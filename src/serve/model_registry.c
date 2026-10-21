/* SPDX-License-Identifier: AGPL-3.0-or-later */
/* Copyright (C) 2026 David (Corey) Herrera */
/**
 * @file model_registry.c
 * @brief Implementation of the multi-model registry (see model_registry.h).
 *
 * Locking rules:
 *   - reg->lock guards the slot table (used/unloading/refs/n). It is NEVER
 *     held across pn_load_on / pn_model_free — a slot is reserved (used, no
 *     model yet) so the name is taken, then the load runs unlocked and the
 *     result is published (or the slot returned) under the lock.
 *   - entry->gen_lock is the per-model generation serializer; the registry
 *     never takes it (the serve chat path does, via trylock).
 *   - reg->drained is signaled by mr_release; mr_unload waits on it until the
 *     entry's refs hit zero, then frees outside the lock. The wait is BOUNDED
 *     (see MR_UNLOAD_DRAIN_MS_DEFAULT) and rolls the entry back on timeout —
 *     an unload that cannot free must refuse, never report success and strand
 *     the dies.
 *
 * @status CPU-validated (tests/test_model_registry.c,
 *         tests/test_model_unload_reclaim.c); die placement numbers become
 *         real on the K80 test machine (die-manager CUDA probe).
 * @issues None
 * @todo None
 */

#include "model_registry.h"

#include <errno.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/stat.h>
#include <time.h>

#include "../accel/backend_identity.h"
#include "../accel/die_layer_split.h"
#include "../accel/die_manager.h"
#include "../core/mem/pn_admission.h"
#include "../formats/gguf_layer_bytes.h"
#include "batch_sched.h"
#include "serve_backend_guard.h"

/* Pre-load admission gate context, used ONLY when the model was launched
 * without --ctx-cap. With a cap set, the KV ring is sized to that cap and this
 * guess is simply wrong: at 8k it demands ~1.1 GiB of KV that a --ctx-cap 2048
 * model will never allocate, which is enough on its own to refuse a 2-die
 * Qwen3.6-35B-A3B Q4_K_M that fits with 1.3 GiB per die to spare. */
#define MR_ADMIT_CTX_GUESS 8192

/* Weight bytes die `idx` of `n_placed` will actually hold.
 *
 * NOT file_bytes / n_placed. A GGUF also carries token_embd — never uploaded,
 * the embedding is a host row copy — and output.weight, which belongs to ONE
 * die. Prorating the file charges every die for both: on Qwen3.6-35B-A3B
 * Q4_K_M that is ~294 MiB of phantom weight per die, and it lands on a gate
 * whose margin is smaller than that.
 *
 * Mirrors the placement the CUDA forward makes (layers split evenly, LM head
 * on the LAST die), read straight off the tensor table. Opening the GGUF here
 * is an mmap + header parse, not a load. Falls back to the prorated share if
 * the file will not open or has no blk.* tensors, so an unfamiliar layout is
 * gated conservatively rather than admitted blind.
 *
 * "Mirrors" is now literal: the layer range comes from pn_die_layer_range, the
 * same partitioner qwen35moe_forward_cuda.c uploads by. It used to be a second
 * copy of the arithmetic, and the copies disagreed — the forward split at
 * n_layer/2 for ANY width while this projected n_layer*idx/n_placed, so a
 * 3-die placement was priced at a third per die and loaded at a half.
 *
 * `n_layers_here_out` / `n_layer_out`, when non-NULL, receive this die's layer
 * count and the model's total — the KV projection needs the same layer share
 * the weights were sliced by, and deriving it twice is how the two drift apart.
 * Both are set to 0 when the layout could not be read, which callers must treat
 * as "no share known" rather than as a zero share. */
static uint64_t mr_die_weight_share(const char *path, int idx, int n_placed,
                                    uint64_t file_bytes,
                                    int *n_layers_here_out, int *n_layer_out)
{
    uint64_t fallback = (n_placed > 0) ? file_bytes / (uint64_t)n_placed
                                       : file_bytes;
    if (n_layers_here_out)
        *n_layers_here_out = 0;
    if (n_layer_out)
        *n_layer_out = 0;
    if (!path || n_placed < 1 || idx < 0 || idx >= n_placed)
        return fallback;

    pn_gguf_file *gf = pn_gguf_open(path);
    if (!gf)
        return fallback;

    uint64_t share = fallback;
    int n_layer = pn_gguf_layer_count(gf);
    int lo = 0, hi = 0;
    if (pn_die_layer_range(n_layer, n_placed, idx, &lo, &hi) == 0) {
        uint64_t layers = pn_gguf_layer_range_bytes(gf, lo, hi);
        if (layers > 0) {
            share = layers;
            if (n_layers_here_out)
                *n_layers_here_out = hi - lo;
            if (n_layer_out)
                *n_layer_out = n_layer;
            if (idx == n_placed - 1) {
                /* The LM head rides with the last die's layers. A TIED model
                 * has no output.weight: its head is token_embd materialized to
                 * F32, so size it as 4 bytes an element, not as the quantized
                 * tensor it was packed as. */
                uint64_t head = pn_gguf_tensor_file_bytes(gf, "output.weight");
                if (head == 0)
                    head = pn_gguf_tensor_numel(gf, "token_embd.weight") *
                           sizeof(float);
                share += head;
            }
        }
    }
    pn_gguf_close(gf);
    return share;
}

/* How long mr_unload waits for in-flight references to drain before refusing.
 *
 * The wait has to be BOUNDED because not every reference drains on its own.
 * A generation turn releases in milliseconds, but pn_session_watch_add takes a
 * reference for the whole lifetime of an open /session/watch connection — a
 * pinned model would park the unload forever. That is not hypothetical: it is
 * what the shutdown path already works around by stopping the watch manager
 * BEFORE mr_free (serve_command.c, serve_run_shutdown), and it is exactly what
 * /admin/models/unload hit in the field — the entry was marked `unloading`
 * (gone from /v1/models and from mr_acquire) while the drain never returned, so
 * pn_dm_release and the model free below it never ran and the dies stayed
 * budgeted AND resident until the whole serve was restarted.
 *
 * 15 s is above a normal turn's release latency and below an operator's
 * patience. Override with PN_UNLOAD_DRAIN_MS (0 = refuse immediately if
 * anything is in flight — what the regression test uses). */
#define MR_UNLOAD_DRAIN_MS_DEFAULT 15000

/* Sentinel for the shutdown path: wait for the drain without a deadline.
 * mr_free is only reached after serve_run_shutdown has quiesced generations and
 * stopped the watch manager, so there is nothing left that cannot drain, and an
 * unbounded wait there keeps shutdown semantics byte-identical to before. */
#define MR_UNLOAD_DRAIN_FOREVER (-1)

static void set_err(char *err, size_t errlen, const char *msg)
{
    if (err && errlen) {
        strncpy(err, msg, errlen - 1);
        err[errlen - 1] = '\0';
    }
}

/* Configured unload drain budget in milliseconds (PN_UNLOAD_DRAIN_MS, else the
 * default). A garbage value falls back; an explicit 0 is honoured. */
static int mr_unload_drain_ms(void)
{
    const char *env = getenv("PN_UNLOAD_DRAIN_MS");
    if (env && *env) {
        char *end = NULL;
        long v = strtol(env, &end, 10);
        if (end != env && v >= 0 && v <= 3600000L)
            return (int)v;
    }
    return MR_UNLOAD_DRAIN_MS_DEFAULT;
}

/* Basename of a path (pointer into the same string). */
static const char *path_basename(const char *path)
{
    const char *slash = strrchr(path, '/');
    return slash ? slash + 1 : path;
}

int mr_init(mr_registry *reg)
{
    if (!reg)
        return -1;
    memset(reg, 0, sizeof(*reg));
    if (pthread_mutex_init(&reg->lock, NULL) != 0)
        return -1;
    if (pthread_cond_init(&reg->drained, NULL) != 0) {
        pthread_mutex_destroy(&reg->lock);
        return -1;
    }
    /* A3: serialize GPU weight uploads so the ~2.4 GB transient never stacks. */
    if (pthread_mutex_init(&reg->upload_lock, NULL) != 0) {
        pthread_cond_destroy(&reg->drained);
        pthread_mutex_destroy(&reg->lock);
        return -1;
    }
    return 0;
}

/* Locked lookup: the slot holding `name`, or NULL. Ready-only when
 * require_ready (skips reserved-but-still-loading slots). */
static mr_entry *find_locked(mr_registry *reg, const char *name,
                             bool require_ready)
{
    for (int i = 0; i < MR_MAX_MODELS; i++) {
        mr_entry *e = &reg->entries[i];
        if (!e->used)
            continue;
        if (require_ready && (!e->model || e->unloading))
            continue;
        if (strncmp(e->name, name, MR_MAX_NAME) == 0)
            return e;
    }
    return NULL;
}

int mr_load(mr_registry *reg, const char *path, const char *name,
            const pn_load_opts *opts, const int *dies, int n_dies,
            int reasoning_budget, int max_sessions, int max_concurrent,
            char *err, size_t errlen)
{
    if (!reg || !path) {
        set_err(err, errlen, "registry/path required");
        return -1;
    }

    /* Resolve the per-model session cap: <=0 → the BS_MAX_USERS default
     * (byte-identical to the pre-cap behavior); otherwise clamp into
     * [1, BS_MAX_USERS] — the batch scheduler's user cap is the hard ceiling,
     * so a request for more sessions than the batcher can serve is capped, not
     * refused (over-request is an operator sizing hint, not an error). */
    int resolved_max_sessions = BS_MAX_USERS;
    if (max_sessions > 0)
        resolved_max_sessions = max_sessions < BS_MAX_USERS
                              ? max_sessions : BS_MAX_USERS;

    /* Resolve the per-model CONCURRENCY cap — a different ceiling from the
     * residency cap above, and the one the batch scheduler enforces. Residency
     * = how many named conversations may hold KV; concurrency = how many
     * requests may decode in the same batched step. Before Phase 7 the
     * scheduler was created with the compile-time BS_MAX_USERS regardless of
     * either, so a model configured for 2 users would happily admit 8 — which
     * is how a burst could drive saber-orch off the ctx_cap-131072 x K=4 ring
     * cliff. Absent/<=0 defaults to the
     * resolved max_sessions, so a single manifest knob still sizes both. */
    int resolved_max_concurrent = resolved_max_sessions;
    if (max_concurrent > 0)
        resolved_max_concurrent = max_concurrent < BS_MAX_USERS
                                ? max_concurrent : BS_MAX_USERS;
    if (!name || !*name)
        name = path_basename(path);
    if (strlen(name) >= MR_MAX_NAME) {
        set_err(err, errlen, "model name too long");
        return -1;
    }
    if (n_dies < 0 || n_dies > PN_MODEL_MAX_DIES || (n_dies > 0 && !dies)) {
        set_err(err, errlen, "bad die list");
        return -1;
    }

    /* A die request this process cannot honour is a REFUSAL, not a quiet CPU
     * placement. The check lives here as well as in pn_serve_main's pre-flight
     * because POST /admin/models/load reaches this function directly with its
     * own `dies` array, and a serve that refused the wrong manifest at startup
     * must not accept the same mistake an hour later over HTTP. Same predicate,
     * one definition (serve_backend_guard.h). n_dies == 0 is always allowed —
     * that is how a CPU serve is asked for. */
    {
        pn_backend_guard_verdict v =
            pn_backend_guard_check(n_dies, pn_backend_supports_die_placement(),
                                   pn_dm_count());
        if (v != PN_BACKEND_GUARD_OK) {
            set_err(err, errlen, pn_backend_guard_short_reason(v));
            return -1;
        }
    }

    struct stat st;
    if (stat(path, &st) != 0 || st.st_size <= 0) {
        set_err(err, errlen, "model file not found");
        return -1;
    }
    uint64_t weights_bytes = (uint64_t)st.st_size;   /* GGUF mmaps ~1:1 */
    /* Model-identity tag for the session warm-tier (Phase 7): FNV-1a over the
     * path, then mixed with the file size and mtime. Two different files, or the
     * same path after a re-quantize/rebuild (size or mtime changes), get
     * distinct tags — so a stale session blob is cleanly rejected on restore. */
    uint64_t weights_version = 1469598103934665603ull;   /* FNV offset basis */
    for (const char *p = path; *p; p++) {
        weights_version ^= (unsigned char)*p;
        weights_version *= 1099511628211ull;             /* FNV prime */
    }
    weights_version ^= weights_bytes + 0x9E3779B97F4A7C15ull +
                       (weights_version << 6) + (weights_version >> 2);
    weights_version ^= (uint64_t)st.st_mtime + 0x9E3779B97F4A7C15ull +
                       (weights_version << 6) + (weights_version >> 2);

    /* --- reserve the name/slot ------------------------------------------ */
    /* Same-name placements are ALLOWED and intentional: an operator places one
     * model id on several cards (e.g. --dies 0,1 and --dies 2,3 under one
     * --name) and mr_acquire load-balances across them. Clients see a single id
     * (oai_handle_models dedupes by name). Load/unload is operator-only (never
     * exposed to clients), so name collisions are an operator concern, not a
     * safety hole — hence no duplicate-name refusal here. */
    pthread_mutex_lock(&reg->lock);
    mr_entry *e = NULL;
    for (int i = 0; i < MR_MAX_MODELS; i++) {
        if (!reg->entries[i].used) {
            e = &reg->entries[i];
            break;
        }
    }
    if (!e) {
        pthread_mutex_unlock(&reg->lock);
        set_err(err, errlen, "model table full");
        return -1;
    }
    memset(e, 0, sizeof(*e));
    e->used = true;                       /* reserved: name taken, not ready */
    strncpy(e->name, name, MR_MAX_NAME - 1);
    pthread_mutex_unlock(&reg->lock);

    /* --- placement + cross-model VRAM budgeting (no registry lock) ------- */
    int placed[PN_MODEL_MAX_DIES];
    int n_placed = 0;
    if (n_dies > 0) {
        memcpy(placed, dies, (size_t)n_dies * sizeof(int));
        n_placed = n_dies;
    } else if (pn_dm_count() > 0) {
        /* Auto-place: one die when the weights fit a die's budget, else a
         * same-card pair (the qwen35moe 2-die split). Ask what ONE die would
         * really hold, not what the file weighs — a tied model's F32 LM head
         * makes its residency exceed its file, so the file size can say "fits
         * one die" about a model that does not. */
        pn_dm_die d0;
        int want = 1;
        uint64_t one_die = mr_die_weight_share(path, 0, 1, weights_bytes,
                                               NULL, NULL);
        if (pn_dm_get(0, &d0) == 0 && one_die > d0.vram_budget)
            want = 2;
        if (pn_dm_place(one_die, want, placed) == 0)
            n_placed = want;
        /* No fit → fall through CPU-resident rather than refusing: the CPU
         * forward still serves (the K80 host does this for GGUF models the
         * GPU path doesn't cover). */
    }

    uint64_t vram_bytes = 0;
    if (n_placed > 0 && pn_dm_count() > 0) {
        /* Admission per candidate die (MemoryAware §6.1): weights share + KV
         * projection must fit the die's remaining budget. B1: Q4 KV is now
         * the default; mirror pn_qwen35moe_kv_q4_on()'s logic (ON unless =0). */
        const char *kv_q4_env = getenv("QWEN35_KV_Q4");
        bool kv_q4 = !(kv_q4_env && kv_q4_env[0] == '0');
        /* The KV ring is sized to --ctx-cap when one is set, so gate against
         * that and fall back to the guess only when it is absent. */
        int admit_ctx = (opts && opts->ctx_cap > 0) ? opts->ctx_cap
                                                    : MR_ADMIT_CTX_GUESS;
        uint64_t need_total = 0;
        for (int i = 0; i < n_placed; i++) {
            pn_dm_die d;
            if (pn_dm_get(placed[i], &d) != 0) {
                pthread_mutex_lock(&reg->lock);
                memset(e, 0, sizeof(*e));
                pthread_mutex_unlock(&reg->lock);
                set_err(err, errlen, "unknown die in placement");
                return -1;
            }
            uint64_t die_free = d.vram_budget > d.vram_used
                              ? d.vram_budget - d.vram_used : 0;
            uint64_t need = 0;
            int n_layers_here = 0, n_layer = 0;
            uint64_t per_die = mr_die_weight_share(path, i, n_placed,
                                                   weights_bytes,
                                                   &n_layers_here, &n_layer);
            /* Pre-load placement guess: the model is not resident yet, so its
             * real KV geometry is unknown — use the conservative generic
             * per-token constant (KV Phase 1 threads real geometry in only where
             * the loaded model is in hand). Measured against the real thing on
             * the one model that matters here, the generic Q4 constant is close:
             * 5898 B/token vs the 6144 B/token Qwen3.6-35B-A3B actually reports
             * from its own geometry, a 4% under-price that is ~15 MiB per die at
             * a 131072 cap — noise beside the scratch reserve. Not worth
             * teaching this path to parse a hybrid's attention layout.
             *
             * Scale by THIS die's layer share, the same way the weights were
             * sliced. Charging every die the whole model's KV double-counted it
             * across a 2-die split. */
            uint64_t kv_bytes = pn_admit_die_kv_bytes(0, kv_q4, admit_ctx,
                                                      n_layers_here, n_layer);
            /* A GPU placement is a GPU load: project weights + KV + on-die
             * scratch, NOT the host activation LUT. die_free comes off
             * vram_budget, which the die manager already set to total minus the
             * per-die reserve — so use the _budgeted entry point, which does not
             * hold it back a second time. */
            if (!pn_admit_model_load_die_budgeted(per_die, kv_bytes, admit_ctx,
                                                  die_free, &need)) {
                pthread_mutex_lock(&reg->lock);
                memset(e, 0, sizeof(*e));
                pthread_mutex_unlock(&reg->lock);
                /* Say WHICH die and BY HOW MUCH. A bare "(admission)" sent an
                 * operator hunting the wrong gate for a whole session: the
                 * numbers are the difference between "buy VRAM" and "the
                 * projection is wrong". */
                snprintf(err, errlen,
                         "die VRAM budget refused the model (admission): "
                         "die %d has %llu MiB free, needs %llu MiB "
                         "(weights %llu + KV@%d + scratch)",
                         placed[i],
                         (unsigned long long)(die_free >> 20),
                         (unsigned long long)(need >> 20),
                         (unsigned long long)(per_die >> 20), admit_ctx);
                return -1;
            }
            need_total += need;
        }
        vram_bytes = need_total;
        if (pn_dm_bind(placed, n_placed, name, vram_bytes) != 0) {
            pthread_mutex_lock(&reg->lock);
            memset(e, 0, sizeof(*e));
            pthread_mutex_unlock(&reg->lock);
            set_err(err, errlen, "die binding refused (placement raced or "
                                 "no longer fits)");
            return -1;
        }
    }

    /* --- the load itself (slow; registry stays serviceable) -------------- */
    /* A3: serialize GPU weight uploads — at most one in-flight at a time so
     * the ~2.4 GB host transient never stacks. The startup loop is sequential
     * already; this guard covers a runtime /admin/models/load that arrives
     * while a startup upload is still in progress. The mutex is held only
     * across pn_load_on (and released immediately after): everything else
     * (admission, slot reservation, publish) stays outside it so a waiting
     * caller sees the live registry and can be refused promptly on its slot
     * attempt once the prior load finishes. */
    pthread_mutex_lock(&reg->upload_lock);
    pn_model *model = pn_load_on(path, opts, n_placed ? placed : NULL,
                                 n_placed);
    pthread_mutex_unlock(&reg->upload_lock);
    pn_state *state = model ? pn_state_new(model) : NULL;
    if (!model || !state) {
        if (model)
            pn_model_free(model);
        if (n_placed > 0)
            pn_dm_release(name);
        pthread_mutex_lock(&reg->lock);
        memset(e, 0, sizeof(*e));
        pthread_mutex_unlock(&reg->lock);
        set_err(err, errlen, model ? "failed to create inference state"
                                   : "failed to load model");
        return -1;
    }

    /* --- publish ---------------------------------------------------------- */
    if (pthread_mutex_init(&e->gen_lock, NULL) != 0) {
        pn_state_free(state);
        pn_model_free(model);
        if (n_placed > 0)
            pn_dm_release(name);
        pthread_mutex_lock(&reg->lock);
        memset(e, 0, sizeof(*e));
        pthread_mutex_unlock(&reg->lock);
        set_err(err, errlen, "failed to init generation lock");
        return -1;
    }
    if (pthread_mutex_init(&e->pc_lock, NULL) != 0) {
        pthread_mutex_destroy(&e->gen_lock);
        pn_state_free(state);
        pn_model_free(model);
        if (n_placed > 0)
            pn_dm_release(name);
        pthread_mutex_lock(&reg->lock);
        memset(e, 0, sizeof(*e));
        pthread_mutex_unlock(&reg->lock);
        set_err(err, errlen, "failed to init prefix-cache lock");
        return -1;
    }
    pthread_mutex_lock(&reg->lock);
    e->path = strdup(path);
    e->model = model;
    e->state = state;
    e->sched = NULL;         /* C3: initialised below after unlock */
    e->prefix_cache = NULL;  /* PC1: initialised below after unlock */
    memcpy(e->dies, placed, (size_t)n_placed * sizeof(int));
    e->n_dies = n_placed;
    e->reasoning_budget = reasoning_budget > 0 ? reasoning_budget : 0;
    e->max_sessions = resolved_max_sessions;
    e->max_concurrent = resolved_max_concurrent;
    e->vram_bytes = vram_bytes;
    e->weights_version = weights_version;
    e->refs = 0;
    e->unloading = false;
    reg->n++;
    pthread_mutex_unlock(&reg->lock);

    /* C3: start the batch scheduler for GPU-resident models when enabled.
     * Done after publish so the model is acquirable while the scheduler
     * starts its driver thread (scheduler creation is fast — no GPU work). */
    if (bs_enabled() && n_placed > 0) {
        /* The scheduler's slot count is the per-model CONCURRENCY cap, not the
         * compile-time BS_MAX_USERS: a model sized for N users must not admit
         * more than N into a batched step. bs_create reserves an extra engine
         * slot as the rebuild scratch on top of this, so the cap is the user
         * slot count exactly. */
        pn_batch_sched *sched = bs_create(model, resolved_max_concurrent, e->name);
        if (sched) {
            fprintf(stderr, "registry: batch scheduler for '%s': "
                    "%d concurrent decode slots (max_concurrent), "
                    "%d resident sessions (max_sessions)\n",
                    e->name, resolved_max_concurrent, resolved_max_sessions);
            /* Publish sched under the registry lock so concurrent acquires
             * see a consistent view (sched==NULL means old path). */
            pthread_mutex_lock(&reg->lock);
            e->sched = sched;
            pthread_mutex_unlock(&reg->lock);
        } else {
            fprintf(stderr, "registry: batch scheduler creation failed for "
                    "'%s'; falling back to single-user mode\n", e->name);
        }
    }

    /* PC1: session prefix cache for GPU-resident models (the SSM
     * snapshot/restore primitive is CUDA-only).  max_sessions = the resolved
     * per-model cap (default BS_MAX_USERS = 8) — bounds both the live-session
     * count and the host footprint (~240 MB/session).  A big model with little
     * die headroom (e.g. saber, ~18 GB across 2 dies) is capped BELOW 8 so 8
     * snapshots can't OOM the card; the default stays byte-identical.  A
     * creation failure is non-fatal: prefix_cache stays NULL and serving falls
     * back to full prefill (byte-identical to today). */
    if (n_placed > 0) {
        pn_prefix_cache *pc = pn_prefix_cache_new(model, resolved_max_sessions);
        if (pc) {
            pthread_mutex_lock(&reg->lock);
            e->prefix_cache = pc;
            pthread_mutex_unlock(&reg->lock);
            fprintf(stderr, "registry: prefix cache enabled for '%s' "
                    "(max_sessions=%d)\n", e->name, resolved_max_sessions);
        } else {
            fprintf(stderr, "registry: prefix cache creation failed for "
                    "'%s'; falling back to full prefill\n", e->name);
        }
    }

    fprintf(stderr, "registry: loaded '%s' (%s)%s\n", e->name, path,
            n_placed ? "" : " [CPU-resident]");
    if (n_placed) {
        fprintf(stderr, "registry: '%s' placed on die%s", e->name,
                n_placed > 1 ? "s" : "");
        for (int i = 0; i < n_placed; i++)
            fprintf(stderr, "%s%d", i ? "," : " ", placed[i]);
        fprintf(stderr, " (%.1f GiB projected)\n",
                (double)vram_bytes / (1024.0 * 1024.0 * 1024.0));
    }
    return 0;
}

mr_entry *mr_acquire(mr_registry *reg, const char *name)
{
    if (!reg)
        return NULL;
    pthread_mutex_lock(&reg->lock);
    mr_entry *e = NULL;
    if (name && *name) {
        /* Multi-card load-balance: among the ready placements that share this
         * name, pick the least-loaded (fewest in-flight refs). One model id
         * thus spreads across every card it is placed on; a single placement
         * is just the degenerate one-candidate case. */
        for (int i = 0; i < MR_MAX_MODELS; i++) {
            mr_entry *c = &reg->entries[i];
            if (!c->used || !c->model || c->unloading)
                continue;
            if (strncmp(c->name, name, MR_MAX_NAME) != 0)
                continue;
            if (!e || c->refs < e->refs)
                e = c;
        }
    } else {
        /* Default model: the lowest-slot resident entry. */
        for (int i = 0; i < MR_MAX_MODELS; i++) {
            mr_entry *c = &reg->entries[i];
            if (c->used && c->model && !c->unloading) {
                e = c;
                break;
            }
        }
    }
    if (e)
        e->refs++;
    pthread_mutex_unlock(&reg->lock);
    return e;
}

void mr_release(mr_registry *reg, mr_entry *entry)
{
    if (!reg || !entry)
        return;
    pthread_mutex_lock(&reg->lock);
    if (entry->refs > 0)
        entry->refs--;
    pthread_cond_broadcast(&reg->drained);
    pthread_mutex_unlock(&reg->lock);
}

int mr_keepalive(mr_registry *reg)
{
    int i, sent = 0;

    if (!reg)
        return 0;
    for (i = 0; i < MR_MAX_MODELS; i++) {
        mr_entry *e = &reg->entries[i];
        bool held = false;

        pthread_mutex_lock(&reg->lock);
        if (e->used && e->model && !e->unloading && !e->sched) {
            e->refs++;
            held = true;
        }
        pthread_mutex_unlock(&reg->lock);
        if (!held)
            continue;
        /* TRYLOCK: a generation in flight IS the keepalive; never wait behind one. */
        if (pthread_mutex_trylock(&e->gen_lock) == 0) {
            const int rc = pn_model_keepalive(e->model);

            pthread_mutex_unlock(&e->gen_lock);
            if (rc > 0)
                sent += rc;
        }
        mr_release(reg, e);
    }
    return sent;
}

int mr_trylock_generation(mr_entry *entry)
{
    int waited_ms;

    /* The idle keepalive holds gen_lock for a few milliseconds; a generation holds it for
     * seconds. Waiting up to 100 ms tells the two apart without turning the keepalive into
     * a spurious 429. */
    for (waited_ms = 0; waited_ms < 100; waited_ms += 2) {
        struct timespec pause = { 0, 2000000L };

        if (pthread_mutex_trylock(&entry->gen_lock) == 0)
            return 0;
        nanosleep(&pause, NULL);
    }
    return pthread_mutex_trylock(&entry->gen_lock);
}

/* Wait for `e->refs` to reach zero, at most `drain_ms` milliseconds (negative =
 * no deadline). Caller holds reg->lock; the lock is held again on return.
 * Returns 0 when the entry drained, -1 on timeout (refs still > 0). */
static int drain_entry_locked(mr_registry *reg, mr_entry *e, int drain_ms)
{
    if (drain_ms < 0) {
        while (e->refs > 0)
            pthread_cond_wait(&reg->drained, &reg->lock);
        return 0;
    }

    struct timespec deadline;
    clock_gettime(CLOCK_REALTIME, &deadline);
    deadline.tv_sec += drain_ms / 1000;
    deadline.tv_nsec += (long)(drain_ms % 1000) * 1000000L;
    if (deadline.tv_nsec >= 1000000000L) {
        deadline.tv_sec += 1;
        deadline.tv_nsec -= 1000000000L;
    }

    while (e->refs > 0) {
        int rc = pthread_cond_timedwait(&reg->drained, &reg->lock, &deadline);
        if (rc == ETIMEDOUT)
            break;
    }
    return e->refs > 0 ? -1 : 0;
}

/* Unload `name`, waiting at most `drain_ms` ms for in-flight references
 * (MR_UNLOAD_DRAIN_FOREVER = no deadline). On a drain timeout the entry is
 * rolled back — still resident, still acquirable, still bound to its dies — and
 * the refusal names the reason. See MR_UNLOAD_DRAIN_MS_DEFAULT for why the
 * public path must never wait indefinitely. */
static int mr_unload_drain(mr_registry *reg, const char *name, int drain_ms,
                           char *err, size_t errlen)
{
    if (!reg || !name || !*name) {
        set_err(err, errlen, "model name required");
        return -1;
    }
    pthread_mutex_lock(&reg->lock);
    mr_entry *e = find_locked(reg, name, false);
    if (!e || !e->model) {
        pthread_mutex_unlock(&reg->lock);
        set_err(err, errlen, "no such model");
        return -1;
    }
    if (e->unloading) {
        pthread_mutex_unlock(&reg->lock);
        set_err(err, errlen, "model is already unloading");
        return -1;
    }
    e->unloading = true;            /* no new acquires while we try to drain */

    if (drain_entry_locked(reg, e, drain_ms) != 0) {
        /* COULD NOT FREE. Put the entry back exactly as it was — resident,
         * acquirable, holding its die binding — and refuse. The alternative
         * (leave it deregistered and return success) is what stranded two dies
         * in the field: the registry said the model was gone while its VRAM and
         * its die budget were still held, and only a full serve restart got
         * them back. */
        int refs = e->refs;
        e->unloading = false;
        pthread_mutex_unlock(&reg->lock);
        /* Kept short on purpose: the caller's reason buffer is 160 bytes and a
         * truncated multi-byte dash would land as mojibake in the JSON error. */
        char msg[160];
        snprintf(msg, sizeof(msg),
                 "model is busy: %d reference(s) in flight after %d ms; NOT "
                 "unloaded (a live turn or an open /session/watch pins it)",
                 refs, drain_ms);
        set_err(err, errlen, msg);
        fprintf(stderr, "registry: REFUSED unload of '%s' — %d in-flight "
                "reference(s) did not drain in %d ms; the model stays resident "
                "(nothing freed, nothing stranded)\n", name, refs, drain_ms);
        return -1;
    }
    /* Committed from here on: the entry drained, so every step below runs. */
    reg->n--;

    /* Detach the resources, free the slot, THEN free outside the lock (a
     * model free can be slow — munmap + GPU buffer teardown). */
    pn_model *model = e->model;
    pn_state *state = e->state;
    pn_batch_sched *sched = e->sched;
    pn_prefix_cache *prefix_cache = e->prefix_cache;
    char *path = e->path;
    int had_dies = e->n_dies;
    char freed_name[MR_MAX_NAME];
    strncpy(freed_name, e->name, MR_MAX_NAME);
    freed_name[MR_MAX_NAME - 1] = '\0';
    pthread_mutex_destroy(&e->gen_lock);
    pthread_mutex_destroy(&e->pc_lock);
    memset(e, 0, sizeof(*e));
    pthread_mutex_unlock(&reg->lock);

    /* PC1: free the session prefix cache (frees every cached pn_state) BEFORE
     * the model/scheduler teardown — its states reference the model. */
    if (prefix_cache)
        pn_prefix_cache_free(prefix_cache);

    /* C3: stop the batch scheduler (joins driver thread) before freeing
     * model state — bs_destroy calls pn_state_free on slot states. */
    if (sched)
        bs_destroy(sched);

    /* Give the die budget back BEFORE the (slow) model free, and never discard
     * the verdict: a release that matches no binding leaves the die budgeted
     * for a model that no longer exists, and the next load onto those dies is
     * refused by admission with nothing in the log to explain it. */
    if (had_dies > 0 && pn_dm_release(freed_name) != 0)
        fprintf(stderr, "registry: WARNING unload '%s' found no die binding to "
                "release — %d die(s) may stay budgeted until restart\n",
                freed_name, had_dies);
    pn_state_free(state);
    pn_model_free(model);
    free(path);
    fprintf(stderr, "registry: unloaded '%s'%s\n", freed_name,
            had_dies > 0 ? " (die budget released)" : "");
    return 0;
}

int mr_unload(mr_registry *reg, const char *name, char *err, size_t errlen)
{
    return mr_unload_drain(reg, name, mr_unload_drain_ms(), err, errlen);
}

uint64_t mr_weights_version(const mr_entry *entry)
{
    return entry ? entry->weights_version : 0;
}

int mr_count(mr_registry *reg)
{
    if (!reg)
        return 0;
    pthread_mutex_lock(&reg->lock);
    int n = reg->n;
    pthread_mutex_unlock(&reg->lock);
    return n;
}

bool mr_model_is_minicpm(const pn_model *model)
{
    return model && model->config.arch == PN_ARCH_LLAMA && model->chat_add_bos;
}

int mr_snapshot(mr_registry *reg, mr_info *out, int max)
{
    if (!reg || !out || max <= 0)
        return 0;
    pthread_mutex_lock(&reg->lock);
    int n = 0;
    for (int i = 0; i < MR_MAX_MODELS && n < max; i++) {
        mr_entry *e = &reg->entries[i];
        if (!e->used || !e->model || e->unloading)
            continue;
        memset(&out[n], 0, sizeof(out[n]));
        strncpy(out[n].name, e->name, MR_MAX_NAME - 1);
        if (e->path) {
            strncpy(out[n].path, e->path, sizeof(out[n].path) - 1);
            out[n].path[sizeof(out[n].path) - 1] = '\0';
        }
        memcpy(out[n].dies, e->dies, sizeof(e->dies));
        out[n].n_dies = e->n_dies;
        out[n].vram_bytes = e->vram_bytes;
        out[n].max_sessions = e->max_sessions;
        out[n].max_concurrent = e->max_concurrent;
        /* CONFIGURED vs SERVED. bs_get_load already reports capacity 1 for a
         * NULL scheduler (the single-user trylock->429 path), which is the
         * whole reason it is asked here rather than trusting e->max_concurrent:
         * the entry keeps the cap the operator ASKED for even when bs_create
         * returned NULL (no CUDA compiled in, QWEN35_BATCH_SLOTS off) or
         * clamped the slot count to the memory-pressure ladder ceiling. Cheap
         * and lock-safe under reg->lock: bs_get_load takes only sched->mu and
         * nothing under sched->mu ever reaches back into the registry, so the
         * two locks cannot form a cycle. */
        bs_load load = {0};
        bs_get_load(e->sched, &load);
        out[n].served_concurrency = load.capacity;
        /* Read the window HERE, under the registry lock, while the entry still
         * holds a live model pointer — the snapshot is a copy-out and its
         * consumers (the /v1/models + /admin writers) hold no ref on the model
         * afterwards. Both calls are pure reads of the model config plus the
         * memwatch snapshot: no allocation, no blocking, safe under the lock. */
        out[n].context_window = pn_effective_ctx_window(e->model);
        out[n].context_window_configured = pn_session_ctx_cap(e->model);
        /* Same reasoning: a lock-free read of per-die counters, no allocation. */
        pn_model_graph_residency_read(e->model, &out[n].graphs);
        pn_model_attn_residency_read(e->model, &out[n].attn);
        out[n].minicpm_tool_dialect = mr_model_is_minicpm(e->model);
        out[n].refs = e->refs;
        n++;
    }
    pthread_mutex_unlock(&reg->lock);
    return n;
}

void mr_free(mr_registry *reg)
{
    if (!reg)
        return;
    /* Unload everything that is resident (drains each in turn). */
    for (;;) {
        char name[MR_MAX_NAME] = "";
        pthread_mutex_lock(&reg->lock);
        for (int i = 0; i < MR_MAX_MODELS; i++) {
            mr_entry *e = &reg->entries[i];
            if (e->used && e->model && !e->unloading) {
                strncpy(name, e->name, MR_MAX_NAME - 1);
                break;
            }
        }
        pthread_mutex_unlock(&reg->lock);
        if (!name[0])
            break;
        /* Shutdown drains without a deadline on purpose — see
         * MR_UNLOAD_DRAIN_FOREVER. A bounded drain here would turn a busy
         * model into an endless refuse/retry loop in this very while(1). */
        mr_unload_drain(reg, name, MR_UNLOAD_DRAIN_FOREVER, NULL, 0);
    }
    pthread_cond_destroy(&reg->drained);
    pthread_mutex_destroy(&reg->lock);
    pthread_mutex_destroy(&reg->upload_lock);
}
