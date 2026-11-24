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
 *   - reg->drained is signaled by mr_release / mr_watch_unpin; mr_unload waits
 *     on it until the entry's live refs hit zero (cancelling turns that
 *     overstay), closes its watches through the unload hook, then frees
 *     outside the lock. Every wait is BOUNDED (see MR_UNLOAD_DRAIN_MS_DEFAULT)
 *     and rolls the entry back on timeout — an unload that cannot free must
 *     refuse, never report success and strand the dies.
 *
 * @status CPU-validated (tests/test_model_registry.c,
 *         tests/test_model_unload_reclaim.c, tests/test_load_budget_release.c); die placement numbers become
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
#include "../core/load_guard.h"
#include "../core/mem/pn_admission.h"
#include "../core/mem/pn_host_admission.h"
#include "../core/mem/pn_ladder.h"
#include "../formats/gguf_layer_bytes.h"
#include "batch_sched.h"
#include "batch_sched_slots.h"
#include "model_session.h"
#include "replica_affinity.h"
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

/* How long mr_unload lets live turns finish before cancelling them.
 *
 * Only LIVE references (generation turns, short route/admin holds) are waited
 * on. An open /session/watch used to hold an mr_acquire reference for the whole
 * life of its connection, so an unload with a chat tab open always timed out
 * and refused ("2 reference(s) in flight after 15000 ms"). Watches now hold a
 * separate pin (mr_watch_pin) that an unload never waits on: it closes them
 * through the unload hook once the turns are gone.
 *
 * Override with PN_UNLOAD_DRAIN_MS (0 = cancel at once). */
#define MR_UNLOAD_DRAIN_MS_DEFAULT 5000

/* How long a cancelled turn gets to unwind before the unload refuses. A turn
 * polls should_abort between prefill chunks and every 8 decode tokens, so this
 * is generous. Override with PN_UNLOAD_CANCEL_MS. */
#define MR_UNLOAD_CANCEL_MS_DEFAULT 10000

/* How long the unload waits for watch pins to drop after the hook closed them.
 * Only a grace sweep caught mid-release can still hold one; it lets go in
 * microseconds. */
#define MR_UNLOAD_WATCH_UNPIN_MS 5000

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

/* A millisecond budget from env var `name`, else `fallback`. A garbage value
 * falls back; an explicit 0 is honoured. */
static int mr_env_ms(const char *name, int fallback)
{
    const char *env = getenv(name);
    if (env && *env) {
        char *end = NULL;
        long v = strtol(env, &end, 10);
        if (end != env && v >= 0 && v <= 3600000L)
            return (int)v;
    }
    return fallback;
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
mr_entry *mr_find_replica_locked(mr_registry *reg, const char *name, int first_die,
                                 bool require_ready)
{
    for (int i = 0; i < MR_MAX_MODELS; i++) {
        mr_entry *e = &reg->entries[i];
        if (!e->used)
            continue;
        if (require_ready && (!e->model || e->unloading))
            continue;
        if (first_die >= 0 && (e->n_dies < 1 || e->dies[0] != first_die))
            continue;
        if (strncmp(e->name, name, MR_MAX_NAME) == 0)
            return e;
    }
    return NULL;
}

/* The load was refused by host-RAM admission: hand back the exact refusal.
 * "admission" in the text is what maps it to HTTP 507 (model_admin.c). */
static void set_host_refusal_err(char *err, size_t errlen)
{
    if (err && errlen)
        snprintf(err, errlen, "host RAM admission refused: %s",
                 pn_host_admit_last_refusal());
}

/* Undo a load that did not publish: free what it built, give back its die
 * reservation, and return the slot. The ONE exit for every failure after the
 * slot was reserved, so no failure path can keep a reservation.
 *
 * The reservation is released by its ref, never by name: replicas of one model
 * share a name, and a by-name release freed the first replica's binding while
 * the failed load's stayed budgeted (K80 bank 2026-11-03: "die 8 has 792 MiB
 * free, needs 10261 MiB" with die 8 empty). */
static void mr_load_abandon(mr_registry *reg, mr_entry *e, pn_dm_binding_ref *binding,
                            pn_batch_sched *sched, pn_state *state, pn_model *model)
{
    bs_destroy(sched);
    if (state)
        pn_state_free(state);
    if (model)
        pn_model_free(model);
    if (binding->slot >= 0 && pn_dm_release_ref(binding) != 0)
        fprintf(stderr, "registry: WARNING failed load of '%s' found no die binding "
                "to release\n", e->name);
    pthread_mutex_lock(&reg->lock);
    memset(e, 0, sizeof(*e));
    pthread_mutex_unlock(&reg->lock);
}

/* The copy number for a new entry of `name`: `want` when free (0 = refused),
 * else, for want == 0, the lowest number no reserved entry of the name uses.
 * Caller holds reg->lock. */
static int replica_number_locked(mr_registry *reg, const char *name, int want)
{
    for (int n = want > 0 ? want : 1;; n++) {
        bool taken = false;
        for (int i = 0; i < MR_MAX_MODELS && !taken; i++) {
            const mr_entry *c = &reg->entries[i];
            taken = c->used && c->replica == n && strncmp(c->name, name, MR_MAX_NAME) == 0;
        }
        if (!taken)
            return n;
        if (want > 0)
            return 0;
    }
}

int mr_load(mr_registry *reg, const char *path, const char *name,
            const pn_load_opts *opts, const int *dies, int n_dies,
            int reasoning_budget, int max_sessions, int max_concurrent,
            char *err, size_t errlen)
{
    return mr_load_replica(reg, path, name, opts, dies, n_dies, reasoning_budget, max_sessions,
                           max_concurrent, 0, err, errlen);
}

int mr_load_replica(mr_registry *reg, const char *path, const char *name,
                    const pn_load_opts *opts, const int *dies, int n_dies,
                    int reasoning_budget, int max_sessions, int max_concurrent,
                    int replica, char *err, size_t errlen)
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
    const bool cpu_only = opts && opts->cpu_only;
    if (cpu_only && n_dies > 0) {
        set_err(err, errlen, "a cpu placement takes no dies");
        return -1;
    }

    /* A die request this process cannot honour is a REFUSAL, not a quiet CPU
     * placement. The check lives here as well as in pn_serve_main's pre-flight
     * because POST /admin/models/load reaches this function directly with its
     * own `dies` array, and a serve that refused the wrong manifest at startup
     * must not accept the same mistake an hour later over HTTP. Same predicate,
     * one definition (serve_backend_guard.h). n_dies == 0 is always allowed
     * here; where it runs is the load guard's call (load_guard.h). */
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
    int number = replica_number_locked(reg, name, replica);
    if (number == 0) {
        pthread_mutex_unlock(&reg->lock);
        if (err && errlen)
            snprintf(err, errlen, "replica %d of '%s' is already loaded", replica, name);
        return -1;
    }
    memset(e, 0, sizeof(*e));
    e->used = true;                       /* reserved: name taken, not ready */
    strncpy(e->name, name, MR_MAX_NAME - 1);
    e->serial = ++reg->next_serial;       /* this placement, never reused */
    e->replica = number;
    pthread_mutex_unlock(&reg->lock);

    /* --- placement + cross-model VRAM budgeting (no registry lock) ------- */
    int placed[PN_MODEL_MAX_DIES];
    int n_placed = 0;
    pn_dm_binding_ref binding = PN_DM_BINDING_NONE;   /* owned here until publish */
    if (n_dies > 0) {
        memcpy(placed, dies, (size_t)n_dies * sizeof(int));
        n_placed = n_dies;
    } else if (!cpu_only && pn_dm_count() > 0) {
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
    }
    /* No die has room: refused, never placed on the host CPU instead (the
     * load guard, load_guard.h). A CPU load is asked for with cpu_only. */
    if (n_placed == 0 && !cpu_only && pn_dm_count() > 0 && pn_load_guard_accel_build()) {
        mr_load_abandon(reg, e, &binding, NULL, NULL, NULL);
        if (err && errlen)
            snprintf(err, errlen, "no die has room for %s (~%llu MiB of weights); "
                     "no CPU fallback: try a smaller quant, free a die, or --cpu",
                     path_basename(path), (unsigned long long)(weights_bytes >> 20));
        return -1;
    }

    uint64_t vram_bytes = 0;
    if (n_placed > 0 && pn_dm_count() > 0) {
        /* Admission per candidate die (MemoryAware §6.1): weights share + KV
         * projection must fit the die's remaining budget. B1: Q4 KV is now
         * the default; mirror pn_qwen35moe_kv_q4_on()'s logic (ON unless =0). */
        const char *kv_q4_env = getenv("QWEN35_KV_Q4");
        bool kv_q4 = !(kv_q4_env && kv_q4_env[0] == '0');
        if (opts && opts->kv_format != PN_KV_FORMAT_DEFAULT)
            kv_q4 = opts->kv_format == PN_KV_FORMAT_Q4;   /* the load's own choice */
        /* The KV ring is sized to --ctx-cap when one is set, so gate against
         * that and fall back to the guess only when it is absent. */
        int admit_ctx = (opts && opts->ctx_cap > 0) ? opts->ctx_cap
                                                    : MR_ADMIT_CTX_GUESS;
        uint64_t need_total = 0;
        for (int i = 0; i < n_placed; i++) {
            pn_dm_die d;
            if (pn_dm_get(placed[i], &d) != 0) {
                mr_load_abandon(reg, e, &binding, NULL, NULL, NULL);
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
                mr_load_abandon(reg, e, &binding, NULL, NULL, NULL);
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
        if (pn_dm_bind_ref(placed, n_placed, name, vram_bytes, &binding) != 0) {
            mr_load_abandon(reg, e, &binding, NULL, NULL, NULL);
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
    /* Host-RAM fit (pn_session_host_fit.h): the load sizes the session window
     * so the registry's own state AND every eager scheduler slot fit the host,
     * or refuses the load with the reason. */
    pn_load_opts host_opts = opts ? *opts : (pn_load_opts){0};
    host_opts.host_sessions = 1 + (bs_enabled() && n_placed > 0
                                   ? resolved_max_concurrent : 0);
    host_opts.card_sessions = bs_enabled() && n_placed > 0 ? resolved_max_concurrent : 0;
    uint64_t refusals = pn_host_admit_thread_refusals();
    uint64_t load_refusals = pn_load_thread_refusals();
    pthread_mutex_lock(&reg->upload_lock);
    pn_model *model = pn_load_on(path, &host_opts, n_placed ? placed : NULL,
                                 n_placed);
    pthread_mutex_unlock(&reg->upload_lock);
    pn_ladder_kv_refusal_clear();
    pn_state *state = model ? pn_state_new(model) : NULL;
    if (!model || !state) {
        mr_load_abandon(reg, e, &binding, NULL, state, model);
        if (pn_load_thread_refusals() != load_refusals)
            set_err(err, errlen, pn_load_refusal_last());
        else if (pn_ladder_kv_refusal())
            set_err(err, errlen, pn_ladder_kv_refusal());
        else if (pn_host_admit_thread_refusals() != refusals)
            set_host_refusal_err(err, errlen);
        else
            set_err(err, errlen, model ? "failed to create inference state"
                                       : "failed to load model");
        return -1;
    }

    /* C3: the batch scheduler for a GPU-resident model carves one block per
     * session (qwen35moe: a KV ring row at ctx_cap + SSM image on its dies;
     * the llama family: host KV — batch_sched_slots.h). Created BEFORE
     * publish so a model whose blocks do not fit is refused here, at load,
     * loudly — never discovered by a request. The slot count is the
     * per-model CONCURRENCY cap (max_concurrent, = max_sessions unless set).
     * An architecture the scheduler cannot hold serves single-user. */
    pn_batch_sched *sched = NULL;
    if (bs_enabled() && n_placed > 0 && !bs_slots_supported(model))
        fprintf(stderr, "registry: '%s': no batched session storage for this "
                "architecture; serving single-user\n", name);
    if (bs_enabled() && n_placed > 0 && bs_slots_supported(model)) {
        /* bs_create prechecks host RAM for eager slot states (llama-family
         * host KV) before allocating any (bs_slots_precheck). */
        sched = bs_create(model, resolved_max_concurrent, name);
        if (!sched) {
            mr_load_abandon(reg, e, &binding, NULL, state, model);
            if (pn_ladder_kv_refusal())
                set_err(err, errlen, pn_ladder_kv_refusal());
            else if (pn_host_admit_thread_refusals() != refusals)
                set_host_refusal_err(err, errlen);
            else
                set_err(err, errlen, "session blocks do not fit (lower "
                                     "--max-sessions or --ctx-cap)");
            return -1;
        }
        /* The scheduler is this model's admission authority: its slots hold
         * the sessions, so it alone can tell a new session from a returning
         * one without seating it on someone else's KV. */
        bs_set_session_admission(sched, resolved_max_sessions,
                                 ms_session_idle_ttl_ms());
        fprintf(stderr, "registry: batch scheduler for '%s': %d session "
                "blocks / concurrent decode slots (max_concurrent), %d "
                "resident sessions (max_sessions)\n", name,
                resolved_max_concurrent, resolved_max_sessions);
    }

    /* --- publish ---------------------------------------------------------- */
    if (pthread_mutex_init(&e->gen_lock, NULL) != 0) {
        mr_load_abandon(reg, e, &binding, sched, state, model);
        set_err(err, errlen, "failed to init generation lock");
        return -1;
    }
    if (pthread_mutex_init(&e->pc_lock, NULL) != 0) {
        pthread_mutex_destroy(&e->gen_lock);
        mr_load_abandon(reg, e, &binding, sched, state, model);
        set_err(err, errlen, "failed to init prefix-cache lock");
        return -1;
    }
    pthread_mutex_lock(&reg->lock);
    e->path = strdup(path);
    e->model = model;
    e->state = state;
    e->sched = sched;        /* C3: created above, before publish */
    e->prefix_cache = NULL;  /* PC1: initialised below after unlock */
    memcpy(e->dies, placed, (size_t)n_placed * sizeof(int));
    e->n_dies = n_placed;
    e->die_binding = binding;   /* committed: the unload releases it now */
    e->reasoning_budget = reasoning_budget > 0 ? reasoning_budget : 0;
    e->max_sessions = resolved_max_sessions;
    e->max_concurrent = resolved_max_concurrent;
    e->vram_bytes = vram_bytes;
    e->weights_version = weights_version;
    e->refs = 0;
    e->unloading = false;
    reg->n++;
    pthread_mutex_unlock(&reg->lock);

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

int mr_watch_pin(mr_registry *reg, mr_entry *entry)
{
    int rc = -1;

    if (!reg || !entry)
        return -1;
    pthread_mutex_lock(&reg->lock);
    if (entry->used && entry->model && !entry->unloading) {
        entry->watch_refs++;
        rc = 0;
    }
    pthread_mutex_unlock(&reg->lock);
    return rc;
}

void mr_watch_unpin(mr_registry *reg, mr_entry *entry)
{
    if (!reg || !entry)
        return;
    pthread_mutex_lock(&reg->lock);
    if (entry->watch_refs > 0)
        entry->watch_refs--;
    pthread_cond_broadcast(&reg->drained);
    pthread_mutex_unlock(&reg->lock);
}

void mr_set_unload_hook(mr_registry *reg, mr_unload_hook_fn fn, void *ctx)
{
    if (!reg)
        return;
    pthread_mutex_lock(&reg->lock);
    while (reg->unload_hook_busy > 0)
        pthread_cond_wait(&reg->drained, &reg->lock);
    reg->unload_hook = fn;
    reg->unload_hook_ctx = ctx;
    pthread_mutex_unlock(&reg->lock);
}

bool mr_turn_cancel_requested(const mr_entry *entry)
{
    return entry && __atomic_load_n(&entry->cancel_turns, __ATOMIC_ACQUIRE) != 0;
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

/* Wait for `*count` (an entry's refs or watch_refs) to reach zero, at most
 * `drain_ms` milliseconds (negative = no deadline). Caller holds reg->lock; the
 * lock is held again on return. Returns 0 when drained, -1 on timeout. */
static int drain_count_locked(mr_registry *reg, const int *count, int drain_ms)
{
    if (drain_ms < 0) {
        while (*count > 0)
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

    while (*count > 0) {
        int rc = pthread_cond_timedwait(&reg->drained, &reg->lock, &deadline);
        if (rc == ETIMEDOUT)
            break;
    }
    return *count > 0 ? -1 : 0;
}

/* Steps 1-2 of an unload (see mr_unload): let live turns finish within
 * `drain_ms`, else ask them to stop and wait `cancel_ms` more. Caller holds
 * reg->lock with e->unloading set. Returns 0 when no live reference is left. A
 * negative drain_ms (shutdown) waits without a deadline and never cancels. */
static int drain_live_turns_locked(mr_registry *reg, mr_entry *e, int drain_ms,
                                   int cancel_ms)
{
    if (drain_count_locked(reg, &e->refs, drain_ms) == 0)
        return 0;
    fprintf(stderr, "registry: unload of '%s' — %d live turn(s) still running "
            "after %d ms; cancelling\n", e->name, e->refs, drain_ms);
    __atomic_store_n(&e->cancel_turns, 1, __ATOMIC_RELEASE);
    return drain_count_locked(reg, &e->refs, cancel_ms);
}

/* Step 3 of an unload: run the unload hook (closes the entry's watches) with
 * the registry lock dropped, then wait for the watch pins to go. Caller holds
 * reg->lock; held again on return. Returns 0 when no watch pin is left. */
static int close_watches_locked(mr_registry *reg, mr_entry *e)
{
    mr_unload_hook_fn hook = reg->unload_hook;
    void *ctx = reg->unload_hook_ctx;

    if (hook) {
        reg->unload_hook_busy++;
        pthread_mutex_unlock(&reg->lock);
        hook(ctx, e);
        pthread_mutex_lock(&reg->lock);
        reg->unload_hook_busy--;
        pthread_cond_broadcast(&reg->drained);
    }
    return drain_count_locked(reg, &e->watch_refs, MR_UNLOAD_WATCH_UNPIN_MS);
}

/* Put a refused unload back exactly as it was — resident, acquirable, holding
 * its die binding — and fill the refusal. Caller holds reg->lock; released on
 * return. The alternative (leave it deregistered and return success) is what
 * stranded two dies in the field: the registry said the model was gone while
 * its VRAM and its die budget were still held. */
static int refuse_unload_unlock(mr_registry *reg, mr_entry *e, const char *name,
                                int waited_ms, char *err, size_t errlen)
{
    int refs = e->refs, pins = e->watch_refs;
    char msg[160];

    e->unloading = false;
    __atomic_store_n(&e->cancel_turns, 0, __ATOMIC_RELEASE);
    pthread_mutex_unlock(&reg->lock);
    /* Kept short on purpose: the caller's reason buffer is 160 bytes and a
     * truncated multi-byte dash would land as mojibake in the JSON error. */
    if (refs > 0)
        snprintf(msg, sizeof(msg), "model is busy: %d live turn(s) did not stop "
                 "within %d ms of a cancel; NOT unloaded", refs, waited_ms);
    else
        snprintf(msg, sizeof(msg), "model is busy: %d session watch(es) did not "
                 "close; NOT unloaded", pins);
    set_err(err, errlen, msg);
    fprintf(stderr, "registry: REFUSED unload of '%s' — %s (nothing freed, "
            "nothing stranded)\n", name, msg);
    return -1;
}

/* Unload `name`: give live turns `drain_ms` to finish, then cancel them and
 * give them `cancel_ms` to unwind (MR_UNLOAD_DRAIN_FOREVER = wait, never
 * cancel — the shutdown path), close its watches, free it. A turn that will not
 * stop rolls the entry back — still resident, acquirable, bound to its dies,
 * watches untouched — and the refusal names the reason. */
static int mr_unload_drain(mr_registry *reg, const char *name, int first_die,
                           int drain_ms, int cancel_ms, char *err, size_t errlen)
{
    if (!reg || !name || !*name) {
        set_err(err, errlen, "model name required");
        return -1;
    }
    pthread_mutex_lock(&reg->lock);
    mr_entry *e = mr_find_replica_locked(reg, name, first_die, false);
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
    e->unloading = true;            /* no new acquires or watch pins */

    if (drain_live_turns_locked(reg, e, drain_ms, cancel_ms) != 0)
        return refuse_unload_unlock(reg, e, name, cancel_ms, err, errlen);
    if (close_watches_locked(reg, e) != 0)
        return refuse_unload_unlock(reg, e, name, MR_UNLOAD_WATCH_UNPIN_MS,
                                    err, errlen);
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
    pn_dm_binding_ref die_binding = e->die_binding;
    uint64_t freed_serial = e->serial;
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
    pn_replica_affinity_forget_serial(freed_serial);
    if (had_dies > 0 && pn_dm_release_ref(&die_binding) != 0)
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
    return mr_unload_replica(reg, name, -1, err, errlen);
}

int mr_unload_replica(mr_registry *reg, const char *name, int first_die, char *err,
                      size_t errlen)
{
    return mr_unload_drain(reg, name, first_die,
                           mr_env_ms("PN_UNLOAD_DRAIN_MS", MR_UNLOAD_DRAIN_MS_DEFAULT),
                           mr_env_ms("PN_UNLOAD_CANCEL_MS", MR_UNLOAD_CANCEL_MS_DEFAULT),
                           err, errlen);
}

uint64_t mr_weights_version(const mr_entry *entry)
{
    return entry ? entry->weights_version : 0;
}

int mr_set_sampling_defaults(mr_registry *reg, const char *name,
                             const pn_model_sampling *defaults)
{
    int found = 0;

    if (!reg || !name || !defaults)
        return -1;
    pthread_mutex_lock(&reg->lock);
    for (int i = 0; i < MR_MAX_MODELS; i++) {
        mr_entry *e = &reg->entries[i];
        if (e->used && strncmp(e->name, name, MR_MAX_NAME) == 0) {
            e->sampling = *defaults;
            found = 1;
        }
    }
    pthread_mutex_unlock(&reg->lock);
    return found ? 0 : -1;
}

int mr_set_reasoning_reserve(mr_registry *reg, const char *name, int reserve)
{
    int found = 0;

    if (!reg || !name)
        return -1;
    pthread_mutex_lock(&reg->lock);
    for (int i = 0; i < MR_MAX_MODELS; i++) {
        mr_entry *e = &reg->entries[i];
        if (e->used && strncmp(e->name, name, MR_MAX_NAME) == 0) {
            e->reasoning_reserve = reserve > 0 ? reserve : 0;
            found = 1;
        }
    }
    pthread_mutex_unlock(&reg->lock);
    return found ? 0 : -1;
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
        out[n].sampling = e->sampling;
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
        /* drop_last: the batched turn checkpoints, or (no scheduler) the
         * single-user turn snapshot / the TT card checkpoint. */
        out[n].think_history_drop_last =
            e->sched ? bs_think_history_drop_last(e->sched)
                     : bs_turn_ckpt_enabled() &&
                       pn_prefix_cache_drop_last_supported(e->prefix_cache);
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
        out[n].watch_refs = e->watch_refs;
        out[n].replica = e->replica;
        /* Seats actually held (scheduler slots / cache entries), not routing
         * rows: those outlive a session's state and count watches. Lock-safe
         * here for the reason given for bs_get_load above. */
        ms_occupancy occ;
        ms_session_occupancy(e, &occ);
        out[n].sessions_used = occ.sessions_used;
        out[n].slots_held = occ.slots_held;
        out[n].seat_free = occ.seat_free;
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
        mr_unload_drain(reg, name, -1, MR_UNLOAD_DRAIN_FOREVER,
                        MR_UNLOAD_DRAIN_FOREVER, NULL, 0);
    }
    pthread_cond_destroy(&reg->drained);
    pthread_mutex_destroy(&reg->lock);
    pthread_mutex_destroy(&reg->upload_lock);
}
