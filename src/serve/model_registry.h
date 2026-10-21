/* SPDX-License-Identifier: AGPL-3.0-or-later */
/* Copyright (C) 2026 David (Corey) Herrera */
/**
 * @file model_registry.h
 * @brief In-memory model registry for multi-model serving:
 *        name → {pn_model handle, pn_state, die assignment, VRAM
 *        footprint, per-model generation lock}. The serve layer routes each
 *        request's `model` field to a resident entry; models load/unload at
 *        runtime so /v1/models is a live list.
 *
 *        Concurrency: entries are refcounted. mr_acquire/mr_release bracket
 *        every use of an entry so mr_unload can drain — it marks the entry
 *        unloading (no new acquires), waits until in-flight refs hit zero,
 *        then frees the model. Generation on one model is serialized by the
 *        entry's own gen_lock (trylock → 429 upstream, exactly the old
 *        single-model behavior); generations on DIFFERENT models run
 *        concurrently (the sampler scratch is _Thread_local).
 *
 *        Placement/VRAM budgeting: mr_load estimates weights from the file
 *        size (GGUF is mmapped 1:1), asks the die manager for a die-set when
 *        one is running (explicit dies win), gates on pn_admit_model_load
 *        against the per-die budget, then pn_load_on + pn_dm_bind. With no
 *        die manager (the Mac / CPU host) entries are CPU-resident (0 dies).
 *
 * @status Registry logic CPU-validated (tests/test_model_registry.c); real
 *         VRAM placement is gated on GPU hardware with the die manager's CUDA probe.
 * @issues None
 * @todo None
 */

#ifndef PN_SERVE_MODEL_REGISTRY_H
#define PN_SERVE_MODEL_REGISTRY_H

#include <pthread.h>
#include <stdbool.h>
#include <stdint.h>

#include "../../include/pantheon.h"
#include "../core/engine.h"
#include "../core/prefix_cache.h"
#include "batch_sched.h"

#define MR_MAX_MODELS 16
#define MR_MAX_NAME   64

/* One resident model. Owned by the registry; use only between mr_acquire and
 * mr_release.
 *
 * C3 (QWEN35_BATCH_SLOTS=1): `sched` replaces the single `state`/`gen_lock`
 * serializer with a multi-slot batch scheduler.  When `sched` is non-NULL, up
 * to BS_MAX_SLOTS users share the model in one batched GPU forward per step;
 * the old trylock→429 is replaced by slot-acquisition (full→429).
 *
 * When QWEN35_BATCH_SLOTS is off (default), `sched` is NULL, `state` and
 * `gen_lock` are used exactly as before — no behaviour change. */
typedef struct {
    char             name[MR_MAX_NAME];
    char            *path;                      /* owned */
    pn_model        *model;
    pn_state        *state;         /* single-user state (sched==NULL path) */
    pthread_mutex_t  gen_lock;      /* single-user generation serializer */
    pn_batch_sched  *sched;         /* C3: multi-slot batch scheduler (may be NULL) */
    pn_prefix_cache *prefix_cache;  /* PC1: session prefix cache (may be NULL);
                                     * keyed by session_id so an agentic turn
                                     * delta-prefills instead of re-prefilling
                                     * the whole context. Guarded per-session by
                                     * the same serializer as generation. */
    pthread_mutex_t  pc_lock;       /* serializes prefix_cache ops (the cache is
                                     * documented thread-unsafe; the batched
                                     * path has concurrent slots). */
    int              dies[PN_MODEL_MAX_DIES];
    int              n_dies;                    /* 0 = CPU-resident */
    int              reasoning_budget;          /* per-model AUTO-path reasoning cap
                                                 * (tokens); 0 = disabled. Set at load
                                                 * from `--reasoning-budget` (or the
                                                 * admin body); a request may override
                                                 * it per-turn. See sg_gen_params. */
    uint64_t         vram_bytes;                /* placement estimate */
    uint64_t         weights_version;           /* model-identity tag for the
                                                 * session warm-tier (Phase 7):
                                                 * derived from path + file size
                                                 * + mtime at load. Stamped into
                                                 * an exported session blob; a
                                                 * restore whose blob carries a
                                                 * different tag is refused (the
                                                 * weights it was captured under
                                                 * no longer match). */
    int              max_sessions;              /* per-model prefix-cache /
                                                 * live-session cap. Set at load
                                                 * from `--max-sessions` (or the
                                                 * admin body); 0/absent → the
                                                 * BS_MAX_USERS default. This is
                                                 * the resolved value actually
                                                 * passed to pn_prefix_cache_new.
                                                 * NOT a concurrency cap — see
                                                 * max_concurrent. */
    int              max_concurrent;            /* per-model BATCH-SLOT cap: how
                                                 * many requests may DECODE at
                                                 * once. Set at load from
                                                 * `--max-concurrent` (or the
                                                 * admin body); 0/absent →
                                                 * max_sessions. Clamped to
                                                 * [1, BS_MAX_USERS] and passed
                                                 * to bs_create as the
                                                 * scheduler's slot count.
                                                 * Distinct from max_sessions:
                                                 * residency is how many
                                                 * conversations may HOLD KV,
                                                 * concurrency is how many may
                                                 * decode simultaneously. */
    int              refs;                      /* in-flight acquires */
    bool             unloading;                 /* draining — no new acquires */
    bool             used;                      /* slot occupancy */
} mr_entry;

/* The registry: a fixed slot table behind one mutex + a drain condvar.
 *
 * A3: upload_lock serializes concurrent GPU weight uploads (the ~2.4 GB
 * transient). The startup loop is sequential already, but a runtime
 * /admin/models/load arriving while a startup upload is in-flight would race
 * and stack two 2.4 GB transients on an 8 GB host. upload_lock prevents that:
 * mr_load acquires it before pn_load_on and releases it after, so at most one
 * GPU weight upload is in flight at any time. The hold is bounded (~55 s on
 * the K80) and well-telegraphed (the caller gets a blocking load, not a 503). */
typedef struct {
    mr_entry        entries[MR_MAX_MODELS];
    int             n;              /* live (used && !unloading) entries */
    pthread_mutex_t lock;
    pthread_cond_t  drained;        /* signaled on every mr_release */
    pthread_mutex_t upload_lock;    /* A3: one GPU upload at a time */
} mr_registry;

/* Copy-out row for /v1/models, `pantheon models`, and the admin surface. */
typedef struct {
    char     name[MR_MAX_NAME];
    char     path[512];
    int      dies[PN_MODEL_MAX_DIES];
    int      n_dies;
    uint64_t vram_bytes;
    int      max_sessions;              /* resolved per-model session cap */
    int      max_concurrent;            /* CONFIGURED per-model batch-slot cap —
                                         * what --max-concurrent asked for. NOT
                                         * what the model actually serves; see
                                         * served_concurrency. Operator surface
                                         * only (/admin). */
    int      served_concurrency;        /* what this model ACTUALLY serves at
                                         * once, read from bs_get_load(): the
                                         * scheduler's slot count, or 1 when
                                         * there is no scheduler (no CUDA build,
                                         * QWEN35_BATCH_SLOTS off, bs_create
                                         * failed, or the memory-pressure ladder
                                         * clamped the slots below the cap).
                                         *
                                         * This is the number /v1/models
                                         * advertises as max_concurrent. It has
                                         * to be the measured one: the incident
                                         * serve advertised max_concurrent 2 on
                                         * a CPU-only binary whose sched was
                                         * NULL, so every second chat took the
                                         * per-model trylock and got 429 while
                                         * the API promised two. */
    int      context_window;            /* EFFECTIVE per-session context window in
                                         * tokens (pn_effective_ctx_window): the
                                         * configured cap after the memory
                                         * ladder's runtime clamp. This is the
                                         * number a CLIENT may plan against, and
                                         * the one /v1/models advertises. */
    int      context_window_configured; /* the operator's --ctx-cap resolution
                                         * (pn_session_ctx_cap), BEFORE the
                                         * ladder. Equal to context_window unless
                                         * the ladder is clamping right now —
                                         * operator surface only (/admin). */
    pn_model_graph_residency graphs;    /* CUDA-graph residency snapshot: how
                                         * many per-layer graph slots are LIVE
                                         * vs FAILED vs INELIGIBLE. A failed
                                         * capture still produces correct
                                         * output, so this counter is the only
                                         * thing that can tell an operator the
                                         * serve quietly got slower. */
    pn_model_attn_residency attn;       /* attention-path residency: how many
                                         * attention layers are running ON-DIE
                                         * vs on the host-softmax fallback, and
                                         * why. Same argument as `graphs`, only
                                         * sharper — the fallback is measured at
                                         * 13.02 -> 1.22 tok/s and still returns
                                         * the right answer. */
    bool     minicpm_tool_dialect;      /* mr_model_is_minicpm: the model speaks
                                         * MiniCPM5's attribute tool dialect.
                                         * /v1/models resolves the dialect from
                                         * this + the name (tc_dialect_for_model). */
    int      refs;
} mr_info;

/* Initialize / destroy. mr_free unloads every resident model (draining each);
 * call only when no more requests can arrive. */
int  mr_init(mr_registry *reg);
void mr_free(mr_registry *reg);

/**
 * Load a model and register it under `name` (NULL → file basename). `dies` /
 * `n_dies` pin an explicit die-set; pass NULL/0 to auto-place via the die
 * manager (or CPU-resident when it is not running). `reasoning_budget` is the
 * per-model AUTO-path reasoning cap in tokens stored on the entry (0 = disabled).
 * `max_sessions` caps the model's prefix cache / live-session count (RESIDENCY:
 * how many named conversations may hold KV at once): <=0 → the BS_MAX_USERS
 * default (byte-identical to the pre-cap behavior), otherwise clamped to
 * [1, BS_MAX_USERS].
 * `max_concurrent` caps the batch scheduler's decode slots (CONCURRENCY: how
 * many requests may decode simultaneously): <=0 → falls back to the resolved
 * `max_sessions`, otherwise clamped to [1, BS_MAX_USERS]. Requests beyond it
 * queue on the scheduler and then get the normal 429/503 backpressure — they
 * are never silently admitted into a wider batch than the model was sized for.
 * Duplicate names, a full table, placement/admission refusal, or a failed
 * pn_load are all clean refusals: returns 0 on success, -1 with a short reason
 * in err[errlen].
 *
 * Loading runs on the caller's thread (a model load can take a while) and
 * does NOT hold the registry lock while pn_load_on runs, so serving continues.
 */
int mr_load(mr_registry *reg, const char *path, const char *name,
            const pn_load_opts *opts, const int *dies, int n_dies,
            int reasoning_budget, int max_sessions, int max_concurrent,
            char *err, size_t errlen);

/**
 * Look up an entry by name and take a reference. NULL/empty name resolves to
 * the default model (the first loaded). Returns NULL if unknown, unloading,
 * or the registry is empty. Every successful mr_acquire MUST be paired with
 * mr_release.
 */
mr_entry *mr_acquire(mr_registry *reg, const char *name);

/** Drop a reference taken by mr_acquire (signals any draining unload). */
void mr_release(mr_registry *reg, mr_entry *entry);

/** PM 5: the serve's idle tick — pn_model_keepalive() on every single-user entry whose
 *  gen_lock is free (trylock; a generation in flight is skipped). @return NOPs sent. */
int mr_keepalive(mr_registry *reg);

/** Take `entry->gen_lock` for a generation: trylock, retried for up to 100 ms so the idle
 *  keepalive's few-millisecond hold is not a 429. @return 0 held, else the trylock error. */
int mr_trylock_generation(mr_entry *entry);

/**
 * Unload `name`: refuse new acquires, wait for in-flight references to drain,
 * release its die binding, and free the model + state.
 *
 * The drain is BOUNDED (PN_UNLOAD_DRAIN_MS, default 15000 ms). Success means
 * the model is gone AND its dies are back: die binding released, scheduler
 * slots and their per-slot KV rings destroyed, prefix cache and model freed. A
 * drain that does not complete in budget is a REFUSAL, not a slow success — the
 * entry is rolled back to fully resident and acquirable and nothing is freed.
 * An unload never leaves a model deregistered with its dies still held.
 *
 * @return 0 on success; -1 with a reason in err[errlen] — unknown name,
 *         already unloading, or "model is busy: N reference(s) still in
 *         flight" (retry once the holders finish).
 *
 * Thread-safe: yes (registry lock)
 * Reentrant: no
 * Signal-safe: no
 */
int mr_unload(mr_registry *reg, const char *name, char *err, size_t errlen);

/**
 * The model-identity tag of a held entry (session warm-tier, Phase 7): a stable
 * value derived from the model file's path + size + mtime at load. An exported
 * session blob carries it; a restore refuses a blob whose tag differs (the
 * weights changed underneath). Must be called on an entry the caller holds a
 * reference to (mr_acquire). Returns 0 for a NULL entry.
 */
uint64_t mr_weights_version(const mr_entry *entry);

/** Number of resident (acquirable) models. */
int mr_count(mr_registry *reg);

/**
 * True when `model` speaks MiniCPM5's tool dialect: a plain-Llama arch whose chat
 * template prepends bos (the MiniCPM5 orchestrator's signature — N2 is
 * PN_ARCH_QWEN35MOE, so the Qwen/N2 path is never mistaken for it). NULL → false.
 *
 * Thread-safe: yes (reads the model config)
 */
bool mr_model_is_minicpm(const pn_model *model);

/** Snapshot up to `max` resident models into out[]; returns the count. */
int mr_snapshot(mr_registry *reg, mr_info *out, int max);

#endif /* PN_SERVE_MODEL_REGISTRY_H */
