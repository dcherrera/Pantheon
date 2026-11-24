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
#include "../accel/die_manager.h"
#include "../core/engine.h"
#include "../core/prefix_cache.h"
#include "batch_sched.h"
#include "sampling_defaults.h"

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
    pn_dm_binding_ref die_binding;              /* this placement's die reservation */
    int              reasoning_budget;          /* per-model AUTO-path reasoning cap
                                                 * (tokens); 0 = disabled. Set at load
                                                 * from `--reasoning-budget` (or the
                                                 * admin body); a request may override
                                                 * it per-turn. See sg_gen_params. */
    int              reasoning_reserve;         /* tokens of max_tokens kept for the
                                                 * answer after a forced close; 0 =
                                                 * PN_REASONING_RESERVE_DEFAULT
                                                 * (reasoning_budget.h). Set right
                                                 * after load (mr_set_reasoning_reserve). */
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
    pn_model_sampling sampling;              /* per-model sampling defaults
                                                 * (A4): a request field that is
                                                 * absent takes these, then the
                                                 * serve fallback. All-absent
                                                 * unless set at load. */
    int              refs;                      /* in-flight acquires: live
                                                 * turns + short admin/route
                                                 * holds. The ONLY count an
                                                 * unload waits out. */
    int              watch_refs;                /* open /session/watch pins
                                                 * (mr_watch_pin). Never delay
                                                 * an unload: it closes them
                                                 * via the unload hook. */
    int              cancel_turns;              /* set by mr_unload once its
                                                 * finish-wait expires; live
                                                 * turns poll it through
                                                 * mr_turn_cancel_requested
                                                 * and abort. Atomic access. */
    uint64_t         serial;                    /* unique per placement (a load);
                                                 * session affinity names a
                                                 * replica by it, so a binding
                                                 * never outlives its replica */
    int              replica;                   /* 1, 2, ...: this copy's number
                                                 * among entries of its name; what
                                                 * a client pins ("replica": N) and
                                                 * X-Pantheon-Replica reports */
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
/* Called by mr_unload once an entry's live turns have drained and it is about
 * to be freed, WITHOUT the registry lock held: the watch manager closes every
 * /session/watch on that entry (terminal `model_unloaded` event) and unpins it.
 * See mr_set_unload_hook. */
typedef void (*mr_unload_hook_fn)(void *ctx, mr_entry *entry);

typedef struct {
    mr_entry        entries[MR_MAX_MODELS];
    int             n;              /* live (used && !unloading) entries */
    pthread_mutex_t lock;
    pthread_cond_t  drained;        /* signaled on every mr_release */
    pthread_mutex_t upload_lock;    /* A3: one GPU upload at a time */
    mr_unload_hook_fn unload_hook;  /* watch manager's close-on-unload */
    void           *unload_hook_ctx;
    int             unload_hook_busy; /* hook calls in progress (guards a
                                       * concurrent mr_set_unload_hook) */
    uint64_t        next_serial;    /* last mr_entry.serial handed out */
} mr_registry;

/* Copy-out row for /v1/models, `pantheon models`, and the admin surface. */
typedef struct {
    char     name[MR_MAX_NAME];
    char     path[512];
    int      dies[PN_MODEL_MAX_DIES];
    int      n_dies;
    uint64_t vram_bytes;
    int      max_sessions;              /* resolved per-model session cap */
    pn_model_sampling sampling;         /* per-model sampling defaults (A4) */
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
    int      refs;                      /* live (in-flight) references */
    int      watch_refs;                /* open /session/watch pins */
    int      replica;                   /* copy number among its name (mr_entry) */
    int      sessions_used;             /* session families holding a seat
                                         * on this replica now (ms_sessions_used:
                                         * scheduler slots or cache entries,
                                         * idle ones included, released or
                                         * reaped ones not) */
    int      slots_held;                /* decode slots held (a family may hold
                                         * more than one; == sessions_used
                                         * without a scheduler) */
    bool     seat_free;                 /* could a NEW session be seated now */
    bool     think_history_drop_last;   /* think_history "drop_last" is served:
                                         * batched turn checkpoints carved
                                         * (bs_think_history_drop_last), or the
                                         * single-user turn snapshot */
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
 * mr_load with an explicit copy number: `replica` >= 1 numbers this copy (a name
 * already holding that number is a refusal); 0 takes the lowest number the name
 * does not use yet. mr_load is mr_load_replica(..., 0, ...).
 *
 * Thread-safe: yes. Reentrant: no. Signal-safe: no
 */
int mr_load_replica(mr_registry *reg, const char *path, const char *name,
                    const pn_load_opts *opts, const int *dies, int n_dies,
                    int reasoning_budget, int max_sessions, int max_concurrent,
                    int replica, char *err, size_t errlen);

/**
 * Look up an entry by name and take a reference. NULL/empty name resolves to
 * the default model (the first loaded). Returns NULL if unknown, unloading,
 * or the registry is empty. Every successful mr_acquire MUST be paired with
 * mr_release.
 */
mr_entry *mr_acquire(mr_registry *reg, const char *name);

/**
 * mr_acquire for a session: when `name` has several replicas (the same model
 * loaded on several die-sets), a session already bound to one goes back to it
 * (its KV lives there); a new session, or a request without one, goes to the
 * least busy replica (live turns + sessions bound to it; ties to the first
 * loaded) and the session is bound there. A bound session that holds no seat
 * on its replica (ms_session_holds_seat: nothing of it lives there — e.g. a
 * watch bound it while only that copy was loaded, or the serve restarted) is
 * placed like a new one on a generation turn (mr_route.place_unseated).
 * NULL/empty name = the default model's name. sid may be NULL. Pair with
 * mr_release.
 *
 * Thread-safe: yes. Reentrant: no. Signal-safe: no
 */
mr_entry *mr_acquire_session(mr_registry *reg, const char *name, const char *sid);

typedef enum {
    MR_ROUTE_OK = 0,
    MR_ROUTE_NO_MODEL,       /* nothing ready under the name */
    MR_ROUTE_NO_REPLICA,     /* the name is loaded, the requested copy is not */
} mr_route_status;

/* One routing decision of mr_acquire_session_on. */
typedef struct {
    int             replica;                   /* in: copy to bind a NEW session
                                                * to; 0 = least busy */
    bool            place_unseated;            /* in: a generation turn — a bound
                                                * session holding no seat on its
                                                * copy is placed like a new one */
    mr_route_status status;                    /* out */
    bool            bound_now;                 /* out: this call bound sid */
    bool            kept_binding;              /* out: sid was bound to another
                                                * copy than `replica`; it stays */
    bool            moved;                     /* out: sid was bound to a copy where
                                                * it held no seat; re-placed */
    int             n_replicas;                /* out: copies of the name */
    int             replicas[MR_MAX_MODELS];   /* out: their numbers, load order */
    char            name[MR_MAX_NAME];         /* out: the name routed on (the default
                                                * model's when none was given) */
} mr_route;

/**
 * mr_acquire_session with a requested copy. A session already bound keeps its
 * copy (kept_binding says the request named another). A new session, or a
 * request without one, goes to copy route->replica when it is > 0 (never to
 * another: a missing copy is MR_ROUTE_NO_REPLICA and NULL), else to the least
 * busy one exactly as mr_acquire_session. route may be NULL (= replica 0).
 * Pair a non-NULL return with mr_release.
 *
 * Thread-safe: yes. Reentrant: no. Signal-safe: no
 */
mr_entry *mr_acquire_session_on(mr_registry *reg, const char *name, const char *sid,
                                mr_route *route);

/**
 * Fail a session over: `from` refused it (no seat, or no slot in time) and it
 * holds no seat there, so take the least busy OTHER ready copy of from's name
 * that has a free seat, move the session's bindings to it and take a
 * reference. NULL when no other copy has a seat (the caller answers the
 * refusal). sid may be NULL. The caller still holds and releases `from`.
 * Pair a non-NULL return with mr_release.
 *
 * Thread-safe: yes. Reentrant: no. Signal-safe: no
 */
mr_entry *mr_failover_session(mr_registry *reg, const mr_entry *from, const char *sid,
                              mr_route *route);

/**
 * Take a reference on every ready copy of `name` (at most `max`), in load
 * order. Returns the count; release each with mr_release.
 *
 * Thread-safe: yes. Reentrant: no. Signal-safe: no
 */
int mr_acquire_replicas(mr_registry *reg, const char *name, mr_entry **out, int max);

/** Drop a reference taken by mr_acquire (signals any draining unload). */
void mr_release(mr_registry *reg, mr_entry *entry);

/**
 * Pin `entry` for an open /session/watch. Unlike mr_acquire this does NOT count
 * as in-flight work: an unload never waits on a watch pin — it closes the watch
 * through the unload hook instead. The caller must already hold an mr_acquire
 * reference on `entry` (so it is resident). @return 0, or -1 when the entry is
 * gone or unloading.
 */
int mr_watch_pin(mr_registry *reg, mr_entry *entry);

/** Drop a pin taken by mr_watch_pin (signals any finishing unload). */
void mr_watch_unpin(mr_registry *reg, mr_entry *entry);

/**
 * Install the unload hook (NULL clears it). One hook per registry: the serve's
 * watch manager. Clearing waits for any hook call in progress to return, so the
 * hook's ctx may be freed right after mr_set_unload_hook(reg, NULL, NULL).
 */
void mr_set_unload_hook(mr_registry *reg, mr_unload_hook_fn fn, void *ctx);

/**
 * True once an unload has asked this entry's live turns to stop (its
 * finish-wait expired). A turn's should_abort polls this so the unload can
 * free the card instead of refusing. Lock-free; safe from any thread holding
 * an mr_acquire reference.
 */
bool mr_turn_cancel_requested(const mr_entry *entry);

/** PM 5: the serve's idle tick — pn_model_keepalive() on every single-user entry whose
 *  gen_lock is free (trylock; a generation in flight is skipped). @return NOPs sent. */
int mr_keepalive(mr_registry *reg);

/** Take `entry->gen_lock` for a generation: trylock, retried for up to 100 ms so the idle
 *  keepalive's few-millisecond hold is not a 429. @return 0 held, else the trylock error. */
int mr_trylock_generation(mr_entry *entry);

/**
 * Unload `name`: refuse new acquires, let live turns finish or cancel them,
 * close every /session/watch on it, release its die binding, and free the
 * model + state.
 *
 *   1. Finish-wait: wait up to PN_UNLOAD_DRAIN_MS (default 5000 ms) for live
 *      references (mr_acquire) to drain. Watch pins are NOT waited on.
 *   2. Cancel: if turns are still live, set the entry's cancel flag (turns
 *      abort at their next should_abort poll) and wait up to
 *      PN_UNLOAD_CANCEL_MS (default 10000 ms) more.
 *   3. Close watches: the unload hook sends each open watch on the entry a
 *      terminal `model_unloaded` event and closes it (in-grace rows too).
 *   4. Free: die binding, scheduler slots + KV, prefix cache, model.
 *
 * A turn that ignores the cancel past step 2 is a REFUSAL: the entry is rolled
 * back to fully resident and acquirable, its watches stay open, nothing is
 * freed. An unload never leaves a model deregistered with its dies held.
 *
 * @return 0 on success; -1 with a reason in err[errlen] — unknown name,
 *         already unloading, or "model is busy: N live turn(s) ..." (a turn
 *         did not stop).
 *
 * Thread-safe: yes (registry lock)
 * Reentrant: no
 * Signal-safe: no
 */
int mr_unload(mr_registry *reg, const char *name, char *err, size_t errlen);

/**
 * mr_unload of one replica: the entry named `name` whose die-set starts at
 * `first_die` (first_die < 0 = the first entry of that name, as mr_unload).
 * Releases exactly that replica's die binding.
 *
 * Thread-safe: yes (registry lock). Reentrant: no. Signal-safe: no
 */
int mr_unload_replica(mr_registry *reg, const char *name, int first_die, char *err,
                      size_t errlen);

/**
 * The entry named `name` whose die-set starts at `first_die` (< 0: any), or
 * NULL. Caller holds reg->lock. require_ready skips loading/unloading slots.
 *
 * Thread-safe: no (caller holds reg->lock). Reentrant: yes. Signal-safe: no
 */
mr_entry *mr_find_replica_locked(mr_registry *reg, const char *name, int first_die,
                                 bool require_ready);

/**
 * The model-identity tag of a held entry (session warm-tier, Phase 7): a stable
 * value derived from the model file's path + size + mtime at load. An exported
 * session blob carries it; a restore refuses a blob whose tag differs (the
 * weights changed underneath). Must be called on an entry the caller holds a
 * reference to (mr_acquire). Returns 0 for a NULL entry.
 */
uint64_t mr_weights_version(const mr_entry *entry);

/**
 * Set the sampling defaults of the resident model `name` (every instance of
 * that name). Call right after mr_load.
 *
 * @return 0, or -1 if no entry has that name
 *
 * Thread-safe: yes (registry lock)
 * Reentrant: no
 * Signal-safe: no
 */
int mr_set_sampling_defaults(mr_registry *reg, const char *name,
                             const pn_model_sampling *defaults);

/**
 * Set the reasoning reserve (sg_gen_params.reasoning_reserve) of the resident
 * model `name` (every instance of that name). Call right after mr_load.
 *
 * @return 0, or -1 if no entry has that name
 *
 * Thread-safe: yes (registry lock)
 * Reentrant: no
 * Signal-safe: no
 */
int mr_set_reasoning_reserve(mr_registry *reg, const char *name, int reserve);

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
