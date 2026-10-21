/* SPDX-License-Identifier: AGPL-3.0-or-later */
/* Copyright (C) 2026 David (Corey) Herrera */
/**
 * @file serve_command.c
 * @brief Implementation of `pantheon serve` (see serve_command.h). Loads one
 *        or more models into the model registry and runs an accept loop that
 *        handles each connection on its own detached thread, so lightweight
 *        requests (e.g. GET /v1/models health checks) respond immediately even
 *        while a chat is generating.
 *
 * Concurrency model:
 *   - IO is concurrent: one detached pthread per accepted connection. accept()
 *     and cheap endpoints never block behind an in-flight generation.
 *   - Generation is serialized PER MODEL: each registry entry owns its
 *     pn_model + pn_state (KV / SSM state) and a gen_lock; the chat path
 *     (openai_chat.c) routes the request's `model` field to an entry and takes
 *     a non-blocking trylock on it — a second chat to the SAME model is
 *     answered with a clean 429 (busy), while chats to DIFFERENT models run
 *     concurrently on their own dies/threads. Health checks and other cheap
 *     routes never touch a lock.
 *   (Batching concurrent chats to one model instead of 429-ing them is future
 *    work — see the multi-user batching task.)
 *
 * Shutdown: SIGINT/SIGTERM are RECORDED by the handler (serve_shutdown.h) and
 * acted on here. The accept loop polls the shutdown wake pipe alongside the
 * listening socket; on a request it stops accepting, drains the in-flight turns
 * — which stop themselves through the abort predicate — and only then frees the
 * models, which is the path that retires a Tenstorrent card's residencies. A
 * drain that does not quiesce ABANDONS the teardown rather than running it
 * under a live thread.
 *
 * Multi-model (Phase 7): `--model` is repeatable; each may be followed by
 * `--name <id>` (served alias) and `--dies <a[,b,...]>` (explicit N-die placement on
 * the K80 test machine). With a die manager running (CUDA present) unnamed placements are
 * assigned automatically. `--served-name` remains the back-compat alias for
 * the FIRST model's name. Runtime load/unload arrives with the admin routes
 * (model_admin.c).
 *
 * Backend honesty: the first line this process writes is its COMPILED backend
 * and build commit, and the second is how many accelerator dies the manager can
 * see. Then any `--dies` that this binary cannot honour is a refusal to start,
 * not a warning (serve_backend_guard.h has the incident it prevents). A CPU
 * serve is asked for by simply not passing `--dies`, and is unaffected.
 *
 * @status Multi-model serving: registry-backed routing, per-model
 *         serialization, live /v1/models. Single `--model` behaves exactly as
 *         the P1 single-model server did.
 * @issues Concurrent chats to one model are 429'd, not batched (deferred to
 *         the batching task).
 * @todo None
 */

#include "serve_command.h"

#include <errno.h>
#include <poll.h>
#include <pthread.h>
#include <signal.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <unistd.h>

#include "../../include/pantheon.h"
#include "../core/engine.h"
#include "../core/mem/pn_memwatch.h"
#include "../accel/backend_identity.h"
#include "../accel/die_manager.h"
#include "http_server.h"
#include "model_admin.h"
#include "model_registry.h"
#include "openai_chat.h"
#include "request_profile.h"
#include "serve_backend_guard.h"
#include "serve_registry.h"
#include "serve_shutdown.h"
#include "session_watch.h"
#include "signal_handlers.h"

#ifdef USE_CUDA
#include "cuda/fc_cuda.h"
#endif

#define SERVE_DEFAULT_HOST "127.0.0.1"
#define SERVE_DEFAULT_PORT 8080

/* One `--model` occurrence and the per-model flags that followed it. */
typedef struct {
    const char *path;
    const char *name;                   /* NULL = file basename */
    int         dies[PN_MODEL_MAX_DIES];
    int         n_dies;                 /* 0 = auto-place / CPU */
    int         reasoning_budget;       /* --reasoning-budget; 0 = disabled */
    int         max_sessions;           /* --max-sessions; 0 = engine default */
    int         max_concurrent;         /* --max-concurrent; batch-slot cap.
                                         * 0 = fall back to max_sessions. */
    int         ctx_cap;                /* --ctx-cap; per-session context cap in
                                         * tokens (KV ring window). 0 = engine
                                         * default (QWEN35_MAX_SEQ / fallback). */
} serve_model_arg;

/* Shared, read-mostly server context handed to every connection thread: the
 * model registry (which owns per-model locks/refcounts internally). */
typedef struct {
    mr_registry          *reg;
    pn_session_watch_mgr *watch;   /* liveness-based session release (§C.4);
                                    * NULL if the watch manager failed to start
                                    * (serve still runs — watches are refused). */
} serve_ctx;

/* Per-connection job: the accepted client fd plus the shared context. Heap
 * allocated by the accept loop and freed by the connection thread. */
typedef struct {
    int        client_fd;
    serve_ctx *ctx;
} connection_job;

static void print_serve_usage(void)
{
    printf("Usage: pantheon serve --model <path> [--name <id>] [--model <path> ...] [options]\n\n");
    printf("Model options (repeat --model to serve several; --name/--dies bind\n");
    printf("to the most recent --model):\n");
    printf("  --model <path>        Model file (GGUF); at least one required\n");
    printf("  --name <id>           Served model id (default: file name)\n");
    printf("  --dies <a[,b,...]>    Pin this model to accelerator die(s) (CUDA/Tenstorrent\n");
    printf("                        builds; CUDA indices are relative to\n");
    printf("                        CUDA_VISIBLE_DEVICES). N dies split the layers N ways,\n");
    printf("                        evenly, LM head on the last. That buys VRAM headroom\n");
    printf("                        for a wider --ctx-cap, NOT speed: each extra die adds\n");
    printf("                        one activation hand-off per step. Use the fewest dies\n");
    printf("                        that hold the model at the cap you want.\n");
    printf("                        Omit to let the engine place it\n");
    printf("                        — and OMIT IT TO SERVE ON THE CPU DELIBERATELY: a\n");
    printf("                        --dies that names dies this binary cannot use is a\n");
    printf("                        hard refusal to start, never a silent CPU fallback.\n");
    printf("  --reasoning-budget <n>  Cap reasoning to <n> tokens on this model: once\n");
    printf("                        the model has generated <n> tokens inside <think>\n");
    printf("                        without closing it, force </think> so it answers\n");
    printf("                        (reasoning-runaway guard; 0/absent = unbounded)\n");
    printf("  --max-sessions <n>    Cap this model's live prefix-cache sessions to <n>\n");
    printf("                        (VRAM-sized; a new session over the cap is refused\n");
    printf("                        429/503; 0/absent = default %d)\n", BS_MAX_USERS);
    printf("  --max-concurrent <n>  Cap this model's SIMULTANEOUS decode slots to <n>\n");
    printf("                        (batch scheduler width; surplus requests queue and\n");
    printf("                        then get 429/503. Distinct from --max-sessions,\n");
    printf("                        which caps resident sessions. 0/absent = the\n");
    printf("                        resolved --max-sessions)\n");
    printf("  --ctx-cap <n>         Per-session context cap in tokens for this model:\n");
    printf("                        the KV ring window. A prompt past it is refused with\n");
    printf("                        context_over_budget (client compacts + retries) — no\n");
    printf("                        silent drop-oldest. 0/absent = engine default.\n");
    printf("  --served-name <id>    Back-compat alias for the FIRST model's --name\n");
    printf("\nGlobal options:\n");
    printf("  --host <addr>         Bind address (default %s)\n", SERVE_DEFAULT_HOST);
    printf("  --port <n>            Port (default %d)\n", SERVE_DEFAULT_PORT);
    printf("  --config <path>       HuggingFace config.json (safetensors models)\n");
    printf("  --vocab <path>        RWKV World tokenizer vocab file\n");
    printf("  --quantize            INT8 quantize weights at load time\n");
    printf("\nEndpoints: GET /v1/models, POST /v1/chat/completions (SSE stream supported),\n");
    printf("           GET /admin/models, POST /admin/models/load, POST /admin/models/unload,\n");
    printf("           GET /admin/dies,\n");
    printf("           POST /admin/session/export, POST /session/restore, POST /session/release,\n");
    printf("           GET /session/estimate\n");
}

/* Parse a comma-separated die list ("0,1") into out[0..max). Returns the
 * count, or -1 on a malformed list. */
static int parse_dies_list(const char *s, int *out, int max)
{
    int n = 0;
    while (*s) {
        char *end = NULL;
        long v = strtol(s, &end, 10);
        if (end == s || v < 0 || n >= max)
            return -1;
        out[n++] = (int)v;
        s = end;
        if (*s == ',')
            s++;
        else if (*s)
            return -1;
    }
    return n > 0 ? n : -1;
}

/* Extract query param `key` from an "a=1&b=two" query string into out[0..cap).
 * Returns 1 if found (out is NUL-terminated), 0 otherwise. No URL-decoding —
 * the Phase-7 params (model name, integers) are plain ASCII. */
static int query_param_str(const char *query, const char *key, char *out,
                           size_t cap)
{
    size_t klen = strlen(key);
    for (const char *p = query; p && *p;) {
        const char *amp = strchr(p, '&');
        const char *eq = strchr(p, '=');
        if (eq && (!amp || eq < amp) && (size_t)(eq - p) == klen &&
            strncmp(p, key, klen) == 0) {
            const char *vstart = eq + 1;
            const char *vend = amp ? amp : vstart + strlen(vstart);
            size_t vlen = (size_t)(vend - vstart);
            if (vlen >= cap) vlen = cap - 1;
            memcpy(out, vstart, vlen);
            out[vlen] = '\0';
            return 1;
        }
        if (!amp) break;
        p = amp + 1;
    }
    if (cap) out[0] = '\0';
    return 0;
}

/* Query param `key` parsed as a double, or `fallback` if absent/malformed. */
static double query_param_num(const char *query, const char *key,
                              double fallback)
{
    char buf[64];
    if (!query_param_str(query, key, buf, sizeof(buf)) || !buf[0])
        return fallback;
    char *end = NULL;
    double v = strtod(buf, &end);
    return (end != buf && *end == '\0') ? v : fallback;
}

/* GET /session/watch?name=&session_id= — the liveness watch (§C.4). The client
 * holds this connection open for the session's life; while it is held the
 * session is ALIVE (never reaped for inactivity), and only the connection
 * closing (client death/drop) frees the slot after a grace. On success the fd
 * is ADOPTED by the watch manager (this returns 1 so the connection thread does
 * NOT close it); on any refusal an error is sent and the fd is closed normally
 * (returns 0). */
static int handle_session_watch(int cfd, serve_ctx *ctx, const hs_request *req)
{
    if (!ctx->watch) {
        hs_send_error(cfd, 503, "watch_unavailable",
                      "session watch manager is not running");
        return 0;
    }
    char name[MR_MAX_NAME] = "";
    char sid[64] = "";
    const char *q = strchr(req->path, '?');
    if (q) {
        query_param_str(q + 1, "name", name, sizeof(name));
        query_param_str(q + 1, "session_id", sid, sizeof(sid));
    }
    /* Fall back to the X-Session-Id header if the query omitted session_id. */
    if (!sid[0] && req->session_id && req->session_id[0])
        snprintf(sid, sizeof(sid), "%s", req->session_id);
    if (!sid[0]) {
        hs_send_error(cfd, 400, "invalid_request_error",
                      "'session_id' query param is required");
        return 0;
    }

    mr_entry *entry = mr_acquire(ctx->reg, name[0] ? name : NULL);
    if (!entry) {
        hs_send_error(cfd, 404, "model_not_found",
                      "the requested model is not loaded (see GET /v1/models)");
        return 0;
    }
    if (!entry->prefix_cache) {
        mr_release(ctx->reg, entry);
        hs_send_error(cfd, 400, "session_caching_disabled",
                      "this model has no session cache to watch");
        return 0;
    }
    /* Route with the entry's resolved name (name[] may be empty → default). */
    int rc = pn_session_watch_add(ctx->watch, cfd, entry,
                                  entry->name, sid);
    mr_release(ctx->reg, entry);   /* the manager took its own reference */
    if (rc != 0) {
        hs_send_error(cfd, 503, "watch_unavailable",
                      "session watch table is full; retry shortly");
        return 0;
    }
    return 1;   /* fd adopted by the watch manager — do not close it */
}

/* Route one parsed request to the matching handler. Cheap routes answer inline
 * (no lock); the chat completion route serializes per model inside
 * oai_handle_chat. Returns 1 if the connection fd was ADOPTED (a held
 * /session/watch — the caller must NOT close it), 0 otherwise. */
static int dispatch(int cfd, serve_ctx *ctx, const hs_request *req)
{
    if (strcmp(req->method, "OPTIONS") == 0) {
        hs_send_json(cfd, 200, "{}");   /* CORS preflight */
        return 0;
    }
    if (strcmp(req->method, "GET") == 0 &&
        strncmp(req->path, "/session/watch", 14) == 0 &&
        (req->path[14] == '\0' || req->path[14] == '?')) {
        return handle_session_watch(cfd, ctx, req);
    }
    if (strcmp(req->method, "GET") == 0 && strcmp(req->path, "/v1/models") == 0) {
        oai_handle_models(cfd, ctx->reg);
        return 0;
    }
    if (strcmp(req->method, "POST") == 0 &&
        strcmp(req->path, "/v1/chat/completions") == 0) {
        /* THE route that drives the accelerator, so it is the one bracketed for
         * the shutdown drain: nothing may retire a card residency while a turn
         * is still submitting descriptors against it. The bracket is wider than
         * the generation itself (it includes the parse and the routing), and
         * that is the safe direction — it can only make a drain wait longer,
         * never let it finish early. */
        if (pn_serve_generation_enter() != 0) {
            hs_send_error_ex(cfd, 503, "server_shutting_down",
                             "the server is shutting down; retry shortly",
                             "Retry-After: 1\r\n");
            return 0;
        }
        oai_handle_chat(cfd, ctx->reg, req->body, req->session_id);
        pn_serve_generation_leave();
        return 0;
    }
    if (strcmp(req->method, "POST") == 0 &&
        strcmp(req->path, "/admin/models/load") == 0) {
        ma_handle_load(cfd, ctx->reg, req->body);
        return 0;
    }
    if (strcmp(req->method, "POST") == 0 &&
        strcmp(req->path, "/admin/models/unload") == 0) {
        ma_handle_unload(cfd, ctx->reg, req->body);
        return 0;
    }
    if (strcmp(req->method, "GET") == 0 && strcmp(req->path, "/admin/models") == 0) {
        ma_handle_models(cfd, ctx->reg);
        return 0;
    }
    if (strcmp(req->method, "GET") == 0 && strcmp(req->path, "/admin/dies") == 0) {
        ma_handle_dies(cfd);
        return 0;
    }
    if (strcmp(req->method, "POST") == 0 &&
        strcmp(req->path, "/admin/session/export") == 0) {
        ma_handle_session_export(cfd, ctx->reg, req->body);
        return 0;
    }
    if (strcmp(req->method, "POST") == 0 &&
        strcmp(req->path, "/session/restore") == 0) {
        oai_handle_session_restore(cfd, ctx->reg, req->body, req->session_id);
        return 0;
    }
    if (strcmp(req->method, "POST") == 0 &&
        strcmp(req->path, "/session/release") == 0) {
        oai_handle_session_release(cfd, ctx->reg, req->body, req->session_id);
        return 0;
    }
    if (strcmp(req->method, "GET") == 0 &&
        strncmp(req->path, "/session/estimate", 17) == 0 &&
        (req->path[17] == '\0' || req->path[17] == '?')) {
        char name[MR_MAX_NAME] = "";
        int tokens = 0;
        double blob_bytes = 0.0, upload_bps = 0.0;
        const char *q = strchr(req->path, '?');
        if (q) {
            query_param_str(q + 1, "name", name, sizeof(name));
            tokens = (int)query_param_num(q + 1, "tokens", 0.0);
            blob_bytes = query_param_num(q + 1, "blob_bytes", 0.0);
            upload_bps = query_param_num(q + 1, "upload_bps", 0.0);
        }
        oai_handle_session_estimate(cfd, ctx->reg, name[0] ? name : NULL,
                                    tokens, blob_bytes, upload_bps);
        return 0;
    }
    if (strcmp(req->method, "GET") == 0 && strcmp(req->path, "/") == 0) {
        /* The health probe answers with the backend too. This is the cheapest
         * route on the server and the one a monitor already hits, so it is the
         * cheapest place for "what am I running" to be a GET instead of an ssh
         * + ldd. Values are compile-time literals + a lock-free die count, so
         * the route stays allocation-free and never blocks. */
        char health[256];
        snprintf(health, sizeof(health),
                 "{\"service\":\"pantheon\",\"status\":\"ok\",\"backend\":\"%s\","
                 "\"build\":\"%s\",\"accelerator\":%s,\"dies_visible\":%d}",
                 pn_backend_token(), pn_backend_build_commit(),
                 pn_backend_has_accelerator() ? "true" : "false", pn_dm_count());
        hs_send_json(cfd, 200, health);
        return 0;
    }
    hs_send_error(cfd, 404, "not_found", "unknown route");
    return 0;
}

/* Read, route, and answer one connection, then close it — UNLESS the route
 * adopted the fd (a held /session/watch, which the watch manager now owns and
 * will close itself). Shared between the per-connection thread and the inline
 * fallback path. */
static void handle_connection(int cfd, serve_ctx *ctx)
{
    hs_request req;
    int adopted = 0;
    pn_reqprof_begin();
    pn_reqprof_phase_start(PN_REQ_HTTP_READ);
    int read_rc = hs_read_request(cfd, &req);
    pn_reqprof_phase_end(PN_REQ_HTTP_READ);
    if (read_rc == 0) {
        fprintf(stderr, "serve: %s %s\n", req.method, req.path);
        pn_reqprof_note_route(req.method, req.path);
        adopted = dispatch(cfd, ctx, &req);
        hs_request_free(&req);
    }
    pn_reqprof_end();
    if (!adopted)
        hs_close(cfd);
}

/* Detached thread entry: own one client connection end to end, then release the
 * job. A slow/blocking client can only stall its own thread, never accept(). */
static void *connection_thread(void *arg)
{
    connection_job *job = arg;
    handle_connection(job->client_fd, job->ctx);
    free(job);
    return NULL;
}

/* Spawn a detached connection thread. On failure (e.g. thread limit) fall back
 * to handling the connection inline so the request is never dropped. */
static void spawn_connection(int cfd, serve_ctx *ctx)
{
    connection_job *job = malloc(sizeof(*job));
    if (!job) { handle_connection(cfd, ctx); return; }
    job->client_fd = cfd;
    job->ctx = ctx;

    pthread_attr_t attr;
    if (pthread_attr_init(&attr) != 0) { handle_connection(cfd, ctx); free(job); return; }
    pthread_attr_setdetachstate(&attr, PTHREAD_CREATE_DETACHED);

    pthread_t tid;
    int rc = pthread_create(&tid, &attr, connection_thread, job);
    pthread_attr_destroy(&attr);
    if (rc != 0) {
        fprintf(stderr, "serve: pthread_create failed (%d); handling inline\n", rc);
        handle_connection(cfd, ctx);
        free(job);
    }
}

/* Milliseconds on CLOCK_MONOTONIC — the drain measures a budget, so it must not
 * move when the wall clock does. */
static uint64_t serve_monotonic_ms(void)
{
    struct timespec ts;

    if (clock_gettime(CLOCK_MONOTONIC, &ts) != 0)
        return 0;
    return (uint64_t)ts.tv_sec * 1000u + (uint64_t)(ts.tv_nsec / 1000000);
}

/* Poll interval of the drain. Short enough that a quiesced serve exits without
 * a perceptible pause, long enough that waiting costs nothing on a 2-core host. */
#define SERVE_DRAIN_POLL_MS 20

/*
 * Wait for every in-flight accelerator-driving turn to finish, and report
 * whether the teardown may proceed.
 *
 * The turns are not interrupted — they are ASKED, through the abort predicate
 * openai_chat.c already arms, which they poll between prefill chunks and every
 * 8 decode tokens. A second signal collapses the budget to zero so an operator
 * hammering Ctrl-C gets out through this path instead of an _exit() from a
 * handler.
 */
static pn_shutdown_drain_decision serve_drain_generations(void)
{
    const uint64_t started = serve_monotonic_ms();
    int reported = 0;

    for (;;) {
        int inflight = pn_serve_generation_inflight();
        unsigned budget = pn_serve_shutdown_escalated()
                              ? 0u : pn_serve_shutdown_drain_budget_ms();
        unsigned elapsed = (unsigned)(serve_monotonic_ms() - started);
        pn_shutdown_drain_decision decision =
            pn_serve_shutdown_drain_decision(inflight, elapsed, budget);

        if (decision != PN_SHUTDOWN_DRAIN_WAIT)
            return decision;
        if (!reported) {
            fprintf(stderr, "serve: waiting up to %u ms for %d in-flight turn%s "
                            "to stop before retiring the accelerator\n",
                    budget, inflight, inflight == 1 ? "" : "s");
            reported = 1;
        }
        usleep(SERVE_DRAIN_POLL_MS * 1000);
    }
}

/*
 * The controlled-shutdown sequence, run on the accept-loop thread once the
 * signal handler's self-pipe has woken it. This is the point where the
 * program's state is known, which is the whole reason the handler defers here.
 *
 * Order is load-bearing:
 *   1. stop accepting, so the in-flight set can only shrink;
 *   2. leave the router, so nothing new is dispatched to this instance;
 *   3. DRAIN — the abort predicate has already told every live turn to stop;
 *   4. only if it quiesced, tear down: the watch manager first (it holds
 *      mr_acquire refs, and mr_free's per-entry drain waits on refs hitting
 *      zero — stopping it after would deadlock the shutdown), then the model
 *      registry, whose per-model free is what reaches pn_llama_tt_free ->
 *      pn_tt_device_close -> pn_tt_residency_release and retires the card's
 *      residencies BY ASKING.
 *
 * On a drain timeout nothing is retired. Retiring under a live thread is worse
 * than not retiring at all: pn_tt_device_close frees the device that thread is
 * still dereferencing, so the tidy-looking teardown ends in a SIGSEGV and a
 * core held with no host attached. What is left instead is exactly what a
 * SIGKILL leaves, which SET_NOC_CLEANUP and recovery-on-acquire already cover.
 */
static void serve_run_shutdown(int listen_fd, pn_session_watch_mgr *watch,
                               mr_registry *reg)
{
    pn_shutdown_drain_decision decision;

    hs_close(listen_fd);
    serve_registry_deregister_all();

    decision = serve_drain_generations();
    if (decision == PN_SHUTDOWN_DRAIN_RETIRE) {
        /* The registry heartbeat thread (Phase G0, only alive when
         * PANTHEON_REGISTRY_URL is set) still takes brief mr_acquire/mr_release
         * pairs here. It is not a starvation risk — mr_acquire skips an entry
         * once mr_unload marks it `unloading` — and the re-raise below follows
         * immediately, so it is left running rather than given a stop path it
         * has never needed. */
        pn_session_watch_stop(watch);
        mr_free(reg);
        pn_dm_shutdown();
        fprintf(stderr, "serve: shutdown complete — models freed and "
                        "accelerator residencies retired\n");
    } else {
        fprintf(stderr,
                "serve: DRAIN TIMED OUT with %d turn(s) still on the "
                "accelerator — NOT retiring residencies under a live thread. "
                "The card is left to SET_NOC_CLEANUP and recovery-on-acquire, "
                "exactly as after a SIGKILL.\n",
                pn_serve_generation_inflight());
    }

#ifdef USE_CUDA
    /* Kepler reclaims VRAM only when the owning process's /dev/nvidiaN fd
     * closes, so this runs on BOTH branches — it is about the host's fd, not
     * about anything a stalled turn is holding. */
    if (fc_cuda_live_ctx_count() > 0)
        fc_cuda_emergency_teardown();
#endif
}

/*
 * Refuse every `--model ... --dies ...` this binary cannot honour, BEFORE any
 * model is loaded.
 *
 * Pre-flighting all specs (rather than letting mr_load refuse the first one) is
 * what makes the failure cheap: a bad manifest is answered in a second instead
 * of after a ~55 s weight upload, and the operator sees every offending model
 * in one go rather than one per restart. mr_load re-checks the same rule for
 * the runtime /admin/models/load path — this is the fast, talkative copy.
 *
 * Returns 1 if the serve must stop (message already printed), 0 to proceed.
 */
static int refuse_dies_without_backend(const serve_model_arg *specs, int n_specs)
{
    const bool die_backend = pn_backend_supports_die_placement();
    const int  dies_visible = pn_dm_count();
    int refused = 0;

    for (int i = 0; i < n_specs; i++) {
        pn_backend_guard_verdict v =
            pn_backend_guard_check(specs[i].n_dies, die_backend, dies_visible);
        char msg[640];

        if (v == PN_BACKEND_GUARD_OK)
            continue;
        pn_backend_guard_message(v, specs[i].name ? specs[i].name : specs[i].path,
                                 specs[i].dies, specs[i].n_dies, msg, sizeof(msg));
        fprintf(stderr, "%s\n", msg);
        refused++;
    }
    return refused > 0;
}

int pn_serve_main(int argc, char **argv)
{
    serve_model_arg specs[MR_MAX_MODELS];
    int n_specs = 0;
    const char *config_path = NULL;
    const char *vocab_path = NULL;
    const char *served_name = NULL;
    const char *host = SERVE_DEFAULT_HOST;
    int port = SERVE_DEFAULT_PORT;
    int quantize = 0;

    /* FIRST line this process writes. The backend is a compile-time constant
     * and the only reason it was ever unknown is that nothing printed it: a
     * CPU-only binary served a 35B MoE off a 2-core Pentium with 8 dies idle,
     * and answering "what am I running" took `ldd` + `strings` on a remote box.
     * One line, one key=value per fact, greppable out of a fleet log. */
    fprintf(stderr, "serve: backend=%s build=%s accelerator=%s\n",
            pn_backend_name(), pn_backend_build_commit(),
            pn_backend_has_accelerator() ? "yes" : "no");

    memset(specs, 0, sizeof(specs));
    for (int i = 1; i < argc; i++) {
        if (strcmp(argv[i], "--help") == 0 || strcmp(argv[i], "-h") == 0) {
            print_serve_usage();
            return 0;
        } else if (strcmp(argv[i], "--model") == 0 && i + 1 < argc) {
            if (n_specs >= MR_MAX_MODELS) {
                fprintf(stderr, "serve: too many --model (max %d)\n", MR_MAX_MODELS);
                return 1;
            }
            specs[n_specs++].path = argv[++i];
        } else if (strcmp(argv[i], "--name") == 0 && i + 1 < argc) {
            if (n_specs == 0) {
                fprintf(stderr, "serve: --name must follow a --model\n");
                return 1;
            }
            specs[n_specs - 1].name = argv[++i];
        } else if (strcmp(argv[i], "--dies") == 0 && i + 1 < argc) {
            if (n_specs == 0) {
                fprintf(stderr, "serve: --dies must follow a --model\n");
                return 1;
            }
            serve_model_arg *m = &specs[n_specs - 1];
            m->n_dies = parse_dies_list(argv[++i], m->dies, PN_MODEL_MAX_DIES);
            if (m->n_dies < 0) {
                fprintf(stderr, "serve: bad --dies list '%s'\n", argv[i]);
                return 1;
            }
        } else if (strcmp(argv[i], "--reasoning-budget") == 0 && i + 1 < argc) {
            if (n_specs == 0) {
                fprintf(stderr, "serve: --reasoning-budget must follow a --model\n");
                return 1;
            }
            /* strtol (not atoi): a typo'd non-numeric arg must ERROR, not silently
             * parse to 0 and disable the guard. */
            const char *rbarg = argv[++i];
            char *rbend = NULL;
            long rb = strtol(rbarg, &rbend, 10);
            if (rbend == rbarg || *rbend != '\0' || rb < 0 || rb > (1 << 20)) {
                fprintf(stderr, "serve: bad --reasoning-budget '%s' "
                        "(want a token count 0..%d)\n", rbarg, 1 << 20);
                return 1;
            }
            specs[n_specs - 1].reasoning_budget = (int)rb;
        } else if (strcmp(argv[i], "--ctx-cap") == 0 && i + 1 < argc) {
            if (n_specs == 0) {
                fprintf(stderr, "serve: --ctx-cap must follow a --model\n");
                return 1;
            }
            /* strtol (mirrors --reasoning-budget): a typo must ERROR, not
             * silently disable the per-model cap. Accept a token count up to the
             * hybrid ring ceiling; 0 = engine default. */
            const char *ccarg = argv[++i];
            char *ccend = NULL;
            long cc = strtol(ccarg, &ccend, 10);
            if (ccend == ccarg || *ccend != '\0' || cc < 0 || cc > (1 << 20)) {
                fprintf(stderr, "serve: bad --ctx-cap '%s' "
                        "(want a token count 0..%d)\n", ccarg, 1 << 20);
                return 1;
            }
            specs[n_specs - 1].ctx_cap = (int)cc;
        } else if (strcmp(argv[i], "--max-sessions") == 0 && i + 1 < argc) {
            if (n_specs == 0) {
                fprintf(stderr, "serve: --max-sessions must follow a --model\n");
                return 1;
            }
            /* strtol (mirrors --reasoning-budget): a typo'd non-numeric arg must
             * ERROR, not silently parse to 0 and revert to the default cap.
             * Accept 0 (= engine default) up to BS_MAX_USERS (the batch
             * scheduler's user-slot ceiling; mr_load clamps anyway). */
            const char *msarg = argv[++i];
            char *msend = NULL;
            long ms = strtol(msarg, &msend, 10);
            if (msend == msarg || *msend != '\0' || ms < 0 || ms > BS_MAX_USERS) {
                fprintf(stderr, "serve: bad --max-sessions '%s' "
                        "(want a session count 0..%d; 0 = default)\n",
                        msarg, BS_MAX_USERS);
                return 1;
            }
            specs[n_specs - 1].max_sessions = (int)ms;
        } else if (strcmp(argv[i], "--max-concurrent") == 0 && i + 1 < argc) {
            if (n_specs == 0) {
                fprintf(stderr, "serve: --max-concurrent must follow a --model\n");
                return 1;
            }
            /* Same strtol discipline as --max-sessions: a typo must ERROR
             * rather than silently widen the batch back to the default. */
            const char *mcarg = argv[++i];
            char *mcend = NULL;
            long mc = strtol(mcarg, &mcend, 10);
            if (mcend == mcarg || *mcend != '\0' || mc < 0 || mc > BS_MAX_USERS) {
                fprintf(stderr, "serve: bad --max-concurrent '%s' "
                        "(want a slot count 0..%d; 0 = use --max-sessions)\n",
                        mcarg, BS_MAX_USERS);
                return 1;
            }
            specs[n_specs - 1].max_concurrent = (int)mc;
        } else if (strcmp(argv[i], "--host") == 0 && i + 1 < argc) {
            host = argv[++i];
        } else if (strcmp(argv[i], "--port") == 0 && i + 1 < argc) {
            port = atoi(argv[++i]);
        } else if (strcmp(argv[i], "--served-name") == 0 && i + 1 < argc) {
            served_name = argv[++i];
        } else if (strcmp(argv[i], "--config") == 0 && i + 1 < argc) {
            config_path = argv[++i];
        } else if (strcmp(argv[i], "--vocab") == 0 && i + 1 < argc) {
            vocab_path = argv[++i];
        } else if (strcmp(argv[i], "--quantize") == 0) {
            quantize = 1;
        } else {
            fprintf(stderr, "serve: unknown option '%s'\n", argv[i]);
            print_serve_usage();
            return 1;
        }
    }

    if (n_specs == 0) {
        fprintf(stderr, "serve: --model is required\n");
        print_serve_usage();
        return 1;
    }
    /* Back-compat: --served-name names the first model when it has no --name. */
    if (served_name && !specs[0].name)
        specs[0].name = served_name;

    /* L15 — serve config hygiene: QWEN35_GPU_FULL_LOGITS restores the full
     * per-token n_vocab (248k) D2H copy over the ~0.4 GB/s host link (K*992 KB
     * per step) — a debug/eval knob that must NEVER be set on the serve path,
     * where it silently caps throughput. The batched serve always presents the
     * sparse device top-k. Guard at start: refuse to serve with it set rather
     * than serve slowly and silently. */
    if (getenv("QWEN35_GPU_FULL_LOGITS")) {
        fprintf(stderr,
            "serve: refusing to start with QWEN35_GPU_FULL_LOGITS set — it forces "
            "the full %s-vocab logits D2H per token (K*992 KB/step over the host "
            "link) and cripples batched decode. Unset it (the serve path uses the "
            "sparse device top-k presentation).\n", "248k");
        return 1;
    }

    /* MemoryAware Phase 1: start the host-signal poll loop BEFORE any load so
     * model-load admission sees live numbers. Non-fatal if it can't start. */
    pn_memwatch_start();

    /* MemoryAware Phase 3 (§8): install crash/shutdown handlers early so any
     * exit — including a crash mid-load — closes the nvidia fd, the only step
     * that reclaims Kepler VRAM. SIGINT/SIGTERM are recorded here and answered
     * by serve_run_shutdown() below (clean CUDA teardown included);
     * SIGSEGV/SIGABRT/SIGBUS _exit() and lean on kernel fd-close + startup
     * stranded-VRAM detection.
     *
     * Arm the shutdown wake pipe FIRST: the handler tests for it to decide
     * whether it may defer, so installing the handlers before the pipe exists
     * leaves a window in which a SIGTERM takes the fallback path. */
    if (pn_serve_shutdown_arm() != 0)
        fprintf(stderr, "serve: could not create the shutdown wake pipe (%s); "
                        "a stop will fall back to the pre-drain path and will "
                        "NOT retire accelerator residencies\n", strerror(errno));
    pn_install_signal_handlers();

#ifdef USE_CUDA
    /* Die manager (Phase 7): probe the resident dies so the registry can
     * auto-place models and budget VRAM across them. Deliberately NO fc_init()
     * here — at this point no model has been placed, so creating the
     * process-default context would pin it to ordinal 0 whatever `--dies` says.
     * The probe is self-contained (bare context per die, destroyed) and the
     * default context is created later by the loader, on the die it owns.
     * Without a GPU (or when the probe fails) serving proceeds CPU-resident. */
    if (fc_driver_available()) {
        if (pn_dm_init_cuda(PN_DM_MAX_DIES) < 0)
            fprintf(stderr, "serve: die probe failed; models stay CPU-resident\n");
    }
#endif

#ifdef USE_TT
    /* Tenstorrent cards, ONE die per device (die_manager.h says why it is not
     * one die per core group). Only when no other family claimed the manager:
     * there is a single die-space and a die id means "K80 ordinal" or
     * "Blackhole device", never both, so a host carrying both families needs a
     * per-model die-space this module does not have. Probing second rather than
     * refusing keeps the common case — one family present — working, and makes
     * the mixed case a visible no-op instead of a silent overwrite. */
    if (pn_dm_count() == 0) {
        if (pn_dm_init_tt(PN_DM_MAX_DIES) < 0) {
            /* REFUSE, do not fall back. Somebody who set FT_TT_DRIVER named a
             * transport; a serve that then answers on the CPU is the silent
             * fallback this project has shipped twice, and the completion text
             * is BYTE-IDENTICAL either way (measured 2026-08-17, same md5 from
             * the card at 4.69 s and the CPU at 22.93 s) — so nothing
             * downstream can catch it. An unset variable is the benign case and
             * still warns and continues. */
            if (pn_dm_tt_driver_explicit()) {
                fprintf(stderr, "serve: REFUSED — FT_TT_DRIVER names a transport and no "
                                "Tenstorrent device could be opened over it. Not falling "
                                "back to the CPU: a CPU answer is indistinguishable from a "
                                "card answer in the text.\n");
                return 1;
            }
            fprintf(stderr, "serve: no Tenstorrent device; models stay CPU-resident\n");
        }
    } else {
        fprintf(stderr, "serve: die manager already holds another accelerator family; "
                        "Tenstorrent devices not registered\n");
    }
#endif

    /* The die count, said out loud, right after the probes that decide it.
     * `registry: '<name>' placed on dies 0,1 (0.0 GiB projected)` was the ONLY
     * trace the incident left, and 0.0 GiB (pn_dm_count()==0 ⇒ vram_bytes 0)
     * reads like a rounding artifact rather than "there is no GPU here". This
     * line cannot be misread. */
    fprintf(stderr, "serve: accelerator dies visible: %d\n", pn_dm_count());

    /* REFUSE, do not warn: the serve had already been running for an unknown
     * period and measurements had already been taken off it before anyone
     * noticed. A warning on a process that then serves for a week is the
     * failure, not the fix. A CPU serve is still fully available — it is asked
     * for by NOT passing --dies (see print_serve_usage). */
    if (refuse_dies_without_backend(specs, n_specs))
        return 1;

    static mr_registry reg;   /* lives for the process (accept loop is infinite) */
    if (mr_init(&reg) != 0) {
        fprintf(stderr, "serve: failed to init model registry\n");
        return 1;
    }

    pn_load_opts opts = { .vocab_path = vocab_path, .config_path = config_path,
                          .quantize = (bool)quantize };
    for (int i = 0; i < n_specs; i++) {
        char err[256];
        /* Per-model context cap (KV Phase 1): the ring is sized to this at load,
         * so set it on the shared opts before each model's load. */
        opts.ctx_cap = specs[i].ctx_cap;
        fprintf(stderr, "serve: loading %s ...\n", specs[i].path);
        if (mr_load(&reg, specs[i].path, specs[i].name, &opts,
                    specs[i].n_dies > 0 ? specs[i].dies : NULL,
                    specs[i].n_dies, specs[i].reasoning_budget,
                    specs[i].max_sessions, specs[i].max_concurrent,
                    err, sizeof(err)) != 0) {
            fprintf(stderr, "serve: failed to load %s: %s\n", specs[i].path, err);
            mr_free(&reg);
            return 1;
        }
    }

    /* PM 6: warm each model once (accelerator families re-armed before the first request). */
    for (int i = 0; i < MR_MAX_MODELS; i++)
        if (reg.entries[i].used && reg.entries[i].model && !reg.entries[i].sched)
            (void)pn_model_warm(reg.entries[i].model);

    /* Liveness-based session release (§C.4): start the watch manager (single
     * poll/grace background thread). If it fails to start, serve still runs —
     * /session/watch is refused (503) and the long backstop is the floor. */
    pn_session_watch_mgr *watch = pn_session_watch_start(&reg);
    if (!watch)
        fprintf(stderr, "serve: session watch manager failed to start; "
                "watches will be refused (long backstop remains the floor)\n");
    else
        /* Protect each model's mode-suffixed slots (`<base>-orch`/`-act`/`-chat`)
         * from the 12 h backstop while a watch on their base is alive. */
        pn_session_watch_install_backstop_hooks(watch);
    serve_ctx ctx = { .reg = &reg, .watch = watch };

    int listen_fd = hs_listen(host, port);
    if (listen_fd < 0) {
        pn_session_watch_stop(watch);
        mr_free(&reg);
        return 1;
    }
    {
        mr_info info[MR_MAX_MODELS];
        int n = mr_snapshot(&reg, info, MR_MAX_MODELS);
        fprintf(stderr, "serve: listening on http://%s:%d/v1  (%d model%s:",
                host, port, n, n == 1 ? "" : "s");
        for (int i = 0; i < n; i++)
            fprintf(stderr, "%s %s", i ? "," : "", info[i].name);
        fprintf(stderr, ")\n");
    }

    /* Dispatch layer (Phase G0): self-register each served model with the
     * router's registry and start heartbeating live load. FULLY gated on
     * PANTHEON_REGISTRY_URL — unset ⇒ this is a no-op and serve is unchanged. */
    serve_registry_start(&reg, host, port);

    /* The accept loop watches the listening socket AND the shutdown wake pipe.
     * The pipe is what makes the deferred teardown correct in a threaded serve:
     * the kernel may deliver SIGTERM to any thread that has it unblocked, so
     * "accept() returns EINTR" is not a guarantee — a connection thread running
     * the handler still wakes us here. */
    struct pollfd watched[2];
    int n_watched = 1;

    watched[0].fd = listen_fd;
    watched[0].events = POLLIN;
    watched[1].fd = pn_serve_shutdown_wake_fd();
    watched[1].events = POLLIN;
    if (watched[1].fd >= 0)
        n_watched = 2;

    while (!pn_serve_shutdown_requested()) {
        /* PM 5: a 1 s tick while idle keeps the accelerator's resident families armed
         * (mr_keepalive), so a request after a pause pays no idle re-arm. */
        int ready = poll(watched, (nfds_t)n_watched, 1000);
        if (ready < 0) {
            if (errno == EINTR) continue;
            fprintf(stderr, "serve: poll() failed: %s\n", strerror(errno));
            break;
        }
        if (ready == 0) {
            (void)mr_keepalive(ctx.reg);
            continue;
        }
        if (n_watched == 2 && (watched[1].revents & POLLIN))
            break;                      /* a shutdown signal was recorded */
        if (watched[0].revents & (POLLERR | POLLNVAL)) {
            /* The listening socket is unusable. Stop rather than spin: poll()
             * reports these every call, so `continue` here would be a busy
             * loop burning a core of a 2-core host. */
            fprintf(stderr, "serve: listening socket failed; stopping\n");
            break;
        }
        if (watched[0].revents & POLLIN) {
            int cfd = hs_accept(listen_fd);
            if (cfd < 0) continue;
            spawn_connection(cfd, &ctx);
        }
    }

    serve_run_shutdown(listen_fd, watch, &reg);

    /* Re-raise with the default disposition so the exit status reflects the
     * signal the operator sent and a parent (systemd) sees a normal
     * signal-terminated exit — same contract the handler used to provide, now
     * after a teardown that actually retired the card. */
    {
        int sig = pn_serve_shutdown_signal_number();

        if (sig > 0) {
            signal(sig, SIG_DFL);
            raise(sig);
        }
    }
    return 0;
}
