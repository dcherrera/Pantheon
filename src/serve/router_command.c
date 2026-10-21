/* SPDX-License-Identifier: AGPL-3.0-or-later */
/* Copyright (C) 2026 David (Corey) Herrera */
/**
 * @file router_command.c
 * @brief Implementation of `pantheon router` (see router_command.h). Same
 *        accept-loop shape as serve_command.c — one detached pthread per
 *        connection — plus a background TTL-sweep thread that evicts instances
 *        whose heartbeat lapsed. No CUDA, no model load: this is the pure
 *        host-side dispatch layer.
 *
 * Routes:
 *   POST   /registry/register     — a serve announces a model instance
 *   POST   /registry/heartbeat    — refresh liveness + load
 *   DELETE /registry/deregister   — graceful removal
 *   GET    /registry/fleet        — model-level fleet view (TUI/ops)
 *   GET    /v1/models             — deduped-by-model_id live model list
 *   POST   /v1/chat/completions   — least-busy dispatch + SSE pass-through
 *   POST   /v1/completions        — same
 *
 * @status G1+G2.
 * @issues None
 * @todo None
 */

#include "router_command.h"

#include <pthread.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <unistd.h>

#include "http_server.h"
#include "registry.h"
#include "router.h"
#include "session_affinity.h"

#define ROUTER_DEFAULT_HOST "127.0.0.1"
#define ROUTER_DEFAULT_PORT 8100
#define ROUTER_SWEEP_INTERVAL_SEC 2
/* Evicted-instance ids reported per sweep so their session bindings are dropped.
 * A single sweep rarely evicts more than a handful; extra evictions still remove
 * from the registry, they just aren't reported this pass (a later sweep is a
 * no-op for them, and their sessions simply re-route least-busy meanwhile). */
#define ROUTER_SWEEP_REPORT_MAX 32

typedef struct {
    reg_registry     *reg;
    session_affinity *sa;
} router_ctx;

/* Args for the background TTL sweep thread (registry + affinity map). */
typedef struct {
    reg_registry     *reg;
    session_affinity *sa;
} sweep_ctx;

typedef struct {
    int         client_fd;
    router_ctx *ctx;
} connection_job;

static volatile int g_running = 1;

static void print_router_usage(void)
{
    printf("Usage: pantheon router [--host <addr>] [--port <n>]\n\n");
    printf("Runs the dispatch layer: model-instance registry + least-busy router.\n");
    printf("Local position (no auth). Serve instances self-register against it.\n\n");
    printf("Options:\n");
    printf("  --host <addr>   Bind address (default %s)\n", ROUTER_DEFAULT_HOST);
    printf("  --port <n>      Port (default %d)\n", ROUTER_DEFAULT_PORT);
    printf("\nEndpoints:\n");
    printf("  POST   /registry/register     POST /registry/heartbeat\n");
    printf("  DELETE /registry/deregister   GET  /registry/fleet\n");
    printf("  GET    /v1/models             POST /v1/chat/completions\n");
    printf("  POST   /v1/completions\n");
}

/* Send a small {"status":"ok"} or an error based on a reg_status. */
static void reply_reg_status(int cfd, reg_status st)
{
    switch (st) {
        case REG_OK:
            hs_send_json(cfd, 200, "{\"status\":\"ok\"}");
            break;
        case REG_BAD_REQUEST:
            hs_send_error(cfd, 400, "invalid_request",
                          "malformed body or missing required field");
            break;
        case REG_FULL:
            hs_send_error(cfd, 503, "registry_full",
                          "instance table is full");
            break;
        case REG_NOT_FOUND:
            hs_send_error(cfd, 404, "not_found", "unknown instance_id");
            break;
    }
}

static void dispatch(int cfd, router_ctx *ctx, const hs_request *req)
{
    if (strcmp(req->method, "OPTIONS") == 0) {
        hs_send_json(cfd, 200, "{}");
        return;
    }

    /* ---- registry control plane ---- */
    if (strcmp(req->method, "POST") == 0 &&
        strcmp(req->path, "/registry/register") == 0) {
        reply_reg_status(cfd, reg_register_json(ctx->reg, req->body));
        return;
    }
    if (strcmp(req->method, "POST") == 0 &&
        strcmp(req->path, "/registry/heartbeat") == 0) {
        reply_reg_status(cfd, reg_heartbeat_json(ctx->reg, req->body));
        return;
    }
    if (strcmp(req->method, "DELETE") == 0 &&
        strcmp(req->path, "/registry/deregister") == 0) {
        char removed[RW_INSTANCE_ID_MAX];
        reg_status st = reg_deregister_json_collect(ctx->reg, req->body,
                                                    removed, sizeof(removed));
        if (st == REG_OK)
            sa_forget_instance(ctx->sa, removed);   /* drop its session bindings */
        reply_reg_status(cfd, st);
        return;
    }
    if (strcmp(req->method, "GET") == 0 &&
        strcmp(req->path, "/registry/fleet") == 0) {
        char *json = reg_fleet_json(ctx->reg);
        if (!json) { hs_send_error(cfd, 500, "server_error", "oom"); return; }
        hs_send_json(cfd, 200, json);
        free(json);
        return;
    }

    /* ---- client plane ---- */
    if (strcmp(req->method, "GET") == 0 && strcmp(req->path, "/v1/models") == 0) {
        char *json = reg_models_json(ctx->reg);
        if (!json) { hs_send_error(cfd, 500, "server_error", "oom"); return; }
        hs_send_json(cfd, 200, json);
        free(json);
        return;
    }
    if (strcmp(req->method, "POST") == 0 &&
        (strcmp(req->path, "/v1/chat/completions") == 0 ||
         strcmp(req->path, "/v1/completions") == 0)) {
        router_handle_completion(cfd, ctx->reg, ctx->sa, req->path, req->body,
                                 req->session_id);
        return;
    }
    if (strcmp(req->method, "GET") == 0 && strcmp(req->path, "/") == 0) {
        hs_send_json(cfd, 200,
                     "{\"service\":\"pantheon-router\",\"status\":\"ok\"}");
        return;
    }
    hs_send_error(cfd, 404, "not_found", "unknown route");
}

static void handle_connection(int cfd, router_ctx *ctx)
{
    hs_request req;
    if (hs_read_request(cfd, &req) == 0) {
        fprintf(stderr, "router: %s %s\n", req.method, req.path);
        dispatch(cfd, ctx, &req);
        hs_request_free(&req);
    }
    hs_close(cfd);
}

static void *connection_thread(void *arg)
{
    connection_job *job = arg;
    handle_connection(job->client_fd, job->ctx);
    free(job);
    return NULL;
}

static void spawn_connection(int cfd, router_ctx *ctx)
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
        handle_connection(cfd, ctx);
        free(job);
    }
}

/* Background TTL sweep: evict instances that stopped heartbeating, and drop the
 * session-affinity bindings that pointed at them (so those sessions re-route
 * least-busy on their next turn instead of pinning to a dead instance). */
static void *sweep_thread(void *arg)
{
    sweep_ctx *sc = arg;
    while (g_running) {
        sleep(ROUTER_SWEEP_INTERVAL_SEC);
        char evicted[ROUTER_SWEEP_REPORT_MAX][RW_INSTANCE_ID_MAX];
        int reported = 0;
        int n = reg_sweep_collect(sc->reg, time(NULL), evicted,
                                  ROUTER_SWEEP_REPORT_MAX, &reported);
        for (int i = 0; i < reported; i++)
            sa_forget_instance(sc->sa, evicted[i]);
        if (n > 0)
            fprintf(stderr, "router: TTL sweep evicted %d stale instance%s\n",
                    n, n == 1 ? "" : "s");
    }
    return NULL;
}

int pn_router_main(int argc, char **argv)
{
    const char *host = ROUTER_DEFAULT_HOST;
    int port = ROUTER_DEFAULT_PORT;

    for (int i = 1; i < argc; i++) {
        if (strcmp(argv[i], "--help") == 0 || strcmp(argv[i], "-h") == 0) {
            print_router_usage();
            return 0;
        } else if (strcmp(argv[i], "--host") == 0 && i + 1 < argc) {
            host = argv[++i];
        } else if (strcmp(argv[i], "--port") == 0 && i + 1 < argc) {
            port = atoi(argv[++i]);
        } else {
            fprintf(stderr, "router: unknown option '%s'\n", argv[i]);
            print_router_usage();
            return 1;
        }
    }

    static reg_registry reg;   /* process lifetime (accept loop is infinite) */
    if (reg_init(&reg) != 0) {
        fprintf(stderr, "router: failed to init registry\n");
        return 1;
    }

    static session_affinity sa;   /* process lifetime; session -> instance map */
    if (sa_init(&sa) != 0) {
        fprintf(stderr, "router: failed to init session-affinity map\n");
        reg_destroy(&reg);
        return 1;
    }

    static sweep_ctx sweeper_ctx;
    sweeper_ctx.reg = &reg;
    sweeper_ctx.sa = &sa;
    pthread_t sweeper;
    if (pthread_create(&sweeper, NULL, sweep_thread, &sweeper_ctx) != 0) {
        fprintf(stderr, "router: failed to start TTL sweep thread\n");
        sa_destroy(&sa);
        reg_destroy(&reg);
        return 1;
    }

    int listen_fd = hs_listen(host, port);
    if (listen_fd < 0) {
        sa_destroy(&sa);
        reg_destroy(&reg);
        return 1;
    }
    fprintf(stderr, "router: dispatch layer listening on http://%s:%d "
                    "(TTL %ds, sweep %ds, session affinity on)\n",
            host, port, RW_TTL_SECONDS, ROUTER_SWEEP_INTERVAL_SEC);

    router_ctx ctx = { .reg = &reg, .sa = &sa };
    for (;;) {
        int cfd = hs_accept(listen_fd);
        if (cfd < 0) continue;
        spawn_connection(cfd, &ctx);
    }

    /* Unreached (infinite accept loop). */
    hs_close(listen_fd);
    sa_destroy(&sa);
    reg_destroy(&reg);
    return 0;
}
