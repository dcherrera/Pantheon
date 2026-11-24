/* SPDX-License-Identifier: AGPL-3.0-or-later */
/* Copyright (C) 2026 David (Corey) Herrera */
/**
 * @file serve_command.h
 * @brief `pantheon serve` entrypoint: load one or more models into the model
 *        registry and run an OpenAI-compatible HTTP server. Invoked from the
 *        CLI dispatch (cli/main.c).
 *
 * @status Multi-model (Phase 7): repeatable --model (+ --name / --dies),
 *         registry-routed /v1/chat/completions, live /v1/models, runtime
 *         load/unload via /admin/models/{load,unload}. Single --model behaves exactly as
 *         the original single-model server.
 * @issues Concurrent chats to ONE model are 429'd, not batched/queued.
 * @todo None
 */

#ifndef PN_SERVE_COMMAND_H
#define PN_SERVE_COMMAND_H

#include "http_server.h"
#include "model_registry.h"

/* Hooks the appliance mode (`pantheon local`, in src/local) plugs into the
 * serve. Every member may be NULL. */
typedef struct {
    /* After the socket is listening: start discovery, print the pairing code,
     * reload the remembered models. Non-zero stops the serve. */
    int  (*on_listening)(mr_registry *reg, const char *host, int port, void *user);
    /* Before the built-in routes: return 1 if the route was handled (response
     * sent), 2 if it was handled and the hook ADOPTED the fd (it answers and
     * closes it later; the serve must not close it), 0 to fall through to the
     * serve's own routes. */
    int  (*route)(int client_fd, mr_registry *reg, const hs_request *req, void *user);
    /* At shutdown, before the models are freed: stop the appliance threads. */
    void (*on_shutdown)(void *user);
    void *user;
} pn_serve_local_hooks;

/**
 * Install the appliance hooks (copied). NULL clears them. Call before
 * pn_serve_main.
 *
 * Thread-safe: no (set before serving)
 * Reentrant: no
 * Signal-safe: no
 */
void pn_serve_set_local_hooks(const pn_serve_local_hooks *hooks);

/**
 * Run the serve subcommand. `argc`/`argv` are the arguments AFTER the "serve"
 * token (argv[0] is "serve"). Blocks serving requests until interrupted.
 *
 * @return process exit code (0 on clean shutdown, non-zero on setup error).
 */
int pn_serve_main(int argc, char **argv);

#endif /* PN_SERVE_COMMAND_H */
