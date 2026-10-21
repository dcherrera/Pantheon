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
 *         load/unload via /admin/models/*. Single --model behaves exactly as
 *         the original single-model server.
 * @issues Concurrent chats to ONE model are 429'd, not batched/queued.
 * @todo None
 */

#ifndef PN_SERVE_COMMAND_H
#define PN_SERVE_COMMAND_H

/**
 * Run the serve subcommand. `argc`/`argv` are the arguments AFTER the "serve"
 * token (argv[0] is "serve"). Blocks serving requests until interrupted.
 *
 * @return process exit code (0 on clean shutdown, non-zero on setup error).
 */
int pn_serve_main(int argc, char **argv);

#endif /* PN_SERVE_COMMAND_H */
