/* SPDX-License-Identifier: AGPL-3.0-or-later */
/* Copyright (C) 2026 David (Corey) Herrera */
/**
 * @file router_command.h
 * @brief `pantheon router` entrypoint (Phase G1+G2): run the dispatch layer —
 *        an in-memory model-instance registry (self-registration + heartbeat +
 *        TTL) plus a least-busy load-balancing router that reverse-proxies
 *        /v1/chat/completions (+ /v1/completions) to the freest live instance.
 *        Local position: no auth. Reuses Pantheon's serve HTTP server.
 *
 * @status G1+G2. CUDA-free, host-side only.
 * @issues None
 * @todo None
 */

#ifndef PN_SERVE_ROUTER_COMMAND_H
#define PN_SERVE_ROUTER_COMMAND_H

/**
 * Run the router subcommand. `argc`/`argv` are the arguments AFTER the "router"
 * token (argv[0] == "router"). Blocks serving requests until interrupted.
 * @return process exit code (0 on clean shutdown, non-zero on setup error).
 */
int pn_router_main(int argc, char **argv);

#endif /* PN_SERVE_ROUTER_COMMAND_H */
