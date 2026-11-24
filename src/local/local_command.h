/* SPDX-License-Identifier: AGPL-3.0-or-later */
/* Copyright (C) 2026 David (Corey) Herrera */
/**
 * @file   local_command.h
 * @brief  `pantheon local` — the appliance mode of the serve. Parses the appliance
 *         flags, installs the appliance hooks (auth, /local routes, discovery,
 *         reload) and runs `pantheon serve --local` on 0.0.0.0. Also
 *         `pantheon local pair-code`, which prints the current pairing code from
 *         the state dir (on the box, or over SSH).
 *
 * @status None
 * @issues None
 * @todo   None
 */

#ifndef PN_LOCAL_COMMAND_H
#define PN_LOCAL_COMMAND_H

#include "local_context.h"

#ifdef __cplusplus
extern "C" {
#endif

/**
 * Fill `c` with the appliance defaults (port 8400, host 0.0.0.0, state and
 * store under ~/.pantheon, the built-in catalog URL) and apply the flags in
 * argv[first..argc).
 * @return 0, 1 for --help, -1 on a bad flag (reported on stderr)
 *
 * Thread-safe: no (getenv). Reentrant: yes. Signal-safe: no
 */
int pn_local_parse_args(pn_local_ctx *c, int argc, char **argv, int first);

/**
 * Run `pantheon local`. argv[0] is "local". Blocks while serving.
 * @return process exit code
 *
 * Thread-safe: no (process entry point)
 * Reentrant:   no
 * Signal-safe: no
 */
int pn_local_main(int argc, char **argv);

#ifdef __cplusplus
}
#endif

#endif /* PN_LOCAL_COMMAND_H */
