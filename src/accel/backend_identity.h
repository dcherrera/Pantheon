/* SPDX-License-Identifier: AGPL-3.0-or-later */
/* Copyright (C) 2026 David (Corey) Herrera */
/**
 * @file backend_identity.h
 * @brief What accelerator backend THIS binary was compiled with, as a value the
 *        process can print. Compile-time constants only — no probing, no
 *        driver, no allocation, safe to call from anywhere at any time.
 *
 *        Why this exists: `make`, `make USE_CUDA=1` and `make USE_TT=1` are
 *        three different programs, and until the per-backend output names
 *        landed they all wrote the same file. A CPU-only binary sitting at the
 *        deployed path served a 35B MoE on a 2-core host with all 8 K80 dies
 *        idle for an unknown period, and answering "what am I running?" took
 *        `ldd` and `strings` on a remote box. The backend is a compile-time
 *        constant; the process should simply say so — in its startup log, on
 *        /v1/models, on the health route, and through `pantheon --backend` so a
 *        deploy script can assert on it before launching anything.
 *
 *        pn_backend_name() is the human string ("CUDA"), pn_backend_token() the
 *        machine one ("cuda") — a deploy script compares the token, a log line
 *        prints the name. They are the same enumeration, never derived from
 *        each other at runtime, so neither can drift into a lie.
 *
 * @status Complete. CUDA/TT/METAL arms are compile-time only, so they are
 *         proved by the preprocessor, not by a hardware run.
 * @issues None
 * @todo None
 */

#ifndef PN_BACKEND_IDENTITY_H
#define PN_BACKEND_IDENTITY_H

#include <stdbool.h>

#ifdef __cplusplus
extern "C" {
#endif

/**
 * Human-readable name of the compiled accelerator backend: "CUDA", "TT",
 * "METAL", "CUDA+TT", ... or "CPU" when none is compiled in.
 *
 * @return A string literal — never NULL, never freed.
 *
 * Thread-safe: yes (returns a literal)
 * Reentrant: yes
 * Signal-safe: yes
 */
const char *pn_backend_name(void);

/**
 * Machine-readable form of the same value, lowercase and shell-comparable:
 * "cuda", "tt", "metal", "cuda+tt", "cpu". This is what `pantheon --backend`
 * prints and what a deploy script asserts against.
 *
 * @return A string literal — never NULL, never freed.
 *
 * Thread-safe: yes (returns a literal)
 * Reentrant: yes
 * Signal-safe: yes
 */
const char *pn_backend_token(void);

/**
 * The commit this binary was built from (short hash, with a "-dirty" suffix
 * when the tree had uncommitted changes), or "unknown" when the build had no
 * git available. Stamped in by the Makefile as -DPN_BUILD_COMMIT.
 *
 * @return A string literal — never NULL, never freed.
 *
 * Thread-safe: yes (returns a literal)
 * Reentrant: yes
 * Signal-safe: yes
 */
const char *pn_backend_build_commit(void);

/**
 * Whether ANY GPU/accelerator backend is compiled in. False means every forward
 * pass in this process runs on the host CPU, whatever the flags say.
 *
 * Thread-safe: yes  Reentrant: yes  Signal-safe: yes
 */
bool pn_backend_has_accelerator(void);

/**
 * Whether this binary carries a backend the DIE MANAGER can probe — i.e. one
 * for which `--dies` means anything. CUDA and Tenstorrent have pn_dm_init_*
 * paths; Metal does not (it is one implicit device, never placed by index), so
 * a Metal build answers false here and true for pn_backend_has_accelerator().
 *
 * This is the predicate the serve's refuse-to-start guard asks, because the
 * question there is not "is there a GPU" but "can a die index be honoured".
 *
 * Thread-safe: yes  Reentrant: yes  Signal-safe: yes
 */
bool pn_backend_supports_die_placement(void);

#ifdef __cplusplus
}
#endif

#endif /* PN_BACKEND_IDENTITY_H */
