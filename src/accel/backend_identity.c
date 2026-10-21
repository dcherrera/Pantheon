/* SPDX-License-Identifier: AGPL-3.0-or-later */
/* Copyright (C) 2026 David (Corey) Herrera */
/**
 * @file backend_identity.c
 * @brief Implementation of the compile-time backend identity (see
 *        backend_identity.h).
 *
 * Everything here is decided by the preprocessor and returned as a string
 * literal. That is deliberate: the whole value of this module is that it CANNOT
 * disagree with the code that was compiled. A runtime probe ("is there a CUDA
 * driver?") answers a different question and is exactly the one that was
 * already misleading — the incident binary had no CUDA compiled in at all, so
 * no probe would have run and no probe would have complained.
 *
 * The combination arms (CUDA+TT etc.) exist because the Makefile permits
 * `make USE_CUDA=1 USE_TT=1`. They are enumerated rather than assembled at
 * runtime so every value is a literal and the accessors stay signal-safe.
 *
 * @status Complete.
 * @issues None
 * @todo None
 */

#include "backend_identity.h"

/* The Makefile stamps the build commit in. A build with no git (a tarball, a
 * copied tree) still compiles — it just reports "unknown" rather than lying. */
#ifndef PN_BUILD_COMMIT
#define PN_BUILD_COMMIT "unknown"
#endif

/* One enumeration, two spellings. Written out per combination instead of
 * concatenated at runtime so both accessors return literals. */
#if defined(USE_CUDA)
#  if defined(USE_TT)
#    define PN_BACKEND_NAME_STR  "CUDA+TT"
#    define PN_BACKEND_TOKEN_STR "cuda+tt"
#  elif defined(USE_METAL)
#    define PN_BACKEND_NAME_STR  "CUDA+METAL"
#    define PN_BACKEND_TOKEN_STR "cuda+metal"
#  else
#    define PN_BACKEND_NAME_STR  "CUDA"
#    define PN_BACKEND_TOKEN_STR "cuda"
#  endif
#elif defined(USE_TT)
#  if defined(USE_METAL)
#    define PN_BACKEND_NAME_STR  "TT+METAL"
#    define PN_BACKEND_TOKEN_STR "tt+metal"
#  else
#    define PN_BACKEND_NAME_STR  "TT"
#    define PN_BACKEND_TOKEN_STR "tt"
#  endif
#elif defined(USE_METAL)
#  define PN_BACKEND_NAME_STR  "METAL"
#  define PN_BACKEND_TOKEN_STR "metal"
#else
#  define PN_BACKEND_NAME_STR  "CPU"
#  define PN_BACKEND_TOKEN_STR "cpu"
#endif

const char *pn_backend_name(void)
{
    return PN_BACKEND_NAME_STR;
}

const char *pn_backend_token(void)
{
    return PN_BACKEND_TOKEN_STR;
}

const char *pn_backend_build_commit(void)
{
    return PN_BUILD_COMMIT;
}

bool pn_backend_has_accelerator(void)
{
#if defined(USE_CUDA) || defined(USE_TT) || defined(USE_METAL)
    return true;
#else
    return false;
#endif
}

bool pn_backend_supports_die_placement(void)
{
    /* Metal is deliberately excluded: die_manager.h has pn_dm_init_cuda and
     * pn_dm_init_tt and nothing else, so on a Metal build `--dies 0` names an
     * index no init path can ever register. */
#if defined(USE_CUDA) || defined(USE_TT)
    return true;
#else
    return false;
#endif
}
