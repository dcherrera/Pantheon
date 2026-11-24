/* SPDX-License-Identifier: AGPL-3.0-or-later */
/* Copyright (C) 2026 David (Corey) Herrera */
/**
 * @file   load_env_scope.h
 * @brief  Environment variables set for the duration of one model load and put back after
 *         (QWEN35_MTP, PN_TT_KV_TYPE, PN_TT_KV_LONG, PN_TT_SPEC: knobs the loaders read
 *         from the environment). The caller serializes loads (model_admin.c holds a lock around
 *         apply ... restore), since the environment is process-wide.
 *
 * @status None
 * @issues None
 * @todo   None
 */

#ifndef PN_SERVE_LOAD_ENV_SCOPE_H
#define PN_SERVE_LOAD_ENV_SCOPE_H

#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif

/** One variable a load sets: `value` NULL = the load leaves it alone. `saved` is filled by
 *  pn_load_env_apply (the previous value, NULL = it was unset) and freed by restore. */
typedef struct {
    const char *name;
    const char *value;
    char       *saved;
} pn_load_env_var;

/**
 * Set every variable with a value, remembering what each replaced.
 *
 * Thread-safe: no (the environment; callers serialize). Reentrant: no. Signal-safe: no
 */
void pn_load_env_apply(pn_load_env_var *vars, size_t n);

/**
 * Put back what pn_load_env_apply replaced: the previous value, or unset when there was none.
 *
 * Thread-safe: no (the environment; callers serialize). Reentrant: no. Signal-safe: no
 */
void pn_load_env_restore(pn_load_env_var *vars, size_t n);

#ifdef __cplusplus
}
#endif

#endif /* PN_SERVE_LOAD_ENV_SCOPE_H */
