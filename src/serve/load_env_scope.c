/* SPDX-License-Identifier: AGPL-3.0-or-later */
/* Copyright (C) 2026 David (Corey) Herrera */
/**
 * @file   load_env_scope.c
 * @brief  Implementation of the load-scoped environment (see load_env_scope.h).
 *
 * @status None
 * @issues None
 * @todo   None
 */

#include "load_env_scope.h"

#include <stdlib.h>
#include <string.h>

void pn_load_env_apply(pn_load_env_var *vars, size_t n)
{
    for (size_t i = 0; i < n; i++) {
        vars[i].saved = NULL;
        if (!vars[i].value)
            continue;
        const char *prev = getenv(vars[i].name);
        vars[i].saved = prev ? strdup(prev) : NULL;
        setenv(vars[i].name, vars[i].value, 1);
    }
}

void pn_load_env_restore(pn_load_env_var *vars, size_t n)
{
    for (size_t i = 0; i < n; i++) {
        if (!vars[i].value)
            continue;
        if (vars[i].saved)
            setenv(vars[i].name, vars[i].saved, 1);
        else
            unsetenv(vars[i].name);
        free(vars[i].saved);
        vars[i].saved = NULL;
    }
}
