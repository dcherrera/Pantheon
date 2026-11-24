/* SPDX-License-Identifier: AGPL-3.0-or-later */
/* Copyright (C) 2026 David (Corey) Herrera */
/**
 * @file   local_caller.c
 * @brief  Implementation of the per-request caller slot (see local_caller.h).
 *
 * @status None
 * @issues None
 * @todo   None
 */

#include "local_caller.h"

#include <string.h>

static _Thread_local pn_local_caller t_caller;
static _Thread_local int t_caller_set;

void pn_local_caller_set(const pn_local_caller *caller)
{
    if (!caller) {
        pn_local_caller_clear();
        return;
    }
    t_caller = *caller;
    t_caller_set = 1;
}

const pn_local_caller *pn_local_caller_get(void)
{
    return t_caller_set ? &t_caller : NULL;
}

void pn_local_caller_clear(void)
{
    memset(&t_caller, 0, sizeof(t_caller));
    t_caller_set = 0;
}
