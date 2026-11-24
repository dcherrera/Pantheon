/* SPDX-License-Identifier: AGPL-3.0-or-later */
/* Copyright (C) 2026 David (Corey) Herrera */
/**
 * @file context_over_budget.c
 * @brief Implementation of the context_over_budget decision (see context_over_budget.h).
 *
 * @status None
 * @issues None
 * @todo None
 */

#include "context_over_budget.h"

bool sg_context_over_budget(pn_admit_status admit, int n_prompt, int fit_tokens,
                            int *kv_cap_out)
{
    if (admit != PN_ADMIT_CAPPED || fit_tokens <= 0 || fit_tokens >= n_prompt)
        return false;
    if (kv_cap_out)
        *kv_cap_out = fit_tokens;
    return true;
}
