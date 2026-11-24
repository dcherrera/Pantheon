/* SPDX-License-Identifier: AGPL-3.0-or-later */
/* Copyright (C) 2026 David (Corey) Herrera */
/**
 * @file context_over_budget.h
 * @brief The `context_over_budget` decision for one prompt: whether pn_admit's
 *        verdict refuses it, and the per-session cap the 422 must quote. The cap
 *        is the one that refused the prompt (pn_admit's fit window: the context
 *        cap, or what host / die memory holds now), never the advertised window,
 *        which can sit far above it on a CPU-resident model (p150, 2026-10-27:
 *        the 422 said 21139 when 1581 refused the prompt).
 *
 * @status None
 * @issues None
 * @todo None
 */

#ifndef PN_SERVE_CONTEXT_OVER_BUDGET_H
#define PN_SERVE_CONTEXT_OVER_BUDGET_H

#include <stdbool.h>

#include "../../include/pantheon.h"

#ifdef __cplusplus
extern "C" {
#endif

/**
 * @param admit       pn_admit's verdict for the prompt
 * @param n_prompt    prompt tokens
 * @param fit_tokens  pn_admit's *fit_tokens_out
 * @param kv_cap_out  on true: the cap that refused the prompt
 * @return true when the prompt is refused as over the per-session budget
 *
 * Thread-safe: yes. Reentrant: yes. Signal-safe: yes
 */
bool sg_context_over_budget(pn_admit_status admit, int n_prompt, int fit_tokens,
                            int *kv_cap_out);

#ifdef __cplusplus
}
#endif

#endif /* PN_SERVE_CONTEXT_OVER_BUDGET_H */
