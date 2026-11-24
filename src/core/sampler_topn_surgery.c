/* SPDX-License-Identifier: AGPL-3.0-or-later */
/* Copyright (C) 2026 David (Corey) Herrera */
/**
 * @file sampler_topn_surgery.c
 * @brief Implementation of the top-N surgery decision (see sampler_topn_surgery.h).
 *
 * status: used by engine.c.
 * issues: none.
 * todo:   none.
 */
#include "sampler_topn_surgery.h"

#include <math.h>
#include <stdlib.h>
#include <string.h>

bool pn_topn_surgery_present(const pn_sampling_params *params)
{
    return params && ((params->logit_bias && params->n_logit_bias > 0) ||
                      params->presence_penalty != 0.0f || params->frequency_penalty != 0.0f);
}

static bool topn_is_survivor(const int *ids, int S, uint32_t token)
{
    for (int j = 0; j < S; j++)
        if ((uint32_t)ids[j] == token)
            return true;
    return false;
}

void pn_topn_surgery_apply(const int *ids, float *vals, int S, const pn_sampling_params *params,
                           const uint32_t *recent, int n_recent)
{
    for (int j = 0; j < S; j++) {
        int count = 0;

        for (int b = 0; params->logit_bias && b < params->n_logit_bias; b++)
            if (params->logit_bias[b].token_id == (uint32_t)ids[j])
                vals[j] += params->logit_bias[b].bias;
        for (int i = 0; recent && i < n_recent; i++)
            if (recent[i] == (uint32_t)ids[j])
                count++;
        if (count > 0)
            vals[j] -= params->frequency_penalty * (float)count + params->presence_penalty;
    }
}

/* RED `topn_surgery_blind` (PN_TT_RED): the outside limit ignores every bias, so a biased
 * token outside the N is never seen — a spec/plain arm carrying such a bias MUST then differ
 * from the dense reference. */
static bool topn_surgery_red_blind(void)
{
    static int red = -1;

    if (red < 0)
        red = getenv("PN_TT_RED") && strcmp(getenv("PN_TT_RED"), "topn_surgery_blind") == 0;
    return red != 0;
}

bool pn_topn_surgery_outside_limit(const int *ids, int S, float v_min,
                                   const pn_sampling_params *params, int n_recent, float *limit)
{
    float best_raise = 0.0f;

    if (topn_surgery_red_blind()) {
        *limit = v_min;
        return true;
    }
    if (n_recent > 0 && (params->presence_penalty < 0.0f || params->frequency_penalty < 0.0f))
        return false;
    for (int b = 0; params->logit_bias && b < params->n_logit_bias; b++) {
        const uint32_t token = params->logit_bias[b].token_id;
        float raise = 0.0f;

        if (topn_is_survivor(ids, S, token))
            continue;
        /* A token named twice is biased twice (pn_apply_logit_bias adds every entry). */
        for (int c = 0; c < params->n_logit_bias; c++)
            if (params->logit_bias[c].token_id == token)
                raise += params->logit_bias[c].bias;
        if (raise > best_raise)
            best_raise = raise;
    }
    *limit = v_min + best_raise;
    return true;
}

bool pn_topn_surgery_raising_bias(const pn_sampling_params *params)
{
    for (int b = 0; params && params->logit_bias && b < params->n_logit_bias; b++)
        if (params->logit_bias[b].bias > 0.0f)
            return true;
    return false;
}

bool pn_topn_surgery_clears(float value, float limit)
{
    return value - limit > 1e-4f * fmaxf(1.0f, fabsf(limit));
}

bool pn_topn_surgery_enabled(void)
{
    static int on = -1;

    if (on < 0)
        on = !(getenv("PN_TT_TOPN_SURGERY") && atoi(getenv("PN_TT_TOPN_SURGERY")) == 0);
    return on != 0;
}
