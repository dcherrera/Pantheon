/* SPDX-License-Identifier: AGPL-3.0-or-later */
/* Copyright (C) 2026 David (Corey) Herrera */
/**
 * @file pn_tt_norm_role_plan.c
 * @brief The persistent norm role's host planning: knobs, step map, step cursor
 *        (pn_tt_norm_role_plan.h).
 *
 * @status P0.
 * @issues None
 * @todo None
 */

#include "pn_tt_norm_role_plan.h"

#include <stddef.h>
#include <string.h>

int pn_tt_norm_role_mode_from(const char *text)
{
    return text && strcmp(text, "1") == 0 ? 1 : 0;
}

int pn_tt_norm_role_exchange_from(const char *text)
{
    if (!text || !*text || strcmp(text, "whole") == 0)
        return 0;
    if (strcmp(text, "faces,stagger") == 0)
        return (int)(PN_TT_NORM_ROLE_EXCHANGE_FACES | PN_TT_NORM_ROLE_EXCHANGE_STAGGER);
    if (strcmp(text, "faces") == 0)
        return (int)PN_TT_NORM_ROLE_EXCHANGE_FACES;
    if (strcmp(text, "stagger") == 0)
        return (int)PN_TT_NORM_ROLE_EXCHANGE_STAGGER;
    return -1;
}

uint32_t pn_tt_norm_role_steps(uint32_t layers)
{
    return 2u * layers + 1u;
}

bool pn_tt_norm_role_step_norm(uint32_t step, uint32_t layers, pn_tt_norm_role_norm *out_norm,
                               uint32_t *out_layer)
{
    if (!out_norm || !out_layer || layers == 0u || step >= pn_tt_norm_role_steps(layers))
        return false;
    if (step % 2u == 1u) {
        *out_norm = PN_TT_NORM_ROLE_POST;
        *out_layer = (step - 1u) / 2u;
    } else if (step / 2u < layers) {
        *out_norm = PN_TT_NORM_ROLE_ATTN;
        *out_layer = step / 2u;
    } else {
        *out_norm = PN_TT_NORM_ROLE_OUTPUT;
        *out_layer = layers;
    }
    return true;
}

void pn_tt_norm_role_cursor_begin(pn_tt_norm_role_cursor *cursor, uint32_t steps)
{
    if (!cursor)
        return;
    cursor->steps = steps;
    cursor->next = 0u;
    cursor->residual_pending = false;
}

int pn_tt_norm_role_cursor_residual(pn_tt_norm_role_cursor *cursor)
{
    if (!cursor || cursor->residual_pending || cursor->next == 0u ||
        cursor->next >= cursor->steps)
        return -1;
    cursor->residual_pending = true;
    return 0;
}

int pn_tt_norm_role_cursor_norm(pn_tt_norm_role_cursor *cursor, uint32_t *out_step)
{
    if (!cursor || !out_step || cursor->next >= cursor->steps ||
        (cursor->next == 0u) == cursor->residual_pending)
        return -1;
    *out_step = cursor->next++;
    cursor->residual_pending = false;
    return 0;
}

int pn_tt_norm_role_cursor_end(const pn_tt_norm_role_cursor *cursor)
{
    return cursor && cursor->next == cursor->steps && !cursor->residual_pending ? 0 : -1;
}
