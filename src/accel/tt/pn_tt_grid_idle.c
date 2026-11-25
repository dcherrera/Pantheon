/* SPDX-License-Identifier: AGPL-3.0-or-later */
/* Copyright (C) 2026 David (Corey) Herrera */
/**
 * @file   pn_tt_grid_idle.c
 * @brief  Grid idle guard — see pn_tt_grid_idle.h.
 *
 * @status None
 * @issues None
 * @todo   None
 */
#include "pn_tt_grid_idle.h"

pn_tt_grid_idle_action pn_tt_grid_idle_decide(double idle_s, double min_nop_s, double limit_s,
                                              bool nop_ok)
{
    if (limit_s <= 0.0 || idle_s < 0.0)
        return PN_TT_GRID_IDLE_NONE;
#ifndef PN_TT_GRID_IDLE_RED
    if (idle_s >= (nop_ok ? limit_s - PN_TT_GRID_IDLE_MARGIN_S : limit_s))
        return PN_TT_GRID_IDLE_REARM;
#endif
    if (nop_ok && idle_s >= min_nop_s && idle_s < limit_s - PN_TT_GRID_IDLE_MARGIN_S)
        return PN_TT_GRID_IDLE_NOP;
    return PN_TT_GRID_IDLE_NONE;
}
