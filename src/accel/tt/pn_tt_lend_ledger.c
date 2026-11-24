/* SPDX-License-Identifier: AGPL-3.0-or-later */
/* Copyright (C) 2026 David (Corey) Herrera */
/**
 * @file pn_tt_lend_ledger.c
 * @brief The L1a lend ledger (pn_tt_lend_ledger.h).
 *
 * @status L1a.5, default on for dense-FFN qwen35.
 * @issues None
 * @todo None
 */
#include "pn_tt_lend_ledger.h"

#include <stdlib.h>
#include <string.h>

bool pn_tt_lend_wanted(uint32_t rows, uint32_t min_rows, bool deep)
{
    return deep && min_rows > 0u && rows >= min_rows;
}

uint32_t pn_tt_lend_min_rows(const char *env, bool dense)
{
    const uint32_t fallback = dense ? PN_TT_LEND_DENSE_MIN_ROWS : 0u;
    char *end = NULL;
    unsigned long parsed;

    if (!env || !*env)
        return fallback;
    parsed = strtoul(env, &end, 10);
    return end && *end == '\0' ? (uint32_t)parsed : fallback;
}

bool pn_tt_lend_deep_on(const char *env, bool lent)
{
#ifdef PN_TT_LEND_DEFAULT_RED
    (void)lent;
    return env && *env && atoi(env) != 0;
#else
    return env && *env ? atoi(env) != 0 : lent;
#endif
}

void pn_tt_lend_ledger_reset(pn_tt_lend_ledger *ledger)
{
    memset(ledger, 0, sizeof(*ledger));
}

bool pn_tt_lend_ledger_add(pn_tt_lend_ledger *ledger, uint32_t member, uint32_t core)
{
    uint32_t index;

    if (ledger->count >= PN_TT_LEND_MAX)
        return false;
    for (index = 0; index < ledger->count; index++)
        if (ledger->core[index] == core)
            return false;
    ledger->member[ledger->count] = member;
    ledger->core[ledger->count] = core;
    ledger->stuck[ledger->count] = false;
    ledger->count++;
    return true;
}

bool pn_tt_lend_ledger_pop(pn_tt_lend_ledger *ledger, uint32_t *member, uint32_t *core)
{
    if (ledger->count == 0u)
        return false;
    ledger->count--;
    *member = ledger->member[ledger->count];
    *core = ledger->core[ledger->count];
    return true;
}

void pn_tt_lend_ledger_mark_stuck(pn_tt_lend_ledger *ledger, uint32_t core)
{
    uint32_t index;

    for (index = 0; index < ledger->count; index++)
        if (ledger->core[index] == core)
            ledger->stuck[index] = true;
}

uint32_t pn_tt_lend_ledger_take_restorable(pn_tt_lend_ledger *ledger, uint32_t *members,
                                           uint32_t *cores)
{
    uint32_t index, written = 0u;

    for (index = 0; index < ledger->count; index++) {
#ifndef PN_TT_LEND_LEDGER_RED
        if (ledger->stuck[index])
            continue;
#endif
        members[written] = ledger->member[index];
        cores[written] = ledger->core[index];
        written++;
    }
    pn_tt_lend_ledger_reset(ledger);
    return written;
}
