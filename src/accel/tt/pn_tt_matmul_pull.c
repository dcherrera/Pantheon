/* SPDX-License-Identifier: AGPL-3.0-or-later */
/* Copyright (C) 2026 David (Corey) Herrera */
/**
 * @file pn_tt_matmul_pull.c
 * @brief The M2 pull's placement in a matmul chain (pn_tt_matmul_pull.h).
 *
 * @status None
 * @issues None
 * @todo None
 */
#include "pn_tt_matmul_pull.h"

void pn_tt_pull_chain_note(pn_tt_pull_chain *chain, bool split, uint32_t members,
                           uint32_t sequence)
{
    if (!chain || !split || members < 2u)
        return;
    chain->members = members;
    chain->target = sequence;
}

pn_tt_pull_words pn_tt_pull_words_for(const pn_tt_pull_chain *chain, bool last, bool red_skip,
                                      bool serial)
{
    pn_tt_pull_words words = { 0u, 0u };

    /* A solo chain has nothing to pull, and a descriptor with more behind it in the chain
     * leaves the waiting to the one the flush awaits. */
    if (!chain || !last || chain->members < 2u)
        return words;
    words.a3 = chain->members | (red_skip ? PN_TT_PULL_RED_SKIP : 0u) |
               (serial ? PN_TT_PULL_SERIAL : 0u);
    words.a4 = chain->target;
    return words;
}
