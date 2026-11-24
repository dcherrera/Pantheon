/* SPDX-License-Identifier: AGPL-3.0-or-later */
/* Copyright (C) 2026 David (Corey) Herrera */
/**
 * @file tt_prefix_cache_link_stubs.c
 * @brief Link stubs for host-only tests that compile src/core/prefix_cache.c without the
 *        qwen35moe TT sources. Under USE_TT, prefix_cache.c's card tier calls three model
 *        functions; these tests never put KV on a card, so the tier is never entered and
 *        the stubs only satisfy the linker. Empty in any other build.
 *
 * @status Used by test_prefix_cache_sys and test_session_watch.
 * @issues None
 * @todo None
 */

#ifdef USE_TT
#include "../src/models/qwen35moe.h"

int pn_qwen35moe_model_tt_kv_on_card(const pn_model *model)
{
    (void)model;
    return 0;
}

int pn_qwen35moe_tt_checkpoint_resume(pn_model *model, pn_state *state, const uint32_t *tokens,
                                      int n)
{
    (void)model; (void)state; (void)tokens; (void)n;
    return 0;
}

int pn_qwen35moe_tt_card_holds(const pn_model *model, const pn_state *state)
{
    (void)model; (void)state;
    return 0;
}

int pn_qwen35moe_tt_checkpoint_available(const pn_model *model)
{
    (void)model;
    return 0;
}
#else
typedef int pn_tt_prefix_cache_link_stubs_unused;
#endif
