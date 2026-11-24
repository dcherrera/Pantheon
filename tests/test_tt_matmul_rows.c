/* SPDX-License-Identifier: AGPL-3.0-or-later */
/* Copyright (C) 2026 David (Corey) Herrera */
/**
 * @file test_tt_matmul_rows.c
 * @brief The A-reuse ring per model and which Qwen3.8 decode matmuls it covers. No card.
 *
 *   1. No preference keeps the old default (128: Ornith's hidden Kt, reuse fires at Kt 128).
 *   2. Qwen3.8 prefers its hidden Kt 5120 / 32 = 160: reuse fires for every Kt-160 class
 *      (GDN in-proj, gate/up, attention q/k/v/gate, LM head) and not for out-proj (192) or
 *      down (544), nor for a multi-row-block descriptor.
 *   3. `PN_TT_A_REUSE` overrides either way; 0 turns it off.
 *   4. Share of Qwen3.8's decode A passes the ring covers (D, from the member shapes), printed.
 *
 * Build: `make tests/test_tt_matmul_rows` (in `make test`; no USE_TT, no card).
 *
 * @status None
 * @issues None
 * @todo None
 */

#include <stdint.h>
#include <stdio.h>

#include "../src/accel/tt/pn_tt_matmul_rows.h"

static int g_failed;
static int g_checks;

static void check(int ok, const char *what)
{
    g_checks++;
    if (!ok) {
        g_failed++;
        printf("FAIL %s\n", what);
    }
}

int main(void)
{
    /* Qwen3.8 decode member shapes (Kt, member Nt, count a token): 16 members, NB 4. */
    static const struct { const char *name; uint32_t kt, nt, per_token; } shapes[] = {
        { "gdn qkv", 160u, 20u, 48u }, { "gdn z", 160u, 12u, 48u }, { "gdn a", 160u, 2u, 48u },
        { "gdn b", 160u, 2u, 48u },    { "attn q+gate", 160u, 24u, 16u },
        { "attn k", 160u, 2u, 16u },   { "attn v", 160u, 2u, 16u },  { "gate", 160u, 34u, 64u },
        { "up", 160u, 34u, 64u },      { "out-proj", 192u, 10u, 64u },
        { "down", 544u, 10u, 64u },    { "lm head", 160u, 485u, 1u },
    };
    const uint32_t q38 = pn_tt_a_reuse_ring(0, 0u, 5120u / 32u);
    double passes = 0.0, saved = 0.0;
    unsigned i;

    check(pn_tt_a_reuse_ring(0, 0u, 0u) == 128u, "no preference keeps 128");
    check(pn_tt_a_reuse_fires(pn_tt_a_reuse_ring(0, 0u, 0u), 128u, 1u), "Ornith Kt 128 fires");
    check(q38 == 160u, "Qwen3.8 ring = 160");
    check(pn_tt_a_reuse_fires(q38, 160u, 1u), "Kt 160 fires");
    check(!pn_tt_a_reuse_fires(q38, 192u, 1u), "out-proj Kt 192 does not fire");
    check(!pn_tt_a_reuse_fires(q38, 544u, 1u), "down Kt 544 does not fire");
    check(!pn_tt_a_reuse_fires(q38, 160u, 2u), "MB 2 does not fire");
    check(pn_tt_a_reuse_ring(1, 0u, 160u) == 0u, "env 0 turns it off");
    check(!pn_tt_a_reuse_fires(pn_tt_a_reuse_ring(1, 0u, 160u), 160u, 1u), "off never fires");
    check(pn_tt_a_reuse_ring(1, 544u, 160u) == 544u, "env overrides the preference");
    for (i = 0; i < sizeof(shapes) / sizeof(shapes[0]); i++) {
        const double p = (double)((shapes[i].nt + 3u) / 4u) * shapes[i].per_token *
                         (double)shapes[i].kt;
        passes += p;
        if (pn_tt_a_reuse_fires(q38, shapes[i].kt, 1u))
            saved += p - (double)shapes[i].per_token * shapes[i].kt;
    }
    printf("Qwen3.8: ring 160 skips %.1f%% of decode A tile reads (D)\n", 100.0 * saved / passes);
    check(saved / passes > 0.5, "ring 160 skips most A reads");
    printf("test_tt_matmul_rows: %d checks, %d failed\n", g_checks, g_failed);
    return g_failed != 0;
}
