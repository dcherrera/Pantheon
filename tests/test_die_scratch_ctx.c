/* SPDX-License-Identifier: AGPL-3.0-or-later */
/* Copyright (C) 2026 David (Corey) Herrera */
/*
 * test_die_scratch_ctx.c — the per-die GPU scratch reserve is priced for the
 * window the model is actually capped to, not for a 32k one.
 *
 * PN_ADMIT_DIE_SCRATCH_BYTES was a single 1.5 GiB constant measured at a 32768
 * window. Most of that is the per-slot on-die KV ring
 * ([nslots][n_head_kv][max_seq][row] per layer), which is proportional to the
 * cap — so charging it flat made a 2048-token model pay a 32k price. That
 * refused a 2-die Qwen3.6-35B-A3B Q4_K_M split whose real per-die peak was
 * 10,570 / 10,879 MiB of 12,206:
 *
 *   [admit] REFUSE model load: die ordinal 6 has 11980 MiB free but its split
 *   share needs ~11399 MiB (20/40 layers + KV@2048 + scratch)
 *
 * The sibling test (test_admission_reserve.c) covers the OTHER half of the same
 * family of bug: the per-die reserve being charged twice on the mr_load path.
 * Here the reserve is charged exactly once and stays charged — this is a
 * repricing, not a loosening.
 */

#include "../src/core/mem/pn_admission.h"

#include <stdio.h>

uint64_t pn_die_reserve_bytes(void);

static int g_failed = 0;

#define CHECK(cond, msg) do { \
    if (!(cond)) { \
        fprintf(stderr, "  FAIL: %s (%s:%d)\n", (msg), __FILE__, __LINE__); \
        g_failed++; \
    } \
} while (0)

#define MIB(n) ((uint64_t)(n) << 20)

/* The K80 test machine's measured Qwen3.6-35B-A3B Q4_K_M split across two K80 dies. */
#define K80_DIE_FREE_MIB     11980   /* raw free, after the CUDA context      */
#define K80_DIE_A_WEIGHTS    9857    /* layers [0,20)                          */
#define K80_DIE_B_WEIGHTS    10254   /* layers [20,40) + the 397 MiB LM head   */
#define K80_KV_2048          6       /* 20/40 layers of KV at a 2048 cap       */
#define K80_KV_32768         96      /* the same share at a 32768 cap          */

/* Mirrors PN_QWEN35MOE_MAX_SEQ_CAP (qwen35moe.h) — the widest window any model
 * here can be capped to, and so the widest the reserve is ever asked about. */
#define ENGINE_MAX_SEQ_CAP    262144

int main(void)
{
    printf("test_die_scratch_ctx\n");
    uint64_t reserve = pn_die_reserve_bytes();
    CHECK(reserve > 0, "there is a per-die reserve to hold back");

    /* --- The reserve is a curve, not a constant ------------------------- */

    /* The one point the 1.5 GiB constant was ever measured at must be
     * unchanged: this repricing fixes the extrapolation, it does not move the
     * calibration. */
    CHECK(pn_admit_die_scratch_bytes(32768) == MIB(1536),
          "a 32768 window still costs exactly the measured 1.5 GiB");

    /* Below it the charge falls, because the per-slot ring does. */
    CHECK(pn_admit_die_scratch_bytes(2048) == MIB(576),
          "a 2048 window costs the base plus its own ring, not a 32k ring");
    CHECK(pn_admit_die_scratch_bytes(2048) <
              pn_admit_die_scratch_bytes(8192),
          "a wider window reserves more");
    CHECK(pn_admit_die_scratch_bytes(8192) <
              pn_admit_die_scratch_bytes(32768),
          "and wider still reserves more again");

    /* Above the calibration point the charge STOPS growing. The slope is sized
     * for the default 8 sessions; a production serve pinned to --max-sessions 2
     * at a 131072 window allocates a quarter of that ring, so letting the line
     * run would refuse a split that runs today. This repricing must only ever
     * charge less than the flat constant, never more. */
    CHECK(pn_admit_die_scratch_bytes(131072) == MIB(1536),
          "a 131072 window costs no more than the constant it replaces");
    CHECK(pn_admit_die_scratch_bytes(ENGINE_MAX_SEQ_CAP) == MIB(1536),
          "and neither does the widest window the engine allows");

    int over = 0, backwards = 0;
    uint64_t prev = 0;
    for (int ctx = 0; ctx <= ENGINE_MAX_SEQ_CAP; ctx += 331) {
        uint64_t s = pn_admit_die_scratch_bytes(ctx);
        if (s > MIB(1536)) over++;
        if (s < prev) backwards++;
        prev = s;
    }
    CHECK(over == 0, "no window anywhere is charged more than the old constant");
    CHECK(backwards == 0, "and the charge never falls as the window widens");

    /* No cap known is the base alone — never zero, and never negative-indexed
     * into the per-token term. */
    CHECK(pn_admit_die_scratch_bytes(0) == PN_ADMIT_DIE_SCRATCH_BASE_BYTES,
          "no window known still reserves the fixed base");
    CHECK(pn_admit_die_scratch_bytes(-1) == PN_ADMIT_DIE_SCRATCH_BASE_BYTES,
          "a garbage window does not underflow the reserve");

    /* --- The K80 test machine case: it fits, so it must be admitted ----------------- */

    uint64_t die_free = MIB(K80_DIE_FREE_MIB);
    uint64_t usable = die_free - reserve;
    uint64_t need = 0;

    /* First reproduce the refusal, so this test is anchored to the real bug:
     * priced at the old flat 1.5 GiB, die A does not fit. */
    CHECK(MIB(K80_DIE_A_WEIGHTS) + MIB(K80_KV_2048) + MIB(1536) > usable,
          "the old flat 32k price refused die A -- the bug, reproduced");

    CHECK(pn_admit_model_load_die(MIB(K80_DIE_A_WEIGHTS), MIB(K80_KV_2048),
                                  2048, die_free, &need),
          "die A admits its 20 layers at a 2048 cap");
    CHECK(need == MIB(K80_DIE_A_WEIGHTS) + MIB(K80_KV_2048) + MIB(576),
          "and the projection is weights + KV + the 2048-window scratch");
    CHECK(need <= usable, "the printed numbers agree it fits");

    /* Die B carries the LM head as well and is the tighter of the two. */
    CHECK(pn_admit_model_load_die(MIB(K80_DIE_B_WEIGHTS), MIB(K80_KV_2048),
                                  2048, die_free, &need),
          "die B admits its 20 layers plus the LM head");

    /* The measured peaks stayed inside the reserve floor, which is why
     * admitting this load was correct rather than merely convenient. */
    CHECK(MIB(10879) <= MIB(12206) - reserve,
          "die B's real peak left the per-die reserve intact");

    /* --- It is a repricing, not a loosening ----------------------------- */

    /* At the window the constant was measured at, the gate refuses exactly what
     * it refused before: the same load capped to 32768 does not fit. */
    CHECK(!pn_admit_model_load_die(MIB(K80_DIE_A_WEIGHTS), MIB(K80_KV_32768),
                                   32768, die_free, &need),
          "the same dies still refuse a 32768 cap");

    /* And the per-die reserve is still held back, exactly once: the admission
     * boundary sits at need + reserve, not at need. */
    uint64_t probe_w = MIB(1000), probe_kv = MIB(10);
    uint64_t probe_need = 0;
    (void)pn_admit_model_load_die(probe_w, probe_kv, 2048, 0, &probe_need);
    CHECK(probe_need == probe_w + probe_kv + MIB(576),
          "the projection does not depend on the free figure");
    CHECK(pn_admit_model_load_die(probe_w, probe_kv, 2048,
                                  probe_need + reserve, NULL),
          "free == need + reserve is the last figure that admits");
    CHECK(!pn_admit_model_load_die(probe_w, probe_kv, 2048,
                                   probe_need + reserve - 1, NULL),
          "one byte into the reserve is a refusal");

    /* --- A die pays the KV of ITS OWN layers ---------------------------- */

    /* Qwen3.6-35B-A3B reports 6144 B/token whole-model from its real geometry.
     * Split 20/40, one die holds half of that. */
    uint64_t whole_2k = 6144ULL * 2048;
    CHECK(pn_admit_die_kv_bytes(6144, true, 2048, 20, 40) == whole_2k / 2,
          "a 20-of-40-layer die pays half the model's KV");
    CHECK(pn_admit_die_kv_bytes(6144, true, 2048, 40, 40) == whole_2k,
          "a die holding every layer pays all of it");

    /* An unreadable layout must charge the WHOLE model's KV -- not zero, and
     * not a guessed share. Gated conservatively rather than admitted blind. */
    CHECK(pn_admit_die_kv_bytes(6144, true, 2048, 0, 0) == whole_2k,
          "an unknown layer count charges the whole model, not nothing");
    CHECK(pn_admit_die_kv_bytes(6144, true, 2048, 0, 40) == whole_2k,
          "and so does a zero layer share");
    CHECK(pn_admit_die_kv_bytes(6144, true, 0, 20, 40) == 0,
          "no window means no KV");

    /* kv_per_token 0 means "not resident yet" -- fall back to the generic
     * constant, never to zero. */
    CHECK(pn_admit_die_kv_bytes(0, true, 2048, 40, 40) ==
              PN_ADMIT_KV_BYTES_PER_TOKEN_Q4 * 2048,
          "a pre-load caller gets the generic constant, not a free pass");

    /* --- The deployed config's shape, at the cap it actually runs ------------- */

    /* qwen36-stock: Qwen3.6-35B-A3B Q3_K_M split over two dies at --ctx-cap
     * 131072. Measured per-die weight projection 7425 MiB against a
     * die-manager budget of 11182 MiB (12,206 total less the 1 GiB reserve).
     *
     * This is the load the CPU-path LUT refused. Priced as the GPU load it is:
     *
     *     8192:   7425 + 23  + 768  =  8216 MiB
     *     32768:  7425 + 92  + 1536 =  9053 MiB
     *     131072: 7425 + 369 + 1536 =  9330 MiB
     *
     * against 11182 MiB. Priced through the CPU-resident gate the same load
     * needed 56363 MiB at 131072 -- 48.2 GiB of it host activation LUT. */
    const uint64_t prod_weights = MIB(7425);
    const uint64_t prod_budget  = MIB(11182);
    uint64_t kv8k   = pn_admit_die_kv_bytes(0, true, 8192,   20, 40);
    uint64_t kv32k  = pn_admit_die_kv_bytes(0, true, 32768,  20, 40);
    uint64_t kv128k = pn_admit_die_kv_bytes(0, true, 131072, 20, 40);
    uint64_t n8k = 0, n32k = 0, n128k = 0;

    CHECK(pn_admit_model_load_die_budgeted(prod_weights, kv8k, 8192,
                                           prod_budget, &n8k),
          "the deployed config's shape admits at an 8192 cap");
    CHECK(pn_admit_model_load_die_budgeted(prod_weights, kv32k, 32768,
                                           prod_budget, &n32k),
          "the deployed config's shape admits at a 32768 cap");
    CHECK(pn_admit_model_load_die_budgeted(prod_weights, kv128k, 131072,
                                           prod_budget, &n128k),
          "the deployed config's shape admits at the 131072 cap it really runs");
    CHECK(n8k < n32k && n32k < n128k,
          "and a wider cap still costs more -- the gate did not go blind");
    CHECK(n128k + MIB(1800) < prod_budget,
          "it admits with real headroom, not by a hair");

    /* Still a gate: the same cap, a die whose weight share is 3 GiB larger,
     * does not fit and must be refused. */
    uint64_t over_need = 0;
    CHECK(!pn_admit_model_load_die_budgeted(MIB(10500), kv128k, 131072,
                                            prod_budget, &over_need),
          "a share that genuinely overruns the budget is still refused");
    CHECK(over_need > prod_budget, "and the printed numbers say why");

    if (g_failed) {
        printf("FAILED: %d check(s)\n", g_failed);
        return 1;
    }
    printf("ALL PASS\n");
    return 0;
}
