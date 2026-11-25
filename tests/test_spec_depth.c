/* SPDX-License-Identifier: AGPL-3.0-or-later */
/* Copyright (C) 2026 David (Corey) Herrera */
/**
 * @file test_spec_depth.c
 * @brief Unit test of the adaptive draft-depth controller (src/core/spec_depth.c): the
 *        rate model, the depth choice, the plain-switch rule and the running updates, and
 *        a simulated decode loop at Qwen3.8's measured K80 costs.
 *
 * @status complete
 * @issues None
 * @todo None
 */
#include "../src/core/spec_depth.h"

#include <math.h>
#include <stdio.h>
#include <stdlib.h>

static int g_fail;

#define CHECK(cond, msg)                                                    \
    do {                                                                    \
        if (!(cond)) {                                                      \
            printf("FAIL %s (%s:%d)\n", msg, __FILE__, __LINE__);           \
            g_fail++;                                                       \
        }                                                                   \
    } while (0)

/* A Qwen3.8 cost/acceptance world: verify + commit of n rows on the line through two depths,
 * the plain decode-slots step, ms an MTP draft, the head's conditional acceptance. */
typedef struct {
    double v_lo[6], v_hi[6], d_lo, d_hi, plain_c, plain_s, draft, acc[PN_SPEC_DEPTH_MAX + 1];
} q38_world;

/* Today: Qwen3.8-27B Q8_0 on K80 dies 8-11 under TP2 (k80-q38-tp2, k80-q38-dflash2-scout
 * bench_spec_verify_gate at 518 / 8198), 8.2 ms an MTP draft; the head's conditional
 * acceptance as measured live at T 1.0 (k80-q38-mtp-accept PN_MTP_PROBE, coupled drafts):
 * general prompts 0.798 / 0.755 / 0.752 (the 4th assumed alike), code-edit replies
 * 0.928 / 0.905 / 0.907. */
static const q38_world q38_tp2_gen = {
    { 0.0, 140.14, 164.14, 178.19, 189.74, 203.75 }, { 0.0, 145.59, 178.54, 199.35, 219.17, 243.61 },
    518, 8198, 154.3, 0.00071, 8.2, { 1.0, 0.798, 0.755, 0.752, 0.75 } };
static const q38_world q38_tp2_edit = {
    { 0.0, 140.14, 164.14, 178.19, 189.74, 203.75 }, { 0.0, 145.59, 178.54, 199.35, 219.17, 243.61 },
    518, 8198, 154.3, 0.00071, 8.2, { 1.0, 0.928, 0.905, 0.907, 0.9 } };
/* Legacy: the 4-die layer split before the verify tiles (q38-nvidia-draft-scout, 512 / 2048),
 * a costly third row and a 1.6x fifth, ~10 ms a draft, conditional acceptance from the head's
 * fixed-depth taus 1.84 / 2.63 / 3.04. Run with its own priors (QWEN35_SPEC_COST, below): it
 * keeps `more`'s look-ahead and the relaxing of untested depths under test. */
static const q38_world q38_legacy = {
    { 0.0, 260.4, 287.8, 340.8, 353.9, 572.0 }, { 0.0, 261.3, 290.8, 351.7, 366.6, 585.1 },
    512, 2048, 259.7, 0.0006, 10.0, { 1.0, 0.843, 0.934, 0.52, 0.5 } };
static const char *q38_legacy_priors =
    "260.10,0.000586;286.80,0.001953;337.17,0.007096;349.67,0.008268;567.63,0.008529;"
    "259.39,0.000586;10,10";
static const q38_world *g_w = &q38_tp2_gen;

static double q38_verify_ms(int n, int pos)
{
    return g_w->v_lo[n] + (g_w->v_hi[n] - g_w->v_lo[n]) / (g_w->d_hi - g_w->d_lo) * (pos - g_w->d_lo);
}

static double q38_plain_ms(int pos) { return g_w->plain_c + g_w->plain_s * (pos - 512); }

static unsigned long long g_rng;

static double q38_uniform(void)
{
    g_rng = g_rng * 6364136223846793005ULL + 1442695040888963407ULL;
    return (double)(g_rng >> 11) * (1.0 / 9007199254740992.0);
}

/* Tokens per second over 512 tokens from pos0. fixed >= 0: that many drafts, no controller.
 * batched: D = 0 is the plain step (the serve's batched slot); otherwise D = 0 is a one-row
 * verify and the one-way plain switch is allowed (the spec_mtp.c loop). *switched = the
 * switch position, -1 = never. */
static double q38_sim(int fixed, int batched, int dense, int pos0, int *switched)
{
    pn_spec_depth c;
    int pos = pos0, n = 0, plain = 0;
    double ms = 0.0;

    pn_spec_depth_init(&c, PN_SPEC_DEPTH_MAX);
    if (dense)
        pn_spec_depth_dense_priors(&c);
    if (batched)
        pn_spec_depth_d0_is_plain(&c);
    g_rng = 7;
    *switched = -1;
    while (n < 512) {
        int d = fixed;
        if (!plain && fixed < 0) {
            d = pn_spec_depth_choose(&c, pos);
            if (!batched && pn_spec_depth_switch_plain(&c, pos)) {
                plain = 1;
                *switched = pos;
            }
        }
        if (plain || (batched && d == 0)) {
            double t = q38_plain_ms(pos);
            if (batched)
                pn_spec_depth_observe_plain(&c, pos, t);
            ms += t;
            n++;
            pos++;
            continue;
        }
        float confs[PN_SPEC_DEPTH_MAX];
        int nd = 0, acc = 0;
        for (int i = 1; i <= d; i++) {
            confs[i - 1] = (float)g_w->acc[i];
            nd = i;
            if (fixed < 0 && i < d && !pn_spec_depth_more(&c, pos, i, confs))
                break;
        }
        while (acc < nd && q38_uniform() < g_w->acc[acc + 1])
            acc++;
        double td = g_w->draft * nd, tv = q38_verify_ms(nd + 1, pos);
        if (fixed < 0) {
            if (nd > 0)
                pn_spec_depth_observe_draft(&c, nd, td);
            pn_spec_depth_observe_verify(&c, nd + 1, pos, tv);
            if (nd > 0)
                pn_spec_depth_observe_accept(&c, nd, acc, confs);
        }
        ms += td + tv;
        n += acc + 1;
        pos += acc + 1;
    }
    return n / (ms / 1e3);
}

static void test_q38_k80(void)
{
    int sw, sw_fixed;

    /* The spec_mtp.c loop on the MoE priors: an unobserved plain step must be priced as slow
     * as the verifies show this model to be, or the MoE plain prior wins the switch
     * after two cycles and the run decodes plain (bench_mtp_e2e `ad` 3.94 tok/s vs d3 7.30).
     * The MoE priors carry Qwen3.6's row shape (4 rows 2.3x one at 19k), Qwen3.8's is 1.4x, so
     * the one running ratio leaves adaptive at 0.83x d3 here; dense models load the dense
     * priors (serve_spec_mtp_cuda.c, bench_mtp_e2e). */
    double d3 = q38_sim(3, 0, 0, 512, &sw_fixed);
    double ad = q38_sim(-1, 0, 0, 512, &sw);
    CHECK(sw < 0, "Qwen3.8, MoE priors: no plain switch");
    CHECK(ad >= 0.80 * d3, "Qwen3.8, MoE priors: adaptive near fixed d3");

    /* Dense priors: adaptive keeps up with the best fixed depth on both loops, short and 16k
     * context, at general acceptance (best d3) and at code-edit acceptance (best d4: the 5-row
     * verify is only ~1.1x the 4-row one under TP2). The legacy world runs on its own priors
     * (env costs) and needs `more` to look past the costly third row to the cheap fourth. */
    const q38_world *worlds[3] = { &q38_tp2_gen, &q38_tp2_edit, &q38_legacy };
    const char *names[3] = { "general", "edit", "legacy" };
    for (int w = 0; w < 3; w++) {
        g_w = worlds[w];
        if (g_w == &q38_legacy)
            setenv("QWEN35_SPEC_COST", q38_legacy_priors, 1);
        for (int batched = 0; batched <= 1; batched++)
            for (int pos = 512; pos <= 16384; pos += 16384 - 512) {
                d3 = q38_sim(3, batched, 1, pos, &sw_fixed);
                double d4 = q38_sim(4, batched, 1, pos, &sw_fixed);
                /* the legacy world's d4 pays the 1.6x fifth row: its bar stays fixed d3 */
                double best = g_w != &q38_legacy && d4 > d3 ? d4 : d3;
                ad = q38_sim(-1, batched, 1, pos, &sw);
                /* edit at 16k: adaptive 13.20 vs d4 13.67 (the pre-TP2 priors locked d3,
                 * 13.04 = 0.954x) — the acceptance it learns sits on the d3/d4 edge there. */
                const double bar = g_w == &q38_tp2_edit ? 0.96 : 0.97;
                if (sw >= 0 || ad < bar * best)
                    printf("  q38 %s batched=%d pos=%d: adaptive %.2f vs d3 %.2f d4 %.2f tok/s, "
                           "switch %d\n", names[w], batched, pos, ad, d3, d4, sw);
                CHECK(sw < 0 && ad >= bar * best,
                      "Qwen3.8, dense priors: adaptive near the best fixed depth");
                if (g_w == &q38_tp2_edit)
                    CHECK(d4 > d3, "Qwen3.8 TP2, edit acceptance: d4 beats d3");
            }
        unsetenv("QWEN35_SPEC_COST");
    }
    g_w = &q38_tp2_gen;
}

int main(void)
{
    pn_spec_depth c;

    unsetenv("QWEN35_SPEC_COST");
    pn_spec_depth_init(&c, 4);

    /* Rate = expected tokens / (draft + verify): D = 0 is one token per 1-row verify. */
    double r0 = pn_spec_depth_rate(&c, 0, 1024);
    CHECK(fabs(r0 - 1.0 / (25.27 + 0.000233 * 1024)) < 1e-12, "D=0 rate");
    double r1 = pn_spec_depth_rate(&c, 1, 1024);
    CHECK(fabs(r1 - 1.8 / (4.5 + 37.24 + 0.000261 * 1024)) < 1e-12, "D=1 rate");

    /* Short context speculates; very long context with low acceptance does not. */
    CHECK(pn_spec_depth_choose(&c, 1024) >= 1, "short ctx drafts");
    for (int i = 1; i <= 4; i++)
        c.accept[i] = 0.2;
    CHECK(pn_spec_depth_choose(&c, 32768) == 0, "low acceptance -> D=0");
    CHECK(c.d0_run == 1, "d0 run counted");
    for (int i = 0; i < 100 && pn_spec_depth_choose(&c, 1024) == 0; i++)
        ;
    CHECK(pn_spec_depth_choose(&c, 1024) >= 1, "a D = 0 run relaxes back to drafting");
    pn_spec_depth_init(&c, 4);

    /* The plain switch: never at 1k or 32k at good acceptance. */
    CHECK(!pn_spec_depth_plain_wins(&c, 1024, 0.9), "no switch at 1k");
    CHECK(!pn_spec_depth_plain_wins(&c, 32768, 0.9), "no switch at 32k, good acceptance");
    c.ratio = 1.6;       /* verifies slower than modelled: an unobserved plain step too */
    for (int i = 1; i <= 4; i++)
        c.accept[i] = 0.1;
    CHECK(!pn_spec_depth_plain_wins(&c, 32768, 0.1), "unobserved plain is as slow: no switch");
    c.plain_c = 20.0;    /* a plain step cheaper than every verify ... */
    CHECK(pn_spec_depth_plain_wins(&c, 32768, 0.1), "... and poor acceptance: switch");
    pn_spec_depth_init(&c, 4);

    /* dmax caps both the choice and the continue rule. */
    pn_spec_depth_init(&c, 1);
    float confs[4] = { 1.0f, 1.0f, 1.0f, 1.0f };
    CHECK(pn_spec_depth_choose(&c, 1024) <= 1, "dmax caps choice");
    CHECK(!pn_spec_depth_more(&c, 1024, 1, confs), "dmax caps more");
    pn_spec_depth_init(&c, 4);
    CHECK(pn_spec_depth_more(&c, 1024, 1, confs), "confident draft continues at 1k");
    confs[0] = 0.05f;
    CHECK(!pn_spec_depth_more(&c, 1024, 1, confs), "unsure draft stops");

    /* A slower verify than modelled moves the running ratio. */
    for (int i = 0; i < 200; i++)
        pn_spec_depth_observe_verify(&c, 2, 1024, 200.0);
    CHECK(c.ratio > 3.0, "ratio tracks");
    pn_spec_depth_observe_accept(&c, 2, 0, confs);
    CHECK(c.accept[1] < 0.80 && fabs(c.accept[2] - 0.79) < 1e-12,
          "only the reached depth updates");

    /* Batched path: D = 0 is the plain step, priced by the plain model and its own ratio. */
    pn_spec_depth_init(&c, 4);
    pn_spec_depth_d0_is_plain(&c);
    CHECK(fabs(pn_spec_depth_rate(&c, 0, 8192) - pn_spec_depth_plain_rate(&c, 8192)) < 1e-15,
          "D=0 rate is the plain rate");
    c.ratio = 1.5;   /* a hot die: the unobserved plain step is priced as hot as the verify */
    CHECK(fabs(pn_spec_depth_plain_rate(&c, 0) - 1.0 / (c.plain_c * 1.5)) < 1e-12,
          "unobserved plain follows the verify ratio");
    c.ratio = 1.0;
    for (int i = 0; i < 200; i++)
        pn_spec_depth_observe_plain(&c, 8192, 2.0 * (c.plain_c + c.plain_s * 8192));
    CHECK(fabs(c.plain_ratio - 2.0) < 1e-3, "plain ratio tracks");
    CHECK(pn_spec_depth_choose(&c, 8192) >= 1, "a slow plain step makes drafting pay");
    for (int i = 0; i < 200; i++)
        pn_spec_depth_observe_extra(&c, 3.0);
    CHECK(fabs(c.extra_ms - 3.0) < 1e-3, "extra tracks");
    CHECK(fabs(pn_spec_depth_rate(&c, 1, 1024) -
               1.8 / (c.draft_first + (c.verify_c[2] + c.verify_s[2] * 1024) + c.extra_ms)) <
              1e-12,
          "extra is in the cycle cost");

    /* Dense priors replace the MoE ones and survive the batched D = 0 arming. */
    pn_spec_depth_init(&c, 4);
    pn_spec_depth_dense_priors(&c);
    pn_spec_depth_d0_is_plain(&c);
    CHECK(c.verify_c[1] > 130.0 && c.verify_c[5] < 1.2 * c.verify_c[4] && c.plain_c > 130.0 &&
              c.draft_first == 8.2, "dense priors (TP2)");
    test_q38_k80();

    /* QWEN35_SPEC_COST overrides every prior. */
    setenv("QWEN35_SPEC_COST", "1,0;2,0;3,0;4,0;5,0;10,0;0.5,0.5", 1);
    pn_spec_depth_init(&c, 4);
    CHECK(c.verify_c[5] == 5.0 && c.plain_c == 10.0 && c.draft_next == 0.5, "env costs");
    pn_spec_depth_dense_priors(&c);
    pn_spec_depth_d0_is_plain(&c);
    CHECK(c.verify_c[5] == 5.0 && c.plain_c == 10.0, "env costs beat dense priors");

    printf(g_fail ? "test_spec_depth: %d FAILED\n" : "test_spec_depth: all passed%.0d\n", g_fail);
    return g_fail ? 1 : 0;
}
