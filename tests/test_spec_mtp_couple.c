/* SPDX-License-Identifier: AGPL-3.0-or-later */
/* Copyright (C) 2026 David (Corey) Herrera */
/**
 * @file test_spec_mtp_couple.c
 * @brief Gate for coupled MTP drafts (spec_mtp_couple.h): the emitted stream stays the plain
 *        sampler's, byte for byte, and coupling raises the acceptance.
 *
 * @status Complete. CPU-only — no card, no model.
 * @issues None
 * @todo None
 *
 * Random cycles at the Qwen3.8 serve setting (T 1.0 / top-k 20 / top-p 0.95, repeat penalty
 * 1.0) and with a repeat penalty (1.1 over a 64 window): each cycle has 5 target rows (top-40
 * pairs, the verify's presented rows) and 4 head rows (top-20 pairs of a perturbed copy).
 * The cycle is run as cuda_spec_draw_rows runs it — drafts from pn_mtp_couple_pick, rows drawn
 * with pn_sample_next_token until the first draw that differs from its draft — and the emitted
 * tokens are checked against the same rows drawn plainly on an identical seeded stream.
 * Checks:
 *   1. byte-identity with plain draws, seeded and unseeded (the unseeded request is seeded
 *      once inside pn_mtp_couple_begin; its plain reference continues that stream);
 *   2. q = p (head pairs = the target's top 20): every draft accepted (>= 0.995);
 *   3. perturbed q: coupled acceptance above the argmax draft's, and the per-row analytic
 *      pn_mtp_couple_overlap matches the live acceptance;
 *   4. greedy requests keep argmax drafts (begin returns false, params untouched);
 *   5. the verify rows are drawn by pn_mtp_draw_row (the sparse-survivor path) against the
 *      plain reference's pn_sample_next_token (the full row): check 1 covers its identity,
 *      and the sparse path must have run.
 * `--selftest`: the red PN_MTP_RED=couple_consume (the peek advances the request's stream)
 * must FAIL check 1.
 */

#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "../src/core/engine.h"
#include "../src/core/sampling_modifiers.h"
#include "../src/core/spec_mtp.h"
#include "../src/core/spec_mtp_couple.h"

#define NV 4096
#define KP 40      /* target pairs a verify row presents */
#define KQ 20      /* head pairs */
#define ROWS 5     /* seed row + 4 drafts */
#define CYCLES 4000
#define WIN 64

static uint64_t g_rng = 0x9E3779B97F4A7C15ull;
static uint32_t xr(void) { g_rng ^= g_rng << 13; g_rng ^= g_rng >> 7; g_rng ^= g_rng << 17; return (uint32_t)g_rng; }
static float xf(void) { return (float)(xr() & 0xFFFFFF) / 16777216.0f; }
static float xn(void) { float u = xf() + 1e-7f, v = xf(); return sqrtf(-2.0f * logf(u)) * cosf(6.2831853f * v); }

typedef struct {
    int pid[ROWS][KP];
    float pval[ROWS][KP];
    int qid[ROWS][KQ];
    float qval[ROWS][KQ];
} cycle_rows;

/* Descending (value, id) pairs, ties by ascending id. */
static void sort_pairs(int *id, float *val, int n)
{
    for (int i = 1; i < n; i++)
        for (int j = i; j > 0 && (val[j] > val[j - 1] || (val[j] == val[j - 1] && id[j] < id[j - 1])); j--) {
            float tv = val[j]; val[j] = val[j - 1]; val[j - 1] = tv;
            int ti = id[j]; id[j] = id[j - 1]; id[j - 1] = ti;
        }
}

/* Target rows: KP distinct ids, a peaked head (scale 2.5). Head rows: the target's top KQ
 * with noise `sigma` on the logits (sigma 0: q = p over the top 20). */
static void make_cycle(cycle_rows *c, float sigma)
{
    for (int r = 0; r < ROWS; r++) {
        for (int i = 0; i < KP; i++) {
            int id, dup;
            do {
                id = (int)(xr() % NV);
                dup = 0;
                for (int j = 0; j < i; j++) dup |= c->pid[r][j] == id;
            } while (dup);
            c->pid[r][i] = id;
            c->pval[r][i] = 2.5f * xn();
        }
        sort_pairs(c->pid[r], c->pval[r], KP);
        for (int i = 0; i < KQ; i++) {
            c->qid[r][i] = c->pid[r][i];
            c->qval[r][i] = c->pval[r][i] + sigma * xn();
        }
        sort_pairs(c->qid[r], c->qval[r], KQ);
    }
}

static pn_sampling_params params_at(float rp, int seeded, uint64_t seed)
{
    pn_sampling_params sp = pn_sampling_defaults();
    sp.temperature = 1.0f;
    sp.top_k = 20;
    sp.top_p = 0.95f;
    sp.repeat_penalty = rp;
    sp.repeat_last_n = WIN;
    sp.rng_seeded = seeded;
    sp.rng_state = seeded ? pn_seed_to_rng_state(seed) : 0;
    return sp;
}

typedef struct {
    long cycles, drafts_checked, accepted, mismatches;
    double overlap_sum;
} run_stats;

/* One coupled cycle on `sp` (spec) and the same rows drawn plainly on `ref` (its stream must
 * stay in lockstep). Each window advances with what its loop emitted. */
static void run_cycle(const cycle_rows *c, float *row, int *prev, int *n_prev, int coupled,
                      pn_sampling_params *sp, uint32_t *recent, int *n_recent,
                      pn_sampling_params *ref, uint32_t *ref_recent, int *ref_n, run_stats *st)
{
    pn_mtp_couple cp;
    uint32_t drafts[ROWS - 1];
    int pick_ids[KQ];

    memset(&cp, 0, sizeof(cp));
    pn_mtp_couple_begin(&cp, sp, recent, *n_recent, WIN, NV, coupled ? ROWS - 1 : 0);
    if (!sp->rng_seeded)
        return;
    /* The reference starts where the request's (possibly just seeded) stream starts. */
    if (st->cycles == 0 && !ref->rng_seeded) {
        ref->rng_seeded = 1;
        ref->rng_state = sp->rng_state;
    }
    for (int i = 0; i < ROWS - 1; i++) {
        for (int j = 0; j < KQ; j++) pick_ids[j] = c->qid[i][j];
        drafts[i] = pn_mtp_couple_pick(&cp, i, pick_ids, c->qval[i], KQ);
    }
    int n_emit = 0;
    uint32_t emit[ROWS];
    for (int r = 0; r < ROWS; r++) {
        if (r < ROWS - 1 && coupled) {
            /* the analytic acceptance of this row's coupled draft */
            uint32_t pidx[KP];
            int pi[KP], qi[KQ];
            float pp[KP], qp[KQ];
            for (int j = 0; j < KP; j++) pidx[j] = (uint32_t)c->pid[r][j];
            int Sp = pn_sampler_pairs_final_probs(pidx, c->pval[r], KP, NV, sp, recent,
                                                  *n_recent, pi, pp, KP);
            int Sq = pn_sampler_pairs_final_probs(cp.ids[r], c->qval[r] /* sorted like ids */,
                                                  KQ, NV, sp, recent, *n_recent, qi, qp, KQ);
            if (Sp > 0 && Sq > 0)
                st->overlap_sum += pn_mtp_couple_overlap(pi, pp, Sp, qi, qp, Sq);
        }
        pn_mtp_present_row(row, prev, n_prev, c->pid[r], c->pval[r], KP);
        uint32_t tok = pn_mtp_draw_row(row, NV, sp, recent, *n_recent);   /* the verify's draw */
        pn_mtp_recent_push(recent, n_recent, WIN, tok);
        emit[n_emit++] = tok;
        if (r < ROWS - 1) {
            st->drafts_checked++;
            st->accepted += tok == drafts[r];
        }
        if (r == ROWS - 1 || tok != drafts[r])
            break;
    }
    /* Plain: the same rows, one draw each, for as many tokens as the cycle emitted. */
    for (int r = 0; r < n_emit; r++) {
        pn_mtp_present_row(row, prev, n_prev, c->pid[r], c->pval[r], KP);
        uint32_t tok = pn_sample_next_token(row, NV, ref, ref_recent, *ref_n);
        pn_mtp_recent_push(ref_recent, ref_n, WIN, tok);
        st->mismatches += tok != emit[r];
    }
    st->cycles++;
}

static run_stats run(float sigma, float rp, int seeded, int coupled, uint64_t seed)
{
    static float row[NV];
    int prev[KP], n_prev = 0;
    uint32_t recent[WIN], ref_recent[WIN];
    int n_recent = 0, ref_n = 0;
    pn_sampling_params sp = params_at(rp, seeded, seed), ref = params_at(rp, seeded, seed);
    run_stats st = { 0 };
    cycle_rows c;

    for (int i = 0; i < NV; i++) row[i] = -INFINITY;
    g_rng = 0x9E3779B97F4A7C15ull ^ seed;
    for (int k = 0; k < CYCLES; k++) {
        make_cycle(&c, sigma);
        run_cycle(&c, row, prev, &n_prev, coupled, &sp, recent, &n_recent, &ref, ref_recent,
                  &ref_n, &st);
    }
    return st;
}

static int fails;
#define CHECK(cond, ...) do { if (!(cond)) { printf("FAIL: "); printf(__VA_ARGS__); printf("\n"); fails++; } else { printf("ok:   "); printf(__VA_ARGS__); printf("\n"); } } while (0)

int main(int argc, char **argv)
{
    const int selftest = argc > 1 && strcmp(argv[1], "--selftest") == 0;

    if (selftest) {
        setenv("PN_MTP_RED", "couple_consume", 1);
        run_stats r = run(0.5f, 1.0f, 1, 1, 7);
        printf("red couple_consume: %ld mismatches over %ld cycles\n", r.mismatches, r.cycles);
        if (r.mismatches == 0) {
            printf("SELFTEST FAIL: the red arm passed the byte-identity gate\n");
            return 1;
        }
        printf("SELFTEST OK: the red arm fails the byte-identity gate\n");
        return 0;
    }

    const unsigned long hits0 = pn_sampler_sparse_hits();
    for (int rpi = 0; rpi < 2; rpi++) {
        const float rp = rpi ? 1.1f : 1.0f;
        run_stats eq = run(0.0f, rp, 1, 1, 11);
        CHECK(eq.mismatches == 0, "rp %.1f q=p seeded: stream identical to plain (%ld cycles)",
              rp, eq.cycles);
        CHECK((double)eq.accepted / eq.drafts_checked >= 0.995,
              "rp %.1f q=p: coupled acceptance %.4f >= 0.995", rp,
              (double)eq.accepted / eq.drafts_checked);
        run_stats cpl = run(0.6f, rp, 1, 1, 13), arg = run(0.6f, rp, 1, 0, 13);
        CHECK(cpl.mismatches == 0 && arg.mismatches == 0,
              "rp %.1f perturbed q: coupled and argmax streams identical to plain", rp);
        double ac = (double)cpl.accepted / cpl.drafts_checked,
               aa = (double)arg.accepted / arg.drafts_checked,
               ov = cpl.overlap_sum / cpl.drafts_checked;
        CHECK(ac > aa + 0.05, "rp %.1f perturbed q: coupled %.4f > argmax %.4f + 0.05", rp, ac,
              aa);
        CHECK(fabs(ov - ac) < 0.02, "rp %.1f analytic overlap %.4f matches live %.4f", rp, ov,
              ac);
        run_stats un = run(0.6f, rp, 0, 1, 17);
        CHECK(un.mismatches == 0 && un.cycles == CYCLES,
              "rp %.1f unseeded: seeded once by begin, stream identical to plain", rp);
    }
    CHECK(pn_sampler_sparse_hits() - hits0 > (unsigned long)CYCLES,
          "verify draws ran the sparse-survivor path (%lu hits)", pn_sampler_sparse_hits() - hits0);
    {
        pn_sampling_params g = params_at(1.0f, 1, 5), before;
        pn_mtp_couple cp;
        memset(&cp, 0, sizeof(cp));
        g.temperature = 0.0f;
        before = g;
        bool on = pn_mtp_couple_begin(&cp, &g, NULL, 0, WIN, NV, 4);
        int ids[2] = { 7, 3 };
        float vals[2] = { 1.0f, 2.0f };
        CHECK(!on && memcmp(&g, &before, sizeof(g)) == 0 &&
              pn_mtp_couple_pick(&cp, 0, ids, vals, 2) == 7,
              "greedy: not coupled, params untouched, the first pair drafted");
    }
    printf(fails ? "test_spec_mtp_couple: %d FAILED\n" : "test_spec_mtp_couple: all passed\n",
           fails);
    return fails ? 1 : 0;
}
