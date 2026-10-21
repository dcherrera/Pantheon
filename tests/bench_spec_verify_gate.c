/* SPDX-License-Identifier: AGPL-3.0-or-later */
/* Copyright (C) 2026 David (Corey) Herrera */
/* Phase 9.1 gate bench: does verifying D rows in ONE forward cost materially
 * less than D sequential single-token forwards?
 *
 * This is the decision gate for speculative decoding on Kepler-class hardware.
 * Pure measurement: it changes no engine behaviour and is not part of the test
 * suite (needs a model and a GPU).
 *
 * Three cost curves, all at the SAME context depth, on ONE model load:
 *
 *   A. SEQUENTIAL single-token forward (`model->forward`) — the legacy
 *      single-user decode step.  The thing speculation must beat D-for-D.
 *   B. SPECULATIVE VERIFY of D rows (`pn_qwen35moe_spec_verify` +
 *      `pn_qwen35moe_spec_accept`) — one forward carrying D candidate rows of
 *      ONE session.  accept(1) is timed with it: that is the WORST-CASE commit
 *      (full SSM rollback); accept(D) is free by construction (the commit
 *      returns immediately when n_accept == n).
 *   C. BATCHED SLOTS step (`pn_qwen35moe_decode_slots`) at K slots — the
 *      serving path's cost as a function of rows-in-one-forward, which is what
 *      a K-user speculative verify would have to pay.
 *
 * The K=1 gate is B(D) vs D x A.  The K=2 gate is C(2D) vs D x C(2): per round
 * two users each advance up to D tokens in one 2D-row forward, versus D
 * ordinary two-row steps.
 *
 * Build (CUDA host): make USE_CUDA=1 tests/bench_spec_verify_gate
 * Run: QWEN35_GPU_SPLIT_LAYER=20 QWEN35_GPU_DIE_A=6 QWEN35_GPU_DIE_B=7 \
 *        QWEN35_KV_Q4=1 QWEN35_MAX_SEQ=8192 QWEN35_FUSE=0x37 \
 *        ./tests/bench_spec_verify_gate <model.gguf> [depth] [reps] [chunk]
 * Env: QWEN35_GATE_ROWS  ("1,2,4,8")  verify widths D to sweep
 *      QWEN35_GATE_SLOTS ("1,2,4,8")  slot counts K to sweep (0 skips C)
 */
#include "../include/pantheon.h"
#include "../src/core/engine.h"
#include "../src/models/qwen35moe.h"
#include "tensor/tensor.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

#ifndef USE_CUDA
int main(void) { fprintf(stderr, "built without USE_CUDA\n"); return 1; }
#else

#define GATE_MAX_POINTS 12
#define GATE_MAX_SLOTS  16
#define GATE_MAX_ROWS   64

static double now_s(void)
{
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (double)ts.tv_sec + (double)ts.tv_nsec * 1e-9;
}

static int argmax_f32(const float *x, int n)
{
    int best = 0;
    for (int i = 1; i < n; i++)
        if (x[i] > x[best]) best = i;
    return best;
}

/* Parse "1,2,4,8" into out[]; returns count (0 on empty/"0"). */
static int parse_int_list(const char *spec, int *out, int max_out)
{
    int n = 0;
    while (spec && *spec && n < max_out) {
        char *end = NULL;
        long v = strtol(spec, &end, 10);
        if (end == spec) break;
        if (v > 0) out[n++] = (int)v;
        spec = (*end == ',') ? end + 1 : end;
    }
    return n;
}

/* Deterministic, non-degenerate seed context so the KV/attention is realistic
 * (not one repeated id).  `user` keeps concurrent slots distinct. */
static void fill_seed_context(uint32_t *buf, int depth, int n_vocab, int user)
{
    uint32_t base = 100u + (uint32_t)(user * 131);
    for (int i = 0; i < depth; i++) {
        uint32_t t = base + (uint32_t)((i * 7u + user * 13u) % 4096u);
        buf[i] = t % (uint32_t)n_vocab;
    }
}

/* Chunked cold prefill of `depth` tokens; returns the last position's argmax
 * (the first decode token) or -1.  Untimed warm-up in every section. */
static int prefill_to_depth(pn_model *model, pn_state *st, const uint32_t *seed,
                            int depth, int chunk, int n_vocab)
{
    int last = -1;
    for (int off = 0; off < depth; off += chunk) {
        int n = depth - off < chunk ? depth - off : chunk;
        tm_tensor *out =
            pn_qwen35moe_forward_batch_cuda(model, seed + off, n, st, NULL);
        if (!out) return -1;
        if (off + n >= depth) last = argmax_f32(tm_data_f32(out), n_vocab);
        tm_free(out);
    }
    return last;
}

/* --- A. sequential single-token forward -------------------------------- */

/* Times `reps` ordinary single-token forwards from `tok`.  Advances st->pos by
 * reps.  Returns ms/forward, or -1 on failure. */
static double measure_sequential_step_ms(pn_model *model, pn_state *st,
                                         uint32_t tok, int reps, int n_vocab)
{
    tm_tensor *out = model->forward(model, tok, st);   /* warm-up, untimed */
    if (!out) return -1.0;
    tok = (uint32_t)argmax_f32(tm_data_f32(out), n_vocab);
    tm_free(out);
    double t0 = now_s();
    for (int i = 0; i < reps; i++) {
        out = model->forward(model, tok, st);
        if (!out) return -1.0;
        tok = (uint32_t)argmax_f32(tm_data_f32(out), n_vocab);
        tm_free(out);
    }
    return 1e3 * (now_s() - t0) / reps;
}

/* --- B. D-row speculative verify ---------------------------------------- */

/* Times `reps` rounds of verify(D) + accept(1) at the state's current depth.
 * accept(1) is the WORST-CASE commit (full SSM rollback from snapshot 0); it
 * advances st->pos by exactly 1 per round, so the depth drifts by `reps`.
 * Splits the round into *verify_ms and *commit_ms.  Returns 0, or -1. */
static int measure_verify_round_ms(pn_model *model, pn_state *st, int rows,
                                   int reps, int n_vocab,
                                   double *verify_ms, double *commit_ms)
{
    uint32_t block[GATE_MAX_ROWS];
    int32_t  argmax_out[GATE_MAX_ROWS];
    double verify_total = 0.0, commit_total = 0.0;
    for (int r = 0; r < rows; r++)
        block[r] = (uint32_t)((1000 + r * 37) % n_vocab);

    if (pn_qwen35moe_spec_verify(model, block, rows, st, argmax_out) != 0)
        return -1;                                     /* warm-up, untimed */
    if (pn_qwen35moe_spec_accept(model, st, 1) != 0)
        return -1;
    for (int i = 0; i < reps; i++) {
        double t0 = now_s();
        if (pn_qwen35moe_spec_verify(model, block, rows, st, argmax_out) != 0)
            return -1;
        double t1 = now_s();
        if (pn_qwen35moe_spec_accept(model, st, 1) != 0)
            return -1;
        double t2 = now_s();
        verify_total += t1 - t0;
        commit_total += t2 - t1;
    }
    *verify_ms = 1e3 * verify_total / reps;
    *commit_ms = 1e3 * commit_total / reps;
    return 0;
}

/* --- C. batched slots step ---------------------------------------------- */

/* Seeds K users to `depth`, binds them, and times `reps` decode_slots steps.
 * Returns ms/step (one step = K rows in one forward), or -1 on failure. */
static double measure_slots_step_ms(pn_model *model, int n_slots, int depth,
                                    int reps, int chunk, int n_vocab)
{
    pn_state *states[GATE_MAX_SLOTS] = { 0 };
    uint32_t  tokens[GATE_MAX_SLOTS];
    float    *logits[GATE_MAX_SLOTS];
    double    step_ms = -1.0;
    uint32_t *seed = malloc((size_t)depth * sizeof(uint32_t));
    if (!seed) return -1.0;

    for (int u = 0; u < n_slots; u++) {
        fill_seed_context(seed, depth, n_vocab, u);
        states[u] = pn_state_new(model);
        if (!states[u]) goto done;
        int first = prefill_to_depth(model, states[u], seed, depth, chunk,
                                     n_vocab);
        if (first < 0) goto done;
        if (pn_qwen35moe_decode_slot_bind(model, states[u], u, n_slots) != 0)
            goto done;
        tokens[u] = (uint32_t)first;
    }
    if (pn_qwen35moe_decode_slots(model, tokens, states, n_slots, logits) != 0)
        goto done;                                     /* warm-up, untimed */
    for (int u = 0; u < n_slots; u++)
        tokens[u] = (uint32_t)argmax_f32(logits[u], n_vocab);

    double t0 = now_s();
    for (int i = 0; i < reps; i++) {
        if (pn_qwen35moe_decode_slots(model, tokens, states, n_slots,
                                      logits) != 0)
            goto done;
        for (int u = 0; u < n_slots; u++)
            tokens[u] = (uint32_t)argmax_f32(logits[u], n_vocab);
    }
    step_ms = 1e3 * (now_s() - t0) / reps;

done:
    for (int u = 0; u < n_slots; u++)
        if (states[u]) pn_state_free(states[u]);
    free(seed);
    return step_ms;
}

/* --- driver -------------------------------------------------------------- */

/* Section A+B: sequential baseline then the verify sweep, on one session. */
static int run_single_session_sections(pn_model *model, int depth, int reps,
                                       int chunk, int n_vocab,
                                       const int *rows, int n_rows,
                                       double *seq_ms_out)
{
    uint32_t *seed = malloc((size_t)depth * sizeof(uint32_t));
    pn_state *st = pn_state_new(model);
    int rc = -1;
    if (!seed || !st) goto done;
    fill_seed_context(seed, depth, n_vocab, 0);
    int first = prefill_to_depth(model, st, seed, depth, chunk, n_vocab);
    if (first < 0) { fprintf(stderr, "prefill failed (depth=%d)\n", depth); goto done; }

    double seq_ms = measure_sequential_step_ms(model, st, (uint32_t)first,
                                               reps, n_vocab);
    if (seq_ms < 0) { fprintf(stderr, "sequential forward failed\n"); goto done; }
    *seq_ms_out = seq_ms;
    printf("\n[A] sequential single-token forward  depth~%d  reps=%d\n",
           depth, reps);
    printf("    %.2f ms/token   %.2f tok/s\n", seq_ms, 1e3 / seq_ms);

    printf("\n[B] speculative verify of D rows (one session, one forward)\n");
    printf("      D   pos   verify_ms  commit_ms   round_ms   ms/row   "
           "round/(D*seq)  breakeven_acc\n");
    for (int i = 0; i < n_rows; i++) {
        int d = rows[i];
        if (d > GATE_MAX_ROWS) continue;
        double verify_ms = 0, commit_ms = 0;
        int pos_before = st->pos;
        if (measure_verify_round_ms(model, st, d, reps, n_vocab,
                                    &verify_ms, &commit_ms) != 0) {
            printf("      %2d   %5d   FAILED\n", d, pos_before);
            continue;
        }
        double round_ms = verify_ms + commit_ms;
        printf("      %2d   %5d   %8.2f   %8.2f   %8.2f  %7.2f   %10.3f   "
               "%10.2f\n",
               d, pos_before, verify_ms, commit_ms, round_ms, round_ms / d,
               round_ms / (d * seq_ms), round_ms / seq_ms);
    }
    rc = 0;
done:
    if (st) pn_state_free(st);
    free(seed);
    return rc;
}

int main(int argc, char **argv)
{
    if (argc < 2) {
        fprintf(stderr,
                "usage: %s <model.gguf> [depth] [reps] [chunk]\n", argv[0]);
        return 1;
    }
    int depth = argc > 2 ? atoi(argv[2]) : 1024;
    int reps  = argc > 3 ? atoi(argv[3]) : 20;
    int chunk = argc > 4 ? atoi(argv[4]) : 256;
    if (depth < 8) depth = 8;
    if (reps < 1) reps = 1;

    int rows[GATE_MAX_POINTS], slots[GATE_MAX_POINTS];
    const char *rows_env = getenv("QWEN35_GATE_ROWS");
    const char *slots_env = getenv("QWEN35_GATE_SLOTS");
    int n_rows = parse_int_list(rows_env ? rows_env : "1,2,4,8", rows,
                                GATE_MAX_POINTS);
    int n_slots = parse_int_list(slots_env ? slots_env : "1,2,4,8", slots,
                                 GATE_MAX_POINTS);

    pn_model *model = pn_load(argv[1]);
    if (!model) { fprintf(stderr, "load failed: %s\n", argv[1]); return 1; }
    int n_vocab = model->config.n_vocab;
    printf("phase 9.1 verify gate: %s\n", argv[1]);
    printf("  n_vocab=%d n_layer=%d depth=%d reps=%d chunk=%d\n",
           n_vocab, model->config.n_layer, depth, reps, chunk);

    double seq_ms = 0;
    if (run_single_session_sections(model, depth, reps, chunk, n_vocab,
                                    rows, n_rows, &seq_ms) != 0) {
        pn_model_free(model);
        return 1;
    }

    if (n_slots > 0) {
        printf("\n[C] batched slots step: K rows of K sessions in one forward\n");
        printf("       K   step_ms   ms/row   step/(K*seq)   agg_tok/s\n");
        for (int i = 0; i < n_slots; i++) {
            int k = slots[i];
            if (k > GATE_MAX_SLOTS) continue;
            double step_ms = measure_slots_step_ms(model, k, depth, reps,
                                                   chunk, n_vocab);
            if (step_ms < 0) { printf("      %2d   FAILED\n", k); continue; }
            printf("      %2d  %8.2f  %7.2f   %10.3f   %9.2f\n",
                   k, step_ms, step_ms / k, step_ms / (k * seq_ms),
                   1e3 * k / step_ms);
        }
    }

    printf("\ndone\n");
    fflush(stdout);
    if (!getenv("QWEN35_EXIT_CLEAN"))
        _Exit(0);
    pn_model_free(model);
    return 0;
}
#endif
