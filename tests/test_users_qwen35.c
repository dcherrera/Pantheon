/* SPDX-License-Identifier: AGPL-3.0-or-later */
/* Copyright (C) 2026 David (Corey) Herrera */
/* Multi-user decode gate + bench for qwen35moe on a K80 host.
 *
 * GATE: K users with DIFFERENT prompts (user 0 = the 12-token reference
 * sequence; the others vary in content AND length so per-user positions
 * differ). Phase 1 decodes each user ALONE (batched prefill + single-token
 * greedy decode — the reference). Phase 2 decodes all K users TOGETHER
 * through pn_qwen35moe_decode_slots (one forward pass per step). Every
 * user's batched token stream must be IDENTICAL to its alone stream —
 * batching changes throughput, not output.
 *
 * BENCH (QWEN35_USERS_BENCH="1,2,4,8"): after the gate, the batch-K
 * throughput curve in the SAME process (one model load): for each K,
 * prefill+bind K users, time n_gen batched greedy steps, report aggregate
 * tok/s, per-user tok/s, and cost(K)/cost(1) (per-token step cost vs the
 * K=1 batched step). Also times the single-token fast path (graphs+fused)
 * as the serving baseline.
 *
 * Build (CUDA host): make USE_CUDA=1 tests/test_users_qwen35
 * Run:  QWEN35_GPU_SPLIT_LAYER=20 QWEN35_GPU_DIE_A=0 QWEN35_GPU_DIE_B=1 \
 *         QWEN35_FUSE=0x37 ./tests/test_users_qwen35 <model.gguf> [K] [n_gen]
 * QWEN35_USERS_DIGEST=1 additionally folds an FNV-1a over every batched
 * decode step's full per-user logits and prints it, so two different BUILDS
 * can be compared bit-exactly (the gate itself only compares two paths inside
 * one process, and an argmax stream can agree while the floats move). It
 * costs ~1% of a step, so leave it off for timing arms.
 *
 * QWEN35_FUSE=0x37 makes the single-token attn path float-identical to the
 * batched kernels (the fused qknorm+rope contraction differs by ~1 ulp —
 * same caveat as test_spec_qwen35); the strict gate runs
 * under it. NOT part of the test suite (needs the model + a CUDA GPU). */
#include "../include/pantheon.h"
#include "../src/core/engine.h"
#include "../src/models/qwen35moe.h"
#include "tensor/tensor.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

#ifndef USE_CUDA
int main(void)
{
    fprintf(stderr, "built without USE_CUDA\n");
    return 1;
}
#else

#define USERS_MAX      16
#define USERS_MAX_GEN  512
#define USERS_PROMPT   12

static double now_s(void)
{
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (double)ts.tv_sec + (double)ts.tv_nsec * 1e-9;
}

static int argmax(const float *x, int n)
{
    int best = 0;
    for (int i = 1; i < n; i++)
        if (x[i] > x[best]) best = i;
    return best;
}

/* FNV-1a, folded across every batched decode step's full logit vector. The
 * gate below compares batched vs alone INSIDE one process; this digest is
 * what makes two DIFFERENT builds comparable — an argmax stream can agree
 * while the floats moved, so a kernel change claiming bit-exactness on the
 * batched decode path has to reproduce this number. */
static unsigned long long fnv1a(unsigned long long h, const void *buf,
                                size_t len)
{
    const unsigned char *p = (const unsigned char *)buf;
    for (size_t i = 0; i < len; i++) {
        h ^= p[i];
        h *= 1099511628211ULL;
    }
    return h;
}

/* User u's prompt: user 0 is the exact 12-token reference sequence; the
 * others shift the token ids and shorten the length so every user decodes
 * at a DIFFERENT position with DIFFERENT content. */
static int user_prompt(int u, uint32_t *out)
{
    static const uint32_t ref_toks[USERS_PROMPT] = {
        9419, 11, 100, 200, 300, 400, 500, 600, 700, 800, 900, 1000
    };
    int len = USERS_PROMPT - (u % 4);
    for (int i = 0; i < len; i++)
        out[i] = ref_toks[i] + (uint32_t)(u * 17);
    return len;
}

/* Batched prefill of one user's prompt; returns the seed token (greedy
 * argmax of the last position) or -1. */
static int prefill_user(pn_model *model, pn_state *st, const uint32_t *prompt,
                        int len, int n_vocab)
{
    tm_tensor *out = pn_qwen35moe_forward_batch_cuda(model, prompt, len, st,
                                                     NULL);
    if (!out)
        return -1;
    int seed = argmax(tm_data_f32(out), n_vocab);
    tm_free(out);
    return seed;
}

/* Reference: decode n_gen greedy tokens for one user ALONE via the
 * single-token forward. stream[0] is the prefill argmax. Returns 0. */
static int decode_alone(pn_model *model, const uint32_t *prompt, int len,
                        int n_gen, int n_vocab, uint32_t *stream,
                        double *dt_out)
{
    pn_state *st = pn_state_new(model);
    if (!st)
        return -1;
    int seed = prefill_user(model, st, prompt, len, n_vocab);
    if (seed < 0) { pn_state_free(st); return -1; }
    stream[0] = (uint32_t)seed;
    double t0 = now_s();
    uint32_t tok = (uint32_t)seed;
    for (int i = 1; i < n_gen; i++) {
        tm_tensor *out = model->forward(model, tok, st);
        if (!out) { pn_state_free(st); return -1; }
        tok = (uint32_t)argmax(tm_data_f32(out), n_vocab);
        tm_free(out);
        stream[i] = tok;
    }
    if (dt_out)
        *dt_out = now_s() - t0;
    pn_state_free(st);
    return 0;
}

/* Batched: prefill + bind K users, then decode n_gen greedy tokens for all
 * of them together (one pass per step). streams[u][i] = user u's i-th
 * token. Returns 0; *dt_out = the timed decode loop (n_gen - 1 steps). */
static int decode_batched(pn_model *model, int K, int n_gen, int n_vocab,
                          uint32_t (*streams)[USERS_MAX_GEN], double *dt_out,
                          unsigned long long *digest_out)
{
    pn_state *sts[USERS_MAX] = { 0 };
    uint32_t toks[USERS_MAX];
    float *logits[USERS_MAX];
    uint32_t prompt[USERS_PROMPT];
    int rc = -1;
    /* QWEN35_USERS_ROWS=scatter: bind user u to persistent row 2K-1-2u of 2K
     * rows (reversed, with a gap between every pair) and step through the row
     * map — the layout the serve's fixed session blocks produce. Must be
     * bit-identical to the identity layout. */
    const char *rows_env = getenv("QWEN35_USERS_ROWS");
    int scatter = rows_env && strcmp(rows_env, "scatter") == 0;
    int rows[USERS_MAX];
    int n_rows = scatter ? 2 * K : K;
    for (int u = 0; u < K; u++)
        rows[u] = scatter ? 2 * K - 1 - 2 * u : u;

    for (int u = 0; u < K; u++) {
        int len = user_prompt(u, prompt);
        sts[u] = pn_state_new(model);
        if (!sts[u])
            goto cleanup;
        int seed = prefill_user(model, sts[u], prompt, len, n_vocab);
        if (seed < 0)
            goto cleanup;
        /* Bind IMMEDIATELY after this user's prefill: the live GPU SSM
         * state is this user's, and the next user's prefill replaces it. */
        if (pn_qwen35moe_decode_slot_bind(model, sts[u], rows[u], n_rows) != 0) {
            fprintf(stderr, "bind failed (user %d)\n", u);
            goto cleanup;
        }
        toks[u] = (uint32_t)seed;
        streams[u][0] = (uint32_t)seed;
    }

    /* Hashing ~1 MB per user per step is ~1% of a step, so the digest is
     * opt-in: run the correctness arm with QWEN35_USERS_DIGEST=1 and the
     * timing arms without it. */
    unsigned long long digest = 1469598103934665603ULL;
    int want_digest = digest_out && getenv("QWEN35_USERS_DIGEST") != NULL;
    double t0 = now_s();
    for (int i = 1; i < n_gen; i++) {
        if (pn_qwen35moe_decode_slots_rows(model, toks, sts, rows, K,
                                           logits) != 0) {
            fprintf(stderr, "decode_slots failed @step %d\n", i);
            goto cleanup;
        }
        for (int u = 0; u < K; u++) {
            toks[u] = (uint32_t)argmax(logits[u], n_vocab);
            streams[u][i] = toks[u];
            if (want_digest)
                digest = fnv1a(digest, logits[u],
                               (size_t)n_vocab * sizeof(float));
        }
    }
    if (dt_out)
        *dt_out = now_s() - t0;
    if (digest_out)
        *digest_out = want_digest ? digest : 0;
    rc = 0;

cleanup:
    for (int u = 0; u < K; u++)
        if (sts[u])
            pn_state_free(sts[u]);
    return rc;
}

int main(int argc, char **argv)
{
    if (argc < 2) {
        fprintf(stderr, "usage: %s <model.gguf> [K] [n_gen]\n", argv[0]);
        return 1;
    }

    pn_model *model = pn_load(argv[1]);
    if (!model) { fprintf(stderr, "load failed\n"); return 1; }
    int n_vocab = model->config.n_vocab;

    int K = argc > 2 ? atoi(argv[2]) : 4;
    if (K < 1) K = 1;
    if (K > USERS_MAX) K = USERS_MAX;
    int n_gen = argc > 3 ? atoi(argv[3]) : 32;
    if (n_gen < 2) n_gen = 2;
    if (n_gen > USERS_MAX_GEN) n_gen = USERS_MAX_GEN;

    static uint32_t ref_streams[USERS_MAX][USERS_MAX_GEN];
    static uint32_t bat_streams[USERS_MAX][USERS_MAX_GEN];
    uint32_t prompt[USERS_PROMPT];
    int failed = 0;

    const char *gate_env = getenv("QWEN35_USERS_GATE");
    int run_gate = !(gate_env && gate_env[0] == '0');

    if (run_gate) {
        fprintf(stderr, "users gate: K=%d n_gen=%d rows=%s\n", K, n_gen,
                getenv("QWEN35_USERS_ROWS") ? getenv("QWEN35_USERS_ROWS")
                                            : "identity");

        /* ---- Phase 1: each user decodes ALONE (the reference). ---- */
        double dt_alone_sum = 0.0;
        for (int u = 0; u < K; u++) {
            int len = user_prompt(u, prompt);
            double dt = 0.0;
            if (decode_alone(model, prompt, len, n_gen, n_vocab,
                             ref_streams[u], &dt) != 0) {
                fprintf(stderr, "alone decode failed (user %d)\n", u);
                return 1;
            }
            dt_alone_sum += dt;
            fprintf(stderr, "alone  u=%d len=%d %.2f tok/s\n", u, len,
                    (n_gen - 1) / dt);
        }

        /* ---- Phase 2: all K users batched. ---- */
        double dt_bat = 0.0;
        unsigned long long dg = 0;
        if (decode_batched(model, K, n_gen, n_vocab, bat_streams,
                           &dt_bat, &dg) != 0)
            return 1;
        fprintf(stderr,
                "batched K=%d: %.3fs for %d steps | %.2f ms/step | "
                "aggregate %.2f tok/s | per-user %.2f tok/s\n",
                K, dt_bat, n_gen - 1, 1000.0 * dt_bat / (n_gen - 1),
                (double)K * (n_gen - 1) / dt_bat, (n_gen - 1) / dt_bat);
        if (dg)
            fprintf(stderr, "batched K=%d logits digest: %016llx\n", K, dg);

        /* ---- Gate: per-user streams identical. ---- */
        for (int u = 0; u < K; u++) {
            int first_diff = -1;
            for (int i = 0; i < n_gen; i++) {
                if (ref_streams[u][i] != bat_streams[u][i]) {
                    first_diff = i;
                    break;
                }
            }
            fprintf(stderr, "user %d alone  :", u);
            for (int i = 0; i < n_gen; i++)
                fprintf(stderr, " %u", ref_streams[u][i]);
            fprintf(stderr, "\nuser %d batched:", u);
            for (int i = 0; i < n_gen; i++)
                fprintf(stderr, " %u", bat_streams[u][i]);
            fprintf(stderr, "\n");
            if (first_diff < 0) {
                fprintf(stderr, "user %d: IDENTICAL (%d tokens)\n", u, n_gen);
            } else {
                fprintf(stderr,
                        "user %d: DIVERGED at %d (alone=%u batched=%u)\n",
                        u, first_diff, ref_streams[u][first_diff],
                        bat_streams[u][first_diff]);
                failed = 1;
            }
        }
        fprintf(stderr, "GATE: %s\n", failed ? "FAIL" : "BIT-EXACT PASS");
    }

    /* ---- Bench: the batch-K throughput curve, one process. ---- */
    const char *bench = getenv("QWEN35_USERS_BENCH");
    if (bench && !failed) {
        char list[128];
        strncpy(list, bench, sizeof(list) - 1);
        list[sizeof(list) - 1] = '\0';

        /* Single-token fast-path baseline (graphs + fused): the K=1
         * serving number the curve is judged against. */
        double dt1 = 0.0;
        int len0 = user_prompt(0, prompt);
        if (decode_alone(model, prompt, len0, n_gen, n_vocab,
                         ref_streams[0], &dt1) != 0)
            return 1;
        double base_tps = (n_gen - 1) / dt1;
        fprintf(stderr, "BENCH single-token path: %.2f tok/s (baseline)\n",
                base_tps);

        double cost1 = 0.0;   /* ms per step at K=1 (batched path) */
        for (char *tk = strtok(list, ","); tk; tk = strtok(NULL, ",")) {
            int k = atoi(tk);
            if (k < 1 || k > USERS_MAX)
                continue;
            double dt = 0.0;
            if (decode_batched(model, k, n_gen, n_vocab, bat_streams,
                               &dt, NULL) != 0)
                return 1;
            double step_ms = 1000.0 * dt / (n_gen - 1);
            if (k == 1 || cost1 == 0.0)
                cost1 = (k == 1) ? step_ms : cost1;
            double agg = (double)k * (n_gen - 1) / dt;
            fprintf(stderr,
                    "BENCH K=%d: %.2f ms/step | aggregate %.2f tok/s | "
                    "per-user %.2f tok/s | vs baseline %.2fx",
                    k, step_ms, agg, (n_gen - 1) / dt, agg / base_tps);
            if (cost1 > 0.0)
                fprintf(stderr, " | cost(K)/cost(1) %.2f (per-token %.2f)",
                        step_ms / cost1, step_ms / cost1 / k);
            fprintf(stderr, "\n");
        }
    }

    fflush(stdout); fflush(stderr);
    if (!getenv("QWEN35_EXIT_CLEAN"))
        _Exit(failed);
    return failed;
}
#endif
