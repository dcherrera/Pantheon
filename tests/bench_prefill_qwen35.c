/* SPDX-License-Identifier: AGPL-3.0-or-later */
/* Copyright (C) 2026 David (Corey) Herrera */
/* Prefill (TTFT) bench for qwen35moe-family models on CUDA.
 *
 * Prefills ONE deterministic prompt of n tokens from a fresh state and reports
 * wall time, tok/s and an FNV-1a hash + argmax of the last position's logits,
 * so two builds (or two paths) can be compared for speed AND bit-exactness.
 *
 * Paths (PREFILL_MODES, default "seq,wide"):
 *   seq   the chunked batched loop (pn_qwen35moe_forward_batch_cuda per chunk,
 *         dies in sequence) — what bench_mtp_e2e does. QWEN35_PROF=1 prints the
 *         BPROF phase split per chunk on this path.
 *   wide  pn_qwen35moe_prefill_range_pipelined_cuda — the serve's default path
 *         (every die on a different chunk).
 *
 * Append mode (PREFILL_DEPTH=D): the shape of an agent tool turn. Per path, a
 * fresh state is prefilled with D prompt tokens (untimed, same path), then the
 * PREFILL_APPENDS ladder (default "512,1024,2048") is appended to that state
 * one after another and each append is timed and hashed on its own, so its
 * depth is D plus the appends before it. n_tokens is ignored in this mode.
 *
 * Build (CUDA host): make USE_CUDA=1 tests/bench_prefill_qwen35
 * Run: QWEN35_GPU_DIES=0,1,2,3 QWEN35_KV_Q4=1 QWEN35_MAX_SEQ=9216 \
 *        ./tests/bench_prefill_qwen35 <model.gguf> <n_tokens> [chunk]
 *      PREFILL_DEPTH=16384 PREFILL_APPENDS=512,1024,2048 QWEN35_MAX_SEQ=20480 \
 *        ./tests/bench_prefill_qwen35 <model.gguf> 0 [chunk]
 * NOT part of the test suite (needs the model + a CUDA GPU). */
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

static double now_s(void)
{
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (double)ts.tv_sec + (double)ts.tv_nsec * 1e-9;
}

static uint64_t fnv1a(const void *p, size_t n)
{
    const unsigned char *b = p;
    uint64_t h = 1469598103934665603ULL;
    for (size_t i = 0; i < n; i++) {
        h ^= b[i];
        h *= 1099511628211ULL;
    }
    return h;
}

static int argmax_f32(const float *x, int n)
{
    int best = 0;
    for (int i = 1; i < n; i++)
        if (x[i] > x[best]) best = i;
    return best;
}

/* Same seed context as bench_spec_verify_gate (user 0). */
static void fill_prompt(uint32_t *buf, int n, int n_vocab)
{
    for (int i = 0; i < n; i++)
        buf[i] = (100u + (uint32_t)((i * 7u) % 4096u)) % (uint32_t)n_vocab;
}

static void report(const char *mode, int depth, int n, int chunk, double dt,
                   tm_tensor *out, int n_vocab)
{
    const float *l = tm_data_f32(out);
    printf("PREFILL mode=%s depth=%d n=%d chunk=%d ms=%.0f tok/s=%.2f "
           "hash=%016llx argmax=%d\n", mode, depth, n, chunk, dt * 1e3, n / dt,
           (unsigned long long)fnv1a(l, (size_t)n_vocab * sizeof(float)),
           argmax_f32(l, n_vocab));
    fflush(stdout);
}

/* Prefill toks[0, n) onto st (which already holds `depth` positions) by one
 * path. Returns the last position's logits (caller frees) or NULL. */
static tm_tensor *prefill_range(pn_model *model, pn_state *st, int wide,
                                const uint32_t *toks, int n, int chunk)
{
    if (wide) {
        tm_tensor *out = NULL;
        int rc = pn_qwen35moe_prefill_range_pipelined_cuda(model, toks, n, st,
                                                           chunk, NULL, NULL,
                                                           &out);
        if (rc != 0 || !out) {
            fprintf(stderr, "wide prefill refused/failed rc=%d\n", rc);
            tm_free(out);
            return NULL;
        }
        return out;
    }
    tm_tensor *last = NULL;
    for (int off = 0; off < n; off += chunk) {
        int m = n - off < chunk ? n - off : chunk;
        tm_tensor *out = pn_qwen35moe_forward_batch_cuda(model, toks + off, m,
                                                         st, NULL);
        if (!out) {
            fprintf(stderr, "seq prefill failed at %d\n", off);
            tm_free(last);
            return NULL;
        }
        tm_free(last);
        last = out;
    }
    return last;
}

/* Append mode: prefill `depth` tokens untimed, then time each append of the
 * comma-separated ladder in turn. toks holds depth + sum(ladder) tokens. */
static int run_append(pn_model *model, const char *mode, int wide,
                      const uint32_t *toks, int depth, const char *ladder,
                      int chunk)
{
    pn_state *st = pn_state_new(model);
    if (!st)
        return -1;
    tm_tensor *out = prefill_range(model, st, wide, toks, depth, chunk);
    int rc = out ? 0 : -1;
    tm_free(out);
    int at = depth;
    for (const char *p = ladder; rc == 0 && p && *p;) {
        int n = atoi(p);
        if (n <= 0)
            break;
        double t0 = now_s();
        out = prefill_range(model, st, wide, toks + at, n, chunk);
        double dt = now_s() - t0;
        if (!out) {
            rc = -1;
            break;
        }
        report(mode, at, n, chunk, dt, out, model->config.n_vocab);
        tm_free(out);
        at += n;
        p = strchr(p, ',');
        if (p)
            p++;
    }
    pn_state_free(st);
    return rc;
}

/* Total tokens a ladder appends. */
static int ladder_tokens(const char *ladder)
{
    int sum = 0;
    for (const char *p = ladder; p && *p;) {
        sum += atoi(p) > 0 ? atoi(p) : 0;
        p = strchr(p, ',');
        if (p)
            p++;
    }
    return sum;
}

static int run_fresh(pn_model *model, const char *mode, int wide,
                     const uint32_t *toks, int n, int chunk)
{
    pn_state *st = pn_state_new(model);
    if (!st)
        return -1;
    double t0 = now_s();
    tm_tensor *out = prefill_range(model, st, wide, toks, n, chunk);
    double dt = now_s() - t0;
    if (out)
        report(mode, 0, n, chunk, dt, out, model->config.n_vocab);
    tm_free(out);
    pn_state_free(st);
    return out ? 0 : -1;
}

int main(int argc, char **argv)
{
    if (argc < 3) {
        fprintf(stderr, "usage: %s <model.gguf> <n_tokens> [chunk]\n",
                argv[0]);
        return 1;
    }
    int n = atoi(argv[2]);
    int chunk = argc > 3 ? atoi(argv[3]) : 128;
    const char *modes = getenv("PREFILL_MODES");
    if (!modes)
        modes = "seq,wide";
    const char *depth_env = getenv("PREFILL_DEPTH");
    int depth = depth_env ? atoi(depth_env) : 0;
    const char *ladder = getenv("PREFILL_APPENDS");
    if (!ladder)
        ladder = "512,1024,2048";
    if (depth > 0)
        n = depth + ladder_tokens(ladder);
    if (n < 1 || chunk < 1)
        return 1;

    pn_model *model = pn_load(argv[1]);
    if (!model) {
        fprintf(stderr, "load failed: %s\n", argv[1]);
        return 1;
    }
    uint32_t *toks = malloc((size_t)n * sizeof(uint32_t));
    if (!toks)
        return 1;
    fill_prompt(toks, n, model->config.n_vocab);

    int rc = 0;
    for (const char *p = modes; *p && rc == 0;) {
        int wide = !strncmp(p, "wide", 4);
        const char *mode = wide ? "wide" : "seq";
        if (!wide && strncmp(p, "seq", 3))
            break;
        rc = depth > 0 ? run_append(model, mode, wide, toks, depth, ladder, chunk)
                       : run_fresh(model, mode, wide, toks, n, chunk);
        p = strchr(p, ',');
        if (!p)
            break;
        p++;
    }
    printf("done rc=%d\n", rc);
    fflush(stdout);
    free(toks);
    if (!getenv("QWEN35_EXIT_CLEAN"))
        _Exit(rc ? 1 : 0);
    pn_model_free(model);
    return rc ? 1 : 0;
}
#endif
