/* SPDX-License-Identifier: AGPL-3.0-or-later */
/* Copyright (C) 2026 David (Corey) Herrera */
/* Prefix-cache gate + latency bench for qwen35moe on a K80 host.
 *
 * The agentic-turn win: across turns the context is mostly unchanged (a small
 * delta appended to a fixed prefix). With caching ON we keep the session's
 * model state and prefill ONLY the delta; with caching OFF we re-prefill the
 * whole context every turn. The two MUST generate identical replies.
 *
 * Phase 1 (single session, THE win): a ~4k-token base context, then several
 * turns each appending a ~200-token delta and greedily generating a short
 * reply. Run twice — cached (pn_prefix_cache: delta-prefill only) vs uncached
 * (fresh pn_state, full re-prefill each turn) — over identical per-turn
 * contexts, comparing the generated tokens argmax-for-argmax and reporting the
 * per-turn prefill latency of each.
 *
 * Phase 2 (multi-session save/restore): two sessions interleaved through ONE
 * cache (they share the GPU's SSM buffers, so each acquire snapshots the other
 * off-GPU and restores this one). Each session's interleaved generation must
 * match that session decoded alone (uncached).
 *
 * Build (CUDA host): make USE_CUDA=1 tests/test_prefix_cache_qwen35
 * Run:  QWEN35_GPU_SPLIT_LAYER=20 QWEN35_GPU_DIE_A=0 QWEN35_GPU_DIE_B=1 \
 *         QWEN35_FUSE=0x37 ./tests/test_prefix_cache_qwen35 <model.gguf> \
 *         [base] [delta] [turns] [gen]
 *   base=4000 delta=200 turns=4 gen=8 by default.
 * QWEN35_FUSE=0x37 makes the single-token decode path float-identical to the
 * batched-prefill kernels (~1-ulp fused qknorm+rope contraction), the same
 * convention as flash_spec_gate.sh / test_spec_qwen35 — the strict argmax gate
 * runs under it.
 * NOT part of the CPU test suite (needs the model + a CUDA GPU). */
#include "../include/pantheon.h"
#include "../src/core/engine.h"
#include "../src/core/prefix_cache.h"
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

/* Deterministic in-vocab pseudo-random token stream (reproducible contexts). */
static void fill_tokens(uint32_t *dst, int n, uint32_t seed, int n_vocab)
{
    uint32_t s = seed * 2654435761u + 12345u;
    for (int i = 0; i < n; i++) {
        s = s * 1664525u + 1013904223u;
        dst[i] = (s >> 8) % (uint32_t)n_vocab;
    }
}

/* Batched-prefill tokens[start..n) from the state's current pos, in chunks of
 * PREFILL_CHUNK (keeps the batch scratch bounded for multi-thousand-token
 * contexts; chunking is argmax-identical — the KV/SSM state is built
 * incrementally exactly as one big batch would). Returns the last-position
 * logits tensor (caller frees), or NULL. */
static int prefill_chunk(void)
{
    const char *v = getenv("QWEN35_PC_CHUNK");
    int c = v ? atoi(v) : 256;
    return c > 0 ? c : 256;
}

static tm_tensor *prefill_delta(pn_model *model, pn_state *st,
                                const uint32_t *tokens, int start, int n)
{
    if (n <= start)
        return NULL;
    int chunk = prefill_chunk();
    tm_tensor *o = NULL;
    for (int p = start; p < n; p += chunk) {
        int m = n - p < chunk ? n - p : chunk;
        if (o) tm_free(o);
        o = pn_qwen35moe_forward_batch_cuda(model, tokens + p, m, st, NULL);
        if (!o) return NULL;
    }
    return o;
}

/* Greedy-generate `gen` tokens from `first` logits (the prefill's last-position
 * logits), writing them to out and advancing the state. The first token comes
 * straight from `first`; the next gen-1 from gen-1 forwards; then ONE more
 * forward feeds the final token so the state ends at pos += gen with every
 * generated token committed to the KV/SSM state (state->pos == committed
 * context length) — exactly a real serving turn, and what lets the next turn
 * delta-prefill only the truly new tokens. Returns 0 / -1. */
static int generate(pn_model *model, pn_state *st, tm_tensor *first,
                    uint32_t *out, int gen, int n_vocab)
{
    uint32_t tok = (uint32_t)argmax(tm_data_f32(first), n_vocab);
    out[0] = tok;
    for (int i = 1; i < gen; i++) {
        tm_tensor *o = model->forward(model, tok, st);
        if (!o) return -1;
        tok = (uint32_t)argmax(tm_data_f32(o), n_vocab);
        out[i] = tok;
        tm_free(o);
    }
    tm_tensor *last = model->forward(model, out[gen - 1], st); /* commit last */
    if (!last) return -1;
    tm_free(last);
    return 0;
}

int main(int argc, char **argv)
{
    if (argc < 2) {
        fprintf(stderr, "usage: %s <model.gguf> [base] [delta] [turns] [gen]\n",
                argv[0]);
        return 1;
    }
    pn_model *model = pn_load(argv[1]);
    if (!model) { fprintf(stderr, "load failed\n"); return 1; }
    int n_vocab = model->config.n_vocab;

    int base  = argc > 2 ? atoi(argv[2]) : 4000;
    int delta = argc > 3 ? atoi(argv[3]) : 200;
    int turns = argc > 4 ? atoi(argv[4]) : 4;
    int gen   = argc > 5 ? atoi(argv[5]) : 8;
    if (base < 1 || delta < 1 || turns < 1 || gen < 1) {
        fprintf(stderr, "bad params\n"); return 1;
    }

    int max_ctx = base + turns * (delta + gen) + 8;
    uint32_t *ctx      = malloc((size_t)max_ctx * sizeof(uint32_t));
    /* per-turn snapshot of the context fed before generation, and each run's
     * generated reply. */
    uint32_t **snap    = malloc((size_t)turns * sizeof(uint32_t *));
    int       *snap_n  = malloc((size_t)turns * sizeof(int));
    uint32_t  *gc      = malloc((size_t)turns * gen * sizeof(uint32_t)); /* cached */
    uint32_t  *gu      = malloc((size_t)turns * gen * sizeof(uint32_t)); /* uncached */
    double    *t_cache = calloc((size_t)turns, sizeof(double));
    double    *t_uncab = calloc((size_t)turns, sizeof(double));
    int       *d_delta = calloc((size_t)turns, sizeof(int));
    if (!ctx || !snap || !snap_n || !gc || !gu || !t_cache || !t_uncab || !d_delta) {
        fprintf(stderr, "oom\n"); return 1;
    }

    fprintf(stderr, "prefix-cache gate: base=%d delta=%d turns=%d gen=%d\n",
            base, delta, turns, gen);

    /* ================= Phase 1a: CACHED run ================= */
    pn_prefix_cache *cache = pn_prefix_cache_new(model, 4);
    if (!cache) { fprintf(stderr, "cache new failed\n"); return 1; }

    int n_ctx = 0;
    fill_tokens(ctx, base, 1u, n_vocab);       /* the fixed base prefix */
    n_ctx = base;
    for (int t = 0; t < turns; t++) {
        if (t > 0) {                            /* append this turn's delta */
            fill_tokens(ctx + n_ctx, delta, 100u + (uint32_t)t, n_vocab);
            n_ctx += delta;
        }
        /* snapshot the context fed before generation (for the reference run) */
        snap_n[t] = n_ctx;
        snap[t] = malloc((size_t)n_ctx * sizeof(uint32_t));
        memcpy(snap[t], ctx, (size_t)n_ctx * sizeof(uint32_t));

        int ds = 0;
        pn_state *st = pn_prefix_cache_acquire(cache, "agent", ctx, n_ctx, &ds);
        if (!st) { fprintf(stderr, "acquire failed t=%d\n", t); return 1; }
        d_delta[t] = n_ctx - ds;

        double t0 = now_s();
        tm_tensor *o = prefill_delta(model, st, ctx, ds, n_ctx);
        t_cache[t] = now_s() - t0;
        if (!o) { fprintf(stderr, "cached prefill failed t=%d\n", t); return 1; }

        if (generate(model, st, o, gc + (size_t)t * gen, gen, n_vocab) != 0) {
            fprintf(stderr, "cached gen failed t=%d\n", t); return 1;
        }
        tm_free(o);
        /* commit the reply into the running context + the session history */
        memcpy(ctx + n_ctx, gc + (size_t)t * gen, (size_t)gen * sizeof(uint32_t));
        n_ctx += gen;
        if (pn_prefix_cache_commit(cache, "agent", ctx, n_ctx) != 0) {
            fprintf(stderr, "commit failed t=%d\n", t); return 1;
        }
    }
    pn_prefix_cache_free(cache);

    /* ================= Phase 1b: UNCACHED reference ================= */
    for (int t = 0; t < turns; t++) {
        pn_state *st = pn_state_new(model);
        if (!st) { fprintf(stderr, "state failed t=%d\n", t); return 1; }
        pn_state_reset(st);
        double t0 = now_s();
        tm_tensor *o = prefill_delta(model, st, snap[t], 0, snap_n[t]);
        t_uncab[t] = now_s() - t0;
        if (!o) { fprintf(stderr, "uncached prefill failed t=%d\n", t); return 1; }
        if (generate(model, st, o, gu + (size_t)t * gen, gen, n_vocab) != 0) {
            fprintf(stderr, "uncached gen failed t=%d\n", t); return 1;
        }
        tm_free(o);
        pn_state_free(st);
    }

    /* ================= Gate + latency report ================= */
    int fail_turn = -1, fail_pos = -1;
    for (int t = 0; t < turns && fail_turn < 0; t++)
        for (int i = 0; i < gen; i++)
            if (gc[(size_t)t * gen + i] != gu[(size_t)t * gen + i]) {
                fail_turn = t; fail_pos = i; break;
            }

    fprintf(stderr, "\nper-turn prefill latency (cached delta vs uncached full):\n");
    for (int t = 0; t < turns; t++)
        fprintf(stderr,
                "  turn %d: ctx=%d delta=%d | cached %.3fs | uncached %.3fs | "
                "speedup %.1fx\n",
                t, snap_n[t], d_delta[t], t_cache[t], t_uncab[t],
                t_cache[t] > 0 ? t_uncab[t] / t_cache[t] : 0.0);

    if (fail_turn < 0)
        fprintf(stderr, "GATE 1 (single session): ARGMAX-EXACT PASS\n");
    else
        fprintf(stderr,
                "GATE 1: FAIL turn=%d pos=%d (cached=%u uncached=%u)\n",
                fail_turn, fail_pos,
                gc[(size_t)fail_turn * gen + fail_pos],
                gu[(size_t)fail_turn * gen + fail_pos]);

    /* ================= Phase 2: multi-session interleave ================= */
    /* Two sessions, short contexts, advanced turn-by-turn through ONE cache in
     * interleaved order — each acquire snapshots the other session off the
     * shared GPU SSM buffers and restores this one. Each session's interleaved
     * reply must match that session decoded alone. */
    int gate2_fail = 0;
    {
        const int b2 = 96, d2 = 24, T2 = 3;
        int cap2 = b2 + T2 * (d2 + gen) + 8;
        uint32_t *cx[2] = { malloc((size_t)cap2 * sizeof(uint32_t)),
                            malloc((size_t)cap2 * sizeof(uint32_t)) };
        int nx[2] = { 0, 0 };
        uint32_t *g2 = malloc((size_t)2 * T2 * gen * sizeof(uint32_t));
        uint32_t **sn = malloc((size_t)2 * T2 * sizeof(uint32_t *));
        int *snn = malloc((size_t)2 * T2 * sizeof(int));
        if (!cx[0] || !cx[1] || !g2 || !sn || !snn) { fprintf(stderr, "oom2\n"); return 1; }
        fill_tokens(cx[0], b2, 7u, n_vocab); nx[0] = b2;
        fill_tokens(cx[1], b2, 9u, n_vocab); nx[1] = b2;

        pn_prefix_cache *c2 = pn_prefix_cache_new(model, 2);
        if (!c2) { fprintf(stderr, "cache2 failed\n"); return 1; }
        const char *sid[2] = { "A", "B" };

        for (int t = 0; t < T2; t++) {
            for (int u = 0; u < 2; u++) {
                if (t > 0) {
                    fill_tokens(cx[u] + nx[u], d2, 200u + (uint32_t)(t * 2 + u),
                                n_vocab);
                    nx[u] += d2;
                }
                int si = (t * 2 + u);
                snn[si] = nx[u];
                sn[si] = malloc((size_t)nx[u] * sizeof(uint32_t));
                memcpy(sn[si], cx[u], (size_t)nx[u] * sizeof(uint32_t));

                int ds = 0;
                pn_state *st = pn_prefix_cache_acquire(c2, sid[u], cx[u], nx[u],
                                                       &ds);
                if (!st) { fprintf(stderr, "acq2 failed\n"); return 1; }
                tm_tensor *o = prefill_delta(model, st, cx[u], ds, nx[u]);
                if (!o) { fprintf(stderr, "pf2 failed\n"); return 1; }
                if (generate(model, st, o, g2 + (size_t)si * gen, gen,
                             n_vocab) != 0) { fprintf(stderr, "gen2\n"); return 1; }
                tm_free(o);
                memcpy(cx[u] + nx[u], g2 + (size_t)si * gen,
                       (size_t)gen * sizeof(uint32_t));
                nx[u] += gen;
                pn_prefix_cache_commit(c2, sid[u], cx[u], nx[u]);
            }
        }
        pn_prefix_cache_free(c2);

        /* Reference: each snapshot decoded alone on a fresh state. */
        for (int si = 0; si < 2 * T2; si++) {
            pn_state *st = pn_state_new(model);
            pn_state_reset(st);
            tm_tensor *o = prefill_delta(model, st, sn[si], 0, snn[si]);
            uint32_t ref[512];
            if (gen > 512) { fprintf(stderr, "gen too large for phase 2\n"); return 1; }
            if (!o || generate(model, st, o, ref, gen, n_vocab) != 0) {
                fprintf(stderr, "ref2 failed si=%d\n", si); return 1;
            }
            tm_free(o);
            pn_state_free(st);
            for (int i = 0; i < gen; i++)
                if (ref[i] != g2[(size_t)si * gen + i]) {
                    fprintf(stderr,
                            "GATE 2: FAIL si=%d pos=%d (interleaved=%u alone=%u)\n",
                            si, i, g2[(size_t)si * gen + i], ref[i]);
                    gate2_fail = 1; break;
                }
            if (gate2_fail) break;
        }
        if (!gate2_fail)
            fprintf(stderr, "GATE 2 (multi-session save/restore): ARGMAX-EXACT PASS\n");
    }

    fflush(stderr);
    int ok = (fail_turn < 0) && !gate2_fail;
    if (!getenv("QWEN35_EXIT_CLEAN"))
        _Exit(ok ? 0 : 1);
    return ok ? 0 : 1;
}
#endif
