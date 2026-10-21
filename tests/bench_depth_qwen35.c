/* SPDX-License-Identifier: AGPL-3.0-or-later */
/* Copyright (C) 2026 David (Corey) Herrera */
/* Context-DEPTH decode bench for qwen35moe (Nex-N2-mini) on a K80 host.
 *
 * Part B of the coding-serving capacity bench.  Isolates STEADY-STATE per-user
 * DECODE throughput as the seeded context DEEPENS — because the 10 full-
 * attention layers grow O(context) while the 30 SSM layers stay context-flat,
 * so the streaming speed a user actually sees degrades with depth.  We want the
 * depth at which per-user decode falls below 8 tok/s (the "conversation" bar).
 *
 * Protocol (per K):
 *   For each user u in [0,K):
 *     1. COLD PREFILL a SEED context of `depth` tokens through the batched
 *        prefill path (chunked in CHUNK-token passes so the K80 never sees a
 *        huge single pass), advancing state->pos to `depth`.  UNTIMED (this is
 *        the warm-up; a real session pays this once, prefix-cached thereafter).
 *     2. bind the slot immediately (GPU SSM image → slot u; host mirror is
 *        stale after a GPU prefill).
 *   Then time ONLY the batched decode loop: n_gen steps of
 *   pn_qwen35moe_decode_slots over all K users at position `depth`+.  Report
 *   per-user decode tok/s ( (n_gen-1)/dt ) and aggregate ( K*(n_gen-1)/dt ).
 *
 * This is decode-at-depth, prefill EXCLUDED — the number a streaming user sees.
 *
 * Build (CUDA host): make USE_CUDA=1 tests/bench_depth_qwen35
 * Run: QWEN35_GPU_SPLIT_LAYER=20 QWEN35_GPU_DIE_A=0 QWEN35_GPU_DIE_B=1 \
 *        QWEN35_FUSE=0x37 \
 *        ./tests/bench_depth_qwen35 <model.gguf> <K> <depth> [n_gen] [chunk]
 * Env: QWEN35_DEPTHS="1024,8192,32768,131072" runs the whole sweep in one
 *      model load at the given K (overrides the <depth> arg's single point).
 *      QWEN35_DEPTH_DIGEST=1 folds an FNV-1a over every decode step's full
 *      per-user logits and prints it with the greedy token stream, so two
 *      BUILDS can be diffed for bit-exactness at depth (costs a host pass over
 *      the logits per step; leave it off for timing arms).
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

#define USERS_MAX 16

static double now_s(void)
{
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (double)ts.tv_sec + (double)ts.tv_nsec * 1e-9;
}

/* FNV-1a over raw float bytes: any bit change in any logit moves it. */
static uint64_t fnv1a(uint64_t h, const void *p, size_t n)
{
    const unsigned char *b = p;
    for (size_t i = 0; i < n; i++) {
        h ^= b[i];
        h *= 1099511628211ULL;
    }
    return h;
}

static int argmax(const float *x, int n)
{
    int best = 0;
    for (int i = 1; i < n; i++)
        if (x[i] > x[best]) best = i;
    return best;
}

/* Deterministic non-degenerate seed context: token ids walk a range so the
 * attention/KV is realistically full (not one repeated id).  Slot offset keeps
 * users distinct.  Kept < n_vocab. */
static void fill_seed(uint32_t *buf, int depth, int n_vocab, int user)
{
    uint32_t base = 100u + (uint32_t)(user * 131);
    for (int i = 0; i < depth; i++)
        buf[i] = base + (uint32_t)((i * 7u + user * 13u) % 4096u);
    for (int i = 0; i < depth; i++)
        if (buf[i] >= (uint32_t)n_vocab) buf[i] %= (uint32_t)n_vocab;
}

/* Chunked cold prefill of `depth` tokens into st.  Returns seed token (argmax
 * of last position) or -1.  Chunked so the batched prefill never sees a huge
 * single pass on the 12 GB K80. */
static int prefill_depth(pn_model *model, pn_state *st, const uint32_t *seed,
                         int depth, int chunk, int n_vocab)
{
    int last = -1;
    for (int off = 0; off < depth; off += chunk) {
        int n = depth - off < chunk ? depth - off : chunk;
        tm_tensor *out =
            pn_qwen35moe_forward_batch_cuda(model, seed + off, n, st, NULL);
        if (!out) return -1;
        if (off + n >= depth) last = argmax(tm_data_f32(out), n_vocab);
        tm_free(out);
    }
    return last;
}

/* Run one (K, depth) point: seed K users to `depth`, then time n_gen batched
 * decode steps.  Fills *per_user and *agg (tok/s), *ok=1 on success. */
static void run_point(pn_model *model, int K, int depth, int n_gen, int chunk,
                      int n_vocab, double *per_user, double *agg, int *ok,
                      double *prefill_s)
{
    pn_state *sts[USERS_MAX] = { 0 };
    uint32_t  toks[USERS_MAX];
    float    *logits[USERS_MAX];
    uint32_t *seed = malloc((size_t)depth * sizeof(uint32_t));
    *ok = 0; *per_user = 0; *agg = 0; *prefill_s = 0;
    if (!seed) { fprintf(stderr, "  OOM host seed buf (depth=%d)\n", depth); return; }

    double pf0 = now_s();
    for (int u = 0; u < K; u++) {
        fill_seed(seed, depth, n_vocab, u);
        sts[u] = pn_state_new(model);
        if (!sts[u]) { fprintf(stderr, "  state_new failed u=%d depth=%d\n", u, depth); goto done; }
        int s = prefill_depth(model, sts[u], seed, depth, chunk, n_vocab);
        if (s < 0) { fprintf(stderr, "  PREFILL FAIL (likely OOM) u=%d depth=%d\n", u, depth); goto done; }
        if (pn_qwen35moe_decode_slot_bind(model, sts[u], u, K) != 0) {
            fprintf(stderr, "  bind fail u=%d depth=%d\n", u, depth); goto done;
        }
        toks[u] = (uint32_t)s;
    }
    *prefill_s = now_s() - pf0;

    int want_digest = getenv("QWEN35_DEPTH_DIGEST") != NULL;
    uint64_t dg = 1469598103934665603ULL;
    if (want_digest)
        printf("DIGEST_TOKENS depth=%d u0:", depth);
    double t0 = now_s();
    for (int i = 1; i < n_gen; i++) {
        if (pn_qwen35moe_decode_slots(model, toks, sts, K, logits) != 0) {
            fprintf(stderr, "  decode_slots FAIL @step %d depth=%d\n", i, depth); goto done;
        }
        for (int u = 0; u < K; u++) {
            if (want_digest)
                dg = fnv1a(dg, logits[u], (size_t)n_vocab * sizeof(float));
            toks[u] = (uint32_t)argmax(logits[u], n_vocab);
        }
        if (want_digest)
            printf(" %u", toks[0]);
    }
    double dt = now_s() - t0;
    if (want_digest)
        printf("\nDIGEST K=%d depth=%d steps=%d logits_fnv1a=%016llx\n", K,
               depth, n_gen - 1, (unsigned long long)dg);
    *per_user = (n_gen - 1) / dt;
    *agg = (double)K * (n_gen - 1) / dt;
    *ok = 1;

done:
    for (int u = 0; u < K; u++)
        if (sts[u]) pn_state_free(sts[u]);
    free(seed);
}

int main(int argc, char **argv)
{
    if (argc < 4) {
        fprintf(stderr,
                "usage: %s <model.gguf> <K> <depth> [n_gen] [chunk]\n"
                "  env QWEN35_DEPTHS=\"1024,8192,...\" sweeps depths at this K\n",
                argv[0]);
        return 1;
    }
    pn_model *model = pn_load(argv[1]);
    if (!model) { fprintf(stderr, "load failed\n"); return 1; }
    int n_vocab = model->config.n_vocab;

    int K = atoi(argv[2]); if (K < 1) K = 1; if (K > USERS_MAX) K = USERS_MAX;
    int n_gen = argc > 4 ? atoi(argv[4]) : 256; if (n_gen < 2) n_gen = 2;
    int chunk = argc > 5 ? atoi(argv[5]) : 512; if (chunk < 8) chunk = 8;

    fprintf(stderr, "depth bench: K=%d n_gen=%d chunk=%d n_vocab=%d\n",
            K, n_gen, chunk, n_vocab);

    const char *depths_env = getenv("QWEN35_DEPTHS");
    char list[256];
    if (depths_env && depths_env[0])
        snprintf(list, sizeof(list), "%s", depths_env);
    else
        snprintf(list, sizeof(list), "%s", argv[3]);

    for (char *tk = strtok(list, ","); tk; tk = strtok(NULL, ",")) {
        int depth = atoi(tk);
        if (depth < 1) continue;
        double per_user = 0, agg = 0, prefill_s = 0; int ok = 0;
        run_point(model, K, depth, n_gen, chunk, n_vocab,
                  &per_user, &agg, &ok, &prefill_s);
        if (ok) {
            fprintf(stderr,
                    "DEPTH K=%d depth=%d: per-user %.2f tok/s | aggregate "
                    "%.2f tok/s | prefill %.1fs | %s8/user\n",
                    K, depth, per_user, agg, prefill_s,
                    per_user >= 8.0 ? ">=" : "<");
            printf("DEPTHBENCH K=%d depth=%d per_user=%.2f agg=%.2f prefill_s=%.1f\n",
                   K, depth, per_user, agg, prefill_s);
        } else {
            fprintf(stderr, "DEPTH K=%d depth=%d: FAILED/OOM (boundary)\n", K, depth);
            printf("DEPTHBENCH K=%d depth=%d FAILED\n", K, depth);
        }
        fflush(stdout); fflush(stderr);
    }

    fflush(stdout); fflush(stderr);
    if (!getenv("QWEN35_EXIT_CLEAN")) _Exit(0);
    return 0;
}
#endif
