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
 *      QWEN35_DEPTH_KS="1,2,3,4" seeds K users ONCE per depth, then times each
 *      listed batch width k (users [0,k) step, the rest idle in their rows).
 *      QWEN35_DEPTH_PROFILE_K=1,4 brackets those widths' timed decode with
 *      cuProfilerStart/Stop (run under nvprof --profile-from-start off).
 *      QWEN35_DEPTH_RESEED=1 re-seeds every user before each width, so user u
 *      decodes the SAME positions at every width: its digest must then match
 *      across widths (batching changes speed, never a row's logits).
 *      QWEN35_DEPTH_FAKE=<n> prefills n real tokens and skips to depth
 *      (garbage KV, real attention cost; timing only, for 64k-128k points).
 *      QWEN35_DEPTH_DIGEST=1 folds an FNV-1a over every decode step's full
 *      logits per user and prints one digest per user, so two BUILDS (or two
 *      batch widths) can be diffed for bit-exactness at depth (costs a host
 *      pass over the logits per step; leave it off for timing arms).
 *      QWEN35_DEPTH_FULL="0,1,4" times each width once per listed count f:
 *      users [0,f) carry full_logits (the serve's tool-grammar row, every
 *      step), the rest the top-k row. Prices the constrained slots' fallback
 *      (QWEN35_PERSLOT_FULL_LOGITS picks which one). Unset: one pass, f = 0.
 * NOT part of the test suite (needs the model + a CUDA GPU). */
#include "../include/pantheon.h"
#include "../src/core/engine.h"
#include "../src/models/qwen35moe.h"
#include "tensor/tensor.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <dlfcn.h>

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

/* QWEN35_DEPTH_PROFILE_K=k[,k2..]: bracket ONLY the timed decode of these widths
 * with cuProfilerStart/Stop, so `nvprof --profile-from-start off` sees decode
 * kernels and none of the seeding prefill.  libcuda is already loaded by
 * foundry; look the symbols up without linking it. */
static void profiler_mark(int k, int start)
{
    const char *e = getenv("QWEN35_DEPTH_PROFILE_K");
    char want[16];
    snprintf(want, sizeof(want), ",%d,", k);
    char list[64];
    snprintf(list, sizeof(list), ",%s,", e ? e : "");
    if (!strstr(list, want))
        return;
    void *h = dlopen("libcuda.so.1", RTLD_NOW | RTLD_NOLOAD);
    int (*fn)(void) = h ? (int (*)(void))dlsym(h, start ? "cuProfilerStart"
                                                       : "cuProfilerStop")
                        : NULL;
    int rc = fn ? fn() : -1;
    fprintf(stderr, "profiler %s for K=%d: rc=%d\n", start ? "start" : "stop", k, rc);
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

/* Seed users [0,K) to `depth` (untimed chunked prefill) and bind each to
 * its slot row in a K-row layout.  Fills sts[] and toks[] (the next token per
 * user).  Returns 0, or -1 with the failure logged. */
static int seed_users(pn_model *model, int K, int depth, int chunk,
                      int n_vocab, pn_state **sts, uint32_t *toks)
{
    uint32_t *seed = malloc((size_t)depth * sizeof(*seed));
    int rc = -1;
    if (!seed) { fprintf(stderr, "  OOM host seed buf (depth=%d)\n", depth); return -1; }
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
    rc = 0;
done:
    free(seed);
    return rc;
}

/* Time n_gen-1 batched decode steps over users [0,k) (already seeded and
 * bound).  With QWEN35_DEPTH_DIGEST set, folds an FNV-1a over each user's
 * full logits row every step and prints one digest PER USER, so a user's
 * stream can be compared across builds and across batch widths.  Returns
 * the elapsed seconds, or a negative value on failure. */
static double time_decode(pn_model *model, pn_state **sts, uint32_t *toks,
                          int k, int depth, int n_gen, int n_vocab)
{
    float *logits[USERS_MAX];
    uint64_t dg[USERS_MAX];
    int want_digest = getenv("QWEN35_DEPTH_DIGEST") != NULL;
    for (int u = 0; u < k; u++)
        dg[u] = 1469598103934665603ULL;
    profiler_mark(k, 1);
    double t0 = now_s();
    for (int i = 1; i < n_gen; i++) {
        if (pn_qwen35moe_decode_slots(model, toks, sts, k, logits) != 0) {
            fprintf(stderr, "  decode_slots FAIL @step %d depth=%d\n", i, depth);
            return -1.0;
        }
        for (int u = 0; u < k; u++) {
            if (want_digest)
                dg[u] = fnv1a(dg[u], logits[u], (size_t)n_vocab * sizeof(float));
            toks[u] = (uint32_t)argmax(logits[u], n_vocab);
        }
    }
    double dt = now_s() - t0;
    profiler_mark(k, 0);
    for (int u = 0; u < k && want_digest; u++)
        printf("DIGEST K=%d depth=%d user=%d steps=%d logits_fnv1a=%016llx\n", k,
               depth, u, n_gen - 1, (unsigned long long)dg[u]);
    return dt;
}

static void report_point(int K, int depth, int n_gen, double dt, double prefill_s)
{
    if (dt <= 0.0) {
        fprintf(stderr, "DEPTH K=%d depth=%d: FAILED/OOM (boundary)\n", K, depth);
        printf("DEPTHBENCH K=%d depth=%d FAILED\n", K, depth);
        return;
    }
    double per_user = (n_gen - 1) / dt;
    double agg = (double)K * (n_gen - 1) / dt;
    fprintf(stderr, "DEPTH K=%d depth=%d: per-user %.2f tok/s | aggregate %.2f tok/s | "
            "step %.1f ms | prefill %.1fs | %s8/user\n", K, depth, per_user, agg,
            1000.0 * dt / (n_gen - 1), prefill_s, per_user >= 8.0 ? ">=" : "<");
    printf("DEPTHBENCH K=%d depth=%d per_user=%.2f agg=%.2f step_ms=%.1f prefill_s=%.1f\n",
           K, depth, per_user, agg, 1000.0 * dt / (n_gen - 1), prefill_s);
    fflush(stdout); fflush(stderr);
}

/* Time width k once per count in QWEN35_DEPTH_FULL (default "0"): users [0,f) carry
 * full_logits, as a tool-grammar slot does for its whole turn. Returns 0, or -1 when a
 * pass failed. */
static int time_full_sweep(pn_model *model, pn_state **sts, uint32_t *toks, int k,
                           int depth, int n_gen, int n_vocab, double prefill_s)
{
    const char *env = getenv("QWEN35_DEPTH_FULL");
    char list[64];
    char *save = NULL;

    snprintf(list, sizeof(list), "%s", env && env[0] ? env : "0");
    for (char *tk = strtok_r(list, ",", &save); tk; tk = strtok_r(NULL, ",", &save)) {
        int f = atoi(tk);
        if (f < 0 || f > k)
            continue;
        for (int u = 0; u < k; u++)
            pn_qwen35moe_state_set_full_logits(sts[u], u < f);
        double dt = time_decode(model, sts, toks, k, depth, n_gen, n_vocab);
        if (env)
            printf("FULLROWS K=%d full=%d ", k, f);
        report_point(k, depth, n_gen, dt, prefill_s);
        if (dt <= 0.0)
            return -1;
    }
    return 0;
}

/* QWEN35_DEPTH_FAKE=<n>: prefill only n real tokens per user, then skip each
 * user to `depth` with pn_qwen35moe_bench_skip_positions (garbage KV, real
 * attention cost). Timing only: digests are meaningless in this mode. */
static int fake_real_tokens(void)
{
    const char *e = getenv("QWEN35_DEPTH_FAKE");
    return e ? atoi(e) : 0;
}

static int seed_depth(int depth)
{
    int n = fake_real_tokens();
    return n > 0 && n < depth ? n : depth;
}

/* Step every user once (so its ring watermarks are current), then skip it to
 * `depth`. No-op unless QWEN35_DEPTH_FAKE is set. Returns 0 / -1. */
static int fake_depth(pn_model *model, pn_state **sts, uint32_t *toks, int K,
                      int depth, int n_vocab)
{
    float *logits[USERS_MAX];
    if (seed_depth(depth) == depth)
        return 0;
    if (pn_qwen35moe_decode_slots(model, toks, sts, K, logits) != 0)
        return -1;
    for (int u = 0; u < K; u++) {
        toks[u] = (uint32_t)argmax(logits[u], n_vocab);
        int add = depth - (int)sts[u]->pos;
        if (add > 0 && pn_qwen35moe_bench_skip_positions(model, sts[u], u, add) != 0) {
            fprintf(stderr, "  fake depth: skip to %d failed for user %d\n", depth, u);
            return -1;
        }
    }
    fprintf(stderr, "  fake depth: %d real tokens, skipped to %d (timing only)\n",
            seed_depth(depth), depth);
    return 0;
}

/* One depth point: seed K users once, then time the batch widths listed in
 * QWEN35_DEPTH_KS (e.g. "1,2,3,4": users [0,k) step together, the rest idle
 * in their rows — the serve's layout), or just width K when it is unset. */
static void run_point(pn_model *model, int K, int depth, int n_gen, int chunk,
                      int n_vocab)
{
    pn_state *sts[USERS_MAX] = { 0 };
    uint32_t  toks[USERS_MAX];
    double pf0 = now_s();
    if (seed_users(model, K, seed_depth(depth), chunk, n_vocab, sts, toks) != 0) {
        report_point(K, depth, n_gen, -1.0, 0.0);
        goto done;
    }
    double prefill_s = now_s() - pf0;
    if (fake_depth(model, sts, toks, K, depth, n_vocab) != 0)
        goto done;
    const char *ks_env = getenv("QWEN35_DEPTH_KS");
    char ks[64];
    snprintf(ks, sizeof(ks), "%s", ks_env && ks_env[0] ? ks_env : "");
    if (!ks[0])
        snprintf(ks, sizeof(ks), "%d", K);
    int reseed = getenv("QWEN35_DEPTH_RESEED") != NULL;
    int first = 1;
    char *save = NULL;
    for (char *tk = strtok_r(ks, ",", &save); tk; tk = strtok_r(NULL, ",", &save)) {
        int k = atoi(tk);
        if (k < 1 || k > K)
            continue;
        if (reseed && !first) {
            for (int u = 0; u < K; u++) {
                pn_state_free(sts[u]);
                sts[u] = NULL;
            }
            if (seed_users(model, K, seed_depth(depth), chunk, n_vocab, sts, toks) != 0 ||
                fake_depth(model, sts, toks, K, depth, n_vocab) != 0)
                break;
        }
        first = 0;
        if (time_full_sweep(model, sts, toks, k, depth, n_gen, n_vocab, prefill_s) != 0)
            break;
    }
done:
    for (int u = 0; u < K; u++)
        if (sts[u]) pn_state_free(sts[u]);
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
        run_point(model, K, depth, n_gen, chunk, n_vocab);
        fflush(stdout); fflush(stderr);
    }

    fflush(stdout); fflush(stderr);
    if (!getenv("QWEN35_EXIT_CLEAN")) _Exit(0);
    return 0;
}
#endif
