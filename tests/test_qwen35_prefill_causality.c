/* SPDX-License-Identifier: AGPL-3.0-or-later */
/* Copyright (C) 2026 David (Corey) Herrera */
/**
 * @file   test_qwen35_prefill_causality.c
 * @brief  Reference-free causality gate for every qwen35 batched-prefill path the build has.
 *
 * Position p's logits cannot depend on any token after p. So the same opening tokens,
 * prefilled once as a short prompt and once as the head of a longer one, must agree at every
 * position both runs can observe. No golden file, no second build: the model is its own
 * oracle. This is the gate the q4 fp16 window ring walked through for the whole life of that
 * feature (plans/results/q4_win_ring_hazard.md): a batched call appended all of its rows
 * before any row read, the ring lapped, and rows read K/V written by LATER rows of the same
 * call. Every A/B stayed green because the defect was on both sides of every A/B.
 *
 * Paths, chosen by the forward the engine selected for the loaded model:
 *
 *   CUDA  `pn_qwen35moe_forward_batch_cuda`, chunked exactly as serve chunks a prompt
 *         (QWEN35_BS_PREFILL_CHUNK, default 128). Run A prefills M tokens, run B N tokens;
 *         the first M per-position argmaxes must match. M is not a multiple of the chunk,
 *         so rows [chunk, M) sit in a short chunk in A and a full one in B; rows [0, chunk)
 *         are chunked identically in both runs and are the control. Under
 *         QWEN35_PREFILL_PIPELINE=1 / QWEN35_PREFILL_PIPELINE_WIDE=1 the 9b / 9c entry points
 *         also prefill probe prompts, whose LAST row must match run B's row at that position.
 *   TT    `pn_qwen35moe_prefill_tt`. It runs no LM head inside a batched pass and returns
 *         only the last row, so a row inside a pass is unobservable. Run A prefills M tokens
 *         and DECODES the rest one at a time, giving logits at every position >= M-1; each
 *         probe prefills K tokens from a fresh state, and its last row (K-1) must match run
 *         A's. The probe's passes hold rows past M that run A's prefill never saw, so a row
 *         reading its future changes the state every later position is computed from.
 *   CPU   none: the CPU qwen35moe forward is one token per call. Nothing to gate.
 *
 * CRITERION. Argmax is required everywhere. Logit digests (FNV-1a over the full row) are
 * also required where the arithmetic is the same in both runs: on CUDA the control rows (the
 * flash prefill is documented argmax-exact, not bit-exact, across batch shapes); on TT only
 * with PN_TT_PREFILL_GDNC=0, because the chunkwise GDN mixers are documented not
 * byte-identical to per-position decode. Digest differences outside that are printed, not
 * scored. GDNC=0 puts the GDN layers on the per-position fallback, which addresses one band
 * only: it needs PN_TT_PREFILL_ROWS=32, and without it the prefill declines and this FAILS.
 *
 * A TT probe counts only when pn_qwen35moe_prefill_tt_last_batched() says passes ran: an
 * aborted pass still returns logits, from the per-token loop.
 *
 * Model: PN_QWEN35_MODEL. Without it, or without a build/card that has a batched path, the
 * test prints SKIP and passes. Knobs: PN_CAUSALITY_N (384, past PN_KVQ4_WMAX = 256),
 * PN_CAUSALITY_M (150). Red proof: tests/causality_red/run_red.sh.
 */

#include "lib/test.h"
#include "tensor/tensor.h"
#include "../include/pantheon.h"
#include "../src/core/engine.h"
#include "../src/models/qwen35moe.h"

#include <stdbool.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>

#if defined(USE_CUDA) || defined(USE_TT)
#define CAUSALITY_HAS_BATCHED_PATH 1
#endif

#define CAUSALITY_DEFAULT_N     384
#define CAUSALITY_DEFAULT_M     150
#define CAUSALITY_DEFAULT_CHUNK 128
/** Mismatch lines printed per comparison before the count takes over. */
#define CAUSALITY_PRINT_LIMIT   10

/** What one logit row is reduced to: enough to score argmax and byte identity. */
typedef struct causality_row {
    int argmax;
    float top;
    uint64_t digest;
} causality_row;

/** The verdict of one comparison. `argmax_bad` and `strict_bad` fail the gate. */
typedef struct causality_score {
    int compared;
    int argmax_bad;
    int strict_bad;
    int digest_diff_unscored;
} causality_score;

static pn_model *g_model;
static uint32_t *g_tokens;

#ifdef CAUSALITY_HAS_BATCHED_PATH
static int g_n;
static int g_m;
static int g_chunk;

static int causality_env_int(const char *name, int fallback) {
    const char *value = getenv(name);

    return (value && *value) ? atoi(value) : fallback;
}

/** FNV-1a over the raw bytes, so two runs can be compared bit-exactly without keeping rows. */
static uint64_t causality_digest(const float *row, int n_vocab) {
    const unsigned char *bytes = (const unsigned char *)row;
    uint64_t hash = 1469598103934665603ULL;

    for (size_t i = 0; i < (size_t)n_vocab * sizeof(float); i++) {
        hash ^= bytes[i];
        hash *= 1099511628211ULL;
    }
    return hash;
}

static causality_row causality_summarize(const float *row, int n_vocab) {
    causality_row out = {0};

    for (int i = 1; i < n_vocab; i++)
        if (row[i] > row[out.argmax])
            out.argmax = i;
    out.top = row[out.argmax];
    out.digest = causality_digest(row, n_vocab);
    return out;
}

/**
 * Deterministic, NON-repeating token stream (the LCG the original gate used). A repeating
 * pattern would mask the defect: reading the wrong position's K/V still yields nearly the
 * right vector when every position carries the same token.
 */
static uint32_t *causality_tokens(int count, int n_vocab) {
    uint32_t *tokens = calloc((size_t)count, sizeof(*tokens));
    uint32_t seed = 0x5eed1234u;
    const int low = 1000;
    const int span = (n_vocab > 32000) ? 30000 : n_vocab - low - 1;

    if (!tokens)
        return NULL;
    for (int i = 0; i < count; i++) {
        seed = seed * 1103515245u + 12345u;
        tokens[i] = (uint32_t)(low + (int)((seed >> 16) % (uint32_t)span));
    }
    return tokens;
}

/** Score `b` against `a` at one position; `strict` makes a digest difference a failure. */
static void causality_score_row(causality_score *score, const char *label, int position,
                                const causality_row *a, const causality_row *b, bool strict) {
    const bool argmax_differs = a->argmax != b->argmax;
    const bool digest_differs = a->digest != b->digest;

    score->compared++;
    if (argmax_differs)
        score->argmax_bad++;
    if (digest_differs && strict)
        score->strict_bad++;
    if (digest_differs && !strict)
        score->digest_diff_unscored++;
    if ((argmax_differs || (digest_differs && strict)) &&
        score->argmax_bad + score->strict_bad <= CAUSALITY_PRINT_LIMIT)
        printf("    %s pos=%d argmax %d (%.6f) vs %d (%.6f) digest %016llx vs %016llx%s\n",
               label, position, a->argmax, (double)a->top, b->argmax, (double)b->top,
               (unsigned long long)a->digest, (unsigned long long)b->digest,
               argmax_differs ? "" : "  [strict: bytes differ]");
}

static void causality_report(const char *path, const causality_score *score) {
    const bool pass = score->argmax_bad == 0 && score->strict_bad == 0;

    printf("    CAUSALITY path=%s N=%d M=%d chunk=%d: %d compared, %d argmax mismatches, "
           "%d strict digest mismatches, %d unscored digest differences -- %s\n",
           path, g_n, g_m, g_chunk, score->compared, score->argmax_bad, score->strict_bad,
           score->digest_diff_unscored, pass ? "PASS" : "FAIL");
}

#ifdef USE_CUDA
/**
 * One chunked batched prefill of g_tokens[0, total) from a fresh state, mirroring
 * serve's sg_prefill_range; the summaries of positions below `keep` land in `out`.
 * @return 0, or -1 if a chunk failed (printed)
 */
static int causality_cuda_prefill(int total, int keep, causality_row *out) {
    const int n_vocab = g_model->config.n_vocab;
    float *all = malloc((size_t)g_chunk * (size_t)n_vocab * sizeof(*all));
    pn_state *state = pn_state_new(g_model);
    int rc = -1;

    if (!all || !state)
        goto cleanup;
    for (int off = 0; off < total; off += g_chunk) {
        const int rows = total - off < g_chunk ? total - off : g_chunk;
        tm_tensor *last = pn_qwen35moe_forward_batch_cuda(g_model, g_tokens + off, rows,
                                                          state, all);

        if (!last) {
            printf("    CAUSALITY cuda: batched prefill failed at %d\n", off);
            goto cleanup;
        }
        tm_free(last);
        for (int i = 0; i < rows && off + i < keep; i++)
            out[off + i] = causality_summarize(all + (size_t)i * n_vocab, n_vocab);
    }
    rc = 0;
cleanup:
    if (state)
        pn_state_free(state);
    free(all);
    return rc;
}

/**
 * Phase 9b: every chunk but the last through pn_qwen35moe_prefill_chunk_cuda (no head, die B
 * left in flight), the last through the ordinary call. Returns the last row's summary.
 */
static int causality_cuda_pipelined(int total, causality_row *out) {
    pn_state *state = pn_state_new(g_model);
    tm_tensor *last = NULL;

    if (!state)
        return -1;
    for (int off = 0; off < total; off += g_chunk) {
        const int rows = total - off < g_chunk ? total - off : g_chunk;

        if (off + rows < total && rows > 1 &&
            pn_qwen35moe_prefill_chunk_cuda(g_model, g_tokens + off, rows, state) == 0)
            continue;
        last = pn_qwen35moe_forward_batch_cuda(g_model, g_tokens + off, rows, state, NULL);
        if (!last) {
            pn_qwen35moe_prefill_drain_cuda(g_model);
            pn_state_free(state);
            return -1;
        }
        if (off + rows < total) {
            tm_free(last);
            last = NULL;
        }
    }
    *out = causality_summarize(tm_data_f32(last), g_model->config.n_vocab);
    tm_free(last);
    pn_state_free(state);
    return 0;
}

/** Phase 9c: the whole range through the n-die software pipeline. 1 = refused (not scored). */
static int causality_cuda_wide(int total, causality_row *out) {
    pn_state *state = pn_state_new(g_model);
    tm_tensor *last = NULL;
    int rc;

    if (!state)
        return -1;
    rc = pn_qwen35moe_prefill_range_pipelined_cuda(g_model, g_tokens, total, state, g_chunk,
                                                   NULL, NULL, &last);
    if (rc == 0 && last)
        *out = causality_summarize(tm_data_f32(last), g_model->config.n_vocab);
    if (last)
        tm_free(last);
    pn_state_free(state);
    return rc == 0 && last ? 0 : (rc == -1 ? 1 : -1);
}

/** The 9b / 9c probes: a K-token range's last row against run B's row K-1 (argmax). */
static int causality_cuda_probes(const causality_row *run_b, causality_score *score) {
    const int probes[] = { g_m + 1, g_n };

    for (size_t i = 0; i < sizeof(probes) / sizeof(probes[0]); i++) {
        causality_row row = {0};
        int rc;

        if (pn_qwen35moe_prefill_pipeline_enabled()) {
            if (causality_cuda_pipelined(probes[i], &row) != 0)
                return -1;
            causality_score_row(score, "pipeline-9b", probes[i] - 1, &run_b[probes[i] - 1],
                                &row, false);
        }
        if (pn_qwen35moe_prefill_pipeline_wide_enabled()) {
            rc = causality_cuda_wide(probes[i], &row);
            if (rc < 0)
                return -1;
            if (rc == 1)
                printf("    CAUSALITY cuda: 9c refused a %d-token range (not scored)\n",
                       probes[i]);
            else
                causality_score_row(score, "pipeline-9c", probes[i] - 1,
                                    &run_b[probes[i] - 1], &row, false);
        }
    }
    return 0;
}

static void test_prefill_causality_cuda(void) {
    causality_row *run_a = NULL, *run_b = NULL;
    causality_score score = {0};
    const int control = g_chunk < g_m ? g_chunk : g_m;

    if (g_model->forward != pn_qwen35moe_forward_cuda) {
        printf("    [SKIP] the engine did not select the CUDA qwen35moe forward\n");
        return;
    }
    /* q4 KV is ON unless QWEN35_KV_Q4 starts with '0' (pn_qwen35moe_kv_q4_on). */
    if ((getenv("QWEN35_KV_Q4") && getenv("QWEN35_KV_Q4")[0] == '0') || g_n <= 256)
        printf("    CAUSALITY cuda: WARNING q4 KV off or N <= PN_KVQ4_WMAX (256) -- the "
               "window-ring hazard cannot occur, this run cannot see it\n");
    run_a = calloc((size_t)g_m, sizeof(*run_a));
    run_b = calloc((size_t)g_n, sizeof(*run_b));
    ASSERT_TRUE(run_a && run_b);
    ASSERT_INT_EQ(0, causality_cuda_prefill(g_m, g_m, run_a));
    ASSERT_INT_EQ(0, causality_cuda_prefill(g_n, g_n, run_b));
    for (int p = 0; p < g_m; p++)
        causality_score_row(&score, "cuda", p, &run_a[p], &run_b[p], p < control);
    ASSERT_INT_EQ(0, causality_cuda_probes(run_b, &score));
    causality_report("cuda", &score);
    free(run_a);
    free(run_b);
    ASSERT_INT_EQ(0, score.argmax_bad);
    ASSERT_INT_EQ(0, score.strict_bad);
}
#endif /* USE_CUDA */

#ifdef USE_TT
/**
 * A fresh K-token TT batched prefill; the last row's summary. -1 = refused, failed, or not
 * batched. Logits alone prove nothing: a pass that aborts hands the prompt to the per-token
 * loop and still returns them, bit-identical to run A — PN_TT_PREFILL_GDNC=0 at the tall
 * strips did exactly that and scored green while no batched pass ran.
 */
static int causality_tt_prefill(int total, pn_state *state, causality_row *out) {
    tm_tensor *last = pn_qwen35moe_prefill_tt(g_model, g_tokens, total, state);
    const int batched = pn_qwen35moe_prefill_tt_last_batched(g_model);

    if (!last || batched <= 0) {
        printf("    CAUSALITY tt: the batched prefill of %d tokens was NOT TAKEN (%s, %d "
               "positions batched) -- a refusal scored as a pass would make this gate "
               "vacuous\n", total, last ? "ran per token" : "declined", batched);
        if (last)
            tm_free(last);
        return -1;
    }
    printf("    CAUSALITY tt: prefill of %d tokens batched %d positions\n", total, batched);
    *out = causality_summarize(tm_data_f32(last), g_model->config.n_vocab);
    tm_free(last);
    return 0;
}

/** Run A: prefill M, then decode the rest one token at a time. `out[p]` for p >= M-1. */
static int causality_tt_run_a(causality_row *out) {
    pn_state *state = pn_state_new(g_model);
    int rc = -1;

    if (!state || causality_tt_prefill(g_m, state, &out[g_m - 1]) != 0)
        goto cleanup;
    for (int p = g_m; p < g_n; p++) {
        tm_tensor *logits = g_model->forward(g_model, g_tokens[p], state);

        if (!logits) {
            printf("    CAUSALITY tt: decode failed at %d\n", p);
            goto cleanup;
        }
        out[p] = causality_summarize(tm_data_f32(logits), g_model->config.n_vocab);
        tm_free(logits);
    }
    rc = 0;
cleanup:
    if (state)
        pn_state_free(state);
    return rc;
}

static void test_prefill_causality_tt(void) {
    /* M+1: one row past M, inside the pass that straddles M. M+17: still that pass's
     * neighbourhood, with a ragged tail pass. N: the whole prompt. */
    const int probes[] = { g_m + 1, g_m + 17, g_n };
    const char *gdnc = getenv("PN_TT_PREFILL_GDNC");
    const bool strict = gdnc && *gdnc && atoi(gdnc) == 0;
    causality_score score = {0};
    causality_row *run_a = NULL;

    if (g_model->forward != pn_qwen35moe_forward_tt) {
        printf("    [SKIP] the engine did not select the Tenstorrent qwen35 forward\n");
        return;
    }
    /* A sparse row is -INFINITY almost everywhere: two would agree for a reason that is not
     * causality. */
    ASSERT_TRUE(getenv("PN_TT_SPARSE_LOGITS") == NULL);
    run_a = calloc((size_t)g_n, sizeof(*run_a));
    ASSERT_NOT_NULL(run_a);
    ASSERT_INT_EQ(0, causality_tt_run_a(run_a));
    for (size_t i = 0; i < sizeof(probes) / sizeof(probes[0]); i++) {
        causality_row row = {0};
        pn_state *state;

        if (probes[i] > g_n)
            continue;
        state = pn_state_new(g_model);
        ASSERT_NOT_NULL(state);
        ASSERT_INT_EQ(0, causality_tt_prefill(probes[i], state, &row));
        pn_state_free(state);
        causality_score_row(&score, "tt", probes[i] - 1, &run_a[probes[i] - 1], &row, strict);
    }
    causality_report(strict ? "tt-strict" : "tt", &score);
    free(run_a);
    ASSERT_INT_EQ(0, score.argmax_bad);
    ASSERT_INT_EQ(0, score.strict_bad);
}
#endif /* USE_TT */

/** Load the model once for every path. Returns false (printed) when the test must SKIP. */
static bool causality_setup(void) {
    const char *path = getenv("PN_QWEN35_MODEL");

    if (!path || !*path || access(path, F_OK) != 0) {
        printf("    [SKIP] set PN_QWEN35_MODEL to the GGUF to run\n");
        return false;
    }
    g_n = causality_env_int("PN_CAUSALITY_N", CAUSALITY_DEFAULT_N);
    g_m = causality_env_int("PN_CAUSALITY_M", CAUSALITY_DEFAULT_M);
    g_chunk = causality_env_int("QWEN35_BS_PREFILL_CHUNK", CAUSALITY_DEFAULT_CHUNK);
    if (g_chunk < 1)
        g_chunk = CAUSALITY_DEFAULT_CHUNK;
    if (g_m < 1 || g_n <= g_m) {
        printf("    CAUSALITY: need PN_CAUSALITY_N > PN_CAUSALITY_M >= 1\n");
        exit(2);
    }
    g_model = pn_load(path);
    if (!g_model) {
        printf("    CAUSALITY: cannot load %s\n", path);
        exit(2);
    }
    g_tokens = causality_tokens(g_n, g_model->config.n_vocab);
    if (!g_tokens)
        exit(2);
    return true;
}
#else
static bool causality_setup(void) {
    printf("    [SKIP] this build has no qwen35 batched-prefill path (CUDA or TT); the CPU "
           "forward is one token per call\n");
    return false;
}
#endif /* CAUSALITY_HAS_BATCHED_PATH */

int main(void) {
    int failed = 0;

    printf("Pantheon qwen35 batched-prefill causality\n");
    printf("=========================================\n\n");
    if (causality_setup()) {
#ifdef USE_CUDA
        RUN_TEST(prefill_causality_cuda);
#endif
#ifdef USE_TT
        RUN_TEST(prefill_causality_tt);
#endif
    }
    TEST_REPORT();
    failed = __ft_failed > 0 ? 1 : 0;
    fflush(stdout);
    fflush(stderr);
#ifdef USE_CUDA
    /* CUDA context teardown wedges on the K80 rig; QWEN35_EXIT_CLEAN=1 asks for it anyway. */
    if (g_model && !getenv("QWEN35_EXIT_CLEAN"))
        _Exit(failed);
#endif
    free(g_tokens);
    if (g_model)
        pn_model_free(g_model);
    return failed;
}
