/* SPDX-License-Identifier: AGPL-3.0-or-later */
/* Copyright (C) 2026 David (Corey) Herrera */
/* test_verify_batch.c - the M2/M3 correctness gate for pn_llama_verify_batch.
 *
 * pn_llama_verify_batch verifies several INDEPENDENT sequences in one forward
 * pass (block b = n_tok[b] candidate tokens for sequence b, starting at that
 * sequence's committed position). Each output row must be the logits the model
 * produces for the prefix ending at that candidate, using only that sequence's
 * own cache. Greedy accept/reject over these rows must then reproduce the plain
 * batched-greedy token stream (proven end to end in test_spec_batch.c).
 *
 * A NOTE ON "BIT-IDENTICAL" (read this): with USE_BLAS the projection / FFN /
 * LM-head GEMMs (pn_qmm) lower to Accelerate cblas_sgemm, whose K-dimension
 * summation is reassociated as a function of the batch size M. So the logits of
 * one row computed in an M-row batch are NOT byte-equal to the same row computed
 * in a batch of a different size (e.g. a batch=1 replay) - they differ at the
 * ~5e-5 level, which is far below any argmax gap and never flips a token (this
 * is exactly the reassociation test_spec_decode.c documents). The emitted TOKEN
 * stream is therefore bit-identical (the real losslessness guarantee), while raw
 * logit bytes are equal only when M is held fixed. This gate is built around
 * what is bitwise-achievable AND still pins down every ragged-indexing bug:
 *
 *   [A] argmax-exact replay: every verify_batch row's ARGMAX equals an
 *       INDEPENDENT per-slot replay (prefill the committed prefix, then run
 *       pn_llama_forward_batch batch=1 over the block one token at a time so the
 *       block's earlier K/V is built by the SAME pn_qmm path), and the max
 *       |logit delta| stays at reassociation noise (<1e-2). A wrong row_pos /
 *       cache_pos / seq_len yields a totally different prediction (delta >> 1 and
 *       argmax flips), so this catches every within-block position bug.
 *   [B] bit-exact cross-cache independence (fixed M): rerun the SAME multi-seq
 *       verify_batch with ONE block's prefix+tokens swapped for different ones of
 *       the SAME length. Because M is unchanged the reassociation is identical,
 *       so every OTHER block's rows must be BYTE-identical and the swapped block
 *       must differ. A wrong row_seq index (reading another sequence's cache)
 *       would leak the swap into a neighbour's rows and break this exactly.
 *   [C] negative control: a reference built against the WRONG sequence's cache
 *       must differ grossly, proving the checks can actually see contamination.
 *
 * Usage: test_verify_batch [model.gguf]
 */

#include "../include/pantheon.h"
#include "../src/core/engine.h"
#include "../src/models/llama.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

#define DELTA_TOL 1e-2f     /* reassociation noise is ~5e-5; a bug is >> 1 */

static int g_nv = 0;

static uint32_t argmax(const float *l, int n) {
    int b = 0;
    for (int i = 1; i < n; i++) if (l[i] > l[b]) b = i;
    return (uint32_t)b;
}

static void rng_fill(uint32_t *p, int n, uint32_t seed) {
    uint32_t x = 0x9e3779b9u * (seed + 1);
    for (int i = 0; i < n; i++) {
        x ^= x << 13; x ^= x >> 17; x ^= x << 5;
        p[i] = (x % (uint32_t)(g_nv - 4)) + 2;   /* valid, non-special ids */
    }
}

/* Replay reference: fresh state, prefill `prefix`, then run forward_batch
 * (batch=1) over block[0..n_blk-1] one token at a time, writing the logits
 * predicted after block[j] into ref[j*nv]. Builds the block's earlier K/V via
 * the same pn_qmm path verify_batch uses, so the ONLY numeric gap left is the
 * batch-size reassociation. */
static int build_reference(pn_model *m, const uint32_t *prefix, int prefix_len,
                           const uint32_t *block, int n_blk, float *ref) {
    pn_state *st = pn_state_new(m);
    if (!st) return -1;
    tm_tensor *pl = pn_llama_prefill(m, prefix, NULL, prefix_len, st);
    if (!pl) { pn_state_free(st); return -1; }
    tm_free(pl);
    for (int j = 0; j < n_blk; j++) {
        uint32_t tok = block[j];
        pn_state *sts[1] = { st };
        if (pn_llama_forward_batch(m, &tok, NULL, sts, 1, ref + (size_t)j * g_nv) != 0) {
            pn_state_free(st); return -1;
        }
    }
    pn_state_free(st);
    return 0;
}

/* Prefill fresh states for every sequence and run ONE ragged verify_batch,
 * writing [M, nv] logits into out. Returns 0 on success, or asserts pos was not
 * advanced (verify never commits). */
static int run_verify(pn_model *m, const uint32_t *const *prefixes,
                      const int *prefix_len, const uint32_t *const *blocks,
                      const int *n_tok, int n_seqs, float *out) {
    int M = 0; for (int b = 0; b < n_seqs; b++) M += n_tok[b];
    pn_state **states = malloc((size_t)n_seqs * sizeof(pn_state *));
    uint32_t *cat = malloc((size_t)M * sizeof(uint32_t));
    int rc = 0, r = 0;
    for (int b = 0; b < n_seqs; b++) {
        states[b] = pn_state_new(m);
        tm_tensor *pl = pn_llama_prefill(m, prefixes[b], NULL, prefix_len[b], states[b]);
        if (!pl) { rc = -1; goto done; }
        tm_free(pl);
        for (int j = 0; j < n_tok[b]; j++) cat[r++] = blocks[b][j];
    }
    if (pn_llama_verify_batch(m, cat, states, n_tok, n_seqs, out) != 0) { rc = -1; goto done; }
    for (int b = 0; b < n_seqs; b++)
        if (states[b]->pos != prefix_len[b]) {
            fprintf(stderr, "  seq %d pos advanced (%d != %d) - verify must not commit\n",
                    b, states[b]->pos, prefix_len[b]);
            rc = -1;
        }
done:
    for (int b = 0; b < n_seqs; b++) if (states[b]) pn_state_free(states[b]);
    free(states); free(cat);
    return rc;
}

/* Check A: every row's argmax equals its independent replay, and the logit
 * delta stays at reassociation noise. */
static int check_argmax_replay(pn_model *m, const char *label,
                               const uint32_t *const *prefixes, const int *prefix_len,
                               const uint32_t *const *blocks, const int *n_tok, int n_seqs) {
    int M = 0; for (int b = 0; b < n_seqs; b++) M += n_tok[b];
    float *vb = malloc((size_t)M * g_nv * sizeof(float));
    if (run_verify(m, prefixes, prefix_len, blocks, n_tok, n_seqs, vb) != 0) {
        fprintf(stderr, "%s: verify run failed\n", label); free(vb); return 1;
    }
    int argmax_mism = 0; float maxd = 0.0f; int r = 0;
    for (int b = 0; b < n_seqs; b++) {
        float *ref = malloc((size_t)n_tok[b] * g_nv * sizeof(float));
        if (build_reference(m, prefixes[b], prefix_len[b], blocks[b], n_tok[b], ref) != 0) {
            fprintf(stderr, "%s: reference build failed\n", label); free(ref); free(vb); return 1;
        }
        for (int j = 0; j < n_tok[b]; j++) {
            const float *row = vb + (size_t)(r + j) * g_nv;
            const float *rr  = ref + (size_t)j * g_nv;
            if (argmax(row, g_nv) != argmax(rr, g_nv)) argmax_mism++;
            for (int i = 0; i < g_nv; i++) {
                float d = fabsf(row[i] - rr[i]); if (d > maxd) maxd = d;
            }
        }
        free(ref); r += n_tok[b];
    }
    int ok = (argmax_mism == 0 && maxd < DELTA_TOL);
    printf("  [A] %-20s seqs=%d M=%2d  argmax-mism=%d  max|delta|=%.2e  %s\n",
           label, n_seqs, M, argmax_mism, maxd, ok ? "OK" : "FAIL");
    free(vb);
    return ok ? 0 : 1;
}

/* Check B: bit-exact cross-cache independence. Swap block `swap`'s prefix and
 * tokens for a different set of the SAME length; M is unchanged so the GEMM
 * reassociation is identical. Every other block's rows must be BYTE-identical,
 * and the swapped block's rows must differ. */
static int check_independence(pn_model *m, const char *label,
                              const uint32_t *const *prefixes, const int *prefix_len,
                              const uint32_t *const *blocks, const int *n_tok, int n_seqs,
                              int swap, const uint32_t *alt_prefix, const uint32_t *alt_block) {
    int M = 0, off_swap = 0; for (int b = 0; b < n_seqs; b++) { if (b < swap) off_swap += n_tok[b]; M += n_tok[b]; }
    float *base = malloc((size_t)M * g_nv * sizeof(float));
    float *alt  = malloc((size_t)M * g_nv * sizeof(float));
    if (run_verify(m, prefixes, prefix_len, blocks, n_tok, n_seqs, base) != 0) {
        free(base); free(alt); return 1;
    }
    const uint32_t *pf2[8]; const uint32_t *bl2[8];
    for (int b = 0; b < n_seqs; b++) { pf2[b] = prefixes[b]; bl2[b] = blocks[b]; }
    pf2[swap] = alt_prefix; bl2[swap] = alt_block;   /* same prefix_len[swap], n_tok[swap] */
    if (run_verify(m, pf2, prefix_len, bl2, n_tok, n_seqs, alt) != 0) {
        free(base); free(alt); return 1;
    }
    int other_diff = 0, swapped_same = 0, r = 0;
    for (int b = 0; b < n_seqs; b++) {
        for (int j = 0; j < n_tok[b]; j++) {
            int eq = memcmp(base + (size_t)(r + j) * g_nv, alt + (size_t)(r + j) * g_nv,
                            (size_t)g_nv * sizeof(float)) == 0;
            if (b == swap) { if (eq) swapped_same++; }
            else           { if (!eq) other_diff++; }
        }
        r += n_tok[b];
    }
    (void)off_swap;
    int ok = (other_diff == 0 && swapped_same == 0);
    printf("  [B] %-20s swap seq %d: other-blocks-changed=%d swapped-unchanged=%d  %s\n",
           label, swap, other_diff, swapped_same,
           ok ? "OK (byte-identical, no contamination)" : "FAIL");
    free(base); free(alt);
    return ok ? 0 : 1;
}

int main(int argc, char **argv) {
    const char *path = (argc > 1) ? argv[1] : "models/lightonocr-lm-Q4_K_M.gguf";
    pn_model *model = pn_load(path);
    if (!model) { fprintf(stderr, "load failed: %s\n", path); return 1; }
    g_nv = model->config.n_vocab;
    printf("model: %s  n_vocab=%d n_layer=%d n_embd=%d arch=%d\n",
           path, g_nv, model->config.n_layer, model->config.n_embd, model->config.arch);

    { /* support gate */
        pn_state *s = pn_state_new(model);
        uint32_t t = 2; int nt = 1;
        float *lg = malloc((size_t)g_nv * sizeof(float));
        int ok = lg && pn_llama_verify_batch(model, &t, &s, &nt, 1, lg) == 0;
        free(lg); pn_state_free(s);
        if (!ok) { printf("[SKIP] model not on the quantized non-MoE verify path\n");
                   pn_model_free(model); return 0; }
    }

    int fail = 0;

    enum { P0 = 32, P1 = 17, P2 = 48, P3 = 5 };
    uint32_t pre0[P0], pre1[P1], pre2[P2], pre3[P3], preX[P1];
    rng_fill(pre0, P0, 10); rng_fill(pre1, P1, 11);
    rng_fill(pre2, P2, 12); rng_fill(pre3, P3, 13); rng_fill(preX, P1, 77);
    uint32_t blk0[9], blk1[9], blk2[9], blk3[9], blkX[9];
    rng_fill(blk0, 9, 100); rng_fill(blk1, 9, 101);
    rng_fill(blk2, 9, 102); rng_fill(blk3, 9, 103); rng_fill(blkX, 9, 200);

    printf("\n== M2: single sequence, block [seed, d0..d_{K-1}] ==\n");
    for (int K = 0; K <= 8; K += 2) {
        const uint32_t *pf[] = { pre0 }; int pl[] = { P0 };
        const uint32_t *bl[] = { blk0 }; int nt[] = { K + 1 };
        char lbl[24]; snprintf(lbl, sizeof(lbl), "K=%d", K);
        fail |= check_argmax_replay(model, lbl, pf, pl, bl, nt, 1);
    }

    printf("\n== M3: ragged multi-sequence (mixed lengths, one plain K=1 block) ==\n");
    {
        const uint32_t *pf[] = { pre0, pre1, pre2 }; int pl[] = { P0, P1, P2 };
        const uint32_t *bl[] = { blk0, blk1, blk2 }; int nt[] = { 5, 1, 8 };
        fail |= check_argmax_replay(model, "3-seq {5,1,8}", pf, pl, bl, nt, 3);
        /* swap the MIDDLE (length-1) block: proves K=1 blocks are isolated too */
        fail |= check_independence(model, "3-seq {5,1,8}", pf, pl, bl, nt, 3, 1, preX, blkX);
    }
    {
        const uint32_t *pf[] = { pre3, pre0, pre1, pre2 }; int pl[] = { P3, P0, P1, P2 };
        const uint32_t *bl[] = { blk3, blk0, blk1, blk2 }; int nt[] = { 4, 4, 2, 6 };
        fail |= check_argmax_replay(model, "4-seq {4,4,2,6}", pf, pl, bl, nt, 4);
        fail |= check_independence(model, "4-seq {4,4,2,6}", pf, pl, bl, nt, 4, 2, preX, blkX);
    }
    {
        const uint32_t *pf[] = { pre0, pre1, pre2, pre3 }; int pl[] = { P0, P1, P2, P3 };
        const uint32_t *bl[] = { blk0, blk1, blk2, blk3 }; int nt[] = { 1, 1, 1, 1 };
        fail |= check_argmax_replay(model, "4-seq all K=1", pf, pl, bl, nt, 4);
    }

    printf("\n== [C] negative control: reference against the WRONG cache must differ ==\n");
    {
        pn_state *sa = pn_state_new(model), *sb = pn_state_new(model);
        tm_tensor *p;
        p = pn_llama_prefill(model, pre0, NULL, P0, sa); tm_free(p);
        p = pn_llama_prefill(model, pre1, NULL, P1, sb); tm_free(p);
        pn_state *states[2] = { sa, sb }; int nt[2] = { 4, 4 };
        uint32_t cat[8];
        for (int j = 0; j < 4; j++) { cat[j] = blk0[j]; cat[4 + j] = blk1[j]; }
        float *vb = malloc((size_t)8 * g_nv * sizeof(float));
        pn_llama_verify_batch(model, cat, states, nt, 2, vb);
        float *ref = malloc((size_t)4 * g_nv * sizeof(float));
        build_reference(model, pre1, P1, blk0, 4, ref);   /* WRONG prefix for seq 0 */
        int gross = 0;
        for (int j = 0; j < 4; j++) {
            float md = 0; for (int i = 0; i < g_nv; i++) { float d = fabsf(vb[(size_t)j*g_nv+i]-ref[(size_t)j*g_nv+i]); if (d>md) md=d; }
            if (md > DELTA_TOL) gross++;
        }
        printf("  wrong-cache reference differs grossly on %d/4 rows  %s\n",
               gross, (gross == 4) ? "GOOD" : "SUSPICIOUS");
        if (gross != 4) fail = 1;
        free(vb); free(ref); pn_state_free(sa); pn_state_free(sb);
    }

    printf("\nRESULT: %s\n", fail ? "FAIL"
        : "PASS (argmax-exact vs replay, bit-exact cross-cache isolation, negative control)");
    pn_model_free(model);
    return fail ? 1 : 0;
}
