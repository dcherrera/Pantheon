/* SPDX-License-Identifier: AGPL-3.0-or-later */
/* Copyright (C) 2026 David (Corey) Herrera */
/**
 * @file test_batch_verify_exact.c
 * @brief Byte-exactness gate for the speculative verify (QWEN35_BATCH_VERIFY):
 *        verified rows' logits and the committed state vs plain decode steps.
 *
 * Per depth D (the seed context is the bench's deterministic one):
 *   REF   prefill D, bind to slot 0, R plain decode-slot steps (the serve's
 *         decode path, K = 1) over block = [first, drafts...] -> L_ref[r]
 *   VER   prefill D, pn_qwen35moe_spec_verify(block) -> L_ver[r] (every row's
 *         full logits via pn_qwen35moe_spec_row_logits), accept(k), bind, two
 *         plain steps over x0, x1 -> L_post[2]
 *   REF2  prefill D, bind, plain steps over block[0..k) then x0, x1 -> L_ref2[2]
 * PASS iff every L_ver[r] == L_ref[r] byte for byte AND L_post == L_ref2 byte
 * for byte (the committed KV/SSM state is exactly k plain steps' state: the
 * two post steps read every committed KV row, the overwritten draft tail and
 * the rolled-back SSM state). The red arm is QWEN35_BATCH_VERIFY=0 (the prefill
 * kernel's reduction order): it MUST report a mismatch.
 *
 * BV_TOPK=40 verifies through pn_qwen35moe_spec_verify_topk (the serve's CUDA spec
 * step) instead, so the rows read afterwards are the ones a grammar draw would fetch
 * after the per-row top-k reduction has run over them (it must restore every row).
 *
 * Build (CUDA host): make USE_CUDA=1 tests/test_batch_verify_exact
 * Run:   <production env> ./tests/test_batch_verify_exact <model.gguf> \
 *          <depth[,depth...]> [rows=5] [accept=2]
 * Prints one "ROW" line per verified row, one "POST" line per depth and
 * "GATE: BYTE-EXACT PASS" / "GATE: FAIL ..." — exit 0 only on PASS.
 * NOT part of the suite (needs the model + a CUDA GPU).
 *
 * @status complete
 * @issues None
 * @todo None
 */
#include "../include/pantheon.h"
#include "../src/core/engine.h"
#include "../src/models/qwen35moe.h"
#include "../src/models/qwen35moe_internal.h"
#include "tensor/tensor.h"
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#ifndef USE_CUDA
int main(void) { fprintf(stderr, "built without USE_CUDA\n"); return 1; }
#else

#define BV_MAX_ROWS 8
#define BV_CHUNK 256

static int argmax_f32(const float *x, int n)
{
    int best = 0;
    for (int i = 1; i < n; i++)
        if (x[i] > x[best]) best = i;
    return best;
}

/* Same deterministic seed as bench_spec_verify_gate (user 0). */
static void fill_seed_context(uint32_t *buf, int depth, int n_vocab)
{
    for (int i = 0; i < depth; i++)
        buf[i] = (100u + (uint32_t)((i * 7u) % 4096u)) % (uint32_t)n_vocab;
}

/* Fresh state prefilled to `depth` in BV_CHUNK chunks; *first = last argmax. */
static pn_state *prefill_new_state(pn_model *model, const uint32_t *seed,
                                   int depth, int n_vocab, uint32_t *first)
{
    pn_state *st = pn_state_new(model);
    if (!st) return NULL;
    for (int off = 0; off < depth; off += BV_CHUNK) {
        int n = depth - off < BV_CHUNK ? depth - off : BV_CHUNK;
        tm_tensor *out = pn_qwen35moe_forward_batch_cuda(model, seed + off, n,
                                                         st, NULL);
        if (!out) { pn_state_free(st); return NULL; }
        if (off + n >= depth)
            *first = (uint32_t)argmax_f32(tm_data_f32(out), n_vocab);
        tm_free(out);
    }
    ((pn_qwen35moe_state *)st->kv_cache)->full_logits = 1;
    return st;
}

/* Bind st to slot 0 and run plain decode-slot steps over toks, copying each
 * step's full logits to out + i * n_vocab. */
static int plain_steps(pn_model *model, pn_state *st, const uint32_t *toks,
                       int n, float *out, int n_vocab)
{
    if (pn_qwen35moe_decode_slot_bind(model, st, 0, 1) != 0)
        return -1;
    for (int i = 0; i < n; i++) {
        float *logits[1];
        pn_state *sts[1] = { st };
        if (pn_qwen35moe_decode_slots(model, &toks[i], sts, 1, logits) != 0)
            return -1;
        memcpy(out + (size_t)i * n_vocab, logits[0],
               (size_t)n_vocab * sizeof(float));
    }
    return 0;
}

/* Byte compare; prints the line and returns 1 on mismatch. */
static int compare_row(const char *tag, int depth, int row, const float *a,
                       const float *b, int n_vocab)
{
    int ndiff = 0;
    double maxd = 0.0;
    for (int i = 0; i < n_vocab; i++) {
        if (memcmp(&a[i], &b[i], sizeof(float)) != 0) {
            ndiff++;
            double d = fabs((double)a[i] - (double)b[i]);
            if (d > maxd) maxd = d;
        }
    }
    printf("%s depth=%d row=%d argmax %d/%d diff_floats=%d max_abs=%.3g %s\n",
           tag, depth, row, argmax_f32(a, n_vocab), argmax_f32(b, n_vocab),
           ndiff, maxd, ndiff ? "MISMATCH" : "exact");
    return ndiff != 0;
}

/* The verify under test: the argmax verify, or with BV_TOPK=k the top-k verify. */
static int verify_rows(pn_model *model, const uint32_t *block, int rows, pn_state *st)
{
    static int ids[BV_MAX_ROWS * 64];
    static float vals[BV_MAX_ROWS * 64];
    int32_t am[BV_MAX_ROWS];
    const char *env = getenv("BV_TOPK");
    int k = env ? atoi(env) : 0;

    if (k < 1)
        return pn_qwen35moe_spec_verify(model, block, rows, st, am);
    if (k > 64)
        return -1;
    return pn_qwen35moe_spec_verify_topk(model, block, rows, st, k, ids, vals);
}

/* One depth: REF, VER, REF2 as in the file header. Returns mismatches, -1. */
static int run_depth(pn_model *model, int depth, int rows, int k, int n_vocab)
{
    size_t vbytes = (size_t)n_vocab * sizeof(float);
    uint32_t *seed = malloc((size_t)depth * sizeof(uint32_t));
    float *ref = malloc(vbytes * rows), *ver = malloc(vbytes * rows);
    float *post = malloc(vbytes * 2), *ref2 = malloc(vbytes * (k + 2));
    uint32_t block[BV_MAX_ROWS], tail[BV_MAX_ROWS + 2], first = 0;
    int bad = -1;
    pn_state *st = NULL;
    if (!seed || !ref || !ver || !post || !ref2) goto done;
    fill_seed_context(seed, depth, n_vocab);

    /* REF */
    if (!(st = prefill_new_state(model, seed, depth, n_vocab, &first))) goto done;
    block[0] = first;
    for (int r = 1; r < rows; r++)
        block[r] = (uint32_t)((1000 + r * 37) % n_vocab);
    if (plain_steps(model, st, block, rows, ref, n_vocab) != 0) goto done;
    pn_state_free(st);

    /* VER */
    if (!(st = prefill_new_state(model, seed, depth, n_vocab, &first))) goto done;
    if (verify_rows(model, block, rows, st) != 0) goto done;
    for (int r = 0; r < rows; r++)
        if (pn_qwen35moe_spec_row_logits(model, r, ver + (size_t)r * n_vocab) != 0)
            goto done;
    if (pn_qwen35moe_spec_accept(model, st, k) != 0) goto done;
    uint32_t x[2] = { (uint32_t)(2000 % n_vocab), (uint32_t)(3001 % n_vocab) };
    if (plain_steps(model, st, x, 2, post, n_vocab) != 0) goto done;
    pn_state_free(st);

    /* REF2 */
    if (!(st = prefill_new_state(model, seed, depth, n_vocab, &first))) goto done;
    memcpy(tail, block, (size_t)k * sizeof(uint32_t));
    tail[k] = x[0];
    tail[k + 1] = x[1];
    if (plain_steps(model, st, tail, k + 2, ref2, n_vocab) != 0) goto done;

    bad = 0;
    for (int r = 0; r < rows; r++)
        bad += compare_row("ROW", depth, r, ref + (size_t)r * n_vocab,
                           ver + (size_t)r * n_vocab, n_vocab);
    for (int i = 0; i < 2; i++)
        bad += compare_row("POST", depth, i, ref2 + (size_t)(k + i) * n_vocab,
                           post + (size_t)i * n_vocab, n_vocab);
done:
    if (st) pn_state_free(st);
    free(seed); free(ref); free(ver); free(post); free(ref2);
    return bad;
}

int main(int argc, char **argv)
{
    if (argc < 3) {
        fprintf(stderr, "usage: %s <model.gguf> <depth[,depth...]> [rows] "
                "[accept]\n", argv[0]);
        return 2;
    }
    int rows = argc > 3 ? atoi(argv[3]) : 5;
    int k = argc > 4 ? atoi(argv[4]) : 2;
    if (rows < 1 || rows > BV_MAX_ROWS || k < 1 || k > rows) {
        fprintf(stderr, "rows in [1,%d], accept in [1,rows]\n", BV_MAX_ROWS);
        return 2;
    }
    pn_model *model = pn_load(argv[1]);
    if (!model) { fprintf(stderr, "load failed\n"); return 2; }
    int n_vocab = model->config.n_vocab;
    printf("batch verify exact: rows=%d accept=%d QWEN35_BATCH_VERIFY=%s BV_TOPK=%s\n",
           rows, k, getenv("QWEN35_BATCH_VERIFY") ? getenv("QWEN35_BATCH_VERIFY")
                                                 : "(default)",
           getenv("BV_TOPK") ? getenv("BV_TOPK") : "(argmax)");
    int total_bad = 0, failed = 0;
    for (const char *p = argv[2]; *p;) {
        char *end = NULL;
        int depth = (int)strtol(p, &end, 10);
        if (end == p) break;
        int bad = run_depth(model, depth, rows, k, n_vocab);
        if (bad < 0) { printf("RUN depth=%d FAILED\n", depth); failed = 1; }
        else total_bad += bad;
        fflush(stdout);
        p = (*end == ',') ? end + 1 : end;
    }
    long vc, vr, vf;
    if (pn_qwen35moe_verify_graph_stats(model, &vc, &vr, &vf) == 0)
        printf("verify graphs: captures=%ld replays=%ld failed=%ld\n", vc, vr, vf);
    int pass = !failed && total_bad == 0;
    if (pass) printf("GATE: BYTE-EXACT PASS\n");
    else printf("GATE: FAIL mismatched=%d run_failed=%d\n", total_bad, failed);
    fflush(stdout);
    if (!getenv("QWEN35_EXIT_CLEAN"))
        _Exit(pass ? 0 : 1);
    pn_model_free(model);
    return pass ? 0 : 1;
}
#endif
