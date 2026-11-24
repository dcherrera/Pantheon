/* SPDX-License-Identifier: AGPL-3.0-or-later */
/* Copyright (C) 2026 David (Corey) Herrera */
/* Exactness gate for keeping an aborted prefill (the serve's
 * bs_partial_prefill_keep / rung 1c aborted-prefill form).
 *
 * A client that hangs up mid-prefill leaves the session's state wherever the
 * pipelined prefill stopped; the serve now keeps that state and the resent
 * request continues from it. That is only correct if the kept state is
 * bit-for-bit the state of an uninterrupted prefill of the same prefix, at the
 * position the engine reports.
 *
 * Path A (abort): the wide pipelined prefill of P[0, N) with an abort probe
 * that fires after ABORT_AFTER polls -> it reports an abort and state->pos ->
 * save into row 0 the way the serve does (bind, forget) -> the pipelined
 * prefill of P[pos, N) -> logits LA -> bind -> GREEDY_N greedy tokens.
 * Path B (no abort): a fresh state, the pipelined prefill of P[0, N) -> LB ->
 * bind row 1 -> GREEDY_N greedy tokens.
 * PASS = LA == LB bytewise and the greedy tokens agree.
 *
 * Falsifier: PABORT_POS_LAG=1 moves state->pos one chunk back after the abort
 * (what the engine reported before 2026-11-03: the last in-flight chunk had
 * run but was not counted); the gate must fail.
 *
 * Build (CUDA host): make USE_CUDA=1 tests/test_prefill_abort_resume_qwen35
 * Run: QWEN35_GPU_DIES=4,5,6,7 QWEN35_MAX_SEQ=4096 \
 *        ./tests/test_prefill_abort_resume_qwen35 <model.gguf>
 * Exit 0 = PASS. NOT part of the CPU test suite (needs the model + a CUDA GPU).
 */
#include "../include/pantheon.h"
#include "../src/core/engine.h"
#include "../src/models/qwen35moe.h"
#include "tensor/tensor.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#ifndef USE_CUDA
int main(void)
{
    fprintf(stderr, "built without USE_CUDA\n");
    return 1;
}
#else

#define PROMPT_LEN  1500
#define CHUNK       128
#define ABORT_AFTER 5    /* polls before the probe fires: mid-range, chunks in flight */
#define GREEDY_N    8

static int argmax(const float *x, int n)
{
    int best = 0;
    for (int i = 1; i < n; i++)
        if (x[i] > x[best]) best = i;
    return best;
}

/* The abort probe: fires on the ABORT_AFTER-th poll. */
static int probe_after(void *ud)
{
    int *polls = (int *)ud;
    return ++(*polls) >= ABORT_AFTER;
}

/* Pipelined prefill of tok[0, n) into `st`; the last logits into `out`.
 * Returns the engine's rc (0 done, 1 aborted, -1 refused/failed). */
static int pipelined(pn_model *model, pn_state *st, const uint32_t *tok, int n,
                     int (*probe)(void *), void *ud, float *out, int n_vocab)
{
    tm_tensor *t = NULL;
    int rc = pn_qwen35moe_prefill_range_pipelined_cuda(model, tok, n, st, CHUNK,
                                                       probe, ud, &t);
    if (rc == 0 && t && out)
        memcpy(out, tm_data_f32(t), (size_t)n_vocab * sizeof(float));
    if (t)
        tm_free(t);
    return rc;
}

/* Bind `st` to `row` and decode `n` tokens greedily from `first`. 0 or -1. */
static int decode(pn_model *model, pn_state *st, int row, uint32_t first, int n,
                  uint32_t *toks, int n_vocab)
{
    if (pn_qwen35moe_decode_slot_bind(model, st, row, 2) != 0)
        return -1;
    pn_state *sts[1] = { st };
    int rows[1] = { row };
    float *logits[1];
    uint32_t tok = first;
    for (int i = 0; i < n; i++) {
        toks[i] = tok;
        if (pn_qwen35moe_decode_slots_rows(model, &tok, sts, rows, 1, logits) != 0)
            return -1;
        tok = (uint32_t)argmax(logits[0], n_vocab);
    }
    return 0;
}

int main(int argc, char **argv)
{
    if (argc < 2) {
        fprintf(stderr, "usage: %s <model.gguf>\n", argv[0]);
        return 1;
    }
    pn_model *model = pn_load(argv[1]);
    if (!model) {
        fprintf(stderr, "load failed\n");
        return 1;
    }
    const int n_vocab = model->config.n_vocab;
    const int lag = getenv("PABORT_POS_LAG") != NULL;
    uint32_t prompt[PROMPT_LEN];
    for (int i = 0; i < PROMPT_LEN; i++)
        prompt[i] = (uint32_t)((1000 + i * 37) % n_vocab);
    float *la = malloc((size_t)n_vocab * sizeof(float));
    float *lb = malloc((size_t)n_vocab * sizeof(float));
    uint32_t ga[GREEDY_N], gb[GREEDY_N];
    pn_state *st = pn_state_new(model);
    pn_state *st2 = pn_state_new(model);
    if (!st || !st2 || !la || !lb ||
        pn_qwen35moe_decode_slots_carve(model, st, 2) != 0) {
        fprintf(stderr, "PABORT: setup failed\n");
        _Exit(1);
    }

    /* Path A. */
    int polls = 0;
    int rc = pipelined(model, st, prompt, PROMPT_LEN, probe_after, &polls, NULL,
                       n_vocab);
    int kept = (int)st->pos;
    printf("PABORT abort rc=%d after %d polls: kept %d of %d tokens\n", rc, polls,
           kept, PROMPT_LEN);
    if (rc != 1 || kept <= 0 || kept >= PROMPT_LEN || kept % CHUNK != 0) {
        fprintf(stderr, "PABORT FAIL: the abort did not stop mid-range on a "
                "chunk boundary\n");
        _Exit(1);
    }
    if (lag)
        st->pos = kept - CHUNK;   /* the falsifier: the pre-fix count */
    /* The serve's keep: save into the row, then the resume forgets the GPU
     * bindings so the next prefill re-seeds from the row. */
    rc = pn_qwen35moe_decode_slot_bind(model, st, 0, 2);
    pn_qwen35moe_gpu_forget_state(model, st);
    int from = (int)st->pos;
    if (rc == 0)
        rc = pipelined(model, st, prompt + from, PROMPT_LEN - from, NULL, NULL,
                       la, n_vocab);
    if (rc == 0)
        rc = decode(model, st, 0, (uint32_t)argmax(la, n_vocab), GREEDY_N, ga,
                    n_vocab);

    /* Path B. */
    if (rc == 0)
        rc = pipelined(model, st2, prompt, PROMPT_LEN, NULL, NULL, lb, n_vocab);
    if (rc == 0)
        rc = decode(model, st2, 1, (uint32_t)argmax(lb, n_vocab), GREEDY_N, gb,
                    n_vocab);
    if (rc != 0) {
        fprintf(stderr, "PABORT FAIL: a step did not run (rc %d)\n", rc);
        fflush(stderr);
        _Exit(1);
    }

    int logits_same = memcmp(la, lb, (size_t)n_vocab * sizeof(float)) == 0;
    int tokens_same = memcmp(ga, gb, sizeof(ga)) == 0;
    int differ = 0;
    for (int i = 0; i < n_vocab; i++)
        differ += la[i] != lb[i];
    printf("PABORT%s logits %s (%d of %d differ), greedy tokens %s:",
           lag ? " [falsifier: pos one chunk back]" : "",
           logits_same ? "IDENTICAL" : "DIFFER", differ, n_vocab,
           tokens_same ? "same" : "differ");
    for (int i = 0; i < GREEDY_N; i++)
        printf(" %u/%u", ga[i], gb[i]);
    printf("\nPABORT %s\n", logits_same && tokens_same ? "PASS" : "FAIL");
    fflush(stdout);
    /* CUDA context teardown wedges on the K80: exit without unwinding. */
    _Exit(logits_same && tokens_same ? 0 : 1);
}
#endif
