/* SPDX-License-Identifier: AGPL-3.0-or-later */
/* Copyright (C) 2026 David (Corey) Herrera */
/* Exactness gate for the turn-start checkpoint (pn_qwen35moe_slot_checkpoint_*,
 * the serve's rung 1c).
 *
 * The serve saves a session's state where the request's own messages end,
 * lets the turn decode in the session's block, and — when the client drops
 * that turn — copies the saved state back and prefills the next request from
 * there. That is only correct if the restored session is bit-for-bit the
 * session that never decoded the turn.
 *
 * Path A (the checkpoint): prefill P[0, C) -> save -> prefill the 3-token
 * opener -> bind row 0 -> decode DECODE_N tokens (more than the 256-token q4
 * fp16 window, so every shadow entry is overwritten) -> restore -> prefill the
 * tail T -> logits LA -> bind -> GREEDY_N greedy tokens.
 * Path B (no turn): a fresh state: prefill P[0, C) -> prefill T -> logits LB
 * -> bind row 1 -> GREEDY_N greedy tokens.
 * PASS = LA == LB bytewise and the greedy tokens agree.
 *
 * Falsifier: TCKPT_SKIP_RESTORE=1 moves the position back WITHOUT the restore
 * (the recurrent state still holds the decoded turn); the gate must fail.
 *
 * Build (CUDA host): make USE_CUDA=1 tests/test_turn_checkpoint_qwen35
 * Run: QWEN35_GPU_DIES=4,5,6,7 QWEN35_MAX_SEQ=4096 \
 *        ./tests/test_turn_checkpoint_qwen35 <model.gguf>
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

#define PROMPT_LEN 600   /* client messages, then the 3-token opener */
#define CUT        597
#define TAIL_LEN   40
#define DECODE_N   300   /* > PN_KVQ4_WMAX: the window is fully overwritten */
#define GREEDY_N   8

static int argmax(const float *x, int n)
{
    int best = 0;
    for (int i = 1; i < n; i++)
        if (x[i] > x[best]) best = i;
    return best;
}

/* Prefill `n` tokens into `st`; copy the last position's logits into `out`
 * (may be NULL). 0 or -1. */
static int prefill(pn_model *model, pn_state *st, const uint32_t *tok, int n,
                   float *out, int n_vocab)
{
    tm_tensor *t = pn_qwen35moe_forward_batch_cuda(model, tok, n, st, NULL);
    if (!t)
        return -1;
    if (out)
        memcpy(out, tm_data_f32(t), (size_t)n_vocab * sizeof(float));
    tm_free(t);
    return 0;
}

/* Bind `st` to `row` and decode `n` tokens greedily from `first`; the tokens
 * stepped land in `toks` when it is not NULL. 0 or -1. */
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
        if (toks)
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
    const int skip_restore = getenv("TCKPT_SKIP_RESTORE") != NULL;
    uint32_t prompt[PROMPT_LEN], tail[TAIL_LEN];
    for (int i = 0; i < PROMPT_LEN; i++)
        prompt[i] = (uint32_t)((1000 + i * 37) % n_vocab);
    for (int i = 0; i < TAIL_LEN; i++)
        tail[i] = (uint32_t)((5000 + i * 53) % n_vocab);
    float *la = malloc((size_t)n_vocab * sizeof(float));
    float *lb = malloc((size_t)n_vocab * sizeof(float));
    float *scratch = malloc((size_t)n_vocab * sizeof(float));
    uint32_t ga[GREEDY_N], gb[GREEDY_N];

    pn_state *st = pn_state_new(model);
    pn_state *st2 = pn_state_new(model);
    if (!st || !st2 || !la || !lb || !scratch ||
        pn_qwen35moe_decode_slots_carve(model, st, 2) != 0 ||
        pn_qwen35moe_slot_checkpoints_carve(model, st, 2) != 0) {
        fprintf(stderr, "TCKPT: setup failed\n");
        _Exit(1);
    }

    /* Path A. */
    int rc = prefill(model, st, prompt, CUT, NULL, n_vocab);
    if (rc == 0 && pn_qwen35moe_slot_checkpoint_save(model, st, 0) != 0) {
        fprintf(stderr, "TCKPT: save refused\n");
        rc = -1;
    }
    if (rc == 0)
        rc = prefill(model, st, prompt + CUT, PROMPT_LEN - CUT, scratch, n_vocab);
    if (rc == 0)
        rc = decode(model, st, 0, (uint32_t)argmax(scratch, n_vocab), DECODE_N,
                    NULL, n_vocab);
    if (rc == 0) {
        if (skip_restore) {
            st->pos = CUT;   /* the falsifier: position back, state not */
        } else if (pn_qwen35moe_slot_checkpoint_restore(model, st, 0, CUT) != 0) {
            fprintf(stderr, "TCKPT: restore refused\n");
            rc = -1;
        }
        pn_qwen35moe_gpu_forget_state(model, st);
    }
    if (rc == 0 && (int)st->pos != CUT) {
        fprintf(stderr, "TCKPT: restore left pos %d\n", (int)st->pos);
        rc = -1;
    }
    /* A restore at a position the checkpoint was not taken at is refused. */
    if (rc == 0 && !skip_restore &&
        pn_qwen35moe_slot_checkpoint_restore(model, st, 0, CUT - 1) == 0) {
        fprintf(stderr, "TCKPT: a wrong-position restore was accepted\n");
        rc = -1;
    }
    if (rc == 0)
        rc = prefill(model, st, tail, TAIL_LEN, la, n_vocab);
    if (rc == 0)
        rc = decode(model, st, 0, (uint32_t)argmax(la, n_vocab), GREEDY_N, ga,
                    n_vocab);

    /* Path B. */
    if (rc == 0)
        rc = prefill(model, st2, prompt, CUT, NULL, n_vocab);
    if (rc == 0)
        rc = prefill(model, st2, tail, TAIL_LEN, lb, n_vocab);
    if (rc == 0)
        rc = decode(model, st2, 1, (uint32_t)argmax(lb, n_vocab), GREEDY_N, gb,
                    n_vocab);
    if (rc != 0) {
        fprintf(stderr, "TCKPT FAIL: a step did not run\n");
        fflush(stderr);
        _Exit(1);
    }

    int logits_same = memcmp(la, lb, (size_t)n_vocab * sizeof(float)) == 0;
    int tokens_same = memcmp(ga, gb, sizeof(ga)) == 0;
    int differ = 0;
    for (int i = 0; i < n_vocab; i++)
        differ += la[i] != lb[i];
    printf("TCKPT%s logits %s (%d of %d differ), greedy tokens %s:",
           skip_restore ? " [falsifier: no restore]" : "",
           logits_same ? "IDENTICAL" : "DIFFER", differ, n_vocab,
           tokens_same ? "same" : "differ");
    for (int i = 0; i < GREEDY_N; i++)
        printf(" %u/%u", ga[i], gb[i]);
    printf("\nTCKPT %s\n", logits_same && tokens_same ? "PASS" : "FAIL");
    fflush(stdout);
    /* CUDA context teardown wedges on the K80: exit without unwinding. */
    _Exit(logits_same && tokens_same ? 0 : 1);
}
#endif
