/* SPDX-License-Identifier: AGPL-3.0-or-later */
/* Copyright (C) 2026 David (Corey) Herrera */
/**
 * @file test_cuda_dense_slots.c
 * @brief Regression gate: a dense-FFN qwen35 model (Ornith, Bonsai) decodes through the
 *        batched decode-slots path the serve runs, and its greedy stream equals the
 *        single-token blocking path's.
 *
 * The batch scratch sized its MoE-only buffers by n_expert / n_expert_used, which are 0
 * for a dense FFN; a 0-byte alloc returns NULL, so the scratch failed and every
 * decode-slots step returned -1 ("[batch_sched] decode_slots failed (n=1)" on the first
 * chat request to Ornith 1.5 9B). The same scratch serves the batched prefill, which
 * fell back to one forward per token without a word.
 *
 * Two greedy arms on one chat prompt:
 *   token — prefill one token at a time with model->forward, decode with model->forward
 *           (the single-token path; what dense models ran before the fix);
 *   slots — batched prefill (pn_qwen35moe_forward_batch_cuda, chunks of 128), bind decode
 *           slot 0, one pn_qwen35moe_decode_slots step per token (the serve's path).
 * PASS when the slots arm runs and both streams are the same n_gen tokens. Fails on the
 * old code: the slots arm cannot take a step.
 *
 * Build (CUDA host): make USE_CUDA=1 tests/test_cuda_dense_slots
 * Run:   ./tests/test_cuda_dense_slots <dense-model.gguf> [n_gen=64]
 * Prints "GATE: PASS" or "GATE: FAIL ..." — exit 0 only on PASS.
 * NOT part of the suite (needs the model + a CUDA GPU).
 *
 * @status complete
 * @issues None
 * @todo None
 */
#include "../include/pantheon.h"
#include "../src/core/chat.h"
#include "../src/core/engine.h"
#include "../src/models/qwen35moe.h"
#include "tensor/tensor.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#ifndef USE_CUDA
int main(void) { fprintf(stderr, "built without USE_CUDA\n"); return 1; }
#else

#define DS_CHUNK   128
#define DS_MAX_GEN 1024

static const char *DS_PROMPT =
    "Write a C function that reverses a singly linked list in place, and explain it in "
    "two sentences.";

static uint32_t greedy(float *logits, int n_vocab)
{
    pn_sampling_params sp = pn_sampling_defaults();
    sp.temperature = 0.0f;
    sp.repeat_penalty = 1.0f;
    return pn_sample_next_token(logits, n_vocab, &sp, NULL, 0);
}

/* Single-token arm. Returns tokens written, or -1. */
static int run_token_arm(pn_model *model, const pn_token_seq *seq, int n_gen, uint32_t *out)
{
    int nv = model->config.n_vocab, n = 0;
    pn_state *st = pn_state_new(model);
    tm_tensor *logits = NULL;

    if (!st)
        return -1;
    for (int i = 0; i < seq->n_tokens; i++) {
        tm_free(logits);
        logits = model->forward(model, seq->tokens[i], st);
        if (!logits)
            goto fail;
    }
    while (n < n_gen) {
        uint32_t t = greedy(tm_data_f32(logits), nv);
        out[n++] = t;
        tm_free(logits);
        logits = model->forward(model, t, st);
        if (!logits)
            goto fail;
    }
    tm_free(logits);
    pn_state_free(st);
    return n;
fail:
    fprintf(stderr, "token arm: forward failed after %d tokens\n", n);
    pn_state_free(st);
    return -1;
}

/* Batched-prefill + decode-slots arm. Returns tokens written, or -1. */
static int run_slots_arm(pn_model *model, const pn_token_seq *seq, int n_gen, uint32_t *out)
{
    int nv = model->config.n_vocab, n = 0;
    uint32_t tok = 0;
    pn_state *st = pn_state_new(model);

    if (!st)
        return -1;
    for (int off = 0; off < seq->n_tokens; off += DS_CHUNK) {
        int c = seq->n_tokens - off < DS_CHUNK ? seq->n_tokens - off : DS_CHUNK;
        tm_tensor *o = pn_qwen35moe_forward_batch_cuda(model, seq->tokens + off, c, st, NULL);
        if (!o) {
            fprintf(stderr, "slots arm: prefill failed at %d\n", off);
            pn_state_free(st);
            return -1;
        }
        if (off + c >= seq->n_tokens)
            tok = greedy(tm_data_f32(o), nv);
        tm_free(o);
    }
    if (pn_qwen35moe_decode_slot_bind(model, st, 0, 1) != 0) {
        fprintf(stderr, "slots arm: bind failed\n");
        pn_state_free(st);
        return -1;
    }
    out[n++] = tok;
    while (n < n_gen) {
        float *logits[1];
        pn_state *sts[1] = { st };
        if (pn_qwen35moe_decode_slots(model, &tok, sts, 1, logits) != 0) {
            fprintf(stderr, "slots arm: decode_slots failed after %d tokens\n", n);
            pn_state_free(st);
            return -1;
        }
        tok = greedy(logits[0], nv);
        out[n++] = tok;
    }
    pn_state_free(st);
    return n;
}

int main(int argc, char **argv)
{
    static uint32_t ref[DS_MAX_GEN], got[DS_MAX_GEN];

    if (argc < 2) {
        fprintf(stderr, "usage: %s <dense-model.gguf> [n_gen=64]\n", argv[0]);
        return 2;
    }
    int n_gen = argc > 2 ? atoi(argv[2]) : 64;
    if (n_gen < 2 || n_gen > DS_MAX_GEN) {
        fprintf(stderr, "bad n_gen\n");
        return 2;
    }
    pn_model *model = pn_load(argv[1]);
    if (!model)
        return 2;
    if ((model->config.arch != PN_ARCH_BONSAI && model->config.arch != PN_ARCH_QWEN35MOE) ||
        model->config.n_experts != 0) {
        fprintf(stderr, "not a dense-FFN qwen35 model (arch %d, %d experts)\n",
                (int)model->config.arch, model->config.n_experts);
        return 2;
    }
    pn_token_seq seq = pn_chat_format_first((pn_chat_style)model->chat_style,
                                            model->tokenizer, NULL, DS_PROMPT);
    if (!seq.tokens)
        return 2;
    printf("dense slots: prompt %d tokens, n_gen %d\n", seq.n_tokens, n_gen);

    int n_ref = run_token_arm(model, &seq, n_gen, ref);
    int n_got = run_slots_arm(model, &seq, n_gen, got);
    int bad = 1;
    if (n_ref != n_gen || n_got != n_gen) {
        printf("GATE: FAIL (run failure: token %d, slots %d of %d)\n", n_ref, n_got, n_gen);
    } else {
        int d = -1;
        for (int i = 0; i < n_gen && d < 0; i++)
            if (ref[i] != got[i])
                d = i;
        if (d >= 0) {
            printf("GATE: FAIL (streams differ at %d: token %u, slots %u)\n", d, ref[d],
                   got[d]);
        } else {
            printf("GATE: PASS (%d identical greedy tokens)\n", n_gen);
            bad = 0;
        }
    }
    free(seq.tokens);
    fflush(stdout);
    _Exit(bad);
}
#endif
