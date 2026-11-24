/* SPDX-License-Identifier: AGPL-3.0-or-later */
/* Copyright (C) 2026 David (Corey) Herrera */
/**
 * @file test_qwen35_slots_fused.c
 * @brief One-slot fused launches on the batched-decode path: engaged, and bit-exact.
 * @status complete
 * @issues none
 * @todo none
 *
 * At one slot the decode-slots path runs the single-token path's fused
 * descriptors (proj4, ssm_out + residual, attn_output + residual, MoE pre and
 * post; gpu_slots_fused_prepare). This test decodes a fixed prompt at K = 1,
 * prints one FNV-1a over every step's presented logits and the token ids, and
 *   - with no argument requires every descriptor built (the fused path really
 *     ran: a build that silently fell back would compare equal to itself);
 *   - with --any only reports the count (for the QWEN35_FUSE=0 run).
 * tests/run_slots_fused_k80.sh runs both and requires the same digest: the
 * fused step must be bit-identical to the unfused slots sequence.
 *
 * Build (CUDA host): make USE_CUDA=1 tests/test_qwen35_slots_fused
 * Run: tests/run_slots_fused_k80.sh <model.gguf> (env picks the dies, e.g.
 *      CUDA_VISIBLE_DEVICES=8,9,10,11 QWEN35_GPU_DIES=0,1,2,3).
 * NOT part of the CPU test suite (needs the model + a CUDA GPU).
 */
#include "../include/pantheon.h"
#include "../src/core/engine.h"
#include "../src/models/qwen35moe.h"
#include "../src/models/qwen35moe_internal.h"
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

#define PROMPT_LEN 96
#define N_STEPS    32

static int argmax(const float *x, int n)
{
    int best = 0;
    for (int i = 1; i < n; i++)
        if (x[i] > x[best]) best = i;
    return best;
}

static uint64_t fnv1a(uint64_t h, const void *p, size_t n)
{
    const unsigned char *b = p;
    for (size_t i = 0; i < n; i++) {
        h ^= b[i];
        h *= 1099511628211ULL;
    }
    return h;
}

int main(int argc, char **argv)
{
    if (argc < 2) {
        fprintf(stderr, "usage: %s <model.gguf> [--any]\n", argv[0]);
        return 1;
    }
    int any = argc > 2 && strcmp(argv[2], "--any") == 0;
    pn_model *model = pn_load(argv[1]);
    if (!model) {
        fprintf(stderr, "load failed\n");
        return 1;
    }
    int n_vocab = model->config.n_vocab;

    uint32_t prompt[PROMPT_LEN];
    for (int i = 0; i < PROMPT_LEN; i++)
        prompt[i] = (uint32_t)((300 + i * 7919) % 60000);
    pn_state *st = pn_state_new(model);
    tm_tensor *out = st ? pn_qwen35moe_forward_batch_cuda(model, prompt, PROMPT_LEN,
                                                          st, NULL) : NULL;
    if (!out) {
        fprintf(stderr, "prefill failed\n");
        return 1;
    }
    uint32_t tok = (uint32_t)argmax(tm_data_f32(out), n_vocab);
    tm_free(out);
    if (pn_qwen35moe_decode_slot_bind(model, st, 0, 1) != 0) {
        fprintf(stderr, "bind failed\n");
        return 1;
    }

    uint64_t dg = 1469598103934665603ULL;
    pn_state *sts[1] = { st };
    printf("TOKENS");
    for (int i = 0; i < N_STEPS; i++) {
        float *logits[1];
        if (pn_qwen35moe_decode_slots(model, &tok, sts, 1, logits) != 0) {
            fprintf(stderr, "decode_slots failed @step %d\n", i);
            return 1;
        }
        dg = fnv1a(dg, logits[0], (size_t)n_vocab * sizeof(float));
        tok = (uint32_t)argmax(logits[0], n_vocab);
        printf(" %u", tok);
    }
    printf("\n");

    int possible = 0;
    int built = pn_qwen35moe_gpu_slots_fused_count_cuda(&possible);
    printf("SLOTSFUSED built=%d of %d steps=%d logits_fnv1a=%016llx\n", built,
           possible, N_STEPS, (unsigned long long)dg);
    pn_state_free(st);
    pn_model_free(model);

    if (!any && (possible == 0 || built != possible)) {
        printf("FAIL: one-slot fused descriptors not all built (%d of %d)\n",
               built, possible);
        return 1;
    }
    printf("PASS\n");
    return 0;
}
#endif
