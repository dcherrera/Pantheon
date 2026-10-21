/* SPDX-License-Identifier: AGPL-3.0-or-later */
/* Copyright (C) 2026 David (Corey) Herrera */
/* Lifetime regression gate: bind a decode slot immediately after a session
 * free.
 *
 * THE BUG THIS EXISTS FOR. gpu_attn_layer holds a raw per-slot back-pointer
 * into a session's state (kv_slot_kv). pn_qwen35moe_state_free released the
 * state without telling the GPU layer, so the NEXT bind into that slot
 * dereferenced freed memory in its lazy-mirror "park copy"
 * (qwen35moe_forward_cuda.c, pn_qwen35moe_decode_slot_bind_cuda). ASAN caught
 * it as a heap-use-after-free; without ASAN the freed 880 bytes read as
 * whatever was recycled into them, and the park copy issued a device-to-host
 * copy through pointers taken out of the dead struct — the `movntps` fault
 * inside libcuda, and the `free(): invalid next size` abort, both reported
 * from other harnesses. One bug, several faces.
 *
 * The serve does exactly this sequence as ordinary operation: batch_sched
 * frees a slot's state on every session release / idle reap and binds a new
 * session into that slot afterwards.
 *
 * WHAT IS ASSERTED, and why it is red without the fix even without a
 * sanitizer: after pn_state_free, NO decode slot may still hold a
 * back-pointer inside the freed state's kv array.
 * pn_qwen35moe_gpu_slot_kv_refs_cuda answers that by comparing addresses and
 * never dereferencing, so the check itself is legal on a freed range — unlike
 * the defect it guards, which read the freed object to decide whether the
 * object was freed. Round 2 then performs the real bind-after-free, which is
 * where ASAN trips on an unfixed build.
 *
 * Build (CUDA host): make DEBUG=1 USE_CUDA=1 tests/test_slot_free_bind_qwen35
 * Run: QWEN35_GPU_SPLIT_LAYER=20 QWEN35_GPU_DIE_A=6 QWEN35_GPU_DIE_B=7 \
 *        QWEN35_FUSE=0x37 ./tests/test_slot_free_bind_qwen35 <model.gguf>
 * Exit 0 = PASS. NOT part of the CPU test suite (needs the model + a CUDA GPU).
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

#define PROMPT_LEN 12
#define N_STEPS    4

static int argmax(const float *x, int n)
{
    int best = 0;
    for (int i = 1; i < n; i++)
        if (x[i] > x[best]) best = i;
    return best;
}

/* One session's whole life on the batched path: new -> prefill -> bind slot ->
 * a few batched decode steps (which is what populates the per-slot
 * bookkeeping the bug leaks) -> free. Records the kv-array range of the state
 * just before the free so the caller can assert nothing still points into it.
 * Returns 0 on success. */
static int session_round(pn_model *model, int n_vocab, int round,
                         const void **kv_base, size_t *kv_bytes)
{
    uint32_t prompt[PROMPT_LEN];
    for (int i = 0; i < PROMPT_LEN; i++)
        prompt[i] = (uint32_t)((100 + i * 7 + round * 29) % n_vocab);

    pn_state *st = pn_state_new(model);
    if (!st) {
        fprintf(stderr, "round %d: state_new failed\n", round);
        return -1;
    }

    tm_tensor *out = pn_qwen35moe_forward_batch_cuda(model, prompt, PROMPT_LEN,
                                                     st, NULL);
    if (!out) {
        fprintf(stderr, "round %d: prefill failed\n", round);
        pn_state_free(st);
        return -1;
    }
    uint32_t tok = (uint32_t)argmax(tm_data_f32(out), n_vocab);
    tm_free(out);

    /* THE BIND. On round 2 this is the bind-after-free: slot 0 still names
     * round 1's (freed) session unless the free path cleared it. */
    if (pn_qwen35moe_decode_slot_bind(model, st, 0, 1) != 0) {
        fprintf(stderr, "round %d: bind failed\n", round);
        pn_state_free(st);
        return -1;
    }

    pn_state *sts[1] = { st };
    float *logits[1];
    for (int i = 0; i < N_STEPS; i++) {
        if (pn_qwen35moe_decode_slots(model, &tok, sts, 1, logits) != 0) {
            fprintf(stderr, "round %d: decode_slots failed @step %d\n",
                    round, i);
            pn_state_free(st);
            return -1;
        }
        tok = (uint32_t)argmax(logits[0], n_vocab);
    }

    /* The range the per-slot back-pointers land in, captured while the state
     * is still alive. Only its ADDRESS is used after the free. */
    const pn_qwen35moe_state *qs = (const pn_qwen35moe_state *)st->kv_cache;
    *kv_base  = (const void *)qs->kv;
    *kv_bytes = (size_t)qs->n_attn * sizeof(*qs->kv);

    pn_state_free(st);
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
    int n_vocab = model->config.n_vocab;
    int fail = 0;

    const void *kv_base = NULL;
    size_t kv_bytes = 0;

    /* Round 1: a normal session, bound and decoded, then freed. */
    if (session_round(model, n_vocab, 1, &kv_base, &kv_bytes) != 0) {
        fprintf(stderr, "SLOTFREE: round 1 could not run\n");
        fflush(stderr);
        _Exit(1);
    }
    if (!kv_base || !kv_bytes) {
        fprintf(stderr, "SLOTFREE: no kv range captured (test is inert)\n");
        fflush(stderr);
        _Exit(1);
    }

    /* Assertion 1 — the invariant, checked without touching freed memory. */
    int refs = pn_qwen35moe_gpu_slot_kv_refs_cuda(kv_base, kv_bytes);
    printf("SLOTFREE dangling_slot_kv_refs=%d\n", refs);
    if (refs != 0) {
        fprintf(stderr,
                "SLOTFREE FAIL: %d decode slot(s) still point into the freed "
                "session state\n", refs);
        fail = 1;
    }

    /* Assertion 2 — the real thing: bind slot 0 again right after that free.
     * This is the line that faulted (the lazy-mirror park copy). It must
     * complete, and under ASAN it must complete without a use-after-free. */
    const void *kv_base2 = NULL;
    size_t kv_bytes2 = 0;
    if (session_round(model, n_vocab, 2, &kv_base2, &kv_bytes2) != 0) {
        fprintf(stderr, "SLOTFREE FAIL: bind-after-free round did not run\n");
        fail = 1;
    } else if (pn_qwen35moe_gpu_slot_kv_refs_cuda(kv_base2, kv_bytes2) != 0) {
        fprintf(stderr, "SLOTFREE FAIL: round 2 left a dangling slot ref\n");
        fail = 1;
    }

    printf("SLOTFREE %s\n", fail ? "FAIL" : "PASS");
    fflush(stdout);
    fflush(stderr);
    /* Same teardown convention as the other qwen35moe CUDA harnesses: CUDA
     * context teardown wedges on the K80, so exit without unwinding.
     * QWEN35_EXIT_CLEAN=1 returns normally instead, which is what lets a
     * sanitizer run its end-of-process checks. */
    if (!getenv("QWEN35_EXIT_CLEAN"))
        _Exit(fail ? 1 : 0);
    return fail ? 1 : 0;
}
#endif
