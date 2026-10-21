/* SPDX-License-Identifier: AGPL-3.0-or-later */
/* Copyright (C) 2026 David (Corey) Herrera */
/*
 * test_2die_swap.c — GATING smoke test for the 2-die barswap context swap
 *
 * The BTC-S37 mining board exposes exactly ONE working MMIO slot, so barswap
 * maps ONE K80 die at a time. To run a model across both dies of a card we
 * time-multiplex: each die keeps its own VRAM allocations and its own CUDA
 * context resident, and bs_acquire(die) swaps which die is wired into the slot.
 * Because only one die is ever in the slot, EACH acquired die appears to CUDA
 * as device ordinal 0 — so we build two independent fc_cuda_ctx, both via
 * fc_cuda_init_device(0), each backed by a different physical die, and switch
 * between them by bs_acquire(die) + the ctx's own push-current (which every
 * fc_*_ctx op already does internally).
 *
 * The OPEN QUESTION this test answers: do two live CUDA contexts (one per die)
 * SURVIVE this acquire/release swapping? i.e. after we work on die A, switch to
 * die B, then switch BACK to die A — is die A's VRAM still intact and is its
 * context still usable for a kernel launch? compute_persist.c proved a SINGLE
 * die's VRAM + context survive a release/re-acquire; this proves it for TWO
 * coexisting contexts alternating in the one slot. This is the gate for the
 * whole 2-die layer-split approach (qwen35moe_forward_cuda.c) — if it fails,
 * that approach needs rethinking, so the output is deliberately diagnostic:
 * active die, buffer checksums and kernel results are printed at every step.
 *
 * Usage (on a K80 host, as root):
 *   make tests/test_2die_swap
 *   ./tests/test_2die_swap <dieA> <dieB>        # e.g. 0 1
 *
 * On a host with no CUDA driver (the Mac) it prints "[SKIP] no CUDA" and
 * exits 0, so it compiles, links and runs everywhere.
 *
 * Teardown discipline: like test_kernel_parity, CUDA's atexit
 * context teardown WEDGES a thread on this barswap rig, so after any CUDA use
 * we _Exit() and let the kernel reclaim the contexts on process death — we
 * never fc_free() / return normally once CUDA is live.
 *
 * @status HOST C only; SKIP path validated on the Mac. The real 2-die swap
 *         runs on a K80 host (operator runs this FIRST — it gates Task 2).
 * @issues None
 * @todo None
 */

#include "cuda/fc_cuda.h"

#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>

#include "barswap/barswap.h"

/* The activation-sized pattern each die holds. Small (single-token scale) so
 * the swap stays cheap, but large enough that a corrupted/stale VRAM page would
 * show up in the checksum. */
#define SWAP_N 2048

/* Bring nvidia up after a die is acquired (it's blacklisted; each submodule
 * needs --ignore-install). Mirrors compute_persist / test_kernel_parity. */
static int swap_load_nvidia(void)
{
    if (system("modprobe --ignore-install nvidia 2>/dev/null") != 0)
        return -1;
    sleep(1);
    if (system("modprobe --ignore-install nvidia_uvm 2>/dev/null") != 0)
        return -1;
    sleep(1);
    return 0;
}

/* Plain f32 checksum (sum) — cheap, order-stable for our element counts, and
 * enough to catch a stale/zeroed/wrong-die page. */
static double checksum(const float *x, int n)
{
    double s = 0.0;
    for (int i = 0; i < n; i++)
        s += x[i];
    return s;
}

/* Fill host buffer with a die-distinguishing ramp: base + i*step. The two dies
 * use different (base, step) so a buffer read back from the WRONG die is
 * obvious in the checksum. */
static void fill_pattern(float *x, int n, float base, float step)
{
    for (int i = 0; i < n; i++)
        x[i] = base + (float)i * step;
}

/* Run fc_silu on ctx over a buffer holding `in`, read the result back, and
 * return its checksum (also fills `out` with the n values). silu(x)=x*sigmoid(x)
 * is monotonic and non-trivial, so a wrong context / stale VRAM yields a
 * visibly wrong checksum. The buffer is allocated, used, and freed here; the
 * resident "kept" buffer is managed by the caller. */
static double silu_checksum(fc_cuda_ctx *ctx, const float *in, int n, float *out)
{
    fc_buffer *x = fc_buffer_from_ptr_ctx(ctx, in, (size_t)n * sizeof(float));
    fc_buffer *y = fc_buffer_alloc_ctx(ctx, (size_t)n * sizeof(float));
    if (!x || !y) {
        fc_buffer_free(x);
        fc_buffer_free(y);
        /* Sentinel that no real checksum can equal, so close_enough() flags it.
         * (-ffast-math makes NAN undefined, hence a large finite magic value.) */
        return -1e30;
    }
    fc_silu_ctx(ctx, x, y, n);
    fc_sync_ctx(ctx);
    fc_buffer_to_cpu(y, out);
    double cs = checksum(out, n);
    fc_buffer_free(x);
    fc_buffer_free(y);
    return cs;
}

/* Reference: what silu_checksum SHOULD produce for a given input, computed on
 * the host. Lets us assert the GPU kernel result, not just self-consistency. */
static double silu_checksum_ref(const float *in, int n)
{
    double s = 0.0;
    for (int i = 0; i < n; i++) {
        double v = in[i];
        s += v * (1.0 / (1.0 + exp(-v)));
    }
    return s;
}

/* Read a resident GPU buffer back and return its checksum. Used to prove die
 * A's VRAM is byte-intact after die B was active in the slot. */
static double resident_checksum(fc_buffer *buf, int n, float *scratch)
{
    fc_buffer_to_cpu(buf, scratch);
    return checksum(scratch, n);
}

/* Float compare with a tolerance scaled to the magnitude (fp32 reductions on
 * Kepler differ slightly from host double accumulation). */
static int close_enough(double a, double b)
{
    double denom = fabs(a) > 1.0 ? fabs(a) : 1.0;
    return fabs(a - b) / denom < 1e-4;
}

int main(int argc, char **argv)
{
    /* Unbuffered: a hung kernel must not swallow the last printed line (a
     * buffered run lost everything on a K80 wedge). */
    setvbuf(stdout, NULL, _IONBF, 0);
    setvbuf(stderr, NULL, _IONBF, 0);

    printf("=== 2-die barswap context-swap smoke test ===\n");

    if (argc < 3) {
        fprintf(stderr, "usage: %s <dieA> <dieB>\n", argv[0]);
        return 2;
    }
    int dieA = atoi(argv[1]);
    int dieB = atoi(argv[2]);
    if (dieA == dieB) {
        fprintf(stderr, "dieA and dieB must differ (got %d %d)\n", dieA, dieB);
        return 2;
    }
    printf("die A = %d   die B = %d   N = %d floats/buffer\n", dieA, dieB, SWAP_N);

    /* Gate on barswap, not CUDA: the Mac has no /dev/bar_swap → nothing to swap
     * → SKIP. CUDA init (cuInit) is deferred until AFTER a die is acquired and
     * nvidia is loaded (below) — cuInit needs a device present, and on a fresh
     * boot nvidia is blacklisted/unloaded until we modprobe it. */
    bs_context *bs = bs_init();
    if (!bs) {
        printf("[SKIP] no CUDA/barswap — host compile/link OK, 2-die swap "
               "deferred to a K80 host\n");
        return 0;
    }
    printf("[step] barswap up: %u dies\n", bs_num_dies(bs));

    /* Host patterns. Die A and die B use distinct ramps so a buffer read back
     * from the wrong die is unmistakable. */
    float *patA = malloc((size_t)SWAP_N * sizeof(float));
    float *patB = malloc((size_t)SWAP_N * sizeof(float));
    float *scratch = malloc((size_t)SWAP_N * sizeof(float));
    if (!patA || !patB || !scratch) {
        fprintf(stderr, "host alloc failed\n");
        _Exit(1);
    }
    fill_pattern(patA, SWAP_N, 0.25f, 0.001f);   /* die A's known pattern */
    fill_pattern(patB, SWAP_N, -1.5f, 0.002f);   /* die B's known pattern */
    double sumA = checksum(patA, SWAP_N);
    double sumB = checksum(patB, SWAP_N);
    double refA = silu_checksum_ref(patA, SWAP_N);
    double refB = silu_checksum_ref(patB, SWAP_N);
    printf("[ref] patA sum=%.6f silu=%.6f | patB sum=%.6f silu=%.6f\n",
           sumA, refA, sumB, refB);

    int fail = 0;

    /* ---- Step 1: acquire die A, init ctxA, upload + run a kernel, KEEP a
     * resident buffer holding patA. ---- */
    printf("\n[step 1] acquire die A (%d), init ctxA on ordinal 0\n", dieA);
    if (!bs_acquire(bs, dieA)) {
        fprintf(stderr, "bs_acquire(A=%d) failed\n", dieA);
        _Exit(1);
    }
    if (swap_load_nvidia() != 0) {
        fprintf(stderr, "nvidia load failed\n");
        _Exit(1);
    }
    printf("       active die now = %d\n", bs_active_die(bs));
    fc_cuda_ctx *ctxA = fc_cuda_init_device(0);
    if (!ctxA || !fc_cuda_ctx_available(ctxA)) {
        fprintf(stderr, "fc_cuda_init_device(0) for die A failed\n");
        _Exit(1);
    }
    /* Resident buffer: stays allocated in die A's VRAM for the whole test. */
    fc_buffer *residentA = fc_buffer_from_ptr_ctx(ctxA, patA,
                                                  (size_t)SWAP_N * sizeof(float));
    if (!residentA) {
        fprintf(stderr, "residentA alloc failed\n");
        _Exit(1);
    }
    double a1_resident = resident_checksum(residentA, SWAP_N, scratch);
    double a1_silu = silu_checksum(ctxA, patA, SWAP_N, scratch);
    printf("       ctxA: resident sum=%.6f (want %.6f) | silu=%.6f (want %.6f)\n",
           a1_resident, sumA, a1_silu, refA);
    if (!close_enough(a1_resident, sumA) || !close_enough(a1_silu, refA)) {
        printf("       [FAIL] die A initial kernel/resident wrong\n");
        fail = 1;
    }

    /* ---- Step 2: acquire die B, init ctxB, upload a DIFFERENT pattern, run a
     * kernel, KEEP a resident buffer holding patB. ---- */
    printf("\n[step 2] acquire die B (%d), init ctxB on ordinal 0\n", dieB);
    if (!bs_acquire(bs, dieB)) {
        fprintf(stderr, "bs_acquire(B=%d) failed\n", dieB);
        _Exit(1);
    }
    printf("       active die now = %d\n", bs_active_die(bs));
    fc_cuda_ctx *ctxB = fc_cuda_init_device(0);
    if (!ctxB || !fc_cuda_ctx_available(ctxB)) {
        fprintf(stderr, "fc_cuda_init_device(0) for die B failed\n");
        _Exit(1);
    }
    fc_buffer *residentB = fc_buffer_from_ptr_ctx(ctxB, patB,
                                                  (size_t)SWAP_N * sizeof(float));
    if (!residentB) {
        fprintf(stderr, "residentB alloc failed\n");
        _Exit(1);
    }
    double b1_resident = resident_checksum(residentB, SWAP_N, scratch);
    double b1_silu = silu_checksum(ctxB, patB, SWAP_N, scratch);
    printf("       ctxB: resident sum=%.6f (want %.6f) | silu=%.6f (want %.6f)\n",
           b1_resident, sumB, b1_silu, refB);
    if (!close_enough(b1_resident, sumB) || !close_enough(b1_silu, refB)) {
        printf("       [FAIL] die B initial kernel/resident wrong\n");
        fail = 1;
    }

    /* ---- Step 3 (THE KEY ASSERTION): swap BACK to die A. ctxA's resident
     * buffer must still hold patA byte-for-byte, and a fresh kernel on ctxA
     * must still produce silu(patA) — proving die A's VRAM + context survived
     * while die B owned the slot. ---- */
    printf("\n[step 3] acquire die A (%d) AGAIN — verify ctxA survived die B\n",
           dieA);
    if (!bs_acquire(bs, dieA)) {
        fprintf(stderr, "bs_acquire(A=%d) re-acquire failed\n", dieA);
        _Exit(1);
    }
    printf("       active die now = %d\n", bs_active_die(bs));
    double a2_resident = resident_checksum(residentA, SWAP_N, scratch);
    double a2_silu = silu_checksum(ctxA, patA, SWAP_N, scratch);
    printf("       ctxA: resident sum=%.6f (want %.6f) | silu=%.6f (want %.6f)\n",
           a2_resident, sumA, a2_silu, refA);
    if (!close_enough(a2_resident, sumA)) {
        printf("       [FAIL] die A VRAM did NOT survive die B being active\n");
        fail = 1;
    }
    if (!close_enough(a2_silu, refA)) {
        printf("       [FAIL] die A context could NOT run a kernel after swap\n");
        fail = 1;
    }

    /* ---- Step 4: swap to die B once more and verify it likewise survived A's
     * re-acquire. ---- */
    printf("\n[step 4] acquire die B (%d) AGAIN — verify ctxB survived\n", dieB);
    if (!bs_acquire(bs, dieB)) {
        fprintf(stderr, "bs_acquire(B=%d) re-acquire failed\n", dieB);
        _Exit(1);
    }
    printf("       active die now = %d\n", bs_active_die(bs));
    double b2_resident = resident_checksum(residentB, SWAP_N, scratch);
    double b2_silu = silu_checksum(ctxB, patB, SWAP_N, scratch);
    printf("       ctxB: resident sum=%.6f (want %.6f) | silu=%.6f (want %.6f)\n",
           b2_resident, sumB, b2_silu, refB);
    if (!close_enough(b2_resident, sumB)) {
        printf("       [FAIL] die B VRAM did NOT survive die A re-acquire\n");
        fail = 1;
    }
    if (!close_enough(b2_silu, refB)) {
        printf("       [FAIL] die B context could NOT run a kernel after swap\n");
        fail = 1;
    }

    printf("\n========================================\n");
    if (fail)
        printf("[FAIL] 2-die context swap — the layer-split approach needs "
               "rethinking (see per-step diagnostics above)\n");
    else
        printf("[PASS] 2-die context swap — two CUDA contexts coexist and "
               "survive barswap acquire/release; Task 2 is unblocked\n");
    printf("========================================\n");

    /* _Exit (not fc_free/return): skip the CUDA atexit teardown that wedges a
     * thread on this barswap rig. The kernel reclaims both contexts and the
     * acquired die on process death. Resident buffers are intentionally NOT
     * freed — process death reclaims them. */
    fflush(stdout);
    _Exit(fail ? 1 : 0);
}
