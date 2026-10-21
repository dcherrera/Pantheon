/* SPDX-License-Identifier: AGPL-3.0-or-later */
/* Copyright (C) 2026 David (Corey) Herrera */
/**
 * @file qwen35moe_forward_metal.c
 * @brief Metal (Apple GPU) backend seam for the Qwen3.5-MoE forward. Phase 0
 *        scaffold: weight residency + the engine-selectable forward entry.
 *        No kernels yet — the forward reports NOT-IMPLEMENTED and the adapter
 *        routes the token to the CPU forward, so selecting this backend is
 *        output-neutral until Phase 1..5 fill the seam.
 *
 *        Structure mirrors qwen35moe_forward_cuda.c: an upload pass that makes
 *        every quantized weight GPU-addressable and attaches the handle to the
 *        weight (pn_qweight.gpu), plus a pn_forward_fn adapter selected by
 *        src/core/engine.c. What it does NOT mirror is the CUDA path's
 *        machinery: unified memory means there is no device allocation, no
 *        H2D/D2H of weights or state, and deliberately no MTLResidencySet
 *        (foundry/metal/qwen35moe_metal_spec.md §2 — wiring the working set was
 *        measured worth ~0.7% and costs the whole memory posture).
 *
 * @status Phase 0 scaffold. Upload + census implemented and measured; forward
 *         is a NOT-IMPLEMENTED seam that falls back to the CPU forward.
 *
 *         The upload and release walks now go through the model's own
 *         pn_qwen35moe_for_each_weight, shared with the kernel-coverage gate,
 *         so all three traverse one list. PN_METAL_COVERAGE=1 runs that gate at
 *         load (qwen35moe_metal_coverage.h) — off by default because it costs a
 *         CPU forward token.
 *
 *         Two compile-time switches exist ONLY to prove the acceptance checks
 *         can fail (never define them in a real build):
 *           PN_METAL_HARNESS_NO_FALLBACK  — the forward claims success and
 *             returns zero logits, so "Metal build output == CPU build output"
 *             is shown to be a real gate. Measured: output becomes "!!!!..."
 *             instead of the CPU answer.
 *           PN_METAL_HARNESS_COPY_LIMIT=N — bounds the unaligned-copy path to
 *             N weights so it can be exercised on a 15 GiB model, proving the
 *             census counters are measurements and not constants.
 * @issues **Zero-copy does not currently reach the weights.** fm_buffer_from_ptr
 *         only takes newBufferWithBytesNoCopy when the pointer is host-page
 *         aligned (16 KiB on Apple Silicon), but GGUF aligns tensor data to
 *         general.alignment (32 bytes) and the expert slices are interior
 *         offsets into merged tensors. Measured on
 *         nex-agi_Nex-N2-mini-Q3_K_M.gguf: 0 of 31,111 weights land page
 *         aligned. Copying instead would duplicate ~15 GiB of weights in a
 *         24 GiB machine, so unaligned weights are counted and SKIPPED, not
 *         copied (see PN_METAL_COPY_UNALIGNED_ENV to opt in for measurement).
 *         The fix is a backend design change, not a bug here: wrap ONE buffer
 *         over the page-aligned mmap region and bind per-tensor byte offsets.
 * @todo
 *   - [ ] Phase 1-5: fill pn_qwen35moe_forward_token_metal with the real
 *         kernel sequence; drop the CPU fallback in the adapter once it lands.
 *   - [ ] Re-do the upload against a single whole-mmap buffer + per-tensor
 *         offsets once fm_metal exposes an offset-bearing buffer view.
 */

/* 1. This module's own header */
#include "qwen35moe.h"

/* 2. Standard library */
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

/* 3. System / platform headers */
#ifdef USE_METAL
#include <unistd.h>
#endif

/* 4. Project headers */
#include "../core/qweight.h"
#include "qwen35moe_internal.h"
#include "qwen35moe_metal_coverage.h"

#ifdef USE_METAL

#include "metal/fm_metal.h"

/* Env knob: copy weights that are not host-page aligned into fresh Metal
 * buffers instead of skipping them. OFF by default and it must stay that way
 * for real models — on this file 15 GiB of weights miss page alignment, so the
 * "fallback copy" is not a fallback, it is a second copy of the whole model in
 * a 24 GiB machine. Exists so the copy path can be exercised deliberately on a
 * small model. */
#define PN_METAL_COPY_UNALIGNED_ENV "PN_METAL_COPY_UNALIGNED"

/* Env knob: run the kernel-coverage gate at model load. Off by default because
 * it costs one CPU forward token to trace the dispatch set. */
#define PN_METAL_COVERAGE_ENV "PN_METAL_COVERAGE"

/* How many distinct unaligned tensor names to name in the census log. The
 * count is the finding; the names are there to show WHICH kinds of tensors
 * miss, and 31k of them would just be noise. */
#define PN_METAL_CENSUS_SAMPLE_NAMES 8

/* ============================================================
 * Page-alignment census
 *
 * fm_buffer_from_ptr silently degrades from newBufferWithBytesNoCopy to a
 * copy when the pointer is not page aligned (foundry/metal/fm_metal.m:172-186).
 * Silently is the problem: the whole memory design of this backend is that the
 * weights stay the mmap'd file, so a degraded upload has to be visible in the
 * log, per tensor, or it looks exactly like a working one.
 * ============================================================ */

typedef struct {
    long page_size;             /* host page size the alignment test uses */
    unsigned long total;        /* weights walked */
    unsigned long zero_copy;    /* page aligned -> newBufferWithBytesNoCopy */
    unsigned long copied;       /* unaligned, copied because the env knob is on */
    unsigned long skipped;      /* unaligned, left un-uploaded (the default) */
    unsigned long already;      /* already had a buffer (idempotent re-run) */
    unsigned long failed;       /* fm_buffer_from_ptr returned NULL */
    uint64_t zero_copy_bytes;
    uint64_t copied_bytes;
    uint64_t skipped_bytes;
    int copy_unaligned;         /* PN_METAL_COPY_UNALIGNED */
    int n_sample;
    char sample[PN_METAL_CENSUS_SAMPLE_NAMES][96];
} pn_metal_upload_census;

/**
 * Record one unaligned weight's name in the bounded sample list.
 * Silently ignored once the sample is full — the counters carry the totals.
 */
static void census_sample_name(pn_metal_upload_census *c, const char *name)
{
    if (c->n_sample >= PN_METAL_CENSUS_SAMPLE_NAMES)
        return;
    snprintf(c->sample[c->n_sample], sizeof(c->sample[0]), "%s", name);
    c->n_sample++;
}

/**
 * Make one quantized weight GPU-addressable, idempotently.
 *
 * Mirrors pn_qmv_gpu_upload's contract (already-resident weights are a no-op,
 * a weight that cannot be uploaded is left alone rather than failing the whole
 * model), but takes the zero-copy path: the weight's bytes ARE the mmap'd GGUF,
 * so a page-aligned pointer is wrapped in place and no memory is duplicated.
 *
 * @param w     Weight to wrap; w->gpu receives the fm_buffer on success.
 * @param name  Diagnostic tensor name, used only for the census log.
 * @param c     Census accumulator (updated for every outcome).
 *
 * Thread-safe: no (mutates w and c)
 * Reentrant: no
 * Signal-safe: no
 */
static void metal_wrap_weight(pn_qweight *w, const char *name,
                              pn_metal_upload_census *c)
{
    if (!w || !w->data || w->rows <= 0 || w->row_stride == 0)
        return;

    c->total++;
    if (w->gpu) {
        c->already++;
        return;
    }

    size_t bytes = (size_t)w->rows * w->row_stride;
    int aligned = ((uintptr_t)w->data % (uintptr_t)c->page_size) == 0;

    if (!aligned) {
        census_sample_name(c, name);
        if (!c->copy_unaligned) {
            c->skipped++;
            c->skipped_bytes += bytes;
            return;
        }
#ifdef PN_METAL_HARNESS_COPY_LIMIT
        /* Harness proof only: bound the copy path so it can be exercised on a
         * 15 GiB model without allocating 15 GiB. Never defined in a real
         * build. */
        if (c->copied >= (unsigned long)PN_METAL_HARNESS_COPY_LIMIT) {
            c->skipped++;
            c->skipped_bytes += bytes;
            return;
        }
#endif
    }

    /* Cast away const: the buffer wraps the mmap'd bytes read-only from our
     * side, but Metal's API takes a mutable pointer. The mapping itself is
     * MAP_PRIVATE and nothing here writes through it. */
    fm_buffer *b = fm_buffer_from_ptr((void *)(uintptr_t)w->data, bytes);
    if (!b) {
        c->failed++;
        return;
    }
    w->gpu = b;
    if (aligned) {
        c->zero_copy++;
        c->zero_copy_bytes += bytes;
    } else {
        c->copied++;
        c->copied_bytes += bytes;
    }
}

/**
 * Release a weight's Metal buffer, if it has one. Symmetric with
 * metal_wrap_weight; safe on a weight that was never uploaded.
 */
static void metal_release_weight(pn_qweight *w)
{
    if (w && w->gpu) {
        fm_buffer_free((fm_buffer *)w->gpu);
        w->gpu = NULL;
    }
}

/* ============================================================
 * Weight walk
 *
 * The walk itself lives in the model (pn_qwen35moe_for_each_weight) so the
 * upload, the release and the kernel-coverage gate all traverse the SAME list.
 * A private copy here is how a newly added weight gets uploaded but not
 * released, or checked but not uploaded.
 * ============================================================ */

/** pn_qwen35moe_weight_fn adapter for the upload pass. */
static void metal_upload_one(const pn_qwen35moe_weight_ref *ref, void *user)
{
    metal_wrap_weight(ref->w, ref->name, (pn_metal_upload_census *)user);
}

/** pn_qwen35moe_weight_fn adapter for the release pass. */
static void metal_release_one(const pn_qwen35moe_weight_ref *ref, void *user)
{
    (void)user;
    metal_release_weight(ref->w);
}

/**
 * Print the census. Always at INFO volume (stderr, one block) — a degraded
 * upload is indistinguishable from a healthy one at runtime, so this is the
 * only place the difference is visible.
 */
static void metal_log_census(const pn_metal_upload_census *c)
{
    fprintf(stderr,
            "metal: qwen35moe weight upload — %lu weights, page=%ld B\n"
            "metal:   zero-copy (page aligned) : %lu (%.2f GiB)\n"
            "metal:   copied  (unaligned)      : %lu (%.2f GiB)\n"
            "metal:   skipped (unaligned)      : %lu (%.2f GiB)\n"
            "metal:   already resident         : %lu\n"
            "metal:   buffer creation failed   : %lu\n",
            c->total, c->page_size,
            c->zero_copy, (double)c->zero_copy_bytes / (1024.0 * 1024.0 * 1024.0),
            c->copied,    (double)c->copied_bytes    / (1024.0 * 1024.0 * 1024.0),
            c->skipped,   (double)c->skipped_bytes   / (1024.0 * 1024.0 * 1024.0),
            c->already, c->failed);

    if (c->skipped == 0 && c->copied == 0)
        return;

    for (int i = 0; i < c->n_sample; i++)
        fprintf(stderr, "metal:   unaligned e.g. %s\n", c->sample[i]);
    fprintf(stderr,
            "metal: NOTE — GGUF aligns tensor data to general.alignment "
            "(32 B by default) and expert slices are interior offsets, so "
            "page alignment is accidental. Until the backend binds one buffer "
            "over the whole mmap with per-tensor offsets, the zero-copy path "
            "in the spec is not reachable for this file.\n");
    if (!c->copy_unaligned)
        fprintf(stderr,
                "metal: unaligned weights were SKIPPED, not copied — copying "
                "would duplicate the model in host RAM. Set %s=1 to copy "
                "anyway (measurement only).\n", PN_METAL_COPY_UNALIGNED_ENV);
}

/* ============================================================
 * Public API
 * ============================================================ */

int pn_qwen35moe_metal_upload(pn_qwen35moe_model *model)
{
    if (!model)
        return -1;
    if (fm_init() != 0 || !fm_available()) {
        /* No Metal device at runtime: graceful skip, exactly like the CUDA
         * upload's no-GPU case. The engine keeps the CPU forward. */
        return -1;
    }

    const pn_qwen35moe_config *cfg = pn_qwen35moe_model_config(model);
    if (!cfg)
        return -1;

    const char *copy_env = getenv(PN_METAL_COPY_UNALIGNED_ENV);
    pn_metal_upload_census census = {0};
    census.page_size = sysconf(_SC_PAGESIZE);
    if (census.page_size <= 0)
        census.page_size = 4096;
    census.copy_unaligned = (copy_env && atoi(copy_env) != 0) ? 1 : 0;

    pn_qwen35moe_for_each_weight(model, metal_upload_one, &census);
    metal_log_census(&census);

    /* Debug knob: prove, at load, that a Metal matvec kernel exists for every
     * quant type this model will dispatch. Off by default because it runs a
     * full CPU forward token to trace the dispatch set. */
    const char *cov_env = getenv(PN_METAL_COVERAGE_ENV);
    if (cov_env && atoi(cov_env) != 0)
        pn_qwen35moe_metal_coverage_check(model, stderr);

    /* A degraded (copy / skip) upload is a reportable finding, not a failure:
     * the weights are still correct and the CPU path is untouched. Only a
     * missing device or a broken model handle is an error. */
    return 0;
}

void pn_qwen35moe_metal_free(pn_qwen35moe_model *model)
{
    pn_qwen35moe_for_each_weight(model, metal_release_one, NULL);
}

int pn_qwen35moe_forward_token_metal(pn_qwen35moe_model *model, uint32_t token,
                                     pn_qwen35moe_state *state, int pos,
                                     float **logits_out)
{
    (void)model; (void)token; (void)state; (void)pos;

    if (logits_out)
        *logits_out = NULL;

#ifdef PN_METAL_HARNESS_NO_FALLBACK
    /* Harness proof only: claim the forward ran and hand back a zero logits
     * row, so the A/B that asserts "Metal output == CPU output" is shown to
     * FAIL when the fallback is removed. Never defined in a real build. */
    if (state && state->logits) {
        memset(state->logits, 0, (size_t)state->n_vocab * sizeof(float));
        if (logits_out)
            *logits_out = state->logits;
        return PN_QWEN35MOE_METAL_OK;
    }
#endif

    /* Phase 0: no kernels exist. Reporting NOT-IMPLEMENTED (rather than
     * producing something) is what keeps the backend honest — the adapter
     * turns this into a CPU token, so selecting Metal today changes nothing
     * about the output. */
    return PN_QWEN35MOE_METAL_ENOTIMPL;
}

/* ============================================================
 * pn_forward_fn adapter (Metal)
 *
 * Mirrors pn_qwen35moe_forward_cuda's adapter shape. The one difference is the
 * NOT-IMPLEMENTED branch: the CUDA adapter has a real kernel path to dispatch
 * to, this one does not yet, so it delegates the whole token to the CPU
 * adapter. Delegating (rather than duplicating the pos/logits bookkeeping) is
 * what makes the fallback bit-identical to a non-Metal build by construction.
 * ============================================================ */

tm_tensor *pn_qwen35moe_forward_metal(pn_model *model, uint32_t token,
                                      pn_state *state)
{
    if (!model || !state)
        return NULL;

    pn_qwen35moe_model *m = (pn_qwen35moe_model *)model->weights;
    pn_qwen35moe_state *s = (pn_qwen35moe_state *)state->kv_cache;
    if (!m || !s)
        return NULL;

    float *logits = NULL;
    int rc = pn_qwen35moe_forward_token_metal(m, token, s, state->pos, &logits);

    if (rc == PN_QWEN35MOE_METAL_ENOTIMPL) {
        static int warned = 0;
        if (!warned) {
            warned = 1;
            fprintf(stderr, "metal: qwen35moe forward not implemented yet "
                            "(Phase 0 scaffold) — running the CPU forward\n");
        }
        return pn_qwen35moe_forward(model, token, state);
    }
    if (rc != PN_QWEN35MOE_METAL_OK || !logits)
        return NULL;

    int64_t shape[1] = { s->n_vocab };
    tm_tensor *out = tm_empty(shape, 1, TM_F32);
    if (!out)
        return NULL;
    memcpy(tm_data_f32(out), logits, (size_t)s->n_vocab * sizeof(float));

    state->pos++;
    return out;
}

#endif /* USE_METAL */
