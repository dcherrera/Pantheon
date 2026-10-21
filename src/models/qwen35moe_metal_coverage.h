/* SPDX-License-Identifier: AGPL-3.0-or-later */
/* Copyright (C) 2026 David (Corey) Herrera */
/**
 * @file qwen35moe_metal_coverage.h
 * @brief Kernel-coverage gate for the qwen35moe Metal backend: asserts that
 *        every weight the forward actually dispatches a kernel at has a Metal
 *        matvec kernel registered for its quant type, and reports the shortfall
 *        as tensors and bytes per type.
 *
 *        The MVP decodes the stock mixed-quant GGUF (Q3_K/Q4_K/Q5_K/Q6_K/Q8_0/
 *        F32 in one file — foundry/metal/qwen35moe_metal_spec.md §5). A missing
 *        or mis-registered kernel for any one of those does not fail loudly: it
 *        produces garbage logits at end-to-end bring-up, where it is easy to
 *        misattribute to a hard kernel like the SSM or attention. This gate
 *        turns that into an immediate, named failure.
 *
 *        WHY IT GATES ON DISPATCH, NOT ON FILE CONTENTS. A census of the GGUF's
 *        tensor types is wrong in both directions:
 *          - false alarm: 361 F32 tensors in the file are RMSNorm weights the
 *            forward consumes elementwise, and token_embd.weight (Q3_K, 218 MB)
 *            is dequantized wholesale at load. None is ever fed to a matvec.
 *          - false pass:  the 256 experts of a layer are slices of ONE merged
 *            tensor, so per-tensor counting hides how the forward reaches them.
 *        So the dispatch set is TRACED: the observer in src/core/qweight.h is
 *        installed at the matvec entry points and one real forward is run. What
 *        the forward dispatches is what the gate checks, by construction, and
 *        it keeps up when the forward changes.
 *
 * @status Expected to FAIL on the stock Q3_K_M file until the four Phase-1
 *         matvec kernels land (plan Phase 1). Failing is the point.
 * @issues None
 * @todo None
 */

#ifndef PN_QWEN35MOE_METAL_COVERAGE_H
#define PN_QWEN35MOE_METAL_COVERAGE_H

#include <stdint.h>
#include <stdio.h>

#include "../formats/gguf.h"
#include "../formats/gguf_dequant.h"
#include "qwen35moe_internal.h"

/* Distinct quant types a single model file can dispatch. The stock Q3_K_M file
 * uses 6; the cap is generous so an unexpected type lands in the table (and
 * fails the gate) rather than being dropped. */
#define PN_METAL_COVERAGE_MAX_TYPES 16

/* Names kept per finding, so a failure report can point at real tensors
 * without printing thousands of them. */
#define PN_METAL_COVERAGE_MAX_SAMPLES 8
#define PN_METAL_COVERAGE_NAME_MAX 128

/* One quant type's dispatch footprint, at GGUF-tensor granularity so the
 * numbers line up with the measured table in the spec. */
typedef struct {
    pn_gguf_quant type;
    char          type_name[PN_GGUF_QUANT_NAME_MAX];
    const char   *kernel_entry;     /* resolved foundry symbol, NULL if none */
    int           covered;          /* 1 when a kernel is registered */
    int           tensors;          /* dispatched GGUF tensors of this type */
    uint64_t      bytes;            /* their total bytes */
    int           weights;          /* dispatched pn_qweight slices */
    unsigned      ops;              /* 1u << pn_qweight_op, OR'd */
    char          example[PN_METAL_COVERAGE_NAME_MAX];  /* first such tensor */
} pn_qwen35moe_metal_coverage_row;

/* Full result of one gate run. */
typedef struct {
    pn_qwen35moe_metal_coverage_row rows[PN_METAL_COVERAGE_MAX_TYPES];
    int      n_rows;

    int      traced;                /* the forward trace ran */
    int      file_tensors;          /* tensors in the GGUF */
    uint64_t file_bytes;

    int      dispatched_tensors;    /* a pn_qweight in them was dispatched */
    uint64_t dispatched_bytes;
    int      dispatched_weights;

    /* Tensors no pn_qweight points into: consumed elementwise or dequantized
     * at load. Reported, never a failure — this is the false alarm a
     * file-contents census would raise. */
    int      excluded_tensors;
    uint64_t excluded_bytes;

    /* Tensors a pn_qweight DOES point into that the trace never dispatched.
     * A failure: either a live dispatch site the observer cannot see, or dead
     * weight. Either way it must not pass silently. */
    int      unaccounted_tensors;
    uint64_t unaccounted_bytes;
    char     unaccounted[PN_METAL_COVERAGE_MAX_SAMPLES][PN_METAL_COVERAGE_NAME_MAX];
    int      n_unaccounted_named;

    /* pn_qweights whose bytes lie outside every GGUF tensor (an owned
     * dequantized copy, or a walk/loader disagreement). Reported so the trace
     * can never be quietly incomplete. */
    int      unresolved_weights;

    int      uncovered_tensors;     /* dispatched, no kernel for the type */
    uint64_t uncovered_bytes;
} pn_qwen35moe_metal_coverage;

/**
 * Run the gate: enumerate the model's weights, trace one real forward to learn
 * the dispatch set, and resolve each dispatched type against the Metal kernel
 * registry.
 *
 * Runs exactly one forward token at position 0 on a scratch state, which is
 * enough: every layer runs every token, and the experts a layer routes to are
 * slices of that layer's own merged tensors, so one token reaches every tensor
 * the forward can dispatch.
 *
 * @param model  Loaded qwen35moe model
 * @param out    Receives the result (caller-owned, plain struct)
 * @return       0 on success, -1 if the model/GGUF is unusable or the trace
 *               could not run (which is itself a gate failure, not a pass)
 *
 * Thread-safe: no (installs the process-wide dispatch observer, runs a forward)
 * Reentrant: no
 * Signal-safe: no
 */
int pn_qwen35moe_metal_coverage_run(pn_qwen35moe_model *model,
                                    pn_qwen35moe_metal_coverage *out);

/**
 * Print the per-type coverage table plus every finding. Written to be readable
 * as a Phase-1 progress meter, not only as a pass/fail.
 */
void pn_qwen35moe_metal_coverage_report(const pn_qwen35moe_metal_coverage *cov,
                                        FILE *f);

/**
 * The verdict: 1 when every dispatched tensor's type has a registered kernel
 * and nothing is unaccounted for, else 0.
 */
int pn_qwen35moe_metal_coverage_passed(const pn_qwen35moe_metal_coverage *cov);

/**
 * Convenience: run + report + verdict. Used by the test and by the debug hook
 * in pn_qwen35moe_metal_upload (PN_METAL_COVERAGE=1).
 *
 * @return 1 if the gate passes, 0 if it fails or could not run
 */
int pn_qwen35moe_metal_coverage_check(pn_qwen35moe_model *model, FILE *f);

#endif /* PN_QWEN35MOE_METAL_COVERAGE_H */
