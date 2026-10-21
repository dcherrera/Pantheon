/* SPDX-License-Identifier: AGPL-3.0-or-later */
/* Copyright (C) 2026 David (Corey) Herrera */
/**
 * @file qwen35moe_metal_coverage.c
 * @brief Implementation of the qwen35moe Metal kernel-coverage gate.
 *
 *        Three passes, in this order:
 *          1. index every GGUF tensor by its byte range, so a weight pointer
 *             can be resolved back to the tensor that contains it (expert
 *             slices are interior offsets into one merged tensor, so pointer
 *             containment is the only way to recover tensor identity);
 *          2. enumerate the model's pn_qweights (pn_qwen35moe_for_each_weight)
 *             and mark the tensors they bind;
 *          3. install the dispatch observer, run ONE real forward token, and
 *             mark the tensors the forward actually dispatched.
 *
 *        A tensor then falls into exactly one class: dispatched (needs a
 *        kernel), bound-but-never-dispatched (unaccounted — a failure), or
 *        bound by nothing (consumed elementwise / dequantized at load —
 *        correctly excluded, and the false alarm a file-contents census
 *        would raise).
 *
 * @status Phase 1 gate. Expected to fail on the stock Q3_K_M file until the
 *         four new matvec kernels land.
 * @issues None
 * @todo None
 */

/* 1. This module's own header */
#include "qwen35moe_metal_coverage.h"

/* 2. Standard library */
#include <stdlib.h>
#include <string.h>

/* 4. Project headers */
#include "../core/qweight.h"
#include "qwen35moe.h"
#include "qwen35moe_metal_kernel_registry.h"

#define GIB (1024.0 * 1024.0 * 1024.0)

/* One GGUF tensor's byte range plus what the model does with it. */
typedef struct {
    const uint8_t *start;
    uint64_t       bytes;
    const char    *name;        /* borrowed from the GGUF (NUL-terminated) */
    pn_gguf_quant  type;
    int            n_bound;     /* pn_qweights whose bytes lie inside */
    int            n_dispatched;/* of those, ones the forward dispatched */
    unsigned       ops;         /* 1u << pn_qweight_op, OR'd */
} cov_tensor;

/* Working state shared by the enumeration walk and the forward trace. */
typedef struct {
    cov_tensor *tensors;        /* sorted by start address */
    int         n_tensors;
    int         unresolved;     /* weights whose bytes are in no tensor */
    int         dispatched_weights;
    int         drop_type;      /* harness: quant type to hide, or -1 */
} cov_ctx;

/*
 * Harness knob: drop one quant type from the dispatch derivation, so the
 * "a dispatched tensor went unaccounted for" branch can be exercised on the
 * real model instead of being taken on trust (see
 * `harnesses-here-measure-nothing`).
 *
 * It can only ever make the gate fail harder. The dropped tensors are still
 * bound by the enumeration pass, so hiding their dispatches moves them from
 * "dispatched" to "unaccounted", which is a failure either way. There is no
 * setting of this variable that turns a red gate green.
 */
#define PN_METAL_COVERAGE_DROP_TYPE_ENV "PN_METAL_COVERAGE_DROP_TYPE"

static int cov_drop_type_from_env(void)
{
    const char *env = getenv(PN_METAL_COVERAGE_DROP_TYPE_ENV);
    return (env && *env) ? atoi(env) : -1;
}

/* ============================================================
 * Tensor index
 * ============================================================ */

static int cov_tensor_cmp(const void *a, const void *b)
{
    const uint8_t *pa = ((const cov_tensor *)a)->start;
    const uint8_t *pb = ((const cov_tensor *)b)->start;
    if (pa < pb) return -1;
    if (pa > pb) return 1;
    return 0;
}

/**
 * Build the address-sorted tensor index from the still-open GGUF.
 * Returns 0 on success, -1 on allocation failure or an empty file.
 */
static int cov_build_index(pn_gguf_file *gguf, cov_ctx *ctx)
{
    if (!gguf || !gguf->tensors || gguf->n_tensors == 0 || !gguf->data)
        return -1;

    ctx->n_tensors = (int)gguf->n_tensors;
    ctx->tensors = calloc((size_t)ctx->n_tensors, sizeof(*ctx->tensors));
    if (!ctx->tensors)
        return -1;

    for (int i = 0; i < ctx->n_tensors; i++) {
        const pn_gguf_tensor_info *ti = &gguf->tensors[i];
        size_t numel = 1;
        for (uint32_t d = 0; d < ti->n_dims && d < 4; d++)
            numel *= (size_t)ti->dims[d];

        cov_tensor *ct = &ctx->tensors[i];
        ct->start = (const uint8_t *)gguf->data + ti->offset;
        ct->bytes = (uint64_t)pn_gguf_type_size(ti->type, numel);
        ct->name  = ti->name.data ? ti->name.data : "<unnamed>";
        ct->type  = ti->type;
    }

    qsort(ctx->tensors, (size_t)ctx->n_tensors, sizeof(*ctx->tensors),
          cov_tensor_cmp);
    return 0;
}

/**
 * Resolve a weight's data pointer to the tensor containing it.
 * Binary search for the last tensor starting at or before `p`, then a
 * containment test — an interior expert slice lands on its merged parent.
 *
 * @return the tensor, or NULL if `p` is outside every tensor
 */
static cov_tensor *cov_tensor_for_ptr(cov_ctx *ctx, const void *p)
{
    const uint8_t *q = (const uint8_t *)p;
    int lo = 0, hi = ctx->n_tensors - 1, found = -1;

    while (lo <= hi) {
        int mid = lo + (hi - lo) / 2;
        if (ctx->tensors[mid].start <= q) {
            found = mid;
            lo = mid + 1;
        } else {
            hi = mid - 1;
        }
    }
    if (found < 0)
        return NULL;

    cov_tensor *ct = &ctx->tensors[found];
    if (q < ct->start + ct->bytes)
        return ct;
    return NULL;
}

/* ============================================================
 * Pass 2 — enumerate the model's weights
 * ============================================================ */

static void cov_bind_weight(const pn_qwen35moe_weight_ref *ref, void *user)
{
    cov_ctx *ctx = (cov_ctx *)user;
    cov_tensor *ct = cov_tensor_for_ptr(ctx, ref->w->data);
    if (!ct) {
        ctx->unresolved++;
        return;
    }
    ct->n_bound++;
}

/* ============================================================
 * Pass 3 — trace one forward
 * ============================================================ */

static void cov_observe_dispatch(const pn_qweight *w, pn_qweight_op op,
                                 void *user)
{
    cov_ctx *ctx = (cov_ctx *)user;
    cov_tensor *ct = cov_tensor_for_ptr(ctx, w->data);
    if (!ct)
        return;   /* counted as unresolved by the enumeration pass */
    if (ctx->drop_type >= 0 && (int)ct->type == ctx->drop_type)
        return;   /* harness: pretend this type is never dispatched */
    ct->n_dispatched++;
    ct->ops |= 1u << (unsigned)op;
    ctx->dispatched_weights++;
}

/**
 * Run one forward token at position 0 with the observer installed.
 *
 * One token is sufficient and this is load-bearing: every layer runs on every
 * token, and a layer's routed experts are slices of that layer's own merged
 * tensors, so whichever 8 of 256 the router picks, the tensor is marked. What
 * one token does NOT cover is a code path gated on position or batch size —
 * see the header's note on prefill.
 *
 * @return 0 on success, -1 if the state could not be built or the forward
 *         returned no logits (either is a gate failure, never a pass)
 */
static int cov_trace_forward(pn_qwen35moe_model *model, cov_ctx *ctx)
{
    void *state = pn_qwen35moe_state_new(model);
    if (!state)
        return -1;

    pn_qweight_set_dispatch_observer(cov_observe_dispatch, ctx);
    float *logits = pn_qwen35moe_forward_token(model, 0,
                                               (pn_qwen35moe_state *)state, 0);
    pn_qweight_set_dispatch_observer(NULL, NULL);

    pn_qwen35moe_state_free(state);
    return logits ? 0 : -1;
}

/* ============================================================
 * Classification + per-type rows
 * ============================================================ */

/** Find (or append) the row for a quant type. NULL when the table is full. */
static pn_qwen35moe_metal_coverage_row *cov_row_for_type(
        pn_qwen35moe_metal_coverage *out, pn_gguf_quant type)
{
    for (int i = 0; i < out->n_rows; i++)
        if (out->rows[i].type == type)
            return &out->rows[i];

    if (out->n_rows >= PN_METAL_COVERAGE_MAX_TYPES)
        return NULL;

    pn_qwen35moe_metal_coverage_row *row = &out->rows[out->n_rows++];
    memset(row, 0, sizeof(*row));
    row->type = type;
    pn_gguf_quant_name(type, row->type_name, sizeof(row->type_name));
    row->covered = pn_qwen35moe_metal_kernel_for_type(type, &row->kernel_entry);
    return row;
}

/** Record an unaccounted tensor's name, up to the sample bound. */
static void cov_name_unaccounted(pn_qwen35moe_metal_coverage *out,
                                 const cov_tensor *ct)
{
    if (out->n_unaccounted_named >= PN_METAL_COVERAGE_MAX_SAMPLES)
        return;
    snprintf(out->unaccounted[out->n_unaccounted_named],
             PN_METAL_COVERAGE_NAME_MAX, "%s", ct->name);
    out->n_unaccounted_named++;
}

/** Fold one tensor into its class and, when dispatched, into its type row. */
static void cov_classify_tensor(const cov_tensor *ct,
                                pn_qwen35moe_metal_coverage *out)
{
    out->file_tensors++;
    out->file_bytes += ct->bytes;

    if (ct->n_dispatched == 0) {
        if (ct->n_bound > 0) {
            out->unaccounted_tensors++;
            out->unaccounted_bytes += ct->bytes;
            cov_name_unaccounted(out, ct);
        } else {
            out->excluded_tensors++;
            out->excluded_bytes += ct->bytes;
        }
        return;
    }

    out->dispatched_tensors++;
    out->dispatched_bytes += ct->bytes;

    pn_qwen35moe_metal_coverage_row *row = cov_row_for_type(out, ct->type);
    if (!row) {
        /* More distinct types than the table holds. Count it uncovered rather
         * than drop it: an unexpected type is exactly what must not pass. */
        out->uncovered_tensors++;
        out->uncovered_bytes += ct->bytes;
        return;
    }
    if (row->tensors == 0)
        snprintf(row->example, sizeof(row->example), "%s", ct->name);
    row->tensors++;
    row->bytes += ct->bytes;
    row->weights += ct->n_dispatched;
    row->ops |= ct->ops;

    if (!row->covered) {
        out->uncovered_tensors++;
        out->uncovered_bytes += ct->bytes;
    }
}

/* Heaviest type first; ties broken by type id so the order is deterministic. */
static int cov_row_cmp(const void *a, const void *b)
{
    const pn_qwen35moe_metal_coverage_row *ra = a, *rb = b;
    if (ra->bytes > rb->bytes) return -1;
    if (ra->bytes < rb->bytes) return 1;
    return (int)ra->type - (int)rb->type;
}

/* ============================================================
 * Public API
 * ============================================================ */

int pn_qwen35moe_metal_coverage_run(pn_qwen35moe_model *model,
                                    pn_qwen35moe_metal_coverage *out)
{
    if (!model || !out)
        return -1;

    memset(out, 0, sizeof(*out));

    cov_ctx ctx = {0};
    ctx.drop_type = cov_drop_type_from_env();
    if (cov_build_index(pn_qwen35moe_model_gguf(model), &ctx) != 0)
        return -1;

    pn_qwen35moe_for_each_weight(model, cov_bind_weight, &ctx);

    int rc = cov_trace_forward(model, &ctx);
    out->traced = (rc == 0);

    for (int i = 0; i < ctx.n_tensors; i++)
        cov_classify_tensor(&ctx.tensors[i], out);

    /* Rank by bytes so the table reads as a progress meter: the type worth the
     * most of the model is the one to build first. */
    qsort(out->rows, (size_t)out->n_rows, sizeof(out->rows[0]), cov_row_cmp);

    out->unresolved_weights = ctx.unresolved;
    out->dispatched_weights = ctx.dispatched_weights;

    free(ctx.tensors);
    return rc;
}

int pn_qwen35moe_metal_coverage_passed(const pn_qwen35moe_metal_coverage *cov)
{
    if (!cov || !cov->traced)
        return 0;
    return cov->uncovered_tensors == 0 &&
           cov->unaccounted_tensors == 0 &&
           cov->unresolved_weights == 0 &&
           cov->dispatched_tensors > 0;
}

/**
 * Human label for the ops a type was dispatched under, e.g. "matvec+dot".
 * Every op present is named: a type reached through more than one op shape
 * needs a kernel for each, so collapsing them would understate the work.
 */
static void cov_ops_label(unsigned ops, char *buf, size_t len)
{
    static const struct { pn_qweight_op op; const char *name; } NAMES[] = {
        { PN_QWEIGHT_OP_MATVEC, "matvec" },
        { PN_QWEIGHT_OP_MATMUL, "matmul" },
        { PN_QWEIGHT_OP_DOT,    "dot"    },
    };
    size_t used = 0;

    buf[0] = '\0';
    for (size_t i = 0; i < sizeof(NAMES) / sizeof(NAMES[0]); i++) {
        if (!(ops & (1u << (unsigned)NAMES[i].op)))
            continue;
        int n = snprintf(buf + used, len - used, "%s%s",
                         used ? "+" : "", NAMES[i].name);
        if (n < 0 || (size_t)n >= len - used)
            return;
        used += (size_t)n;
    }
    if (used == 0)
        snprintf(buf, len, "?");
}

/** The per-type table — the Phase-1 progress meter. */
static void cov_report_rows(const pn_qwen35moe_metal_coverage *cov, FILE *f)
{
    fprintf(f, "metal-coverage:   %-6s %8s %8s %11s  %-13s %s\n",
            "type", "tensors", "covered", "bytes", "op", "kernel");
    for (int i = 0; i < cov->n_rows; i++) {
        const pn_qwen35moe_metal_coverage_row *r = &cov->rows[i];
        char frac[24], ops[32];
        snprintf(frac, sizeof(frac), "%d/%d", r->covered ? r->tensors : 0,
                 r->tensors);
        cov_ops_label(r->ops, ops, sizeof(ops));
        fprintf(f, "metal-coverage:   %-6s %8d %8s %8.2f GiB  %-13s %s\n",
                r->type_name, r->tensors, frac, (double)r->bytes / GIB, ops,
                r->covered ? r->kernel_entry : "MISSING");
    }
}

/** Every finding that makes the verdict what it is, named. */
static void cov_report_findings(const pn_qwen35moe_metal_coverage *cov, FILE *f)
{
    for (int i = 0; i < cov->n_rows; i++) {
        const pn_qwen35moe_metal_coverage_row *r = &cov->rows[i];
        if (r->covered)
            continue;
        fprintf(f, "metal-coverage: MISSING KERNEL  %-6s  %s"
                   "  (+%d more tensor(s), %d dispatch site(s), %.2f GiB)\n",
                r->type_name, r->example, r->tensors - 1, r->weights,
                (double)r->bytes / GIB);
    }

    /* Registration is per quant type, which is the matvec question. A type that
     * is ALSO reached through a different op shape needs a kernel for that
     * shape too, and this gate cannot see whether the registered entry point
     * serves it — so say so out loud rather than let the `op` column carry it. */
    for (int i = 0; i < cov->n_rows; i++) {
        const pn_qwen35moe_metal_coverage_row *r = &cov->rows[i];
        if (!r->covered || !(r->ops & (1u << PN_QWEIGHT_OP_DOT)))
            continue;
        fprintf(f, "metal-coverage: NOTE  %s is also dispatched as a scalar dot; "
                   "%s covers the matvec shape only — the dot needs its own "
                   "kernel and this gate does not check it\n",
                r->type_name, r->kernel_entry);
    }

    if (cov->unaccounted_tensors > 0) {
        fprintf(f, "metal-coverage: UNACCOUNTED  %d tensor(s) (%.2f GiB) are "
                   "bound to a pn_qweight but were never dispatched — either a "
                   "dispatch site the observer cannot see, or dead weight\n",
                cov->unaccounted_tensors,
                (double)cov->unaccounted_bytes / GIB);
        for (int i = 0; i < cov->n_unaccounted_named; i++)
            fprintf(f, "metal-coverage:   unaccounted e.g. %s\n",
                    cov->unaccounted[i]);
    }

    if (cov->unresolved_weights > 0)
        fprintf(f, "metal-coverage: UNRESOLVED  %d weight(s) point outside "
                   "every GGUF tensor — the trace cannot vouch for them\n",
                cov->unresolved_weights);

    if (!cov->traced)
        fprintf(f, "metal-coverage: TRACE FAILED — the dispatch set is unknown, "
                   "so this is a failure, not an empty pass\n");
}

void pn_qwen35moe_metal_coverage_report(const pn_qwen35moe_metal_coverage *cov,
                                        FILE *f)
{
    if (!cov || !f)
        return;

    uint64_t covered_bytes = cov->dispatched_bytes - cov->uncovered_bytes;
    int covered_tensors = cov->dispatched_tensors - cov->uncovered_tensors;
    double pct = cov->dispatched_bytes
               ? 100.0 * (double)covered_bytes / (double)cov->dispatched_bytes
               : 0.0;

    fprintf(f, "metal-coverage: qwen35moe Metal matvec coverage "
               "(dispatch set traced from one forward)\n");
    cov_report_rows(cov, f);
    fprintf(f, "metal-coverage:   TOTAL  %d/%d dispatched tensors covered, "
               "%.2f/%.2f GiB (%.1f%%)\n",
            covered_tensors, cov->dispatched_tensors,
            (double)covered_bytes / GIB, (double)cov->dispatched_bytes / GIB,
            pct);
    fprintf(f, "metal-coverage:   dispatch sites: %d pn_qweight(s) across %d "
               "tensor(s)\n", cov->dispatched_weights, cov->dispatched_tensors);
    fprintf(f, "metal-coverage:   excluded: %d of %d file tensors (%.2f GiB) "
               "are never fed to a kernel — norms, embeddings; not a gap\n",
            cov->excluded_tensors, cov->file_tensors,
            (double)cov->excluded_bytes / GIB);

    cov_report_findings(cov, f);

    fprintf(f, "metal-coverage: %s\n",
            pn_qwen35moe_metal_coverage_passed(cov) ? "PASS" : "FAIL");
}

int pn_qwen35moe_metal_coverage_check(pn_qwen35moe_model *model, FILE *f)
{
    pn_qwen35moe_metal_coverage cov = {0};
    int rc = pn_qwen35moe_metal_coverage_run(model, &cov);
    if (rc != 0 && cov.file_tensors == 0) {
        fprintf(f ? f : stderr,
                "metal-coverage: could not run (no model or no GGUF backing)\n");
        return 0;
    }
    pn_qwen35moe_metal_coverage_report(&cov, f ? f : stderr);
    return pn_qwen35moe_metal_coverage_passed(&cov);
}
