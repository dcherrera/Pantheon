/* SPDX-License-Identifier: AGPL-3.0-or-later */
/* Copyright (C) 2026 David (Corey) Herrera */
/**
 * @file qwen35moe_metal_kernel_registry.c
 * @brief Implementation of the quant-type -> Metal matvec entry point lookup.
 *
 * @status Phase 1. Q4_K (fm_matvec_q4_k_at) and F32 (fm_sgemv) resolve today;
 *         the four Phase-1 kernels do not exist yet, so their types report
 *         uncovered.
 * @issues None
 * @todo None
 */

/* 1. This module's own header */
#include "qwen35moe_metal_kernel_registry.h"

/* 2. Standard library */
#include <stddef.h>

/* 3. System / platform headers */
#ifdef __APPLE__
#include <dlfcn.h>
#endif

/*
 * Candidate entry-point names per quant type, most specific first. Several are
 * listed per type because a kernel may land in either the plain or the
 * offset-bearing (`_at`, fm_bufref) form, and because the codebase is not
 * consistent about `q3k` vs `q3_k`. The names come from
 * foundry/metal/qwen35moe_metal_plan.md Phase 1 and from the entry points
 * fm_metal.h already exports.
 *
 * Deliberately NOT listed: fm_matvec_q8 as the Q8_0 kernel. It is a
 * single-scale int8 dot, not the block-scaled Q8_0 layout (plan Phase 1:
 * "Write a new one; do not extend it"), so accepting it here would mark 94
 * tensors covered by a kernel that would read their bytes wrong — precisely
 * the silent-garbage failure this gate exists to prevent.
 */
typedef struct {
    pn_gguf_quant type;
    const char *candidates[4];
} pn_metal_kernel_entry;

static const pn_metal_kernel_entry KERNEL_TABLE[] = {
    { GGUF_QUANT_F32,  { "fm_sgemv", NULL, NULL, NULL } },
    { GGUF_QUANT_Q3_K, { "fm_matvec_q3k_at", "fm_matvec_q3k",
                         "fm_matvec_q3_k_at", "fm_matvec_q3_k" } },
    { GGUF_QUANT_Q4_K, { "fm_matvec_q4_k_at", "fm_matvec_q4_k", NULL, NULL } },
    { GGUF_QUANT_Q5_K, { "fm_matvec_q5k_at", "fm_matvec_q5k",
                         "fm_matvec_q5_k_at", "fm_matvec_q5_k" } },
    { GGUF_QUANT_Q6_K, { "fm_matvec_q6k_at", "fm_matvec_q6k",
                         "fm_matvec_q6_k_at", "fm_matvec_q6_k" } },
    { GGUF_QUANT_Q8_0, { "fm_matvec_q8_0_at", "fm_matvec_q8_0", NULL, NULL } },
};

#define KERNEL_TABLE_LEN (sizeof(KERNEL_TABLE) / sizeof(KERNEL_TABLE[0]))

/*
 * Is `symbol` present in the running image? RTLD_DEFAULT searches the main
 * executable first, which is where the Makefile's whole-program link puts
 * foundry's Metal objects.
 *
 * Off Apple there is no Metal at all, so nothing is registered by definition —
 * and this also keeps a Linux build (a K80 host) from needing -ldl.
 */
static int symbol_is_linked(const char *symbol)
{
#ifdef __APPLE__
    return dlsym(RTLD_DEFAULT, symbol) != NULL;
#else
    (void)symbol;
    return 0;
#endif
}

int pn_qwen35moe_metal_kernel_for_type(pn_gguf_quant type,
                                       const char **name_out)
{
    if (name_out)
        *name_out = NULL;

    for (size_t i = 0; i < KERNEL_TABLE_LEN; i++) {
        if (KERNEL_TABLE[i].type != type)
            continue;
        for (size_t c = 0; c < 4; c++) {
            const char *cand = KERNEL_TABLE[i].candidates[c];
            if (!cand)
                break;
            if (symbol_is_linked(cand)) {
                if (name_out)
                    *name_out = cand;
                return 1;
            }
        }
        return 0;
    }
    return 0;   /* type not in the table at all: no kernel is even planned */
}
