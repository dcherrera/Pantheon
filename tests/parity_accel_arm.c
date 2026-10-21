/* SPDX-License-Identifier: AGPL-3.0-or-later */
/* Copyright (C) 2026 David (Corey) Herrera */
/**
 * @file parity_accel_arm.c
 * @brief Implementation of the accelerator-arm and CPU-oracle selection declared in
 *        parity_accel_arm.h.
 *
 * @status Working
 * @issues None
 * @todo None
 */

#include "parity_accel_arm.h"

#include "../src/core/engine.h"
#include "../src/models/bonsai.h"
#include "../src/models/llama.h"
#include "../src/models/qwen35moe.h"

#include <stdio.h>
#include <stdlib.h>

int parity_identity_control_active(void)
{
    const char *env = getenv("PN_PARITY_IDENTITY");

    return env && atoi(env) != 0;
}

/**
 * A human name for an accelerator forward, for the report line only.
 *
 * Selection does NOT depend on this: an unrecognised forward is still a valid accelerator
 * arm and is reported as "ACCEL". Naming and selecting were the same table once, which is
 * how an unlisted family became "no accelerator" instead of "an accelerator I cannot name".
 */
static const char *parity_accel_name(pn_forward_fn accel)
{
#ifdef USE_CUDA
    if (accel == pn_llama_forward_cuda) return "CUDA/llama";
    if (accel == pn_qwen35moe_forward_cuda) return "CUDA/qwen35moe";
    /* Bonsai/Ornith has its OWN entry symbol that delegates to the qwen35moe CUDA forward, so
     * the two compare unequal and an Ornith CUDA arm reported itself as the anonymous "ACCEL".
     * The dump carries this string as its label, which is what a reader is told the arm was. */
    if (accel == pn_bonsai_forward_cuda) return "CUDA/bonsai";
#endif
#ifdef USE_TT
    if (accel == pn_llama_forward_tt) return "TT/llama";
    if (accel == pn_qwen35moe_forward_tt) return "TT/qwen35moe";
#endif
    (void)accel;
    return "ACCEL";
}

int parity_require_cpu_oracle(pn_model *model, pn_forward_fn expected, const char *tool)
{
    pn_forward_fn cpu = parity_select_cpu_oracle(model, NULL);

    if (cpu && cpu == expected)
        return 0;
    fprintf(stderr, "%s: this harness's probes are written for one architecture and this "
                    "model's CPU oracle is a different forward. Refusing rather than reading "
                    "the wrong weight struct.\n", tool ? tool : "parity");
    return -1;
}

int parity_accel_seam_enable(pn_model *model, int enable)
{
    if (!model || !model->weights)
        return 0;
    /* The qwen35/bonsai family is the one that has a seam. Both architectures load the same
     * `pn_qwen35moe_model` weight struct — the engine's own bonsai branch selects
     * `pn_qwen35moe_forward_tt` over `pn_bonsai_forward` — so the test is the weight struct,
     * not the architecture enum. */
    if (model->config.arch == PN_ARCH_QWEN35MOE || model->config.arch == PN_ARCH_BONSAI)
        return pn_qwen35moe_model_accel_mv_enable((pn_qwen35moe_model *)model->weights,
                                                  enable);
    return 0;
}

pn_forward_fn parity_select_cpu_oracle(pn_model *model, const char **name_out)
{
    pn_forward_fn cpu;

    if (!model)
        return NULL;
    cpu = pn_get_forward_fn(model->config.arch);
    if (!cpu) {
        fprintf(stderr, "parity: architecture %d has no CPU forward in the engine registry — "
                        "there is nothing to score against. Refusing.\n", (int)model->config.arch);
        return NULL;
    }
    if (name_out)
        *name_out = "CPU";
    return cpu;
}

pn_forward_fn parity_select_accel_arm(pn_model *model, const char **name_out)
{
    pn_forward_fn cpu, accel;
    const char *name;

    if (!model)
        return NULL;
    cpu = parity_select_cpu_oracle(model, NULL);
    if (!cpu)
        return NULL;

    /* The engine already chose. An accelerator arm is simply a forward that is not the
     * architecture's CPU one — no table to keep current, and no family it silently omits. */
    accel = model->forward;
    name = parity_accel_name(accel);
    if (parity_identity_control_active()) {
        accel = cpu;
        name = "CPU-IDENTITY-CONTROL";
        fprintf(stderr, "PN_PARITY_IDENTITY: both arms are the CPU oracle. This run measures "
                        "the HARNESS, not the accelerator.\n");
    } else if (accel == cpu) {
        fprintf(stderr, "WARNING: the engine left this model on its CPU forward. Parity would "
                        "be CPU-vs-CPU and meaningless — refusing.\n");
        return NULL;
    }
    if (name_out)
        *name_out = name;
    return accel;
}
