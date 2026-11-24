/* SPDX-License-Identifier: AGPL-3.0-or-later */
/* Copyright (C) 2026 David (Corey) Herrera */
/**
 * @file probe_layers.c
 * @brief Per-layer residual stream of ONE forward (token id argv[2], position 0), printed as
 *        |x|, a checksum and a NaN/Inf count per layer, then the argmax logit. Two builds
 *        (e.g. arm64 NEON vs x86-64 scalar) are bisected by diffing the two printouts.
 *
 *   tests/qwen3_parity/probe_layers <model.gguf> [token]
 *
 * @status Working
 * @issues None
 * @todo None
 */
#include <math.h>
#include <stdio.h>
#include <stdlib.h>

#include "include/pantheon.h"
#include "src/core/engine.h"
#include "src/models/llama.h"
#include "src/models/llama_residual_trace.h"

static int probe_argmax(const float *l, int n)
{
    int best = 0;
    for (int i = 1; i < n; i++)
        if (l[i] > l[best]) best = i;
    return best;
}

int main(int argc, char **argv)
{
    if (argc < 2) {
        fprintf(stderr, "usage: %s model.gguf [token]\n", argv[0]);
        return 2;
    }
    uint32_t token = argc > 2 ? (uint32_t)atoi(argv[2]) : 785u;
    pn_model *model = pn_load(argv[1]);
    pn_state *state = model ? pn_state_new(model) : NULL;
    if (!state)
        return 1;
    pn_residual_trace trace;
    if (pn_residual_trace_alloc(&trace, model->config.n_layer + 1, model->config.n_embd) != 0)
        return 1;
    pn_residual_trace_capture_into(&trace);
    pn_output *out = pn_infer(model, state, token);
    pn_residual_trace_capture_into(NULL);
    for (int r = 0; r < trace.recorded; r++) {
        const float *x = trace.rows + (size_t)r * trace.width;
        double norm = 0.0, sum = 0.0;
        int bad = 0;
        for (int i = 0; i < trace.width; i++) {
            if (!isfinite(x[i])) { bad++; continue; }
            norm += (double)x[i] * x[i];
            sum += x[i] * (double)(i % 7 + 1);
        }
        printf("L%02d |x|=%.6f chk=%.6f nonfinite=%d x0=%.6f\n", r, sqrt(norm), sum, bad, x[0]);
    }
    if (out && out->data) {
        const float *l = (const float *)out->data;
        int best = 0;
        for (int i = 1; i < model->config.n_vocab; i++)
            if (l[i] > l[best]) best = i;
        printf("argmax %d logit %.6f\n", best, l[best]);
    }
    pn_output_free(out);

    /* The same five tokens sequentially (decode path) and as one batched prefill. */
    static const uint32_t seq[] = { 785, 6722, 315, 9625, 374 };
    pn_state_reset(state);
    for (int i = 0; i < 5; i++) {
        pn_residual_trace_capture_into(&trace);
        pn_output *o = pn_infer(model, state, seq[i]);
        pn_residual_trace_capture_into(NULL);
        for (int r = 0; r < trace.recorded; r++) {
            int bad = 0;
            for (int j = 0; j < trace.width; j++)
                bad += !isfinite(trace.rows[(size_t)r * trace.width + j]);
            if (bad) {
                printf("seq pos %d: first non-finite residual at L%02d (%d values)\n", i, r, bad);
                break;
            }
        }
        printf("seq pos %d argmax %d logit %.6f\n", i, probe_argmax(o->data, model->config.n_vocab),
               o->data[probe_argmax(o->data, model->config.n_vocab)]);
        pn_output_free(o);
    }
    pn_state_reset(state);
    tm_tensor *pl = pn_llama_prefill(model, seq, NULL, 5, state);
    if (pl) {
        const float *d = (const float *)pl->data;
        printf("prefill argmax %d logit %.6f\n", probe_argmax(d, model->config.n_vocab),
               d[probe_argmax(d, model->config.n_vocab)]);
        tm_free(pl);
    } else {
        printf("prefill unsupported\n");
    }
    pn_residual_trace_free(&trace);
    pn_state_free(state);
    pn_model_free(model);
    return 0;
}
