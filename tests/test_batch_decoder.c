/* SPDX-License-Identifier: AGPL-3.0-or-later */
/* Copyright (C) 2026 David (Corey) Herrera */
/* Validate the continuous-batch VLM decoder against sequential pn_describe_embeds.
 *
 * Two pages with different (synthetic) image embeds are decoded greedily both
 * ways: once each via pn_describe_embeds (reference), and both together via the
 * pn_batch_decoder continuous-batching API. Each page's batched transcription
 * must be byte-identical to its sequential one — proving prefill + batched
 * decode + per-sequence sampling/EOS/detok all match, with independent caches.
 */

#include "pantheon.h"
#include "src/core/engine.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static float *make_embeds(int n_img, int n_embd, uint32_t seed) {
    float *e = malloc((size_t)n_img * n_embd * sizeof(float));
    uint32_t s = seed;
    for (int i = 0; i < n_img * n_embd; i++) {
        s ^= s << 13; s ^= s >> 17; s ^= s << 5;
        e[i] = ((float)(s % 2000) / 1000.0f - 1.0f) * 0.5f;
    }
    return e;
}

int main(int argc, char **argv) {
    const char *path = argc > 1 ? argv[1] : "models/lightonocr-lm-Q4_K_M.gguf";
    int n_img = argc > 2 ? atoi(argv[2]) : 64;
    int max_tokens = argc > 3 ? atoi(argv[3]) : 48;

    pn_model *m = pn_load(path);
    if (!m) { fprintf(stderr, "load failed\n"); return 1; }
    int n_embd = m->config.n_embd;

    pn_sampling_params sp = pn_sampling_defaults();
    sp.temperature = 0.0f;   /* greedy: deterministic */
    const char *prompt = "Convert this page to markdown.";

    float *eA = make_embeds(n_img, n_embd, 11);
    float *eB = make_embeds(n_img, n_embd, 22);

    /* Reference: sequential describe of each page. */
    pn_state *r = pn_state_new(m);
    char *refA = pn_describe_embeds(m, r, eA, n_img, n_embd, prompt, max_tokens, &sp, NULL, NULL, NULL, NULL);
    char *refB = pn_describe_embeds(m, r, eB, n_img, n_embd, prompt, max_tokens, &sp, NULL, NULL, NULL, NULL);
    pn_state_free(r);

    /* Batched: prefill both, admit into the decoder, step to completion. */
    pn_state *sA = pn_state_new(m), *sB = pn_state_new(m);
    uint32_t tA = 0, tB = 0;
    pn_vlm_prefill_embeds(m, sA, eA, n_img, n_embd, prompt, &sp, &tA);
    pn_vlm_prefill_embeds(m, sB, eB, n_img, n_embd, prompt, &sp, &tB);

    pn_batch_decoder *d = pn_batch_decoder_new(m, 2, max_tokens, &sp);
    pn_batch_decoder_admit(d, sA, 0, tA);
    pn_batch_decoder_admit(d, sB, 1, tB);

    char *batA = NULL, *batB = NULL;
    pn_page_out out[2];
    while (pn_batch_decoder_active(d) > 0) {
        int nf = pn_batch_decoder_step(d, out, 2);
        if (nf < 0) { fprintf(stderr, "batched decode unsupported\n"); return 1; }
        for (int i = 0; i < nf; i++) {
            if (out[i].page_id == 0) batA = out[i].text; else batB = out[i].text;
        }
    }
    pn_batch_decoder_free(d);
    pn_state_free(sA); pn_state_free(sB);

    int okA = refA && batA && strcmp(refA, batA) == 0;
    int okB = refB && batB && strcmp(refB, batB) == 0;
    printf("page A: %s (%zu chars)\n", okA ? "match" : "MISMATCH", refA ? strlen(refA) : 0);
    printf("page B: %s (%zu chars)\n", okB ? "match" : "MISMATCH", refB ? strlen(refB) : 0);
    printf("%s\n", (okA && okB) ? "PASS: batch decoder == sequential describe" : "FAIL");

    free(eA); free(eB); free(refA); free(refB); free(batA); free(batB);
    pn_model_free(m);
    return (okA && okB) ? 0 : 1;
}
