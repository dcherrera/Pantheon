/* SPDX-License-Identifier: AGPL-3.0-or-later */
/* Copyright (C) 2026 David (Corey) Herrera */
/* Vision-encoder benchmark + embedding dump for flash-attention validation.
 * Builds a fixed deterministic image, runs pn_vision_encode best-of-N, prints
 * timings, and writes the image-token embeddings to a binary file for cosine
 * comparison across builds. */
#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <time.h>
#include "src/formats/gguf.h"
#include "src/models/pixtral_vision.h"

static double now_ms(void) {
    struct timespec ts; clock_gettime(CLOCK_MONOTONIC, &ts);
    return ts.tv_sec * 1000.0 + ts.tv_nsec / 1e6;
}

int main(int argc, char **argv) {
    const char *mmproj = argc > 1 ? argv[1] : "models/mmproj-F16.gguf";
    const char *outfile = argc > 2 ? argv[2] : NULL;
    int dim_img = argc > 3 ? atoi(argv[3]) : 1024;
    int nruns = argc > 4 ? atoi(argv[4]) : 3;

    pn_gguf_file *g = pn_gguf_open(mmproj);
    if (!g) { fprintf(stderr, "cannot open %s\n", mmproj); return 1; }
    pn_vision_model *m = pn_vision_load(g);
    pn_gguf_close(g);
    if (!m) { fprintf(stderr, "vision load failed\n"); return 1; }

    int W = dim_img, H = dim_img;
    uint8_t *rgb = malloc((size_t)W * H * 3);
    /* deterministic pseudo-random content (fixed seed) */
    uint32_t s = 0x12345678u;
    for (size_t i = 0; i < (size_t)W * H * 3; i++) {
        s = s * 1664525u + 1013904223u;
        rgb[i] = (uint8_t)((s >> 24) ^ (i * 31u));
    }

    float *embd = NULL; int pd = 0, M = 0;
    double best = 1e30;
    for (int r = 0; r < nruns; r++) {
        free(embd); embd = NULL;
        double t0 = now_ms();
        M = pn_vision_encode(m, rgb, W, H, &embd, &pd);
        double dt = now_ms() - t0;
        if (dt < best) best = dt;
        fprintf(stderr, "run %d: %.1f ms  (tokens=%d dim=%d)\n", r, dt, M, pd);
    }
    fprintf(stderr, "BEST vision encode: %.1f ms\n", best);
    printf("%.1f\n", best);

    if (outfile && embd && M > 0) {
        FILE *f = fopen(outfile, "wb");
        if (f) {
            int32_t hdr[2] = { M, pd };
            fwrite(hdr, sizeof(int32_t), 2, f);
            fwrite(embd, sizeof(float), (size_t)M * pd, f);
            fclose(f);
            fprintf(stderr, "wrote %d x %d embeddings to %s\n", M, pd, outfile);
        }
    }
    free(embd); free(rgb); pn_vision_free(m);
    return 0;
}
