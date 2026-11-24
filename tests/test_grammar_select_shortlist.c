/* SPDX-License-Identifier: AGPL-3.0-or-later */
/* Copyright (C) 2026 David (Corey) Herrera */
/**
 * @file test_grammar_select_shortlist.c
 * @brief gms_select_shortlist (with its SSE block prefilter) returns the scalar select's
 *        shortlist: the same ids in the same order and the same bound.
 *
 * @status Complete. CPU-only.
 * @issues None
 * @todo None
 *
 * Reference: the scalar loop the prefilter replaced, copied here. Rows at 248,320 entries:
 * random; quantised (heavy ties at the root); ramps up and down (constant heap churn);
 * signed zeros near the top; -inf sprinkled; a top-N row (<= GMS_MAX finite); short rows
 * (n_vocab not a multiple of 4). The bound compares by value: under -ffast-math
 * (-fno-signed-zeros) the sign of a zero maximum is the compiler's choice even in the scalar
 * loop, and every consumer reads the bound arithmetically, where -0 and +0 are the same. NaN
 * and +inf rows are out of scope for the same reason (-ffinite-math-only).
 * `--selftest`: PN_GMS_SELECT_RED=1 (the block skip passes over entries just above the root)
 * must mismatch. On a build without SSE2 the select is the scalar loop: selftest SKIP.
 */
#include "../src/core/grammar_mask_shortlist.h"

#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define VOCAB 248320

static uint64_t g_rng = 0x9e3779b97f4a7c15ull;
static uint32_t rnd(void)
{
    g_rng ^= g_rng << 13; g_rng ^= g_rng >> 7; g_rng ^= g_rng << 17;
    return (uint32_t)(g_rng >> 11);
}
static float rnd_unit(void) { return (float)(rnd() % 16777216u) / 16777216.0f; }

static float from_bits(uint32_t b) { float f; memcpy(&f, &b, 4); return f; }

static int ref_neg_inf(const float *p) { uint32_t b; memcpy(&b, p, 4); return b == 0xff800000u; }

static void ref_heap_down(uint32_t *heap, const float *logits, int n, int at)
{
    for (;;) {
        int least = at, l = 2 * at + 1, r = l + 1;
        if (l < n && logits[heap[l]] < logits[heap[least]]) least = l;
        if (r < n && logits[heap[r]] < logits[heap[least]]) least = r;
        if (least == at) return;
        uint32_t t = heap[at]; heap[at] = heap[least]; heap[least] = t;
        at = least;
    }
}

/* The scalar select before the prefilter, verbatim in effect. */
static int ref_select(const float *logits, int n_vocab, uint32_t *ids, float *bound)
{
    int n = 0;
    float left_out = from_bits(0xff800000u);

    for (int i = 0; i < n_vocab; i++) {
        if (ref_neg_inf(&logits[i])) continue;
        if (n < GMS_MAX) {
            ids[n++] = (uint32_t)i;
            if (n == GMS_MAX)
                for (int j = GMS_MAX / 2 - 1; j >= 0; j--) ref_heap_down(ids, logits, n, j);
            continue;
        }
        if (logits[i] <= logits[ids[0]]) {
            if (logits[i] > left_out) left_out = logits[i];
            continue;
        }
        if (logits[ids[0]] > left_out) left_out = logits[ids[0]];
        ids[0] = (uint32_t)i;
        ref_heap_down(ids, logits, n, 0);
    }
    if (n == 0) return 0;
    if (n < GMS_MAX || ref_neg_inf(&left_out)) {
        float min = logits[ids[0]];
        for (int i = 1; i < n; i++)
            if (logits[ids[i]] < min) min = logits[ids[i]];
        *bound = min;
    } else {
        *bound = left_out;
    }
    return n;
}

/* kind: 0 random, 1 quantised, 2 ramp up, 3 ramp down, 4 signed zeros at the top,
 * 5 -inf sprinkled, 6 top-N row, 7 mostly -inf with zeros. */
static void make_row(float *row, int n, int kind)
{
    for (int i = 0; i < n; i++) {
        float v = rnd_unit() * 8.0f - 12.0f;
        if (kind == 1) v = (float)((int)(v * 2.0f)) * 0.5f;
        if (kind == 2) v = -20.0f + 30.0f * (float)i / (float)n;
        if (kind == 3) v = 10.0f - 30.0f * (float)i / (float)n;
        if (kind == 6 || kind == 7) v = from_bits(0xff800000u);
        row[i] = v;
    }
    if (kind == 4 || kind == 7) {
        /* The top of the row sits at or below zero, so zeros decide the left-out max. */
        for (int i = 0; i < n; i++) if (row[i] > -0.5f) row[i] = -1.0f;
        for (int k = 0; k < 400; k++)
            row[rnd() % (uint32_t)n] = (rnd() & 1) ? -0.0f : 0.0f;
        for (int k = 0; kind == 4 && k < 300; k++)
            row[rnd() % (uint32_t)n] = 0.5f + rnd_unit();
    }
    if (kind == 5)
        for (int k = 0; k < 40; k++)
            row[rnd() % (uint32_t)n] = from_bits(0xff800000u);
    if (kind == 6)
        for (int k = 0; k < 200; k++) row[rnd() % (uint32_t)n] = rnd_unit() * 10.0f;
}

int main(int argc, char **argv)
{
    const int selftest = argc > 1 && strcmp(argv[1], "--selftest") == 0;
#if !defined(__SSE2__)
    if (selftest) {
        printf("test_grammar_select_shortlist --selftest: SKIP (no SSE2 path)\n");
        return 0;
    }
#endif
    if (selftest)
        setenv("PN_GMS_SELECT_RED", "1", 1);
    float *row = malloc((size_t)VOCAB * sizeof(float));
    uint32_t a[GMS_MAX], b[GMS_MAX];
    int rows = 0, bad = 0;
    static const int sizes[] = { VOCAB, VOCAB - 1, VOCAB - 3, 1031, 257, 300 };

    if (!row)
        return 2;
    for (int s = 0; s < 6; s++)
        for (int kind = 0; kind < 8; kind++)
            for (int rep = 0; rep < (sizes[s] == VOCAB ? 6 : 3); rep++) {
                const int n = sizes[s];
                float ba = 0.0f, bb = 0.0f;
                make_row(row, n, kind);
                const int na = ref_select(row, n, a, &ba);
                const int nb = gms_select_shortlist(row, n, b, &bb);
                rows++;
                if (na != nb || memcmp(a, b, (size_t)na * 4) != 0 ||
                    (na > 0 && !(ba == bb))) {
                    if (++bad <= 5)
                        fprintf(stderr, "mismatch n %d kind %d rep %d: count %d/%d bound %a/%a\n",
                                n, kind, rep, na, nb, (double)ba, (double)bb);
                }
            }
    free(row);
    printf("test_grammar_select_shortlist%s: rows %d, mismatched %d\n",
           selftest ? " --selftest" : "", rows, bad);
    if (selftest) {
        printf(bad ? "SELFTEST PASS (red build caught)\n" : "SELFTEST FAIL (red build not caught)\n");
        return bad ? 0 : 1;
    }
    printf(bad ? "FAIL\n" : "PASS\n");
    return bad ? 1 : 0;
}
