/* SPDX-License-Identifier: AGPL-3.0-or-later */
/* Copyright (C) 2026 David (Corey) Herrera */
/**
 * @file qwen36_oracle_compare.c
 * @brief Compare two qwen36_oracle_dump runs position by position: argmax agreement, the
 *   largest logit / logsumexp / next-token-logit differences, top-16 overlap, and (with both
 *   runs' .logits.f32) the max |dlogit| over the whole vocabulary and the mean KL(a || b).
 *   Optionally the two runs' routing dumps: per-call top-8 SET agreement and where the flips
 *   sit (the selection margin).
 *
 *   usage: qwen36_oracle_compare <prefix_a> <prefix_b> [--n-vocab V] [--routing K] [--layers L]
 *          [--max-dlogit X] [--min-argmax F]
 *   exit 0 when the bounds given hold (max |dlogit| <= X, argmax agreement >= F), else 1.
 *   `--per-position 1` prints each position's KL.
 *   `--positions P` compares only the first P positions (and their P * L routing calls): the
 *   teacher-forced span. Past it the two runs' greedy tails feed DIFFERENT tokens once an argmax
 *   differs, so everything after that is not a like-for-like comparison.
 *
 * @status None
 * @issues None
 * @todo None
 */
#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define CMP_TOP 16

typedef struct {
    int32_t position, input_token, target_token, argmax;
    float logsumexp, target_logit;
    int32_t top_ids[CMP_TOP];
    float top_logits[CMP_TOP];
} cmp_record;

static cmp_record *cmp_read_records(const char *prefix, size_t *n)
{
    char path[1024];
    FILE *f;
    long bytes;
    cmp_record *r;

    snprintf(path, sizeof(path), "%s.pos.bin", prefix);
    f = fopen(path, "rb");
    if (!f || fseek(f, 0, SEEK_END) != 0 || (bytes = ftell(f)) < 0) {
        perror(path);
        return NULL;
    }
    rewind(f);
    *n = (size_t)bytes / sizeof(cmp_record);
    r = malloc(*n * sizeof(cmp_record) + 1);
    if (r && fread(r, sizeof(cmp_record), *n, f) != *n) {
        free(r);
        r = NULL;
    }
    fclose(f);
    return r;
}

/** Whole-vocabulary logits: max |a - b| and mean KL(softmax a || softmax b). */
static int cmp_per_position;
static double cmp_median_kl = -1.0;

static int cmp_double_order(const void *x, const void *y)
{
    double a = *(const double *)x, b = *(const double *)y;

    return (a > b) - (a < b);
}

static void cmp_full_logits(const char *pa, const char *pb, size_t positions, long n_vocab, double *max_dl,
                            double *mean_kl)
{
    char path[1024];
    FILE *fa, *fb;
    float *a = malloc((size_t)n_vocab * sizeof(float)), *b = malloc((size_t)n_vocab * sizeof(float));
    double *kls = malloc((positions + 1u) * sizeof(double));
    double kl_sum = 0.0;
    size_t done = 0;

    *max_dl = -1.0;
    *mean_kl = -1.0;
    snprintf(path, sizeof(path), "%s.logits.f32", pa);
    fa = fopen(path, "rb");
    snprintf(path, sizeof(path), "%s.logits.f32", pb);
    fb = fopen(path, "rb");
    if (!fa || !fb || !a || !b)
        goto out;
    *max_dl = 0.0;
    for (; done < positions; done++) {
        double ma = -3e38, mb = -3e38, sa = 0.0, sb = 0.0, kl = 0.0;

        if (fread(a, sizeof(float), (size_t)n_vocab, fa) != (size_t)n_vocab ||
            fread(b, sizeof(float), (size_t)n_vocab, fb) != (size_t)n_vocab)
            break;
        for (long v = 0; v < n_vocab; v++) {
            *max_dl = fmax(*max_dl, fabs((double)a[v] - b[v]));
            ma = fmax(ma, a[v]);
            mb = fmax(mb, b[v]);
        }
        for (long v = 0; v < n_vocab; v++) {
            sa += exp(a[v] - ma);
            sb += exp(b[v] - mb);
        }
        for (long v = 0; v < n_vocab; v++) {
            double la = a[v] - ma - log(sa), lb = b[v] - mb - log(sb);

            kl += exp(la) * (la - lb);
        }
        kl_sum += kl;
        if (kls)
            kls[done] = kl;
        if (cmp_per_position)
            printf("  position %zu: KL %.4g\n", done, kl);
    }
    *mean_kl = done ? kl_sum / (double)done : -1.0;
    if (kls && done) {
        qsort(kls, done, sizeof(double), cmp_double_order);
        cmp_median_kl = done % 2u ? kls[done / 2u] : 0.5 * (kls[done / 2u - 1u] + kls[done / 2u]);
    }
out:
    free(kls);
    if (fa)
        fclose(fa);
    if (fb)
        fclose(fb);
    free(a);
    free(b);
}

/** Routing dumps: fraction of calls whose top-k SETS agree; flips binned by margin and, with
 *  `layers` > 0, by layer (call % layers: a decode token runs the layers in order). */
static void cmp_routing(const char *pa, const char *pb, int k, int layers, size_t max_calls)
{
    size_t by_layer[256] = {0};
    char path[1024];
    FILE *fa, *fb;
    size_t record = (size_t)k * 8u + 4u, calls = 0, same = 0, flips_near = 0;
    unsigned char *ra = malloc(record), *rb = malloc(record);

    snprintf(path, sizeof(path), "%s.routing.bin", pa);
    fa = fopen(path, "rb");
    snprintf(path, sizeof(path), "%s.routing.bin", pb);
    fb = fopen(path, "rb");
    while (fa && fb && ra && rb && (!max_calls || calls < max_calls) && fread(ra, record, 1, fa) == 1 &&
           fread(rb, record, 1, fb) == 1) {
        int32_t ia[64], ib[64];
        float pa_k, next_a;
        int match = 1;

        memcpy(ia, ra, (size_t)k * 4u);
        memcpy(ib, rb, (size_t)k * 4u);
        memcpy(&pa_k, ra + (size_t)k * 4u + (size_t)(k - 1) * 4u, 4);
        memcpy(&next_a, ra + (size_t)k * 8u, 4);
        for (int i = 0; i < k && match; i++) {
            int found = 0;

            for (int j = 0; j < k; j++)
                found |= ia[i] == ib[j];
            match = found;
        }
        if (!match && layers > 0 && layers <= 256)
            by_layer[calls % (size_t)layers]++;
        calls++;
        same += (size_t)match;
        if (!match && pa_k - next_a < 1e-3f)
            flips_near++;
    }
    printf("routing: %zu calls, top-%d SET agreement %.6f (%zu flips, %zu at a margin < 1e-3)\n", calls, k,
           calls ? (double)same / (double)calls : 0.0, calls - same, flips_near);
    if (layers > 0 && layers <= 256 && calls > same) {
        printf("routing flips by layer:");
        for (int l = 0; l < layers; l++)
            if (by_layer[l])
                printf(" L%d:%zu", l, by_layer[l]);
        printf("\n");
    }
    if (fa)
        fclose(fa);
    if (fb)
        fclose(fb);
    free(ra);
    free(rb);
}

int main(int argc, char **argv)
{
    long n_vocab = 0;
    int routing_k = 0, routing_layers = 0;
    size_t positions_limit = 0;
    double max_dlogit_bound = -1.0, min_argmax = -1.0;
    size_t na = 0, nb = 0, n, same_argmax = 0, top_overlap = 0;
    double d_lse = 0.0, d_target = 0.0, max_dl = -1.0, mean_kl = -1.0;
    cmp_record *a, *b;
    int ok = 1;

    if (argc < 3) {
        fprintf(stderr, "usage: %s <prefix_a> <prefix_b> [--n-vocab V] [--routing K] [--max-dlogit X] "
                        "[--min-argmax F]\n", argv[0]);
        return 2;
    }
    for (int i = 3; i + 1 < argc; i += 2) {
        if (strcmp(argv[i], "--n-vocab") == 0)
            n_vocab = atol(argv[i + 1]);
        else if (strcmp(argv[i], "--routing") == 0)
            routing_k = atoi(argv[i + 1]);
        else if (strcmp(argv[i], "--layers") == 0)
            routing_layers = atoi(argv[i + 1]);
        else if (strcmp(argv[i], "--positions") == 0)
            positions_limit = (size_t)atol(argv[i + 1]);
        else if (strcmp(argv[i], "--per-position") == 0)
            cmp_per_position = atoi(argv[i + 1]);
        else if (strcmp(argv[i], "--max-dlogit") == 0)
            max_dlogit_bound = atof(argv[i + 1]);
        else if (strcmp(argv[i], "--min-argmax") == 0)
            min_argmax = atof(argv[i + 1]);
    }
    a = cmp_read_records(argv[1], &na);
    b = cmp_read_records(argv[2], &nb);
    if (!a || !b || na == 0 || (na != nb && !positions_limit)) {
        fprintf(stderr, "qwen36_oracle_compare: %zu vs %zu positions (unequal runs need --positions)\n", na, nb);
        return 1;
    }
    n = na < nb ? na : nb;
    if (positions_limit && positions_limit < n)
        n = positions_limit;
    for (size_t p = 0; p < n; p++) {
        same_argmax += a[p].argmax == b[p].argmax;
        d_lse = fmax(d_lse, fabs((double)a[p].logsumexp - b[p].logsumexp));
        if (a[p].target_token >= 0)
            d_target = fmax(d_target, fabs((double)a[p].target_logit - b[p].target_logit));
        for (int i = 0; i < CMP_TOP; i++)
            for (int j = 0; j < CMP_TOP; j++)
                top_overlap += a[p].top_ids[i] == b[p].top_ids[j];
    }
    if (n_vocab > 0)
        cmp_full_logits(argv[1], argv[2], n, n_vocab, &max_dl, &mean_kl);
    printf("%zu positions: argmax agreement %.6f (%zu differ), max |d logsumexp| %.3g, max |d next-token "
           "logit| %.3g, top-16 overlap %.4f", n, (double)same_argmax / (double)n, n - same_argmax, d_lse, d_target,
           (double)top_overlap / (double)(n * CMP_TOP));
    if (n_vocab > 0)
        printf(", max |dlogit| %.3g, mean KL %.3g, median KL %.3g", max_dl, mean_kl, cmp_median_kl);
    printf("\n");
    if (routing_k > 0)
        cmp_routing(argv[1], argv[2], routing_k, routing_layers,
                    positions_limit && routing_layers > 0 ? positions_limit * (size_t)routing_layers : 0u);
    if (max_dlogit_bound >= 0.0 && !(max_dl >= 0.0 && max_dl <= max_dlogit_bound))
        ok = 0;
    if (min_argmax >= 0.0 && (double)same_argmax / (double)n < min_argmax)
        ok = 0;
    printf("%s\n", ok ? "WITHIN BOUNDS" : "OUT OF BOUNDS");
    free(a);
    free(b);
    return ok ? 0 : 1;
}
