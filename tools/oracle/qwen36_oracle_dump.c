/* SPDX-License-Identifier: AGPL-3.0-or-later */
/* Copyright (C) 2026 David (Corey) Herrera */
/**
 * @file qwen36_oracle_dump.c
 * @brief The card-format CPU oracle's dumper: run a
 *   model token by token over a token list (teacher-forced), then greedy for G more tokens,
 *   and write a compact record a position — argmax, logsumexp, the next token's logit, the
 *   top 16 — plus optionally every logit. With `PN_FTTW_ORACLE=<trunk.fttw>` the weights are
 *   the artifact's tiles; with `QWEN35_ROUTING_DUMP=<file>` every MoE call's top-8 lands too.
 *
 *   usage: qwen36_oracle_dump <model.gguf> <tokens.npy|tokens.txt> <out_prefix>
 *              [--offset O] [--count N] [--greedy G] [--full-logits] [--prefill L]
 *   `--jobs FILE`: several sequences on ONE load, a line each `<tokens> <prefix> <count> <prefill>
 *   <greedy>` (tokens/out_prefix positionals then unused), each with a fresh state.
 *   Split bands (PN_SPLIT_LAYERS, qwen35moe_forward.c): a `%s` in PN_SPLIT_RESID_IN /
 *   PN_SPLIT_RESID_OUT becomes each sequence's out_prefix basename, so every sequence of a
 *   `--jobs` run reads and writes its own residual file (a single run expands it the same way).
 *   PN_ACT_STATS accumulates every sequence of the load into its one file.
 *   `--prefill L` (qwen36 plan S1): the first L tokens go through the BATCHED prefill (on the
 *   card, pn_infer_prefill), whose logits are the record of position L-1; positions L.. then
 *   run token by token. The records start at position L-1 (compare against the same slice).
 *   writes: <prefix>.pos.bin  (oracle_record a position), <prefix>.greedy.txt,
 *           <prefix>.logits.f32 (with --full-logits: n_vocab floats a position)
 *   Run with QWEN35_KV_Q4=0 (fp32 KV).
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
#include <time.h>

#include "include/pantheon.h"

#define ORACLE_TOP 16
#define ORACLE_MAX_TOKENS 1048576
#define ORACLE_PATH_BYTES 4096

/** The split residual knobs whose `%s` is expanded per sequence, and their templates. */
static const char *const ORACLE_SPLIT_ENV[2] = { "PN_SPLIT_RESID_IN", "PN_SPLIT_RESID_OUT" };
static char *g_split_template[2];

/** One position. Little-endian, 152 bytes, no padding. */
typedef struct {
    int32_t position;
    int32_t input_token;
    int32_t target_token; /**< the next token of the list; -1 in the greedy tail */
    int32_t argmax;
    float logsumexp;
    float target_logit;   /**< 0 in the greedy tail (target_token -1) */
    int32_t top_ids[ORACLE_TOP];
    float top_logits[ORACLE_TOP];
} oracle_record;

/** Token ids from a .npy (int32 / int64, 1-D) or a whitespace-separated text file. */
static int32_t *oracle_read_tokens(const char *path, size_t *count)
{
    FILE *f = fopen(path, "rb");
    int32_t *ids = f ? malloc(ORACLE_MAX_TOKENS * sizeof(int32_t)) : NULL;
    unsigned char magic[10];
    size_t n = 0;

    if (!ids) {
        perror(path);
        if (f)
            fclose(f);
        return NULL;
    }
    if (fread(magic, 1, 10, f) == 10 && memcmp(magic, "\x93NUMPY", 6) == 0) {
        unsigned header = (unsigned)magic[8] | ((unsigned)magic[9] << 8);
        char *dict = calloc(header + 1u, 1);
        int wide = dict && fread(dict, 1, header, f) == header && strstr(dict, "<i8") != NULL;

        for (int64_t v64; n < ORACLE_MAX_TOKENS; n++) {
            int32_t v32;

            if (wide ? fread(&v64, 8, 1, f) != 1 : fread(&v32, 4, 1, f) != 1)
                break;
            ids[n] = wide ? (int32_t)v64 : v32;
        }
        free(dict);
    } else {
        rewind(f);
        while (n < ORACLE_MAX_TOKENS && fscanf(f, "%d", &ids[n]) == 1)
            n++;
    }
    fclose(f);
    *count = n;
    return ids;
}

/** Fill the record's argmax / logsumexp / top-16 from one logits row. */
static void oracle_summarise(const float *logits, int n_vocab, oracle_record *r)
{
    double max = -3.0e38, sum = 0.0;

    /* -3e38, not -INFINITY: the build is -ffast-math (finite math assumed). */
    for (int i = 0; i < ORACLE_TOP; i++) {
        r->top_ids[i] = -1;
        r->top_logits[i] = -3.0e38f;
    }
    for (int v = 0; v < n_vocab; v++) {
        if (logits[v] > max)
            max = logits[v];
        if (logits[v] > r->top_logits[ORACLE_TOP - 1]) {
            int slot = ORACLE_TOP - 1;

            while (slot > 0 && logits[v] > r->top_logits[slot - 1]) {
                r->top_logits[slot] = r->top_logits[slot - 1];
                r->top_ids[slot] = r->top_ids[slot - 1];
                slot--;
            }
            r->top_logits[slot] = logits[v];
            r->top_ids[slot] = v;
        }
    }
    for (int v = 0; v < n_vocab; v++)
        sum += exp((double)logits[v] - max);
    r->argmax = r->top_ids[0];
    r->logsumexp = (float)(max + log(sum));
}

static double oracle_now(void)
{
    struct timespec ts;

    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (double)ts.tv_sec + (double)ts.tv_nsec * 1e-9;
}

/** Run one position, write its record (and logits), return the argmax or -1. */
static int oracle_step(pn_model *model, pn_state *state, int position, int32_t token, int32_t target,
                       FILE *records, FILE *logits_out)
{
    pn_output *out = pn_infer(model, state, (uint32_t)token);
    oracle_record r;

    if (!out || out->type != PN_OUT_LOGITS || !out->data) {
        fprintf(stderr, "qwen36_oracle_dump: pn_infer failed at position %d\n", position);
        pn_output_free(out);
        return -1;
    }
    memset(&r, 0, sizeof(r));
    r.position = position;
    r.input_token = token;
    r.target_token = target;
    oracle_summarise(out->data, out->n, &r);
    r.target_logit = target >= 0 && target < out->n ? out->data[target] : 0.0f;
    fwrite(&r, sizeof(r), 1, records);
    if (logits_out)
        fwrite(out->data, sizeof(float), (size_t)out->n, logits_out);
    pn_output_free(out);
    return r.argmax;
}

/** Keep every split knob that holds a `%s` as a template (read once, before any sequence). */
static void oracle_split_templates_capture(void)
{
    for (int k = 0; k < 2; k++) {
        const char *value = getenv(ORACLE_SPLIT_ENV[k]);

        if (value && strstr(value, "%s"))
            g_split_template[k] = strdup(value);
    }
}

/** Point each templated knob at this sequence: every `%s` -> the basename of `prefix`.
 *  @return 0, or -1 (an expansion too long, or no memory for setenv) */
static int oracle_split_templates_apply(const char *prefix)
{
    const char *base = strrchr(prefix, '/');

    base = base ? base + 1 : prefix;
    for (int k = 0; k < 2; k++) {
        const char *in = g_split_template[k];
        char path[ORACLE_PATH_BYTES];
        size_t n = 0;
        const size_t base_len = strlen(base);

        if (!in)
            continue;
        while (*in && n + 1 < sizeof(path)) {
            if (in[0] == '%' && in[1] == 's') {
                if (n + base_len + 1 > sizeof(path))
                    break;
                memcpy(path + n, base, base_len);
                n += base_len;
                in += 2;
            } else {
                path[n++] = *in++;
            }
        }
        path[n] = '\0';
        if (*in || setenv(ORACLE_SPLIT_ENV[k], path, 1) != 0) {
            fprintf(stderr, "qwen36_oracle_dump: %s='%s' does not expand for '%s'\n",
                    ORACLE_SPLIT_ENV[k], g_split_template[k], base);
            return -1;
        }
    }
    return 0;
}

/** One sequence on a loaded model: its own state, its own three output files. */
static int oracle_run_sequence(pn_model *model, const int32_t *ids, size_t total, size_t offset,
                               size_t count, int greedy, int full, int prefill,
                               const char *prefix)
{
    char path[1024];
    FILE *records, *greedy_out, *logits_out = NULL;
    pn_state *state = pn_state_new(model);
    double t0;
    int rc = 1;

    if (offset >= total || oracle_split_templates_apply(prefix) != 0) {
        pn_state_free(state);
        return 1;
    }
    if (count == 0 || offset + count > total)
        count = total - offset;
    snprintf(path, sizeof(path), "%s.pos.bin", prefix);
    records = fopen(path, "wb");
    snprintf(path, sizeof(path), "%s.greedy.txt", prefix);
    greedy_out = fopen(path, "w");
    if (full) {
        snprintf(path, sizeof(path), "%s.logits.f32", prefix);
        logits_out = fopen(path, "wb");
    }
    if (!state || !records || !greedy_out || (full && !logits_out))
        goto done;

    t0 = oracle_now();
    int next = -1;
    size_t first = 0;
    if (prefill >= 2 && (size_t)prefill <= count) {
        uint32_t *batch = malloc((size_t)prefill * sizeof(uint32_t));
        pn_output *out;
        oracle_record r;

        for (int i = 0; batch && i < prefill; i++)
            batch[i] = (uint32_t)ids[offset + (size_t)i];
        out = batch ? pn_infer_prefill(model, state, batch, prefill) : NULL;
        free(batch);
        if (!out || out->type != PN_OUT_LOGITS || !out->data) {
            fprintf(stderr, "qwen36_oracle_dump: the batched prefill of %d tokens was NOT taken\n",
                    prefill);
            pn_output_free(out);
            goto done;
        }
        memset(&r, 0, sizeof(r));
        r.position = prefill - 1;
        r.input_token = ids[offset + (size_t)prefill - 1u];
        r.target_token = (size_t)prefill < count ? ids[offset + (size_t)prefill] : -1;
        oracle_summarise(out->data, out->n, &r);
        r.target_logit = r.target_token >= 0 && r.target_token < out->n ? out->data[r.target_token] : 0.0f;
        fwrite(&r, sizeof(r), 1, records);
        if (logits_out)
            fwrite(out->data, sizeof(float), (size_t)out->n, logits_out);
        next = r.argmax;
        pn_output_free(out);
        first = (size_t)prefill;
        fprintf(stderr, "qwen36_oracle_dump: batched prefill of %d tokens in %.1f s\n", prefill,
                oracle_now() - t0);
    }
    for (size_t p = first; p < count; p++) {
        int32_t target = offset + p + 1 < total && p + 1 < count ? ids[offset + p + 1] : -1;

        next = oracle_step(model, state, (int)p, ids[offset + p], target, records, logits_out);
        if (next < 0)
            goto done;
    }
    for (int g = 0; g < greedy; g++) {
        fprintf(greedy_out, "%d\n", next);
        next = oracle_step(model, state, (int)(count + (size_t)g), next, -1, records, logits_out);
        if (next < 0)
            goto done;
    }
    fprintf(stderr, "qwen36_oracle_dump: %s: %zu teacher-forced + %d greedy positions in %.1f s = %.2f s/token\n",
            prefix, count, greedy, oracle_now() - t0,
            (oracle_now() - t0) / (double)(count + (size_t)greedy));
    rc = 0;

done:
    if (records)
        fclose(records);
    if (greedy_out)
        fclose(greedy_out);
    if (logits_out)
        fclose(logits_out);
    pn_state_free(state);
    return rc;
}

/** `--jobs <file>`: one sequence a line, `<tokens> <out_prefix> <count> <prefill> <greedy>`,
 *  all on ONE model load (each with a fresh state). */
static int oracle_run_jobs(pn_model *model, const char *jobs_path, int full)
{
    char tokens_path[1024], prefix[1024];
    unsigned long count;
    int prefill, greedy, rc = 0;
    FILE *jobs = fopen(jobs_path, "r");

    if (!jobs) {
        perror(jobs_path);
        return 1;
    }
    while (fscanf(jobs, "%1023s %1023s %lu %d %d", tokens_path, prefix, &count, &prefill,
                  &greedy) == 5) {
        size_t total = 0;
        int32_t *ids = oracle_read_tokens(tokens_path, &total);

        if (!ids || oracle_run_sequence(model, ids, total, 0, count, greedy, full, prefill,
                                        prefix) != 0)
            rc = 1;
        free(ids);
    }
    fclose(jobs);
    return rc;
}

int main(int argc, char **argv)
{
    size_t offset = 0, count = 0, total = 0;
    int greedy = 0, full = 0, rc = 1, prefill = 0;
    const char *jobs = NULL;
    int32_t *ids = NULL;
    pn_model *model;

    if (argc < 4) {
        fprintf(stderr, "usage: %s <model.gguf> <tokens> <out_prefix> [--offset O] [--count N] "
                        "[--greedy G] [--full-logits] [--prefill L] [--jobs FILE]\n", argv[0]);
        return 2;
    }
    for (int i = 4; i < argc; i++) {
        if (strcmp(argv[i], "--offset") == 0 && i + 1 < argc)
            offset = strtoull(argv[++i], NULL, 10);
        else if (strcmp(argv[i], "--count") == 0 && i + 1 < argc)
            count = strtoull(argv[++i], NULL, 10);
        else if (strcmp(argv[i], "--greedy") == 0 && i + 1 < argc)
            greedy = atoi(argv[++i]);
        else if (strcmp(argv[i], "--full-logits") == 0)
            full = 1;
        else if (strcmp(argv[i], "--prefill") == 0 && i + 1 < argc)
            prefill = atoi(argv[++i]);
        else if (strcmp(argv[i], "--jobs") == 0 && i + 1 < argc)
            jobs = argv[++i];
    }
    if (!jobs) {
        ids = oracle_read_tokens(argv[2], &total);
        if (!ids || offset >= total)
            return 1;
    }
    oracle_split_templates_capture();
    model = pn_load(argv[1]);
    if (model)
        rc = jobs ? oracle_run_jobs(model, jobs, full)
                  : oracle_run_sequence(model, ids, total, offset, count, greedy, full, prefill,
                                        argv[3]);
    pn_model_free(model);
    free(ids);
    return rc;
}
