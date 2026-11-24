/* SPDX-License-Identifier: AGPL-3.0-or-later */
/* Copyright (C) 2026 David (Corey) Herrera */
/*
 * Mask-computation cost of `field_patterns`: the time to ask the tool grammar
 * about every token of a vocabulary (one full mask, what the serve does per
 * masked step) with the walk inside a `finding` value, with and without the
 * client's pattern on that value, in both dialects.
 *
 * The vocabulary is synthetic and deterministic (the repo has no tokenizer
 * fixture and Mac model loads are off-limits for a bench): 248,320 pieces — the
 * Qwen3.5/3.6 vocab size — drawn as space-led English subwords, code fragments,
 * random alphanumeric runs (1-12 bytes), punctuation, digits, newline pieces and
 * UTF-8 letters, plus the call skeleton's own tokens.
 *
 * Usage: tests/bench_field_pattern_mask [reps]   (default 15; median reported)
 * Built with -DFPM_BASELINE it runs only the no-pattern rows and never calls the
 * apply functions, so it links against the grammar sources from before
 * `field_patterns` existed (the cost of the bigger walk state when no pattern
 * is set).
 *
 * Pure C, libc-only.
 */

#include "../src/serve/json_reader.h"
#include "../src/serve/tool_grammar.h"
#include "../src/serve/tool_grammar_json.h"
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

#define VOCAB 248320
#define PIECE_MAX 16

static const char *TOOLS_JSON =
    "[{\"type\":\"function\",\"function\":{\"name\":\"read\",\"parameters\":{"
    "\"type\":\"object\",\"properties\":{"
    "\"path\":{\"type\":\"string\",\"minLength\":1},"
    "\"finding\":{\"type\":\"string\",\"minLength\":1}},"
    "\"required\":[\"path\",\"finding\"]}}}]";

static const char *FIELD_JSON =
    "{\"read\":{\"finding\":\"-|[^\\\\n]+ from: [^\\\\s,:]+:[0-9]+(-[0-9]+)?"
    "(, [^\\\\s,:]+:[0-9]+(-[0-9]+)?)*\"}}";

static char g_vocab[VOCAB][PIECE_MAX];
static uint64_t g_rng = 0x9E3779B97F4A7C15ull;

static uint32_t next_random(void) {
    g_rng ^= g_rng << 13; g_rng ^= g_rng >> 7; g_rng ^= g_rng << 17;
    return (uint32_t)(g_rng >> 20);
}

static void random_run(char *out, int len, const char *alphabet) {
    int n = (int)strlen(alphabet);
    for (int i = 0; i < len; i++) out[i] = alphabet[next_random() % (uint32_t)n];
    out[len] = '\0';
}

static void build_vocab(void) {
    static const char *WORDS[] = { " the", " from", " of", " is", " to", " read", " value",
        " line", " file", " cap", " buffer", " size", " returns", " when", " and", "ing",
        "tion", "ed", " a", " in" };
    static const char *CODE[] = { "_", ".c", ".h", "::", "(", ")", "{", "}", ";", "->", "==",
        "</", "<", ">", "\"", "\\", "/", ",", ", ", ":", " =", "[", "]", "```" };
    static const char *SKELETON[] = { "<tool_call>", "</tool_call>", "<function=", "</function>",
        "<parameter=", "</parameter>", "\n", "\n\n", " ", "</think>", "\"}}", "\n```" };
    static const char *LETTERS[] = { "\xc3\xa9", "\xc3\xbc", "\xe2\x82\xac", "\xe4\xb8\xad" };
    int i = 0;
    for (size_t k = 0; k < sizeof(SKELETON) / sizeof(SKELETON[0]); k++)
        snprintf(g_vocab[i++], PIECE_MAX, "%s", SKELETON[k]);
    for (; i < VOCAB; i++) {
        uint32_t kind = next_random() % 100;
        char *p = g_vocab[i];
        if (kind < 25) snprintf(p, PIECE_MAX, "%s", WORDS[next_random() % 20]);
        else if (kind < 40) snprintf(p, PIECE_MAX, "%s", CODE[next_random() % 23]);
        else if (kind < 85) {
            int lead = next_random() % 2;
            if (lead) p[0] = ' ';
            random_run(p + lead, 1 + (int)(next_random() % 11),
                       "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789");
        } else if (kind < 90) random_run(p, 1 + (int)(next_random() % 4), "0123456789");
        else if (kind < 94) random_run(p, 1 + (int)(next_random() % 3), "!#$%&*+-.;=?@^`|~");
        else if (kind < 97) snprintf(p, PIECE_MAX, "%s", (next_random() % 2) ? ".\n" : "\n\n\n");
        else snprintf(p, PIECE_MAX, "%s", LETTERS[next_random() % 4]);
    }
}

static double now_ns(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (double)ts.tv_sec * 1e9 + (double)ts.tv_nsec;
}

static int compare_doubles(const void *a, const void *b) {
    double x = *(const double *)a, y = *(const double *)b;
    return (x > y) - (x < y);
}

typedef bool (*allowed_fn)(const void *st, const char *text);
static bool xml_allowed(const void *st, const char *t) { return tg_token_allowed(st, t); }
static bool json_allowed(const void *st, const char *t) { return tgj_token_allowed(st, t); }

/* Median ms for one full-vocab mask; *allowed_out = tokens allowed. */
static double time_mask(const void *st, allowed_fn allowed, int reps, int *allowed_out) {
    double samples[64];
    if (reps > 64) reps = 64;
    for (int r = 0; r < reps; r++) {
        int allowed_count = 0;
        double t0 = now_ns();
        for (int i = 0; i < VOCAB; i++) allowed_count += allowed(st, g_vocab[i]);
        samples[r] = (now_ns() - t0) / 1e6;
        *allowed_out = allowed_count;
    }
    qsort(samples, (size_t)reps, sizeof(double), compare_doubles);
    return samples[reps / 2];
}

static void bench_xml(const jr_value *tools, const jr_value *field, const char *at, int reps) {
    tg_grammar *g = tg_build(tools);
#ifndef FPM_BASELINE
    if (field && !tg_apply_field_patterns(g, field)) { fprintf(stderr, "apply failed\n"); exit(1); }
#endif
    tg_state *st = tg_state_new(g);
    tg_advance(st, "<tool_call>\n<function=read>\n<parameter=path>\na.c\n</parameter>\n"
                   "<parameter=finding>\n");
    tg_advance(st, at);
    int allowed = 0;
    double ms = time_mask(st, xml_allowed, reps, &allowed);
    printf("  xml   %-8s %-26s %7.2f ms/mask  %6.1f ns/token  allowed %d\n",
           field ? "pattern" : "none", at, ms, ms * 1e6 / VOCAB, allowed);
    tg_state_free(st);
    tg_free(g);
}

static void bench_json(const jr_value *tools, const jr_value *field, const char *at, int reps) {
    tgj_grammar *g = tgj_build(tools);
#ifndef FPM_BASELINE
    if (field && !tgj_apply_field_patterns(g, field)) { fprintf(stderr, "apply failed\n"); exit(1); }
#endif
    tgj_state *st = tgj_state_new(g);
    tgj_advance(st, "\n```json\n{\"function\": \"read\", \"arguments\": {\"path\": \"a.c\", "
                    "\"finding\": \"");
    tgj_advance(st, at);
    int allowed = 0;
    double ms = time_mask(st, json_allowed, reps, &allowed);
    printf("  json  %-8s %-26s %7.2f ms/mask  %6.1f ns/token  allowed %d\n",
           field ? "pattern" : "none", at, ms, ms * 1e6 / VOCAB, allowed);
    tgj_state_free(st);
    tgj_free(g);
}

int main(int argc, char **argv) {
    int reps = argc > 1 ? atoi(argv[1]) : 15;
    if (reps < 1) reps = 1;
    build_vocab();
    jr_value *tools = jr_parse(TOOLS_JSON);
    jr_value *field = jr_parse(FIELD_JSON);
    static const char *POSITIONS[] = { "the cap is", "x from: src/a.c:1", "x from: src/a.c" };
    printf("vocab %d synthetic pieces, median of %d masks\n", VOCAB, reps);
    for (size_t i = 0; i < sizeof(POSITIONS) / sizeof(POSITIONS[0]); i++) {
        bench_xml(tools, NULL, POSITIONS[i], reps);
#ifndef FPM_BASELINE
        bench_xml(tools, field, POSITIONS[i], reps);
#endif
        bench_json(tools, NULL, POSITIONS[i], reps);
#ifndef FPM_BASELINE
        bench_json(tools, field, POSITIONS[i], reps);
#endif
    }
    jr_free(field);
    jr_free(tools);
    return 0;
}
