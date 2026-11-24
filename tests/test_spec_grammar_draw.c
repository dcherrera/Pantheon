/* SPDX-License-Identifier: AGPL-3.0-or-later */
/* Copyright (C) 2026 David (Corey) Herrera */
/**
 * @file test_spec_grammar_draw.c
 * @brief Gate for the MTP grammar draw (pn_mtp_draw_rows_grammar, spec_mtp.h): verify rows
 *        drawn over a scratch clone of the grammar walk give plain grammar decoding's stream,
 *        byte for byte.
 *
 * @status Complete. CPU-only — no card, no model.
 * @issues None
 * @todo None
 *
 * A toy grammar (text per token id; the mask is live in bands of positions, and a live mask
 * allows a quarter of the vocabulary that moves with the walk; every 29th position allows
 * nothing, so the fail-closed end fires) over dense synthetic rows, one per position.
 * PLAIN draws every position with pn_sample_next_token over the live walk (the batched driver's
 * plain step). SPEC runs cycles as the batched driver does: a seed, 0-4 drafts (the plain
 * tokens, some corrupted), pn_mtp_draw_rows_grammar over a clone of the live walk, then the
 * live walk advanced over the emitted tokens (driver_spec_deposit). Checks, greedy and seeded
 * (T 0.6 / top-k 20 / top-p 0.95), without and with a repeat penalty: every emitted token,
 * the RNG stream and the walk equal plain's at the same position; masked rows and rejected
 * drafts both occur (non-vacuous).
 * `--selftest`: the reds PN_MTP_RED=grammar_live_walk (draws do not advance the scratch walk)
 * and PN_MTP_RED=1 (drafts accepted unchecked) must each FAIL the identity.
 */
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "../src/core/engine.h"
#include "../src/core/sampling_modifiers.h"
#include "../src/core/spec_mtp.h"

#define NV 512          /* vocabulary */
#define POSITIONS 600   /* positions decoded per run */
#define WINDOW 64       /* repeat window */
#define END_TOKEN 3     /* the fail-closed forced token */

static int g_checks, g_fail;
#define CHECK(c, ...) do { g_checks++; if (!(c)) { g_fail++; printf("FAIL: "); \
    printf(__VA_ARGS__); printf("\n"); } else { printf("ok: "); printf(__VA_ARGS__); \
    printf("\n"); } } while (0)

typedef struct {
    long len;          /* tokens advanced over */
    uint64_t hash;     /* fold of the advanced texts (state identity) */
} toy_walk;

static char g_text[NV][4];

static const char *toy_text(void *user, uint32_t id)
{
    (void)user;
    return id < NV ? g_text[id] : NULL;
}

static int toy_refresh(void *st)
{
    const toy_walk *w = st;
    return (w->len / 9) % 3 != 0;   /* bands: 9 free positions, then 18 masked */
}

static int toy_allowed(void *st, const char *text)
{
    const toy_walk *w = st;
    if (w->len % 29 == 28)
        return 0;                   /* nothing legal: the fail-closed end */
    return (text[0] - 'a' + (int)(w->hash % 4)) % 4 == 0;
}

static void toy_advance(void *st, const char *text)
{
    toy_walk *w = st;
    w->len++;
    for (const char *c = text; *c; c++)
        w->hash = w->hash * 1099511628211ULL ^ (unsigned char)*c;
}

static void *toy_clone(void *st)
{
    toy_walk *w = malloc(sizeof(*w));
    if (w)
        *w = *(const toy_walk *)st;
    return w;
}

static void toy_clone_free(void *st) { free(st); }

static pn_tool_grammar_hook toy_hook(toy_walk *st)
{
    pn_tool_grammar_hook h;
    memset(&h, 0, sizeof(h));
    h.token_text = toy_text;
    h.allowed = toy_allowed;
    h.advance = toy_advance;
    h.refresh = toy_refresh;
    h.clone = toy_clone;
    h.clone_free = toy_clone_free;
    h.st = st;
    h.n_vocab = NV;
    h.fail_closed = 1;
    h.forced_token = END_TOKEN;
    return h;
}

static uint64_t g_rng;
static double unit_rand(void)
{
    g_rng = g_rng * 6364136223846793005ULL + 1442695040888963407ULL;
    return (double)(g_rng >> 11) / 9007199254740992.0;
}

/* Row p: dense logits, deterministic in p (both arms read the same row for a position). */
static float *g_rows;
static void make_rows(void)
{
    g_rows = malloc((size_t)POSITIONS * NV * sizeof(float));
    g_rng = 12345;
    for (size_t i = 0; i < (size_t)POSITIONS * NV; i++)
        g_rows[i] = (float)(unit_rand() * 12.0 - 6.0) * (i % NV < 40 ? 1.6f : 1.0f);
    for (int i = 0; i < NV; i++) {
        g_text[i][0] = (char)('a' + i % 23);
        g_text[i][1] = (char)('a' + (i / 23) % 23);
        g_text[i][2] = '\0';
    }
}

static int fetch_row(void *ctx, int row, float *out)
{
    const int base = *(const int *)ctx;
    if (base + row >= POSITIONS)
        return -1;
    memcpy(out, g_rows + (size_t)(base + row) * NV, NV * sizeof(float));
    return 0;
}

static pn_sampling_params params_at(float temp, float rp)
{
    pn_sampling_params sp = pn_sampling_defaults();
    sp.temperature = temp;
    sp.top_k = 20;
    sp.top_p = 0.95f;
    sp.repeat_penalty = rp;
    sp.repeat_last_n = WINDOW;
    sp.rng_seeded = 1;
    sp.rng_state = pn_seed_to_rng_state(777);
    return sp;
}

typedef struct {
    uint32_t tok[POSITIONS];
    uint64_t rng[POSITIONS];       /* stream state after drawing position p */
    toy_walk walk[POSITIONS];      /* walk after advancing over position p's token */
    long masked;
} plain_run;

static void run_plain(float temp, float rp, plain_run *out)
{
    pn_sampling_params sp = params_at(temp, rp);
    toy_walk w = { 0, 0 };
    pn_tool_grammar_hook h = toy_hook(&w);
    uint32_t recent[WINDOW];
    int n_recent = 0;
    float *row = malloc(NV * sizeof(float));

    sp.tool_grammar = &h;
    out->masked = 0;
    for (int p = 0; p < POSITIONS; p++) {
        out->masked += toy_refresh(&w);
        memcpy(row, g_rows + (size_t)p * NV, NV * sizeof(float));
        out->tok[p] = pn_sample_next_token(row, NV, &sp, recent, n_recent);
        pn_mtp_recent_push(recent, &n_recent, WINDOW, out->tok[p]);
        out->rng[p] = sp.rng_state;
        out->walk[p] = w;
    }
    free(row);
}

typedef struct {
    long cycles, drafted, accepted, mismatches;
} spec_run;

/* The batched driver's cycles against plain's stream: returns its counters. */
static spec_run run_spec(float temp, float rp, const plain_run *plain, uint64_t seed)
{
    pn_sampling_params sp = params_at(temp, rp);
    toy_walk live = { 0, 0 };
    pn_tool_grammar_hook hook = toy_hook(&live);
    uint32_t recent[WINDOW], toks[PN_SPEC_DEPTH_MAX + 1], emit[PN_SPEC_DEPTH_MAX + 1];
    int n_recent = 0, p = 0;
    float *row = malloc(NV * sizeof(float));
    spec_run r = { 0, 0, 0, 0 };

    g_rng = seed;
    while (p < POSITIONS && r.mismatches == 0) {
        int nd = (int)(unit_rand() * (PN_SPEC_DEPTH_MAX + 1));
        if (p + nd >= POSITIONS)
            nd = POSITIONS - p - 1;
        toks[0] = p > 0 ? plain->tok[p - 1] : 0;
        for (int i = 0; i < nd; i++)
            toks[i + 1] = unit_rand() < 0.75 ? plain->tok[p + i] : (plain->tok[p + i] + 1) % NV;
        pn_tool_grammar_hook scratch = hook;
        scratch.st = toy_clone(&live);
        sp.tool_grammar = &scratch;
        int base = p;
        int ne = pn_mtp_draw_rows_grammar(fetch_row, &base, toks, nd, row, NV, &sp, recent,
                                          &n_recent, WINDOW, emit);
        toy_clone_free(scratch.st);
        r.cycles++;
        r.drafted += nd;
        r.accepted += ne - 1;
        for (int j = 0; j < ne; j++) {   /* driver_spec_deposit: the live walk takes each */
            const char *text = toy_text(NULL, emit[j]);
            if (text)
                toy_advance(&live, text);
            if (emit[j] != plain->tok[p + j] || live.len != plain->walk[p + j].len ||
                live.hash != plain->walk[p + j].hash)
                r.mismatches++;
        }
        p += ne;
        if (r.mismatches == 0 && sp.rng_state != plain->rng[p - 1])
            r.mismatches++;
    }
    free(row);
    return r;
}

static void gate(float temp, float rp, const char *label)
{
    static plain_run plain;
    run_plain(temp, rp, &plain);
    spec_run r = run_spec(temp, rp, &plain, 99);
    CHECK(r.mismatches == 0, "%s: tokens, RNG stream and walk equal plain (%ld cycles)",
          label, r.cycles);
    CHECK(plain.masked > POSITIONS / 3 && r.accepted < r.drafted && r.accepted > 0,
          "%s: non-vacuous (masked rows %ld, drafts %ld, accepted %ld)", label, plain.masked,
          r.drafted, r.accepted);
}

static int selftest_red(const char *red)
{
    static plain_run plain;
    setenv("PN_MTP_RED", red, 1);
    run_plain(0.6f, 1.0f, &plain);
    spec_run r = run_spec(0.6f, 1.0f, &plain, 99);
    unsetenv("PN_MTP_RED");
    printf("red %s: %ld mismatches over %ld cycles\n", red, r.mismatches, r.cycles);
    return r.mismatches > 0;
}

int main(int argc, char **argv)
{
    make_rows();
    if (argc > 1 && strcmp(argv[1], "--selftest") == 0) {
        int ok = selftest_red("grammar_live_walk") && selftest_red("1");
        printf(ok ? "SELFTEST OK: both reds fail the identity\n"
                  : "SELFTEST FAIL: a red arm passed the identity gate\n");
        return ok ? 0 : 1;
    }
    gate(0.0f, 1.0f, "greedy");
    gate(0.6f, 1.0f, "seeded T0.6");
    gate(0.6f, 1.1f, "seeded T0.6 repeat 1.1");
    printf("\n%s: %d checks, %d failures\n", g_fail ? "FAILED" : "PASSED", g_checks, g_fail);
    free(g_rows);
    return g_fail ? 1 : 0;
}
