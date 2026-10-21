/* SPDX-License-Identifier: AGPL-3.0-or-later */
/* Copyright (C) 2026 David (Corey) Herrera */
/*
 * Unit tests for speculative decoding under the tool-call grammar
 * (src/serve/grammar_spec_chooser.c) with the real ornith JSON grammar in its
 * tool_choice "required" think-gated mode.
 *
 * The central test is an EQUIVALENCE SIMULATION: a deterministic toy model
 * (logits = a hash of the token prefix, over a small vocab of real ornith
 * fragments) is decoded twice —
 *   plain : one token at a time, grammar mask on the dense row, argmax;
 *   spec  : seed + drafts verified in one cycle exactly as
 *           pn_qwen35moe_tt_spec_step_ex accepts them (a draft is accepted iff
 *           the chooser picks it; DEFER ends the cycle at the previous row),
 *           rows handed over as top-N pairs, drafts mixing the true
 *           continuation with grammar-breaking junk —
 * and the two token streams must be identical for every trial.  Every
 * completed call must parse with the serve's own parser (tc_parse_tool_calls).
 *
 * The simulation is SELF-FALSIFYING: the same spec loop with the grammar
 * ignored (mask never active) must DIVERGE from plain decoding in some trial,
 * or the test exits 2 (VACUOUS) — a comparison that cannot fail proves nothing.
 *
 * Pure C, libc-only — no engine/model/device.
 */

#include "lib/test.h"
#include "../src/serve/grammar_spec_chooser.h"
#include "../src/core/grammar_mask_shortlist.h"
#include "../src/serve/tool_grammar_json.h"
#include "../src/serve/tool_calling.h"
#include "../src/serve/json_reader.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static const char *TOOLS_JSON =
    "[{\"type\":\"function\",\"function\":{\"name\":\"read\","
    "\"parameters\":{\"type\":\"object\",\"properties\":{"
    "\"path\":{\"type\":\"string\",\"minLength\":1}},\"required\":[\"path\"]}}},"
    "{\"type\":\"function\",\"function\":{\"name\":\"add\","
    "\"parameters\":{\"type\":\"object\",\"properties\":{"
    "\"a\":{\"type\":\"integer\"},\"b\":{\"type\":\"integer\"}},"
    "\"required\":[\"a\",\"b\"]}}}]";

/* Toy vocab: real ornith fragments, reasoning text, and junk the grammar must
 * refuse once the call is compelled.  Id 0 is the end token (no text). */
static const char *VOCAB[] = {
    "", "Let", " me", " think", "{", "</think>", "\n\n", "```json\n",
    "{\"function\": \"read\"", ", \"arguments\": {", "\"path\": \"", "src",
    "/main.c", "\"}}", "\n```", "Sure, here it is", " ", "\"", "```", "}",
    "{\"function\": \"add\"", "\"a\": 1, \"b\": 2}}", "ok", "\n",
};
#define VOCAB_N ((int)(sizeof(VOCAB) / sizeof(VOCAB[0])))
#define END_ID 0
#define THINK_CLOSE_ID 5
#define TOP_N 4
#define MAX_SEQ 128

/* ---- grammar ops over tgj_state ---- */
static void ops_copy(void *d, const void *s) { tgj_state_copy(d, s); }
static bool ops_allowed(const void *st, const char *t) { return tgj_token_allowed(st, t); }
static void ops_advance(void *st, const char *t) { tgj_advance(st, t); }
static bool ops_mask(const void *st) { return tgj_mask_active(st); }
static bool ops_mask_never(const void *st) { (void)st; return false; }
static bool ops_complete(const void *st) { return tgj_complete(st); }
static const gsc_grammar_ops OPS = { ops_copy, ops_allowed, ops_advance, ops_mask, ops_complete };
static const gsc_grammar_ops OPS_IGNORING_GRAMMAR = {
    ops_copy, ops_allowed, ops_advance, ops_mask_never, ops_complete };

static const char *vocab_text(void *ctx, uint32_t id) {
    (void)ctx;
    return id < (uint32_t)VOCAB_N ? VOCAB[id] : NULL;
}

/* ---- the toy model: logits are a hash of (salt, prefix) ---- */
static uint64_t mix64(uint64_t x) {
    x += 0x9E3779B97F4A7C15ull;
    x = (x ^ (x >> 30)) * 0xBF58476D1CE4E5B9ull;
    x = (x ^ (x >> 27)) * 0x94D049BB133111EBull;
    return x ^ (x >> 31);
}

static void toy_logits(uint64_t salt, const uint32_t *seq, int n, float *out) {
    uint64_t h = mix64(salt);
    for (int i = 0; i < n; i++) h = mix64(h ^ seq[i]);
    for (int v = 0; v < VOCAB_N; v++)
        out[v] = (float)(mix64(h + (uint64_t)v) >> 40) / (float)(1u << 24);
    /* Close the think block after a while, and favour the fragments that finish
     * a call, so most trials reach a complete call inside MAX_SEQ. */
    if (n >= 6 + (int)(h % 8)) out[THINK_CLOSE_ID] += 0.6f;
    static const int CALL_FRAGMENTS[] = { 7, 8, 9, 10, 13, 14 };
    for (int i = 0; i < (int)(sizeof(CALL_FRAGMENTS) / sizeof(CALL_FRAGMENTS[0])); i++)
        out[CALL_FRAGMENTS[i]] += 0.3f;
    out[END_ID] -= 0.5f;   /* an early end token would make trials trivial */
}

/* Argmax, lowest id on a tie; under `walk`'s active mask only allowed ids. */
static uint32_t masked_argmax(const float *logits, const tgj_state *walk) {
    bool masking = walk && tgj_mask_active(walk);
    int best = -1;
    for (int v = 0; v < VOCAB_N; v++) {
        if (masking && VOCAB[v][0] && !tgj_token_allowed(walk, VOCAB[v])) continue;
        if (best < 0 || logits[v] > logits[best]) best = v;
    }
    return (uint32_t)best;
}

/* The card's top-N (index, value) pairs of a dense row, in id order. */
static uint32_t top_n_pairs(const float *logits, uint32_t *index, float *value) {
    bool taken[VOCAB_N] = { false };
    for (int k = 0; k < TOP_N; k++) {
        int best = -1;
        for (int v = 0; v < VOCAB_N; v++)
            if (!taken[v] && (best < 0 || logits[v] > logits[best])) best = v;
        taken[best] = true;
    }
    uint32_t n = 0;
    for (int v = 0; v < VOCAB_N; v++)
        if (taken[v]) { index[n] = (uint32_t)v; value[n] = logits[v]; n++; }
    return n;
}

/* Plain constrained decode.  Returns the token count. */
static int decode_plain(const tgj_grammar *g, uint64_t salt, uint32_t *seq) {
    tgj_state *walk = tgj_state_new_required(g, true);
    float logits[VOCAB_N];
    int n = 0;
    while (n < MAX_SEQ) {
        toy_logits(salt, seq, n, logits);
        uint32_t tok = masked_argmax(logits, walk);
        if (tok == END_ID) break;
        seq[n++] = tok;
        tgj_advance(walk, VOCAB[tok]);
        if (tgj_complete(walk)) break;
    }
    tgj_state_free(walk);
    return n;
}

/* A row under an active mask, decided as the serve does: from the row's top-N
 * by the shortlist rule (greedy), else — row 0 only — the dense row at the
 * cycle's start, masked; a later undecidable row defers. */
typedef struct {
    uint64_t salt;
    const uint32_t *seq;
    int n;
    unsigned long dense_rows;
} masked_ctx;

static bool walk_allows(void *ctx, uint32_t id) {
    return !VOCAB[id][0] || tgj_token_allowed((const tgj_state *)ctx, VOCAB[id]);
}

static uint32_t masked_pick(void *ctx, void *walk, int row, const uint32_t *index,
                            const float *value, uint32_t count) {
    masked_ctx *m = (masked_ctx *)ctx;
    float sparse[VOCAB_N], logits[VOCAB_N];
    uint32_t ids[GMS_MAX];
    float bound;
    gms_sampling greedy = { .temperature = 0.0f, .top_k = 1, .repeat_penalty = 1.0f };
    uint32_t neg = 0xff800000u;
    for (int v = 0; v < VOCAB_N; v++) memcpy(&sparse[v], &neg, sizeof(float));
    for (uint32_t i = 0; i < count; i++) sparse[index[i]] = value[i];
    int n = gms_select_shortlist(sparse, VOCAB_N, ids, &bound);
    if (gms_mask_from_shortlist(sparse, VOCAB_N, ids, n, bound, &greedy, walk_allows, walk)) {
        int best = -1;
        for (int v = 0; v < VOCAB_N; v++)
            if (sparse[v] > -1e30f && (best < 0 || sparse[v] > sparse[best])) best = v;
        return (uint32_t)best;
    }
    if (row > 0) return GSC_DEFER;
    m->dense_rows++;
    toy_logits(m->salt, m->seq, m->n, logits);
    return masked_argmax(logits, (const tgj_state *)walk);
}

/* Drafts: the true plain continuation, with junk mixed in by the hash. */
static int make_drafts(const tgj_grammar *g, uint64_t salt, const uint32_t *seq, int n,
                       uint32_t *drafts) {
    uint32_t truth[MAX_SEQ + 4];
    int nt = decode_plain(g, salt, truth);
    int nd = 0;
    uint64_t h = mix64(salt ^ (uint64_t)n * 7919u);
    for (int j = 0; j < 3; j++) {
        h = mix64(h);
        uint32_t d = (h % 3 == 0 || n + j >= nt) ? (uint32_t)(h >> 33) % VOCAB_N
                                                 : truth[n + j];
        if (d == END_ID) break;     /* the serve cuts drafts at an end token */
        drafts[nd++] = d;
    }
    (void)seq;
    return nd;
}

/* One speculative verify cycle, accepting exactly as spec_step_ex does. */
static int verify_cycle(gsc_chooser *c, uint64_t salt, uint32_t *seq, int n,
                        const uint32_t *drafts, int nd, uint32_t *emit) {
    uint32_t index[TOP_N];
    float value[TOP_N], logits[VOCAB_N];
    uint32_t row_seq[MAX_SEQ + 8];
    int n_accept = 0;
    uint32_t next;
    memcpy(row_seq, seq, (size_t)n * sizeof(*seq));
    gsc_begin_cycle(c);
    for (;;) {
        for (int i = 0; i < n_accept; i++) row_seq[n + i] = drafts[i];
        toy_logits(salt, row_seq, n + n_accept, logits);
        uint32_t cnt = top_n_pairs(logits, index, value);
        uint32_t tok = gsc_choose(c, index, value, cnt);
        if (tok == GSC_DEFER) {
            if (n_accept == 0) return -1;   /* spec_step_ex fails the cycle */
            n_accept--;
            next = drafts[n_accept];
            break;
        }
        if (n_accept < nd && tok == drafts[n_accept]) { n_accept++; continue; }
        next = tok;
        break;
    }
    for (int i = 0; i < n_accept; i++) emit[i] = drafts[i];
    emit[n_accept] = next;
    return n_accept + 1;
}

/* Speculative constrained decode through the chooser.  Returns the count. */
static int decode_spec(const tgj_grammar *g, uint64_t salt, const gsc_grammar_ops *ops,
                       uint32_t *seq, unsigned long *cycles) {
    tgj_state *live = tgj_state_new_required(g, true);
    tgj_state *walk = tgj_state_clone(live);
    masked_ctx dctx = { salt, seq, 0, 0 };
    gsc_chooser c = { .ops = ops, .live = live, .walk = walk, .text = vocab_text,
                      .masked = masked_pick, .masked_ctx = &dctx };
    float logits[VOCAB_N];
    int n = 0;
    bool stop = false;
    while (n < MAX_SEQ && !stop) {
        uint32_t drafts[4], emit[8];
        int nd = make_drafts(g, salt, seq, n, drafts);
        nd = gsc_truncate_drafts(&c, drafts, nd);
        int ne;
        if (n == 0 || nd < 1) {   /* no cycle: one plain step */
            toy_logits(salt, seq, n, logits);
            emit[0] = masked_argmax(logits, ops == &OPS ? live : NULL);
            ne = 1;
        } else {
            dctx.n = n;
            ne = verify_cycle(&c, salt, seq, n, drafts, nd, emit);
            (*cycles)++;
            if (ne < 0) {             /* a refused cycle: report a divergence */
                seq[n++] = END_ID;
                break;
            }
        }
        for (int i = 0; i < ne && !stop; i++) {
            if (emit[i] == END_ID || n >= MAX_SEQ) { stop = true; break; }
            seq[n++] = emit[i];
            tgj_advance(live, VOCAB[emit[i]]);
            if (tgj_complete(live)) stop = true;
        }
    }
    tgj_state_free(walk);
    tgj_state_free(live);
    return n;
}

/* The text the model emitted after its think block. */
static void content_after_think(const uint32_t *seq, int n, char *out, size_t cap) {
    size_t len = 0;
    int start = 0;
    for (int i = 0; i < n; i++)
        if (seq[i] == THINK_CLOSE_ID) { start = i + 1; break; }
    out[0] = '\0';
    for (int i = start; i < n; i++) {
        size_t t = strlen(VOCAB[seq[i]]);
        if (len + t + 1 >= cap) break;
        memcpy(out + len, VOCAB[seq[i]], t);
        len += t;
        out[len] = '\0';
    }
}

/* ============================================================
 * Spec == plain over many trials; completed calls parse.
 * ============================================================ */
static void test_spec_equals_plain_constrained(void) {
    jr_value *tools = jr_parse(TOOLS_JSON);
    tgj_grammar *g = tgj_build(tools);
    int mismatches = 0, completed = 0, parsed = 0, ignoring_mismatches = 0;
    unsigned long cycles = 0;
    for (uint64_t salt = 1; salt <= 400; salt++) {
        uint32_t plain[MAX_SEQ + 8], spec[MAX_SEQ + 8], loose[MAX_SEQ + 8];
        int np = decode_plain(g, salt, plain);
        int ns = decode_spec(g, salt, &OPS, spec, &cycles);
        int nl = decode_spec(g, salt, &OPS_IGNORING_GRAMMAR, loose, &cycles);
        if (np != ns || memcmp(plain, spec, (size_t)np * sizeof(*plain)) != 0) mismatches++;
        if (np != nl || memcmp(plain, loose, (size_t)np * sizeof(*plain)) != 0)
            ignoring_mismatches++;
        tgj_state *w = tgj_state_new_required(g, true);
        for (int i = 0; i < np; i++) tgj_advance(w, VOCAB[plain[i]]);
        if (tgj_complete(w)) {
            char text[4096];
            tc_tool_call *calls = NULL;
            int n_calls = 0;
            completed++;
            content_after_think(plain, np, text, sizeof(text));
            if (tc_parse_tool_calls(text, tools, TC_DIALECT_ORNITH_JSON, &calls, &n_calls) == 0 &&
                n_calls == 1 && tc_tool_name_known(tools, calls[0].name))
                parsed++;
            tc_tool_calls_free(calls, n_calls);
        }
        tgj_state_free(w);
    }
    printf("    trials=400 cycles=%lu completed=%d parsed=%d mismatches=%d "
           "grammar-ignoring mismatches=%d\n", cycles, completed, parsed, mismatches,
           ignoring_mismatches);
    if (ignoring_mismatches == 0) {
        printf("VACUOUS: ignoring the grammar never changed the output\n");
        exit(2);
    }
    ASSERT_INT_EQ(mismatches, 0);
    ASSERT_TRUE(completed > 100);
    ASSERT_INT_EQ(parsed, completed);
    tgj_free(g);
    jr_free(tools);
}

/* ============================================================
 * Truncation: a grammar-breaking draft is cut, and nothing follows completion.
 * ============================================================ */
static void test_truncate_drafts(void) {
    jr_value *tools = jr_parse(TOOLS_JSON);
    tgj_grammar *g = tgj_build(tools);
    tgj_state *live = tgj_state_new_required(g, true);
    tgj_state *walk = tgj_state_clone(live);
    gsc_chooser c = { .ops = &OPS, .live = live, .walk = walk, .text = vocab_text };

    /* Reasoning: anything goes, '{' included. */
    uint32_t think[3] = { 4, 15, 18 };
    ASSERT_INT_EQ(gsc_truncate_drafts(&c, think, 3), 3);

    tgj_advance(live, "</think>");
    tgj_advance(live, "\n\n");
    uint32_t prose[3] = { 7, 15, 8 };          /* fence, then junk */
    ASSERT_INT_EQ(gsc_truncate_drafts(&c, prose, 3), 1);
    uint32_t good[3] = { 7, 8, 9 };
    ASSERT_INT_EQ(gsc_truncate_drafts(&c, good, 3), 3);

    tgj_advance(live, "```json\n");
    tgj_advance(live, "{\"function\": \"read\"");
    tgj_advance(live, ", \"arguments\": {");
    uint32_t finish[4] = { 10, 11, 13, 14 };   /* path, value, close, fence close */
    ASSERT_INT_EQ(gsc_truncate_drafts(&c, finish, 4), 4);
    uint32_t past[3] = { 10, 13, 14 };         /* "": an empty path is refused */
    ASSERT_INT_EQ(gsc_truncate_drafts(&c, past, 3), 1);
    tgj_advance(live, "\"path\": \"");
    tgj_advance(live, "src");
    uint32_t over[3] = { 13, 14, 22 };          /* complete after 14 → 22 cut */
    ASSERT_INT_EQ(gsc_truncate_drafts(&c, over, 3), 2);
    ASSERT_TRUE(c.drafts_cut >= 4);

    tgj_state_free(walk);
    tgj_state_free(live);
    tgj_free(g);
    jr_free(tools);
}

/* Routing: an unmasked row goes to the base chooser, a masked row to the masked
 * pick (with the private walk and the row index), and the masked pick's DEFER
 * ends the cycle. */
static int g_base_calls, g_masked_calls, g_masked_last_row;
static uint32_t base_first(void *ctx, int row, const uint32_t *index, const float *value,
                           uint32_t count) {
    (void)ctx; (void)row; (void)value; (void)count;
    g_base_calls++;
    return index[0];
}
static uint32_t masked_fence_then_defer(void *ctx, void *walk, int row, const uint32_t *index,
                                        const float *value, uint32_t count) {
    (void)ctx; (void)index; (void)value; (void)count;
    g_masked_calls++;
    g_masked_last_row = row;
    return tgj_in_call((const tgj_state *)walk) ? GSC_DEFER : 7u;   /* fence, then defer */
}

static void test_rows_route_by_mask(void) {
    jr_value *tools = jr_parse(TOOLS_JSON);
    tgj_grammar *g = tgj_build(tools);
    tgj_state *live = tgj_state_new_required(g, true);
    tgj_state *walk = tgj_state_clone(live);
    gsc_chooser c = { .ops = &OPS, .live = live, .walk = walk, .text = vocab_text,
                      .base = base_first, .masked = masked_fence_then_defer };
    uint32_t idx_close[2] = { 5, 22 };   /* "</think>" first */
    uint32_t idx_any[2] = { 15, 22 };
    float val[2] = { 1.0f, 0.5f };

    g_base_calls = g_masked_calls = 0;
    gsc_begin_cycle(&c);
    ASSERT_TRUE(gsc_choose(&c, idx_close, val, 2) == 5u);   /* reasoning: base */
    ASSERT_INT_EQ(g_base_calls, 1);
    ASSERT_TRUE(gsc_choose(&c, idx_any, val, 2) == 7u);     /* compelled: masked pick */
    ASSERT_INT_EQ(g_masked_calls, 1);
    ASSERT_INT_EQ(g_masked_last_row, 1);
    ASSERT_TRUE(gsc_choose(&c, idx_any, val, 2) == GSC_DEFER);
    ASSERT_INT_EQ(g_masked_last_row, 2);
    ASSERT_INT_EQ(g_base_calls, 1);

    tgj_state_free(walk);
    tgj_state_free(live);
    tgj_free(g);
    jr_free(tools);
}

/* Nothing past the call is committed: the row after the one that completed
 * the call defers, so the completing token ends the cycle uncommitted. */
static void test_defer_after_completion(void) {
    jr_value *tools = jr_parse(TOOLS_JSON);
    tgj_grammar *g = tgj_build(tools);
    tgj_state *live = tgj_state_new_required(g, true);
    const char *frags[] = { "</think>", "```json\n", "{\"function\": \"read\"",
                            ", \"arguments\": {", "\"path\": \"", "src", "\"}}" };
    for (int i = 0; i < 7; i++) tgj_advance(live, frags[i]);
    tgj_state *walk = tgj_state_clone(live);
    masked_ctx m = { 0, NULL, 0, 0 };
    gsc_chooser c = { .ops = &OPS, .live = live, .walk = walk, .text = vocab_text,
                      .masked = masked_pick, .masked_ctx = &m };
    uint32_t idx[2] = { 14, 22 };
    float val[2] = { 2.0f, 1.0f };
    gsc_begin_cycle(&c);
    ASSERT_TRUE(gsc_choose(&c, idx, val, 2) == 14u);  /* the fence close completes it */
    ASSERT_TRUE(c.completed);
    ASSERT_TRUE(gsc_choose(&c, idx, val, 2) == GSC_DEFER);
    tgj_state_free(walk);
    tgj_state_free(live);
    tgj_free(g);
    jr_free(tools);
}

int main(void) {
    printf("\n=== grammar_spec_chooser (speculative decoding under the grammar) ===\n\n");
    RUN_TEST(truncate_drafts);
    RUN_TEST(rows_route_by_mask);
    RUN_TEST(defer_after_completion);
    RUN_TEST(spec_equals_plain_constrained);
    TEST_REPORT();
    return __ft_failed == 0 ? 0 : 1;
}
