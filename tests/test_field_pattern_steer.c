/* SPDX-License-Identifier: AGPL-3.0-or-later */
/* Copyright (C) 2026 David (Corey) Herrera */
/**
 * @file test_field_pattern_steer.c
 * @brief A refused close on a patterned value is steered into the pattern's
 *        forced continuation (" from: " for Argus's `finding`), instead of only
 *        being refused.
 *
 * @status Complete.  CPU-only — no device, no model.
 * @issues None
 * @todo None
 *
 * The defect (rm4 runs, 2026-11-17): with `field_patterns` on `finding`, a model
 * that tried to close a fact with no " from: " had its close masked and simply
 * wrote on — the pattern lets almost any byte into the fact — until the 8,192
 * token cap; the serve dropped the call (call_incomplete).
 *
 * Part 1 checks tg_steer / tgj_steer directly: which refusals steer, which do
 * not, and that a steered walk allows only the forced literal.  Part 2 runs a
 * scripted "model" through the real sampler (pn_sample_next_token) in both
 * dialects: its top token at the end of the fact is the close; with the steer
 * armed the call must come out as "<fact> from: Makefile:3" and complete, and
 * without it the model writes on and the call never closes.  A value whose
 * close is legal draws the same tokens with and without the steer armed, and
 * the steer is never consulted.
 */
#include "../src/core/engine.h"
#include "../src/serve/json_reader.h"
#include "../src/serve/tool_grammar.h"
#include "../src/serve/tool_grammar_json.h"

#include <stdint.h>
#include <stdio.h>
#include <string.h>

static int g_fail = 0, g_checks = 0;
#define CHECK(cond, ...) do { \
    g_checks++; \
    if (!(cond)) { g_fail++; printf("  FAIL: "); printf(__VA_ARGS__); printf("\n"); } \
} while (0)

/* `read` as Argus declares it, with the client's `finding` pattern. */
static const char TOOLS[] =
    "[{\"type\":\"function\",\"function\":{\"name\":\"read\",\"parameters\":{"
    "\"type\":\"object\",\"properties\":{"
    "\"path\":{\"type\":\"string\",\"minLength\":1},"
    "\"finding\":{\"type\":\"string\",\"minLength\":1}},"
    "\"required\":[\"path\",\"finding\"]}}}]";
static const char FIELD[] =
    "{\"read\":{\"finding\":\"-|[^\\\\n]+ from: [^\\\\s,:]+:[0-9]+(-[0-9]+)?"
    "(, [^\\\\s,:]+:[0-9]+(-[0-9]+)?)*\"}}";

static const char XML_HEAD[] = "<tool_call>\n<function=read>\n"
                               "<parameter=path>\nMakefile\n</parameter>\n"
                               "<parameter=finding>\n";
static const char JSON_HEAD[] =
    "\n```json\n{\"function\": \"read\", \"arguments\": {\"path\": \"Makefile\", \"finding\": \"";

/* ── Part 1: the grammar entry points ── */

static tg_state *xml_at_value(tg_grammar *g, const char *value) {
    tg_state *st = tg_state_new(g);
    tg_advance(st, XML_HEAD);
    tg_advance(st, value);
    return st;
}

static tgj_state *json_at_value(tgj_grammar *g, const char *value) {
    tgj_state *st = tgj_state_new(g);
    tgj_advance(st, JSON_HEAD);
    tgj_advance(st, value);
    return st;
}

static void test_xml_steer(tg_grammar *g) {
    const char *fact = "the Makefile builds with gcc";

    /* The trained tail's newline, refused on a from:-less fact: steered. */
    tg_state *st = xml_at_value(g, fact);
    CHECK(!tg_token_allowed(st, "\n"), "xml: the close of a from:-less fact is refused");
    CHECK(tg_steer(st, "\n"), "xml: a refused \"\\n\" close steers");
    /* Only the forced literal now. */
    CHECK(!tg_token_allowed(st, "\n"), "xml: steered: \"\\n\" still refused");
    CHECK(!tg_token_allowed(st, " and"), "xml: steered: other text refused");
    CHECK(!tg_token_allowed(st, "</parameter>"), "xml: steered: the close refused");
    CHECK(tg_token_allowed(st, " from"), "xml: steered: \" from\" allowed");
    CHECK(tg_token_allowed(st, " "), "xml: steered: \" \" allowed");
    CHECK(tg_token_allowed(st, " from: Makefile:3"), "xml: steered: literal then a path allowed");
    CHECK(!tg_steer(st, "\n"), "xml: no second steer while one is pending");
    tg_advance(st, " from");
    CHECK(!tg_token_allowed(st, " "), "xml: steered: after \" from\" only ':' (got ' ')");
    tg_advance(st, ": Makefile:3");
    CHECK(tg_token_allowed(st, "\n</parameter>\n</function>\n</tool_call>"),
          "xml: steered value closes once it matches");
    tg_advance(st, "\n</parameter>\n</function>\n</tool_call>");
    CHECK(tg_complete(st), "xml: steered call completes");
    tg_state_free(st);

    /* The close as a whole token, and a token that writes on and then closes. */
    st = xml_at_value(g, fact);
    CHECK(tg_steer(st, "\n</parameter>"), "xml: \"\\n</parameter>\" steers");
    tg_state_free(st);
    st = xml_at_value(g, fact);
    CHECK(tg_steer(st, "</parameter>"), "xml: \"</parameter>\" (no newline) steers");
    tg_state_free(st);
    st = xml_at_value(g, fact);
    CHECK(tg_steer(st, ".\n"), "xml: \".\\n\" steers");
    tg_state_free(st);

    /* Not steered: an allowed token, a non-close refusal, an empty value, a
     * value already mid-"</parameter>", a matching value, no pattern. */
    st = xml_at_value(g, fact);
    CHECK(!tg_steer(st, " and"), "xml: an allowed token does not steer");
    tg_state_free(st);
    /* After "from: " a path class comes next: the close is refused but no
     * byte is forced, so there is nothing to steer to. */
    st = xml_at_value(g, "gcc from: ");
    CHECK(!tg_token_allowed(st, "\n"), "xml: a close right after \"from: \" is refused");
    CHECK(!tg_steer(st, "\n"), "xml: a refusal with no forced bytes does not steer");
    tg_state_free(st);
    st = xml_at_value(g, "");
    CHECK(!tg_steer(st, "</parameter>"), "xml: an empty value does not steer");
    tg_state_free(st);
    st = xml_at_value(g, "gcc</");
    CHECK(!tg_steer(st, "parameter>"), "xml: a value mid-\"</parameter>\" does not steer");
    tg_state_free(st);
    st = xml_at_value(g, "gcc from: Makefile:3");
    CHECK(tg_token_allowed(st, "\n"), "xml: a matching value closes");
    CHECK(!tg_steer(st, "\n"), "xml: a matching value does not steer");
    tg_state_free(st);
    st = tg_state_new(g);
    tg_advance(st, "<tool_call>\n<function=read>\n<parameter=path>\nMake");
    CHECK(!tg_steer(st, "\n</parameter>"), "xml: an unpatterned value does not steer");
    tg_state_free(st);
}

static void test_json_steer(tgj_grammar *g) {
    const char *fact = "the Makefile builds with gcc";

    tgj_state *st = json_at_value(g, fact);
    CHECK(!tgj_token_allowed(st, "\""), "json: the close of a from:-less fact is refused");
    CHECK(tgj_steer(st, "\"}}"), "json: a refused \"\\\"}}\" close steers");
    CHECK(!tgj_token_allowed(st, "\""), "json: steered: the quote still refused");
    CHECK(!tgj_token_allowed(st, " and"), "json: steered: other text refused");
    CHECK(tgj_token_allowed(st, " from"), "json: steered: \" from\" allowed");
    CHECK(tgj_token_allowed(st, " from: Makefile:3\""), "json: steered: literal, path, close");
    tgj_advance(st, " from: Makefile:3\"}}\n```");
    CHECK(tgj_complete(st), "json: steered call completes");
    tgj_state_free(st);

    st = json_at_value(g, fact);
    CHECK(tgj_steer(st, "\""), "json: '\"' steers");
    tgj_state_free(st);
    st = json_at_value(g, fact);
    CHECK(tgj_steer(st, ".\","), "json: '.\",' steers");
    tgj_state_free(st);

    st = json_at_value(g, fact);
    CHECK(!tgj_steer(st, " and"), "json: an allowed token does not steer");
    tgj_state_free(st);
    st = json_at_value(g, "gcc");
    CHECK(!tgj_token_allowed(st, "\\q"), "json: an invalid escape is refused");
    CHECK(!tgj_steer(st, "\\q"), "json: a non-close refusal does not steer");
    tgj_state_free(st);
    st = json_at_value(g, "gcc from: ");
    CHECK(!tgj_steer(st, "\""), "json: a refusal with no forced bytes does not steer");
    tgj_state_free(st);
    st = json_at_value(g, "");
    CHECK(!tgj_steer(st, "\""), "json: an empty value does not steer");
    tgj_state_free(st);
    st = json_at_value(g, " ");
    CHECK(!tgj_steer(st, "\""), "json: a whitespace-only value does not steer");
    tgj_state_free(st);
    st = json_at_value(g, "gcc\\");
    CHECK(!tgj_steer(st, "\""), "json: an escaped quote is not a close");
    tgj_state_free(st);
    st = json_at_value(g, "gcc from: Makefile:3");
    CHECK(!tgj_steer(st, "\""), "json: a matching value does not steer");
    tgj_state_free(st);
}

/* ── Part 2: a scripted model through the sampler ── */

enum {
    T_FACT, T_LEGAL, T_DASH, T_NL, T_AND, T_FROM, T_COLON, T_SP, T_PATH_SP, T_PATH,
    T_PCLOSE, T_FCLOSE, T_TCLOSE, T_JCLOSE, T_FENCE, T_FILLER, N_NAMED
};
#define VOCAB 512   /* wide enough not to read as a device's sparse top-k row */
static const char *const TEXT[N_NAMED] = {
    [T_FACT] = "the Makefile builds with gcc", [T_LEGAL] = "gcc from: Makefile:3",
    [T_DASH] = "-", [T_NL] = "\n", [T_AND] = " and", [T_FROM] = " from", [T_COLON] = ":",
    [T_SP] = " ", [T_PATH_SP] = " Makefile:3", [T_PATH] = "Makefile:3",
    [T_PCLOSE] = "</parameter>", [T_FCLOSE] = "\n</function>", [T_TCLOSE] = "\n</tool_call>",
    [T_JCLOSE] = "\"}}", [T_FENCE] = "\n```", [T_FILLER] = "zz",
};

static const char *tok_text(void *user, uint32_t id) {
    (void)user;
    if (id >= VOCAB) return NULL;
    return id < N_NAMED ? TEXT[id] : TEXT[T_FILLER];
}

static bool ends_with(const char *s, const char *tail) {
    size_t n = strlen(s), m = strlen(tail);
    return n >= m && memcmp(s + n - m, tail, m) == 0;
}

/* The model: what it ranks first after `text` (everything since the head).  It
 * opens with `first`, completes a " from" it sees written, and otherwise wants
 * to close; " and" (writing on) is always its runner-up. */
static uint32_t model_top(bool json, uint32_t first, const char *text) {
    if (!*text) return first;
    if (ends_with(text, " from")) return T_COLON;
    if (ends_with(text, "from:")) return T_PATH_SP;
    if (ends_with(text, "from: ")) return T_PATH;
    if (json) return ends_with(text, "}}") ? T_FENCE : T_JCLOSE;
    if (ends_with(text, "</function>")) return T_TCLOSE;
    if (ends_with(text, "</parameter>")) return T_FCLOSE;
    if (ends_with(text, "\n")) return T_PCLOSE;
    return T_NL;
}

typedef struct {
    bool json;
    void *st;
} walk_ref;

static int steer_calls, steer_moves;
static int xml_allowed(void *st, const char *t) { return tg_token_allowed((tg_state *)st, t); }
static int xml_live(void *st) { return tg_mask_active((tg_state *)st); }
static void xml_advance(void *st, const char *t) { tg_advance((tg_state *)st, t); }
static int xml_steer(void *st, const char *t) {
    steer_calls++;
    int moved = tg_steer((tg_state *)st, t);
    steer_moves += moved;
    return moved;
}
static int json_allowed(void *st, const char *t) { return tgj_token_allowed((tgj_state *)st, t); }
static int json_live(void *st) { return tgj_mask_active((tgj_state *)st); }
static void json_advance(void *st, const char *t) { tgj_advance((tgj_state *)st, t); }
static int json_steer(void *st, const char *t) {
    steer_calls++;
    int moved = tgj_steer((tgj_state *)st, t);
    steer_moves += moved;
    return moved;
}

typedef struct {
    bool steer;            /* arm hook.steer (what the serve does with field_patterns) */
    bool driver_advance;   /* the batched driver's shape: the sampler advances */
    float temperature;
} run_opts;

/* Decode up to 40 tokens from the value's start; the text since the head goes
 * to `out`.  Returns whether the call completed. */
static bool run(bool json, const void *grammar, uint32_t first, run_opts o, char *out, size_t cap)
{
    void *st;
    if (json) {
        st = tgj_state_new((const tgj_grammar *)grammar);
        tgj_advance(st, JSON_HEAD);
    } else {
        st = tg_state_new((const tg_grammar *)grammar);
        tg_advance(st, XML_HEAD);
    }
    pn_tool_grammar_hook hook;
    memset(&hook, 0, sizeof(hook));
    hook.st = st;
    hook.token_text = tok_text;
    hook.allowed = json ? json_allowed : xml_allowed;
    hook.refresh = json ? json_live : xml_live;
    hook.n_vocab = VOCAB;
    if (o.steer) hook.steer = json ? json_steer : xml_steer;
    if (o.driver_advance) hook.advance = json ? json_advance : xml_advance;

    pn_sampling_params sp = pn_sampling_defaults();
    sp.temperature = o.temperature;
    sp.top_k = 1;
    sp.tool_grammar = &hook;
    sp.rng_seeded = 1;
    sp.rng_state = 0x9E3779B97F4A7C15ull;

    out[0] = '\0';
    bool done = false;
    for (int step = 0; step < 40 && !done; step++) {
        float row[VOCAB];
        for (int i = 0; i < VOCAB; i++) row[i] = -10.0f;
        row[T_FROM] = 1.0f;
        row[T_COLON] = row[T_SP] = row[T_PATH] = row[T_PATH_SP] = 0.0f;
        row[T_AND] = 5.0f;
        row[model_top(json, first, out)] = 10.0f;
        uint32_t tok = pn_sample_next_token(row, VOCAB, &sp, NULL, 0);
        if (!o.driver_advance) {
            if (json) tgj_advance(st, TEXT[tok < N_NAMED ? tok : T_FILLER]);
            else tg_advance(st, TEXT[tok < N_NAMED ? tok : T_FILLER]);
        }
        strncat(out, tok_text(NULL, tok), cap - strlen(out) - 1);
        done = json ? tgj_complete(st) : tg_complete(st);
    }
    if (json) tgj_state_free(st);
    else tg_state_free(st);
    return done;
}

static void test_scripted(bool json, const void *g) {
    const char *dialect = json ? "json" : "xml";
    char out[1024], ctl[1024];
    const char *want = json ? "the Makefile builds with gcc from: Makefile:3\"}}\n```"
                            : "the Makefile builds with gcc from: Makefile:3\n</parameter>"
                              "\n</function>\n</tool_call>";
    const run_opts shapes[] = {
        { true, false, 0.0f }, { true, false, 0.7f }, { true, true, 0.0f }, { true, true, 0.7f },
    };
    for (size_t i = 0; i < sizeof(shapes) / sizeof(shapes[0]); i++) {
        steer_moves = 0;
        bool done = run(json, g, T_FACT, shapes[i], out, sizeof(out));
        CHECK(done && strcmp(out, want) == 0,
              "%s steer (driver_advance=%d T=%.1f): completed=%d, got \"%s\"", dialect,
              shapes[i].driver_advance, (double)shapes[i].temperature, done, out);
        CHECK(steer_moves == 1, "%s steer: %d steers, want 1", dialect, steer_moves);
    }

    /* The defect, unarmed: the close stays refused and the model writes on. */
    run_opts off = { false, false, 0.0f };
    bool done = run(json, g, T_FACT, off, out, sizeof(out));
    CHECK(!done && strstr(out, " and and"), "%s unarmed: should write on unclosed, got \"%s\"",
          dialect, out);

    /* A value whose close is legal: same draws armed or not, steer never asked. */
    const uint32_t legal[] = { T_DASH, T_LEGAL };
    for (size_t i = 0; i < 2; i++) {
        steer_calls = 0;
        run_opts on = { true, false, 0.0f };
        bool a = run(json, g, legal[i], on, out, sizeof(out));
        bool b = run(json, g, legal[i], off, ctl, sizeof(ctl));
        CHECK(a && b && strcmp(out, ctl) == 0, "%s legal \"%s\": armed \"%s\" vs unarmed \"%s\"",
              dialect, TEXT[legal[i]], out, ctl);
        CHECK(steer_calls == 0, "%s legal \"%s\": steer asked %d times", dialect,
              TEXT[legal[i]], steer_calls);
    }
}

int main(void)
{
    jr_value *tools = jr_parse(TOOLS);
    jr_value *field = jr_parse(FIELD);
    tg_grammar *g = tg_build(tools);
    tgj_grammar *gj = tgj_build(tools);
    if (!g || !gj || !tg_apply_field_patterns(g, field) || !tgj_apply_field_patterns(gj, field)) {
        printf("FAIL: grammar build\n");
        return 1;
    }

    test_xml_steer(g);
    test_json_steer(gj);
    test_scripted(false, g);
    test_scripted(true, gj);

    tg_free(g);
    tgj_free(gj);
    jr_free(field);
    jr_free(tools);
    printf("test_field_pattern_steer: %d/%d checks passed\n", g_checks - g_fail, g_checks);
    return g_fail ? 1 : 0;
}
