/* SPDX-License-Identifier: AGPL-3.0-or-later */
/* Copyright (C) 2026 David (Corey) Herrera */
/**
 * @file test_grammar_repeat_penalty.c
 * @brief While the tool grammar's mask is live, the repetition penalty does not
 *        touch the draw.  Outside the mask it still does.
 *
 * @status Complete.  CPU-only — no device, no model.
 * @issues None
 * @todo None
 *
 * The defect (Qwen3.6-35B-A3B, Q3_K_M and Q8_0, K80 serve, 2026-10-28): closing
 * tags leaked into tool arguments — `read {"path":"</parameter_unnamed>.teamide/…"}`.
 * A logprobs replay of the recorded request showed the cause.  After
 * `<parameter=path>` the model's own top token was "\n" (logprob 0.00), yet the
 * draw was "\n\n" (-14.27).  The grammar allows both.  The sampler's repetition
 * penalty (1.1, applied once PER OCCURRENCE over the last 64 tokens) had divided
 * the "\n" logit by 1.1^n, because a tool call repeats "\n" and ">" on every line.
 * The model, pushed into a blank line, wrote "</parameter"; the empty-value floor
 * then masked ">", so it wrote "_unnamed>" instead and carried on.  A greedy replay
 * (no penalty) of the same request emitted a clean call.
 *
 * The two scenarios below use the recorded margins: the "\n" vs "\n\n" step at the
 * start of a value (gap 14.27) and the ">" vs "_" step that closes a value (gap
 * 12.76).  top_k = 1 makes the draw the post-penalty argmax, so the verdict is
 * deterministic.  Scenario 4 repeats the close on Ornith's JSON dialect.  The control runs the same row with the grammar's mask OFF (AUTO,
 * no opener yet): the penalty must still apply there.
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

enum { T_NL, T_NLNL, T_LT_SLASH, T_PARAMETER, T_GT, T_UNDERSCORE, T_PATH, T_QUOTE, T_FILLER, N_NAMED };
/* Wide enough that the row is not mistaken for a device's sparse top-k row. */
#define VOCAB 512
static const char *const TEXT[N_NAMED] = {
    [T_NL] = "\n", [T_NLNL] = "\n\n", [T_LT_SLASH] = "</", [T_PARAMETER] = "parameter",
    [T_GT] = ">", [T_UNDERSCORE] = "_", [T_PATH] = "src/main.c", [T_QUOTE] = "\"",
    [T_FILLER] = "zz",
};

static const char *tok_text(void *user, uint32_t id) {
    (void)user;
    if (id >= VOCAB) return NULL;
    return id < N_NAMED ? TEXT[id] : TEXT[T_FILLER];
}
/* Hook callbacks for the two tool dialects: Qwen XML (tool_grammar) and Ornith's JSON
 * (tool_grammar_json).  The exemption keys on the hook, so both must see it. */
static int xml_allowed(void *st, const char *t) { return tg_token_allowed((tg_state *)st, t); }
static int xml_live(void *st) { return tg_mask_active((tg_state *)st); }
static int json_allowed(void *st, const char *t) { return tgj_token_allowed((tgj_state *)st, t); }
static int json_live(void *st) { return tgj_mask_active((tgj_state *)st); }
typedef struct {
    int (*allowed)(void *, const char *);
    int (*live)(void *);
} dialect_ops;
static const dialect_ops XML = { xml_allowed, xml_live };
static const dialect_ops JSON = { json_allowed, json_live };

/* The `read` tool as Argus declares it: path is required and may not be empty. */
static const char TOOLS[] =
    "[{\"type\":\"function\",\"function\":{\"name\":\"read\",\"parameters\":{"
    "\"type\":\"object\",\"properties\":{\"path\":{\"type\":\"string\",\"minLength\":1}},"
    "\"required\":[\"path\"]}}}]";

/* One draw: logits[want] = top, logits[rival] = top - gap, the rest far below;
 * `want` occurs `occurrences` times in the recent window. */
static uint32_t draw(const dialect_ops *ops, void *st, uint32_t want, uint32_t rival,
                     float gap, int occurrences)
{
    float row[VOCAB];
    uint32_t recent[64];
    for (int i = 0; i < VOCAB; i++) row[i] = -10.0f;
    row[want] = 36.0f;
    row[rival] = 36.0f - gap;
    int n = 0;
    for (; n < occurrences; n++) recent[n] = want;
    recent[n++] = T_FILLER;

    pn_tool_grammar_hook hook;
    memset(&hook, 0, sizeof(hook));
    hook.token_text = tok_text;
    hook.allowed = ops->allowed;
    hook.refresh = ops->live;
    hook.st = st;
    hook.n_vocab = VOCAB;

    pn_sampling_params sp = pn_sampling_defaults();   /* repeat_penalty 1.1 */
    sp.temperature = 0.7f;
    sp.top_k = 1;
    sp.tool_grammar = &hook;
    sp.rng_seeded = 1;
    sp.rng_state = 0x9E3779B97F4A7C15ull;
    uint32_t tok = pn_sample_next_token(row, VOCAB, &sp, recent, n);
    CHECK(sp.repeat_penalty == 1.1f, "the caller's repeat_penalty was not restored (%g)",
          (double)sp.repeat_penalty);
    return tok;
}

int main(void)
{
    jr_value *tools = jr_parse(TOOLS);
    tg_grammar *g = tg_build(tools);
    if (!g) { printf("FAIL: grammar build\n"); return 1; }

    /* 1. The value's opening newline, mask live (REQUIRED walk, inside the call). */
    {
        tg_state *st = tg_state_new_required(g, true);
        tg_advance(st, "Let me read the task file.\n</think>\n\n<tool_call>\n"
                       "<function=read>\n<parameter=path>");
        CHECK(tg_mask_active(st), "mask should be live after <parameter=path>");
        uint32_t t = draw(&XML, st, T_NL, T_NLNL, 14.27f, 6);
        CHECK(t == T_NL, "value start: drew '%s', want the model's own \"\\n\"", TEXT[t]);
        tg_state_free(st);
    }

    /* 2. The '>' that closes "</parameter" after real content, mask live. */
    {
        tg_state *st = tg_state_new_required(g, true);
        tg_advance(st, "</think>\n<tool_call>\n<function=read>\n<parameter=path>\n"
                       "src/main.c\n</parameter");
        CHECK(tg_mask_active(st), "mask should be live inside the close tag");
        uint32_t t = draw(&XML, st, T_GT, T_UNDERSCORE, 12.76f, 6);
        CHECK(t == T_GT, "value close: drew '%s', want \">\"", TEXT[t]);
        tg_state_free(st);
    }

    /* 3. Control: AUTO walk, no opener yet, so the mask is off.  The repetition
     * penalty still applies there: "\n" divided by 1.1^6 loses to "\n\n". */
    {
        tg_state *st = tg_state_new(g);
        tg_advance(st, "Some prose before any call.");
        CHECK(!tg_mask_active(st), "mask should be off in AUTO prose");
        uint32_t t = draw(&XML, st, T_NL, T_NLNL, 14.27f, 6);
        CHECK(t == T_NLNL, "prose: drew '%s', want the penalised runner-up \"\\n\\n\"", TEXT[t]);
        tg_state_free(st);
    }

    /* 4. Ornith's JSON dialect, mask live: the '"' that closes a string value. */
    {
        tgj_grammar *gj = tgj_build(tools);
        tgj_state *st = gj ? tgj_state_new_required(gj, true) : NULL;
        CHECK(st != NULL, "JSON grammar build");
        if (st) {
            tgj_advance(st, "</think>\n{\"function\": \"read\", \"arguments\": "
                            "{\"path\": \"src/main.c");
            CHECK(tgj_mask_active(st), "JSON mask should be live inside the value");
            uint32_t t = draw(&JSON, st, T_QUOTE, T_UNDERSCORE, 12.76f, 6);
            CHECK(t == T_QUOTE, "JSON value close: drew '%s', want '\"'", TEXT[t]);
            tgj_state_free(st);
        }
        tgj_free(gj);
    }

    tg_free(g);
    jr_free(tools);
    printf("test_grammar_repeat_penalty: %d/%d checks passed\n", g_checks - g_fail, g_checks);
    return g_fail ? 1 : 0;
}
