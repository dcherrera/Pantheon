/* SPDX-License-Identifier: AGPL-3.0-or-later */
/* Copyright (C) 2026 David (Corey) Herrera */
/*
 * Unit tests for the streaming reasoning/content/tool-call splitter
 * (src/serve/serve_emitter.c).
 *
 * Focus is Phase 2 wrapper-tolerant STREAM detection: a bare "<function=" opener
 * (no "<tool_call>" wrapper), which nex-n2-mini at Q3_K_M emits most of the time,
 * must be diverted into the raw `tool` channel (→ sg_result.tool_raw) instead of
 * being streamed out as content — including when the opener straddles two token
 * deltas (em_feed calls).  With tools NOT armed, a "<function=" stays content.
 * The canonical "<tool_call>" wrapper is detected regardless.
 *
 * The emitter depends only on libc (no engine/CUDA), so it links standalone.
 */

#include "lib/test.h"
#include "../src/serve/serve_emitter.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

/* Delta collector: concatenates every content delta (is_reasoning==0) the
 * emitter streams, so a test can assert what was (and was not) emitted as
 * visible answer content. */
typedef struct { char buf[4096]; size_t len; } sink;

static void sink_cb(const char *text, int is_reasoning,
                    const sg_token_logprob *const *lps, int n_lps,
                    void *ud)
{
    (void)lps; (void)n_lps;
    if (is_reasoning) return;   /* only track answer content for these tests */
    sink *s = (sink *)ud;
    size_t n = strlen(text);
    if (s->len + n < sizeof(s->buf)) {
        memcpy(s->buf + s->len, text, n);
        s->len += n;
        s->buf[s->len] = '\0';
    }
}

/* Drive the emitter out of the <think> block so we are on the content channel
 * (generation begins inside <think>; "</think>" flips to content). */
static void enter_content(emitter *em)
{
    em_feed(em, "</think>");
}

/* ============================================================
 * Bare "<function=" opener mid-stream (tools armed) → captured to tool channel,
 * NOT streamed as content.
 * ============================================================ */

static void test_bare_opener_captured(void) {
    sink s = {0};
    emitter em;
    em_init(&em, sink_cb, &s, NULL, 0, /*has_tools=*/true, TC_DIALECT_QWEN_XML);
    enter_content(&em);

    em_feed(&em, "Sure, let me help. ");
    em_feed(&em, "<function=glob>\n<parameter=pattern>README.md</parameter>\n</function>");
    em_flush(&em);

    /* The pre-opener prose streamed as content; the call did NOT. */
    ASSERT_STR_EQ(s.buf, "Sure, let me help. ");
    /* The tool channel captured from the opener onward, verbatim. */
    ASSERT_NOT_NULL(em.tool.data);
    ASSERT_STR_EQ(em.tool.data,
        "<function=glob>\n<parameter=pattern>README.md</parameter>\n</function>");
    em_free(&em);
}

/* ============================================================
 * Bare "<function=" opener SPLIT across two token deltas → still captured
 * (the hold-back must span the boundary). Several split points.
 * ============================================================ */

static void feed_split_and_check(const char *a, const char *b) {
    sink s = {0};
    emitter em;
    em_init(&em, sink_cb, &s, NULL, 0, /*has_tools=*/true, TC_DIALECT_QWEN_XML);
    enter_content(&em);

    em_feed(&em, a);
    em_feed(&em, b);
    em_flush(&em);

    /* Nothing of the opener leaked to content. */
    ASSERT_TRUE(strstr(s.buf, "<function") == NULL);
    ASSERT_TRUE(strstr(s.buf, "function=") == NULL);
    /* The whole call landed in the tool channel. */
    ASSERT_NOT_NULL(em.tool.data);
    ASSERT_STR_EQ(em.tool.data,
        "<function=bash>\n<parameter=cmd>ls</parameter>\n</function>");
    em_free(&em);
}

static void test_split_opener_captured(void) {
    const char *full =
        "<function=bash>\n<parameter=cmd>ls</parameter>\n</function>";
    /* Split right in the middle of "<function=" (after "<fun"). */
    feed_split_and_check("<fun", "ction=bash>\n<parameter=cmd>ls</parameter>\n</function>");
    /* Split after "<function" (before "="). */
    feed_split_and_check("<function", "=bash>\n<parameter=cmd>ls</parameter>\n</function>");
    /* Split after just "<". */
    feed_split_and_check("<", "function=bash>\n<parameter=cmd>ls</parameter>\n</function>");
    (void)full;
}

/* Split with leading content before the (split) opener: prose streams, opener
 * held across the boundary, call captured. */
static void test_split_opener_with_prefix(void) {
    sink s = {0};
    emitter em;
    em_init(&em, sink_cb, &s, NULL, 0, /*has_tools=*/true, TC_DIALECT_QWEN_XML);
    enter_content(&em);

    em_feed(&em, "ok <func");
    em_feed(&em, "tion=glob><parameter=pattern>*.c</parameter></function>");
    em_flush(&em);

    ASSERT_STR_EQ(s.buf, "ok ");
    ASSERT_NOT_NULL(em.tool.data);
    ASSERT_STR_EQ(em.tool.data,
        "<function=glob><parameter=pattern>*.c</parameter></function>");
    em_free(&em);
}

/* ============================================================
 * Tools NOT armed → a "<function=" stays content (no diversion).
 * ============================================================ */

static void test_tools_off_stays_content(void) {
    sink s = {0};
    emitter em;
    em_init(&em, sink_cb, &s, NULL, 0, /*has_tools=*/false, TC_DIALECT_QWEN_XML);
    enter_content(&em);

    const char *txt =
        "Use <function=foo> in your code like <function=bar>.";
    em_feed(&em, txt);
    em_flush(&em);

    /* Everything streamed as content; nothing captured to the tool channel. */
    ASSERT_STR_EQ(s.buf, txt);
    ASSERT_NULL(em.tool.data);
    em_free(&em);
}

/* Even split across tokens, tools-off keeps "<function=" as content. */
static void test_tools_off_split_stays_content(void) {
    sink s = {0};
    emitter em;
    em_init(&em, sink_cb, &s, NULL, 0, /*has_tools=*/false, TC_DIALECT_QWEN_XML);
    enter_content(&em);

    em_feed(&em, "call <func");
    em_feed(&em, "tion=x> now");
    em_flush(&em);

    ASSERT_STR_EQ(s.buf, "call <function=x> now");
    ASSERT_NULL(em.tool.data);
    em_free(&em);
}

/* ============================================================
 * Canonical "<tool_call>" wrapper still diverts (regression), even tools-off:
 * "<tool_call>" detection is unconditional.
 * ============================================================ */

static void test_wrapped_opener_captured(void) {
    sink s = {0};
    emitter em;
    em_init(&em, sink_cb, &s, NULL, 0, /*has_tools=*/false, TC_DIALECT_QWEN_XML);
    enter_content(&em);

    em_feed(&em, "here: <tool_call><function=glob></function></tool_call>");
    em_flush(&em);

    ASSERT_STR_EQ(s.buf, "here: ");
    ASSERT_NOT_NULL(em.tool.data);
    ASSERT_STR_EQ(em.tool.data,
        "<tool_call><function=glob></function></tool_call>");
    em_free(&em);
}

/* ============================================================
 * Earliest-opener-wins: a "<tool_call>" that precedes a bare "<function=" opens
 * on the wrapper (captures from it, so the inner "<function=" is inside tool).
 * ============================================================ */

static void test_wrapped_before_bare(void) {
    sink s = {0};
    emitter em;
    em_init(&em, sink_cb, &s, NULL, 0, /*has_tools=*/true, TC_DIALECT_QWEN_XML);
    enter_content(&em);

    em_feed(&em, "<tool_call>\n<function=glob></function>\n</tool_call>");
    em_flush(&em);

    ASSERT_STR_EQ(s.buf, "");
    ASSERT_NOT_NULL(em.tool.data);
    ASSERT_STR_EQ(em.tool.data,
        "<tool_call>\n<function=glob></function>\n</tool_call>");
    em_free(&em);
}

/* ============================================================
 * Plausible-shape guard: a stray "<function=" with no name char after '=' is
 * NOT treated as an opener (stays content) even with tools armed.
 * ============================================================ */

static void test_stray_fn_fragment_stays_content(void) {
    sink s = {0};
    emitter em;
    em_init(&em, sink_cb, &s, NULL, 0, /*has_tools=*/true, TC_DIALECT_QWEN_XML);
    enter_content(&em);

    /* "<function=>" (empty name) and "<function= " (space) are not openers. */
    const char *txt = "the tag <function=> or <function= is malformed";
    em_feed(&em, txt);
    em_flush(&em);

    ASSERT_STR_EQ(s.buf, txt);
    ASSERT_NULL(em.tool.data);
    em_free(&em);
}

/* A real opener AFTER a stray fragment is still found (scan skips the stray). */
static void test_real_opener_after_stray(void) {
    sink s = {0};
    emitter em;
    em_init(&em, sink_cb, &s, NULL, 0, /*has_tools=*/true, TC_DIALECT_QWEN_XML);
    enter_content(&em);

    em_feed(&em, "junk <function=> then <function=glob></function>");
    em_flush(&em);

    ASSERT_STR_EQ(s.buf, "junk <function=> then ");
    ASSERT_NOT_NULL(em.tool.data);
    ASSERT_STR_EQ(em.tool.data, "<function=glob></function>");
    em_free(&em);
}

/* ============================================================
 * No opener at all → pure content, nothing captured.
 * ============================================================ */

static void test_no_opener_all_content(void) {
    sink s = {0};
    emitter em;
    em_init(&em, sink_cb, &s, NULL, 0, /*has_tools=*/true, TC_DIALECT_QWEN_XML);
    enter_content(&em);

    const char *txt = "Just a normal answer with no tool calls.";
    em_feed(&em, txt);
    em_flush(&em);

    ASSERT_STR_EQ(s.buf, txt);
    ASSERT_NULL(em.tool.data);
    em_free(&em);
}

/* ============================================================
 * MiniCPM5 attribute-style opener "<function name=...>" (tools armed) →
 * captured to the tool channel from "<function" onward, NOT streamed as
 * content. Distinct from the Qwen "<function=" form.
 * ============================================================ */

static void test_minicpm_opener_captured(void) {
    sink s = {0};
    emitter em;
    em_init(&em, sink_cb, &s, NULL, 0, /*has_tools=*/true, TC_DIALECT_QWEN_XML);
    enter_content(&em);

    em_feed(&em, "Dispatching now. ");
    em_feed(&em, "<function name=\"dispatch_agent\">"
                 "<param name=\"subtask\">write a haiku</param></function>");
    em_flush(&em);

    ASSERT_STR_EQ(s.buf, "Dispatching now. ");
    ASSERT_NOT_NULL(em.tool.data);
    ASSERT_STR_EQ(em.tool.data,
        "<function name=\"dispatch_agent\">"
        "<param name=\"subtask\">write a haiku</param></function>");
    em_free(&em);
}

/* The MiniCPM opener SPLIT across two deltas is still captured (the hold-back
 * must span "<function " across the boundary). */
static void test_minicpm_opener_split(void) {
    sink s = {0};
    emitter em;
    em_init(&em, sink_cb, &s, NULL, 0, /*has_tools=*/true, TC_DIALECT_QWEN_XML);
    enter_content(&em);

    em_feed(&em, "ok <func");
    em_feed(&em, "tion name=\"f\"><param name=\"x\">y</param></function>");
    em_flush(&em);

    ASSERT_STR_EQ(s.buf, "ok ");
    ASSERT_TRUE(strstr(s.buf, "<function") == NULL);
    ASSERT_NOT_NULL(em.tool.data);
    ASSERT_STR_EQ(em.tool.data,
        "<function name=\"f\"><param name=\"x\">y</param></function>");
    em_free(&em);
}

/* With tools OFF, "<function name=" stays content (never diverted). */
static void test_minicpm_opener_tools_off_stays_content(void) {
    sink s = {0};
    emitter em;
    em_init(&em, sink_cb, &s, NULL, 0, /*has_tools=*/false, TC_DIALECT_QWEN_XML);
    enter_content(&em);

    const char *txt = "the tag <function name=\"x\"> is literal here";
    em_feed(&em, txt);
    em_flush(&em);

    ASSERT_STR_EQ(s.buf, txt);
    ASSERT_NULL(em.tool.data);
    em_free(&em);
}

/* The grammar-forced dialect (bare "<function=" opener, but each value wrapped in
 * <value>…</value>) diverts EXACTLY like a plain bare call — the emitter captures
 * from the opener to end raw (the <value> wrapper is inside the span), nothing
 * leaks to content. The parser then unwraps <value>. Streamed across two deltas. */
static void test_forced_value_wrapper_captured(void) {
    sink s = {0};
    emitter em;
    em_init(&em, sink_cb, &s, NULL, 0, /*has_tools=*/true, TC_DIALECT_QWEN_XML);
    enter_content(&em);

    em_feed(&em, "Delegating. ");
    em_feed(&em, "<function=dispatch_agent>\n<parameter=subtask>\n");
    em_feed(&em, "<value>\nbuild the parser module\n</value>\n</parameter>\n</function>");
    em_flush(&em);

    ASSERT_STR_EQ(s.buf, "Delegating. ");
    ASSERT_TRUE(strstr(s.buf, "<function") == NULL);
    ASSERT_TRUE(strstr(s.buf, "<value>") == NULL);
    ASSERT_NOT_NULL(em.tool.data);
    ASSERT_STR_EQ(em.tool.data,
        "<function=dispatch_agent>\n<parameter=subtask>\n"
        "<value>\nbuild the parser module\n</value>\n</parameter>\n</function>");
    em_free(&em);
}

/* Live MiniCPM/grammar-forced token split: the model emits "<function" and "="
 * as SEPARATE tokens, so the bare "<function=" opener is completed only when the
 * "=" token lands — with nothing after it yet. The plausibility gate rejects a
 * "<function=" with no name char, so without a hold-back the complete opener
 * flushes to content and the divert never fires (the observed tool_calls:null).
 * The opener must be held until the following name char arrives, then diverted. */
static void test_split_opener_after_equals(void) {
    sink s = {0};
    emitter em;
    em_init(&em, sink_cb, &s, NULL, 0, /*has_tools=*/true, TC_DIALECT_QWEN_XML);
    enter_content(&em);

    /* Exactly the K80 test machine token stream: </think> \n\n <function (empty) = dis ... */
    em_feed(&em, "\n\n");
    em_feed(&em, "<function");
    em_feed(&em, "");
    em_feed(&em, "=");
    em_feed(&em, "dispatch_agent>\n<parameter=subtask>\n");
    em_feed(&em, "<value>build it</value>\n</paramet>\n</function>");
    em_flush(&em);

    /* Nothing of the call leaked to content. */
    ASSERT_TRUE(strstr(s.buf, "<function") == NULL);
    ASSERT_NOT_NULL(em.tool.data);
    ASSERT_STR_EQ(em.tool.data,
        "<function=dispatch_agent>\n<parameter=subtask>\n"
        "<value>build it</value>\n</paramet>\n</function>");
    em_free(&em);
}

/* A "<function=" at the buffer end that is NEVER followed (stream ends) flushes
 * to content — it was a stray prose fragment, not a real opener. */
static void test_trailing_fn_open_never_followed_is_content(void) {
    sink s = {0};
    emitter em;
    em_init(&em, sink_cb, &s, NULL, 0, /*has_tools=*/true, TC_DIALECT_QWEN_XML);
    enter_content(&em);

    em_feed(&em, "the literal <function=");
    em_flush(&em);   /* stream ends here — held opener must flush as content */

    ASSERT_STR_EQ(s.buf, "the literal <function=");
    ASSERT_NULL(em.tool.data);
    em_free(&em);
}

/* ============================================================
 * Reasoning-budget force-close: the AUTO-path budget in sg_run_completion, once
 * over budget, force-DECODES the "</think>" token(s) and feeds each through
 * em_feed exactly like a sampled token.  This models the emitter side of that:
 * reasoning tokens route to the reasoning channel, then the force-injected
 * "</think>" flips the emitter to content, and the following answer streams as
 * content — proving the mechanism the force-decode relies on.
 * ============================================================ */

/* Delta collector that keeps BOTH channels separately so this test can assert
 * the reasoning/content split the budget force-close produces. */
typedef struct { char r[4096]; size_t rl; char c[4096]; size_t cl; } split_sink;

static void split_cb(const char *text, int is_reasoning,
                     const sg_token_logprob *const *lps, int n_lps,
                     void *ud)
{
    (void)lps; (void)n_lps;
    split_sink *s = (split_sink *)ud;
    char *buf = is_reasoning ? s->r : s->c;
    size_t *len = is_reasoning ? &s->rl : &s->cl;
    size_t n = strlen(text);
    if (*len + n < 4096) { memcpy(buf + *len, text, n); *len += n; buf[*len] = '\0'; }
}

static void test_budget_force_close_flips_to_content(void) {
    split_sink s = {0};
    emitter em;
    em_init(&em, split_cb, &s, NULL, 0, /*has_tools=*/false, TC_DIALECT_QWEN_XML);
    /* em_init starts in_think=1 (generation begins inside <think>). */
    ASSERT_TRUE(em.in_think == 1);

    /* Reasoning runs away — every token routes to the reasoning channel. */
    em_feed(&em, "let me think about ");
    em_feed(&em, "this problem carefully");
    ASSERT_TRUE(em.in_think == 1);
    ASSERT_STR_EQ(s.c, "");   /* nothing on the content channel yet */

    /* Budget tripped: sg_run_completion force-decodes "</think>" and feeds its
     * decoded text here.  The emitter's own THINK_CLOSE match flips to content. */
    em_feed(&em, "</think>");
    ASSERT_TRUE(em.in_think == 0);

    /* The answer now streams as CONTENT, not reasoning_content. */
    em_feed(&em, "The answer is 42.");
    em_flush(&em);
    ASSERT_STR_EQ(s.c, "The answer is 42.");
    ASSERT_STR_EQ(s.r, "let me think about this problem carefully");
    em_free(&em);
}

/* Qwen3.8 eval run 3 (2026-11-14): with the close one token (248069), the same text
 * spelled out of ordinary tokens (`</` `think` `>\`) is reasoning — it was routed to
 * content mid-thought, and the rest of the reasoning became the answer. */
static void test_spelled_think_close_stays_reasoning(void) {
    split_sink s = {0};
    emitter em;
    em_init(&em, split_cb, &s, NULL, 0, /*has_tools=*/true, TC_DIALECT_QWEN_XML);
    em_set_think_close_token(&em);

    em_feed(&em, "write \"\\n");
    em_feed(&em, "</");
    em_feed(&em, "think");
    em_feed(&em, ">\\");
    em_feed(&em, "n\"");
    ASSERT_TRUE(em.in_think == 1);
    ASSERT_STR_EQ(s.c, "");

    em_feed(&em, "</think>");
    ASSERT_TRUE(em.in_think == 0);
    em_feed(&em, "Done.");
    em_flush(&em);
    ASSERT_STR_EQ(s.r, "write \"\\n</think>\\n\"");
    ASSERT_STR_EQ(s.c, "Done.");
    em_free(&em);
}

/* ============================================================
 * Ornith JSON dialect (TC_DIALECT_ORNITH_JSON): a ```json fence / plausible
 * bare "{" diverts to the tool channel; XML openers are NOT watched.
 * ============================================================ */

/* A ```json fence after reasoning diverts the whole block to the tool channel. */
static void test_ornith_fence_captured(void) {
    sink s = {0};
    emitter em;
    em_init(&em, sink_cb, &s, NULL, 0, /*has_tools=*/true, TC_DIALECT_ORNITH_JSON);
    enter_content(&em);

    em_feed(&em, "Let me read that. ");
    em_feed(&em, "```json\n{\"function\":\"read\",\"arguments\":{\"path\":\"/x\"}}\n```");
    em_flush(&em);

    ASSERT_STR_EQ(s.buf, "Let me read that. ");
    ASSERT_NOT_NULL(em.tool.data);
    ASSERT_TRUE(strstr(em.tool.data, "```json") != NULL);
    ASSERT_TRUE(strstr(em.tool.data, "\"path\":\"/x\"") != NULL);
    em_free(&em);
}

/* A bare plausible "{" (no fence) diverts too. */
static void test_ornith_bare_brace_captured(void) {
    sink s = {0};
    emitter em;
    em_init(&em, sink_cb, &s, NULL, 0, /*has_tools=*/true, TC_DIALECT_ORNITH_JSON);
    enter_content(&em);

    em_feed(&em, "{\"function\":\"read\",\"arguments\":{\"path\":\"/y\"}}");
    em_flush(&em);

    ASSERT_STR_EQ(s.buf, "");   /* the whole object was diverted */
    ASSERT_NOT_NULL(em.tool.data);
    ASSERT_TRUE(strstr(em.tool.data, "\"path\":\"/y\"") != NULL);
    em_free(&em);
}

/* The fence split across two feeds still diverts (hold-back spans the boundary). */
static void test_ornith_fence_split(void) {
    sink s = {0};
    emitter em;
    em_init(&em, sink_cb, &s, NULL, 0, /*has_tools=*/true, TC_DIALECT_ORNITH_JSON);
    enter_content(&em);

    em_feed(&em, "ok ```js");
    em_feed(&em, "on\n{\"function\":\"read\",\"arguments\":{\"path\":\"/z\"}}\n```");
    em_flush(&em);

    ASSERT_STR_EQ(s.buf, "ok ");
    ASSERT_NOT_NULL(em.tool.data);
    ASSERT_TRUE(strstr(em.tool.data, "\"path\":\"/z\"") != NULL);
    em_free(&em);
}

/* A plausible "{" split right after the brace (key on the next feed) still
 * diverts — the undecided-brace hold-back covers it. */
static void test_ornith_brace_split(void) {
    sink s = {0};
    emitter em;
    em_init(&em, sink_cb, &s, NULL, 0, /*has_tools=*/true, TC_DIALECT_ORNITH_JSON);
    enter_content(&em);

    em_feed(&em, "answer: {");
    em_feed(&em, "\"function\":\"read\",\"arguments\":{\"path\":\"/w\"}}");
    em_flush(&em);

    ASSERT_STR_EQ(s.buf, "answer: ");
    ASSERT_NOT_NULL(em.tool.data);
    ASSERT_TRUE(strstr(em.tool.data, "\"path\":\"/w\"") != NULL);
    em_free(&em);
}

/* A stray prose "{" that does NOT open a function/name key stays content. */
static void test_ornith_prose_brace_stays_content(void) {
    sink s = {0};
    emitter em;
    em_init(&em, sink_cb, &s, NULL, 0, /*has_tools=*/true, TC_DIALECT_ORNITH_JSON);
    enter_content(&em);

    em_feed(&em, "The set is {1, 2, 3} of numbers.");
    em_flush(&em);

    ASSERT_STR_EQ(s.buf, "The set is {1, 2, 3} of numbers.");
    ASSERT_TRUE(em.tool.data == NULL || em.tool.len == 0);
    em_free(&em);
}

/* Tools OFF: even the ornith dialect never diverts (has_tools gate). */
static void test_ornith_tools_off_stays_content(void) {
    sink s = {0};
    emitter em;
    em_init(&em, sink_cb, &s, NULL, 0, /*has_tools=*/false, TC_DIALECT_ORNITH_JSON);
    enter_content(&em);

    em_feed(&em, "{\"function\":\"read\",\"arguments\":{\"path\":\"/x\"}}");
    em_flush(&em);

    ASSERT_STR_EQ(s.buf, "{\"function\":\"read\",\"arguments\":{\"path\":\"/x\"}}");
    ASSERT_TRUE(em.tool.data == NULL || em.tool.len == 0);
    em_free(&em);
}

/* Regression: under the ornith dialect an XML "<function=" opener is NOT
 * diverted (it is not ornith's shape) — it stays as content. */
static void test_ornith_ignores_xml_opener(void) {
    sink s = {0};
    emitter em;
    em_init(&em, sink_cb, &s, NULL, 0, /*has_tools=*/true, TC_DIALECT_ORNITH_JSON);
    enter_content(&em);

    em_feed(&em, "<function=read>\n<parameter=path>/x</parameter>\n</function>");
    em_flush(&em);

    ASSERT_TRUE(strstr(s.buf, "<function=read>") != NULL);
    ASSERT_TRUE(em.tool.data == NULL || em.tool.len == 0);
    em_free(&em);
}

/* ============================================================
 * AUTO reasoning-budget TRIP timing (regression for the saber runaway).
 *
 * sg_run_completion caps AUTO-path reasoning at reasoning_budget tokens: while
 * the emitter is inside <think>, every generated token counts, and once the
 * count reaches the budget the loop force-decodes "</think>" so the model
 * concludes.  The bug: the loop counted a token off the PRE-feed think state
 * (was_think) but decided whether to fire off the POST-feed em.in_think.  When
 * the model self-emits "</think>" on the very token that reaches the budget,
 * em_feed clears em.in_think in the same step, so the (em.in_think && rtok >=
 * budget) fire guard is already false — the trip is STARVED, the cap never
 * fires, and the model runs on to max_tokens (the ~8 tok/s never-concludes
 * runaway).  The fix latches the trip off was_think (as the count is) so the
 * boundary token can't starve it.
 *
 * This models the loop's counting + latch decision over the real emitter with a
 * budget of 3 and a token stream whose 3rd think token is exactly "</think>".
 * old_fire = the buggy post-feed guard; new_fire = the fix's pre-feed latch.
 * ============================================================ */
static void test_reason_budget_trips_on_closing_token(void) {
    split_sink s = {0};
    emitter em;
    em_init(&em, split_cb, &s, NULL, 0, /*has_tools=*/false, TC_DIALECT_QWEN_XML);
    ASSERT_TRUE(em.in_think == 1);

    /* Budget 3: the 3rd think token IS the close.  Feed one "token" per step. */
    const char *toks[] = { "reason A ", "reason B ", "</think>" };
    const int budget = 3;
    int rtok = 0;
    int old_fire = 0;   /* buggy: gate on POST-feed em.in_think */
    int new_fire = 0;   /* fixed: latch on PRE-feed was_think */

    for (int i = 0; i < 3; i++) {
        int was_think = em.in_think;
        em_feed(&em, toks[i]);
        if (was_think) {
            rtok++;
            /* Fix: latch the trip here, off was_think — survives a same-token
             * close that just cleared em.in_think. */
            if (rtok >= budget) new_fire = 1;
        }
        /* Old buggy guard, evaluated after em_feed. */
        if (em.in_think && rtok >= budget) old_fire = 1;
    }

    /* The regression: the OLD post-feed guard never armed (in_think went 0 on the
     * same token that reached the budget), so the reasoning cap silently failed. */
    ASSERT_INT_EQ(old_fire, 0);
    /* The fix: the pre-feed latch DID arm — the cap fires and the model concludes. */
    ASSERT_INT_EQ(new_fire, 1);
    /* And the budget really was reached exactly on the closing token. */
    ASSERT_INT_EQ(rtok, budget);
    em_free(&em);
}

/* Sanity: a genuine still-in-think overflow (no self-close) fires under BOTH the
 * old and new logic — the fix must not regress the plain runaway case. */
static void test_reason_budget_trips_still_in_think(void) {
    split_sink s = {0};
    emitter em;
    em_init(&em, split_cb, &s, NULL, 0, /*has_tools=*/false, TC_DIALECT_QWEN_XML);

    const char *toks[] = { "ramble 1 ", "ramble 2 ", "ramble 3 ", "ramble 4 " };
    const int budget = 3;
    int rtok = 0, old_fire = 0, new_fire = 0;
    for (int i = 0; i < 4; i++) {
        int was_think = em.in_think;
        em_feed(&em, toks[i]);
        if (was_think) { rtok++; if (rtok >= budget) new_fire = 1; }
        if (em.in_think && rtok >= budget) old_fire = 1;
    }
    /* Never closed <think>, so both strategies arm at the budget — parity. */
    ASSERT_INT_EQ(old_fire, 1);
    ASSERT_INT_EQ(new_fire, 1);
    ASSERT_TRUE(em.in_think == 1);
    em_free(&em);
}

/* ============================================================
 * A tool-call opener INSIDE <think> is never a tool call.
 * ============================================================ */

/* nous_fncall_prompt.py:136-147 splits on the LAST "</think>" and parses only
 * what follows — verbatim comment: "# Do not parse <tool_call> in thought!!!".
 * We get the same guarantee structurally rather than by a post-hoc split: the
 * emitter's in_think branch scans for "</think>" and NOTHING else, so opener
 * detection is not even armed while reasoning is open. The whole call text below
 * is reasoning, the tool channel stays empty, and the parser is never handed it.
 *
 * This is a STANDING guard, not a regression from a known bug: it is the only
 * thing keeping a model that talks about calling a tool inside its reasoning
 * from having that talk executed. */
static void test_tool_opener_inside_think_is_not_diverted(void) {
    sink s = {0};
    emitter em;
    em_init(&em, sink_cb, &s, NULL, 0, /*has_tools=*/true, TC_DIALECT_QWEN_XML);
    /* NOTE: no enter_content() — generation begins inside <think>. */

    em_feed(&em, "I could emit ");
    em_feed(&em, "<tool_call>\n<function=read>\n<parameter=path>\n/etc/hostname\n"
                 "</parameter>\n</function>\n</tool_call>");
    em_feed(&em, " but I will check first.");
    ASSERT_TRUE(em.in_think == 1);
    ASSERT_NULL(em.tool.data);          /* nothing was ever diverted */

    /* Only once reasoning actually closes does an opener arm. */
    em_feed(&em, "</think>");
    ASSERT_TRUE(em.in_think == 0);
    em_feed(&em, "<function=read>\n<parameter=path>\n/etc/hostname\n</parameter>\n</function>");
    em_flush(&em);
    ASSERT_NOT_NULL(em.tool.data);
    ASSERT_STR_EQ(s.buf, "");           /* the real call never reached content */
    em_free(&em);
}

int main(void) {
    printf("\n=== serve_emitter (stream tool-call detection) ===\n\n");
    RUN_TEST(tool_opener_inside_think_is_not_diverted);
    RUN_TEST(budget_force_close_flips_to_content);
    RUN_TEST(reason_budget_trips_on_closing_token);
    RUN_TEST(reason_budget_trips_still_in_think);
    RUN_TEST(bare_opener_captured);
    RUN_TEST(split_opener_captured);
    RUN_TEST(split_opener_with_prefix);
    RUN_TEST(tools_off_stays_content);
    RUN_TEST(tools_off_split_stays_content);
    RUN_TEST(wrapped_opener_captured);
    RUN_TEST(wrapped_before_bare);
    RUN_TEST(stray_fn_fragment_stays_content);
    RUN_TEST(real_opener_after_stray);
    RUN_TEST(no_opener_all_content);
    RUN_TEST(minicpm_opener_captured);
    RUN_TEST(minicpm_opener_split);
    RUN_TEST(spelled_think_close_stays_reasoning);
    RUN_TEST(minicpm_opener_tools_off_stays_content);
    RUN_TEST(forced_value_wrapper_captured);
    RUN_TEST(split_opener_after_equals);
    RUN_TEST(trailing_fn_open_never_followed_is_content);
    RUN_TEST(ornith_fence_captured);
    RUN_TEST(ornith_bare_brace_captured);
    RUN_TEST(ornith_fence_split);
    RUN_TEST(ornith_brace_split);
    RUN_TEST(ornith_prose_brace_stays_content);
    RUN_TEST(ornith_tools_off_stays_content);
    RUN_TEST(ornith_ignores_xml_opener);
    TEST_REPORT();
    return __ft_failed == 0 ? 0 : 1;
}
