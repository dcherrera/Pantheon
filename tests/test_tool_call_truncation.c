/* SPDX-License-Identifier: AGPL-3.0-or-later */
/* Copyright (C) 2026 David (Corey) Herrera */
/*
 * TRUNCATED TOOL CALLS — the grammar and the parser must agree about where a
 * parameter value ended, and a call the grammar never closed must be visibly
 * incomplete rather than a call with an argument quietly missing.
 *
 * This is the only test that links BOTH src/serve/tool_grammar.c and
 * src/serve/tool_calling.c, because the defect it covers exists only in the gap
 * between them.  Neither half is wrong on its own:
 *
 *   • the grammar's value terminator was "\n</parameter>\n", plus a guard that
 *     refused to let it complete on an empty value by consuming its chars as
 *     CONTENT instead;
 *   • the parser's value terminator is strstr(vstart, "</parameter>") — any
 *     occurrence, no padding required.
 *
 * On the one emission the model actually produces when it gives up — a value of
 * a single "\n" — those two rules disagree about a byte sequence both of them
 * saw.  The guard ate the terminator's leading "\n" as content, which made
 * "</parameter>" break the match and be swallowed as ordinary value text, so the
 * walk stayed inside TG_VALUE for the rest of the turn.  The parser found that
 * same "</parameter>" immediately, read the value as "", and dropped the key.
 * The client got `read {}` / `write {"path":…}` with no `content` — a
 * complete-looking call missing a required argument — while the serve's own
 * grammar state said the call had never finished.
 *
 * Measured: 9/9 live truncated calls were "grammar ACCEPTS char-for-char,
 * tg_complete == NO", both controls complete.
 *
 * Every defect case here is paired with the healthy neighbour it must not be
 * confused with, and the sharpest of those neighbours is a LEGAL EMPTY VALUE:
 * `edit`'s new_string:"" is a deletion (the client's schema code
 * argument_is_absent), and it is byte-for-byte identical to the truncation up to
 * the point where the model closes the call.  Only the close separates them,
 * which is exactly why the verdict is tg_complete() and not "was the value
 * empty".
 *
 * Pure C string/FSM work — no engine/model/CUDA deps, fully local.
 */

#include "lib/test.h"
#include "../src/serve/tool_calling.h"
#include "../src/serve/tool_grammar.h"
#include "../src/serve/json_reader.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

/* The client's real shapes, trimmed to the tools that appear in the captured raws:
 * `read` (1 required arg — the `{}` case), `write` (2 — the "path kept, content
 * lost" case) and `edit` (3 — where new_string:"" is a legal deletion). */
static const char *TRUNC_TOOLS =
    "[{\"type\":\"function\",\"function\":{\"name\":\"read\","
    "\"parameters\":{\"type\":\"object\",\"properties\":{"
    "\"path\":{\"type\":\"string\"}},\"required\":[\"path\"]}}},"
    "{\"type\":\"function\",\"function\":{\"name\":\"write\","
    "\"parameters\":{\"type\":\"object\",\"properties\":{"
    "\"path\":{\"type\":\"string\"},\"content\":{\"type\":\"string\"}},"
    "\"required\":[\"path\",\"content\"]}}},"
    "{\"type\":\"function\",\"function\":{\"name\":\"edit\","
    "\"parameters\":{\"type\":\"object\",\"properties\":{"
    "\"path\":{\"type\":\"string\"},\"old_string\":{\"type\":\"string\"},"
    "\"new_string\":{\"type\":\"string\"}},"
    "\"required\":[\"path\",\"old_string\",\"new_string\"]}}}]";

/* What the DECODER saw, for one raw emission. */
typedef struct {
    bool accepted;   /* every char was an allowed continuation (nothing masked) */
    bool complete;   /* tg_complete — the call closed */
    bool in_call;    /* tg_in_call — the walk is inside the skeleton */
} decode_verdict;

/* Walk `text` through the grammar one char at a time in FORCED mode
 * (tool_choice=required, which is how the orchestrator runs), asking
 * tg_token_allowed BEFORE each advance so `accepted` really means "the mask
 * would never have blocked this".  `accepted` is the load-bearing half: it is
 * what makes a failing case a GRAMMAR/PARSER DISAGREEMENT rather than merely
 * text the grammar rejected. */
static decode_verdict decode(const tg_grammar *g, const char *text)
{
    decode_verdict v = { true, false, false };
    tg_state *st = tg_state_new_forced(g, NULL);
    if (!st) { v.accepted = false; return v; }
    tg_reasoning_end(st);
    for (const char *p = text; *p; p++) {
        char one[2] = { *p, '\0' };
        if (!tg_token_allowed(st, one)) v.accepted = false;
        tg_advance(st, one);
    }
    v.complete = tg_complete(st);
    v.in_call  = tg_in_call(st);
    tg_state_free(st);
    return v;
}

/* Parse `text` and return the FIRST call's arguments JSON (owned by the caller),
 * or NULL when no call came out.  `*n_out` reports how many calls were parsed so
 * a test can tell "no call" from "a call with nothing in it". */
static char *parse_first_arguments(const char *text, const jr_value *tools,
                                   int *n_out, char **name_out)
{
    tc_tool_call *calls = NULL;
    int n = 0;
    char *args = NULL;
    if (tc_parse_tool_calls(text, tools, TC_DIALECT_QWEN_XML, &calls, &n) != 0)
        n = 0;
    *n_out = n;
    if (name_out) *name_out = NULL;
    if (n > 0) {
        args = calls[0].arguments ? strdup(calls[0].arguments) : NULL;
        if (name_out && calls[0].name) *name_out = strdup(calls[0].name);
    }
    tc_tool_calls_free(calls, n);
    return args;
}

/* ============================================================
 * THE TWO SHAPES THAT REACHED THE CLIENT AS BROKEN CALLS
 * ============================================================ */

/* VERBATIM from the live serve log (tool_arg_truncation.md, log:274) — the raw
 * that reached the client as `read {}`.  Note the close carries NO padding newlines:
 * "</parameter></function>", which the old grammar could not match either.
 *
 * The assertions in order are the whole finding: the grammar never objected to a
 * single character (so no amount of masking was going to stop this), the call
 * nonetheless never closed, and the parser turned it into a named call whose one
 * required argument is gone. */
static void test_live_read_empty_path(void) {
    jr_value *tools = jr_parse(TRUNC_TOOLS);
    tg_grammar *g = tg_build(tools);
    const char *raw =
        "<tool_call>\n<function=read>\n<parameter=path>\n\n"
        "</parameter></function>\n";

    decode_verdict v = decode(g, raw);
    ASSERT_TRUE(v.accepted);      /* the grammar accepted it char-for-char */
    ASSERT_FALSE(v.complete);     /* …and it is NOT a finished call */
    ASSERT_TRUE(v.in_call);       /* …and the walk is inside the skeleton */

    /* What the parser makes of the same bytes: a `read` whose `path` is the
     * empty string the model actually closed around.  It used to be `{}` — the
     * argument dropped and therefore ABSENT, which is a claim about a member
     * that was in fact present.  Either way shipping it is the defect, and the
     * three assertions above are what the serve refuses on (openai_chat.c
     * collect_tool_calls, asserted on the wire in test_chat_request.c).  The
     * empty `path` that survives that gate being off is the empty-path wedge's shape,
     * and it is the client's void-argument guard that answers it. */
    int n = 0; char *name = NULL;
    char *args = parse_first_arguments(raw, tools, &n, &name);
    ASSERT_INT_EQ(n, 1);
    ASSERT_STR_EQ(name, "read");
    ASSERT_STR_EQ(args, "{\"path\":\"\"}");
    free(args); free(name);

    tg_free(g);
    jr_free(tools);
}

/* The 2-required-argument shape: `path` survives, `content` arrives EMPTY, and
 * the model stops without closing the call.
 *
 * This is the residue the parameter-level empty-value rule cannot see.  `content`
 * was closed — `</parameter>` is right there — so as bytes it is a deliberate
 * empty value and the parser now keeps it.  What is truncated is the CALL, which
 * only the grammar can say (`v.complete` false, asserted below), and that is the
 * gate this shape is caught by. */
static void test_live_write_loses_content(void) {
    jr_value *tools = jr_parse(TRUNC_TOOLS);
    tg_grammar *g = tg_build(tools);
    const char *raw =
        "<function=write>\n"
        "<parameter=path>\n/tmp/x.c\n</parameter>\n"
        "<parameter=content>\n\n</parameter>\n";

    decode_verdict v = decode(g, raw);
    ASSERT_TRUE(v.accepted);
    ASSERT_FALSE(v.complete);
    ASSERT_TRUE(v.in_call);

    int n = 0; char *name = NULL;
    char *args = parse_first_arguments(raw, tools, &n, &name);
    ASSERT_INT_EQ(n, 1);
    ASSERT_STR_EQ(name, "write");
    /* `content` is EMPTY, not gone — and the difference matters, because gone is
     * a claim the bytes do not support. */
    ASSERT_STR_EQ(args, "{\"path\":\"/tmp/x.c\",\"content\":\"\"}");
    free(args); free(name);

    tg_free(g);
    jr_free(tools);
}

/* The end token landing inside a value (three captured raws end this way).  The
 * value here is NOT empty, so no empty-value rule of any kind would catch it —
 * only "the call never closed" does. */
static void test_end_token_inside_a_value(void) {
    jr_value *tools = jr_parse(TRUNC_TOOLS);
    tg_grammar *g = tg_build(tools);
    const char *raw =
        "<function=write>\n"
        "<parameter=path>\n/tmp/x.c\n</parameter>\n"
        "<parameter=content>\n<|im_end|>";

    decode_verdict v = decode(g, raw);
    ASSERT_TRUE(v.accepted);
    ASSERT_FALSE(v.complete);
    ASSERT_TRUE(v.in_call);

    tg_free(g);
    jr_free(tools);
}

/* ============================================================
 * THE NEIGHBOURS THAT MUST NOT MOVE
 * ============================================================ */

/* CONTROL — a healthy multi-parameter call is complete and keeps both arguments.
 * Without this the suite would "pass" under a grammar that never completes
 * anything, which is precisely the failure mode being fixed. */
static void test_healthy_multi_parameter_call(void) {
    jr_value *tools = jr_parse(TRUNC_TOOLS);
    tg_grammar *g = tg_build(tools);
    const char *raw =
        "<tool_call>\n<function=write>\n"
        "<parameter=path>\n/tmp/x.c\n</parameter>\n"
        "<parameter=content>\nhello\n</parameter>\n"
        "</function>\n</tool_call>";

    decode_verdict v = decode(g, raw);
    ASSERT_TRUE(v.accepted);
    ASSERT_TRUE(v.complete);

    int n = 0; char *name = NULL;
    char *args = parse_first_arguments(raw, tools, &n, &name);
    ASSERT_INT_EQ(n, 1);
    ASSERT_STR_EQ(name, "write");
    ASSERT_STR_EQ(args, "{\"path\":\"/tmp/x.c\",\"content\":\"hello\"}");
    free(args); free(name);

    tg_free(g);
    jr_free(tools);
}

/* THE TRAP — a LEGAL EMPTY VALUE.  `edit` with new_string:"" is a deletion, and
 * The client's schema layer is explicit that an empty member is present, not absent
 * (argument_is_absent, the client's schema code).  Its bytes are identical
 * to test_live_read_empty_path's right up to the close, so any rule of the form
 * "an empty value means the model gave up" turns this correct call into an error.
 *
 * It must be COMPLETE: the model closed the parameter and then closed the call.
 * That is the whole difference, and it is the only one available. */
static void test_legal_empty_value_completes(void) {
    jr_value *tools = jr_parse(TRUNC_TOOLS);
    tg_grammar *g = tg_build(tools);
    const char *raw =
        "<function=edit>\n"
        "<parameter=path>\n/tmp/x.c\n</parameter>\n"
        "<parameter=old_string>\ndead line\n</parameter>\n"
        "<parameter=new_string>\n\n</parameter>\n"
        "</function>";

    decode_verdict v = decode(g, raw);
    ASSERT_TRUE(v.accepted);
    ASSERT_TRUE(v.complete);      /* NOT a truncation → never gated */

    /* And the deletion SURVIVES: `new_string` reaches the client present and
     * empty, which is the one shape the client can read as a deletion.  Dropped, it
     * arrived absent — indistinguishable from a call cut short before
     * `new_string`, charged against the truncated-call budget, corrected, and
     * re-asked until the task was abandoned over a call that was right every
     * time (the client's schema code argument_is_absent, reducer.c
     * LOOP_GUARD_TRUNCATED_CALL_BUDGET). */
    int n = 0; char *name = NULL;
    char *args = parse_first_arguments(raw, tools, &n, &name);
    ASSERT_INT_EQ(n, 1);
    ASSERT_STR_EQ(name, "edit");
    ASSERT_STR_EQ(args, "{\"path\":\"/tmp/x.c\",\"old_string\":\"dead line\","
                        "\"new_string\":\"\"}");
    free(args); free(name);

    tg_free(g);
    jr_free(tools);
}

/* The naive early close stays REJECTED — the model cannot skip a required
 * argument by closing the call.  This is the guard that already worked
 * (tool_grammar.c TG_BODY, commit 5baeeec) and the one thing in this area the
 * measurement confirmed was doing its job; the value-side change must not have
 * loosened it. */
static void test_naive_early_close_still_rejected(void) {
    jr_value *tools = jr_parse(TRUNC_TOOLS);
    tg_grammar *g = tg_build(tools);

    /* No parameters at all. */
    decode_verdict bare = decode(g, "<function=read>\n</function>");
    ASSERT_FALSE(bare.accepted);
    ASSERT_FALSE(bare.complete);

    /* One of two required arguments, then close. */
    decode_verdict half = decode(g,
        "<function=write>\n<parameter=path>\n/tmp/x.c\n</parameter>\n</function>");
    ASSERT_FALSE(half.accepted);
    ASSERT_FALSE(half.complete);

    tg_free(g);
    jr_free(tools);
}

/* ============================================================
 * THE MECHANISM ITSELF — the close is no longer swallowed
 * ============================================================ */

/* The direct statement of the fix, with no parser involved.
 *
 * Feed a value and its close, and then ask what may follow.  Once "</parameter>"
 * has been consumed the walk is at a BODY BOUNDARY, where the only legal
 * continuations are another "<parameter=" or "</function>" — so ordinary prose
 * must be MASKED.  Under the old guard the close had been absorbed as value text
 * and the walk was still in free-text VALUE, where prose is allowed; that single
 * difference is the whole divergence.
 *
 * Both the padded and unpadded closes are checked, because the live raws use the
 * unpadded one. */
static void test_close_returns_the_walk_to_the_skeleton(void) {
    jr_value *tools = jr_parse(TRUNC_TOOLS);
    tg_grammar *g = tg_build(tools);

    static const char *const openers[] = {
        "<function=read>\n<parameter=path>\n/tmp/x.c\n</parameter>\n",  /* padded   */
        "<function=read>\n<parameter=path>\n/tmp/x.c</parameter>",      /* unpadded */
        "<function=read>\n<parameter=path>\n\n</parameter>",            /* empty    */
    };
    for (size_t i = 0; i < sizeof(openers) / sizeof(openers[0]); i++) {
        tg_state *st = tg_state_new_forced(g, NULL);
        ASSERT_NOT_NULL(st);
        tg_reasoning_end(st);
        tg_advance(st, openers[i]);
        ASSERT_TRUE(tg_in_call(st));
        /* At a body boundary: prose is off-grammar… */
        ASSERT_FALSE(tg_token_allowed(st, "The bash tool seems to be stale"));
        /* …and the two real continuations are not. */
        ASSERT_TRUE(tg_token_allowed(st, "</function>"));
        ASSERT_TRUE(tg_token_allowed(st, "\n<parameter=path>"));
        ASSERT_FALSE(tg_complete(st));   /* the CALL is still open */
        tg_state_free(st);
    }

    tg_free(g);
    jr_free(tools);
}

/* The value boundary the grammar uses is the one the parser uses, across the
 * padding variants the model actually emits.  Each of these closes the call, and
 * the parser reads the same value out of each — which is the property that makes
 * "the grammar did not complete" mean "the model truncated" and nothing else. */
static void test_value_boundary_agrees_with_the_parser(void) {
    jr_value *tools = jr_parse(TRUNC_TOOLS);
    tg_grammar *g = tg_build(tools);

    static const char *const shapes[] = {
        "<function=read>\n<parameter=path>\n/tmp/x.c\n</parameter>\n</function>",
        "<function=read>\n<parameter=path>\n/tmp/x.c</parameter></function>",
        "<function=read>\n<parameter=path>\n/tmp/x.c\n</parameter></function>",
        "<function=read>\n<parameter=path>\n/tmp/x.c</parameter>\n</function>",
    };
    for (size_t i = 0; i < sizeof(shapes) / sizeof(shapes[0]); i++) {
        decode_verdict v = decode(g, shapes[i]);
        ASSERT_TRUE(v.accepted);
        ASSERT_TRUE(v.complete);

        int n = 0; char *name = NULL;
        char *args = parse_first_arguments(shapes[i], tools, &n, &name);
        ASSERT_INT_EQ(n, 1);
        ASSERT_STR_EQ(args, "{\"path\":\"/tmp/x.c\"}");
        free(args); free(name);
    }

    tg_free(g);
    jr_free(tools);
}

/* ============================================================
 * Main
 * ============================================================ */

int main(void) {
    printf("\n  truncated tool calls — grammar/parser agreement + the close gate\n\n");

    RUN_TEST(live_read_empty_path);
    RUN_TEST(live_write_loses_content);
    RUN_TEST(end_token_inside_a_value);

    RUN_TEST(healthy_multi_parameter_call);
    RUN_TEST(legal_empty_value_completes);
    RUN_TEST(naive_early_close_still_rejected);

    RUN_TEST(close_returns_the_walk_to_the_skeleton);
    RUN_TEST(value_boundary_agrees_with_the_parser);

    TEST_REPORT();
    return __ft_failed > 0 ? 1 : 0;
}
