/* SPDX-License-Identifier: AGPL-3.0-or-later */
/* Copyright (C) 2026 David (Corey) Herrera */
/*
 * Unit tests for the tool-call decoding grammar (src/serve/tool_grammar.c).
 *
 * The grammar is the Phase-5 correctness gate: with a tools schema it must
 * ACCEPT the model's trained XML emission (wrapped <tool_call>… and the bare
 * <function=… form) and REJECT the off-contract drift (the JSON {"tool_call":…}
 * habit, unknown function names, unknown parameter keys, a value outside a
 * parameter's schema `enum`), while leaving free prose UNCONSTRAINED (the model
 * may legitimately answer with no tool call).
 *
 * Pure C, libc-only (json_reader for the tools DOM) — fully local, no CUDA.
 */

#include "lib/test.h"
#include "../src/serve/tool_grammar.h"
#include "../src/serve/json_reader.h"
#include "../src/serve/tool_calling.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

/* Two tools: glob(pattern, path) and read(path).  No `required` list on either,
 * so these stay the fixture for everything that is not about requiredness.
 *
 * Deliberately NO "minLength" on any parameter: this is the schema shape that
 * says nothing about emptiness, and the grammar must not invent a floor it was
 * never given.  CLIENT_FLOORED_JSON below is the same tools as the client really sends
 * them, and is what the empty-value tests use. */
static const char *TOOLS_JSON =
    "[{\"type\":\"function\",\"function\":{\"name\":\"glob\","
    "\"parameters\":{\"type\":\"object\",\"properties\":{"
    "\"pattern\":{\"type\":\"string\"},\"path\":{\"type\":\"string\"}}}}},"
    "{\"type\":\"function\",\"function\":{\"name\":\"read\","
    "\"parameters\":{\"type\":\"object\",\"properties\":{"
    "\"path\":{\"type\":\"string\"}}}}}]";

/* The client's REAL shapes for the two tools this contract turns on
 * (the client's schema code), verbatim in the one keyword
 * that matters:
 *
 *   bash.command      required, "minLength":1  → empty is the live failure
 *   edit.new_string   required, NO minLength   → empty is the DELETION
 *
 * Both keys are `required`.  That is the whole reason requiredness cannot be the
 * empty-value predicate and `minLength` is. */
static const char *CLIENT_FLOORED_JSON =
    "[{\"type\":\"function\",\"function\":{\"name\":\"bash\","
    "\"parameters\":{\"type\":\"object\",\"properties\":{"
    "\"command\":{\"type\":\"string\",\"minLength\":1}},"
    "\"required\":[\"command\"]}}},"
    "{\"type\":\"function\",\"function\":{\"name\":\"edit\","
    "\"parameters\":{\"type\":\"object\",\"properties\":{"
    "\"path\":{\"type\":\"string\",\"minLength\":1},"
    "\"old_string\":{\"type\":\"string\",\"minLength\":1},"
    "\"new_string\":{\"type\":\"string\"}},"
    "\"required\":[\"path\",\"old_string\",\"new_string\"]}}}]";

/* The client's REAL switch_mode schema (the client's schema code) — the
 * live victim of the dropped-`enum` gap: before the enum was compiled, a VALUE
 * of "banana" was a complete, well-formed switch_mode call. */
static const char *SWITCH_MODE_JSON =
    "[{\"type\":\"function\",\"function\":{\"name\":\"switch_mode\","
    "\"parameters\":{\"type\":\"object\",\"properties\":"
    "{\"mode\":{\"type\":\"string\","
    "\"enum\":[\"chat\",\"plan\",\"act\",\"orchestrate\"]},"
    "\"reason\":{\"type\":\"string\"}},"
    "\"required\":[\"mode\",\"reason\"]}}}]";

/* log(msg REQUIRED, level enum OPTIONAL) — an enum on a key the model may
 * legally leave out entirely. */
static const char *ENUM_OPTIONAL_JSON =
    "[{\"type\":\"function\",\"function\":{\"name\":\"log\","
    "\"parameters\":{\"type\":\"object\",\"properties\":{"
    "\"msg\":{\"type\":\"string\"},"
    "\"level\":{\"type\":\"string\",\"enum\":[\"info\",\"warn\",\"error\"]}},"
    "\"required\":[\"msg\"]}}}]";

/* Enum shapes that must DEGRADE to a free-text VALUE rather than reject:
 * non-string members, one non-string among strings, an empty member, an empty
 * list, and string members contradicting a non-string `type`.  `notype` has
 * string members and no declared type — that one DOES compile. */
static const char *ENUM_DEGRADE_JSON =
    "[{\"type\":\"function\",\"function\":{\"name\":\"cfg\","
    "\"parameters\":{\"type\":\"object\",\"properties\":{"
    "\"nums\":{\"type\":\"integer\",\"enum\":[1,2,3]},"
    "\"mixed\":{\"type\":\"string\",\"enum\":[\"a\",7]},"
    "\"blank\":{\"type\":\"string\",\"enum\":[\"a\",\"\"]},"
    "\"nolist\":{\"type\":\"string\",\"enum\":[]},"
    "\"clash\":{\"type\":\"integer\",\"enum\":[\"a\",\"b\"]},"
    "\"notype\":{\"enum\":[\"yes\",\"no\"]}}}}}]";

/* Feed a whole string through the grammar one CHAR at a time, checking that
 * every char stays allowed (mirrors the per-token mask at char granularity —
 * the strictest possible feed).  Returns the final state (caller frees). */
static tg_state *feed_ok(tg_grammar *g, const char *s, bool *all_allowed) {
    tg_state *st = tg_state_new(g);
    char one[2] = {0, 0};
    *all_allowed = true;
    for (const char *p = s; *p; p++) {
        one[0] = *p;
        if (!tg_token_allowed(st, one)) { *all_allowed = false; break; }
        tg_advance(st, one);
    }
    return st;
}

/* Does feeding `s` (char by char) hit a DISALLOWED char at some point? */
static bool feed_rejects(tg_grammar *g, const char *s) {
    bool ok;
    tg_state *st = feed_ok(g, s, &ok);
    tg_state_free(st);
    return !ok;
}

/* Forced-mode variants.  A forced state (required-any or pinned function) with
 * reasoning already ended — the opener is compelled from the first char. */
static tg_state *feed_forced(tg_state *st, const char *s, bool *all_allowed) {
    char one[2] = {0, 0};
    *all_allowed = true;
    for (const char *p = s; *p; p++) {
        one[0] = *p;
        if (!tg_token_allowed(st, one)) { *all_allowed = false; break; }
        tg_advance(st, one);
    }
    return st;
}

/* Feed `s` into a forced (required-any) state whose reasoning has ended; does it
 * get rejected somewhere? */
static bool forced_rejects(tg_grammar *g, const char *s) {
    tg_state *st = tg_state_new_forced(g, NULL);
    tg_reasoning_end(st);
    bool ok;
    feed_forced(st, s, &ok);
    tg_state_free(st);
    return !ok;
}

/* ============================================================
 * ACCEPT: canonical wrapped call.
 * ============================================================ */
static void test_wrapped_accepts(void) {
    jr_value *tools = jr_parse(TOOLS_JSON);
    tg_grammar *g = tg_build(tools);
    ASSERT_NOT_NULL(g);

    const char *call =
        "<tool_call>\n<function=glob>\n"
        "<parameter=pattern>\n*.md\n</parameter>\n"
        "</function>\n</tool_call>";
    bool ok;
    tg_state *st = feed_ok(g, call, &ok);
    ASSERT_TRUE(ok);
    ASSERT_TRUE(tg_complete(st));

    tg_state_free(st);
    tg_free(g);
    jr_free(tools);
}

/* ============================================================
 * ACCEPT: bare (no wrapper) call — the Q3 model's favored form.
 * ============================================================ */
static void test_bare_accepts(void) {
    jr_value *tools = jr_parse(TOOLS_JSON);
    tg_grammar *g = tg_build(tools);

    const char *call =
        "<function=read>\n<parameter=path>\nREADME.md\n</parameter>\n</function>";
    bool ok;
    tg_state *st = feed_ok(g, call, &ok);
    ASSERT_TRUE(ok);
    ASSERT_TRUE(tg_complete(st));

    tg_state_free(st);
    tg_free(g);
    jr_free(tools);
}

/* ============================================================
 * ACCEPT: prose before the call (reasoning region) is unconstrained.
 * ============================================================ */
static void test_prose_then_call(void) {
    jr_value *tools = jr_parse(TOOLS_JSON);
    tg_grammar *g = tg_build(tools);

    const char *txt =
        "Let me look. I'll list the files < first > maybe. "
        "<function=glob>\n<parameter=path>\nsrc\n</parameter>\n</function>";
    bool ok;
    tg_state *st = feed_ok(g, txt, &ok);
    ASSERT_TRUE(ok);            /* prose stayed FREE, call completed */
    ASSERT_TRUE(tg_complete(st));

    tg_state_free(st);
    tg_free(g);
    jr_free(tools);
}

/* ============================================================
 * ACCEPT: pure prose answer, NO tool call → never forced into a call.
 * ============================================================ */
static void test_pure_prose_no_call(void) {
    jr_value *tools = jr_parse(TOOLS_JSON);
    tg_grammar *g = tg_build(tools);

    const char *txt = "The answer is 42. No tools needed here.";
    bool ok;
    tg_state *st = feed_ok(g, txt, &ok);
    ASSERT_TRUE(ok);                 /* every char allowed */
    ASSERT_FALSE(tg_in_call(st));    /* still FREE, not inside a call */
    ASSERT_FALSE(tg_complete(st));

    tg_state_free(st);
    tg_free(g);
    jr_free(tools);
}

/* ============================================================
 * REJECT: the off-contract JSON habit {"tool_call":…}.
 * Once the model opens "<function=" (or "<tool_call>") the skeleton engages;
 * feeding the JSON form as if it were a call must be rejected.  We test that
 * the JSON, fed after an opener, breaks — i.e. the grammar won't emit it.
 * ============================================================ */
static void test_json_form_rejected(void) {
    jr_value *tools = jr_parse(TOOLS_JSON);
    tg_grammar *g = tg_build(tools);

    /* After the opener the grammar demands a NAME then '>'; JSON braces/quotes
     * are not part of any tool name, so this is rejected. */
    ASSERT_TRUE(feed_rejects(g, "<function={\"name\":\"glob\"}"));
    /* The bare JSON tool_call form: "<tool_call>{...}" — after "<tool_call>\n"
     * the grammar demands "<function=", not '{'. */
    ASSERT_TRUE(feed_rejects(g, "<tool_call>\n{\"tool_call\":"));

    tg_free(g);
    jr_free(tools);
}

/* ============================================================
 * REJECT: unknown function name.
 * ============================================================ */
static void test_unknown_function_rejected(void) {
    jr_value *tools = jr_parse(TOOLS_JSON);
    tg_grammar *g = tg_build(tools);

    /* "grep" is not a tool → the 'g' is live (glob starts with g) but the name
     * cannot terminate as "grep", and 'r' after "g" is not live (glob → 'l'). */
    ASSERT_TRUE(feed_rejects(g, "<function=grep>"));
    /* a name that is a valid prefix but closed early ("glo>") is not exact. */
    ASSERT_TRUE(feed_rejects(g, "<function=glo>"));

    tg_free(g);
    jr_free(tools);
}

/* ============================================================
 * REJECT: unknown parameter key for a valid function.
 * ============================================================ */
static void test_unknown_param_rejected(void) {
    jr_value *tools = jr_parse(TOOLS_JSON);
    tg_grammar *g = tg_build(tools);

    /* read has only "path"; "mode" is not a key. */
    ASSERT_TRUE(feed_rejects(g,
        "<function=read>\n<parameter=mode>"));
    /* glob's key "pattern" mistyped as "patern" is rejected. */
    ASSERT_TRUE(feed_rejects(g,
        "<function=glob>\n<parameter=patern>"));

    tg_free(g);
    jr_free(tools);
}

/* ============================================================
 * ACCEPT: a name/key split across "tokens" still matches (cross-boundary).
 * ============================================================ */
static void test_cross_token_name(void) {
    jr_value *tools = jr_parse(TOOLS_JSON);
    tg_grammar *g = tg_build(tools);

    tg_state *st = tg_state_new(g);
    /* opener + partial name in one token, rest of the name in the next. */
    ASSERT_TRUE(tg_token_allowed(st, "<function=gl"));  tg_advance(st, "<function=gl");
    ASSERT_TRUE(tg_token_allowed(st, "ob>\n"));         tg_advance(st, "ob>\n");
    ASSERT_TRUE(tg_token_allowed(st, "<parameter=pattern>\n")); tg_advance(st, "<parameter=pattern>\n");
    ASSERT_TRUE(tg_token_allowed(st, "x\n</parameter>\n"));     tg_advance(st, "x\n</parameter>\n");
    ASSERT_TRUE(tg_token_allowed(st, "</function>"));   tg_advance(st, "</function>");
    ASSERT_TRUE(tg_complete(st));
    /* an unknown-name split must still be rejected at the divergent char. */
    tg_state *bad = tg_state_new(g);
    tg_advance(bad, "<function=gl");
    ASSERT_FALSE(tg_token_allowed(bad, "ee>"));  /* "glee" diverges at 2nd 'e' */

    tg_state_free(st);
    tg_state_free(bad);
    tg_free(g);
    jr_free(tools);
}

/* ============================================================
 * ACCEPT: multi-parameter call (two <parameter=…> blocks).
 * ============================================================ */
static void test_multi_param(void) {
    jr_value *tools = jr_parse(TOOLS_JSON);
    tg_grammar *g = tg_build(tools);

    const char *call =
        "<function=glob>\n"
        "<parameter=pattern>\n*.c\n</parameter>\n"
        "<parameter=path>\nsrc\n</parameter>\n"
        "</function>";
    bool ok;
    tg_state *st = feed_ok(g, call, &ok);
    ASSERT_TRUE(ok);
    ASSERT_TRUE(tg_complete(st));

    tg_state_free(st);
    tg_free(g);
    jr_free(tools);
}

/* ============================================================
 * FORCED mode: after reasoning, prose/JSON are REJECTED from the start —
 * the opener is compelled (this is the Phase-5 gap being closed).
 * ============================================================ */
static void test_forced_rejects_prose_and_json(void) {
    jr_value *tools = jr_parse(TOOLS_JSON);
    tg_grammar *g = tg_build(tools);

    /* Plain prose refusal — the classic Q3 failure — must be masked out. */
    ASSERT_TRUE(forced_rejects(g, "I cannot do that."));
    /* The off-contract JSON drift, from the very first char. */
    ASSERT_TRUE(forced_rejects(g, "{\"tool_call\":{\"name\":\"glob\"}}"));
    /* Even a stray non-'<' leading char is rejected (opener required). */
    ASSERT_TRUE(forced_rejects(g, "Sure! <function=glob>"));

    tg_free(g);
    jr_free(tools);
}

/* FORCED mode ACCEPTS a valid call fed from the start (no reasoning). */
static void test_forced_accepts_call(void) {
    jr_value *tools = jr_parse(TOOLS_JSON);
    tg_grammar *g = tg_build(tools);

    tg_state *st = tg_state_new_forced(g, NULL);
    tg_reasoning_end(st);
    const char *call =
        "<function=read>\n<parameter=path>\nREADME.md\n</parameter>\n</function>";
    bool ok;
    feed_forced(st, call, &ok);
    ASSERT_TRUE(ok);
    ASSERT_TRUE(tg_complete(st));

    tg_state_free(st);
    tg_free(g);
    jr_free(tools);
}

/* FORCED mode: reasoning is allowed FIRST (before tg_reasoning_end); then the
 * opener is compelled.  Mirrors the serve loop: <think> content is unconstrained
 * until "</think>" flips the phase, after which only the call is valid. */
static void test_forced_reasoning_then_call(void) {
    jr_value *tools = jr_parse(TOOLS_JSON);
    tg_grammar *g = tg_build(tools);

    tg_state *st = tg_state_new_forced(g, NULL);
    bool ok;
    /* Reasoning phase: arbitrary prose is allowed (mask inactive here). */
    feed_forced(st, "The user wants a file listing. Let me use glob.", &ok);
    ASSERT_TRUE(ok);
    ASSERT_FALSE(tg_mask_active(st));   /* not yet forced */
    /* "</think>" fires reasoning-end → opener now required. */
    tg_reasoning_end(st);
    ASSERT_TRUE(tg_mask_active(st));    /* forced: post-reasoning FREE is masked */
    /* Now a valid call is accepted; prose would be rejected. */
    ASSERT_TRUE(tg_token_allowed(st, "<function=glob>"));
    ASSERT_FALSE(tg_token_allowed(st, "Actually, no."));
    feed_forced(st, "<function=glob>\n<parameter=pattern>\n*.md\n</parameter>\n</function>", &ok);
    ASSERT_TRUE(ok);
    ASSERT_TRUE(tg_complete(st));
    ASSERT_FALSE(tg_mask_active(st));   /* complete → constraint released */

    tg_state_free(st);
    tg_free(g);
    jr_free(tools);
}

/* FORCED-FUNCTION mode pins the NAME: the target function is accepted; any other
 * tool name is rejected even though it is a valid tool. */
static void test_forced_function_pins_name(void) {
    jr_value *tools = jr_parse(TOOLS_JSON);
    tg_grammar *g = tg_build(tools);

    /* Pin "read": a "read" call is accepted, a "glob" call is rejected. */
    tg_state *good = tg_state_new_forced(g, "read");
    ASSERT_NOT_NULL(good);
    tg_reasoning_end(good);
    bool ok;
    feed_forced(good, "<function=read>\n<parameter=path>\nx\n</parameter>\n</function>", &ok);
    ASSERT_TRUE(ok);
    ASSERT_TRUE(tg_complete(good));
    tg_state_free(good);

    tg_state *bad = tg_state_new_forced(g, "read");
    tg_reasoning_end(bad);
    feed_forced(bad, "<function=glob>", &ok);
    ASSERT_FALSE(ok);   /* "glob" rejected: name pinned to "read" */
    tg_state_free(bad);

    /* A pin that is not a known tool → NULL (serve falls back to AUTO). */
    ASSERT_NULL(tg_state_new_forced(g, "not_a_tool"));

    tg_free(g);
    jr_free(tools);
}

/* ============================================================
 * BUG 1 — forcing must engage from the grammar's OWN "</think>" detection,
 * WITHOUT any external tg_reasoning_end() call (the emitter's <think> flag is
 * unreliable on the Q3 model).  The grammar watches its own token stream:
 * reasoning text flows freely, then "</think>" self-fires the forced constraint,
 * and the model is compelled into a valid call.
 * ============================================================ */
static void test_forced_self_detects_think_close(void) {
    jr_value *tools = jr_parse(TOOLS_JSON);
    tg_grammar *g = tg_build(tools);

    tg_state *st = tg_state_new_forced(g, NULL);
    bool ok;
    /* Reasoning phase (mask inactive) — arbitrary prose is allowed. */
    feed_forced(st, "I should read the file to answer.", &ok);
    ASSERT_TRUE(ok);
    ASSERT_FALSE(tg_mask_active(st));   /* grace: not yet forced */
    /* The model emits "</think>" — the grammar detects it ITSELF (no external
     * tg_reasoning_end call) and flips into the forced constraint. */
    feed_forced(st, "</think>", &ok);
    ASSERT_TRUE(ok);
    ASSERT_TRUE(tg_mask_active(st));    /* self-detected close → opener compelled */
    /* Post-close: prose is now impossible, a valid call is compelled. */
    ASSERT_FALSE(tg_token_allowed(st, "Sorry, I can't."));
    ASSERT_TRUE(tg_token_allowed(st, "<function=read>"));
    feed_forced(st, "<function=read>\n<parameter=path>\nREADME.md\n</parameter>\n</function>", &ok);
    ASSERT_TRUE(ok);
    ASSERT_TRUE(tg_complete(st));

    tg_state_free(st);
    tg_free(g);
    jr_free(tools);
}

/* BUG 1 — "</think>" split across token boundaries is still detected (the
 * rolling tracker survives token splits, like a real BPE stream). */
static void test_forced_think_close_split_tokens(void) {
    jr_value *tools = jr_parse(TOOLS_JSON);
    tg_grammar *g = tg_build(tools);

    tg_state *st = tg_state_new_forced(g, NULL);
    bool ok;
    feed_forced(st, "reasoning</th", &ok);   ASSERT_TRUE(ok);
    ASSERT_FALSE(tg_mask_active(st));         /* mid-"</think>": still grace */
    feed_forced(st, "ink>", &ok);            ASSERT_TRUE(ok);
    ASSERT_TRUE(tg_mask_active(st));          /* completed across the split */
    feed_forced(st, "<function=read>\n<parameter=path>\nx\n</parameter>\n</function>", &ok);
    ASSERT_TRUE(ok);
    ASSERT_TRUE(tg_complete(st));

    tg_state_free(st);
    tg_free(g);
    jr_free(tools);
}

/* BUG 1 — a call opened DIRECTLY during reasoning (no "</think>" first) still
 * engages the skeleton (auto-like), so the model's own well-formed call is
 * honored even when it never closes think. */
static void test_forced_direct_call_no_think(void) {
    jr_value *tools = jr_parse(TOOLS_JSON);
    tg_grammar *g = tg_build(tools);

    tg_state *st = tg_state_new_forced(g, NULL);
    bool ok;
    /* No external reasoning-end; the model opens a call straight away. */
    feed_forced(st, "<function=read>\n<parameter=path>\nREADME.md\n</parameter>\n</function>", &ok);
    ASSERT_TRUE(ok);
    ASSERT_TRUE(tg_complete(st));   /* skeleton engaged mid-grace, call completed */

    tg_state_free(st);
    tg_free(g);
    jr_free(tools);
}

/* AUTO mode is UNCHANGED by the forced-mode work: prose stays FREE, and the mask
 * only arms inside a call (tg_mask_active == tg_in_call). */
static void test_auto_mode_unchanged(void) {
    jr_value *tools = jr_parse(TOOLS_JSON);
    tg_grammar *g = tg_build(tools);

    tg_state *st = tg_state_new(g);
    tg_reasoning_end(st);   /* no-op in AUTO mode */
    bool ok;
    feed_forced(st, "The answer is 42, no tools needed.", &ok);
    ASSERT_TRUE(ok);                  /* prose unconstrained */
    ASSERT_FALSE(tg_mask_active(st)); /* AUTO: not inside a call → no mask */
    ASSERT_FALSE(tg_in_call(st));
    /* Once a call opens, the mask arms (unchanged auto behavior). */
    tg_advance(st, "<function=glob>");
    ASSERT_TRUE(tg_in_call(st));
    ASSERT_TRUE(tg_mask_active(st));

    tg_state_free(st);
    tg_free(g);
    jr_free(tools);
}

/* ============================================================
 * EMPTY VALUE: the parameter CLOSES, and the call's fate is decided by whether
 * the model went on to close the CALL — not by how much whitespace the value
 * held.
 *
 * THIS TEST USED TO ASSERT THE OPPOSITE (test_empty_value_rejected): that an
 * empty/whitespace value could never complete a call, because TG_VALUE refused to
 * let the terminator start until a non-whitespace char had been seen.  That
 * guarantee was never real, and holding it here hid the defect it caused:
 *
 *   • Refusing to START the terminator is done by CONSUMING those chars as
 *     content, which is an ACCEPT.  tg_token_allowed("\n</parameter>\n") returned
 *     true at an empty value, so nothing was ever masked and the model emitted it
 *     freely — 9 times out of 9 on the live fleet.
 *   • Once eaten, the "</parameter>" broke the terminator match and was swallowed
 *     as value text, so the walk stayed in TG_VALUE for the rest of the turn.
 *     The parser's strstr found that SAME "</parameter>" and closed the value at
 *     it.  Grammar and parser disagreed about a byte sequence both had seen, and
 *     the argument vanished: `read {}`, `write` without its `content`.
 *
 * An empty value is also legal on its own terms — `edit`'s `new_string:""` is a
 * deletion (the client's schema code argument_is_absent) — so rejecting it
 * would break a correct call to catch an incorrect one that the reject does not
 * actually catch.  The truncation signal moved to where it is decidable:
 * tg_complete().
 *
 * SCOPE, since a floor now exists.  Everything below runs on TOOLS_JSON, whose
 * parameters declare no `minLength` — the schema shape that says nothing about
 * emptiness.  A schema that DOES say so is the next test, and the two together
 * are the contract: the grammar refuses an empty value where the schema forbids
 * one, and nowhere else.
 * ============================================================ */
static void test_empty_value_closes_the_parameter(void) {
    jr_value *tools = jr_parse(TOOLS_JSON);
    tg_grammar *g = tg_build(tools);
    bool ok;

    /* An empty value, with the model going on to close the call: accepted AND
     * complete.  The parameter ended at "</parameter>", exactly where the parser
     * ends it, and the call ended at "</function>". */
    tg_state *empty = feed_ok(g,
        "<function=read>\n<parameter=path>\n\n</parameter>\n</function>", &ok);
    ASSERT_TRUE(ok);
    ASSERT_TRUE(tg_complete(empty));
    tg_state_free(empty);

    /* THE DEFECT SHAPE — the same empty value, but the model STOPPED after the
     * close instead of ending the call.  Every char is still on-grammar (nothing
     * to mask: it is a legal prefix), and the walk is back on the SKELETON rather
     * than stranded inside the value, so the incompleteness is visible. */
    tg_state *cut = feed_ok(g,
        "<function=read>\n<parameter=path>\n\n</parameter>\n", &ok);
    ASSERT_TRUE(ok);
    ASSERT_FALSE(tg_complete(cut));   /* the CALL never closed → truncation */
    ASSERT_TRUE(tg_in_call(cut));     /* and it is inside a call, not adrift */
    tg_state_free(cut);

    /* A single content char before the terminator IS accepted (non-empty). */
    tg_state *st = feed_ok(g,
        "<function=read>\n<parameter=path>\nx\n</parameter>\n</function>", &ok);
    ASSERT_TRUE(ok);
    ASSERT_TRUE(tg_complete(st));
    tg_state_free(st);

    /* Whitespace-only values in their several shapes — all close the parameter,
     * none strand the walk.  The old guard treated these as content and lost the
     * close; the value the parser reads out of each is "" either way. */
    tg_state *st2 = feed_ok(g,
        "<function=read>\n<parameter=path>\n\n\n</parameter>\n</function>", &ok);
    ASSERT_TRUE(ok);
    ASSERT_TRUE(tg_complete(st2));
    tg_state_free(st2);

    tg_state *st3 = feed_ok(g,
        "<function=read>\n<parameter=path>\n \n</parameter>\n</function>", &ok);
    ASSERT_TRUE(ok);
    ASSERT_TRUE(tg_complete(st3));
    tg_state_free(st3);

    tg_state *st4 = feed_ok(g,
        "<function=read>\n<parameter=path>\n \t \n</parameter>\n</function>", &ok);
    ASSERT_TRUE(ok);
    ASSERT_TRUE(tg_complete(st4));
    tg_state_free(st4);

    /* CONTROL: leading/trailing whitespace around REAL content is still fine —
     * the fix must not reject values that merely have padding, or every natural
     * "\n<value>\n" emission would break. */
    tg_state *st5 = feed_ok(g,
        "<function=read>\n<parameter=path>\n  /tmp/x.c  \n</parameter>\n</function>", &ok);
    ASSERT_TRUE(ok);
    ASSERT_TRUE(tg_complete(st5));
    tg_state_free(st5);

    /* CONTROL: a value that CONTAINS the terminator's leading newline but keeps
     * going still runs to the real close — the terminator tracker must re-seed,
     * not commit, on a partial match. */
    tg_state *st6 = feed_ok(g,
        "<function=read>\n<parameter=path>\na\n</par b </parameter>\n</function>", &ok);
    ASSERT_TRUE(ok);
    ASSERT_TRUE(tg_complete(st6));
    tg_state_free(st6);

    tg_free(g);
    jr_free(tools);
}

/* ============================================================
 * FLOORED VALUE: "</parameter>" is OFF-GRAMMAR while the value is empty.
 *
 * THE LIVE FAILURE THIS EXISTS FOR.  qwen36-bf16, tool grammar armed and masking
 * (masked_steps non-zero), emitted
 *
 *     [tool_call bash]  {"command":""}
 *     [tool_result bash]  (no output)
 *
 * SEVEN consecutive times — seven full decodes at ~10 tok/s producing nothing —
 * until the run hit its no-progress guard and stopped.  A model that emits an
 * empty required argument does not recover on its own: the empty call teaches it
 * nothing, so it emits the same call again.  That is why this is prevented at
 * emission and not corrected afterwards.
 *
 * WHY THE ASSERTION IS `feed_rejects` AND NOT `!tg_complete`.  Those are two
 * different guarantees and only one of them masks a token.  The guard removed in
 * 08f78d6 satisfied `!tg_complete` — it kept the walk from ever finishing — while
 * ACCEPTING every char, so the model emitted the empty value anyway and the walk
 * was stranded mid-value for the rest of the turn.  Requiring the char to be
 * REFUSED is requiring that tg_token_allowed says no, which is the only form of
 * this constraint that changes what the model can draw.
 * ============================================================ */
static void test_empty_value_off_grammar_when_floored(void) {
    jr_value *tools = jr_parse(CLIENT_FLOORED_JSON);
    tg_grammar *g = tg_build(tools);
    bool ok;

    /* The live shape: bash.command, closed around nothing. */
    ASSERT_TRUE(feed_rejects(g,
        "<function=bash>\n<parameter=command>\n</parameter>"));
    /* The unpadded spelling, and the doubled-newline one. */
    ASSERT_TRUE(feed_rejects(g,
        "<function=bash>\n<parameter=command></parameter>"));
    ASSERT_TRUE(feed_rejects(g,
        "<function=bash>\n<parameter=command>\n\n</parameter>"));
    /* Whitespace is not content: the parser trims it, so a floor satisfied by
     * bytes the host deletes is not a floor. */
    ASSERT_TRUE(feed_rejects(g,
        "<function=bash>\n<parameter=command>\n \n</parameter>"));
    ASSERT_TRUE(feed_rejects(g,
        "<function=bash>\n<parameter=command>\n \t \r\n</parameter>"));

    /* CONTROL: one real char is enough, and the call completes. */
    tg_state *good = feed_ok(g,
        "<function=bash>\n<parameter=command>\nls\n</parameter>\n</function>", &ok);
    ASSERT_TRUE(ok);
    ASSERT_TRUE(tg_complete(good));
    tg_state_free(good);

    /* CONTROL: padding around real content is untouched. */
    tg_state *padded = feed_ok(g,
        "<function=bash>\n<parameter=command>\n  ls -la  \n</parameter>\n</function>",
        &ok);
    ASSERT_TRUE(ok);
    ASSERT_TRUE(tg_complete(padded));
    tg_state_free(padded);

    /* CONTROL: content that merely BEGINS like the terminator is real content,
     * so the close after it is legal.  The floor credits a broken terminator
     * prefix retroactively; without that these would be read as empty. */
    tg_state *lt = feed_ok(g,
        "<function=bash>\n<parameter=command>\n</par x\n</parameter>\n</function>",
        &ok);
    ASSERT_TRUE(ok);
    ASSERT_TRUE(tg_complete(lt));
    tg_state_free(lt);

    tg_state *lone = feed_ok(g,
        "<function=bash>\n<parameter=command>\n<\n</parameter>\n</function>", &ok);
    ASSERT_TRUE(ok);
    ASSERT_TRUE(tg_complete(lone));
    tg_state_free(lone);

    tg_free(g);
    jr_free(tools);
}

/* ============================================================
 * THE OTHER SIDE OF THE SAME PREDICATE: a required string with NO minLength may
 * still be empty, and `edit`'s new_string is exactly that key.
 *
 * Deleting a line is an ordinary operation and `new_string:""` is how it is
 * spelled.  A guard keyed on `required` — or an unconditional one, which is what
 * the JSON dialect shipped for a while — makes that call unemittable, and under
 * tool_choice=required an unemittable correct call is a wedge in its own right.
 * The floored siblings in the SAME tool are still refused, so the verdict is per
 * key and not the guard being off.
 * ============================================================ */
static void test_unfloored_value_may_still_be_empty(void) {
    jr_value *tools = jr_parse(CLIENT_FLOORED_JSON);
    tg_grammar *g = tg_build(tools);
    bool ok;

    tg_state *del = feed_ok(g,
        "<function=edit>\n<parameter=path>\n/tmp/x.c\n</parameter>\n"
        "<parameter=old_string>\ndrop me\n</parameter>\n"
        "<parameter=new_string>\n</parameter>\n</function>", &ok);
    ASSERT_TRUE(ok);
    ASSERT_TRUE(tg_complete(del));
    tg_state_free(del);

    ASSERT_TRUE(feed_rejects(g,
        "<function=edit>\n<parameter=path>\n</parameter>"));
    ASSERT_TRUE(feed_rejects(g,
        "<function=edit>\n<parameter=path>\n/tmp/x.c\n</parameter>\n"
        "<parameter=old_string>\n</parameter>"));

    tg_free(g);
    jr_free(tools);
}

/* The close is honoured WITHOUT the trained padding newlines, which is how the
 * live fleet actually emits it ("</parameter></function>",
 * tool_arg_truncation.md log:274).  The parser has always read that shape; the
 * grammar used to swallow it whole. */
static void test_unpadded_close_is_grammatical(void) {
    jr_value *tools = jr_parse(TOOLS_JSON);
    tg_grammar *g = tg_build(tools);
    bool ok;

    tg_state *tight = feed_ok(g,
        "<function=read>\n<parameter=path>\n/tmp/x.c</parameter></function>", &ok);
    ASSERT_TRUE(ok);
    ASSERT_TRUE(tg_complete(tight));
    tg_state_free(tight);

    /* Two parameters, no padding between them either. */
    tg_state *two = feed_ok(g,
        "<function=glob>\n<parameter=pattern>\n*.c</parameter>"
        "<parameter=path>\nsrc</parameter></function>", &ok);
    ASSERT_TRUE(ok);
    ASSERT_TRUE(tg_complete(two));
    tg_state_free(two);

    tg_free(g);
    jr_free(tools);
}

/* ============================================================
 * MISSING-REQUIRED-ARG IMPOSSIBILITY.  The empty-VALUE guard above only binds a
 * key the model DID emit; leaving a required key out entirely used to be a legal
 * path ("<function=NAME>\n</function>" completed with zero parameters), and the
 * parser turned that into `{}`.  Live on the fleet 2026-08-06, WITH the grammar
 * on, that is exactly what reached the client as `dispatch_agent {}` — refused for a
 * missing "subtask" at the cost of a wasted round trip.  The XML dialect now
 * enforces `required` the way the JSON dialect always has.
 * ============================================================ */

/* dispatch_agent(subtask REQUIRED) + check_agents(agent_ids, none required) —
 * The client's real orchestrator schema, which is where this was observed. */
static const char *REQUIRED_TOOLS_JSON =
    "[{\"type\":\"function\",\"function\":{\"name\":\"dispatch_agent\","
    "\"parameters\":{\"type\":\"object\",\"properties\":{"
    "\"subtask\":{\"type\":\"string\"}},\"required\":[\"subtask\"]}}},"
    "{\"type\":\"function\",\"function\":{\"name\":\"check_agents\","
    "\"parameters\":{\"type\":\"object\",\"properties\":{"
    "\"agent_ids\":{\"type\":\"array\"}},\"required\":[]}}}]";

static void test_missing_required_arg_rejected(void) {
    jr_value *tools = jr_parse(REQUIRED_TOOLS_JSON);
    tg_grammar *g = tg_build(tools);
    bool ok;

    /* THE DEFECT: a no-argument call must not be completable. */
    tg_state *bare = feed_ok(g, "<function=dispatch_agent>\n</function>", &ok);
    ASSERT_FALSE(ok);                  /* the '/' of "</function>" is masked out */
    ASSERT_FALSE(tg_complete(bare));
    tg_state_free(bare);

    /* Supplying the required key completes normally. */
    tg_state *full = feed_ok(g,
        "<function=dispatch_agent>\n<parameter=subtask>\ndo it\n</parameter>\n"
        "</function>", &ok);
    ASSERT_TRUE(ok);
    ASSERT_TRUE(tg_complete(full));
    tg_state_free(full);

    /* A tool with an EMPTY `required` list is unaffected: no-arg still closes. */
    tg_state *none = feed_ok(g, "<function=check_agents>\n</function>", &ok);
    ASSERT_TRUE(ok);
    ASSERT_TRUE(tg_complete(none));
    tg_state_free(none);

    /* Same guarantee under forced decoding (tool_choice "required"), which is
     * the mode the orchestrator actually runs in. */
    tg_state *f = tg_state_new_forced(g, NULL);
    tg_reasoning_end(f);
    feed_forced(f, "<function=dispatch_agent>\n</function>", &ok);
    ASSERT_FALSE(ok);
    ASSERT_FALSE(tg_complete(f));
    tg_state_free(f);

    /* No schema `required` at all (the pre-existing TOOLS_JSON) keeps the old
     * permissive behavior — this change only binds what the schema declares. */
    jr_value *loose_tools = jr_parse(TOOLS_JSON);
    tg_grammar *lg = tg_build(loose_tools);
    tg_state *loose = feed_ok(lg, "<function=read>\n</function>", &ok);
    ASSERT_TRUE(ok);
    ASSERT_TRUE(tg_complete(loose));
    tg_state_free(loose);
    tg_free(lg);
    jr_free(loose_tools);

    tg_free(g);
    jr_free(tools);
}

/* Each call in a stream is judged on its own: a satisfied call must not let a
 * following empty one through (seen_keys resets when a NAME is matched). */
static void test_required_resets_between_calls(void) {
    jr_value *tools = jr_parse(REQUIRED_TOOLS_JSON);
    tg_grammar *g = tg_build(tools);
    bool ok;

    tg_state *st = feed_ok(g, "<function=check_agents>\n</function>", &ok);
    ASSERT_TRUE(ok);

    /* Start a SECOND call that omits its required key — still rejected. */
    char one[2] = {0, 0};
    const char *second = "\n<function=dispatch_agent>\n</function>";
    bool second_ok = true;
    for (const char *p = second; *p; p++) {
        one[0] = *p;
        if (!tg_token_allowed(st, one)) { second_ok = false; break; }
        tg_advance(st, one);
    }
    ASSERT_FALSE(second_ok);
    tg_state_free(st);

    tg_free(g);
    jr_free(tools);
}

/* ============================================================
 * FORCED EARLY-ARM (BUG 1 hardening): the mask must engage the instant the model
 * starts a tool-call opener DURING reasoning — WITHOUT waiting for "</think>".
 * ============================================================ */
static void test_forced_early_arm_on_opener(void) {
    jr_value *tools = jr_parse(TOOLS_JSON);
    tg_grammar *g = tg_build(tools);

    tg_state *st = tg_state_new_forced(g, NULL);
    bool ok;
    /* Reasoning grace: prose unconstrained, mask off. */
    feed_forced(st, "thinking about it", &ok);
    ASSERT_TRUE(ok);
    ASSERT_FALSE(tg_mask_active(st));
    /* Model starts an opener mid-reasoning (no "</think>") — mask arms NOW. */
    feed_forced(st, "<function=", &ok);
    ASSERT_TRUE(ok);
    ASSERT_TRUE(tg_mask_active(st));      /* early-armed on opener progress */
    /* From here only a valid NAME is allowed; garbage name is rejected. */
    ASSERT_FALSE(tg_token_allowed(st, "base_agent_agent>"));
    ASSERT_TRUE(tg_token_allowed(st, "read>"));

    tg_state_free(st);
    tg_free(g);
    jr_free(tools);
}

/* ============================================================
 * FORCED REASONING BUDGET: a model that degenerates in <think> and NEVER emits
 * "</think>" must still be compelled — after TG_REASON_BUDGET grace chars the
 * opener is required (prose becomes impossible), so a repetition loop cannot
 * stream unconstrained forever.
 * ============================================================ */
static void test_forced_reasoning_budget(void) {
    jr_value *tools = jr_parse(TOOLS_JSON);
    tg_grammar *g = tg_build(tools);

    tg_state *st = tg_state_new_forced(g, NULL);
    /* Stream a long degenerate ramble (spaces — allowed in grace) with NO
     * "</think>" and NO opener, past the internal reasoning budget (8192 chars).
     * Feed a generous 16384 so we clear the budget regardless.  Every space stays
     * in FREE grace and counts against the budget; mask stays off until it trips. */
    char one[2] = {' ', 0};
    bool ever_masked = false;
    for (int i = 0; i < 16384; i++) {
        if (!tg_token_allowed(st, one)) break;   /* prose disallowed once budget trips */
        tg_advance(st, one);
        if (tg_mask_active(st)) { ever_masked = true; break; }
    }
    /* The grace is spent: the mask armed (opener compelled) and non-opener prose
     * is impossible; a valid opener is still accepted. */
    ASSERT_TRUE(ever_masked);
    ASSERT_TRUE(tg_mask_active(st));
    ASSERT_FALSE(tg_token_allowed(st, "more prose"));
    ASSERT_TRUE(tg_token_allowed(st, "<function=read>"));

    tg_state_free(st);

    /* Opener-RICH ramble must ALSO trip the budget.  The live degeneration is full
     * of "<tools>"/"<function=" in the model's OWN reasoning; the char counter must
     * not be starved by mid-opener chars (the bug that let a 3500-token ramble run
     * without arming in serving).  Feed a '<'-littered stream and confirm the mask
     * still arms within a bounded number of chars. */
    tg_state *st2 = tg_state_new_forced(g, NULL);
    const char *chunk = "<tools> <function reasoning about <tool_call> stuff ";
    bool armed2 = false;
    for (int i = 0; i < 2000 && !armed2; i++) {
        for (const char *p = chunk; *p; p++) {
            char one2[2] = { *p, 0 };
            if (!tg_token_allowed(st2, one2)) break;
            tg_advance(st2, one2);
            if (tg_mask_active(st2)) { armed2 = true; break; }
        }
    }
    ASSERT_TRUE(armed2);              /* budget tripped despite opener-rich text */
    ASSERT_TRUE(tg_mask_active(st2));
    tg_state_free(st2);

    /* PROSE-at-trip regression (the exact bug seen in serving): when the budget trips on a
     * NON-whitespace prose char, arming must NOT drive the state to TG_DEAD — a
     * DEAD state reports the mask INACTIVE, silently disengaging the constraint so
     * the ramble streams on unmasked forever.  Feed continuous letters (no spaces,
     * no openers) well past the budget and assert the mask arms AND STAYS armed. */
    tg_state *st3 = tg_state_new_forced(g, NULL);
    char one3[2] = {'a', 0};
    bool armed3 = false;
    for (int i = 0; i < 12000; i++) {
        one3[0] = "abcdefgh"[i & 7];   /* pure letters: never whitespace, never '<' */
        /* In the sampler the mask (when active) would forbid this letter; here we
         * just drive the LIVE state to observe arming, so advance unconditionally. */
        tg_advance(st3, one3);
        if (tg_mask_active(st3)) { armed3 = true; break; }
    }
    ASSERT_TRUE(armed3);              /* budget armed the mask on prose */
    /* Critically: the state is NOT dead — the mask is genuinely active and a valid
     * opener is still reachable (arming survived the prose char that tripped it). */
    ASSERT_TRUE(tg_mask_active(st3));
    ASSERT_TRUE(tg_token_allowed(st3, "<function=read>"));
    ASSERT_FALSE(tg_token_allowed(st3, "still rambling"));
    tg_state_free(st3);

    tg_free(g);
    jr_free(tools);
}

/* ============================================================
 * ENUM — the live switch_mode victim, XML dialect.  Every declared mode is
 * still emittable and completes; anything off the list is structurally
 * impossible.  BOTH halves matter: a grammar that rejected every value would
 * pass the reject half on its own.
 * ============================================================ */
static void test_enum_closes_switch_mode(void) {
    jr_value *tools = jr_parse(SWITCH_MODE_JSON);
    tg_grammar *g = tg_build(tools);
    ASSERT_NOT_NULL(g);

    static const char *const MODES[] = { "chat", "plan", "act", "orchestrate" };
    for (int i = 0; i < 4; i++) {
        char call[256];
        snprintf(call, sizeof(call),
                 "<function=switch_mode>\n"
                 "<parameter=mode>\n%s\n</parameter>\n"
                 "<parameter=reason>\nthe job needs it\n</parameter>\n"
                 "</function>", MODES[i]);
        bool ok;
        tg_state *st = feed_ok(g, call, &ok);
        ASSERT_TRUE(ok);                 /* every legal mode still accepted */
        ASSERT_TRUE(tg_complete(st));
        tg_state_free(st);
    }

    /* The bug this closes. */
    ASSERT_TRUE(feed_rejects(g,
        "<function=switch_mode>\n<parameter=mode>\nbanana"));
    /* Off-set values that share a prefix die where they diverge. */
    ASSERT_TRUE(feed_rejects(g,
        "<function=switch_mode>\n<parameter=mode>\nplans"));
    /* A strict PREFIX of a member is not a member: the terminator cannot start. */
    ASSERT_TRUE(feed_rejects(g,
        "<function=switch_mode>\n<parameter=mode>\npla\n</parameter>"));
    /* An empty enum VALUE is impossible (no member is empty). */
    ASSERT_TRUE(feed_rejects(g,
        "<function=switch_mode>\n<parameter=mode>\n\n</parameter>"));

    /* A member split across token boundaries still matches. */
    tg_state *split = tg_state_new(g);
    tg_advance(split, "<function=switch_mode>\n<parameter=mode>\norch");
    ASSERT_TRUE(tg_token_allowed(split, "estrate\n</parameter>\n"));
    ASSERT_FALSE(tg_token_allowed(split, "id\n</parameter>\n"));  /* "orchid" */
    tg_state_free(split);

    tg_free(g);
    jr_free(tools);
}

/* An enum on an OPTIONAL key constrains the value WITHOUT making the key
 * mandatory: omitting it entirely still completes. */
static void test_enum_on_optional_key(void) {
    jr_value *tools = jr_parse(ENUM_OPTIONAL_JSON);
    tg_grammar *g = tg_build(tools);
    bool ok;

    tg_state *a = feed_ok(g,
        "<function=log>\n<parameter=msg>\nhi\n</parameter>\n</function>", &ok);
    ASSERT_TRUE(ok);                 /* level omitted → still a complete call */
    ASSERT_TRUE(tg_complete(a));
    tg_state_free(a);

    tg_state *b = feed_ok(g,
        "<function=log>\n<parameter=msg>\nhi\n</parameter>\n"
        "<parameter=level>\nwarn\n</parameter>\n</function>", &ok);
    ASSERT_TRUE(ok);
    ASSERT_TRUE(tg_complete(b));
    tg_state_free(b);

    /* Present but off-set is still impossible. */
    ASSERT_TRUE(feed_rejects(g,
        "<function=log>\n<parameter=msg>\nhi\n</parameter>\n"
        "<parameter=level>\nloud"));

    tg_free(g);
    jr_free(tools);
}

/* Enums we cannot represent DEGRADE to a free-text VALUE — they never reject a
 * legal call (the TG_REQ_MAX_KEYS precedent). */
static void test_enum_degrades_when_unrepresentable(void) {
    jr_value *tools = jr_parse(ENUM_DEGRADE_JSON);
    tg_grammar *g = tg_build(tools);
    ASSERT_NOT_NULL(g);
    bool ok;

    /* nums / mixed / blank / nolist / clash: all dropped → free text, so text
     * that is nowhere in the enum is accepted rather than rejected. */
    static const char *const DROPPED[] = { "nums", "mixed", "blank", "nolist", "clash" };
    for (int i = 0; i < 5; i++) {
        char call[256];
        snprintf(call, sizeof(call),
                 "<function=cfg>\n<parameter=%s>\nanything at all\n</parameter>\n"
                 "</function>", DROPPED[i]);
        tg_state *st = feed_ok(g, call, &ok);
        ASSERT_TRUE(ok);
        ASSERT_TRUE(tg_complete(st));
        tg_state_free(st);
    }

    /* notype: string members and NO declared type → the enum DOES apply. */
    tg_state *d = feed_ok(g,
        "<function=cfg>\n<parameter=notype>\nyes\n</parameter>\n</function>", &ok);
    ASSERT_TRUE(ok);
    ASSERT_TRUE(tg_complete(d));
    tg_state_free(d);
    ASSERT_TRUE(feed_rejects(g, "<function=cfg>\n<parameter=notype>\nmaybe"));

    tg_free(g);
    jr_free(tools);
}

/* An enum larger than the cap degrades to free text rather than rejecting. */
static void test_enum_over_cap_degrades(void) {
    char schema[8192];
    int n = snprintf(schema, sizeof(schema),
                     "[{\"type\":\"function\",\"function\":{\"name\":\"pick\","
                     "\"parameters\":{\"type\":\"object\",\"properties\":{"
                     "\"v\":{\"type\":\"string\",\"enum\":[");
    for (int i = 0; i < 65; i++)                       /* 65 > TG_ENUM_MAX_MEMBERS */
        n += snprintf(schema + n, sizeof(schema) - (size_t)n,
                      "%s\"m%03d\"", i ? "," : "", i);
    snprintf(schema + n, sizeof(schema) - (size_t)n,
             "]}},\"required\":[\"v\"]}}}]");

    jr_value *tools = jr_parse(schema);
    tg_grammar *g = tg_build(tools);
    ASSERT_NOT_NULL(g);

    bool ok;
    tg_state *st = feed_ok(g,
        "<function=pick>\n<parameter=v>\nanything\n</parameter>\n</function>", &ok);
    ASSERT_TRUE(ok);                 /* off-list accepted → the enum was dropped */
    ASSERT_TRUE(tg_complete(st));
    tg_state_free(st);

    tg_free(g);
    jr_free(tools);
}

/* ============================================================
 * LANDING WINDOW (tg_reasoning_end_injected) — the frame-capture cause fix.
 *
 * When a turn spends its reasoning_budget the SERVE force-decodes "</think>"
 * itself.  The grammar cannot tell that close from the model's own, so it used
 * to compel the tool-call opener from the very next char — while the model was
 * still mid-sentence.  The model then emitted the compelled
 * "<function=NAME>\n<parameter=KEY>\n" and dumped the rest of its thought,
 * often a COMPLETE tool call, into the free-text VALUE.  Recorded live:
 * a recorded native-XML capture and three recorded Hermes-shape captures.
 *
 * tg_reasoning_end_injected grants a bounded AUTO-like grace so the model can
 * land: finish its sentence, emit its own close, and open the call itself.
 * ============================================================ */

/* The prose the model actually wrote after being cut off (the Hermes lead-in
 * capture, the text that was captured into the `path` argument). */
static const char CAPTURED_LEADIN_PROSE[] =
    "\n\nLet me look at the Makefile to understand the build structure:\n\n";

/* After an INJECTED close the model's unfinished sentence must still be legal.
 * This is the assertion the bug fails: pre-fix, the first prose char is
 * off-grammar and the model is forced to open a frame around its thought. */
static void test_landing_window_lets_the_model_finish(void) {
    jr_value *tools = jr_parse(TOOLS_JSON);
    tg_grammar *g = tg_build(tools);
    tg_state *st = tg_state_new_forced(g, NULL);

    /* The serve force-decoded "</think>" and told the grammar it was injected. */
    tg_reasoning_end_injected(st);
    ASSERT_INT_EQ(tg_debug_landing_chars(st), 512);
    /* AUTO-like during the window: no mask, prose allowed. */
    ASSERT_FALSE(tg_mask_active(st));
    ASSERT_TRUE(tg_token_allowed(st, "Let me look at the Makefile"));

    bool ok;
    feed_forced(st, CAPTURED_LEADIN_PROSE, &ok);
    ASSERT_TRUE(ok);                       /* the whole sentence stayed legal */
    ASSERT_FALSE(tg_in_call(st));          /* and was NOT swallowed into a frame */

    tg_state_free(st);
    tg_free(g);
    jr_free(tools);
}

/* End to end on the real capture: after the injected close the model writes its
 * prose AND THEN its own call — which now lands as a real call instead of as the
 * string value of a compelled `read`.  The path is the one the model asked for. */
static void test_landing_window_recovers_the_real_capture(void) {
    jr_value *tools = jr_parse(TOOLS_JSON);
    tg_grammar *g = tg_build(tools);
    tg_state *st = tg_state_new_forced(g, NULL);
    bool ok;

    tg_reasoning_end_injected(st);
    feed_forced(st, CAPTURED_LEADIN_PROSE, &ok);
    ASSERT_TRUE(ok);
    /* The model now opens the call it meant to make all along. */
    feed_forced(st, "<tool_call>\n<function=read>\n<parameter=path>\n"
                    "lumen/Makefile\n</parameter>\n</function>\n"
                    "</tool_call>", &ok);
    ASSERT_TRUE(ok);
    ASSERT_TRUE(tg_complete(st));          /* a COMPLETE call, not a captured one */

    tg_state_free(st);
    tg_free(g);
    jr_free(tools);
}

/* The window ends the moment the model emits its OWN "</think>" — it has now
 * genuinely concluded, so the opener is compelled again immediately.  This is
 * what actually terminates the window in every recorded capture (all of them
 * open with the model's own close). */
static void test_landing_window_ends_on_models_own_close(void) {
    jr_value *tools = jr_parse(TOOLS_JSON);
    tg_grammar *g = tg_build(tools);
    tg_state *st = tg_state_new_forced(g, NULL);
    bool ok;

    tg_reasoning_end_injected(st);
    ASSERT_INT_EQ(tg_debug_landing_chars(st), 512);
    /* Exactly the shape seen live: the model writes the close it had not
     * reached yet as the first thing after being cut off. */
    feed_forced(st, "</think>", &ok);
    ASSERT_TRUE(ok);
    ASSERT_INT_EQ(tg_debug_landing_chars(st), 0);   /* window closed early */
    /* Compelled again: prose is off-grammar, an opener is not. */
    ASSERT_FALSE(tg_token_allowed(st, "Let me look"));
    ASSERT_TRUE(tg_token_allowed(st, "<function="));
    ASSERT_TRUE(tg_mask_active(st));

    tg_state_free(st);
    tg_free(g);
    jr_free(tools);
}

/* The window is BOUNDED: a model that just keeps rambling is compelled again
 * once the char budget runs out, so the runaway the reasoning budget exists to
 * stop is not reopened.  tool_choice=required is preserved, only deferred. */
static void test_landing_window_closes_at_budget(void) {
    jr_value *tools = jr_parse(TOOLS_JSON);
    tg_grammar *g = tg_build(tools);
    tg_state *st = tg_state_new_forced(g, NULL);

    tg_reasoning_end_injected(st);
    /* Ramble with no close and no opener.  Spaces are legal in the window and
     * each one spends budget. */
    char one[2] = {'x', 0};
    int fed = 0;
    for (int i = 0; i < 4096; i++) {
        if (!tg_token_allowed(st, one)) break;
        tg_advance(st, one);
        fed++;
        if (tg_debug_landing_chars(st) == 0) break;
    }
    ASSERT_INT_EQ(tg_debug_landing_chars(st), 0);
    ASSERT_INT_EQ(fed, 512);                  /* spent exactly the budget */
    /* Compelled again. */
    ASSERT_FALSE(tg_token_allowed(st, "still rambling"));
    ASSERT_TRUE(tg_token_allowed(st, "<function="));

    tg_state_free(st);
    tg_free(g);
    jr_free(tools);
}

/* THE COMMON CASE, untouched: a model that closes its own reasoning gets NO
 * window — the grammar self-detects "</think>" in the stream exactly as before
 * and compels the opener from the next char.  Byte-identical to pre-fix
 * behavior, which is the bar: only an INJECTED close changes anything. */
static void test_landing_window_absent_on_self_close(void) {
    jr_value *tools = jr_parse(TOOLS_JSON);
    tg_grammar *g = tg_build(tools);
    tg_state *st = tg_state_new_forced(g, NULL);
    bool ok;

    feed_forced(st, "thinking it through</think>", &ok);
    ASSERT_TRUE(ok);
    ASSERT_INT_EQ(tg_debug_landing_chars(st), 0);   /* no grace granted */
    ASSERT_TRUE(tg_forced_reasoning_ended(st));
    ASSERT_FALSE(tg_token_allowed(st, "prose"));    /* compelled immediately */
    ASSERT_TRUE(tg_token_allowed(st, "<function="));

    tg_state_free(st);
    tg_free(g);
    jr_free(tools);
}

/* AUTO mode never compels an opener, so an injected close must not change it
 * either — prose stays free and the mask stays off, window or not. */
static void test_landing_window_absent_in_auto_mode(void) {
    jr_value *tools = jr_parse(TOOLS_JSON);
    tg_grammar *g = tg_build(tools);
    tg_state *st = tg_state_new(g);          /* AUTO */
    bool ok;

    tg_reasoning_end_injected(st);
    feed_forced(st, "The answer is 42, no tools needed.", &ok);
    ASSERT_TRUE(ok);
    ASSERT_FALSE(tg_mask_active(st));
    ASSERT_TRUE(tg_token_allowed(st, "more prose"));

    tg_state_free(st);
    tg_free(g);
    jr_free(tools);
}

/* A call opened DURING the window still engages the skeleton — the grace makes
 * prose legal, it does not stop the grammar from constraining a real call.  An
 * unknown tool name is still impossible mid-window. */
static void test_landing_window_opener_still_engages(void) {
    jr_value *tools = jr_parse(TOOLS_JSON);
    tg_grammar *g = tg_build(tools);
    tg_state *st = tg_state_new_forced(g, NULL);
    bool ok;

    tg_reasoning_end_injected(st);
    feed_forced(st, "ok: <function=", &ok);
    ASSERT_TRUE(ok);
    ASSERT_TRUE(tg_mask_active(st));         /* early-armed on opener progress */
    ASSERT_FALSE(tg_token_allowed(st, "not_a_tool>"));
    ASSERT_TRUE(tg_token_allowed(st, "read>"));

    tg_state_free(st);
    tg_free(g);
    jr_free(tools);
}

/* ============================================================
 * Build guards: empty / non-array tools → NULL grammar.
 * ============================================================ */
static void test_build_guards(void) {
    jr_value *empty = jr_parse("[]");
    ASSERT_NULL(tg_build(empty));
    jr_free(empty);

    jr_value *obj = jr_parse("{\"x\":1}");
    ASSERT_NULL(tg_build(obj));
    jr_free(obj);
}


/* ============================================================
 * THE NEAR-MISS OPENER — measured on qwen36-bf16 (BF16 full weights, 7 tools,
 * 558-token prompt) with the grammar OFF.  The model does not drift to JSON and
 * does not answer in prose: it invents a tool-call frame that is one character
 * away from the real opener.
 *
 *     <tool_calls>\n<tool>\n<tool_name>bash</tool_name>…
 *
 * This is the case the AUTO/FORCED split actually turns on, and it is NOT the
 * "grammar armed, then went DEAD" story.  In AUTO the mask is inactive for the
 * whole of TG_FREE (tg_mask_active only engages at tg_in_call), so every one of
 * those characters is sampled unconstrained and the walk simply never enters the
 * skeleton — an armed AUTO grammar would have changed nothing about this turn.
 * FORCED engages the moment an opener prefix starts (free_open > 0), so the 's'
 * that breaks "<tool_call" is masked and the frame cannot be invented.
 * ============================================================ */
static void test_auto_never_masks_a_near_miss_opener(void) {
    jr_value *tools = jr_parse(TOOLS_JSON);
    tg_grammar *g = tg_build(tools);

    tg_state *st = tg_state_new(g);          /* AUTO */
    const char *near_miss = "<tool_calls>";
    char one[2] = {0, 0};
    bool masked_anywhere = false;
    for (const char *p = near_miss; *p; p++) {
        if (tg_mask_active(st)) masked_anywhere = true;
        one[0] = *p;
        /* AUTO never rejects in FREE: the near miss is fully sampleable. */
        ASSERT_TRUE(tg_token_allowed(st, one));
        tg_advance(st, one);
    }
    ASSERT_FALSE(masked_anywhere);   /* the constraint never engaged */
    ASSERT_FALSE(tg_in_call(st));    /* and the walk never entered a call */
    ASSERT_FALSE(tg_is_dead(st));    /* NOT dead — it is still plain FREE */
    tg_state_free(st);

    /* AUTO IS NOT USELESS — it just cannot reach INSIDE the opener.  The other
     * half of the same live defect gets the wrapper exactly right and invents
     * only the inner tag ("<tool_call>\n<bash>\n<parameter=command>", four of
     * them in one fleet log).  There the opener DOES complete, the walk enters
     * the skeleton, and AUTO masks "<bash>" out — so the two sub-classes have
     * opposite outcomes under the same mode, and only FORCED covers both. */
    tg_state *inner = tg_state_new(g);
    tg_advance(inner, "<tool_call>");
    ASSERT_TRUE(tg_in_call(inner));
    ASSERT_TRUE(tg_mask_active(inner));
    ASSERT_FALSE(tg_token_allowed(inner, "\n<glob>"));      /* the invented tag */
    ASSERT_TRUE(tg_token_allowed(inner, "\n<function=glob>"));
    tg_state_free(inner);

    /* FORCED, same bytes: the mask arms on '<' and the 's' is off-grammar. */
    tg_state *fs = tg_state_new_forced(g, NULL);
    tg_reasoning_end(fs);
    tg_advance(fs, "<tool_call");
    ASSERT_TRUE(tg_mask_active(fs));          /* early-arm on the opener prefix */
    ASSERT_FALSE(tg_token_allowed(fs, "s"));  /* the near miss is unreachable */
    ASSERT_TRUE(tg_token_allowed(fs, ">"));   /* the real opener still is */
    tg_state_free(fs);

    tg_free(g);
    jr_free(tools);
}

/* ============================================================
 * DEAD IS NOT FREE.  tg_in_call() is false for both, which is why "the AUTO
 * grammar switched itself off" was unobservable: a walk that went off-grammar
 * mid-call and a walk that never started one reported the same thing.  A DEAD
 * walk also reports its mask INACTIVE, so every remaining token of the turn is
 * sampled with no constraint at all.
 * ============================================================ */
static void test_dead_walk_is_distinguishable_from_free(void) {
    jr_value *tools = jr_parse(TOOLS_JSON);
    tg_grammar *g = tg_build(tools);

    /* FREE: prose, no call. Not dead. */
    tg_state *free_st = tg_state_new(g);
    tg_advance(free_st, "The answer is 42.");
    ASSERT_FALSE(tg_in_call(free_st));
    ASSERT_FALSE(tg_is_dead(free_st));
    tg_state_free(free_st);

    /* DEAD: a real opener, then an off-grammar byte inside the skeleton. */
    tg_state *dead_st = tg_state_new(g);
    tg_advance(dead_st, "<function=glob>");
    ASSERT_TRUE(tg_in_call(dead_st));
    ASSERT_TRUE(tg_mask_active(dead_st));
    tg_advance(dead_st, "\nnot-a-parameter");
    ASSERT_TRUE(tg_is_dead(dead_st));
    ASSERT_FALSE(tg_in_call(dead_st));    /* indistinguishable from FREE here */
    ASSERT_FALSE(tg_mask_active(dead_st));/* the constraint is off for the rest */
    tg_state_free(dead_st);

    tg_free(g);
    jr_free(tools);
}

/* ============================================================
 * CONTAINER-TYPED VALUES: the frame was constrained, the ARGUMENT was not.
 *
 * The client's REAL todowrite schema (the client's schema code emit_todo_tool):
 * `todos` is {"type":"array","items":{"type":"object",…}}, the only tool in the
 * live toolset with a nested shape.  `read` and `bash` — flat, single-string
 * arguments — were flawless for a whole run; todowrite failed three times with
 * the same malformation and never recovered, because the bytes inside
 * <parameter=todos> were free text.
 *
 * tool_calling.c splices a container-typed value into the call's `arguments`
 * object RAW, so one malformed value does not produce a bad argument — it
 * produces an `arguments` string that will not parse at all, taking every other
 * argument of the call with it.
 * ============================================================ */
static const char *TODOWRITE_JSON =
    "[{\"type\":\"function\",\"function\":{\"name\":\"todowrite\","
    "\"parameters\":{\"type\":\"object\",\"properties\":"
    "{\"todos\":{\"type\":\"array\",\"items\":{\"type\":\"object\","
    "\"properties\":{\"content\":{\"type\":\"string\",\"minLength\":1},"
    "\"status\":{\"type\":\"string\","
    "\"enum\":[\"pending\",\"in_progress\",\"completed\"]}},"
    "\"required\":[\"content\",\"status\"]}}},"
    "\"required\":[\"todos\"]}}}]";

/* Wrap a `todos` VALUE in the trained XML call the model actually emits. */
static char *todowrite_call(const char *value) {
    static char buf[2048];
    snprintf(buf, sizeof(buf),
             "<function=todowrite>\n<parameter=todos>\n%s\n</parameter>\n"
             "</function>", value);
    return buf;
}

static void test_malformed_nested_argument_is_rejected(void) {
    jr_value *tools = jr_parse(TODOWRITE_JSON);
    tg_grammar *g = tg_build(tools);
    ASSERT_NOT_NULL(g);

    /* The live failures, verbatim in shape: a bare value where a key was owed. */
    ASSERT_TRUE(feed_rejects(g, todowrite_call(
        "[{\"content\":\"Remove idx_points_scope\",\"status\":\"in_progress\"},"
        "{\"content\":\"\",\"\"}]")));
    ASSERT_TRUE(feed_rejects(g, todowrite_call(
        "[{\"content\":\"a\",\"status\":\"pending\"},"
        "{\"content\":\"_verify the fix works end-to-end_\",\"_verify\"}]")));

    /* What the model meant is still reachable, end to end. */
    bool ok = false;
    tg_state *st = feed_ok(g, todowrite_call(
        "[{\"content\":\"Remove idx_points_scope\",\"status\":\"in_progress\"},"
        "{\"content\":\"verify the fix works\",\"status\":\"pending\"}]"), &ok);
    ASSERT_TRUE(ok);
    ASSERT_TRUE(tg_complete(st));
    tg_state_free(st);

    /* An empty list is a legal, reachable emission — the container-typed rule
     * bans "" (a string, wrong type), not emptiness. */
    ok = false;
    tg_state *st2 = feed_ok(g, todowrite_call("[]"), &ok);
    ASSERT_TRUE(ok);
    ASSERT_TRUE(tg_complete(st2));
    tg_state_free(st2);

    /* Wrong JSON type for an "array" parameter: well-formed, still refused. */
    ASSERT_TRUE(feed_rejects(g, todowrite_call("{\"a\":1}")));
    ASSERT_TRUE(feed_rejects(g, todowrite_call("\"buy milk\"")));
    /* Unclosed: "</parameter>" is not a legal continuation until the value closes. */
    ASSERT_TRUE(feed_rejects(g, todowrite_call("[{\"content\":\"a\"")));

    tg_free(g);
    jr_free(tools);
}

/* The grammar and the parser must not disagree about where a value ended.
 * parse_call_arguments finds the value's end with strstr(v, "</parameter>"), so
 * those bytes INSIDE a JSON string would cut the value mid-structure and hand
 * write_typed_value an un-parseable prefix.  Off-grammar, therefore never drawn. */
static void test_terminator_inside_a_json_string_is_off_grammar(void) {
    jr_value *tools = jr_parse(TODOWRITE_JSON);
    tg_grammar *g = tg_build(tools);

    ASSERT_TRUE(feed_rejects(g, todowrite_call(
        "[{\"content\":\"a</parameter>b\",\"status\":\"pending\"}]")));
    /* A lone '<' inside a string is ordinary text and must stay legal — the
     * guard is the whole literal, not the character. */
    bool ok = false;
    tg_state *st = feed_ok(g, todowrite_call(
        "[{\"content\":\"a < b\",\"status\":\"pending\"}]"), &ok);
    ASSERT_TRUE(ok);
    ASSERT_TRUE(tg_complete(st));
    tg_state_free(st);

    tg_free(g);
    jr_free(tools);
}

/* Scope guard: a STRING parameter keeps its free-text VALUE.  `write`'s
 * `content` and `bash`'s `command` carry braces, brackets and quotes all day —
 * constraining them would break every file the model writes. */
static void test_string_parameters_stay_free_text(void) {
    jr_value *tools = jr_parse(TOOLS_JSON);   /* glob/read: all string params */
    tg_grammar *g = tg_build(tools);

    bool ok = false;
    tg_state *st = feed_ok(g,
        "<function=read>\n<parameter=path>\n{ not [ json \" at all\n"
        "</parameter>\n</function>", &ok);
    ASSERT_TRUE(ok);
    ASSERT_TRUE(tg_complete(st));
    tg_state_free(st);

    tg_free(g);
    jr_free(tools);
}

/* ============================================================
 * THE BUDGET MUST TRIP AT A TOKEN BOUNDARY, NOT MID-TOKEN.
 *
 * Live regression, qwen36-bf16, one recorded session, 2026-08-26: THREE
 * turns ended
 *
 *   [tool_grammar] turn end: the walk went DEAD in FORCED mode — the mask
 *   reported inactive from that point and the rest of this turn decoded
 *   UNCONSTRAINED (dead_turns=1..3)
 *   [tool_grammar] turn mask density: masked_steps=0 sparse_rows=0
 *
 * masked_steps=0 is the whole story: the walk died BEFORE the mask was ever
 * evaluated, so nothing the model wrote for the rest of that turn was
 * constrained.  And because tg_forced_reasoning_ended() kept saying "reasoning
 * is over" from a walk that had already given up, the serve flipped the emitter
 * to the CONTENT channel (serve_generation.c) — so the rest of the model's think
 * block was recorded as an ANSWER, mid-sentence.
 *
 * The cause is a granularity mismatch this suite could not see.  The budget trip
 * consumes its char and returns, on the stated promise "arm cleanly, then mask
 * the next step".  That promise holds only if the next CHAR is the next STEP.
 * It is not: tg_advance walks a whole token, so the next char of the SAME token
 * is judged under `forced` with no mask in between, and one prose char there is
 * TG_FREE_BREAK → TG_DEAD.
 *
 * Every pre-existing budget test fed the grammar ONE CHAR AT A TIME, which is
 * the one granularity at which the bug cannot occur.  These feed tokens.
 * ============================================================ */
static void test_reason_budget_trips_mid_token_without_dying(void) {
    jr_value *tools = jr_parse(TOOLS_JSON);
    tg_grammar *g = tg_build(tools);
    tg_state *st = tg_state_new_forced(g, NULL);

    /* Realistic BPE pieces: multi-char, prose, no opener, no "</think>".  The
     * budget (8192 chars) cannot land on a token boundary with a 9-char piece,
     * so the trip is always mid-token — exactly the live shape. */
    const char *piece = " thinking";
    ASSERT_TRUE(strlen(piece) > 1);
    bool armed = false;
    for (int i = 0; i < 4096 && !armed; i++) {
        tg_advance(st, piece);
        armed = tg_mask_active(st);
        if (tg_is_dead(st)) break;
    }

    /* THE RED ASSERTION.  Before the fix the walk is DEAD here and `armed` never
     * became true — mask inactive for the rest of the turn, masked_steps=0. */
    ASSERT_FALSE(tg_is_dead(st));
    ASSERT_TRUE(armed);
    ASSERT_TRUE(tg_mask_active(st));

    /* And the compelled region really is compelled from the next token on. */
    ASSERT_TRUE(tg_token_allowed(st, "<function=read>"));
    ASSERT_FALSE(tg_token_allowed(st, " still rambling"));

    tg_state_free(st);
    tg_free(g);
    jr_free(tools);
}

/* The same hazard at the OTHER end of the landing window: the window expires
 * mid-token, and the remaining chars of that token must not be judged under the
 * compelled rule either. */
static void test_landing_window_expires_mid_token_without_dying(void) {
    jr_value *tools = jr_parse(TOOLS_JSON);
    tg_grammar *g = tg_build(tools);
    tg_state *st = tg_state_new_forced(g, NULL);

    tg_reasoning_end_injected(st);
    const char *piece = " landing";   /* 8 chars; 512 % 8 == 0 would boundary-align */
    const char *odd   = " land";      /* 5 chars; 512 % 5 != 0 → expires mid-token */
    (void)piece;
    for (int i = 0; i < 512 && tg_debug_landing_chars(st) > 0; i++) {
        tg_advance(st, odd);
        if (tg_is_dead(st)) break;
    }
    ASSERT_FALSE(tg_is_dead(st));
    ASSERT_INT_EQ(tg_debug_landing_chars(st), 0);
    ASSERT_TRUE(tg_mask_active(st));
    ASSERT_TRUE(tg_token_allowed(st, "<function=read>"));

    tg_state_free(st);
    tg_free(g);
    jr_free(tools);
}

/* A DEAD walk has no standing to tell the serve that reasoning is over.
 *
 * sg_tg_forced_ended() (serve_generation.c) drives exactly one thing: it flips
 * the emitter out of its <think> channel so a COMPELLED call is parsed as a tool
 * call instead of landing in reasoning_content.  A dead walk compels nothing —
 * its mask is inactive and every remaining token is unconstrained — so answering
 * "yes" there routes the model's ongoing THOUGHT into the content channel and
 * the transcript records it as an answer.  That is the leak.
 *
 * Everything else in this module already refuses to speak for a dead walk
 * (tg_mask_active, and the serve's own tool-call verdict); this makes the
 * reasoning-ended report consistent with them. */
static void test_dead_walk_does_not_report_reasoning_ended(void) {
    jr_value *tools = jr_parse(TOOLS_JSON);
    tg_grammar *g = tg_build(tools);
    tg_state *st = tg_state_new_forced(g, NULL);

    tg_reasoning_end(st);
    ASSERT_TRUE(tg_forced_reasoning_ended(st));   /* alive + compelling: yes */

    tg_advance(st, "prose");                      /* off-grammar while compelled */
    ASSERT_TRUE(tg_is_dead(st));
    ASSERT_FALSE(tg_mask_active(st));             /* the constraint is off … */
    ASSERT_FALSE(tg_forced_reasoning_ended(st));  /* … so this must be off too */

    tg_state_free(st);
    tg_free(g);
    jr_free(tools);
}

/* The sampler's full mask rejects a token whose first 1 or 2 bytes kill the walk
 * without walking the rest (engine.c grammar_first_char_verdicts) — sound only
 * because DEAD is absorbing.  Checked at every prefix of a forced XML call. */

/* ============================================================
 * tool_choice "required" with thinking ON (tg_state_new_required, think_open):
 * the think block is NEVER judged — a "<tool_call>" or "<function=" written
 * while thinking is thought, not a call, and a think that never closes never
 * arms the mask — and from the token after "</think>" the call is compelled.
 * ============================================================ */
static void test_required_think_gated_reasoning_is_free(void) {
    jr_value *tools = jr_parse(TOOLS_JSON);
    tg_grammar *g = tg_build(tools);
    tg_state *st = tg_state_new_required(g, true);
    ASSERT_NOT_NULL(st);
    ASSERT_FALSE(tg_mask_active(st));

    const char *thought =
        "I could emit <tool_call>\n<function=read>\n<parameter=path>\nx\n</parameter> "
        "or <function=glob> -- let me think about <function= more.";
    char one[2] = {0, 0};
    bool ok;
    for (const char *p = thought; *p; p++) {
        one[0] = *p;
        ASSERT_TRUE(tg_token_allowed(st, one));
        tg_advance(st, one);
        ASSERT_FALSE(tg_mask_active(st));
    }
    ASSERT_FALSE(tg_in_call(st));

    /* A think that never closes: far past the legacy TG_REASON_BUDGET the mask
     * stays off — reasoning is bounded by max_tokens / reasoning_budget. */
    for (int i = 0; i < 30000; i++) {
        one[0] = "ab <f=>\n"[i % 8];
        tg_advance(st, one);
        if (tg_mask_active(st)) break;
    }
    ASSERT_FALSE(tg_mask_active(st));
    ASSERT_FALSE(tg_forced_reasoning_ended(st));

    /* The close arrives glued to the text after it: the rest of that token is
     * still grace, and the call is compelled from the next token. */
    tg_advance(st, "done.</think>\n\nSure");
    ASSERT_FALSE(tg_is_dead(st));
    ASSERT_TRUE(tg_mask_active(st));
    ASSERT_TRUE(tg_forced_reasoning_ended(st));
    ASSERT_FALSE(tg_token_allowed(st, "Let me see the rest of the file."));
    ASSERT_FALSE(tg_token_allowed(st, "{\"name\": \"read\""));
    feed_forced(st, "\n<tool_call>\n<function=read>\n<parameter=path>\nsrc/main.c\n"
                    "</parameter>\n</function>\n</tool_call>", &ok);
    ASSERT_TRUE(ok);
    ASSERT_TRUE(tg_complete(st));
    ASSERT_FALSE(tg_mask_active(st));
    tg_state_free(st);
    tg_free(g);
    jr_free(tools);
}

/* "required" with NO open think block (thinking off): compelled from token 0. */
static void test_required_without_think_compels_from_start(void) {
    jr_value *tools = jr_parse(TOOLS_JSON);
    tg_grammar *g = tg_build(tools);
    tg_state *st = tg_state_new_required(g, false);
    ASSERT_TRUE(tg_mask_active(st));
    ASSERT_FALSE(tg_token_allowed(st, "Sure, let me look."));
    ASSERT_TRUE(tg_token_allowed(st, "\n<tool_call>"));
    ASSERT_TRUE(tg_token_allowed(st, "<function=read>"));
    tg_state_free(st);
    tg_free(g);
    jr_free(tools);
}

/* The serve's reasoning_budget force-close: the injected "</think>" ends the
 * think-gated block and grants the landing window; the model's own close then
 * compels the call. */
static void test_required_injected_close_lands(void) {
    jr_value *tools = jr_parse(TOOLS_JSON);
    tg_grammar *g = tg_build(tools);
    tg_state *st = tg_state_new_required(g, true);
    bool ok;
    feed_forced(st, "still thinking about <function=", &ok);
    ASSERT_FALSE(tg_mask_active(st));
    tg_advance(st, "</think>");            /* the serve's injected close */
    tg_reasoning_end_injected(st);
    ASSERT_FALSE(tg_mask_active(st));      /* landing: the sentence may finish */
    feed_forced(st, " and then I read it.", &ok);
    ASSERT_TRUE(ok);
    feed_forced(st, "</think>", &ok);      /* the model's own close */
    ASSERT_TRUE(ok);
    ASSERT_TRUE(tg_mask_active(st));
    ASSERT_FALSE(tg_token_allowed(st, "prose"));
    tg_state_free(st);
    tg_free(g);
    jr_free(tools);
}

/* Parse `text` as the serve would (Qwen XML dialect) and return the one call's
 * "name:arguments", or "" when it does not yield exactly one call. */
static void parse_one_call(const char *text, const jr_value *tools, char *out, size_t cap) {
    tc_tool_call *calls = NULL;
    int n = 0;
    out[0] = '\0';
    if (tc_parse_tool_calls(text, tools, TC_DIALECT_QWEN_XML, &calls, &n) == 0 && n == 1)
        snprintf(out, cap, "%s:%s", calls[0].name, calls[0].arguments);
    tc_tool_calls_free(calls, n);
}

/* A "required" turn may end only where that is legal: never before the call
 * opens, and inside the call only where the serve can close it — and the
 * closed call parses to what the model wrote. */
static void test_required_end_and_closing_text(void) {
    jr_value *tools = jr_parse(CLIENT_FLOORED_JSON);
    tg_grammar *g = tg_build(tools);
    char closing[TG_CLOSING_MAX], text[512], got[512];
    bool ok;

    tg_state *st = tg_state_new_required(g, true);
    ASSERT_TRUE(tg_end_allowed(st));                  /* thinking: free */
    feed_forced(st, "plan</think>\n\n", &ok);
    ASSERT_FALSE(tg_end_allowed(st));                 /* compelled, no call yet */
    ASSERT_INT_EQ(tg_closing_text(st, closing, TG_CLOSING_MAX), -1);
    feed_forced(st, "<tool_call>\n<function=ba", &ok);
    ASSERT_FALSE(tg_end_allowed(st));                 /* name unfinished */
    feed_forced(st, "sh>\n<parameter=command>\n", &ok);
    ASSERT_TRUE(ok);
    ASSERT_FALSE(tg_end_allowed(st));                 /* floored value still empty */
    feed_forced(st, "ls -la", &ok);
    ASSERT_TRUE(tg_end_allowed(st));
    ASSERT_INT_EQ(tg_closing_text(st, closing, TG_CLOSING_MAX),
                  (int)strlen("\n</parameter>\n</function>\n</tool_call>"));
    ASSERT_STR_EQ(closing, "\n</parameter>\n</function>\n</tool_call>");
    tg_advance(st, closing);
    ASSERT_TRUE(tg_complete(st));
    ASSERT_TRUE(tg_end_allowed(st));                  /* done: free again */
    tg_state_free(st);

    /* The closed call is the call the parser reads from the unclosed text. */
    snprintf(text, sizeof(text), "<tool_call>\n<function=bash>\n<parameter=command>\nls -la%s",
             "\n</parameter>\n</function>\n</tool_call>");
    parse_one_call(text, tools, got, sizeof(got));
    ASSERT_STR_EQ(got, "bash:{\"command\":\"ls -la\"}");
    parse_one_call("<tool_call>\n<function=bash>\n<parameter=command>\nls -la", tools,
                   text, sizeof(text));
    ASSERT_STR_EQ(text, got);

    /* Part-way through "</function>": the rest of the literal, then the wrapper. */
    st = tg_state_new_required(g, false);
    feed_forced(st, "<tool_call>\n<function=bash>\n<parameter=command>\nls\n</parameter>\n"
                    "</func", &ok);
    ASSERT_TRUE(ok);
    ASSERT_INT_EQ(tg_closing_text(st, closing, TG_CLOSING_MAX),
                  (int)strlen("tion>\n</tool_call>"));
    ASSERT_STR_EQ(closing, "tion>\n</tool_call>");
    tg_state_free(st);

    /* Bare form (no wrapper): closed without "</tool_call>". */
    st = tg_state_new_required(g, false);
    feed_forced(st, "<function=bash>\n<parameter=command>\npwd", &ok);
    ASSERT_INT_EQ(tg_closing_text(st, closing, TG_CLOSING_MAX),
                  (int)strlen("\n</parameter>\n</function>"));
    ASSERT_STR_EQ(closing, "\n</parameter>\n</function>");
    tg_state_free(st);

    /* A required key still missing: no closer, the turn may not end. */
    st = tg_state_new_required(g, false);
    feed_forced(st, "<tool_call>\n<function=edit>\n<parameter=path>\na.c", &ok);
    ASSERT_TRUE(ok);
    ASSERT_FALSE(tg_end_allowed(st));
    feed_forced(st, "\n</parameter>\n", &ok);
    ASSERT_FALSE(tg_end_allowed(st));
    tg_state_free(st);
    tg_free(g);
    jr_free(tools);

    /* An enum value is closable only once it is a whole member. */
    tools = jr_parse(SWITCH_MODE_JSON);
    g = tg_build(tools);
    st = tg_state_new_required(g, false);
    feed_forced(st, "<tool_call>\n<function=switch_mode>\n<parameter=reason>\nr\n"
                    "</parameter>\n<parameter=mode>\npl", &ok);
    ASSERT_TRUE(ok);
    ASSERT_FALSE(tg_end_allowed(st));
    feed_forced(st, "an", &ok);
    ASSERT_TRUE(tg_end_allowed(st));
    ASSERT_INT_EQ(tg_closing_text(st, closing, TG_CLOSING_MAX),
                  (int)strlen("\n</parameter>\n</function>\n</tool_call>"));
    tg_advance(st, closing);
    ASSERT_TRUE(tg_complete(st));
    tg_state_free(st);
    tg_free(g);
    jr_free(tools);
}

static void test_first_char_prefilter_is_sound(void) {
    jr_value *tools = jr_parse(TOOLS_JSON);
    tg_grammar *g = tg_build(tools);
    static const char *TOKENS[] = {
        "<tool_call>", "<", "<function=", "read", ">\n", "<parameter=", "path", "</parameter>",
        "</function>", "\n", " ", "Sure", "</think>", "src/main.c", "{\"", "</tool_call>",
        "<function=glob>", "\u00e9", NULL };
    const char *script = "let me think</think>\n<tool_call>\n<function=read>\n"
                         "<parameter=path>\nsrc/main.c\n</parameter>\n</function>\n</tool_call>";
    char one[2] = { 0, 0 };
    int checked = 0, violations = 0;
    /* The legacy forced walk and the think-gated "required" walk. */
    for (int w = 0; w < 2; w++) {
        tg_state *st = w ? tg_state_new_required(g, true) : tg_state_new_forced(g, NULL);
        for (const char *p = script; ; p++) {
            for (int t = 0; TOKENS[t]; t++) {
                char two[3] = { TOKENS[t][0], TOKENS[t][0] ? TOKENS[t][1] : 0, 0 };
                one[0] = TOKENS[t][0];
                if (tg_token_allowed(st, TOKENS[t]) && (!tg_token_allowed(st, one) || !tg_token_allowed(st, two))) violations++;
                checked++;
            }
            if (!*p) break;
            one[0] = *p;
            tg_advance(st, one);
        }
        ASSERT_TRUE(tg_complete(st));
        tg_state_free(st);
    }
    ASSERT_INT_EQ(violations, 0);
    ASSERT_TRUE(checked > 2000);
    tg_free(g);
    jr_free(tools);
}

int main(void) {
    printf("\n=== tool_grammar (grammar-constrained tool decoding) ===\n\n");
    RUN_TEST(wrapped_accepts);
    RUN_TEST(first_char_prefilter_is_sound);
    RUN_TEST(bare_accepts);
    RUN_TEST(prose_then_call);
    RUN_TEST(pure_prose_no_call);
    RUN_TEST(json_form_rejected);
    RUN_TEST(unknown_function_rejected);
    RUN_TEST(unknown_param_rejected);
    RUN_TEST(cross_token_name);
    RUN_TEST(multi_param);
    RUN_TEST(forced_rejects_prose_and_json);
    RUN_TEST(forced_accepts_call);
    RUN_TEST(forced_reasoning_then_call);
    RUN_TEST(forced_function_pins_name);
    RUN_TEST(forced_self_detects_think_close);
    RUN_TEST(forced_think_close_split_tokens);
    RUN_TEST(forced_direct_call_no_think);
    RUN_TEST(auto_mode_unchanged);
    RUN_TEST(auto_never_masks_a_near_miss_opener);
    RUN_TEST(dead_walk_is_distinguishable_from_free);
    RUN_TEST(empty_value_closes_the_parameter);
    RUN_TEST(empty_value_off_grammar_when_floored);
    RUN_TEST(unfloored_value_may_still_be_empty);
    RUN_TEST(unpadded_close_is_grammatical);
    RUN_TEST(missing_required_arg_rejected);
    RUN_TEST(required_resets_between_calls);
    RUN_TEST(forced_early_arm_on_opener);
    RUN_TEST(forced_reasoning_budget);
    RUN_TEST(reason_budget_trips_mid_token_without_dying);
    RUN_TEST(landing_window_expires_mid_token_without_dying);
    RUN_TEST(dead_walk_does_not_report_reasoning_ended);
    RUN_TEST(landing_window_lets_the_model_finish);
    RUN_TEST(landing_window_recovers_the_real_capture);
    RUN_TEST(landing_window_ends_on_models_own_close);
    RUN_TEST(landing_window_closes_at_budget);
    RUN_TEST(landing_window_absent_on_self_close);
    RUN_TEST(landing_window_absent_in_auto_mode);
    RUN_TEST(landing_window_opener_still_engages);
    RUN_TEST(enum_closes_switch_mode);
    RUN_TEST(enum_on_optional_key);
    RUN_TEST(enum_degrades_when_unrepresentable);
    RUN_TEST(enum_over_cap_degrades);
    RUN_TEST(malformed_nested_argument_is_rejected);
    RUN_TEST(terminator_inside_a_json_string_is_off_grammar);
    RUN_TEST(string_parameters_stay_free_text);
    RUN_TEST(build_guards);
    RUN_TEST(required_think_gated_reasoning_is_free);
    RUN_TEST(required_without_think_compels_from_start);
    RUN_TEST(required_injected_close_lands);
    RUN_TEST(required_end_and_closing_text);
    TEST_REPORT();
    return __ft_failed == 0 ? 0 : 1;
}
