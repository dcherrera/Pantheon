/* SPDX-License-Identifier: AGPL-3.0-or-later */
/* Copyright (C) 2026 David (Corey) Herrera */
/*
 * Unit tests for the JSON tool-call decoding grammar
 * (src/serve/tool_grammar_json.c) — ornith's dialect.
 *
 * The grammar must ACCEPT ornith's real emission (a fenced OR bare JSON object
 * {"function":NAME,"arguments":{KEY:value,…}}, "name"/"parameters" spellings
 * too) and REJECT the off-contract drift: an unknown NAME/KEY, an empty value,
 * a missing required key, an empty {} when a key is required, and a value
 * outside a parameter's schema `enum`.  Free prose stays UNCONSTRAINED in AUTO;
 * FORCED compels a call.  This is the Phase-2 gate: the `required` guarantee for
 * ornith, structurally.
 *
 * Pure C, libc-only (json_reader for the tools DOM) — fully local, no CUDA.
 */

#include "lib/test.h"
#include "../src/serve/tool_grammar_json.h"
#include "../src/serve/json_reader.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

/* Two tools:
 *   read(path: string, minLength 1)       — path required, may not be empty
 *   add(a: integer, b: integer)           — both required
 *
 * `path` carries "minLength":1 because the client's real schema does
 * (the client's schema code) and because that keyword is now load-
 * bearing: it is the ONLY thing that separates `read`'s path (empty is the
 * empty-path wedge) from `edit`'s new_string (empty is the deletion).  A
 * fixture that omitted it would be asserting the grammar's behaviour for a
 * schema the client never sends.  See EMPTY_OK_TOOLS_JSON for the other side. */
static const char *TOOLS_JSON =
    "[{\"type\":\"function\",\"function\":{\"name\":\"read\","
    "\"parameters\":{\"type\":\"object\",\"properties\":{"
    "\"path\":{\"type\":\"string\",\"minLength\":1}},\"required\":[\"path\"]}}},"
    "{\"type\":\"function\",\"function\":{\"name\":\"add\","
    "\"parameters\":{\"type\":\"object\",\"properties\":{"
    "\"a\":{\"type\":\"integer\"},\"b\":{\"type\":\"integer\"}},"
    "\"required\":[\"a\",\"b\"]}}}]";

/* The client's REAL `edit` shape (the client's schema code): all three
 * parameters are `required`, `path` and `old_string` carry "minLength":1, and
 * `new_string` deliberately does NOT — because an empty new_string is what a
 * deletion IS.  Requiredness is therefore not the empty-value predicate, and
 * this fixture is what proves the grammar does not treat it as one. */
static const char *EMPTY_OK_TOOLS_JSON =
    "[{\"type\":\"function\",\"function\":{\"name\":\"edit\","
    "\"parameters\":{\"type\":\"object\",\"properties\":{"
    "\"path\":{\"type\":\"string\",\"minLength\":1},"
    "\"old_string\":{\"type\":\"string\",\"minLength\":1},"
    "\"new_string\":{\"type\":\"string\"}},"
    "\"required\":[\"path\",\"old_string\",\"new_string\"]}}}]";

/* The client's REAL switch_mode schema (the client's schema code) — the
 * live victim of the dropped-`enum` gap: before the enum was compiled, this
 * grammar accepted "mode":"banana" as a complete, well-formed call. */
static const char *SWITCH_MODE_JSON =
    "[{\"type\":\"function\",\"function\":{\"name\":\"switch_mode\","
    "\"parameters\":{\"type\":\"object\",\"properties\":"
    "{\"mode\":{\"type\":\"string\","
    "\"enum\":[\"chat\",\"plan\",\"act\",\"orchestrate\"],"
    "\"description\":\"the mode to switch to\"},"
    "\"reason\":{\"type\":\"string\",\"description\":\"one short line\"}},"
    "\"required\":[\"mode\",\"reason\"]}}}]";

/* log(msg: string REQUIRED, level: string enum OPTIONAL) — an enum on a key the
 * model may legally leave out entirely. */
static const char *ENUM_OPTIONAL_JSON =
    "[{\"type\":\"function\",\"function\":{\"name\":\"log\","
    "\"parameters\":{\"type\":\"object\",\"properties\":{"
    "\"msg\":{\"type\":\"string\"},"
    "\"level\":{\"type\":\"string\",\"enum\":[\"info\",\"warn\",\"error\"]}},"
    "\"required\":[\"msg\"]}}}]";

/* Every enum shape that must DEGRADE to the plain type rather than reject:
 *   nums    — non-string members
 *   mixed   — one non-string member among strings
 *   blank   — an empty member (unsatisfiable if compiled)
 *   nolist  — an empty enum array
 *   clash   — string members on an integer-typed key (schema contradiction)
 *   notype  — string members with NO declared type: this one DOES compile. */
static const char *ENUM_DEGRADE_JSON =
    "[{\"type\":\"function\",\"function\":{\"name\":\"cfg\","
    "\"parameters\":{\"type\":\"object\",\"properties\":{"
    "\"nums\":{\"type\":\"integer\",\"enum\":[1,2,3]},"
    "\"mixed\":{\"type\":\"string\",\"enum\":[\"a\",7]},"
    "\"blank\":{\"type\":\"string\",\"enum\":[\"a\",\"\"]},"
    "\"nolist\":{\"type\":\"string\",\"enum\":[]},"
    "\"clash\":{\"type\":\"integer\",\"enum\":[\"a\",\"b\"]},"
    "\"notype\":{\"enum\":[\"yes\",\"no\"]}},"
    "\"required\":[\"nums\"]}}}]";

/* get_weather(city: string, unit: string) — only city required (unit optional). */
static const char *WEATHER_JSON =
    "[{\"type\":\"function\",\"function\":{\"name\":\"get_weather\","
    "\"parameters\":{\"type\":\"object\",\"properties\":{"
    "\"city\":{\"type\":\"string\"},\"unit\":{\"type\":\"string\"}},"
    "\"required\":[\"city\"]}}}]";

/* Feed a whole string char-by-char through an AUTO state; each char must stay
 * allowed.  Returns the final state (caller frees). */
static tgj_state *feed_ok(tgj_grammar *g, const char *s, bool *all_allowed) {
    tgj_state *st = tgj_state_new(g);
    char one[2] = {0, 0};
    *all_allowed = true;
    for (const char *p = s; *p; p++) {
        one[0] = *p;
        if (!tgj_token_allowed(st, one)) { *all_allowed = false; break; }
        tgj_advance(st, one);
    }
    return st;
}

static bool feed_rejects(tgj_grammar *g, const char *s) {
    bool ok;
    tgj_state *st = feed_ok(g, s, &ok);
    tgj_state_free(st);
    return !ok;
}

/* Feed into an already-created state (forced/other). */
static void feed_into(tgj_state *st, const char *s, bool *all_allowed) {
    char one[2] = {0, 0};
    *all_allowed = true;
    for (const char *p = s; *p; p++) {
        one[0] = *p;
        if (!tgj_token_allowed(st, one)) { *all_allowed = false; break; }
        tgj_advance(st, one);
    }
}

static bool forced_rejects(tgj_grammar *g, const char *s) {
    tgj_state *st = tgj_state_new_forced(g, NULL);
    tgj_reasoning_end(st);
    bool ok;
    feed_into(st, s, &ok);
    tgj_state_free(st);
    return !ok;
}

/* ============================================================
 * ACCEPT: fenced single call.
 * ============================================================ */
static void test_fenced_accepts(void) {
    jr_value *tools = jr_parse(TOOLS_JSON);
    tgj_grammar *g = tgj_build(tools);
    ASSERT_NOT_NULL(g);

    const char *call =
        "```json\n{\"function\": \"read\", \"arguments\": "
        "{\"path\": \"/etc/hostname\"}}\n```";
    bool ok;
    tgj_state *st = feed_ok(g, call, &ok);
    ASSERT_TRUE(ok);
    ASSERT_TRUE(tgj_complete(st));

    tgj_state_free(st);
    tgj_free(g);
    jr_free(tools);
}

/* ============================================================
 * ACCEPT: bare (unfenced) call.
 * ============================================================ */
static void test_bare_accepts(void) {
    jr_value *tools = jr_parse(TOOLS_JSON);
    tgj_grammar *g = tgj_build(tools);

    const char *call =
        "{\"function\":\"read\",\"arguments\":{\"path\":\"README.md\"}}";
    bool ok;
    tgj_state *st = feed_ok(g, call, &ok);
    ASSERT_TRUE(ok);
    ASSERT_TRUE(tgj_complete(st));

    tgj_state_free(st);
    tgj_free(g);
    jr_free(tools);
}

/* ============================================================
 * ACCEPT: the "name"/"parameters" key spellings.
 * ============================================================ */
static void test_name_and_parameters_keys(void) {
    jr_value *tools = jr_parse(TOOLS_JSON);
    tgj_grammar *g = tgj_build(tools);

    const char *call =
        "{\"name\":\"read\",\"parameters\":{\"path\":\"x\"}}";
    bool ok;
    tgj_state *st = feed_ok(g, call, &ok);
    ASSERT_TRUE(ok);
    ASSERT_TRUE(tgj_complete(st));

    tgj_state_free(st);
    tgj_free(g);
    jr_free(tools);
}

/* ============================================================
 * ACCEPT: integer-typed args (numeric literals), both required present.
 * ============================================================ */
static void test_integer_args(void) {
    jr_value *tools = jr_parse(TOOLS_JSON);
    tgj_grammar *g = tgj_build(tools);

    const char *call =
        "{\"function\":\"add\",\"arguments\":{\"a\": 5, \"b\": 37}}";
    bool ok;
    tgj_state *st = feed_ok(g, call, &ok);
    ASSERT_TRUE(ok);
    ASSERT_TRUE(tgj_complete(st));

    tgj_state_free(st);
    tgj_free(g);
    jr_free(tools);
}

/* An integer param cannot take a quoted string value. */
static void test_integer_rejects_string(void) {
    jr_value *tools = jr_parse(TOOLS_JSON);
    tgj_grammar *g = tgj_build(tools);
    ASSERT_TRUE(feed_rejects(g,
        "{\"function\":\"add\",\"arguments\":{\"a\": \"5\""));
    tgj_free(g);
    jr_free(tools);
}

/* ============================================================
 * REJECT: unknown function name / unknown parameter key.
 * ============================================================ */
static void test_unknown_name_rejected(void) {
    jr_value *tools = jr_parse(TOOLS_JSON);
    tgj_grammar *g = tgj_build(tools);
    /* "reads" diverges from "read" at the trailing 's' before the closing quote
     * (read is a valid prefix but the exact-match on '"' fails). */
    ASSERT_TRUE(feed_rejects(g, "{\"function\":\"reads\""));
    ASSERT_TRUE(feed_rejects(g, "{\"function\":\"glob\""));
    tgj_free(g);
    jr_free(tools);
}

static void test_unknown_key_rejected(void) {
    jr_value *tools = jr_parse(TOOLS_JSON);
    tgj_grammar *g = tgj_build(tools);
    /* read has only "path"; "mode" is not a key. */
    ASSERT_TRUE(feed_rejects(g,
        "{\"function\":\"read\",\"arguments\":{\"mode\""));
    tgj_free(g);
    jr_free(tools);
}

/* ============================================================
 * REJECT (structurally): empty value for a required string.
 * ============================================================ */
static void test_empty_value_impossible(void) {
    jr_value *tools = jr_parse(TOOLS_JSON);
    tgj_grammar *g = tgj_build(tools);
    /* `"path": ""` — the closing quote after zero content chars is DEAD, because
     * read.path declares "minLength":1. */
    ASSERT_TRUE(feed_rejects(g,
        "{\"function\":\"read\",\"arguments\":{\"path\": \"\""));
    tgj_free(g);
    jr_free(tools);
}

/* ============================================================
 * ACCEPT: an empty value for a string with NO length floor.
 *
 * The mirror of the test above, and the reason the floor is per-key rather than
 * blanket.  `edit`'s new_string is `required` AND legitimately empty — a
 * deletion — so a guard keyed on requiredness, or an unconditional one (which is
 * what this dialect shipped for a while), makes a LEGAL call unemittable.  Under
 * tool_choice=required that is its own wedge: the model must emit a call, and
 * the only correct one is masked.
 * ============================================================ */
static void test_empty_value_allowed_without_a_floor(void) {
    jr_value *tools = jr_parse(EMPTY_OK_TOOLS_JSON);
    tgj_grammar *g = tgj_build(tools);
    bool ok;
    tgj_state *st = feed_ok(g,
        "{\"function\":\"edit\",\"arguments\":{\"path\":\"/tmp/x.c\","
        "\"old_string\":\"drop me\\n\",\"new_string\":\"\"}}", &ok);
    ASSERT_TRUE(ok);
    ASSERT_TRUE(tgj_complete(st));
    tgj_state_free(st);

    /* CONTROL: the floored siblings in the SAME tool are still refused, so this
     * is a per-key verdict and not the guard having been switched off. */
    ASSERT_TRUE(feed_rejects(g,
        "{\"function\":\"edit\",\"arguments\":{\"path\":\"\""));
    ASSERT_TRUE(feed_rejects(g,
        "{\"function\":\"edit\",\"arguments\":{\"path\":\"/tmp/x.c\","
        "\"old_string\":\"\""));
    tgj_free(g);
    jr_free(tools);
}

/* ============================================================
 * REJECT: a WHITESPACE-ONLY value for a required string.
 *
 * The guard above used to key on "any in-quote char", so `" "` and `"\n"` passed
 * it while carrying nothing.  Unlike the XML dialect the ornith parser does not
 * trim, so this forwarded `{"path":" "}` verbatim — a different shape from the
 * XML path's fabricated `{"path":""}`, but the same wedge once it reached the
 * tool (the client's require_string() skips whitespace before testing for empty).
 *
 * The escape cases are the second half of the hole and survive fixing only the
 * plain-char site: the escape pair used to set val_content without decoding it.
 * ============================================================ */
static void test_whitespace_only_value_impossible(void) {
    jr_value *tools = jr_parse(TOOLS_JSON);
    tgj_grammar *g = tgj_build(tools);
    const char *head = "{\"function\":\"read\",\"arguments\":{\"path\": ";

    char buf[256];
    const char *ws_values[] = {
        "\" \"",        /* one space */
        "\"  \\t \"",   /* mixed run */
        "\"\\n\"",      /* escaped newline — the escape-pair hole */
        "\"\\t\"",      /* escaped tab */
        "\"\\r\"",      /* escaped CR */
        "\"\\f\"",      /* escaped form feed */
    };
    for (size_t i = 0; i < sizeof(ws_values) / sizeof(ws_values[0]); i++) {
        snprintf(buf, sizeof(buf), "%s%s", head, ws_values[i]);
        ASSERT_TRUE(feed_rejects(g, buf));
    }

    /* CONTROL: real content still accepted, including with whitespace padding
     * and an escape that is NOT whitespace (\\b, \\", \\\\). */
    snprintf(buf, sizeof(buf), "%s\"  /tmp/x.c  \"", head);
    ASSERT_FALSE(feed_rejects(g, buf));
    snprintf(buf, sizeof(buf), "%s\"\\b\"", head);
    ASSERT_FALSE(feed_rejects(g, buf));
    snprintf(buf, sizeof(buf), "%s\"a\\nb\"", head);
    ASSERT_FALSE(feed_rejects(g, buf));

    tgj_free(g);
    jr_free(tools);
}

/* ============================================================
 * REJECT (structurally): empty {} arguments when a key is required — the outer
 * object cannot close until the required key is present.
 * ============================================================ */
static void test_empty_args_impossible_when_required(void) {
    jr_value *tools = jr_parse(TOOLS_JSON);
    tgj_grammar *g = tgj_build(tools);
    bool ok;
    /* "arguments":{} then attempt to close — the '}' that would close the arg
     * object is rejected because required_left > 0 (path outstanding). */
    tgj_state *st = feed_ok(g,
        "{\"function\":\"read\",\"arguments\":{}", &ok);
    ASSERT_FALSE(ok);   /* the inner '}' closing an empty args object is DEAD */
    tgj_state_free(st);

    /* The add tool needs BOTH a and b: closing after only a is impossible. */
    ASSERT_TRUE(feed_rejects(g,
        "{\"function\":\"add\",\"arguments\":{\"a\": 5}"));
    tgj_free(g);
    jr_free(tools);
}

/* An OPTIONAL-only extra can be omitted: get_weather with just city closes. */
static void test_optional_key_omittable(void) {
    jr_value *tools = jr_parse(WEATHER_JSON);
    tgj_grammar *g = tgj_build(tools);
    const char *call =
        "{\"function\":\"get_weather\",\"arguments\":{\"city\":\"Paris\"}}";
    bool ok;
    tgj_state *st = feed_ok(g, call, &ok);
    ASSERT_TRUE(ok);
    ASSERT_TRUE(tgj_complete(st));
    tgj_state_free(st);
    tgj_free(g);
    jr_free(tools);
}

/* ============================================================
 * ACCEPT: a key split across token boundaries still matches.
 * ============================================================ */
static void test_cross_token_key(void) {
    jr_value *tools = jr_parse(WEATHER_JSON);
    tgj_grammar *g = tgj_build(tools);

    tgj_state *st = tgj_state_new(g);
    ASSERT_TRUE(tgj_token_allowed(st, "{\"function\":\"get_"));
    tgj_advance(st, "{\"function\":\"get_");
    ASSERT_TRUE(tgj_token_allowed(st, "weather\","));
    tgj_advance(st, "weather\",");
    ASSERT_TRUE(tgj_token_allowed(st, "\"arguments\":{\"ci"));
    tgj_advance(st, "\"arguments\":{\"ci");
    ASSERT_TRUE(tgj_token_allowed(st, "ty\":\"NY\"}}"));
    tgj_advance(st, "ty\":\"NY\"}}");
    ASSERT_TRUE(tgj_complete(st));
    tgj_state_free(st);

    /* A key split that diverges is still rejected. */
    tgj_state *bad = tgj_state_new(g);
    tgj_advance(bad, "{\"function\":\"get_weather\",\"arguments\":{\"ci");
    ASSERT_FALSE(tgj_token_allowed(bad, "x\""));   /* "cix" not a key */
    tgj_state_free(bad);

    tgj_free(g);
    jr_free(tools);
}

/* ============================================================
 * AUTO: free prose before the call is unconstrained; pure prose never forced.
 * ============================================================ */
static void test_prose_then_call_auto(void) {
    jr_value *tools = jr_parse(TOOLS_JSON);
    tgj_grammar *g = tgj_build(tools);

    const char *txt =
        "Let me read that file for you.\n"
        "```json\n{\"function\":\"read\",\"arguments\":{\"path\":\"a.txt\"}}\n```";
    bool ok;
    tgj_state *st = feed_ok(g, txt, &ok);
    ASSERT_TRUE(ok);
    ASSERT_TRUE(tgj_complete(st));
    tgj_state_free(st);

    /* Pure prose, no call → never forced, never in a call. */
    tgj_state *p = feed_ok(g, "The answer is 42, no tools needed.", &ok);
    ASSERT_TRUE(ok);
    ASSERT_FALSE(tgj_in_call(p));
    ASSERT_FALSE(tgj_complete(p));
    ASSERT_FALSE(tgj_mask_active(p));   /* AUTO: FREE prose unmasked */
    tgj_state_free(p);

    tgj_free(g);
    jr_free(tools);
}

/* ============================================================
 * FORCED: after reasoning, prose is rejected; a valid call is compelled.
 * ============================================================ */
static void test_forced_rejects_prose(void) {
    jr_value *tools = jr_parse(TOOLS_JSON);
    tgj_grammar *g = tgj_build(tools);

    ASSERT_TRUE(forced_rejects(g, "I cannot do that."));
    ASSERT_TRUE(forced_rejects(g, "Sure! {\"function\":\"read\"}"));

    /* A valid call from the start (reasoning ended) is accepted. */
    tgj_state *st = tgj_state_new_forced(g, NULL);
    tgj_reasoning_end(st);
    bool ok;
    feed_into(st,
        "```json\n{\"function\":\"read\",\"arguments\":{\"path\":\"x\"}}\n```",
        &ok);
    ASSERT_TRUE(ok);
    ASSERT_TRUE(tgj_complete(st));
    tgj_state_free(st);

    tgj_free(g);
    jr_free(tools);
}

/* FORCED: reasoning is allowed first, then "</think>" self-fires the constraint. */
static void test_forced_self_detects_think_close(void) {
    jr_value *tools = jr_parse(TOOLS_JSON);
    tgj_grammar *g = tgj_build(tools);

    tgj_state *st = tgj_state_new_forced(g, NULL);
    bool ok;
    feed_into(st, "I should read the file.", &ok);
    ASSERT_TRUE(ok);
    ASSERT_FALSE(tgj_mask_active(st));   /* grace */
    feed_into(st, "</think>", &ok);
    ASSERT_TRUE(ok);
    ASSERT_TRUE(tgj_mask_active(st));    /* self-detected → opener compelled */
    ASSERT_FALSE(tgj_token_allowed(st, "Sorry, I can't."));
    ASSERT_TRUE(tgj_token_allowed(st, "```json\n{"));
    feed_into(st,
        "```json\n{\"function\":\"read\",\"arguments\":{\"path\":\"a\"}}\n```",
        &ok);
    ASSERT_TRUE(ok);
    ASSERT_TRUE(tgj_complete(st));
    tgj_state_free(st);

    tgj_free(g);
    jr_free(tools);
}

/* FORCED-FUNCTION pins NAME: the target is accepted; another tool is rejected. */
static void test_forced_function_pins_name(void) {
    jr_value *tools = jr_parse(TOOLS_JSON);
    tgj_grammar *g = tgj_build(tools);

    tgj_state *good = tgj_state_new_forced(g, "read");
    ASSERT_NOT_NULL(good);
    tgj_reasoning_end(good);
    bool ok;
    feed_into(good, "{\"function\":\"read\",\"arguments\":{\"path\":\"x\"}}", &ok);
    ASSERT_TRUE(ok);
    ASSERT_TRUE(tgj_complete(good));
    tgj_state_free(good);

    tgj_state *bad = tgj_state_new_forced(g, "read");
    tgj_reasoning_end(bad);
    feed_into(bad, "{\"function\":\"add\"", &ok);
    ASSERT_FALSE(ok);   /* pinned to read → "add" rejected */
    tgj_state_free(bad);

    ASSERT_NULL(tgj_state_new_forced(g, "not_a_tool"));

    tgj_free(g);
    jr_free(tools);
}

/* ============================================================
 * 0-BUDGET FORCED ARMING (the ornith reasoning_budget:0 edge).  A forced state
 * with reasoning ended IMMEDIATELY (no <think> grace at all, mirroring a
 * 0-budget model that opens straight into the answer channel) must arm the
 * opener from the very first token — prose is impossible, the fence/`{` opener
 * survives.  This exercises the path the N2 (budget 2000) grammar never did.
 * ============================================================ */
static void test_zero_budget_forced_arming(void) {
    jr_value *tools = jr_parse(TOOLS_JSON);
    tgj_grammar *g = tgj_build(tools);

    /* Reasoning ends before any token — the 0-budget shape: the very first
     * content token is already under the forced constraint. */
    tgj_state *st = tgj_state_new_forced(g, NULL);
    tgj_reasoning_end(st);
    ASSERT_TRUE(tgj_mask_active(st));     /* armed with zero grace */
    ASSERT_FALSE(tgj_token_allowed(st, "Here is the fix:"));
    ASSERT_TRUE(tgj_token_allowed(st, "```json"));
    ASSERT_TRUE(tgj_token_allowed(st, "{\"function\":\"read\""));
    /* And it drives a full call to completion. */
    bool ok;
    feed_into(st,
        "{\"function\":\"read\",\"arguments\":{\"path\":\"/etc/hostname\"}}", &ok);
    ASSERT_TRUE(ok);
    ASSERT_TRUE(tgj_complete(st));
    tgj_state_free(st);

    /* Also verify the budget backstop trips for a runaway 0-grace-expecting model
     * that never emits </think>: a long prose ramble must arm the mask in bounded
     * time (not stream unconstrained forever). */
    tgj_state *st2 = tgj_state_new_forced(g, NULL);
    char one[2] = {'a', 0};
    bool armed = false;
    for (int i = 0; i < 12000; i++) {
        one[0] = "abcdefgh"[i & 7];
        tgj_advance(st2, one);
        if (tgj_mask_active(st2)) { armed = true; break; }
    }
    ASSERT_TRUE(armed);
    ASSERT_TRUE(tgj_mask_active(st2));
    ASSERT_TRUE(tgj_token_allowed(st2, "```json\n{"));
    ASSERT_FALSE(tgj_token_allowed(st2, "still rambling"));
    tgj_state_free(st2);

    tgj_free(g);
    jr_free(tools);
}

/* ============================================================
 * AUTO mode unchanged by forced work: mask only inside a call.
 * ============================================================ */
static void test_auto_mode_masking(void) {
    jr_value *tools = jr_parse(TOOLS_JSON);
    tgj_grammar *g = tgj_build(tools);

    tgj_state *st = tgj_state_new(g);
    tgj_reasoning_end(st);   /* no-op in AUTO */
    bool ok;
    feed_into(st, "no tools needed", &ok);
    ASSERT_TRUE(ok);
    ASSERT_FALSE(tgj_mask_active(st));
    ASSERT_FALSE(tgj_in_call(st));
    /* Once the object opens, the mask arms. */
    tgj_advance(st, "{\"function\":\"read");
    ASSERT_TRUE(tgj_in_call(st));
    ASSERT_TRUE(tgj_mask_active(st));
    tgj_state_free(st);

    tgj_free(g);
    jr_free(tools);
}

/* ============================================================
 * Malformed / truncated → never a complete call.
 * ============================================================ */
static void test_truncated_incomplete(void) {
    jr_value *tools = jr_parse(TOOLS_JSON);
    tgj_grammar *g = tgj_build(tools);
    bool ok;
    /* Missing the closing braces → not complete. */
    tgj_state *st = feed_ok(g,
        "{\"function\":\"read\",\"arguments\":{\"path\":\"x\"", &ok);
    ASSERT_TRUE(ok);                 /* valid prefix so far */
    ASSERT_FALSE(tgj_complete(st));  /* but not a complete call */
    tgj_state_free(st);
    tgj_free(g);
    jr_free(tools);
}

/* ============================================================
 * ENUM — the live switch_mode victim.  Every declared mode is still emittable
 * and completes; anything off the list is structurally impossible.  BOTH halves
 * matter: a grammar that rejected every value would pass the reject half alone.
 * ============================================================ */
static void test_enum_closes_switch_mode(void) {
    jr_value *tools = jr_parse(SWITCH_MODE_JSON);
    tgj_grammar *g = tgj_build(tools);
    ASSERT_NOT_NULL(g);

    static const char *const MODES[] = { "chat", "plan", "act", "orchestrate" };
    for (int i = 0; i < 4; i++) {
        char call[256];
        snprintf(call, sizeof(call),
                 "{\"function\":\"switch_mode\",\"arguments\":"
                 "{\"mode\":\"%s\",\"reason\":\"the job needs it\"}}", MODES[i]);
        bool ok;
        tgj_state *st = feed_ok(g, call, &ok);
        ASSERT_TRUE(ok);                 /* every legal mode still accepted */
        ASSERT_TRUE(tgj_complete(st));
        tgj_state_free(st);
    }

    /* The bug this closes: "banana" was a complete, well-formed switch_mode. */
    ASSERT_TRUE(feed_rejects(g,
        "{\"function\":\"switch_mode\",\"arguments\":{\"mode\":\"banana\""));
    /* An off-set value that shares a prefix dies where it diverges. */
    ASSERT_TRUE(feed_rejects(g,
        "{\"function\":\"switch_mode\",\"arguments\":{\"mode\":\"plans\""));
    /* A strict PREFIX of a member is not a member: the closing quote is DEAD. */
    ASSERT_TRUE(feed_rejects(g,
        "{\"function\":\"switch_mode\",\"arguments\":{\"mode\":\"pla\""));
    /* And empty stays impossible. */
    ASSERT_TRUE(feed_rejects(g,
        "{\"function\":\"switch_mode\",\"arguments\":{\"mode\":\"\""));

    tgj_free(g);
    jr_free(tools);
}

/* An enum member split across token boundaries still matches (the state carries
 * the accumulator), and a split that diverges is still rejected. */
static void test_enum_cross_token(void) {
    jr_value *tools = jr_parse(SWITCH_MODE_JSON);
    tgj_grammar *g = tgj_build(tools);

    tgj_state *st = tgj_state_new(g);
    tgj_advance(st, "{\"function\":\"switch_mode\",\"arguments\":{\"mode\":\"orch");
    ASSERT_TRUE(tgj_token_allowed(st, "estrate\""));
    tgj_advance(st, "estrate\",\"reason\":\"x\"}}");
    ASSERT_TRUE(tgj_complete(st));
    tgj_state_free(st);

    tgj_state *bad = tgj_state_new(g);
    tgj_advance(bad, "{\"function\":\"switch_mode\",\"arguments\":{\"mode\":\"orch");
    ASSERT_FALSE(tgj_token_allowed(bad, "id\""));   /* "orchid" is not a mode */
    tgj_state_free(bad);

    tgj_free(g);
    jr_free(tools);
}

/* An enum on an OPTIONAL key constrains the value WITHOUT making the key
 * mandatory: omitting it entirely still completes. */
static void test_enum_on_optional_key(void) {
    jr_value *tools = jr_parse(ENUM_OPTIONAL_JSON);
    tgj_grammar *g = tgj_build(tools);
    bool ok;

    tgj_state *a = feed_ok(g, "{\"function\":\"log\",\"arguments\":{\"msg\":\"hi\"}}", &ok);
    ASSERT_TRUE(ok);                 /* level omitted → still a complete call */
    ASSERT_TRUE(tgj_complete(a));
    tgj_state_free(a);

    tgj_state *b = feed_ok(g,
        "{\"function\":\"log\",\"arguments\":{\"msg\":\"hi\",\"level\":\"warn\"}}", &ok);
    ASSERT_TRUE(ok);
    ASSERT_TRUE(tgj_complete(b));
    tgj_state_free(b);

    /* Present but off-set is still impossible. */
    ASSERT_TRUE(feed_rejects(g,
        "{\"function\":\"log\",\"arguments\":{\"msg\":\"hi\",\"level\":\"loud\""));

    tgj_free(g);
    jr_free(tools);
}

/* Enums we cannot represent DEGRADE to the declared type — they never reject a
 * legal call (the TG_REQ_MAX_KEYS precedent). */
static void test_enum_degrades_when_unrepresentable(void) {
    jr_value *tools = jr_parse(ENUM_DEGRADE_JSON);
    tgj_grammar *g = tgj_build(tools);
    ASSERT_NOT_NULL(g);
    bool ok;

    /* nums: integer type + numeric members → dropped, still a plain integer, so
     * a value outside the list is accepted (unconstrained, not rejected). */
    tgj_state *a = feed_ok(g, "{\"function\":\"cfg\",\"arguments\":{\"nums\":99}}", &ok);
    ASSERT_TRUE(ok);
    ASSERT_TRUE(tgj_complete(a));
    tgj_state_free(a);

    /* mixed / blank / nolist: dropped → plain strings, any non-empty text ok. */
    tgj_state *b = feed_ok(g,
        "{\"function\":\"cfg\",\"arguments\":{\"nums\":1,\"mixed\":\"zzz\","
        "\"blank\":\"zzz\",\"nolist\":\"zzz\"}}", &ok);
    ASSERT_TRUE(ok);
    ASSERT_TRUE(tgj_complete(b));
    tgj_state_free(b);

    /* clash: string members on an integer key → the declared type wins, so the
     * value is a number and the enum's own members are NOT accepted. */
    ASSERT_TRUE(feed_rejects(g,
        "{\"function\":\"cfg\",\"arguments\":{\"nums\":1,\"clash\":\"a\""));
    tgj_state *c = feed_ok(g,
        "{\"function\":\"cfg\",\"arguments\":{\"nums\":1,\"clash\":42}}", &ok);
    ASSERT_TRUE(ok);
    ASSERT_TRUE(tgj_complete(c));
    tgj_state_free(c);

    /* notype: string members and NO declared type → the enum DOES apply. */
    tgj_state *d = feed_ok(g,
        "{\"function\":\"cfg\",\"arguments\":{\"nums\":1,\"notype\":\"yes\"}}", &ok);
    ASSERT_TRUE(ok);
    ASSERT_TRUE(tgj_complete(d));
    tgj_state_free(d);
    ASSERT_TRUE(feed_rejects(g,
        "{\"function\":\"cfg\",\"arguments\":{\"nums\":1,\"notype\":\"maybe\""));

    tgj_free(g);
    jr_free(tools);
}

/* An enum larger than the cap degrades to unconstrained rather than rejecting. */
static void test_enum_over_cap_degrades(void) {
    char schema[8192];
    int n = snprintf(schema, sizeof(schema),
                     "[{\"type\":\"function\",\"function\":{\"name\":\"pick\","
                     "\"parameters\":{\"type\":\"object\",\"properties\":{"
                     "\"v\":{\"type\":\"string\",\"enum\":[");
    for (int i = 0; i < 65; i++)                       /* 65 > TGJ_ENUM_MAX_MEMBERS */
        n += snprintf(schema + n, sizeof(schema) - (size_t)n,
                      "%s\"m%03d\"", i ? "," : "", i);
    snprintf(schema + n, sizeof(schema) - (size_t)n,
             "]}},\"required\":[\"v\"]}}}]");

    jr_value *tools = jr_parse(schema);
    tgj_grammar *g = tgj_build(tools);
    ASSERT_NOT_NULL(g);

    bool ok;
    /* Off-list value accepted → the enum was dropped, not enforced. */
    tgj_state *st = feed_ok(g,
        "{\"function\":\"pick\",\"arguments\":{\"v\":\"anything\"}}", &ok);
    ASSERT_TRUE(ok);
    ASSERT_TRUE(tgj_complete(st));
    tgj_state_free(st);

    tgj_free(g);
    jr_free(tools);
}

/* ============================================================
 * Build guards: empty / non-array tools → NULL grammar.
 * ============================================================ */
static void test_build_guards(void) {
    jr_value *empty = jr_parse("[]");
    ASSERT_NULL(tgj_build(empty));
    jr_free(empty);
    jr_value *obj = jr_parse("{\"x\":1}");
    ASSERT_NULL(tgj_build(obj));
    jr_free(obj);
}


/* ============================================================
 * The compelled region must start at a TOKEN boundary.
 *
 * The XML dialect's twin (test_tool_grammar.c
 * reason_budget_trips_mid_token_without_dying) and the same live failure:
 * tgj_advance walks a WHOLE token, so a reasoning budget that trips mid-token
 * would judge that token's remaining chars under the compelled rule with no
 * mask in between — and one prose char there is TGJ_DEAD, which reports the
 * mask INACTIVE for every remaining token of the turn.  Every other test in this
 * file feeds one char at a time, the one granularity where it cannot happen.
 * ============================================================ */
static void test_reason_budget_trips_mid_token_without_dying(void) {
    jr_value *tools = jr_parse(TOOLS_JSON);
    tgj_grammar *g = tgj_build(tools);
    tgj_state *st = tgj_state_new_forced(g, NULL);

    const char *piece = " thinking";   /* multi-char, prose, no opener, no close */
    bool armed = false;
    for (int i = 0; i < 4096 && !armed; i++) {
        tgj_advance(st, piece);
        armed = tgj_mask_active(st);
        if (tgj_is_dead(st)) break;
    }
    ASSERT_FALSE(tgj_is_dead(st));
    ASSERT_TRUE(armed);
    ASSERT_TRUE(tgj_mask_active(st));
    ASSERT_TRUE(tgj_token_allowed(st, "{"));
    ASSERT_FALSE(tgj_token_allowed(st, " still rambling"));

    tgj_state_free(st);
    tgj_free(g);
    jr_free(tools);
}

/* A DEAD walk must not tell the serve that reasoning is over — that bit flips
 * the emitter's channel, and a dead walk is compelling nothing, so the flip
 * routes the model's ongoing thought into the content channel. */
static void test_dead_walk_does_not_report_reasoning_ended(void) {
    jr_value *tools = jr_parse(TOOLS_JSON);
    tgj_grammar *g = tgj_build(tools);
    tgj_state *st = tgj_state_new_forced(g, NULL);

    tgj_reasoning_end(st);
    ASSERT_TRUE(tgj_forced_reasoning_ended(st));
    tgj_advance(st, "prose");
    ASSERT_TRUE(tgj_is_dead(st));
    ASSERT_FALSE(tgj_mask_active(st));
    ASSERT_FALSE(tgj_forced_reasoning_ended(st));

    tgj_state_free(st);
    tgj_free(g);
    jr_free(tools);
}

/* ============================================================
 * tool_choice "required" with thinking ON (tgj_state_new_required, think_open):
 * the think block is NEVER constrained — a '{' or a ```json fence written while
 * thinking is thought, not a call, and a think that never closes never arms the
 * mask (max_tokens bounds it) — and from the token after "</think>" the call is
 * compelled.
 * ============================================================ */
static void test_required_think_gated_reasoning_is_free(void) {
    jr_value *tools = jr_parse(TOOLS_JSON);
    tgj_grammar *g = tgj_build(tools);
    tgj_state *st = tgj_state_new_required(g, true);
    ASSERT_NOT_NULL(st);
    ASSERT_FALSE(tgj_mask_active(st));

    const char *thought =
        "The user wants the file. I could call {\"function\":\"read\"} or emit "
        "```json\n{ ... } — let me think about { and } more.";
    bool ok;
    char one[2] = {0, 0};
    for (const char *p = thought; *p; p++) {
        one[0] = *p;
        ASSERT_TRUE(tgj_token_allowed(st, one));
        tgj_advance(st, one);
        ASSERT_FALSE(tgj_mask_active(st));
    }
    ASSERT_FALSE(tgj_in_call(st));

    /* A think that never closes: far past the legacy 8192-char budget, the
     * mask stays off — reasoning is bounded by max_tokens, not by the grammar. */
    for (int i = 0; i < 30000; i++) {
        one[0] = "abc {}`\n"[i % 8];
        tgj_advance(st, one);
        if (tgj_mask_active(st)) break;
    }
    ASSERT_FALSE(tgj_mask_active(st));
    ASSERT_FALSE(tgj_forced_reasoning_ended(st));

    /* The model closes its think block: from the next token the call is due. */
    feed_into(st, "</think>", &ok);
    ASSERT_TRUE(ok);
    ASSERT_TRUE(tgj_mask_active(st));
    ASSERT_TRUE(tgj_forced_reasoning_ended(st));
    ASSERT_FALSE(tgj_token_allowed(st, "Let me see the rest of the file."));
    ASSERT_FALSE(tgj_token_allowed(st, "{\"function\":\"write\""));
    feed_into(st, "\n\n```json\n{\"function\": \"read\", \"arguments\": "
                  "{\"path\": \"src/main.c\"}}\n```", &ok);
    ASSERT_TRUE(ok);
    ASSERT_TRUE(tgj_complete(st));
    ASSERT_FALSE(tgj_mask_active(st));
    tgj_state_free(st);

    tgj_free(g);
    jr_free(tools);
}

/* "required" with NO open think block (thinking off): compelled from token 0. */
static void test_required_without_think_compels_from_start(void) {
    jr_value *tools = jr_parse(TOOLS_JSON);
    tgj_grammar *g = tgj_build(tools);
    tgj_state *st = tgj_state_new_required(g, false);
    ASSERT_TRUE(tgj_mask_active(st));
    ASSERT_FALSE(tgj_token_allowed(st, "Sure, let me look."));
    ASSERT_TRUE(tgj_token_allowed(st, "```json"));
    ASSERT_TRUE(tgj_token_allowed(st, "{\"name\": \"add\""));
    tgj_state_free(st);
    tgj_free(g);
    jr_free(tools);
}

/* The serve's reasoning_budget force-close ends a think-gated block through
 * tgj_reasoning_end (it injects "</think>" too, but the explicit end alone must
 * be enough): the call is compelled right after. */
static void test_required_think_gated_external_end(void) {
    jr_value *tools = jr_parse(TOOLS_JSON);
    tgj_grammar *g = tgj_build(tools);
    tgj_state *st = tgj_state_new_required(g, true);
    bool ok;
    feed_into(st, "still thinking about {", &ok);
    ASSERT_FALSE(tgj_mask_active(st));
    tgj_reasoning_end(st);
    ASSERT_TRUE(tgj_mask_active(st));
    ASSERT_FALSE(tgj_token_allowed(st, "prose"));
    tgj_state_free(st);
    tgj_free(g);
    jr_free(tools);
}

/* The sampler's full mask rejects a token whose first 1 or 2 bytes kill the walk
 * without walking the rest (engine.c grammar_first_char_verdicts).  Sound only
 * because DEAD is absorbing: check it at every prefix of a real required call,
 * for a spread of token texts. */
static void test_first_char_prefilter_is_sound(void) {
    jr_value *tools = jr_parse(TOOLS_JSON);
    tgj_grammar *g = tgj_build(tools);
    static const char *TOKENS[] = {
        "```json", "```", "{\"", "{\"function", "\":", " \"", "read", "\"}}",
        "\n```", "Sure", " the", "</think>", "\n\n", "{", "}", ",", " ", "\"path\": \"",
        "src/main.c", "\\n", "\"", "1", "true", "]", "\u00e9", "<tool_call>", NULL };
    const char *script = "think {about} it</think>\n\n```json\n{\"function\": \"read\", "
                         "\"arguments\": {\"path\": \"src/main.c\"}}\n```";
    tgj_state *st = tgj_state_new_required(g, true);
    char one[2] = { 0, 0 };
    int checked = 0, violations = 0;
    for (const char *p = script; ; p++) {
        for (int t = 0; TOKENS[t]; t++) {
            char two[3] = { TOKENS[t][0], TOKENS[t][0] ? TOKENS[t][1] : 0, 0 };
            one[0] = TOKENS[t][0];
            if (tgj_token_allowed(st, TOKENS[t]) && (!tgj_token_allowed(st, one) || !tgj_token_allowed(st, two))) violations++;
            checked++;
        }
        if (!*p) break;
        one[0] = *p;
        tgj_advance(st, one);
    }
    ASSERT_INT_EQ(violations, 0);
    ASSERT_TRUE(checked > 1000);
    tgj_state_free(st);
    tgj_free(g);
    jr_free(tools);
}

/* A "required" turn may end only where that is legal: never before the call
 * opens (the turn must make a call), and inside the call only where the serve
 * can close it — and the closing text it gets must complete the call. */
static void test_required_end_and_closing_text(void) {
    jr_value *tools = jr_parse(TOOLS_JSON);
    tgj_grammar *g = tgj_build(tools);
    char closing[16];
    bool ok;

    tgj_state *st = tgj_state_new_required(g, true);
    ASSERT_TRUE(tgj_end_allowed(st));                 /* thinking: free */
    feed_into(st, "plan</think>\n\n", &ok);
    ASSERT_FALSE(tgj_end_allowed(st));                /* compelled, no call yet */
    ASSERT_INT_EQ(tgj_closing_text(st, closing, 16), -1);
    feed_into(st, "```json\n{\"function\": \"read\", \"arguments\": {", &ok);
    ASSERT_FALSE(tgj_end_allowed(st));                /* required `path` missing */
    feed_into(st, "\"path\": \"a.c\n```", &ok);  /* ornith: fence inside the string */
    ASSERT_TRUE(ok);
    ASSERT_TRUE(tgj_end_allowed(st));
    ASSERT_INT_EQ(tgj_closing_text(st, closing, 16), 7);
    ASSERT_STR_EQ(closing, "\"}}\n```");
    tgj_advance(st, closing);
    ASSERT_TRUE(tgj_complete(st));
    ASSERT_TRUE(tgj_end_allowed(st));                 /* done: free again */
    tgj_state_free(st);

    /* Bare object, integer arguments: closed by the braces alone. */
    st = tgj_state_new_required(g, false);
    feed_into(st, "{\"name\":\"add\",\"parameters\":{\"a\": 1, \"b\": 22", &ok);
    ASSERT_TRUE(ok);
    ASSERT_INT_EQ(tgj_closing_text(st, closing, 16), 2);
    ASSERT_STR_EQ(closing, "}}");
    /* A key with no value yet cannot be closed. */
    tgj_state_free(st);
    st = tgj_state_new_required(g, false);
    feed_into(st, "{\"name\":\"add\",\"parameters\":{\"a\": 1, \"b\":", &ok);
    ASSERT_FALSE(tgj_end_allowed(st));
    tgj_state_free(st);

    tgj_free(g);
    jr_free(tools);
}

int main(void) {
    printf("\n=== tool_grammar_json (ornith JSON grammar) ===\n\n");
    RUN_TEST(fenced_accepts);
    RUN_TEST(bare_accepts);
    RUN_TEST(name_and_parameters_keys);
    RUN_TEST(integer_args);
    RUN_TEST(integer_rejects_string);
    RUN_TEST(unknown_name_rejected);
    RUN_TEST(unknown_key_rejected);
    RUN_TEST(empty_value_impossible);
    RUN_TEST(empty_value_allowed_without_a_floor);
    RUN_TEST(whitespace_only_value_impossible);
    RUN_TEST(empty_args_impossible_when_required);
    RUN_TEST(optional_key_omittable);
    RUN_TEST(cross_token_key);
    RUN_TEST(prose_then_call_auto);
    RUN_TEST(forced_rejects_prose);
    RUN_TEST(forced_self_detects_think_close);
    RUN_TEST(forced_function_pins_name);
    RUN_TEST(zero_budget_forced_arming);
    RUN_TEST(reason_budget_trips_mid_token_without_dying);
    RUN_TEST(dead_walk_does_not_report_reasoning_ended);
    RUN_TEST(auto_mode_masking);
    RUN_TEST(truncated_incomplete);
    RUN_TEST(enum_closes_switch_mode);
    RUN_TEST(enum_cross_token);
    RUN_TEST(enum_on_optional_key);
    RUN_TEST(enum_degrades_when_unrepresentable);
    RUN_TEST(enum_over_cap_degrades);
    RUN_TEST(build_guards);
    RUN_TEST(required_think_gated_reasoning_is_free);
    RUN_TEST(required_without_think_compels_from_start);
    RUN_TEST(required_think_gated_external_end);
    RUN_TEST(first_char_prefilter_is_sound);
    RUN_TEST(required_end_and_closing_text);
    TEST_REPORT();
    return __ft_failed == 0 ? 0 : 1;
}
