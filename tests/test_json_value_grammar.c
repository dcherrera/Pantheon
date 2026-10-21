/* SPDX-License-Identifier: AGPL-3.0-or-later */
/* Copyright (C) 2026 David (Corey) Herrera */
/*
 * Unit tests for the incremental JSON-value walk (src/serve/json_value_grammar.c).
 *
 * This is the piece that makes a container-typed tool argument unfailable: the
 * tool grammar runs it over the bytes inside <parameter=KEY>…</parameter> when
 * the parameter's schema says "array"/"object", and tool_calling.c runs the same
 * walk over the finished text before splicing it into `arguments` raw.  Both
 * layers must agree about what "well-formed" means, so both call THIS.
 *
 * The anchor case is the live one: three consecutive `todowrite` calls from
 * qwen36-bf16, frame grammar armed and masking correctly, all three killed by
 * the same malformation inside a free-text parameter body.
 *
 * Pure C, libc-only — no engine/model/CUDA.
 */

#include "lib/test.h"
#include "../src/serve/json_value_grammar.h"
#include <stdio.h>
#include <string.h>

/* ── the live failures ──
 *
 * Reconstructed from the run: the `todos` parameter is declared
 * {"type":"array","items":{"type":"object",…}} (the client's schema code
 * emit_todo_tool), so `write_typed_value` splices these bytes into the call's
 * `arguments` object RAW.  Each one therefore broke the WHOLE call, not one
 * argument. */

/* Failure 1: a bare value where a key was owed, after the comma. */
static const char *LIVE_BAD_1 =
    "[{\"content\":\"Remove idx_points_scope\",\"status\":\"in_progress\"},"
    "{\"content\":\"\",\"\"}]";

/* Failure 2: the same shape with a longer bare string. */
static const char *LIVE_BAD_2 =
    "[{\"content\":\"a\",\"status\":\"pending\"},"
    "{\"content\":\"_verify the fix works end-to-end_\",\"_verify\"}]";

/* What the model MEANT both times — must stay reachable. */
static const char *LIVE_GOOD =
    "[{\"content\":\"Remove idx_points_scope\",\"status\":\"in_progress\"},"
    "{\"content\":\"verify the fix works end-to-end\",\"status\":\"pending\"}]";

/* Feed `s` char by char; returns the char index at which the walk DIED, or -1
 * if it never died.  Feeding one char at a time is the strictest possible
 * granularity — the sampler masks whole tokens, so anything rejected here is
 * rejected there too. */
static int death_index(const char *s, char container) {
    jvg_walk w;
    jvg_begin(&w, container);
    for (int i = 0; s[i]; i++) {
        jvg_step(&w, s[i]);
        if (jvg_is_dead(&w)) return i;
    }
    return -1;
}

/* ── the regression these tests exist for ── */

static void test_live_todowrite_malformation_is_rejected(void) {
    /* Both live failures must die, and die INSIDE the value — not merely be
     * reported wrong at the end. */
    int d1 = death_index(LIVE_BAD_1, '[');
    ASSERT_INT_NE(d1, -1);
    /* It dies on the '}' that follows the bare string: the string was read as a
     * key, so a ':' was owed. */
    ASSERT_TRUE(LIVE_BAD_1[d1] == '}');

    int d2 = death_index(LIVE_BAD_2, '[');
    ASSERT_INT_NE(d2, -1);
    ASSERT_TRUE(LIVE_BAD_2[d2] == '}');

    /* And the well-formed intent is still reachable. */
    ASSERT_INT_EQ(death_index(LIVE_GOOD, '['), -1);
    ASSERT_TRUE(jvg_text_is_json_value(LIVE_GOOD, '['));
    ASSERT_FALSE(jvg_text_is_json_value(LIVE_BAD_1, '['));
    ASSERT_FALSE(jvg_text_is_json_value(LIVE_BAD_2, '['));
}

/* ── the structural rules ── */

static void test_wellformed_values_accepted(void) {
    static const char *ok[] = {
        "{}", "[]", "[1,2,3]", "{\"a\":1}", "[{\"a\":[true,false,null]}]",
        "  [ 1 , 2 ]  ", "{\"a\":{\"b\":{\"c\":[]}}}",
        "[-1,0,1.5,1e10,1E-10,-0.5e+3]",
        "[\"\\\"\",\"\\\\\",\"\\/\",\"\\b\",\"\\f\",\"\\n\",\"\\r\",\"\\t\"]",
        "[\"\\u00e9\"]", "[\"a < b\"]", NULL,
    };
    for (int i = 0; ok[i]; i++) {
        if (!jvg_text_is_json_value(ok[i], 0)) {
            printf("    should have been accepted: %s\n", ok[i]);
            ASSERT_TRUE(0);
        }
    }
}

static void test_malformed_values_rejected(void) {
    static const char *bad[] = {
        "{",  "[",  "]", "}", "",
        "[1,]",                 /* trailing comma in an array */
        "{\"a\":1,}",           /* trailing comma in an object */
        "{\"a\"}",              /* key with no value */
        "{\"a\":}",             /* colon with no value */
        "{a:1}",                /* unquoted key */
        "{\"a\":1,\"b\"}",      /* second member missing its value */
        "[1,2",                 /* never closed */
        "[1,2]]",               /* trailing junk */
        "[1,2] x",              /* trailing junk after whitespace */
        "[1 2]",                /* missing comma */
        "[-]", "[1.]", "[1e]", "[01x]",   /* malformed numbers */
        "[tru]", "[truex]",     /* malformed literals */
        "[\"unterminated]",
        "[\"bad\\escape\"]",
        "{]", "[}",             /* mismatched close */
        NULL,
    };
    for (int i = 0; bad[i]; i++) {
        if (jvg_text_is_json_value(bad[i], 0)) {
            printf("    should have been rejected: %s\n", bad[i]);
            ASSERT_TRUE(0);
        }
    }
}

static void test_raw_control_char_in_string_rejected(void) {
    /* A literal newline inside a JSON string is illegal and would make the raw
     * splice un-parseable at the client. */
    ASSERT_FALSE(jvg_text_is_json_value("[\"a\nb\"]", '['));
    ASSERT_TRUE(jvg_text_is_json_value("[\"a\\nb\"]", '['));
}

static void test_required_container_is_enforced(void) {
    /* An "array" parameter may not be handed an object, a string or a number —
     * all three are well-formed JSON and all three are the wrong type.  This is
     * what guarantees write_typed_value's raw-splice branch is the one taken. */
    ASSERT_TRUE(jvg_text_is_json_value("[1]", '['));
    ASSERT_FALSE(jvg_text_is_json_value("{\"a\":1}", '['));
    ASSERT_FALSE(jvg_text_is_json_value("\"a list\"", '['));
    ASSERT_FALSE(jvg_text_is_json_value("3", '['));

    ASSERT_TRUE(jvg_text_is_json_value("{\"a\":1}", '{'));
    ASSERT_FALSE(jvg_text_is_json_value("[1]", '{'));

    /* With no container demanded, any JSON value is fine. */
    ASSERT_TRUE(jvg_text_is_json_value("3", 0));
    ASSERT_TRUE(jvg_text_is_json_value("\"a\"", 0));
    ASSERT_TRUE(jvg_text_is_json_value("null", 0));

    /* Leading whitespace is padding either way (the parser trims it). */
    ASSERT_TRUE(jvg_text_is_json_value("  \n [1]", '['));
}

static void test_completion_is_reported_at_the_closing_char(void) {
    jvg_walk w;
    jvg_begin(&w, '[');
    const char *s = "[1,2]";
    for (int i = 0; s[i]; i++) {
        jvg_step(&w, s[i]);
        /* Complete EXACTLY at the final ']' and not one char earlier — the tool
         * grammar keys "</parameter> is now legal" off this bit. */
        ASSERT_INT_EQ(jvg_is_complete(&w) ? 1 : 0, s[i + 1] == '\0' ? 1 : 0);
    }
    /* Anything after a complete value is off-grammar. */
    jvg_step(&w, '0');
    ASSERT_TRUE(jvg_is_dead(&w));
}

static void test_number_reconsume_at_a_boundary(void) {
    /* A number is only ended by the char after it, and that char is structural.
     * All three boundaries must work. */
    ASSERT_TRUE(jvg_text_is_json_value("[1,2]", '['));
    ASSERT_TRUE(jvg_text_is_json_value("[1 ]", '['));
    ASSERT_TRUE(jvg_text_is_json_value("{\"a\":1}", '{'));
    ASSERT_TRUE(jvg_text_is_json_value("[[1]]", '['));
}

static void test_depth_cap_is_a_wall_not_a_crash(void) {
    char deep[2 * JVG_MAX_DEPTH + 8];
    int n = 0;
    for (int i = 0; i < JVG_MAX_DEPTH + 1; i++) deep[n++] = '[';
    deep[n] = '\0';
    jvg_walk w;
    jvg_begin(&w, '[');
    for (int i = 0; deep[i]; i++) jvg_step(&w, deep[i]);
    ASSERT_TRUE(jvg_is_dead(&w));          /* the (cap+1)th open is not allowed */
    ASSERT_INT_EQ(jvg_depth(&w), JVG_MAX_DEPTH);

    /* Exactly at the cap is fine. */
    n = 0;
    for (int i = 0; i < JVG_MAX_DEPTH; i++) deep[n++] = '[';
    for (int i = 0; i < JVG_MAX_DEPTH; i++) deep[n++] = ']';
    deep[n] = '\0';
    ASSERT_TRUE(jvg_text_is_json_value(deep, '['));
}

static void test_state_is_a_copyable_pod(void) {
    /* The whole design rests on this: the mask clones the walk per candidate
     * token and throws the clone away, so a clone must be independent and the
     * struct must stay small enough for ~248k copies a step. */
    ASSERT_TRUE(sizeof(jvg_walk) <= 32);

    jvg_walk live;
    jvg_begin(&live, '[');
    jvg_step(&live, '[');
    jvg_walk probe = live;
    jvg_step(&probe, '}');                 /* kill the CLONE */
    ASSERT_TRUE(jvg_is_dead(&probe));
    ASSERT_FALSE(jvg_is_dead(&live));      /* the live walk is untouched */
    jvg_step(&live, ']');
    ASSERT_TRUE(jvg_is_complete(&live));
}

static void test_null_and_empty_inputs(void) {
    ASSERT_FALSE(jvg_is_dead(NULL));
    ASSERT_FALSE(jvg_is_complete(NULL));
    ASSERT_INT_EQ(jvg_depth(NULL), 0);
    ASSERT_TRUE(strcmp(jvg_phase_name(NULL), "none") == 0);
    ASSERT_FALSE(jvg_text_is_json_value(NULL, 0));
    ASSERT_FALSE(jvg_text_is_json_value("", 0));
    jvg_step(NULL, 'x');   /* must not crash */
    jvg_begin(NULL, '[');  /* must not crash */
}

int main(void) {
    printf("\n=== json_value_grammar (incremental JSON-value walk) ===\n\n");
    RUN_TEST(live_todowrite_malformation_is_rejected);
    RUN_TEST(wellformed_values_accepted);
    RUN_TEST(malformed_values_rejected);
    RUN_TEST(raw_control_char_in_string_rejected);
    RUN_TEST(required_container_is_enforced);
    RUN_TEST(completion_is_reported_at_the_closing_char);
    RUN_TEST(number_reconsume_at_a_boundary);
    RUN_TEST(depth_cap_is_a_wall_not_a_crash);
    RUN_TEST(state_is_a_copyable_pod);
    RUN_TEST(null_and_empty_inputs);
    TEST_REPORT();
    return __ft_failed == 0 ? 0 : 1;
}
