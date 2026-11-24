/* SPDX-License-Identifier: AGPL-3.0-or-later */
/* Copyright (C) 2026 David (Corey) Herrera */
/*
 * Unit tests for `field_patterns` in the JSON tool-call grammar
 * (tgj_apply_field_patterns; src/serve/tool_grammar_json.c,
 * src/serve/field_pattern.c).
 *
 * The pattern judges the DECODED string: escapes are decoded before their bytes
 * are fed (`\"` is `"`, `\n` a newline, `\uXXXX` UTF-8, surrogate pairs joined),
 * every char that leaves no way to a match is masked, and the closing quote is
 * legal only on a match.  Same feed style as tests/test_tool_grammar_json_reasoned.c.
 *
 * Pure C, libc-only — fully local, no CUDA.
 */

#include "lib/test.h"
#include "../src/serve/tool_grammar_json.h"
#include "../src/serve/json_reader.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

/* read/grep carry the client's pattern on `finding`; bash has none; `note` has a
 * UTF-8 pattern for the \u decoding; read's integer, enum and unknown-type keys
 * must ignore the entries the field gives them. */
static const char *TOOLS_JSON =
    "[{\"type\":\"function\",\"function\":{\"name\":\"read\",\"parameters\":{"
    "\"type\":\"object\",\"properties\":{"
    "\"path\":{\"type\":\"string\",\"minLength\":1},"
    "\"finding\":{\"type\":\"string\",\"minLength\":1},"
    "\"limit\":{\"type\":\"integer\"},"
    "\"mode\":{\"type\":\"string\",\"enum\":[\"fast\",\"full\"]},"
    "\"ref\":{\"type\":\"uuid\"}},"
    "\"required\":[\"path\",\"finding\"]}}},"
    "{\"type\":\"function\",\"function\":{\"name\":\"grep\",\"parameters\":{"
    "\"type\":\"object\",\"properties\":{"
    "\"pattern\":{\"type\":\"string\",\"minLength\":1},"
    "\"finding\":{\"type\":\"string\",\"minLength\":1}},"
    "\"required\":[\"pattern\",\"finding\"]}}},"
    "{\"type\":\"function\",\"function\":{\"name\":\"bash\",\"parameters\":{"
    "\"type\":\"object\",\"properties\":{"
    "\"command\":{\"type\":\"string\",\"minLength\":1},"
    "\"finding\":{\"type\":\"string\",\"minLength\":1}},"
    "\"required\":[\"command\",\"finding\"]}}},"
    "{\"type\":\"function\",\"function\":{\"name\":\"note\",\"parameters\":{"
    "\"type\":\"object\",\"properties\":{\"text\":{\"type\":\"string\"}},"
    "\"required\":[\"text\"]}}}]";

#define FINDING_PATTERN_JSON \
    "\"-|[^\\\\n]+ from: [^\\\\s,:]+:[0-9]+(-[0-9]+)?(, [^\\\\s,:]+:[0-9]+(-[0-9]+)?)*\""

static const char *FIELD_JSON =
    "{\"read\":{\"finding\":" FINDING_PATTERN_JSON ",\"limit\":\"x\",\"mode\":\"x\","
    "\"ref\":\"x\"},"
    "\"grep\":{\"finding\":" FINDING_PATTERN_JSON "},"
    "\"note\":{\"text\":\"caf\xc3\xa9|\xf0\x9f\x98\x80|A\"},\"nope\":{\"finding\":\"x\"}}";

static const char *READ_HEAD =
    "\n```json\n{\"function\": \"read\", \"arguments\": {\"path\": \"a.c\", \"finding\": \"";
static const char *GREP_HEAD =
    "\n```json\n{\"function\": \"grep\", \"arguments\": {\"pattern\": \"foo\", \"finding\": \"";
static const char *BASH_HEAD =
    "\n```json\n{\"function\": \"bash\", \"arguments\": {\"command\": \"ls\", \"finding\": \"";
static const char *NOTE_HEAD = "\n```json\n{\"function\": \"note\", \"arguments\": {\"text\": \"";
static const char *VALUE_TAIL = "\"}}\n```";

typedef struct {
    jr_value    *tools;
    jr_value    *field;
    tgj_grammar *g;
} fixture;

static fixture fixture_open(const char *field_json) {
    fixture f = { jr_parse(TOOLS_JSON), NULL, NULL };
    f.g = tgj_build(f.tools);
    if (field_json) {
        f.field = jr_parse(field_json);
        /* A failed apply leaves no grammar, so every walk of the test fails. */
        if (!tgj_apply_field_patterns(f.g, f.field)) { tgj_free(f.g); f.g = NULL; }
    }
    return f;
}

static void fixture_close(fixture *f) {
    tgj_free(f->g);
    jr_free(f->field);
    jr_free(f->tools);
}

static bool feed_chars(tgj_state *st, const char *s) {
    char one[2] = { 0, 0 };
    for (const char *p = s; *p; p++) {
        one[0] = *p;
        if (!tgj_token_allowed(st, one)) return false;
        tgj_advance(st, one);
    }
    return true;
}

static tgj_state *walk_to_value(const fixture *f, const char *head) {
    tgj_state *st = tgj_state_new(f->g);
    if (feed_chars(st, head)) return st;
    tgj_state_free(st);
    return NULL;   /* every later check on a NULL walk fails */
}

/* Does the (still JSON-escaped) `value` then VALUE_TAIL complete the call? */
static bool value_completes(const fixture *f, const char *head, const char *value) {
    tgj_state *st = walk_to_value(f, head);
    bool ok = feed_chars(st, value) && feed_chars(st, VALUE_TAIL) && tgj_complete(st);
    tgj_state_free(st);
    return ok;
}

/* 1. A violating value is masked at its first bad char: an escaped newline is a
 *    newline, refused by [^\n] the moment its escape letter arrives. */
static void test_violation_masked_at_first_bad_char(void) {
    fixture f = fixture_open(FIELD_JSON);
    tgj_state *st = walk_to_value(&f, READ_HEAD);
    ASSERT_TRUE(feed_chars(st, "x\\"));
    ASSERT_FALSE(tgj_token_allowed(st, "n"));
    ASSERT_FALSE(tgj_token_allowed(st, "nmore"));
    ASSERT_TRUE(tgj_token_allowed(st, "t"));   /* a tab is not a newline */
    tgj_state_free(st);
    st = walk_to_value(&f, READ_HEAD);
    ASSERT_TRUE(feed_chars(st, "x from: a.c:10"));
    ASSERT_FALSE(tgj_token_allowed(st, "\\n"));   /* no trimming in JSON */
    ASSERT_FALSE(tgj_token_allowed(st, "\n"));    /* nor a raw newline byte */
    tgj_state_free(st);
    fixture_close(&f);
}

/* 2. A matching value closes, including `-`. */
static void test_matching_value_closes(void) {
    fixture f = fixture_open(FIELD_JSON);
    ASSERT_TRUE(value_completes(&f, READ_HEAD, "x from: a.c:10"));
    ASSERT_TRUE(value_completes(&f, READ_HEAD, "-"));
    tgj_state *st = walk_to_value(&f, READ_HEAD);
    const char *pieces[] = { "x", " from", ":", " a", ".c", ":", "10", "\"}}", "\n```", NULL };
    for (int i = 0; pieces[i]; i++) {
        ASSERT_TRUE(tgj_token_allowed(st, pieces[i]));
        tgj_advance(st, pieces[i]);
    }
    ASSERT_TRUE(tgj_complete(st));
    tgj_state_free(st);
    fixture_close(&f);
}

/* 3. The closing quote is masked in a non-accepting state, for the sampler and
 *    the serve's closer. */
static void test_close_masked_when_not_accepting(void) {
    fixture f = fixture_open(FIELD_JSON);
    tgj_state *st = tgj_state_new_required(f.g, false);
    ASSERT_TRUE(feed_chars(st, READ_HEAD));
    ASSERT_TRUE(feed_chars(st, "x from: a.c"));
    char closing[32];
    ASSERT_FALSE(tgj_token_allowed(st, "\""));
    ASSERT_FALSE(tgj_token_allowed(st, "\"}}"));
    ASSERT_INT_EQ(tgj_closing_text(st, closing, (int)sizeof(closing)), -1);
    ASSERT_FALSE(tgj_end_allowed(st));
    ASSERT_TRUE(feed_chars(st, ":7"));
    ASSERT_TRUE(tgj_closing_text(st, closing, (int)sizeof(closing)) > 0);
    tgj_state_free(st);
    ASSERT_FALSE(value_completes(&f, READ_HEAD, "x"));
    ASSERT_FALSE(value_completes(&f, READ_HEAD, "x from: a.c:10,b.h:5"));
    ASSERT_FALSE(value_completes(&f, READ_HEAD, "-x"));
    ASSERT_FALSE(value_completes(&f, READ_HEAD, " - "));   /* no trimming */
    fixture_close(&f);
}

/* 4. grep cites several ranges. */
static void test_grep_multi_range_closes(void) {
    fixture f = fixture_open(FIELD_JSON);
    ASSERT_TRUE(value_completes(&f, GREP_HEAD, "x from: a.c:10-20, b.h:5"));
    ASSERT_FALSE(value_completes(&f, GREP_HEAD, "x from: a.c:1, "));
    fixture_close(&f);
}

/* 5. Unpatterned fields keep today's walk, invalid escapes included. */
static void test_unpatterned_fields_unchanged(void) {
    fixture f = fixture_open(FIELD_JSON);
    ASSERT_TRUE(value_completes(&f, BASH_HEAD, "anything\\nat all"));
    ASSERT_TRUE(value_completes(&f, BASH_HEAD, "odd \\q escape"));
    ASSERT_FALSE(value_completes(&f, READ_HEAD, "odd \\q from: a.c:1"));
    const char *keys = "\n```json\n{\"function\": \"read\", \"arguments\": {\"limit\": 12, "
                       "\"mode\": \"fast\", \"ref\": \"abc\", \"path\": \"a b\", "
                       "\"finding\": \"";
    ASSERT_TRUE(value_completes(&f, keys, "-"));
    fixture_close(&f);
}

/* 6. Escapes are decoded before the pattern sees them. */
static void test_escapes_decoded(void) {
    fixture f = fixture_open(FIELD_JSON);
    ASSERT_TRUE(value_completes(&f, READ_HEAD, "say \\\"hi\\\" from: a.c:1"));
    ASSERT_TRUE(value_completes(&f, READ_HEAD, "x from: src\\/a.c:1"));
    ASSERT_TRUE(value_completes(&f, READ_HEAD, "x\\\\y from: a.c:1"));
    ASSERT_FALSE(value_completes(&f, READ_HEAD, "x from:\\ta.c:1"));   /* \t is not ' ' */
    ASSERT_TRUE(value_completes(&f, NOTE_HEAD, "caf\\u00e9"));
    ASSERT_TRUE(value_completes(&f, NOTE_HEAD, "caf\\u00E9"));
    ASSERT_TRUE(value_completes(&f, NOTE_HEAD, "caf\xc3\xa9"));          /* raw UTF-8 */
    ASSERT_TRUE(value_completes(&f, NOTE_HEAD, "\\ud83d\\ude00"));       /* a surrogate pair */
    ASSERT_TRUE(value_completes(&f, NOTE_HEAD, "\\u0041"));
    ASSERT_FALSE(value_completes(&f, NOTE_HEAD, "\\ud83d"));             /* lone high half */
    ASSERT_FALSE(value_completes(&f, NOTE_HEAD, "\\ud83dA"));            /* ... is not dropped */
    ASSERT_FALSE(value_completes(&f, NOTE_HEAD, "cafe"));
    tgj_state *st = walk_to_value(&f, NOTE_HEAD);
    ASSERT_TRUE(feed_chars(st, "caf"));
    ASSERT_FALSE(tgj_token_allowed(st, "e"));
    ASSERT_TRUE(feed_chars(st, "\\u00"));
    ASSERT_FALSE(tgj_token_allowed(st, "\""));   /* a \u owes four hex digits */
    ASSERT_FALSE(tgj_token_allowed(st, "zz"));
    ASSERT_FALSE(tgj_token_allowed(st, "e8"));   /* decodes to è */
    ASSERT_TRUE(tgj_token_allowed(st, "e9\""));
    tgj_state_free(st);
    fixture_close(&f);
}

/* 7. clone / copy mid-escape carry the pattern state (speculation / MTP). */
static void test_clone_and_copy_carry_pattern(void) {
    fixture f = fixture_open(FIELD_JSON);
    tgj_state *st = walk_to_value(&f, NOTE_HEAD);
    ASSERT_TRUE(feed_chars(st, "\\ud83d\\ude0"));
    tgj_state *clone = tgj_state_clone(st);
    ASSERT_TRUE(feed_chars(clone, "0"));
    ASSERT_TRUE(tgj_token_allowed(clone, "\""));
    ASSERT_FALSE(tgj_token_allowed(st, "\""));   /* live walk untouched */
    tgj_state *other = walk_to_value(&f, BASH_HEAD);
    tgj_state_copy(other, st);
    ASSERT_FALSE(tgj_token_allowed(other, "1"));   /* U+1F601 is not in the pattern */
    ASSERT_TRUE(feed_chars(other, "0"));
    ASSERT_TRUE(feed_chars(other, VALUE_TAIL));
    ASSERT_TRUE(tgj_complete(other));
    tgj_state_free(other);
    tgj_state_free(clone);
    tgj_state_free(st);
    fixture_close(&f);
}

/* 8. A field that binds nothing leaves every mask byte-identical to no field. */
static void test_noop_field_masks_identical(void) {
    static const char *TOKENS[] = {
        "\"", "\"}}", "}", "\\", "\\n", "\\u00", "x", "-", " from", ":", "10", ",", ".",
        "\n```", "{", "\"finding\"", "</think>", "Sure", NULL };
    static const char *FIELDS[] = {
        "null", "{}", "{\"nope\":{\"finding\":\"x\"}}", "{\"read\":{\"nope\":\"x\"}}",
        "{\"read\":{\"finding\":7}}", "{\"read\":{\"limit\":\"x\",\"mode\":\"x\"}}",
        "{\"read\":\"x\"}", "[\"read\"]" };
    char script[512];
    snprintf(script, sizeof(script), "I will read it.</think>%sx\\ny \\q \\u00e9 from a.c%s",
             READ_HEAD, VALUE_TAIL);
    fixture base = fixture_open(NULL);
    int mismatches = 0, checked = 0;
    for (size_t fi = 0; fi < sizeof(FIELDS) / sizeof(FIELDS[0]); fi++) {
        fixture f = fixture_open(FIELDS[fi]);
        for (int mode = 0; mode < 2; mode++) {
            tgj_state *a = mode ? tgj_state_new(base.g) : tgj_state_new_required(base.g, true);
            tgj_state *b = mode ? tgj_state_new(f.g) : tgj_state_new_required(f.g, true);
            char one[2] = { 0, 0 };
            for (const char *p = script; ; p++) {
                for (int t = 0; TOKENS[t]; t++, checked++)
                    if (tgj_token_allowed(a, TOKENS[t]) != tgj_token_allowed(b, TOKENS[t]))
                        mismatches++;
                if (tgj_end_allowed(a) != tgj_end_allowed(b)) mismatches++;
                if (!*p) break;
                one[0] = *p;
                tgj_advance(a, one);
                tgj_advance(b, one);
            }
            ASSERT_TRUE(tgj_complete(b));
            tgj_state_free(a);
            tgj_state_free(b);
        }
        fixture_close(&f);
    }
    ASSERT_INT_EQ(mismatches, 0);
    ASSERT_TRUE(checked > 5000);
    fixture_close(&base);
}

/* 9. A pattern that does not compile fails the apply. */
static void test_bad_pattern_fails_apply(void) {
    jr_value *tools = jr_parse(TOOLS_JSON);
    jr_value *field = jr_parse("{\"read\":{\"finding\":\"[z-a]\"}}");
    tgj_grammar *g = tgj_build(tools);
    ASSERT_FALSE(tgj_apply_field_patterns(g, field));
    tgj_free(g);
    jr_free(field);
    jr_free(tools);
}

int main(void) {
    printf("\n=== tool_grammar_json field_patterns ===\n\n");
    RUN_TEST(violation_masked_at_first_bad_char);
    RUN_TEST(matching_value_closes);
    RUN_TEST(close_masked_when_not_accepting);
    RUN_TEST(grep_multi_range_closes);
    RUN_TEST(unpatterned_fields_unchanged);
    RUN_TEST(escapes_decoded);
    RUN_TEST(clone_and_copy_carry_pattern);
    RUN_TEST(noop_field_masks_identical);
    RUN_TEST(bad_pattern_fails_apply);
    TEST_REPORT();
    return __ft_failed == 0 ? 0 : 1;
}
