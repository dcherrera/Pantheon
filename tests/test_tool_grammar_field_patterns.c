/* SPDX-License-Identifier: AGPL-3.0-or-later */
/* Copyright (C) 2026 David (Corey) Herrera */
/*
 * Unit tests for `field_patterns` in the XML tool-call grammar
 * (tg_apply_field_patterns; src/serve/tool_grammar.c, src/serve/field_pattern.c).
 *
 * A pattern on a tool's string property binds that value WHOLE: every byte that
 * leaves no way to a matching value is masked, and "</parameter>" closes only on
 * a match.  The value judged is the one the parser hands the client — the bytes
 * before "</parameter>", whitespace-trimmed — so the trained framing newlines
 * are not part of it.  Same feed style as tests/test_tool_grammar_reasoned.c:
 * char-by-char walks plus whole token pieces.
 *
 * Pure C, libc-only — fully local, no CUDA.
 */

#include "lib/test.h"
#include "../src/serve/tool_grammar.h"
#include "../src/serve/json_reader.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

/* The client's code-reading tools (`finding` carries the pattern), `bash` (no
 * pattern), on `read` an integer, an enum and an unknown-type key the field
 * also names — all ignored (not free-text strings) — and `tag`, whose pattern
 * ends in '<' so a value can close right after a terminator look-alike. */
static const char *TOOLS_JSON =
    "[{\"type\":\"function\",\"function\":{\"name\":\"read\",\"parameters\":{"
    "\"type\":\"object\",\"properties\":{"
    "\"path\":{\"type\":\"string\",\"minLength\":1},"
    "\"finding\":{\"type\":\"string\",\"minLength\":1},"
    "\"limit\":{\"type\":\"integer\"},"
    "\"mode\":{\"type\":\"string\",\"enum\":[\"fast\",\"full\"]},"
    "\"ref\":{\"type\":\"uuid\"}},"
    "\"required\":[\"path\",\"finding\"]}}},"
    "{\"type\":\"function\",\"function\":{\"name\":\"tag\",\"parameters\":{"
    "\"type\":\"object\",\"properties\":{\"text\":{\"type\":\"string\"}}}}},"
    "{\"type\":\"function\",\"function\":{\"name\":\"grep\",\"parameters\":{"
    "\"type\":\"object\",\"properties\":{"
    "\"pattern\":{\"type\":\"string\",\"minLength\":1},"
    "\"finding\":{\"type\":\"string\",\"minLength\":1}},"
    "\"required\":[\"pattern\",\"finding\"]}}},"
    "{\"type\":\"function\",\"function\":{\"name\":\"bash\",\"parameters\":{"
    "\"type\":\"object\",\"properties\":{"
    "\"command\":{\"type\":\"string\",\"minLength\":1},"
    "\"finding\":{\"type\":\"string\",\"minLength\":1}},"
    "\"required\":[\"command\",\"finding\"]}}}]";

/* The pattern the client sends, as JSON. */
#define FINDING_PATTERN_JSON \
    "\"-|[^\\\\n]+ from: [^\\\\s,:]+:[0-9]+(-[0-9]+)?(, [^\\\\s,:]+:[0-9]+(-[0-9]+)?)*\""

/* The request field: the pattern on read/grep, plus entries that must be
 * ignored — an unknown tool, an unknown property, an integer and an enum key. */
static const char *FIELD_JSON =
    "{\"read\":{\"finding\":" FINDING_PATTERN_JSON ",\"nope\":\"x\",\"limit\":\"x\","
    "\"mode\":\"x\",\"ref\":\"x\"},\"grep\":{\"finding\":" FINDING_PATTERN_JSON "},"
    "\"tag\":{\"text\":\"a<|b\"},"
    "\"outline\":{\"finding\":" FINDING_PATTERN_JSON "},\"nope\":{\"finding\":\"x\"}}";

static const char *READ_HEAD = "<tool_call>\n<function=read>\n"
                               "<parameter=path>\na.c\n</parameter>\n"
                               "<parameter=finding>\n";
static const char *GREP_HEAD = "<tool_call>\n<function=grep>\n"
                               "<parameter=pattern>\nfoo\n</parameter>\n"
                               "<parameter=finding>\n";
static const char *BASH_HEAD = "<tool_call>\n<function=bash>\n"
                               "<parameter=command>\nls\n</parameter>\n"
                               "<parameter=finding>\n";
static const char *TAG_HEAD = "<tool_call>\n<function=tag>\n<parameter=text>\n";
static const char *VALUE_TAIL = "\n</parameter>\n</function>\n</tool_call>";

typedef struct {
    jr_value   *tools;
    jr_value   *field;
    tg_grammar *g;
} fixture;

static fixture fixture_open(const char *field_json) {
    fixture f = { jr_parse(TOOLS_JSON), NULL, NULL };
    f.g = tg_build(f.tools);
    if (field_json) {
        f.field = jr_parse(field_json);
        /* A failed apply leaves no grammar, so every walk of the test fails. */
        if (!tg_apply_field_patterns(f.g, f.field)) { tg_free(f.g); f.g = NULL; }
    }
    return f;
}

static void fixture_close(fixture *f) {
    tg_free(f->g);
    jr_free(f->field);
    jr_free(f->tools);
}

/* Feed `s` one char at a time; false at the first char the mask would refuse. */
static bool feed_chars(tg_state *st, const char *s) {
    char one[2] = { 0, 0 };
    for (const char *p = s; *p; p++) {
        one[0] = *p;
        if (!tg_token_allowed(st, one)) return false;
        tg_advance(st, one);
    }
    return true;
}

/* A fresh AUTO walk positioned at the start of a value, after `head` (NULL if
 * the head is refused). */
static tg_state *walk_to_value(const fixture *f, const char *head) {
    tg_state *st = tg_state_new(f->g);
    if (feed_chars(st, head)) return st;
    tg_state_free(st);
    return NULL;   /* every later check on a NULL walk fails */
}

/* Does `value` then VALUE_TAIL complete the call, walked char by char? */
static bool value_completes(const fixture *f, const char *head, const char *value) {
    tg_state *st = walk_to_value(f, head);
    bool ok = feed_chars(st, value) && feed_chars(st, VALUE_TAIL) && tg_complete(st);
    tg_state_free(st);
    return ok;
}

/* 1. A violating value is masked at the first byte that cannot lead to a match:
 *    an interior newline once the fact can no longer close, or the first byte
 *    after a trailing newline that can only be trailing. */
static void test_violation_masked_at_first_bad_char(void) {
    fixture f = fixture_open(FIELD_JSON);
    tg_state *st = walk_to_value(&f, READ_HEAD);
    ASSERT_TRUE(feed_chars(st, "x"));
    ASSERT_FALSE(tg_token_allowed(st, "\n"));      /* "x" cannot close: no trailing ws */
    ASSERT_FALSE(tg_token_allowed(st, "\ny"));
    ASSERT_TRUE(tg_token_allowed(st, " from"));
    tg_state_free(st);

    st = walk_to_value(&f, READ_HEAD);
    ASSERT_TRUE(feed_chars(st, "x from: a.c:10\n"));   /* accepting: "\n" may trail */
    ASSERT_FALSE(tg_token_allowed(st, "m"));
    ASSERT_FALSE(tg_token_allowed(st, "more"));
    ASSERT_TRUE(tg_token_allowed(st, "\n"));
    ASSERT_TRUE(tg_token_allowed(st, "</parameter>"));
    tg_state_free(st);

    st = walk_to_value(&f, READ_HEAD);
    ASSERT_TRUE(feed_chars(st, "-\n"));
    ASSERT_FALSE(tg_token_allowed(st, "x"));
    tg_state_free(st);
    fixture_close(&f);
}

/* 2. A matching value closes, including the "nothing new" `-`. */
static void test_matching_value_closes(void) {
    fixture f = fixture_open(FIELD_JSON);
    ASSERT_TRUE(value_completes(&f, READ_HEAD, "x from: a.c:10"));
    ASSERT_TRUE(value_completes(&f, READ_HEAD, "-"));
    ASSERT_TRUE(value_completes(&f, READ_HEAD, "the cap is `4096` from: src/io/buf.c:88-91"));
    /* Whole token pieces, as a real sampler draws them. */
    tg_state *st = walk_to_value(&f, READ_HEAD);
    const char *pieces[] = { "x", " from", ":", " a", ".c", ":", "10", "\n", "</parameter>",
                             "\n", "</function>", "\n", "</tool_call>", NULL };
    for (int i = 0; pieces[i]; i++) {
        ASSERT_TRUE(tg_token_allowed(st, pieces[i]));
        tg_advance(st, pieces[i]);
    }
    ASSERT_TRUE(tg_complete(st));
    tg_state_free(st);
    fixture_close(&f);
}

/* 3. In a non-accepting state the close is masked — for the sampler and for the
 *    serve's own closer (tg_closing_text / tg_end_allowed). */
static void test_close_masked_when_not_accepting(void) {
    fixture f = fixture_open(FIELD_JSON);
    tg_state *st = tg_state_new_required(f.g, false);
    ASSERT_TRUE(feed_chars(st, READ_HEAD));
    ASSERT_TRUE(feed_chars(st, "x from: a.c"));
    char closing[TG_CLOSING_MAX];
    ASSERT_FALSE(tg_token_allowed(st, "</parameter>"));
    ASSERT_FALSE(tg_token_allowed(st, "\n</parameter>"));
    ASSERT_INT_EQ(tg_closing_text(st, closing, (int)sizeof(closing)), -1);
    ASSERT_FALSE(tg_end_allowed(st));
    ASSERT_TRUE(tg_token_allowed(st, ":7"));
    ASSERT_TRUE(feed_chars(st, ":7"));
    ASSERT_TRUE(tg_closing_text(st, closing, (int)sizeof(closing)) > 0);
    ASSERT_TRUE(tg_end_allowed(st));
    tg_state_free(st);

    ASSERT_FALSE(value_completes(&f, READ_HEAD, "x"));
    ASSERT_FALSE(value_completes(&f, READ_HEAD, "x from: a.c:10,b.h:5"));
    ASSERT_FALSE(value_completes(&f, READ_HEAD, "x from: a b.c:1"));
    ASSERT_FALSE(value_completes(&f, READ_HEAD, "-x"));
    ASSERT_FALSE(value_completes(&f, READ_HEAD, "x from: a.c:10."));
    fixture_close(&f);
}

/* 4. grep cites several ranges. */
static void test_grep_multi_range_closes(void) {
    fixture f = fixture_open(FIELD_JSON);
    ASSERT_TRUE(value_completes(&f, GREP_HEAD, "x from: a.c:10-20, b.h:5"));
    ASSERT_TRUE(value_completes(&f, GREP_HEAD, "x from: a.c:1, b.h:5, c/d.c:7-9"));
    ASSERT_FALSE(value_completes(&f, GREP_HEAD, "x from: a.c:1, "));
    fixture_close(&f);
}

/* 5. A field with no pattern takes any value it took before: bash's finding, the
 *    integer and enum keys the field named, read's path. */
static void test_unpatterned_fields_unchanged(void) {
    fixture f = fixture_open(FIELD_JSON);
    ASSERT_TRUE(value_completes(&f, BASH_HEAD, "anything\nat all"));
    ASSERT_TRUE(value_completes(&f, BASH_HEAD, "x"));
    ASSERT_FALSE(value_completes(&f, BASH_HEAD, " "));   /* the non-empty rule still holds */
    const char *limit = "<tool_call>\n<function=read>\n<parameter=limit>\n12\n</parameter>\n"
                        "<parameter=mode>\nfast\n</parameter>\n<parameter=ref>\nabc\n"
                        "</parameter>\n<parameter=path>\na b\nc\n"
                        "</parameter>\n<parameter=finding>\n";
    ASSERT_TRUE(value_completes(&f, limit, "-"));
    fixture_close(&f);
}

/* 6. The framing: leading and trailing whitespace is not part of the value. */
static void test_whitespace_is_trimmed(void) {
    fixture f = fixture_open(FIELD_JSON);
    ASSERT_TRUE(value_completes(&f, READ_HEAD, "\n  -  \n"));
    ASSERT_TRUE(value_completes(&f, READ_HEAD, " \t x from: a.c:3 \n"));
    fixture_close(&f);
}

/* 7. Bytes that start like the terminator are content until it completes; a
 *    completed terminator judges the value before it. */
static void test_terminator_lookalikes(void) {
    fixture f = fixture_open(FIELD_JSON);
    ASSERT_TRUE(value_completes(&f, READ_HEAD, "a</b> <c from: d.c:2"));
    /* "a<" then "</parameter>": the second '<' restarts the match, and the value
     * it closes is "a<" (matches), not "a" — and "b<" is not "b". */
    ASSERT_TRUE(value_completes(&f, TAG_HEAD, "b"));
    ASSERT_FALSE(value_completes(&f, TAG_HEAD, "a"));
    tg_state *st = walk_to_value(&f, TAG_HEAD);
    ASSERT_TRUE(feed_chars(st, "a<"));
    ASSERT_TRUE(tg_token_allowed(st, "</parameter>"));
    tg_state_free(st);
    st = walk_to_value(&f, TAG_HEAD);
    ASSERT_TRUE(feed_chars(st, "b<"));
    ASSERT_FALSE(tg_token_allowed(st, "</parameter>"));
    tg_state_free(st);
    st = walk_to_value(&f, READ_HEAD);
    ASSERT_TRUE(feed_chars(st, "x from: a.c:1</paramete"));
    ASSERT_TRUE(tg_token_allowed(st, "r>"));
    ASSERT_TRUE(tg_token_allowed(st, "rs"));   /* breaks the match: content again */
    tg_state_free(st);
    st = walk_to_value(&f, READ_HEAD);
    ASSERT_TRUE(feed_chars(st, "x from: a.c</paramete"));
    ASSERT_FALSE(tg_token_allowed(st, "r>"));
    tg_state_free(st);
    fixture_close(&f);
}

/* 8. clone / copy mid-value carry the pattern state (speculation / MTP). */
static void test_clone_and_copy_carry_pattern(void) {
    fixture f = fixture_open(FIELD_JSON);
    tg_state *st = walk_to_value(&f, READ_HEAD);
    ASSERT_TRUE(feed_chars(st, "x from: a.c"));
    tg_state *clone = tg_state_clone(st);
    ASSERT_FALSE(tg_token_allowed(clone, "</parameter>"));
    ASSERT_TRUE(feed_chars(clone, ":1"));
    ASSERT_TRUE(tg_token_allowed(clone, "</parameter>"));
    ASSERT_FALSE(tg_token_allowed(st, "</parameter>"));   /* live walk untouched */

    tg_state *other = walk_to_value(&f, BASH_HEAD);
    tg_state_copy(other, clone);
    ASSERT_TRUE(tg_token_allowed(other, "</parameter>"));
    ASSERT_FALSE(tg_token_allowed(other, "\nmore"));
    ASSERT_TRUE(feed_chars(other, VALUE_TAIL));
    ASSERT_TRUE(tg_complete(other));
    tg_state_free(other);
    tg_state_free(clone);
    tg_state_free(st);
    fixture_close(&f);
}

/* 9. A field that binds nothing leaves every mask byte-identical to no field. */
static void test_noop_field_masks_identical(void) {
    static const char *TOKENS[] = {
        "</function>", "<parameter=", "<parameter=finding>", "path", "\n", " ", "x", "-",
        "</parameter>", "\n</parameter>", "</tool_call>", "<tool_call>", "<function=read>",
        " from", ":", "10", ",", ".", "</think>", "Sure", "</", NULL };
    static const char *FIELDS[] = {
        "null", "{}", "{\"nope\":{\"finding\":\"x\"}}", "{\"read\":{\"nope\":\"x\"}}",
        "{\"read\":{\"finding\":7}}", "{\"read\":{\"limit\":\"x\",\"mode\":\"x\"}}",
        "{\"read\":\"x\"}", "[\"read\"]" };
    char script[512];
    snprintf(script, sizeof(script), "I will read it.</think>\n%sx\ny from a.c%s",
             READ_HEAD, VALUE_TAIL);
    fixture base = fixture_open(NULL);
    int mismatches = 0, checked = 0;
    for (size_t fi = 0; fi < sizeof(FIELDS) / sizeof(FIELDS[0]); fi++) {
        fixture f = fixture_open(FIELDS[fi]);
        for (int mode = 0; mode < 2; mode++) {
            tg_state *a = mode ? tg_state_new(base.g) : tg_state_new_required(base.g, true);
            tg_state *b = mode ? tg_state_new(f.g) : tg_state_new_required(f.g, true);
            char one[2] = { 0, 0 };
            for (const char *p = script; ; p++) {
                for (int t = 0; TOKENS[t]; t++, checked++)
                    if (tg_token_allowed(a, TOKENS[t]) != tg_token_allowed(b, TOKENS[t]))
                        mismatches++;
                if (tg_end_allowed(a) != tg_end_allowed(b)) mismatches++;
                if (!*p) break;
                one[0] = *p;
                tg_advance(a, one);
                tg_advance(b, one);
            }
            ASSERT_TRUE(tg_complete(b));
            tg_state_free(a);
            tg_state_free(b);
        }
        fixture_close(&f);
    }
    ASSERT_INT_EQ(mismatches, 0);
    ASSERT_TRUE(checked > 5000);
    fixture_close(&base);
}

/* 10. A pattern that does not compile fails the apply (the parse refuses it
 *     first; this is the backstop). */
static void test_bad_pattern_fails_apply(void) {
    jr_value *tools = jr_parse(TOOLS_JSON);
    jr_value *field = jr_parse("{\"read\":{\"finding\":\"(ab\"}}");
    tg_grammar *g = tg_build(tools);
    ASSERT_FALSE(tg_apply_field_patterns(g, field));
    tg_free(g);
    jr_free(field);
    jr_free(tools);
}

int main(void) {
    printf("\n=== tool_grammar field_patterns (XML) ===\n\n");
    RUN_TEST(violation_masked_at_first_bad_char);
    RUN_TEST(matching_value_closes);
    RUN_TEST(close_masked_when_not_accepting);
    RUN_TEST(grep_multi_range_closes);
    RUN_TEST(unpatterned_fields_unchanged);
    RUN_TEST(whitespace_is_trimmed);
    RUN_TEST(terminator_lookalikes);
    RUN_TEST(clone_and_copy_carry_pattern);
    RUN_TEST(noop_field_masks_identical);
    RUN_TEST(bad_pattern_fails_apply);
    TEST_REPORT();
    return __ft_failed == 0 ? 0 : 1;
}
