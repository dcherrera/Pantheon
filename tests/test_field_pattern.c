/* SPDX-License-Identifier: AGPL-3.0-or-later */
/* Copyright (C) 2026 David (Corey) Herrera */
/*
 * Unit tests for the field_pattern regex (src/serve/field_pattern.c): every
 * syntax element against a whole-value match table, the compile errors (and the
 * state limit), the client's `finding` pattern, and the prefix-liveness property
 * the grammars rely on (a non-empty state set can still be completed).
 *
 * Pure C, libc-only.
 */

#include "lib/test.h"
#include "../src/serve/field_pattern.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

/* The pattern the client sends on `finding` of read/grep/outline. */
static const char *FINDING_PATTERN =
    "-|[^\\n]+ from: [^\\s,:]+:[0-9]+(-[0-9]+)?(, [^\\s,:]+:[0-9]+(-[0-9]+)?)*";

typedef struct {
    const char *pattern;
    const char *text;
    bool        match;
} match_case;

static bool matches(const char *pattern, const char *text, size_t len) {
    char err[128];
    field_pattern *p = fp_compile(pattern, err, sizeof(err));
    if (!p) { printf("    compile failed: %s: %s\n", pattern, err); return false; }
    bool ok = fp_full_match(p, text, len);
    fp_free(p);
    return ok;
}

static void test_syntax_table(void) {
    static const match_case CASES[] = {
        { "abc", "abc", true }, { "abc", "ab", false }, { "abc", "abcd", false },
        { "", "", true }, { "", "a", false },
        { "a.c", "abc", true }, { "a.c", "a\nc", false }, { "a.c", "a\xc3\xa9", false },
        { "a..", "a\xc3\xa9", true },
        { "[abc]", "b", true }, { "[abc]", "d", false },
        { "[a-c]x", "cx", true }, { "[a-c]x", "dx", false },
        { "[^a-c]", "d", true }, { "[^a-c]", "b", false }, { "[^\\n]", "\n", false },
        { "[^,]+", "\xe2\x82\xac\xc3\xa9", true },
        { "[]a]", "]", true }, { "[^]a]", "]", false }, { "[-a]", "-", true },
        { "[a-]", "-", true }, { "[\\]]", "]", true }, { "[\\-z]", "-", true },
        { "[\\d_]", "7", true }, { "[\\d_]", "_", true }, { "[\\d_]", "a", false },
        { "[\\S]", " ", false }, { "[\\W]", "-", true }, { "[\\W]", "a", false },
        { "\\d", "5", true }, { "\\d", "a", false }, { "\\D", "a", true }, { "\\D", "5", false },
        { "\\s", " ", true }, { "\\s", "\t", true }, { "\\s", "\v", true }, { "\\s", "x", false },
        { "\\S", "x", true }, { "\\S", "\n", false },
        { "\\w", "_", true }, { "\\w", "Z", true }, { "\\w", "-", false },
        { "\\W", "-", true }, { "\\W", "a", false },
        { "a\\nb", "a\nb", true }, { "a\\tb", "a\tb", true }, { "\\r\\f", "\r\f", true },
        { "\\\\", "\\", true }, { "\\.", ".", true }, { "\\.", "a", false },
        { "\\(\\)\\[\\]\\{\\}\\*\\+\\?\\|\\^\\$", "()[]{}*+?|^$", true },
        { "(ab)c", "abc", true }, { "(ab)+", "ababab", true }, { "(ab)+", "aba", false },
        { "a|b", "b", true }, { "a|b", "ab", false }, { "ab|cd|", "", true },
        { "(a|bc)d", "bcd", true }, { "(a|bc)d", "acd", false },
        { "a*", "", true }, { "a*", "aaaa", true }, { "a*", "ab", false },
        { "a+", "", false }, { "a+", "aaa", true },
        { "ab?c", "ac", true }, { "ab?c", "abc", true }, { "ab?c", "abbc", false },
        { "a{3}", "aaa", true }, { "a{3}", "aa", false }, { "a{3}", "aaaa", false },
        { "a{2,}", "a", false }, { "a{2,}", "aaaaa", true },
        { "a{1,3}", "", false }, { "a{1,3}", "aaa", true }, { "a{1,3}", "aaaa", false },
        { "a{0}", "", true }, { "a{0,0}b", "b", true },
        { "(a{2}|b){2}", "aab", true }, { "(a{2}|b){2}", "ab", false },
        { "^abc$", "abc", true }, { "^abc$", "xabc", false }, { "a\\$", "a$", true },
        { "x]}", "x]}", true },
    };
    int failures = 0;
    for (size_t i = 0; i < sizeof(CASES) / sizeof(CASES[0]); i++) {
        const match_case *c = &CASES[i];
        if (matches(c->pattern, c->text, strlen(c->text)) != c->match) {
            printf("    MISMATCH /%s/ on \"%s\": want %d\n", c->pattern, c->text, c->match);
            failures++;
        }
    }
    ASSERT_INT_EQ(failures, 0);
    /* A NUL byte is just a byte. */
    ASSERT_TRUE(matches("a.b", "a\0b", 3));
}

static void test_compile_errors(void) {
    static const char *BAD[] = {
        "(ab", "ab)", "a(b|c", "[abc", "[z-a]", "[a-\\d]", "*a", "+", "a**", "a+?",
        "a{2}{3}", "a{3,2}", "a{", "a{x}", "a{,3}", "a{101}", "a^b", "a$b", "(a$)",
        "\\x41", "\\b", "\\1", "a\\", "[^\\s\\S]", "{2}",
    };
    char err[128];
    int compiled = 0;
    for (size_t i = 0; i < sizeof(BAD) / sizeof(BAD[0]); i++) {
        err[0] = '\0';
        field_pattern *p = fp_compile(BAD[i], err, sizeof(err));
        if (p) { printf("    compiled but should not: /%s/\n", BAD[i]); compiled++; fp_free(p); }
        else if (!err[0]) { printf("    no message for /%s/\n", BAD[i]); compiled++; }
    }
    ASSERT_INT_EQ(compiled, 0);
    ASSERT_NULL(fp_compile(NULL, err, sizeof(err)));
    /* The message names the offending offset. */
    ASSERT_NULL(fp_compile("ab(cd", err, sizeof(err)));
    ASSERT_STR_EQ(err, "unbalanced '(' at byte 2");
    ASSERT_NULL(fp_compile("[z-a]", err, sizeof(err)));
    ASSERT_STR_EQ(err, "bad range in class at byte 1");
    ASSERT_NULL(fp_compile("a+?", err, sizeof(err)));
    ASSERT_STR_EQ(err, "stacked or lazy quantifier at byte 2");
}

static void test_state_limit(void) {
    char err[128];
    char longest[2048];
    /* 100 + 27 atoms + the start state = 128: fits.  One more atom does not. */
    field_pattern *p = fp_compile("a{100}b{27}", err, sizeof(err));
    ASSERT_NOT_NULL(p);
    ASSERT_INT_EQ(fp_state_count(p), FP_MAX_STATES);
    fp_free(p);
    ASSERT_NULL(fp_compile("a{100}b{28}", err, sizeof(err)));
    ASSERT_STR_EQ(err, "pattern needs more than 128 states");
    ASSERT_NULL(fp_compile("(a{50}){50}", err, sizeof(err)));
    /* A pattern of empty repeats makes no states but is still bounded in time. */
    ASSERT_NULL(fp_compile("(((((){100}){100}){100}){100}){100}", err, sizeof(err)));
    ASSERT_STR_EQ(err, "pattern too complex");
    memset(longest, 'a', 1025);
    longest[1025] = '\0';
    ASSERT_NULL(fp_compile(longest, err, sizeof(err)));
}

static void test_finding_pattern(void) {
    static const match_case CASES[] = {
        { NULL, "-", true },
        { NULL, "x from: a.c:10", true },
        { NULL, "x from: a.c:10-20, b.h:5", true },
        { NULL, "the cap is `4096` from: src/io/buf.c:88-91", true },
        { NULL, "x", false },
        { NULL, "x from: a.c", false },
        { NULL, "x from: a.c:10,b.h:5", false },
        { NULL, "x from: a.c:10.", false },
        { NULL, "x from: a b.c:1", false },
        { NULL, "-x", false },
        { NULL, "", false },
        { NULL, "x\ny from: a.c:1", false },
        { NULL, "x from: a.c:1, ", false },
    };
    char err[128];
    field_pattern *p = fp_compile(FINDING_PATTERN, err, sizeof(err));
    ASSERT_NOT_NULL(p);
    if (!p) return;
    ASSERT_INT_EQ(fp_state_count(p), 22);
    int failures = 0;
    for (size_t i = 0; i < sizeof(CASES) / sizeof(CASES[0]); i++)
        if (fp_full_match(p, CASES[i].text, strlen(CASES[i].text)) != CASES[i].match) {
            printf("    MISMATCH finding on \"%s\": want %d\n", CASES[i].text, CASES[i].match);
            failures++;
        }
    ASSERT_INT_EQ(failures, 0);
    fp_free(p);
}

/* Liveness: stepping reports an empty set at exactly the first byte no
 * completion can follow, and a prefix with a live set can still be finished. */
static void test_prefix_liveness(void) {
    char err[128];
    field_pattern *p = fp_compile(FINDING_PATTERN, err, sizeof(err));
    ASSERT_NOT_NULL(p);
    if (!p) return;
    const char *text = "x from: a.c:10\nmore";
    fp_set s = fp_start(p);
    size_t died_at = 0;
    for (size_t i = 0; text[i]; i++)
        if (!fp_step(p, &s, (unsigned char)text[i])) { died_at = i; break; }
    ASSERT_INT_EQ((int)died_at, 14);   /* the '\n' */
    ASSERT_TRUE(fp_set_empty(&s));
    /* "...:10." is not dead: the fact part may itself hold " from: ", so only
     * the close can refuse it. */
    s = fp_start(p);
    for (const char *c = "x from: a.c:10."; *c; c++) ASSERT_TRUE(fp_step(p, &s, (unsigned char)*c));
    ASSERT_FALSE(fp_accepting(p, &s));
    s = fp_start(p);
    for (const char *c = "x from: a.c:10"; *c; c++) fp_step(p, &s, (unsigned char)*c);
    ASSERT_TRUE(fp_accepting(p, &s));
    fp_step(p, &s, ',');
    ASSERT_FALSE(fp_accepting(p, &s));
    ASSERT_FALSE(fp_set_empty(&s));
    for (const char *c = " b.h:5"; *c; c++) fp_step(p, &s, (unsigned char)*c);
    ASSERT_TRUE(fp_accepting(p, &s));
    fp_free(p);
}

/* The forced continuation of `prefix` under `pattern` (fp_forced_continuation),
 * into `out`. */
static int forced_after(const char *pattern, const char *prefix, char *out, int cap) {
    char err[128];
    field_pattern *p = fp_compile(pattern, err, sizeof(err));
    if (!p) { printf("    compile failed: %s: %s\n", pattern, err); out[0] = '\0'; return -1; }
    fp_set s = fp_start(p);
    for (const char *c = prefix; *c; c++) fp_step(p, &s, (unsigned char)*c);
    int n = fp_forced_continuation(p, &s, out, cap);
    fp_free(p);
    return n;
}

/* The steer's literal: the bytes every shortest completion shares, stopping
 * where a class offers more than one byte or the walk accepts. */
static void test_forced_continuation(void) {
    char out[32];
    /* A fact with no from: yet -> " from: " (the path class then offers many). */
    ASSERT_INT_EQ(forced_after(FINDING_PATTERN, "the Makefile builds with gcc", out, 32), 7);
    ASSERT_STR_EQ(out, " from: ");
    /* Part of it already written: only the rest. */
    forced_after(FINDING_PATTERN, "gcc fr", out, 32);
    ASSERT_STR_EQ(out, "om: ");
    /* Trailing space already there. */
    forced_after(FINDING_PATTERN, "gcc ", out, 32);
    ASSERT_STR_EQ(out, "from: ");
    /* A path with no line: the ':' (the digits are a class). */
    forced_after(FINDING_PATTERN, "gcc from: Makefile", out, 32);
    ASSERT_STR_EQ(out, ":");
    /* "...:10." may itself be fact text, so " from: " again. */
    forced_after(FINDING_PATTERN, "x from: a.c:10.", out, 32);
    ASSERT_STR_EQ(out, " from: ");
    /* Accepting: nothing forced. */
    ASSERT_INT_EQ(forced_after(FINDING_PATTERN, "gcc from: Makefile:3", out, 32), 0);
    ASSERT_STR_EQ(out, "");
    ASSERT_INT_EQ(forced_after(FINDING_PATTERN, "-", out, 32), 0);
    /* After "gcc from: " the path class offers many bytes: nothing forced. */
    ASSERT_INT_EQ(forced_after(FINDING_PATTERN, "gcc from: ", out, 32), 0);
    /* The empty prefix: the shortest completion is "-". */
    forced_after(FINDING_PATTERN, "", out, 32);
    ASSERT_STR_EQ(out, "-");
    /* Generic shapes: a literal, a branch point, a counted repeat, a cut. */
    forced_after("abc", "", out, 32);
    ASSERT_STR_EQ(out, "abc");
    forced_after("a(b|c)d", "", out, 32);
    ASSERT_STR_EQ(out, "a");
    forced_after("a(b|c)d", "ab", out, 32);
    ASSERT_STR_EQ(out, "d");
    forced_after("x{3}y", "x", out, 32);
    ASSERT_STR_EQ(out, "xxy");
    /* A longer branch is not forced while a shorter one decides. */
    forced_after("ab|acd", "a", out, 32);
    ASSERT_STR_EQ(out, "b");
    ASSERT_INT_EQ(forced_after("abcdef", "", out, 4), 3);
    ASSERT_STR_EQ(out, "abc");
    /* A dead prefix forces nothing. */
    ASSERT_INT_EQ(forced_after("abc", "x", out, 32), 0);
}

int main(void) {
    printf("\n=== field_pattern ===\n\n");
    RUN_TEST(syntax_table);
    RUN_TEST(compile_errors);
    RUN_TEST(state_limit);
    RUN_TEST(finding_pattern);
    RUN_TEST(prefix_liveness);
    RUN_TEST(forced_continuation);
    TEST_REPORT();
    return __ft_failed == 0 ? 0 : 1;
}
