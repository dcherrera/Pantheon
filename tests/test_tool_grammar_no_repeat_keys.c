/* SPDX-License-Identifier: AGPL-3.0-or-later */
/* Copyright (C) 2026 David (Corey) Herrera */
/*
 * Each parameter key at most once per call, in both tool-call grammars
 * (src/serve/tool_grammar.c, src/serve/tool_grammar_json.c), and the head+tail
 * raw excerpt on the [tool_call_dropped] line (src/serve/tool_raw_excerpt.h).
 *
 * Live, rm4 pbthink 4 (XML): `finding` closed valid, then the model wrote `keep`
 * and `finding` again and again until max_tokens — the grammar let any key of
 * the tool reopen, so once every required key was in nothing forced
 * "</function>".  The JSON dialect refused a repeat only at its closing quote and
 * still took ',' after the last key: a dead end the mask cannot leave.
 *
 * The scripted models here pick, each step, their intended token if the mask
 * allows it and otherwise the first allowed fallback — the same choice a
 * sampler makes when the model's top token is masked.
 *
 * Pure C, libc-only — fully local, no CUDA.
 */

#include "lib/test.h"
#include "../src/serve/tool_grammar.h"
#include "../src/serve/tool_grammar_json.h"
#include "../src/serve/tool_raw_excerpt.h"
#include "../src/serve/json_reader.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

/* `read` as Argus sends it: path + finding required, keep optional, limit an
 * optional integer.  `grep` has keys sharing a prefix (path / pattern), and
 * `with`, whose 'h' sits where path's does: a check of one position at a time
 * would let "pat" + "h" spell a written `path`. */
static const char *TOOLS_JSON =
    "[{\"type\":\"function\",\"function\":{\"name\":\"read\",\"parameters\":{"
    "\"type\":\"object\",\"properties\":{"
    "\"path\":{\"type\":\"string\",\"minLength\":1},"
    "\"finding\":{\"type\":\"string\",\"minLength\":1},"
    "\"keep\":{\"type\":\"string\"},"
    "\"limit\":{\"type\":\"integer\"}},"
    "\"required\":[\"path\",\"finding\"]}}},"
    "{\"type\":\"function\",\"function\":{\"name\":\"grep\",\"parameters\":{"
    "\"type\":\"object\",\"properties\":{"
    "\"pattern\":{\"type\":\"string\",\"minLength\":1},"
    "\"path\":{\"type\":\"string\"},"
    "\"with\":{\"type\":\"string\"}},"
    "\"required\":[\"pattern\"]}}},"
    "{\"type\":\"function\",\"function\":{\"name\":\"noop\",\"parameters\":{"
    "\"type\":\"object\",\"properties\":{}}}}]";

/* ═════════════════════════════ XML dialect ═════════════════════════════ */

static bool xml_feed(tg_state *st, const char *s) {
    char one[2] = { 0, 0 };
    for (const char *p = s; *p; p++) {
        one[0] = *p;
        if (!tg_token_allowed(st, one)) return false;
        tg_advance(st, one);
    }
    return true;
}

static tg_state *xml_walk(const tg_grammar *g, const char *prefix) {
    tg_state *st = tg_state_new(g);
    if (xml_feed(st, prefix)) return st;
    tg_state_free(st);
    return NULL;
}

static const char *XML_READ_ALL =
    "<tool_call>\n<function=read>\n"
    "<parameter=path>\na.c\n</parameter>\n"
    "<parameter=finding>\nx from: a.c:1-100\n</parameter>\n"
    "<parameter=keep>\nyes\n</parameter>\n"
    "<parameter=limit>\n5\n</parameter>\n";

/* 1. A key the call already wrote cannot be opened again; an unwritten optional
 *    key still can, and so can the close. */
static void test_xml_seen_key_not_reopened(void) {
    jr_value *tools = jr_parse(TOOLS_JSON);
    tg_grammar *g = tg_build(tools);
    tg_state *st = xml_walk(g, "<tool_call>\n<function=read>\n"
                               "<parameter=path>\na.c\n</parameter>\n"
                               "<parameter=finding>\nx from: a.c:1-100\n</parameter>\n"
                               "<parameter=keep>\nyes\n</parameter>\n");
    ASSERT_TRUE(st != NULL);
    ASSERT_FALSE(tg_token_allowed(st, "<parameter=keep>"));
    ASSERT_FALSE(tg_token_allowed(st, "<parameter=finding>"));
    ASSERT_FALSE(tg_token_allowed(st, "<parameter=path>"));
    ASSERT_FALSE(tg_token_allowed(st, "<parameter=k"));      /* masked at the first char */
    ASSERT_TRUE(tg_token_allowed(st, "<parameter=limit>"));  /* optional, unwritten */
    ASSERT_TRUE(tg_token_allowed(st, "</function>"));
    tg_state_free(st);
    tg_free(g);
    jr_free(tools);
}

/* 2. Every key written: "<parameter=" is off-grammar, "</function>" the only
 *    way on — and the call completes through it. */
static void test_xml_all_keys_forces_close(void) {
    jr_value *tools = jr_parse(TOOLS_JSON);
    tg_grammar *g = tg_build(tools);
    tg_state *st = xml_walk(g, XML_READ_ALL);
    ASSERT_TRUE(st != NULL);
    ASSERT_TRUE(tg_token_allowed(st, "<"));
    ASSERT_FALSE(tg_token_allowed(st, "<p"));
    ASSERT_FALSE(tg_token_allowed(st, "<parameter="));
    ASSERT_TRUE(tg_token_allowed(st, "</"));
    ASSERT_TRUE(xml_feed(st, "</function>\n</tool_call>"));
    ASSERT_TRUE(tg_complete(st));
    tg_state_free(st);
    tg_free(g);
    jr_free(tools);
}

/* 3. Keys sharing a prefix: with `path` written, "pat" still leads to `pattern`,
 *    but the walk refuses the 'h' that would spell `path` (no dead end at '>'). */
static void test_xml_shared_prefix(void) {
    jr_value *tools = jr_parse(TOOLS_JSON);
    tg_grammar *g = tg_build(tools);
    tg_state *st = xml_walk(g, "<function=grep>\n<parameter=path>\nsrc\n</parameter>\n");
    ASSERT_TRUE(st != NULL);
    ASSERT_TRUE(tg_token_allowed(st, "<parameter=pat"));
    ASSERT_FALSE(tg_token_allowed(st, "<parameter=path"));
    ASSERT_FALSE(tg_token_allowed(st, "<parameter=path>"));
    ASSERT_TRUE(xml_feed(st, "<parameter=pattern>\nfoo\n</parameter>\n</function>"));
    ASSERT_TRUE(tg_complete(st));
    tg_state_free(st);
    tg_free(g);
    jr_free(tools);
}

/* 4. A tool with no parameters: "<parameter=" is refused at once. */
static void test_xml_no_keys(void) {
    jr_value *tools = jr_parse(TOOLS_JSON);
    tg_grammar *g = tg_build(tools);
    tg_state *st = xml_walk(g, "<function=noop>\n");
    ASSERT_TRUE(st != NULL);
    ASSERT_FALSE(tg_token_allowed(st, "<p"));
    ASSERT_TRUE(xml_feed(st, "</function>"));
    ASSERT_TRUE(tg_complete(st));
    tg_state_free(st);
    tg_free(g);
    jr_free(tools);
}

/* The rm4 pbthink 4 model as a ranked vocabulary: it always prefers to write
 * keep, then finding, again; the close ranks last.  Each step takes the
 * highest-ranked token the mask allows — the masked argmax. */
static const char *const XML_RANKED[] = {
    "<parameter=keep>\nyes\n</parameter>\n",
    "<parameter=finding>\nx from: a.c:1-100\n</parameter>\n",
    "<parameter=limit>\n5\n</parameter>\n",
    "</function>\n</tool_call>",
    NULL
};

/* Run the ranked model for at most `max_tokens` steps after a call holding all
 * its required keys; returns the steps it took to complete, or -1 (truncated at
 * max_tokens, or the mask left no token). */
static int xml_scripted_run(const tg_grammar *g, int max_tokens) {
    tg_state *st = xml_walk(g, "<tool_call>\n<function=read>\n"
                               "<parameter=path>\na.c\n</parameter>\n"
                               "<parameter=finding>\nx from: a.c:1-100\n</parameter>\n");
    if (!st) return -1;
    int steps = 0;
    while (!tg_complete(st) && steps < max_tokens) {
        const char *pick = NULL;
        for (int i = 0; !pick && XML_RANKED[i]; i++)
            if (tg_token_allowed(st, XML_RANKED[i])) pick = XML_RANKED[i];
        if (!pick) break;
        tg_advance(st, pick);
        steps++;
    }
    int r = tg_complete(st) ? steps : -1;
    tg_state_free(st);
    return r;
}

/* 5. The live drop, scripted: the call now completes in a few tokens instead of
 *    running to max_tokens. */
static void test_xml_scripted_loop_closes(void) {
    jr_value *tools = jr_parse(TOOLS_JSON);
    tg_grammar *g = tg_build(tools);
    int steps = xml_scripted_run(g, 400);
    ASSERT_TRUE(steps > 0);
    ASSERT_INT_EQ(steps, 3);   /* keep, limit (once each), then the close */
    tg_free(g);
    jr_free(tools);
}

/* ═════════════════════════════ JSON dialect ════════════════════════════ */

static bool json_feed(tgj_state *st, const char *s) {
    char one[2] = { 0, 0 };
    for (const char *p = s; *p; p++) {
        one[0] = *p;
        if (!tgj_token_allowed(st, one)) return false;
        tgj_advance(st, one);
    }
    return true;
}

static tgj_state *json_walk(const tgj_grammar *g, const char *prefix) {
    tgj_state *st = tgj_state_new(g);
    if (json_feed(st, prefix)) return st;
    tgj_state_free(st);
    return NULL;
}

static const char *JSON_READ_HEAD =
    "\n```json\n{\"function\": \"read\", \"arguments\": {\"path\": \"a.c\", "
    "\"finding\": \"x from: a.c:1-100\", \"keep\": \"yes\"";

/* 6. A written key is refused at its first char that no unwritten key shares;
 *    an unwritten optional key is still open. */
static void test_json_seen_key_not_reopened(void) {
    jr_value *tools = jr_parse(TOOLS_JSON);
    tgj_grammar *g = tgj_build(tools);
    tgj_state *st = json_walk(g, JSON_READ_HEAD);
    ASSERT_TRUE(st != NULL);
    ASSERT_TRUE(tgj_token_allowed(st, ", "));
    ASSERT_FALSE(tgj_token_allowed(st, ", \"k"));
    ASSERT_FALSE(tgj_token_allowed(st, ", \"keep\""));
    ASSERT_FALSE(tgj_token_allowed(st, ", \"finding"));
    ASSERT_TRUE(tgj_token_allowed(st, ", \"limit\": 5"));
    ASSERT_TRUE(tgj_token_allowed(st, "}}"));
    tgj_state_free(st);
    tgj_free(g);
    jr_free(tools);
}

/* 7. Every key written: ',' is off-grammar (it led nowhere), '}' closes. */
static void test_json_all_keys_forces_close(void) {
    jr_value *tools = jr_parse(TOOLS_JSON);
    tgj_grammar *g = tgj_build(tools);
    tgj_state *st = json_walk(g, JSON_READ_HEAD);
    ASSERT_TRUE(json_feed(st, ", \"limit\": 5"));
    ASSERT_FALSE(tgj_token_allowed(st, ","));
    ASSERT_FALSE(tgj_token_allowed(st, " ,"));
    ASSERT_TRUE(json_feed(st, "}}\n```"));
    ASSERT_TRUE(tgj_complete(st));
    tgj_state_free(st);
    tgj_free(g);
    jr_free(tools);
}

/* 8. Shared prefix, and a tool with no keys. */
static void test_json_shared_prefix_and_no_keys(void) {
    jr_value *tools = jr_parse(TOOLS_JSON);
    tgj_grammar *g = tgj_build(tools);
    tgj_state *st = json_walk(g, "{\"function\": \"grep\", \"arguments\": {\"path\": \"src\"");
    ASSERT_TRUE(st != NULL);
    ASSERT_TRUE(tgj_token_allowed(st, ", \"pat"));
    ASSERT_FALSE(tgj_token_allowed(st, ", \"path"));
    ASSERT_TRUE(json_feed(st, ", \"pattern\": \"foo\"}}"));
    ASSERT_TRUE(tgj_complete(st));
    tgj_state_free(st);

    st = json_walk(g, "{\"function\": \"noop\", \"arguments\": {");
    ASSERT_TRUE(st != NULL);
    ASSERT_FALSE(tgj_token_allowed(st, "\""));
    ASSERT_TRUE(json_feed(st, "}}"));
    ASSERT_TRUE(tgj_complete(st));
    tgj_state_free(st);
    tgj_free(g);
    jr_free(tools);
}

/* The same model in JSON, in fine tokens (a key split from its quotes), so a
 * repeated key is spelled one piece at a time. */
static const char *const JSON_RANKED[] = {
    ",", " \"yes\"", " 5", " \"", "keep", "finding", "limit", "\":", "}", "\n```",
    NULL
};

static int json_scripted_run(const tgj_grammar *g, int max_tokens) {
    tgj_state *st = json_walk(g, "\n```json\n{\"function\": \"read\", \"arguments\": "
                                 "{\"path\": \"a.c\", \"finding\": \"x from: a.c:1-100\"");
    if (!st) return -1;
    int steps = 0;
    while (!tgj_complete(st) && steps < max_tokens) {
        const char *pick = NULL;
        for (int i = 0; !pick && JSON_RANKED[i]; i++)
            if (tgj_token_allowed(st, JSON_RANKED[i])) pick = JSON_RANKED[i];
        if (!pick) break;
        tgj_advance(st, pick);
        steps++;
    }
    int r = tgj_complete(st) ? steps : -1;
    tgj_state_free(st);
    return r;
}

/* 9. Scripted: the JSON call completes instead of walking into a dead end. */
static void test_json_scripted_loop_closes(void) {
    jr_value *tools = jr_parse(TOOLS_JSON);
    tgj_grammar *g = tgj_build(tools);
    int steps = json_scripted_run(g, 400);
    ASSERT_INT_EQ(steps, 13);   /* keep, limit, then "}}" and the fence */
    tgj_free(g);
    jr_free(tools);
}

/* ═════════════════════════ dropped-call raw excerpt ═════════════════════ */

static char *excerpt(const char *raw, size_t head, size_t tail) {
    char *buf = NULL;
    size_t len = 0;
    FILE *f = open_memstream(&buf, &len);
    if (!f) return NULL;
    tool_raw_excerpt_print(f, raw, head, tail);
    fclose(f);
    return buf;
}

/* 10. Short text: whole, no tail field.  Long text: head and tail, the tail being
 *     the LAST bytes; never overlapping the head; tail start on a UTF-8 boundary. */
static void test_raw_excerpt(void) {
    char *s = excerpt("abc", 8, 8);
    ASSERT_STR_EQ(s, "raw_len=3 raw=<<<abc>>>");
    free(s);
    s = excerpt("12345678", 8, 8);
    ASSERT_STR_EQ(s, "raw_len=8 raw=<<<12345678>>>");
    free(s);

    s = excerpt("HEADHEAD-middle-TAILTAIL", 8, 8);
    ASSERT_STR_EQ(s, "raw_len=24 raw=<<<HEADHEAD…[truncated]>>> raw_tail=<<<TAILTAIL>>>");
    free(s);

    s = excerpt("HEADHEADxyz", 8, 8);   /* tail would overlap the head: rest only */
    ASSERT_STR_EQ(s, "raw_len=11 raw=<<<HEADHEAD…[truncated]>>> raw_tail=<<<xyz>>>");
    free(s);

    /* "é" is 2 bytes; a tail of 3 would start on its continuation byte. */
    s = excerpt("HEADHEAD----\xc3\xa9zz", 8, 3);
    ASSERT_STR_EQ(s, "raw_len=16 raw=<<<HEADHEAD…[truncated]>>> raw_tail=<<<zz>>>");
    free(s);

    /* The live sizes: a 5000-byte loop shows its first and last 800. */
    char big[5001];
    memset(big, 'k', 5000);
    memcpy(big, "START", 5);
    memcpy(big + 4995, "ENDED", 5);
    big[5000] = '\0';
    s = excerpt(big, TOOL_RAW_EXCERPT_HEAD, TOOL_RAW_EXCERPT_TAIL);
    ASSERT_TRUE(s && strncmp(s, "raw_len=5000 raw=<<<START", 25) == 0);
    const char *tail = s ? strstr(s, "raw_tail=<<<") : NULL;
    ASSERT_TRUE(tail != NULL);
    ASSERT_INT_EQ(tail ? (int)strlen(tail) : 0, (int)strlen("raw_tail=<<<>>>") + 800);
    ASSERT_TRUE(tail && strcmp(tail + strlen(tail) - 8, "ENDED>>>") == 0);
    free(s);
}

int main(void) {
    printf("\n=== tool grammar: each key once per call ===\n\n");
    RUN_TEST(xml_seen_key_not_reopened);
    RUN_TEST(xml_all_keys_forces_close);
    RUN_TEST(xml_shared_prefix);
    RUN_TEST(xml_no_keys);
    RUN_TEST(xml_scripted_loop_closes);
    RUN_TEST(json_seen_key_not_reopened);
    RUN_TEST(json_all_keys_forces_close);
    RUN_TEST(json_shared_prefix_and_no_keys);
    RUN_TEST(json_scripted_loop_closes);
    RUN_TEST(raw_excerpt);
    TEST_REPORT();
    return __ft_failed == 0 ? 0 : 1;
}
