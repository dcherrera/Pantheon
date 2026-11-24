/* SPDX-License-Identifier: AGPL-3.0-or-later */
/* Copyright (C) 2026 David (Corey) Herrera */
/*
 * Unit tests for `required_if_reasoned` in the JSON tool-call grammar
 * (tgj_require_if_reasoned, tgj_watch_reasoning; src/serve/tool_grammar_json.c).
 * The XML twin, and the tracker's own "first call" rule, are in
 * tests/test_tool_grammar_reasoned.c.
 *
 * A listed key is required on the reply's FIRST call when the reply REASONED
 * (its think block held a non-whitespace char before its close); otherwise it is
 * optional.  Same feed style as tests/test_tool_grammar_json.c.
 *
 * Pure C, libc-only — fully local, no CUDA.
 */

#include "lib/test.h"
#include "../src/serve/tool_grammar_json.h"
#include "../src/serve/json_reader.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

/* read(conclusion, path REQUIRED) and glob(conclusion, pattern REQUIRED), minLength
 * 1 on every key. */
static const char *TOOLS_JSON =
    "[{\"type\":\"function\",\"function\":{\"name\":\"read\","
    "\"parameters\":{\"type\":\"object\",\"properties\":{"
    "\"conclusion\":{\"type\":\"string\",\"minLength\":1},"
    "\"path\":{\"type\":\"string\",\"minLength\":1}},"
    "\"required\":[\"path\"]}}},"
    "{\"type\":\"function\",\"function\":{\"name\":\"glob\","
    "\"parameters\":{\"type\":\"object\",\"properties\":{"
    "\"conclusion\":{\"type\":\"string\",\"minLength\":1},"
    "\"pattern\":{\"type\":\"string\",\"minLength\":1}},"
    "\"required\":[\"pattern\"]}}}]";

static const char *FIELD_JSON = "[\"conclusion\",\"no_such_key\"]";

static const char *CALL_HEAD =
    "\n```json\n{\"function\": \"read\", \"arguments\": {\"path\": \"src/hdr.c\"";
static const char *CONCLUSION_ARG = ", \"conclusion\": \"the length check reads 4 bytes\"";
static const char *CALL_TAIL = "}}\n```";

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
        tgj_require_if_reasoned(f.g, f.field);
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

/* After CALL_HEAD, may the arguments object close now (without the conclusion)? */
static bool can_close_without_key(tgj_state *st) {
    char closing[16];
    return tgj_token_allowed(st, "}") &&
           tgj_closing_text(st, closing, (int)sizeof(closing)) > 0;
}

/* 1. Reasoned reply, first call: '}' is masked without the key; with it, closes. */
static void test_reasoned_first_call_owes_key(void) {
    fixture f = fixture_open(FIELD_JSON);
    tgj_state *st = tgj_state_new_required(f.g, true);
    tgj_set_think_close_token(st);
    tgj_advance(st, "The header says 8 bytes.");
    tgj_advance(st, "</think>");
    ASSERT_TRUE(tgj_reply_reasoned(st));
    ASSERT_TRUE(feed_chars(st, CALL_HEAD));
    ASSERT_FALSE(can_close_without_key(st));
    ASSERT_FALSE(tgj_end_allowed(st));
    ASSERT_TRUE(tgj_token_allowed(st, ", \"conclusion\""));
    tgj_state *empty = tgj_state_clone(st);
    ASSERT_TRUE(feed_chars(empty, ", \"conclusion\": \" "));
    ASSERT_FALSE(tgj_token_allowed(empty, "\""));   /* non-empty rule holds */
    tgj_state_free(empty);
    ASSERT_TRUE(feed_chars(st, CONCLUSION_ARG));
    ASSERT_TRUE(feed_chars(st, CALL_TAIL));
    ASSERT_TRUE(tgj_complete(st));
    ASSERT_INT_EQ(tgj_calls_named(st), 1);
    tgj_state_free(st);
    fixture_close(&f);
}

/* 2. The counter: 0 before the NAME, 1 once the call reached it and after it
 *    closed (the serve stops at the first complete call). */
static void test_call_counter(void) {
    fixture f = fixture_open(FIELD_JSON);
    tgj_state *st = tgj_state_new_required(f.g, true);
    ASSERT_TRUE(feed_chars(st, "thinking</think>"));
    ASSERT_INT_EQ(tgj_calls_named(st), 0);
    ASSERT_TRUE(feed_chars(st, CALL_HEAD));
    ASSERT_INT_EQ(tgj_calls_named(st), 1);
    ASSERT_TRUE(feed_chars(st, CONCLUSION_ARG));
    ASSERT_TRUE(feed_chars(st, CALL_TAIL));
    ASSERT_TRUE(tgj_complete(st));
    ASSERT_INT_EQ(tgj_calls_named(st), 1);
    tgj_state_free(st);
    fixture_close(&f);
}

/* 3. Unreasoned reply (whitespace-only think, text close or close token). */
static void test_unreasoned_reply_closes_without_key(void) {
    fixture f = fixture_open(FIELD_JSON);
    tgj_state *text = tgj_state_new_required(f.g, true);
    ASSERT_TRUE(feed_chars(text, "\n\n</think>\n\n"));
    ASSERT_FALSE(tgj_reply_reasoned(text));
    ASSERT_TRUE(feed_chars(text, CALL_HEAD));
    ASSERT_TRUE(can_close_without_key(text));
    ASSERT_TRUE(feed_chars(text, CALL_TAIL));
    ASSERT_TRUE(tgj_complete(text));

    tgj_state *token = tgj_state_new_required(f.g, true);
    tgj_set_think_close_token(token);
    tgj_advance(token, " \n\t\n");
    tgj_advance(token, "</think>");
    ASSERT_FALSE(tgj_reply_reasoned(token));
    ASSERT_TRUE(feed_chars(token, CALL_HEAD));
    ASSERT_TRUE(can_close_without_key(token));
    /* A partial close broken off by whitespace was text. */
    tgj_state *partial = tgj_state_new_required(f.g, true);
    ASSERT_TRUE(feed_chars(partial, "\n</thi"));
    ASSERT_FALSE(tgj_reply_reasoned(partial));
    ASSERT_TRUE(feed_chars(partial, "\n</think>"));
    ASSERT_TRUE(tgj_reply_reasoned(partial));
    tgj_state_free(partial);
    tgj_state_free(token);
    tgj_state_free(text);
    fixture_close(&f);
}

/* 4. No think block opened: never reasoned. */
static void test_no_think_block_closes_without_key(void) {
    fixture f = fixture_open(FIELD_JSON);
    tgj_state *req = tgj_state_new_required(f.g, false);
    ASSERT_TRUE(feed_chars(req, CALL_HEAD));
    ASSERT_FALSE(tgj_reply_reasoned(req));
    ASSERT_TRUE(can_close_without_key(req));

    tgj_state *autost = tgj_state_new(f.g);
    ASSERT_TRUE(feed_chars(autost, "Let me read the header first."));
    ASSERT_TRUE(feed_chars(autost, CALL_HEAD));
    ASSERT_FALSE(tgj_reply_reasoned(autost));
    ASSERT_TRUE(can_close_without_key(autost));
    tgj_state_free(autost);
    tgj_state_free(req);
    fixture_close(&f);
}

/* 5. AUTO with an open think block: reasoned → owed; whitespace-only → not; an
 *    opener ('{' or the fence) straight inside the block is the call. */
static void test_auto_watched_think_block(void) {
    fixture f = fixture_open(FIELD_JSON);
    tgj_state *reasoned = tgj_state_new(f.g);
    tgj_watch_reasoning(reasoned);
    ASSERT_TRUE(feed_chars(reasoned, "Check the header.</think>\n\nReading it."));
    ASSERT_TRUE(tgj_reply_reasoned(reasoned));
    ASSERT_TRUE(feed_chars(reasoned, CALL_HEAD));
    ASSERT_FALSE(can_close_without_key(reasoned));
    ASSERT_TRUE(feed_chars(reasoned, CONCLUSION_ARG));
    ASSERT_TRUE(feed_chars(reasoned, CALL_TAIL));
    ASSERT_TRUE(tgj_complete(reasoned));

    tgj_state *empty = tgj_state_new(f.g);
    tgj_watch_reasoning(empty);
    ASSERT_TRUE(feed_chars(empty, "\n\n</think>\n\nI will read it."));
    ASSERT_FALSE(tgj_reply_reasoned(empty));
    ASSERT_TRUE(feed_chars(empty, CALL_HEAD));
    ASSERT_TRUE(can_close_without_key(empty));

    /* The close token is not reasoning, even where AUTO walks its chars. */
    tgj_state *token = tgj_state_new(f.g);
    tgj_set_think_close_token(token);
    tgj_watch_reasoning(token);
    tgj_advance(token, "\n\n");
    tgj_advance(token, "</think>");
    ASSERT_TRUE(feed_chars(token, "\n\nI will read it."));
    ASSERT_FALSE(tgj_reply_reasoned(token));
    tgj_state_free(token);

    tgj_state *fence = tgj_state_new(f.g);
    tgj_watch_reasoning(fence);
    ASSERT_TRUE(feed_chars(fence, CALL_HEAD));
    ASSERT_FALSE(tgj_reply_reasoned(fence));
    ASSERT_TRUE(can_close_without_key(fence));
    tgj_state *brace = tgj_state_new(f.g);
    tgj_watch_reasoning(brace);
    ASSERT_TRUE(feed_chars(brace, "\n{\"function\": \"read\", \"arguments\": {\"path\": \"a\""));
    ASSERT_FALSE(tgj_reply_reasoned(brace));
    ASSERT_TRUE(can_close_without_key(brace));
    tgj_state_free(brace);
    tgj_state_free(fence);
    tgj_state_free(empty);
    tgj_state_free(reasoned);
    fixture_close(&f);
}

/* A reasoning-end call (the serve's budget close) with text before it = reasoned. */
static void test_force_closed_reasoning_is_reasoned(void) {
    fixture f = fixture_open(FIELD_JSON);
    tgj_state *st = tgj_state_new_required(f.g, true);
    ASSERT_TRUE(feed_chars(st, " </thi"));
    ASSERT_FALSE(tgj_reply_reasoned(st));
    tgj_reasoning_end(st);
    ASSERT_TRUE(tgj_reply_reasoned(st));
    tgj_state_free(st);
    fixture_close(&f);
}

/* 6. No field / empty / unknown-only / non-array: masks byte-identical. */
static void test_absent_field_masks_identical(void) {
    static const char *TOKENS[] = {
        "}", ",", ", \"conclusion\"", "\"", " ", "\n", "```", "```json", "{", "x",
        "</think>", "Sure", "}}\n```", NULL };
    static const char *FIELDS[] = { "null", "[]", "[\"no_such_key\"]", "{\"a\":1}", "[7]" };
    char script[512];
    snprintf(script, sizeof(script), "I should read it.</think>\n%s%s", CALL_HEAD, CALL_TAIL);
    fixture base = fixture_open(NULL);
    int mismatches = 0, checked = 0;
    for (int fi = 0; fi < 5; fi++) {
        fixture f = fixture_open(FIELDS[fi]);
        for (int mode = 0; mode < 2; mode++) {
            tgj_state *a = mode ? tgj_state_new(base.g) : tgj_state_new_required(base.g, true);
            tgj_state *b = mode ? tgj_state_new(f.g) : tgj_state_new_required(f.g, true);
            if (mode) tgj_watch_reasoning(b);
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
    ASSERT_TRUE(checked > 1000);
    fixture_close(&base);
}

/* 7. clone / copy mid-reasoning carry `reasoned` and the call counter. */
static void test_clone_and_copy_carry_state(void) {
    fixture f = fixture_open(FIELD_JSON);
    tgj_state *st = tgj_state_new_required(f.g, true);
    ASSERT_TRUE(feed_chars(st, "  "));
    tgj_state *before = tgj_state_clone(st);
    ASSERT_TRUE(feed_chars(st, "The offset is wrong."));
    tgj_state *mid = tgj_state_clone(st);
    ASSERT_TRUE(tgj_reply_reasoned(mid));
    ASSERT_FALSE(tgj_reply_reasoned(before));
    ASSERT_TRUE(feed_chars(mid, "</think>"));
    ASSERT_TRUE(feed_chars(mid, CALL_HEAD));
    ASSERT_FALSE(can_close_without_key(mid));
    ASSERT_TRUE(feed_chars(before, "</think>"));
    ASSERT_TRUE(feed_chars(before, CALL_HEAD));
    ASSERT_TRUE(can_close_without_key(before));

    tgj_state_copy(st, mid);
    ASSERT_TRUE(tgj_reply_reasoned(st));
    ASSERT_INT_EQ(tgj_calls_named(st), 1);
    ASSERT_FALSE(can_close_without_key(st));
    tgj_state_free(mid);
    tgj_state_free(before);
    tgj_state_free(st);
    fixture_close(&f);
}

int main(void) {
    printf("\n=== tool_grammar_json required_if_reasoned ===\n\n");
    RUN_TEST(reasoned_first_call_owes_key);
    RUN_TEST(call_counter);
    RUN_TEST(unreasoned_reply_closes_without_key);
    RUN_TEST(no_think_block_closes_without_key);
    RUN_TEST(auto_watched_think_block);
    RUN_TEST(force_closed_reasoning_is_reasoned);
    RUN_TEST(absent_field_masks_identical);
    RUN_TEST(clone_and_copy_carry_state);
    TEST_REPORT();
    return __ft_failed == 0 ? 0 : 1;
}
