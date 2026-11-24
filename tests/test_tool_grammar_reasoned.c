/* SPDX-License-Identifier: AGPL-3.0-or-later */
/* Copyright (C) 2026 David (Corey) Herrera */
/*
 * Unit tests for `required_if_reasoned` in the XML tool-call grammar
 * (tg_require_if_reasoned, tg_watch_reasoning; src/serve/tool_grammar.c) and the
 * shared tracker (src/serve/reasoned_reply_tracker.h).
 *
 * A listed key is required on the reply's FIRST call when the reply REASONED
 * (its think block held a non-whitespace char before its close); otherwise it is
 * optional.  Same feed style as tests/test_tool_grammar.c: whole token pieces,
 * the "</think>" close token, and char-by-char feeds.
 *
 * Pure C, libc-only — fully local, no CUDA.
 */

#include "lib/test.h"
#include "../src/serve/tool_grammar.h"
#include "../src/serve/reasoned_reply_tracker.h"
#include "../src/serve/json_reader.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

/* read(conclusion, path REQUIRED) and glob(conclusion, pattern REQUIRED), with the
 * client's minLength 1 on every key, so the non-empty rule applies to the
 * reasoned-only key exactly as to a `required` one. */
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

/* The request field as the client sends it, plus an unknown name (ignored). */
static const char *FIELD_JSON = "[\"conclusion\",\"no_such_key\"]";

static const char *CALL_HEAD = "\n<tool_call>\n<function=read>\n"
                               "<parameter=path>\nsrc/hdr.c\n</parameter>\n";
static const char *CONCLUSION_PARAM =
    "<parameter=conclusion>\nthe length check reads 4 bytes\n</parameter>\n";
static const char *CALL_TAIL = "</function>\n</tool_call>";

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
        tg_require_if_reasoned(f.g, f.field);
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

/* After CALL_HEAD, may the call close now (without the conclusion)? */
static bool can_close_without_key(tg_state *st) {
    char closing[TG_CLOSING_MAX];
    return tg_token_allowed(st, "</function>") &&
           tg_closing_text(st, closing, (int)sizeof(closing)) > 0;
}

/* 1. Reasoned reply, first call: without the key the close is masked (and the
 *    serve cannot close it either); with the key it closes. */
static void test_reasoned_first_call_owes_key(void) {
    fixture f = fixture_open(FIELD_JSON);
    tg_state *st = tg_state_new_required(f.g, true);
    tg_set_think_close_token(st);
    tg_advance(st, "The header says 8 bytes.");
    tg_advance(st, "</think>");
    ASSERT_TRUE(tg_reply_reasoned(st));
    ASSERT_TRUE(feed_chars(st, CALL_HEAD));
    ASSERT_FALSE(can_close_without_key(st));
    ASSERT_FALSE(tg_end_allowed(st));
    ASSERT_TRUE(tg_token_allowed(st, "<parameter=conclusion>"));
    /* The existing non-empty rule holds for the reasoned-only key too. */
    tg_state *empty = tg_state_clone(st);
    ASSERT_TRUE(feed_chars(empty, "<parameter=conclusion>\n \n"));
    ASSERT_FALSE(tg_token_allowed(empty, "</parameter>"));
    tg_state_free(empty);
    ASSERT_TRUE(feed_chars(st, CONCLUSION_PARAM));
    ASSERT_TRUE(feed_chars(st, CALL_TAIL));
    ASSERT_TRUE(tg_complete(st));
    ASSERT_INT_EQ(tg_calls_named(st), 1);
    tg_state_free(st);
    fixture_close(&f);
}

/* 2. Later calls of the same reply do not owe the key.  The grammar walks one call
 *    per reply (the serve stops at the first complete call), so the "first call"
 *    rule is checked on the tracker, and the walk's counter is checked to be 1
 *    once its call closed. */
static void test_second_call_does_not_owe_key(void) {
    reasoned_reply r = { 0 };
    reasoned_reply_begin(&r);
    reasoned_reply_char(&r, 'x', false);
    ASSERT_TRUE(reasoned_reply_call_named(&r));
    ASSERT_FALSE(reasoned_reply_call_named(&r));
    ASSERT_FALSE(reasoned_reply_call_named(&r));
    ASSERT_INT_EQ(r.calls_named, 3);

    fixture f = fixture_open(FIELD_JSON);
    tg_state *st = tg_state_new_required(f.g, true);
    ASSERT_TRUE(feed_chars(st, "thinking</think>"));
    ASSERT_INT_EQ(tg_calls_named(st), 0);
    ASSERT_TRUE(feed_chars(st, CALL_HEAD));
    ASSERT_INT_EQ(tg_calls_named(st), 1);
    ASSERT_TRUE(feed_chars(st, CONCLUSION_PARAM));
    ASSERT_TRUE(feed_chars(st, CALL_TAIL));
    ASSERT_TRUE(tg_complete(st));
    ASSERT_INT_EQ(tg_calls_named(st), 1);
    tg_state_free(st);
    fixture_close(&f);
}

/* 3. Unreasoned reply (empty / whitespace-only think): the first call closes
 *    without the key — text close, close token, and a spelled close in token mode
 *    (which IS reasoning: its text is not the close). */
static void test_unreasoned_reply_closes_without_key(void) {
    fixture f = fixture_open(FIELD_JSON);
    tg_state *text = tg_state_new_required(f.g, true);
    ASSERT_TRUE(feed_chars(text, "\n\n</think>\n\n"));
    ASSERT_FALSE(tg_reply_reasoned(text));
    ASSERT_TRUE(feed_chars(text, CALL_HEAD));
    ASSERT_TRUE(can_close_without_key(text));
    ASSERT_TRUE(feed_chars(text, CALL_TAIL));
    ASSERT_TRUE(tg_complete(text));

    tg_state *token = tg_state_new_required(f.g, true);
    tg_set_think_close_token(token);
    tg_advance(token, " \n\t\r\n");
    tg_advance(token, "</think>");
    ASSERT_FALSE(tg_reply_reasoned(token));
    ASSERT_TRUE(feed_chars(token, CALL_HEAD));
    ASSERT_TRUE(can_close_without_key(token));

    tg_state *spelled = tg_state_new_required(f.g, true);
    tg_set_think_close_token(spelled);
    tg_advance(spelled, "\n</");
    tg_advance(spelled, "think>");
    ASSERT_TRUE(tg_reply_reasoned(spelled));

    /* A partial close that breaks off was text. */
    tg_state *partial = tg_state_new_required(f.g, true);
    ASSERT_TRUE(feed_chars(partial, "\n</thi"));
    ASSERT_FALSE(tg_reply_reasoned(partial));
    ASSERT_TRUE(feed_chars(partial, "\n</think>"));   /* broken by whitespace */
    ASSERT_TRUE(tg_reply_reasoned(partial));
    tg_state_free(partial);
    tg_state_free(spelled);
    tg_state_free(token);
    tg_state_free(text);
    fixture_close(&f);
}

/* 4. No think block opened: never reasoned, the first call closes without it. */
static void test_no_think_block_closes_without_key(void) {
    fixture f = fixture_open(FIELD_JSON);
    tg_state *req = tg_state_new_required(f.g, false);
    ASSERT_TRUE(feed_chars(req, CALL_HEAD));
    ASSERT_FALSE(tg_reply_reasoned(req));
    ASSERT_TRUE(can_close_without_key(req));

    /* AUTO without tg_watch_reasoning: prose before the call is not reasoning. */
    tg_state *autost = tg_state_new(f.g);
    ASSERT_TRUE(feed_chars(autost, "Let me read the header first."));
    ASSERT_TRUE(feed_chars(autost, CALL_HEAD));
    ASSERT_FALSE(tg_reply_reasoned(autost));
    ASSERT_TRUE(can_close_without_key(autost));
    tg_state_free(autost);
    tg_state_free(req);
    fixture_close(&f);
}

/* 5. AUTO with an open think block (tg_watch_reasoning): reasoned → owed;
 *    whitespace-only → not; an opener written straight inside the think block is
 *    the call, not reasoning. */
static void test_auto_watched_think_block(void) {
    fixture f = fixture_open(FIELD_JSON);
    tg_state *reasoned = tg_state_new(f.g);
    tg_watch_reasoning(reasoned);
    ASSERT_TRUE(feed_chars(reasoned, "Check the header.</think>\n\nReading it."));
    ASSERT_TRUE(tg_reply_reasoned(reasoned));
    ASSERT_TRUE(feed_chars(reasoned, CALL_HEAD));
    ASSERT_FALSE(can_close_without_key(reasoned));
    ASSERT_TRUE(feed_chars(reasoned, CONCLUSION_PARAM));
    ASSERT_TRUE(feed_chars(reasoned, CALL_TAIL));
    ASSERT_TRUE(tg_complete(reasoned));

    tg_state *empty = tg_state_new(f.g);
    tg_watch_reasoning(empty);
    ASSERT_TRUE(feed_chars(empty, "\n\n</think>\n\nI will read it."));
    ASSERT_FALSE(tg_reply_reasoned(empty));
    ASSERT_TRUE(feed_chars(empty, CALL_HEAD));
    ASSERT_TRUE(can_close_without_key(empty));

    /* The close token is not reasoning, even where AUTO walks its chars. */
    tg_state *token = tg_state_new(f.g);
    tg_set_think_close_token(token);
    tg_watch_reasoning(token);
    tg_advance(token, "\n\n");
    tg_advance(token, "</think>");
    ASSERT_TRUE(feed_chars(token, "\n\nI will read it."));
    ASSERT_FALSE(tg_reply_reasoned(token));
    tg_state_free(token);

    tg_state *opener = tg_state_new(f.g);
    tg_watch_reasoning(opener);
    ASSERT_TRUE(feed_chars(opener, "\n<x"));          /* a broken '<' is text */
    ASSERT_TRUE(tg_reply_reasoned(opener));
    tg_state *direct = tg_state_new(f.g);
    tg_watch_reasoning(direct);
    ASSERT_TRUE(feed_chars(direct, CALL_HEAD));
    ASSERT_FALSE(tg_reply_reasoned(direct));
    ASSERT_TRUE(can_close_without_key(direct));
    tg_state_free(direct);
    tg_state_free(opener);
    tg_state_free(empty);
    tg_state_free(reasoned);
    fixture_close(&f);
}

/* A budget force-close with text in the block = reasoned. */
static void test_force_closed_reasoning_is_reasoned(void) {
    fixture f = fixture_open(FIELD_JSON);
    tg_state *st = tg_state_new_required(f.g, true);
    ASSERT_TRUE(feed_chars(st, "still working out the off"));
    tg_advance(st, "</think>");
    tg_reasoning_end_injected(st);
    ASSERT_TRUE(tg_reply_reasoned(st));
    tg_state *ended = tg_state_new_required(f.g, true);
    ASSERT_TRUE(feed_chars(ended, " </thi"));
    ASSERT_FALSE(tg_reply_reasoned(ended));
    tg_reasoning_end_injected(ended);   /* close never walked (no close token) */
    ASSERT_TRUE(tg_reply_reasoned(ended));
    tg_state_free(ended);
    tg_state_free(st);
    fixture_close(&f);
}

/* 6. No field / empty field / only unknown names: masks byte-identical to a
 *    grammar that never heard of the field, on a reasoned stream. */
static void test_absent_field_masks_identical(void) {
    static const char *TOKENS[] = {
        "</function>", "<parameter=", "<parameter=conclusion>", "path", "\n", " ",
        "</parameter>", "</tool_call>", "<tool_call>", "<function=read>", "x", "</think>",
        "Sure", NULL };
    static const char *FIELDS[] = { NULL, "[]", "[\"no_such_key\"]", "{\"a\":1}", "[7]" };
    char script[512];
    snprintf(script, sizeof(script), "I should read it.</think>\n%s%s", CALL_HEAD, CALL_TAIL);
    fixture base = fixture_open(NULL);
    int mismatches = 0, checked = 0;
    for (int fi = 0; fi < 5; fi++) {
        fixture f = fixture_open(FIELDS[fi] ? FIELDS[fi] : "null");
        for (int mode = 0; mode < 2; mode++) {
            tg_state *a = mode ? tg_state_new(base.g) : tg_state_new_required(base.g, true);
            tg_state *b = mode ? tg_state_new(f.g) : tg_state_new_required(f.g, true);
            if (mode) tg_watch_reasoning(b);
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
    ASSERT_TRUE(checked > 1000);
    fixture_close(&base);
}

/* 7. clone / copy mid-reasoning carry `reasoned` and the call counter. */
static void test_clone_and_copy_carry_state(void) {
    fixture f = fixture_open(FIELD_JSON);
    tg_state *st = tg_state_new_required(f.g, true);
    ASSERT_TRUE(feed_chars(st, "  "));
    tg_state *before = tg_state_clone(st);
    ASSERT_TRUE(feed_chars(st, "The offset is wrong."));
    tg_state *mid = tg_state_clone(st);
    ASSERT_TRUE(tg_reply_reasoned(mid));
    ASSERT_FALSE(tg_reply_reasoned(before));
    ASSERT_TRUE(feed_chars(mid, "</think>"));
    ASSERT_TRUE(feed_chars(mid, CALL_HEAD));
    ASSERT_FALSE(can_close_without_key(mid));
    ASSERT_TRUE(feed_chars(before, "</think>"));
    ASSERT_TRUE(feed_chars(before, CALL_HEAD));
    ASSERT_TRUE(can_close_without_key(before));

    tg_state_copy(st, mid);
    ASSERT_TRUE(tg_reply_reasoned(st));
    ASSERT_INT_EQ(tg_calls_named(st), 1);
    ASSERT_FALSE(can_close_without_key(st));
    tg_state_free(mid);
    tg_state_free(before);
    tg_state_free(st);
    fixture_close(&f);
}

int main(void) {
    printf("\n=== tool_grammar required_if_reasoned (XML) ===\n\n");
    RUN_TEST(reasoned_first_call_owes_key);
    RUN_TEST(second_call_does_not_owe_key);
    RUN_TEST(unreasoned_reply_closes_without_key);
    RUN_TEST(no_think_block_closes_without_key);
    RUN_TEST(auto_watched_think_block);
    RUN_TEST(force_closed_reasoning_is_reasoned);
    RUN_TEST(absent_field_masks_identical);
    RUN_TEST(clone_and_copy_carry_state);
    TEST_REPORT();
    return __ft_failed == 0 ? 0 : 1;
}
