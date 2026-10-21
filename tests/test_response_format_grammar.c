/* SPDX-License-Identifier: AGPL-3.0-or-later */
/* Copyright (C) 2026 David (Corey) Herrera */
/**
 * @file test_response_format_grammar.c
 * @brief Unit tests for the `response_format` decoding grammar
 *        (src/serve/response_format_grammar.c).
 *
 * Drives the FSM exactly the way the constrained sampler does — ask
 * rfg_token_allowed() for a candidate, and only rfg_advance() when it said yes —
 * so a test that passes here is a statement about what the sampler can emit, not
 * about a parser run after the fact.
 *
 * @status Covers json_object: acceptance of valid JSON, rejection of prose and
 *         of every malformed shape a model actually produces, the reasoning
 *         grace, the mask gating, and multi-char token boundaries.
 * @issues None
 * @todo None
 */

#include <stdio.h>
#include <string.h>

#include "lib/test.h"
#include "../src/serve/response_format_grammar.h"

/* Offer `text` as one candidate token: allowed → advance (what the sampler
 * does), rejected → the live state is untouched (what the mask does). */
static bool feed_token(rfg_state *st, const char *text)
{
    if (!rfg_token_allowed(st, text)) return false;
    rfg_advance(st, text);
    return true;
}

/* Feed a whole string in `chunk`-char tokens.  Returns false at the first chunk
 * the grammar refuses (so a test can assert exactly where a shape dies). */
static bool feed_chunks(rfg_state *st, const char *text, int chunk)
{
    size_t n = strlen(text);
    char buf[64];
    for (size_t i = 0; i < n; i += (size_t)chunk) {
        size_t len = n - i < (size_t)chunk ? n - i : (size_t)chunk;
        if (len >= sizeof(buf)) len = sizeof(buf) - 1;
        memcpy(buf, text + i, len);
        buf[len] = '\0';
        if (!feed_token(st, buf)) return false;
    }
    return true;
}

/* A state with no reasoning grace (a non-think model): constrained from token 0. */
static rfg_state *fresh(void)
{
    return rfg_state_new(RFG_FORMAT_JSON_OBJECT, false);
}

/* Accept a valid document at several token boundaries — the grammar has to hold
 * for ANY tokenization, and a real vocab emits multi-char pieces like `{"`. */
static void accepts(const char *json)
{
    for (int chunk = 1; chunk <= 5; chunk++) {
        rfg_state *st = fresh();
        ASSERT_NOT_NULL(st);
        ASSERT_TRUE(feed_chunks(st, json, chunk));
        ASSERT_TRUE(rfg_complete(st));
        rfg_state_free(st);
    }
}

/* Reject a malformed document: at SOME point the grammar must refuse, and it
 * must not report completion. */
static void rejects(const char *json)
{
    rfg_state *st = fresh();
    ASSERT_NOT_NULL(st);
    bool all_allowed = feed_chunks(st, json, 1);
    ASSERT_FALSE(all_allowed && rfg_complete(st));
    rfg_state_free(st);
}

/* ── acceptance ── */

static void test_accepts_flat_object(void)
{
    accepts("{\"a\":1}");
    accepts("{\"name\":\"pantheon\",\"ok\":true}");
}

static void test_accepts_empty_object(void)
{
    accepts("{}");
}

static void test_accepts_nested_and_arrays(void)
{
    accepts("{\"a\":{\"b\":[1,2,{\"c\":null}]},\"d\":[]}");
}

static void test_accepts_all_scalar_types(void)
{
    accepts("{\"s\":\"x\",\"i\":-12,\"f\":3.5e-2,\"t\":true,\"f2\":false,\"n\":null}");
}

static void test_accepts_whitespace(void)
{
    accepts("  {\n  \"a\" : 1 ,\n  \"b\" : [ 2 ]\n}");
}

static void test_accepts_string_escapes(void)
{
    accepts("{\"a\":\"line\\nbreak \\\"quoted\\\" \\u00e9\"}");
}

/* ── rejection: the shapes a model actually produces ── */

static void test_rejects_prose(void)
{
    rfg_state *st = fresh();
    /* The whole point: a model that would have answered in prose cannot. */
    ASSERT_FALSE(feed_token(st, "Sure"));
    ASSERT_FALSE(feed_token(st, "Here"));
    ASSERT_FALSE(rfg_complete(st));
    /* But the JSON opener is available, so the mask always has a survivor. */
    ASSERT_TRUE(feed_token(st, "{"));
    rfg_state_free(st);
}

static void test_rejects_markdown_fence(void)
{
    rfg_state *st = fresh();
    ASSERT_FALSE(feed_token(st, "```json\n{"));
    rfg_state_free(st);
}

static void test_rejects_top_level_array(void)
{
    /* json_object means an OBJECT — a bare array is not it. */
    rfg_state *st = fresh();
    ASSERT_FALSE(feed_token(st, "["));
    rfg_state_free(st);
}

static void test_rejects_trailing_comma(void)
{
    rejects("{\"a\":1,}");
    rejects("{\"a\":[1,]}");
}

static void test_rejects_unquoted_key(void)
{
    rejects("{a:1}");
}

static void test_rejects_missing_colon(void)
{
    rejects("{\"a\" 1}");
}

static void test_rejects_incomplete_number(void)
{
    rejects("{\"a\":1e}");
    rejects("{\"a\":-}");
    rejects("{\"a\":1.}");
}

static void test_rejects_bad_literal(void)
{
    rejects("{\"a\":tru}");
    rejects("{\"a\":TRUE}");
}

static void test_rejects_bad_escape(void)
{
    rejects("{\"a\":\"\\q\"}");
    rejects("{\"a\":\"\\u00zz\"}");
}

static void test_rejects_raw_control_char_in_string(void)
{
    rejects("{\"a\":\"x\ny\"}");
}

static void test_rejects_unbalanced_close(void)
{
    rejects("{\"a\":[1}");
    rejects("{\"a\":{]}");
}

/* ── completion + mask gating ── */

static void test_complete_stops_the_walk(void)
{
    rfg_state *st = fresh();
    ASSERT_TRUE(feed_chunks(st, "{\"a\":1}", 1));
    ASSERT_TRUE(rfg_complete(st));
    /* Once complete nothing further is a valid continuation — the serve stops
     * decoding here, and a stray token could only invalidate the content. */
    ASSERT_FALSE(rfg_token_allowed(st, "x"));
    /* …and the mask goes inactive, so EOS becomes sampleable again. */
    ASSERT_FALSE(rfg_mask_active(st));
    rfg_state_free(st);
}

static void test_mask_active_while_open(void)
{
    rfg_state *st = fresh();
    ASSERT_TRUE(rfg_mask_active(st));      /* before the object opens */
    ASSERT_TRUE(feed_token(st, "{"));
    ASSERT_TRUE(rfg_mask_active(st));      /* inside the object */
    ASSERT_TRUE(rfg_in_json(st));
    ASSERT_TRUE(feed_token(st, "}"));
    ASSERT_FALSE(rfg_mask_active(st));     /* done */
    ASSERT_FALSE(rfg_in_json(st));
    rfg_state_free(st);
}

/* ── the reasoning grace (think-style models) ── */

static void test_reasoning_grace_is_unconstrained(void)
{
    rfg_state *st = rfg_state_new(RFG_FORMAT_JSON_OBJECT, true);
    ASSERT_NOT_NULL(st);
    ASSERT_FALSE(rfg_mask_active(st));     /* the think block is free */
    ASSERT_FALSE(rfg_reasoning_ended(st));
    ASSERT_TRUE(feed_token(st, "The user wants "));
    ASSERT_TRUE(feed_token(st, "a JSON object. Let me think... "));
    ASSERT_FALSE(rfg_mask_active(st));

    /* "</think>" is self-detected in the raw stream — the serve emitter's think
     * flag is not reliable on every model, which is why this is here. */
    ASSERT_TRUE(feed_token(st, "</think>"));
    ASSERT_TRUE(rfg_reasoning_ended(st));
    ASSERT_TRUE(rfg_mask_active(st));

    /* From here the answer is constrained. */
    ASSERT_FALSE(feed_token(st, "Sure!"));
    ASSERT_TRUE(feed_chunks(st, "{\"a\":1}", 2));
    ASSERT_TRUE(rfg_complete(st));
    rfg_state_free(st);
}

static void test_reasoning_close_split_across_tokens(void)
{
    rfg_state *st = rfg_state_new(RFG_FORMAT_JSON_OBJECT, true);
    ASSERT_TRUE(feed_token(st, "thinking</th"));
    ASSERT_FALSE(rfg_reasoning_ended(st));
    ASSERT_TRUE(feed_token(st, "ink>"));
    ASSERT_TRUE(rfg_reasoning_ended(st));
    ASSERT_TRUE(rfg_mask_active(st));
    rfg_state_free(st);
}

static void test_explicit_reasoning_end(void)
{
    /* The reasoning-budget force-close path calls this directly. */
    rfg_state *st = rfg_state_new(RFG_FORMAT_JSON_OBJECT, true);
    ASSERT_FALSE(rfg_mask_active(st));
    rfg_reasoning_end(st);
    ASSERT_TRUE(rfg_reasoning_ended(st));
    ASSERT_TRUE(rfg_mask_active(st));
    ASSERT_FALSE(feed_token(st, "prose"));
    rfg_state_free(st);
}

static void test_text_format_builds_no_state(void)
{
    /* {"type":"text"} is the default and must cost nothing. */
    ASSERT_NULL(rfg_state_new(RFG_FORMAT_TEXT, false));
}

static void test_depth_cap_is_not_a_crash(void)
{
    rfg_state *st = fresh();
    ASSERT_TRUE(feed_token(st, "{"));
    /* Open key/value pairs until the cap refuses another container. */
    bool refused = false;
    for (int i = 0; i < RFG_MAX_DEPTH + 8; i++) {
        if (!feed_token(st, "\"a\":")) { refused = true; break; }
        if (!feed_token(st, "{"))      { refused = true; break; }
    }
    ASSERT_TRUE(refused);
    ASSERT_FALSE(rfg_complete(st));
    rfg_state_free(st);
}

static void test_null_state_is_safe(void)
{
    /* The serve calls these unconditionally when the feature is off. */
    ASSERT_TRUE(rfg_token_allowed(NULL, "x"));
    ASSERT_FALSE(rfg_complete(NULL));
    ASSERT_FALSE(rfg_mask_active(NULL));
    ASSERT_FALSE(rfg_in_json(NULL));
    rfg_advance(NULL, "x");
    rfg_reasoning_end(NULL);
    rfg_state_free(NULL);
}

int main(void)
{
    printf("\n=== response_format grammar (json_object constrained decoding) ===\n\n");
    RUN_TEST(accepts_flat_object);
    RUN_TEST(accepts_empty_object);
    RUN_TEST(accepts_nested_and_arrays);
    RUN_TEST(accepts_all_scalar_types);
    RUN_TEST(accepts_whitespace);
    RUN_TEST(accepts_string_escapes);
    RUN_TEST(rejects_prose);
    RUN_TEST(rejects_markdown_fence);
    RUN_TEST(rejects_top_level_array);
    RUN_TEST(rejects_trailing_comma);
    RUN_TEST(rejects_unquoted_key);
    RUN_TEST(rejects_missing_colon);
    RUN_TEST(rejects_incomplete_number);
    RUN_TEST(rejects_bad_literal);
    RUN_TEST(rejects_bad_escape);
    RUN_TEST(rejects_raw_control_char_in_string);
    RUN_TEST(rejects_unbalanced_close);
    RUN_TEST(complete_stops_the_walk);
    RUN_TEST(mask_active_while_open);
    RUN_TEST(reasoning_grace_is_unconstrained);
    RUN_TEST(reasoning_close_split_across_tokens);
    RUN_TEST(explicit_reasoning_end);
    RUN_TEST(text_format_builds_no_state);
    RUN_TEST(depth_cap_is_not_a_crash);
    RUN_TEST(null_state_is_safe);
    TEST_REPORT();
    return __ft_failed == 0 ? 0 : 1;
}
