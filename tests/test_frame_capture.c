/* SPDX-License-Identifier: AGPL-3.0-or-later */
/* Copyright (C) 2026 David (Corey) Herrera */
/**
 * @file test_frame_capture.c
 * @brief Tests for FRAME CAPTURE handling in the tool-call parser
 *        (src/serve/tool_calling.c).
 *
 *        When a turn exhausts its `reasoning_budget` the serve injects the
 *        "</think>" token itself; the grammar cannot tell that close from the
 *        model's own, so FORCED mode compels the tool-call opener while the model
 *        is still mid-thought.  Whatever it was still going to say — often a
 *        complete tool call — lands in the free-text parameter VALUE.
 *
 *        Every FX_* literal comes from tests/frame_capture_fixtures.h: synthetic
 *        content in the shapes observed in real serving.  FX_NATIVE_XML_CAPTURE
 *        is the control that rules the dialect explanation out: the captured
 *        text there is a NATIVE Qwen-XML call.
 *
 *        The suite asserts three behaviours:
 *          RECOVER — a value that IS a complete call becomes that call, with the
 *                    exact tool and arguments the model asked for.
 *          REJECT  — a value beginning with the reasoning close (and only that)
 *                    surfaces no call at all.
 *          PASS    — everything else, including the captured scratch-text values
 *                    from the same sessions, is untouched.
 *
 *        Pure C string parsing, no model/engine deps — fully local.
 *
 * @status None
 * @issues None
 * @todo None
 */

#include "lib/test.h"
#include "frame_capture_fixtures.h"
#include "../src/serve/tool_calling.h"
#include "../src/serve/json_reader.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

/* ============================================================
 * Tool schemas — a typical agent client's shapes
 * ============================================================ */

static const char *READ_TOOLS =
    "[{\"type\":\"function\",\"function\":{\"name\":\"read\",\"parameters\":"
    "{\"type\":\"object\",\"properties\":{\"path\":{\"type\":\"string\"}},"
    "\"required\":[\"path\"]}}}]";

static const char *GLOB_TOOLS =
    "[{\"type\":\"function\",\"function\":{\"name\":\"glob\",\"parameters\":"
    "{\"type\":\"object\",\"properties\":{\"pattern\":{\"type\":\"string\"}},"
    "\"required\":[\"pattern\"]}}}]";

/* grep + read: the FX_NATIVE_XML_CAPTURE shape is a `read` call captured inside a
 * `grep` frame, so both must be declared for that recovery to be legal. */
static const char *GREP_READ_TOOLS =
    "[{\"type\":\"function\",\"function\":{\"name\":\"grep\",\"parameters\":"
    "{\"type\":\"object\",\"properties\":{\"pattern\":{\"type\":\"string\"}},"
    "\"required\":[\"pattern\"]}}},"
    "{\"type\":\"function\",\"function\":{\"name\":\"read\",\"parameters\":"
    "{\"type\":\"object\",\"properties\":{\"path\":{\"type\":\"string\"}},"
    "\"required\":[\"path\"]}}}]";

/* read + write, so a two-parameter call with a big free-text `content` can be
 * exercised: that is the argument most likely to legitimately MENTION the
 * frame's vocabulary (an agent editing this very source file). */
static const char *WRITE_TOOLS =
    "[{\"type\":\"function\",\"function\":{\"name\":\"read\",\"parameters\":"
    "{\"type\":\"object\",\"properties\":{\"path\":{\"type\":\"string\"}},"
    "\"required\":[\"path\"]}}},"
    "{\"type\":\"function\",\"function\":{\"name\":\"write\",\"parameters\":"
    "{\"type\":\"object\",\"properties\":{\"path\":{\"type\":\"string\"},"
    "\"content\":{\"type\":\"string\"}},"
    "\"required\":[\"path\",\"content\"]}}}]";

/* ============================================================
 * Helpers
 * ============================================================ */

/* Rebuild the raw model emission that produces an argument: the compelled
 * wrapped XML frame with `value` sitting in the parameter's VALUE slot.  The
 * fixture is the post-dup_trim value, so re-wrapping it reproduces the bytes
 * the parser sees. */
static char *xml_frame(const char *tool, const char *key, const char *value)
{
    size_t n = strlen(tool) + strlen(key) + strlen(value) + 128;
    char *buf = malloc(n);
    if (!buf) return NULL;
    snprintf(buf, n,
             "<tool_call>\n<function=%s>\n<parameter=%s>\n%s\n</parameter>\n"
             "</function>\n</tool_call>",
             tool, key, value);
    return buf;
}

/* Parse one reconstructed frame under the QWEN_XML dialect. */
static int parse_frame(const char *tools_json, const char *tool, const char *key,
                       const char *value, tc_tool_call **calls, int *n)
{
    jr_value *tools = jr_parse(tools_json);
    char *text = xml_frame(tool, key, value);
    int rc = tc_parse_tool_calls(text, tools, TC_DIALECT_QWEN_XML, calls, n);
    free(text);
    jr_free(tools);
    return rc;
}

/* ============================================================
 * RECOVER — captured calls
 * ============================================================ */

/* THE CONTROL.  A native Qwen-XML `read` call captured inside a `grep` frame.
 * Recovering this needs no theory about dialects at all — the captured call is
 * in the grammar's own dialect. */
static void test_native_xml_capture_recovered(void) {
    tc_tool_call *calls = NULL; int n = -1;
    ASSERT_INT_EQ(parse_frame(GREP_READ_TOOLS, "grep", "pattern",
                              FX_NATIVE_XML_CAPTURE, &calls, &n), 0);
    ASSERT_INT_EQ(n, 1);
    ASSERT_NOT_NULL(calls);
    ASSERT_STR_EQ(calls[0].name, "read");
    ASSERT_STR_EQ(calls[0].arguments, "{\"path\":\"lumen/serve.log\"}");
    ASSERT_TRUE(strncmp(calls[0].id, "call_", 5) == 0);
    tc_tool_calls_free(calls, n);
}

static void test_hermes_leadin_capture_recovered(void) {
    tc_tool_call *calls = NULL; int n = -1;
    ASSERT_INT_EQ(parse_frame(READ_TOOLS, "read", "path", FX_HERMES_LEADIN_CAPTURE, &calls, &n), 0);
    ASSERT_INT_EQ(n, 1);
    ASSERT_NOT_NULL(calls);
    ASSERT_STR_EQ(calls[0].name, "read");
    ASSERT_STR_EQ(calls[0].arguments, "{\"path\":\"lumen/Makefile\"}");
    ASSERT_TRUE(strncmp(calls[0].id, "call_", 5) == 0);
    tc_tool_calls_free(calls, n);
}

static void test_overclosed_capture_recovered(void) {
    tc_tool_call *calls = NULL; int n = -1;
    ASSERT_INT_EQ(parse_frame(READ_TOOLS, "read", "path", FX_OVERCLOSED_CAPTURE, &calls, &n), 0);
    ASSERT_INT_EQ(n, 1);
    ASSERT_STR_EQ(calls[0].name, "read");
    ASSERT_STR_EQ(calls[0].arguments,
                  "{\"path\":\"lumen/include/lumen/settings_cache.h\"}");
    tc_tool_calls_free(calls, n);
}

static void test_absolute_path_capture_recovered(void) {
    tc_tool_call *calls = NULL; int n = -1;
    ASSERT_INT_EQ(parse_frame(READ_TOOLS, "read", "path", FX_ABSOLUTE_PATH_CAPTURE, &calls, &n), 0);
    ASSERT_INT_EQ(n, 1);
    ASSERT_STR_EQ(calls[0].name, "read");
    ASSERT_STR_EQ(calls[0].arguments,
                  "{\"path\":\"/home/user/src/lumen/include/settings_cache.h\"}");
    tc_tool_calls_free(calls, n);
}

/* The recovered call must carry no trace of the outer frame. */
static void test_recovered_arguments_drop_the_outer_frame(void) {
    tc_tool_call *calls = NULL; int n = -1;
    ASSERT_INT_EQ(parse_frame(READ_TOOLS, "read", "path", FX_HERMES_LEADIN_CAPTURE, &calls, &n), 0);
    ASSERT_INT_EQ(n, 1);
    ASSERT_NULL(strstr(calls[0].arguments, "<tool_call>"));
    ASSERT_NULL(strstr(calls[0].arguments, "</think>"));
    ASSERT_NULL(strstr(calls[0].arguments, "call_function"));
    tc_tool_calls_free(calls, n);
}

/* A capture that names a DIFFERENT tool than the armed frame routes to the tool
 * the model actually asked for — that is the whole point. */
static void test_capture_across_tools_routes_to_the_named_tool(void) {
    const char *v =
        "<tool_call>call_function{\"func\":\"write\",\"args\":"
        "{\"path\":\"/tmp/a.txt\",\"content\":\"hi\"}}";
    tc_tool_call *calls = NULL; int n = -1;
    ASSERT_INT_EQ(parse_frame(WRITE_TOOLS, "read", "path", v, &calls, &n), 0);
    ASSERT_INT_EQ(n, 1);
    ASSERT_STR_EQ(calls[0].name, "write");
    ASSERT_STR_EQ(calls[0].arguments,
                  "{\"path\":\"/tmp/a.txt\",\"content\":\"hi\"}");
    tc_tool_calls_free(calls, n);
}

/* ============================================================
 * REJECT — a leading reasoning close, and nothing else
 * ============================================================ */

/* The value is the force-closed model's own "</think>" plus an unfinished
 * thought: unmistakably a captured frame, with no call in it.  No tool call is
 * surfaced rather than dispatching the fragment as a path. */
static void test_leading_reasoning_close_rejected(void) {
    const char *v = "</think>\n\nI should look at the Makefile next.";
    tc_tool_call *calls = NULL; int n = -1;
    ASSERT_INT_EQ(parse_frame(READ_TOOLS, "read", "path", v, &calls, &n), 0);
    ASSERT_INT_EQ(n, 0);
    tc_tool_calls_free(calls, n);
}

/* An inner call naming a tool the request never declared is NOT recovered — the
 * tool_name_known gate holds and no invented call reaches the client. */
static void test_undeclared_inner_tool_never_recovered(void) {
    const char *v =
        "<tool_call>call_function{\"func\":\"rm_rf\",\"args\":{\"path\":\"/\"}}";
    tc_tool_call *calls = NULL; int n = -1;
    ASSERT_INT_EQ(parse_frame(READ_TOOLS, "read", "path", v, &calls, &n), 0);
    /* Not recovered, and NOT rejected either: "<tool_call>" is not a rejecting
     * marker (that collision with real `write`/`edit` content is untested), so
     * the outer call passes through as the harmless read it already was. */
    ASSERT_INT_EQ(n, 1);
    ASSERT_STR_EQ(calls[0].name, "read");
    ASSERT_NULL(strstr(calls[0].arguments, "rm_rf\":"));
    tc_tool_calls_free(calls, n);
}

/* ============================================================
 * PASS — captured scratch text, and legitimate content
 * ============================================================ */

/* A bash pipeline captured into a `read` path.  The model
 * wanted `bash`, but the value carries no frame marker and no complete call, so
 * there is nothing to recover — the call passes through exactly as before.
 * Recovering it would mean GUESSING the tool from shell metacharacters, which
 * misfires on legitimate `bash`/`grep` arguments. */
static void test_shell_in_read_path_untouched(void) {
    tc_tool_call *calls = NULL; int n = -1;
    ASSERT_INT_EQ(parse_frame(READ_TOOLS, "read", "path", FX_SHELL_IN_READ_PATH, &calls, &n), 0);
    ASSERT_INT_EQ(n, 1);
    ASSERT_STR_EQ(calls[0].name, "read");
    ASSERT_NOT_NULL(strstr(calls[0].arguments, "settings_cache.c.md 2>/dev/null"));
    tc_tool_calls_free(calls, n);
}

/* Shell pipeline + dangling fence in a `glob` pattern. */
static void test_shell_in_glob_pattern_untouched(void) {
    tc_tool_call *calls = NULL; int n = -1;
    ASSERT_INT_EQ(parse_frame(GLOB_TOOLS, "glob", "pattern", FX_SHELL_IN_GLOB, &calls, &n), 0);
    ASSERT_INT_EQ(n, 1);
    ASSERT_STR_EQ(calls[0].name, "glob");
    ASSERT_NOT_NULL(strstr(calls[0].arguments, "lume-4471093821"));
    tc_tool_calls_free(calls, n);
}

/* Over 5,000 chars of analysis, C source with braces, and a
 * truncated SEARCH/REPLACE block.  The braces must NOT be mistaken for a JSON
 * call — exactly one `glob` call comes out, unchanged. */
static void test_analysis_dump_untouched(void) {
    tc_tool_call *calls = NULL; int n = -1;
    ASSERT_TRUE(strlen(FX_ANALYSIS_DUMP) > 5000);
    ASSERT_INT_EQ(parse_frame(GLOB_TOOLS, "glob", "pattern", FX_ANALYSIS_DUMP, &calls, &n), 0);
    ASSERT_INT_EQ(n, 1);
    ASSERT_STR_EQ(calls[0].name, "glob");
    ASSERT_NOT_NULL(strstr(calls[0].arguments, "<<<<<<< SEARCH"));
    tc_tool_calls_free(calls, n);
}

/* THE common-case guard: an agent writing a file whose CONTENT mentions the
 * frame's vocabulary (this repo's own sources do exactly that).  No complete
 * call runs to the end, so the write is dispatched intact. */
static void test_write_content_mentioning_the_frame_is_kept(void) {
    const char *v =
        "/* the parser watches these */\n"
        "static const char TC_OPEN[] = \"<tool_call>\";\n"
        "static const char FN_OPEN[] = \"<function=\";\n"
        "static const char PARAM_OPEN[] = \"<parameter=\";\n";
    tc_tool_call *calls = NULL; int n = -1;
    ASSERT_INT_EQ(parse_frame(WRITE_TOOLS, "write", "content", v, &calls, &n), 0);
    ASSERT_INT_EQ(n, 1);
    ASSERT_STR_EQ(calls[0].name, "write");
    ASSERT_NOT_NULL(strstr(calls[0].arguments, "TC_OPEN"));
    ASSERT_NOT_NULL(strstr(calls[0].arguments, "PARAM_OPEN"));
    tc_tool_calls_free(calls, n);
}

/* Same guard at the START of the content: a file whose first bytes are the frame
 * vocabulary is still written.  Only a leading "</think>" rejects. */
static void test_write_content_starting_with_frame_literal_is_kept(void) {
    const char *v = "<tool_call> is the wrapper opener this parser watches for.";
    tc_tool_call *calls = NULL; int n = -1;
    ASSERT_INT_EQ(parse_frame(WRITE_TOOLS, "write", "content", v, &calls, &n), 0);
    ASSERT_INT_EQ(n, 1);
    ASSERT_STR_EQ(calls[0].name, "write");
    ASSERT_NOT_NULL(strstr(calls[0].arguments, "wrapper opener"));
    tc_tool_calls_free(calls, n);
}

/* The "runs to the end" gate: a value that CONTAINS a complete call but keeps
 * going afterwards is documentation, not a capture. */
static void test_value_containing_a_call_as_content_is_kept(void) {
    const char *v =
        "for example <tool_call>call_function{\"func\":\"read\",\"args\":"
        "{\"path\":\"/x\"}} is what that shape looks like in practice";
    tc_tool_call *calls = NULL; int n = -1;
    ASSERT_INT_EQ(parse_frame(WRITE_TOOLS, "write", "content", v, &calls, &n), 0);
    ASSERT_INT_EQ(n, 1);
    ASSERT_STR_EQ(calls[0].name, "write");
    ASSERT_NOT_NULL(strstr(calls[0].arguments, "in practice"));
    tc_tool_calls_free(calls, n);
}

/* A plain JSON object in a value, with NO frame marker, is never inspected —
 * a `write` payload that happens to be JSON stays a write. */
static void test_plain_json_value_without_marker_is_kept(void) {
    const char *v = "{\"func\": \"read\", \"args\": {\"path\": \"/etc/hostname\"}}";
    tc_tool_call *calls = NULL; int n = -1;
    ASSERT_INT_EQ(parse_frame(WRITE_TOOLS, "write", "content", v, &calls, &n), 0);
    ASSERT_INT_EQ(n, 1);
    ASSERT_STR_EQ(calls[0].name, "write");
    ASSERT_NOT_NULL(strstr(calls[0].arguments, "etc/hostname"));
    tc_tool_calls_free(calls, n);
}

/* Ordinary turns must still parse: an ordinary call is completely unaffected
 * by the capture path. */
static void test_ordinary_call_unaffected(void) {
    tc_tool_call *calls = NULL; int n = -1;
    ASSERT_INT_EQ(parse_frame(READ_TOOLS, "read", "path",
                              "lumen/src/settings/settings_cache.c",
                              &calls, &n), 0);
    ASSERT_INT_EQ(n, 1);
    ASSERT_STR_EQ(calls[0].name, "read");
    ASSERT_STR_EQ(calls[0].arguments,
                  "{\"path\":\"lumen/src/settings/settings_cache.c\"}");
    tc_tool_calls_free(calls, n);
}

/* ============================================================
 * Main
 * ============================================================ */

int main(void) {
    printf("\n=== Forced-close FRAME CAPTURE ===\n\n");

    RUN_TEST(native_xml_capture_recovered);
    RUN_TEST(hermes_leadin_capture_recovered);
    RUN_TEST(overclosed_capture_recovered);
    RUN_TEST(absolute_path_capture_recovered);
    RUN_TEST(recovered_arguments_drop_the_outer_frame);
    RUN_TEST(capture_across_tools_routes_to_the_named_tool);

    RUN_TEST(leading_reasoning_close_rejected);
    RUN_TEST(undeclared_inner_tool_never_recovered);

    RUN_TEST(shell_in_read_path_untouched);
    RUN_TEST(shell_in_glob_pattern_untouched);
    RUN_TEST(analysis_dump_untouched);
    RUN_TEST(write_content_mentioning_the_frame_is_kept);
    RUN_TEST(write_content_starting_with_frame_literal_is_kept);
    RUN_TEST(value_containing_a_call_as_content_is_kept);
    RUN_TEST(plain_json_value_without_marker_is_kept);
    RUN_TEST(ordinary_call_unaffected);

    TEST_REPORT();
    return __ft_failed > 0 ? 1 : 0;
}
