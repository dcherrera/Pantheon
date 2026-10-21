/* SPDX-License-Identifier: AGPL-3.0-or-later */
/* Copyright (C) 2026 David (Corey) Herrera */
/*
 * Unit tests for the STRUCTURALLY-WRONG tool-call shapes and for the truncated-
 * JSON repair (src/serve/tool_calling.c).
 *
 * These are the two serve-side items from the audit of Qwen's own agent
 * harnesses:
 *
 *   #4  detect output that is structurally wrong rather than merely unparseable
 *       — a JSON body closed with this dialect's own "</parameter></function>"
 *       tail (qwen-code geminiChat.ts:1358-1377), a "</think>" leaking into the
 *       tool-call region (converter.ts:1664), and a call with no name / no such
 *       name (converter.ts:1699-1711) — and report it, never execute it.
 *
 *   #11 close a truncated argument object instead of losing the call
 *       (streamingToolCallParser.ts:294-323, :409-439).
 *
 * Each detector is paired here with the NEIGHBOUR it must not be confused with,
 * because every one of these tests is really a test that healthy traffic is
 * untouched. Pure C string parsing, no model/engine deps — fully local.
 */

#include "lib/test.h"
#include "../src/serve/tool_calling.h"
#include "../src/serve/json_reader.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

/* `read` takes a string `path`; `write` takes a string `path` and an OBJECT
 * `patch`, so a legitimate call can end in "}\n</parameter>\n</function>" — the
 * exact bytes the leaked-tail detector keys on. That collision is the point. */
static const char *DEFECT_TOOLS =
    "[{\"type\":\"function\",\"function\":{\"name\":\"read\","
    "\"parameters\":{\"type\":\"object\","
    "\"properties\":{\"path\":{\"type\":\"string\"}},"
    "\"required\":[\"path\"]}}},"
    "{\"type\":\"function\",\"function\":{\"name\":\"write\","
    "\"parameters\":{\"type\":\"object\","
    "\"properties\":{\"path\":{\"type\":\"string\"},"
    "\"patch\":{\"type\":\"object\"}},"
    "\"required\":[\"path\"]}}}]";

/* ============================================================
 * #4a  LEAKED DIALECT TAIL — the model mixed two frames
 * ============================================================ */

/* The defect itself: JSON arguments where the <parameter=KEY> blocks belong,
 * finished off with the XML frame's own closing tail. Parsed as-is this yields
 * a `read {}` — a named call carrying nothing, indistinguishable from a call
 * with a deliberately empty argument bag, and unusable either way. */
static void test_leaked_tail_dropped_and_named(void) {
    jr_value *tools = jr_parse(DEFECT_TOOLS);
    const char *text =
        "<tool_call>\n<function=read>\n"
        "{\"path\": \"/etc/hostname\"}\n"
        "</parameter>\n</function>\n</tool_call>";

    tc_tool_call *calls = NULL; int n = -1;
    tc_parse_error err = {0};
    ASSERT_INT_EQ(tc_parse_tool_calls_diagnosed(text, tools, TC_DIALECT_QWEN_XML,
                                                &calls, &n, &err, NULL), 0);
    ASSERT_INT_EQ(n, 0);
    ASSERT_INT_EQ(err.kind, TC_PARSE_LEAKED_TOOL_TAGS);
    ASSERT_STR_EQ(err.function_name, "read");
    ASSERT_STR_EQ(tc_parse_failure_code(err.kind), "leaked_tool_tags");

    tc_parse_error_free(&err);
    tc_tool_calls_free(calls, n);
    jr_free(tools);
}

/* NEIGHBOUR 1 — the same closing bytes, but from a HEALTHY call. `patch` is a
 * schema-typed object, so its value legitimately ends in '}' and the call
 * legitimately ends "}\n</parameter>\n</function>". qwen-code's regex alone
 * would reject this; what saves it is that a <parameter= opened the close. */
static void test_leaked_tail_json_valued_parameter_survives(void) {
    jr_value *tools = jr_parse(DEFECT_TOOLS);
    const char *text =
        "<tool_call>\n<function=write>\n"
        "<parameter=patch>\n{\"op\": \"set\"}\n</parameter>\n"
        "</function>\n</tool_call>";

    tc_tool_call *calls = NULL; int n = -1;
    tc_parse_error err = {0};
    ASSERT_INT_EQ(tc_parse_tool_calls_diagnosed(text, tools, TC_DIALECT_QWEN_XML,
                                                &calls, &n, &err, NULL), 0);
    ASSERT_INT_EQ(n, 1);
    ASSERT_STR_EQ(calls[0].name, "write");
    ASSERT_STR_EQ(calls[0].arguments, "{\"patch\":{\"op\": \"set\"}}");
    ASSERT_INT_EQ(err.kind, TC_PARSE_OK);

    tc_parse_error_free(&err);
    tc_tool_calls_free(calls, n);
    jr_free(tools);
}

/* NEIGHBOUR 2 — STRING AWARENESS, the property qwen-code's hasLeakedToolCallTags
 * exists to provide. Here the ONLY "} </parameter> </function>" run in the block
 * sits inside a quoted JSON value, so it is content the model chose to write,
 * not a frame it leaked. A scanner that did not track string state would find
 * the pattern, drop the call and report a defect that is not there. */
static void test_leaked_tail_inside_a_string_is_not_a_leak(void) {
    jr_value *tools = jr_parse(DEFECT_TOOLS);
    const char *text =
        "<function=read>\n"
        "{\"note\": \"see } </parameter> </function>\"}\n";

    tc_tool_call *calls = NULL; int n = -1;
    tc_parse_error err = {0};
    ASSERT_INT_EQ(tc_parse_tool_calls_diagnosed(text, tools, TC_DIALECT_QWEN_XML,
                                                &calls, &n, &err, NULL), 0);
    ASSERT_INT_EQ(n, 1);
    ASSERT_STR_EQ(calls[0].name, "read");
    ASSERT_INT_EQ(err.kind, TC_PARSE_OK);

    tc_parse_error_free(&err);
    tc_tool_calls_free(calls, n);
    jr_free(tools);
}

/* NEIGHBOUR 3 — no "</function>" means no dialect tail, so there are not two
 * frames to have mixed. The block is handled exactly as it was before. */
static void test_leaked_tail_needs_the_function_close(void) {
    jr_value *tools = jr_parse(DEFECT_TOOLS);
    const char *text = "<function=read>\n{\"path\": \"/etc/hostname\"}\n</parameter>\n";

    tc_tool_call *calls = NULL; int n = -1;
    tc_parse_error err = {0};
    ASSERT_INT_EQ(tc_parse_tool_calls_diagnosed(text, tools, TC_DIALECT_QWEN_XML,
                                                &calls, &n, &err, NULL), 0);
    ASSERT_INT_EQ(n, 1);
    ASSERT_INT_EQ(err.kind, TC_PARSE_OK);

    tc_parse_error_free(&err);
    tc_tool_calls_free(calls, n);
    jr_free(tools);
}

/* ============================================================
 * #4b  REASONING LEAK — "</think>" inside the tool-call region
 * ============================================================ */

/* The value the model wrote into the parameter is the thought it had not
 * finished, opened by the reasoning close it was about to write. The call was
 * already being rejected; the change is that the rejection now has a name, so
 * the turn no longer reads as "the model chose not to act". */
static void test_reasoning_leak_named(void) {
    jr_value *tools = jr_parse(DEFECT_TOOLS);
    const char *text =
        "<function=read>\n<parameter=path>\n"
        "</think>\n\nI should look at the log before deciding.\n"
        "</parameter>\n</function>";

    tc_tool_call *calls = NULL; int n = -1;
    tc_parse_error err = {0};
    ASSERT_INT_EQ(tc_parse_tool_calls_diagnosed(text, tools, TC_DIALECT_QWEN_XML,
                                                &calls, &n, &err, NULL), 0);
    ASSERT_INT_EQ(n, 0);
    ASSERT_INT_EQ(err.kind, TC_PARSE_REASONING_LEAK);
    ASSERT_STR_EQ(err.function_name, "read");
    ASSERT_STR_EQ(tc_parse_failure_code(err.kind), "reasoning_leak");

    tc_parse_error_free(&err);
    tc_tool_calls_free(calls, n);
    jr_free(tools);
}

/* NEIGHBOUR — the same leading "</think>", but the value goes on to carry a
 * COMPLETE call. That call is recovered and returned, so nothing is absent and
 * nothing is reported. An error here would be a warning riding a call that was
 * emitted, which this channel deliberately never carries. */
static void test_reasoning_leak_silent_when_call_recovered(void) {
    jr_value *tools = jr_parse(DEFECT_TOOLS);
    const char *text =
        "<function=write>\n<parameter=patch>\n"
        "</think>\n\n<function=read>\n<parameter=path>\n/etc/hostname\n"
        "</parameter>\n</function>";

    tc_tool_call *calls = NULL; int n = -1;
    tc_parse_error err = {0};
    ASSERT_INT_EQ(tc_parse_tool_calls_diagnosed(text, tools, TC_DIALECT_QWEN_XML,
                                                &calls, &n, &err, NULL), 0);
    ASSERT_INT_EQ(n, 1);
    ASSERT_STR_EQ(calls[0].name, "read");
    ASSERT_STR_EQ(calls[0].arguments, "{\"path\":\"/etc/hostname\"}");
    ASSERT_INT_EQ(err.kind, TC_PARSE_OK);

    tc_parse_error_free(&err);
    tc_tool_calls_free(calls, n);
    jr_free(tools);
}

/* ============================================================
 * #4c  MALFORMED NAME — no name, or no such tool
 * ============================================================ */

static void test_call_name_missing_named(void) {
    jr_value *tools = jr_parse(DEFECT_TOOLS);
    const char *text = "<tool_call>\n<function=>\n</function>\n</tool_call>";

    tc_tool_call *calls = NULL; int n = -1;
    tc_parse_error err = {0};
    ASSERT_INT_EQ(tc_parse_tool_calls_diagnosed(text, tools, TC_DIALECT_QWEN_XML,
                                                &calls, &n, &err, NULL), 0);
    ASSERT_INT_EQ(n, 0);
    ASSERT_INT_EQ(err.kind, TC_PARSE_CALL_NAME_MISSING);
    /* No name existed, so none is claimed — the field is absent on the wire. */
    ASSERT_NULL(err.function_name);
    ASSERT_STR_EQ(tc_parse_failure_code(err.kind), "call_name_missing");

    tc_parse_error_free(&err);
    tc_tool_calls_free(calls, n);
    jr_free(tools);
}

static void test_call_name_unknown_named(void) {
    jr_value *tools = jr_parse(DEFECT_TOOLS);
    const char *text =
        "<function=serach_files>\n<parameter=path>\n/etc\n</parameter>\n</function>";

    tc_tool_call *calls = NULL; int n = -1;
    tc_parse_error err = {0};
    ASSERT_INT_EQ(tc_parse_tool_calls_diagnosed(text, tools, TC_DIALECT_QWEN_XML,
                                                &calls, &n, &err, NULL), 0);
    ASSERT_INT_EQ(n, 0);
    ASSERT_INT_EQ(err.kind, TC_PARSE_CALL_NAME_UNKNOWN);
    ASSERT_STR_EQ(err.function_name, "serach_files");
    ASSERT_STR_EQ(tc_parse_failure_code(err.kind), "call_name_unknown");

    tc_parse_error_free(&err);
    tc_tool_calls_free(calls, n);
    jr_free(tools);
}

/* A good call standing beside a bad name: the good one still comes through, and
 * the bad one is still reported. One mis-emission must not cost the whole turn
 * — the same rule the abandoned-call detector follows. */
static void test_bad_name_does_not_cost_the_good_call(void) {
    jr_value *tools = jr_parse(DEFECT_TOOLS);
    const char *text =
        "<function=read>\n<parameter=path>\n/etc/hostname\n</parameter>\n</function>\n"
        "<function=serach_files>\n<parameter=path>\n/etc\n</parameter>\n</function>";

    tc_tool_call *calls = NULL; int n = -1;
    tc_parse_error err = {0};
    ASSERT_INT_EQ(tc_parse_tool_calls_diagnosed(text, tools, TC_DIALECT_QWEN_XML,
                                                &calls, &n, &err, NULL), 0);
    ASSERT_INT_EQ(n, 1);
    ASSERT_STR_EQ(calls[0].name, "read");
    ASSERT_INT_EQ(err.kind, TC_PARSE_CALL_NAME_UNKNOWN);

    tc_parse_error_free(&err);
    tc_tool_calls_free(calls, n);
    jr_free(tools);
}

/* NO schema to gate against is permissive, exactly as tool_name_known already
 * promises: with no `tools` array nothing can be "unknown", so nothing is
 * reported and the call comes through. */
static void test_no_schema_reports_nothing(void) {
    const char *text =
        "<function=anything_at_all>\n<parameter=path>\n/etc\n</parameter>\n</function>";

    tc_tool_call *calls = NULL; int n = -1;
    tc_parse_error err = {0};
    ASSERT_INT_EQ(tc_parse_tool_calls_diagnosed(text, NULL, TC_DIALECT_QWEN_XML,
                                                &calls, &n, &err, NULL), 0);
    ASSERT_INT_EQ(n, 1);
    ASSERT_INT_EQ(err.kind, TC_PARSE_OK);

    tc_parse_error_free(&err);
    tc_tool_calls_free(calls, n);
}

/* ============================================================
 * #11  TRUNCATED-JSON REPAIR (ORNITH_JSON dialect)
 * ============================================================ */

/* The plainest case: the arguments object closed, the outer object did not. One
 * '}' is owed and there is exactly one way to write it. */
static void test_repair_closes_outer_object(void) {
    jr_value *tools = jr_parse(DEFECT_TOOLS);
    const char *text =
        "```json\n{\"function\": \"read\", \"arguments\": {\"path\": \"/etc/hostname\"}";

    tc_tool_call *calls = NULL; int n = -1;
    tc_parse_repair rep = {0};
    ASSERT_INT_EQ(tc_parse_tool_calls_diagnosed(text, tools, TC_DIALECT_ORNITH_JSON,
                                                &calls, &n, NULL, &rep), 0);
    ASSERT_INT_EQ(n, 1);
    ASSERT_STR_EQ(calls[0].name, "read");
    ASSERT_STR_EQ(calls[0].arguments, "{\"path\":\"/etc/hostname\"}");
    ASSERT_INT_EQ(rep.count, 1);
    ASSERT_STR_EQ(rep.added, "}");

    tc_parse_repair_free(&rep);
    tc_tool_calls_free(calls, n);
    jr_free(tools);
}

/* The cut landed inside a string value. qwen-code's own first repair is
 * JSON.parse(buffer + '"'); this is that, plus the closers the string was
 * nested in. The value keeps only what the model got out — which is why the
 * '"' is visible in `added`, so the log says so. */
static void test_repair_closes_open_string(void) {
    jr_value *tools = jr_parse(DEFECT_TOOLS);
    const char *text =
        "{\"function\": \"read\", \"arguments\": {\"path\": \"/etc/host";

    tc_tool_call *calls = NULL; int n = -1;
    tc_parse_repair rep = {0};
    ASSERT_INT_EQ(tc_parse_tool_calls_diagnosed(text, tools, TC_DIALECT_ORNITH_JSON,
                                                &calls, &n, NULL, &rep), 0);
    ASSERT_INT_EQ(n, 1);
    ASSERT_STR_EQ(calls[0].arguments, "{\"path\":\"/etc/host\"}");
    ASSERT_INT_EQ(rep.count, 1);
    ASSERT_STR_EQ(rep.added, "\"}}");

    tc_parse_repair_free(&rep);
    tc_tool_calls_free(calls, n);
    jr_free(tools);
}

/* The cut landed just AFTER a complete string value, so only the two braces are
 * owed and no value is at risk at all — the cleanest repair there is. It is here
 * because it is also the shape that catches a walk which forgets that a string's
 * CLOSING quote is a significant byte: miss that and the repair slices the value
 * back off before closing. */
static void test_repair_closes_after_a_complete_value(void) {
    jr_value *tools = jr_parse(DEFECT_TOOLS);
    const char *text = "{\"function\": \"read\", \"arguments\": {\"path\": \"/etc/hostname\"";

    tc_tool_call *calls = NULL; int n = -1;
    tc_parse_repair rep = {0};
    ASSERT_INT_EQ(tc_parse_tool_calls_diagnosed(text, tools, TC_DIALECT_ORNITH_JSON,
                                                &calls, &n, NULL, &rep), 0);
    ASSERT_INT_EQ(n, 1);
    ASSERT_STR_EQ(calls[0].arguments, "{\"path\":\"/etc/hostname\"}");
    ASSERT_INT_EQ(rep.count, 1);
    ASSERT_STR_EQ(rep.added, "}}");   /* no '"' — nothing was closed early */

    tc_parse_repair_free(&rep);
    tc_tool_calls_free(calls, n);
    jr_free(tools);
}

/* A closing markdown fence with no closing brace: the fence is frame, not JSON,
 * and parse_ornith_json already steps over the OPENING one. */
static void test_repair_ignores_trailing_fence(void) {
    jr_value *tools = jr_parse(DEFECT_TOOLS);
    const char *text =
        "```json\n{\"function\": \"read\", \"arguments\": {\"path\": \"/x\"}\n```";

    tc_tool_call *calls = NULL; int n = -1;
    tc_parse_repair rep = {0};
    ASSERT_INT_EQ(tc_parse_tool_calls_diagnosed(text, tools, TC_DIALECT_ORNITH_JSON,
                                                &calls, &n, NULL, &rep), 0);
    ASSERT_INT_EQ(n, 1);
    ASSERT_STR_EQ(calls[0].arguments, "{\"path\":\"/x\"}");
    ASSERT_INT_EQ(rep.count, 1);

    tc_parse_repair_free(&rep);
    tc_tool_calls_free(calls, n);
    jr_free(tools);
}

/* ── The declines. Each one is a place where closing would mean inventing. ── */

/* After a ':' the next thing owed is a VALUE. Repairing here would put a made-up
 * one under a real tool's name. */
static void test_repair_declines_dangling_colon(void) {
    jr_value *tools = jr_parse(DEFECT_TOOLS);
    const char *text = "```json\n{\"function\":\"read\",\"arguments\":{\"path\":";

    tc_tool_call *calls = NULL; int n = -1;
    tc_parse_repair rep = {0};
    ASSERT_INT_EQ(tc_parse_tool_calls_diagnosed(text, tools, TC_DIALECT_ORNITH_JSON,
                                                &calls, &n, NULL, &rep), 0);
    ASSERT_INT_EQ(n, 0);
    ASSERT_INT_EQ(rep.count, 0);

    tc_parse_repair_free(&rep);
    tc_tool_calls_free(calls, n);
    jr_free(tools);
}

/* One character later: the string is open but empty. Closing it produces
 * {"path":""} — the empty-required-argument defect this parser already drops on
 * the XML side, and it must not be manufactured here. */
static void test_repair_declines_empty_open_string(void) {
    jr_value *tools = jr_parse(DEFECT_TOOLS);
    const char *text = "{\"function\":\"read\",\"arguments\":{\"path\":\"";

    tc_tool_call *calls = NULL; int n = -1;
    tc_parse_repair rep = {0};
    ASSERT_INT_EQ(tc_parse_tool_calls_diagnosed(text, tools, TC_DIALECT_ORNITH_JSON,
                                                &calls, &n, NULL, &rep), 0);
    ASSERT_INT_EQ(n, 0);
    ASSERT_INT_EQ(rep.count, 0);

    tc_parse_repair_free(&rep);
    tc_tool_calls_free(calls, n);
    jr_free(tools);
}

/* An argument bag that was opened and never filled would close to `{}` — a
 * named call carrying nothing, which is the exact dead end the whole diagnostic
 * layer exists to stop. Declined. */
static void test_repair_declines_empty_argument_bag(void) {
    jr_value *tools = jr_parse(DEFECT_TOOLS);
    const char *text = "{\"function\":\"read\",\"arguments\":{";

    tc_tool_call *calls = NULL; int n = -1;
    tc_parse_repair rep = {0};
    ASSERT_INT_EQ(tc_parse_tool_calls_diagnosed(text, tools, TC_DIALECT_ORNITH_JSON,
                                                &calls, &n, NULL, &rep), 0);
    ASSERT_INT_EQ(n, 0);
    ASSERT_INT_EQ(rep.count, 0);

    tc_parse_repair_free(&rep);
    tc_tool_calls_free(calls, n);
    jr_free(tools);
}

/* After a ',' a KEY is owed. Fabricating one is out of the question. */
static void test_repair_declines_trailing_comma(void) {
    jr_value *tools = jr_parse(DEFECT_TOOLS);
    const char *text = "{\"function\":\"read\",\"arguments\":{\"path\":\"/x\",";

    tc_tool_call *calls = NULL; int n = -1;
    tc_parse_repair rep = {0};
    ASSERT_INT_EQ(tc_parse_tool_calls_diagnosed(text, tools, TC_DIALECT_ORNITH_JSON,
                                                &calls, &n, NULL, &rep), 0);
    ASSERT_INT_EQ(n, 0);
    ASSERT_INT_EQ(rep.count, 0);

    tc_parse_repair_free(&rep);
    tc_tool_calls_free(calls, n);
    jr_free(tools);
}

/* The cut landed on a backslash: whatever it was going to escape is unknown, so
 * appending a quote would change what the escape means. */
static void test_repair_declines_mid_escape(void) {
    jr_value *tools = jr_parse(DEFECT_TOOLS);
    const char *text = "{\"function\":\"read\",\"arguments\":{\"path\":\"/tmp/a\\";

    tc_tool_call *calls = NULL; int n = -1;
    tc_parse_repair rep = {0};
    ASSERT_INT_EQ(tc_parse_tool_calls_diagnosed(text, tools, TC_DIALECT_ORNITH_JSON,
                                                &calls, &n, NULL, &rep), 0);
    ASSERT_INT_EQ(n, 0);
    ASSERT_INT_EQ(rep.count, 0);

    tc_parse_repair_free(&rep);
    tc_tool_calls_free(calls, n);
    jr_free(tools);
}

/* Half a \uXXXX escape: same reason. */
static void test_repair_declines_partial_unicode_escape(void) {
    jr_value *tools = jr_parse(DEFECT_TOOLS);
    const char *text = "{\"function\":\"read\",\"arguments\":{\"path\":\"/tmp/\\u00";

    tc_tool_call *calls = NULL; int n = -1;
    tc_parse_repair rep = {0};
    ASSERT_INT_EQ(tc_parse_tool_calls_diagnosed(text, tools, TC_DIALECT_ORNITH_JSON,
                                                &calls, &n, NULL, &rep), 0);
    ASSERT_INT_EQ(n, 0);
    ASSERT_INT_EQ(rep.count, 0);

    tc_parse_repair_free(&rep);
    tc_tool_calls_free(calls, n);
    jr_free(tools);
}

/* A key cut off mid-word closes to {"pa"} — not valid JSON. The strict re-parse
 * is the backstop behind the decline rules, and it holds on its own. */
static void test_repair_declines_truncated_key(void) {
    jr_value *tools = jr_parse(DEFECT_TOOLS);
    const char *text = "{\"function\":\"read\",\"arguments\":{\"pa";

    tc_tool_call *calls = NULL; int n = -1;
    tc_parse_repair rep = {0};
    ASSERT_INT_EQ(tc_parse_tool_calls_diagnosed(text, tools, TC_DIALECT_ORNITH_JSON,
                                                &calls, &n, NULL, &rep), 0);
    ASSERT_INT_EQ(n, 0);
    ASSERT_INT_EQ(rep.count, 0);

    tc_parse_repair_free(&rep);
    tc_tool_calls_free(calls, n);
    jr_free(tools);
}

/* A repaired call is still a call: the name gate applies to it unchanged, so a
 * truncation cannot smuggle in a tool the request never declared. */
static void test_repair_still_gates_the_name(void) {
    jr_value *tools = jr_parse(DEFECT_TOOLS);
    const char *text = "{\"function\":\"serach_files\",\"arguments\":{\"path\":\"/x\"}";

    tc_tool_call *calls = NULL; int n = -1;
    tc_parse_repair rep = {0};
    ASSERT_INT_EQ(tc_parse_tool_calls_diagnosed(text, tools, TC_DIALECT_ORNITH_JSON,
                                                &calls, &n, NULL, &rep), 0);
    ASSERT_INT_EQ(n, 0);
    ASSERT_INT_EQ(rep.count, 0);

    tc_parse_repair_free(&rep);
    tc_tool_calls_free(calls, n);
    jr_free(tools);
}

/* ── The repair must be invisible to everything that already worked. ── */

/* A call that parses cleanly takes the identical path with and without the
 * repair out-param: same count, same name, same argument bytes, and the repair
 * counter never moves. The truncated branch is the ONLY place the new code is
 * reachable from, and this is the assertion that says so. */
static void test_clean_parse_is_unchanged_by_repair(void) {
    jr_value *tools = jr_parse(DEFECT_TOOLS);
    const char *text =
        "Let me look.\n```json\n{\"function\": \"read\", "
        "\"arguments\": {\"path\": \"/etc/hostname\"}}\n```";

    tc_tool_call *plain = NULL; int np = -1;
    ASSERT_INT_EQ(tc_parse_tool_calls(text, tools, TC_DIALECT_ORNITH_JSON,
                                      &plain, &np), 0);

    tc_tool_call *diag = NULL; int nd = -1;
    tc_parse_repair rep = {0};
    ASSERT_INT_EQ(tc_parse_tool_calls_diagnosed(text, tools, TC_DIALECT_ORNITH_JSON,
                                                &diag, &nd, NULL, &rep), 0);
    ASSERT_INT_EQ(np, 1);
    ASSERT_INT_EQ(nd, np);
    ASSERT_STR_EQ(diag[0].name, plain[0].name);
    ASSERT_STR_EQ(diag[0].arguments, plain[0].arguments);
    ASSERT_INT_EQ(rep.count, 0);
    ASSERT_NULL(rep.added);

    tc_parse_repair_free(&rep);
    tc_tool_calls_free(diag, nd);
    tc_tool_calls_free(plain, np);
    jr_free(tools);
}

/* The plain entry point never repairs. `rep` both enables and records the
 * repair, so a caller that is not accounting for one does not silently get one
 * — and serve_prompt_tokens.c, which only counts tokens, keeps today's answer. */
static void test_repair_is_off_without_the_out_param(void) {
    jr_value *tools = jr_parse(DEFECT_TOOLS);
    const char *text =
        "{\"function\": \"read\", \"arguments\": {\"path\": \"/etc/hostname\"}";

    tc_tool_call *calls = NULL; int n = -1;
    ASSERT_INT_EQ(tc_parse_tool_calls(text, tools, TC_DIALECT_ORNITH_JSON,
                                      &calls, &n), 0);
    ASSERT_INT_EQ(n, 0);

    tc_tool_calls_free(calls, n);
    jr_free(tools);
}

/* The XML dialect is untouched by any of this: its own truncation tolerance
 * (an unclosed </parameter> falls through to the block limit) already handles
 * the same event, and no JSON repair runs on that path. */
static void test_xml_dialect_untouched_by_repair(void) {
    jr_value *tools = jr_parse(DEFECT_TOOLS);
    const char *text = "<function=read>\n<parameter=path>\n/etc/hostname\n";

    tc_tool_call *calls = NULL; int n = -1;
    tc_parse_repair rep = {0};
    ASSERT_INT_EQ(tc_parse_tool_calls_diagnosed(text, tools, TC_DIALECT_QWEN_XML,
                                                &calls, &n, NULL, &rep), 0);
    ASSERT_INT_EQ(n, 1);
    ASSERT_STR_EQ(calls[0].arguments, "{\"path\":\"/etc/hostname\"}");
    ASSERT_INT_EQ(rep.count, 0);

    tc_parse_repair_free(&rep);
    tc_tool_calls_free(calls, n);
    jr_free(tools);
}

/* ============================================================
 * A MALFORMED CONTAINER-TYPED VALUE MUST NOT BREAK THE WHOLE CALL
 *
 * write_typed_value splices an "object"/"array" value into the arguments object
 * RAW.  Unvalidated, that is not a bad ARGUMENT — it is an `arguments` string
 * that does not parse, so the client loses every other argument of the call too
 * and gets no indication which member was at fault.  Live, three consecutive
 * `todowrite` calls (qwen36-bf16, 2026-08-26): the client reported only "arguments
 * were not valid JSON" and the model, given nothing to correct, reproduced the
 * same malformation twice more.
 *
 * These tests are the DEFENCE IN DEPTH half (pantheon-14): the decoding grammar
 * now makes the malformation unreachable, but the grammar is off by default and
 * a request with no tools never builds one, so the parser must not trust these
 * bytes either.
 * ============================================================ */

/* `todos` is a schema-typed ARRAY, `path` an ordinary string — so one call can
 * carry both a spliced value and a normal one, which is what makes "the rest of
 * the object survives" testable. */
static const char *CONTAINER_TOOLS =
    "[{\"type\":\"function\",\"function\":{\"name\":\"todowrite\","
    "\"parameters\":{\"type\":\"object\",\"properties\":"
    "{\"path\":{\"type\":\"string\"},"
    "\"todos\":{\"type\":\"array\"}},"
    "\"required\":[\"todos\"]}}}]";

static void test_malformed_container_arg_keeps_the_object_parseable(void) {
    jr_value *tools = jr_parse(CONTAINER_TOOLS);
    /* The live malformation: a bare value where a key was owed. */
    const char *text =
        "<function=todowrite>\n"
        "<parameter=path>\nplan.md\n</parameter>\n"
        "<parameter=todos>\n"
        "[{\"content\":\"a\",\"status\":\"in_progress\"},{\"content\":\"\",\"\"}]\n"
        "</parameter>\n</function>";

    tc_tool_call *calls = NULL; int n = -1;
    tc_parse_error err = {0};
    ASSERT_INT_EQ(tc_parse_tool_calls_diagnosed(text, tools, TC_DIALECT_QWEN_XML,
                                                &calls, &n, &err, NULL), 0);
    ASSERT_INT_EQ(n, 1);
    ASSERT_STR_EQ(calls[0].name, "todowrite");

    /* THE ASSERTION.  Before the fix this string was un-parseable, so `path` —
     * a perfectly good argument the model got right — was lost with it. */
    jr_value *args = jr_parse(calls[0].arguments);
    ASSERT_NOT_NULL(args);
    ASSERT_STR_EQ(jr_as_string(jr_object_get(args, "path")), "plan.md");
    /* `todos` arrives as a STRING carrying the model's bytes: the wrong type,
     * which the tool's own schema check names precisely, rather than a corrupt
     * object that names nothing. */
    ASSERT_INT_EQ(jr_kind(jr_object_get(args, "todos")), JR_STRING);
    jr_free(args);

    tc_parse_error_free(&err);
    tc_tool_calls_free(calls, n);
    jr_free(tools);
}

/* NEIGHBOUR — a WELL-FORMED array is still spliced raw and still arrives as a
 * real array.  The gate must not cost healthy traffic its types. */
static void test_wellformed_container_arg_is_still_spliced_raw(void) {
    jr_value *tools = jr_parse(CONTAINER_TOOLS);
    const char *text =
        "<function=todowrite>\n"
        "<parameter=todos>\n"
        "[{\"content\":\"a\",\"status\":\"in_progress\"}]\n"
        "</parameter>\n</function>";

    tc_tool_call *calls = NULL; int n = -1;
    tc_parse_error err = {0};
    ASSERT_INT_EQ(tc_parse_tool_calls_diagnosed(text, tools, TC_DIALECT_QWEN_XML,
                                                &calls, &n, &err, NULL), 0);
    ASSERT_INT_EQ(n, 1);
    jr_value *args = jr_parse(calls[0].arguments);
    ASSERT_NOT_NULL(args);
    const jr_value *todos = jr_object_get(args, "todos");
    ASSERT_INT_EQ(jr_kind(todos), JR_ARRAY);
    ASSERT_INT_EQ(jr_array_len(todos), 1);
    ASSERT_STR_EQ(jr_as_string(jr_object_get(jr_array_get(todos, 0), "status")),
                  "in_progress");
    jr_free(args);

    tc_parse_error_free(&err);
    tc_tool_calls_free(calls, n);
    jr_free(tools);
}

/* ============================================================
 * Main
 * ============================================================ */

int main(void) {
    printf("\n  tool-call defect detection + JSON repair tests\n\n");

    RUN_TEST(malformed_container_arg_keeps_the_object_parseable);
    RUN_TEST(wellformed_container_arg_is_still_spliced_raw);

    RUN_TEST(leaked_tail_dropped_and_named);
    RUN_TEST(leaked_tail_json_valued_parameter_survives);
    RUN_TEST(leaked_tail_inside_a_string_is_not_a_leak);
    RUN_TEST(leaked_tail_needs_the_function_close);

    RUN_TEST(reasoning_leak_named);
    RUN_TEST(reasoning_leak_silent_when_call_recovered);

    RUN_TEST(call_name_missing_named);
    RUN_TEST(call_name_unknown_named);
    RUN_TEST(bad_name_does_not_cost_the_good_call);
    RUN_TEST(no_schema_reports_nothing);

    RUN_TEST(repair_closes_outer_object);
    RUN_TEST(repair_closes_open_string);
    RUN_TEST(repair_closes_after_a_complete_value);
    RUN_TEST(repair_ignores_trailing_fence);
    RUN_TEST(repair_declines_dangling_colon);
    RUN_TEST(repair_declines_empty_open_string);
    RUN_TEST(repair_declines_empty_argument_bag);
    RUN_TEST(repair_declines_trailing_comma);
    RUN_TEST(repair_declines_mid_escape);
    RUN_TEST(repair_declines_partial_unicode_escape);
    RUN_TEST(repair_declines_truncated_key);
    RUN_TEST(repair_still_gates_the_name);
    RUN_TEST(clean_parse_is_unchanged_by_repair);
    RUN_TEST(repair_is_off_without_the_out_param);
    RUN_TEST(xml_dialect_untouched_by_repair);

    TEST_REPORT();
    return __ft_failed > 0 ? 1 : 0;
}
