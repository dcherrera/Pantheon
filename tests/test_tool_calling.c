/* SPDX-License-Identifier: AGPL-3.0-or-later */
/* Copyright (C) 2026 David (Corey) Herrera */
/*
 * Unit tests for the tool-call parser (src/serve/tool_calling.c).
 *
 * Focus of these tests is tc_parse_tool_calls' wrapper-tolerant parsing: the
 * canonical wrapped <tool_call><function=…></function></tool_call> form AND a
 * bare <function=…></function> block with no wrapper (which the Q3 nex-n2-mini
 * emits most of the time) must both parse into identical structured calls, and
 * text with no <function= at all must yield zero calls. Pure C string parsing,
 * no model/engine deps — fully local (no CUDA/BLAS needed).
 *
 * The last section covers the ABANDONED call (tc_parse_tool_calls_diagnosed)
 * against the fixture in abandoned_call_fixture.h, together with the
 * neighbours it must NOT be confused with — an empty argument, a well-formed
 * call, a turn with no opener, and a merely-malformed close over a good value.
 */

#include "lib/test.h"
#include "abandoned_call_fixture.h"
#include "../src/serve/tool_calling.h"
#include "../src/serve/json_reader.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

/* A one-function tools schema (get_weather: location string, unit string) so
 * parameter values get schema-typed exactly like the serve path does. */
static const char *TOOLS_JSON =
    "[{\"type\":\"function\",\"function\":{\"name\":\"get_weather\","
    "\"description\":\"Get the weather.\",\"parameters\":{\"type\":\"object\","
    "\"properties\":{\"location\":{\"type\":\"string\"},"
    "\"unit\":{\"type\":\"string\"},\"days\":{\"type\":\"integer\"}},"
    "\"required\":[\"location\"]}}}]";

/* ============================================================
 * Wrapped form (canonical) still parses — regression guard.
 * ============================================================ */

static void test_wrapped_single(void) {
    jr_value *tools = jr_parse(TOOLS_JSON);
    const char *text =
        "<tool_call>\n<function=get_weather>\n"
        "<parameter=location>\nParis\n</parameter>\n"
        "</function>\n</tool_call>";

    tc_tool_call *calls = NULL; int n = -1;
    int rc = tc_parse_tool_calls(text, tools, TC_DIALECT_QWEN_XML, &calls, &n);
    ASSERT_INT_EQ(rc, 0);
    ASSERT_INT_EQ(n, 1);
    ASSERT_NOT_NULL(calls);
    ASSERT_STR_EQ(calls[0].name, "get_weather");
    ASSERT_STR_EQ(calls[0].arguments, "{\"location\":\"Paris\"}");
    ASSERT_TRUE(strncmp(calls[0].id, "call_", 5) == 0);

    tc_tool_calls_free(calls, n);
    jr_free(tools);
}

/* ============================================================
 * Bare form (no <tool_call> wrapper) — the Phase 1 fix.
 * ============================================================ */

static void test_bare_single(void) {
    jr_value *tools = jr_parse(TOOLS_JSON);
    const char *text =
        "<function=get_weather>\n"
        "<parameter=location>\nParis\n</parameter>\n"
        "</function>";

    tc_tool_call *calls = NULL; int n = -1;
    int rc = tc_parse_tool_calls(text, tools, TC_DIALECT_QWEN_XML, &calls, &n);
    ASSERT_INT_EQ(rc, 0);
    ASSERT_INT_EQ(n, 1);
    ASSERT_NOT_NULL(calls);
    ASSERT_STR_EQ(calls[0].name, "get_weather");
    ASSERT_STR_EQ(calls[0].arguments, "{\"location\":\"Paris\"}");
    ASSERT_TRUE(strncmp(calls[0].id, "call_", 5) == 0);

    tc_tool_calls_free(calls, n);
    jr_free(tools);
}

/* Bare and wrapped forms of the same call yield identical name + arguments
 * (only the random id differs) — proves output parity. */
static void test_bare_matches_wrapped(void) {
    jr_value *tools = jr_parse(TOOLS_JSON);
    const char *bare =
        "<function=get_weather>\n<parameter=location>\nParis\n</parameter>\n"
        "<parameter=unit>\ncelsius\n</parameter>\n</function>";
    const char *wrapped =
        "<tool_call>\n<function=get_weather>\n<parameter=location>\nParis\n</parameter>\n"
        "<parameter=unit>\ncelsius\n</parameter>\n</function>\n</tool_call>";

    tc_tool_call *cb = NULL, *cw = NULL; int nb = -1, nw = -1;
    ASSERT_INT_EQ(tc_parse_tool_calls(bare, tools, TC_DIALECT_QWEN_XML, &cb, &nb), 0);
    ASSERT_INT_EQ(tc_parse_tool_calls(wrapped, tools, TC_DIALECT_QWEN_XML, &cw, &nw), 0);
    ASSERT_INT_EQ(nb, 1);
    ASSERT_INT_EQ(nw, 1);
    ASSERT_STR_EQ(cb[0].name, cw[0].name);
    ASSERT_STR_EQ(cb[0].arguments, cw[0].arguments);
    ASSERT_STR_EQ(cb[0].arguments, "{\"location\":\"Paris\",\"unit\":\"celsius\"}");

    tc_tool_calls_free(cb, nb);
    tc_tool_calls_free(cw, nw);
    jr_free(tools);
}

/* Integer schema typing survives the bare path (days -> bare literal). */
static void test_bare_typed_integer(void) {
    jr_value *tools = jr_parse(TOOLS_JSON);
    const char *text =
        "<function=get_weather>\n<parameter=location>\nParis\n</parameter>\n"
        "<parameter=days>\n3\n</parameter>\n</function>";

    tc_tool_call *calls = NULL; int n = -1;
    ASSERT_INT_EQ(tc_parse_tool_calls(text, tools, TC_DIALECT_QWEN_XML, &calls, &n), 0);
    ASSERT_INT_EQ(n, 1);
    ASSERT_STR_EQ(calls[0].arguments, "{\"location\":\"Paris\",\"days\":3}");

    tc_tool_calls_free(calls, n);
    jr_free(tools);
}

/* ============================================================
 * Multiple calls in one message (bare + wrapped).
 * ============================================================ */

static void test_bare_multi(void) {
    jr_value *tools = jr_parse(TOOLS_JSON);
    const char *text =
        "<function=get_weather>\n<parameter=location>\nParis\n</parameter>\n</function>\n"
        "<function=get_weather>\n<parameter=location>\nTokyo\n</parameter>\n</function>";

    tc_tool_call *calls = NULL; int n = -1;
    ASSERT_INT_EQ(tc_parse_tool_calls(text, tools, TC_DIALECT_QWEN_XML, &calls, &n), 0);
    ASSERT_INT_EQ(n, 2);
    ASSERT_STR_EQ(calls[0].arguments, "{\"location\":\"Paris\"}");
    ASSERT_STR_EQ(calls[1].arguments, "{\"location\":\"Tokyo\"}");

    tc_tool_calls_free(calls, n);
    jr_free(tools);
}

static void test_wrapped_multi(void) {
    jr_value *tools = jr_parse(TOOLS_JSON);
    const char *text =
        "<tool_call>\n<function=get_weather>\n<parameter=location>\nParis\n</parameter>\n</function>\n</tool_call>\n"
        "<tool_call>\n<function=get_weather>\n<parameter=location>\nTokyo\n</parameter>\n</function>\n</tool_call>";

    tc_tool_call *calls = NULL; int n = -1;
    ASSERT_INT_EQ(tc_parse_tool_calls(text, tools, TC_DIALECT_QWEN_XML, &calls, &n), 0);
    ASSERT_INT_EQ(n, 2);
    ASSERT_STR_EQ(calls[0].arguments, "{\"location\":\"Paris\"}");
    ASSERT_STR_EQ(calls[1].arguments, "{\"location\":\"Tokyo\"}");

    tc_tool_calls_free(calls, n);
    jr_free(tools);
}

/* ============================================================
 * Real-world noise tolerance.
 * ============================================================ */

/* Natural-language reasoning before the first bare <function= is skipped. */
static void test_reasoning_then_bare_call(void) {
    jr_value *tools = jr_parse(TOOLS_JSON);
    const char *text =
        "I'll check the weather for you now.\n"
        "<function=get_weather>\n<parameter=location>\nParis\n</parameter>\n</function>";

    tc_tool_call *calls = NULL; int n = -1;
    ASSERT_INT_EQ(tc_parse_tool_calls(text, tools, TC_DIALECT_QWEN_XML, &calls, &n), 0);
    ASSERT_INT_EQ(n, 1);
    ASSERT_STR_EQ(calls[0].name, "get_weather");
    ASSERT_STR_EQ(calls[0].arguments, "{\"location\":\"Paris\"}");

    tc_tool_calls_free(calls, n);
    jr_free(tools);
}

/* A doubled/again </function> after a valid bare call must not crash or emit a
 * second garbage call. */
static void test_bare_doubled_close(void) {
    jr_value *tools = jr_parse(TOOLS_JSON);
    const char *text =
        "<function=get_weather>\n<parameter=location>\nParis\n</parameter>\n"
        "</function>\n</function>";

    tc_tool_call *calls = NULL; int n = -1;
    ASSERT_INT_EQ(tc_parse_tool_calls(text, tools, TC_DIALECT_QWEN_XML, &calls, &n), 0);
    ASSERT_INT_EQ(n, 1);
    ASSERT_STR_EQ(calls[0].arguments, "{\"location\":\"Paris\"}");

    tc_tool_calls_free(calls, n);
    jr_free(tools);
}

/* A trailing </tool_call> with no opener alongside a bare call: since a real
 * TC_OPEN is present in this shape only if wrapped, the bare scan handles it and
 * the stray closer is inert. */
static void test_bare_call_stray_closer(void) {
    jr_value *tools = jr_parse(TOOLS_JSON);
    const char *text =
        "<function=get_weather>\n<parameter=location>\nParis\n</parameter>\n"
        "</function>\n</tool_call>";

    tc_tool_call *calls = NULL; int n = -1;
    ASSERT_INT_EQ(tc_parse_tool_calls(text, tools, TC_DIALECT_QWEN_XML, &calls, &n), 0);
    ASSERT_INT_EQ(n, 1);
    ASSERT_STR_EQ(calls[0].arguments, "{\"location\":\"Paris\"}");

    tc_tool_calls_free(calls, n);
    jr_free(tools);
}

/* ============================================================
 * No false positives.
 * ============================================================ */

/* Plain answer text with no <function= at all -> zero calls. */
static void test_no_tools_text(void) {
    jr_value *tools = jr_parse(TOOLS_JSON);
    const char *text = "The weather in Paris is sunny and 18 degrees celsius.";

    tc_tool_call *calls = NULL; int n = -1;
    ASSERT_INT_EQ(tc_parse_tool_calls(text, tools, TC_DIALECT_QWEN_XML, &calls, &n), 0);
    ASSERT_INT_EQ(n, 0);
    ASSERT_NULL(calls);

    tc_tool_calls_free(calls, n);
    jr_free(tools);
}

/* Empty string -> zero calls. */
static void test_empty(void) {
    tc_tool_call *calls = (tc_tool_call *)0x1; int n = -1;
    ASSERT_INT_EQ(tc_parse_tool_calls("", NULL, TC_DIALECT_QWEN_XML, &calls, &n), 0);
    ASSERT_INT_EQ(n, 0);
    ASSERT_NULL(calls);
}

/* NULL text -> zero calls, no crash. */
static void test_null_text(void) {
    tc_tool_call *calls = (tc_tool_call *)0x1; int n = -1;
    ASSERT_INT_EQ(tc_parse_tool_calls(NULL, NULL, TC_DIALECT_QWEN_XML, &calls, &n), 0);
    ASSERT_INT_EQ(n, 0);
    ASSERT_NULL(calls);
}

/* A stray "<function=" with no '>'-terminated name must not crash or emit a
 * garbage call. */
static void test_garbage_unterminated_name(void) {
    jr_value *tools = jr_parse(TOOLS_JSON);
    const char *text = "here is a stray <function= tag with no close at all";

    tc_tool_call *calls = NULL; int n = -1;
    ASSERT_INT_EQ(tc_parse_tool_calls(text, tools, TC_DIALECT_QWEN_XML, &calls, &n), 0);
    ASSERT_INT_EQ(n, 0);
    ASSERT_NULL(calls);

    tc_tool_calls_free(calls, n);
    jr_free(tools);
}

/* An empty-name "<function=>" (has a '>' but no name) emits no call. */
static void test_garbage_empty_name(void) {
    jr_value *tools = jr_parse(TOOLS_JSON);
    const char *text = "<function=></function>";

    tc_tool_call *calls = NULL; int n = -1;
    ASSERT_INT_EQ(tc_parse_tool_calls(text, tools, TC_DIALECT_QWEN_XML, &calls, &n), 0);
    ASSERT_INT_EQ(n, 0);

    tc_tool_calls_free(calls, n);
    jr_free(tools);
}

/* ============================================================
 * Name-set gate (defense-in-depth) — a hallucinated tool name that is NOT in
 * the request's tools schema must NEVER surface as a tool_call, even when it is
 * otherwise well-formed XML.  This is the parser-side backstop for the observed
 * saber-orch degeneration: PANTHEON_TOOL_GRAMMAR=1 constrains the NAME only once
 * the armed skeleton is entered, but a model that degenerates in the reasoning
 * region (or a fail-open step) can leave a garbage "<function=NAME>" in the raw
 * text — the parser must reject any NAME outside the tools set.
 * ============================================================ */

/* The live-observed failure: a garbage hallucinated name with empty args, in an
 * otherwise well-formed bare block, is dropped (0 calls). */
static void test_hallucinated_name_rejected(void) {
    jr_value *tools = jr_parse(TOOLS_JSON);
    const char *text =
        "<function=base_agent_agent_agent_tool_tool_call_call_action_p2555577766>\n"
        "</function>";

    tc_tool_call *calls = NULL; int n = -1;
    ASSERT_INT_EQ(tc_parse_tool_calls(text, tools, TC_DIALECT_QWEN_XML, &calls, &n), 0);
    ASSERT_INT_EQ(n, 0);
    ASSERT_NULL(calls);

    tc_tool_calls_free(calls, n);
    jr_free(tools);
}

/* Wrapped form with a hallucinated name is likewise dropped. */
static void test_hallucinated_name_rejected_wrapped(void) {
    jr_value *tools = jr_parse(TOOLS_JSON);
    const char *text =
        "<tool_call>\n<function=not_a_real_tool>\n"
        "<parameter=location>\nParis\n</parameter>\n"
        "</function>\n</tool_call>";

    tc_tool_call *calls = NULL; int n = -1;
    ASSERT_INT_EQ(tc_parse_tool_calls(text, tools, TC_DIALECT_QWEN_XML, &calls, &n), 0);
    ASSERT_INT_EQ(n, 0);

    tc_tool_calls_free(calls, n);
    jr_free(tools);
}

/* A KNOWN name is unaffected — a garbage block adjacent to a valid one keeps the
 * valid call and drops only the bogus one. */
static void test_known_name_kept_bogus_dropped(void) {
    jr_value *tools = jr_parse(TOOLS_JSON);
    const char *text =
        "<function=bogus_name>\n</function>\n"
        "<function=get_weather>\n"
        "<parameter=location>\nParis\n</parameter>\n"
        "</function>";

    tc_tool_call *calls = NULL; int n = -1;
    ASSERT_INT_EQ(tc_parse_tool_calls(text, tools, TC_DIALECT_QWEN_XML, &calls, &n), 0);
    ASSERT_INT_EQ(n, 1);
    ASSERT_NOT_NULL(calls);
    ASSERT_STR_EQ(calls[0].name, "get_weather");
    ASSERT_STR_EQ(calls[0].arguments, "{\"location\":\"Paris\"}");

    tc_tool_calls_free(calls, n);
    jr_free(tools);
}

/* With NO tools schema the gate is permissive (nothing to validate against) —
 * preserves the prior behavior for schema-less parses. */
static void test_no_schema_permissive(void) {
    const char *text =
        "<function=anything_goes>\n"
        "<parameter=x>\ny\n</parameter>\n"
        "</function>";

    tc_tool_call *calls = NULL; int n = -1;
    ASSERT_INT_EQ(tc_parse_tool_calls(text, NULL, TC_DIALECT_QWEN_XML, &calls, &n), 0);
    ASSERT_INT_EQ(n, 1);
    ASSERT_NOT_NULL(calls);
    ASSERT_STR_EQ(calls[0].name, "anything_goes");

    tc_tool_calls_free(calls, n);
    jr_free(NULL);
}

/* ============================================================
 * MiniCPM5 attribute-style dialect: bare, unwrapped
 *   <function name="NAME"><param name="KEY">VALUE</param>…</function>
 * (quoted-attribute name/keys, optional CDATA values). Disjoint from the
 * Qwen "<function=" / "<parameter=" forms above.
 * ============================================================ */

static void test_minicpm_single(void) {
    jr_value *tools = jr_parse(TOOLS_JSON);
    const char *text =
        "<function name=\"get_weather\">"
        "<param name=\"location\">Paris</param></function>";

    tc_tool_call *calls = NULL; int n = -1;
    ASSERT_INT_EQ(tc_parse_tool_calls(text, tools, TC_DIALECT_QWEN_XML, &calls, &n), 0);
    ASSERT_INT_EQ(n, 1);
    ASSERT_NOT_NULL(calls);
    ASSERT_STR_EQ(calls[0].name, "get_weather");
    ASSERT_STR_EQ(calls[0].arguments, "{\"location\":\"Paris\"}");
    ASSERT_TRUE(strncmp(calls[0].id, "call_", 5) == 0);

    tc_tool_calls_free(calls, n);
    jr_free(tools);
}

/* Typed multi-param: integer stays a JSON literal, string stays quoted. */
static void test_minicpm_typed_multi(void) {
    jr_value *tools = jr_parse(TOOLS_JSON);
    const char *text =
        "<function name=\"get_weather\">"
        "<param name=\"location\">Paris</param>"
        "<param name=\"days\">3</param></function>";

    tc_tool_call *calls = NULL; int n = -1;
    ASSERT_INT_EQ(tc_parse_tool_calls(text, tools, TC_DIALECT_QWEN_XML, &calls, &n), 0);
    ASSERT_INT_EQ(n, 1);
    ASSERT_STR_EQ(calls[0].name, "get_weather");
    ASSERT_STR_EQ(calls[0].arguments,
                  "{\"location\":\"Paris\",\"days\":3}");

    tc_tool_calls_free(calls, n);
    jr_free(tools);
}

/* CDATA-wrapped multi-line value is taken verbatim (wrapper stripped). */
static void test_minicpm_cdata(void) {
    jr_value *tools = jr_parse(TOOLS_JSON);
    const char *text =
        "<function name=\"get_weather\">"
        "<param name=\"location\"><![CDATA[Paris\n& Berlin]]></param>"
        "</function>";

    tc_tool_call *calls = NULL; int n = -1;
    ASSERT_INT_EQ(tc_parse_tool_calls(text, tools, TC_DIALECT_QWEN_XML, &calls, &n), 0);
    ASSERT_INT_EQ(n, 1);
    ASSERT_STR_EQ(calls[0].arguments, "{\"location\":\"Paris\\n& Berlin\"}");

    tc_tool_calls_free(calls, n);
    jr_free(tools);
}

/* Single-quoted attributes are accepted. */
static void test_minicpm_single_quotes(void) {
    jr_value *tools = jr_parse(TOOLS_JSON);
    const char *text =
        "<function name='get_weather'>"
        "<param name='location'>Paris</param></function>";

    tc_tool_call *calls = NULL; int n = -1;
    ASSERT_INT_EQ(tc_parse_tool_calls(text, tools, TC_DIALECT_QWEN_XML, &calls, &n), 0);
    ASSERT_INT_EQ(n, 1);
    ASSERT_STR_EQ(calls[0].name, "get_weather");
    ASSERT_STR_EQ(calls[0].arguments, "{\"location\":\"Paris\"}");

    tc_tool_calls_free(calls, n);
    jr_free(tools);
}

/* Leading reasoning prose before the call is skipped; hallucinated names are
 * still gated by the schema. */
static void test_minicpm_reasoning_prefix(void) {
    jr_value *tools = jr_parse(TOOLS_JSON);
    const char *text =
        "Let me check the weather for you.\n"
        "<function name=\"get_weather\">"
        "<param name=\"location\">Paris</param></function>";

    tc_tool_call *calls = NULL; int n = -1;
    ASSERT_INT_EQ(tc_parse_tool_calls(text, tools, TC_DIALECT_QWEN_XML, &calls, &n), 0);
    ASSERT_INT_EQ(n, 1);
    ASSERT_STR_EQ(calls[0].name, "get_weather");

    tc_tool_calls_free(calls, n);
    jr_free(tools);
}

/* The native preamble renderer emits the <tools> block + <function name=…>
 * instructions (not the Qwen <parameter=…> form). */
static void test_minicpm_preamble(void) {
    jr_value *tools = jr_parse(TOOLS_JSON);
    char *pre = tc_render_tools_system_minicpm(tools, NULL);
    ASSERT_NOT_NULL(pre);
    ASSERT_TRUE(strstr(pre, "<tools>") != NULL);
    ASSERT_TRUE(strstr(pre, "<function name=\"function-name\">") != NULL);
    ASSERT_TRUE(strstr(pre, "<param name=\"param-name\">") != NULL);
    ASSERT_TRUE(strstr(pre, "get_weather") != NULL);   /* the tool JSON */
    /* Must NOT carry the Qwen dialect's example form. */
    ASSERT_TRUE(strstr(pre, "<parameter=") == NULL);
    free(pre);
    jr_free(tools);
}

/* ============================================================
 * Grammar-forced dialect: the body-style <function=NAME> / <parameter=KEY>
 * (like the bare Qwen form) but with an extra <value>…</value> wrapper around
 * each value:
 *   <function=NAME>\n<parameter=KEY>\n<value>\nVALUE\n</value>\n</parameter>\n</function>
 * The generation grammar accepts the wrapper as free VALUE text, so the parser
 * must peel it back to the bare VALUE.  Regression: without the fix the value
 * captured as the literal "<value>\nVALUE\n</value>" text.
 * ============================================================ */

static void test_forced_value_wrapper_bare(void) {
    jr_value *tools = jr_parse(TOOLS_JSON);
    const char *text =
        "<function=get_weather>\n<parameter=location>\n"
        "<value>\nParis\n</value>\n</parameter>\n</function>";

    tc_tool_call *calls = NULL; int n = -1;
    ASSERT_INT_EQ(tc_parse_tool_calls(text, tools, TC_DIALECT_QWEN_XML, &calls, &n), 0);
    ASSERT_INT_EQ(n, 1);
    ASSERT_NOT_NULL(calls);
    ASSERT_STR_EQ(calls[0].name, "get_weather");
    ASSERT_STR_EQ(calls[0].arguments, "{\"location\":\"Paris\"}");

    tc_tool_calls_free(calls, n);
    jr_free(tools);
}

/* Same forced dialect INSIDE a <tool_call> wrapper (the grammar's wrapped form
 * with the <value> wrapper). */
static void test_forced_value_wrapper_wrapped(void) {
    jr_value *tools = jr_parse(TOOLS_JSON);
    const char *text =
        "<tool_call>\n<function=get_weather>\n<parameter=location>\n"
        "<value>\nParis\n</value>\n</parameter>\n</function>\n</tool_call>";

    tc_tool_call *calls = NULL; int n = -1;
    ASSERT_INT_EQ(tc_parse_tool_calls(text, tools, TC_DIALECT_QWEN_XML, &calls, &n), 0);
    ASSERT_INT_EQ(n, 1);
    ASSERT_STR_EQ(calls[0].name, "get_weather");
    ASSERT_STR_EQ(calls[0].arguments, "{\"location\":\"Paris\"}");

    tc_tool_calls_free(calls, n);
    jr_free(tools);
}

/* A multi-line value inside the <value> wrapper keeps its interior newlines. */
static void test_forced_value_wrapper_multiline(void) {
    jr_value *tools = jr_parse(TOOLS_JSON);
    const char *text =
        "<function=get_weather>\n<parameter=location>\n"
        "<value>\nParis\nand Berlin\n</value>\n</parameter>\n</function>";

    tc_tool_call *calls = NULL; int n = -1;
    ASSERT_INT_EQ(tc_parse_tool_calls(text, tools, TC_DIALECT_QWEN_XML, &calls, &n), 0);
    ASSERT_INT_EQ(n, 1);
    ASSERT_STR_EQ(calls[0].arguments, "{\"location\":\"Paris\\nand Berlin\"}");

    tc_tool_calls_free(calls, n);
    jr_free(tools);
}

/* Typed integer survives the <value> wrapper (unwrapped, then schema-typed). */
static void test_forced_value_wrapper_typed_integer(void) {
    jr_value *tools = jr_parse(TOOLS_JSON);
    const char *text =
        "<function=get_weather>\n<parameter=location>\n"
        "<value>\nParis\n</value>\n</parameter>\n"
        "<parameter=days>\n<value>\n3\n</value>\n</parameter>\n</function>";

    tc_tool_call *calls = NULL; int n = -1;
    ASSERT_INT_EQ(tc_parse_tool_calls(text, tools, TC_DIALECT_QWEN_XML, &calls, &n), 0);
    ASSERT_INT_EQ(n, 1);
    ASSERT_STR_EQ(calls[0].arguments, "{\"location\":\"Paris\",\"days\":3}");

    tc_tool_calls_free(calls, n);
    jr_free(tools);
}

/* An empty <value></value> inner yields an EMPTY arg, not a missing one: the
 * model wrote both halves of the wrapper, so it closed the value deliberately,
 * and the parameter-level close is the whole empty-value test. */
static void test_forced_value_wrapper_empty_inner(void) {
    jr_value *tools = jr_parse(TOOLS_JSON);
    const char *text =
        "<function=get_weather>\n<parameter=location>\n"
        "<value>\n</value>\n</parameter>\n</function>";

    tc_tool_call *calls = NULL; int n = -1;
    ASSERT_INT_EQ(tc_parse_tool_calls(text, tools, TC_DIALECT_QWEN_XML, &calls, &n), 0);
    ASSERT_INT_EQ(n, 1);
    /* name kept, and the closed-around-nothing value is kept as "". */
    ASSERT_STR_EQ(calls[0].arguments, "{\"location\":\"\"}");

    tc_tool_calls_free(calls, n);
    jr_free(tools);
}

/* The forced-dialect value and the plain bare value of the same call yield
 * identical structured args (only the id differs) — proves the wrapper is fully
 * transparent to the client. */
static void test_forced_value_matches_plain(void) {
    jr_value *tools = jr_parse(TOOLS_JSON);
    const char *wrapped_val =
        "<function=get_weather>\n<parameter=location>\n"
        "<value>\nParis\n</value>\n</parameter>\n</function>";
    const char *plain_val =
        "<function=get_weather>\n<parameter=location>\nParis\n</parameter>\n</function>";

    tc_tool_call *cw = NULL, *cp = NULL; int nw = -1, np = -1;
    ASSERT_INT_EQ(tc_parse_tool_calls(wrapped_val, tools, TC_DIALECT_QWEN_XML, &cw, &nw), 0);
    ASSERT_INT_EQ(tc_parse_tool_calls(plain_val, tools, TC_DIALECT_QWEN_XML, &cp, &np), 0);
    ASSERT_INT_EQ(nw, 1);
    ASSERT_INT_EQ(np, 1);
    ASSERT_STR_EQ(cw[0].arguments, cp[0].arguments);

    tc_tool_calls_free(cw, nw);
    tc_tool_calls_free(cp, np);
    jr_free(tools);
}

/* Live MiniCPM-Q8 quirk: the value is <value>-wrapped AND the </parameter> close
 * is emitted malformed ("</paramet>"). PARAM_CLOSE is then never found, so a
 * naive scan drags the wrapper + junk into the value. Anchoring on </value> (the
 * model closes it reliably) recovers the clean value regardless. */
static void test_forced_value_wrapper_malformed_param_close(void) {
    jr_value *tools = jr_parse(TOOLS_JSON);
    const char *text =
        "<function=get_weather>\n<parameter=location>\n"
        "<value>Paris</value>\n</paramet>\n</function>";

    tc_tool_call *calls = NULL; int n = -1;
    ASSERT_INT_EQ(tc_parse_tool_calls(text, tools, TC_DIALECT_QWEN_XML, &calls, &n), 0);
    ASSERT_INT_EQ(n, 1);
    ASSERT_STR_EQ(calls[0].name, "get_weather");
    ASSERT_STR_EQ(calls[0].arguments, "{\"location\":\"Paris\"}");

    tc_tool_calls_free(calls, n);
    jr_free(tools);
}

/* Multiple <value>-wrapped params, each with the malformed close, still parse. */
static void test_forced_value_wrapper_multi_malformed(void) {
    jr_value *tools = jr_parse(TOOLS_JSON);
    const char *text =
        "<function=get_weather>\n"
        "<parameter=location>\n<value>Paris</value>\n</paramet>\n"
        "<parameter=days>\n<value>3</value>\n</paramet>\n</function>";

    tc_tool_call *calls = NULL; int n = -1;
    ASSERT_INT_EQ(tc_parse_tool_calls(text, tools, TC_DIALECT_QWEN_XML, &calls, &n), 0);
    ASSERT_INT_EQ(n, 1);
    ASSERT_STR_EQ(calls[0].arguments, "{\"location\":\"Paris\",\"days\":3}");

    tc_tool_calls_free(calls, n);
    jr_free(tools);
}

/* A bare value that merely CONTAINS "<value>" mid-text (not a full enclosing
 * wrapper) is NOT unwrapped — only a value that both starts with <value> and
 * ends with </value> is a wrapper. */
static void test_forced_value_partial_not_unwrapped(void) {
    jr_value *tools = jr_parse(TOOLS_JSON);
    const char *text =
        "<function=get_weather>\n<parameter=location>\n"
        "Paris <value> not a wrapper\n</parameter>\n</function>";

    tc_tool_call *calls = NULL; int n = -1;
    ASSERT_INT_EQ(tc_parse_tool_calls(text, tools, TC_DIALECT_QWEN_XML, &calls, &n), 0);
    ASSERT_INT_EQ(n, 1);
    ASSERT_STR_EQ(calls[0].arguments, "{\"location\":\"Paris <value> not a wrapper\"}");

    tc_tool_calls_free(calls, n);
    jr_free(tools);
}

/* ============================================================
 * Ornith JSON dialect (TC_DIALECT_ORNITH_JSON) — Phase 1.
 * The model emits a markdown-fenced or bare {"function"|"name":NAME,
 * "arguments":{…}} object instead of XML.
 * ============================================================ */

/* A read-tool schema (path string) matching a live agent client's envelope. */
static const char *READ_TOOLS =
    "[{\"type\":\"function\",\"function\":{\"name\":\"read\","
    "\"parameters\":{\"type\":\"object\","
    "\"properties\":{\"path\":{\"type\":\"string\"}},"
    "\"required\":[\"path\"]}}}]";

static void test_ornith_fenced_single(void) {
    jr_value *tools = jr_parse(READ_TOOLS);
    const char *text =
        "```json\n{\n  \"function\": \"read\",\n"
        "  \"arguments\": {\n    \"path\": \"/etc/hostname\"\n  }\n}\n```";

    tc_tool_call *calls = NULL; int n = -1;
    ASSERT_INT_EQ(tc_parse_tool_calls(text, tools, TC_DIALECT_ORNITH_JSON, &calls, &n), 0);
    ASSERT_INT_EQ(n, 1);
    ASSERT_STR_EQ(calls[0].name, "read");
    ASSERT_STR_EQ(calls[0].arguments, "{\"path\":\"/etc/hostname\"}");
    ASSERT_TRUE(strncmp(calls[0].id, "call_", 5) == 0);

    tc_tool_calls_free(calls, n);
    jr_free(tools);
}

/* A bare (unfenced) object with reasoning prose before it. */
static void test_ornith_bare_with_prose(void) {
    jr_value *tools = jr_parse(READ_TOOLS);
    const char *text =
        "Let me read that file for you.\n"
        "{\"function\": \"read\", \"arguments\": {\"path\": \"/tmp/a.txt\"}}";

    tc_tool_call *calls = NULL; int n = -1;
    ASSERT_INT_EQ(tc_parse_tool_calls(text, tools, TC_DIALECT_ORNITH_JSON, &calls, &n), 0);
    ASSERT_INT_EQ(n, 1);
    ASSERT_STR_EQ(calls[0].name, "read");
    ASSERT_STR_EQ(calls[0].arguments, "{\"path\":\"/tmp/a.txt\"}");

    tc_tool_calls_free(calls, n);
    jr_free(tools);
}

/* The "name" key spelling (some emissions use it instead of "function"). */
static void test_ornith_name_key(void) {
    jr_value *tools = jr_parse(READ_TOOLS);
    const char *text =
        "```json\n{\"name\": \"read\", \"arguments\": {\"path\": \"/x\"}}\n```";

    tc_tool_call *calls = NULL; int n = -1;
    ASSERT_INT_EQ(tc_parse_tool_calls(text, tools, TC_DIALECT_ORNITH_JSON, &calls, &n), 0);
    ASSERT_INT_EQ(n, 1);
    ASSERT_STR_EQ(calls[0].name, "read");
    ASSERT_STR_EQ(calls[0].arguments, "{\"path\":\"/x\"}");

    tc_tool_calls_free(calls, n);
    jr_free(tools);
}

/* Schema typing: integer args stay JSON literals through the round-trip
 * (arguments objects are re-serialized from the parsed DOM). */
static void test_ornith_typed_args(void) {
    jr_value *tools = jr_parse(TOOLS_JSON);   /* get_weather: location, unit, days(int) */
    const char *text =
        "{\"function\": \"get_weather\", \"arguments\": "
        "{\"location\": \"Paris\", \"days\": 3}}";

    tc_tool_call *calls = NULL; int n = -1;
    ASSERT_INT_EQ(tc_parse_tool_calls(text, tools, TC_DIALECT_ORNITH_JSON, &calls, &n), 0);
    ASSERT_INT_EQ(n, 1);
    ASSERT_STR_EQ(calls[0].name, "get_weather");
    ASSERT_STR_EQ(calls[0].arguments, "{\"location\":\"Paris\",\"days\":3}");

    tc_tool_calls_free(calls, n);
    jr_free(tools);
}

/* Parallel calls: two fenced JSON blocks in one emission. */
static void test_ornith_parallel_blocks(void) {
    jr_value *tools = jr_parse(READ_TOOLS);
    const char *text =
        "```json\n{\"function\":\"read\",\"arguments\":{\"path\":\"/a\"}}\n```\n"
        "```json\n{\"function\":\"read\",\"arguments\":{\"path\":\"/b\"}}\n```";

    tc_tool_call *calls = NULL; int n = -1;
    ASSERT_INT_EQ(tc_parse_tool_calls(text, tools, TC_DIALECT_ORNITH_JSON, &calls, &n), 0);
    ASSERT_INT_EQ(n, 2);
    ASSERT_STR_EQ(calls[0].arguments, "{\"path\":\"/a\"}");
    ASSERT_STR_EQ(calls[1].arguments, "{\"path\":\"/b\"}");

    tc_tool_calls_free(calls, n);
    jr_free(tools);
}

/* An unknown function name is dropped (never surfaced as a bogus call). */
static void test_ornith_unknown_name_dropped(void) {
    jr_value *tools = jr_parse(READ_TOOLS);
    const char *text =
        "{\"function\":\"delete_everything\",\"arguments\":{\"path\":\"/\"}}";

    tc_tool_call *calls = NULL; int n = -1;
    ASSERT_INT_EQ(tc_parse_tool_calls(text, tools, TC_DIALECT_ORNITH_JSON, &calls, &n), 0);
    ASSERT_INT_EQ(n, 0);

    tc_tool_calls_free(calls, n);
    jr_free(tools);
}

/* Malformed / truncated JSON → zero calls (never a bogus call). */
static void test_ornith_truncated_zero(void) {
    jr_value *tools = jr_parse(READ_TOOLS);
    const char *text = "```json\n{\"function\":\"read\",\"arguments\":{\"path\":";

    tc_tool_call *calls = NULL; int n = -1;
    ASSERT_INT_EQ(tc_parse_tool_calls(text, tools, TC_DIALECT_ORNITH_JSON, &calls, &n), 0);
    ASSERT_INT_EQ(n, 0);

    tc_tool_calls_free(calls, n);
    jr_free(tools);
}

/* Plain prose with no JSON object → zero calls. */
static void test_ornith_prose_only(void) {
    jr_value *tools = jr_parse(READ_TOOLS);
    const char *text = "I don't think I need a tool for this one.";

    tc_tool_call *calls = NULL; int n = -1;
    ASSERT_INT_EQ(tc_parse_tool_calls(text, tools, TC_DIALECT_ORNITH_JSON, &calls, &n), 0);
    ASSERT_INT_EQ(n, 0);

    tc_tool_calls_free(calls, n);
    jr_free(tools);
}

/* Render→parse round-trip for ornith: history render produces a fenced JSON
 * block the ornith parser reads back into the identical call. */
static void test_ornith_render_roundtrip(void) {
    jr_value *tools = jr_parse(READ_TOOLS);
    const char *calls_json =
        "[{\"id\":\"call_1\",\"type\":\"function\",\"function\":"
        "{\"name\":\"read\",\"arguments\":\"{\\\"path\\\":\\\"/etc/hostname\\\"}\"}}]";
    jr_value *tc = jr_parse(calls_json);

    char *rendered = tc_render_assistant_tool_calls(tc, true, TC_DIALECT_ORNITH_JSON);
    ASSERT_NOT_NULL(rendered);
    ASSERT_TRUE(strstr(rendered, "```json") != NULL);

    tc_tool_call *calls = NULL; int n = -1;
    ASSERT_INT_EQ(tc_parse_tool_calls(rendered, tools, TC_DIALECT_ORNITH_JSON, &calls, &n), 0);
    ASSERT_INT_EQ(n, 1);
    ASSERT_STR_EQ(calls[0].name, "read");
    ASSERT_STR_EQ(calls[0].arguments, "{\"path\":\"/etc/hostname\"}");

    tc_tool_calls_free(calls, n);
    free(rendered);
    jr_free(tc);
    jr_free(tools);
}

/* A bare "{" in a QWEN_XML request must NOT be parsed as a call (dialect gate:
 * the JSON parser only runs for ORNITH_JSON). */
static void test_json_ignored_under_xml_dialect(void) {
    jr_value *tools = jr_parse(READ_TOOLS);
    const char *text =
        "{\"function\":\"read\",\"arguments\":{\"path\":\"/etc/hostname\"}}";

    tc_tool_call *calls = NULL; int n = -1;
    ASSERT_INT_EQ(tc_parse_tool_calls(text, tools, TC_DIALECT_QWEN_XML, &calls, &n), 0);
    ASSERT_INT_EQ(n, 0);

    tc_tool_calls_free(calls, n);
    jr_free(tools);
}

/* Dialect resolver (tc_dialect_for_model) — name/arch table. */
static void test_dialect_resolver(void) {
    ASSERT_INT_EQ(tc_dialect_for_model("ornith", false), TC_DIALECT_ORNITH_JSON);
    ASSERT_INT_EQ(tc_dialect_for_model("Ornith-1.0-9B", false), TC_DIALECT_ORNITH_JSON);
    ASSERT_INT_EQ(tc_dialect_for_model("saber-orch", false), TC_DIALECT_QWEN_XML);
    ASSERT_INT_EQ(tc_dialect_for_model("nex-n2-mini", false), TC_DIALECT_QWEN_XML);
    ASSERT_INT_EQ(tc_dialect_for_model("minicpm5", true), TC_DIALECT_MINICPM_XML);
    /* Name table wins over the minicpm flag for an ornith model. */
    ASSERT_INT_EQ(tc_dialect_for_model("ornith", true), TC_DIALECT_ORNITH_JSON);
    ASSERT_INT_EQ(tc_dialect_for_model(NULL, false), TC_DIALECT_QWEN_XML);
}

/* ============================================================
 * Abandoned call (TC_PARSE_CALL_ABANDONED).
 *
 * The defect: an opener and a <parameter=KEY> were emitted, the parameter
 * was never closed, and no value was ever written — the model stopped in the
 * middle of the call.  That used to parse to `{}`, which is byte-identical to
 * what a deliberately EMPTY argument produces, so a client could not tell the
 * two apart.  The neighbour tests below are the point of the whole fix: three
 * conditions, three different answers.
 * ============================================================ */

static void test_abandoned_call_fixture(void) {
    jr_value *tools = jr_parse(READ_TOOLS);
    tc_tool_call *calls = NULL; int n = -1;
    tc_parse_error err = {0};

    /* The 57-byte abandoned emission, as the diagnostic reports it. */
    ASSERT_INT_EQ(tc_parse_tool_calls_diagnosed(FX_ABANDONED_READ_PATH, tools,
                                                TC_DIALECT_QWEN_XML,
                                                &calls, &n, &err, NULL), 0);
    ASSERT_INT_EQ(n, 0);                       /* not a call — a truncation */
    ASSERT_INT_EQ(err.kind, TC_PARSE_CALL_ABANDONED);
    ASSERT_STR_EQ(tc_parse_failure_code(err.kind), "call_abandoned");
    ASSERT_STR_EQ(err.function_name, "read");
    ASSERT_STR_EQ(err.parameter_name, "path");   /* names the key left open */
    ASSERT_TRUE(err.message != NULL);
    ASSERT_TRUE(strstr(err.message, "read") != NULL);
    ASSERT_TRUE(strstr(err.message, "abandoned") != NULL);
    ASSERT_TRUE(strstr(err.message, "'path'") != NULL);

    tc_parse_error_free(&err);
    ASSERT_INT_EQ(err.kind, TC_PARSE_OK);        /* reset, nothing to leak */
    tc_tool_calls_free(calls, n);
    jr_free(tools);
}

/* Both entry points agree on the RESULT: the abandoned call is absent either
 * way. tc_parse_tool_calls just does not say why. */
static void test_abandoned_legacy_entry_point_agrees(void) {
    jr_value *tools = jr_parse(READ_TOOLS);
    tc_tool_call *calls = NULL; int n = -1;

    ASSERT_INT_EQ(tc_parse_tool_calls(FX_ABANDONED_READ_PATH, tools,
                                      TC_DIALECT_QWEN_XML, &calls, &n), 0);
    ASSERT_INT_EQ(n, 0);

    tc_tool_calls_free(calls, n);
    jr_free(tools);
}

/* The call ends at END OF TEXT rather than at </function> — same shape, same
 * verdict. The close tag is not what makes it abandoned; the missing
 * </parameter> plus the missing value is. */
static void test_abandoned_at_end_of_text(void) {
    jr_value *tools = jr_parse(READ_TOOLS);
    const char *text = "<function=read>\n<parameter=path>\n";

    tc_tool_call *calls = NULL; int n = -1;
    tc_parse_error err = {0};
    ASSERT_INT_EQ(tc_parse_tool_calls_diagnosed(text, tools, TC_DIALECT_QWEN_XML,
                                                &calls, &n, &err, NULL), 0);
    ASSERT_INT_EQ(n, 0);
    ASSERT_INT_EQ(err.kind, TC_PARSE_CALL_ABANDONED);
    ASSERT_STR_EQ(err.parameter_name, "path");

    tc_parse_error_free(&err);
    tc_tool_calls_free(calls, n);
    jr_free(tools);
}

/* NEIGHBOUR 1 — a genuinely EMPTY argument. The parameter IS closed, so the
 * model finished writing the argument; the call is emitted, and the argument
 * with it, empty. This is the condition that used to be indistinguishable from
 * abandoned — and dropping the value put it right back, because a member the
 * serve deleted and one the model never sent read identically at the client. */
static void test_abandoned_neighbour_empty_value_still_a_call(void) {
    jr_value *tools = jr_parse(READ_TOOLS);
    const char *text =
        "<tool_call>\n<function=read>\n<parameter=path>\n\n</parameter>\n"
        "</function>";

    tc_tool_call *calls = NULL; int n = -1;
    tc_parse_error err = {0};
    ASSERT_INT_EQ(tc_parse_tool_calls_diagnosed(text, tools, TC_DIALECT_QWEN_XML,
                                                &calls, &n, &err, NULL), 0);
    ASSERT_INT_EQ(n, 1);
    ASSERT_STR_EQ(calls[0].name, "read");
    ASSERT_STR_EQ(calls[0].arguments, "{\"path\":\"\"}");
    ASSERT_INT_EQ(err.kind, TC_PARSE_OK);      /* NOT abandoned */

    tc_parse_error_free(&err);
    tc_tool_calls_free(calls, n);
    jr_free(tools);
}

/* NEIGHBOUR 2 — a well-formed call is untouched: still one call, still its
 * arguments, and no error. */
static void test_abandoned_neighbour_wellformed_untouched(void) {
    jr_value *tools = jr_parse(READ_TOOLS);
    const char *text =
        "<tool_call>\n<function=read>\n<parameter=path>\n/etc/hostname\n"
        "</parameter>\n</function>";

    tc_tool_call *calls = NULL; int n = -1;
    tc_parse_error err = {0};
    ASSERT_INT_EQ(tc_parse_tool_calls_diagnosed(text, tools, TC_DIALECT_QWEN_XML,
                                                &calls, &n, &err, NULL), 0);
    ASSERT_INT_EQ(n, 1);
    ASSERT_STR_EQ(calls[0].arguments, "{\"path\":\"/etc/hostname\"}");
    ASSERT_INT_EQ(err.kind, TC_PARSE_OK);

    tc_parse_error_free(&err);
    tc_tool_calls_free(calls, n);
    jr_free(tools);
}

/* NEIGHBOUR 3 — no opener at all is untouched: zero calls AND no error, which
 * is a different answer from the abandoned zero-calls-plus-error. */
static void test_abandoned_neighbour_no_opener_untouched(void) {
    jr_value *tools = jr_parse(READ_TOOLS);
    const char *text = "I could read that file, but let me think first.";

    tc_tool_call *calls = NULL; int n = -1;
    tc_parse_error err = {0};
    ASSERT_INT_EQ(tc_parse_tool_calls_diagnosed(text, tools, TC_DIALECT_QWEN_XML,
                                                &calls, &n, &err, NULL), 0);
    ASSERT_INT_EQ(n, 0);
    ASSERT_INT_EQ(err.kind, TC_PARSE_OK);

    tc_parse_error_free(&err);
    tc_tool_calls_free(calls, n);
    jr_free(tools);
}

/* NEIGHBOUR 4 — the missing close tag ALONE is not abandonment. The live
 * MiniCPM-Q8 "</paramet>" quirk drops the close over a perfectly good value;
 * that call must still come through, unflagged. */
static void test_abandoned_unclosed_but_valued_is_a_call(void) {
    jr_value *tools = jr_parse(READ_TOOLS);
    const char *text =
        "<function=read>\n<parameter=path>\n/etc/hostname\n</function>";

    tc_tool_call *calls = NULL; int n = -1;
    tc_parse_error err = {0};
    ASSERT_INT_EQ(tc_parse_tool_calls_diagnosed(text, tools, TC_DIALECT_QWEN_XML,
                                                &calls, &n, &err, NULL), 0);
    ASSERT_INT_EQ(n, 1);
    ASSERT_STR_EQ(calls[0].arguments, "{\"path\":\"/etc/hostname\"}");
    ASSERT_INT_EQ(err.kind, TC_PARSE_OK);

    tc_parse_error_free(&err);
    tc_tool_calls_free(calls, n);
    jr_free(tools);
}

/* A healthy call BEFORE an abandoned one: the good call survives and the
 * abandoned one is reported, so one truncation does not cost the whole turn. */
static void test_abandoned_after_a_good_call(void) {
    jr_value *tools = jr_parse(READ_TOOLS);
    const char *text =
        "<function=read>\n<parameter=path>\n/etc/hostname\n</parameter>\n</function>\n"
        "<function=read>\n<parameter=path>\n</function>";

    tc_tool_call *calls = NULL; int n = -1;
    tc_parse_error err = {0};
    ASSERT_INT_EQ(tc_parse_tool_calls_diagnosed(text, tools, TC_DIALECT_QWEN_XML,
                                                &calls, &n, &err, NULL), 0);
    ASSERT_INT_EQ(n, 1);
    ASSERT_STR_EQ(calls[0].arguments, "{\"path\":\"/etc/hostname\"}");
    ASSERT_INT_EQ(err.kind, TC_PARSE_CALL_ABANDONED);
    ASSERT_STR_EQ(err.parameter_name, "path");

    tc_parse_error_free(&err);
    tc_tool_calls_free(calls, n);
    jr_free(tools);
}

/* ============================================================
 * Main
 * ============================================================ */

int main(void) {
    printf("\n  tool-call parser tests\n\n");

    RUN_TEST(wrapped_single);
    RUN_TEST(bare_single);
    RUN_TEST(bare_matches_wrapped);
    RUN_TEST(bare_typed_integer);
    RUN_TEST(bare_multi);
    RUN_TEST(wrapped_multi);
    RUN_TEST(reasoning_then_bare_call);
    RUN_TEST(bare_doubled_close);
    RUN_TEST(bare_call_stray_closer);
    RUN_TEST(no_tools_text);
    RUN_TEST(empty);
    RUN_TEST(null_text);
    RUN_TEST(garbage_unterminated_name);
    RUN_TEST(garbage_empty_name);
    RUN_TEST(hallucinated_name_rejected);
    RUN_TEST(hallucinated_name_rejected_wrapped);
    RUN_TEST(known_name_kept_bogus_dropped);
    RUN_TEST(no_schema_permissive);

    RUN_TEST(minicpm_single);
    RUN_TEST(minicpm_typed_multi);
    RUN_TEST(minicpm_cdata);
    RUN_TEST(minicpm_single_quotes);
    RUN_TEST(minicpm_reasoning_prefix);
    RUN_TEST(minicpm_preamble);

    RUN_TEST(forced_value_wrapper_bare);
    RUN_TEST(forced_value_wrapper_wrapped);
    RUN_TEST(forced_value_wrapper_multiline);
    RUN_TEST(forced_value_wrapper_typed_integer);
    RUN_TEST(forced_value_wrapper_empty_inner);
    RUN_TEST(forced_value_matches_plain);
    RUN_TEST(forced_value_wrapper_malformed_param_close);
    RUN_TEST(forced_value_wrapper_multi_malformed);
    RUN_TEST(forced_value_partial_not_unwrapped);

    RUN_TEST(ornith_fenced_single);
    RUN_TEST(ornith_bare_with_prose);
    RUN_TEST(ornith_name_key);
    RUN_TEST(ornith_typed_args);
    RUN_TEST(ornith_parallel_blocks);
    RUN_TEST(ornith_unknown_name_dropped);
    RUN_TEST(ornith_truncated_zero);
    RUN_TEST(ornith_prose_only);
    RUN_TEST(ornith_render_roundtrip);
    RUN_TEST(json_ignored_under_xml_dialect);
    RUN_TEST(dialect_resolver);

    RUN_TEST(abandoned_call_fixture);
    RUN_TEST(abandoned_legacy_entry_point_agrees);
    RUN_TEST(abandoned_at_end_of_text);
    RUN_TEST(abandoned_neighbour_empty_value_still_a_call);
    RUN_TEST(abandoned_neighbour_wellformed_untouched);
    RUN_TEST(abandoned_neighbour_no_opener_untouched);
    RUN_TEST(abandoned_unclosed_but_valued_is_a_call);
    RUN_TEST(abandoned_after_a_good_call);

    TEST_REPORT();
    return __ft_failed > 0 ? 1 : 0;
}
