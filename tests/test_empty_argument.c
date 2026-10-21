/* SPDX-License-Identifier: AGPL-3.0-or-later */
/* Copyright (C) 2026 David (Corey) Herrera */
/*
 * EMPTY IS NOT ABSENT — the one bit the serve owes the client about a parameter
 * that carried no characters.
 *
 * THE DEFECT.  parse_call_arguments dropped any value that trimmed to empty, so
 * `edit` performing a deletion — `new_string:""`, which is what a deletion IS —
 * left the serve as `{"path":…,"old_string":…}` with the member gone.  At the
 * client a gone member is an ABSENT one (the client's schema code
 * argument_is_absent), absent is the signature of a call cut short, and the client
 * charges a cut-short call against a per-task budget of 6 that ENDS the task
 * (LOOP_GUARD_TRUNCATED_CALL_BUDGET).  So a correct deletion was corrected,
 * re-asked, re-emitted identically, and eventually abandoned the task.  Deleting
 * a line of code is an ordinary operation.
 *
 * THE DISCRIMINATOR, and why it is `closed` and not "was the value empty".  The
 * two events that produce zero value characters are:
 *   • the model wrote `<parameter=k></parameter>` — it CLOSED the parameter, so
 *     the emptiness is the argument; and
 *   • the turn ended inside the value region — no terminator was ever written,
 *     so nothing about the argument is known.
 * That is a fact about the bytes, at parameter granularity, and it holds with no
 * decoding grammar running at all (PANTHEON_TOOL_GRAMMAR is off by default).
 * tg_complete() is the same question one level up, about the whole call, and the
 * two layer: test_tool_call_truncation.c owns that half.
 *
 * WHAT THIS DOES NOT UNDO.  The empty-path wedge was `{"path":""}` reaching a `read`,
 * and that shape is deliberately reachable again — it is what the model emitted.
 * Hiding it as `{}` is what made "the model sent an empty path" and "the model
 * never sent a path" the same bytes.  Whether an empty value is USABLE is a
 * question about the parameter (`path` never, `new_string` always) and is
 * answered at the client, which has the tool's meaning; it cannot be answered
 * about a member that never arrived.
 *
 * Pure C string work — no engine/model/CUDA deps, fully local.
 */

#include "lib/test.h"
#include "../src/serve/tool_calling.h"
#include "../src/serve/json_reader.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

/* The client's real shapes for the tools this contract turns on, plus one non-string
 * parameter — an empty value on an `integer` must still produce valid JSON. */
static const char *EMPTY_ARG_TOOLS =
    "[{\"type\":\"function\",\"function\":{\"name\":\"read\","
    "\"parameters\":{\"type\":\"object\",\"properties\":{"
    "\"path\":{\"type\":\"string\"}},\"required\":[\"path\"]}}},"
    "{\"type\":\"function\",\"function\":{\"name\":\"write\","
    "\"parameters\":{\"type\":\"object\",\"properties\":{"
    "\"path\":{\"type\":\"string\"},\"content\":{\"type\":\"string\"}},"
    "\"required\":[\"path\",\"content\"]}}},"
    "{\"type\":\"function\",\"function\":{\"name\":\"edit\","
    "\"parameters\":{\"type\":\"object\",\"properties\":{"
    "\"path\":{\"type\":\"string\"},\"old_string\":{\"type\":\"string\"},"
    "\"new_string\":{\"type\":\"string\"}},"
    "\"required\":[\"path\",\"old_string\",\"new_string\"]}}},"
    "{\"type\":\"function\",\"function\":{\"name\":\"edit_lines\","
    "\"parameters\":{\"type\":\"object\",\"properties\":{"
    "\"path\":{\"type\":\"string\"},\"start_line\":{\"type\":\"integer\"},"
    "\"new_text\":{\"type\":\"string\"}},"
    "\"required\":[\"path\",\"start_line\"]}}}]";

/* Parse `raw` and hand back the first call's name + arguments (both owned by the
 * caller), along with the diagnosis.  `*n` is the call count. */
static char *parse_one(const char *raw, const jr_value *tools, int *n,
                       char **name_out, tc_parse_error *err)
{
    tc_tool_call *calls = NULL;
    *n = -1;
    *name_out = NULL;
    if (tc_parse_tool_calls_diagnosed(raw, tools, TC_DIALECT_QWEN_XML, &calls, n,
                                      err, NULL) != 0)
        return NULL;
    char *args = NULL;
    if (*n > 0) {
        args = calls[0].arguments ? strdup(calls[0].arguments) : NULL;
        *name_out = calls[0].name ? strdup(calls[0].name) : NULL;
    }
    tc_tool_calls_free(calls, *n);
    return args;
}

/* ============================================================
 * THE DELETION — the call this whole change exists for.
 * ============================================================ */

/* `edit` with an empty `new_string` is a DELETION, and it must arrive as a
 * present-and-empty member.  Everything the client does with it turns on that:
 * present-and-empty is a value it can judge, absent is a truncation it charges. */
static void test_edit_deletion_keeps_new_string_present(void) {
    jr_value *tools = jr_parse(EMPTY_ARG_TOOLS);
    const char *raw =
        "<tool_call>\n<function=edit>\n"
        "<parameter=path>\n/tmp/x.c\n</parameter>\n"
        "<parameter=old_string>\ndead line\n</parameter>\n"
        "<parameter=new_string>\n</parameter>\n"
        "</function>\n</tool_call>";

    int n = 0; char *name = NULL; tc_parse_error err = {0};
    char *args = parse_one(raw, tools, &n, &name, &err);
    ASSERT_INT_EQ(n, 1);
    ASSERT_STR_EQ(name, "edit");
    ASSERT_STR_EQ(args, "{\"path\":\"/tmp/x.c\",\"old_string\":\"dead line\","
                        "\"new_string\":\"\"}");
    ASSERT_INT_EQ(err.kind, TC_PARSE_OK);

    free(args); free(name);
    tc_parse_error_free(&err);
    jr_free(tools);
}

/* The same deletion written with the whitespace padding the model actually
 * emits.  dup_trim reduces "\n\n" to "", so this is byte-different from the test
 * above and value-identical — the padding must not change the verdict. */
static void test_edit_deletion_with_whitespace_padding(void) {
    jr_value *tools = jr_parse(EMPTY_ARG_TOOLS);
    const char *raw =
        "<function=edit>\n"
        "<parameter=path>\n/tmp/x.c\n</parameter>\n"
        "<parameter=old_string>\ndead line\n</parameter>\n"
        "<parameter=new_string>\n \t\n</parameter>\n"
        "</function>";

    int n = 0; char *name = NULL; tc_parse_error err = {0};
    char *args = parse_one(raw, tools, &n, &name, &err);
    ASSERT_INT_EQ(n, 1);
    ASSERT_STR_EQ(args, "{\"path\":\"/tmp/x.c\",\"old_string\":\"dead line\","
                        "\"new_string\":\"\"}");
    ASSERT_INT_EQ(err.kind, TC_PARSE_OK);

    free(args); free(name);
    tc_parse_error_free(&err);
    jr_free(tools);
}

/* `write` with empty `content` — creating an empty file — is the same contract on
 * the two-argument tool, and it is the shape that used to arrive as `write` with
 * `content` MISSING, the exact fixture the client's truncated-call detector was built
 * from. */
static void test_write_empty_content_keeps_the_member(void) {
    jr_value *tools = jr_parse(EMPTY_ARG_TOOLS);
    const char *raw =
        "<function=write>\n"
        "<parameter=path>\n/tmp/empty.txt\n</parameter>\n"
        "<parameter=content>\n</parameter>\n"
        "</function>";

    int n = 0; char *name = NULL; tc_parse_error err = {0};
    char *args = parse_one(raw, tools, &n, &name, &err);
    ASSERT_INT_EQ(n, 1);
    ASSERT_STR_EQ(args, "{\"path\":\"/tmp/empty.txt\",\"content\":\"\"}");
    ASSERT_INT_EQ(err.kind, TC_PARSE_OK);

    free(args); free(name);
    tc_parse_error_free(&err);
    jr_free(tools);
}

/* ============================================================
 * THE OTHER SIDE — an UNCLOSED empty value is still dropped, and still named.
 * ============================================================ */

/* The turn ended inside the value: no `</parameter>`, no content.  Nothing is
 * known about this argument, so nothing is claimed — the member stays out AND the
 * call is reported abandoned.  This is the half of the old behaviour that was
 * always right, and it is what keeps the fix from being "emit empties". */
static void test_unclosed_empty_value_is_still_dropped(void) {
    jr_value *tools = jr_parse(EMPTY_ARG_TOOLS);
    const char *raw =
        "<tool_call>\n<function=read>\n<parameter=path>\n\n</function>";

    int n = 0; char *name = NULL; tc_parse_error err = {0};
    char *args = parse_one(raw, tools, &n, &name, &err);
    ASSERT_INT_EQ(n, 0);                          /* not emitted at all */
    ASSERT_INT_EQ(err.kind, TC_PARSE_CALL_ABANDONED);
    ASSERT_STR_EQ(err.function_name, "read");
    ASSERT_STR_EQ(err.parameter_name, "path");

    free(args); free(name);
    tc_parse_error_free(&err);
    jr_free(tools);
}

/* An unclosed empty value SIDE BY SIDE with a good one: `path` arrived and is
 * kept, `content` never closed and is dropped — the pre-existing truncation
 * shape, which the client names through the schema's required list.  The empty
 * rule must not turn "we never heard about content" into `content:""`. */
static void test_unclosed_empty_beside_a_good_value(void) {
    jr_value *tools = jr_parse(EMPTY_ARG_TOOLS);
    const char *raw =
        "<function=write>\n"
        "<parameter=path>\n/tmp/x.c\n</parameter>\n"
        "<parameter=content>\n";

    int n = 0; char *name = NULL; tc_parse_error err = {0};
    char *args = parse_one(raw, tools, &n, &name, &err);
    ASSERT_INT_EQ(n, 0);                          /* abandoned → not emitted */
    ASSERT_INT_EQ(err.kind, TC_PARSE_CALL_ABANDONED);
    ASSERT_STR_EQ(err.parameter_name, "content");

    free(args); free(name);
    tc_parse_error_free(&err);
    jr_free(tools);
}

/* ============================================================
 * SHAPES — the empty value must survive every wrapper, and stay valid JSON.
 * ============================================================ */

/* The forced `<value></value>` wrapper closes the value too, so it is the same
 * deliberate emission and keeps the member.  It is a SEPARATE close from
 * `</parameter>`: the live MiniCPM quirk truncates `</parameter>` to
 * "</paramet>", and a value the model plainly finished must not become absent
 * because of a malformed outer tag. */
static void test_value_wrapper_empty_inner_keeps_the_member(void) {
    jr_value *tools = jr_parse(EMPTY_ARG_TOOLS);
    const char *raw =
        "<function=read>\n<parameter=path>\n<value>\n</value>\n</paramet>\n"
        "</function>";

    int n = 0; char *name = NULL; tc_parse_error err = {0};
    char *args = parse_one(raw, tools, &n, &name, &err);
    ASSERT_INT_EQ(n, 1);
    ASSERT_STR_EQ(args, "{\"path\":\"\"}");
    ASSERT_INT_EQ(err.kind, TC_PARSE_OK);

    free(args); free(name);
    tc_parse_error_free(&err);
    jr_free(tools);
}

/* A closed empty value on a NON-STRING parameter.  write_typed_value must not
 * splice `"start_line":` followed by nothing — an empty integer falls back to a
 * JSON string, so the arguments object still parses.  Asserted by parsing it. */
static void test_empty_value_on_a_typed_parameter_is_valid_json(void) {
    jr_value *tools = jr_parse(EMPTY_ARG_TOOLS);
    const char *raw =
        "<function=edit_lines>\n"
        "<parameter=path>\n/tmp/x.c\n</parameter>\n"
        "<parameter=start_line>\n</parameter>\n"
        "</function>";

    int n = 0; char *name = NULL; tc_parse_error err = {0};
    char *args = parse_one(raw, tools, &n, &name, &err);
    ASSERT_INT_EQ(n, 1);
    ASSERT_STR_EQ(args, "{\"path\":\"/tmp/x.c\",\"start_line\":\"\"}");
    jr_value *parsed = jr_parse(args);
    ASSERT_NOT_NULL(parsed);                      /* it is JSON, not a fragment */
    ASSERT_STR_EQ(jr_as_string(jr_object_get(parsed, "start_line")), "");
    jr_free(parsed);

    free(args); free(name);
    tc_parse_error_free(&err);
    jr_free(tools);
}

/* ============================================================
 * THE NEIGHBOUR THAT MUST STILL LOOK WRONG — the empty-path wedge.
 * ============================================================ */

/* `{"path":""}` on a `read` is the wedge, and it still reaches the client as
 * `{"path":""}`.  That is the point: the client can see an empty path and refuse
 * it (loop_guard_args_void, file_tools' require_string).  Emitted as `{}` it was
 * indistinguishable from a `read` that never sent a path — the same collision,
 * one layer down. */
static void test_empty_path_still_arrives_as_an_empty_path(void) {
    jr_value *tools = jr_parse(EMPTY_ARG_TOOLS);
    const char *raw =
        "<tool_call>\n<function=read>\n<parameter=path>\n\n</parameter>\n"
        "</function>\n</tool_call>";

    int n = 0; char *name = NULL; tc_parse_error err = {0};
    char *args = parse_one(raw, tools, &n, &name, &err);
    ASSERT_INT_EQ(n, 1);
    ASSERT_STR_EQ(name, "read");
    ASSERT_STR_EQ(args, "{\"path\":\"\"}");
    ASSERT_INT_EQ(err.kind, TC_PARSE_OK);   /* it is a call, not a parse failure */

    free(args); free(name);
    tc_parse_error_free(&err);
    jr_free(tools);
}

/* A call with NO parameters at all is still `{}` — the empty rule is about a
 * parameter that was opened and closed, not about an empty argument bag. */
static void test_no_parameters_is_still_an_empty_object(void) {
    jr_value *tools = jr_parse(EMPTY_ARG_TOOLS);
    const char *raw = "<function=read>\n</function>";

    int n = 0; char *name = NULL; tc_parse_error err = {0};
    char *args = parse_one(raw, tools, &n, &name, &err);
    ASSERT_INT_EQ(n, 1);
    ASSERT_STR_EQ(args, "{}");

    free(args); free(name);
    tc_parse_error_free(&err);
    jr_free(tools);
}

/* A healthy call is byte-identical to what it always was. */
static void test_good_values_are_untouched(void) {
    jr_value *tools = jr_parse(EMPTY_ARG_TOOLS);
    const char *raw =
        "<function=edit>\n"
        "<parameter=path>\n/tmp/x.c\n</parameter>\n"
        "<parameter=old_string>\nold\n</parameter>\n"
        "<parameter=new_string>\nnew\n</parameter>\n"
        "</function>";

    int n = 0; char *name = NULL; tc_parse_error err = {0};
    char *args = parse_one(raw, tools, &n, &name, &err);
    ASSERT_INT_EQ(n, 1);
    ASSERT_STR_EQ(args, "{\"path\":\"/tmp/x.c\",\"old_string\":\"old\","
                        "\"new_string\":\"new\"}");
    ASSERT_INT_EQ(err.kind, TC_PARSE_OK);

    free(args); free(name);
    tc_parse_error_free(&err);
    jr_free(tools);
}

/* ============================================================
 * Main
 * ============================================================ */

int main(void) {
    printf("\n  empty tool arguments — empty is present, absent is not\n\n");

    RUN_TEST(edit_deletion_keeps_new_string_present);
    RUN_TEST(edit_deletion_with_whitespace_padding);
    RUN_TEST(write_empty_content_keeps_the_member);
    RUN_TEST(unclosed_empty_value_is_still_dropped);
    RUN_TEST(unclosed_empty_beside_a_good_value);
    RUN_TEST(value_wrapper_empty_inner_keeps_the_member);
    RUN_TEST(empty_value_on_a_typed_parameter_is_valid_json);
    RUN_TEST(empty_path_still_arrives_as_an_empty_path);
    RUN_TEST(no_parameters_is_still_an_empty_object);
    RUN_TEST(good_values_are_untouched);

    TEST_REPORT();
    return __ft_failed > 0 ? 1 : 0;
}
