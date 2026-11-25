/* SPDX-License-Identifier: AGPL-3.0-or-later */
/* Copyright (C) 2026 David (Corey) Herrera */
/**
 * @file tool_calling.c
 * @brief Implementation of the OpenAI tool-calling glue (see tool_calling.h).
 *        Renders the Qwen3-Coder XML tool section the Nex-N2-mini expects (and
 *        the MiniCPM5-native section via tc_render_tools_system_minicpm),
 *        renders assistant/tool history back into that dialect, and parses the
 *        model's tool-call output into schema-typed structured calls.
 *
 *        Parsing accepts THREE forms: the canonical wrapped Qwen form
 *        (<tool_call><function=NAME>…</function></tool_call>); a bare
 *        <function=NAME>…</function> block with no wrapper (the Q3 nex-n2-mini
 *        emits the bare form most of the time); and MiniCPM5's attribute-style
 *        <function name="NAME"><param name="KEY">VALUE</param>…</function>
 *        (unwrapped, CDATA-tolerant). All yield identical structured calls
 *        (ids, names, JSON arguments).
 *
 *        It also recovers a FRAME CAPTURE: when a turn is force-closed
 *        mid-thought and the compelled XML frame swallows the call the model was
 *        still writing, the captured call replaces the outer frame (see
 *        FRAME_MARKERS and recover_captured_call).
 *
 *        And it names an ABANDONED call: a <parameter=KEY> opened, never closed,
 *        with no value bytes in it, is a truncated emission rather than a call.
 *        It is dropped and reported through tc_parse_tool_calls_diagnosed
 *        instead of parsing to the `{}` a deliberately empty argument also
 *        produces (see tc_parse_failure and tests/abandoned_call_fixture.h).
 *        The same channel names the other structurally-wrong shapes — a JSON
 *        body closed with this dialect's own tail (body_leaks_dialect_tail), a
 *        "</think>" leaking into a parameter value, and an opener whose name is
 *        missing or is not a declared tool.
 *
 *        Finally it CLOSES a truncated JSON argument object when everything left
 *        open can be closed without inventing a key or a value
 *        (json_repair_truncated), and reports each such rescue so the serve can
 *        count how often the model is being cut off mid-call.
 *
 * @status Complete for the serve layer's needs.
 * @issues Frame-capture recovery is a SAFETY NET over a serve-side cause (the
 *         reasoning-budget "</think>" injection), not a fix for it, and it can
 *         only reach values that ARE a complete call.  A captured value that is
 *         free-form scratch text (a shell pipeline in a `read` path, an analysis
 *         dump in a `glob` pattern) carries no call and no marker, and passes
 *         through unchanged.
 * @todo
 *   - [ ] Gate the capture layer on the serve's `reason_forced` flag once it is
 *         plumbed through — that is the zero-false-positive signal, and it would
 *         retire the content markers as the trigger.
 */

#include "tool_calling.h"

#include <ctype.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "json_reader.h"
#include "json_value_grammar.h"
#include "json_writer.h"
#include "request_id.h"

/* ============================================================
 * Dialect resolution (per-model tool-call dialect selection)
 * ============================================================ */

/* Case-insensitive prefix test: does `s` begin with `prefix`? */
static bool has_prefix_ci(const char *s, const char *prefix)
{
    if (!s) return false;
    while (*prefix) {
        if (tolower((unsigned char)*s) != tolower((unsigned char)*prefix))
            return false;
        s++;
        prefix++;
    }
    return true;
}

tc_dialect tc_dialect_for_model(const char *name, bool is_minicpm)
{
    /* Name table wins first — an explicit per-model declaration that survives a
     * new same-arch sibling (ornith and saber are near-identical qwen35 arches,
     * so arch alone cannot separate them). */
    if (has_prefix_ci(name, "ornith"))
        return TC_DIALECT_ORNITH_JSON;
    if (is_minicpm)
        return TC_DIALECT_MINICPM_XML;
    return TC_DIALECT_QWEN_XML;
}

bool tc_dialect_grammar_models(tc_dialect dialect)
{
    return dialect == TC_DIALECT_QWEN_XML || dialect == TC_DIALECT_ORNITH_JSON;
}

/* Literal fragments of the tools system block, copied verbatim from the model's
 * chat_template so the prompt the model sees is byte-identical to the oracle.
 * Source: Qwen chat_template (Qwen team, Alibaba Cloud), Apache-2.0; see NOTICE. */
static const char TOOLS_HEAD[] =
    "# Tools\n\nYou have access to the following functions:\n\n<tools>";
static const char TOOLS_FORMAT[] =
    "\n</tools>\n\n"
    "If you choose to call a function ONLY reply in the following format with NO "
    "suffix:\n\n"
    "<tool_call>\n<function=example_function_name>\n"
    "<parameter=example_parameter_1>\nvalue_1\n</parameter>\n"
    "<parameter=example_parameter_2>\nThis is the value for the second parameter\n"
    "that can span\nmultiple lines\n</parameter>\n</function>\n</tool_call>\n\n"
    "<IMPORTANT>\nReminder:\n"
    "- Function calls MUST follow the specified format: an inner "
    "<function=...></function> block must be nested within "
    "<tool_call></tool_call> XML tags\n"
    "- Required parameters MUST be specified\n"
    "- You may provide optional reasoning for your function call in natural "
    "language BEFORE the function call, but NOT after\n"
    "- If there is no function call available, answer the question like normal "
    "with your current knowledge and do not tell the user about function calls\n"
    "</IMPORTANT>";

static const char TC_OPEN[]   = "<tool_call>";
static const char TC_CLOSE[]  = "</tool_call>";
static const char FN_OPEN[]   = "<function=";
static const char FN_CLOSE[]  = "</function>";
static const char PARAM_OPEN[]  = "<parameter=";
static const char PARAM_CLOSE[] = "</parameter>";
/* Grammar-forced (MiniCPM-armed) dialect wraps each parameter value in an extra
 * <value>…</value> block:
 *   <function=NAME>\n<parameter=KEY>\n<value>\nVALUE\n</value>\n</parameter>\n</function>
 * The generation grammar treats VALUE as free text terminated by "\n</parameter>\n",
 * so it accepts this wrapper verbatim — the parser must peel it back off to the
 * bare VALUE (the model emits it because its own trained format uses <value>). */
static const char VALUE_OPEN[]  = "<value>";
static const char VALUE_CLOSE[] = "</value>";

/* MiniCPM5's attribute-style dialect (SGLang minicpm5 detector / the model's
 * chat_template): a bare, unwrapped
 *   <function name="NAME"><param name="KEY">VALUE</param>...</function>
 * with the name/keys as double- or single-quoted attributes, and multi-line /
 * special-char values optionally wrapped in <![CDATA[ ... ]]>. */
static const char FN_ATTR_OPEN[]  = "<function ";
static const char FN_ATTR_CLOSE[] = "</function>";
static const char PARAM_ATTR_OPEN[]  = "<param ";
static const char PARAM_ATTR_CLOSE[] = "</param>";
static const char CDATA_OPEN[]  = "<![CDATA[";
static const char CDATA_CLOSE[] = "]]>";

/* ── FRAME CAPTURE: a tool-call frame landing inside a parameter VALUE ──
 *
 * CAUSE (confirmed in the serve, not a dialect story).  Every think-style model
 * on the fleet runs a per-model `reasoning_budget`.  When a turn spends that
 * budget inside <think> without concluding, the SERVE injects the "</think>"
 * token itself — batch_sched.c driver_reason_step (batched path) and
 * serve_generation.c sg_run_completion (single-user).  That injected close is
 * fed straight into the grammar (driver_grammar_advance_injected), which cannot
 * tell it from a close the MODEL chose: TG_FREE's think_term matcher fires,
 * reasoning_ended is set, and FORCED mode compels the tool-call opener from the
 * very next char.  The model — still mid-thought — is made to emit
 * "<function=NAME>\n<parameter=KEY>\n", and everything it was still going to say
 * lands in the free-text VALUE.
 *
 * The control that settles it is a recorded native-XML capture, a session that never
 * wedged, on nex-n2-mini:
 *
 *     <parameter=pattern>
 *     </think>\n\n<function=read>\n<parameter=path>\nlumen/serve.log
 *     </parameter>
 *
 * The captured text is a NATIVE Qwen-XML call — the exact dialect the grammar
 * armed — preceded by the model's own "</think>".  No dialect drift occurred, so
 * the capture is not about dialects; it is about being cut off mid-thought.  The
 * leading "</think>" is the signature of the whole family: the model had not yet
 * written the close it was about to write, so it wrote it into the value.
 *
 * These literals are the frame's own vocabulary.  They are a TRIGGER for
 * inspection, never a verdict on their own — see recover_captured_call for the
 * gates, and note the two things this trigger CANNOT do:
 *   • it is not complete.  Three recorded defects carry no marker at all (a
 *     shell pipeline in a `read` path, "TaermIDE"/"A rgus", "grep i"), so
 *     content marks can never cover the whole family;
 *   • it is a symptom, not the cause.  The zero-false-positive signal is the
 *     serve's own `reason_forced` (batch_sched.h:307) / `reason_over_budget`
 *     (serve_generation.c:1744) — true exactly on the turns where the close was
 *     injected.  That flag is currently surfaced NOWHERE.  Gating this whole
 *     layer on it is the right end state; until it is plumbed through, the
 *     trigger below stays deliberately conservative.
 *
 * NOTE the deliberate layering: this detector lives in the PARSER, not in the
 * grammar.  Making a nested opener off-grammar would mask the very tokens that
 * carry the model's unfinished thought, leaving a differently-corrupted value
 * and nothing to recover — and a TG_DEAD walk reports its mask INACTIVE
 * (tg_mask_active), which would release the constraint entirely. */
static const char THINK_CLOSE[]   = "</think>";
static const char CALL_FUNCTION[] = "call_function{";
static const char *const FRAME_MARKERS[] = {
    TC_OPEN,        /* "<tool_call>"    — the wrapper opener, restarted inline */
    FN_OPEN,        /* "<function="     — the bare opener, restarted inline    */
    PARAM_OPEN,     /* "<parameter="    — a body boundary, restarted inline    */
    THINK_CLOSE,    /* "</think>"       — the reasoning close, written inline  */
    CALL_FUNCTION,  /* "call_function{" — the Hermes-lineage call verb         */
};
#define N_FRAME_MARKERS (sizeof(FRAME_MARKERS) / sizeof(*FRAME_MARKERS))

/* Malloc a whitespace-trimmed copy of [start,end). Never returns NULL for a
 * valid range unless OOM (returns NULL then). */
static char *dup_trim(const char *start, const char *end)
{
    while (start < end && isspace((unsigned char)*start)) start++;
    while (end > start && isspace((unsigned char)end[-1])) end--;
    size_t n = (size_t)(end - start);
    char *out = malloc(n + 1);
    if (!out) return NULL;
    memcpy(out, start, n);
    out[n] = '\0';
    return out;
}

/* Peel an enclosing <value>…</value> wrapper off a whitespace-trimmed value.
 * The grammar-forced dialect emits <parameter=KEY>\n<value>\nVALUE\n</value>\n
 * so, after dup_trim, `val` may be exactly "<value>…</value>" — return an owned
 * copy of the inner VALUE (itself whitespace-trimmed) in that case.  A value with
 * no wrapper (or only a partial one) is passed through unchanged.  NULL on OOM.
 * Sets *unwrapped=1 iff a wrapper was actually peeled (so the caller can drop a
 * now-empty <value></value> without changing the plain-value/auto behavior). */
static char *unwrap_value(char *val, int *unwrapped)
{
    if (unwrapped) *unwrapped = 0;
    if (!val) return NULL;
    size_t len = strlen(val);
    size_t olen = strlen(VALUE_OPEN), clen = strlen(VALUE_CLOSE);
    if (len < olen + clen) return val;
    if (memcmp(val, VALUE_OPEN, olen) != 0) return val;
    if (memcmp(val + len - clen, VALUE_CLOSE, clen) != 0) return val;
    char *inner = dup_trim(val + olen, val + len - clen);
    if (!inner) return val;   /* OOM → keep the (wrapped) original rather than lose it */
    free(val);
    if (unwrapped) *unwrapped = 1;
    return inner;
}

/* ============================================================
 * Tools system-block rendering
 * ============================================================ */

char *tc_render_tools_system(const jr_value *tools, const char *system_content)
{
    if (jr_kind(tools) != JR_ARRAY || jr_array_len(tools) == 0) return NULL;

    json_writer jw;
    if (!jw_init(&jw)) return NULL;
    jw_raw(&jw, TOOLS_HEAD);
    int n = jr_array_len(tools);
    for (int i = 0; i < n; i++) {
        jw_raw(&jw, "\n");
        jr_to_json_tojson(jr_array_get(tools, i), &jw);   /* tool | tojson */
    }
    jw_raw(&jw, TOOLS_FORMAT);
    if (system_content && *system_content) {
        char *trimmed = dup_trim(system_content, system_content + strlen(system_content));
        if (trimmed && *trimmed) {
            jw_raw(&jw, "\n\n");
            jw_raw(&jw, trimmed);
        }
        free(trimmed);
    }
    char *result = jw.oom ? NULL : strdup(jw.data);
    jw_free(&jw);
    return result;
}

/* MiniCPM5's native tools preamble (from the model's chat_template): tools are
 * listed as JSON inside <tools>…</tools>, and a call is emitted UNWRAPPED as
 * <function name="NAME"><param name="KEY">VALUE</param>…</function>. Rendering
 * the Qwen preamble instead makes MiniCPM improvise a garbled hybrid (it echoes
 * the tool's JSON schema as nested <parameter>/<property> tags), so the
 * orchestrator model MUST be prompted in its own dialect.
 * Source: openbmb/MiniCPM5 chat_template (OpenBMB), Apache-2.0; see NOTICE. */
static const char MC_TOOLS_HEAD[] =
    "# Tools\n\nYou are provided with function signatures within "
    "<tools></tools> XML tags:\n<tools>";
static const char MC_TOOLS_FORMAT[] =
    "\n</tools>\n\n"
    "Tool usage guidelines:\n"
    "- You may call zero or more functions. If no function calls are needed, "
    "just answer normally and do not include any <function ... </function>.\n"
    "- When calling a function, return an XML object within "
    "<function ... </function> using:\n"
    "<function name=\"function-name\"><param name=\"param-name\">param-value"
    "</param></function>\n"
    "- param-value may be multi-line. If it contains <, & or newline "
    "characters, wrap it in a CDATA block: "
    "<param name=\"param-name\"><![CDATA[...multi-line value...]]></param>";

char *tc_render_tools_system_minicpm(const jr_value *tools,
                                     const char *system_content)
{
    if (jr_kind(tools) != JR_ARRAY || jr_array_len(tools) == 0) return NULL;

    json_writer jw;
    if (!jw_init(&jw)) return NULL;
    jw_raw(&jw, MC_TOOLS_HEAD);
    int n = jr_array_len(tools);
    for (int i = 0; i < n; i++) {
        jw_raw(&jw, "\n");
        jr_to_json(jr_array_get(tools, i), &jw);   /* tool | tojson */
    }
    jw_raw(&jw, MC_TOOLS_FORMAT);
    if (system_content && *system_content) {
        char *trimmed = dup_trim(system_content,
                                 system_content + strlen(system_content));
        if (trimmed && *trimmed) {
            jw_raw(&jw, "\n\n");
            jw_raw(&jw, trimmed);
        }
        free(trimmed);
    }
    char *result = jw.oom ? NULL : strdup(jw.data);
    jw_free(&jw);
    return result;
}

/* Ornith's native tools preamble: it lists tools as JSON in <tools>…</tools>
 * (identical to the Qwen head) but instructs the JSON-object call form Ornith's
 * RL training actually emits — a markdown-fenced {"function":NAME,"arguments":
 * {…}} block — so the model's AUTO emission stays consistent with what the
 * ORNITH_JSON parser accepts (rather than the XML its GGUF template advertises
 * but the model does not use). */
static const char ORN_TOOLS_FORMAT[] =
    "\n</tools>\n\n"
    "To use a tool, reply with a single markdown-fenced JSON object in this "
    "exact form:\n\n"
    "```json\n"
    "{\n"
    "  \"function\": \"example_function_name\",\n"
    "  \"arguments\": {\n"
    "    \"example_parameter_1\": \"value_1\",\n"
    "    \"example_parameter_2\": \"value_2\"\n"
    "  }\n"
    "}\n"
    "```\n\n"
    "<IMPORTANT>\nReminder:\n"
    "- Use the EXACT tool names and parameter names from the <tools> list above "
    "— do not invent tool names, and use the real parameter keys (e.g. the "
    "declared \"path\"), never the value as the key or a made-up key\n"
    "- When the task requires reading, writing, editing a file, or running a "
    "command, you MUST call the matching tool with the JSON block — do NOT just "
    "print code or the file contents in your answer; the code only takes effect "
    "when written through the tool\n"
    "- The call MUST be a JSON object with a \"function\" name and an "
    "\"arguments\" object; ALL required parameters MUST be present with real "
    "values (never an empty arguments object)\n"
    "- You may provide brief reasoning BEFORE the JSON block, but the block "
    "itself must be valid JSON with no trailing text\n"
    "- If no tool is needed, answer normally and do not mention tools\n"
    "</IMPORTANT>";

char *tc_render_tools_system_ornith(const jr_value *tools,
                                    const char *system_content)
{
    if (jr_kind(tools) != JR_ARRAY || jr_array_len(tools) == 0) return NULL;

    json_writer jw;
    if (!jw_init(&jw)) return NULL;
    jw_raw(&jw, TOOLS_HEAD);
    int n = jr_array_len(tools);
    for (int i = 0; i < n; i++) {
        jw_raw(&jw, "\n");
        jr_to_json(jr_array_get(tools, i), &jw);   /* tool | tojson */
    }
    jw_raw(&jw, ORN_TOOLS_FORMAT);
    if (system_content && *system_content) {
        char *trimmed = dup_trim(system_content,
                                 system_content + strlen(system_content));
        if (trimmed && *trimmed) {
            jw_raw(&jw, "\n\n");
            jw_raw(&jw, trimmed);
        }
        free(trimmed);
    }
    char *result = jw.oom ? NULL : strdup(jw.data);
    jw_free(&jw);
    return result;
}

/* ============================================================
 * Assistant tool-call history rendering (JSON args -> XML)
 * ============================================================ */

/* Append a history parameter value the way the Qwen template re-renders it:
 * `args_value | string if args_value is string else args_value | tojson`.  A
 * string goes in bare; anything else as Python json.dumps writes it (", " and
 * ": " separators, null as "null"). */
static void append_history_value(json_writer *jw, const jr_value *v)
{
    if (jr_kind(v) == JR_STRING) jw_raw(jw, jr_as_string(v));
    else                         jr_to_json_tojson(v, jw);
}

/* Render one call's <function=…>…</function> body (name + parameters). */
static void render_one_call(json_writer *jw, const jr_value *fn)
{
    const char *name = jr_as_string(jr_object_get(fn, "name"));
    jw_raw(jw, "<function=");
    jw_raw(jw, name ? name : "");
    jw_raw(jw, ">\n");
    /* arguments arrive as a JSON object (HF style, used as-is) or a stringified
     * JSON object (OpenAI wire format, parsed here and freed below). */
    const jr_value *av = jr_object_get(fn, "arguments");
    jr_value *parsed = NULL;
    const jr_value *args = av;
    if (jr_kind(av) != JR_OBJECT) {
        const char *args_str = jr_as_string(av);
        parsed = args_str ? jr_parse(args_str) : NULL;
        args = parsed;
    }
    if (jr_kind(args) == JR_OBJECT) {
        int n = jr_object_member_count(args);
        for (int i = 0; i < n; i++) {
            const char *key = jr_object_key_at(args, i);
            jw_raw(jw, "<parameter=");
            jw_raw(jw, key ? key : "");
            jw_raw(jw, ">\n");
            append_history_value(jw, jr_object_value_at(args, i));
            jw_raw(jw, "\n</parameter>\n");
        }
    }
    jr_free(parsed);
    jw_raw(jw, "</function>\n");
}

/* Render one call as Ornith's markdown-fenced JSON block: ```json {"function":
 * NAME,"arguments":{…}} ```. A stringified `arguments` (the same shape
 * tc_parse_tool_calls produced) is emitted verbatim inside the block; a JSON
 * object `arguments` is written as compact JSON, the same shape. */
static void render_one_call_ornith(json_writer *jw, const jr_value *fn)
{
    const char *name = jr_as_string(jr_object_get(fn, "name"));
    const jr_value *av = jr_object_get(fn, "arguments");
    jw_raw(jw, "```json\n{\"function\": ");
    jw_string(jw, name ? name : "");
    jw_raw(jw, ", \"arguments\": ");
    if (jr_kind(av) == JR_OBJECT) {
        jr_to_json(av, jw);
    } else {
        const char *args_str = jr_as_string(av);
        jw_raw(jw, (args_str && *args_str) ? args_str : "{}");
    }
    jw_raw(jw, "}\n```");
}

char *tc_render_assistant_tool_calls(const jr_value *tool_calls,
                                     bool content_empty, tc_dialect dialect)
{
    if (jr_kind(tool_calls) != JR_ARRAY || jr_array_len(tool_calls) == 0) return NULL;

    json_writer jw;
    if (!jw_init(&jw)) return NULL;
    int n = jr_array_len(tool_calls);
    for (int i = 0; i < n; i++) {
        const jr_value *call = jr_array_get(tool_calls, i);
        const jr_value *fn = jr_object_get(call, "function");
        if (jr_kind(fn) != JR_OBJECT) fn = call;   /* tolerate flattened shape */
        if (i == 0) jw_raw(&jw, content_empty ? "" : "\n\n");
        else        jw_raw(&jw, "\n");
        if (dialect == TC_DIALECT_ORNITH_JSON) {
            render_one_call_ornith(&jw, fn);
        } else {
            jw_raw(&jw, "<tool_call>\n");
            render_one_call(&jw, fn);
            jw_raw(&jw, "</tool_call>");
        }
    }
    char *result = jw.oom ? NULL : strdup(jw.data);
    jw_free(&jw);
    return result;
}

char *tc_join_assistant_history(const char *content, const jr_value *tool_calls,
                                tc_dialect dialect)
{
    char *trimmed = dup_trim(content ? content : "",
                             content ? content + strlen(content) : "");
    if (!trimmed) return NULL;
    char *calls = tc_render_assistant_tool_calls(tool_calls, trimmed[0] == '\0',
                                                 dialect);
    char *joined = NULL;
    if (calls) {
        size_t content_len = strlen(trimmed), calls_len = strlen(calls);
        joined = malloc(content_len + calls_len + 1);
        if (joined) {
            memcpy(joined, trimmed, content_len);
            memcpy(joined + content_len, calls, calls_len + 1);
        }
    }
    free(calls);
    free(trimmed);
    return joined;
}

/* ============================================================
 * Model-output parsing (XML -> structured calls)
 * ============================================================ */

/* Is `fname` one of the request's declared tool names?  Defense-in-depth name
 * gate: the generation-time grammar (tool_grammar.c) enumerates the tool names
 * as a closed alternation, but it only engages once a clean tool-call opener
 * appears — a model that DEGENERATES in the reasoning/prose region (e.g. the Q3
 * saber-orch emitting a garbage "<function=base_agent_agent_...>" name it never
 * routed through the armed skeleton), or a step that fails open on 0 survivors,
 * can still leave a bogus name in the raw text.  This gate ensures the PARSER
 * never surfaces a tool_call whose name is outside the request's tools set, so a
 * hallucinated name can never reach the client.  A NULL/empty tools array (no
 * schema to validate against) is permissive → returns true. */
bool tc_tool_name_known(const jr_value *tools, const char *fname)
{
    if (jr_kind(tools) != JR_ARRAY) return true;   /* no schema → don't gate */
    int n = jr_array_len(tools);
    if (n <= 0) return true;
    for (int i = 0; i < n; i++) {
        const jr_value *fn = jr_object_get(jr_array_get(tools, i), "function");
        const char *name = jr_as_string(jr_object_get(fn, "name"));
        if (name && strcmp(name, fname) == 0) return true;
    }
    return false;
}

/* JSON-schema "type" for parameter `pname` of function `fname`, or NULL. */
static const char *param_type(const jr_value *tools, const char *fname,
                              const char *pname)
{
    if (jr_kind(tools) != JR_ARRAY) return NULL;
    int n = jr_array_len(tools);
    for (int i = 0; i < n; i++) {
        const jr_value *fn = jr_object_get(jr_array_get(tools, i), "function");
        const char *name = jr_as_string(jr_object_get(fn, "name"));
        if (!name || strcmp(name, fname) != 0) continue;
        const jr_value *params = jr_object_get(fn, "parameters");
        const jr_value *props = jr_object_get(params, "properties");
        const jr_value *prop = jr_object_get(props, pname);
        return jr_as_string(jr_object_get(prop, "type"));
    }
    return NULL;
}

/* Write a parsed parameter value into the arguments object, JSON-typed per its
 * schema `type`. Unknown / string types are emitted as JSON strings.
 *
 * THE RAW SPLICE IS THE ONLY UNCHECKED PATH, AND IT IS NO LONGER UNCHECKED.
 * Every branch here either escapes the value into a JSON string — where any
 * byte at all is safe — or VALIDATES it before handing the writer bytes it will
 * not touch: "integer"/"number" through strtod, "boolean" through strcmp, and
 * now "object"/"array" through the same well-formedness walk the decoding
 * grammar runs (json_value_grammar.h).  One definition of "well-formed", shared
 * by the constraint and this re-check, so the two cannot disagree — the failure
 * mode this module has hit twice.
 *
 * WHY DEGRADE TO A STRING RATHER THAN REJECT THE CALL.  Splicing malformed bytes
 * does not corrupt one ARGUMENT, it corrupts the whole `arguments` OBJECT: the
 * client gets a string that will not parse, so every other argument of that call
 * is lost too, and nothing on the wire says which member was at fault.  That is
 * exactly what three live `todowrite` calls did.  Writing the bytes as a JSON
 * string instead keeps the object parseable and moves the failure to where the
 * knowledge is — the tool's own schema check, which can say "todos must be an
 * array" and name it.  Rejecting the call outright was the other candidate
 * (pantheon-14) and is worse here: under tool_choice=required the model must
 * emit something, so a rejection spends a whole turn (~200 s at fleet decode
 * rates) to produce strictly less information than a typed refusal does. */
static void write_typed_value(json_writer *jw, const char *val, const char *type,
                              const char *fname, const char *key)
{
    if (!type || strcmp(type, "string") == 0) { jw_string(jw, val); return; }
    if (strcmp(type, "integer") == 0 || strcmp(type, "number") == 0) {
        if (*val) {
            char *end = NULL;
            strtod(val, &end);
            if (end != val && *end == '\0') { jw_raw(jw, val); return; }
        }
        jw_string(jw, val);
        return;
    }
    if (strcmp(type, "boolean") == 0) {
        if (strcmp(val, "true") == 0 || strcmp(val, "false") == 0) jw_raw(jw, val);
        else jw_string(jw, val);
        return;
    }
    if (strcmp(type, "object") == 0 || strcmp(type, "array") == 0) {
        char want = (type[0] == 'a') ? '[' : '{';
        if (jvg_text_is_json_value(val, want)) { jw_raw(jw, val); return; }
        /* SAY SO.  Reached only with the decoding grammar off (it is off by
         * default, and a request with no tools never builds one) or on a
         * dialect it does not model — with the grammar armed, the bytes that
         * get here cannot be malformed. */
        fprintf(stderr, "[tool_arg_not_json] %s.%s is declared \"%s\" but its "
                "value is not well-formed JSON — emitted as a string so the rest "
                "of the arguments object still parses [len=%zu val=<<<%.200s%s>>>]\n",
                fname ? fname : "?", key ? key : "?", type, strlen(val), val,
                strlen(val) > 200 ? "…[truncated]" : "");
        jw_string(jw, val);
        return;
    }
    jw_string(jw, val);
}

/* Does `v` contain any tool-call FRAME marker at all?  Cheap trigger for the
 * capture path: true only means "worth inspecting", never "reject". */
static bool value_has_frame_marker(const char *v)
{
    for (size_t i = 0; i < N_FRAME_MARKERS; i++)
        if (strstr(v, FRAME_MARKERS[i])) return true;
    return false;
}

/* Does `v` BEGIN (after leading whitespace) with the REASONING CLOSE?
 *
 * This is the one marker that rejects, and it is deliberately the only one.
 * "</think>" is a reasoning-channel control token: it is never the opening bytes
 * of a real argument, but it IS what a model writes first when its think block
 * was force-closed out from under it (a recorded native-XML capture, and the
 * Hermes lead-in capture).  So a value that starts with it is a captured frame, full stop.
 *
 * The other markers are NOT rejected on, even leading.  "<tool_call>",
 * "<function=" and "<parameter=" are plausible opening bytes of a file being
 * written or an edit fragment — Pantheon's own tool_calling.c holds all of them
 * as string literals — and that collision is UNTESTED against real traffic:
 * there is not a single `write` or `edit` call in the whole history DB.  When a
 * leading "<tool_call>" really is a capture it is almost always followed by a
 * complete call, which recover_captured_call handles without needing to reject.
 * Rejecting on the untested markers would trade a recorded failure for an
 * unmeasured one. */
static bool value_starts_with_reasoning_close(const char *v)
{
    while (*v && isspace((unsigned char)*v)) v++;
    return strncmp(v, THINK_CLOSE, strlen(THINK_CLOSE)) == 0;
}

/* Recover a captured call out of the parameter value `value`.  Defined below (it
 * re-enters push_call for a captured XML call); declared here so push_call can
 * consult it.  Returns the number of calls pushed (0 = nothing recoverable), or
 * -1 on OOM (sets *oom). */
static int recover_captured_call(const char *value, const jr_value *tools,
                                 tc_tool_call **calls, int *n, int *cap,
                                 int *oom);

/* Build the JSON arguments object for one <function=…>…</function> body spanning
 * [body,limit). Returns an owned string, or NULL on OOM.
 *
 * `misroute_out` (nullable) receives an owned copy of the FIRST parameter value
 * carrying a tool-call frame marker — the caller decides whether that value is a
 * recoverable misroute, a rejectable frame fragment, or just content.
 *
 * `abandoned_key_out` (nullable) receives an owned copy of the FIRST parameter
 * key that was opened, never closed with </parameter> before `limit`, and holds
 * no value bytes — the ABANDONED shape (see tc_parse_failure). */
static char *parse_call_arguments(const char *body, const char *limit,
                                  const jr_value *tools, const char *fname,
                                  char **misroute_out, char **abandoned_key_out)
{
    json_writer jw;
    if (!jw_init(&jw)) return NULL;
    jw_raw(&jw, "{");
    const char *pp = body;
    int first = 1;
    while (1) {
        const char *pk = strstr(pp, PARAM_OPEN);
        if (!pk || pk >= limit) break;
        const char *kstart = pk + strlen(PARAM_OPEN);
        const char *kend = memchr(kstart, '>', (size_t)(limit - kstart));
        if (!kend) break;
        char *key = dup_trim(kstart, kend);
        const char *vstart = kend + 1;
        /* Was this parameter CLOSED before the call ended?  The answer is the
         * structural half of the abandoned test below AND the whole of the
         * empty-value decision, so it is recorded rather than folded into the
         * `vend` fallback: falling back to `limit` reads the same either way,
         * but "there was no </parameter>" is a fact the value alone cannot
         * report.  A `<value>…</value>` wrapper closes the value too — the
         * branch below promotes that terminator into this flag. */
        const char *pclose = strstr(vstart, PARAM_CLOSE);
        int closed = (pclose && pclose <= limit);
        const char *vend = closed ? pclose : limit;

        /* Grammar-forced <value>…</value> wrapper: if the value region opens with
         * <value> (after leading whitespace), take </value> as the value boundary
         * directly.  This both peels the wrapper AND survives a malformed
         * </parameter> close (the live MiniCPM-Q8 quirk emits a truncated
         * "</paramet>", so PARAM_CLOSE isn't found and vend fell through to the
         * enclosing </function> — dragging the wrapper + junk into the value).
         * The model reliably closes </value>, so anchor on it when present. */
        int unwrapped = 0;
        char *val = NULL;
        const char *vs = vstart;
        while (vs < vend && isspace((unsigned char)*vs)) vs++;
        size_t volen = strlen(VALUE_OPEN);
        if ((size_t)(vend - vs) >= volen && memcmp(vs, VALUE_OPEN, volen) == 0) {
            const char *inner = vs + volen;
            const char *vclose = strstr(inner, VALUE_CLOSE);
            const char *inner_end = (vclose && vclose <= limit) ? vclose : vend;
            val = dup_trim(inner, inner_end);
            unwrapped = 1;
            /* Advance past </value> so the next PARAM_OPEN scan resumes cleanly
             * even when </parameter> was malformed (vend would otherwise sit at
             * the enclosing </function>).  A found </value> is ALSO a close: it
             * is the terminator the model reliably emits, and the live MiniCPM
             * "</paramet>" quirk means </parameter> can be absent from a value
             * the model plainly finished. */
            if (vclose && vclose <= limit) {
                const char *after = vclose + strlen(VALUE_CLOSE);
                const char *pc = strstr(after, PARAM_CLOSE);
                vend = (pc && pc <= limit) ? pc : after;
                closed = 1;
            }
        } else {
            val = dup_trim(vstart, vend);
            val = unwrap_value(val, &unwrapped);   /* peel a same-line <value>…</value> wrapper */
        }
        /* Flag (don't alter) a value that carries the frame's own vocabulary.
         * The value is still written normally: if the caller cannot recover a
         * real call from it and it does not START with a marker, this stays
         * exactly the call it is today. */
        if (misroute_out && !*misroute_out && val && *val &&
            value_has_frame_marker(val))
            *misroute_out = strdup(val);
        /* ABANDONED: the parameter was opened, never closed, and nothing was
         * written into it.  Both halves are needed.  Without `!closed` this
         * would also fire on a deliberately empty <parameter=k></parameter>,
         * which is a complete emission; without the empty test it would fire on
         * the live "</paramet>" quirk, whose value is perfectly good.  Only the
         * conjunction means the model stopped mid-call. */
        if (abandoned_key_out && !*abandoned_key_out && !closed &&
            key && *key && (!val || !*val))
            *abandoned_key_out = strdup(key);
        /* EMPTY-VALUE POLICY — `closed` is the whole test.
         *
         * An empty value the model CLOSED is a deliberate emission and is emitted
         * as `""`.  This module's own contract has always said so ("a complete, if
         * useless, argument, and that call is still emitted"), but the code said
         * otherwise: dropping it made the member ABSENT, and absent is a different
         * claim.  `edit`'s `new_string:""` is a legal deletion; arriving with
         * `new_string` gone it reads as a call cut short, and the client charges it
         * against a truncated-call budget that ends the task after six — a task
         * abandoned over a call that was right every time.  Deleting a line is an
         * ordinary operation, so `{"path":"…"}` → `{"path":"…","new_string":""}`.
         *
         * An empty value the model did NOT close is still dropped.  That is the
         * truncation shape: the turn ended inside the value region, there is no
         * terminator, and nothing was written.  It is also reported — the abandoned
         * test directly above fires on exactly this conjunction.
         *
         * WHY THIS IS SAFE FOR the empty-path wedge.  That wedge was `{"path":""}` on a
         * `read`, and it is REACHABLE AGAIN by design — the empty value is the
         * model's, and hiding it as `{}` is what made the two indistinguishable in
         * the first place.  What changed is who answers it: an empty value that is
         * meaningless for the parameter it landed on is a TOOL-LEVEL judgement
         * (`path` can never be empty, `new_string` always can), and the client makes it
         * with the schema in hand — loop_guard_args_void still trips on
         * `{"path":""}`, and file_tools' require_string still refuses it.  Neither
         * can make that judgement about a member it never received.
         *
         * WHY A TRUNCATED CALL DOES NOT ARRIVE HERE AS A CLOSED EMPTY VALUE.  Two
         * independent gates, at different granularities.  Call-level: with the tool
         * grammar running, a turn that ended mid-call is SG_TOOL_GRAMMAR_INCOMPLETE
         * and every call parsed out of it is refused before the wire
         * (openai_chat.c, TC_PARSE_CALL_INCOMPLETE).  Parameter-level — and this is
         * the one that holds when no grammar ran at all (PANTHEON_TOOL_GRAMMAR is
         * off by default, and a request with no tools never builds one) — `closed`
         * is a fact about THESE BYTES: a cut inside a value leaves no terminator,
         * so the value stays dropped exactly as before.  The residue the two do not
         * cover is a cut AFTER a closed parameter but before `</function>`; there
         * the empty value is still one the model deliberately closed, and the
         * arguments that never got emitted are genuinely absent — the pre-existing
         * truncation shape, which the client names via tool_schema_missing_required. */
        if (key && val && (*val || closed)) {
            if (!first) jw_raw(&jw, ",");
            first = 0;
            jw_string(&jw, key);
            jw_raw(&jw, ":");
            write_typed_value(&jw, val, param_type(tools, fname, key),
                              fname, key);
        }
        free(key);
        free(val);
        if (vend >= limit) break;
        pp = vend + strlen(PARAM_CLOSE);
    }
    jw_raw(&jw, "}");
    char *result = jw.oom ? NULL : strdup(jw.data);
    jw_free(&jw);
    return result;
}

const char *tc_parse_failure_code(tc_parse_failure kind)
{
    switch (kind) {
    case TC_PARSE_CALL_ABANDONED:    return "call_abandoned";
    case TC_PARSE_LEAKED_TOOL_TAGS:  return "leaked_tool_tags";
    case TC_PARSE_REASONING_LEAK:    return "reasoning_leak";
    case TC_PARSE_CALL_NAME_MISSING: return "call_name_missing";
    case TC_PARSE_CALL_NAME_UNKNOWN: return "call_name_unknown";
    case TC_PARSE_CALL_INCOMPLETE:   return "call_incomplete";
    case TC_PARSE_NO_CALL_PARSED:    return "no_call_parsed";
    case TC_PARSE_OK:                break;
    }
    return NULL;
}

void tc_parse_error_free(tc_parse_error *err)
{
    if (!err) return;
    free(err->function_name);
    free(err->parameter_name);
    free(err->message);
    err->function_name = NULL;
    err->parameter_name = NULL;
    err->message = NULL;
    err->kind = TC_PARSE_OK;
}

void tc_parse_repair_free(tc_parse_repair *rep)
{
    if (!rep) return;
    free(rep->added);
    free(rep->repaired_json);
    rep->added = NULL;
    rep->repaired_json = NULL;
    rep->count = 0;
}

/* Record a dropped call on `err`; the FIRST one wins (a truncated or mis-framed
 * emission makes whatever follows it dependent, not corroborating).  Best-effort
 * on the strings: an OOM costs the detail, never the `kind`, so a client
 * switching on the code still learns what happened.  A NULL `fname`/`pname` is
 * stored as NULL so the wire writer omits the field entirely — "<function=>" has
 * no name to report, and saying "" would claim it does. */
static void note_failure(tc_parse_error *err, tc_parse_failure kind,
                         const char *fname, const char *pname, const char *msg)
{
    if (!err || err->kind != TC_PARSE_OK) return;
    err->kind = kind;
    if (fname) err->function_name = strdup(fname);
    if (pname) err->parameter_name = strdup(pname);
    if (msg)   err->message = strdup(msg);
}

/* Record an abandoned call: the model opened a parameter and stopped. */
static void note_abandoned_call(tc_parse_error *err, const char *fname,
                                const char *key)
{
    char msg[256];
    snprintf(msg, sizeof(msg),
             "%s: call abandoned — parameter '%s' was opened but never given a value",
             fname ? fname : "?", key ? key : "?");
    note_failure(err, TC_PARSE_CALL_ABANDONED, fname ? fname : "",
                 key ? key : "", msg);
}

/* Record the two-frame mix: a JSON body closed with this dialect's own tail. */
static void note_leaked_tool_tags(tc_parse_error *err, const char *fname)
{
    char msg[256];
    snprintf(msg, sizeof(msg),
             "%s: mixed tool-call frames — a JSON body closed with "
             "</parameter></function>, which no <parameter= ever opened",
             fname ? fname : "?");
    note_failure(err, TC_PARSE_LEAKED_TOOL_TAGS, fname, NULL, msg);
}

/* Record a reasoning close that landed inside the tool-call region.  No
 * parameter name is reported: the leak is a fact about the FRAME (the whole call
 * was thought, not arguments), not about one key being wrong. */
static void note_reasoning_leak(tc_parse_error *err, const char *fname)
{
    char msg[256];
    snprintf(msg, sizeof(msg),
             "%s: reasoning leaked into the call — a parameter value begins with "
             "</think> and carries no recoverable call",
             fname ? fname : "?");
    note_failure(err, TC_PARSE_REASONING_LEAK, fname, NULL, msg);
}

/* Record an opener whose name is absent ("<function=>") or not a declared tool. */
static void note_bad_call_name(tc_parse_error *err, const char *name)
{
    char msg[256];
    if (name && *name) {
        snprintf(msg, sizeof(msg),
                 "%s: no such tool — the name is not in the request's tools", name);
        note_failure(err, TC_PARSE_CALL_NAME_UNKNOWN, name, NULL, msg);
        return;
    }
    note_failure(err, TC_PARSE_CALL_NAME_MISSING, NULL, NULL,
                 "tool call opened with no function name (\"<function=>\")");
}

/* THE TWO-FRAME MIX.  Does the function body [body,fn_close) consist of a JSON
 * body closed with THIS dialect's own "</parameter></function>" tail — i.e. the
 * model wrote OpenAI/Hermes-style JSON arguments where the <parameter=KEY>
 * blocks belong, then finished the call in the XML frame it was prompted in?
 *
 * qwen-code detects the same shape with
 *   LEAKED_TOOL_CALL_TAGS = /[}\]]\s*<\/parameter>\s*<\/function>/iy
 * scanned by hasLeakedToolCallTags (geminiChat.ts:1358-1377), which walks the
 * text tracking JSON string state and only tests the regex at a '}'/']' found
 * OUTSIDE a string.  That string-awareness is reproduced below and it is not
 * decoration: a value like {"content":"cfg} </parameter></function>"} matches
 * the pattern textually, and a strstr would reject a perfectly good call over
 * bytes the model deliberately put inside a quoted argument.
 *
 * ONE ADDITION their version does not need, and cannot be dropped here.  Their
 * stream is pure JSON — a "</parameter>" in it is leakage by construction.  Ours
 * is the XML dialect, where "…}\n</parameter>\n</function>" is ALSO the exact,
 * correct ending of a healthy call whose last argument has an object/array type.
 * What separates them is whether anything ever OPENED that close: a real call
 * writes <parameter=KEY> first.  So the tail only counts as a leak when the body
 * carries no PARAM_OPEN at all — tested with the same unguarded strstr that
 * parse_call_arguments itself uses to find parameters, so the two can never
 * disagree about whether this body has any.
 *
 * `fn_close` is the block's "</function>" (push_call's fend); it is NULL-safe and
 * verified to actually BE that tag, because the caller passes the block limit
 * when there was no close tag at all and a missing tail is not a mix. */
static bool body_leaks_dialect_tail(const char *body, const char *fn_close)
{
    if (!fn_close || strncmp(fn_close, FN_CLOSE, strlen(FN_CLOSE)) != 0)
        return false;
    const char *opener = strstr(body, PARAM_OPEN);
    if (opener && opener < fn_close) return false;   /* a real call's own close */

    size_t pclen = strlen(PARAM_CLOSE);
    bool in_str = false, escaped = false;
    for (const char *p = body; p < fn_close; p++) {
        if (in_str) {
            if (escaped)         escaped = false;
            else if (*p == '\\') escaped = true;
            else if (*p == '"')  in_str = false;
            continue;
        }
        if (*p == '"') { in_str = true; continue; }
        if (*p != '}' && *p != ']') continue;
        const char *q = p + 1;
        while (q < fn_close && isspace((unsigned char)*q)) q++;
        if ((size_t)(fn_close - q) < pclen || memcmp(q, PARAM_CLOSE, pclen) != 0)
            continue;
        q += pclen;
        while (q < fn_close && isspace((unsigned char)*q)) q++;
        if (q == fn_close) return true;   /* …} </parameter> </function> */
    }
    return false;
}

void tc_new_call_id(char *buf, size_t len)
{
    pn_request_id_new("call_", buf, len);
}

/* Parse one <function=…>…</function> block whose opener is at `fp` (pointing at
 * FN_OPEN) into a call, appending it to *calls (growing as needed). `limit`
 * bounds the block (the enclosing </tool_call>, or end-of-text for a bare call);
 * the name/</function> are searched within [fp,limit). On success returns the
 * position just past this block's </function> (or `limit` if none), so the
 * caller can continue scanning. Returns NULL on OOM after freeing *calls (and
 * sets *oom); a malformed block (no '>'-terminated name) returns NULL with *oom
 * left 0 — the caller stops that scan without failing the whole parse.
 *
 * `depth` is 0 for a call found in the model's own output and 1 for one
 * recovered out of a parameter VALUE; the misroute and abandoned checks run at
 * depth 0 only, so recovery cannot recurse indefinitely.
 *
 * `err` (nullable) receives the reason a started call was deliberately NOT
 * pushed — today only the abandoned shape. */
static const char *push_call(const char *fp, const char *limit,
                             const jr_value *tools,
                             tc_tool_call **calls, int *n, int *cap, int *oom,
                             int depth, tc_parse_error *err)
{
    const char *nstart = fp + strlen(FN_OPEN);
    const char *nend = memchr(nstart, '>', (size_t)(limit - nstart));
    if (!nend) return NULL;                    /* no name terminator → give up */
    char *name = dup_trim(nstart, nend);
    const char *fend = strstr(nend, FN_CLOSE);
    if (!fend || fend > limit) fend = limit;
    const char *next = (fend < limit) ? fend + strlen(FN_CLOSE) : limit;

    if (name && *name && tc_tool_name_known(tools, name)) {
        /* TWO-FRAME MIX.  The body is JSON closed with the XML tail, so there is
         * no <parameter= for parse_call_arguments to read and it would build the
         * `{}` that is byte-identical to a deliberately empty argument bag — the
         * dead end this whole diagnostic layer exists to stop.  Checked BEFORE
         * parsing: the arguments it would produce carry no information. */
        if (depth == 0 && body_leaks_dialect_tail(nend, fend)) {
            note_leaked_tool_tags(err, name);
            free(name);
            return next;
        }
        char *misroute = NULL;
        char *abandoned = NULL;
        char *args = parse_call_arguments(nend, fend, tools, name,
                                          depth == 0 ? &misroute : NULL,
                                          depth == 0 ? &abandoned : NULL);
        if (!args) {
            free(name); free(misroute); free(abandoned);
            *oom = 1;
            return NULL;
        }
        if (misroute) {
            /* FRAME CAPTURE.  Recover first — the model asked for a specific,
             * correct action and was simply cut off mid-thought, so rejecting it
             * only forces a re-roll that will hit the same budget again.  Reject
             * only on the one unambiguous marker (a leading reasoning close).
             * Anything else keeps the call exactly as it is today. */
            int rec = recover_captured_call(misroute, tools, calls, n, cap, oom);
            if (*oom) {
                free(misroute); free(abandoned); free(name); free(args);
                return NULL;
            }
            if (rec > 0) {
                free(misroute); free(abandoned); free(name); free(args);
                return next;   /* the outer XML frame was never a real call */
            }
            if (value_starts_with_reasoning_close(misroute)) {
                /* REASONING LEAK (qwen-code converter.ts:1664 detects the same
                 * "</think> inside the tool-call region" event).  The frame is
                 * rejected exactly as before — but SAY SO.  Dropping it silently
                 * left the client with a turn that produced neither a call nor a
                 * reason, which reads identically to a model that chose not to
                 * act.  Nothing is reported on the `rec > 0` path above: there a
                 * real call was recovered, so nothing is absent. */
                note_reasoning_leak(err, name);
                free(misroute); free(abandoned); free(name); free(args);
                return next;
            }
            free(misroute);
        }
        /* ABANDONED CALL.  The model opened an argument and stopped, so this is
         * a truncation, not a call.  Report it and drop it rather than emitting
         * the `{}` it parses to: that `{}` is byte-identical to the one a
         * deliberately empty argument produces, and a client cannot act on
         * either — which is exactly how the live `read {}` became a dead end. */
        if (abandoned) {
            note_abandoned_call(err, name, abandoned);
            free(abandoned); free(name); free(args);
            return next;
        }
        if (*n >= *cap) {
            int ncap = *cap ? *cap * 2 : 4;
            tc_tool_call *grown = realloc(*calls, (size_t)ncap * sizeof(**calls));
            if (!grown) { free(name); free(args); *oom = 1; return NULL; }
            *calls = grown;
            *cap = ncap;
        }
        char id[TC_CALL_ID_MAX];
        tc_new_call_id(id, sizeof(id));
        (*calls)[*n].id = strdup(id);
        (*calls)[*n].name = name;
        (*calls)[*n].arguments = args;
        (*n)++;
    } else {
        /* empty name (a stray "<function=>") OR a hallucinated name outside the
         * request's tools set → drop the block, never surface a bogus call.
         * The drop is unchanged; what is new is NAMING it.  qwen-code raises
         * MALFORMED_TOOL_CALL for exactly this ("a tool call with no name",
         * converter.ts:1699-1711), and the two cases are worth telling apart:
         * a missing name is a broken emission, an unknown one is the model
         * inventing a tool — only the second is something a client can put in
         * front of the model as a correction. */
        if (depth == 0) note_bad_call_name(err, name);
        free(name);
    }
    return next;
}

/* ============================================================
 * MiniCPM5 attribute-style parsing (<function name="…"><param name="…">…)
 * ============================================================ */

/* Extract the value of a `name="…"` / `name='…'` attribute inside the tag
 * [tag_start, tag_end) (tag_end points at the '>' terminating the open tag).
 * Returns an owned trimmed copy, or NULL if the attribute is absent. */
static char *attr_value(const char *tag_start, const char *tag_end)
{
    const char *p = tag_start;
    while (p < tag_end) {
        const char *nm = memmem(p, (size_t)(tag_end - p), "name", 4);
        if (!nm || nm >= tag_end) return NULL;
        const char *q = nm + 4;
        while (q < tag_end && isspace((unsigned char)*q)) q++;
        if (q >= tag_end || *q != '=') { p = nm + 4; continue; }
        q++;
        while (q < tag_end && isspace((unsigned char)*q)) q++;
        if (q >= tag_end || (*q != '"' && *q != '\'')) { p = nm + 4; continue; }
        char quote = *q++;
        const char *vend = memchr(q, quote, (size_t)(tag_end - q));
        if (!vend) return NULL;
        return dup_trim(q, vend);
    }
    return NULL;
}

/* Copy a param body [start,end) into an owned string, stripping an enclosing
 * <![CDATA[ … ]]> wrapper (CDATA content is taken verbatim, not whitespace-
 * trimmed); a plain body is whitespace-trimmed. NULL on OOM. */
static char *param_body_value(const char *start, const char *end)
{
    const char *s = start;
    while (s < end && isspace((unsigned char)*s)) s++;
    if ((size_t)(end - s) >= strlen(CDATA_OPEN) &&
        memcmp(s, CDATA_OPEN, strlen(CDATA_OPEN)) == 0) {
        const char *cs = s + strlen(CDATA_OPEN);
        const char *ce = memmem(cs, (size_t)(end - cs), CDATA_CLOSE,
                                strlen(CDATA_CLOSE));
        const char *cend = ce ? ce : end;
        size_t n = (size_t)(cend - cs);
        char *out = malloc(n + 1);
        if (!out) return NULL;
        memcpy(out, cs, n);
        out[n] = '\0';
        return out;
    }
    return dup_trim(start, end);
}

/* Build the JSON arguments object for one MiniCPM <function> body spanning
 * [body,limit), reading each <param name="KEY">VALUE</param>. Owned string or
 * NULL on OOM. */
static char *parse_call_arguments_attr(const char *body, const char *limit,
                                       const jr_value *tools, const char *fname)
{
    json_writer jw;
    if (!jw_init(&jw)) return NULL;
    jw_raw(&jw, "{");
    const char *pp = body;
    int first = 1;
    while (1) {
        const char *pk = memmem(pp, (size_t)(limit - pp), PARAM_ATTR_OPEN,
                                strlen(PARAM_ATTR_OPEN));
        if (!pk || pk >= limit) break;
        const char *tag_end = memchr(pk, '>', (size_t)(limit - pk));
        if (!tag_end || tag_end >= limit) break;
        char *key = attr_value(pk + strlen(PARAM_ATTR_OPEN), tag_end);
        const char *vstart = tag_end + 1;
        const char *vend = memmem(vstart, (size_t)(limit - vstart),
                                  PARAM_ATTR_CLOSE, strlen(PARAM_ATTR_CLOSE));
        if (!vend || vend > limit) vend = limit;
        char *val = param_body_value(vstart, vend);
        if (key && *key && val) {
            if (!first) jw_raw(&jw, ",");
            first = 0;
            jw_string(&jw, key);
            jw_raw(&jw, ":");
            write_typed_value(&jw, val, param_type(tools, fname, key),
                              fname, key);
        }
        free(key);
        free(val);
        if (vend >= limit) break;
        pp = vend + strlen(PARAM_ATTR_CLOSE);
    }
    jw_raw(&jw, "}");
    char *result = jw.oom ? NULL : strdup(jw.data);
    jw_free(&jw);
    return result;
}

/* Parse one MiniCPM "<function name=…>…</function>" block whose opener is at
 * `fp` (pointing at FN_ATTR_OPEN). `end` bounds the whole text. On success
 * returns the position just past this block's </function>; NULL on OOM (sets
 * *oom, frees *calls) or a malformed opener (no '>' / no name → *oom left 0). */
static const char *push_call_attr(const char *fp, const char *end,
                                  const jr_value *tools,
                                  tc_tool_call **calls, int *n, int *cap, int *oom)
{
    const char *tag_end = memchr(fp, '>', (size_t)(end - fp));
    if (!tag_end) return NULL;                 /* no open-tag terminator */
    char *name = attr_value(fp + strlen(FN_ATTR_OPEN), tag_end);
    const char *body = tag_end + 1;
    const char *fend = memmem(body, (size_t)(end - body), FN_ATTR_CLOSE,
                              strlen(FN_ATTR_CLOSE));
    const char *limit = fend ? fend : end;
    const char *next = fend ? fend + strlen(FN_ATTR_CLOSE) : end;

    if (name && *name && tc_tool_name_known(tools, name)) {
        char *args = parse_call_arguments_attr(body, limit, tools, name);
        if (!args) { free(name); *oom = 1; return NULL; }
        if (*n >= *cap) {
            int ncap = *cap ? *cap * 2 : 4;
            tc_tool_call *grown = realloc(*calls, (size_t)ncap * sizeof(**calls));
            if (!grown) { free(name); free(args); *oom = 1; return NULL; }
            *calls = grown;
            *cap = ncap;
        }
        char id[TC_CALL_ID_MAX];
        tc_new_call_id(id, sizeof(id));
        (*calls)[*n].id = strdup(id);
        (*calls)[*n].name = name;
        (*calls)[*n].arguments = args;
        (*n)++;
    } else {
        free(name);   /* empty/hallucinated name → drop the block */
    }
    return next;
}

/* True when `text` carries a MiniCPM attribute-style "<function name=…>" tag
 * (as opposed to the Qwen "<function=NAME>" form). Used to select the parser. */
static bool has_attr_function(const char *text)
{
    return memmem(text, strlen(text), FN_ATTR_OPEN, strlen(FN_ATTR_OPEN)) != NULL;
}

/* ============================================================
 * Ornith JSON-object parsing ({"function"|"name":NAME,"arguments":{…}})
 * ============================================================ */

/* Given `open` pointing at a top-level '{', return the position just PAST its
 * matching '}', or NULL if the object never closes (truncated). Skips braces
 * that appear inside JSON strings (honoring backslash escapes). */
static const char *json_object_end(const char *open, const char *end)
{
    int depth = 0;
    bool in_str = false;
    for (const char *p = open; p < end; p++) {
        char c = *p;
        if (in_str) {
            if (c == '\\') { p++; continue; }   /* skip the escaped char */
            if (c == '"') in_str = false;
            continue;
        }
        if (c == '"') { in_str = true; continue; }
        if (c == '{') depth++;
        else if (c == '}') {
            depth--;
            if (depth == 0) return p + 1;
        }
    }
    return NULL;   /* unbalanced / truncated */
}

/* The function NAME out of a JSON call object, under any of its spellings.
 * "function" is the form the ornith preamble asks for, "name" the OpenAI/Hermes
 * tool-schema form, and "func" the Hermes `call_function{…}` form the Q3 base
 * bleeds through with (three recorded Hermes-shape captures).  All three name the same
 * thing, and the result still has to be one of the request's declared tools
 * (tc_tool_name_known), so accepting the aliases only turns a silent no-op into the
 * call the model asked for. */
static const char *json_call_name(const jr_value *obj)
{
    const char *name = jr_as_string(jr_object_get(obj, "function"));
    if (!name) name = jr_as_string(jr_object_get(obj, "name"));
    if (!name) name = jr_as_string(jr_object_get(obj, "func"));
    return name;
}

/* The argument bag out of a JSON call object, under any of its spellings. */
static const jr_value *json_call_args(const jr_value *obj)
{
    const jr_value *args = jr_object_get(obj, "arguments");
    if (jr_kind(args) != JR_OBJECT) args = jr_object_get(obj, "parameters");
    if (jr_kind(args) != JR_OBJECT) args = jr_object_get(obj, "args");
    return args;
}

/* Append one call built from a parsed JSON call object `obj` (which carries a
 * function-name string and an arguments object, under any of the spellings
 * json_call_name/json_call_args accept) to *calls. Returns 0 on success
 * (including a silently-dropped bad-shape/unknown-name object), -1 on OOM. The
 * arguments object is re-serialized (compact) so the stored string is
 * self-owned and independent of the DOM. */
static int push_call_json(const jr_value *obj, const jr_value *tools,
                          tc_tool_call **calls, int *n, int *cap)
{
    if (jr_kind(obj) != JR_OBJECT) return 0;
    const char *name = json_call_name(obj);
    if (!name || !*name || !tc_tool_name_known(tools, name)) return 0;

    const jr_value *args = json_call_args(obj);

    json_writer jw;
    if (!jw_init(&jw)) return -1;
    if (jr_kind(args) == JR_OBJECT) jr_to_json(args, &jw);
    else                            jw_raw(&jw, "{}");
    if (jw.oom) { jw_free(&jw); return -1; }
    char *args_str = strdup(jw.data);
    jw_free(&jw);
    if (!args_str) return -1;

    if (*n >= *cap) {
        int ncap = *cap ? *cap * 2 : 4;
        tc_tool_call *grown = realloc(*calls, (size_t)ncap * sizeof(**calls));
        if (!grown) { free(args_str); return -1; }
        *calls = grown;
        *cap = ncap;
    }
    char id[TC_CALL_ID_MAX];
    tc_new_call_id(id, sizeof(id));
    (*calls)[*n].id = strdup(id);
    (*calls)[*n].name = strdup(name);
    (*calls)[*n].arguments = args_str;
    (*n)++;
    return 0;
}

/* Push every call carried by ONE parsed top-level JSON object: either the
 * {"tool_calls":[…]} wrapper form (each element a call) or the object itself
 * when it names a function. A non-object, or an object that names nothing,
 * pushes nothing. Returns 0 on success, -1 on OOM. */
static int push_calls_from_json_object(const jr_value *obj, const jr_value *tools,
                                       tc_tool_call **calls, int *n, int *cap)
{
    if (jr_kind(obj) != JR_OBJECT) return 0;
    const jr_value *arr = jr_object_get(obj, "tool_calls");
    if (jr_kind(arr) == JR_ARRAY) {
        int m = jr_array_len(arr);
        for (int i = 0; i < m; i++)
            if (push_call_json(jr_array_get(arr, i), tools, calls, n, cap) != 0)
                return -1;
        return 0;
    }
    if (json_call_name(obj))
        return push_call_json(obj, tools, calls, n, cap);
    return 0;
}

/* How deep a nesting the repair below models. A call object is
 * {function, arguments:{…}} — two levels — so anything past this is not the
 * shape we are recovering and declining costs nothing. */
#define TC_JSON_REPAIR_MAX_DEPTH 32

/* CLOSE WHAT A TRUNCATED JSON OBJECT LEFT OPEN — the recovery qwen-code performs
 * with JSON.parse(buffer + '"') and then jsonrepair
 * (streamingToolCallParser.ts:294-323 and :409-439).  We cannot take their
 * second step: jsonrepair guesses (it will invent separators, quote bare words
 * and close a dangling key with an empty value), and an invented empty value is
 * precisely the `{"path":""}` defect this parser already fights.  So this closes
 * and NOTHING else.
 *
 * [open,end) is a '{'-rooted object that never closed.  Appended, in order:
 *   • one '"' when a string is still open, and
 *   • the matching '}' / ']' for every container still open, innermost first.
 * Nothing is inserted, replaced or removed inside the text.
 *
 * It DECLINES (returns NULL) whenever closing would require inventing something:
 *   • the truncation sits after a ',' ':' '{' or '[' — the next thing owed is a
 *     KEY or a VALUE, and there is nothing to close, only something to make up.
 *     `{"path":` repaired to `{"path":""}` and `"arguments":{` repaired to
 *     `"arguments":{}` are both a fabricated argument bag wearing a real call's
 *     name, which is worse than no call at all;
 *   • the open string has no content bytes yet (same fabricated-empty-value
 *     case, reached one character later);
 *   • the cut landed mid-escape ('\' with nothing after it, or a '\uXXXX'
 *     missing digits) — closing there changes what the escape means;
 *   • the nesting is deeper than TC_JSON_REPAIR_MAX_DEPTH, or a closer appeared
 *     with nothing open (not a truncation at all).
 *
 * Trailing whitespace and markdown-fence backticks are excluded from the output
 * when no string is open — parse_ornith_json already ignores a LEADING ```json
 * fence, so ignoring the closing one is the same rule, not a new liberty.  They
 * are never trimmed while a string is open, where those bytes are content.
 *
 * WHAT THIS BUYS AND WHAT IT COSTS.  A string closed early keeps only the bytes
 * the model got out, so a path cut at "/etc/host" is repaired to "/etc/host" and
 * not to what the model meant.  That is the same trade qwen-code takes, and it
 * is the reason `added` is reported: a '"' in it marks exactly the calls whose
 * values may be short.  A wrong path fails loudly at the tool and the model can
 * correct it; a lost call is a turn that silently did nothing.
 *
 * @param open       points at the '{' that opens the truncated object
 * @param end        one past the last byte of the text
 * @param added_out  receives the owned string of appended bytes (caller frees)
 * @return  owned repaired JSON text (caller frees), or NULL when declined/OOM
 */
static char *json_repair_truncated(const char *open, const char *end,
                                   char **added_out)
{
    char closers[TC_JSON_REPAIR_MAX_DEPTH];
    int depth = 0;
    bool in_str = false, escaped = false;
    int hex_owed = 0;              /* digits still owed to a "\uXXXX" escape */
    size_t str_len = 0;            /* content bytes in the string now open */
    const char *last_sig = NULL;   /* last byte NOT INSIDE a string that is
                                    * neither whitespace nor a fence backtick.
                                    * Both quotes of a string count: they are the
                                    * string's boundaries, and the CLOSING one is
                                    * routinely the last real byte before the cut
                                    * ({"path":"/x" owes only its braces). Miss it
                                    * and body_end would slice the value away. */

    for (const char *p = open; p < end; p++) {
        char c = *p;
        if (in_str) {
            str_len++;
            if (hex_owed)     { hex_owed--; continue; }
            if (escaped)      { escaped = false; if (c == 'u') hex_owed = 4; continue; }
            if (c == '\\')    { escaped = true; continue; }
            if (c == '"')     { in_str = false; str_len = 0; last_sig = p; }
            continue;
        }
        if (c == '"') { in_str = true; str_len = 0; last_sig = p; continue; }
        if (c == '{' || c == '[') {
            if (depth >= TC_JSON_REPAIR_MAX_DEPTH) return NULL;
            closers[depth++] = (c == '{') ? '}' : ']';
        } else if (c == '}' || c == ']') {
            if (depth == 0) return NULL;   /* closed with nothing open */
            depth--;
        }
        if (!isspace((unsigned char)c) && c != '`') last_sig = p;
    }

    if (depth == 0) return NULL;           /* balanced — not a truncation */
    if (in_str) {
        if (escaped || hex_owed || str_len == 0) return NULL;
    } else {
        if (!last_sig) return NULL;
        char c = *last_sig;
        if (c == ',' || c == ':' || c == '{' || c == '[') return NULL;
    }

    const char *body_end = in_str ? end : last_sig + 1;
    size_t body_len = (size_t)(body_end - open);
    size_t add_len = (in_str ? 1u : 0u) + (size_t)depth;
    char *added = malloc(add_len + 1);
    char *out = malloc(body_len + add_len + 1);
    if (!added || !out) { free(added); free(out); return NULL; }

    size_t k = 0;
    if (in_str) added[k++] = '"';
    for (int i = depth - 1; i >= 0; i--) added[k++] = closers[i];
    added[k] = '\0';
    memcpy(out, open, body_len);
    memcpy(out + body_len, added, add_len);
    out[body_len + add_len] = '\0';
    *added_out = added;
    return out;
}

/* Repair the truncated object at `open` and push whatever call it carries.
 * Returns 1 when a call was recovered (and `rep` updated), 0 when the repair was
 * declined or carried nothing, -1 on OOM. */
static int repair_truncated_json_call(const char *open, const char *end,
                                      const jr_value *tools,
                                      tc_tool_call **calls, int *n, int *cap,
                                      tc_parse_repair *rep)
{
    char *added = NULL;
    char *fixed = json_repair_truncated(open, end, &added);
    if (!fixed) return 0;

    jr_value *obj = jr_parse(fixed);   /* STRICT re-parse: the repair is only
                                        * ever accepted if it produced real JSON */
    int before = *n;
    int rc = push_calls_from_json_object(obj, tools, calls, n, cap);
    jr_free(obj);
    if (rc != 0) { free(added); free(fixed); return -1; }
    if (*n == before) { free(added); free(fixed); return 0; }

    if (rep) {
        rep->count++;
        free(rep->added);
        free(rep->repaired_json);
        rep->added = added;
        rep->repaired_json = fixed;
        return 1;
    }
    free(added);
    free(fixed);
    return 1;
}

/* Push every call carried by ONE balanced top-level object [open,obj_end) —
 * the per-object step of parse_ornith_json, shared with tc_parse_ornith_object
 * so the incremental streamer and the end-of-turn parse cannot disagree about
 * what an object yields.  Returns 0 on success, -1 on OOM. */
static int push_calls_from_object_slice(const char *open, const char *obj_end,
                                        const jr_value *tools,
                                        tc_tool_call **calls, int *n, int *cap)
{
    char *slice = dup_trim(open, obj_end);
    jr_value *obj = slice ? jr_parse(slice) : NULL;
    free(slice);
    int rc = push_calls_from_json_object(obj, tools, calls, n, cap);
    jr_free(obj);
    return rc;
}

int tc_parse_ornith_object(const char *open, const char *obj_end,
                           const jr_value *tools, tc_tool_call **out, int *n_out)
{
    *out = NULL;
    *n_out = 0;
    int cap = 0;
    if (push_calls_from_object_slice(open, obj_end, tools, out, n_out, &cap) != 0) {
        tc_tool_calls_free(*out, *n_out);
        *out = NULL;
        *n_out = 0;
        return -1;
    }
    return 0;
}

/* Parse Ornith's JSON emission: scan `text` for top-level '{' objects, parse
 * each balanced object, and emit a call for any that carries a function name +
 * arguments. Handles a markdown ```json fence (the '{' is found regardless of
 * the fence), a bare object, repeated objects (parallel calls), and an
 * "arguments" whose value is itself an array of {function,arguments} objects
 * (an alternate parallel shape). Malformed JSON → 0 calls; a TRUNCATED trailing
 * object is repaired first when it can be closed without guessing, and yields 0
 * calls otherwise. Returns 0 on success, -1 on OOM — the caller owns *calls in
 * both cases and frees it.
 *
 * `tail_out` (nullable) receives the position just past the LAST object that
 * yielded a call, so the misroute recovery can tell "this value IS a call" from
 * "this value CONTAINS a call as content".
 *
 * `rep` both ENABLES and records the truncated-object repair; NULL declines to
 * repair at all.  The two are one switch on purpose — a repair that nobody
 * accounts for is exactly the silent papering-over this is meant to avoid — and
 * NULL is what the capture-recovery path wants: it is asking whether a parameter
 * value IS a complete call, and repairing a truncated one there would answer a
 * different question. */
static int parse_ornith_json(const char *text, const jr_value *tools,
                             tc_tool_call **calls, int *n, int *cap,
                             const char **tail_out, tc_parse_repair *rep)
{
    const char *end = text + strlen(text);
    const char *p = text;
    while ((p = memchr(p, '{', (size_t)(end - p))) != NULL) {
        const char *obj_end = json_object_end(p, end);
        if (!obj_end) {
            /* Truncated trailing object. A truncation is TERMINAL — everything
             * to `end` belongs to this object — so this is the last thing to
             * try either way, and the scan stops after it exactly as before. */
            if (rep) {
                int rc = repair_truncated_json_call(p, end, tools, calls, n, cap,
                                                    rep);
                if (rc < 0) return -1;
                if (rc > 0 && tail_out) *tail_out = end;
            }
            break;
        }
        int before = *n;
        if (push_calls_from_object_slice(p, obj_end, tools, calls, n, cap) != 0)
            return -1;
        if (tail_out && *n > before) *tail_out = obj_end;
        /* Continue AFTER this object so repeated/parallel blocks are all seen. */
        p = obj_end;
    }
    return 0;
}

/* After a recovered call, what may still legally trail it inside the value:
 * whitespace, an over-closed brace or tag ('}' / '>' — recorded captures emit
 * "}}}" where two braces close the object), a markdown fence, and the frame
 * closers the model started to write.  Anything else means the value merely
 * CONTAINED a call as content, so it is not a misroute and must not be
 * rewritten. */
static bool recovered_tail_is_clean(const char *p, const char *end)
{
    size_t tclen = strlen(TC_CLOSE), fclen = strlen(FN_CLOSE);
    while (p < end) {
        size_t rem = (size_t)(end - p);
        if (rem >= tclen && memcmp(p, TC_CLOSE, tclen) == 0) { p += tclen; continue; }
        if (rem >= fclen && memcmp(p, FN_CLOSE, fclen) == 0) { p += fclen; continue; }
        if (isspace((unsigned char)*p) || *p == '}' || *p == '>' || *p == '`') {
            p++;
            continue;
        }
        return false;
    }
    return true;
}

/* Recover the call the model ACTUALLY asked for out of a captured parameter
 * value.
 *
 * `value` is one parameter's whitespace-trimmed VALUE that carried a frame
 * marker.  If it is itself a complete call — captured XML
 * (<function=NAME>…</function>, the NATIVE shape seen in a recorded native-XML capture)
 * or a JSON call object, fenced or bare, in any of the spellings
 * json_call_name/json_call_args accept — that call is pushed and the outer frame
 * is discarded by the caller.
 *
 * Note what this does NOT claim.  It does not decide that any shape is a given
 * model's "native dialect"; it only reads back a call the model finished
 * writing in a slot that was never meant to hold one.  The native-XML case needs
 * no dialect assumption at all, and the `func`/`args` spelling is accepted by
 * json_call_name/json_call_args as one more alias, not as a model property.
 *
 * Two gates keep this from hijacking legitimate content:
 *   • the caller only reaches here when the value carries a frame marker, so a
 *     value that is merely a JSON object (a real `write` payload, say) is never
 *     inspected at all;
 *   • the recovered call must run to the END of the value (recovered_tail_is_clean).
 *     A value that CONTAINS a call and then continues is content, not a frame.
 * Recovered names still pass tc_tool_name_known, so an unknown tool cannot appear.
 *
 * @return  number of calls pushed (0 = nothing recoverable), -1 on OOM (*oom set)
 */
static int recover_captured_call(const char *value, const jr_value *tools,
                                 tc_tool_call **calls, int *n, int *cap,
                                 int *oom)
{
    const char *end = value + strlen(value);
    const char *tail = NULL;
    int before = *n;

    const char *fp = strstr(value, FN_OPEN);
    if (fp) {
        const char *next = push_call(fp, end, tools, calls, n, cap, oom, 1, NULL);
        if (*oom) return -1;
        tail = next ? next : end;
    } else if (parse_ornith_json(value, tools, calls, n, cap, &tail, NULL) != 0) {
        /* NULL `rep`: no truncated-object repair here.  This asks whether the
         * value IS a complete call, and a repaired truncation is not one. */
        *oom = 1;
        return -1;
    }

    if (*n == before) return 0;                       /* nothing parsed */
    if (tail && recovered_tail_is_clean(tail, end)) return *n - before;

    /* Parsed, but the value continues past it → content, not a frame. Undo. */
    for (int i = before; i < *n; i++) {
        free((*calls)[i].id);
        free((*calls)[i].name);
        free((*calls)[i].arguments);
    }
    *n = before;
    return 0;
}

int tc_parse_tool_calls(const char *text, const jr_value *tools,
                        tc_dialect dialect, tc_tool_call **out, int *n_out)
{
    return tc_parse_tool_calls_diagnosed(text, tools, dialect, out, n_out,
                                         NULL, NULL);
}

int tc_parse_tool_calls_diagnosed(const char *text, const jr_value *tools,
                                  tc_dialect dialect, tc_tool_call **out,
                                  int *n_out, tc_parse_error *err,
                                  tc_parse_repair *repair)
{
    *out = NULL;
    *n_out = 0;
    if (err) *err = (tc_parse_error){0};
    if (repair) *repair = (tc_parse_repair){0};
    if (!text) return 0;

    tc_tool_call *calls = NULL;
    int n = 0, cap = 0, oom = 0;
    const char *end = text + strlen(text);

    if (dialect == TC_DIALECT_ORNITH_JSON) {
        /* Ornith emits a markdown-fenced or bare JSON object rather than XML.
         * Parse that shape into the same structured calls. */
        if (parse_ornith_json(text, tools, &calls, &n, &cap, NULL, repair) != 0) {
            tc_tool_calls_free(calls, n);
            tc_parse_repair_free(repair);
            return -1;
        }
    } else if (has_attr_function(text)) {
        /* MiniCPM5 attribute-style: bare, unwrapped
         * <function name="NAME"><param name="KEY">VALUE</param>…</function>.
         * No <tool_call> wrapper and no "<function=" form, so it is disjoint
         * from the Qwen/N2 paths below — dispatched here when present. */
        const char *p = text;
        while ((p = memmem(p, (size_t)(end - p), FN_ATTR_OPEN,
                           strlen(FN_ATTR_OPEN))) != NULL) {
            const char *next = push_call_attr(p, end, tools, &calls, &n, &cap, &oom);
            if (oom) { tc_tool_calls_free(calls, n); return -1; }
            if (!next) break;         /* malformed opener (no '>') → stop */
            p = (next > p) ? next : p + strlen(FN_ATTR_OPEN);
        }
    } else if (strstr(text, TC_OPEN) != NULL) {
        /* Wrapped form: one <function=…> per <tool_call>…</tool_call>. Unchanged
         * behaviour — a trailing </tool_call> with no opener is simply never a
         * TC_OPEN match, and a wrapper missing its </function> falls back to the
         * </tool_call> (or end) as the block limit. */
        const char *p = text;
        while ((p = strstr(p, TC_OPEN)) != NULL) {
            const char *tc_end = strstr(p, TC_CLOSE);
            const char *limit = tc_end ? tc_end : end;
            const char *fp = strstr(p, FN_OPEN);
            if (!fp || fp >= limit) { p += strlen(TC_OPEN); continue; }
            const char *next = push_call(fp, limit, tools, &calls, &n, &cap, &oom,
                                         0, err);
            if (oom) { tc_tool_calls_free(calls, n); tc_parse_error_free(err); return -1; }
            (void)next;   /* wrapped path advances by the wrapper, not the block */
            if (!tc_end) break;
            p = tc_end + strlen(TC_CLOSE);
        }
    } else {
        /* Bare form: no <tool_call> wrapper at all — scan every bare
         * <function=…></function> block directly. Leading reasoning text before
         * the first opener is skipped; a doubled/again </function> just yields no
         * further FN_OPEN and ends the scan; a stray "<function=" with no '>' or
         * no close is dropped without emitting a garbage call. */
        const char *p = text;
        while ((p = strstr(p, FN_OPEN)) != NULL) {
            const char *next = push_call(p, end, tools, &calls, &n, &cap, &oom,
                                         0, err);
            if (oom) { tc_tool_calls_free(calls, n); tc_parse_error_free(err); return -1; }
            if (!next) break;         /* malformed opener (no name '>') → stop */
            p = (next > p) ? next : p + strlen(FN_OPEN);
        }
    }

    *out = calls;
    *n_out = n;
    return 0;
}

void tc_tool_calls_free(tc_tool_call *calls, int n)
{
    if (!calls) return;
    for (int i = 0; i < n; i++) {
        free(calls[i].id);
        free(calls[i].name);
        free(calls[i].arguments);
    }
    free(calls);
}
