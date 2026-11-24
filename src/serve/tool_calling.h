/* SPDX-License-Identifier: AGPL-3.0-or-later */
/* Copyright (C) 2026 David (Corey) Herrera */
/**
 * @file tool_calling.h
 * @brief OpenAI tool-calling glue for the serve layer, matching the Nex-N2-mini
 *        (qwen35moe) chat_template. The model uses the Qwen3-Coder XML dialect:
 *        tools are listed as JSON inside a <tools>…</tools> system block, and a
 *        call is emitted as
 *            <tool_call>
 *            <function=NAME>
 *            <parameter=KEY>
 *            VALUE
 *            </parameter>
 *            </function>
 *            </tool_call>
 *        This module renders that prompt section, renders assistant tool-call
 *        history back into the same XML, and parses model output into structured
 *        calls whose arguments are typed per the tools' JSON schema.
 *
 * @status Renders tools + tool/assistant history; parses single and parallel
 *         calls; scalar/bool/number/object/array typing from the schema; names
 *         an ABANDONED call (tc_parse_tool_calls_diagnosed) instead of letting
 *         it pass as an empty-argument one; names the other STRUCTURALLY WRONG
 *         shapes (leaked dialect tail, reasoning leak, missing/unknown name);
 *         and closes a truncated JSON argument object when what is open can be
 *         closed without guessing (tc_parse_repair).  TC_PARSE_CALL_INCOMPLETE
 *         is the serve-layer companion: a call whose decoding grammar never
 *         closed is refused rather than emitted with an argument missing.  An
 *         empty value the model CLOSED is emitted as "" rather than dropped, so
 *         The client sees empty-and-present instead of absent.
 * @issues None
 * @todo None
 */

#ifndef PN_SERVE_TOOL_CALLING_H
#define PN_SERVE_TOOL_CALLING_H

#include "json_reader.h"

/* Per-model tool-call dialect. A resident model gets the dialect it ACTUALLY
 * emits at runtime, resolved ONCE post-route in chat_apply_tool_dialect
 * (openai_chat.c) from the model name/arch (see tc_dialect_for_model) and then
 * threaded to the parser, the stream emitter, and the grammar setup. The GGUF
 * chat_template is NOT the discriminator: ornith's template claims Qwen-XML but
 * its RL-learned habit emits a markdown-fenced JSON object, so the dialect must
 * be declared per model rather than inferred from the template.
 *   QWEN_XML    — n2 / saber: <tool_call><function=NAME><parameter=KEY>VALUE…
 *                 (the historical default).
 *   MINICPM_XML — <function name="NAME"><param name="KEY">VALUE</param>…
 *                 (folded in from the pre-existing MiniCPM special case).
 *   ORNITH_JSON — a markdown-fenced (```json …```) or bare JSON object
 *                 {"function"|"name":NAME,"arguments":{…}} (Ornith-1.0-9B). */
typedef enum {
    TC_DIALECT_QWEN_XML = 0,
    TC_DIALECT_MINICPM_XML,
    TC_DIALECT_ORNITH_JSON,
} tc_dialect;

/* One tool call parsed out of model output (owned strings). `arguments` is a
 * stringified JSON object, ready to place in the OpenAI response verbatim. */
typedef struct {
    char *id;         /* "call_XXXXXXXX" */
    char *name;       /* function name */
    char *arguments;  /* stringified JSON object, e.g. {"location":"Paris"} */
} tc_tool_call;

/* Why a call the model STARTED never became a usable call.  This is a REASON
 * FOR AN ABSENCE, not a warning riding a call that was still emitted: when it
 * is set, the call it describes is deliberately NOT in the result array.
 *
 * TC_PARSE_CALL_ABANDONED is the shape recorded live on 2026-08-18 (three times
 * byte-identical, tests/abandoned_call_fixture.h): the model opened a tool,
 * opened <parameter=KEY>, wrote no value bytes at all, and then jumped to
 * </function> — skipping </parameter> entirely.  Both halves are required and
 * both are structural, not stylistic:
 *   • the parameter was never closed before the call ended (by </function>, by
 *     end of text, or by anything else), AND
 *   • the value region between the opener and the call's end is empty once
 *     whitespace is discounted.
 * The conjunction is what makes this a fact rather than a guess.  A parameter
 * that IS closed around an empty value is a different event — the model wrote a
 * complete argument that happens to be empty — and that call is emitted WITH the
 * empty value: `{"path":""}`, not `{}`.  A parameter whose close tag is malformed
 * but whose value carries content (the live MiniCPM-Q8 "</paramet>" quirk) is a
 * truncated close over a good value, and that call is emitted unchanged.  Only
 * "no close AND no value" means the model stopped in the middle of writing the
 * call.
 *
 * EMPTY IS NOT ABSENT, AND THE SERVE MUST NOT CONFLATE THEM.  This header said
 * "a complete, if useless, argument" while the code dropped that argument, which
 * made it absent — a different claim, and the wrong one.  `edit`'s
 * `new_string:""` is a legal deletion; arriving as `{"path":…,"old_string":…}` it
 * is indistinguishable from a call cut short before `new_string`, and the client
 * charges it against a truncated-call budget that ends the task.  Whether an
 * empty value is USEFUL is a question about the parameter (`path` never, and
 * `{"path":""}` is the empty-path wedge; `new_string` always), so it is
 * answered where the tool's meaning lives — the client — and it can only be
 * answered about a member that actually arrived.
 *
 * The other four kinds are the same species — a call the model STARTED that is
 * deliberately not in the result array — for the shapes that are structurally
 * wrong rather than merely unparseable.  Each is a decidable fact about the
 * bytes, never a guess about intent:
 *
 *   TC_PARSE_LEAKED_TOOL_TAGS — the model mixed TWO frames: it wrote a JSON body
 *     where the <parameter=KEY> blocks belong and then closed with this dialect's
 *     own "</parameter></function>" tail.  Detected only when the body carries NO
 *     <parameter= opener at all, so the "</parameter>" it closes with has nothing
 *     to close — a legitimate call whose LAST argument is a JSON object ends in
 *     the same "}</parameter></function>" bytes and must not be touched.
 *
 *   TC_PARSE_REASONING_LEAK — the parameter value the model wrote BEGINS with
 *     "</think>": a reasoning close landed inside the tool-call region, so the
 *     value is captured thought rather than an argument.  Reported only when no
 *     real call could be recovered out of that value; when one can be, the
 *     recovered call is returned and nothing is reported.
 *
 *   TC_PARSE_CALL_NAME_MISSING — "<function=>": an opener with no name in it.
 *   TC_PARSE_CALL_NAME_UNKNOWN — a name outside the request's `tools` set (the
 *     tc_tool_name_known gate).  Both were dropped SILENTLY before; the drop is
 *     still right, but the client could not tell it from a turn that simply
 *     chose not to call anything.
 *
 * TC_PARSE_CALL_INCOMPLETE and TC_PARSE_NO_CALL_PARSED are serve-layer verdicts
 * and are NOT raised by this module — the parser sees only text, and both turn on
 * something outside it.
 *
 *   TC_PARSE_CALL_INCOMPLETE — the DECODING GRAMMAR entered a call and the turn
 *     ended before that call closed.  This is the truncation, stated by the one
 *     component that can state it: the parser is a text scanner and a truncated
 *     call is, by construction, text that scans perfectly — it is a legal PREFIX
 *     of a good call.  The grammar knows it is a prefix; nothing else does.  So
 *     the serve asks the grammar (tg_complete / SG_TOOL_GRAMMAR_INCOMPLETE in
 *     serve_generation.h) and refuses to hand back a call the model never
 *     finished, rather than shipping the argument-shaped remains of one.
 *     Measured in real serving: 9/9 truncated calls were "grammar accepted
 *     char-for-char, tg_complete == NO", and both controls were complete.
 *     NOTE what this deliberately does NOT do: nothing is masked to make the
 *     model keep emitting.  A model that has stopped, compelled to continue,
 *     writes a corrupted file instead of failing cleanly.  Detect and report.
 *
 *   TC_PARSE_NO_CALL_PARSED is the catch-all: raw tool text exists, zero calls
 *     came out, and none of the kinds above explained why (the parser cannot know
 *     that the stream emitter diverted a tool region at all). */
typedef enum {
    TC_PARSE_OK = 0,
    TC_PARSE_CALL_ABANDONED,
    TC_PARSE_LEAKED_TOOL_TAGS,
    TC_PARSE_REASONING_LEAK,
    TC_PARSE_CALL_NAME_MISSING,
    TC_PARSE_CALL_NAME_UNKNOWN,
    TC_PARSE_CALL_INCOMPLETE,
    TC_PARSE_NO_CALL_PARSED,
} tc_parse_failure;

/* A structured description of the failure above (owned strings, all nullable).
 * Zero-initialize before use; release with tc_parse_error_free. */
typedef struct {
    tc_parse_failure kind;
    char *function_name;   /* the tool the model opened, e.g. "read" */
    char *parameter_name;  /* the key left open, e.g. "path" */
    char *message;         /* human-readable, e.g. "read: call abandoned — …" */
} tc_parse_error;

/* A JSON argument object that parsed ONLY AFTER REPAIR — the opposite kind of
 * event from tc_parse_error above.  Nothing is absent here: a call the model
 * truncated mid-JSON was recovered by closing what it left open.  It is reported
 * so the serve can COUNT it, because a silent repair would hide the model defect
 * it is papering over.
 *
 * The repair is deliberately narrow (see json_repair_truncated in the .c): it
 * appends ONLY closers — one '"' for a string still open, then the matching
 * '}'/']' for each container still open, innermost first — and then re-parses
 * strictly.  It never invents a key, a value, or a separator, and it DECLINES
 * outright when the truncation point would require one (a trailing ',' ':' '{'
 * or '[', a half-written escape).  Whatever it cannot close this way fails
 * exactly as it does today.
 *
 * `added` is the literal byte string appended, which is also the honest record
 * of the risk taken: a '"' in it means a STRING was closed, so the value carries
 * only as much text as the model got out before it was cut off.
 *
 * `count` counts repairs that produced a call — the number of tool calls that
 * would otherwise have been lost outright. */
typedef struct {
    int   count;          /* repairs that yielded ≥1 call (0 = none happened) */
    char *added;          /* bytes appended by the LAST repair, e.g. "\"}}" */
    char *repaired_json;  /* the LAST repaired object text, for the log */
} tc_parse_repair;

/**
 * Stable wire code for a parse failure, for a client to switch on.
 *
 * @param kind  the failure kind
 * @return  a static string ("call_abandoned", "leaked_tool_tags",
 *          "reasoning_leak", "call_name_missing", "call_name_unknown",
 *          "call_incomplete", "no_call_parsed"), or NULL for TC_PARSE_OK
 *
 * Thread-safe: yes (returns static storage)
 */
const char *tc_parse_failure_code(tc_parse_failure kind);

/** Free the owned members of `err` and reset it to TC_PARSE_OK. NULL-safe. */
void tc_parse_error_free(tc_parse_error *err);

/** Free the owned members of `rep` and reset its count to 0. NULL-safe. */
void tc_parse_repair_free(tc_parse_repair *rep);

/**
 * Resolve the tool-call dialect for a resident model from its registry name and
 * a caller-computed MiniCPM flag. The name table wins first ("ornith*" →
 * ORNITH_JSON), then the MiniCPM flag → MINICPM_XML, else the Qwen-XML default.
 * This is the single source of truth the post-route resolver
 * (chat_apply_tool_dialect) calls, so a new same-arch model can be declared by
 * name without touching the parser/grammar. `is_minicpm` folds in the existing
 * model_is_minicpm(arch,add_bos) test (passed by the caller, which owns the
 * pn_model). A NULL name defaults to Qwen-XML.
 *
 * @param name        registry model name (may be NULL)
 * @param is_minicpm  true if the caller's model matches the MiniCPM arch test
 * @return  the resolved tc_dialect
 *
 * Thread-safe: yes (pure function of its arguments)
 */
tc_dialect tc_dialect_for_model(const char *name, bool is_minicpm);

/**
 * Does a tool grammar model this dialect's call format?  QWEN_XML is the
 * tool_grammar.c skeleton, ORNITH_JSON is tool_grammar_json.c.  MINICPM_XML's
 * attribute form (`<function name="…"><param name="…">`) is not modelled, so a
 * "required" turn cannot be guaranteed a parsed call for it.
 *
 * @param dialect  the resolved dialect
 * @return  true when a grammar walks exactly that dialect's call
 *
 * Thread-safe: yes (pure function of its argument)
 */
bool tc_dialect_grammar_models(tc_dialect dialect);

/**
 * Render the system-block *content* for the tools preamble, matching the
 * model's chat_template (the "# Tools … <tools>…</tools> … <IMPORTANT>…"
 * text). If `system_content` is non-NULL and non-empty it is appended after a
 * blank line, exactly as the template merges a leading system message.
 *
 * @param tools           OpenAI `tools` array (JR_ARRAY of tool objects).
 * @param system_content  Optional leading system message text, or NULL.
 * @return  Owned NUL-terminated string (caller frees), or NULL on OOM / if
 *          `tools` is not a non-empty array.
 *
 * Thread-safe: yes (no shared state)
 */
char *tc_render_tools_system(const jr_value *tools, const char *system_content);

/**
 * MiniCPM5 variant of tc_render_tools_system. MiniCPM5 uses a different tool
 * dialect than the Qwen/N2 default: tools are listed in <tools>…</tools> and a
 * call is emitted UNWRAPPED as
 *     <function name="NAME"><param name="KEY">VALUE</param>…</function>
 * (multi-line/special values CDATA-wrapped). Rendering the Qwen preamble to a
 * MiniCPM model makes it improvise a garbled hybrid, so the orchestrator model
 * must be prompted with this native preamble. Same ownership/return contract as
 * tc_render_tools_system.
 */
char *tc_render_tools_system_minicpm(const jr_value *tools,
                                     const char *system_content);

/**
 * Ornith variant of tc_render_tools_system. Ornith's RL-learned tool habit emits
 * a markdown-fenced JSON object {"function":NAME,"arguments":{…}} rather than the
 * Qwen XML its template advertises. This preamble shows that JSON form so the
 * model's AUTO emission is consistent with what the ORNITH_JSON parser accepts.
 * Same ownership/return contract as tc_render_tools_system.
 */
char *tc_render_tools_system_ornith(const jr_value *tools,
                                    const char *system_content);

/**
 * Render an assistant history message's `tool_calls` into the model's XML call
 * syntax, to be appended after the assistant text when re-feeding history so a
 * multi-step tool loop reconstructs the same prompt the model produced.
 *
 * @param tool_calls    OpenAI assistant.tool_calls array (JR_ARRAY), or NULL.
 * @param content_empty true if the assistant message had no text content
 *                      (controls the leading separator, per the template).
 * @param dialect       the resident model's dialect: QWEN_XML/MINICPM_XML render
 *                      the XML form; ORNITH_JSON renders the fenced JSON block the
 *                      model emits, so a multi-turn loop feeds ornith its own
 *                      dialect back.
 * @return  Owned string (possibly empty), or NULL on OOM / no calls.
 *
 * Thread-safe: yes
 */
char *tc_render_assistant_tool_calls(const jr_value *tool_calls,
                                     bool content_empty, tc_dialect dialect);

/**
 * One assistant history turn's content as the template renders it: content
 * `|trim`med, then its tool calls, with the "\n\n" separator only when the
 * trimmed content is non-empty.  The ONE join of content and calls: the request
 * parse (combine_assistant_tool_calls) and the serve's own re-render projection
 * (sg_project_client_rerender) both go through it, so they cannot drift.
 *
 * @param content    the message's text content, or NULL.
 * @param tool_calls OpenAI assistant.tool_calls array (JR_ARRAY).
 * @param dialect    as tc_render_assistant_tool_calls.
 * @return  Owned string, or NULL on OOM / no calls.
 *
 * Thread-safe: yes
 */
char *tc_join_assistant_history(const char *content, const jr_value *tool_calls,
                                tc_dialect dialect);

/**
 * Parse model output into structured calls. Accepts BOTH forms the model emits:
 * the canonical wrapped <tool_call><function=NAME>…</function></tool_call> and a
 * bare <function=NAME>…</function> block with no <tool_call> wrapper (the Q3
 * nex-n2-mini uses the bare form most of the time). Both yield identical calls.
 * Real-world noise is tolerated: leading natural-language reasoning before the
 * first <function=, a trailing </tool_call> with no opener, and a doubled
 * </function>. Text with no <function= at all yields zero calls.
 * Each parameter value is typed using the matching tool's JSON schema
 * (string → JSON string; integer/number/boolean → literal when it parses;
 * object/array → passed through when it looks like JSON; unknown → string).
 * An ABANDONED call (see tc_parse_failure) yields NO call: the model stopped
 * mid-emission, so there is nothing to run. Use tc_parse_tool_calls_diagnosed
 * to learn that that is why the call is missing.
 *
 * ORNITH_JSON dialect: instead of the XML forms, parse a markdown-fenced
 * (```json …```) OR bare top-level JSON object {"function"|"name":NAME,
 * "arguments":{…}} (single or parallel) into the same structured calls.
 * Malformed or truncated JSON yields zero calls (never a bogus call), matching
 * the XML path's tolerance. Truncated-object REPAIR is not available through
 * this entry point — it comes with the tc_parse_repair out-param on
 * tc_parse_tool_calls_diagnosed, so a caller only ever gets a repair it is
 * accounting for.
 *
 * @param text    Raw model output containing a wrapped or bare tool call.
 * @param tools   The request `tools` array, for schema-driven typing (may be
 *                NULL — everything then defaults to string).
 * @param dialect the resident model's dialect (selects the XML vs JSON parser).
 * @param out     Set to an owned array of parsed calls (free with
 *                tc_tool_calls_free); NULL if none found.
 * @param n_out   Set to the number of calls (0 if none).
 * @return  0 on success (including zero calls), -1 on OOM.
 *
 * Thread-safe: yes (uses rand() for ids — not cryptographic)
 */
int tc_parse_tool_calls(const char *text, const jr_value *tools,
                        tc_dialect dialect, tc_tool_call **out, int *n_out);

/**
 * tc_parse_tool_calls plus the reason a started call is missing from `out`, and
 * the record of any argument object that parsed only after repair. The calls
 * returned are the same calls in the same order — passing NULL for both
 * out-params is byte-for-byte the plain entry point.
 *
 * At most one failure is reported (the first): a truncated or mis-framed
 * emission makes whatever follows it in the same text dependent rather than
 * corroborating. The failure kinds are reported for the Qwen-XML forms (wrapped
 * and bare) only — the ones whose <function=NAME> / <parameter=KEY> …
 * </parameter> vocabulary makes "opened but never closed" and "closed but never
 * opened" decidable questions. MiniCPM's attribute form and the ORNITH_JSON
 * object form still drop a bad call silently here; the serve layer's
 * TC_PARSE_NO_CALL_PARSED catch-all covers them.
 *
 * @param err  Zero-initialized tc_parse_error, or NULL to opt out. On return,
 *             `kind` is TC_PARSE_OK when nothing was dropped. Owned strings —
 *             release with tc_parse_error_free.
 * @param repair  Zero-initialized tc_parse_repair, or NULL. It both ENABLES and
 *             records the truncated-object repair: passing NULL declines to
 *             repair at all, so a caller that is not accounting for a repair
 *             never silently gets one. On return, `count` is 0 when no repair
 *             fired. Owned strings — release with tc_parse_repair_free.
 * @return  0 on success (including zero calls), -1 on OOM.
 *
 * Thread-safe: yes (uses rand() for ids — not cryptographic)
 */
int tc_parse_tool_calls_diagnosed(const char *text, const jr_value *tools,
                                  tc_dialect dialect, tc_tool_call **out,
                                  int *n_out, tc_parse_error *err,
                                  tc_parse_repair *repair);

/**
 * The calls ONE balanced ORNITH_JSON top-level object yields — exactly what
 * tc_parse_tool_calls produces for that object when it is one of several in a
 * text (the same per-object code runs in both).  Used by the incremental
 * tool-call streamer (tool_call_stream.h) the moment an object closes, so what
 * it has already put on the wire can be checked against the parse.
 *
 * @param open     the object's opening '{'
 * @param obj_end  one past its matching '}'
 * @param tools    the request `tools` array (name gate), may be NULL
 * @param out      set to an owned array (free with tc_tool_calls_free) or NULL
 * @param n_out    set to the number of calls (0 when the object carries none)
 * @return  0 on success (including zero calls), -1 on OOM.
 *
 * Thread-safe: yes (uses rand() for ids — not cryptographic)
 */
int tc_parse_ornith_object(const char *open, const char *obj_end,
                           const jr_value *tools, tc_tool_call **out, int *n_out);

/**
 * True when `fname` names one of the request's tools (a NULL or empty `tools`
 * array has no schema to check against and accepts every name).  The same gate
 * every parser path applies before it emits a call.
 */
bool tc_tool_name_known(const jr_value *tools, const char *fname);

/** Write a fresh "call_XXXXXXXX" id into buf (rand()-based, not cryptographic). */
void tc_new_call_id(char *buf, size_t len);

/** Free an array returned by tc_parse_tool_calls. NULL-safe. */
void tc_tool_calls_free(tc_tool_call *calls, int n);

#endif /* PN_SERVE_TOOL_CALLING_H */
