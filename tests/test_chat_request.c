/* SPDX-License-Identifier: AGPL-3.0-or-later */
/* Copyright (C) 2026 David (Corey) Herrera */
/*
 * Unit tests for openai_chat.c's request PARSE (tool_choice handling + the
 * forced_name lifetime) and its response SHAPE (where session_conflict lands in
 * the emitted JSON, on both the streaming and non-streaming paths).
 *
 * ── Request parse: tool_choice + forced_name LIFETIME (n2 tool-calling
 *    Phase 5 — BUG 2 regression).
 *
 * BUG 2 was an intermittent request failure introduced with the new tool_choice
 * parsing.  The forced-function name (from the object form
 * {"type":"function","function":{"name":"X"}}) was kept as a BORROWED pointer
 * into the request's jr_value DOM — memory the caller frees with jr_free(root).
 * Any change to the free ordering, or a path that holds gen.forced_name past the
 * DOM, turns that alias into a use-after-free.  The fix OWNS the name on the
 * request (strdup, freed in chat_request_free), so it never aliases freed DOM.
 *
 * These tests parse a request, FREE the DOM, then still read gen.forced_name the
 * way the generator does — proving the kept string is request-owned and stable.
 * They also cover the string forms (auto/none/required) and the tools gate.
 *
 * ── Response shape: session_conflict.  See the block above those tests.
 *
 * parse_chat_request, send_completion_json and send_usage_chunk are all static,
 * so the .c is #included directly (the engine-side handlers it also defines are
 * satisfied by tiny link stubs below — neither path calls them; the two hs_send_*
 * stubs capture the emitted bytes).  Pure C, libc-only; no CUDA, no model.
 */

#include "lib/test.h"

#include <unistd.h>   /* dup/dup2/read/close — the stderr capture below */

/* Pull in the unit under test (statics reachable). Rename its main out. */
#define main oai_chat_main_unused
#include "../src/serve/openai_chat.c"
#undef main
#include "../src/serve/serve_emitter.h"

/* ── Link stubs — the request-PARSE path never invokes these; they only satisfy
 * the linker for the handler functions compiled into this TU. ── */
void bs_notify_pressure(pn_batch_sched *a, int b) { (void)a; (void)b; }
bs_acquire_result bs_slot_acquire_or_wait(pn_batch_sched *a, int b,
        int *c, int *d, const char *e) {
    (void)a; (void)b; (void)c; (void)d; (void)e; return BS_ACQUIRE_OK;
}
void hs_begin_sse(int a) { (void)a; }
int hs_client_alive(int a) { (void)a; return 1; }
void hs_end_sse(int a) { (void)a; }
void hs_send_error(int a, int b, const char *c, const char *d) {
    (void)a; (void)b; (void)c; (void)d;
}
void hs_send_error_ex(int a, int b, const char *c, const char *d, const char *e) {
    (void)a; (void)b; (void)c; (void)d; (void)e;
}
/* ── Wire capture — the two sinks openai_chat.c serializes a response into.
 * The response-shape tests at the bottom read back the exact bytes a client
 * would receive, so a field that stops arriving fails the build. ── */
static char wire_json[8192];   /* last hs_send_json body  (non-streaming)     */
static char wire_sse[8192];    /* last hs_send_sse event  (streaming chunk)   */
void hs_send_json(int a, int b, const char *c) {
    (void)a; (void)b;
    snprintf(wire_json, sizeof(wire_json), "%s", c ? c : "");
}
/* EVERY streamed byte, in order, as a client receives it (events and SSE
 * comments) — the tool-call streaming tests fold this the way a client does. */
static json_writer wire_all;
void hs_send_sse(int a, const char *b) {
    (void)a;
    snprintf(wire_sse, sizeof(wire_sse), "%s", b ? b : "");
    if (wire_all.data) {
        jw_raw(&wire_all, "data: ");
        jw_raw(&wire_all, b ? b : "");
        jw_raw(&wire_all, "\n\n");
    }
}
void hs_send_sse_comment(int a, const char *b) {
    (void)a;
    if (wire_all.data) {
        jw_raw(&wire_all, ": ");
        jw_raw(&wire_all, b);
        jw_raw(&wire_all, "\n\n");
    }
}
mr_entry *mr_acquire(mr_registry *a, const char *b) { (void)a; (void)b; return NULL; }
int mr_count(mr_registry *a) { (void)a; return 0; }
void mr_release(mr_registry *a, mr_entry *b) { (void)a; (void)b; }
int mr_snapshot(mr_registry *a, mr_info *b, int c) { (void)a; (void)b; (void)c; return 0; }
bool mr_model_is_minicpm(const pn_model *a) { (void)a; return false; }
void pn_memwatch_read(pn_mem_state *a) { (void)a; }
bool pn_memwatch_running(void) { return false; }
/* Set by a test to say whether the stubbed generation ENFORCED response_format,
 * so the streaming path's status resolution can be driven both ways. */
static int stub_response_format_ok = 0;
/* Tokens the stubbed generation emits through a REAL emitter (wired exactly as
 * serve_generation.c wires it, tool tap included), so run_streaming is driven
 * end to end.  NULL = emit nothing (the response-shape tests). */
static const char *const *stub_pieces = NULL;
static int stub_n_pieces = 0;
/* wire_all.len when the LAST piece had been fed — i.e. what the client had
 * received while the model was still generating. */
static size_t stub_wire_len_before_end = 0;
int sg_run_completion(pn_model *a, pn_state *b, const sg_message *c, int d,
        const sg_gen_params *e, sg_delta_fn f, void *g, sg_result *h,
        const sg_prefix_ctx *i) {
    (void)a; (void)b; (void)c; (void)d; (void)i;
    if (h) h->response_format_ok = stub_response_format_ok;
    if (!stub_pieces) return 0;
    emitter em;
    em_init(&em, f, g, e->stop, e->n_stop, e->has_tools, e->dialect);
    em_set_tool_tap(&em, e->tool_tap, e->tool_tap_ud);
    em.in_think = 0;
    for (int k = 0; k < stub_n_pieces; k++) em_feed(&em, stub_pieces[k]);
    stub_wire_len_before_end = wire_all.len;
    em_flush(&em);
    h->content = em.content.data ? em.content.data : strdup("");
    h->tool_raw = em.tool.data;
    h->finish_reason = "stop";
    em.content.data = NULL;
    em.tool.data = NULL;
    em_free(&em);
    return 0;
}
int sg_run_completion_batched(pn_model *a, pn_batch_sched *b, int c,
        const sg_message *d, int e, const sg_gen_params *f, sg_delta_fn g,
        void *h, sg_result *i, const char *j) {
    (void)a; (void)b; (void)c; (void)d; (void)e; (void)f; (void)g; (void)h;
    (void)i; (void)j; return 0;
}
void sg_result_free(sg_result *a) { (void)a; }
bool sg_tool_choice_required_with_thinking(tc_dialect d) { (void)d; return false; }
/* Phase-7 warm-tier externals the session handlers reference (never called on
 * the request-parse path this test exercises). */
double sg_prefill_tok_per_s_measured(void) { return 0.0; }
uint64_t mr_weights_version(const mr_entry *a) { (void)a; return 0; }
unsigned char *pn_base64_decode(const char *a, size_t *b) { (void)a; if (b) *b = 0; return NULL; }
char *pn_base64_encode(const unsigned char *a, size_t b) { (void)a; (void)b; return NULL; }
void *pn_qwen35moe_snap_deserialize(const unsigned char *a, size_t b, uint64_t *c) {
    (void)a; (void)b; if (c) *c = 0; return NULL;
}
void pn_qwen35moe_state_snapshot_free(void *a) { (void)a; }
pn_session_admit pn_prefix_cache_admit(pn_prefix_cache *a, const char *b,
        uint64_t c, uint64_t d, int *e) {
    (void)a; (void)b; (void)c; (void)d; if (e) *e = 0;
    return PN_SESSION_ADMIT_OK;
}
void pn_prefix_cache_touch(pn_prefix_cache *a, const char *b, uint64_t c) {
    (void)a; (void)b; (void)c;
}
int pn_prefix_cache_import_snapshot(pn_prefix_cache *a, const char *b,
        const void *c, const uint32_t *d, int e) {
    (void)a; (void)b; (void)c; (void)d; (void)e; return 0;
}

/* Bodies exercised. */
static const char *BODY_FUNCTION =
    "{\"model\":\"nex\",\"messages\":[{\"role\":\"user\",\"content\":\"read it\"}],"
    "\"tools\":[{\"type\":\"function\",\"function\":{\"name\":\"read\","
    "\"parameters\":{\"type\":\"object\",\"properties\":{\"path\":{\"type\":\"string\"}}}}}],"
    "\"tool_choice\":{\"type\":\"function\",\"function\":{\"name\":\"read\"}}}";

static const char *BODY_REQUIRED =
    "{\"messages\":[{\"role\":\"user\",\"content\":\"x\"}],"
    "\"tools\":[{\"type\":\"function\",\"function\":{\"name\":\"read\"}}],"
    "\"tool_choice\":\"required\"}";

/* ============================================================
 * BUG 2: the forced_name must survive the DOM being freed.  We parse, free the
 * jr_value tree, then read gen.forced_name the way the generator does.  With the
 * borrowed-pointer regression this is a use-after-free; with the owned copy it
 * is stable ("read").  Looping stresses the allocator so a dangling read would
 * reliably read reused/poisoned bytes.
 * ============================================================ */
static void test_forced_name_survives_dom_free(void) {
    for (int i = 0; i < 256; i++) {
        jr_value *root = jr_parse(BODY_FUNCTION);
        ASSERT_NOT_NULL(root);
        chat_request req;
        const char *err = NULL;
        ASSERT_TRUE(parse_chat_request(root, &req, &err) == 0);
        ASSERT_TRUE(req.gen.tool_choice == SG_TOOL_CHOICE_FUNCTION);
        ASSERT_NOT_NULL(req.gen.forced_name);
        ASSERT_TRUE(strcmp(req.gen.forced_name, "read") == 0);

        /* Free the DOM FIRST — exactly what oai_handle_chat does after the
         * completion.  forced_name must NOT point into it. */
        jr_free(root);

        /* Now read the kept name (generator does this via sg_tg_setup). */
        ASSERT_TRUE(strlen(req.gen.forced_name) == 4);
        ASSERT_TRUE(strcmp(req.gen.forced_name, "read") == 0);

        chat_request_free(&req);
    }
}

/* tool_choice string forms resolve correctly and carry no forced_name. */
/* The prefix-cache session key: `session_id`, then OpenAI's `prompt_cache_key`, then
 * opencode's `promptCacheKey`; empty strings don't count. Without one the serve caches
 * nothing, so a client using either alias would re-prefill every turn. */
static void test_session_key_aliases(void) {
    struct { const char *fields; const char *want; } cases[] = {
        { "\"session_id\":\"a\"",                                  "a" },
        { "\"prompt_cache_key\":\"b\"",                            "b" },
        { "\"promptCacheKey\":\"c\"",                              "c" },
        { "\"promptCacheKey\":\"c\",\"session_id\":\"a\"",           "a" },
        { "\"promptCacheKey\":\"c\",\"prompt_cache_key\":\"b\"",     "b" },
        { "\"session_id\":\"\",\"promptCacheKey\":\"c\"",            "c" },
        { "\"user\":\"u\"",                                        NULL },
    };
    for (unsigned k = 0; k < sizeof(cases) / sizeof(cases[0]); k++) {
        char body[512];
        snprintf(body, sizeof(body),
            "{\"messages\":[{\"role\":\"user\",\"content\":\"x\"}],%s}", cases[k].fields);
        jr_value *root = jr_parse(body);
        ASSERT_NOT_NULL(root);
        chat_request req; const char *err = NULL;
        ASSERT_TRUE(parse_chat_request(root, &req, &err) == 0);
        if (cases[k].want) {
            ASSERT_NOT_NULL(req.session_id);
            ASSERT_TRUE(strcmp(req.session_id, cases[k].want) == 0);
        } else {
            ASSERT_TRUE(req.session_id == NULL);
        }
        chat_request_free(&req);
        jr_free(root);
    }
}

static void test_tool_choice_string_forms(void) {
    struct { const char *tc; sg_tool_choice_kind want; bool tools; } cases[] = {
        { "\"required\"", SG_TOOL_CHOICE_REQUIRED, true },
        { "\"auto\"",     SG_TOOL_CHOICE_AUTO,     true },
        { "\"none\"",     SG_TOOL_CHOICE_AUTO,     false }, /* none → tools dropped → AUTO */
        { "\"bogus\"",    SG_TOOL_CHOICE_AUTO,     true },  /* unknown → auto */
    };
    for (unsigned k = 0; k < sizeof(cases) / sizeof(cases[0]); k++) {
        char body[512];
        snprintf(body, sizeof(body),
            "{\"messages\":[{\"role\":\"user\",\"content\":\"x\"}],"
            "\"tools\":[{\"type\":\"function\",\"function\":{\"name\":\"read\"}}],"
            "\"tool_choice\":%s}", cases[k].tc);
        jr_value *root = jr_parse(body);
        ASSERT_NOT_NULL(root);
        chat_request req; const char *err = NULL;
        ASSERT_TRUE(parse_chat_request(root, &req, &err) == 0);
        ASSERT_TRUE(req.gen.tool_choice == cases[k].want);
        ASSERT_TRUE((req.tools != NULL) == cases[k].tools);
        ASSERT_NULL(req.gen.forced_name);
        chat_request_free(&req);
        jr_free(root);
    }
}

/* "required" with tools present is honored; forced_name stays NULL. */
static void test_required_no_forced_name(void) {
    jr_value *root = jr_parse(BODY_REQUIRED);
    ASSERT_NOT_NULL(root);
    chat_request req; const char *err = NULL;
    ASSERT_TRUE(parse_chat_request(root, &req, &err) == 0);
    ASSERT_TRUE(req.gen.tool_choice == SG_TOOL_CHOICE_REQUIRED);
    ASSERT_NULL(req.gen.forced_name);
    chat_request_free(&req);
    jr_free(root);
}

/* No tools present → forcing is meaningless: tool_choice downgrades to AUTO even
 * when the request asked for a specific function. */
static void test_forced_without_tools_downgrades(void) {
    const char *body =
        "{\"messages\":[{\"role\":\"user\",\"content\":\"x\"}],"
        "\"tool_choice\":{\"type\":\"function\",\"function\":{\"name\":\"read\"}}}";
    jr_value *root = jr_parse(body);
    ASSERT_NOT_NULL(root);
    chat_request req; const char *err = NULL;
    ASSERT_TRUE(parse_chat_request(root, &req, &err) == 0);
    ASSERT_NULL(req.tools);
    ASSERT_TRUE(req.gen.tool_choice == SG_TOOL_CHOICE_AUTO);
    ASSERT_NULL(req.gen.forced_name);
    chat_request_free(&req);
    jr_free(root);
}

/* Malformed object form (missing name) does not force and does not crash. */
static void test_malformed_object_form(void) {
    const char *bodies[] = {
        "{\"messages\":[{\"role\":\"user\",\"content\":\"x\"}],"
        "\"tools\":[{\"type\":\"function\",\"function\":{\"name\":\"read\"}}],"
        "\"tool_choice\":{}}",
        "{\"messages\":[{\"role\":\"user\",\"content\":\"x\"}],"
        "\"tools\":[{\"type\":\"function\",\"function\":{\"name\":\"read\"}}],"
        "\"tool_choice\":{\"type\":\"function\"}}",
        "{\"messages\":[{\"role\":\"user\",\"content\":\"x\"}],"
        "\"tools\":[{\"type\":\"function\",\"function\":{\"name\":\"read\"}}],"
        "\"tool_choice\":{\"type\":\"function\",\"function\":{\"name\":\"\"}}}",
    };
    for (unsigned k = 0; k < sizeof(bodies) / sizeof(bodies[0]); k++) {
        jr_value *root = jr_parse(bodies[k]);
        ASSERT_NOT_NULL(root);
        chat_request req; const char *err = NULL;
        ASSERT_TRUE(parse_chat_request(root, &req, &err) == 0);
        ASSERT_TRUE(req.gen.tool_choice == SG_TOOL_CHOICE_AUTO); /* not FUNCTION */
        ASSERT_NULL(req.gen.forced_name);
        chat_request_free(&req);
        jr_free(root);
    }
}

/* ============================================================
 * RESPONSE SHAPE — session_conflict on the wire.
 *
 * The serve detects a second conversation wearing one session id and reports it
 * with "session_conflict":true INSIDE the `usage` object — the object a client
 * already reads for kv_used/kv_cap (an agent client's SSE parser reads the
 * terminal usage chunk).  It shipped emitted OUTSIDE `usage` on the
 * non-streaming path (the usage object is closed by the jw_printf format, so an
 * appended field lands as a sibling of `usage`, not a member), which is why no
 * client ever saw it.  These tests parse the emitted bytes and assert the field
 * is a MEMBER of `usage` on BOTH paths — a substring check would have passed
 * through the whole bug.
 * ============================================================ */

/* A minimal successful result, as the generator fills it. */
static sg_result wire_test_result(int session_conflict) {
    static char content[] = "hi";
    sg_result res = {0};
    res.content           = content;
    res.finish_reason     = "stop";
    res.prompt_tokens     = 33;
    res.completion_tokens = 4;
    res.kv_used_tokens    = 33;
    res.kv_cap_tokens     = 32000;
    res.session_conflict  = session_conflict;
    return res;
}

/* Non-streaming: the flag rides `usage`, not the enclosing completion object. */
static void test_nonstream_usage_carries_session_conflict(void) {
    sg_result res = wire_test_result(1);
    wire_json[0] = '\0';
    chat_choice ch = { .res = res };
    send_completion_json(-1, "cmpl-1", 0, "test-model", &ch, 1, NULL, NULL);

    jr_value *root = jr_parse(wire_json);
    ASSERT_NOT_NULL(root);
    const jr_value *usage = jr_object_get(root, "usage");
    ASSERT_NOT_NULL(usage);
    ASSERT_TRUE(jr_as_bool(jr_object_get(usage, "session_conflict"), false));
    /* Not a top-level sibling: a client reading `usage` must find it. */
    ASSERT_NULL(jr_object_get(root, "session_conflict"));
    /* The fields it rides beside are untouched. */
    ASSERT_INT_EQ((int)jr_as_number(jr_object_get(usage, "kv_used"), -1), 33);
    ASSERT_INT_EQ((int)jr_as_number(jr_object_get(usage, "kv_cap"), -1), 32000);
    jr_free(root);
}

/* No substitution detected → the field is absent everywhere (never a latched or
 * always-present "false" a client could misread as proof of correctness). */
static void test_nonstream_usage_omits_session_conflict_when_clean(void) {
    sg_result res = wire_test_result(0);
    wire_json[0] = '\0';
    chat_choice ch = { .res = res };
    send_completion_json(-1, "cmpl-2", 0, "test-model", &ch, 1, NULL, NULL);

    jr_value *root = jr_parse(wire_json);
    ASSERT_NOT_NULL(root);
    const jr_value *usage = jr_object_get(root, "usage");
    ASSERT_NOT_NULL(usage);
    ASSERT_NULL(jr_object_get(usage, "session_conflict"));
    ASSERT_NULL(jr_object_get(root, "session_conflict"));
    jr_free(root);
}

/* Streaming: the terminal usage chunk (stream_options.include_usage) carries it
 * in the same place — this is the path the client actually uses. */
static void test_stream_usage_chunk_carries_session_conflict(void) {
    stream_ctx sc = { .client_fd = -1, .id = "cmpl-3", .created = 0,
                      .model_id = "test-model", .role_sent = true };
    wire_sse[0] = '\0';
    send_usage_chunk(&sc, 33, 4, 0, 33, 32000, 1);

    jr_value *root = jr_parse(wire_sse);
    ASSERT_NOT_NULL(root);
    const jr_value *usage = jr_object_get(root, "usage");
    ASSERT_NOT_NULL(usage);
    ASSERT_TRUE(jr_as_bool(jr_object_get(usage, "session_conflict"), false));
    ASSERT_INT_EQ((int)jr_as_number(jr_object_get(usage, "kv_used"), -1), 33);
    ASSERT_INT_EQ((int)jr_as_number(jr_object_get(usage, "total_tokens"), -1), 37);
    jr_free(root);
}

static void test_stream_usage_chunk_omits_session_conflict_when_clean(void) {
    stream_ctx sc = { .client_fd = -1, .id = "cmpl-4", .created = 0,
                      .model_id = "test-model", .role_sent = true };
    wire_sse[0] = '\0';
    send_usage_chunk(&sc, 33, 4, 0, 33, 32000, 0);

    jr_value *root = jr_parse(wire_sse);
    ASSERT_NOT_NULL(root);
    const jr_value *usage = jr_object_get(root, "usage");
    ASSERT_NOT_NULL(usage);
    ASSERT_NULL(jr_object_get(usage, "session_conflict"));
    jr_free(root);
}

/* ============================================================
 * RESPONSE SHAPE — prompt_tokens_details.cached_tokens on the wire.
 *
 * The serve knew how many prompt tokens it reused (the prefix cache logs it to
 * stderr) and told the client nothing, so a client's prefill rate was
 * prompt_tokens / TTFT — a number that RISES the more the serve caches (measured
 * 409 tok/s against a true 39.7 at 150 of 166 reused). These tests read the
 * emitted bytes back and assert the OpenAI-standard nesting on BOTH paths: a
 * MEMBER named cached_tokens inside a `prompt_tokens_details` object inside
 * `usage`. They also pin the two properties a client depends on — the field is
 * present on a COLD turn carrying 0 (absent and zero are different answers), and
 * cached_tokens never exceeds prompt_tokens.
 * ============================================================ */

/* Fetch usage.prompt_tokens_details.cached_tokens out of an emitted body, or
 * -1 when either object is missing. */
static int wire_cached_tokens(const jr_value *root) {
    const jr_value *usage = jr_object_get(root, "usage");
    if (!usage) return -1;
    const jr_value *details = jr_object_get(usage, "prompt_tokens_details");
    if (!details) return -1;
    const jr_value *cached = jr_object_get(details, "cached_tokens");
    if (!cached) return -1;
    return (int)jr_as_number(cached, -1);
}

/* Warm turn, non-streaming: the reused count reaches the client, nested where
 * an OpenAI client already looks for it. */
static void test_nonstream_usage_reports_cached_tokens(void) {
    sg_result res = wire_test_result(0);
    res.prompt_tokens = 166;
    res.cached_prompt_tokens = 150;
    wire_json[0] = '\0';
    chat_choice ch = { .res = res };
    send_completion_json(-1, "cmpl-5", 0, "test-model", &ch, 1, NULL, NULL);

    jr_value *root = jr_parse(wire_json);
    ASSERT_NOT_NULL(root);
    ASSERT_INT_EQ(wire_cached_tokens(root), 150);
    /* The invariant a rate calculation divides by: computed = prompt - cached
     * must stay positive, so cached can never exceed prompt. */
    const jr_value *usage = jr_object_get(root, "usage");
    int prompt = (int)jr_as_number(jr_object_get(usage, "prompt_tokens"), -1);
    ASSERT_INT_EQ(prompt, 166);
    ASSERT_TRUE(wire_cached_tokens(root) <= prompt);
    /* Not a top-level sibling, and not flattened beside prompt_tokens. */
    ASSERT_NULL(jr_object_get(root, "prompt_tokens_details"));
    ASSERT_NULL(jr_object_get(usage, "cached_tokens"));
    jr_free(root);
}

/* Cold turn, non-streaming: PRESENT and 0.  If this ever goes back to being
 * omitted, a client cannot distinguish "nothing was reused" from "this serve
 * does not report reuse" and has to fall back to the inflated rate. */
static void test_nonstream_usage_reports_zero_cached_tokens_when_cold(void) {
    sg_result res = wire_test_result(0);
    res.cached_prompt_tokens = 0;
    wire_json[0] = '\0';
    chat_choice ch = { .res = res };
    send_completion_json(-1, "cmpl-6", 0, "test-model", &ch, 1, NULL, NULL);

    jr_value *root = jr_parse(wire_json);
    ASSERT_NOT_NULL(root);
    ASSERT_INT_EQ(wire_cached_tokens(root), 0);
    jr_free(root);
}

/* Warm turn, streaming terminal usage chunk — the path benchmarks measure. */
static void test_stream_usage_chunk_reports_cached_tokens(void) {
    stream_ctx sc = { .client_fd = -1, .id = "cmpl-7", .created = 0,
                      .model_id = "test-model", .role_sent = true };
    wire_sse[0] = '\0';
    send_usage_chunk(&sc, 166, 4, 150, 170, 32000, 0);

    jr_value *root = jr_parse(wire_sse);
    ASSERT_NOT_NULL(root);
    ASSERT_INT_EQ(wire_cached_tokens(root), 150);
    const jr_value *usage = jr_object_get(root, "usage");
    ASSERT_TRUE(wire_cached_tokens(root) <=
                (int)jr_as_number(jr_object_get(usage, "prompt_tokens"), -1));
    /* The fields it rides beside are untouched. */
    ASSERT_INT_EQ((int)jr_as_number(jr_object_get(usage, "total_tokens"), -1), 170);
    ASSERT_INT_EQ((int)jr_as_number(jr_object_get(usage, "kv_used"), -1), 170);
    jr_free(root);
}

/* Cold turn, streaming: same present-and-zero rule as the non-streaming body. */
static void test_stream_usage_chunk_reports_zero_cached_tokens_when_cold(void) {
    stream_ctx sc = { .client_fd = -1, .id = "cmpl-8", .created = 0,
                      .model_id = "test-model", .role_sent = true };
    wire_sse[0] = '\0';
    send_usage_chunk(&sc, 166, 4, 0, 170, 32000, 0);

    jr_value *root = jr_parse(wire_sse);
    ASSERT_NOT_NULL(root);
    ASSERT_INT_EQ(wire_cached_tokens(root), 0);
    jr_free(root);
}

/* ============================================================
 * OpenAI FIELDS THAT USED TO BE SILENTLY IGNORED
 *
 * response_format / seed / logit_bias / presence_penalty / frequency_penalty /
 * logprobs / top_logprobs all parsed to nothing before this change: a client
 * sending response_format:{"type":"json_object"} believed it was guaranteed JSON
 * and quietly was not.  These tests assert the parse RESULT (what the generator
 * actually receives) and, for response_format, the reported status on BOTH the
 * streaming and non-streaming wire — parsed back out of the emitted bytes, not
 * substring-matched.
 * ============================================================ */

/* Parse a body and hand back the request; caller frees both. */
static jr_value *parse_body(const char *body, chat_request *req) {
    jr_value *root = jr_parse(body);
    if (!root) return NULL;
    const char *err = NULL;
    if (parse_chat_request(root, req, &err) != 0) { jr_free(root); return NULL; }
    return root;
}

#define BODY_HEAD "{\"messages\":[{\"role\":\"user\",\"content\":\"x\"}]"

static void test_response_format_json_object_is_applied(void) {
    chat_request req; 
    jr_value *root = parse_body(
        BODY_HEAD ",\"response_format\":{\"type\":\"json_object\"}}", &req);
    ASSERT_NOT_NULL(root);
    ASSERT_TRUE(req.gen.response_format == SG_RESPONSE_FORMAT_JSON_OBJECT);
    ASSERT_NOT_NULL(req.response_format_status);
    ASSERT_STR_EQ(req.response_format_status, "applied");
    chat_request_free(&req);
    jr_free(root);
}

static void test_response_format_text_is_a_noop(void) {
    chat_request req;
    jr_value *root = parse_body(
        BODY_HEAD ",\"response_format\":{\"type\":\"text\"}}", &req);
    ASSERT_NOT_NULL(root);
    ASSERT_TRUE(req.gen.response_format == SG_RESPONSE_FORMAT_TEXT);
    /* The explicit default constrains nothing and needs no report. */
    ASSERT_NULL(req.response_format_status);
    chat_request_free(&req);
    jr_free(root);
}

static void test_response_format_absent_is_a_noop(void) {
    chat_request req;
    jr_value *root = parse_body(BODY_HEAD "}", &req);
    ASSERT_NOT_NULL(root);
    ASSERT_TRUE(req.gen.response_format == SG_RESPONSE_FORMAT_TEXT);
    ASSERT_NULL(req.response_format_status);
    chat_request_free(&req);
    jr_free(root);
}

static void test_response_format_json_schema_downgrades(void) {
    /* The shape is not enforced, but the reply is still constrained to valid
     * JSON and the client is TOLD the schema did not apply. */
    chat_request req;
    jr_value *root = parse_body(
        BODY_HEAD ",\"response_format\":{\"type\":\"json_schema\",\"json_schema\":"
        "{\"name\":\"r\",\"schema\":{\"type\":\"object\"}}}}", &req);
    ASSERT_NOT_NULL(root);
    ASSERT_TRUE(req.gen.response_format == SG_RESPONSE_FORMAT_JSON_OBJECT);
    ASSERT_STR_EQ(req.response_format_status, "schema_downgraded");
    chat_request_free(&req);
    jr_free(root);
}

static void test_response_format_unknown_type_is_reported(void) {
    chat_request req;
    jr_value *root = parse_body(
        BODY_HEAD ",\"response_format\":{\"type\":\"xml\"}}", &req);
    ASSERT_NOT_NULL(root);
    ASSERT_TRUE(req.gen.response_format == SG_RESPONSE_FORMAT_TEXT);
    ASSERT_STR_EQ(req.response_format_status, "ignored_unknown_type");
    chat_request_free(&req);
    jr_free(root);
}

/* THE PRECEDENCE RULE.  Tools and response_format are two masks over one decode
 * and the sampler has one hook.  Tools win — and the client is told, so the
 * conflict is visible rather than silent. */
static void test_response_format_yields_to_tools(void) {
    chat_request req;
    jr_value *root = parse_body(
        BODY_HEAD ",\"tools\":[{\"type\":\"function\",\"function\":{\"name\":\"read\"}}]"
        ",\"response_format\":{\"type\":\"json_object\"}}", &req);
    ASSERT_NOT_NULL(root);
    ASSERT_NOT_NULL(req.tools);
    ASSERT_TRUE(req.gen.response_format == SG_RESPONSE_FORMAT_TEXT);
    ASSERT_STR_EQ(req.response_format_status, "ignored_tools_present");
    chat_request_free(&req);
    jr_free(root);
}

/* tool_choice:"none" NULLs the tools, so nothing is competing for the mask and
 * the JSON constraint applies after all. */
static void test_response_format_applies_when_tool_choice_none(void) {
    chat_request req;
    jr_value *root = parse_body(
        BODY_HEAD ",\"tools\":[{\"type\":\"function\",\"function\":{\"name\":\"read\"}}]"
        ",\"tool_choice\":\"none\""
        ",\"response_format\":{\"type\":\"json_object\"}}", &req);
    ASSERT_NOT_NULL(root);
    ASSERT_NULL(req.tools);
    ASSERT_TRUE(req.gen.response_format == SG_RESPONSE_FORMAT_JSON_OBJECT);
    ASSERT_STR_EQ(req.response_format_status, "applied");
    chat_request_free(&req);
    jr_free(root);
}

static void test_seed_is_parsed(void) {
    chat_request req;
    jr_value *root = parse_body(BODY_HEAD ",\"seed\":12345}", &req);
    ASSERT_NOT_NULL(root);
    ASSERT_TRUE(req.gen.seed_set);
    ASSERT_TRUE(req.gen.seed == 12345u);
    chat_request_free(&req);
    jr_free(root);

    /* Absent → unset, so the global RNG is used exactly as before. */
    root = parse_body(BODY_HEAD "}", &req);
    ASSERT_NOT_NULL(root);
    ASSERT_FALSE(req.gen.seed_set);
    chat_request_free(&req);
    jr_free(root);

    /* seed 0 is a REAL seed, not "absent". */
    root = parse_body(BODY_HEAD ",\"seed\":0}", &req);
    ASSERT_NOT_NULL(root);
    ASSERT_TRUE(req.gen.seed_set);
    ASSERT_TRUE(req.gen.seed == 0u);
    chat_request_free(&req);
    jr_free(root);
}

/* `enable_thinking` is the one reasoning control the Qwen3.x template exposes,
 * and it is BINARY.  Absent must stay DEFAULT — that is what keeps every
 * existing client byte-identical, and it is the half a "the field parses" test
 * would miss. */
static void test_enable_thinking_is_parsed(void) {
    chat_request req;
    jr_value *root = parse_body(BODY_HEAD ",\"enable_thinking\":false}", &req);
    ASSERT_NOT_NULL(root);
    ASSERT_TRUE(req.gen.thinking == SG_THINKING_OFF);
    chat_request_free(&req);
    jr_free(root);

    root = parse_body(BODY_HEAD ",\"enable_thinking\":true}", &req);
    ASSERT_NOT_NULL(root);
    ASSERT_TRUE(req.gen.thinking == SG_THINKING_ON);
    chat_request_free(&req);
    jr_free(root);

    /* Absent → DEFAULT: the env still decides, exactly as before. */
    root = parse_body(BODY_HEAD "}", &req);
    ASSERT_NOT_NULL(root);
    ASSERT_TRUE(req.gen.thinking == SG_THINKING_DEFAULT);
    chat_request_free(&req);
    jr_free(root);

    /* A non-boolean is NOT a silent "false" — it leaves the default standing
     * rather than quietly switching a client's reasoning off. */
    root = parse_body(BODY_HEAD ",\"enable_thinking\":\"low\"}", &req);
    ASSERT_NOT_NULL(root);
    ASSERT_TRUE(req.gen.thinking == SG_THINKING_DEFAULT);
    chat_request_free(&req);
    jr_free(root);
}

static void test_penalties_are_parsed_and_clamped(void) {
    chat_request req;
    jr_value *root = parse_body(
        BODY_HEAD ",\"presence_penalty\":1.5,\"frequency_penalty\":-0.25}", &req);
    ASSERT_NOT_NULL(root);
    ASSERT_FLT_NEAR(req.gen.presence_penalty, 1.5f, 1e-6f);
    ASSERT_FLT_NEAR(req.gen.frequency_penalty, -0.25f, 1e-6f);
    chat_request_free(&req);
    jr_free(root);

    /* Out of OpenAI's documented range → clamped, never passed through raw. */
    root = parse_body(
        BODY_HEAD ",\"presence_penalty\":99,\"frequency_penalty\":-99}", &req);
    ASSERT_NOT_NULL(root);
    ASSERT_FLT_NEAR(req.gen.presence_penalty, 2.0f, 1e-6f);
    ASSERT_FLT_NEAR(req.gen.frequency_penalty, -2.0f, 1e-6f);
    chat_request_free(&req);
    jr_free(root);
}

static void test_logit_bias_is_parsed_and_owned(void) {
    chat_request req;
    jr_value *root = parse_body(
        BODY_HEAD ",\"logit_bias\":{\"100\":-100,\"7\":2.5,\"bogus\":1,\"9\":\"x\"}}",
        &req);
    ASSERT_NOT_NULL(root);
    /* Non-numeric key and non-numeric value are skipped, not fatal. */
    ASSERT_INT_EQ(req.gen.n_logit_bias, 2);
    ASSERT_NOT_NULL(req.gen.logit_bias);

    /* The map must survive the DOM being freed — the same lifetime rule
     * forced_name needed (BUG 2); the generator reads it after jr_free(root). */
    jr_free(root);
    ASSERT_TRUE(req.gen.logit_bias[0].token_id == 100u);
    ASSERT_FLT_NEAR(req.gen.logit_bias[0].bias, -100.0f, 1e-6f);
    ASSERT_TRUE(req.gen.logit_bias[1].token_id == 7u);
    ASSERT_FLT_NEAR(req.gen.logit_bias[1].bias, 2.5f, 1e-6f);
    chat_request_free(&req);
}

static void test_logprobs_are_parsed(void) {
    chat_request req;
    jr_value *root = parse_body(
        BODY_HEAD ",\"logprobs\":true,\"top_logprobs\":5}", &req);
    ASSERT_NOT_NULL(root);
    ASSERT_TRUE(req.gen.logprobs);
    ASSERT_INT_EQ(req.gen.logprobs_top, 5);
    chat_request_free(&req);
    jr_free(root);

    /* Over the cap → clamped (the sampler writes into fixed-size arrays). */
    root = parse_body(BODY_HEAD ",\"logprobs\":true,\"top_logprobs\":500}", &req);
    ASSERT_NOT_NULL(root);
    ASSERT_INT_EQ(req.gen.logprobs_top, PN_LOGPROBS_MAX_TOP);
    chat_request_free(&req);
    jr_free(root);

    /* Absent → off, sampler untouched. */
    root = parse_body(BODY_HEAD "}", &req);
    ASSERT_NOT_NULL(root);
    ASSERT_FALSE(req.gen.logprobs);
    chat_request_free(&req);
    jr_free(root);
}

/* ── response_format_status ON THE WIRE, both paths ──
 * The last bug in this file was a field that landed correctly on one path and
 * not the other, so both are asserted by PARSING the emitted bytes. */

static void test_nonstream_reports_response_format_status(void) {
    sg_result res = wire_test_result(0);
    wire_json[0] = '\0';
    chat_choice ch = { .res = res };
    send_completion_json(-1, "cmpl-rf", 0, "test-model", &ch, 1, "applied", NULL);

    jr_value *root = jr_parse(wire_json);
    ASSERT_NOT_NULL(root);
    const jr_value *status = jr_object_get(root, "response_format_status");
    ASSERT_NOT_NULL(status);
    ASSERT_STR_EQ(jr_as_string(status), "applied");
    /* It must not have disturbed the object it sits beside. */
    ASSERT_NOT_NULL(jr_object_get(root, "usage"));
    ASSERT_NOT_NULL(jr_object_get(root, "choices"));
    jr_free(root);
}

static void test_nonstream_omits_status_when_not_requested(void) {
    sg_result res = wire_test_result(0);
    wire_json[0] = '\0';
    chat_choice ch = { .res = res };
    send_completion_json(-1, "cmpl-rf2", 0, "test-model", &ch, 1, NULL, NULL);

    jr_value *root = jr_parse(wire_json);
    ASSERT_NOT_NULL(root);
    ASSERT_NULL(jr_object_get(root, "response_format_status"));
    jr_free(root);
}

static void test_stream_chunk_reports_response_format_status(void) {
    stream_ctx sc = { .client_fd = -1, .id = "cmpl-rf3", .created = 0,
                      .model_id = "test-model", .role_sent = true,
                      .response_format_status = "ignored_tools_present" };
    wire_sse[0] = '\0';
    send_chunk(&sc, "", "stop", NULL, 0);

    jr_value *root = jr_parse(wire_sse);
    ASSERT_NOT_NULL(root);
    const jr_value *status = jr_object_get(root, "response_format_status");
    ASSERT_NOT_NULL(status);
    ASSERT_STR_EQ(jr_as_string(status), "ignored_tools_present");
    /* Still a well-formed chunk: choices[0].finish_reason survives. */
    const jr_value *choice = jr_array_get(jr_object_get(root, "choices"), 0);
    ASSERT_NOT_NULL(choice);
    ASSERT_STR_EQ(jr_as_string(jr_object_get(choice, "finish_reason")), "stop");
    jr_free(root);
}

static void test_stream_chunk_omits_status_when_not_requested(void) {
    stream_ctx sc = { .client_fd = -1, .id = "cmpl-rf4", .created = 0,
                      .model_id = "test-model", .role_sent = true };
    wire_sse[0] = '\0';
    send_chunk(&sc, "", "stop", NULL, 0);

    jr_value *root = jr_parse(wire_sse);
    ASSERT_NOT_NULL(root);
    ASSERT_NULL(jr_object_get(root, "response_format_status"));
    jr_free(root);
}

/* ── OpenAI `n` (multiple completions) ── */

static void test_n_defaults_to_one(void) {
    chat_request req;
    jr_value *root = parse_body(BODY_HEAD "}", &req);
    ASSERT_NOT_NULL(root);
    ASSERT_INT_EQ(req.n_choices, 1);
    ASSERT_NULL(req.n_status);
    chat_request_free(&req);
    jr_free(root);
}

static void test_n_is_honoured(void) {
    chat_request req;
    jr_value *root = parse_body(BODY_HEAD ",\"n\":3}", &req);
    ASSERT_NOT_NULL(root);
    ASSERT_INT_EQ(req.n_choices, 3);
    ASSERT_NULL(req.n_status);
    chat_request_free(&req);
    jr_free(root);
}

static void test_n_over_the_ceiling_is_reported_not_silent(void) {
    chat_request req;
    jr_value *root = parse_body(BODY_HEAD ",\"n\":500}", &req);
    ASSERT_NOT_NULL(root);
    ASSERT_INT_EQ(req.n_choices, OAI_MAX_CHOICES);
    /* The clamp must be VISIBLE — a silent cap is the defect class this whole
     * change exists to remove. */
    ASSERT_NOT_NULL(req.n_status);
    ASSERT_STR_EQ(req.n_status, "capped_to_8");
    chat_request_free(&req);
    jr_free(root);
}

static void test_n_zero_and_negative_floor_at_one(void) {
    chat_request req;
    jr_value *root = parse_body(BODY_HEAD ",\"n\":0}", &req);
    ASSERT_NOT_NULL(req.n_choices == 1 ? root : NULL);
    chat_request_free(&req);
    jr_free(root);
    root = parse_body(BODY_HEAD ",\"n\":-4}", &req);
    ASSERT_NOT_NULL(root);
    ASSERT_INT_EQ(req.n_choices, 1);
    chat_request_free(&req);
    jr_free(root);
}

/* A seeded n>1 request must produce n DIFFERENT completions — reusing one RNG
 * stream would make them identical, which is not what `seed` means. */
static void test_seeded_choices_get_distinct_streams(void) {
    chat_request req;
    jr_value *root = parse_body(BODY_HEAD ",\"seed\":7,\"n\":3}", &req);
    ASSERT_NOT_NULL(root);
    sg_gen_params a = gen_params_for_choice(&req, 0);
    sg_gen_params b = gen_params_for_choice(&req, 1);
    sg_gen_params c = gen_params_for_choice(&req, 2);
    ASSERT_TRUE(a.seed_set && b.seed_set && c.seed_set);
    ASSERT_TRUE(a.seed != b.seed);
    ASSERT_TRUE(b.seed != c.seed);
    /* …and re-deriving choice 0 gives the same stream, so the whole REQUEST is
     * still reproducible. */
    ASSERT_TRUE(gen_params_for_choice(&req, 0).seed == a.seed);
    chat_request_free(&req);
    jr_free(root);
}

/* The wire shape of an n>1 response: n choices with ascending indices, one
 * prompt charged once, completion tokens summed. */
static void test_nonstream_emits_all_choices(void) {
    sg_result r0 = wire_test_result(0);
    sg_result r1 = wire_test_result(0);
    static char alt[] = "yo";
    r1.content = alt;
    r1.completion_tokens = 6;
    chat_choice chs[2] = { { .res = r0 }, { .res = r1 } };
    wire_json[0] = '\0';
    send_completion_json(-1, "cmpl-n", 0, "test-model", chs, 2, NULL, NULL);

    jr_value *root = jr_parse(wire_json);
    ASSERT_NOT_NULL(root);
    const jr_value *choices = jr_object_get(root, "choices");
    ASSERT_INT_EQ(jr_array_len(choices), 2);
    const jr_value *c0 = jr_array_get(choices, 0);
    const jr_value *c1 = jr_array_get(choices, 1);
    ASSERT_INT_EQ((int)jr_as_number(jr_object_get(c0, "index"), -1), 0);
    ASSERT_INT_EQ((int)jr_as_number(jr_object_get(c1, "index"), -1), 1);
    ASSERT_STR_EQ(jr_as_string(jr_object_get(jr_object_get(c1, "message"),
                                             "content")), "yo");
    const jr_value *usage = jr_object_get(root, "usage");
    /* The prompt was prefilled once; the completions add up. */
    ASSERT_INT_EQ((int)jr_as_number(jr_object_get(usage, "prompt_tokens"), -1), 33);
    ASSERT_INT_EQ((int)jr_as_number(jr_object_get(usage, "completion_tokens"), -1),
                  4 + 6);
    ASSERT_INT_EQ((int)jr_as_number(jr_object_get(usage, "total_tokens"), -1),
                  33 + 10);
    jr_free(root);
}

static void test_nonstream_reports_n_shortfall(void) {
    sg_result r0 = wire_test_result(0);
    chat_choice ch = { .res = r0 };
    wire_json[0] = '\0';
    send_completion_json(-1, "cmpl-n2", 0, "test-model", &ch, 1, NULL,
                         "partial_slot_unavailable");
    jr_value *root = jr_parse(wire_json);
    ASSERT_NOT_NULL(root);
    ASSERT_STR_EQ(jr_as_string(jr_object_get(root, "n_status")),
                  "partial_slot_unavailable");
    jr_free(root);
}

static void test_stream_chunk_carries_the_choice_index(void) {
    stream_ctx sc = { .client_fd = -1, .id = "cmpl-n3", .created = 0,
                      .model_id = "test-model", .role_sent = true,
                      .choice_index = 2 };
    wire_sse[0] = '\0';
    send_chunk(&sc, "\"content\":\"x\"", NULL, NULL, 0);
    jr_value *root = jr_parse(wire_sse);
    ASSERT_NOT_NULL(root);
    const jr_value *choice = jr_array_get(jr_object_get(root, "choices"), 0);
    ASSERT_INT_EQ((int)jr_as_number(jr_object_get(choice, "index"), -1), 2);
    jr_free(root);
}

/* ── the STREAMING path really sets response_format_status ──
 * The wire-shape tests above set the field on the ctx by hand, which proves the
 * writer works but NOT that the streaming run path populates it.  That gap is
 * exactly how the previous bug shipped (correct on one path, absent on the
 * other), so drive run_streaming end-to-end against the stubbed generator and
 * read the closing chunk back. */
/* Drive run_streaming end-to-end against the stubbed generator and read the
 * closing chunk back.  Asserts BOTH directions: the streaming path must report
 * "applied" only when the generation actually enforced the constraint, and
 * "not_enforced" when it did not.  A status that claims a guarantee it did not
 * deliver is the defect this whole change exists to remove. */
static void run_streaming_status_case(int enforced, const char *expect) {
    chat_request req;
    jr_value *root = parse_body(
        BODY_HEAD ",\"response_format\":{\"type\":\"json_object\"}}", &req);
    ASSERT_NOT_NULL(root);
    ASSERT_STR_EQ(req.response_format_status, "applied");   /* the INTENT */

    stub_response_format_ok = enforced;
    stream_ctx sc = { .client_fd = -1, .id = "cmpl-s", .created = 0,
                      .model_id = "test-model" };
    wire_sse[0] = '\0';
    run_streaming(-1, NULL, NULL, &req, &req.gen, &sc, NULL);

    /* wire_sse holds the LAST chunk written — the closing one. */
    jr_value *chunk = jr_parse(wire_sse);
    ASSERT_NOT_NULL(chunk);
    ASSERT_STR_EQ(jr_as_string(jr_object_get(chunk, "response_format_status")),
                  expect);
    jr_free(chunk);
    chat_request_free(&req);
    jr_free(root);
}

static void test_run_streaming_reports_enforced_status(void) {
    run_streaming_status_case(1, "applied");
}

static void test_run_streaming_reports_unenforced_status(void) {
    run_streaming_status_case(0, "not_enforced");
}


static void test_status_applied_requires_actual_enforcement(void) {
    chat_request req;
    jr_value *root = parse_body(
        BODY_HEAD ",\"response_format\":{\"type\":\"json_object\"}}", &req);
    ASSERT_NOT_NULL(root);
    ASSERT_STR_EQ(req.response_format_status, "applied");   /* the INTENT */

    sg_result enforced = {0};
    enforced.response_format_ok = 1;
    ASSERT_STR_EQ(wire_response_format_status(&req, &enforced), "applied");

    /* The grammar did not close an object → the reply is not guaranteed JSON,
     * so the wire must NOT claim it was. */
    sg_result not_enforced = {0};
    not_enforced.response_format_ok = 0;
    ASSERT_STR_EQ(wire_response_format_status(&req, &not_enforced),
                  "not_enforced");
    chat_request_free(&req);
    jr_free(root);
}

static void test_status_schema_downgrade_also_requires_enforcement(void) {
    chat_request req;
    jr_value *root = parse_body(
        BODY_HEAD ",\"response_format\":{\"type\":\"json_schema\",\"json_schema\":"
        "{\"name\":\"r\",\"schema\":{\"type\":\"object\"}}}}", &req);
    ASSERT_NOT_NULL(root);
    sg_result ok = {0};   ok.response_format_ok = 1;
    sg_result bad = {0};  bad.response_format_ok = 0;
    ASSERT_STR_EQ(wire_response_format_status(&req, &ok), "schema_downgraded");
    ASSERT_STR_EQ(wire_response_format_status(&req, &bad), "not_enforced");
    chat_request_free(&req);
    jr_free(root);
}

/* The already-honest dispositions describe the REQUEST, not the generation, so
 * they must survive untouched (there was nothing to enforce). */
static void test_status_ignored_reasons_pass_through(void) {
    chat_request req;
    jr_value *root = parse_body(
        BODY_HEAD ",\"tools\":[{\"type\":\"function\",\"function\":{\"name\":\"read\"}}]"
        ",\"response_format\":{\"type\":\"json_object\"}}", &req);
    ASSERT_NOT_NULL(root);
    sg_result res = {0};   /* response_format_ok 0 — nothing was constrained */
    ASSERT_STR_EQ(wire_response_format_status(&req, &res),
                  "ignored_tools_present");
    chat_request_free(&req);
    jr_free(root);

    root = parse_body(BODY_HEAD ",\"response_format\":{\"type\":\"xml\"}}", &req);
    ASSERT_NOT_NULL(root);
    ASSERT_STR_EQ(wire_response_format_status(&req, &res),
                  "ignored_unknown_type");
    chat_request_free(&req);
    jr_free(root);
}

static void test_status_absent_stays_absent(void) {
    chat_request req;
    jr_value *root = parse_body(BODY_HEAD "}", &req);
    ASSERT_NOT_NULL(root);
    sg_result res = {0};
    ASSERT_NULL(wire_response_format_status(&req, &res));
    chat_request_free(&req);
    jr_free(root);
}

/* The wire writer must carry the RESOLVED status, so a non-enforcing turn is
 * visibly non-enforcing in the body a client parses. */
static void test_nonstream_body_carries_not_enforced(void) {
    sg_result res = wire_test_result(0);
    res.response_format_ok = 0;
    chat_choice ch = { .res = res };
    chat_request req;
    jr_value *root = parse_body(
        BODY_HEAD ",\"response_format\":{\"type\":\"json_object\"}}", &req);
    ASSERT_NOT_NULL(root);
    wire_json[0] = '\0';
    send_completion_json(-1, "cmpl-ne", 0, "test-model", &ch, 1,
                         wire_response_format_status(&req, &ch.res), NULL);
    jr_value *body = jr_parse(wire_json);
    ASSERT_NOT_NULL(body);
    ASSERT_STR_EQ(jr_as_string(jr_object_get(body, "response_format_status")),
                  "not_enforced");
    jr_free(body);
    chat_request_free(&req);
    jr_free(root);
}

/* ============================================================
 * RESPONSE SHAPE — tool_call_error on the wire.
 *
 * Three conditions the client used to be able to confuse, asserted through the
 * bytes it actually parses:
 *   1. ABANDONED    — the model opened <parameter=path> and stopped:
 *                     no `tool_calls`, a `tool_call_error` naming `path`.
 *   2. EMPTY ARG    — the model closed an empty value: a real `tool_calls`
 *                     entry with `{"path":""}` arguments, and NO
 *                     `tool_call_error`.
 *   3. NO CALL      — neither field.
 * Conditions 1 and 2 were byte-identical before this (both `read {}`), and the
 * `{}` half of that collision is why 2 now carries the empty value instead of
 * dropping it: a dropped member and a member that never arrived read the same.
 * ============================================================ */

static const char *WIRE_READ_TOOLS =
    "[{\"type\":\"function\",\"function\":{\"name\":\"read\","
    "\"parameters\":{\"type\":\"object\","
    "\"properties\":{\"path\":{\"type\":\"string\"}},"
    "\"required\":[\"path\"]}}}]";

/* Run one turn whose tool channel held `tool_raw` all the way to the emitted
 * non-streaming body, and return that body parsed. Caller jr_frees it. */
static jr_value *wire_emit_tool_turn_dialect(const char *tool_raw,
                                             tc_dialect dialect) {
    jr_value *tools = jr_parse(WIRE_READ_TOOLS);
    chat_request req = {0};
    req.tools = tools;
    req.gen.dialect = dialect;
    chat_choice ch = { .res = wire_test_result(0) };
    ch.res.tool_raw = (char *)tool_raw;   /* borrowed; not freed through res */
    collect_tool_calls(&req, &ch.res, &ch.calls, &ch.n_calls, &ch.tool_error);
    wire_json[0] = '\0';
    send_completion_json(-1, "cmpl-tc", 0, "test-model", &ch, 1, NULL, NULL);
    tc_tool_calls_free(ch.calls, ch.n_calls);
    tc_parse_error_free(&ch.tool_error);
    jr_free(tools);
    return jr_parse(wire_json);
}

/* The XML dialect every other wire test uses. */
static jr_value *wire_emit_tool_turn(const char *tool_raw) {
    return wire_emit_tool_turn_dialect(tool_raw, TC_DIALECT_QWEN_XML);
}

static const jr_value *wire_first_choice(const jr_value *root) {
    return jr_array_get(jr_object_get(root, "choices"), 0);
}

/* 1. ABANDONED — reported, and NOT dressed up as a call. */
static void test_nonstream_body_reports_abandoned_call(void) {
    jr_value *root = wire_emit_tool_turn(
        "<tool_call>\n<function=read>\n<parameter=path>\n\n</function>");
    ASSERT_NOT_NULL(root);
    const jr_value *choice = wire_first_choice(root);
    ASSERT_NOT_NULL(choice);
    ASSERT_NULL(jr_object_get(jr_object_get(choice, "message"), "tool_calls"));
    const jr_value *err = jr_object_get(choice, "tool_call_error");
    ASSERT_NOT_NULL(err);
    ASSERT_STR_EQ(jr_as_string(jr_object_get(err, "code")), "call_abandoned");
    ASSERT_STR_EQ(jr_as_string(jr_object_get(err, "function")), "read");
    ASSERT_STR_EQ(jr_as_string(jr_object_get(err, "parameter")), "path");
    ASSERT_TRUE(jr_as_string(jr_object_get(err, "message")) != NULL);
    jr_free(root);
}

/* 2. EMPTY ARGUMENT — still a call, still carrying the argument, and NOT flagged.
 *
 * `{"path":""}` on the wire, not `{}`.  Empty and absent are different claims and
 * The client acts differently on each: absent is what a truncated call looks like
 * and is charged against a budget that ends the task, while empty is a value the
 * client can judge against the tool (`edit`'s `new_string:""` is a deletion,
 * `read`'s `path:""` is the empty-path wedge). */
static void test_nonstream_body_keeps_empty_argument_call(void) {
    jr_value *root = wire_emit_tool_turn(
        "<tool_call>\n<function=read>\n<parameter=path>\n\n</parameter>\n"
        "</function>");
    ASSERT_NOT_NULL(root);
    const jr_value *choice = wire_first_choice(root);
    ASSERT_NOT_NULL(choice);
    const jr_value *tcs =
        jr_object_get(jr_object_get(choice, "message"), "tool_calls");
    ASSERT_NOT_NULL(tcs);
    ASSERT_INT_EQ(jr_array_len(tcs), 1);
    ASSERT_STR_EQ(jr_as_string(jr_object_get(
                      jr_object_get(jr_array_get(tcs, 0), "function"), "arguments")),
                  "{\"path\":\"\"}");
    ASSERT_NULL(jr_object_get(choice, "tool_call_error"));
    jr_free(root);
}

/* 3. NO CALL AT ALL — neither field; the turn is ordinary text.
 *
 * The fixture is an EMPTY tool channel, which is what "no call" actually looks
 * like: the emitter only ever writes into `tool_raw` after it has seen a
 * tool-call opener, so ordinary prose ends up in `content` and never here.  (It
 * used to pass prose as `tool_raw`, a state the emitter cannot produce; that
 * shape now means something specific — see the no_call_parsed test below.) */
static void test_nonstream_body_silent_when_no_call(void) {
    jr_value *root = wire_emit_tool_turn(NULL);
    ASSERT_NOT_NULL(root);
    const jr_value *choice = wire_first_choice(root);
    ASSERT_NOT_NULL(choice);
    ASSERT_NULL(jr_object_get(jr_object_get(choice, "message"), "tool_calls"));
    ASSERT_NULL(jr_object_get(choice, "tool_call_error"));
    jr_free(root);
}

/* A tool FRAME was emitted and nothing survived parsing: the emitter diverted a
 * region (so `tool_raw` is non-empty) but no call came out of it and no more
 * specific reason was found.  Without this the whole diverted region vanishes —
 * it never reaches `content` either — and the client sees a turn that did
 * nothing, with no way to know why.  qwen-code's analogue is the
 * MALFORMED_TOOL_CALL raised on a tool_calls finish with zero completed calls
 * (converter.ts:1699-1711). */
static void test_nonstream_body_reports_unparseable_frame(void) {
    jr_value *root = wire_emit_tool_turn("<tool_call>\nnothing parseable here\n");
    ASSERT_NOT_NULL(root);
    const jr_value *choice = wire_first_choice(root);
    ASSERT_NOT_NULL(choice);
    ASSERT_NULL(jr_object_get(jr_object_get(choice, "message"), "tool_calls"));
    const jr_value *err = jr_object_get(choice, "tool_call_error");
    ASSERT_NOT_NULL(err);
    ASSERT_STR_EQ(jr_as_string(jr_object_get(err, "code")), "no_call_parsed");
    jr_free(root);
}

/* A truncated JSON argument object reaches the wire as a real tool call rather
 * than as nothing: the serve closes the one '}' the model never got to write.
 * This is the whole path — collect_tool_calls asks for the repair, logs it
 * ([tool_json_repair] on stderr), and the call is serialized normally. */
static void test_nonstream_body_repairs_truncated_json_call(void) {
    jr_value *root = wire_emit_tool_turn_dialect(
        "```json\n{\"function\": \"read\", \"arguments\": {\"path\": \"/etc/hostname\"}",
        TC_DIALECT_ORNITH_JSON);
    ASSERT_NOT_NULL(root);
    const jr_value *choice = wire_first_choice(root);
    ASSERT_NOT_NULL(choice);
    const jr_value *tcs =
        jr_object_get(jr_object_get(choice, "message"), "tool_calls");
    ASSERT_NOT_NULL(tcs);
    ASSERT_INT_EQ(jr_array_len(tcs), 1);
    ASSERT_STR_EQ(jr_as_string(jr_object_get(
                      jr_object_get(jr_array_get(tcs, 0), "function"), "arguments")),
                  "{\"path\":\"/etc/hostname\"}");
    ASSERT_STR_EQ(jr_as_string(jr_object_get(choice, "finish_reason")), "tool_calls");
    ASSERT_NULL(jr_object_get(choice, "tool_call_error"));
    jr_free(root);
}

/* A call the model named but the request never declared: dropped as before (a
 * hallucinated tool must never reach the client), but now NAMED, because "you
 * called a tool that does not exist" is the one drop a client can put back in
 * front of the model as a correction. */
static void test_nonstream_body_reports_unknown_tool_name(void) {
    jr_value *root = wire_emit_tool_turn(
        "<tool_call>\n<function=serach_files>\n<parameter=path>\n/etc\n"
        "</parameter>\n</function>\n</tool_call>");
    ASSERT_NOT_NULL(root);
    const jr_value *choice = wire_first_choice(root);
    ASSERT_NOT_NULL(choice);
    ASSERT_NULL(jr_object_get(jr_object_get(choice, "message"), "tool_calls"));
    const jr_value *err = jr_object_get(choice, "tool_call_error");
    ASSERT_NOT_NULL(err);
    ASSERT_STR_EQ(jr_as_string(jr_object_get(err, "code")), "call_name_unknown");
    ASSERT_STR_EQ(jr_as_string(jr_object_get(err, "function")), "serach_files");
    jr_free(root);
}

/* A healthy call is untouched: the field never appears on a good turn. */
static void test_nonstream_body_clean_call_unflagged(void) {
    jr_value *root = wire_emit_tool_turn(
        "<tool_call>\n<function=read>\n<parameter=path>\n/etc/hostname\n"
        "</parameter>\n</function>");
    ASSERT_NOT_NULL(root);
    const jr_value *choice = wire_first_choice(root);
    ASSERT_NOT_NULL(choice);
    const jr_value *tcs =
        jr_object_get(jr_object_get(choice, "message"), "tool_calls");
    ASSERT_NOT_NULL(tcs);
    ASSERT_INT_EQ(jr_array_len(tcs), 1);
    ASSERT_STR_EQ(jr_as_string(jr_object_get(choice, "finish_reason")),
                  "tool_calls");
    ASSERT_NULL(jr_object_get(choice, "tool_call_error"));
    jr_free(root);
}

/* ============================================================
 * TRUNCATION GATE — a call the DECODING GRAMMAR never closed.
 *
 * The one thing the text cannot tell you.  A truncated tool call is a legal
 * PREFIX of a good one, so every fixture below parses cleanly into a named call
 * with a required argument silently absent — which is exactly what the live
 * fleet handed to the client (`read {}`, `write` with no `content`) while the serve's
 * own grammar state said the call had never finished.  So the verdict rides in
 * on sg_result.tool_grammar_verdict, and these tests assert what the client
 * actually parses.
 * ============================================================ */

/* Same turn as wire_emit_tool_turn, with the grammar's end-of-turn verdict set. */
static jr_value *wire_emit_tool_turn_verdict(const char *tool_raw, int verdict) {
    jr_value *tools = jr_parse(WIRE_READ_TOOLS);
    chat_request req = {0};
    req.tools = tools;
    req.gen.dialect = TC_DIALECT_QWEN_XML;
    chat_choice ch = { .res = wire_test_result(0) };
    ch.res.tool_raw = (char *)tool_raw;   /* borrowed; not freed through res */
    ch.res.tool_grammar_verdict = verdict;
    collect_tool_calls(&req, &ch.res, &ch.calls, &ch.n_calls, &ch.tool_error);
    wire_json[0] = '\0';
    send_completion_json(-1, "cmpl-tc", 0, "test-model", &ch, 1, NULL, NULL);
    tc_tool_calls_free(ch.calls, ch.n_calls);
    tc_parse_error_free(&ch.tool_error);
    jr_free(tools);
    return jr_parse(wire_json);
}

/* THE DEFECT, verbatim off the live serve log (tool_arg_truncation.md log:274):
 * the model emitted `path` as a single newline and stopped.  It parses into
 * `read {}` — a named call with its one required argument gone — and used to be
 * sent as one.  It must now be absent from `tool_calls` and NAMED on the error
 * channel instead. */
static void test_nonstream_body_refuses_incomplete_call(void) {
    jr_value *root = wire_emit_tool_turn_verdict(
        "<tool_call>\n<function=read>\n<parameter=path>\n\n"
        "</parameter></function>\n", SG_TOOL_GRAMMAR_INCOMPLETE);
    ASSERT_NOT_NULL(root);
    const jr_value *choice = wire_first_choice(root);
    ASSERT_NOT_NULL(choice);
    ASSERT_NULL(jr_object_get(jr_object_get(choice, "message"), "tool_calls"));
    const jr_value *err = jr_object_get(choice, "tool_call_error");
    ASSERT_NOT_NULL(err);
    ASSERT_STR_EQ(jr_as_string(jr_object_get(err, "code")), "call_incomplete");
    ASSERT_STR_EQ(jr_as_string(jr_object_get(err, "function")), "read");
    ASSERT_TRUE(jr_as_string(jr_object_get(err, "message")) != NULL);
    jr_free(root);
}

/* The 2-required-argument shape: `path` parsed, `content` emptied and dropped.
 * A call carrying a real-looking `path` is the more dangerous of the two — a
 * client that only checks "did I get a call" would run it. */
static void test_nonstream_body_refuses_partial_arguments(void) {
    jr_value *root = wire_emit_tool_turn_verdict(
        "<function=read>\n<parameter=path>\n/etc/hostname\n</parameter>\n",
        SG_TOOL_GRAMMAR_INCOMPLETE);
    ASSERT_NOT_NULL(root);
    const jr_value *choice = wire_first_choice(root);
    ASSERT_NOT_NULL(choice);
    ASSERT_NULL(jr_object_get(jr_object_get(choice, "message"), "tool_calls"));
    ASSERT_STR_EQ(jr_as_string(jr_object_get(
                      jr_object_get(choice, "tool_call_error"), "code")),
                  "call_incomplete");
    jr_free(root);
}

/* CONTROL — the grammar CLOSED the call: untouched, and unflagged.  Without this
 * the gate could be "refuse everything" and still look green. */
static void test_nonstream_body_keeps_complete_call(void) {
    jr_value *root = wire_emit_tool_turn_verdict(
        "<tool_call>\n<function=read>\n<parameter=path>\n/etc/hostname\n"
        "</parameter>\n</function>", SG_TOOL_GRAMMAR_COMPLETE);
    ASSERT_NOT_NULL(root);
    const jr_value *choice = wire_first_choice(root);
    ASSERT_NOT_NULL(choice);
    const jr_value *tcs =
        jr_object_get(jr_object_get(choice, "message"), "tool_calls");
    ASSERT_NOT_NULL(tcs);
    ASSERT_INT_EQ(jr_array_len(tcs), 1);
    ASSERT_NULL(jr_object_get(choice, "tool_call_error"));
    jr_free(root);
}

/* CONTROL — NO tool grammar ran (PANTHEON_TOOL_GRAMMAR off, or a dialect this
 * grammar does not model, or a walk that never entered a call).  ABSENT is not a
 * soft "incomplete": with no verdict to gate on, the turn must behave exactly as
 * it did before this gate existed. */
static void test_nonstream_body_ungated_without_a_grammar(void) {
    jr_value *root = wire_emit_tool_turn_verdict(
        "<tool_call>\n<function=read>\n<parameter=path>\n\n"
        "</parameter></function>\n", SG_TOOL_GRAMMAR_ABSENT);
    ASSERT_NOT_NULL(root);
    const jr_value *choice = wire_first_choice(root);
    ASSERT_NOT_NULL(choice);
    ASSERT_NOT_NULL(jr_object_get(jr_object_get(choice, "message"), "tool_calls"));
    ASSERT_NULL(jr_object_get(choice, "tool_call_error"));
    jr_free(root);
}

/* A reason the PARSER already named wins — it is the same event seen closer up,
 * and it carries the parameter name.  The gate must not overwrite it. */
static void test_incomplete_does_not_mask_a_named_reason(void) {
    jr_value *root = wire_emit_tool_turn_verdict(
        "<tool_call>\n<function=read>\n<parameter=path>\n\n</function>",
        SG_TOOL_GRAMMAR_INCOMPLETE);
    ASSERT_NOT_NULL(root);
    const jr_value *choice = wire_first_choice(root);
    ASSERT_NOT_NULL(choice);
    ASSERT_NULL(jr_object_get(jr_object_get(choice, "message"), "tool_calls"));
    const jr_value *err = jr_object_get(choice, "tool_call_error");
    ASSERT_STR_EQ(jr_as_string(jr_object_get(err, "code")), "call_abandoned");
    ASSERT_STR_EQ(jr_as_string(jr_object_get(err, "parameter")), "path");
    jr_free(root);
}


/* ============================================================
 * THE TOOL-SYNTAX LEAK.  Everything in collect_tool_calls hangs off
 * `res->tool_raw`, which the emitter only fills once it recognises a real
 * opener.  A model that invents a NEAR-MISS frame matches no opener, is never
 * diverted, and lands in `content` — so the turn ends with no call, no defect
 * code, and NOTHING in the log.  That silence is what made this class of bug
 * cost a session to find: it reads exactly like a turn that chose to answer in
 * prose.
 *
 * The captured text is verbatim from qwen36-bf16 (BF16 full weights, 7 tools,
 * 558-token prompt, grammar off).  stderr is redirected to a temp file so the
 * assertion is on what the fleet log actually receives, not on the call being
 * spelled right.
 * ============================================================ */
static char *capture_leak_stderr(const char *content, const char *tool_raw,
                                 int with_tools) {
    static char buf[4096];
    char path[] = "/tmp/pn_tool_leak_XXXXXX";
    int fd = mkstemp(path);
    if (fd < 0) return NULL;
    fflush(stderr);
    int saved = dup(fileno(stderr));
    dup2(fd, fileno(stderr));

    jr_value *tools = with_tools
        ? jr_parse("[{\"type\":\"function\",\"function\":{\"name\":\"bash\","
                   "\"parameters\":{\"type\":\"object\",\"properties\":"
                   "{\"command\":{\"type\":\"string\"}}}}}]")
        : NULL;
    chat_request req = {0};
    req.tools = tools;
    sg_result res = {0};
    res.content  = (char *)content;
    res.tool_raw = (char *)tool_raw;

    tc_tool_call *calls = NULL; int n = 0; tc_parse_error err = {0};
    collect_tool_calls(&req, &res, &calls, &n, &err);
    tc_tool_calls_free(calls, n);
    tc_parse_error_free(&err);
    jr_free(tools);

    fflush(stderr);
    dup2(saved, fileno(stderr));
    close(saved);
    lseek(fd, 0, SEEK_SET);
    ssize_t got = read(fd, buf, sizeof(buf) - 1);
    buf[got > 0 ? got : 0] = '\0';
    close(fd);
    remove(path);
    return buf;
}

static const char INVENTED_FRAME[] =
    "<tool_calls>\n<tool>\n<tool_name>bash</tool_name>\n<tool>\n"
    "<parameter_name>command</parameter_name>\n"
    "<parameter_value>sed -n '175,200p' src/context/context_store.c"
    "</parameter_value>\n</tool>\n</tool_call>";

static void test_invented_tool_frame_is_reported(void) {
    const char *log = capture_leak_stderr(INVENTED_FRAME, NULL, 1);
    ASSERT_TRUE(log != NULL);
    ASSERT_TRUE(strstr(log, "[tool_syntax_leak]") != NULL);
    /* The raw text has to be IN the line — the whole point is that the client
     * only ever records the parsed result, so this is the one place the bytes
     * that produced the empty turn still exist. */
    ASSERT_TRUE(strstr(log, "<tool_name>bash</tool_name>") != NULL);
}

/* A plain prose answer is the common case and must stay silent, or the signal is
 * worthless. */
static void test_plain_prose_answer_is_silent(void) {
    const char *log = capture_leak_stderr(
        "The file reads a config and returns 0 on success.", NULL, 1);
    ASSERT_TRUE(log != NULL);
    ASSERT_TRUE(strstr(log, "[tool_syntax_leak]") == NULL);
}

/* A turn that DID divert a frame is `no_call_parsed`'s business, not this one:
 * the two silences are different and must not be conflated. */
static void test_diverted_frame_is_not_a_leak(void) {
    const char *log = capture_leak_stderr("", "<tool_call>\n<bash>\n", 1);
    ASSERT_TRUE(log != NULL);
    ASSERT_TRUE(strstr(log, "[tool_syntax_leak]") == NULL);
    ASSERT_TRUE(strstr(log, "no_call_parsed") != NULL);
}

/* No tools on the request → the model was never offered a call, and tool-ish
 * prose (a code answer ABOUT tool calling) is not a defect. */
static void test_no_tools_no_report(void) {
    const char *log = capture_leak_stderr(INVENTED_FRAME, NULL, 0);
    ASSERT_TRUE(log != NULL);
    ASSERT_TRUE(strstr(log, "[tool_syntax_leak]") == NULL);
}

/* ── tool calls stream WHILE they are generated (run_streaming end to end) ──
 * The stubbed generation feeds a real emitter whose tool tap is armed by
 * run_streaming, exactly as serve_generation.c does.  The old behaviour sent
 * nothing of a call until the turn ended — the client's stall watchdog then
 * killed long `write` calls — so these assert on what was on the wire BEFORE
 * the end, and that the streamed fragments add up to the parsed arguments. */

typedef struct {
    char name[32];
    json_writer args;
    int  fragments;
    bool seen;
} folded_call;

/* Fold `len` bytes of SSE the way a client does (tool_calls by index). */
static int fold_wire(const char *wire, size_t len, folded_call *calls, int max,
                     char *finish, size_t finish_len, int *keepalives)
{
    int n = 0;
    *keepalives = 0;
    if (finish_len) finish[0] = '\0';
    char *copy = strndup(wire, len);
    for (char *line = strtok(copy, "\n"); line; line = strtok(NULL, "\n")) {
        if (line[0] == ':') { (*keepalives)++; continue; }
        if (strncmp(line, "data: ", 6) != 0) continue;
        jr_value *ev = jr_parse(line + 6);
        const jr_value *ch = jr_array_get(jr_object_get(ev, "choices"), 0);
        const jr_value *tcs = jr_object_get(jr_object_get(ch, "delta"), "tool_calls");
        for (int i = 0; i < jr_array_len(tcs); i++) {
            const jr_value *tc = jr_array_get(tcs, i);
            int idx = (int)jr_as_number(jr_object_get(tc, "index"), -1);
            if (idx < 0 || idx >= max) continue;
            folded_call *fc = &calls[idx];
            if (!fc->seen) { fc->seen = true; jw_init(&fc->args); if (idx >= n) n = idx + 1; }
            const jr_value *fn = jr_object_get(tc, "function");
            const char *name = jr_as_string(jr_object_get(fn, "name"));
            if (name) snprintf(fc->name, sizeof(fc->name), "%s", name);
            const char *args = jr_as_string(jr_object_get(fn, "arguments"));
            if (args && *args) { jw_raw(&fc->args, args); fc->fragments++; }
        }
        const char *fr = jr_as_string(jr_object_get(ch, "finish_reason"));
        if (fr && finish_len) snprintf(finish, finish_len, "%s", fr);
        jr_free(ev);
    }
    free(copy);
    return n;
}

static void free_folded(folded_call *calls, int n)
{
    for (int i = 0; i < n; i++) if (calls[i].seen) jw_free(&calls[i].args);
}

#define TOOLS_WRITE ",\"tools\":[{\"type\":\"function\",\"function\":{\"name\":\"write\"," \
    "\"parameters\":{\"type\":\"object\",\"properties\":{\"path\":{\"type\":\"string\"}," \
    "\"content\":{\"type\":\"string\"}},\"required\":[\"path\",\"content\"]}}}]"

static const char *const ORNITH_WRITE_PIECES[] = {
    "I'll write it.\n\n", "```", "json\n", "{\n  \"", "function", "\": \"",
    "write", "\",\n  \"", "arguments", "\": {\n    \"path\": \"", "a.py",
    "\",\n    \"content\": \"", "import os\\n", "\\n", "def main():\\n",
    "    print(\\\"hi\\\")\\n", "    return 0\\n", "\\n", "if __name__ == ",
    "\\\"__main__\\\":\\n", "    main()\\n", "\"\n  }\n}", "\n```",
};

/* Run one streaming completion over `pieces`; leaves the wire in wire_all. */
static void run_stub_stream(chat_request *req, const char *const *pieces, int n,
                            double keepalive_s)
{
    jw_free(&wire_all);
    jw_init(&wire_all);
    stub_pieces = pieces;
    stub_n_pieces = n;
    stream_ctx sc = { .client_fd = -1, .id = "cmpl-tc", .created = 0,
                      .model_id = "test-model", .keepalive_s = keepalive_s };
    run_streaming(-1, NULL, NULL, req, &req->gen, &sc, NULL);
    stub_pieces = NULL;
    stub_n_pieces = 0;
}

static void test_stream_tool_call_reaches_wire_before_turn_ends(void) {
    chat_request req;
    jr_value *root = parse_body(BODY_HEAD TOOLS_WRITE ",\"stream\":true}", &req);
    ASSERT_NOT_NULL(root);
    req.gen.dialect = TC_DIALECT_ORNITH_JSON;   /* what chat_apply_tool_dialect sets for ornith */
    int n = (int)(sizeof(ORNITH_WRITE_PIECES) / sizeof(*ORNITH_WRITE_PIECES));
    run_stub_stream(&req, ORNITH_WRITE_PIECES, n, 0);

    /* What the client had BEFORE generation ended: the opening delta and
     * argument fragments — not silence. */
    folded_call early[4] = {0};
    char fin[32];
    int ka = 0;
    int n_early = fold_wire(wire_all.data, stub_wire_len_before_end, early, 4,
                            fin, sizeof(fin), &ka);
    ASSERT_INT_EQ(n_early, 1);
    ASSERT_STR_EQ(early[0].name, "write");
    ASSERT_TRUE(early[0].fragments >= 5);
    ASSERT_TRUE(strstr(early[0].args.data, "def main()") != NULL);

    /* The whole stream: one call whose fragments concatenate to the parsed
     * arguments (what the non-streaming body carries), finish tool_calls. */
    folded_call all[4] = {0};
    int n_all = fold_wire(wire_all.data, wire_all.len, all, 4, fin, sizeof(fin), &ka);
    ASSERT_INT_EQ(n_all, 1);
    json_writer raw;
    jw_init(&raw);
    for (int k = 0; k < n; k++) jw_raw(&raw, ORNITH_WRITE_PIECES[k]);
    tc_tool_call *calls = NULL;
    int n_calls = 0;
    tc_parse_tool_calls(strstr(raw.data, "```"), req.tools, TC_DIALECT_ORNITH_JSON,
                        &calls, &n_calls);
    ASSERT_INT_EQ(n_calls, 1);
    ASSERT_STR_EQ(all[0].args.data, calls[0].arguments);
    ASSERT_STR_EQ(fin, "tool_calls");
    tc_tool_calls_free(calls, n_calls);
    jw_free(&raw);
    free_folded(early, n_early);
    free_folded(all, n_all);
    chat_request_free(&req);
    jr_free(root);
}

/* A call that cannot be streamed early (the XML dialect) must still not leave
 * the wire silent: keepalive comments while it is written, the call itself
 * at the end exactly as before. */
static void test_stream_keepalive_while_call_is_held(void) {
    static const char *const pieces[] = {
        "<tool_call>\n", "<function=write>\n", "<parameter=path>\na.txt\n",
        "</parameter>\n", "<parameter=content>\nhello\n", "</parameter>\n",
        "</function>\n", "</tool_call>",
    };
    chat_request req;
    jr_value *root = parse_body(BODY_HEAD TOOLS_WRITE ",\"stream\":true}", &req);
    ASSERT_NOT_NULL(root);
    run_stub_stream(&req, pieces, (int)(sizeof(pieces) / sizeof(*pieces)), 1e-9);

    folded_call early[4] = {0};
    char fin[32];
    int ka = 0;
    int n_early = fold_wire(wire_all.data, stub_wire_len_before_end, early, 4,
                            fin, sizeof(fin), &ka);
    ASSERT_INT_EQ(n_early, 0);              /* held: nothing of the call yet … */
    ASSERT_TRUE(ka >= 1);                   /* … but the stream is not silent */

    folded_call all[4] = {0};
    int n_all = fold_wire(wire_all.data, wire_all.len, all, 4, fin, sizeof(fin), &ka);
    ASSERT_INT_EQ(n_all, 1);
    ASSERT_STR_EQ(all[0].name, "write");
    ASSERT_STR_EQ(all[0].args.data, "{\"path\":\"a.txt\",\"content\":\"hello\"}");
    ASSERT_STR_EQ(fin, "tool_calls");
    free_folded(all, n_all);
    chat_request_free(&req);
    jr_free(root);
}

int main(void) {
    printf("\n=== chat_request (tool_choice parse + forced_name lifetime) ===\n\n");
    RUN_TEST(forced_name_survives_dom_free);
    RUN_TEST(tool_choice_string_forms);
    RUN_TEST(session_key_aliases);
    RUN_TEST(required_no_forced_name);
    RUN_TEST(forced_without_tools_downgrades);
    RUN_TEST(malformed_object_form);
    printf("\n=== tool-syntax leak (a frame that matched no opener) ===\n\n");
    RUN_TEST(invented_tool_frame_is_reported);
    RUN_TEST(plain_prose_answer_is_silent);
    RUN_TEST(diverted_frame_is_not_a_leak);
    RUN_TEST(no_tools_no_report);
    printf("\n=== openai fields (response_format / seed / penalties / bias / logprobs) ===\n\n");
    RUN_TEST(response_format_json_object_is_applied);
    RUN_TEST(response_format_text_is_a_noop);
    RUN_TEST(response_format_absent_is_a_noop);
    RUN_TEST(response_format_json_schema_downgrades);
    RUN_TEST(response_format_unknown_type_is_reported);
    RUN_TEST(response_format_yields_to_tools);
    RUN_TEST(response_format_applies_when_tool_choice_none);
    RUN_TEST(seed_is_parsed);
    RUN_TEST(enable_thinking_is_parsed);
    RUN_TEST(penalties_are_parsed_and_clamped);
    RUN_TEST(logit_bias_is_parsed_and_owned);
    RUN_TEST(logprobs_are_parsed);
    RUN_TEST(nonstream_reports_response_format_status);
    RUN_TEST(nonstream_omits_status_when_not_requested);
    RUN_TEST(stream_chunk_reports_response_format_status);
    RUN_TEST(stream_chunk_omits_status_when_not_requested);
    RUN_TEST(run_streaming_reports_enforced_status);
    RUN_TEST(run_streaming_reports_unenforced_status);
    RUN_TEST(n_defaults_to_one);
    RUN_TEST(n_is_honoured);
    RUN_TEST(n_over_the_ceiling_is_reported_not_silent);
    RUN_TEST(n_zero_and_negative_floor_at_one);
    RUN_TEST(seeded_choices_get_distinct_streams);
    RUN_TEST(nonstream_emits_all_choices);
    RUN_TEST(nonstream_reports_n_shortfall);
    RUN_TEST(stream_chunk_carries_the_choice_index);
    RUN_TEST(status_applied_requires_actual_enforcement);
    RUN_TEST(status_schema_downgrade_also_requires_enforcement);
    RUN_TEST(status_ignored_reasons_pass_through);
    RUN_TEST(status_absent_stays_absent);
    RUN_TEST(nonstream_body_carries_not_enforced);

    RUN_TEST(nonstream_body_reports_abandoned_call);
    RUN_TEST(nonstream_body_keeps_empty_argument_call);
    RUN_TEST(nonstream_body_silent_when_no_call);
    RUN_TEST(nonstream_body_reports_unparseable_frame);
    RUN_TEST(nonstream_body_repairs_truncated_json_call);
    RUN_TEST(nonstream_body_reports_unknown_tool_name);
    RUN_TEST(nonstream_body_clean_call_unflagged);
    RUN_TEST(nonstream_body_refuses_incomplete_call);
    RUN_TEST(nonstream_body_refuses_partial_arguments);
    RUN_TEST(nonstream_body_keeps_complete_call);
    RUN_TEST(nonstream_body_ungated_without_a_grammar);
    RUN_TEST(incomplete_does_not_mask_a_named_reason);
    printf("\n=== chat response shape (session_conflict on the wire) ===\n\n");
    RUN_TEST(nonstream_usage_carries_session_conflict);
    RUN_TEST(nonstream_usage_omits_session_conflict_when_clean);
    RUN_TEST(stream_usage_chunk_carries_session_conflict);
    RUN_TEST(stream_usage_chunk_omits_session_conflict_when_clean);
    printf("\n=== chat response shape (cached_tokens on the wire) ===\n\n");
    RUN_TEST(nonstream_usage_reports_cached_tokens);
    RUN_TEST(nonstream_usage_reports_zero_cached_tokens_when_cold);
    RUN_TEST(stream_usage_chunk_reports_cached_tokens);
    RUN_TEST(stream_usage_chunk_reports_zero_cached_tokens_when_cold);
    RUN_TEST(stream_tool_call_reaches_wire_before_turn_ends);
    RUN_TEST(stream_keepalive_while_call_is_held);
    TEST_REPORT();
    return __ft_failed == 0 ? 0 : 1;
}
