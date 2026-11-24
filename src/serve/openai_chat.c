/* SPDX-License-Identifier: AGPL-3.0-or-later */
/* Copyright (C) 2026 David (Corey) Herrera */
/**
 * @file openai_chat.c
 * @brief Implementation of the OpenAI-compatible endpoint handlers
 *        (see openai_chat.h).
 *
 * Reasoning handling: the model emits <think>…</think>. We surface the thinking
 * as a separate `reasoning_content` field (message.reasoning_content in the
 * non-streaming reply; delta.reasoning_content in stream chunks) and the answer
 * as `content` — matching the llama.cpp oracle's `--reasoning-format deepseek`,
 * which is exactly what the user's opencode `nex` provider already consumes.
 *
 * Tool calling: when the request carries `tools`, they are rendered into the
 * prompt in the resident model's tool dialect (see tool_calling.c) — the
 * Qwen3-Coder XML form by default, or the MiniCPM5 <function name=…> form when
 * the routed model is MiniCPM5 (chat_apply_tool_dialect swaps the preamble
 * once the model is known). The model's tool-call output is parsed into
 * structured calls, and the reply carries OpenAI `tool_calls` (non-stream) /
 * the tool-call delta sequence (stream) with finish_reason "tool_calls".
 * Chat-only requests are unchanged.
 *
 * Multi-model routing: the request's `model` field selects the resident
 * registry entry (route_model below — single resident model always wins for
 * back-compat; several require an exact name, else 404). Generation is
 * serialized per entry (gen_lock trylock → 429), so different models serve
 * concurrently.
 *
 * OpenAI sampling/decoding fields: the request's `response_format`, `seed`,
 * `logit_bias`, `presence_penalty`, `frequency_penalty`, `logprobs` /
 * `top_logprobs` and `n` are parsed here and threaded into sg_gen_params. They
 * were previously accepted and silently ignored, which is worse than rejecting
 * them — a client sending `response_format:{"type":"json_object"}` believed it
 * was guaranteed JSON and quietly was not. Whatever the serve actually did with
 * `response_format` comes back as a top-level `response_format_status`, and a
 * short `n` comes back as `n_status`, on BOTH the streaming and non-streaming
 * paths.
 *
 * Tools vs response_format: they are two masks over one decode and the sampler
 * has one grammar hook, so TOOLS WIN and the client is told
 * ("ignored_tools_present"). See resolve_response_format.
 *
 * Repetition collapse: a turn the decode loop stopped because the model was
 * looping (repetition_loop_detector.h) comes back with finish_reason
 * SG_FINISH_REPETITION_LOOP on BOTH paths, and that value OUTRANKS "tool_calls"
 * — a caller must be able to tell a degenerated turn from a finished one, and a
 * collapse reported as a tidy stop is the defect this exists to close.
 *
 * @status /v1/models and /v1/chat/completions (streaming + non-streaming),
 *         including tool calling and multi-step tool loops, work — now
 *         registry-routed (Phase 7 multi-model). The OpenAI sampling fields
 *         above all take effect on both paths.
 * @issues `response_format:{"type":"json_schema"}` constrains the reply to valid
 *         JSON but does NOT enforce the schema's shape; it is reported as
 *         "schema_downgraded" rather than pretending to comply.
 *         `n` is capped at OAI_MAX_CHOICES (reported as "capped_to_8"), and each
 *         extra completion is a full re-decode with no prefix-cache reuse — on a
 *         K80 that is minutes per choice.
 * @todo None
 */

#include "openai_chat.h"

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

#include "batch_sched.h"
#include "http_server.h"
#include "field_patterns_request.h"
#include "json_reader.h"
#include "session_key.h"
#include "attn_residency_json.h"
#include "graph_residency_json.h"
#include "json_writer.h"
#include "model_registry.h"
#include "request_profile.h"
#include "model_session.h"
#include "replica_affinity.h"
#include "replica_pin.h"
#include "serve_generation.h"
#include "serve_request_dump.h"
#include "serve_shutdown.h"
#include "tool_call_stream.h"
#include "tool_calling.h"
#include "tool_raw_excerpt.h"
#include "base64.h"
#include "sampling_defaults.h"
#include "../accel/backend_identity.h"
#include "../core/engine.h"
#include "../core/mem/pn_host_admission.h"
#include "../core/mem/pn_mem_state.h"
#include "../core/mem/pn_memwatch.h"
#include "../core/prefix_cache.h"
#include "../core/session_snapshot.h"
#include "../core/think_history.h"
#include "../models/qwen35moe.h"

#define OAI_DEFAULT_MAX_TOKENS  1024

/* Ceiling on OpenAI `n`.  Every extra completion is another full decode on a
 * K80 — minutes, and another decode slot — so an unbounded `n` is a trivial way
 * to wedge the fleet.  Over the ceiling the request is NOT refused and NOT
 * silently trimmed: it returns this many completions and says so in `n_status`. */
#define OAI_MAX_CHOICES      8
#define OAI_MAX_CHOICES_STR "8"

/* Parsed request: sampling params + message array + stream flag + the model
 * routing name. Owns the messages/stop arrays it allocates and any synthesized
 * message-content strings (`owned`). Borrowed strings point into the jr_value
 * tree, which must outlive this struct. */
typedef struct {
    sg_message      *messages;
    int              n_messages;
    sg_gen_params    gen;
    bool             stream;
    bool             include_usage; /* stream_options.include_usage: emit a
                                     * terminal usage chunk on the SSE stream
                                     * (OpenAI-standard; needed for metering). */
    const char      *model;      /* borrowed "model" field (NULL if absent) */
    const char      *session_id; /* borrowed "session_id" body field (NULL if
                                  * absent); PC0.1 prefix-cache key. */
    const jr_value  *tools;      /* borrowed; NULL if absent or tool_choice=none */
    char           **owned;      /* synthesized content strings to free */
    int              n_owned;
    int              preamble_slot; /* messages[] index of the synthetic tools
                                     * system turn, or -1 if none — lets the
                                     * post-route dialect fixup replace it. */
    const char      *sys_content;   /* borrowed leading system message content
                                     * merged into the preamble (NULL if none). */
    const jr_value  *raw_messages;  /* borrowed request messages[] DOM (outlives
                                     * this struct); re-walked post-route by
                                     * chat_apply_tool_dialect to re-render
                                     * assistant tool_calls in the model's dialect. */
    int              msg_start;     /* index in raw_messages where the copied
                                     * history begins (1 past a leading system
                                     * message when the tools preamble absorbed
                                     * it, else 0) — maps messages[] back to
                                     * raw_messages for the re-render. */
    bool             reasoning_budget_set; /* request carried a reasoning_budget /
                                     * max_reasoning_tokens field — its value in
                                     * gen.reasoning_budget then OVERRIDES the
                                     * per-model default (0 = explicit disable). */
    const char      *response_format_status; /* what actually happened to the
                                     * request's `response_format` — "applied",
                                     * "schema_downgraded", "ignored_tools_present"
                                     * or "ignored_unknown_type".  NULL when the
                                     * request carried no response_format.  Static
                                     * strings; reported on BOTH the streaming and
                                     * non-streaming wire so a client never has to
                                     * guess whether its guarantee held. */
    int              n_choices;      /* OpenAI `n` — how many completions to
                                     * produce (default 1).  Each one is a full
                                     * decode, so n > 1 costs n x the generation
                                     * time; see OAI_MAX_CHOICES. */
    const char      *n_status;       /* NULL, or why the delivered choice count
                                     * differs from the requested `n` (static
                                     * string, reported on the wire). */
    pn_logit_bias_entry *logit_bias; /* OWNED parsed `logit_bias` map; gen.logit_bias
                                     * borrows it, so it must outlive the completion
                                     * (freed in chat_request_free). */
    pn_model_sampling sampling;   /* the sampling fields this request CARRIED;
                                     * re-resolved against the routed model's
                                     * defaults (sampling_defaults.h). */
} chat_request;

/* Take ownership of `s` for later free and return it as a borrowed pointer.
 * Frees `s` and returns NULL on allocation failure. */
static const char *req_own(chat_request *req, char *s)
{
    if (!s) return NULL;
    char **grown = realloc(req->owned, (size_t)(req->n_owned + 1) * sizeof(char *));
    if (!grown) { free(s); return NULL; }
    req->owned = grown;
    req->owned[req->n_owned++] = s;
    return s;
}

/* Generate a short pseudo-unique completion id (chatcmpl-XXXXXXXX). */
static void make_completion_id(char *buf, size_t len)
{
    static const char hex[] = "0123456789abcdef";
    char rnd[9];
    for (int i = 0; i < 8; i++) rnd[i] = hex[rand() & 0xF];
    rnd[8] = '\0';
    snprintf(buf, len, "chatcmpl-%s", rnd);
}



/* Measured cold-prefill throughput (tokens/sec) for the restart estimate the
 * serve reports on a capacity refuse. Measured K80 prefill sits ~17 tok/s
 * (Qwen3.6 Q3_K); a conservative single constant keeps the estimate honest
 * without a per-model probe (Phase 3 can refine). Override via
 * PN_PREFILL_TOK_PER_S. */
static double oai_prefill_tok_per_s(void)
{
    const char *env = getenv("PN_PREFILL_TOK_PER_S");
    if (env && *env) {
        char *end = NULL;
        double v = strtod(env, &end);
        if (end != env && *end == '\0' && v > 0.0)
            return v;
    }
    /* Phase 7: prefer the live measured rate once we have one; fall back to the
     * conservative constant until the first real prefill lands. */
    double measured = sg_prefill_tok_per_s_measured();
    if (measured > 0.0)
        return measured;
    return 17.0;
}

/* Collect stop strings from the request "stop" field (string or array). Returns
 * a malloc'd array of borrowed const char* (caller frees the array, not the
 * strings); sets *n. */
static char **parse_stops(const jr_value *stop, int *n)
{
    *n = 0;
    if (!stop) return NULL;
    if (jr_kind(stop) == JR_STRING) {
        char **arr = malloc(sizeof(char *));
        if (!arr) return NULL;
        arr[0] = (char *)jr_as_string(stop);
        *n = 1;
        return arr;
    }
    if (jr_kind(stop) == JR_ARRAY) {
        int count = jr_array_len(stop);
        if (count == 0) return NULL;
        char **arr = malloc((size_t)count * sizeof(char *));
        if (!arr) return NULL;
        for (int i = 0; i < count; i++) {
            const char *s = jr_as_string(jr_array_get(stop, i));
            if (s) arr[(*n)++] = (char *)s;
        }
        return arr;
    }
    return NULL;
}

/* ============================================================
 * OpenAI sampling / decoding field parsing
 * ============================================================ */

/* Clamp a client-supplied number into [lo, hi], mapping NaN to 0.  A raw
 * (int)/(float) cast of an unbounded or NaN JSON number is undefined behaviour,
 * and every one of these fields is client-controlled.
 *
 * NaN test is isnan(), not the `v == v` self-compare idiom: this TU builds
 * with -ffast-math, which implies -ffinite-math-only, and the compiler folds
 * `v == v` to constant true (proven — the whole check compiles to `return 0`
 * unconditionally), so the self-compare form silently never detects NaN. */
static double clamp_number(double v, double lo, double hi)
{
    if (isnan(v)) return 0.0;
    if (v < lo) return lo;
    if (v > hi) return hi;
    return v;
}

/*
 * Resolve `response_format` against the rest of the request.
 *
 * Precedence rule (the one thing that MUST be defined rather than left to
 * chance): a tool-call grammar and a response-format grammar are two masks over
 * the same decode and the sampler has ONE hook — they cannot both run.  When the
 * request carries tools, TOOLS WIN: the model's job that turn is to decide
 * whether to call one, and a JSON-object constraint would make the trained
 * tool-call syntax unreachable, breaking the thing the fleet depends on.  That
 * is also OpenAI's shape — `response_format` describes the assistant's CONTENT,
 * and a turn that produces `tool_calls` produces no content to constrain.  The
 * request is NOT refused: tools work normally and `response_format_status` comes
 * back "ignored_tools_present" so the client can SEE it instead of inferring it
 * from unparseable output.
 *
 * `json_schema` is accepted and decoded under the json_object constraint — the
 * reply is valid JSON but its SHAPE is not enforced — and reported as
 * "schema_downgraded".  An unrecognized type reports "ignored_unknown_type".
 *
 * Sets req->gen.response_format and req->response_format_status.  Must be called
 * AFTER req->tools has been resolved.
 */
static void resolve_response_format(chat_request *req, const jr_value *root)
{
    const jr_value *rf = jr_object_get(root, "response_format");
    if (jr_kind(rf) != JR_OBJECT) return;      /* absent → text, status NULL */

    const char *type = jr_as_string(jr_object_get(rf, "type"));
    if (!type || strcmp(type, "text") == 0)
        return;   /* the explicit default: nothing to constrain, nothing to say */

    bool json_object = (strcmp(type, "json_object") == 0);
    bool json_schema = (strcmp(type, "json_schema") == 0);
    if (!json_object && !json_schema) {
        req->response_format_status = "ignored_unknown_type";
        return;
    }
    if (req->tools) {
        req->response_format_status = "ignored_tools_present";
        return;
    }
    req->gen.response_format    = SG_RESPONSE_FORMAT_JSON_OBJECT;
    req->response_format_status = json_schema ? "schema_downgraded" : "applied";
}

/*
 * Parse `logit_bias` — an object mapping a decimal token-id string to a bias.
 *
 * Stored as an OWNED array on the request (gen.logit_bias borrows it) because
 * the map must outlive the jr_value DOM the caller frees — the same lifetime
 * hazard the forced tool name hit (BUG 2).  A non-numeric key or value is
 * skipped rather than failing the request.  Biases are clamped to OpenAI's
 * documented [-100, 100].
 */
static void parse_logit_bias(chat_request *req, const jr_value *root)
{
    const jr_value *map = jr_object_get(root, "logit_bias");
    if (jr_kind(map) != JR_OBJECT) return;
    int n = jr_object_member_count(map);
    if (n <= 0) return;

    pn_logit_bias_entry *entries = calloc((size_t)n, sizeof(*entries));
    if (!entries) return;   /* OOM → the bias is simply not applied */

    int out = 0;
    for (int i = 0; i < n; i++) {
        const char *key = jr_object_key_at(map, i);
        if (!key || !*key) continue;
        char *end = NULL;
        long id = strtol(key, &end, 10);
        if (end == key || !end || *end != '\0' || id < 0) continue;
        const jr_value *v = jr_object_value_at(map, i);
        if (jr_kind(v) != JR_NUMBER) continue;
        entries[out].token_id = (uint32_t)id;
        entries[out].bias =
            (float)clamp_number(jr_as_number(v, 0.0), -100.0, 100.0);
        out++;
    }
    if (out == 0) { free(entries); return; }
    req->logit_bias       = entries;
    req->gen.logit_bias   = entries;
    req->gen.n_logit_bias = out;
}

/*
 * Parse `seed`, `presence_penalty`, `frequency_penalty`, `logprobs`,
 * `top_logprobs` and `logit_bias`.  All optional; absent leaves the generation
 * byte-identical to before these fields existed.
 */
static void parse_openai_sampling_fields(chat_request *req, const jr_value *root)
{
    const jr_value *seed = jr_object_get(root, "seed");
    if (jr_kind(seed) == JR_NUMBER) {
        /* The wire value is a JSON double; clamp to the exactly-representable
         * integer range so a huge seed maps to a stable stream rather than an
         * undefined cast. */
        double sv = clamp_number(jr_as_number(seed, 0.0),
                                 -9007199254740992.0, 9007199254740992.0);
        req->gen.seed_set = true;
        req->gen.seed = (uint64_t)(int64_t)sv;
    }

    req->gen.frequency_penalty = (float)clamp_number(
        jr_as_number(jr_object_get(root, "frequency_penalty"), 0.0), -2.0, 2.0);

    req->gen.logprobs = jr_as_bool(jr_object_get(root, "logprobs"), false);
    req->gen.logprobs_top = (int)clamp_number(
        jr_as_number(jr_object_get(root, "top_logprobs"), 0.0),
        0.0, (double)PN_LOGPROBS_MAX_TOP);

    parse_logit_bias(req, root);

    /* `n`: each completion is a SEPARATE full decode of the same prompt, so the
     * cost is linear in n and, on this hardware, minutes per extra choice.  It
     * is honoured rather than ignored, and the ceiling is a resource guard, not
     * a silent clamp — a request over it still runs, delivers OAI_MAX_CHOICES
     * completions, and is TOLD so in `n_status`. */
    req->n_choices = (int)clamp_number(jr_as_number(jr_object_get(root, "n"), 1.0),
                                       1.0, 1.0e6);
    if (req->n_choices > OAI_MAX_CHOICES) {
        req->n_choices = OAI_MAX_CHOICES;
        req->n_status  = "capped_to_" OAI_MAX_CHOICES_STR;
    }
}

/*
 * The `response_format_status` actually put on the wire, resolved against what
 * the generation DID rather than what the request asked for.
 *
 * `req->response_format_status` is the request-time INTENT ("applied" /
 * "schema_downgraded" when we meant to constrain).  Intent is not evidence: if
 * the JSON grammar did not close a complete object, the reply is not JSON and
 * saying "applied" would claim a guarantee that did not hold — the same
 * silent-degradation shape as ignoring the field outright, just louder.  So a
 * constrain-intent turn that did not enforce reports "not_enforced" instead.
 * The already-honest dispositions ("ignored_tools_present",
 * "ignored_unknown_type") pass through unchanged.
 */
static const char *wire_response_format_status(const chat_request *req,
                                               const sg_result *res)
{
    const char *intent = req->response_format_status;
    if (!intent) return NULL;
    bool meant_to_constrain = (strcmp(intent, "applied") == 0 ||
                               strcmp(intent, "schema_downgraded") == 0);
    if (!meant_to_constrain) return intent;
    if (res && res->response_format_ok) return intent;
    return "not_enforced";
}

/* An assistant message's trimmed text content and the render of its tool_calls
 * as one owned string (tc_join_assistant_history), registered on the request.
 * Returns the combined borrowed string (falls back to `content` on OOM). */
static const char *combine_assistant_tool_calls(chat_request *req,
                                                const char *content,
                                                const jr_value *tool_calls,
                                                tc_dialect dialect)
{
    char *combined = tc_join_assistant_history(content, tool_calls, dialect);
    if (!combined) return content ? content : "";
    const char *owned = req_own(req, combined);
    return owned ? owned : (content ? content : "");
}

/* A message's text as the chat template renders it: a string as is; an array's
 * text parts concatenated in order (image/video parts skipped: no vision);
 * else NULL. */
static const char *message_text(chat_request *req, const jr_value *content)
{
    jr_type k = jr_kind(content);
    if (k == JR_STRING)
        return jr_as_string(content);
    if (k == JR_ARRAY) {
        int n = jr_array_len(content);
        char *buf = NULL;
        size_t len = 0, cap = 0;
        for (int i = 0; i < n; i++) {
            const jr_value *item = jr_array_get(content, i);
            if (jr_kind(item) != JR_OBJECT)
                continue;
            const char *type = jr_as_string(jr_object_get(item, "type"));
            if (type && (strcmp(type, "image") == 0 || strcmp(type, "video") == 0))
                continue;
            if (jr_object_get(item, "image") ||
                jr_object_get(item, "image_url") ||
                jr_object_get(item, "video"))
                continue;
            const char *txt = jr_as_string(jr_object_get(item, "text"));
            if (!txt)
                continue;
            size_t tlen = strlen(txt);
            if (len + tlen > cap) {
                size_t new_cap = cap ? cap * 2 : 256;
                while (new_cap < len + tlen)
                    new_cap *= 2;
                char *tmp = realloc(buf, new_cap);
                if (!tmp) { free(buf); return NULL; }
                buf = tmp;
                cap = new_cap;
            }
            memcpy(buf + len, txt, tlen);
            len += tlen;
        }
        if (!buf) {
            buf = malloc(1);
            if (!buf) return NULL;
            buf[0] = '\0';
        }
        return req_own(req, buf);
    }
    return NULL;
}

/* Build the sg_message[] the generator consumes. When `tools` is present the
 * first message becomes a synthetic system turn carrying the tools preamble
 * (merging any leading system message), and assistant history tool_calls are
 * expanded back into the model's XML syntax. Returns 0 on success. */
static int build_messages(const jr_value *msgs, chat_request *req,
                          const char **err)
{
    int n = jr_array_len(msgs);
    /* +1 headroom for a possible synthetic system turn. */
    req->messages = calloc((size_t)n + 1, sizeof(sg_message));
    if (!req->messages) { *err = "out of memory"; return -1; }

    const jr_value *first = jr_array_get(msgs, 0);
    const char *first_role = jr_as_string(jr_object_get(first, "role"));
    bool first_is_system = first_role && strcmp(first_role, "system") == 0;

    int out = 0;
    int start = 0;
    req->preamble_slot = -1;
    req->raw_messages = msgs;
    if (req->tools) {
        const char *sys = first_is_system
            ? message_text(req, jr_object_get(first, "content")) : NULL;
        /* Render the DEFAULT (Qwen/N2) preamble here; a post-route fixup
         * (chat_apply_tool_dialect) swaps it for the model's native dialect
         * once the resident model is known (e.g. MiniCPM5). Record the slot +
         * merged system content so the fixup can re-render. */
        req->sys_content = sys;
        const char *preamble = req_own(req, tc_render_tools_system(req->tools, sys));
        req->messages[out].role = "system";
        req->messages[out].content = preamble ? preamble : "";
        req->preamble_slot = out;
        out++;
        if (first_is_system) start = 1;
    }

    req->msg_start = start;
    for (int i = start; i < n; i++) {
        const jr_value *m = jr_array_get(msgs, i);
        const char *role = jr_as_string(jr_object_get(m, "role"));
        const char *content = message_text(req, jr_object_get(m, "content"));
        const jr_value *tool_calls = jr_object_get(m, "tool_calls");
        if (!role) role = "user";
        req->messages[out].role = role;
        /* An assistant turn may carry back the `reasoning_content` this serve
         * streamed for it.  Replaying it into the prompt is what lets the prefix
         * cache append-resume: the committed history's forced <think> opener is
         * otherwise unreproducible from `content` alone.  Borrowed from the
         * request DOM, exactly like `content`. */
        req->messages[out].reasoning =
            jr_as_string(jr_object_get(m, "reasoning_content"));
        if (strcmp(role, "assistant") == 0 && jr_kind(tool_calls) == JR_ARRAY &&
            jr_array_len(tool_calls) > 0) {
            /* Render in the DEFAULT (Qwen XML) dialect here — the model is not
             * yet routed. chat_apply_tool_dialect re-renders these slots in the
             * model's native dialect once known (e.g. ornith's JSON). */
            req->messages[out].content =
                combine_assistant_tool_calls(req, content, tool_calls,
                                             TC_DIALECT_QWEN_XML);
        } else {
            req->messages[out].content = content ? content : "";
        }
        out++;
    }
    req->n_messages = out;
    return 0;
}

/* Defined with its streaming sibling below; declared here because the default
 * arming happens during the parse. */
static int oai_shutdown_requested(void *ud);

/* `think_history` (core/think_history.h): what the session's KV must hold of
 * earlier think blocks. Absent → KEEP (today's ladder). A value this serve does
 * not know is ignored the same way, so a newer client still gets an answer, and
 * logged once per process so the mismatch is visible without flooding. */
static void parse_think_history(const jr_value *value, sg_gen_params *gen)
{
    static int unknown_logged;
    if (!value || pn_think_history_parse(jr_as_string(value),
                                         &gen->think_history))
        return;
    if (__atomic_exchange_n(&unknown_logged, 1, __ATOMIC_RELAXED) == 0)
        fprintf(stderr, "[serve] think_history: unknown value %s ignored "
                "(known: as_sent, drop_last); logged once\n",
                jr_as_string(value) ? jr_as_string(value) : "(not a string)");
}

/* Parse the request DOM into a chat_request. Returns 0 on success, -1 with a
 * human message in *err on failure. */
static int parse_chat_request(const jr_value *root, chat_request *req,
                              const char **err)
{
    memset(req, 0, sizeof(*req));
    if (jr_kind(root) != JR_OBJECT) { *err = "request body must be a JSON object"; return -1; }

    const jr_value *messages = jr_object_get(root, "messages");
    if (jr_kind(messages) != JR_ARRAY || jr_array_len(messages) == 0) {
        *err = "'messages' must be a non-empty array";
        return -1;
    }

    /* tool_choice: string form ("auto"|"none"|"required") OR the object form
     * {"type":"function","function":{"name":"X"}} pinning a specific function.
     * Default (absent) is "auto".  "none" disables tools entirely; "required"
     * and the object form drive Phase-5 forced-call decoding (compel a call). */
    const jr_value *tc_val = jr_object_get(root, "tool_choice");
    const char *tc_str = jr_as_string(tc_val);           /* NULL for the object form */
    sg_tool_choice_kind choice = SG_TOOL_CHOICE_AUTO;
    const char *forced_name = NULL;
    if (tc_str) {
        if (strcmp(tc_str, "none") == 0)          choice = SG_TOOL_CHOICE_NONE;
        else if (strcmp(tc_str, "required") == 0) choice = SG_TOOL_CHOICE_REQUIRED;
        else                                      choice = SG_TOOL_CHOICE_AUTO;  /* "auto" or unknown */
    } else if (jr_kind(tc_val) == JR_OBJECT) {
        const char *type = jr_as_string(jr_object_get(tc_val, "type"));
        const char *name = jr_as_string(
            jr_object_get(jr_object_get(tc_val, "function"), "name"));
        if (type && strcmp(type, "function") == 0 && name && *name) {
            choice = SG_TOOL_CHOICE_FUNCTION;
            /* OWN the forced name (BUG 2).  A borrowed pointer into the request's
             * jr_value DOM is a lifetime hazard: it aliases memory the caller
             * frees with jr_free(root), so any change to the free ordering (or a
             * path that retains gen.forced_name past the DOM) is a use-after-free.
             * Copy it onto the request (freed in chat_request_free) so the string
             * we keep is ours and never dangles.  On OOM, req_own frees the copy
             * and returns NULL → fall back to "required" (still forces a call). */
            forced_name = req_own(req, strdup(name));
            choice = forced_name ? SG_TOOL_CHOICE_FUNCTION
                                 : SG_TOOL_CHOICE_REQUIRED;
        }
    }

    bool tools_disabled = (choice == SG_TOOL_CHOICE_NONE);
    const jr_value *tools = jr_object_get(root, "tools");
    req->tools = (!tools_disabled && jr_kind(tools) == JR_ARRAY &&
                  jr_array_len(tools) > 0) ? tools : NULL;
    /* Arm the stream emitter's bare-"<function=" tool-call detection only when
     * this request actually carried tools (a wrapped "<tool_call>" is detected
     * regardless).  Avoids capturing a literal "<function=" in a normal answer. */
    req->gen.has_tools = (req->tools != NULL);
    /* Borrow the tools DOM for grammar-constrained decoding (consumed when the
     * grammar arms — sg_tool_grammar_arms; otherwise ignored). */
    req->gen.tools = req->tools;
    /* Keys required only on the first call of a reply that reasoned (borrowed
     * like `tools`; read by the grammar build, never rendered into the prompt). */
    req->gen.required_if_reasoned =
        req->tools ? jr_object_get(root, "required_if_reasoned") : NULL;
    /* Whole-value patterns on tool string fields (borrowed the same way; read by
     * the grammar build, never rendered).  A malformed field or a pattern that
     * does not compile is refused here, with its reason, rather than dropped. */
    const jr_value *field_patterns = jr_object_get(root, "field_patterns");
    char fp_err[256];
    if (field_patterns_validate(field_patterns, fp_err, sizeof(fp_err)) != 0) {
        *err = req_own(req, strdup(fp_err));
        if (!*err) *err = "'field_patterns' is malformed";
        return -1;
    }
    req->gen.field_patterns =
        (req->tools && jr_kind(field_patterns) == JR_OBJECT) ? field_patterns : NULL;
    /* Thread the resolved choice down to Phase-5 forced-call decoding.  With no
     * tools present, forcing is meaningless → fall back to AUTO.  forced_name
     * (when set) is request-OWNED (strdup'd above), so it stays valid for the
     * whole completion regardless of the jr_value DOM's lifetime. */
    req->gen.tool_choice = req->tools ? choice : SG_TOOL_CHOICE_AUTO;
    req->gen.forced_name =
        (req->gen.tool_choice == SG_TOOL_CHOICE_FUNCTION) ? forced_name : NULL;
    /* The DEFAULT is the shutdown check, not NULL: every turn — streaming or
     * not — has to be stoppable by a SIGTERM, or the drain that precedes the
     * accelerator teardown times out and the residencies are left held. The
     * streaming paths replace it with the wider check that also folds in client
     * disconnect. */
    req->gen.should_abort = oai_shutdown_requested;
    req->gen.abort_ud = NULL;

    if (build_messages(messages, req, err) != 0) return -1;

    /* temperature / top_p / top_k / presence_penalty: request value, else the
     * serve fallback here; re-resolved against the model's defaults once routed. */
    pn_model_sampling_from_json(root, &req->sampling);
    pn_model_sampling_apply(&req->sampling, NULL, &req->gen);
    /* Accept both max_tokens and the newer max_completion_tokens. */
    double mt = jr_as_number(jr_object_get(root, "max_tokens"),
              jr_as_number(jr_object_get(root, "max_completion_tokens"),
                           OAI_DEFAULT_MAX_TOKENS));
    /* Clamp: (int)mt on a huge/NaN value is UB, and downstream `max_tokens + 8`
     * (reasoning-budget headroom) could signed-overflow. 1<<20 tokens is far past
     * any real request. Non-positive falls to the serve default downstream. */
    if (!(mt >= 0.0)) mt = 0.0;                 /* NaN / negative → 0 (serve default) */
    if (mt > (double)(1 << 20)) mt = (double)(1 << 20);
    req->gen.max_tokens = (int)mt;
    req->gen.stop = parse_stops(jr_object_get(root, "stop"), &req->gen.n_stop);
    /* Per-request reasoning-budget override (accept either name).  When present it
     * wins over the per-model default; a negative value is clamped to 0 (disabled).
     * Absent → reasoning_budget_set stays false and the per-model default is used
     * at handle time.  gen.reasoning_budget defaults to 0 (memset above). */
    const jr_value *rb = jr_object_get(root, "reasoning_budget");
    if (jr_kind(rb) != JR_NUMBER)
        rb = jr_object_get(root, "max_reasoning_tokens");
    if (jr_kind(rb) == JR_NUMBER) {
        int v = (int)jr_as_number(rb, 0);
        req->gen.reasoning_budget = v > 0 ? v : 0;
        req->reasoning_budget_set = true;
    }
    /* Per-request thinking control.  `enable_thinking` is the Qwen3.x chat
     * template's own variable name (and llama.cpp's), and it is BINARY there —
     * the template's generation prompt is either "<think>\n" or the closed
     * "<think>\n\n</think>\n\n".  The template exposes no graded reasoning
     * level, so we do not invent one: a `reasoning_effort` whose middle values
     * all decoded to the same prompt would be a dial that does nothing.
     * Absent or non-boolean → SG_THINKING_DEFAULT (0 from the memset), which is
     * byte-identical to the pre-parameter serve. */
    const jr_value *et = jr_object_get(root, "enable_thinking");
    if (jr_kind(et) == JR_BOOL)
        req->gen.thinking = jr_as_bool(et, true) ? SG_THINKING_ON
                                                 : SG_THINKING_OFF;
    parse_think_history(jr_object_get(root, "think_history"), &req->gen);
    req->stream = jr_as_bool(jr_object_get(root, "stream"), false);
    /* OpenAI stream_options.include_usage: when set, the stream ends with an
     * extra chunk carrying `usage` (and empty `choices`), so a proxy can meter
     * streamed generations (which otherwise carry no token counts). */
    req->include_usage = jr_as_bool(
        jr_object_get(jr_object_get(root, "stream_options"), "include_usage"),
        false);
    req->model = jr_as_string(jr_object_get(root, "model"));
    /* PC0.1: a body session key (borrowed) overrides the X-Session-Id header the
     * caller passes in separately. `session_id`, `prompt_cache_key` (OpenAI) or
     * `promptCacheKey` (opencode) — see session_key.h. */
    req->session_id = pn_body_session_key(root);

    /* OpenAI sampling/decoding fields.  response_format resolves LAST because it
     * has to see the settled `tools` (tools win the constrained-decode slot). */
    parse_openai_sampling_fields(req, root);
    resolve_response_format(req, root);
    return 0;
}

static void chat_request_free(chat_request *req)
{
    for (int i = 0; i < req->n_owned; i++) free(req->owned[i]);
    free(req->owned);
    free(req->messages);
    free(req->gen.stop);
    free(req->logit_bias);
}

/* Parse the model's raw tool-call text into structured calls, if the request
 * allowed tools and the model emitted one. Sets *calls / *n (0 if none). */
/* A parsed call carries nothing usable: no members at all, or every value blank
 * once whitespace is discounted.  Cheap string test — the arguments are the JSON
 * this function just built, so "{}" is the whole of the degenerate case plus the
 * empty/whitespace-valued forms that survive it. */
static bool tc_args_degenerate(const char *args)
{
    if (!args) return true;
    const char *p = args;
    while (*p == ' ' || *p == '\t' || *p == '\n' || *p == '\r') p++;
    if (p[0] != '{') return false;              /* not an object — leave it alone */
    p++;
    while (*p == ' ' || *p == '\t' || *p == '\n' || *p == '\r') p++;
    if (*p == '}') return true;                 /* {} — no members */
    /* Any member whose value is an empty or whitespace-only string.  Walk the
     * body properly rather than skipping a char class: a naive skip of
     * ' ' \\ t n r would read {"path":"nnn"} as empty, because the escape letters
     * are also ordinary letters.  Only a BACKSLASH-introduced t/n/r/f is
     * whitespace; a bare one is content. */
    for (const char *q = strstr(args, "\":"); q; q = strstr(q + 1, "\":")) {
        const char *v = q + 2;
        while (*v == ' ' || *v == '\t' || *v == '\n' || *v == '\r') v++;
        if (*v != '"') continue;                /* non-string value — fine */
        v++;
        bool has_content = false;
        for (; *v && *v != '"'; v++) {
            if (*v == '\\' && v[1]) {
                char e = *++v;
                if (e != 't' && e != 'n' && e != 'r' && e != 'f')
                    has_content = true;          /* \" \\ \/ \uXXXX … are content */
                continue;
            }
            if (*v != ' ' && *v != '\t' && *v != '\n' && *v != '\r')
                has_content = true;
        }
        if (!has_content) return true;
    }
    return false;
}

/* Parse the model's raw tool-call text into structured calls, if the request
 * allowed tools and the model emitted one. Sets *calls / *n (0 if none).
 *
 * DIAGNOSTIC: when parsing yields a call with nothing usable in it, log the RAW
 * text that produced it.  The client only ever records the parsed result, so a
 * `read {}` in the client's history is a dead end — there is no way to tell whether
 * the model emitted an empty value, a mangled one that failed to extract, or a
 * whole tool call in the wrong frame.  This is the only place both halves exist
 * at once.  It fires only on the degenerate case, so it costs nothing on healthy
 * traffic and cannot flood the log the way a per-step trace would.
 *
 * `err` (nullable) carries back the reason a call the model STARTED is absent
 * from *calls — the abandoned shape and the other structurally-wrong ones. The
 * caller puts it on the wire so the client can tell them from the empty-argument
 * case, which still yields a call.
 *
 * NO-CALL CATCH-ALL. The emitter only fills `tool_raw` once it has seen a
 * tool-call opener, so raw text with ZERO calls out of it means a frame was
 * emitted and nothing survived parsing. That is the serve-side reading of
 * qwen-code's MALFORMED_TOOL_CALL "finish_reason=tool_calls with no completed
 * calls" (converter.ts:1699-1711) — our own finish_reason is derived from the
 * call count, so it can never contradict it, but the diverted-and-lost text can.
 * Raised only when the parser named no more specific reason.
 *
 * TRUNCATION GATE. This is also the only place the DECODING GRAMMAR's end-of-turn
 * verdict (res->tool_grammar_verdict) meets the calls parsed out of the text, and
 * a call the grammar never closed is refused here rather than handed over. The
 * text alone cannot show it: a truncated call is a legal prefix of a good one, so
 * it parses cleanly into a call with a required argument silently absent. */
/* THE DEFECT THE WHOLE DIAGNOSTIC LAYER ABOVE CANNOT SEE.
 *
 * Everything in collect_tool_calls hangs off `res->tool_raw`, and the emitter
 * only fills that once it recognises an OPENER ("<tool_call>", a plausible
 * "<function=").  A model that invents a NEAR-MISS frame therefore matches no
 * opener, is never diverted, and lands in `content` as prose — so the turn ends
 * with zero calls, finish_reason "stop", and not one line in the log.  Measured
 * on qwen36-bf16 (BF16 full weights, 7 tools, 558-token prompt, grammar off):
 *
 *     <tool_calls>\n<tool>\n<tool_name>bash</tool_name>\n<tool>
 *     <parameter_name>command</parameter_name>
 *     <parameter_value>sed -n '175,200p' …</parameter_value>\n</tool>\n</tool_call>
 *
 * `no_call_parsed` does NOT cover this: that code fires when a frame WAS
 * diverted and nothing parsed out of it.  Here nothing was ever diverted.  The
 * two silences are different and the operator needs to tell them apart.
 *
 * The test is deliberately crude — does the answer text carry a tool-call tag at
 * all — because it only ever runs on a turn that produced NO call, and a real
 * opener in `content` is impossible by construction (the emitter would have
 * taken it).  What is left is exactly the invented dialects. */
static void note_content_tool_syntax(const char *content)
{
    static const char *const MARKERS[] = {
        "<tool_call", "</tool_call", "<function", "</function",
        "<parameter", "</parameter", "<tool_name", "<tool>",
    };
    if (!content || !*content) return;
    for (size_t i = 0; i < sizeof(MARKERS) / sizeof(*MARKERS); i++) {
        const char *hit = strstr(content, MARKERS[i]);
        if (!hit) continue;
        size_t len = strlen(content);
        fprintf(stderr, "[tool_syntax_leak] the answer carries tool-call syntax "
                "that matched no opener, so it was never diverted and no call "
                "was parsed — marker='%s' content_len=%zu content=<<<%.600s%s>>>\n",
                MARKERS[i], len, content, len > 600 ? "…[truncated]" : "");
        return;
    }
}

static void collect_tool_calls(const chat_request *req, const sg_result *res,
                               tc_tool_call **calls, int *n, tc_parse_error *err)
{
    *calls = NULL;
    *n = 0;
    if (err) *err = (tc_parse_error){0};
    if (req->tools && !(res->tool_raw && *res->tool_raw))
        note_content_tool_syntax(res->content);
    if (req->tools && res->tool_raw && *res->tool_raw) {
        tc_parse_repair repair = {0};
        tc_parse_tool_calls_diagnosed(res->tool_raw, req->tools, req->gen.dialect,
                                      calls, n, err, &repair);
        size_t rawlen = strlen(res->tool_raw);
        /* Every repair is logged, and only ever after one fired, so healthy
         * traffic costs nothing and the count is a straight grep. `added` is the
         * whole story of what was assumed: a '"' in it means a string value was
         * closed early and may be short. */
        if (repair.count > 0)
            fprintf(stderr,
                    "[tool_json_repair] repairs=%d added=<<<%s>>> "
                    "repaired=<<<%.600s>>> raw_len=%zu raw=<<<%.600s%s>>>\n",
                    repair.count, repair.added ? repair.added : "",
                    repair.repaired_json ? repair.repaired_json : "",
                    rawlen, res->tool_raw, rawlen > 600 ? "…[truncated]" : "");
        tc_parse_repair_free(&repair);
        /* TRUNCATION GATE.  The grammar entered a call and the turn ended before
         * it closed, so whatever the parser just built is the wreckage of a call
         * the model never finished — and it does not LOOK like wreckage: a
         * truncated emission is a legal prefix, so it parses into a named call
         * with a required argument quietly absent (`read {}`, `write` with no
         * `content`).  The serve knew it was incomplete and emitted it anyway;
         * that is the defect.  Drop the calls and say why, on the same channel as
         * every other "a call the model started is deliberately not here" reason.
         *
         * Placed BEFORE the no_call_parsed catch-all so the specific reason wins,
         * and it does not overwrite a reason the parser already named (abandoned,
         * leaked tail, …) — those are the same event seen closer up.
         *
         * Nothing is masked to keep the model generating.  A model that has
         * stopped, forced to continue, produces a corrupted file instead of a
         * clean failure; the client retries on this code instead. */
        if (res->tool_grammar_verdict == SG_TOOL_GRAMMAR_INCOMPLETE) {
            char msg[256];
            /* Copied, not borrowed: the calls array owns `name` and is freed
             * immediately below, before `err` is filled in. */
            char *fname = (*n > 0 && (*calls)[0].name)
                              ? strdup((*calls)[0].name) : NULL;
            snprintf(msg, sizeof(msg),
                     "%s: call incomplete — the decoding grammar was still inside "
                     "the call when the turn ended", fname ? fname : "tool call");
            tc_tool_calls_free(*calls, *n);
            *calls = NULL;
            *n = 0;
            if (err && err->kind == TC_PARSE_OK) {
                err->kind = TC_PARSE_CALL_INCOMPLETE;
                err->function_name = fname;   /* ownership moves to `err` */
                err->message = strdup(msg);
            } else {
                free(fname);
            }
        }
        if (err && err->kind == TC_PARSE_OK && *n == 0) {
            err->kind = TC_PARSE_NO_CALL_PARSED;
            err->message = strdup("a tool-call frame was emitted but no call "
                                  "could be parsed out of it");
        }
        if (err && err->kind != TC_PARSE_OK) {
            flockfile(stderr);   /* one line, even with other requests logging */
            fprintf(stderr, "[tool_call_dropped] code=%s %s ",
                    tc_parse_failure_code(err->kind),
                    err->message ? err->message : "(no message)");
            tool_raw_excerpt_print(stderr, res->tool_raw, TOOL_RAW_EXCERPT_HEAD,
                                   TOOL_RAW_EXCERPT_TAIL);
            fputc('\n', stderr);
            funlockfile(stderr);
        }
        for (int i = 0; i < *n; i++) {
            if (!tc_args_degenerate((*calls)[i].arguments))
                continue;
            fprintf(stderr,
                    "[tool_degenerate] tool=%s args=%s raw_len=%zu raw=<<<%.600s%s>>>\n",
                    (*calls)[i].name ? (*calls)[i].name : "?",
                    (*calls)[i].arguments ? (*calls)[i].arguments : "(null)",
                    rawlen, res->tool_raw, rawlen > 600 ? "…[truncated]" : "");
        }
    }
}

/* Re-render every assistant-history tool_calls turn in `dialect` (the initial
 * pass in build_messages rendered them as Qwen XML, before the model was known).
 * Walks raw_messages in step with the flattened messages[] so each combined
 * "text + rendered calls" slot is replaced. Only needed for a non-XML dialect;
 * a no-op run cost is avoided by the caller. */
static void rerender_assistant_history(chat_request *req, tc_dialect dialect)
{
    int n = jr_array_len(req->raw_messages);
    int out = req->preamble_slot >= 0 ? 1 : 0;   /* slot 0 is the tools preamble */
    for (int i = req->msg_start; i < n; i++, out++) {
        const jr_value *m = jr_array_get(req->raw_messages, i);
        const char *role = jr_as_string(jr_object_get(m, "role"));
        const jr_value *tool_calls = jr_object_get(m, "tool_calls");
        if (!role || strcmp(role, "assistant") != 0) continue;
        if (jr_kind(tool_calls) != JR_ARRAY || jr_array_len(tool_calls) == 0)
            continue;
        const char *content = message_text(req, jr_object_get(m, "content"));
        req->messages[out].content =
            combine_assistant_tool_calls(req, content, tool_calls, dialect);
    }
}

/* Post-route dialect resolver + fixup. The tools preamble + assistant history
 * were rendered in the default (Qwen/N2) dialect before the model was known.
 * Once routed, resolve the resident model's dialect (tc_dialect_for_model) and,
 * when it differs from the XML default, install the model's native prompt
 * preamble and re-render its assistant history. Stores the resolved dialect on
 * req->gen so the emitter, grammar setup, and parser all speak it. No-op for a
 * tools-less request beyond storing the dialect.
 *
 * Phase 2: ornith's `tool_choice:"required"`/named-function is NO LONGER
 * downgraded to AUTO — the schema-constraining JSON grammar (tool_grammar_json,
 * built in sg_tg_setup for TC_DIALECT_ORNITH_JSON) honors `required` natively,
 * so ornith runs FORCED exactly like saber does under its XML grammar. */
static void chat_apply_tool_dialect(chat_request *req, const pn_model *model,
                                    const char *model_name)
{
    tc_dialect dialect =
        tc_dialect_for_model(model_name, mr_model_is_minicpm(model));
    req->gen.dialect = dialect;
    if (!req->tools) return;

    if (req->preamble_slot < 0) return;   /* no tools preamble to swap */
    char *preamble = NULL;
    if (dialect == TC_DIALECT_MINICPM_XML)
        preamble = tc_render_tools_system_minicpm(req->tools, req->sys_content);
    else if (dialect == TC_DIALECT_ORNITH_JSON)
        preamble = tc_render_tools_system_ornith(req->tools, req->sys_content);
    if (preamble) {
        const char *owned = req_own(req, preamble);
        if (owned) req->messages[req->preamble_slot].content = owned;
    }
    /* Assistant history was rendered XML in build_messages; re-render for a
     * non-XML dialect so a multi-turn loop feeds the model its own dialect. */
    if (dialect == TC_DIALECT_ORNITH_JSON)
        rerender_assistant_history(req, dialect);
}

/* ============================================================
 * GET /v1/models
 * ============================================================ */

/* `think_history` — the request values this model honours
 * (core/think_history.h). as_sent on every path: the batched ladder resumes at
 * the system prefix, the single-user cache and the TT card checkpoint stop at
 * the first think block, and a request without a session is always cold.
 * drop_last where a turn-start state is kept: the batched scheduler's turn
 * checkpoints (CUDA), the single-user turn snapshot, or the TT card's band
 * checkpoint. Always present, so a client
 * never has to guess from absence. */
static void write_think_history_props(json_writer *jw, const mr_info *info)
{
    jw_raw(jw, info->think_history_drop_last
                   ? ",\"think_history\":[\"as_sent\",\"drop_last\"]"
                   : ",\"think_history\":[\"as_sent\"]");
}

void oai_handle_models(int client_fd, mr_registry *reg)
{
    mr_info info[MR_MAX_MODELS];
    int n = mr_snapshot(reg, info, MR_MAX_MODELS);

    json_writer jw;
    if (!jw_init(&jw)) { hs_send_error(client_fd, 500, "server_error", "oom"); return; }
    /* `backend` / `build` — the compiled accelerator backend and the commit,
     * where a CLIENT can read them. Additive top-level keys: an OpenAI-shaped
     * client walks `data` and never sees them, and anyone measuring this serve
     * can now tell a CUDA answer from a CPU answer without `ldd` on a remote
     * box. The completion text is byte-identical either way — which is exactly
     * why nothing downstream could ever catch it. */
    jw_raw(&jw, "{\"object\":\"list\",\"backend\":");
    jw_string(&jw, pn_backend_token());
    jw_raw(&jw, ",\"build\":");
    jw_string(&jw, pn_backend_build_commit());
    jw_raw(&jw, ",\"data\":[");
    int emitted = 0;
    for (int i = 0; i < n; i++) {
        /* Dedupe by name: a model placed on several cards (multi-card LB)
         * appears ONCE — clients see one logical model id, not per-card copies. */
        bool seen = false;
        for (int j = 0; j < i; j++)
            if (strcmp(info[j].name, info[i].name) == 0) { seen = true; break; }
        if (seen)
            continue;
        if (emitted++) jw_raw(&jw, ",");
        jw_raw(&jw, "{\"id\":");
        jw_string(&jw, info[i].name);
        /* `max_concurrent` is the SERVED concurrency (bs_get_load's capacity),
         * not the configured cap. A client uses this number to decide how many
         * requests to have in flight, so advertising a cap the serve cannot
         * honour is worse than advertising nothing: the incident serve promised
         * 2 while its scheduler was NULL (no CUDA compiled in), so every second
         * chat hit the per-model trylock and came back 429. The configured cap
         * is still visible to an OPERATOR on /admin/models — same split as
         * context_window vs context_window_configured, for the same reason. */
        jw_printf(&jw, ",\"object\":\"model\",\"created\":0,\"owned_by\":\"pantheon\""
                  ",\"max_sessions\":%d,\"max_concurrent\":%d",
                  info[i].max_sessions, info[i].served_concurrency);
        /* `context_window` — the per-session window a client may actually use,
         * in tokens. EFFECTIVE, not the operator's --ctx-cap: it is the
         * post-ladder number, the same one this model's completions report as
         * `kv_cap`, so the client's plan and the serve's enforcement agree.
         *
         * Additive, and omitted when unknown (0): a plain OpenAI-compatible
         * client ignores the extra key, and a client that reads it never has to
         * special-case a zero. The name matches the client's own `context_window` —
         * see docs/reference/http-api.md. */
        if (info[i].context_window > 0)
            jw_printf(&jw, ",\"context_window\":%d", info[i].context_window);
        /* `tool_choice_required_with_thinking` — a client may send
         * tool_choice "required" with thinking ON and rely on it: the think
         * block stays free and the turn ends in a parsed tool call (the
         * think-gated grammar, sg_tool_choice_required_with_thinking).  False
         * for a dialect no grammar models (MiniCPM).  Always present,
         * true/false, so a client never has to guess from absence. */
        tc_dialect dialect =
            tc_dialect_for_model(info[i].name, info[i].minicpm_tool_dialect);
        jw_printf(&jw, ",\"tool_choice_required_with_thinking\":%s",
                  sg_tool_choice_required_with_thinking(dialect) ? "true" : "false");
        write_think_history_props(&jw, &info[i]);
        /* `graphs` — CUDA-graph residency, same additive/omit-when-absent rule.
         * A lost graph costs speed only (output stays correct), so this is the
         * one place a client or a poller can see that a model is running
         * degraded. `ineligible` is NOT a fault: a model whose MoE experts are
         * not Q3_K captures nothing by design. */
        gr_write_summary(&jw, &info[i].graphs);
        /* `attention` — same rule again, and the sharper of the two: a layer
         * on the host-softmax fallback returns identical tokens an order of
         * magnitude slower. `host` > 0 means this model is serving degraded. */
        ar_write_summary(&jw, &info[i].attn);
        jw_raw(&jw, "}");
    }
    jw_raw(&jw, "]}");
    hs_send_json(client_fd, 200, jw.data);
    jw_free(&jw);
}

/* ============================================================
 * POST /v1/chat/completions — non-streaming
 * ============================================================ */

/* ============================================================
 * OpenAI `logprobs` on the wire
 * ============================================================ */

/* Emit one token's entry: {"token":…,"logprob":…,"bytes":[…],"top_logprobs":[…]}.
 * `bytes` is the token's raw UTF-8, which is how a client reconstructs a token
 * that is not valid UTF-8 on its own (a split multi-byte character). */
static void write_logprob_entry(json_writer *jw, const char *token,
                                float logprob, bool with_top,
                                const sg_token_logprob *lp)
{
    jw_raw(jw, "{\"token\":");
    jw_string(jw, token);
    /* -inf/NaN are not valid JSON numbers; a token the model gave zero mass is
     * reported at the float floor instead of poisoning the whole body.
     * isnan(), not `logprob != logprob`: this TU builds with -ffast-math and
     * the self-compare idiom folds to constant false (proven), so it never
     * caught a NaN logprob and would have written literal `nan` into the
     * JSON body. */
    if (isnan(logprob))                jw_raw(jw, ",\"logprob\":0");
    else if (logprob < -1.0e30f)       jw_raw(jw, ",\"logprob\":-9999");
    else                               jw_printf(jw, ",\"logprob\":%.6f", logprob);
    jw_raw(jw, ",\"bytes\":[");
    for (const unsigned char *p = (const unsigned char *)(token ? token : "");
         *p; p++) {
        if (p != (const unsigned char *)token) jw_raw(jw, ",");
        jw_printf(jw, "%u", (unsigned)*p);
    }
    jw_raw(jw, "]");
    if (with_top && lp) {
        jw_raw(jw, ",\"top_logprobs\":[");
        for (int i = 0; i < lp->n_top; i++) {
            if (i) jw_raw(jw, ",");
            write_logprob_entry(jw, lp->top_token[i], lp->top_logprob[i],
                                false, NULL);
        }
        jw_raw(jw, "]");
    }
    jw_raw(jw, "}");
}

/* Append `"logprobs":{"content":[…]}` — the per-choice OpenAI object.  `entries`
 * is an array of BORROWED per-token records. */
static void write_logprobs_object(json_writer *jw,
                                  const sg_token_logprob *const *entries,
                                  int n)
{
    jw_raw(jw, ",\"logprobs\":{\"content\":[");
    for (int i = 0; i < n; i++) {
        if (i) jw_raw(jw, ",");
        write_logprob_entry(jw, entries[i]->token, entries[i]->logprob, true,
                            entries[i]);
    }
    jw_raw(jw, "]}");
}

/* Non-streaming variant: the result owns a flat array rather than a pointer
 * list, so the whole completion's logprobs are emitted in one go. */
static void write_logprobs_object_flat(json_writer *jw,
                                       const sg_token_logprob *entries, int n)
{
    jw_raw(jw, ",\"logprobs\":{\"content\":[");
    for (int i = 0; i < n; i++) {
        if (i) jw_raw(jw, ",");
        write_logprob_entry(jw, entries[i].token, entries[i].logprob, true,
                            &entries[i]);
    }
    jw_raw(jw, "]}");
}

/* Append a "tool_calls":[…] array (the OpenAI function-call shape) to `jw`. */
static void write_tool_calls_array(json_writer *jw, const tc_tool_call *calls,
                                   int n)
{
    jw_raw(jw, "\"tool_calls\":[");
    for (int i = 0; i < n; i++) {
        if (i) jw_raw(jw, ",");
        jw_raw(jw, "{\"id\":");
        jw_string(jw, calls[i].id);
        jw_raw(jw, ",\"type\":\"function\",\"function\":{\"name\":");
        jw_string(jw, calls[i].name);
        jw_raw(jw, ",\"arguments\":");
        jw_string(jw, calls[i].arguments);   /* stringified JSON per OpenAI */
        jw_raw(jw, "}}");
    }
    jw_raw(jw, "]");
}

/* Append a `tool_call_error` object: the structured reason a call the model
 * STARTED is not in `tool_calls`.  Its presence is the whole signal — a client
 * seeing it knows the model truncated mid-call, which is a different event from
 * a call that arrived with an empty argument (that one still comes through as a
 * tool_call, with `{}` arguments) and from a turn that never called a tool at
 * all (neither field present). */
static void write_tool_call_error(json_writer *jw, const tc_parse_error *err)
{
    jw_raw(jw, "\"tool_call_error\":{\"code\":");
    jw_string(jw, tc_parse_failure_code(err->kind));
    if (err->function_name) {
        jw_raw(jw, ",\"function\":");
        jw_string(jw, err->function_name);
    }
    if (err->parameter_name) {
        jw_raw(jw, ",\"parameter\":");
        jw_string(jw, err->parameter_name);
    }
    if (err->message) {
        jw_raw(jw, ",\"message\":");
        jw_string(jw, err->message);
    }
    jw_raw(jw, "}");
}

/* Did the repetition-loop detector end this turn?  The finish_reason string is
 * the whole signal (serve_generation.h), and both response shapes have to read
 * it the same way — the last bug in this file was a field that landed on one
 * transport and not the other. */
static bool finish_is_repetition_loop(const sg_result *res)
{
    return res && res->finish_reason &&
           strcmp(res->finish_reason, SG_FINISH_REPETITION_LOOP) == 0;
}

/* One completed choice: the generation result plus its parsed tool calls.  `n`
 * completions produce an array of these, and n == 1 is simply an array of one —
 * the single-completion path is not special-cased anywhere below. */
typedef struct {
    sg_result     res;
    tc_tool_call *calls;
    int           n_calls;
    tc_parse_error tool_error;   /* why a started call is missing from `calls` */
} chat_choice;

/* Append one `choices[]` element. */
static void write_choice(json_writer *jw, int index, const chat_choice *ch)
{
    const sg_result *res = &ch->res;
    jw_printf(jw, "{\"index\":%d,\"message\":{\"role\":\"assistant\",\"content\":",
              index);
    /* A pure tool call carries content:null; any text before the call is kept. */
    if (ch->n_calls > 0 && (!res->content || !res->content[0]))
        jw_raw(jw, "null");
    else
        jw_string(jw, res->content);
    if (res->reasoning && res->reasoning[0]) {
        jw_raw(jw, ",\"reasoning_content\":");
        jw_string(jw, res->reasoning);
    }
    if (ch->n_calls > 0) {
        jw_raw(jw, ",");
        write_tool_calls_array(jw, ch->calls, ch->n_calls);
    }
    jw_raw(jw, "}");
    /* Per the OpenAI shape `logprobs` is a member of the CHOICE, beside
     * `message` — not inside it. */
    if (res->logprobs && res->n_logprobs > 0)
        write_logprobs_object_flat(jw, res->logprobs, res->n_logprobs);
    jw_raw(jw, ",\"finish_reason\":");
    /* A repetition collapse OUTRANKS "tool_calls".  If the turn degenerated, the
     * caller has to learn that even when a call was parsed out of the wreckage —
     * reporting "tool_calls" would hand back a call to execute and hide the one
     * fact that explains the turn.  Every other case keeps the pre-existing
     * precedence (a parsed call wins over the generation's own stop reason). */
    jw_string(jw, (ch->n_calls > 0 && !finish_is_repetition_loop(res))
                      ? "tool_calls" : res->finish_reason);
    /* Beside `finish_reason`, in the CHOICE: it explains this completion's
     * missing call, and with `n` > 1 each completion answers for itself. */
    if (ch->tool_error.kind != TC_PARSE_OK) {
        jw_raw(jw, ",");
        write_tool_call_error(jw, &ch->tool_error);
    }
    jw_raw(jw, "}");
}

static void send_completion_json(int client_fd, const char *id, long created,
                                 const char *model_id,
                                 const chat_choice *choices, int n_choices,
                                 const char *response_format_status,
                                 const char *n_status)
{
    if (n_choices <= 0) {
        hs_send_error(client_fd, 500, "server_error", "no completion produced");
        return;
    }
    json_writer jw;
    if (!jw_init(&jw)) { hs_send_error(client_fd, 500, "server_error", "oom"); return; }
    jw_raw(&jw, "{\"id\":");
    jw_string(&jw, id);
    jw_printf(&jw, ",\"object\":\"chat.completion\",\"created\":%ld,\"model\":", created);
    jw_string(&jw, model_id);
    jw_raw(&jw, ",\"choices\":[");
    for (int i = 0; i < n_choices; i++) {
        if (i) jw_raw(&jw, ",");
        write_choice(&jw, i, &choices[i]);
    }
    jw_raw(&jw, "]");

    /* Usage across the whole request: the prompt was paid for ONCE (every choice
     * decoded from the same prefill), the completions add up. */
    const sg_result *first = &choices[0].res;
    int completion_total = 0;
    for (int i = 0; i < n_choices; i++)
        completion_total += choices[i].res.completion_tokens;
    /* `session_conflict` is written INSIDE the usage object (hence the %s before
     * its closing brace, not a jw_raw after it): it tells the client it is
     * running two conversations under one session id, so the serve can resume
     * neither and both re-prefill in full every turn. Only the client can fix
     * that, so the signal has to be on the wire and not just in the serve log —
     * and it has to be where the client looks, which is `usage`, the object it
     * already reads for kv_used/kv_cap. Appending it after the brace instead put
     * it beside `usage` rather than in it, and no client ever saw it. */
    /* `prompt_tokens_details.cached_tokens` is OpenAI's own spelling for "of
     * those prompt tokens, this many were reused, not computed". UNCONDITIONAL
     * (0 on a cold turn, never omitted): a client cannot tell a serve that
     * reported no reuse from one that does not report reuse at all, so an absent
     * field forces it back to prompt_tokens/TTFT — a prefill rate that rises
     * with every cache hit. Absent and zero are different answers. */
    /* `resume` + `prefill_tokens`: the rung this turn resumed on and what it
     * prefilled, so a client metering a think_history request (or any turn)
     * does not need the serve log. Unconditional, like cached_tokens. */
    jw_printf(&jw, ",\"usage\":{\"prompt_tokens\":%d,\"completion_tokens\":%d,"
                   "\"total_tokens\":%d,"
                   "\"prompt_tokens_details\":{\"cached_tokens\":%d},"
                   "\"kv_used\":%d,\"kv_cap\":%d,"
                   "\"resume\":\"%s\",\"prefill_tokens\":%d%s}",
              first->prompt_tokens, completion_total,
              first->prompt_tokens + completion_total,
              first->cached_prompt_tokens,
              first->kv_used_tokens, first->kv_cap_tokens,
              pn_resume_kind_name(first->resume_kind), first->prefill_tokens,
              first->session_conflict ? ",\"session_conflict\":true" : "");
    /* Tell the client when admission clamped its context (MemoryAware §6.2
     * rung 3): the oldest prompt tokens were dropped to the fitting window, so
     * the model did not see the full conversation. Top-level, mirroring the
     * streaming path, where it rides the closing chunk and not `usage`. */
    if (first->context_capped)
        jw_raw(&jw, ",\"context_capped\":true");
    /* What actually happened to the request's `response_format`.  Top-level and
     * unconditional (not inside `usage`, which a streaming client only sees when
     * it opted into stream_options.include_usage) so the answer is on the wire
     * for EVERY client that sent the field — the point of the field is a
     * guarantee, and a guarantee that silently did not apply is the bug. */
    if (response_format_status) {
        jw_raw(&jw, ",\"response_format_status\":");
        jw_string(&jw, response_format_status);
    }
    /* Same rule for `n`: if the request asked for more completions than were
     * delivered, say so rather than letting the client count choices and guess. */
    if (n_status) {
        jw_raw(&jw, ",\"n_status\":");
        jw_string(&jw, n_status);
    }
    jw_raw(&jw, "}");
    hs_send_json(client_fd, 200, jw.data);
    jw_free(&jw);
}

/* Free everything a choice owns. */
static void chat_choice_free(chat_choice *ch)
{
    tc_tool_calls_free(ch->calls, ch->n_calls);
    ch->calls = NULL;
    ch->n_calls = 0;
    tc_parse_error_free(&ch->tool_error);
    sg_result_free(&ch->res);
}

/* ============================================================
 * POST /v1/chat/completions — streaming (SSE)
 * ============================================================ */

typedef struct {
    int         client_fd;
    const char *id;
    long        created;
    const char *model_id;
    bool        role_sent;
    bool        context_capped;   /* admission clamped the prompt window —
                                   * surfaced on the closing chunk(s) */
    const char *response_format_status;  /* set just before the CLOSING chunk (so
                                   * it rides that chunk and not all of them),
                                   * mirroring context_capped.  NULL = the request
                                   * carried no response_format. */
    const tc_parse_error *tool_error;  /* why a started call is absent from the
                                   * tool-call deltas.  Borrowed, set just before
                                   * the CLOSING chunk so it rides that chunk —
                                   * same placement rule as the two above.  NULL
                                   * (or TC_PARSE_OK) = nothing was abandoned. */
    int         choice_index;     /* OpenAI `n`: which completion these chunks
                                   * belong to.  Always 0 for the ordinary
                                   * single-completion request, so the wire shape
                                   * is unchanged for every client today. */
    tcs_streamer *tool_stream;    /* this completion's tool-call streamer (set by
                                   * the run paths; tool_call_stream.h) */
    double      last_write_s;     /* monotonic time of the last SSE write */
    double      keepalive_s;      /* silence that earns a keepalive comment while
                                   * a tool call is being written; 0 = default */
    bool        client_gone;      /* a write to the stream failed (closed, reset,
                                   * or not read within the send timeout): the
                                   * turn's should_abort stops it */
} stream_ctx;

/* Longest silence on the wire while the model writes a tool call.  Whatever
 * cannot be streamed yet (tool_call_stream.h: a held object, an XML-dialect
 * call) is covered by an SSE comment this often, far inside any client stall
 * watchdog (typically tens of seconds). */
#define OAI_TOOL_KEEPALIVE_S 5.0

static double oai_now_s(void)
{
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (double)ts.tv_sec + (double)ts.tv_nsec / 1e9;
}

/* Emit one chat.completion.chunk with the given delta body (already the inner
 * JSON of "delta", e.g. "\"content\":\"hi\""). */
static void send_chunk(stream_ctx *sc, const char *delta_json,
                       const char *finish_reason,
                       const sg_token_logprob *const *logprobs, int n_logprobs)
{
    json_writer jw;
    if (!jw_init(&jw)) return;
    jw_raw(&jw, "{\"id\":");
    jw_string(&jw, sc->id);
    jw_printf(&jw, ",\"object\":\"chat.completion.chunk\",\"created\":%ld,\"model\":",
              sc->created);
    jw_string(&jw, sc->model_id);
    jw_printf(&jw, ",\"choices\":[{\"index\":%d,\"delta\":{", sc->choice_index);
    jw_raw(&jw, delta_json ? delta_json : "");
    jw_raw(&jw, "}");
    /* Same placement as the non-streaming body: a member of the CHOICE, beside
     * `delta`.  Carries only the tokens whose bytes are in THIS delta — the
     * emitter holds a partial tail back, so a client cannot pair them by
     * counting deltas. */
    if (logprobs && n_logprobs > 0)
        write_logprobs_object(&jw, logprobs, n_logprobs);
    jw_raw(&jw, ",\"finish_reason\":");
    if (finish_reason) jw_string(&jw, finish_reason);
    else jw_raw(&jw, "null");
    /* Same member of the same object as the non-streaming body's copy, so a
     * client reads the abandoned call the same way on both transports. */
    if (sc->tool_error && sc->tool_error->kind != TC_PARSE_OK) {
        jw_raw(&jw, ",");
        write_tool_call_error(&jw, sc->tool_error);
    }
    jw_raw(&jw, "}]");
    /* MemoryAware §6.2 rung 3: tell the streaming client its context was
     * clamped (set only once generation finished, so it rides the closing
     * chunk(s), mirroring the non-streaming `context_capped` field). */
    if (sc->context_capped)
        jw_raw(&jw, ",\"context_capped\":true");
    /* Same signal as the non-streaming body's `response_format_status`, on the
     * same terms: top-level, so a streaming client sees it whether or not it
     * asked for a usage chunk. */
    if (sc->response_format_status) {
        jw_raw(&jw, ",\"response_format_status\":");
        jw_string(&jw, sc->response_format_status);
    }
    jw_raw(&jw, "}");
    if (hs_send_sse(sc->client_fd, jw.data) != 0)
        sc->client_gone = true;
    sc->last_write_s = oai_now_s();
    jw_free(&jw);
}

/* Emit the OpenAI terminal usage chunk: an extra chat.completion.chunk with an
 * empty `choices` array and a `usage` object. Sent last (just before [DONE])
 * only when the request set stream_options.include_usage, so a metering proxy
 * can read token counts off the stream. Mirrors the non-streaming `usage`.
 * KV Phase 1: also carries kv_used/kv_cap (the session's true KV occupancy vs
 * its per-session cap) so the client can trigger compaction at ~80% BEFORE the
 * reject-backstop fires. kv_used reflects committed KV — reasoning tokens the
 * client never counts are in there, so this beats the client's own estimate.
 * `cached_tokens` rides here too, and this is the emission that matters for it:
 * benchmarks stream, so a fix to the non-streaming body alone would leave every
 * measured prefill rate inflated. */
static void send_usage_chunk(stream_ctx *sc, const sg_result *res)
{
    json_writer jw;
    if (!jw_init(&jw)) return;
    jw_raw(&jw, "{\"id\":");
    jw_string(&jw, sc->id);
    jw_printf(&jw, ",\"object\":\"chat.completion.chunk\",\"created\":%ld,\"model\":",
              sc->created);
    jw_string(&jw, sc->model_id);
    jw_printf(&jw, ",\"choices\":[],\"usage\":{\"prompt_tokens\":%d,"
                   "\"completion_tokens\":%d,\"total_tokens\":%d,"
                   "\"prompt_tokens_details\":{\"cached_tokens\":%d},"
                   "\"kv_used\":%d,\"kv_cap\":%d,"
                   "\"resume\":\"%s\",\"prefill_tokens\":%d%s}}",
              res->prompt_tokens, res->completion_tokens,
              res->prompt_tokens + res->completion_tokens,
              res->cached_prompt_tokens, res->kv_used_tokens,
              res->kv_cap_tokens, pn_resume_kind_name(res->resume_kind),
              res->prefill_tokens,
              /* Same signal as the non-streaming body's "session_conflict":
               * two conversations are sharing one session id, so neither can
               * resume. Rides `usage` because that is the object a streaming
               * client already reads for kv_used/kv_cap. */
              res->session_conflict ? ",\"session_conflict\":true" : "");
    if (hs_send_sse(sc->client_fd, jw.data) != 0)
        sc->client_gone = true;
    jw_free(&jw);
}

/* KV Phase 1: the machine-readable "context over budget" signal. The prompt
 * exceeds the model's per-session KV cap; the serve refused rather than silently
 * drop-oldest. The client reads type == "context_over_budget" + the
 * kv_used/kv_cap numbers, compacts its history, and retries. 422 Unprocessable
 * (not 503/429) tells the client this is NOT a transient blip to blind-retry —
 * it must shed context first. */
static void send_context_over_budget_http(int client_fd, int kv_used,
                                           int kv_cap)
{
    char msg[224];
    snprintf(msg, sizeof(msg),
             "context over budget: %d tokens exceeds the %d-token per-session "
             "cap; compact the conversation and retry (kv_used=%d kv_cap=%d)",
             kv_used, kv_cap, kv_used, kv_cap);
    hs_send_error(client_fd, 422, "context_over_budget", msg);
}

/* SSE variant: the stream is already open, so the reject must ride the stream as
 * a terminal error event carrying the same machine-readable type + kv numbers,
 * then the stream ends. */
static void send_context_over_budget_sse(int client_fd, int kv_used, int kv_cap)
{
    char json[320];
    snprintf(json, sizeof(json),
             "{\"error\":{\"message\":\"context over budget; compact and "
             "retry\",\"type\":\"context_over_budget\",\"code\":null,"
             "\"kv_used\":%d,\"kv_cap\":%d}}",
             kv_used, kv_cap);
    hs_send_sse(client_fd, json);
    hs_end_sse(client_fd);
}

/* A turn refused to protect host memory, on an open SSE stream: an OpenAI
 * error event carrying the refusal (the caller ends the stream). */
static void send_memory_constrained_sse(int client_fd, const char *reason)
{
    json_writer jw;
    if (!jw_init(&jw))
        return;
    jw_raw(&jw, "{\"error\":{\"message\":");
    jw_string(&jw, reason && reason[0] ? reason
                                       : "server memory-constrained; retry shortly");
    jw_raw(&jw, ",\"type\":\"server_memory_constrained\",\"code\":null}}");
    hs_send_sse(client_fd, jw.data);
    jw_free(&jw);
}

static void ensure_role_sent(stream_ctx *sc)
{
    if (!sc->role_sent) {
        send_chunk(sc, "\"role\":\"assistant\"", NULL, NULL, 0);
        sc->role_sent = true;
    }
}

/* serve_generation delta callback: forward each token delta as an SSE chunk,
 * carrying the logprobs of the tokens whose bytes this delta contains. */
static void stream_delta_cb(const char *text, int is_reasoning,
                            const sg_token_logprob *const *logprobs,
                            int n_logprobs, void *ud)
{
    stream_ctx *sc = ud;
    ensure_role_sent(sc);
    json_writer jw;
    if (!jw_init(&jw)) return;
    jw_raw(&jw, is_reasoning ? "\"reasoning_content\":" : "\"content\":");
    jw_string(&jw, text);
    send_chunk(sc, jw.data, NULL, logprobs, n_logprobs);
    jw_free(&jw);
}

/* tool_call_stream sink: the OpenAI opening delta of one call (index, id,
 * type, name, empty arguments).  Clients concatenate the argument fragments
 * that follow under the same index. */
static void stream_tool_call_begin(void *ud, int index, const char *id,
                                   const char *name)
{
    stream_ctx *sc = ud;
    ensure_role_sent(sc);
    json_writer jw;
    if (!jw_init(&jw)) return;
    jw_printf(&jw, "\"tool_calls\":[{\"index\":%d,\"id\":", index);
    jw_string(&jw, id);
    jw_raw(&jw, ",\"type\":\"function\",\"function\":{\"name\":");
    jw_string(&jw, name);
    jw_raw(&jw, ",\"arguments\":\"\"}}]");
    send_chunk(sc, jw.data, NULL, NULL, 0);
    jw_free(&jw);
}

/* tool_call_stream sink: one argument fragment of call `index`. */
static void stream_tool_call_arguments(void *ud, int index, const char *bytes,
                                       size_t n)
{
    stream_ctx *sc = ud;
    json_writer jw;
    if (!jw_init(&jw)) return;
    jw_printf(&jw, "\"tool_calls\":[{\"index\":%d,\"function\":{\"arguments\":",
              index);
    jw_string_n(&jw, bytes, n);
    jw_raw(&jw, "}}]");
    send_chunk(sc, jw.data, NULL, NULL, 0);
    jw_free(&jw);
}

/* tool_call_stream sink: tool-call bytes were generated.  If the wire has been
 * silent for the keepalive interval (the part being written cannot be sent
 * yet), send an SSE comment so a client watchdog sees a live stream. */
static void stream_tool_call_progress(void *ud)
{
    stream_ctx *sc = ud;
    double interval = sc->keepalive_s > 0 ? sc->keepalive_s : OAI_TOOL_KEEPALIVE_S;
    double now = oai_now_s();
    if (now - sc->last_write_s < interval) return;
    if (hs_send_sse_comment(sc->client_fd, "keepalive") != 0)
        sc->client_gone = true;
    sc->last_write_s = now;
}

/* sg_gen_params.tool_tap → the streamer. */
static void stream_tool_tap(const char *bytes, size_t n, void *ud)
{
    tcs_feed(ud, bytes, n);
}

/* Set up `ts` to stream this completion's tool calls into `sc`, and arm the
 * generation params so the emitter feeds it. */
static bool stream_tool_calls_begin(stream_ctx *sc, tcs_streamer *ts,
                                    const chat_request *req, sg_gen_params *g)
{
    tcs_sink sink = { stream_tool_call_begin, stream_tool_call_arguments,
                      stream_tool_call_progress, sc };
    if (!tcs_init(ts, req->gen.dialect, req->tools, &sink)) return false;
    sc->tool_stream = ts;
    g->tool_tap = stream_tool_tap;
    g->tool_tap_ud = ts;
    return true;
}

/* Undo stream_tool_calls_begin. */
static void stream_tool_calls_end(stream_ctx *sc, tcs_streamer *ts)
{
    tcs_free(ts);
    sc->tool_stream = NULL;
}

/*
 * Generation params for choice `index` of an `n`-completion request.
 *
 * A `seed` makes the REQUEST reproducible — it does not mean the n completions
 * should be identical, which is what reusing one RNG stream would produce.  So
 * each choice offsets the seed; re-running the same request with the same seed
 * still replays all n completions exactly.  (With temperature 0 every choice is
 * the argmax and they coincide regardless — that is inherent to greedy decoding,
 * not something the seed can fix.)
 */
static sg_gen_params gen_params_for_choice(const chat_request *req, int index)
{
    sg_gen_params g = req->gen;
    if (g.seed_set) g.seed += (uint64_t)index;
    return g;
}

/* What a routed turn's should_abort looks at: the client fd (streaming only —
 * NULL where a disconnect is not acted on) and the model entry it runs on. Both
 * outlive the synchronous completion. */
typedef struct {
    const int      *client_fd;
    const mr_entry *entry;
    const bool     *client_gone;   /* the stream's failed-write flag, or NULL */
} oai_turn_abort_ctx;

/* should_abort for a routed turn: true once the process has been asked to stop,
 * once an unload of this model has cancelled its live turns
 * (mr_turn_cancel_requested), or — streaming — once the client disconnected.
 *
 * The shutdown arm is what makes SIGTERM able to retire the card residencies:
 * the accept loop cannot tear the accelerator down while a turn is still
 * submitting descriptors against it, so the turn has to stop being asked to
 * finish first. The unload arm is the same idea for one model: an unload that
 * outwaited its finish budget asks the turn to stop so it can free the card
 * instead of refusing. The generation loop polls this between prefill chunks
 * and every 8 decode tokens, which bounds how long a stop waits. */
static int oai_turn_should_abort(void *ud)
{
    const oai_turn_abort_ctx *a = ud;

    if (pn_serve_shutdown_requested())
        return 1;
    if (a && mr_turn_cancel_requested(a->entry))
        return 1;
    if (a && a->client_gone && *a->client_gone)
        return 1;   /* a write already failed: the slot goes back now */
    return a && a->client_fd && !hs_client_alive(*a->client_fd);
}

/* The streaming arming of a routed turn's abort: same entry, plus the client
 * fd. `out` must outlive the completion (the caller's stack frame). */
static void arm_streaming_abort(sg_gen_params *g, oai_turn_abort_ctx *out,
                                const int *client_fd, const bool *client_gone)
{
    const oai_turn_abort_ctx *routed = g->should_abort == oai_turn_should_abort
                                     ? g->abort_ud : NULL;

    out->client_fd = client_fd;
    out->client_gone = client_gone;
    out->entry = routed ? routed->entry : NULL;
    g->should_abort = oai_turn_should_abort;
    g->abort_ud = out;
}

/* Was this aborted turn stopped by an unload of its model (not a disconnect)? */
static bool turn_cancelled_by_unload(const sg_gen_params *g)
{
    const oai_turn_abort_ctx *a = g->should_abort == oai_turn_should_abort
                                ? g->abort_ud : NULL;

    return a && mr_turn_cancel_requested(a->entry);
}

/* The terminal SSE error a streaming turn gets when an unload cancels it. */
static void send_model_unloaded_sse(int client_fd)
{
    hs_send_sse(client_fd, "{\"error\":{\"message\":\"the model was unloaded; "
                "this turn was cancelled\",\"type\":\"model_unloaded\","
                "\"code\":\"model_unloaded\"}}");
}

/* The terminal SSE error of a turn the decode watchdog stopped: no token came
 * for bs_decode_stall_ms(), the slot is released, and the stream ends here
 * instead of staying open with nothing on it. */
static void send_generation_stalled_sse(int client_fd)
{
    hs_send_sse(client_fd, "{\"error\":{\"message\":\"generation stalled: no token "
                "for too long; the turn was stopped, retry it\",\"type\":"
                "\"generation_stalled\",\"code\":\"generation_stalled\"}}");
}

/* should_abort for a turn with no client-liveness check (the non-streaming
 * paths, where a disconnect is not acted on): true only once the process has
 * been asked to stop. Without this a non-streaming completion would run to its
 * max_tokens through a SIGTERM and time the drain out — losing the retirement
 * in exactly the mid-generation case it exists for. */
static int oai_shutdown_requested(void *ud)
{
    (void)ud;
    return pn_serve_shutdown_requested();
}

/* Emit one completion's closing chunks into an ALREADY-OPEN SSE stream: the
 * tool-call deltas (if any) and the finish chunk.  Shared by both run paths so a
 * change to the closing shape cannot land on one and not the other — the last
 * bug in this file was exactly that (a field on one path only). */
static void stream_finish_choice(stream_ctx *sc, chat_request *req,
                                 sg_result *res, int rc)
{
    ensure_role_sent(sc);
    sc->context_capped = (rc == 0 && res->context_capped);
    /* Rides the closing chunk(s), exactly like context_capped.  Resolved against
     * what the generation actually did — see wire_response_format_status. */
    sc->response_format_status = wire_response_format_status(req, res);

    tc_tool_call *calls = NULL;
    int n_calls = 0;
    tc_parse_error tool_error = {0};
    if (rc == 0) collect_tool_calls(req, res, &calls, &n_calls, &tool_error);
    if (n_calls > 0) {
        /* Whatever the streamer has not already put on the wire — all of it on
         * a path or dialect that streams nothing early. */
        tcs_streamer fallback;
        tcs_streamer *ts = sc->tool_stream;
        if (!ts) {
            tcs_sink sink = { stream_tool_call_begin, stream_tool_call_arguments,
                              NULL, sc };
            ts = tcs_init(&fallback, req->gen.dialect, NULL, &sink) ? &fallback : NULL;
        }
        if (ts) tcs_finish(ts, calls, n_calls);
        if (ts == &fallback) tcs_free(&fallback);
        /* Set AFTER the deltas so it rides the CLOSING chunk only, exactly like
         * context_capped — a per-delta copy would repeat one event N times. */
        sc->tool_error = &tool_error;
        /* Same precedence as the non-streaming body: a degenerated turn reports
         * the degeneration even when a call was parsed out of it. */
        send_chunk(sc, "", finish_is_repetition_loop(res) ? res->finish_reason
                                                          : "tool_calls",
                   NULL, 0);
    } else {
        sc->tool_error = &tool_error;
        send_chunk(sc, "", rc == 0 ? res->finish_reason : "stop", NULL, 0);
    }
    sc->tool_error = NULL;         /* the borrow ends with this stack frame */
    tc_parse_error_free(&tool_error);
    tc_tool_calls_free(calls, n_calls);
}

/*
 * Stream ONE completion into an already-open SSE stream (single-user path).
 *
 * The SSE lifecycle (hs_begin_sse / hs_end_sse) belongs to the CALLER: an `n` > 1
 * request streams several completions down one stream, told apart by
 * choices[].index.  Returns the sg rc so the caller can stop after a failure it
 * has already reported on the stream.
 */
static int run_streaming(int client_fd, pn_model *model, pn_state *state,
                         chat_request *req, const sg_gen_params *gen,
                         stream_ctx *sc, const sg_prefix_ctx *pc)
{
    sg_gen_params g = *gen;
    oai_turn_abort_ctx abort_ctx;
    /* Free the model slot the instant the client disconnects (e.g. Esc) instead
     * of finishing a cold prefill nobody will read — that lingering slot is what
     * made the next turn hit a "busy" empty response. */
    arm_streaming_abort(&g, &abort_ctx, &client_fd, &sc->client_gone);
    tcs_streamer ts;
    bool tool_stream = stream_tool_calls_begin(sc, &ts, req, &g);
    sg_result res = {0};
    int rc = sg_run_completion(model, state, req->messages, req->n_messages,
                               &g, stream_delta_cb, sc, &res, pc);
    if (turn_cancelled_by_unload(&g)) {
        /* Checked before rc: a cancel landing mid-decode ends the loop like a
         * clean "stop" (rc 0), and closing it as one would pass a cut-off answer
         * off as complete. */
        fprintf(stderr, "[serve] model unloading — turn cancelled\n");
        send_model_unloaded_sse(client_fd);
        rc = SG_ABORTED;
    } else if (rc == SG_ABORTED) {
        /* Client is gone and the slot is released; nothing to send. */
        fprintf(stderr, "[serve] client disconnected — turn aborted, slot released\n");
    } else if (rc == SG_ADMIT_DENIED) {
        send_memory_constrained_sse(client_fd, res.admit_reason);
    } else if (rc == SG_CONTEXT_OVER_BUDGET) {
        send_context_over_budget_sse(client_fd, res.kv_used_tokens,
                                     res.kv_cap_tokens);
    } else {
        stream_finish_choice(sc, req, &res, rc);
        if (req->include_usage)
            send_usage_chunk(sc, &res);
    }
    if (tool_stream) stream_tool_calls_end(sc, &ts);
    sg_result_free(&res);
    return rc;
}

/*
 * Stream ONE completion into an already-open SSE stream (C3 batched path).
 * `slot` is a pre-acquired slot index (bs_slot_acquire already called); the
 * completion takes ownership of it.  SSE lifecycle belongs to the caller, as on
 * the single-user path.
 */
static int run_streaming_batched(int client_fd, pn_model *model,
                                 pn_batch_sched *sched, chat_request *req,
                                 const sg_gen_params *gen, stream_ctx *sc,
                                 int slot, const char *sid)
{
    sg_gen_params g = *gen;
    /* Same arming as run_streaming, and for the same reason — except this is the
     * path the fleet actually runs, so until now an abandoned request burned its
     * whole cold prefill (~205 s at 8k) holding a decode slot.  A foreign client
     * with a fixed timeout would retry into a second slot and repeat it; the
     * server, not the client, has to be the one that stops. */
    oai_turn_abort_ctx abort_ctx;
    arm_streaming_abort(&g, &abort_ctx, &client_fd, &sc->client_gone);
    tcs_streamer ts;
    bool tool_stream = stream_tool_calls_begin(sc, &ts, req, &g);
    sg_result res = {0};
    int rc = sg_run_completion_batched(model, sched, slot, req->messages,
                                       req->n_messages, &g,
                                       stream_delta_cb, sc, &res, sid);
    if (turn_cancelled_by_unload(&g)) {
        /* Before rc, for the reason run_streaming gives. */
        fprintf(stderr, "[serve] model unloading — batched turn cancelled\n");
        send_model_unloaded_sse(client_fd);
        rc = SG_ABORTED;
    } else if (rc == SG_ABORTED) {
        /* Client is gone and the slot is released; nothing to send. */
        fprintf(stderr, "[serve] client disconnected — batched turn aborted, "
                "slot released\n");
    } else if (rc == SG_STALLED) {
        send_generation_stalled_sse(client_fd);
    } else if (rc == SG_ADMIT_DENIED) {
        send_memory_constrained_sse(client_fd, res.admit_reason);
    } else if (rc == SG_CONTEXT_OVER_BUDGET) {
        send_context_over_budget_sse(client_fd, res.kv_used_tokens,
                                     res.kv_cap_tokens);
    } else {
        stream_finish_choice(sc, req, &res, rc);
        if (req->include_usage)
            send_usage_chunk(sc, &res);
    }
    if (tool_stream) stream_tool_calls_end(sc, &ts);
    sg_result_free(&res);
    return rc;
}

/* Resolve the request's `model` field to a resident registry entry (with a
 * reference taken). Routing rule: with exactly one resident model every
 * request routes to it regardless of the field (the single-model server's
 * behavior, so existing clients keep working); with several, an exact name
 * match is required (a missing field falls back to the default model) and an
 * unknown name is a clean 404. A requested copy (route->replica) binds a new
 * session to exactly that copy (replica_pin.h). Every response after routing
 * carries X-Pantheon-Replica. Returns NULL after answering the client. */
static mr_entry *route_model(int client_fd, mr_registry *reg,
                             const char *name, const char *sid, mr_route *route)
{
    mr_entry *entry = NULL;

    /* Replicas of one name: the session's own replica, else the requested one,
     * else the least busy. */
    if (route->replica == 0 && mr_count(reg) <= 1) {
        entry = mr_acquire(reg, NULL);
        pn_replica_pin_announce(entry);
    } else {
        entry = pn_replica_pin_acquire(client_fd, reg, mr_count(reg) <= 1 ? NULL : name,
                                       sid, route);
        if (!entry && route->status == MR_ROUTE_NO_REPLICA)
            return NULL;   /* answered: the copy is not loaded */
    }
    if (entry)
        return entry;
    if (mr_count(reg) == 0)
        hs_send_error(client_fd, 503, "no_models_loaded",
                      "no model is currently loaded");
    else
        hs_send_error(client_fd, 404, "model_not_found",
                      "the requested model is not loaded (see GET /v1/models)");
    return NULL;
}

/* Log a session-capacity 429 with who holds the seats, so a refusal while
 * sessions_used reads below max_sessions (a family holding several slots) is
 * visible in the serve log. */
static void log_session_capacity_refuse(const mr_entry *entry, const char *sid)
{
    ms_occupancy o;

    ms_session_occupancy(entry, &o);
    fprintf(stderr, "[admission] 429 session capacity: sid=%s model=%s replica=%d "
            "sessions=%d/%d slots_held=%d slots_free=%d seat_free=%d\n",
            sid && sid[0] ? sid : "-", entry->name, entry->replica,
            o.sessions_used, o.max_sessions, o.slots_held, o.slots_free,
            (int)o.seat_free);
}

/* Send the Phase-5 "model at session capacity" 429 with a Retry-After and a
 * re-prefill restart estimate. `restart_tokens` is the committed length the
 * refused (returning/expired) session would re-prefill; 0 for a truly new sid. */
static void send_session_capacity_refuse(int client_fd, const mr_route *route,
                                         const mr_entry *entry, const char *sid,
                                         int restart_tokens)
{
    int restart_s = (int)((double)restart_tokens
                          / oai_prefill_tok_per_s() + 0.5);
    log_session_capacity_refuse(entry, sid);
    pn_replica_pin_send_full(client_fd, route, entry, restart_s);
}

/* Take a decode slot (see acquire_decode_slot) without answering the client. */
static bs_acquire_result acquire_decode_slot_quiet(pn_batch_sched *sched, const char *sid,
                                                   int *slot_out, int *retry_after)
{
    /* Refresh the pressure snapshot so the scheduler sees the current host state
     * at decision time (memwatch polls every 250 ms; this forces an immediate
     * read without waiting for the next tick). */
    if (pn_memwatch_running()) {
        pn_mem_state ms;
        pn_memwatch_read(&ms);
        bs_notify_pressure(sched, (int)ms.level);
    }
    static const int SLOT_WAIT_MS = 30000;  /* 30 s max queue wait */
    *retry_after = 0;
    *slot_out = -1;
    return bs_slot_acquire_or_wait(sched, SLOT_WAIT_MS, slot_out, retry_after, sid);
}

/* The OpenAI error body for a failed slot acquire. */
static void send_acquire_refusal(int client_fd, bs_acquire_result ar, int retry_after,
                                 const mr_route *route, const mr_entry *entry,
                                 const char *sid)
{
    if (ar == BS_ACQUIRE_AT_CAPACITY) {
        send_session_capacity_refuse(client_fd, route, entry, sid, 0);
        return;
    }

    int http_code;
    const char *err_code, *err_msg;
    switch (ar) {
    case BS_ACQUIRE_503:
        http_code = 503;
        err_code  = "server_memory_constrained";
        err_msg   = "server memory-constrained and queue full; retry shortly";
        break;
    case BS_ACQUIRE_TIMEOUT:
        http_code = 429;
        err_code  = "server_busy";
        err_msg   = "request timed out waiting for a decode slot; retry shortly";
        break;
    case BS_ACQUIRE_SHUTDOWN:
        http_code = 503;
        err_code  = "server_shutting_down";
        err_msg   = "server is shutting down";
        break;
    default: /* BS_ACQUIRE_429 */
        http_code = 429;
        err_code  = "server_busy";
        err_msg   = "all decode slots occupied and queue full; retry shortly";
        break;
    }
    if (retry_after > 0) {
        char ra_hdr[64];
        snprintf(ra_hdr, sizeof(ra_hdr), "Retry-After: %d\r\n", retry_after);
        hs_send_error_ex(client_fd, http_code, err_code, err_msg, ra_hdr);
    } else {
        hs_send_error(client_fd, http_code, err_code, err_msg);
    }
}

/*
 * Acquire one decode slot on the batched path, waiting in the scheduler's
 * bounded FIFO when every slot is busy and gating new prefills on host memory
 * pressure (D3 backpressure).  429 only when the queue is full; 503 when the
 * queue is full AND memory is CRITICAL.  The scheduler also admits the
 * session (max_sessions): a new session with no seat gets the immediate
 * 429 model_at_session_capacity, never a wait or another session's slot.
 *
 * `answer_client` sends the OpenAI error body on failure.  It is false for the
 * second and later completions of an `n` > 1 request: a response is already in
 * flight (SSE) or already has usable choices, so an error body would corrupt it
 * — the caller reports the shortfall in `n_status` instead.
 */
static bs_acquire_result acquire_decode_slot(int client_fd,
                                             pn_batch_sched *sched,
                                             const mr_route *route,
                                             const mr_entry *entry,
                                             const char *sid, int *slot_out,
                                             bool answer_client)
{
    int retry_after = 0;
    bs_acquire_result ar = acquire_decode_slot_quiet(sched, sid, slot_out, &retry_after);
    if (ar != BS_ACQUIRE_OK && answer_client)
        send_acquire_refusal(client_fd, ar, retry_after, route, entry, sid);
    return ar;
}

/*
 * The first completion's slot, failing over between copies: when this copy
 * refuses (no seat, queue full, no slot in time) a session that holds no seat
 * here and named no copy, the least busy other copy with a free seat is tried
 * before the client hears a 429 — a full replica never refuses a session a free
 * one could seat.  On a move *entry (and the scheduler, model and abort target
 * derived from it) switch to the new copy; the old reference is released.
 * Answers the client on failure, like acquire_decode_slot.
 */
static bs_acquire_result acquire_with_failover(int client_fd, mr_registry *reg,
                                               mr_route *route, mr_entry **entry,
                                               oai_turn_abort_ctx *turn_abort,
                                               const char *sid, int *slot_out)
{
    int retry_after = 0;
    mr_entry *at = *entry;
    bs_acquire_result ar = acquire_decode_slot_quiet(at->sched, sid, slot_out, &retry_after);

    while (ar != BS_ACQUIRE_OK && ar != BS_ACQUIRE_SHUTDOWN && route->replica <= 0 &&
           !ms_session_holds_seat(at, sid)) {
        mr_entry *next = mr_failover_session(reg, at, sid, route);
        if (!next || !next->sched) {
            if (next)
                mr_release(reg, next);
            break;
        }
        mr_release(reg, at);
        at = next;
        *entry = at;
        turn_abort->entry = at;
        pn_replica_pin_announce(at);
        ar = acquire_decode_slot_quiet(at->sched, sid, slot_out, &retry_after);
    }
    if (ar != BS_ACQUIRE_OK)
        send_acquire_refusal(client_fd, ar, retry_after, route, at, sid);
    return ar;
}

void oai_handle_chat(int client_fd, mr_registry *reg, const char *body,
                     const char *session_id, const char *replica_header)
{
    if (!body) { hs_send_error(client_fd, 400, "invalid_request_error", "empty request body"); return; }
    sg_request_dump_body(body);

    pn_reqprof_phase_start(PN_REQ_JSON_PARSE);
    jr_value *root = jr_parse(body);
    if (!root) {
        pn_reqprof_phase_end(PN_REQ_JSON_PARSE);
        hs_send_error(client_fd, 400, "invalid_request_error", "malformed JSON");
        return;
    }

    chat_request req;
    const char *err = NULL;
    int parse_rc = parse_chat_request(root, &req, &err);
    pn_reqprof_phase_end(PN_REQ_JSON_PARSE);
    if (parse_rc != 0) {
        hs_send_error(client_fd, 400, "invalid_request_error", err ? err : "bad request");
        chat_request_free(&req);
        jr_free(root);
        return;
    }

    /* The copy the client asks for (body "replica" wins over the header). A
     * turn may re-place a bound session that holds no seat on its copy. */
    mr_route route = { .place_unseated = true };
    char pin_err[96];
    if (pn_replica_pin_parse(root, replica_header, &route.replica, pin_err,
                             sizeof(pin_err)) != 0) {
        hs_send_error(client_fd, 400, "invalid_request_error", pin_err);
        chat_request_free(&req);
        jr_free(root);
        return;
    }

    /* Route to the resident model. The held reference keeps mr_unload from
     * freeing the model mid-generation. */
    pn_reqprof_phase_start(PN_REQ_ROUTE);
    mr_entry *entry = route_model(client_fd, reg, req.model,
                                  (req.session_id && req.session_id[0]) ? req.session_id
                                                                        : session_id,
                                  &route);
    if (!entry) {
        pn_reqprof_phase_end(PN_REQ_ROUTE);
        chat_request_free(&req);
        jr_free(root);
        return;
    }
    /* Pantheon Local: a session belongs to the person who first used it (§4.8). */
    if (ms_session_principal_check(entry, (req.session_id && req.session_id[0]) ? req.session_id
                                                                                 : session_id,
                                   hs_thread_principal(), hs_thread_principal_admin(),
                                   MS_PRINCIPAL_BIND) != MS_OK) {
        pn_reqprof_phase_end(PN_REQ_ROUTE);
        hs_send_error(client_fd, 403, "session_not_yours", MS_NOT_YOURS_MESSAGE);
        mr_release(reg, entry);
        chat_request_free(&req);
        jr_free(root);
        return;
    }

    pn_model        *model    = entry->model;
    const char      *model_id = entry->name;
    pn_batch_sched  *sched    = entry->sched;   /* non-NULL when C3 is active */

    /* Every turn on this entry is now cancellable by an unload of it (plus the
     * shutdown check parse armed); the streaming paths add client liveness. */
    oai_turn_abort_ctx turn_abort = { .client_fd = NULL, .entry = entry };
    req.gen.should_abort = oai_turn_should_abort;
    req.gen.abort_ud = &turn_abort;

    /* Now that the resident model is known, resolve its tool-call dialect and
     * swap the tools preamble / re-render history into it (e.g. MiniCPM5's
     * <function name=…> or ornith's JSON). No-op for the default Qwen/N2 dialect
     * or a tools-less request beyond storing the dialect on req->gen. */
    chat_apply_tool_dialect(&req, model, model_id);
    pn_reqprof_phase_end(PN_REQ_ROUTE);

    /* Resolve the AUTO-path reasoning budget: request override (if the body
     * carried reasoning_budget / max_reasoning_tokens) ELSE the per-model default
     * (from --reasoning-budget) ELSE 0 (disabled). */
    if (!req.reasoning_budget_set)
        req.gen.reasoning_budget = entry->reasoning_budget;
    req.gen.reasoning_reserve = entry->reasoning_reserve;
    /* Absent sampling fields take the model's defaults (A4). */
    pn_model_sampling_apply(&req.sampling, &entry->sampling, &req.gen);
    /* Name the model on the params purely so an abnormal-event log (the
     * reasoning force-close) can say WHICH model it happened to.  Borrowed from
     * the registry entry, which outlives this request (it is ref-held). */
    req.gen.model_name = entry->name;

    /* PC0.1: effective prefix-cache session id — the "session_id" body field
     * wins over the X-Session-Id header; empty/absent → NULL (no caching, full
     * prefill, byte-identical to today). */
    const char *sid = (req.session_id && req.session_id[0]) ? req.session_id
                    : (session_id && session_id[0])         ? session_id
                                                            : NULL;

    char id[32];
    make_completion_id(id, sizeof(id));
    long created = (long)time(NULL);

    /* ── Phase 5: session admission — REFUSE at capacity + idle TTL ──
     * A new session past the model's max_sessions gets a clean 429 (never
     * queued, never seated on a live session's state — an existing session
     * always resumes); idle unwatched sessions are reaped first.  The BATCHED
     * path admits inside the scheduler, atomically with the slot claim
     * (acquire_decode_slot): its slots hold the sessions.  The SINGLE-USER
     * path admits below, under gen_lock: its generation mutates the same
     * per-session cache states the reaper frees. */

    /* ── C3 batched path (QWEN35_BATCH_SLOTS=1 and model is GPU-placed) ── */
    if (sched) {
        stream_ctx sc = { .client_fd = client_fd, .id = id, .created = created,
                          .model_id = model_id };
        chat_choice *choices = NULL;
        int n_delivered = 0;
        bool sse_open = false;
        const char *n_status = req.n_status;

        if (!req.stream) {
            choices = calloc((size_t)req.n_choices, sizeof(*choices));
            if (!choices) {
                hs_send_error(client_fd, 500, "server_error", "oom");
                mr_release(reg, entry);
                chat_request_free(&req);
                jr_free(root);
                return;
            }
        }

        for (int ci = 0; ci < req.n_choices; ci++) {
            /* D1+D3: acquire a slot BEFORE opening SSE so a 429/503 can be sent
             * cleanly without an in-progress SSE stream to abort. */
            int slot = -1;
            bs_acquire_result ar;
            if (ci == 0) {
                ar = acquire_with_failover(client_fd, reg, &route, &entry, &turn_abort,
                                           sid, &slot);
                /* A failover moved the turn: everything below runs on the new copy
                 * (same model file, same name). */
                sched = entry->sched;
                model = entry->model;
                model_id = entry->name;
                sc.model_id = model_id;
                req.gen.model_name = entry->name;
            } else {
                ar = acquire_decode_slot(client_fd, sched, &route, entry, sid, &slot, false);
            }
            if (ar != BS_ACQUIRE_OK) {
                /* The FIRST choice failing is the whole request failing, and
                 * acquire_decode_slot already answered.  A LATER choice failing
                 * cannot inject an error into a response already in flight — the
                 * client gets the completions that did land, and n_status says
                 * the rest did not. */
                if (ci == 0) {
                    pn_replica_pin_undo(&route, entry, sid);
                    goto batched_done;
                }
                n_status = "partial_slot_unavailable";
                break;
            }

            /* Only the FIRST completion belongs to the session: the client keeps
             * at most one of the n, so committing a later one would leave the
             * warm prefix describing a conversation that never happened.  The
             * extra completions therefore full-prefill — the documented cost. */
            const char *choice_sid = (ci == 0) ? sid : NULL;
            sg_gen_params gen = gen_params_for_choice(&req, ci);

            if (req.stream) {
                if (!sse_open) {
                    sc.client_gone = hs_begin_sse(client_fd) != 0;
                    sse_open = true;
                }
                sc.choice_index = ci;
                sc.role_sent    = false;
                int rc = run_streaming_batched(client_fd, model, sched, &req,
                                               &gen, &sc, slot, choice_sid);
                n_delivered++;
                if (rc != 0) break;   /* already reported on the stream */
            } else {
                chat_choice *ch = &choices[n_delivered];
                int rc = sg_run_completion_batched(model, sched, slot,
                                                   req.messages, req.n_messages,
                                                   &gen, NULL, NULL, &ch->res,
                                                   choice_sid);
                if (turn_cancelled_by_unload(&gen))
                    rc = SG_ABORTED;   /* a mid-decode cancel returns 0 */
                if (rc != 0) {
                    if (n_delivered > 0) {
                        /* Some completions already succeeded — return them
                         * rather than throwing the work away, and say why the
                         * rest are missing. */
                        n_status = "partial_generation_failed";
                        sg_result_free(&ch->res);
                        memset(ch, 0, sizeof(*ch));
                        break;
                    }
                    if (rc == SG_ADMIT_DENIED)
                        hs_send_error(client_fd, 503, "server_memory_constrained",
                                      ch->res.admit_reason[0]
                                          ? ch->res.admit_reason
                                          : "server memory-constrained; retry shortly");
                    else if (rc == SG_CONTEXT_OVER_BUDGET)
                        send_context_over_budget_http(client_fd,
                                                      ch->res.kv_used_tokens,
                                                      ch->res.kv_cap_tokens);
                    else if (turn_cancelled_by_unload(&gen))
                        hs_send_error(client_fd, 503, "model_unloaded",
                                      "the model was unloaded; this turn was "
                                      "cancelled");
                    else if (rc == SG_STALLED)
                        hs_send_error(client_fd, 503, "generation_stalled",
                                      "generation stalled: no token for too long; "
                                      "the turn was stopped, retry it");
                    else
                        hs_send_error(client_fd, 500, "server_error",
                                      "generation failed");
                    sg_result_free(&ch->res);
                    memset(ch, 0, sizeof(*ch));
                    goto batched_done;
                }
                collect_tool_calls(&req, &ch->res, &ch->calls, &ch->n_calls,
                                   &ch->tool_error);
                n_delivered++;
            }
        }

        if (req.stream) {
            if (sse_open) hs_end_sse(client_fd);
        } else if (n_delivered > 0) {
            send_completion_json(client_fd, id, created, model_id, choices,
                                 n_delivered,
                                 wire_response_format_status(&req,
                                                             &choices[0].res),
                                 n_status);
        }

    batched_done:
        /* `choices` is allocated only on the NON-streaming path — a streaming
         * request counts delivered completions but stores none, so this must be
         * guarded on the ARRAY, not on the count. */
        for (int ci = 0; choices && ci < n_delivered; ci++)
            chat_choice_free(&choices[ci]);
        free(choices);
        /* No gen_lock to release — the slot scheduler manages concurrency. */
        mr_release(reg, entry);
        chat_request_free(&req);
        jr_free(root);
        return;
    }

    /* ── Legacy single-user path (sched == NULL) ── */
    /* Serialize on ITS lock only: chats to different models run concurrently
     * (sampler scratch is _Thread_local); a second chat to the SAME model
     * gets the immediate 429, exactly the old single-model behavior. */
    if (mr_trylock_generation(entry) != 0) {
        hs_send_error(client_fd, 429, "server_busy",
                      "a generation is already in progress on this model; "
                      "retry shortly");
        pn_replica_pin_undo(&route, entry, sid);
        mr_release(reg, entry);
        chat_request_free(&req);
        jr_free(root);
        return;
    }

    /* ── Phase 5 admission, single-user path — under gen_lock ──
     * gen_lock (just taken) is the single-user path's cache serializer: the
     * generation below runs its acquire/commit with pc_lock==NULL, relying on
     * gen_lock. Running admission (which reaps idle slots — freeing their
     * pn_states) under the SAME lock means the reaper can never race a live
     * generation on this model. A concurrent request fails the trylock above and
     * returns 429 without ever reaching the reaper. */
    if (entry->prefix_cache && sid) {
        int restart_tokens = 0;
        pn_session_admit v = pn_prefix_cache_admit(entry->prefix_cache, sid,
                                                   ms_now_ms(),
                                                   ms_session_idle_ttl_ms(),
                                                   &restart_tokens);
        if (v == PN_SESSION_ADMIT_AT_CAPACITY) {
            pthread_mutex_unlock(&entry->gen_lock);
            send_session_capacity_refuse(client_fd, &route, entry, sid, restart_tokens);
            pn_replica_pin_undo(&route, entry, sid);
            mr_release(reg, entry);
            chat_request_free(&req);
            jr_free(root);
            return;
        }
    }
    pn_state *state = entry->state;

    /* PC2: prefix-cache context — active only when the client sent a session id
     * and the model has a cache. NULL sid → full prefill (byte-identical). The
     * gen_lock already serializes this model's single-user path, so the cache
     * op (thread-unsafe) is safe under it; pc_lock stays NULL here. */
    sg_prefix_ctx pcx = { .cache = entry->prefix_cache, .sid = sid,
                          .pc_lock = NULL };
    const sg_prefix_ctx *pc = (entry->prefix_cache && sid) ? &pcx : NULL;

    {
        stream_ctx sc = { .client_fd = client_fd, .id = id, .created = created,
                          .model_id = model_id };
        chat_choice *choices = NULL;
        int n_delivered = 0;
        bool sse_open = false;
        const char *n_status = req.n_status;

        if (!req.stream)
            choices = calloc((size_t)req.n_choices, sizeof(*choices));

        for (int ci = 0; ci < req.n_choices && (req.stream || choices); ci++) {
            /* Only the FIRST completion belongs to the session — see the batched
             * path for why the extras run uncached. */
            const sg_prefix_ctx *choice_pc = (ci == 0) ? pc : NULL;
            sg_gen_params gen = gen_params_for_choice(&req, ci);

            if (req.stream) {
                if (!sse_open) {
                    sc.client_gone = hs_begin_sse(client_fd) != 0;
                    sse_open = true;
                }
                sc.choice_index = ci;
                sc.role_sent    = false;
                int rc = run_streaming(client_fd, model, state, &req, &gen, &sc,
                                       choice_pc);
                n_delivered++;
                if (rc != 0) break;   /* already reported on the stream */
            } else {
                chat_choice *ch = &choices[n_delivered];
                int rc = sg_run_completion(model, state, req.messages,
                                           req.n_messages, &gen, NULL, NULL,
                                           &ch->res, choice_pc);
                if (turn_cancelled_by_unload(&gen))
                    rc = SG_ABORTED;   /* a mid-decode cancel returns 0 */
                if (rc != 0) {
                    if (n_delivered > 0) {
                        n_status = "partial_generation_failed";
                        sg_result_free(&ch->res);
                        memset(ch, 0, sizeof(*ch));
                        break;
                    }
                    if (rc == SG_ADMIT_DENIED)
                        hs_send_error(client_fd, 503, "server_memory_constrained",
                                      ch->res.admit_reason[0]
                                          ? ch->res.admit_reason
                                          : "server memory-constrained; retry shortly");
                    else if (rc == SG_CONTEXT_OVER_BUDGET)
                        send_context_over_budget_http(client_fd,
                                                      ch->res.kv_used_tokens,
                                                      ch->res.kv_cap_tokens);
                    else if (turn_cancelled_by_unload(&gen))
                        hs_send_error(client_fd, 503, "model_unloaded",
                                      "the model was unloaded; this turn was "
                                      "cancelled");
                    else
                        hs_send_error(client_fd, 500, "server_error",
                                      "generation failed");
                    sg_result_free(&ch->res);
                    memset(ch, 0, sizeof(*ch));
                    break;
                }
                collect_tool_calls(&req, &ch->res, &ch->calls, &ch->n_calls,
                                   &ch->tool_error);
                n_delivered++;
            }
        }

        if (req.stream) {
            if (sse_open) hs_end_sse(client_fd);
        } else if (n_delivered > 0) {
            send_completion_json(client_fd, id, created, model_id, choices,
                                 n_delivered,
                                 wire_response_format_status(&req,
                                                             &choices[0].res),
                                 n_status);
        } else if (!choices) {
            hs_send_error(client_fd, 500, "server_error", "oom");
        }
        /* `choices` is allocated only on the NON-streaming path — a streaming
         * request counts delivered completions but stores none, so this must be
         * guarded on the ARRAY, not on the count. */
        for (int ci = 0; choices && ci < n_delivered; ci++)
            chat_choice_free(&choices[ci]);
        free(choices);
    }

    /* Phase 5: mark the session active so the idle-TTL reaper measures from the
     * END of this turn. Under gen_lock (serializes this model's cache ops). */
    if (entry->prefix_cache && sid)
        pn_prefix_cache_touch(entry->prefix_cache, sid, ms_now_ms());

    pthread_mutex_unlock(&entry->gen_lock);
    mr_release(reg, entry);
    chat_request_free(&req);
    jr_free(root);
}

/* ============================================================
 * POST /session/restore — session warm-tier IMPORT (Phase 7)
 * ============================================================ */

/* Parse tokens[] (a JSON array of ints) into a malloc'd uint32_t array. Returns
 * the array (caller frees) and sets *n; NULL on a missing/empty/bad array. */
static uint32_t *parse_token_array(const jr_value *arr, int *n)
{
    *n = 0;
    if (jr_kind(arr) != JR_ARRAY)
        return NULL;
    int len = jr_array_len(arr);
    if (len <= 0)
        return NULL;
    uint32_t *toks = malloc((size_t)len * sizeof(uint32_t));
    if (!toks)
        return NULL;
    for (int i = 0; i < len; i++)
        toks[i] = (uint32_t)jr_as_number(jr_array_get(arr, i), 0);
    *n = len;
    return toks;
}

/* Restore a client-held session blob into a live slot. Body:
 *   {"name":"model","session_id":"sid","blob":"<base64>","tokens":[...]}
 * `tokens` is the committed history the blob covers (the client owns it); its
 * length must equal the blob's snapshot pos. On a weights_version mismatch or a
 * bad blob, answer a clean error so the client falls back to message-replay. */
void oai_handle_session_restore(int client_fd, mr_registry *reg,
                                const char *body, const char *session_id,
                                const char *replica_header)
{
    if (!body) { hs_send_error(client_fd, 400, "invalid_request_error", "empty request body"); return; }
    jr_value *root = jr_parse(body);
    if (!root || jr_kind(root) != JR_OBJECT) {
        hs_send_error(client_fd, 400, "invalid_request_error", "malformed JSON");
        jr_free(root);
        return;
    }
    const char *name = jr_as_string(jr_object_get(root, "name"));
    const char *body_sid = jr_as_string(jr_object_get(root, "session_id"));
    const char *sid = (body_sid && body_sid[0]) ? body_sid
                    : (session_id && session_id[0]) ? session_id : NULL;
    const char *blob_b64 = jr_as_string(jr_object_get(root, "blob"));
    if (!sid || !blob_b64 || !blob_b64[0]) {
        hs_send_error(client_fd, 400, "invalid_request_error",
                      "'session_id' and 'blob' are required");
        jr_free(root);
        return;
    }
    int n_tok = 0;
    uint32_t *tokens = parse_token_array(jr_object_get(root, "tokens"), &n_tok);
    if (!tokens) {
        hs_send_error(client_fd, 400, "invalid_request_error",
                      "'tokens' (committed history) is required");
        jr_free(root);
        return;
    }

    /* A restore binds the session like its first turn: honour a requested copy. */
    mr_route route = {0};
    char pin_err[96];
    if (pn_replica_pin_parse(root, replica_header, &route.replica, pin_err,
                             sizeof(pin_err)) != 0) {
        free(tokens);
        hs_send_error(client_fd, 400, "invalid_request_error", pin_err);
        jr_free(root);
        return;
    }
    mr_entry *entry = pn_replica_pin_acquire(client_fd, reg, name, sid, &route);
    if (!entry) {
        free(tokens);
        if (route.status != MR_ROUTE_NO_REPLICA)
            hs_send_error(client_fd, 404, "model_not_found",
                          "the requested model is not loaded (see GET /v1/models)");
        jr_free(root);
        return;
    }
    /* A model with neither a scheduler nor a prefix cache has nowhere to put a
     * session (the facade reports MS_NO_CACHE for it too; refusing here avoids
     * decoding a large blob for nothing). */
    if (!entry->sched && !entry->prefix_cache) {
        mr_release(reg, entry); free(tokens);
        hs_send_error(client_fd, 400, "session_caching_disabled",
                      "this model has no session cache to restore into");
        jr_free(root);
        return;
    }
    if (ms_session_principal_check(entry, sid, hs_thread_principal(),
                                   hs_thread_principal_admin(), MS_PRINCIPAL_BIND) != MS_OK) {
        mr_release(reg, entry); free(tokens);
        hs_send_error(client_fd, 403, "session_not_yours", MS_NOT_YOURS_MESSAGE);
        jr_free(root);
        return;
    }

    /* Decode + deserialize the blob, then validate its weights_version — all
     * BEFORE handing it to the facade, so the (potentially large) untrusted
     * parse runs unlocked. The weights_version guard is the one thing that must
     * happen on BOTH paths and before any state is touched: a blob captured
     * under a different model build must never be written into a live slot. */
    size_t blob_len = 0;
    char admit_reason[PN_HOST_ADMIT_REASON_LEN];
    /* Decoded blob (3/4 of the text) + the snapshot it parses into (~the blob
     * again) are both live at once. */
    if (pn_host_admit_check((uint64_t)strlen(blob_b64) / 4 * 3 * 2,
                            "session import", "shorten the session or free "
                            "host memory", admit_reason,
                            sizeof(admit_reason)) != 0) {
        mr_release(reg, entry); free(tokens);
        hs_send_error(client_fd, 503, "server_memory_constrained", admit_reason);
        jr_free(root);
        return;
    }
    unsigned char *blob = pn_base64_decode(blob_b64, &blob_len);
    if (!blob) {
        mr_release(reg, entry); free(tokens);
        hs_send_error(client_fd, 400, "bad_session_blob",
                      "blob is not valid base64; replay the session instead");
        jr_free(root);
        return;
    }
    /* The arch picks the snapshot family, and the two families' blobs carry
     * different magics — so a blob captured on the other family is refused at
     * parse rather than misread as this one. */
    pn_arch arch = entry->model->config.arch;
    uint64_t blob_wv = 0;
    void *snap = pn_session_snapshot_deserialize(arch, blob, blob_len, &blob_wv);
    free(blob);
    if (!snap) {
        mr_release(reg, entry); free(tokens);
        hs_send_error(client_fd, 400, "bad_session_blob",
                      "session blob is corrupt or wrong format; replay instead");
        jr_free(root);
        return;
    }
    if (blob_wv != mr_weights_version(entry)) {
        pn_session_snapshot_free(arch, snap);
        mr_release(reg, entry); free(tokens);
        hs_send_error(client_fd, 409, "weights_version_mismatch",
                      "session blob was captured under different weights; "
                      "replay the session instead");
        jr_free(root);
        return;
    }

    /* Install the state through the session facade, which routes to the store
     * this model actually generates from: the scheduler's warm decode slot on
     * the batched path, the prefix cache on the single-user one. Restoring into
     * the prefix cache unconditionally — what this used to do — reported
     * success on a batched model while writing to a store no batched generation
     * ever reads (and allocating an unused pn_state to hold it). Locking and
     * admission live inside the facade. */
    ms_result msr = ms_session_restore(entry, sid, snap, tokens, n_tok);
    pn_session_snapshot_free(arch, snap);

    if (msr != MS_OK) {
        if (msr == MS_BUSY)
            log_session_capacity_refuse(entry, sid);
        if (msr == MS_BUSY && route.replica > 0) {
            pn_replica_pin_send_full(client_fd, &route, entry,
                                     (int)((double)n_tok / oai_prefill_tok_per_s() + 0.5));
            pn_replica_pin_undo(&route, entry, sid);
        } else if (msr == MS_BUSY) {
            hs_send_error(client_fd, 429, "model_at_session_capacity",
                          "model at session capacity; retry shortly");
            pn_replica_pin_undo(&route, entry, sid);
        } else if (msr == MS_NO_CACHE) {
            hs_send_error(client_fd, 400, "session_caching_disabled",
                          "this model has no session cache to restore into");
        } else if (msr == MS_BAD_SNAPSHOT) {
            hs_send_error(client_fd, 400, "bad_session_blob",
                          "snapshot did not fit this model (dim/pos mismatch); "
                          "replay instead");
        } else {
            hs_send_error(client_fd, 500, "server_error",
                          "session restore failed");
        }
        mr_release(reg, entry); free(tokens);
        jr_free(root);
        return;
    }

    json_writer jw;
    if (!jw_init(&jw)) { mr_release(reg, entry); free(tokens); hs_send_error(client_fd, 500, "server_error", "oom"); jr_free(root); return; }
    jw_raw(&jw, "{\"restored\":true,\"session_id\":");
    jw_string(&jw, sid);
    jw_printf(&jw, ",\"n_tok\":%d}", n_tok);
    hs_send_json(client_fd, 200, jw.data);
    jw_free(&jw);

    mr_release(reg, entry);
    free(tokens);
    jr_free(root);
}

/* POST /session/release {name, session_id} — free a session's serve slot on a
 * client's clean shutdown, returning capacity immediately instead of waiting out
 * the idle-TTL. Goes through the session facade, so a BATCHED session's warm
 * decode slot is actually dropped (it used to clear only the prefix cache, which
 * holds nothing on that path — a departed session stayed bound to its slot).
 * Idempotent: an unknown/already-freed session still returns 200 released:true —
 * The client is best-effort and never retries. */
void oai_handle_session_release(int client_fd, mr_registry *reg,
                                const char *body, const char *session_id)
{
    if (!body) { hs_send_error(client_fd, 400, "invalid_request_error", "empty request body"); return; }
    jr_value *root = jr_parse(body);
    if (!root || jr_kind(root) != JR_OBJECT) {
        hs_send_error(client_fd, 400, "invalid_request_error", "malformed JSON");
        jr_free(root);
        return;
    }
    const char *name = jr_as_string(jr_object_get(root, "name"));
    const char *body_sid = jr_as_string(jr_object_get(root, "session_id"));
    const char *sid = (body_sid && body_sid[0]) ? body_sid
                    : (session_id && session_id[0]) ? session_id : NULL;
    if (!sid) {
        hs_send_error(client_fd, 400, "invalid_request_error",
                      "'session_id' is required");
        jr_free(root);
        return;
    }

    mr_entry *entry = mr_acquire_session(reg, name, sid);
    if (!entry) {
        hs_send_error(client_fd, 404, "model_not_found",
                      "the requested model is not loaded (see GET /v1/models)");
        jr_free(root);
        return;
    }
    pn_replica_pin_announce(entry);
    if (ms_session_principal_check(entry, sid, hs_thread_principal(),
                                   hs_thread_principal_admin(), MS_PRINCIPAL_CHECK) != MS_OK) {
        mr_release(reg, entry);
        hs_send_error(client_fd, 403, "session_not_yours", MS_NOT_YOURS_MESSAGE);
        jr_free(root);
        return;
    }
    /* Nothing to free → the release still "succeeds" (the client's intent —
     * the slot is gone — is satisfied). */
    ms_session_release(entry, sid);
    pn_replica_affinity_forget(sid);

    json_writer jw;
    if (!jw_init(&jw)) { mr_release(reg, entry); hs_send_error(client_fd, 500, "server_error", "oom"); jr_free(root); return; }
    jw_raw(&jw, "{\"released\":true,\"session_id\":");
    jw_string(&jw, sid);
    jw_raw(&jw, "}");
    hs_send_json(client_fd, 200, jw.data);
    jw_free(&jw);

    mr_release(reg, entry);
    jr_free(root);
}

/* ============================================================
 * GET /session/estimate?name=&tokens=N — warm-vs-cold restart estimate (Phase 7)
 * ============================================================ */

/* Answer a session restart estimate: cold (re-prefill N tokens at the measured
 * prefill rate) vs warm (upload a blob of ~N-token size at a client-provided or
 * default LAN rate, plus a host→GPU term). Pure arithmetic from measured
 * constants, so the client UX can show "reload ~2s vs re-prefill ~230s". */
void oai_handle_session_estimate(int client_fd, mr_registry *reg,
                                 const char *name, int tokens,
                                 double blob_bytes, double upload_bytes_per_s)
{
    if (tokens <= 0) {
        hs_send_error(client_fd, 400, "invalid_request_error",
                      "'tokens' query param (> 0) is required");
        return;
    }
    /* Resolve the model just to confirm it exists / echo its id (optional). */
    mr_entry *entry = mr_acquire(reg, name);
    const char *model_id = entry ? entry->name : "";

    double prefill_rate = oai_prefill_tok_per_s();          /* tok/s */
    double cold_s = (double)tokens / prefill_rate;

    /* Warm: blob upload time + a host→GPU upload term. If the caller did not
     * pass blob_bytes, estimate it from the per-token cost measured in Phase 3
     * (saber ~30 MiB / ornith ~130 MiB per session are near-constant; a
     * per-token byte figure is a rough linear proxy for the estimate UX). */
    static const double DEFAULT_UPLOAD_BPS = 100.0e6 / 8.0;  /* ~100 Mb LAN */
    static const double H2GPU_BPS = 3.0e9;                   /* ~3 GB/s PCIe */
    static const double BYTES_PER_TOKEN = 16384.0;           /* ~16 KiB/token proxy */
    double bytes = blob_bytes > 0.0 ? blob_bytes
                                    : (double)tokens * BYTES_PER_TOKEN;
    double up_bps = upload_bytes_per_s > 0.0 ? upload_bytes_per_s
                                             : DEFAULT_UPLOAD_BPS;
    double warm_s = bytes / up_bps + bytes / H2GPU_BPS;

    json_writer jw;
    if (!jw_init(&jw)) { if (entry) mr_release(reg, entry); hs_send_error(client_fd, 500, "server_error", "oom"); return; }
    jw_raw(&jw, "{\"model\":");
    jw_string(&jw, model_id);
    jw_printf(&jw, ",\"tokens\":%d,\"prefill_tok_per_s\":%.2f,"
              "\"cold_reprefill_s\":%.2f,\"warm_restore_s\":%.3f,"
              "\"blob_bytes_est\":%.0f}",
              tokens, prefill_rate, cold_s, warm_s, bytes);
    hs_send_json(client_fd, 200, jw.data);
    jw_free(&jw);
    if (entry)
        mr_release(reg, entry);
}
