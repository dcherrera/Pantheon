/* SPDX-License-Identifier: AGPL-3.0-or-later */
/* Copyright (C) 2026 David (Corey) Herrera */
/**
 * @file serve_prompt_tokens.c
 * @brief Implementation of the ChatML prompt tokenizer and the client-re-render
 *        projection (see serve_prompt_tokens.h).
 *
 * @status Complete.
 * @issues None
 * @todo None
 */

#include "serve_prompt_tokens.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "../core/bpe_pretokenize.h"
#include "../core/chat.h"
#include "../core/engine.h"
#include "json_reader.h"
#include "tokenizer/tokenizer.h"

/* ============================================================
 * Token builder — mirrors src/core/chat.c's ChatML(+think) layout, but over a
 * full OpenAI messages[] array (multi-turn history), reusing the same
 * tokenization primitives (special-token lookup + qwen35 BPE text encode).
 * ============================================================ */

typedef struct {
    uint32_t *tokens;
    int len;
    int cap;
} token_builder;

static bool tb_init(token_builder *tb)
{
    tb->cap = 256;
    tb->len = 0;
    tb->tokens = malloc((size_t)tb->cap * sizeof(uint32_t));
    return tb->tokens != NULL;
}

static void tb_push(token_builder *tb, uint32_t id)
{
    if (tb->len >= tb->cap) {
        tb->cap *= 2;
        uint32_t *grown = realloc(tb->tokens, (size_t)tb->cap * sizeof(uint32_t));
        if (!grown) return;
        tb->tokens = grown;
    }
    tb->tokens[tb->len++] = id;
}

static void tb_push_special(token_builder *tb, ft_tokenizer *tok, const char *s)
{
    int id = ft_token_to_id(tok, s);
    if (id >= 0) tb_push(tb, (uint32_t)id);
}

/* Encode plain text with the model's tokenizer (qwen35 BPE for the mini, else
 * the SentencePiece path) and append the ids. */
static void tb_push_text(token_builder *tb, ft_tokenizer *tok, const char *text)
{
    if (!text || !*text) return;
    int n = 0;
    int *ids = (ft_get_model_type(tok) == 0)
                   ? pn_bpe_encode(tok, text, &n)
                   : ft_encode(tok, text, &n);
    if (!ids) return;
    for (int i = 0; i < n; i++)
        if (ids[i] >= 0) tb_push(tb, (uint32_t)ids[i]);
    free(ids);
}

/* The tool-call wrapper.  The vocab holds these as SINGLE special ids
 * (nex-n2-mini: "<tool_call>" is id 248058), and tb_push_text would byte-encode
 * the rendered STRING into the four ordinary ids "<","tool","_call",">".  Both
 * are legal tokenizations of the same characters, so the ONLY thing that matters
 * is that the prompt builder and the re-render projection pick the same one —
 * which they do by both going through sg_encode_assistant_content.  Assistant
 * and system turns only: the system turn carries the tools preamble, whose
 * format instructions spell <tool_call> out, and the chat template tokenizes
 * those as the special ids.  A user/tool message that happens to contain the
 * literal text stays ordinary text. */
static const char *const TOOL_WRAPPER_MARKERS[] = { "<tool_call>", "</tool_call>" };
#define TOOL_WRAPPER_MARKER_COUNT \
    (sizeof(TOOL_WRAPPER_MARKERS) / sizeof(TOOL_WRAPPER_MARKERS[0]))

/* Encode assistant-turn content, emitting any tool-call wrapper marker as its
 * special id.  A marker the vocab does not know (id < 0) is left alone and
 * encodes as ordinary text, so a model without these tokens tokenizes
 * byte-identically to a plain tb_push_text. */
static void tb_push_assistant_content(token_builder *tb, ft_tokenizer *tok,
                                      const char *content)
{
    if (!content || !*content) return;
    const char *pos = content;
    while (*pos) {
        const char *best = NULL;
        size_t best_len = 0;
        int best_id = -1;
        for (size_t i = 0; i < TOOL_WRAPPER_MARKER_COUNT; i++) {
            int id = ft_token_to_id(tok, TOOL_WRAPPER_MARKERS[i]);
            if (id < 0) continue;
            const char *at = strstr(pos, TOOL_WRAPPER_MARKERS[i]);
            if (at && (!best || at < best)) {
                best = at;
                best_len = strlen(TOOL_WRAPPER_MARKERS[i]);
                best_id = id;
            }
        }
        if (!best) {
            tb_push_text(tb, tok, pos);
            return;
        }
        if (best > pos) {
            size_t head_len = (size_t)(best - pos);
            char *head = malloc(head_len + 1);
            if (head) {
                memcpy(head, pos, head_len);
                head[head_len] = '\0';
                tb_push_text(tb, tok, head);
                free(head);
            }
        }
        tb_push(tb, (uint32_t)best_id);
        pos = best + best_len;
    }
}

/* Emit one "<|im_start|>{role}\n{content}<|im_end|>\n" block.  An assistant
 * turn's content goes through tb_push_assistant_content so a rendered tool call
 * reproduces the ids the projection recorded; a system turn does too, so the
 * tools preamble's <tool_call> example is the special id the template gives
 * (see TOOL_WRAPPER_MARKERS). */
static void tb_push_turn(token_builder *tb, ft_tokenizer *tok,
                         const char *role, const char *content)
{
    char header[64];
    snprintf(header, sizeof(header), "%s\n", role);
    tb_push_special(tb, tok, "<|im_start|>");
    tb_push_text(tb, tok, header);
    if (strcmp(role, "assistant") == 0 || strcmp(role, "system") == 0)
        tb_push_assistant_content(tb, tok, content);
    else
        tb_push_text(tb, tok, content);
    tb_push_special(tb, tok, "<|im_end|>");
    tb_push_text(tb, tok, "\n");
}

/* The separator after "</think>" in a history think block, as the chat
 * template writes it. */
static const char THINK_TAIL_WS[] = "\n\n";

/* Whitespace Jinja's |trim strips from both ends of a string. */
static int is_jinja_trim_whitespace(char c)
{
    return c == ' ' || c == '\t' || c == '\n' || c == '\r' || c == '\v' ||
           c == '\f';
}

/* Emit the think block of a HISTORY assistant turn for a client that resent
 * the reasoning: "<think>\n" + trim(reasoning) + "\n</think>\n\n", as the
 * template writes it, so a client that echoes reasoning_content verbatim and
 * one that strips it render the same bytes.  <think> and </think> go in as
 * SPECIAL ids: tb_push_text would byte-encode them as ordinary text and never
 * reach id 248068.  The middle is ONE text push: Qwen's pre-tokenizer joins
 * punctuation with a following newline, so pushing it in pieces can tokenize
 * differently.
 *
 * This is the ONE definition of that span.  Both the prompt builder (below) and
 * the reasoning-bearing resume projection (sg_encode_history_think_block) go
 * through it, for the same reason tb_push_assistant_content is shared — a
 * second implementation is free to drift, and a one-token drift costs a full
 * re-prefill. */
static void tb_push_history_think_block(token_builder *tb, ft_tokenizer *tok,
                                        const char *reasoning)
{
    size_t begin = 0, end = strlen(reasoning);
    while (begin < end && is_jinja_trim_whitespace(reasoning[begin])) begin++;
    while (end > begin && is_jinja_trim_whitespace(reasoning[end - 1])) end--;

    size_t trimmed_len = end - begin;
    char *middle = malloc(trimmed_len + 3); /* "\n" + trimmed + "\n" + NUL */
    tb_push_special(tb, tok, "<think>");
    if (!middle) {
        fprintf(stderr, "serve: think block alloc failed; reasoning sent untrimmed\n");
        tb_push_text(tb, tok, "\n");
        tb_push_text(tb, tok, reasoning);
    } else {
        middle[0] = '\n';
        memcpy(middle + 1, reasoning + begin, trimmed_len);
        middle[trimmed_len + 1] = '\n';
        middle[trimmed_len + 2] = '\0';
        tb_push_text(tb, tok, middle);
        free(middle);
    }
    tb_push_special(tb, tok, "</think>");
    tb_push_text(tb, tok, THINK_TAIL_WS);
}

/* Emit one assistant turn whose reasoning the client resent verbatim:
 * "<|im_start|>assistant\n<think>{reasoning}</think>{content}<|im_end|>\n". */
static void tb_push_assistant_think_turn(token_builder *tb, ft_tokenizer *tok,
                                         const char *reasoning,
                                         const char *content)
{
    tb_push_special(tb, tok, "<|im_start|>");
    tb_push_text(tb, tok, "assistant\n");
    tb_push_history_think_block(tb, tok, reasoning);
    tb_push_assistant_content(tb, tok, content);
    tb_push_special(tb, tok, "<|im_end|>");
    tb_push_text(tb, tok, "\n");
}

/* Encode head + body + tail as ONE text push.  Qwen's pre-tokenizer joins
 * punctuation with a following newline, so pushing the pieces separately can
 * tokenize differently from the template's single string. */
static void tb_push_text_joined(token_builder *tb, ft_tokenizer *tok,
                                const char *head, const char *body,
                                const char *tail)
{
    size_t head_len = strlen(head), body_len = strlen(body), tail_len = strlen(tail);
    char *joined = malloc(head_len + body_len + tail_len + 1);
    if (!joined) {
        fprintf(stderr, "serve: prompt text alloc failed; pushed in pieces\n");
        tb_push_text(tb, tok, head);
        tb_push_text(tb, tok, body);
        tb_push_text(tb, tok, tail);
        return;
    }
    memcpy(joined, head, head_len);
    memcpy(joined + head_len, body, body_len);
    memcpy(joined + head_len + body_len, tail, tail_len + 1);
    tb_push_text(tb, tok, joined);
    free(joined);
}

/* Emit one tool result per the model's chat_template: a tool message is fed
 * back inside a <tool_response> wrapper on a user turn, and CONSECUTIVE tool
 * messages share that one turn:
 *   "<|im_start|>user" + ("\n<tool_response>\n" + content + "\n</tool_response>")...
 *   + "<|im_end|>\n"
 * `opens_turn` / `closes_turn` say whether this result is the first / last of
 * its run.  <tool_response> and </tool_response> are added tokens in the Qwen
 * vocabs (248066 / 248067) that an HF tokenizer splits out of any text, so they
 * go in as their ids when the vocab has both; a vocab without them gets the
 * same characters as one text run. */
static void tb_push_tool_response(token_builder *tb, ft_tokenizer *tok,
                                  const char *content, bool opens_turn,
                                  bool closes_turn)
{
    int open_id = ft_token_to_id(tok, "<tool_response>");
    int close_id = ft_token_to_id(tok, "</tool_response>");
    const char *lead = opens_turn ? "user\n" : "\n";
    if (opens_turn) tb_push_special(tb, tok, "<|im_start|>");
    if (open_id >= 0 && close_id >= 0) {
        tb_push_text(tb, tok, lead);
        tb_push(tb, (uint32_t)open_id);
        tb_push_text_joined(tb, tok, "\n", content, "\n");
        tb_push(tb, (uint32_t)close_id);
    } else {
        char head[32];
        snprintf(head, sizeof(head), "%s<tool_response>\n", lead);
        tb_push_text_joined(tb, tok, head, content, "\n</tool_response>");
    }
    if (closes_turn) {
        tb_push_special(tb, tok, "<|im_end|>");
        tb_push_text(tb, tok, "\n");
    }
}

/* A copy of `text` with Jinja |trim applied (the whitespace
 * is_jinja_trim_whitespace names, from both ends).  Owned; NULL on OOM. */
static char *dup_jinja_trimmed(const char *text)
{
    size_t begin = 0, end = strlen(text);
    while (begin < end && is_jinja_trim_whitespace(text[begin])) begin++;
    while (end > begin && is_jinja_trim_whitespace(text[end - 1])) end--;
    char *trimmed = malloc(end - begin + 1);
    if (!trimmed) return NULL;
    memcpy(trimmed, text + begin, end - begin);
    trimmed[end - begin] = '\0';
    return trimmed;
}

/* A message's content as the model's template renders it: |trim'd when the
 * template trims (model->chat_trim_content), else as sent.  Returns an owned
 * copy in *owned (free it) when it trimmed, NULL otherwise. */
static const char *template_content(const struct pn_model *model,
                                    const char *content, char **owned)
{
    *owned = NULL;
    if (!model->chat_trim_content || !content) return content;
    *owned = dup_jinja_trimmed(content);
    if (!*owned) {
        fprintf(stderr, "serve: content trim alloc failed; sent untrimmed\n");
        return content;
    }
    return *owned;
}

bool sg_no_think_enabled(void)
{
    static int cached = -1;
    if (cached < 0) {
        const char *e = getenv("PANTHEON_NO_THINK");
        cached = (e && e[0] == '1') ? 1 : 0;
    }
    return cached != 0;
}

bool sg_chat_style_is_think(int chat_style)
{
    return chat_style == PN_CHAT_CHATML_THINK ||
           chat_style == PN_CHAT_CHATML_THINK_NL;
}

bool sg_thinking_suppressed(sg_thinking_mode mode)
{
    if (mode == SG_THINKING_OFF) return true;
    if (mode == SG_THINKING_ON)  return false;
    return sg_no_think_enabled();      /* SG_THINKING_DEFAULT → the env decides */
}

/* Push the forced generation prefill for the assistant turn about to be
 * generated — the tokens the model did not choose.  Returns how many. */
static int tb_push_generation_prefill(token_builder *tb, ft_tokenizer *tok,
                                      int chat_style, sg_thinking_mode thinking)
{
    int before = tb->len;
    if (sg_chat_style_is_think(chat_style) && sg_thinking_suppressed(thinking)) {
        /* No-Think: prefill a CLOSED empty think block so a hybrid
         * Think/No-Think model (MiniCPM5, saber-orch) skips reasoning and acts
         * DIRECTLY.  These four pushes are byte-for-byte the Qwen3.x template's
         * enable_thinking=false serialization ("<think>\n\n</think>\n\n"), which
         * is the only reasoning control that family's template exposes — hence a
         * BINARY switch and not a graded effort level.  Requested per-request via
         * `enable_thinking`, or process-wide via PANTHEON_NO_THINK. */
        tb_push_special(tb, tok, "<think>");
        tb_push_text(tb, tok, "\n\n");
        tb_push_special(tb, tok, "</think>");
        tb_push_text(tb, tok, "\n\n");
    } else if (chat_style == PN_CHAT_CHATML_THINK) {
        tb_push_special(tb, tok, "<think>");
    } else if (chat_style == PN_CHAT_CHATML_THINK_NL) {
        /* Bonsai-27B: the model's own template prefills '<think>\n' — the
         * newline is part of the trained prefill.  Prefilling bare <think>
         * leaves "\n\n</think>\n\n" (the template's NON-thinking serialization)
         * as the natural greedy continuation, so the model deterministically
         * closes the think block empty (Phase 3b bug). */
        tb_push_special(tb, tok, "<think>");
        tb_push_text(tb, tok, "\n");
    }
    return tb->len - before;
}

uint32_t *sg_build_prompt_tokens(struct pn_model *model,
                                 const sg_message *messages, int n_messages,
                                 sg_thinking_mode thinking,
                                 int *n_out, int *sys_len_out,
                                 int *think_prefill_out)
{
    sg_prompt_bounds bounds;
    uint32_t *tokens = sg_build_prompt_tokens_bounded(model, messages,
                                                      n_messages, thinking,
                                                      n_out, &bounds);
    if (sys_len_out) *sys_len_out = tokens ? bounds.sys_len : 0;
    if (think_prefill_out) *think_prefill_out = tokens ? bounds.think_prefill : 0;
    return tokens;
}

/* Is messages[index] a tool result?  False outside [0, n_messages). */
static bool message_is_tool(const sg_message *messages, int n_messages, int index)
{
    if (index < 0 || index >= n_messages) return false;
    return messages[index].role && strcmp(messages[index].role, "tool") == 0;
}

uint32_t *sg_build_prompt_tokens_bounded(struct pn_model *model,
                                         const sg_message *messages,
                                         int n_messages,
                                         sg_thinking_mode thinking, int *n_out,
                                         sg_prompt_bounds *bounds)
{
    ft_tokenizer *tok = model->tokenizer;
    token_builder tb;
    memset(bounds, 0, sizeof(*bounds));
    if (!tb_init(&tb)) return NULL;

    /* Templates that lead with {{ bos_token }} (e.g. MiniCPM5) need BOS at
     * position 0 — the model was trained with it and degenerates to whitespace
     * without it.  Detected at load (model->chat_add_bos). */
    if (model->chat_add_bos) {
        int bid = ft_bos_id(tok);
        if (bid >= 0) tb_push(&tb, (uint32_t)bid);
    }

    for (int i = 0; i < n_messages; i++) {
        const char *role = messages[i].role ? messages[i].role : "user";
        char *trimmed = NULL;
        const char *content = template_content(
            model, messages[i].content ? messages[i].content : "", &trimmed);
        const char *reasoning = messages[i].reasoning;
        if (strcmp(role, "tool") == 0)
            tb_push_tool_response(&tb, tok, content,
                                  !message_is_tool(messages, n_messages, i - 1),
                                  !message_is_tool(messages, n_messages, i + 1));
        else if (reasoning && *reasoning && strcmp(role, "assistant") == 0 &&
                 sg_chat_style_is_think(model->chat_style))
            tb_push_assistant_think_turn(&tb, tok, reasoning, content);
        else
            tb_push_turn(&tb, tok, role, content);
        free(trimmed);
        if (i == 0 && strcmp(role, "system") == 0)
            bounds->sys_len = tb.len;
    }

    bounds->turn_start = tb.len;
    tb_push_special(&tb, tok, "<|im_start|>");
    tb_push_text(&tb, tok, "assistant\n");
    bounds->think_prefill = tb_push_generation_prefill(&tb, tok,
                                                       model->chat_style,
                                                       thinking);
    *n_out = tb.len;
    return tb.tokens;
}

uint32_t *sg_encode_history_think_block(struct pn_model *model,
                                        const char *reasoning, int *n_out)
{
    *n_out = 0;
    if (!model || !reasoning || !reasoning[0]) return NULL;
    /* Match sg_build_prompt_tokens exactly: a non-think chat style renders an
     * assistant history turn WITHOUT a think block whatever `reasoning` says, so
     * projecting one here would predict a prompt the builder never produces. */
    if (!sg_chat_style_is_think(model->chat_style)) return NULL;
    token_builder tb;
    if (!tb_init(&tb)) return NULL;
    tb_push_history_think_block(&tb, model->tokenizer, reasoning);
    if (tb.len <= 0) {
        free(tb.tokens);
        return NULL;
    }
    *n_out = tb.len;
    return tb.tokens;
}

uint32_t *sg_encode_assistant_content(struct pn_model *model, const char *text,
                                      int *n_out)
{
    *n_out = 0;
    if (!model || !text || !text[0]) return NULL;
    token_builder tb;
    if (!tb_init(&tb)) return NULL;
    tb_push_assistant_content(&tb, model->tokenizer, text);
    if (tb.len <= 0) {
        free(tb.tokens);
        return NULL;
    }
    *n_out = tb.len;
    return tb.tokens;
}

/* ============================================================
 * Client-re-render projection
 * ============================================================ */

/* Re-render the serve's OWN parsed calls, with this turn's content, into
 * assistant-history syntax through the very function the next request's
 * `tool_calls` array will go through (tc_join_assistant_history, reached from
 * combine_assistant_tool_calls in openai_chat.c).
 *
 * The detour via JSON is deliberate and is the whole point: the calls are put
 * back into the exact OpenAI wire shape the reply is about to carry, then handed
 * to the shared join.  Rendering them any other way would be a SECOND
 * implementation of the history syntax, free to drift from the one that
 * actually runs next turn — and a one-token drift costs a full re-prefill.
 * Returns an owned string, or NULL on OOM / no calls. */
static char *render_own_turn_as_history(const char *content,
                                        const tc_tool_call *calls, int n,
                                        tc_dialect dialect)
{
    if (!calls || n <= 0) return NULL;
    json_writer jw;
    if (!jw_init(&jw)) return NULL;
    jw_raw(&jw, "[");
    for (int i = 0; i < n; i++) {
        if (i) jw_raw(&jw, ",");
        jw_raw(&jw, "{\"id\":");
        jw_string(&jw, calls[i].id ? calls[i].id : "");
        jw_raw(&jw, ",\"type\":\"function\",\"function\":{\"name\":");
        jw_string(&jw, calls[i].name ? calls[i].name : "");
        jw_raw(&jw, ",\"arguments\":");
        jw_string(&jw, calls[i].arguments ? calls[i].arguments : "{}");
        jw_raw(&jw, "}}");
    }
    jw_raw(&jw, "]");
    char *rendered = NULL;
    if (!jw.oom) {
        jr_value *dom = jr_parse(jw.data);
        rendered = tc_join_assistant_history(content, dom, dialect);
        jr_free(dom);
    }
    jw_free(&jw);
    return rendered;
}

uint32_t *sg_project_client_rerender(struct pn_model *model,
                                     const char *content, const char *tool_raw,
                                     const void *tools, tc_dialect dialect,
                                     int *n_out)
{
    *n_out = 0;
    if (!model) return NULL;
    if (!content) content = "";

    char *joined = NULL;
    if (tool_raw && tool_raw[0]) {
        tc_tool_call *calls = NULL;
        int n_calls = 0;
        if (tc_parse_tool_calls(tool_raw, (const jr_value *)tools, dialect,
                                &calls, &n_calls) == 0 && n_calls > 0)
            joined = render_own_turn_as_history(content, calls, n_calls, dialect);
        tc_tool_calls_free(calls, n_calls);
        /* A diverted block that will not parse back is a turn whose re-render we
         * cannot predict — disable the rung rather than record a guess. */
        if (!joined) return NULL;
    }

    /* The prompt builder then applies the template's content |trim to the
     * whole message (template_content); do the same here. */
    char *trimmed = NULL;
    const char *text = template_content(model, joined ? joined : content, &trimmed);
    uint32_t *ids = sg_encode_assistant_content(model, text, n_out);
    free(trimmed);
    free(joined);
    return ids;
}
