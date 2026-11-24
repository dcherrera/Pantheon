/* SPDX-License-Identifier: AGPL-3.0-or-later */
/* Copyright (C) 2026 David (Corey) Herrera */
/**
 * @file serve_prompt_tokens.h
 * @brief The ChatML prompt tokenizer for the serve layer, and the projection of
 *        what a client will resend for the turn the serve just generated.
 *
 * Two halves of ONE contract, which is why they live in one module:
 *
 *   sg_build_prompt_tokens()  turns an OpenAI messages[] array into the token
 *                             stream the model is prefilled with.
 *   sg_project_client_rerender()  predicts the token stream the SAME builder
 *                             will produce next turn for the assistant turn just
 *                             generated.
 *   sg_encode_history_think_block()  the same prediction for the turn's
 *                             `reasoning_content`, for a client that sends
 *                             reasoning BACK (the collapse fix).
 *
 * The prefix cache lives or dies on those two agreeing.  A session can only
 * resume (rather than re-prefill its whole history) when the committed state's
 * client-visible projection is a token-exact prefix of the next request's
 * context — so the projection must not GUESS what the client will send.  It
 * renders the assistant turn through the very same functions the next request
 * will go through (tc_join_assistant_history → sg_encode_assistant_content),
 * which makes agreement structural rather than lucky.
 *
 * Why that matters, concretely: a model's own sampled ids for a piece of text
 * are NOT a function of the text.  "<tool_call>" has two tokenizations in the
 * n2/saber vocab — the special id 248058 and the four ordinary ids
 * "<","tool","_call",">" — and the char-level tool grammar accepts either, so
 * which one lands in the KV is a sampling accident.  Any projection built from
 * the generated ids therefore diverges from the client's deterministic
 * re-render on some turns; one built from the re-render diverges on none.
 *
 * @status Complete.  Extracted from serve_generation.c (2026-08-09) together
 *         with the re-render projection that replaced the byte-offset /
 *         wrapper-guessing projection.  The think prefill is now per-REQUEST
 *         (sg_thinking_suppressed) rather than only process-wide; the env
 *         PANTHEON_NO_THINK remains the default when a request says nothing.
 *         sg_encode_history_think_block extends the same one-encoder rule to
 *         the reasoning channel, so a client that resends `reasoning_content`
 *         resumes instead of re-prefilling.
 * @issues A turn that emits BOTH prose content and a tool call is projected
 *         with the prose included; the client drops it (`content:null`) when it
 *         resends such a turn, so that turn misses the rung and re-prefills.
 *         Fails safe, never wrong.  Not reachable under forced tool_choice
 *         (all prose lands in the reasoning channel).
 * @todo None
 */

#ifndef PN_SERVE_PROMPT_TOKENS_H
#define PN_SERVE_PROMPT_TOKENS_H

#include <stdbool.h>
#include <stdint.h>

#include "serve_generation.h"   /* sg_message */
#include "tool_calling.h"       /* tc_dialect */

struct pn_model;

/**
 * Is this chat style one that prefills a "<think>" opener into the assistant
 * turn (a reasoning model)?
 *
 * @param chat_style  a pn_chat_style value
 * @return  true for the think-prefilling styles
 *
 * Thread-safe: yes (pure)
 */
bool sg_chat_style_is_think(int chat_style);

/**
 * No-Think mode (env PANTHEON_NO_THINK=1), read once and cached.  When set, a
 * hybrid Think/No-Think reasoning model is prefilled with a CLOSED empty think
 * block so it acts directly instead of reasoning.
 *
 * @return  true when the env var is set to "1"
 *
 * Thread-safe: yes (idempotent cache of a getenv)
 */
bool sg_no_think_enabled(void);

/**
 * Resolve one request's thinking control to the single question the rest of the
 * pipeline asks: is this completion's think block prefilled CLOSED?
 *
 * The prompt builder, the response-format grammar's reasoning grace and the
 * stream emitter's starting channel must all get the SAME answer — if they
 * disagree, the model's first generated token is routed to the wrong channel
 * and the answer/tool-call channel comes back empty.  So they all call this.
 *
 * @param mode  the request's resolved sg_thinking_mode
 * @return  true when reasoning is suppressed (prefill "<think>\n\n</think>\n\n")
 *
 * Thread-safe: yes (pure, apart from the cached getenv in sg_no_think_enabled)
 */
bool sg_thinking_suppressed(sg_thinking_mode mode);

/**
 * Build the full prompt token stream from messages[], then open the assistant
 * turn (prefilling "<think>" when the model is a reasoning model).
 *
 * An assistant message that carries `reasoning` renders as
 * "<think>{reasoning}</think>{content}" rather than bare "{content}", which
 * reproduces the token stream this serve committed for that turn.
 *
 * @param model              resident model (tokenizer + chat style + BOS rule)
 * @param messages           the OpenAI messages[] as sg_message
 * @param n_messages         how many
 * @param thinking           the request's thinking control; SG_THINKING_DEFAULT
 *                           reproduces the pre-parameter behavior exactly
 * @param n_out              receives the token count
 * @param sys_len_out        optional; receives the SYSTEM-PREFIX token boundary
 *                           — the leading tokens covering the first message's
 *                           whole turn when it is a system message, else 0.
 *                           Those tokens are byte-identical across a session's
 *                           turns, so the prefix cache can snapshot there (SYS1).
 * @param think_prefill_out  optional; receives the FORCED generation prefill
 *                           length (tokens pushed after "<|im_start|>assistant\n"
 *                           that the model did not choose and a client cannot
 *                           reproduce).
 * @return  Owned token array (caller frees), or NULL on failure.
 *
 * Thread-safe: yes (no shared state beyond the tokenizer, which is read-only)
 */
uint32_t *sg_build_prompt_tokens(struct pn_model *model,
                                 const sg_message *messages, int n_messages,
                                 sg_thinking_mode thinking,
                                 int *n_out, int *sys_len_out,
                                 int *think_prefill_out);

/** Where a built prompt's parts start, in prompt-token positions. */
typedef struct {
    int sys_len;        /* system-prefix boundary (see sg_build_prompt_tokens) */
    int think_prefill;  /* forced generation prefill length after the opener */
    int turn_start;     /* where the assistant opener "<|im_start|>assistant\n"
                         * begins: the end of the client's own messages.  A next
                         * request that drops or rewrites this turn's reply still
                         * starts with prompt[0, turn_start). */
} sg_prompt_bounds;

/**
 * sg_build_prompt_tokens that also reports `turn_start` (sg_prompt_bounds).
 * Same tokens, same sys_len and think_prefill as sg_build_prompt_tokens.
 *
 * @param bounds  receives the boundaries (required)
 * @return  Owned token array (caller frees), or NULL on failure.
 *
 * Thread-safe: yes
 */
uint32_t *sg_build_prompt_tokens_bounded(struct pn_model *model,
                                         const sg_message *messages,
                                         int n_messages,
                                         sg_thinking_mode thinking, int *n_out,
                                         sg_prompt_bounds *bounds);

/**
 * Encode the think block of a HISTORY assistant turn as the prompt builder will
 * render it next turn when the client resends this turn's `reasoning_content`:
 * "<think>" + the emitter-trimmed lead newline + {reasoning} + "</think>" + the
 * trailing blank line, with both tags as SPECIAL ids.
 *
 * This is the reasoning-bearing half of the resume contract.  A client that
 * sends reasoning back cannot reproduce the token ids the model SAMPLED inside
 * its think block — the same "a model's ids are not a function of its text"
 * problem sg_project_client_rerender solves for the answer channel — so the
 * projection is built from the reasoning TEXT the serve streamed, put through
 * the very function build_prompt() will use.  The client therefore only has to
 * echo the reasoning STRING byte-for-byte; the ids are re-derived here.
 *
 * @param model      resident model (tokenizer + chat style)
 * @param reasoning  the `reasoning_content` this turn streamed (the emitter's
 *                   accumulated reasoning channel, verbatim)
 * @param n_out      receives the token count (0 when nothing is projected)
 * @return  Owned token array (caller frees), or NULL when there is no think
 *          block to project — empty reasoning, or a non-think chat style, whose
 *          history turns the builder renders WITHOUT one whatever the client
 *          sends.  The caller then leaves the reasoning-bearing rung disabled.
 *
 * Thread-safe: yes
 */
uint32_t *sg_encode_history_think_block(struct pn_model *model,
                                        const char *reasoning, int *n_out);

/**
 * Encode ASSISTANT-turn content, promoting the tool-call wrapper markers
 * ("<tool_call>" / "</tool_call>") to the special ids the vocab holds for them.
 * Exposed because it is the single definition of "how assistant text becomes
 * tokens" that both the prompt builder and the re-render projection must use —
 * if they ever differ, every tool turn re-prefills its whole conversation.
 *
 * @param model  resident model (for its tokenizer)
 * @param text   the assistant turn's content (NULL/empty → zero tokens)
 * @param n_out  receives the token count
 * @return  Owned token array (caller frees), or NULL on OOM / empty input.
 *
 * Thread-safe: yes
 */
uint32_t *sg_encode_assistant_content(struct pn_model *model, const char *text,
                                      int *n_out);

/**
 * Project what a faithful client will RESEND for the assistant turn the serve
 * has just generated: the serve's own reply, re-rendered into prompt tokens by
 * the same path the next request will take.
 *
 * The turn's reply is `content` plus, when the model emitted a tool call, the
 * calls parsed out of `tool_raw`.  Content and calls are joined with
 * tc_join_assistant_history — the exact function build_messages() will call on
 * the client's message next turn — the template's content |trim is applied as
 * the builder applies it (model->chat_trim_content), and the result is
 * tokenized with sg_encode_assistant_content, the exact function the builder
 * uses.
 * So the projection is not a prediction of the client so much as a replay of the
 * deterministic pipeline the client's message is about to enter.
 *
 * @param model    resident model (tokenizer + dialect target)
 * @param content  the visible content channel of the reply ("" when the whole
 *                 turn was reasoning + a tool call)
 * @param tool_raw the raw diverted tool-call text, or NULL when the turn made
 *                 no call
 * @param tools    borrowed request `tools` array (jr_value*), for schema-typed
 *                 argument parsing; may be NULL
 * @param dialect  the resident model's tool-call dialect
 * @param n_out    receives the token count (0 when there is nothing to project)
 * @return  Owned token array (caller frees), or NULL when the turn has no
 *          reproducible reply — a pure-reasoning runaway, or a tool block that
 *          would not parse back.  The caller must then leave the rung disabled
 *          rather than record a projection it cannot stand behind.
 *
 * Thread-safe: yes
 */
uint32_t *sg_project_client_rerender(struct pn_model *model,
                                     const char *content, const char *tool_raw,
                                     const void *tools, tc_dialect dialect,
                                     int *n_out);

#endif /* PN_SERVE_PROMPT_TOKENS_H */
