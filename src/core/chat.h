/* SPDX-License-Identifier: AGPL-3.0-or-later */
/* Copyright (C) 2026 David (Corey) Herrera */
#ifndef PN_CHAT_H
#define PN_CHAT_H

#include <stdbool.h>
#include <stdint.h>
#include "tokenizer/tokenizer.h"

/* Supported chat template styles */
typedef enum {
    PN_CHAT_CHATML,        /* Qwen2.5, SmolLM2 */
    PN_CHAT_CHATML_THINK,  /* ChatML + reasoning prefill (Nex-N2, Qwen3): assistant turn opens with <think> */
    PN_CHAT_MISTRAL,       /* Mistral, Mixtral */
    PN_CHAT_PHI3,          /* Phi-3 */
    PN_CHAT_NONE,          /* Raw passthrough */
    PN_CHAT_CHATML_THINK_NL, /* ChatML + reasoning prefill "<think>\n" (Bonsai-27B):
                              * the model's own template opens the assistant turn
                              * with {{- '<think>\n' }} — the newline is part of the
                              * trained prefill. Prefilling bare <think> (the _THINK
                              * form) leaves "\n\n</think>\n\n" — the template's
                              * NON-thinking serialization — as the natural greedy
                              * continuation, so the model deterministically closes
                              * the think block empty and reasoning is forfeited. */
} pn_chat_style;

/* Token sequence (caller frees tokens) */
typedef struct {
    uint32_t *tokens;
    int n_tokens;
} pn_token_seq;

/* Detect chat style from Jinja2 template string (GGUF metadata) */
pn_chat_style pn_chat_detect_style(const char *jinja_template);

/* Does the template |trim message content (Qwen3.5+ family, Ornith)?  The
 * prompt builder then trims every message's content as the template does;
 * templates without |trim (Qwen2.5, Qwen3) keep their content as sent. */
bool pn_chat_template_trims_content(const char *jinja_template);

/* Fallback: detect from architecture enum value */
pn_chat_style pn_chat_style_from_arch(int arch);

/* Format first turn: system (optional) + user + assistant prefix → token IDs */
pn_token_seq pn_chat_format_first(pn_chat_style style, ft_tokenizer *tok,
                                   const char *system, const char *user);

/* Format turn 1+: EOT + user + assistant prefix → token IDs */
pn_token_seq pn_chat_format_turn(pn_chat_style style, ft_tokenizer *tok,
                                  const char *user);

/* Human-readable style name */
const char* pn_chat_style_name(pn_chat_style style);

#endif /* PN_CHAT_H */
