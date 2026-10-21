/* SPDX-License-Identifier: AGPL-3.0-or-later */
/* Copyright (C) 2026 David (Corey) Herrera */
#include "chat.h"
#include "engine.h"
#include "bpe_pretokenize.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

/* ============================================================
 * Sequence builder — accumulates token IDs
 * ============================================================ */

typedef struct {
    uint32_t *tokens;
    int len;
    int cap;
} seq_builder;

static void seq_init(seq_builder *s) {
    s->cap = 256;
    s->tokens = malloc(s->cap * sizeof(uint32_t));
    s->len = 0;
}

static void seq_push_id(seq_builder *s, int id) {
    if (s->len >= s->cap) {
        s->cap *= 2;
        s->tokens = realloc(s->tokens, s->cap * sizeof(uint32_t));
    }
    s->tokens[s->len++] = (uint32_t)id;
}

static void seq_push_ids(seq_builder *s, int *ids, int n) {
    for (int i = 0; i < n; i++) {
        seq_push_id(s, ids[i]);
    }
}

/*
 * Push a special token by looking up its string in the vocab.
 * Uses ft_token_to_id which does exact hash lookup.
 */
static void seq_push_special(seq_builder *s, ft_tokenizer *tok, const char *token_str) {
    int id = ft_token_to_id(tok, token_str);
    if (id < 0) {
        fprintf(stderr, "chat: special token '%s' not found in vocab\n", token_str);
        return;
    }
    seq_push_id(s, id);
}

/*
 * Push a text segment, encoding it properly for the tokenizer type.
 * BPE models need GPT-2 byte encoding first.
 * SentencePiece models encode directly (no leading space added here).
 */
static void seq_push_text(seq_builder *s, ft_tokenizer *tok, const char *text) {
    if (!text || !*text) return;

    int is_bpe = (ft_get_model_type(tok) == 0);
    int n = 0;
    int *ids;

    if (is_bpe) {
        /* qwen35 pre-tokenization + GPT-2 byte BPE (matches llama.cpp). */
        ids = pn_bpe_encode(tok, text, &n);
    } else {
        ids = ft_encode(tok, text, &n);
    }

    if (ids) {
        seq_push_ids(s, ids, n);
        free(ids);
    }
}

/*
 * Push text encoded via SentencePiece path — adds leading space for word boundary.
 * Used for the initial text in Mistral format where SentencePiece BOS handling
 * is done separately.
 */
static void seq_push_text_sp(seq_builder *s, ft_tokenizer *tok, const char *text, int add_bos) {
    if (!text || !*text) return;

    int is_bpe = (ft_get_model_type(tok) == 0);
    int n = 0;
    int *ids;

    if (is_bpe) {
        char *encoded = gpt2_bytes_encode(text);
        if (!encoded) return;
        ids = ft_encode_with_special(tok, encoded, &n, add_bos, false);
        free(encoded);
    } else {
        ids = ft_encode_with_special(tok, text, &n, add_bos, false);
    }

    if (ids) {
        seq_push_ids(s, ids, n);
        free(ids);
    }
}

static pn_token_seq seq_finish(seq_builder *s) {
    pn_token_seq result;
    result.tokens = s->tokens;
    result.n_tokens = s->len;
    return result;
}

/* ============================================================
 * Style detection
 * ============================================================ */

pn_chat_style pn_chat_detect_style(const char *jinja_template) {
    if (!jinja_template) return PN_CHAT_NONE;

    if (strstr(jinja_template, "<|im_start|>")) {
        /* Reasoning models (Nex-N2, Qwen3) prefill the assistant turn with
         * <think> via `add_generation_prompt`; match that for oracle parity.
         *
         * The prefill must match the MODEL'S OWN template exactly:
         *   Nex-N2:  {{- '<think>' }}    -> prefill <think>        (_THINK)
         *   Bonsai:  {{- '<think>\n' }}  -> prefill <think> + "\n" (_THINK_NL)
         * The distinguishing literal is the Jinja string '<think>\n' with the
         * escaped newline immediately before the closing quote — Nex-N2's
         * non-thinking branch '<think>\n\n</think>\n\n' does NOT match it. */
        if (strstr(jinja_template, "<think>")) {
            if (strstr(jinja_template, "'<think>\\n'"))
                return PN_CHAT_CHATML_THINK_NL;
            return PN_CHAT_CHATML_THINK;
        }
        return PN_CHAT_CHATML;
    }
    if (strstr(jinja_template, "[INST]"))        return PN_CHAT_MISTRAL;
    if (strstr(jinja_template, "<|system|>") ||
        strstr(jinja_template, "<|user|>"))      return PN_CHAT_PHI3;

    return PN_CHAT_NONE;
}

pn_chat_style pn_chat_style_from_arch(int arch) {
    switch (arch) {
        case 7:  /* PN_ARCH_QWEN3 */
        case 1:  /* PN_ARCH_QWEN2 */
        case 0:  /* PN_ARCH_LLAMA (SmolLM2 uses llama arch) */
            return PN_CHAT_CHATML;
        case 3:  /* PN_ARCH_MISTRAL */
        case 4:  /* PN_ARCH_MIXTRAL */
            return PN_CHAT_MISTRAL;
        case 2:  /* PN_ARCH_PHI */
            return PN_CHAT_PHI3;
        default:
            return PN_CHAT_NONE;
    }
}

const char* pn_chat_style_name(pn_chat_style style) {
    switch (style) {
        case PN_CHAT_CHATML:       return "ChatML";
        case PN_CHAT_CHATML_THINK: return "ChatML+think";
        case PN_CHAT_CHATML_THINK_NL: return "ChatML+think\\n";
        case PN_CHAT_MISTRAL: return "Mistral";
        case PN_CHAT_PHI3:    return "Phi-3";
        case PN_CHAT_NONE:    return "none";
    }
    return "unknown";
}

/* ============================================================
 * ChatML format
 *
 * Turn 0: <|im_start|>system\n{sys}<|im_end|>\n
 *         <|im_start|>user\n{user}<|im_end|>\n
 *         <|im_start|>assistant\n
 *
 * Turn 1+: <|im_end|>\n
 *          <|im_start|>user\n{user}<|im_end|>\n
 *          <|im_start|>assistant\n
 * ============================================================ */

/* think: append the <think> reasoning prefill after the assistant header,
 * matching the model's `add_generation_prompt` template branch.
 * think == 2 (_THINK_NL): the model's template prefills '<think>\n' — the
 * newline is part of the trained prefill and steers greedy decode INTO the
 * reasoning branch (see PN_CHAT_CHATML_THINK_NL in chat.h). */
static pn_token_seq format_chatml_first(ft_tokenizer *tok,
                                         const char *system, const char *user,
                                         int think) {
    seq_builder s;
    seq_init(&s);

    if (system && *system) {
        seq_push_special(&s, tok, "<|im_start|>");
        seq_push_text(&s, tok, "system\n");
        seq_push_text(&s, tok, system);
        seq_push_special(&s, tok, "<|im_end|>");
        seq_push_text(&s, tok, "\n");
    }

    seq_push_special(&s, tok, "<|im_start|>");
    seq_push_text(&s, tok, "user\n");
    seq_push_text(&s, tok, user);
    seq_push_special(&s, tok, "<|im_end|>");
    seq_push_text(&s, tok, "\n");

    seq_push_special(&s, tok, "<|im_start|>");
    seq_push_text(&s, tok, "assistant\n");
    if (think) {
        seq_push_special(&s, tok, "<think>");
        if (think == 2)
            seq_push_text(&s, tok, "\n");
    }

    return seq_finish(&s);
}

static pn_token_seq format_chatml_turn(ft_tokenizer *tok, const char *user,
                                       int think) {
    seq_builder s;
    seq_init(&s);

    seq_push_special(&s, tok, "<|im_end|>");
    seq_push_text(&s, tok, "\n");

    seq_push_special(&s, tok, "<|im_start|>");
    seq_push_text(&s, tok, "user\n");
    seq_push_text(&s, tok, user);
    seq_push_special(&s, tok, "<|im_end|>");
    seq_push_text(&s, tok, "\n");

    seq_push_special(&s, tok, "<|im_start|>");
    seq_push_text(&s, tok, "assistant\n");
    if (think) {
        seq_push_special(&s, tok, "<think>");
        if (think == 2)
            seq_push_text(&s, tok, "\n");
    }

    return seq_finish(&s);
}

/* ============================================================
 * Mistral format
 *
 * Turn 0: [BOS] [INST] {user} [/INST]
 * Turn 1+: </s> [INST] {user} [/INST]
 *
 * [INST] and [/INST] are plain text, not special tokens.
 * BOS is added by SentencePiece, </s> is the EOS special token.
 * ============================================================ */

static pn_token_seq format_mistral_first(ft_tokenizer *tok, const char *user) {
    seq_builder s;
    seq_init(&s);

    /* Build the text: "[INST] {user} [/INST]" and let tokenizer add BOS */
    size_t ulen = strlen(user);
    char *text = malloc(8 + ulen + 9 + 1);
    sprintf(text, "[INST] %s [/INST]", user);

    seq_push_text_sp(&s, tok, text, 1);  /* add_bos=1 for SentencePiece */
    free(text);

    return seq_finish(&s);
}

static pn_token_seq format_mistral_turn(ft_tokenizer *tok, const char *user) {
    seq_builder s;
    seq_init(&s);

    /* EOS token to end previous assistant turn */
    int eos_id = ft_eos_id(tok);
    if (eos_id >= 0) seq_push_id(&s, eos_id);

    /* " [INST] {user} [/INST]" — space before [INST] for SentencePiece */
    size_t ulen = strlen(user);
    char *text = malloc(9 + ulen + 9 + 1);
    sprintf(text, " [INST] %s [/INST]", user);

    seq_push_text(&s, tok, text);
    free(text);

    return seq_finish(&s);
}

/* ============================================================
 * Phi-3 format
 *
 * Turn 0: <|system|>\n{sys}<|end|>\n
 *         <|user|>\n{user}<|end|>\n
 *         <|assistant|>\n
 *
 * Turn 1+: <|end|>\n
 *          <|user|>\n{user}<|end|>\n
 *          <|assistant|>\n
 * ============================================================ */

static pn_token_seq format_phi3_first(ft_tokenizer *tok,
                                       const char *system, const char *user) {
    seq_builder s;
    seq_init(&s);

    if (system && *system) {
        seq_push_special(&s, tok, "<|system|>");
        seq_push_text(&s, tok, "\n");
        seq_push_text(&s, tok, system);
        seq_push_special(&s, tok, "<|end|>");
        seq_push_text(&s, tok, "\n");
    }

    seq_push_special(&s, tok, "<|user|>");
    seq_push_text(&s, tok, "\n");
    seq_push_text(&s, tok, user);
    seq_push_special(&s, tok, "<|end|>");
    seq_push_text(&s, tok, "\n");

    seq_push_special(&s, tok, "<|assistant|>");
    seq_push_text(&s, tok, "\n");

    return seq_finish(&s);
}

static pn_token_seq format_phi3_turn(ft_tokenizer *tok, const char *user) {
    seq_builder s;
    seq_init(&s);

    seq_push_special(&s, tok, "<|end|>");
    seq_push_text(&s, tok, "\n");

    seq_push_special(&s, tok, "<|user|>");
    seq_push_text(&s, tok, "\n");
    seq_push_text(&s, tok, user);
    seq_push_special(&s, tok, "<|end|>");
    seq_push_text(&s, tok, "\n");

    seq_push_special(&s, tok, "<|assistant|>");
    seq_push_text(&s, tok, "\n");

    return seq_finish(&s);
}

/* ============================================================
 * Public API
 * ============================================================ */

pn_token_seq pn_chat_format_first(pn_chat_style style, ft_tokenizer *tok,
                                   const char *system, const char *user) {
    switch (style) {
        case PN_CHAT_CHATML:
            return format_chatml_first(tok, system, user, 0);
        case PN_CHAT_CHATML_THINK:
            return format_chatml_first(tok, system, user, 1);
        case PN_CHAT_CHATML_THINK_NL:
            return format_chatml_first(tok, system, user, 2);
        case PN_CHAT_MISTRAL:
            return format_mistral_first(tok, user);
        case PN_CHAT_PHI3:
            return format_phi3_first(tok, system, user);
        case PN_CHAT_NONE: {
            /* Passthrough: just encode the user text with BOS */
            seq_builder s;
            seq_init(&s);
            seq_push_text_sp(&s, tok, user, 1);
            return seq_finish(&s);
        }
    }
    /* Unreachable, but satisfy compiler */
    pn_token_seq empty = {NULL, 0};
    return empty;
}

pn_token_seq pn_chat_format_turn(pn_chat_style style, ft_tokenizer *tok,
                                  const char *user) {
    switch (style) {
        case PN_CHAT_CHATML:
            return format_chatml_turn(tok, user, 0);
        case PN_CHAT_CHATML_THINK:
            return format_chatml_turn(tok, user, 1);
        case PN_CHAT_CHATML_THINK_NL:
            return format_chatml_turn(tok, user, 2);
        case PN_CHAT_MISTRAL:
            return format_mistral_turn(tok, user);
        case PN_CHAT_PHI3:
            return format_phi3_turn(tok, user);
        case PN_CHAT_NONE: {
            seq_builder s;
            seq_init(&s);
            seq_push_text(&s, tok, user);
            return seq_finish(&s);
        }
    }
    pn_token_seq empty = {NULL, 0};
    return empty;
}
