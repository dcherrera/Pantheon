/* SPDX-License-Identifier: AGPL-3.0-or-later */
/* Copyright (C) 2026 David (Corey) Herrera */
/*
 * Unit tests for chat template system (src/core/chat.c)
 *
 * Tests style detection, token sequence formatting for ChatML, Mistral, Phi-3.
 * Uses a mock tokenizer with added special tokens to verify correct output.
 */

#include "lib/test.h"
#include "../src/core/chat.h"
#include "../src/core/engine.h"
#include "tokenizer/tokenizer.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

/* ============================================================
 * Style detection tests
 * ============================================================ */

static void test_detect_chatml(void) {
    pn_chat_style s = pn_chat_detect_style(
        "{% for message in messages %}{{'<|im_start|>' + message['role']}}");
    ASSERT_INT_EQ(s, PN_CHAT_CHATML);
}

static void test_detect_mistral(void) {
    pn_chat_style s = pn_chat_detect_style(
        "{% if messages[0]['role'] == 'system' %}{% set system_message = messages[0]['content'] %}{% endif %}{% for message in messages %}{% if message['role'] == 'user' %}{{ '[INST] ' + message['content'] + ' [/INST]' }}{% endif %}{% endfor %}");
    ASSERT_INT_EQ(s, PN_CHAT_MISTRAL);
}

static void test_detect_phi3(void) {
    pn_chat_style s = pn_chat_detect_style(
        "{% for message in messages %}<|user|>\n{{ message['content'] }}<|end|>\n<|assistant|>\n{% endfor %}");
    ASSERT_INT_EQ(s, PN_CHAT_PHI3);
}

static void test_detect_none(void) {
    pn_chat_style s = pn_chat_detect_style("some random template string");
    ASSERT_INT_EQ(s, PN_CHAT_NONE);
}

static void test_detect_null(void) {
    pn_chat_style s = pn_chat_detect_style(NULL);
    ASSERT_INT_EQ(s, PN_CHAT_NONE);
}

/* ============================================================
 * Style from arch tests
 * ============================================================ */

static void test_arch_qwen2(void) {
    ASSERT_INT_EQ(pn_chat_style_from_arch(1), PN_CHAT_CHATML);
}

static void test_arch_llama(void) {
    ASSERT_INT_EQ(pn_chat_style_from_arch(0), PN_CHAT_CHATML);
}

static void test_arch_mistral(void) {
    ASSERT_INT_EQ(pn_chat_style_from_arch(3), PN_CHAT_MISTRAL);
}

static void test_arch_mixtral(void) {
    ASSERT_INT_EQ(pn_chat_style_from_arch(4), PN_CHAT_MISTRAL);
}

static void test_arch_phi(void) {
    ASSERT_INT_EQ(pn_chat_style_from_arch(2), PN_CHAT_PHI3);
}

static void test_arch_unknown(void) {
    ASSERT_INT_EQ(pn_chat_style_from_arch(99), PN_CHAT_NONE);
}

/* ============================================================
 * Style name tests
 * ============================================================ */

static void test_style_name(void) {
    ASSERT_STR_EQ(pn_chat_style_name(PN_CHAT_CHATML), "ChatML");
    ASSERT_STR_EQ(pn_chat_style_name(PN_CHAT_MISTRAL), "Mistral");
    ASSERT_STR_EQ(pn_chat_style_name(PN_CHAT_PHI3), "Phi-3");
    ASSERT_STR_EQ(pn_chat_style_name(PN_CHAT_NONE), "none");
}

/* ============================================================
 * ChatML formatting tests (BPE tokenizer)
 *
 * We build a minimal BPE tokenizer with the special tokens.
 * The exact token IDs depend on insertion order, but we verify
 * that special tokens appear at the right positions.
 * ============================================================ */

static ft_tokenizer* make_chatml_tok(void) {
    ft_tokenizer *tok = ft_new();

    /* Add basic ASCII chars as single-char tokens (GPT-2 style BPE) */
    for (int c = 0; c < 256; c++) {
        char s[2] = {(char)c, 0};
        ft_add_token(tok, s);
    }

    /* Add some common word tokens */
    ft_add_token(tok, "system");
    ft_add_token(tok, "user");
    ft_add_token(tok, "assistant");
    ft_add_token(tok, "Hello");

    /* Add special tokens */
    int im_start = ft_add_token(tok, "<|im_start|>");
    int im_end = ft_add_token(tok, "<|im_end|>");

    ft_set_bos_token(tok, "<|im_start|>");
    ft_set_eos_token(tok, "<|im_end|>");

    (void)im_start;
    (void)im_end;

    return tok;
}

static void test_chatml_first_with_system(void) {
    ft_tokenizer *tok = make_chatml_tok();

    pn_token_seq seq = pn_chat_format_first(PN_CHAT_CHATML, tok, "You are helpful.", "Hi");
    ASSERT_TRUE(seq.tokens != NULL);
    ASSERT_TRUE(seq.n_tokens > 0);

    /* First token should be <|im_start|> */
    int im_start_id = ft_token_to_id(tok, "<|im_start|>");
    ASSERT_INT_EQ(seq.tokens[0], im_start_id);

    /* Should contain <|im_end|> tokens */
    int im_end_id = ft_token_to_id(tok, "<|im_end|>");
    int found_end = 0;
    for (int i = 0; i < seq.n_tokens; i++) {
        if ((int)seq.tokens[i] == im_end_id) found_end++;
    }
    /* 2 im_end tokens: one after system, one after user */
    ASSERT_INT_EQ(found_end, 2);

    /* Last token should be part of "assistant\n" text */
    /* Second to last <|im_start|> is the assistant prefix */
    int n_starts = 0;
    for (int i = 0; i < seq.n_tokens; i++) {
        if ((int)seq.tokens[i] == im_start_id) n_starts++;
    }
    /* 3 im_start tokens: system, user, assistant */
    ASSERT_INT_EQ(n_starts, 3);

    free(seq.tokens);
    ft_free(tok);
}

static void test_chatml_first_no_system(void) {
    ft_tokenizer *tok = make_chatml_tok();

    pn_token_seq seq = pn_chat_format_first(PN_CHAT_CHATML, tok, NULL, "Hi");
    ASSERT_TRUE(seq.tokens != NULL);
    ASSERT_TRUE(seq.n_tokens > 0);

    int im_start_id = ft_token_to_id(tok, "<|im_start|>");
    int im_end_id = ft_token_to_id(tok, "<|im_end|>");

    /* Without system: 2 im_start (user + assistant), 1 im_end (user) */
    int n_starts = 0, n_ends = 0;
    for (int i = 0; i < seq.n_tokens; i++) {
        if ((int)seq.tokens[i] == im_start_id) n_starts++;
        if ((int)seq.tokens[i] == im_end_id) n_ends++;
    }
    ASSERT_INT_EQ(n_starts, 2);
    ASSERT_INT_EQ(n_ends, 1);

    free(seq.tokens);
    ft_free(tok);
}

static void test_chatml_turn(void) {
    ft_tokenizer *tok = make_chatml_tok();

    pn_token_seq seq = pn_chat_format_turn(PN_CHAT_CHATML, tok, "How are you?");
    ASSERT_TRUE(seq.tokens != NULL);
    ASSERT_TRUE(seq.n_tokens > 0);

    int im_start_id = ft_token_to_id(tok, "<|im_start|>");
    int im_end_id = ft_token_to_id(tok, "<|im_end|>");

    /* Turn format: <|im_end|>\n <|im_start|>user\n{text}<|im_end|>\n <|im_start|>assistant\n */
    /* First token should be im_end (closing previous assistant turn) */
    ASSERT_INT_EQ(seq.tokens[0], im_end_id);

    int n_starts = 0, n_ends = 0;
    for (int i = 0; i < seq.n_tokens; i++) {
        if ((int)seq.tokens[i] == im_start_id) n_starts++;
        if ((int)seq.tokens[i] == im_end_id) n_ends++;
    }
    /* 2 im_start (user, assistant), 2 im_end (prev turn close, user close) */
    ASSERT_INT_EQ(n_starts, 2);
    ASSERT_INT_EQ(n_ends, 2);

    free(seq.tokens);
    ft_free(tok);
}

/* ============================================================
 * Phi-3 formatting tests (SentencePiece tokenizer)
 * ============================================================ */

static ft_tokenizer* make_phi3_tok(void) {
    ft_tokenizer *tok = ft_new();
    ft_set_model_type(tok, 1);  /* Unigram / SentencePiece */

    /* Add basic tokens with scores */
    ft_add_token(tok, "<unk>");     /* 0 */
    ft_add_token(tok, "<s>");       /* 1 */
    ft_add_token(tok, "</s>");      /* 2 */

    /* SentencePiece word separator ▁ */
    ft_add_token(tok, "\xe2\x96\x81");  /* 3: ▁ */

    /* Common subwords */
    ft_add_token(tok, "\xe2\x96\x81" "You");    /* 4: ▁You */
    ft_add_token(tok, "\xe2\x96\x81" "are");    /* 5: ▁are */
    ft_add_token(tok, "\xe2\x96\x81" "help");   /* 6 */
    ft_add_token(tok, "ful");                   /* 7 */
    ft_add_token(tok, "\xe2\x96\x81" "Hi");     /* 8: ▁Hi */
    ft_add_token(tok, "\xe2\x96\x81" "Hello");  /* 9 */

    /* Byte fallback tokens */
    for (int b = 0; b < 256; b++) {
        char hex[8];
        snprintf(hex, sizeof(hex), "<0x%02X>", b);
        ft_add_token(tok, hex);
    }

    /* Special tokens for Phi-3 */
    int sys_id = ft_add_token(tok, "<|system|>");
    int usr_id = ft_add_token(tok, "<|user|>");
    int asst_id = ft_add_token(tok, "<|assistant|>");
    int end_id = ft_add_token(tok, "<|end|>");

    (void)sys_id; (void)usr_id; (void)asst_id; (void)end_id;

    /* Single-char tokens for newline etc */
    ft_add_token(tok, "\n");

    ft_set_bos_token(tok, "<s>");
    ft_set_eos_token(tok, "</s>");

    /* Set scores (all zero for simplicity — just need Viterbi to work) */
    int n = ft_vocab_size(tok);
    float *scores = calloc(n, sizeof(float));
    /* Give real tokens higher scores than byte fallbacks */
    for (int i = 3; i < 10; i++) scores[i] = -1.0f;
    for (int i = 10; i < 10 + 256; i++) scores[i] = -100.0f;  /* byte fallback: very low */
    ft_set_scores(tok, scores, n);
    free(scores);

    return tok;
}

static void test_phi3_first_with_system(void) {
    ft_tokenizer *tok = make_phi3_tok();

    pn_token_seq seq = pn_chat_format_first(PN_CHAT_PHI3, tok, "You are helpful.", "Hi");
    ASSERT_TRUE(seq.tokens != NULL);
    ASSERT_TRUE(seq.n_tokens > 0);

    /* First token should be <|system|> */
    int sys_id = ft_token_to_id(tok, "<|system|>");
    ASSERT_INT_EQ(seq.tokens[0], sys_id);

    /* Should contain <|end|> tokens */
    int end_id = ft_token_to_id(tok, "<|end|>");
    int found_end = 0;
    for (int i = 0; i < seq.n_tokens; i++) {
        if ((int)seq.tokens[i] == end_id) found_end++;
    }
    /* 2 <|end|> tokens: after system, after user */
    ASSERT_INT_EQ(found_end, 2);

    /* Should contain <|assistant|> */
    int asst_id = ft_token_to_id(tok, "<|assistant|>");
    int found_asst = 0;
    for (int i = 0; i < seq.n_tokens; i++) {
        if ((int)seq.tokens[i] == asst_id) found_asst++;
    }
    ASSERT_INT_EQ(found_asst, 1);

    free(seq.tokens);
    ft_free(tok);
}

static void test_phi3_turn(void) {
    ft_tokenizer *tok = make_phi3_tok();

    pn_token_seq seq = pn_chat_format_turn(PN_CHAT_PHI3, tok, "Hello");
    ASSERT_TRUE(seq.tokens != NULL);
    ASSERT_TRUE(seq.n_tokens > 0);

    /* First token should be <|end|> (closing previous turn) */
    int end_id = ft_token_to_id(tok, "<|end|>");
    ASSERT_INT_EQ(seq.tokens[0], end_id);

    /* Should have <|user|> and <|assistant|> */
    int usr_id = ft_token_to_id(tok, "<|user|>");
    int asst_id = ft_token_to_id(tok, "<|assistant|>");
    int found_usr = 0, found_asst = 0;
    for (int i = 0; i < seq.n_tokens; i++) {
        if ((int)seq.tokens[i] == usr_id) found_usr++;
        if ((int)seq.tokens[i] == asst_id) found_asst++;
    }
    ASSERT_INT_EQ(found_usr, 1);
    ASSERT_INT_EQ(found_asst, 1);

    free(seq.tokens);
    ft_free(tok);
}

/* ============================================================
 * Mistral formatting tests (SentencePiece tokenizer)
 * ============================================================ */

static ft_tokenizer* make_mistral_tok(void) {
    ft_tokenizer *tok = ft_new();
    ft_set_model_type(tok, 1);  /* Unigram / SentencePiece */

    ft_add_token(tok, "<unk>");   /* 0 */
    ft_add_token(tok, "<s>");     /* 1 */
    ft_add_token(tok, "</s>");    /* 2 */

    ft_add_token(tok, "\xe2\x96\x81");  /* 3: ▁ */
    ft_add_token(tok, "\xe2\x96\x81" "Hello"); /* 4 */
    ft_add_token(tok, "\xe2\x96\x81" "[");     /* 5 */
    ft_add_token(tok, "INST");              /* 6 */
    ft_add_token(tok, "]");                 /* 7 */
    ft_add_token(tok, "\xe2\x96\x81" "[/");    /* 8 */

    /* Byte fallback tokens */
    for (int b = 0; b < 256; b++) {
        char hex[8];
        snprintf(hex, sizeof(hex), "<0x%02X>", b);
        ft_add_token(tok, hex);
    }

    ft_add_token(tok, "\n");

    ft_set_bos_token(tok, "<s>");
    ft_set_eos_token(tok, "</s>");

    int n = ft_vocab_size(tok);
    float *scores = calloc(n, sizeof(float));
    for (int i = 3; i < 9; i++) scores[i] = -1.0f;
    for (int i = 9; i < 9 + 256; i++) scores[i] = -100.0f;
    ft_set_scores(tok, scores, n);
    free(scores);

    return tok;
}

static void test_mistral_first(void) {
    ft_tokenizer *tok = make_mistral_tok();

    pn_token_seq seq = pn_chat_format_first(PN_CHAT_MISTRAL, tok, NULL, "Hello");
    ASSERT_TRUE(seq.tokens != NULL);
    ASSERT_TRUE(seq.n_tokens > 0);

    /* First token should be BOS (id 1) since SentencePiece adds it */
    int bos_id = ft_bos_id(tok);
    ASSERT_INT_EQ(seq.tokens[0], bos_id);

    free(seq.tokens);
    ft_free(tok);
}

static void test_mistral_turn(void) {
    ft_tokenizer *tok = make_mistral_tok();

    pn_token_seq seq = pn_chat_format_turn(PN_CHAT_MISTRAL, tok, "Hello");
    ASSERT_TRUE(seq.tokens != NULL);
    ASSERT_TRUE(seq.n_tokens > 0);

    /* First token should be EOS (</s>) from previous turn */
    int eos_id = ft_eos_id(tok);
    ASSERT_INT_EQ(seq.tokens[0], eos_id);

    free(seq.tokens);
    ft_free(tok);
}

/* ============================================================
 * None (passthrough) style tests
 * ============================================================ */

static void test_none_passthrough(void) {
    ft_tokenizer *tok = make_mistral_tok();  /* any tokenizer works */

    pn_token_seq seq = pn_chat_format_first(PN_CHAT_NONE, tok, NULL, "Hello");
    ASSERT_TRUE(seq.tokens != NULL);
    ASSERT_TRUE(seq.n_tokens > 0);

    /* Should start with BOS for SentencePiece */
    int bos_id = ft_bos_id(tok);
    ASSERT_INT_EQ(seq.tokens[0], bos_id);

    free(seq.tokens);
    ft_free(tok);
}

/* ============================================================
 * Main
 * ============================================================ */

int main(void) {
    printf("\n  chat template tests\n\n");

    /* Style detection */
    RUN_TEST(detect_chatml);
    RUN_TEST(detect_mistral);
    RUN_TEST(detect_phi3);
    RUN_TEST(detect_none);
    RUN_TEST(detect_null);

    /* Arch mapping */
    RUN_TEST(arch_qwen2);
    RUN_TEST(arch_llama);
    RUN_TEST(arch_mistral);
    RUN_TEST(arch_mixtral);
    RUN_TEST(arch_phi);
    RUN_TEST(arch_unknown);

    /* Style names */
    RUN_TEST(style_name);

    /* ChatML formatting */
    RUN_TEST(chatml_first_with_system);
    RUN_TEST(chatml_first_no_system);
    RUN_TEST(chatml_turn);

    /* Phi-3 formatting */
    RUN_TEST(phi3_first_with_system);
    RUN_TEST(phi3_turn);

    /* Mistral formatting */
    RUN_TEST(mistral_first);
    RUN_TEST(mistral_turn);

    /* None passthrough */
    RUN_TEST(none_passthrough);

    TEST_REPORT();
    return __ft_failed > 0 ? 1 : 0;
}
