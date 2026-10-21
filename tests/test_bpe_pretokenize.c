/* SPDX-License-Identifier: AGPL-3.0-or-later */
/* Copyright (C) 2026 David (Corey) Herrera */
/*
 * Unit tests for qwen35 BPE pre-tokenization (src/core/bpe_pretokenize.c).
 *
 * These verify that BPE merges respect pre-token boundaries the way
 * llama.cpp's qwen35 pre-tokenizer regex does — the property that gates
 * tokenizer parity with the llama.cpp oracle (build step A7).
 *
 * We use a synthetic GPT-2-style BPE tokenizer: every byte maps to its
 * GPT-2 codepoint as a single-char token, plus a few merge rules. We then
 * assert which merges fire, which directly observes the pre-token split:
 *   - A "ĠĠ" (double-space) merge must NOT fire across a "  word" boundary,
 *     because qwen35 splits "  word" into " " + " word" (the trailing space
 *     joins the following word), so the two spaces are never adjacent in one
 *     pre-token.
 *   - It MUST fire for a pure run of spaces with no following word.
 */

#include "lib/test.h"
#include "../src/core/bpe_pretokenize.h"
#include "../src/core/engine.h"
#include "tokenizer/tokenizer.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

/* GPT-2 codepoint for a raw space (0x20 -> U+0120 'Ġ'). */
#define GP_SPACE 0x0120
/* GPT-2 codepoint for a raw newline (0x0A -> U+010A 'Ċ'). */
#define GP_NEWLINE 0x010A

/* Build a synthetic BPE tokenizer: token id == GPT-2 codepoint for every
 * single byte, plus optional merges added by the caller afterwards. */
static ft_tokenizer *make_byte_tok(void) {
    ft_tokenizer *tok = ft_new();
    ft_set_model_type(tok, 0); /* BPE */
    /* Add the 256 GPT-2 single-codepoint tokens in codepoint order so that
     * ft_token_to_id returns a stable, known id per codepoint. We add them
     * as the UTF-8 of each GPT-2 codepoint. */
    for (int b = 0; b < 256; b++) {
        char raw[2] = {(char)b, 0};
        char *enc = gpt2_bytes_encode(raw); /* 1 codepoint, 1-2 UTF-8 bytes */
        ft_add_token(tok, enc);
        free(enc);
    }
    return tok;
}

/* Count how many ids in the output equal `id`. */
static int count_id(const int *ids, int n, int id) {
    int c = 0;
    for (int i = 0; i < n; i++) if (ids[i] == id) c++;
    return c;
}

/* The id of a token whose string is the GPT-2 encoding of `raw`. */
static int id_of_raw(ft_tokenizer *tok, const char *raw) {
    char *enc = gpt2_bytes_encode(raw);
    int id = ft_token_to_id(tok, enc);
    free(enc);
    return id;
}

static void test_double_space_does_not_cross_word(void) {
    ft_tokenizer *tok = make_byte_tok();
    /* Add a merge "Ġ Ġ" -> "ĠĠ" (two spaces). */
    char *sp = gpt2_bytes_encode(" ");
    ft_add_merge(tok, sp, sp);
    int dbl = id_of_raw(tok, "  "); /* the merged double-space token */
    int single = id_of_raw(tok, " ");
    free(sp);

    /* "a  b": qwen35 splits as "a" + " " + " b". The two spaces are in
     * different pre-tokens, so the ĠĠ merge must NOT fire. */
    int n = 0;
    int *ids = pn_bpe_encode_qwen35(tok, "a  b", &n);
    ASSERT_INT_EQ(count_id(ids, n, dbl), 0);
    ASSERT_INT_EQ(count_id(ids, n, single), 2);
    free(ids);
    ft_free(tok);
}

static void test_double_space_run_at_end_merges(void) {
    ft_tokenizer *tok = make_byte_tok();
    char *sp = gpt2_bytes_encode(" ");
    ft_add_merge(tok, sp, sp);
    int dbl = id_of_raw(tok, "  ");
    free(sp);

    /* "a  " (trailing run, no following word): the two spaces stay in one
     * pre-token (\s+ at end of string) and DO merge. */
    int n = 0;
    int *ids = pn_bpe_encode_qwen35(tok, "a  ", &n);
    ASSERT_INT_EQ(count_id(ids, n, dbl), 1);
    free(ids);
    ft_free(tok);
}

static void test_single_leading_space_joins_word(void) {
    ft_tokenizer *tok = make_byte_tok();
    /* Merge "Ġ w" so " w" becomes one token when they are adjacent. */
    char *sp = gpt2_bytes_encode(" ");
    char *w = gpt2_bytes_encode("w");
    ft_add_merge(tok, sp, w);
    char *spw_enc = gpt2_bytes_encode(" w");
    int spw = ft_token_to_id(tok, spw_enc);
    free(spw_enc);
    free(sp); free(w);

    /* " world" — single leading space attaches to the word's first char,
     * so the "Ġw" merge fires. */
    int n = 0;
    int *ids = pn_bpe_encode_qwen35(tok, " world", &n);
    ASSERT_INT_EQ(count_id(ids, n, spw), 1);
    free(ids);
    ft_free(tok);
}

static void test_digits_split_individually(void) {
    ft_tokenizer *tok = make_byte_tok();
    /* Merge "1 2" -> "12". qwen35 emits each digit as its own pre-token
     * (\p{N} matches a single digit), so the merge must NOT fire. */
    char *one = gpt2_bytes_encode("1");
    char *two = gpt2_bytes_encode("2");
    ft_add_merge(tok, one, two);
    char *twelve_enc = gpt2_bytes_encode("12");
    int twelve = ft_token_to_id(tok, twelve_enc);
    free(twelve_enc); free(one); free(two);

    int n = 0;
    int *ids = pn_bpe_encode_qwen35(tok, "12", &n);
    ASSERT_INT_EQ(count_id(ids, n, twelve), 0);
    ASSERT_INT_EQ(n, 2);
    free(ids);
    ft_free(tok);
}

static void test_newline_is_own_pretoken(void) {
    ft_tokenizer *tok = make_byte_tok();
    /* Merge a letter with a following newline ("b" + "Ċ"). The "\s*[\r\n]+"
     * alternative isolates newlines, so "b\n" must not merge. */
    char *b = gpt2_bytes_encode("b");
    char *nl = gpt2_bytes_encode("\n");
    ft_add_merge(tok, b, nl);
    char *bnl_enc = gpt2_bytes_encode("b\n");
    int bnl = ft_token_to_id(tok, bnl_enc);
    free(bnl_enc); free(b); free(nl);

    int n = 0;
    int *ids = pn_bpe_encode_qwen35(tok, "ab\n", &n);
    ASSERT_INT_EQ(count_id(ids, n, bnl), 0);
    free(ids);
    ft_free(tok);
}

/* ============================================================
 * llama-bpe (Llama-3) pre-tokenizer tests.
 *
 * These assert the three ways llama-bpe's split differs from qwen35:
 *   - digits group 1..3 per pre-token (\p{N}{1,3}), not one-at-a-time;
 *   - the space/word and space/symbol handling matches Llama-3 boundaries.
 * The merge-firing trick is the same as the qwen35 tests: a merge fires iff
 * its two halves land in the same pre-token, so which merges fire directly
 * observes the split boundaries (vocab-independent, hand-verifiable).
 * ============================================================ */

static void test_llama3_digits_group_by_three(void) {
    ft_tokenizer *tok = make_byte_tok();
    /* Merge "1 2" -> "12" and "2 3" -> "23". In llama-bpe "\p{N}{1,3}" groups
     * up to three digits into ONE pre-token, so within "123" the digits are
     * adjacent and a two-digit merge CAN fire (qwen35 would forbid it). */
    char *one = gpt2_bytes_encode("1");
    char *two = gpt2_bytes_encode("2");
    ft_add_merge(tok, one, two);
    char *twelve_enc = gpt2_bytes_encode("12");
    int twelve = ft_token_to_id(tok, twelve_enc);
    free(twelve_enc); free(one); free(two);

    int n = 0;
    int *ids = pn_bpe_encode_llama3(tok, "123", &n);
    /* "123" is a single 3-digit pre-token, so the "12" merge fires once. */
    ASSERT_INT_EQ(count_id(ids, n, twelve), 1);
    free(ids);
    ft_free(tok);
}

static void test_llama3_digits_split_after_three(void) {
    ft_tokenizer *tok = make_byte_tok();
    /* Merge "3 4" -> "34": digits 3 and 4 straddle the 3-digit boundary in
     * "12345" ("123" | "45"), so this merge must NOT fire. */
    char *three = gpt2_bytes_encode("3");
    char *four = gpt2_bytes_encode("4");
    ft_add_merge(tok, three, four);
    char *tf_enc = gpt2_bytes_encode("34");
    int tf = ft_token_to_id(tok, tf_enc);
    free(tf_enc); free(three); free(four);

    int n = 0;
    int *ids = pn_bpe_encode_llama3(tok, "12345", &n);
    ASSERT_INT_EQ(count_id(ids, n, tf), 0);
    free(ids);
    ft_free(tok);
}

static void test_qwen35_digits_never_group(void) {
    /* Regression guard: the qwen35 path still emits one digit per pre-token,
     * so the same "12" merge that fires under llama-bpe must NOT fire here. */
    ft_tokenizer *tok = make_byte_tok();
    char *one = gpt2_bytes_encode("1");
    char *two = gpt2_bytes_encode("2");
    ft_add_merge(tok, one, two);
    char *twelve_enc = gpt2_bytes_encode("12");
    int twelve = ft_token_to_id(tok, twelve_enc);
    free(twelve_enc); free(one); free(two);

    int n = 0;
    int *ids = pn_bpe_encode_qwen35(tok, "123", &n);
    ASSERT_INT_EQ(count_id(ids, n, twelve), 0);
    ASSERT_INT_EQ(n, 3);
    free(ids);
    ft_free(tok);
}

static void test_llama3_contraction_case_insensitive(void) {
    ft_tokenizer *tok = make_byte_tok();
    /* "don't": "n" is a letter-run pre-token ending before the apostrophe;
     * "'t" is its own contraction pre-token. A merge of "n" + "'" must NOT
     * fire because they are in different pre-tokens. Also verify the
     * uppercase "'T" is recognized (case-insensitive). */
    char *nq = gpt2_bytes_encode("n");
    char *apo = gpt2_bytes_encode("'");
    ft_add_merge(tok, nq, apo);
    char *nap_enc = gpt2_bytes_encode("n'");
    int nap = ft_token_to_id(tok, nap_enc);
    free(nap_enc); free(nq); free(apo);

    int n = 0;
    int *ids = pn_bpe_encode_llama3(tok, "don't", &n);
    ASSERT_INT_EQ(count_id(ids, n, nap), 0);
    free(ids);

    /* Uppercase apostrophe-T is still a contraction (own pre-token). */
    n = 0;
    ids = pn_bpe_encode_llama3(tok, "DON'T", &n);
    ASSERT_INT_EQ(count_id(ids, n, nap), 0);
    free(ids);
    ft_free(tok);
}

static void test_llama3_leading_space_joins_word(void) {
    /* Shared with qwen35: a single leading space attaches to the next word. */
    ft_tokenizer *tok = make_byte_tok();
    char *sp = gpt2_bytes_encode(" ");
    char *w = gpt2_bytes_encode("w");
    ft_add_merge(tok, sp, w);
    char *spw_enc = gpt2_bytes_encode(" w");
    int spw = ft_token_to_id(tok, spw_enc);
    free(spw_enc); free(sp); free(w);

    int n = 0;
    int *ids = pn_bpe_encode_llama3(tok, " world", &n);
    ASSERT_INT_EQ(count_id(ids, n, spw), 1);
    free(ids);
    ft_free(tok);
}

static void test_llama3_space_before_digits_is_own_token(void) {
    ft_tokenizer *tok = make_byte_tok();
    /* In "Hello world 123", the space before "123" is NOT consumed by the
     * digit alt (which has no leading space) nor the symbol alt (a digit is
     * not a symbol); it is a lone-space \s+ pre-token. So a merge of " " + "1"
     * must NOT fire. */
    char *sp = gpt2_bytes_encode(" ");
    char *one = gpt2_bytes_encode("1");
    ft_add_merge(tok, sp, one);
    char *sp1_enc = gpt2_bytes_encode(" 1");
    int sp1 = ft_token_to_id(tok, sp1_enc);
    free(sp1_enc); free(sp); free(one);

    int n = 0;
    int *ids = pn_bpe_encode_llama3(tok, "world 123", &n);
    ASSERT_INT_EQ(count_id(ids, n, sp1), 0);
    free(ids);
    ft_free(tok);
}

static void test_llama3_space_before_symbols_joins(void) {
    ft_tokenizer *tok = make_byte_tok();
    /* "world !!!": the space before "!" IS consumed by the symbol alt
     * ( ?[^\s\p{L}\p{N}]+ ), so " " + "!" land in one pre-token and merge. */
    char *sp = gpt2_bytes_encode(" ");
    char *bang = gpt2_bytes_encode("!");
    ft_add_merge(tok, sp, bang);
    char *spb_enc = gpt2_bytes_encode(" !");
    int spb = ft_token_to_id(tok, spb_enc);
    free(spb_enc); free(sp); free(bang);

    int n = 0;
    int *ids = pn_bpe_encode_llama3(tok, "world !!!", &n);
    ASSERT_INT_EQ(count_id(ids, n, spb), 1);
    free(ids);
    ft_free(tok);
}

static void test_pre_from_name_llama_bpe(void) {
    /* Dispatch: the metadata identifier maps to the Llama-3 pre-type. */
    ASSERT_INT_EQ(pn_bpe_pre_from_name("llama-bpe"), PN_BPE_PRE_LLAMA3);
    ASSERT_INT_EQ(pn_bpe_pre_from_name("qwen35"), PN_BPE_PRE_QWEN35);
    ASSERT_INT_EQ(pn_bpe_pre_from_name("nonexistent-pre"), PN_BPE_PRE_UNKNOWN);
}

int main(void) {
    printf("\n  qwen35 + llama-bpe BPE pre-tokenization tests\n\n");

    RUN_TEST(double_space_does_not_cross_word);
    RUN_TEST(double_space_run_at_end_merges);
    RUN_TEST(single_leading_space_joins_word);
    RUN_TEST(digits_split_individually);
    RUN_TEST(newline_is_own_pretoken);

    RUN_TEST(llama3_digits_group_by_three);
    RUN_TEST(llama3_digits_split_after_three);
    RUN_TEST(qwen35_digits_never_group);
    RUN_TEST(llama3_contraction_case_insensitive);
    RUN_TEST(llama3_leading_space_joins_word);
    RUN_TEST(llama3_space_before_digits_is_own_token);
    RUN_TEST(llama3_space_before_symbols_joins);
    RUN_TEST(pre_from_name_llama_bpe);

    TEST_REPORT();
    return __ft_failed > 0 ? 1 : 0;
}
