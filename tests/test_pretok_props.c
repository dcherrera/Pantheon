/* SPDX-License-Identifier: AGPL-3.0-or-later */
/* Copyright (C) 2026 David (Corey) Herrera */
/*
 * Property tests for the BPE pre-tokenizer (src/core/bpe_pretokenize.c).
 *
 * Boundary oracle: a unigram tokenizer whose vocab is the 256 GPT-2 byte
 * characters (id == byte value) plus a raw ASCII " " (id 256). The unigram
 * path of ft_encode prepends " " to every string it encodes, and GPT-2
 * byte-encoding never produces a raw 0x20, so id 256 appears exactly once at
 * the start of each pre-token. The output therefore spells out the pre-token
 * split byte for byte, with no merge-ordering ambiguity.
 *
 * Covers: the worked-example table in both modes, the per-code-point class
 * table over 1..0x10FFFF (+ samples above), invalid UTF-8, whitespace,
 * contractions, digit grouping, partition of random byte strings, and the
 * driver contract (NULL/empty, long pieces, pre-type dispatch, the once-only
 * unknown-pre warning).
 */

#include "lib/test.h"
#include "../src/core/bpe_pretokenize.h"
#include "../src/core/engine.h"
#include "tokenizer/tokenizer.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>

#define SP_ID 256

static ft_tokenizer *g_tok;

static ft_tokenizer *make_oracle_tok(void) {
    ft_tokenizer *tok = ft_new();
    ft_set_model_type(tok, 1); /* unigram: prepends " " per encoded piece */
    for (int b = 0; b < 256; b++) {
        char raw[2] = {(char)b, 0};
        if (b == 0) {
            /* gpt2_bytes_encode cannot see a NUL; text never contains one,
             * but keep ids aligned with a placeholder that never matches. */
            ft_add_token(tok, "<nul-placeholder>");
            continue;
        }
        char *enc = gpt2_bytes_encode(raw);
        ft_add_token(tok, enc);
        free(enc);
    }
    ft_add_token(tok, " ");
    return tok;
}

/* Split `s[0..len)` with mode (llama?) into piece lengths. Returns count, or -1
 * if the oracle output does not spell the input exactly. */
static int split_lens(const char *s, int llama, int *lens, int max) {
    int n = 0;
    int *ids = llama ? pn_bpe_encode_llama3(g_tok, s, &n) : pn_bpe_encode_qwen35(g_tok, s, &n);
    if (!ids) return -1;
    int pieces = -1;
    size_t byte = 0, slen = strlen(s);
    for (int i = 0; i < n; i++) {
        if (ids[i] == SP_ID) {
            if (pieces >= 0 && lens[pieces] == 0) { free(ids); return -1; } /* empty piece */
            pieces++;
            if (pieces >= max) { free(ids); return -1; }
            lens[pieces] = 0;
            continue;
        }
        if (pieces < 0 || byte >= slen || ids[i] != (unsigned char)s[byte]) { free(ids); return -1; }
        byte++;
        lens[pieces]++;
    }
    free(ids);
    if (byte != slen) return -1;
    if (pieces >= 0 && lens[pieces] == 0) return -1;
    return pieces + 1;
}

/* Render a split as "a|b|c" with pieces copied raw. */
static void render(const char *s, int llama, char *out, size_t cap) {
    int lens[512];
    int n = split_lens(s, llama, lens, 512);
    if (n < 0) { snprintf(out, cap, "<oracle error>"); return; }
    size_t o = 0, p = 0;
    for (int i = 0; i < n; i++) {
        if (i && o + 1 < cap) out[o++] = '|';
        for (int k = 0; k < lens[i] && o + 1 < cap; k++) out[o++] = s[p++];
    }
    out[o] = 0;
}

static int g_split_fail;

static void expect_split(const char *in, int llama, const char *want) {
    char got[4096];
    render(in, llama, got, sizeof got);
    if (strcmp(got, want) != 0) {
        printf("    split mismatch (%s) for \"", llama ? "L" : "Q");
        for (const char *c = in; *c; c++) printf((unsigned char)*c < 0x20 || (unsigned char)*c >= 0x7F ? "\\x%02X" : "%c", (unsigned char)*c);
        printf("\": got [%s] want [%s]\n", got, want);
        g_split_fail++;
    }
}

static void expect_both(const char *in, const char *want) {
    expect_split(in, 0, want);
    expect_split(in, 1, want);
}

/* ------------------------------------------------------------------ */

static void test_worked_examples(void) {
    g_split_fail = 0;
    expect_both("Hello world", "Hello| world");
    expect_both("a  b", "a| | b");
    expect_both("a  ", "a|  ");
    expect_both("  ", "  ");
    expect_split("12", 0, "1|2");
    expect_split("12", 1, "12");
    expect_split("12345", 0, "1|2|3|4|5");
    expect_split("12345", 1, "123|45");
    expect_both("don't", "don|'t");
    expect_both("DON'T", "DON|'T");
    expect_both("we'LL go", "we|'LL| go");
    expect_both("x!!\n\ny", "x|!!\n\n|y");
    expect_both("world !!!", "world| !!!");
    expect_both("\n\n  foo", "\n\n| | foo");
    expect_both("\n \n  x", "\n \n| | x");
    expect_both("\n  ", "\n|  ");
    expect_both("a\t5", "a|\t|5");
    expect_both("\tfoo", "\tfoo");
    expect_split("e\xCC\x81x", 0, "e\xCC\x81x");
    expect_split("e\xCC\x81x", 1, "e|\xCC\x81x");
    expect_split("e\xCC\x81!", 0, "e\xCC\x81|!");
    expect_split("e\xCC\x81!", 1, "e|\xCC\x81!");
    expect_split("1\xD9\xA0\xD9\xA1", 0, "1|\xD9\xA0\xD9\xA1");
    expect_split("1\xD9\xA0\xD9\xA1", 1, "1\xD9\xA0\xD9\xA1");
    expect_both("foo'", "foo|'");
    expect_both("\xC3\x0A" "a", "\xC3\x0A" "a");
    ASSERT_INT_EQ(g_split_fail, 0);
}

static void test_invalid_utf8(void) {
    g_split_fail = 0;
    expect_both("a\x80", "a|\x80");          /* stray continuation: not a letter */
    expect_both("!\x80", "!\x80");           /* ... it is a symbol */
    expect_both("a\xBF", "a|\xBF");
    expect_both("a\xC3", "a\xC3");           /* lone lead at EOS: Latin-1 letter */
    expect_both("a\xAA\xB5\xBA", "a\xAA\xB5\xBA");
    expect_both("a\xD7", "a|\xD7");
    expect_both("a\xF7", "a|\xF7");
    expect_both("\n\x85\n", "\n\x85\n");     /* lone 0x85 / 0xA0 are whitespace */
    expect_both("\n\xA0\n", "\n\xA0\n");
    expect_both("a\xE2\x82", "a\xE2|\x82");  /* E2 82 at EOS: two width-1 code points */
    expect_both("a\xF0\x9F\x98", "a\xF0|\x9F\x98");  /* truncated 4-byte at EOS */
    expect_both("a\xF0\x9F\x98\x80", "a|\xF0\x9F\x98\x80"); /* U+1F600: symbol */
    expect_both("a\xF8", "a\xF8");
    expect_both("a\xFF", "a\xFF");
    expect_both("a\xC0\xAF", "a|\xC0\xAF");  /* overlong '/' is a symbol */
    expect_both("!\xC0\xAF", "!\xC0\xAF");
    expect_both("a\xC1\x81", "a\xC1\x81");   /* overlong 'A' is a letter */
    expect_both("a\xED\xA0\x80", "a|\xED\xA0\x80");
    expect_both("!\xED\xA0\x80", "!\xED\xA0\x80");
    expect_both("!\xF4\x90\x80\x80", "!\xF4\x90\x80\x80");
    expect_both("x\xC3\x0A\xC3", "x\xC3\x0A\xC3");
    ASSERT_INT_EQ(g_split_fail, 0);
}

static void test_whitespace(void) {
    g_split_fail = 0;
    expect_both("\r\n\r\nx", "\r\n\r\n|x");
    expect_both("a\r\n\r\n", "a|\r\n\r\n");
    expect_both("\t \n x", "\t \n| x");
    expect_both("a \t", "a| \t");
    expect_both("a \xE3\x80\x80" "b", "a| |\xE3\x80\x80" "b");   /* U+3000 dropped by R6 */
    expect_both("a \xC2\xA0" "b", "a| |\xC2\xA0" "b");           /* NBSP dropped by R6 */
    expect_both("\xC2\xA0" "foo", "\xC2\xA0" "foo");              /* NBSP as R2 prefix */
    expect_both("a\xE3\x80\x80\xE3\x80\x80" "1", "a|\xE3\x80\x80|\xE3\x80\x80|1");
    expect_both("a \n", "a| \n");
    expect_both("x\x0B\x0C" "y", "x|\x0B|\x0C" "y");
    expect_both("x\x1C" "y", "x|\x1C" "y");                      /* 0x1C is a symbol */
    ASSERT_INT_EQ(g_split_fail, 0);
}

static void test_contractions(void) {
    g_split_fail = 0;
    static const char *sfx[] = { "s", "t", "m", "d", "re", "ve", "ll" };
    for (int i = 0; i < 7; i++) {
        size_t n = strlen(sfx[i]);
        for (int mask = 0; mask < (1 << n); mask++) {
            char c[3] = {0};
            for (size_t k = 0; k < n; k++)
                c[k] = (char)((mask >> k) & 1 ? sfx[i][k] - 32 : sfx[i][k]);
            char in[32], want[32];
            snprintf(in, sizeof in, "x'%s y", c);
            snprintf(want, sizeof want, "x|'%s| y", c);
            expect_both(in, want);
        }
    }
    expect_both("x'", "x|'");
    expect_both("x'r", "x|'r");
    expect_both("'sxyz", "'s|xyz");
    expect_both("x'rx", "x|'rx");
    expect_both("x\xE2\x80\x99s", "x|\xE2\x80\x99s");   /* U+2019 is not a contraction */
    ASSERT_INT_EQ(g_split_fail, 0);
}

static void test_digits(void) {
    g_split_fail = 0;
    for (int n = 1; n <= 7; n++) {
        char in[16], wq[32], wl[32];
        size_t q = 0, l = 0;
        for (int i = 0; i < n; i++) {
            in[i] = (char)('1' + i);
            if (i) wq[q++] = '|';
            wq[q++] = in[i];
            if (i && i % 3 == 0) wl[l++] = '|';
            wl[l++] = in[i];
        }
        in[n] = 0; wq[q] = 0; wl[l] = 0;
        expect_split(in, 0, wq);
        expect_split(in, 1, wl);
    }
    /* ASCII, fullwidth 1, Arabic-Indic 1, ASCII */
    const char *mix = "1\xEF\xBC\x91\xD9\xA1" "2";
    expect_split(mix, 0, "1|\xEF\xBC\x91|\xD9\xA1|2");
    expect_split(mix, 1, "1\xEF\xBC\x91\xD9\xA1|2");
    /* Extended Arabic-Indic and Devanagari digits */
    expect_split("1\xDB\xB1", 1, "1\xDB\xB1");
    expect_split("1\xE0\xA5\xA7", 1, "1|\xE0\xA5\xA7"); /* U+0967: letter only */
    ASSERT_INT_EQ(g_split_fail, 0);
}

/* ---- class table sweep ---- */

typedef struct { unsigned lo, hi; } rng;

static const rng X_LETTER[] = {
    {'A','Z'},{'a','z'},{0xAA,0xAA},{0xB5,0xB5},{0xBA,0xBA},{0xC0,0xD6},{0xD8,0xF6},{0xF8,0xFF},
    {0x100,0x2AF},{0x370,0x3FF},{0x400,0x4FF},{0x530,0x58F},{0x590,0x5FF},{0x600,0x6FF},
    {0x900,0x97F},{0x1100,0x11FF},{0x3040,0x309F},{0x30A0,0x30FF},{0x3400,0x4DBF},
    {0x4E00,0x9FFF},{0xAC00,0xD7AF},{0xF900,0xFAFF},{0x20000,0x2FA1F},
};
static const rng X_NUMBER[] = { {'0','9'},{0x660,0x669},{0x6F0,0x6F9},{0xFF10,0xFF19} };
static const rng X_MARK[] = {
    {0x300,0x36F},{0x483,0x489},{0x591,0x5BD},{0x64B,0x65F},{0x900,0x903},{0x93A,0x94F},
    {0x1AB0,0x1AFF},{0x1DC0,0x1DFF},{0x20D0,0x20FF},{0xFE20,0xFE2F},
};
static const rng X_SPACE[] = {
    {0x20,0x20},{0x09,0x09},{0x0A,0x0A},{0x0D,0x0D},{0x0B,0x0B},{0x0C,0x0C},{0x85,0x85},
    {0xA0,0xA0},{0x1680,0x1680},{0x2000,0x200A},{0x2028,0x2029},{0x202F,0x202F},
    {0x205F,0x205F},{0x3000,0x3000},
};

static int in_set(unsigned cp, const rng *r, size_t n) {
    for (size_t i = 0; i < n; i++) if (cp >= r[i].lo && cp <= r[i].hi) return 1;
    return 0;
}
#define IN(cp, set) in_set((cp), set, sizeof(set) / sizeof(set[0]))

static int utf8_put(unsigned cp, char *o) {
    if (cp < 0x80) { o[0] = (char)cp; return 1; }
    if (cp < 0x800) { o[0] = (char)(0xC0 | (cp >> 6)); o[1] = (char)(0x80 | (cp & 0x3F)); return 2; }
    if (cp < 0x10000) {
        o[0] = (char)(0xE0 | (cp >> 12)); o[1] = (char)(0x80 | ((cp >> 6) & 0x3F));
        o[2] = (char)(0x80 | (cp & 0x3F)); return 3;
    }
    o[0] = (char)(0xF0 | (cp >> 18)); o[1] = (char)(0x80 | ((cp >> 12) & 0x3F));
    o[2] = (char)(0x80 | ((cp >> 6) & 0x3F)); o[3] = (char)(0x80 | (cp & 0x3F)); return 4;
}

/* Is the first piece of `pre`+cp+`post` longer than `pre`? */
static int joined(const char *pre, unsigned cp, const char *post, int llama) {
    char buf[32];
    size_t pl = strlen(pre);
    memcpy(buf, pre, pl);
    int w = utf8_put(cp, buf + pl);
    strcpy(buf + pl + w, post);
    int lens[8];
    int n = split_lens(buf, llama, lens, 8);
    if (n < 0) return -1;
    return lens[0] > (int)pl;
}

static int check_cp(unsigned cp) {
    int letter = IN(cp, X_LETTER), number = IN(cp, X_NUMBER);
    int mark = IN(cp, X_MARK), space = IN(cp, X_SPACE);
    int nl = cp == 0x0A || cp == 0x0D;
    int symq = !space && !letter && !number && !mark;
    int syml = !space && !letter && !number;
    int bad = 0;
    bad |= joined("a", cp, "", 1) != letter;
    bad |= joined("a", cp, "", 0) != (letter || mark);
    bad |= joined("1", cp, "", 1) != number;
    bad |= joined("\n", cp, "\n", 0) != space;
    /* '!' joins a following symbol run (R4), newlines after it, or acts as the
     * one-code-point prefix of a letter run (R2). */
    bad |= joined("!", cp, "", 0) != (symq || nl || letter || mark);
    bad |= joined("!", cp, "", 1) != (syml || nl || letter);
    if (bad) printf("    class mismatch at U+%04X\n", cp);
    return bad;
}

static void test_class_table(void) {
    int fails = 0;
    for (unsigned cp = 1; cp <= 0x10FFFF && fails < 10; cp++)
        fails += check_cp(cp);
    /* Values above U+10FFFF (leads F4..F7) decode as-is and belong to no set. */
    for (unsigned cp = 0x110000; cp <= 0x1FFFFF && fails < 10; cp += 0x1111)
        fails += check_cp(cp);
    fails += check_cp(0x1FFFFF);
    ASSERT_INT_EQ(fails, 0);
}

/* ---- partition of random byte strings ---- */

static size_t ref_width(const unsigned char *s, size_t len, size_t p) {
    unsigned c = s[p];
    if (c < 0x80) return 1;
    if ((c & 0xE0) == 0xC0 && p + 1 < len) return 2;
    if ((c & 0xF0) == 0xE0 && p + 2 < len) return 3;
    if ((c & 0xF8) == 0xF0 && p + 3 < len) return 4;
    return 1;
}

static void test_partition_random(void) {
    static const unsigned char alphabet[] = {
        'a', 'Z', '1', ' ', '\n', '\r', '\t', '\'', 's', '!', '.', 0x80, 0xA0, 0xBF, 0xC3, 0xCC,
        0xD9, 0xE2, 0xE3, 0xED, 0xF0, 0xF4, 0xF8, 0xFF, 0x85, 0x81, 0x82, 0x99, 'l', 'e',
    };
    unsigned seed = 12345;
    int fails = 0;
    for (int iter = 0; iter < 20000 && fails < 5; iter++) {
        char s[64];
        seed = seed * 1103515245u + 12345u;
        int len = 1 + (int)((seed >> 16) % 40);
        for (int i = 0; i < len; i++) {
            seed = seed * 1103515245u + 12345u;
            s[i] = (char)alphabet[(seed >> 16) % sizeof alphabet];
        }
        s[len] = 0;
        for (int llama = 0; llama < 2; llama++) {
            int lens[64];
            int n = split_lens(s, llama, lens, 64);
            if (n < 0) { fails++; printf("    partition broken (iter %d)\n", iter); continue; }
            /* Every boundary must sit on a code-point start of the decoder walk. */
            size_t p = 0, b = 0;
            for (int k = 0; k < n; k++) {
                b += (size_t)lens[k];
                while (p < b) p += ref_width((const unsigned char *)s, (size_t)len, p);
                if (p != b) { fails++; printf("    mid-code-point split (iter %d)\n", iter); break; }
            }
        }
    }
    ASSERT_INT_EQ(fails, 0);
}

/* ---- driver ---- */

static void test_driver_null_and_empty(void) {
    int n = -7;
    int *ids = pn_bpe_encode_qwen35(NULL, "abc", &n);
    ASSERT_NOT_NULL(ids); ASSERT_INT_EQ(n, 0); free(ids);
    n = -7;
    ids = pn_bpe_encode_llama3(g_tok, NULL, &n);
    ASSERT_NOT_NULL(ids); ASSERT_INT_EQ(n, 0); free(ids);
    n = -7;
    ids = pn_bpe_encode(g_tok, "", &n);
    ASSERT_NOT_NULL(ids); ASSERT_INT_EQ(n, 0); free(ids);
    n = -7;
    ids = pn_bpe_encode(NULL, "x", &n);
    ASSERT_NOT_NULL(ids); ASSERT_INT_EQ(n, 0); free(ids);
}

static void test_driver_long_piece(void) {
    size_t L = 10000;
    char *s = malloc(L + 1);
    memset(s, 'a', L);
    s[L] = 0;
    int lens[4];
    ASSERT_INT_EQ(split_lens(s, 0, lens, 4), 1);
    ASSERT_INT_EQ(lens[0], (int)L);
    s[0] = ' ';
    ASSERT_INT_EQ(split_lens(s, 1, lens, 4), 1);
    free(s);
}

static void test_pre_from_name_table(void) {
    ASSERT_INT_EQ(pn_bpe_pre_from_name(NULL), PN_BPE_PRE_QWEN35);
    ASSERT_INT_EQ(pn_bpe_pre_from_name("qwen35"), PN_BPE_PRE_QWEN35);
    ASSERT_INT_EQ(pn_bpe_pre_from_name("qwen2"), PN_BPE_PRE_QWEN35);
    ASSERT_INT_EQ(pn_bpe_pre_from_name("gpt-2"), PN_BPE_PRE_QWEN35);
    ASSERT_INT_EQ(pn_bpe_pre_from_name("default"), PN_BPE_PRE_QWEN35);
    ASSERT_INT_EQ(pn_bpe_pre_from_name("llama-bpe"), PN_BPE_PRE_LLAMA3);
    ASSERT_INT_EQ(pn_bpe_pre_from_name("llama3"), PN_BPE_PRE_LLAMA3);
    ASSERT_INT_EQ(pn_bpe_pre_from_name("llama-v3"), PN_BPE_PRE_LLAMA3);
    ASSERT_INT_EQ(pn_bpe_pre_from_name(""), PN_BPE_PRE_UNKNOWN);
    ASSERT_INT_EQ(pn_bpe_pre_from_name("Qwen35"), PN_BPE_PRE_UNKNOWN);
    ASSERT_INT_EQ(pn_bpe_pre_from_name("llama"), PN_BPE_PRE_UNKNOWN);
}

/* Run pn_bpe_encode with pre-type `pre` on "12"; return piece count, and capture
 * anything written to stderr into `err`. */
static int dispatch_pieces(int pre, char *err, size_t cap) {
    fflush(stderr);
    int pfd[2];
    if (pipe(pfd) != 0) return -1;
    int saved = dup(2);
    dup2(pfd[1], 2);
    close(pfd[1]);

    ft_set_pre_type(g_tok, pre);
    int n = 0;
    int *ids = pn_bpe_encode(g_tok, "12", &n);
    ft_set_pre_type(g_tok, 0);

    fflush(stderr);
    dup2(saved, 2);
    close(saved);
    ssize_t r = read(pfd[0], err, cap - 1);
    err[r > 0 ? r : 0] = 0;
    close(pfd[0]);

    int pieces = 0;
    for (int i = 0; i < n; i++) if (ids[i] == SP_ID) pieces++;
    free(ids);
    return pieces;
}

static void test_dispatch_and_warn_once(void) {
    static const char warn[] =
        "bpe: tokenizer.ggml.pre is unsupported; falling back to the qwen35/GPT-2 "
        "pre-tokenizer (token IDs may not match the reference for this model)\n";
    char err[512];
    ASSERT_INT_EQ(dispatch_pieces(0, err, sizeof err), 2);
    ASSERT_INT_EQ((int)strlen(err), 0);
    ASSERT_INT_EQ(dispatch_pieces(1, err, sizeof err), 1);
    ASSERT_INT_EQ((int)strlen(err), 0);
    ASSERT_INT_EQ(dispatch_pieces(7, err, sizeof err), 2);
    ASSERT_INT_EQ((int)strlen(err), 0);
    ASSERT_INT_EQ(dispatch_pieces(-1, err, sizeof err), 2);
    ASSERT_INT_EQ((int)strlen(err), 0);
    ASSERT_INT_EQ(dispatch_pieces(2, err, sizeof err), 2);
    ASSERT_INT_EQ(strcmp(err, warn), 0);
    for (int i = 0; i < 5; i++) {
        ASSERT_INT_EQ(dispatch_pieces(2, err, sizeof err), 2);
        ASSERT_INT_EQ((int)strlen(err), 0);
    }
    /* Direct entry points never warn. */
    int n = 0;
    int *ids = pn_bpe_encode_qwen35(g_tok, "x", &n);
    free(ids);
}

int main(void) {
    printf("\n  BPE pre-tokenizer property tests\n\n");
    g_tok = make_oracle_tok();

    RUN_TEST(dispatch_and_warn_once); /* first: the warning flag is process-wide */
    RUN_TEST(worked_examples);
    RUN_TEST(invalid_utf8);
    RUN_TEST(whitespace);
    RUN_TEST(contractions);
    RUN_TEST(digits);
    RUN_TEST(class_table);
    RUN_TEST(partition_random);
    RUN_TEST(driver_null_and_empty);
    RUN_TEST(driver_long_piece);
    RUN_TEST(pre_from_name_table);

    ft_free(g_tok);
    TEST_REPORT();
    return __ft_failed > 0 ? 1 : 0;
}
