/* SPDX-License-Identifier: AGPL-3.0-or-later */
/* Copyright (C) 2026 David (Corey) Herrera */
/**
 * @file   bpe_pretokenize.c
 * @brief  Pre-token splitter (qwen35/GPT-2 and llama-bpe modes) feeding ft_encode.
 *
 * Rewritten from a behaviour specification and measured behaviour;
 * the implementer did not see the previous code. The split is a hand-written
 * scanner over a fixed character-class table, tried rule by rule at each position:
 * contraction, letter run, digits, symbol run, whitespace, single code point.
 *
 * @status None
 * @issues - Decoder does not validate continuation bytes (a lead byte swallows whatever
 *           follows, e.g. C3 0A is one code point); stray high bytes classify as Latin-1.
 *         - Class table is coarse: e.g. 0x1E00-0x1EFF, Thai, fullwidth Latin are symbols;
 *           Arabic-Indic digits are both letters and numbers.
 *         - Contractions are case-insensitive in both modes, ASCII apostrophe only.
 *         - Unknown pre-type warning flag is a non-thread-safe static.
 * @todo   None
 */
#include "bpe_pretokenize.h"
#include "engine.h"

#include <stdint.h>
#include <stddef.h>
#include <stdbool.h>
#include <stdlib.h>
#include <string.h>
#include <stdio.h>

/* ---------------------------------------------------------------------------
 * Name -> pre-type
 * ------------------------------------------------------------------------- */

int pn_bpe_pre_from_name(const char *pre) {
    if (pre == NULL) return PN_BPE_PRE_QWEN35;
    static const char *const qwen_names[] = { "qwen35", "qwen2", "gpt-2", "default" };
    static const char *const llama_names[] = { "llama-bpe", "llama3", "llama-v3" };
    for (size_t i = 0; i < sizeof qwen_names / sizeof qwen_names[0]; i++)
        if (strcmp(pre, qwen_names[i]) == 0) return PN_BPE_PRE_QWEN35;
    for (size_t i = 0; i < sizeof llama_names / sizeof llama_names[0]; i++)
        if (strcmp(pre, llama_names[i]) == 0) return PN_BPE_PRE_LLAMA3;
    return PN_BPE_PRE_UNKNOWN;
}

/* ---------------------------------------------------------------------------
 * Lenient UTF-8 decode: width and code point at byte offset p.
 * Continuation bytes are masked, never checked.
 * ------------------------------------------------------------------------- */

static size_t decode_at(const unsigned char *s, size_t len, size_t p, uint32_t *cp) {
    uint32_t c = s[p];
    if (c < 0x80) {
        *cp = c;
        return 1;
    }
    if ((c & 0xE0) == 0xC0 && p + 1 < len) {
        *cp = ((c & 0x1F) << 6) | (s[p + 1] & 0x3Fu);
        return 2;
    }
    if ((c & 0xF0) == 0xE0 && p + 2 < len) {
        *cp = ((c & 0x0F) << 12) | ((s[p + 1] & 0x3Fu) << 6) | (s[p + 2] & 0x3Fu);
        return 3;
    }
    if ((c & 0xF8) == 0xF0 && p + 3 < len) {
        *cp = ((c & 0x07) << 18) | ((s[p + 1] & 0x3Fu) << 12) |
              ((s[p + 2] & 0x3Fu) << 6) | (s[p + 3] & 0x3Fu);
        return 4;
    }
    *cp = c;
    return 1;
}

/* ---------------------------------------------------------------------------
 * Character classes. Each set is independent; they overlap on purpose.
 * ------------------------------------------------------------------------- */

typedef struct { uint32_t lo, hi; } cp_range;

static bool in_ranges(uint32_t cp, const cp_range *r, size_t n) {
    for (size_t i = 0; i < n; i++)
        if (cp >= r[i].lo && cp <= r[i].hi) return true;
    return false;
}

#define N_RANGES(a) (sizeof(a) / sizeof((a)[0]))

static const cp_range letter_ranges[] = {
    { 0x0041, 0x005A }, { 0x0061, 0x007A },
    { 0x00AA, 0x00AA }, { 0x00B5, 0x00B5 }, { 0x00BA, 0x00BA },
    { 0x00C0, 0x00D6 }, { 0x00D8, 0x00F6 }, { 0x00F8, 0x00FF },
    { 0x0100, 0x02AF },
    { 0x0370, 0x03FF },
    { 0x0400, 0x04FF },
    { 0x0530, 0x058F },
    { 0x0590, 0x05FF },
    { 0x0600, 0x06FF },
    { 0x0900, 0x097F },
    { 0x1100, 0x11FF },
    { 0x3040, 0x309F },
    { 0x30A0, 0x30FF },
    { 0x3400, 0x4DBF },
    { 0x4E00, 0x9FFF },
    { 0xAC00, 0xD7AF },
    { 0xF900, 0xFAFF },
    { 0x20000, 0x2FA1F },
};

static const cp_range number_ranges[] = {
    { 0x0030, 0x0039 },
    { 0x0660, 0x0669 },
    { 0x06F0, 0x06F9 },
    { 0xFF10, 0xFF19 },
};

static const cp_range mark_ranges[] = {
    { 0x0300, 0x036F },
    { 0x0483, 0x0489 },
    { 0x0591, 0x05BD },
    { 0x064B, 0x065F },
    { 0x0900, 0x0903 },
    { 0x093A, 0x094F },
    { 0x1AB0, 0x1AFF },
    { 0x1DC0, 0x1DFF },
    { 0x20D0, 0x20FF },
    { 0xFE20, 0xFE2F },
};

static const cp_range space_ranges[] = {
    { 0x0009, 0x000D },   /* tab, LF, VT, FF, CR */
    { 0x0020, 0x0020 },
    { 0x0085, 0x0085 },
    { 0x00A0, 0x00A0 },
    { 0x1680, 0x1680 },
    { 0x2000, 0x200A },
    { 0x2028, 0x2029 },
    { 0x202F, 0x202F },
    { 0x205F, 0x205F },
    { 0x3000, 0x3000 },
};

static bool cls_letter(uint32_t cp) { return in_ranges(cp, letter_ranges, N_RANGES(letter_ranges)); }
static bool cls_number(uint32_t cp) { return in_ranges(cp, number_ranges, N_RANGES(number_ranges)); }
static bool cls_mark(uint32_t cp)   { return in_ranges(cp, mark_ranges, N_RANGES(mark_ranges)); }
static bool cls_space(uint32_t cp)  { return in_ranges(cp, space_ranges, N_RANGES(space_ranges)); }
static bool cls_newline(uint32_t cp) { return cp == 0x0A || cp == 0x0D; }

/* Mode-dependent derived predicates. `llama` selects the llama-bpe rules. */

static bool is_run(uint32_t cp, bool llama) {
    if (cls_letter(cp)) return true;
    return !llama && cls_mark(cp);
}

static bool is_sym(uint32_t cp, bool llama) {
    if (cls_space(cp) || cls_letter(cp) || cls_number(cp)) return false;
    if (!llama && cls_mark(cp)) return false;
    return true;
}

static bool is_prefix(uint32_t cp) {
    return !cls_newline(cp) && !cls_letter(cp) && !cls_number(cp);
}

/* ---------------------------------------------------------------------------
 * One pre-token: returns the exclusive end offset of the pre-token at pos.
 * ------------------------------------------------------------------------- */

static unsigned char ascii_lower(unsigned char c) {
    return (c >= 'A' && c <= 'Z') ? (unsigned char)(c + ('a' - 'A')) : c;
}

static size_t scan_contraction(const unsigned char *s, size_t len, size_t pos) {
    if (s[pos] != '\'' || pos + 1 >= len) return 0;
    unsigned char a = ascii_lower(s[pos + 1]);
    if (a == 's' || a == 't' || a == 'm' || a == 'd') return pos + 2;
    if (pos + 2 < len) {
        unsigned char b = ascii_lower(s[pos + 2]);
        if ((a == 'r' && b == 'e') || (a == 'v' && b == 'e') || (a == 'l' && b == 'l'))
            return pos + 3;
    }
    return 0;
}

static size_t scan_letters(const unsigned char *s, size_t len, size_t pos,
                           uint32_t cp0, size_t w0, bool llama) {
    size_t start;
    uint32_t next;
    if (is_prefix(cp0) && pos + w0 < len &&
        (decode_at(s, len, pos + w0, &next), is_run(next, llama))) {
        start = pos + w0;
    } else if (is_run(cp0, llama)) {
        start = pos;
    } else {
        return 0;
    }
    size_t p = start;
    while (p < len) {
        uint32_t cp;
        size_t w = decode_at(s, len, p, &cp);
        if (!is_run(cp, llama)) break;
        p += w;
    }
    return p;
}

static size_t scan_digits(const unsigned char *s, size_t len, size_t pos,
                          uint32_t cp0, size_t w0, bool llama) {
    if (!cls_number(cp0)) return 0;
    size_t p = pos + w0;
    if (!llama) return p;
    for (int extra = 0; extra < 2 && p < len; extra++) {
        uint32_t cp;
        size_t w = decode_at(s, len, p, &cp);
        if (!cls_number(cp)) break;
        p += w;
    }
    return p;
}

static size_t scan_symbols(const unsigned char *s, size_t len, size_t pos,
                           uint32_t cp0, size_t w0, bool llama) {
    size_t start = pos;
    uint32_t cp = cp0;
    if (cp0 == 0x20 && pos + w0 < len) {
        uint32_t next;
        decode_at(s, len, pos + w0, &next);
        if (is_sym(next, llama)) {
            start = pos + w0;
            cp = next;
        }
    }
    if (!is_sym(cp, llama)) return 0;
    size_t p = start;
    while (p < len) {
        uint32_t c;
        size_t w = decode_at(s, len, p, &c);
        if (!is_sym(c, llama)) break;
        p += w;
    }
    while (p < len) {
        uint32_t c;
        size_t w = decode_at(s, len, p, &c);
        if (!cls_newline(c)) break;
        p += w;
    }
    return p;
}

static size_t scan_space(const unsigned char *s, size_t len, size_t pos, uint32_t cp0) {
    if (!cls_space(cp0)) return 0;
    size_t p = pos, last_start = pos, nl_end = 0, n = 0;
    while (p < len) {
        uint32_t c;
        size_t w = decode_at(s, len, p, &c);
        if (!cls_space(c)) break;
        last_start = p;
        p += w;
        n++;
        if (cls_newline(c)) nl_end = p;
    }
    if (nl_end != 0) return nl_end;          /* stop after the last newline */
    if (p < len && n >= 2) return last_start; /* leave one space for the next word */
    return p;
}

static size_t scan_one(const unsigned char *s, size_t len, size_t pos, bool llama) {
    uint32_t cp0;
    size_t w0 = decode_at(s, len, pos, &cp0);
    size_t end;
    if ((end = scan_contraction(s, len, pos)) != 0) return end;
    if ((end = scan_letters(s, len, pos, cp0, w0, llama)) != 0) return end;
    if ((end = scan_digits(s, len, pos, cp0, w0, llama)) != 0) return end;
    if ((end = scan_symbols(s, len, pos, cp0, w0, llama)) != 0) return end;
    if ((end = scan_space(s, len, pos, cp0)) != 0) return end;
    return pos + w0;
}

/* ---------------------------------------------------------------------------
 * Driver
 * ------------------------------------------------------------------------- */

static int *encode_with_mode(ft_tokenizer *tok, const char *text, int *length, bool llama) {
    *length = 0;
    if (tok == NULL || text == NULL) return calloc(1, sizeof(int));

    const unsigned char *s = (const unsigned char *)text;
    size_t len = strlen(text);

    int *ids = NULL;
    size_t n_ids = 0, cap_ids = 0;
    char *piece = NULL;
    size_t cap_piece = 0;

    size_t pos = 0;
    while (pos < len) {
        size_t end = scan_one(s, len, pos, llama);
        if (end <= pos) end = pos + 1;
        size_t plen = end - pos;

        if (plen + 1 > cap_piece) {
            size_t want = cap_piece ? cap_piece : 256;
            while (want < plen + 1) want *= 2;
            char *grown = realloc(piece, want);
            if (grown == NULL) goto fail;
            piece = grown;
            cap_piece = want;
        }
        memcpy(piece, s + pos, plen);
        piece[plen] = '\0';

        char *enc = gpt2_bytes_encode(piece);
        if (enc == NULL) goto fail;
        int sub_n = 0;
        int *sub = ft_encode(tok, enc, &sub_n);
        free(enc);
        if (sub == NULL) goto fail;

        if (sub_n > 0) {
            size_t need = n_ids + (size_t)sub_n;
            if (need > cap_ids) {
                size_t want = cap_ids ? cap_ids : 256;
                while (want < need) want *= 2;
                int *grown = realloc(ids, want * sizeof(int));
                if (grown == NULL) {
                    free(sub);
                    goto fail;
                }
                ids = grown;
                cap_ids = want;
            }
            memcpy(ids + n_ids, sub, (size_t)sub_n * sizeof(int));
            n_ids = need;
        }
        free(sub);
        pos = end;
    }

    free(piece);
    *length = (int)n_ids;
    if (n_ids == 0) {
        free(ids);
        return calloc(1, sizeof(int));
    }
    return ids;

fail:
    free(piece);
    free(ids);
    return NULL;
}

int *pn_bpe_encode_qwen35(ft_tokenizer *tok, const char *text, int *length) {
    return encode_with_mode(tok, text, length, false);
}

int *pn_bpe_encode_llama3(ft_tokenizer *tok, const char *text, int *length) {
    return encode_with_mode(tok, text, length, true);
}

int *pn_bpe_encode(ft_tokenizer *tok, const char *text, int *length) {
    int pre = tok ? ft_get_pre_type(tok) : PN_BPE_PRE_QWEN35;
    if (pre == PN_BPE_PRE_LLAMA3) return pn_bpe_encode_llama3(tok, text, length);
    if (pre == PN_BPE_PRE_UNKNOWN) {
        static bool warned = false;
        if (!warned) {
            warned = true;
            fprintf(stderr, "bpe: tokenizer.ggml.pre is unsupported; falling back to the "
                            "qwen35/GPT-2 pre-tokenizer (token IDs may not match the "
                            "reference for this model)\n");
        }
    }
    return pn_bpe_encode_qwen35(tok, text, length);
}
