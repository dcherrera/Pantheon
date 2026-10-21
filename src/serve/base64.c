/* SPDX-License-Identifier: AGPL-3.0-or-later */
/* Copyright (C) 2026 David (Corey) Herrera */
/**
 * @file base64.c
 * @brief Implementation of the Base64 encode/decode in base64.h.
 *
 * @status Complete; unit-tested (tests/test_base64.c).
 * @issues None
 * @todo None
 */

#include "base64.h"

#include <stdlib.h>
#include <string.h>

static const char B64_ALPHABET[] =
    "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/";

char *pn_base64_encode(const unsigned char *src, size_t len)
{
    size_t out_len = ((len + 2) / 3) * 4;
    char *out = malloc(out_len + 1);
    if (!out)
        return NULL;

    size_t o = 0;
    size_t i = 0;
    while (i + 3 <= len) {
        unsigned v = ((unsigned)src[i] << 16) | ((unsigned)src[i + 1] << 8) |
                     (unsigned)src[i + 2];
        out[o++] = B64_ALPHABET[(v >> 18) & 0x3F];
        out[o++] = B64_ALPHABET[(v >> 12) & 0x3F];
        out[o++] = B64_ALPHABET[(v >> 6) & 0x3F];
        out[o++] = B64_ALPHABET[v & 0x3F];
        i += 3;
    }
    size_t rem = len - i;
    if (rem == 1) {
        unsigned v = (unsigned)src[i] << 16;
        out[o++] = B64_ALPHABET[(v >> 18) & 0x3F];
        out[o++] = B64_ALPHABET[(v >> 12) & 0x3F];
        out[o++] = '=';
        out[o++] = '=';
    } else if (rem == 2) {
        unsigned v = ((unsigned)src[i] << 16) | ((unsigned)src[i + 1] << 8);
        out[o++] = B64_ALPHABET[(v >> 18) & 0x3F];
        out[o++] = B64_ALPHABET[(v >> 12) & 0x3F];
        out[o++] = B64_ALPHABET[(v >> 6) & 0x3F];
        out[o++] = '=';
    }
    out[o] = '\0';
    return out;
}

/* Reverse map: alphabet char -> 6-bit value, 0xFF for invalid. Built once. */
static int b64_value(unsigned char ch)
{
    if (ch >= 'A' && ch <= 'Z') return ch - 'A';
    if (ch >= 'a' && ch <= 'z') return ch - 'a' + 26;
    if (ch >= '0' && ch <= '9') return ch - '0' + 52;
    if (ch == '+') return 62;
    if (ch == '/') return 63;
    return -1;
}

unsigned char *pn_base64_decode(const char *src, size_t *out_len)
{
    if (out_len)
        *out_len = 0;
    if (!src)
        return NULL;
    size_t n = strlen(src);
    /* A valid base64 body length is a multiple of 4 (padding included). */
    if (n % 4 != 0)
        return NULL;

    size_t pad = 0;
    if (n >= 1 && src[n - 1] == '=') pad++;
    if (n >= 2 && src[n - 2] == '=') pad++;
    size_t dec_len = n / 4 * 3 - pad;

    unsigned char *out = malloc(dec_len ? dec_len : 1);
    if (!out)
        return NULL;

    size_t o = 0;
    for (size_t i = 0; i < n; i += 4) {
        int q[4];
        for (int k = 0; k < 4; k++) {
            char ch = src[i + k];
            if (ch == '=') {
                /* Padding is only legal in the final quad's last two slots. */
                if (i + 4 != n || k < 2) { free(out); return NULL; }
                q[k] = 0;
            } else {
                q[k] = b64_value((unsigned char)ch);
                if (q[k] < 0) { free(out); return NULL; }
            }
        }
        unsigned v = ((unsigned)q[0] << 18) | ((unsigned)q[1] << 12) |
                     ((unsigned)q[2] << 6) | (unsigned)q[3];
        if (o < dec_len) out[o++] = (unsigned char)((v >> 16) & 0xFF);
        if (o < dec_len) out[o++] = (unsigned char)((v >> 8) & 0xFF);
        if (o < dec_len) out[o++] = (unsigned char)(v & 0xFF);
    }
    if (out_len)
        *out_len = dec_len;
    return out;
}
