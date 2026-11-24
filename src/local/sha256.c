/* SPDX-License-Identifier: AGPL-3.0-or-later */
/* Copyright (C) 2026 David (Corey) Herrera */
/**
 * @file   sha256.c
 * @brief  Implementation of SHA-256 and HMAC-SHA256 (see sha256.h). The
 *         transform, update and final are ported from GiTeam app/src/util/hash.c.
 *
 * @status None
 * @issues None
 * @todo   None
 */

#include "sha256.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static const uint32_t SHA256_K[64] = {
    0x428a2f98, 0x71374491, 0xb5c0fbcf, 0xe9b5dba5, 0x3956c25b, 0x59f111f1, 0x923f82a4,
    0xab1c5ed5, 0xd807aa98, 0x12835b01, 0x243185be, 0x550c7dc3, 0x72be5d74, 0x80deb1fe,
    0x9bdc06a7, 0xc19bf174, 0xe49b69c1, 0xefbe4786, 0x0fc19dc6, 0x240ca1cc, 0x2de92c6f,
    0x4a7484aa, 0x5cb0a9dc, 0x76f988da, 0x983e5152, 0xa831c66d, 0xb00327c8, 0xbf597fc7,
    0xc6e00bf3, 0xd5a79147, 0x06ca6351, 0x14292967, 0x27b70a85, 0x2e1b2138, 0x4d2c6dfc,
    0x53380d13, 0x650a7354, 0x766a0abb, 0x81c2c92e, 0x92722c85, 0xa2bfe8a1, 0xa81a664b,
    0xc24b8b70, 0xc76c51a3, 0xd192e819, 0xd6990624, 0xf40e3585, 0x106aa070, 0x19a4c116,
    0x1e376c08, 0x2748774c, 0x34b0bcb5, 0x391c0cb3, 0x4ed8aa4a, 0x5b9cca4f, 0x682e6ff3,
    0x748f82ee, 0x78a5636f, 0x84c87814, 0x8cc70208, 0x90befffa, 0xa4506ceb, 0xbef9a3f7,
    0xc67178f2
};

#define ROTR(x, n)    (((x) >> (n)) | ((x) << (32 - (n))))
#define CH(x, y, z)   (((x) & (y)) ^ (~(x) & (z)))
#define MAJ(x, y, z)  (((x) & (y)) ^ ((x) & (z)) ^ ((y) & (z)))
#define EP0(x)        (ROTR(x, 2) ^ ROTR(x, 13) ^ ROTR(x, 22))
#define EP1(x)        (ROTR(x, 6) ^ ROTR(x, 11) ^ ROTR(x, 25))
#define SIG0(x)       (ROTR(x, 7) ^ ROTR(x, 18) ^ ((x) >> 3))
#define SIG1(x)       (ROTR(x, 17) ^ ROTR(x, 19) ^ ((x) >> 10))

static void sha256_transform(uint32_t state[8], const uint8_t block[64])
{
    uint32_t w[64];

    for (int i = 0; i < 16; i++)
        w[i] = ((uint32_t)block[i * 4] << 24) | ((uint32_t)block[i * 4 + 1] << 16) |
               ((uint32_t)block[i * 4 + 2] << 8) | ((uint32_t)block[i * 4 + 3]);
    for (int i = 16; i < 64; i++)
        w[i] = SIG1(w[i - 2]) + w[i - 7] + SIG0(w[i - 15]) + w[i - 16];

    uint32_t a = state[0], b = state[1], c = state[2], d = state[3];
    uint32_t e = state[4], f = state[5], g = state[6], h = state[7];
    for (int i = 0; i < 64; i++) {
        uint32_t t1 = h + EP1(e) + CH(e, f, g) + SHA256_K[i] + w[i];
        uint32_t t2 = EP0(a) + MAJ(a, b, c);
        h = g; g = f; f = e; e = d + t1;
        d = c; c = b; b = a; a = t1 + t2;
    }
    state[0] += a; state[1] += b; state[2] += c; state[3] += d;
    state[4] += e; state[5] += f; state[6] += g; state[7] += h;
}

void pn_sha256_init(pn_sha256_ctx *ctx)
{
    static const uint32_t iv[8] = { 0x6a09e667, 0xbb67ae85, 0x3c6ef372, 0xa54ff53a,
                                    0x510e527f, 0x9b05688c, 0x1f83d9ab, 0x5be0cd19 };
    memcpy(ctx->state, iv, sizeof(iv));
    ctx->count = 0;
    memset(ctx->buffer, 0, sizeof(ctx->buffer));
}

void pn_sha256_update(pn_sha256_ctx *ctx, const void *data, size_t len)
{
    const uint8_t *bytes = data;
    size_t used = (size_t)(ctx->count % 64);

    ctx->count += len;
    if (used > 0) {
        size_t space = 64 - used;
        if (len < space) {
            memcpy(ctx->buffer + used, bytes, len);
            return;
        }
        memcpy(ctx->buffer + used, bytes, space);
        sha256_transform(ctx->state, ctx->buffer);
        bytes += space;
        len -= space;
    }
    while (len >= 64) {
        sha256_transform(ctx->state, bytes);
        bytes += 64;
        len -= 64;
    }
    if (len > 0)
        memcpy(ctx->buffer, bytes, len);
}

void pn_sha256_final(pn_sha256_ctx *ctx, uint8_t digest[PN_SHA256_LEN])
{
    uint64_t bits = ctx->count * 8;
    size_t used = (size_t)(ctx->count % 64);

    ctx->buffer[used++] = 0x80;
    if (used > 56) {
        memset(ctx->buffer + used, 0, 64 - used);
        sha256_transform(ctx->state, ctx->buffer);
        used = 0;
    }
    memset(ctx->buffer + used, 0, 56 - used);
    for (int i = 0; i < 8; i++)
        ctx->buffer[56 + i] = (uint8_t)(bits >> (56 - i * 8));
    sha256_transform(ctx->state, ctx->buffer);
    for (int i = 0; i < 8; i++) {
        digest[i * 4]     = (uint8_t)(ctx->state[i] >> 24);
        digest[i * 4 + 1] = (uint8_t)(ctx->state[i] >> 16);
        digest[i * 4 + 2] = (uint8_t)(ctx->state[i] >> 8);
        digest[i * 4 + 3] = (uint8_t)(ctx->state[i]);
    }
}

void pn_sha256(const void *data, size_t len, uint8_t digest[PN_SHA256_LEN])
{
    pn_sha256_ctx ctx;

    pn_sha256_init(&ctx);
    pn_sha256_update(&ctx, data, len);
    pn_sha256_final(&ctx, digest);
}

/* memset that the compiler may not drop (the bytes are dead afterwards). */
static void wipe_bytes(void *p, size_t n)
{
    volatile uint8_t *v = p;

    while (n--)
        *v++ = 0;
}

void pn_hmac_sha256(const void *key, size_t klen, const void *msg, size_t mlen,
                    uint8_t mac[PN_SHA256_LEN])
{
    enum { BLOCK = 64 };
    uint8_t k0[BLOCK] = {0}, pad[BLOCK], inner[PN_SHA256_LEN];
    pn_sha256_ctx ctx;

    if (klen > BLOCK)
        pn_sha256(key, klen, k0);
    else if (klen)
        memcpy(k0, key, klen);
    for (int i = 0; i < BLOCK; i++)
        pad[i] = k0[i] ^ 0x36;
    pn_sha256_init(&ctx);
    pn_sha256_update(&ctx, pad, BLOCK);
    pn_sha256_update(&ctx, msg, mlen);
    pn_sha256_final(&ctx, inner);
    for (int i = 0; i < BLOCK; i++)
        pad[i] = k0[i] ^ 0x5c;
    pn_sha256_init(&ctx);
    pn_sha256_update(&ctx, pad, BLOCK);
    pn_sha256_update(&ctx, inner, PN_SHA256_LEN);
    pn_sha256_final(&ctx, mac);
    wipe_bytes(k0, sizeof(k0));
    wipe_bytes(pad, sizeof(pad));
    wipe_bytes(inner, sizeof(inner));
    wipe_bytes(&ctx, sizeof(ctx));
}

void pn_hex_encode(const uint8_t *bytes, size_t n, char *hex)
{
    static const char digits[] = "0123456789abcdef";

    for (size_t i = 0; i < n; i++) {
        hex[i * 2] = digits[bytes[i] >> 4];
        hex[i * 2 + 1] = digits[bytes[i] & 0x0f];
    }
    hex[n * 2] = '\0';
}

static int hex_value(char c)
{
    if (c >= '0' && c <= '9') return c - '0';
    if (c >= 'a' && c <= 'f') return c - 'a' + 10;
    if (c >= 'A' && c <= 'F') return c - 'A' + 10;
    return -1;
}

int pn_hex_decode(const char *hex, uint8_t *bytes, size_t n)
{
    for (size_t i = 0; i < n; i++) {
        int hi = hex_value(hex[i * 2]);
        int lo = hi < 0 ? -1 : hex_value(hex[i * 2 + 1]);
        if (hi < 0 || lo < 0)
            return -1;
        bytes[i] = (uint8_t)((hi << 4) | lo);
    }
    return 0;
}

int pn_sha256_file(const char *path, uint8_t digest[PN_SHA256_LEN])
{
    enum { CHUNK = 1 << 20 };
    FILE *f = fopen(path, "rb");
    uint8_t *buf = NULL;
    pn_sha256_ctx ctx;
    int rc = -1;

    if (!f)
        return -1;
    buf = malloc(CHUNK);
    if (!buf)
        goto done;
    pn_sha256_init(&ctx);
    for (;;) {
        size_t n = fread(buf, 1, CHUNK, f);
        if (n > 0)
            pn_sha256_update(&ctx, buf, n);
        if (n < CHUNK)
            break;
    }
    if (!ferror(f)) {
        pn_sha256_final(&ctx, digest);
        rc = 0;
    }
done:
    free(buf);
    fclose(f);
    return rc;
}
