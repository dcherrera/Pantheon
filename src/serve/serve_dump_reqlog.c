/* SPDX-License-Identifier: AGPL-3.0-or-later */
/* Copyright (C) 2026 David (Corey) Herrera */
/**
 * @file serve_dump_reqlog.c
 * @brief Prompt hash + request line for the hidden-dump request log. See serve_dump_reqlog.h.
 *
 * status: complete.
 * issues: none.
 * todo:   none.
 */
#include "serve_dump_reqlog.h"

#include "../local/sha256.h"

int sg_dump_reqlog_request(pn_dump_reqlog_request *r, const uint32_t *prompt, int n_prompt)
{
    pn_sha256_ctx ctx;
    uint8_t digest[PN_SHA256_LEN], le[4];
    char hex[PN_SHA256_HEX_LEN + 1];
    int i;

    pn_sha256_init(&ctx);
    for (i = 0; i < n_prompt; i++) {
        le[0] = (uint8_t)prompt[i];
        le[1] = (uint8_t)(prompt[i] >> 8);
        le[2] = (uint8_t)(prompt[i] >> 16);
        le[3] = (uint8_t)(prompt[i] >> 24);
        pn_sha256_update(&ctx, le, sizeof(le));
    }
    pn_sha256_final(&ctx, digest);
    pn_hex_encode(digest, sizeof(digest), hex);
    r->prompt_sha256_hex = hex;
    return pn_dump_reqlog_write_request(r);
}
