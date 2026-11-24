/* SPDX-License-Identifier: AGPL-3.0-or-later */
/* Copyright (C) 2026 David (Corey) Herrera */
/**
 * @file serve_request_dump.c
 * @brief Implementation of the request-body capture (see serve_request_dump.h).
 *
 * status: used by openai_chat.c.
 * issues: none.
 * todo:   none.
 */
#include "serve_request_dump.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/time.h>

void sg_request_dump_body(const char *body)
{
    static unsigned long dump_sequence = 0;
    const char *dir = getenv("PN_SERVE_REQ_DUMP");
    struct timeval now;
    char path[4096];
    FILE *out;
    size_t length;

    if (!dir || !dir[0] || !body)
        return;
    gettimeofday(&now, NULL);
    snprintf(path, sizeof(path), "%s/req_%lld_%04lu.json", dir,
             (long long)now.tv_sec * 1000LL + now.tv_usec / 1000, dump_sequence++);
    out = fopen(path, "wb");
    if (!out) {
        fprintf(stderr, "[req_dump] cannot write %s\n", path);
        return;
    }
    length = strlen(body);
    if (fwrite(body, 1, length, out) != length)
        fprintf(stderr, "[req_dump] short write %s\n", path);
    fclose(out);
}
