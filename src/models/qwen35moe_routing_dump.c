/* SPDX-License-Identifier: AGPL-3.0-or-later */
/* Copyright (C) 2026 David (Corey) Herrera */
/**
 * @file   qwen35moe_routing_dump.c
 * @brief  Implementation of the routing dump (see the header for the record).
 *
 * @status None
 * @issues None
 * @todo   None
 */
#include "qwen35moe_routing_dump.h"

#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>

/** The dump file, opened on first use; NULL when the variable is unset or unopenable. */
static FILE *routing_dump_file(void)
{
    static int tried;
    static FILE *file;
    const char *path;

    if (tried)
        return file;
    tried = 1;
    path = getenv("QWEN35_ROUTING_DUMP");
    if (!path || !*path)
        return NULL;
    file = fopen(path, "wb");
    if (!file)
        fprintf(stderr, "qwen35moe: QWEN35_ROUTING_DUMP=%s cannot be opened — no routing dump\n", path);
    return file;
}

void pn_qwen35moe_routing_dump(const float *probs, int n_expert, const int *top_idx, int k)
{
    FILE *file = routing_dump_file();
    float next = -1.0f;

    if (!file)
        return;
    for (int e = 0; e < n_expert; e++) {
        int chosen = 0;

        for (int s = 0; s < k && !chosen; s++)
            chosen = top_idx[s] == e;
        if (!chosen && probs[e] > next)
            next = probs[e];
    }
    for (int s = 0; s < k; s++) {
        int32_t idx = top_idx[s];

        fwrite(&idx, sizeof(idx), 1, file);
    }
    for (int s = 0; s < k; s++)
        fwrite(&probs[top_idx[s]], sizeof(float), 1, file);
    fwrite(&next, sizeof(next), 1, file);
}
