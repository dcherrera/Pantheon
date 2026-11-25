/* SPDX-License-Identifier: AGPL-3.0-or-later */
/* Copyright (C) 2026 David (Corey) Herrera */
/**
 * @file usage_record.c
 * @brief Implementation of usage_record.h.
 *
 * @status Complete
 * @issues None
 * @todo None
 */

#include "usage_record.h"

#include <stdio.h>

int pn_usage_line_format(char *buf, size_t cap, const char *id, const char *tenant,
                         const char *model, const pn_usage_counts *u,
                         const char *finish)
{
    static const pn_usage_counts none = {0};

    if (!u)
        u = &none;
    return snprintf(buf, cap,
                    "[usage] id=%s tenant=%s model=%s prompt=%d cached=%d "
                    "completion=%d choices=%d finish=%s",
                    id && id[0] ? id : "-", tenant && tenant[0] ? tenant : "-",
                    model && model[0] ? model : "-", u->prompt, u->cached,
                    u->completion, u->choices, finish && finish[0] ? finish : "-");
}

void pn_usage_line_log(const char *id, const char *tenant, const char *model,
                       const pn_usage_counts *u, const char *finish)
{
    char line[512];

    pn_usage_line_format(line, sizeof(line), id, tenant, model, u, finish);
    fprintf(stderr, "%s\n", line);
}
