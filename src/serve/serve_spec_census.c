/* SPDX-License-Identifier: AGPL-3.0-or-later */
/* Copyright (C) 2026 David (Corey) Herrera */
/**
 * @file serve_spec_census.c
 * @brief Implementation of the per-request speculation census (see serve_spec_census.h).
 *
 * status: used by serve_generation.c.
 * issues: none.
 * todo:   none.
 */
#include "serve_spec_census.h"

#include <stdio.h>
#include <string.h>

void sg_spec_census_begin(sg_spec_census *c, const char *mode)
{
    memset(c, 0, sizeof(*c));
    c->mode = mode ? mode : "off";
}

void sg_spec_census_plain(sg_spec_census *c, sg_plain_reason reason)
{
    if ((int)reason >= 0 && reason < SG_PLAIN_REASONS)
        c->plain[reason]++;
}

void sg_spec_census_verified(sg_spec_census *c)
{
    c->from_verify++;
}

void sg_spec_census_step(sg_spec_census *c, int masked, double ms)
{
    c->region_tokens[masked ? 1 : 0]++;
    c->region_ms[masked ? 1 : 0] += ms;
}

void sg_spec_census_report(const sg_spec_census *c, int generated)
{
    unsigned long plain = 0;
    int r;

    for (r = 0; r < SG_PLAIN_REASONS; r++)
        plain += c->plain[r];
    fprintf(stderr, "[spec] census: mode=%s gen=%d verify=%lu plain=%lu "
                    "(trip %lu cap %lu nodraft %lu failed %lu off %lu) "
                    "| free %lu tok %.1f ms | masked %lu tok %.1f ms\n",
            c->mode, generated, c->from_verify, plain, c->plain[SG_PLAIN_TRIP],
            c->plain[SG_PLAIN_CAP], c->plain[SG_PLAIN_NODRAFT], c->plain[SG_PLAIN_FAILED],
            c->plain[SG_PLAIN_OFF], c->region_tokens[0], c->region_ms[0], c->region_tokens[1],
            c->region_ms[1]);
}
