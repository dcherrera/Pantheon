/* SPDX-License-Identifier: AGPL-3.0-or-later */
/* Copyright (C) 2026 David (Corey) Herrera */
/**
 * @file attn_residency_json.c
 * @brief Implementation of the attention-path residency JSON writers
 *        (see attn_residency_json.h).
 *
 * @status None
 * @issues None
 * @todo None
 */

#include "attn_residency_json.h"

/* One class object: {"ondie":..,"host":..,"total":..}. `gave_up` and
 * `impossible` are emitted only when non-zero — they are the loud cases and a
 * zero would be noise on every healthy read. */
static void ar_write_counts(json_writer *jw, const pn_attn_residency *c)
{
    jw_printf(jw, "{\"ondie\":%d,\"host\":%d,\"total\":%d",
              c->ondie, c->host, c->total);
    if (c->gave_up > 0)
        jw_printf(jw, ",\"gave_up\":%d", c->gave_up);
    if (c->impossible > 0)
        jw_printf(jw, ",\"impossible\":%d", c->impossible);
    jw_raw(jw, "}");
}

/* `,"reasons":{"<name>":n,...}` over the reasons any layer is CURRENTLY on.
 * Omitted entirely when nothing is on the host path, so a healthy model does
 * not carry an object full of zeros. */
static void ar_write_reasons(json_writer *jw, const pn_attn_residency *c)
{
    if (c->host == 0)
        return;
    jw_raw(jw, ",\"reasons\":{");
    int first = 1;
    for (int i = 1; i < PN_ATTN_HOST_REASON_COUNT; i++) {
        if (c->by_reason[i] == 0)
            continue;
        jw_printf(jw, "%s\"%s\":%d", first ? "" : ",",
                  pn_attn_host_reason_name((unsigned char)i), c->by_reason[i]);
        first = 0;
    }
    jw_raw(jw, "}");
}

void ar_write_summary(json_writer *jw, const pn_model_attn_residency *r)
{
    if (!jw || !r || r->all.total == 0)
        return;
    jw_raw(jw, ",\"attention\":");
    ar_write_counts(jw, &r->all);
}

void ar_write_detail(json_writer *jw, const pn_model_attn_residency *r)
{
    if (!jw || !r || r->all.total == 0)
        return;
    jw_printf(jw, ",\"attention\":{\"ondie\":%d,\"host\":%d,\"total\":%d,"
                  "\"gave_up\":%d,\"impossible\":%d,"
                  "\"host_engagements\":%u,\"ring_alloc_failures\":%u",
              r->all.ondie, r->all.host, r->all.total,
              r->all.gave_up, r->all.impossible,
              r->all.lifetime_host_engagements,
              r->all.lifetime_alloc_failures);
    ar_write_reasons(jw, &r->all);
    jw_raw(jw, "}");
}
