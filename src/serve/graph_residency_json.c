/* SPDX-License-Identifier: AGPL-3.0-or-later */
/* Copyright (C) 2026 David (Corey) Herrera */
/**
 * @file graph_residency_json.c
 * @brief Implementation of the graphs_live/graphs_total JSON writers
 *        (see graph_residency_json.h).
 *
 * @status None
 * @issues None
 * @todo None
 */

#include "graph_residency_json.h"

/* One class object: {"live":..,"total":..,"failed":..,"ineligible":..,
 * "pending":..}. `gave_up` is emitted only when non-zero — it is the loud
 * case (a slot that spent its retry budget and stopped trying) and a zero
 * would just be noise on every healthy read. */
static void gr_write_counts(json_writer *jw, const pn_graph_residency *c)
{
    jw_printf(jw, "{\"live\":%d,\"total\":%d,\"failed\":%d,"
                  "\"ineligible\":%d,\"pending\":%d",
              c->live, c->total, c->failed, c->ineligible, c->pending);
    if (c->gave_up > 0)
        jw_printf(jw, ",\"gave_up\":%d", c->gave_up);
    jw_raw(jw, "}");
}

/* Emit `,"<name>":{...}` for a class the model has actually exercised. */
static void gr_write_class(json_writer *jw, const char *name,
                           const pn_graph_residency *c)
{
    if (c->total == 0)
        return;
    jw_printf(jw, ",\"%s\":", name);
    gr_write_counts(jw, c);
}

void gr_write_summary(json_writer *jw, const pn_model_graph_residency *r)
{
    if (!jw || !r || r->all.total == 0)
        return;
    jw_raw(jw, ",\"graphs\":");
    gr_write_counts(jw, &r->all);
}

void gr_write_detail(json_writer *jw, const pn_model_graph_residency *r)
{
    if (!jw || !r || r->all.total == 0)
        return;
    jw_printf(jw, ",\"graphs\":{\"live\":%d,\"total\":%d,\"failed\":%d,"
                  "\"gave_up\":%d,\"ineligible\":%d,\"pending\":%d,"
                  "\"captures\":%u,\"capture_failures\":%u",
              r->all.live, r->all.total, r->all.failed, r->all.gave_up,
              r->all.ineligible, r->all.pending,
              r->all.lifetime_captures, r->all.lifetime_failures);
    gr_write_class(jw, "batched_attn", &r->batched_attn);
    gr_write_class(jw, "batched_ssm", &r->batched_ssm);
    gr_write_class(jw, "single_ssm", &r->single_ssm);
    gr_write_class(jw, "segment", &r->segment);
    jw_raw(jw, "}");
}
