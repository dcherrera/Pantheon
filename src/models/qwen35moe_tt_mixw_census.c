/* SPDX-License-Identifier: AGPL-3.0-or-later */
/* Copyright (C) 2026 David (Corey) Herrera */
/**
 * @file qwen35moe_tt_mixw_census.c
 * @brief Implementation of the qwen35 dense hybrid's weight placement census
 *        (qwen35moe_tt_mixw_census.h).
 *
 * @status None
 * @issues None
 * @todo None
 */

#include "qwen35moe_tt_mixw_census.h"

#include <stdio.h>
#include <string.h>

/* The converter's refusals (foundry tt/convert ft_tt_ledger_map_refusal()), word for word in
 * substance, so the fit refuses what the converter refuses. */
static const char MIXW_REFUSE_SSM_GATE[] =
    "ssm_alpha / ssm_beta stay where the ledger puts them (tiny, unmeasured under block float)";
static const char MIXW_REFUSE_EMBEDDING[] =
    "token_embd is a host-side row table: a Bfp4 copy frees no card DRAM and the runtime has "
    "no Bfp4 row type";

/** Append one placement; names are copied into the census. @return 0, or -1 when full. */
static int mixw_add(pn_qwen35_tt_mixw_census *c, const char *placement, const char *group,
                    const char *refusal, uint32_t rows, uint32_t cols, uint32_t parts,
                    pn_tt_mixw_placement_kind kind)
{
    pn_tt_mixw_placement *p;

    if (c->count >= PN_QWEN35_TT_MIXW_MAX_PLACEMENTS)
        return -1;
    p = &c->placements[c->count];
    snprintf(c->names[c->count][0], PN_QWEN35_TT_MIXW_NAME_BYTES, "%s", placement);
    snprintf(c->names[c->count][1], PN_QWEN35_TT_MIXW_NAME_BYTES, "%s", group);
    p->placement_name = c->names[c->count][0];
    p->group_name = c->names[c->count][1];
    p->map_refusal = refusal;
    p->tile_rows = rows;
    p->tile_cols = cols;
    p->parts = parts;
    p->kind = kind;
    c->count++;
    return 0;
}

/** One trunk block's N-major entries, in the directory's (name) order. */
static int mixw_add_block(pn_qwen35_tt_mixw_census *c, const pn_qwen35_tt_mixw_shape *s,
                          uint32_t layer)
{
    const uint32_t e = s->embd_tiles, h = s->head_tiles;
    const struct { const char *suffix; uint32_t rows, cols, parts; const char *refusal; } ssm[] = {
        {"attn_gate", e, s->inner_tiles, 1, NULL},  {"attn_qkv", e, s->conv_tiles, 1, NULL},
        {"ffn_down", s->ffn_tiles, e, 1, NULL},     {"ffn_gate", e, s->ffn_tiles, 1, NULL},
        {"ffn_up", e, s->ffn_tiles, 1, NULL},
        {"ssm_alpha", e, s->dt_tiles, 1, MIXW_REFUSE_SSM_GATE},
        {"ssm_beta", e, s->dt_tiles, 1, MIXW_REFUSE_SSM_GATE},
        {"ssm_out", s->inner_tiles, e, 1, NULL},
    }, attn[] = {
        {"attn_k", e, s->kv_tiles, 1, NULL},        {"attn_output", h, e, 1, NULL},
        {"attn_q", e, 2u * h, s->split_q_at_load ? 2u : 1u, NULL},
        {"attn_v", e, s->kv_tiles, 1, NULL},        {"ffn_down", s->ffn_tiles, e, 1, NULL},
        {"ffn_gate", e, s->ffn_tiles, 1, NULL},     {"ffn_up", e, s->ffn_tiles, 1, NULL},
    };
    const bool attention = (layer + 1u) % s->attn_interval == 0u;
    const size_t n = attention ? sizeof(attn) / sizeof(attn[0]) : sizeof(ssm) / sizeof(ssm[0]);
    char name[PN_QWEN35_TT_MIXW_NAME_BYTES];

    for (size_t i = 0; i < n; i++) {
        const char *suffix = attention ? attn[i].suffix : ssm[i].suffix;

        snprintf(name, sizeof(name), "blk.%u.%s.weight", layer, suffix);
        if (mixw_add(c, name, name, attention ? attn[i].refusal : ssm[i].refusal,
                     attention ? attn[i].rows : ssm[i].rows,
                     attention ? attn[i].cols : ssm[i].cols,
                     attention ? attn[i].parts : ssm[i].parts, PN_TT_MIXW_PLACE_LOAD) != 0)
            return -1;
    }
    return 0;
}

/** Two staged halves of one `attn_q` (pn_qwen35_tt_attn_split_q): Q, then the gate. */
static int mixw_add_q_halves(pn_qwen35_tt_mixw_census *c, const pn_qwen35_tt_mixw_shape *s,
                             const char *placement_prefix, const char *group)
{
    char name[PN_QWEN35_TT_MIXW_NAME_BYTES];

    snprintf(name, sizeof(name), "%s (Q half)", placement_prefix);
    if (mixw_add(c, name, group, NULL, s->embd_tiles, s->head_tiles, 1u,
                 PN_TT_MIXW_PLACE_STAGED) != 0)
        return -1;
    snprintf(name, sizeof(name), "%s (gate half)", placement_prefix);
    return mixw_add(c, name, group, NULL, s->embd_tiles, s->head_tiles, 1u,
                    PN_TT_MIXW_PLACE_STAGED);
}

/** The MTP head in pn_qwen35_tt_mtp_arm() order; GGUF block L = n_layer. */
static int mixw_add_mtp(pn_qwen35_tt_mixw_census *c, const pn_qwen35_tt_mixw_shape *s)
{
    const uint32_t e = s->embd_tiles, L = s->n_layer;
    const struct { const char *placement, *group_suffix; uint32_t rows, cols; } place[] = {
        {"mtp.eh_proj_e.weight", "nextn.eh_proj", e, e},
        {"mtp.eh_proj_h.weight", "nextn.eh_proj", e, e},
        {"mtp.attn_k.weight", "attn_k", e, s->kv_tiles},
        {"mtp.attn_v.weight", "attn_v", e, s->kv_tiles},
        {"mtp.attn_output.weight", "attn_output", s->head_tiles, e},
        {"mtp.ffn_gate.weight", "ffn_gate", e, s->ffn_tiles},
        {"mtp.ffn_up.weight", "ffn_up", e, s->ffn_tiles},
        {"mtp.ffn_down.weight", "ffn_down", s->ffn_tiles, e},
    };
    char group[PN_QWEN35_TT_MIXW_NAME_BYTES];

    for (size_t i = 0; i < sizeof(place) / sizeof(place[0]); i++) {
        snprintf(group, sizeof(group), "blk.%u.%s.weight", L, place[i].group_suffix);
        if (mixw_add(c, place[i].placement, group, NULL, place[i].rows, place[i].cols, 1u,
                     PN_TT_MIXW_PLACE_STAGED) != 0)
            return -1;
    }
    snprintf(group, sizeof(group), "blk.%u.attn_q.weight", L);
    if (mixw_add_q_halves(c, s, "mtp.attn_q.weight", group) != 0)
        return -1;
    if (s->draft_cols == 0u)
        return 0;
    return mixw_add(c, "mtp.draft_vocab.weight",
                    s->draft_repacked ? "mtp.draft_vocab.weight" : "output.weight", NULL, e,
                    s->draft_cols, 1u, PN_TT_MIXW_PLACE_STAGED);
}

int pn_qwen35_tt_mixw_census_build(const pn_qwen35_tt_mixw_shape *s,
                                   pn_qwen35_tt_mixw_census *out)
{
    char name[PN_QWEN35_TT_MIXW_NAME_BYTES];

    if (!s || !out || s->n_layer == 0u || s->attn_interval == 0u || s->embd_tiles == 0u)
        return -1;
    out->count = 0u;
    if (mixw_add(out, "output.weight", "output.weight", NULL, s->embd_tiles, s->vocab_tiles, 1u,
                 PN_TT_MIXW_PLACE_LOAD) != 0 ||
        mixw_add(out, "token_embd.weight", "token_embd.weight", MIXW_REFUSE_EMBEDDING,
                 s->vocab_tiles, s->embd_tiles, 1u, PN_TT_MIXW_PLACE_HOST) != 0)
        return -1;
    for (uint32_t layer = 0; layer < s->n_layer; layer++)
        if (mixw_add_block(out, s, layer) != 0)
            return -1;
    for (uint32_t layer = 0; !s->split_q_at_load && layer < s->n_layer; layer++) {
        if ((layer + 1u) % s->attn_interval != 0u)
            continue;
        snprintf(name, sizeof(name), "blk.%u.attn_q.weight", layer);
        if (mixw_add_q_halves(out, s, name, name) != 0)
            return -1;
    }
    return s->mtp ? mixw_add_mtp(out, s) : 0;
}
