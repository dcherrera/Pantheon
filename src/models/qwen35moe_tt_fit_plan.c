/* SPDX-License-Identifier: AGPL-3.0-or-later */
/* Copyright (C) 2026 David (Corey) Herrera */
/**
 * @file qwen35moe_tt_fit_plan.c
 * @brief Implementation of main's DRAM floor and the fit-policy trigger
 *        (qwen35moe_tt_fit_plan.h).
 *
 * @status None
 * @issues None
 * @todo None
 */

#include "qwen35moe_tt_fit_plan.h"

#include <stdint.h>
#include <string.h>

/** A card tile is 32 x 32; fp32 tiles (the GDN family) are 4 KiB, bf16 strip tiles 2 KiB. */
#define FIT_TILE_DIM 32u
#define FIT_FP32_TILE_BYTES 4096ull
#define FIT_BF16_TILE_BYTES 2048ull

static uint64_t fit_tiles(uint32_t features)
{
    return (features + FIT_TILE_DIM - 1u) / FIT_TILE_DIM;
}

uint64_t pn_qwen35_tt_fit_tall_tiles(const pn_qwen35_tt_fit_shape *s)
{
    const uint64_t embd = fit_tiles(s->n_embd), ffn = fit_tiles(s->ffn);
    const uint64_t inner = fit_tiles(s->d_inner), heads = fit_tiles(s->n_head * s->head_dim);
    const uint64_t kv = fit_tiles(s->n_head_kv * s->head_dim);
    const uint64_t gated_copies = s->outproj_mirrors ? s->mirrors : 1u;
    /* pn_qwen35_tt_strips in prefill_strips_of order: hidden x2, normed (mirrored), project,
     * ssm_qkv, ssm_z, alpha, beta, ssm_gated, attn_qg, attn_k, attn_v, attn_out, ffn_gate,
     * ffn_up, ffn_swiglu (mirrored); the logit row has no tall copy. */
    uint64_t tiles = 3u * embd + embd * s->mirrors + fit_tiles(s->conv_dim) + inner +
                     2u * fit_tiles(s->v_heads) + inner * gated_copies + 2u * heads + 2u * kv +
                     heads + 2u * ffn + ffn * s->mirrors;

    /* The attention arm's strips: q, q_rot, z, gate, gated (mirrored) and k_rot. */
    if (s->attn_on_card)
        tiles += 4u * heads + heads * gated_copies + kv;
    return tiles;
}

/** K + V bytes of one position of one attention layer: heads x dim elements of K at K's tile
 *  size plus as many of V at V's (2,048 B bf16; 1,088 Bfp8_b; 576 Bfp4_b — pn_tt_kv_format.h;
 *  K and V may differ). */
static uint64_t kv_position_bytes(const pn_qwen35_tt_fit_shape *s)
{
    const uint64_t k_tile = s->kv_k_tile_bytes ? s->kv_k_tile_bytes : FIT_BF16_TILE_BYTES;
    const uint64_t v_tile = s->kv_v_tile_bytes ? s->kv_v_tile_bytes : FIT_BF16_TILE_BYTES;

    return (uint64_t)s->n_head_kv * s->head_dim * (k_tile + v_tile) / 1024ull;
}

void pn_qwen35_tt_fit_floor_compute(const pn_qwen35_tt_fit_shape *s, uint64_t weight_bytes,
                                    uint64_t joint_q_bytes, pn_qwen35_tt_fit_floor *out)
{
    const uint64_t conv_tiles = fit_tiles(s->conv_dim);
    const uint64_t state_tiles = fit_tiles(s->ssm_state);
    const uint64_t state = (uint64_t)s->v_heads * state_tiles * state_tiles * FIT_FP32_TILE_BYTES;
    const uint64_t window = conv_tiles * (s->conv_taps - 1u) * FIT_FP32_TILE_BYTES;
    const uint64_t conv_weight = conv_tiles * s->conv_taps * FIT_FP32_TILE_BYTES;

    memset(out, 0, sizeof(*out));
    out->weights = weight_bytes;
    /* Main uploads the joint AND stages its two halves beside it for on-card attention. */
    out->second_q = s->attn_on_card ? joint_q_bytes : 0u;
    out->kv = s->attn_on_card ? (uint64_t)s->n_attn * kv_position_bytes(s) * s->ctx : 0u;
    if (s->gdn_on_card && s->conv_taps > 0u) {
        out->gdn = (uint64_t)s->n_ssm * (state + window + conv_weight);
        /* The conv constant is the row-replicated weight plus a band: at least the weight. */
        out->gdnc = (uint64_t)s->n_ssm * conv_weight * s->gdnc_replicas;
        out->reserve = s->state_split ? (uint64_t)s->n_ssm * state : 0u;
    }
    out->tall = s->tall_rows ? pn_qwen35_tt_fit_tall_tiles(s) * s->tall_rows * FIT_BF16_TILE_BYTES
                             : 0u;
    /* MTP on: the head, its own KV region (one attention layer), and one twin set of every
     * GDN layer's window + state per verify row (pn_tt_gdn_snapshot_reserve). */
    if (s->mtp_bytes) {
        out->mtp = s->mtp_bytes - (s->draft_view ? s->mtp_draft_bytes : 0u);
        if (s->attn_on_card)
            out->mtp += kv_position_bytes(s) * s->ctx;
        if (s->gdn_on_card && s->conv_taps > 0u)
            out->mtp += (uint64_t)s->mtp_rows * s->n_ssm * (state + window);
    }
    if (s->gate_tables && s->gdn_on_card)
        out->gate_tables = pn_qwen35_tt_fit_gate_table_bytes(s->n_ssm, s->v_heads);
    out->total = out->weights + out->second_q + out->kv + out->gdn + out->gdnc + out->tall +
                 out->reserve + out->mtp + out->gate_tables;
}

uint64_t pn_qwen35_tt_fit_gate_table_bytes(uint32_t layers, uint32_t heads)
{
    const uint64_t stripes = PN_QWEN35_TT_FIT_TABLE_STRIPES;
    const uint64_t slots = (uint64_t)layers * ((heads + stripes - 1u) / stripes) * stripes;

    return (slots + stripes) * PN_QWEN35_TT_FIT_GATE_TABLE_SLOT + (uint64_t)layers * 256u;
}

uint32_t pn_qwen35_tt_fit_table_stripes(uint64_t *free_bytes, uint32_t banks,
                                        const uint64_t *layer_slot, uint32_t layers,
                                        uint32_t heads, uint64_t beta_bytes,
                                        uint64_t keep_free, uint32_t *bank_out)
{
    if (!free_bytes || !layer_slot || !bank_out || banks == 0u ||
        banks > PN_QWEN35_TT_FIT_MAX_BANKS || heads == 0u)
        return 0u;
    for (uint32_t stripes = PN_QWEN35_TT_FIT_TABLE_STRIPES; stripes >= 1u; stripes /= 2u) {
        uint64_t trial[PN_QWEN35_TT_FIT_MAX_BANKS];
        bool beta_in[PN_QWEN35_TT_FIT_MAX_BANKS] = {false};
        bool ok = stripes <= banks;

        memcpy(trial, free_bytes, banks * sizeof(trial[0]));
        for (uint32_t l = 0; ok && l < layers; l++) {
            const uint64_t stripe_bytes =
                (uint64_t)((heads + stripes - 1u) / stripes) * layer_slot[l];
            bool used[PN_QWEN35_TT_FIT_MAX_BANKS] = {false};

            for (uint32_t s = 0; ok && s < stripes; s++) {
                uint32_t best = banks;
                uint64_t need;

                for (uint32_t b = 0; b < banks; b++)
                    if (!used[b] && (best == banks || trial[b] > trial[best]))
                        best = b;
                need = stripe_bytes + (beta_in[best] ? 0u : beta_bytes);
                if (trial[best] < need + keep_free) {
                    ok = false;
                    break;
                }
                trial[best] -= need;
                used[best] = beta_in[best] = true;
                bank_out[l * PN_QWEN35_TT_FIT_TABLE_STRIPES + s] = best;
            }
        }
        if (ok) {
            memcpy(free_bytes, trial, banks * sizeof(trial[0]));
            return stripes;
        }
    }
    return 0u;
}

bool pn_qwen35_tt_fit_policy_wanted(const pn_qwen35_tt_fit_floor *floor, uint64_t card_bytes)
{
    return floor && floor->total + PN_QWEN35_TT_FIT_HEADROOM > card_bytes;
}

/* ---- The bank rule ---- */

uint64_t pn_qwen35_tt_fit_bank_floor(const pn_qwen35_tt_fit_shape *s,
                                     const pn_qwen35_tt_fit_floor *floor, bool checkpoint,
                                     uint32_t banks)
{
    const uint64_t head_kv = s->mtp_bytes && s->attn_on_card ? kv_position_bytes(s) * s->ctx : 0u;
    const uint32_t kv_layers = s->n_attn + (head_kv ? 1u : 0u);
    uint64_t kv_bank = 0u, even;

    if (banks == 0u)
        return 0u;
    even = floor->total - floor->kv - head_kv;
    if (checkpoint)
        even += pn_qwen35_tt_fit_snapshot_bytes(s);
    if (floor->kv)
        kv_bank = pn_qwen35_tt_fit_kv_bank_bytes(s, kv_layers, banks, true);
    if (kv_bank == 0u)
        even += floor->kv + head_kv;
    return (even + banks - 1u) / banks + kv_bank;
}

int64_t pn_qwen35_tt_fit_bank_low(const pn_qwen35_tt_fit_shape *s,
                                  const pn_qwen35_tt_fit_floor *floor, bool checkpoint,
                                  const uint64_t *free_bytes, uint32_t banks, uint32_t *bank_out)
{
    int64_t need, low = INT64_MAX;

    if (banks == 0u || banks > PN_QWEN35_TT_FIT_MAX_BANKS)
        return INT64_MIN;
    need = (int64_t)(pn_qwen35_tt_fit_bank_floor(s, floor, checkpoint, banks) +
                     PN_QWEN35_TT_FIT_HEADROOM / banks);
    for (uint32_t b = 0; b < banks; b++)
        if ((int64_t)free_bytes[b] - need < low) {
            low = (int64_t)free_bytes[b] - need;
            if (bank_out)
                *bank_out = b;
        }
    return low;
}

bool pn_qwen35_tt_fit_policy_wanted_banks(const pn_qwen35_tt_fit_shape *s,
                                          const pn_qwen35_tt_fit_floor *floor, bool checkpoint,
                                          const uint64_t *free_bytes, uint32_t banks)
{
    uint64_t card = 0u;

    if (banks == 0u || banks > PN_QWEN35_TT_FIT_MAX_BANKS)
        return false;
    for (uint32_t b = 0; b < banks; b++)
        card += free_bytes[b];
    return pn_qwen35_tt_fit_policy_wanted(floor, card) ||
           pn_qwen35_tt_fit_bank_low(s, floor, checkpoint, free_bytes, banks, NULL) <
               (int64_t)PN_QWEN35_TT_TIER_HEADROOM;
}

/* ---- The bank budget ---- */

/** The recurrent state of one SSM layer: v_heads x (S/32)^2 fp32 tiles. */
static uint64_t fit_layer_state(const pn_qwen35_tt_fit_shape *s)
{
    const uint64_t state_tiles = fit_tiles(s->ssm_state);

    return (uint64_t)s->v_heads * state_tiles * state_tiles * FIT_FP32_TILE_BYTES;
}

uint64_t pn_qwen35_tt_fit_state_bytes(const pn_qwen35_tt_fit_shape *s)
{
    return s ? (uint64_t)s->n_ssm * fit_layer_state(s) : 0u;
}

uint64_t pn_qwen35_tt_fit_snapshot_bytes(const pn_qwen35_tt_fit_shape *s)
{
    if (!s || s->conv_taps == 0u)
        return 0u;
    return (uint64_t)s->n_ssm * (fit_layer_state(s) + fit_tiles(s->conv_dim) *
                                                          (s->conv_taps - 1u) *
                                                          FIT_FP32_TILE_BYTES);
}

uint64_t pn_qwen35_tt_fit_kv_bank_bytes(const pn_qwen35_tt_fit_shape *s, uint32_t kv_layers,
                                        uint32_t banks, bool larger)
{
    const uint32_t set = banks / 2u;
    uint64_t k_tile, v_tile, big, small, group_tiles;
    uint32_t many, few;

    if (!s || set == 0u || banks % 2u != 0u || s->n_head_kv == 0u || s->n_head_kv % set != 0u)
        return 0u;
    k_tile = s->kv_k_tile_bytes ? s->kv_k_tile_bytes : FIT_BF16_TILE_BYTES;
    v_tile = s->kv_v_tile_bytes ? s->kv_v_tile_bytes : FIT_BF16_TILE_BYTES;
    big = k_tile > v_tile ? k_tile : v_tile;
    small = k_tile > v_tile ? v_tile : k_tile;
    /* One bank's group of one layer: its heads x dim tiles x position tiles. */
    group_tiles = (uint64_t)(s->n_head_kv / set) * fit_tiles(s->head_dim) * fit_tiles(s->ctx);
    many = (kv_layers + 1u) / 2u;
    few = kv_layers / 2u;
    return group_tiles * (larger ? many * big + few * small : few * big + many * small);
}

/*
 * THE GATE-BLOCK ARRAY, DERIVED. pn_tt_gdngate_arm(max_rows) sets max_bands = ceil(max_rows /
 * 32) and block_stride = align64(ftgg_block_bytes(heads)), ftgg_block_bytes(h) = (16 header
 * words + h x 64 words: a 32-row decay run and a 32-row beta run a head) x 4 B
 * (foundry tt/kernels/resident_gdngate_protocol.h). Each band holds 2 blocks, one per DELTA
 * sub-chunk (the gdnc band loop splits a 32-row band at C = 16), and every SSM layer owns its
 * own array. The DELTA descriptor reads its gate at the gdnc DELTA constant's packed NoC word,
 * so the whole array is ONE allocation in THAT bank. The prefill asks for one pass of the
 * tall strips: rows = tall tile rows x 32 = 2,048 by default. Qwen3.8, 48 layers x 48 heads:
 * stride (16 + 3,072) x 4 = 12,352 B, x 2 x 64 bands = 1,581,056 B a layer, 75,890,688 B in
 * all (72.4 MiB in one bank) — why the fit policy arms the family FLAT only (max_rows 0) today:
 * its census has no 72 MiB in any one bank at 30,720.
 */
#define FIT_GATE_HEADER_WORDS 16u
#define FIT_GATE_HEAD_WORDS 64u
#define FIT_GATE_SUBCHUNKS 2u
#define FIT_DRAM_ALIGN 64ull

uint64_t pn_qwen35_tt_fit_gate_array_bytes(uint32_t layers, uint32_t heads, uint32_t rows_a_pass)
{
    const uint64_t block = (FIT_GATE_HEADER_WORDS + (uint64_t)heads * FIT_GATE_HEAD_WORDS) * 4u;
    const uint64_t stride = (block + FIT_DRAM_ALIGN - 1u) & ~(FIT_DRAM_ALIGN - 1u);

    return (uint64_t)layers * FIT_GATE_SUBCHUNKS * fit_tiles(rows_a_pass) * stride;
}

/* FT_TT_OP_GDNC_GATE_STRIDE (foundry tt/ft_tt_op_gdnc.h) and PN_TT_GDNC_STAGE_SLOTS. */
#define FIT_GDNC_GATE_FLOATS 160u
#define FIT_GDNC_STAGE_SLOTS 2u

uint64_t pn_qwen35_tt_fit_gdnc_gate_bytes(uint32_t v_heads)
{
    return (uint64_t)FIT_GDNC_STAGE_SLOTS * v_heads * FIT_GDNC_GATE_FLOATS * sizeof(float);
}

/** The KV growth a bank can see from `anchor` to `load` (see the header). */
static int bank_fit_kv_growth(const pn_qwen35_tt_bank_load *anchor,
                              const pn_qwen35_tt_bank_load *load, uint32_t banks, int64_t *out)
{
    const pn_qwen35_tt_fit_shape *a = &anchor->shape, *l = &load->shape;
    const bool same_role = a->kv_k_tile_bytes == l->kv_k_tile_bytes &&
                           a->kv_v_tile_bytes == l->kv_v_tile_bytes &&
                           anchor->kv_layers == load->kv_layers;
    const uint64_t to = pn_qwen35_tt_fit_kv_bank_bytes(l, load->kv_layers, banks, true);
    const uint64_t from = pn_qwen35_tt_fit_kv_bank_bytes(a, anchor->kv_layers, banks, same_role);

    if ((load->kv_layers && !to) || (anchor->kv_layers && !from))
        return -1;
    *out = (int64_t)to - (int64_t)from;
    return 0;
}

int pn_qwen35_tt_bank_fit_compute(const pn_qwen35_tt_bank_load *anchor, uint64_t anchor_free,
                                  const pn_qwen35_tt_bank_load *load, uint32_t banks,
                                  uint64_t headroom, pn_qwen35_tt_bank_fit *out)
{
    int64_t even;

    if (!anchor || !load || !out || banks == 0u || banks > PN_QWEN35_TT_FIT_MAX_BANKS)
        return -1;
    memset(out, 0, sizeof(*out));
    if (bank_fit_kv_growth(anchor, load, banks, &out->kv_growth) != 0)
        return -1;
    /* Even classes divide over the banks; a growth rounds up, a saving down. */
    even = (int64_t)load->even_bytes - (int64_t)anchor->even_bytes;
    even = even >= 0 ? (even + (int64_t)banks - 1) / (int64_t)banks : even / (int64_t)banks;
    for (uint32_t b = 0; b < banks; b++) {
        out->free_bank[b] = (int64_t)anchor_free + (int64_t)anchor->weight_bank[b] -
                            (int64_t)load->weight_bank[b] - out->kv_growth - even;
        if (out->free_bank[b] < out->free_bank[out->binding_bank])
            out->binding_bank = b;
    }
    /* The one-bank class's bank is chosen at run time (the emptiest then): charge it where it
     * hurts most. A growth over the anchor only — the anchor's own sits in its measurement. */
    if (load->one_bank_bytes > anchor->one_bank_bytes)
        out->free_bank[out->binding_bank] -=
            (int64_t)(load->one_bank_bytes - anchor->one_bank_bytes);
    out->binding_free = out->free_bank[out->binding_bank];
    out->fits = out->binding_free >= 0 && (uint64_t)out->binding_free >= headroom;
    return 0;
}

/* ---- The levers ---- */

uint64_t pn_qwen35_tt_fit_policy_bank_floor(const pn_qwen35_tt_fit_shape *s,
                                            const pn_qwen35_tt_fit_floor *floor, bool checkpoint,
                                            uint32_t banks)
{
    const uint64_t head_kv = s->mtp_bytes && s->attn_on_card ? kv_position_bytes(s) * s->ctx : 0u;
    const uint32_t kv_layers = s->n_attn + (head_kv ? 1u : 0u);
    const uint64_t set = pn_qwen35_tt_fit_snapshot_bytes(s);
    uint64_t kv_bank = 0u, even;

    if (banks == 0u)
        return 0u;
    even = floor->total - floor->kv - head_kv - floor->second_q;
    /* D: the decode conv weight is gdnc copy 0's (pn_tt_gdn_conv_share.h). */
    if (s->gdn_on_card && s->conv_taps > 0u)
        even -= (uint64_t)s->n_ssm * fit_tiles(s->conv_dim) * s->conv_taps * FIT_FP32_TILE_BYTES;
    /* C: no tall joint attn_qg (two heads-widths) with attention on the card. */
    if (s->tall_rows && s->attn_on_card)
        even -= 2u * fit_tiles(s->n_head * s->head_dim) * s->tall_rows * FIT_BF16_TILE_BYTES;
    /* The compact verify map: rows - 1 slots (reserve_mtp_slots), the floor priced rows. */
    if (s->mtp_bytes && s->mtp_rows > 0u && s->gdn_on_card)
        even -= set;
    if (checkpoint)
        even += set;
    if (floor->kv)
        kv_bank = pn_qwen35_tt_fit_kv_bank_bytes(s, kv_layers, banks, true);
    if (kv_bank == 0u)
        even += floor->kv + head_kv;
    return (even + banks - 1u) / banks + kv_bank;
}

int64_t pn_qwen35_tt_fit_policy_bank_low(const pn_qwen35_tt_fit_shape *s, uint64_t weight_bytes,
                                         uint64_t joint_q_bytes, bool checkpoint,
                                         const uint64_t *free_bytes, uint32_t banks)
{
    pn_qwen35_tt_fit_floor floor;
    int64_t need, low = INT64_MAX;
    bool single;

    if (!s || !free_bytes || banks == 0u || banks > PN_QWEN35_TT_FIT_MAX_BANKS)
        return INT64_MIN;
    pn_qwen35_tt_fit_floor_compute(s, weight_bytes, joint_q_bytes, &floor);
    single = s->tall_rows && s->tall_rows <= 32u && !s->outproj_mirrors;
    need = (int64_t)(pn_qwen35_tt_fit_policy_bank_floor(s, &floor, checkpoint, banks) +
                     (single ? PN_QWEN35_TT_FIT_LEVER_ALLOWANCE_SINGLE
                             : PN_QWEN35_TT_FIT_LEVER_ALLOWANCE));
    for (uint32_t b = 0; b < banks; b++)
        if ((int64_t)free_bytes[b] - need < low)
            low = (int64_t)free_bytes[b] - need;
    return low;
}

/** The tall steps of the ladder, main's set first (rows 0 = keep the shape's). */
static const struct {
    uint32_t rows;
    bool outproj_mirrors;
} fit_tall_steps[] = { { 0u, true }, { 32u, true }, { 32u, false }, { 16u, false } };

/** The ladder over E, V, G/F at the shape's KV tiles: the first step that keeps the headroom.
 *  `out` gets the last step tried; `*tried` whether any was. */
static bool fit_ladder(const pn_qwen35_tt_fit_shape *shape, uint64_t weight_bytes,
                       uint64_t joint_q_bytes, bool checkpoint, const uint64_t *free_bytes,
                       uint32_t banks, const pn_qwen35_tt_fit_pins *pins,
                       pn_qwen35_tt_fit_levers *out, int64_t *low, bool *tried)
{
    pn_qwen35_tt_fit_shape s;

    for (size_t t = 0; t < sizeof(fit_tall_steps) / sizeof(fit_tall_steps[0]); t++) {
        s = *shape;
        if (fit_tall_steps[t].rows) {
            /* A pinned tall set, or one already at or under this step, takes no tall step. */
            if (pins->tall_rows || shape->tall_rows <= fit_tall_steps[t].rows)
                continue;
            s.tall_rows = fit_tall_steps[t].rows;
            if (!pins->outproj_mirrors)
                s.outproj_mirrors = fit_tall_steps[t].outproj_mirrors && shape->outproj_mirrors;
        }
        for (uint32_t v = 0; v < 2u; v++) {
            /* V only where there is a draft copy to drop and the operator left it free. */
            if (v == 1u && (pins->draft_view || shape->draft_view || shape->mtp_draft_bytes == 0u))
                break;
            s.draft_view = shape->draft_view || v == 1u;
            for (uint32_t r = shape->gdnc_replicas; r >= 1u; r--) {
                if (pins->gdnc_replicas && r != shape->gdnc_replicas)
                    break;
                s.gdnc_replicas = r;
                *low = pn_qwen35_tt_fit_policy_bank_low(&s, weight_bytes, joint_q_bytes,
                                                        checkpoint, free_bytes, banks);
                out->gdnc_replicas = s.gdnc_replicas;
                out->tall_rows = s.tall_rows;
                out->outproj_mirrors = s.outproj_mirrors;
                out->draft_view = s.draft_view;
                *tried = true;
                if (*low >= (int64_t)PN_QWEN35_TT_TIER_HEADROOM)
                    return true;
            }
        }
    }
    return false;
}

bool pn_qwen35_tt_fit_levers_pick(const pn_qwen35_tt_fit_shape *shape, uint64_t weight_bytes,
                                  uint64_t joint_q_bytes, bool checkpoint,
                                  const uint64_t *free_bytes, uint32_t banks,
                                  const pn_qwen35_tt_fit_pins *pins,
                                  pn_qwen35_tt_fit_levers *out, int64_t *low_out)
{
    const pn_qwen35_tt_fit_pins none = { false, false, false, false, 0u, 0u };
    int64_t low = INT64_MIN;
    bool tried = false, fits;

    if (!shape || !out)
        return false;
    if (!pins)
        pins = &none;
    out->gdnc_replicas = shape->gdnc_replicas;
    out->tall_rows = shape->tall_rows;
    out->outproj_mirrors = shape->outproj_mirrors;
    out->draft_view = shape->draft_view;
    out->kv_long = false;
    fits = fit_ladder(shape, weight_bytes, joint_q_bytes, checkpoint, free_bytes, banks, pins,
                      out, &low, &tried);
    /* K: the whole ladder again at the long type's tiles, only when the load's own type cannot
     * meet the rule and the long type is smaller (it is never a quality trade for nothing). */
    if (!fits && pins->kv_long_k_tile && pins->kv_long_v_tile &&
        (uint64_t)pins->kv_long_k_tile + pins->kv_long_v_tile <
            (uint64_t)shape->kv_k_tile_bytes + shape->kv_v_tile_bytes) {
        pn_qwen35_tt_fit_shape s = *shape;

        s.kv_k_tile_bytes = pins->kv_long_k_tile;
        s.kv_v_tile_bytes = pins->kv_long_v_tile;
        out->kv_long = true;
        fits = fit_ladder(&s, weight_bytes, joint_q_bytes, checkpoint, free_bytes, banks, pins,
                          out, &low, &tried);
    }
    if (low_out)
        *low_out = tried ? low : INT64_MIN;
    return fits;
}
