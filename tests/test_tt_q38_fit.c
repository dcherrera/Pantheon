/* SPDX-License-Identifier: AGPL-3.0-or-later */
/* Copyright (C) 2026 David (Corey) Herrera */
/**
 * @file test_tt_q38_fit.c
 * @brief Qwen3.8-27B Bfp8 on one p150 at ctx 32768: the zero-cost fit (options
 *        A, B, C) checked against the real census.
 *
 * Opens no device. It drives the production planning code with Qwen3.8's shapes:
 *   - A: `pn_tt_weight_upload_parts()` — the load-time whole-or-halves decision;
 *   - C: `pn_qwen35_tt_tall_requests()` — which strips get a tall prefill copy;
 *   - B: `pn_tt_strip_placement_order()` + `_simulate()` — mirrors first, then emptiest bank.
 *
 * The bank state before the tall set comes from the `tt-full-load` census
 * (the serve logs at 16384 and 32768, "DRAM arm: MTP head"): card free
 * and the fullest bank's free, each printed to 0.001 GiB and taken here at the LOW end. The
 * other seven banks are unknown, so every check runs two spreads consistent with the census:
 * EVEN (the rest shared equally) and SKEWED (seven banks at the fullest bank's free, one bank
 * holding the remainder). Every starting value of the strip cursor is tried.
 *
 * Build: `make USE_TT=1 tests/test_tt_q38_fit` (Mac or Linux, no card).
 *
 * @status None
 * @issues None
 * @todo None
 */

#include <math.h>
#include <stdbool.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#ifdef USE_TT

#include "../src/accel/tt/pn_tt_kv_banks.h"
#include "../src/accel/tt/pn_tt_kv_format.h"
#include "../src/accel/tt/pn_tt_strip_placement.h"
#include "../src/accel/tt/pn_tt_weight_split.h"
#include "../src/models/qwen35moe_tt_fit_plan.h"
#include "../src/models/qwen35moe_tt_internal.h"
#include "../src/local/catalog_manifest.h"
#include "../src/local/load_options.h"
#include "../src/local/local_file_io.h"
#include "../src/serve/load_env_scope.h"

/* ---- Qwen3.8-27B geometry (GGUF metadata, serve log; artifact list qwen38_ttconv) ---- */
#define Q38_LAYERS 64
#define Q38_ATTN_EVERY 4 /* layers 3, 7, ... 63 are full attention: 16 of 64 */
#define Q38_EMBD_TILES 160u   /* 5120 */
#define Q38_HEADS 24u
#define Q38_HEAD_TILES 8u     /* head_dim 256 */
#define Q38_KV_TILES 32u      /* 4 kv heads x 256 */
#define Q38_FFN_TILES 544u    /* 17408 */
#define Q38_INNER_TILES 192u  /* d_inner 6144 */
#define Q38_CONV_TILES 320u   /* conv_dim 10240 */
#define Q38_DT_TILES 2u       /* dt_rank 48 */
#define Q38_VOCAB_TILES 7760u /* 248320 */
#define Q38_BFP8_TILE 1088ull

/* ---- p150 and the census ---- */
#define P150_BANKS 8u
#define P150_BANK_BYTES 4278190080ull
#define GIB (1024.0 * 1024.0 * 1024.0)
#define CENSUS_ROUNDING_GIB 0.0005 /* the log prints GiB to 3 decimals */
#define TALL_BYTES_PER_TILE (64ull * 2048ull) /* 64 rows of bf16 32x32 tiles */
#define MATMUL_MEMBERS 16u /* the joint attn_q's shards: 2 per bank, so 1/8 per bank */

static int g_checks, g_failures;

static void check(bool ok, const char *what)
{
    g_checks++;
    if (!ok) {
        g_failures++;
        printf("FAIL %s\n", what);
    }
}

/* ================================================================
 * The artifact's matmul tensors (497, 25,970.2 MiB in the load log)
 * ================================================================ */

typedef struct {
    char name[64];
    uint32_t tile_rows, tile_cols;
} q38_tensor;

static uint32_t q38_tensor_list(q38_tensor *out)
{
    static const struct { const char *suffix; uint32_t rows, cols; } SSM[] = {
        {"attn_qkv", Q38_EMBD_TILES, Q38_CONV_TILES},
        {"attn_gate", Q38_EMBD_TILES, Q38_INNER_TILES},
        {"ssm_alpha", Q38_EMBD_TILES, Q38_DT_TILES},   {"ssm_beta", Q38_EMBD_TILES, Q38_DT_TILES},
        {"ssm_out", Q38_INNER_TILES, Q38_EMBD_TILES},
    }, ATTN[] = {
        {"attn_q", Q38_EMBD_TILES, 2u * Q38_HEADS * Q38_HEAD_TILES},
        {"attn_k", Q38_EMBD_TILES, Q38_KV_TILES},      {"attn_v", Q38_EMBD_TILES, Q38_KV_TILES},
        {"attn_output", Q38_HEADS * Q38_HEAD_TILES, Q38_EMBD_TILES},
    }, FFN[] = {
        {"ffn_gate", Q38_EMBD_TILES, Q38_FFN_TILES},   {"ffn_up", Q38_EMBD_TILES, Q38_FFN_TILES},
        {"ffn_down", Q38_FFN_TILES, Q38_EMBD_TILES},
    };
    uint32_t n = 0;

    for (int layer = 0; layer < Q38_LAYERS; layer++) {
        const bool attention = (layer + 1) % Q38_ATTN_EVERY == 0;
        const size_t mixers =
            attention ? sizeof(ATTN) / sizeof(ATTN[0]) : sizeof(SSM) / sizeof(SSM[0]);

        for (size_t i = 0; i < mixers; i++) {
            snprintf(out[n].name, sizeof(out[n].name), "blk.%d.%s.weight", layer,
                     attention ? ATTN[i].suffix : SSM[i].suffix);
            out[n].tile_rows = attention ? ATTN[i].rows : SSM[i].rows;
            out[n++].tile_cols = attention ? ATTN[i].cols : SSM[i].cols;
        }
        for (size_t i = 0; i < sizeof(FFN) / sizeof(FFN[0]); i++) {
            snprintf(out[n].name, sizeof(out[n].name), "blk.%d.%s.weight", layer, FFN[i].suffix);
            out[n].tile_rows = FFN[i].rows;
            out[n++].tile_cols = FFN[i].cols;
        }
    }
    snprintf(out[n].name, sizeof(out[n].name), "output.weight");
    out[n].tile_rows = Q38_EMBD_TILES;
    out[n++].tile_cols = Q38_VOCAB_TILES;
    return n;
}

/** The upload plan's totals: all bytes, and bytes of whole (joint) attn_q uploads. */
typedef struct {
    uint64_t bytes, joint_q_bytes;
    uint32_t joint_q, q_halves, gate_halves;
} q38_upload_totals;

static q38_upload_totals q38_upload_plan(const pn_tt_weight_split_rule *rule)
{
    static q38_tensor tensors[512];
    pn_tt_weight_upload_part parts[PN_TT_WEIGHT_SPLIT_MAX_PARTS];
    q38_upload_totals t = {0};
    const uint32_t n = q38_tensor_list(tensors);

    for (uint32_t i = 0; i < n; i++) {
        const uint64_t bytes =
            (uint64_t)tensors[i].tile_rows * tensors[i].tile_cols * Q38_BFP8_TILE;
        const bool is_q = strstr(tensors[i].name, ".attn_q.weight") != NULL;
        const uint32_t count = pn_tt_weight_upload_parts(rule, tensors[i].name,
                                                         tensors[i].tile_cols, bytes, parts);

        check(count > 0u, "every Qwen3.8 matmul tensor has an upload plan");
        for (uint32_t p = 0; p < count; p++) {
            t.bytes += parts[p].bytes;
            if (is_q && parts[p].part < 0) {
                t.joint_q++;
                t.joint_q_bytes += parts[p].bytes;
            }
            if (is_q && parts[p].part == 0)
                t.q_halves++;
            if (is_q && parts[p].part == 1)
                t.gate_halves++;
            if (is_q && parts[p].part >= 0) {
                check(parts[p].tile_cols == Q38_HEADS * Q38_HEAD_TILES, "a half is 192 tiles");
                check(parts[p].runs[1].first ==
                          2u * Q38_HEAD_TILES + (uint32_t)parts[p].part * Q38_HEAD_TILES,
                      "head 1 of a half starts at its field in the joint");
            }
        }
    }
    check(n == 497u, "497 matmul tensors, as the load log counts");
    return t;
}

static bool q38_split_rule(pn_tt_weight_split_rule *rule)
{
    return pn_tt_weight_split_rule_init(rule, ".attn_q.weight", Q38_HEADS, 2u * Q38_HEAD_TILES,
                                        Q38_HEAD_TILES) == 0;
}

/* ================================================================
 * The tall prefill set (option C) from the decode strips
 * ================================================================ */

static void q38_decode_strips(pn_qwen35_tt *g)
{
    pn_qwen35_tt_strips *s = &g->card;
    pn_qwen35_tt_attn_strips *a = &g->attn_card;
    struct { pn_tt_activation *strip; uint32_t tiles, replicas; } set[] = {
        {&s->hidden[0], Q38_EMBD_TILES, 1}, {&s->hidden[1], Q38_EMBD_TILES, 1},
        {&s->normed, Q38_EMBD_TILES, P150_BANKS}, {&s->project, Q38_EMBD_TILES, 1},
        {&s->ssm_qkv, Q38_CONV_TILES, 1}, {&s->ssm_z, Q38_INNER_TILES, 1},
        {&s->ssm_alpha, Q38_DT_TILES, 1}, {&s->ssm_beta, Q38_DT_TILES, 1},
        {&s->ssm_gated, Q38_INNER_TILES, 1}, {&s->attn_qg, 2u * Q38_HEADS * Q38_HEAD_TILES, 1},
        {&s->attn_k, Q38_KV_TILES, 1}, {&s->attn_v, Q38_KV_TILES, 1},
        {&s->attn_out, Q38_INNER_TILES, 1}, {&s->ffn_gate, Q38_FFN_TILES, 1},
        {&s->ffn_up, Q38_FFN_TILES, 1}, {&s->ffn_swiglu, Q38_FFN_TILES, P150_BANKS},
        {&s->logits, Q38_VOCAB_TILES, 1},
        {&a->q, Q38_INNER_TILES, 1}, {&a->q_rot, Q38_INNER_TILES, 1}, {&a->k_rot, Q38_KV_TILES, 1},
        {&a->z, Q38_INNER_TILES, 1}, {&a->gate, Q38_INNER_TILES, 1},
        {&a->gated, Q38_INNER_TILES, 1},
    };

    for (size_t i = 0; i < sizeof(set) / sizeof(set[0]); i++) {
        set[i].strip->tile_count = set[i].tiles;
        set[i].strip->replica_count = set[i].replicas;
    }
}

typedef struct {
    pn_tt_strip_request requests[64];
    size_t count;
    uint64_t tiles, mirror_tiles;
    bool attn_qg_tall, logits_tall;
} q38_tall_set;

static q38_tall_set q38_tall(bool attn_on_card)
{
    static pn_qwen35_tt g;
    q38_tall_set t = {0};

    memset(&g, 0, sizeof(g));
    q38_decode_strips(&g);
    g.attn_on_card = attn_on_card;
    t.count = pn_qwen35_tt_tall_requests(&g, P150_BANKS, true, true, t.requests, 64);
    for (size_t i = 0; i < t.count; i++) {
        t.tiles += (uint64_t)t.requests[i].tile_count * t.requests[i].replicas;
        t.mirror_tiles += (uint64_t)t.requests[i].tile_count * (t.requests[i].replicas - 1u);
    }
    /* The request list is in prefill_strips_of order: attn_qg is 10th, logits 17th. */
    t.attn_qg_tall = t.requests[9].tile_count != 0u;
    t.logits_tall = t.requests[16].tile_count != 0u;
    return t;
}

/* ================================================================
 * Placement against the census
 * ================================================================ */

typedef struct {
    int ctx;
    double free_gib, fullest_free_gib; /* "DRAM arm: MTP head" line */
} q38_census;

static const q38_census CENSUS_16K = {16384, 1.870, 0.220};
static const q38_census CENSUS_32K = {32768, 0.870, 0.095};

/** Per-bank free before the tall set, `freed_per_bank` added to each bank (option A). */
static void q38_bank_spread(const q38_census *c, bool skewed, uint64_t freed_per_bank,
                            uint64_t *free_bytes)
{
    const uint64_t total = (uint64_t)((c->free_gib - CENSUS_ROUNDING_GIB) * GIB);
    const uint64_t fullest = (uint64_t)((c->fullest_free_gib - CENSUS_ROUNDING_GIB) * GIB);

    for (uint32_t b = 0; b < P150_BANKS; b++)
        free_bytes[b] = skewed ? fullest : (total - fullest) / (P150_BANKS - 1u);
    free_bytes[P150_BANKS - 1u] = fullest;
    if (skewed)
        free_bytes[0] = total - (uint64_t)(P150_BANKS - 1u) * fullest;
    for (uint32_t b = 0; b < P150_BANKS; b++)
        free_bytes[b] += freed_per_bank;
}

/**
 * Place the tall set in production order for every cursor start and both spreads.
 * @return true when all fit; `*margin_gib` = least free left in any bank over all runs
 */
static bool q38_place(const q38_census *c, const q38_tall_set *t, uint64_t freed_per_bank,
                      double *margin_gib, double *card_left_gib)
{
    size_t order[64];
    bool all = true;

    *margin_gib = 1e9;
    if (pn_tt_strip_placement_order(t->requests, t->count, order) != t->count)
        return false;
    for (int skewed = 0; skewed < 2; skewed++) {
        for (uint32_t start = 0; start < P150_BANKS; start++) {
            uint64_t free_bytes[P150_BANKS];
            uint32_t cursor = start;
            size_t failed = 0;
            uint64_t left = 0;

            q38_bank_spread(c, skewed != 0, freed_per_bank, free_bytes);
            if (pn_tt_strip_placement_simulate(t->requests, order, t->count,
                                               PN_TT_STRIP_POLICY_MIRRORS_FIRST,
                                               TALL_BYTES_PER_TILE, free_bytes, P150_BANKS,
                                               &cursor, &failed) != 0) {
                all = false;
                continue;
            }
            for (uint32_t b = 0; b < P150_BANKS; b++) {
                left += free_bytes[b];
                if ((double)free_bytes[b] / GIB < *margin_gib)
                    *margin_gib = (double)free_bytes[b] / GIB;
            }
            *card_left_gib = (double)left / GIB;
        }
    }
    return all;
}

/** Test (2) for one context: mirrored strips ordered first, and the set fits every bank. */
static void q38_check_context(const q38_census *c, const q38_tall_set *t, uint64_t freed_per_bank,
                              const char *arm)
{
    size_t order[64];
    bool mirrors_first = pn_tt_strip_placement_order(t->requests, t->count, order) == t->count;
    bool seen_unmirrored = false;
    double margin = 0.0, card_left = 0.0;
    char what[160];
    bool fits;

    for (size_t i = 0; mirrors_first && i < t->count; i++) {
        const pn_tt_strip_request *r = &t->requests[order[i]];

        if (r->tile_count == 0u)
            continue;
        if (r->replicas <= 1u)
            seen_unmirrored = true;
        else if (seen_unmirrored)
            mirrors_first = false;
    }
    snprintf(what, sizeof(what),
             "%s ctx %d: every mirrored tall strip placed before any unmirrored one", arm, c->ctx);
    check(mirrors_first, what);
    fits = q38_place(c, t, freed_per_bank, &margin, &card_left);
    snprintf(what, sizeof(what), "%s ctx %d: the tall set fits, no bank over 4,278,190,080 B", arm,
             c->ctx);
    check(fits, what);
    if (fits)
        printf("  %s ctx %5d: fits; card free after the tall set %.3f GiB, "
               "fullest bank %.3f GiB free\n", arm, c->ctx, card_left, margin);
    else
        printf("  %s ctx %5d: DOES NOT FIT\n", arm, c->ctx);
}

/* ================================================================
 * THE CONDITIONAL POLICY: main's placement unless main cannot fit
 * ================================================================ */

/** A model the fit policy is decided for: its shape, artifact bytes and the census at the
 *  "DRAM arm: MTP head" stage (the state the tall set is placed into). */
typedef struct {
    const char *name;
    pn_qwen35_tt_fit_shape shape;
    uint64_t weight_bytes, joint_q_bytes;
    q38_census census;
    bool policy_expected;
} fit_model;

/* Free DRAM when the artifact is open: 8 x 4,278,190,080 B less the 0.016 GiB of scratch
 * rows the census reports as non-weight right after the upload. */
#define CARD_FREE_AT_OPEN ((uint64_t)P150_BANKS * P150_BANK_BYTES - (uint64_t)(0.016 * GIB))

static pn_qwen35_tt_fit_shape fit_shape_of(uint32_t embd, uint32_t ffn, uint32_t heads,
                                           uint32_t kv_heads, uint32_t n_attn, uint32_t n_ssm,
                                           uint32_t d_inner, uint32_t v_heads, uint32_t conv_dim,
                                           uint32_t ctx)
{
    pn_qwen35_tt_fit_shape s = {0};

    s.n_embd = embd; s.ffn = ffn; s.n_head = heads; s.n_head_kv = kv_heads; s.head_dim = 256u;
    s.n_attn = n_attn; s.n_ssm = n_ssm; s.d_inner = d_inner; s.v_heads = v_heads;
    s.conv_dim = conv_dim; s.conv_taps = 4u; s.ssm_state = 128u; s.ctx = ctx;
    s.tall_rows = 64u; s.mirrors = P150_BANKS; s.outproj_mirrors = true;
    s.gdnc_replicas = 8u; s.attn_on_card = true; s.gdn_on_card = true; s.state_split = true;
    return s;
}

/** Bytes of `layers` joint attn_q tensors: (embd/32) x (2*heads*256/32) tiles of `tile`. */
static uint64_t joint_q_bytes(uint32_t layers, uint32_t embd, uint32_t heads, uint64_t tile)
{
    return (uint64_t)layers * (embd / 32u) * (2u * heads * 8u) * tile;
}

/** Fill the decode strips the tall set is built from, from a shape (qwen35_tt_alloc_strips). */
static void shape_decode_strips(pn_qwen35_tt *g, const pn_qwen35_tt_fit_shape *s)
{
    const uint32_t embd = s->n_embd / 32u, inner = s->d_inner / 32u, ffn = s->ffn / 32u;
    const uint32_t heads = s->n_head * s->head_dim / 32u, kv = s->n_head_kv * s->head_dim / 32u;
    const uint32_t dt = (s->v_heads + 31u) / 32u;
    pn_qwen35_tt_strips *c = &g->card;
    pn_qwen35_tt_attn_strips *a = &g->attn_card;
    struct { pn_tt_activation *strip; uint32_t tiles, replicas; } set[] = {
        {&c->hidden[0], embd, 1}, {&c->hidden[1], embd, 1}, {&c->normed, embd, s->mirrors},
        {&c->project, embd, 1}, {&c->ssm_qkv, s->conv_dim / 32u, 1}, {&c->ssm_z, inner, 1},
        {&c->ssm_alpha, dt, 1}, {&c->ssm_beta, dt, 1}, {&c->ssm_gated, inner, 1},
        {&c->attn_qg, 2u * heads, 1}, {&c->attn_k, kv, 1}, {&c->attn_v, kv, 1},
        {&c->attn_out, heads, 1}, {&c->ffn_gate, ffn, 1}, {&c->ffn_up, ffn, 1},
        {&c->ffn_swiglu, ffn, s->mirrors}, {&c->logits, Q38_VOCAB_TILES, 1},
        {&a->q, heads, 1}, {&a->q_rot, heads, 1}, {&a->k_rot, kv, 1}, {&a->z, heads, 1},
        {&a->gate, heads, 1}, {&a->gated, heads, 1},
    };

    memset(g, 0, sizeof(*g));
    for (size_t i = 0; i < sizeof(set) / sizeof(set[0]); i++) {
        set[i].strip->tile_count = set[i].tiles;
        set[i].strip->replica_count = set[i].replicas;
    }
    g->attn_on_card = s->attn_on_card;
}

/** Main's tall set, as main's pn_qwen35_tt_prefill_tall_alloc loop built it: every strip in
 *  prefill_strips_of order but the logit row, the gated pair at the device's mirror count. */
static size_t main_tall_requests(pn_qwen35_tt *g, pn_tt_strip_request *out)
{
    pn_tt_activation *all[] = {
        &g->card.hidden[0], &g->card.hidden[1], &g->card.normed, &g->card.project,
        &g->card.ssm_qkv, &g->card.ssm_z, &g->card.ssm_alpha, &g->card.ssm_beta,
        &g->card.ssm_gated, &g->card.attn_qg, &g->card.attn_k, &g->card.attn_v,
        &g->card.attn_out, &g->card.ffn_gate, &g->card.ffn_up, &g->card.ffn_swiglu,
        &g->card.logits, &g->attn_card.q, &g->attn_card.q_rot, &g->attn_card.k_rot,
        &g->attn_card.z, &g->attn_card.gate, &g->attn_card.gated,
    };
    const size_t n = sizeof(all) / sizeof(all[0]);

    for (size_t i = 0; i < n; i++) {
        const bool gated = all[i] == &g->card.ssm_gated || all[i] == &g->attn_card.gated;

        out[i].tile_count = all[i] == &g->card.logits ? 0u : all[i]->tile_count;
        out[i].replicas = gated ? P150_BANKS : all[i]->replica_count;
    }
    return n;
}

/** Main's order (input order, every copy 0 round-robin) for every cursor start and spread:
 *  true when it fits all of them, `*any_fail` when at least one fails. */
static bool main_order_fits(const q38_census *c, const pn_tt_strip_request *req, size_t n,
                            bool *any_fail)
{
    size_t order[64];
    bool all = true;

    for (size_t i = 0; i < n; i++)
        order[i] = i;
    *any_fail = false;
    for (int skewed = 0; skewed < 2; skewed++)
        for (uint32_t start = 0; start < P150_BANKS; start++) {
            uint64_t free_bytes[P150_BANKS];
            uint32_t cursor = start;

            q38_bank_spread(c, skewed != 0, 0u, free_bytes);
            if (pn_tt_strip_placement_simulate(req, order, n, PN_TT_STRIP_POLICY_ROUND_ROBIN,
                                               TALL_BYTES_PER_TILE, free_bytes, P150_BANKS,
                                               &cursor, NULL) != 0) {
                all = false;
                *any_fail = true;
            }
        }
    return all;
}

/** One model: the floor's verdict, and with main's placement, that the planner's tall set is
 *  main's request for request and that main's order is what the tall stage keeps. */
static void check_fit_model(const fit_model *fm)
{
    static pn_qwen35_tt g;
    pn_tt_strip_request mine[64], ref[64];
    pn_qwen35_tt_fit_floor floor;
    size_t n_mine, n_ref;
    uint64_t tiles = 0u;
    bool wanted, same = true, any_fail;
    char what[200];

    pn_qwen35_tt_fit_floor_compute(&fm->shape, fm->weight_bytes, fm->joint_q_bytes, &floor);
    wanted = pn_qwen35_tt_fit_policy_wanted(&floor, CARD_FREE_AT_OPEN);
    printf("  %-28s floor %.3f GiB, slack %+.3f GiB -> %s\n", fm->name, floor.total / GIB,
           ((double)CARD_FREE_AT_OPEN - (double)floor.total) / GIB,
           wanted ? "FIT POLICY" : "main's placement");
    snprintf(what, sizeof(what), "%s: fit policy %s", fm->name,
             fm->policy_expected ? "engages" : "stays off (main's placement)");
    check(wanted == fm->policy_expected, what);

    shape_decode_strips(&g, &fm->shape);
    n_mine = pn_qwen35_tt_tall_requests(&g, P150_BANKS, true, false, mine, 64);
    n_ref = main_tall_requests(&g, ref);
    for (size_t i = 0; i < n_ref && i < n_mine; i++) {
        same = same && mine[i].tile_count == ref[i].tile_count &&
               mine[i].replicas == ref[i].replicas;
        tiles += (uint64_t)mine[i].tile_count * mine[i].replicas;
    }
    snprintf(what, sizeof(what), "%s: without the policy the tall set is main's, strip by strip",
             fm->name);
    check(same && n_mine == n_ref, what);
    snprintf(what, sizeof(what), "%s: the floor's tall set is the planner's (%llu tiles)",
             fm->name, (unsigned long long)tiles);
    check(pn_qwen35_tt_fit_tall_tiles(&fm->shape) == tiles, what);
    if (fm->policy_expected || fm->shape.tall_rows == 0u)
        return;
    snprintf(what, sizeof(what), "%s: main's tall order fits every spread, so it is kept",
             fm->name);
    check(main_order_fits(&fm->census, ref, n_ref, &any_fail), what);
}

static void check_conditional_policy(uint64_t q38_weights, uint64_t q38_joint_q)
{
    /* Bfp4 tile 576 B vs Bfp8 1088 B: a Bonsai-27B Bfp4 artifact is Qwen3.8's tiles at 576. */
    const uint64_t bonsai4 = q38_weights / Q38_BFP8_TILE * 576u;
    const fit_model models[] = {
        {"Ornith-1.5-9B 32k", fit_shape_of(4096, 12288, 16, 4, 8, 24, 4096, 32, 8192, 32768),
         (uint64_t)(8.093 * GIB), joint_q_bytes(8, 4096, 16, Q38_BFP8_TILE),
         {32768, 21.377, 2.658}, false},
        {"Qwen3.6-35B-A3B 36,864", fit_shape_of(2048, 512, 16, 2, 10, 30, 4096, 32, 8192, 36864),
         (uint64_t)(26.316 * GIB), joint_q_bytes(10, 2048, 16, Q38_BFP8_TILE),
         {36864, 3.205, 0.368}, false},
        {"Bonsai-27B Bfp4 32k", fit_shape_of(5120, 17408, 24, 4, 16, 48, 6144, 48, 10240, 32768),
         bonsai4, q38_joint_q / Q38_BFP8_TILE * 576u,
         {32768, 1.867 + (double)(q38_weights - bonsai4) / GIB,
          0.219 + (double)(q38_weights - bonsai4) / GIB / P150_BANKS}, false},
        {"Qwen3.8-27B rows=0 32k", fit_shape_of(5120, 17408, 24, 4, 16, 48, 6144, 48, 10240,
                                                32768),
         q38_weights, q38_joint_q, {32768, 0.870, 0.095}, false},
        {"Qwen3.8-27B 16k", fit_shape_of(5120, 17408, 24, 4, 16, 48, 6144, 48, 10240, 16384),
         q38_weights, q38_joint_q, CENSUS_16K, true},
        {"Qwen3.8-27B 32k", fit_shape_of(5120, 17408, 24, 4, 16, 48, 6144, 48, 10240, 32768),
         q38_weights, q38_joint_q, CENSUS_32K, true},
        {"Qwen3.8-27B 34,816", fit_shape_of(5120, 17408, 24, 4, 16, 48, 6144, 48, 10240, 34816),
         q38_weights, q38_joint_q, CENSUS_32K, true},
        {"Qwen3.8-27B 36,864", fit_shape_of(5120, 17408, 24, 4, 16, 48, 6144, 48, 10240, 36864),
         q38_weights, q38_joint_q, CENSUS_32K, true},
    };
    fit_model rows0 = models[3];
    bool any_fail;
    static pn_qwen35_tt g;
    pn_tt_strip_request ref[64];
    size_t n;

    rows0.shape.tall_rows = 0u;
    for (size_t i = 0; i < sizeof(models) / sizeof(models[0]); i++)
        check_fit_model(i == 3 ? &rows0 : &models[i]);

    /* A FLOOR: at most what main allocated. The p150 "DRAM after arm" used on main (Ornith 32k
     * 11.580, Qwen3.6 36,864 29.218, Qwen3.8 rows=0 32k 31.005; raw serve logs),
     * against the floor less its post-load reserve, which main had not taken by then. */
    {
        const struct { size_t model; double used_gib; } measured[] = {
            {0, 11.580}, {1, 29.218}, {3, 31.005},
        };

        for (size_t i = 0; i < sizeof(measured) / sizeof(measured[0]); i++) {
            const fit_model *fm = measured[i].model == 3 ? &rows0 : &models[measured[i].model];
            pn_qwen35_tt_fit_floor floor;
            char what[160];

            pn_qwen35_tt_fit_floor_compute(&fm->shape, fm->weight_bytes, fm->joint_q_bytes,
                                           &floor);
            snprintf(what, sizeof(what), "%s: floor %.3f GiB <= main's measured %.3f GiB",
                     fm->name, (floor.total - floor.reserve) / GIB, measured[i].used_gib);
            check((floor.total - floor.reserve) / GIB <= measured[i].used_gib, what);
        }
    }

    /* Main's order at Qwen3.8 16k is the p150 failure (main refused at 16k): the tall stage's
     * dry-run of main's order must see it, or it would keep main's order and refuse. */
    shape_decode_strips(&g, &models[4].shape);
    n = main_tall_requests(&g, ref);
    (void)main_order_fits(&CENSUS_16K, ref, n, &any_fail);
    check(any_fail, "Qwen3.8 16k: the dry-run of main's tall order fails, as main did on the p150");
}

/* ================================================================
 * (6) MTP on: a dense head enters the floor
 * ================================================================ */

/* Gate 5 on the p150 (Pantheon 852730e4): with the MTP head on, main's placement at 8,192 refused
 * at the tall set ("arm: MTP head" 1.553 GiB free, the head 0.786 GiB) — the floor had not
 * counted the head, so the fit policy stayed off. */
#define Q38_MTP_ARTIFACT 451323072ull /* Qwen3.8-27B-Q8_0.bfp8.mtp.fttw, bytes on disk */
#define Q38_MTP_DRAFT_COLS 2057ull    /* draft vocabulary: 2,048 tiles + the 9-tile special tail */
#define Q38_MTP_ARM_GIB 0.786         /* "arm: MTP head" used minus "arm: attention + KV" */

static void check_mtp_floor(uint64_t q38_weights, uint64_t q38_joint_q)
{
    const uint64_t head = Q38_MTP_ARTIFACT + Q38_MTP_DRAFT_COLS * Q38_EMBD_TILES * Q38_BFP8_TILE;
    /* Hand-worked: one KV layer = 4 kv heads x 256 x K+V x 2 B = 4,096 B a position; a verify
     * slot = 48 GDN layers x (state 48 heads x 4 x 4 fp32 tiles = 3 MiB + window 320 tiles x 3
     * taps x 4 KiB = 3.75 MiB) = 339,738,624 B. */
    const uint64_t slot = 48ull * (3145728ull + 3932160ull);
    pn_qwen35_tt_fit_shape off = fit_shape_of(5120, 17408, 24, 4, 16, 48, 6144, 48, 10240, 8192);
    pn_qwen35_tt_fit_shape on = off, orn;
    pn_qwen35_tt_fit_floor f_off, f_on, f_orn;
    char what[160];

    on.mtp_bytes = head;
    on.mtp_rows = 4u;
    pn_qwen35_tt_fit_floor_compute(&off, q38_weights, q38_joint_q, &f_off);
    pn_qwen35_tt_fit_floor_compute(&on, q38_weights, q38_joint_q, &f_on);
    printf("  Qwen3.8 8,192: floor %.3f GiB without MTP, %.3f GiB with it (MTP %.3f)\n",
           f_off.total / GIB, f_on.total / GIB, f_on.mtp / GIB);
    check(slot == 339738624ull, "a verify slot is 324 MiB");
    snprintf(what, sizeof(what), "MTP term = head + 8,192 x 4 KiB + 4 slots (%llu B)",
             (unsigned long long)f_on.mtp);
    check(f_on.mtp == head + 8192ull * 4096ull + 4ull * slot && f_off.mtp == 0u, what);
    check(f_on.total - f_off.total == f_on.mtp, "MTP is the only term that moves");
    check(!pn_qwen35_tt_fit_policy_wanted(&f_off, CARD_FREE_AT_OPEN),
          "Qwen3.8 8,192 without MTP: main's placement");
    check(pn_qwen35_tt_fit_policy_wanted(&f_on, CARD_FREE_AT_OPEN),
          "Qwen3.8 8,192 with MTP: the fit policy engages");
    check((double)head / GIB <= Q38_MTP_ARM_GIB, "the head term is at most the measured arm");
    /* Ornith-1.5-9B 32k with its MTP head (0.3 GiB is above its artifact): main's placement. */
    orn = fit_shape_of(4096, 12288, 16, 4, 8, 24, 4096, 32, 8192, 32768);
    orn.mtp_bytes = (uint64_t)(0.3 * GIB) + Q38_MTP_DRAFT_COLS * 128u * Q38_BFP8_TILE;
    orn.mtp_rows = 4u;
    pn_qwen35_tt_fit_floor_compute(&orn, (uint64_t)(8.093 * GIB),
                                   joint_q_bytes(8, 4096, 16, Q38_BFP8_TILE), &f_orn);
    check(!pn_qwen35_tt_fit_policy_wanted(&f_orn, CARD_FREE_AT_OPEN),
          "Ornith 32k with MTP: main's placement");
}

/* Block-float KV (pn_tt_kv_format.h): a position of one attention layer is 4,096 B bf16,
 * 2,176 B Bfp8_b (1,088 B tiles), 1,152 B Bfp4_b (576 B tiles), and K and V scale each by its
 * own tile in a mixed cache (K8V4 2,176 / 2 + 1,152 / 2 = 1,664 B); trunk and the MTP head's
 * 17th layer both scale. Hand-worked at 30,720: 16 x 4,096 x 30,720 = 1.875 GiB bf16. */
typedef struct kv_type_case {
    const char *name;          /* the PN_TT_KV_TYPE value */
    uint32_t k_tile, v_tile;   /* 0 = bf16 */
    uint64_t per_position;     /* K + V bytes of one position of one layer, by hand */
    const char *program;       /* the attention program that reads it */
} kv_type_case;

static const kv_type_case KV_TYPES[] = {
    { "bf16", 0u, 0u, 4096u, "attention" },
    { "bfp8", 1088u, 1088u, 2176u, "attention_bfp8" },
    { "bfp4", 576u, 576u, 1152u, "attention_bfp4" },
    { "k8v4", 1088u, 576u, 1664u, "attention_k8v4" },
    { "k16v4", 2048u, 576u, 2624u, "attention_k16v4" },
    { "k4v16", 576u, 2048u, 2624u, "attention_k4v16" },
    { "k16v8", 2048u, 1088u, 3136u, "attention_k16v8" },
    /* K4H: the Bfp4 cache, K Hadamard-rotated (pn_tt_kv_rotate.h); Bfp4's program. */
    { "k4h", 576u, 576u, 1152u, "attention_bfp4" },
};
#define KV_TYPE_COUNT (sizeof(KV_TYPES) / sizeof(KV_TYPES[0]))
#define MIB (1024.0 * 1024.0)
/* Measured (#56 + #106, p150, 2026-11-12, probes at 30720, "combo"): Bfp4 KV at 30,720 +
 * prefix checkpoint + MTP k1 leaves the fullest bank 0.038 GiB free. */
#define BFP4_K1_CKPT_FREE_MIB 38.0

/** One layer's K (or V) group at 30,720: 1 KV head (G = 4) x 960 tiles x 8 dim tiles. */
static uint64_t kv_group_bytes(uint32_t tile)
{
    return 960ull * 8u * (tile ? tile : 2048u);
}

/**
 * The 17 layers' K and V groups over the 8 banks as pn_tt_kv_banks_alloc() places them (G = 4,
 * the next 4 banks for one set, the 4 after for the other, banks of a set padded level),
 * starting from `start` (per-bank cursors). `balanced` false = K then V on every layer, the
 * placement a mixed cache would get without pn_tt_kv_layer_v_first(). Returns the KV bytes of
 * the bank that holds the most.
 */
static uint64_t kv_bank_share(uint32_t k_tile, uint32_t v_tile, const uint64_t start[P150_BANKS],
                              bool balanced)
{
    uint64_t cursor[P150_BANKS], kv[P150_BANKS] = { 0 }, most = 0;

    memcpy(cursor, start, sizeof(cursor));
    for (uint32_t layer = 0; layer < 17u; layer++) {
        uint64_t top[2] = { 0, 0 };

        for (uint32_t b = 0; b < P150_BANKS; b++)
            if (cursor[b] > top[b / 4u])
                top[b / 4u] = cursor[b];
        const bool v_first = balanced && pn_tt_kv_layer_v_first(kv_group_bytes(k_tile),
                                                                kv_group_bytes(v_tile), top[0],
                                                                top[1]);
        for (uint32_t b = 0; b < P150_BANKS; b++) {
            const bool first_set = b < 4u;
            const uint64_t bytes = kv_group_bytes(first_set != v_first ? k_tile : v_tile);

            cursor[b] = top[b / 4u] + bytes;
            kv[b] += bytes;
        }
    }
    for (uint32_t b = 0; b < P150_BANKS; b++)
        if (kv[b] > most)
            most = kv[b];
    return most;
}

/**
 * The fit of each type at 30,720 + prefix checkpoint + MTP k1 (spec §3): the fullest bank's
 * KV against Bfp4's, from the measured Bfp4 free space. `kv17[t]` is the fit floor's KV of
 * type t over the 17 layers (16 trunk + the MTP head's). Average model (KV spread evenly) and
 * the allocator's worst case (the fullest bank in the set that takes 9 of the 17 larger
 * regions). K8V4 is the one mixed type that fits; it fits only because a mixed cache's
 * allocation alternates its larger region between the bank sets.
 */
static void check_kv_type_fit(const uint64_t kv17[KV_TYPE_COUNT])
{
    static const uint64_t level[P150_BANKS] = { 0 };
    const double bfp4_avg = (double)kv17[2] / P150_BANKS / MIB; /* KV_TYPES[2] = bfp4 */
    const double bfp4_worst = (double)kv_bank_share(576u, 576u, level, true) / MIB;
    char what[200];

    for (size_t t = 0; t < KV_TYPE_COUNT; t++) {
        const kv_type_case *c = &KV_TYPES[t];
        const double avg = (double)kv17[t] / P150_BANKS / MIB;
        const double worst = (double)kv_bank_share(c->k_tile, c->v_tile, level, true) / MIB;
        const double spare_avg = BFP4_K1_CKPT_FREE_MIB - (avg - bfp4_avg);
        const double spare_worst = BFP4_K1_CKPT_FREE_MIB - (worst - bfp4_worst);

        printf("  %-5s 30,720 + ckpt + k1: KV/bank avg %.1f MiB, worst bank %.1f MiB -> spare "
               "%.1f (avg) / %.1f (worst) MiB\n", c->name, avg, worst, spare_avg, spare_worst);
        if (strcmp(c->name, "k8v4") == 0) {
            snprintf(what, sizeof(what), "K8V4 fits 30,720 + ckpt + k1: ~6 MiB spare on average "
                     "(%.1f), > 0 at the worst bank (%.1f)", spare_avg, spare_worst);
            check(spare_avg > 5.0 && spare_avg < 7.0 && spare_worst > 0.0, what);
            check(fabs(avg - 103.6) < 0.1, "K8V4 KV per bank at 30,720 is 103.6 MiB (spec §3)");
            snprintf(what, sizeof(what), "K8V4 placed K-then-V on every layer would NOT fit "
                     "(worst bank %.1f MiB short): the allocator must alternate",
                     -(BFP4_K1_CKPT_FREE_MIB -
                       ((double)kv_bank_share(c->k_tile, c->v_tile, level, false) / MIB -
                        bfp4_worst)));
            check(BFP4_K1_CKPT_FREE_MIB -
                          ((double)kv_bank_share(c->k_tile, c->v_tile, level, false) / MIB -
                           bfp4_worst) < 0.0,
                  what);
        } else if (strcmp(c->name, "k4h") == 0) {
            /* Bfp4's bytes exactly: H rides the matmul residency (68 KiB, not KV). */
            check(fabs(avg - bfp4_avg) < 1e-9 && fabs(worst - bfp4_worst) < 1e-9,
                  "k4h KV per bank = Bfp4's at 30,720");
        } else if (strcmp(c->name, "bfp4") != 0) {
            snprintf(what, sizeof(what), "%s does not fit 30,720 + ckpt + k1", c->name);
            check(spare_avg < 0.0, what);
        }
    }
    check(!pn_tt_kv_layer_v_first(kv_group_bytes(576u), kv_group_bytes(576u), 9u, 1u),
          "a uniform cache always places K then V (every earlier placement unchanged)");
}

/** The knob: each PN_TT_KV_TYPE value names its K and V tiles and its program; others refuse. */
static void check_kv_type_knob(void)
{
    static const char *const refused[] = { "k8v8", "k8v16", "bfp6", "BF16", "k4v4" };
    pn_tt_kv_type type;
    char what[200];

    for (size_t t = 0; t < KV_TYPE_COUNT; t++) {
        const kv_type_case *c = &KV_TYPES[t];
        const bool parsed = pn_tt_kv_type_parse(c->name, &type);
        const char *program = pn_tt_kv_type_attention_program(type);
        const uint32_t k_tile = c->k_tile ? c->k_tile : 2048u;
        const uint32_t v_tile = c->v_tile ? c->v_tile : 2048u;

        snprintf(what, sizeof(what), "PN_TT_KV_TYPE=%s: K %u / V %u B tiles, program %s",
                 c->name, k_tile, v_tile, c->program);
        check(parsed && pn_tt_kv_format_tile_bytes(type.k_format) == k_tile &&
                  pn_tt_kv_format_tile_bytes(type.v_format) == v_tile &&
                  strcmp(pn_tt_kv_type_name(type), c->name) == 0 && program &&
                  strcmp(program, c->program) == 0,
              what);
    }
    for (size_t r = 0; r < sizeof(refused) / sizeof(refused[0]); r++) {
        snprintf(what, sizeof(what), "PN_TT_KV_TYPE=%s is refused (stays bf16)", refused[r]);
        check(!pn_tt_kv_type_parse(refused[r], &type) && type.k_format == PN_TT_KV_FORMAT_BF16 &&
                  type.v_format == PN_TT_KV_FORMAT_BF16,
              what);
    }
    check(pn_tt_kv_type_parse("", &type) && type.k_format == PN_TT_KV_FORMAT_BF16,
          "an empty PN_TT_KV_TYPE is bf16");
}

/**
 * The shipped catalog's Qwen3.8 p150 profiles run the shapes checked here: Pantheon Local sets
 * the profile's PN_TT_KV_TYPE / PN_TT_SPEC for its load (load_options.h, model_admin.c), and
 * the TT load reads them back as the KV type and the verify rows the fit prices. p150-one must
 * be K8V4 at 30,720 + ckpt + k1 (2 verify rows): the one mixed type check_kv_type_fit() fits
 * there. p150-one-bf16 loads with no MTP head (PN_TT_SPEC=0). The catalog's p150 KV names are
 * exactly the knob's.
 */
static void check_catalog_profiles(void)
{
    pn_catalog_manifest *m = calloc(1, sizeof(*m));
    char *text = NULL, err[256], spec[8];
    pn_tt_kv_type type;

    for (size_t t = 0; t < KV_TYPE_COUNT; t++)
        check(pn_cat_kv_type_allowed("p150", KV_TYPES[t].name) &&
                  pn_tt_kv_type_parse(KV_TYPES[t].name, &type),
              "every PN_TT_KV_TYPE value is a p150 catalog kv_type");
    check(!pn_cat_kv_type_allowed("p150", "k8v8") && !pn_tt_kv_type_parse("k8v8", &type),
          "a name the knob refuses, the catalog refuses");
    if (!m || pn_local_read_file("catalog/catalog.json", 4u << 20, &text, NULL) != 0 ||
        pn_cat_manifest_parse(text, m, err, sizeof(err)) != 0) {
        check(false, "catalog/catalog.json parses");
        free(text);
        free(m);
        return;
    }
    const pn_cat_model *q = pn_cat_find_model(m, "qwen3.8-27b-q8");
    const pn_cat_profile *k8v4 = q ? pn_cat_find_profile(q, "p150-one") : NULL;
    const pn_cat_profile *bf16 = q ? pn_cat_find_profile(q, "p150-one-bf16") : NULL;

    check(k8v4 && bf16, "Qwen3.8 p150-one and p150-one-bf16 are in the catalog");
    if (k8v4 && bf16) {
        pn_load_env_var env[] = {
            { PN_TT_KV_TYPE_ENV, pn_local_tt_kv_type_of("p150", k8v4->options.kv_type), NULL },
            { "PN_TT_SPEC", pn_local_tt_spec_of("p150", k8v4->mtp, k8v4->mtp_drafts, spec,
                                                sizeof(spec)), NULL },
        };
        pn_load_env_apply(env, 2);
        type = pn_tt_kv_type_requested();
        const uint32_t rows = pn_qwen35_tt_mtp_verify_rows();
        pn_load_env_restore(env, 2);
        printf("  catalog Qwen3.8 p150-one: KV %s, %u verify rows, ctx %d\n",
               pn_tt_kv_type_name(type), rows, k8v4->ctx_cap);
        check(strcmp(pn_tt_kv_type_name(type), "k8v4") == 0 && rows == 2u &&
                  k8v4->ctx_cap == 30720 && k8v4->options.ctx_max == 30720,
              "catalog Qwen3.8 p150-one loads as K8V4 + MTP k1 at 30,720 (the fitting shape)");
        pn_load_env_var off[] = {
            { PN_TT_KV_TYPE_ENV, pn_local_tt_kv_type_of("p150", bf16->options.kv_type), NULL },
            { "PN_TT_SPEC", pn_local_tt_spec_of("p150", bf16->mtp, bf16->mtp_drafts, spec,
                                                sizeof(spec)), NULL },
        };
        check(off[0].value == NULL && off[1].value && strcmp(off[1].value, "0") == 0 &&
                  bf16->ctx_cap == 30720,
              "catalog Qwen3.8 p150-one-bf16 loads bf16 with no MTP head (PN_TT_SPEC=0)");
    }
    free(text);
    free(m);
}

static void check_kv_format_floor(uint64_t q38_weights, uint64_t q38_joint_q)
{
    pn_qwen35_tt_fit_shape base = fit_shape_of(5120, 17408, 24, 4, 16, 48, 6144, 48, 10240,
                                               30720);
    pn_qwen35_tt_fit_floor bf16_floor;
    uint64_t kv17[KV_TYPE_COUNT];
    char what[200];

    base.mtp_bytes = 1u; /* any head: its KV layer is what is checked */
    base.mtp_rows = 2u;
    pn_qwen35_tt_fit_floor_compute(&base, q38_weights, q38_joint_q, &bf16_floor);
    for (size_t t = 0; t < KV_TYPE_COUNT; t++) {
        const kv_type_case *c = &KV_TYPES[t];
        pn_qwen35_tt_fit_shape s = base;
        pn_qwen35_tt_fit_floor floor;

        s.kv_k_tile_bytes = c->k_tile;
        s.kv_v_tile_bytes = c->v_tile;
        pn_qwen35_tt_fit_floor_compute(&s, q38_weights, q38_joint_q, &floor);
        snprintf(what, sizeof(what), "KV %s: trunk KV = 16 x %llu x 30,720 (%.3f GiB)", c->name,
                 (unsigned long long)c->per_position, floor.kv / GIB);
        check(floor.kv == 16ull * c->per_position * 30720ull, what);
        check(floor.mtp - bf16_floor.mtp + KV_TYPES[0].per_position * 30720ull ==
                  c->per_position * 30720ull,
              "the MTP head's KV layer scales the same way");
        kv17[t] = floor.kv + floor.kv / 16u; /* + the head's layer, checked just above */
        printf("  KV %-5s (K %4u / V %4u B tiles): trunk KV %.3f GiB, frees %.3f GiB vs bf16 "
               "(with the head's layer)\n", c->name, c->k_tile ? c->k_tile : 2048u,
               c->v_tile ? c->v_tile : 2048u, floor.kv / GIB,
               (double)(bf16_floor.total - floor.total) / GIB);
    }
    check_kv_type_fit(kv17);
    check_kv_type_knob();
    check_catalog_profiles();
}

int main(void)
{
    pn_tt_weight_split_rule rule;
    q38_upload_totals before, after;
    q38_tall_set tall_old, tall;
    uint64_t freed, freed_per_bank;
    double tall_gib, tall_old_gib, mirror_gib, margin_32k;

    /* (1) A: the joint attn_q is not uploaded; both halves of every one are. */
    check(q38_split_rule(&rule), "the attn_q split rule builds at 24 heads x 16 tiles");
    before = q38_upload_plan(NULL);
    after = q38_upload_plan(&rule);
    check(before.joint_q == 16u && before.q_halves == 0u,
          "without the rule: 16 joint attn_q, no halves");
    check(after.joint_q == 0u, "A: no joint attn_q is planned for upload");
    check(after.q_halves == 16u && after.gate_halves == 16u,
          "A: the Q and gate halves of all 16 are");
    check(after.bytes == before.bytes, "A: the halves replace the joint byte for byte");
    freed = before.joint_q_bytes - after.joint_q_bytes;
    freed_per_bank = freed / P150_BANKS;
    check((MATMUL_MEMBERS % P150_BANKS) == 0u &&
          (2u * Q38_HEADS * Q38_HEAD_TILES) % MATMUL_MEMBERS == 0u,
          "the joint's 16 even shards sit 2 to a bank, so A frees 1/8 of it in every bank");

    /* The host block and the seam re-interleave the halves: head h's Q then its gate. */
    {
        float q[2 * 4], gate[2 * 4], qg[4 * 4];

        for (int i = 0; i < 8; i++) {
            q[i] = (float)i;
            gate[i] = (float)(100 + i);
        }
        pn_qwen35_tt_interleave_qg(q, gate, 2u, 4u, qg);
        check(qg[0] == 0.0f && qg[3] == 3.0f && qg[4] == 100.0f && qg[8] == 4.0f &&
              qg[12] == 104.0f && qg[15] == 107.0f, "halves re-interleave to the joint layout");
    }

    /* (3) C: no tall attn_qg with attention on the card; no tall logit row ever. */
    tall_old = q38_tall(false);
    tall = q38_tall(true);
    check(tall_old.attn_qg_tall && !tall.attn_qg_tall, "C: the tall attn_qg strip is absent");
    check(!tall_old.logits_tall && !tall.logits_tall, "the tall set has no logit row");

    /* (4) Totals against the spec, to 0.001 GiB. */
    tall_old_gib = (double)(tall_old.tiles * TALL_BYTES_PER_TILE) / GIB;
    tall_gib = (double)(tall.tiles * TALL_BYTES_PER_TILE) / GIB;
    mirror_gib = (double)(tall.mirror_tiles * TALL_BYTES_PER_TILE) / GIB;
    printf("  weights %.3f GiB (%.1f MiB); A frees %.3f GiB (%.1f MiB a bank)\n",
           (double)before.bytes / GIB, (double)before.bytes / (1024.0 * 1024.0),
           (double)freed / GIB, (double)freed_per_bank / (1024.0 * 1024.0));
    printf("  tall set %.3f GiB (%llu tiles), with C %.3f GiB; mirrors %.3f GiB (%llu tiles)\n",
           tall_old_gib, (unsigned long long)tall_old.tiles, tall_gib, mirror_gib,
           (unsigned long long)tall.mirror_tiles);
    check(tall_old.tiles == 12228u && tall_old.mirror_tiles == 7616u,
          "12,228 tall tiles, 7,616 of them mirrors");
    check((double)before.bytes / (1024.0 * 1024.0) > 25970.15 &&
          (double)before.bytes / (1024.0 * 1024.0) < 25970.25,
          "weights 25,970.2 MiB, as the load log");
    check((double)freed / GIB > 0.9955 && (double)freed / GIB < 0.9965, "A frees 0.996 GiB");
    check(tall_old_gib > 1.4925 && tall_old_gib < 1.4935, "tall set 1.493 GiB");
    check(mirror_gib > 0.9295 && mirror_gib < 0.9305, "mirrors 0.930 GiB");
    check(tall_old_gib - tall_gib > 0.0465 && tall_old_gib - tall_gib < 0.0475,
          "C saves 0.047 GiB");
    margin_32k = CENSUS_32K.free_gib + (double)freed / GIB - tall_gib;
    printf("  32k card margin after A+B+C: %.3f GiB\n", margin_32k);
    check(margin_32k > 0.415 && margin_32k < 0.425, "32k margin ~0.42 GiB");

    /* (5) The conditional policy: main's placement for Ornith, Qwen3.6 and Bonsai; the fit
     * policy for Qwen3.8 Bfp8 with its tall set. */
    check_conditional_policy(before.bytes, before.joint_q_bytes);
    check_mtp_floor(before.bytes, before.joint_q_bytes);
    check_kv_format_floor(before.bytes, before.joint_q_bytes);

    /* (2) B: mirrors first and every bank holds at 16k and 32k, with A and C. */
    q38_check_context(&CENSUS_16K, &tall, freed_per_bank, "A+B+C");
    q38_check_context(&CENSUS_32K, &tall, freed_per_bank, "A+B+C");
    /* The spec's "16k fits with B alone". */
    q38_check_context(&CENSUS_16K, &tall_old, 0u, "B    ");

    printf("test_tt_q38_fit: %d checks, %d failed\n", g_checks, g_failures);
    return g_failures ? 1 : 0;
}

#else

int main(void)
{
    printf("test_tt_q38_fit: needs USE_TT=1 (no card)\n");
    return 1;
}

#endif /* USE_TT */
