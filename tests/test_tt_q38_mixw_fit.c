/* SPDX-License-Identifier: AGPL-3.0-or-later */
/* Copyright (C) 2026 David (Corey) Herrera */
/**
 * @file test_tt_q38_mixw_fit.c
 * @brief Qwen3.8-27B on one p150: a per-tensor Bfp8/Bfp4 format map priced per DRAM bank,
 *        against the real census.
 *
 * Opens no device and needs no USE_TT. It checks:
 *   1. NO MAP REPRODUCES THE BFP8 CENSUS: 497 N-major entries, 27,231,682,560 B (the load log's
 *      25,970.2 MiB), and with the scratch rows the
 *      card's "DRAM after weight upload" line: 25.377 GiB used, fullest bank 1 at 3.185 GiB.
 *      An explicit all-bfp8_b map changes no byte in any bank.
 *   2. A UNIFORM BFP4 MAP gives the spec §3.1 rows class by class and its 14.30 GiB total.
 *   3. THE OUTPUT-EXACT MTP SUBSET (MTP block + draft-vocabulary copy at Bfp4) frees 0.355 GiB
 *      and leaves every trunk byte where it was.
 *   4. REFUSALS: a map that splits a group (one placement named alone, or the draft copy
 *      apart from `output.weight` without a staging repack), unknown or repeated names,
 *      groups the converter will not map, and formats no dense resident program serves.
 *   5. THE 256K TIER: every
 *      card class against the measured 30,720 + ckpt + k1 K8V4 census, then the bank budget
 *      (qwen35moe_tt_fit_plan.h) at 262,144 for M-LL (192 FFN matrices Bfp4) and M-LL-e (layers
 *      0, 1, 62, 63 FFN kept Bfp8), with the prefill gate-block array, the DFlash2 variants and
 *      the largest bf16-KV context. M-LL + ckpt + k1 keeps >= 128 MiB in its fullest bank;
 *      no map at 262,144 is refused.
 *   6. THE TIER PROFILE IN THE TEST CATALOG (usable plan 5.1,
 *      `tests/local/fixtures/catalog_q38_tier.json`): ctx_max 262,144, K8V4, MTP k1; its
 *      memory model at 262,144 is this budget's card total (gates FLAT) within 1 MiB a bank, its
 *      fixed bytes are the default's minus exactly the M-LL bytes, it fits a p150 with the load
 *      margin, and its files follow the card-artifact and MTP companion names. The default model
 *      in it is catalog/catalog.json's. Red: the default's memory model at 262,144 does not fit.
 *
 * Build: `make tests/test_tt_q38_mixw_fit` (any host; run from the repo root).
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

#include "../src/accel/tt/pn_tt_mixw_fit.h"
#include "../src/local/catalog_manifest.h"
#include "../src/local/load_options.h"
#include "../src/models/qwen35moe_tt_fit_plan.h"
#include "../src/models/qwen35moe_tt_mixw_census.h"

#define GIB (1024.0 * 1024.0 * 1024.0)
#define MIB (1024.0 * 1024.0)
#define CENSUS_ROUNDING_GIB 0.0005 /* logs and the spec print GiB to 3 decimals */
#define P150_BANK_BYTES 4278190080ull
#define Q38_TRUNK_NMAJOR_BYTES 27231682560ull /* list_trunk.txt: the 497 N-major entries */
#define Q38_MTP_MATMUL_BYTES 451215360ull     /* 414,720 tiles; the .mtp.fttw is 451,323,072 B */
#define Q38_MTP_ARTIFACT 451323072ull       /* the .mtp.fttw on disk (test_tt_q38_fit.c) */
#define Q38_DRAFT_COLS 2057u                  /* 2,048 + the 9-tile special tail */
#define Q38_SCRATCH_BANK0 (544ull * 2048ull)  /* pn_tt_alloc_scratch: max Kt x bf16 tile */
#define Q38_SCRATCH_BANK1 (7760ull * 2048ull) /* max Nt x bf16 tile */

static int g_checks, g_failures;

static void check(bool ok, const char *what)
{
    g_checks++;
    if (!ok) {
        g_failures++;
        printf("FAIL %s\n", what);
    }
}

static bool near_gib(uint64_t bytes, double want_gib, double tolerance)
{
    const double diff = (double)bytes / GIB - want_gib;

    return diff <= tolerance && diff >= -tolerance;
}

static const pn_tt_mixw_card Q38_CARD = {8u, 16u, 2u, PN_TT_MIXW_TILE_BYTES_BFP8};

static pn_qwen35_tt_mixw_shape q38_shape(bool mtp, bool draft_repacked)
{
    pn_qwen35_tt_mixw_shape s;

    memset(&s, 0, sizeof(s));
    s.n_layer = 64u;
    s.attn_interval = 4u;
    s.embd_tiles = 160u;
    s.ffn_tiles = 544u;
    s.conv_tiles = 320u;
    s.inner_tiles = 192u;
    s.dt_tiles = 2u;
    s.head_tiles = 192u;
    s.kv_tiles = 32u;
    s.vocab_tiles = 7760u;
    s.split_q_at_load = true; /* the fit policy: Qwen3.8 Bfp8 takes it at every context */
    s.mtp = mtp;
    s.draft_cols = mtp ? Q38_DRAFT_COLS : 0u;
    s.draft_repacked = draft_repacked;
    return s;
}

static pn_qwen35_tt_mixw_census g_census, g_census_b;

static void build(pn_qwen35_tt_mixw_census *c, bool mtp, bool repacked)
{
    const pn_qwen35_tt_mixw_shape s = q38_shape(mtp, repacked);

    check(pn_qwen35_tt_mixw_census_build(&s, c) == 0, "the Qwen3.8 census builds");
}

/* ================================================================
 * Maps
 * ================================================================ */

static pn_tt_mixw_map_entry g_map[PN_QWEN35_TT_MIXW_MAX_PLACEMENTS];

/** Every group a map may name, once, at `format`; `filter` NULL = all, else a name prefix. */
static size_t map_all(const pn_qwen35_tt_mixw_census *c, const char *format, const char *filter)
{
    size_t n = 0u;

    for (size_t i = 0; i < c->count; i++) {
        const pn_tt_mixw_placement *p = &c->placements[i];
        bool seen = false;

        if (p->map_refusal || p->kind == PN_TT_MIXW_PLACE_HOST ||
            (filter && strncmp(p->group_name, filter, strlen(filter)) != 0))
            continue;
        for (size_t j = 0; j < n && !seen; j++)
            seen = strcmp(g_map[j].tensor_name, p->group_name) == 0;
        if (!seen) {
            g_map[n].tensor_name = p->group_name;
            g_map[n++].format_name = format;
        }
    }
    return n;
}

static int price(const pn_qwen35_tt_mixw_census *c, size_t map_count, pn_tt_mixw_fit *fit,
                 uint32_t *tiles, char *reason)
{
    return pn_tt_mixw_fit_price(c->placements, c->count, map_count ? g_map : NULL, map_count,
                                &Q38_CARD, fit, tiles, reason, PN_TT_MIXW_REASON_BYTES);
}

static void print_banks(const char *label, const pn_tt_mixw_fit *fit)
{
    printf("  %-26s", label);
    for (uint32_t b = 0; b < Q38_CARD.bank_count; b++)
        printf(" %8.2f", (double)fit->bank_bytes[b] / MIB);
    printf("  MiB; fullest bank %u\n", fit->fullest_bank);
}

/* ================================================================
 * (1) No map: the uniform Bfp8 census
 * ================================================================ */

static void test_bfp8_census(pn_tt_mixw_fit *trunk8)
{
    pn_tt_mixw_fit explicit8;
    char reason[PN_TT_MIXW_REASON_BYTES] = "";
    uint64_t used = 0u, fullest = 0u;
    uint32_t fullest_bank = 0u, load = 0u;
    size_t n;

    build(&g_census, false, false);
    for (size_t i = 0; i < g_census.count; i++)
        load += g_census.placements[i].kind == PN_TT_MIXW_PLACE_LOAD;
    check(load == 497u, "497 N-major trunk entries, as list_trunk.txt");
    check(price(&g_census, 0u, trunk8, NULL, reason) == 0, "the empty map prices");
    check(trunk8->load_bytes == Q38_TRUNK_NMAJOR_BYTES && trunk8->card_bytes == trunk8->load_bytes,
          "no map: 27,231,682,560 B of trunk weights, all on the card");
    check(trunk8->host_bytes == 1350860800ull, "token_embd is 1,350,860,800 B on the host");
    for (uint32_t b = 0; b < Q38_CARD.bank_count; b++) {
        const uint64_t bank = trunk8->bank_bytes[b] + (b == 0u ? Q38_SCRATCH_BANK0 : 0u) +
                              (b == 1u ? Q38_SCRATCH_BANK1 : 0u);

        used += bank;
        if (bank > fullest) {
            fullest = bank;
            fullest_bank = b;
        }
    }
    printf("  after weight upload: %.3f GiB used; fullest bank %u %.4f GiB used %.4f GiB free\n",
           (double)used / GIB, fullest_bank, (double)fullest / GIB,
           (double)(P150_BANK_BYTES - fullest) / GIB);
    check(near_gib(used, 25.377, CENSUS_ROUNDING_GIB), "card census: 25.377 GiB used");
    check(fullest_bank == 1u, "card census: bank 1 is the fullest");
    check(near_gib(fullest, 3.185, CENSUS_ROUNDING_GIB) &&
              near_gib(P150_BANK_BYTES - fullest, 0.799, CENSUS_ROUNDING_GIB),
          "card census: bank 1 3.185 GiB used, 0.799 GiB free");

    n = map_all(&g_census, "bfp8_b", NULL);
    check(price(&g_census, n, &explicit8, NULL, reason) == 0, "an all-bfp8_b map prices");
    check(memcmp(explicit8.bank_bytes, trunk8->bank_bytes, sizeof(explicit8.bank_bytes)) == 0,
          "an all-bfp8_b map moves no byte in any bank");
    print_banks("Bfp8 trunk", trunk8);

    n = map_all(&g_census, "bfp4_b", NULL);
    check(price(&g_census, n, &explicit8, NULL, reason) == 0, "uniform Bfp4 on the trunk prices");
    print_banks("uniform Bfp4 trunk", &explicit8);
}

/* ================================================================
 * (2) Uniform Bfp4: the spec §3.1 rows
 * ================================================================ */

typedef struct {
    const char *label;
    double bfp8_gib, bfp4_gib;
    uint64_t bytes8, bytes4;
} spec_row;

enum {
    ROW_GATE_UP, ROW_DOWN, ROW_QKV, ROW_GATE, ROW_SSM_OUT, ROW_Q, ROW_KV, ROW_OUT, ROW_HEAD,
    ROW_EMBD, ROW_MTP, ROW_DRAFT, ROW_COUNT, ROW_NONE = ROW_COUNT
};

static bool ends_with(const char *name, const char *suffix)
{
    const size_t a = strlen(name), b = strlen(suffix);

    return a >= b && strcmp(name + a - b, suffix) == 0;
}

/** The spec §3.1 row a placement is counted in (ssm_alpha/beta are in none). */
static int spec_row_of(const pn_tt_mixw_placement *p)
{
    static const struct { const char *suffix; int row; } trunk[] = {
        {".ffn_gate.weight", ROW_GATE_UP}, {".ffn_up.weight", ROW_GATE_UP},
        {".ffn_down.weight", ROW_DOWN},    {".attn_qkv.weight", ROW_QKV},
        {".attn_gate.weight", ROW_GATE},   {".ssm_out.weight", ROW_SSM_OUT},
        {".attn_q.weight", ROW_Q},         {".attn_k.weight", ROW_KV},
        {".attn_v.weight", ROW_KV},        {".attn_output.weight", ROW_OUT},
        {"output.weight", ROW_HEAD},       {"token_embd.weight", ROW_EMBD},
    };

    if (strcmp(p->placement_name, "mtp.draft_vocab.weight") == 0)
        return ROW_DRAFT;
    if (strncmp(p->group_name, "blk.64.", 7) == 0)
        return ROW_MTP;
    for (size_t i = 0; i < sizeof(trunk) / sizeof(trunk[0]); i++)
        if (ends_with(p->placement_name, trunk[i].suffix))
            return trunk[i].row;
    return ROW_NONE;
}

/** Least bytes freed in any bank going from `from` to `to`. */
static uint64_t least_freed(const pn_tt_mixw_fit *from, const pn_tt_mixw_fit *to)
{
    uint64_t least = UINT64_MAX;

    for (uint32_t b = 0; b < Q38_CARD.bank_count; b++)
        if (from->bank_bytes[b] - to->bank_bytes[b] < least)
            least = from->bank_bytes[b] - to->bank_bytes[b];
    return least;
}

static void test_uniform_bfp4(const pn_tt_mixw_fit *trunk8, pn_tt_mixw_fit *all8)
{
    static uint32_t tiles[PN_QWEN35_TT_MIXW_MAX_PLACEMENTS];
    spec_row rows[ROW_COUNT] = {
        {"ffn_gate + ffn_up", 11.289, 5.977, 0, 0}, {"ffn_down", 5.645, 2.988, 0, 0},
        {"GDN attn_qkv", 2.490, 1.318, 0, 0},       {"GDN attn_gate", 1.494, 0.791, 0, 0},
        {"GDN ssm_out", 1.494, 0.791, 0, 0},        {"attn_q (q + gate)", 0.996, 0.527, 0, 0},
        {"attn_k + attn_v", 0.166, 0.088, 0, 0},    {"attn_output", 0.498, 0.264, 0, 0},
        {"output (LM head)", 1.258, 0.666, 0, 0},   {"token_embd (host)", 1.258, 0.666, 0, 0},
        {"MTP block", 0.420, 0.222, 0, 0},          {"MTP draft-vocab copy", 0.333, 0.177, 0, 0},
    };
    const uint64_t alpha_beta8 = 96ull * 160u * 2u * PN_TT_MIXW_TILE_BYTES_BFP8;
    pn_tt_mixw_fit fit4;
    char reason[PN_TT_MIXW_REASON_BYTES] = "", what[200];
    uint64_t total4 = 0u;
    bool rot_ok = true;
    size_t n;

    build(&g_census, true, false);
    check(price(&g_census, 0u, all8, NULL, reason) == 0, "Bfp8 with MTP prices");
    n = map_all(&g_census, "bfp4_b", NULL);
    check(price(&g_census, n, &fit4, tiles, reason) == 0, "the uniform Bfp4 map prices");
    for (size_t i = 0; i < g_census.count; i++) {
        const pn_tt_mixw_placement *p = &g_census.placements[i];
        const int r = spec_row_of(p);

        if (r == ROW_NONE)
            continue;
        rows[r].bytes8 += pn_tt_mixw_placement_bytes(p, PN_TT_MIXW_TILE_BYTES_BFP8);
        /* token_embd is unmappable: its spec row is what Bfp4 WOULD be, on the host. */
        rows[r].bytes4 += pn_tt_mixw_placement_bytes(
            p, p->kind == PN_TT_MIXW_PLACE_HOST ? PN_TT_MIXW_TILE_BYTES_BFP4 : tiles[i]);
    }
    printf("  %-22s %9s %9s   (spec §3.1)\n", "class", "Bfp8 GiB", "Bfp4 GiB");
    for (int r = 0; r < ROW_COUNT; r++) {
        printf("  %-22s %9.3f %9.3f\n", rows[r].label, rows[r].bytes8 / GIB, rows[r].bytes4 / GIB);
        snprintf(what, sizeof(what), "§3.1 row %s: %.3f / %.3f GiB", rows[r].label,
                 rows[r].bfp8_gib, rows[r].bfp4_gib);
        check(near_gib(rows[r].bytes8, rows[r].bfp8_gib, CENSUS_ROUNDING_GIB) &&
                  near_gib(rows[r].bytes4, rows[r].bfp4_gib, CENSUS_ROUNDING_GIB), what);
        if (r != ROW_DRAFT)
            total4 += rows[r].bytes4; /* the spec's total row has no draft copy */
    }
    printf("  spec total (token_embd at Bfp4 on the host, no draft copy): %.3f GiB\n",
           total4 / GIB);
    check(near_gib(total4, 14.30, 0.005), "§3.1 total matmul weights at Bfp4: 14.30 GiB");
    check(fit4.host_bytes == 1350860800ull,
          "token_embd stays 1,350,860,800 B on the host: a map cannot move it");
    check(all8->load_bytes == trunk8->load_bytes && all8->staged_bytes ==
              Q38_MTP_MATMUL_BYTES + (uint64_t)Q38_DRAFT_COLS * 160u * PN_TT_MIXW_TILE_BYTES_BFP8,
          "Bfp8 MTP: 451,215,360 B of matmuls + the 358,082,560 B draft copy");
    /* Hand-worked rotation: 497 trunk entries move the cursor from bank 2 to 3, the 8 MTP
     * weights back to 3, its two attn_q halves to 5; the 2,057-column copy gives members 0..8
     * one extra column, and members 0 and 8 both sit in bank 5: one column more there. */
    for (uint32_t b = 0; b < Q38_CARD.bank_count; b++)
        rot_ok = rot_ok && all8->bank_bytes[b] - trunk8->bank_bytes[b] ==
                               (all8->staged_bytes - 160ull * 1088u) / 8u +
                                   (b == 5u ? 160ull * 1088u : 0u);
    check(rot_ok && all8->fullest_bank == 5u,
          "rotation: the MTP head is even but bank 5, one draft column (174,080 B) fuller");
    check(fit4.card_bytes == (all8->card_bytes - alpha_beta8) / PN_TT_MIXW_TILE_BYTES_BFP8 *
                                     PN_TT_MIXW_TILE_BYTES_BFP4 + alpha_beta8,
          "uniform Bfp4 = every card tile at 576 B but ssm_alpha/beta (96 x 320 tiles)");

    printf("  Bfp4 on the card: %.3f GiB (trunk %.3f + staged MTP %.3f) vs Bfp8 %.3f; "
           "%u placements at 576 B\n", fit4.card_bytes / GIB, fit4.load_bytes / GIB,
           fit4.staged_bytes / GIB, all8->card_bytes / GIB, fit4.bfp4_placements);
    print_banks("Bfp8 trunk + MTP", all8);
    print_banks("uniform Bfp4 trunk + MTP", &fit4);
    printf("  fullest bank: Bfp8 %u %.2f MiB -> Bfp4 %u %.2f MiB; least freed in any bank "
           "%.2f MiB\n", all8->fullest_bank, all8->bank_bytes[all8->fullest_bank] / MIB,
           fit4.fullest_bank, fit4.bank_bytes[fit4.fullest_bank] / MIB,
           least_freed(all8, &fit4) / MIB);
}

/* ================================================================
 * (3) The output-exact MTP subset
 * ================================================================ */

static void test_mtp_subset(const pn_tt_mixw_fit *trunk8, const pn_tt_mixw_fit *all8)
{
    pn_tt_mixw_fit subset;
    char reason[PN_TT_MIXW_REASON_BYTES] = "";
    uint64_t freed;
    size_t n;
    bool frees_every_bank = true;

    /* The copy is staged from output.weight's bytes: without a repack it is in that group. */
    build(&g_census, true, false);
    n = map_all(&g_census, "bfp4_b", "blk.64.");
    g_map[n].tensor_name = "mtp.draft_vocab.weight";
    g_map[n++].format_name = "bfp4_b";
    check(price(&g_census, n, &subset, NULL, reason) != 0 &&
              strstr(reason, "splits that group") && strstr(reason, "output.weight"),
          "MTP subset without a staging repack: the draft copy apart from output.weight is a split");
    printf("  refused: %s\n", reason);

    build(&g_census_b, true, true);
    n = map_all(&g_census_b, "bfp4_b", "blk.64.");
    check(n == 8u, "the MTP block is 8 map lines (eh_proj, attn q/k/v/output, ffn x3)");
    g_map[n].tensor_name = "mtp.draft_vocab.weight";
    g_map[n++].format_name = "bfp4_b";
    check(price(&g_census_b, n, &subset, NULL, reason) == 0,
          "MTP subset with the staging repack prices");
    freed = all8->card_bytes - subset.card_bytes;
    check(subset.load_bytes == trunk8->load_bytes, "the subset moves no trunk byte");
    for (uint32_t b = 0; b < Q38_CARD.bank_count; b++)
        frees_every_bank = frees_every_bank && subset.bank_bytes[b] - trunk8->bank_bytes[b] <
                                       all8->bank_bytes[b] - trunk8->bank_bytes[b];
    check(frees_every_bank, "the subset frees bytes in every bank");
    check(near_gib(freed, 0.355, CENSUS_ROUNDING_GIB), "the subset frees 0.355 GiB (spec §3.1)");
    print_banks("MTP subset trunk + MTP", &subset);
    printf("  MTP subset: frees %.3f GiB (%.2f MiB a bank avg); MTP + copy per bank: Bfp8 ",
           freed / GIB, freed / MIB / Q38_CARD.bank_count);
    for (uint32_t b = 0; b < Q38_CARD.bank_count; b++)
        printf("%.2f%s", (all8->bank_bytes[b] - trunk8->bank_bytes[b]) / MIB,
               b + 1u < Q38_CARD.bank_count ? "/" : "");
    printf(" -> Bfp4 ");
    for (uint32_t b = 0; b < Q38_CARD.bank_count; b++)
        printf("%.2f%s", (subset.bank_bytes[b] - trunk8->bank_bytes[b]) / MIB,
               b + 1u < Q38_CARD.bank_count ? "/" : "");
    printf(" MiB; fullest bank %u %.2f MiB (Bfp8 %u %.2f)\n", subset.fullest_bank,
           subset.bank_bytes[subset.fullest_bank] / MIB, all8->fullest_bank,
           all8->bank_bytes[all8->fullest_bank] / MIB);
}

/* ================================================================
 * (3b) The fit floor (qwen35moe_tt_fit_plan.h) fed from a priced map
 * ================================================================ */

/** Bytes of the trunk's attn_q groups (the floor's second copy) at the map's tiles. */
static uint64_t trunk_q_bytes(const pn_qwen35_tt_mixw_census *c, const uint32_t *tiles)
{
    uint64_t bytes = 0u;

    for (size_t i = 0; i < c->count; i++)
        if (c->placements[i].kind == PN_TT_MIXW_PLACE_LOAD &&
            ends_with(c->placements[i].placement_name, ".attn_q.weight"))
            bytes += pn_tt_mixw_placement_bytes(&c->placements[i], tiles[i]);
    return bytes;
}

/** Main's floor at 32k with MTP (4 verify rows), from the priced census. */
static pn_qwen35_tt_fit_floor floor_of(const pn_qwen35_tt_mixw_census *c, size_t map_count,
                                       pn_tt_mixw_fit *fit)
{
    static uint32_t tiles[PN_QWEN35_TT_MIXW_MAX_PLACEMENTS];
    pn_qwen35_tt_fit_shape s;
    pn_qwen35_tt_fit_floor floor;
    char reason[PN_TT_MIXW_REASON_BYTES] = "";

    memset(&s, 0, sizeof(s));
    s.n_embd = 5120u; s.ffn = 17408u; s.n_head = 24u; s.n_head_kv = 4u; s.head_dim = 256u;
    s.n_attn = 16u; s.n_ssm = 48u; s.d_inner = 6144u; s.v_heads = 48u; s.conv_dim = 10240u;
    s.conv_taps = 4u; s.ssm_state = 128u; s.ctx = 32768u; s.tall_rows = 64u; s.mirrors = 8u;
    s.outproj_mirrors = true; s.gdnc_replicas = 8u; s.attn_on_card = true;
    s.gdn_on_card = true; s.state_split = true; s.mtp_rows = 4u;
    check(price(c, map_count, fit, tiles, reason) == 0, "the floor's map prices");
    /* The head term counts the MTP artifact's fp32 vectors too: they do not change format. */
    s.mtp_bytes = fit->staged_bytes + (Q38_MTP_ARTIFACT - Q38_MTP_MATMUL_BYTES);
    pn_qwen35_tt_fit_floor_compute(&s, fit->load_bytes, trunk_q_bytes(c, tiles), &floor);
    return floor;
}

static void test_fit_floor(void)
{
    pn_tt_mixw_fit fit;
    pn_qwen35_tt_fit_floor f8, f4, fsub;
    size_t n;

    build(&g_census, true, false);
    f8 = floor_of(&g_census, 0u, &fit);
    check(f8.weights == Q38_TRUNK_NMAJOR_BYTES && f8.second_q == 1069547520ull,
          "Bfp8 floor: the weights and joint attn_q test_tt_q38_fit.c feeds it");
    /* test_tt_q38_fit.c's MTP term: head + one KV layer x ctx + 4 verify slots of 324 MiB. */
    check(f8.mtp == Q38_MTP_ARTIFACT + (uint64_t)Q38_DRAFT_COLS * 160u * 1088u +
                        32768ull * 4096ull + 4ull * 339738624ull,
          "Bfp8 floor: the MTP head term is test_tt_q38_fit.c's");
    n = map_all(&g_census, "bfp4_b", NULL);
    f4 = floor_of(&g_census, n, &fit);
    build(&g_census_b, true, true);
    n = map_all(&g_census_b, "bfp4_b", "blk.64.");
    g_map[n].tensor_name = "mtp.draft_vocab.weight";
    g_map[n++].format_name = "bfp4_b";
    fsub = floor_of(&g_census_b, n, &fit);
    printf("  main's floor at 32k + MTP: Bfp8 %.3f GiB, uniform Bfp4 %.3f GiB, MTP subset %.3f "
           "GiB (card 31.875)\n", f8.total / GIB, f4.total / GIB, fsub.total / GIB);
    check(f8.total - fsub.total == 380846080ull,
          "the subset lowers the floor by exactly its 380,846,080 freed bytes");
}

/* ================================================================
 * (4) Refusals and the groups
 * ================================================================ */

static void expect_refused(const pn_qwen35_tt_mixw_census *c, const char *name,
                           const char *format, const char *needle, const char *what)
{
    pn_tt_mixw_fit fit;
    char reason[PN_TT_MIXW_REASON_BYTES] = "";

    g_map[0].tensor_name = name;
    g_map[0].format_name = format;
    check(price(c, 1u, &fit, NULL, reason) != 0 && strstr(reason, needle), what);
}

static void test_refusals(void)
{
    pn_tt_mixw_fit fit;
    char reason[PN_TT_MIXW_REASON_BYTES] = "";

    build(&g_census, true, false);
    expect_refused(&g_census, "mtp.eh_proj_e.weight", "bfp4_b", "splits that group",
                   "eh_proj_e named alone splits blk.64.nextn.eh_proj.weight");
    expect_refused(&g_census, "mtp.attn_q.weight (Q half)", "bfp4_b", "splits that group",
                   "one MTP attn_q half named alone is a split");
    expect_refused(&g_census, "token_embd.weight", "bfp4_b", "host-side row table",
                   "token_embd is refused (host-side, the converter refuses it)");
    expect_refused(&g_census, "blk.0.ssm_alpha.weight", "bfp4_b", "ssm_alpha",
                   "ssm_alpha is refused (the converter refuses it)");
    expect_refused(&g_census, "blk.0.ffn_up.weight", "bfp6_b", "only bfp8_b and bfp4_b",
                   "bfp6_b has no dense resident program");
    expect_refused(&g_census, "blk.64.ffn_up.weight.", "bfp4_b", "not a tensor",
                   "an unknown name is refused");
    g_map[0].tensor_name = g_map[1].tensor_name = "blk.0.ffn_up.weight";
    g_map[0].format_name = "bfp4_b";
    g_map[1].format_name = "bfp8_b";
    check(price(&g_census, 2u, &fit, NULL, reason) != 0 && strstr(reason, "listed twice"),
          "a name listed twice is refused");

    /* Without the fit policy's split: the joint + two staged halves share one format. */
    {
        pn_qwen35_tt_mixw_shape s = q38_shape(false, false);
        static uint32_t tiles[PN_QWEN35_TT_MIXW_MAX_PLACEMENTS];
        uint32_t at576 = 0u;

        s.split_q_at_load = false;
        check(pn_qwen35_tt_mixw_census_build(&s, &g_census_b) == 0, "the unsplit census builds");
        check(pn_tt_mixw_group_size(g_census_b.placements, g_census_b.count,
                                    "blk.3.attn_q.weight") == 3u,
              "unsplit: blk.3.attn_q is the joint + its Q and gate copies");
        g_map[0].tensor_name = "blk.3.attn_q.weight";
        g_map[0].format_name = "bfp4_b";
        check(price(&g_census_b, 1u, &fit, tiles, reason) == 0, "unsplit attn_q map prices");
        for (size_t i = 0; i < g_census_b.count; i++)
            at576 += tiles[i] == PN_TT_MIXW_TILE_BYTES_BFP4;
        check(at576 == 3u && fit.bfp4_placements == 3u, "all three take the group's format");
    }
}

/** The groups a map moves as one (more than one placement or part), MTP on, split at load. */
static void print_groups(void)
{
    uint32_t trunk_q = 0u, groups = 0u;

    build(&g_census, true, false);
    printf("  groups of more than one placement (MTP on, split at load, no staging repack):\n");
    for (size_t i = 0; i < g_census.count; i++) {
        const pn_tt_mixw_placement *p = &g_census.placements[i];
        const size_t members = pn_tt_mixw_group_size(g_census.placements, g_census.count,
                                                     p->group_name);
        bool first = true;

        for (size_t j = 0; j < i && first; j++)
            first = strcmp(g_census.placements[j].group_name, p->group_name) != 0;
        if (!first || (members < 2u && p->parts < 2u))
            continue;
        groups++;
        if (p->kind == PN_TT_MIXW_PLACE_LOAD && p->parts > 1u) {
            trunk_q++;
            continue;
        }
        printf("    %-30s %zu placements\n", p->group_name, members);
    }
    printf("    blk.N.attn_q.weight x %u        1 entry of 2 parts each (the halves)\n", trunk_q);
    check(trunk_q == 16u && groups == 19u,
          "19 groups move as one: 16 trunk attn_q, output, MTP eh_proj, MTP attn_q");
}

/* ================================================================
 * (5) The 256k tier: every card class, then the bank budget
 * ================================================================ */

/* THE ANCHOR: Bfp8
 * trunk + MTP, K8V4, 30,720 + prefix checkpoint + MTP k1 (one compact verify slot), the fit
 * policy (A + B + C + D), gate family not armed (main: 32 heads at most). Each stage's card-wide
 * "used" GiB as printed (3 decimals), and the fullest bank's free after the last stage, taken
 * at the LOW end of its rounding: "bank 4 ... 0.003 GiB free" -> 0.0025 GiB.
 * RoPE tables do not grow with context: pn_tt_attention.c builds one 32-row block at a time
 * (decode, chunk1, the verify's rows, 62 tall slots: 4 bf16 tiles = 8 KiB each, ~0.5 MiB in
 * all), inside the census's small arms. Spec §3's +8 MiB a bank at 262k assumed a whole-context
 * table the code does not allocate. */
#define ANCHOR_CTX 30720u
#define ANCHOR_FULLEST_FREE 2684354ull /* 0.0025 GiB */
#define K8V4_K_TILE 1088u
#define K8V4_V_TILE 576u
#define Q38_KV_LAYERS 17u            /* 16 attention layers + the MTP head's */
#define Q38_DFLASH_KV_LAYERS 16u     /* DFlash2 replaces the MTP head: no 17th region */
#define TIER_CTX 262144u
#define GATE_ROWS_A_PASS (64u * 32u) /* PN_TT_PREFILL_TILE_ROWS 64 x 32 rows */
/* pn_tt_gdngate.c gdngate_upload_constants: ssm_a[64] | ssm_dt[64] fp32 a layer, in bank 0. */
#define GATE_CONSTANT_BYTES (48ull * 128u * 4u)
/* DFlash2 drafter + head + taps at Bfp8 (spec §3, E), card-wide, even like the trunk. */
#define DFLASH_DRAFTER_BYTES ((uint64_t)(2.43 * GIB))
#define DFLASH_VERIFY_ROWS 8u

enum { ST_UPLOAD, ST_STRIPS, ST_GDN, ST_ATTN, ST_MTP, ST_TALL, ST_RESERVE, ST_SLOT, ST_CKPT,
       ST_COUNT };
static const struct {
    const char *stage;
    double used_gib, small_max_gib; /* census; the stage's arms no shape term prices */
} ANCHOR_STAGES[ST_COUNT] = {
    {"after weight upload", 25.377, 0.0}, {"activation strips", 25.479, 0.110},
    {"gdn state + strip bridge", 27.689, 0.007}, {"attention + KV", 28.583, 0.086},
    {"MTP head", 29.370, 0.040}, {"tall prefill strips", 30.815, 0.0},
    {"prefill buffers", 30.960, 0.005}, {"MTP verify slots", 31.276, 0.0},
    {"prefix checkpoint", 31.592, 0.0},
};

static pn_qwen35_tt_fit_shape q38_fit_shape(uint32_t ctx, uint32_t k_tile, uint32_t v_tile)
{
    pn_qwen35_tt_fit_shape s;

    memset(&s, 0, sizeof(s));
    s.n_embd = 5120u; s.ffn = 17408u; s.n_head = 24u; s.n_head_kv = 4u; s.head_dim = 256u;
    s.n_attn = 16u; s.n_ssm = 48u; s.d_inner = 6144u; s.v_heads = 48u; s.conv_dim = 10240u;
    s.conv_taps = 4u; s.ssm_state = 128u; s.ctx = ctx; s.tall_rows = 64u; s.mirrors = 8u;
    s.outproj_mirrors = true; s.gdnc_replicas = 8u; s.attn_on_card = true;
    s.gdn_on_card = true; s.state_split = true;
    s.kv_k_tile_bytes = k_tile;
    s.kv_v_tile_bytes = v_tile;
    return s;
}

/** M-LL: every trunk blk.N.ffn_gate|ffn_up|ffn_down.weight at bfp4_b; `edge` keeps layers 0, 1,
 *  62, 63 at Bfp8 (M-LL-e). The MTP block (blk.64) stays Bfp8. */
static size_t map_mll(const pn_qwen35_tt_mixw_census *c, bool edge)
{
    size_t n = 0u;

    for (size_t i = 0; i < c->count; i++) {
        const char *name = c->placements[i].group_name;
        unsigned layer = 0u;
        char kind[16] = "";

        if (c->placements[i].kind != PN_TT_MIXW_PLACE_LOAD ||
            sscanf(name, "blk.%u.ffn_%15[a-z].weight", &layer, kind) != 2 || layer >= 64u ||
            (strcmp(kind, "gate") != 0 && strcmp(kind, "up") != 0 && strcmp(kind, "down") != 0) ||
            (edge && (layer < 2u || layer > 61u)))
            continue;
        g_map[n].tensor_name = name;
        g_map[n++].format_name = "bfp4_b";
    }
    return n;
}

/** The card classes of one load (bytes), card-wide. */
typedef struct {
    uint64_t trunk, staged, drafter, gdn, gdnc, kv, tall, reserve, verify, ckpt, gate;
    uint64_t second_q, conv_own; /* main's placement only (tier_policy_off) */
    uint64_t small[ST_COUNT];
} tier_classes;

/** Every class a shape term prices, for Qwen3.8 under the fit policy with D (conv share). */
static void tier_price_classes(const pn_qwen35_tt_fit_shape *s, const pn_tt_mixw_fit *weights,
                               uint64_t drafter, uint32_t kv_layers, uint64_t verify,
                               uint64_t gate, tier_classes *t)
{
    const uint64_t conv_tiles = (s->conv_dim + 31u) / 32u;
    /* pn_tt_gdnc.c: the row-replicated conv constant (taps planes + 2 x taps band tiles) a
     * layer, x replicas; the DELTA constant (4 program tiles + 1 a v-head) and GATE block in
     * each of 2 staging slots. D drops decode's own conv weight: state + window remain. */
    const uint64_t gdnc_constant = (conv_tiles * s->conv_taps + 2u * s->conv_taps) * 4096u;

    memset(t, 0, sizeof(*t));
    t->trunk = weights->load_bytes;
    t->staged = weights->staged_bytes;
    t->drafter = drafter;
    t->gdn = pn_qwen35_tt_fit_snapshot_bytes(s);
    t->gdnc = (uint64_t)s->n_ssm * gdnc_constant * s->gdnc_replicas +
              2ull * (4u + s->v_heads) * 4096u + pn_qwen35_tt_fit_gdnc_gate_bytes(s->v_heads);
    t->kv = 4u * (pn_qwen35_tt_fit_kv_bank_bytes(s, kv_layers, 8u, true) +
                  pn_qwen35_tt_fit_kv_bank_bytes(s, kv_layers, 8u, false));
    /* C: no tall attn_qg (2 x heads tiles) under the fit policy. */
    t->tall = (pn_qwen35_tt_fit_tall_tiles(s) - 2u * (s->n_head * s->head_dim / 32u)) *
              s->tall_rows * 2048u;
    t->reserve = pn_qwen35_tt_fit_state_bytes(s); /* the GDNC state split */
    t->verify = verify;
    t->ckpt = pn_qwen35_tt_fit_snapshot_bytes(s);
    t->gate = gate;
}

static uint64_t tier_small(const tier_classes *t)
{
    uint64_t small = 0u;

    for (int i = 0; i < ST_COUNT; i++)
        small += t->small[i];
    return small;
}

static uint64_t tier_total(const tier_classes *t)
{
    return t->trunk + t->staged + t->drafter + t->gdn + t->gdnc + t->kv + t->tall + t->reserve +
           t->verify + t->ckpt + t->gate + t->second_q + t->conv_own + tier_small(t);
}

/** The anchor's classes against its census, stage by stage: no class may exceed its stage, and
 *  what is left over is the stage's small arms (which the tier keeps unchanged). */
static void test_anchor_census(const pn_tt_mixw_fit *all8, tier_classes *anchor)
{
    const pn_qwen35_tt_fit_shape s = q38_fit_shape(ANCHOR_CTX, K8V4_K_TILE, K8V4_V_TILE);
    uint64_t computed[ST_COUNT];
    char what[200];

    tier_price_classes(&s, all8, 0u, Q38_KV_LAYERS, pn_qwen35_tt_fit_snapshot_bytes(&s), 0u,
                       anchor);
    computed[ST_UPLOAD] = anchor->trunk + Q38_SCRATCH_BANK0 + Q38_SCRATCH_BANK1;
    computed[ST_STRIPS] = 0u;
    computed[ST_GDN] = anchor->gdn + anchor->gdnc;
    computed[ST_ATTN] = anchor->kv;
    computed[ST_MTP] = anchor->staged;
    computed[ST_TALL] = anchor->tall;
    computed[ST_RESERVE] = anchor->reserve;
    computed[ST_SLOT] = anchor->verify;
    computed[ST_CKPT] = anchor->ckpt;
    printf("  anchor 30,720 + ckpt + k1 K8V4 (p150 census): stage, measured, priced, left\n");
    for (int i = 0; i < ST_COUNT; i++) {
        const double measured = ANCHOR_STAGES[i].used_gib - (i ? ANCHOR_STAGES[i - 1].used_gib
                                                              : 0.0);
        const double left = measured - computed[i] / GIB;

        anchor->small[i] = left > 0.0 ? (uint64_t)(left * GIB) : 0u;
        printf("    %-26s %7.3f %7.3f %7.3f GiB\n", ANCHOR_STAGES[i].stage, measured,
               computed[i] / GIB, left);
        snprintf(what, sizeof(what), "census stage '%s': priced %.4f GiB within the measured "
                 "%.3f, %.4f GiB left (at most %.3f)", ANCHOR_STAGES[i].stage, computed[i] / GIB,
                 measured, left, ANCHOR_STAGES[i].small_max_gib);
        check(left >= -0.0011 && left <= ANCHOR_STAGES[i].small_max_gib + 0.0011, what);
    }
    anchor->small[ST_UPLOAD] = Q38_SCRATCH_BANK0 + Q38_SCRATCH_BANK1;
    check(near_gib(tier_total(anchor), 31.592, 0.002),
          "every anchor class sums to the census's 31.592 GiB used");
}

/* BLOCK 1 (2026-11-18):
 * the GPTQ M-LL tier at 262,144, K8V4, prefix checkpoint, MTP k1 (PN_TT_SPEC=1), gates FLAT.
 * Main's floor fits the card there (no "main's placement does not fit" line), so the load runs
 * MAIN'S PLACEMENT, not the fit policy the anchor was measured under: attn_q whole plus its
 * staged halves (the second copy), each layer's own decode conv weight (no D), the tall attn_qg,
 * and the verify's GDN slots allocated at the first verify rather than reserved at load. The
 * fullest bank after the arm: "bank 4 3.870 GiB used 0.114 GiB free" (low end 0.1135). */
#define BLOCK1_FULLEST_FREE 121870295ull /* 0.1135 GiB */
#define BLOCK1_CARD_FREE ((uint64_t)(31.859 * GIB)) /* free at the decide, as printed */
#define Q38_VERIFY_ROWS_K1 2u /* pn_qwen35_tt_mtp_verify_rows() at PN_TT_SPEC=1 */
static const double BLOCK1_USED_GIB[ST_COUNT] = {
    17.409, 17.511, 19.955, 27.943, 28.729, 30.222, 30.366, 30.366 /* no slot stage */, 30.683,
};

/** What main's placement adds to (and moves out of) the fit policy's classes: the second attn_q,
 *  the decode conv weights (one taps x conv_dim/32 fp32 plane set a layer), the tall attn_qg
 *  (2 x heads tile columns x tall rows), and no verify slots at load. */
static void tier_policy_off(const pn_qwen35_tt_fit_shape *s, uint64_t joint_q, tier_classes *t)
{
    t->second_q = joint_q;
    t->conv_own = (uint64_t)s->n_ssm * s->conv_taps * ((s->conv_dim + 31u) / 32u) * 4096u;
    t->tall += 2ull * (s->n_head * s->head_dim / 32u) * s->tall_rows * 2048u;
    t->verify = 0u;
}

/** The fit policy's tall set at the tier (no attn_qg): main's adds the qg on top. */
static uint64_t ct_flat_tall(const pn_tt_mixw_fit *weights)
{
    const pn_qwen35_tt_fit_shape s = q38_fit_shape(TIER_CTX, K8V4_K_TILE, K8V4_V_TILE);
    tier_classes t;

    tier_price_classes(&s, weights, 0u, Q38_KV_LAYERS, 0u, 0u, &t);
    return t.tall;
}

/** Main's floor for a load, as pn_qwen35_tt_fit_decide computes it. */
static pn_qwen35_tt_fit_floor tier_floor(const pn_tt_mixw_fit *weights, uint64_t joint_q,
                                         uint32_t ctx, uint32_t verify_rows)
{
    pn_qwen35_tt_fit_shape s = q38_fit_shape(ctx, K8V4_K_TILE, K8V4_V_TILE);
    pn_qwen35_tt_fit_floor floor;

    s.mtp_bytes = weights->staged_bytes + (Q38_MTP_ARTIFACT - Q38_MTP_MATMUL_BYTES);
    s.mtp_rows = verify_rows;
    pn_qwen35_tt_fit_floor_compute(&s, weights->load_bytes, joint_q, &floor);
    return floor;
}

/** Stage `i` of a load's classes: what each census stage allocates (test_anchor_census). */
static uint64_t tier_stage(const tier_classes *t, int i)
{
    switch (i) {
    case ST_UPLOAD: return t->trunk;
    case ST_GDN: return t->gdn + t->gdnc + t->conv_own;
    case ST_ATTN: return t->kv + t->second_q;
    case ST_MTP: return t->staged;
    case ST_TALL: return t->tall;
    case ST_RESERVE: return t->reserve;
    case ST_SLOT: return t->verify;
    case ST_CKPT: return t->ckpt;
    default: return 0u;
    }
}

/**
 * Block 1 against the classes, stage by stage (the anchor's small arms carried over): the fit
 * policy's classes miss it by the second attn_q, the conv weights and the tall attn_qg; main's
 * placement matches every stage to the census's rounding. Returns the policy-off classes.
 */
static void test_block1_census(const pn_tt_mixw_fit *all8, const pn_tt_mixw_fit *mll,
                               uint64_t joint_q, const tier_classes *ca, tier_classes *ct)
{
    const pn_qwen35_tt_fit_shape s = q38_fit_shape(TIER_CTX, K8V4_K_TILE, K8V4_V_TILE);
    const pn_qwen35_tt_fit_floor tier = tier_floor(mll, joint_q, TIER_CTX, Q38_VERIFY_ROWS_K1);
    /* attn_q stays Bfp8 under M-LL: the joint bytes are the same on both trunks */
    const pn_qwen35_tt_fit_floor bfp8 = tier_floor(all8, joint_q, ANCHOR_CTX, Q38_VERIFY_ROWS_K1);
    tier_classes on;
    int worst_on = 0, worst_off = 0;
    double miss_on = 0.0, miss_off = 0.0;
    char what[220];

    snprintf(what, sizeof(what), "main's floor: M-LL at 262,144 k1 %.3f GiB + 0.75 headroom fits "
             "the %.3f GiB free (main's placement, as block 1 ran); Bfp8 at 30,720 %.3f does not "
             "(the anchor's fit policy)", tier.total / GIB, BLOCK1_CARD_FREE / GIB,
             bfp8.total / GIB);
    printf("  %s\n", what);
    check(!pn_qwen35_tt_fit_policy_wanted(&tier, BLOCK1_CARD_FREE) &&
              pn_qwen35_tt_fit_policy_wanted(&bfp8, BLOCK1_CARD_FREE), what);

    tier_price_classes(&s, mll, 0u, Q38_KV_LAYERS, pn_qwen35_tt_fit_snapshot_bytes(&s), 0u, &on);
    memcpy(on.small, ca->small, sizeof(on.small));
    *ct = on;
    tier_policy_off(&s, joint_q, ct);
    check(ct->second_q == 1069547520ull && ct->conv_own == 251658240ull &&
              ct->tall - on.tall == 50331648ull,
          "main's placement adds attn_q's 1,069,547,520 B copy, 48 x 5 MiB conv weights and the "
          "48 MiB tall attn_qg");
    printf("  block 1, M-LL 262,144 k1 (p150 census): stage, measured, fit policy, main's\n");
    for (int i = 0; i < ST_COUNT; i++) {
        const double measured = BLOCK1_USED_GIB[i] - (i ? BLOCK1_USED_GIB[i - 1] : 0.0);
        const double p_on = (tier_stage(&on, i) + ca->small[i]) / GIB;
        const double p_off = (tier_stage(ct, i) + ca->small[i]) / GIB;

        printf("    %-26s %7.3f %7.3f %7.3f GiB\n", ANCHOR_STAGES[i].stage, measured, p_on, p_off);
        if (fabs(measured - p_on) > miss_on) {
            miss_on = fabs(measured - p_on);
            worst_on = i;
        }
        if (fabs(measured - p_off) > miss_off) {
            miss_off = fabs(measured - p_off);
            worst_off = i;
        }
    }
    snprintf(what, sizeof(what), "RED: the fit policy's classes miss block 1 (worst '%s' by "
             "%.3f GiB)", ANCHOR_STAGES[worst_on].stage, miss_on);
    check(miss_on > 0.2, what);
    snprintf(what, sizeof(what), "GATE: main's placement matches every block-1 stage within the "
             "census rounding (worst '%s' by %.1f MiB)", ANCHOR_STAGES[worst_off].stage,
             miss_off * 1024.0);
    check(miss_off <= 2.0 * CENSUS_ROUNDING_GIB + 0.0011, what);
    snprintf(what, sizeof(what), "block 1 at load: %.3f GiB priced vs 30.683 measured (the "
             "verify's %.0f MiB comes at the first verify)", (tier_total(ct)) / GIB,
             on.verify / MIB);
    check(near_gib(tier_total(ct), 30.683, 0.002), what);
}

typedef struct {
    const char *label;
    pn_qwen35_tt_bank_load load;
    pn_qwen35_tt_bank_fit fit;
} tier_variant;

static void tier_load(pn_qwen35_tt_bank_load *l, const pn_tt_mixw_fit *weights, uint64_t drafter,
                      uint32_t ctx, uint32_t k_tile, uint32_t v_tile, uint32_t kv_layers,
                      uint64_t verify, uint64_t one_bank)
{
    memset(l, 0, sizeof(*l));
    l->shape = q38_fit_shape(ctx, k_tile, v_tile);
    l->kv_layers = kv_layers;
    for (uint32_t b = 0; b < Q38_CARD.bank_count; b++)
        l->weight_bank[b] = weights->bank_bytes[b] + drafter / Q38_CARD.bank_count;
    /* verify state + the prefix checkpoint twin: the even classes that differ by load */
    l->even_bytes = verify + pn_qwen35_tt_fit_snapshot_bytes(&l->shape);
    l->one_bank_bytes = one_bank;
}

static void print_variant(const tier_variant *v, const pn_qwen35_tt_bank_load *anchor)
{
    printf("  %-44s fullest bank %u: %8.1f MiB free (%s 128 MiB)\n", v->label,
           v->fit.binding_bank, v->fit.binding_free / MIB, v->fit.fits ? ">=" : "<");
    printf("    %-20s", "bank");
    for (uint32_t b = 0; b < Q38_CARD.bank_count; b++)
        printf(" %8u", b);
    printf("\n    %-20s", "weights MiB");
    for (uint32_t b = 0; b < Q38_CARD.bank_count; b++)
        printf(" %8.1f", v->load.weight_bank[b] / MIB);
    printf("\n    %-20s", "freed vs anchor");
    for (uint32_t b = 0; b < Q38_CARD.bank_count; b++)
        printf(" %8.1f", ((double)anchor->weight_bank[b] - (double)v->load.weight_bank[b]) / MIB);
    printf("\n    %-20s %8.1f MiB a bank (KV), %.1f MiB a bank (verify + ckpt), %.1f MiB in "
           "bank %u (one-bank)\n", "growth", v->fit.kv_growth / MIB,
           ((double)v->load.even_bytes - (double)anchor->even_bytes) / Q38_CARD.bank_count / MIB,
           ((double)v->load.one_bank_bytes - (double)anchor->one_bank_bytes) / MIB,
           v->fit.binding_bank);
    printf("    %-20s", "free >= MiB");
    for (uint32_t b = 0; b < Q38_CARD.bank_count; b++)
        printf(" %8.1f", v->fit.free_bank[b] / MIB);
    printf("\n");
}

static void price_variant(tier_variant *v, const pn_qwen35_tt_bank_load *anchor)
{
    check(pn_qwen35_tt_bank_fit_compute(anchor, ANCHOR_FULLEST_FREE, &v->load,
                                        Q38_CARD.bank_count, PN_QWEN35_TT_TIER_HEADROOM,
                                        &v->fit) == 0, "the bank budget prices");
    print_variant(v, anchor);
}

/** Largest context (1,024 steps) at which `kv` tiles fit `base` with the headroom. */
static uint32_t tier_max_ctx(const tier_variant *base, const pn_qwen35_tt_bank_load *anchor,
                             uint32_t k_tile, uint32_t v_tile)
{
    uint32_t best = 0u;

    for (uint32_t ctx = 1024u; ctx <= TIER_CTX; ctx += 1024u) {
        tier_variant v = *base;

        v.load.shape = q38_fit_shape(ctx, k_tile, v_tile);
        if (pn_qwen35_tt_bank_fit_compute(anchor, ANCHOR_FULLEST_FREE, &v.load,
                                          Q38_CARD.bank_count, PN_QWEN35_TT_TIER_HEADROOM,
                                          &v.fit) != 0 || !v.fit.fits)
            break;
        best = ctx;
    }
    return best;
}

/** Card-wide classes (spec §3's table, from the shapes), anchor and the tier. */
static void print_class_table(const tier_classes *a, const tier_classes *t, const char *label)
{
    const struct { const char *name; uint64_t av, tv; } rows[] = {
        {"trunk matmul weights", a->trunk, t->trunk},
        {"MTP block + draft copy", a->staged, t->staged},
        {"GDN decode (state+window)", a->gdn, t->gdn},
        {"GDNC conv x8, delta, GATE", a->gdnc, t->gdnc},
        {"tall prefill strips", a->tall, t->tall},
        {"GDNC state split", a->reserve, t->reserve},
        {"DFlash2 drafter", a->drafter, t->drafter},
        {"scratch + small arms", tier_small(a), tier_small(t)},
        {"prefix checkpoint twin", a->ckpt, t->ckpt},
        {"verify state", a->verify, t->verify},
        {"KV", a->kv, t->kv},
        {"gate-block array + consts", a->gate, t->gate},
    };

    printf("  card classes, GiB          anchor 30,720   %s\n", label);
    for (size_t i = 0; i < sizeof(rows) / sizeof(rows[0]); i++)
        printf("    %-28s %8.3f %8.3f\n", rows[i].name, rows[i].av / GIB, rows[i].tv / GIB);
    printf("    %-28s %8.3f %8.3f  (free %.3f / %.3f GiB, %.1f / %.1f MiB a bank average)\n",
           "total", tier_total(a) / GIB, tier_total(t) / GIB,
           (8.0 * P150_BANK_BYTES - tier_total(a)) / GIB,
           (8.0 * P150_BANK_BYTES - tier_total(t)) / GIB,
           (8.0 * P150_BANK_BYTES - tier_total(a)) / 8.0 / MIB,
           (8.0 * P150_BANK_BYTES - tier_total(t)) / 8.0 / MIB);
}

/* ================================================================
 * (6) The tier profile in the test catalog (plan 5.1)
 * ================================================================ */

#define TIER_CATALOG "tests/local/fixtures/catalog_q38_tier.json"
#define REAL_CATALOG "catalog/catalog.json"
#define TIER_MODEL "qwen3.8-27b-tier256k"
#define TIER_PROFILE "p150-one-256k"
#define DEFAULT_MODEL "qwen3.8-27b-q8"
#define DEFAULT_PROFILE "p150-one"
/* The GPTQ M-LL artifact (gptq.bfp4ffn.fttw). */
#define TIER_ARTIFACT_BYTES 20045796760ull
#define TIER_ARTIFACT_SHA "e537e151a5631d5cd528494fd573e2db82a7d38a4b7ceab378478898bb2425a5"
#define Q38_KV_PER_TOKEN 28288ull

static bool parse_catalog(const char *path, pn_catalog_manifest *out)
{
    char err[256] = "", *json = NULL;
    long n = -1;
    FILE *f = fopen(path, "rb");
    bool ok = false;

    if (f && fseek(f, 0, SEEK_END) == 0 && (n = ftell(f)) > 0 && fseek(f, 0, SEEK_SET) == 0 &&
        (json = malloc((size_t)n + 1u)) && fread(json, 1u, (size_t)n, f) == (size_t)n) {
        json[n] = '\0';
        ok = pn_cat_manifest_parse(json, out, err, sizeof(err)) == 0;
        if (!ok)
            printf("  %s: %s\n", path, err);
    }
    if (f)
        fclose(f);
    free(json);
    return ok;
}

static const pn_cat_file *find_file(const pn_cat_model *m, const char *suffix, bool weights)
{
    for (int i = 0; m && i < m->n_files; i++) {
        const size_t n = strlen(m->files[i].filename), k = strlen(suffix);

        if ((strcmp(m->files[i].role, "weights") == 0) == weights && n >= k &&
            strcmp(m->files[i].filename + n - k, suffix) == 0)
            return &m->files[i];
    }
    return NULL;
}

/** `<stem>.<rung>.fttw`, rung non-empty with no '.' (gguf_fttw_overlay.c card_artifact_name). */
static bool card_artifact_name(const char *name, const char *stem)
{
    const size_t n = strlen(name), s = strlen(stem);

    return n > s + 1u + 5u && strncmp(name, stem, s) == 0 && name[s] == '.' &&
           strcmp(name + n - 5u, ".fttw") == 0 && !memchr(name + s + 1u, '.', n - 5u - s - 1u);
}

static void test_tier_profile(uint64_t priced_total, uint64_t priced_kv, uint64_t mll_freed,
                              const pn_qwen35_tt_bank_fit *flat)
{
    static pn_catalog_manifest tier, real;
    const pn_cat_model *tm, *dm, *rm;
    const pn_cat_profile *tp = NULL, *dp = NULL, *rp = NULL;
    const pn_cat_file *weights, *trunk, *mtp, *dmtp;
    const uint64_t card = 8ull * P150_BANK_BYTES;
    pn_load_option opts[PN_LOAD_OPTIONS_MAX];
    char stem[128] = "", want[160], what[240];
    uint64_t need = 0u, default_need;
    int n_artifacts = 0, n_opts;
    bool grid_has_tier = false;
    double diff_mib;

    check(parse_catalog(TIER_CATALOG, &tier) && parse_catalog(REAL_CATALOG, &real),
          "the tier test catalog and the real catalog parse");
    tm = pn_cat_find_model(&tier, TIER_MODEL);
    dm = pn_cat_find_model(&tier, DEFAULT_MODEL);
    rm = pn_cat_find_model(&real, DEFAULT_MODEL);
    if (tm) tp = pn_cat_find_profile(tm, TIER_PROFILE);
    if (dm) dp = pn_cat_find_profile(dm, DEFAULT_PROFILE);
    if (rm) rp = pn_cat_find_profile(rm, DEFAULT_PROFILE);
    check(tp && dp && rp, "the tier profile and the default profile are listed");
    if (!tp || !dp || !rp)
        return;

    /* The default arm is the shipped one, unchanged. */
    check(dm->n_files == rm->n_files && dp->options.mem_fixed == rp->options.mem_fixed &&
              dp->options.mem_kv_per_token == rp->options.mem_kv_per_token &&
              dp->options.ctx_max == rp->options.ctx_max && dp->ctx_cap == rp->ctx_cap &&
              strcmp(dp->options.kv_type, rp->options.kv_type) == 0,
          "the default model in the test catalog is catalog/catalog.json's");
    for (int i = 0; i < dm->n_files && i < rm->n_files; i++)
        check(strcmp(dm->files[i].sha256, rm->files[i].sha256) == 0 &&
                  strcmp(dm->files[i].filename, rm->files[i].filename) == 0,
              "a default file is catalog/catalog.json's (name and sha256)");

    /* The tier profile: 262,144, K8V4, prefix checkpoint (the TT default) + MTP k1. */
    check(tp->options.present && tp->options.ctx_max == (int)TIER_CTX &&
              tp->ctx_cap == (int)TIER_CTX && tp->max_sessions == 1,
          "tier profile: ctx_max and the load's ctx_cap are 262,144, one session");
    check(strcmp(tp->options.kv_type, "k8v4") == 0 && tp->mtp && tp->mtp_drafts == 1,
          "tier profile: K8V4 KV, MTP on with 1 draft (k1)");
    check(tp->options.mem_kv_per_token == Q38_KV_PER_TOKEN &&
              tp->options.mem_kv_per_token * TIER_CTX == priced_kv,
          "tier profile: 28,288 B a token, 262,144 x that = the budget's KV class");
    check(tp->options.mem_fixed + mll_freed == dp->options.mem_fixed,
          "tier profile: fixed bytes = the default's minus exactly the 8,556,380,160 M-LL frees");

    /* Its memory model at 262,144 against the budget (1 MiB a bank = 8 MiB card-wide). */
    need = pn_load_option_need(&tp->options, (int)TIER_CTX, 1);
    diff_mib = ((double)need - (double)priced_total) / MIB;
    snprintf(what, sizeof(what), "FIT LINE: catalog need at 262,144 %.3f GiB vs plan 1.1's card "
             "total (gates FLAT) %.3f GiB: %+.2f MiB, within 1 MiB a bank", need / GIB,
             priced_total / GIB, diff_mib);
    check(diff_mib >= -8.0 && diff_mib <= 8.0, what);
    snprintf(what, sizeof(what), "the catalog's average free a bank (%.1f MiB) is at least plan "
             "1.1's per-bank lower bound (%.1f MiB)", (double)(card - need) / 8.0 / MIB,
             flat->binding_free / MIB);
    check(need < card && (int64_t)((card - need) / 8u) >= flat->binding_free, what);
    check(need + PN_LOAD_FIT_MARGIN_BYTES <= card, "tier need + the load margin fits a p150");
    default_need = pn_load_option_need(&dp->options, (int)TIER_CTX, 1);
    check(default_need + PN_LOAD_FIT_MARGIN_BYTES > card,
          "RED: the default (Bfp8) memory model at 262,144 does not fit a p150");
    n_opts = pn_load_options_grid(tp, opts, PN_LOAD_OPTIONS_MAX);
    for (int i = 0; i < n_opts; i++)
        grid_has_tier |= opts[i].ctx_cap == (int)TIER_CTX && opts[i].need_per_unit == need;
    check(grid_has_tier, "the load picker offers 262,144 at that need");
    printf("  tier profile %s: need at 262,144 %.3f GiB (fixed %.3f + KV %.3f), plan 1.1 %.3f "
           "GiB (%+.2f MiB); card %.3f GiB; 1.1 lower bound %.1f MiB in every bank\n",
           TIER_PROFILE, need / GIB, tp->options.mem_fixed / GIB, priced_kv / GIB,
           priced_total / GIB, diff_mib, card / GIB, flat->binding_free / MIB);

    /* Files: same metadata GGUF and MTP bytes, the GPTQ trunk under companion names. */
    weights = find_file(tm, ".gguf", true);
    trunk = find_file(tm, ".bfp4ffn.fttw", false);
    mtp = find_file(tm, ".mtp.fttw", false);
    dmtp = find_file(dm, ".mtp.fttw", false);
    check(weights && trunk && mtp && dmtp, "tier files: metadata GGUF, trunk, MTP head");
    if (!weights || !trunk || !mtp || !dmtp)
        return;
    check(strcmp(weights->sha256, find_file(dm, ".gguf", true)->sha256) == 0 &&
              strcmp(mtp->sha256, dmtp->sha256) == 0 && mtp->size == dmtp->size,
          "tier files: the default's metadata GGUF and MTP head bytes");
    check(strcmp(trunk->sha256, TIER_ARTIFACT_SHA) == 0 && trunk->size == TIER_ARTIFACT_BYTES,
          "tier trunk: the plan 1.6 GPTQ artifact (sha256 e537e151..., 20,045,796,760 B)");
    snprintf(stem, sizeof(stem), "%.*s", (int)(strlen(weights->filename) - 5u),
             weights->filename);
    for (int i = 0; i < tm->n_files; i++)
        n_artifacts += card_artifact_name(tm->files[i].filename, stem);
    check(n_artifacts == 1 && card_artifact_name(trunk->filename, stem),
          "tier trunk is the one <stem>.<rung>.fttw beside the metadata GGUF");
    snprintf(want, sizeof(want), "%.*s.mtp.fttw", (int)(strlen(trunk->filename) - 5u),
             trunk->filename);
    check(strcmp(mtp->filename, want) == 0, "tier MTP head is named <trunk>.mtp.fttw");
}

static void test_tier(const pn_tt_mixw_fit *all8)
{
    static uint32_t tiles[PN_QWEN35_TT_MIXW_MAX_PLACEMENTS];
    const pn_qwen35_tt_fit_shape s30 = q38_fit_shape(ANCHOR_CTX, K8V4_K_TILE, K8V4_V_TILE);
    const uint64_t slot = pn_qwen35_tt_fit_snapshot_bytes(&s30);
    const uint64_t rows8 = DFLASH_VERIFY_ROWS * pn_qwen35_tt_fit_state_bytes(&s30);
    const uint64_t gate = pn_qwen35_tt_fit_gate_array_bytes(48u, 48u, GATE_ROWS_A_PASS) +
                          GATE_CONSTANT_BYTES;
    pn_tt_mixw_fit mll, mlle, mll_trunk;
    pn_qwen35_tt_bank_load anchor;
    tier_classes ca, ct, cb;
    tier_variant v[8];
    uint64_t joint_q;
    char reason[PN_TT_MIXW_REASON_BYTES] = "", what[200];
    size_t n;

    /* The two allocations main does not price, derived in qwen35moe_tt_fit_plan.c. */
    check(pn_qwen35_tt_fit_gate_array_bytes(48u, 48u, GATE_ROWS_A_PASS) == 75890688ull,
          "gate-block array: 48 layers x 2 x 64 bands x 12,352 B = 75,890,688 B, one bank");
    check(pn_qwen35_tt_fit_gate_array_bytes(48u, 48u, 0u) == 0u, "FLAT only: no array");
    check(pn_qwen35_tt_fit_gdnc_gate_bytes(48u) == 61440ull,
          "GDNC GATE at 48 heads: 2 slots x 48 x 160 fp32 = 61,440 B");
    check(slot == 339738624ull && pn_qwen35_tt_fit_state_bytes(&s30) == 150994944ull,
          "a snapshot set is 324 MiB; state alone 144 MiB");
    check(pn_qwen35_tt_fit_kv_bank_bytes(&s30, 17u, 8u, true) +
              pn_qwen35_tt_fit_kv_bank_bytes(&s30, 17u, 8u, false) ==
              28288ull * ANCHOR_CTX / 4u,
          "KV over the banks: 4 of each role = 28,288 B a token (spec §3)");

    test_anchor_census(all8, &ca);

    /* M-LL and M-LL-e on the trunk + MTP census (Bfp8 elsewhere). */
    build(&g_census, true, false);
    n = map_mll(&g_census, false);
    check(n == 192u && price(&g_census, n, &mll, tiles, reason) == 0 &&
              mll.bfp4_placements == 192u, "M-LL: 192 FFN matrices at bfp4_b price");
    snprintf(what, sizeof(what), "M-LL frees 7.969 GiB, %.1f MiB in every bank (spec: 7.968, "
             "1,020)", (all8->bank_bytes[0] - mll.bank_bytes[0]) / MIB);
    check(all8->card_bytes - mll.card_bytes == 8556380160ull &&
              least_freed(all8, &mll) == 8556380160ull / 8u, what);
    joint_q = trunk_q_bytes(&g_census, tiles);
    test_block1_census(all8, &mll, joint_q, &ca, &cb);
    n = map_mll(&g_census, true);
    check(n == 180u && price(&g_census, n, &mlle, NULL, reason) == 0,
          "M-LL-e: 180 FFN matrices at bfp4_b (layers 0, 1, 62, 63 kept) price");
    build(&g_census_b, false, false);
    n = map_mll(&g_census_b, false);
    check(price(&g_census_b, n, &mll_trunk, NULL, reason) == 0, "M-LL without MTP prices");

    tier_load(&anchor, all8, 0u, ANCHOR_CTX, K8V4_K_TILE, K8V4_V_TILE, Q38_KV_LAYERS, slot, 0u);
    v[0].label = "anchor against itself (30,720 Bfp8 ckpt k1)";
    v[0].load = anchor;
    v[1].label = "no map 262,144 ckpt k1 + gate array (RED)";
    tier_load(&v[1].load, all8, 0u, TIER_CTX, K8V4_K_TILE, K8V4_V_TILE, Q38_KV_LAYERS, slot,
              gate);
    v[2].label = "M-LL 262,144 ckpt k1, gates FLAT";
    tier_load(&v[2].load, &mll, 0u, TIER_CTX, K8V4_K_TILE, K8V4_V_TILE, Q38_KV_LAYERS, slot, 0u);
    v[3].label = "M-LL 262,144 ckpt k1 + gate array";
    tier_load(&v[3].load, &mll, 0u, TIER_CTX, K8V4_K_TILE, K8V4_V_TILE, Q38_KV_LAYERS, slot,
              gate);
    v[4].label = "M-LL-e 262,144 ckpt k1, gates FLAT";
    tier_load(&v[4].load, &mlle, 0u, TIER_CTX, K8V4_K_TILE, K8V4_V_TILE, Q38_KV_LAYERS, slot,
              0u);
    v[5].label = "M-LL-e 262,144 ckpt k1 + gate array";
    tier_load(&v[5].load, &mlle, 0u, TIER_CTX, K8V4_K_TILE, K8V4_V_TILE, Q38_KV_LAYERS, slot,
              gate);
    v[6].label = "M-LL DFlash2 recompute-commit + gate array";
    tier_load(&v[6].load, &mll_trunk, DFLASH_DRAFTER_BYTES, TIER_CTX, K8V4_K_TILE, K8V4_V_TILE,
              Q38_DFLASH_KV_LAYERS, 0u, gate);
    v[7].label = "M-LL DFlash2 8 fp32 rows + gate array";
    tier_load(&v[7].load, &mll_trunk, DFLASH_DRAFTER_BYTES, TIER_CTX, K8V4_K_TILE, K8V4_V_TILE,
              Q38_DFLASH_KV_LAYERS, rows8, gate);
    printf("  bank budget vs the anchor (fullest bank %.1f MiB free, low end), per bank:\n",
           ANCHOR_FULLEST_FREE / MIB);
    for (size_t i = 0; i < sizeof(v) / sizeof(v[0]); i++)
        price_variant(&v[i], &anchor);

    check(v[0].fit.binding_free == (int64_t)ANCHOR_FULLEST_FREE && v[0].fit.kv_growth == 0 &&
              !v[0].fit.fits, "the anchor priced against itself is the anchor (no 128 MiB)");
    check(!v[1].fit.fits && v[1].fit.binding_free < 0,
          "RED: no map at 262,144 + ckpt + k1 is refused (fullest bank negative)");
    /* Hand-worked: 0.0025 GiB + 1,069,547,520 freed - 231,424 x 3,600 B (a 9 K + 8 V bank) -
     * the array and its constants. */
    check(v[3].fit.binding_free == (int64_t)ANCHOR_FULLEST_FREE + 1069547520 -
                                       231424ll * 3600ll - (int64_t)gate,
          "M-LL + gate array: the hand-worked fullest bank");
    snprintf(what, sizeof(what), "GATE: M-LL 262,144 + ckpt + k1 + gate array keeps >= 128 MiB "
             "in its fullest bank (%.1f)", v[3].fit.binding_free / MIB);
    check(v[3].fit.fits && v[2].fit.fits, what);
    check(v[2].fit.binding_free - v[3].fit.binding_free == (int64_t)gate,
          "the gate array costs its bytes in the fullest bank");

    /* Main's placement (block 1), priced the same way: its extra classes as even classes. Not a
     * strict per-bank bound (the second attn_q follows the weight rotation, not an even split),
     * so it is checked against the measured fullest bank as an estimate, not to 1 MiB. */
    {
        const uint64_t extra = cb.second_q + cb.conv_own + (cb.tall - ct_flat_tall(&mll));
        tier_variant m[2];

        m[0].label = "M-LL 262,144 main's placement, at load";
        tier_load(&m[0].load, &mll, 0u, TIER_CTX, K8V4_K_TILE, K8V4_V_TILE, Q38_KV_LAYERS, 0u, 0u);
        m[0].load.even_bytes += extra;
        m[1] = m[0];
        m[1].label = "M-LL 262,144 main's placement, first verify";
        m[1].load.even_bytes += slot;
        price_variant(&m[0], &anchor);
        price_variant(&m[1], &anchor);
        snprintf(what, sizeof(what), "main's placement at load: %.1f MiB priced (even split), "
                 "%.1f measured in the fullest bank (block 1, low end); within 16 MiB",
                 m[0].fit.binding_free / MIB, BLOCK1_FULLEST_FREE / MIB);
        check(llabs(m[0].fit.binding_free - (int64_t)BLOCK1_FULLEST_FREE) <= 16ll * 1048576ll,
              what);
        snprintf(what, sizeof(what), "RED for the 128 MiB rule: main's placement after the first "
                 "verify leaves %.1f MiB (priced) / ~%.1f (measured less the slot's share)",
                 m[1].fit.binding_free / MIB, (BLOCK1_FULLEST_FREE - slot / 8u) / MIB);
        check(!m[1].fit.fits && BLOCK1_FULLEST_FREE - slot / 8u < PN_QWEN35_TT_TIER_HEADROOM,
              what);
    }

    /* bf16 KV on the M-LL tier, ckpt + k1, with and without the gate array. */
    {
        const uint32_t bf16_gate = tier_max_ctx(&v[3], &anchor, 0u, 0u);
        const uint32_t bf16_flat = tier_max_ctx(&v[2], &anchor, 0u, 0u);
        const uint32_t k8v4_gate = tier_max_ctx(&v[3], &anchor, K8V4_K_TILE, K8V4_V_TILE);

        printf("  largest context with >= 128 MiB a bank on M-LL + ckpt + k1: bf16 KV %u (gate "
               "array) / %u (FLAT); K8V4 %u (gate array)\n", bf16_gate, bf16_flat, k8v4_gate);
        check(bf16_gate > ANCHOR_CTX && bf16_gate < TIER_CTX && bf16_flat >= bf16_gate,
              "bf16 KV fits the tier above 30,720 and below 262,144");
        check(k8v4_gate == TIER_CTX, "K8V4 reaches 262,144 on the tier");
    }

    /* Spec §3's card-wide table from the classes (the tier keeps the anchor's small arms). */
    tier_price_classes(&v[3].load.shape, &mll, 0u, Q38_KV_LAYERS, slot, gate, &ct);
    memcpy(ct.small, ca.small, sizeof(ct.small));
    print_class_table(&ca, &ct, "M-LL 262,144 + gate array");
    tier_price_classes(&v[6].load.shape, &mll_trunk, DFLASH_DRAFTER_BYTES, Q38_DFLASH_KV_LAYERS,
                       0u, gate, &ct);
    memcpy(ct.small, ca.small, sizeof(ct.small));
    print_class_table(&ca, &ct, "DFlash2 recompute");

    /* The profile loads with gates FLAT (no prefill gate array): price that variant. */
    tier_price_classes(&v[2].load.shape, &mll, 0u, Q38_KV_LAYERS, slot, 0u, &ct);
    memcpy(ct.small, ca.small, sizeof(ct.small));
    test_tier_profile(tier_total(&ct), ct.kv, all8->card_bytes - mll.card_bytes, &v[2].fit);
}

int main(void)
{
    static pn_tt_mixw_fit trunk8, all8;

    test_bfp8_census(&trunk8);
    test_uniform_bfp4(&trunk8, &all8);
    test_mtp_subset(&trunk8, &all8);
    test_fit_floor();
    test_refusals();
    print_groups();
    test_tier(&all8);
    printf("test_tt_q38_mixw_fit: %d checks, %d failed\n", g_checks, g_failures);
    return g_failures ? 1 : 0;
}
