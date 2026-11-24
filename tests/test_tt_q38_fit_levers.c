/* SPDX-License-Identifier: AGPL-3.0-or-later */
/* Copyright (C) 2026 David (Corey) Herrera */
/**
 * @file test_tt_q38_fit_levers.c
 * @brief The fit policy's levers (qwen35moe_tt_fit_plan.h): GDNC conv copies, tall bands and
 *        unmirrored gated strips, picked at load so every DRAM bank keeps 128 MiB (#148 rule).
 *
 * Opens no device. It drives the production picker with Qwen3.8's shapes and the decide's real
 * starting state (8 banks less the open's scratch rows in banks 0 and 1):
 *   1. HAND-WORKED: the policy bank floor against main's (option A's second attn_q, D's conv
 *      weight, C's tall attn_qg and the compact verify map off; the checkpoint's set on).
 *   2. CALIBRATION: the estimate sits 10-40 MiB under every measured fullest bank under the
 *      policy (p150): the 30,720 default (3 MiB, checkpoint in, k1 serve), the
 *      262,144 tier (243 MiB, fitbanks block), and the lever loads of the q8k
 *      block (3 / 4 + V copies at 30,720; 16 and 32 single bands at 102,400).
 *   3. TODAY'S DEFAULT: 30,720 without levers is short of the rule (RED for today's fit); the
 *      picker takes conv copies only and keeps the 64-band tall set.
 *   4. LONG CONTEXT: the ladder at 65,536 and 131,072, pins, and loads with room keeping main's
 *      levers.
 *   5. THE LONG PROFILE (catalog/catalog.json `p150-one-100k`): ctx 102,400 fits with 1 copy, 32
 *      single bands and the draft in place; its fixed bytes are the default's less exactly what
 *      those levers free; the default's memory model at 102,400 does not fit a p150 (red).
 *   6. LEVER K (the long-context KV type, k4h: Bfp4 K + V): only when the load's own type fails
 *      the whole ladder; then the whole ladder again at its tiles, fewest other levers first.
 *
 * Build: `make tests/test_tt_q38_fit_levers` (any host).
 *
 * @status None
 * @issues None
 * @todo None
 */

#include <stdbool.h>
#include <stdint.h>
#include <stdio.h>
#include <string.h>

#include <stdlib.h>

#include "../src/local/catalog_manifest.h"
#include "../src/local/load_options.h"
#include "../src/models/qwen35moe_tt_fit_plan.h"

#define GIB (1024.0 * 1024.0 * 1024.0)
#define MIB (1024.0 * 1024.0)
#define P150_BANKS 8u
#define P150_BANK_BYTES 4278190080ull
#define SCRATCH_BANK0 (544ull * 2048ull)
#define SCRATCH_BANK1 (7760ull * 2048ull)

/* Qwen3.8-27B (test_tt_q38_fit_banks.c): the Bfp8 trunk, the joint attn_q, M-LL's FFN saving,
 * the dense MTP head as fit_shape_mtp prices it. */
#define Q38_TRUNK_BYTES 27231682560ull
#define Q38_JOINT_Q_BYTES 1069547520ull
#define Q38_MLL_FREED 8556380160ull
#define Q38_MTP_DRAFT (2057ull * 160ull * 1088ull) /* the staged draft-vocabulary copy */
#define Q38_MTP_HEAD (451323072ull + Q38_MTP_DRAFT)
#define Q38_SNAPSHOT_SET 339738624ull /* 48 x (3 MiB state + 3.75 MiB window) */
#define Q38_CONV_WEIGHTS (48ull * 320u * 4u * 4096u) /* 48 x 5 MiB */
#define Q38_TALL_QG (2ull * 192u * 64u * 2048u)      /* joint attn_qg at 64 bands */
#define K8V4_K 1088u
#define K8V4_V 576u
#define K4H_K 576u /* k4h: Bfp4_b K (rotated) and V */
#define K4H_V 576u
#define VERIFY_ROWS_K1 2u

/* The measured fullest banks under the fit policy (p150) after "reserve: prefix checkpoint",
 * each at the low end of its 3-decimal GiB print. */
#define DEFAULT_30720_FULLEST 2684354ull        /* 8 copies: "0.003 GiB free" (p7_2 a.k1) */
#define DEFAULT_30720_R6_FULLEST 66035122ull    /* 6 copies: "0.062 GiB free" (q38xl k1r6) */
#define TIER_262144_FULLEST (243ull * 1048576ull) /* fitbanks: 243 MiB */
/* With the levers, the exact gate tables armed: after "arm: exact gate tables", the state the
 * bank rule checks (card 2026-11-19). */
#define GIB_LOW(milli) ((uint64_t)(((milli) - 0.5) / 1000.0 * GIB))
#define LEVERS_30720_R3_FULLEST GIB_LOW(149.0)      /* mine30: 3 copies, 64 mirrored */
#define LEVERS_30720_R4V_FULLEST GIB_LOW(161.0)     /* v1: 4 copies, draft in place */
#define LEVERS_102400_16S_FULLEST GIB_LOW(180.0)    /* mineL: 1 copy, 16 single, draft in place */
#define LEVERS_102400_32S_FULLEST GIB_LOW(148.0)    /* b32: 1 copy, 32 single, draft in place */

static int g_checks, g_failures;

static void check(bool ok, const char *what)
{
    g_checks++;
    if (!ok) {
        g_failures++;
        printf("FAIL %s\n", what);
    }
}

static void decide_free(uint64_t *free_bytes)
{
    for (uint32_t b = 0; b < P150_BANKS; b++)
        free_bytes[b] = P150_BANK_BYTES;
    free_bytes[0] -= SCRATCH_BANK0;
    free_bytes[1] -= SCRATCH_BANK1;
}

/** Qwen3.8-27B at `ctx`, K8V4, MTP k1, main's levers (8 copies, 64 bands, mirrored gated). */
static pn_qwen35_tt_fit_shape q38(uint32_t ctx, bool tables)
{
    pn_qwen35_tt_fit_shape s;

    memset(&s, 0, sizeof(s));
    s.n_embd = 5120; s.ffn = 17408; s.n_head = 24; s.n_head_kv = 4; s.head_dim = 256;
    s.n_attn = 16; s.n_ssm = 48; s.d_inner = 6144; s.v_heads = 48; s.conv_dim = 10240;
    s.conv_taps = 4; s.ssm_state = 128; s.ctx = ctx;
    s.kv_k_tile_bytes = K8V4_K; s.kv_v_tile_bytes = K8V4_V;
    s.tall_rows = 64u; s.mirrors = P150_BANKS; s.outproj_mirrors = true;
    s.gdnc_replicas = 8u; s.attn_on_card = true; s.gdn_on_card = true; s.state_split = true;
    s.mtp_bytes = Q38_MTP_HEAD; s.mtp_rows = VERIFY_ROWS_K1; s.mtp_draft_bytes = Q38_MTP_DRAFT;
    s.gate_tables = tables;
    return s;
}

static int64_t low_of(const pn_qwen35_tt_fit_shape *s, uint64_t weights)
{
    uint64_t free_bytes[P150_BANKS];

    decide_free(free_bytes);
    return pn_qwen35_tt_fit_policy_bank_low(s, weights, Q38_JOINT_Q_BYTES, true, free_bytes,
                                            P150_BANKS);
}

static bool pick(const pn_qwen35_tt_fit_shape *s, const pn_qwen35_tt_fit_pins *pins,
                 pn_qwen35_tt_fit_levers *out, int64_t *low)
{
    uint64_t free_bytes[P150_BANKS];

    decide_free(free_bytes);
    return pn_qwen35_tt_fit_levers_pick(s, Q38_TRUNK_BYTES, Q38_JOINT_Q_BYTES, true, free_bytes,
                                        P150_BANKS, pins, out, low);
}

/* ================================================================
 * (1) Hand-worked
 * ================================================================ */

static void test_hand_worked(void)
{
    const pn_qwen35_tt_fit_shape s = q38(30720u, false);
    pn_qwen35_tt_fit_shape r1 = s;
    pn_qwen35_tt_fit_floor floor, floor_r1;
    uint64_t main_bank, policy_bank;
    char what[200];

    pn_qwen35_tt_fit_floor_compute(&s, Q38_TRUNK_BYTES, Q38_JOINT_Q_BYTES, &floor);
    main_bank = pn_qwen35_tt_fit_bank_floor(&s, &floor, true, P150_BANKS);
    policy_bank = pn_qwen35_tt_fit_policy_bank_floor(&s, &floor, true, P150_BANKS);
    /* Main's even part less A (1,069,547,520), D (251,658,240), C (50,331,648) and one compact
     * verify set (339,738,624): 1,711,276,032 B, an exact multiple of 8. */
    snprintf(what, sizeof(what), "policy bank floor = main's - (A + D + C + one verify set) / 8 "
             "= main's - %llu B", (unsigned long long)(main_bank - policy_bank));
    check(main_bank - policy_bank ==
              (Q38_JOINT_Q_BYTES + Q38_CONV_WEIGHTS + Q38_TALL_QG + Q38_SNAPSHOT_SET) / 8u,
          what);
    check(pn_qwen35_tt_fit_policy_bank_floor(&s, &floor, true, P150_BANKS) -
                  pn_qwen35_tt_fit_policy_bank_floor(&s, &floor, false, P150_BANKS) ==
              Q38_SNAPSHOT_SET / 8u,
          "the checkpoint costs its set's even share");
    r1.gdnc_replicas = 1u;
    pn_qwen35_tt_fit_floor_compute(&r1, Q38_TRUNK_BYTES, Q38_JOINT_Q_BYTES, &floor_r1);
    check(policy_bank - pn_qwen35_tt_fit_policy_bank_floor(&r1, &floor_r1, true, P150_BANKS) ==
              7u * Q38_CONV_WEIGHTS / 8u,
          "E 8 -> 1 frees seven conv-constant copies' even share (210 MiB a bank)");
    check(pn_qwen35_tt_fit_policy_bank_floor(&s, &floor, true, 0u) == 0u &&
              low_of(&s, Q38_TRUNK_BYTES) != INT64_MIN,
          "no banks: no floor");
}

/* ================================================================
 * (2) Calibration against the measured fullest banks
 * ================================================================ */

/** The estimate sits under the measurement, by at least 10 MiB and at most 40. */
static void calibrated(const char *name, int64_t estimate, uint64_t measured)
{
    char what[200];

    printf("  %s: estimate %.1f MiB, measured %.1f\n", name, estimate / MIB, measured / MIB);
    snprintf(what, sizeof(what), "%s: estimate %.1f MiB is 10-40 MiB under the measured %.1f",
             name, estimate / MIB, measured / MIB);
    check(estimate <= (int64_t)measured - 10ll * 1048576ll &&
              estimate >= (int64_t)measured - 40ll * 1048576ll, what);
}

static void test_calibration(void)
{
    const pn_qwen35_tt_fit_shape d = q38(30720u, false), t = q38(262144u, false);
    pn_qwen35_tt_fit_shape r6 = d;

    r6.gdnc_replicas = 6u;
    calibrated("30,720 default, 8 copies", low_of(&d, Q38_TRUNK_BYTES), DEFAULT_30720_FULLEST);
    calibrated("30,720 default, 6 copies", low_of(&r6, Q38_TRUNK_BYTES), DEFAULT_30720_R6_FULLEST);
    calibrated("tier 262,144", low_of(&t, Q38_TRUNK_BYTES - Q38_MLL_FREED), TIER_262144_FULLEST);
    {
        pn_qwen35_tt_fit_shape a = q38(30720u, true), b = a, c = q38(102400u, true), e;

        a.gdnc_replicas = 3u;
        calibrated("30,720, 3 copies", low_of(&a, Q38_TRUNK_BYTES), LEVERS_30720_R3_FULLEST);
        b.gdnc_replicas = 4u; b.draft_view = true;
        calibrated("30,720, 4 copies + V", low_of(&b, Q38_TRUNK_BYTES), LEVERS_30720_R4V_FULLEST);
        c.gdnc_replicas = 1u; c.draft_view = true; c.outproj_mirrors = false; e = c;
        c.tall_rows = 16u;
        calibrated("102,400, 1 copy, 16 single, V", low_of(&c, Q38_TRUNK_BYTES),
                   LEVERS_102400_16S_FULLEST);
        e.tall_rows = 32u;
        calibrated("102,400, 1 copy, 32 single, V", low_of(&e, Q38_TRUNK_BYTES),
                   LEVERS_102400_32S_FULLEST);
    }
}

/* ================================================================
 * (3) Today's default at 30,720
 * ================================================================ */

static void test_default(void)
{
    const pn_qwen35_tt_fit_shape s = q38(30720u, true);
    pn_qwen35_tt_fit_levers lv;
    int64_t low = 0;
    bool fits;
    char what[200];

    check(low_of(&s, Q38_TRUNK_BYTES) < (int64_t)PN_QWEN35_TT_TIER_HEADROOM,
          "RED for today's fit: the 30,720 default with main's levers is under 128 MiB a bank");
    fits = pick(&s, NULL, &lv, &low);
    printf("  30,720 + tables: copies %u, bands %u, gated %s, ~%.1f MiB\n", lv.gdnc_replicas,
           lv.tall_rows, lv.outproj_mirrors ? "mirrored" : "single", low / MIB);
    snprintf(what, sizeof(what), "the default fits with copies alone: 8 -> %u, 64 bands, gated "
             "mirrored, the draft copy kept (~%.1f MiB)", lv.gdnc_replicas, low / MIB);
    check(fits && lv.gdnc_replicas == 3u && lv.tall_rows == 64u && lv.outproj_mirrors &&
              !lv.draft_view && low >= (int64_t)PN_QWEN35_TT_TIER_HEADROOM, what);
    /* One copy more would not: the picker took the fewest levers. */
    {
        pn_qwen35_tt_fit_shape more = s;

        more.gdnc_replicas = lv.gdnc_replicas + 1u;
        check(low_of(&more, Q38_TRUNK_BYTES) < (int64_t)PN_QWEN35_TT_TIER_HEADROOM,
              "one copy more is short: the least lever that fits");
    }
}

/* ================================================================
 * (4) Long context, pins, room
 * ================================================================ */

static void test_long(void)
{
    pn_qwen35_tt_fit_shape s = q38(65536u, true);
    pn_qwen35_tt_fit_pins pins = { false, false, false, false, 0u, 0u };
    pn_qwen35_tt_fit_levers lv;
    int64_t low = 0;
    bool fits;
    char what[220];

    fits = pick(&s, NULL, &lv, &low);
    printf("  65,536: copies %u, bands %u, gated %s, draft %s, ~%.1f MiB\n", lv.gdnc_replicas,
           lv.tall_rows, lv.outproj_mirrors ? "mirrored" : "single",
           lv.draft_view ? "in place" : "staged", low / MIB);
    snprintf(what, sizeof(what), "65,536: 64 bands do not fit even with one copy and the draft "
             "in place; 32 mirrored bands with 2 copies and the draft copy do (copies %u, bands "
             "%u)", lv.gdnc_replicas, lv.tall_rows);
    check(fits && lv.gdnc_replicas == 2u && lv.tall_rows == 32u && lv.outproj_mirrors &&
              !lv.draft_view, what);

    s = q38(131072u, true);
    fits = pick(&s, NULL, &lv, &low);
    printf("  131,072: copies %u, bands %u, gated %s, ~%.1f MiB (%s)\n", lv.gdnc_replicas,
           lv.tall_rows, lv.outproj_mirrors ? "mirrored" : "single", low / MIB,
           fits ? "fits" : "SHORT");
    snprintf(what, sizeof(what), "131,072: every lever (1 copy, 16 single bands, draft in place) "
             "is still short (~%.1f MiB) and the picker says so", low / MIB);
    check(!fits && lv.gdnc_replicas == 1u && lv.tall_rows == 16u && !lv.outproj_mirrors &&
              lv.draft_view && low < (int64_t)PN_QWEN35_TT_TIER_HEADROOM, what);

    /* Pins: the operator's copy count stays, the ladder moves to the tall steps. */
    s = q38(30720u, true);
    s.gdnc_replicas = 8u;
    pins.gdnc_replicas = true;
    fits = pick(&s, &pins, &lv, &low);
    snprintf(what, sizeof(what), "copies pinned at 8: the tall steps instead, to 32 single bands "
             "(bands %u, %s, draft %s)", lv.tall_rows,
             lv.outproj_mirrors ? "mirrored" : "single", lv.draft_view ? "in place" : "staged");
    check(fits && lv.gdnc_replicas == 8u && lv.tall_rows == 32u && !lv.outproj_mirrors &&
              !lv.draft_view, what);
    s.ctx = 65536u;
    pins.draft_view = true;
    fits = pick(&s, &pins, &lv, &low);
    check(!lv.draft_view && lv.gdnc_replicas == 8u && lv.tall_rows == 16u && !fits,
          "65,536, copies and the draft copy pinned: the tall set takes every step, still short");
    s.ctx = 30720u;
    pins.draft_view = false;
    pins.gdnc_replicas = false;
    pins.tall_rows = true;
    s.tall_rows = 0u;
    fits = pick(&s, &pins, &lv, &low);
    check(lv.tall_rows == 0u && lv.gdnc_replicas <= 8u && fits,
          "tall set pinned off: no tall step, copies only");

    /* Room: the 4-bit tier's weights at 30,720 keep main's levers. */
    s = q38(30720u, true);
    {
        uint64_t free_bytes[P150_BANKS];

        decide_free(free_bytes);
        fits = pn_qwen35_tt_fit_levers_pick(&s, Q38_TRUNK_BYTES - Q38_MLL_FREED,
                                            Q38_JOINT_Q_BYTES, true, free_bytes, P150_BANKS,
                                            NULL, &lv, &low);
    }
    check(fits && lv.gdnc_replicas == 8u && lv.tall_rows == 64u && lv.outproj_mirrors,
          "a load with room keeps 8 copies and the 64-band mirrored set");
    check(!pn_qwen35_tt_fit_levers_pick(&s, Q38_TRUNK_BYTES, Q38_JOINT_Q_BYTES, true, NULL,
                                        P150_BANKS, NULL, &lv, &low) && low == INT64_MIN,
          "no bank state: nothing picked");
}

/* ================================================================
 * (5) The long profile in catalog/catalog.json
 * ================================================================ */

#define LONG_CTX 102400u
#define CATALOG "catalog/catalog.json"

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

/** The policy's tall set in bytes (option C's attn_qg off). */
static uint64_t policy_tall(const pn_qwen35_tt_fit_shape *s)
{
    return (pn_qwen35_tt_fit_tall_tiles(s) - 2u * 192u) * s->tall_rows * 2048u;
}

static void test_long_profile(void)
{
    static pn_catalog_manifest cat;
    const pn_cat_model *m;
    const pn_cat_profile *dp = NULL, *lp = NULL;
    pn_qwen35_tt_fit_shape main_levers = q38(LONG_CTX, true), picked;
    pn_qwen35_tt_fit_levers lv;
    const uint64_t card = 8ull * P150_BANK_BYTES;
    uint64_t freed, need, default_need;
    pn_load_option opts[PN_LOAD_OPTIONS_MAX];
    int n_opts;
    bool offered = false;
    int64_t low = 0;
    char what[240];

    check(parse_catalog(CATALOG, &cat), "catalog/catalog.json parses");
    m = pn_cat_find_model(&cat, "qwen3.8-27b-q8");
    if (m) {
        dp = pn_cat_find_profile(m, "p150-one");
        lp = pn_cat_find_profile(m, "p150-one-100k");
    }
    check(dp && lp, "qwen3.8-27b-q8 lists p150-one and p150-one-100k");
    if (!dp || !lp)
        return;
    check(lp->options.present && lp->ctx_cap == (int)LONG_CTX && lp->options.ctx_max == (int)LONG_CTX &&
              lp->max_sessions == 1 && strcmp(lp->options.kv_type, "k8v4") == 0 && lp->mtp &&
              lp->mtp_drafts == 1 && lp->options.mem_kv_per_token == dp->options.mem_kv_per_token,
          "p150-one-100k: ctx 102,400, one session, K8V4, MTP k1, the default's KV a token");
    check(pick(&main_levers, NULL, &lv, &low) && lv.gdnc_replicas == 1u && lv.tall_rows == 32u &&
              !lv.outproj_mirrors && lv.draft_view,
          "102,400 fits with 1 copy, 32 single bands, the draft in place (16 bands not needed)");
    printf("  102,400: copies %u, bands %u, gated %s, draft %s, ~%.1f MiB\n", lv.gdnc_replicas,
           lv.tall_rows, lv.outproj_mirrors ? "mirrored" : "single",
           lv.draft_view ? "in place" : "staged", low / MIB);
    picked = main_levers;
    picked.gdnc_replicas = lv.gdnc_replicas;
    picked.tall_rows = lv.tall_rows;
    picked.outproj_mirrors = lv.outproj_mirrors;
    freed = (uint64_t)(main_levers.gdnc_replicas - picked.gdnc_replicas) * Q38_CONV_WEIGHTS +
            policy_tall(&main_levers) - policy_tall(&picked) + Q38_MTP_DRAFT;
    snprintf(what, sizeof(what), "p150-one-100k fixed = the default's %llu - the levers' %llu B",
             (unsigned long long)dp->options.mem_fixed, (unsigned long long)freed);
    check(lp->options.mem_fixed + freed == dp->options.mem_fixed, what);
    need = pn_load_option_need(&lp->options, (int)LONG_CTX, 1);
    default_need = pn_load_option_need(&dp->options, (int)LONG_CTX, 1);
    printf("  p150-one-100k: need at 102,400 %.3f GiB (fixed %.3f), card %.3f; the default's "
           "model %.3f\n", need / GIB, lp->options.mem_fixed / GIB, card / GIB, default_need / GIB);
    check(need + PN_LOAD_FIT_MARGIN_BYTES <= card, "its need at 102,400 + the load margin fits a p150");
    check(default_need + PN_LOAD_FIT_MARGIN_BYTES > card,
          "RED: the default profile's memory model at 102,400 does not fit a p150");
    n_opts = pn_load_options_grid(lp, opts, PN_LOAD_OPTIONS_MAX);
    for (int i = 0; i < n_opts; i++)
        offered |= opts[i].ctx_cap == (int)LONG_CTX && opts[i].need_per_unit == need;
    check(offered, "the load picker offers 102,400 at that need");
}

/* ================================================================
 * (6) Lever K: the long-context KV type
 * ================================================================ */

static void test_kv_long(void)
{
    pn_qwen35_tt_fit_shape s = q38(131072u, true);
    pn_qwen35_tt_fit_pins pins = { false, false, false, false, K4H_K, K4H_V };
    pn_qwen35_tt_fit_levers lv, base;
    int64_t low = 0, low_base = 0;
    bool fits;
    char what[240];

    fits = pick(&s, &pins, &lv, &low);
    printf("  131,072 + k4h: copies %u, bands %u, gated %s, draft %s, KV %s, ~%.1f MiB (%s)\n",
           lv.gdnc_replicas, lv.tall_rows, lv.outproj_mirrors ? "mirrored" : "single",
           lv.draft_view ? "in place" : "staged", lv.kv_long ? "k4h" : "k8v4", low / MIB,
           fits ? "fits" : "SHORT");
    snprintf(what, sizeof(what), "131,072: K8V4 is short with every lever, k4h fits (copies %u, "
             "bands %u, %s, draft %s, ~%.1f MiB)", lv.gdnc_replicas, lv.tall_rows,
             lv.outproj_mirrors ? "mirrored" : "single", lv.draft_view ? "in place" : "staged",
             low / MIB);
    check(fits && lv.kv_long && low >= (int64_t)PN_QWEN35_TT_TIER_HEADROOM, what);
    /* Fewest other levers: the same step at the load's own type is short, and the ladder step
     * before it at k4h tiles is short too (copies one more, same tall step and draft). */
    {
        pn_qwen35_tt_fit_shape t = s;

        t.gdnc_replicas = lv.gdnc_replicas;
        t.tall_rows = lv.tall_rows;
        t.outproj_mirrors = lv.outproj_mirrors;
        t.draft_view = lv.draft_view;
        check(low_of(&t, Q38_TRUNK_BYTES) < (int64_t)PN_QWEN35_TT_TIER_HEADROOM,
              "131,072: the picked levers at K8V4 tiles are short (K is needed)");
        t.kv_k_tile_bytes = K4H_K;
        t.kv_v_tile_bytes = K4H_V;
        check(low_of(&t, Q38_TRUNK_BYTES) == low, "the picked estimate is the k4h shape's");
        if (lv.gdnc_replicas < s.gdnc_replicas) {
            t.gdnc_replicas = lv.gdnc_replicas + 1u;
            check(low_of(&t, Q38_TRUNK_BYTES) < (int64_t)PN_QWEN35_TT_TIER_HEADROOM,
                  "131,072 + k4h: one more conv copy at the picked step is short (fewest levers)");
        }
    }

    /* Only when needed: a load its own type fits never takes K. */
    s = q38(30720u, true);
    fits = pick(&s, &pins, &lv, &low);
    (void)pick(&s, NULL, &base, &low_base);
    check(fits && !lv.kv_long && lv.gdnc_replicas == base.gdnc_replicas &&
              lv.tall_rows == base.tall_rows && low == low_base,
          "30,720 keeps K8V4: the long type changes nothing where the ladder fits");
    s = q38(102400u, true);
    fits = pick(&s, &pins, &lv, &low);
    snprintf(what, sizeof(what), "102,400 (p150-one-100k) fits at K8V4 with the levers: no K "
             "(%s)", lv.kv_long ? "took K" : "K8V4");
    check(fits && !lv.kv_long, what);

    /* No long type, or one not smaller than the load's: K is never a trade for nothing. */
    s = q38(131072u, true);
    pins.kv_long_k_tile = pins.kv_long_v_tile = 0u;
    fits = pick(&s, &pins, &lv, &low);
    check(!fits && !lv.kv_long, "131,072 without a long type: short, K8V4, as before");
    pins.kv_long_k_tile = K8V4_K;
    pins.kv_long_v_tile = K8V4_V;
    fits = pick(&s, &pins, &lv, &low);
    check(!fits && !lv.kv_long, "a long type no smaller than the load's own is not taken");
}

int main(void)
{
    printf("test_tt_q38_fit_levers: the fit policy's levers\n");
    printf("(1) hand-worked\n");
    test_hand_worked();
    printf("(2) calibration\n");
    test_calibration();
    printf("(3) today's default at 30,720\n");
    test_default();
    printf("(4) long context, pins, room\n");
    test_long();
    printf("(5) the long profile\n");
    test_long_profile();
    printf("(6) lever K: the long-context KV type\n");
    test_kv_long();
    printf("%d checks, %d failures\n", g_checks, g_failures);
    return g_failures ? 1 : 0;
}
