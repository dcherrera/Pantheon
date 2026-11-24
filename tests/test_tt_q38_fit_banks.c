/* SPDX-License-Identifier: AGPL-3.0-or-later */
/* Copyright (C) 2026 David (Corey) Herrera */
/**
 * @file test_tt_q38_fit_banks.c
 * @brief The fit policy's bank rule (qwen35moe_tt_fit_plan.h): the policy also engages when
 *        main's placement would leave a DRAM bank under 128 MiB, verify slots and the prefix
 *        checkpoint included (David's card 20261118-1010).
 *
 * Opens no device. It drives the production trigger with Qwen3.8's shapes and the decide's real
 * starting state (8 banks less the open's scratch rows in banks 0 and 1):
 *   1. HAND-WORKED: the bank floor (KV at its larger bank role, the rest even, the checkpoint's
 *      set), the fullest bank, and the bank the decide names; a bank short on its own engages
 *      the policy with the card-wide slack untouched.
 *   2. THE TIER (the 256k tier run, block 1, p150): GPTQ M-LL at 262,144, K8V4, MTP k1, prefix
 *      checkpoint. The whole-card rule keeps main's placement (as block 1 ran); the bank rule
 *      engages the policy, and its estimate is at most block 1's fullest bank after the first
 *      verify. Red: the whole-card rule alone.
 *   3. TODAY'S LOADS: the 30,720 default (and its bf16 profile) keeps the fit policy; Ornith 32k,
 *      Qwen3.6 36,864 and Bonsai Bfp4 32k keep main's placement, and each estimate is at most
 *      its measured fullest bank. Qwen3.8 rows=0 32k, whose fullest bank measured 97 MiB under
 *      main's placement, now takes the policy.
 *
 * Build: `make tests/test_tt_q38_fit_banks` (any host).
 *
 * @status None
 * @issues None
 * @todo None
 */

#include <stdbool.h>
#include <stdint.h>
#include <stdio.h>
#include <string.h>

#include "../src/models/qwen35moe_tt_fit_plan.h"

#define GIB (1024.0 * 1024.0 * 1024.0)
#define MIB (1024.0 * 1024.0)
#define P150_BANKS 8u
#define P150_BANK_BYTES 4278190080ull
/* pn_tt_alloc_scratch at open (test_tt_q38_mixw_fit.c): max Kt x bf16 tile in bank 0, max Nt in
 * bank 1. The banks' sum is block 1's "31.859 GiB free" at the decide. */
#define SCRATCH_BANK0 (544ull * 2048ull)
#define SCRATCH_BANK1 (7760ull * 2048ull)

/* Qwen3.8-27B (test_tt_q38_mixw_fit.c): the Bfp8 trunk's 497 matmuls, the joint attn_q's 16
 * tensors, the bytes M-LL's 192 FFN matrices at bfp4_b free, and the dense MTP head as the
 * decide's fit_shape_mtp prices it (the .mtp.fttw + 2,057 draft columns x 160 x Bfp8). */
#define Q38_TRUNK_BYTES 27231682560ull
#define Q38_JOINT_Q_BYTES 1069547520ull
#define Q38_MLL_FREED 8556380160ull
#define Q38_MTP_HEAD (451323072ull + 2057ull * 160ull * 1088ull)
#define Q38_BFP8_TILE 1088ull
#define K8V4_K 1088u
#define K8V4_V 576u
#define VERIFY_ROWS_K1 2u /* pn_qwen35_tt_mtp_verify_rows() at PN_TT_SPEC=1 */
/* Hand-worked: a snapshot set = 48 layers x (3 MiB state + 3.75 MiB window) (test_tt_q38_fit.c) */
#define Q38_SNAPSHOT_SET 339738624ull
#define WHOLE_CARD_SHARE (768ull * 1048576ull / 8u)

/* Block 1: the fullest bank
 * after the load, prefix checkpoint in, "bank 4 ... 0.114 GiB free" (low end 0.1135 GiB). */
#define BLOCK1_FULLEST_FREE 121870295ull

static int g_checks, g_failures;

static void check(bool ok, const char *what)
{
    g_checks++;
    if (!ok) {
        g_failures++;
        printf("FAIL %s\n", what);
    }
}

/** The decide's free bytes: every bank empty but the open's scratch rows. */
static void decide_free(uint64_t *free_bytes)
{
    for (uint32_t b = 0; b < P150_BANKS; b++)
        free_bytes[b] = P150_BANK_BYTES;
    free_bytes[0] -= SCRATCH_BANK0;
    free_bytes[1] -= SCRATCH_BANK1;
}

static pn_qwen35_tt_fit_shape shape_of(uint32_t embd, uint32_t ffn, uint32_t heads,
                                       uint32_t kv_heads, uint32_t n_attn, uint32_t n_ssm,
                                       uint32_t d_inner, uint32_t v_heads, uint32_t conv_dim,
                                       uint32_t ctx)
{
    pn_qwen35_tt_fit_shape s;

    memset(&s, 0, sizeof(s));
    s.n_embd = embd; s.ffn = ffn; s.n_head = heads; s.n_head_kv = kv_heads; s.head_dim = 256u;
    s.n_attn = n_attn; s.n_ssm = n_ssm; s.d_inner = d_inner; s.v_heads = v_heads;
    s.conv_dim = conv_dim; s.conv_taps = 4u; s.ssm_state = 128u; s.ctx = ctx;
    s.tall_rows = 64u; s.mirrors = P150_BANKS; s.outproj_mirrors = true;
    s.gdnc_replicas = 8u; s.attn_on_card = true; s.gdn_on_card = true; s.state_split = true;
    return s;
}

/** Qwen3.8-27B at `ctx`, K8V4 unless `bf16`, with the MTP head's `rows` verify rows (0 = off). */
static pn_qwen35_tt_fit_shape q38(uint32_t ctx, bool bf16, uint32_t rows)
{
    pn_qwen35_tt_fit_shape s = shape_of(5120, 17408, 24, 4, 16, 48, 6144, 48, 10240, ctx);

    s.kv_k_tile_bytes = bf16 ? 0u : K8V4_K;
    s.kv_v_tile_bytes = bf16 ? 0u : K8V4_V;
    s.mtp_bytes = rows ? Q38_MTP_HEAD : 0u;
    s.mtp_rows = rows;
    return s;
}

static uint64_t joint_q(uint32_t layers, uint32_t embd, uint32_t heads, uint64_t tile)
{
    return (uint64_t)layers * (embd / 32u) * (2u * heads * 8u) * tile;
}

static uint64_t sum(const uint64_t *free_bytes)
{
    uint64_t card = 0u;

    for (uint32_t b = 0; b < P150_BANKS; b++)
        card += free_bytes[b];
    return card;
}

/* ================================================================
 * (1) Hand-worked
 * ================================================================ */

static void test_hand_worked(void)
{
    const pn_qwen35_tt_fit_shape s = q38(262144u, false, VERIFY_ROWS_K1);
    pn_qwen35_tt_fit_shape no_verify = s;
    /* One bank's group of one layer at 262,144: 1 KV head x 8 dim tiles x 8,192 position tiles.
     * 17 regions (16 + the head's) alternate the sets, so the larger role is 9 K + 8 V groups. */
    const uint64_t group = 8ull * 8192ull;
    const uint64_t kv_larger = 9u * group * K8V4_K + 8u * group * K8V4_V;
    const uint64_t kv_all = 17ull * 1664ull * 262144ull; /* K8V4: 1,664 B a position a layer */
    pn_qwen35_tt_fit_floor floor, floor_nv;
    uint64_t free_bytes[P150_BANKS], even, bank_floor;
    uint32_t bank = 99u;
    int64_t low;
    char what[200];

    pn_qwen35_tt_fit_floor_compute(&s, Q38_TRUNK_BYTES - Q38_MLL_FREED, Q38_JOINT_Q_BYTES, &floor);
    check(kv_larger == 943718400ull && kv_all == floor.kv + 1664ull * 262144ull &&
              kv_larger > kv_all / P150_BANKS,
          "KV: 943,718,400 B in the larger role, above the even share of the 17 regions");
    even = floor.total - kv_all + Q38_SNAPSHOT_SET;
    bank_floor = (even + 7u) / 8u + kv_larger;
    snprintf(what, sizeof(what), "bank floor = ceil((floor - KV + checkpoint set) / 8) + KV larger "
             "role = %llu B", (unsigned long long)bank_floor);
    check(pn_qwen35_tt_fit_bank_floor(&s, &floor, true, P150_BANKS) == bank_floor, what);
    check(pn_qwen35_tt_fit_bank_floor(&s, &floor, true, P150_BANKS) -
                  pn_qwen35_tt_fit_bank_floor(&s, &floor, false, P150_BANKS) ==
              Q38_SNAPSHOT_SET / 8u,
          "the checkpoint costs its set's even share (42,467,328 B a bank)");
    no_verify.mtp_rows = 0u;
    pn_qwen35_tt_fit_floor_compute(&no_verify, Q38_TRUNK_BYTES - Q38_MLL_FREED, Q38_JOINT_Q_BYTES,
                                   &floor_nv);
    check(pn_qwen35_tt_fit_bank_floor(&s, &floor, true, P150_BANKS) -
                  pn_qwen35_tt_fit_bank_floor(&no_verify, &floor_nv, true, P150_BANKS) ==
              2u * Q38_SNAPSHOT_SET / 8u,
          "k1's two verify slots cost their even share (84,934,656 B a bank)");

    decide_free(free_bytes);
    low = pn_qwen35_tt_fit_bank_low(&s, &floor, true, free_bytes, P150_BANKS, &bank);
    check(low == (int64_t)(P150_BANK_BYTES - SCRATCH_BANK1) - (int64_t)bank_floor -
                     (int64_t)WHOLE_CARD_SHARE && bank == 1u,
          "fullest bank = bank 1 (the scratch's 15 MiB) less the bank floor less 96 MiB");

    /* A bank short on its own: Qwen3.6 36,864 keeps ~200 MiB a bank; take 150 MiB from bank 5. */
    {
        const pn_qwen35_tt_fit_shape q36 = shape_of(2048, 512, 16, 2, 10, 30, 4096, 32, 8192,
                                                    36864);
        pn_qwen35_tt_fit_floor f36;
        int64_t before;

        pn_qwen35_tt_fit_floor_compute(&q36, (uint64_t)(26.316 * GIB),
                                       joint_q(10, 2048, 16, Q38_BFP8_TILE), &f36);
        check(pn_qwen35_tt_fit_kv_bank_bytes(&q36, 10u, P150_BANKS, true) == 0u &&
                  pn_qwen35_tt_fit_bank_floor(&q36, &f36, false, P150_BANKS) ==
                      (f36.total + 7u) / 8u,
              "Qwen3.6 (2 KV heads, no bank-rule placement): every term spread evenly");
        before = pn_qwen35_tt_fit_bank_low(&q36, &f36, true, free_bytes, P150_BANKS, NULL);
        free_bytes[5] -= 150ull * 1048576ull;
        low = pn_qwen35_tt_fit_bank_low(&q36, &f36, true, free_bytes, P150_BANKS, &bank);
        snprintf(what, sizeof(what), "bank 5 150 MiB short: %.1f -> %.1f MiB, named; the "
                 "whole-card rule still keeps main's, the bank rule engages",
                 before / MIB, low / MIB);
        check(before >= (int64_t)PN_QWEN35_TT_TIER_HEADROOM && bank == 5u &&
                  /* bank 1 bound before by its scratch rows; bank 5 had them free */
                  low == before + (int64_t)SCRATCH_BANK1 - 150ll * 1048576ll &&
                  !pn_qwen35_tt_fit_policy_wanted(&f36, sum(free_bytes)) &&
                  pn_qwen35_tt_fit_policy_wanted_banks(&q36, &f36, true, free_bytes, P150_BANKS),
              what);
    }
    check(pn_qwen35_tt_fit_bank_low(&s, &floor, true, free_bytes, 0u, NULL) == INT64_MIN &&
              !pn_qwen35_tt_fit_policy_wanted_banks(&s, &floor, true, free_bytes, 0u) &&
              !pn_qwen35_tt_fit_policy_wanted_banks(&s, &floor, true, free_bytes,
                                                    PN_QWEN35_TT_FIT_MAX_BANKS + 1u),
          "no banks (or too many): no estimate, no policy");
}

/* ================================================================
 * (2) The tier at 262,144 (block 1)
 * ================================================================ */

static void test_tier(void)
{
    const pn_qwen35_tt_fit_shape s = q38(262144u, false, VERIFY_ROWS_K1);
    /* Main's first verify at k1 takes rows = 2 slots (legacy map) on top of the load. */
    const uint64_t after_verify = BLOCK1_FULLEST_FREE - 2u * Q38_SNAPSHOT_SET / P150_BANKS;
    pn_qwen35_tt_fit_floor floor;
    uint64_t free_bytes[P150_BANKS];
    uint32_t bank = 99u;
    int64_t low;
    char what[240];

    decide_free(free_bytes);
    pn_qwen35_tt_fit_floor_compute(&s, Q38_TRUNK_BYTES - Q38_MLL_FREED, Q38_JOINT_Q_BYTES, &floor);
    low = pn_qwen35_tt_fit_bank_low(&s, &floor, true, free_bytes, P150_BANKS, &bank);
    printf("  tier 262,144 M-LL K8V4 k1 + ckpt: floor %.3f GiB of %.3f free; fullest bank %u "
           "~%.1f MiB (block 1: %.1f at load, ~%.1f after the first verify)\n",
           floor.total / GIB, sum(free_bytes) / GIB, bank, low / MIB, BLOCK1_FULLEST_FREE / MIB,
           after_verify / MIB);
    snprintf(what, sizeof(what), "main's floor %.3f GiB (#147: 30.741) and the decide's %.3f GiB "
             "free (block 1: 31.859)", floor.total / GIB, sum(free_bytes) / GIB);
    check(floor.total / GIB > 30.740 && floor.total / GIB < 30.742 &&
              sum(free_bytes) / GIB > 31.8585 && sum(free_bytes) / GIB < 31.8595, what);
    check(!pn_qwen35_tt_fit_policy_wanted(&floor, sum(free_bytes)),
          "RED for the old rule: the whole-card rule keeps main's placement (as block 1 ran)");
    check(pn_qwen35_tt_fit_policy_wanted_banks(&s, &floor, true, free_bytes, P150_BANKS),
          "the bank rule engages the fit policy at the tier");
    snprintf(what, sizeof(what), "the estimate (%.1f MiB) is at most block 1's fullest bank "
             "after the first verify (~%.1f MiB), and under 128", low / MIB, after_verify / MIB);
    check(low <= (int64_t)after_verify && low < (int64_t)PN_QWEN35_TT_TIER_HEADROOM, what);
    /* Without the checkpoint, the estimate still engages it (the slots alone bind). */
    check(pn_qwen35_tt_fit_policy_wanted_banks(&s, &floor, false, free_bytes, P150_BANKS),
          "the tier engages it with the checkpoint off too");
}

/* ================================================================
 * (3) Today's loads
 * ================================================================ */

typedef struct {
    const char *name;
    pn_qwen35_tt_fit_shape shape;
    uint64_t weights, joint_q;
    double measured_fullest_gib; /* 0 = not measured under main's placement */
    bool old_rule, new_rule;
} load_case;

static void test_todays_loads(void)
{
    const uint64_t bonsai4 = Q38_TRUNK_BYTES / Q38_BFP8_TILE * 576u;
    pn_qwen35_tt_fit_shape rows0 = q38(32768u, true, 0u);
    load_case cases[] = {
        /* The shipped default (catalog p150-one): Bfp8, 30,720, K8V4, MTP k1; and p150-one-bf16. */
        {"Qwen3.8 default 30,720", q38(30720u, false, VERIFY_ROWS_K1), Q38_TRUNK_BYTES,
         Q38_JOINT_Q_BYTES, 0.0, true, true},
        {"Qwen3.8 bf16 30,720", q38(30720u, true, 0u), Q38_TRUNK_BYTES, Q38_JOINT_Q_BYTES, 0.0,
         true, true},
        /* Main's placement on the p150, fullest bank from the census (test_tt_q38_fit.c). */
        {"Ornith-1.5-9B 32k", shape_of(4096, 12288, 16, 4, 8, 24, 4096, 32, 8192, 32768),
         (uint64_t)(8.093 * GIB), joint_q(8, 4096, 16, Q38_BFP8_TILE), 2.658, false, false},
        {"Qwen3.6-35B-A3B 36,864", shape_of(2048, 512, 16, 2, 10, 30, 4096, 32, 8192, 36864),
         (uint64_t)(26.316 * GIB), joint_q(10, 2048, 16, Q38_BFP8_TILE), 0.368, false, false},
        {"Bonsai-27B Bfp4 32k", shape_of(5120, 17408, 24, 4, 16, 48, 6144, 48, 10240, 32768),
         bonsai4, Q38_JOINT_Q_BYTES / Q38_BFP8_TILE * 576u, 0.0, false, false},
        {"Qwen3.8 rows=0 32k", rows0, Q38_TRUNK_BYTES, Q38_JOINT_Q_BYTES, 0.095, false, true},
    };
    uint64_t free_bytes[P150_BANKS];
    char what[220];

    cases[5].shape.tall_rows = 0u;
    decide_free(free_bytes);
    for (size_t i = 0; i < sizeof(cases) / sizeof(cases[0]); i++) {
        const load_case *c = &cases[i];
        /* The census loads ran without the checkpoint; the default and the tier carry it. */
        const bool ckpt = c->measured_fullest_gib == 0.0;
        pn_qwen35_tt_fit_floor floor;
        bool old_rule, new_rule;
        int64_t low;

        pn_qwen35_tt_fit_floor_compute(&c->shape, c->weights, c->joint_q, &floor);
        old_rule = pn_qwen35_tt_fit_policy_wanted(&floor, sum(free_bytes));
        new_rule = pn_qwen35_tt_fit_policy_wanted_banks(&c->shape, &floor, ckpt, free_bytes,
                                                        P150_BANKS);
        low = pn_qwen35_tt_fit_bank_low(&c->shape, &floor, ckpt, free_bytes, P150_BANKS, NULL);
        printf("  %-24s floor %.3f GiB, fullest bank ~%8.1f MiB: whole-card %s, banks %s\n",
               c->name, floor.total / GIB, low / MIB, old_rule ? "FIT" : "main",
               new_rule ? "FIT" : "main");
        snprintf(what, sizeof(what), "%s: whole-card rule %s, both rules %s", c->name,
                 c->old_rule ? "fit policy" : "main's", c->new_rule ? "fit policy" : "main's");
        check(old_rule == c->old_rule && new_rule == c->new_rule, what);
        if (c->measured_fullest_gib > 0.0) {
            snprintf(what, sizeof(what), "%s: estimate %.1f MiB <= measured fullest bank %.1f "
                     "MiB", c->name, low / MIB, c->measured_fullest_gib * 1024.0);
            check(low <= (int64_t)(c->measured_fullest_gib * GIB), what);
        }
    }
    check(0.095 * 1024.0 < PN_QWEN35_TT_TIER_HEADROOM / MIB,
          "rows=0 32k's fullest bank measured 97 MiB: under 128, so the policy is right there");
}

/* ================================================================
 * (4) The exact card gates' tables (qwen35moe_gate_tables.h)
 * ================================================================ */

/* Measured fullest banks (p150): the tier under the policy after its first verify (fitbanks
 * block, q38-tt-fitpolicy-banks: 242 MiB), and the 32k oracle dump under the policy, no MTP
 * (its "DRAM after arm" line: fullest bank 0.147 GiB, 1.370 GiB free). */
#define TIER_POLICY_FULLEST (242ull * 1048576ull)
#define DUMP32K_FULLEST ((uint64_t)(0.147 * GIB))
#define DUMP32K_FREE ((uint64_t)(1.370 * GIB))
/* Qwen3.8-27B's tables as built on the p150 (2026-11-19 card run): every layer packed at its
 * widest, <= 7,829 words a head (8,004 on the Mac's libm); priced here at 8,004 for all. */
#define Q38_SLOT ((8004ull * 4u + 63u) & ~63ull)
#define Q38_BETA_BYTES (((16ull + 7240ull) * 4u + 63u) & ~63ull)

/** Every bank's share at S stripes: layers x ceil(48 / S) slots + one beta copy. */
static uint64_t q38_bank_share(uint32_t stripes)
{
    return 48ull * ((48u + stripes - 1u) / stripes) * Q38_SLOT + Q38_BETA_BYTES;
}

/** Every layer's stripes on distinct banks. */
static bool stripes_distinct(const uint32_t *bank_of, uint32_t stripes)
{
    for (uint32_t l = 0; l < 48u; l++)
        for (uint32_t i = 0; i < stripes; i++)
            for (uint32_t j = i + 1u; j < stripes; j++)
                if (bank_of[l * 8u + i] == bank_of[l * 8u + j])
                    return false;
    return true;
}

static void test_gate_tables(void)
{
    const uint64_t priced = pn_qwen35_tt_fit_gate_table_bytes(48u, 48u);
    uint64_t slot[48], free_bytes[P150_BANKS], before[P150_BANKS];
    uint32_t bank_of[48 * PN_QWEN35_TT_FIT_TABLE_STRIPES];
    char what[240];

    snprintf(what, sizeof(what), "priced at the cap: (48 x 48 + 8 beta copies) x 40 KiB + 48 "
             "blocks of 256 B = %llu B (%.1f MiB)", (unsigned long long)priced, priced / MIB);
    check(priced == 2312ull * 40960ull + 48ull * 256ull, what);
    check(pn_qwen35_tt_fit_gate_table_bytes(2u, 13u) == (2ull * 16u + 8u) * 40960u + 512u,
          "13 heads price as 16 (a whole stripe at 8)");
    for (int pass = 0; pass < 2; pass++) {
        /* The 30,720 default and the tier, both with the k1 verify slots and the checkpoint. */
        pn_qwen35_tt_fit_shape s = q38(pass ? 262144u : 30720u, false, VERIFY_ROWS_K1);
        const uint64_t weights = pass ? Q38_TRUNK_BYTES - Q38_MLL_FREED : Q38_TRUNK_BYTES;
        pn_qwen35_tt_fit_floor without, with;
        int64_t low_without, low_with;

        decide_free(free_bytes);
        pn_qwen35_tt_fit_floor_compute(&s, weights, Q38_JOINT_Q_BYTES, &without);
        s.gate_tables = true;
        pn_qwen35_tt_fit_floor_compute(&s, weights, Q38_JOINT_Q_BYTES, &with);
        low_without = pn_qwen35_tt_fit_bank_low(&s, &without, true, free_bytes, P150_BANKS, NULL);
        low_with = pn_qwen35_tt_fit_bank_low(&s, &with, true, free_bytes, P150_BANKS, NULL);
        snprintf(what, sizeof(what), "%s: the floor grows by the tables (%.1f MiB), the fullest "
                 "bank estimate by their even share (%.1f -> %.1f MiB), and the policy still "
                 "engages", pass ? "tier 262,144" : "default 30,720", with.gate_tables / MIB,
                 low_without / MIB, low_with / MIB);
        check(with.total - without.total == priced && with.gate_tables == priced &&
                  low_without - low_with == (int64_t)((with.total - with.kv + 7u) / 8u -
                                                      (without.total - without.kv + 7u) / 8u) &&
                  pn_qwen35_tt_fit_policy_wanted_banks(&s, &with, true, free_bytes, P150_BANKS),
              what);
    }
    for (uint32_t l = 0; l < 48u; l++)
        slot[l] = Q38_SLOT;
    /* THE TIER under the policy: verify slots and checkpoint already reserved, so the arm keeps
     * the bare 128 MiB. Every bank at the measured fullest (the worst case): 8 stripes. */
    for (uint32_t b = 0; b < P150_BANKS; b++)
        free_bytes[b] = before[b] = TIER_POLICY_FULLEST;
    {
        uint64_t lowest = UINT64_MAX;
        bool distinct = true, even = true;
        const uint32_t stripes = pn_qwen35_tt_fit_table_stripes(
            free_bytes, P150_BANKS, slot, 48u, 48u, Q38_BETA_BYTES, PN_QWEN35_TT_TIER_HEADROOM,
            bank_of);

        for (uint32_t b = 0; b < P150_BANKS; b++) {
            if (free_bytes[b] < lowest)
                lowest = free_bytes[b];
            even = even && before[b] - free_bytes[b] == q38_bank_share(8u);
        }
        distinct = stripes_distinct(bank_of, stripes);
        snprintf(what, sizeof(what), "tier: %u stripes, every layer over distinct banks, each "
                 "bank %.1f MiB of tables, every bank keeps %.1f MiB (>= 128, verify slots in)",
                 stripes, q38_bank_share(8u) / MIB, lowest / MIB);
        check(stripes == 8u && distinct && even && lowest >= PN_QWEN35_TT_TIER_HEADROOM, what);
    }
    /* THE 32k DUMP under the policy (measured): the fullest bank at 147 MiB, the rest sharing
     * the card's other free bytes. Its 8-stripe share still leaves it over 128 MiB. */
    free_bytes[0] = before[0] = DUMP32K_FULLEST;
    for (uint32_t b = 1; b < P150_BANKS; b++)
        free_bytes[b] = before[b] = (DUMP32K_FREE - DUMP32K_FULLEST) / (P150_BANKS - 1u);
    {
        const uint32_t stripes = pn_qwen35_tt_fit_table_stripes(
            free_bytes, P150_BANKS, slot, 48u, 48u, Q38_BETA_BYTES, PN_QWEN35_TT_TIER_HEADROOM,
            bank_of);

        snprintf(what, sizeof(what), "32k dump: %u stripes, every layer over distinct banks, "
                 "the fullest bank %.1f -> %.1f MiB", stripes, before[0] / MIB,
                 free_bytes[0] / MIB);
        check(stripes == 8u && stripes_distinct(bank_of, stripes) &&
                  free_bytes[0] >= PN_QWEN35_TT_TIER_HEADROOM, what);
    }
    /* ONE BANK TOO FULL FOR ITS SHARE: 8 stripes would need it, so 4 over the other banks. */
    free_bytes[0] = PN_QWEN35_TT_TIER_HEADROOM + q38_bank_share(8u) / 2u;
    for (uint32_t b = 1; b < P150_BANKS; b++)
        free_bytes[b] = PN_QWEN35_TT_TIER_HEADROOM + 4u * q38_bank_share(8u);
    before[0] = free_bytes[0];
    {
        const uint32_t stripes = pn_qwen35_tt_fit_table_stripes(
            free_bytes, P150_BANKS, slot, 48u, 48u, Q38_BETA_BYTES, PN_QWEN35_TT_TIER_HEADROOM,
            bank_of);
        bool untouched = true;

        for (uint32_t l = 0; l < 48u; l++)
            for (uint32_t i = 0; i < stripes; i++)
                untouched = untouched && bank_of[l * 8u + i] != 0u;
        snprintf(what, sizeof(what), "a bank short of its 8-stripe share: %u stripes, that bank "
                 "untouched", stripes);
        check(stripes == 4u && untouched && free_bytes[0] == before[0], what);
    }
    /* MAIN'S PLACEMENT with k1: the verify slots are still to come, so the arm keeps their share
     * on top of 128 MiB — 2 snapshot sets / 8 banks. */
    for (uint32_t b = 0; b < P150_BANKS; b++)
        free_bytes[b] = before[b] =
            PN_QWEN35_TT_TIER_HEADROOM + 2u * Q38_SNAPSHOT_SET / 8u + 8ull * 1048576ull;
    {
        const uint32_t stripes = pn_qwen35_tt_fit_table_stripes(
            free_bytes, P150_BANKS, slot, 48u, 48u, Q38_BETA_BYTES,
            PN_QWEN35_TT_TIER_HEADROOM + 2u * Q38_SNAPSHOT_SET / 8u, bank_of);

        check(stripes == 0u && memcmp(free_bytes, before, sizeof(before)) == 0,
              "RED: banks 8 MiB over 128 MiB + the verify slots' share cannot take 66 MiB of "
              "tables at any striping — refused, free bytes untouched (the host gates stay)");
    }
    check(pn_qwen35_tt_fit_table_stripes(free_bytes, 0u, slot, 48u, 48u, Q38_BETA_BYTES, 0u,
                                         bank_of) == 0u,
          "no banks: refused");
}

int main(void)
{
    printf("test_tt_q38_fit_banks: the fit policy's bank rule\n");
    printf("(1) hand-worked\n");
    test_hand_worked();
    printf("(2) the tier at 262,144 (block 1)\n");
    test_tier();
    printf("(3) today's loads\n");
    test_todays_loads();
    printf("(4) the exact card gates' tables\n");
    test_gate_tables();
    printf("%d checks, %d failures\n", g_checks, g_failures);
    return g_failures ? 1 : 0;
}
