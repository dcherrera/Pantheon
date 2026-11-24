/* SPDX-License-Identifier: AGPL-3.0-or-later */
/* Copyright (C) 2026 David (Corey) Herrera */
/**
 * @file test_tt_q38_reserve.c
 * @brief Qwen3.8-27B Bfp8 on one p150: which contexts the load accepts once every post-load
 *        prefill buffer is counted at load, where the prefix checkpoint fits, and that a
 *        prefill which aborts part way invalidates its session.
 *
 * Opens no device. It drives the production planning code:
 *   - the tall set under the fit policy: `pn_qwen35_tt_tall_requests()` +
 *     `pn_tt_strip_placement_simulate()` (mirrors first);
 *   - the mandatory reserve and the checkpoint twins: `pn_tt_bank_need_*()`
 *     (pn_tt_prefill_reserve.h), the arithmetic `pn_qwen35_tt_prefill_reserve()` decides with;
 *   - the abort: `pn_qwen35_tt_prefill_abort()`, the function every abort site calls.
 *
 * THE BANK STATE. The p150 measured the fit-policy load (PR #97) at the "DRAM arm: MTP head"
 * stage: 32,768 -> 1.867 GiB free, fullest bank 0.219; 34,816 -> 1.742 / 0.203; 36,864 ->
 * 1.617 / 0.188. Those three are exactly the KV at 64 KiB a position over the
 * card and 8 KiB a position in the fullest bank, so other contexts are derived the same way.
 * The other seven banks are unknown: every check runs an EVEN and a SKEWED spread consistent
 * with the census (pn test_tt_q38_fit.c) and every strip-cursor start, and takes the worst.
 *
 * Build: `make USE_TT=1 tests/test_tt_q38_reserve` (Mac or Linux, no card).
 *
 * @status None
 * @issues None
 * @todo None
 */

#include <stdbool.h>
#include <stdint.h>
#include <stdio.h>
#include <string.h>

#ifdef USE_TT

#include "../src/accel/tt/pn_tt_prefill_reserve.h"
#include "../src/accel/tt/pn_tt_strip_placement.h"
#include "../src/models/qwen35moe_tt_internal.h"

#define BANKS 8u
#define GIB (1024.0 * 1024.0 * 1024.0)
#define MIB (1024.0 * 1024.0)
#define ROUNDING_GIB 0.0005
#define TALL_BYTES_PER_TILE (64ull * 2048ull)

/* Qwen3.8-27B GDN geometry: 48 SSM layers, 48 v-heads of 128 x 128 fp32, conv_dim 10240. */
#define SSM_LAYERS 48u
#define GDNC_MEMBERS 8u
#define STATE_BYTES (48ull * 4u * 4u * 4096u)           /* 3.00 MiB a layer */
#define WINDOW_BYTES (320ull * 3u * 4096u)              /* 3.75 MiB a layer */
#define SLICE_BYTES (STATE_BYTES / GDNC_MEMBERS)        /* 384 KiB a member */
#define ALT_BYTES (2ull * 320u * 4096u + 48ull * 4u * 4096u) /* PIPE 2: 2 streams + y */
#define ROPE_SLOT_BYTES 8192ull                         /* a 32-row, 64-dim rotary block */
#define ROPE_SLOTS 62u
/* Option D: decode's own 5.00 MiB conv weight a layer, round-robin, 6 layers a bank. */
#define D_FREED_PER_BANK (6ull * 5u * 1024u * 1024u)

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
 * The bank state before the tall set, from the p150 census
 * ================================================================ */

static void census_at(int ctx, double *free_gib, double *fullest_gib)
{
    const double positions = 32768.0 - (double)ctx;

    *free_gib = 1.867 + positions * 65536.0 / GIB;
    *fullest_gib = 0.219 + positions * 8192.0 / GIB;
}

static void bank_spread(int ctx, bool skewed, uint64_t extra_per_bank, uint64_t *free_bytes)
{
    double free_gib, fullest_gib;
    uint64_t total, fullest;

    census_at(ctx, &free_gib, &fullest_gib);
    total = (uint64_t)((free_gib - ROUNDING_GIB) * GIB);
    fullest = (uint64_t)((fullest_gib - ROUNDING_GIB) * GIB);
    for (uint32_t b = 0; b < BANKS; b++)
        free_bytes[b] = skewed ? fullest : (total - fullest) / (BANKS - 1u);
    free_bytes[BANKS - 1u] = fullest;
    if (skewed)
        free_bytes[0] = total - (uint64_t)(BANKS - 1u) * fullest;
    for (uint32_t b = 0; b < BANKS; b++)
        free_bytes[b] += extra_per_bank;
}

/* ================================================================
 * The fit-policy tall set and the post-load needs
 * ================================================================ */

static size_t q38_tall_requests(pn_tt_strip_request *out)
{
    static pn_qwen35_tt g;
    pn_qwen35_tt_strips *c = &g.card;
    pn_qwen35_tt_attn_strips *a = &g.attn_card;
    struct { pn_tt_activation *strip; uint32_t tiles, replicas; } set[] = {
        {&c->hidden[0], 160, 1}, {&c->hidden[1], 160, 1}, {&c->normed, 160, BANKS},
        {&c->project, 160, 1}, {&c->ssm_qkv, 320, 1}, {&c->ssm_z, 192, 1},
        {&c->ssm_alpha, 2, 1}, {&c->ssm_beta, 2, 1}, {&c->ssm_gated, 192, 1},
        {&c->attn_qg, 384, 1}, {&c->attn_k, 32, 1}, {&c->attn_v, 32, 1},
        {&c->attn_out, 192, 1}, {&c->ffn_gate, 544, 1}, {&c->ffn_up, 544, 1},
        {&c->ffn_swiglu, 544, BANKS}, {&c->logits, 7760, 1}, {&a->q, 192, 1},
        {&a->q_rot, 192, 1}, {&a->k_rot, 32, 1}, {&a->z, 192, 1}, {&a->gate, 192, 1},
        {&a->gated, 192, 1},
    };

    memset(&g, 0, sizeof(g));
    for (size_t i = 0; i < sizeof(set) / sizeof(set[0]); i++) {
        set[i].strip->tile_count = set[i].tiles;
        set[i].strip->replica_count = set[i].replicas;
    }
    g.attn_on_card = true;
    return pn_qwen35_tt_tall_requests(&g, BANKS, true, true, out, 64);
}

/** The GDN region banks: weight, window, state per layer from one round-robin cursor. */
static void gdn_region_banks(uint32_t *window_bank, uint32_t *state_bank)
{
    for (uint32_t layer = 0; layer < SSM_LAYERS; layer++) {
        window_bank[layer] = (3u * layer + 1u) % BANKS;
        state_bank[layer] = (3u * layer + 2u) % BANKS;
    }
}

/** The mandatory reserve. PIPE 2's streams are pinned to one bank on the card; which one the
 *  census cannot say, so here they count as an any-bank allocation. */
static void mandatory_need(pn_tt_bank_need *need)
{
    uint32_t window_bank[SSM_LAYERS], state_bank[SSM_LAYERS];

    gdn_region_banks(window_bank, state_bank);
    pn_tt_bank_need_init(need, BANKS);
    pn_tt_bank_need_state_split(need, state_bank, SSM_LAYERS, GDNC_MEMBERS, SLICE_BYTES);
    pn_tt_bank_need_any(need, ALT_BYTES);
    for (uint32_t slot = 0; slot < ROPE_SLOTS; slot++)
        pn_tt_bank_need_any(need, ROPE_SLOT_BYTES);
}

static void checkpoint_need(pn_tt_bank_need *need)
{
    uint32_t window_bank[SSM_LAYERS], state_bank[SSM_LAYERS];

    gdn_region_banks(window_bank, state_bank);
    pn_tt_bank_need_init(need, BANKS);
    pn_tt_bank_need_twins(need, window_bank, state_bank, SSM_LAYERS, WINDOW_BYTES, STATE_BYTES);
}

/** What one context gets over every spread and cursor start (the worst case). */
typedef struct {
    bool tall, mandatory, checkpoint;
    double fullest_after_reserve_mib;
} ctx_verdict;

static ctx_verdict judge_ctx(int ctx, uint64_t extra_per_bank)
{
    pn_tt_strip_request requests[64];
    const size_t n = q38_tall_requests(requests);
    size_t order[64];
    pn_tt_bank_need mandatory, twins;
    ctx_verdict v = {true, true, true, 1e9};

    (void)pn_tt_strip_placement_order(requests, n, order);
    mandatory_need(&mandatory);
    checkpoint_need(&twins);
    for (int skewed = 0; skewed < 2; skewed++)
        for (uint32_t start = 0; start < BANKS; start++) {
            uint64_t free_bytes[BANKS];
            uint32_t cursor = start;

            bank_spread(ctx, skewed != 0, extra_per_bank, free_bytes);
            if (pn_tt_strip_placement_simulate(requests, order, n,
                                               PN_TT_STRIP_POLICY_MIRRORS_FIRST,
                                               TALL_BYTES_PER_TILE, free_bytes, BANKS, &cursor,
                                               NULL) != 0) {
                v.tall = v.mandatory = v.checkpoint = false;
                continue;
            }
            if (!pn_tt_bank_need_fits(&mandatory, free_bytes, NULL, NULL)) {
                v.mandatory = v.checkpoint = false;
                continue;
            }
            pn_tt_bank_need_take(&mandatory, free_bytes);
            for (uint32_t b = 0; b < BANKS; b++)
                if ((double)free_bytes[b] / MIB < v.fullest_after_reserve_mib)
                    v.fullest_after_reserve_mib = (double)free_bytes[b] / MIB;
            if (!pn_tt_bank_need_fits(&twins, free_bytes, NULL, NULL))
                v.checkpoint = false;
        }
    return v;
}

/* ================================================================
 * (1) Load-time refusal; (2) where the checkpoint fits
 * ================================================================ */

static void check_load_refusal(void)
{
    const ctx_verdict v32 = judge_ctx(32768, 0u), v34 = judge_ctx(34816, 0u);
    const ctx_verdict v36 = judge_ctx(36864, 0u);
    pn_tt_bank_need need;

    mandatory_need(&need);
    printf("  mandatory reserve %.1f MiB (state split %.1f MiB = %.1f MiB a bank)\n",
           (double)pn_tt_bank_need_total(&need) / MIB,
           (double)SSM_LAYERS * GDNC_MEMBERS * SLICE_BYTES / MIB,
           (double)need.bank[0] / MIB);
    printf("  32,768: tall %d, reserve %d -> %s (fullest bank %.1f MiB after it)\n", v32.tall,
           v32.mandatory, v32.mandatory ? "LOADS" : "REFUSED", v32.fullest_after_reserve_mib);
    printf("  34,816: tall %d, reserve %d -> %s\n", v34.tall, v34.mandatory,
           v34.mandatory ? "LOADS" : "REFUSED");
    printf("  36,864: tall %d, reserve %d -> %s\n", v36.tall, v36.mandatory,
           v36.mandatory ? "LOADS" : "REFUSED");
    check(v32.tall && v32.mandatory, "32,768 loads with every prefill buffer reserved");
    check(v34.tall && !v34.mandatory,
          "34,816 places its tall set but REFUSES at load: the state split does not fit");
    check(!v36.tall && !v36.mandatory, "36,864 REFUSES at load: the tall set does not fit");
    check(!v32.checkpoint, "32,768: the prefix checkpoint does not fit (served with it OFF)");
}

/** The largest context (1,024 steps) at which tall + reserve + checkpoint fit, `extra` bytes
 *  added to every bank (option D). */
static int largest_ctx_with_checkpoint(uint64_t extra_per_bank)
{
    for (int ctx = 36864; ctx >= 8192; ctx -= 1024) {
        const ctx_verdict v = judge_ctx(ctx, extra_per_bank);

        if (v.checkpoint)
            return ctx;
    }
    return 0;
}

/** Option D above 30,720: the checkpoint is OFF cleanly (the load still reserves the
 *  mandatory buffers), and what the extra 30 MiB a bank does to the load-time refusals.
 *  D_FREED_PER_BANK is measured on the production placement by test_tt_q38_conv_share.c. */
static void check_d_contexts(void)
{
    static const int ctxs[] = { 30720, 31744, 32768, 34816, 36864 };
    ctx_verdict v[5];

    for (int i = 0; i < 5; i++) {
        v[i] = judge_ctx(ctxs[i], D_FREED_PER_BANK);
        printf("  with D %d: tall %d, reserve %d, checkpoint %d -> %s (fullest bank %.1f MiB "
               "after the reserve)\n", ctxs[i], v[i].tall, v[i].mandatory, v[i].checkpoint,
               !v[i].mandatory ? "REFUSED" : v[i].checkpoint ? "LOADS, checkpoint ON"
                                                            : "LOADS, checkpoint OFF",
               v[i].mandatory ? v[i].fullest_after_reserve_mib : 0.0);
    }
    check(v[0].mandatory && v[0].checkpoint, "with D 30,720 loads with the checkpoint ON");
    check(v[1].mandatory && !v[1].checkpoint, "with D 31,744 loads with the checkpoint OFF");
    check(v[2].mandatory && !v[2].checkpoint, "with D 32,768 loads with the checkpoint OFF");
    /* D's 30 MiB a bank also clears 34,816's state-split shortfall (13.7 MiB without it). */
    check(v[3].mandatory && !v[3].checkpoint, "with D 34,816 loads with the checkpoint OFF");
    /* 36,864 is not asserted: this worst-case spread refuses it, the card's real spread
     * reserved it on the p150 (fullest bank ~0 MiB after the reserve). The load decides. */
}

static void check_checkpoint_sweep(void)
{
    pn_tt_bank_need twins;
    const int today = largest_ctx_with_checkpoint(0u);
    const int with_d = largest_ctx_with_checkpoint(D_FREED_PER_BANK);
    const ctx_verdict v32_d = judge_ctx(32768, D_FREED_PER_BANK);

    checkpoint_need(&twins);
    printf("  checkpoint twins %.1f MiB (%.1f MiB a bank)\n",
           (double)pn_tt_bank_need_total(&twins) / MIB, (double)twins.bank[0] / MIB);
    printf("  largest ctx with the checkpoint: today %d, with option D %d; 32k with D: %s\n",
           today, with_d, v32_d.checkpoint ? "fits" : "does NOT fit");
    check(today > 0 && today < 32768, "the checkpoint fits below 32k today");
    check(with_d >= today, "option D never lowers the checkpoint's largest context");
    check(with_d == 30720, "with option D the checkpoint fits up to 30,720 (David's ship cap)");
    check_d_contexts();
}

/* ================================================================
 * (3) A prefill that aborts part way invalidates its session
 * ================================================================ */

static void check_abort_invalidates(void)
{
    static pn_qwen35_tt g;
    pn_state state;
    tm_tensor *out;

    memset(&g, 0, sizeof(g));
    memset(&state, 0, sizeof(state));
    /* A session mid-conversation: the card holds it, with a checkpoint at 384 it could resume
     * from, and a fault (PN_TT_RED=prefill_pass_fail on the card) aborts its next prefill after
     * the first pass wrote 2,048 positions. */
    state.pos = 2432;
    g.card_owner = &state;
    g.ckpt.valid = true;
    g.ckpt.restored = true;
    g.ckpt.pos = 384;
    g.ckpt.owner = &state;
    g.ckpt.pending_at = 2048;
    g.prefill_last_batched = 2048;
    out = pn_qwen35_tt_prefill_abort(&g, &state, "fault injection: forced pass failure", 2048);

    check(out == NULL, "an aborted prefill returns no logits");
    check(g.prefill_aborted, "the abort is flagged for the caller to fail the request");
    check(state.pos == 0, "the session is rewound to position 0, not left at 2,432");
    check(!g.ckpt.valid && !g.ckpt.restored && g.ckpt.pending_at == 0,
          "the prefix checkpoint is dropped and no capture is pending");
    check(g.card_owner == NULL, "the card no longer claims to hold the session");
    check(!pn_qwen35_tt_ckpt_restored_at(&g, &state, 384),
          "the session cannot resume from its old checkpoint");
    check(g.prefill_last_batched == -1, "no batched positions are reported");
}

int main(void)
{
    check_load_refusal();
    check_checkpoint_sweep();
    check_abort_invalidates();
    printf("test_tt_q38_reserve: %d checks, %d failed\n", g_checks, g_failures);
    return g_failures ? 1 : 0;
}

#else

int main(void)
{
    printf("test_tt_q38_reserve: needs USE_TT=1 (no card)\n");
    return 1;
}

#endif /* USE_TT */
