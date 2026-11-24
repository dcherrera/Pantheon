/* SPDX-License-Identifier: AGPL-3.0-or-later */
/* Copyright (C) 2026 David (Corey) Herrera */
/**
 * @file qwen35moe_fit_tt.c
 * @brief The qwen35 family's TT load-time fit: whether main's placement fits (the fit policy,
 *        qwen35moe_tt_fit_plan.h) and the post-load prefill reserve (pn_tt_prefill_reserve.h).
 *
 * Both answer one question at LOAD rather than at the first prompt: can this context run on
 * this card? Main's placement when it can; the fit options when only they make it fit; and a
 * refusal, with the shortfall named, when nothing does — never a serve that fails its first
 * prefill and keeps answering over a half-written state.
 *
 * @status None
 * @issues None
 * @todo None
 */

#ifdef USE_TT

#include "qwen35moe_tt_internal.h"
#include "qwen35moe_tt_fit_plan.h"

#include <stdio.h>
#include <stdlib.h>
#include <sys/stat.h>

#include "../accel/tt/pn_tt_gdn.h"
#include "../accel/tt/pn_tt_gdn_checkpoint.h"
#include "../accel/tt/pn_tt_gdnc.h"
#include "../accel/tt/pn_tt_kv_format.h"
#include "../accel/tt/pn_tt_replica_map.h"
#include "tt/kernels/resident_gdngate_protocol.h"

#define FIT_GIB (1024.0 * 1024.0 * 1024.0)
#define FIT_MIB (1024.0 * 1024.0)

/** A dense model's MTP head, when one resolved (pn_qwen35_tt_mtp_resolve): its artifact, the
 *  draft-vocabulary LM-head columns its arm stages, and its verify rows. The MoE head keeps its
 *  own guard at arm (pn_qwen35_tt_mtp_arm), so it does not enter the floor. */
static void fit_shape_mtp(pn_qwen35_tt *g, const pn_qwen35moe_config *cfg,
                          pn_qwen35_tt_fit_shape *s)
{
    uint32_t rows = 0u, cols = 0u, tile = 0u;
    struct stat st;

    if (!g->mtp.path[0] || !cfg->dense_ffn || cfg->n_expert > 0 || !s->attn_on_card ||
        stat(g->mtp.path, &st) != 0)
        return;
    s->mtp_bytes = (uint64_t)st.st_size;
    if (pn_tt_device_artifact_tensor_tiles(g->device, "output.weight", &rows, &cols, &tile) == 0)
        s->mtp_draft_bytes = (uint64_t)pn_qwen35_tt_mtp_draft_columns(cols) * rows * tile;
    s->mtp_bytes += s->mtp_draft_bytes;
    s->draft_view = pn_qwen35_tt_mtp_draft_view_env() > 0;
    s->mtp_rows = pn_qwen35_tt_mtp_verify_rows();
}

/** The load's shape for main's floor. Attention's default is declared here because the
 *  floor (and option A) must know it before the weights go up; the resident arm reads the
 *  same selection afterwards. */
static pn_qwen35_tt_fit_shape fit_shape(pn_qwen35_tt *g, const pn_qwen35moe_config *cfg,
                                        int n_ssm, int n_attn, uint32_t banks)
{
    pn_qwen35_tt_fit_shape s = {0};
    const char *env = getenv("PN_TT_GDNC_CONST_REPLICAS");
    const uint32_t replicas = env && *env ? (uint32_t)strtoul(env, NULL, 10) : 8u;

    pn_tt_device_oncard_default(g->device, PN_TT_ONCARD_ATTENTION, true);
    s.n_embd = (uint32_t)cfg->n_embd;
    s.ffn = (uint32_t)(cfg->dense_ffn ? cfg->ffn_length : cfg->shared_ff);
    s.n_head = (uint32_t)cfg->n_head;
    s.n_head_kv = (uint32_t)cfg->n_head_kv;
    s.head_dim = (uint32_t)cfg->head_dim;
    s.n_attn = (uint32_t)n_attn;
    s.n_ssm = (uint32_t)n_ssm;
    s.conv_dim = (uint32_t)cfg->conv_dim;
    s.conv_taps = (uint32_t)cfg->ssm_conv;
    s.ssm_state = (uint32_t)cfg->ssm_state;
    s.v_heads = (uint32_t)cfg->ssm_dt_rank;
    s.d_inner = (uint32_t)cfg->ssm_d_inner;
    s.ctx = (uint32_t)pn_qwen35moe_max_seq(cfg);
    /* The load's type: the fit policy's choice when it set one (pn_tt_device_set_kv_type). */
    s.kv_k_tile_bytes = pn_tt_kv_format_tile_bytes(pn_tt_device_kv_type(g->device).k_format);
    s.kv_v_tile_bytes = pn_tt_kv_format_tile_bytes(pn_tt_device_kv_type(g->device).v_format);
    s.tall_rows = pn_qwen35_tt_tall_rows_wanted();
    s.mirrors = pn_tt_activation_replica_count(g->device);
    s.outproj_mirrors = pn_qwen35_tt_outproj_mirrors_wanted();
    s.gdnc_replicas = replicas < 1u ? 1u : (replicas > banks ? banks : replicas);
    s.attn_on_card = pn_tt_device_oncard_requested(g->device, PN_TT_ONCARD_ATTENTION);
    s.gdn_on_card = pn_tt_device_oncard_requested(g->device, PN_TT_ONCARD_GDN);
    s.state_split = pn_tt_gdnc_state_split_enabled();
    /* Priced where the tables ARE the card gates: more V-heads than gdngate's cap, so its det
     * gates (Ornith, Qwen3.6) never arm instead. The arm's own bank rule still guards. */
    s.gate_tables = pn_qwen35_tt_gate_tables_wanted() && s.v_heads > kGgMaxHeads;
    fit_shape_mtp(g, cfg, &s);
    return s;
}

static bool fit_env_set(const char *name)
{
    const char *env = getenv(name);

    return env && *env;
}

/** Under the fit policy, the levers its own placement needs for the bank rule (lever E, G, F:
 *  qwen35moe_tt_fit_plan.h), from the bank state at the decide. */
static void fit_decide_levers(pn_qwen35_tt *g, const pn_qwen35_tt_fit_shape *shape,
                              uint64_t weights, uint64_t joint_q, bool ckpt,
                              const uint64_t *free_bytes, uint32_t banks)
{
    pn_tt_kv_type long_type;
    const bool long_ok = pn_tt_kv_type_long(&long_type);
    /* K: the profile's long-context type (PN_TT_KV_LONG, set by Local from options.kv_long). */
    const pn_qwen35_tt_fit_pins pins = {
        fit_env_set("PN_TT_GDNC_CONST_REPLICAS"), fit_env_set("PN_TT_PREFILL_TILE_ROWS"),
        fit_env_set("PN_TT_OUTPROJ_MIRRORS"), pn_qwen35_tt_mtp_draft_view_env() >= 0,
        long_ok ? pn_tt_kv_format_tile_bytes(long_type.k_format) : 0u,
        long_ok ? pn_tt_kv_format_tile_bytes(long_type.v_format) : 0u };
    pn_qwen35_tt_fit_levers levers;
    int64_t low = 0;
    const bool fits = pn_qwen35_tt_fit_levers_pick(shape, weights, joint_q, ckpt, free_bytes,
                                                   banks, &pins, &levers, &low);

    g->fit_levers = true;
    g->fit_tall_rows = levers.tall_rows;
    g->fit_outproj_mirrors = levers.outproj_mirrors;
    g->fit_draft_view = levers.draft_view;
    pn_tt_gdnc_request_replicas(g->device, levers.gdnc_replicas);
    /* Before the attention arm builds the KV layout (pn_tt_device_set_kv_type). */
    if (levers.kv_long)
        pn_tt_device_set_kv_type(g->device, long_type);
    fprintf(stderr, "qwen35_tt: fit levers at ctx %u — GDNC conv copies %u (of %u), tall strips "
                    "%u bands%s%s%s%s; the fullest bank is estimated to keep ~%.0f MiB (%s %.0f "
                    "MiB a bank)\n", shape->ctx, levers.gdnc_replicas, shape->gdnc_replicas,
            levers.tall_rows, levers.outproj_mirrors ? "" : ", gated strips unmirrored",
            levers.draft_view ? ", MTP draft reads the LM head in place" : "",
            levers.kv_long ? ", KV " : "", levers.kv_long ? pn_tt_kv_type_name(long_type) : "",
            (double)low / FIT_MIB, fits ? "rule" : "SHORT of the rule,",
            (double)PN_QWEN35_TT_TIER_HEADROOM / FIT_MIB);
}

void pn_qwen35_tt_fit_decide(pn_qwen35_tt *g, const pn_qwen35moe_config *cfg, int n_ssm,
                             int n_attn)
{
    uint64_t free_bytes[PN_TT_FIT_MAX_BANKS], card = 0u, weights = 0u, joint_q = 0u;
    const uint32_t banks = pn_tt_device_dram_free_banks(g->device, free_bytes,
                                                        PN_TT_FIT_MAX_BANKS);
    const uint32_t head_tiles = (uint32_t)(cfg->head_dim + 31) / 32u;
    pn_qwen35_tt_fit_shape shape;
    pn_qwen35_tt_fit_floor floor;
    uint32_t low_bank = 0u;
    int64_t low;
    bool ckpt;

    g->fit_policy = false;
    g->fit_levers = false;
    if (banks == 0u || banks > PN_QWEN35_TT_FIT_MAX_BANKS ||
        pn_tt_device_artifact_matmul_bytes(g->device, ".attn_q.weight", &weights, &joint_q) != 0)
        return;
    for (uint32_t b = 0; b < banks; b++)
        card += free_bytes[b];
    shape = fit_shape(g, cfg, n_ssm, n_attn, banks);
    pn_qwen35_tt_fit_floor_compute(&shape, weights, joint_q, &floor);
    /* The checkpoint is reserved at load under either placement (reserve_checkpoint). */
    ckpt = shape.gdn_on_card && pn_qwen35_tt_ckpt_enabled();
    if (!pn_qwen35_tt_fit_policy_wanted_banks(&shape, &floor, ckpt, free_bytes, banks))
        return;
    low = pn_qwen35_tt_fit_bank_low(&shape, &floor, ckpt, free_bytes, banks, &low_bank);
    fprintf(stderr, "qwen35_tt: main's placement does not fit this card at ctx %u — its floor "
                    "is %.3f GiB of %.3f GiB free (weights %.3f, second attn_q %.3f, KV %.3f, "
                    "gdn %.3f, gdnc %.3f, tall %.3f, reserve %.3f, MTP %.3f, gate tables "
                    "%.3f), and leaves DRAM "
                    "bank %u ~%.1f MiB (checkpoint %s); the rules are %.3f GiB of headroom and "
                    "%.0f MiB a bank. FIT POLICY on: attn_q uploaded as its halves, tall strips "
                    "mirrors first, no tall attn_qg\n", shape.ctx, (double)floor.total / FIT_GIB,
            (double)card / FIT_GIB, (double)floor.weights / FIT_GIB,
            (double)floor.second_q / FIT_GIB, (double)floor.kv / FIT_GIB,
            (double)floor.gdn / FIT_GIB, (double)floor.gdnc / FIT_GIB,
            (double)floor.tall / FIT_GIB, (double)floor.reserve / FIT_GIB,
            (double)floor.mtp / FIT_GIB, (double)floor.gate_tables / FIT_GIB, low_bank, (double)low / FIT_MIB, ckpt ? "in" : "off",
            (double)PN_QWEN35_TT_FIT_HEADROOM / FIT_GIB,
            (double)PN_QWEN35_TT_TIER_HEADROOM / FIT_MIB);
    g->fit_policy = true;
    fit_decide_levers(g, &shape, weights, joint_q, ckpt, free_bytes, banks);
    if (!shape.attn_on_card)
        return;
    /* A shape the rule refuses keeps the joint upload; the attention arm stages the halves. */
    if (pn_tt_device_split_weights_at_load(g->device, ".attn_q.weight", (uint32_t)cfg->n_head,
                                           2u * head_tiles, head_tiles) != 0)
        fprintf(stderr, "qwen35_tt: attn_q is uploaded whole (its halves are staged later)\n");
}

/** Print a shortfall: `what`, the bank (or "any bank"), and by how much. */
static void reserve_report_short(const char *what, const pn_tt_bank_need *need,
                                 uint32_t short_bank, uint64_t short_bytes)
{
    char where[32];

    if (short_bank < need->bank_count)
        snprintf(where, sizeof(where), "bank %u", short_bank);
    else
        snprintf(where, sizeof(where), "any bank");
    fprintf(stderr, "qwen35_tt: %s needs %.1f MiB of card DRAM and %s is %.1f MiB short\n",
            what, (double)pn_tt_bank_need_total(need) / FIT_MIB, where,
            (double)short_bytes / FIT_MIB);
}

/** The prefix checkpoint's twins, counted first and allocated only when every bank holds them. */
static void reserve_checkpoint(pn_qwen35_tt *g, uint64_t *free_bytes, uint32_t banks)
{
    pn_tt_bank_need need;
    uint32_t short_bank = 0u;
    uint64_t short_bytes = 0u;

    g->ckpt_off = false;
    if (!g->gdn_on_card || !pn_qwen35_tt_ckpt_enabled())
        return;
    pn_tt_bank_need_init(&need, banks);
    pn_tt_gdn_checkpoint_need(g->device, &need);
    if (pn_tt_bank_need_fits(&need, free_bytes, &short_bank, &short_bytes) &&
        pn_tt_gdn_checkpoint_reserve(g->device) == 0) {
        pn_tt_device_report_dram_stage(g->device, "reserve: prefix checkpoint");
        return;
    }
    reserve_report_short("the prefix checkpoint", &need, short_bank, short_bytes);
    fprintf(stderr, "qwen35_tt: prefix checkpoint OFF at this context — a resumed turn "
                    "re-prefills its whole conversation. Output is unaffected.\n");
    g->ckpt_off = true;
}

/** Under the fit policy, the dense MTP head's verify slots, counted and allocated at load (main
 *  allocates them at the first verify, where a bank already full fails it). The verify is then
 *  capped at these rows, the serve's prompt-lookup drafts included (lossless: fewer drafts
 *  verified). The verify takes the COMPACT slot map here — rows - 1 slots, the last row
 *  writing the buffer of the state the verify started from, which nothing reads again
 *  (pn_tt_gdn_verify_slots.h). Short: MTP OFF, plain decode. */
static void reserve_mtp_slots(pn_qwen35_tt *g, uint64_t *free_bytes, uint32_t banks)
{
    const uint32_t rows = pn_qwen35_tt_mtp_verify_rows();
    uint32_t slots;
    pn_tt_bank_need need;
    uint32_t short_bank = 0u;
    uint64_t short_bytes = 0u;
    char what[96];

    if (!g->fit_policy || g->moe || !g->mtp.on || !g->gdn_on_card)
        return;
    pn_tt_gdn_verify_compact_set(g->device, true);
    slots = pn_tt_gdn_verify_slots(g->device, rows);
    pn_tt_bank_need_init(&need, banks);
    pn_tt_gdn_snapshot_need(g->device, slots, &need);
    if (pn_tt_bank_need_fits(&need, free_bytes, &short_bank, &short_bytes) &&
        pn_tt_gdn_snapshot_reserve(g->device, slots) == 0) {
        g->spec_rows_cap = rows;
        fprintf(stderr, "qwen35_tt: MTP verify — %u GDN snapshot slots reserved at load (%.1f "
                        "MiB, compact map: %u rows); verifies capped at %u rows\n", slots,
                (double)pn_tt_bank_need_total(&need) / FIT_MIB, rows, rows);
        pn_tt_device_report_dram_stage(g->device, "reserve: MTP verify slots");
        (void)pn_tt_device_dram_free_banks(g->device, free_bytes, PN_TT_FIT_MAX_BANKS);
        return;
    }
    snprintf(what, sizeof(what), "the MTP verify (%u GDN snapshot slots)", slots);
    reserve_report_short(what, &need, short_bank, short_bytes);
    fprintf(stderr, "qwen35_tt: MTP OFF at this context — plain decode. Lower --ctx-cap or "
                    "PN_TT_SPEC. Output is unaffected.\n");
    pn_tt_gdn_verify_compact_set(g->device, false);
    g->mtp.on = false;
}

uint64_t pn_qwen35_tt_gate_tables_keep_free(pn_qwen35_tt *g)
{
    uint64_t keep = PN_QWEN35_TT_TIER_HEADROOM;

    /* Under main's placement the MTP verify's snapshot slots come at the first verify, after
     * the tables: a bank keeps their share too. (Under the fit policy they are already
     * reserved — reserve_mtp_slots.) */
    if (!g->fit_policy && !g->moe && g->mtp.on && g->gdn_on_card) {
        uint64_t free_bytes[PN_TT_FIT_MAX_BANKS], widest = 0u;
        const uint32_t banks = pn_tt_device_dram_free_banks(g->device, free_bytes,
                                                            PN_TT_FIT_MAX_BANKS);
        pn_tt_bank_need need;

        pn_tt_bank_need_init(&need, banks);
        pn_tt_gdn_snapshot_need(g->device,
                                pn_tt_gdn_verify_slots(g->device, pn_qwen35_tt_mtp_verify_rows()),
                                &need);
        for (uint32_t b = 0; b < banks; b++)
            if (need.bank[b] > widest)
                widest = need.bank[b];
        keep += widest + (banks ? (need.any + banks - 1u) / banks : 0u);
    }
    return keep;
}

int pn_qwen35_tt_prefill_reserve(pn_qwen35_tt *g)
{
    uint64_t free_bytes[PN_TT_FIT_MAX_BANKS];
    const uint32_t banks = pn_tt_device_dram_free_banks(g->device, free_bytes,
                                                        PN_TT_FIT_MAX_BANKS);
    pn_tt_bank_need need;
    uint32_t short_bank = 0u;
    uint64_t short_bytes = 0u;

    if (banks == 0u)
        return -1;
    pn_tt_bank_need_init(&need, banks);
    pn_tt_gdnc_prefill_need(g->device, &need);
    pn_tt_rope_prefill_need(g->device, &need);
    if (!pn_tt_bank_need_fits(&need, free_bytes, &short_bank, &short_bytes)) {
        reserve_report_short("the batched prefill's own buffers (GDNC state split, PIPE 2 "
                             "streams, tall rope tables)", &need, short_bank, short_bytes);
        fprintf(stderr, "qwen35_tt: REFUSED — this context cannot run its first prefill on "
                        "this card. Lower --ctx-cap.\n");
        return -1;
    }
    if (pn_tt_gdnc_reserve_prefill(g->device) != 0 || pn_tt_rope_reserve_prefill(g->device) != 0) {
        fprintf(stderr, "qwen35_tt: REFUSED — a counted prefill buffer did not allocate\n");
        return -1;
    }
    pn_tt_device_report_dram_stage(g->device, "reserve: prefill buffers");
    if (pn_tt_device_dram_free_banks(g->device, free_bytes, PN_TT_FIT_MAX_BANKS) != banks)
        return -1;
    reserve_mtp_slots(g, free_bytes, banks);
    reserve_checkpoint(g, free_bytes, banks);
    return 0;
}

#endif /* USE_TT */
