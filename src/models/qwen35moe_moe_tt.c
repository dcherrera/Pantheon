/* SPDX-License-Identifier: AGPL-3.0-or-later */
/* Copyright (C) 2026 David (Corey) Herrera */
/**
 * @file qwen35moe_moe_tt.c
 * @brief The Qwen3.6 MoE block in the resident TT forward: router, routed gate/up, SwiGLU,
 *        routing-weight scale, routed down + weighted sum, the shared expert and its sigmoid
 *        gate — every step on the card.
 *
 * One layer, one decode row, in the CPU reference's order (qwen35moe_moe.c):
 *   1. route: logits = ffn_gate_inp . x (fp32), top-k, renormalised det-exp weights, the
 *      shared gate sigmoid(ffn_gate_inp_shexp . x) — the router residency, one descriptor;
 *   2. gate/up: the k routed experts (expert residencies, indices read on the card) and the
 *      shared expert (the decode matmul group) into ONE strip each, rank r at tiles
 *      [r * E, (r + 1) * E), the shared expert after the k ranks;
 *   3. h = silu(gate) * up over the whole strip, then h * scale (the router's scale strip:
 *      w_r per rank, the shared gate after) — two elementwise descriptors;
 *   4. routed down + weighted sum: ONE K-concatenated descriptor per member, sum over ranks in
 *      rank order in DST; the shared down on the decode group;
 *   5. project = shared + routed. The caller adds the residual.
 * The card computes w_r * h_r in bf16 before the down projection where the CPU scales the
 * fp32 down output; that is card arithmetic, measured against the oracle (spec §8 B).
 *
 * @status Decode (one row). Prefill (5.6) and verify rows (5.5) are not built: the batched
 *         prefill and speculative verify decline on a MoE model.
 * @issues The shared + routed sum is a third add a layer (fusable).
 * @todo None
 */

#ifdef USE_TT

#include "qwen35moe_tt_internal.h"

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "../accel/tt/pn_tt_matmul_group.h"
#include "../accel/tt/pn_tt_moe.h"
#include "tt/kernels/resident_matmul_moe_protocol.h"
#include "../core/qweight.h"
#include "../memory/arena.h"
#include "qwen35moe_routing_dump.h"

/** One layer's routed experts and router on the card. */
struct pn_qwen35_tt_moe_layer {
    pn_tt_moe_stack gate, up, down;
    pn_tt_moe_router router;
};

/**
 * THE PER-LAYER GATE (`PN_TT_MOE_CHECK=1`): after every MoE layer the card's input row, its
 * routing slot and its output come back and the CPU card-format oracle (the host model over
 * the same artifact, `PN_FTTW_ORACLE`) runs the same block on the SAME bf16 input. Reported
 * per layer: routing set agreement with the oracle's top-k margin, routing-weight error and
 * the output's relative L2 error; and a digest of every card output, which two green runs
 * must share and every red must move. Slow (the oracle is a host CPU block): gates only.
 */
typedef struct {
    pn_qwen35moe_model *m;
    pn_arena *arena;
    float *x, *card, *host, *logits;
    uint64_t layers, routing_equal, routing_near_tie, routing_flip;
    uint64_t digest;
    double worst_rel, sum_rel, worst_weight;
    uint64_t tokens;
} pn_qwen35_tt_moe_check;

/** The block's strips: gate / up / h / scaled are k*E + S tiles (E expert ff, S shared ff,
 *  in tiles); shared and routed are n_embd wide. */
struct pn_qwen35_tt_moe_strips {
    pn_tt_activation gate, up, h, scaled;
    pn_tt_activation shared, routed;
    uint32_t routed_tiles; /**< k * E */
    pn_qwen35_tt_moe_check *check; /**< NULL unless PN_TT_MOE_CHECK=1 */
    bool verify_group;             /**< PN_TT_MOE_GROUP_VERIFY=1 (diagnostic) */
    uint64_t group_checks, group_incomplete;
    /* BATCHED PREFILL (plan S1): one group's LIST-mode gate / up outputs (256 * E tiles x
     * `prefill_bands` bands; SwiGLU and scale run in place in `prefill_gate`) and the pass's
     * routed output (n_embd / 32 tiles x the tall height). */
    bool prefill;
    uint32_t prefill_bands;
    pn_tt_activation prefill_gate, prefill_up, prefill_routed;
};

/** A routing set's margin: the k-th selected probability minus the best unselected one. */
#define PN_QWEN35_TT_MOE_NEAR_TIE 1e-3

uint32_t pn_qwen35_tt_moe_eltwise_tiles(const pn_qwen35moe_config *cfg)
{
    return (uint32_t)(cfg->n_expert_used * cfg->expert_ff + cfg->shared_ff) / 32u;
}

/** The strip `s` seen from tile `first` for `count` tiles, band 0 only, primary copy. */
static pn_tt_activation moe_view(const pn_tt_activation *s, uint32_t first, uint32_t count)
{
    pn_tt_activation v = *s;

    v.address = s->address + first * 2048u;
    v.tile_count = count;
    v.replica_count = 1u;
    v.replica_bank[0] = v.bank;
    v.replica_address[0] = v.address;
    v.tile_rows = 1u;
    v.active_rows = 1u;
    v.row_offset = 0u;
    return v;
}

/** The shared gate vector as floats (F32 in the GGUF, or the loader's widened copy). */
static const float *moe_shexp_gate(const pn_qwen35moe_moe *moe)
{
    if (moe->gate_inp_shexp_f32)
        return moe->gate_inp_shexp_f32;
    if (moe->ffn_gate_inp_shexp.type != GGUF_QUANT_F32)
        return NULL;
    return (const float *)moe->ffn_gate_inp_shexp.data;
}

static const pn_qwen35moe_moe *moe_of_layer(pn_qwen35moe_model *m, int layer)
{
    const int idx = pn_qwen35moe_model_layer_index(m, layer);

    if (pn_qwen35moe_model_layer_type(m, layer) == PN_QWEN35MOE_LAYER_ATTN) {
        const pn_qwen35moe_attn_layer *l = pn_qwen35moe_model_attn_layer(m, idx);

        return l ? &l->moe : NULL;
    }
    {
        const pn_qwen35moe_ssm_layer *l = pn_qwen35moe_model_ssm_layer(m, idx);

        return l ? &l->moe : NULL;
    }
}

int pn_qwen35_tt_moe_bind(pn_qwen35_tt *g, pn_qwen35moe_model *m, const pn_qwen35moe_config *cfg)
{
    char name[128];
    int layer;

    if (cfg->expert_weights_scale != 1.0f) {
        fprintf(stderr, "qwen35_tt: expert_weights_scale %g is not folded into the router "
                        "writer's weights — refused\n", (double)cfg->expert_weights_scale);
        return -1;
    }
    g->moe_model = m;
    /* One slot more than the trunk: the MTP head's MoE layer (pn_qwen35_tt_moe_bind_mtp). */
    g->moe_layers = calloc((size_t)cfg->n_layer + 1u, sizeof(*g->moe_layers));
    if (!g->moe_layers)
        return -1;
    for (layer = 0; layer < cfg->n_layer; layer++) {
        struct pn_qwen35_tt_moe_layer *e = &g->moe_layers[layer];
        const pn_qwen35moe_moe *moe = moe_of_layer(m, layer);
        const float *shexp = moe ? moe_shexp_gate(moe) : NULL;
        struct { const char *suffix; pn_tt_moe_stack *stack; } stacks[] = {
            { "ffn_gate_exps", &e->gate }, { "ffn_up_exps", &e->up },
            { "ffn_down_exps", &e->down },
        };

        if (!shexp) {
            fprintf(stderr, "qwen35_tt: layer %d has no F32 shared-expert gate\n", layer);
            return -1;
        }
        for (size_t i = 0; i < sizeof(stacks) / sizeof(stacks[0]); i++) {
            snprintf(name, sizeof(name), "blk.%d.%s.weight", layer, stacks[i].suffix);
            if (pn_tt_moe_place_stack(g->device, name, stacks[i].stack) != 0)
                return -1;
        }
        snprintf(name, sizeof(name), "blk.%d.ffn_gate_inp.weight", layer);
        if (pn_tt_moe_place_router(g->device, name, shexp, &e->router) != 0)
            return -1;
        if (e->gate.experts != (uint32_t)cfg->n_expert ||
            e->router.tile_cols != (uint32_t)cfg->n_expert / 32u + 1u ||
            e->gate.tile_cols * 32u != (uint32_t)cfg->expert_ff ||
            e->down.tile_cols * 32u != (uint32_t)cfg->n_embd) {
            fprintf(stderr, "qwen35_tt: layer %d's expert stacks disagree with the config "
                            "(%u experts, ff %u, down out %u)\n", layer, e->gate.experts,
                    e->gate.tile_cols * 32u, e->down.tile_cols * 32u);
            return -1;
        }
    }
    fprintf(stderr, "qwen35_tt: MoE — %d layers x 3 expert stacks (FOLD layout) + router placed; "
                    "fold + upload %.1f s\n", cfg->n_layer, pn_tt_moe_fold_seconds(g->device));
    pn_tt_device_report_dram_stage(g->device, "MoE experts + routers placed");
    return 0;
}

int pn_qwen35_tt_moe_bind_mtp(pn_qwen35_tt *g, const pn_qwen35moe_config *cfg)
{
    struct pn_qwen35_tt_moe_layer *e;
    const float *shexp;
    struct { const char *name; pn_tt_moe_stack *stack; } stacks[3];

    if (!g || !g->moe_layers || !cfg)
        return -1;
    e = &g->moe_layers[cfg->n_layer];
    stacks[0].name = "mtp.ffn_gate_exps.weight";
    stacks[0].stack = &e->gate;
    stacks[1].name = "mtp.ffn_up_exps.weight";
    stacks[1].stack = &e->up;
    stacks[2].name = "mtp.ffn_down_exps.weight";
    stacks[2].stack = &e->down;
    shexp = pn_tt_device_extra_vector(g->device, "mtp.ffn_gate_inp_shexp.weight",
                                      (uint32_t)cfg->n_embd);
    if (!shexp) {
        fprintf(stderr, "qwen35_tt: the MTP artifact has no F32 shared-expert gate\n");
        return -1;
    }
    for (size_t i = 0; i < 3; i++)
        if (pn_tt_moe_place_extra_stack(g->device, stacks[i].name, stacks[i].stack) != 0)
            return -1;
    if (pn_tt_moe_place_extra_router(g->device, "mtp.ffn_gate_inp.weight", shexp, &e->router) != 0)
        return -1;
    if (e->gate.experts != (uint32_t)cfg->n_expert ||
        e->router.tile_cols != (uint32_t)cfg->n_expert / 32u + 1u ||
        e->gate.tile_cols * 32u != (uint32_t)cfg->expert_ff ||
        e->down.tile_cols * 32u != (uint32_t)cfg->n_embd ||
        e->gate.stride != g->moe_layers[0].gate.stride) {
        fprintf(stderr, "qwen35_tt: the MTP head's expert stacks disagree with the trunk's\n");
        return -1;
    }
    g->moe_mtp = true;
    fprintf(stderr, "qwen35_tt: MTP head's MoE layer placed (3 expert stacks + router, %.1f s "
                    "fold + upload so far)\n", pn_tt_moe_fold_seconds(g->device));
    return 0;
}

/** The batched prefill's group height: `PN_TT_MOE_PREFILL_BANDS` (1..4, default 4). */
static uint32_t moe_prefill_bands(void)
{
    const char *env = getenv("PN_TT_MOE_PREFILL_BANDS");
    const long bands = env && *env ? atol(env) : (long)FT_TT_MOE_ROWS_BANDS_MAX;

    /* 1, 2 or 4 (see pn_qwen35_tt_moe_prefill_ffn: a 3-band chunk does not tile the ring). */
    return bands == 1 || bands == 2 ? (uint32_t)bands : FT_TT_MOE_ROWS_BANDS_MAX;
}

/**
 * BATCHED PREFILL STRIPS (plan S1), only when `PN_TT_MOE_PREFILL` is not 0 (default on): two
 * group strips of 256 * E tiles x the group height, and the pass's routed output strip at
 * the tall height (64 bands). ~2 x 32 + 8 MiB at the defaults.
 */
static int moe_prefill_arm(pn_qwen35_tt *g, const pn_qwen35moe_config *cfg,
                           struct pn_qwen35_tt_moe_strips *s)
{
    const char *env = getenv("PN_TT_MOE_PREFILL");
    const uint32_t wide = FT_TT_MOE_LIST_MAX * (uint32_t)cfg->expert_ff / 32u;

    s->prefill = !(env && *env && atoi(env) == 0);
    if (!s->prefill)
        return 0;
    s->prefill_bands = moe_prefill_bands();
    if (pn_tt_activation_alloc_rows(g->device, wide, s->prefill_bands, &s->prefill_gate) != 0 ||
        pn_tt_activation_alloc_rows(g->device, wide, s->prefill_bands, &s->prefill_up) != 0 ||
        pn_tt_activation_alloc_rows(g->device, (uint32_t)cfg->n_embd / 32u,
                                    FT_TT_MOE_SHEXP_BANDS_MAX, &s->prefill_routed) != 0) {
        fprintf(stderr, "qwen35_tt: no DRAM for the MoE batched-prefill strips — prompts run "
                        "one position at a time\n");
        s->prefill = false;
        return 0;
    }
    fprintf(stderr, "qwen35_tt: MoE batched prefill ARMED — groups of %u bands (%u rows), LIST "
                    "mode over 256 entries\n", s->prefill_bands, s->prefill_bands * 32u);
    return 0;
}

bool pn_qwen35_tt_moe_prefill_ready(const pn_qwen35_tt *g)
{
    return g && g->moe && g->moe_strips && g->moe_strips->prefill;
}

int pn_qwen35_tt_moe_arm(pn_qwen35_tt *g, const pn_qwen35moe_config *cfg)
{
    struct pn_qwen35_tt_moe_strips *s;
    pn_tt_moe_shape shape;
    const uint32_t wide = pn_qwen35_tt_moe_eltwise_tiles(cfg);
    const uint32_t embd = (uint32_t)cfg->n_embd / 32u;

    if (!g->moe_layers)
        return -1;
    memset(&shape, 0, sizeof(shape));
    shape.hidden_tiles = embd;
    shape.experts = (uint32_t)cfg->n_expert;
    shape.k = (uint32_t)cfg->n_expert_used;
    shape.expert_tiles = (uint32_t)cfg->expert_ff / 32u;
    shape.shexp_tiles = (uint32_t)cfg->shared_ff / 32u;
    shape.stride = g->moe_layers[0].gate.stride;
    if (pn_tt_moe_arm(g->device, &shape) != 0)
        return -1;
    s = calloc(1, sizeof(*s));
    if (!s)
        return -1;
    g->moe_strips = s;
    s->routed_tiles = shape.k * shape.expert_tiles;
    if (pn_tt_activation_alloc(g->device, wide, &s->gate) != 0 ||
        pn_tt_activation_alloc(g->device, wide, &s->up) != 0 ||
        pn_tt_activation_alloc(g->device, wide, &s->h) != 0 ||
        pn_tt_activation_alloc(g->device, wide, &s->scaled) != 0 ||
        pn_tt_activation_alloc(g->device, embd, &s->shared) != 0 ||
        pn_tt_activation_alloc(g->device, embd, &s->routed) != 0)
        return -1;
    g->moe = true;
    s->verify_group = getenv("PN_TT_MOE_GROUP_VERIFY") && atoi(getenv("PN_TT_MOE_GROUP_VERIFY")) != 0;
    if (moe_prefill_arm(g, cfg, s) != 0)
        return -1;
    pn_tt_device_report_dram_stage(g->device, "arm: MoE block");
    if (getenv("PN_TT_MOE_CHECK") && atoi(getenv("PN_TT_MOE_CHECK")) == 1 && g->moe_model) {
        pn_qwen35_tt_moe_check *c = calloc(1, sizeof(*c));

        if (!c)
            return -1;
        s->check = c;
        c->m = g->moe_model;
        c->arena = pn_arena_new(16u << 20);
        c->x = calloc((size_t)cfg->n_embd, sizeof(float));
        c->card = calloc((size_t)cfg->n_embd, sizeof(float));
        c->host = calloc((size_t)cfg->n_embd, sizeof(float));
        c->logits = calloc((size_t)cfg->n_expert, sizeof(float));
        c->digest = 1469598103934665603ull;
        if (!c->arena || !c->x || !c->card || !c->host || !c->logits)
            return -1;
        fprintf(stderr, "qwen35_tt: PN_TT_MOE_CHECK=1 — every MoE layer is checked against the "
                        "CPU card-format oracle (slow; a gate, not a run)\n");
    }
    return 0;
}

/** FNV-1a over the card output's bytes. */
static uint64_t moe_digest(uint64_t h, const float *v, int n)
{
    const unsigned char *b = (const unsigned char *)v;

    for (size_t i = 0; i < (size_t)n * sizeof(float); i++)
        h = (h ^ b[i]) * 1099511628211ull;
    return h;
}

/** The oracle's top-k over softmax probabilities (strict >, lowest index on a tie) and its
 *  margin: k-th selected minus best unselected. */
static double moe_host_topk(const float *logits, int n, int k, uint32_t *idx, float *weight)
{
    double max = logits[0], sum = 0.0, wsum = 0.0, best_out = 0.0;
    double p[1024];
    bool taken[1024] = { false };

    if (n > 1024)
        return 0.0;
    for (int e = 1; e < n; e++)
        if (logits[e] > max)
            max = logits[e];
    for (int e = 0; e < n; e++) {
        p[e] = exp((double)logits[e] - max);
        sum += p[e];
    }
    for (int r = 0; r < k; r++) {
        int best = -1;

        for (int e = 0; e < n; e++)
            if (!taken[e] && (best < 0 || p[e] > p[best]))
                best = e;
        taken[best] = true;
        idx[r] = (uint32_t)best;
        wsum += p[best];
    }
    for (int r = 0; r < k; r++)
        weight[r] = (float)(p[idx[r]] / wsum);
    for (int e = 0; e < n; e++)
        if (!taken[e] && p[e] > best_out)
            best_out = p[e];
    return (p[idx[k - 1]] - best_out) / sum;
}

/** Score one layer's card output against the oracle on the same input. */
static int moe_check_layer(pn_qwen35_tt *g, const pn_qwen35moe_config *cfg, int layer)
{
    pn_qwen35_tt_moe_check *c = g->moe_strips->check;
    const pn_qwen35moe_moe *moe = moe_of_layer(c->m, layer);
    uint32_t card_idx[PN_TT_MOE_MAX_K], host_idx[PN_TT_MOE_MAX_K];
    float card_w[PN_TT_MOE_MAX_K], host_w[PN_TT_MOE_MAX_K];
    const int n = cfg->n_embd, k = cfg->n_expert_used;
    double err = 0.0, norm = 0.0, margin, wdiff = 0.0;
    int same = 0, attached;

    if (!moe ||
        pn_tt_activation_download(g->device, &g->card.normed, c->x, (uint32_t)n) != 0 ||
        pn_tt_activation_download(g->device, &g->card.project, c->card, (uint32_t)n) != 0 ||
        pn_tt_moe_read_slot(g->device, card_idx, card_w) != 0)
        return -1;
    attached = pn_qwen35moe_model_accel_mv_enable(c->m, 0);
    pn_arena_reset(c->arena);
    pn_qmv(&moe->ffn_gate_inp, c->x, c->logits);
    margin = moe_host_topk(c->logits, cfg->n_expert, k, host_idx, host_w);
    if (pn_qwen35moe_moe_block(moe, cfg, c->x, c->arena, c->host, NULL) != 0) {
        if (attached == 1)
            (void)pn_qwen35moe_model_accel_mv_enable(c->m, 1);
        return -1;
    }
    if (attached == 1)
        (void)pn_qwen35moe_model_accel_mv_enable(c->m, 1);
    for (int r = 0; r < k; r++)
        for (int q = 0; q < k; q++)
            if (card_idx[r] == host_idx[q]) {
                same++;
                if (fabs((double)card_w[r] - host_w[q]) > wdiff)
                    wdiff = fabs((double)card_w[r] - host_w[q]);
            }
    for (int j = 0; j < n; j++) {
        err += ((double)c->card[j] - c->host[j]) * ((double)c->card[j] - c->host[j]);
        norm += (double)c->host[j] * c->host[j];
    }
    err = norm > 0.0 ? sqrt(err / norm) : 0.0;
    c->layers++;
    if (same == k)
        c->routing_equal++;
    else if (margin < PN_QWEN35_TT_MOE_NEAR_TIE)
        c->routing_near_tie++;
    else
        c->routing_flip++;
    if (same == k) {
        c->sum_rel += err;
        if (err > c->worst_rel)
            c->worst_rel = err;
        if (wdiff > c->worst_weight)
            c->worst_weight = wdiff;
    }
    c->digest = moe_digest(c->digest, c->card, n);
    fprintf(stderr, "moe_check t%llu L%d: routing %d/%d (margin %.2e)%s, weights max |d| %.2e, "
                    "out relL2 %.3e\n", (unsigned long long)c->tokens, layer, same, k, margin,
            same == k ? "" : (margin < PN_QWEN35_TT_MOE_NEAR_TIE ? " NEAR-TIE" : " FLIP"),
            wdiff, err);
    return 0;
}

/** The gate's summary line (at release). */
static void moe_check_report(const pn_qwen35_tt_moe_check *c)
{
    if (!c || c->layers == 0u)
        return;
    fprintf(stderr, "moe_check SUMMARY: %llu layer-rows, routing equal %llu, near-tie %llu, "
                    "FLIP %llu; on equal routing out relL2 mean %.3e worst %.3e, weights worst "
                    "%.2e; digest %016llx\n",
            (unsigned long long)c->layers, (unsigned long long)c->routing_equal,
            (unsigned long long)c->routing_near_tie, (unsigned long long)c->routing_flip,
            c->routing_equal ? c->sum_rel / (double)c->routing_equal : 0.0, c->worst_rel,
            c->worst_weight, (unsigned long long)c->digest);
}

/**
 * `QWEN35_ROUTING_DUMP=<file>` on the card: the CARD's selection in the CPU oracle's record
 * (qwen35moe_routing_dump.h) — indices from the routing slot, probabilities from the card's own
 * fp32 logits — so qwen36_oracle_compare scores card routing against the oracle's. Reads back
 * two small buffers a layer: a gate instrument, not a run.
 */
static int moe_routing_dump(pn_qwen35_tt *g, const pn_qwen35moe_config *cfg)
{
    static int armed = -1;
    float logits[1024 + 32], probs[1024], weights[PN_TT_MOE_MAX_K];
    uint32_t idx[PN_TT_MOE_MAX_K];
    int top[PN_TT_MOE_MAX_K];
    const int n = cfg->n_expert;
    double max, sum = 0.0;

    if (armed < 0)
        armed = getenv("QWEN35_ROUTING_DUMP") && *getenv("QWEN35_ROUTING_DUMP") ? 1 : 0;
    if (!armed)
        return 0;
    if (n > 1024 || pn_tt_moe_read_slot(g->device, idx, weights) != 0 ||
        pn_tt_moe_read_logits(g->device, logits, (uint32_t)(n / 32 + 1) * 32u) != 0)
        return -1;
    max = logits[0];
    for (int e = 1; e < n; e++)
        if (logits[e] > max)
            max = logits[e];
    for (int e = 0; e < n; e++)
        sum += exp((double)logits[e] - max);
    for (int e = 0; e < n; e++)
        probs[e] = (float)(exp((double)logits[e] - max) / sum);
    for (int r = 0; r < cfg->n_expert_used; r++)
        top[r] = (int)idx[r];
    pn_qwen35moe_routing_dump(probs, n, top, cfg->n_expert_used);
    return 0;
}

int pn_qwen35_tt_moe_ffn(pn_qwen35_tt *g, const pn_qwen35_tt_layer *d, int layer)
{
    const struct pn_qwen35_tt_moe_layer *e = &g->moe_layers[layer];
    struct pn_qwen35_tt_moe_strips *s = g->moe_strips;
    const uint32_t r = s->routed_tiles;
    const uint32_t sh = s->gate.tile_count - r;
    pn_tt_activation shexp_gate = moe_view(&s->gate, r, sh);
    pn_tt_activation shexp_up = moe_view(&s->up, r, sh);
    pn_tt_activation shexp_scaled = moe_view(&s->scaled, r, sh);
    pn_tt_activation routed_scaled = moe_view(&s->scaled, 0u, r);

    if (pn_tt_moe_route(g->device, &e->router, &g->card.normed) != 0 ||
        moe_routing_dump(g, pn_qwen35moe_model_config(g->moe_model)) != 0)
        return -1;
    if (pn_qwen35_tt_project(g, &d->ffn_gate, &g->card.normed, &shexp_gate, false) != 0 ||
        pn_qwen35_tt_project(g, &d->ffn_up, &g->card.normed, &shexp_up, false) != 0 ||
        pn_tt_moe_gate_up(g->device, &e->gate, &e->up, &g->card.normed, &s->gate, &s->up) != 0 ||
        pn_tt_matvec_flush(g->device) != 0 || pn_tt_moe_await(g->device) != 0)
        return -1;
    /* PN_TT_MOE_PULL_VERIFY: is the decode group (the shared expert) really done here? */
    if (s->verify_group) {
        const uint32_t short_members = pn_tt_matmul_group_incomplete(g->device);

        s->group_checks++;
        s->group_incomplete += short_members != 0u;
    }
    if (pn_tt_silu_mul(g->device, &s->gate, &s->up, &s->h) != 0 ||
        pn_tt_mul(g->device, &s->h, pn_tt_moe_scale_strip(g->device), &s->scaled) != 0)
        return -1;
    if (pn_qwen35_tt_project(g, &d->ffn_down, &shexp_scaled, &s->shared, false) != 0 ||
        pn_tt_moe_down(g->device, &e->down, &routed_scaled, &s->routed) != 0 ||
        pn_tt_matvec_flush(g->device) != 0 || pn_tt_moe_await(g->device) != 0)
        return -1;
    if (pn_tt_add(g->device, &s->shared, &s->routed, &g->card.project) != 0)
        return -1;
    /* The host check knows the trunk's layers only (moe_layers[n_layer] is the MTP head's). */
    if (s->check && layer < pn_qwen35moe_model_config(g->moe_model)->n_layer) {
        if (layer == 0)
            s->check->tokens++;
        return moe_check_layer(g, pn_qwen35moe_model_config(s->check->m), layer);
    }
    return 0;
}

/** `strip` narrowed to bands [band, band + span): a copy, the caller's strip untouched. */
static int moe_band_view(const pn_tt_activation *strip, uint32_t band, uint32_t span,
                         pn_tt_activation *out)
{
    *out = *strip;
    return pn_tt_activation_seek(out, strip->row_offset + band) == 0 &&
                   pn_tt_activation_span(out, span) == 0
               ? 0
               : -1;
}

/** A contiguous region of the scale buffer seen as `tiles` x `bands` (tile-row-major). */
static pn_tt_activation moe_scale_view(const pn_tt_activation *buffer, uint32_t first_tile,
                                       uint32_t tiles, uint32_t bands)
{
    pn_tt_activation v = *buffer;

    v.address = buffer->address + first_tile * 2048u;
    v.tile_count = tiles;
    v.tile_rows = bands;
    v.active_rows = bands;
    v.row_offset = 0u;
    v.replica_count = 1u;
    v.replica_bank[0] = v.bank;
    v.replica_address[0] = v.address;
    return v;
}

/** One routed group: route its bands, LIST gate/up, SwiGLU x scale in place, LIST down. */
static int moe_prefill_group(pn_qwen35_tt *g, const struct pn_qwen35_tt_moe_layer *e,
                             uint32_t first, uint32_t bands)
{
    struct pn_qwen35_tt_moe_strips *s = g->moe_strips;
    const pn_tt_activation *buffer = pn_tt_moe_scale_buffer(g->device);
    pn_tt_activation normed, gate, up, routed, scale;

    if (!buffer || moe_band_view(&g->card.normed, first, bands, &normed) != 0 ||
        moe_band_view(&s->prefill_gate, 0u, bands, &gate) != 0 ||
        moe_band_view(&s->prefill_up, 0u, bands, &up) != 0 ||
        moe_band_view(&s->prefill_routed, first, bands, &routed) != 0)
        return -1;
    scale = moe_scale_view(buffer, 0u, gate.tile_count, bands);
    if (pn_tt_moe_route_rows(g->device, &e->router, &normed, bands, first) != 0 ||
        pn_tt_moe_gate_up_rows(g->device, &e->gate, &e->up, &normed, bands, FT_TT_MOE_LIST_MAX,
                               &gate, &up) != 0 ||
        pn_tt_moe_await(g->device) != 0 ||
        pn_tt_silu_mul(g->device, &gate, &up, &gate) != 0 ||
        pn_tt_mul(g->device, &gate, &scale, &gate) != 0 ||
        pn_tt_moe_down_rows(g->device, &e->down, &gate, bands, FT_TT_MOE_LIST_MAX, &routed) != 0 ||
        pn_tt_moe_await(g->device) != 0)
        return -1;
    return 0;
}

int pn_qwen35_tt_moe_prefill_ffn(pn_qwen35_tt *g, const pn_qwen35_tt_layer *d, int layer,
                                 uint32_t bands)
{
    struct pn_qwen35_tt_moe_strips *s = g ? g->moe_strips : NULL;
    const pn_tt_activation *buffer = pn_tt_moe_scale_buffer(g ? g->device : NULL);
    pn_qwen35_tt_strips *c = g ? &g->card : NULL;
    pn_tt_activation routed, shexp_scale;

    if (!s || !s->prefill || !buffer || bands == 0u || bands > FT_TT_MOE_SHEXP_BANDS_MAX)
        return -1;
    for (uint32_t first = 0u, span; first < bands; first += span) {
        /* A group is 1, 2 or 4 bands, never 3: the expert residencies' CB_IN0 moves `16 * MB`
         * pages a block, and every chunk size must tile the ring from wherever the last
         * descriptor left the write pointer (a multiple of 64 pages). */
        span = bands - first < s->prefill_bands ? bands - first : s->prefill_bands;
        if (span == 3u)
            span = 2u;
        if (moe_prefill_group(g, &g->moe_layers[layer], first, span) != 0)
            return -1;
    }
    /* The shared expert over the whole pass, its sigmoid gate a per-row scale (the router
     * wrote one tile set a band), then shared + routed. */
    shexp_scale = moe_scale_view(buffer, FT_TT_MOE_SHEXP_SCALE_TILE0, c->ffn_swiglu.tile_count,
                                 bands);
    if (moe_band_view(&s->prefill_routed, 0u, bands, &routed) != 0 ||
        pn_qwen35_tt_project(g, &d->ffn_gate, &c->normed, &c->ffn_gate, false) != 0 ||
        pn_qwen35_tt_project(g, &d->ffn_up, &c->normed, &c->ffn_up, false) != 0 ||
        pn_tt_matvec_flush(g->device) != 0 ||
        pn_tt_silu_mul(g->device, &c->ffn_gate, &c->ffn_up, &c->ffn_swiglu) != 0 ||
        pn_tt_mul(g->device, &c->ffn_swiglu, &shexp_scale, &c->ffn_swiglu) != 0 ||
        pn_qwen35_tt_project(g, &d->ffn_down, &c->ffn_swiglu, &c->project, true) != 0 ||
        pn_tt_matvec_flush(g->device) != 0 ||
        pn_tt_add(g->device, &c->project, &routed, &c->project) != 0)
        return -1;
    return 0;
}

int pn_qwen35_tt_moe_verify_ffn(pn_qwen35_tt *g, const pn_qwen35_tt_layer *d, int layer,
                                uint32_t rows)
{
    struct pn_qwen35_tt_moe_strips *s = g ? g->moe_strips : NULL;
    const pn_tt_activation *buffer = pn_tt_moe_scale_buffer(g ? g->device : NULL);
    pn_qwen35_tt_strips *c = g ? &g->card : NULL;
    const struct pn_qwen35_tt_moe_layer *e;
    pn_tt_activation normed, gate, up, routed, scale, shexp_scale;
    uint32_t entries, tiles;

    if (!s || !s->prefill || !buffer || rows == 0u || rows > FT_TT_MOE_CONCAT_ROWS_MAX)
        return -1;
    e = &g->moe_layers[layer];
    {
        const pn_qwen35moe_config *cfg = pn_qwen35moe_model_config(g->moe_model);

        entries = rows * (uint32_t)cfg->n_expert_used;
        tiles = entries * ((uint32_t)cfg->expert_ff / 32u);
    }
    /* One band: the verify rows. Every view is band 0 of its strip. */
    if (moe_band_view(&c->normed, 0u, 1u, &normed) != 0 ||
        moe_band_view(&s->prefill_routed, 0u, 1u, &routed) != 0)
        return -1;
    gate = moe_view(&s->prefill_gate, 0u, tiles);
    up = moe_view(&s->prefill_up, 0u, tiles);
    scale = moe_scale_view(buffer, 0u, tiles, 1u);
    shexp_scale = moe_scale_view(buffer, FT_TT_MOE_SHEXP_SCALE_TILE0, c->ffn_swiglu.tile_count, 1u);
    if (pn_tt_moe_route_concat(g->device, &e->router, &normed, rows) != 0 ||
        pn_tt_moe_gate_up_rows(g->device, &e->gate, &e->up, &normed, 1u, entries, &gate, &up) != 0 ||
        pn_tt_moe_await(g->device) != 0 || pn_tt_silu_mul(g->device, &gate, &up, &gate) != 0 ||
        pn_tt_mul(g->device, &gate, &scale, &gate) != 0 ||
        pn_tt_moe_down_rows(g->device, &e->down, &gate, 1u, entries, &routed) != 0 ||
        pn_tt_moe_await(g->device) != 0)
        return -1;
    /* The shared expert and shared + routed: the prefill pass's, at one band. RED
     * moe_verify_drop_routed: the verify rows lose their routed experts — any accepted draft
     * then rests on a wrong row, and speculation must DIFFER from plain decode. */
    const char *red = getenv("PN_TT_RED");
    const bool drop_routed = red && strcmp(red, "moe_verify_drop_routed") == 0;

    if (pn_qwen35_tt_project(g, &d->ffn_gate, &c->normed, &c->ffn_gate, false) != 0 ||
        pn_qwen35_tt_project(g, &d->ffn_up, &c->normed, &c->ffn_up, false) != 0 ||
        pn_tt_matvec_flush(g->device) != 0 ||
        pn_tt_silu_mul(g->device, &c->ffn_gate, &c->ffn_up, &c->ffn_swiglu) != 0 ||
        pn_tt_mul(g->device, &c->ffn_swiglu, &shexp_scale, &c->ffn_swiglu) != 0 ||
        pn_qwen35_tt_project(g, &d->ffn_down, &c->ffn_swiglu, &c->project, true) != 0 ||
        pn_tt_matvec_flush(g->device) != 0 ||
        (!drop_routed && pn_tt_add(g->device, &c->project, &routed, &c->project) != 0))
        return -1;
    return 0;
}

void pn_qwen35_tt_moe_release(pn_qwen35_tt *g)
{
    if (!g)
        return;
    if (g->moe_strips && g->moe_strips->check) {
        pn_qwen35_tt_moe_check *c = g->moe_strips->check;

        moe_check_report(c);
        pn_arena_free(c->arena);
        free(c->x);
        free(c->card);
        free(c->host);
        free(c->logits);
        free(c);
    }
    if (g->moe)
        pn_tt_moe_report(g->device, (double)g->tokens);
    if (g->moe_strips && g->moe_strips->verify_group)
        fprintf(stderr, "qwen35_tt: MoE GROUP VERIFY — the decode group (shared expert) had a member "
                        "short of its issued count after %llu of %llu gate/up phases\n",
                (unsigned long long)g->moe_strips->group_incomplete,
                (unsigned long long)g->moe_strips->group_checks);
    free(g->moe_layers);
    free(g->moe_strips);
    g->moe_layers = NULL;
    g->moe_strips = NULL;
    g->moe = false;
}

#endif /* USE_TT */
