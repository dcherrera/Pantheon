/* SPDX-License-Identifier: AGPL-3.0-or-later */
/* Copyright (C) 2026 David (Corey) Herrera */
/**
 * @file qwen35moe_forward_tt.c
 * @brief The `qwen35` hybrid stack — 24 Gated-Delta-Net layers and 8 full-attention
 *        layers — with its MATMULS on a Tenstorrent Blackhole card.
 *
 *        This is the first rung of the ladder, and the rung is chosen deliberately.
 *
 * ---------------------------------------------------------------------------
 * WHAT THIS MOVES, AND WHY IT IS EXACTLY THE MATMULS AND NOTHING ELSE
 * ---------------------------------------------------------------------------
 * `llama_forward_tt.c` moved a whole family at a time — norms, then SwiGLU, then the
 * attention triple — against a model whose every layer is the same shape. This model is
 * a HYBRID: layer `i` is full attention when `(i+1) % full_attention_interval == 0` and a
 * Gated-Delta-Net recurrence otherwise, and the recurrence carries 2 MiB of fp32 state per
 * layer from one token to the next. Moving the recurrence and the matmuls in one step
 * would give a wrong first answer several independent candidates at once, which is the
 * situation this project's own method exists to avoid: **one new thing at a time, against
 * a working oracle.**
 *
 * So the seam is `qwen35moe_mv()` (qwen35moe_internal.h) and nothing else changes. The
 * conv1d, the L2 norms, the GQA head repeat, the delta-rule recurrence, the gated norm,
 * the attention softmax and every RMSNorm run the SAME HOST CODE as the CPU oracle, on the
 * same fp32 state, in the same order. The two arms therefore differ in the matmuls alone,
 * and the only expected divergence is the one the weights carry: Q8_0 dequantised to fp32
 * on the host, against Bfp8_b tiles and a bf16 activation row on the card.
 *
 * ---------------------------------------------------------------------------
 * THE COST OF THIS RUNG, STATED BEFORE IT IS MEASURED
 * ---------------------------------------------------------------------------
 * `pn_tt_matvec()` takes a host row in and hands a host row back, so every one of this
 * model's **249 matmuls a token** crosses the link twice. Card->host is the slow direction
 * (~4.7 MB/s, no device->host DMA on Blackhole) and it is what will bind here — exactly as
 * it bound `llama_forward_tt.c` at the same rung, where 0.412 s of a 0.568 s token was the
 * link. That is not a defect of this file; it is the measurement that prices every step
 * after it, and it is why the next rung keeps the activation resident and the one after
 * that moves the recurrence on-card.
 *
 * 249 = 24 SSM layers x 5 (attn_qkv, attn_gate, ssm_alpha, ssm_beta, ssm_out)
 *     +  8 attention layers x 4 (attn_q, attn_k, attn_v, attn_output)
 *     + 32 layers x 3 (the dense SwiGLU: ffn_gate, ffn_up, ffn_down)
 *     +  1 LM head.
 * The count is checked at upload, because a binding that silently did not happen is a
 * matmul that silently ran on the CPU, and this project has twice had a card arm and a CPU
 * arm produce identical bytes.
 *
 * ---------------------------------------------------------------------------
 * WHAT IS DELIBERATELY NOT HERE
 * ---------------------------------------------------------------------------
 * - **`ssm_conv1d` is not bound and must not be.** It is a depthwise causal convolution,
 *   not a matvec: the artifact holds it as an 8192x4 matmul tile grid because the converter
 *   classifies every 2-D tensor, but multiplying an activation by it computes nothing this
 *   model wants. It stays on the host until there is a conv op.
 * - **MoE is refused, not attempted.** `n_expert > 0` is SABER, and its router and expert
 *   gather are a different phase. This file serves the DENSE branch (`dense_ffn`).
 * - **The embedding table is not on the card** for the reason `llama_forward_tt.c` gives:
 *   it is tiled block-float and no on-card op reads it.
 *
 * ---------------------------------------------------------------------------
 * AND THE RUNG ABOVE IT, WHICH IS NOW THE DEFAULT
 * ---------------------------------------------------------------------------
 * The projection-only arm above is still built, still selectable (`PN_TT_RESIDENT=0`) and
 * still the reference every measurement on this model was taken against. It is no longer
 * what runs: `qwen35moe_resident_tt.c` keeps the residual stream, both RMSNorms, the dense
 * SwiGLU and the LM head in card DRAM, which is 74.5% of that arm's readback. This file
 * owns the residency, the bindings, the counters and the choice between the two arms; that
 * one owns the forward.
 *
 * @status Two arms. The RESIDENT one (default) keeps activations in card DRAM and crosses
 *         the link only for the Gated-Delta-Net's and the attention core's operands. The
 *         PROJECTION-ONLY one (`PN_TT_RESIDENT=0`) round-trips every matvec and is the
 *         reference. Both refuse rather than falling back when the geometry, the
 *         architecture branch or a binding is not what they expect.
 * @issues None
 * @todo
 *   - [ ] The WHOLE Gated-Delta-Net block on the card — conv1d + SiLU, the L2 norms, the
 *         GQA repeat, the delta rule against a state resident in card DRAM, and the gated
 *         norm. It has to move as ONE piece: the conv1d is width-preserving so moving it
 *         alone changes nothing, and moving the L2 norm and the GQA repeat with it is
 *         WORSE than doing nothing (the repeat doubles q and k). This is the floor the
 *         resident arm cannot reach past.
 */

#ifdef USE_TT

#include <stdio.h>
#include <time.h>
#include <stdlib.h>
#include <string.h>

#include "qwen35moe.h"
#include "qwen35moe_internal.h"
#include "qwen35moe_tt_internal.h"

#include "../accel/tt/pn_tt_dispatch.h"
#include "../accel/tt/pn_tt_device.h"
#include "../accel/tt/pn_tt_matmul_group.h"
#include "../accel/tt/pn_tt_render.h"
#include "../core/bench.h"
#include "../core/engine.h"
#include "../core/qweight.h"
#include "tt/convert/ft_tt_artifact_cache.h"
#include "tt/ft_tt_memmap.h"

/* RED `spec_lookup_unverified` (serve): the next verify cycle accepts its first draft unverified
 * and wrong — armed only on a cycle whose drafts came from prompt lookup, so it DIFFERS only if
 * lookup drafts reach the output. */
static bool g_red_accept_next_cycle;

void pn_qwen35moe_tt_red_accept_next_cycle(void)
{
    g_red_accept_next_cycle = true;
}

/** An artifact to use INSTEAD of the cache. Same override semantics, and deliberately the
 *  same spelling, as the llama backend's: used verbatim, provenance unchecked, refuses
 *  rather than quietly reconverting. */
#define PN_QWEN35_TT_ARTIFACT_ENV "PN_TT_ARTIFACT"

/** Where cached artifacts live. Unset keeps them beside the checkpoint. */
#define PN_QWEN35_TT_CACHE_DIR_ENV "PN_TT_ARTIFACT_CACHE_DIR"

/** Force a rung by name. An instrument: a forced rung still has to fit. */
#define PN_QWEN35_TT_RUNG_ENV "PN_TT_RUNG"

/** Card DRAM held back from the weight budget, in MiB. */
#define PN_QWEN35_TT_RESERVE_MIB_ENV "PN_TT_CARD_RESERVE_MIB"
#define PN_QWEN35_TT_RESERVE_MIB_DEFAULT 512u

/** Report the per-token matmul split on stderr every N tokens. 0 (default) is silent. */
#define PN_QWEN35_TT_REPORT_ENV "PN_TT_REPORT"

/** A rung name that names nothing, kept distinct from "not set". */
#define PN_QWEN35_TT_ENV_BAD (-2)

/* ============================================================
 * The seam
 * ============================================================ */

/**
 * `qwen35moe_mv`'s backend: run this matvec on the card, or decline.
 *
 * Declining is CORRECT — the seam runs the host matvec — and that is precisely why every
 * decline is counted. A backend that quietly declined every weight would produce the right
 * answer at CPU speed and look like a working card.
 */
static int qwen35_tt_mv(void *ctx, const pn_qweight *w, const float *x, float *out)
{
    pn_qwen35_tt *g = (pn_qwen35_tt *)ctx;
    int i;

    if (!g || !g->device || !w)
        return -1;
    for (i = 0; i < g->n_bindings; i++) {
        if (g->bindings[i].key != w)
            continue;
        if (pn_tt_matvec(g->device, &g->bindings[i].card, x, out) != 0) {
            g->mv_failed++;
            if (!g->failure_reported) {
                g->failure_reported = 1;
                fprintf(stderr, "qwen35_tt: a card matvec FAILED and the host ran it "
                                "instead. The answer is still correct; the run is no "
                                "longer a measurement of the card.\n");
            }
            return -1;
        }
        g->mv_on_card++;
        return 0;
    }
    g->mv_on_host++;
    return -1;
}

/* ============================================================
 * Residency
 * ============================================================ */

/**
 * Bind one artifact tensor to one host weight, reporting a miss by name.
 *
 * ONE walk fills BOTH tables: the seam's `pn_qweight *`-keyed table for the projection-only
 * arm, and `card_out` — the per-layer handle the resident arm indexes by layer number. A
 * second walk would be a second place for a tensor name to be wrong, and only one of the two
 * would be checked against the count.
 */
static int qwen35_tt_bind(pn_qwen35_tt *g, const char *name, const pn_qweight *key,
                          pn_tt_weight *card_out)
{
    pn_qwen35_tt_binding *slot;

    if (g->n_bindings >= g->n_expected) {
        fprintf(stderr, "qwen35_tt: more bindings than the %d this geometry predicts — "
                        "the layer walk and the count disagree\n", g->n_expected);
        return -1;
    }
    slot = &g->bindings[g->n_bindings];
    if (pn_tt_device_find_weight(g->device, name, &slot->card) != 0) {
        fprintf(stderr, "qwen35_tt: the artifact has no %s\n", name);
        return -1;
    }
    slot->key = key;
    g->n_bindings++;
    if (card_out)
        *card_out = slot->card;
    return 0;
}

/**
 * How many matmul weights this geometry has, derived and not assumed.
 *
 * Checked against the walk afterwards. The two are computed from different things — this
 * from the layer counts, the walk from the tensors that actually bound — so a layer whose
 * type was misread cannot satisfy both.
 */
static int qwen35_tt_expected_bindings(const pn_qwen35moe_config *cfg, int n_ssm, int n_attn)
{
    return n_ssm * 5      /* attn_qkv, attn_gate, ssm_alpha, ssm_beta, ssm_out */
         + n_attn * 4     /* attn_q (Q and its gate, joint), attn_k, attn_v, attn_output */
         + cfg->n_layer * 3 /* the dense SwiGLU on every layer */
         + 1;             /* the LM head */
}

/** Every matmul weight of one layer, by the GGUF names the converter preserved. */
static int qwen35_tt_bind_layer(pn_qwen35_tt *g, pn_qwen35moe_model *m,
                                const pn_qwen35moe_config *cfg, int layer)
{
    pn_qwen35_tt_layer *d = &g->layers[layer];
    const pn_qwen35moe_moe *moe;
    char name[128];
    int idx = pn_qwen35moe_model_layer_index(m, layer);
    int i;

    d->is_attention = pn_qwen35moe_model_layer_type(m, layer) == PN_QWEN35MOE_LAYER_ATTN;
    if (d->is_attention) {
        pn_qwen35moe_attn_layer *l = pn_qwen35moe_model_attn_layer_mut(m, idx);
        struct { const char *suffix; const pn_qweight *key; pn_tt_weight *card; } bind[] = {
            { "attn_q", &l->attn_q, &d->attn_q },
            { "attn_k", &l->attn_k, &d->attn_k },
            { "attn_v", &l->attn_v, &d->attn_v },
            { "attn_output", &l->attn_output, &d->attn_output },
        };
        if (!l)
            return -1;
        for (i = 0; i < (int)(sizeof(bind) / sizeof(bind[0])); i++) {
            snprintf(name, sizeof(name), "blk.%d.%s.weight", layer, bind[i].suffix);
            if (qwen35_tt_bind(g, name, bind[i].key, bind[i].card) != 0)
                return -1;
        }
        moe = &l->moe;
    } else {
        pn_qwen35moe_ssm_layer *l = pn_qwen35moe_model_ssm_layer_mut(m, idx);
        struct { const char *suffix; const pn_qweight *key; pn_tt_weight *card; } bind[] = {
            { "attn_qkv", &l->attn_qkv, &d->attn_qkv },
            { "attn_gate", &l->attn_gate, &d->attn_gate },
            { "ssm_alpha", &l->ssm_alpha, &d->ssm_alpha },
            { "ssm_beta", &l->ssm_beta, &d->ssm_beta },
            { "ssm_out", &l->ssm_out, &d->ssm_out },
        };
        if (!l)
            return -1;
        for (i = 0; i < (int)(sizeof(bind) / sizeof(bind[0])); i++) {
            snprintf(name, sizeof(name), "blk.%d.%s.weight", layer, bind[i].suffix);
            if (qwen35_tt_bind(g, name, bind[i].key, bind[i].card) != 0)
                return -1;
        }
        /* ssm_conv1d is NOT bound, on purpose — see the file note. A depthwise conv is not
         * a matvec, and the artifact holding it as one is a property of the converter's
         * classifier, not a claim that multiplying by it means anything. */
        moe = &l->moe;
    }

    (void)cfg;
    {
        struct { const char *suffix; const pn_qweight *key; pn_tt_weight *card; } ffn[] = {
            { "ffn_gate", &moe->ffn_gate_shexp, &d->ffn_gate },
            { "ffn_up", &moe->ffn_up_shexp, &d->ffn_up },
            { "ffn_down", &moe->ffn_down_shexp, &d->ffn_down },
        };
        for (i = 0; i < (int)(sizeof(ffn) / sizeof(ffn[0])); i++) {
            snprintf(name, sizeof(name), "blk.%d.%s.weight", layer, ffn[i].suffix);
            if (qwen35_tt_bind(g, name, ffn[i].key, ffn[i].card) != 0)
                return -1;
        }
    }
    return 0;
}

/** A forced rung by name, or -1 for "the fit decides", or ENV_BAD. */
static int qwen35_tt_forced_rung(void)
{
    static const struct { const char *name; ft_tt_ledger_mode mode; } NAMES[] = {
        {"bfp8", FT_TT_LEDGER_BFP8}, {"bfp4", FT_TT_LEDGER_BFP4},
    };
    const char *name = getenv(PN_QWEN35_TT_RUNG_ENV);
    size_t i;

    if (!name || !*name)
        return -1;
    for (i = 0; i < sizeof(NAMES) / sizeof(NAMES[0]); i++) {
        if (strcmp(name, NAMES[i].name) == 0)
            return (int)NAMES[i].mode;
    }
    fprintf(stderr, "qwen35_tt: REFUSED — %s='%s' names neither bfp8 nor bfp4\n",
            PN_QWEN35_TT_RUNG_ENV, name);
    return PN_QWEN35_TT_ENV_BAD;
}

/**
 * The recurrent state this model needs in addition to its KV cache.
 *
 * It is a FIXED cost per sequence and does not grow with context — that is the whole point
 * of a linear-attention layer — but it is not small, and leaving it out of the capacity
 * plan would let the selector choose a rung whose weights fit and whose run does not.
 * Per SSM layer: `(ssm_conv - 1) * conv_dim` conv floats + `ssm_state^2 * ssm_dt_rank`
 * recurrence floats.
 */
static uint64_t qwen35_tt_recurrent_state_bytes(const pn_qwen35moe_config *cfg, int n_ssm)
{
    uint64_t conv = (uint64_t)(cfg->ssm_conv - 1) * (uint64_t)cfg->conv_dim;
    uint64_t recur = (uint64_t)cfg->ssm_state * (uint64_t)cfg->ssm_state *
                     (uint64_t)cfg->ssm_dt_rank;
    return (conv + recur) * sizeof(float) * (uint64_t)n_ssm;
}

/**
 * Turn the `.gguf` path into a card-ready artifact, converting and caching if need be.
 *
 * The same one function the llama backend uses, with this model's own KV arithmetic and
 * with the recurrent state charged to the run-time reserve rather than forgotten.
 */
static int qwen35_tt_resolve_artifact(pn_model *model, pn_qwen35moe_model *m,
                                      const pn_qwen35moe_config *cfg, int n_ssm,
                                      char *out, size_t out_bytes)
{
    const char *override_path = getenv(PN_QWEN35_TT_ARTIFACT_ENV);
    const char *reserve_mib = getenv(PN_QWEN35_TT_RESERVE_MIB_ENV);
    ft_tt_artifact_cache_request request;
    ft_tt_artifact_cache_result result;
    char line[FT_TT_CAPACITY_REASON_BYTES];
    unsigned long reserve;
    int forced;
    int rc;

    if (override_path && *override_path) {
        if (snprintf(out, out_bytes, "%s", override_path) >= (int)out_bytes) {
            fprintf(stderr, "qwen35_tt: REFUSED — %s is longer than %zu bytes\n",
                    PN_QWEN35_TT_ARTIFACT_ENV, out_bytes);
            return -1;
        }
        fprintf(stderr, "qwen35_tt: artifact %s (%s override — its provenance is NOT "
                        "checked against the checkpoint)\n", out, PN_QWEN35_TT_ARTIFACT_ENV);
        return 0;
    }
    if (!model->path || !*model->path) {
        fprintf(stderr, "qwen35_tt: REFUSED — the model has no checkpoint path to convert "
                        "from, and %s is not set\n", PN_QWEN35_TT_ARTIFACT_ENV);
        return -1;
    }

    forced = qwen35_tt_forced_rung();
    if (forced == PN_QWEN35_TT_ENV_BAD)
        return -1;

    reserve = PN_QWEN35_TT_RESERVE_MIB_DEFAULT;
    if (reserve_mib && *reserve_mib)
        reserve = strtoul(reserve_mib, NULL, 10);

    memset(&request, 0, sizeof(request));
    request.gguf_path = model->path;
    request.cache_dir = getenv(PN_QWEN35_TT_CACHE_DIR_ENV);
    request.budget.card_bytes = (uint64_t)FT_TT_MEM_DRAM_SIZE * 8ull;
    request.budget.runtime_state_bytes = (uint64_t)reserve * 1024ull * 1024ull +
                                         qwen35_tt_recurrent_state_bytes(cfg, n_ssm);
    request.budget.kv_bytes_per_token = pn_qwen35moe_kv_bytes_per_token(m, false);
    request.budget.context_tokens =
        (uint32_t)(cfg->ctx_cap > 0 ? cfg->ctx_cap : model->config.n_ctx);
    request.progress_style = FT_TT_CACHE_PROGRESS_STDERR;
    request.forced_mode = forced;
    /* The pairing is a property of the CHECKPOINT and the selector reads it off the file.
     * For this family it answers `source`, and for this model that is not merely right but
     * load-bearing: `attn_q` holds Q and its output gate INTERLEAVED PER HEAD
     * ([h0 Q, h0 gate, h1 Q, ...], qwen35moe_attn.c), so a split-half row permutation over
     * that tensor would permute the gate as well and there would be no error anywhere. */
    request.rope_pairing = -1;

    rc = ft_tt_artifact_cache_ensure(&request, &result);
    if (rc != FT_TT_ARTIFACT_CACHE_OK) {
        fprintf(stderr, "qwen35_tt: REFUSED — %s\n", result.reason);
        return -1;
    }
    fprintf(stderr, "qwen35_tt: %s — %s\n", result.produced ? "converted" : "cached",
            ft_tt_capacity_selection_line(&result.selection, &request.budget, line,
                                          sizeof(line)));
    if (result.produced && result.had_artifact && result.note[0])
        fprintf(stderr, "qwen35_tt: the cached artifact was NOT reusable and was not "
                        "served — %s\n", result.note);
    fprintf(stderr, "qwen35_tt: rotary pairing %s (%s); KV %llu B/token, %u tokens "
                    "reserved, recurrent state %.2f MiB\n",
            result.rope_pairing == FT_TT_ROPE_PAIRING_SPLIT_HALF ? "split-half" : "source",
            result.rope_reason ? result.rope_reason : "",
            (unsigned long long)request.budget.kv_bytes_per_token,
            request.budget.context_tokens,
            (double)qwen35_tt_recurrent_state_bytes(cfg, n_ssm) / (1024.0 * 1024.0));

    if (snprintf(out, out_bytes, "%s", result.artifact_path) >= (int)out_bytes) {
        fprintf(stderr, "qwen35_tt: REFUSED — the cached artifact's path does not fit\n");
        return -1;
    }
    return 0;
}

/* ============================================================
 * Public entry points
 * ============================================================ */

int pn_qwen35moe_tt_upload(pn_model *model)
{
    char artifact[FT_TT_ARTIFACT_CACHE_PATH_BYTES];
    const pn_qwen35moe_config *cfg;
    pn_qwen35moe_config *cfg_mut;
    pn_qwen35moe_model *m;
    pn_qwen35_tt *g = NULL;
    const char *report;
    int n_ssm, n_attn;
    int layer;

    if (!model || !model->weights)
        return -1;
    m = (pn_qwen35moe_model *)model->weights;
    cfg = pn_qwen35moe_model_config(m);
    if (!cfg)
        return -1;

    /* Two refusals, and they are different sentences on purpose: telling the SABER user
     * their model does not fit would send them to fix the wrong thing. */
    if (!cfg->dense_ffn || cfg->n_expert > 0) {
        fprintf(stderr, "qwen35_tt: this backend serves the DENSE qwen35 branch; this "
                        "checkpoint is the MoE one (%d experts) and its router and expert "
                        "gather are not built here. Staying on CPU.\n", cfg->n_expert);
        return -1;
    }
    if (cfg->n_embd % 32 != 0 || cfg->ffn_length % 32 != 0) {
        fprintf(stderr, "qwen35_tt: REFUSED — a card tile is 32 wide and this geometry "
                        "(n_embd %d, ffn %d) does not divide by it\n",
                cfg->n_embd, cfg->ffn_length);
        return -1;
    }

    n_ssm = pn_qwen35moe_model_n_ssm(m);
    n_attn = pn_qwen35moe_model_n_attn(m);
    if (n_ssm + n_attn != cfg->n_layer) {
        fprintf(stderr, "qwen35_tt: REFUSED — %d SSM + %d attention layers is not the %d "
                        "this model declares\n", n_ssm, n_attn, cfg->n_layer);
        return -1;
    }

    g = calloc(1, sizeof(*g));
    if (!g)
        return -1;
    report = getenv(PN_QWEN35_TT_REPORT_ENV);
    g->report_every = report ? atoi(report) : 0;
    g->n_expected = qwen35_tt_expected_bindings(cfg, n_ssm, n_attn);
    g->n_layers = cfg->n_layer;
    {
        const char *sparse = getenv(PN_QWEN35_TT_SPARSE_ENV);
        g->sparse_env = (sparse && *sparse) ? (atoi(sparse) != 0) : -1;
        g->sparse_logits = g->sparse_env == 1;
    }

    /* The card is opened BEFORE the artifact is resolved, and the order is load-bearing:
     * resolving can mean a ten-minute conversion, and doing that on a machine whose card
     * then fails to open would burn the ten minutes to end up on the CPU anyway. */
    if (pn_tt_device_open(0, &g->device) != 0)
        goto fail;
    /* M1: depth 4 on this model (+0.6-1.0% decode, +0.6% prefill, three rounds). Before the
     * artifact loads, because the matmul residency arms there. */
    pn_tt_device_prefer_prefetch_tiles(g->device, 4u);
    /* M4: pipelined matmul reader + 3-chunk CBs (+1.7% decode, +1.0% prefill). */
    pn_tt_device_prefer_matmul_pipe(g->device, true);
    if (qwen35_tt_resolve_artifact(model, m, cfg, n_ssm, artifact, sizeof(artifact)) != 0)
        goto fail;
    if (pn_tt_device_load_artifact(g->device, artifact) != 0)
        goto fail;
    pn_qwen35_tt_mtp_resolve(g, artifact);

    g->bindings = calloc((size_t)g->n_expected, sizeof(*g->bindings));
    g->layers = calloc((size_t)cfg->n_layer, sizeof(*g->layers));
    if (!g->bindings || !g->layers)
        goto fail;
    for (layer = 0; layer < cfg->n_layer; layer++) {
        if (qwen35_tt_bind_layer(g, m, cfg, layer) != 0)
            goto fail;
    }
    if (qwen35_tt_bind(g, "output.weight", pn_qwen35moe_model_output(m), &g->output) != 0)
        goto fail;

    if (g->n_bindings != g->n_expected) {
        fprintf(stderr, "qwen35_tt: REFUSED — bound %d weights where the geometry predicts "
                        "%d. A missing binding is a matmul that would run on the CPU while "
                        "the run reported the card.\n", g->n_bindings, g->n_expected);
        goto fail;
    }

    /* The shape the artifact says must be the shape the engine says. A silent disagreement
     * is a matmul with the right tile counts and the wrong meaning. */
    {
        pn_tt_weight head = g->bindings[g->n_bindings - 1].card;
        if ((int)head.out_features != cfg->n_vocab || (int)head.in_features != cfg->n_embd) {
            fprintf(stderr, "qwen35_tt: REFUSED — the artifact's LM head is %u x %u and the "
                            "engine says %d x %d\n", head.in_features, head.out_features,
                    cfg->n_embd, cfg->n_vocab);
            goto fail;
        }
    }

    cfg_mut = pn_qwen35moe_model_config_mut(m);
    cfg_mut->mv.fn = qwen35_tt_mv;
    cfg_mut->mv.ctx = g;
    pn_qwen35moe_model_set_gpu(m, g);

    /* Never fatal: a card that cannot arm the on-card families still serves this model on
     * the projection-only arm, slower and not wrong. */
    if (pn_qwen35_tt_resident_arm(g, m, cfg) != 0)
        goto fail;

    fprintf(stderr, "qwen35_tt: %d matmul weights resident (%.2f GiB), %d SSM + %d "
                    "attention layers, %d matmuls a token, %s arm\n",
            (int)pn_tt_device_weight_count(g->device),
            (double)pn_tt_device_weight_bytes(g->device) / (1024.0 * 1024.0 * 1024.0),
            n_ssm, n_attn, g->n_bindings,
            g->resident ? "RESIDENT-ACTIVATION" : "projection-only");
    /* AFTER every family has armed, so the line covers the KV regions, the GDN carried state
     * and the activation mirrors as well as the weights — which is what a capacity question
     * has to be answered from, and what the weight figure alone cannot say. */
    pn_tt_device_report_dram(g->device, "after arm");
    return 0;

fail:
    /* DETACH THE SEAM BEFORE FREEING WHAT IT POINTS AT. The three lines above this label
     * install `qwen35_tt_mv` on the SHARED CONFIG with `g` as its context, and the arm below
     * them is the one that can fail — so a refusal there used to free `g` and leave
     * `cfg->mv.fn` pointing into it. The caller then falls back to the CPU forward, which
     * calls `qwen35moe_mv()` on the very first projection, which reads the freed context.
     *
     * MEASURED 2026-08-21: on-card attention asked for BY NAME with a head-norm slot it could
     * not have refuses correctly, the engine prints "using CPU forward" — and the process
     * dies with SIGSEGV before the first token. The refusal is meant to cost ~0.05 tok/s and
     * it cost a crash, which is not a price anyone chose. This is the same "an accelerator is
     * not always a forward" seam that made a parity harness score the card against itself:
     * on this family the accelerator is a HOOK, so tearing it down means tearing down the
     * hook and not only the object.
     *
     * FORGET rather than detach: detaching only moves the hook into a shadow copy that a
     * later `parity_accel_seam_enable(m, 1)` would put straight back. */
    pn_qwen35moe_model_accel_mv_forget(m);
    pn_qwen35moe_model_set_gpu(m, NULL);
    if (g) {
        pn_qwen35_tt_resident_release(g);
        free(g->layers);
        free(g->bindings);
        if (g->device)
            pn_tt_device_close(g->device);
        free(g);
    }
    return -1;
}

int pn_qwen35moe_model_tt_loaded(const pn_model *model)
{
    const pn_qwen35moe_model *m;

    if (!model || !model->weights)
        return 0;
    m = (const pn_qwen35moe_model *)model->weights;
    return pn_qwen35moe_model_gpu(m) != NULL;
}

int pn_qwen35moe_model_tt_kv_on_card(const pn_model *model)
{
    const pn_qwen35moe_model *m;
    const pn_qwen35_tt *g;

    if (!model || !model->weights)
        return 0;
    m = (const pn_qwen35moe_model *)model->weights;
    g = (const pn_qwen35_tt *)pn_qwen35moe_model_gpu(m);
    return g && pn_qwen35_tt_attn_is_on_card(g) ? 1 : 0;
}

void pn_qwen35moe_tt_free(pn_model *model)
{
    pn_qwen35moe_model *m;
    pn_qwen35_tt *g;

    if (!model || !model->weights)
        return;
    m = (pn_qwen35moe_model *)model->weights;
    g = (pn_qwen35_tt *)pn_qwen35moe_model_gpu(m);
    if (!g)
        return;
    /* FORGET, not detach — the shadow copy `..._mv_enable(m, 0)` keeps would still name `g`
     * after the free below, and a parity harness re-enabling the seam would restore it. */
    pn_qwen35moe_model_accel_mv_forget(m);
    pn_qwen35moe_model_set_gpu(m, NULL);
    pn_qwen35_tt_resident_release(g);
    pn_qwen35_tt_prefill_release(g);
    pn_qwen35_tt_ckpt_release(g);
    free(g->layers);
    free(g->bindings);
    /* Closes the residency. It never parks the core — parking a resident triad breaks its
     * next boot, whether or not every processor stopped cleanly (DEV_NOTES.md §1). */
    if (g->device)
        pn_tt_device_close(g->device);
    free(g);
}

void pn_qwen35moe_tt_matmul_counts(const pn_model *model, uint64_t *on_card,
                                   uint64_t *on_host, uint64_t *failed)
{
    const pn_qwen35moe_model *m;
    const pn_qwen35_tt *g = NULL;

    if (model && model->weights) {
        m = (const pn_qwen35moe_model *)model->weights;
        g = (const pn_qwen35_tt *)pn_qwen35moe_model_gpu(m);
    }
    if (on_card)
        *on_card = g ? g->mv_on_card : 0;
    if (on_host)
        *on_host = g ? g->mv_on_host : 0;
    if (failed)
        *failed = g ? g->mv_failed : 0;
}

/**
 * Where the time went, per token, split by machine rather than summed.
 *
 * "A token takes N seconds" is not a finding; the attribution is. `upload` is a host tilize
 * plus a ~287 MB/s write, `dispatch` is the kernel launch and the card's own compute, and
 * `readback` is the ~5 MB/s return. At this rung every matvec pays all three, so the split
 * is the argument for the next rung and the number that prices it.
 *
 * `resident` and `oneshot` are EVIDENCE and not telemetry: both dispatch paths produce
 * identical results from identical kernels, so a run that silently fell back to one-shot
 * looks exactly like a resident one from its output alone.
 *
 * The last line is the one the resident arm is read from: `oncard` is the seconds and the
 * descriptor count of the norms, the SwiGLUs, the residual adds and the top-N, which are
 * work that used to be host arithmetic OUTSIDE all four phases. Moving an op onto the card
 * moves its cost INTO `oncard`, so a rung that halves the readback and grows `oncard` has
 * not lost the difference — it has started reporting work it was always doing.
 */
/**
 * The wall against the phase sum, with the difference broken out by name.
 *
 * The four device counters time the SEAMS and only the seams. Host arithmetic between two
 * seams — the Gated-Delta-Net recurrence, the attention core, the embedding gather, the
 * sparse logit row — is in none of them, so "dispatch + readback + on-card" is a lower bound
 * on the token and quoting it AS the token prices the next rung against the wrong
 * denominator. `unattributed` is what neither the counters nor the four named regions claim;
 * it is malloc, the arena resets, the per-descriptor host bookkeeping and the timer calls
 * themselves, and if it ever grows past a few percent something else has moved into it.
 *
 * `forward_wall` still excludes sampling and de-tokenisation, which happen in the caller's
 * generate loop. Those are read by subtracting the cumulative figure printed here from the
 * CLI's own decode wall.
 */
static void qwen35_tt_report_wall(const pn_qwen35_tt *g, double upload, double dispatch,
                                  double readback, double oncard)
{
    double tokens = (double)(g->tokens ? g->tokens : 1);
    double phases = upload + dispatch + readback + oncard;
    double named = g->host_recurrence_seconds + g->host_attention_seconds +
                   g->host_embed_seconds + g->host_logit_seconds;
    double wall = g->forward_wall_seconds;
    double unattributed = wall - phases - named;

    fprintf(stderr, "qwen35_tt:   WALL %.4f s/token (%.2f tok/s) = phases %.4f + host GDN "
                    "%.4f + host attn %.4f + embed %.4f + logit row %.4f + unattributed "
                    "%.4f\n",
            wall / tokens, wall > 0.0 ? tokens / wall : 0.0, phases / tokens,
            g->host_recurrence_seconds / tokens, g->host_attention_seconds / tokens,
            g->host_embed_seconds / tokens, g->host_logit_seconds / tokens,
            unattributed / tokens);
    fprintf(stderr, "qwen35_tt:   the phase sum is %.1f%% of the wall; %.4f s/token "
                    "(%.1f%%) is host work OUTSIDE all four counters. Cumulative forward "
                    "wall %.3f s over %llu forwards\n",
            wall > 0.0 ? 100.0 * phases / wall : 0.0, (wall - phases) / tokens,
            wall > 0.0 ? 100.0 * (wall - phases) / wall : 0.0, wall,
            (unsigned long long)g->tokens);
}

/**
 * The on-card phase PER FAMILY, and the GDN's two phases apart.
 *
 * "162 descriptors a token, 247 µs each" is an average over families whose work differs by
 * three orders of magnitude — a 4,096-element RMS norm and a 32-head delta rule are both one
 * descriptor. Every re-pricing of the on-card term has so far been a SUBTRACTION between two
 * runs, which cannot tell one family from another and cannot tell CONV from DELTA at all.
 * These counters are per residency and were always being kept; this only prints them.
 */
static void qwen35_tt_report_residencies(const pn_qwen35_tt *g)
{
    static const char *const phase_name[3] = { "?", "conv", "delta" };
    const double tokens = (double)(g->tokens ? g->tokens : 1);
    unsigned index;
    unsigned phase;

    for (index = 0;; index++) {
        const char *name = NULL;
        uint64_t ops = 0;
        double seconds = 0.0;
        uint32_t core = 0;

        if (!pn_tt_device_residency_stat(g->device, index, &name, &ops, &seconds, &core))
            break;
        if (ops == 0u)
            continue;
        fprintf(stderr, "qwen35_tt:   residency %-10s core %3u — %.4f s/token over %.1f "
                        "descriptors/token, %.1f us each\n",
                name ? name : "?", core, seconds / tokens, (double)ops / tokens,
                1e6 * seconds / (double)ops);
    }
    {
        double issue = 0.0, wait = 0.0;
        uint64_t issues = 0, waits = 0, wait_descriptors = 0;

        pn_tt_device_dispatch_split(g->device, &issue, &issues, &wait, &waits,
                                    &wait_descriptors);
        if (issues != 0u)
            fprintf(stderr, "qwen35_tt:   dispatch ISSUE %.4f s/token over %.1f issues/token, "
                            "%.1f us each; WAIT %.4f s/token over %.1f waits/token, %.1f us "
                            "each covering %.2f descriptors\n",
                    issue / tokens, (double)issues / tokens, 1e6 * issue / (double)issues,
                    wait / tokens, (double)waits / tokens,
                    waits ? 1e6 * wait / (double)waits : 0.0,
                    waits ? (double)wait_descriptors / (double)waits : 0.0);
        /* The device-level split is only fed by the CHAINED form (`pn_tt_matvec_enqueue`),
         * and this model dispatches through `pn_tt_matvec_dispatch`, which enqueues and
         * flushes in one call and touches neither counter — so on this arm the line above
         * is silent and the group's own clocks are the only ones that see the matmul. They
         * are printed unconditionally for that reason. */
        pn_tt_matmul_group_time_report(g->device);
        pn_tt_matmul_group_read_report(g->device, tokens);
        if (g->tokens > g->idle_clock_from && g->idle_clock_from > 0u)
            pn_tt_device_idle_clock(g->device, 1, (double)(g->tokens - g->idle_clock_from));
        pn_tt_dispatch_report();
        pn_tt_device_wait_report(g->device, tokens);
    }
    {
        /* THE AWAIT IN PCIe ACCESSES, SPIN APART FROM SETTLE. `settle` is the term the
         * completion shape moves — **16.0 control / 5.0 window / 0.0 line** — and it is the
         * only one an arm may be scored on. `poll` is the wait loop and is CARD-BOUND: it is
         * how the host waits, it runs ~44 reads an await on this model, and it swamps the
         * settle term in any pooled ratio. Printed unconditionally, because a knob that did
         * not arm has to be visible as a number and not as an absence. */
        uint64_t awaits = 0, poll = 0, settle = 0;
        unsigned fast = 0;

        pn_tt_device_await_transactions(g->device, &awaits, &poll, &settle, &fast);
        if (awaits != 0u)
            fprintf(stderr, "qwen35_tt:   residency AWAIT %.1f awaits/token — SETTLE %.2f "
                            "reads each (%.1f/token), POLL %.1f reads each (%.1f/token); "
                            "%u residencies on the one-transaction path\n",
                    (double)awaits / tokens, (double)settle / (double)awaits,
                    (double)settle / tokens, (double)poll / (double)awaits,
                    (double)poll / tokens, fast);
    }
    {
        /* THE ISSUE IN PCIe ACCESSES, AND ITS THREE CALLS TIMED APART. The whole point of the
         * per-call split is that a per-ACCESS model and a per-CALL model predict opposite
         * things about the wide shape, and the recorded 2.08/4.84 µs pair fits neither. Three
         * calls at 16, 8 and 1 accesses over-determine it. Printed unconditionally so a knob
         * that did not arm is visible as a number and not as an absence. */
        uint64_t issues = 0, accesses = 0;
        double param = 0.0, ring = 0.0, head = 0.0;
        unsigned wide = 0;

        pn_tt_device_issue_transactions(g->device, &issues, &accesses, &param, &ring, &head,
                                        &wide);
        if (issues != 0u)
            fprintf(stderr, "qwen35_tt:   residency ISSUE %.1f issues/token — %.2f accesses "
                            "each (%.1f/token), %.4f s/token = param %.2f + ring %.2f + head "
                            "%.2f us each; %u residencies on the wide shape\n",
                    (double)issues / tokens, (double)accesses / (double)issues,
                    (double)accesses / tokens, (param + ring + head) / tokens,
                    1e6 * param / (double)issues, 1e6 * ring / (double)issues,
                    1e6 * head / (double)issues, wide);
    }
    {
        /* WHICH WINDOW ACTUALLY CARRIED IT — the one witness of the issue shape that is not
         * this file's own bookkeeping. `wide` moves the parameter block and the ring slot from
         * the UC window to the WC one, so the write bytes must MOVE BETWEEN THESE TWO NUMBERS
         * by the amount the issue counters claim. A knob that armed nothing leaves them where
         * they were, and no amount of derived counting could tell you that. */
        uint64_t reg_r = 0, reg_w = 0, bulk_r = 0, bulk_w = 0;
        uint64_t reg_aims = 0, bulk_aims = 0, aim_failures = 0;
        double aim_seconds = 0.0, aim_verify_seconds = 0.0;

        pn_tt_device_window_traffic(g->device, &reg_r, &reg_w, &bulk_r, &bulk_w);
        pn_tt_device_window_aims(g->device, &reg_aims, &bulk_aims, &aim_failures);
        pn_tt_device_window_aim_seconds(g->device, &aim_seconds, &aim_verify_seconds);
        fprintf(stderr, "qwen35_tt:   windows — register(UC/word32) read %.1f write %.1f B/token "
                        "%llu aims, bulk(WC/block16) read %.1f write %.1f B/token %llu aims, "
                        "%llu aim failures\n",
                (double)reg_r / tokens, (double)reg_w / tokens,
                (unsigned long long)reg_aims, (double)bulk_r / tokens,
                (double)bulk_w / tokens, (unsigned long long)bulk_aims,
                (unsigned long long)aim_failures);
        /* WHAT THE AIMS COST. The issue path re-aims a window ONCE PER DESCRIPTOR, so this is
         * a per-token term and not setup — and the readback's share of it is what decides
         * whether the fix is fewer readbacks or more windows. */
        if (reg_aims + bulk_aims != 0u)
            fprintf(stderr, "qwen35_tt:   window AIMS %.1f/token — %.4f s/token, %.2f us each, "
                            "of which the READBACK is %.4f s/token (%.1f%%, %.2f us each)\n",
                    (double)(reg_aims + bulk_aims) / tokens, aim_seconds / tokens,
                    1e6 * aim_seconds / (double)(reg_aims + bulk_aims),
                    aim_verify_seconds / tokens,
                    aim_seconds > 0.0 ? 100.0 * aim_verify_seconds / aim_seconds : 0.0,
                    1e6 * aim_verify_seconds / (double)(reg_aims + bulk_aims));
    }
    {
        /* WHETHER THE BANK IS ABSORBING THE ROTATION, measured rather than inferred from the
         * aim count falling. The window COUNTS are what the knob achieved — a bank that could
         * not claim what it asked for runs short and says so — and `misaims` is the red arm's
         * own counter, printed unconditionally so an arm that never fired reads as a 0 and
         * not as an absence. `audit` must be 0 on every arm but the red one. */
        uint32_t reg_windows = 0u, bulk_windows = 0u;
        uint64_t rhit = 0, rmiss = 0, bhit = 0, bmiss = 0, misaims = 0, audit = 0;

        pn_tt_device_window_bank(g->device, &reg_windows, &bulk_windows, &rhit, &rmiss, &bhit,
                                 &bmiss, &misaims, &audit);
        if (reg_windows + bulk_windows != 0u)
            fprintf(stderr, "qwen35_tt:   window BANK — register %u windows %.1f hit + %.1f "
                            "miss/token (%.2f%%), bulk %u windows %.1f hit + %.1f miss/token "
                            "(%.2f%%), %llu misaims injected, %llu audit failures\n",
                    reg_windows, (double)rhit / tokens, (double)rmiss / tokens,
                    (rhit + rmiss) ? 100.0 * (double)rhit / (double)(rhit + rmiss) : 0.0,
                    bulk_windows, (double)bhit / tokens, (double)bmiss / tokens,
                    (bhit + bmiss) ? 100.0 * (double)bhit / (double)(bhit + bmiss) : 0.0,
                    (unsigned long long)misaims, (unsigned long long)audit);
    }
    for (phase = 1u; phase < 3u; phase++) {
        double seconds = 0.0, recovered_seconds = 0.0;
        uint64_t ops = 0, recovered_ops = 0, rearms = 0;

        pn_tt_device_gdn_phase_stats(g->device, phase, &seconds, &ops);
        pn_tt_device_gdn_phase_recovery(g->device, phase, &recovered_seconds, &recovered_ops,
                                        &rearms);
        if (ops == 0u && recovered_ops == 0u)
            continue;
        /* THE RE-ARM COUNT PRINTS EVEN AT ZERO. The `us each` beside it is the CLEAN figure,
         * and a contaminated one that reads as clean is the whole defect: PHASE CONV read
         * 4,502 us a descriptor against 738.2 with nothing behind it but a circular-buffer
         * write pointer that made every CONV after a DELTA refuse, and no line in this report
         * could have said so. */
        fprintf(stderr, "qwen35_tt:   gdn phase %-5s          — %.4f s/token over %.1f "
                        "descriptors/token, %.1f us each; %llu replayed over %llu re-arms "
                        "(%.4f s)\n",
                phase_name[phase], seconds / tokens, (double)ops / tokens,
                ops ? 1e6 * seconds / (double)ops : 0.0,
                (unsigned long long)recovered_ops, (unsigned long long)rearms,
                recovered_seconds);
    }
}

/**
 * The on-card phase PER SHAPE, and each family's fixed/per-tile split where it can be solved.
 *
 * The line above this one reports the elementwise residency as 136 descriptors at one mean.
 * Those 136 are six functions over three widths — 64 residual adds at 128 tiles, 32 SwiGLUs
 * at 384, 24 GDN output gates at 128, and the attention gate's SIGMOID and MUL at 256 apiece.
 * A mean over that cannot price any lever against any of them, which is the same defect the
 * per-family report was written to fix, one level down.
 *
 * The FIT is the second thing this buys. A family running several widths through one program
 * solves `per_descriptor = fixed + per_tile * tiles` from its own ordinary descriptors — and
 * the intercept is exactly what merging or removing descriptors recovers. It is NOT a host
 * round trip: it is everything that does not scale with the strip, host and card together.
 * A family that ran one width is refused rather than divided.
 */
static void qwen35_tt_report_census(const pn_qwen35_tt *g)
{
    static const char *const families[] = { "eltwise", "render-widen", "render-narrow",
                                            "reduce", "head-norm", "topk" };
    const pn_tt_descriptor_census *census = pn_tt_device_census(g->device);
    const double tokens = (double)(g->tokens ? g->tokens : 1);
    uint32_t index;
    size_t family;

    if (!census || census->row_count == 0u)
        return;
    for (index = 0; index < census->row_count; index++) {
        const pn_tt_descriptor_census_row *row = NULL;

        if (!pn_tt_descriptor_census_row_at(census, index, &row) || !row || row->ops == 0u)
            continue;
        fprintf(stderr, "qwen35_tt:   census %-13s %-9s Wt=%-5u — %.4f s/token over %.1f "
                        "descriptors/token, %.1f us each, %.3f us/tile\n",
                row->family, row->function, row->tile_count, row->seconds / tokens,
                (double)row->ops / tokens, 1e6 * row->seconds / (double)row->ops,
                1e6 * row->seconds / ((double)row->ops * (double)row->tile_count));
    }
    for (family = 0; family < sizeof(families) / sizeof(families[0]); family++) {
        double fixed = 0.0, per_tile = 0.0;
        uint32_t shapes = 0;

        if (pn_tt_descriptor_census_fit(census, families[family], &fixed, &per_tile,
                                        &shapes) != 0)
            continue;
        fprintf(stderr, "qwen35_tt:   census FIT %-13s over %u widths — fixed %.1f us/descriptor "
                        "+ %.3f us/tile\n", families[family], shapes, 1e6 * fixed,
                1e6 * per_tile);
    }
    /* Printed even at zero. A census that quietly dropped rows reads exactly like a census of
     * a cheaper model, and the reader has to be able to see that it did not. The re-arm count
     * is here for the same reason: every `us each` above is the CLEAN figure, and a run with
     * refusals in it has to be readable AS one. */
    {
        uint64_t recovered_ops = 0, rearms = 0;
        double recovered_seconds = 0.0;

        (void)pn_tt_descriptor_census_family_recovery(census, NULL, &recovered_ops,
                                                      &recovered_seconds, &rearms);
        fprintf(stderr, "qwen35_tt:   census %u keys, %llu dropped (%.4f s), %llu refused, "
                        "%llu replayed over %llu re-arms (%.4f s)\n",
                census->row_count, (unsigned long long)census->dropped,
                census->dropped_seconds, (unsigned long long)census->refused,
                (unsigned long long)recovered_ops, (unsigned long long)rearms,
                recovered_seconds);
    }
}

/**
 * The elementwise split PER MEMBER — silent when the family runs on one core.
 *
 * Reported apart from the census because the census keys on the WHOLE descriptor's width, by
 * design: that keeps the `fixed + per_tile * tiles` fit comparable across a member ladder.
 * What it therefore cannot show is the shape of the set, and "it stopped at N" is exactly the
 * finding that needs it — a knee where every member costs the same is the cores; a knee where
 * one member costs more than the others is that member being starved.
 */
static void qwen35_tt_report_eltwise_split(const pn_qwen35_tt *g)
{
    const double tokens = (double)(g->tokens ? g->tokens : 1);
    unsigned index;

    for (index = 1u; index < 64u; index++) {
        uint32_t core = 0u;
        uint64_t ops = 0;
        double seconds = 0.0;

        /* Member 1 existing is what makes this a SPLIT; a lone member 0 is the pre-split
         * path and has nothing to say here. */
        if (!pn_tt_device_eltwise_member_stat(g->device, index, NULL, NULL, NULL))
            break;
        if (index == 1u) {
            pn_tt_device_eltwise_member_stat(g->device, 0u, &core, &ops, &seconds);
            fprintf(stderr, "qwen35_tt:   eltwise split member 0 core %-3u — %.4f s/token "
                            "await over %.1f descriptors/token\n",
                    core, seconds / tokens, (double)ops / tokens);
        }
        pn_tt_device_eltwise_member_stat(g->device, index, &core, &ops, &seconds);
        fprintf(stderr, "qwen35_tt:   eltwise split member %-1u core %-3u — %.4f s/token "
                        "await over %.1f descriptors/token\n",
                index, core, seconds / tokens, (double)ops / tokens);
    }
}

/**
 * The strip bridge's split PER MEMBER — silent when a direction runs on one core.
 *
 * Same argument as the elementwise one above, and printed for the same reason: a SPLIT line
 * at arming time is a claim about what was asked for, and a later stage can still take the
 * members away. Measured 2026-08-26 on the elementwise family, where `pn_tt_publish_replica_
 * map()` COLLAPSED every split above two members and the arming line still said 16 — three
 * census ladders read as a core-scaling knee that was a table size. A per-member line that
 * is ABSENT is the only thing that catches that.
 */
static void qwen35_tt_report_render_split(const pn_qwen35_tt *g)
{
    const double tokens = (double)(g->tokens ? g->tokens : 1);
    uint32_t direction;

    for (direction = 0; direction < PN_TT_RENDER_INDEX_COUNT; direction++) {
        uint32_t members = pn_tt_render_members(g->device, direction);
        uint32_t member;

        if (members < 2u)
            continue;
        for (member = 0; member < members; member++) {
            const char *name = "?";
            uint32_t core = 0u;
            uint64_t ops = 0;
            double seconds = 0.0;

            if (!pn_tt_render_member_stat(g->device, direction, member, &name, &core, &ops,
                                          &seconds))
                break;
            fprintf(stderr, "qwen35_tt:   %s split member %-1u core %-3u — %.4f s/token "
                            "await over %.1f descriptors/token\n",
                    name, member, core, seconds / tokens, (double)ops / tokens);
        }
    }
}

static void qwen35_tt_report(const pn_qwen35_tt *g)
{
    double upload = 0.0, dispatch = 0.0, readback = 0.0;
    uint64_t resident = 0, oneshot = 0, restarts = 0;
    double per_token;

    pn_tt_device_phase_seconds(g->device, &upload, &dispatch, &readback);
    pn_tt_device_dispatch_counts(g->device, &resident, &oneshot, &restarts);
    per_token = g->tokens ? (upload + dispatch + readback) / (double)g->tokens : 0.0;

    fprintf(stderr, "qwen35_tt: token %llu — %llu matmuls on the card, %llu on the host, "
                    "%llu refused\n", (unsigned long long)g->tokens,
            (unsigned long long)g->mv_on_card, (unsigned long long)g->mv_on_host,
            (unsigned long long)g->mv_failed);
    fprintf(stderr, "qwen35_tt:   phases %.3f s upload + %.3f s dispatch + %.3f s readback "
                    "= %.3f s/token on the card path (%.2f tok/s if it were all)\n",
            upload / (double)(g->tokens ? g->tokens : 1),
            dispatch / (double)(g->tokens ? g->tokens : 1),
            readback / (double)(g->tokens ? g->tokens : 1), per_token,
            per_token > 0.0 ? 1.0 / per_token : 0.0);
    fprintf(stderr, "qwen35_tt:   %llu resident dispatches, %llu one-shot, %llu restarts; "
                    "%.2f MiB read back, %.2f MiB written\n",
            (unsigned long long)resident, (unsigned long long)oneshot,
            (unsigned long long)restarts,
            (double)pn_tt_device_bytes_read(g->device) / (1024.0 * 1024.0),
            (double)pn_tt_device_bytes_written(g->device) / (1024.0 * 1024.0));
    {
        double oncard = 0.0;
        uint64_t ops = 0, oncard_restarts = 0;

        pn_tt_device_oncard_stats(g->device, &oncard, &ops, &oncard_restarts);
        fprintf(stderr, "qwen35_tt:   on-card ops %.3f s/token over %llu descriptors "
                        "(%llu restarts); readback %.2f KiB/token\n",
                oncard / (double)(g->tokens ? g->tokens : 1), (unsigned long long)ops,
                (unsigned long long)oncard_restarts,
                (double)pn_tt_device_bytes_read(g->device) /
                    (1024.0 * (double)(g->tokens ? g->tokens : 1)));
        qwen35_tt_report_wall(g, upload, dispatch, readback, oncard);
    }
    qwen35_tt_report_residencies(g);
    qwen35_tt_report_eltwise_split(g);
    qwen35_tt_report_render_split(g);
    qwen35_tt_report_census(g);
    /* The on-card GDN verdict rides the same cadence, so a long run shows the gate moving
     * rather than only reporting it at teardown — a run killed on the link never reaches the
     * release path, and a gate whose only output is at exit measures nothing then. */
    pn_qwen35_tt_gdn_parity_report(g);
    pn_qwen35_tt_gdn_normcheck_report(g);
}

/**
 * SAY IT WHEN A PROMPT IS BEING PREFILLED ONE POSITION AT A TIME.
 *
 * The chunkwise prefill ships ON and is worth ~24x, and it is reached by the caller ASKING
 * for it (`pn_qwen35moe_prefill_tt`). A caller that does not ask gets a correct answer at
 * ~20x the wall, and until 2026-08-26 two of the three prefill entry points in this tree did
 * not ask — with nothing anywhere saying so. This is the counter that says so, and it lives
 * in the forward rather than at a call site precisely because the defect WAS a call site.
 *
 * `run` is consecutive single-position forwards. Generation is also single-position, so the
 * threshold is what separates a prompt from a completion; a chat turn of 64 generated tokens
 * on a model that HAS prefilled once will not trip it, because a batched pass clears the run
 * and sets `prefill_passes`.
 *
 * @param g    the residency, for its counters
 * @param pos  the position this forward is about to compute
 */
static void qwen35_tt_note_sequential_prefill(pn_qwen35_tt *g, int pos)
{
    if (pos == 0)
        g->sequential_prefill_run = 0;
    g->sequential_prefill_run++;
    if (g->sequential_prefill_said || g->prefill_passes != 0u ||
        g->sequential_prefill_run < PN_QWEN35_TT_SEQUENTIAL_PREFILL_ALARM)
        return;
    g->sequential_prefill_said = true;
    fprintf(stderr,
            "qwen35_tt: PREFILL IS RUNNING PER TOKEN — %llu consecutive single-position "
            "forwards and NOT ONE batched pass. The chunkwise prefill is armed and nobody "
            "asked it; this prompt costs roughly 20x what it should. The caller must go "
            "through pn_qwen35moe_prefill_tt(), or set PN_TT_PREFILL=0 to say the "
            "per-token path is what was wanted.\n",
            (unsigned long long)g->sequential_prefill_run);
}

/** Wrap the logit row the chosen arm produced in the tensor the engine expects. */
static tm_tensor *qwen35_tt_wrap_logits(const float *logits, int n_vocab)
{
    int64_t shape[1] = { n_vocab };
    tm_tensor *out = tm_empty(shape, 1, TM_F32);

    if (!out)
        return NULL;
    memcpy(tm_data_f32(out), logits, (size_t)n_vocab * sizeof(float));
    return out;
}

/**
 * A MEASUREMENT KNOB, NOT A FEATURE: `PN_TT_IDLE_PROBE=<forward>:<ms>` sleeps `ms` before the
 * `<forward>`-th TT forward of this process (0-based), so every residency on the card sits idle
 * across the gap exactly as it does in a `serve` between requests. It exists to prove the idle
 * retire: with the retire ON the token after the gap must be byte-identical; with it OFF
 * (`PN_TT_RESIDENCY_IDLE_RETIRE_MS=0`) the first descriptors after a gap past the idle bound
 * must be refused. Read once; said out loud when armed.
 */
static void qwen35_tt_idle_probe(void)
{
    static long at = -2, ms = 0, calls = 0;

    if (at == -2) {
        const char *value = getenv("PN_TT_IDLE_PROBE");

        at = -1;
        if (value && sscanf(value, "%ld:%ld", &at, &ms) == 2 && at >= 0 && ms > 0)
            fprintf(stderr, "qwen35_tt: IDLE PROBE armed — %ld ms of card idle before TT "
                            "forward %ld\n", ms, at);
        else
            at = -1;
    }
    if (at >= 0 && calls == at) {
        struct timespec nap = { ms / 1000, (ms % 1000) * 1000000L };

        fprintf(stderr, "qwen35_tt: IDLE PROBE — sleeping %ld ms before forward %ld\n", ms, at);
        (void)nanosleep(&nap, NULL);
    }
    calls++;
}

tm_tensor *pn_qwen35moe_forward_tt(pn_model *model, uint32_t token, pn_state *state)
{
    pn_qwen35moe_model *m;
    pn_qwen35moe_state *s;
    pn_qwen35_tt *g;
    struct fw_pool *pool = NULL;
    tm_tensor *out;
    double forward_started_ms;

    if (!model || !model->weights || !state)
        return NULL;
    m = (pn_qwen35moe_model *)model->weights;
    g = (pn_qwen35_tt *)pn_qwen35moe_model_gpu(m);
    if (!g) {
        fprintf(stderr, "qwen35_tt: the TT forward was selected with no residency "
                        "attached\n");
        return NULL;
    }
#ifdef USE_THREADS
    pool = model->pool;
#endif
    qwen35_tt_idle_probe();
    forward_started_ms = pn_time_ms();

    if (g->resident) {
        const float *logits;

        s = (pn_qwen35moe_state *)state->kv_cache;
        if (!s)
            return NULL;
        /* The residual stream, both RMSNorms, the dense SwiGLU and the LM head stay in card
         * DRAM; the Gated-Delta-Net recurrence and the attention core run on the host,
         * through the same two functions the CPU oracle calls. */
        qwen35_tt_note_sequential_prefill(g, state->pos);
        pn_qwen35_tt_ckpt_note(g, state, state->pos);
        logits = pn_qwen35_tt_resident_forward(g, m, token, s, state->pos, pool);
        if (!logits)
            return NULL;
        {
            double wrap_started_ms = pn_time_ms();
            out = qwen35_tt_wrap_logits(logits, s->n_vocab);
            g->host_logit_seconds += (pn_time_ms() - wrap_started_ms) / 1000.0;
        }
        /* STEP 4 S2b: this position's hidden is the MTP catch-up owed before the next draft. */
        if (out && g->mtp.on) {
            if (pn_tt_row_copy(g->device, &g->card.hidden[g->card.slot], 0u, &g->mtp.h, 0u) == 0) {
                g->mtp.pend_base = state->pos;
                g->mtp.pend_n = 1;
                g->mtp.pend_tall = false;
            } else {
                g->mtp.pend_n = 0;
            }
        }
        if (out)
            state->pos++;
    } else {
        /* The whole hybrid stack, unchanged. The matmuls leave through the seam installed at
         * upload; everything else is the CPU oracle's own arithmetic on its own fp32 state,
         * which is what makes this arm diagnosable against it. */
        out = pn_qwen35moe_forward(model, token, state);
    }

    g->forward_wall_seconds += (pn_time_ms() - forward_started_ms) / 1000.0;
    g->tokens++;
    if (g->tokens == 2u && g->idle_clock_from == 0u) {
        pn_tt_device_idle_clock(g->device, 0, 0.0);
        g->idle_clock_from = 2u;
    }
    if (g->tokens == 2u || g->tokens == 24u || g->tokens == 47u) {
        char label[32];

        snprintf(label, sizeof(label), "decode forward %llu", (unsigned long long)g->tokens);
        pn_tt_device_telemetry_log(g->device, label);
    }
    if (g->report_every > 0 && g->tokens % (uint64_t)g->report_every == 0)
        qwen35_tt_report(g);
    return out;
}

/* =======================================================================================
 * STEP 4 S2 — one speculative step (greedy, lossless)
 * ======================================================================================= */

int pn_qwen35moe_tt_spec_step(pn_model *model, pn_state *state, uint32_t seed,
                              const uint32_t *drafts, int n_drafts, uint32_t *emit)
{
    return pn_qwen35moe_tt_spec_step_ex(model, state, seed, drafts, n_drafts, emit, NULL, NULL);
}

/* This request's logit rows: `pairs` = 0 dense, else the card's exact top-`pairs`. */
static void qwen35_tt_set_rows(pn_model *model, uint32_t pairs)
{
    pn_qwen35_tt *g;

    if (!model || !model->weights || model->forward != pn_qwen35moe_forward_tt)
        return;
    g = (pn_qwen35_tt *)pn_qwen35moe_model_gpu((pn_qwen35moe_model *)model->weights);
    if (!g || !g->resident)
        return;
    g->sparse_logits = g->sparse_capable && pairs > 0u;
    g->topk_pairs = pairs > 0u ? pairs : PN_QWEN35_TT_TOPK_GREEDY;
}

int pn_qwen35moe_tt_dense_row(void *model_ctx, float *out, int n_vocab)
{
    pn_model *model = (pn_model *)model_ctx;
    pn_qwen35_tt *g;

    if (!model || !model->weights || !out)
        return -1;
    g = (pn_qwen35_tt *)pn_qwen35moe_model_gpu((pn_qwen35moe_model *)model->weights);
    if (!g || !g->resident)
        return -1;
    return pn_tt_activation_download(g->device, &g->card.logits, out, (uint32_t)n_vocab);
}

int pn_qwen35moe_tt_dense_chunks(void *model_ctx, const unsigned char *need, float *out,
                                 int n_vocab)
{
    pn_model *model = (pn_model *)model_ctx;
    pn_qwen35_tt *g;

    if (!model || !model->weights || !out || !need)
        return -1;
    g = (pn_qwen35_tt *)pn_qwen35moe_model_gpu((pn_qwen35moe_model *)model->weights);
    if (!g || !g->resident)
        return -1;
    return pn_tt_activation_download_chunks(g->device, &g->card.logits, need, out,
                                            (uint32_t)n_vocab);
}

bool pn_qwen35moe_tt_greedy_exact_on_topn(const pn_sampling_params *params)
{
    return params && params->temperature <= 0.0f && !params->tool_grammar &&
           !(params->logit_bias && params->n_logit_bias > 0) &&
           params->presence_penalty == 0.0f && params->frequency_penalty == 0.0f;
}

uint32_t pn_qwen35moe_tt_select_rows(pn_model *model, const pn_sampling_params *params,
                                     bool needs_full_logits)
{
    pn_qwen35_tt *g;
    uint32_t pairs = 0u;
    const bool greedy = params && params->temperature <= 0.0f;

    if (!model || !model->weights || model->forward != pn_qwen35moe_forward_tt || !params)
        return 0u;
    g = (pn_qwen35_tt *)pn_qwen35moe_model_gpu((pn_qwen35moe_model *)model->weights);
    if (!g)
        return 0u;
    if (g->sparse_env == 1) {
        pairs = greedy ? PN_QWEN35_TT_TOPK_GREEDY : PN_QWEN35_TT_TOPK_PAIRS;
    } else if (g->sparse_env == -1 && !needs_full_logits) {
        if (pn_qwen35moe_tt_greedy_exact_on_topn(params))
            pairs = PN_QWEN35_TT_TOPK_GREEDY;
        /* Sampled (llama.cpp/HF order): the renormalised top-k set lies inside the top-64
         * unless the penalties reach past it — the sampler checks that per draw and refills
         * the dense row (pn_sampler_topn_row_exact). `PN_TT_SAMPLED_TOPN=0`: dense. */
        else if (!greedy && !params->tool_grammar && !params->logprobs &&
                 !(params->logit_bias && params->n_logit_bias > 0) && params->top_k >= 1 &&
                 params->top_k <= (int)PN_QWEN35_TT_TOPK_PAIRS &&
                 !(getenv("PN_TT_SAMPLED_TOPN") && atoi(getenv("PN_TT_SAMPLED_TOPN")) == 0))
            pairs = PN_QWEN35_TT_TOPK_PAIRS;
    }
    qwen35_tt_set_rows(model, pairs);
    return g->sparse_logits ? pairs : 0u;
}

bool pn_qwen35moe_tt_spec_can_sample(pn_model *model)
{
    pn_qwen35_tt *g;

    if (!model || !model->weights)
        return false;
    g = (pn_qwen35_tt *)pn_qwen35moe_model_gpu((pn_qwen35moe_model *)model->weights);
    return g && g->resident && g->sparse_logits &&
           !(getenv("PN_TT_SPEC_HEADS") && atoi(getenv("PN_TT_SPEC_HEADS")) == 0);
}

int pn_qwen35moe_tt_spec_step_ex(pn_model *model, pn_state *state, uint32_t seed,
                                 const uint32_t *drafts, int n_drafts, uint32_t *emit,
                                 pn_spec_choose_fn choose, void *choose_ctx)
{
    pn_qwen35moe_model *m;
    const pn_qwen35moe_config *cfg;
    pn_qwen35moe_state *s;
    pn_qwen35_tt *g;
    uint32_t tokens[16], argmax[16], forced[16], next = 0u;
    int rows, n_accept = 0, i, committed;
    double started_ms;

    if (!model || !model->weights || !state || !emit || n_drafts < 0 || n_drafts > 15 ||
        (choose && !pn_qwen35moe_tt_spec_can_sample(model)))
        return -1;
    m = (pn_qwen35moe_model *)model->weights;
    cfg = pn_qwen35moe_model_config(m);
    g = (pn_qwen35_tt *)pn_qwen35moe_model_gpu(m);
    s = (pn_qwen35moe_state *)state->kv_cache;
    if (!g || !g->resident || !s || !cfg)
        return -1;
    pn_qwen35_tt_ckpt_note(g, state, state->pos);
    /* The verify's heads need the top-N rows: refuse BEFORE any card work — a verify that
     * ran and then failed would leave the carried state advanced past the committed token
     * (the empty-env run: spec failed after the pass, greedy DIFFERED). */
    if (!g->sparse_logits && !(getenv("PN_TT_SPEC_HEADS") && atoi(getenv("PN_TT_SPEC_HEADS")) == 0))
        return -1;
    /* The idle clock's window starts at the first speculative cycle when decode is spec. */
    if (g->idle_clock_from == 0u) {
        pn_tt_device_idle_clock(g->device, 0, 0.0);
        g->idle_clock_from = g->tokens > 0u ? g->tokens : 1u;
    }
    started_ms = pn_time_ms();
    tokens[0] = seed;
    for (i = 0; i < n_drafts; i++)
        tokens[1 + i] = drafts[i];
    rows = n_drafts + 1;
    /* S2d MEASURE: `PN_TT_DISPATCH_OBSERVE_SPEC=1` + `PN_TT_DISPATCH_RECORD=1000000+c` records
     * verify cycle c in the dispatch recorder's observe mode (2000000+c: its MTP drafting). */
    if (getenv("PN_TT_DISPATCH_OBSERVE_SPEC"))
        pn_tt_dispatch_token_begin(g->device, 1000000u + g->spec_cycles);
    {
        /* S2d: the verify runs as STREAMED puller segments (`PN_TT_SPEC_STREAM=0` off). */
        const bool streamed = !getenv("PN_TT_DISPATCH_OBSERVE_SPEC") &&
                              pn_tt_dispatch_stream_begin(g->device);
        const int vrc = pn_qwen35_tt_spec_verify(g, m, cfg, s, tokens, rows, state->pos, argmax);

        if ((streamed && !pn_tt_dispatch_stream_end()) || vrc != 0)
            return -1;
    }
    if (getenv("PN_TT_DISPATCH_OBSERVE_SPEC"))
        pn_tt_dispatch_token_end(g->device, 1000000u + g->spec_cycles);
    /* argmax[r] is the greedy token AFTER tokens[r]; draft i sits at row i+1, so it is
     * accepted while it equals argmax[i]. RED `spec_accept_unverified`: the first draft is
     * accepted whatever the verify said, and on the first cycle it is a token the verify
     * certainly rejected (argmax + 1) — so the red cannot be vacuous at high acceptance. */
    {
        const char *red = getenv("PN_TT_RED");
        const bool red_lookup = g_red_accept_next_cycle;

        g_red_accept_next_cycle = false;
        if ((red && strcmp(red, "spec_accept_unverified") == 0) || red_lookup) {
            for (i = 0; i < n_drafts; i++)
                forced[i] = drafts[i];
            if ((g->spec_cycles == 0u || red_lookup) && n_drafts > 0)
                forced[0] = (argmax[0] + 1u) % (uint32_t)s->n_vocab;
            drafts = forced;
            n_accept = n_drafts > 0 ? 1 : 0;
        }
        /* SAMPLING (`choose`): row r's token is the chooser's draw over row r's sparse
         * logits, drawn once per EMITTED token in order; a draft is accepted iff the draw
         * equals it — plain sampling's tokens, RNG stream and repeat window exactly. */
        for (;;) {
            const uint32_t token =
                choose ? choose(choose_ctx, g->spec_topk_index[n_accept],
                                g->spec_topk_value[n_accept], g->spec_topk_count[n_accept])
                       : argmax[n_accept];

            if (token == PN_SPEC_CHOOSE_DEFER) {
                /* Row n_accept's top-N cannot decide it: the draft row n_accept-1 accepted
                 * becomes the next (uncommitted) token. Only rows >= 1 may defer. */
                if (n_accept == 0)
                    return -1;
                n_accept--;
                next = drafts[n_accept];
                g->spec_deferred++;
                break;
            }
            if (n_accept < n_drafts && token == drafts[n_accept]) {
                n_accept++;
                continue;
            }
            next = token;
            break;
        }
        for (i = 0; i < n_accept; i++)
            emit[i] = drafts[i];
    }
    if (getenv("PN_TT_SPEC_TRACE")) {
        fprintf(stderr, "SPEC pos %d seed %u |", state->pos, seed);
        for (i = 0; i < n_drafts; i++)
            fprintf(stderr, " d%u", drafts[i]);
        fprintf(stderr, " |");
        for (i = 0; i < rows; i++)
            fprintf(stderr, " a%u", argmax[i]);
        fprintf(stderr, " | accept %d\n", n_accept);
    }
    emit[n_accept] = next;
    committed = n_accept + 1;  /* the seed + the accepted drafts */
    {
        const double t = pn_time_ms();

        /* RED `spec_rollback_skip`: keep the state after the LAST verify row — must DIFFER. */
        const char *red = getenv("PN_TT_RED");

        if (committed < rows && !(red && strcmp(red, "spec_rollback_skip") == 0) &&
            pn_qwen35_tt_spec_rollback(g, cfg, committed - 1) != 0)
            return -1;
        g->spec_rollback_ms += pn_time_ms() - t;
    }
    /* STEP 4 S2b: the committed rows' hiddens (the verify set, rows 0..committed-1) are the
     * MTP catch-up owed before the next draft; each is followed by an accepted draft, the last
     * by the next seed. */
    if (g->mtp.on) {
        g->mtp.pend_base = state->pos;
        g->mtp.pend_n = committed;
        g->mtp.pend_tall = true;
        for (i = 0; i < n_accept; i++)
            g->mtp.pend_tok[i] = drafts[i];
    }
    state->pos += committed;
    /* The verify issued outside the dispatch model's per-token shape: re-learn it before any
     * plain forward replays again. */
    pn_tt_dispatch_sequence_reset();
    g->spec_cycles++;
    g->spec_drafted += (uint64_t)n_drafts;
    g->spec_accepted += (uint64_t)n_accept;
    g->forward_wall_seconds += (pn_time_ms() - started_ms) / 1000.0;
    g->spec_total_ms += pn_time_ms() - started_ms;
    g->tokens += (uint64_t)committed;
    if (g->report_every > 0 && g->tokens >= (uint64_t)g->report_every && !g->spec_reported) {
        uint64_t mr;
        double mr_ms, cyc = (double)(g->spec_cycles ? g->spec_cycles : 1);

        g->spec_reported = true;
        pn_tt_move_stats(&mr, &mr_ms);
        fprintf(stderr, "qwen35_tt: SPEC — %llu cycles, %llu drafted, %llu accepted (%.2f), "
                        "%.2f tokens a cycle | ms a cycle: total %.1f = pass %.1f (row copies "
                        "%.1f x%.0f) + heads %.1f + mtp %.1f (%.1f steps, %.1f heads) + rollback "
                        "%.1f\n",
                (unsigned long long)g->spec_cycles, (unsigned long long)g->spec_drafted,
                (unsigned long long)g->spec_accepted,
                g->spec_drafted ? (double)g->spec_accepted / (double)g->spec_drafted : 0.0,
                (double)(g->spec_cycles + g->spec_accepted) / cyc,
                (g->spec_total_ms + g->mtp.ms) / cyc,
                g->spec_pass_ms / cyc, mr_ms / cyc, (double)mr / cyc, g->spec_head_ms / cyc,
                g->mtp.ms / cyc, (double)g->mtp.steps / cyc, (double)g->mtp.heads / cyc,
                g->spec_rollback_ms / cyc);
        {
            unsigned c;
            double clocked = 0.0;

            fprintf(stderr, "qwen35_tt: SPEC verify pass by card clock, ms a cycle:");
            for (c = 0; c < g->spec_clock_count; c++) {
                clocked += g->spec_clock_seconds[c];
                if (g->spec_clock_seconds[c] * 1000.0 / cyc >= 0.5)
                    fprintf(stderr, " %s %.1f", g->spec_clock_names[c],
                            g->spec_clock_seconds[c] * 1000.0 / cyc);
            }
            fprintf(stderr, " | clocked %.1f of pass %.1f = gdn rows %.1f + attention rows %.1f "
                            "+ batched %.1f\n", clocked * 1000.0 / cyc, g->spec_pass_ms / cyc,
                    g->spec_ssm_rows_ms / cyc, g->spec_attn_rows_ms / cyc,
                    (g->spec_pass_ms - g->spec_ssm_rows_ms - g->spec_attn_rows_ms) / cyc);
            fprintf(stderr, "qwen35_tt: SPEC gdn rows by section, ms a cycle: pulls %.1f conv %.1f "
                            "l2 %.1f delta %.1f tail %.1f\n",
                    g->spec_gdn_part_ms[0] / cyc, g->spec_gdn_part_ms[1] / cyc,
                    g->spec_gdn_part_ms[2] / cyc, g->spec_gdn_part_ms[3] / cyc,
                    g->spec_gdn_part_ms[4] / cyc);
            pn_tt_dispatch_stream_report(cyc);
        }
        qwen35_tt_report(g);
    }
    return n_accept + 1;
}

/* =======================================================================================
 * STEP 4 S2b — MTP drafts
 * ======================================================================================= */

static int qwen35_tt_mtp_draft_steps(pn_qwen35_tt *g, pn_qwen35moe_model *m,
                                     const pn_qwen35moe_config *cfg, pn_qwen35moe_state *s,
                                     pn_state *state, uint32_t seed, int k, uint32_t *drafts,
                                     int n);

/** `PN_TT_SPEC_ADAPT`: the confidence below which drafting stops. DEFAULT 0.6
 *  (run_serve_spec_adapt.sh round 2: g256 +5.5%, m256 +2.6%, t2 +2.0%, code/g48 flat, all
 *  IDENTICAL); `=0` off. */
static float qwen35_tt_spec_adapt_tau(void)
{
    static float tau = -1.0f;

    if (tau < 0.0f) {
        const char *env = getenv("PN_TT_SPEC_ADAPT");

        tau = env && *env ? (float)atof(env) : 0.6f;
        if (tau < 0.0f)
            tau = 0.0f;
    }
    return tau;
}

/** Drafts `from..k-1`, each one MTP step from the previous draft (the catch-up made draft 0).
 *  @return k, or the count of drafts that are still good when a step failed */
static int qwen35_tt_mtp_draft_tail(pn_qwen35_tt *g, pn_qwen35moe_model *m,
                                    const pn_qwen35moe_config *cfg, pn_qwen35moe_state *s,
                                    pn_state *state, uint32_t *drafts, int from, int k)
{
    const float tau = qwen35_tt_spec_adapt_tau();
    int i;

    for (i = from; i < k; i++) {
        /* ADAPTIVE COUNT (`PN_TT_SPEC_ADAPT=tau`): a draft the head is unsure of is the last
         * one worth a verify row; drafting past it spends an MTP step and a row on tokens the
         * verify will almost surely refuse. The drafts are the same tokens either way. */
        if (tau > 0.0f && g->mtp.last_conf < tau) {
            g->mtp.adapt_stops++;
            return i;
        }
        if (pn_qwen35_tt_mtp_step(g, m, cfg, s, drafts[i - 1], NULL, 0u, state->pos - 1 + i,
                                  true, &drafts[i]) != 0)
            return i;
    }
    return k;
}

int pn_qwen35moe_tt_mtp_draft(pn_model *model, pn_state *state, uint32_t seed, int k,
                              uint32_t *drafts)
{
    pn_qwen35moe_model *m;
    const pn_qwen35moe_config *cfg;
    pn_qwen35moe_state *s;
    pn_qwen35_tt *g;
    int n;

    if (!model || !model->weights || !state || !drafts || k < 1 || k > 15)
        return -1;
    m = (pn_qwen35moe_model *)model->weights;
    cfg = pn_qwen35moe_model_config(m);
    g = (pn_qwen35_tt *)pn_qwen35moe_model_gpu(m);
    s = (pn_qwen35moe_state *)state->kv_cache;
    if (!g || !g->mtp.on || !s || !cfg)
        return -1;
    n = g->mtp.pend_n;
    /* The catch-up must end exactly where the seed sits, or the drafts are for elsewhere. */
    if (n < 1 || g->mtp.pend_base + n != state->pos)
        return -1;
    g->mtp.pend_n = 0;
    {
        /* The drafts are decode time: booked into the forward wall the reports divide. */
        const double before = g->mtp.ms;
        int got;

        bool streamed = false;

        if (getenv("PN_TT_DISPATCH_OBSERVE_SPEC"))
            pn_tt_dispatch_token_begin(g->device, 2000000u + g->spec_cycles);
        else
            streamed = pn_tt_dispatch_stream_begin(g->device);
        got = qwen35_tt_mtp_draft_steps(g, m, cfg, s, state, seed, k, drafts, n);
        if (streamed && !pn_tt_dispatch_stream_end())
            got = -1;
        if (getenv("PN_TT_DISPATCH_OBSERVE_SPEC"))
            pn_tt_dispatch_token_end(g->device, 2000000u + g->spec_cycles);

        g->forward_wall_seconds += (g->mtp.ms - before) / 1000.0;
        return got;
    }
}

static int qwen35_tt_mtp_draft_steps(pn_qwen35_tt *g, pn_qwen35moe_model *m,
                                     const pn_qwen35moe_config *cfg, pn_qwen35moe_state *s,
                                     pn_state *state, uint32_t seed, int k, uint32_t *drafts,
                                     int n)
{
    int i;

    /* S2f: the committed rows' catch-up as ONE pass (`PN_TT_MTP_BATCH=0`: one step a row). */
    if (g->mtp.pend_tall && n >= 2 &&
        !(getenv("PN_TT_MTP_BATCH") && atoi(getenv("PN_TT_MTP_BATCH")) == 0)) {
        uint32_t next[16];

        for (i = 0; i < n - 1; i++)
            next[i] = g->mtp.pend_tok[i];
        next[n - 1] = seed;
        if (pn_qwen35_tt_mtp_catchup_rows(g, m, cfg, s, next, n, g->mtp.pend_base,
                                          &drafts[0]) != 0)
            return -1;
        n = 0;
    }
    for (i = 0; i < n; i++) {
        const uint32_t tok = i < n - 1 ? g->mtp.pend_tok[i] : seed;
        const pn_tt_activation *src =
            g->mtp.pend_tall ? &g->card_tall.hidden[g->card_tall.slot] : NULL;

        if (pn_qwen35_tt_mtp_step(g, m, cfg, s, tok, src, (uint32_t)i, g->mtp.pend_base + i,
                                  i == n - 1, &drafts[0]) != 0)
            return -1;
    }
    return qwen35_tt_mtp_draft_tail(g, m, cfg, s, state, drafts, 1, k);
}

#endif /* USE_TT */
