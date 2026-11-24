/* SPDX-License-Identifier: AGPL-3.0-or-later */
/* Copyright (C) 2026 David (Corey) Herrera */
/**
 * @file qwen35moe_tt_internal.h
 * @brief The state the two halves of the `qwen35` Tenstorrent backend share:
 *        `qwen35moe_forward_tt.c` (residency, bindings, counters) and
 *        `qwen35moe_resident_tt.c` (the resident-activation forward).
 *
 *        Private to those two translation units. Nothing outside `src/models/` includes it.
 *
 * @status None
 * @issues None
 * @todo None
 */

#ifndef PN_QWEN35MOE_TT_INTERNAL_H
#define PN_QWEN35MOE_TT_INTERNAL_H

/** PM 5: a resident family idle this long gets a keepalive NOP (the retire limit is 5 s). */
#define PN_QWEN35_TT_KEEPALIVE_IDLE_S 2.0


#ifdef USE_TT

#include <stdbool.h>
#include <stdint.h>

#include "qwen35moe.h"
#include "qwen35moe_internal.h"

#include "../accel/tt/pn_tt_device.h"
#include "../accel/tt/pn_tt_fp32_strip.h"
#include "../accel/tt/pn_tt_norm_role_plan.h"
#include "../accel/tt/pn_tt_prefill_reserve.h"
#include "../accel/tt/pn_tt_strip_placement.h"
#include "../core/engine.h"

/** How many (index, value) pairs the card's exact top-N returns when the logit row stays
 *  resident. Matches the llama backend's, and eight is well past what a greedy decode or a
 *  temperature sampler over a survivor set needs. */
/** Top-N array capacity (the kernel's ceiling); the per-request N is `topk_pairs`. */
#define PN_QWEN35_TT_TOPK_PAIRS 64u
/** N for greedy rows and MTP drafts. */
#define PN_QWEN35_TT_TOPK_GREEDY 8u

/** Present the LM head as the card's exact top-N instead of reading the whole logit row.
 *  Deliberately the same spelling as the llama backend's. */
#define PN_QWEN35_TT_SPARSE_ENV "PN_TT_SPARSE_LOGITS"

/**
 * One host weight and the card weight that serves it.
 *
 * Keyed by the ADDRESS of the `pn_qweight` the model already holds, because that is the only
 * thing the `qwen35moe_mv` seam is handed. The alternative — re-deriving a tensor name at
 * every matvec — would put a string compare on the hot path and, worse, would make the seam
 * depend on a naming convention rather than on the binding that was actually checked at load.
 */
typedef struct {
    const pn_qweight *key;
    pn_tt_weight card;
    /** A joint Q+gate weight uploaded as its halves (`card` = Q, `gate` = gate): the seam runs
     *  both and re-interleaves per head (`split_heads` x `split_head_dim`). */
    bool split;
    pn_tt_weight gate;
    uint32_t split_heads;
    uint32_t split_head_dim;
} pn_qwen35_tt_binding;

/**
 * Every card-side handle one layer needs, resolved once at load.
 *
 * A layer is EITHER an SSM layer or an attention layer, so the two projection sets alias in
 * meaning rather than in storage: `is_attention` says which five (or four) of the first nine
 * fields are live. Keeping both sets in one struct costs 5 words a layer and lets the
 * resident forward index by absolute layer number, which is what every other structure here
 * is keyed by.
 */
typedef struct {
    bool is_attention;

    /* SSM layers */
    pn_tt_weight attn_qkv;   /**< x -> the fused q|k|v stream, conv_dim wide */
    pn_tt_weight attn_gate;  /**< x -> z, the gated norm's multiplier */
    pn_tt_weight ssm_alpha;
    pn_tt_weight ssm_beta;
    pn_tt_weight ssm_out;    /**< gated -> the block output */

    /* Attention layers */
    /** x -> Q and its output gate, INTERLEAVED per head. When `attn_q_split` it is the SHAPE
     *  only (no DRAM): the joint tensor was uploaded as its two halves at load. */
    pn_tt_weight attn_q;
    bool attn_q_split;
    pn_tt_weight attn_k;
    pn_tt_weight attn_v;
    pn_tt_weight attn_output;
    /** The two DE-INTERLEAVED halves of `attn_q`, uploaded at load in the joint tensor's place
     *  when on-card attention is requested (`attn_q_split`), else staged by the attention arm.
     *  A joint projection cannot be read in place by either the QK-norm or flash decode, and
     *  permuting the WEIGHT costs one upload at load and nothing per token —
     *  `src/accel/tt/pn_tt_column_runs.h` prices that against the alternatives. */
    pn_tt_weight attn_q_only;
    pn_tt_weight attn_gate_only;
    /** The per-head QK-RMSNorm constants, at the ATTENTION head width. One gamma each,
     *  shared across heads, which is what `qwen35moe_qk_norm_heads()` does. */
    pn_tt_norm_constants attn_q_norm;
    pn_tt_norm_constants attn_k_norm;

    /* Every layer */
    pn_tt_weight ffn_gate;
    pn_tt_weight ffn_up;
    pn_tt_weight ffn_down;

    pn_tt_norm_constants attn_norm;  /**< pre-block RMSNorm */
    pn_tt_norm_constants post_norm;  /**< pre-FFN RMSNorm */
} pn_qwen35_tt_layer;

/**
 * Every activation of one token, resident in card DRAM.
 *
 * `hidden` is TWO strips and not one because the residual add is a stream op: the reader
 * runs ahead of the writer, so writing the result back over an input would be safe today and
 * would be an unstated dependency on that lead. Two strips and a slot index cost a few KiB
 * of a 31 GiB card and depend on nothing. Same argument as the llama backend's.
 */
typedef struct {
    pn_tt_activation hidden[2];
    pn_tt_activation normed;    /**< every RMSNorm's output; every projection's input */
    pn_tt_activation project;   /**< ssm_out / attn_output / ffn_down output, pre-residual */

    /* The SSM block's four projection outputs and its one projection input. */
    pn_tt_activation ssm_qkv;
    pn_tt_activation ssm_z;
    pn_tt_activation ssm_alpha;
    pn_tt_activation ssm_beta;
    pn_tt_activation ssm_gated;

    /* The attention block's three projection outputs and its one projection input. */
    pn_tt_activation attn_qg;
    pn_tt_activation attn_k;
    pn_tt_activation attn_v;
    pn_tt_activation attn_out;

    pn_tt_activation ffn_gate;
    pn_tt_activation ffn_up;
    pn_tt_activation ffn_swiglu;

    pn_tt_activation logits;

    int slot; /**< which `hidden` holds the live residual stream */
} pn_qwen35_tt_strips;

/**
 * The extra strips the ON-CARD attention block needs, allocated only when it arms.
 *
 * `q_rot` and `k_rot` are separate from `q` and `k` rather than rotated in place because
 * RoPE is a stream op whose reader runs ahead of its writer: an in-place rotation would be
 * safe today and an unstated dependency on that lead. The same argument `hidden[2]` is
 * written on.
 */
typedef struct {
    pn_tt_activation q;     /**< the de-interleaved Q half, then QK-normed in place */
    pn_tt_activation q_rot; /**< after partial NeoX RoPE — what flash decode reads */
    pn_tt_activation k_rot;
    pn_tt_activation z;     /**< the de-interleaved output-gate half, raw */
    pn_tt_activation gate;  /**< sigmoid(z); the elementwise pair's intermediate */
    pn_tt_activation gated; /**< sigmoid(z) * attn — what `attn_output` reads */
} pn_qwen35_tt_attn_strips;

/** The model-level handle, stored on the model's `gpu` slot. */
typedef struct {
    pn_tt_device *device;

    /* The `qwen35moe_mv` seam's table — the PROJECTION-ONLY arm. Still built, because it is
     * the arm the resident forward falls back to when the on-card families do not arm, and
     * because it is the reference this project's ladder was measured on. */
    pn_qwen35_tt_binding *bindings;
    int n_bindings;
    int n_expected;
    /** Qwen3.6 MoE CENSUS arm (`PN_TT_Q36_CENSUS=1`, the fit):
     *  the MoE checkpoint armed with its SHARED expert bound as the dense FFN and the routed
     *  experts not uploaded — every non-weight buffer at 3.6's geometry, for the DRAM census.
     *  Its tokens are NOT the model's; the serve says so on every arm. */
    bool moe_census;
    /** The ROUTED-EXPERT block on the card (qwen35moe_moe_tt.c, qwen36 plan Phase 5.4): the
     *  MoE checkpoint with its router, experts and shared expert on Tensix. `moe_layers`
     *  holds each layer's stacks and router; the shared expert binds as ffn_gate/up/down. */
    bool moe;
    struct pn_qwen35_tt_moe_layer *moe_layers; /**< n_layer + 1: the last is the MTP head's */
    bool moe_mtp;                              /**< moe_layers[n_layer] placed (plan S6) */
    uint32_t spec_rows_cap;                    /**< verify rows reserved (MoE; dense under the fit policy); 0 = no cap */
    struct pn_qwen35_tt_moe_strips *moe_strips;
    pn_qwen35moe_model *moe_model; /**< for the per-layer gate's CPU oracle */

    /* The RESIDENT arm. `resident` is false when any of norm / SwiGLU / residual failed to
     * arm; the forward then runs the seam arm and says so once. */
    bool resident;
    /** The logit row as the card's exact top-N (effective, per request). `sparse_env`:
     *  PN_TT_SPARSE_LOGITS as set (1/0), or -1 unset = sparse for GREEDY requests only (the
     *  greedy argmax over the top-N is the dense argmax; a sampled request keeps the dense
     *  row). `sparse_capable`: the sampling residency armed. */
    bool sparse_logits;
    int sparse_env;
    bool sparse_capable;
    uint32_t topk_pairs;   /**< this request's N: 8 greedy, 64 sampled (top_k <= 64) */
    uint64_t spec_deferred; /**< spec-sampling cycles cut short by an inexact top-N row */
    pn_qwen35_tt_layer *layers;
    int n_layers;
    pn_tt_weight output;
    pn_tt_norm_constants output_norm;
    pn_qwen35_tt_strips card;

    /* Host scratch the resident forward reads projections into. Allocated once. */
    float *h_qkv;     /**< conv_dim */
    float *h_z;       /**< ssm_d_inner */
    float *h_alpha;   /**< ssm_dt_rank, rounded up to a tile */
    float *h_beta;    /**< ssm_dt_rank, rounded up to a tile */
    float *h_gated;   /**< ssm_d_inner */
    float *h_qg;      /**< 2 * head_dim * n_head */
    float *h_qg_parts; /**< the same width: the Q half then the gate half, before interleave */
    float *h_k;       /**< head_dim * n_head_kv */
    float *h_v;       /**< head_dim * n_head_kv */
    float *h_attn;    /**< head_dim * n_head */
    float *h_embed;   /**< n_embd */
    float *h_logits;  /**< n_vocab */

    /* The ON-CARD Gated-Delta-Net arm's host scratch. Allocated only when it arms, so a run
     * without `PN_TT_ONCARD=...,gdn` pays nothing for it. `gdn_qn`/`gdn_kn` are per K-HEAD
     * and NOT GQA-repeated — the repeat is the card reader's modulo. */
    bool gdn_on_card;   /**< the two card phases replace the host recurrence */
    float *gdn_conv_out; /**< conv_dim, PHASE CONV's answer */
    float *gdn_qn;      /**< ssm_n_group * ssm_state, L2-normalised */
    float *gdn_kn;      /**< likewise */
    float *gdn_y;       /**< ssm_dt_rank * ssm_state, PHASE DELTA's answer */
    float *gdn_decay;   /**< ssm_dt_rank, already through softplus and exp */
    float *gdn_beta;    /**< ssm_dt_rank, already through the sigmoid */

    /* THE STRIP BRIDGE — the whole Gated-Delta-Net block on the card, with no activation
     * crossing PCIe. `PN_TT_GDN_STRIP`, and it needs `gdn_on_card`, both render residencies
     * and the Wt = ssm_state/32 head-norm slot; any of those missing leaves the arm off and
     * the host-mediated recurrence above running unchanged.
     *
     * What still crosses per layer is `alpha` and `beta` DOWN (two tiles, 256 B) and the gate
     * block UP (256 B), against 49,408 B. Readback 1478.09 -> 326.09 KiB/token. */
    bool gdn_strip;
    pn_tt_norm_constants gdn_l2;   /**< the q/k L2 norm's constants, one for the model */
    pn_tt_norm_constants *gdn_rms; /**< the gated RMS norm's, ONE PER LAYER: gamma is
                                    *   `ssm_norm` and that is a per-layer weight */
    bool *gdn_rms_ready;
    pn_tt_fp32_strip gdn_y_norm;   /**< RMSNorm(y), fp32, before the narrow render */
    uint32_t gdn_strip_checks;     /**< PN_TT_GDN_STRIP_CHECK layers scored so far */
    float *gdn_probe;              /**< the in-place proof's own scratch; NULL unless asked */
    pn_tt_activation gdn_y_bridge; /**< the same values as a bf16 strip, for the gate */
    /** THE CHUNK's gate scratch: `32 * ssm_dt_rank` each, ROW-MAJOR BY POSITION, because
     *  `beta = sigmoid(.)` and `decay = exp(ssm_a * softplus(.))` need a `pow` and so a `log`
     *  and the SFPU has neither — so they are host arithmetic for 32 positions at once now
     *  rather than for one. NULL unless the chunkwise family armed. */
    float *gdn_chunk_decay;
    /** PM 7: the streamed band loop sets this — a chunk's output gate (its last op) is left
     *  pending and the NEXT chunk settles it before its narrow rewrites the gate's operand. */
    bool gdn_chunk_defer_gate;
    /** PM 7 ITEM 5: this layer's gates were computed ON THE CARD (the GDN GATE family): the
     *  band loop reads nothing back, the chunk skips the host gate math, and each DELTA
     *  descriptor points at its card-built block of band `gdn_card_gate_band`. */
    bool gdn_card_gate;
    uint32_t gdn_card_gate_band;
    /** PM 7 ITEMS 5+6: the gates are ft_det_math everywhere (set when gdngate arms): the hot
     *  paths compute them on the card (no alpha/beta read back), every other TT path computes
     *  the SAME integer function on the host — so prefill, verify and plain decode agree. */
    bool gdn_det_gates;
    /** The layer in flight runs on the EXACT card gates (pn_tt_gdnc_table_arm): its DELTA
     *  descriptors look the host's gates up; set with `gdn_card_gate` by the band loop. */
    bool gdn_table_gate;
    /** PM 7 ITEM 7 step 1: the streamed band loop's PIPELINE — band b's gated RMS norm, narrow
     *  and output gate run (deferred) under band b+1's CONV / L2 / DELTA instead of before them.
     *  Same ops, same operands, same order per buffer: bit-exact. `gdn_pipe_stage` 1 = the
     *  previous band's RMS is in flight; its z / gated views are kept for the narrow + gate. */
    bool gdn_pipe;
    int gdn_pipe_stage;
    int gdn_pipe_layer;
    pn_tt_activation gdn_pipe_z, gdn_pipe_gated;
    /** Prefill lane PIPE 2 (`PN_TT_GDN_PIPE=2`): band b's DELTA is left IN FLIGHT and band
     *  b+1's widen + CONV queue behind it on the second buffer parity; band b's RMS / narrow /
     *  gate follow one band later. `gdn_pipe2_inflight` = band (layer `gdn_pipe2_layer`,
     *  parity `gdn_pipe2_parity`, views `gdn_pipe2_z` / `_gated`) has its DELTA queued. */
    bool gdn_pipe2;
    bool gdn_pipe2_inflight;
    uint32_t gdn_pipe2_parity;
    int gdn_pipe2_layer;
    int gdn_pipe2_vheads;
    pn_tt_activation gdn_pipe2_z, gdn_pipe2_gated;
    float *gdn_chunk_beta;
    /** THE PHASE-ISOLATION ARM's row shuttle: `32 * conv_dim` and `32 * v_heads * state`
     *  floats. NULL unless `PN_TT_PREFILL_GDNC_PHASE` names something other than `both`, so
     *  neither a shipped process nor the measured chunkwise arm pays anything for it. It
     *  exists to say WHICH of the two chunk kernels owns a divergence —
     *  `qwen35moe_gdn_strip_tt.c`. */
    float *gdn_split_stream;
    float *gdn_split_y;

    /* The ON-CARD GATE: a shadow host recurrence run beside the card's, on the same inputs
     * and from the same zero initial condition. NULL unless `PN_TT_GDN_PARITY` is set,
     * because it costs a second full recurrence a layer — the very thing the on-card arm
     * exists to remove. `qwen35moe_gdn_parity_tt.c`. */
    struct pn_qwen35_gdn_parity *gdn_parity;

    /* The PER-HEAD NORM PROBE: the two norms this block still runs on the host, run on the
     * card BESIDE them and scored per head. NULL unless `PN_TT_GDN_NORMCHECK` is set.
     * `qwen35moe_gdn_normcheck_tt.c`. */
    struct pn_qwen35_gdn_normcheck *gdn_normcheck;

    /* ON-CARD ATTENTION — `qwen35moe_attn_tt.c`. OPT-IN, and `PN_TT_ONCARD=all` does NOT
     * select it: this arm moves the KV cache into card DRAM at bf16 and changes the order
     * the attention sum accumulates in, so it is a different forward rather than a faster
     * one. `attn_compare` runs the host arm beside it and diffs PER HEAD. */
    bool attn_on_card;
    bool attn_compare;
    pn_qwen35_tt_attn_strips attn_card;
    /* PHASE G option 3: the PREFILL-ONLY tall set (`PN_TT_PREFILL_TILE_ROWS`). Swapped with
     * `card` / `attn_card` for the batched passes and back before the tail forward, so decode
     * never sees a tall strip. `tall_rows` 0 = prefill shares the decode strips. */
    pn_qwen35_tt_strips card_tall;
    pn_qwen35_tt_attn_strips attn_card_tall;
    uint32_t tall_rows;
    bool tall_active;
    /** Positions the last `pn_qwen35moe_prefill_tt()` call ran through batched passes; -1
     *  when it declined or a pass aborted (the prompt then ran per token). */
    int prefill_last_batched;
    /* STEP 4 S2 accounting */
    uint64_t spec_cycles, spec_drafted, spec_accepted;
    double spec_pass_ms, spec_head_ms, spec_rollback_ms, spec_total_ms;
    double spec_ssm_rows_ms, spec_attn_rows_ms; /**< the per-row mixer loops inside the pass */
    /** The gdn rows by section (host wall, each section ends on an await): alpha/beta pulls,
     *  widen + CONV rows, the q/k L2 norm, gates + DELTA rows, gated norm + gate + flush. */
    double spec_gdn_part_ms[5];
    double spec_gdn_mark_ms; /**< where the running section of spec_gdn_part_ms began */
    /* The verify pass's wall attributed to card clocks (pn_tt_device_card_clocks, by name). */
    const char *spec_clock_names[PN_TT_CARD_CLOCKS_MAX];
    double spec_clock_seconds[PN_TT_CARD_CLOCKS_MAX];
    unsigned spec_clock_count;
    bool spec_reported;
    uint64_t idle_clock_from; /**< tokens when the idle-clock window opened (0 = not yet) */
    /* STEP 4 S2b: the MTP draft head (`PN_TT_MTP=<artifact>`, tools/build_mtp_gguf.py). */
    struct {
        bool on;
        /** The MTP artifact this run will arm, or "" (none): `PN_TT_MTP`, else the trunk
         *  artifact's sibling `<trunk minus .fttw>.mtp.fttw` when that file exists
         *  (pn_qwen35_tt_mtp_resolve, before the attention arm reserves its KV region). */
        char path[1024];
        pn_qwen35_tt_layer layer;  /**< the nextn block: attention + dense FFN */
        pn_tt_weight eh_e, eh_h;   /**< eh_proj's embedding half and hidden half */
        pn_tt_norm_constants enorm, hnorm, head_norm;
        pn_tt_activation h, e;     /**< decode strips: the hidden it chains on; the embedding */
        int attn_index;            /**< its KV region, after the trunk's */
        /* The catch-up owed before the next draft: positions pend_base.., one a row, the
         * hidden of row i from the verify set (`pend_tall`) or `h` (one row), followed by
         * token pend_tok[i] (the last one is the next seed, supplied by the draft call). */
        int pend_base, pend_n;
        bool pend_tall;
        uint32_t pend_tok[16];
        /* TTFT lane (qwen35moe_mtp_prompt_defer_tt.c, `PN_TT_MTP_PROMPT_DEFER`): the LAST
         * prompt pass's MTP layer, recorded by the pass (pending), owed once the tail forward
         * has made the logits, and run before anything that reads the MTP KV or overwrites
         * the tall hidden it consumes. `prompt_owner` = the pn_qwen35moe_state it belongs to. */
        bool prompt_pending, prompt_owed, prompt_grid_kept;
        const void *prompt_owner;
        int prompt_base, prompt_rows, prompt_cap;
        uint32_t *prompt_next;
        uint64_t prompt_settled, prompt_dropped;
        double prompt_ms;
        uint64_t steps, heads;
        double ms;
        /** The last draft's confidence: its softmax over the head's top-k values (1 on the
         *  dense path). Read by the adaptive draft count (`PN_TT_SPEC_ADAPT`). */
        float last_conf;
        uint64_t adapt_stops;      /**< cycles the adaptive count ended early */
        /* SAMPLED drafts (pn_qwen35moe_tt_mtp_draft_ex): each head's top-N goes to `draw`,
         * which picks the draft; `draw_index` counts the heads of the current draft call. */
        pn_spec_draft_fn draw;
        void *draw_ctx;
        int draw_index;
        /* S2c: the DRAFT VOCABULARY (`PN_TT_MTP_VOCAB_TILES=N`): the LM head's first N tile
         * columns (ids < 32N, BPE rank ~ frequency) + the special-token tail, staged as its own
         * weight; its logits strip is pre-filled -inf so top-k never picks an unwritten tile.
         * Drafts only — the verify still uses the whole head. */
        bool sub;
        pn_tt_weight head_sub;
        pn_tt_activation logits_sub;
        uint32_t sub_tiles, sub_tail_first;
        /* THE TREE PROBE (`PN_TT_SPEC_TREE_PROBE=1`, diagnostics, no card work): each head of
         * the current draft call keeps its top pairs; the verify then books, per draft
         * position, how often it was reached / accepted and the rank of the emitted token
         * among the head's pairs where the draft was refused — what a sibling candidate at
         * that position would have bought (qwen35_tt_tree_probe). */
        int probe_heads;
        uint32_t probe_ids[16][PN_QWEN35_TT_TOPK_GREEDY];
        float probe_vals[16][PN_QWEN35_TT_TOPK_GREEDY];
        uint32_t probe_n[16], probe_best[16];
        uint64_t probe_cycles, probe_rows, probe_tokens, probe_skipped;
        uint64_t probe_reach[16], probe_acc[16];
        uint64_t probe_rank[16][PN_QWEN35_TT_TOPK_GREEDY + 1]; /**< [pos][rank], last = miss */
    } mtp;
    float *h_attn_card;  /**< the card answer, for the crossed control only */

    /* THE BATCHED PREFILL's host scratch — one [32 x width] buffer per strip it has to row
     * select through. NULL until a batched prefill runs, so a decode-only process pays
     * nothing for it. `qwen35moe_prefill_tt.c` owns the definition. */
    struct pn_qwen35_tt_batch *batch;

    uint32_t topk_index[PN_QWEN35_TT_TOPK_PAIRS];
    float topk_value[PN_QWEN35_TT_TOPK_PAIRS];
    uint32_t topk_count;
    /** LAUNCH lane (`PN_TT_ROW_RESET`): the host row the last sparse head wrote — `-INFINITY`
     *  everywhere but `row_index[0..row_count)` — or NULL when that is not known (a dense
     *  head ran, or none yet). The next sparse head resets those entries instead of the row. */
    float *row_sparse;
    uint32_t row_index[PN_QWEN35_TT_TOPK_PAIRS];
    uint32_t row_count;
    /** Speculative sampling: every verify row's top-N, kept by pn_qwen35_tt_head_rows(). */
    uint32_t spec_topk_index[16][PN_QWEN35_TT_TOPK_PAIRS];
    float spec_topk_value[16][PN_QWEN35_TT_TOPK_PAIRS];
    uint32_t spec_topk_count[16];

    /* The evidence that the card was used. Byte-identical output is NOT evidence — the card
     * and a silent CPU fallback have returned the same md5 in this project — so the split is
     * counted and reported, and an acceptance run reads `on_host == 0`. */
    uint64_t mv_on_card;
    uint64_t mv_on_host;   /**< a weight the table does not hold; should be 0 */
    uint64_t mv_failed;    /**< the card was asked and refused; should be 0 */
    int failure_reported;  /**< the first failure prints; the rest only count */
    /** S3a: the last out-projection already wrote hidden + projection into the spare strip
     *  (pn_qwen35_tt_project_out), so the next residual step only flips the slot. Cleared by
     *  every pn_qwen35_tt_project() and by the residual step that consumes it. */
    int residual_fused;
    /** P0 persistent norm role (accel/tt/pn_tt_norm_role.h): `norm_role_tried` once the arm
     *  was attempted; `norm_role_active` for the forward in flight, whose norms are role steps
     *  and whose residual steps fold into them (`norm_role_cursor`). */
    bool norm_role_tried;
    bool norm_role_active;
    pn_tt_norm_role_cursor norm_role_cursor;

    /* The wall-clock reconciliation. The four device counters time the SEAMS — a tilize and
     * a link write, a dispatch, a link read and a de-tilize, an on-card descriptor — and
     * nothing else. Everything the host computes BETWEEN two seams is invisible to all four,
     * so a phase sum quoted as a token understates it by however much host arithmetic the
     * rung still owns. These accumulate the named host regions, so whatever is left over is a
     * real unknown rather than a region nobody thought to time.
     *
     * All seconds, cumulative over `tokens`. `forward_wall_seconds` brackets the whole
     * forward, so `forward_wall - (upload+dispatch+readback+oncard) - the four below` is the
     * unattributed slice. Sampling and de-tokenisation are OUTSIDE the forward and therefore
     * outside `forward_wall_seconds` too — they are read by subtracting it from the CLI's
     * own generate-loop wall. */
    double forward_wall_seconds;    /**< entry to exit of pn_qwen35moe_forward_tt */
    double host_recurrence_seconds; /**< pn_qwen35moe_ssm_recurrence, 24 layers a token */
    double host_attention_seconds;  /**< pn_qwen35moe_attn_core, 8 layers a token */
    double host_embed_seconds;      /**< the embedding-row gather and dequant */
    double host_logit_seconds;      /**< the sparse -INFINITY fill and the logit-row copy */

    int report_every;
    uint64_t tokens;

    /* ------------------------------------------------------------------
     * DID ANY CALLER ACTUALLY ASK FOR THE BATCHED PREFILL?
     *
     * The chunkwise prefill shipped ON by default on 2026-08-25 at ~24x, and on 2026-08-26 it
     * was found to reach exactly ONE of the three entry points that prefill a prompt: the
     * chat CLI (`pn_stream_tokens`) and the serve (`sg_prompt_chunk`) both had per-token
     * loops that never called `pn_qwen35moe_prefill_tt()` at all. Nothing refused, nothing
     * errored, nothing printed — a 1,630-token prompt simply took 295 s and 1,632 forwards,
     * and two sessions read that as the rung having regressed.
     *
     * A COUNTER IN THE MODEL CATCHES IT WHATEVER THE CALLER IS. `prefill_passes` is bumped by
     * the batched path; `sequential_prefill_run` counts consecutive single-position forwards
     * from position 0. When the run crosses `PN_QWEN35_TT_SEQUENTIAL_PREFILL_ALARM` with no
     * batched pass ever having run, the forward says so ONCE. A wall figure has no
     * denominator; this does.
     * ------------------------------------------------------------------ */
    uint64_t prefill_passes;          /**< batched passes this process has run */
    uint64_t sequential_prefill_run;  /**< consecutive one-position forwards */
    bool sequential_prefill_said;     /**< the alarm prints once, then only counts */

    /* ------------------------------------------------------------------
     * MULTI-TURN: the batched prefill's checkpoint (qwen35moe_prefill_checkpoint_tt.c).
     * The carried GDN state at the last full PASS boundary of the last prefill, copied on the
     * card, with the token ids it covers. The attention K/V rows below `pos` are the card's
     * own and stay valid while the same sequence keeps running at or above `pos`.
     * ------------------------------------------------------------------ */
    struct pn_qwen35_tt_checkpoint {
        bool valid;
        bool restored;        /**< restored for `owner` at `pos`, the prefill not yet run */
        int pos;              /**< positions [0, pos), a multiple of the pass rows */
        const pn_state *owner;
        uint32_t *tokens;     /**< tokens[0, pos) */
        int capacity;
        /* A capture under way in this prefill: every SSM layer's state is copied when its
         * band loop reaches `pending_at`; the checkpoint is valid once all of them were. */
        int pending_at;       /**< 0 = no capture in this prefill */
        int pending_layers;
        double capture_ms;
        unsigned long long verify_words;
        long long verify_differ; /**< PN_TT_CKPT_VERIFY tally; -1 once a read failed */
        /* The next prefill of `ceiling_owner` checkpoints at or below `ceiling` (the end of
         * the history its next turn resends; pn_qwen35moe_tt_checkpoint_ceiling). Used once. */
        int ceiling;
        const pn_state *ceiling_owner;
    } ckpt;
    /** The pn_state whose sequence the card's K/V and carried state hold now (the last one a
     *  forward, a verify or a prefill ran for). */
    const pn_state *card_owner;
    /** The FIT POLICY (qwen35moe_tt_fit_plan.h): main's placement does not fit this card, so
     *  options A (at load), B and C (tall strips) apply. False = main's placement exactly. */
    bool fit_policy;
    /** The fit policy's LEVERS were picked at the decide (qwen35moe_tt_fit_plan.h): the tall
     *  set takes `fit_tall_rows` bands and `fit_outproj_mirrors`, the GDNC its copy count
     *  (pn_tt_gdnc_request_replicas). False = the env / defaults. */
    bool fit_levers;
    uint32_t fit_tall_rows;
    bool fit_outproj_mirrors;
    /** Lever V: the MTP draft reads its LM-head columns of `output` in place (no staged copy). */
    bool fit_draft_view;
    /** The load could not reserve the prefix checkpoint's twins: no checkpoint is taken and no
     *  prefill retries the reservation (qwen35moe_fit_tt.c). */
    bool ckpt_off;
    /** The last `pn_qwen35moe_prefill_tt()` ABORTED part way: the card's state was partly
     *  written, the session was invalidated, and the caller must fail the request rather than
     *  replay it per token (pn_qwen35_tt_prefill_abort). */
    bool prefill_aborted;
} pn_qwen35_tt;

/* ============================================================
 * MULTI-TURN CHECKPOINT — qwen35moe_prefill_checkpoint_tt.c
 * ============================================================ */

/** `PN_TT_PREFILL_CKPT=0` turns capture and resume off (the control). Default on. */
bool pn_qwen35_tt_ckpt_enabled(void);

/** Every card forward, verify and prefill: a forward for another sequence, or below the
 *  checkpoint, invalidates it (the K/V below it is about to be rewritten). */
void pn_qwen35_tt_ckpt_note(pn_qwen35_tt *g, const pn_state *state, int pos);
/** Forget the card's sequence (a warm-up ran): no state owns it, no checkpoint is valid. */
void pn_qwen35_tt_ckpt_forget(pn_qwen35_tt *g);

/** May a batched prefill start at `pos` != 0 for `state`? Only right after a resume. */
bool pn_qwen35_tt_ckpt_restored_at(const pn_qwen35_tt *g, const pn_state *state, int pos);

/**
 * Before a prefill's passes: the batched rows run from `base` (0, or the restored checkpoint)
 * to `batch_end`; the capture position is the last 32-row band boundary in that range, and at
 * or below the ceiling the serve set for this state (qwen35moe_checkpoint_band.h). Each
 * SSM layer's state is copied there by pn_qwen35_tt_ckpt_band_done(). A capture overwrites the
 * previous checkpoint, so that one is dropped first.
 */
void pn_qwen35_tt_ckpt_plan(pn_qwen35_tt *g, const pn_state *state, int base, int batch_end);

/** The chunk band loop finished a band of SSM layer `ssm_index` ending at absolute position
 *  `end`: at the capture position, copy that layer's state. */
void pn_qwen35_tt_ckpt_band_done(pn_qwen35_tt *g, int ssm_index, int end);

/** After the passes: valid when every layer was captured; `tokens` are this prefill's, from
 *  `base`. */
void pn_qwen35_tt_ckpt_commit(pn_qwen35_tt *g, const pn_state *state, const uint32_t *tokens,
                              int base, int n_ssm);

/** Free the host side. */
void pn_qwen35_tt_ckpt_release(pn_qwen35_tt *g);

/* ============================================================
 * PREFILL alpha/beta PREFETCH — qwen35moe_prefill_prefetch_tt.c
 * ============================================================ */

/** `PN_TT_PREFILL_AB_PREFETCH=0` turns it off. Default on. */
bool pn_qwen35_tt_ab_prefetch_enabled(void);
/** Before band b's chunk mixers: read band `band` (= b+1) of `ssm_alpha` / `ssm_beta` (`rows`
 *  rows) into the host rows given, inside the CONV and DELTA windows. */
void pn_qwen35_tt_ab_prefetch_arm(pn_qwen35_tt *g, int ssm_index, int band, int rows,
                                  float *alpha, uint32_t alpha_width, float *beta,
                                  uint32_t beta_width);
/** After band b's mixers: clear a window hook that never ran. */
void pn_qwen35_tt_ab_prefetch_settle(pn_qwen35_tt *g);
/** Were band `band`'s alpha (`beta` false) or beta rows of this layer already read? */
bool pn_qwen35_tt_ab_prefetched(const pn_qwen35_tt *g, int ssm_index, int band, bool beta);
/** Forget any pending read (a new pass). */
void pn_qwen35_tt_ab_prefetch_reset(void);
/** One line under PN_TT_PREFILL_REPORT. */
void pn_qwen35_tt_ab_prefetch_report(void);

/* PHASE W — the alpha/beta reads on the READER THREAD (pn_tt_reader.h): every band of a layer
 * posted once its projections land, taken band by band. `post` returns false when the reader
 * is not available (the prefetch above then runs); `take` returns -1 when a band's read
 * failed (the caller reads it inline); `end` drains — call it on EVERY exit. */
bool pn_qwen35_tt_ab_reader_post(pn_qwen35_tt *g, int rows, float *alpha, uint32_t alpha_width,
                                 float *beta, uint32_t beta_width);
int pn_qwen35_tt_ab_reader_take(pn_qwen35_tt *g, int band);
void pn_qwen35_tt_ab_reader_end(pn_qwen35_tt *g);
/* PHASE W wavefront: the NEXT SSM layer's bands [first, first + count), gated on the card
 * having executed its alpha/beta projections; `promote` makes them the current layer's once
 * every band is posted; `abandon` waits out everything (abort / end of pass). */
struct pn_tt_reader_gate;
bool pn_qwen35_tt_ab_reader_post_next(pn_qwen35_tt *g, int ssm_index, int first, int count,
                                      int rows, float *alpha, uint32_t alpha_width, float *beta,
                                      uint32_t beta_width, const struct pn_tt_reader_gate *gate);
bool pn_qwen35_tt_ab_reader_promote(int ssm_index);
void pn_qwen35_tt_ab_reader_abandon(pn_qwen35_tt *g);
/* PM 6: the speculative VERIFY's alpha/beta through the reader (gated on member 0, the stream
 * kicked) instead of a stream boundary; `wait` before the host math (`PN_TT_SPEC_AB_READER=0`
 * off). post false = read them the old way. */
bool pn_qwen35_tt_ab_verify_post(pn_qwen35_tt *g, int rows, float *alpha, uint32_t alpha_width,
                                 float *beta, uint32_t beta_width);
int pn_qwen35_tt_ab_verify_wait(pn_qwen35_tt *g);

/* PHASE W2 (qwen35moe_prefill_overlap_tt.c): an SSM layer's FFN on the grid, a super-band at a
 * time, under the GDN chain. */
/** The next SSM layer a super-band job also prepares (`PN_TT_PREFILL_WAVEFRONT=1`): its weights,
 *  index, and the host rows its alpha/beta reads land in. `d` NULL = none. */
typedef struct pn_qwen35_tt_overlap_next {
    const pn_qwen35_tt_layer *d;
    int ssm_index;
    int rows;
    float *alpha, *beta;
    uint32_t alpha_width, beta_width;
} pn_qwen35_tt_overlap_next;

uint32_t pn_qwen35_tt_overlap_bands(void);
/** G.3: the super-band span for a pass of `bands` (`PN_TT_PREFILL_OVERLAP=S` fixes it, 0 off;
 *  unset: half the pass, at least 13). The replica map is keyed with the same span. */
uint32_t pn_qwen35_tt_overlap_span(uint32_t bands);
bool pn_qwen35_tt_wavefront_enabled(void);
bool pn_qwen35_tt_overlap_layer(pn_qwen35_tt *g, int ssm_index, uint32_t bands);
/** O.1: plan layer `d`'s super-band jobs over `bands` (`pipe_depth`: bands the chain's gates
 *  trail their band). @return 0, or -1 (no plan; the caller runs the layer without overlap). */
int pn_qwen35_tt_overlap_layer_begin(pn_qwen35_tt *g, const pn_qwen35_tt_layer *d,
                                     uint32_t bands, uint32_t pipe_depth,
                                     const pn_qwen35_tt_overlap_next *next,
                                     const pn_tt_activation *mixer, const pn_tt_weight *out_weight);
/** A: may an attention layer of `bands` run its out-proj and FFN under its band loop
 *  (`PN_TT_PREFILL_OV_ATTN`, default on)? */
bool pn_qwen35_tt_overlap_attention(pn_qwen35_tt *g, uint32_t bands);
/** Run the steps due after band `band` (`band == bands`: the layer's tail, everything left). */
int pn_qwen35_tt_overlap_tick(pn_qwen35_tt *g, uint32_t band);
/** Drop the layer's plan (an error path): the grid waited, nothing more runs. */
void pn_qwen35_tt_overlap_abort(pn_qwen35_tt *g);
/** Has the wavefront already run SSM layer `ssm_index`'s attn-norm, in-proj and read posts for
 *  all `bands`? `forget` drops it (a new pass, an abort). */
bool pn_qwen35_tt_overlap_prepared(int ssm_index, uint32_t bands);

/** PM 7 ITEM 5: the prepared layer's gate blocks were issued on the card (all `bands`). */
bool pn_qwen35_tt_overlap_gates_prepared(int ssm_index, uint32_t bands);

/** PM 7 ITEM 5: the wavefront prepares the next layer's gates on the card (not reads). */
void pn_qwen35_tt_overlap_set_card_gates(bool on);
void pn_qwen35_tt_overlap_forget(void);
double pn_qwen35_tt_overlap_take_seconds(void);
void pn_qwen35_tt_overlap_report(void);

/** Consecutive single-position forwards that mean "this is a prefill, and it is not batched".
 *  64 is well past any chat turn's generation and far short of a real prompt. */
#define PN_QWEN35_TT_SEQUENTIAL_PREFILL_ALARM 64u

/**
 * Bring up the resident-activation arm: the on-card op residencies, the strips and the norm
 * constants.
 *
 * Never fatal. On any refusal it leaves `g->resident` false and the caller runs the
 * projection-only arm, which is slower and not wrong. Implemented in
 * `qwen35moe_resident_tt.c`.
 *
 * @return 0 whether or not the arm came up; -1 only on a bad argument
 *
 * Thread-safe: no    Reentrant: no    Signal-safe: no
 */
int pn_qwen35_tt_resident_arm(pn_qwen35_tt *g, pn_qwen35moe_model *m,
                              const pn_qwen35moe_config *cfg);

/**
 * The MoE block (qwen35moe_moe_tt.c). `bind` places every layer's expert stacks and router
 * after the dense load; `arm` boots the block's residencies and allocates its strips (inside
 * the resident arm, after the on-card families); `ffn` replaces the dense SwiGLU for one layer
 * and leaves shared + routed in `card.project`; `release` frees the host side.
 */
int pn_qwen35_tt_moe_bind(pn_qwen35_tt *g, pn_qwen35moe_model *m, const pn_qwen35moe_config *cfg);
int pn_qwen35_tt_moe_arm(pn_qwen35_tt *g, const pn_qwen35moe_config *cfg);
int pn_qwen35_tt_moe_ffn(pn_qwen35_tt *g, const pn_qwen35_tt_layer *d, int layer);
/** BATCHED PREFILL (plan S1): the MoE FFN of one layer over a pass's `bands` bands (the tall
 *  strips seeked to the pass): routed groups in LIST mode + the shared expert, into
 *  `card.project`. `ready` says whether the prefill strips were armed. */
bool pn_qwen35_tt_moe_prefill_ready(const pn_qwen35_tt *g);
int pn_qwen35_tt_moe_prefill_ffn(pn_qwen35_tt *g, const pn_qwen35_tt_layer *d, int layer,
                                 uint32_t bands);
/** The MoE FFN of a verify pass (plan S6): CONCAT routing over the pass's `rows` (one band),
 *  so every row's output is the decode token's, bit for bit. */
/** Place the MTP head's MoE layer (extra artifact: mtp.ffn_*_exps, mtp.ffn_gate_inp and the
 *  shared gate) as moe_layers[n_layer]. 0, or -1 (no MTP on the MoE). */
int pn_qwen35_tt_moe_bind_mtp(pn_qwen35_tt *g, const pn_qwen35moe_config *cfg);
int pn_qwen35_tt_moe_verify_ffn(pn_qwen35_tt *g, const pn_qwen35_tt_layer *d, int layer,
                                uint32_t rows);
void pn_qwen35_tt_moe_release(pn_qwen35_tt *g);
/** Tiles the MoE block's SwiGLU / scale descriptors carry (k * expert ff + shared ff) / 32. */
uint32_t pn_qwen35_tt_moe_eltwise_tiles(const pn_qwen35moe_config *cfg);

/** Release whatever `pn_qwen35_tt_resident_arm` allocated on the host side. */
void pn_qwen35_tt_resident_release(pn_qwen35_tt *g);

/**
 * One token through the resident-activation forward.
 *
 * The residual stream, both RMSNorms, the dense SwiGLU and the LM head stay in card DRAM;
 * only the values the host still computes — the Gated-Delta-Net recurrence's operands and
 * the attention block's — cross the link, and each crosses it exactly once.
 *
 * @return the logit row, owned by `g`, or NULL on failure
 *
 * Thread-safe: no    Reentrant: no    Signal-safe: no
 */
float *pn_qwen35_tt_resident_forward(pn_qwen35_tt *g, pn_qwen35moe_model *m,
                                     uint32_t token, pn_qwen35moe_state *s, int pos,
                                     struct fw_pool *pool);

/* ============================================================
 * The ON-CARD Gated-Delta-Net arm — qwen35moe_gdn_tt.c
 *
 * OPT-IN, and it has never executed on silicon. It replaces `pn_qwen35moe_ssm_recurrence()`
 * with the same six steps, two of them run as card descriptors over state that lives in card
 * DRAM across tokens. Until the two per-head norms move onto a second narrow reduce residency
 * it is SLOWER than the host recurrence, for the reason `pn_tt_gdn.h` states as a number.
 * ============================================================ */

/**
 * Arm the GDN residency, allocate every SSM layer's carried state, and TRANSPOSE each layer's
 * conv weight into the tap-major planes the card reads.
 *
 * Reads `PN_TT_ONCARD` through `pn_tt_device_oncard_requested()`, so an unset variable leaves
 * `g->gdn_on_card` false and the host recurrence in place.
 *
 * @return 0 whether or not it armed; -1 on a bad argument or a staging failure
 *
 * Thread-safe: no    Reentrant: no    Signal-safe: no
 */
int pn_qwen35_tt_gdn_arm(pn_qwen35_tt *g, pn_qwen35moe_model *m,
                         const pn_qwen35moe_config *cfg);

/** Release the host scratch `pn_qwen35_tt_gdn_arm` allocated. Safe on NULL and on an arm
 *  that never came up. */
void pn_qwen35_tt_gdn_release(pn_qwen35_tt *g);

/* ============================================================
 * ON-CARD ATTENTION — implemented in qwen35moe_attn_tt.c
 * ------------------------------------------------------------
 * The op family (`src/accel/tt/pn_tt_attention.c`) already existed and is green on silicon
 * for MiniCPM5-1B. What this file adds is Ornith's SHAPE: the de-interleaved Q/gate
 * projections, the per-head QK-RMSNorms at head_dim/32 tile columns, and the SIGMOID output
 * gate. See the file note for why each landed where it did.
 * ============================================================ */

/**
 * Bring up on-card attention if `PN_TT_ONCARD` names it.
 *
 * Stages the two de-interleaved halves of every attention layer's joint Q+gate projection,
 * uploads the QK-norm constants, arms the per-head reduce residency at the ATTENTION head
 * width and the RoPE/memory/attention triple, and allocates the extra strips.
 *
 * A REFUSAL, not a fallback: attention was asked for by name, so a missing program, an
 * unexpressible rotary width or a residency that will not arm returns -1 rather than leaving
 * the block on the host with different numerics and nothing saying so.
 *
 * @return 0 (armed, or never asked for), -1 on a refusal
 *
 * Thread-safe: no    Reentrant: no    Signal-safe: no
 */
int pn_qwen35_tt_attn_arm(pn_qwen35_tt *g, pn_qwen35moe_model *m,
                          const pn_qwen35moe_config *cfg);

/** Did the on-card attention arm come up for this run? */
bool pn_qwen35_tt_attn_is_on_card(const pn_qwen35_tt *g);

/** `model`'s qwen35 TT residency, or NULL: no residency, or not a qwen35-family model at all
 *  (its weights are then not a pn_qwen35moe_model, so nothing may read them as one). */
pn_qwen35_tt *pn_qwen35_tt_of(const pn_model *model);

/**
 * One attention layer entirely on the card, leaving the block output in `g->card.project`.
 *
 * NOTHING crosses the link: q, k, v, the gate and the answer all stay in DRAM strips, and
 * the KV cache is the card's.
 *
 * @param attn_index  the ATTENTION-layer ordinal (`pn_qwen35moe_model_layer_index()`), which
 *                    is what indexes the card's per-layer K/V regions
 * @param seq_len     LIVE cached positions, i.e. `min(pos + 1, max_seq)`
 * @return            0, or -1 with the reason on stderr
 *
 * Thread-safe: no    Reentrant: no    Signal-safe: no
 */
int pn_qwen35_tt_attn_block_card(pn_qwen35_tt *g, const pn_qwen35_tt_layer *d,
                                 const pn_qwen35moe_config *cfg, int attn_index, int pos,
                                 int seq_len);

/**
 * THE TWO HALVES OF THAT BLOCK, SPLIT AT THE POSITION SEAM.
 *
 * `..._project_card` is the part a batched prefill runs ONCE for up to 32 rows: four
 * projections over `g->card.normed`, per-row by construction. `..._core_card` is the part
 * that cannot be batched — the QK-norms, the rotary, the K/V append and flash decode each
 * take ONE position — so it runs once per row over a strip whose logical row 0 the caller
 * has just rewritten. The output projection stays with the caller, because the batched arm
 * defers it until all the rows' gates are back in one strip.
 *
 * `pn_qwen35_tt_attn_block_card()` is exactly these two plus that projection, so the shipped
 * decode path is byte-for-byte the same sequence it was.
 *
 * @return 0, or -1 with the reason on stderr
 *
 * Thread-safe: no    Reentrant: no    Signal-safe: no
 */
int pn_qwen35_tt_attn_project_card(pn_qwen35_tt *g, const pn_qwen35_tt_layer *d);

/** @copydoc pn_qwen35_tt_attn_project_card */
int pn_qwen35_tt_attn_core_card(pn_qwen35_tt *g, const pn_qwen35_tt_layer *d,
                                const pn_qwen35moe_config *cfg, int attn_index, int pos,
                                int seq_len);

/** Prefill lane ATTENTION PIPE (`PN_TT_ATTN_PIPE`): the chunk core split — QK norms + RoPE
 *  (`prep`), then K/V appends + attention left in flight (`issue`, pn_tt_attention_settle()
 *  before its output is read). */
int pn_qwen35_tt_attn_chunk_prep(pn_qwen35_tt *g, const pn_qwen35_tt_layer *d,
                                 const pn_qwen35moe_config *cfg, int base_pos, int rows);
int pn_qwen35_tt_attn_chunk_issue(pn_qwen35_tt *g, int attn_index, int base_pos, int rows);
/** Deep-queued prefill attention (pn_tt_attn_deep.h): a band's K/V appends (awaited), and its
 *  head descriptors queued on the members (nothing awaited; pn_tt_attention_chunk_drain()). */
int pn_qwen35_tt_attn_chunk_append(pn_qwen35_tt *g, int attn_index, int base_pos, int rows);
int pn_qwen35_tt_attn_chunk_queue(pn_qwen35_tt *g, int attn_index, int base_pos, int rows);

/** K4H (pn_tt_kv_rotate.h): after RoPE, `q_rot * H -> q` and `k_rot * H -> attn_k` when the
 *  KV type rotates (else a no-op); then `_query()` / `_key()` name the strips the attention
 *  reads and the append writes from: `q` / `attn_k` rotating, `q_rot` / `k_rot` otherwise. */
int pn_qwen35_tt_attn_rotate(pn_qwen35_tt *g, const pn_qwen35moe_config *cfg);
pn_tt_activation *pn_qwen35_tt_attn_query(pn_qwen35_tt *g);
pn_tt_activation *pn_qwen35_tt_attn_key(pn_qwen35_tt *g);

/**
 * The same block over a CHUNK of `rows` consecutive positions, in rows 0..rows-1 of the
 * strips the projections already wrote.
 *
 * EVERY STEP EXCEPT ATTENTION ALREADY BATCHED AND NONE OF THEM CHANGES HERE. The two
 * per-head QK-RMSNorms are `row_blocks = heads` over a `[1, heads*S]` strip and a row reduce
 * reduces ALONG the row, so row t of row block h is position t of head h; the sigmoid gate
 * is per element. RoPE, the KV append and attention take the chunk forms, and that is the
 * whole difference — which is why this is a sibling of `_core_card` rather than a mode
 * inside it: the shipped decode sequence stays byte for byte what it was.
 *
 * `base_pos` MUST be a multiple of 32 and `rows` 1..32.
 *
 * @return 0, or -1 with the reason on stderr
 *
 * Thread-safe: no    Reentrant: no    Signal-safe: no
 */
int pn_qwen35_tt_attn_core_chunk(pn_qwen35_tt *g, const pn_qwen35_tt_layer *d,
                                 const pn_qwen35moe_config *cfg, int attn_index, int base_pos,
                                 int rows);

/** Free the batched prefill's host scratch. Safe on NULL and on a run that never batched.
 *  The entry point itself is declared in `qwen35moe.h`, because the engine calls it. */
void pn_qwen35_tt_prefill_release(pn_qwen35_tt *g);
/** PHASE G option 3: allocate the prefill-only tall strip set (`PN_TT_PREFILL_TILE_ROWS`),
 *  after the decode strips and the attention strips exist. 0 also when the knob is off. */
int pn_qwen35_tt_prefill_tall_alloc(pn_qwen35_tt *g);

/**
 * The prefill-only tall set as placement requests, one per strip in the batched passes' strip
 * order: each strip's width and mirror count, with `tile_count` 0 for a strip that gets no
 * tall copy (the logit row; under `fit_policy`, option C, also the joint `attn_qg` when
 * attention is on the card). The gated out-projection strips take `device_replicas` mirrors
 * when `outproj_mirrors`. Without `fit_policy` the set is exactly main's.
 * @return requests written, at most `max`
 */
size_t pn_qwen35_tt_tall_requests(pn_qwen35_tt *g, uint32_t device_replicas,
                                  bool outproj_mirrors, bool fit_policy,
                                  pn_tt_strip_request *out, size_t max);
/** The prefill-only tall set's height from `PN_TT_PREFILL_TILE_ROWS` (default 64): 0 when off
 *  or not a power of two in 2..64. */
uint32_t pn_qwen35_tt_tall_rows_wanted(void);
/** `PN_TT_OUTPROJ_MIRRORS` (default on): the gated tall strips are mirrored. */
bool pn_qwen35_tt_outproj_mirrors_wanted(void);

/**
 * THE TT LOAD-TIME FIT (qwen35moe_fit_tt.c).
 *
 * `fit_decide`: with the artifact open and no weight placed, compute main's DRAM floor
 * (qwen35moe_tt_fit_plan.h) and, when it does not leave the card's headroom or leaves its
 * fullest bank under the bank headroom, take the fit policy: `attn_q` goes up as its halves
 * (option A). Never fails the load; prints the decision.
 *
 * `prefill_reserve`: after the tall strips, count every buffer a batched prefill used to
 * allocate later (pn_tt_prefill_reserve.h) against the real per-bank free bytes, and allocate
 * them. Mandatory buffers short -> -1 (the load refuses, the shortfall named). The prefix
 * checkpoint short -> served with the checkpoint OFF (`g->ckpt_off`), said once.
 */
void pn_qwen35_tt_fit_decide(pn_qwen35_tt *g, const pn_qwen35moe_config *cfg, int n_ssm,
                             int n_attn);
int pn_qwen35_tt_prefill_reserve(pn_qwen35_tt *g);

/** The exact card gates (qwen35moe_gate_tables.h) are wanted: `PN_TT_GDN_TABLE_GATES` unset or
 *  not `0`. The fit floor prices them; the arm then builds, checks and uploads them. */
bool pn_qwen35_tt_gate_tables_wanted(void);

/** THE EXACT CARD GATES in decode and the verify, for layer `ssm_index`: the tables armed (no
 *  det gates), `PN_TT_GDN_TABLE_DECODE` not 0, and the layer's decode table block bound — then
 *  the next DELTA descriptors name it (pn_tt_gdn_table_gate, cleared by the strip function
 *  after its DELTAs) and the caller reads no alpha/beta back and passes NULL gates. false:
 *  the host gates, nothing changed. */
bool pn_qwen35_tt_table_decode_gates(pn_qwen35_tt *g, int ssm_index);

/** What a DRAM bank must keep when the gate tables go in: the 128 MiB bank rule, plus the MTP
 *  verify slots' share when they are still to come (main's placement allocates them at the
 *  first verify). */
uint64_t pn_qwen35_tt_gate_tables_keep_free(pn_qwen35_tt *g);

/** L2: mirror the decode `ssm_gated` / attention `gated` strips from the DRAM left after the
 *  arm and republish the replica map (qwen35moe_outproj_mirrors_tt.c). 0 also when off or
 *  declined; -1 on a failed allocation or publish. */
int pn_qwen35_tt_decode_outproj_mirrors(pn_qwen35_tt *g);

/**
 * A batched prefill that failed PART WAY: the card's carried state and K/V hold some of the
 * prompt. Logs `reason`, drops the prefix checkpoint, forgets that the card holds `state`'s
 * sequence, rewinds `state` to position 0 and sets `g->prefill_aborted`, so the caller fails
 * the request (pn_qwen35moe_prefill_tt_aborted) and the next request starts a fresh sequence,
 * which re-zeroes the carried state. Never reused.
 * @return NULL always
 */
tm_tensor *pn_qwen35_tt_prefill_abort(pn_qwen35_tt *g, pn_state *state, const char *reason,
                                      int positions_done);

/** STEP 4 S2 — the speculative verify (qwen35moe_prefill_tt.c). */
int pn_qwen35_tt_spec_verify(pn_qwen35_tt *g, pn_qwen35moe_model *m,
                             const pn_qwen35moe_config *cfg, pn_qwen35moe_state *st,
                             const uint32_t *tokens, int rows, int base_pos,
                             uint32_t *argmax_out);
int pn_qwen35_tt_spec_rollback(pn_qwen35_tt *g, const pn_qwen35moe_config *cfg, int rows,
                               int last_kept);
/** STEP 4 S2b — the MTP draft head (qwen35moe_resident_tt.c). arm: place the second
 *  artifact's weights (after the attention arm, which reserved its KV region). step: one MTP
 *  position — hidden row `h_row` of `h_src` (NULL: the chained `mtp.h`) and the embedding of
 *  `token` through eh_proj, the nextn block at `pos`, and with `want_head` its head's argmax.
 *  prefill_layer (qwen35moe_prefill_tt.c, inside a batched pass): the same over a pass's rows,
 *  writing the MTP KV for the prompt. */
void pn_qwen35_tt_mtp_resolve(pn_qwen35_tt *g, const char *trunk_artifact);
/** Rows a default verify takes: the MTP drafts a cycle (`PN_TT_SPEC`, 3 unless set) + the seed. */
uint32_t pn_qwen35_tt_mtp_verify_rows(void);
/** LM-head columns the MTP arm stages for its draft vocabulary out of `total` (0 = the whole
 *  head, read in place; `PN_TT_MTP_VOCAB_TILES`), as pn_qwen35_tt_mtp_arm takes them. */
uint32_t pn_qwen35_tt_mtp_draft_columns(uint32_t total);

/** `PN_TT_MTP_DRAFT_VIEW`: 1 the draft reads the LM head in place (lever V), 0 it stages its
 *  copy, -1 unset (the fit decides). */
int pn_qwen35_tt_mtp_draft_view_env(void);
int pn_qwen35_tt_mtp_arm(pn_qwen35_tt *g, pn_qwen35moe_model *m, const pn_qwen35moe_config *cfg);
int pn_qwen35_tt_mtp_step(pn_qwen35_tt *g, pn_qwen35moe_model *m, const pn_qwen35moe_config *cfg,
                          pn_qwen35moe_state *st, uint32_t token, const pn_tt_activation *h_src,
                          uint32_t h_row, int pos, bool want_head, uint32_t *argmax_out);
/** `PN_TT_DUMP_HIDDEN` + `_LAST=1` under speculative decode: the COMMITTED verify rows' last-trunk-layer
 *  hidden (row copy + one row read each), records as the per-token dump's. 0 when off. */
int pn_qwen35_tt_dump_verify_rows(pn_qwen35_tt *g, const pn_qwen35moe_config *cfg,
                                  const pn_tt_activation *src, const uint32_t *tokens, int rows,
                                  int pos);
/** `PN_TT_SPEC_TREE_PROBE=1`: the tree probe is on (read once). */
bool pn_qwen35_tt_tree_probe_on(void);
int pn_qwen35_tt_mtp_head_row(pn_qwen35_tt *g, pn_qwen35moe_state *st,
                              const pn_tt_activation *src, uint32_t row, uint32_t *argmax_out);
/** S2f: the MTP catch-up over `rows` committed verify rows in ONE pass (qwen35moe_prefill_tt.c):
 *  the verify set's hiddens + the embeddings of `next[]`, at base_pos.., then the draft head on
 *  the last row. @return 0, or -1 */
int pn_qwen35_tt_mtp_catchup_rows(pn_qwen35_tt *g, pn_qwen35moe_model *m,
                                  const pn_qwen35moe_config *cfg, pn_qwen35moe_state *st,
                                  const uint32_t *next, int rows, int base_pos,
                                  uint32_t *argmax_out);
/** TTFT lane: the prompt's MTP layer over `rows` rows at base_pos, run OUTSIDE the prefill
 *  (qwen35moe_prefill_tt.c): tall strips in, the layer on the hidden the last pass left, flush,
 *  grid end, decode strips back. Bit-identical to the in-pass layer. @return 0, or -1 */
int pn_qwen35_tt_prefill_mtp_prompt_run(pn_qwen35_tt *g, pn_qwen35moe_model *m,
                                        const pn_qwen35moe_config *cfg, const uint32_t *next,
                                        int rows, int base_pos);
/** TTFT lane (qwen35moe_mtp_prompt_defer_tt.c). wanted: defer the last pass's layer.
 *  owe: the pass records it (pending). arm: the tail forward is done, the layer is owed by
 *  `owner`. settle: run it now if `owner` holds it, else drop it. drop: forget it (the card's
 *  sequence moved on) and end the grid kept for it. idle: the serve's idle tick (ends a kept
 *  grid while it is owed or forgotten in a stream). free: buffers. */
bool pn_qwen35_tt_mtp_prompt_defer_wanted(const pn_qwen35_tt *g);
int pn_qwen35_tt_mtp_prompt_owe(pn_qwen35_tt *g, const uint32_t *next, int rows, int base_pos);
void pn_qwen35_tt_mtp_prompt_arm(pn_qwen35_tt *g, const void *owner);
int pn_qwen35_tt_mtp_prompt_settle(pn_qwen35_tt *g, pn_qwen35moe_model *m,
                                   const pn_qwen35moe_config *cfg, const void *owner);
void pn_qwen35_tt_mtp_prompt_drop(pn_qwen35_tt *g, const char *why);
/** drop, from inside a stream: no card access (the idle tick ends the kept grid). */
void pn_qwen35_tt_mtp_prompt_forget(pn_qwen35_tt *g, const char *why);
void pn_qwen35_tt_mtp_prompt_idle(pn_qwen35_tt *g);
void pn_qwen35_tt_mtp_prompt_free(pn_qwen35_tt *g);
int pn_qwen35_tt_attn_split_q(pn_qwen35_tt *g, const pn_qwen35moe_config *cfg, const char *name,
                              bool extra, pn_qwen35_tt_layer *slot);
/** Q `[h0 | h1 | ...]` and gate halves back into the joint `[h0 Q | h0 gate | h1 Q | ...]`. */
void pn_qwen35_tt_interleave_qg(const float *q, const float *gate, uint32_t heads,
                                uint32_t head_dim, float *qg);
/** The decode head (final norm, LM head, top-k into st->logits) on the decode hidden's row 0,
 *  under the single-row promise — exactly what a single-position forward runs. */
int pn_qwen35_tt_head_row(pn_qwen35_tt *g, pn_qwen35moe_state *st);
/** S2c: rows 0..rows-1 of `hidden` through the head together; each row's greedy token. */
int pn_qwen35_tt_head_rows(pn_qwen35_tt *g, pn_qwen35moe_state *st,
                           const pn_tt_activation *hidden, int rows, uint32_t *argmax_out);

/**
 * THE CROSSED CONTROL: diff the card's attention answer against the host arm's, PER HEAD.
 *
 * Off unless `PN_TT_ATTN_COMPARE` is set, because it costs the readback the block exists to
 * remove. It reports max|delta|, WHICH head and lane, and how many heads are past 1e-3
 * relative — not a PCC, because every failure this path can have is a permutation or a
 * localised wrong element and PCC is blind to the first and saturates on the second.
 */
void pn_qwen35_tt_attn_compare(pn_qwen35_tt *g, const pn_qwen35moe_config *cfg, int layer,
                               const float *host_attn);

/** Free what `pn_qwen35_tt_attn_arm` allocated. Safe on NULL and on an arm that never came
 *  up. The card strips and the KV regions belong to the device and go with it. */
void pn_qwen35_tt_attn_release(pn_qwen35_tt *g);

/** One projection, ISSUED (or issued and awaited when `wait`). Shared with
 *  `qwen35moe_attn_tt.c` so both blocks count the same way — the on-card/on-host split is
 *  the evidence that the card ran the forward, and two counters would be two answers. */
int pn_qwen35_tt_project(pn_qwen35_tt *g, const pn_tt_weight *w, const pn_tt_activation *in,
                         const pn_tt_activation *out, bool wait);

/**
 * S3a: an out-projection (`ssm_out`, `attn_output`, `ffn_down`) whose result the next residual
 * step adds to the stream. With `PN_TT_FUSE_RESADD=1` and a shape the RESADD kernel serves it
 * writes hidden + bf16(in . w) straight into the spare residual strip and sets
 * `residual_fused`, so the residual step only flips the slot; otherwise it is today's
 * pn_qwen35_tt_project(w, in, project, wait) and the residual step adds. Not fused while the
 * attention compare arm reads `project`.
 *
 * @return 0, or -1 (the projection failed; the token is refused, as pn_qwen35_tt_project)
 */
int pn_qwen35_tt_project_out(pn_qwen35_tt *g, const pn_tt_weight *w, const pn_tt_activation *in);

/**
 * S3b: the dense SwiGLU's gate, up and silu(gate) * up as ONE fused projection when
 * `PN_TT_FUSE_SWIGLU=1` and the shape fits (pn_tt_matvec_enqueue_swiglu()), else the
 * unfused gate and up (chained), a flush and pn_tt_silu_mul() — today's sequence, byte for
 * byte. Either way `swiglu` holds silu(in . gate) * (in . up) once the next flush (the
 * down projection's) has run. Counted as two projections either way.
 *
 * @return 0, or -1 (a projection failed; the token is refused, as pn_qwen35_tt_project)
 */
int pn_qwen35_tt_swiglu(pn_qwen35_tt *g, const pn_tt_weight *gate, const pn_tt_weight *up,
                        const pn_tt_activation *in, const pn_tt_activation *gate_out,
                        const pn_tt_activation *up_out, const pn_tt_activation *swiglu);

/**
 * One SSM layer's recurrence, with `conv1d + SiLU` and the delta rule on the card.
 *
 * Argument for argument the same as `pn_qwen35moe_ssm_recurrence()` minus the state, the
 * arena and the thread pool — the state is the card's, and there is no per-head host loop
 * left to spread over threads.
 *
 * @param ssm_index  the SSM-layer ordinal, i.e. `pn_qwen35moe_model_layer_index()`
 * @return           0, or -1 with the reason on stderr
 *
 * Thread-safe: no    Reentrant: no    Signal-safe: no
 */
int pn_qwen35_tt_gdn_recurrence(pn_qwen35_tt *g, const pn_qwen35moe_ssm_layer *l,
                                const pn_qwen35moe_config *cfg, int ssm_index,
                                const float *qkv, const float *z, const float *beta_raw,
                                const float *alpha, float *gated);

/**
 * Arm THE STRIP BRIDGE — the same block with no activation crossing PCIe.
 *
 * Needs the GDN family armed, both `render_*` residencies and a Wt = `ssm_state/32` head-norm
 * slot. Any of them missing leaves `g->gdn_strip` false and the host-mediated recurrence
 * above running; it is never fatal, because a slower correct forward is the right failure.
 *
 * @return 0 whether or not the arm came up; -1 only on a bad argument or an allocation
 */
int pn_qwen35_tt_gdn_strip_arm(pn_qwen35_tt *g, const pn_qwen35moe_config *cfg);

/** Free what `pn_qwen35_tt_gdn_strip_arm` allocated. */
void pn_qwen35_tt_gdn_strip_release(pn_qwen35_tt *g);

/**
 * ONE SSM LAYER, END TO END ON THE CARD.
 *
 * `qkv`, `z` and `gated` are CARD STRIPS, not host arrays — that is the whole difference and
 * the whole of the 49,408 B a layer. `beta_raw` and `alpha` are still host floats, because
 * `beta = sigmoid(.)` and `decay = exp(ssm_a * softplus(.))` need a `pow` and so a `log`, and
 * the SFPU has neither; they are two tiles down and 256 B up.
 *
 * @param ssm_index  the SSM-layer ordinal, i.e. `pn_qwen35moe_model_layer_index()`
 * @return           0, or -1 with the reason on stderr
 *
 * Thread-safe: no    Reentrant: no    Signal-safe: no
 */
int pn_qwen35_tt_gdn_strip_rows(pn_qwen35_tt *g, const pn_qwen35moe_ssm_layer *l,
                                const pn_qwen35moe_config *cfg, int ssm_index, int rows,
                                const pn_tt_activation *qkv, const pn_tt_activation *z,
                                const float *beta_raw, size_t beta_stride,
                                const float *alpha, size_t alpha_stride,
                                const pn_tt_activation *gated);
int pn_qwen35_tt_gdn_strip_layer(pn_qwen35_tt *g, const pn_qwen35moe_ssm_layer *l,
                                 const pn_qwen35moe_config *cfg, int ssm_index,
                                 const pn_tt_activation *qkv, const pn_tt_activation *z,
                                 const float *beta_raw, const float *alpha,
                                 const pn_tt_activation *gated);

/**
 * THE SAME SSM LAYER OVER A CHUNK of `rows` consecutive positions, in rows 0..rows-1 of the
 * strips the projections already wrote.
 *
 * EVERY STEP EXCEPT THE TWO RECURRENCES ALREADY BATCHED AND NONE OF THEM CHANGES HERE. The
 * render is per tile, the two per-head norms are `row_blocks = heads` over a `[1, heads*S]`
 * strip and a row reduce reduces ALONG the row, and the `silu(z) *` gate is per element — so
 * row t of row block h is position t of head h and 32 positions cost one descriptor exactly as
 * one position does. Only PHASE CONV and PHASE DELTA change, and they change to the CHUNKWISE
 * family, which advances the same window and the same recurrent state.
 *
 * `rows` must be at least `pn_tt_gdnc_min_chunk_rows()` — a chunk shorter than the conv window
 * cannot REPLACE it, only shift it, and both the host encoder and the device refuse. The
 * caller routes a shorter tail through `pn_qwen35_tt_gdn_strip_layer()` position by position,
 * which is the same arithmetic on the same bytes.
 *
 * `beta_raw` and `alpha` are `rows` ROWS of host floats at strides `beta_stride`/`alpha_stride`
 * — the shape `pn_tt_activation_download_rows()` produces.
 *
 * @return 0, or -1 with the reason on stderr
 *
 * Thread-safe: no    Reentrant: no    Signal-safe: no
 */
/** PM 7 ITEM 5: the DELTA sub-chunk length the card gate blocks are cut at (the band loop's
 *  own split; 32 when the split is off). */
uint32_t pn_qwen35_tt_gdn_gate_sub_rows(void);

/** PM 7 ITEM 5: the shipped chunk arm (both mixers chunkwise, no isolation split) — the only
 *  one the card gate blocks serve. */
bool pn_qwen35_tt_gdn_card_gate_possible(void);

/** PM 7 ITEM 7 step 1: finish the pipeline's pending band (its RMS, narrow and gate), awaited.
 *  @return 0 or -1. Safe when nothing is pending. */
int pn_qwen35_tt_gdn_pipe_drain(pn_qwen35_tt *g);

/** Prefill lane PIPE 2: finish only the TAIL (narrow + gate) of the band whose RMS is pending,
 *  leaving an in-flight DELTA alone (what an overlap mid-step needs). `pn_qwen35_tt_gdn_pipe_drain`
 *  finishes everything, the in-flight band included. */
int pn_qwen35_tt_gdn_pipe_drain_tail(pn_qwen35_tt *g);

int pn_qwen35_tt_gdn_strip_chunk(pn_qwen35_tt *g, const pn_qwen35moe_ssm_layer *l,
                                 const pn_qwen35moe_config *cfg, int ssm_index,
                                 const pn_tt_activation *qkv, const pn_tt_activation *z,
                                 const float *beta_raw, uint32_t beta_stride,
                                 const float *alpha, uint32_t alpha_stride, int rows,
                                 const pn_tt_activation *gated);

/** Is the chunkwise SSM path usable for a chunk of `rows` positions this run? False when the
 *  `gdnc` family did not arm, when the strip bridge is off, or when `rows` is below the conv
 *  window's floor — every one of which is a reason to run the positions one at a time. */
bool pn_qwen35_tt_gdn_chunk_available(const pn_qwen35_tt *g, int rows);

/* ============================================================
 * THE ON-CARD GATE — implemented in qwen35moe_gdn_parity_tt.c
 * ------------------------------------------------------------
 * A shadow host recurrence run beside the card's, on the same inputs, from the same zero
 * initial condition, never resynced. Two numbers a layer: `conv` isolates PHASE CONV and
 * `gated` prices the whole block. The bar is 1e-3 MAX-NORM RELATIVE and it is the CUDA arm's
 * — the oracle accumulates its dots in fp64 on purpose, so bit-equality was never available.
 * ============================================================ */

/** Opaque; the parity file owns every byte of it. */
struct pn_qwen35_gdn_parity;

/**
 * Arm the shadow recurrence if `PN_TT_GDN_PARITY` names it and the GDN arm actually came up.
 *
 * @return 0 whether or not it armed; -1 only on a bad argument or an allocation failure
 *
 * Thread-safe: no    Reentrant: no    Signal-safe: no
 */
int pn_qwen35_tt_gdn_parity_arm(pn_qwen35_tt *g, const pn_qwen35moe_config *cfg);

/** Free it. Safe on NULL and on a run that never armed. */
void pn_qwen35_tt_gdn_parity_release(pn_qwen35_tt *g);

/**
 * Compare ONE SSM layer, after the card arm has written `g->gdn_conv_out` and `g->h_gated`.
 *
 * Advances the shadow state by exactly one token. A no-op returning 0 when parity is not
 * armed, so the call site needs no guard.
 *
 * @param layer  the SSM-layer ordinal, for the report only
 * @param state  the HOST arm's own per-layer state, which the on-card arm leaves untouched
 * @return       0, or -1 if the shadow recurrence itself failed
 */
int pn_qwen35_tt_gdn_parity_layer(pn_qwen35_tt *g, const pn_qwen35moe_ssm_layer *l,
                                  const pn_qwen35moe_config *cfg, int layer,
                                  const float *qkv, const float *z, const float *beta_raw,
                                  const float *alpha, pn_qwen35moe_ssm_state *state,
                                  pn_arena *arena, struct fw_pool *pool);

/** Close one token: fold its layers into the running worst and print one line. */
void pn_qwen35_tt_gdn_parity_token(pn_qwen35_tt *g, int position);

/** The verdict, once, at teardown. */
void pn_qwen35_tt_gdn_parity_report(const pn_qwen35_tt *g);

/* ================================================================
 * THE PER-HEAD NORM PROBE — implemented in qwen35moe_gdn_normcheck_tt.c
 * ================================================================
 * The two norms of this block that are NOT on the card, run on the card anyway and scored
 * against the oracle's own `pn_qwen35moe_l2_norm()` and `pn_qwen35moe_gated_norm()`. A
 * probe, not a path: nothing downstream reads what it computes.
 */
struct pn_qwen35_gdn_normcheck;

/** Arm the per-head reduce residency and the probe's strips. `PN_TT_GDN_NORMCHECK=1`, and
 *  a card that cannot arm it leaves `g->gdn_normcheck` NULL rather than failing the run. */
int pn_qwen35_tt_gdn_normcheck_arm(pn_qwen35_tt *g, const pn_qwen35moe_config *cfg);

/** Score ONE SSM layer's two norms. `z` and `gated` are the block's own, after the host
 *  has written `gated` — so the comparison is against the value the forward actually used. */
int pn_qwen35_tt_gdn_normcheck_layer(pn_qwen35_tt *g, const pn_qwen35moe_ssm_layer *l,
                                     const pn_qwen35moe_config *cfg, int ssm_index,
                                     const float *z, const float *gated);

/** One line: the worst head of each norm, and how many elements passed 1e-3 relative. */
void pn_qwen35_tt_gdn_normcheck_report(const pn_qwen35_tt *g);

/** Free the probe's strips and constants. Safe on a run that never armed it. */
void pn_qwen35_tt_gdn_normcheck_release(pn_qwen35_tt *g);

/** `PN_TT_ROW_RESET` (LAUNCH lane, default on, `=0` off): sparse host logit rows are reset,
 *  not refilled; the plain forward's tensor is built from the pairs instead of copied and
 *  carries their ids for the sampler (engine.h pn_sampler_row_hint). */
bool pn_qwen35_tt_row_reset_on(void);

#endif /* USE_TT */

#endif /* PN_QWEN35MOE_TT_INTERNAL_H */
