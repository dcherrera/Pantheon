/* SPDX-License-Identifier: AGPL-3.0-or-later */
/* Copyright (C) 2026 David (Corey) Herrera */
/**
 * @file llama_attention_trace.h
 * @brief Capture the GQA softmax's SHAPE — per head, per layer — out of the llama forward,
 *        together with the magnitude that shape is supposed to explain.
 *
 *        The ChatML defect was traced to `attn_output`'s INPUT being 1.2-3.8x larger under a
 *        chat framing at unchanged PCC, while `attn_v`'s input is identical between framings.
 *        The stated mechanism — less cancellation in the weighted average of V, i.e. more
 *        CONCENTRATED attention weights — was inferred from those magnitudes and never
 *        measured. This records both halves at once so the inference can be tested rather
 *        than repeated:
 *
 *          - the SHAPE of each head's weight vector (entropy, collision, max weight),
 *          - WHERE its mass sits (weight on the position-0 sink, on the current token, and
 *            the relative position of the peak), and
 *          - that head's contribution to `attn_output`'s input RMS, plus the
 *            no-cancellation reference the weights average over.
 *
 *        Shape and location are independent, and only the pair identifies the cause. A head
 *        that parks 60% of its mass on a near-zero-magnitude prefix sink and a head that puts
 *        60% on a full-magnitude content token have the SAME entropy and the same collision,
 *        and outputs that differ by a factor of several — so entropy alone cannot answer the
 *        question this exists to answer, and recording it alone would look like a null result.
 *
 *        `cancellation` is the quantity that closes (or refutes) the link. For a head whose
 *        output is o = sum_s p_s v_s, define
 *
 *            vbar_rms = sqrt( sum_s p_s * ||v_s||^2 / head_dim )
 *
 *        — the per-component RMS of the V rows the head averaged, weighted by the attention
 *        itself, so a head that attends to unusually large V rows is not mistaken for a head
 *        that cancels less. Then `out_rms / vbar_rms` is exactly how much of that magnitude
 *        SURVIVED the weighted average. If the V rows were mutually uncorrelated, that ratio
 *        would be sqrt(sum_s p_s^2) — the square root of the collision probability — so
 *        `collision` gives the concentration-only PREDICTION of the same number, and the two
 *        can be compared directly. Concentration explaining the magnitude means those two
 *        move together; a measured cancellation that does not track sqrt(collision) means the
 *        magnitude comes from WHICH vectors are averaged, not from how many.
 *
 * The recorded weights are the post-softmax ones the forward actually used, read out of the
 * same scores buffer `compute_attn_heads` left them in — nothing is recomputed, so this
 * cannot drift from the arithmetic it claims to describe.
 *
 * ASSUMES A CONTIGUOUS CACHE (no sliding-window wraparound): positions 0..seq_len-1. The
 * only caller is a debug harness stepping a fresh sequence, which is exactly that case.
 *
 * @status Working
 * @issues None
 * @todo None
 */
#ifndef PN_LLAMA_ATTENTION_TRACE_H
#define PN_LLAMA_ATTENTION_TRACE_H

/** One query head's attention at one position of one layer. */
typedef struct {
    double entropy;    /**< -sum p ln p, in nats. ln(seq_len) is the uniform maximum */
    double collision;  /**< sum p^2; its reciprocal is the effective number of positions */
    double max_weight; /**< the largest single weight */
    double sink_weight;/**< the weight on position 0 — the attention sink every framing has,
                            and the first token a chat template differs in */
    double out_rms;    /**< RMS over head_dim of this head's slice of `attn_out` */
    double vbar_rms;   /**< sqrt(sum_s p_s ||v_s||^2 / head_dim) — no-cancellation reference */
    double v0_rms;     /**< RMS of the position-0 V row this head averages over. A sink whose
                            V row is small drags vbar_rms down in proportion to sink_weight */
    double top_pos_rel;/**< argmax position / (seq_len-1) — WHERE the peak sits. Near 0 is a
                            prefix sink, near 1 is recency. Entropy cannot tell these apart */
    double self_weight;/**< the weight on the current token, the other slot a head can park on */
} pn_attn_head_record;

/** One forward's attention shape: n_layer * n_head head records plus per-layer scalars. */
typedef struct {
    pn_attn_head_record *heads;      /**< n_layer * n_head, row-major by layer, owned */
    double              *normed_rms; /**< n_layer — RMS of `attn_v`'s input, owned */
    int                  n_layer;
    int                  n_head;
    int                  seq_len;    /**< keys attended at the position just captured */
    int                  recorded;   /**< layers the forward now running has written */
} pn_attention_trace;

/** Everything one layer's capture reads. Pointers are the forward's own live buffers. */
typedef struct {
    const float *scores;   /**< [n_head, max_seq] post-softmax weights */
    const float *attn_out; /**< [n_head, head_dim] the weighted V sums */
    const float *v_cache;  /**< [n_kv_head, max_seq, head_dim] */
    const float *normed;   /**< [n_embd] the vector the Q/K/V projections read */
    int n_head;
    int n_kv_groups;
    int head_dim;
    int max_seq;
    int seq_len;
    int n_embd;
} pn_attention_trace_layer;

/**
 * Allocate a capture buffer.
 *
 * @return 0 on success, -1 on allocation failure
 *
 * Thread-safe: yes   Reentrant: yes   Signal-safe: no
 */
int pn_attention_trace_alloc(pn_attention_trace *trace, int n_layer, int n_head);

void pn_attention_trace_free(pn_attention_trace *trace);

/**
 * Direct the next forward's per-layer attention records into `trace`, or NULL to stop.
 *
 * One process-wide slot, for the same reason the residual trace has one: the forward
 * signature is the engine's public contract and a debug instrument does not get to change
 * it. The only caller is a harness running one forward at a time.
 *
 * Thread-safe: no   Reentrant: no   Signal-safe: no
 */
void pn_attention_trace_capture_into(pn_attention_trace *trace);

/** @return whether a capture is armed, so the forward can skip work it only does to record. */
int pn_attention_trace_armed(void);

/**
 * Record one layer's attention shape. Called by a forward implementation immediately after
 * the heads are computed and BEFORE the output projection consumes `attn_out`.
 */
void pn_attention_trace_record(int layer, const pn_attention_trace_layer *in);

/** @return the record for (layer, head) of a completed capture, or NULL if never written. */
const pn_attn_head_record *pn_attention_trace_head(const pn_attention_trace *trace, int layer,
                                                   int head);

#endif /* PN_LLAMA_ATTENTION_TRACE_H */
