/* SPDX-License-Identifier: AGPL-3.0-or-later */
/* Copyright (C) 2026 David (Corey) Herrera */
/**
 * @file qwen35moe_prefill_overlap_tt.c
 * @brief PHASE W2 (gate W3a): an SSM layer's FFN runs on the prefill GRID, one SUPER-BAND of
 *        rows at a time, while the GDN chain mixes the next super-band.
 *
 * WHY. A tall prefill runs each layer as projections over every row, then the GDN chain band by
 * band, then the out-projection and the FFN over every row — so while the chain runs (2.08 s
 * of a 1,630-token prompt) the 48-core grid idles, and while the FFN runs (1.0 s) the chain's
 * cores idle. The FFN is position-independent: rows [a, b) of it need only rows [a, b) of the
 * mixer's output. So once the chain has finished super-band j (`PN_TT_PREFILL_OVERLAP` bands),
 * that super-band's out-projection, residual, post-norm, FFN and residual can run while the
 * chain moves on.
 *
 * WHAT OVERLAPS. Only the grid's projections (gate, up, down) run DETACHED
 * (pn_tt_matmul_grid_issue_detached): the chain's ops flush the matvec chain on entry, and a
 * detached projection is not on it. The short ops between them — the out-projection (decode
 * group), the residual adds, the post-norm, the SwiGLU (wide eltwise) — run synchronously
 * inside the chain at the moment their input is ready, polled between bands.
 *
 * WHY BIT-EXACT. Every op is today's op over a row sub-range of the same strips: each output
 * row's arithmetic (its K reduction, its norm row, its elementwise tile) never depended on the
 * other rows, and the M split is over rows. The mirrors of `normed` and `ffn_swiglu` written at
 * a super-band base are keyed by `pn_tt_replica_map_set_superband()`.
 *
 * Knobs: `PN_TT_PREFILL_OVERLAP=S` bands a super-band (default 13, 0 off), `PN_TT_PREFILL_OVERLAP_LAYERS=N`
 * only the first N SSM layers (the W3a gate). RED `prefill_overlap_shift`: the FFN of each
 * super-band reads the mixer rows one band late — must DIFFER.
 *
 * @status Gate W3a.
 * @issues Attention layers are not overlapped.
 * @todo None
 */

#ifdef USE_TT

#include "qwen35moe_tt_internal.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "../accel/tt/pn_tt_matmul_group.h"
#include "../accel/tt/pn_tt_gdngate.h"
#include "../accel/tt/pn_tt_reader.h"
#include "../core/bench.h"

enum { OV_IDLE = 0, OV_GATE_UP, OV_DOWN };

static struct {
    pn_qwen35_tt *g;
    const pn_qwen35_tt_layer *d;
    pn_qwen35_tt_overlap_next next; /**< the next SSM layer the job also prepares, or none */
    int prepared_ssm;               /**< that layer's index once a super-band was prepared */
    uint32_t prepared_bands;        /**< bands of it prepared so far */
    bool card_gates;                /**< PM 7: prepare gate blocks on the card, not reads */
    bool prepared_card_gates;       /**< the prepared layer's gates were issued on the card */
    int state;
    int s0;              /**< the hidden slot holding the layer's INPUT */
    uint32_t band, span; /**< the super-band's first band and band count */
    double sync_seconds; /**< time the chain spent on the job's synchronous steps */
    unsigned long long superbands;
} g_ov;

bool pn_qwen35_tt_wavefront_enabled(void)
{
    static int cached = -1;

    if (cached < 0) {
        const char *env = getenv("PN_TT_PREFILL_WAVEFRONT");

        /* DEFAULT ON (run_w_wavefront.sh 2026-10-08: serve TTFT 2.73 -> 2.69 s, CLI wall 2.973
         * -> 2.924, IDENTICAL, reds DIFFER); `PN_TT_PREFILL_WAVEFRONT=0` off. */
        cached = env && *env ? atoi(env) != 0 : 1;
    }
    return cached != 0;
}

uint32_t pn_qwen35_tt_overlap_bands(void)
{
    static int cached = -1;

    if (cached < 0) {
        const char *env = getenv("PN_TT_PREFILL_OVERLAP");

        /* DEFAULT 13 inside the streamed loop (run_w_stream3.sh 2026-10-08: CLI wall 3.22 ->
         * 2.95 s; S 17 serve TTFT 2.98 -> 2.77 s); `PN_TT_PREFILL_OVERLAP=0` off. */
        cached = env && *env ? atoi(env) : 13;
        if (cached < 0)
            cached = 0;
    }
    return (uint32_t)cached;
}

bool pn_qwen35_tt_overlap_layer(pn_qwen35_tt *g, int ssm_index, uint32_t bands)
{
    static int limit = -2;
    const uint32_t sb = pn_qwen35_tt_overlap_bands();

    if (limit == -2) {
        const char *env = getenv("PN_TT_PREFILL_OVERLAP_LAYERS");

        limit = env && *env ? atoi(env) : -1;
    }
    return g && sb && bands > sb && pn_tt_matmul_grid_is_running(g->device) &&
           (limit < 0 || ssm_index < limit);
}

/** Strip `strip` narrowed to bands [band, band + span) — a COPY, so the chain's own strips,
 *  which the band loop keeps seeking, are never touched. */
static int ov_view(const pn_tt_activation *strip, uint32_t band, uint32_t span,
                   pn_tt_activation *out)
{
    *out = *strip;
    return pn_tt_activation_seek(out, band) == 0 && pn_tt_activation_span(out, span) == 0 ? 0
                                                                                           : -1;
}

/** Step 1: out-projection, residual, post-norm (all synchronous), gate + up (detached). */
static int ov_start(void)
{
    pn_qwen35_tt *g = g_ov.g;
    pn_qwen35_tt_strips *s = &g->card;
    const bool red = getenv("PN_TT_RED") && strcmp(getenv("PN_TT_RED"), "prefill_overlap_shift") == 0;
    const uint32_t in_band = red && g_ov.band + g_ov.span < s->ssm_gated.tile_rows
                                 ? g_ov.band + 1u : g_ov.band;
    pn_tt_activation gated, project, h_in, h_mid, normed, gate, up;

    if (ov_view(&s->ssm_gated, in_band, g_ov.span, &gated) != 0 ||
        ov_view(&s->project, g_ov.band, g_ov.span, &project) != 0 ||
        ov_view(&s->hidden[g_ov.s0], g_ov.band, g_ov.span, &h_in) != 0 ||
        ov_view(&s->hidden[g_ov.s0 ^ 1], g_ov.band, g_ov.span, &h_mid) != 0 ||
        ov_view(&s->normed, g_ov.band, g_ov.span, &normed) != 0 ||
        ov_view(&s->ffn_gate, g_ov.band, g_ov.span, &gate) != 0 ||
        ov_view(&s->ffn_up, g_ov.band, g_ov.span, &up) != 0)
        return -1;
    if (pn_qwen35_tt_project(g, &g_ov.d->ssm_out, &gated, &project, true) != 0 ||
        pn_tt_add(g->device, &h_in, &project, &h_mid) != 0 ||
        pn_tt_rms_norm(g->device, &g_ov.d->post_norm, &h_mid, &normed) != 0 ||
        pn_tt_matmul_grid_issue_detached(g->device, &g_ov.d->ffn_gate, &normed, &gate) != 0 ||
        pn_tt_matmul_grid_issue_detached(g->device, &g_ov.d->ffn_up, &normed, &up) != 0)
        return -1;
    g_ov.state = OV_GATE_UP;
    return 0;
}

/** Step 2: SwiGLU (synchronous) and down (detached). */
static int ov_swiglu_down(void)
{
    pn_qwen35_tt *g = g_ov.g;
    pn_qwen35_tt_strips *s = &g->card;
    pn_tt_activation gate, up, swiglu, project;

    if (ov_view(&s->ffn_gate, g_ov.band, g_ov.span, &gate) != 0 ||
        ov_view(&s->ffn_up, g_ov.band, g_ov.span, &up) != 0 ||
        ov_view(&s->ffn_swiglu, g_ov.band, g_ov.span, &swiglu) != 0 ||
        ov_view(&s->project, g_ov.band, g_ov.span, &project) != 0)
        return -1;
    if (pn_tt_silu_mul(g->device, &gate, &up, &swiglu) != 0 ||
        pn_tt_matmul_grid_issue_detached(g->device, &g_ov.d->ffn_down, &swiglu, &project) != 0)
        return -1;
    g_ov.state = OV_DOWN;
    return 0;
}

/**
 * Step 4 (the WAVEFRONT): this super-band of the NEXT SSM layer — its attn-norm, its q/k/v and z
 * projections (grid, detached), its alpha/beta (decode group) and their reads, gated on the card
 * having run them. So the reader starts the next layer while this one still mixes. Every op is
 * the one the next layer's start would have run, over these rows.
 */
static int ov_prepare_next(void)
{
    pn_qwen35_tt *g = g_ov.g;
    pn_qwen35_tt_strips *s = &g->card;
    const pn_qwen35_tt_layer *n = g_ov.next.d;
    pn_tt_activation h, normed, qkv, z, alpha, beta;
    pn_tt_reader_gate gate;

    if (ov_view(&s->hidden[g_ov.s0], g_ov.band, g_ov.span, &h) != 0 ||
        ov_view(&s->normed, g_ov.band, g_ov.span, &normed) != 0 ||
        ov_view(&s->ssm_qkv, g_ov.band, g_ov.span, &qkv) != 0 ||
        ov_view(&s->ssm_z, g_ov.band, g_ov.span, &z) != 0 ||
        ov_view(&s->ssm_alpha, g_ov.band, g_ov.span, &alpha) != 0 ||
        ov_view(&s->ssm_beta, g_ov.band, g_ov.span, &beta) != 0)
        return -1;
    if (pn_tt_rms_norm(g->device, &n->attn_norm, &h, &normed) != 0 ||
        pn_tt_matmul_grid_issue_detached(g->device, &n->attn_qkv, &normed, &qkv) != 0 ||
        pn_tt_matmul_grid_issue_detached(g->device, &n->attn_gate, &normed, &z) != 0 ||
        pn_qwen35_tt_project(g, &n->ssm_alpha, &normed, &alpha, false) != 0 ||
        pn_qwen35_tt_project(g, &n->ssm_beta, &normed, &beta, false) != 0 ||
        pn_tt_matvec_flush(g->device) != 0)
        return -1;
    /* PM 7 ITEM 5: with the gates on the card these rows' gate blocks are computed there,
     * after the flush above (the projections that wrote alpha/beta are done) — no reads. */
    if (g_ov.card_gates) {
        if (pn_tt_gdngate_issue_bands(g->device, (uint32_t)g_ov.next.ssm_index, &s->ssm_alpha,
                                      &s->ssm_beta, (uint32_t)g_ov.next.rows,
                                      pn_qwen35_tt_gdn_gate_sub_rows(), g_ov.band,
                                      g_ov.span) != 0)
            return -1;
    } else if (!pn_tt_matmul_group_member0_completion(g->device, &gate.x, &gate.y,
                                                      &gate.control_address, &gate.target) ||
               !pn_qwen35_tt_ab_reader_post_next(g, g_ov.next.ssm_index, (int)g_ov.band,
                                                 (int)g_ov.span, g_ov.next.rows,
                                                 g_ov.next.alpha, g_ov.next.alpha_width,
                                                 g_ov.next.beta, g_ov.next.beta_width, &gate)) {
        return -1;
    }
    if (g_ov.band == 0u) {
        g_ov.prepared_ssm = g_ov.next.ssm_index;
        g_ov.prepared_bands = 0u;
        g_ov.prepared_card_gates = g_ov.card_gates;
    }
    if (g_ov.prepared_ssm == g_ov.next.ssm_index && g_ov.prepared_bands == g_ov.band)
        g_ov.prepared_bands += g_ov.span;
    return 0;
}

/** Step 3: the FFN residual (synchronous): hidden[s0] = hidden[s0^1] + project; then step 4. */
static int ov_finish(void)
{
    pn_qwen35_tt *g = g_ov.g;
    pn_qwen35_tt_strips *s = &g->card;
    pn_tt_activation h_mid, project, h_out;

    if (ov_view(&s->hidden[g_ov.s0 ^ 1], g_ov.band, g_ov.span, &h_mid) != 0 ||
        ov_view(&s->project, g_ov.band, g_ov.span, &project) != 0 ||
        ov_view(&s->hidden[g_ov.s0], g_ov.band, g_ov.span, &h_out) != 0 ||
        pn_tt_add(g->device, &h_mid, &project, &h_out) != 0)
        return -1;
    g_ov.state = OV_IDLE;
    if (g_ov.next.d && ov_prepare_next() != 0) {
        g_ov.prepared_ssm = -1; /* the next layer runs its own start */
        return -1;
    }
    return 0;
}

bool pn_qwen35_tt_overlap_prepared(int ssm_index, uint32_t bands)
{
    return g_ov.prepared_ssm == ssm_index && g_ov.prepared_bands >= bands;
}

bool pn_qwen35_tt_overlap_gates_prepared(int ssm_index, uint32_t bands)
{
    return pn_qwen35_tt_overlap_prepared(ssm_index, bands) && g_ov.prepared_card_gates;
}

void pn_qwen35_tt_overlap_set_card_gates(bool on)
{
    g_ov.card_gates = on;
}

void pn_qwen35_tt_overlap_forget(void)
{
    g_ov.prepared_ssm = -1;
    g_ov.prepared_bands = 0u;
}

/** Run the job's next step if its detached projections are done; `block` waits for them. */
static int ov_step(bool block)
{
    const double at = pn_time_ms();
    int rc = 0, done;

    if (g_ov.state == OV_IDLE)
        return 0;
    done = block ? (pn_tt_matmul_grid_detached_wait(g_ov.g->device) == FT_TT_OK ? 1 : -1)
                 : pn_tt_matmul_grid_detached_poll(g_ov.g->device);
    if (done < 0)
        rc = -1;
    else if (done == 1)
        rc = g_ov.state == OV_GATE_UP ? ov_swiglu_down() : ov_finish();
    g_ov.sync_seconds += (pn_time_ms() - at) / 1000.0;
    return rc;
}

int pn_qwen35_tt_overlap_advance(pn_qwen35_tt *g)
{
    (void)g;
    return ov_step(false);
}

int pn_qwen35_tt_overlap_drain(pn_qwen35_tt *g)
{
    (void)g;
    while (g_ov.state != OV_IDLE)
        if (ov_step(true) != 0) {
            g_ov.state = OV_IDLE;
            return -1;
        }
    return 0;
}

int pn_qwen35_tt_overlap_superband(pn_qwen35_tt *g, const pn_qwen35_tt_layer *d, uint32_t band,
                                   uint32_t span, const pn_qwen35_tt_overlap_next *next)
{
    double at;
    int rc;

    if (pn_qwen35_tt_overlap_drain(g) != 0)
        return -1;
    at = pn_time_ms();
    g_ov.g = g;
    g_ov.d = d;
    g_ov.s0 = g->card.slot;
    g_ov.band = band;
    g_ov.span = span;
    if (next && next->d && pn_qwen35_tt_wavefront_enabled())
        g_ov.next = *next;
    else
        memset(&g_ov.next, 0, sizeof(g_ov.next));
    g_ov.superbands++;
    rc = ov_start();
    g_ov.sync_seconds += (pn_time_ms() - at) / 1000.0;
    if (rc != 0)
        g_ov.state = OV_IDLE;
    return rc;
}

double pn_qwen35_tt_overlap_take_seconds(void)
{
    const double seconds = g_ov.sync_seconds;

    g_ov.sync_seconds = 0.0;
    return seconds;
}

void pn_qwen35_tt_overlap_report(void)
{
    if (g_ov.superbands)
        fprintf(stderr, "qwen35_tt: prefill OVERLAP — %llu super-bands of %u bands, FFN on the "
                        "grid under the GDN chain\n",
                g_ov.superbands, pn_qwen35_tt_overlap_bands());
}

#endif /* USE_TT */
