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
 * WHAT OVERLAPS. The grid's projections (out, gate, up, down, and the wavefront's q/k/v/z) run
 * DETACHED (pn_tt_matmul_grid_issue_detached): the chain's ops flush the matvec chain on entry,
 * and a detached projection is not on it. The short ops between them — the residual adds, the
 * post-norm, the SwiGLU — run synchronously inside the chain.
 *
 * WHERE (O.1, qwen35moe_prefill_ov_plan.h). Streamed, the band loop is ONE in-order puller
 * program, and every wait for the grid stalls the chain bands queued behind it. So each
 * super-band job's steps land at band offsets after its last band (`PN_TT_PREFILL_OV_AT=a,b,c`,
 * default 2,5,10 at 13 bands under PIPE 2) — each wait for the grid comes bands after the step
 * it waits on — and the layer's last job runs at the loop's end INSIDE the stream.
 *
 * G.2. Each wait names its step's grid MARK (pn_tt_grid_marks.h), not every detached projection,
 * and the next job's out-proj is queued before this job's FINISH waits for its down, so the grid
 * holds work through the synchronous steps. Under table gates the next layer's alpha/beta go to
 * the grid detached too. `PN_TT_GRID_MARKS=0` waits for everything (O.1).
 *
 * WHY BIT-EXACT. Every op is today's op over a row sub-range of the same strips: each output
 * row's arithmetic (its K reduction, its norm row, its elementwise tile) never depended on the
 * other rows, and the M split is over rows. The mirrors of `normed` and `ffn_swiglu` written at
 * a super-band base are keyed by `pn_tt_replica_map_set_superband()`.
 *
 * Knobs: `PN_TT_PREFILL_OVERLAP=S` bands a super-band (default 13, 0 off), `PN_TT_PREFILL_OVERLAP_LAYERS=N`
 * only the first N SSM layers (the W3a gate), `PN_TT_PREFILL_OV_AT`. RED `prefill_overlap_shift`:
 * the FFN of each super-band reads the mixer rows one band late — must DIFFER. RED
 * `prefill_ov_out_early`: OUT without the pipe's depth (reads `gated` before its last gates
 * ran) — must DIFFER.
 *
 * @status Gate W3a; O.1 placement.
 * @issues Attention layers are not overlapped.
 * @todo None
 */

#ifdef USE_TT

#include "qwen35moe_tt_internal.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "../accel/tt/pn_tt_eltwise_group.h"
#include "../accel/tt/pn_tt_matmul_group.h"
#include "../accel/tt/pn_tt_gdnc.h"
#include "../accel/tt/pn_tt_gdngate.h"
#include "../accel/tt/pn_tt_reader.h"
#include "../core/bench.h"
#include "qwen35moe_prefill_ov_plan.h"

static struct {
    pn_qwen35_tt *g;
    const pn_qwen35_tt_layer *d;
    pn_qwen35_tt_overlap_next next; /**< the next SSM layer the job also prepares, or none */
    int prepared_ssm;               /**< that layer's index once a super-band was prepared */
    uint32_t prepared_bands;        /**< bands of it prepared so far */
    bool card_gates;                /**< PM 7: prepare gate blocks on the card, not reads */
    bool prepared_card_gates;       /**< the prepared layer's gates went to the card */
    bool active;                    /**< a layer's plan is running */
    uint32_t ticked;                /**< the next band whose steps have not run */
    pn_ov_plan plan;                /**< O.1: where each job's steps land */
    const pn_tt_activation *mixer;  /**< the mixer's output strip (SSM `gated`, attention `gated`) */
    const pn_tt_weight *out_weight; /**< its out-projection */
    int s0;              /**< the hidden slot holding the layer's INPUT */
    uint32_t job;        /**< the job being stepped */
    uint32_t band, span; /**< its first band and band count */
    uint32_t mark[PN_OV_PLAN_MAX_JOBS][3]; /**< G.2: each job's OUT / UP / DOWN grid marks */
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

uint32_t pn_qwen35_tt_overlap_span(uint32_t bands)
{
    const char *env = getenv("PN_TT_PREFILL_OVERLAP");
    const uint32_t half = (bands + 1u) / 2u;

    if (pn_qwen35_tt_overlap_bands() == 0u)
        return 0u;
    if (env && *env)
        return pn_qwen35_tt_overlap_bands();
    /* G.3: unset, half the pass (two jobs), at least 13 — span 32 at 64 bands measured an SSM
     * layer-pass of 87.7 ms against 91.3 at 13 (bigger grid jobs), bit-identical. */
    return half > 13u ? half : 13u;
}

bool pn_qwen35_tt_overlap_attention(pn_qwen35_tt *g, uint32_t bands)
{
    static int cached = -1;

    if (cached < 0) {
        const char *env = getenv("PN_TT_PREFILL_OV_ATTN");

        /* A: an attention layer's out-proj and FFN under its own band loop; `=0` off. */
        cached = env && *env ? atoi(env) != 0 : 1;
    }
    return cached && g && pn_qwen35_tt_overlap_span(bands) &&
           bands > pn_qwen35_tt_overlap_span(bands) && pn_tt_matmul_grid_is_running(g->device);
}

bool pn_qwen35_tt_overlap_layer(pn_qwen35_tt *g, int ssm_index, uint32_t bands)
{
    static int limit = -2;
    const uint32_t sb = pn_qwen35_tt_overlap_span(bands);

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

/** Wait for every detached grid projection (the layer's tail). */
static int ov_grid_wait(void)
{
    return pn_tt_matmul_grid_detached_wait(g_ov.g->device) == FT_TT_OK ? 0 : -1;
}

/** G.2: wait for the job's step `stage` (OUT 0, UP 1, DOWN 2) — that grid projection and
 *  what was issued before it, not what was issued after (pn_tt_grid_marks.h). */
static int ov_wait_step(uint32_t stage)
{
    return pn_tt_matmul_grid_wait_mark(g_ov.g->device, g_ov.mark[g_ov.job][stage]) == FT_TT_OK
               ? 0 : -1;
}

/** G.2: the grid mark of the projection just issued, as the job's step `stage`. */
static void ov_mark_step(uint32_t stage)
{
    g_ov.mark[g_ov.job][stage] = pn_tt_matmul_grid_last_mark(g_ov.g->device);
}

/** OUT: the job's `gated` rows settled (its last gates ran `pipe_depth` bands ago), then its
 *  out-projection — detached on the grid when `gated` is mirrored, else synchronous. */
static int ov_out(void)
{
    pn_qwen35_tt *g = g_ov.g;
    pn_qwen35_tt_strips *s = &g->card;
    const bool red = getenv("PN_TT_RED") && strcmp(getenv("PN_TT_RED"), "prefill_overlap_shift") == 0;
    const uint32_t in_band = red && g_ov.band + g_ov.span < g_ov.mixer->tile_rows
                                 ? g_ov.band + 1u : g_ov.band;
    pn_tt_activation gated, project;

    if (ov_view(g_ov.mixer, in_band, g_ov.span, &gated) != 0 ||
        ov_view(&s->project, g_ov.band, g_ov.span, &project) != 0 ||
        pn_tt_eltwise_group_settle(g->device) != 0)
        return -1;
    g_ov.mark[g_ov.job][0] = PN_TT_GRID_MARK_NONE; /* synchronous: nothing to wait for */
    if (gated.replica_count > 1u) {
        if (pn_tt_matmul_grid_issue_detached(g->device, g_ov.out_weight, &gated, &project) !=
            FT_TT_OK)
            return -1;
        ov_mark_step(0u);
        return 0;
    }
    return pn_qwen35_tt_project(g, g_ov.out_weight, &gated, &project, true);
}

/** UP: residual, post-norm (synchronous), gate + up (detached). */
static int ov_up(void)
{
    pn_qwen35_tt *g = g_ov.g;
    pn_qwen35_tt_strips *s = &g->card;
    pn_tt_activation project, h_in, h_mid, normed, gate, up;

    if (ov_view(&s->project, g_ov.band, g_ov.span, &project) != 0 ||
        ov_view(&s->hidden[g_ov.s0], g_ov.band, g_ov.span, &h_in) != 0 ||
        ov_view(&s->hidden[g_ov.s0 ^ 1], g_ov.band, g_ov.span, &h_mid) != 0 ||
        ov_view(&s->normed, g_ov.band, g_ov.span, &normed) != 0 ||
        ov_view(&s->ffn_gate, g_ov.band, g_ov.span, &gate) != 0 ||
        ov_view(&s->ffn_up, g_ov.band, g_ov.span, &up) != 0)
        return -1;
    if (ov_wait_step(0u) != 0 || pn_tt_add(g->device, &h_in, &project, &h_mid) != 0 ||
        pn_tt_rms_norm(g->device, &g_ov.d->post_norm, &h_mid, &normed) != 0 ||
        pn_tt_matmul_grid_issue_detached(g->device, &g_ov.d->ffn_gate, &normed, &gate) != 0 ||
        pn_tt_matmul_grid_issue_detached(g->device, &g_ov.d->ffn_up, &normed, &up) != 0)
        return -1;
    ov_mark_step(1u);
    g_ov.superbands++;
    return 0;
}

/** DOWN: SwiGLU (synchronous), down (detached). */
static int ov_down(void)
{
    pn_qwen35_tt *g = g_ov.g;
    pn_qwen35_tt_strips *s = &g->card;
    pn_tt_activation gate, up, swiglu, project;

    if (ov_view(&s->ffn_gate, g_ov.band, g_ov.span, &gate) != 0 ||
        ov_view(&s->ffn_up, g_ov.band, g_ov.span, &up) != 0 ||
        ov_view(&s->ffn_swiglu, g_ov.band, g_ov.span, &swiglu) != 0 ||
        ov_view(&s->project, g_ov.band, g_ov.span, &project) != 0)
        return -1;
    if (ov_wait_step(1u) != 0 || pn_tt_silu_mul(g->device, &gate, &up, &swiglu) != 0 ||
        pn_tt_matmul_grid_issue_detached(g->device, &g_ov.d->ffn_down, &swiglu, &project) != 0)
        return -1;
    ov_mark_step(2u);
    return 0;
}

/** The layer runs the exact table gates: nothing but the next layer's DELTA reads alpha/beta. */
static bool ov_table_gates(void)
{
    return g_ov.card_gates && pn_tt_gdnc_table_ready(g_ov.g->device);
}

/** The next layer's alpha/beta. Under the exact table gates (G.2) DETACHED on the grid by rows:
 *  nothing reads them before the next layer, and on the matvec chain they would force a full
 *  grid drain every few jobs (the chain's 6-projection bound). Else on the chain, as before. */
static int ov_prepare_alpha_beta(const pn_qwen35_tt_layer *n, const pn_tt_activation *normed,
                                 const pn_tt_activation *alpha, const pn_tt_activation *beta)
{
    pn_qwen35_tt *g = g_ov.g;

    if (ov_table_gates() && normed->replica_count > 1u &&
        pn_tt_matmul_grid_issue_detached(g->device, &n->ssm_alpha, normed, alpha) == FT_TT_OK)
        return pn_tt_matmul_grid_issue_detached(g->device, &n->ssm_beta, normed, beta) == FT_TT_OK
                   ? 0 : -1;
    return (pn_qwen35_tt_project(g, &n->ssm_alpha, normed, alpha, false) != 0 ||
            pn_qwen35_tt_project(g, &n->ssm_beta, normed, beta, false) != 0) ? -1 : 0;
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
    int rc;

    if (ov_view(&s->hidden[g_ov.s0], g_ov.band, g_ov.span, &h) != 0 ||
        ov_view(&s->normed, g_ov.band, g_ov.span, &normed) != 0 ||
        ov_view(&s->ssm_qkv, g_ov.band, g_ov.span, &qkv) != 0 ||
        ov_view(&s->ssm_z, g_ov.band, g_ov.span, &z) != 0 ||
        ov_view(&s->ssm_alpha, g_ov.band, g_ov.span, &alpha) != 0 ||
        ov_view(&s->ssm_beta, g_ov.band, g_ov.span, &beta) != 0)
        return -1;
    /* O.0: alpha/beta by rows on the grid only under card gates — the host-gate reader below
     * is gated on member 0's count, which a grid piece never moves. O.1: under the exact
     * table gates nothing reads these rows before the next layer, so no flush here (the
     * puller would stand on q/k/v/z); the layer's tail flushes once. */
    pn_tt_matmul_grid_unsplit_allow(g->device, g_ov.card_gates);
    rc = (pn_tt_rms_norm(g->device, &n->attn_norm, &h, &normed) != 0 ||
          pn_tt_matmul_grid_issue_detached(g->device, &n->attn_qkv, &normed, &qkv) != 0 ||
          pn_tt_matmul_grid_issue_detached(g->device, &n->attn_gate, &normed, &z) != 0 ||
          ov_prepare_alpha_beta(n, &normed, &alpha, &beta) != 0 ||
          (!ov_table_gates() && pn_tt_matvec_flush(g->device) != 0)) ? -1 : 0;
    pn_tt_matmul_grid_unsplit_allow(g->device, false);
    if (rc != 0)
        return -1;
    /* PM 7 ITEM 5: with the gates on the card these rows' gate blocks are computed there,
     * after the flush above (the projections that wrote alpha/beta are done) — no reads. */
    if (g_ov.card_gates) {
        /* The exact card gates issue nothing: each DELTA looks its gates up from the strips. */
        if (!pn_tt_gdnc_table_ready(g->device) &&
            pn_tt_gdngate_issue_bands(g->device, (uint32_t)g_ov.next.ssm_index, &s->ssm_alpha,
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

/** FINISH: the FFN residual (synchronous), hidden[s0] = hidden[s0^1] + project; then the
 *  wavefront's next-layer prepare. */
static int ov_finish(void)
{
    pn_qwen35_tt *g = g_ov.g;
    pn_qwen35_tt_strips *s = &g->card;
    pn_tt_activation h_mid, project, h_out;

    if (ov_view(&s->hidden[g_ov.s0 ^ 1], g_ov.band, g_ov.span, &h_mid) != 0 ||
        ov_view(&s->project, g_ov.band, g_ov.span, &project) != 0 ||
        ov_view(&s->hidden[g_ov.s0], g_ov.band, g_ov.span, &h_out) != 0 ||
        ov_wait_step(2u) != 0 || pn_tt_add(g->device, &h_mid, &project, &h_out) != 0)
        return -1;
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

int pn_qwen35_tt_overlap_layer_begin(pn_qwen35_tt *g, const pn_qwen35_tt_layer *d,
                                     uint32_t bands, uint32_t pipe_depth,
                                     const pn_qwen35_tt_overlap_next *next,
                                     const pn_tt_activation *mixer, const pn_tt_weight *out_weight)
{
    const uint32_t span = pn_qwen35_tt_overlap_span(bands);
    uint32_t at[3];
    const uint32_t *use = NULL;
    const char *env = getenv("PN_TT_PREFILL_OV_AT");

    if (env && *env) {
        if (pn_ov_plan_parse_at(env, at) != 0) {
            fprintf(stderr, "qwen35_tt: PN_TT_PREFILL_OV_AT=%s is not three increasing band "
                            "offsets a,b,c — the defaults run\n", env);
        } else {
            use = at;
        }
    }
    /* RED `prefill_ov_out_early`: no pipe-depth floor — OUT reads `gated` before the last
     * band's gate ran. */
    if (getenv("PN_TT_RED") && strcmp(getenv("PN_TT_RED"), "prefill_ov_out_early") == 0) {
        static const uint32_t early[3] = { 0u, 3u, 8u };

        pipe_depth = 0u;
        use = early;
    }
    if (pn_ov_plan_build(&g_ov.plan, bands, span, pipe_depth, use) != 0)
        return -1;
    g_ov.g = g;
    g_ov.d = d;
    g_ov.mixer = mixer;
    g_ov.out_weight = out_weight;
    g_ov.s0 = g->card.slot;
    if (next && next->d && pn_qwen35_tt_wavefront_enabled())
        g_ov.next = *next;
    else
        memset(&g_ov.next, 0, sizeof(g_ov.next));
    for (uint32_t k = 0; k < PN_OV_PLAN_MAX_JOBS; k++)
        for (uint32_t i = 0; i < 3u; i++)
            g_ov.mark[k][i] = PN_TT_GRID_MARK_NONE;
    g_ov.ticked = 0u;
    g_ov.active = true;
    return 0;
}

/** One planned action on its job's rows. */
static int ov_run(const pn_ov_action *action)
{
    g_ov.job = action->job;
    pn_ov_plan_job_bands(&g_ov.plan, action->job, &g_ov.band, &g_ov.span);
    switch (action->stage) {
    case PN_OV_STAGE_OUT:
        return ov_out();
    case PN_OV_STAGE_UP:
        return ov_up();
    case PN_OV_STAGE_DOWN:
        return ov_down();
    case PN_OV_STAGE_FINISH:
        return ov_finish();
    default:
        return -1;
    }
}

int pn_qwen35_tt_overlap_tick(pn_qwen35_tt *g, uint32_t band)
{
    pn_ov_action due[PN_OV_PLAN_MAX_JOBS * PN_OV_STAGE_COUNT];
    const double at = pn_time_ms();
    size_t n, i;
    int rc = 0;

    (void)g;
    if (!g_ov.active)
        return 0;
    /* Every band since the last tick, so a loop that ended early still runs the whole plan. */
    for (; g_ov.ticked <= band && g_ov.ticked <= g_ov.plan.bands && rc == 0; g_ov.ticked++) {
        n = pn_ov_plan_due(&g_ov.plan, g_ov.ticked, due, sizeof(due) / sizeof(due[0]));
        for (i = 0; i < n && rc == 0; i++)
            rc = ov_run(&due[i]);
    }
    /* The tail: every projection the wavefront left unflushed is done before the next layer. */
    if (rc == 0 && band >= g_ov.plan.bands &&
        (pn_tt_matvec_flush(g_ov.g->device) != 0 || ov_grid_wait() != 0))
        rc = -1;
    if (band >= g_ov.plan.bands || rc != 0)
        g_ov.active = false;
    g_ov.sync_seconds += (pn_time_ms() - at) / 1000.0;
    return rc;
}

void pn_qwen35_tt_overlap_abort(pn_qwen35_tt *g)
{
    if (g_ov.active && g)
        (void)pn_tt_matmul_grid_detached_wait(g->device);
    g_ov.active = false;
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
        fprintf(stderr, "qwen35_tt: prefill OVERLAP — %llu super-bands (span %u at 64 bands), "
                        "FFN on the grid under the GDN chain\n",
                g_ov.superbands, pn_qwen35_tt_overlap_span(64u));
}

#endif /* USE_TT */
