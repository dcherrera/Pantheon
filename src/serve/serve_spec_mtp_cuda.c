/* SPDX-License-Identifier: AGPL-3.0-or-later */
/* Copyright (C) 2026 David (Corey) Herrera */
/**
 * @file serve_spec_mtp_cuda.c
 * @brief Implementation of the serve's CUDA MTP cycle (serve_spec_mtp_cuda.h).
 *
 * @status complete
 * @issues None
 * @todo None
 */
#include "serve_spec_mtp_cuda.h"

#ifdef USE_CUDA

#include "../core/spec_mtp_step.h"
#include "../models/bonsai.h"
#include "../models/qwen35moe.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

double sg_mtp_cuda_now_ms(void)
{
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (double)ts.tv_sec * 1e3 + (double)ts.tv_nsec * 1e-6;
}

static int mtp_cuda_dmax(void)
{
    static int v = -1;
    if (v < 0) {
        const char *env = getenv("QWEN35_MTP_DMAX");
        v = env ? atoi(env) : PN_SPEC_DEPTH_MAX;
        if (v < 0) v = 0;
        if (v > PN_SPEC_DEPTH_MAX) v = PN_SPEC_DEPTH_MAX;
    }
    return v;
}

bool sg_qwen35_cuda_forward(const pn_model *model)
{
    return model && (model->forward == pn_qwen35moe_forward_cuda ||
                     model->forward == pn_bonsai_forward_cuda);
}

bool sg_mtp_cuda_available(pn_model *model)
{
    return sg_qwen35_cuda_forward(model) && pn_qwen35moe_mtp_ready(model);
}

int sg_mtp_cuda_begin(sg_mtp_cuda *mtp, const pn_model *model, int max_recent)
{
    uint32_t *keep = mtp->recent;
    int keep_cap = mtp->recent_cap, need = max_recent > 0 ? max_recent : 1;

    memset(mtp, 0, sizeof(*mtp));
    if (keep && keep_cap >= need) {
        mtp->recent = keep;
        mtp->recent_cap = keep_cap;
    } else {
        free(keep);
        mtp->recent = malloc((size_t)need * sizeof(uint32_t));
        mtp->recent_cap = mtp->recent ? need : 0;
    }
    if (!mtp->recent)
        return -1;
    mtp->max_recent = max_recent;
    pn_spec_depth_init(&mtp->depth, mtp_cuda_dmax());
    if (pn_qwen35moe_dense_ffn(model))
        pn_spec_depth_dense_priors(&mtp->depth);
    return 0;
}

typedef struct {
    sg_mtp_cuda *mtp;
    int pos;
} mtp_cuda_more_ctx;

/* The controller's per-draft continue decision from the head's confidence. */
static int mtp_cuda_more(void *ctx, int drafted, float conf)
{
    mtp_cuda_more_ctx *m = (mtp_cuda_more_ctx *)ctx;
    if (drafted >= 1 && drafted <= PN_SPEC_DEPTH_MAX)
        m->mtp->confs[drafted - 1] = conf;
    return pn_spec_depth_more(&m->mtp->depth, m->pos, drafted, m->mtp->confs);
}

/* Up to d drafts after `seed`, cut before the first stop token. Returns the count (0 when
 * the head could not draft: the cycle is then a one-row verify). A sampled request's drafts
 * are coupled to the uniforms their rows will draw with (spec_mtp_couple.h); the window row
 * 0 draws under is recent[0..n_recent). */
static int mtp_cuda_drafts(sg_mtp_cuda *mtp, pn_model *model, pn_state *state, uint32_t seed,
                           int d, int eos_id, int im_end, uint32_t *drafts,
                           pn_sampling_params *params, const uint32_t *recent, int n_recent)
{
    mtp_cuda_more_ctx more = { .mtp = mtp, .pos = state->pos };
    double t0 = sg_mtp_cuda_now_ms();
    bool coupled = pn_mtp_couple_begin(&mtp->couple, params, recent, n_recent, mtp->max_recent,
                                       model->config.n_vocab, d);
    bool pick = coupled || pn_mtp_couple_probe_on();
    int nd = pn_qwen35moe_mtp_draft_pick(model, state, seed, d, drafts, mtp->confs,
                                         mtp_cuda_more, &more,
                                         pick ? pn_mtp_couple_pick : NULL,
                                         pick ? &mtp->couple : NULL);

    if (nd < 1) {
        mtp->draft_failures++;
        return 0;
    }
    mtp->ms_draft += sg_mtp_cuda_now_ms() - t0;
    pn_spec_depth_observe_draft(&mtp->depth, nd, sg_mtp_cuda_now_ms() - t0);
    for (int i = 0; i < nd; i++)
        if ((int)drafts[i] == eos_id || (int)drafts[i] == im_end)
            return i;
    return nd;
}

int sg_mtp_cuda_choose(sg_mtp_cuda *mtp, int pos, int cap)
{
    int d = pn_spec_depth_choose(&mtp->depth, pos);

    if (d > cap)
        d = cap > 0 ? cap : 0;
    return d;
}

int sg_mtp_cuda_cycle(sg_mtp_cuda *mtp, pn_model *model, pn_state *state, uint32_t seed,
                      int depth, int cap, int eos_id, int im_end, sg_mtp_cut_fn cut,
                      void *cut_ctx, pn_sampling_params *params, const uint32_t *recent,
                      int n_recent, uint32_t *emit)
{
    uint32_t drafts[PN_SPEC_DEPTH_MAX];
    const int pos = state->pos;
    int d = depth >= 0 ? (depth < cap ? depth : (cap > 0 ? cap : 0))
                       : sg_mtp_cuda_choose(mtp, pos, cap), nd = 0;

    if (n_recent > mtp->max_recent)
        n_recent = mtp->max_recent;
    if (recent != mtp->recent)
        memcpy(mtp->recent, recent, (size_t)n_recent * sizeof(uint32_t));
    mtp->couple.n_drafts = 0;
    /* Prompt lookup: up to 4 drafts (the verify's 5 rows) copied from a recurring suffix. */
    uint32_t looked[PN_SPEC_DEPTH_MAX];
    const int lcap = cap < PN_SPEC_DEPTH_MAX ? cap : PN_SPEC_DEPTH_MAX;
    int nl = mtp->lookup && lcap > 0 ? sg_spec_lookup_draft(mtp->lookup, lcap, looked) : 0;
    bool looked_up = false;
    if (nl > 0 && mtp->lookup->mode == 1) {
        memcpy(drafts, looked, (size_t)nl * sizeof(*drafts));
        nd = nl;
        looked_up = true;
    } else {
        if (d > 0)
            nd = mtp_cuda_drafts(mtp, model, state, seed, d, eos_id, im_end, drafts, params,
                                 mtp->recent, n_recent);
        if (nl > 0 && nd >= 1 && sg_spec_lookup_agrees(mtp->lookup, looked[0], drafts[0])) {
            memcpy(drafts, looked, (size_t)nl * sizeof(*drafts));
            nd = nl;
            looked_up = true;
            mtp->couple.n_drafts = 1;   /* only draft 0 is the head's (the probe) */
        }
    }
    if (looked_up)
        for (int i = 0; i < nd; i++)
            if ((int)drafts[i] == eos_id || (int)drafts[i] == im_end) {
                nd = i;
                break;
            }
    if (nd > 0 && cut) {
        int keep = cut(cut_ctx, drafts, nd);
        mtp->grammar_cuts += keep < nd;
        nd = keep < nd ? (keep > 0 ? keep : 0) : nd;
    }
    double t0 = sg_mtp_cuda_now_ms();
    int ne = pn_qwen35moe_cuda_spec_step_ex(model, state, seed, drafts, nd, emit, params,
                                            mtp->recent, &n_recent, mtp->max_recent,
                                            &mtp->couple);
    if (ne < 1)
        return -1;
    const double verify_ms = sg_mtp_cuda_now_ms() - t0;
    mtp->ms_verify += verify_ms;
    pn_spec_depth_observe_verify(&mtp->depth, nd + 1, pos, verify_ms);
    if (looked_up) {
        /* The head's acceptance stays the head's: a lookup cycle books only its own. */
        sg_spec_lookup_report(mtp->lookup, nd, ne - 1);
        mtp->lookup_cycles++;
        mtp->lookup_drafted += nd;
        mtp->lookup_accepted += ne - 1;
    } else if (nd > 0) {
        pn_spec_depth_observe_accept(&mtp->depth, nd, ne - 1, mtp->confs);
    }
    mtp->cycles++;
    mtp->drafted += nd;
    mtp->accepted += ne - 1;
    mtp->depth_hist[nd]++;
    return ne;
}

void sg_mtp_cuda_report(const sg_mtp_cuda *mtp, int generated)
{
    if (mtp->cycles == 0) {
        if (mtp->plain_steps)
            fprintf(stderr, "[spec] mtp-cuda request: 0 cycles, plain steps %ld at %.1f ms, "
                            "generated %d\n", mtp->plain_steps,
                    mtp->ms_plain / mtp->plain_steps, generated);
        return;
    }
    fprintf(stderr, "[spec] mtp-cuda request: %ld cycles, %ld drafts, %ld accepted (%.3f), "
                    "%.2f tokens a cycle, depth hist %ld/%ld/%ld/%ld/%ld, draft failures %ld, "
                    "grammar cuts %ld, generated %d, ms/cycle draft %.1f verify %.1f bind %.1f "
                    "cycle %.1f, plain steps %ld at %.1f ms (%ld unfed)\n",
            mtp->cycles, mtp->drafted, mtp->accepted,
            mtp->drafted ? (double)mtp->accepted / (double)mtp->drafted : 0.0,
            (double)(mtp->cycles + mtp->accepted) / (double)mtp->cycles, mtp->depth_hist[0],
            mtp->depth_hist[1], mtp->depth_hist[2], mtp->depth_hist[3], mtp->depth_hist[4],
            mtp->draft_failures, mtp->grammar_cuts, generated, mtp->ms_draft / mtp->cycles,
            mtp->ms_verify / mtp->cycles, mtp->ms_bind / mtp->cycles,
            mtp->ms_cycle / mtp->cycles, mtp->plain_steps,
            mtp->plain_steps ? mtp->ms_plain / mtp->plain_steps : 0.0,
            mtp->plain_feed_misses);
    if (mtp->grammar_rows)
        fprintf(stderr, "[spec] mtp-cuda grammar draw: rows %ld, masked %ld\n",
                mtp->grammar_rows, mtp->grammar_masked);
    if (mtp->lookup_cycles)
        fprintf(stderr, "[spec] mtp-cuda lookup cycles %ld, drafted %ld, accepted %ld\n",
                mtp->lookup_cycles, mtp->lookup_drafted, mtp->lookup_accepted);
    if (mtp->couple.coupled_drafts)
        fprintf(stderr, "[spec] mtp-cuda coupled drafts %ld, argmax drafts %ld\n",
                mtp->couple.coupled_drafts, mtp->couple.argmax_drafts);
}

void sg_mtp_cuda_end(sg_mtp_cuda *mtp)
{
    free(mtp->recent);
    mtp->recent = NULL;
    mtp->recent_cap = 0;
}

#endif /* USE_CUDA */
