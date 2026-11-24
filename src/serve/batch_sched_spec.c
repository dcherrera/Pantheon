/* SPDX-License-Identifier: AGPL-3.0-or-later */
/* Copyright (C) 2026 David (Corey) Herrera */
/**
 * @file batch_sched_spec.c
 * @brief Implementation of solo-slot MTP speculation for the batched driver
 *        (batch_sched_spec.h).
 *
 * @status complete
 * @issues None
 * @todo None
 */
#include "batch_sched_spec.h"

#ifdef USE_CUDA

#include "../core/engine.h"
#include "../core/spec_mtp.h"
#include "../models/qwen35moe.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

/* PN_MTP_SERVE_SPEC=0: an MTP model decodes plain (the head stays loaded). */
static bool bs_spec_env_on(void)
{
    static int v = -1;
    if (v < 0) {
        const char *env = getenv("PN_MTP_SERVE_SPEC");
        v = !(env && atoi(env) == 0);
    }
    return v;
}

/* PN_GRAMMAR_ARM_WINDOW=1 (as batch_sched.c): a grammar slot's rows are the top-k rows
 * except while the mask is live — the only case speculation may share with plain decode. */
static bool bs_spec_arm_window_on(void)
{
    static int v = -1;
    if (v < 0) {
        const char *env = getenv("PN_GRAMMAR_ARM_WINDOW");
        v = (env && env[0] && env[0] != '0') ? 1 : 0;
    }
    return v;
}

/* An AUTO tool grammar the driver can look ahead over (pn_tool_grammar_hook.clone). */
static bool bs_spec_grammar_lookahead(const pn_tool_grammar_hook *tgh)
{
    return tgh->clone && tgh->clone_free && tgh->refresh && tgh->advance && tgh->token_text &&
           tgh->st && bs_spec_arm_window_on();
}

/* PN_MTP_GRAMMAR=1 turns the grammar draw on. Default off: exact, but no gain yet at real
 * tool-turn depth (~19k), where a multi-row verify costs ~2.7 plain steps. Off, a tool-grammar
 * slot never speculates (off:tool-grammar unless the arming-window look-ahead applies). */
static bool bs_spec_grammar_env_on(void)
{
    static int v = -1;
    if (v < 0) {
        const char *env = getenv("PN_MTP_GRAMMAR");
        v = env && atoi(env) != 0;
    }
    return v;
}

bool bs_spec_grammar_draws(const pn_tool_grammar_hook *tgh)
{
    /* Window off: the owner armed the full row for the whole turn, so plain decode draws every
     * row of this slot dense and the cycle can fetch the same rows. A steer moves the walk from
     * inside the sampler, which the deposit's re-advance would not replay: plain only. */
    return tgh && tgh->clone && tgh->clone_free && tgh->refresh && tgh->advance &&
           tgh->token_text && tgh->st && !tgh->steer && !bs_spec_arm_window_on() &&
           bs_spec_grammar_env_on();
}

bool bs_spec_eligible(const pn_batch_sched *sched, const bs_slot *sl)
{
    const pn_tool_grammar_hook *tgh = sl->sp.tool_grammar;

    if (tgh && !bs_spec_grammar_draws(tgh) &&
        (!bs_spec_grammar_lookahead(tgh) || tgh->refresh(tgh->st)))
        return false;   /* no grammar draw or look-ahead, or the mask is live for this row */
    return sched->mtp && sched->driver_sample && bs_spec_env_on() && sl->mtp_on &&
           !sl->pd_stepped &&
           !sl->finished && sl->budget >= 1 && sl->ring && sl->ring_w < sl->ring_cap &&
           !sl->sp.logprobs && !(sl->sp.logit_bias && sl->sp.n_logit_bias > 0) &&
           sl->reason_close_i >= sl->reason_close_n;
}

int bs_spec_grammar_cut(void *ctx, const uint32_t *drafts, int nd)
{
    const pn_tool_grammar_hook *tgh = (const pn_tool_grammar_hook *)ctx;
    void *walk = tgh->clone(tgh->st);
    int keep = 0;

    if (!walk)
        return 0;
    for (; keep < nd; keep++) {
        const char *text = tgh->token_text(tgh->user, drafts[keep]);
        if (text)
            tgh->advance(walk, text);
        if (tgh->refresh(walk))
            break;   /* the row after this draft would be masked */
    }
    tgh->clone_free(walk);
    return keep;
}

int bs_spec_grammar_complete_cut(void *ctx, const uint32_t *drafts, int nd)
{
    const pn_tool_grammar_hook *tgh = (const pn_tool_grammar_hook *)ctx;
    void *walk = tgh->clone(tgh->st);
    int keep = 0;

    if (!walk)
        return 0;
    for (; keep < nd; keep++) {
        const char *text = tgh->token_text(tgh->user, drafts[keep]);
        /* A draft the live mask refuses cannot equal its row's draw: drop it and the rest. */
        if (text && tgh->allowed && tgh->refresh(walk) && !tgh->allowed(walk, text))
            break;
        if (text)
            tgh->advance(walk, text);
        if (tgh->complete && tgh->complete(walk))
            break;   /* this draft would close the call: only the cycle's last token may */
    }
    tgh->clone_free(walk);
    return keep;
}

const char *bs_spec_plain_reason(const pn_batch_sched *sched, const pn_sampling_params *sp)
{
    if (!sched->mtp)
        return NULL;
    if (!sched->driver_sample)
        return "off:owner-sampling";
    if (!bs_spec_env_on())
        return "off:env";
    if (sp->tool_grammar && !bs_spec_grammar_draws(sp->tool_grammar) &&
        !bs_spec_grammar_lookahead(sp->tool_grammar))
        return "off:tool-grammar";
    if (sp->logprobs)
        return "off:logprobs";
    if (sp->logit_bias && sp->n_logit_bias > 0)
        return "off:logit-bias";
    return NULL;
}

void bs_spec_after_plain_step(pn_batch_sched *sched, bs_slot *sl, uint32_t token, double ms)
{
    sl->mtp.plain_steps++;
    sl->mtp.ms_plain += ms;
    pn_spec_depth_observe_plain(&sl->mtp.depth, sl->state->pos - 1, ms);
    /* The step's output row joins the head's backlog, so the next cycle drafts from an
     * unbroken history instead of restarting it (a failure only costs drafts). */
    if (pn_qwen35moe_mtp_push_rows(sched->model, sl->state, &token, 1) != 0)
        sl->mtp.plain_feed_misses++;
}

bool bs_spec_shadow_on(void)
{
    static int v = -1;
    if (v < 0) {
        const char *env = getenv("PN_MTP_SHADOW");
        v = (env && env[0] && env[0] != '0') ? 1 : 0;
        if (v)
            fprintf(stderr, "[spec] PN_MTP_SHADOW=1: grammar slots draft beside plain steps "
                            "(acceptance probe, slower)\n");
    }
    return v;
}

/* Score probe draft `q` against the emitted tokens; returns true when its outcome is known
 * (a mismatch seen, every draft matched, or `final`), booking it in the histogram. */
static bool bs_spec_shadow_score(sg_mtp_cuda *mtp, const bs_slot *sl,
                                 const sg_mtp_shadow_draft *q, bool final)
{
    int m = 0;

    while (m < q->n && q->at + 1 + m < sl->ring_w && sl->ring[q->at + 1 + m] == q->d[m])
        m++;
    if (!final && m < q->n && q->at + 1 + m >= sl->ring_w)
        return false;   /* still matching: the next token decides */
    mtp->shadow_hist[q->masked ? 1 : 0][m]++;
    return true;
}

static void bs_spec_shadow_drain(bs_slot *sl, bool final)
{
    sg_mtp_cuda *mtp = &sl->mtp;
    int keep = 0;

    for (int i = 0; i < mtp->shadow_n; i++)
        if (!bs_spec_shadow_score(mtp, sl, &mtp->shadow[i], final))
            mtp->shadow[keep++] = mtp->shadow[i];
    mtp->shadow_n = keep;
}

void bs_spec_shadow_step(pn_batch_sched *sched, bs_slot *sl)
{
    const pn_tool_grammar_hook *tgh = sl->sp.tool_grammar;
    sg_mtp_cuda *mtp = &sl->mtp;
    float confs[PN_SPEC_DEPTH_MAX];

    bs_spec_shadow_drain(sl, false);
    if (!sl->ring || sl->ring_w < 1 || mtp->shadow_n >= SG_MTP_SHADOW_PENDING)
        return;
    sg_mtp_shadow_draft *q = &mtp->shadow[mtp->shadow_n];
    const double t0 = sg_mtp_cuda_now_ms();
    int nd = pn_qwen35moe_mtp_draft(sched->model, sl->state, sl->next_token, PN_SPEC_DEPTH_MAX,
                                    q->d, confs, NULL, NULL);
    mtp->shadow_ms += sg_mtp_cuda_now_ms() - t0;
    if (nd < 1)
        return;
    mtp->shadow_drafts++;
    for (q->n = 0; q->n < nd; q->n++)   /* a cycle cuts drafts before a stop token */
        if ((int)q->d[q->n] == sched->eos_id || (int)q->d[q->n] == sched->im_end_id)
            break;
    q->at = sl->ring_w - 1;
    q->masked = tgh && tgh->refresh && tgh->st && tgh->refresh(tgh->st);
    mtp->shadow_n++;
}

void bs_spec_shadow_report(bs_slot *sl, int slot)
{
    const long *u = sl->mtp.shadow_hist[0], *k = sl->mtp.shadow_hist[1];

    bs_spec_shadow_drain(sl, true);
    long nu = u[0] + u[1] + u[2] + u[3] + u[4], nk = k[0] + k[1] + k[2] + k[3] + k[4];
    if (nu + nk == 0)
        return;
    fprintf(stderr, "[spec] shadow drafts slot=%d generated %d: unmasked rows %ld accept "
                    "%ld/%ld/%ld/%ld/%ld, masked rows %ld accept %ld/%ld/%ld/%ld/%ld, "
                    "draft ms %.1f\n",
            slot, sl->ring_w, nu, u[0], u[1], u[2], u[3], u[4], nk, k[0], k[1], k[2], k[3],
            k[4], sl->mtp.shadow_drafts ? sl->mtp.shadow_ms / sl->mtp.shadow_drafts : 0.0);
}

bool bs_spec_stress_plain(const bs_slot *sl)
{
    static int every = -1;
    if (every < 0) {
        const char *env = getenv("QWEN35_MTP_STRESS_PLAIN");
        every = env ? atoi(env) : 0;
    }
    return every > 0 && (sl->mtp.cycles + sl->mtp.plain_steps) % every == every - 1;
}

void bs_spec_arm_controller(bs_slot *sl)
{
    static int v = -1;
    if (v < 0) {
        const char *env = getenv("QWEN35_MTP_D0_PLAIN");
        v = !(env && atoi(env) == 0);
    }
    if (v)
        pn_spec_depth_d0_is_plain(&sl->mtp.depth);
}

int bs_spec_draft_cap(const bs_slot *sl)
{
    int cap = sl->budget - 1;

    if (sl->ring_cap - sl->ring_w - 1 < cap)
        cap = sl->ring_cap - sl->ring_w - 1;
    /* Stop short of the close window: inside it any line end may close. */
    if (sl->reason_in_think && sl->reason_budget > 0 && !sl->reason_forced &&
        sl->reason_soft - sl->reason_toks - 1 < cap)
        cap = sl->reason_soft - sl->reason_toks - 1;
    return cap > 0 ? cap : 0;
}

int bs_spec_stage_window(bs_slot *sl, uint32_t seed)
{
    int n = sl->n_recent < sl->mtp.max_recent ? sl->n_recent : sl->mtp.max_recent;

    if (n > 0)
        memcpy(sl->mtp.recent, sl->recent, (size_t)n * sizeof(uint32_t));
    pn_mtp_recent_push(sl->mtp.recent, &n, sl->mtp.max_recent, seed);
    return n;
}

/* A cycle under the tool grammar: the draws run over a scratch clone of the live walk (the
 * live walk is advanced by the deposit, over the emitted tokens), drafts are cut before the
 * call's closing token, and the rows are drawn dense (spec_mtp_step.h). Returns the emitted
 * count, or -1. The solo slot owns the sampler counters for the cycle, so their delta is its
 * masked-row count. */
static int bs_spec_cycle_grammar(pn_batch_sched *sched, bs_slot *sl, uint32_t seed, int depth,
                                 int cap, pn_sampling_params *sp, int n_recent, uint32_t *emit)
{
    const pn_tool_grammar_hook *tgh = sp->tool_grammar;
    pn_tool_grammar_hook scratch = *tgh;
    const unsigned long masked0 = pn_sampler_grammar_mask_steps();

    scratch.st = tgh->clone(tgh->st);
    if (!scratch.st)
        return -1;
    sp->tool_grammar = &scratch;
    int ne = sg_mtp_cuda_cycle(&sl->mtp, sched->model, sl->state, seed, depth, cap,
                               sched->eos_id, sched->im_end_id, bs_spec_grammar_complete_cut,
                               (void *)tgh, sp, sl->mtp.recent, n_recent, emit);
    sp->tool_grammar = tgh;
    tgh->clone_free(scratch.st);
    if (ne > 0) {
        sl->mtp.grammar_rows += ne;
        sl->mtp.grammar_masked += (long)(pn_sampler_grammar_mask_steps() - masked0);
    }
    return ne;
}

int bs_spec_cycle_gpu(pn_batch_sched *sched, bs_slot *sl, int gpu_row, uint32_t seed,
                      int depth, int cap, pn_sampling_params *sp, int n_recent, uint32_t *emit)
{
    pn_state *st = sl->state;

    /* The shared buffers hold this state only up to its last cycle; plain steps since then
     * advanced the slot's block instead. The recurrent image re-seeds from the block; the
     * shared KV ring keeps its rows and tops up only the plain-stepped ones (D2D). A slot
     * that never cycled drops the whole binding (its block is the only source). */
    if (sl->spec_live_pos < 0)
        pn_qwen35moe_gpu_forget_state(sched->model, st);
    else if (sl->spec_live_pos != st->pos)
        pn_qwen35moe_gpu_forget_ssm(sched->model, st);
    sl->spec_live_pos = -1;
    const pn_tool_grammar_hook *tgh = sp->tool_grammar;
    int ne;
    if (bs_spec_grammar_draws(tgh))
        ne = bs_spec_cycle_grammar(sched, sl, seed, depth, cap, sp, n_recent, emit);
    else {
        sp->tool_grammar = NULL;   /* every row is outside the mask: drawn as a hook-free row */
        ne = sg_mtp_cuda_cycle(&sl->mtp, sched->model, st, seed, depth, cap, sched->eos_id,
                               sched->im_end_id, tgh ? bs_spec_grammar_cut : NULL,
                               (void *)tgh, sp, sl->mtp.recent, n_recent, emit);
        sp->tool_grammar = tgh;
    }
    if (ne < 1)
        return -1;
    const double t_bind = sg_mtp_cuda_now_ms();
    if (pn_qwen35moe_decode_slot_bind(sched->model, st, gpu_row, sched->cap) != 0)
        return -1;
    const double bind_ms = sg_mtp_cuda_now_ms() - t_bind;
    sl->mtp.ms_bind += bind_ms;
    pn_spec_depth_observe_extra(&sl->mtp.depth, bind_ms);   /* a plain step pays none */
    sl->spec_live_pos = st->pos;
    return ne;
}

#endif /* USE_CUDA */
