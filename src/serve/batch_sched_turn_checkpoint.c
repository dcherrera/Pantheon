/* SPDX-License-Identifier: AGPL-3.0-or-later */
/* Copyright (C) 2026 David (Corey) Herrera */
/**
 * @file batch_sched_turn_checkpoint.c
 * @brief Implementation of the turn-start checkpoint (rung 1c) — see
 *        batch_sched_turn_checkpoint.h for the design.
 *
 * @status Complete
 * @issues None
 * @todo None
 */

#include "batch_sched_turn_checkpoint.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "batch_sched.h"
#include "batch_sched_slots.h"
#include "request_profile.h"

bool bs_turn_ckpt_enabled(void)
{
    static int cached = -1;
    if (cached < 0) {
        const char *env = getenv("QWEN35_TURN_CHECKPOINT");
        cached = (env && env[0] == '0') ? 0 : 1;   /* default on (David, 2026-11-03) */
        if (cached)
            fprintf(stderr, "[batch_sched] turn-start checkpoint enabled "
                    "(QWEN35_TURN_CHECKPOINT=0 turns it off)\n");
    }
    return cached != 0;
}

/* *dst = a[0, a_n) ++ b[0, b_n), growing the buffer. 0, or -1 on OOM. */
static int ckpt_store_tokens(uint32_t **dst, int *cap, int *n,
                             const uint32_t *a, int a_n,
                             const uint32_t *b, int b_n)
{
    int want = a_n + b_n;
    if (*cap < want) {
        uint32_t *grown = realloc(*dst, (size_t)want * sizeof(uint32_t));
        if (!grown)
            return -1;
        *dst = grown;
        *cap = want;
    }
    if (a_n > 0)
        memcpy(*dst, a, (size_t)a_n * sizeof(uint32_t));
    if (b_n > 0)
        memcpy(*dst + a_n, b, (size_t)b_n * sizeof(uint32_t));
    *n = want;
    return 0;
}

/* Save slot `idx`'s state at client position `cut` into card checkpoint
 * `ck_index`, described by `ck`. `what` names it in the log. */
static int ckpt_capture_into(pn_batch_sched *sched, int idx, bs_turn_ckpt *ck,
                             int ck_index, const char *what, const char *sid,
                             const uint32_t *prompt, int cut)
{
    bs_slot *sl = &sched->slots[idx];
    (void)ck_index;   /* card-only (USE_CUDA) */
    /* The KV holds hist[0, ct_keep) then prompt[ct_prompt_from ..) (the
     * think-elision seeding; both 0 on every other rung). */
    int keep = sl->ct_keep;
    int from = sl->ct_prompt_from;
    int kv_n = keep + (cut - from);
    if (cut < from || keep > sl->hist_n || (int)sl->state->pos != kv_n) {
        fprintf(stderr, "[batch_sched] %s skipped slot=%d sid=%s: "
                "state pos %d is not the checkpoint position %d\n", what, idx,
                sid, (int)sl->state->pos, kv_n);
        return -1;
    }
#ifdef USE_CUDA
    bs_slots_prefill_drain(sched);
    if (bs_slots_checkpoint_save(sched, sl->state, ck_index) != 0) {
        fprintf(stderr, "[batch_sched] %s save failed slot=%d "
                "sid=%s at pos %d\n", what, idx, sid, kv_n);
        return -1;
    }
#endif
    /* The card now holds the new checkpoint, so the old bookkeeping no longer
     * describes it whatever happens below. */
    ck->valid = false;
    if (ckpt_store_tokens(&ck->client, &ck->client_cap, &ck->client_n,
                          prompt, cut, NULL, 0) != 0 ||
        ckpt_store_tokens(&ck->kv, &ck->kv_cap, &ck->kv_n, sl->hist, keep,
                          prompt + from, cut - from) != 0)
        return -1;
    snprintf(ck->sid, sizeof(ck->sid), "%s", sid);
    ck->valid = true;
    if (pn_reqprof_enabled())
        fprintf(stderr, "[reqprof] %s saved slot=%d sid=%s: "
                "client %d tokens, kv pos %d\n", what, idx, sid, cut, kv_n);
    return 0;
}

int bs_turn_ckpt_capture(pn_batch_sched *sched, int idx, const char *sid,
                         const uint32_t *prompt, int cut)
{
    if (!bs_turn_ckpt_enabled() || !sched || idx < 0 || idx >= sched->cap ||
        !sid || !sid[0] || !prompt || cut <= 0)
        return -1;
    return ckpt_capture_into(sched, idx, &sched->slots[idx].ckpt, idx,
                             "turn-checkpoint", sid, prompt, cut);
}

bool bs_turn_ckpt_fits(const bs_turn_ckpt *ck, const char *sid,
                       const uint32_t *tokens, int n_ctx)
{
    return ck && ck->valid && sid && sid[0] && tokens &&
           strncmp(ck->sid, sid, sizeof(ck->sid)) == 0 &&
           ck->client_n > 0 && ck->client_n < n_ctx &&
           memcmp(ck->client, tokens,
                  (size_t)ck->client_n * sizeof(uint32_t)) == 0;
}

/* PN_REQPROF: why a checkpoint this session holds did not apply. */
static void ckpt_log_miss(int idx, const char *sid, const bs_turn_ckpt *ck,
                          const uint32_t *tokens, int n_ctx)
{
    if (!pn_reqprof_enabled() || !ck->valid || !sid ||
        strncmp(ck->sid, sid, sizeof(ck->sid)) != 0)
        return;
    int n = ck->client_n < n_ctx ? ck->client_n : n_ctx;
    int first_diff = -1;
    for (int i = 0; i < n && first_diff < 0; i++)
        if (ck->client[i] != tokens[i])
            first_diff = i;
    fprintf(stderr, "[reqprof] turn-checkpoint MISS slot=%d sid=%s: client_n=%d "
            "n_ctx=%d first_diff=%d\n", idx, sid, ck->client_n, n_ctx,
            first_diff);
}

/* Make the slot's warm bookkeeping describe the restored record `ck` (its
 * checkpoint, or the prefix an aborted prefill kept). */
static int ckpt_publish_history(pn_batch_sched *sched, bs_slot *sl,
                                const bs_turn_ckpt *ck, const char *sid)
{
    int rc = 0;
    pthread_mutex_lock(&sched->mu);
    if (sl->hist_cap < ck->kv_n) {
        uint32_t *grown = realloc(sl->hist, (size_t)ck->kv_n * sizeof(uint32_t));
        if (grown) {
            sl->hist = grown;
            sl->hist_cap = ck->kv_n;
        } else {
            rc = -1;
        }
    }
    if (rc == 0) {
        memcpy(sl->hist, ck->kv, (size_t)ck->kv_n * sizeof(uint32_t));
        sl->hist_n = ck->kv_n;
        snprintf(sl->sid, sizeof(sl->sid), "%s", sid);
    } else {
        sl->sid[0] = '\0';
        sl->hist_n = 0;
    }
    /* The projections described the dropped turn's reply. */
    sl->vis_n = 0;
    sl->vis_think_n = 0;
    pthread_mutex_unlock(&sched->mu);
    return rc;
}

/* Put slot `idx` back at card checkpoint `ck_index` described by `ck` when
 * `tokens` start with its client tokens. 1 resumed, 0 not applicable. */
static int ckpt_resume_from(pn_batch_sched *sched, int idx, bs_turn_ckpt *ck,
                            int ck_index, const char *what, const char *sid,
                            const uint32_t *tokens, int n_ctx, int *delta)
{
    bs_slot *sl = &sched->slots[idx];
    (void)ck_index;   /* card-only (USE_CUDA) */
    if (!bs_turn_ckpt_fits(ck, sid, tokens, n_ctx)) {
        ckpt_log_miss(idx, sid, ck, tokens, n_ctx);
        return 0;
    }
    /* The KV ends this turn kv_n - client_n past the n_ctx admission cleared
     * (the think blocks rung 1b kept); refuse rather than overrun the window. */
    int cap = pn_session_ctx_cap(sched->model);
    if (cap > 0 && ck->kv_n + (n_ctx - ck->client_n) > cap)
        return 0;
#ifdef USE_CUDA
    if (bs_slots_checkpoint_restore(sched, sl->state, idx, ck_index,
                                    ck->kv_n) != 0) {
        fprintf(stderr, "[batch_sched] %s restore refused slot=%d "
                "sid=%s (the card no longer holds it)\n", what, idx, sid);
        bs_turn_ckpt_forget(ck);
        return 0;
    }
    /* Same re-seed contract as rung 1: the next prefill re-seeds the shared
     * buffers from the slot's block, which now holds the checkpoint. The MTP
     * head's backlog described the dropped turn; it restarts here. */
    bs_slots_forget(sched, sl->state);
    bs_slots_mtp_rewind(sched, sl->state);
#else
    /* Non-CUDA builds never run the scheduler (bs_enabled is false); this arm
     * keeps the host tests of the ladder's bookkeeping meaningful. */
    sl->state->pos = ck->kv_n;
#endif
    if (ckpt_publish_history(sched, sl, ck, sid) != 0)
        return 0;   /* OOM: the re-prefilling rungs overwrite the state */
    sl->ct_keep = ck->kv_n;
    sl->ct_prompt_from = ck->client_n;
    *delta = ck->client_n;
    fprintf(stderr, "[batch_sched] %s resume slot=%d sid=%s: "
            "resume at %d (kv %d), prefill %d of %d tokens\n", what, idx, sid,
            ck->client_n, ck->kv_n, n_ctx - ck->client_n, n_ctx);
    return 1;
}

int bs_turn_ckpt_resume(pn_batch_sched *sched, int idx, const char *sid,
                        const uint32_t *tokens, int n_ctx, int *delta)
{
    if (!bs_turn_ckpt_enabled() || !sched || idx < 0 || idx >= sched->cap ||
        !delta)
        return 0;
    return ckpt_resume_from(sched, idx, &sched->slots[idx].ckpt, idx,
                            "turn-checkpoint", sid, tokens, n_ctx, delta);
}

bool bs_think_history_drop_last(const pn_batch_sched *sched)
{
    return sched && sched->turn_ckpt_on && bs_turn_ckpt_enabled();
}

/* Is the checkpoint's KV exactly its client tokens (no think block that a
 * projection resume kept before it)? */
static bool ckpt_kv_is_client(const bs_turn_ckpt *ck)
{
    return ck->kv_n == ck->client_n &&
           memcmp(ck->kv, ck->client, (size_t)ck->kv_n * sizeof(uint32_t)) == 0;
}

int bs_turn_ckpt_resume_exact(pn_batch_sched *sched, int idx, const char *sid,
                              const uint32_t *tokens, int n_ctx, int *delta)
{
    if (!bs_turn_ckpt_enabled() || !sched || idx < 0 || idx >= sched->cap ||
        !delta)
        return 0;
    bs_turn_ckpt *ck = &sched->slots[idx].ckpt;
    if (bs_turn_ckpt_fits(ck, sid, tokens, n_ctx) && !ckpt_kv_is_client(ck)) {
        fprintf(stderr, "[batch_sched] drop_last: turn checkpoint slot=%d "
                "sid=%s holds %d KV tokens for %d request tokens (think blocks "
                "an earlier projection kept); not resumed\n", idx, sid,
                ck->kv_n, ck->client_n);
        return 0;
    }
    return ckpt_resume_from(sched, idx, ck, idx, "turn-checkpoint", sid,
                            tokens, n_ctx, delta);
}

/* ── Rung 1d: the divergence checkpoint ── */

/* Re-prefilling at least this many tokens past the resume point is worth a
 * checkpoint save (~68 MiB of device copies) at the divergence point. */
#define BS_DIV_CKPT_MIN_GAIN 512

bool bs_div_ckpt_wanted(void)
{
    static int cached = -1;
    if (cached < 0) {
        const char *env = getenv("QWEN35_DIV_CHECKPOINT");
        cached = (env && env[0] == '0') ? 0 : 1;
    }
    return cached != 0;
}

bool bs_div_ckpt_enabled(const pn_batch_sched *sched)
{
    return sched && sched->div_ckpt_on && bs_turn_ckpt_enabled();
}

/* First index where a[0, a_n) and b[0, b_n) differ. */
static int ckpt_first_diff(const uint32_t *a, int a_n, const uint32_t *b,
                           int b_n)
{
    int m = a_n < b_n ? a_n : b_n;
    for (int i = 0; i < m; i++)
        if (a[i] != b[i])
            return i;
    return m;
}

int bs_div_point(const pn_batch_sched *sched, int idx, const char *sid,
                 const uint32_t *tokens, int n_ctx)
{
    if (!bs_div_ckpt_enabled(sched) || idx < 0 || idx >= sched->cap || !sid ||
        !sid[0] || !tokens)
        return -1;
    const bs_slot *sl = &sched->slots[idx];
    if (sl->sid[0] && strncmp(sl->sid, sid, sizeof(sl->sid)) == 0 &&
        sl->hist_n > 0)
        return ckpt_first_diff(sl->hist, sl->hist_n, tokens, n_ctx);
    if (sl->ckpt.valid && strncmp(sl->ckpt.sid, sid, sizeof(sl->ckpt.sid)) == 0)
        return ckpt_first_diff(sl->ckpt.client, sl->ckpt.client_n, tokens,
                               n_ctx);
    return -1;
}

void bs_div_ckpt_plan(pn_batch_sched *sched, int idx, int div_point, int delta,
                      int n_ctx)
{
    if (!sched || idx < 0 || idx >= sched->cap)
        return;
    bs_slot *sl = &sched->slots[idx];
    sl->div_cut = -1;
    /* KV rows from the resume position on are rewritten by this prefill. */
    if (sl->divck.valid && (int)sl->state->pos < sl->divck.kv_n)
        bs_turn_ckpt_forget(&sl->divck);
    if (!bs_div_ckpt_enabled(sched) || div_point < 0 || div_point >= n_ctx ||
        div_point - delta < BS_DIV_CKPT_MIN_GAIN)
        return;
    sl->div_cut = div_point;
}

int bs_div_ckpt_cut(const pn_batch_sched *sched, int idx)
{
    if (!bs_div_ckpt_enabled(sched) || idx < 0 || idx >= sched->cap)
        return -1;
    return sched->slots[idx].div_cut;
}

int bs_div_ckpt_capture(pn_batch_sched *sched, int idx, const char *sid,
                        const uint32_t *prompt, int cut)
{
    if (!bs_div_ckpt_enabled(sched) || idx < 0 || idx >= sched->cap || !sid ||
        !sid[0] || !prompt || cut <= 0)
        return -1;
    bs_slot *sl = &sched->slots[idx];
    sl->div_cut = -1;
    int rc = ckpt_capture_into(sched, idx, &sl->divck, sched->cap + idx,
                               "divergence-checkpoint", sid, prompt, cut);
    if (rc == 0)
        fprintf(stderr, "[batch_sched] divergence-checkpoint saved slot=%d "
                "sid=%s at %d (kv %d)\n", idx, sid, cut, sl->divck.kv_n);
    return rc;
}

int bs_div_ckpt_resume(pn_batch_sched *sched, int idx, const char *sid,
                       const uint32_t *tokens, int n_ctx, int *delta)
{
    if (!bs_div_ckpt_enabled(sched) || idx < 0 || idx >= sched->cap || !delta)
        return 0;
    return ckpt_resume_from(sched, idx, &sched->slots[idx].divck,
                            sched->cap + idx, "divergence-checkpoint", sid,
                            tokens, n_ctx, delta);
}

int bs_partial_prefill_keep(pn_batch_sched *sched, int idx, const char *sid,
                            const uint32_t *prompt, int n_prompt, int sys_len)
{
    if (!bs_turn_ckpt_enabled() || !sched || idx < 0 || idx >= sched->cap ||
        !sid || !sid[0] || !prompt)
        return -1;
    bs_slot *sl = &sched->slots[idx];
    bs_turn_ckpt *pp = &sl->partial;
    bs_turn_ckpt_forget(pp);
    /* The state holds hist[0, keep) then prompt[from, client_n): the seeding
     * of whichever rung started this prefill (both 0 on a cold one). */
    int keep = sl->ct_keep;
    int from = sl->ct_prompt_from;
    int kv_n = (int)sl->state->pos;
    int client_n = from + (kv_n - keep);
    /* At or below the system prefix there is nothing to keep: the SYS1
     * snapshot resumes there for free. An abort right after its restore
     * (before any chunk) would otherwise keep a block that never received
     * the restored rows, and the resume would be refused. */
    if (kv_n <= 0 || kv_n < keep || keep > sl->hist_n || client_n > n_prompt ||
        (sys_len > 0 && client_n <= sys_len))
        return -1;
#ifdef USE_CUDA
    if (bs_slots_save_live(sched, sl->state, idx) != 0) {
        fprintf(stderr, "[batch_sched] slot %d: saving the aborted prefill "
                "failed; not kept\n", idx);
        return -1;
    }
#endif
    if (ckpt_store_tokens(&pp->client, &pp->client_cap, &pp->client_n,
                          prompt, client_n, NULL, 0) != 0 ||
        ckpt_store_tokens(&pp->kv, &pp->kv_cap, &pp->kv_n, sl->hist, keep,
                          prompt + from, client_n - from) != 0)
        return -1;
    snprintf(pp->sid, sizeof(pp->sid), "%s", sid);
    pp->valid = true;
    fprintf(stderr, "[batch_sched] slot %d: prefill aborted at %d of %d "
            "tokens; keeping that prefix for sid=%s\n", idx, client_n,
            n_prompt, sid);
    return 0;
}

int bs_partial_prefill_resume(pn_batch_sched *sched, int idx, const char *sid,
                              const uint32_t *tokens, int n_ctx, int *delta)
{
    if (!bs_turn_ckpt_enabled() || !sched || idx < 0 || idx >= sched->cap ||
        !delta)
        return 0;
    bs_slot *sl = &sched->slots[idx];
    const bs_turn_ckpt *pp = &sl->partial;
    if (!bs_turn_ckpt_fits(pp, sid, tokens, n_ctx) ||
        (int)sl->state->pos != pp->kv_n)
        return 0;
    int cap = pn_session_ctx_cap(sched->model);
    if (cap > 0 && pp->kv_n + (n_ctx - pp->client_n) > cap)
        return 0;
#ifdef USE_CUDA
    if (!bs_slots_live_resumable(sched, sl->state, idx)) {
        fprintf(stderr, "[batch_sched] partial-prefill resume refused slot=%d "
                "sid=%s (the block no longer holds it)\n", idx, sid);
        return 0;
    }
    /* Rung 1's re-seed contract: the prefill re-seeds from the block. */
    bs_slots_forget(sched, sl->state);
#endif
    if (ckpt_publish_history(sched, sl, pp, sid) != 0)
        return 0;
    sl->ct_keep = pp->kv_n;
    sl->ct_prompt_from = pp->client_n;
    *delta = pp->client_n;
    fprintf(stderr, "[batch_sched] partial-prefill resume slot=%d sid=%s: "
            "resume at %d (kv %d), prefill %d of %d tokens\n", idx, sid,
            pp->client_n, pp->kv_n, n_ctx - pp->client_n, n_ctx);
    return 1;
}

void bs_turn_ckpt_forget(bs_turn_ckpt *ck)
{
    if (ck)
        ck->valid = false;
}

void bs_turn_ckpt_free(bs_turn_ckpt *ck)
{
    if (!ck)
        return;
    free(ck->client);
    free(ck->kv);
    memset(ck, 0, sizeof(*ck));
}
