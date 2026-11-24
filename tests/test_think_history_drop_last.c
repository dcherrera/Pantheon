/* SPDX-License-Identifier: AGPL-3.0-or-later */
/* Copyright (C) 2026 David (Corey) Herrera */
/**
 * @file test_think_history_drop_last.c
 * @brief `think_history: "drop_last"` on the batched ladder (CPU, no model):
 *        the request resumes at the last turn's start checkpoint (rung 1c),
 *        the slot's KV and history are the request as rendered, the turn saves
 *        its own checkpoint, and the chain continues; no usable checkpoint →
 *        the system prefix; without the flag rung 1b still hits.
 *
 * The checkpoint copy itself (GDN + conv images, MTP backlog reset) is card
 * work and is the bank gate; here the non-CUDA arm only moves state->pos.
 *
 * @status Complete
 * @issues None
 * @todo None
 */

#include "think_history_ladder_harness.h"
#include "../src/models/qwen35moe_checkpoint_band.h"

/* Turn 1: 8 system + 9 conversation tokens, then "<|im_start|>assistant\n"
 * (17, 18) and the forced "<think>" (19): turn_start 17, think prefill 1. */
#define SYS_LEN     8
#define T1_N        20
#define T1_CUT      17
#define IM_START    151644u
#define ASSISTANT   77091u
#define THINK_OPEN  248068u
#define THINK_CLOSE 248069u

static const uint32_t SYS_A[SYS_LEN] = { 1, 2, 3, 4, 5, 6, 7, 8 };

static void turn1_prompt(uint32_t *p)
{
    memcpy(p, SYS_A, sizeof(SYS_A));
    for (int i = SYS_LEN; i < T1_CUT; i++)
        p[i] = (uint32_t)(1000 + i);
    p[17] = IM_START;
    p[18] = ASSISTANT;
    p[19] = THINK_OPEN;
}

/* Turn 1 generated a think block, its close and a 4-token tool call. */
static const uint32_t T1_GEN[] = { 900, 901, 902, THINK_CLOSE, 500, 501, 502,
                                   503 };
#define T1_GEN_N ((int)(sizeof(T1_GEN) / sizeof(T1_GEN[0])))
#define T1_KV_N  (T1_N + T1_GEN_N)

static pn_model qwen35_model(void)
{
    pn_model m;
    memset(&m, 0, sizeof(m));
    m.config.arch = PN_ARCH_QWEN35MOE;
    return m;
}

/* Finish a turn whose prompt `p` (n tokens) the ladder just resumed: save the
 * checkpoint at `cut`, commit prompt + gen, record the think-elided projection
 * and publish the warm history, which is the KV:
 * hist[0, ct_keep) ++ p[ct_prompt_from, n) ++ gen. */
static void finish_turn(pn_batch_sched *bs, int s, const uint32_t *p, int n,
                        int cut, const uint32_t *gen, int n_gen)
{
    bs_slot *sl = &bs->slots[s];
    int keep = sl->ct_keep, from = sl->ct_prompt_from;
    sl->state->pos = keep + (cut - from);
    CHECK(bs_turn_ckpt_capture(bs, s, "A", p, cut) == 0,
          "turn checkpoint saved at the turn start");
    int kv_n = keep + (n - from) + n_gen;
    uint32_t *kv = malloc((size_t)kv_n * sizeof(uint32_t));
    memcpy(kv, sl->hist, (size_t)keep * sizeof(uint32_t));
    memcpy(kv + keep, p + from, (size_t)(n - from) * sizeof(uint32_t));
    memcpy(kv + keep + (n - from), gen, (size_t)n_gen * sizeof(uint32_t));
    harness_seed_committed(sl, p, n, gen, n_gen);
    const uint32_t rerender[] = { 500, 501, 502, 503 };
    bs_slot_note_visible(bs, s, p, n, 1, NULL, 0, rerender, 4);
    bs_slot_publish_projections(sl, kv_n);
    harness_publish_warm(sl, "A", kv, kv_n);
    free(kv);
    bs_prefill_abort(bs, s);
}

/* Slot 0 committed turn 1 for "A" from a cold prefill: system snapshot,
 * checkpoint at 17 whose KV is exactly the request's tokens. */
static pn_batch_sched *sched_after_think_turn(pn_model *model)
{
    pn_batch_sched *bs = harness_sched(model, 2);
    bs->turn_ckpt_on = true;
    uint32_t p1[T1_N];
    turn1_prompt(p1);
    bs_resume_outcome out;
    int s = bs_slot_acquire_biased(bs, "A");
    bs_prefill_resume_think(bs, s, "A", p1, T1_N, SYS_LEN,
                            PN_THINK_HISTORY_KEEP, &out);
    bs->slots[s].state->pos = SYS_LEN;
    CHECK(out.capture_sys == 1 &&
          bs_prefill_capture_sys(bs, s, "A", p1, SYS_LEN) == 0,
          "turn 1: system snapshot");
    finish_turn(bs, s, p1, T1_N, T1_CUT, T1_GEN, T1_GEN_N);
    return bs;
}

/* Request 2 with turn 1's think block dropped: turn 1's messages, its tool
 * call as content (the think-elided projection, 23 tokens), a 6-token tool
 * result and a 3-token opener. */
#define R2_N   32
#define R2_CUT 29
static void request2(uint32_t *r)
{
    turn1_prompt(r);
    const uint32_t call[] = { 500, 501, 502, 503 };
    memcpy(r + 19, call, sizeof(call));
    for (int i = 23; i < R2_CUT; i++)
        r[i] = (uint32_t)(3000 + i);
    r[29] = IM_START;
    r[30] = ASSISTANT;
    r[31] = THINK_OPEN;
}

static bs_resume_outcome resume(pn_batch_sched *bs, const uint32_t *tokens,
                                int n, pn_think_history th)
{
    bs_resume_outcome out;
    int s = bs_slot_acquire_biased(bs, "A");
    bs_prefill_resume_think(bs, s, "A", tokens, n, SYS_LEN, th, &out);
    bs_prefill_abort(bs, s);
    return out;
}

static void test_without_flag_projection_still_hits(void)
{
    printf("test: without the flag, rung 1b still hits\n");
    pn_model model = qwen35_model();
    pn_batch_sched *bs = sched_after_think_turn(&model);
    uint32_t r2[R2_N];
    request2(r2);
    CHECK(memcmp(bs->slots[0].vis, r2, 23 * sizeof(uint32_t)) == 0,
          "request 2 is the think-elided projection + its tail");
    bs_resume_outcome out = resume(bs, r2, R2_N, PN_THINK_HISTORY_KEEP);
    CHECK(out.kind == PN_RESUME_PROJECTION && out.delta_start == 23,
          "projection resume at 23 over the 28-token KV");
    bs_destroy(bs);
}

static void test_drop_last_resumes_at_turn_checkpoint(void)
{
    printf("test: drop_last resumes at rung 1c, KV = the request\n");
    pn_model model = qwen35_model();
    pn_batch_sched *bs = sched_after_think_turn(&model);
    uint32_t r2[R2_N];
    request2(r2);
    bs_resume_outcome out;
    int s = bs_slot_acquire_biased(bs, "A");
    bs_prefill_resume_think(bs, s, "A", r2, R2_N, SYS_LEN,
                            PN_THINK_HISTORY_DROP_LAST, &out);
    bs_slot *sl = &bs->slots[s];
    CHECK(out.kind == PN_RESUME_TURN_CKPT && out.delta_start == T1_CUT &&
          out.capture_sys == 0, "resume at 17, reported as turn_ckpt");
    CHECK((int)sl->state->pos == T1_CUT && sl->hist_n == T1_CUT &&
          memcmp(sl->hist, r2, T1_CUT * sizeof(uint32_t)) == 0,
          "state and history are the request's first 17 tokens");
    CHECK(sl->ct_keep == T1_CUT && sl->ct_prompt_from == T1_CUT,
          "the log joins the request at 17");
    CHECK((int)sl->state->pos + (R2_N - out.delta_start) == R2_N,
          "KV length after the prefill == the rendered request (32)");
    CHECK(sl->vis_n == 0 && sl->vis_think_n == 0,
          "the dropped turn's projections are gone");
    bs_prefill_abort(bs, s);

    /* The turn saves its own checkpoint and the chain continues. */
    s = bs_slot_acquire_biased(bs, "A");
    bs_prefill_resume_think(bs, s, "A", r2, R2_N, SYS_LEN,
                            PN_THINK_HISTORY_DROP_LAST, &out);
    const uint32_t gen2[] = { 910, THINK_CLOSE, 600, 601 };
    finish_turn(bs, s, r2, R2_N, R2_CUT, gen2, 4);
    CHECK(sl->ckpt.kv_n == R2_CUT && sl->ckpt.client_n == R2_CUT &&
          memcmp(sl->ckpt.kv, r2, R2_CUT * sizeof(uint32_t)) == 0,
          "turn 2's checkpoint KV is exactly the request's 29 tokens");
    uint32_t r3[40];
    memcpy(r3, r2, R2_CUT * sizeof(uint32_t));
    for (int i = R2_CUT; i < 40; i++)
        r3[i] = (uint32_t)(4000 + i);
    out = resume(bs, r3, 40, PN_THINK_HISTORY_DROP_LAST);
    CHECK(out.kind == PN_RESUME_TURN_CKPT && out.delta_start == R2_CUT,
          "request 3 resumes at turn 2's checkpoint (29)");
    bs_destroy(bs);
}

static void test_drop_last_skips_exact_append(void)
{
    printf("test: drop_last skips rung 1 (exact append)\n");
    pn_model model = qwen35_model();
    pn_batch_sched *bs = sched_after_think_turn(&model);
    uint32_t r[T1_KV_N + 4];
    memcpy(r, bs->slots[0].hist, T1_KV_N * sizeof(uint32_t));
    for (int i = T1_KV_N; i < T1_KV_N + 4; i++)
        r[i] = (uint32_t)(4000 + i);
    bs_resume_outcome keep = resume(bs, r, T1_KV_N + 4, PN_THINK_HISTORY_KEEP);
    CHECK(keep.kind == PN_RESUME_EXACT && keep.delta_start == T1_KV_N,
          "control: no flag, rung 1 resumes at 28");
    bs_destroy(bs);
    bs = sched_after_think_turn(&model);
    bs_resume_outcome dl = resume(bs, r, T1_KV_N + 4,
                                  PN_THINK_HISTORY_DROP_LAST);
    CHECK(dl.kind == PN_RESUME_TURN_CKPT && dl.delta_start == T1_CUT,
          "drop_last: the turn checkpoint instead (17)");
    bs_destroy(bs);
}

static void test_drop_last_no_match_is_sys1(void)
{
    printf("test: drop_last with no checkpoint match resumes at SYS1\n");
    pn_model model = qwen35_model();
    pn_batch_sched *bs = sched_after_think_turn(&model);
    uint32_t r2[R2_N];
    request2(r2);
    r2[12] ^= 1u;   /* an earlier message changed */
    bs_resume_outcome out = resume(bs, r2, R2_N, PN_THINK_HISTORY_DROP_LAST);
    CHECK(out.kind == PN_RESUME_SYS_PREFIX && out.delta_start == SYS_LEN,
          "system prefix, reported as sys_prefix");
    CHECK(!bs->slots[0].ckpt.valid && bs->slots[0].hist_n == 0,
          "checkpoint and history dropped, rebuilt from this turn");
    bs_destroy(bs);
}

/* Turn 2 under no flag resumes by projection (the KV keeps turn 1's think
 * block) and checkpoints at 29: that checkpoint's KV is 34 tokens. */
static void projection_turn2(pn_batch_sched *bs, const uint32_t *r2)
{
    bs_resume_outcome out;
    int s = bs_slot_acquire_biased(bs, "A");
    bs_prefill_resume_think(bs, s, "A", r2, R2_N, SYS_LEN,
                            PN_THINK_HISTORY_KEEP, &out);
    CHECK(out.kind == PN_RESUME_PROJECTION, "turn 2: projection resume");
    const uint32_t gen2[] = { 910, THINK_CLOSE, 600, 601 };
    finish_turn(bs, s, r2, R2_N, R2_CUT, gen2, 4);
    CHECK(bs->slots[s].ckpt.kv_n == 34 && bs->slots[s].ckpt.client_n == R2_CUT,
          "turn 2's checkpoint holds 34 KV tokens for 29 request tokens");
}

static void test_drop_last_refuses_checkpoint_over_kept_think(void)
{
    printf("test: drop_last refuses a checkpoint whose KV kept think blocks\n");
    pn_model model = qwen35_model();
    uint32_t r2[R2_N], r3[38];
    request2(r2);
    memcpy(r3, r2, R2_CUT * sizeof(uint32_t));
    for (int i = R2_CUT; i < 38; i++)
        r3[i] = (uint32_t)(5000 + i);
    pn_batch_sched *bs = sched_after_think_turn(&model);
    projection_turn2(bs, r2);
    bs->slots[0].vis_n = 0;   /* request 3 drops turn 2's reply */
    bs_resume_outcome keep = resume(bs, r3, 38, PN_THINK_HISTORY_KEEP);
    CHECK(keep.kind == PN_RESUME_TURN_CKPT && keep.delta_start == R2_CUT &&
          bs->slots[0].ct_keep == 34,
          "control: no flag, rung 1c resumes over the 34-token KV");
    bs_destroy(bs);
    bs = sched_after_think_turn(&model);
    projection_turn2(bs, r2);
    bs->slots[0].vis_n = 0;
    bs_resume_outcome dl = resume(bs, r3, 38, PN_THINK_HISTORY_DROP_LAST);
    CHECK(dl.kind == PN_RESUME_SYS_PREFIX && dl.delta_start == SYS_LEN,
          "drop_last: not KV = request there, so the system prefix");
    bs_destroy(bs);
}

static void test_drop_last_capability(void)
{
    printf("test: drop_last is served only with turn checkpoints carved\n");
    pn_model model = qwen35_model();
    pn_batch_sched *bs = harness_sched(&model, 1);
    CHECK(!bs_think_history_drop_last(NULL), "no scheduler: not served");
    CHECK(!bs_think_history_drop_last(bs), "no checkpoints carved: not served");
    bs->turn_ckpt_on = true;
    CHECK(bs_think_history_drop_last(bs), "checkpoints carved: served");
    bs_destroy(bs);
}

/* ---- single-user: the turn snapshot in the prefix cache ---- */

/* Stamp every SSM image of `st` with `seed` (stands in for a prefill). */
static void stamp_ssm(pn_state *st, float seed)
{
    pn_qwen35moe_state *s = (pn_qwen35moe_state *)st->kv_cache;
    for (int i = 0; i < s->n_ssm; i++) {
        s->ssm[i].conv_state[0] = seed;
        s->ssm[i].ssm_state[0] = seed + 0.5f;
    }
}

static bool ssm_is(const pn_state *st, float seed)
{
    const pn_qwen35moe_state *s = (const pn_qwen35moe_state *)st->kv_cache;
    for (int i = 0; i < s->n_ssm; i++)
        if (s->ssm[i].conv_state[0] != seed ||
            s->ssm[i].ssm_state[0] != seed + 0.5f)
            return false;
    return true;
}

/* Session "u" turn 1 on the single-user cache: system snapshot at 8, the
 * state stamped 1.0 at the turn start 17 (turn snapshot saved there when
 * `capture`), then 2.0 by the end of the turn; history = prompt + gen. */
static pn_prefix_cache *su_after_think_turn(pn_model *model, bool capture)
{
    pn_prefix_cache *c = pn_prefix_cache_new(model, 2);
    uint32_t p1[T1_N], hist[T1_KV_N];
    turn1_prompt(p1);
    pn_prefix_acquire_result out;
    pn_state *st = harness_acquire(c, "u", p1, T1_N, SYS_LEN, T1_N, true,
                                   &out);
    st->pos = SYS_LEN;
    CHECK(out.capture_sys &&
          pn_prefix_cache_capture_sys(c, "u", p1, SYS_LEN) == 0,
          "single-user turn 1: system snapshot");
    stamp_ssm(st, 1.0f);
    st->pos = T1_CUT;
    if (capture)
        CHECK(pn_prefix_cache_capture_turn(c, "u", p1, T1_CUT) == 0,
              "single-user turn 1: turn snapshot at 17");
    stamp_ssm(st, 2.0f);
    memcpy(hist, p1, sizeof(p1));
    memcpy(hist + T1_N, T1_GEN, sizeof(T1_GEN));
    st->pos = T1_KV_N;
    CHECK(pn_prefix_cache_commit(c, "u", hist, T1_KV_N) == 0,
          "single-user turn 1: commit");
    return c;
}

/* request 2's as_sent ceiling: its first think block (none before the
 * forced generation prefill at 31). */
static int r2_ceiling(const uint32_t *r2)
{
    return pn_think_history_first_think(r2, R2_N - 1, (int)THINK_OPEN);
}

static void test_single_user_drop_last_restores_turn_snapshot(void)
{
    printf("test: single-user drop_last restores the turn snapshot\n");
    pn_model model = qwen35_model();
    pn_prefix_cache *c = su_after_think_turn(&model, true);
    uint32_t r2[R2_N];
    request2(r2);
    pn_prefix_acquire_result out;
    pn_state *st = harness_acquire(c, "u", r2, R2_N, SYS_LEN, r2_ceiling(r2),
                                   true, &out);
    CHECK(out.kind == PN_RESUME_TURN_CKPT && out.delta_start == T1_CUT,
          "resume at 17, reported as turn_ckpt");
    CHECK(st && st->pos == T1_CUT && ssm_is(st, 1.0f),
          "state back at 17 with the turn start's recurrent images");
    CHECK(st->pos + (R2_N - out.delta_start) == R2_N,
          "KV length after the prefill == the rendered request (32)");
    pn_prefix_cache_free(c);
}

static void test_single_user_without_flag_no_restore(void)
{
    printf("test: single-user without drop_last never restores the snapshot\n");
    pn_model model = qwen35_model();
    pn_prefix_cache *c = su_after_think_turn(&model, true);
    uint32_t r2[R2_N];
    request2(r2);
    pn_prefix_acquire_result out;
    pn_state *st = harness_acquire(c, "u", r2, R2_N, SYS_LEN, R2_N, false,
                                   &out);
    CHECK(out.kind == PN_RESUME_SYS_PREFIX && out.delta_start == SYS_LEN &&
          st->pos == SYS_LEN, "no flag: the system prefix, as before");
    /* That resume rewrote rows past 8: the snapshot is gone. */
    st->pos = T1_KV_N;
    pn_prefix_cache_commit(c, "u", r2, R2_N);
    st->pos = R2_N;
    harness_acquire(c, "u", r2, R2_N, SYS_LEN, r2_ceiling(r2), true, &out);
    CHECK(out.kind != PN_RESUME_TURN_CKPT,
          "a snapshot below the resume point is dropped");
    pn_prefix_cache_free(c);
}

static void test_single_user_no_snapshot_is_as_sent(void)
{
    printf("test: single-user drop_last without a snapshot is as_sent\n");
    pn_model model = qwen35_model();
    pn_prefix_cache *c = su_after_think_turn(&model, false);
    uint32_t r2[R2_N];
    request2(r2);
    pn_prefix_acquire_result out;
    harness_acquire(c, "u", r2, R2_N, SYS_LEN, r2_ceiling(r2), true, &out);
    CHECK(out.kind == PN_RESUME_SYS_PREFIX && out.delta_start == SYS_LEN,
          "system prefix, reported as sys_prefix");
    pn_prefix_cache_free(c);

    c = su_after_think_turn(&model, true);
    r2[12] ^= 1u;   /* an earlier message changed: the snapshot does not fit */
    harness_acquire(c, "u", r2, R2_N, SYS_LEN, r2_ceiling(r2), true, &out);
    CHECK(out.kind == PN_RESUME_SYS_PREFIX && out.delta_start == SYS_LEN,
          "a snapshot the request does not start with: system prefix");
    pn_prefix_cache_free(c);
}

static void test_recurrent_snapshot_is_not_exportable(void)
{
    printf("test: a recurrent snapshot holds no KV rows and is not exported\n");
    pn_model model = qwen35_model();
    pn_state *st = pn_state_new(&model);
    void *snap = pn_qwen35moe_state_snapshot_recurrent(st->kv_cache, 17);
    unsigned char *bytes = NULL;
    size_t len = 0;
    CHECK(snap && pn_qwen35moe_state_snapshot_pos(snap) == 17,
          "captured at 17");
    CHECK(pn_qwen35moe_snap_serialize(snap, 1, &bytes, &len) == -1 && !bytes,
          "serialize refuses it");
    pn_qwen35moe_state_snapshot_free(snap);
    pn_state_free(st);
}

/* ---- Tenstorrent: the band checkpoint's ceiling ---- */

static void test_tt_ceiling_on_think_history_turns(void)
{
    printf("test: TT checkpoint sits at or below turn_start on think_history turns\n");
    CHECK(pn_think_history_tt_ceiling(PN_THINK_HISTORY_KEEP, 1023, 1026) == 1026,
          "no flag: the end of the resent history, as before");
    CHECK(pn_think_history_tt_ceiling(PN_THINK_HISTORY_DROP_LAST, 1023, 1026) ==
          1023 && pn_think_history_tt_ceiling(PN_THINK_HISTORY_AS_SENT, 1023,
                                              1026) == 1023,
          "think_history: turn_start");
    /* The case that matters: the header pushes the history past a band. */
    int keep_at = pn_qwen35_ckpt_band_at(0, 1027, 1026);
    int drop_at = pn_qwen35_ckpt_band_at(
        0, 1027, pn_think_history_tt_ceiling(PN_THINK_HISTORY_DROP_LAST, 1023, 1026));
    CHECK(keep_at == 1024,
          "no flag: band 1024, above the turn start a drop_last request keeps");
    CHECK(drop_at == 992, "think_history: band 992, resumable");
    int worst = 0;
    bool below = true;
    for (int ts = 33; ts < 4096; ts++) {
        int at = pn_qwen35_ckpt_band_at(
            0, ts + 4, pn_think_history_tt_ceiling(PN_THINK_HISTORY_DROP_LAST, ts,
                                                   ts + 3));
        below = below && at <= ts && at > 0;
        if (ts - at > worst)
            worst = ts - at;
    }
    CHECK(below && worst == 31,
          "every turn_start: the band is at or below it, at most 31 tokens back");
}

int main(void)
{
    test_without_flag_projection_still_hits();
    test_drop_last_resumes_at_turn_checkpoint();
    test_drop_last_skips_exact_append();
    test_drop_last_no_match_is_sys1();
    test_drop_last_refuses_checkpoint_over_kept_think();
    test_drop_last_capability();
    test_single_user_drop_last_restores_turn_snapshot();
    test_single_user_without_flag_no_restore();
    test_single_user_no_snapshot_is_as_sent();
    test_recurrent_snapshot_is_not_exportable();
    test_tt_ceiling_on_think_history_turns();
    if (failures) {
        printf("=== %d CHECK(s) FAILED ===\n", failures);
        return 1;
    }
    printf("=== all tests passed ===\n");
    return 0;
}
