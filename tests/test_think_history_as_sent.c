/* SPDX-License-Identifier: AGPL-3.0-or-later */
/* Copyright (C) 2026 David (Corey) Herrera */
/**
 * @file test_think_history_as_sent.c
 * @brief `think_history: "as_sent"` on the resume ladders (CPU, no model):
 *        after a think-bearing turn the request resumes at the system prefix
 *        and the KV holds exactly the request as rendered; without the flag the
 *        `vis` projection rung still hits.
 *
 * Batched (bs_prefill_resume_think over a hand-assembled scheduler): every
 * warm rung (1 exact, 1b projection, the aborted-prefill rung, 1c turn
 * checkpoint, 1d divergence checkpoint) is skipped and reported as such; the
 * slot's history, projections and checkpoint are rebuilt from the request.
 * Single-user (pn_prefix_cache_acquire_bounded): no tier resumes past the
 * request's first think block. The card work (CUDA/TT) is the bank gate.
 *
 * @status Complete
 * @issues None
 * @todo None
 */

#include "think_history_ladder_harness.h"

/* Turn 1: 8 system tokens, 9 conversation tokens, a 3-token assistant opener
 * ending in the forced "<think>" (turn_start 17, think prefill 1). */
#define SYS_LEN     8
#define T1_N        20
#define T1_CUT      17
#define THINK_OPEN  248068u
#define THINK_CLOSE 248069u

static const uint32_t SYS_A[SYS_LEN] = { 1, 2, 3, 4, 5, 6, 7, 8 };

static void turn1_prompt(uint32_t *p)
{
    memcpy(p, SYS_A, sizeof(SYS_A));
    for (int i = SYS_LEN; i < T1_CUT; i++)
        p[i] = (uint32_t)(1000 + i);
    p[17] = 151644u;      /* <|im_start|> */
    p[18] = 77091u;       /* assistant\n */
    p[19] = THINK_OPEN;   /* forced think prefill */
}

/* What turn 1 generated: a think block, its close, a 4-token reply. */
static const uint32_t T1_GEN[] = { 900, 901, 902, THINK_CLOSE, 500, 501, 502,
                                   503 };
#define T1_GEN_N ((int)(sizeof(T1_GEN) / sizeof(T1_GEN[0])))
#define T1_KV_N  (T1_N + T1_GEN_N)

/* A scheduler whose slot 0 committed turn 1 for session "A": system snapshot,
 * turn checkpoint at 17, both projections, warm history of 28 tokens. */
static pn_batch_sched *sched_after_think_turn(pn_model *model)
{
    pn_batch_sched *bs = harness_sched(model, 2);
    uint32_t p1[T1_N];
    turn1_prompt(p1);
    bs_resume_outcome out;
    int s = bs_slot_acquire_biased(bs, "A");
    bs_prefill_resume_think(bs, s, "A", p1, T1_N, SYS_LEN,
                            PN_THINK_HISTORY_KEEP, &out);
    CHECK(out.delta_start == 0 && out.capture_sys == 1 &&
          out.kind == PN_RESUME_COLD, "turn 1: cold, capture the system prefix");
    bs_slot *sl = &bs->slots[s];
    sl->state->pos = SYS_LEN;
    CHECK(bs_prefill_capture_sys(bs, s, "A", p1, SYS_LEN) == 0,
          "turn 1: system snapshot");
    sl->state->pos = T1_CUT;
    CHECK(bs_turn_ckpt_capture(bs, s, "A", p1, T1_CUT) == 0,
          "turn 1: turn checkpoint");
    sl->state->pos = T1_N;
    harness_seed_committed(sl, p1, T1_N, T1_GEN, T1_GEN_N);
    const uint32_t think_block[] = { THINK_OPEN, 950, THINK_CLOSE };
    const uint32_t rerender[] = { 500, 501, 502, 503 };
    bs_slot_note_visible(bs, s, p1, T1_N, 1, think_block, 3, rerender, 4);
    bs_slot_publish_projections(sl, sl->ct_n);
    harness_publish_warm(sl, "A", sl->ct, T1_KV_N);
    bs_prefill_abort(bs, s);
    return bs;
}

/* Request 2 as a reasoning-dropping client sends it: the think-elided
 * projection (23 tokens) + a 6-token tool result + a 3-token opener. */
#define R2_N 32
static void request2(const pn_batch_sched *bs, uint32_t *r)
{
    memcpy(r, bs->slots[0].vis, 23 * sizeof(uint32_t));
    for (int i = 23; i < R2_N; i++)
        r[i] = (uint32_t)(3000 + i);
}

/* Run the ladder for `tokens` on session "A" and release the slot. */
static bs_resume_outcome resume(pn_batch_sched *bs, const uint32_t *tokens,
                                int n, pn_think_history th)
{
    bs_resume_outcome out;
    int s = bs_slot_acquire_biased(bs, "A");
    bs_prefill_resume_think(bs, s, "A", tokens, n, SYS_LEN, th, &out);
    bs_prefill_abort(bs, s);
    return out;
}

static pn_model qwen35_model(void)
{
    pn_model m;
    memset(&m, 0, sizeof(m));
    m.config.arch = PN_ARCH_QWEN35MOE;
    return m;
}

static void test_without_flag_projection_still_hits(void)
{
    printf("test: without the flag, rung 1b (vis) still hits\n");
    pn_model model = qwen35_model();
    pn_batch_sched *bs = sched_after_think_turn(&model);
    uint32_t r2[R2_N];
    request2(bs, r2);
    bs_resume_outcome out = resume(bs, r2, R2_N, PN_THINK_HISTORY_KEEP);
    CHECK(out.kind == PN_RESUME_PROJECTION && out.delta_start == 23,
          "projection resume at 23, reported as projection");
    CHECK(bs->slots[0].ct_keep == T1_KV_N,
          "the KV keeps turn 1's think block (28 rows before the tail)");
    bs_destroy(bs);
}

static void test_as_sent_resumes_at_sys1(void)
{
    printf("test: as_sent after a think-bearing turn resumes at SYS1\n");
    pn_model model = qwen35_model();
    pn_batch_sched *bs = sched_after_think_turn(&model);
    uint32_t r2[R2_N];
    request2(bs, r2);
    bs_resume_outcome out;
    int s = bs_slot_acquire_biased(bs, "A");
    bs_prefill_resume_think(bs, s, "A", r2, R2_N, SYS_LEN,
                            PN_THINK_HISTORY_AS_SENT, &out);
    bs_slot *sl = &bs->slots[s];
    CHECK(out.kind == PN_RESUME_SYS_PREFIX && out.delta_start == SYS_LEN,
          "resume at the system prefix, reported as sys_prefix");
    CHECK(out.capture_sys == 0, "the system snapshot is reused, not recaptured");
    CHECK((int)sl->state->pos == SYS_LEN && sl->ct_keep == 0 &&
          sl->ct_prompt_from == 0,
          "state at 8 and the log is the request: KV ends at the rendered 32");
    CHECK((int)sl->state->pos + (R2_N - out.delta_start) == R2_N,
          "KV length after the prefill == the rendered request");
    CHECK(sl->hist_n == 0 && sl->vis_n == 0 && sl->vis_think_n == 0 &&
          !sl->ckpt.valid,
          "history, projections and checkpoint are dropped for this turn");
    /* This turn's checkpoint is rebuilt from the request alone. */
    sl->state->pos = R2_N - 3;
    CHECK(bs_turn_ckpt_capture(bs, s, "A", r2, R2_N - 3) == 0 &&
          sl->ckpt.kv_n == sl->ckpt.client_n &&
          memcmp(sl->ckpt.kv, r2, (size_t)(R2_N - 3) * sizeof(uint32_t)) == 0,
          "the new checkpoint's KV is the request's own tokens");
    bs_prefill_abort(bs, s);
    bs_destroy(bs);
}

static void test_as_sent_skips_exact_append(void)
{
    printf("test: as_sent skips rung 1 (exact append)\n");
    pn_model model = qwen35_model();
    pn_batch_sched *bs = sched_after_think_turn(&model);
    uint32_t r[T1_KV_N + 4];
    memcpy(r, bs->slots[0].hist, T1_KV_N * sizeof(uint32_t));
    for (int i = T1_KV_N; i < T1_KV_N + 4; i++)
        r[i] = (uint32_t)(4000 + i);
    bs_resume_outcome keep = resume(bs, r, T1_KV_N + 4, PN_THINK_HISTORY_KEEP);
    CHECK(keep.kind == PN_RESUME_EXACT && keep.delta_start == T1_KV_N,
          "control: no flag, rung 1 resumes at 28 (exact)");
    bs_destroy(bs);
    bs = sched_after_think_turn(&model);
    bs_resume_outcome as = resume(bs, r, T1_KV_N + 4, PN_THINK_HISTORY_AS_SENT);
    CHECK(as.kind == PN_RESUME_SYS_PREFIX && as.delta_start == SYS_LEN,
          "as_sent: system prefix instead");
    bs_destroy(bs);
}

static void test_as_sent_skips_turn_checkpoint(void)
{
    printf("test: as_sent skips rung 1c (turn checkpoint)\n");
    pn_model model = qwen35_model();
    pn_batch_sched *bs = sched_after_think_turn(&model);
    uint32_t p1[T1_N], r[T1_CUT + 9];
    turn1_prompt(p1);
    memcpy(r, p1, T1_CUT * sizeof(uint32_t));
    for (int i = T1_CUT; i < T1_CUT + 9; i++)
        r[i] = (uint32_t)(5000 + i);
    bs_resume_outcome keep = resume(bs, r, T1_CUT + 9, PN_THINK_HISTORY_KEEP);
    CHECK(keep.kind == PN_RESUME_TURN_CKPT && keep.delta_start == T1_CUT,
          "control: no flag, rung 1c resumes at 17 (turn_ckpt)");
    bs_destroy(bs);
    bs = sched_after_think_turn(&model);
    bs_resume_outcome as = resume(bs, r, T1_CUT + 9, PN_THINK_HISTORY_AS_SENT);
    CHECK(as.kind == PN_RESUME_SYS_PREFIX && as.delta_start == SYS_LEN,
          "as_sent: system prefix instead");
    bs_destroy(bs);
}

/* Request 2 under no flag resumes by projection, prefills 5 tokens and is
 * aborted: the slot keeps that prefix (client 28, kv 32). */
static void abort_request2_midway(pn_batch_sched *bs, const uint32_t *r2)
{
    bs_resume_outcome out;
    int s = bs_slot_acquire_biased(bs, "A");
    bs_prefill_resume_think(bs, s, "A", r2, R2_N, SYS_LEN,
                            PN_THINK_HISTORY_KEEP, &out);
    bs->slots[s].state->pos = T1_KV_N + 5;
    CHECK(bs_partial_prefill_keep(bs, s, "A", r2, R2_N, SYS_LEN) == 0,
          "aborted prefill kept");
    bs_prefill_abort(bs, s);
}

static void test_as_sent_skips_partial_prefill(void)
{
    printf("test: as_sent skips the aborted-prefill rung\n");
    pn_model model = qwen35_model();
    pn_batch_sched *bs = sched_after_think_turn(&model);
    uint32_t r2[R2_N];
    request2(bs, r2);
    abort_request2_midway(bs, r2);
    bs_resume_outcome keep = resume(bs, r2, R2_N, PN_THINK_HISTORY_KEEP);
    CHECK(keep.kind == PN_RESUME_PARTIAL && keep.delta_start == 28,
          "control: no flag, the retry resumes at 28 (partial)");
    bs_destroy(bs);
    bs = sched_after_think_turn(&model);
    abort_request2_midway(bs, r2);
    bs_resume_outcome as = resume(bs, r2, R2_N, PN_THINK_HISTORY_AS_SENT);
    CHECK(as.kind == PN_RESUME_SYS_PREFIX && as.delta_start == SYS_LEN,
          "as_sent: system prefix instead");
    CHECK(!bs->slots[0].partial.valid, "the aborted-prefill record is dropped");
    bs_destroy(bs);
}

/* Give slot 0 a divergence checkpoint at client 12 (kv 12). */
static void add_div_checkpoint(pn_batch_sched *bs)
{
    bs_slot *sl = &bs->slots[0];
    uint32_t p1[T1_N];
    turn1_prompt(p1);
    bs->div_ckpt_on = true;
    int hist_pos = (int)sl->state->pos;
    sl->state->pos = 12;
    CHECK(bs_div_ckpt_capture(bs, 0, "A", p1, 12) == 0,
          "divergence checkpoint at 12");
    sl->state->pos = hist_pos;
}

static void test_as_sent_skips_div_checkpoint(void)
{
    printf("test: as_sent skips rung 1d (divergence checkpoint)\n");
    pn_model model = qwen35_model();
    uint32_t p1[T1_N], r[24];
    turn1_prompt(p1);
    memcpy(r, p1, 12 * sizeof(uint32_t));
    for (int i = 12; i < 24; i++)
        r[i] = (uint32_t)(6000 + i);
    pn_batch_sched *bs = sched_after_think_turn(&model);
    add_div_checkpoint(bs);
    bs_resume_outcome keep = resume(bs, r, 24, PN_THINK_HISTORY_KEEP);
    CHECK(keep.kind == PN_RESUME_DIV_CKPT && keep.delta_start == 12,
          "control: no flag, rung 1d resumes at 12 (div_ckpt)");
    bs_destroy(bs);
    bs = sched_after_think_turn(&model);
    add_div_checkpoint(bs);
    bs_resume_outcome as = resume(bs, r, 24, PN_THINK_HISTORY_AS_SENT);
    CHECK(as.kind == PN_RESUME_SYS_PREFIX && as.delta_start == SYS_LEN,
          "as_sent: system prefix instead");
    bs_destroy(bs);
}

static void test_as_sent_cold_without_snapshot(void)
{
    printf("test: as_sent with no system snapshot is a cold prefill\n");
    pn_model model = qwen35_model();
    pn_batch_sched *bs = sched_after_think_turn(&model);
    uint32_t r2[R2_N];
    request2(bs, r2);
    r2[7] = 77;   /* a changed system prompt: the snapshot does not apply */
    bs_resume_outcome out = resume(bs, r2, R2_N, PN_THINK_HISTORY_AS_SENT);
    CHECK(out.kind == PN_RESUME_COLD && out.delta_start == 0 &&
          out.capture_sys == 1, "cold, reported as cold, capture requested");
    bs_destroy(bs);
}

/* ---- single-user: pn_prefix_cache_acquire_bounded ---- */

/* A cache whose session "u" committed turn 1 (with its system snapshot when
 * the arch has one). */
static pn_prefix_cache *cache_after_think_turn(pn_model *model, uint32_t *hist)
{
    pn_prefix_cache *c = pn_prefix_cache_new(model, 2);
    uint32_t p1[T1_N];
    turn1_prompt(p1);
    pn_prefix_acquire_result out;
    pn_state *st = harness_acquire(c, "u", p1, T1_N, SYS_LEN, T1_N, false,
                                   &out);
    CHECK(st && out.delta_start == 0 && out.kind == PN_RESUME_COLD,
          "single-user turn 1: cold");
    st->pos = SYS_LEN;
    if (out.capture_sys)
        CHECK(pn_prefix_cache_capture_sys(c, "u", p1, SYS_LEN) == 0,
              "single-user turn 1: system snapshot");
    memcpy(hist, p1, sizeof(p1));
    memcpy(hist + T1_N, T1_GEN, sizeof(T1_GEN));
    st->pos = T1_KV_N;
    CHECK(pn_prefix_cache_commit(c, "u", hist, T1_KV_N) == 0,
          "single-user turn 1: commit");
    return c;
}

static void test_single_user_stops_at_first_think(void)
{
    printf("test: single-user as_sent never resumes past a think block\n");
    pn_model model = qwen35_model();
    uint32_t hist[T1_KV_N], r[T1_KV_N + 4];
    pn_prefix_cache *c = cache_after_think_turn(&model, hist);
    /* The request carries turn 1's think block byte for byte. */
    memcpy(r, hist, sizeof(hist));
    for (int i = T1_KV_N; i < T1_KV_N + 4; i++)
        r[i] = (uint32_t)(7000 + i);
    int n = T1_KV_N + 4;
    pn_prefix_acquire_result out;
    harness_acquire(c, "u", r, n, SYS_LEN, n, false, &out);
    CHECK(out.kind == PN_RESUME_EXACT && out.delta_start == T1_KV_N,
          "control: no ceiling, append resumes at 28 (exact)");
    pn_prefix_cache_commit(c, "u", hist, T1_KV_N);
    int ceiling = pn_think_history_first_think(r, n - 1, (int)THINK_OPEN);
    CHECK(ceiling == 19, "the request's first think block is at 19");
    harness_acquire(c, "u", r, n, SYS_LEN, ceiling, false, &out);
    CHECK(out.kind == PN_RESUME_SYS_PREFIX && out.delta_start == SYS_LEN,
          "as_sent: system prefix, not past the think block");
    pn_prefix_cache_free(c);
}

static void test_single_user_no_think_no_cost(void)
{
    printf("test: single-user as_sent with no think block costs nothing\n");
    pn_model model = qwen35_model();
    uint32_t hist[T1_KV_N], r[T1_KV_N + 4];
    pn_prefix_cache *c = cache_after_think_turn(&model, hist);
    memcpy(r, hist, sizeof(hist));
    for (int i = T1_KV_N; i < T1_KV_N + 4; i++)
        r[i] = (uint32_t)(7000 + i);
    int n = T1_KV_N + 4;
    /* A model with no think token: the ceiling is the whole history. */
    int ceiling = pn_think_history_first_think(r, n - 1, -1);
    pn_prefix_acquire_result out;
    harness_acquire(c, "u", r, n, SYS_LEN, ceiling, false, &out);
    CHECK(ceiling == n - 1 && out.kind == PN_RESUME_EXACT &&
          out.delta_start == T1_KV_N, "append resume as before");
    pn_prefix_cache_free(c);
}

static void test_single_user_lcp_rewind_clamped(void)
{
    printf("test: single-user as_sent clamps the plain-KV lcp rewind\n");
    pn_model model;
    memset(&model, 0, sizeof(model));
    model.config.arch = PN_ARCH_LLAMA;
    uint32_t hist[T1_KV_N], r[30];
    pn_prefix_cache *c = cache_after_think_turn(&model, hist);
    memcpy(r, hist, 24 * sizeof(uint32_t));   /* diverges at 24 */
    for (int i = 24; i < 30; i++)
        r[i] = (uint32_t)(8000 + i);
    pn_prefix_acquire_result out;
    harness_acquire(c, "u", r, 30, SYS_LEN, 30, false, &out);
    CHECK(out.kind == PN_RESUME_PARTIAL && out.delta_start == 24,
          "control: no ceiling, rewind to the lcp 24 (partial)");
    pn_prefix_cache_free(c);
    c = cache_after_think_turn(&model, hist);
    harness_acquire(c, "u", r, 30, SYS_LEN, 19, false, &out);
    CHECK(out.kind == PN_RESUME_PARTIAL && out.delta_start == 19,
          "as_sent: rewind stops at the think block (19)");
    pn_prefix_cache_free(c);
}

int main(void)
{
    test_without_flag_projection_still_hits();
    test_as_sent_resumes_at_sys1();
    test_as_sent_skips_exact_append();
    test_as_sent_skips_turn_checkpoint();
    test_as_sent_skips_partial_prefill();
    test_as_sent_skips_div_checkpoint();
    test_as_sent_cold_without_snapshot();
    test_single_user_stops_at_first_think();
    test_single_user_no_think_no_cost();
    test_single_user_lcp_rewind_clamped();
    if (failures) {
        printf("=== %d CHECK(s) FAILED ===\n", failures);
        return 1;
    }
    printf("=== all tests passed ===\n");
    return 0;
}
