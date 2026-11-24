/* SPDX-License-Identifier: AGPL-3.0-or-later */
/* Copyright (C) 2026 David (Corey) Herrera */
/*
 * test_qwen35_die_projection.c — the shipped K80 profile still fits, and the die
 * gate never projects less than the build really holds.
 *
 * Profile: qwen3.6-35b-a3b-q8 "k80-two-boards" — 4 K80 dies, 10/40 layers each,
 * ctx 65,536, 4 sessions, q4 KV, MTP on. Pantheon main ba71dd38 refused it:
 *
 *   [admit] REFUSE model load: die ordinal 3 has 12076 MiB free but its share
 *   needs ~11207 MiB (10/40 layers + KV@65536 + scratch)
 *
 * K-G1 (711a6da2) had made the Q8_0 layer weights honest (+501 MiB/die); the
 * same gate still left out the MTP head (907 MiB on the last die) and priced the
 * session rings through a 1536 MiB clamped scratch, so 7bcd8a7 passed only
 * because those errors cancelled. Measured on the bank with 7bcd8a7 (dies
 * 8-11, after 1k/8k/36k-token greedy requests): the last die holds 11547 MiB of
 * 12206 (~128 MiB of it the CUDA context, outside die_free), the first 9982.
 *
 * Bytes below are the GGUF's own (tools: blk.30-39 at the repacked size 9036 MiB,
 * blk.40 = the MTP head 907 MiB, output.weight 546 MiB repacked).
 */

#include "../src/models/qwen35moe_die_projection.h"

#include <stdio.h>

static int g_failed = 0;

#define CHECK(cond, msg) do { \
    if (!(cond)) { \
        fprintf(stderr, "  FAIL: %s (%s:%d)\n", (msg), __FILE__, __LINE__); \
        g_failed++; \
    } \
} while (0)

#define MIB(n) ((uint64_t)(n) << 20)

/* Measured on the K80 bank, Qwen3.6-35B-A3B Q8_0 (qwen3.6-35b-a3b-q8-0bd466629210). */
#define Q8_LAST_LAYERS_UPLOAD   9474518272ULL   /* blk.30-39, Q8_0 at 36 B / 32 */
#define Q8_FIRST_LAYERS_UPLOAD  9481873408ULL   /* blk.0-9 */
#define Q8_HEAD_UPLOAD          (248320ULL * 2304 + 2048 * 4)  /* + output_norm */
#define Q8_MTP_UPLOAD           950712320ULL     /* blk.40 */
#define K80_DIE_FREE            MIB(12076)      /* free once the context exists */
#define K80_MARGIN              MIB(256)        /* PN_DIE_LOAD_MARGIN_BYTES */
#define K80_LAST_RESIDENT       MIB(11547 - 128) /* 7bcd8a7 serving, minus context */
#define K80_FIRST_RESIDENT      MIB(9982 - 128)

static pn_qwen35_die_shape q36_q8_die(int last)
{
    pn_qwen35_die_shape s = {
        .layer_weight_bytes = last ? Q8_LAST_LAYERS_UPLOAD : Q8_FIRST_LAYERS_UPLOAD,
        .extra_weight_bytes = last ? Q8_HEAD_UPLOAD : 0,
        .mtp_weight_bytes = last ? Q8_MTP_UPLOAD : 0,
        .n_attn = last ? 3 : 2,          /* attention every 4th layer: 31,35,39 / 3,7 */
        .n_ssm = last ? 7 : 8,
        .session_rows = 4,
        .max_seq = 65536,
        .kv_q4 = 1,
        .n_head_kv = 2, .head_dim = 256,
        .ssm_conv = 4, .conv_dim = 8192, .ssm_state = 128, .ssm_dt_rank = 32,
        .scratch_bytes = MIB(512),       /* PN_ADMIT_DIE_SCRATCH_BASE_BYTES */
    };
    return s;
}

static void test_session_block_matches_carve(void)
{
    pn_qwen35_die_shape s = q36_q8_die(1);
    /* Per attention layer: K+V rings 2 heads x 65536 x 144 B, + fp16 sink(8) and
     * window(256) shadows; per SSM layer 3x8192 f32 conv + 128x128x32 f32 state. */
    uint64_t attn = 2ULL * 2 * 65536 * 144 + 2ULL * 2 * (8 + 256) * 256 * 2;
    uint64_t ssm = 3ULL * 8192 * 4 + 128ULL * 128 * 32 * 4;
    CHECK(pn_qwen35_ssm_image_bytes(&s) == ssm, "SSM image bytes");
    CHECK(pn_qwen35_session_block_bytes(&s) == 3 * attn + 7 * ssm, "session block bytes");
    CHECK(pn_qwen35_checkpoint_row_bytes(&s) == 7 * ssm + 3ULL * 2 * 2 * (8 + 256) * 256 * 2,
          "checkpoint row bytes");
}

static void test_shipped_profile_fits(void)
{
    for (int last = 0; last <= 1; last++) {
        pn_qwen35_die_shape s = q36_q8_die(last);
        pn_qwen35_die_need n;
        pn_qwen35_die_project(&s, &n);
        uint64_t resident = last ? K80_LAST_RESIDENT : K80_FIRST_RESIDENT;
        printf("  die %s: need %llu MiB (weights %llu + sessions %llu + scratch %llu), "
               "served %llu, spare %lld MiB\n", last ? "last " : "first",
               (unsigned long long)(n.total >> 20), (unsigned long long)(n.weights >> 20),
               (unsigned long long)(n.sessions >> 20), (unsigned long long)(n.scratch >> 20),
               (unsigned long long)(resident >> 20),
               (long long)((int64_t)(K80_DIE_FREE - K80_MARGIN - n.total) >> 20));
        CHECK(n.total >= resident, "projection is below what the die really holds");
        CHECK(n.total + K80_MARGIN <= K80_DIE_FREE, "shipped 65,536 profile no longer fits");
    }
}

static void test_session_rows(void)
{
    pn_qwen35_die_shape s = q36_q8_die(1);
    pn_qwen35_die_need four, one, none;
    pn_qwen35_die_project(&s, &four);
    s.session_rows = 1;
    pn_qwen35_die_project(&s, &one);
    s.session_rows = 0;
    pn_qwen35_die_project(&s, &none);
    CHECK(four.sessions == 4 * one.sessions, "rows scale the session term");
    CHECK(none.sessions == one.sessions, "a single-user serve still charges one ring");
}

int main(void)
{
    printf("test_qwen35_die_projection\n");
    test_session_block_matches_carve();
    test_shipped_profile_fits();
    test_session_rows();
    if (g_failed) {
        fprintf(stderr, "test_qwen35_die_projection: %d failure(s)\n", g_failed);
        return 1;
    }
    printf("test_qwen35_die_projection: ok\n");
    return 0;
}
