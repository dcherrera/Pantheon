/* SPDX-License-Identifier: AGPL-3.0-or-later */
/* Copyright (C) 2026 David (Corey) Herrera */
/**
 * @file test_mtp_ring_clear.c
 * @brief The MTP head's KV ring must never hand a draft a row nothing wrote. Host-side, no
 *        GPU, no model.
 *
 * WHAT IT GUARDS
 *   2026-11-14 Xid 31 on a K80 host. The head keeps ONE
 *   q4 ring per replica, attends [0, pos], and its backlog resets on a session switch. A long
 *   resumed session reset at ~26k, above every row any session had written, so its draft
 *   read recycled cuMemAlloc bytes: a q4 scale that decodes as NaN poisons the softmax and
 *   every draft logit comes out NaN.
 *
 *   The test replays that on host buffers laid out exactly like the device ring (the
 *   pn_kvq4_store_row / pn_kvq4_load_row mirror of the kernels' sink / window / q4 choice):
 *   buffers filled with recycled bytes, cleared with the same pn_mtp_ring_clear and the
 *   same sizes mtp_alloc uses, one session writes a prefix, a second resumes far above it,
 *   and the second session's draft attends [0, pos]. The output must be finite, the
 *   session's own rows must read back as written, and the other session's rows must
 *   survive (the clear is at allocation only, never on a backlog reset).
 *
 * RED-FIRST
 *   `--no-clear` skips the clear: the ring as mtp_alloc left it before the fix. Its draft
 *   reads NaN, so its checks MUST fail; the arm inverts and passes only when they do.
 */

#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "../src/models/qwen35moe_kv_q4.h"
#include "../src/models/qwen35moe_mtp_ring.h"

static int failures = 0;
#define CHECK(cond, msg) do { \
    if (cond) { printf("  PASS: %s\n", msg); } \
    else { printf("  FAIL: %s (%s:%d)\n", msg, __FILE__, __LINE__); failures++; } \
} while (0)

static int no_clear = 0;

/* Qwen3.5/3.6 MoE head geometry; max_seq is not a multiple of the zero chunk, so the
 * doubling's short last copy is exercised. */
enum { N_HEAD_KV = 2, HEAD_DIM = 256, N_HEAD = 16, MAX_SEQ = 3001 };

typedef struct {
    uint8_t *buf[PN_MTP_RING_BUFFERS];
    size_t size[PN_MTP_RING_BUFFERS];
    long puts, copies;
} host_ring;

static int host_put(void *ops_ctx, void *buf, const void *src, size_t bytes) {
    ((host_ring *)ops_ctx)->puts++;
    memcpy(buf, src, bytes);
    return 0;
}

static int host_copy(void *ops_ctx, void *buf, size_t dst_off, size_t src_off, size_t bytes) {
    ((host_ring *)ops_ctx)->copies++;
    memcpy((uint8_t *)buf + dst_off, (uint8_t *)buf + src_off, bytes);
    return 0;
}

/* Allocate like mtp_alloc (same sizes) and fill with a "recycled" 16-bit pattern. */
static int ring_alloc(host_ring *r, uint16_t junk) {
    pn_mtp_ring_sizes rs = pn_mtp_ring_sizes_for(N_HEAD_KV, HEAD_DIM, MAX_SEQ);

    memset(r, 0, sizeof(*r));
    for (int i = 0; i < PN_MTP_RING_BUFFERS; i++) {
        r->size[i] = i < 2 ? rs.cache_bytes : i < 4 ? rs.sink_bytes : rs.win_bytes;
        r->buf[i] = malloc(r->size[i]);
        if (!r->buf[i])
            return -1;
        for (size_t b = 0; b + 1 < r->size[i]; b += 2)
            memcpy(r->buf[i] + b, &junk, sizeof(junk));
    }
    if (no_clear)
        return 0;
    pn_mtp_ring_fill_ops ops = { host_put, host_copy, r };
    return pn_mtp_ring_clear((void *const *)r->buf, &rs, &ops);
}

static void ring_free(host_ring *r) {
    for (int i = 0; i < PN_MTP_RING_BUFFERS; i++)
        free(r->buf[i]);
}

/* A deterministic finite K or V row for (session, position, head). */
static void make_row(float *dst, int session, int pos, int h, int is_v) {
    for (int e = 0; e < HEAD_DIM; e++)
        dst[e] = 0.5f * sinf(0.013f * (float)(pos + 1) * (float)(e + 1) + 0.7f * (float)session +
                             1.3f * (float)h + (is_v ? 2.1f : 0.0f));
}

/* mtp_rows_kv's append: K and V for positions [pos0, pos0 + n) of one session. */
static void ring_append(host_ring *r, int session, int pos0, int n) {
    float row[HEAD_DIM];

    for (int p = pos0; p < pos0 + n; p++) {
        for (int h = 0; h < N_HEAD_KV; h++) {
            make_row(row, session, p, h, 0);
            pn_kvq4_store_row(r->buf[0], (uint16_t *)r->buf[2], (uint16_t *)r->buf[4], h,
                              MAX_SEQ, p, row, HEAD_DIM, pn_kvq4_sink(), pn_kvq4_win());
            make_row(row, session, p, h, 1);
            pn_kvq4_store_row(r->buf[1], (uint16_t *)r->buf[3], (uint16_t *)r->buf[5], h,
                              MAX_SEQ, p, row, HEAD_DIM, pn_kvq4_sink(), pn_kvq4_win());
        }
    }
}

/* mtp_row_full's attention: one query at pos over keys [0, pos], every head. Returns 1 when
 * all outputs are finite. */
static int draft_attention_finite(const host_ring *r, int pos) {
    static float scores[MAX_SEQ];
    float q[HEAD_DIM], k[HEAD_DIM], v[HEAD_DIM], out[HEAD_DIM];
    const int groups = N_HEAD / N_HEAD_KV, sink = pn_kvq4_sink(), win = pn_kvq4_win();
    int finite = 1;

    for (int qh = 0; qh < N_HEAD; qh++) {
        const int h = qh / groups;
        float m = -3.0e38f, l = 0.0f;

        make_row(q, 99, pos, qh, 0);
        for (int s = 0; s <= pos; s++) {
            float dot = 0.0f;

            pn_kvq4_load_row(r->buf[0], (const uint16_t *)r->buf[2],
                             (const uint16_t *)r->buf[4], h, MAX_SEQ, s, pos, k, HEAD_DIM,
                             sink, win);
            for (int e = 0; e < HEAD_DIM; e++)
                dot += q[e] * k[e];
            scores[s] = dot / 16.0f;
            m = fmaxf(m, scores[s]);
        }
        memset(out, 0, sizeof(out));
        for (int s = 0; s <= pos; s++) {
            float p = expf(scores[s] - m);

            l += p;
            pn_kvq4_load_row(r->buf[1], (const uint16_t *)r->buf[3],
                             (const uint16_t *)r->buf[5], h, MAX_SEQ, s, pos, v, HEAD_DIM,
                             sink, win);
            for (int e = 0; e < HEAD_DIM; e++)
                out[e] += p * v[e];
        }
        for (int e = 0; e < HEAD_DIM; e++)
            if (!isfinite(out[e] / l))
                finite = 0;
    }
    return finite;
}

/* Max |loaded - written| over a session's rows at [pos0, pos0 + n), read as of cur_pos. */
static float rows_err(const host_ring *r, int session, int pos0, int n, int cur_pos) {
    float want[HEAD_DIM], got[HEAD_DIM], err = 0.0f;

    for (int p = pos0; p < pos0 + n; p++) {
        for (int h = 0; h < N_HEAD_KV; h++) {
            make_row(want, session, p, h, 0);
            pn_kvq4_load_row(r->buf[0], (const uint16_t *)r->buf[2],
                             (const uint16_t *)r->buf[4], h, MAX_SEQ, p, cur_pos, got,
                             HEAD_DIM, pn_kvq4_sink(), pn_kvq4_win());
            for (int e = 0; e < HEAD_DIM; e++)
                err = fmaxf(err, isfinite(got[e]) ? fabsf(got[e] - want[e]) : 3.0e38f);
        }
    }
    return err;
}

/* The incident: session A wrote [0, 600); session B resumes at 2900 (the backlog reset
 * there), appends [2900, 2910) and drafts at 2909, attending [0, 2909]. */
static void test_resumed_session_draft(uint16_t junk, const char *label) {
    host_ring r;
    char msg[160];

    printf("\n-- resumed-session draft over a ring of %s --\n", label);
    if (ring_alloc(&r, junk) != 0) {
        CHECK(0, "ring allocation + clear");
        return;
    }
    ring_append(&r, 1, 0, 600);
    ring_append(&r, 2, 2900, 10);
    snprintf(msg, sizeof(msg), "draft at 2909 attends [0, 2909] and stays finite (%s)", label);
    CHECK(draft_attention_finite(&r, 2909), msg);
    CHECK(rows_err(&r, 2, 2900, 10, 2909) < 0.1f, "the session's own rows read back as written");
    CHECK(rows_err(&r, 1, 0, 600, 2909) < 0.1f,
          "the other session's rows survive (no clear on reset)");
    ring_free(&r);
}

/* Every byte of every buffer is zero after the clear, with few uploads. */
static void test_clear_is_total(void) {
    host_ring r;
    int all_zero = 1;

    printf("\n-- the clear covers every byte --\n");
    if (ring_alloc(&r, 0xA5A5) != 0) {
        CHECK(0, "ring allocation + clear");
        return;
    }
    for (int i = 0; i < PN_MTP_RING_BUFFERS; i++)
        for (size_t b = 0; b < r.size[i]; b++)
            if (r.buf[i][b]) {
                all_zero = 0;
                break;
            }
    CHECK(all_zero, "all six ring buffers are zero");
    CHECK(no_clear || r.puts == PN_MTP_RING_BUFFERS, "one zero upload per buffer, the rest on-device");
    ring_free(&r);
}

static void test_sizes_and_arguments(void) {
    pn_mtp_ring_sizes rs = pn_mtp_ring_sizes_for(N_HEAD_KV, HEAD_DIM, 65536);
    pn_mtp_ring_sizes bad = pn_mtp_ring_sizes_for(0, HEAD_DIM, 65536);
    pn_mtp_ring_fill_ops ops = { host_put, host_copy, NULL };
    host_ring scratch = {0};

    printf("\n-- sizes and arguments --\n");
    ops.ops_ctx = &scratch;
    CHECK(rs.cache_bytes == (size_t)2 * 65536 * 144, "q4 ring: n_head_kv * max_seq * 144 B");
    CHECK(rs.sink_bytes == (size_t)2 * 256 * PN_KVQ4_SINKMAX * 2, "sink shadow size");
    CHECK(rs.win_bytes == (size_t)2 * 256 * PN_KVQ4_WMAX * 2, "window shadow size");
    CHECK(bad.cache_bytes == 0 && bad.sink_bytes == 0 && bad.win_bytes == 0,
          "a non-positive geometry sizes to zero");
    CHECK(pn_mtp_ring_zero_fill(NULL, 16, &ops) == -1, "NULL buffer is refused");
    CHECK(pn_mtp_ring_zero_fill(&scratch, 0, &ops) == 0 && scratch.puts == 0,
          "zero bytes is a no-op");
}

int main(int argc, char **argv) {
    for (int i = 1; i < argc; i++)
        if (strcmp(argv[i], "--no-clear") == 0)
            no_clear = 1;

    printf("=== MTP head ring: no draft reads an unwritten row ===\n");
    if (no_clear)
        printf("MODE: --no-clear (the ring as mtp_alloc left it before the fix — MUST fail)\n");

    test_resumed_session_draft(0xFFFF, "0xFFFF (fp16 NaN scales)");
    test_resumed_session_draft(0x7C00, "0x7C00 (fp16 inf scales)");
    test_clear_is_total();
    test_sizes_and_arguments();

    printf("\n%d check(s) failed\n", failures);
    if (no_clear) {
        printf(failures > 0 ? "[PASS] the no-clear arm failed, as it must\n"
                            : "[FAIL] the no-clear arm PASSED — the gate is vacuous\n");
        return failures > 0 ? 0 : 1;
    }
    printf(failures == 0 ? "[PASS]\n" : "[FAIL]\n");
    return failures == 0 ? 0 : 1;
}
