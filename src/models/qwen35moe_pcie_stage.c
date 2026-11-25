/* SPDX-License-Identifier: AGPL-3.0-or-later */
/* Copyright (C) 2026 David (Corey) Herrera */
/**
 * @file   qwen35moe_pcie_stage.c
 * @brief  Pinned / async / peer / event-chained staging of a two-stage decode step's per-step
 *         copies. See qwen35moe_pcie_stage.h for the levers and the ordering rules.
 *
 * Red build (-DPN_PCIE_RED, never in a release): PN_PCIE_RED=<arm> breaks one ordering rule
 * so the exactness gates must fail. Arms: `hop_wait` (stage 1's uploads do not wait for the
 * hop's device-to-host copy), `d2h_order` (the hop's ready event is recorded before its copy
 * is queued), `peer` (the peer copy to rank 1 is skipped), `reuse` (the input rows are
 * handed back for rewriting without waiting for the uploads that read them).
 *
 * @status Default on (each flag unset = on).
 * @issues None
 * @todo   None
 */
#include "qwen35moe_pcie_stage.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

/* On by default: unset, "" or "1" is on, anything else (e.g. "0") turns the lever off. */
int qw_env_flag_default_on(const char *v)
{
    return !v || v[0] == 0 || (v[0] == '1' && v[1] == 0);
}

#define qw_pcie_flag_on qw_env_flag_default_on

qw_pcie_flags qw_pcie_flags_parse(const char *pin, const char *fanout, const char *hop_event,
                                  const char *pos)
{
    qw_pcie_flags f = { qw_pcie_flag_on(pin), qw_pcie_flag_on(fanout),
                        qw_pcie_flag_on(hop_event), qw_pcie_flag_on(pos) };
    return f;
}

qw_pcie_flags qw_pcie_flags_get(void)
{
    static int latched;
    static qw_pcie_flags f;
    if (!latched) {
        f = qw_pcie_flags_parse(getenv("QWEN35_PCIE_PIN"), getenv("QWEN35_PCIE_FANOUT"),
                                getenv("QWEN35_PCIE_HOP_EVENT"), getenv("QWEN35_PCIE_POS"));
        latched = 1;
    }
    return f;
}

int qw_pcie_flags_any(qw_pcie_flags f)
{
    return f.pin || f.fanout || f.hop_event || f.pos;
}

#ifdef USE_CUDA

#define QW_PCIE_DIES 4

/** A host buffer the dies read with async copies, and the events behind those reads. */
typedef struct {
    void *host;
    fc_event *read[QW_PCIE_DIES];   /* recorded on die d's stream after its read */
    unsigned armed;                 /* bit d: read[d] was recorded since the last wait */
} qw_pcie_host_buf;

struct qw_pcie_stage {
    fc_cuda_ctx *ctx[QW_PCIE_DIES];
    qw_pcie_flags f;
    int pinned;
    size_t rows_cap, small_cap;
    qw_pcie_host_buf in, hop, small[QW_PCIE_DIES];
    fc_event *hop_ready;                    /* on ctx[0]: the hop's rows are on the host */
    fc_event *fence[QW_PCIE_DIES];          /* on rank 1 of a pair: its work so far */
    fc_event *peer_done[QW_PCIE_DIES];      /* on rank 0 of a pair: its peer copy */
};

#ifdef PN_PCIE_RED
static int qw_pcie_red(const char *arm)
{
    const char *v = getenv("PN_PCIE_RED");
    return v && strcmp(v, arm) == 0;
}
#else
#define qw_pcie_red(arm) 0
#endif

static void *qw_pcie_host_alloc(qw_pcie_stage *st, size_t bytes)
{
    void *p = st->pinned ? fc_host_alloc(st->ctx[0], bytes) : malloc(bytes);
    if (p)
        memset(p, 0, bytes);
    return p;
}

static void qw_pcie_host_release(qw_pcie_stage *st, void *p)
{
    if (st->pinned)
        fc_host_free(st->ctx[0], p);
    else
        free(p);
}

/** Wait until every recorded device read of b has completed. */
static int qw_pcie_wait_reads(qw_pcie_host_buf *b)
{
    for (int d = 0; d < QW_PCIE_DIES; d++)
        if ((b->armed >> d) & 1u)
            if (fc_event_query(b->read[d]) != 0 && fc_event_sync(b->read[d]) != 0)
                return -1;
    b->armed = 0;
    return 0;
}

/** Record that die d's stream has queued a read of b. */
static int qw_pcie_mark_read(qw_pcie_stage *st, qw_pcie_host_buf *b, int d)
{
    if (fc_event_record_ctx(st->ctx[d], b->read[d]) != 0)
        return -1;
    b->armed |= 1u << d;
    return 0;
}

/* Rows in `b` to both ranks of the pair at `base` (dst0 on ctx[base], dst1 on
 * ctx[base + 1]). Each rank's stream first waits on `after` when given. */
static int qw_pcie_pair_fan(qw_pcie_stage *st, int base, qw_pcie_host_buf *b, fc_buffer *dst0,
                            fc_buffer *dst1, size_t bytes, fc_event *after)
{
    fc_cuda_ctx *r0 = st->ctx[base], *r1 = st->ctx[base + 1];

    if (after && !qw_pcie_red("hop_wait") && fc_event_wait_ctx(r0, after) != 0)
        return -1;
    if (fc_buffer_from_cpu_n_async(dst0, b->host, bytes) != 0 ||
        qw_pcie_mark_read(st, b, base) != 0)
        return -1;
    if (st->f.fanout) {
        /* Rank 1's earlier work may still read dst1: queue the write behind it. */
        if (fc_event_record_ctx(r1, st->fence[base + 1]) != 0 ||
            fc_event_wait_ctx(r0, st->fence[base + 1]) != 0)
            return -1;
        if (!qw_pcie_red("peer") && fc_buffer_copy_peer_async(dst1, dst0, bytes, r0) != 0)
            return -1;
        return fc_event_record_ctx(r0, st->peer_done[base]) == 0 &&
               fc_event_wait_ctx(r1, st->peer_done[base]) == 0 ? 0 : -1;
    }
    if (after && !qw_pcie_red("hop_wait") && fc_event_wait_ctx(r1, after) != 0)
        return -1;
    if (fc_buffer_from_cpu_n_async(dst1, b->host, bytes) != 0)
        return -1;
    return qw_pcie_mark_read(st, b, base + 1);
}

static int qw_pcie_events_create(qw_pcie_stage *st, qw_pcie_host_buf *b)
{
    for (int d = 0; d < QW_PCIE_DIES; d++)
        if (!(b->read[d] = fc_event_create_ctx(st->ctx[d])))
            return -1;
    return 0;
}

static void qw_pcie_events_free(qw_pcie_host_buf *b)
{
    for (int d = 0; d < QW_PCIE_DIES; d++)
        fc_event_free(b->read[d]);
}

qw_pcie_stage *qw_pcie_stage_create(fc_cuda_ctx *const ctx[4], size_t rows_bytes,
                                    size_t small_bytes, qw_pcie_flags f)
{
    qw_pcie_stage *st = calloc(1, sizeof(*st));
    if (!st)
        return NULL;
    for (int d = 0; d < QW_PCIE_DIES; d++)
        st->ctx[d] = ctx[d];
    st->f = f;
    st->rows_cap = rows_bytes;
    st->small_cap = f.pos ? small_bytes : 0;
    /* Pinned only on request (and only if the driver gives it): the other levers alone
     * stage through pageable memory, which the driver copies before an upload returns. */
    void *probe = f.pin ? fc_host_alloc(ctx[0], 64) : NULL;
    st->pinned = probe != NULL;
    if (probe)
        fc_host_free(ctx[0], probe);
    if (f.pin && !st->pinned)
        fprintf(stderr, "[qwen35moe] QWEN35_PCIE_PIN: no pinned host memory, staging pageable\n");

    st->in.host = qw_pcie_host_alloc(st, rows_bytes);
    st->hop.host = qw_pcie_host_alloc(st, rows_bytes);
    int ok = st->in.host && st->hop.host && qw_pcie_events_create(st, &st->in) == 0 &&
             qw_pcie_events_create(st, &st->hop) == 0 &&
             (st->hop_ready = fc_event_create_ctx(ctx[0])) != NULL;
    for (int d = 0; ok && d < QW_PCIE_DIES; d++) {
        st->fence[d] = fc_event_create_ctx(ctx[d]);
        st->peer_done[d] = fc_event_create_ctx(ctx[d]);
        ok = st->fence[d] && st->peer_done[d];
        if (ok && st->small_cap) {
            st->small[d].host = qw_pcie_host_alloc(st, st->small_cap);
            ok = st->small[d].host && qw_pcie_events_create(st, &st->small[d]) == 0;
        }
    }
    if (!ok) {
        fprintf(stderr, "[qwen35moe] QWEN35_PCIE: staging setup failed; legacy copies\n");
        qw_pcie_stage_free(st);
        return NULL;
    }
    return st;
}

void qw_pcie_stage_free(qw_pcie_stage *st)
{
    if (!st)
        return;
    qw_pcie_host_release(st, st->in.host);
    qw_pcie_host_release(st, st->hop.host);
    qw_pcie_events_free(&st->in);
    qw_pcie_events_free(&st->hop);
    fc_event_free(st->hop_ready);
    for (int d = 0; d < QW_PCIE_DIES; d++) {
        fc_event_free(st->fence[d]);
        fc_event_free(st->peer_done[d]);
        qw_pcie_host_release(st, st->small[d].host);
        qw_pcie_events_free(&st->small[d]);
    }
    free(st);
}

size_t qw_pcie_stage_rows_cap(const qw_pcie_stage *st)
{
    return st ? st->rows_cap : 0;
}

qw_pcie_flags qw_pcie_stage_flags(const qw_pcie_stage *st)
{
    qw_pcie_flags none = { 0, 0, 0, 0 };
    return st ? st->f : none;
}

int qw_pcie_stage_pinned(const qw_pcie_stage *st)
{
    return st ? st->pinned : 0;
}

float *qw_pcie_stage_input_rows(qw_pcie_stage *st)
{
    if (qw_pcie_red("reuse"))
        return st->in.host;
    return qw_pcie_wait_reads(&st->in) == 0 ? st->in.host : NULL;
}

int qw_pcie_stage_fan_in(qw_pcie_stage *st, fc_buffer *dst0, fc_buffer *dst1, size_t bytes)
{
    if (bytes > st->rows_cap)
        return -1;
    return qw_pcie_pair_fan(st, 0, &st->in, dst0, dst1, bytes, NULL);
}

int qw_pcie_stage_hop(qw_pcie_stage *st, fc_buffer *src0, fc_buffer *dst2, fc_buffer *dst3,
                      size_t bytes)
{
    if (bytes > st->rows_cap || qw_pcie_wait_reads(&st->hop) != 0)
        return -1;
    if (st->f.hop_event) {
        /* Into pinned memory the copy is queued; into pageable it returns once landed. */
        if (qw_pcie_red("d2h_order") && fc_event_record_ctx(st->ctx[0], st->hop_ready) != 0)
            return -1;
        if (fc_buffer_to_cpu_n_async(src0, st->hop.host, bytes) != 0)
            return -1;
        if (!qw_pcie_red("d2h_order") && fc_event_record_ctx(st->ctx[0], st->hop_ready) != 0)
            return -1;
        return qw_pcie_pair_fan(st, 2, &st->hop, dst2, dst3, bytes, st->hop_ready);
    }
    /* The caller drained ctx[0]; a blocking copy returns once the rows are on the host. */
    fc_buffer_to_cpu_n(src0, st->hop.host, bytes);
    return qw_pcie_pair_fan(st, 2, &st->hop, dst2, dst3, bytes, NULL);
}

int qw_pcie_stage_upload_small(qw_pcie_stage *st, int die, fc_buffer *dst, const void *src,
                               size_t bytes)
{
    if (die < 0 || die >= QW_PCIE_DIES || bytes > st->small_cap)
        return 1;
    qw_pcie_host_buf *b = &st->small[die];
    if (qw_pcie_wait_reads(b) != 0)
        return -1;
    memcpy(b->host, src, bytes);
    if (fc_buffer_from_cpu_n_async(dst, b->host, bytes) != 0)
        return -1;
    return qw_pcie_mark_read(st, b, die);
}

#endif /* USE_CUDA */
