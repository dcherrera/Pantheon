/* SPDX-License-Identifier: AGPL-3.0-or-later */
/* Copyright (C) 2026 David (Corey) Herrera */
/**
 * @file pn_tt_dispatch.c
 * @brief STEP 3 (A) — the dispatch RECORDER and its host-side SHADOW. See pn_tt_dispatch.h.
 *
 * @status A1 observe mode; A1b model + host-side shadow verify.
 * @issues None
 * @todo A2 the puller's dry run; A3 live replay.
 *
 * WHY THE TRANSPORT AND NOT THE CALL SITES. Every host access to the card goes through one of
 * the runtime's three accessors (`noc`, `bulk_l1`, `bulk_dram`). Wrapping them sees ALL of a
 * token — including any write no queue API knows about — so "the token is queue traffic plus
 * these boundaries" is a measured statement, not an inventory of call sites. Every resident
 * queue sits at the same L1 layout (control 0x170000, head word +0x8, completion window
 * +0x10, ring +0x100, rotating parameter slots +0x9000, parameter cache +0xA000, group block
 * +0xF000), so the address alone classifies an access.
 *
 * THE MODEL (A1b). Two recorded tokens t0, t0+1 must be the same SHAPE (event for event: class,
 * accessor, core, size; the address too except ring slots). Then every queue word is one of:
 *   - a ring descriptor's TOKEN: `nonce ^ ((index+1) * 0x9E3779B9)` (ft_tt_queue_token_at),
 *     index = the queue's descriptor count, which advances by a fixed n_q a token;
 *   - a ring SLOT address: ring + (index & (capacity-1)) * 32;
 *   - anything else (head values, poll counts, parameter words, a4/a5 sequences, KV
 *     positions): LINEAR in the token, `w(t0) + (w(t0+1) - w(t0)) * k`.
 * Boundary accesses (DRAM, non-queue L1) are the host's: only their shape is modelled. The
 * SHADOW predicts every queue write and every awaited count of each later token from the
 * model alone and compares it, byte for byte, with what the host really did. RED
 * `PN_TT_RED=dispatch_skip_patch` drops the linear strides; it must mismatch.
 */
#ifdef USE_TT

#include "pn_tt_dispatch.h"

#include <execinfo.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "pn_tt_device_internal.h"
#include "pn_tt_dispatch_hoist.h"
#include "pn_tt_host_gap.h"
#include "pn_tt_precondition.h"
#include "../../core/bench.h"
#include "../../core/sample_profiler.h"

#include "tt/ft_tt_op_l1_layout.h"
#include "tt/kernels/resident_precondition_protocol.h"
#include "tt/kernels/resident_puller_abi.h"

enum {
    TD_ACC_NOC = 0,
    TD_ACC_BULK_L1 = 1,
    TD_ACC_BULK_DRAM = 2,
    TD_ACCESSORS = 3
};

typedef enum {
    TD_Q_HEAD,        /* write control word 2 */
    TD_Q_RING,        /* write ring slot(s) */
    TD_Q_PARAM_ROT,   /* write a rotating parameter slot */
    TD_Q_PARAM_CACHE, /* write a parameter-cache block */
    TD_Q_OTHER_W,     /* any other queue-region write */
    TD_Q_POLL,        /* read the completion window (words 4..15) */
    TD_Q_GROUP_R,     /* read a group completion block */
    TD_Q_OTHER_R,     /* any other queue-region read */
    TD_B_L1_W,        /* BOUNDARY: L1 write outside the queue region */
    TD_B_L1_R,
    TD_B_DRAM_W,
    TD_B_DRAM_R,
    TD_CLASSES
} td_class;

static const char *const td_class_name[TD_CLASSES] = {
    "q-head", "q-ring", "q-param-rot", "q-param-cache", "q-other-w", "q-poll", "q-group-r",
    "q-other-r", "B-l1-w", "B-l1-r", "B-dram-w", "B-dram-r"
};

#define TD_QUEUE_RING_OFFSET  0x100u
#define TD_TOKEN_MULTIPLIER   0x9E3779B9u
#define TD_DATA_WORDS         128u /* 512 B: the longest queue write (8 ring slots) */
#define TD_MAX_QUEUES         256u

typedef struct {
    uint8_t cls;
    uint8_t acc;
    uint16_t x, y;
    uint32_t addr;
    uint32_t bytes;
    uint32_t repeats;  /* merged consecutive identical reads (a poll loop) */
    uint32_t data_at;  /* index into the token's word pool (writes: payload, reads: last data) */
    uint32_t words;    /* words stored at data_at */
    uint32_t queue;    /* ring/head/poll: the queue's table index */
    uint32_t index;    /* ring: descriptor index of the first slot written */
} td_event;

typedef struct {
    td_event *events;
    size_t count, capacity;
    uint32_t *pool;
    size_t pool_count, pool_capacity;
} td_token;

typedef struct {
    uint16_t x, y;
    uint32_t pending;  /* ring descriptors staged since the last head write */
    size_t pending_from; /* first ring event of those */
    uint32_t nonce;
    bool nonce_known;
    uint32_t mask;     /* capacity - 1, inferred */
    uint32_t base[2];  /* head value at the start of each model token */
} td_queue;

typedef struct {
    pn_tt_device *device;
    ft_tt_noc_access original[TD_ACCESSORS];
    bool wrapped;
    bool recording;
    td_token *into;
    td_token model[2];
    td_token current;
    td_queue queues[TD_MAX_QUEUES];
    uint32_t queue_count;
    bool model_ok;
    bool puller_ready;
    /* A3 live replay */
    bool replaying;      /* this forward's queue traffic is the puller's */
    bool internal;       /* our own puller enqueue/await: pass straight through */
    bool replay_dead;    /* a divergence was seen: never replay again this process */
    size_t cursor;       /* the model event the host is at */
    bool polling;        /* the host is inside the poll loop of event `cursor` */
    uint32_t next_segment, awaited_segment, replay_k;
    uint64_t replayed_tokens, divergences, launched_segments, puller_rearms;
    uint64_t first_forward;
    uint64_t sequence_first; /* the forward a sequence began at (1 = the process's first) */
    bool sequence_reset;     /* pn_tt_dispatch_sequence_reset() was called */
    uint32_t relearns;       /* windows slid this sequence (two model tokens not one shape) */
    uint64_t shape_key;      /* the coming token's shape key (pn_tt_dispatch_shape_key) */
    uint64_t anchor_key;     /* the key the model is learned under */
    uint64_t reanchors;      /* learning re-anchored at a key change */
    /* shadow tallies */
    uint64_t checked_tokens, shape_failures;
    uint64_t compared[TD_CLASSES], mismatched[TD_CLASSES];
    unsigned mismatches_printed;
} td_recorder;

static td_recorder g_td;

static bool td_is_boundary(uint8_t cls)
{
    return cls >= TD_B_L1_W;
}

static uint8_t td_classify(uint8_t acc, uint64_t addr, bool write)
{
    uint64_t offset;

    if (acc == TD_ACC_BULK_DRAM)
        return write ? TD_B_DRAM_W : TD_B_DRAM_R;
    if (addr < FT_TT_OP_QUEUE_BASE || addr >= FT_TT_OP_QUEUE_BASE + 0x10000u)
        return write ? TD_B_L1_W : TD_B_L1_R;
    offset = addr - FT_TT_OP_QUEUE_BASE;
    if (write) {
        if (offset == 8u)
            return TD_Q_HEAD;
        /* The ring is at most 64 slots (0x100..0x900); the parameter cache's overflow
         * region is 0x2000..0x9000 (ft_tt_queue.h), its primary region 0xA000..0xF000. */
        if (offset >= TD_QUEUE_RING_OFFSET && offset < TD_QUEUE_RING_OFFSET + 64u * 32u)
            return TD_Q_RING;
        if (offset >= 0x9000u && offset < 0xA000u)
            return TD_Q_PARAM_ROT;
        if ((offset >= 0xA000u && offset < 0xF000u) ||
            (offset >= FT_TT_QUEUE_PARAM_CACHE_OVERFLOW_OFFSET && offset < 0x9000u))
            return TD_Q_PARAM_CACHE;
        return TD_Q_OTHER_W;
    }
    if (offset >= 0x10u && offset < 0x40u)
        return TD_Q_POLL;
    if (offset >= 0xF000u && offset < 0xF900u)
        return TD_Q_GROUP_R;
    return TD_Q_OTHER_R;
}

static uint32_t td_queue_of(uint16_t x, uint16_t y)
{
    uint32_t i;

    for (i = 0; i < g_td.queue_count; i++)
        if (g_td.queues[i].x == x && g_td.queues[i].y == y)
            return i;
    if (g_td.queue_count == TD_MAX_QUEUES)
        return TD_MAX_QUEUES;
    memset(&g_td.queues[i], 0, sizeof(g_td.queues[i]));
    g_td.queues[i].x = x;
    g_td.queues[i].y = y;
    return g_td.queue_count++;
}

static bool td_reserve(td_token *t, size_t words)
{
    if (t->count == t->capacity) {
        size_t grow = t->capacity ? t->capacity * 2u : 16384u;
        td_event *more = realloc(t->events, grow * sizeof(*more));

        if (!more)
            return false;
        t->events = more;
        t->capacity = grow;
    }
    if (t->pool_count + words > t->pool_capacity) {
        size_t grow = t->pool_capacity ? t->pool_capacity * 2u : 262144u;
        uint32_t *more;

        while (grow < t->pool_count + words)
            grow *= 2u;
        more = realloc(t->pool, grow * sizeof(*more));
        if (!more)
            return false;
        t->pool = more;
        t->pool_capacity = grow;
    }
    return true;
}

static void td_note(uint8_t acc, uint16_t x, uint16_t y, uint64_t addr, const void *data,
                    size_t bytes, bool write)
{
    td_token *t = g_td.into;
    td_event *last, *e;
    const uint8_t cls = td_classify(acc, addr, write);
    size_t words = (bytes + 3u) / 4u;

    if (!g_td.recording || !t)
        return;
    if (words > TD_DATA_WORDS)
        words = TD_DATA_WORDS;
    last = t->count ? &t->events[t->count - 1u] : NULL;
    /* A LONG poll reads the ERROR word (control word 9) every 128 polls to find a refusal.
     * It is part of the same await: fold it in, so the poll after it merges too. */
    if (!write && last && last->cls == TD_Q_POLL && addr == FT_TT_OP_QUEUE_BASE + 9u * 4u &&
        bytes == 4u) {
        /* (any core's: the matmul's group check reads every pending member's error word) */
        last->repeats++;
        return;
    }
    /* A poll loop is ONE event: the same read repeated; keep the LAST answer. */
    if (!write && last && last->cls == cls && last->acc == acc && last->x == x &&
        last->y == y && last->addr == (uint32_t)addr && last->bytes == (uint32_t)bytes) {
        last->repeats++;
        memcpy(&t->pool[last->data_at], data, last->words * 4u > bytes ? bytes : last->words * 4u);
        return;
    }
    if (!td_reserve(t, words)) {
        g_td.recording = false;
        return;
    }
    e = &t->events[t->count++];
    memset(e, 0, sizeof(*e));
    e->cls = cls;
    e->acc = acc;
    e->x = x;
    e->y = y;
    e->addr = (uint32_t)addr;
    e->bytes = (uint32_t)bytes;
    e->repeats = 1u;
    e->data_at = (uint32_t)t->pool_count;
    e->words = (uint32_t)words;
    memset(&t->pool[t->pool_count], 0, words * 4u);
    memcpy(&t->pool[t->pool_count], data, bytes < words * 4u ? bytes : words * 4u);
    t->pool_count += words;
}

static bool td_stream_write(uint8_t acc, uint16_t x, uint16_t y, uint64_t addr,
                            const void *src, size_t bytes);
static bool td_stream_read(uint8_t acc, uint16_t x, uint16_t y, uint64_t addr, void *dst,
                           size_t bytes);
static bool g_streaming;

#define TD_WRAPPERS(index)                                                                     \
    static int td_write_##index(void *user, uint16_t x, uint16_t y, uint64_t addr,              \
                                const void *src, size_t bytes)                                  \
    {                                                                                           \
        (void)user;                                                                             \
        if (g_streaming && !g_td.internal && td_stream_write(index, x, y, addr, src, bytes))   \
            return FT_TT_OK;                                                                    \
        if (g_td.replaying && !g_td.internal && td_replay_access(index, x, y, addr, src, bytes, \
                                                                 true))                         \
            return FT_TT_OK;                                                                    \
        td_note(index, x, y, addr, src, bytes, true);                                           \
        return g_td.original[index].write(g_td.original[index].user, x, y, addr, src, bytes);  \
    }                                                                                           \
    static int td_read_##index(void *user, uint16_t x, uint16_t y, uint64_t addr, void *dst,    \
                               size_t bytes)                                                    \
    {                                                                                           \
        int rc_;                                                                                \
        (void)user;                                                                             \
        if (g_streaming && !g_td.internal && td_stream_read(index, x, y, addr, dst, bytes))    \
            return FT_TT_OK;                                                                    \
        if (g_td.replaying && !g_td.internal &&                                                 \
            td_replay_access(index, x, y, addr, dst, bytes, false))                             \
            return FT_TT_OK;                                                                    \
        rc_ = g_td.original[index].read(g_td.original[index].user, x, y, addr, dst, bytes);     \
        td_note(index, x, y, addr, dst, bytes, false);                                          \
        return rc_;                                                                             \
    }

static bool td_replay_access(uint8_t acc, uint16_t x, uint16_t y, uint64_t addr,
                             const void *data, size_t bytes, bool write);

TD_WRAPPERS(0)
TD_WRAPPERS(1)
TD_WRAPPERS(2)

static ft_tt_noc_access *td_accessor(pn_tt_device *device, unsigned index)
{
    switch (index) {
    case TD_ACC_NOC:
        return &device->runtime.noc;
    case TD_ACC_BULK_L1:
        return &device->runtime.bulk_l1;
    default:
        return &device->runtime.bulk_dram;
    }
}

static void td_wrap(pn_tt_device *device)
{
    static int (*const writes[TD_ACCESSORS])(void *, uint16_t, uint16_t, uint64_t, const void *,
                                             size_t) = { td_write_0, td_write_1, td_write_2 };
    static int (*const reads[TD_ACCESSORS])(void *, uint16_t, uint16_t, uint64_t, void *,
                                            size_t) = { td_read_0, td_read_1, td_read_2 };
    unsigned index;

    if (g_td.wrapped)
        return;
    g_td.device = device;
    for (index = 0; index < TD_ACCESSORS; index++) {
        ft_tt_noc_access *access = td_accessor(device, index);

        g_td.original[index] = *access;
        if (!access->write || !access->read)
            continue;
        access->write = writes[index];
        access->read = reads[index];
    }
    g_td.wrapped = true;
}

static void td_unwrap(void)
{
    unsigned index;

    if (!g_td.wrapped)
        return;
    for (index = 0; index < TD_ACCESSORS; index++)
        *td_accessor(g_td.device, index) = g_td.original[index];
    g_td.wrapped = false;
}

static long td_env_long(const char *name)
{
    const char *value = getenv(name);

    return (value && *value) ? strtol(value, NULL, 10) : 0;
}

/** `PN_TT_DISPATCH`: on-card dispatch (A, the single puller). DEFAULT 1; 0 = host-issued. */
static long td_dispatch_live(void)
{
    const char *value = getenv("PN_TT_DISPATCH");

    return (value && *value) ? strtol(value, NULL, 10) : 1;
}

/* ---------------------------------------------------------------------------------------
 * Resolve: queue identity, descriptor indices, the count each poll waits for
 * --------------------------------------------------------------------------------------- */

/** Attach every ring / head / poll event to its queue and give ring events their index.
 *  @return false if the token's queue traffic is not the shape the model assumes */
static bool td_resolve(td_token *t)
{
    size_t i;
    uint32_t q;

    for (q = 0; q < g_td.queue_count; q++)
        g_td.queues[q].pending = 0u;
    for (i = 0; i < t->count; i++) {
        td_event *e = &t->events[i];

        if (e->cls != TD_Q_RING && e->cls != TD_Q_HEAD && e->cls != TD_Q_POLL &&
            e->cls != TD_Q_PARAM_ROT && e->cls != TD_Q_PARAM_CACHE)
            continue;
        q = td_queue_of(e->x, e->y);
        if (q == TD_MAX_QUEUES)
            return false;
        e->queue = q;
        if (e->cls == TD_Q_RING) {
            if (e->bytes % 32u != 0u)
                return false;
            if (g_td.queues[q].pending == 0u)
                g_td.queues[q].pending_from = i;
            g_td.queues[q].pending += e->bytes / 32u;
        } else if (e->cls == TD_Q_HEAD) {
            uint32_t head = t->pool[e->data_at];
            uint32_t index = head - g_td.queues[q].pending;
            size_t k;

            /* The ring writes since the last head hold indices head-pending .. head-1. */
            for (k = g_td.queues[q].pending_from; g_td.queues[q].pending && k < i; k++) {
                td_event *r = &t->events[k];

                if (r->cls != TD_Q_RING || r->queue != q)
                    continue;
                r->index = index;
                index += r->bytes / 32u;
            }
            g_td.queues[q].pending = 0u;
        }
    }
    return true;
}

static uint32_t td_token_at(uint32_t nonce, uint32_t index)
{
    uint32_t token = nonce ^ ((index + 1u) * TD_TOKEN_MULTIPLIER);

    if (token == 0u || token == (uint32_t)FT_TT_QUEUE_STATE_HOST_POISON)
        token = nonce ^ 0xA5A5A5A5u;
    return token;
}

/** Nonce and ring capacity per queue, from the model tokens. */
/** The queue's ring capacity - 1, read from its control block (word 1), or 0 if the read
 *  fails or the word is not a power of two (the caller then infers it). */
static uint32_t td_queue_capacity_mask(const td_queue *queue)
{
    const ft_tt_noc_access *bulk = &g_td.original[TD_ACC_BULK_L1];
    uint32_t words[4];

    if (!bulk->read || bulk->read(bulk->user, queue->x, queue->y, FT_TT_OP_QUEUE_BASE, words,
                                  sizeof(words)) != FT_TT_OK)
        return 0u;
    if (words[1] == 0u || (words[1] & (words[1] - 1u)) != 0u || words[1] > 0x10000u)
        return 0u;
    return words[1] - 1u;
}

static bool td_learn_queues_at(size_t *failed_at)
{
    uint32_t q;
    unsigned m;
    size_t i;

    for (q = 0; q < g_td.queue_count; q++) {
        g_td.queues[q].nonce_known = false;
        g_td.queues[q].mask = 0u;
    }
    for (m = 0; m < 2u; m++) {
        const td_token *t = &g_td.model[m];

        for (i = 0; i < t->count; i++) {
            const td_event *e = &t->events[i];
            td_queue *queue;
            uint32_t slot, d;

            if (e->cls != TD_Q_RING)
                continue;
            queue = &g_td.queues[e->queue];
            if (!queue->nonce_known) {
                queue->nonce = t->pool[e->data_at + 1u] ^ ((e->index + 1u) * TD_TOKEN_MULTIPLIER);
                queue->nonce_known = true;
            }
            for (d = 0; d < e->bytes / 32u; d++)
                if (t->pool[e->data_at + d * 8u + 1u] != td_token_at(queue->nonce, e->index + d)) {
                    *failed_at = i + (size_t)m * 1000000000u;
                    return false;
                }
            slot = (e->addr - FT_TT_OP_QUEUE_BASE - TD_QUEUE_RING_OFFSET) / 32u;
            /* THE CAPACITY IS READ, NOT INFERRED. "The smallest power of two every recorded
             * slot agrees with" is underdetermined for a queue with one descriptor a token:
             * two tokens at indices 144, 145 of a 16-slot ring agree with capacity 2, and the
             * replay then put index 146 in slot 0 (a multi-turn delta prefill after spec
             * cycles: DIVERGED at k=2, a puller segment failed, the turn came back empty).
             * RED `dispatch_mask_infer` keeps the inference. */
            if (queue->mask == 0u && !pn_tt_red("dispatch_mask_infer"))
                queue->mask = td_queue_capacity_mask(queue);
            if ((e->index & queue->mask) != slot && queue->mask != 0u &&
                !pn_tt_red("dispatch_mask_infer")) {
                *failed_at = i + (size_t)m * 1000000000u + 500000000u;
                return false;
            }
            while ((e->index & queue->mask) != slot) {
                queue->mask = queue->mask * 2u + 1u;
                if (queue->mask > 0xFFFFu)
                    return false;
            }
        }
    }
    return true;
}

/* ---------------------------------------------------------------------------------------
 * The shadow: predict a later token from the model, compare byte for byte
 * --------------------------------------------------------------------------------------- */

static bool td_same_shape(const td_event *a, const td_event *b)
{
    return a->cls == b->cls && a->acc == b->acc && a->x == b->x && a->y == b->y &&
           a->bytes == b->bytes && (a->cls == TD_Q_RING || a->addr == b->addr);
}

static void td_mismatch(uint64_t forward, size_t i, const char *what, uint32_t want,
                        uint32_t got)
{
    if (g_td.mismatches_printed >= 12u)
        return;
    g_td.mismatches_printed++;
    fprintf(stderr, "pn_tt: DISPATCH SHADOW forward %llu event %zu %s: predicted 0x%08x, the "
                    "host wrote 0x%08x\n",
            (unsigned long long)forward, i, what, want, got);
}

static void td_shadow_check(uint64_t forward)
{
    const td_token *a = &g_td.model[0], *b = &g_td.model[1], *t = &g_td.current;
    const uint32_t k = (uint32_t)(forward - g_td.first_forward);
    const bool skip_patch = pn_tt_red("dispatch_skip_patch");
    size_t i;

    g_td.checked_tokens++;
    if (t->count != a->count) {
        g_td.shape_failures++;
        fprintf(stderr, "pn_tt: DISPATCH SHADOW forward %llu: %zu events, the model has %zu\n",
                (unsigned long long)forward, t->count, a->count);
        return;
    }
    for (i = 0; i < t->count; i++) {
        const td_event *ea = &a->events[i], *eb = &b->events[i], *et = &t->events[i];
        bool bad = false;
        uint32_t w;

        if (!td_same_shape(ea, et)) {
            g_td.shape_failures++;
            if (g_td.mismatches_printed < 12u) {
                g_td.mismatches_printed++;
                fprintf(stderr, "pn_tt: DISPATCH SHADOW forward %llu event %zu: shape %s "
                                "(%u,%u) 0x%x %u B, the model has %s (%u,%u) 0x%x %u B\n",
                        (unsigned long long)forward, i, td_class_name[et->cls], et->x, et->y,
                        et->addr, et->bytes, td_class_name[ea->cls], ea->x, ea->y, ea->addr,
                        ea->bytes);
            }
            return;
        }
        if (td_is_boundary(et->cls) || et->cls == TD_Q_GROUP_R || et->cls == TD_Q_OTHER_R)
            continue;
        g_td.compared[et->cls]++;
        if (et->cls == TD_Q_RING) {
            const td_queue *queue = &g_td.queues[ea->queue];
            const uint32_t step = eb->index - ea->index;
            const uint32_t index = ea->index + step * k;
            const uint32_t addr = FT_TT_OP_QUEUE_BASE + TD_QUEUE_RING_OFFSET +
                                  (index & queue->mask) * 32u;
            uint32_t d, word;

            if (et->addr != addr) {
                td_mismatch(forward, i, "ring slot address", addr, et->addr);
                bad = true;
            }
            for (d = 0; !bad && d < et->bytes / 32u; d++)
                for (word = 0; word < 8u; word++) {
                    const uint32_t at = d * 8u + word;

                    w = word == 1u ? td_token_at(queue->nonce, index + d)
                                   : a->pool[ea->data_at + at] +
                                         (skip_patch ? 0u
                                                     : (b->pool[eb->data_at + at] -
                                                        a->pool[ea->data_at + at]) * k);
                    if (w != t->pool[et->data_at + at]) {
                        td_mismatch(forward, i, word == 1u ? "ring token" : "ring word", w,
                                    t->pool[et->data_at + at]);
                        bad = true;
                        break;
                    }
                }
        } else {
            /* Writes: every word linear. Polls: only the word that holds COMPLETED_COUNT
             * (control word 5) is the await's target; the rest is the card's own state. */
            uint32_t word, first = 0u, last = et->words;

            if (et->cls == TD_Q_POLL) {
                const uint32_t offset = et->addr - FT_TT_OP_QUEUE_BASE;

                if (offset > 0x14u || offset + et->bytes <= 0x14u)
                    continue; /* a poll that does not read the count (an error word) */
                first = (0x14u - offset) / 4u;
                last = first + 1u;
            }
            for (word = first; word < last; word++) {
                w = a->pool[ea->data_at + word] +
                    (skip_patch ? 0u
                                : (b->pool[eb->data_at + word] - a->pool[ea->data_at + word]) *
                                      k);
                if (w != t->pool[et->data_at + word]) {
                    td_mismatch(forward, i, td_class_name[et->cls], w,
                                t->pool[et->data_at + word]);
                    bad = true;
                    break;
                }
            }
        }
        if (bad)
            g_td.mismatched[et->cls]++;
    }
}

static void td_shadow_report(void)
{
    unsigned cls;
    uint64_t total = 0u, bad = 0u;

    for (cls = 0; cls < TD_CLASSES; cls++) {
        total += g_td.compared[cls];
        bad += g_td.mismatched[cls];
    }
    fprintf(stderr, "pn_tt: DISPATCH SHADOW — model from forwards %llu/%llu (%s), %u queues; "
                    "%llu later tokens checked, %llu shape failures, %llu of %llu predicted "
                    "events MISMATCHED%s\n",
            (unsigned long long)g_td.first_forward, (unsigned long long)g_td.first_forward + 1u,
            g_td.model_ok ? "learned" : "REFUSED", g_td.queue_count,
            (unsigned long long)g_td.checked_tokens, (unsigned long long)g_td.shape_failures,
            (unsigned long long)bad, (unsigned long long)total,
            pn_tt_red("dispatch_skip_patch") ? " [RED dispatch_skip_patch]" : "");
    for (cls = 0; cls < TD_CLASSES; cls++)
        if (g_td.compared[cls])
            fprintf(stderr, "pn_tt: DISPATCH SHADOW   %-13s %9llu compared %7llu mismatched\n",
                    td_class_name[cls], (unsigned long long)g_td.compared[cls],
                    (unsigned long long)g_td.mismatched[cls]);
}

/* ---------------------------------------------------------------------------------------
 * A2: the puller's PROGRAM, built from the model, and its on-card DRY RUN
 * --------------------------------------------------------------------------------------- */

#define TD_PROGRAM_BASE      0x26000u  /* the CB arena: the puller has no CBs */
#define TD_PROGRAM_LIMIT     0x160000u /* the puller's scratch starts here */
#define TD_PULL_MAGIC        0x524C5044u
#define TD_PULL_OPCODE       0x14u
#define TD_PULLER_DEFAULT_CORE 89u

typedef struct {
    uint32_t *words;
    size_t count, capacity;
    uint32_t segments;
    uint32_t segment_offset[256];
    bool ok;
} td_program;

static td_program g_prog;
static uint64_t g_ring_patch_word[8]; /* which descriptor words move token to token */

static void td_emit(uint32_t word)
{
    if (!g_prog.ok)
        return;
    if (g_prog.count == g_prog.capacity) {
        size_t grow = g_prog.capacity ? g_prog.capacity * 2u : 65536u;
        uint32_t *more = realloc(g_prog.words, grow * sizeof(*more));

        if (!more) {
            g_prog.ok = false;
            return;
        }
        g_prog.words = more;
        g_prog.capacity = grow;
    }
    g_prog.words[g_prog.count++] = word;
}

/** Emit base words, then {word, stride} for every word that moves between the two models. */
static void td_emit_words(const uint32_t *base, const uint32_t *next, uint32_t n,
                          int skip_every_8th_1)
{
    uint32_t i, patches = 0u;
    size_t header_at = g_prog.count - 4u; /* caller emitted the entry's 4 header-ish words */

    (void)header_at;
    for (i = 0; i < n; i++)
        td_emit(base[i]);
    for (i = 0; i < n; i++) {
        if (skip_every_8th_1 && (i % 8u) == 1u)
            continue; /* the ring token: computed on the card */
        if (next[i] != base[i]) {
            td_emit(i);
            td_emit(next[i] - base[i]);
            patches++;
        }
    }
    (void)patches;
}

static uint32_t td_count_patches(const uint32_t *base, const uint32_t *next, uint32_t n,
                                 int ring)
{
    uint32_t i, patches = 0u;

    for (i = 0; i < n; i++)
        if (!(ring && (i % 8u) == 1u) && next[i] != base[i])
            patches++;
    return patches;
}

/** `PN_TT_DISPATCH_MCAST`: STEP 3 B, DEFAULT 1 (run_d_mcast.sh: +1.0% steady, IDENTICAL). */
static long td_mcast_on(void)
{
    const char *value = getenv("PN_TT_DISPATCH_MCAST");

    return (value && *value) ? strtol(value, NULL, 10) : 1;
}

/* ---------------------------------------------------------------------------------------
 * STEP 3 B: MULTICAST GROUPS. Ring slots whose bytes are identical on several queues (the
 * matmul members 1..N share a nonce, an index and a parameter-cache layout) go out as ONE
 * multicast to a rectangle of <= 4 member cores (the recorded wedge rule), and their heads
 * as one 4-byte multicast. `PN_TT_DISPATCH_MCAST` (default 1).
 * --------------------------------------------------------------------------------------- */

#define TD_MAX_GROUPS 64u

typedef struct {
    uint32_t queues[4];
    uint32_t count;
    uint16_t x_lo, y_lo, x_hi, y_hi;
} td_group;

static td_group g_groups[TD_MAX_GROUPS];
static uint32_t g_group_count;
static int32_t *g_merge; /* per model event: -1 plain, >= 0 group leader, -2 merged away */
static uint64_t g_merged_rings, g_merged_heads;

static bool td_ring_same(size_t i, size_t j)
{
    const td_token *a = &g_td.model[0], *b = &g_td.model[1];
    const td_event *ai = &a->events[i], *aj = &a->events[j];
    const td_event *bi = &b->events[i], *bj = &b->events[j];
    const td_queue *qi = &g_td.queues[ai->queue], *qj = &g_td.queues[aj->queue];
    uint32_t w;

    if (ai->bytes != aj->bytes || ai->addr != aj->addr || ai->index != aj->index ||
        bi->index != bj->index || qi->nonce != qj->nonce || qi->mask != qj->mask)
        return false;
    for (w = 0; w < ai->words; w++)
        if (a->pool[ai->data_at + w] != a->pool[aj->data_at + w] ||
            b->pool[bi->data_at + w] != b->pool[bj->data_at + w])
            return false;
    return true;
}

static bool td_head_same(size_t i, size_t j)
{
    const td_token *a = &g_td.model[0], *b = &g_td.model[1];

    return a->pool[a->events[i].data_at] == a->pool[a->events[j].data_at] &&
           b->pool[b->events[i].data_at] == b->pool[b->events[j].data_at];
}

/** Partition the queues that share a nonce with another queue into row runs of <= 4
 *  adjacent cores (same y, consecutive x), every cell of a run a sharing queue. */
static void td_find_groups(void)
{
    bool used[TD_MAX_QUEUES];
    uint32_t q, r;

    memset(used, 0, sizeof(used));
    g_group_count = 0u;
    for (q = 0; q < g_td.queue_count; q++) {
        uint32_t peers = 0u;

        for (r = 0; r < g_td.queue_count; r++)
            if (r != q && g_td.queues[r].nonce == g_td.queues[q].nonce)
                peers++;
        if (peers == 0u)
            used[q] = true; /* not a sharing queue */
    }
    for (q = 0; q < g_td.queue_count && g_group_count < TD_MAX_GROUPS; q++) {
        td_group *g = &g_groups[g_group_count];
        uint32_t x;

        if (used[q])
            continue;
        memset(g, 0, sizeof(*g));
        g->queues[0] = q;
        g->count = 1u;
        g->x_lo = g->x_hi = g_td.queues[q].x;
        g->y_lo = g->y_hi = g_td.queues[q].y;
        used[q] = true;
        /* extend to the right along the row */
        for (x = g_td.queues[q].x + 1u; g->count < 4u; x++) {
            for (r = 0; r < g_td.queue_count; r++)
                if (!used[r] && g_td.queues[r].y == g->y_lo && g_td.queues[r].x == x &&
                    g_td.queues[r].nonce == g_td.queues[q].nonce)
                    break;
            if (r == g_td.queue_count)
                break;
            g->queues[g->count++] = r;
            g->x_hi = (uint16_t)x;
            used[r] = true;
        }
        if (g->count >= 2u)
            g_group_count++;
    }
}

/** Mark, window by window (between awaits / boundaries), each group whose members' k-th
 *  ring writes are byte-identical (and k-th heads equal) as ONE multicast. */
static bool td_plan_merges(void)
{
    const td_token *a = &g_td.model[0];
    size_t i, from = 0u;

    free(g_merge);
    g_merge = malloc(a->count * sizeof(*g_merge));
    if (!g_merge)
        return false;
    for (i = 0; i < a->count; i++)
        g_merge[i] = -1;
    g_merged_rings = g_merged_heads = 0u;
    if (td_mcast_on() <= 0)
        return true;
    td_find_groups();
    for (i = 0; i <= a->count; i++) {
        size_t j, k;
        uint32_t g;

        if (i < a->count && !td_is_boundary(a->events[i].cls) && a->events[i].cls != TD_Q_POLL)
            continue;
        /* window [from, i) */
        for (g = 0; g < g_group_count; g++) {
            const td_group *grp = &g_groups[g];
            uint32_t occurrence;

            for (occurrence = 0;; occurrence++) {
                size_t ring_at[4], head_at[4];
                uint32_t m, found = 0u;

                for (m = 0; m < grp->count; m++) {
                    uint32_t seen_r = 0u, seen_h = 0u;

                    ring_at[m] = head_at[m] = (size_t)-1;
                    for (j = from; j < i; j++) {
                        const td_event *e = &a->events[j];

                        if (e->queue != grp->queues[m])
                            continue;
                        if (e->cls == TD_Q_RING && seen_r++ == occurrence)
                            ring_at[m] = j;
                        if (e->cls == TD_Q_HEAD && seen_h++ == occurrence)
                            head_at[m] = j;
                    }
                    if (ring_at[m] != (size_t)-1 && head_at[m] != (size_t)-1)
                        found++;
                }
                if (found == 0u)
                    break;
                if (found != grp->count)
                    continue;
                for (k = 1; k < grp->count; k++)
                    if (!td_ring_same(ring_at[0], ring_at[k]) ||
                        !td_head_same(head_at[0], head_at[k]))
                        break;
                if (k != grp->count)
                    continue;
                /* the LEADER is the earliest ring / head of the group in the window */
                {
                    size_t lr = ring_at[0], lh = head_at[0];

                    for (k = 1; k < grp->count; k++) {
                        if (ring_at[k] < lr)
                            lr = ring_at[k];
                        if (head_at[k] < lh)
                            lh = head_at[k];
                    }
                    for (k = 0; k < grp->count; k++) {
                        g_merge[ring_at[k]] = -2;
                        g_merge[head_at[k]] = -2;
                    }
                    g_merge[lr] = (int32_t)g;
                    g_merge[lh] = (int32_t)g;
                    g_merged_rings += grp->count;
                    g_merged_heads += grp->count;
                }
            }
        }
        from = i + 1u;
    }
    return true;
}

/* ---------------------------------------------------------------------------------------
 * L10: HOISTING (pn_tt_dispatch_hoist.h). Each emitted entry is noted with what it writes or
 * waits for; the planner reorders the entries so a descriptor's parameter block and ring slot
 * go out while the puller awaits the core before it, behind a FLUSH of the queued heads.
 * `PN_TT_DISPATCH_HOIST` (0 = the host's order).
 * --------------------------------------------------------------------------------------- */

typedef struct {
    pn_tt_hoist_item *items;
    pn_tt_pre_item *pre; /* S2: the same entry as the precondition planner sees it */
    size_t *word_at;   /* first program word of each item's entry */
    size_t count, capacity;
    pn_tt_hoist_stats stats;
    bool planned;
} td_hoist_state;

static td_hoist_state g_hoist;

static long td_hoist_on(void)
{
    const char *value = getenv("PN_TT_DISPATCH_HOIST");

    /* DEFAULT 1 (run_l10_hoist.sh: plain 20.13 -> 20.22 tok/s twice, IDENTICAL, reds green) */
    return (value && *value) ? strtol(value, NULL, 10) : 1;
}

/* ---------------------------------------------------------------------------------------
 * S2: ON-CARD COMPLETION (pn_tt_precondition.h). The replay program and the decode / verify
 * streams turn an AWAIT followed by heads into wait sidecar entries on those heads'
 * descriptors plus the same AWAIT a window later. `PN_TT_ONCARD_COMPLETION` (default 1, David 2026-11-17; 0 = the pre-S2
 * programs, byte for byte), `PN_TT_ONCARD_LOOKAHEAD` (windows ahead, default 1), RED
 * `PN_TT_RED=precondition_early` (every entry one count short: must DIFFER).
 * --------------------------------------------------------------------------------------- */

static bool td_oncard_on(void)
{
    static int on = -1;

    if (on < 0) {
        const char *value = getenv("PN_TT_ONCARD_COMPLETION");

        on = value && *value ? atoi(value) != 0 : 1;
    }
    return on == 1;
}

static uint32_t td_oncard_lookahead(void)
{
    static uint32_t lookahead;

    if (lookahead == 0u) {
        const char *value = getenv("PN_TT_ONCARD_LOOKAHEAD");
        const long parsed = value && *value ? strtol(value, NULL, 10) : 1;

        lookahead = parsed < 1 ? 1u
                               : parsed > (long)PN_TT_PRE_MAX_LOOKAHEAD ? PN_TT_PRE_MAX_LOOKAHEAD
                                                                        : (uint32_t)parsed;
    }
    return lookahead;
}

/** A plan's actions, as the planner hands them out (the replay keeps its program's; a stream
 *  under `PN_TT_ONCARD_VERIFY=1` its own, for the check at its end). */
typedef struct {
    pn_tt_pre_action *a;
    size_t n, cap;
    bool oom;
} td_oncard_actions;

static void td_oncard_append(td_oncard_actions *list, const pn_tt_pre_action *action)
{
    if (list->n == list->cap) {
        const size_t grow = list->cap ? list->cap * 2u : 8192u;
        pn_tt_pre_action *more = realloc(list->a, grow * sizeof(*more));

        if (!more) {
            list->oom = true;
            return;
        }
        list->a = more;
        list->cap = grow;
    }
    list->a[list->n++] = *action;
}

static struct {
    td_oncard_actions plan; /* the replay program's, in program order */
    pn_tt_pre_stats stats;
    pn_tt_pre_report report;
    bool planned;           /* the uploaded replay program is the S2 one */
} g_oncard;

static void td_oncard_record(void *user, const pn_tt_pre_action *action)
{
    (void)user;
    td_oncard_append(&g_oncard.plan, action);
}

/** `PN_TT_DISPATCH_LAZY_HEADS`: the puller skips the write barrier after a head flush inside a
 *  segment (resident_puller.c args[4] bit 6; one barrier at the segment end). */
static uint32_t td_lazy_heads_bit(void)
{
    static int on = -1;

    if (on < 0) {
        const char *value = getenv("PN_TT_DISPATCH_LAZY_HEADS");

        on = (value && *value) ? atoi(value) != 0 : 0;
    }
    return on ? 64u : 0u;
}

/** The descriptor index a rotating parameter block at event `i` belongs to: the next ring
 *  write on the same queue. UINT32_MAX when there is none. */
static uint32_t td_param_descriptor(size_t i)
{
    const td_token *a = &g_td.model[0];
    size_t j;

    for (j = i + 1u; j < a->count; j++)
        if (a->events[j].cls == TD_Q_RING && a->events[j].queue == a->events[i].queue)
            return a->events[j].index;
    return UINT32_MAX;
}

/** Note the entry the builder just emitted for event `i` (at word `entry_at`). `kind` forces
 *  a kind (segment end); 0 classifies the event. `pre`: the entry as S2's planner sees it. */
static void td_hoist_note(size_t i, size_t entry_at, uint8_t kind, const pn_tt_pre_item *pre)
{
    const td_token *a = &g_td.model[0];
    pn_tt_hoist_item *it;

    if (g_hoist.count == g_hoist.capacity) {
        const size_t grow = g_hoist.capacity ? g_hoist.capacity * 2u : 8192u;
        pn_tt_hoist_item *items = realloc(g_hoist.items, grow * sizeof(*items));
        pn_tt_pre_item *pres = items ? realloc(g_hoist.pre, grow * sizeof(*pres)) : NULL;
        size_t *word_at = pres ? realloc(g_hoist.word_at, grow * sizeof(*word_at)) : NULL;

        if (items)
            g_hoist.items = items;
        if (pres)
            g_hoist.pre = pres;
        if (!items || !pres || !word_at) {
            g_prog.ok = false;
            return;
        }
        g_hoist.word_at = word_at;
        g_hoist.capacity = grow;
    }
    g_hoist.pre[g_hoist.count] = *pre;
    it = &g_hoist.items[g_hoist.count];
    g_hoist.word_at[g_hoist.count++] = entry_at;
    memset(it, 0, sizeof(*it));
    it->group = -1;
    it->kind = kind ? kind : PN_TT_HOIST_FIXED;
    if (kind)
        return;
    {
        const td_event *e = &a->events[i];
        const uint32_t *w = &a->pool[e->data_at];

        it->queue = (uint16_t)e->queue;
        if ((e->cls == TD_Q_RING || e->cls == TD_Q_HEAD) && g_merge[i] >= 0) {
            it->group = (int16_t)g_merge[i];
            it->queue = (uint16_t)g_groups[g_merge[i]].queues[0];
        }
        switch (e->cls) {
        case TD_Q_RING: {
            const uint32_t last = (e->bytes / 32u - 1u) * 8u;

            it->kind = PN_TT_HOIST_RING;
            it->index = e->index;
            it->slots = e->bytes / 32u;
            /* a3 > 1: member 0 pulls that many members up to sequence a4; a5: this core's
             * group sequence (resident_matmul_writer.c) */
            if ((w[last + 5u] & 0x7FFFFFFFu) > 1u)
                it->pull_sequence = w[last + 6u];
            it->group_sequence = w[last + 7u];
            break;
        }
        case TD_Q_HEAD:
            it->kind = PN_TT_HOIST_HEAD;
            break;
        case TD_Q_POLL: {
            const uint32_t offset = e->addr - FT_TT_OP_QUEUE_BASE;

            it->kind = PN_TT_HOIST_AWAIT;
            it->index = w[(0x14u - offset) / 4u];
            break;
        }
        case TD_Q_PARAM_ROT:
            it->index = td_param_descriptor(i);
            if (it->index != UINT32_MAX) {
                it->kind = PN_TT_HOIST_WRITE;
                it->address = e->addr;
            }
            break;
        default:
            break;
        }
    }
}

/** The queue index of core (x, y) in `count` queues found by `queue_of`, or -1. */
static int32_t td_find_queue(uint16_t x, uint16_t y, uint32_t count,
                             uint16_t (*xy_of)(uint32_t q, int which))
{
    uint32_t q;

    for (q = 0; q < count; q++)
        if (xy_of(q, 0) == x && xy_of(q, 1) == y)
            return (int32_t)q;
    return -1;
}

/** S1: the members of one split vector family's planted pull table (pn_tt_group_pull.h) get
 *  its member 0's queue as their leader. Nothing when the table is not live. */
static void td_hoist_pull_leaders(const pn_tt_group_pull *pull, uint32_t count,
                                  uint16_t (*xy_of)(uint32_t, int), int32_t *leader_of)
{
    int32_t leader;
    uint32_t m;

    if (!pn_tt_group_pull_covers(pull, pull->members))
        return;
    leader = td_find_queue(pull->core_x[0], pull->core_y[0], count, xy_of);
    if (leader < 0)
        return;
    for (m = 1; m < pull->members; m++) {
        const int32_t q = td_find_queue(pull->core_x[m], pull->core_y[m], count, xy_of);

        if (q >= 0)
            leader_of[q] = leader;
    }
}

/** Members whose completion member 0's PULL implies: the decode matmul group's
 *  (resident_matmul_writer.c pull_group) and every split vector family's (S1). Each gets its
 *  member 0's queue as its leader. Nothing for a pull that is not armed or that a red arm
 *  skips or shortens. */
static void td_hoist_leaders(pn_tt_device *device, uint32_t count, uint16_t (*xy_of)(uint32_t, int),
                      int32_t *leader_of)
{
    const pn_tt_matmul_group *group = &device->group;
    int32_t leader;
    uint32_t m;

    for (m = 0; m < count; m++)
        leader_of[m] = -1;
    if (!pn_tt_red("group_pull_short")) {
        td_hoist_pull_leaders(&device->eltwise_pull, count, xy_of, leader_of);
        td_hoist_pull_leaders(&device->reduce_pull, count, xy_of, leader_of);
        td_hoist_pull_leaders(&device->attention_pull, count, xy_of, leader_of);
        td_hoist_pull_leaders(&device->gdn_pull, count, xy_of, leader_of);
        td_hoist_pull_leaders(&device->memory_pull, count, xy_of, leader_of);
        for (m = 0; m < (uint32_t)PN_TT_RENDER_DIRECTIONS; m++)
            td_hoist_pull_leaders(&device->render[m].pull, count, xy_of, leader_of);
        for (m = 0; m < PN_TT_HEAD_NORM_MAX_WIDTHS; m++)
            td_hoist_pull_leaders(&device->head_norm[m].pull, count, xy_of, leader_of);
    }
    if (group->member_count < 2u || !group->pull || !group->pull_table_ok ||
        group->red_pull_skip)
        return;
    leader = td_find_queue(group->member[0].core_x, group->member[0].core_y, count, xy_of);
    if (leader < 0)
        return;
    for (m = 1; m < group->member_count; m++) {
        const int32_t q = td_find_queue(group->member[m].core_x, group->member[m].core_y,
                                        count, xy_of);

        if (q >= 0)
            leader_of[q] = leader;
    }
}

static uint16_t td_queue_xy(uint32_t q, int which)
{
    return which ? g_td.queues[q].y : g_td.queues[q].x;
}

/** Per queue: capacity, per-token step, pull leader. */
static void td_hoist_queues(pn_tt_hoist_queue *queues)
{
    const td_token *a = &g_td.model[0], *b = &g_td.model[1];
    int32_t leader_of[TD_MAX_QUEUES];
    uint32_t q;
    size_t i;

    td_hoist_leaders(g_td.device, g_td.queue_count, td_queue_xy, leader_of);
    for (q = 0; q < g_td.queue_count; q++) {
        queues[q].capacity = g_td.queues[q].mask + 1u;
        queues[q].step = 0u;
        queues[q].leader = leader_of[q];
    }
    for (i = 0; i < a->count; i++)
        if (a->events[i].cls == TD_Q_RING && queues[a->events[i].queue].step == 0u)
            queues[a->events[i].queue].step = b->events[i].index - a->events[i].index;
}

/** Where an assembly of the reordered program stands. */
typedef struct {
    uint32_t *words;
    size_t at, entries_from;
    uint32_t offsets[256], segments;
    bool open, ok;
} td_assembly;

/** Append an entry's words, opening a segment first when none is open. */
static void td_assemble_words(td_assembly *as, const uint32_t *w, size_t n)
{
    if (!as->open) {
        as->ok = as->segments < 256u;
        if (!as->ok)
            return;
        as->offsets[as->segments++] = (uint32_t)((as->entries_from + as->at) * 4u);
        as->open = true;
    }
    memcpy(&as->words[as->at], w, n * sizeof(*w));
    as->at += n;
}

/** Append noted entry `origin` as the builder emitted it, or a FLUSH for the mark. */
static void td_assemble_item(td_assembly *as, uint32_t origin)
{
    size_t from, to;

    if (origin == PN_TT_HOIST_FLUSH_MARK) {
        as->words[as->at++] = FT_TT_PULLER_ENTRY_FLUSH;
        return;
    }
    from = g_hoist.word_at[origin];
    to = origin + 1u < g_hoist.count ? g_hoist.word_at[origin + 1u] : g_prog.count;
    td_assemble_words(as, &g_prog.words[from], to - from);
    if (g_hoist.items[origin].kind == PN_TT_HOIST_SEGMENT)
        as->open = false;
}

/** S2: one action of the precondition plan; an item is `order[item]`. */
static void td_assemble_action(td_assembly *as, const pn_tt_pre_action *a, const uint32_t *order)
{
    uint32_t w[FT_TT_PULLER_PRECOND_WORDS];

    switch (a->kind) {
    case PN_TT_PRE_OUT_ITEM:
        td_assemble_item(as, order[a->item]);
        break;
    case PN_TT_PRE_OUT_AWAIT:
        w[0] = 4u;
        w[1] = a->queue;
        w[2] = a->count;
        w[3] = a->count_step;
        td_assemble_words(as, w, 4u);
        break;
    default:
        w[0] = FT_TT_PULLER_ENTRY_PRECOND;
        w[1] = a->group >= 0 ? 0x80000000u | (uint32_t)a->group : a->queue;
        w[2] = a->index;
        w[3] = a->index_step;
        w[4] = a->producer;
        w[5] = a->count;
        w[6] = a->count_step;
        td_assemble_words(as, w, FT_TT_PULLER_PRECOND_WORDS);
        break;
    }
}

/** `PN_TT_ONCARD_DUMP=<file>`: the replay's planning problem (queues, groups, the items as the
 *  planner sees them, after the hoist) and its plan, for an off-card re-plan and re-check
 *  (tests/test_tt_precondition --replay <file>). Rewritten at every program build. */
static void td_oncard_dump(const pn_tt_pre_config *config, const pn_tt_hoist_queue *queues,
                           const pn_tt_hoist_group *groups, const pn_tt_pre_item *in, size_t n)
{
    const char *path = getenv("PN_TT_ONCARD_DUMP");
    uint32_t capacity[TD_MAX_QUEUES], step[TD_MAX_QUEUES];
    int32_t leader[TD_MAX_QUEUES];
    bool honours[TD_MAX_QUEUES];
    pn_tt_pre_dump d;
    uint32_t q;
    FILE *f;

    if (!path || !*path)
        return;
    for (q = 0; q < g_td.queue_count; q++) {
        capacity[q] = queues[q].capacity;
        step[q] = queues[q].step;
        leader[q] = queues[q].leader;
        honours[q] = true;
    }
    memset(&d, 0, sizeof(d));
    d.config = *config;
    d.queue_count = g_td.queue_count;
    d.capacity = capacity;
    d.step = step;
    d.leader = leader;
    d.honours = honours;
    d.group_count = g_group_count;
    d.groups = (pn_tt_hoist_group *)(uintptr_t)groups;
    d.items = (pn_tt_pre_item *)(uintptr_t)in;
    d.n_items = n;
    d.actions = g_oncard.plan.a;
    d.n_actions = g_oncard.plan.n;
    f = fopen(path, "w");
    if (!f || pn_tt_precondition_dump_write(f, &d) != 0)
        fprintf(stderr, "pn_tt: DISPATCH — ONCARD dump to %s failed\n", path);
    else
        fprintf(stderr, "pn_tt: DISPATCH — ONCARD dump: %zu items, %zu actions -> %s\n", n,
                d.n_actions, path);
    if (f)
        fclose(f);
}

/** S2: plan the (hoisted) order into g_oncard.plan and check it. Carried from the previous
 *  token like the hoist; refused (false, the order stands) when the check fails — except
 *  under RED `precondition_early`, whose refused plan is the point. */
static bool td_oncard_plan(const uint32_t *order, size_t planned, const pn_tt_hoist_queue *queues,
                           const pn_tt_hoist_group *groups)
{
    const pn_tt_pre_config config = { td_oncard_lookahead(), pn_tt_red("precondition_early") };
    pn_tt_pre_queue_info info[TD_MAX_QUEUES];
    pn_tt_pre_item *in = malloc((planned ? planned : 1u) * sizeof(*in));
    pn_tt_pre_planner *p =
        in ? pn_tt_precondition_create(g_td.queue_count, &config, td_oncard_record, NULL) : NULL;
    bool ok = p != NULL;
    size_t k;
    uint32_t q;

    g_oncard.plan.n = 0u;
    g_oncard.plan.oom = false;
    for (q = 0; ok && q < g_td.queue_count; q++) {
        info[q].capacity = queues[q].capacity;
        info[q].honours = true; /* every resident reader waits (foundry check_precondition_wait.sh) */
        pn_tt_precondition_queue(p, q, queues[q].capacity, queues[q].step, queues[q].leader, true);
    }
    if (ok)
        pn_tt_precondition_groups(p, groups, g_group_count);
    for (k = 0; ok && k < planned; k++) {
        static const pn_tt_pre_item flush = { .kind = PN_TT_PRE_NEUTRAL, .group = -1 };

        in[k] = order[k] == PN_TT_HOIST_FLUSH_MARK ? flush : g_hoist.pre[order[k]];
    }
    if (ok)
        pn_tt_precondition_carry(p, in, planned);
    for (k = 0; ok && k < planned; k++)
        ok = pn_tt_precondition_push(p, (uint32_t)k, &in[k]) == 0;
    if (p)
        g_oncard.stats = *pn_tt_precondition_stats(p);
    pn_tt_precondition_destroy(p);
    ok = ok && !g_oncard.plan.oom;
    if (ok)
        td_oncard_dump(&config, queues, groups, in, planned);
    if (ok && !pn_tt_precondition_verify(in, planned, g_oncard.plan.a, g_oncard.plan.n, info,
                                         g_td.queue_count, groups, g_group_count,
                                         &g_oncard.report)) {
        fprintf(stderr, "pn_tt: DISPATCH — ONCARD plan refused by its check (%zu failures; "
                        "first: %s; held %zu, publish %zu)%s\n", g_oncard.report.failures,
                g_oncard.report.first, g_oncard.stats.by_held, g_oncard.stats.by_publish,
                config.red_early ? " — RED precondition_early runs it anyway" : "; the order "
                                                                               "stands");
        ok = config.red_early;
    }
    if (!ok)
        g_oncard.plan.n = 0u;
    free(in);
    return ok;
}

/** Reorder the entries from word `entries_from` to the end by the plan (the hoist, then S2's
 *  preconditions). Any refusal leaves the host's order (and its segment table) as emitted. */
static bool td_hoist_program(size_t entries_from)
{
    pn_tt_hoist_queue queues[TD_MAX_QUEUES];
    pn_tt_hoist_group groups[TD_MAX_GROUPS];
    const bool hoist = td_hoist_on() > 0, oncard = td_oncard_on();
    uint32_t *order = NULL;
    size_t planned = 0u, k;
    bool preconditioned = false;
    td_assembly as;

    g_hoist.planned = false;
    g_oncard.planned = false;
    /* The DRY run checks the hoisted program too: hoisting keeps each stream (writes, heads,
     * awaits) in the host's order, which is all the dry comparison reads. */
    if (!hoist && !oncard)
        return true;
    for (k = 0; k < g_group_count; k++) {
        memcpy(groups[k].queues, g_groups[k].queues, sizeof(groups[k].queues));
        groups[k].count = g_groups[k].count;
    }
    td_hoist_queues(queues);
    memset(&as, 0, sizeof(as));
    as.entries_from = entries_from;
    order = malloc(2u * (g_hoist.count + 1u) * sizeof(*order));
    /* room for a FLUSH and, under S2, an entry beside every noted item */
    as.words = malloc((g_prog.count - entries_from +
                       (1u + (oncard ? FT_TT_PULLER_PRECOND_WORDS : 0u)) * g_hoist.count + 1u) *
                      sizeof(*as.words));
    if (order && as.words) {
        if (hoist)
            planned = pn_tt_hoist_plan(g_hoist.items, g_hoist.count, queues, g_td.queue_count,
                                       groups, g_group_count, order, &g_hoist.stats);
        else
            for (planned = 0u; planned < g_hoist.count; planned++)
                order[planned] = (uint32_t)planned;
    }
    as.ok = planned != 0u;
    preconditioned = as.ok && oncard && td_oncard_plan(order, planned, queues, groups);
    for (k = 0; preconditioned && as.ok && k < g_oncard.plan.n; k++)
        td_assemble_action(&as, &g_oncard.plan.a[k], order);
    for (k = 0; !preconditioned && as.ok && k < planned; k++)
        td_assemble_item(&as, order[k]);
    if (as.ok && as.segments == g_prog.segments) {
        g_prog.count = entries_from;
        for (k = 0; k < as.at; k++)
            td_emit(as.words[k]);
        memcpy(g_prog.segment_offset, as.offsets, as.segments * sizeof(*as.offsets));
        g_hoist.planned = g_prog.ok && hoist;
        g_oncard.planned = g_prog.ok && preconditioned;
    } else {
        fprintf(stderr, "pn_tt: DISPATCH — HOIST plan refused (%zu entries, %u of %u segments); "
                        "the host's order stands\n", planned, as.segments, g_prog.segments);
    }
    free(order);
    free(as.words);
    return g_prog.ok;
}

/** S2: event `i` as the precondition planner sees it. `first` / `first_step` hold, per queue
 *  and then per multicast group, the first ring index staged since its last head (a head's
 *  first descriptor) and that index's per-token step. */
static pn_tt_pre_item td_pre_item(size_t i, uint32_t *first, uint32_t *first_step)
{
    const td_token *a = &g_td.model[0], *b = &g_td.model[1];
    const td_event *ea = &a->events[i], *eb = &b->events[i];
    const uint32_t *wa = &a->pool[ea->data_at], *wb = &b->pool[eb->data_at];
    const bool grouped = (ea->cls == TD_Q_RING || ea->cls == TD_Q_HEAD) && g_merge[i] >= 0;
    const uint32_t target = grouped ? TD_MAX_QUEUES + (uint32_t)g_merge[i] : ea->queue;
    pn_tt_pre_item it;

    memset(&it, 0, sizeof(it));
    it.kind = PN_TT_PRE_BARRIER; /* parameter cache, any other queue word: untracked */
    it.group = grouped ? (int16_t)g_merge[i] : -1;
    it.queue = grouped ? (uint16_t)g_groups[g_merge[i]].queues[0] : (uint16_t)ea->queue;
    switch (ea->cls) {
    case TD_Q_RING: {
        const uint32_t last = (ea->bytes / 32u - 1u) * 8u;

        it.kind = PN_TT_PRE_RING;
        it.index = ea->index;
        it.slots = ea->bytes / 32u;
        it.step = eb->index - ea->index;
        if ((wa[last + 5u] & 0x7FFFFFFFu) > 1u)
            it.pull_sequence = wa[last + 6u];
        it.group_sequence = wa[last + 7u];
        if (first[target] == UINT32_MAX) {
            first[target] = ea->index;
            first_step[target] = it.step;
        }
        break;
    }
    case TD_Q_HEAD:
        it.kind = PN_TT_PRE_HEAD;
        it.index = first[target];
        it.slots = first[target] == UINT32_MAX ? 0u : wa[0] - first[target];
        it.step = first_step[target];
        first[target] = UINT32_MAX;
        break;
    case TD_Q_POLL: {
        const uint32_t word = (0x14u - (ea->addr - FT_TT_OP_QUEUE_BASE)) / 4u;

        it.kind = PN_TT_PRE_AWAIT;
        it.index = wa[word];
        it.step = wb[word] - wa[word];
        break;
    }
    case TD_Q_PARAM_ROT:
        it.index = td_param_descriptor(i);
        if (it.index != UINT32_MAX) {
            it.kind = PN_TT_PRE_WRITE;
            it.address = ea->addr;
        }
        break;
    default:
        break;
    }
    return it;
}

/** Build the puller program from the two model tokens. @return false = refuse replay */
static bool td_build_program(void)
{
    const td_token *a = &g_td.model[0], *b = &g_td.model[1];
    static uint32_t first[TD_MAX_QUEUES + TD_MAX_GROUPS], first_step[TD_MAX_QUEUES + TD_MAX_GROUPS];
    static const pn_tt_pre_item segment_end = { .kind = PN_TT_PRE_SEGMENT, .group = -1 };
    pn_tt_pre_item pre;
    size_t queue_table, segment_table, group_table, entries_from, i;
    bool in_segment = false;
    uint32_t q;

    g_prog.count = 0u;
    g_prog.segments = 0u;
    g_prog.ok = true;
    for (i = 0; i < 16u; i++)
        td_emit(0u);
    queue_table = g_prog.count;
    for (q = 0; q < g_td.queue_count; q++) {
        td_emit((uint32_t)g_td.queues[q].x | ((uint32_t)g_td.queues[q].y << 16));
        td_emit(g_td.queues[q].nonce);
        td_emit(g_td.queues[q].mask);
        td_emit(0u);
    }
    segment_table = g_prog.count;
    for (i = 0; i < 256u; i++)
        td_emit(0u);
    if (!td_plan_merges())
        return false;
    group_table = g_prog.count;
    for (q = 0; q < g_group_count; q++) {
        td_emit(g_groups[q].queues[0]);
        td_emit(g_groups[q].x_lo);
        td_emit(g_groups[q].y_lo);
        td_emit(g_groups[q].x_hi);
        td_emit(g_groups[q].y_hi);
        td_emit(g_groups[q].count);
        td_emit(0u);
        td_emit(0u);
    }
    entries_from = g_prog.count;
    g_hoist.count = 0u;
    memset(first, 0xFF, sizeof(first));
    for (i = 0; i < a->count && g_prog.ok; i++) {
        const td_event *ea = &a->events[i], *eb = &b->events[i];
        const uint32_t *wa = &a->pool[ea->data_at], *wb = &b->pool[eb->data_at];
        const size_t entry_at = g_prog.count;

        if (td_is_boundary(ea->cls)) {
            if (in_segment) {
                td_emit(5u); /* segment end */
                td_hoist_note(i, entry_at, PN_TT_HOIST_SEGMENT, &segment_end);
            }
            in_segment = false;
            continue;
        }
        if (ea->cls == TD_Q_GROUP_R || ea->cls == TD_Q_OTHER_R)
            return false; /* a read the model has no meaning for */
        if (g_merge[i] == -2)
            continue; /* written by its group's multicast */
        if (!in_segment) {
            if (g_prog.segments == 256u)
                return false;
            g_prog.segment_offset[g_prog.segments++] = (uint32_t)(g_prog.count * 4u);
            in_segment = true;
        }
        switch (ea->cls) {
        case TD_Q_RING: {
            const uint32_t n = ea->bytes / 32u;
            const uint32_t patches = td_count_patches(wa, wb, 8u * n, 1);
            uint32_t w8;

            for (w8 = 0; w8 < 8u * n; w8++)
                if ((w8 % 8u) != 1u && wa[w8] != wb[w8])
                    g_ring_patch_word[w8 % 8u]++;

            td_emit(2u | (n << 8) | (patches << 20));
            td_emit(g_merge[i] >= 0 ? 0x80000000u | (uint32_t)g_merge[i] : ea->queue);
            td_emit(ea->index);
            td_emit(eb->index - ea->index);
            td_emit_words(wa, wb, 8u * n, 1);
            break;
        }
        case TD_Q_HEAD:
            td_emit(3u);
            td_emit(g_merge[i] >= 0 ? 0x80000000u | (uint32_t)g_merge[i] : ea->queue);
            td_emit(wa[0]);
            td_emit(wb[0] - wa[0]);
            break;
        case TD_Q_POLL: {
            const uint32_t offset = ea->addr - FT_TT_OP_QUEUE_BASE;
            uint32_t word;

            if (offset > 0x14u || offset + ea->bytes <= 0x14u)
                break; /* not a count read */
            word = (0x14u - offset) / 4u;
            td_emit(4u);
            td_emit(ea->queue);
            td_emit(wa[word]);
            td_emit(wb[word] - wa[word]);
            break;
        }
        default: { /* parameter blocks and any other queue-region write */
            const uint32_t n = ea->bytes / 4u;
            const uint32_t patches = td_count_patches(wa, wb, n, 0);

            if (ea->bytes % 4u != 0u || n > 0xFFFu)
                return false;
            td_emit(1u | (n << 8) | (patches << 20));
            td_emit((uint32_t)ea->x | ((uint32_t)ea->y << 16));
            td_emit(ea->addr);
            td_emit_words(wa, wb, n, 0);
            break;
        }
        }
        pre = td_pre_item(i, first, first_step);
        if (g_prog.count != entry_at)
            td_hoist_note(i, entry_at, 0u, &pre);
    }
    if (in_segment) {
        const size_t entry_at = g_prog.count;

        td_emit(5u);
        td_hoist_note(a->count, entry_at, PN_TT_HOIST_SEGMENT, &segment_end);
    }
    if (g_prog.ok && !td_hoist_program(entries_from))
        return false;
    td_emit(6u);
    if (!g_prog.ok)
        return false;
    for (i = 0; i < g_prog.segments; i++)
        g_prog.words[segment_table + i] = g_prog.segment_offset[i];
    g_prog.words[0] = TD_PULL_MAGIC;
    g_prog.words[1] = 1u;
    g_prog.words[2] = g_td.queue_count;
    g_prog.words[3] = (uint32_t)(queue_table * 4u);
    g_prog.words[4] = g_prog.segments;
    g_prog.words[5] = (uint32_t)(segment_table * 4u);
    g_prog.words[14] = g_group_count;
    g_prog.words[15] = (uint32_t)(group_table * 4u);
    /* The log after the program, 16-byte aligned, up to the scratch. */
    g_prog.words[6] = (uint32_t)((TD_PROGRAM_BASE + g_prog.count * 4u + 15u) & ~15u);
    if (g_prog.words[6] >= TD_PROGRAM_LIMIT)
        return false;
    g_prog.words[7] = TD_PROGRAM_LIMIT - g_prog.words[6];
    while (g_prog.count % 4u)
        td_emit(0u);
    return g_prog.ok;
}

static bool td_puller_arm(pn_tt_device *device)
{
    const uint32_t core = pn_tt_env_u32("PN_TT_PULLER_CORE", TD_PULLER_DEFAULT_CORE);

    if (device->puller.running)
        return true;
    if (!pn_tt_prepare_family_on_core(device, &device->puller, "puller", "puller",
                                      "PN_TT_PULLER_CORE", core,
                                      (1u << FT_TT_PROC_DM0) | (1u << FT_TT_PROC_DM1)) ||
        pn_tt_residency_start(&device->puller) != 0) {
        fprintf(stderr, "pn_tt: DISPATCH — the puller did not arm on core %u\n", core);
        return false;
    }
    fprintf(stderr, "pn_tt: DISPATCH — puller residency up on core %u\n", core);
    return true;
}

static void ts_table_forget(void);

static bool td_upload_program(pn_tt_device *device)
{
    const ft_tt_noc_access *bulk = &device->runtime.bulk_l1;

    ts_table_forget(); /* the recorded program may run over the shared queue table */
    if (bulk->write(bulk->user, device->puller.core_x, device->puller.core_y, TD_PROGRAM_BASE,
                    g_prog.words, g_prog.count * 4u) != FT_TT_OK) {
        fprintf(stderr, "pn_tt: DISPATCH — the program did not upload\n");
        return false;
    }
    fprintf(stderr, "pn_tt: DISPATCH — ring words that move per token: opcode %llu a0 %llu "
                    "a1 %llu a2 %llu a3 %llu a4 %llu a5 %llu\n",
            (unsigned long long)g_ring_patch_word[0], (unsigned long long)g_ring_patch_word[2],
            (unsigned long long)g_ring_patch_word[3], (unsigned long long)g_ring_patch_word[4],
            (unsigned long long)g_ring_patch_word[5], (unsigned long long)g_ring_patch_word[6],
            (unsigned long long)g_ring_patch_word[7]);
    fprintf(stderr, "pn_tt: DISPATCH — MULTICAST %u groups; per token %llu ring writes and %llu "
                    "heads go out as group multicasts\n",
            g_group_count, (unsigned long long)g_merged_rings,
            (unsigned long long)g_merged_heads);
    {
        uint32_t gi;

        for (gi = 0; gi < g_group_count && gi < 8u; gi++)
            fprintf(stderr, "pn_tt: DISPATCH —   group %u: (%u,%u)..(%u,%u), %u cores\n", gi,
                    g_groups[gi].x_lo, g_groups[gi].y_lo, g_groups[gi].x_hi, g_groups[gi].y_hi,
                    g_groups[gi].count);
    }
    fprintf(stderr, "pn_tt: DISPATCH — program %zu B, %u queues, %u segments, log at 0x%x "
                    "(%u B)\n",
            g_prog.count * 4u, g_td.queue_count, g_prog.segments, g_prog.words[6],
            g_prog.words[7]);
    if (g_hoist.planned)
        fprintf(stderr, "pn_tt: DISPATCH — HOIST: %zu writes moved under an await, %zu left in "
                        "place, %zu flushes\n",
                g_hoist.stats.moved, g_hoist.stats.blocked, g_hoist.stats.flushes);
    if (g_oncard.planned)
        fprintf(stderr, "pn_tt: DISPATCH — ONCARD (lookahead %u): %zu awaits a token, %zu "
                        "converted to %zu sidecar entries and realized a window late, %zu kept "
                        "in place (realized by write %zu, barrier %zu, segment %zu, run %zu, "
                        "chain %zu, publish %zu, held %zu); check: %zu heads, %zu writes, "
                        "%zu segments, %zu failures%s\n",
                td_oncard_lookahead(), g_oncard.stats.awaits_in, g_oncard.stats.windows,
                g_oncard.stats.preconds,
                g_oncard.stats.awaits_in - g_oncard.stats.merged - g_oncard.stats.windows,
                g_oncard.stats.by_write, g_oncard.stats.by_barrier, g_oncard.stats.by_segment,
                g_oncard.stats.by_run, g_oncard.stats.by_chain, g_oncard.stats.by_publish,
                g_oncard.stats.by_held,
                g_oncard.report.heads,
                g_oncard.report.writes, g_oncard.report.segments, g_oncard.report.failures,
                pn_tt_red("precondition_early") ? " [RED precondition_early]" : "");
    return true;
}

static uint64_t g_dry_checked, g_dry_failed, g_dry_words;

/** S2 dry run: the log's AWAIT (4) and PRECOND records, in program order, against the plan's
 *  realized AWAITs and sidecar entries at token offset k. (The host's polls themselves are the
 *  shadow's to check: td_shadow_check compares each against the model.) */
static bool td_dry_check_oncard(const uint32_t *log, size_t log_words, uint32_t k,
                                uint64_t forward)
{
    size_t at = 0u, j = 0u;

    for (; at + 4u <= log_words; at += 4u + log[at + 3u]) {
        const uint32_t type = log[at];
        const pn_tt_pre_action *a;
        uint32_t want[4], xy, addr = 0u, n = 1u, w;

        if (type != 4u && type != FT_TT_PULLER_ENTRY_PRECOND)
            continue;
        while (j < g_oncard.plan.n && g_oncard.plan.a[j].kind == PN_TT_PRE_OUT_ITEM)
            j++;
        a = j < g_oncard.plan.n ? &g_oncard.plan.a[j++] : NULL;
        if (!a || (type == 4u) != (a->kind == PN_TT_PRE_OUT_AWAIT)) {
            fprintf(stderr, "pn_tt: DISPATCH DRY forward %llu: log record type %u at word %zu is "
                            "not the plan's next await / entry\n", (unsigned long long)forward,
                    type, at);
            return false;
        }
        xy = (uint32_t)g_td.queues[a->queue].x | ((uint32_t)g_td.queues[a->queue].y << 16);
        want[0] = a->count + a->count_step * k;
        if (type != 4u) {
            const td_queue *target = &g_td.queues[a->group >= 0 ? g_groups[a->group].queues[0]
                                                                : a->queue];
            const td_queue *producer = &g_td.queues[a->producer];

            want[0] = a->index + a->index_step * k;
            want[1] = ftq_pre_xy_word(producer->x, producer->y);
            want[2] = a->count + a->count_step * k;
            want[3] = producer->nonce;
            addr = FT_TT_OP_QUEUE_BASE + ftq_pre_entry_offset(want[0], target->mask);
            xy = a->group >= 0 ? 0x80000000u | (uint32_t)a->group : xy;
            n = 4u;
        }
        if (log[at + 1u] != xy || log[at + 3u] != n || (type != 4u && log[at + 2u] != addr)) {
            fprintf(stderr, "pn_tt: DISPATCH DRY forward %llu: %s record at word %zu names xy 0x%x "
                            "addr 0x%x n %u, the plan xy 0x%x addr 0x%x n %u\n",
                    (unsigned long long)forward, type == 4u ? "await" : "entry", at, log[at + 1u],
                    log[at + 2u], log[at + 3u], xy, addr, n);
            return false;
        }
        for (w = 0; w < n; w++)
            if (log[at + 4u + w] != want[w]) {
                fprintf(stderr, "pn_tt: DISPATCH DRY forward %llu: %s record at word %zu word %u: "
                                "the puller 0x%08x, the plan 0x%08x\n",
                        (unsigned long long)forward, type == 4u ? "await" : "entry", at, w,
                        log[at + 4u + w], want[w]);
                return false;
            }
    }
    while (j < g_oncard.plan.n && g_oncard.plan.a[j].kind == PN_TT_PRE_OUT_ITEM)
        j++;
    if (j != g_oncard.plan.n) {
        fprintf(stderr, "pn_tt: DISPATCH DRY forward %llu: the log ran out before the plan's "
                        "await / entry %zu\n", (unsigned long long)forward, j);
        return false;
    }
    return true;
}

/** Run the whole token k on the puller in DRY-RUN mode and compare its log, stream for
 *  stream (non-head writes, heads, awaits), with what the host really did this token. */
static void td_dry_check(pn_tt_device *device, uint64_t forward)
{
    const ft_tt_noc_access *bulk = &device->runtime.bulk_l1;
    const uint32_t k = (uint32_t)(forward - g_td.first_forward);
    const td_token *t = &g_td.current;
    uint32_t params[FT_TT_QUEUE_PARAM_WORDS];
    uint32_t header[16], *log = NULL;
    size_t i, at_w = 0u, at_h = 0u, at_a = 0u, log_words;
    size_t heads_at[2] = { 0u, 0u };
    bool bad = false;
    (void)heads_at;

    memset(params, 0, sizeof(params));
    params[0] = TD_PROGRAM_BASE;
    params[1] = pn_tt_red("dispatch_dry_k") ? k + 1u : k;
    params[2] = 0u;
    params[3] = g_prog.segments - 1u;
    params[4] = 1u;
    g_dry_checked++;
    if (pn_tt_residency_submit(&device->puller, TD_PULL_OPCODE, params, 5000u) != 0 ||
        bulk->read(bulk->user, device->puller.core_x, device->puller.core_y, TD_PROGRAM_BASE,
                   header, sizeof(header)) != FT_TT_OK) {
        g_dry_failed++;
        fprintf(stderr, "pn_tt: DISPATCH DRY forward %llu — the puller did not complete\n",
                (unsigned long long)forward);
        return;
    }
    if (header[8] != 0u) {
        g_dry_failed++;
        fprintf(stderr, "pn_tt: DISPATCH DRY forward %llu — puller error %u at entry +0x%x\n",
                (unsigned long long)forward, header[8], header[9]);
        return;
    }
    log_words = header[10] / 4u;
    log = malloc((log_words + 4u) * 4u);
    if (!log || bulk->read(bulk->user, device->puller.core_x, device->puller.core_y,
                           g_prog.words[6], log, ((header[10] + 15u) & ~15u)) != FT_TT_OK) {
        free(log);
        g_dry_failed++;
        return;
    }
    g_dry_words += log_words;
    /* Three cursors into the log, one per stream; each advances past records of its type. */
    for (i = 0; i < t->count && !bad; i++) {
        const td_event *e = &t->events[i];
        const uint32_t *w = &t->pool[e->data_at];
        size_t *cursor;
        uint32_t want_type, n, xy = (uint32_t)e->x | ((uint32_t)e->y << 16), word;

        if (td_is_boundary(e->cls) || e->cls == TD_Q_GROUP_R || e->cls == TD_Q_OTHER_R)
            continue;
        if (e->cls == TD_Q_POLL && g_oncard.planned)
            continue; /* S2 moved the awaits: td_dry_check_oncard() reads them */
        /* STEP 3 B: a multicast group's ring slot / head is ONE log record naming the group
         * (bit 31 | group); the members' own events were merged into the leader's. */
        if (g_merge && (e->cls == TD_Q_RING || e->cls == TD_Q_HEAD)) {
            if (g_merge[i] == -2)
                continue;
            if (g_merge[i] >= 0)
                xy = 0x80000000u | (uint32_t)g_merge[i];
        }
        if (e->cls == TD_Q_POLL) {
            const uint32_t offset = e->addr - FT_TT_OP_QUEUE_BASE;

            if (offset > 0x14u || offset + e->bytes <= 0x14u)
                continue;
            want_type = 4u;
            cursor = &at_a;
            w += (0x14u - offset) / 4u;
            n = 1u;
        } else if (e->cls == TD_Q_HEAD) {
            want_type = 3u;
            cursor = &at_h;
            n = 1u;
        } else {
            want_type = e->cls == TD_Q_RING ? 2u : 1u;
            cursor = &at_w;
            n = e->bytes / 4u;
        }
        while (*cursor < log_words) {
            const uint32_t type = log[*cursor];
            const bool match_stream = (want_type == 4u && type == 4u) ||
                                      (want_type == 3u && type == 3u) ||
                                      (want_type <= 2u && type <= 2u);

            if (match_stream)
                break;
            *cursor += 4u + log[*cursor + 3u];
        }
        if (*cursor >= log_words) {
            bad = true;
            fprintf(stderr, "pn_tt: DISPATCH DRY forward %llu event %zu: the log ran out\n",
                    (unsigned long long)forward, i);
            break;
        }
        if (log[*cursor] != want_type || log[*cursor + 1u] != xy ||
            (want_type != 4u && log[*cursor + 2u] != e->addr) || log[*cursor + 3u] != n) {
            bad = true;
            fprintf(stderr, "pn_tt: DISPATCH DRY forward %llu event %zu %s: log has type %u "
                            "xy 0x%x addr 0x%x n %u, the host did xy 0x%x addr 0x%x n %u\n",
                    (unsigned long long)forward, i, td_class_name[e->cls], log[*cursor],
                    log[*cursor + 1u], log[*cursor + 2u], log[*cursor + 3u], xy, e->addr, n);
            break;
        }
        for (word = 0; word < n; word++)
            if (log[*cursor + 4u + word] != w[word]) {
                bad = true;
                fprintf(stderr, "pn_tt: DISPATCH DRY forward %llu event %zu %s word %u: the "
                                "puller 0x%08x, the host 0x%08x\n",
                        (unsigned long long)forward, i, td_class_name[e->cls], word,
                        log[*cursor + 4u + word], w[word]);
                break;
            }
        *cursor += 4u + n;
    }
    if (!bad && g_oncard.planned)
        bad = !td_dry_check_oncard(log, log_words, k, forward);
    free(log);
    if (bad)
        g_dry_failed++;
}

/* ---------------------------------------------------------------------------------------
 * STEP 4 S2: the puller's data movers for the speculative verify
 * --------------------------------------------------------------------------------------- */

#define TD_ROWCOPY_OPCODE 0x15u
#define TD_FILL_OPCODE    0x16u
#define TD_COPY_OPCODE    0x17u
/* One FILL descriptor's ceiling: the kernel's `bytes` is 32-bit; 256 MiB keeps each one
 *  well inside the submit timeout. */
#define TD_FILL_MAX_BYTES (256u << 20)

/* S2 data-mover accounting (pn_tt_move_stats). */
static struct { uint64_t rows; double row_ms; } g_move;

void pn_tt_move_stats(uint64_t *rows, double *row_ms)
{
    *rows = g_move.rows;
    *row_ms = g_move.row_ms;
}

/* ---------------------------------------------------------------------------------------
 * S2d: STREAMED SEGMENTS. No model: while streaming, the host runs its code exactly as
 * always, but its queue writes (parameter blocks, ring slots, heads), its DRAM writes and its
 * aligned L1 writes are CAPTURED as literal puller entries, and its completion polls are
 * answered at once (count = the queue's last head, token from the queue's nonce) while an
 * AWAIT entry makes the puller wait for the same on the card. A host READ of data (or
 * anything the stream cannot express) is a boundary: the captured segment is uploaded to the
 * puller in one write, run, awaited, and the access then goes to the card for real. So the
 * card's await->issue latency inside a segment is the puller's, not the host's round trip.
 * --------------------------------------------------------------------------------------- */

/* The program arena is cut into TS_REGIONS regions; a captured chunk is launched into the
 * next one WITHOUT waiting once it reaches TS_CHUNK_WORDS (the card runs it while the host
 * captures the next), so only a host data read waits. A region is reused only after the
 * PULL descriptor that ran it has completed. */
#define TS_REGIONS 12u
#define TS_INFLIGHT 6u
#define TS_MAX_QUEUES 256u
/* LAUNCH lane: the SHARED queue table (`PN_TT_STREAM_SHARED_TABLE`) sits above the regions, one
 * 4-word row a queue. */
#define TS_TABLE_BYTES (TS_MAX_QUEUES * 16u)
#define TS_TABLE_BASE (TD_PROGRAM_LIMIT - TS_TABLE_BYTES)
#define TS_REGION_WORDS (((TS_TABLE_BASE - TD_PROGRAM_BASE) / 4u / TS_REGIONS) & ~3u)
#define TS_ENTRY_BASE (16u + 4u * TS_MAX_QUEUES + 4u)
#define TS_MAX_WORDS TS_REGION_WORDS
/* A chunk launches early (`PN_TT_STREAM_CHUNK` words, default 512: run_s2d_chunk.sh) so the card starts while
 * the host captures the rest. */
static uint32_t ts_chunk_words(void)
{
    static uint32_t chunk;

    if (chunk == 0u) {
        const char *env = getenv("PN_TT_STREAM_CHUNK");
        const uint32_t cap = (uint32_t)(TS_REGION_WORDS - TS_ENTRY_BASE - 4096u);

        chunk = env && atoi(env) > 0 ? (uint32_t)atoi(env) : 512u;
        if (chunk > cap)
            chunk = cap;
    }
    return chunk;
}
/* After a boundary the card is idle: launch the FIRST chunk sooner (`PN_TT_STREAM_FIRST`
 * words, default = the chunk size). */
static bool g_ts_fresh;
/* HOST GAP: why the next full flush leaves the card idle (pn_tt_host_gap.h). */
static const char *g_ts_gap_why = "flush";
static uint32_t ts_first_words(void)
{
    static uint32_t first;

    if (first == 0u) {
        const char *env = getenv("PN_TT_STREAM_FIRST");

        first = env && atoi(env) > 0 ? (uint32_t)atoi(env) : ts_chunk_words();
    }
    return first;
}
#define TS_CHUNK_WORDS (g_ts_fresh ? ts_first_words() : ts_chunk_words())

typedef struct {
    uint16_t x, y;
    uint32_t nonce;
    bool nonce_known;
    uint32_t issued;     /* the last head written (or read) */
    bool issued_known;
    uint32_t ring_token; /* the token word of the last ring slot written since the head */
    bool ring_pending;
    uint32_t heartbeat;
    /* L10 B2 (hoisting in the stream): a head captured in THIS stream makes `issued` exact;
     * slots staged since it give a write's descriptor index; the ring capacity, read once a
     * nonce (a re-arm is a new nonce). */
    bool fresh;
    uint32_t staged;
    uint32_t capacity, capacity_nonce;
} ts_queue;

static struct {
    pn_tt_device *device;
    uint32_t *words; /* entries at TS_ENTRY_BASE (the capture buffer) */
    uint32_t *out;   /* the compact program a launch uploads */
    size_t count;    /* entry words */
    ts_queue queues[TS_MAX_QUEUES];
    uint32_t queue_count;
    bool failed;
    uint32_t region;            /* the next region to launch into */
    uint32_t region_done_at[TS_REGIONS]; /* puller issued count that ran the region (0 free) */
    uint32_t launched, completed; /* PULLs launched / known complete (puller counts) */
    uint64_t segments, entries, boundaries, polls_answered, launched_words, async_launches;
    double flush_ms, upload_ms, capture_ms, wait_ms;
    /* S2e census: entries by kind, words by kind, and WRITEs a recent one duplicates
     * (same address, bytes and data, another core): multicast candidates. */
    uint64_t n_param, n_ring, n_dram, n_l1, n_head, n_await, n_rowcopy;
    uint64_t w_param, w_ring, w_dram, w_l1, dup_writes, dup_words, dup_same_row;
    struct { uint16_t x, y; uint32_t addr, bytes; uint64_t hash; } recent[32];
    uint32_t recent_at;
    double last_flush_end;
    uint32_t puller_busy0, puller_await0; /* control words 62/63 at stream begin */
    uint64_t bound_by_class[TD_CLASSES], bound_bytes[TD_CLASSES];
    double bound_ms_by_class[TD_CLASSES];
    struct { uint16_t x, y; uint32_t addr, bytes; uint64_t n; double ms; } l1r[16];
    uint32_t l1r_count;
    double bound_drain_ms, bound_access_ms;
    double puller_busy_ms, puller_await_ms;
    /* PM 8 AWAIT census (`PN_TT_PULLER_AWAIT_CENSUS=1`): the puller's per-queue table read at
     * stream begin and end, the differences summed here by queue index. */
    uint32_t census0[2u * TS_MAX_QUEUES];
    double census_ms[TS_MAX_QUEUES];
    uint64_t census_awaits[TS_MAX_QUEUES];
    /* L10 B2: the hoist in the stream (`PN_TT_STREAM_HOIST`) */
    pn_tt_hoist_tracker *hoist;
    int64_t await_at;          /* the chunk's last AWAIT entry (word offset), -1 none */
    bool after_await;          /* a non-HEAD entry follows it: nothing may move above it */
    bool flushed;              /* a FLUSH already stands before it */
    size_t await_undo_from, await_undo_to;
    uint64_t hoisted, hoist_blocked, hoist_flushes;
    int32_t leader_of[TS_MAX_QUEUES];
    uint32_t leader_queues;    /* queue_count leader_of was computed for */
    /* LAUNCH lane: the shared queue table — rows on the puller (0 = none since the stream
     * began), the nonce each row carries, and the rows written / table drains a run. */
    uint32_t table_rows;
    uint32_t table_nonce[TS_MAX_QUEUES];
    uint64_t table_writes, table_drains, table_words_saved;
    /* S2 (pn_tt_precondition.h) in a decode-time stream: the planner, each queue's ring slots
     * captured since its last head, the queues it was told about, the last AWAIT pushed (a
     * poll loop is one AWAIT), and under `PN_TT_ONCARD_VERIFY=1` the stream's items and
     * actions for the check at its end. */
    pn_tt_pre_planner *pre;
    uint32_t pre_items;
    uint32_t pre_staged[TS_MAX_QUEUES];
    bool pre_set[TS_MAX_QUEUES];
    pn_tt_pre_queue_info pre_info[TS_MAX_QUEUES];
    bool pre_last_await;
    uint32_t pre_last_q, pre_last_count;
    uint64_t n_precond;
    bool pre_verify;
    pn_tt_pre_item *vlog;
    size_t vlog_n, vlog_cap;
    td_oncard_actions vplan;
    uint64_t verified, verify_failed;
} g_ts;

/* The puller kernel's PULL_AWAIT_TABLE (resident_puller.c). */
#define TS_AWAIT_TABLE 0x16D000u
/* The puller kernel's PULL_POLL_BUFFER (resident_puller.c): the line its last AWAIT read. */
#define TS_POLL_BUFFER 0x16C000u

static bool ts_census_enabled(void)
{
    static int enabled = -1;

    if (enabled < 0) {
        const char *env = getenv("PN_TT_PULLER_AWAIT_CENSUS");

        enabled = env && atoi(env) != 0;
    }
    return enabled == 1;
}

static void ts_census_read(uint32_t *table)
{
    const ft_tt_noc_access *noc = &g_td.original[TD_ACC_NOC];
    pn_tt_residency *puller = &g_ts.device->puller;

    (void)noc->read(noc->user, puller->core_x, puller->core_y, TS_AWAIT_TABLE, table,
                    2u * TS_MAX_QUEUES * sizeof(uint32_t));
}

/** Add the puller's per-queue AWAIT time since the stream began to the census. */
static void ts_census_accumulate(void)
{
    uint32_t now[2u * TS_MAX_QUEUES];
    uint32_t q;

    ts_census_read(now);
    for (q = 0; q < TS_MAX_QUEUES; q++) {
        g_ts.census_ms[q] += (double)(uint32_t)(now[2u * q] - g_ts.census0[2u * q]) * 16.0 /
                             1.35e6;
        g_ts.census_awaits[q] += (uint32_t)(now[2u * q + 1u] - g_ts.census0[2u * q + 1u]);
    }
}

void pn_tt_dispatch_census_restart(void)
{
    memset(g_ts.census_ms, 0, sizeof(g_ts.census_ms));
    memset(g_ts.census_awaits, 0, sizeof(g_ts.census_awaits));
}

/** Print the census by family (the core's label), totals since the last restart. */
static void ts_census_report(void)
{
    double total = 0.0;
    uint32_t q;

    if (!ts_census_enabled())
        return;
    for (q = 0; q < g_ts.queue_count; q++)
        total += g_ts.census_ms[q];
    fprintf(stderr, "pn_tt: AWAIT CENSUS — %.1f ms the puller spent in AWAIT polls, by queue:\n",
            total);
    for (q = 0; q < g_ts.queue_count; q++)
        if (g_ts.census_awaits[q] != 0u)
            fprintf(stderr, "pn_tt:   await (%2u,%2u) %-22s %8.1f ms over %7llu awaits "
                            "(%.2f us each)\n",
                    g_ts.queues[q].x, g_ts.queues[q].y,
                    pn_tt_core_label(g_ts.queues[q].x, g_ts.queues[q].y), g_ts.census_ms[q],
                    (unsigned long long)g_ts.census_awaits[q],
                    g_ts.census_ms[q] * 1000.0 / (double)g_ts.census_awaits[q]);
}

void pn_tt_dispatch_census_print(void)
{
    ts_census_report();
}

void pn_tt_dispatch_report_await_retries(pn_tt_device *device)
{
    const ft_tt_noc_access *noc;
    uint32_t words[4];

    if (!device || !device->puller.running)
        return;
    noc = g_td.wrapped ? &g_td.original[TD_ACC_NOC] : td_accessor(device, TD_ACC_NOC);
    if (!noc->read || noc->read(noc->user, device->puller.core_x, device->puller.core_y,
                                FT_TT_PULLER_AWAIT_RETRIES, words, sizeof(words)) != FT_TT_OK)
        return;
    fprintf(stderr, "pn_tt: puller AWAIT retries — %u mismatching poll(s) read again over %u "
                    "AWAIT(s) since the puller armed\n",
            words[0], words[1]);
}

/* LAUNCH lane: (x, y) -> queue index + 1 (0 = none yet), so a captured write finds its
 * queue in one load instead of a scan of ~140 (three lookups an issued descriptor). Valid
 * while `g_ts_queue_at_count` == g_ts.queue_count; a reset of the queue list empties it. */
static uint16_t g_ts_queue_at[64][64];
static uint32_t g_ts_queue_at_count;

static uint32_t ts_queue_of(uint16_t x, uint16_t y)
{
    uint32_t i;

    if (g_ts_queue_at_count != g_ts.queue_count) {
        memset(g_ts_queue_at, 0, sizeof(g_ts_queue_at));
        for (i = 0; i < g_ts.queue_count; i++)
            if (g_ts.queues[i].x < 64u && g_ts.queues[i].y < 64u)
                g_ts_queue_at[g_ts.queues[i].x][g_ts.queues[i].y] = (uint16_t)(i + 1u);
        g_ts_queue_at_count = g_ts.queue_count;
    }
    if (x < 64u && y < 64u && g_ts_queue_at[x][y] != 0u)
        return g_ts_queue_at[x][y] - 1u;
    for (i = 0; i < g_ts.queue_count; i++)
        if (g_ts.queues[i].x == x && g_ts.queues[i].y == y)
            return i;
    if (g_ts.queue_count == TS_MAX_QUEUES)
        return TS_MAX_QUEUES;
    memset(&g_ts.queues[i], 0, sizeof(g_ts.queues[i]));
    g_ts.queues[i].x = x;
    g_ts.queues[i].y = y;
    if (x < 64u && y < 64u)
        g_ts_queue_at[x][y] = (uint16_t)(i + 1u);
    g_ts_queue_at_count = ++g_ts.queue_count;
    return i;
}

static bool ts_room(size_t words)
{
    return g_ts.words && TS_ENTRY_BASE + g_ts.count + words + 2u <= TS_MAX_WORDS;
}

static bool ts_flush(void);
static bool ts_launch(bool wait);
static bool ts_pre_push_kind(uint8_t kind);

static bool ts_emit(const uint32_t *w, size_t n)
{
    if ((!ts_room(n) || g_ts.count >= TS_CHUNK_WORDS) && !ts_launch(false))
        return false;
    if (!ts_room(n))
        return false;
    memcpy(&g_ts.words[TS_ENTRY_BASE + g_ts.count], w, n * 4u);
    g_ts.count += n;
    g_ts.entries++;
    return true;
}

/** The line the puller's failed AWAIT read last, its token named against the stream's nonce:
 *  a token one ahead of the count is a read between a writer's token and count stores. */
static void ts_report_poll_line(uint32_t q)
{
    const ft_tt_noc_access *bulk = &g_td.original[TD_ACC_BULK_L1];
    const pn_tt_residency *puller = &g_ts.device->puller;
    const uint32_t nonce = g_ts.queues[q].nonce;
    uint32_t line[4];
    const char *token = "unknown: the stream did not know the nonce";

    if (bulk->read(bulk->user, puller->core_x, puller->core_y, TS_POLL_BUFFER, line,
                   sizeof(line)) != FT_TT_OK)
        return;
    if (g_ts.queues[q].nonce_known && line[1] != 0u)
        token = line[2] == td_token_at(nonce, line[1] - 1u) ? "count - 1's (consistent)"
                : line[2] == td_token_at(nonce, line[1])
                    ? "count's: read between the writer's token and count stores"
                    : "neither: another nonce";
    fprintf(stderr, "pn_tt: STREAM STUCK —     the puller's last line read: count %u token %08x "
                    "state 0x%x — the token is %s\n",
            line[1], line[2], line[3], token);
}

/** Wait until the puller has completed `target` PULLs (its completed count, read directly). */
/**
 * A STUCK PULLER, named (q38-tt-regression-bisect: two 28k batched prefills aborted after a
 * silent 20 s wait on the puller, the grid idled out behind it). For every region the puller
 * has entered: its error word, the entry it is on (`header[9]` is written as each entry
 * starts), the last count an AWAIT read and the entry's first words; for an AWAIT entry
 * (type 4: queue xy, wanted) the awaited core's control block too. Read only.
 */
static void ts_report_stuck_puller(const char *where, uint32_t target)
{
    pn_tt_residency *puller = &g_ts.device->puller;
    const ft_tt_noc_access *bulk = &g_td.original[TD_ACC_BULK_L1];
    uint32_t fields[8], header[16], entry[4], control[8];

    if (bulk->read(bulk->user, puller->core_x, puller->core_y, FT_TT_OP_QUEUE_BASE, fields,
                   sizeof(fields)) == FT_TT_OK)
        fprintf(stderr, "pn_tt: STREAM STUCK (%s) — puller (%u,%u) head %u tail %u completed %u "
                        "state 0x%x; waited for PULL %u, %u launched\n",
                where, puller->core_x, puller->core_y, fields[2], fields[3], fields[5], fields[7],
                target, g_ts.launched);
    for (uint32_t r = 0; r < TS_REGIONS; r++) {
        const uint32_t base = TD_PROGRAM_BASE + r * TS_REGION_WORDS * 4u;

        if (bulk->read(bulk->user, puller->core_x, puller->core_y, base, header,
                       sizeof(header)) != FT_TT_OK || header[9] == 0u)
            continue;
        if (bulk->read(bulk->user, puller->core_x, puller->core_y, base + header[9], entry,
                       sizeof(entry)) != FT_TT_OK)
            memset(entry, 0, sizeof(entry));
        fprintf(stderr, "pn_tt: STREAM STUCK —   region %u%s: error %u, entry +0x%x = %08x %08x "
                        "%08x %08x, last await read %u\n",
                r, r == (g_ts.region + TS_REGIONS - 1u) % TS_REGIONS ? " (last launched)" : "",
                header[8], header[9], entry[0], entry[1], entry[2], entry[3], header[11]);
        /* An AWAIT's word 1 is the queue's TABLE INDEX (ts_emit_await), not its xy. */
        if ((entry[0] & 0xFFu) == 4u && entry[1] < g_ts.queue_count &&
            bulk->read(bulk->user, g_ts.queues[entry[1]].x, g_ts.queues[entry[1]].y,
                       FT_TT_OP_QUEUE_BASE, control, sizeof(control)) == FT_TT_OK)
            fprintf(stderr, "pn_tt: STREAM STUCK —     AWAIT queue %u core (%u,%u) %s for %u: head "
                            "%u tail %u completed %u state 0x%x; the stream knew nonce %08x "
                            "(%s), issued %u (%s)\n",
                    entry[1], g_ts.queues[entry[1]].x, g_ts.queues[entry[1]].y,
                    pn_tt_core_label(g_ts.queues[entry[1]].x, g_ts.queues[entry[1]].y),
                    entry[2], control[2], control[3], control[5], control[7],
                    g_ts.queues[entry[1]].nonce,
                    g_ts.queues[entry[1]].nonce_known ? "known" : "unknown",
                    g_ts.queues[entry[1]].issued,
                    g_ts.queues[entry[1]].issued_known ? "known" : "unknown");
        if ((entry[0] & 0xFFu) == 4u && header[8] != 0u && entry[1] < g_ts.queue_count)
            ts_report_poll_line(entry[1]);
    }
}

/**
 * The FIRST stall, named (q38-tt-grid-abort): a puller that has not finished `target` after 4 s,
 * before the cores it starves reach their own idle exit (~8 s). The 20 s dump comes after
 * that exit and names the AWAIT on the dead cores, not the one that stalled first. Once a
 * process (a long stall prints once); read only.
 */
static void ts_report_early_stall(const char *where, uint32_t target)
{
    static bool reported;
    char tag[64];

    if (reported)
        return;
    reported = true;
    snprintf(tag, sizeof(tag), "%s, EARLY at 4 s", where);
    ts_report_stuck_puller(tag, target);
    {
        const ft_tt_noc_access *bulk = &g_td.original[TD_ACC_BULK_L1];
        uint32_t control[8], shown = 0u;

        for (uint32_t q = 0; q < g_ts.queue_count && shown < 32u; q++) {
            if (bulk->read(bulk->user, g_ts.queues[q].x, g_ts.queues[q].y, FT_TT_OP_QUEUE_BASE,
                           control, sizeof(control)) != FT_TT_OK)
                continue;
            if (control[5] >= control[2] && control[7] != (uint32_t)FT_TT_QUEUE_STATE_EXIT)
                continue; /* caught up and alive */
            shown++;
            fprintf(stderr, "pn_tt: STREAM STUCK —     queue %u core (%u,%u) %s: head %u tail %u "
                            "completed %u state 0x%x; the stream knew issued %u (%s)\n",
                    q, g_ts.queues[q].x, g_ts.queues[q].y,
                    pn_tt_core_label(g_ts.queues[q].x, g_ts.queues[q].y), control[2], control[3],
                    control[5], control[7], g_ts.queues[q].issued,
                    g_ts.queues[q].issued_known ? "known" : "unknown");
        }
        fprintf(stderr, "pn_tt: STREAM STUCK — %u of %u stream queues behind or not running\n",
                shown, g_ts.queue_count);
    }
}

/** Wait up to `ms` for the puller to have completed `target` PULLs (no failure on timeout). */
static bool ts_poll_puller(uint32_t target, double ms)
{
    pn_tt_residency *puller = &g_ts.device->puller;
    const ft_tt_noc_access *noc = &g_td.original[TD_ACC_NOC];
    const double started = pn_time_ms();
    uint32_t fields[8];

    for (;;) {
        if (noc->read(noc->user, puller->core_x, puller->core_y, FT_TT_OP_QUEUE_BASE + 0x10u,
                      fields, 16u) != FT_TT_OK)
            return false;
        if ((int32_t)(fields[1] - target) >= 0)
            return true;
        if (pn_time_ms() - started > ms)
            return false;
    }
}

static bool ts_wait_puller(uint32_t target)
{
    pn_tt_residency *puller = &g_ts.device->puller;
    const ft_tt_noc_access *noc = &g_td.original[TD_ACC_NOC];
    const double started = pn_time_ms();
    uint32_t fields[8];

    if ((int32_t)(g_ts.completed - target) >= 0)
        return true;
    for (;;) {
        if (noc->read(noc->user, puller->core_x, puller->core_y, FT_TT_OP_QUEUE_BASE + 0x10u,
                      fields, 16u) != FT_TT_OK)
            return false;
        if (fields[3] == (uint32_t)FT_TT_QUEUE_STATE_HOST_POISON)
            return false;
        if ((int32_t)(fields[1] - target) >= 0) {
            g_ts.completed = fields[1];
            break;
        }
        if (pn_time_ms() - started > 4000.0)
            ts_report_early_stall("region wait", target);
        if (pn_time_ms() - started > 20000.0) {
            ts_report_stuck_puller("region wait", target);
            return false;
        }
    }
    g_ts.wait_ms += pn_time_ms() - started;
    return true;
}

/** HOST GAP (`PN_TT_HOST_GAP_STARVE=N`, timing only): every Nth launch reads the puller's
 *  completed count first; equal to every PULL launched = the puller had run out of program
 *  and was waiting on the host's capture (a STARVED launch). A starved puller is spinning in
 *  its idle loop: its idle-spin word (control word 10, reset when work arrives) says how long
 *  it has waited, in spins; `ts_starve_calibrate` prices a spin once, on an idle puller. */
static uint64_t g_starve_sampled, g_starve_empty, g_starve_spins;
/* by launches since the last full flush (0 = the first chunk after a boundary .. 4 = 4+) */
static uint64_t g_starve_at_sampled[5], g_starve_at_empty[5];
static uint32_t g_ts_since_flush;
static double g_starve_spins_per_us;
static uint32_t ts_starve_every(void)
{
    static uint32_t every = 0xFFFFFFFFu;

    if (every == 0xFFFFFFFFu)
        every = getenv("PN_TT_HOST_GAP_STARVE") ? (uint32_t)atoi(getenv("PN_TT_HOST_GAP_STARVE"))
                                                : 0u;
    return every;
}

static void ts_starve_sample(pn_tt_residency *puller)
{
    static uint32_t n;
    const ft_tt_noc_access *noc = &g_td.original[TD_ACC_NOC];
    const uint32_t every = ts_starve_every();
    uint32_t fields[8];

    if (every == 0u || ++n % every != 0u)
        return;
    if (noc->read(noc->user, puller->core_x, puller->core_y, FT_TT_OP_QUEUE_BASE + 0x10u, fields,
                  32u) != FT_TT_OK)
        return;
    g_starve_sampled++;
    g_starve_at_sampled[g_ts_since_flush < 4u ? g_ts_since_flush : 4u]++;
    if (fields[1] == g_ts.launched) {
        g_starve_empty++;
        g_starve_at_empty[g_ts_since_flush < 4u ? g_ts_since_flush : 4u]++;
        g_starve_spins += fields[6];
    }
}

/** Once, on a puller the host knows is idle (every PULL awaited): idle spins per us. */
static void ts_starve_calibrate(pn_tt_residency *puller)
{
    const ft_tt_noc_access *noc = &g_td.original[TD_ACC_NOC];
    uint32_t a, b;
    double t0, t1;

    if (ts_starve_every() == 0u || g_starve_spins_per_us > 0.0)
        return;
    t0 = pn_time_ms();
    if (noc->read(noc->user, puller->core_x, puller->core_y, FT_TT_OP_QUEUE_BASE + 40u, &a, 4u) !=
        FT_TT_OK)
        return;
    while (pn_time_ms() - t0 < 2.0)
        ;
    t1 = pn_time_ms();
    if (noc->read(noc->user, puller->core_x, puller->core_y, FT_TT_OP_QUEUE_BASE + 40u, &b, 4u) !=
            FT_TT_OK ||
        b <= a)
        return;
    g_starve_spins_per_us = (double)(b - a) / ((t1 - t0) * 1000.0);
}

static void ts_table_forget(void)
{
    g_ts.table_rows = 0u;
}

/** `PN_TT_STREAM_SHARED_TABLE` (default 1; 0 = every launch carries its own queue table). */
static bool ts_shared_table_on(void)
{
    static int on = -1;

    if (on < 0)
        on = !getenv("PN_TT_STREAM_SHARED_TABLE") || atoi(getenv("PN_TT_STREAM_SHARED_TABLE")) != 0;
    return on != 0;
}

/** THE SHARED QUEUE TABLE: one copy on the puller at TS_TABLE_BASE instead of ~130 queue rows
 *  (~520 words) in every launch's program. A row only ever GROWS the table in place (a launch
 *  in flight indexes only the queues it knew); a row whose nonce changed (a re-armed queue)
 *  would change what an in-flight launch reads, so the puller is drained first. Forgotten at
 *  every stream begin (a replay program may have overwritten the arena). */
static bool ts_table_sync(const ft_tt_noc_access *bulk, pn_tt_residency *puller)
{
    uint32_t rows[4u * TS_MAX_QUEUES];
    uint32_t q, first = g_ts.table_rows;

    for (q = 0; q < g_ts.table_rows; q++)
        if (g_ts.table_nonce[q] != g_ts.queues[q].nonce)
            break;
    if (q < g_ts.table_rows) {
        if (!ts_wait_puller(g_ts.launched))
            return false;
        g_ts.table_drains++;
        first = 0u;
    }
    if (first == g_ts.queue_count)
        return true;
    for (q = first; q < g_ts.queue_count; q++) {
        rows[4u * (q - first)] = (uint32_t)g_ts.queues[q].x | ((uint32_t)g_ts.queues[q].y << 16);
        rows[4u * (q - first) + 1u] = g_ts.queues[q].nonce;
        rows[4u * (q - first) + 2u] = 0xFFFFu;
        rows[4u * (q - first) + 3u] = 0u;
        g_ts.table_nonce[q] = g_ts.queues[q].nonce;
    }
    /* RED `stream_table_nonce`: the shared rows carry wrong nonces; the puller's await token
     * check must refuse (proves the launches read THIS table). */
    if (pn_tt_red("stream_table_nonce"))
        for (q = first; q < g_ts.queue_count; q++)
            rows[4u * (q - first) + 1u] ^= 1u;
    if (bulk->write(bulk->user, puller->core_x, puller->core_y, TS_TABLE_BASE + first * 16u, rows,
                    (size_t)(g_ts.queue_count - first) * 16u) != FT_TT_OK)
        return false;
    g_ts.table_rows = g_ts.queue_count;
    g_ts.table_writes++;
    return true;
}

/** Upload the captured chunk into the next region and launch it; `wait`: and wait for every
 *  launched chunk (the puller's own await, so its completion bookkeeping stays exact). */
static bool ts_launch(bool wait)
{
    pn_tt_device *device = g_ts.device;
    pn_tt_residency *puller;
    const ft_tt_noc_access *bulk = &g_td.original[TD_ACC_BULK_L1];
    uint32_t params[FT_TT_QUEUE_PARAM_WORDS];
    uint32_t *w = g_ts.words, q, base;
    size_t total;
    double started;
    bool ok = true;

    if (g_ts.failed)
        return false;
    started = pn_time_ms();
    puller = &device->puller;
    g_td.internal = true;
    if (g_ts.count > 0u) {
        const uint32_t r = g_ts.region;

        /* the region must be free, and the puller's ring must have room */
        if (g_ts.region_done_at[r] != 0u && !ts_wait_puller(g_ts.region_done_at[r]))
            ok = false;
        if (ok && g_ts.launched - g_ts.completed >= TS_INFLIGHT &&
            !ts_wait_puller(g_ts.launched - TS_INFLIGHT + 1u))
            ok = false;
        base = TD_PROGRAM_BASE + r * TS_REGION_WORDS * 4u;
        if (g_ts.last_flush_end > 0.0)
            g_ts.capture_ms += started - g_ts.last_flush_end;
        w[TS_ENTRY_BASE + g_ts.count++] = 5u; /* segment end */
        w[TS_ENTRY_BASE + g_ts.count++] = 6u; /* program end */
        {
            /* COMPACT: header, the queues actually known, the segment table, the entries —
             * a fixed 256-queue table was 1,044 words a launch. SHARED: no table here, the
             * header points at the puller's one copy (an offset from this header). */
            const bool shared = ok && ts_shared_table_on() && ts_table_sync(bulk, puller);
            const uint32_t table = 16u, seg = table + (shared ? 0u : 4u * g_ts.queue_count);
            const uint32_t entries = (seg + 1u + 3u) & ~3u;
            uint32_t *o = g_ts.out;

            if (ok && ts_shared_table_on() && !shared)
                ok = false;
            memset(o, 0, (size_t)entries * 4u);
            o[0] = TD_PULL_MAGIC;
            o[1] = 1u;
            o[2] = g_ts.queue_count;
            o[3] = shared ? TS_TABLE_BASE - base : table * 4u;
            o[4] = 1u;
            o[5] = seg * 4u;
            for (q = 0; !shared && q < g_ts.queue_count; q++) {
                o[table + 4u * q] = (uint32_t)g_ts.queues[q].x | ((uint32_t)g_ts.queues[q].y << 16);
                o[table + 4u * q + 1u] = g_ts.queues[q].nonce;
                o[table + 4u * q + 2u] = 0xFFFFu;
            }
            if (shared)
                g_ts.table_words_saved += 4u * g_ts.queue_count;
            o[seg] = entries * 4u;
            memcpy(&o[entries], &w[TS_ENTRY_BASE], g_ts.count * 4u);
            total = entries + g_ts.count;
            o[6] = base + (uint32_t)(TS_REGION_WORDS * 4u);
            o[7] = 0u;
            w = o;
        }
        {
            const double up = pn_time_ms();

            if (ok && bulk->write(bulk->user, puller->core_x, puller->core_y, base, w,
                                  (total * 4u + 15u) & ~(size_t)15u) != FT_TT_OK)
                ok = false;
            g_ts.upload_ms += pn_time_ms() - up;
        }
        ts_starve_sample(puller);
        memset(params, 0, sizeof(params));
        params[0] = base;
        params[4] = td_lazy_heads_bit();
        /* Each queued PULL keeps ITS OWN parameter block: with one slot, a launch queued
         * behind another rewrote the base the earlier one had not read yet (chunked launches
         * skipped regions — error 4 at an await). */
        puller->param_slot = 1u + (g_ts.launched % 8u);
        if (ok && pn_tt_residency_enqueue(puller, TD_PULL_OPCODE, params, 1u) != 0)
            ok = false;
        puller->param_slot = 0u;
        g_ts_fresh = false;
        if (ok) {
            pn_tt_host_gap_card_busy();
            g_ts.launched++;
            g_ts.region_done_at[r] = g_ts.launched;
            g_ts.region = (r + 1u) % TS_REGIONS;
            g_ts.async_launches++;
            g_ts_since_flush++;
        }
        g_ts.launched_words += total;
        g_ts.count = 0u;
        g_ts.await_at = -1;
        g_ts.segments++;
    }
    if (ok && wait && g_ts.launched != g_ts.completed) {
        /* Name a stall at 4 s, then wait the rest of the 20 s as before. */
        const bool early = ts_poll_puller(g_ts.launched, 4000.0);

        if (!early)
            ts_report_early_stall("flush", g_ts.launched);
        if (pn_tt_residency_await(puller, early ? 20000u : 16000u) != 0) {
            ts_report_stuck_puller("flush", g_ts.launched);
            ok = false;
        } else {
            g_ts.completed = g_ts.launched;
        }
    }
    if (ok && wait)
        pn_tt_host_gap_card_idle(g_ts_gap_why);
    g_td.internal = false;
    g_ts.flush_ms += pn_time_ms() - started;
    if (wait) {
        g_ts.last_flush_end = pn_time_ms();
        g_ts_fresh = true;
        g_ts_since_flush = 0u;
    }
    if (!ok)
        g_ts.failed = true;
    return ok;
}

/** Everything captured runs on the card and is done (S2: every converted AWAIT first). */
static bool ts_flush(void)
{
    if (!ts_pre_push_kind(PN_TT_PRE_SEGMENT))
        return false;
    return ts_launch(true);
}

/** `PN_TT_STREAM_DIRECT` (default 1): WRITE DIRECT entries (no scratch copy on the puller). */
static bool ts_direct(void)
{
    static int direct = -1;

    if (direct < 0)
        direct = !getenv("PN_TT_STREAM_DIRECT") || atoi(getenv("PN_TT_STREAM_DIRECT")) != 0;
    return direct != 0;
}

static bool ts_write_entry(uint16_t x, uint16_t y, uint64_t addr, const void *src, size_t bytes)
{
    const uint8_t *p = src;

    while (bytes) {
        const size_t chunk = bytes > 4000u * 4u ? 4000u * 4u : bytes;
        uint32_t head[3];

        if ((!ts_room(3u + chunk / 4u) || g_ts.count >= TS_CHUNK_WORDS) && !ts_launch(false))
            return false;
        if (!ts_room(6u + chunk / 4u))
            return false;
        if (ts_direct()) {
            /* WRITE DIRECT: the payload starts at a 16 B boundary of the uploaded program
             * (entries start 16 B-aligned), as its destination does. */
            const uint32_t pad = (uint32_t)((4u - (g_ts.count + 3u) % 4u) % 4u);

            head[0] = 8u | ((uint32_t)(chunk / 4u) << 8) | (pad << 20);
            head[1] = (uint32_t)x | ((uint32_t)y << 16);
            head[2] = (uint32_t)addr;
            memcpy(&g_ts.words[TS_ENTRY_BASE + g_ts.count], head, sizeof(head));
            memset(&g_ts.words[TS_ENTRY_BASE + g_ts.count + 3u], 0, pad * 4u);
            memcpy(&g_ts.words[TS_ENTRY_BASE + g_ts.count + 3u + pad], p, chunk);
            g_ts.count += 3u + pad + chunk / 4u;
        } else {
            head[0] = 1u | ((uint32_t)(chunk / 4u) << 8);
            head[1] = (uint32_t)x | ((uint32_t)y << 16);
            head[2] = (uint32_t)addr;
            memcpy(&g_ts.words[TS_ENTRY_BASE + g_ts.count], head, sizeof(head));
            memcpy(&g_ts.words[TS_ENTRY_BASE + g_ts.count + 3u], p, chunk);
            g_ts.count += 3u + chunk / 4u;
        }
        g_ts.entries++;
        p += chunk;
        addr += chunk;
        bytes -= chunk;
    }
    return true;
}

/** `PN_TT_STREAM_CENSUS=1`: the S2e duplicate-write census (a byte hash of every ring slot
 *  and parameter block the host captures, ~0.5 MB a cycle). Off by default since the LAUNCH
 *  lane: it is host time in the capture path, which starves the puller mid-stream. */
static bool ts_dup_census_on(void)
{
    static int on = -1;

    if (on < 0)
        on = getenv("PN_TT_STREAM_CENSUS") && atoi(getenv("PN_TT_STREAM_CENSUS")) != 0;
    return on != 0;
}

static void ts_census(uint8_t cls, uint16_t x, uint16_t y, uint64_t addr, const void *src,
                      size_t bytes)
{
    const uint8_t *p = src;
    uint64_t hash = 1469598103934665603ull;
    uint32_t i;
    size_t b;

    switch (cls) {
    case TD_Q_RING: g_ts.n_ring++; g_ts.w_ring += bytes / 4u; break;
    case TD_Q_PARAM_ROT:
    case TD_Q_PARAM_CACHE: g_ts.n_param++; g_ts.w_param += bytes / 4u; break;
    case TD_B_DRAM_W: g_ts.n_dram++; g_ts.w_dram += bytes / 4u; return;
    default: g_ts.n_l1++; g_ts.w_l1 += bytes / 4u; return;
    }
    if (!ts_dup_census_on())
        return;
    for (b = 0; b < bytes; b++)
        hash = (hash ^ p[b]) * 1099511628211ull;
    for (i = 0; i < 32u; i++) {
        const uint32_t j = i;

        if (g_ts.recent[j].hash == hash && g_ts.recent[j].addr == (uint32_t)addr &&
            g_ts.recent[j].bytes == (uint32_t)bytes &&
            (g_ts.recent[j].x != x || g_ts.recent[j].y != y)) {
            g_ts.dup_writes++;
            g_ts.dup_words += bytes / 4u;
            if (g_ts.recent[j].y == y)
                g_ts.dup_same_row++;
            break;
        }
    }
    g_ts.recent[g_ts.recent_at % 32u].x = x;
    g_ts.recent[g_ts.recent_at % 32u].y = y;
    g_ts.recent[g_ts.recent_at % 32u].addr = (uint32_t)addr;
    g_ts.recent[g_ts.recent_at % 32u].bytes = (uint32_t)bytes;
    g_ts.recent[g_ts.recent_at % 32u].hash = hash;
    g_ts.recent_at++;
}

/* ---------------------------------------------------------------------------------------
 * L10 B2: HOISTING IN THE STREAM. The rule is the replay's (pn_tt_dispatch_hoist.h), driven
 * one captured entry at a time: a rotating parameter block or a ring slot of a queue whose
 * head this stream has seen goes in front of the chunk's last AWAIT (only HEADs after it),
 * behind a FLUSH, when what it overwrites is proven finished there. `PN_TT_STREAM_HOIST`.
 * --------------------------------------------------------------------------------------- */

static bool ts_hoist_on(void)
{
    static int on = -1;

    if (on < 0) {
        const char *env = getenv("PN_TT_STREAM_HOIST");

        on = env && *env ? atoi(env) != 0 : 0;
    }
    return on == 1;
}

/** The stream hoist, unless S2 plans this stream (its lagged AWAITs already put a window
 *  between a write and the await before it). */
static bool ts_hoist_active(void)
{
    return ts_hoist_on() && !g_ts.pre;
}

static uint16_t ts_queue_xy(uint32_t q, int which)
{
    return which ? g_ts.queues[q].y : g_ts.queues[q].x;
}

/** The queue's ring capacity (control word 1), read once a nonce; 0 unknown. */
static uint32_t ts_queue_capacity(ts_queue *queue)
{
    td_queue probe;

    if (queue->capacity != 0u && queue->capacity_nonce == queue->nonce)
        return queue->capacity;
    memset(&probe, 0, sizeof(probe));
    probe.x = queue->x;
    probe.y = queue->y;
    queue->capacity = td_queue_capacity_mask(&probe) + 1u;
    if (queue->capacity == 1u)
        queue->capacity = 0u;
    queue->capacity_nonce = queue->nonce;
    return queue->capacity;
}

/** A stream begins: every bound unknown (a host-issued gap or a re-arm may lie behind it). */
static void ts_hoist_begin(void)
{
    uint32_t q;

    g_ts.await_at = -1;
    g_ts.leader_queues = UINT32_MAX;
    for (q = 0; q < g_ts.queue_count; q++)
        g_ts.queues[q].fresh = false;
    if (!ts_hoist_active())
        return;
    if (!g_ts.hoist)
        g_ts.hoist = pn_tt_hoist_tracker_create(TS_MAX_QUEUES);
    if (g_ts.hoist)
        pn_tt_hoist_tracker_forget(g_ts.hoist);
}

/** The tracker item for a queue write, or kind FIXED. */
static pn_tt_hoist_item ts_hoist_item(uint8_t cls, uint32_t q, uint64_t addr, const void *src,
                                      size_t bytes)
{
    pn_tt_hoist_item it;
    const ts_queue *queue = &g_ts.queues[q];

    memset(&it, 0, sizeof(it));
    it.kind = PN_TT_HOIST_FIXED;
    it.group = -1;
    it.queue = (uint16_t)q;
    if (!queue->fresh)
        return it;
    if (cls == TD_Q_PARAM_ROT) {
        it.kind = PN_TT_HOIST_WRITE;
        it.index = queue->issued + queue->staged;
        it.address = (uint32_t)addr;
    } else if (cls == TD_Q_RING && bytes % 32u == 0u && bytes > 0u) {
        uint32_t w[8];

        memcpy(w, (const uint8_t *)src + bytes - 32u, sizeof(w));
        it.kind = PN_TT_HOIST_RING;
        it.index = queue->issued + queue->staged;
        it.slots = (uint32_t)(bytes / 32u);
        if ((w[5] & 0x7FFFFFFFu) > 1u)
            it.pull_sequence = w[6];
        it.group_sequence = w[7];
    }
    return it;
}

/** Put one WRITE DIRECT in front of the chunk's last AWAIT (FLUSH first, once). */
static bool ts_hoist_insert(uint16_t x, uint16_t y, uint64_t addr, const void *src, size_t bytes)
{
    const size_t at = (size_t)g_ts.await_at + (g_ts.flushed ? 0u : 1u);
    const uint32_t pad = (uint32_t)((4u - (at + 3u) % 4u) % 4u);
    const size_t words = 3u + pad + bytes / 4u + (g_ts.flushed ? 0u : 1u);
    uint32_t *base = &g_ts.words[TS_ENTRY_BASE];
    const size_t tail = g_ts.count - (size_t)g_ts.await_at;

    if (!ts_direct() || bytes > 4000u * 4u || !ts_room(words + 4u))
        return false;
    memmove(&base[(size_t)g_ts.await_at + words], &base[g_ts.await_at], tail * 4u);
    if (!g_ts.flushed) {
        base[g_ts.await_at] = FT_TT_PULLER_ENTRY_FLUSH;
        g_ts.hoist_flushes++;
    }
    base[at] = 8u | ((uint32_t)(bytes / 4u) << 8) | (pad << 20);
    base[at + 1u] = (uint32_t)x | ((uint32_t)y << 16);
    base[at + 2u] = (uint32_t)addr;
    memset(&base[at + 3u], 0, pad * 4u);
    memcpy(&base[at + 3u + pad], src, bytes);
    g_ts.count += words;
    g_ts.await_at += (int64_t)words;
    g_ts.flushed = true;
    g_ts.entries++;
    return true;
}

/** Capture a queue write, in front of the last await when the rule allows. */
static bool ts_hoist_write(uint8_t cls, uint32_t q, uint16_t x, uint16_t y, uint64_t addr,
                           const void *src, size_t bytes)
{
    pn_tt_hoist_item it;
    pn_tt_hoist_need need[4];
    uint32_t needs;
    bool moved = false;

    if (!ts_hoist_active() || !g_ts.hoist)
        return ts_write_entry(x, y, addr, src, bytes);
    it = ts_hoist_item(cls, q, addr, src, bytes);
    if (it.kind == PN_TT_HOIST_RING || it.kind == PN_TT_HOIST_WRITE) {
        const uint32_t capacity = ts_queue_capacity(&g_ts.queues[q]);
        const int32_t *leader_of = g_ts.leader_of;

        if (g_ts.leader_queues != g_ts.queue_count) {
            td_hoist_leaders(g_ts.device, g_ts.queue_count, ts_queue_xy, g_ts.leader_of);
            g_ts.leader_queues = g_ts.queue_count;
        }
        pn_tt_hoist_tracker_queue(g_ts.hoist, q, capacity, 0u, leader_of[q]);
        if (it.kind == PN_TT_HOIST_RING && leader_of[q] >= 0)
            pn_tt_hoist_tracker_queue(g_ts.hoist, (uint32_t)leader_of[q],
                                      ts_queue_capacity(&g_ts.queues[leader_of[q]]), 0u, -1);
        needs = pn_tt_hoist_tracker_needs(g_ts.hoist, &it, need);
        if (g_ts.await_at >= 0 && !g_ts.after_await &&
            pn_tt_hoist_tracker_met(g_ts.hoist, need, needs, g_ts.await_undo_from,
                                    g_ts.await_undo_to))
            moved = ts_hoist_insert(x, y, addr, src, bytes);
        if (moved)
            g_ts.hoisted++;
        else
            g_ts.hoist_blocked++;
    }
    if (!moved) {
        const uint32_t launched = g_ts.launched;

        if (!ts_write_entry(x, y, addr, src, bytes))
            return false;
        g_ts.after_await = g_ts.launched == launched ? true : g_ts.after_await;
    }
    (void)pn_tt_hoist_tracker_apply(g_ts.hoist, &it);
    if (it.kind == PN_TT_HOIST_RING || (cls == TD_Q_RING && g_ts.queues[q].fresh))
        g_ts.queues[q].staged += (uint32_t)(bytes / 32u);
    return true;
}

/** A HEAD captured: the queue's count is exact from here (its writes may move). */
static void ts_hoist_head(uint32_t q, uint32_t value)
{
    g_ts.queues[q].fresh = true;
    g_ts.queues[q].staged = 0u;
    (void)value;
}

/** An AWAIT appended at word `at` of the chunk. */
static void ts_hoist_await(uint32_t q, uint32_t wanted, size_t at)
{
    pn_tt_hoist_item it;

    g_ts.await_at = -1;
    if (!ts_hoist_active() || !g_ts.hoist || !g_ts.queues[q].fresh)
        return;
    memset(&it, 0, sizeof(it));
    it.kind = PN_TT_HOIST_AWAIT;
    it.group = -1;
    it.queue = (uint16_t)q;
    it.index = wanted;
    pn_tt_hoist_tracker_drop_undo(g_ts.hoist);
    g_ts.await_undo_from = pn_tt_hoist_tracker_apply(g_ts.hoist, &it);
    g_ts.await_undo_to = pn_tt_hoist_tracker_undo_mark(g_ts.hoist);
    g_ts.await_at = (int64_t)at;
    g_ts.after_await = false;
    g_ts.flushed = false;
}

/* ---------------------------------------------------------------------------------------
 * S2 IN THE STREAM: each captured queue access is an item for the precondition planner, fed
 * before the access's own entry, so the realized AWAITs and the sidecar entries it hands out
 * land in front of it. Entries are WRITE DIRECTs of the 16-byte sidecar slot (the stream's
 * values are literal; the replay uses the puller's PRECOND entry for its per-token patches).
 * --------------------------------------------------------------------------------------- */

static void ts_pre_emit(void *user, const pn_tt_pre_action *a)
{
    (void)user;
    if (g_ts.pre_verify)
        td_oncard_append(&g_ts.vplan, a);
    if (g_ts.failed)
        return;
    if (a->kind == PN_TT_PRE_OUT_AWAIT) {
        const uint32_t entry[4] = { 4u, a->queue, a->count, 0u };

        g_ts.n_await++;
        if (!ts_emit(entry, 4u))
            g_ts.failed = true;
    } else if (a->kind == PN_TT_PRE_OUT_PRECOND) {
        const ts_queue *target = &g_ts.queues[a->queue], *producer = &g_ts.queues[a->producer];
        const uint32_t words[4] = { a->index, ftq_pre_xy_word(producer->x, producer->y), a->count,
                                    producer->nonce };

        g_ts.n_precond++;
        /* A coordinate past six bits would leave the slot untagged — no wait at all. */
        if (words[1] == 0u || target->capacity == 0u ||
            !ts_write_entry(target->x, target->y,
                            FT_TT_OP_QUEUE_BASE + ftq_pre_entry_offset(a->index,
                                                                       target->capacity - 1u),
                            words, sizeof(words)))
            g_ts.failed = true;
    }
}

/** Tell the planner about queue `q` the first time this stream names it: its ring capacity
 *  (read once a nonce), its pull leader, and whether an entry can be addressed to it. */
static void ts_pre_setup(uint32_t q)
{
    uint32_t capacity;

    if (q >= TS_MAX_QUEUES || g_ts.pre_set[q])
        return;
    capacity = ts_queue_capacity(&g_ts.queues[q]);
    if (g_ts.leader_queues != g_ts.queue_count) {
        td_hoist_leaders(g_ts.device, g_ts.queue_count, ts_queue_xy, g_ts.leader_of);
        g_ts.leader_queues = g_ts.queue_count;
    }
    g_ts.pre_info[q].capacity = capacity;
    g_ts.pre_info[q].honours = capacity != 0u;
    pn_tt_precondition_queue(g_ts.pre, q, capacity, 0u, g_ts.leader_of[q],
                             g_ts.pre_info[q].honours);
    g_ts.pre_set[q] = true;
}

/** Feed one item. @return false (the stream failed) when the planner refused it. */
static bool ts_pre_push(const pn_tt_pre_item *it)
{
    if (g_ts.failed)
        return false;
    if (it->kind == PN_TT_PRE_HEAD || it->kind == PN_TT_PRE_RING ||
        it->kind == PN_TT_PRE_WRITE || it->kind == PN_TT_PRE_AWAIT)
        ts_pre_setup(it->queue);
    if (g_ts.pre_verify) {
        if (g_ts.vlog_n == g_ts.vlog_cap) {
            const size_t grow = g_ts.vlog_cap ? g_ts.vlog_cap * 2u : 16384u;
            pn_tt_pre_item *more = realloc(g_ts.vlog, grow * sizeof(*more));

            if (more) {
                g_ts.vlog = more;
                g_ts.vlog_cap = grow;
            } else {
                g_ts.pre_verify = false; /* the check is off for this stream, not the plan */
            }
        }
        if (g_ts.pre_verify)
            g_ts.vlog[g_ts.vlog_n++] = *it;
    }
    g_ts.pre_last_await = it->kind == PN_TT_PRE_AWAIT;
    g_ts.pre_last_q = it->queue;
    g_ts.pre_last_count = it->index;
    if (pn_tt_precondition_push(g_ts.pre, g_ts.pre_items++, it) != 0)
        g_ts.failed = true;
    return !g_ts.failed;
}

/** The planner's item for a captured queue write (`cls` RING / PARAM_ROT / PARAM_CACHE). A
 *  ring slot or parameter block of a queue whose head this stream has not seen has no known
 *  index: an untracked write (everything realized first). */
static pn_tt_pre_item ts_pre_write_item(uint8_t cls, uint32_t q, uint64_t addr, const void *src,
                                        size_t bytes)
{
    const ts_queue *queue = &g_ts.queues[q];
    pn_tt_pre_item it;

    memset(&it, 0, sizeof(it));
    it.kind = PN_TT_PRE_BARRIER;
    it.group = -1;
    it.queue = (uint16_t)q;
    if (!queue->fresh)
        return it;
    it.index = queue->issued + g_ts.pre_staged[q];
    if (cls == TD_Q_RING && bytes % 32u == 0u && bytes > 0u) {
        uint32_t w[8];

        memcpy(w, (const uint8_t *)src + bytes - 32u, sizeof(w));
        it.kind = PN_TT_PRE_RING;
        it.slots = (uint32_t)(bytes / 32u);
        if ((w[5] & 0x7FFFFFFFu) > 1u)
            it.pull_sequence = w[6];
        it.group_sequence = w[7];
    } else if (cls == TD_Q_PARAM_ROT) {
        it.kind = PN_TT_PRE_WRITE;
        it.address = (uint32_t)addr;
    }
    return it;
}

/** An item of one kind and nothing else (a data write, a row copy, a stream's segment end). */
static bool ts_pre_push_kind(uint8_t kind)
{
    pn_tt_pre_item it;

    if (!g_ts.pre)
        return true;
    memset(&it, 0, sizeof(it));
    it.kind = kind;
    it.group = -1;
    return ts_pre_push(&it);
}

static bool td_stream_write(uint8_t acc, uint16_t x, uint16_t y, uint64_t addr,
                            const void *src, size_t bytes)
{
    const uint8_t cls = td_classify(acc, addr, true);
    const bool aligned = (addr % 16u) == 0u && (bytes % 4u) == 0u && bytes > 0u;

    if (aligned && cls != TD_Q_HEAD)
        ts_census(cls, x, y, addr, src, bytes);

    if (g_ts.failed)
        return false;
    switch (cls) {
    case TD_Q_RING:
    case TD_Q_PARAM_ROT:
    case TD_Q_PARAM_CACHE: {
        const uint32_t q = ts_queue_of(x, y);

        if (q == TS_MAX_QUEUES || !aligned)
            break;
        if (cls == TD_Q_RING && bytes >= 32u) {
            /* the LAST slot's token: with the next head it names the queue's nonce */
            memcpy(&g_ts.queues[q].ring_token, (const uint8_t *)src + bytes - 32u + 4u, 4u);
            g_ts.queues[q].ring_pending = true;
        }
        if (g_ts.pre) {
            const pn_tt_pre_item it = ts_pre_write_item(cls, q, addr, src, bytes);

            if (!ts_pre_push(&it))
                return false;
            if (cls == TD_Q_RING)
                g_ts.pre_staged[q] += (uint32_t)(bytes / 32u);
        }
        return ts_hoist_write(cls, q, x, y, addr, src, bytes);
    }
    case TD_Q_HEAD: {
        const uint32_t q = ts_queue_of(x, y);
        uint32_t value, entry[4];

        if (q == TS_MAX_QUEUES || bytes != 4u)
            break;
        memcpy(&value, src, 4u);
        if (g_ts.queues[q].ring_pending) {
            g_ts.queues[q].nonce = g_ts.queues[q].ring_token ^ (value * TD_TOKEN_MULTIPLIER);
            g_ts.queues[q].nonce_known = true;
            g_ts.queues[q].ring_pending = false;
        }
        g_ts.queues[q].issued = value;
        g_ts.queues[q].issued_known = true;
        ts_hoist_head(q, value);
        if (g_ts.pre) {
            /* The batch's ring slots were captured just before (an enqueue writes slots, then
             * the head, in one call), so its first descriptor is the head less them. */
            pn_tt_pre_item it;

            memset(&it, 0, sizeof(it));
            it.kind = PN_TT_PRE_HEAD;
            it.group = -1;
            it.queue = (uint16_t)q;
            it.slots = g_ts.pre_staged[q];
            it.index = it.slots ? value - it.slots : UINT32_MAX;
            g_ts.pre_staged[q] = 0u;
            if (!ts_pre_push(&it))
                return false;
        }
        entry[0] = 3u;
        entry[1] = q;
        entry[2] = value;
        entry[3] = 0u;
        g_ts.n_head++;
        return ts_emit(entry, 4u);
    }
    case TD_B_DRAM_W:
    case TD_B_L1_W:
        if (!aligned)
            break;
        g_ts.after_await = true;
        if (!ts_pre_push_kind(PN_TT_PRE_BARRIER))
            return false;
        return ts_write_entry(x, y, addr, src, bytes);
    default:
        break;
    }
    /* Not expressible: run what was captured, then do this one for real. */
    g_ts.boundaries++;
    g_ts.bound_by_class[cls]++;
    {
        const double t = pn_time_ms();

        (void)ts_flush();
        g_ts.bound_drain_ms += pn_time_ms() - t;
    }
    return false;
}

/** The puller AWAITs queue `q` reaching `count` — once a run (S2: the planner places it). */
static bool ts_emit_await(uint32_t q, uint32_t count)
{
    const size_t last = g_ts.count;
    uint32_t entry[4];

    if (g_ts.pre) {
        pn_tt_pre_item it;

        memset(&it, 0, sizeof(it));
        it.kind = PN_TT_PRE_AWAIT;
        it.group = -1;
        it.queue = (uint16_t)q;
        it.index = count;
        if (g_ts.pre_last_await && g_ts.pre_last_q == q && g_ts.pre_last_count == count)
            return true;
        return ts_pre_push(&it);
    }
    if (last >= 4u && g_ts.words[TS_ENTRY_BASE + last - 4u] == 4u &&
        g_ts.words[TS_ENTRY_BASE + last - 3u] == q &&
        g_ts.words[TS_ENTRY_BASE + last - 2u] == count)
        return true;
    entry[0] = 4u;
    entry[1] = q;
    entry[2] = count;
    entry[3] = 0u;
    g_ts.n_await++;
    if (!ts_emit(entry, 4u))
        return false;
    ts_hoist_await(q, count, g_ts.count - 4u);
    return true;
}

static bool td_stream_read(uint8_t acc, uint16_t x, uint16_t y, uint64_t addr, void *dst,
                           size_t bytes)
{
    const uint8_t cls = td_classify(acc, addr, false);

    if (!g_ts.failed && cls == TD_Q_POLL && (bytes % 4u) == 0u) {
        const uint32_t q = ts_queue_of(x, y);
        ts_queue *queue = q < TS_MAX_QUEUES ? &g_ts.queues[q] : NULL;

        if (queue && queue->nonce_known && queue->issued_known && queue->issued > 0u) {
            uint32_t window[16];
            const uint32_t first = (uint32_t)((addr - FT_TT_OP_QUEUE_BASE) / 4u);
            uint32_t token = queue->nonce ^ (queue->issued * TD_TOKEN_MULTIPLIER);

            if (token == 0u || token == (uint32_t)FT_TT_QUEUE_STATE_HOST_POISON)
                token = queue->nonce ^ 0xA5A5A5A5u;
            /* RED `stream_await_early`: the puller waits for one descriptor LESS than the host
             * was told is done — the next family starts on a producer's unfinished output; must
             * DIFFER (never more than one behind, so no ring overrun). */
            if (!ts_emit_await(q, queue->issued - (pn_tt_red("stream_await_early") &&
                                                           queue->issued > 1u ? 1u : 0u)))
                return false;
            memset(window, 0, sizeof(window));
            window[FT_TT_QUEUE_FIELD_COMPLETED_COUNT] = queue->issued;
            window[FT_TT_QUEUE_FIELD_COMPLETED_TOKEN] = token;
            window[FT_TT_QUEUE_FIELD_STATE] = (uint32_t)FT_TT_QUEUE_STATE_IDLE;
            window[FT_TT_QUEUE_FIELD_HEARTBEAT] = ++queue->heartbeat;
            window[FT_TT_QUEUE_FIELD_HEAD] = queue->issued;
            window[FT_TT_QUEUE_FIELD_TAIL] = queue->issued;
            if (first + bytes / 4u <= 16u) {
                memcpy(dst, &window[first], bytes);
                g_ts.polls_answered++;
                return true;
            }
        }
    }
    /* A real read: everything captured must have run on the card first. */
    g_ts.boundaries++;
    g_ts.bound_by_class[cls]++;
    {
        const double t = pn_time_ms();
        bool flushed;

        g_ts_gap_why = td_class_name[cls];
        flushed = ts_flush();
        g_ts_gap_why = "flush";
        g_ts.bound_drain_ms += pn_time_ms() - t;
        if (!flushed)
            return false;
    }
    {
        const double t_read = pn_time_ms();
        const int rc = g_td.original[acc].read(g_td.original[acc].user, x, y, addr, dst, bytes);

        g_ts.bound_access_ms += pn_time_ms() - t_read;
        g_ts.bound_ms_by_class[cls] += pn_time_ms() - t_read;
        g_ts.bound_bytes[cls] += bytes;
        if (cls == TD_B_L1_R) {
            uint32_t i;
            static int traced;

            /* `PN_TT_STREAM_TRACE_READ=<bytes>`: who reads it (once, a backtrace). */
            if (!traced && getenv("PN_TT_STREAM_TRACE_READ") &&
                (size_t)atoi(getenv("PN_TT_STREAM_TRACE_READ")) == bytes) {
                void *frames[16];

                traced = 1;
                fprintf(stderr, "pn_tt: STREAM traced read (%u,%u) 0x%llx %zu B:\n", x, y,
                        (unsigned long long)addr, bytes);
                backtrace_symbols_fd(frames, backtrace(frames, 16), 2);
            }
            for (i = 0; i < g_ts.l1r_count; i++)
                if (g_ts.l1r[i].addr == (uint32_t)addr && g_ts.l1r[i].bytes == (uint32_t)bytes)
                    break;
            if (i == g_ts.l1r_count && i < 16u) {
                g_ts.l1r[i].x = x;
                g_ts.l1r[i].y = y;
                g_ts.l1r[i].addr = (uint32_t)addr;
                g_ts.l1r[i].bytes = (uint32_t)bytes;
                g_ts.l1r_count++;
            }
            if (i < 16u) {
                g_ts.l1r[i].n++;
                g_ts.l1r[i].ms += pn_time_ms() - t_read;
            }
        }

        /* A real completion read teaches the queue's nonce and count. */
        if (rc == FT_TT_OK && cls == TD_Q_POLL && (bytes % 4u) == 0u) {
            const uint32_t q = ts_queue_of(x, y);
            const uint32_t first = (uint32_t)((addr - FT_TT_OP_QUEUE_BASE) / 4u);

            if (q < TS_MAX_QUEUES && first <= FT_TT_QUEUE_FIELD_COMPLETED_COUNT &&
                first + bytes / 4u > FT_TT_QUEUE_FIELD_COMPLETED_TOKEN) {
                const uint32_t *f = (const uint32_t *)dst;
                const uint32_t count = f[FT_TT_QUEUE_FIELD_COMPLETED_COUNT - first];
                const uint32_t token = f[FT_TT_QUEUE_FIELD_COMPLETED_TOKEN - first];

                if (count > 0u && !g_ts.queues[q].nonce_known) {
                    g_ts.queues[q].nonce = token ^ (count * TD_TOKEN_MULTIPLIER);
                    g_ts.queues[q].nonce_known = true;
                }
            }
        }
        if (rc != FT_TT_OK)
            g_ts.failed = true;
        /* the caller's own read must not run twice */
        return rc == FT_TT_OK;
    }
}

static bool td_stream_rowcopy(const uint32_t *params)
{
    uint32_t entry[10];

    g_ts.after_await = true;
    if (!ts_pre_push_kind(PN_TT_PRE_BARRIER))
        return false;
    entry[0] = 7u;
    memcpy(&entry[1], params, 9u * 4u);
    g_ts.n_rowcopy++;
    return ts_emit(entry, 10u);
}

static void ts_puller_clocks(uint32_t *busy, uint32_t *await);

/** pn_tt_dispatch_stream_begin_decode() is the caller: S2 may plan this stream. */
static bool g_ts_decode;

/** `PN_TT_ONCARD_VERIFY=1`: every S2 stream is checked at its end (pn_tt_precondition_verify). */
static bool ts_pre_verify_on(void)
{
    static int on = -1;

    if (on < 0)
        on = getenv("PN_TT_ONCARD_VERIFY") && atoi(getenv("PN_TT_ONCARD_VERIFY")) != 0;
    return on == 1;
}

/** S2 for a decode-time stream: a fresh planner (every bound unknown, as the hoist's). */
static void ts_pre_begin(void)
{
    const pn_tt_pre_config config = { td_oncard_lookahead(), pn_tt_red("precondition_early") };

    g_ts.pre = NULL;
    if (!g_ts_decode || !td_oncard_on())
        return;
    g_ts.pre = pn_tt_precondition_create(TS_MAX_QUEUES, &config, ts_pre_emit, NULL);
    if (!g_ts.pre) {
        fprintf(stderr, "pn_tt: STREAM — ONCARD planner not created; this stream awaits as "
                        "today\n");
        return;
    }
    g_ts.pre_items = 0u;
    g_ts.pre_last_await = false;
    memset(g_ts.pre_staged, 0, sizeof(g_ts.pre_staged));
    memset(g_ts.pre_set, 0, sizeof(g_ts.pre_set));
    memset(g_ts.pre_info, 0, sizeof(g_ts.pre_info));
    g_ts.pre_verify = ts_pre_verify_on();
    g_ts.vlog_n = 0u;
    g_ts.vplan.n = 0u;
    g_ts.vplan.oom = false;
}

/** The stream ended (everything realized by its last flush): check it if asked, drop the
 *  planner. A failed check is reported; the stream already ran, so it is also counted. */
static void ts_pre_end(void)
{
    pn_tt_pre_report report;

    if (!g_ts.pre)
        return;
    if (g_ts.pre_verify && !g_ts.vplan.oom) {
        g_ts.verified++;
        if (!pn_tt_precondition_verify(g_ts.vlog, g_ts.vlog_n, g_ts.vplan.a, g_ts.vplan.n,
                                       g_ts.pre_info, TS_MAX_QUEUES, NULL, 0u, &report) &&
            g_ts.verify_failed++ < 8u)
            fprintf(stderr, "pn_tt: STREAM — ONCARD check FAILED on stream %llu (%zu failures; "
                            "first: %s)\n", (unsigned long long)g_ts.verified, report.failures,
                    report.first);
    }
    pn_tt_precondition_destroy(g_ts.pre);
    g_ts.pre = NULL;
}

bool pn_tt_dispatch_stream_begin_decode(pn_tt_device *device)
{
    bool ok;

    g_ts_decode = true;
    ok = pn_tt_dispatch_stream_begin(device);
    g_ts_decode = false;
    return ok;
}

bool pn_tt_dispatch_stream_begin(pn_tt_device *device)
{
    const char *env = getenv("PN_TT_SPEC_STREAM");

    if (!device || (env && atoi(env) == 0) || !td_puller_arm(device) || g_td.replaying ||
        g_td.recording)
        return false;
    if (!g_ts.words) {
        g_ts.words = calloc(TS_MAX_WORDS, sizeof(uint32_t));
        g_ts.out = calloc(TS_MAX_WORDS, sizeof(uint32_t));
        if (!g_ts.words || !g_ts.out)
            return false;
    }
    g_ts.device = device;
    g_ts.count = 0u;
    g_ts.failed = false;
    /* every region is free between streams (each stream ends with a full wait) */
    memset(g_ts.region_done_at, 0, sizeof(g_ts.region_done_at));
    g_ts.launched = g_ts.completed = device->puller.queue.issued;
    g_ts.last_flush_end = pn_time_ms();
    ts_pre_begin(); /* before the hoist: an S2 stream does not hoist */
    ts_hoist_begin();
    ts_table_forget();
    /* The streamed program overwrites the puller's recorded decode program: re-learn it. */
    g_td.sequence_reset = true;
    td_wrap(device);
    ts_puller_clocks(&g_ts.puller_busy0, &g_ts.puller_await0);
    if (ts_census_enabled())
        ts_census_read(g_ts.census0);
    g_streaming = true;
    /* `PN_TT_STREAM_PROFILE=1` (with `PN_PROFILE`): sample the host only inside streams. */
    if (getenv("PN_TT_STREAM_PROFILE"))
        pn_sample_profiler_window(1);
    return true;
}

static void ts_puller_clocks(uint32_t *busy, uint32_t *await)
{
    const ft_tt_noc_access *noc = &g_td.original[TD_ACC_NOC];
    pn_tt_residency *puller = &g_ts.device->puller;
    uint32_t w[2] = { 0u, 0u };

    (void)noc->read(noc->user, puller->core_x, puller->core_y, FT_TT_OP_QUEUE_BASE + 62u * 4u,
                    w, 8u);
    *busy = w[0];
    *await = w[1];
}

bool pn_tt_dispatch_stream_end(void)
{
    bool ok;

    if (!g_streaming)
        return true;
    g_ts_gap_why = "end";
    ok = ts_flush() && !g_ts.failed;
    g_ts_gap_why = "flush";
    ts_pre_end();
    if (ok)
        ts_starve_calibrate(&g_ts.device->puller);
    {
        uint32_t busy, wait;

        ts_puller_clocks(&busy, &wait);
        g_ts.puller_busy_ms += (double)(uint32_t)(busy - g_ts.puller_busy0) * 16.0 / 1.35e6;
        g_ts.puller_await_ms += (double)(uint32_t)(wait - g_ts.puller_await0) * 16.0 / 1.35e6;
        if (ts_census_enabled())
            ts_census_accumulate();
    }
    g_streaming = false;
    if (getenv("PN_TT_STREAM_PROFILE"))
        pn_sample_profiler_window(0);
    td_unwrap();
    /* What the host thinks it issued stays what it issued; a failure leaves no trust. */
    if (!ok)
        g_ts.queue_count = 0u;
    return ok;
}

int pn_tt_dispatch_stream_await(uint16_t x, uint16_t y, uint32_t count)
{
    uint32_t q;

    if (!g_streaming)
        return 0;
    if (g_ts.failed)
        return -1;
    q = ts_queue_of(x, y);
    if (q >= TS_MAX_QUEUES || !g_ts.queues[q].nonce_known || !g_ts.queues[q].issued_known ||
        count > g_ts.queues[q].issued)
        return -1;
    if (count == 0u)
        return 1;
    return ts_emit_await(q, count) ? 1 : -1;
}

void pn_tt_dispatch_stream_queue_rearmed(uint16_t x, uint16_t y)
{
    for (uint32_t q = 0; q < g_ts.queue_count; q++)
        if (g_ts.queues[q].x == x && g_ts.queues[q].y == y) {
            g_ts.queues[q].nonce_known = false;
            g_ts.queues[q].issued_known = false;
            g_ts.queues[q].ring_pending = false;
            g_ts.queues[q].fresh = false;
            g_ts.queues[q].staged = 0u;
            g_ts.queues[q].capacity_nonce = 0u;
        }
}

bool pn_tt_dispatch_stream_kick(void)
{
    if (!g_streaming || g_ts.failed)
        return g_streaming ? false : true;
    return g_ts.count == 0u || ts_launch(false);
}

void pn_tt_dispatch_stream_report(double cycles)
{
    ts_census_report();
    if (g_ts.segments == 0u || cycles <= 0.0)
        return;
    fprintf(stderr, "pn_tt: STREAM — a cycle: %.1f segments, %.0f entries, %.1f boundaries, %.0f "
                    "polls answered, %.0f program words, %.1f launches; %.1f ms in launch+wait (upload "
                    "%.1f, region waits %.1f), %.1f ms host between flushes\n",
            (double)g_ts.segments / cycles, (double)g_ts.entries / cycles,
            (double)g_ts.boundaries / cycles, (double)g_ts.polls_answered / cycles,
            (double)g_ts.launched_words / cycles, (double)g_ts.async_launches / cycles,
            g_ts.flush_ms / cycles, g_ts.upload_ms / cycles, g_ts.wait_ms / cycles,
            g_ts.capture_ms / cycles);
    fprintf(stderr, "pn_tt: STREAM census a cycle: param %.0f (%.0f w), ring %.0f (%.0f w), dram %.0f "
                    "(%.0f w), l1 %.0f (%.0f w), head %.0f, await %.0f, rowcopy %.0f; "
                    "duplicate writes %.0f (%.0f w), %.0f in the same NoC row\n",
            g_ts.n_param / cycles, g_ts.w_param / cycles, g_ts.n_ring / cycles,
            g_ts.w_ring / cycles, g_ts.n_dram / cycles, g_ts.w_dram / cycles,
            g_ts.n_l1 / cycles, g_ts.w_l1 / cycles, g_ts.n_head / cycles, g_ts.n_await / cycles,
            g_ts.n_rowcopy / cycles, g_ts.dup_writes / cycles, g_ts.dup_words / cycles,
            g_ts.dup_same_row / cycles);
    if (g_ts.table_writes)
        fprintf(stderr, "pn_tt: STREAM shared queue table: %.0f words a cycle not uploaded, "
                        "%.1f table writes and %.2f drains a cycle\n",
                (double)g_ts.table_words_saved / cycles, (double)g_ts.table_writes / cycles,
                (double)g_ts.table_drains / cycles);
    {
        unsigned c;

        fprintf(stderr, "pn_tt: STREAM boundaries a cycle:");
        for (c = 0; c < TD_CLASSES; c++)
            if (g_ts.bound_by_class[c])
                fprintf(stderr, " %s %.1f (%.0f B, %.1f ms)", td_class_name[c],
                        g_ts.bound_by_class[c] / cycles, g_ts.bound_bytes[c] / cycles,
                        g_ts.bound_ms_by_class[c] / cycles);
        fprintf(stderr, " | drain %.1f ms, the accesses %.1f ms\n", g_ts.bound_drain_ms / cycles,
                g_ts.bound_access_ms / cycles);
        fprintf(stderr, "pn_tt: STREAM L1 reads a cycle:");
        for (c = 0; c < g_ts.l1r_count; c++)
            fprintf(stderr, " [(%u,%u) 0x%x %u B x%.1f %.2f ms]", g_ts.l1r[c].x, g_ts.l1r[c].y,
                    g_ts.l1r[c].addr, g_ts.l1r[c].bytes, g_ts.l1r[c].n / cycles,
                    g_ts.l1r[c].ms / cycles);
        fprintf(stderr, "\n");
    }
    if (g_starve_sampled) {
        const double share = (double)g_starve_empty / (double)g_starve_sampled;
        const double idle_us = g_starve_empty && g_starve_spins_per_us > 0.0
                                   ? (double)g_starve_spins / (double)g_starve_empty /
                                         g_starve_spins_per_us
                                   : 0.0;

        fprintf(stderr, "pn_tt: STREAM STARVED launches: %llu of %llu sampled found the puller "
                        "with nothing left to run; idle at the launch %.1f us each (%.1f spins/us)"
                        " -> ~%.2f ms a cycle\n", (unsigned long long)g_starve_empty,
                (unsigned long long)g_starve_sampled, idle_us, g_starve_spins_per_us,
                share * ((double)g_ts.async_launches / cycles) * idle_us / 1000.0);
        fprintf(stderr, "pn_tt: STREAM STARVED by launches since a flush: 0 %llu/%llu, 1 %llu/%llu, "
                        "2 %llu/%llu, 3 %llu/%llu, 4+ %llu/%llu\n",
                (unsigned long long)g_starve_at_empty[0], (unsigned long long)g_starve_at_sampled[0],
                (unsigned long long)g_starve_at_empty[1], (unsigned long long)g_starve_at_sampled[1],
                (unsigned long long)g_starve_at_empty[2], (unsigned long long)g_starve_at_sampled[2],
                (unsigned long long)g_starve_at_empty[3], (unsigned long long)g_starve_at_sampled[3],
                (unsigned long long)g_starve_at_empty[4], (unsigned long long)g_starve_at_sampled[4]);
    }
    fprintf(stderr, "pn_tt: STREAM puller a cycle: %.1f ms inside PULLs, %.1f of them in AWAIT "
                    "polls (%.1f issuing)\n", g_ts.puller_busy_ms / cycles,
            g_ts.puller_await_ms / cycles, (g_ts.puller_busy_ms - g_ts.puller_await_ms) / cycles);
    if (ts_hoist_on())
        fprintf(stderr, "pn_tt: STREAM HOIST a cycle: %.0f writes moved under an await, %.0f "
                        "left in place, %.0f flushes\n", g_ts.hoisted / cycles,
                g_ts.hoist_blocked / cycles, g_ts.hoist_flushes / cycles);
    if (g_ts.n_precond)
        fprintf(stderr, "pn_tt: STREAM ONCARD a cycle: %.0f sidecar entries, %.0f real awaits "
                        "(lookahead %u); %llu streams checked, %llu FAILED%s\n",
                g_ts.n_precond / cycles, g_ts.n_await / cycles, td_oncard_lookahead(),
                (unsigned long long)g_ts.verified, (unsigned long long)g_ts.verify_failed,
                pn_tt_red("precondition_early") ? " [RED precondition_early]" : "");
}

/**
 * Submit one data-mover descriptor (ROWCOPY / FILL / COPY) to the puller and wait for it.
 *
 * AN IDLE PULLER IS RETIRED BY THE SUBMIT ITSELF (residency_idle_retire: idle past the limit,
 * ~5 s), which then refuses the descriptor. Every data mover after a user's pause hit that: the
 * FILL fell back to the 0.43 s link zero and the checkpoint restore failed (run_serve_prefill_
 * ckpt.sh, 12 s pause). Re-arm once and resubmit, re-uploading the replay program when there is
 * one, as td_launch_segment() does. RED `puller_no_rearm` keeps the refusal.
 */
static int td_puller_submit(pn_tt_device *device, uint32_t opcode, const uint32_t *params)
{
    if (!td_puller_arm(device))
        return -1;
    pn_tt_host_gap_card_busy();
    if (pn_tt_residency_submit(&device->puller, opcode, params, 5000u) == 0) {
        pn_tt_host_gap_card_idle("puller-op");
        return 0;
    }
    if (pn_tt_red("puller_no_rearm") || device->puller.running || !td_puller_arm(device) ||
        (g_prog.count > 0u && !td_upload_program(device)))
        return -1;
    g_td.puller_rearms++;
    fprintf(stderr, "pn_tt: DISPATCH — the puller was idle-retired; re-armed for a data "
                    "mover\n");
    return pn_tt_residency_submit(&device->puller, opcode, params, 5000u);
}

int pn_tt_row_copy(pn_tt_device *device, const pn_tt_activation *src, uint32_t src_row,
                   const pn_tt_activation *dst, uint32_t dst_row)
{
    uint32_t params[FT_TT_QUEUE_PARAM_WORDS];
    uint32_t replica;
    double started;

    if (!device || !src || !dst || src->tile_count == 0u || src->tile_count != dst->tile_count ||
        src->bank >= device->dram_bank_count || !td_puller_arm(device))
        return -1;
    /* EVERY MIRROR of the destination: a stale mirror is a plausible wrong answer. */
    for (replica = 0; replica < dst->replica_count; replica++) {
        const uint32_t bank = dst->replica_bank[replica];

        memset(params, 0, sizeof(params));
        params[0] = device->dram[src->bank].x;
        params[1] = device->dram[src->bank].y;
        params[2] = src->address;
        params[3] = src_row;
        params[4] = device->dram[bank].x;
        params[5] = device->dram[bank].y;
        params[6] = dst->replica_address[replica];
        params[7] = dst_row;
        params[8] = src->tile_count;
        started = pn_time_ms();
        if (g_streaming) {
            /* S2d: inline in the streamed segment (the puller runs it in order). */
            if (!td_stream_rowcopy(params))
                return -1;
        } else if (td_puller_submit(device, TD_ROWCOPY_OPCODE, params) != 0)
            return -1;
        g_move.row_ms += pn_time_ms() - started;
        g_move.rows++;
    }
    return 0;
}

int pn_tt_fill_dram_oncard(pn_tt_device *device, uint32_t bank, uint32_t address,
                           size_t bytes, uint32_t value)
{
    uint32_t params[FT_TT_QUEUE_PARAM_WORDS];
    size_t offset;

    if (!device || bank >= device->dram_bank_count || bytes == 0u || (address & 63u) != 0u ||
        (bytes & 63u) != 0u || g_streaming || pn_tt_matvec_flush(device) != 0 ||
        !td_puller_arm(device))
        return -1;
    for (offset = 0; offset < bytes; offset += TD_FILL_MAX_BYTES) {
        const size_t n = bytes - offset < TD_FILL_MAX_BYTES ? bytes - offset : TD_FILL_MAX_BYTES;

        memset(params, 0, sizeof(params));
        params[0] = device->dram[bank].x;
        params[1] = device->dram[bank].y;
        params[2] = address + (uint32_t)offset;
        params[3] = (uint32_t)n;
        params[4] = value;
        if (td_puller_submit(device, TD_FILL_OPCODE, params) != 0)
            return -1;
    }
    return 0;
}

int pn_tt_copy_dram_oncard(pn_tt_device *device, uint32_t src_bank, uint32_t src_address,
                           uint32_t dst_bank, uint32_t dst_address, size_t bytes)
{
    uint32_t params[FT_TT_QUEUE_PARAM_WORDS];
    size_t offset;

    if (!device || src_bank >= device->dram_bank_count || dst_bank >= device->dram_bank_count ||
        bytes == 0u || ((src_address | dst_address) & 63u) != 0u || (bytes & 63u) != 0u ||
        g_streaming || pn_tt_matvec_flush(device) != 0 || !td_puller_arm(device))
        return -1;
    for (offset = 0; offset < bytes; offset += TD_FILL_MAX_BYTES) {
        const size_t n = bytes - offset < TD_FILL_MAX_BYTES ? bytes - offset : TD_FILL_MAX_BYTES;

        memset(params, 0, sizeof(params));
        params[0] = device->dram[src_bank].x;
        params[1] = device->dram[src_bank].y;
        params[2] = src_address + (uint32_t)offset;
        params[3] = device->dram[dst_bank].x;
        params[4] = device->dram[dst_bank].y;
        params[5] = dst_address + (uint32_t)offset;
        params[6] = (uint32_t)n;
        if (td_puller_submit(device, TD_COPY_OPCODE, params) != 0)
            return -1;
    }
    return 0;
}

/* ---------------------------------------------------------------------------------------
 * Observe-mode report (A1)
 * --------------------------------------------------------------------------------------- */

static void td_report(const td_token *t, uint64_t forward)
{
    uint64_t events[TD_CLASSES] = { 0 }, bytes[TD_CLASSES] = { 0 }, reads[TD_CLASSES] = { 0 };
    size_t i, segments = 0u, queue_run = 0u, longest = 0u;
    unsigned cls;

    for (i = 0; i < t->count; i++) {
        const td_event *e = &t->events[i];

        events[e->cls]++;
        bytes[e->cls] += (uint64_t)e->bytes * e->repeats;
        reads[e->cls] += e->repeats;
        if (td_is_boundary(e->cls)) {
            if (queue_run) {
                segments++;
                if (queue_run > longest)
                    longest = queue_run;
            }
            queue_run = 0u;
        } else {
            queue_run++;
        }
    }
    if (queue_run) {
        segments++;
        if (queue_run > longest)
            longest = queue_run;
    }
    fprintf(stderr, "pn_tt: DISPATCH RECORD forward %llu — %zu events (poll loops merged), %zu "
                    "queue segments between boundaries, longest %zu events\n",
            (unsigned long long)forward, t->count, segments, longest);
    for (cls = 0; cls < TD_CLASSES; cls++)
        if (events[cls])
            fprintf(stderr, "pn_tt: DISPATCH RECORD   %-13s %7llu events %9llu accesses %10llu B\n",
                    td_class_name[cls], (unsigned long long)events[cls],
                    (unsigned long long)reads[cls], (unsigned long long)bytes[cls]);
}

/* ---------------------------------------------------------------------------------------
 * A3: LIVE REPLAY. The host runs its forward exactly as always; its queue WRITES are
 * swallowed here (the puller makes them), its completion POLLS and every boundary access go
 * to the card for real. So the host's own counters, checks and data flow are unchanged, and
 * the puller is what the card waits on. Each segment is launched when the host reaches its
 * first event; the previous segment is awaited first (errors surface there, and the
 * puller's ring cannot overrun).
 * --------------------------------------------------------------------------------------- */

static int g_event_segment_ready; /* event -> segment table built */
static int32_t *g_event_segment;

static void td_index_segments(void)
{
    const td_token *a = &g_td.model[0];
    size_t i;
    int32_t segment = -1;
    bool in_segment = false;

    free(g_event_segment);
    g_event_segment = malloc(a->count * sizeof(*g_event_segment));
    if (!g_event_segment)
        return;
    for (i = 0; i < a->count; i++) {
        if (td_is_boundary(a->events[i].cls)) {
            in_segment = false;
            g_event_segment[i] = -1;
            continue;
        }
        if (!in_segment) {
            segment++;
            in_segment = true;
        }
        g_event_segment[i] = segment;
    }
    g_event_segment_ready = 1;
}

/** The model's prediction for one queue WRITE at token offset k, byte for byte. */
static bool td_predict_match(size_t i, uint32_t k, uint64_t addr, const void *data, size_t bytes)
{
    const td_token *a = &g_td.model[0], *b = &g_td.model[1];
    const td_event *ea = &a->events[i], *eb = &b->events[i];
    const uint32_t *wa = &a->pool[ea->data_at], *wb = &b->pool[eb->data_at];
    uint32_t words[TD_DATA_WORDS];
    const uint32_t n = (uint32_t)(bytes / 4u);
    uint32_t word;

    if (n > TD_DATA_WORDS || bytes % 4u)
        return false;
    memcpy(words, data, bytes);
    if (ea->cls == TD_Q_RING) {
        const td_queue *queue = &g_td.queues[ea->queue];
        const uint32_t index = ea->index + (eb->index - ea->index) * k;

        if (addr != FT_TT_OP_QUEUE_BASE + TD_QUEUE_RING_OFFSET + (index & queue->mask) * 32u)
            return false;
        for (word = 0; word < n; word++) {
            const uint32_t want = (word % 8u) == 1u
                                      ? td_token_at(queue->nonce, index + word / 8u)
                                      : wa[word] + (wb[word] - wa[word]) * k;

            if (want != words[word])
                return false;
        }
        return true;
    }
    if (addr != ea->addr)
        return false;
    for (word = 0; word < n; word++)
        if (wa[word] + (wb[word] - wa[word]) * k != words[word])
            return false;
    return true;
}

static void td_diverge(const char *why, size_t i)
{
    if (g_td.divergences++ < 8u)
        fprintf(stderr, "pn_tt: DISPATCH REPLAY DIVERGED at forward k=%u event %zu: %s — replay "
                        "is OFF from the next token\n", g_td.replay_k, i, why);
    g_td.replay_dead = true;
}

static bool td_await_puller(void);

static bool td_launch_segment(uint32_t segment)
{
    uint32_t params[FT_TT_QUEUE_PARAM_WORDS];
    pn_tt_residency *puller = &g_td.device->puller;
    bool ok = true;

    /* THE PULLER'S RING HOLDS A FEW: a segment is launched only once the previous ones are
     * done (every segment ends at a boundary, which awaited them already). */
    if (g_td.awaited_segment != g_td.next_segment && !td_await_puller())
        return false;
    g_td.internal = true;
    memset(params, 0, sizeof(params));
    params[0] = TD_PROGRAM_BASE;
    params[1] = g_td.replay_k;
    params[2] = segment;
    params[3] = segment;
    /* REDS (resident_puller.c args[4]): patches at k+1 must DIFFER; a flipped token must be
     * REFUSED by the host's completion check; dropped awaits must DIFFER or refuse. */
    params[4] = (pn_tt_red("dispatch_skip_patch") ? 2u : 0u) |
                (pn_tt_red("dispatch_bad_token") ? 4u : 0u) |
                (pn_tt_red("dispatch_no_wait") ? 8u : 0u) |
                (pn_tt_red("dispatch_mcast_short") ? 32u : 0u) | td_lazy_heads_bit();
    if (ok && pn_tt_residency_enqueue(puller, TD_PULL_OPCODE, params, 1u) != 0) {
        /* AN IDLE PULLER IS RETIRED BY THE ENQUEUE (residency_idle_retire): a prefill longer
         * than the idle limit (~5 s: the serve's 1,630-token prompt is 7 s) between two
         * replayed tokens left the next token with no puller and the request failed
         * ("a segment did not launch"). Re-arm it, re-upload the program, retry once. */
        ok = !pn_tt_red("dispatch_no_rearm") && !puller->running && td_puller_arm(g_td.device) && td_upload_program(g_td.device) &&
             pn_tt_residency_enqueue(puller, TD_PULL_OPCODE, params, 1u) == 0;
        if (ok)
            g_td.puller_rearms++;
    }
    g_td.internal = false;
    if (ok) {
        pn_tt_host_gap_card_busy();
        g_td.next_segment = segment + 1u;
        g_td.launched_segments++;
    }
    return ok;
}

/** The completion window the model predicts for poll event `i` at token offset k: every
 *  word linear, the token word recomputed from the queue's nonce. */
static void td_synthesize_poll(size_t i, uint32_t k, void *dst, size_t bytes)
{
    const td_token *a = &g_td.model[0], *b = &g_td.model[1];
    const td_event *ea = &a->events[i], *eb = &b->events[i];
    const uint32_t *wa = &a->pool[ea->data_at], *wb = &b->pool[eb->data_at];
    const uint32_t offset = ea->addr - FT_TT_OP_QUEUE_BASE;
    uint32_t words[TD_DATA_WORDS];
    uint32_t n = (uint32_t)((bytes + 3u) / 4u), word, count = 0u;
    bool have_count = false;

    if (n > ea->words)
        n = ea->words;
    for (word = 0; word < n; word++) {
        words[word] = wa[word] + (wb[word] - wa[word]) * k;
        if (offset + word * 4u == 0x14u) {
            count = words[word];
            have_count = true;
        }
    }
    for (word = 0; word < n; word++)
        if (offset + word * 4u == 0x18u && have_count && count > 0u)
            words[word] = td_token_at(g_td.queues[ea->queue].nonce, count - 1u);
        else if (offset + word * 4u == 0x24u)
            words[word] = 0u; /* error: none (the puller checked the card's) */
    memcpy(dst, words, bytes < n * 4u ? bytes : n * 4u);
}

/** Wait for every launched segment on the puller (the card's own checks ran there). */
static bool td_await_puller(void)
{
    pn_tt_residency *puller = &g_td.device->puller;
    bool ok = true;

    if (g_td.next_segment == 0u || g_td.awaited_segment == g_td.next_segment)
        return true;
    g_td.internal = true;
    if (pn_tt_residency_await(puller, 5000u) != 0) {
        uint32_t header[16];
        const ft_tt_noc_access *bulk = &g_td.original[TD_ACC_BULK_L1];

        if (bulk->read(bulk->user, puller->core_x, puller->core_y, TD_PROGRAM_BASE, header,
                       sizeof(header)) == FT_TT_OK)
        {
            uint32_t q, w0 = 0u, w1 = 0u;

            fprintf(stderr, "pn_tt: DISPATCH REPLAY — segment %u failed on the puller: error %u "
                            "at entry +0x%x (last await count %u)\n",
                    g_td.next_segment - 1u, header[8], header[9], header[11]);
            if (header[9] / 4u + 1u < g_prog.count) {
                w0 = g_prog.words[header[9] / 4u];
                w1 = g_prog.words[header[9] / 4u + 1u];
            }
            fprintf(stderr, "pn_tt: DISPATCH REPLAY — that entry: type %u count %u queue word "
                            "0x%08x\n", w0 & 0xFFu, (w0 >> 8) & 0xFFFu, w1);
            for (q = 0; q < g_td.queue_count; q++) {
                uint32_t control[8];

                if (bulk->read(bulk->user, g_td.queues[q].x, g_td.queues[q].y,
                               FT_TT_OP_QUEUE_BASE, control, sizeof(control)) == FT_TT_OK)
                    fprintf(stderr, "pn_tt: DISPATCH REPLAY —   queue %u (%u,%u): head %u tail %u "
                                    "completed %u state 0x%x\n", q, g_td.queues[q].x,
                            g_td.queues[q].y, control[2], control[3], control[5], control[7]);
            }
        }
        ok = false;
    }
    g_td.internal = false;
    g_td.awaited_segment = g_td.next_segment;
    if (ok)
        pn_tt_host_gap_card_idle("segment");
    return ok;
}

static bool td_replay_access(uint8_t acc, uint16_t x, uint16_t y, uint64_t addr,
                             const void *data, size_t bytes, bool write)
{
    const td_token *a = &g_td.model[0];
    const uint8_t cls = td_classify(acc, addr, write);
    const td_event *e;

    if (g_td.cursor >= a->count) {
        td_diverge("the host made more accesses than the model has", g_td.cursor);
        g_td.replaying = false;
        return false;
    }
    e = &a->events[g_td.cursor];
    /* HOST GAP: before the token's first launch the card is idle — book the host time up to
     * each event under the class it reached (timing only; a no-op with the clock off). */
    if (g_td.next_segment == 0u)
        pn_tt_host_gap_split(td_class_name[cls]);
    if (cls != e->cls || x != e->x || y != e->y || bytes != e->bytes ||
        (cls != TD_Q_RING && addr != e->addr)) {
        if (g_td.divergences < 8u)
            fprintf(stderr, "pn_tt: DISPATCH REPLAY — the host did %s (%u,%u) 0x%llx %zu B; the "
                            "model's event %zu is %s (%u,%u) 0x%x %u B\n",
                    td_class_name[cls], x, y, (unsigned long long)addr, bytes, g_td.cursor,
                    td_class_name[e->cls], e->x, e->y, e->addr, e->bytes);
        td_diverge("the access is not the model's next event", g_td.cursor);
        g_td.replaying = false;
        return false;
    }
    if (td_is_boundary(cls)) {
        /* THE HOST TOUCHES DATA: everything the puller was asked for must be done. */
        if (!td_await_puller()) {
            td_diverge("the puller did not complete before a boundary", g_td.cursor);
            g_td.replaying = false;
            return false;
        }
        g_td.cursor++;
        return false;
    }
    if (g_event_segment[g_td.cursor] >= 0 &&
        (uint32_t)g_event_segment[g_td.cursor] == g_td.next_segment &&
        !td_launch_segment(g_td.next_segment)) {
        td_diverge("a segment did not launch", g_td.cursor);
        g_td.replaying = false;
        return false;
    }
    if (!write) {
        /* A COMPLETION POLL: answered from the model. The puller awaits the same count on
         * the card and checks the same token before it issues anything after it; the host
         * would only race it (a puller ahead of the host leaves the count PAST the host's). */
        td_synthesize_poll(g_td.cursor, g_td.replay_k, (void *)data, bytes);
        g_td.cursor++;
        return true;
    }
    if (!td_predict_match(g_td.cursor, g_td.replay_k, addr, data, bytes)) {
        if (g_td.divergences < 8u) {
            const td_event *ea = &a->events[g_td.cursor];
            const uint32_t *wa = &a->pool[ea->data_at];
            const uint32_t *wb = &g_td.model[1].pool[g_td.model[1].events[g_td.cursor].data_at];
            uint32_t word, host[TD_DATA_WORDS];
            const uint32_t n = (uint32_t)(bytes / 4u) < TD_DATA_WORDS ? (uint32_t)(bytes / 4u)
                                                                      : TD_DATA_WORDS;

            memcpy(host, data, n * 4u);
            fprintf(stderr, "pn_tt: DISPATCH REPLAY — mispredicted %s (%u,%u) 0x%llx %zu B",
                    td_class_name[cls], x, y, (unsigned long long)addr, bytes);
            if (cls == TD_Q_RING) {
                const td_queue *queue = &g_td.queues[ea->queue];
                const uint32_t index = ea->index +
                                       (g_td.model[1].events[g_td.cursor].index - ea->index) *
                                           g_td.replay_k;

                fprintf(stderr, " | model index %u mask %u -> slot address 0x%x",
                        index, queue->mask,
                        FT_TT_OP_QUEUE_BASE + TD_QUEUE_RING_OFFSET + (index & queue->mask) * 32u);
            }
            for (word = 0; word < n && word < 16u; word++)
                if (cls != TD_Q_RING && host[word] != wa[word] + (wb[word] - wa[word]) * g_td.replay_k)
                    fprintf(stderr, " w%u host %08x model %08x", word, host[word],
                            wa[word] + (wb[word] - wa[word]) * g_td.replay_k);
            fprintf(stderr, "\n");
        }
        td_diverge("the host's write is not the model's prediction (the puller wrote the "
                   "model's)", g_td.cursor);
    }
    g_td.cursor++;
    return true; /* swallowed: the puller wrote it */
}

static void td_replay_begin(pn_tt_device *device, uint64_t forward)
{
    td_wrap(device);
    g_td.cursor = 0u;
    g_td.polling = false;
    g_td.next_segment = 0u;
    g_td.awaited_segment = 0u;
    g_td.replay_k = (uint32_t)(forward - g_td.first_forward);
    g_td.replaying = true;
}

static void td_replay_end(void)
{
    if (!g_td.replaying)
        return;
    if (g_td.polling) {
        g_td.cursor++;
        g_td.polling = false;
    }
    g_td.replaying = false;
    if (g_td.cursor != g_td.model[0].count)
        td_diverge("the token ended before the model did", g_td.cursor);
    if (!td_await_puller())
        td_diverge("the last segment did not complete", g_td.cursor);
    g_td.replayed_tokens++;
    td_unwrap();
}

/** Why the two model tokens gave no model: the reason and the first events around it (both
 *  tokens), so a refused token names its own cause (qwen36 plan S2). */
static void td_explain_no_model(const char *why, size_t at)
{
    const size_t index = at % 500000000u;

    fprintf(stderr, "pn_tt: DISPATCH SHADOW — no model: %s (at %s%zu)\n", why,
            at >= 1000000000u ? "model 1 event " : "event ", index);
    for (unsigned m = 0; m < 2u; m++) {
        const td_token *t = &g_td.model[m];

        for (size_t i = index > 2u ? index - 2u : 0u; i < t->count && i <= index + 2u; i++) {
            const td_event *e = &t->events[i];

            fprintf(stderr, "pn_tt:   model %u event %zu: %s acc %u core (%u,%u) addr 0x%x bytes %u "
                            "queue %u index %u%s\n", m, i, td_class_name[e->cls], (unsigned)e->acc,
                    (unsigned)e->x, (unsigned)e->y, (unsigned)e->addr, (unsigned)e->bytes,
                    (unsigned)e->queue, (unsigned)e->index, i == index ? "   <--" : "");
        }
    }
}

/* ---------------------------------------------------------------------------------------
 * Token hooks
 * --------------------------------------------------------------------------------------- */

void pn_tt_dispatch_token_begin(pn_tt_device *device, uint64_t forward)
{
    const long observe = td_env_long("PN_TT_DISPATCH_RECORD");
    const long live = td_dispatch_live();
    const long shadow = live > 0 ? (td_env_long("PN_TT_DISPATCH_FROM") > 0
                                        ? td_env_long("PN_TT_DISPATCH_FROM") : 4)
                                 : td_env_long("PN_TT_DISPATCH_SHADOW");
    td_token *into = NULL;

    if (!device)
        return;
    /* A SHAPE-KEY CHANGE re-anchors the model exactly as a new sequence does: the token's
     * descriptors are not the linear continuation of the model's (split-K's plan at the new
     * depth), and the puller writes a prediction BEFORE the host's write can be compared, so
     * a replayed token across the change puts the wrong descriptors on the card. That token
     * and the learning pair after it are host-issued (exact either way). */
    if (g_td.shape_key != g_td.anchor_key) {
        if (g_td.reanchors++ < 8u && live > 0)
            fprintf(stderr, "pn_tt: DISPATCH — the token's shape key changed at forward %llu "
                            "(the attention plan): replay re-learns from here\n",
                    (unsigned long long)forward);
        g_td.anchor_key = g_td.shape_key;
        g_td.sequence_reset = true;
    }
    /* A NEW SEQUENCE re-anchors the model: positions restart, so every position patch would
     * be wrong. The model is re-learned on the sequence's own forwards `shadow`, `shadow+1`
     * (counted from its first forward) and replay resumes after them. */
    if (g_td.sequence_reset) {
        g_td.sequence_reset = false;
        g_td.sequence_first = forward;
        g_td.model_ok = false;
        g_td.puller_ready = false;
        g_td.relearns = 0u;
    }
    if (g_td.sequence_first == 0u)
        g_td.sequence_first = 1u;
    {
        const uint64_t first = g_td.sequence_first + (uint64_t)(shadow > 0 ? shadow : 1) - 1u;

        if (live > 0 && g_td.puller_ready && !g_td.replay_dead && forward >= first + 2u) {
            td_replay_begin(device, forward);
            return;
        }
        if (observe > 0 && forward == (uint64_t)observe)
            into = &g_td.current;
        if (shadow > 0 && forward >= first) {
            g_td.first_forward = first;
            if (forward == first)
                into = &g_td.model[0];
            else if (forward == first + 1u)
                into = &g_td.model[1];
            else if (g_td.model_ok)
                into = &g_td.current;
        }
    }
    if (!into)
        return;
    td_wrap(device);
    into->count = 0u;
    into->pool_count = 0u;
    g_td.into = into;
    g_td.recording = true;
}

/**
 * H: two model tokens that are not one shape (the first decode forward still carries the
 * prefill's tail) slide the window one forward instead of ending replay for the sequence:
 * the second token becomes the first of the next pair, and learning retries on the next
 * forward. `PN_TT_DISPATCH_RELEARN` slides at most (default 8, 0 = never, the old behaviour).
 */
static void td_slide_window(void)
{
    const long limit = getenv("PN_TT_DISPATCH_RELEARN") ? td_env_long("PN_TT_DISPATCH_RELEARN")
                                                         : 8;
    td_token keep;

    if (limit <= 0 || g_td.relearns >= (uint32_t)limit)
        return;
    keep = g_td.model[0];
    g_td.model[0] = g_td.model[1];
    g_td.model[1] = keep;
    g_td.sequence_first++;
    g_td.relearns++;
    fprintf(stderr, "pn_tt: DISPATCH SHADOW — window slid (%u of %ld): learning retries on "
                    "the next forward\n", g_td.relearns, limit);
}

void pn_tt_dispatch_token_end(pn_tt_device *device, uint64_t forward)
{
    const long observe = td_env_long("PN_TT_DISPATCH_RECORD");
    const long live = td_dispatch_live();
    const long shadow = live > 0 ? 1 : td_env_long("PN_TT_DISPATCH_SHADOW");

    if (g_td.replaying) {
        td_replay_end();
        return;
    }
    if (!g_td.recording)
        return;
    g_td.recording = false;
    td_unwrap();
    if (observe > 0 && forward == (uint64_t)observe)
        td_report(g_td.into, forward);
    if (shadow <= 0)
        return;
    if (g_td.into == &g_td.model[1]) {
        const char *why = NULL;
        size_t at = 0u;

        if (g_td.model[0].count != g_td.model[1].count)
            why = "event counts differ";
        else if (!td_resolve(&g_td.model[0]) || !td_resolve(&g_td.model[1]))
            why = "a queue write did not resolve (queue table full or a ring write not 32 B)";
        else if (!td_learn_queues_at(&at))
            why = "a ring token or slot did not decode (learn queues)";
        g_td.model_ok = why == NULL;
        if (g_td.model_ok) {
            size_t i;

            for (i = 0; i < g_td.model[0].count; i++)
                if (!td_same_shape(&g_td.model[0].events[i], &g_td.model[1].events[i])) {
                    g_td.model_ok = false;
                    why = "an event changed shape";
                    at = i;
                    break;
                }
        }
        if (why)
            td_explain_no_model(why, at);
        if (g_td.model_ok && (td_env_long("PN_TT_DISPATCH_DRY") > 0 || live > 0)) {
            g_td.puller_ready = td_build_program() && td_puller_arm(device) &&
                                td_upload_program(device);
            if (!g_td.puller_ready)
                fprintf(stderr, "pn_tt: DISPATCH DRY — no program or no puller\n");
            else
                td_index_segments();
            if (!g_event_segment_ready)
                g_td.puller_ready = false;
        }
        if (!g_td.model_ok) {
            fprintf(stderr, "pn_tt: DISPATCH SHADOW — the two model tokens are not one shape "
                            "(%zu vs %zu events) or a token word did not decode; no model\n",
                    g_td.model[0].count, g_td.model[1].count);
            td_slide_window();
        }
    } else if (g_td.into == &g_td.current && g_td.model_ok) {
        if (td_resolve(&g_td.current)) {
            td_shadow_check(forward);
            if (g_td.puller_ready &&
                (forward - g_td.first_forward) % (uint64_t)td_env_long("PN_TT_DISPATCH_DRY") == 0u)
                td_dry_check(device, forward);
        } else {
            g_td.shape_failures++;
        }
    }
}

void pn_tt_dispatch_sequence_reset(void)
{
    g_td.sequence_reset = true;
}

void pn_tt_dispatch_shape_key(uint64_t key)
{
    g_td.shape_key = key;
}

bool pn_tt_dispatch_quiet(void)
{
    return !g_streaming && !g_td.replaying && !g_td.recording;
}

void pn_tt_dispatch_report(void)
{
    if (td_env_long("PN_TT_DISPATCH_SHADOW") > 0)
        td_shadow_report();
    if (td_dispatch_live() > 0 && g_td.replayed_tokens && g_td.device &&
        g_td.device->puller.running) {
        uint32_t header[16];
        const ft_tt_noc_access *bulk = &g_td.device->runtime.bulk_l1;

        if (bulk->read(bulk->user, g_td.device->puller.core_x, g_td.device->puller.core_y,
                       TD_PROGRAM_BASE, header, sizeof(header)) == FT_TT_OK)
            fprintf(stderr, "pn_tt: DISPATCH PULLER — busy %.2f ms/token inside PULL "
                            "descriptors, of which %.2f ms awaiting targets\n",
                    (double)header[12] * 16.0 / 1.35e6 / (double)g_td.replayed_tokens,
                    (double)header[13] * 16.0 / 1.35e6 / (double)g_td.replayed_tokens);
    }
    if (td_dispatch_live() > 0)
        fprintf(stderr, "pn_tt: DISPATCH REPLAY — %llu tokens replayed by the puller, %llu "
                        "segments launched, %llu divergences%s%s\n",
                (unsigned long long)g_td.replayed_tokens,
                (unsigned long long)g_td.launched_segments,
                (unsigned long long)g_td.divergences,
                g_td.replay_dead ? " — replay OFF" : "",
                getenv("PN_TT_RED") ? " [RED armed]" : "");
    if (td_dispatch_live() > 0 && g_td.reanchors)
        fprintf(stderr, "pn_tt: DISPATCH REPLAY — %llu re-anchor(s) at a shape-key change "
                        "(the attention plan); learning restarted there\n",
                (unsigned long long)g_td.reanchors);
    if (td_env_long("PN_TT_DISPATCH_DRY") > 0)
        fprintf(stderr, "pn_tt: DISPATCH DRY — %llu tokens replayed on the puller in dry-run, "
                        "%llu FAILED, %llu log words compared; program %zu B, %u segments%s\n",
                (unsigned long long)g_dry_checked, (unsigned long long)g_dry_failed,
                (unsigned long long)g_dry_words, g_prog.count * 4u, g_prog.segments,
                pn_tt_red("dispatch_dry_k") ? " [RED dispatch_dry_k]" : "");
}

#endif /* USE_TT */
