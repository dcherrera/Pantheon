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
#include "../../core/bench.h"

#include "tt/ft_tt_op_l1_layout.h"

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
        if (offset >= TD_QUEUE_RING_OFFSET && offset < 0x9000u)
            return TD_Q_RING;
        if (offset >= 0x9000u && offset < 0xA000u)
            return TD_Q_PARAM_ROT;
        if (offset >= 0xA000u && offset < 0xF000u)
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

static bool td_learn_queues(void)
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
                if (t->pool[e->data_at + d * 8u + 1u] != td_token_at(queue->nonce, e->index + d))
                    return false;
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
                !pn_tt_red("dispatch_mask_infer"))
                return false;
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

/** Build the puller program from the two model tokens. @return false = refuse replay */
static bool td_build_program(void)
{
    const td_token *a = &g_td.model[0], *b = &g_td.model[1];
    size_t queue_table, segment_table, group_table, i;
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
    for (i = 0; i < a->count && g_prog.ok; i++) {
        const td_event *ea = &a->events[i], *eb = &b->events[i];
        const uint32_t *wa = &a->pool[ea->data_at], *wb = &b->pool[eb->data_at];

        if (td_is_boundary(ea->cls)) {
            if (in_segment)
                td_emit(5u); /* segment end */
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
    }
    if (in_segment)
        td_emit(5u);
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

static bool td_upload_program(pn_tt_device *device)
{
    const ft_tt_noc_access *bulk = &device->runtime.bulk_l1;

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
    return true;
}

static uint64_t g_dry_checked, g_dry_failed, g_dry_words;

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
#define TS_REGION_WORDS (((TD_PROGRAM_LIMIT - TD_PROGRAM_BASE) / 4u / TS_REGIONS) & ~3u)
#define TS_MAX_QUEUES 256u
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
} g_ts;

/* The puller kernel's PULL_AWAIT_TABLE (resident_puller.c). */
#define TS_AWAIT_TABLE 0x16D000u

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

/** Print the census by family (the core's label), totals over the run. */
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

static uint32_t ts_queue_of(uint16_t x, uint16_t y)
{
    uint32_t i;

    for (i = 0; i < g_ts.queue_count; i++)
        if (g_ts.queues[i].x == x && g_ts.queues[i].y == y)
            return i;
    if (g_ts.queue_count == TS_MAX_QUEUES)
        return TS_MAX_QUEUES;
    memset(&g_ts.queues[i], 0, sizeof(g_ts.queues[i]));
    g_ts.queues[i].x = x;
    g_ts.queues[i].y = y;
    return g_ts.queue_count++;
}

static bool ts_room(size_t words)
{
    return g_ts.words && TS_ENTRY_BASE + g_ts.count + words + 2u <= TS_MAX_WORDS;
}

static bool ts_flush(void);
static bool ts_launch(bool wait);

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

/** Wait until the puller has completed `target` PULLs (its completed count, read directly). */
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
        if (pn_time_ms() - started > 20000.0)
            return false;
    }
    g_ts.wait_ms += pn_time_ms() - started;
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
             * a fixed 256-queue table was 1,044 words a launch. */
            const uint32_t table = 16u, seg = table + 4u * g_ts.queue_count;
            const uint32_t entries = (seg + 1u + 3u) & ~3u;
            uint32_t *o = g_ts.out;

            memset(o, 0, (size_t)entries * 4u);
            o[0] = TD_PULL_MAGIC;
            o[1] = 1u;
            o[2] = g_ts.queue_count;
            o[3] = table * 4u;
            o[4] = 1u;
            o[5] = seg * 4u;
            for (q = 0; q < g_ts.queue_count; q++) {
                o[table + 4u * q] = (uint32_t)g_ts.queues[q].x | ((uint32_t)g_ts.queues[q].y << 16);
                o[table + 4u * q + 1u] = g_ts.queues[q].nonce;
                o[table + 4u * q + 2u] = 0xFFFFu;
            }
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
        memset(params, 0, sizeof(params));
        params[0] = base;
        /* Each queued PULL keeps ITS OWN parameter block: with one slot, a launch queued
         * behind another rewrote the base the earlier one had not read yet (chunked launches
         * skipped regions — error 4 at an await). */
        puller->param_slot = 1u + (g_ts.launched % 8u);
        if (ok && pn_tt_residency_enqueue(puller, TD_PULL_OPCODE, params, 1u) != 0)
            ok = false;
        puller->param_slot = 0u;
        g_ts_fresh = false;
        if (ok) {
            g_ts.launched++;
            g_ts.region_done_at[r] = g_ts.launched;
            g_ts.region = (r + 1u) % TS_REGIONS;
            g_ts.async_launches++;
        }
        g_ts.launched_words += total;
        g_ts.count = 0u;
        g_ts.segments++;
    }
    if (ok && wait && g_ts.launched != g_ts.completed) {
        if (pn_tt_residency_await(puller, 20000u) != 0) {
            uint32_t header[16];

            for (q = 0; q < TS_REGIONS; q++)
                if (bulk->read(bulk->user, puller->core_x, puller->core_y,
                               TD_PROGRAM_BASE + q * TS_REGION_WORDS * 4u, header,
                               sizeof(header)) == FT_TT_OK && header[8] != 0u)
                    fprintf(stderr, "pn_tt: STREAM — region %u failed on the puller: error %u at "
                                    "entry +0x%x (last await count %u)\n", q, header[8],
                            header[9], header[11]);
            ok = false;
        } else {
            g_ts.completed = g_ts.launched;
        }
    }
    g_td.internal = false;
    g_ts.flush_ms += pn_time_ms() - started;
    if (wait) {
        g_ts.last_flush_end = pn_time_ms();
        g_ts_fresh = true;
    }
    if (!ok)
        g_ts.failed = true;
    return ok;
}

/** Everything captured runs on the card and is done. */
static bool ts_flush(void)
{
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
        return ts_write_entry(x, y, addr, src, bytes);
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

static bool td_stream_read(uint8_t acc, uint16_t x, uint16_t y, uint64_t addr, void *dst,
                           size_t bytes)
{
    const uint8_t cls = td_classify(acc, addr, false);

    if (!g_ts.failed && cls == TD_Q_POLL && (bytes % 4u) == 0u) {
        const uint32_t q = ts_queue_of(x, y);
        ts_queue *queue = q < TS_MAX_QUEUES ? &g_ts.queues[q] : NULL;

        if (queue && queue->nonce_known && queue->issued_known && queue->issued > 0u) {
            uint32_t window[16], entry[4];
            const uint32_t first = (uint32_t)((addr - FT_TT_OP_QUEUE_BASE) / 4u);
            const size_t last = g_ts.count;
            uint32_t token = queue->nonce ^ (queue->issued * TD_TOKEN_MULTIPLIER);

            if (token == 0u || token == (uint32_t)FT_TT_QUEUE_STATE_HOST_POISON)
                token = queue->nonce ^ 0xA5A5A5A5u;
            /* The puller waits for the same count (once a run of polls). */
            if (!(last >= 4u && g_ts.words[TS_ENTRY_BASE + last - 4u] == 4u &&
                  g_ts.words[TS_ENTRY_BASE + last - 3u] == q &&
                  g_ts.words[TS_ENTRY_BASE + last - 2u] == queue->issued)) {
                entry[0] = 4u;
                entry[1] = q;
                /* RED `stream_await_early`: the puller waits for one descriptor LESS than the
                 * host was told is done — the next family starts on a producer's unfinished
                 * output; must DIFFER (never more than one behind, so no ring overrun). */
                entry[2] = queue->issued - (pn_tt_red("stream_await_early") && queue->issued > 1u
                                                ? 1u : 0u);
                entry[3] = 0u;
                g_ts.n_await++;
                if (!ts_emit(entry, 4u))
                    return false;
            }
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
        const bool flushed = ts_flush();

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

    entry[0] = 7u;
    memcpy(&entry[1], params, 9u * 4u);
    g_ts.n_rowcopy++;
    return ts_emit(entry, 10u);
}

static void ts_puller_clocks(uint32_t *busy, uint32_t *await);

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
    /* The streamed program overwrites the puller's recorded decode program: re-learn it. */
    g_td.sequence_reset = true;
    td_wrap(device);
    ts_puller_clocks(&g_ts.puller_busy0, &g_ts.puller_await0);
    if (ts_census_enabled())
        ts_census_read(g_ts.census0);
    g_streaming = true;
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
    ok = ts_flush() && !g_ts.failed;
    {
        uint32_t busy, wait;

        ts_puller_clocks(&busy, &wait);
        g_ts.puller_busy_ms += (double)(uint32_t)(busy - g_ts.puller_busy0) * 16.0 / 1.35e6;
        g_ts.puller_await_ms += (double)(uint32_t)(wait - g_ts.puller_await0) * 16.0 / 1.35e6;
        if (ts_census_enabled())
            ts_census_accumulate();
    }
    g_streaming = false;
    td_unwrap();
    /* What the host thinks it issued stays what it issued; a failure leaves no trust. */
    if (!ok)
        g_ts.queue_count = 0u;
    return ok;
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
    fprintf(stderr, "pn_tt: STREAM puller a cycle: %.1f ms inside PULLs, %.1f of them in AWAIT "
                    "polls (%.1f issuing)\n", g_ts.puller_busy_ms / cycles,
            g_ts.puller_await_ms / cycles, (g_ts.puller_busy_ms - g_ts.puller_await_ms) / cycles);
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
    if (pn_tt_residency_submit(&device->puller, opcode, params, 5000u) == 0)
        return 0;
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
                (pn_tt_red("dispatch_mcast_short") ? 32u : 0u);
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
    /* A NEW SEQUENCE re-anchors the model: positions restart, so every position patch would
     * be wrong. The model is re-learned on the sequence's own forwards `shadow`, `shadow+1`
     * (counted from its first forward) and replay resumes after them. */
    if (g_td.sequence_reset) {
        g_td.sequence_reset = false;
        g_td.sequence_first = forward;
        g_td.model_ok = false;
        g_td.puller_ready = false;
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
        g_td.model_ok = g_td.model[0].count == g_td.model[1].count &&
                        td_resolve(&g_td.model[0]) && td_resolve(&g_td.model[1]) &&
                        td_learn_queues();
        if (g_td.model_ok) {
            size_t i;

            for (i = 0; i < g_td.model[0].count; i++)
                if (!td_same_shape(&g_td.model[0].events[i], &g_td.model[1].events[i])) {
                    g_td.model_ok = false;
                    break;
                }
        }
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
        if (!g_td.model_ok)
            fprintf(stderr, "pn_tt: DISPATCH SHADOW — the two model tokens are not one shape "
                            "(%zu vs %zu events) or a token word did not decode; no model\n",
                    g_td.model[0].count, g_td.model[1].count);
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
    if (td_env_long("PN_TT_DISPATCH_DRY") > 0)
        fprintf(stderr, "pn_tt: DISPATCH DRY — %llu tokens replayed on the puller in dry-run, "
                        "%llu FAILED, %llu log words compared; program %zu B, %u segments%s\n",
                (unsigned long long)g_dry_checked, (unsigned long long)g_dry_failed,
                (unsigned long long)g_dry_words, g_prog.count * 4u, g_prog.segments,
                pn_tt_red("dispatch_dry_k") ? " [RED dispatch_dry_k]" : "");
}

#endif /* USE_TT */
