/* SPDX-License-Identifier: AGPL-3.0-or-later */
/* Copyright (C) 2026 David (Corey) Herrera */
/**
 * @file test_tt_group_pull.c
 * @brief S1 group pull (src/accel/tt/pn_tt_group_pull.h), no card. Each split family —
 *   eltwise (decode split, wide split, PM 7 deferral), reduce + reduce-member, render widen
 *   (and its PM 8 deferral), head-norm, attention lanes, GDN DELTA (and its S2c deferral); the
 *   KV append split at its words and deal — is driven through its real issue path against a FAKE
 *   CARD that runs queued descriptors the way the resident writers do: a member-0 descriptor
 *   whose a3 names more than one member completes only after every member it pulls (read off
 *   the pull table in member 0's L1) has posted the sequence in a4, and every descriptor stores
 *   its a5 in control word 56. A core runs only when its completion window is polled or a
 *   pull reads it, so a member nobody vouches for stays unfinished.
 *
 *   Knob on: the table on member 0, every member's done word zeroed once, the a3/a4/a5 words,
 *   EXACTLY ONE completion await (member 0's), every member finished when the call returns.
 *   Knob off (`PN_TT_GROUP_PULL=0`): a3/a4/a5 all zero and every member awaited. A table that
 *   does not read back: per-member awaits. The first card run's failures, each reproduced: a
 *   member idle-retired between a narrow and a wide split (it must not be pulled), a re-arm
 *   that loses member 0's table (it must be planted again), and prefill splits (out of the
 *   default scope). The RED child (`PN_TT_RED=group_pull_short`, its own
 *   process: PN_TT_RED is read once) runs the knob-on checks and must FAIL them.
 * Needs USE_TT=1, NOT a card.
 */
#include <stdbool.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#ifdef USE_TT

#include "../src/accel/tt/pn_tt_attention_lanes.h"
#include "../src/accel/tt/pn_tt_device_internal.h"
#include "../src/accel/tt/pn_tt_eltwise_group.h"
#include "../src/accel/tt/pn_tt_gdn.h"
#include "../src/accel/tt/pn_tt_group_pull.h"
#include "../src/accel/tt/pn_tt_head_norm.h"
#include "../src/accel/tt/pn_tt_kv_append_split.h"
#include "../src/accel/tt/pn_tt_render.h"
#include "tt/ft_tt_op_l1_layout.h"

#define CORES 24u
#define QUEUE_WORDS (0x10000u / 4u)
#define RING_SLOTS 16u
#define WORD_DONE 56u

static int g_checks, g_failures;

static void check(bool ok, const char *what)
{
    g_checks++;
    if (!ok) {
        g_failures++;
        printf("FAIL %s\n", what);
    }
}

/* ================================================================
 * The fake card: the queue region of every core, and the resident writers' rule
 * ================================================================ */

static uint32_t g_l1[CORES][QUEUE_WORDS];
static uint16_t g_x[CORES], g_y[CORES];
static uint32_t g_cores;
static uint32_t g_polls[CORES];        /* completion-window reads (host awaits) */
static uint32_t g_table_writes[CORES]; /* writes at control + 0xF900 */
static uint32_t g_done_zeroes[CORES];  /* writes at control + 0xE0 */
static bool g_corrupt_table;           /* the table reads back wrong */

static int fake_core(uint16_t x, uint16_t y)
{
    for (uint32_t c = 0; c < g_cores; c++)
        if (g_x[c] == x && g_y[c] == y)
            return (int)c;
    return -1;
}

/** Run core `c`'s published descriptors in order, as its writer would. */
static void fake_run(int c, uint32_t depth)
{
    uint32_t *control = g_l1[c];

    while (control[FT_TT_QUEUE_FIELD_COMPLETED_COUNT] < control[FT_TT_QUEUE_FIELD_HEAD]) {
        const uint32_t index = control[FT_TT_QUEUE_FIELD_COMPLETED_COUNT];
        const uint32_t *d = &control[(0x100u + (index & (control[1] - 1u)) * 32u) / 4u];
        const uint32_t pulled = d[5] & 0x3FFFFFFFu;

        if (pulled > 1u && (d[5] & 0x80000000u) == 0u) {
            const uint32_t *table = &control[PN_TT_GROUP_PULL_TABLE_OFFSET / 4u];

            if (table[0] != PN_TT_GROUP_PULL_MAGIC || pulled > table[1] || depth > 2u) {
                control[FT_TT_QUEUE_FIELD_ERROR] = 7u; /* kErrBadParameters */
                return;
            }
            for (uint32_t m = 1u; m < pulled; m++) {
                const int member = fake_core((uint16_t)(table[2u + m] & 0xFFFFu),
                                             (uint16_t)(table[2u + m] >> 16));

                if (member < 0)
                    return;
                fake_run(member, depth + 1u);
                if ((int32_t)(g_l1[member][WORD_DONE] - d[6]) < 0) {
                    control[FT_TT_QUEUE_FIELD_ERROR] = 4u; /* kErrSemWaitTimeout */
                    return;
                }
            }
        }
        control[WORD_DONE] = d[7];
        control[FT_TT_QUEUE_FIELD_COMPLETED_TOKEN] = d[1];
        control[FT_TT_QUEUE_FIELD_STATE] = FT_TT_QUEUE_STATE_IDLE;
        control[FT_TT_QUEUE_FIELD_COMPLETED_COUNT] = index + 1u;
    }
}

static int fake_write(void *user, uint16_t x, uint16_t y, uint64_t address, const void *src,
                      size_t bytes)
{
    const int c = fake_core(x, y);
    const uint64_t offset = address - FT_TT_OP_QUEUE_BASE;

    (void)user;
    if (c < 0 || address < FT_TT_OP_QUEUE_BASE || offset + bytes > 0x10000u)
        return FT_TT_OK; /* DRAM and the rest of L1 are not modelled */
    if (offset == PN_TT_GROUP_PULL_TABLE_OFFSET)
        g_table_writes[c]++;
    if (offset == PN_TT_GROUP_PULL_DONE_OFFSET)
        g_done_zeroes[c]++;
    memcpy((uint8_t *)g_l1[c] + offset, src, bytes);
    /* A stop request retires the triad at once: every processor's status slot reads EXIT. */
    if (offset == FT_TT_QUEUE_FIELD_STOP_REQUEST * 4u && bytes == 4u && g_l1[c][4] != 0u)
        for (uint32_t p = 0; p < FT_TT_PROC_COUNT; p++)
            g_l1[c][(FT_TT_QUEUE_STATUS_OFFSET + p * FT_TT_QUEUE_STATUS_BYTES) / 4u] =
                FT_TT_QUEUE_STATE_EXIT;
    return FT_TT_OK;
}

static int fake_read(void *user, uint16_t x, uint16_t y, uint64_t address, void *dst,
                     size_t bytes)
{
    const int c = fake_core(x, y);
    const uint64_t offset = address - FT_TT_OP_QUEUE_BASE;

    (void)user;
    if (c < 0 || address < FT_TT_OP_QUEUE_BASE || offset + bytes > 0x10000u)
        return FT_TT_ERROR;
    if (offset >= 0x10u && offset < 0x20u) {
        g_polls[c]++;
        fake_run(c, 0u);
    }
    memcpy(dst, (const uint8_t *)g_l1[c] + offset, bytes);
    if (offset == PN_TT_GROUP_PULL_TABLE_OFFSET && g_corrupt_table && bytes >= 8u)
        ((uint32_t *)dst)[1] ^= 1u;
    return FT_TT_OK;
}

/* ================================================================
 * Fake device and residencies
 * ================================================================ */

static pn_tt_device *fake_device(void)
{
    pn_tt_device *d = calloc(1, sizeof(*d));

    if (!d)
        return NULL;
    d->runtime.noc.write = fake_write;
    d->runtime.noc.read = fake_read;
    d->runtime.bulk_l1.write = fake_write;
    d->runtime.bulk_l1.read = fake_read;
    d->dram_bank_count = 8u;
    for (uint32_t b = 0; b < 8u; b++) {
        d->dram[b].x = (uint16_t)(30u + b);
        d->dram[b].y = 1u;
    }
    memset(g_l1, 0, sizeof(g_l1));
    g_cores = 0u;
    g_corrupt_table = false;
    d->single_row = true; /* a decode forward: the default pull scope */
    return d;
}

/** A running residency on a fresh core, with a STALE done word an earlier process left. */
static void fake_arm(pn_tt_device *d, pn_tt_residency *r, const char *name)
{
    const uint32_t c = g_cores++;

    memset(r, 0, sizeof(*r));
    snprintf(r->name, sizeof(r->name), "%s", name);
    r->runtime = &d->runtime;
    r->core_index = 100u + c;
    r->core_x = (uint16_t)(1u + c);
    r->core_y = 2u;
    g_x[c] = r->core_x;
    g_y[c] = r->core_y;
    if (ft_tt_queue_init(&d->runtime.noc, r->core_x, r->core_y, FT_TT_OP_QUEUE_BASE, RING_SLOTS,
                         0x1234u + c, &r->queue) != FT_TT_OK)
        return;
    g_l1[c][WORD_DONE] = 0x7FFFFFF0u;
    r->running = true;
    r->processor_mask = PN_TT_ONCARD_TRIAD_MASK;
    r->last_alive_seconds = pn_tt_now_seconds();
    r->await_shape = PN_TT_AWAIT_LINE;
    r->issue_shape = PN_TT_ISSUE_WORD;
}

static void reset_counters(void)
{
    memset(g_polls, 0, sizeof(g_polls));
    memset(g_table_writes, 0, sizeof(g_table_writes));
    memset(g_done_zeroes, 0, sizeof(g_done_zeroes));
}

static ft_tt_op_operand operand(const pn_tt_device *d, uint32_t bank, uint32_t address,
                                uint32_t tile_bytes)
{
    ft_tt_op_operand o;

    o.x = (uint16_t)d->dram[bank].x;
    o.y = (uint16_t)d->dram[bank].y;
    o.address = address;
    o.tile_bytes = tile_bytes;
    return o;
}

/** The last descriptor core `c` was sent, word `w` (a3 = 5, a4 = 6, a5 = 7). */
static uint32_t last_word(const pn_tt_residency *r, uint32_t w)
{
    const int c = fake_core(r->core_x, r->core_y);
    const uint32_t index = r->queue.issued - 1u;

    return g_l1[c][(0x100u + (index & (RING_SLOTS - 1u)) * 32u) / 4u + w];
}

static uint32_t polls(const pn_tt_residency *r)
{
    return g_polls[fake_core(r->core_x, r->core_y)];
}

/* ================================================================
 * The checks every family shares
 * ================================================================ */

/**
 * After one split of `set[0..members)` has returned. `table_members`: what the planted table
 * names (the eltwise set's whole width). `pulled`: the knob is on and the table read back.
 */
static void check_split(const char *family, pn_tt_residency *const *set, uint32_t members,
                        uint32_t table_members, bool pulled, bool first)
{
    const int c0 = fake_core(set[0]->core_x, set[0]->core_y);
    const uint32_t *table = &g_l1[c0][PN_TT_GROUP_PULL_TABLE_OFFSET / 4u];
    const uint32_t sequence = last_word(set[0], 7u);
    char what[160];
    bool table_ok = table[0] == PN_TT_GROUP_PULL_MAGIC && table[1] == table_members;
    bool words_ok = true, finished = true, zeroed = true, settled = true;
    uint32_t member_polls = 0u;

    for (uint32_t m = 0; m < table_members; m++)
        if (table[2u + m] != ((uint32_t)set[m]->core_x | ((uint32_t)set[m]->core_y << 16)))
            table_ok = false;
    for (uint32_t m = 0; m < members; m++) {
        const int c = fake_core(set[m]->core_x, set[m]->core_y);

        if (pulled) {
            words_ok = words_ok && last_word(set[m], 7u) == sequence && sequence != 0u &&
                       last_word(set[m], 5u) == (m == 0u ? members : 0u) &&
                       last_word(set[m], 6u) == (m == 0u ? sequence : 0u);
        } else {
            words_ok = words_ok && last_word(set[m], 5u) == 0u && last_word(set[m], 6u) == 0u &&
                       last_word(set[m], 7u) == 0u;
        }
        if (g_l1[c][FT_TT_QUEUE_FIELD_COMPLETED_COUNT] != set[m]->queue.issued)
            finished = false;
        if (first && pulled && g_done_zeroes[c] != 1u)
            zeroed = false;
        if (m > 0u) {
            member_polls += polls(set[m]);
            settled = settled && set[m]->unawaited == 0u;
        }
    }
    snprintf(what, sizeof(what), "%s: %s", family,
             pulled ? "the table on member 0 names every member, in order"
                    : "no table planted (a refused one once, never again)");
    if (pulled)
        check(table_ok && (!first || g_table_writes[c0] == 1u), what);
    else
        check(g_table_writes[c0] <= (first && g_corrupt_table ? 1u : 0u), what);
    snprintf(what, sizeof(what), "%s: %s", family,
             pulled ? "member 0 a3 = members, a4 = a5 = the sequence; members a5 only"
                    : "a3 / a4 / a5 all zero (today's descriptors)");
    check(words_ok, what);
    if (first && pulled) {
        snprintf(what, sizeof(what), "%s: every member's stale done word zeroed once", family);
        check(zeroed, what);
    }
    snprintf(what, sizeof(what), "%s: %s", family,
             pulled ? "ONE await, member 0's; no member polled"
                    : "every member awaited on its own");
    if (pulled)
        check(polls(set[0]) >= 1u && member_polls == 0u, what);
    else
        check(member_polls >= members - 1u, what);
    snprintf(what, sizeof(what), "%s: every member finished when the call returned", family);
    check(finished, what);
    snprintf(what, sizeof(what), "%s: the host holds nothing unawaited on any member", family);
    check(settled, what);
}

static void set_knob(bool on)
{
    if (on)
        unsetenv("PN_TT_GROUP_PULL");
    else
        setenv("PN_TT_GROUP_PULL", "0", 1);
}

/* ================================================================
 * The families
 * ================================================================ */

static void eltwise_request(const pn_tt_device *d, uint32_t tiles, ft_tt_op_eltwise_request *r)
{
    memset(r, 0, sizeof(*r));
    r->function = FT_TT_OP_VECTOR_FN_SILU_MUL;
    r->tile_count = tiles;
    r->prefetch_tiles = 8u;
    r->input_a = operand(d, 0u, 0x100000u, PN_TT_TILE_BF16);
    r->input_b = operand(d, 1u, 0x100000u, PN_TT_TILE_BF16);
    r->output = operand(d, 2u, 0x100000u, PN_TT_TILE_BF16);
}

static void test_eltwise(bool knob, bool corrupt)
{
    pn_tt_device *d = fake_device();
    pn_tt_residency *set[6];
    ft_tt_op_eltwise_request request;
    const bool pulled = knob && !corrupt;
    const char *family = corrupt ? "eltwise (table refused)" : "eltwise";

    if (!d) {
        check(false, "fake device");
        return;
    }
    set_knob(knob);
    g_corrupt_table = corrupt;
    fake_arm(d, &d->eltwise, "eltwise");
    for (uint32_t m = 0; m < 5u; m++)
        fake_arm(d, &d->eltwise_member[m], "eltwise-member");
    d->eltwise_members = 6u;
    d->eltwise_decode_members = 4u; /* members 4 and 5 are WIDE */
    d->eltwise_depth = 8u;
    for (uint32_t m = 0; m < 6u; m++)
        set[m] = m == 0u ? &d->eltwise : &d->eltwise_member[m - 1u];

    /* The decode split: four of the six-wide table. */
    reset_counters();
    eltwise_request(d, 128u, &request);
    check(pn_tt_eltwise_group_submit(d, &request) == 0, "eltwise decode submit");
    check_split(family, set, 4u, pulled ? 6u : 0u, pulled, true);
    check(polls(set[4]) == 0u && polls(set[5]) == 0u, "eltwise: a decode split leaves the wide "
                                                      "members alone");

    /* The wide split: all six, one sequence on. */
    reset_counters();
    eltwise_request(d, 2048u, &request);
    check(pn_tt_eltwise_group_submit(d, &request) == 0, "eltwise wide submit");
    check_split(corrupt ? "eltwise wide (table refused)" : "eltwise wide", set, 6u,
                pulled ? 6u : 0u, pulled, false);
    if (pulled)
        check(last_word(set[0], 7u) == 2u && g_table_writes[fake_core(set[0]->core_x,
                                                                       set[0]->core_y)] == 0u,
              "eltwise wide: sequence 2, the table is not planted again");

    /* PM 7: deferred, then settled — the await moves, it does not multiply. */
    reset_counters();
    eltwise_request(d, 128u, &request);
    pn_tt_eltwise_group_defer_next(d);
    check(pn_tt_eltwise_group_submit(d, &request) == 0 && polls(set[0]) == 0u,
          "eltwise deferred: nothing awaited at the issue");
    check(pn_tt_eltwise_group_settle(d) == 0, "eltwise deferred settle");
    check_split(corrupt ? "eltwise deferred (table refused)" : "eltwise deferred", set, 4u,
                pulled ? 6u : 0u, pulled, false);
    free(d);
}

static void test_reduce(bool knob)
{
    pn_tt_device *d = fake_device();
    pn_tt_residency *set[2];
    pn_tt_norm_constants constants = { 3u, 0x200000u, 8u };

    pn_tt_activation in, out;

    if (!d) {
        check(false, "fake device");
        return;
    }
    set_knob(knob);
    fake_arm(d, &d->reduce, "reduce");
    fake_arm(d, &d->reduce_member[0], "reduce-member");
    d->reduce_members = 2u;
    d->reduce_tile_cols = 8u;
    set[0] = &d->reduce;
    set[1] = &d->reduce_member[0];
    memset(&in, 0, sizeof(in));
    in.bank = 4u;
    in.address = 0x300000u;
    in.tile_count = 8u;
    out = in;
    out.bank = 5u;
    reset_counters();
    check(pn_tt_rms_norm(d, &constants, &in, &out) == 0, "reduce submit");
    check_split("reduce", set, 2u, knob ? 2u : 0u, knob, true);
    free(d);
}

/** Member m's parameter word `w` of its last descriptor (descriptor a0 is the block's L1). */
static uint32_t last_parameter(const pn_tt_residency *r, uint32_t w)
{
    const int c = fake_core(r->core_x, r->core_y);
    const uint32_t block = last_word(r, 2u);

    return g_l1[c][(block - FT_TT_OP_QUEUE_BASE) / 4u + w];
}

/** N (2026-11-19): the norm split over 4 members — one pull over all four, member m finishing
 *  slice m of 4 (kRedFinal byte 3 = 0x40 | m). */
static void test_reduce_wide(void)
{
    pn_tt_device *d = fake_device();
    pn_tt_residency *set[4];
    pn_tt_norm_constants constants = { 3u, 0x200000u, 8u };
    pn_tt_activation in, out;
    bool slices_ok = true;

    if (!d) {
        check(false, "fake device");
        return;
    }
    set_knob(true);
    fake_arm(d, &d->reduce, "reduce");
    set[0] = &d->reduce;
    for (uint32_t m = 1u; m < 4u; m++) {
        fake_arm(d, &d->reduce_member[m - 1u], "reduce-member");
        set[m] = &d->reduce_member[m - 1u];
    }
    d->reduce_members = 4u;
    d->reduce_tile_cols = 8u;
    memset(&in, 0, sizeof(in));
    in.bank = 4u;
    in.address = 0x300000u;
    in.tile_count = 8u;
    out = in;
    out.bank = 5u;
    reset_counters();
    check(pn_tt_rms_norm(d, &constants, &in, &out) == 0, "reduce x4 submit");
    check_split("reduce x4", set, 4u, 4u, true, true);
    for (uint32_t m = 0; m < 4u; m++)
        slices_ok = slices_ok &&
                    (last_parameter(set[m], FT_TT_OP_REDUCE_FINAL) >> 24) == (0x40u | m);
    check(slices_ok, "reduce x4: member m finishes slice m of 4");
    free(d);
}

static void test_render(bool knob)
{
    pn_tt_device *d = fake_device();
    pn_tt_render_residency *slot;
    pn_tt_residency *set[2];

    if (!d) {
        check(false, "fake device");
        return;
    }
    set_knob(knob);
    slot = &d->render[PN_TT_RENDER_WIDEN];
    fake_arm(d, &slot->residency, "render-widen");
    fake_arm(d, &slot->member[0], "render-widen-member");
    slot->members = 2u;
    slot->in_tile_bytes = PN_TT_TILE_BF16;
    slot->out_tile_bytes = PN_TT_TILE_FP32;
    slot->tiles = 64u;
    slot->depth = 8u;
    slot->zero_bank = 6u;
    slot->zero_address = 0x400000u;
    set[0] = &slot->residency;
    set[1] = &slot->member[0];
    reset_counters();
    check(pn_tt_render_widen(d, operand(d, 0u, 0x500000u, PN_TT_TILE_BF16),
                             operand(d, 1u, 0x500000u, PN_TT_TILE_FP32), 64u) == 0,
          "render submit");
    check_split("render-widen", set, 2u, knob ? 2u : 0u, knob, true);

    /* PM 8: the residencies defer; the settle completes the window. */
    reset_counters();
    pn_tt_render_defer(d, false, true);
    check(pn_tt_render_widen(d, operand(d, 0u, 0x500000u, PN_TT_TILE_BF16),
                             operand(d, 1u, 0x500000u, PN_TT_TILE_FP32), 64u) == 0 &&
              polls(set[0]) == 0u && polls(set[1]) == 0u,
          "render deferred: nothing awaited at the issue");
    check(pn_tt_render_settle(d, false) == 0, "render deferred settle");
    pn_tt_render_defer(d, false, false);
    check_split("render-widen deferred", set, 2u, knob ? 2u : 0u, knob, false);
    free(d);
}

static void test_head_norm(bool knob)
{
    pn_tt_device *d = fake_device();
    pn_tt_head_norm_residency *slot;
    pn_tt_norm_constants constants = { 3u, 0x200000u, 4u };
    pn_tt_residency *set[2];

    if (!d) {
        check(false, "fake device");
        return;
    }
    set_knob(knob);
    slot = &d->head_norm[0];
    fake_arm(d, &slot->residency, "head-norm");
    fake_arm(d, &slot->member[0], "head-norm-member");
    slot->tile_cols = 4u;
    slot->tile_bytes = PN_TT_TILE_BF16;
    slot->members = 2u;
    d->head_norm_count = 1u;
    set[0] = &slot->residency;
    set[1] = &slot->member[0];
    reset_counters();
    check(pn_tt_head_norm_operands(d, FT_TT_OP_REDUCE_FN_RMS_NORM, &constants, 16u,
                                   operand(d, 0u, 0x600000u, PN_TT_TILE_BF16),
                                   operand(d, 1u, 0x600000u, PN_TT_TILE_BF16), 64u, 64u) == 0,
          "head-norm submit");
    check_split("head-norm", set, 2u, knob ? 2u : 0u, knob, true);
    free(d);
}

/* ---- Stage B: attention lanes, GDN, the KV append split ---- */

static void test_attention_lanes(bool knob)
{
    pn_tt_device *d = fake_device();
    uint32_t words[8][FT_TT_OP_ATTENTION_WORDS];
    pn_tt_residency *set[4];

    if (!d) {
        check(false, "fake device");
        return;
    }
    set_knob(knob);
    memset(words, 0, sizeof(words));
    fake_arm(d, &d->attention, "attention");
    for (uint32_t m = 0; m < 3u; m++)
        fake_arm(d, &d->attention_member[m], "attention-member");
    d->attention_members = 4u;
    d->attn_kv_heads = 8u;
    for (uint32_t m = 0; m < 4u; m++)
        set[m] = m == 0u ? &d->attention : &d->attention_member[m - 1u];
    reset_counters();
    /* Eight KV groups over four lanes: two rounds, the second at sequence 2. */
    check(pn_tt_attention_issue_lanes(d, words, 8u) == 0, "attention lanes issue");
    check_split("attention lanes", set, 4u, knob ? 4u : 0u, knob, true);
    if (knob)
        check(last_word(set[0], 7u) == 2u, "attention lanes: one sequence a round");
    free(d);
}

static void test_gdn(bool knob)
{
    pn_tt_device *d = fake_device();
    pn_tt_residency *set[2];
    const pn_tt_gdn_shape shape = { 128u, 4u, 2u, 1024u, 4u, 1u };

    if (!d) {
        check(false, "fake device");
        return;
    }
    set_knob(knob);
    fake_arm(d, &d->gdn, "gdn");
    fake_arm(d, &d->gdn_member[0], "gdn-member");
    d->gdn_members = 2u;
    d->gdn_shape = shape;
    d->gdn_region = calloc(1u, sizeof(*d->gdn_region));
    if (!d->gdn_region) {
        check(false, "gdn region");
        free(d);
        return;
    }
    d->gdn_region[0].state.bank = 2u;
    d->gdn_region[0].state.address = 0x700000u;
    d->gdn_stream_out.bank = 3u;
    d->gdn_stream_out.address = 0x710000u;
    d->gdn_y.bank = 3u;
    d->gdn_y.address = 0x720000u;
    d->gdn_gate.bank = 4u;
    d->gdn_gate.address = 0x730000u;
    set[0] = &d->gdn;
    set[1] = &d->gdn_member[0];
    reset_counters();
    check(pn_tt_gdn_delta_descriptor_only(d, 0u) == 0, "gdn DELTA split submit");
    check_split("gdn DELTA", set, 2u, knob ? 2u : 0u, knob, true);

    /* S2c: deferred rows, awaited together — `gdn` alone when every row pulled. */
    reset_counters();
    check(pn_tt_gdn_defer(d, true) == 0 && pn_tt_gdn_delta_descriptor_only(d, 0u) == 0 &&
              pn_tt_gdn_delta_descriptor_only(d, 0u) == 0 && polls(set[0]) == 0u,
          "gdn deferred: two rows, nothing awaited at the issue");
    check(pn_tt_gdn_await_deferred(d) == 0, "gdn deferred await");
    (void)pn_tt_gdn_defer(d, false);
    check_split("gdn DELTA deferred", set, 2u, knob ? 2u : 0u, knob, false);
    free(d->gdn_region);
    free(d);
}

/** The append split's words (a member may run several descriptors), then the same deal run
 *  through the fake card the way pn_tt_kv_append_chunk_pair() issues and awaits it. */
static void test_kv_append(void)
{
    pn_tt_device *d = fake_device();
    pn_tt_group_pull pull;
    pn_tt_residency *set[4];
    uint32_t words[FT_TT_QUEUE_PARAM_WORDS];
    bool bad[4] = { false, false, false, false };
    bool ok = true, finished = true;
    uint32_t sequence, index;
    pn_tt_group_pull_words w;

    /* Six descriptors over four members: 0 and 4 on member 0, 1 and 5 on member 1. */
    w = pn_tt_kv_append_pull_words(0u, 6u, 4u, 9u);
    check(w.a3 == 0u && w.a4 == 0u && w.a5 == 8u, "kv append: member 0's first posts 8, no pull");
    w = pn_tt_kv_append_pull_words(4u, 6u, 4u, 9u);
    check(w.a3 == 4u && w.a4 == 9u && w.a5 == 9u, "kv append: member 0's last pulls all four");
    w = pn_tt_kv_append_pull_words(1u, 6u, 4u, 9u);
    check(w.a3 == 0u && w.a5 == 8u, "kv append: member 1's first does not satisfy the pull");
    w = pn_tt_kv_append_pull_words(5u, 6u, 4u, 9u);
    check(w.a3 == 0u && w.a5 == 9u, "kv append: member 1's last posts 9");
    w = pn_tt_kv_append_pull_words(0u, 3u, 4u, 9u);
    check(w.a3 == 3u && w.a4 == 9u, "kv append: three descriptors pull only the three used");
    if (!d) {
        check(false, "fake device");
        return;
    }
    set_knob(true);
    memset(&pull, 0, sizeof(pull));
    memset(words, 0, sizeof(words));
    fake_arm(d, &d->memory, "memory");
    for (uint32_t m = 0; m < 3u; m++)
        fake_arm(d, &d->memory_member[m], "memory-member");
    for (uint32_t m = 0; m < 4u; m++)
        set[m] = m == 0u ? &d->memory : &d->memory_member[m - 1u];
    reset_counters();
    ok = pn_tt_group_pull_decide(&pull, true, set, 4u, 4u, "memory");
    sequence = pn_tt_group_pull_next(&pull);
    for (index = 0u; ok && index < 6u; index++) {
        w = pn_tt_kv_append_pull_words(index, 6u, 4u, sequence);
        ok = pn_tt_residency_enqueue_pull(set[pn_tt_kv_append_member_of(index, 4u)],
                                          FT_TT_QUEUE_OP_KV_APPEND, words, w.a3, w.a4,
                                          w.a5) == 0;
    }
    ok = ok && pn_tt_group_pull_await(&pull, true, set, 4u, bad, NULL) == 0;
    for (uint32_t m = 0; m < 4u; m++)
        if (g_l1[fake_core(set[m]->core_x, set[m]->core_y)][FT_TT_QUEUE_FIELD_COMPLETED_COUNT] !=
            set[m]->queue.issued)
            finished = false;
    check(ok && polls(set[0]) >= 1u && polls(set[1]) + polls(set[2]) + polls(set[3]) == 0u,
          "kv append: ONE await, member 0's");
    check(finished, "kv append: every member finished when the await returned");
    free(d);
}

/* ---- The first card run's failures (p150, 2026-11-16), each one reproduced ---- */

/** A fresh device with `n` armed residencies of the eltwise set (any family's set behaves the
 *  same through the helpers), and a zeroed pull. */
static pn_tt_device *helper_set(uint32_t n, pn_tt_residency **set, pn_tt_group_pull *pull)
{
    pn_tt_device *d = fake_device();

    if (!d)
        return NULL;
    set_knob(true);
    memset(pull, 0, sizeof(*pull));
    for (uint32_t m = 0; m < n; m++) {
        set[m] = m == 0u ? &d->eltwise : &d->eltwise_member[m - 1u];
        fake_arm(d, set[m], m == 0u ? "eltwise" : "eltwise-member");
    }
    return d;
}

static bool finished_clean(const pn_tt_residency *r)
{
    const int c = fake_core(r->core_x, r->core_y);

    return g_l1[c][FT_TT_QUEUE_FIELD_COMPLETED_COUNT] == r->queue.issued &&
           g_l1[c][FT_TT_QUEUE_FIELD_ERROR] == 0u;
}

/**
 * A NARROW split (CONV over 2 of the gdn set; the eltwise decode split) leaves the rest of the
 * set idle; the enqueue's idle retire then stops them at the WIDE split (DELTA over all 4; the
 * prefill SwiGLU). Before the fix member 0 had already been sent a pull of them and spun on
 * members that never got a descriptor: the puller timed out in prefill and gdn DELTA reported
 * "error". Now the split goes out unpulled and the refused members re-arm as they always did.
 */
static void test_member_retired_between_issues(void)
{
    pn_tt_residency *set[4];
    pn_tt_group_pull pull;
    uint32_t words[4][FT_TT_QUEUE_PARAM_WORDS];
    bool bad[4] = { false, false, false, false };
    pn_tt_device *d = helper_set(4u, set, &pull);
    bool pulled;

    if (!d) {
        check(false, "fake device");
        return;
    }
    memset(words, 0, sizeof(words));
    /* The narrow split: members 0 and 1 of the 4-wide table, pulled. */
    pulled = pn_tt_group_pull_decide(&pull, true, set, 4u, 2u, "retire");
    (void)pn_tt_group_pull_enqueue(&pull, pulled, set, 2u, FT_TT_QUEUE_OP_GDN, words[0], bad);
    check(pulled && pn_tt_group_pull_await(&pull, pulled, set, 2u, bad, NULL) == 0,
          "retire: the narrow split (2 of 4) pulls and completes");
    /* Members 2 and 3 sat idle past the retire limit while it ran. */
    set[2]->last_alive_seconds = 1e-3;
    set[3]->last_alive_seconds = 1e-3;
    reset_counters();
    pulled = pn_tt_group_pull_decide(&pull, true, set, 4u, 4u, "retire");
    check(!pulled, "retire: a split with an idle-retired member goes out UNPULLED");
    (void)pn_tt_group_pull_enqueue(&pull, pulled, set, 4u, FT_TT_QUEUE_OP_GDN, words[0], bad);
    (void)pn_tt_group_pull_await(&pull, pulled, set, 4u, bad, NULL);
    check(bad[2] && bad[3] && !bad[0] && !bad[1],
          "retire: the retired members refuse (and re-arm), members 0 and 1 do not");
    check(last_word(set[0], 5u) == 0u && finished_clean(set[0]),
          "retire: member 0 was never sent a pull of a member that got no descriptor");
    free(d);
}

/**
 * A re-arm keeps the core but starts a new queue (new nonce), and member 0's L1 is not
 * promised to keep the table, nor a member's its done word. Before the fix the table was
 * planted once per set of cores: member 0, re-armed with its table gone, refused every pulled
 * descriptor; a member re-armed with a stale done word would satisfy the pull early.
 */
static void test_rearm_replants(void)
{
    pn_tt_residency *set[3];
    pn_tt_group_pull pull;
    uint32_t words[3][FT_TT_QUEUE_PARAM_WORDS];
    bool bad[3] = { false, false, false };
    pn_tt_device *d = helper_set(3u, set, &pull);
    int c0, c2;
    bool pulled;

    if (!d) {
        check(false, "fake device");
        return;
    }
    memset(words, 0, sizeof(words));
    c0 = fake_core(set[0]->core_x, set[0]->core_y);
    c2 = fake_core(set[2]->core_x, set[2]->core_y);
    pulled = pn_tt_group_pull_decide(&pull, true, set, 3u, 3u, "rearm");
    (void)pn_tt_group_pull_enqueue(&pull, pulled, set, 3u, FT_TT_QUEUE_OP_ELTWISE, words[0], bad);
    check(pulled && pn_tt_group_pull_await(&pull, pulled, set, 3u, bad, NULL) == 0,
          "rearm: the first split pulls");
    /* Member 0 and member 2 re-armed on their own cores: new queues, member 0's table gone,
     * member 2's done word left far ahead of the family's sequence. */
    memset(&g_l1[c0][PN_TT_GROUP_PULL_TABLE_OFFSET / 4u], 0, PN_TT_GROUP_PULL_TABLE_WORDS * 4u);
    (void)ft_tt_queue_init(&d->runtime.noc, set[0]->core_x, set[0]->core_y,
                           FT_TT_OP_QUEUE_BASE, RING_SLOTS, 0x9001u, &set[0]->queue);
    (void)ft_tt_queue_init(&d->runtime.noc, set[2]->core_x, set[2]->core_y,
                           FT_TT_OP_QUEUE_BASE, RING_SLOTS, 0x9002u, &set[2]->queue);
    g_l1[c2][WORD_DONE] = 0x7FFFFFF0u;
    reset_counters();
    pulled = pn_tt_group_pull_decide(&pull, true, set, 3u, 3u, "rearm");
    check(pulled && g_table_writes[c0] == 1u && g_done_zeroes[c2] == 1u &&
              g_l1[c2][WORD_DONE] == 0u,
          "rearm: a new nonce plants again — the table, and the re-armed member's done word");
    (void)pn_tt_group_pull_enqueue(&pull, pulled, set, 3u, FT_TT_QUEUE_OP_ELTWISE, words[0], bad);
    check(pn_tt_group_pull_await(&pull, pulled, set, 3u, bad, NULL) == 0 &&
              finished_clean(set[0]) && polls(set[1]) == 0u && polls(set[2]) == 0u,
          "rearm: the next split pulls and completes on member 0's await alone");
    free(d);
}

/** PREFILL: out of the default scope (its descriptors are long, and its idle members are the
 *  ones retired and re-armed). PN_TT_GROUP_PULL=2 puts it back in. */
static void test_prefill_scope(void)
{
    for (uint32_t everywhere = 0; everywhere < 2u; everywhere++) {
        pn_tt_device *d = fake_device();
        pn_tt_residency *set[4];
        ft_tt_op_eltwise_request request;

        if (!d) {
            check(false, "fake device");
            return;
        }
        if (everywhere)
            setenv("PN_TT_GROUP_PULL", "2", 1);
        else
            unsetenv("PN_TT_GROUP_PULL");
        d->single_row = false;
        d->verify_rows = 0u;
        fake_arm(d, &d->eltwise, "eltwise");
        for (uint32_t m = 0; m < 3u; m++)
            fake_arm(d, &d->eltwise_member[m], "eltwise-member");
        d->eltwise_members = 4u;
        d->eltwise_decode_members = 4u;
        d->eltwise_depth = 8u;
        for (uint32_t m = 0; m < 4u; m++)
            set[m] = m == 0u ? &d->eltwise : &d->eltwise_member[m - 1u];
        reset_counters();
        eltwise_request(d, 128u, &request);
        check(pn_tt_eltwise_group_submit(d, &request) == 0, "prefill eltwise submit");
        check_split(everywhere ? "prefill at PN_TT_GROUP_PULL=2" : "prefill (default scope)", set,
                    4u, everywhere ? 4u : 0u, everywhere != 0u, true);
        free(d);
    }
    unsetenv("PN_TT_GROUP_PULL");
}

/** Member 0's pull refused on the card (its table is gone): every member is awaited on its
 *  own, member 0 is reported bad for the re-arm, and the next issue plants the table again. */
static void test_member0_refused(void)
{
    pn_tt_device *d = fake_device();
    pn_tt_group_pull pull;
    pn_tt_residency *set[3];
    uint32_t words[3][FT_TT_QUEUE_PARAM_WORDS];
    bool bad[3] = { false, false, false };
    int c0;

    if (!d) {
        check(false, "fake device");
        return;
    }
    set_knob(true);
    memset(&pull, 0, sizeof(pull));
    memset(words, 0, sizeof(words));
    fake_arm(d, &d->eltwise, "eltwise");
    fake_arm(d, &d->eltwise_member[0], "eltwise-member");
    fake_arm(d, &d->eltwise_member[1], "eltwise-member");
    set[0] = &d->eltwise;
    set[1] = &d->eltwise_member[0];
    set[2] = &d->eltwise_member[1];
    c0 = fake_core(set[0]->core_x, set[0]->core_y);
    check(pn_tt_group_pull_decide(&pull, true, set, 3u, 3u, "refusal"),
          "refusal: the table plants");
    g_l1[c0][PN_TT_GROUP_PULL_TABLE_OFFSET / 4u] = 0u; /* L1 lost it */
    reset_counters();
    check(pn_tt_group_pull_enqueue(&pull, true, set, 3u, FT_TT_QUEUE_OP_ELTWISE, words[0],
                                   bad) == 0,
          "refusal: the split enqueues");
    check(pn_tt_group_pull_await(&pull, true, set, 3u, bad, NULL) != 0 && bad[0] && !bad[1] &&
              !bad[2],
          "refusal: member 0 is reported bad, the members are not");
    check(polls(set[1]) >= 1u && polls(set[2]) >= 1u,
          "refusal: every member awaited on its own");
    check(pull.members == 0u && !pull.table_ok, "refusal: the table is forgotten");
    reset_counters();
    check(pn_tt_group_pull_decide(&pull, true, set, 3u, 3u, "refusal") &&
              g_table_writes[c0] == 1u &&
              g_l1[c0][PN_TT_GROUP_PULL_TABLE_OFFSET / 4u] == PN_TT_GROUP_PULL_MAGIC,
          "refusal: the next issue plants it again");
    free(d);
}

/* ================================================================
 * The words on their own
 * ================================================================ */

static void test_words(void)
{
    pn_tt_group_pull_words w = pn_tt_group_pull_words_for(0u, 4u, 9u, false);

    check(w.a3 == 4u && w.a4 == 9u && w.a5 == 9u, "member 0: a3 members, a4 = a5 = sequence");
    w = pn_tt_group_pull_words_for(3u, 4u, 9u, false);
    check(w.a3 == 0u && w.a4 == 0u && w.a5 == 9u, "member 3: a5 only");
    w = pn_tt_group_pull_words_for(0u, 4u, 9u, true);
    check(w.a3 == 3u && w.a4 == 9u, "RED group_pull_short: member 0 one member short");
}

static void run_families(void)
{
    test_eltwise(true, false);
    test_reduce(true);
    test_render(true);
    test_head_norm(true);
    test_attention_lanes(true);
    test_gdn(true);
    test_kv_append();
}

int main(int argc, char **argv)
{
    if (argc > 1 && strcmp(argv[1], "--red-child") == 0) {
        run_families();
        printf("test_tt_group_pull (red child): %d checks, %d failed\n", g_checks, g_failures);
        return g_failures ? 1 : 0;
    }
    test_words();
    test_reduce_wide();
    run_families();
    test_eltwise(false, false);
    test_reduce(false);
    test_render(false);
    test_head_norm(false);
    test_attention_lanes(false);
    test_gdn(false);
    test_eltwise(true, true);
    test_member0_refused();
    test_member_retired_between_issues();
    test_rearm_replants();
    test_prefill_scope();
    {
        char command[1024];
        int rc;

        snprintf(command, sizeof(command),
                 "PN_TT_RED=group_pull_short %s --red-child > /dev/null 2>&1", argv[0]);
        rc = system(command);
        check(rc != 0, "the RED child (group_pull_short) FAILS the knob-on checks");
    }
    printf("test_tt_group_pull: %d checks, %d failed\n", g_checks, g_failures);
    return g_failures ? 1 : 0;
}

#else
int main(void)
{
    printf("test_tt_group_pull: needs USE_TT=1\n");
    return 0;
}
#endif
