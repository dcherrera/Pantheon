/* SPDX-License-Identifier: AGPL-3.0-or-later */
/* Copyright (C) 2026 David (Corey) Herrera */
/**
 * @file test_tt_swiglu_fuse.c
 * @brief S3b (fused SwiGLU in the gate/up members), no card:
 *   DEFAULT (a child, both knobs unset): PN_TT_FUSE_SWIGLU is ON, PN_TT_FUSE_RESADD off.
 *   KNOB OFF (this process, PN_TT_FUSE_SWIGLU=0): the program is rtriad_bfp8; the fused entry points refuse and
 *     write NOTHING; the members' table is never written; a plain projection's descriptors
 *     carry no fused bit and a1 = a2 = 0.
 *   KNOB ON (a child, PN_TT_FUSE_SWIGLU=1): the program is rtriad_bfp8_swiglu; every matmul
 *     member's replica table fans a write anywhere in the decode or the tall `ffn_swiglu` out
 *     to every mirror at the same offset (the kernel's lookup, ftr_resolve_destinations); a
 *     strip mirrored after the publish is NOT covered; one fused projection is ONE descriptor
 *     per member — the gate member's words + FLAG|ROW0 over word 15, the up shard in a1/a2 —
 *     and every unservable shape is refused with nothing written.
 *   REDS (each must FAIL the child's checks): PN_TT_RED=fuse_swiglu_mirror_skip (one mirror
 *     left out of the table), PN_TT_RED=fuse_swiglu_no_stage (bit 26 set on the words).
 *   S3a, RESADD (a second child, PN_TT_FUSE_RESADD=1 alone): the fused program is armed; one
 *     out-projection with the residual add fused is ONE descriptor per member — the
 *     projection's words + RESADD over word 15, the residual at the member's first column in
 *     a1/a2 — and every unservable shape is refused with nothing written. Knob off (this
 *     process): pn_tt_matvec_resadd refuses and writes nothing.
 *     RED PN_TT_RED=fuse_resadd_skip (bit 28 set) must FAIL the child.
 * Needs USE_TT=1, NOT a card.
 */
#include <stdbool.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#ifdef USE_TT

#include "../src/accel/tt/pn_tt_device_internal.h"
#include "../src/accel/tt/pn_tt_replica_map.h"
#include "../src/accel/tt/pn_tt_swiglu_fuse.h"
#include "../src/accel/tt/pn_tt_resadd_fuse.h"
#include "tt/ft_tt_op_l1_layout.h"
#include "tt/kernels/resident_matmul_block_protocol.h"
#include "tt/kernels/resident_matmul_swiglu_protocol.h"
#include "tt/kernels/resident_matmul_resadd_protocol.h"
#include "tt/kernels/resident_output_replicas.h"

#define BANKS 8u
#define MEMBERS 16u
#define KT 8u
#define NT 544u
#define COLS (NT / MEMBERS)

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
 * Fake device: per-core replica tables, every other write logged
 * ================================================================ */

typedef struct logged_write {
    uint16_t x, y;
    uint32_t address, bytes;
    uint32_t words[16];
} logged_write;

static logged_write g_log[4096];
static uint32_t g_log_count;
static uint32_t g_table[MEMBERS][FT_TT_REPLICA_TABLE_WORDS];
static uint32_t g_table_writes;

static int core_slot(uint16_t x, uint16_t y)
{
    if (y == 2u && x >= 1u && x < 1u + MEMBERS)
        return (int)x - 1;
    return -1;
}

static int fake_write(void *user, uint16_t x, uint16_t y, uint64_t address, const void *src,
                      size_t bytes)
{
    const int slot = core_slot(x, y);

    (void)user;
    if (slot >= 0 && address == FT_TT_REPLICA_TABLE_ADDRESS && bytes <= sizeof(g_table[0])) {
        memcpy(g_table[slot], src, bytes);
        g_table_writes++;
        return FT_TT_OK;
    }
    if (g_log_count < 4096u) {
        logged_write *w = &g_log[g_log_count++];

        w->x = x;
        w->y = y;
        w->address = (uint32_t)address;
        w->bytes = (uint32_t)bytes;
        memset(w->words, 0, sizeof(w->words));
        memcpy(w->words, src, bytes < sizeof(w->words) ? bytes : sizeof(w->words));
    }
    return FT_TT_OK;
}

static int fake_read(void *user, uint16_t x, uint16_t y, uint64_t address, void *dst,
                     size_t bytes)
{
    const int slot = core_slot(x, y);

    (void)user;
    if (slot >= 0 && address == FT_TT_REPLICA_TABLE_ADDRESS && bytes <= sizeof(g_table[0])) {
        memcpy(dst, g_table[slot], bytes);
        return FT_TT_OK;
    }
    memset(dst, 0, bytes);
    return FT_TT_OK;
}

static pn_tt_device *fake_device(void)
{
    pn_tt_device *d = calloc(1, sizeof(*d));
    uint32_t m;

    if (!d)
        return NULL;
    d->dram_bank_count = BANKS;
    for (uint32_t b = 0; b < BANKS; b++) {
        d->dram[b].x = (uint16_t)(30u + b);
        d->dram[b].y = 1u;
    }
    d->runtime.noc.write = fake_write;
    d->runtime.noc.read = fake_read;
    d->runtime.bulk_l1.write = fake_write;
    d->runtime.bulk_l1.read = fake_read;
    d->group.member_count = MEMBERS;
    d->group.column_block = 4u;
    d->group.row_block = 1u;
    d->group.dst_section_tiles = 8u;
    for (m = 0; m < MEMBERS; m++) {
        pn_tt_matmul_member *slot = &d->group.member[m];

        slot->core_index = m;
        slot->core_x = (uint16_t)(1u + m);
        slot->core_y = 2u;
        slot->queue = &slot->own_queue;
        slot->running = true;
        (void)ft_tt_queue_init(&d->runtime.noc, slot->core_x, slot->core_y, FT_TT_OP_QUEUE_BASE,
                               8u, 0x1234u, slot->queue);
    }
    d->resident_depth = 4u;
    d->resident_tile_bytes = 1088u;
    d->resident_running = true;
    g_log_count = 0u;
    memset(g_table, 0, sizeof(g_table));
    g_table_writes = 0u;
    return d;
}

/** A strip of `tiles` x `rows`, mirrored `copies` times from bank `bank0` at `base`. */
static pn_tt_activation strip(uint32_t tiles, uint32_t rows, uint32_t copies, uint32_t bank0,
                              uint32_t base)
{
    pn_tt_activation s;

    memset(&s, 0, sizeof(s));
    s.tile_count = tiles;
    s.tile_rows = rows;
    s.active_rows = 1u;
    s.replica_count = copies;
    for (uint32_t i = 0; i < copies; i++) {
        s.replica_bank[i] = (bank0 + i) % BANKS;
        s.replica_address[i] = base + i * 0x08000000u;
    }
    s.bank = s.replica_bank[0];
    s.address = s.replica_address[0];
    return s;
}

static pn_tt_weight weight(uint32_t base, uint32_t bank_shift)
{
    pn_tt_weight w;

    memset(&w, 0, sizeof(w));
    w.tile_rows = KT;
    w.tile_cols = NT;
    w.tile_bytes = 1088u;
    w.shard_count = MEMBERS;
    for (uint32_t m = 0; m < MEMBERS; m++) {
        w.shard_tile_cols[m] = COLS;
        w.shard_column_first[m] = m * COLS;
        w.shard_address[m] = base + m * 0x10000u;
        w.shard_bank[m] = (m + bank_shift) % BANKS;
        w.shard_x[m] = (uint16_t)(30u + w.shard_bank[m]);
        w.shard_y[m] = 1u;
    }
    return w;
}

/** The descriptor (8 words) and its parameter block (16 words) member `m` was last sent. */
static bool sent_to(uint32_t m, uint32_t *descriptor, uint32_t *parameters)
{
    bool have = false;

    for (uint32_t i = 0; i < g_log_count; i++)
        if (core_slot(g_log[i].x, g_log[i].y) == (int)m && g_log[i].bytes == 32u &&
            g_log[i].words[0] == FT_TT_QUEUE_OP_MATMUL) {
            memcpy(descriptor, g_log[i].words, 8u * sizeof(uint32_t));
            have = true;
        }
    if (!have)
        return false;
    for (uint32_t i = 0; i < g_log_count; i++)
        if (core_slot(g_log[i].x, g_log[i].y) == (int)m && g_log[i].bytes == 64u &&
            g_log[i].address == descriptor[2]) {
            memcpy(parameters, g_log[i].words, 16u * sizeof(uint32_t));
            return true;
        }
    return false;
}

static uint32_t descriptors_logged(void)
{
    uint32_t n = 0u;

    for (uint32_t i = 0; i < g_log_count; i++)
        n += g_log[i].bytes == 32u && g_log[i].words[0] == FT_TT_QUEUE_OP_MATMUL ? 1u : 0u;
    return n;
}

/* ================================================================
 * Pure helpers (any knob)
 * ================================================================ */

static void test_helpers(void)
{
    pn_tt_weight gate = weight(0x200000u, 0u), up = weight(0x800000u, 3u), odd;
    pn_tt_device row;
    uint32_t a1 = 0u, a2 = 0u;

    check(pn_tt_swiglu_shards_pair(&gate, &up), "gate and up on one shard map pair");
    odd = up;
    odd.shard_tile_cols[3]++, odd.shard_tile_cols[4]--;
    check(!pn_tt_swiglu_shards_pair(&gate, &odd), "a different split of the columns does not");
    odd = up;
    odd.shard_count = 1u;
    check(!pn_tt_swiglu_shards_pair(&gate, &odd), "an unsplit weight does not");
    odd = up;
    odd.tile_rows = KT * 2u;
    check(!pn_tt_swiglu_shards_pair(&gate, &odd), "a different Kt does not");
    check(pn_tt_swiglu_block_bits(true, false) ==
              (FT_TT_MATMUL_SWIGLU_FLAG | FT_TT_MATMUL_SWIGLU_SILU_ROW0) &&
              pn_tt_swiglu_block_bits(false, true) ==
                  (FT_TT_MATMUL_SWIGLU_FLAG | FT_TT_MATMUL_SWIGLU_RED_NO_STAGE) &&
              (pn_tt_swiglu_block_bits(true, true) & 0x00FFFFFFu) == 0u,
          "word-15 bits are 24..26 and never touch the block word");
    pn_tt_swiglu_up_words(&up, 5u, &a1, &a2);
    check(a1 == up.shard_address[5] && ftsw_up_x(a2) == up.shard_x[5] &&
              ftsw_up_y(a2) == up.shard_y[5],
          "a1/a2 are member 5's up shard base and bank door");
    memset(&row, 0, sizeof(row));
    check(!pn_tt_eltwise_silu_row0(&row), "silu row 0: not for a multi-row forward");
    row.single_row = true;
    check(pn_tt_eltwise_silu_row0(&row), "silu row 0: a single-token forward");
    row.single_row = false;
    row.verify_rows = 4u;
    check(pn_tt_eltwise_silu_row0(&row), "silu row 0: a verify of 4 rows");
    row.verify_rows = 5u;
    check(!pn_tt_eltwise_silu_row0(&row), "silu row 0: not a verify of 5 rows (eltwise's rule)");

    /* S3a helpers. */
    {
        pn_tt_activation res = strip(160u, 1u, 1u, 3u, 0x300000u);
        pn_tt_activation out = strip(160u, 1u, 1u, 5u, 0x340000u);
        pn_tt_activation mirrored = strip(160u, 1u, BANKS, 5u, 0x380000u);

        check(pn_tt_resadd_block_bits(false) == FT_TT_MATMUL_RESADD_FLAG &&
                  pn_tt_resadd_block_bits(true) ==
                      (FT_TT_MATMUL_RESADD_FLAG | FT_TT_MATMUL_RESADD_RED_SKIP) &&
                  (pn_tt_resadd_block_bits(true) & (0x00FFFFFFu | FT_TT_MATMUL_SWIGLU_BITS)) == 0u,
              "resadd word-15 bits are 27..28, clear of the block word and the SwiGLU bits");
        pn_tt_resadd_words(&res, 70u, 33u, 1u, &a1, &a2);
        check(a1 == res.address + 70u * 2048u && ftra_residual_x(a2) == 33u &&
                  ftra_residual_y(a2) == 1u,
              "resadd a1/a2: the residual at the member's first column, and its bank door");
        check(pn_tt_resadd_strips_pair(&res, &out, 160u), "resadd: two unmirrored strips pair");
        check(!pn_tt_resadd_strips_pair(&res, &res, 160u), "resadd: not in place");
        check(!pn_tt_resadd_strips_pair(&res, &mirrored, 160u),
              "resadd: not into a mirrored output (the writer writes one destination)");
        check(!pn_tt_resadd_strips_pair(&res, &out, 161u), "resadd: not past the strip's width");
    }
}

/* ================================================================
 * Knob off: nothing moves
 * ================================================================ */

static void test_knob_off(void)
{
    pn_tt_device *d = fake_device();
    pn_tt_weight gate = weight(0x200000u, 0u), up = weight(0x800000u, 3u);
    pn_tt_activation normed = strip(KT, 2u, BANKS, 0u, 0x100000u);
    pn_tt_activation swiglu = strip(NT, 2u, BANKS, 2u, 0x400000u);
    const pn_tt_activation *fused[1] = { &swiglu };
    uint32_t descriptor[8], parameters[16];
    bool clean = true;

    if (!d) {
        check(false, "fake device");
        return;
    }
    check(strcmp(pn_tt_resident_program_name(), "rtriad_bfp8") == 0,
          "knob off: the matmul group arms rtriad_bfp8");
    pn_tt_replica_map_set_fused_swiglu(fused, 1u);
    check(pn_tt_replica_map_publish_fused(d) == 0 && g_table_writes == 0u,
          "knob off: the members' table is never written");
    d->single_row = true;
    check(pn_tt_matvec_enqueue_swiglu(d, &gate, &up, &normed, &swiglu) == 1 && g_log_count == 0u,
          "knob off: the fused projection is refused (run unfused), nothing written");
    check(pn_tt_matmul_group_enqueue_swiglu(d, &gate, &up, &normed, &swiglu) == FT_TT_EINVAL &&
              g_log_count == 0u,
          "knob off: the group refuses it too");
    check(pn_tt_matmul_group_enqueue(d, &gate, &normed, &swiglu) == FT_TT_OK &&
              descriptors_logged() == MEMBERS,
          "knob off: a plain projection is one descriptor per member");
    for (uint32_t m = 0; m < MEMBERS; m++)
        if (!sent_to(m, descriptor, parameters) ||
            (parameters[FT_TT_QUEUE_MATMUL_COLUMN_BLOCK] & 0xFF000000u) != 0u ||
            descriptor[3] != 0u || descriptor[4] != 0u)
            clean = false;
    check(clean, "knob off: no fused bit in word 15, a1 = a2 = 0 (today's words)");
    {
        pn_tt_activation res = strip(NT, 1u, 1u, 3u, 0x300000u);
        pn_tt_activation out = strip(NT, 1u, 1u, 5u, 0x340000u);

        g_log_count = 0u;
        check(pn_tt_matvec_resadd(d, &gate, &normed, &res, &out) == 1 && g_log_count == 0u,
              "knob off: the residual-add projection is refused (run unfused), nothing written");
        check(pn_tt_matmul_group_enqueue_resadd(d, &gate, &normed, &res, &out) == FT_TT_EINVAL &&
                  g_log_count == 0u,
              "knob off: the group refuses it too");
    }
    pn_tt_replica_map_set_fused_swiglu(NULL, 0u);
    free(d);
}

/* ================================================================
 * Knob on (child): the table and the descriptors
 * ================================================================ */

/** Destinations member `m`'s table resolves for a write `offset` bytes into `s`'s band 0, and
 *  whether each lands at the same offset in its own copy. */
static uint32_t fan_out(const pn_tt_device *d, uint32_t m, const pn_tt_activation *s,
                        uint32_t offset, bool *same)
{
    uint32_t x[FT_TT_REPLICA_MAX_DESTINATIONS], y[FT_TT_REPLICA_MAX_DESTINATIONS];
    uint32_t base[FT_TT_REPLICA_MAX_DESTINATIONS];
    const uint32_t n = ftr_resolve_destinations(g_table[m], d->dram[s->replica_bank[0]].x,
                                                d->dram[s->replica_bank[0]].y,
                                                s->replica_address[0] + offset, x, y, base);

    *same = true;
    for (uint32_t i = 0; i < n; i++)
        if (i >= s->replica_count || x[i] != d->dram[s->replica_bank[i]].x ||
            base[i] != s->replica_address[i] + offset)
            *same = false;
    return n;
}

static void test_fused_table(pn_tt_device *d, pn_tt_activation *decode, pn_tt_activation *tall)
{
    const pn_tt_activation *fused[2] = { decode, tall };
    pn_tt_activation other = strip(NT, 2u, BANKS, 5u, 0x600000u);
    pn_tt_activation late = strip(NT, 2u, 1u, 6u, 0x700000u);
    const pn_tt_activation *with_late[3] = { decode, tall, &late };
    bool same = false, all = true;
    uint32_t n, m;

    pn_tt_replica_map_set_fused_swiglu(fused, 2u);
    check(pn_tt_replica_map_publish_fused(d) == 0 && g_table_writes == MEMBERS,
          "knob on: the table lands on all 16 members");
    for (m = 0; m < MEMBERS; m++) {
        n = fan_out(d, m, decode, m * COLS * 2048u, &same);
        all = all && n == BANKS && same;
        n = fan_out(d, m, decode, (NT + m * COLS) * 2048u, &same);
        all = all && n == BANKS && same;
        n = fan_out(d, m, tall, (5u * NT + m * COLS + COLS - 1u) * 2048u, &same);
        all = all && n == BANKS && same;
    }
    check(all, "every member: its slice of decode bands 0/1 and tall band 5 reaches all 8 copies");
    n = fan_out(d, 0u, decode, 2u * NT * 2048u, &same);
    check(n == 1u, "one tile past the decode strip: not fanned out");
    check(pn_tt_replica_map_fused_covers(d, decode) && pn_tt_replica_map_fused_covers(d, tall),
          "the decode and the tall swiglu are covered");
    check(!pn_tt_replica_map_fused_covers(d, &other), "an unregistered mirrored strip is not");
    other.replica_count = 1u;
    check(pn_tt_replica_map_fused_covers(d, &other), "an unmirrored strip needs no table");
    /* Mirrored AFTER the publish: the table on the cards lacks it. */
    pn_tt_replica_map_set_fused_swiglu(with_late, 3u);
    (void)pn_tt_replica_map_publish_fused(d);
    late = strip(NT, 2u, BANKS, 6u, 0x700000u);
    check(!pn_tt_replica_map_fused_covers(d, &late),
          "a strip mirrored after the publish is NOT covered until the next one");
    (void)pn_tt_replica_map_publish_fused(d);
    check(pn_tt_replica_map_fused_covers(d, &late), "... and is after it");
    pn_tt_replica_map_set_fused_swiglu(fused, 2u);
    (void)pn_tt_replica_map_publish_fused(d);
    d->group.member[7].running = false;
    pn_tt_replica_map_set_fused_swiglu(fused, 2u);
    check(pn_tt_replica_map_publish_fused(d) == 0 && !pn_tt_replica_map_fused_covers(d, decode),
          "a member not running: nothing published, the fused path refused until the arm");
    d->group.member[7].running = true;
    check(pn_tt_replica_map_publish_fused(d) == 0 && pn_tt_replica_map_fused_covers(d, decode),
          "the arm's re-plant covers it again");
}

static void test_fused_issue(pn_tt_device *d, pn_tt_activation *decode)
{
    pn_tt_weight gate = weight(0x200000u, 0u), up = weight(0x800000u, 3u), odd = up;
    pn_tt_activation normed = strip(KT, 2u, BANKS, 0u, 0x100000u);
    pn_tt_activation two_rows = normed, unregistered = strip(NT, 2u, BANKS, 5u, 0x600000u);
    pn_tt_activation two_out;
    uint32_t descriptor[8], parameters[16], m;
    bool words = true, a12 = true, bits = true, pipe = true;
    int rc;

    d->single_row = true;
    g_log_count = 0u;
    rc = pn_tt_matmul_group_enqueue_swiglu(d, &gate, &up, &normed, decode);
    check(rc == FT_TT_OK && descriptors_logged() == MEMBERS,
          "one fused projection = ONE descriptor per member (16, not 32 + 4 eltwise)");
    for (m = 0; m < MEMBERS; m++) {
        if (!sent_to(m, descriptor, parameters)) {
            words = false;
            continue;
        }
        words = words && parameters[FT_TT_QUEUE_MATMUL_TILE_ROWS] == 1u &&
                parameters[FT_TT_QUEUE_MATMUL_TILE_INNER] == KT &&
                parameters[FT_TT_QUEUE_MATMUL_TILE_COLS] == COLS &&
                parameters[FT_TT_QUEUE_MATMUL_B_X] == gate.shard_x[m] &&
                parameters[FT_TT_QUEUE_MATMUL_B_BASE] == gate.shard_address[m] &&
                parameters[FT_TT_QUEUE_MATMUL_C_X] == d->dram[decode->bank].x &&
                parameters[FT_TT_QUEUE_MATMUL_C_BASE] == decode->address + m * COLS * 2048u &&
                parameters[FT_TT_QUEUE_MATMUL_A_BASE] ==
                    normed.replica_address[m % normed.replica_count];
        a12 = a12 && descriptor[3] == up.shard_address[m] &&
              descriptor[4] == ftsw_up_xy(up.shard_x[m], up.shard_y[m]);
        bits = bits && (parameters[FT_TT_QUEUE_MATMUL_COLUMN_BLOCK] & FT_TT_MATMUL_SWIGLU_BITS) ==
                           (FT_TT_MATMUL_SWIGLU_FLAG | FT_TT_MATMUL_SWIGLU_SILU_ROW0) &&
               (parameters[FT_TT_QUEUE_MATMUL_COLUMN_BLOCK] & 0x00FFFFFFu) == ftmm_block_word(1u, 4u);
        pipe = pipe && (parameters[FT_TT_QUEUE_MATMUL_B_N_MAJOR] & (1u << 8)) != 0u;
    }
    check(words, "each member: Mt 1, its 34 columns, the GATE shard in B, swiglu + first col in C");
    check(a12, "each member: a1/a2 = its UP shard's base and bank door");
    check(bits, "word 15 = the block word | FLAG | SILU_ROW0 (no RED bit)");
    check(pipe, "the pipelined reader is asked for (the fused reader needs it)");

    g_log_count = 0u;
    odd.shard_tile_cols[0]++, odd.shard_tile_cols[1]--;
    check(pn_tt_matmul_group_enqueue_swiglu(d, &gate, &odd, &normed, decode) == FT_TT_EINVAL,
          "refused: up split differently from gate");
    two_rows.active_rows = 2u;
    two_out = *decode;
    two_out.active_rows = 2u;
    check(pn_tt_matmul_group_enqueue_swiglu(d, &gate, &up, &two_rows, &two_out) == FT_TT_EINVAL,
          "refused: two tile rows (a prefill band pair)");
    check(pn_tt_matmul_group_enqueue_swiglu(d, &gate, &up, &normed, &unregistered) ==
              FT_TT_EINVAL,
          "refused: an output the members' table does not cover");
    d->group.g0_a1 = 0x80000010u;
    check(pn_tt_matmul_group_enqueue_swiglu(d, &gate, &up, &normed, decode) == FT_TT_EINVAL,
          "refused: a G0 stash (a1 is the stash's)");
    d->group.g0_a1 = 0u;
    d->group.cooperative = true;
    check(pn_tt_matmul_group_enqueue_swiglu(d, &gate, &up, &normed, decode) == FT_TT_EINVAL,
          "refused: a cooperating residency");
    d->group.cooperative = false;
    check(g_log_count == 0u, "a refusal writes nothing");
}

static int run_fused_child(void)
{
    pn_tt_device *d = fake_device();
    pn_tt_activation decode = strip(NT, 2u, BANKS, 2u, 0x400000u);
    pn_tt_activation tall = strip(NT, 64u, BANKS, 4u, 0x00800000u);

    if (!d) {
        check(false, "fake device");
        return 1;
    }
    check(strcmp(pn_tt_resident_program_name(), "rtriad_bfp8_swiglu") == 0,
          "knob on: the matmul group arms rtriad_bfp8_swiglu");
    tall.replica_address[0] = 0x00800000u; /* distinct from every decode copy */
    test_fused_table(d, &decode, &tall);
    test_fused_issue(d, &decode);
    pn_tt_replica_map_set_fused_swiglu(NULL, 0u);
    free(d);
    printf("test_tt_swiglu_fuse (knob-on child): %d checks, %d failed\n", g_checks, g_failures);
    return g_failures ? 1 : 0;
}

/* ================================================================
 * S3a RESADD (child, PN_TT_FUSE_RESADD=1): the descriptors
 * ================================================================ */

static int run_resadd_child(void)
{
    pn_tt_device *d = fake_device();
    pn_tt_weight w = weight(0x200000u, 0u);
    pn_tt_activation in = strip(KT, 2u, BANKS, 0u, 0x100000u);
    pn_tt_activation res = strip(NT, 2u, 1u, 3u, 0x300000u);
    pn_tt_activation out = strip(NT, 2u, 1u, 5u, 0x340000u);
    pn_tt_activation mirrored = strip(NT, 2u, BANKS, 5u, 0x380000u);
    pn_tt_activation two_rows = in, two_out = out;
    uint32_t descriptor[8], parameters[16], m;
    bool words = true, a12 = true, bits = true;
    int rc;

    if (!d) {
        check(false, "fake device");
        return 1;
    }
    check(strcmp(pn_tt_resident_program_name(), "rtriad_bfp8_swiglu") == 0,
          "resadd on: the matmul group arms the fused program");
    check(pn_tt_fuse_resadd_mode() == 1 && pn_tt_fuse_swiglu_mode() == 0,
          "resadd on, swiglu off: only the residual add fuses");
    d->single_row = true;
    rc = pn_tt_matmul_group_enqueue_resadd(d, &w, &in, &res, &out);
    check(rc == FT_TT_OK && descriptors_logged() == MEMBERS,
          "one residual-add projection = ONE descriptor per member (and no eltwise ADD)");
    for (m = 0; m < MEMBERS; m++) {
        if (!sent_to(m, descriptor, parameters)) {
            words = false;
            continue;
        }
        words = words && parameters[FT_TT_QUEUE_MATMUL_TILE_ROWS] == 1u &&
                parameters[FT_TT_QUEUE_MATMUL_TILE_INNER] == KT &&
                parameters[FT_TT_QUEUE_MATMUL_TILE_COLS] == COLS &&
                parameters[FT_TT_QUEUE_MATMUL_B_X] == w.shard_x[m] &&
                parameters[FT_TT_QUEUE_MATMUL_B_BASE] == w.shard_address[m] &&
                parameters[FT_TT_QUEUE_MATMUL_C_X] == d->dram[out.bank].x &&
                parameters[FT_TT_QUEUE_MATMUL_C_BASE] == out.address + m * COLS * 2048u &&
                (parameters[FT_TT_QUEUE_MATMUL_B_N_MAJOR] & (1u << 8)) != 0u;
        a12 = a12 && descriptor[3] == res.address + m * COLS * 2048u &&
              descriptor[4] == ftra_residual_xy(d->dram[res.bank].x, d->dram[res.bank].y);
        bits = bits &&
               (parameters[FT_TT_QUEUE_MATMUL_COLUMN_BLOCK] & 0xFF000000u) ==
                   FT_TT_MATMUL_RESADD_FLAG &&
               (parameters[FT_TT_QUEUE_MATMUL_COLUMN_BLOCK] & 0x00FFFFFFu) ==
                   ftmm_block_word(1u, 4u);
    }
    check(words, "each member: Mt 1, its 34 columns, its shard in B, the OUTPUT + first col in C");
    check(a12, "each member: a1/a2 = the RESIDUAL at its first column and the residual's bank");
    check(bits, "word 15 = the block word | RESADD (no RED bit, no SwiGLU bit)");

    g_log_count = 0u;
    check(pn_tt_matmul_group_enqueue_resadd(d, &w, &in, &res, &mirrored) == FT_TT_EINVAL,
          "refused: a mirrored output");
    check(pn_tt_matmul_group_enqueue_resadd(d, &w, &in, &res, &res) == FT_TT_EINVAL,
          "refused: in place");
    two_rows.active_rows = 2u;
    two_out.active_rows = 2u;
    check(pn_tt_matmul_group_enqueue_resadd(d, &w, &two_rows, &res, &two_out) == FT_TT_EINVAL,
          "refused: two tile rows (a prefill band pair)");
    d->group.g0_a1 = 0x80000010u;
    check(pn_tt_matmul_group_enqueue_resadd(d, &w, &in, &res, &out) == FT_TT_EINVAL,
          "refused: a G0 stash (a1 is the stash's)");
    d->group.g0_a1 = 0u;
    d->group.cooperative = true;
    check(pn_tt_matmul_group_enqueue_resadd(d, &w, &in, &res, &out) == FT_TT_EINVAL,
          "refused: a cooperating residency");
    d->group.cooperative = false;
    w.shard_count = 1u;
    check(pn_tt_matmul_group_enqueue_resadd(d, &w, &in, &res, &out) == FT_TT_EINVAL,
          "refused: an unsplit weight");
    check(g_log_count == 0u, "a refusal writes nothing");
    free(d);
    printf("test_tt_swiglu_fuse (resadd child): %d checks, %d failed\n", g_checks, g_failures);
    return g_failures ? 1 : 0;
}

int main(int argc, char **argv)
{
    char command[1024];
    int rc;

    if (argc > 1 && strcmp(argv[1], "--fused-child") == 0)
        return run_fused_child();
    if (argc > 1 && strcmp(argv[1], "--resadd-child") == 0)
        return run_resadd_child();
    if (argc > 1 && strcmp(argv[1], "--default-child") == 0) {
        check(pn_tt_fuse_swiglu_mode() == 1 && pn_tt_fuse_resadd_mode() == 0 &&
                  strcmp(pn_tt_resident_program_name(), "rtriad_bfp8_swiglu") == 0,
              "default: SwiGLU fusion on (the fused program), residual-add fusion off");
        printf("test_tt_swiglu_fuse (default child): %d checks, %d failed\n", g_checks,
               g_failures);
        return g_failures ? 1 : 0;
    }
    /* The knob-off half runs in THIS process: today's path is PN_TT_FUSE_SWIGLU=0. */
    setenv("PN_TT_FUSE_SWIGLU", "0", 1);
    unsetenv("PN_TT_FUSE_RESADD");
    test_helpers();
    test_knob_off();
    /* The knob and PN_TT_RED are read once a process, so the knob-on half is its own process;
     * a PN_TT_RED set by the caller (a red run) is inherited by it. */
    snprintf(command, sizeof(command), "PN_TT_FUSE_SWIGLU=1 PN_TT_MATMUL_PIPE=1 %s --fused-child",
             argv[0]);
    rc = system(command);
    check(rc == 0, "the knob-on child");
    snprintf(command, sizeof(command),
             "PN_TT_FUSE_SWIGLU=0 PN_TT_FUSE_RESADD=1 PN_TT_MATMUL_PIPE=1 %s --resadd-child",
             argv[0]);
    rc = system(command);
    check(rc == 0, "the resadd child");
    snprintf(command, sizeof(command),
             "env -u PN_TT_FUSE_SWIGLU -u PN_TT_FUSE_RESADD %s --default-child", argv[0]);
    rc = system(command);
    check(rc == 0, "the default child");
    printf("test_tt_swiglu_fuse: %d checks, %d failed\n", g_checks, g_failures);
    return g_failures ? 1 : 0;
}

#else
int main(void)
{
    printf("test_tt_swiglu_fuse: needs USE_TT=1\n");
    return 0;
}
#endif
