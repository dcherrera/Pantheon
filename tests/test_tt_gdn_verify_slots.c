/* SPDX-License-Identifier: AGPL-3.0-or-later */
/* Copyright (C) 2026 David (Corey) Herrera */
/**
 * @file test_tt_gdn_verify_slots.c
 * @brief The verify's GDN snapshot-slot map (pn_tt_gdn_verify_slots.h), legacy and compact,
 *        against a model of the foundry GDN kernels' read and write order; and the largest
 *        Qwen3.8 MTP contexts the compact map buys on one p150.
 *
 * (1) BOOKKEEPING + HAZARDS. Buffers carry a symbolic state: S_j = j, garbage = -1. Each row
 *     computes its output from what it READ, so a buffer clobbered before a read propagates.
 *     The kernel order modelled (foundry b5fee7b, tt/kernels):
 *     - per-row walk: descriptor r reads the live part, then writes its target; every read of
 *       a descriptor completes (NoC read barrier) before its compute, and descriptors run in
 *       order, so a later descriptor's write follows every earlier descriptor's read.
 *     - multi-row DELTA (read_one_delta_rows): the state is read per head at row 0 only; a
 *       head's writes follow its read. Only S_0's buffer is read.
 *     - multi-row CONV (read_one_conv_rows): rows in order; row r pass A reads sequence
 *       elements r..r+taps-2, pass B elements r+1..r+taps-2 in order, element e < taps-1 being
 *       LIVE window plane e (S_0's buffer). Row r's rolled-window plane p is written after its
 *       pass-B tap p was read. WORST CASE: a row's writes land before any later row's reads.
 *     Every rows 1..16, every walk mix, every rollback point, conv taps 2..8. The naive
 *     "row 0 in place" map is run through the same model and must be flagged (the model can
 *     fail).
 * (2) RED: `PN_TT_RED=gdn_compact_rollback_slot` must fail (1).
 * (3) CENSUS at the profiles: Gate 5's bank-0 model: free
 *     after the mandatory reserve with the MTP head = 191.5 MiB - 8.5 MiB per 1,024 positions;
 *     a twin set (a verify slot or the checkpoint) = 40.5 MiB a bank. Legacy must reproduce
 *     Gate 5's measured table; compact's predictions are printed and checked.
 *
 * Build: `make tests/test_tt_gdn_verify_slots` (no card, no USE_TT needed).
 *
 * @status None
 * @issues None
 * @todo None
 */

#include <stdbool.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>

#include "../src/accel/tt/pn_tt_gdn_verify_slots.h"

#define MAX_ROWS 16
#define MAX_BUFFERS (MAX_ROWS + 1)

static int g_checks, g_failures;

static void check(bool ok, const char *what)
{
    g_checks++;
    if (!ok) {
        g_failures++;
        printf("FAIL %s\n", what);
    }
}

/** One part (window or state) of one layer: buffers, the live pointer, the slot pointers. */
typedef struct {
    int content[MAX_BUFFERS];
    int live;
    int slot[MAX_ROWS];
    int slots;
    bool hazard; /**< a read saw a buffer a write had already clobbered */
} verify_part;

/** How the map under test is chosen: the production map, or the naive row-0-in-place one. */
typedef enum { MAP_LEGACY, MAP_COMPACT, MAP_NAIVE_ROW0 } map_kind;

static void part_init(verify_part *p, uint32_t slots)
{
    p->slots = (int)slots;
    p->live = 0;
    p->content[0] = 0;
    for (int i = 1; i < MAX_BUFFERS; i++)
        p->content[i] = -1;
    for (int s = 0; s < MAX_ROWS; s++)
        p->slot[s] = s < (int)slots ? s + 1 : -1;
    p->hazard = false;
}

static int naive_rows_out_slot(uint32_t row)
{
    return row == 0u ? -1 : (int)row - 1;
}

static int map_rows_out(map_kind map, uint32_t rows, uint32_t row)
{
    if (map == MAP_NAIVE_ROW0)
        return naive_rows_out_slot(row);
    return pn_gdn_verify_rows_out_slot(map == MAP_COMPACT, rows, row);
}

/** The per-row walk: in order, each row reads the live part then writes its target. */
static bool walk_rows(verify_part *p, bool compact, uint32_t rows)
{
    for (uint32_t r = 0; r < rows; r++) {
        const int slot = pn_gdn_verify_row_slot(compact, rows, r);
        const int read = p->content[p->live];
        const int out = read < 0 ? -1 : read + 1;
        int t;

        if (slot >= p->slots)
            return false;
        if (slot < 0) {
            p->content[p->live] = out;
            continue;
        }
        p->content[p->slot[slot]] = out;
        t = p->live;
        p->live = p->slot[slot];
        p->slot[slot] = t;
    }
    return true;
}

/** A multi-row descriptor's target buffer for row r (live = S_0's buffer at its start). */
static int rows_target(const verify_part *p, map_kind map, uint32_t rows, uint32_t row)
{
    const int slot = map_rows_out(map, rows, row);

    return slot < 0 ? p->live : (slot < p->slots ? p->slot[slot] : -2);
}

/** Multi-row DELTA: S_0 read per head at row 0 (before that head's writes); rows carried. */
static bool walk_delta_multi(verify_part *p, map_kind map, uint32_t rows)
{
    const int start = p->content[p->live];

    for (uint32_t r = 0; r < rows; r++) {
        const int target = rows_target(p, map, rows, r);

        if (target == -2)
            return false;
        p->content[target] = start < 0 ? -1 : start + (int)r + 1;
    }
    return true;
}

/** Multi-row CONV at `taps`, worst-case timing: row w's writes before rows > w read. */
static bool walk_conv_multi(verify_part *p, map_kind map, uint32_t rows, uint32_t taps)
{
    const int s0_buffer = p->live, start = p->content[p->live];
    bool clobbered[MAX_ROWS] = { false }; /* S_0's buffer, plane by plane: < taps - 1 planes */

    for (uint32_t r = 0; r < rows; r++) {
        const int target = rows_target(p, map, rows, r);
        bool bad = start < 0;

        if (target == -2)
            return false;
        /* Pass A: elements r..r+taps-2; element e < taps-1 is S_0's plane e. */
        for (uint32_t e = r; e + 1u < taps && e <= r + taps - 2u; e++)
            bad |= clobbered[e];
        /* Pass B, tap q (q = 0..taps-3) reads element q+1+r, then plane q is written. */
        for (uint32_t q = 0; q + 2u < taps; q++) {
            const uint32_t e = q + 1u + r;

            if (e + 1u < taps)
                bad |= clobbered[e];
            if (target == s0_buffer)
                clobbered[q] = true;
        }
        if (target == s0_buffer)
            clobbered[taps - 2u] = true; /* the newest plane: the row's own stream */
        p->hazard |= bad;
        p->content[target] = bad ? -1 : start + (int)r + 1;
    }
    return true;
}

static void multi_trades(verify_part *p, map_kind map, uint32_t rows)
{
    if (map == MAP_COMPACT || map == MAP_NAIVE_ROW0)
        return;
    for (uint32_t r = 0; r < rows; r++) {
        const int t = p->live;

        p->live = p->slot[r];
        p->slot[r] = t;
    }
}

/** The naive map's end state is fixed up by pointer moves only (no data), as a real
 *  implementation would: live <- slot rows-2 (S_rows), S_1's buffer into slot 1 mod (rows-1). */
static void naive_fixup(verify_part *p, uint32_t rows)
{
    int moved[MAX_ROWS];

    if (rows < 2u)
        return;
    for (uint32_t j = 1; j < rows; j++) /* S_j sits in slot j-2 (j >= 2), S_1 in the old live */
        moved[j % (rows - 1u)] = j == 1u ? p->live : p->slot[j - 2u];
    p->live = p->slot[rows - 2u];
    for (uint32_t s = 0; s + 1u < rows; s++)
        p->slot[s] = moved[s];
}

/** Run one verify of `rows` on a part: `multi` picks the multi-row descriptor (if rows >= 2)
 *  over the per-row walk; `conv` the CONV kernel model over the DELTA one. */
static bool run_part(verify_part *p, map_kind map, uint32_t rows, bool multi, bool conv,
                     uint32_t taps)
{
    const bool compact = map != MAP_LEGACY;

    if (!multi || rows < 2u)
        return walk_rows(p, compact, rows);
    if (!(conv ? walk_conv_multi(p, map, rows, taps) : walk_delta_multi(p, map, rows)))
        return false;
    multi_trades(p, map, rows);
    if (map == MAP_NAIVE_ROW0)
        naive_fixup(p, rows);
    return true;
}

/** Every rollback point of one verify, on copies: live must become S_(j+1). */
static bool rollbacks_ok(const verify_part *after, bool compact, uint32_t rows)
{
    if (after->content[after->live] != (int)rows)
        return false;
    for (uint32_t j = 0; j + 1u < rows; j++) {
        const uint32_t slot = pn_gdn_verify_rollback_slot(compact, rows, j);

        if ((int)slot >= after->slots || after->content[after->slot[slot]] != (int)j + 1)
            return false;
    }
    return true;
}

/** (1): every rows, walk mix and taps under `map`. @return combinations that failed */
static int bookkeeping_failures(map_kind map, bool count_hazards)
{
    int failed = 0;

    for (uint32_t rows = 1; rows <= MAX_ROWS; rows++)
        for (uint32_t taps = 2; taps <= 8; taps++)
            for (int mix = 0; mix < 4; mix++) {
                const bool compact = map != MAP_LEGACY;
                const uint32_t slots = pn_gdn_verify_slot_count(compact, rows);
                verify_part window, state;
                bool ok;

                part_init(&window, slots);
                part_init(&state, slots);
                ok = run_part(&window, map, rows, (mix & 1) != 0, true, taps) &&
                     run_part(&state, map, rows, (mix & 2) != 0, false, taps);
                ok = ok && rollbacks_ok(&window, compact, rows) &&
                     rollbacks_ok(&state, compact, rows) &&
                     !(count_hazards && (window.hazard || state.hazard));
                failed += !ok;
            }
    return failed;
}

static void check_bookkeeping(void)
{
    char what[160];
    int legacy, compact, naive, red;

    check(pn_gdn_verify_slot_count(false, 4u) == 4u && pn_gdn_verify_slot_count(true, 4u) == 3u,
          "4 rows (3 drafts): legacy 4 slots, compact 3");
    check(pn_gdn_verify_slot_count(true, 2u) == 1u && pn_gdn_verify_row_slot(true, 1u, 0u) == -1,
          "compact: 1 draft = 1 slot; a one-row verify writes in place");
    legacy = bookkeeping_failures(MAP_LEGACY, true);
    compact = bookkeeping_failures(MAP_COMPACT, true);
    snprintf(what, sizeof(what), "legacy map: every rows/mix/taps/rollback exact (%d failed)",
             legacy);
    check(legacy == 0, what);
    snprintf(what, sizeof(what), "compact map: every rows/mix/taps/rollback exact, no read of a "
             "clobbered buffer (%d failed)", compact);
    check(compact == 0, what);
    /* NEGATIVE CONTROL: the write-up's row-0-in-place map. The model must flag it (the
     * multi-row CONV re-reads live planes for rows 1..taps-2). */
    naive = bookkeeping_failures(MAP_NAIVE_ROW0, true);
    printf("  naive row-0-in-place map: %d of %d combinations wrong under the kernel model\n",
           naive, MAX_ROWS * 7 * 4);
    check(naive > 0, "the model flags the row-0-in-place map (it can fail)");
    setenv("PN_TT_RED", "gdn_compact_rollback_slot", 1);
    red = bookkeeping_failures(MAP_COMPACT, true);
    unsetenv("PN_TT_RED");
    printf("  RED gdn_compact_rollback_slot: %d combinations wrong\n", red);
    check(red > 0, "RED gdn_compact_rollback_slot is caught");
}

/* ================================================================
 * (3) The census at the profiles (Gate 5 bank-0 model)
 * ================================================================ */

#define BANK0_FREE_AT_0_MIB 191.5
#define BANK0_MIB_PER_1K 8.5
#define TWIN_MIB 40.5

/** Largest context (1,024 steps, up to 36,864) whose bank 0 holds `twins` sets; 0 = none. */
static uint32_t largest_ctx(uint32_t twins)
{
    uint32_t best = 0u;

    for (uint32_t ctx = 1024u; ctx <= 36864u; ctx += 1024u)
        if (BANK0_FREE_AT_0_MIB - BANK0_MIB_PER_1K * (double)ctx / 1024.0 >=
            (double)twins * TWIN_MIB)
            best = ctx;
    return best;
}

static void check_census(void)
{
    static const struct { uint32_t drafts; bool ckpt; uint32_t legacy, compact; } table[] = {
        /* legacy = Gate 5 measured (run or probe); compact = this change's prediction */
        { 3, true, 0, 3072 },    { 3, false, 3072, 8192 },
        { 2, true, 3072, 8192 }, { 2, false, 8192, 13312 },
        { 1, true, 8192, 13312 }, { 1, false, 13312, 17408 },
    };
    char what[160];

    printf("  largest context (bank-0 model)   legacy   compact\n");
    for (size_t i = 0; i < sizeof(table) / sizeof(table[0]); i++) {
        const uint32_t rows = table[i].drafts + 1u, ck = table[i].ckpt ? 1u : 0u;
        const uint32_t legacy = largest_ctx(pn_gdn_verify_slot_count(false, rows) + ck);
        const uint32_t compact = largest_ctx(pn_gdn_verify_slot_count(true, rows) + ck);

        printf("  %u draft(s), checkpoint %-3s       %6u    %6u\n", table[i].drafts,
               table[i].ckpt ? "on" : "off", legacy, compact);
        snprintf(what, sizeof(what), "%u drafts ckpt %d: legacy %u (Gate 5), compact %u",
                 table[i].drafts, (int)ck, table[i].legacy, table[i].compact);
        check(legacy == table[i].legacy && compact == table[i].compact, what);
    }
}

int main(void)
{
    check_bookkeeping();
    check_census();
    printf("test_tt_gdn_verify_slots: %d checks, %d failed\n", g_checks, g_failures);
    return g_failures ? 1 : 0;
}
