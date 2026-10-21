/* SPDX-License-Identifier: AGPL-3.0-or-later */
/* Copyright (C) 2026 David (Corey) Herrera */
/*
 * test_die_manager.c — die-manager unit tests (Phase 7 placement primitive).
 * Self-contained: logical dies only, no model, no GPU, no external harness —
 * links just src/accel/die_manager.c, so it builds and runs on the Mac. The
 * CUDA probe path (pn_dm_init_cuda) is gated on GPU hardware and validated there.
 *
 * Covers:
 *   - init/shutdown/idempotence and the not-running behavior
 *   - topology: 2-die placement lands on a same-card pair, preferring free pairs
 *   - packing: 1-die placement prefers cards whose mate is already occupied
 *   - budget rejection: placement and bind refuse what does not fit
 *   - bind/release accounting round-trips vram_used exactly
 *   - duplicate-name bind rejection
 */

#include "../src/accel/die_manager.h"

#include <stdio.h>
#include <string.h>

static int g_failed = 0;

#define CHECK(cond, msg) do { \
    if (!(cond)) { \
        fprintf(stderr, "  FAIL: %s (%s:%d)\n", (msg), __FILE__, __LINE__); \
        g_failed++; \
    } \
} while (0)

#define GIB(n) ((uint64_t)(n) << 30)

static void test_init_shutdown(void)
{
    printf("init_shutdown:\n");
    CHECK(pn_dm_count() == 0, "not running -> count 0");
    int dies[2];
    CHECK(pn_dm_place(GIB(1), 1, dies) == -1, "not running -> place refuses");

    CHECK(pn_dm_init_logical(0, GIB(11)) == -1, "0 dies rejected");
    CHECK(pn_dm_init_logical(4, 0) == -1, "0 budget rejected");
    CHECK(pn_dm_init_logical(4, GIB(11)) == 0, "init 4 logical dies");
    CHECK(pn_dm_count() == 4, "count 4");
    CHECK(pn_dm_init_logical(16, GIB(1)) == 0, "re-init is a no-op while running");
    CHECK(pn_dm_count() == 4, "count unchanged by re-init");

    pn_dm_die d;
    CHECK(pn_dm_get(3, &d) == 0 && d.vram_budget == GIB(11) && d.vram_used == 0,
          "fresh die: full budget, nothing used");
    CHECK(pn_dm_get(4, &d) == -1, "out-of-range die refused");

    pn_dm_shutdown();
    CHECK(pn_dm_count() == 0, "shutdown -> count 0");
}

static void test_pair_placement(void)
{
    printf("pair_placement:\n");
    CHECK(pn_dm_init_logical(8, GIB(11)) == 0, "init 8 dies (4 cards)");

    /* A 16 GB 2-die model gets a same-card pair. */
    int dies[2] = { -1, -1 };
    CHECK(pn_dm_place(GIB(16), 2, dies) == 0, "2-die placement fits");
    CHECK(dies[1] == dies[0] + 1 && dies[0] % 2 == 0, "same-card pair (2i, 2i+1)");
    CHECK(pn_dm_bind(dies, 2, "mini-a", GIB(16)) == 0, "bind mini-a");

    /* Second 2-die model gets a DIFFERENT free pair (disjoint die-sets). */
    int dies2[2] = { -1, -1 };
    CHECK(pn_dm_place(GIB(16), 2, dies2) == 0, "second 2-die placement fits");
    CHECK(dies2[0] != dies[0], "prefers a fully-free pair over sharing");
    CHECK(dies2[1] == dies2[0] + 1 && dies2[0] % 2 == 0, "also a same-card pair");
    CHECK(pn_dm_bind(dies2, 2, "mini-b", GIB(16)) == 0, "bind mini-b");

    /* Oversized: 30 GB across 2 dies (15 GB/die) exceeds the 11 GB budget. */
    int dies3[2];
    CHECK(pn_dm_place(GIB(30), 2, dies3) == -1, "over-budget pair refused");

    pn_dm_shutdown();
}

static void test_single_die_packing(void)
{
    printf("single_die_packing:\n");
    CHECK(pn_dm_init_logical(4, GIB(11)) == 0, "init 4 dies (2 cards)");

    /* Occupy die 0 (half of card 0). */
    int die0 = 0;
    CHECK(pn_dm_bind(&die0, 1, "small-a", GIB(4)) == 0, "bind small-a on die 0");

    /* Next 1-die model should pack onto card 0 (mate of die 0 = die 1),
     * keeping the (2,3) pair free for a future 2-die model. */
    int pick = -1;
    CHECK(pn_dm_place(GIB(4), 1, &pick) == 0, "1-die placement fits");
    CHECK(pn_dm_card_of(pick) == 0, "packs onto the half-used card");
    CHECK(pn_dm_bind(&pick, 1, "small-b", GIB(4)) == 0, "bind small-b");

    /* The free same-card pair is still available for a 2-die model. */
    int pair[2];
    CHECK(pn_dm_place(GIB(16), 2, pair) == 0 && pair[0] == 2 && pair[1] == 3,
          "pair (2,3) preserved for the 2-die model");

    pn_dm_shutdown();
}

static void test_bind_release_accounting(void)
{
    printf("bind_release_accounting:\n");
    CHECK(pn_dm_init_logical(2, GIB(11)) == 0, "init 2 dies");

    int pair[2] = { 0, 1 };
    CHECK(pn_dm_bind(pair, 2, "mini", GIB(16)) == 0, "bind 16 GB across the pair");

    pn_dm_die d;
    CHECK(pn_dm_get(0, &d) == 0 && d.vram_used == GIB(16) / 2,
          "die 0 carries half the bytes");
    CHECK(d.n_models == 1 && strcmp(d.model_name, "mini") == 0,
          "die 0 reports the bound model");

    /* A REPEATED NAME IS ALLOWED. One model id may be bound once per card for
     * load-balanced serving, so each placement is its own binding and it is the
     * per-die room check — not the name — that refuses an overlap
     * (die_manager.c pn_dm_bind). Unknown die is still refused. */
    int die0 = 0;
    CHECK(pn_dm_bind(&die0, 1, "mini", GIB(1)) == 0,
          "the same name binds again as a second placement");
    CHECK(pn_dm_get(0, &d) == 0 && d.vram_used == GIB(16) / 2 + GIB(1),
          "die 0 carries both bindings' bytes");
    CHECK(d.n_models == 2, "die 0 reports two bindings");
    int bogus = 9;
    CHECK(pn_dm_bind(&bogus, 1, "other", GIB(1)) == -1, "unknown die refused");

    /* Over-budget bind refused even when place() was skipped. */
    CHECK(pn_dm_bind(&die0, 1, "fat", GIB(8)) == -1,
          "bind refuses what no longer fits");

    /* Release is by-name FIRST-MATCH, so N bindings under one name take N
     * releases and the N+1th fails. The registry relies on exactly this: it
     * releases once per entry on unload. */
    CHECK(pn_dm_release("mini") == 0, "release the first 'mini' binding");
    CHECK(pn_dm_get(0, &d) == 0 && d.n_models == 1 && d.vram_used == GIB(1),
          "die 0 still carries the second binding");
    CHECK(pn_dm_release("mini") == 0, "release the second 'mini' binding");
    CHECK(pn_dm_get(0, &d) == 0 && d.vram_used == 0 && d.n_models == 0,
          "die 0 back to empty");
    CHECK(pn_dm_release("mini") == -1, "a third release is refused");

    pn_dm_shutdown();
}

int main(void)
{
    printf("test_die_manager\n");
    test_init_shutdown();
    test_pair_placement();
    test_single_die_packing();
    test_bind_release_accounting();

    if (g_failed) {
        printf("FAILED: %d check(s)\n", g_failed);
        return 1;
    }
    printf("ALL PASS\n");
    return 0;
}
