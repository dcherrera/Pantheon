/* SPDX-License-Identifier: AGPL-3.0-or-later */
/* Copyright (C) 2026 David (Corey) Herrera */
/*
 * test_local_unit_place.c — Pantheon Local B4/B5: profile request → free units.
 * Fits; the same-card rule; one-GPU requests prefer an idle card (two busy GPUs
 * on one K80 board throttle); refusals carry the reason the app shows.
 */

#include "local_test_support.h"

#include "../src/local/unit_place.h"

#define GIB(n) ((uint64_t)(n) << 30)

/* A fake K80 box: n_cards boards, busy[] marks units with a model. */
static void make_k80(pn_hw_report *hw, int n_cards, const int *busy, int n_busy)
{
    memset(hw, 0, sizeof(*hw));
    hw->family = PN_HW_FAMILY_K80;
    for (int c = 0; c < n_cards; c++) {
        hw->cards[c].id = c;
        snprintf(hw->cards[c].type, sizeof(hw->cards[c].type), "k80");
        hw->cards[c].first_unit = 2 * c;
        hw->cards[c].n_units = 2;
        for (int k = 0; k < 2; k++) {
            pn_hw_unit *u = &hw->units[2 * c + k];
            u->id = u->die = 2 * c + k;
            u->card = c;
            snprintf(u->type, sizeof(u->type), "k80-gpu");
            u->mem_total = u->mem_free = GIB(11);
        }
    }
    hw->n_cards = n_cards;
    hw->n_units = 2 * n_cards;
    for (int i = 0; i < n_busy; i++) {
        hw->units[busy[i]].n_models = 1;
        hw->units[busy[i]].mem_free = GIB(2);
    }
}

static pn_unit_request req(const char *type, int count, bool same_card)
{
    pn_unit_request r;
    memset(&r, 0, sizeof(r));
    snprintf(r.unit_type, sizeof(r.unit_type), "%s", type);
    r.count = count;
    r.same_card = same_card;
    return r;
}

static void test_pair_same_card(void)
{
    pn_hw_report hw;
    int ids[8], n = 0, busy[] = { 0, 3 };
    char why[128];

    printf("pair_same_card:\n");
    make_k80(&hw, 3, busy, 2);   /* card0 half busy, card1 half busy, card2 idle */
    pn_unit_request r = req("k80-gpu", 2, true);
    CHECK(pn_unit_place(&hw, &r, ids, &n, why, sizeof(why)) == 0, "fits");
    CHECK(n == 2 && ids[0] == 4 && ids[1] == 5, "lands on card 2 (units 4,5)");

    int busy2[] = { 0, 3, 4 };
    make_k80(&hw, 3, busy2, 3);  /* no card has 2 free */
    CHECK(pn_unit_place(&hw, &r, ids, &n, why, sizeof(why)) == -1, "refused");
    CHECK(strcmp(why, "needs 2 free k80-gpu on one card, you have 1") == 0, why);
}

static void test_single_prefers_idle_card(void)
{
    pn_hw_report hw;
    int ids[8], n = 0, busy[] = { 0 };
    char why[128];

    printf("single_prefers_idle_card:\n");
    make_k80(&hw, 2, busy, 1);   /* unit 1's mate is busy; card 1 idle */
    pn_unit_request r = req("k80-gpu", 1, false);
    CHECK(pn_unit_place(&hw, &r, ids, &n, why, sizeof(why)) == 0 && n == 1, "fits");
    CHECK(ids[0] == 2, "picks idle card 1, not unit 1 next to a busy GPU");

    int busy2[] = { 0, 2 };
    make_k80(&hw, 2, busy2, 2);  /* every card half busy */
    CHECK(pn_unit_place(&hw, &r, ids, &n, why, sizeof(why)) == 0 && n == 1 && ids[0] == 1,
          "falls back to a half-busy card");
}

static void test_short_and_memory(void)
{
    pn_hw_report hw;
    int ids[8], n = 0, busy[] = { 0, 1, 2 };
    char why[128];

    printf("short_and_memory:\n");
    make_k80(&hw, 2, busy, 3);
    pn_unit_request r = req("k80-gpu", 2, false);
    CHECK(pn_unit_place(&hw, &r, ids, &n, why, sizeof(why)) == -1, "1 free < 2 refused");
    CHECK(strcmp(why, "needs 2 free k80-gpu, you have 1") == 0, why);
    r = req("p150", 1, false);
    CHECK(pn_unit_place(&hw, &r, ids, &n, why, sizeof(why)) == -1, "no p150 on a K80 box");
    make_k80(&hw, 1, NULL, 0);
    r = req("k80-gpu", 1, false);
    r.min_mem_per_unit = GIB(12);
    CHECK(pn_unit_place(&hw, &r, ids, &n, why, sizeof(why)) == -1, "12 GiB minimum refused");
    CHECK(pn_unit_count_free(&hw, "k80-gpu", GIB(11)) == 2, "11 GiB minimum: 2 fit");
}

static void test_cpu_and_p150(void)
{
    pn_hw_report hw;
    int ids[8], n = 7;
    char why[128];

    printf("cpu_and_p150:\n");
    memset(&hw, 0, sizeof(hw));
    hw.n_cards = hw.n_units = 1;
    hw.cards[0].n_units = 1;
    snprintf(hw.units[0].type, sizeof(hw.units[0].type), "cpu");
    pn_unit_request r = req("cpu", 1, false);
    CHECK(pn_unit_place(&hw, &r, ids, &n, why, sizeof(why)) == 0 && n == 0, "cpu: no dies");
    snprintf(hw.units[0].type, sizeof(hw.units[0].type), "p150");
    hw.units[0].mem_free = GIB(31);
    r = req("p150", 1, false);
    CHECK(pn_unit_place(&hw, &r, ids, &n, why, sizeof(why)) == 0 && n == 1 && ids[0] == 0,
          "p150 = unit 0");
    hw.units[0].n_models = 1;
    CHECK(pn_unit_place(&hw, &r, ids, &n, why, sizeof(why)) == -1, "busy p150 refused");
}

/* Q8_0 on two K80 boards: 4 units, 2 per card = two whole boards, idle boards first,
 * in card order (the split's head lands on the last board's second GPU). Without the
 * rule a 4-unit request spreads one unit per idle board. */
static void test_two_whole_boards(void)
{
    pn_hw_report hw;
    int ids[8], n = 0, busy[] = { 6 };
    char why[160];

    printf("two_whole_boards:\n");
    make_k80(&hw, 4, busy, 1);   /* cards 0-2 idle, card 3 half busy */
    pn_unit_request r = req("k80-gpu", 4, false);
    r.per_card = 2;
    CHECK(pn_unit_place(&hw, &r, ids, &n, why, sizeof(why)) == 0, "fits");
    CHECK(n == 4 && ids[0] == 0 && ids[1] == 1 && ids[2] == 2 && ids[3] == 3,
          "two idle boards, card order (units 0,1,2,3)");

    int busy2[] = { 0, 2 };
    make_k80(&hw, 4, busy2, 2);  /* cards 0,1 half busy: only cards 2,3 whole */
    CHECK(pn_unit_place(&hw, &r, ids, &n, why, sizeof(why)) == 0 && n == 4 && ids[0] == 4 &&
              ids[3] == 7, "skips the half-busy boards (units 4-7)");

    int busy3[] = { 0, 2, 4 };
    make_k80(&hw, 4, busy3, 3);  /* 5 free units, only card 3 has 2 free */
    CHECK(pn_unit_place(&hw, &r, ids, &n, why, sizeof(why)) == -1 && n == 0, "refused");
    CHECK(strstr(why, "2 on each of 2 cards; 1 card(s) have 2 free") != NULL, why);

    make_k80(&hw, 3, NULL, 0);   /* six free units on three idle boards */
    r.per_card = 0;
    CHECK(pn_unit_place(&hw, &r, ids, &n, why, sizeof(why)) == 0 && n == 4 &&
              ids[0] == 0 && ids[1] == 2 && ids[2] == 4,
          "no rule: one per idle board first (why the profile needs per_card)");
}

int main(void)
{
    test_pair_same_card();
    test_two_whole_boards();
    test_single_prefers_idle_card();
    test_short_and_memory();
    test_cpu_and_p150();
    return lt_report("test_local_unit_place");
}
