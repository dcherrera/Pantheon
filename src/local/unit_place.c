/* SPDX-License-Identifier: AGPL-3.0-or-later */
/* Copyright (C) 2026 David (Corey) Herrera */
/**
 * @file   unit_place.c
 * @brief  Implementation of unit placement (see unit_place.h).
 *
 * @status None
 * @issues None
 * @todo   None
 */

#include "unit_place.h"

#include <stdio.h>
#include <string.h>

static bool unit_is_free(const pn_hw_unit *u, const char *type, uint64_t min_mem)
{
    return strcmp(u->type, type) == 0 && pn_hw_unit_available(u) && u->mem_free >= min_mem;
}

int pn_unit_count_free(const pn_hw_report *hw, const char *type, uint64_t min_mem)
{
    int n = 0;

    for (int i = 0; i < hw->n_units; i++)
        n += unit_is_free(&hw->units[i], type, min_mem);
    return n;
}

/* Free units of `type` on one card. */
static int card_free_units(const pn_hw_report *hw, const pn_hw_card *c, const char *type,
                           uint64_t min_mem, int *ids, int max)
{
    int n = 0;

    for (int i = 0; i < c->n_units && n < max; i++) {
        const pn_hw_unit *u = &hw->units[c->first_unit + i];
        if (unit_is_free(u, type, min_mem))
            ids[n++] = u->id;
    }
    return n;
}

/* Most free units of `type` that any single card has. */
static int best_card_free(const pn_hw_report *hw, const char *type, uint64_t min_mem)
{
    int best = 0, ids[PN_HW_MAX_UNITS];

    for (int c = 0; c < hw->n_cards; c++) {
        int n = card_free_units(hw, &hw->cards[c], type, min_mem, ids, PN_HW_MAX_UNITS);
        if (n > best)
            best = n;
    }
    return best;
}

/* Fill units_out from the cards whose units are ALL free first (a lone unit on
 * an idle card runs at full clock), then from any card. */
static int place_spread(const pn_hw_report *hw, const pn_unit_request *req, int *out)
{
    int n = 0, ids[PN_HW_MAX_UNITS];

    for (int pass = 0; pass < 2 && n < req->count; pass++) {
        for (int c = 0; c < hw->n_cards && n < req->count; c++) {
            const pn_hw_card *card = &hw->cards[c];
            int k = card_free_units(hw, card, req->unit_type, req->min_mem_per_unit, ids,
                                    PN_HW_MAX_UNITS);
            if (k == 0 || (pass == 0 && k < card->n_units))
                continue;
            for (int i = 0; i < k && n < req->count; i++) {
                bool taken = false;
                for (int j = 0; j < n; j++)
                    taken |= out[j] == ids[i];
                if (!taken)
                    out[n++] = ids[i];
                if (pass == 0)
                    break;   /* one per idle card on the first pass */
            }
        }
    }
    return n;
}

/* `per_card` groups: cards with that many free units, fully idle cards first (a
 * board whose other GPU is busy throttles both), lowest card id first within a
 * pass; units in card order, so a layer split walks board by board and ends on the
 * last card's second GPU. Returns the cards found (may be short). */
static int place_per_card(const pn_hw_report *hw, const pn_unit_request *req, int *out,
                          int *n_out)
{
    int cards = 0, want = req->count / req->per_card, ids[PN_HW_MAX_UNITS];
    bool used[PN_HW_MAX_CARDS] = { false };

    *n_out = 0;
    for (int pass = 0; pass < 2 && cards < want; pass++) {
        for (int c = 0; c < hw->n_cards && cards < want; c++) {
            const pn_hw_card *card = &hw->cards[c];
            int k = card_free_units(hw, card, req->unit_type, req->min_mem_per_unit, ids,
                                    PN_HW_MAX_UNITS);
            if (used[c] || k < req->per_card || (pass == 0 && k < card->n_units))
                continue;
            used[c] = true;
            cards++;
        }
    }
    for (int c = 0; c < hw->n_cards; c++) {
        if (!used[c])
            continue;
        card_free_units(hw, &hw->cards[c], req->unit_type, req->min_mem_per_unit, ids,
                        PN_HW_MAX_UNITS);
        for (int i = 0; i < req->per_card; i++)
            out[(*n_out)++] = ids[i];
    }
    return cards;
}

int pn_unit_place(const pn_hw_report *hw, const pn_unit_request *req, int *units_out,
                  int *n_out, char *reason, size_t reason_cap)
{
    const char *t = req->unit_type;
    int have = pn_unit_count_free(hw, t, req->min_mem_per_unit);

    *n_out = 0;
    if (req->count < 1 || req->count > PN_UNIT_PLACE_MAX) {
        snprintf(reason, reason_cap, "bad unit count %d", req->count);
        return -1;
    }
    if (strcmp(t, "cpu") == 0) {
        if (have < 1) {
            snprintf(reason, reason_cap, "needs the cpu unit free, it is in use");
            return -1;
        }
        return 0;   /* CPU-resident: no dies */
    }
    if (req->same_card) {
        int best = best_card_free(hw, t, req->min_mem_per_unit);
        if (best < req->count) {
            snprintf(reason, reason_cap, "needs %d free %s on one card, you have %d",
                     req->count, t, best);
            return -1;
        }
        for (int c = 0; c < hw->n_cards; c++) {
            int n = card_free_units(hw, &hw->cards[c], t, req->min_mem_per_unit, units_out,
                                    req->count);
            if (n == req->count) {
                *n_out = n;
                return 0;
            }
        }
    }
    if (have < req->count) {
        snprintf(reason, reason_cap, "needs %d free %s, you have %d", req->count, t, have);
        return -1;
    }
    if (req->per_card > 0 && !req->same_card) {
        int want = req->count / req->per_card;
        int got = place_per_card(hw, req, units_out, n_out);
        if (got < want) {
            *n_out = 0;
            snprintf(reason, reason_cap, "needs %d free %s, %d on each of %d cards; %d card(s) "
                     "have %d free", req->count, t, req->per_card, want, got, req->per_card);
            return -1;
        }
        return 0;
    }
    *n_out = place_spread(hw, req, units_out);
    return 0;
}
