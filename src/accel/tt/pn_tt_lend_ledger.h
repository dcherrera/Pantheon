/* SPDX-License-Identifier: AGPL-3.0-or-later */
/* Copyright (C) 2026 David (Corey) Herrera */
/**
 * @file pn_tt_lend_ledger.h
 * @brief L1a prefill role table, the bookkeeping half:
 *        which decode residencies lent their cores to prefill attention for one prompt, and which
 *        of those cores may be booted again when the prompt ends.
 *
 * Pure (no card): the device glue (pn_tt_prefill_lend.h) records into it and asks it. The rules it
 * owns: a prompt lends only when it is long enough and deep attention is on; every lent core is
 * restored once; a core whose borrowed attention member did not exit is NEVER booted again
 * (booting firmware under a running triad is the wedge).
 *
 * @status L1a.5: default on for dense-FFN qwen35 (Qwen3.8) prompts of 16,384+ batched rows.
 * @issues None
 * @todo None
 */
#ifndef PN_TT_LEND_LEDGER_H
#define PN_TT_LEND_LEDGER_H

#include <stdbool.h>
#include <stdint.h>

/** The most residencies one prompt lends (the decode GDN family's maximum). */
#define PN_TT_LEND_MAX 32u

typedef struct pn_tt_lend_ledger {
    uint32_t count;
    uint32_t member[PN_TT_LEND_MAX]; /**< the lending family's member index */
    uint32_t core[PN_TT_LEND_MAX];
    bool stuck[PN_TT_LEND_MAX];      /**< its borrowed member did not exit: never boot it */
} pn_tt_lend_ledger;

/** Does a prompt of `rows` batched rows lend? Only with deep attention (the lent members serve
 *  its queue) and at least `min_rows` rows (the swap costs ~0.2 s; spec §4). */
bool pn_tt_lend_wanted(uint32_t rows, uint32_t min_rows, bool deep);

/** The default row floor for a dense-FFN qwen35 model (Qwen3.8). p150, exact: 32,768 rows 134.2 ->
 *  113.4 s, 7,680 rows 25.06 -> 25.49 s; break-even ~12k (spec §4d). MoE models stay off. */
#define PN_TT_LEND_DENSE_MIN_ROWS 16384u

/** The row floor: `env` (`PN_TT_PREFILL_LEND`) when set and a number, else
 *  PN_TT_LEND_DENSE_MIN_ROWS for a dense-FFN model and 0 (off) otherwise. */
uint32_t pn_tt_lend_min_rows(const char *env, bool dense);

/** Is deep-queued attention on? `env` (`PN_TT_ATTN_DEEP`) when set; unset, exactly when this
 *  prompt has lent cores (without them deep only loses the FFN overlap). Ask with `lent` true for
 *  "may this prompt lend". */
bool pn_tt_lend_deep_on(const char *env, bool lent);

/** Empty the ledger. */
void pn_tt_lend_ledger_reset(pn_tt_lend_ledger *ledger);

/** Record that `member` lent `core`. @return false when full or the core is already lent */
bool pn_tt_lend_ledger_add(pn_tt_lend_ledger *ledger, uint32_t member, uint32_t core);

/** Take the last entry back out (it went to another role). @return false when empty */
bool pn_tt_lend_ledger_pop(pn_tt_lend_ledger *ledger, uint32_t *member, uint32_t *core);

/** The borrowed member on `core` did not exit: its core must not be booted again. */
void pn_tt_lend_ledger_mark_stuck(pn_tt_lend_ledger *ledger, uint32_t core);

/**
 * The residencies to restore, in lending order, skipping stuck cores; empties the ledger so a
 * second call restores nothing.
 * @return how many entries were written to `members` / `cores` (each PN_TT_LEND_MAX long)
 */
uint32_t pn_tt_lend_ledger_take_restorable(pn_tt_lend_ledger *ledger, uint32_t *members,
                                           uint32_t *cores);

#endif /* PN_TT_LEND_LEDGER_H */
