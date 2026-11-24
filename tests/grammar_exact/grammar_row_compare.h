/* SPDX-License-Identifier: AGPL-3.0-or-later */
/* Copyright (C) 2026 David (Corey) Herrera */
/**
 * @file grammar_row_compare.h
 * @brief Draw a grammar-MASKED row two ways through the real sampler and compare the tokens:
 *        the plain path (the dense row, the full-vocabulary mask) and the Tenstorrent top-N
 *        path (the row's top-64, decided by the shortlist / outside bound, else chunks or a
 *        refill at row 0, a deferral at a verify row past 0).
 *
 * The grammar is a bitmap: token i's text is "@@<i>", and "@" / "@@" are live prefixes, so
 * the sampler's prefix filter and verdicts see exactly verdict[i]. The top-N row carries the
 * TT kernel's top-64 (value desc, id asc); the dense row, the chunks and the device's tail
 * counts come from the dense row, as the card would serve them.
 *
 * Shared by tests/test_grammar_exact_bound.c (synthetic rows) and
 * tools/grammar_exact/grammar_row_census.c (real masked rows dumped on the Mac).
 *
 * @status Test support (CPU only).
 * @issues None
 * @todo None
 */
#ifndef PN_GRAMMAR_ROW_COMPARE_H
#define PN_GRAMMAR_ROW_COMPARE_H

#include <stdint.h>

#include "../../include/pantheon.h"

#define GRC_TOPN 64

/** How the top-N side decides a row. */
typedef enum {
    GRC_ROW0 = 0,       /**< plain decode / verify row 0: sample_token with every callback */
    GRC_ROW_LATER,      /**< a verify row past 0: the shortlist only (no counts), else defer */
} grc_route;

typedef struct {
    int n_vocab;
    const float *dense;             /**< the raw row */
    const unsigned char *verdict;   /**< 1 = the grammar allows the token */
    pn_sampling_params params;      /**< temperature, top_k, top_p, penalties (no callbacks) */
    const uint32_t *recent;
    int n_recent;
    int topn;                       /**< the top-N row's N (0 = GRC_TOPN; a gate knob below it) */
} grc_row;

typedef struct {
    unsigned long draws;        /**< seeds compared (decided draws only for GRC_ROW_LATER) */
    unsigned long mismatches;   /**< top-N token != plain token */
    unsigned long deferred;     /**< GRC_ROW_LATER: undecided (the verify would defer) */
    unsigned long bound;        /**< draws the outside bound decided */
    unsigned long shortlist;    /**< draws any shortlist rule decided */
    unsigned long refills;      /**< whole dense rows fetched */
    unsigned long chunk_steps;  /**< draws decided by a chunk fetch */
} grc_result;

/** Build the bitmap grammar's token texts for a vocabulary (once; 0 on success). */
int grc_init(int n_vocab);

/**
 * Draw `n_seeds` seeded tokens each way and accumulate into *out.
 * @param grammar_off_subject  selftest: the top-N side drops the grammar (must mismatch)
 */
void grc_compare(const grc_row *row, grc_route route, int n_seeds, uint64_t seed_base,
                 int grammar_off_subject, grc_result *out);

#endif /* PN_GRAMMAR_ROW_COMPARE_H */
