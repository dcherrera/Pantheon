/* SPDX-License-Identifier: AGPL-3.0-or-later */
/* Copyright (C) 2026 David (Corey) Herrera */
/**
 * @file pn_tt_superband.h
 * @brief The prefill overlap's super-band jobs over a pass: ONE rule for the plan
 *        (qwen35moe_prefill_ov_plan.c) and the replica map (pn_tt_replica_map.c), whose keys must
 *        name exactly the row ranges the jobs' producers write.
 *
 * `bands` split into jobs of `span` bands, the last one shorter — except that a last job of
 * fewer than 2 bands is folded into the one before it (a 1-band job is a decode-shaped
 * projection on the grid; span 21 over 64 bands deadlocked on it, q38-gdn-chain G.3).
 *
 * Pure, header-only.
 *
 * @status G.3 of the prefill chain work.
 * @issues None
 * @todo   None
 */
#ifndef PN_TT_SUPERBAND_H
#define PN_TT_SUPERBAND_H

#include <stdint.h>

/** Jobs over `bands` at `span` (0 when either is 0). */
static inline uint32_t pn_tt_superband_jobs(uint32_t bands, uint32_t span)
{
    uint32_t jobs;

    if (bands == 0u || span == 0u)
        return 0u;
    jobs = (bands + span - 1u) / span;
#ifndef PN_TT_SUPERBAND_RED
    if (jobs > 1u && bands - (jobs - 1u) * span < 2u)
        jobs--;
#endif
    return jobs;
}

/** Job `job`'s bands: [*first, *first + *count); count 0 past the last job. */
static inline void pn_tt_superband_job(uint32_t bands, uint32_t span, uint32_t job,
                                       uint32_t *first, uint32_t *count)
{
    const uint32_t jobs = pn_tt_superband_jobs(bands, span);

    if (job >= jobs) {
        *first = bands;
        *count = 0u;
        return;
    }
    *first = job * span;
    *count = job + 1u == jobs ? bands - *first : span;
}

#endif /* PN_TT_SUPERBAND_H */
