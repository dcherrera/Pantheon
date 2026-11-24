/* SPDX-License-Identifier: AGPL-3.0-or-later */
/* Copyright (C) 2026 David (Corey) Herrera */
/**
 * @file   batch_sched_prefill_share.h
 * @brief  Decode share while a prefill runs beside decoding sessions: how long
 *         a prefiller stands aside at each chunk yield so the other slots on
 *         the model keep decoding.
 *
 *         Before this a yield let the driver run ONE decode step per prefill
 *         chunk. A chunk is a 1024-token wide sub-range (seconds on a 4-die
 *         K80 copy) and a step is ~50-60 ms, so a long re-prefill (an agent's
 *         context compaction, 12-25k tokens) left its neighbours with one
 *         token per chunk for minutes.
 *
 *         Now the prefiller, after a chunk that held the GPU for `held_ms`,
 *         waits `held_ms * share / (100 - share)` before it re-queues for the
 *         turn, while the driver steps the decoders freely. `share` is the
 *         percentage of GPU time decoding gets while both run
 *         (QWEN35_BS_PREFILL_DECODE_SHARE, default 50; 0 = the old one step per
 *         chunk). The wait ends early when nothing is left to decode.
 *
 *         Scheduling only: chunk boundaries and every forward are unchanged,
 *         so each session's tokens are bit-identical at any share.
 *
 * @status Complete
 * @issues None
 * @todo   None
 */

#ifndef PN_SERVE_BATCH_SCHED_PREFILL_SHARE_H
#define PN_SERVE_BATCH_SCHED_PREFILL_SHARE_H

#include <stdlib.h>

/* Default decode share (percent) and the cap that keeps a prefill moving. */
#define BS_PREFILL_DECODE_SHARE_DEFAULT 50
#define BS_PREFILL_DECODE_SHARE_MAX     90

/* Parse a QWEN35_BS_PREFILL_DECODE_SHARE value: NULL/empty = the default,
 * otherwise clamped to [0, BS_PREFILL_DECODE_SHARE_MAX]. */
static inline int bs_prefill_decode_share_parse(const char *value)
{
    if (!value || !value[0])
        return BS_PREFILL_DECODE_SHARE_DEFAULT;
    int pct = atoi(value);
    if (pct < 0)
        pct = 0;
    if (pct > BS_PREFILL_DECODE_SHARE_MAX)
        pct = BS_PREFILL_DECODE_SHARE_MAX;
    return pct;
}

/* How long (ms) the prefiller stands aside after holding the GPU `held_ms`, so
 * decode gets `share_pct` percent of the time. 0 for share 0 or no hold. */
static inline double bs_prefill_decode_window_ms(double held_ms, int share_pct)
{
    if (share_pct <= 0 || held_ms <= 0.0)
        return 0.0;
    if (share_pct > BS_PREFILL_DECODE_SHARE_MAX)
        share_pct = BS_PREFILL_DECODE_SHARE_MAX;
    return held_ms * (double)share_pct / (double)(100 - share_pct);
}

#endif /* PN_SERVE_BATCH_SCHED_PREFILL_SHARE_H */
