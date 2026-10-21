/* SPDX-License-Identifier: AGPL-3.0-or-later */
/* Copyright (C) 2026 David (Corey) Herrera */
/**
 * @file repetition_loop_detector.c
 * @brief Implementation of the decode-time repetition-cycle detector
 *        (see repetition_loop_detector.h for the thresholds and the reasoning).
 *
 * Two stages, so the common case (text that is not looping) costs almost
 * nothing and the confirmed case costs one compare per token:
 *
 *   PROPOSE.  A rolling polynomial hash over the last RLD_SHINGLE_TOKENS token
 *   ids is kept per position.  When the current window's hash equals the hash
 *   stored at position i-L, the stream MIGHT have period L.  The scan runs L
 *   from 1 upward and takes the FIRST hit, so the distance proposed is the
 *   fundamental period and never a multiple of it — this is the single most
 *   important false-positive defence, because every legitimate repeating
 *   structure (a zero-filled table row, a separator line, a repeated log line)
 *   has a small fundamental period and is rejected on the spot by min_period.
 *
 *   CONFIRM.  A hash hit is verified against the real token ids (a hash
 *   collision must not be able to end a turn), then the detector LOCKS: from
 *   there each new token only has to equal the token `period` back.  One
 *   mismatch — one novel token anywhere in the cycle — unlocks immediately and
 *   the run count is lost, so a "cycle" broken by real content never
 *   accumulates.  Firing needs min_repeats back-to-back copies.
 *
 * @status Complete.
 * @issues None
 * @todo None
 */

#include "repetition_loop_detector.h"

#include <stdlib.h>
#include <string.h>

/* Odd multiplier for the rolling window hash (the FNV-1a 64-bit prime).  Any
 * odd base gives a full-period polynomial hash mod 2^64; this one is already
 * the project's habitual choice. */
#define RLD_HASH_BASE 0x100000001b3ULL

/* Defaults.  See the header for why these numbers and not smaller ones. */
#define RLD_DEFAULT_MIN_PERIOD   48
#define RLD_DEFAULT_MAX_PERIOD   512
#define RLD_DEFAULT_MIN_REPEATS  12

/* Hard ceiling on max_period, so an operator typo cannot ask for a gigabyte of
 * ring.  4096 tokens is ~16 KB of text in one cycle — far past anything a
 * degeneration has ever produced here. */
#define RLD_MAX_PERIOD_CEILING   4096

/* Read one non-negative integer env var, falling back to `fallback` when unset,
 * empty, or unparseable.  Deliberately strict: a garbage value must not
 * silently become 0 and disable a guard the operator thought they were tuning. */
static int rld_env_int(const char *name, int fallback)
{
    const char *raw = getenv(name);
    if (!raw || !raw[0])
        return fallback;
    char *end = NULL;
    long v = strtol(raw, &end, 10);
    if (end == raw || (end && *end) || v <= 0 || v > 1000000L)
        return fallback;
    return (int)v;
}

bool rld_enabled(void)
{
    static int cached = -1;
    if (cached < 0) {
        const char *env = getenv("PANTHEON_REPETITION_STOP");
        cached = (env && env[0] == '0' && !env[1]) ? 0 : 1;
    }
    return cached != 0;
}

void rld_default_config(rld_config *cfg)
{
    static rld_config cached;
    static int resolved = 0;

    if (!cfg)
        return;
    if (!resolved) {
        rld_config c;
        c.min_period  = rld_env_int("PANTHEON_REPETITION_MIN_PERIOD",
                                    RLD_DEFAULT_MIN_PERIOD);
        c.max_period  = rld_env_int("PANTHEON_REPETITION_MAX_PERIOD",
                                    RLD_DEFAULT_MAX_PERIOD);
        c.min_repeats = rld_env_int("PANTHEON_REPETITION_MIN_REPEATS",
                                    RLD_DEFAULT_MIN_REPEATS);
        if (c.min_repeats < 2) c.min_repeats = 2;
        if (c.max_period > RLD_MAX_PERIOD_CEILING) c.max_period = RLD_MAX_PERIOD_CEILING;
        if (c.min_period > c.max_period) c.min_period = c.max_period;
        cached = c;
        resolved = 1;
    }
    *cfg = cached;
}

/* Smallest power of two >= n (n >= 1). */
static size_t rld_next_pow2(size_t n)
{
    size_t p = 1;
    while (p < n)
        p <<= 1;
    return p;
}

bool rld_init(rld_detector *rd, const rld_config *cfg)
{
    if (!rd)
        return false;
    memset(rd, 0, sizeof(*rd));

    if (cfg) {
        rd->cfg = *cfg;
        if (rd->cfg.min_repeats < 2) rd->cfg.min_repeats = 2;
        if (rd->cfg.max_period > RLD_MAX_PERIOD_CEILING)
            rd->cfg.max_period = RLD_MAX_PERIOD_CEILING;
        if (rd->cfg.min_period < 1) rd->cfg.min_period = 1;
        if (rd->cfg.min_period > rd->cfg.max_period)
            rd->cfg.min_period = rd->cfg.max_period;
    } else {
        rld_default_config(&rd->cfg);
    }
    if (rd->cfg.max_period < 1)
        return false;

    /* The verify step reaches back max_period + RLD_SHINGLE_TOKENS - 1 tokens,
     * so the ring must hold one more than that for the oldest entry to survive
     * the write of the newest. */
    size_t need = (size_t)rd->cfg.max_period + RLD_SHINGLE_TOKENS + 1;
    size_t cap  = rld_next_pow2(need);

    rd->token_ring   = calloc(cap, sizeof(*rd->token_ring));
    rd->shingle_ring = calloc(cap, sizeof(*rd->shingle_ring));
    if (!rd->token_ring || !rd->shingle_ring) {
        rld_free(rd);        /* inert, not fatal: the turn runs unguarded */
        return false;
    }
    rd->ring_mask = cap - 1;

    rd->shingle_power = 1;
    for (int i = 0; i < RLD_SHINGLE_TOKENS - 1; i++)
        rd->shingle_power *= RLD_HASH_BASE;

    return true;
}

void rld_free(rld_detector *rd)
{
    if (!rd)
        return;
    free(rd->token_ring);
    free(rd->shingle_ring);
    rd->token_ring = NULL;
    rd->shingle_ring = NULL;
    rd->ring_mask = 0;
}

/* Confirm a hash-proposed period against the real token ids: the last
 * RLD_SHINGLE_TOKENS tokens must each equal the token `period` earlier.  A
 * 2^-64 hash collision must never be able to end a live turn, so the proposal
 * is never trusted on its own.  Runs once per lock, not per token. */
static bool rld_verify_period(const rld_detector *rd, long last, int period)
{
    size_t mask = rd->ring_mask;
    for (int j = 0; j < RLD_SHINGLE_TOKENS; j++) {
        long here = last - j;
        long back = here - period;
        if (back < 0)
            return false;
        if (rd->token_ring[(size_t)here & mask] !=
            rd->token_ring[(size_t)back & mask])
            return false;
    }
    return true;
}

/* PROPOSE: the smallest L in [1, max_period] at which the current window
 * recurs, or 0 when it does not recur at all.  Smallest-first is deliberate —
 * see the file header. */
static int rld_propose_period(const rld_detector *rd, long last)
{
    size_t mask = rd->ring_mask;
    long oldest_valid = RLD_SHINGLE_TOKENS - 1;   /* first full window */

    for (int period = 1; period <= rd->cfg.max_period; period++) {
        long prev = last - period;
        if (prev < oldest_valid)
            break;
        if (rd->shingle_ring[(size_t)prev & mask] == rd->rolling_hash)
            return period;
    }
    return 0;
}

bool rld_feed(rld_detector *rd, uint32_t token)
{
    if (!rd || !rd->token_ring || rd->tripped)
        return false;

    long   last = rd->n_fed;
    size_t mask = rd->ring_mask;

    /* Roll the window hash forward over the token entering at `last`, dropping
     * the one leaving RLD_SHINGLE_TOKENS back.  +1 on every id so that token 0
     * still contributes (it is a real token on some tokenizers). */
    if (last >= RLD_SHINGLE_TOKENS) {
        uint32_t leaving =
            rd->token_ring[(size_t)(last - RLD_SHINGLE_TOKENS) & mask];
        rd->rolling_hash -= rd->shingle_power * ((uint64_t)leaving + 1u);
    }
    rd->rolling_hash = rd->rolling_hash * RLD_HASH_BASE + ((uint64_t)token + 1u);

    rd->token_ring[(size_t)last & mask]   = token;
    rd->shingle_ring[(size_t)last & mask] = rd->rolling_hash;
    rd->n_fed = last + 1;

    if (last < RLD_SHINGLE_TOKENS - 1)
        return false;   /* the first window is not complete yet */

    /* CONFIRM: locked onto a period — one compare per token.  A single novel
     * token breaks the cycle and drops the whole run, which is what keeps
     * "repeated structure with varying content" (a JSON array of distinct
     * objects, a table with distinct rows) from ever accumulating. */
    if (rd->locked_period > 0) {
        long back = last - rd->locked_period;
        if (back < 0 || rd->token_ring[(size_t)back & mask] != token) {
            rd->locked_period = 0;
            rd->verified_run  = 0;
            return false;
        }
        rd->verified_run++;
        /* `verified_run` counts positions matching one period back, so N
         * back-to-back copies of the block show up as (N-1) periods of run. */
        long needed = (long)rd->locked_period * (rd->cfg.min_repeats - 1);
        if (rd->verified_run >= needed) {
            rd->tripped      = true;
            rd->cycle_tokens = rd->locked_period;
            rd->repeats      = (int)(rd->verified_run / rd->locked_period) + 1;
            return true;
        }
        return false;
    }

    int period = rld_propose_period(rd, last);
    if (period == 0)
        return false;
    /* The FUNDAMENTAL period is below the bar: this is structured repetition
     * (a table row, a padding line, a repeated log line), not a collapsed
     * paragraph.  Reject and do not consider any multiple of it. */
    if (period < rd->cfg.min_period)
        return false;
    if (!rld_verify_period(rd, last, period))
        return false;

    rd->locked_period = period;
    rd->verified_run  = RLD_SHINGLE_TOKENS;   /* the window we just verified */
    return false;
}
