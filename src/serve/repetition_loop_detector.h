/* SPDX-License-Identifier: AGPL-3.0-or-later */
/* Copyright (C) 2026 David (Corey) Herrera */
/**
 * @file repetition_loop_detector.h
 * @brief Decode-time detector for a degenerate repeating CYCLE in the generated
 *        token stream — the "repetition collapse" a heavily quantised model
 *        falls into at long context, where it stops producing new text and
 *        re-emits one paragraph over and over until the token budget runs out.
 *
 * The measured failure (saber-orch, Qwen3.6-35B-A3B at Q3_K_M, ~40k context):
 * a 14,535-byte "answer" whose distinct content was ~470 bytes — one paragraph
 * cycled 31 times — generated to completion and reported as a normal finish.
 *
 * ── What is detected ────────────────────────────────────────────────────────
 * A CYCLE, not a repeated token and not a repeated line: a block of `period`
 * tokens reproduced byte-for-byte, back to back, with ZERO novel tokens in
 * between, `min_repeats` times.  Detection runs on TOKEN IDS, never on decoded
 * text: the ids are what the loop already has in hand (no allocation, no UTF-8
 * boundary handling), and the emitter holds a partial tail back so the decoded
 * text is not even available token-by-token.
 *
 * ── Why the thresholds are where they are ───────────────────────────────────
 * FALSE POSITIVES ARE THE REAL RISK.  This model's job is largely reading and
 * writing code, and legitimate output repeats: zero-filled tables, `return
 * NULL;` a hundred times, markdown separator rows, ASCII art, a log excerpt
 * where one line recurs 200 times.  Truncating a legitimate long code block is
 * a WORSE failure than the collapse it prevents — it corrupts good output
 * silently, where the collapse at least produces obvious garbage.  Two
 * properties keep the false-positive rate down:
 *
 *   1. The detector always locks onto the SMALLEST distance at which the
 *      recent window recurs, i.e. the FUNDAMENTAL period, never a multiple of
 *      it.  So a zero-filled table whose repeating unit is one 7-token row is
 *      proposed as period 7 — and rejected — rather than being read as a
 *      48-token "cycle" three rows long.  Every legitimate repeating structure
 *      we can think of has a repeating unit that is a LINE or a ROW, which is
 *      exactly where this rule puts it.
 *   2. A minimum period (default 48 tokens ≈ 190 chars, several lines) on top
 *      of that, plus a minimum repeat count (default 12).  Together they mean
 *      the detector cannot fire until the model has emitted at least
 *      48 × 12 = 576 tokens of byte-identical, back-to-back repeated content
 *      with no novel token anywhere in it.
 *
 * Deliberately NOT detected: short-period degeneration ("the the the", one
 * sentence looping).  That is precisely the region legitimate output lives in,
 * and the repeat/frequency penalties in the sampler already push against it.
 *
 * ── Cost ────────────────────────────────────────────────────────────────────
 * Bounded per token, no allocation in the hot loop.  Per generated token: one
 * rolling-hash update (~4 integer ops) and, while NOT locked onto a candidate
 * cycle, one scan of the last `max_period` shingle hashes — a contiguous 4 KB
 * u64 array, so it is a sequential compare sweep and not a data structure walk.
 * Once locked the per-token cost is a single integer compare.
 *
 * MEASURED (M4, -O3, 2M tokens): 209 ns/token never-locked (the steady state
 * for normal output), 3.6 ns/token while locked.  Against a ~25 ms K80 decode
 * step that is under 0.001% even after allowing an order of magnitude for the
 * a K80 host's much slower CPU — and on the batched path it runs on the request
 * thread that already detokenizes and streams each token, which is off the
 * batch's critical path entirely.
 *
 * State is two rings sized to the next power of two above `max_period + 32`:
 * 12 KB for the default 512-token cap, allocated once per completion.
 *
 * Depends only on libc — unit-tested standalone in
 * tests/test_repetition_loop_detector.c, no engine/CUDA.
 *
 * @status Complete.  Wired into all three decode loops in serve_generation.c
 *         (single-user, batched driver-sampling, batched legacy lockstep); a
 *         trip ends the turn with finish_reason SG_FINISH_REPETITION_LOOP.
 * @issues None
 * @todo None
 */

#ifndef PN_REPETITION_LOOP_DETECTOR_H
#define PN_REPETITION_LOOP_DETECTOR_H

#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>

/* Length of the fingerprint window whose recurrence proposes a candidate cycle.
 * 32 tokens (~120 chars) is long enough that an accidental recurrence is
 * vanishingly unlikely, and short enough that the whole window fits well inside
 * the smallest cycle we are willing to act on. */
#define RLD_SHINGLE_TOKENS 32

/* Detector thresholds.  See the header comment for the reasoning; every one of
 * them is overridable at runtime (rld_default_config). */
typedef struct {
    int min_period;    /* shortest cycle we will act on, in tokens */
    int max_period;    /* longest cycle we track (bounds the ring memory) */
    int min_repeats;   /* back-to-back byte-identical copies required to fire */
} rld_config;

/* Detector state for ONE completion.  Zero-initialize and rld_init; feed every
 * generated token in order; rld_free when the turn ends. */
typedef struct {
    rld_config cfg;

    uint32_t *token_ring;      /* token id of absolute index i at [i & mask] */
    uint64_t *shingle_ring;    /* hash of the RLD_SHINGLE_TOKENS-token window
                                * ENDING at absolute index i, at [i & mask] */
    size_t    ring_mask;       /* ring capacity - 1 (capacity is a power of 2) */

    long      n_fed;           /* absolute index of the next token to be fed */
    uint64_t  rolling_hash;    /* hash of the window ending at the last token */
    uint64_t  shingle_power;   /* B^(RLD_SHINGLE_TOKENS-1), to drop the token
                                * leaving the window */

    int       locked_period;   /* candidate cycle length; 0 = not locked */
    long      verified_run;    /* consecutive positions confirmed equal to the
                                * token `locked_period` back */

    bool      tripped;         /* a loop was detected (latched) */
    int       cycle_tokens;    /* the detected cycle length, in tokens */
    int       repeats;         /* how many back-to-back copies were confirmed */
} rld_detector;

/**
 * Is loop detection enabled for this process?
 *
 * ON by default; `PANTHEON_REPETITION_STOP=0` turns it off.  On by default
 * because at these thresholds a trip means the model has already emitted 576+
 * tokens of byte-identical, back-to-back repeated content with no novel token
 * in it — output that is worthless to the caller whether or not the model was
 * degenerate — and because the failure it catches is live in production and is
 * currently reported to the caller as a normal, tidy finish.
 *
 * Read once and cached; process-constant.
 *
 * Thread-safe: yes (idempotent lazy init of an int)
 * Reentrant: yes
 * Signal-safe: no (getenv)
 */
bool rld_enabled(void);

/**
 * Fill `cfg` with the process-wide thresholds, resolved once from the
 * environment and clamped to sane ranges:
 *   PANTHEON_REPETITION_MIN_PERIOD   (default 48, min 1)
 *   PANTHEON_REPETITION_MAX_PERIOD   (default 512, clamped to [min_period, 4096])
 *   PANTHEON_REPETITION_MIN_REPEATS  (default 12, min 2)
 *
 * @param cfg  Output; untouched when NULL.
 *
 * Thread-safe: yes (idempotent lazy init)
 * Reentrant: yes
 * Signal-safe: no (getenv)
 */
void rld_default_config(rld_config *cfg);

/**
 * Initialize a detector.  Allocates the two rings.
 *
 * @param rd   Detector to initialize (fully overwritten).
 * @param cfg  Thresholds, or NULL for rld_default_config().
 * @return     true when the detector is live; false when it is inert (a NULL
 *             `rd`, or the ring allocation failed).  An inert detector is safe
 *             to feed and free — it simply never fires, so a caller never has
 *             to branch on the return value to stay correct.
 *
 * Thread-safe: yes (touches only `rd`)
 * Reentrant: yes
 * Signal-safe: no (malloc)
 */
bool rld_init(rld_detector *rd, const rld_config *cfg);

/**
 * Feed one generated token id, in generation order.
 *
 * @param rd     Detector (NULL or inert → always false).
 * @param token  The token id just generated.
 * @return       true EXACTLY ONCE — on the token at which the loop is
 *               confirmed.  The caller should end the turn and report
 *               SG_FINISH_REPETITION_LOOP.  Every later call returns false, so
 *               a caller that keeps feeding cannot double-report.
 *
 * After a trip, `cycle_tokens` and `repeats` describe what was detected.
 *
 * Thread-safe: no (mutates `rd`; one detector per completion)
 * Reentrant: no
 * Signal-safe: yes (no allocation, no libc calls)
 */
bool rld_feed(rld_detector *rd, uint32_t token);

/** Release the detector's rings.  NULL-safe; safe to call twice. */
void rld_free(rld_detector *rd);

#endif /* PN_REPETITION_LOOP_DETECTOR_H */
