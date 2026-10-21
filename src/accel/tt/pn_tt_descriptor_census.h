/* SPDX-License-Identifier: AGPL-3.0-or-later */
/* Copyright (C) 2026 David (Corey) Herrera */
/**
 * @file pn_tt_descriptor_census.h
 * @brief Attribute on-card descriptor time by (FAMILY, FUNCTION, TILE COUNT) — one level
 *        below the per-residency report, with no card and no device in it.
 *
 * ---------------------------------------------------------------------------
 * WHY THIS EXISTS: "293.8 us EACH" IS THE SAME AVERAGE, ONE LEVEL DOWN
 * ---------------------------------------------------------------------------
 * `qwen35_tt_report_residencies()` was written because "162 descriptors a token, 247 us
 * each" averaged over families whose work differs by three orders of magnitude. It replaced
 * that with a line per residency — and the line it produced for the elementwise family reads
 * **136 descriptors/token, 293.8 us each**, which is the identical mistake at the next level:
 *
 *   - the eltwise residency runs SIX different functions (silu, sigmoid, add, mul, sub,
 *     silu_mul), unary and binary, over at least THREE different widths (`n_embd/32`, the
 *     FFN width, the attention gate's width);
 *   - a unary SIGMOID over the gate and a binary SILU_MUL over the FFN are both "one
 *     descriptor", and one reads half the bytes of the other;
 *   - the two RENDER directions run the same three kernel sources in two other programs at
 *     two other page sizes, and they are **not in the residency table at all** — the
 *     bridge's ~0.022 s/token has only ever been a SUBTRACTION between runs.
 *
 * So the on-card term's largest non-matmul family has never been decomposed, and the second
 * largest is invisible to the instrument. This file is the missing key.
 *
 * ---------------------------------------------------------------------------
 * AND WHAT THE KEY BUYS THAT A COUNT CANNOT
 * ---------------------------------------------------------------------------
 * A family that runs the SAME program over SEVERAL widths prices its own per-descriptor
 * fixed cost, from one ordinary run, with no extra card arm:
 *
 *     seconds_per_descriptor(width) = fixed + per_tile * tile_count
 *
 * `pn_tt_descriptor_census_fit()` solves that line by weighted least squares over a family's
 * rows. **The intercept is what merging descriptors removes and the slope is what it does
 * not** — which is the one question a "N descriptors x 293.8 us" figure cannot answer, and
 * the one that three levers here have already been scoped wrong from (two of which measured
 * exactly 0 when built).
 *
 * The intercept is NOT "the host round trip" and must not be reported as one. It is
 * everything that does not scale with the strip: the parameter write, the ring write, the
 * head bump, the first poll's PCIe latency, the kernel's own per-descriptor prologue, the CB
 * rebase and the completion post. Separating THOSE needs
 * `pn_tt_residency_submit_repeat()`, which is a different instrument and a card arm.
 *
 * ---------------------------------------------------------------------------
 * WHAT IT REFUSES, AND WHY EACH REFUSAL IS LOUD
 * ---------------------------------------------------------------------------
 * A census that silently drops rows reads exactly like a census of a cheaper model. So:
 *
 *   - a table overflow does NOT fold into an "other" bucket — it increments `dropped`, and
 *     any reader that prints a census must print `dropped` beside it;
 *   - a fit over ONE distinct width is refused, not answered. Two unknowns need two
 *     equations, and a line through one point is the arithmetic that produced the retired
 *     "958 ns per single-tile op". TWO guards enforce this — the named shape count and the
 *     normal equation's determinant — and they are REDUNDANT WITH EACH OTHER: mutating either
 *     one away leaves the suite green and only removing BOTH goes red (4 failures). They are
 *     both kept, and that redundancy is stated here rather than left as a check nobody can
 *     prove red on its own;
 *   - a non-finite duration is refused at record time, because `failed = value > limit` is
 *     false for a NaN and a NaN that reaches a fit poisons every figure downstream.
 *
 * ---------------------------------------------------------------------------
 * A DESCRIPTOR THAT REFUSED AND WAS REPLAYED IS NOT A DESCRIPTOR'S COST
 * ---------------------------------------------------------------------------
 * Every resident family in this backend submits under a recovery: if the residency declines
 * the descriptor the host RE-ARMS it — a full teardown and relaunch — and submits again, and
 * the caller's stopwatch spans all three. Folding that bracket into `seconds / ops` prices a
 * teardown as arithmetic, and it has already produced a wrong headline number: PHASE CONV read
 * **4,502 us a descriptor against 738.2** for no reason but a circular-buffer write pointer
 * that made every CONV after a DELTA refuse. The recorded "CONV is DATA-DEPENDENT, 4,571 us on
 * a degenerate stream" has the same shape and no surviving harness.
 *
 * So `ops` and `seconds` are the CLEAN descriptors — first attempt, retired — and a replayed
 * one goes to `recovered_ops` / `recovered_seconds` / `rearms` instead. Three consequences,
 * and the middle one is the point:
 *
 *   - `pn_tt_descriptor_census_fit()` sees clean rows only, so a run with refusals fits the
 *     op rather than the recovery;
 *   - a reader that prints `us each` MUST print `rearms` beside it, exactly as it must print
 *     `dropped` — a contaminated figure that looks like a clean one is the whole defect;
 *   - `pn_tt_descriptor_census_family_totals()` answers the CLEAN totals, so wall-time
 *     accounting that wants everything adds `pn_tt_descriptor_census_family_recovery()`.
 *
 * No device, no allocation, no I/O — deliberately, so it is provable with no card.
 *
 * @status None
 * @issues None
 * @todo None
 */

#ifndef PN_TT_DESCRIPTOR_CENSUS_H
#define PN_TT_DESCRIPTOR_CENSUS_H

#include <stdbool.h>
#include <stdint.h>

/** Distinct (family, function, tile count) keys the table holds. Ornith's decode path uses
 *  well under half of these; the margin is for a hybrid arm that runs more shapes. */
#define PN_TT_DESCRIPTOR_CENSUS_ROWS 64u

/** Key strings are COPIED, not borrowed, so a caller may pass a stack buffer. */
#define PN_TT_DESCRIPTOR_CENSUS_NAME_BYTES 24u

/** One key and everything measured against it. */
typedef struct pn_tt_descriptor_census_row {
    char family[PN_TT_DESCRIPTOR_CENSUS_NAME_BYTES];   /**< "eltwise", "render-widen", ... */
    char function[PN_TT_DESCRIPTOR_CENSUS_NAME_BYTES]; /**< "add", "silu_mul", "widen", ... */
    uint32_t tile_count;                               /**< tiles the descriptor covers */
    uint64_t ops;      /**< CLEAN descriptors on this key — first attempt, retired */
    double seconds;    /**< summed submit wall time of those clean descriptors */
    /** Descriptors the residency REFUSED and the host replayed after a re-arm, and the whole
     *  bracket they cost. Never folded into `seconds`: a teardown is not a descriptor's work,
     *  and `seconds / ops` is quoted as one. */
    uint64_t recovered_ops;
    double recovered_seconds;
    /** Re-arms performed inside those brackets. `>= recovered_ops`; a caller that re-arms more
     *  than once for one descriptor says so. */
    uint64_t rearms;
} pn_tt_descriptor_census_row;

/** The whole instrument. Zero-initialise it; there is no constructor. */
typedef struct pn_tt_descriptor_census {
    pn_tt_descriptor_census_row rows[PN_TT_DESCRIPTOR_CENSUS_ROWS];
    uint32_t row_count;
    /** Descriptors that matched no row and found no free slot, and their time. NEVER folded
     *  into a row: an "other" bucket makes a truncated census look complete. */
    uint64_t dropped;
    double dropped_seconds;
    /** Descriptors refused at record time — a bad width or a non-finite duration. Kept apart
     *  from `dropped` because they mean a defect upstream, not a table too small. */
    uint64_t refused;
} pn_tt_descriptor_census;

/**
 * Add one descriptor's time to its key, creating the row on first sight.
 *
 * @param census      the table; NULL is refused
 * @param family      residency-level name, truncated to the key width
 * @param function    what the descriptor computed, truncated to the key width
 * @param tile_count  tiles the descriptor covers; 0 is refused
 * @param seconds     measured submit wall time; non-finite or negative is refused
 * @return            0, or -1 with `refused` or `dropped` incremented
 *
 * Thread-safe: no    Reentrant: no    Signal-safe: no
 */
int pn_tt_descriptor_census_record(pn_tt_descriptor_census *census, const char *family,
                                   const char *function, uint32_t tile_count, double seconds);

/**
 * Add one REPLAYED descriptor's whole bracket — refusal, re-arm, resubmit — to its key.
 *
 * Identical keying and identical refusals to `pn_tt_descriptor_census_record()`; the only
 * difference is which pair of counters moves, and that difference is the point. Nothing this
 * function records ever reaches `seconds / ops` or the fit.
 *
 * @param census      the table; NULL is refused
 * @param family      residency-level name, truncated to the key width
 * @param function    what the descriptor computed, truncated to the key width
 * @param tile_count  tiles the descriptor covers; 0 is refused
 * @param seconds     the WHOLE bracket, not the replay alone; non-finite or negative refused
 * @param rearms      re-arms performed inside the bracket; 0 is refused, because a recovered
 *                    descriptor that re-armed nothing is a caller that mislabelled a clean one
 * @return            0, or -1 with `refused` or `dropped` incremented
 *
 * Thread-safe: no    Reentrant: no    Signal-safe: no
 */
int pn_tt_descriptor_census_record_recovered(pn_tt_descriptor_census *census,
                                             const char *family, const char *function,
                                             uint32_t tile_count, double seconds,
                                             uint64_t rearms);

/**
 * Read row `index` back, in first-sight order.
 *
 * @return false once `index` reaches `row_count`, which is the loop's terminator
 */
bool pn_tt_descriptor_census_row_at(const pn_tt_descriptor_census *census, uint32_t index,
                                    const pn_tt_descriptor_census_row **out_row);

/**
 * Sum one family's rows.
 *
 * @param family     matched exactly against the recorded key; NULL sums EVERY row
 * @return           0, or -1 when the family has no rows (so a caller cannot print zeros
 *                   for a family that never ran and read it as "free")
 */
int pn_tt_descriptor_census_family_totals(const pn_tt_descriptor_census *census,
                                          const char *family, uint64_t *out_ops,
                                          double *out_seconds, uint32_t *out_shapes);

/**
 * The same family's REPLAYED descriptors — what `..._family_totals()` deliberately leaves out.
 *
 * A reader that prints a per-descriptor figure has to print `out_rearms` beside it. It is
 * answered as 0/0/0 rather than refused when the family ran clean, because "the family has no
 * recovery" and "the family never ran" are different facts and only the second is a refusal.
 *
 * @param family              matched exactly against the recorded key; NULL sums EVERY row
 * @param out_recovered_ops   descriptors that refused and were replayed
 * @param out_recovered_seconds  their whole brackets, summed
 * @param out_rearms          re-arms performed inside them
 * @return                    0, or -1 when the family has no rows at all
 */
int pn_tt_descriptor_census_family_recovery(const pn_tt_descriptor_census *census,
                                            const char *family, uint64_t *out_recovered_ops,
                                            double *out_recovered_seconds,
                                            uint64_t *out_rearms);

/**
 * Solve `seconds_per_descriptor = fixed + per_tile * tile_count` for one family, by least
 * squares over its rows weighted by descriptor count.
 *
 * REFUSED unless the family shows at least two DISTINCT tile counts. Read the header's
 * "what the key buys" block before quoting the intercept as a host round trip — it is not
 * one, and this function cannot tell you that it is.
 *
 * CLEAN ROWS ONLY. A row whose descriptors all refused and were replayed contributes no
 * equation, so a family that never retired a first attempt is refused rather than fitted to
 * its own recovery path.
 *
 * @param out_fixed_seconds   intercept, seconds per descriptor at zero tiles
 * @param out_seconds_per_tile slope, seconds per tile
 * @param out_shapes          distinct tile counts the fit used; NULL to ignore
 * @return                    0, or -1 (no such family, one shape only, or a degenerate
 *                            normal equation)
 */
int pn_tt_descriptor_census_fit(const pn_tt_descriptor_census *census, const char *family,
                                double *out_fixed_seconds, double *out_seconds_per_tile,
                                uint32_t *out_shapes);

#endif /* PN_TT_DESCRIPTOR_CENSUS_H */
