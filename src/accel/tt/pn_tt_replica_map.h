/* SPDX-License-Identifier: AGPL-3.0-or-later */
/* Copyright (C) 2026 David (Corey) Herrera */
/**
 * @file pn_tt_replica_map.h
 * @brief Mirror the HOT activation strips across DRAM banks, and tell the producers.
 *
 * ---------------------------------------------------------------------------
 * THE PROBLEM, MEASURED
 * ---------------------------------------------------------------------------
 * The Blackhole's 510.8 GB/s is EIGHT banks of ~59. Every member of a column-split matmul
 * reads the WHOLE A strip, so N members are N readers of ONE bank. Per-bank during Ornith
 * decode, counted from the descriptors:
 *
 *     b0 5.8   b1 3.1   b2 43.1   b3 4.0   b4 4.9   b5 4.0   b6 3.1   b7 14.0  GB/s
 *
 * One bank at 73% of its own ceiling, seven nearly idle, the card at 16% of peak. That —
 * and not a core-scaling knee — is why matmul card work is FLAT from 4 members to 16 and
 * why the shipped setting is four cores on a 140-Tensix part. Adding readers to a saturated
 * bank does nothing.
 *
 * The discriminating null is already recorded: the SAME bank 4 MiB apart reproduces the
 * control to three decimals, while a DIFFERENT bank per member restores 1.90x / 1.53x a
 * doubling. So it is the bank, not address locality and not a row buffer.
 *
 * ---------------------------------------------------------------------------
 * WHY MIRRORS, AND NOT THE THREE CHEAPER THINGS
 * ---------------------------------------------------------------------------
 * 1. **More DOORS of the same bank** — already shipped (`PN_TT_ACTIVATION_DOORS`) and
 *    already exhausted. A door is a NoC port and lifts the ~45 GB/s ENDPOINT ceiling to the
 *    ~59 GB/s BANK ceiling. The bank ceiling is the one being hit.
 * 2. **A different ALLOCATION POLICY** — there is none that helps. The traffic is N readers
 *    of one strip, so no placement of that one strip spreads it; only more than one copy of
 *    the bytes does, or a layout that puts different PARTS of the strip in different banks.
 * 3. **INTERLEAVING the strip across banks by tile** — genuinely cheaper in bytes, and
 *    rejected. It moves the change into `resident_matmul_reader.c`, the one kernel the
 *    `cmp` gate rests on and the one with four program variants; it needs a bank table
 *    inside that reader plus a descriptor word to point at it; it breaks every OTHER
 *    consumer of the strip (the host uploads, the GDN op, attention, the elementwise
 *    reader), because the strip would no longer be contiguous anywhere; and it cuts A's
 *    request size from 16,384 B to 2,048, into a band this card is measurably worse in.
 *    Mirroring keeps replica 0 at the strip's own address, so every one of those consumers
 *    is untouched and the matmul reader does not change at all.
 * 4. **COOPERATIVE multicast**, where a leader reads A once and broadcasts it — implemented,
 *    re-timed cleanly on 2026-08-19 with the residencies moved off the rectangle, and a
 *    NULL: control 0.5116, rect 0.5125, coop 0.5115 s/token.
 *
 * What mirroring costs is WRITES, which are the cheap direction on this card, and they are
 * card-to-card. The strips are 256 KiB (`normed`) and 768 KiB (`ffn_swiglu`); at 8 mirrors
 * that is ~112 MiB/token of extra NoC writes spread over eight banks.
 *
 * ---------------------------------------------------------------------------
 * WHICH STRIPS, AND WHY ONLY THOSE
 * ---------------------------------------------------------------------------
 * Counted from Ornith's own shapes, A's bytes divide per layer as
 *
 *     normed 70.6%   ffn_swiglu 17.6%   ssm_gated / attn_out 11.8%
 *
 * `normed` is the RMS-norm output that feeds all seven projections of every layer;
 * `ffn_swiglu` is `silu(gate)*up` and feeds `ffn_down`. Their producers are the REDUCE and
 * ELTWISE residencies — and both of those drain through the SAME kernel,
 * `resident_vector_writer.c`, so ONE writer change covers 88% of A. The remaining 12% is
 * produced by the host in the shipped forward, where `pn_tt_activation_upload()` fans out
 * on its own.
 *
 * ---------------------------------------------------------------------------
 * THE FAILURE MODE THIS FILE EXISTS TO PREVENT
 * ---------------------------------------------------------------------------
 * A mirror nobody refreshes is STALE, and a matmul member that reads it returns a plausible
 * wrong answer — every kernel completes, every residency is healthy, no error word is set.
 * So the map is published to every core that could produce a mirrored strip, and it is READ
 * BACK and compared rather than trusted to the write's return code.
 *
 * @status Written 2026-08-19 and ON at 8 the same day, gated on the 48-token oracle across
 *         thirteen mirrored arms. `PN_TT_ACTIVATION_REPLICAS=1` is the control.
 * @issues None
 * @todo None
 */

#ifndef PN_TT_REPLICA_MAP_H
#define PN_TT_REPLICA_MAP_H

#ifdef USE_TT

#include <stdbool.h>
#include <stdint.h>

#include "pn_tt_device.h"
#include "pn_tt_residency.h"

/** `PN_TT_ACTIVATION_REPLICAS=N` mirrors the hot strips into N distinct DRAM banks. */
#define PN_TT_ACTIVATION_REPLICAS_ENV "PN_TT_ACTIVATION_REPLICAS"

/**
 * EIGHT — one per DRAM bank — SINCE 2026-08-19, and it is a measured default rather than a
 * shape argument.
 *
 * Gated on the 48-token oracle and nothing softer: thirteen mirrored arms across member
 * counts 1..32 and mirror counts 2/4/8 are all `cmp` IDENTICAL (189 B, md5 `80fc6259`),
 * every one at `company=0` with zero residency re-arms, beside eleven unmirrored controls
 * and a red arm (`PN_TT_RED=replica_stale`) that DIFFERS.
 *
 * What it buys, at 16 members: matmul card work **0.0959 -> 0.0595 s/token (1.61x)** and
 * the per-bank table flat — b2 41.6 GB/s against seven banks at 11-19 becomes 25.5-29.0
 * across all eight. What it costs is the fan-out: the on-card families go 0.040 -> 0.042
 * s/token, i.e. **2 ms against the 36 ms it returns.**
 *
 * `PN_TT_ACTIVATION_REPLICAS=1` is the control arm and is the pre-replication forward word
 * for word — the map is then empty and every producer's writer misses it.
 */
#define PN_TT_ACTIVATION_REPLICAS_DEFAULT 8u

/** How many mirrors this run asked for, clamped to the bank count. 1 means off. */
uint32_t pn_tt_activation_replica_count(const pn_tt_device *device);

/**
 * Publish the replica map to every residency that could produce one of these strips.
 *
 * The map is keyed on the strip's PRIMARY (bank coordinate, address), which is what the
 * producing descriptor already carries in its output words — the REDUCE family has no free
 * parameter word left, so the fan-out cannot be addressed per descriptor.
 *
 * Strips with one replica are skipped rather than refused: a caller passes its whole strip
 * list and the ones that were not mirrored simply do not appear in the map.
 *
 * WRITTEN TO THE REDUCE, ELTWISE AND HEAD-NORM CORES, because those are the residencies
 * running `resident_vector_writer.c`. A core that never produces a mirrored strip is
 * harmless to tell: the lookup is keyed on the exact destination triple, so it misses.
 *
 * VERIFIED BY READ-BACK. A half-written map is a stale mirror and a stale mirror is a
 * plausible wrong answer; there is no refusal anywhere downstream to catch it.
 *
 * ---------------------------------------------------------------------------
 * `split_output` — AND WHY THE ELEMENTWISE SPLIT CANNOT BE TOLD FROM THE OTHERS
 * ---------------------------------------------------------------------------
 * **MEASURED ON SILICON 2026-08-24, AND IT COST THE LEVER ITS `cmp`.** The lookup is the
 * EXACT triple, and the header above says what a miss does: "a miss is the shipped path word
 * for word". `PN_TT_ELTWISE_CORES=N` hands member `m` the SAME descriptor with
 * `C_BASE + m * slice_bytes` — so member 0 hits the table and mirrors its slice, and every
 * member above it MISSES and leaves its slice stale in seven banks, which the matmul members
 * then read. The completion is clean, every kernel retires, nothing anywhere refuses. The
 * arms: `cmp` IDENTICAL at one member, 204 B at two, 8 B at four, 57 B at sixteen — wrong in
 * proportion to `(N-1)/N` of the strip, and the discriminating null is
 * `PN_TT_ACTIVATION_REPLICAS=1`, which is IDENTICAL at every member count because there is
 * then no table to miss.
 *
 * So a strip a SPLIT family produces needs one set per slice, not one set. `split_output` is
 * that strip — named by the caller, because nothing here can infer a producer, and the
 * distinction is load-bearing: `normed` is the REDUCE's output and reduce is never split, so
 * slicing it would spend the four-set table on entries no descriptor can hit.
 *
 * **AND THE MEMBERS ARE PUBLISHED TO.** They run `resident_vector_writer.c` like every other
 * core in this list; before this they were armed with no table at all, which is not "no
 * mirroring" but "whatever that L1 region held", and the file header is explicit that a map
 * left by a previous process is the difference between missing and fanning out to somebody
 * else's addresses.
 *
 * The table holds `FT_TT_REPLICA_MAX_SETS`. When a split needs more than that, the SPLIT is
 * collapsed to one core here and said out loud — the alternative is a correct-looking map
 * that silently covers some slices and not others.
 *
 * @param strips        pointers to the strips that may be mirrored, in any order
 * @param strip_count   how many
 * @param split_output  the one of them the ELEMENTWISE family produces, or NULL
 * @return 0, or -1 with the reason on stderr
 *
 * Thread-safe: no    Reentrant: no    Signal-safe: no
 */
int pn_tt_publish_replica_map(pn_tt_device *device, const pn_tt_activation *const *strips,
                              uint32_t strip_count, const pn_tt_activation *split_output);

/**
 * OUT-PROJECTIONS ON THE GRID (2026-10-27). `ranged`: every TALL mirrored strip (more than 4
 * bands) is keyed by ONE range record (any write inside it fans out at the same offset)
 * instead of the span / super-band keys; and the `gated` strips (`ssm_gated`, attention
 * `gated`, written band by band) get a range record in EVERY writer's table, so they can be
 * mirrored and their out-projections take the prefill grid. Republish after changing it.
 * `ranged` false = the keys above, word for word.
 */
void pn_tt_replica_map_set_ranged(bool ranged, const pn_tt_activation *const *gated,
                                  uint32_t gated_count);

/**
 * L2: the DECODE out-projection inputs
 * (`ssm_gated`, attention `gated`), mirrored late (pn_tt_late_mirrors.h), each keyed by ONE
 * range record in every writer's table — the eltwise members write slices and verify rows
 * write bands, and a range covers both. Pointers to the model's strip SLOTS: a strip taller
 * than 4 bands there (the tall set swapped in for a batched pass) is skipped, and one with a
 * single replica adds nothing. Republish after changing it. NULL / 0 clears it.
 */
void pn_tt_replica_map_set_decode_ranged(const pn_tt_activation *const *strips, uint32_t count);

/** Strips the fused-SwiGLU members' table can name (decode + tall `ffn_swiglu`, spare). */
#define PN_TT_REPLICA_MAP_FUSED_MAX 4u

/**
 * S3b: register the strips a FUSED SwiGLU descriptor writes, BY SLOT (pointer): a tall swap
 * exchanges the contents of the decode and tall strip structs, so the pair of slot pointers
 * names both strips whatever the swap state, and late mirrors are seen as they land. Clears
 * the published flag; pn_tt_replica_map_publish_fused() sets it.
 *
 * Thread-safe: no    Reentrant: no    Signal-safe: no
 */
void pn_tt_replica_map_set_fused_swiglu(const pn_tt_activation *const *strips, uint32_t count);

/**
 * S3b: publish (and read back) the matmul members' replica table — one RANGE record per
 * registered mirrored strip, keyed at its band-0 primary over every band — to every running
 * member of the matmul group. A no-op with the knob off or nothing registered. Called by
 * pn_tt_publish_replica_map() and after every matmul-group arm (a re-arm re-plants it).
 *
 * @return 0, or -1 (a write or read-back failed; the fused path is then refused)
 *
 * Thread-safe: no    Reentrant: no    Signal-safe: no
 */
int pn_tt_replica_map_publish_fused(pn_tt_device *device);

/**
 * S3b: may a fused descriptor write `output`? True for an unmirrored strip (no fan-out
 * needed), or a mirrored one that a published members' table names.
 *
 * Thread-safe: no    Reentrant: no    Signal-safe: no
 */
bool pn_tt_replica_map_fused_covers(const pn_tt_device *device, const pn_tt_activation *output);

/** PHASE G: the prefill span (bands a pass's ops cover) the keys of a TALL strip (> 4 bands)
 *  are published for; republish after changing it. 0/1 = decode only. */
void pn_tt_replica_map_set_span(uint32_t span);

/** PHASE W2: producers also write SUPER-BANDS of `bands` bands at band j*bands of the prefill
 *  span; key those too (0 = off). Republish after changing it. */
void pn_tt_replica_map_set_superband(uint32_t bands);

/**
 * Write an EMPTY table to one residency that runs `resident_vector_writer.c` and was armed
 * AFTER `pn_tt_publish_replica_map()` had already run.
 *
 * "Empty" and "never written" are different states on this card, and only one of them is
 * safe. The table lives at a FIXED L1 address inside a region every program owns, so a
 * residency nobody published to reads whatever is at `FT_TT_REPLICA_TABLE_ADDRESS` — the
 * magic word protects a zeroed or erased region, but it does not protect against a REAL
 * table another process wrote on that core. An empty table with the magic word and a zero
 * set count is a miss for every descriptor, which is the shipped path word for word, and it
 * is a written fact rather than an assumption.
 *
 * The merged gate's output (`ssm_gated`) is genuinely not mirrored — A's bytes are `normed`
 * 70.6% / `ffn_swiglu` 17.6% and only those two are replicated — so empty is the CORRECT
 * table here and not a stub.
 *
 * @return 0, or -1 with the reason on stderr. A residency that is not running is a no-op.
 *
 * Thread-safe: no    Reentrant: no    Signal-safe: no
 */
int pn_tt_publish_empty_replica_map(pn_tt_device *device, const pn_tt_residency *residency,
                                    const char *label);

/**
 * The persistent norm role's members (pn_tt_norm_role.h): the reduce residency's table, built
 * from the strips the last pn_tt_publish_replica_map() was given. Called at the role's arm and
 * by every publish after it. No members running: nothing written.
 *
 * @return 0, or -1 with the reason on stderr
 *
 * Thread-safe: no    Reentrant: no    Signal-safe: no
 */
int pn_tt_replica_map_publish_role(pn_tt_device *device);

#endif /* USE_TT */
#endif /* PN_TT_REPLICA_MAP_H */
