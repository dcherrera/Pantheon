/* SPDX-License-Identifier: AGPL-3.0-or-later */
/* Copyright (C) 2026 David (Corey) Herrera */
/**
 * @file pn_tt_gdn.h
 * @brief The GATED-DELTA-NET block on the card: one resident core running
 *        `foundry/tt/ft_tt_op_gdn.h`'s two phases over per-layer state that lives in card
 *        DRAM and is never downloaded.
 *
 *        Model-agnostic on purpose. It knows S, the head counts, the conv width and the tap
 *        count; it does not know what a Qwen3.5 layer is. `src/models/qwen35moe_gdn_tt.c`
 *        is the caller and owns the host arithmetic this op deliberately does not do.
 *
 * ---------------------------------------------------------------------------
 * WHAT THIS OWNS AND WHAT IT DOES NOT
 * ---------------------------------------------------------------------------
 * The op is TWO descriptors a layer — `conv1d + SiLU` with the rolling window, then the
 * delta rule over every v-head — and between them sit two per-head NORMS that are NOT in it:
 * the q/k L2 norm, and the output gated norm. Those are ordinary reduces at a SECOND width
 * (`S/32` = 4 tile columns against the model's `n_embd/32` = 128) and they live on the narrow
 * `reduce_fp32` residency `pn_tt_head_norm.h` places.
 *
 * **THERE ARE TWO CALLERS AND THEY ARE NOT THE SAME ARM.**
 *
 *   HOST-MEDIATED  `pn_tt_gdn_conv()` / `pn_tt_gdn_delta()` take and return host float
 *                  arrays, so the two norms between them run on the CPU. That arm is SLOWER
 *                  than the host recurrence and it is arithmetic, not a defect: the host
 *                  still pulls the 49,408 B of projections AND now pulls `conv_out` (32,768 B)
 *                  and `y` (16,384 B) a layer. It is the CORRECTNESS vehicle — the thing the
 *                  on-card gate scores against the CPU oracle — and nothing else.
 *   THE STRIP      `pn_tt_gdn_conv_oncard()` / `pn_tt_gdn_delta_oncard()` take nothing and
 *   BRIDGE         return nothing: every operand is already in card DRAM, the norms run on
 *                  the card between them, and the only bytes that cross are `alpha`/`beta`
 *                  down and 256 B of gate block up. `qwen35moe_gdn_strip_tt.c` is the caller
 *                  and readback goes 1478.09 -> 326.09 KiB/token.
 *
 * ---------------------------------------------------------------------------
 * WHAT LIVES ON THE CARD, PER SSM LAYER, AND WHAT IT COSTS
 * ---------------------------------------------------------------------------
 * Every size comes from `ft_tt_op_gdn_*_bytes()` so the capacity figure and the allocation
 * are one expression. At Ornith's shape (S=128, 32 v-heads, 16 k-heads, conv_dim 8192,
 * 4 taps):
 *
 *   the recurrent state   2 MiB   [v_heads, S, S] fp32, oracle-transposed, READ AND WRITTEN
 *   the conv window       3 MiB   taps-1 TAP-MAJOR PLANES of conv_dim/32 tiles, R/W
 *   the conv weight       4 MiB   taps planes, a CONSTANT staged once, TRANSPOSED at load
 *                         -----
 *                         9 MiB a layer, 216 MiB over 24 SSM layers
 *
 * The 50.25 MiB the capacity plan reserves counted the window as `conv_dim * (taps-1)` fp32
 * ELEMENTS. A plane is a one-row tilized strip and the tile quantum is 32 rows, so the window
 * is 32x that; carried state alone is 120 MiB. See `ft_tt_op_gdn.h`.
 *
 * Plus ONE shared set of per-token staging buffers — the qkv stream in, the stream out (which
 * is CONV's output AND DELTA's input, one buffer since both are the same row-0 strip), the
 * gate block and `y` — 2.6 MiB total, because a token runs one layer at a time and nothing
 * carries between them.
 *
 * @status SHIPS ON. On silicon (p150a, 2026-08-21) the strip-bridge arm is `cmp` IDENTICAL
 *         to the 48-token oracle at **326.09 KiB/token and 3.79 tok/s**, against the control's
 *         1478.09 and 2.20; the host-mediated arm beside it is IDENTICAL too, at 2630.09 and
 *         1.43. `PN_TT_ONCARD=all` selects the family now — but the block WITHOUT the bridge
 *         is a wall regression, so a DEFAULT arm that loses the bridge gives the recurrence
 *         back to the host and a NAMED one keeps it.
 * @issues **EVERY NUMBER MEASURED ON A CONTESTED CORE IS VOID.** Core 75 is a matmul group
 *         member at `PN_TT_MATMUL_CORES=4` and members run upward from it, so the contested
 *         set MOVES with the count. While two residencies were retiring each other the same
 *         CONV descriptor scored 1.0e-2 max-norm with 1957 of 8192 channels off; on an
 *         uncontested core it is 6.872e-04 with 3. A contested core does not fail, it
 *         computes badly, and both arms log a healthy "residency up".
 *
 *         The carried state is reset when the forward is asked for position 0. A caller that
 *         starts mid-sequence from a prefix cache would inherit the previous sequence's
 *         state; the host arm has the same shape of problem and solves it by owning the
 *         state per sequence, which this cannot while there is one region set per model.
 * @todo
 *   - [ ] An all-Float32 elementwise program, so the strip bridge's `silu(z) *` runs in fp32
 *         and only the final `gated` is narrowed. Modelled at 1.909e-03 against 1.023e-02.
 */

#ifndef PN_TT_GDN_H
#define PN_TT_GDN_H

#ifdef USE_TT

#include <stdint.h>

#include "pn_tt_device.h"
#include "tt/ft_tt_op_dispatch.h"

/** The shape one model's GDN block has. Every field is read off the model's config, never
 *  assumed — `k_heads` must divide `v_heads` because the GQA repeat is `vh % k_heads`. */
typedef struct pn_tt_gdn_shape {
    uint32_t state_size;  /**< S; head_k_dim == head_v_dim, a multiple of 32 */
    uint32_t v_heads;
    uint32_t k_heads;
    uint32_t conv_dim;    /**< 2*key_dim + val_dim, a multiple of 32 */
    uint32_t conv_taps;   /**< ssm_conv; the window is taps-1 deep */
    uint32_t layer_count; /**< SSM layers, each with its own carried state */
} pn_tt_gdn_shape;

/**
 * Arm the GDN residency (one more core) and allocate every layer's carried state.
 *
 * Never fatal on its own — a card without the `gdn` program leaves the recurrence on the
 * host — but `PN_TT_ONCARD=gdn` makes a MISSING program a refusal, exactly like the other
 * families (pn_tt_device_arm_oncard).
 *
 * The conv WEIGHT is allocated here and left uninitialised: it is a per-layer constant and
 * the caller uploads it with `pn_tt_gdn_conv_weight_upload()` once, at load.
 *
 * @return 0 (even when nothing armed), -1 on a bad argument
 *
 * Thread-safe: no    Reentrant: no    Signal-safe: no
 */
int pn_tt_device_arm_gdn(pn_tt_device *device, const pn_tt_gdn_shape *shape);

/** Did the GDN family actually arm for this run? */
bool pn_tt_gdn_is_armed(const pn_tt_device *device);

/** L1a (prefill rework): how many decode GDN residencies (`gdn` + head-split members) could be
 *  lent to prefill; 0 when the family is not armed. */
uint32_t pn_tt_gdn_lend_count(const pn_tt_device *device);

/**
 * L1a: retire decode GDN residency `member_index` (0 = `gdn`) so its core can serve prefill.
 * Its carried state is in DRAM and untouched. @return its core, or UINT32_MAX when its firmware
 * is not on a core or it did not retire (then nothing may boot on that core)
 */
uint32_t pn_tt_gdn_lend(pn_tt_device *device, uint32_t member_index);

/** L1a: arm decode GDN residency `member_index` fresh on `core` again (its firmware re-booted).
 *  @return true when it is running */
bool pn_tt_gdn_restore(pn_tt_device *device, uint32_t member_index, uint32_t core);

/**
 * TRANSPOSE one layer's conv1d weight into tap-major planes and upload it. Once, at load.
 *
 * `weight` is the oracle's own `layer->ssm_conv1d`: ROW-MAJOR `[conv_dim, taps]`, i.e.
 * `w[c*taps + t]`. The card wants `taps` PLANES, plane `t` holding `w[.][t]` laid out exactly
 * like the qkv stream. **That is a transpose and not a reshape** — a reshape gives a wrong
 * answer with no error anywhere, which is why this is a named function taking the oracle's
 * order rather than a memcpy at the call site.
 *
 * @param layer   0..layer_count-1, the SSM-layer ordinal the two calls below also take
 * @param weight  `conv_dim * taps` floats in the oracle's order
 * @return        0, or -1 with the reason on stderr
 */
int pn_tt_gdn_conv_weight_upload(pn_tt_device *device, uint32_t layer, const float *weight);

/** Zero every layer's carried state — the conv window and the recurrent state. The start of
 *  a sequence, and the state this op is armed with. */
int pn_tt_gdn_state_reset(pn_tt_device *device);

/** STEP 4 S2 — the verify's per-row states, copy-free. Reserve `slots` snapshot slots (every
 *  layer's window + state, on the live region's bank); route the decode CONV/DELTA writes of
 *  the state to slot `slot` (-1: back in place); trade layer `layer`'s live window + state with
 *  slot `slot`'s. @return 0 or -1 */
int pn_tt_gdn_snapshot_reserve(pn_tt_device *device, uint32_t slots);

/** Allocate the prefix checkpoint's twin set now (it is otherwise allocated at the first
 *  capture, in each layer's own bank). 0, or -1. */
int pn_tt_gdn_checkpoint_reserve(pn_tt_device *device);

/**
 * PROJECT, allocating nothing: the least free bytes any DRAM bank would keep after `slots`
 * snapshot slots (those not yet reserved), the checkpoint set if `checkpoint` and not yet
 * held — both in each layer's own bank — and `stacks` blocks of `stack_bytes`, each into the
 * bank with the most free (pn_tt_moe_place_stack's rule). 0, or -1.
 */
int pn_tt_gdn_project_bank_free(pn_tt_device *device, uint32_t slots, bool checkpoint,
                                uint32_t stacks, uint64_t stack_bytes, uint64_t *out_min_free);
int pn_tt_gdn_state_out(pn_tt_device *device, int slot);
/** The verify's slot map (pn_tt_gdn_verify_slots.h): COMPACT (rows - 1 slots) or legacy. */
void pn_tt_gdn_verify_compact_set(pn_tt_device *device, bool compact);
/** Snapshot slots a verify of `rows` rows needs under this device's map. */
uint32_t pn_tt_gdn_verify_slots(const pn_tt_device *device, uint32_t rows);
/** Per-row walk, row `row` of `rows`: route its state write (pn_tt_gdn_state_out) to the slot
 *  the map names, or in place. @return 0 or -1 */
int pn_tt_gdn_verify_row_out(pn_tt_device *device, uint32_t rows, uint32_t row);
/** Per-row walk, after row `row`'s descriptor: trade `layer`'s live part(s) with that slot
 *  (nothing when the row wrote in place). @return 0 or -1 */
int pn_tt_gdn_verify_row_trade(pn_tt_device *device, uint32_t layer, uint32_t rows,
                               uint32_t row, bool window, bool state);
/** The slot holding the state after row `last_kept` of the last `rows`-row verify. */
uint32_t pn_tt_gdn_verify_rollback_slot(const pn_tt_device *device, uint32_t rows,
                                        uint32_t last_kept);
int pn_tt_gdn_layer_swap(pn_tt_device *device, uint32_t layer, uint32_t slot);
int pn_tt_gdn_layer_swap_part(pn_tt_device *device, uint32_t layer, uint32_t slot, bool window,
                              bool state);
/** S2c: the row select for the next CONV / DELTA descriptors (0 = the decode row-0 walk). */
int pn_tt_gdn_set_row(pn_tt_device *device, uint32_t row);
/** S2c: queue split CONV/DELTA descriptors without awaiting (per-row gate blocks); await them
 *  all with pn_tt_gdn_await_deferred() before anything reads their output. */
int pn_tt_gdn_defer(pn_tt_device *device, bool defer);
int pn_tt_gdn_await_deferred(pn_tt_device *device);
/** S2d: every row's gate block (decay[rows][H], beta[rows][H]) in ONE write, then DELTA
 *  descriptors that use the staged blocks (no per-row upload). */
int pn_tt_gdn_gate_upload_rows(pn_tt_device *device, uint32_t rows, const float *decay,
                               const float *beta);
int pn_tt_gdn_delta_descriptor_only(pn_tt_device *device, uint32_t layer);

/** THE EXACT CARD GATES in decode: until cleared (`address` 0), DELTA descriptors
 *  (pn_tt_gdn_delta_descriptor_only, pn_tt_gdn_delta_rows_oncard) name the table block at
 *  `bank`:`address` (pn_tt_gdnc_table_decode_block) and the reader builds their gate blocks
 *  from the strips it names — nothing uploaded, nothing read back. */
void pn_tt_gdn_table_gate(pn_tt_device *device, uint32_t bank, uint32_t address);
/** MULTI-ROW DELTA: verify rows 0..rows-1 in one descriptor per member (see pn_tt_gdn.c). */
int pn_tt_gdn_delta_rows_oncard(pn_tt_device *device, uint32_t layer, uint32_t rows);

/** PM 6 MULTI-ROW CONV: the CONV of verify rows [0, rows) in ONE descriptor per member (strip
 *  mode): row r reads the window its one-row descriptor would have, writes its output to row r
 *  and its rolled window to the slot the verify map names (legacy: slot r, then the window
 *  swaps in row order — live = after the last row, slot r = before row r). -2 when refused before anything ran (the caller may
 *  take the per-row walk), -1 on a card failure. */
int pn_tt_gdn_conv_rows_oncard(pn_tt_device *device, uint32_t layer, uint32_t rows);

/**
 * The per-phase clean/replayed figures on stderr, ONCE, at release.
 *
 * NOT a duplicate of the `qwen35_tt: gdn phase` report line. That one runs under
 * `tokens % PN_TT_REPORT == 0`, so an arm that CORRUPTS the model's output — which every
 * data arm on this op does by construction — ends the run below the threshold and the
 * instrument that would have scored the arm does not print at all. Measured 2026-08-27.
 * Silent when the family ran nothing.
 */
void pn_tt_gdn_report_phases(const pn_tt_device *device);

/**
 * PHASE CONV: `conv_out = SiLU(conv1d(qkv))`, with the window rolled ON THE CARD.
 *
 * @param qkv       `conv_dim` floats, the fused projection
 * @param conv_out  `conv_dim` floats, written
 * @return          0, or -1 with the reason on stderr
 */
int pn_tt_gdn_conv(pn_tt_device *device, uint32_t layer, const float *qkv, float *conv_out);

/**
 * PHASE DELTA: the four-pass delta rule over every v-head, with the `[S,S]` state read,
 * updated and written back ON THE CARD.
 *
 * **`q` and `k` are PER K-HEAD and must NOT be GQA-repeated.** The reader does the repeat as
 * one modulo; staging the repeated 32-head form doubles them and is the 1.33x-worse row of
 * the split-point table.
 *
 * @param q      `k_heads * state_size` floats, L2-normalised, un-repeated
 * @param k      likewise
 * @param v      `v_heads * state_size` floats
 * @param decay  `v_heads` floats — `exp(g_h)`, already through the host `softplus` and `exp`
 *               because there is no SFPU `log`
 * @param beta   `v_heads` floats, already through the sigmoid
 * @param y      `v_heads * state_size` floats, written
 * @return       0, or -1 with the reason on stderr
 */
int pn_tt_gdn_delta(pn_tt_device *device, uint32_t layer, const float *q, const float *k,
                    const float *v, const float *decay, const float *beta, float *y);

/* ============================================================================
 * THE ON-CARD CHAIN — the same two descriptors with no host array anywhere
 * ============================================================================
 * The five calls below are the whole of A2. The two phases become
 *
 *     widen(ssm_qkv bf16)  ->  gdn_stream_in
 *     pn_tt_gdn_conv_oncard()                       -> the strip at pn_tt_gdn_stream_operand()
 *     an L2 head norm over its first pn_tt_gdn_key_tiles() tiles, IN PLACE
 *     pn_tt_gdn_delta_oncard()                      -> the strip at pn_tt_gdn_y_operand()
 *     an RMS head norm, narrow(), then silu(z) * on the elementwise residency
 *
 * and the only bytes that cross the link for the block are `alpha`/`beta` down and the
 * 256-byte gate block up. The activations do not: readback 1478.09 -> 326.09 KiB/token.
 *
 * The L2 norm runs IN PLACE over the conv output because PHASE CONV's output buffer IS PHASE
 * DELTA's input buffer — the two are the same 256-tile row-0 strip and q, k and v are already
 * where `ftg_delta_stream_base()` reads them. In place is safe here for a pipeline reason and
 * not a hopeful one: the reduce reader prefetches ONE row block, so it has read block `b`
 * before the writer can have written block `b`. There is a discriminating arm for it.
 * ============================================================================ */

/** PHASE CONV over whatever is in the stream-in buffer, writing the stream buffer. */
int pn_tt_gdn_conv_oncard(pn_tt_device *device, uint32_t layer);

/** PHASE DELTA over whatever is in the stream buffer, writing `y`. Uploads the 256-byte gate
 *  block and nothing else. */
int pn_tt_gdn_delta_oncard(pn_tt_device *device, uint32_t layer, const float *decay,
                           const float *beta);

/** PM 7 ITEM 6: where the next DELTA reads its gate — `rows` rows (1 = the single gate, or a
 *  deferred row's block) — so the card can write it there (gdngate FLAT). @return 0 or -1. */
int pn_tt_gdn_gate_target(pn_tt_device *device, uint32_t rows, uint32_t *bank,
                          uint32_t *address, uint32_t *row_stride);

/** `decay[v_heads]` then `beta[v_heads]`, flat fp32, into the gate buffer. Public because the
 *  on-card path uploads it without staging anything else. */
int pn_tt_gdn_gate_upload(pn_tt_device *device, const float *decay, const float *beta);

/** PHASE CONV's INPUT — where the widened `ssm_qkv` strip has to land. */
ft_tt_op_operand pn_tt_gdn_stream_in_operand(const pn_tt_device *device);

/** PHASE CONV's OUTPUT and PHASE DELTA's INPUT, which are ONE buffer. */
ft_tt_op_operand pn_tt_gdn_stream_operand(const pn_tt_device *device);

/** PHASE DELTA's OUTPUT. */
ft_tt_op_operand pn_tt_gdn_y_operand(const pn_tt_device *device);

/**
 * Read logical row 0 of `tiles` tiles of the stream buffer, starting at `first_tile`.
 *
 * A PROBE, and the only reader of it: on the shipped on-card path nothing downloads this
 * buffer at all. It exists so the in-place q/k L2 norm can be scored against its own preimage
 * rather than asserted — see `qwen35moe_gdn_strip_tt.c`.
 *
 * @param out  `tiles * 32` floats
 * @return     0, or -1
 */
int pn_tt_gdn_stream_read(pn_tt_device *device, uint32_t first_tile, uint32_t tiles,
                          float *out);

/* ============================================================================
 * THE PHASE-ISOLATION SEAM — for attributing a chunkwise divergence, and nothing else
 * ============================================================================
 * The chunkwise family (`pn_tt_gdnc.h`) writes rows 0..C-1 of these same three buffers; the
 * decode family writes row 0 and ZERO ROWS 1..31, because it computes whole tiles over an
 * input whose rows 1..31 are padding. So a CHUNK descriptor's answer cannot simply be handed
 * to the next DECODE descriptor and back: each decode descriptor would wipe the other rows.
 *
 * The two calls below are what lets one phase run per position over a chunk's rows while the
 * other stays chunkwise: shuttle row `t` to row 0, run the decode descriptor, take row 0
 * away again, and restore the decode family's zero-rows invariant in between. Every move is
 * an exact fp32 copy over the link, so a hybrid arm's divergence from the sequential path is
 * attributable to the ONE chunk kernel it still runs.
 *
 * SLOW ON PURPOSE — this is a quality instrument, not a speed path, and it is only ever
 * reached from `PN_TT_PREFILL_GDNC_PHASE`.
 * ============================================================================ */

/** Which of the family's three row-major fp32 strips a row move addresses. */
typedef enum {
    PN_TT_GDN_STRIP_STREAM_IN = 0, /**< PHASE CONV's input */
    PN_TT_GDN_STRIP_STREAM_OUT,    /**< PHASE CONV's output, PHASE DELTA's input */
    PN_TT_GDN_STRIP_Y              /**< PHASE DELTA's output */
} pn_tt_gdn_strip_id;

/**
 * Move logical row `row` of the first `tiles` tiles of one strip, in either direction.
 *
 * @param values  `tiles * 32` floats
 * @return        0, or -1 on an unarmed family, a bad strip or a row past 31
 *
 * Thread-safe: no    Reentrant: no    Signal-safe: no
 */
int pn_tt_gdn_strip_row_move(pn_tt_device *device, pn_tt_gdn_strip_id strip, uint32_t row,
                             uint32_t tiles, float *values, bool reading);

/** Zero one whole strip — the decode family's precondition, which a chunk descriptor's
 *  whole-tile write is exactly what breaks. */
int pn_tt_gdn_strip_zero(pn_tt_device *device, pn_tt_gdn_strip_id strip);

/** `conv_dim/32` — the whole stream. */
uint32_t pn_tt_gdn_stream_tiles(const pn_tt_device *device);

/** `2 * k_heads * state_size/32` — q then k, the prefix the L2 norm covers. */
uint32_t pn_tt_gdn_key_tiles(const pn_tt_device *device);

/** The GDN's V-heads: the live columns of the alpha/beta strips. 0 before the arm. */
uint32_t pn_tt_gdn_v_heads(const pn_tt_device *device);

/** `v_heads * state_size/32` — v, and the same count as `y`. */
uint32_t pn_tt_gdn_value_tiles(const pn_tt_device *device);

#endif /* USE_TT */
#endif /* PN_TT_GDN_H */
