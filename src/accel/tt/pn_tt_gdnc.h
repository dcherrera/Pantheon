/* SPDX-License-Identifier: AGPL-3.0-or-later */
/* Copyright (C) 2026 David (Corey) Herrera */
/**
 * @file pn_tt_gdnc.h
 * @brief The CHUNKWISE Gated-Delta-Net block on the card: the same two phases the decode
 *        family runs, over C <= 32 POSITIONS AT ONCE, on `foundry/tt/ft_tt_op_gdnc.h`.
 *
 * ---------------------------------------------------------------------------
 * IT BORROWS THE DECODE FAMILY'S CARRIED STATE, BYTE FOR BYTE
 * ---------------------------------------------------------------------------
 * This is the single most important property of this file and it was designed in rather than
 * discovered: the rolling conv window and the `[v_heads, S, S]` recurrent state the chunkwise
 * kernels read and write are **the buffers `pn_tt_gdn.c` allocated**, at the same addresses,
 * in the same layout (`ftgc_plane_tile()` is `ftg_conv_plane_tile()`, `ftgc_state_tile()` is
 * `ftg_state_tile()`). So a prompt that prefills in chunks and then decodes one token at a
 * time is ONE sequence with no transcription at the seam, and the `1..taps-2` tail a chunk
 * cannot carry is run through the decode family on the same bytes.
 *
 * The consequence for arming: **this family REQUIRES the `gdn` family to be armed** and
 * refuses otherwise. It owns no carried state of its own.
 *
 * ---------------------------------------------------------------------------
 * WHAT IT DOES OWN, AND WHY EACH IS A SECOND BUFFER
 * ---------------------------------------------------------------------------
 *   the CONV constant   PER LAYER. `taps` ROW-REPLICATED tap-major planes plus the `2*taps`
 *                       band tiles riding at the end of them. The decode family's weight
 *                       carries `w[c][tap]` in ROW 0 with zeros below; a chunk needs it at
 *                       every row, because the tap coefficient is constant across positions
 *                       and the broadcast surface is BROADCAST_COL only. Same tile count,
 *                       same addresses, different `valid_rows` — and a SEPARATE buffer,
 *                       because the shipped decode path's byte-identity is a hard gate and
 *                       is not worth re-deriving through a shared operand.
 *                       4.03 MiB a layer at Ornith, 96.8 MiB over 24.
 *   the DELTA constant  ONE for the model: the four program constants (`I`, `2I`, the
 *                       strictly-lower mask, the `scale` column) and one decay-ratio tile per
 *                       v-head. 147 KiB. The decay tiles are rebuilt every layer of every
 *                       chunk; the four constants are written once.
 *   the GATE block      ONE for the model, `v_heads * 160` floats, and it MUST sit in the
 *                       DELTA constant's bank — it shares that operand's packed NoC
 *                       coordinate word and nothing on the device can catch a violation.
 *
 * ---------------------------------------------------------------------------
 * THE DECAY-RATIO MATRIX IS BUILT ON THE HOST, DELIBERATELY
 * ---------------------------------------------------------------------------
 * `ft_tt_op_gdnc_build_decay()` is a product recurrence: every entry in (0,1], no cumulative
 * reciprocal, no logarithm. Both alternatives were modelled through this card's Tf32
 * truncating unpack and both are worse — an on-card doubling construction costs the whole
 * precision advantage of going chunkwise (y 8.6e-03 -> 1.7e-02), and `exp(log P_t - log P_s)`
 * is safe against OVERFLOW and not against the UNPACK, because cumulative logs reach -50 at
 * Ornith's decays and a Tf32 truncation of -50 is 2.4% error in D before the exp is reached.
 * **It costs 128 KiB a layer a chunk on the link, which is 0.33 ms a position at 32 rows, and
 * that is the price of the accuracy. Do not move it to the card.**
 *
 * ---------------------------------------------------------------------------
 * WHAT THE CALLER OWES
 * ---------------------------------------------------------------------------
 *   1. `chunk_rows >= conv_taps - 1` for PHASE CONV. Below that the chunk cannot REPLACE the
 *      window, only shift it, and both the host encoder and the device refuse. The caller
 *      routes a shorter tail through `pn_tt_gdn_conv_oncard()` / `pn_tt_gdn_delta_oncard()`,
 *      which is the same arithmetic on the same bytes. `pn_tt_gdnc_min_chunk_rows()` is that
 *      floor. PHASE DELTA has no floor.
 *   2. **ROWS `chunk_rows..31` OF THE STREAM MUST BE ZERO.** They are computed and discarded,
 *      and `SiLU(0) = 0` cannot surprise anyone — a stale value there would propagate through
 *      arithmetic nobody inspects. In `qwen35moe_prefill_tt.c` this holds for free and by
 *      construction: `pn_tt_activation_upload_rows()` tilizes at `valid_rows = rows`, so the
 *      pass's `hidden` strip is zero above the chunk, an RMS norm of a zero row is zero, and
 *      an `Mt=1` matvec of a zero row is zero — so every projection is zero above the chunk.
 *
 * @status NEW, default OFF behind `PN_TT_PREFILL_GDNC`. The three kernels are green on
 *         silicon (`foundry/tt/tests/test_phase6_gdnc.c`): PHASE CONV 681.4 us a descriptor
 *         flat in C, PHASE DELTA 435.5 us at 32 v-heads, both parity-checked over 103
 *         positions as 32+32+32+7.
 * @issues The `gdnc` residency is ONE core and PHASE DELTA is not head-split here. The
 *         decode family measured its DELTA embarrassingly parallel in the head and the
 *         chunkwise one fits `13.8 us + N x 13.18`, so the split is available and unbuilt.
 * @todo None
 */

#ifndef PN_TT_GDNC_H
#define PN_TT_GDNC_H

#ifdef USE_TT

#include <stdbool.h>
#include <stdint.h>

#include "pn_tt_device.h"
#include "pn_tt_prefill_reserve.h"

/**
 * Arm the chunkwise residency and allocate the three buffers above.
 *
 * Reads the shape off the ALREADY-ARMED `gdn` family, so the two cannot disagree about
 * `conv_dim`, the tap count or the head counts — which is what makes sharing the window and
 * the recurrent state safe.
 *
 * Never fatal on its own: a card without the `gdnc` program, a core already held, or DRAM
 * that will not take the constants all leave the family unarmed and the caller on the
 * per-position path.
 *
 * @return 0 (even when nothing armed), -1 on a bad argument or a failed allocation
 *
 * Thread-safe: no    Reentrant: no    Signal-safe: no
 */
int pn_tt_device_arm_gdnc(pn_tt_device *device);

/**
 * The conv-constant copy count the model's fit picked (lever E, qwen35moe_tt_fit_plan.h), read
 * when the family arms; 0 = the default 8. `PN_TT_GDNC_CONST_REPLICAS` still wins. Call before
 * pn_tt_device_arm_gdnc.
 *
 * Thread-safe: no    Reentrant: no    Signal-safe: yes
 */
void pn_tt_gdnc_request_replicas(pn_tt_device *device, uint32_t replicas);

/** Did the chunkwise family actually arm for this run? */
bool pn_tt_gdnc_is_armed(const pn_tt_device *device);

/** The host's 1/sqrt(S) word, as the COMPACT block's header carries it. */
uint32_t pn_tt_gdnc_compact_scale_bits(const pn_tt_device *device);

/** One exact-gate table's slot cap in bytes (src/models/qwen35moe_gate_tables.h
 *  PN_QWEN35_GATE_TABLE_SLOT_WORDS x 4). */
#define PN_TT_GDNC_TABLE_SLOT_BYTES 40960u
/** The most banks a layer's decay tables stripe over (ftgc_table.h kFtgtMaxStripes). */
#define PN_TT_GDNC_TABLE_STRIPES 8u

/**
 * THE EXACT CARD GATES: upload the host's lookup tables (src/models/qwen35moe_gate_tables.h) —
 * `decay` is layers x heads tables at `head_stride_words` apart, `layer_words[l]` the widest of
 * layer l's (its heads are packed at that, 64 B aligned) — STRIPED: layer l's head h in bank
 * `stripe_bank[l * PN_TT_GDNC_TABLE_STRIPES + h % stripes]` (distinct banks a layer, the
 * placement pn_qwen35_tt_fit_table_stripes chose), with one beta copy in every bank that holds
 * a stripe; then allocate the per-layer table blocks in DELTA constant slot 0's bank.
 * @return 0, or -1 (the host gates stay).
 */
int pn_tt_gdnc_table_arm(pn_tt_device *device, uint32_t layers, uint32_t heads,
                         const uint32_t *decay, size_t head_stride_words,
                         const uint32_t *layer_words, const uint32_t *beta, uint32_t beta_words,
                         uint32_t stripes, const uint32_t *stripe_bank);

/** Did the exact card gates arm? */
bool pn_tt_gdnc_table_ready(const pn_tt_device *device);

/** Layer `layer`'s table block (ftgc_table.h) for the alpha/beta strips, from their band-0
 *  addresses whatever band they are seeked to. @return 0, or -1. */
int pn_tt_gdnc_table_block(const pn_tt_device *device, uint32_t layer,
                           const pn_tt_activation *alpha, const pn_tt_activation *beta,
                           uint32_t out[48]);

/** Put layer `layer`'s block on the card if it changed. Before the layer's band loop: nothing
 *  queued may be reading it. */
int pn_tt_gdnc_table_bind(pn_tt_device *device, uint32_t layer, const pn_tt_activation *alpha,
                          const pn_tt_activation *beta);

/** The most decode table blocks a layer keeps (one per distinct strip address pair). */
#define PN_TT_GDNC_TABLE_DECODE_BLOCKS 8u

/** DECODE: layer `layer`'s table block naming `alpha`/`beta` AS THEY STAND (the decode rows
 *  are rows of their current band), for pn_tt_gdn_table_gate. Written once per address pair
 *  and never rewritten, so a queued decode DELTA never sees its block change.
 *  @return 0 with `bank`/`address`, or -1 (no tables, a misaligned strip, the cache full: the
 *  caller keeps the host gates). */
int pn_tt_gdnc_table_decode_block(pn_tt_device *device, uint32_t layer,
                                  const pn_tt_activation *alpha, const pn_tt_activation *beta,
                                  uint32_t *bank, uint32_t *address);

/** One DELTA descriptor whose gates the reader looks up from band `band` of the bound strips. */
int pn_tt_gdnc_delta_oncard_table(pn_tt_device *device, uint32_t layer, uint32_t first_row,
                                  uint32_t chunk_rows, uint32_t band);

/** PM 7 ITEM 5: one DELTA descriptor whose COMPACT gate block the card already computed (the
 *  GDN GATE family) at `gate_address`, in the DELTA constant bank. Nothing is staged. */
int pn_tt_gdnc_delta_oncard_card_gate(pn_tt_device *device, uint32_t layer, uint32_t first_row,
                                      uint32_t chunk_rows, uint32_t gate_address);

/** The shortest chunk PHASE CONV will run — `conv_taps - 1`. Shorter tails go through the
 *  decode family. 0 when the family is not armed. */
uint32_t pn_tt_gdnc_min_chunk_rows(const pn_tt_device *device);

/**
 * TRANSPOSE one layer's conv1d weight into ROW-REPLICATED tap-major planes and upload it,
 * then write the `2*taps` band tiles behind them. Once, at load.
 *
 * `weight` is the oracle's own `layer->ssm_conv1d`: ROW-MAJOR `[conv_dim, taps]`. Same
 * transpose `pn_tt_gdn_conv_weight_upload()` does, with `valid_rows` 1 -> 32.
 *
 * @param layer   0..layer_count-1, the SSM-layer ordinal
 * @return        0, or -1 with the reason on stderr
 */
int pn_tt_gdnc_conv_weight_upload(pn_tt_device *device, uint32_t layer, const float *weight);

/**
 * PHASE CONV over `chunk_rows` positions of the GDN stream-in buffer, writing the stream
 * buffer and rolling the shared window ONCE, to the chunk's last `taps-1` positions.
 *
 * @param chunk_rows  `conv_taps-1` .. 32
 * @return 0, or -1 with the reason on stderr
 */
int pn_tt_gdnc_conv_oncard(pn_tt_device *device, uint32_t layer, uint32_t chunk_rows);

/**
 * PHASE DELTA over the same `chunk_rows` positions, reading q|k|v out of the stream buffer
 * and writing `y`, with the shared recurrent state advanced ONCE across the whole chunk.
 *
 * Uploads the per-head decay-ratio tiles (128 KiB) and the gate block (20 KiB) and nothing
 * else.
 *
 * @param decay  `chunk_rows * v_heads` floats, ROW-MAJOR by position — `exp(g)`, already
 *               through the host `softplus` and `exp` because there is no SFPU `log`
 * @param beta   likewise, already through the sigmoid
 * @return 0, or -1 with the reason on stderr
 */
int pn_tt_gdnc_delta_oncard(pn_tt_device *device, uint32_t layer, uint32_t chunk_rows,
                            const float *decay, const float *beta);

/**
 * PHASE DELTA over rows `[first_row, first_row + chunk_rows)` of the SAME 32-row strip — the
 * IN-PLACE SUB-CHUNK, and the reason a shorter chunk costs one extra descriptor instead of a
 * row shuttle.
 *
 * `first_row = 0` IS `pn_tt_gdnc_delta_oncard()` — same operands, same encoding, same staged
 * bytes — so a caller that never sub-chunks is unchanged by construction.
 *
 * WHY IT EXISTS. PHASE DELTA owns 100% of the chunkwise prefill's quality variance and the
 * chunk length is the whole of it: six arms against one oracle, C = 16 passing all six
 * criteria with the best figures of the ladder and C = 32 failing worst-document. Everything
 * else in the SSM layer batches at 32 and is unaffected by C, so the arm that keeps the speed
 * and takes the quality is ONE PHASE CONV over the whole pass, ONE L2 norm over the whole
 * pass, and `ceil(rows / C)` of these.
 *
 * @param first_row   a multiple of 16, with `first_row + chunk_rows <= 32`
 * @param decay,beta  `chunk_rows * v_heads` floats for THIS sub-chunk's rows, row-major by
 *                    position. The caller slices; nothing here reads outside the sub-chunk
 * @return 0, or -1 with the reason on stderr
 */
int pn_tt_gdnc_delta_oncard_at(pn_tt_device *device, uint32_t layer, uint32_t first_row,
                               uint32_t chunk_rows, const float *decay, const float *beta);

/**
 * Stage `(layer, first_row, chunk_rows)`'s decay tiles and gate block into that sub-chunk's
 * slot AHEAD of its descriptor, which then uses them instead of staging. Meant to run from the
 * overlap hook, i.e. while the card is busy with something that does not read that slot.
 *
 * @return 0, or -1 (nothing is marked staged)
 */
int pn_tt_gdnc_delta_prestage_at(pn_tt_device *device, uint32_t layer, uint32_t first_row,
                                 uint32_t chunk_rows, const float *decay, const float *beta);

/**
 * ONE-SHOT: run `fn(context)` inside the next gdnc descriptor, between its enqueue and its
 * await — host work hidden under card time. Its seconds are booked to the stage line, not the
 * phase. A failing `fn` fails that descriptor's call.
 */
void pn_tt_gdnc_set_overlap(pn_tt_device *device, int (*fn)(void *context), void *context);

/**
 * ONE-SHOT, a second slot: `fn(context)` runs in the next gdnc window AFTER the first slot's
 * hook. Its result is ignored (its owner redoes the work inline when it did not happen), and it
 * may set this slot again for the window after. NULL clears it.
 */
void pn_tt_gdnc_set_overlap_aux(pn_tt_device *device, int (*fn)(void *context), void *context);

/**
 * Prefill lane — PER-MEMBER STATE BANKS (`PN_TT_GDNC_STATE_BANKS`, default on, `=0` off). The 8 DELTA members all
 * read and write one layer's recurrent state (2 MiB) from ONE bank twice a band. With this on,
 * `enter` copies member m's heads into a buffer in bank (layer + 1 + m) % banks, every DELTA of
 * that layer reads and writes there, `sync` copies them back into `gdn_region[layer].state`
 * (the decode family's buffer: the tail forward, the checkpoint and decode read it), `exit`
 * syncs and drops the slices. Same bytes, other addresses: bit-exact. All three are card-side
 * copies and refuse inside a puller stream. No-ops (0) when off or unsplit.
 */
bool pn_tt_gdnc_state_split_enabled(void);
/** `bands` = the layer's band count; below `PN_TT_GDNC_STATE_BANKS_MIN_BANDS` (default 4) the
 *  copies cost more than they save (a 25-token prompt: +3 ms TTFT) and enter is a no-op. */
int pn_tt_gdnc_state_split_enter(pn_tt_device *device, uint32_t layer, uint32_t bands);
int pn_tt_gdnc_state_split_sync(pn_tt_device *device, uint32_t layer);
int pn_tt_gdnc_state_split_exit(pn_tt_device *device, uint32_t layer);

/**
 * The DRAM a batched prefill takes from this family after the load: the state-split slices
 * (when the split is enabled and usable) and PIPE 2's alternate streams. `prefill_need` adds
 * them to `need` per bank; `reserve_prefill` allocates them now, so the prefill finds them in
 * place (pn_tt_prefill_reserve.h). Nothing when the family is not armed.
 * @return reserve: 0, or -1 when a buffer did not allocate
 */
void pn_tt_gdnc_prefill_need(const pn_tt_device *device, pn_tt_bank_need *need);
int pn_tt_gdnc_reserve_prefill(pn_tt_device *device);

/**
 * Prefill lane PIPE 2 helpers. `set_parity` swaps the second stream_in / stream_out / y triple in
 * (1) or back (0); every operand accessor and every descriptor issued after it uses that
 * parity. `defer` makes the members' awaits record-only (pn_tt_residency defer_await) until
 * `settle` awaits them all. `serialize_next` marks the next DELTA descriptor(s) with the
 * reader's wait-for-the-previous-completion flag while on.
 */
int pn_tt_gdnc_set_parity(pn_tt_device *device, uint32_t parity);
void pn_tt_gdnc_defer(pn_tt_device *device, bool on);
int pn_tt_gdnc_settle(pn_tt_device *device);
void pn_tt_gdnc_serialize_next(pn_tt_device *device, bool on);

/** Retire the residency and free what arming allocated. Safe on a device that never armed. */
void pn_tt_gdnc_release(pn_tt_device *device);

#endif /* USE_TT */
#endif /* PN_TT_GDNC_H */
