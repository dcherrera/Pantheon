/* SPDX-License-Identifier: AGPL-3.0-or-later */
/* Copyright (C) 2026 David (Corey) Herrera */
/**
 * @file pn_tt_dispatch.h
 * @brief STEP 3 (A) — on-card dispatch: the RECORDER. Captures every host<->card access a
 *        decode token makes, at the transport, classified by address.
 *
 * @status A1: observe mode (`PN_TT_DISPATCH_RECORD=<token>`): wraps the runtime's three
 *         accessors for one decode token and prints what the token is made of — queue writes
 *         (parameter blocks, ring slots, head bumps), completion polls, and every OTHER access
 *         (the segment boundaries a puller has to hand back to the host).
 *         A1b (`PN_TT_DISPATCH_SHADOW=<forward>`): learn a model from two tokens and predict
 *         every later token's queue writes and awaited counts, compared byte for byte.
 * @issues None
 * @todo A2: the puller's dry run; A3: live replay.
 */
#ifndef PN_TT_DISPATCH_H
#define PN_TT_DISPATCH_H

#ifdef USE_TT

#include <stdbool.h>
#include <stdint.h>

#include "pn_tt_device.h"

/** Call at the start of a decode forward (before the embedding upload); `forward` counts
 *  forwards from 1. Arms the recorder when `forward` is the one asked for. */
void pn_tt_dispatch_token_begin(pn_tt_device *device, uint64_t forward);

/** Call after the head (top-k read back). Disarms and, after the recorded token, reports. */
void pn_tt_dispatch_token_end(pn_tt_device *device, uint64_t forward);

/** A new sequence (the carried state was zeroed): the model is re-learned on its forwards. */
void pn_tt_dispatch_sequence_reset(void);
/** No stream, no replay and no recording is in progress: the host may issue outside a token
 *  (the keepalive's NOPs). */
bool pn_tt_dispatch_quiet(void);

/** STEP 4 S2 (verify data movers on the puller, armed on demand):
 *  row `src_row` of every tile of `src` -> row `dst_row` of `dst` (all of dst's mirrors). */
int pn_tt_row_copy(pn_tt_device *device, const pn_tt_activation *src, uint32_t src_row,
                   const pn_tt_activation *dst, uint32_t dst_row);
/** FILL on the puller (`resident_puller.c` kOpFill): every 32-bit word of DRAM
 *  [address, address + bytes) of `bank` becomes `value`, card side. address and bytes 64 B
 *  multiples. -1 (nothing written) when the puller cannot arm, mid-stream, or misaligned. */
int pn_tt_fill_dram_oncard(pn_tt_device *device, uint32_t bank, uint32_t address,
                           size_t bytes, uint32_t value);
/** COPY on the puller (`resident_puller.c` kOpCopy): DRAM [src_address, + bytes) of
 *  `src_bank` -> [dst_address, + bytes) of `dst_bank`, card side (the multi-turn checkpoint of
 *  the carried GDN state). Addresses and bytes 64 B multiples. Flushes first. -1 when the
 *  puller cannot arm, mid-stream, misaligned, or the descriptor failed (a puller program
 *  without the opcode refuses it: the copy is then not done, and the caller must not trust
 *  the destination). */
int pn_tt_copy_dram_oncard(pn_tt_device *device, uint32_t src_bank, uint32_t src_address,
                           uint32_t dst_bank, uint32_t dst_address, size_t bytes);
/** Cumulative S2 row-copy count and host wall time. */
void pn_tt_move_stats(uint64_t *rows, double *row_ms);

/** S2d: STREAMED segments around a speculative verify / MTP drafting (pn_tt_dispatch.c).
 *  `PN_TT_SPEC_STREAM=0` off. begin returns false when not streaming; end returns false if a
 *  segment failed on the card (the caller must then treat its work as failed). */
bool pn_tt_dispatch_stream_begin(pn_tt_device *device);
bool pn_tt_dispatch_stream_end(void);
void pn_tt_dispatch_stream_report(double cycles);
/** PM 6: launch what the stream has captured so far WITHOUT waiting (the card starts on it
 *  while the host waits on something else). true when nothing was streaming or the launch
 *  went; false if the stream has failed. */
bool pn_tt_dispatch_stream_kick(void);

/** A1b: print the shadow's tallies (`PN_TT_DISPATCH_SHADOW=<first model forward>`). */
void pn_tt_dispatch_report(void);

#endif /* USE_TT */
#endif /* PN_TT_DISPATCH_H */
