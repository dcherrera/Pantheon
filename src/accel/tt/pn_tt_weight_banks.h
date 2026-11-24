/* SPDX-License-Identifier: AGPL-3.0-or-later */
/* Copyright (C) 2026 David (Corey) Herrera */
/**
 * @file pn_tt_weight_banks.h
 * @brief WHERE a sharded matmul weight's member panels live in DRAM (tt-place lane), and
 *        which weights a timing probe touches.
 *
 * `PN_TT_WEIGHT_BANKS`:
 *   unset / `rotate`  the shipped placement: member m's panel of every tensor in bank
 *                     `(cursor + m) % 8`, the cursor advancing one bank per tensor, doors
 *                     from the stripe chooser
 *   `near`            member m's panels of EVERY tensor in the bank and door nearest member
 *                     m's core on the weight reader's NoC (0), two members a bank
 *                     (ft_tt_dram_nearest.h)
 *   `nearbank`        the nearest banks, doors from the stripe chooser
 *   `far`, `fixed`    CONTROLS: the farthest balanced map; the rotation's cursor-0 banks for
 *                     every tensor (same bank set, no per-tensor rotation) — they separate
 *                     NoC distance from "the same banks every projection"
 *
 * The same bytes are uploaded either way, so every mode is bit-exact by construction: only
 * the (bank, address, door) triple in the descriptor moves.
 *
 * `PN_TT_MATMUL_PROBE_ONLY=<classes>`: restricts `PN_TT_MATMUL_PROBE` (timing only) to the
 * weights whose CLASS is listed — the name between `blk.<n>.` and `.weight` (`ffn_gate`,
 * `attn_qkv`, ...), or the whole name less `.weight` otherwise (`output`). Unset: every
 * weight, which is what the probe always did.
 *
 * @status NEW 2026-10-22
 * @issues None
 * @todo None
 */
#ifndef PN_TT_WEIGHT_BANKS_H
#define PN_TT_WEIGHT_BANKS_H

#include <stdbool.h>
#include <stdint.h>

#include "pn_tt_device_internal.h"
#include "tt/ft_tt_dram_stripe.h"

#define PN_TT_WEIGHT_BANKS_ENV "PN_TT_WEIGHT_BANKS"
#define PN_TT_MATMUL_PROBE_ONLY_ENV "PN_TT_MATMUL_PROBE_ONLY"

/**
 * Banks and doors for `members` panels of one tensor under the configured mode.
 *
 * @return 1 when a non-rotation mode filled `banks`/`doors`; 0 when the rotation applies
 *         (the caller's own arithmetic, unchanged); -1 on a refusal (reason on stderr)
 */
int pn_tt_weight_banks_choose(pn_tt_device *device, uint32_t members, uint32_t *banks,
                              ft_tt_dram_endpoint *doors);

/** Does the timing probe apply to the weight called `name`? (see the file header) */
uint8_t pn_tt_weight_probe_selected(const char *name);

/** The weight's CLASS: `blk.12.ffn_gate.weight` -> `ffn_gate`, `output.weight` -> `output`. */
void pn_tt_weight_class(const char *name, char *out, uint32_t out_size);

#endif /* PN_TT_WEIGHT_BANKS_H */
