/* SPDX-License-Identifier: AGPL-3.0-or-later */
/* Copyright (C) 2026 David (Corey) Herrera */
/**
 * @file pn_tt_unmirrored_probe.h
 * @brief PRICE PROBE (2026-10-26, timing only): what a multi-row projection whose input is
 *        NOT mirrored (the out-projections: `ssm_gated`, attention `gated`) would cost if it
 *        were — before building the mirrors.
 *
 * `PN_TT_UNMIRRORED_PROBE=1`: such a projection reads A from 8 banks (the strip's own bank
 * and the next 7, same address) and so takes the prefill grid, as a mirrored input would.
 * `=2`: the same spread A, but it stays on the decode group (the mirror's effect alone).
 * Every bank but the first holds OTHER bytes there: the text DIFFERS by design. Read cost is
 * the same whatever the bytes are, so the timing is the mirrored projection's, minus the
 * producer's fan-out writes (priced separately).
 *
 * @status Probe; off by default.
 * @issues None
 * @todo None
 */
#ifndef PN_TT_UNMIRRORED_PROBE_H
#define PN_TT_UNMIRRORED_PROBE_H

#ifdef USE_TT

#include <stdbool.h>
#include <stdint.h>

#include "pn_tt_device.h"

/** 0 off, 1 spread A + grid, 2 spread A on the decode group. Read once. */
uint32_t pn_tt_unmirrored_probe_mode(void);

/**
 * When the probe is on and `input` is a multi-row, unmirrored strip, fill `out` with a copy
 * of it whose A reads spread over `bank_count` banks at the same address, and return true.
 */
bool pn_tt_unmirrored_probe_view(const pn_tt_activation *input, uint32_t tile_rows,
                                 uint32_t bank_count, pn_tt_activation *out);

#endif /* USE_TT */
#endif /* PN_TT_UNMIRRORED_PROBE_H */
