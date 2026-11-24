/* SPDX-License-Identifier: AGPL-3.0-or-later */
/* Copyright (C) 2026 David (Corey) Herrera */
/**
 * @file pn_tt_late_mirrors.h
 * @brief Mirror an already-allocated activation strip AFTER every load-time allocation, from
 *        the DRAM that is left, keeping a reserve in every bank.
 *
 * `pn_tt_activation_alloc_replicated()` takes its mirrors at the strip's own allocation, early
 * in the arm, in fixed banks — so a strip mirrored there moves every later allocation and can
 * be the one that makes a bank overflow. Qwen3.8-27B with MTP at ctx 30,720 leaves its fullest
 * bank 3 MiB free after the arm (p150, `DRAM after arm`). The decode out-projection
 * inputs (`ssm_gated`, attention `gated`: 192 tiles x 2 bands = 768 KiB a copy) are mirrored
 * here, last, into whichever banks still hold the copy plus the reserve; a bank that does not
 * is skipped and the strip simply gets fewer mirrors. Nothing allocated before moves.
 *
 * The strip and every new mirror are zeroed: the caller asserts the strip is per-layer scratch
 * (its producer rewrites row 0 before any read; rows 1-31 stay zero), so zero is its content.
 *
 * Keeping the mirrors current is the producer's job (pn_tt_replica_map.h); the caller must
 * register the strip (`pn_tt_replica_map_set_decode_ranged`) and republish the map.
 *
 * @status L2. Card A/B staged, not run.
 * @issues None
 * @todo None
 */
#ifndef PN_TT_LATE_MIRRORS_H
#define PN_TT_LATE_MIRRORS_H

#ifdef USE_TT

#include <stdint.h>

#include "pn_tt_device.h"

/**
 * Give `strip` up to `replicas` copies (primary included) in distinct banks with at least
 * `reserve_bytes` free after the copy. Banks are tried from the strip's own bank + 1 upward.
 *
 * @param strip          an allocated, UNMIRRORED strip (replica_count 1)
 * @param replicas       copies wanted, primary included; clamped to the bank count and to
 *                       PN_TT_ACTIVATION_MAX_REPLICAS
 * @param reserve_bytes  free bytes every used bank must still hold afterwards
 * @return the strip's replica count afterwards (1 = no bank had room, nothing changed), or -1
 *         on bad arguments, an already-mirrored strip, or a failed allocation / zeroing (the
 *         strip is then left unmirrored)
 *
 * Thread-safe: no    Reentrant: no    Signal-safe: no
 */
int pn_tt_activation_mirror_late(pn_tt_device *device, pn_tt_activation *strip,
                                 uint32_t replicas, uint64_t reserve_bytes);

#endif /* USE_TT */
#endif /* PN_TT_LATE_MIRRORS_H */
