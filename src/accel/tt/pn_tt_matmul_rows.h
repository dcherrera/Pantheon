/* SPDX-License-Identifier: AGPL-3.0-or-later */
/* Copyright (C) 2026 David (Corey) Herrera */
/**
 * @file pn_tt_matmul_rows.h
 * @brief Pure decisions for the resident decode matmul's A side: the A-reuse ring and whether
 *        a descriptor's A read is skipped after its first column pass. No device, so the host
 *        test (tests/test_tt_matmul_rows.c) runs the code the card path runs.
 *
 * BYTES A-REUSE: the reader skips a pass's A read when CB_IN0's ring is EXACTLY the
 * descriptor's Kt and MB = 1 (resident_matmul_reader.c). The ring is one size per residency, so
 * it is a model preference: the hidden Kt, the K of most of the model's bytes.
 *
 * @status None
 * @issues None
 * @todo None
 */

#ifndef PN_TT_MATMUL_ROWS_H
#define PN_TT_MATMUL_ROWS_H

#include <stdbool.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

/** Default ring when no model states one: Ornith's hidden Kt (4096 / 32). */
#define PN_TT_A_REUSE_DEFAULT_TILES 128u

/**
 * The A-reuse ring in pages.
 * @param env_set    1 when `PN_TT_A_REUSE` is set (it overrides; 0 = off)
 * @param env_tiles  its value
 * @param preferred  the model's preference (0 = none)
 */
uint32_t pn_tt_a_reuse_ring(int env_set, uint32_t env_tiles, uint32_t preferred);

/** The kernel's condition: the ring is exactly Kt and the descriptor walks one row block. */
bool pn_tt_a_reuse_fires(uint32_t ring, uint32_t tile_inner, uint32_t row_block);

#ifdef __cplusplus
}
#endif

#endif /* PN_TT_MATMUL_ROWS_H */
