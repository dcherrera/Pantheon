/* SPDX-License-Identifier: AGPL-3.0-or-later */
/* Copyright (C) 2026 David (Corey) Herrera */
/**
 * @file qwen35moe_checkpoint_band.h
 * @brief Where a multi-turn prefill checkpoint is taken: the position arithmetic of
 *        qwen35moe_prefill_checkpoint_tt.c, host-only so it is unit-testable without a card.
 *
 * The checkpoint is the carried GDN state at a 32-row band boundary of a batched prefill
 * (see qwen35moe_prefill_checkpoint_tt.c for why only a band boundary is bit-exact). A later
 * turn can resume from it only when its prompt starts with every token below it.
 *
 * THE CEILING. A chat prompt ends with tokens the template FORCED after the assistant header
 * — the generation prompt, "<think>\n" on Ornith. When the client resends that turn as
 * history the template renders it without them, so the next prompt agrees with this one only
 * up to the header's end. A checkpoint above that point can never be resumed: the next turn
 * re-prefills the whole conversation (35k tokens: 79 s instead of ~2 s). The ceiling is the
 * header's end; the checkpoint is the last band boundary at or below it.
 *
 * @status Complete.
 * @issues None
 * @todo None
 */

#ifndef PN_QWEN35MOE_CHECKPOINT_BAND_H
#define PN_QWEN35MOE_CHECKPOINT_BAND_H

/** Rows of a chunk-mixer band: the only positions a checkpoint can sit at. */
#define PN_QWEN35_CKPT_BAND_ROWS 32

/**
 * The position to checkpoint in a batched prefill that runs rows [base, batch_end).
 *
 * @param base       where the passes start (0, or a restored checkpoint) — a band multiple
 * @param batch_end  one past the last batched row
 * @param ceiling    no checkpoint above this position (the end of the history the next turn
 *                   will resend); <= 0 means none
 * @return the last band boundary in (base, min(batch_end, ceiling)], or 0 when there is none
 *         (the checkpoint at `base`, if any, still stands)
 *
 * Thread-safe: yes (pure)
 */
int pn_qwen35_ckpt_band_at(int base, int batch_end, int ceiling);

#endif /* PN_QWEN35MOE_CHECKPOINT_BAND_H */
