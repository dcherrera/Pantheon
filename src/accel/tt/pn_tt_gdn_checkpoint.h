/* SPDX-License-Identifier: AGPL-3.0-or-later */
/* Copyright (C) 2026 David (Corey) Herrera */
/**
 * @file pn_tt_gdn_checkpoint.h
 * @brief The multi-turn checkpoint of the carried GDN state: every SSM layer's conv window and
 *        recurrent state copied, ON THE CARD, to a second region set and back.
 *
 * A hybrid model cannot rewind by position: the recurrent state is a fold of the whole prefix.
 * The model layer takes a checkpoint at a prefill PASS boundary (`qwen35moe_prefill_tt.c`) and
 * a later turn restores it and batch-prefills from there with the same passes a cold run would
 * take, so its output is the cold run's. The attention K/V needs no copy: rows below the
 * checkpoint are never rewritten while it is valid (the model layer enforces that).
 *
 * The copies are the puller's COPY op (`resident_puller.c` kOpCopy): ~120 MB card side instead
 * of 22 s over a 5.3 MB/s readback. The checkpoint stays in card DRAM; nothing is written to
 * the host's disk (bank-never-solos: a blob for the user's machine is a separate export).
 *
 * status: per-layer capture, restore and a sampled byte verify.
 * issues: one checkpoint a device (one sequence on the card at a time).
 * todo:   none.
 */
#ifndef PN_TT_GDN_CHECKPOINT_H
#define PN_TT_GDN_CHECKPOINT_H

#ifdef USE_TT

#include "pn_tt_device.h"

/**
 * Copy SSM layer `layer`'s live window + state into its checkpoint twin (the set is allocated
 * on first use). Called between two 32-row bands of a prefill pass, when that layer's state is
 * the state at the checkpoint position. Flushes the card first. RED `ckpt_copy_short` leaves
 * each buffer's last 4 KiB uncopied.
 *
 * @return 0, or -1 (the checkpoint must then be treated as absent)
 */
int pn_tt_gdn_checkpoint_capture_layer(pn_tt_device *device, uint32_t layer);

/**
 * Copy every layer's checkpoint twin back over the live window + state, and restart the
 * dispatch model's learning (positions move backwards). RED `ckpt_restore_skip` reports
 * success and copies nothing: the live state stays the turn's end and the output must DIFFER.
 *
 * @return 0, or -1 (nothing captured, or a copy failed: the live state is then UNDEFINED and
 *         the caller must start the sequence over)
 */
int pn_tt_gdn_checkpoint_restore(pn_tt_device *device);

/**
 * `PN_TT_CKPT_VERIFY=1`: compare the first and last 64 KiB of layer `layer`'s live window and
 * state with its checkpoint twin over the link. Call right after that layer's capture, or after
 * a restore. The gate of the copy is bytes, not an argmax.
 *
 * @return the 32-bit words that differ (0 when the knob is off), or -1 when a read failed;
 *         `*words` grows by the words compared
 */
long long pn_tt_gdn_checkpoint_verify_layer(pn_tt_device *device, uint32_t layer,
                                            unsigned long long *words);

/** Is `PN_TT_CKPT_VERIFY` set? */
bool pn_tt_gdn_checkpoint_verify_wanted(void);

#endif /* USE_TT */
#endif /* PN_TT_GDN_CHECKPOINT_H */
