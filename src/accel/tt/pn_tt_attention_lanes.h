/* SPDX-License-Identifier: AGPL-3.0-or-later */
/* Copyright (C) 2026 David (Corey) Herrera */
/**
 * @file pn_tt_attention_lanes.h
 * @brief DECODE ATTENTION LANES (2026-10-26): the per-KV-group flash-decode descriptors of
 *        one decode / verify attention call spread over the attention residency and its
 *        head-split members (cores 84 + 75..), instead of running in series on core 84.
 *
 * Each descriptor is byte for byte the one the single-core path sends, and every member runs
 * the same `attention` program in the same L1 shape, so each group's output tiles are the
 * same bytes; the groups write disjoint output tiles and read the KV cache only. What changes
 * is WHEN they run: all lanes are enqueued, then all are awaited, so the consumer of the
 * output is still ordered after every group.
 *
 * Decode attention is one descriptor per KV group and its cost grows with the context
 * (it walks the group's whole K/V), so at long context it is the cycle's largest term and
 * one core did all of it.
 *
 * @status Default on, 4 lanes (2026-10-26): `PN_TT_ATTN_DECODE_LANES=<n>`, `=1` (or 0) off,
 *         the single-core path. n is capped by the armed members (`PN_TT_ATTN_CORES`,
 *         default 16) and by the KV groups.
 * @issues None
 * @todo None
 */
#ifndef PN_TT_ATTENTION_LANES_H
#define PN_TT_ATTENTION_LANES_H

#ifdef USE_TT

#include <stdbool.h>
#include <stdint.h>

#include "pn_tt_device.h"
#include "tt/ft_tt_op_attention.h"

/** The most KV groups one lane-split call takes (Ornith has 4). */
#define PN_TT_ATTN_LANE_MAX_GROUPS 16u

/** Lanes the decode attention runs on: 1 = the single-core path. */
uint32_t pn_tt_attention_decode_lanes(const pn_tt_device *device);

/**
 * Run `groups` encoded flash-decode descriptors, group g on lane `g % lanes` (lane 0 the
 * `attention` residency, lane k member k), every lane enqueued before any is awaited. A lane
 * that refuses is re-armed and its descriptors resubmitted alone, after every other lane was
 * awaited (an unawaited lane would desynchronise its cursor).
 * @return 0, or -1 when a lane could not be re-armed
 */
int pn_tt_attention_issue_lanes(pn_tt_device *device,
                                uint32_t words[][FT_TT_OP_ATTENTION_WORDS], uint32_t groups);

/* Defined in pn_tt_attention.c: retire + re-arm the `attention` residency / member `index`. */
bool pn_tt_attention_main_rearm(pn_tt_device *device);
bool pn_tt_attention_member_rearm(pn_tt_device *device, uint32_t index);

#endif /* USE_TT */
#endif /* PN_TT_ATTENTION_LANES_H */
