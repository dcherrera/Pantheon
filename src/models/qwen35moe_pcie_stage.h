/* SPDX-License-Identifier: AGPL-3.0-or-later */
/* Copyright (C) 2026 David (Corey) Herrera */
/**
 * @file   qwen35moe_pcie_stage.h
 * @brief  Host staging of a two-stage (TP2 x 2) decode step's small per-step copies: the
 *         stage-0 input rows to both ranks, the stage hop (stage 0 rank 0 to both stage-1
 *         ranks) and the position uploads, with each PCIe lever behind its own flag.
 *
 * WHY. On a board behind a narrow upstream link every per-step copy pays the link's latency.
 * Today each is a blocking pageable copy: the driver copies the rows into its own staging
 * buffer first, the host waits for every copy, and the same rows cross the link once per
 * rank. These levers cut that without changing a byte the GPU sees (each on by default,
 * "=0" turns it off):
 *   QWEN35_PCIE_PIN=1        host rows in pinned memory, uploads async (no staging copy, the
 *                            host does not wait for the DMA);
 *   QWEN35_PCIE_FANOUT=1     one upload to rank 0, then an on-board peer copy to rank 1 (one
 *                            link crossing instead of two);
 *   QWEN35_PCIE_HOP_EVENT=1  the hop is event-chained: stage 0's rows go to the host and on
 *                            to stage 1 in stream order, with no host wait on stage 0;
 *   QWEN35_PCIE_POS=1        position uploads through per-die staging, async.
 * The fifth lever, how the host waits for the GPU, is foundry's FC_CUDA_WAIT.
 *
 * ORDER. An async upload reads its host source when its stream reaches it, so a host buffer
 * is rewritten only after an event recorded behind every read of it has completed. A peer
 * write into rank 1's buffer is queued behind rank 1's own earlier work (a fence event), and
 * rank 1's later work waits for the copy. Stage 1's uploads wait on an event recorded behind
 * the hop's device-to-host copy.
 *
 * Exact by construction: the same bytes land in the same buffers before the same kernels.
 *
 * @status Default on: a flag unset, "" or "1" is on, anything else (e.g. "0") is off; all
 *         four "0" = the legacy blocking copies, not this module.
 * @issues One step thread: a stage is driven by one thread at a time.
 * @todo   None
 */
#ifndef PN_QWEN35MOE_PCIE_STAGE_H
#define PN_QWEN35MOE_PCIE_STAGE_H

#include <stddef.h>

/** Which staging levers are on. */
typedef struct {
    int pin;        /* QWEN35_PCIE_PIN */
    int fanout;     /* QWEN35_PCIE_FANOUT */
    int hop_event;  /* QWEN35_PCIE_HOP_EVENT */
    int pos;        /* QWEN35_PCIE_POS */
} qw_pcie_flags;

/**
 * Parses the four flag values (each NULL, "" or "1" = on; anything else, e.g. "0", = off).
 *
 * @return The flags (pure; no environment access)
 */
qw_pcie_flags qw_pcie_flags_parse(const char *pin, const char *fanout, const char *hop_event,
                                  const char *pos);

/**
 * One default-on env flag: NULL, "" or "1" = on; anything else (e.g. "0") = off. The same
 * parse the four levers use; QWEN35_SHORT_ROW and QWEN35_TOPK_FAST share it.
 *
 * @return 1 on, 0 off (pure)
 */
int qw_env_flag_default_on(const char *v);

/** The flags from the environment, read once per process. */
qw_pcie_flags qw_pcie_flags_get(void);

/** 1 when any staging lever is on (the step then routes its copies through this module). */
int qw_pcie_flags_any(qw_pcie_flags f);

#ifdef USE_CUDA

#include "cuda/fc_cuda.h"

typedef struct qw_pcie_stage qw_pcie_stage;

/**
 * Creates the staging for one two-stage step. ctx[0], ctx[1] are stage 0's ranks, ctx[2],
 * ctx[3] stage 1's; each pair must have peer access enabled when f.fanout is on.
 *
 * @param ctx         The four dies' contexts, in stage-rank order
 * @param rows_bytes  Capacity of the row buffers (largest rows copy the step makes)
 * @param small_bytes Capacity of each die's position staging (0 when f.pos is off)
 * @param f           The levers to apply
 * @return            The staging, or NULL (allocation or event creation failed)
 */
qw_pcie_stage *qw_pcie_stage_create(fc_cuda_ctx *const ctx[4], size_t rows_bytes,
                                    size_t small_bytes, qw_pcie_flags f);

/** Frees the staging (NULL is a no-op). Every device copy using it must have completed. */
void qw_pcie_stage_free(qw_pcie_stage *st);

/** Capacity of the row buffers in bytes. */
size_t qw_pcie_stage_rows_cap(const qw_pcie_stage *st);

/** The levers this staging applies. */
qw_pcie_flags qw_pcie_stage_flags(const qw_pcie_stage *st);

/** 1 when the row buffers are pinned (0 = fc_host_alloc unavailable, malloc'd). */
int qw_pcie_stage_pinned(const qw_pcie_stage *st);

/**
 * The host rows for stage 0's input, once every earlier device read of them has completed.
 *
 * @return Host buffer of qw_pcie_stage_rows_cap bytes, or NULL on a device error
 */
float *qw_pcie_stage_input_rows(qw_pcie_stage *st);

/**
 * Stage 0's input: `bytes` of the input rows to dst0 (on ctx[0]) and dst1 (on ctx[1]), queued
 * on their streams. Returns at once.
 *
 * @return 0, or -1 (a device call failed: the step must fail)
 */
int qw_pcie_stage_fan_in(qw_pcie_stage *st, fc_buffer *dst0, fc_buffer *dst1, size_t bytes);

/**
 * The stage hop: `bytes` of src0 (stage 0 rank 0, on ctx[0]) to dst2 (ctx[2]) and dst3
 * (ctx[3]). With hop_event it is queued behind ctx[0]'s work and returns at once; otherwise
 * the caller has drained ctx[0], and the rows are on the host before the uploads are queued.
 *
 * @return 0, or -1 (a device call failed: the step must fail)
 */
int qw_pcie_stage_hop(qw_pcie_stage *st, fc_buffer *src0, fc_buffer *dst2, fc_buffer *dst3,
                      size_t bytes);

/**
 * A small upload (positions) to dst on die `die` (0-3) through that die's staging, async.
 *
 * @return 0, -1 (device error), or 1 (bytes above the staging: caller copies directly)
 */
int qw_pcie_stage_upload_small(qw_pcie_stage *st, int die, fc_buffer *dst, const void *src,
                               size_t bytes);

#endif /* USE_CUDA */

#endif /* PN_QWEN35MOE_PCIE_STAGE_H */
