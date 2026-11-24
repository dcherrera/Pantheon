/* SPDX-License-Identifier: AGPL-3.0-or-later */
/* Copyright (C) 2026 David (Corey) Herrera */
/**
 * @file   act_stats.h
 * @brief  PN_ACT_STATS: per matmul weight tensor, accumulate sum x_j^2 per input channel over
 *         every activation row the CPU forward multiplies it by (llama.cpp imatrix semantics),
 *         and optionally the full X^T X for named tensors. The mixw tier's imatrix source.
 *
 *   PN_ACT_STATS=<file>             on (default off); <file> gets the imatrix at end of run
 *   PN_ACT_STATS_FULL=<a,b,...>     also X^T X for these GGUF tensor names, in <file>.xtx
 *   PN_ACT_STATS_XTX_PATH=<path>    the X^T X file goes here instead (a FIFO works: see below)
 *   PN_ACT_STATS_XTX_F32=1          f32 X^T X accumulators (half the RAM); the file stays f64
 *
 * One hook: the qweight activation observer (pn_qmv / pn_qmv_group / pn_qmm / pn_qmm_prefill),
 * keyed by the weight's data pointer, which pn_load maps back to its GGUF tensor name (FTTW
 * overlay tensors included). Both files are opened (truncated) when the stats start and
 * written once, when the owning model is freed. Format and merge rule: act_stats_file.h
 * (tools/oracle/act_stats_merge sums the files of split halves or separate runs).
 *
 * X^T X is accumulated in batches (act_stats_xtx.h) and written STRICTLY SEQUENTIALLY: the
 * header's counts are known at the write, nothing is sought or patched, so the file can be a
 * named pipe a consumer reads in one pass (H never touches disk). A FIFO path is opened at the
 * write, not at the start. Tensors go out in PN_ACT_STATS_FULL order, printed to stderr.
 *
 * Dispatches on a weight with no GGUF name (e.g. MoE expert slices, the MTP eh_proj halves)
 * are counted and reported at the end, never guessed at.
 *
 * Coverage: the CPU forward, which for qwen35 is one token at a time (decode, the per-token
 * prompt loop, the oracle's teacher-forced positions), plus any batched CPU caller of pn_qmm /
 * pn_qmm_prefill. A matmul an accelerator serves (qwen35moe_mv's card arm, the TT batched
 * prefill) never reaches the observer and is not recorded. Observer calls must come from one
 * thread at a time (the qwen35 CPU forward dispatches every matmul from its calling thread).
 *
 * @status None
 * @issues None
 * @todo   None
 */
#ifndef PN_ACT_STATS_H
#define PN_ACT_STATS_H

#include "../formats/gguf.h"

#ifdef __cplusplus
extern "C" {
#endif

/**
 * Start accumulating: map every 2-D tensor of `gguf` (data + offset -> name), open `path` (and
 * `path`.xtx when full_list names any tensor), install the activation observer.
 *
 * @param path       imatrix output file
 * @param full_list  comma-separated tensor names for X^T X, or NULL / ""
 * @param gguf       the loaded model's GGUF (FTTW overlay attached if any)
 * @param owner      the model whose pn_act_stats_finish writes the files
 * @return 0, or -1 (already active, a FULL name not in the GGUF, an X^T X that cannot be
 *         allocated, an output that cannot be opened; stderr says which; nothing installed)
 *
 * Thread-safe: no   Reentrant: no   Signal-safe: no
 */
int pn_act_stats_begin(const char *path, const char *full_list, const pn_gguf_file *gguf,
                       const void *owner);

/** X^T X options of pn_act_stats_begin_ex(). */
typedef struct pn_act_stats_options {
    const char *xtx_path; /**< the X^T X file; NULL or "" = `path`.xtx */
    int         xtx_f32;  /**< nonzero: f32 accumulators (the file stays f64) */
} pn_act_stats_options;

/**
 * pn_act_stats_begin with X^T X options (NULL = the defaults). Same contract otherwise.
 *
 * Thread-safe: no   Reentrant: no   Signal-safe: no
 */
int pn_act_stats_begin_ex(const char *path, const char *full_list, const pn_gguf_file *gguf,
                          const void *owner, const pn_act_stats_options *options);

/**
 * pn_act_stats_begin_ex from PN_ACT_STATS / PN_ACT_STATS_FULL / PN_ACT_STATS_XTX_PATH /
 * PN_ACT_STATS_XTX_F32.
 *
 * @return 0 knob off (nothing done), 1 started, -1 refused (stderr)
 *
 * Thread-safe: no   Reentrant: no   Signal-safe: no
 */
int pn_act_stats_begin_from_env(const pn_gguf_file *gguf, const void *owner);

/**
 * Skip (paused != 0) or resume accumulating. The split oracle's first half pauses around the
 * LM head, whose input there is a mid-stack residual, not the model's.
 *
 * Thread-safe: no   Reentrant: no   Signal-safe: no
 */
void pn_act_stats_set_paused(int paused);

/** @return 1 while stats are being accumulated, else 0. Thread-safe: no. */
int pn_act_stats_active(void);

/**
 * If `owner` started the stats: uninstall the observer, write the imatrix (and X^T X) files,
 * report unnamed / mismatched dispatches on stderr, free everything. Otherwise a no-op.
 *
 * @return 0 written or nothing to do, -1 a write failed (stderr)
 *
 * Thread-safe: no   Reentrant: no   Signal-safe: no
 */
int pn_act_stats_finish(const void *owner);

#ifdef __cplusplus
}
#endif

#endif /* PN_ACT_STATS_H */
