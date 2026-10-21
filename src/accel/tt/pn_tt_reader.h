/* SPDX-License-Identifier: AGPL-3.0-or-later */
/* Copyright (C) 2026 David (Corey) Herrera */
/**
 * @file pn_tt_reader.h
 * @brief A READER THREAD on its own window bank: card->host strip reads (with the host
 *        de-tilize) taken off the thread that issues descriptors.
 *
 * WHY. The prefill's GDN chain step reads `ssm_alpha`/`ssm_beta` down a 32-row band at a time
 * (the host owes softplus/exp/sigmoid): 2,448 tile reads, ~0.94 s of a 1,630-token prefill at
 * the link's ~5.3 MB/s readback, and a PIO read stalls the CPU that issues it. Hidden under the
 * gdnc windows (the alpha/beta prefetch) the reads still EXTEND those windows — the chain is
 * 2.08 s, 1.0 s of it unclocked. The host has two cores and the second one sits idle. A read
 * is the same bytes whichever core issues it, so the answer is bit-identical by construction.
 *
 * HOW IT STAYS SAFE. The thread owns a THIRD udrv window bank
 * (`ft_tt_runtime_udrv_reader_access()`); it never touches the issuing thread's banks, the
 * device's counters or a strip struct (addresses are resolved at post time on the caller's
 * thread). One producer (the caller), one consumer (the thread), a fixed ring, C11 atomics.
 * `PN_TT_READER=0` disables it; so does a transport without the reader bank (tt-kmd), and
 * every caller then reads inline as before. `PN_TT_READER_THREADS=N` (default 1): N workers,
 * each on its own bank, jobs dealt round-robin. Two read no faster (the link serialises PIO
 * reads: 1.95 s summed vs 1.10 for one) — kept as the measured control. Idle workers sleep.
 *
 * @status Shipped (PHASE W); gated by run_w1a_eltwise_wide.sh, run_w_stream2.sh.
 * @issues None
 * @todo None
 */

#ifndef PN_TT_READER_H
#define PN_TT_READER_H

#ifdef USE_TT

#include <stdbool.h>
#include <stdint.h>

typedef struct pn_tt_device pn_tt_device;

/** True when posts will run on the reader thread (starting it on first use). */
bool pn_tt_reader_available(pn_tt_device *device);

/**
 * Queue a read of `tiles` bf16 tiles at (`bank`, `address`) and the de-tilize of their first
 * `rows` rows into `dst` (row stride `count` floats, tile t's columns at t*32) — exactly what
 * `pn_tt_activation_download_rows()` computes for one band.
 *
 * @return a ticket > 0, or 0 when nothing was queued (the caller reads inline)
 *
 * Thread-safe: no (one producer)
 */
uint64_t pn_tt_reader_post_rows(pn_tt_device *device, uint32_t bank, uint32_t address,
                                uint32_t tiles, uint32_t rows, float *dst, uint32_t count);

/** A card-side precondition: the queue at (x, y, control_address) has EXECUTED `target`
 *  descriptors. The reader polls it on its own bank before the read (PHASE W wavefront: a
 *  streamed host never learns when a projection finished, the card does). */
typedef struct pn_tt_reader_gate {
    uint16_t x, y;
    uint32_t control_address;
    uint32_t target;
} pn_tt_reader_gate;

/** As pn_tt_reader_post_rows(), read only once `gate` holds (NULL = at once). A gate not met
 *  within 5 s fails the job (the caller re-reads inline). */
uint64_t pn_tt_reader_post_rows_gated(pn_tt_device *device, uint32_t bank, uint32_t address,
                                      uint32_t tiles, uint32_t rows, float *dst,
                                      uint32_t count, const pn_tt_reader_gate *gate);

/** Wait until `ticket` is done. @return 0, or -1 when that read failed (dst is unspecified). */
int pn_tt_reader_wait(pn_tt_device *device, uint64_t ticket);

/** Wait until everything posted is done (abort paths: nothing may still write a buffer). */
void pn_tt_reader_drain(pn_tt_device *device);

/** Drain with every pending GATE failing at once (an abort: the gated work may never run). */
void pn_tt_reader_drain_cancelled(pn_tt_device *device);

/** Stop and join the thread. Safe when it never started. Called by pn_tt_device_close(). */
void pn_tt_reader_close(pn_tt_device *device);

/** Seconds the WAITERS spent blocked, and reads done (for the prefill report). */
void pn_tt_reader_stats(const pn_tt_device *device, double *out_wait_seconds,
                        uint64_t *out_reads, double *out_read_seconds);

#endif /* USE_TT */
#endif /* PN_TT_READER_H */
