/* SPDX-License-Identifier: AGPL-3.0-or-later */
/* Copyright (C) 2026 David (Corey) Herrera */
/**
 * @file   load_progress.h
 * @brief  Live progress of model loads and unloads (Pantheon Local H2n). The
 *         caller of a load / unload opens a progress slot on its own thread
 *         (pn_load_progress_begin); the engine's upload and free paths on that
 *         thread report per-die device bytes and the layer reached; anyone may
 *         read every open slot at any time. A thread with no open slot makes
 *         the engine's reports no-ops, so plain `serve` loads pay nothing.
 *
 * @status None
 * @issues None
 * @todo   None
 */

#ifndef PN_LOAD_PROGRESS_H
#define PN_LOAD_PROGRESS_H

#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

#define PN_LOAD_PROGRESS_SLOTS     8
#define PN_LOAD_PROGRESS_UNITS     8
#define PN_LOAD_PROGRESS_TAG_MAX   64

typedef enum {
    PN_LOAD_PHASE_LOADING = 1,
    PN_LOAD_PHASE_UNLOADING,
} pn_load_phase;

/* One device the load or unload touches. */
typedef struct {
    int      die;
    uint64_t bytes;     /* device bytes the model holds there now */
    uint64_t total;     /* bytes it will hold (load) / held at the start (unload) */
} pn_load_progress_unit;

typedef struct {
    char                  tag[PN_LOAD_PROGRESS_TAG_MAX];   /* the caller's name */
    int                   replica;
    pn_load_phase         phase;
    int                   layer;          /* layers uploaded so far */
    int                   n_layers;       /* 0 = not reported */
    pn_load_progress_unit units[PN_LOAD_PROGRESS_UNITS];
    int                   n_units;
} pn_load_progress;

/**
 * Open a slot on the calling thread for `tag` / `replica` over `dies` (totals
 * start at 0, or at `totals[i]` when given). Ends any slot this thread had.
 * @return 0, or -1 when every slot is taken (progress is then not reported)
 *
 * Thread-safe: yes. Reentrant: no. Signal-safe: no
 */
int pn_load_progress_begin(const char *tag, int replica, pn_load_phase phase, const int *dies,
                           const uint64_t *totals, int n_dies);

/**
 * Close the calling thread's slot (no-op without one).
 *
 * Thread-safe: yes. Reentrant: no. Signal-safe: no
 */
void pn_load_progress_end(void);

/**
 * Engine side, on the loading thread: the bytes a die will hold, the bytes it
 * holds now, the layers uploaded. A die not in the slot is added (up to
 * PN_LOAD_PROGRESS_UNITS). A total below the bytes reported is raised to them.
 *
 * Thread-safe: yes. Reentrant: yes. Signal-safe: no
 */
void pn_load_progress_set_total(int die, uint64_t total);
void pn_load_progress_set_bytes(int die, uint64_t bytes);
void pn_load_progress_set_layer(int layer, int n_layers);

/**
 * Copy every open slot.
 * @return slots written
 *
 * Thread-safe: yes. Reentrant: yes. Signal-safe: no
 */
int pn_load_progress_list(pn_load_progress *out, int max);

/**
 * The open slot for tag / replica.
 * @return 0, or -1 when none is open
 *
 * Thread-safe: yes. Reentrant: yes. Signal-safe: no
 */
int pn_load_progress_find(const char *tag, int replica, pn_load_progress *out);

/**
 * Bytes open slots report on `die` (summed), and whether any slot touches it.
 *
 * Thread-safe: yes. Reentrant: yes. Signal-safe: no
 */
uint64_t pn_load_progress_die_bytes(int die, int *active);

#ifdef __cplusplus
}
#endif

#endif /* PN_LOAD_PROGRESS_H */
