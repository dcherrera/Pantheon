/* SPDX-License-Identifier: AGPL-3.0-or-later */
/* Copyright (C) 2026 David (Corey) Herrera */
/**
 * @file qwen35moe_gate_tables.h
 * @brief The EXACT card gates' lookup tables: the host's own GDN gates
 *        (pn_qwen35moe_gdn_gates, libm) evaluated on every bf16 code, per (layer, head) for
 *        decay and once for beta, in foundry's windowed table format (tt/kernels/ftgc_table.h).
 *
 * WHY EXACT. The prefill's host gates read alpha/beta as bf16 widened by `<< 16` and are
 * elementwise, so a table of the host's answer on all 65,536 codes IS the host's gate. Built
 * by calling the same compiled body the prefill calls (noinline, qwen35moe_ssm.c) with every
 * head on the same code; `pn_qwen35_gate_tables_check()` then holds the lane independence that
 * relies on (a vectorised expf whose lanes saw their neighbours would fail it) over mixed rows
 * — every code in every head position for one layer, random rows for all — and any mismatch
 * REFUSES the tables: the caller stays on the host gates and the load continues.
 *
 * THE CAP. A head's table must fit PN_QWEN35_GATE_TABLE_SLOT_WORDS (40 KiB): the fit model
 * prices that slot, data-independently, before the weights go up. Qwen3.8-27B's widest is
 * 8,004 words (Mac libm); a model whose gates need more than the cap refuses (host gates).
 * On the card each layer's heads are packed at that LAYER's widest table, 64 B aligned.
 *
 * No device: tested on any host (tests/test_tt_gate_tables.c).
 *
 * @status None
 * @issues None
 * @todo None
 */
#ifndef QWEN35MOE_GATE_TABLES_H
#define QWEN35MOE_GATE_TABLES_H

#include <stdbool.h>
#include <stdint.h>

#include "qwen35moe_internal.h"

/** One head's table slot, the cap: 10,240 words (40 KiB, a multiple of the NoC's 64 B). */
#define PN_QWEN35_GATE_TABLE_SLOT_WORDS 10240u
/** The most SSM layers a table set carries (per-layer widths below). */
#define PN_QWEN35_GATE_TABLE_MAX_LAYERS 64u

typedef struct pn_qwen35_gate_tables {
    uint32_t layers, heads;
    /** layers x heads slots; head h of layer l at `(l * heads + h) * SLOT_WORDS`. */
    uint32_t *decay;
    /** One slot: the beta table (layer- and head-independent). */
    uint32_t *beta;
    uint32_t widest; /**< words of the widest table built */
    uint32_t layer_widest[PN_QWEN35_GATE_TABLE_MAX_LAYERS]; /**< per layer, decay only */
} pn_qwen35_gate_tables;

/**
 * Build every table from `layers[0..n_layers)` at `heads` V-heads. @return 0, or -1 (out of
 * memory, a table over the cap, a builder proof that failed, or beta answers that differ
 * between heads) with `out` left empty.
 *
 * Thread-safe: yes (distinct `out`)    Reentrant: yes    Signal-safe: no
 */
int pn_qwen35_gate_tables_build(const pn_qwen35moe_ssm_layer *const *layers, uint32_t n_layers,
                                uint32_t heads, pn_qwen35_gate_tables *out);

/**
 * The lane check: `pn_qwen35moe_gdn_gates` on mixed rows against the tables — every non-NaN
 * code in every head position of layer 0, then `random_rows` random rows per layer.
 * @return words that differ (0 = the tables are the host's gates)
 *
 * Thread-safe: yes    Reentrant: yes    Signal-safe: no
 */
uint64_t pn_qwen35_gate_tables_check(const pn_qwen35_gate_tables *tables,
                                     const pn_qwen35moe_ssm_layer *const *layers,
                                     uint32_t random_rows, uint32_t seed);

/**
 * Build, check, and decide: the tables when both pass, else NULL (said on stderr) — the caller
 * then keeps the host gates and the load goes on. `red` corrupts one entry after the build
 * (`PN_TT_RED=gate_table_corrupt`): the check must refuse.
 *
 * Thread-safe: yes    Reentrant: yes    Signal-safe: no
 */
pn_qwen35_gate_tables *pn_qwen35_gate_tables_prepare(const pn_qwen35moe_ssm_layer *const *layers,
                                                     uint32_t n_layers, uint32_t heads,
                                                     bool red);

/** Free what build (or prepare) made; NULL is fine. */
void pn_qwen35_gate_tables_free(pn_qwen35_gate_tables *tables);

/** The value `tables` gives head `head` of layer `layer` for `code` (decay), or beta. */
uint32_t pn_qwen35_gate_tables_decay(const pn_qwen35_gate_tables *tables, uint32_t layer,
                                     uint32_t head, uint32_t code);
uint32_t pn_qwen35_gate_tables_beta(const pn_qwen35_gate_tables *tables, uint32_t code);

#endif /* QWEN35MOE_GATE_TABLES_H */
