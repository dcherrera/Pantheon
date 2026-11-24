/* SPDX-License-Identifier: AGPL-3.0-or-later */
/* Copyright (C) 2026 David (Corey) Herrera */
/**
 * @file grammar_row_dump.h
 * @brief Instrument: write every grammar-MASKED dense row the sampler sees to a file, so the
 *        exact masked-row decision (grammar_mask_shortlist.h) can be priced and checked
 *        offline on real rows (tools/grammar_exact/).
 *
 * `PN_GRAMMAR_ROW_DUMP=<path>` (unset = off, nothing runs). One record per masked step on a
 * DENSE row (a backend without top-N rows, e.g. the Mac CPU build): the sampling parameters,
 * the repeat window, the row's top-64 in the TT kernel's order (value desc, id asc) with each
 * entry's grammar verdict, and the count of allowed tokens. The full raw row and the verdict
 * of every token are added when the top-64 shortlist rule alone (gms_mask_from_shortlist, no
 * outside bound) cannot decide the draw — the rows the outside bound is for — or always with
 * `PN_GRAMMAR_ROW_DUMP_ALL=1`.
 *
 * Record layout (host endianness): gro_header, recent[n_recent] u32, top_ids[64] u32,
 * top_vals[64] f32, top_allowed[64] u8, then if has_row: row[n_vocab] f32, verdict[n_vocab] u8.
 *
 * @status Instrument only (the draw is untouched). Used by sample_token (engine.c).
 * @issues None
 * @todo None
 */
#ifndef PN_GRAMMAR_ROW_DUMP_H
#define PN_GRAMMAR_ROW_DUMP_H

#include <stdint.h>

#include "../../include/pantheon.h"

#define GRO_MAGIC 0x31445247u   /* "GRD1" */
#define GRO_TOPN 64

typedef struct {
    uint32_t magic;
    int32_t n_vocab;
    float temperature, top_p;
    int32_t top_k;
    float repeat_penalty, presence_penalty, frequency_penalty;
    int32_t has_logit_bias;
    int32_t n_recent;
    int32_t n_allowed;     /**< tokens the grammar allows on the whole row */
    int32_t has_row;       /**< 1: the full row and every verdict follow */
} gro_header;

/**
 * Append one record (a no-op unless PN_GRAMMAR_ROW_DUMP is set).
 *
 * @param logits   the dense raw row, before the mask
 * @param verdict  1 = allowed, per token (n_vocab entries)
 *
 * Thread-safe: yes (one lock around the file)
 */
void pn_grammar_row_dump(const float *logits, int n_vocab, const unsigned char *verdict,
                         const pn_sampling_params *params, const uint32_t *recent, int n_recent);

/** Is the dump armed? (Callers skip computing verdicts they would not otherwise need.) */
int pn_grammar_row_dump_enabled(void);

#endif /* PN_GRAMMAR_ROW_DUMP_H */
