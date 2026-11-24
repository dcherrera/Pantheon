/* SPDX-License-Identifier: AGPL-3.0-or-later */
/* Copyright (C) 2026 David (Corey) Herrera */
/**
 * @file serve_spec_lookup.h
 * @brief Prompt-lookup drafts for the serve's speculative decode: the request's own context
 *        (prompt + everything emitted) searched for the current suffix, its continuation
 *        proposed ahead of the model's MTP drafts.
 *
 * Training-free and host-only (`pn_ngram_draft`, microseconds a cycle). It pays where the
 * answer repeats its context — code edits, quoted text, summaries of a long turn, a re-asked
 * multi-turn question — and proposes nothing on free text, where the MTP head drafts instead.
 * Every draft is verified by the model before it is emitted, so the output is the plain
 * decode's whatever is drafted.
 *
 * Gating: a draft needs a suffix match of at least `min_ngram` tokens; the throttle
 * (`pn_spec_throttle`) stops proposing after lookup cycles that accept too little and probes
 * again later.
 *
 * Env: `PN_TT_SPEC_LOOKUP` (2 = the DEFAULT, David 2026-10-07: only when its first draft is the
 * MTP head's first draft, which the catch-up computes anyway; 1 lookup whenever it matches;
 * 0 off), `PN_TT_SPEC_LOOKUP_K` (drafts a lookup cycle, default 6), `PN_TT_SPEC_LOOKUP_MIN` (shortest suffix match, default 3).
 *
 * status: written for the Tenstorrent serve; model-agnostic.
 * issues: none.
 * todo:   none.
 */
#ifndef PN_SERVE_SPEC_LOOKUP_H
#define PN_SERVE_SPEC_LOOKUP_H

#include <stdbool.h>
#include <stdint.h>

#include "../core/ngram_draft.h"

typedef struct {
    bool on;
    int mode;           /**< PN_TT_SPEC_LOOKUP: 1 matched, 2 matched and agreeing with MTP */
    uint32_t *history;  /**< prompt + every emitted token, oldest first */
    int count, capacity;
    int k;              /**< drafts a lookup cycle proposes at most */
    int min_ngram;      /**< shortest suffix match that may draft */
    pn_spec_throttle throttle;
    uint64_t cycles, drafted, accepted; /**< lookup cycles only */
} sg_spec_lookup;

/**
 * Read the env and copy the prompt. Off (no allocation) when `PN_TT_SPEC_LOOKUP=0`.
 * @param max_new  tokens the request may still emit (sizes the history once)
 * @return 0, or -1 on OOM (left off)
 */
int sg_spec_lookup_init(sg_spec_lookup *lookup, const uint32_t *prompt, int n_prompt,
                        int max_new);

/**
 * The CUDA MTP cycle's lookup (serve_spec_mtp_cuda.h): `QWEN35_MTP_LOOKUP` (0 = off, the
 * default; 1 lookup whenever it matches, in place of the head's drafts; 2 only when its first
 * draft is the head's first draft), `QWEN35_MTP_LOOKUP_K` (drafts a lookup cycle, default and
 * at most 4: the verify's 5 rows), `QWEN35_MTP_LOOKUP_MIN` (shortest suffix match, default 5).
 * @return 0, or -1 on OOM (left off)
 */
int sg_spec_lookup_init_cuda(sg_spec_lookup *lookup, const uint32_t *prompt, int n_prompt,
                             int max_new);

/** Append one emitted token to the history. */
void sg_spec_lookup_push(sg_spec_lookup *lookup, uint32_t token);

/**
 * Drafts for the token after the history's end, at most min(k, cap).
 * @return the number written to `drafts`; 0 when off, throttled or no match long enough
 */
int sg_spec_lookup_draft(sg_spec_lookup *lookup, int cap, uint32_t *drafts);

/** May lookup drafts starting with `looked_first` replace the MTP drafts whose first is
 *  `mtp_first`? Always in mode 1; only when they agree in mode 2. */
bool sg_spec_lookup_agrees(const sg_spec_lookup *lookup, uint32_t looked_first,
                           uint32_t mtp_first);

/** A lookup cycle's outcome: `drafted` proposed, `accepted` of them kept. */
void sg_spec_lookup_report(sg_spec_lookup *lookup, int drafted, int accepted);

/** Free the history and print the tallies once when any lookup cycle ran. */
void sg_spec_lookup_free(sg_spec_lookup *lookup);

#endif /* PN_SERVE_SPEC_LOOKUP_H */
