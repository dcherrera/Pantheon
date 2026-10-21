/* SPDX-License-Identifier: AGPL-3.0-or-later */
/* Copyright (C) 2026 David (Corey) Herrera */
/**
 * @file grammar_spec_chooser.h
 * @brief Speculative decoding under a tool-call grammar: draft truncation and a
 *        verify-row chooser whose output equals plain constrained decoding.
 *
 * A speculative cycle verifies [seed, d0, d1, ...] in one pass and gets, per row
 * r, the card's exact top-N (index, value) pairs for the position after row r's
 * input.  Plain decoding would pick each token with the grammar mask applied to
 * the DENSE row.  This chooser makes the same pick, row by row, from a private
 * copy of the grammar walk advanced over the tokens it has already chosen:
 *
 *   - mask inactive at row r (reasoning, prose)   → the unconstrained pick (the
 *     base chooser / the verify's greedy rule) — the mask is a no-op there.
 *   - mask active                                 → the caller's masked pick
 *     with this private walk.  It must return exactly the plain constrained
 *     step's token: the serve decides from the row's top-N when the shortlist
 *     rule proves that exact (grammar_mask_shortlist.h), else row 0 downloads
 *     the dense row and masks it as plain decode does, and a later row DEFERS —
 *     the cycle ends at the previous row and the next cycle meets this position
 *     as its row 0.
 *   - the row after the one that COMPLETED the call → DEFER, so nothing past the
 *     call is committed to the state.
 *
 * A draft is accepted iff it equals the chosen token, so a draft that breaks the
 * grammar can never be accepted.  gsc_truncate_drafts additionally cuts drafts
 * at the first one the grammar rejects (it would cost a verify row for nothing)
 * and right after the one that completes the call.
 *
 * Pure C, libc-only: the grammar, the token text, the base pick and the masked
 * pick all arrive as callbacks, so tests/test_grammar_spec_chooser.c drives it
 * with the real JSON grammar and synthetic rows, no device.
 *
 * @status Used by the Tenstorrent single-user serve loop (serve_generation.c,
 *         sg_spec_cycle) for tool_choice "required".
 * @issues None
 * @todo None
 */
#ifndef PN_GRAMMAR_SPEC_CHOOSER_H
#define PN_GRAMMAR_SPEC_CHOOSER_H

#include <stdbool.h>
#include <stdint.h>

/** Same value as qwen35moe.h's PN_SPEC_CHOOSE_DEFER. */
#define GSC_DEFER 0xFFFFFFFFu

/** Grammar walk operations (one dialect's tg_ or tgj_ functions behind void pointers). */
typedef struct {
    void (*copy)(void *dst, const void *src);
    bool (*allowed)(const void *st, const char *text);
    void (*advance)(void *st, const char *text);
    bool (*mask_active)(const void *st);
    bool (*complete)(const void *st);
} gsc_grammar_ops;

/** A token's decoded text (NULL/"" = no text: always allowed, no advance). */
typedef const char *(*gsc_token_text_fn)(void *ctx, uint32_t id);

/** Unconstrained pick for verify row `row` (sampling mode).  May return
 *  GSC_DEFER for row > 0. */
typedef uint32_t (*gsc_base_choose_fn)(void *ctx, int row, const uint32_t *index,
                                       const float *value, uint32_t count);

/** A row under an active mask: the plain constrained step's token for this row
 *  with `walk` as the grammar state, or GSC_DEFER when the row's top-N cannot
 *  decide it exactly (only for row > 0 — row 0 must decide). */
typedef uint32_t (*gsc_masked_choose_fn)(void *ctx, void *walk, int row, const uint32_t *index,
                                         const float *value, uint32_t count);

typedef struct {
    const gsc_grammar_ops *ops;
    const void *live;              /**< the serve's walk, already past the seed */
    void *walk;                    /**< caller-owned scratch of the same type */
    gsc_token_text_fn text;
    void *text_ctx;
    gsc_base_choose_fn base;       /**< NULL = greedy (largest value, lowest id) */
    void *base_ctx;
    gsc_masked_choose_fn masked;
    void *masked_ctx;
    int row;                       /**< next verify row this cycle */
    bool completed;                /**< a chosen token completed the call */
    unsigned long rows_masked;     /**< rows chosen under an active mask */
    unsigned long rows_deferred;   /**< rows deferred (undecidable, or past the call) */
    unsigned long drafts_cut;      /**< drafts removed by gsc_truncate_drafts */
} gsc_chooser;

/**
 * Cut `drafts` to the prefix worth verifying from the live walk: stop before the
 * first draft the grammar rejects while its mask is active, and right after the
 * draft that completes the call.  Uses (and clobbers) c->walk.
 * @return the number of drafts kept (0..n).
 */
int gsc_truncate_drafts(gsc_chooser *c, const uint32_t *drafts, int n);

/** Start a verify cycle: c->walk = c->live, row 0. */
void gsc_begin_cycle(gsc_chooser *c);

/**
 * pn_spec_choose_fn-compatible chooser (ctx = gsc_chooser*).  Called once per
 * emitted row, in order.  Returns the chosen token or GSC_DEFER (never at row 0).
 */
uint32_t gsc_choose(void *ctx, const uint32_t *index, const float *value, uint32_t count);

#endif /* PN_GRAMMAR_SPEC_CHOOSER_H */
