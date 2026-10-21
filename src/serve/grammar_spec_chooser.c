/* SPDX-License-Identifier: AGPL-3.0-or-later */
/* Copyright (C) 2026 David (Corey) Herrera */
/*
 * Speculative decoding under a tool-call grammar — see grammar_spec_chooser.h
 * for the exactness argument behind each branch.  Pure C, libc-only.
 */
#include "grammar_spec_chooser.h"

#include <stddef.h>

/* Text of a token, or NULL when it has none (an end/special token). */
static const char *gsc_token_text(const gsc_chooser *c, uint32_t id)
{
    const char *text = c->text ? c->text(c->text_ctx, id) : NULL;
    return (text && *text) ? text : NULL;
}

/* The sampler's own rule: a token with no text is always allowed. */
static bool gsc_token_allowed(const gsc_chooser *c, const void *walk, uint32_t id)
{
    const char *text = gsc_token_text(c, id);
    return !text || c->ops->allowed(walk, text);
}

static void gsc_advance_walk(gsc_chooser *c, uint32_t id)
{
    const char *text = gsc_token_text(c, id);
    if (text) c->ops->advance(c->walk, text);
    if (c->ops->complete(c->walk)) c->completed = true;
}

int gsc_truncate_drafts(gsc_chooser *c, const uint32_t *drafts, int n)
{
    c->ops->copy(c->walk, c->live);
    if (c->ops->complete(c->walk)) {
        c->drafts_cut += (unsigned long)n;
        return 0;
    }
    for (int i = 0; i < n; i++) {
        if (c->ops->mask_active(c->walk) && !gsc_token_allowed(c, c->walk, drafts[i])) {
            c->drafts_cut += (unsigned long)(n - i);
            return i;
        }
        const char *text = gsc_token_text(c, drafts[i]);
        if (text) c->ops->advance(c->walk, text);
        if (c->ops->complete(c->walk)) {
            c->drafts_cut += (unsigned long)(n - i - 1);
            return i + 1;
        }
    }
    return n;
}

void gsc_begin_cycle(gsc_chooser *c)
{
    c->ops->copy(c->walk, c->live);
    c->row = 0;
    c->completed = false;
}

/* The verify's greedy rule over a sparse row: largest value, lowest id on a tie
 * (qwen35moe_resident_tt.c spec head). */
static uint32_t gsc_greedy_pick(const uint32_t *index, const float *value, uint32_t count)
{
    uint32_t best = index[0];
    float best_value = value[0];
    for (uint32_t i = 1; i < count; i++) {
        if (value[i] > best_value || (value[i] == best_value && index[i] < best)) {
            best = index[i];
            best_value = value[i];
        }
    }
    return best;
}

uint32_t gsc_choose(void *ctx, const uint32_t *index, const float *value, uint32_t count)
{
    gsc_chooser *c = (gsc_chooser *)ctx;
    const int row = c->row++;
    uint32_t token;

    if (c->completed) {
        c->rows_deferred++;
        return GSC_DEFER;
    }
    if (!c->ops->mask_active(c->walk)) {
        if (c->base)
            token = c->base(c->base_ctx, row, index, value, count);
        else
            token = count ? gsc_greedy_pick(index, value, count) : (row > 0 ? GSC_DEFER : 0u);
    } else {
        c->rows_masked++;
        token = c->masked(c->masked_ctx, c->walk, row, index, value, count);
    }
    if (token == GSC_DEFER) {
        c->rows_deferred++;
        return GSC_DEFER;
    }
    gsc_advance_walk(c, token);
    return token;
}
