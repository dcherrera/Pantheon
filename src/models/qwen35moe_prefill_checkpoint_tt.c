/* SPDX-License-Identifier: AGPL-3.0-or-later */
/* Copyright (C) 2026 David (Corey) Herrera */
/**
 * @file qwen35moe_prefill_checkpoint_tt.c
 * @brief MULTI-TURN PREFIX/STATE CACHING for Ornith on the card: the batched prefill's
 *        checkpoint, and the resume a later turn takes from it.
 *
 * ---------------------------------------------------------------------------
 * WHY A CHECKPOINT AND NOT THE TURN'S END
 * ---------------------------------------------------------------------------
 * A hybrid model cannot rewind by longest common prefix: every GDN layer's state is a fold of
 * the whole prefix. And the batched prefill is not bit-identical to the per-token path (the
 * chunk mixers run 32 rows at a time), so the state at a turn's end (decode, per token) is
 * not the state a cold prefill of the next prompt passes through. The one state that is: the
 * state at a 32-row BAND boundary of a batched prefill.
 *
 * The passes are 2,048 rows (the tall strips), so a chat prompt is ONE pass, and inside it each
 * SSM layer runs its chunk mixer one 32-row band at a time. Every op of a pass is row-local or
 * runs per 32-row band at an absolute band position, so the state after band j is the same
 * whatever pass it ran in. `qwen35moe_prefill_tt.c` therefore copies each SSM layer's state
 * (conv window + recurrent state, card side, `pn_tt_gdn_checkpoint.c`) when its band loop
 * reaches the LAST 32-row boundary of the batched range, and a later turn whose prompt starts
 * with those tokens restores it and prefills from there: the same bands at the same positions,
 * so a restored-then-tail run is the cold run.
 *
 * The boundary is also kept at or below the end of the history the next turn will resend
 * (the serve's ceiling, qwen35moe_checkpoint_band.h): the template's generation prompt
 * ("<think>\n") is not in the re-rendered history, and a checkpoint that covers it can never
 * be resumed — 1 turn in 32 re-prefilled its whole conversation before the ceiling.
 *
 * ---------------------------------------------------------------------------
 * WHAT KEEPS IT VALID
 * ---------------------------------------------------------------------------
 * The attention K/V is not copied: its rows below the checkpoint are the card's own and stay
 * right while the SAME sequence keeps running AT OR ABOVE the checkpoint. Every card forward,
 * verify and prefill calls `pn_qwen35_tt_ckpt_note()`, and a forward for another pn_state, or
 * at a position below the checkpoint, drops it.
 *
 * `pn_qwen35moe_tt_card_holds()` answers the same question for the prefix cache's APPEND tier,
 * which had no card check at all: session A, then session B, then A's exact-history append ran
 * per token over B's K/V and state. RED `card_holds_skip` restores that (must DIFFER).
 *
 * The checkpoint lives in card DRAM only. Nothing is written to the host's disk: user data
 * never rests on the serving host (a blob for the user's machine would be a separate export).
 *
 * @status Per-layer capture at the last band boundary; resume through the serve's prefix cache.
 * @issues One checkpoint a model (one sequence on the card).
 * @todo None
 */

#ifdef USE_TT

#include "qwen35moe_tt_internal.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "../accel/tt/pn_tt_gdn_checkpoint.h"
#include "qwen35moe.h"
#include "qwen35moe_checkpoint_band.h"
#include "../core/bench.h"

/** `PN_TT_RED=<name>`, spelled here: a model file does not reach into the device layer. */
static bool ckpt_red(const char *name)
{
    const char *value = getenv("PN_TT_RED");

    return value && *value && strcmp(value, name) == 0;
}

bool pn_qwen35_tt_ckpt_enabled(void)
{
    const char *env = getenv("PN_TT_PREFILL_CKPT");

    return !(env && *env && atoi(env) == 0);
}

static void ckpt_drop(pn_qwen35_tt *g, const char *why)
{
    if (!g->ckpt.valid)
        return;
    fprintf(stderr, "qwen35_tt: prefill checkpoint at %d dropped — %s\n", g->ckpt.pos, why);
    g->ckpt.valid = false;
    g->ckpt.restored = false;
}

void pn_qwen35_tt_ckpt_forget(pn_qwen35_tt *g)
{
    if (!g)
        return;
    if (g->ckpt.valid)
        ckpt_drop(g, "a warm-up ran on the card");
    g->card_owner = NULL;
}

void pn_qwen35_tt_ckpt_note(pn_qwen35_tt *g, const pn_state *state, int pos)
{
    if (!g)
        return;
    if (g->ckpt.valid && state != g->ckpt.owner)
        ckpt_drop(g, "the card ran another sequence");
    else if (g->ckpt.valid && pos < g->ckpt.pos)
        ckpt_drop(g, "a forward below it rewrites its K/V");
    if (g->ckpt.restored && pos != g->ckpt.pos)
        g->ckpt.restored = false;
    g->card_owner = state;
}

bool pn_qwen35_tt_ckpt_restored_at(const pn_qwen35_tt *g, const pn_state *state, int pos)
{
    return g && g->ckpt.valid && g->ckpt.restored && g->ckpt.owner == state &&
           g->ckpt.pos == pos && pos > 0;
}

/** Make room for `n` checkpoint tokens. */
static int ckpt_tokens_reserve(pn_qwen35_tt *g, int n)
{
    uint32_t *grown;

    if (n <= g->ckpt.capacity)
        return 0;
    grown = realloc(g->ckpt.tokens, (size_t)n * sizeof(*grown));
    if (!grown)
        return -1;
    g->ckpt.tokens = grown;
    g->ckpt.capacity = n;
    return 0;
}

void pn_qwen35_tt_ckpt_plan(pn_qwen35_tt *g, const pn_state *state, int base, int batch_end)
{
    int at, ceiling;

    if (!g)
        return;
    /* The ceiling is for THIS prefill of its state only: a later prefill (another request,
     * a warm-up) must not inherit it. */
    ceiling = (state && g->ckpt.ceiling_owner == state) ? g->ckpt.ceiling : 0;
    g->ckpt.ceiling = 0;
    g->ckpt.ceiling_owner = NULL;
    g->ckpt.pending_at = 0;
    g->ckpt.pending_layers = 0;
    g->ckpt.capture_ms = 0.0;
    g->ckpt.verify_words = 0u;
    g->ckpt.verify_differ = 0;
    /* `ckpt_off`: the load could not reserve the twins (qwen35moe_fit_tt.c). No retry here:
     * a retry would take DRAM a pinned buffer needs, and it failed at load already. */
    if (!state || !pn_qwen35_tt_ckpt_enabled() || g->ckpt_off || !g->gdn_on_card ||
        base % 32 != 0)
        return;
    at = pn_qwen35_ckpt_band_at(base, batch_end, ceiling);
    if (at == 0)
        return; /* no new full band: the checkpoint at `base`, if any, still stands */
    /* The tokens below `base` are the restored checkpoint's, which the resume matched against
     * this prompt. A prefill from anywhere else has a prefix this file does not know. */
    if (base != 0 && !(g->ckpt.valid && g->ckpt.owner == state && g->ckpt.pos == base)) {
        ckpt_drop(g, "a prefill from an unknown base");
        return;
    }
    if (ckpt_tokens_reserve(g, at) != 0) {
        ckpt_drop(g, "no host memory for its tokens");
        return;
    }
    /* The copies overwrite the stored state layer by layer: never a half-taken checkpoint. */
    g->ckpt.valid = false;
    g->ckpt.restored = false;
    g->ckpt.pending_at = at;
}

void pn_qwen35_tt_ckpt_band_done(pn_qwen35_tt *g, int ssm_index, int end)
{
    const double started = pn_time_ms();
    long long differ;

    if (!g || g->ckpt.pending_at == 0 || end != g->ckpt.pending_at)
        return;
    if (pn_tt_gdn_checkpoint_capture_layer(g->device, (uint32_t)ssm_index) != 0) {
        g->ckpt.pending_at = 0; /* abandoned: the commit finds nothing to commit */
        return;
    }
    g->ckpt.capture_ms += pn_time_ms() - started;
    g->ckpt.pending_layers++;
    differ = pn_tt_gdn_checkpoint_verify_layer(g->device, (uint32_t)ssm_index,
                                               &g->ckpt.verify_words);
    if (differ < 0 || g->ckpt.verify_differ < 0)
        g->ckpt.verify_differ = -1;
    else
        g->ckpt.verify_differ += differ;
}

void pn_qwen35_tt_ckpt_commit(pn_qwen35_tt *g, const pn_state *state, const uint32_t *tokens,
                              int base, int n_ssm)
{
    const int at = g ? g->ckpt.pending_at : 0;

    if (!g || at == 0)
        return;
    g->ckpt.pending_at = 0;
    if (g->ckpt.pending_layers != n_ssm) {
        fprintf(stderr, "qwen35_tt: prefill checkpoint at %d NOT taken — %d of %d layers "
                        "captured\n", at, g->ckpt.pending_layers, n_ssm);
        return;
    }
    memcpy(g->ckpt.tokens + base, tokens, (size_t)(at - base) * sizeof(*tokens));
    g->ckpt.pos = at;
    g->ckpt.owner = state;
    g->ckpt.valid = true;
    fprintf(stderr, "qwen35_tt: prefill checkpoint at %d — %d layers copied in %.1f ms\n", at,
            n_ssm, g->ckpt.capture_ms);
    pn_tt_device_report_dram_stage(g->device, "after prefill checkpoint");
    if (pn_tt_gdn_checkpoint_verify_wanted())
        fprintf(stderr, "pn_tt: gdn checkpoint — VERIFY after capture: %lld of %llu sampled "
                        "words DIFFER\n", g->ckpt.verify_differ, g->ckpt.verify_words);
}

void pn_qwen35_tt_ckpt_release(pn_qwen35_tt *g)
{
    if (!g)
        return;
    free(g->ckpt.tokens);
    memset(&g->ckpt, 0, sizeof(g->ckpt));
}

tm_tensor *pn_qwen35_tt_prefill_abort(pn_qwen35_tt *g, pn_state *state, const char *reason,
                                      int positions_done)
{
    fprintf(stderr, "qwen35_tt: BATCHED PREFILL ABORTED after %d positions — %s. The card's "
                    "carried state and K/V are PARTLY WRITTEN, so the session is INVALIDATED: "
                    "this request fails, and the next one starts a fresh sequence.\n",
            positions_done, reason ? reason : "(no reason)");
    if (g) {
        g->ckpt.pending_at = 0;
        ckpt_drop(g, "a prefill aborted over it");
        g->ckpt.owner = NULL;
        g->card_owner = NULL;
        g->prefill_last_batched = -1;
        g->prefill_aborted = true;
    }
    if (state)
        state->pos = 0;
    return NULL;
}

/** The resident model behind `model`, or NULL. */
static pn_qwen35_tt *ckpt_model(const pn_model *model)
{
    const pn_qwen35moe_model *m;

    if (!model || !model->weights || model->forward != pn_qwen35moe_forward_tt)
        return NULL;
    m = (const pn_qwen35moe_model *)model->weights;
    return (pn_qwen35_tt *)pn_qwen35moe_model_gpu((pn_qwen35moe_model *)m);
}

int pn_qwen35moe_tt_checkpoint_resume(pn_model *model, pn_state *state, const uint32_t *tokens,
                                      int n)
{
    pn_qwen35_tt *g = ckpt_model(model);
    const int pos = g ? g->ckpt.pos : 0;

    if (!g || !state || !tokens || !g->resident || !g->gdn_on_card ||
        !pn_qwen35_tt_ckpt_enabled() || !g->ckpt.valid)
        return 0;
    /* The card must hold this sequence (its K/V below `pos`), and a prompt token must be left
     * over: the last one is the ordinary forward that produces the logits. */
    if (g->ckpt.owner != state || g->card_owner != state || pos >= n ||
        memcmp(tokens, g->ckpt.tokens, (size_t)pos * sizeof(*tokens)) != 0)
        return 0;
    if (pn_tt_gdn_checkpoint_restore(g->device) != 0) {
        ckpt_drop(g, "the restore copy failed");
        return -1;
    }
    if (pn_tt_gdn_checkpoint_verify_wanted()) {
        const int n_ssm = pn_qwen35moe_model_n_ssm((const pn_qwen35moe_model *)model->weights);
        unsigned long long words = 0u;
        long long differ = 0, one;
        uint32_t layer;

        for (layer = 0; layer < (uint32_t)n_ssm; layer++) {
            one = pn_tt_gdn_checkpoint_verify_layer(g->device, layer, &words);
            differ = one < 0 || differ < 0 ? -1 : differ + one;
        }
        fprintf(stderr, "pn_tt: gdn checkpoint — VERIFY after restore: %lld of %llu sampled "
                        "words DIFFER\n", differ, words);
    }
    state->pos = pos;
    g->ckpt.restored = true;
    return pos;
}

void pn_qwen35moe_tt_checkpoint_ceiling(pn_model *model, const pn_state *state, int ceiling)
{
    pn_qwen35_tt *g = ckpt_model(model);

    if (!g)
        return;
    g->ckpt.ceiling = ceiling;
    g->ckpt.ceiling_owner = state;
}

int pn_qwen35moe_tt_checkpoint_available(const pn_model *model)
{
    const pn_qwen35_tt *g = ckpt_model(model);

    return g && g->resident && g->gdn_on_card && !g->ckpt_off && pn_qwen35_tt_ckpt_enabled();
}

int pn_qwen35moe_tt_card_holds(const pn_model *model, const pn_state *state)
{
    const pn_qwen35_tt *g = ckpt_model(model);

    if (g && ckpt_red("card_holds_skip"))
        return 1;
    return g && state && g->card_owner == state ? 1 : 0;
}

#endif /* USE_TT */
