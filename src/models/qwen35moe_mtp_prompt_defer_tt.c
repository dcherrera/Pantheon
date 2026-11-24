/* SPDX-License-Identifier: AGPL-3.0-or-later */
/* Copyright (C) 2026 David (Corey) Herrera */
/**
 * @file qwen35moe_mtp_prompt_defer_tt.c
 * @brief TTFT lane: the prompt's MTP layer runs AFTER the first token, not before it.
 *
 * status: built 2026-10-22 (`PN_TT_MTP_PROMPT_DEFER`, default on, `=0` off).
 * issues: none known.
 * todo:   none.
 *
 * ---------------------------------------------------------------------------
 * WHY IT CAN MOVE
 * ---------------------------------------------------------------------------
 * The batched prefill ran the MTP block over the prompt (its K/V, so a draft attends to the
 * whole context) inside the last pass, before the tail forward made the logits: ~0.185 s of a
 * 1,630-token TTFT. Nothing the first token needs reads it — only the first MTP draft does.
 * So the last pass RECORDS it (the next-token ids, rows, base position) and leaves the tall
 * hidden it consumes in place; the tail forward runs on the decode strips, which do not touch
 * the tall set; the first token is sampled and emitted; and the layer runs at the first of:
 *
 *   - the first MTP draft (pn_qwen35moe_tt_mtp_draft*),        the speculative path;
 *   - the next forward of that sequence (pn_qwen35moe_forward_tt), the plain path, and the
 *     serve's commit of the last token of a max_tokens=1 request;
 *   - (a verify or a catch-up overwrites the tall hidden; both run inside a stream after the
 *     first draft, so one that ever ran first DROPS it, loudly — the text is unaffected);
 *   - the next prefill: a RESUMED turn of the same sequence runs it first (its MTP K/V below
 *     the checkpoint is this layer's); any other prefill starts a sequence and drops it.
 *
 * The same layer, the same rows, the same hidden and the same K/V positions, so the MTP K/V
 * and every draft are byte-identical to the in-pass layer (the grid serves its projections when
 * it is still running, the matmul group otherwise — bit-identical, PHASE G). The text never
 * depended on it: the verify is lossless.
 *
 * NOT DEFERRED: a pass that is not the prompt's last (its hidden is overwritten by the next
 * pass), no tall strips (the tail forward would overwrite the hidden), and the restage /
 * residue-poison instruments (they rewrite the tall set after the passes).
 *
 * THE GRID is kept running from the prefill until the layer runs (its begin is 0.038 s; the
 * layer runs milliseconds after the first token). The serve's idle tick ends it when the layer
 * is still owed (an EOS first token), and the layer then begins it again if it needs it.
 */

#ifdef USE_TT

#include "qwen35moe_tt_internal.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "../accel/tt/pn_tt_matmul_group.h"
#include "../core/bench.h"
#include "qwen35moe.h"

static bool mtp_prompt_env_set(const char *name)
{
    const char *env = getenv(name);

    return env && *env && atoi(env) != 0;
}

bool pn_qwen35_tt_mtp_prompt_defer_wanted(const pn_qwen35_tt *g)
{
    const char *env = getenv("PN_TT_MTP_PROMPT_DEFER");

    if (env && *env && atoi(env) == 0)
        return false;
    return g && g->mtp.on && g->tall_rows != 0u &&
           !mtp_prompt_env_set("PN_TT_PREFILL_RESTAGE") &&
           !mtp_prompt_env_set("PN_TT_PREFILL_RESIDUE_POISON");
}

int pn_qwen35_tt_mtp_prompt_owe(pn_qwen35_tt *g, const uint32_t *next, int rows, int base_pos)
{
    if (rows < 1)
        return -1;
    if (rows > g->mtp.prompt_cap) {
        uint32_t *grown = realloc(g->mtp.prompt_next, (size_t)rows * sizeof(uint32_t));

        if (!grown)
            return -1;
        g->mtp.prompt_next = grown;
        g->mtp.prompt_cap = rows;
    }
    memcpy(g->mtp.prompt_next, next, (size_t)rows * sizeof(uint32_t));
    g->mtp.prompt_rows = rows;
    g->mtp.prompt_base = base_pos;
    g->mtp.prompt_pending = true;
    g->mtp.prompt_owed = false;
    g->mtp.prompt_grid_kept = true;
    return 0;
}

void pn_qwen35_tt_mtp_prompt_arm(pn_qwen35_tt *g, const void *owner)
{
    if (!g || !g->mtp.prompt_pending)
        return;
    g->mtp.prompt_pending = false;
    g->mtp.prompt_owed = true;
    g->mtp.prompt_owner = owner;
}

void pn_qwen35_tt_mtp_prompt_forget(pn_qwen35_tt *g, const char *why)
{
    if (!g || !(g->mtp.prompt_pending || g->mtp.prompt_owed))
        return;
    g->mtp.prompt_pending = false;
    g->mtp.prompt_owed = false;
    g->mtp.prompt_dropped++;
    fprintf(stderr, "qwen35_tt: MTP PROMPT LAYER dropped — %s\n", why);
}

static void mtp_prompt_grid_release(pn_qwen35_tt *g)
{
    if (g->mtp.prompt_grid_kept)
        pn_tt_matmul_grid_end(g->device);
    g->mtp.prompt_grid_kept = false;
}

void pn_qwen35_tt_mtp_prompt_drop(pn_qwen35_tt *g, const char *why)
{
    if (!g)
        return;
    pn_qwen35_tt_mtp_prompt_forget(g, why);
    mtp_prompt_grid_release(g);
}

int pn_qwen35_tt_mtp_prompt_settle(pn_qwen35_tt *g, pn_qwen35moe_model *m,
                                   const pn_qwen35moe_config *cfg, const void *owner)
{
    const double started = pn_time_ms();
    const char *red = getenv("PN_TT_RED");
    int rc;

    if (!g || !g->mtp.prompt_owed)
        return 0;
    if (owner != g->mtp.prompt_owner) {
        pn_qwen35_tt_mtp_prompt_drop(g, "another sequence took the card");
        return 0;
    }
    g->mtp.prompt_owed = false;
    /* RED `mtp_prompt_defer_skip`: the layer never runs — the first drafts attend to MTP K/V
     * this prompt did not write. The drafts must DIFFER (PN_TT_SPEC_TRACE). */
    if (red && strcmp(red, "mtp_prompt_defer_skip") == 0) {
        mtp_prompt_grid_release(g);
        return 0;
    }
    g->mtp.prompt_grid_kept = false; /* the run ends the grid */
    rc = pn_qwen35_tt_prefill_mtp_prompt_run(g, m, cfg, g->mtp.prompt_next, g->mtp.prompt_rows,
                                             g->mtp.prompt_base);
    g->mtp.prompt_settled++;
    g->mtp.prompt_ms += pn_time_ms() - started;
    if (getenv("PN_TT_PREFILL_REPORT"))
        fprintf(stderr, "qwen35_tt: MTP PROMPT LAYER after the first token — %d rows at %d in "
                        "%.1f ms%s\n", g->mtp.prompt_rows, g->mtp.prompt_base,
                pn_time_ms() - started, rc == 0 ? "" : " — FAILED");
    if (rc != 0)
        fprintf(stderr, "qwen35_tt: the deferred MTP prompt layer failed on the card; the "
                        "drafts of this sequence attend to K/V it did not write\n");
    return rc;
}

void pn_qwen35_tt_mtp_prompt_idle(pn_qwen35_tt *g)
{
    if (g)
        mtp_prompt_grid_release(g);
}

void pn_qwen35_tt_mtp_prompt_free(pn_qwen35_tt *g)
{
    if (!g)
        return;
    free(g->mtp.prompt_next);
    g->mtp.prompt_next = NULL;
    g->mtp.prompt_cap = 0;
    g->mtp.prompt_pending = false;
    g->mtp.prompt_owed = false;
}

#endif /* USE_TT */
