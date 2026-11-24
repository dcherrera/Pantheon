/* SPDX-License-Identifier: AGPL-3.0-or-later */
/* Copyright (C) 2026 David (Corey) Herrera */
/**
 * @file qwen35moe_prefill_tt.c
 * @brief BATCHED PREFILL for Ornith: up to 32 prompt positions through one set of
 *        projections, with the two mixers still one position at a time.
 *
 * ---------------------------------------------------------------------------
 * WHY THIS IS NOT THE LLAMA BATCHED PREFILL
 * ---------------------------------------------------------------------------
 * `llama_forward_tt.c` batches a prompt by putting 32 positions in rows 0..31 of every strip
 * and letting the card multiply the rows it was multiplying anyway. It can do that because it
 * REFUSES on-card attention: with attention on the host, every position-dependent step is
 * host arithmetic over host rows, and nothing on the card ever needs to know which row is
 * which token.
 *
 * Ornith is the opposite arm. Both of its mixers run WHOLE ON THE CARD, and both are
 * single-position by construction:
 *
 *   - the Gated-Delta-Net op advances a carried state in card DRAM, one position per pair of
 *     descriptors, and the recurrence is sequential in time by definition;
 *   - `pn_tt_rope`, `pn_tt_kv_append` and flash decode each take ONE `pos`.
 *
 * And every one of those ops reads LOGICAL ROW 0 of its input strip. **There is no on-card
 * row gather** — the op set has BROADCAST_COL and nothing that moves row `r` of a tile to row
 * 0 — so a 32-row pass can batch the norms, the nine projections a layer, the SwiGLU and the
 * residual adds, and it must then reach each of its 32 rows THROUGH THE LINK: pull the
 * mixer's input down as whole tiles once a layer, put row `r` back as row 0, run the mixer,
 * pull its row-0 answer back, and re-upload all 32 rows for the output projection.
 *
 * That trade is the whole of this file and it is not obviously a win, which is why it is
 * measured rather than argued and why it defaults OFF:
 *
 *   WHAT IT SAVES   the projections, the model RMS norms, the SwiGLU and the residual adds
 *                   stop being paid per position. A matvec is `Mt=1` and the FPU already
 *                   multiplies 32 rows against every weight tile it reads; 31 of them were
 *                   zero padding.
 *   WHAT IT COSTS   one n_embd-wide row-0 read and one strip-wide row-0 write per LAYER per
 *                   POSITION — the traffic Phase A spent the whole program removing
 *                   (readback 1478.09 -> 6.09 KiB/token).
 *
 * `PN_TT_PREFILL_REPORT=1` prints the split, so the arm answers its own question.
 *
 * ---------------------------------------------------------------------------
 * WHAT IS AND IS NOT BATCHED, AND WHY EACH
 * ---------------------------------------------------------------------------
 *   BATCHED   `pn_tt_rms_norm`      submits `row_blocks = 1`, and one row block IS 32 rows,
 *                                   each reduced over its own row.
 *             every projection      `C[m][n]` sums over k and reads only row m of A.
 *             `pn_tt_add`,          per element, so per row already.
 *             `pn_tt_silu_mul`
 *   PER ROW   the GDN block         a recurrence; row r's answer depends on row r-1's state.
 *             the attention core    rotary, K/V append and the flash decode each name a
 *                                   position, and row r must attend rows <= r.
 *
 * The per-row loops run STRICTLY IN POSITION ORDER, and that is a correctness requirement
 * rather than a convenience: the loop order IS the causal mask and IS the recurrence's time
 * axis. `PN_TT_RED=prefill_causal` runs it backwards, which is the mutation that proves so.
 *
 * ---------------------------------------------------------------------------
 * NO LM HEAD RUNS IN A BATCHED PASS
 * ---------------------------------------------------------------------------
 * The caller hands the LAST prompt token to the ordinary single-token forward, which is what
 * produces the logits — so the head, the top-N and the sparse-logit presentation are literally
 * the shipping code and cannot diverge. It also makes the acceptance argument short: if the
 * card state this writes is what the sequential path writes, the logits are what the
 * sequential path produces, because they come out of the same function reading the same card.
 *
 * @status Written, default OFF. `PN_TT_PREFILL=1` arms it.
 * @issues The gather is host-mediated because the op set has no row gather. If the link makes
 *   that a net loss, the answer is an on-card row-select descriptor (or the chunkwise
 *   parallel delta rule, which subsumes it) and not a tuning of this file.
 * @todo None
 */

#ifdef USE_TT

#include "qwen35moe_tt_internal.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "../accel/tt/pn_tt_attention_grid.h"
#include "../accel/tt/pn_tt_dispatch.h"
#include "../accel/tt/pn_tt_eltwise_group.h"
#include "../accel/tt/pn_tt_gdn.h"
#include "../accel/tt/pn_tt_gdnc.h"
#include "../accel/tt/pn_tt_gdngate.h"
#include "../accel/tt/pn_tt_head_norm.h"
#include "../accel/tt/pn_tt_lend_ledger.h"
#include "../accel/tt/pn_tt_matmul_group.h"
#include "../accel/tt/pn_tt_prefill_lend.h"
#include "../accel/tt/pn_tt_replica_map.h"
#include "../accel/tt/pn_tt_residency.h"
#include "../accel/tt/pn_tt_strip_placement.h"
#include "../core/bench.h"
#include "qwen35moe.h"
#include "qwen35moe_prefill_tt_wall.h"
#include "../core/sample_profiler.h"

/**
 * **ON BY DEFAULT as of 2026-08-25.** Only an explicit `0` puts prefill back on the per-token
 * path.
 *
 * IT WAS OFF UNTIL THE MEASUREMENT SAID WHICH WAY IT GOES, and it now has. The batched
 * plumbing on its own is NUMERICALLY EXACT — `PN_TT_PREFILL=1` with both GDN mixers still the
 * sequential decode family scores BIT-IDENTICALLY to the per-token path over 339 positions,
 * every figure to every printed digit — so the batched projections, the model norms, the
 * SwiGLU, the residual adds and the host row select contribute EXACTLY ZERO. What it buys is
 * the batched attention and the projections; what makes it worth having is
 * `PN_TT_PREFILL_GDNC`, which is on with it.
 */
#define PN_QWEN35_TT_PREFILL_ENV "PN_TT_PREFILL"

/** Rows of one batched pass. 32 is the tile and the default; 1 turns the batched path into
 *  the sequential computation THROUGH THE BATCHED CODE, which is the control that separates a
 *  plumbing defect from a batching defect. */
#define PN_QWEN35_TT_PREFILL_ROWS_ENV "PN_TT_PREFILL_ROWS"

/** Print the transfer / sequential / batched split at the end of the prefill. */
#define PN_QWEN35_TT_PREFILL_REPORT_ENV "PN_TT_PREFILL_REPORT"

/** Batched rows from which a prefill starts the PHASE G grid (`PN_TT_PREFILL_GRID_MIN_ROWS`).
 *  The serve warm-up (PN_QWEN35_TT_WARM_ROWS) is above it, so it boots the grid before the
 *  first request. */
#define PN_QWEN35_TT_GRID_MIN_ROWS 384

/**
 * `PN_TT_PREFILL_ATTN_CHUNK=0` puts the ATTENTION layers back on the per-position path while
 * the SSM layers stay chunked. ON by default, because the chunk is bit-exact there.
 *
 * IT IS AN ISOLATION KNOB AND IT EXISTS BECAUSE THE TWO CHUNK PATHS CANNOT OTHERWISE BE
 * SEPARATED. `PN_TT_PREFILL_ROWS` moves both at once and it moves them differently: attention
 * needs a chunk whose first position is a multiple of 32, so at ROWS=4 only the FIRST pass
 * takes the attention chunk while every pass takes the SSM one. A chunk-size sweep read
 * through that confound cannot say which of the two a divergence belongs to.
 */
#define PN_QWEN35_TT_PREFILL_ATTN_CHUNK_ENV "PN_TT_PREFILL_ATTN_CHUNK"

/**
 * The host scratch a batched pass row-selects through. One `[rows x width]` fp32 buffer per
 * strip that has to cross, allocated on the first batched prefill and kept for the process.
 */
struct pn_qwen35_tt_batch {
    int rows;

    float *hidden;   /**< n_embd, the embedding rows going up */
    float *ssm_qkv;  /**< the fused q|k|v stream, conv_dim */
    float *ssm_z;
    float *ssm_alpha;
    float *ssm_beta;
    float *ssm_gated;
    float *attn_q;
    float *attn_z;
    float *attn_k;
    float *attn_v;
    float *attn_gated;

    /* The arm's own accounting. Seconds, cumulative over the prefill. */
    double transfer_seconds;
    double sequential_seconds;
    double batched_seconds;
    /* G0: `sequential` ATTRIBUTED. Each sequential bracket snapshots every card clock the
     * device keeps (`pn_tt_device_card_clocks()`) on entry and exit and adds the difference
     * here, entry by entry, so the report can say which clocks the bracket's wall contains
     * and how much of it NO clock contains. */
    const char *sequential_clock_names[PN_TT_CARD_CLOCKS_MAX];
    double sequential_clock_seconds[PN_TT_CARD_CLOCKS_MAX];
    unsigned sequential_clock_count;
    /** MEASURE (2026-10-05): `batched` split by bracket site — see kPrefillBatchedNames. */
    double batched_site_seconds[7];
    /** PHASE G: the widest span any pass of this prefill wrote — what the restage zeroes. */
    uint32_t max_span;
    double sequential_unmatched_seconds; /**< brackets a residency armed inside */
    uint64_t passes;
    uint64_t rows_done;
};

/** Same knob and same shape as the device layer's `pn_tt_red`. Every mutation here is HOST
 *  ordering only — no transfer changes length, so none can reach a zero-length NoC hazard.
 *
 *    prefill_causal   the per-row loop runs in REVERSE, so row r's mixer runs after rows
 *                     later than it. For attention that breaks the mask; for the GDN block it
 *                     feeds the recurrence its positions out of order. The loop order is the
 *                     only thing enforcing either, which is what this proves.
 *    prefill_rowfill  every row is served row 0's projection output — exactly the bug of a
 *                     batch whose transfer still only moves logical row 0.
 *    prefill_pos      every row runs the mixer at the pass's FIRST position, so 32 K/V
 *                     entries land on one slot and the rotary is wrong for 31 of them.
 */
static bool prefill_red(const char *name)
{
    const char *value = getenv("PN_TT_RED");

    return value && *value && strcmp(value, name) == 0;
}

/** Default ON: only an explicit `0` puts the attention layers back per position. */
/** `PN_TT_PREFILL_STAGE_ZERO=1`: zero a staged strip's whole span before its upload (the
 *  pre-2026-10-22 behaviour), not only the bands past it. */
static bool prefill_stage_zero_all(void)
{
    const char *env = getenv("PN_TT_PREFILL_STAGE_ZERO");

    return env && *env && atoi(env) != 0;
}

static bool prefill_attn_chunk_enabled(void)
{
    const char *env = getenv(PN_QWEN35_TT_PREFILL_ATTN_CHUNK_ENV);

    return !(env && *env && atoi(env) == 0);
}

/** How many 32-row bands a pass has. Every strip is allocated the same height, so hidden[0]
 *  speaks for all of them; a model whose strips did not arm at all reports one. */
static uint32_t prefill_bands(pn_qwen35_tt *g)
{
    const uint32_t rows = g->card.hidden[0].tile_rows;

    return rows ? rows : 1u;
}

/**
 * How many bands a pass of `rows` positions actually OCCUPIES, which is what the widened ops
 * must span — not how many the strips have.
 *
 * THE TWO DIFFER ON THE TAIL PASS AND IT IS NOT FREE. A 28-row tail at two bands spans one;
 * telling the matmul to span two would sweep the weights for a whole band nobody wrote, which
 * is measured: `PN_TT_PREFILL_ROWS=32` at two bands ran the shipped answer at 126.576 GB/token
 * against 68.510 — the byte figures DOUBLED for the same 1,629 positions.
 */
static uint32_t prefill_span_for(pn_qwen35_tt *g, int rows)
{
    const uint32_t bands = prefill_bands(g);
    const uint32_t want = (uint32_t)((rows + 31) / 32);

    if (rows <= 0)
        return 1u;
    return want < bands ? want : bands;
}

/** The GDN chunk mixer, asked about a whole PASS rather than a chunk: every band of it has to
 *  be a legal chunk, because every band is a separate descriptor. At one band this is the
 *  single call it replaced. */
static bool prefill_chunk_available(pn_qwen35_tt *g, int rows)
{
    const uint32_t bands = prefill_bands(g);
    int done = 0;

    for (uint32_t band = 0; band < bands && done < rows; band++) {
        const int take = (rows - done) < 32 ? (rows - done) : 32;

        if (!pn_qwen35_tt_gdn_chunk_available(g, take))
            return false;
        done += take;
    }
    return done == rows;
}

/**
 * How many of `n_tokens` (from absolute position `base`) the batched passes take; the rest,
 * at least the last token, run the ordinary forward.
 *
 * THE CHUNK FLOOR CUT. The GDN chunk mixer refuses a band shorter than `conv_taps - 1` (3), and
 * a pass whose LAST band is that short drops the WHOLE pass onto the per-position fallback,
 * which cannot address a second band: the pass fails, the positions it half wrote run per token
 * over it, and the answer is wrong as well as 20x slow — every prompt whose batched length is
 * 1 or 2 past a multiple of 32 (2 in 32 lengths; found 2026-10-07 at 1,795 tokens). Those one
 * or two positions go to the ordinary forward instead. Decided on ABSOLUTE positions, so a
 * resumed prefill (base a multiple of 32) cuts exactly where a cold one of the same prompt does.
 */
static int prefill_batched_rows(pn_qwen35_tt *g, int base, int n_tokens)
{
    int end = base + n_tokens - 1;
    const int residue = end % 32;

    if (end > 32 && residue > 0 && !pn_qwen35_tt_gdn_chunk_available(g, residue) &&
        !prefill_red("prefill_floor_cut_off"))
        end -= residue;
    return end > base ? end - base : 0;
}

static bool prefill_enabled(void)
{
    const char *env = getenv(PN_QWEN35_TT_PREFILL_ENV);

    return !(env && *env && atoi(env) == 0);
}

/**
 * Every strip a batched pass writes into, in ONE list.
 *
 * It was already written out twice — once in `prefill_restage()` and once, in pieces, in the
 * two `prefill_zero()` calls — and the band seek needs a third reader. A strip missing from
 * this list is a strip the seek leaves pointing at band 0 while the rest of the pass has moved
 * to band 1, which is the projections of one chunk mixed with the positions of another and
 * nothing on the card that would say so. So there is one list and three callers.
 */
#define PN_QWEN35_TT_MAX_STRIPS 32u

static size_t prefill_strips_of(pn_qwen35_tt_strips *s, pn_qwen35_tt_attn_strips *a,
                                pn_tt_activation **out, size_t max, bool keep_empty)
{
    pn_tt_activation *all[] = {
        &s->hidden[0], &s->hidden[1], &s->normed, &s->project, &s->ssm_qkv, &s->ssm_z,
        &s->ssm_alpha, &s->ssm_beta, &s->ssm_gated, &s->attn_qg, &s->attn_k, &s->attn_v,
        &s->attn_out, &s->ffn_gate, &s->ffn_up, &s->ffn_swiglu, &s->logits,
        &a->q, &a->q_rot, &a->k_rot, &a->z, &a->gate, &a->gated,
    };
    const size_t count = sizeof(all) / sizeof(all[0]);
    size_t n = 0;

    for (size_t i = 0; i < count && n < max; i++)
        if (keep_empty || all[i]->tile_count)
            out[n++] = all[i];
    return n;
}

static size_t prefill_strips(pn_qwen35_tt *g, pn_tt_activation **out, size_t max)
{
    return prefill_strips_of(&g->card, &g->attn_card, out, max, false);
}

/* pn_tt_device_internal.h PN_TT_STRIP_MAX_TILE_ROWS; pn_tt_activation_alloc_rows refuses more. */
#define QWEN35_TT_TALL_MAX_ROWS 64u

/* One band of one bf16 strip tile column: 32 x 32 x 2 B (pn_tt_device_internal.h
 * PN_TT_TILE_BF16), so a strip of `rows` bands takes `rows` of these a tile column. */
#define QWEN35_TT_BF16_TILE_BYTES 2048u

/* The span the tall strips' mirrors were last keyed for (prefill_pass); 0 = re-key. */
static uint32_t g_prefill_published_span;

/**
 * OUT-PROJECTIONS ON THE GRID (2026-10-27, `PN_TT_OUTPROJ_MIRRORS`, default on; `=0` off):
 * the tall `ssm_gated` and attention `gated` strips are mirrored like `normed`, so ssm_out
 * and attn_output (Kt 128 x Nt 128) leave the 16-member decode group, where all 16 read A
 * from one bank, for the 48-core grid with the M split. Their producers write band by band,
 * so the mirrors are kept by RANGE records (pn_tt_replica_map.h). Decode strips unchanged.
 */
static bool pn_qwen35_tt_outproj_mirrors(void)
{
    static int cached = -1;

    if (cached < 0) {
        const char *env = getenv("PN_TT_OUTPROJ_MIRRORS");

        cached = env && *env ? atoi(env) != 0 : 1;
    }
    return cached != 0;
}

static uint32_t g_card_tall_gated_replicas(const pn_qwen35_tt *g)
{
    return g->card_tall.ssm_gated.replica_count;
}

size_t pn_qwen35_tt_tall_requests(pn_qwen35_tt *g, uint32_t device_replicas,
                                  bool outproj_mirrors, bool fit_policy,
                                  pn_tt_strip_request *out, size_t max)
{
    pn_tt_activation *small[PN_QWEN35_TT_MAX_STRIPS];
    const size_t n = prefill_strips_of(&g->card, &g->attn_card, small,
                                       max < PN_QWEN35_TT_MAX_STRIPS ? max
                                                                     : PN_QWEN35_TT_MAX_STRIPS,
                                       true);

    for (size_t i = 0; i < n; i++) {
        const bool gated = small[i] == &g->card.ssm_gated || small[i] == &g->attn_card.gated;
        /* OPTION C: the joint `attn_qg` strip is written only by the HOST attention block,
         * and the batched passes run only with attention on the card (pn_qwen35_tt_prefill
         * declines otherwise), so its tall copy is never read: 0.047 GiB on Qwen3.8-27B. */
        const bool unread = fit_policy && small[i] == &g->card.attn_qg && g->attn_on_card;

        out[i].tile_count = small[i]->tile_count;
        out[i].replicas = small[i]->replica_count;
        if (small[i] == &g->card.logits || unread)
            out[i].tile_count = 0u;
        /* Never the decode strip's own count: its late mirrors (L2) are decode-only. */
        if (gated)
            out[i].replicas = outproj_mirrors ? device_replicas : 1u;
    }
    return n;
}

/**
 * PHASE G option 3 — PREFILL-ONLY TALL STRIPS (`PN_TT_PREFILL_TILE_ROWS`, a power of two
 * above the decode height, at most QWEN35_TT_TALL_MAX_ROWS; 0/unset = off).
 *
 * G1 made EVERY strip tall, and decode paid -1.4% for it (matmul host ISSUE +3 us, window
 * aims +20 a token) with the cause unidentified. This keeps the decode strips at their own
 * height and gives the batched passes a SECOND set of the same widths and mirror counts at
 * the tall height, swapped into `card` / `attn_card` for the passes and swapped back before
 * the tail forward. The logit strip is not copied: no batched pass runs the LM head.
 *
 * MAIN'S PLACEMENT WHENEVER IT FITS. The set is dry-run first in main's order against the
 * allocator's real per-bank free bytes and strip cursor (exact: every size is a 64 B
 * multiple). It fits -> main's loop, byte for byte. It does not, or the load already took the
 * FIT POLICY (`g->fit_policy`, qwen35moe_tt_fit_plan.h) -> options B and C:
 *   - B, mirrors first (pn_tt_strip_placement.h): a mirrored strip has a copy pinned to every
 *     bank and the allocator cannot reclaim, so the mirrors go before any unmirrored strip
 *     can fill a bank they need; each unmirrored strip then goes to the emptiest bank. On
 *     Qwen3.8-27B at 16k main's order failed with 1.870 GiB free.
 *   - C, no tall `attn_qg` with attention on the card (pn_qwen35_tt_tall_requests).
 */
static bool qwen35_tt_tall_main_fits(const pn_qwen35_tt *g, const pn_tt_strip_request *requests,
                                     size_t n, uint32_t rows)
{
    uint64_t free_bytes[PN_TT_FIT_MAX_BANKS];
    size_t order[PN_QWEN35_TT_MAX_STRIPS];
    const uint32_t banks = pn_tt_device_dram_free_banks(g->device, free_bytes,
                                                        PN_TT_FIT_MAX_BANKS);
    uint32_t cursor = pn_tt_device_strip_cursor(g->device);

    for (size_t i = 0; i < n; i++)
        order[i] = i;
    return banks > 0u &&
           pn_tt_strip_placement_simulate(requests, order, n, PN_TT_STRIP_POLICY_ROUND_ROBIN,
                                          (uint64_t)rows * QWEN35_TT_BF16_TILE_BYTES,
                                          free_bytes, banks, &cursor, NULL) == 0;
}

/** The set's allocation order: main's input order, or mirrors first under the fit policy. */
static size_t qwen35_tt_tall_order(pn_qwen35_tt *g, uint32_t rows, bool outproj_mirrors,
                                   pn_tt_strip_request *requests, size_t *order)
{
    const uint32_t replicas = pn_tt_activation_replica_count(g->device);
    size_t n = pn_qwen35_tt_tall_requests(g, replicas, outproj_mirrors, false, requests,
                                          PN_QWEN35_TT_MAX_STRIPS);

    if (!g->fit_policy && !qwen35_tt_tall_main_fits(g, requests, n, rows)) {
        fprintf(stderr, "qwen35_tt: main's tall-strip placement does not fit this card — FIT "
                        "POLICY on: mirrors first, unmirrored strips in the emptiest bank, no "
                        "tall attn_qg\n");
        g->fit_policy = true;
    }
    if (!g->fit_policy) {
        for (size_t i = 0; i < n; i++)
            order[i] = i;
        return n;
    }
    n = pn_qwen35_tt_tall_requests(g, replicas, outproj_mirrors, true, requests,
                                   PN_QWEN35_TT_MAX_STRIPS);
    return pn_tt_strip_placement_order(requests, n, order);
}

uint32_t pn_qwen35_tt_tall_rows_wanted(void)
{
    /* DEFAULT 64 (G option 3, shipped with on-card dispatch: run_d_ceilings.sh). */
    const char *env = getenv("PN_TT_PREFILL_TILE_ROWS");
    const uint32_t rows = env ? (uint32_t)strtoul(env, NULL, 10) : 64u;

    return rows >= 2u && (rows & (rows - 1u)) == 0u && rows <= QWEN35_TT_TALL_MAX_ROWS ? rows
                                                                                       : 0u;
}

bool pn_qwen35_tt_outproj_mirrors_wanted(void)
{
    return pn_qwen35_tt_outproj_mirrors();
}

int pn_qwen35_tt_prefill_tall_alloc(pn_qwen35_tt *g)
{
    pn_tt_activation *tall[PN_QWEN35_TT_MAX_STRIPS];
    pn_tt_strip_request requests[PN_QWEN35_TT_MAX_STRIPS];
    size_t order[PN_QWEN35_TT_MAX_STRIPS];
    /* DEFAULT 64 (G option 3, shipped with on-card dispatch: run_d_ceilings.sh). */
    const char *env = getenv("PN_TT_PREFILL_TILE_ROWS");
    /* The fit's levers G and F when it picked them (the env pins them there too). */
    const uint32_t rows = env ? (uint32_t)strtoul(env, NULL, 10)
                              : g->fit_levers ? g->fit_tall_rows : 64u;
    const bool outproj_mirrors = g->fit_levers ? g->fit_outproj_mirrors
                                               : pn_qwen35_tt_outproj_mirrors();
    size_t n;

    g->tall_rows = 0u;
    if (rows == 0u)
        return 0;
    if ((rows & (rows - 1u)) != 0u || rows > QWEN35_TT_TALL_MAX_ROWS ||
        rows <= g->card.hidden[0].tile_rows) {
        fprintf(stderr, "qwen35_tt: PN_TT_PREFILL_TILE_ROWS=%u must be a power of two above "
                        "the decode height %u and at most %u — prefill shares the decode "
                        "strips\n", rows, g->card.hidden[0].tile_rows, QWEN35_TT_TALL_MAX_ROWS);
        return 0;
    }
    memset(&g->card_tall, 0, sizeof(g->card_tall));
    memset(&g->attn_card_tall, 0, sizeof(g->attn_card_tall));
    (void)prefill_strips_of(&g->card_tall, &g->attn_card_tall, tall, PN_QWEN35_TT_MAX_STRIPS,
                            true);
    n = qwen35_tt_tall_order(g, rows, outproj_mirrors, requests, order);
    for (size_t step = 0; step < n; step++) {
        const pn_tt_strip_request *strip = &requests[order[step]];
        pn_tt_activation *out = tall[order[step]];

        if (strip->tile_count == 0u)
            continue;
        if ((strip->replicas > 1u || !g->fit_policy
                 ? pn_tt_activation_alloc_replicated_rows(g->device, strip->tile_count,
                                                          strip->replicas, rows, out)
                 : pn_tt_activation_alloc_rows_emptiest(g->device, strip->tile_count, rows,
                                                        out)) != 0) {
            fprintf(stderr, "qwen35_tt: the prefill-only tall strips did not allocate\n");
            return -1;
        }
    }
    if (outproj_mirrors) {
        const pn_tt_activation *gated[2] = { &g->card.ssm_gated, &g->attn_card.gated };

        pn_tt_replica_map_set_ranged(true, gated, 2u);
        fprintf(stderr, "qwen35_tt: OUT-PROJECTIONS ON THE GRID — tall ssm_gated / attention "
                        "gated mirrored x%u (range records); PN_TT_OUTPROJ_MIRRORS=0 off\n",
                g_card_tall_gated_replicas(g));
    }
    g->tall_rows = rows;
    pn_tt_device_set_prefill_tile_rows(g->device, rows);
    fprintf(stderr, "qwen35_tt: prefill-only TALL strips — %u bands for the batched passes, "
                    "decode keeps %u\n", rows, g->card.hidden[0].tile_rows);
    /* S3b: the tall `ffn_swiglu` now exists; put its record on the matmul members so a verify's
     * fused SwiGLU fans out into its mirrors (a no-op with PN_TT_FUSE_SWIGLU off). */
    return pn_tt_replica_map_publish_fused(g->device);
}

static void prefill_tall_swap(pn_qwen35_tt *g)
{
    pn_qwen35_tt_strips s = g->card;
    pn_qwen35_tt_attn_strips a = g->attn_card;

    g->card = g->card_tall;
    g->card_tall = s;
    g->attn_card = g->attn_card_tall;
    g->attn_card_tall = a;
    g->tall_active = !g->tall_active;
}

/* STEP 4 S2: a batched pass whose mixers run the DECODE kernels row by row (the verify). */
static bool g_verify_mode;

static void prefill_tall_enter(pn_qwen35_tt *g)
{
    if (g->tall_rows == 0u || g->tall_active)
        return;
    prefill_tall_swap(g);
    g_prefill_published_span = 0u; /* the first pass keys the tall mirrors */
}

/** Swap the decode strips back and re-key the producers' mirror map for them. RED
 *  `prefill_tall_map_stale`: skip the re-key, so decode's mirrors go stale — must DIFFER. */
static int prefill_tall_exit(pn_qwen35_tt *g)
{
    const pn_tt_activation *mirrored[2];

    if (!g->tall_active)
        return 0;
    prefill_tall_swap(g);
    if (prefill_red("prefill_tall_map_stale"))
        return 0;
    mirrored[0] = &g->card.normed;
    mirrored[1] = &g->card.ffn_swiglu;
    pn_tt_replica_map_set_span(0u);
    return pn_tt_publish_replica_map(g->device, mirrored, 2u, &g->card.ffn_swiglu);
}


/**
 * Point every strip at band `band` and tell every op to span `span` bands.
 *
 * TWO SETTINGS AND NOT ONE, because the ops divide in two. The matmul, the RMS norms and the
 * elementwise family are tile-row-general and take `span`; the per-head norms, RoPE, the KV
 * append, the attention chunk and both GDN mixers have already spent their row axis and can
 * only be run once per band, so they take `band` and a span of one. See
 * `pn_tt_activation_seek()`.
 */
static int prefill_seek(pn_qwen35_tt *g, uint32_t band, uint32_t span)
{
    pn_tt_activation *strips[PN_QWEN35_TT_MAX_STRIPS];
    const size_t n = prefill_strips(g, strips, PN_QWEN35_TT_MAX_STRIPS);

    /* RED PROOF `PN_TT_RED=prefill_band0`: every MIXER pass runs on band 0, so positions
     * 32..63 are mixed from band 0's projections and written over band 0's answer. THE
     * PROJECTIONS ARE UNTOUCHED — the span-2 calls keep their band-0 seek either way — so
     * this arm mutates the seek and nothing else, and it is the ONLY evidence that band 1
     * carries the second half of the pass rather than being computed and discarded. It MUST
     * DIFFER; an IDENTICAL here means a two-band pass is a one-band pass with waste. */
    if (span == 1u && prefill_red("prefill_band0"))
        band = 0u;

    for (size_t i = 0; i < n; i++)
        if (pn_tt_activation_seek(strips[i], band) != 0 ||
            pn_tt_activation_span(strips[i], span) != 0)
            return -1;
    return 0;
}

/** Rows of one batched pass: the tall strips' height while they are swapped in or about to
 *  be, the decode strips' otherwise, narrowed by `PN_TT_PREFILL_ROWS`. */
static int prefill_rows_wanted(pn_qwen35_tt *g)
{
    const char *env = getenv(PN_QWEN35_TT_PREFILL_ROWS_ENV);
    const uint32_t bands = (g->tall_rows && !g->tall_active) ? g->tall_rows : prefill_bands(g);
    const int ceiling = (int)(bands * 32u);
    int rows = env && *env ? atoi(env) : ceiling;

    if (rows < 1) rows = 1;
    /* THE CEILING IS THE ALLOCATION, not the tile. A pass may carry as many positions as the
     * strips have bands; asking for more would run the mixers over a band nothing wrote. */
    if (rows > ceiling) rows = ceiling;
    return rows;
}

/** The strip's full width in floats. Transfers move whole tiles, so the padding rides along
 *  and the round trip is exact rather than nearly so. */
static uint32_t prefill_width(const pn_tt_activation *strip)
{
    return strip->tile_count * 32u;
}

static double prefill_now(void)
{
    return pn_time_ms() / 1000.0;
}

/** batched + sequential + transfer so far (cumulative over the process); 0 before the first
 *  pass allocated the scratch. The wall regions report their movement. */
static double prefill_accounted(const pn_qwen35_tt *g)
{
    const struct pn_qwen35_tt_batch *b = g->batch;

    return b ? b->batched_seconds + b->sequential_seconds + b->transfer_seconds : 0.0;
}

#define PREFILL_WALL_OPEN(g, region)  pn_prefill_wall_open((region), prefill_accounted(g))
#define PREFILL_WALL_CLOSE(g, region) pn_prefill_wall_close((region), prefill_accounted(g))

/** The `batched` bracket sites, in kPrefillBatchedNames order. */
static const char *const kPrefillBatchedNames[7] = { "ssm-in-proj", "ssm-out-proj",
                                                     "attn-proj", "attn-out-proj", "ffn",
                                                     "attn-norm", "post-norm" };

static void prefill_batched_add(struct pn_qwen35_tt_batch *b, unsigned site, double seconds)
{
    b->batched_seconds += seconds;
    if (site < 7u)
        b->batched_site_seconds[site] += seconds;
}

/** Every card clock at the open of a `sequential` bracket. */
typedef struct {
    const char *names[PN_TT_CARD_CLOCKS_MAX];
    double seconds[PN_TT_CARD_CLOCKS_MAX];
    unsigned count;
    double mark;
} prefill_sequential_bracket;

/** Open a `sequential` bracket: every card clock, then the wall mark. */
/**
 * PHASE W chain step — `PN_TT_PREFILL_STREAM` (default on, `=0` off): a layer's band loop runs as STREAMED puller
 * segments (pn_tt_dispatch_stream_begin, the S2d machinery): the host's queue writes become
 * puller entries and its completion polls are answered at once while the puller awaits the
 * same counts on the card, so the chain stops paying a host round trip per op. The alpha/beta
 * reads are on the reader thread's own window bank and are not boundaries. While streaming,
 * the sequential bracket does not snapshot card clocks (each would be a boundary).
 */
static bool g_prefill_streaming;

static bool prefill_stream_wanted(void)
{
    static int cached = -1;

    if (cached < 0) {
        const char *env = getenv("PN_TT_PREFILL_STREAM");

        /* DEFAULT ON (run_w_stream.sh 2026-10-08: serve 1,630 TTFT 3.24 -> 2.98 s, IDENTICAL). */
        cached = env && *env ? atoi(env) != 0 : 1;
    }
    return cached != 0;
}

static void prefill_stream_open(pn_qwen35_tt *g)
{
    g_prefill_streaming = prefill_stream_wanted() && !g_verify_mode &&
                          pn_tt_dispatch_stream_begin(g->device);
}

/** @return 0, or -1 when a streamed segment failed on the card (the work is then untrusted). */
static int prefill_stream_close(void)
{
    const bool was = g_prefill_streaming;

    g_prefill_streaming = false;
    return (!was || pn_tt_dispatch_stream_end()) ? 0 : -1;
}

static void prefill_sequential_open(const pn_qwen35_tt *g, prefill_sequential_bracket *open)
{
    open->count = g_prefill_streaming ? 0u
                                      : pn_tt_device_card_clocks(g->device, open->names,
                                                                 open->seconds,
                                                                 PN_TT_CARD_CLOCKS_MAX);
    open->mark = prefill_now();
}

/** Add `seconds` to the entry called `name`, BY NAME — split members repeat member 0's name
 *  and sum into one family row, and an index would not survive the table changing shape
 *  between two brackets. */
static void prefill_sequential_book(struct pn_qwen35_tt_batch *b, const char *name,
                                    double seconds)
{
    unsigned index;

    for (index = 0; index < b->sequential_clock_count; index++)
        if (strcmp(b->sequential_clock_names[index], name) == 0)
            break;
    if (index == b->sequential_clock_count) {
        if (index >= PN_TT_CARD_CLOCKS_MAX) {
            b->sequential_unmatched_seconds += seconds;
            return;
        }
        b->sequential_clock_names[index] = name;
        b->sequential_clock_seconds[index] = 0.0;
        b->sequential_clock_count++;
    }
    b->sequential_clock_seconds[index] += seconds;
}

/**
 * Close it: the wall into `sequential_seconds`, each clock's movement into its entry.
 *
 * A residency that ARMS inside the bracket changes the table's shape, and a clock matched by
 * position would then be subtracted from its neighbour. So the snapshots must agree entry by
 * entry on the name; when they do not, the whole bracket goes to `unmatched` — counted, never
 * guessed at.
 */
static void prefill_sequential_close(pn_qwen35_tt *g, const prefill_sequential_bracket *open)
{
    struct pn_qwen35_tt_batch *b = g->batch;
    const double wall = prefill_now() - open->mark;
    const char *names[PN_TT_CARD_CLOCKS_MAX];
    double after[PN_TT_CARD_CLOCKS_MAX];
    unsigned count, index;

    b->sequential_seconds += wall;
    if (g_prefill_streaming)
        return;
    count = pn_tt_device_card_clocks(g->device, names, after, PN_TT_CARD_CLOCKS_MAX);
    for (index = 0; index < count && count == open->count; index++)
        if (names[index] != open->names[index])
            break;
    if (count != open->count || index != count) {
        b->sequential_unmatched_seconds += wall;
        return;
    }
    for (index = 0; index < count; index++)
        prefill_sequential_book(b, names[index], after[index] - open->seconds[index]);
}

void pn_qwen35_tt_prefill_release(pn_qwen35_tt *g)
{
    struct pn_qwen35_tt_batch *b;

    if (!g || !g->batch)
        return;
    b = g->batch;
    free(b->hidden); free(b->ssm_qkv); free(b->ssm_z); free(b->ssm_alpha); free(b->ssm_beta);
    free(b->ssm_gated); free(b->attn_q); free(b->attn_z); free(b->attn_k); free(b->attn_v);
    free(b->attn_gated);
    free(b);
    g->batch = NULL;
}

static int prefill_alloc(pn_qwen35_tt *g, const pn_qwen35moe_config *cfg, int rows)
{
    const pn_qwen35_tt_strips *s = &g->card;
    const pn_qwen35_tt_attn_strips *a = &g->attn_card;
    struct pn_qwen35_tt_batch *b;

    if (g->batch && g->batch->rows >= rows)
        return 0;
    pn_qwen35_tt_prefill_release(g);
    b = calloc(1, sizeof(*b));
    if (!b)
        return -1;
    b->hidden     = calloc((size_t)rows * (size_t)cfg->n_embd, sizeof(float));
    b->ssm_qkv    = calloc((size_t)rows * prefill_width(&s->ssm_qkv), sizeof(float));
    b->ssm_z      = calloc((size_t)rows * prefill_width(&s->ssm_z), sizeof(float));
    b->ssm_alpha  = calloc((size_t)rows * prefill_width(&s->ssm_alpha), sizeof(float));
    b->ssm_beta   = calloc((size_t)rows * prefill_width(&s->ssm_beta), sizeof(float));
    b->ssm_gated  = calloc((size_t)rows * prefill_width(&s->ssm_gated), sizeof(float));
    b->attn_q     = calloc((size_t)rows * prefill_width(&a->q), sizeof(float));
    b->attn_z     = calloc((size_t)rows * prefill_width(&a->z), sizeof(float));
    b->attn_k     = calloc((size_t)rows * prefill_width(&s->attn_k), sizeof(float));
    b->attn_v     = calloc((size_t)rows * prefill_width(&s->attn_v), sizeof(float));
    b->attn_gated = calloc((size_t)rows * prefill_width(&a->gated), sizeof(float));
    if (!b->hidden || !b->ssm_qkv || !b->ssm_z || !b->ssm_alpha || !b->ssm_beta ||
        !b->ssm_gated || !b->attn_q || !b->attn_z || !b->attn_k || !b->attn_v ||
        !b->attn_gated) {
        g->batch = b;
        pn_qwen35_tt_prefill_release(g);
        return -1;
    }
    b->rows = rows;
    g->batch = b;
    return 0;
}

/** Download `rows` whole tile-rows of a strip into a host buffer, timed as TRANSFER. */
static int prefill_pull(pn_qwen35_tt *g, const pn_tt_activation *strip, float *host, int rows)
{
    struct pn_qwen35_tt_batch *b = g->batch;
    double mark = prefill_now();
    int rc = pn_tt_activation_download_rows(g->device, strip, host, prefill_width(strip),
                                            (uint32_t)rows);

    b->transfer_seconds += prefill_now() - mark;
    return rc;
}

/** Put ONE host row back as the strip's logical row 0 — the row select, done through the
 *  link because the card has no op that can do it. */
static int prefill_push_row(pn_qwen35_tt *g, const pn_tt_activation *strip, const float *host,
                            int row)
{
    struct pn_qwen35_tt_batch *b = g->batch;
    uint32_t width = prefill_width(strip);
    int take = prefill_red("prefill_rowfill") ? 0 : row;
    double mark = prefill_now();
    int rc = pn_tt_activation_upload(g->device, strip, host + (size_t)take * width, width);

    b->transfer_seconds += prefill_now() - mark;
    return rc;
}

/** Read a mixer's row-0 answer into row `row` of a host buffer. */
static int prefill_pull_row(pn_qwen35_tt *g, const pn_tt_activation *strip, float *host,
                            int row)
{
    struct pn_qwen35_tt_batch *b = g->batch;
    uint32_t width = prefill_width(strip);
    double mark = prefill_now();
    int rc = pn_tt_activation_download(g->device, strip, host + (size_t)row * width, width);

    b->transfer_seconds += prefill_now() - mark;
    return rc;
}

/** Put all `rows` host rows up as a batch, so the next projection serves every one of them. */
static int prefill_push(pn_qwen35_tt *g, const pn_tt_activation *strip, const float *host,
                        int rows)
{
    struct pn_qwen35_tt_batch *b = g->batch;
    double mark = prefill_now();
    int rc = pn_tt_activation_upload_rows(g->device, strip, host, prefill_width(strip),
                                          (uint32_t)rows);

    b->transfer_seconds += prefill_now() - mark;
    return rc;
}

/**
 * ZERO A MIXER'S WORKING SET, and this is a CORRECTNESS step rather than hygiene.
 *
 * `pn_tt_activation_upload()` writes logical row 0 and nothing else, so after a 32-row pull
 * the row select leaves rows 1..31 carrying the OTHER 31 positions' projection outputs. In
 * decode those rows are the tile quantum's zeros, and every claim that makes a mixer's row-0
 * answer independent of them rests on that: an op that mixes across rows anywhere — the delta
 * rule's outer product, a broadcast, the K/V append — would read live neighbours instead of
 * zeros and there is nothing on the card that would say so.
 *
 * A whole-strip zero costs one write of the strip, in the 200-390 MB/s direction, ONCE a
 * layer rather than once a row. That is ~0.08 ms a position against the 3+ ms a position the
 * row select already pays, so restoring the invariant is free at this rung and the arm does
 * not have to depend on an unproven independence.
 */
static int prefill_zero(pn_qwen35_tt *g, const pn_tt_activation *const *strips, size_t n)
{
    struct pn_qwen35_tt_batch *b = g->batch;
    double mark = prefill_now();

    for (size_t i = 0; i < n; i++)
        if (strips[i]->tile_count &&
            pn_tt_activation_zero_bands(g->device, strips[i],
                                        prefill_bands(g) > 4u ? b->max_span : 0u) != 0)
            return -1;
    b->transfer_seconds += prefill_now() - mark;
    return 0;
}

/** hidden = hidden + project, on the card, into the spare strip. Per tile, so all 32 rows.
 *  When a verify's out-projection already did that (S3a, prefill_project_out), only the flip. */
static int prefill_residual(pn_qwen35_tt *g)
{
    int rc;

    if (g->residual_fused) {
        g->residual_fused = 0;
        g->card.slot ^= 1;
        return 0;
    }
    PREFILL_WALL_OPEN(g, PN_PREFILL_WALL_RESIDUAL);
    rc = pn_tt_add(g->device, &g->card.hidden[g->card.slot], &g->card.project,
                   &g->card.hidden[g->card.slot ^ 1]);
    PREFILL_WALL_CLOSE(g, PN_PREFILL_WALL_RESIDUAL);
    if (rc != 0)
        return -1;
    g->card.slot ^= 1;
    return 0;
}

/** An out-projection into `project`, which prefill_residual() adds next. S3a: the VERIFY rows
 *  (MTP k1's verify runs this pass) take the fused residual add when it fits; a prefill pass
 *  proper keeps the projection and the add. */
static int prefill_project_out(pn_qwen35_tt *g, const pn_tt_weight *w, const pn_tt_activation *in)
{
    if (g_verify_mode)
        return pn_qwen35_tt_project_out(g, w, in);
    return pn_qwen35_tt_project(g, w, in, &g->card.project, true);
}

/**
 * One SSM layer over `rows` positions: four projections ONCE, then the recurrence `rows`
 * times through the link, then the output projection ONCE.
 */
/** Every exit from the chunk band loop: nothing the reader thread or a detached grid
 *  projection still owes may land after the loop has returned. */
static void prefill_chain_settle(pn_qwen35_tt *g)
{
    g->gdn_chunk_defer_gate = false;
    g->gdn_card_gate = false;
    g->gdn_table_gate = false;
    (void)pn_qwen35_tt_gdn_pipe_drain(g);
    g->gdn_pipe = false;
    g->gdn_pipe2 = false;
    (void)pn_tt_eltwise_group_settle(g->device);
    (void)prefill_stream_close();
    pn_qwen35_tt_overlap_abort(g);
    pn_qwen35_tt_ab_reader_abandon(g);
    pn_qwen35_tt_overlap_forget();
}

/** Close the loop's stream, its wait booked as SEQUENTIAL (the card running the streamed
 *  chain). @return 0, or -1 when a segment failed. */
static int prefill_stream_finish(pn_qwen35_tt *g)
{
    const double at = prefill_now();
    const int rc = prefill_stream_close();

    g->batch->sequential_seconds += prefill_now() - at;
    return rc;
}

static bool prefill_env_on(const char *name);

/** PM 7 ITEM 7 step 1: `PN_TT_GDN_PIPE=0` runs the band chain serially. */
static bool prefill_gdn_pipe_wanted(void)
{
    const char *env = getenv("PN_TT_GDN_PIPE");

    return !(env && *env && atoi(env) == 0);
}

/** Prefill lane PIPE 2 — band b's DELTA left in flight under band b+1's widen + CONV, on two
 *  buffer parities (qwen35moe_gdn_strip_tt.c). DEFAULT (run_pg_pipe2.sh, 2026-10-22: serve warm
 *  1,630 TTFT 2,258 -> 2,196 ms, 730 1,152 -> 1,127, IDENTICAL, reds DIFFER); `PN_TT_GDN_PIPE=1`
 *  is the PM 7 pipeline, `=0` the serial chain. */
static bool prefill_gdn_pipe2_wanted(void)
{
    const char *env = getenv("PN_TT_GDN_PIPE");

    return !(env && *env && atoi(env) != 2);
}

/** The verify's deferred GDN rows (`PN_TT_SPEC_DEFER=0` off) — gdn_strip_rows' own reading. */
static bool prefill_spec_defer(void)
{
    const char *env = getenv("PN_TT_SPEC_DEFER");

    return !env || atoi(env) != 0;
}

/** ITEM 6: `PN_TT_GDNGATE_DECODE=0` — det gates on the host from read alpha/beta. */
static bool prefill_card_decode_gates(void)
{
    const char *env = getenv("PN_TT_GDNGATE_DECODE");

    return !(env && *env && atoi(env) == 0);
}

/**
 * PM 7 ITEM 5: put layer `ssm_index`'s gate blocks on the card — already issued by the W2
 * wavefront for a prepared layer, else issued here for every band (not awaited; the band loop
 * settles them once). @return true when the layer runs on card gates, false = the host path.
 */
static bool prefill_card_gates_begin(pn_qwen35_tt *g, int ssm_index, int rows,
                                     bool gates_prepared)
{
    g->gdn_table_gate = false;
    if (g_verify_mode || !pn_qwen35_tt_gdn_card_gate_possible())
        return false;
    /* THE EXACT CARD GATES: nothing to issue — each DELTA looks its gates up from the strips.
     * The layer's table block is bound to the strips here, before the stream opens. */
    if (!g->gdn_det_gates && pn_tt_gdnc_table_ready(g->device)) {
        g->gdn_table_gate = pn_tt_gdnc_table_bind(g->device, (uint32_t)ssm_index,
                                                  &g->card.ssm_alpha, &g->card.ssm_beta) == 0;
        return g->gdn_table_gate;
    }
    if (!pn_tt_gdngate_enabled(g->device))
        return false;
    if (!gates_prepared &&
        pn_tt_gdngate_issue(g->device, (uint32_t)ssm_index, &g->card.ssm_alpha,
                            &g->card.ssm_beta, (uint32_t)rows,
                            pn_qwen35_tt_gdn_gate_sub_rows()) != 0)
        return false;
    /* `PN_TT_GDNGATE_VERIFY=1` (gates only): every block read back and compared word for word
     * with the host build of it from host-read alpha/beta — the card == host proof on the
     * model's own data. Reads the link: never in a timing arm. */
    if (prefill_env_on("PN_TT_GDNGATE_VERIFY")) {
        const long bad = pn_tt_gdngate_settle(g->device) == 0
                             ? pn_tt_gdngate_verify_layer(g->device, (uint32_t)ssm_index,
                                                          &g->card.ssm_alpha, &g->card.ssm_beta,
                                                          (uint32_t)rows,
                                                          pn_qwen35_tt_gdn_gate_sub_rows())
                             : -1;

        fprintf(stderr, "qwen35_tt: gdngate VERIFY layer %d — %ld word(s) differ over %d rows\n",
                ssm_index, bad, rows);
    }
    return true;
}

/** An env knob that is on when set to anything but 0. */
static bool prefill_env_on(const char *name)
{
    const char *value = getenv(name);

    return value && *value && atoi(value) != 0;
}

/**
 * Returns 0, -1, or 1 when the layer's out-projection, residuals, post-norm and FFN were
 * already run inside the band loop (PHASE W2 overlap) — the caller then skips them.
 */
static int prefill_ssm_layer(pn_qwen35_tt *g, pn_qwen35moe_model *m,
                             const pn_qwen35_tt_layer *d, const pn_qwen35moe_config *cfg,
                             int ssm_index, int base_pos, int rows,
                             const pn_qwen35_tt_layer *next_d, int next_ssm_index)
{
    struct pn_qwen35_tt_batch *b = g->batch;
    pn_qwen35_tt_strips *s = &g->card;
    const pn_qwen35moe_ssm_layer *l = pn_qwen35moe_model_ssm_layer(m, ssm_index);
    uint32_t alpha_w = prefill_width(&s->ssm_alpha);
    uint32_t beta_w = prefill_width(&s->ssm_beta);
    /* PHASE W WAVEFRONT: the previous layer's super-band jobs already ran this layer's
     * attn-norm, in-proj and read posts — skip them. */
    const bool prepared = !g_verify_mode &&
                          pn_qwen35_tt_overlap_prepared(ssm_index, prefill_span_for(g, rows));
    /* Read BEFORE pn_qwen35_tt_overlap_forget() below: the wavefront's own record of having
     * issued this layer's gate blocks (run_pm7_item6_diag.sh: read after it, every layer
     * re-issued at its start — 24 of 88 issues, 0.146 s of settles). */
    const bool gates_prepared = prepared &&
                                pn_qwen35_tt_overlap_gates_prepared(ssm_index,
                                                                    prefill_span_for(g, rows));
    const pn_qwen35_tt_overlap_next next = {
        next_d, next_ssm_index, rows, b->ssm_alpha, b->ssm_beta, alpha_w, beta_w,
    };
    double mark;

    if (!l)
        return -1;
    pn_qwen35_tt_overlap_forget();
    mark = prefill_now();
    /* O.0: alpha/beta may go to the grid by rows — no stream is open here, so the flush below
     * is a real host wait and any reader posted after it reads finished rows. */
    pn_tt_matmul_grid_unsplit_allow(g->device, !g_prefill_streaming);
    if (!prepared &&
        (pn_qwen35_tt_project(g, &d->attn_qkv, &s->normed, &s->ssm_qkv, false) != 0 ||
         pn_qwen35_tt_project(g, &d->attn_gate, &s->normed, &s->ssm_z, false) != 0 ||
         pn_qwen35_tt_project(g, &d->ssm_alpha, &s->normed, &s->ssm_alpha, false) != 0 ||
         pn_qwen35_tt_project(g, &d->ssm_beta, &s->normed, &s->ssm_beta, false) != 0 ||
         pn_tt_matvec_flush(g->device) != 0)) {
        pn_tt_matmul_grid_unsplit_allow(g->device, false);
        return -1;
    }
    pn_tt_matmul_grid_unsplit_allow(g->device, false);
    prefill_batched_add(b, 0, prefill_now() - mark); /* ssm-in-proj */

    /* STEP 4 S2: THE VERIFY. Row r of the pass goes through the DECODE GDN kernels exactly as
     * a single-position forward would: its qkv/z rows are copied on the card into row 0 of the
     * decode strips (rows 1-31 there are the decode invariant's zeros), the decode strip layer
     * runs, and its gated row 0 is copied back into row r. S0 proved this arithmetic
     * bit-identical to per-token. Row r writes its new carried state into snapshot slot r and
     * the two trade places (pn_tt_gdn_layer_swap): slot r keeps the state BEFORE row r for a
     * rollback, with no copy. */
    if (g_verify_mode) {
        /* The projections land first, so the row timer holds only the row loop. */
        const double rows_started = pn_tt_matvec_flush(g->device) == 0 ? pn_time_ms() : -1.0;

        if (rows_started < 0.0)
            return -1;
        /* PM 6: the reads go to the reader (gated on the card running these projections) and
         * the stream keeps flowing; gdn_strip_rows waits for them before the host math. The
         * S2 row-copy path still reads inline. */
        /* ITEM 6: det gates computed on the card for a deferred verify — no reads at all. */
        const bool card = !getenv("PN_TT_SPEC_ROWCOPY") && prefill_spec_defer() &&
                          ((g->gdn_det_gates && prefill_card_decode_gates()) ||
                           pn_qwen35_tt_table_decode_gates(g, ssm_index));

        if (!card &&
            (getenv("PN_TT_SPEC_ROWCOPY") ||
             !pn_qwen35_tt_ab_verify_post(g, rows, b->ssm_alpha, alpha_w, b->ssm_beta,
                                          beta_w)) &&
            (prefill_pull(g, &s->ssm_alpha, b->ssm_alpha, rows) != 0 ||
             prefill_pull(g, &s->ssm_beta, b->ssm_beta, rows) != 0))
            return -1;
        g->spec_gdn_part_ms[0] += pn_time_ms() - rows_started;
        /* S2c (`PN_TT_SPEC_ROWCOPY=1` keeps the S2 form: decode strips + puller row copies):
         * the decode recurrences with the kernels' ROW SELECT on the verify strips, the
         * row-wise ops once over all rows — no row copies, no per-row flushes. */
        if (!getenv("PN_TT_SPEC_ROWCOPY")) {
            if (pn_qwen35_tt_gdn_strip_rows(g, l, cfg, ssm_index, rows, &s->ssm_qkv, &s->ssm_z,
                                            card ? NULL : b->ssm_beta, beta_w,
                                            card ? NULL : b->ssm_alpha, alpha_w,
                                            &s->ssm_gated) != 0 ||
                pn_tt_matvec_flush(g->device) != 0)
                return -1;
            g->spec_gdn_part_ms[4] += pn_time_ms() - g->spec_gdn_mark_ms;
        } else
        for (int r = 0; r < rows; r++) {
            pn_qwen35_tt_strips *v = &g->card_tall; /* the VERIFY rows: swapped out below */
            int rc;

            prefill_tall_swap(g); /* g->card = decode strips, g->card_tall = verify rows */
            v = &g->card_tall;
            rc = (pn_tt_row_copy(g->device, &v->ssm_qkv, (uint32_t)r, &g->card.ssm_qkv, 0u) != 0 ||
                  pn_tt_row_copy(g->device, &v->ssm_z, (uint32_t)r, &g->card.ssm_z, 0u) != 0)
                     ? -1 : 0;
            /* Row r's new state goes to the slot the verify map names (legacy: slot r), then
             * trades places with the live one. */
            if (rc == 0)
                rc = pn_tt_gdn_verify_row_out(g->device, (uint32_t)rows, (uint32_t)r);
            if (rc == 0) {
                pn_tt_device_set_single_row(g->device, true);
                rc = pn_qwen35_tt_gdn_strip_layer(g, l, cfg, ssm_index, &g->card.ssm_qkv,
                                                  &g->card.ssm_z,
                                                  b->ssm_beta + (size_t)r * beta_w,
                                                  b->ssm_alpha + (size_t)r * alpha_w,
                                                  &g->card.ssm_gated);
                pn_tt_device_set_single_row(g->device, false);
            }
            if (rc == 0 && pn_tt_matvec_flush(g->device) != 0)
                rc = -1;
            if (pn_tt_gdn_state_out(g->device, -1) != 0 ||
                (rc == 0 && pn_tt_gdn_verify_row_trade(g->device, (uint32_t)ssm_index,
                                                       (uint32_t)rows, (uint32_t)r, true,
                                                       true) != 0))
                rc = -1;
            if (rc == 0)
                rc = pn_tt_row_copy(g->device, &g->card.ssm_gated, 0u, &v->ssm_gated,
                                    (uint32_t)r);
            prefill_tall_swap(g);
            if (rc != 0)
                return -1;
        }
        g->spec_ssm_rows_ms += pn_time_ms() - rows_started;
        mark = prefill_now();
        if (prefill_project_out(g, &d->ssm_out, &s->ssm_gated) != 0)
            return -1;
        prefill_batched_add(b, 1, prefill_now() - mark); /* ssm-out-proj */
        return 0;
    }

    /* THE CHUNK PATH: THE 24 SSM LAYERS STOP ROW-SELECTING THROUGH THE LINK.
     *
     * `ssm_qkv` and `ssm_z` never come down and never go back up — the two chunkwise mixers
     * read the strip the projections already wrote, with row t carrying position `base+t`, and
     * advance the SAME conv window and the SAME recurrent state the decode family owns. What
     * still crosses is `alpha` and `beta` DOWN (two tiles) and the per-head decay tiles and
     * gate block UP (128 + 20 KiB a layer a chunk), because `beta = sigmoid(.)` and
     * `decay = exp(.)` need a `log` the SFPU does not have, and because the decay-ratio matrix
     * is built on the host as a product recurrence ON PURPOSE — a card-built one costs the
     * whole precision advantage of going chunkwise. That was 0.2063 s/row of transfer.
     *
     * THE FLOOR IS THE CONV WINDOW'S. A chunk shorter than `taps-1` cannot REPLACE the window,
     * only shift it, and both the host encoder and the device refuse — so those positions go
     * through the per-position path below, which is the same arithmetic on the same bytes and
     * leaves the window and the state exactly where a chunk would have. That is the ONE seam
     * in this file and it costs nothing, because the two families share the buffers.
     *
     * The two red arms about the per-position LOOP select the fallback path, because a chunk
     * has no such loop; the chunk path's own carry arms are `gdnc_window_wipe` and
     * `gdnc_state_wipe` in pn_tt_gdnc.c, and both are INVISIBLE IN CHUNK 0. */
    if (prefill_chunk_available(g, rows) && !prefill_red("prefill_causal") &&
        !prefill_red("prefill_rowfill")) {
        const uint32_t bands = prefill_span_for(g, rows);
        /* PHASE W: every band's alpha/beta read posted to the READER THREAD now (all of them
         * are projected); the band loop only waits on its own. Off: the prefetch below. */
        /* PM 7 ITEM 5: the gates on the card — no alpha/beta crosses the link at all. */
        const bool card_gates = prefill_card_gates_begin(g, ssm_index, rows, gates_prepared);
        const bool reader = !card_gates &&
                            ((prepared && pn_qwen35_tt_ab_reader_promote(ssm_index)) ||
                             pn_qwen35_tt_ab_reader_post(g, rows, b->ssm_alpha, alpha_w,
                                                         b->ssm_beta, beta_w));
        bool overlap = pn_qwen35_tt_overlap_layer(g, ssm_index, bands);

        /* The layer's gate blocks, settled ONCE before the first band, by the HOST and before
         * the stream opens: as a puller AWAIT inside the stream it stalled ~8.5 s a layer
         * (run_pm7_item5_gates.sh round 2: TTFT 204 s), while a host await is the gate time
         * itself — and a prepared layer's blocks are long done. */
        if (card_gates && !g->gdn_table_gate && pn_tt_gdngate_settle(g->device) != 0)
            return -1;
        /* Prefill lane: each DELTA member's heads of this layer's state in its own bank for
         * the band loop (`PN_TT_GDNC_STATE_BANKS`; a card copy, so before the stream opens). */
        if (pn_tt_gdnc_state_split_enter(g->device, (uint32_t)ssm_index, bands) != 0)
            return -1;
        prefill_stream_open(g);
        g->gdn_card_gate = card_gates;
        /* PM 7 ITEM 7 step 1: the pipelined band (`PN_TT_GDN_PIPE=0` off). */
        g->gdn_pipe = card_gates && g_prefill_streaming && prefill_gdn_pipe_wanted();
        g->gdn_pipe2 = g->gdn_pipe && prefill_gdn_pipe2_wanted();
        /* O.1: the super-band jobs' steps placed behind the chain's gates (PIPE 2 trails 2). */
        if (overlap && pn_qwen35_tt_overlap_layer_begin(g, d, bands,
                                                        g->gdn_pipe2 ? 2u : g->gdn_pipe ? 1u : 0u,
                                                        &next, &s->ssm_gated, &d->ssm_out) != 0)
            overlap = false;
        pn_qwen35_tt_overlap_set_card_gates(card_gates);
        /* PM 7: streamed, each band's output gate is left pending (its AWAIT moves to the next
         * band's narrow, or to the settles below); `PN_TT_ELTWISE_DEFER=0` off. */
        g->gdn_chunk_defer_gate = g_prefill_streaming;
        int done = 0;

        /* ONE BAND AT A TIME, IN POSITION ORDER, and the order is the correctness claim: the
         * conv window and the `[S,S]` recurrent state are card-resident and carry from one
         * descriptor to the next, so band 1's mixer must see band 0's already run. That is
         * also why the GDN mixers cannot simply be widened the way the projections were —
         * `chunk_rows > 32` is refused by name in five places because the intra-chunk decay
         * matrix is ONE 32x32 tile, and above 32 the DELTA term is O(C^2). The band loop runs
         * the mixer the shape it already has, over different bytes. */
        for (uint32_t band = 0; band < bands && done < rows; band++) {
            const int take = (rows - done) < 32 ? (rows - done) : 32;
            int rc;

            if (prefill_seek(g, band, 1u) != 0) {
                prefill_chain_settle(g);
                return -1;
            }
            /* Band b's reads: the reader thread's, or done under band b-1's mixers (the
             * prefetch), or inline. A failed reader read is re-read inline. None at all when
             * the card computed the gates. */
            g->gdn_card_gate_band = band;
            if (card_gates) {
                /* nothing to read */
            } else if (reader) {
                const double waited = prefill_now();
                const int took = pn_qwen35_tt_ab_reader_take(g, (int)band);

                b->transfer_seconds += prefill_now() - waited;
                if (took != 0 &&
                    (prefill_pull(g, &s->ssm_alpha, b->ssm_alpha + (size_t)done * alpha_w,
                                  take) != 0 ||
                     prefill_pull(g, &s->ssm_beta, b->ssm_beta + (size_t)done * beta_w,
                                  take) != 0)) {
                    prefill_chain_settle(g);
                    return -1;
                }
            } else if ((!pn_qwen35_tt_ab_prefetched(g, ssm_index, (int)band, false) &&
                 prefill_pull(g, &s->ssm_alpha, b->ssm_alpha + (size_t)done * alpha_w, take) != 0) ||
                (!pn_qwen35_tt_ab_prefetched(g, ssm_index, (int)band, true) &&
                 prefill_pull(g, &s->ssm_beta, b->ssm_beta + (size_t)done * beta_w, take) != 0))
                return -1;
            pn_qwen35_tt_ab_prefetch_reset();
            if (!reader && !card_gates && band + 1u < bands && done + take < rows) {
                const int next = (rows - done - take) < 32 ? (rows - done - take) : 32;

                pn_qwen35_tt_ab_prefetch_arm(g, ssm_index, (int)band + 1, next,
                                             b->ssm_alpha + (size_t)(done + take) * alpha_w,
                                             alpha_w,
                                             b->ssm_beta + (size_t)(done + take) * beta_w,
                                             beta_w);
            }
            prefill_sequential_bracket bracket;
            prefill_sequential_open(g, &bracket);
            rc = pn_qwen35_tt_gdn_strip_chunk(g, l, cfg, ssm_index, &s->ssm_qkv, &s->ssm_z,
                                              b->ssm_beta + (size_t)done * beta_w, beta_w,
                                              b->ssm_alpha + (size_t)done * alpha_w, alpha_w,
                                              take, &s->ssm_gated);
            prefill_sequential_close(g, &bracket);
            pn_qwen35_tt_ab_prefetch_settle(g);
            if (rc != 0) {
                prefill_chain_settle(g);
                return -1;
            }
            done += take;
            /* MULTI-TURN: this layer's state is the state at `base_pos + done`. The capture is
             * a puller COPY, which refuses mid-stream: the stream closes around it. */
            if (g_prefill_streaming && g->ckpt.pending_at == base_pos + done) {
                if (pn_qwen35_tt_gdn_pipe_drain(g) != 0 ||
                    pn_tt_eltwise_group_settle(g->device) != 0 || prefill_stream_finish(g) != 0 ||
                    pn_tt_gdnc_state_split_sync(g->device, (uint32_t)ssm_index) != 0) {
                    prefill_chain_settle(g);
                    return -1;
                }
                pn_qwen35_tt_ckpt_band_done(g, ssm_index, base_pos + done);
                prefill_stream_open(g);
            } else {
                if (!g_prefill_streaming && g->ckpt.pending_at == base_pos + done &&
                    pn_tt_gdnc_state_split_sync(g->device, (uint32_t)ssm_index) != 0) {
                    prefill_chain_settle(g);
                    return -1;
                }
                pn_qwen35_tt_ckpt_band_done(g, ssm_index, base_pos + done);
            }
            /* PHASE W2 / O.1: the super-band jobs' steps due after this band (plan in
             * qwen35moe_prefill_ov_plan.h). Streamed, each one's waits are puller AWAITs, and
             * the plan places them bands after the grid work they wait on. */
            if (overlap) {
                rc = pn_qwen35_tt_overlap_tick(g, band);
                prefill_batched_add(b, 4, pn_qwen35_tt_overlap_take_seconds());
                if (rc != 0) {
                    prefill_chain_settle(g);
                    return -1;
                }
            }
        }
        pn_qwen35_tt_ab_prefetch_reset();
        g->gdn_chunk_defer_gate = false;
        g->gdn_card_gate = false;
        g->gdn_table_gate = false;
        if (pn_qwen35_tt_gdn_pipe_drain(g) != 0 || pn_tt_eltwise_group_settle(g->device) != 0) {
            g->gdn_pipe = false;
            prefill_chain_settle(g);
            return -1;
        }
        /* O.1: the layer's last jobs (the tail of the plan) INSIDE the stream. */
        if (overlap) {
            const int orc = pn_qwen35_tt_overlap_tick(g, bands);

            prefill_batched_add(b, 4, pn_qwen35_tt_overlap_take_seconds());
            if (orc != 0) {
                g->gdn_pipe = false;
                prefill_chain_settle(g);
                return -1;
            }
        }
        if (prefill_stream_finish(g) != 0) {
            g->gdn_pipe = false;
            prefill_chain_settle(g);
            return -1;
        }
        g->gdn_pipe = false;
        g->gdn_pipe2 = false;
        pn_qwen35_tt_ab_reader_end(g);
        /* The state back where the decode family, the tail forward and the next turn read it. */
        if (pn_tt_gdnc_state_split_exit(g->device, (uint32_t)ssm_index) != 0)
            return -1;
        if (overlap) {
            if (prefill_seek(g, 0u, bands) != 0)
                return -1;
            return 1;
        }
        if (prefill_seek(g, 0u, bands) != 0)
            return -1;

        mark = prefill_now();
        if (prefill_project_out(g, &d->ssm_out, &s->ssm_gated) != 0)
            return -1;
        prefill_batched_add(b, 1, prefill_now() - mark); /* ssm-out-proj */
        return 0;
    }

    if (prefill_pull(g, &s->ssm_qkv, b->ssm_qkv, rows) != 0 ||
        prefill_pull(g, &s->ssm_z, b->ssm_z, rows) != 0 ||
        prefill_pull(g, &s->ssm_alpha, b->ssm_alpha, rows) != 0 ||
        prefill_pull(g, &s->ssm_beta, b->ssm_beta, rows) != 0)
        return -1;
    {
        const pn_tt_activation *live[] = { &s->ssm_qkv, &s->ssm_z, &s->ssm_gated };

        if (prefill_zero(g, live, sizeof(live) / sizeof(live[0])) != 0)
            return -1;
    }

    for (int i = 0; i < rows; i++) {
        int r = prefill_red("prefill_causal") ? rows - 1 - i : i;
        int rc;

        if (prefill_push_row(g, &s->ssm_qkv, b->ssm_qkv, r) != 0 ||
            prefill_push_row(g, &s->ssm_z, b->ssm_z, r) != 0)
            return -1;
        prefill_sequential_bracket bracket;
        prefill_sequential_open(g, &bracket);
        rc = pn_qwen35_tt_gdn_strip_layer(g, l, cfg, ssm_index, &s->ssm_qkv, &s->ssm_z,
                                          b->ssm_beta + (size_t)r * beta_w,
                                          b->ssm_alpha + (size_t)r * alpha_w,
                                          &s->ssm_gated);
        prefill_sequential_close(g, &bracket);
        if (rc != 0)
            return -1;
        if (prefill_pull_row(g, &s->ssm_gated, b->ssm_gated, r) != 0)
            return -1;
    }
    if (prefill_push(g, &s->ssm_gated, b->ssm_gated, rows) != 0)
        return -1;

    mark = prefill_now();
    if (prefill_project_out(g, &d->ssm_out, &s->ssm_gated) != 0)
        return -1;
    prefill_batched_add(b, 1, prefill_now() - mark); /* ssm-out-proj */
    return 0;
}

/** One attention layer over `rows` positions. Same shape as the SSM one: project once,
 *  rotate/append/attend `rows` times, project out once. */
/** Prefill lane ATTENTION PIPE (`PN_TT_ATTN_PIPE=0` off): band b's attention in flight while
 *  band b+1's QK norms + RoPE run; its K/V appends wait for it (no append under a running
 *  attention), its sigmoid gate is left pending. Same ops on the same band rows: bit-exact.
 *  DEFAULT ON from 2 bands (run_pg_attnpipe.sh, 2026-10-22: serve warm 1,630 TTFT 2,103 ->
 *  2,050 ms, IDENTICAL, red DIFFERS); one band has nothing to overlap. */
static bool prefill_attn_pipe_wanted(uint32_t bands)
{
    const char *env = getenv("PN_TT_ATTN_PIPE");

    return bands >= 2u && !(env && *env && atoi(env) == 0);
}

static bool prefill_attn_deep_wanted(const pn_qwen35_tt *g, uint32_t bands);
static double g_prefill_attn_grid_seconds;
static uint32_t g_prefill_attn_grid_swaps;
/* Deep attention's host phases (prep + append, queue, drain, gates), cumulative: is a long
 * prompt's attention enqueue-bound or card-bound? Timing only. */
static double g_prefill_attn_deep_seconds[4];
static uint32_t g_prefill_attn_deep_passes;

/** C: `PN_TT_ATTN_GRID_DEPTH=N` (default 0, off) lends the
 *  48 grid cores to the deep attention of every layer-pass whose first position is at least N
 *  (break-even ~26k at 2,048-row passes). Needs `PN_TT_ATTN_DEEP=1`. */
static uint32_t prefill_attn_grid_depth(void)
{
    const char *env = getenv("PN_TT_ATTN_GRID_DEPTH");

    return env && *env ? (uint32_t)strtoul(env, NULL, 10) : 0u;
}

/** Lend the grid to this layer-pass's attention when C applies. @return members borrowed */
static uint32_t prefill_attn_grid_borrow(pn_qwen35_tt *g, int base_pos, uint32_t bands)
{
    const uint32_t depth = prefill_attn_grid_depth();
    uint32_t cores[PN_TT_ATTN_GRID_MAX_CORES];
    uint32_t lent, armed;
    double at;

    /* L1a's lent cores already fill the borrowed set for this prompt. */
    if (depth == 0u || (uint32_t)base_pos < depth || !prefill_attn_deep_wanted(g, bands) ||
        pn_tt_prefill_lend_active(g->device))
        return 0u;
    at = prefill_now();
    lent = pn_tt_matmul_grid_lend(g->device, cores, PN_TT_ATTN_GRID_MAX_CORES);
    armed = lent ? pn_tt_attention_grid_arm(g->device, cores, lent) : 0u;
    g_prefill_attn_grid_seconds += prefill_now() - at;
    if (lent && !armed)
        (void)pn_tt_matmul_grid_begin(g->device);
    return armed;
}

/** Park the borrowed members and boot the grid again. @return 0, -1 when a member did not exit */
static int prefill_attn_grid_return(pn_qwen35_tt *g)
{
    const double at = prefill_now();
    int rc = pn_tt_attention_grid_park_all(g->device, NULL, NULL);

    /* A member that did not exit leaves its core unbootable: the grid then refuses by name
     * (pn_tt_core_holder / the boot guard) and the decode group serves the projections. */
    (void)pn_tt_matmul_grid_begin(g->device);
    g_prefill_attn_grid_seconds += prefill_now() - at;
    g_prefill_attn_grid_swaps++;
    return rc;
}

/** DEEP-QUEUED ATTENTION (pn_tt_attn_deep.h):
 *  `PN_TT_ATTN_DEEP=0|1`; unset, on exactly while this prompt has L1a's lent cores (alone it only
 *  loses the FFN overlap: 8k 25.06 -> 26.77 s). */
static bool prefill_attn_deep_wanted(const pn_qwen35_tt *g, uint32_t bands)
{
    return bands >= 2u &&
           pn_tt_lend_deep_on(getenv("PN_TT_ATTN_DEEP"), pn_tt_prefill_lend_active(g->device));
}

/**
 * The attention layer-pass with every band's K/V appended before any band's attention: prep +
 * append band by band, then every (band, head) descriptor queued over the members with no
 * barrier, one drain, then the gates. Each descriptor is the shipped one; a band's later rows
 * sit past its live length and take the dead mask, as the stale rows they replace did.
 *
 * A failed drain re-runs the ATTENTION ONLY through the shipped per-band call, which re-arms a
 * member that refused (the norms ran in place and must not run twice).
 */
static int prefill_attn_deep(pn_qwen35_tt *g, const pn_qwen35_tt_layer *d,
                             const pn_qwen35moe_config *cfg, int attn_index, int base_pos,
                             int rows, uint32_t bands)
{
    pn_qwen35_tt_strips *s = &g->card;
    pn_qwen35_tt_attn_strips *a = &g->attn_card;
    prefill_sequential_bracket bracket;
    /* RED `attn_deep_skip`: the last band is never queued, so its attention output is the
     * previous layer's — must DIFFER. (Queueing a band before its own append would race the
     * append and is not a deterministic red.) */
    const bool skip_last = prefill_red("attn_deep_skip");
    double mark = prefill_now(), now;
    uint32_t band;
    int done, rc = 0;

    prefill_sequential_open(g, &bracket);
    for (band = 0, done = 0; rc == 0 && band < bands && done < rows; band++) {
        const int take = (rows - done) < 32 ? (rows - done) : 32;
        const int chunk_base = prefill_red("prefill_pos") ? 0 : base_pos + done;

        if (prefill_seek(g, band, 1u) != 0 ||
            pn_qwen35_tt_attn_chunk_prep(g, d, cfg, chunk_base, take) != 0 ||
            pn_qwen35_tt_attn_chunk_append(g, attn_index, chunk_base, take) != 0)
            rc = -1;
        done += take;
    }
    now = prefill_now();
    g_prefill_attn_deep_seconds[0] += now - mark;
    mark = now;
    for (band = 0, done = 0; rc == 0 && band < bands && done < rows; band++) {
        const int take = (rows - done) < 32 ? (rows - done) : 32;
        const int chunk_base = prefill_red("prefill_pos") ? 0 : base_pos + done;
        const bool last = band + 1u == bands || done + take >= rows;

        if (!(skip_last && last) &&
            (prefill_seek(g, band, 1u) != 0 ||
             pn_qwen35_tt_attn_chunk_queue(g, attn_index, chunk_base, take) != 0))
            rc = -1;
        done += take;
    }
    if (rc != 0)
        fprintf(stderr, "qwen35_tt: deep attention could not queue layer %d (prep, append or a "
                        "member's enqueue failed)\n", attn_index);
    now = prefill_now();
    g_prefill_attn_deep_seconds[1] += now - mark;
    mark = now;
    if (pn_tt_attention_chunk_drain(g->device) != 0 && rc == 0) {
        fprintf(stderr, "qwen35_tt: deep attention drain failed at layer %d — re-running the "
                        "layer's attention band by band\n", attn_index);
        for (band = 0, done = 0; rc == 0 && band < bands && done < rows; band++) {
            const int take = (rows - done) < 32 ? (rows - done) : 32;
            const int chunk_base = prefill_red("prefill_pos") ? 0 : base_pos + done;

            if (prefill_seek(g, band, 1u) != 0 ||
                pn_tt_attention_chunk(g->device, (uint32_t)attn_index, &a->q_rot, &s->attn_out,
                                      (uint32_t)chunk_base, (uint32_t)take) != 0)
                rc = -1;
            done += take;
        }
    }
    now = prefill_now();
    g_prefill_attn_deep_seconds[2] += now - mark;
    mark = now;
    for (band = 0, done = 0; rc == 0 && band < bands && done < rows; band++) {
        const int take = (rows - done) < 32 ? (rows - done) : 32;

        if (prefill_seek(g, band, 1u) != 0)
            rc = -1;
        if (rc == 0) {
            pn_tt_eltwise_group_defer_next(g->device);
            rc = pn_tt_sigmoid_mul(g->device, &a->z, &s->attn_out, &a->gate, &a->gated);
        }
        done += take;
    }
    if (pn_tt_eltwise_group_settle(g->device) != 0)
        rc = -1;
    g_prefill_attn_deep_seconds[3] += prefill_now() - mark;
    g_prefill_attn_deep_passes++;
    prefill_sequential_close(g, &bracket);
    return rc;
}

static int prefill_attn_pipelined(pn_qwen35_tt *g, const pn_qwen35_tt_layer *d,
                                  const pn_qwen35moe_config *cfg, int attn_index, int base_pos,
                                  int rows, uint32_t bands, bool overlap)
{
    struct pn_qwen35_tt_batch *b = g->batch;
    pn_qwen35_tt_strips *s = &g->card;
    pn_qwen35_tt_attn_strips *a = &g->attn_card;
    prefill_sequential_bracket bracket;
    int done = 0, rc = 0;

    prefill_sequential_open(g, &bracket);
    if (prefill_seek(g, 0u, 1u) != 0 ||
        pn_qwen35_tt_attn_chunk_prep(g, d, cfg, prefill_red("prefill_pos") ? 0 : base_pos,
                                     rows < 32 ? rows : 32) != 0)
        rc = -1;
    for (uint32_t band = 0; rc == 0 && band < bands && done < rows; band++) {
        const int take = (rows - done) < 32 ? (rows - done) : 32;
        const int chunk_base = prefill_red("prefill_pos") ? 0 : base_pos + done;
        pn_tt_activation z, out, gate, gated;

        if (prefill_seek(g, band, 1u) != 0 ||
            pn_qwen35_tt_attn_chunk_issue(g, attn_index, chunk_base, take) != 0) {
            rc = -1;
            break;
        }
        z = a->z;
        out = s->attn_out;
        gate = a->gate;
        gated = a->gated;
        done += take;
        if (band + 1u < bands && done < rows) {
            const int next = (rows - done) < 32 ? (rows - done) : 32;

            if (prefill_seek(g, band + 1u, 1u) != 0 ||
                pn_qwen35_tt_attn_chunk_prep(g, d, cfg,
                                             prefill_red("prefill_pos") ? 0 : base_pos + done,
                                             next) != 0)
                rc = -1;
        }
        if (pn_tt_attention_settle(g->device) != 0)
            rc = -1;
        /* RED `attn_pipe_gate_shift`: band b's gate runs on whatever band the strips are
         * seeked to now (b+1 after a prep) — must DIFFER. */
        if (rc == 0 && prefill_red("attn_pipe_gate_shift")) {
            z = a->z;
            out = s->attn_out;
            gate = a->gate;
            gated = a->gated;
        }
        if (rc == 0) {
            pn_tt_eltwise_group_defer_next(g->device);
            rc = pn_tt_sigmoid_mul(g->device, &z, &out, &gate, &gated);
        }
        /* A: the super-band jobs' steps due after this band (the gate just issued is band
         * b's, so the plan's pipe depth is 0). */
        if (rc == 0 && overlap) {
            rc = pn_qwen35_tt_overlap_tick(g, band);
            prefill_batched_add(b, 4, pn_qwen35_tt_overlap_take_seconds());
        }
    }
    (void)pn_tt_attention_settle(g->device);
    if (pn_tt_eltwise_group_settle(g->device) != 0)
        rc = -1;
    prefill_sequential_close(g, &bracket);
    return rc;
}

/** A: the overlap of an attention layer's out-proj and FFN under its band loop, planned; the
 *  wavefront prepares the next SSM layer only under the exact table gates (nothing waits on a
 *  reader then). @return true when the plan runs. */
static bool prefill_attn_overlap_begin(pn_qwen35_tt *g, const pn_qwen35_tt_layer *d, int rows,
                                       uint32_t bands, const pn_qwen35_tt_layer *next_d,
                                       int next_ssm_index)
{
    struct pn_qwen35_tt_batch *b = g->batch;
    const bool table = !g_verify_mode && pn_qwen35_tt_gdn_card_gate_possible() &&
                       !g->gdn_det_gates && pn_tt_gdnc_table_ready(g->device);
    pn_qwen35_tt_overlap_next next = { NULL, -1, rows, NULL, NULL, 0u, 0u };

    /* The plan's FFN steps are the dense SwiGLU's. */
    if (g->moe || !pn_qwen35_tt_overlap_attention(g, bands))
        return false;
    if (table && next_d)
        next = (pn_qwen35_tt_overlap_next){ next_d, next_ssm_index, rows, b->ssm_alpha,
                                            b->ssm_beta, prefill_width(&g->card.ssm_alpha),
                                            prefill_width(&g->card.ssm_beta) };
    pn_qwen35_tt_overlap_set_card_gates(table);
    return pn_qwen35_tt_overlap_layer_begin(g, d, bands, 0u, &next, &g->attn_card.gated,
                                            &d->attn_output) == 0;
}

static int prefill_attn_layer(pn_qwen35_tt *g, const pn_qwen35_tt_layer *d,
                              const pn_qwen35moe_config *cfg, int attn_index, int base_pos,
                              int rows, bool allow_overlap, const pn_qwen35_tt_layer *next_d,
                              int next_ssm_index)
{
    struct pn_qwen35_tt_batch *b = g->batch;
    pn_qwen35_tt_strips *s = &g->card;
    pn_qwen35_tt_attn_strips *a = &g->attn_card;
    const int max_seq = pn_qwen35moe_max_seq(cfg);
    double mark = prefill_now();

    if (pn_qwen35_tt_attn_project_card(g, d) != 0 || pn_tt_matvec_flush(g->device) != 0)
        return -1;
    prefill_batched_add(b, 2, prefill_now() - mark); /* attn-proj */

    /* STEP 4 S2: THE VERIFY — the decode attention core, row by row, at position base+r. */
    if (g_verify_mode && !getenv("PN_TT_SPEC_ROWCOPY") &&
        !(getenv("PN_TT_SPEC_ATTN_ROWS") && atoi(getenv("PN_TT_SPEC_ATTN_ROWS")) == 0)) {
        /* S2c: the row-wise steps once over every row (QK norms, rotary at each row's own
         * position by the decode table's angles, the gate), the K/V appends straight from row
         * r of the verify strips, and only flash decode per row through the decode strips —
         * one row copy each way. Row r attends to positions <= base+r, so appending every
         * row first changes nothing it reads. */
        const double rows_started = pn_time_ms();
        pn_qwen35_tt_attn_strips *da = &g->attn_card_tall; /* the decode strips, swapped out */
        pn_qwen35_tt_strips *ds = &g->card_tall;
        int rc = 0, rows_done = 0;

        if (pn_tt_head_norm_bf16(g->device, FT_TT_OP_REDUCE_FN_RMS_NORM, &d->attn_q_norm,
                                 (uint32_t)cfg->n_head, &a->q, &a->q) != 0 ||
            pn_tt_head_norm_bf16(g->device, FT_TT_OP_REDUCE_FN_RMS_NORM, &d->attn_k_norm,
                                 (uint32_t)cfg->n_head_kv, &s->attn_k, &s->attn_k) != 0 ||
            pn_tt_rope_rows(g->device, &a->q, &a->q_rot, (uint32_t)cfg->n_head,
                            (uint32_t)base_pos, (uint32_t)rows) != 0 ||
            pn_tt_rope_rows(g->device, &s->attn_k, &a->k_rot, (uint32_t)cfg->n_head_kv,
                            (uint32_t)base_pos, (uint32_t)rows) != 0 ||
            pn_qwen35_tt_attn_rotate(g, cfg) != 0)
            return -1;
        for (int r = 0; rc == 0 && r < rows; r++)
            rc = (pn_tt_kv_append_row(g->device, (uint32_t)attn_index, false,
                                      pn_qwen35_tt_attn_key(g), (uint32_t)r,
                                      (uint32_t)(base_pos + r)) != 0 ||
                  pn_tt_kv_append_row(g->device, (uint32_t)attn_index, true, &s->attn_v,
                                      (uint32_t)r, (uint32_t)(base_pos + r)) != 0)
                     ? -1 : 0;
        /* S2f: flash decode for every row in ONE descriptor a group (ROWS mode) when every
         * row ends in the same position tile (`PN_TT_SPEC_ATTN_MULTI=0`: one a row). */
        if (rc == 0 && base_pos / 32 == (base_pos + rows - 1) / 32 && base_pos + rows < max_seq &&
            !(getenv("PN_TT_SPEC_ATTN_MULTI") && atoi(getenv("PN_TT_SPEC_ATTN_MULTI")) == 0)) {
            rc = pn_tt_attention_rows(g->device, (uint32_t)attn_index,
                                      pn_qwen35_tt_attn_query(g), &s->attn_out,
                                      (uint32_t)base_pos, (uint32_t)rows);
            rows_done = rows;
        }
        for (int r = rows_done; rc == 0 && r < rows; r++) {
            const int pos = base_pos + r;
            const int seq_len = (pos + 1 < max_seq) ? pos + 1 : max_seq;

            rc = (pn_tt_row_copy(g->device, pn_qwen35_tt_attn_query(g), (uint32_t)r,
                                 &da->q_rot, 0u) != 0 ||
                  pn_tt_attention(g->device, (uint32_t)attn_index, &da->q_rot, &ds->attn_out,
                                  (uint32_t)seq_len) != 0 ||
                  pn_tt_row_copy(g->device, &ds->attn_out, 0u, &s->attn_out, (uint32_t)r) != 0)
                     ? -1 : 0;
        }
        if (rc != 0 || pn_tt_sigmoid_mul(g->device, &a->z, &s->attn_out, &a->gate,
                                          &a->gated) != 0)
            return -1;
        g->spec_attn_rows_ms += pn_time_ms() - rows_started;
        mark = prefill_now();
        if (prefill_project_out(g, &d->attn_output, &a->gated) != 0)
            return -1;
        prefill_batched_add(b, 3, prefill_now() - mark); /* attn-out-proj */
        return 0;
    }
    if (g_verify_mode) {
        const double rows_started = pn_time_ms();

        for (int r = 0; r < rows; r++) {
            const int pos = base_pos + r;
            const int seq_len = (pos + 1 < max_seq) ? pos + 1 : max_seq;
            pn_qwen35_tt_strips *vs;
            pn_qwen35_tt_attn_strips *va;
            int rc;

            prefill_tall_swap(g);
            vs = &g->card_tall;
            va = &g->attn_card_tall;
            rc = (pn_tt_row_copy(g->device, &va->q, (uint32_t)r, &g->attn_card.q, 0u) != 0 ||
                  pn_tt_row_copy(g->device, &va->z, (uint32_t)r, &g->attn_card.z, 0u) != 0 ||
                  pn_tt_row_copy(g->device, &vs->attn_k, (uint32_t)r, &g->card.attn_k, 0u) != 0 ||
                  pn_tt_row_copy(g->device, &vs->attn_v, (uint32_t)r, &g->card.attn_v, 0u) != 0)
                     ? -1 : 0;
            if (rc == 0) {
                pn_tt_device_set_single_row(g->device, true);
                rc = pn_qwen35_tt_attn_core_card(g, d, cfg, attn_index, pos, seq_len);
                pn_tt_device_set_single_row(g->device, false);
            }
            if (rc == 0 && pn_tt_matvec_flush(g->device) != 0)
                rc = -1;
            if (rc == 0)
                rc = pn_tt_row_copy(g->device, &g->attn_card.gated, 0u, &va->gated, (uint32_t)r);
            prefill_tall_swap(g);
            if (rc != 0)
                return -1;
        }
        g->spec_attn_rows_ms += pn_time_ms() - rows_started;
        mark = prefill_now();
        if (prefill_project_out(g, &d->attn_output, &a->gated) != 0)
            return -1;
        prefill_batched_add(b, 3, prefill_now() - mark); /* attn-out-proj */
        return 0;
    }

    /* THE CHUNK PATH: NO ROW SELECT AT ALL.
     *
     * Every strip the attention block needs is already on the card with row t carrying
     * position `base_pos + t` — the projections put it there — so the four whole-tile pulls,
     * the nine-strip re-zero, the 4 x rows row-0 pushes and the rows row-0 pulls all go
     * away. That is ~48 KiB up and ~32 KiB down PER POSITION removed from a 5.3 MB/s link,
     * and it is the whole reason batching cost 3.4x what it saved.
     *
     * The alignment condition is the one the three chunk ops share (RoPE's row-phase
     * contract, a whole KV tile, one causal mask tile), and a prefill pass always starts on
     * a multiple of 32. The red arms still have to be able to reorder and mis-position the
     * work, so a named one falls back to the per-position path rather than being ignored. */
    if ((base_pos % 32) == 0 && rows > 0 && rows <= (int)(prefill_bands(g) * 32u) &&
        prefill_attn_chunk_enabled() &&
        !prefill_red("prefill_causal") && !prefill_red("prefill_rowfill")) {
        /* RED PROOF: the whole chunk claims to start at position 0, so RoPE rotates by the
         * wrong angle and the chunk attends to the wrong span of cache. INVISIBLE ON THE
         * FIRST PASS, where base_pos IS 0 — the same structure both GDN mixers recorded, and
         * the reason the gate runs more than one pass.
         *
         * `prefill_causal` and `prefill_rowfill` are about the per-position LOOP, which a
         * chunk does not have, so they select the fallback path above and still fire there.
         * The chunk path's own causality arm is on the card, in
         * foundry/tt/tests/test_phase6_attention.c: `causal_off` points the descriptor at a
         * constants operand whose causal tiles are all live. */
        const uint32_t bands = prefill_span_for(g, rows);
        int done = 0;

        const double mixer_mark = prefill_now();

        /* C: the grid is lent BEFORE the streamed segment opens and returned after it ends —
         * parking, booting and resets are host NoC writes, not puller entries. */
        const uint32_t borrowed = prefill_attn_grid_borrow(g, base_pos, bands);

        prefill_stream_open(g);
        if (g_prefill_streaming && prefill_attn_pipe_wanted(bands)) {
            /* Deep-queued attention (PN_TT_ATTN_DEEP) has no band loop for A's FFN to sit
             * under: it runs without the overlap, and the out-proj follows it. */
            const bool deep = prefill_attn_deep_wanted(g, bands);
            const bool overlap =
                !deep && allow_overlap &&
                prefill_attn_overlap_begin(g, d, rows, bands, next_d, next_ssm_index);
            int rc = deep ? prefill_attn_deep(g, d, cfg, attn_index, base_pos, rows, bands)
                          : prefill_attn_pipelined(g, d, cfg, attn_index, base_pos, rows, bands,
                                                   overlap);

            /* A: the layer's last jobs (the plan's tail) INSIDE the stream. */
            if (rc == 0 && overlap) {
                rc = pn_qwen35_tt_overlap_tick(g, bands);
                prefill_batched_add(b, 4, pn_qwen35_tt_overlap_take_seconds());
            }
            if (rc != 0 || prefill_stream_finish(g) != 0 ||
                (borrowed && prefill_attn_grid_return(g) != 0) || prefill_seek(g, 0u, bands) != 0) {
                (void)prefill_stream_close();
                pn_qwen35_tt_overlap_abort(g);
                return -1;
            }
            mark = prefill_now();
            pn_prefill_wall_attn_mixer(mark - mixer_mark);
            if (overlap)
                return 1; /* the plan ran the out-proj, residuals, post-norm and FFN */
            if (prefill_project_out(g, &d->attn_output, &a->gated) != 0)
                return -1;
            prefill_batched_add(b, 3, prefill_now() - mark); /* attn-out-proj */
            return 0;
        }
        /* ONE BAND AT A TIME, for a reason the GDN loop does not have: the chunk-mode
         * attention descriptor REFUSES a non-zero `query_row` (ft_tt_op_attention.c) and there
         * is no tile-row field in the protocol at all, so the second band is reached the only
         * way it can be — by sliding the operand address, which is exactly what the seek did.
         * RoPE's table is one 32-position block and the KV append's `source_row0` caps at 32
         * for the same reason. Each band carries its own first position, so the mask, the
         * rotary angle and the K/V slots are the ones the per-32 path would have used. */
        for (uint32_t band = 0; band < bands && done < rows; band++) {
            const int take = (rows - done) < 32 ? (rows - done) : 32;
            const int chunk_base = prefill_red("prefill_pos") ? 0 : base_pos + done;
            int rc;

            if (prefill_seek(g, band, 1u) != 0) {
                (void)prefill_stream_close();
                return -1;
            }
            prefill_sequential_bracket bracket;
            prefill_sequential_open(g, &bracket);
            rc = pn_qwen35_tt_attn_core_chunk(g, d, cfg, attn_index, chunk_base, take);
            prefill_sequential_close(g, &bracket);
            if (rc != 0) {
                (void)prefill_stream_close();
                return -1;
            }
            done += take;
        }
        if (prefill_stream_finish(g) != 0 || prefill_seek(g, 0u, bands) != 0)
            return -1;

        mark = prefill_now();
        if (prefill_project_out(g, &d->attn_output, &a->gated) != 0)
            return -1;
        prefill_batched_add(b, 3, prefill_now() - mark); /* attn-out-proj */
        return 0;
    }

    if (prefill_pull(g, &a->q, b->attn_q, rows) != 0 ||
        prefill_pull(g, &a->z, b->attn_z, rows) != 0 ||
        prefill_pull(g, &s->attn_k, b->attn_k, rows) != 0 ||
        prefill_pull(g, &s->attn_v, b->attn_v, rows) != 0)
        return -1;
    {
        const pn_tt_activation *live[] = { &a->q, &a->z, &s->attn_k, &s->attn_v, &a->q_rot,
                                           &a->k_rot, &s->attn_out, &a->gate, &a->gated };

        if (prefill_zero(g, live, sizeof(live) / sizeof(live[0])) != 0)
            return -1;
    }

    for (int i = 0; i < rows; i++) {
        int r = prefill_red("prefill_causal") ? rows - 1 - i : i;
        int pos = prefill_red("prefill_pos") ? base_pos : base_pos + r;
        int seq_len = (pos + 1 < max_seq) ? pos + 1 : max_seq;
        int rc;

        if (prefill_push_row(g, &a->q, b->attn_q, r) != 0 ||
            prefill_push_row(g, &a->z, b->attn_z, r) != 0 ||
            prefill_push_row(g, &s->attn_k, b->attn_k, r) != 0 ||
            prefill_push_row(g, &s->attn_v, b->attn_v, r) != 0)
            return -1;
        prefill_sequential_bracket bracket;
        prefill_sequential_open(g, &bracket);
        rc = pn_qwen35_tt_attn_core_card(g, d, cfg, attn_index, pos, seq_len);
        prefill_sequential_close(g, &bracket);
        if (rc != 0)
            return -1;
        if (prefill_pull_row(g, &a->gated, b->attn_gated, r) != 0)
            return -1;
    }
    if (prefill_push(g, &a->gated, b->attn_gated, rows) != 0)
        return -1;

    mark = prefill_now();
    if (prefill_project_out(g, &d->attn_output, &a->gated) != 0)
        return -1;
    prefill_batched_add(b, 3, prefill_now() - mark); /* attn-out-proj */
    return 0;
}

/** The dense SwiGLU FFN, entirely batched: nothing in it is position-dependent. */
static int prefill_ffn(pn_qwen35_tt *g, const pn_qwen35_tt_layer *d)
{
    struct pn_qwen35_tt_batch *b = g->batch;
    pn_qwen35_tt_strips *s = &g->card;
    double mark = prefill_now();
    int rc = -1;

    /* S3b: the VERIFY rows (MTP k1's verify runs this pass) take the fused SwiGLU when it is
     * armed; a prefill pass proper keeps the three ops. */
    if (g_verify_mode) {
        if (pn_qwen35_tt_swiglu(g, &d->ffn_gate, &d->ffn_up, &s->normed, &s->ffn_gate,
                                &s->ffn_up, &s->ffn_swiglu) == 0 &&
            prefill_project_out(g, &d->ffn_down, &s->ffn_swiglu) == 0)
            rc = 0;
    } else if (pn_qwen35_tt_project(g, &d->ffn_gate, &s->normed, &s->ffn_gate, false) == 0 &&
               pn_qwen35_tt_project(g, &d->ffn_up, &s->normed, &s->ffn_up, false) == 0 &&
               pn_tt_matvec_flush(g->device) == 0 &&
               pn_tt_silu_mul(g->device, &s->ffn_gate, &s->ffn_up, &s->ffn_swiglu) == 0 &&
               pn_qwen35_tt_project(g, &d->ffn_down, &s->ffn_swiglu, &s->project, true) == 0) {
        rc = 0;
    }
    prefill_batched_add(b, 4, prefill_now() - mark); /* ffn */
    return rc;
}

/** The MoE FFN of a pass (plan S1), timed as the dense FFN site. */
static int prefill_moe_ffn(pn_qwen35_tt *g, const pn_qwen35_tt_layer *d, int layer, int rows)
{
    double mark = prefill_now();
    /* A verify (one band of draft rows) routes CONCAT so each row is the decode token's.
     * RED moe_verify_list: the verify takes the prefill's union LIST instead (ascending expert
     * order), which is NOT the decode token's sum order — speculation must then DIFFER. */
    const int rc = g_verify_mode && rows <= 32 && !prefill_red("moe_verify_list")
                       ? pn_qwen35_tt_moe_verify_ffn(g, d, layer, (uint32_t)rows)
                       : pn_qwen35_tt_moe_prefill_ffn(g, d, layer, prefill_span_for(g, rows));

    prefill_batched_add(g->batch, 4, prefill_now() - mark); /* ffn */
    return rc;
}

/** PHASE G: TALL strips (a whole prompt a pass) key their mirrors by the span the pass's
 *  producers write (pn_tt_replica_map.c), so a pass whose span differs republishes. */
static int prefill_publish_tall_span(pn_qwen35_tt *g, uint32_t bands)
{
    pn_qwen35_tt_strips *s = &g->card;
    const pn_tt_activation *mirrored[2] = { &s->normed, &s->ffn_swiglu };

    if (prefill_bands(g) <= 4u || bands == g_prefill_published_span)
        return 0;
    pn_tt_replica_map_set_span(bands);
    pn_tt_replica_map_set_superband(pn_qwen35_tt_overlap_span(bands));
    if (pn_tt_publish_replica_map(g->device, mirrored, 2u, &s->ffn_swiglu) != 0)
        return -1;
    g_prefill_published_span = bands;
    return 0;
}

/** `PN_TT_PREFILL_GRID_MIN_ROWS`: the batched rows from which the prefill grid boots. */
static int prefill_grid_min_rows(void)
{
    const char *env = getenv("PN_TT_PREFILL_GRID_MIN_ROWS");

    return env ? atoi(env) : PN_QWEN35_TT_GRID_MIN_ROWS;
}

/** ONE batched pass: `rows` consecutive positions from `base_pos` through the whole stack,
 *  leaving the card's carried state and K/V where the sequential path would leave them, and
 *  computing no logits at all. */
/**
 * STEP 4 S2b: the MTP block over a pass's rows — its KV for the prompt, so a draft attends to
 * the whole context. Row r is (trunk hidden r, embedding of `next[r]`) at base_pos + r; the
 * trunk's hidden is consumed (the logits come from the tail forward, not from this pass).
 * Chunked attention: a draft only proposes, so the parity-gated prefill path is fine here.
 */
static int prefill_mtp_layer(pn_qwen35_tt *g, pn_qwen35moe_model *m,
                             const pn_qwen35moe_config *cfg, const uint32_t *next, int rows,
                             int base_pos)
{
    struct pn_qwen35_tt_batch *b = g->batch;
    pn_qwen35_tt_strips *s = &g->card;
    const pn_qwen35_tt_layer *d = &g->mtp.layer;
    const uint32_t bands = prefill_span_for(g, rows);
    pn_tt_activation *e = &s->hidden[s->slot ^ 1];
    int staged = 0;

    for (int r = 0; r < rows; r++)
        pn_qwen35moe_model_embed_row(m, next[r], b->hidden + (size_t)r * (size_t)cfg->n_embd);
    /* Prefill lane: as the hidden stage — the upload below rewrites bands [0, bands). */
    if (prefill_seek(g, 0u, bands) != 0 ||
        (!(g_verify_mode && bands == 1u) &&
         (prefill_stage_zero_all()
              ? pn_tt_activation_zero_bands(g->device, e,
                                            prefill_bands(g) > 4u ? b->max_span : 0u)
              : pn_tt_activation_zero_band_range(g->device, e, bands,
                                                 prefill_bands(g) > 4u ? b->max_span
                                                                       : prefill_bands(g))) !=
             0))
        return -1;
    for (uint32_t band = 0; band < bands && staged < rows; band++) {
        const int take = (rows - staged) < 32 ? (rows - staged) : 32;

        if (prefill_seek(g, band, 1u) != 0 ||
            prefill_push(g, e, b->hidden + (size_t)staged * (size_t)cfg->n_embd, take) != 0)
            return -1;
        staged += take;
    }
    if (prefill_seek(g, 0u, bands) != 0)
        return -1;
    if (pn_tt_rms_norm(g->device, &g->mtp.hnorm, &s->hidden[s->slot], &s->normed) != 0 ||
        pn_qwen35_tt_project(g, &g->mtp.eh_h, &s->normed, &s->project, true) != 0 ||
        pn_tt_rms_norm(g->device, &g->mtp.enorm, e, &s->normed) != 0 ||
        pn_qwen35_tt_project(g, &g->mtp.eh_e, &s->normed, &s->hidden[s->slot], true) != 0 ||
        prefill_residual(g) != 0)
        return -1;
    if (pn_tt_rms_norm(g->device, &d->attn_norm, &s->hidden[s->slot], &s->normed) != 0 ||
        prefill_attn_layer(g, d, cfg, g->mtp.attn_index, base_pos, rows, false, NULL, -1) != 0 ||
        prefill_residual(g) != 0 ||
        pn_tt_rms_norm(g->device, &d->post_norm, &s->hidden[s->slot], &s->normed) != 0 ||
        (g->moe ? (g->moe_mtp ? prefill_moe_ffn(g, d, cfg->n_layer, rows) : -1)
                : prefill_ffn(g, d)) != 0 ||
        prefill_residual(g) != 0)
        return -1;
    return 0;
}

int pn_qwen35_tt_prefill_mtp_prompt_run(pn_qwen35_tt *g, pn_qwen35moe_model *m,
                                        const pn_qwen35moe_config *cfg, const uint32_t *next,
                                        int rows, int base_pos)
{
    int rc;

    if (!g || g->tall_rows == 0u || g->tall_active || !g->batch || g->batch->rows < rows)
        return -1;
    if (!pn_tt_matmul_grid_is_running(g->device) && rows >= prefill_grid_min_rows())
        (void)pn_tt_matmul_grid_begin(g->device);
    prefill_tall_enter(g);
    rc = prefill_publish_tall_span(g, prefill_span_for(g, rows));
    if (rc == 0)
        rc = prefill_mtp_layer(g, m, cfg, next, rows, base_pos);
    if (rc == 0 && pn_tt_matvec_flush(g->device) != 0)
        rc = -1;
    pn_tt_matmul_grid_end(g->device);
    if (prefill_seek(g, 0u, 1u) != 0)
        rc = -1;
    if (prefill_tall_exit(g) != 0)
        rc = -1;
    pn_tt_matmul_group_param_cache_reset(g->device);
    /* Issued outside the dispatch model's per-token shape (as a verify is): re-learn it before
     * any plain forward replays again. */
    pn_tt_dispatch_sequence_reset();
    return rc;
}

static int prefill_pass(pn_qwen35_tt *g, pn_qwen35moe_model *m,
                        const pn_qwen35moe_config *cfg, const uint32_t *tokens, int rows,
                        int base_pos, bool last)
{
    struct pn_qwen35_tt_batch *b = g->batch;
    pn_qwen35_tt_strips *s = &g->card;
    double mark;

    const uint32_t bands = prefill_span_for(g, rows);
    int staged = 0;

    if (prefill_publish_tall_span(g, bands) != 0)
        return -1;

    PREFILL_WALL_OPEN(g, PN_PREFILL_WALL_EMBED);
    for (int r = 0; r < rows; r++)
        pn_qwen35moe_model_embed_row(m, tokens[r],
                                     b->hidden + (size_t)r * (size_t)cfg->n_embd);
    PREFILL_WALL_CLOSE(g, PN_PREFILL_WALL_EMBED);
    s->slot = 0;
    /* A SHORT PASS LEAVES BANDS NOBODY WRITES, and an Mt-band matmul reads all of them. The
     * tilizer zeroes the rows above `take` INSIDE the last band it writes; the bands PAST it
     * would still carry the previous pass's positions, so the whole strip is re-staged zero
     * first. An Mt=2 matmul of a zero band is a zero band and every op after it preserves
     * that, exactly as the 31 padding rows have always been preserved. Once a pass, not once
     * a layer: only the pass's FIRST strip can carry a previous pass's residue. */
    /* PHASE G: tall strips are zeroed over the bands written (this pass's, or a previous
     * pass's if wider), never the whole allocation. */
    if (bands > b->max_span)
        b->max_span = bands;
    /* S2f: a verify (<= 32 rows, one band) does not zero: the band's upload rewrites every
     * tile (rows above the count zero) and no op reads a band it does not span — the zeroing
     * was the prompt's whole span (51 bands after a 1,629-row prompt) every cycle. */
    PREFILL_WALL_OPEN(g, PN_PREFILL_WALL_STAGE);
    /* Prefill lane: the band-by-band upload below rewrites EVERY tile of bands [0, bands)
     * (the tilizer zeroes the rows above a band's count), so only the bands past the pass's
     * span need the zero — a 1,630-row prompt zeroed 13 MB over the link (0.045 s) for
     * nothing. `PN_TT_PREFILL_STAGE_ZERO=1` zeroes the whole span as before. */
    if (prefill_seek(g, 0u, bands) != 0 ||
        (!(g_verify_mode && bands == 1u) && !prefill_red("prefill_stage_skip_band1") &&
         (prefill_stage_zero_all()
              ? pn_tt_activation_zero_bands(g->device, &s->hidden[0],
                                            prefill_bands(g) > 4u ? b->max_span : 0u)
              : pn_tt_activation_zero_band_range(g->device, &s->hidden[0], bands,
                                                 prefill_bands(g) > 4u ? b->max_span
                                                                       : prefill_bands(g))) !=
             0))
        return -1;
    /* THE UPLOAD IS PER BAND AND THE TILIZER IS WHY. `ft_tt_tilize_strip()` takes `valid_rows`
     * 1..32 and writes ONE tile-row, zeroing the rows above the count — which is the strip
     * invariant, restored per band. A 64-row upload would need `ft_tt_tilize_matrix()` and
     * would buy one fewer host call; the invariant is worth more than the call. */
    for (uint32_t band = 0; band < bands && staged < rows; band++) {
        const int take = (rows - staged) < 32 ? (rows - staged) : 32;

        /* RED `prefill_stage_skip_band1`: a prompt pass reads band 1 of hidden[0] without
         * writing it (no zero, no upload) — the residue-read class the restage gate exists
         * to see. Must DIFFER (NaN under PN_TT_PREFILL_RESIDUE_POISON). */
        if (band == 1u && !g_verify_mode && prefill_red("prefill_stage_skip_band1")) {
            staged += take;
            continue;
        }
        if (prefill_seek(g, band, 1u) != 0 ||
            prefill_push(g, &s->hidden[0],
                         b->hidden + (size_t)staged * (size_t)cfg->n_embd, take) != 0)
            return -1;
        staged += take;
    }
    if (prefill_seek(g, 0u, bands) != 0)
        return -1;
    PREFILL_WALL_CLOSE(g, PN_PREFILL_WALL_STAGE);

    PREFILL_WALL_OPEN(g, PN_PREFILL_WALL_LAYERS);
    for (int layer = 0; layer < cfg->n_layer; layer++) {
        const pn_qwen35_tt_layer *d = &g->layers[layer];
        int idx = pn_qwen35moe_model_layer_index(m, layer);
        const double layer_mark = prefill_now();

        mark = layer_mark;
        /* (W wavefront: a prepared SSM layer's attn-norm already ran, super-band by super-band.) */
        if (!(!d->is_attention && !g_verify_mode &&
              pn_qwen35_tt_overlap_prepared(idx, prefill_span_for(g, rows))) &&
            pn_tt_rms_norm(g->device, &d->attn_norm, &s->hidden[s->slot], &s->normed) != 0)
            return -1;
        prefill_batched_add(b, 5, prefill_now() - mark); /* attn-norm */

        if (d->is_attention) {
            const bool next_ssm = layer + 1 < cfg->n_layer && !g->layers[layer + 1].is_attention;
            int arc;

            pn_qwen35_tt_overlap_forget();
            arc = prefill_attn_layer(
                g, d, cfg, idx, base_pos, rows, true, next_ssm ? &g->layers[layer + 1] : NULL,
                next_ssm ? pn_qwen35moe_model_layer_index(m, layer + 1) : -1);
            if (arc < 0)
                return -1;
            if (arc == 1) { /* A: the band loop already ran the rest of the layer */
                if (!g_verify_mode)
                    (void)pn_tt_device_keepalive(g->device, PN_QWEN35_TT_KEEPALIVE_IDLE_S);
                pn_prefill_wall_layer(true, prefill_now() - layer_mark);
                continue;
            }
        } else {
            const bool next_ssm = layer + 1 < cfg->n_layer && !g->layers[layer + 1].is_attention;
            const int src = prefill_ssm_layer(
                g, m, d, cfg, idx, base_pos, rows, next_ssm ? &g->layers[layer + 1] : NULL,
                next_ssm ? pn_qwen35moe_model_layer_index(m, layer + 1) : -1);

            if (src < 0)
                return -1;
            if (src == 1) { /* PHASE W2: the band loop already ran the rest of the layer */
                if (!g_verify_mode)
                    (void)pn_tt_device_keepalive(g->device, PN_QWEN35_TT_KEEPALIVE_IDLE_S);
                pn_prefill_wall_layer(false, prefill_now() - layer_mark);
                continue;
            }
        }
        if (prefill_residual(g) != 0)
            return -1;

        mark = prefill_now();
        if (pn_tt_rms_norm(g->device, &d->post_norm, &s->hidden[s->slot], &s->normed) != 0)
            return -1;
        prefill_batched_add(b, 6, prefill_now() - mark); /* post-norm */

        if ((g->moe ? prefill_moe_ffn(g, d, layer, rows) : prefill_ffn(g, d)) != 0 ||
            prefill_residual(g) != 0)
            return -1;
        /* PM 5 KEEPALIVE, per layer: one tall pass is seconds (tens at 8k) during which the
         * decode-only families (gdn, topk, ...) would idle past the retire limit. */
        if (!g_verify_mode)
            (void)pn_tt_device_keepalive(g->device, PN_QWEN35_TT_KEEPALIVE_IDLE_S);
        pn_prefill_wall_layer(d->is_attention, prefill_now() - layer_mark);
    }
    PREFILL_WALL_CLOSE(g, PN_PREFILL_WALL_LAYERS);
    /* A prompt pass never holds the prompt's last token, so every row has a next one. */
    /* `PN_TT_MTP_PROMPT=0`: no MTP KV for the prompt (prices what the prefill cost buys). */
    /* TTFT lane: the LAST pass only RECORDS it (qwen35moe_mtp_prompt_defer_tt.c) — it runs
     * after the first token, on the hidden this pass leaves in the tall set. */
    PREFILL_WALL_OPEN(g, PN_PREFILL_WALL_MTP);
    if (g->mtp.on && !g_verify_mode &&
        !(getenv("PN_TT_MTP_PROMPT") && atoi(getenv("PN_TT_MTP_PROMPT")) == 0)) {
        if (last && pn_qwen35_tt_mtp_prompt_defer_wanted(g)) {
            if (pn_qwen35_tt_mtp_prompt_owe(g, tokens + 1, rows, base_pos) != 0)
                return -1;
        } else if (prefill_mtp_layer(g, m, cfg, tokens + 1, rows, base_pos) != 0) {
            return -1;
        }
    }
    PREFILL_WALL_CLOSE(g, PN_PREFILL_WALL_MTP);
    b->passes++;
    b->rows_done += (uint64_t)rows;
    /* The forward's own alarm reads this: a process that has run a batched pass is a process
     * whose callers ask, so its single-position forwards are generation and not a prefill. */
    g->prefill_passes++;
    g->sequential_prefill_run = 0;
    return 0;
}

/**
 * The strips' state after the last pass. `PN_TT_PREFILL_RESTAGE=1` zeroes every band the
 * prompt wrote (the pre-2026-10-07 default: 2.38 s of a 1,630-token prefill, ~700 MB of zeros
 * over the link); `PN_TT_PREFILL_RESIDUE_POISON=1` fills the WHOLE tall allocation with 0xFF
 * (bf16 NaN) instead, the test that nothing reads the residue; default: neither.
 *
 * WHY THE ZERO IS NOT NEEDED ON THE TALL SET. With prefill-only tall strips (`tall_active`)
 * the decode strips are a different allocation the prompt never touched, so the residue is
 * read by nothing but the NEXT batched pass (a prompt, a verify, an MTP catch-up), and every
 * one of those writes each band it spans before reading it: hidden[0] is zeroed over the
 * pass's span and uploaded band by band (the tilizer zeroes rows above the count), the mixer
 * working sets are zeroed in `prefill_zero()`, every other strip is a producer's output over
 * the span. The S2f verify already relies on it (it never zeroes). Strips SHARED with decode
 * (tall strips off) keep the zero: there the next reader is a decode token.
 */
static bool prefill_restage_env(const char *name)
{
    const char *env = getenv(name);

    return env && *env && atoi(env) != 0;
}

static int prefill_restage(pn_qwen35_tt *g)
{
    pn_tt_activation *strips[PN_QWEN35_TT_MAX_STRIPS];
    const bool poison = g->tall_active && prefill_restage_env("PN_TT_PREFILL_RESIDUE_POISON");
    const bool zero = !g->tall_active || prefill_restage_env("PN_TT_PREFILL_RESTAGE");
    size_t n;

    /* BAND 0 AND A SPAN OF ONE FIRST, because everything after the prefill is a decode token
     * and a decode token is band 0. A strip left pointing at band 1 would send every later
     * projection into the wrong half of its own buffer, with every kernel retiring clean —
     * and `pn_tt_activation_zero()` zeroes the WHOLE allocation from band 0 whatever the seek
     * is, so the restage itself is correct either way. This line is about what comes after. */
    if (prefill_seek(g, 0u, 1u) != 0)
        return -1;
    n = prefill_strips(g, strips, PN_QWEN35_TT_MAX_STRIPS);
    for (size_t i = 0; (zero || poison) && i < n; i++)
        if (pn_tt_activation_fill_bands(g->device, strips[i],
                                        poison ? 0u : g->batch->max_span,
                                        poison ? 0xFFu : 0u) != 0)
            return -1;
    g->batch->max_span = 0u;
    return 0;
}

/**
 * A REFUSAL THAT SAYS ITS NAME, because a silent one is a shipped default degrading 20x.
 *
 * `pn_qwen35moe_prefill_tt()` returns NULL to mean "not taken" and the engine's per-token
 * loop then runs the whole prompt. That fallback is CORRECT and it is also ~20x slower than
 * the path that ships on by default, so a NULL nobody can see is indistinguishable from the
 * batched prefill having run. Measured 2026-08-26 on a clean checkout of main: a 1,630-token
 * prompt took 295 s and ~1,632 forwards, and the only line on stderr was the carried-state
 * reset — which is printed by the batched path on its way IN.
 *
 * @param reason  what stopped it, in the caller's own terms
 * @return        NULL always, so a refusal site is one `return`
 */
static tm_tensor *prefill_declined(const char *reason)
{
    fprintf(stderr, "qwen35_tt: BATCHED PREFILL DECLINED — %s. Nothing was touched and the "
                    "caller's per-token loop will produce the same answer, at roughly 20x "
                    "the wall.\n", reason);
    return NULL;
}

/* THE OTHER CLASS, AND IT IS NOT THE SAME EVENT. A pass that failed part way has already
 * written the card's conv window, recurrent state and K/V for the positions it did complete,
 * and `state->pos` has already advanced for every pass that succeeded before it. A per-token
 * replay over that state would be a wrong answer, so every such site below calls
 * pn_qwen35_tt_prefill_abort(): the session is invalidated and the request fails (the callers
 * ask pn_qwen35moe_prefill_tt_aborted). */

/** FAULT INJECTION `PN_TT_RED=prefill_pass_fail`: the second batched pass of a prompt (the
 *  first if the prompt has one) reports failure after the first wrote its state. */
static bool prefill_fault(int done)
{
    return prefill_red("prefill_pass_fail") && done > 0;
}

/**
 * G0: `sequential` ATTRIBUTED, with its denominator. Every card clock that moved inside a
 * sequential bracket, as seconds and as a share of `sequential`, then what NO clock contains
 * (host work between descriptors, transfers inside the mixers, anything unclocked) and the
 * brackets a residency armed inside, which are counted rather than split.
 */
static void prefill_report_sequential(const struct pn_qwen35_tt_batch *b)
{
    const double total = b->sequential_seconds > 0.0 ? b->sequential_seconds : 1.0;
    double clocked = 0.0;
    unsigned index;

    for (index = 0; index < b->sequential_clock_count; index++)
        if (b->sequential_clock_seconds[index] > 0.0)
            clocked += b->sequential_clock_seconds[index];
    fprintf(stderr, "qwen35_tt: sequential ATTRIBUTED — %.4f s of %.4f s clocked (%.1f%%), "
                    "%.4f s unclocked (%.1f%%), %.4f s in brackets a residency armed inside\n",
            clocked, b->sequential_seconds, 100.0 * clocked / total,
            b->sequential_seconds - clocked - b->sequential_unmatched_seconds,
            100.0 * (b->sequential_seconds - clocked - b->sequential_unmatched_seconds) / total,
            b->sequential_unmatched_seconds);
    for (index = 0; index < b->sequential_clock_count; index++)
        if (b->sequential_clock_seconds[index] > 0.0)
            fprintf(stderr, "qwen35_tt:   sequential clock %-16s %.4f s (%.1f%%)\n",
                    b->sequential_clock_names[index], b->sequential_clock_seconds[index],
                    100.0 * b->sequential_clock_seconds[index] / total);
}

/** MEASURE: `batched` by bracket site (each = its projections + flush, or its op). */
static void prefill_report_batched(const struct pn_qwen35_tt_batch *b)
{
    const double total = b->batched_seconds > 0.0 ? b->batched_seconds : 1.0;
    unsigned site;

    for (site = 0; site < 7u; site++)
        fprintf(stderr, "qwen35_tt:   batched site %-14s %.4f s (%.1f%%)\n",
                kPrefillBatchedNames[site], b->batched_site_seconds[site],
                100.0 * b->batched_site_seconds[site] / total);
}

static void prefill_report(const pn_qwen35_tt *g)
{
    const struct pn_qwen35_tt_batch *b = g->batch;
    double rows = (double)(b->rows_done ? b->rows_done : 1);

    fprintf(stderr, "qwen35_tt: batched prefill — %llu rows over %llu passes; "
                    "batched %.4f s (%.4f s/row) + sequential %.4f s (%.4f s/row) + "
                    "transfer %.4f s (%.4f s/row)\n",
            (unsigned long long)b->rows_done, (unsigned long long)b->passes,
            b->batched_seconds, b->batched_seconds / rows,
            b->sequential_seconds, b->sequential_seconds / rows,
            b->transfer_seconds, b->transfer_seconds / rows);
    prefill_report_sequential(b);
    prefill_report_batched(b);
}

tm_tensor *pn_qwen35moe_prefill_tt(pn_model *model, const uint32_t *tokens, int n_tokens,
                                   pn_state *state)
{
    pn_qwen35moe_model *m;
    const pn_qwen35moe_config *cfg;
    pn_qwen35_tt *g;
    tm_tensor *last = NULL;
    int rows, done, base, batched;
    bool resumed;

    /* NOT A DECLINE THIS FILE OWNS: the caller asks every model, and a prompt of one token
     * has no batch in it. Saying so on every llama turn would be noise. */
    if (!model || !state || !model->weights || !tokens || n_tokens < 2)
        return NULL;
    if (model->forward != pn_qwen35moe_forward_tt)
        return NULL;
    m = (pn_qwen35moe_model *)model->weights;
    g = (pn_qwen35_tt *)pn_qwen35moe_model_gpu(m);
    cfg = pn_qwen35moe_model_config(m);
    if (g) {
        g->prefill_last_batched = -1;
        g->prefill_aborted = false;
    }
    if (!prefill_enabled())
        return prefill_declined(PN_QWEN35_TT_PREFILL_ENV "=0 was asked for");
    if (!g || !cfg || !g->resident)
        return prefill_declined("the model is not resident on the card");
    if (g->moe && !pn_qwen35_tt_moe_prefill_ready(g))
        return prefill_declined("the MoE batched prefill is off (PN_TT_MOE_PREFILL=0) or has no "
                                "DRAM for its strips");
    /* REFUSALS, each of which hands the prompt back UNTOUCHED so the caller's per-token loop
     * runs. A refusal that half-ran a pass would be a wrong answer with no symptom.
     *
     * Both mixers must be the ON-CARD ones. With the host recurrence or host attention the
     * batch would have to carry `pn_qwen35moe_state`'s per-layer host state through a row
     * loop it was never written for, and the arm would be measuring a third forward. */
    if (!g->gdn_strip)
        return prefill_declined("the GDN strip bridge did not arm");
    if (!g->attn_on_card)
        return prefill_declined("attention is not on the card");
    /* Every instrument below assumes ONE position per forward: the crossed control runs a
     * whole second attention block, and the GDN gate and normcheck run shadow recurrences off
     * host state a batched pass does not maintain. */
    if (g->attn_compare || g->gdn_parity || g->gdn_normcheck)
        return prefill_declined("a per-forward instrument is armed "
                                "(PN_TT_ATTN_COMPARE / PN_TT_GDN_PARITY / PN_TT_GDN_NORMCHECK)");
    /* A continuation is batched only from a restored checkpoint: its passes are a cold run's
     * (qwen35moe_prefill_checkpoint_tt.c). Any other position runs per token. */
    resumed = pn_qwen35_tt_ckpt_restored_at(g, state, state->pos);
    if (state->pos != 0 && !resumed)
        return prefill_declined("the sequence does not start at position 0");
    base = state->pos;
    /* TTFT lane: a layer the previous prompt still owes runs first when this prompt RESUMES
     * that sequence (its MTP K/V below the checkpoint is that layer's); any other prefill
     * starts a sequence, which rewrites all of it. */
    if (resumed)
        (void)pn_qwen35_tt_mtp_prompt_settle(g, m, cfg, state->kv_cache);
    pn_qwen35_tt_mtp_prompt_drop(g, "a new sequence started on the card");
    pn_qwen35_tt_ckpt_note(g, state, state->pos);
    /* PHASE G0 MEASURE: one synthetic projection on layer 0's ffn_up, then the prefill as
     * usual (G0 uses its own buffers). Run several times for a best-of. */
    if (getenv("PN_TT_G0")) {
        int rep;

        for (rep = 0; rep < 3; rep++)
            (void)pn_tt_matmul_group_g0(g->device, &g->layers[0].ffn_up);
    }
    /* ON RECORD AND IT COSTS 4,096 SEQUENTIAL POSITIONS BEFORE IT BITES: `max_seq` defaults to
     * 4096, so a longer prompt used to run the per-token path to position 4096 and then fail
     * at `the KV append at position 4096 of layer 0 is not encodable`, with nothing having
     * said the batch declined. QWEN35_MAX_SEQ is the knob and it is named here. */
    if (state->pos + n_tokens > pn_qwen35moe_max_seq(cfg)) {
        char why[192];

        snprintf(why, sizeof(why), "the prompt is %d tokens and max_seq is %d — raise "
                                   "QWEN35_MAX_SEQ, or the per-token loop will run to %d and "
                                   "then fail at the KV append",
                 n_tokens, pn_qwen35moe_max_seq(cfg), pn_qwen35moe_max_seq(cfg));
        return prefill_declined(why);
    }

    /* A SECOND BAND NEEDS BOTH CHUNK PATHS, and the arms that select the per-position
     * fallback are named here rather than discovered inside a layer. The fallback moves ONE
     * host row at a time through `pn_tt_activation_upload()`, which writes logical row 0 of
     * whichever band the strip is pointed at and can address nothing else — so at more than
     * one band it would run every band's positions through band 0. A decline hands the prompt
     * back untouched; an abort would not.
     *
     * Decided on the PASS height, which is the TALL strips' when they exist: deciding on the
     * decode strips let `PN_TT_STRIP_TILE_ROWS=1` and `PN_TT_PREFILL_GDNC=0` through to a
     * 64-band pass that failed, ran the prompt per token and still returned logits — the
     * causality gate scored those arms green while no batched pass had run. */
    if (prefill_rows_wanted(g) > 32 &&
        (prefill_red("prefill_causal") || prefill_red("prefill_rowfill") ||
         !prefill_attn_chunk_enabled() || !pn_qwen35_tt_gdn_chunk_available(g, 32)))
        return prefill_declined("a pass wider than 32 rows needs both chunk paths — the "
                                "per-position fallback moves one host row at a time and "
                                "cannot address a second band (" PN_QWEN35_TT_PREFILL_ROWS_ENV
                                "=32 runs it)");

    pn_prefill_wall_set_rearm_clock(pn_tt_residency_lifecycle_seconds);
    pn_prefill_wall_begin(prefill_accounted(g));
    pn_sample_profiler_window(1);
    PREFILL_WALL_OPEN(g, PN_PREFILL_WALL_SETUP);
    prefill_tall_enter(g);
    rows = prefill_rows_wanted(g);
    if (prefill_alloc(g, cfg, rows) != 0) {
        (void)prefill_tall_exit(g);
        return prefill_declined("the host row scratch could not be allocated");
    }

    /* The card's carried state is per sequence and there is one region set per model, so a
     * new sequence zeroes it exactly where the host arm would get a fresh calloc. This is the
     * same seam `pn_qwen35_tt_resident_forward()` takes at pos 0, and the refusal above keeps
     * the batched arm to sequences that start there. */
    PREFILL_WALL_OPEN(g, PN_PREFILL_WALL_STATE_RESET);
    if (!resumed && pn_tt_gdn_state_reset(g->device) != 0) {
        (void)prefill_tall_exit(g);
        return prefill_declined("the carried state could not be zeroed");
    }
    PREFILL_WALL_CLOSE(g, PN_PREFILL_WALL_STATE_RESET);
    /* PHASE G: the prefill grid (`PN_TT_PREFILL_GRID`), started for this prefill only. */
    pn_tt_device_telemetry_log(g->device, "prefill start");
    /* The grid pays only when a projection has rows to share: booting 48 cores for a short
     * prompt cost ~40 ms and served 0 projections (run_d_defaults.sh). `PN_TT_PREFILL_GRID_MIN_ROWS`
     * (default 64) batched rows or more. */
    /* By THIS call's rows: grid and matmul group are bit-identical (pf64 with the grid off
     * IDENTICAL, run_prefill_ab_prefetch.sh), so a resumed tail may skip the boot. */
    /* DEFAULT 768 (run_prefill_grid_min.sh): at 508 rows the grid LOST 40 ms, at 1,629 it wins
     * 0.32 s; a resumed 113-row tail without it: TTFT 618 -> 453 ms, IDENTICAL. */
    /* Prefill lane (run_pg_park.sh, run_pg_defaults.sh): with the grid kept booted
     * (`PN_TT_PREFILL_GRID_PARK` default 0) its begin is 0.038 s; at 514 rows the grid wins
     * ~60 ms, at 298 it loses ~17-30, at 154 ~65 — floor 384. */
    if (n_tokens - 1 >= prefill_grid_min_rows()) {
        PREFILL_WALL_OPEN(g, PN_PREFILL_WALL_GRID_BEGIN);
        (void)pn_tt_matmul_grid_begin(g->device);
        PREFILL_WALL_CLOSE(g, PN_PREFILL_WALL_GRID_BEGIN);
    }
    PREFILL_WALL_CLOSE(g, PN_PREFILL_WALL_SETUP);

    /* Batch everything but the LAST token; that one goes through the ordinary forward, which
     * is what produces the logits and what keeps the head unmodified. */
    batched = prefill_batched_rows(g, base, n_tokens);
    /* L1a: the decode GDN cores serve this prompt's deep attention (PN_TT_PREFILL_LEND; on by
     * default for a dense-FFN model's long prompt, pn_tt_lend_min_rows). */
    (void)pn_tt_prefill_lend_begin(
        g->device, (uint32_t)batched,
        pn_tt_lend_min_rows(getenv("PN_TT_PREFILL_LEND"), pn_qwen35moe_dense_ffn(model) != 0),
        pn_tt_lend_deep_on(getenv("PN_TT_ATTN_DEEP"), true));
    pn_qwen35_tt_ckpt_plan(g, state, base, base + batched);
    done = 0;
    while (batched - done >= rows) {
        if (prefill_fault(done) ||
            prefill_pass(g, m, cfg, tokens + done, rows, state->pos, done + rows == batched) !=
                0) {
            (void)pn_tt_prefill_lend_end(g->device);
            pn_tt_matmul_grid_end(g->device);
            (void)prefill_tall_exit(g);
            return pn_qwen35_tt_prefill_abort(g, state, "a batched pass failed on the card",
                                              done);
        }
        done += rows;
        state->pos += rows;
        /* PM 5 KEEPALIVE: a long prompt leaves the decode-only families idle for seconds. */
        (void)pn_tt_device_keepalive(g->device, PN_QWEN35_TT_KEEPALIVE_IDLE_S);
    }
    /* THE SHORT TAIL PASS IS AN ABORT LIKE ANY OTHER. It used to be absorbed: the tail loop
     * below then ran its positions per token on top of the state the failed pass half wrote,
     * and Qwen3.8 at 34,816 served that corrupted answer on the p150. */
    if (batched - done > 0) {
        if (prefill_fault(done) ||
            prefill_pass(g, m, cfg, tokens + done, batched - done, state->pos, true) != 0) {
            (void)pn_tt_prefill_lend_end(g->device);
            pn_tt_matmul_grid_end(g->device);
            (void)prefill_tall_exit(g);
            return pn_qwen35_tt_prefill_abort(g, state, "the short tail pass failed", done);
        }
        state->pos += batched - done;
        done = batched;
    }
    pn_qwen35_tt_ckpt_commit(g, state, tokens, base, pn_qwen35moe_model_n_ssm(m));
    PREFILL_WALL_OPEN(g, PN_PREFILL_WALL_FLUSH);
    if (pn_tt_matvec_flush(g->device) != 0) {
        pn_qwen35_tt_mtp_prompt_drop(g, "the last pass did not complete");
        (void)pn_tt_prefill_lend_end(g->device);
        pn_tt_matmul_grid_end(g->device);
        (void)prefill_tall_exit(g);
        return pn_qwen35_tt_prefill_abort(g, state, "the last pass did not complete", done);
    }
    PREFILL_WALL_CLOSE(g, PN_PREFILL_WALL_FLUSH);
    /* L1a: every lent core back to its decode residency before anything decodes. */
    if (pn_tt_prefill_lend_end(g->device) != 0) {
        pn_qwen35_tt_mtp_prompt_drop(g, "a lent decode core did not come back");
        pn_tt_matmul_grid_end(g->device);
        (void)prefill_tall_exit(g);
        return pn_qwen35_tt_prefill_abort(g, state, "a lent decode core did not come back", done);
    }
    /* TTFT lane: a deferred MTP layer keeps the grid running for its projections. */
    PREFILL_WALL_OPEN(g, PN_PREFILL_WALL_GRID_END);
    if (!g->mtp.prompt_pending)
        pn_tt_matmul_grid_end(g->device);
    PREFILL_WALL_CLOSE(g, PN_PREFILL_WALL_GRID_END);
    if (pn_tt_device_dram_report_wanted()) {
        char tag[64];

        snprintf(tag, sizeof(tag), "after prefill to position %d", state->pos);
        pn_tt_device_report_dram(g->device, tag);
    }
    PREFILL_WALL_OPEN(g, PN_PREFILL_WALL_RESTAGE);
    if (prefill_restage(g) != 0) {
        pn_qwen35_tt_mtp_prompt_drop(g, "the strips could not be re-zeroed");
        (void)prefill_tall_exit(g);
        return pn_qwen35_tt_prefill_abort(
            g, state, "the strips could not be re-zeroed after the last pass", done);
    }
    PREFILL_WALL_CLOSE(g, PN_PREFILL_WALL_RESTAGE);
    PREFILL_WALL_OPEN(g, PN_PREFILL_WALL_TALL_EXIT);
    if (prefill_tall_exit(g) != 0) {
        pn_qwen35_tt_mtp_prompt_drop(g, "the decode mirror map did not republish");
        return pn_qwen35_tt_prefill_abort(
            g, state, "the decode strips' mirror map did not republish", done);
    }
    PREFILL_WALL_CLOSE(g, PN_PREFILL_WALL_TALL_EXIT);
    /* STEP 3a: everything is flushed (restage zeroes through the flush); give decode a clean
     * parameter cache rather than one full of this prefill's blocks. */
    PREFILL_WALL_OPEN(g, PN_PREFILL_WALL_PARAM_RESET);
    pn_tt_matmul_group_param_cache_reset(g->device);
    PREFILL_WALL_CLOSE(g, PN_PREFILL_WALL_PARAM_RESET);

    PREFILL_WALL_OPEN(g, PN_PREFILL_WALL_TAIL);
    for (int i = done; i < n_tokens; i++) {
        if (last) tm_free(last);
        last = pn_qwen35moe_forward_tt(model, tokens[i], state);
        if (!last) {
            pn_qwen35_tt_mtp_prompt_drop(g, "the tail forward failed");
            return pn_qwen35_tt_prefill_abort(
                g, state, "the tail forward that produces the logits failed", state->pos);
        }
    }
    PREFILL_WALL_CLOSE(g, PN_PREFILL_WALL_TAIL);
    /* TTFT lane: the logits exist; the prompt's MTP layer is now owed (the first draft, the
     * next forward or a resumed prefill runs it — qwen35moe_mtp_prompt_defer_tt.c). */
    pn_qwen35_tt_mtp_prompt_arm(g, state->kv_cache);
    g->prefill_last_batched = done;
    pn_sample_profiler_window(0);
    if (getenv(PN_QWEN35_TT_PREFILL_REPORT_ENV)) {
        prefill_report(g);
        pn_qwen35_tt_ab_prefetch_report();
        pn_qwen35_tt_overlap_report();
        fprintf(stderr, "qwen35_tt: prefill gate DEFER — %llu chain gate descriptor(s) deferred "
                        "(cumulative)\n",
                (unsigned long long)pn_tt_eltwise_group_deferred(g->device));
        pn_tt_gdngate_report(g->device);
        pn_prefill_wall_report(n_tokens, prefill_accounted(g));
        if (g_prefill_attn_grid_swaps)
            fprintf(stderr, "qwen35_tt: prefill ATTENTION GRID (C) — %u layer-passes on the "
                            "borrowed grid cores, %.4f s lending and returning (cumulative)\n",
                    g_prefill_attn_grid_swaps, g_prefill_attn_grid_seconds);
        if (g_prefill_attn_deep_passes)
            fprintf(stderr, "qwen35_tt: prefill DEEP ATTENTION — %u layer-passes: prep+append "
                            "%.4f s, queue %.4f s, drain %.4f s, gates %.4f s (host, "
                            "cumulative)\n", g_prefill_attn_deep_passes,
                    g_prefill_attn_deep_seconds[0], g_prefill_attn_deep_seconds[1],
                    g_prefill_attn_deep_seconds[2], g_prefill_attn_deep_seconds[3]);
        pn_tt_dispatch_census_print(); /* the prompt's own AWAIT census */
        /* The streamed segments so far (cumulative over the process): puller busy vs AWAIT,
         * starved launches (`PN_TT_HOST_GAP_STARVE=N`), boundaries — host- or card-bound. */
        pn_tt_dispatch_stream_report(1.0);
    }
    pn_tt_dispatch_census_restart(); /* the AWAIT census then covers decode only */
    return last;
}

int pn_qwen35moe_prefill_tt_aborted(const pn_model *model)
{
    const pn_qwen35_tt *g = NULL;

    if (model && model->weights && model->forward == pn_qwen35moe_forward_tt)
        g = (const pn_qwen35_tt *)pn_qwen35moe_model_gpu(
            (const pn_qwen35moe_model *)model->weights);
    return g && g->prefill_aborted ? 1 : 0;
}

int pn_qwen35moe_prefill_tt_last_batched(const pn_model *model)
{
    const pn_qwen35_tt *g = NULL;

    if (model && model->weights && model->forward == pn_qwen35moe_forward_tt)
        g = (const pn_qwen35_tt *)pn_qwen35moe_model_gpu(
            (const pn_qwen35moe_model *)model->weights);
    return g ? g->prefill_last_batched : -1;
}

/* =======================================================================================
 * STEP 4 S2 — THE SPECULATIVE VERIFY
 * ======================================================================================= */

/**
 * Run `rows` tokens (the seed, then the drafts) at positions `base_pos..` through ONE batched
 * pass whose projections, norms, SwiGLU and residuals serve every row and whose GDN and
 * attention mixers are the DECODE kernels run row by row (S0: bit-identical to per-token),
 * then each row's head through the DECODE head path (its hidden row copied to row 0), giving
 * each row's exact greedy token in `argmax_out[r]`. Leaves the carried state after the LAST
 * row; the earlier rows' states are in the slots pn_tt_gdn_verify_slots.h names, for
 * pn_qwen35_tt_spec_rollback().
 * K/V for every row is written at its position (a rejected row's is overwritten later).
 * @return 0, or -1 (nothing is reliable; the caller must stop speculating)
 */
int pn_qwen35_tt_spec_verify(pn_qwen35_tt *g, pn_qwen35moe_model *m,
                             const pn_qwen35moe_config *cfg, pn_qwen35moe_state *st,
                             const uint32_t *tokens, int rows, int base_pos,
                             uint32_t *argmax_out)
{
    int r, rc = 0;
    double t0 = pn_time_ms(), t1;

    if (!g || !m || !cfg || !st || !tokens || rows < 1 || rows > 16 || !argmax_out ||
        g->tall_rows == 0u)
        return -1;
    /* TTFT lane: the pass overwrites the tall hidden an owed MTP prompt layer reads. The
     * first draft always runs it before any verify (both entry points draft first); a verify
     * runs inside its stream, where a prompt-sized layer must not, so here it is dropped. */
    pn_qwen35_tt_mtp_prompt_forget(g, "a verify ran before the first draft");
    if (pn_tt_gdn_snapshot_reserve(g->device, pn_tt_gdn_verify_slots(g->device,
                                                                     (uint32_t)rows)) != 0)
        return -1;
    prefill_tall_enter(g);
    if (prefill_alloc(g, cfg, rows) != 0) {
        (void)prefill_tall_exit(g);
        return -1;
    }
    {
        const char *names0[PN_TT_CARD_CLOCKS_MAX], *names1[PN_TT_CARD_CLOCKS_MAX];
        double sec0[PN_TT_CARD_CLOCKS_MAX], sec1[PN_TT_CARD_CLOCKS_MAX];
        unsigned n0 = pn_tt_device_card_clocks(g->device, names0, sec0, PN_TT_CARD_CLOCKS_MAX);
        unsigned n1, i, j;

        g_verify_mode = true;
        /* S2c: the live-row promise — half tiles at <= 16 rows, eltwise E1 at <= 4
         * (`PN_TT_SPEC_HALF=0` off). */
        pn_tt_device_set_verify_rows(g->device,
                                     (!getenv("PN_TT_SPEC_HALF") ||
                                      atoi(getenv("PN_TT_SPEC_HALF")) != 0) ? (uint32_t)rows : 0u);
        rc = prefill_pass(g, m, cfg, tokens, rows, base_pos, false);
        pn_tt_device_set_verify_rows(g->device, 0u);
        g_verify_mode = false;
        n1 = pn_tt_device_card_clocks(g->device, names1, sec1, PN_TT_CARD_CLOCKS_MAX);
        for (i = 0; i < n1 && i < n0; i++) {
            for (j = 0; j < g->spec_clock_count; j++)
                if (strcmp(g->spec_clock_names[j], names1[i]) == 0)
                    break;
            if (j == g->spec_clock_count) {
                if (j == PN_TT_CARD_CLOCKS_MAX)
                    continue;
                g->spec_clock_names[j] = names1[i];
                g->spec_clock_seconds[j] = 0.0;
                g->spec_clock_count++;
            }
            g->spec_clock_seconds[j] += sec1[i] - sec0[i];
        }
    }
    if (rc == 0 && pn_tt_matvec_flush(g->device) != 0)
        rc = -1;
    /* Back on the decode strips with the decode mirror map republished FIRST: the decode
     * head's norm and LM head read replicated activations, and under the tall map they would
     * read the tall mirrors (the first S2a round emitted one token forever). */
    if (prefill_tall_exit(g) != 0)
        rc = -1;
    t1 = pn_time_ms();
    g->spec_pass_ms += t1 - t0;
    /* S2c: every row's head in one LM-head pass (`PN_TT_SPEC_HEADS=0`: the S2 form below). */
    if (rc == 0 && !(getenv("PN_TT_SPEC_HEADS") && atoi(getenv("PN_TT_SPEC_HEADS")) == 0)) {
        rc = pn_qwen35_tt_head_rows(g, st, &g->card_tall.hidden[g->card_tall.slot], rows,
                                    argmax_out);
        g->spec_head_ms += pn_time_ms() - t1;
        return rc;
    }
    /* THE HEADS: row r's hidden (still in the tall set) into the decode hidden's row 0, then
     * the DECODE head (norm, LM head, top-k) exactly as a single-position forward runs it. */
    for (r = 0; rc == 0 && r < rows; r++) {
        int v, best = 0;

        if (pn_tt_row_copy(g->device, &g->card_tall.hidden[g->card_tall.slot], (uint32_t)r,
                           &g->card.hidden[g->card.slot], 0u) != 0 ||
            pn_qwen35_tt_head_row(g, st) != 0) {
            rc = -1;
            break;
        }
        /* The engine's greedy rule: strict >, the lowest index wins a tie. */
        for (v = 1; v < st->n_vocab; v++)
            if (st->logits[v] > st->logits[best])
                best = v;
        argmax_out[r] = (uint32_t)best;
    }
    g->spec_head_ms += pn_time_ms() - t1;
    return rc;
}

/** Restore the carried state after row `last_kept` (0-based) of the last verify of `rows` rows,
 *  every layer: the state before row last_kept+1, i.e. after row last_kept, is in the slot the
 *  verify map names (legacy: slot last_kept+1). */
int pn_qwen35_tt_spec_rollback(pn_qwen35_tt *g, const pn_qwen35moe_config *cfg, int rows,
                               int last_kept)
{
    const uint32_t slot = pn_tt_gdn_verify_rollback_slot(g->device, (uint32_t)rows,
                                                         (uint32_t)last_kept);
    int layer, ssm = 0;

    for (layer = 0; layer < cfg->n_layer; layer++) {
        if (g->layers[layer].is_attention)
            continue;
        if (pn_tt_gdn_layer_swap(g->device, (uint32_t)ssm, slot) != 0)
            return -1;
        ssm++;
    }
    return 0;
}

int pn_qwen35_tt_mtp_catchup_rows(pn_qwen35_tt *g, pn_qwen35moe_model *m,
                                  const pn_qwen35moe_config *cfg, pn_qwen35moe_state *st,
                                  const uint32_t *next, int rows, int base_pos,
                                  uint32_t *argmax_out)
{
    const double started = pn_time_ms();
    int rc;

    if (!g->mtp.on || rows < 1 || rows > 16 || g->tall_rows == 0u)
        return -1;
    pn_qwen35_tt_mtp_prompt_forget(g, "a catch-up ran before the first draft");
    /* The verify set holds the committed rows' hiddens: enter it as it was left. */
    prefill_tall_enter(g);
    if (prefill_alloc(g, cfg, rows) != 0) {
        (void)prefill_tall_exit(g);
        return -1;
    }
    {
        const uint32_t bands = prefill_span_for(g, rows);
        pn_qwen35_tt_strips *s = &g->card;

        rc = 0;
        if (prefill_bands(g) > 4u && bands != g_prefill_published_span) {
            const pn_tt_activation *mirrored[2] = { &s->normed, &s->ffn_swiglu };

            pn_tt_replica_map_set_span(bands);
            rc = pn_tt_publish_replica_map(g->device, mirrored, 2u, &s->ffn_swiglu);
            g_prefill_published_span = bands;
        }
    }
    /* Exact per row is not required of a draft, but the verify's row paths make it so. */
    g_verify_mode = true;
    pn_tt_device_set_verify_rows(g->device, (uint32_t)rows);
    if (rc == 0)
        rc = prefill_mtp_layer(g, m, cfg, next, rows, base_pos);
    pn_tt_device_set_verify_rows(g->device, 0u);
    g_verify_mode = false;
    if (rc == 0 && pn_tt_matvec_flush(g->device) != 0)
        rc = -1;
    if (prefill_tall_exit(g) != 0)
        rc = -1;
    g->mtp.steps++;
    g->mtp.ms += pn_time_ms() - started;
    if (rc != 0)
        return -1;
    return pn_qwen35_tt_mtp_head_row(g, st, &g->card_tall.hidden[g->card_tall.slot],
                                     (uint32_t)rows - 1u, argmax_out);
}

/* PM 7: 192 rows, not 48. The prompt's SwiGLU must reach the WIDE eltwise members (W1a: a
 * descriptor of >= 2,048 tiles; 192 rows = 6 tile rows x 384 = 2,304): 48 rows (768 tiles)
 * left all 13 idle since the load, so the first long request retired and re-armed them
 * (run_serve_warm.sh: first 1,630 TTFT 2.73 s vs 2.65 after). Prefill lane: 390, so its 389
 * batched rows pass the grid floor (PN_QWEN35_TT_GRID_MIN_ROWS) and the warm-up BOOTS the grid,
 * which then stays booted for the first request. `PN_TT_WARM_ROWS` (2..390) overrides. */
#define PN_QWEN35_TT_WARM_ROWS 390

static void warm_discard_token(const char *token, void *user_data)
{
    (void)token;
    (void)user_data;
}

int pn_qwen35moe_warm_tt(pn_model *model)
{
    const char *env = getenv("PN_TT_WARM");
    const char *rows_env = getenv("PN_TT_WARM_ROWS");
    const int rows = rows_env && *rows_env ? atoi(rows_env) : PN_QWEN35_TT_WARM_ROWS;
    pn_sampling_params params;
    pn_qwen35_tt *g;
    pn_state *state;
    uint32_t tokens[PN_QWEN35_TT_WARM_ROWS];
    double started;
    int i, n;

    if (!model || !model->weights || model->forward != pn_qwen35moe_forward_tt ||
        (env && *env && atoi(env) == 0))
        return 0;
    g = (pn_qwen35_tt *)pn_qwen35moe_model_gpu((pn_qwen35moe_model *)model->weights);
    if (!g || !g->device)
        return 0;
    /* A batched pass (every chunk family) and then two decode tokens (every decode family).
     * The ids are arbitrary vocabulary; nothing is kept. */
    if (rows < 2 || rows > PN_QWEN35_TT_WARM_ROWS)
        return 0;
    for (i = 0; i < rows; i++)
        tokens[i] = 1000u + (uint32_t)i * 7u;
    state = pn_state_new(model);
    if (!state)
        return -1;
    params = pn_sampling_defaults();
    params.temperature = 0.0f;
    started = pn_time_ms();
    n = pn_stream_tokens(model, state, tokens, rows, 2, &params, warm_discard_token, NULL);
    pn_state_free(state);
    /* The card held the warm-up's sequence: no session may take it for its own. */
    pn_qwen35_tt_ckpt_forget(g);
    fprintf(stderr, "qwen35_tt: warm-up — %d-row prefill + %d token(s) in %.0f ms; every "
                    "family re-armed before the first request\n", rows, n,
            pn_time_ms() - started);
    return n < 0 ? -1 : 0;
}

/** PM 5: the serve's idle tick — keep this model's resident families armed (see
 *  pn_tt_device_keepalive). @return NOPs sent, 0 if none were due or not a TT model, -1 */
int pn_qwen35moe_keepalive_tt(pn_model *model)
{
    pn_qwen35_tt *g;

    if (!model || !model->weights || model->forward != pn_qwen35moe_forward_tt)
        return 0;
    g = (pn_qwen35_tt *)pn_qwen35moe_model_gpu((pn_qwen35moe_model *)model->weights);
    if (!g || !g->device)
        return 0;
    pn_qwen35_tt_mtp_prompt_idle(g);
    return pn_tt_device_keepalive(g->device, PN_QWEN35_TT_KEEPALIVE_IDLE_S);
}

#endif /* USE_TT */
