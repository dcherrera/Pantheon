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

#include "../accel/tt/pn_tt_dispatch.h"
#include "../accel/tt/pn_tt_eltwise_group.h"
#include "../accel/tt/pn_tt_gdn.h"
#include "../accel/tt/pn_tt_gdngate.h"
#include "../accel/tt/pn_tt_head_norm.h"
#include "../accel/tt/pn_tt_matmul_group.h"
#include "../accel/tt/pn_tt_replica_map.h"
#include "../accel/tt/pn_tt_residency.h"
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

/* The span the tall strips' mirrors were last keyed for (prefill_pass); 0 = re-key. */
static uint32_t g_prefill_published_span;

/**
 * PHASE G option 3 — PREFILL-ONLY TALL STRIPS (`PN_TT_PREFILL_TILE_ROWS`, a power of two
 * above the decode height, at most QWEN35_TT_TALL_MAX_ROWS; 0/unset = off).
 *
 * G1 made EVERY strip tall, and decode paid -1.4% for it (matmul host ISSUE +3 us, window
 * aims +20 a token) with the cause unidentified. This keeps the decode strips at their own
 * height and gives the batched passes a SECOND set of the same widths and mirror counts at
 * the tall height, swapped into `card` / `attn_card` for the passes and swapped back before
 * the tail forward. The logit strip is not copied: no batched pass runs the LM head.
 */
int pn_qwen35_tt_prefill_tall_alloc(pn_qwen35_tt *g)
{
    pn_tt_activation *small[PN_QWEN35_TT_MAX_STRIPS], *tall[PN_QWEN35_TT_MAX_STRIPS];
    /* DEFAULT 64 (G option 3, shipped with on-card dispatch: run_d_ceilings.sh). */
    const char *env = getenv("PN_TT_PREFILL_TILE_ROWS");
    const uint32_t rows = env ? (uint32_t)strtoul(env, NULL, 10) : 64u;
    size_t n, i;

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
    n = prefill_strips_of(&g->card, &g->attn_card, small, PN_QWEN35_TT_MAX_STRIPS, true);
    (void)prefill_strips_of(&g->card_tall, &g->attn_card_tall, tall, PN_QWEN35_TT_MAX_STRIPS,
                            true);
    for (i = 0; i < n; i++) {
        if (small[i]->tile_count == 0u || small[i] == &g->card.logits)
            continue;
        if (pn_tt_activation_alloc_replicated_rows(g->device, small[i]->tile_count,
                                                   small[i]->replica_count, rows,
                                                   tall[i]) != 0) {
            fprintf(stderr, "qwen35_tt: the prefill-only tall strips did not allocate\n");
            return -1;
        }
    }
    g->tall_rows = rows;
    pn_tt_device_set_prefill_tile_rows(g->device, rows);
    fprintf(stderr, "qwen35_tt: prefill-only TALL strips — %u bands for the batched passes, "
                    "decode keeps %u\n", rows, g->card.hidden[0].tile_rows);
    return 0;
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

/** hidden = hidden + project, on the card, into the spare strip. Per tile, so all 32 rows. */
static int prefill_residual(pn_qwen35_tt *g)
{
    int rc;

    PREFILL_WALL_OPEN(g, PN_PREFILL_WALL_RESIDUAL);
    rc = pn_tt_add(g->device, &g->card.hidden[g->card.slot], &g->card.project,
                   &g->card.hidden[g->card.slot ^ 1]);
    PREFILL_WALL_CLOSE(g, PN_PREFILL_WALL_RESIDUAL);
    if (rc != 0)
        return -1;
    g->card.slot ^= 1;
    return 0;
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
    (void)pn_qwen35_tt_gdn_pipe_drain(g);
    g->gdn_pipe = false;
    (void)pn_tt_eltwise_group_settle(g->device);
    (void)prefill_stream_close();
    (void)pn_qwen35_tt_overlap_drain(g);
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
    if (g_verify_mode || !pn_tt_gdngate_enabled(g->device) ||
        !pn_qwen35_tt_gdn_card_gate_possible())
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
    if (!prepared &&
        (pn_qwen35_tt_project(g, &d->attn_qkv, &s->normed, &s->ssm_qkv, false) != 0 ||
         pn_qwen35_tt_project(g, &d->attn_gate, &s->normed, &s->ssm_z, false) != 0 ||
         pn_qwen35_tt_project(g, &d->ssm_alpha, &s->normed, &s->ssm_alpha, false) != 0 ||
         pn_qwen35_tt_project(g, &d->ssm_beta, &s->normed, &s->ssm_beta, false) != 0 ||
         pn_tt_matvec_flush(g->device) != 0))
        return -1;
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
        const bool card = g->gdn_det_gates && !getenv("PN_TT_SPEC_ROWCOPY") &&
                          prefill_spec_defer() && prefill_card_decode_gates();

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
            /* Row r's new state goes to slot r, then trades places with the live one. */
            if (rc == 0)
                rc = pn_tt_gdn_state_out(g->device, r);
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
                (rc == 0 && pn_tt_gdn_layer_swap(g->device, (uint32_t)ssm_index, (uint32_t)r) != 0))
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
        if (pn_qwen35_tt_project(g, &d->ssm_out, &s->ssm_gated, &s->project, true) != 0)
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
        const uint32_t overlap_bands = pn_qwen35_tt_overlap_bands();
        const bool overlap = pn_qwen35_tt_overlap_layer(g, ssm_index, bands);

        /* The layer's gate blocks, settled ONCE before the first band, by the HOST and before
         * the stream opens: as a puller AWAIT inside the stream it stalled ~8.5 s a layer
         * (run_pm7_item5_gates.sh round 2: TTFT 204 s), while a host await is the gate time
         * itself — and a prepared layer's blocks are long done. */
        if (card_gates && pn_tt_gdngate_settle(g->device) != 0)
            return -1;
        prefill_stream_open(g);
        g->gdn_card_gate = card_gates;
        /* PM 7 ITEM 7 step 1: the pipelined band (`PN_TT_GDN_PIPE=0` off). */
        g->gdn_pipe = card_gates && g_prefill_streaming && prefill_gdn_pipe_wanted();
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
                    pn_tt_eltwise_group_settle(g->device) != 0 || prefill_stream_finish(g) != 0) {
                    prefill_chain_settle(g);
                    return -1;
                }
                pn_qwen35_tt_ckpt_band_done(g, ssm_index, base_pos + done);
                prefill_stream_open(g);
            } else {
                pn_qwen35_tt_ckpt_band_done(g, ssm_index, base_pos + done);
            }
            /* PHASE W2: a finished super-band's FFN goes to the grid; otherwise the job in
             * flight takes its next step if its projections are done. */
            /* STREAMED, a poll is answered at once and its AWAIT stalls the puller there, so
             * the job's next step is SCHEDULED instead: mid-super-band (the SwiGLU + down) and
             * at its close (the residual) — the grid runs under the chain ops issued between. */
            if (overlap) {
                const bool boundary = (band + 1u) % overlap_bands == 0u && band + 1u < bands &&
                                      done < rows;
                const bool step = !g_prefill_streaming ||
                                  (band + 1u) % overlap_bands == overlap_bands / 2u;

                /* The job reads `gated` (out-proj) and the grid rewrites z: gate settled. */
                if ((boundary || step) && (pn_qwen35_tt_gdn_pipe_drain(g) != 0 ||
                                           pn_tt_eltwise_group_settle(g->device) != 0)) {
                    prefill_chain_settle(g);
                    return -1;
                }
                rc = boundary ? pn_qwen35_tt_overlap_superband(g, d, band + 1u - overlap_bands,
                                                               overlap_bands, &next)
                     : step   ? pn_qwen35_tt_overlap_advance(g) : 0;
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
        if (pn_qwen35_tt_gdn_pipe_drain(g) != 0 || pn_tt_eltwise_group_settle(g->device) != 0 ||
            prefill_stream_finish(g) != 0) {
            g->gdn_pipe = false;
            prefill_chain_settle(g);
            return -1;
        }
        g->gdn_pipe = false;
        pn_qwen35_tt_ab_reader_end(g);
        if (overlap) {
            const uint32_t last = ((bands - 1u) / overlap_bands) * overlap_bands;
            int orc = pn_qwen35_tt_overlap_superband(g, d, last, bands - last, &next);

            if (orc == 0)
                orc = pn_qwen35_tt_overlap_drain(g);
            prefill_batched_add(b, 4, pn_qwen35_tt_overlap_take_seconds());
            if (orc != 0 || prefill_seek(g, 0u, bands) != 0)
                return -1;
            return 1;
        }
        if (prefill_seek(g, 0u, bands) != 0)
            return -1;

        mark = prefill_now();
        if (pn_qwen35_tt_project(g, &d->ssm_out, &s->ssm_gated, &s->project, true) != 0)
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
    if (pn_qwen35_tt_project(g, &d->ssm_out, &s->ssm_gated, &s->project, true) != 0)
        return -1;
    prefill_batched_add(b, 1, prefill_now() - mark); /* ssm-out-proj */
    return 0;
}

/** One attention layer over `rows` positions. Same shape as the SSM one: project once,
 *  rotate/append/attend `rows` times, project out once. */
static int prefill_attn_layer(pn_qwen35_tt *g, const pn_qwen35_tt_layer *d,
                              const pn_qwen35moe_config *cfg, int attn_index, int base_pos,
                              int rows)
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
                            (uint32_t)base_pos, (uint32_t)rows) != 0)
            return -1;
        for (int r = 0; rc == 0 && r < rows; r++)
            rc = (pn_tt_kv_append_row(g->device, (uint32_t)attn_index, false, &a->k_rot,
                                      (uint32_t)r, (uint32_t)(base_pos + r)) != 0 ||
                  pn_tt_kv_append_row(g->device, (uint32_t)attn_index, true, &s->attn_v,
                                      (uint32_t)r, (uint32_t)(base_pos + r)) != 0)
                     ? -1 : 0;
        /* S2f: flash decode for every row in ONE descriptor a group (ROWS mode) when every
         * row ends in the same position tile (`PN_TT_SPEC_ATTN_MULTI=0`: one a row). */
        if (rc == 0 && base_pos / 32 == (base_pos + rows - 1) / 32 && base_pos + rows < max_seq &&
            !(getenv("PN_TT_SPEC_ATTN_MULTI") && atoi(getenv("PN_TT_SPEC_ATTN_MULTI")) == 0)) {
            rc = pn_tt_attention_rows(g->device, (uint32_t)attn_index, &a->q_rot, &s->attn_out,
                                      (uint32_t)base_pos, (uint32_t)rows);
            rows_done = rows;
        }
        for (int r = rows_done; rc == 0 && r < rows; r++) {
            const int pos = base_pos + r;
            const int seq_len = (pos + 1 < max_seq) ? pos + 1 : max_seq;

            rc = (pn_tt_row_copy(g->device, &a->q_rot, (uint32_t)r, &da->q_rot, 0u) != 0 ||
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
        if (pn_qwen35_tt_project(g, &d->attn_output, &a->gated, &s->project, true) != 0)
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
        if (pn_qwen35_tt_project(g, &d->attn_output, &a->gated, &s->project, true) != 0)
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

        prefill_stream_open(g);
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
        if (pn_qwen35_tt_project(g, &d->attn_output, &a->gated, &s->project, true) != 0)
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
    if (pn_qwen35_tt_project(g, &d->attn_output, &a->gated, &s->project, true) != 0)
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

    if (pn_qwen35_tt_project(g, &d->ffn_gate, &s->normed, &s->ffn_gate, false) == 0 &&
        pn_qwen35_tt_project(g, &d->ffn_up, &s->normed, &s->ffn_up, false) == 0 &&
        pn_tt_matvec_flush(g->device) == 0 &&
        pn_tt_silu_mul(g->device, &s->ffn_gate, &s->ffn_up, &s->ffn_swiglu) == 0 &&
        pn_qwen35_tt_project(g, &d->ffn_down, &s->ffn_swiglu, &s->project, true) == 0)
        rc = 0;
    prefill_batched_add(b, 4, prefill_now() - mark); /* ffn */
    return rc;
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
    if (prefill_seek(g, 0u, bands) != 0 ||
        (!(g_verify_mode && bands == 1u) &&
         pn_tt_activation_zero_bands(g->device, e, prefill_bands(g) > 4u ? b->max_span : 0u) != 0))
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
        prefill_attn_layer(g, d, cfg, g->mtp.attn_index, base_pos, rows) != 0 ||
        prefill_residual(g) != 0 ||
        pn_tt_rms_norm(g->device, &d->post_norm, &s->hidden[s->slot], &s->normed) != 0 ||
        prefill_ffn(g, d) != 0 || prefill_residual(g) != 0)
        return -1;
    return 0;
}

static int prefill_pass(pn_qwen35_tt *g, pn_qwen35moe_model *m,
                        const pn_qwen35moe_config *cfg, const uint32_t *tokens, int rows,
                        int base_pos)
{
    struct pn_qwen35_tt_batch *b = g->batch;
    pn_qwen35_tt_strips *s = &g->card;
    double mark;

    const uint32_t bands = prefill_span_for(g, rows);
    int staged = 0;

    /* PHASE G: TALL strips (a whole prompt a pass) key their mirrors by the span the pass's
     * producers write (pn_tt_replica_map.c), so a pass whose span differs republishes. */
    if (prefill_bands(g) > 4u && bands != g_prefill_published_span) {
        const pn_tt_activation *mirrored[2] = { &s->normed, &s->ffn_swiglu };

        pn_tt_replica_map_set_span(bands);
        pn_tt_replica_map_set_superband(pn_qwen35_tt_overlap_bands());
        if (pn_tt_publish_replica_map(g->device, mirrored, 2u, &s->ffn_swiglu) != 0)
            return -1;
        g_prefill_published_span = bands;
    }

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
    if (prefill_seek(g, 0u, bands) != 0 ||
        (!(g_verify_mode && bands == 1u) && !prefill_red("prefill_stage_skip_band1") &&
         pn_tt_activation_zero_bands(g->device, &s->hidden[0],
                                     prefill_bands(g) > 4u ? b->max_span : 0u) != 0))
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

        mark = prefill_now();
        /* (W wavefront: a prepared SSM layer's attn-norm already ran, super-band by super-band.) */
        if (!(!d->is_attention && !g_verify_mode &&
              pn_qwen35_tt_overlap_prepared(idx, prefill_span_for(g, rows))) &&
            pn_tt_rms_norm(g->device, &d->attn_norm, &s->hidden[s->slot], &s->normed) != 0)
            return -1;
        prefill_batched_add(b, 5, prefill_now() - mark); /* attn-norm */

        if (d->is_attention) {
            pn_qwen35_tt_overlap_forget();
            if (prefill_attn_layer(g, d, cfg, idx, base_pos, rows) != 0)
                return -1;
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
                continue;
            }
        }
        if (prefill_residual(g) != 0)
            return -1;

        mark = prefill_now();
        if (pn_tt_rms_norm(g->device, &d->post_norm, &s->hidden[s->slot], &s->normed) != 0)
            return -1;
        prefill_batched_add(b, 6, prefill_now() - mark); /* post-norm */

        if (prefill_ffn(g, d) != 0 || prefill_residual(g) != 0)
            return -1;
        /* PM 5 KEEPALIVE, per layer: one tall pass is seconds (tens at 8k) during which the
         * decode-only families (gdn, topk, ...) would idle past the retire limit. */
        if (!g_verify_mode)
            (void)pn_tt_device_keepalive(g->device, PN_QWEN35_TT_KEEPALIVE_IDLE_S);
    }
    PREFILL_WALL_CLOSE(g, PN_PREFILL_WALL_LAYERS);
    /* A prompt pass never holds the prompt's last token, so every row has a next one. */
    /* `PN_TT_MTP_PROMPT=0`: no MTP KV for the prompt (prices what the prefill cost buys). */
    PREFILL_WALL_OPEN(g, PN_PREFILL_WALL_MTP);
    if (g->mtp.on && !g_verify_mode &&
        !(getenv("PN_TT_MTP_PROMPT") && atoi(getenv("PN_TT_MTP_PROMPT")) == 0) &&
        prefill_mtp_layer(g, m, cfg, tokens + 1, rows, base_pos) != 0)
        return -1;
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

/**
 * THE OTHER CLASS, AND IT IS NOT THE SAME EVENT. A pass that failed part way has already
 * written the card's conv window, recurrent state and K/V for the positions it did complete,
 * and `state->pos` has already advanced for every pass that succeeded before it. The caller
 * cannot tell this from a refusal — it sees the same NULL — so its per-token replay runs the
 * prompt again over state that is neither empty nor complete.
 *
 * @param reason         what failed
 * @param positions_done positions whose state is already on the card
 */
static tm_tensor *prefill_aborted(const char *reason, int positions_done)
{
    fprintf(stderr, "qwen35_tt: BATCHED PREFILL ABORTED after %d positions — %s. The card's "
                    "carried state and K/V are PARTLY WRITTEN; the caller's per-token replay "
                    "cannot undo them, so this answer is WRONG and not merely slow.\n",
            positions_done, reason);
    return NULL;
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
    bool resumed, tail_aborted = false;

    /* NOT A DECLINE THIS FILE OWNS: the caller asks every model, and a prompt of one token
     * has no batch in it. Saying so on every llama turn would be noise. */
    if (!model || !state || !model->weights || !tokens || n_tokens < 2)
        return NULL;
    if (model->forward != pn_qwen35moe_forward_tt)
        return NULL;
    m = (pn_qwen35moe_model *)model->weights;
    g = (pn_qwen35_tt *)pn_qwen35moe_model_gpu(m);
    cfg = pn_qwen35moe_model_config(m);
    if (g)
        g->prefill_last_batched = -1;
    if (!prefill_enabled())
        return prefill_declined(PN_QWEN35_TT_PREFILL_ENV "=0 was asked for");
    if (!g || !cfg || !g->resident)
        return prefill_declined("the model is not resident on the card");
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
    if (n_tokens - 1 >= (getenv("PN_TT_PREFILL_GRID_MIN_ROWS")
                             ? atoi(getenv("PN_TT_PREFILL_GRID_MIN_ROWS")) : 768))
    {
        PREFILL_WALL_OPEN(g, PN_PREFILL_WALL_GRID_BEGIN);
        (void)pn_tt_matmul_grid_begin(g->device);
        PREFILL_WALL_CLOSE(g, PN_PREFILL_WALL_GRID_BEGIN);
    }
    PREFILL_WALL_CLOSE(g, PN_PREFILL_WALL_SETUP);

    /* Batch everything but the LAST token; that one goes through the ordinary forward, which
     * is what produces the logits and what keeps the head unmodified. */
    batched = prefill_batched_rows(g, base, n_tokens);
    pn_qwen35_tt_ckpt_plan(g, state, base, base + batched);
    done = 0;
    while (batched - done >= rows) {
        if (prefill_pass(g, m, cfg, tokens + done, rows, state->pos) != 0) {
            pn_tt_matmul_grid_end(g->device);
            (void)prefill_tall_exit(g);
            return prefill_aborted("a batched pass failed on the card", done);
        }
        done += rows;
        state->pos += rows;
        /* PM 5 KEEPALIVE: a long prompt leaves the decode-only families idle for seconds. */
        (void)pn_tt_device_keepalive(g->device, PN_QWEN35_TT_KEEPALIVE_IDLE_S);
    }
    /* THE SHORT TAIL PASS IS THE ONE FAILURE THIS FUNCTION ABSORBS, and absorbing it silently
     * was wrong: a pass that failed part way has already written some of those positions, and
     * the tail loop below then runs them AGAIN on top of that state. Loud, and it keeps the
     * old behaviour rather than turning a working prompt into a refusal. */
    if (batched - done > 0) {
        if (prefill_pass(g, m, cfg, tokens + done, batched - done, state->pos) == 0) {
            state->pos += batched - done;
            done = batched;
        } else {
            tail_aborted = true;
            (void)prefill_aborted("the short tail pass failed; the remaining positions run "
                                  "per token OVER the state it half wrote", done);
        }
    }
    pn_qwen35_tt_ckpt_commit(g, state, tokens, base, pn_qwen35moe_model_n_ssm(m));
    PREFILL_WALL_OPEN(g, PN_PREFILL_WALL_FLUSH);
    if (pn_tt_matvec_flush(g->device) != 0) {
        pn_tt_matmul_grid_end(g->device);
        (void)prefill_tall_exit(g);
        return prefill_aborted("the last pass did not complete", done);
    }
    PREFILL_WALL_CLOSE(g, PN_PREFILL_WALL_FLUSH);
    PREFILL_WALL_OPEN(g, PN_PREFILL_WALL_GRID_END);
    pn_tt_matmul_grid_end(g->device);
    PREFILL_WALL_CLOSE(g, PN_PREFILL_WALL_GRID_END);
    PREFILL_WALL_OPEN(g, PN_PREFILL_WALL_RESTAGE);
    if (prefill_restage(g) != 0) {
        (void)prefill_tall_exit(g);
        return prefill_aborted("the strips could not be re-zeroed after the last pass", done);
    }
    PREFILL_WALL_CLOSE(g, PN_PREFILL_WALL_RESTAGE);
    PREFILL_WALL_OPEN(g, PN_PREFILL_WALL_TALL_EXIT);
    if (prefill_tall_exit(g) != 0)
        return prefill_aborted("the decode strips' mirror map did not republish", done);
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
        if (!last)
            return prefill_aborted("the tail forward that produces the logits failed",
                                   state->pos);
    }
    PREFILL_WALL_CLOSE(g, PN_PREFILL_WALL_TAIL);
    g->prefill_last_batched = tail_aborted ? -1 : done;
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
    }
    return last;
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
 * row; snapshots of every earlier row are in slots 0..rows-2 for pn_qwen35_tt_spec_rollback().
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
    if (pn_tt_gdn_snapshot_reserve(g->device, (uint32_t)rows) != 0)
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
        rc = prefill_pass(g, m, cfg, tokens, rows, base_pos);
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

/** Restore the carried state after row `last_kept` (0-based) of the last verify, every layer:
 *  slot last_kept+1 holds the state before row last_kept+1, i.e. after row last_kept. */
int pn_qwen35_tt_spec_rollback(pn_qwen35_tt *g, const pn_qwen35moe_config *cfg, int last_kept)
{
    int layer, ssm = 0;

    for (layer = 0; layer < cfg->n_layer; layer++) {
        if (g->layers[layer].is_attention)
            continue;
        if (pn_tt_gdn_layer_swap(g->device, (uint32_t)ssm, (uint32_t)last_kept + 1u) != 0)
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
 * (run_serve_warm.sh: first 1,630 TTFT 2.73 s vs 2.65 after). `PN_TT_WARM_ROWS` (2..192)
 * overrides, for the A/B. */
#define PN_QWEN35_TT_WARM_ROWS 192

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
    return g && g->device ? pn_tt_device_keepalive(g->device, PN_QWEN35_TT_KEEPALIVE_IDLE_S) : 0;
}

#endif /* USE_TT */
