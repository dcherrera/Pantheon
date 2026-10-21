/* SPDX-License-Identifier: AGPL-3.0-or-later */
/* Copyright (C) 2026 David (Corey) Herrera */
/**
 * @file qwen35moe_gdn_strip_tt.c
 * @brief THE STRIP BRIDGE: the whole Gated-Delta-Net block on the card, with no activation
 *        crossing PCIe. This is the file that removes 78% of a decode token's readback.
 *
 * ---------------------------------------------------------------------------
 * THE NUMBER THIS FILE EXISTS FOR
 * ---------------------------------------------------------------------------
 * A decode token was 0.390 s of HOST against 0.050 s of card, and 0.29 s of the host half was
 * TRANSFER. The Gated-Delta-Net block owned **49,408 B a layer over 24 SSM layers** of it:
 * `ssm_qkv` and `ssm_z` came down, the recurrence ran, `gated` went back up. `qwen35moe_gdn_tt.c`
 * moved the two heavy PHASES onto the card and made the transfer WORSE, because the per-head
 * L2 norm sits between them and forced `conv_out` and `y` across as well — 1.32 tok/s armed
 * against 1.95 unarmed, `cmp`-identical and correct, and a wall regression for one reason
 * only.
 *
 * Nothing about that seam was arithmetic. It was a FORMAT and a LAYOUT:
 *
 *   FORMAT  every projection writes a bf16 strip; the GDN op and the per-head reduces are
 *           Float32 programs, because the recurrence carries state across tokens. Fixed by
 *           `pn_tt_render.h` — two resident cores that convert a strip in card DRAM.
 *   LAYOUT  PHASE DELTA used to take three tilized `[heads, S]` matrices, one row per head,
 *           which is not the geometry a projection or a reduce produces. It takes HEAD-MAJOR
 *           ROW-0 STRIPS now, which is exactly PHASE CONV's output geometry — so the delta
 *           rule reads the convolution's answer IN PLACE and the L2 norm between them writes
 *           back over its own input.
 *
 * With both, the block is six descriptors and two 256-byte host transfers a layer:
 *
 *     widen(ssm_qkv)          bf16 strip  -> the GDN stream buffer, fp32
 *     PHASE CONV                          -> conv_out, which IS the delta input
 *     L2 head norm, IN PLACE, over its first `2*k_heads*state_tiles` tiles
 *     PHASE DELTA                         -> y, an fp32 row-0 strip
 *     RMS head norm                       -> y_norm, fp32
 *     narrow(y_norm)          fp32        -> a bf16 strip
 *     silu(z) * that, on the shipped elementwise residency -> `ssm_gated`
 *
 * `alpha` and `beta` still come DOWN (two tiles, 256 B) and the gate block goes UP (256 B),
 * because `beta = sigmoid(.)` and `decay = exp(ssm_a * softplus(.))` need a `pow` and so a
 * `log`, and the SFPU has neither. That is the whole of the block's link traffic:
 * **readback 1478.09 -> 326.09 KiB/token.**
 *
 * ---------------------------------------------------------------------------
 * WHY THE L2 NORM IN PLACE IS A DECISION AND NOT A HOPE
 * ---------------------------------------------------------------------------
 * Its input and its output are the same 256-tile buffer: `conv_out`'s q and k slices, normed
 * back over themselves, with v left alone beside them. That is safe by PIPELINE ORDER — the
 * reduce reader prefetches ONE row block, so it has read block `b` before the writer can have
 * written block `b` — and the alternative is a whole extra fp32 copy descriptor a layer to
 * move `v` into a second buffer.
 *
 * `PN_TT_GDN_STRIP_CHECK=1` is the arm that makes that a measurement rather than a claim: it
 * reads the conv output's q|k PREIMAGE off the card, runs the descriptor, reads the ANSWER
 * back, and scores it per head against `pn_qwen35moe_l2_norm()` on the preimage. A reduce
 * racing its own input cannot pass that — it would be normalising values it had already
 * overwritten. It also checks that `v`, which this descriptor must not touch, is byte for
 * byte what PHASE CONV left there.
 *
 * @status SHIPPED ON by default, MEASURED ON SILICON (p150a, 2026-08-21), against a control
 *         re-taken on the same binary and the same blob with `PN_TT_PROGRAM_DIR` unset:
 *
 *           arm        readback KiB/token   wall tok/s   cmp vs the 48-token oracle
 *           control          1478.09           2.20      IDENTICAL, 189 B, md5 80fc6259
 *           gdn, no bridge   2630.09           1.43      IDENTICAL
 *           THE BRIDGE        326.09           3.79      IDENTICAL
 *
 *         **4.53x the readback and 1.72x the wall, with the output BYTE-IDENTICAL** — so the
 *         narrowing render and both per-head norms moving on-card cost nothing the 48-token
 *         completion can see, and no parity ratio is needed to justify the default.
 *         Decomposed: readback 0.287 -> 0.065 s/token, wall 0.4543 -> 0.2639, and the bridge's
 *         own card time is ~0.022 s/token over 48 render descriptors — 22 ms spent to remove
 *         222 ms. 0 refusals, 0 restarts, company=0 on every arm.
 * @issues None
 * @todo None
 */

#ifdef USE_TT

#include "qwen35moe_tt_internal.h"

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "../accel/tt/pn_tt_fp32_strip.h"
#include "../core/bench.h"
#include "../accel/tt/pn_tt_eltwise_group.h"
#include "../accel/tt/pn_tt_gate.h"
#include "../accel/tt/pn_tt_gdn.h"
#include "../accel/tt/pn_tt_gdnc.h"
#include "../accel/tt/pn_tt_gdngate.h"
#include "tt/kernels/resident_gdngate_protocol.h"
#include "tt/kernels/ft_det_math.h"
#include "../accel/tt/pn_tt_head_norm.h"
#include "../accel/tt/pn_tt_render.h"

/**
 * ON BY DEFAULT; `PN_TT_GDN_STRIP=0` turns it off.
 *
 * It flipped on the two numbers this file exists for, measured on a p150a against a control
 * re-taken on the same binary and the same blob: readback **1478.09 -> 326.09 KiB/token** and
 * wall **2.20 -> 3.79 tok/s**, with the 48-token completion `cmp` IDENTICAL to the oracle —
 * 189 B, md5 80fc6259 — so the output gate is satisfied by BYTE IDENTITY rather than by a
 * parity ratio. `0` reproduces the host-mediated arm at 1.43 tok/s, which is what makes the
 * gain attributable.
 */
#define PN_QWEN35_TT_GDN_STRIP_ENV "PN_TT_GDN_STRIP"

/** The in-place proof below. Budgeted, off by default, nothing downstream reads it. */
#define PN_QWEN35_TT_GDN_STRIP_CHECK_ENV    "PN_TT_GDN_STRIP_CHECK"
#define PN_QWEN35_TT_GDN_STRIP_CHECK_BUDGET 24

/**
 * WHICH OF THE TWO CHUNK KERNELS RUNS. `both` is the arm as it was measured and the default.
 *
 * A CPU oracle scored the chunkwise prefill on 2026-08-24 and it FAILED: the mean is unbiased
 * (t = +0.433) and the SPREAD doubles, `dnll_sd` 0.1943 -> 0.3776. `PN_TT_PREFILL=1` alone —
 * batched projections, batched attention, both GDN mixers still the decode family — scores
 * BIT-IDENTICALLY to the per-token path, so all of that variance belongs to PHASE CONV, to
 * PHASE DELTA, or to their interaction. Nothing else could separate the two: they share the
 * conv window, the recurrent state and all three buffers, and `PN_TT_PREFILL_GDNC` moves them
 * together.
 *
 *   both    the arm as it stands, both mixers chunkwise
 *   conv    PHASE CONV chunkwise, PHASE DELTA per position over the chunk's rows
 *   delta   PHASE CONV per position, PHASE DELTA chunkwise
 *   none    BOTH per position, through the SAME row shuttle — **the control**, and the only
 *           thing that makes the other two readable. It must score bit-identically to the
 *           sequential arm; if it does not, the shuttle is the divergence and no attribution
 *           it produces means anything.
 *
 * The three non-default modes are SLOW — a row of the stream crosses the link per position
 * per layer, which is the traffic the chunk exists to remove. They are a quality instrument.
 */
#define PN_QWEN35_TT_GDNC_PHASE_ENV "PN_TT_PREFILL_GDNC_PHASE"

typedef enum {
    PN_GDNC_PHASE_BOTH = 0,
    PN_GDNC_PHASE_CONV,
    PN_GDNC_PHASE_DELTA,
    PN_GDNC_PHASE_NONE
} pn_gdnc_phase;

/**
 * THE TWO MIXERS' CHUNK LENGTH, INSIDE A 32-ROW PASS. 0 (the default) is the whole pass.
 *
 * `PN_TT_PREFILL_ROWS` cannot answer the chunk-length question. It shortens the PASS, which
 * moves the attention chunk with it, and — measured 2026-08-24 — it WEDGES core 90: with the
 * SSM chunked the decode `gdn` family goes unused for a whole prefill, its first descriptor
 * after it is refused, and TRISC0 spins out its bound. The boot guard refuses rather than
 * wedging worse and rung 3 clears it, but the arm produces no dump. At the default 32 the
 * same arm runs clean.
 *
 * So the sub-chunk runs inside a 32-row pass: the pass, the attention chunk, the projections
 * and the decode family all stay exactly where the passing `both` arm leaves them, and the
 * ONLY variable is C. It uses the same row shuttle, which the `none` control proved
 * byte-exact — and `PN_TT_PREFILL_GDNC_ROWS=32` reproduces the `both` arm, which is this
 * knob's own control.
 */
#define PN_QWEN35_TT_GDNC_ROWS_ENV "PN_TT_PREFILL_GDNC_ROWS"

/** 0 when unset or out of range; said out loud once when it is not. */
static int gdnc_sub_rows(void)
{
    static int rows = -1;
    const char *env;

    if (rows >= 0)
        return rows;
    env = getenv(PN_QWEN35_TT_GDNC_ROWS_ENV);
    rows = env && *env ? atoi(env) : 0;
    if (rows < 0 || rows > 32)
        rows = 0;
    if (rows != 0)
        fprintf(stderr, "qwen35_tt: gdnc SUB-CHUNK ARMED — the two GDN mixers run at C = %d "
                        "inside a 32-row pass, through the row shuttle. An attribution "
                        "instrument, and SLOW.\n", rows);
    return rows;
}

/**
 * PHASE DELTA'S CHUNK LENGTH, IN PLACE. **16 by default — this is the shipped rung.**
 *
 * ---------------------------------------------------------------------------
 * WHAT IT IS, AND WHY IT IS NOT `PN_TT_PREFILL_GDNC_ROWS`
 * ---------------------------------------------------------------------------
 * `PN_TT_PREFILL_GDNC_ROWS` above answers the chunk-length QUESTION through a row shuttle: a
 * row of the stream crosses the link per position per layer, which is the traffic the chunk
 * exists to remove. It is an instrument and it is slow. This knob is the ANSWER: PHASE DELTA
 * runs `ceil(rows/C)` descriptors over the SAME 32-row strip, each owning rows
 * `[first_row, first_row + C)` IN PLACE, and nothing crosses the link at all.
 *
 * ---------------------------------------------------------------------------
 * WHY 16, MEASURED
 * ---------------------------------------------------------------------------
 * PHASE DELTA owns 100% of the chunkwise arm's quality variance and the chunk length is the
 * whole of it. `A = I + A_off` is WELL CONDITIONED (`max|A^-1| = 1.0`, `cond ~ 30`) but
 * `A_off`'s powers are not monotone: at C = 32, `max|A_off^8| = 3.6e+04` and the Newton
 * iterates reach 1.7e+06 before collapsing to ~1, so the answer is a cancellation of
 * 1e5-sized intermediates and a 4.9e-4 Tf32 unpack truncation leaves a residue it cannot
 * remove. At C = 16 the intermediates never get large enough for that to bite. The ladder,
 * scored against one cached oracle at 339 positions: C = 4 ratio 0.993285 / sd 0.1747;
 * C = 8 1.000834 / 0.1705; **C = 16 0.995699 / t -0.477 / 4.13% / KL 0.00460 / sd 0.1664 /
 * worst-doc 1.0289 — 6/6 and the best figures of the whole ladder**; C = 32 FAILS
 * worst-document.
 *
 * ---------------------------------------------------------------------------
 * PHASE CONV STAYS AT THE WHOLE PASS, AND THAT IS NOT A COMPROMISE
 * ---------------------------------------------------------------------------
 * The convolution is a fixed `taps` FIR, not a length-sensitive recurrence: output row t reads
 * input rows `t-(taps-1)..t`, which for `t >= taps-1` are inside the strip whatever the
 * chunking, and the band tiles already select them. A sub-chunked CONV and a whole-pass CONV
 * accumulate the same `2*taps` terms in the same tap order, differing only in which of each
 * pair carries the value and which carries an exact zero. So CONV runs ONCE — one descriptor
 * instead of two — and the measured `chunkwise CONV alone PASSES 6/6 and beats what ships` is
 * the arm that says the whole-pass CONV is fine. The L2 norm between them runs once for the
 * same reason: a ROW reduce reduces ALONG the row, so every position is normalised
 * independently and 32 rows at once is 32 rows one at a time.
 *
 * `0` or anything `>= rows` is the whole pass, which is the C = 32 arm and this knob's own
 * control: it must be BYTE-IDENTICAL to the dump the sub-chunk path did not exist for.
 * Anything not a multiple of 16 is refused, because `first_row` must land on a tile FACE
 * boundary for the writer's `y` write to stay a contiguous suffix.
 */
#define PN_QWEN35_TT_DELTA_ROWS_ENV     "PN_TT_PREFILL_DELTA_ROWS"
#define PN_QWEN35_TT_DELTA_ROWS_DEFAULT 16

/** Read once, and said out loud whenever it is not the default. */
static int delta_sub_rows(void)
{
    static int rows = -1;
    const char *env;

    if (rows >= 0)
        return rows;
    env = getenv(PN_QWEN35_TT_DELTA_ROWS_ENV);
    rows = (env && *env) ? atoi(env) : PN_QWEN35_TT_DELTA_ROWS_DEFAULT;
    if (rows < 0 || rows > 32 || (rows % 16) != 0) {
        fprintf(stderr, "qwen35_tt: %s=%s is not 0, 16 or 32 — `first_row` must land on a "
                        "tile face boundary. Staying at %d.\n",
                PN_QWEN35_TT_DELTA_ROWS_ENV, env ? env : "", PN_QWEN35_TT_DELTA_ROWS_DEFAULT);
        rows = PN_QWEN35_TT_DELTA_ROWS_DEFAULT;
    }
    if (rows != PN_QWEN35_TT_DELTA_ROWS_DEFAULT)
        fprintf(stderr, "qwen35_tt: PHASE DELTA chunk length is C = %d, not the shipped %d\n",
                rows, PN_QWEN35_TT_DELTA_ROWS_DEFAULT);
    return rows;
}

/** `PN_TT_GDNC_STAGE_OVERLAP`: stage each DELTA sub-chunk's slot under the card time of the
 *  descriptor before it (CONV for sub-chunk 0, sub-chunk 0's DELTA for sub-chunk 1). ON by
 *  default: 234.0 -> 244.4 tok/s, 2,448 of 2,448 stagings hidden, IDENTICAL, read-back 0. */
static bool gdnc_stage_overlap(void)
{
    static int cached = -1;

    if (cached < 0) {
        const char *value = getenv("PN_TT_GDNC_STAGE_OVERLAP");

        cached = (value && *value) ? (strcmp(value, "0") != 0) : 1;
        fprintf(stderr, "qwen35_tt: PHASE DELTA staging %s\n",
                cached ? "runs UNDER the previous gdnc descriptor's card time (overlap)"
                       : "runs between descriptors (serial)");
    }
    return cached != 0;
}

/** One sub-chunk to prestage from the overlap hook. */
typedef struct {
    pn_tt_device *device;
    uint32_t layer, first_row, chunk_rows;
    const float *decay, *beta;
} gdn_prestage_job;

static int gdn_prestage_run(void *context)
{
    const gdn_prestage_job *job = (const gdn_prestage_job *)context;

    return pn_tt_gdnc_delta_prestage_at(job->device, job->layer, job->first_row,
                                        job->chunk_rows, job->decay, job->beta);
}

/** The sub-chunk starting at `first` as a prestage job. */
static gdn_prestage_job gdn_prestage_job_at(pn_qwen35_tt *g, int ssm_index, int v_heads,
                                            int rows, int sub, int first)
{
    gdn_prestage_job job;

    job.device = g->device;
    job.layer = (uint32_t)ssm_index;
    job.first_row = (uint32_t)first;
    job.chunk_rows = (uint32_t)((rows - first < sub) ? rows - first : sub);
    job.decay = g->gdn_chunk_decay + (size_t)first * v_heads;
    job.beta = g->gdn_chunk_beta + (size_t)first * v_heads;
    return job;
}

/**
 * PHASE DELTA over the pass, at the shipped chunk length, IN PLACE.
 *
 * The recurrent state carries across sub-chunks exactly as it carries across passes — neither
 * this file nor the kernels touch it in between — and the strip is untouched, so `decay` and
 * `beta` are just sliced. One iteration when C covers the pass, and then this is byte for
 * byte the descriptor that shipped.
 */
uint32_t pn_qwen35_tt_gdn_gate_sub_rows(void)
{
    const int sub = delta_sub_rows();

    return sub == 0 ? 32u : (uint32_t)sub;
}

/** PM 7 ITEM 5: the band's DELTA descriptors on the card-built gate blocks — the same split
 *  as below (`ftgg_sub_live_rows()` is that split), nothing staged. */
static int gdn_delta_card_gate(pn_qwen35_tt *g, int ssm_index, int rows)
{
    const uint32_t sub_rows = pn_qwen35_tt_gdn_gate_sub_rows();

    for (uint32_t sub = 0; sub < 2u; sub++) {
        const uint32_t live = ftgg_sub_live_rows((uint32_t)rows, sub_rows, sub);

        if (live == 0u)
            continue;
        if (pn_tt_gdnc_delta_oncard_card_gate(
                g->device, (uint32_t)ssm_index, sub * sub_rows, live,
                pn_tt_gdngate_block_address(g->device, (uint32_t)ssm_index,
                                            g->gdn_card_gate_band, sub)) != 0)
            return -1;
    }
    return 0;
}

static int gdn_delta_in_place(pn_qwen35_tt *g, const pn_qwen35moe_config *cfg, int ssm_index,
                              int rows)
{
    const int v_heads = cfg->ssm_dt_rank;
    const int sub = delta_sub_rows();
    int first;

    if (g->gdn_card_gate)
        return gdn_delta_card_gate(g, ssm_index, rows);

    if (sub == 0 || sub >= rows)
        return pn_tt_gdnc_delta_oncard(g->device, (uint32_t)ssm_index, (uint32_t)rows,
                                       g->gdn_chunk_decay, g->gdn_chunk_beta);
    for (first = 0; first < rows; first += sub) {
        const int live = (rows - first < sub) ? rows - first : sub;
        gdn_prestage_job next;

        /* The NEXT sub-chunk stages under this one's card time, into the other slot. */
        if (gdnc_stage_overlap() && first + sub < rows) {
            next = gdn_prestage_job_at(g, ssm_index, v_heads, rows, sub, first + sub);
            pn_tt_gdnc_set_overlap(g->device, gdn_prestage_run, &next);
        }
        if (pn_tt_gdnc_delta_oncard_at(g->device, (uint32_t)ssm_index, (uint32_t)first,
                                       (uint32_t)live,
                                       g->gdn_chunk_decay + (size_t)first * v_heads,
                                       g->gdn_chunk_beta + (size_t)first * v_heads) != 0)
            return -1;
    }
    return 0;
}

/** Read once and SAID OUT LOUD once — an isolation arm that silently did not fire is the
 *  failure mode this whole measurement is exposed to. */
static pn_gdnc_phase gdnc_phase(void)
{
    static pn_gdnc_phase phase = PN_GDNC_PHASE_BOTH;
    static bool read = false;
    const char *env;

    if (read)
        return phase;
    read = true;
    env = getenv(PN_QWEN35_TT_GDNC_PHASE_ENV);
    if (!env || !*env || strcmp(env, "both") == 0)
        return phase;
    if (strcmp(env, "conv") == 0)
        phase = PN_GDNC_PHASE_CONV;
    else if (strcmp(env, "delta") == 0)
        phase = PN_GDNC_PHASE_DELTA;
    else if (strcmp(env, "none") == 0)
        phase = PN_GDNC_PHASE_NONE;
    else
        fprintf(stderr, "qwen35_tt: %s=%s is not one of both|conv|delta|none — staying on "
                        "`both`\n", PN_QWEN35_TT_GDNC_PHASE_ENV, env);
    if (phase != PN_GDNC_PHASE_BOTH)
        fprintf(stderr, "qwen35_tt: gdnc phase split ARMED — %s. PHASE %s is chunkwise, the "
                        "other runs PER POSITION through the row shuttle. This is an "
                        "attribution instrument and it is SLOW.\n",
                env, phase == PN_GDNC_PHASE_CONV    ? "CONV"
                     : phase == PN_GDNC_PHASE_DELTA ? "DELTA"
                                                    : "NEITHER");
    return phase;
}

bool pn_qwen35_tt_gdn_card_gate_possible(void)
{
    return gdnc_phase() == PN_GDNC_PHASE_BOTH && gdnc_sub_rows() == 0;
}

int pn_qwen35_tt_gdn_strip_arm(pn_qwen35_tt *g, const pn_qwen35moe_config *cfg)
{
    const uint32_t state = (uint32_t)cfg->ssm_state;
    uint32_t stream_tiles;
    uint32_t value_tiles;
    const char *want;

    if (!g || !g->device || !cfg)
        return -1;
    /* Default ON: only an explicit `0` turns it off. */
    want = getenv(PN_QWEN35_TT_GDN_STRIP_ENV);
    if (want && *want && atoi(want) == 0) {
        fprintf(stderr, "qwen35_tt: %s=0 — the strip bridge is OFF by request; the block's "
                        "activations cross the link again\n", PN_QWEN35_TT_GDN_STRIP_ENV);
        return 0;
    }
    if (!g->gdn_on_card) {
        fprintf(stderr, "qwen35_tt: %s needs the GDN family armed — add `gdn` to "
                        "PN_TT_ONCARD. The block stays on the host.\n",
                PN_QWEN35_TT_GDN_STRIP_ENV);
        return 0;
    }
    /* THE TWO EXISTING GDN PROBES SCORE THE HOST-MEDIATED ARM AND CANNOT SEE THIS ONE.
     * `PN_TT_GDN_PARITY` runs a shadow host recurrence beside `pn_qwen35_tt_gdn_recurrence()`
     * and `PN_TT_GDN_NORMCHECK` scores the two per-head norms against `gdn_conv_out`/`gdn_y`
     * — neither function is on this path, so with the bridge on they would run ZERO
     * comparisons and report a clean sheet. That is a gate passing because it never fired,
     * which this project has now hit seven times, so the bridge STANDS DOWN and says why
     * rather than letting a probe read as green. */
    if (getenv("PN_TT_GDN_PARITY") || getenv("PN_TT_GDN_NORMCHECK")) {
        fprintf(stderr, "qwen35_tt: PN_TT_GDN_PARITY / PN_TT_GDN_NORMCHECK score the "
                        "HOST-MEDIATED arm and would compare nothing at all against the strip "
                        "bridge — standing the bridge down so the probe measures what it "
                        "claims to. Unset it to run the bridge.\n");
        return 0;
    }
    stream_tiles = pn_tt_gdn_stream_tiles(g->device);
    value_tiles = pn_tt_gdn_value_tiles(g->device);
    if (stream_tiles == 0u || value_tiles == 0u)
        return 0;

    /* The widest strip each direction sees: the qkv stream one way, `y_norm` the other. */
    if (pn_tt_device_arm_render(g->device, stream_tiles, value_tiles) != 0)
        return 0;
    if (pn_tt_device_arm_head_norm(g->device, state) != 0)
        return 0;
    /* THE MERGED GATE, OPT-IN. It takes `render_narrow` out of the DECODE token; the render
     * stays armed because the chunkwise prefill path still narrows through it, and a residency
     * that armed is a residency the stats table reports at zero ops rather than one that
     * silently was not there. */
    if (pn_tt_device_arm_gate_narrow(g->device, value_tiles) != 0)
        return 0;
    if (!pn_tt_render_widen_is_armed(g->device) || !pn_tt_render_narrow_is_armed(g->device) ||
        !pn_tt_head_norm_is_armed(g->device, state)) {
        fprintf(stderr, "qwen35_tt: %s asked for the strip bridge and one of the three "
                        "residencies it needs did not arm — the block stays on the "
                        "host-mediated path\n", PN_QWEN35_TT_GDN_STRIP_ENV);
        return 0;
    }

    g->gdn_rms = calloc((size_t)cfg->n_layer, sizeof(*g->gdn_rms));
    g->gdn_rms_ready = calloc((size_t)cfg->n_layer, sizeof(*g->gdn_rms_ready));
    if (!g->gdn_rms || !g->gdn_rms_ready)
        return -1;
    /* The L2 member takes eps*eps: the oracle clamps the MAGNITUDE at eps and the eps tile
     * folds into the SQUARE. foundry `ft_tt_op_vector.h` carries the derivation. It is ONE
     * buffer for the model — the L2 norm carries no gamma, so nothing in it is per layer. */
    if (pn_tt_head_norm_constants_upload(g->device, FT_TT_OP_REDUCE_FN_L2_NORM, NULL, state,
                                         cfg->rms_norm_eps * cfg->rms_norm_eps,
                                         &g->gdn_l2) != 0)
        return -1;
    if (pn_tt_fp32_strip_alloc(g->device, value_tiles, &g->gdn_y_norm) != 0 ||
        pn_tt_activation_alloc(g->device, value_tiles, &g->gdn_y_bridge) != 0)
        return -1;
    /* The in-place probe's scratch — three q|k widths and two v widths, allocated only when
     * the probe is asked for so the shipped arm pays nothing for it. */
    if (getenv(PN_QWEN35_TT_GDN_STRIP_CHECK_ENV)) {
        g->gdn_probe = calloc((size_t)3u * pn_tt_gdn_key_tiles(g->device) * 32u +
                                  (size_t)2u * value_tiles * 32u,
                              sizeof(*g->gdn_probe));
        if (!g->gdn_probe)
            return -1;
    }

    /* THE CHUNK's gate scratch, one tile-row of positions wide. Allocated only when the
     * chunkwise family armed, so a decode-only process pays nothing for it. */
    if (pn_tt_gdnc_is_armed(g->device)) {
        g->gdn_chunk_decay = calloc(32u * (size_t)cfg->ssm_dt_rank, sizeof(float));
        g->gdn_chunk_beta = calloc(32u * (size_t)cfg->ssm_dt_rank, sizeof(float));
        if (!g->gdn_chunk_decay || !g->gdn_chunk_beta)
            return -1;
    }

    /* THE PHASE-ISOLATION ARM's row shuttle, allocated only when it is asked for. */
    if (pn_tt_gdnc_is_armed(g->device) &&
        (gdnc_phase() != PN_GDNC_PHASE_BOTH || gdnc_sub_rows() != 0)) {
        g->gdn_split_stream = calloc(32u * (size_t)stream_tiles * 32u, sizeof(float));
        g->gdn_split_y = calloc(32u * (size_t)value_tiles * 32u, sizeof(float));
        if (!g->gdn_split_stream || !g->gdn_split_y)
            return -1;
    }

    g->gdn_strip = true;
    fprintf(stderr, "qwen35_tt: THE STRIP BRIDGE IS ON — the Gated-Delta-Net block runs whole "
                    "on the card. Per SSM layer the link carries alpha+beta DOWN (256 B) and "
                    "the gate block UP (256 B); the activations do not cross at all.\n");
    return 0;
}

void pn_qwen35_tt_gdn_strip_release(pn_qwen35_tt *g)
{
    if (!g)
        return;
    free(g->gdn_rms);
    free(g->gdn_rms_ready);
    free(g->gdn_probe);
    free(g->gdn_chunk_decay);
    free(g->gdn_chunk_beta);
    free(g->gdn_split_stream);
    free(g->gdn_split_y);
    g->gdn_rms = NULL;
    g->gdn_rms_ready = NULL;
    g->gdn_probe = NULL;
    g->gdn_chunk_decay = NULL;
    g->gdn_chunk_beta = NULL;
    g->gdn_split_stream = NULL;
    g->gdn_split_y = NULL;
    g->gdn_strip = false;
}

/**
 * THE IN-PLACE PROOF, and it is the only thing in this file that reads the card.
 *
 * The q/k L2 norm writes over its own input. That is safe by pipeline order, and this is what
 * turns "by pipeline order" into a number: score the card's answer, per head, against the host
 * oracle applied to the PREIMAGE this function read a moment earlier. A reduce that raced its
 * own input would be normalising values it had already overwritten, and the two would not
 * agree to Tf32 truncation.
 *
 * `v` is checked too, and it is the cheaper half of the same question: this descriptor covers
 * the first `2*k_heads*state_tiles` tiles of a 256-tile buffer, so if its write pointer walked
 * past its row blocks, `v` is where it would land — and `v` is PHASE DELTA's operand.
 *
 * A PROBE: budgeted, off by default, and nothing downstream reads it.
 */

/** Max-norm relative error of `got` against `want`, PER HEAD. A per-head fault is invisible
 *  in a per-tensor figure — this family scored 1.000e+00 for three rounds on exactly that. */
static double gdn_strip_worst_head(const float *want, const float *got, int heads, int width)
{
    double worst = 0.0;
    int head;

    for (head = 0; head < heads; head++) {
        double max_abs = 0.0, max_want = 0.0;
        int index;

        for (index = 0; index < width; index++) {
            const double diff = fabs((double)got[(size_t)head * width + index] -
                                     (double)want[(size_t)head * width + index]);
            const double magnitude = fabs((double)want[(size_t)head * width + index]);

            if (diff > max_abs)
                max_abs = diff;
            if (magnitude > max_want)
                max_want = magnitude;
        }
        if (max_want > 0.0 && max_abs / max_want > worst)
            worst = max_abs / max_want;
    }
    return worst;
}

/**
 * Read the conv-output q|k PREIMAGE, run the descriptor, score the answer. Returns the
 * descriptor's own result so the caller cannot accidentally skip a failure.
 */
static int gdn_strip_l2_checked(pn_qwen35_tt *g, const pn_qwen35moe_config *cfg, int ssm_index,
                                ft_tt_op_operand stream, uint32_t stream_tiles)
{
    const int state = cfg->ssm_state;
    const int k_heads = cfg->ssm_n_group;
    const uint32_t key_tiles = pn_tt_gdn_key_tiles(g->device);
    const uint32_t value_tiles = pn_tt_gdn_value_tiles(g->device);
    const size_t key_count = (size_t)key_tiles * 32u;
    const size_t value_count = (size_t)value_tiles * 32u;
    /* ITS OWN SCRATCH, sized from the two counts it actually reads. The first spelling of
     * this probe borrowed `gdn_qn`, which is `k_heads * state` — HALF the q|k strip — and
     * overran it: the arm aborted and printed a worst head of 1.7e+00 against a path whose
     * 48-token completion is byte-identical to the oracle. A probe that reports a failure it
     * caused itself is worse than no probe. */
    float *preimage = g->gdn_probe;
    float *want = preimage + key_count;
    float *got = want + key_count;
    float *v_before = got + key_count;
    float *v_after = v_before + value_count;
    double worst;
    int head;
    int rc;

    if (!g->gdn_probe)
        return -1;
    if (pn_tt_gdn_stream_read(g->device, 0u, key_tiles, preimage) != 0 ||
        pn_tt_gdn_stream_read(g->device, key_tiles, value_tiles, v_before) != 0)
        return -1;
    /* The oracle's own function on the preimage, per k-head: q's heads then k's. */
    memcpy(want, preimage, key_count * sizeof(float));
    for (head = 0; head < 2 * k_heads; head++)
        pn_qwen35moe_l2_norm(want + (size_t)head * state, state, cfg->rms_norm_eps);

    rc = pn_tt_head_norm_operands(g->device, FT_TT_OP_REDUCE_FN_L2_NORM, &g->gdn_l2,
                                  (uint32_t)(2 * k_heads), stream, stream, stream_tiles,
                                  stream_tiles);
    if (rc != 0)
        return rc;
    if (pn_tt_gdn_stream_read(g->device, 0u, key_tiles, got) != 0 ||
        pn_tt_gdn_stream_read(g->device, key_tiles, value_tiles, v_after) != 0)
        return -1;

    worst = gdn_strip_worst_head(want, got, 2 * k_heads, state);
    fprintf(stderr, "qwen35_tt: [strip] ssm %2d  L2 IN PLACE worst head %.3e  v untouched %s\n",
            ssm_index, worst,
            memcmp(v_before, v_after, value_count * sizeof(float)) == 0 ? "YES" : "NO");
    return 0;
}

/** This layer's gated-RMS constants, built on first use. `gamma` is `ssm_norm`, per layer. */
static int gdn_strip_rms_constants(pn_qwen35_tt *g, const pn_qwen35moe_ssm_layer *l,
                                   const pn_qwen35moe_config *cfg, int ssm_index)
{
    if (g->gdn_rms_ready[ssm_index])
        return 0;
    if (pn_tt_head_norm_constants_upload(g->device, FT_TT_OP_REDUCE_FN_RMS_NORM, l->ssm_norm,
                                         (uint32_t)cfg->ssm_state, cfg->rms_norm_eps,
                                         &g->gdn_rms[ssm_index]) != 0)
        return -1;
    g->gdn_rms_ready[ssm_index] = true;
    return 0;
}

/** One position's gates on the HOST: the engine's libm arithmetic, or (det gates) the same
 *  integer ft_det_math the card runs — on float bits that are bf16 << 16, as the card sees. */
static void gdn_gate_host(const pn_qwen35_tt *g, const pn_qwen35moe_ssm_layer *l,
                          const float *alpha, const float *beta_raw, int v_heads, float *decay,
                          float *beta)
{
    int head;

    if (!g->gdn_det_gates) {
        pn_qwen35moe_beta_sigmoid(beta_raw, v_heads, beta);
        pn_qwen35moe_decay_gate(l, alpha, v_heads, decay);
        for (head = 0; head < v_heads; head++)
            decay[head] = expf(decay[head]);
        return;
    }
    for (head = 0; head < v_heads; head++) {
        uint32_t a, dt, ssm_a, b, out;

        memcpy(&a, &alpha[head], sizeof(a));
        memcpy(&dt, &l->ssm_dt[head], sizeof(dt));
        memcpy(&ssm_a, &l->ssm_a[head], sizeof(ssm_a));
        memcpy(&b, &beta_raw[head], sizeof(b));
        out = ft_det_gdn_gate(a, dt, ssm_a);
        memcpy(&decay[head], &out, sizeof(out));
        out = ft_det_gdn_beta(b);
        memcpy(&beta[head], &out, sizeof(out));
    }
}

/** ITEM 6: this many rows' gates straight into the gate buffer the next DELTA reads, computed
 *  on the card from band 0 of the alpha/beta strips. */
static int gdn_gate_card(pn_qwen35_tt *g, int ssm_index, uint32_t rows)
{
    uint32_t bank, address, stride;

    if (pn_tt_gdn_gate_target(g->device, rows, &bank, &address, &stride) != 0)
        return -1;
    return pn_tt_gdngate_flat(g->device, (uint32_t)ssm_index, &g->card.ssm_alpha,
                              &g->card.ssm_beta, rows, bank, address, stride);
}

int pn_qwen35_tt_gdn_strip_layer(pn_qwen35_tt *g, const pn_qwen35moe_ssm_layer *l,
                                 const pn_qwen35moe_config *cfg, int ssm_index,
                                 const pn_tt_activation *qkv, const pn_tt_activation *z,
                                 const float *beta_raw, const float *alpha,
                                 const pn_tt_activation *gated)
{
    const int v_heads = cfg->ssm_dt_rank;
    const uint32_t stream_tiles = pn_tt_gdn_stream_tiles(g->device);
    const uint32_t value_tiles = pn_tt_gdn_value_tiles(g->device);
    ft_tt_op_operand stream;

    /* ITEM 6: `alpha == beta_raw == NULL` asks for the gates ON THE CARD (det gates only). */
    const bool card = !alpha && !beta_raw && g && g->gdn_det_gates;

    if (!g || !l || !cfg || !qkv || !z || (!card && (!beta_raw || !alpha)) || !gated)
        return -1;
    if (!g->gdn_strip || ssm_index < 0)
        return -1;
    if (gdn_strip_rms_constants(g, l, cfg, ssm_index) != 0)
        return -1;

    /* 1 and 2, the oracle's own, on the two operands that still cross. `decay = exp(g)` is
     * applied HERE because the card has no `log` and therefore no `pow`. */
    if (!card)
        gdn_gate_host(g, l, alpha, beta_raw, v_heads, g->gdn_decay, g->gdn_beta);

    /* 3. The bf16 projection becomes the fp32 stream PHASE CONV reads, in card DRAM. */
    if (pn_tt_render_widen(g->device, pn_tt_operand(g->device, qkv),
                           pn_tt_gdn_stream_in_operand(g->device), stream_tiles) != 0)
        return -1;
    if (pn_tt_gdn_conv_oncard(g->device, (uint32_t)ssm_index) != 0)
        return -1;

    /* 4. The q/k L2 norm, IN PLACE over the first `2*k_heads*state_tiles` tiles of the conv
     * output — one row block per head, `state_tiles` tile columns wide, exactly the
     * `[1, heads*S]` reduce `pn_tt_head_norm.h` describes. `v` is the rest of the same buffer
     * and this descriptor never reaches it. The GQA repeat is NOT materialised: the DELTA
     * reader's `vh % k_heads` is the repeat, and staging the 32-head form is the 1.33x-WORSE
     * row of the split-point table. */
    stream = pn_tt_gdn_stream_operand(g->device);
    if (g->gdn_strip_checks < PN_QWEN35_TT_GDN_STRIP_CHECK_BUDGET &&
        getenv(PN_QWEN35_TT_GDN_STRIP_CHECK_ENV)) {
        g->gdn_strip_checks++;
        if (gdn_strip_l2_checked(g, cfg, ssm_index, stream, stream_tiles) != 0)
            return -1;
    } else if (pn_tt_head_norm_operands(g->device, FT_TT_OP_REDUCE_FN_L2_NORM, &g->gdn_l2,
                                        (uint32_t)(2 * cfg->ssm_n_group), stream, stream,
                                        stream_tiles, stream_tiles) != 0) {
        return -1;
    }

    /* 5. PHASE DELTA over that same buffer, plus 256 B of gate block (ITEM 6: written by the
     * card, not uploaded). */
    if (card ? (gdn_gate_card(g, ssm_index, 1u) != 0 ||
                pn_tt_gdn_delta_descriptor_only(g->device, (uint32_t)ssm_index) != 0)
             : pn_tt_gdn_delta_oncard(g->device, (uint32_t)ssm_index, g->gdn_decay,
                                      g->gdn_beta) != 0)
        return -1;

    /* 6. The gated norm, in two halves at two formats: the RMS in the Float32 reduce family
     * where the recurrence's precision lives, then the render, then `silu(z) *` on the SHIPPED
     * bf16 elementwise residency. The render is the term `gated` is bound by — see
     * pn_tt_render.h for what it costs and pn_tt_gate.h for what removes it. */
    if (pn_tt_head_norm_operands(g->device, FT_TT_OP_REDUCE_FN_RMS_NORM, &g->gdn_rms[ssm_index],
                                 (uint32_t)v_heads, pn_tt_gdn_y_operand(g->device),
                                 pn_tt_fp32_strip_operand(g->device, &g->gdn_y_norm),
                                 value_tiles, g->gdn_y_norm.tile_count) != 0)
        return -1;
    /* THE MERGED GATE, WHEN IT IS ARMED: one descriptor instead of two, reading `y_norm` as
     * fp32 straight into SrcB. The unpack that narrows it is the SAME truncating fp32->bf16
     * conversion `render_narrow` performs one descriptor earlier — verified off-hardware to be
     * the same `unpack_src / unpack_dst / tile_size` triple — so this is expected to be `cmp`
     * IDENTICAL and a DIFFERS is a finding, not a cost. */
    if (pn_tt_gate_narrow_is_armed(g->device))
        return pn_tt_gate_narrow(g->device,
                                 pn_tt_operand(g->device, z),
                                 pn_tt_fp32_strip_operand(g->device, &g->gdn_y_norm),
                                 pn_tt_operand(g->device, gated), value_tiles);
    if (pn_tt_render_narrow(g->device, pn_tt_fp32_strip_operand(g->device, &g->gdn_y_norm),
                            pn_tt_operand(g->device, &g->gdn_y_bridge), value_tiles) != 0)
        return -1;
    return pn_tt_silu_mul(g->device, z, &g->gdn_y_bridge, gated);
}

/** PM 5: the multi-row DELTA verify mode, ON unless `PN_TT_SPEC_DELTA_ROWS=0`. */
static bool delta_multi_row_enabled(void)
{
    static int cached = -1;

    if (cached < 0) {
        const char *value = getenv("PN_TT_SPEC_DELTA_ROWS");

        cached = (value && *value) ? (strcmp(value, "0") != 0) : 1;
        fprintf(stderr, "qwen35_tt: verify DELTA rows %s\n",
                cached ? "in ONE multi-row descriptor (PN_TT_SPEC_DELTA_ROWS=0 off)"
                       : "one descriptor a row");
    }
    return cached != 0;
}

/** PM 6: the multi-row CONV verify mode, ON unless `PN_TT_SPEC_CONV_ROWS=0`. */
static bool conv_multi_row_enabled(void)
{
    static int cached = -1;

    if (cached < 0) {
        const char *value = getenv("PN_TT_SPEC_CONV_ROWS");

        cached = (value && *value) ? (strcmp(value, "0") != 0) : 1;
        fprintf(stderr, "qwen35_tt: verify CONV rows %s\n",
                cached ? "in ONE multi-row descriptor (PN_TT_SPEC_CONV_ROWS=0 off)"
                       : "one descriptor a row");
    }
    return cached != 0;
}

/**
 * STEP 4 S2c: the same six steps over the first `rows` rows of a strip (the speculative
 * verify), with the DECODE recurrences run row by row through the kernels' row select
 * (`FT_TT_OP_GDN_ROW_SHIFT`) and everything row-wise run ONCE over all rows: the widen, the
 * q/k L2 norm, the gated RMS norm and the gate. Every CONV runs before the L2 norm and every
 * DELTA after it, exactly the per-row order of each row's own data. Row r's new window and
 * state go to snapshot slot r (STATE_OUT) and trade places with the live ones, so slot r keeps
 * the state before row r (the rollback). Bit-exactness per row is the gate's to show.
 */
int pn_qwen35_tt_gdn_strip_rows(pn_qwen35_tt *g, const pn_qwen35moe_ssm_layer *l,
                                const pn_qwen35moe_config *cfg, int ssm_index, int rows,
                                const pn_tt_activation *qkv, const pn_tt_activation *z,
                                const float *beta_raw, size_t beta_stride,
                                const float *alpha, size_t alpha_stride,
                                const pn_tt_activation *gated)
{
    const int v_heads = cfg->ssm_dt_rank;
    const uint32_t stream_tiles = pn_tt_gdn_stream_tiles(g->device);
    const uint32_t value_tiles = pn_tt_gdn_value_tiles(g->device);
    ft_tt_op_operand stream;
    int r, rc = 0;
    /* S2c: the rows' CONVs (then DELTAs) queue behind each other and are awaited once
     * (`PN_TT_SPEC_DEFER=0`: one await a row). */
    const bool defer = !getenv("PN_TT_SPEC_DEFER") || atoi(getenv("PN_TT_SPEC_DEFER")) != 0;
    /* PRICING PROBE, TIMING ONLY (the text it produces is not evidence): bit 0 runs the DELTA
     * of row 0 alone, bit 1 the CONV of row 0 alone — the ceiling of a multi-row mode. */
    const uint32_t probe = getenv("PN_TT_PROBE_GDN_ROWS") ? (uint32_t)atoi(getenv("PN_TT_PROBE_GDN_ROWS")) : 0u;
    const int conv_rows = (probe & 2u) ? 1 : rows;
    int delta_rows = (probe & 1u) ? 1 : rows;
    double mark = pn_time_ms();

    if (!g->gdn_strip || ssm_index < 0 || rows < 1 || rows > 16)
        return -1;
    if (gdn_strip_rms_constants(g, l, cfg, ssm_index) != 0)
        return -1;
    if (pn_tt_render_widen(g->device, pn_tt_operand(g->device, qkv),
                           pn_tt_gdn_stream_in_operand(g->device), stream_tiles) != 0)
        return -1;
    /* PM 6 MULTI-ROW CONV: every row in ONE descriptor per member (not deferred: one await). */
    r = 0;
    if (conv_rows > 1 && conv_multi_row_enabled()) {
        const int multi = pn_tt_gdn_conv_rows_oncard(g->device, (uint32_t)ssm_index,
                                                     (uint32_t)conv_rows);

        if (multi == -1)
            return -1;
        if (multi == 0)
            r = conv_rows; /* done; a refusal (-2) leaves r at 0 and the per-row walk runs */
    }
    (void)pn_tt_gdn_defer(g->device, defer);
    for (; rc == 0 && r < conv_rows; r++)
        rc = (pn_tt_gdn_set_row(g->device, (uint32_t)r) != 0 ||
              pn_tt_gdn_state_out(g->device, r) != 0 ||
              pn_tt_gdn_conv_oncard(g->device, (uint32_t)ssm_index) != 0 ||
              pn_tt_gdn_layer_swap_part(g->device, (uint32_t)ssm_index, (uint32_t)r, true,
                                        false) != 0)
                 ? -1 : 0;
    (void)pn_tt_gdn_defer(g->device, false);
    if (pn_tt_gdn_await_deferred(g->device) != 0)
        rc = -1;
    (void)pn_tt_gdn_state_out(g->device, -1);
    (void)pn_tt_gdn_set_row(g->device, 0u);
    if (rc != 0)
        return -1;
    g->spec_gdn_part_ms[1] += pn_time_ms() - mark;
    mark = pn_time_ms();
    stream = pn_tt_gdn_stream_operand(g->device);
    if (pn_tt_head_norm_operands(g->device, FT_TT_OP_REDUCE_FN_L2_NORM, &g->gdn_l2,
                                 (uint32_t)(2 * cfg->ssm_n_group), stream, stream,
                                 stream_tiles, stream_tiles) != 0)
        return -1;
    g->spec_gdn_part_ms[2] += pn_time_ms() - mark;
    mark = pn_time_ms();
    /* S2d: every row's gates computed first and staged in ONE write (deferred mode), then the
     * DELTAs queue on the staged blocks. */
    /* PM 6: the alpha/beta reads (reader thread) land before the host math needs them.
     * ITEM 6: with det gates on a deferred verify there are no reads — the card writes every
     * row's gate block itself. */
    if (defer && g->gdn_det_gates && !alpha && !beta_raw) {
        (void)pn_tt_gdn_defer(g->device, true);
        rc = gdn_gate_card(g, ssm_index, (uint32_t)rows);
        if (rc == 0 && rows > 1 && delta_rows == rows && delta_multi_row_enabled()) {
            (void)pn_tt_gdn_defer(g->device, false);
            rc = pn_tt_gdn_delta_rows_oncard(g->device, (uint32_t)ssm_index, (uint32_t)rows);
            delta_rows = 0;
        }
        for (r = 0; rc == 0 && r < delta_rows; r++)
            rc = (pn_tt_gdn_set_row(g->device, (uint32_t)r) != 0 ||
                  pn_tt_gdn_state_out(g->device, r) != 0 ||
                  pn_tt_gdn_delta_descriptor_only(g->device, (uint32_t)ssm_index) != 0 ||
                  pn_tt_gdn_layer_swap_part(g->device, (uint32_t)ssm_index, (uint32_t)r, false,
                                            true) != 0)
                     ? -1 : 0;
    } else if (pn_qwen35_tt_ab_verify_wait(g) != 0) {
        return -1;
    } else if (defer) {
        float decay[16 * 64], beta[16 * 64];

        if (v_heads > 64)
            return -1;
        for (r = 0; r < rows; r++)
            gdn_gate_host(g, l, alpha + (size_t)r * alpha_stride,
                          beta_raw + (size_t)r * beta_stride, v_heads, decay + r * v_heads,
                          beta + r * v_heads);
        (void)pn_tt_gdn_defer(g->device, true);
        rc = pn_tt_gdn_gate_upload_rows(g->device, (uint32_t)rows, decay, beta);
        /* PM 5 MULTI-ROW DELTA (`PN_TT_SPEC_DELTA_ROWS=0` off): every row in ONE descriptor
         * per member — q/k/v and the state read once a head, the rows carried in L1. */
        if (rc == 0 && rows > 1 && delta_rows == rows && delta_multi_row_enabled()) {
            (void)pn_tt_gdn_defer(g->device, false);
            rc = pn_tt_gdn_delta_rows_oncard(g->device, (uint32_t)ssm_index, (uint32_t)rows);
            delta_rows = 0;
        }
        for (r = 0; rc == 0 && r < delta_rows; r++)
            rc = (pn_tt_gdn_set_row(g->device, (uint32_t)r) != 0 ||
                  pn_tt_gdn_state_out(g->device, r) != 0 ||
                  pn_tt_gdn_delta_descriptor_only(g->device, (uint32_t)ssm_index) != 0 ||
                  pn_tt_gdn_layer_swap_part(g->device, (uint32_t)ssm_index, (uint32_t)r, false,
                                            true) != 0)
                     ? -1 : 0;
    } else
    for (r = 0; rc == 0 && r < rows; r++) {
        gdn_gate_host(g, l, alpha + (size_t)r * alpha_stride, beta_raw + (size_t)r * beta_stride,
                      v_heads, g->gdn_decay, g->gdn_beta);
        rc = (pn_tt_gdn_set_row(g->device, (uint32_t)r) != 0 ||
              pn_tt_gdn_state_out(g->device, r) != 0 ||
              pn_tt_gdn_delta_oncard(g->device, (uint32_t)ssm_index, g->gdn_decay,
                                     g->gdn_beta) != 0 ||
              pn_tt_gdn_layer_swap_part(g->device, (uint32_t)ssm_index, (uint32_t)r, false,
                                        true) != 0)
                 ? -1 : 0;
    }
    (void)pn_tt_gdn_defer(g->device, false);
    if (pn_tt_gdn_await_deferred(g->device) != 0)
        rc = -1;
    (void)pn_tt_gdn_state_out(g->device, -1);
    (void)pn_tt_gdn_set_row(g->device, 0u);
    if (rc != 0)
        return -1;
    g->spec_gdn_part_ms[3] += pn_time_ms() - mark;
    g->spec_gdn_mark_ms = pn_time_ms();
    if (pn_tt_head_norm_operands(g->device, FT_TT_OP_REDUCE_FN_RMS_NORM, &g->gdn_rms[ssm_index],
                                 (uint32_t)v_heads, pn_tt_gdn_y_operand(g->device),
                                 pn_tt_fp32_strip_operand(g->device, &g->gdn_y_norm),
                                 value_tiles, g->gdn_y_norm.tile_count) != 0)
        return -1;
    if (pn_tt_gate_narrow_is_armed(g->device))
        return pn_tt_gate_narrow(g->device, pn_tt_operand(g->device, z),
                                 pn_tt_fp32_strip_operand(g->device, &g->gdn_y_norm),
                                 pn_tt_operand(g->device, gated), value_tiles);
    if (pn_tt_render_narrow(g->device, pn_tt_fp32_strip_operand(g->device, &g->gdn_y_norm),
                            pn_tt_operand(g->device, &g->gdn_y_bridge), value_tiles) != 0)
        return -1;
    return pn_tt_silu_mul(g->device, z, &g->gdn_y_bridge, gated);
}

/* ================================================================
 * THE CHUNK — the same six steps, with the two recurrences over 32 positions at once
 * ================================================================ */

/**
 * PHASE CONV per position across the chunk's rows, leaving the buffer a chunk descriptor's
 * shape. The `conv` half of the phase-isolation arm — see `PN_TT_PREFILL_GDNC_PHASE`.
 *
 * THE ZERO IN THE LOOP IS THE POINT, not defensive coding. A decode descriptor computes WHOLE
 * TILES over an input whose rows 1..31 are padding, so it writes `SiLU(0) = 0` into rows 1..31
 * of its output and it expects rows 1..31 of its input to be zero. Handing it a chunk's buffer
 * unmodified would both feed it 31 other positions and destroy the rows already answered.
 */
static int gdn_split_conv(pn_qwen35_tt *g, int ssm_index, int rows, uint32_t stream_tiles)
{
    const size_t width = (size_t)stream_tiles * 32u;
    int t;

    /* The chunk's whole input comes off the card FIRST, because the loop below wipes it. */
    for (t = 0; t < rows; t++)
        if (pn_tt_gdn_strip_row_move(g->device, PN_TT_GDN_STRIP_STREAM_IN, (uint32_t)t,
                                     stream_tiles, g->gdn_split_stream + (size_t)t * width,
                                     true) != 0)
            return -1;

    for (t = 0; t < rows; t++) {
        float *row = g->gdn_split_stream + (size_t)t * width;

        if (pn_tt_gdn_strip_zero(g->device, PN_TT_GDN_STRIP_STREAM_IN) != 0 ||
            pn_tt_gdn_strip_row_move(g->device, PN_TT_GDN_STRIP_STREAM_IN, 0u, stream_tiles,
                                     row, false) != 0 ||
            pn_tt_gdn_conv_oncard(g->device, (uint32_t)ssm_index) != 0 ||
            pn_tt_gdn_strip_row_move(g->device, PN_TT_GDN_STRIP_STREAM_OUT, 0u, stream_tiles,
                                     row, true) != 0)
            return -1;
    }

    /* Reassembled the way a chunk descriptor leaves it: rows 0..C-1 answered, the rest zero. */
    if (pn_tt_gdn_strip_zero(g->device, PN_TT_GDN_STRIP_STREAM_OUT) != 0)
        return -1;
    for (t = 0; t < rows; t++)
        if (pn_tt_gdn_strip_row_move(g->device, PN_TT_GDN_STRIP_STREAM_OUT, (uint32_t)t,
                                     stream_tiles, g->gdn_split_stream + (size_t)t * width,
                                     false) != 0)
            return -1;
    return 0;
}

/** PHASE DELTA per position across the chunk's rows. The `delta` half, and the same argument
 *  about the zero — `y` is whole-tile written too, so every position would wipe the last. */
static int gdn_split_delta(pn_qwen35_tt *g, const pn_qwen35moe_config *cfg, int ssm_index,
                           int rows, uint32_t stream_tiles, uint32_t value_tiles)
{
    const size_t width = (size_t)stream_tiles * 32u;
    const size_t y_width = (size_t)value_tiles * 32u;
    const int v_heads = cfg->ssm_dt_rank;
    int t;

    for (t = 0; t < rows; t++)
        if (pn_tt_gdn_strip_row_move(g->device, PN_TT_GDN_STRIP_STREAM_OUT, (uint32_t)t,
                                     stream_tiles, g->gdn_split_stream + (size_t)t * width,
                                     true) != 0)
            return -1;

    for (t = 0; t < rows; t++)
        if (pn_tt_gdn_strip_zero(g->device, PN_TT_GDN_STRIP_STREAM_OUT) != 0 ||
            pn_tt_gdn_strip_row_move(g->device, PN_TT_GDN_STRIP_STREAM_OUT, 0u, stream_tiles,
                                     g->gdn_split_stream + (size_t)t * width, false) != 0 ||
            pn_tt_gdn_delta_oncard(g->device, (uint32_t)ssm_index,
                                   g->gdn_chunk_decay + (size_t)t * v_heads,
                                   g->gdn_chunk_beta + (size_t)t * v_heads) != 0 ||
            pn_tt_gdn_strip_row_move(g->device, PN_TT_GDN_STRIP_Y, 0u, value_tiles,
                                     g->gdn_split_y + (size_t)t * y_width, true) != 0)
            return -1;

    if (pn_tt_gdn_strip_zero(g->device, PN_TT_GDN_STRIP_Y) != 0)
        return -1;
    for (t = 0; t < rows; t++)
        if (pn_tt_gdn_strip_row_move(g->device, PN_TT_GDN_STRIP_Y, (uint32_t)t, value_tiles,
                                     g->gdn_split_y + (size_t)t * y_width, false) != 0)
            return -1;
    return 0;
}

/**
 * The two mixers at a SHORTER chunk length than the pass, `sub` rows at a time.
 *
 * Each sub-chunk is put at rows 0..sub-1 with the rest of the buffer zero, because that is
 * where both chunk kernels read; the window and the recurrent state carry across sub-chunks
 * exactly as they carry across passes, since neither this file nor the kernels touch them in
 * between. `sub == rows` is one iteration and reproduces the whole-pass arm.
 */
static int gdn_chunk_subchunked(pn_qwen35_tt *g, const pn_qwen35moe_config *cfg, int ssm_index,
                                int rows, int sub, uint32_t stream_tiles, uint32_t value_tiles)
{
    const size_t width = (size_t)stream_tiles * 32u;
    const size_t y_width = (size_t)value_tiles * 32u;
    const int v_heads = cfg->ssm_dt_rank;
    int first;
    int t;

    for (t = 0; t < rows; t++)
        if (pn_tt_gdn_strip_row_move(g->device, PN_TT_GDN_STRIP_STREAM_IN, (uint32_t)t,
                                     stream_tiles, g->gdn_split_stream + (size_t)t * width,
                                     true) != 0)
            return -1;

    for (first = 0; first < rows; first += sub) {
        const int live = (rows - first < sub) ? rows - first : sub;

        if (pn_tt_gdn_strip_zero(g->device, PN_TT_GDN_STRIP_STREAM_IN) != 0)
            return -1;
        for (t = 0; t < live; t++)
            if (pn_tt_gdn_strip_row_move(g->device, PN_TT_GDN_STRIP_STREAM_IN, (uint32_t)t,
                                         stream_tiles,
                                         g->gdn_split_stream + (size_t)(first + t) * width,
                                         false) != 0)
                return -1;
        if (pn_tt_gdnc_conv_oncard(g->device, (uint32_t)ssm_index, (uint32_t)live) != 0)
            return -1;
        /* The same L2 descriptor as every other arm: rows at and above `live` are zero and a
         * zero row normalises to a zero row. */
        if (pn_tt_head_norm_operands(g->device, FT_TT_OP_REDUCE_FN_L2_NORM, &g->gdn_l2,
                                     (uint32_t)(2 * cfg->ssm_n_group),
                                     pn_tt_gdn_stream_operand(g->device),
                                     pn_tt_gdn_stream_operand(g->device), stream_tiles,
                                     stream_tiles) != 0)
            return -1;
        if (pn_tt_gdnc_delta_oncard(g->device, (uint32_t)ssm_index, (uint32_t)live,
                                    g->gdn_chunk_decay + (size_t)first * v_heads,
                                    g->gdn_chunk_beta + (size_t)first * v_heads) != 0)
            return -1;
        for (t = 0; t < live; t++)
            if (pn_tt_gdn_strip_row_move(g->device, PN_TT_GDN_STRIP_Y, (uint32_t)t, value_tiles,
                                         g->gdn_split_y + (size_t)(first + t) * y_width,
                                         true) != 0)
                return -1;
    }

    if (pn_tt_gdn_strip_zero(g->device, PN_TT_GDN_STRIP_Y) != 0)
        return -1;
    for (t = 0; t < rows; t++)
        if (pn_tt_gdn_strip_row_move(g->device, PN_TT_GDN_STRIP_Y, (uint32_t)t, value_tiles,
                                     g->gdn_split_y + (size_t)t * y_width, false) != 0)
            return -1;
    return 0;
}

/**
 * Steps 3-5 of the chunk: PHASE CONV, the q/k L2 norm, PHASE DELTA.
 *
 * The L2 norm between them is the SAME descriptor in every arm and that is deliberate — a row
 * reduce reduces ALONG the row, so row t of row block h is position t of head h and one
 * position and 32 are the same arithmetic per row. It is not a candidate for the divergence
 * and running it batched in all four arms keeps the comparison to the two mixers.
 */
static int gdn_chunk_mixers(pn_qwen35_tt *g, const pn_qwen35moe_config *cfg, int ssm_index,
                            int rows, uint32_t stream_tiles, uint32_t value_tiles)
{
    const pn_gdnc_phase phase = gdnc_phase();
    const int sub = gdnc_sub_rows();

    if ((phase != PN_GDNC_PHASE_BOTH || sub != 0) &&
        (!g->gdn_split_stream || !g->gdn_split_y))
        return -1;
    /* The sub-chunk answers a different question and the two do not compose: it needs BOTH
     * mixers chunkwise, at a shorter C. */
    if (sub != 0 && phase == PN_GDNC_PHASE_BOTH)
        return gdn_chunk_subchunked(g, cfg, ssm_index, rows, sub, stream_tiles, value_tiles);

    if (phase == PN_GDNC_PHASE_BOTH || phase == PN_GDNC_PHASE_CONV) {
        gdn_prestage_job first;
        const int delta_sub = delta_sub_rows();

        /* Sub-chunk 0's staging hides under CONV, which reads neither slot. */
        if (phase == PN_GDNC_PHASE_BOTH && gdnc_stage_overlap() && !g->gdn_card_gate) {
            first = gdn_prestage_job_at(g, ssm_index, cfg->ssm_dt_rank, rows,
                                        (delta_sub == 0 || delta_sub >= rows) ? rows : delta_sub,
                                        0);
            pn_tt_gdnc_set_overlap(g->device, gdn_prestage_run, &first);
        }
        if (pn_tt_gdnc_conv_oncard(g->device, (uint32_t)ssm_index, (uint32_t)rows) != 0) {
            pn_tt_gdnc_set_overlap(g->device, NULL, NULL);
            return -1;
        }
    } else if (gdn_split_conv(g, ssm_index, rows, stream_tiles) != 0) {
        return -1;
    }

    if (pn_tt_head_norm_operands(g->device, FT_TT_OP_REDUCE_FN_L2_NORM, &g->gdn_l2,
                                 (uint32_t)(2 * cfg->ssm_n_group),
                                 pn_tt_gdn_stream_operand(g->device),
                                 pn_tt_gdn_stream_operand(g->device), stream_tiles,
                                 stream_tiles) != 0)
        return -1;

    if (phase == PN_GDNC_PHASE_BOTH)
        return gdn_delta_in_place(g, cfg, ssm_index, rows);
    /* THE PHASE-ISOLATION ARMS KEEP THE WHOLE-PASS DESCRIPTOR. They exist to attribute the
     * variance between the two mixers at ONE chunk length; sub-chunking one of them inside an
     * arm whose other half is per-position would answer neither question. */
    if (phase == PN_GDNC_PHASE_DELTA)
        return pn_tt_gdnc_delta_oncard(g->device, (uint32_t)ssm_index, (uint32_t)rows,
                                       g->gdn_chunk_decay, g->gdn_chunk_beta);
    return gdn_split_delta(g, cfg, ssm_index, rows, stream_tiles, value_tiles);
}

bool pn_qwen35_tt_gdn_chunk_available(const pn_qwen35_tt *g, int rows)
{
    uint32_t floor;

    if (!g || !g->gdn_strip || !g->gdn_chunk_decay || !g->gdn_chunk_beta)
        return false;
    /* An isolation arm that quietly fell back to the chunk path would be a measurement of the
     * thing not under test — the one way this attribution can lie and read plausible. */
    if ((gdnc_phase() != PN_GDNC_PHASE_BOTH || gdnc_sub_rows() != 0) &&
        (!g->gdn_split_stream || !g->gdn_split_y))
        return false;
    floor = pn_tt_gdnc_min_chunk_rows(g->device);
    /* A ZERO FLOOR MEANS THE FAMILY IS NOT ARMED, not that any chunk will do — the floor is
     * `conv_taps - 1` and a family that armed has at least two taps. */
    if (floor == 0u)
        return false;
    return rows > 0 && rows <= 32 && (uint32_t)rows >= floor;
}

/* =======================================================================================
 * PM 7 ITEM 7 step 1 — THE PIPELINED BAND (streamed, card gates)
 *
 * Per band the chain was widen, CONV, L2, DELTA, gated RMS, narrow, gate — each awaited before
 * the next, so the puller stood still through all seven. Only CONV (window) and DELTA (state)
 * carry across bands; RMS / narrow / gate of band b depend on band b's DELTA alone. So band b's
 * tail runs under band b+1's head, its awaits moved to just before the op that rewrites what it
 * reads:
 *     band b+1:  widen | CONV | settle RMS(b), settle gate(b-1), narrow(b) | L2 | DELTA |
 *                settle narrow(b), gate(b) | RMS(b+1)
 * WAR hazards, each closed by the settle before it: DELTA(b+1) rewrites y (RMS(b) settled),
 * RMS(b+1) rewrites y_norm (narrow(b) settled), narrow(b) rewrites the bridge (gate(b-1)
 * settled). Every op is the one the serial chain ran, on the same bytes: bit-exact.
 * ======================================================================================= */

/** One `strcmp` against `PN_TT_RED`. */
static bool gdn_strip_env_red(const char *name)
{
    const char *value = getenv("PN_TT_RED");

    return value && *value && strcmp(value, name) == 0;
}

/** The pending band's narrow (deferred). Needs its RMS done and the gate before it done. */
static int gdn_pipe_narrow(pn_qwen35_tt *g, uint32_t value_tiles)
{
    if (pn_tt_head_norm_settle(g->device, g->gdn_rms[g->gdn_pipe_layer].tile_cols) != 0 ||
        pn_tt_eltwise_group_settle(g->device) != 0)
        return -1;
    pn_tt_render_defer(g->device, true, true);
    {
        const int rc = pn_tt_render_narrow(g->device,
                                           pn_tt_fp32_strip_operand(g->device, &g->gdn_y_norm),
                                           pn_tt_operand(g->device, &g->gdn_y_bridge),
                                           value_tiles);

        pn_tt_render_defer(g->device, true, false);
        return rc;
    }
}

/** The pending band's gate (deferred), once its narrow is done. */
static int gdn_pipe_gate(pn_qwen35_tt *g)
{
    if (pn_tt_render_settle(g->device, true) != 0)
        return -1;
    pn_tt_eltwise_group_defer_next(g->device);
    return pn_tt_silu_mul(g->device, &g->gdn_pipe_z, &g->gdn_y_bridge, &g->gdn_pipe_gated);
}

/** Band b's RMS (deferred), its tail then pending with z / gated kept for the next band. */
static int gdn_pipe_rms(pn_qwen35_tt *g, int ssm_index, int v_heads, uint32_t value_tiles,
                        const pn_tt_activation *z, const pn_tt_activation *gated)
{
    int rc;

    pn_tt_head_norm_defer(g->device, g->gdn_rms[ssm_index].tile_cols, true);
    rc = pn_tt_head_norm_operands(g->device, FT_TT_OP_REDUCE_FN_RMS_NORM,
                                  &g->gdn_rms[ssm_index], (uint32_t)v_heads,
                                  pn_tt_gdn_y_operand(g->device),
                                  pn_tt_fp32_strip_operand(g->device, &g->gdn_y_norm),
                                  value_tiles, g->gdn_y_norm.tile_count);
    pn_tt_head_norm_defer(g->device, g->gdn_rms[ssm_index].tile_cols, false);
    g->gdn_pipe_z = *z;
    g->gdn_pipe_gated = *gated;
    g->gdn_pipe_layer = ssm_index;
    g->gdn_pipe_stage = rc == 0 ? 1 : 0;
    return rc;
}

int pn_qwen35_tt_gdn_pipe_drain(pn_qwen35_tt *g)
{
    int rc = 0;

    if (!g || g->gdn_pipe_stage == 0)
        return 0;
    g->gdn_pipe_stage = 0;
    if (gdn_pipe_narrow(g, pn_tt_gdn_value_tiles(g->device)) != 0 || gdn_pipe_gate(g) != 0)
        rc = -1;
    /* Everything awaited: whatever reads `gated` next sees it written. */
    if (pn_tt_render_settle(g->device, true) != 0 || pn_tt_eltwise_group_settle(g->device) != 0 ||
        pn_tt_head_norm_settle(g->device, g->gdn_rms[g->gdn_pipe_layer].tile_cols) != 0)
        rc = -1;
    return rc;
}

/** PM 8 TIMING PROBE `PN_TT_PROBE_CHAIN=conv|delta|gdnc`: the piped band skips that gdnc
 *  phase's descriptors — the chain's floor without them. The text is garbage by design. */
static bool gdn_probe_skips(const char *phase)
{
    static const char *probe = (const char *)1;

    if (probe == (const char *)1)
        probe = getenv("PN_TT_PROBE_CHAIN");
    return probe && *probe &&
           (strcmp(probe, phase) == 0 || strcmp(probe, "gdnc") == 0);
}

/** The pipelined band: widen, CONV, [narrow(prev)], L2, DELTA, [gate(prev)], RMS(b). */
static int gdn_strip_chunk_piped(pn_qwen35_tt *g, const pn_qwen35moe_config *cfg, int ssm_index,
                                 const pn_tt_activation *qkv, const pn_tt_activation *z,
                                 int rows, const pn_tt_activation *gated)
{
    const uint32_t stream_tiles = pn_tt_gdn_stream_tiles(g->device);
    const uint32_t value_tiles = pn_tt_gdn_value_tiles(g->device);
    const bool pending = g->gdn_pipe_stage == 1;
    ft_tt_op_operand stream;

    /* A band of another layer must not inherit this one's tail. */
    if (pending && g->gdn_pipe_layer != ssm_index && pn_qwen35_tt_gdn_pipe_drain(g) != 0)
        return -1;
    if (pn_tt_render_widen(g->device, pn_tt_operand(g->device, qkv),
                           pn_tt_gdn_stream_in_operand(g->device), stream_tiles) != 0 ||
        (!gdn_probe_skips("conv") &&
         pn_tt_gdnc_conv_oncard(g->device, (uint32_t)ssm_index, (uint32_t)rows) != 0))
        return -1;
    if (g->gdn_pipe_stage == 1 && gdn_pipe_narrow(g, value_tiles) != 0)
        return -1;
    stream = pn_tt_gdn_stream_operand(g->device);
    if (pn_tt_head_norm_operands(g->device, FT_TT_OP_REDUCE_FN_L2_NORM, &g->gdn_l2,
                                 (uint32_t)(2 * cfg->ssm_n_group), stream, stream,
                                 stream_tiles, stream_tiles) != 0 ||
        (!gdn_probe_skips("delta") && gdn_delta_card_gate(g, ssm_index, rows) != 0))
        return -1;
    /* RED `gdn_pipe_gate_skip`: the pending band's gate is never issued (its `gated` rows stay
     * whatever they were) — must DIFFER: proves the pipelined gate is what writes them. */
    if (g->gdn_pipe_stage == 1 && !gdn_strip_env_red("gdn_pipe_gate_skip") &&
        gdn_pipe_gate(g) != 0)
        return -1;
    return gdn_pipe_rms(g, ssm_index, cfg->ssm_dt_rank, value_tiles, z, gated);
}

int pn_qwen35_tt_gdn_strip_chunk(pn_qwen35_tt *g, const pn_qwen35moe_ssm_layer *l,
                                 const pn_qwen35moe_config *cfg, int ssm_index,
                                 const pn_tt_activation *qkv, const pn_tt_activation *z,
                                 const float *beta_raw, uint32_t beta_stride,
                                 const float *alpha, uint32_t alpha_stride, int rows,
                                 const pn_tt_activation *gated)
{
    const int v_heads = cfg->ssm_dt_rank;
    const uint32_t stream_tiles = pn_tt_gdn_stream_tiles(g->device);
    const uint32_t value_tiles = pn_tt_gdn_value_tiles(g->device);
    int position;

    if (!g || !l || !cfg || !qkv || !z || !beta_raw || !alpha || !gated)
        return -1;
    if (ssm_index < 0 || !pn_qwen35_tt_gdn_chunk_available(g, rows))
        return -1;
    {
        const double at = pn_time_ms() / 1000.0;
        const int rc = gdn_strip_rms_constants(g, l, cfg, ssm_index);

        pn_tt_device_host_clock_add(g->device, PN_TT_HOST_CLOCK_GDN_CONSTANTS,
                                    pn_time_ms() / 1000.0 - at);
        if (rc != 0)
            return -1;
    }

    /* 1 and 2, the oracle's own, now for every position of the chunk. The result is ROW-MAJOR
     * BY POSITION, which is the layout `pn_tt_gdnc_delta_oncard()` transposes per head into a
     * decay-ratio tile and a gate run. `decay = exp(g)` is applied HERE because the card has
     * no `log` and therefore no `pow`. (PM 7 ITEM 5: or not at all — the card computed them.) */
    if (!g->gdn_card_gate) {
        const double at = pn_time_ms() / 1000.0;

        for (position = 0; position < rows; position++) {
            float *decay = g->gdn_chunk_decay + (size_t)position * v_heads;
            float *beta = g->gdn_chunk_beta + (size_t)position * v_heads;

            gdn_gate_host(g, l, alpha + (size_t)position * alpha_stride,
                          beta_raw + (size_t)position * beta_stride, v_heads, decay, beta);
        }
        pn_tt_device_host_clock_add(g->device, PN_TT_HOST_CLOCK_GDN_MATH,
                                    pn_time_ms() / 1000.0 - at);
    }

    if (g->gdn_pipe && g->gdn_card_gate)
        return gdn_strip_chunk_piped(g, cfg, ssm_index, qkv, z, rows, gated);

    /* 3. The bf16 projection becomes the fp32 stream, WHOLE TILES — the render is per tile, so
     * all 32 rows cross in the same descriptor one row used to. Rows at and above the chunk are
     * zero and the host owes that: `pn_tt_activation_upload_rows()` tilizes the pass's hidden
     * strip at `valid_rows = rows`, an RMS norm of a zero row is zero and an `Mt=1` matvec of a
     * zero row is zero, so every projection is already zero above the chunk. */
    if (pn_tt_render_widen(g->device, pn_tt_operand(g->device, qkv),
                           pn_tt_gdn_stream_in_operand(g->device), stream_tiles) != 0)
        return -1;

    /* 4 and 5. PHASE CONV, the q/k L2 norm IN PLACE, PHASE DELTA — the three steps
     * `PN_TT_PREFILL_GDNC_PHASE` can split between the chunkwise and the decode families. The
     * L2 norm is the same descriptor in every arm; the norm's own reason is in
     * `gdn_chunk_mixers()`. */
    if (gdn_chunk_mixers(g, cfg, ssm_index, rows, stream_tiles, value_tiles) != 0)
        return -1;

    /* 6. The gated norm and the output gate, both unchanged for the same reasons as 4. */
    if (pn_tt_head_norm_operands(g->device, FT_TT_OP_REDUCE_FN_RMS_NORM, &g->gdn_rms[ssm_index],
                                 (uint32_t)v_heads, pn_tt_gdn_y_operand(g->device),
                                 pn_tt_fp32_strip_operand(g->device, &g->gdn_y_norm),
                                 value_tiles, g->gdn_y_norm.tile_count) != 0)
        return -1;
    /* PM 7: the previous chunk's gate (deferred) still reads `gdn_y_bridge`: settled before
     * the narrow rewrites it. Streamed, this is where its puller AWAIT now lands — the widen,
     * CONV, L2, DELTA and gated norm above ran under it. */
    if (pn_tt_eltwise_group_settle(g->device) != 0)
        return -1;
    if (pn_tt_render_narrow(g->device, pn_tt_fp32_strip_operand(g->device, &g->gdn_y_norm),
                            pn_tt_operand(g->device, &g->gdn_y_bridge), value_tiles) != 0)
        return -1;
    if (g->gdn_chunk_defer_gate)
        pn_tt_eltwise_group_defer_next(g->device);
    return pn_tt_silu_mul(g->device, z, &g->gdn_y_bridge, gated);
}

#endif /* USE_TT */
