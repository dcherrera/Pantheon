/* SPDX-License-Identifier: AGPL-3.0-or-later */
/* Copyright (C) 2026 David (Corey) Herrera */
/**
 * @file qwen35moe_prefill_prefetch_tt.c
 * @brief The batched prefill's alpha/beta READS for the next 32-row band, hidden under the
 *        current band's GDN chunk descriptors.
 *
 * Every SSM layer of a pass reads `ssm_alpha` and `ssm_beta` down a band at a time (the host
 * owes `softplus`/`exp`/`sigmoid`, which the SFPU cannot do) — 1,224 band-layers and ~1.05 s of
 * a 1,630-token prefill at the link's ~5.3 MB/s readback, every one of them with the card idle.
 * All of a layer's bands are projected before its band loop starts, so band b+1's rows are
 * already on the card while band b's mixers run. This reads them then: alpha inside the CONV
 * descriptor's enqueue/await window (the gdnc AUX overlap slot — the first slot is the DELTA
 * staging's), beta inside the DELTA's (the alpha hook arms the beta one). Same bytes, same host buffers, earlier: bit-identical by
 * construction. A read that did not happen (no window opened, a failure) is just done by the
 * band itself, as before.
 *
 * `PN_TT_PREFILL_AB_PREFETCH=0` off. RED `prefill_ab_prefetch_stale`: the prefetch reads the
 * CURRENT band's rows into the next band's place (must DIFFER).
 *
 * @status Written; gated by run_prefill_ab_prefetch.sh.
 * @issues The first band of each layer is still read inline (its rows exist only after that
 *   layer's projections).
 * @todo None
 */

#ifdef USE_TT

#include "qwen35moe_tt_internal.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "../accel/tt/pn_tt_gdnc.h"
#include "../accel/tt/pn_tt_reader.h"
#include "../accel/tt/pn_tt_dispatch.h"
#include "../accel/tt/pn_tt_matmul_group.h"
#include "tt/ft_tt_op_l1_layout.h"
#include "../core/bench.h"

/** One band's pending reads. Static: one prefill runs at a time on a model's card. */
static struct {
    pn_qwen35_tt *g;
    int ssm_index, band, rows;
    uint32_t alpha_width, beta_width;
    float *alpha, *beta;      /**< host rows of that band */
    bool alpha_done, beta_done;
    bool armed;
    double seconds;           /**< reads done inside a window (cumulative, reported) */
    unsigned long long reads;
} g_ab;

static bool ab_red(const char *name)
{
    const char *value = getenv("PN_TT_RED");

    return value && *value && strcmp(value, name) == 0;
}

bool pn_qwen35_tt_ab_prefetch_enabled(void)
{
    const char *env = getenv("PN_TT_PREFILL_AB_PREFETCH");

    return !(env && *env && atoi(env) == 0);
}

/** Read `rows` of band `band` of `strip` (only that strip is seeked; the band loop re-seeks
 *  every strip at its next band). */
static int ab_read(pn_tt_activation *strip, int band, float *host, uint32_t width, int rows)
{
    const double at = pn_time_ms();
    const int read_band = ab_red("prefill_ab_prefetch_stale") ? band - 1 : band;
    int rc;

    if (pn_tt_activation_seek(strip, (uint32_t)read_band) != 0)
        return -1;
    rc = pn_tt_activation_download_rows(g_ab.g->device, strip, host, width, (uint32_t)rows);
    g_ab.seconds += (pn_time_ms() - at) / 1000.0;
    g_ab.reads++;
    return rc;
}

static int ab_hook_beta(void *context)
{
    (void)context;
    if (!g_ab.armed || g_ab.beta_done)
        return 0;
    g_ab.beta_done = ab_read(&g_ab.g->card.ssm_beta, g_ab.band, g_ab.beta, g_ab.beta_width,
                             g_ab.rows) == 0;
    return g_ab.beta_done ? 0 : -1;
}

static int ab_hook_alpha(void *context)
{
    (void)context;
    if (!g_ab.armed || g_ab.alpha_done)
        return 0;
    g_ab.alpha_done = ab_read(&g_ab.g->card.ssm_alpha, g_ab.band, g_ab.alpha, g_ab.alpha_width,
                              g_ab.rows) == 0;
    /* The next gdnc window (the DELTA) takes the beta read. */
    pn_tt_gdnc_set_overlap_aux(g_ab.g->device, ab_hook_beta, NULL);
    return g_ab.alpha_done ? 0 : -1;
}

void pn_qwen35_tt_ab_prefetch_arm(pn_qwen35_tt *g, int ssm_index, int band, int rows,
                                  float *alpha, uint32_t alpha_width, float *beta,
                                  uint32_t beta_width)
{
    if (!g || !pn_qwen35_tt_ab_prefetch_enabled() || rows <= 0)
        return;
    g_ab.g = g;
    g_ab.ssm_index = ssm_index;
    g_ab.band = band;
    g_ab.rows = rows;
    g_ab.alpha = alpha;
    g_ab.beta = beta;
    g_ab.alpha_width = alpha_width;
    g_ab.beta_width = beta_width;
    g_ab.alpha_done = false;
    g_ab.beta_done = false;
    g_ab.armed = true;
    pn_tt_gdnc_set_overlap_aux(g->device, ab_hook_alpha, NULL);
}

void pn_qwen35_tt_ab_prefetch_settle(pn_qwen35_tt *g)
{
    if (!g || !g_ab.armed)
        return;
    /* A window that never opened leaves its hook set: it must not fire on some later
     * descriptor (the next layer has overwritten these strips by then). */
    pn_tt_gdnc_set_overlap_aux(g->device, NULL, NULL);
}

bool pn_qwen35_tt_ab_prefetched(const pn_qwen35_tt *g, int ssm_index, int band, bool beta)
{
    if (!g || !g_ab.armed || g_ab.g != g || g_ab.ssm_index != ssm_index || g_ab.band != band)
        return false;
    return beta ? g_ab.beta_done : g_ab.alpha_done;
}

void pn_qwen35_tt_ab_prefetch_reset(void)
{
    g_ab.armed = false;
}

/* ---- PHASE W: the reader-thread mode ---------------------------------------------------- */

/** A tall pass is at most 64 bands (QWEN35_TT_TALL_MAX_ROWS). */
#define AB_READER_MAX_BANDS 64

/** One layer's tickets: [0] alpha, [1] beta, per band; 0 = not posted. */
typedef struct ab_reader_slot {
    uint64_t ticket[2][AB_READER_MAX_BANDS];
    int ssm_index;
    int bands;            /**< bands the layer has */
    int posted;           /**< bands posted so far */
    bool armed;
} ab_reader_slot;

/** `cur` is the layer whose band loop is running; `nxt` the next SSM layer, whose bands the
 *  W2 wavefront posts super-band by super-band while `cur` still mixes. */
static struct {
    ab_reader_slot cur, nxt;
    double wait_seconds;
} g_abr;

/** The DRAM address of band `band` of `strip`, whatever band it is seeked to — computed, so the
 *  strip struct the issuing thread keeps moving is never touched by the post. */
static uint32_t ab_band_address(const pn_tt_activation *strip, int band)
{
    const uint32_t band_bytes = strip->tile_count * FT_TT_OP_TILE_BYTES_BF16;

    return strip->address - strip->row_offset * band_bytes + (uint32_t)band * band_bytes;
}

/** Post bands [first, first + count) of the alpha/beta strips into `slot`, gated or not. */
static bool ab_post_bands(pn_qwen35_tt *g, ab_reader_slot *slot, int first, int count, int rows,
                          float *alpha, uint32_t alpha_width, float *beta, uint32_t beta_width,
                          const pn_tt_reader_gate *gate)
{
    const pn_tt_activation *a = &g->card.ssm_alpha, *b = &g->card.ssm_beta;
    /* RED `prefill_reader_stale`: band b is read from band b-1's rows — must DIFFER. */
    const bool stale = ab_red("prefill_reader_stale");
    int band;

    for (band = first; band < first + count; band++) {
        const int done = band * 32;
        const int take = rows - done < 32 ? rows - done : 32;
        const int src = stale && band > 0 ? band - 1 : band;

        if (take <= 0 || band >= AB_READER_MAX_BANDS)
            return false;
        slot->ticket[0][band] = pn_tt_reader_post_rows_gated(
            g->device, a->bank, ab_band_address(a, src), a->tile_count, (uint32_t)take,
            alpha + (size_t)done * alpha_width, alpha_width, gate);
        slot->ticket[1][band] = pn_tt_reader_post_rows_gated(
            g->device, b->bank, ab_band_address(b, src), b->tile_count, (uint32_t)take,
            beta + (size_t)done * beta_width, beta_width, gate);
        slot->posted = band + 1;
    }
    return true;
}

bool pn_qwen35_tt_ab_reader_post(pn_qwen35_tt *g, int rows, float *alpha, uint32_t alpha_width,
                                 float *beta, uint32_t beta_width)
{
    const pn_tt_activation *a = &g->card.ssm_alpha, *b = &g->card.ssm_beta;
    const int bands = (rows + 31) / 32;

    g_abr.cur.armed = false;
    if (!g || !pn_tt_reader_available(g->device) || rows <= 0 || bands > AB_READER_MAX_BANDS ||
        a->replica_count > 1u || b->replica_count > 1u)
        return false;
    g_abr.cur.bands = bands;
    g_abr.cur.posted = 0;
    if (!ab_post_bands(g, &g_abr.cur, 0, bands, rows, alpha, alpha_width, beta, beta_width,
                       NULL))
        return false;
    g_abr.cur.armed = true;
    return true;
}

bool pn_qwen35_tt_ab_reader_post_next(pn_qwen35_tt *g, int ssm_index, int first, int count,
                                      int rows, float *alpha, uint32_t alpha_width, float *beta,
                                      uint32_t beta_width, const pn_tt_reader_gate *gate)
{
    const int bands = (rows + 31) / 32;

    if (!g || !pn_tt_reader_available(g->device) || bands > AB_READER_MAX_BANDS)
        return false;
    if (first == 0) {
        memset(&g_abr.nxt, 0, sizeof(g_abr.nxt));
        g_abr.nxt.ssm_index = ssm_index;
        g_abr.nxt.bands = bands;
    }
    if (g_abr.nxt.ssm_index != ssm_index || first != g_abr.nxt.posted ||
        !ab_post_bands(g, &g_abr.nxt, first, count, rows, alpha, alpha_width, beta, beta_width,
                       gate)) {
        g_abr.nxt.armed = false;
        return false;
    }
    g_abr.nxt.armed = g_abr.nxt.posted == g_abr.nxt.bands;
    return true;
}

bool pn_qwen35_tt_ab_reader_promote(int ssm_index)
{
    if (!g_abr.nxt.armed || g_abr.nxt.ssm_index != ssm_index)
        return false;
    g_abr.cur = g_abr.nxt;
    memset(&g_abr.nxt, 0, sizeof(g_abr.nxt));
    return true;
}

int pn_qwen35_tt_ab_reader_take(pn_qwen35_tt *g, int band)
{
    const double at = pn_time_ms();
    int rc;

    if (!g_abr.cur.armed || band < 0 || band >= g_abr.cur.bands ||
        g_abr.cur.ticket[0][band] == 0u || g_abr.cur.ticket[1][band] == 0u)
        return -1;
    rc = (pn_tt_reader_wait(g->device, g_abr.cur.ticket[0][band]) == 0 &&
          pn_tt_reader_wait(g->device, g_abr.cur.ticket[1][band]) == 0) ? 0 : -1;
    g_abr.wait_seconds += (pn_time_ms() - at) / 1000.0;
    return rc;
}

void pn_qwen35_tt_ab_reader_end(pn_qwen35_tt *g)
{
    int band;

    /* Only THIS layer's reads: the next layer's are the wavefront's and must keep running. */
    if (g_abr.cur.armed && g)
        for (band = g_abr.cur.bands - 1; band >= 0; band--)
            if (g_abr.cur.ticket[1][band]) {
                (void)pn_tt_reader_wait(g->device, g_abr.cur.ticket[1][band]);
                break;
            }
    g_abr.cur.armed = false;
}

void pn_qwen35_tt_ab_reader_abandon(pn_qwen35_tt *g)
{
    pn_qwen35_tt_ab_reader_end(g);
    if (g)
        pn_tt_reader_drain_cancelled(g->device);
    memset(&g_abr.nxt, 0, sizeof(g_abr.nxt));
}

/* ---- PM 6: the VERIFY's alpha/beta on the reader (no stream boundary) ------------------- */

/** One verify layer's two reads, posted after its projections and taken before the host math. */
static struct {
    uint64_t ticket[2];
    pn_qwen35_tt *g;
    float *alpha, *beta;
    uint32_t alpha_width, beta_width;
    int rows;
    bool armed;
    double wait_seconds;
    unsigned long long posts;
} g_abv;

static bool ab_verify_enabled(void)
{
    static int cached = -1;

    if (cached < 0) {
        const char *env = getenv("PN_TT_SPEC_AB_READER");

        cached = env && *env ? atoi(env) != 0 : 1;
    }
    return cached != 0;
}

bool pn_qwen35_tt_ab_verify_post(pn_qwen35_tt *g, int rows, float *alpha, uint32_t alpha_width,
                                 float *beta, uint32_t beta_width)
{
    const pn_tt_activation *a = &g->card.ssm_alpha, *b = &g->card.ssm_beta;
    pn_tt_reader_gate gate;
    /* RED `spec_ab_ungated`: the reads go out with no gate — before the card has run the
     * projections they read; must DIFFER. */
    const bool ungated = ab_red("spec_ab_ungated");

    g_abv.armed = false;
    if (!g || !ab_verify_enabled() || rows <= 0 || rows > 32 || a->replica_count > 1u ||
        b->replica_count > 1u || !pn_tt_reader_available(g->device) ||
        !pn_tt_matmul_group_member0_completion(g->device, &gate.x, &gate.y,
                                               &gate.control_address, &gate.target))
        return false;
    g_abv.ticket[0] = pn_tt_reader_post_rows_gated(g->device, a->bank, ab_band_address(a, 0),
                                                   a->tile_count, (uint32_t)rows, alpha,
                                                   alpha_width, ungated ? NULL : &gate);
    g_abv.ticket[1] = pn_tt_reader_post_rows_gated(g->device, b->bank, ab_band_address(b, 0),
                                                   b->tile_count, (uint32_t)rows, beta,
                                                   beta_width, ungated ? NULL : &gate);
    if (g_abv.ticket[0] == 0u || g_abv.ticket[1] == 0u) {
        pn_tt_reader_drain_cancelled(g->device);
        return false;
    }
    g_abv.g = g;
    g_abv.alpha = alpha;
    g_abv.beta = beta;
    g_abv.alpha_width = alpha_width;
    g_abv.beta_width = beta_width;
    g_abv.rows = rows;
    g_abv.armed = true;
    g_abv.posts++;
    /* The projections the gate waits on are still in the host's capture: launch them. */
    (void)pn_tt_dispatch_stream_kick();
    return true;
}

int pn_qwen35_tt_ab_verify_wait(pn_qwen35_tt *g)
{
    const double at = pn_time_ms();
    int rc = 0;

    if (!g_abv.armed || g_abv.g != g)
        return 0;
    g_abv.armed = false;
    /* What the host captured since (widen, CONV, L2) runs while the reads land. */
    (void)pn_tt_dispatch_stream_kick();
    if (pn_tt_reader_wait(g->device, g_abv.ticket[0]) != 0 ||
        pn_tt_reader_wait(g->device, g_abv.ticket[1]) != 0) {
        /* A failed read is re-read through the stream (a boundary: slower, the same bytes). */
        rc = (pn_tt_activation_download_rows(g->device, &g->card.ssm_alpha, g_abv.alpha,
                                             g_abv.alpha_width, (uint32_t)g_abv.rows) == 0 &&
              pn_tt_activation_download_rows(g->device, &g->card.ssm_beta, g_abv.beta,
                                             g_abv.beta_width, (uint32_t)g_abv.rows) == 0)
                 ? 0 : -1;
    }
    g_abv.wait_seconds += (pn_time_ms() - at) / 1000.0;
    return rc;
}

void pn_qwen35_tt_ab_prefetch_report(void)
{
    if (g_abv.posts)
        fprintf(stderr, "qwen35_tt: verify alpha/beta READER — %llu layers, %.3f s waited "
                        "(cumulative)\n", g_abv.posts, g_abv.wait_seconds);
    if (g_abr.wait_seconds > 0.0)
        fprintf(stderr, "qwen35_tt: prefill alpha/beta READER — %.3f s waited on the reader "
                        "thread (cumulative)\n", g_abr.wait_seconds);
    if (g_ab.reads == 0u)
        return;
    fprintf(stderr, "qwen35_tt: prefill alpha/beta PREFETCH — %llu reads under gdnc windows "
                    "(%.3f s, cumulative)\n", g_ab.reads, g_ab.seconds);
}

#endif /* USE_TT */
