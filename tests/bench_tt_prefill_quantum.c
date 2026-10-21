/* SPDX-License-Identifier: AGPL-3.0-or-later */
/* Copyright (C) 2026 David (Corey) Herrera */
/**
 * @file bench_tt_prefill_quantum.c
 * @brief THE THREE MEASUREMENTS PHASE 4 IS GATED ON, on Ornith's own strips.
 *
 * ---------------------------------------------------------------------------
 * WHY THIS EXISTS BEFORE ANY PREFILL CODE
 * ---------------------------------------------------------------------------
 * A batched prefill for this model is not the llama one. Ornith is 24 Gated-Delta-Net layers
 * and 8 attention layers, and BOTH mixers are single-position on the card by construction:
 * the GDN op advances a carried state, and `pn_tt_rope` / `pn_tt_kv_append` / flash decode
 * each take ONE position. So a 32-row pass can batch the norms, the projections, the SwiGLU
 * and the residual adds, and it must then hand each of the 32 rows to a mixer that reads
 * LOGICAL ROW 0. There is no on-card row gather, so the only way to reach row `r` is to bring
 * the strip down and put row `r` back up as row 0.
 *
 * That makes the whole design turn on ONE number nobody here has measured: **what a
 * WHOLE-TILE download costs against a row-0 download.** The row-0 reader moves 128 B a tile
 * (two 64 B chunks); the batched reader moves 2048 B. If the link is transaction-bound the
 * batched read is nearly free per token and a host-mediated row select is affordable; if it
 * is byte-bound at the row-0 rate the design is dead before it is written, because Phase A
 * spent this whole program removing exactly this traffic.
 *
 * Three things are timed, all on the strips the shipped forward already owns:
 *
 *   1. **TRANSFERS** — `pn_tt_activation_{upload,download}` (row 0) against
 *      `..._rows` (32 rows), at Ornith's four real widths. Reported as µs and as B/s in both
 *      directions, because this project has one measured figure for the card->host direction
 *      (~5 MB/s at 128 B a tile) and none at all for whole tiles.
 *
 *   2. **THE TILE QUANTUM, MEASURED RATHER THAN INHERITED** — the same projection with rows
 *      1..31 ZERO and with rows 1..31 carrying real activations. The descriptor is identical
 *      either way, so the claim "32 positions cost what one position costs" is a claim about
 *      DATA DEPENDENCE, not about descriptors — and this card has already produced one
 *      3.7x data-dependent timing (PHASE CONV, 1222.7 µs on real activations against 4571 on
 *      a degenerate stream). A tile quantum that holds for a `[K+1, N]` verify strip of
 *      near-duplicate rows need not hold for 32 unrelated prompt positions.
 *
 *   3. **THE SEQUENTIAL TERMS** — one GDN block (the strip bridge, six descriptors) and one
 *      on-card attention block, timed alone. These are the terms a 32-row pass CANNOT batch,
 *      so they set the ceiling of the whole item: 32 rows cost `flat + 32 * sequential`.
 *
 * ---------------------------------------------------------------------------
 * PROVING IT CAN FAIL
 * ---------------------------------------------------------------------------
 * `--selftest` runs the transfer arms against a KNOWN answer: it uploads 32 distinct rows,
 * reads them back with the batched reader and compares, and reads row 0 with the row-0 reader
 * and compares that. A bench whose transfers do not move the right bytes is measuring the
 * cost of being wrong. It also checks that `pn_tt_activation_upload_rows` re-zeroes rows
 * `rows..31`, which is the strip invariant every Mt=1 op downstream depends on.
 *
 * @status Working
 * @issues None
 * @todo None
 */

#include "../include/pantheon.h"
#include "../src/core/engine.h"

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

#ifndef USE_TT
int main(void)
{
    fprintf(stderr, "bench_tt_prefill_quantum: built without USE_TT — nothing to measure.\n");
    return 77;
}
#else

#include "../src/models/qwen35moe.h"
#include "../src/models/qwen35moe_tt_internal.h"

#define BENCH_ROWS 32u

static double bench_now(void)
{
    struct timespec ts;

    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (double)ts.tv_sec + (double)ts.tv_nsec * 1e-9;
}

/** A deterministic stream. Distinct per (row, column) so a transfer that moves row 0
 *  everywhere, or transposes, cannot pass the self-test. */
static void bench_fill(float *values, uint32_t rows, uint32_t width, uint32_t salt)
{
    for (uint32_t r = 0; r < rows; r++)
        for (uint32_t c = 0; c < width; c++)
            values[(size_t)r * width + c] =
                (float)((int)((r * 7919u + c * 104729u + salt) % 2003u) - 1000) * 0.0009765625f;
}

/** Time `reps` calls of a transfer and print µs and B/s. `bytes` is the payload the link
 *  actually moved, which for the row-0 reader is 128 B a tile and NOT `width * 4`. */
static void bench_report(const char *what, uint32_t width, uint32_t rows, double seconds,
                         int reps, double bytes_per_call)
{
    double per = seconds / (double)reps;

    printf("QUANTUM %-28s width %5u rows %2u  %9.1f us/call  %8.2f MB/s  (%.0f B/call)\n",
           what, width, rows, per * 1e6, bytes_per_call / per / 1e6, bytes_per_call);
}

/** The card-side bytes a strip transfer moves: whole tiles for the batched pair, two 64-byte
 *  chunks a tile for the row-0 pair. bf16 strips are 2048 B a tile. */
static double bench_bytes(uint32_t tiles, uint32_t rows, uint32_t tile_bytes)
{
    return rows > 1u ? (double)tiles * (double)tile_bytes : (double)tiles * 128.0;
}

static int bench_transfers(pn_qwen35_tt *g, const pn_qwen35moe_config *cfg, int reps)
{
    struct { const char *name; const pn_tt_activation *strip; uint32_t width; } arms[] = {
        { "hidden (n_embd)",   &g->card.hidden[0], (uint32_t)cfg->n_embd },
        { "ssm_qkv (conv_dim)", &g->card.ssm_qkv,  g->card.ssm_qkv.tile_count * 32u },
        { "ssm_gated",         &g->card.ssm_gated, g->card.ssm_gated.tile_count * 32u },
        { "ffn_swiglu (n_ff)", &g->card.ffn_swiglu, g->card.ffn_swiglu.tile_count * 32u },
    };
    size_t widest = 0;
    float *host;

    for (size_t i = 0; i < sizeof(arms) / sizeof(arms[0]); i++)
        if (arms[i].width > widest)
            widest = arms[i].width;
    host = calloc((size_t)BENCH_ROWS * widest, sizeof(float));
    if (!host)
        return -1;

    for (size_t i = 0; i < sizeof(arms) / sizeof(arms[0]); i++) {
        const pn_tt_activation *strip = arms[i].strip;
        uint32_t width = arms[i].width;
        uint32_t tiles = strip->tile_count;
        double mark;

        if (tiles == 0u)
            continue;
        bench_fill(host, BENCH_ROWS, width, (uint32_t)i);

        mark = bench_now();
        for (int r = 0; r < reps; r++)
            if (pn_tt_activation_upload(g->device, strip, host, width) != 0)
                goto fail;
        bench_report("upload row0", width, 1u, bench_now() - mark, reps,
                     bench_bytes(tiles, 1u, 2048u));

        mark = bench_now();
        for (int r = 0; r < reps; r++)
            if (pn_tt_activation_upload_rows(g->device, strip, host, width, BENCH_ROWS) != 0)
                goto fail;
        bench_report("upload 32 rows", width, BENCH_ROWS, bench_now() - mark, reps,
                     bench_bytes(tiles, BENCH_ROWS, 2048u));

        mark = bench_now();
        for (int r = 0; r < reps; r++)
            if (pn_tt_activation_download(g->device, strip, host, width) != 0)
                goto fail;
        bench_report("download row0", width, 1u, bench_now() - mark, reps,
                     bench_bytes(tiles, 1u, 2048u));

        mark = bench_now();
        for (int r = 0; r < reps; r++)
            if (pn_tt_activation_download_rows(g->device, strip, host, width, BENCH_ROWS) != 0)
                goto fail;
        bench_report("download 32 rows", width, BENCH_ROWS, bench_now() - mark, reps,
                     bench_bytes(tiles, BENCH_ROWS, 2048u));
    }
    free(host);
    return 0;
fail:
    free(host);
    fprintf(stderr, "QUANTUM REFUSED: a transfer failed\n");
    return -1;
}

/** THE SELF-TEST. Without it the timings above are the cost of moving unknown bytes. */
static int bench_selftest(pn_qwen35_tt *g, const pn_qwen35moe_config *cfg)
{
    const pn_tt_activation *strip = &g->card.hidden[0];
    uint32_t width = (uint32_t)cfg->n_embd;
    float *sent = calloc((size_t)BENCH_ROWS * width, sizeof(float));
    float *got = calloc((size_t)BENCH_ROWS * width, sizeof(float));
    int bad_rows = 0, bad_row0 = 0, bad_zero = 0, rc = -1;

    if (!sent || !got)
        goto done;
    bench_fill(sent, BENCH_ROWS, width, 0x5eedu);
    if (pn_tt_activation_upload_rows(g->device, strip, sent, width, BENCH_ROWS) != 0)
        goto done;
    if (pn_tt_activation_download_rows(g->device, strip, got, width, BENCH_ROWS) != 0)
        goto done;
    /* bf16 on the card, fp32 here: the round trip is a TRUNCATION, so agreement is to the
     * bf16 grid and not to the bit. A row that came back as another row would be off by far
     * more than that, which is what this is looking for. */
    for (uint32_t r = 0; r < BENCH_ROWS; r++)
        for (uint32_t c = 0; c < width; c++) {
            double a = sent[(size_t)r * width + c], b = got[(size_t)r * width + c];
            if (fabs(a - b) > 0.02 * (fabs(a) + 1e-3))
                bad_rows++;
        }
    memset(got, 0, (size_t)BENCH_ROWS * width * sizeof(float));
    if (pn_tt_activation_download(g->device, strip, got, width) != 0)
        goto done;
    for (uint32_t c = 0; c < width; c++)
        if (fabs(sent[c] - got[c]) > 0.02 * (fabs(sent[c]) + 1e-3))
            bad_row0++;

    /* THE STRIP INVARIANT: an upload of 4 rows must leave rows 4..31 ZERO, or a later Mt=1
     * op reads a previous pass's positions as padding. */
    if (pn_tt_activation_upload_rows(g->device, strip, sent, width, 4u) != 0)
        goto done;
    memset(got, 0, (size_t)BENCH_ROWS * width * sizeof(float));
    if (pn_tt_activation_download_rows(g->device, strip, got, width, BENCH_ROWS) != 0)
        goto done;
    for (uint32_t r = 4; r < BENCH_ROWS; r++)
        for (uint32_t c = 0; c < width; c++)
            if (got[(size_t)r * width + c] != 0.0f)
                bad_zero++;

    printf("QUANTUM selftest  32-row round trip %s (%d bad)  row-0 reader %s (%d bad)  "
           "rows 4..31 zeroed after a 4-row upload %s (%d bad)\n",
           bad_rows == 0 ? "OK" : "BAD", bad_rows, bad_row0 == 0 ? "OK" : "BAD", bad_row0,
           bad_zero == 0 ? "OK" : "BAD", bad_zero);
    rc = (bad_rows || bad_row0 || bad_zero) ? -1 : 0;
done:
    free(sent);
    free(got);
    return rc;
}

/**
 * THE TILE QUANTUM. The same four SSM projections, run with rows 1..31 zero and then with
 * rows 1..31 carrying real activations. Identical descriptors; the only variable is the DATA
 * the FPU multiplies in the 31 rows it was already multiplying.
 */
static int bench_quantum(pn_qwen35_tt *g, const pn_qwen35moe_config *cfg, int reps)
{
    const pn_qwen35_tt_layer *d = NULL;
    pn_qwen35_tt_strips *s = &g->card;
    uint32_t width = (uint32_t)cfg->n_embd;
    float *host = calloc((size_t)BENCH_ROWS * width, sizeof(float));
    double padded, live;
    int rc = -1;

    for (int i = 0; i < g->n_layers; i++)
        if (!g->layers[i].is_attention) { d = &g->layers[i]; break; }
    if (!host || !d)
        goto done;

    /* WARM UP FIRST, AND IT IS NOT A COURTESY. The first flush after arming retires and
     * replays the matmul residency ("a chained projection did not complete — retiring the
     * residency ... and replaying 4 projection(s)"), which lands entirely on whichever arm
     * runs first and reads as a 277x tile-quantum effect that is not one. Two discarded
     * rounds put both arms on a warm residency. */
    bench_fill(host, 1u, width, 11u);
    if (pn_tt_activation_upload(g->device, &s->normed, host, width) != 0)
        goto done;
    for (int warm = 0; warm < 2; warm++) {
        (void)pn_qwen35_tt_project(g, &d->attn_qkv, &s->normed, &s->ssm_qkv, false);
        (void)pn_qwen35_tt_project(g, &d->attn_gate, &s->normed, &s->ssm_z, false);
        (void)pn_qwen35_tt_project(g, &d->ssm_alpha, &s->normed, &s->ssm_alpha, false);
        (void)pn_qwen35_tt_project(g, &d->ssm_beta, &s->normed, &s->ssm_beta, false);
        (void)pn_tt_matvec_flush(g->device);
    }
    padded = bench_now();
    for (int r = 0; r < reps; r++) {
        if (pn_qwen35_tt_project(g, &d->attn_qkv, &s->normed, &s->ssm_qkv, false) != 0 ||
            pn_qwen35_tt_project(g, &d->attn_gate, &s->normed, &s->ssm_z, false) != 0 ||
            pn_qwen35_tt_project(g, &d->ssm_alpha, &s->normed, &s->ssm_alpha, false) != 0 ||
            pn_qwen35_tt_project(g, &d->ssm_beta, &s->normed, &s->ssm_beta, false) != 0 ||
            pn_tt_matvec_flush(g->device) != 0)
            goto done;
    }
    padded = bench_now() - padded;

    bench_fill(host, BENCH_ROWS, width, 11u);
    if (pn_tt_activation_upload_rows(g->device, &s->normed, host, width, BENCH_ROWS) != 0)
        goto done;
    live = bench_now();
    for (int r = 0; r < reps; r++) {
        if (pn_qwen35_tt_project(g, &d->attn_qkv, &s->normed, &s->ssm_qkv, false) != 0 ||
            pn_qwen35_tt_project(g, &d->attn_gate, &s->normed, &s->ssm_z, false) != 0 ||
            pn_qwen35_tt_project(g, &d->ssm_alpha, &s->normed, &s->ssm_alpha, false) != 0 ||
            pn_qwen35_tt_project(g, &d->ssm_beta, &s->normed, &s->ssm_beta, false) != 0 ||
            pn_tt_matvec_flush(g->device) != 0)
            goto done;
    }
    live = bench_now() - live;

    printf("QUANTUM ssm projections x4   1 live row %.1f us   32 live rows %.1f us   "
           "ratio %.4f   (per position: %.1f us -> %.1f us)\n",
           1e6 * padded / reps, 1e6 * live / reps, live / padded,
           1e6 * padded / reps, 1e6 * live / reps / (double)BENCH_ROWS);
    rc = 0;
done:
    free(host);
    return rc;
}

/** The two terms a 32-row pass cannot batch, timed alone: one GDN block and, if it is armed,
 *  one on-card attention block. `32 * these + the flat part` is the whole item's ceiling. */
static int bench_sequential(pn_qwen35_tt *g, pn_qwen35moe_model *m,
                            const pn_qwen35moe_config *cfg, int reps)
{
    pn_qwen35_tt_strips *s = &g->card;
    const pn_qwen35_tt_layer *d_ssm = NULL, *d_attn = NULL;
    int ssm_layer = -1, attn_layer = -1;
    double mark;

    for (int i = 0; i < g->n_layers; i++) {
        if (!g->layers[i].is_attention && !d_ssm) { d_ssm = &g->layers[i]; ssm_layer = i; }
        if (g->layers[i].is_attention && !d_attn) { d_attn = &g->layers[i]; attn_layer = i; }
    }
    /* NON-FATAL, and reported as a refusal rather than a zero. These two blocks are reached
     * here OUTSIDE a forward — no embedding, no norms, no carried state at position 0 — so a
     * residency that will not take a descriptor in that context is a fact about the bench and
     * not about the block. The residency counters in a real run are the authority on their
     * cost; this is a cross-check, so a refusal must read as one. */
    if (d_ssm && g->gdn_strip) {
        const pn_qwen35moe_ssm_layer *l =
            pn_qwen35moe_model_ssm_layer(m, pn_qwen35moe_model_layer_index(m, ssm_layer));
        int failed = 0;

        memset(g->h_alpha, 0, sizeof(float) * 32u);
        memset(g->h_beta, 0, sizeof(float) * 32u);
        mark = bench_now();
        for (int r = 0; r < reps && l && !failed; r++)
            failed = pn_qwen35_tt_gdn_strip_layer(g, l, cfg,
                                                  pn_qwen35moe_model_layer_index(m, ssm_layer),
                                                  &s->ssm_qkv, &s->ssm_z, g->h_beta,
                                                  g->h_alpha, &s->ssm_gated) != 0;
        if (failed || !l)
            printf("QUANTUM gdn strip block      REFUSED outside a forward — read it off the "
                   "residency counters instead\n");
        else
            printf("QUANTUM gdn strip block      %.1f us/call  -> %.4f s over 24 layers\n",
                   1e6 * (bench_now() - mark) / reps, 24.0 * (bench_now() - mark) / reps);
    }
    if (d_attn && g->attn_on_card) {
        int idx = pn_qwen35moe_model_layer_index(m, attn_layer);
        int failed = 0;

        mark = bench_now();
        for (int r = 0; r < reps && !failed; r++)
            failed = pn_qwen35_tt_attn_block_card(g, d_attn, cfg, idx, r + 1, r + 2) != 0;
        if (failed)
            printf("QUANTUM attention block      REFUSED outside a forward — read it off the "
                   "residency counters instead\n");
        else
            printf("QUANTUM attention block      %.1f us/call  -> %.4f s over 8 layers "
                   "(at trivial context)\n",
                   1e6 * (bench_now() - mark) / reps, 8.0 * (bench_now() - mark) / reps);
    } else {
        printf("QUANTUM attention block      NOT ON CARD on this arm\n");
    }
    return 0;
}

int main(int argc, char **argv)
{
    const char *model_path = NULL;
    int reps = 20;
    pn_model *model;
    pn_qwen35moe_model *m;
    pn_qwen35_tt *g;
    const pn_qwen35moe_config *cfg;
    int rc = 1;

    for (int i = 1; i < argc; i++) {
        if (strcmp(argv[i], "--reps") == 0 && i + 1 < argc)
            reps = atoi(argv[++i]);
        else if (!model_path)
            model_path = argv[i];
    }
    if (!model_path || reps < 1) {
        fprintf(stderr, "usage: %s <model.gguf> [--reps N]\n", argv[0]);
        return 2;
    }
    model = pn_load(model_path);
    if (!model) {
        fprintf(stderr, "QUANTUM REFUSED: cannot load %s\n", model_path);
        return 2;
    }
    /* THE VACUOUS-RUN GUARD. Timing host memcpys against each other would answer nothing. */
    if (model->forward != pn_qwen35moe_forward_tt) {
        fprintf(stderr, "QUANTUM REFUSED: the engine did not select the qwen35 Tenstorrent "
                        "forward. Set PN_TT_ARTIFACT and check the card.\n");
        pn_model_free(model);
        return 2;
    }
    m = (pn_qwen35moe_model *)model->weights;
    g = (pn_qwen35_tt *)pn_qwen35moe_model_gpu(m);
    cfg = pn_qwen35moe_model_config(m);
    if (!g || !cfg || !g->resident) {
        fprintf(stderr, "QUANTUM REFUSED: no resident arm — there are no card strips to "
                        "time.\n");
        pn_model_free(model);
        return 2;
    }
    printf("QUANTUM model n_embd %d n_layer %d  strips: hidden %u ssm_qkv %u ssm_z %u "
           "ssm_gated %u ffn_swiglu %u tiles;  gdn_strip %d attn_on_card %d\n",
           cfg->n_embd, cfg->n_layer, g->card.hidden[0].tile_count, g->card.ssm_qkv.tile_count,
           g->card.ssm_z.tile_count, g->card.ssm_gated.tile_count,
           g->card.ffn_swiglu.tile_count, (int)g->gdn_strip, (int)g->attn_on_card);

    /* ORDER IS LOAD-BEARING, AND IT WAS MEASURED THE HARD WAY. A resident triad has an IDLE
     * SPIN LIMIT — the transport says so itself ("leaves core 71 resident until its idle spin
     * limit, measured at over 40 s") — and the transfer arms below take ~40 s of link with no
     * descriptor in them. Running them first retired the matmul residency mid-bench and every
     * projection after it came back `resident matmul Mt=1 Kt=128 Nt=256 refused`, which reads
     * as a tile-quantum result and is a bench artefact. The card arms run FIRST, while the
     * residencies are warm, and the transfers run last. */
    if (bench_selftest(g, cfg) != 0) {
        fprintf(stderr, "QUANTUM REFUSED: the transfers do not move the right bytes; the "
                        "timings below would be the cost of being wrong.\n");
        goto done;
    }
    if (bench_quantum(g, cfg, reps) != 0)
        goto done;
    if (bench_sequential(g, m, cfg, reps) != 0)
        goto done;
    if (bench_transfers(g, cfg, reps) != 0)
        goto done;
    rc = 0;
done:
    printf("QUANTUM VERDICT %s\n", rc == 0 ? "MEASURED" : "REFUSED");
    pn_model_free(model);
    return rc;
}

#endif /* USE_TT */
