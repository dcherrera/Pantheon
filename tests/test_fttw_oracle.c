/* SPDX-License-Identifier: AGPL-3.0-or-later */
/* Copyright (C) 2026 David (Corey) Herrera */
/**
 * @file test_fttw_oracle.c
 * @brief The card-format oracle's weight path: a
 *   metadata GGUF with an FTTW artifact attached reads every matmul weight, the embedding and
 *   the expert stacks from the artifact's tiles, and pn_qmv / pn_qmm over them equal a plain
 *   fp32 matvec of the source values.
 *
 * Fixtures are built here with foundry's own converter (fp32, bfp8, moe_bfp6 and bfp4 ledgers).
 * fp32 tiles must be EXACT; block-float tiles within their format step; bfp4 tiles EXACT against
 * f32 math on their ft_bfp_unpack_tiles values.
 *
 * PN_MIXW_BFP4 repack (gguf_fttw_repack.h): a one-tensor map turns
 * only that tensor's tiles into ft_bfp_pack_tiles(Bfp4) of its decoded Bfp8 values, and the
 * oracle reads them exactly; an empty map leaves the table, every artifact byte and every matvec
 * output identical; a name the model lacks (another layer), a typo, a non-Bfp8 tensor, another
 * format, a duplicate and a malformed line are refused with nothing changed. Run with `--red`
 * (PN_TILED_RED=1 is set by the Makefile rule): one tile per column decoded with its exponent
 * off by one, and the exact checks must FAIL.
 *
 * Card artifact beside a metadata GGUF (pn_gguf_attach_card_artifact, Pantheon Local's p150
 * install of Qwen3.6): `<stem>.gguf` + `<stem>.<rung>.fttw` attaches that artifact and nothing
 * else (not the `<stem>.<rung>.mtp.fttw` head, not another stem's); a full GGUF, or a
 * metadata GGUF with no artifact, is untouched; two candidates are refused.
 *
 * @status None
 * @issues None
 * @todo None
 */
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>

#include "../src/accel/simd.h"
#include "../src/core/qweight.h"
#include "../src/formats/gguf.h"
#include "../src/formats/gguf_dequant.h"
#include "../src/formats/gguf_fttw_overlay.h"
#include "../src/formats/gguf_fttw_repack.h"
#include "../src/formats/gguf_tiled.h"
#include "tt/convert/ft_bfp.h"
#include "tt/convert/ft_gguf_write.h"
#include "tt/convert/ft_tt_weight_convert.h"

#define K 64
#define N 96
#define VOCAB 128
#define E 4
#define EN 64
#define NP 48 /* N not whole tiles: Qwen3.8's ssm_alpha/ssm_beta are 5120 -> 48 */

static int checks, failures;

static void check(int ok, const char *what)
{
    checks++;
    failures += !ok;
    printf("  %s  %s\n", ok ? "ok  " : "FAIL", what);
}

static float value(unsigned seed, unsigned i)
{
    unsigned h = (i * 2654435761u) ^ (seed * 0x9E3779B9u);

    h ^= h >> 15;
    h *= 0x2C1B3C6Du;
    h ^= h >> 13;
    return ((float)(h % 2001u) - 1000.0f) / 997.0f;
}

static ft_gguf_write_tensor tensor(const char *name, unsigned dims, uint64_t d0, uint64_t d1, uint64_t d2,
                                   const float *data)
{
    ft_gguf_write_tensor t;

    memset(&t, 0, sizeof(t));
    t.name = name;
    t.n_dims = dims;
    t.dims[0] = d0;
    t.dims[1] = dims > 1 ? d1 : 1;
    t.dims[2] = dims > 2 ? d2 : 1;
    t.dims[3] = 1;
    t.data = data;
    t.byte_count = d0 * t.dims[1] * t.dims[2] * sizeof(float);
    return t;
}

static int write_gguf(const char *path, const ft_gguf_write_tensor *t, unsigned count)
{
    ft_gguf_kv_buffer kv;
    int rc;

    memset(&kv, 0, sizeof(kv));
    ft_gguf_kv_add_string(&kv, "general.architecture", "qwen35moe");
    rc = ft_gguf_write_file(path, kv.bytes, kv.used, kv.count, t, count, 32u);
    ft_gguf_kv_free(&kv);
    return rc;
}

static int convert(const char *gguf, const char *out, ft_tt_ledger_mode mode)
{
    ft_tt_convert_options o;

    memset(&o, 0, sizeof(o));
    o.mode = mode;
    o.scope = FT_TT_CONVERT_SCOPE_ALL;
    return ft_tt_convert_gguf(gguf, out, &o);
}

/** max |pn_qmv(w) - reference| over the rows, where reference[r] = pn_dot_f32(W[r], x). */
static double matvec_error(const pn_qweight *w, const float *source, const float *x)
{
    float *out = calloc((size_t)w->rows, sizeof(float));
    double worst = 0.0;

    pn_qmv(w, x, out);
    for (int r = 0; r < w->rows; r++)
        worst = fmax(worst, fabs((double)out[r] - pn_dot_f32(source + (size_t)r * (size_t)w->cols, x, w->cols)));
    free(out);
    return worst;
}

/** One artifact arm: `exact` demands bit-equal outputs; otherwise within `tolerance`. */
static void arm(const char *label, const char *fttw, const float *qkv, const float *embd, const float *exps,
                const float *alpha, const float *x, int exact, double tolerance)
{
    char what[160];
    pn_gguf_file *g = pn_gguf_open("build/fttw_oracle_meta.gguf");
    pn_qweight w, expert;
    float row[K], *batch_out = calloc(3u * N, sizeof(float)), *xs = malloc(3u * K * sizeof(float));
    double err, worst_row = 0.0, batch_worst = 0.0;
    int attached;

    printf("\n== %s ==\n", label);
    attached = g ? pn_gguf_attach_fttw(g, fttw) : -1;
    snprintf(what, sizeof(what), "attached 5 tiled tensors, the tied head included (got %d)", attached);
    check(attached == 5, what);
    if (attached != 5 || pn_qweight_from_gguf(&w, g, "blk.0.attn_qkv.weight") != 0) {
        pn_gguf_close(g);
        free(batch_out);
        free(xs);
        return;
    }
    err = matvec_error(&w, qkv, x);
    snprintf(what, sizeof(what), "attn_qkv [%d x %d] matvec: max |err| %.3g (%s)", w.rows, w.cols, err,
             exact ? "must be 0" : "within the format step");
    check(exact ? err == 0.0 : err <= tolerance, what);

    for (int i = 0; i < 3 * K; i++)
        xs[i] = value(9, (unsigned)i);
    pn_qmm(&w, xs, 3, batch_out);
    for (int b = 0; b < 3; b++) {
        float one[N];

        pn_qmv(&w, xs + b * K, one);
        for (int r = 0; r < N; r++)
            batch_worst = fmax(batch_worst, fabs((double)batch_out[b * N + r] - one[r]));
    }
    check(batch_worst == 0.0, "pn_qmm over 3 rows == 3 x pn_qmv, bit for bit");

    for (unsigned t = 0; t < VOCAB; t += 37) {
        if (pn_gguf_dequantize_row(g, "token_embd.weight", t, K, row) != 0)
            worst_row = 1e30;
        for (int c = 0; c < K; c++)
            worst_row = fmax(worst_row, fabs((double)row[c] - embd[t * K + c]));
    }
    snprintf(what, sizeof(what), "embedding rows (source layout) max |err| %.3g", worst_row);
    check(exact ? worst_row == 0.0 : worst_row <= tolerance, what);

    {
        pn_gguf_tensor_info *info = pn_gguf_find_tensor(g, "blk.0.ffn_gate_exps.weight");
        size_t row_stride = pn_gguf_type_size(info->type, K);
        double worst_expert = 0.0;

        for (int e = 0; e < E; e++) {
            expert = (pn_qweight){ .data = (const char *)g->data + info->offset + (size_t)e * EN * row_stride,
                                   .type = info->type, .rows = EN, .cols = K, .row_stride = row_stride };
            worst_expert = fmax(worst_expert, matvec_error(&expert, exps + (size_t)e * EN * K, x));
        }
        snprintf(what, sizeof(what), "expert slices at base + e * rows * row_stride: max |err| %.3g", worst_expert);
        check(exact ? worst_expert == 0.0 : worst_expert <= tolerance, what);
    }
    {
        /* N = 48: two stored tile columns, the second half zero padding. Every one of the 48
         * outputs is checked against the source (row 47 is in the partial column), and the
         * output buffer has a guard past row 47 that must stay untouched. */
        pn_gguf_tensor_info *info = pn_gguf_find_tensor(g, "blk.0.ssm_alpha.weight");
        pn_qweight a;
        float out[3 * NP + 1], one[NP + 1];
        double padded_batch = 0.0;

        check(info && (info->dims[1] == NP), "ssm_alpha dims [64, 48] attached (N = 48)");
        if (info && pn_qweight_from_gguf(&a, g, "blk.0.ssm_alpha.weight") == 0) {
            err = matvec_error(&a, alpha, x);
            snprintf(what, sizeof(what), "ssm_alpha [%d x %d] (N not whole tiles) matvec: max |err| %.3g (%s)",
                     a.rows, a.cols, err, exact ? "must be 0" : "within the format step");
            check(a.rows == NP && (exact ? err == 0.0 : err <= tolerance), what);
            one[NP] = 12345.0f;
            pn_qmv(&a, x, one);
            check(one[NP] == 12345.0f, "ssm_alpha matvec writes rows 0..47 only (guard intact)");
            out[3 * NP] = 12345.0f;
            pn_qmm(&a, xs, 3, out);
            for (int b = 0; b < 3; b++) {
                pn_qmv(&a, xs + b * K, one);
                for (int r = 0; r < NP; r++)
                    padded_batch = fmax(padded_batch, fabs((double)out[b * NP + r] - one[r]));
            }
            check(padded_batch == 0.0 && out[3 * NP] == 12345.0f,
                  "ssm_alpha pn_qmm over 3 rows == 3 x pn_qmv, bit for bit, guard intact");
        } else {
            check(0, "ssm_alpha opens as a pn_qweight");
        }
    }
    pn_gguf_close(g);
    free(batch_out);
    free(xs);
}

/**
 * W'[n][k] of an N-major tiled tensor ([rows x cols], cols = K whole tiles, rows padded to a
 * tile), straight from foundry's ft_bfp_unpack_tiles: the values the card multiplies. Tile
 * (g, kt) holds tile[k * 32 + n] = W[32g + n][32kt + k].
 */
static float *nmajor_decoded(ft_bfp_format format, const void *bytes, int rows, int cols)
{
    const int kts = cols / 32, groups = (rows + 31) / 32;
    float *tiles = malloc((size_t)groups * kts * 1024u * sizeof(float));
    float *w = malloc((size_t)rows * cols * sizeof(float));

    if (!tiles || !w || ft_bfp_unpack_tiles(format, bytes, (size_t)groups * kts, tiles) != FT_BFP_OK) {
        free(tiles);
        free(w);
        return NULL;
    }
    for (int g = 0; g < groups; g++)
        for (int kt = 0; kt < kts; kt++)
            for (int k = 0; k < 32; k++)
                for (int n = 0; n < 32 && 32 * g + n < rows; n++)
                    w[(size_t)(32 * g + n) * cols + 32 * kt + k] = tiles[((size_t)g * kts + kt) * 1024u + k * 32 + n];
    free(tiles);
    return w;
}

/**
 * pn_qmv over `name` (experts: every slice) must equal f32 math on the ft_bfp-decoded values
 * of its own tiles, bit for bit; `source` (the GGUF values) bounds the decode within `step`.
 */
static void exact_on_decoded(pn_gguf_file *g, const char *name, ft_bfp_format format, const float *source,
                             const float *x, double step)
{
    pn_gguf_tensor_info *info = pn_gguf_find_tensor(g, name);
    const int experts = info && info->n_dims > 2 ? (int)info->dims[2] : 1;
    const int rows = info ? (int)info->dims[1] : 0, cols = info ? (int)info->dims[0] : 0;
    double err = 0.0, off_source = 0.0;
    char what[200];

    if (!info) {
        check(0, name);
        return;
    }
    for (int e = 0; e < experts; e++) {
        size_t row_stride = pn_gguf_type_size(info->type, (size_t)cols);
        size_t slice_bytes = (size_t)((rows + 31) / 32) * (cols / 32) * pn_tiled_tile_bytes(info->type);
        const char *base = (const char *)g->data + info->offset + (size_t)e * slice_bytes;
        pn_qweight w = { .data = base, .type = info->type, .rows = rows, .cols = cols, .row_stride = row_stride };
        float *decoded = nmajor_decoded(format, base, rows, cols);

        if (!decoded) {
            check(0, "decode the reference tiles");
            return;
        }
        err = fmax(err, matvec_error(&w, decoded, x));
        for (size_t i = 0; i < (size_t)rows * cols; i++)
            off_source = fmax(off_source, fabs((double)decoded[i] - source[(size_t)e * rows * cols + i]));
        free(decoded);
    }
    snprintf(what, sizeof(what), "%s [%d x %d%s]: oracle matvec == f32 math on ft_bfp_unpack_tiles values "
             "(max |err| %.3g, must be 0); decode within %.3g of the source (%.3g)", name, rows, cols,
             experts > 1 ? " x 4 experts" : "", err, step, off_source);
    check(err == 0.0 && off_source <= step, what);
}

/**
 * The Bfp4 arm: the bfp4 ledger rung (no embedding: Bfp4 in source layout is refused by the
 * overlay) attaches as PN_QUANT_TILED_BFP4 and the oracle reads its exact tile values.
 * Bfp4_b: 3 magnitude bits; |source| < 1.01 shares exponent 0, a step of 0.25, rounded: 0.125.
 */
static void bfp4_arm(const float *qkv, const float *exps, const float *alpha, const float *x)
{
    pn_gguf_file *g = pn_gguf_open("build/fttw_oracle_meta.gguf");
    pn_gguf_tensor_info *info;
    int attached;

    printf("\n== bfp4 tiles: EXACT on the decoded values ==\n");
    attached = g ? pn_gguf_attach_fttw(g, "build/fttw_oracle.bfp4.fttw") : -1;
    info = attached > 0 ? pn_gguf_find_tensor(g, "blk.0.attn_qkv.weight") : NULL;
    check(attached == 3 && info && info->type == PN_QUANT_TILED_BFP4,
          "attached 3 tiled tensors, attn_qkv as PN_QUANT_TILED_BFP4");
    if (attached == 3) {
        exact_on_decoded(g, "blk.0.attn_qkv.weight", FT_BFP_BFP4_B, qkv, x, 0.126);
        exact_on_decoded(g, "blk.0.ffn_gate_exps.weight", FT_BFP_BFP4_B, exps, x, 0.126);
        exact_on_decoded(g, "blk.0.ssm_alpha.weight", FT_BFP_BFP4_B, alpha, x, 0.126);
    }
    pn_gguf_close(g);
}

/** Write `text` to `path`; 0 on success. */
static int write_text(const char *path, const char *text)
{
    FILE *f = fopen(path, "w");
    int ok = f && fputs(text, f) >= 0;

    if (f && fclose(f) != 0)
        ok = 0;
    return ok ? 0 : -1;
}

/** Tiles an N-major tensor stores (N padded to a tile, times experts). */
static size_t stored_tiles(const pn_gguf_tensor_info *t)
{
    size_t tiles = (size_t)(t->dims[0] / 32) * (size_t)((t->dims[1] + 31) / 32);

    for (uint32_t d = 2; d < t->n_dims; d++)
        tiles *= (size_t)t->dims[d];
    return tiles;
}

/** The bfp8 fixture attached to the metadata GGUF, the map at `map_text` applied; *rc = repack rc. */
static pn_gguf_file *open_repacked(const char *map_text, int *rc)
{
    pn_gguf_file *g = pn_gguf_open("build/fttw_oracle_meta.gguf");

    *rc = -9;
    if (!g || pn_gguf_attach_fttw(g, "build/fttw_oracle.bfp8.fttw") != 5 ||
        write_text("build/fttw_mixw.map", map_text) != 0) {
        pn_gguf_close(g);
        return NULL;
    }
    *rc = pn_gguf_repack_bfp4_from_map_file(g, "build/fttw_mixw.map");
    return g;
}

/** Same tensor table (types, offsets) and the same artifact bytes outside [skip, skip + skip_bytes). */
static int same_as_base(const pn_gguf_file *g, const pn_gguf_file *base, size_t skip, size_t skip_bytes)
{
    const unsigned char *a = g->overlay_base, *b = base->overlay_base;

    if (g->n_tensors != base->n_tensors || g->overlay_size != base->overlay_size)
        return 0;
    for (uint64_t i = 0; i < g->n_tensors; i++)
        if (g->tensors[i].type != base->tensors[i].type || g->tensors[i].offset != base->tensors[i].offset)
            return 0;
    return memcmp(a, b, skip) == 0 &&
           memcmp(a + skip + skip_bytes, b + skip + skip_bytes, g->overlay_size - skip - skip_bytes) == 0;
}

/**
 * `name` was repacked: its type is Bfp4, its first tiles * 576 bytes equal ft_bfp_pack_tiles
 * (Bfp4) of ft_bfp_unpack_tiles (Bfp8) of the base's bytes, and nothing else in the artifact
 * or the tensor table changed (the stale tail of its span included).
 */
static void check_repacked(const pn_gguf_file *g, const pn_gguf_file *base, const char *name)
{
    pn_gguf_tensor_info *t = pn_gguf_find_tensor((pn_gguf_file *)g, name);
    pn_gguf_tensor_info *b = pn_gguf_find_tensor((pn_gguf_file *)base, name);
    const size_t tiles = b ? stored_tiles(b) : 0;
    float *values = malloc(tiles * 1024u * sizeof(float) + 1u);
    unsigned char *want = malloc(tiles * 576u + 1u);
    const unsigned char *at = t ? (const unsigned char *)g->data + t->offset : NULL;
    const unsigned char *from = b ? (const unsigned char *)base->data + b->offset : NULL;
    char what[200];
    int ok = t && b && values && want && t->type == PN_QUANT_TILED_BFP4 &&
             ft_bfp_unpack_tiles(FT_BFP_BFP8_B, from, tiles, values) == FT_BFP_OK &&
             ft_bfp_pack_tiles(FT_BFP_BFP4_B, values, tiles, want) == FT_BFP_OK && memcmp(at, want, tiles * 576u) == 0;

    snprintf(what, sizeof(what), "%s: now TILED_BFP4, its %zu tiles == ft_bfp_pack_tiles(bfp4, unpack(bfp8))", name,
             tiles);
    check(ok, what);
    if (ok) {
        pn_gguf_tensor_info saved = *t;

        t->type = b->type; /* compare the rest of the table and artifact with this span masked */
        ok = memcmp(at + tiles * 576u, from + tiles * 576u, tiles * (1088u - 576u)) == 0 &&
             same_as_base(g, base, (size_t)(at - (const unsigned char *)g->overlay_base), tiles * 1088u);
        *t = saved;
    }
    snprintf(what, sizeof(what), "%s: every other tensor, type and artifact byte unchanged", name);
    check(ok, what);
    free(values);
    free(want);
}

/** The artifact file on disk still equals the base's (never repacked) mapping. */
static int file_untouched(const pn_gguf_file *base)
{
    FILE *f = fopen("build/fttw_oracle.bfp8.fttw", "rb");
    unsigned char *bytes = malloc(base->overlay_size + 1u);
    int same = f && bytes && fread(bytes, 1, base->overlay_size + 1u, f) == base->overlay_size &&
               memcmp(bytes, base->overlay_base, base->overlay_size) == 0;

    if (f)
        fclose(f);
    free(bytes);
    return same;
}

/** Every 2-D tensor's matvec output, bit for bit, equal on g and base (the forward's only weight reads). */
static int matvecs_identical(pn_gguf_file *g, pn_gguf_file *base, const float *x)
{
    static const char *names[] = { "blk.0.attn_qkv.weight", "blk.0.ssm_alpha.weight" };
    float a[N], b[N];

    for (size_t i = 0; i < sizeof(names) / sizeof(names[0]); i++) {
        pn_qweight wa, wb;

        if (pn_qweight_from_gguf(&wa, g, names[i]) != 0 || pn_qweight_from_gguf(&wb, base, names[i]) != 0)
            return 0;
        pn_qmv(&wa, x, a);
        pn_qmv(&wb, x, b);
        if (memcmp(a, b, (size_t)wa.rows * sizeof(float)) != 0)
            return 0;
    }
    return 1;
}

/** A refused map: rc -1 and nothing changed at all. */
static void check_refused(const pn_gguf_file *base, const char *map_text, const char *what)
{
    int rc;
    pn_gguf_file *g = open_repacked(map_text, &rc);
    char line[200];

    snprintf(line, sizeof(line), "refused, nothing changed: %s (rc %d)", what, rc);
    check(g && rc == -1 && same_as_base(g, base, 0, 0), line);
    pn_gguf_close(g);
}

/**
 * PN_MIXW_BFP4's repack on the bfp8 fixture: a one-tensor map
 * changes that tensor's tiles only, to exactly ft_bfp_pack_tiles of the decoded Bfp8 values,
 * and the oracle reads them; an empty map changes nothing; bad maps are refused untouched.
 */
static void mixw_repack(const float *qkv, const float *exps, const float *alpha, const float *x)
{
    pn_gguf_file *base = pn_gguf_open("build/fttw_oracle_meta.gguf");
    pn_gguf_file *g;
    int rc;

    printf("\n== PN_MIXW_BFP4: Bfp8 -> Bfp4 repack in memory ==\n");
    if (!base || pn_gguf_attach_fttw(base, "build/fttw_oracle.bfp8.fttw") != 5) {
        check(0, "base attach");
        pn_gguf_close(base);
        return;
    }
    /* Bfp8 then Bfp4: within 0.125 (Bfp4) + 1/128 (Bfp8's rounding) of the source. */
    g = open_repacked("# one tensor\nblk.0.attn_qkv.weight bfp4_b   # trailing comment\n\n", &rc);
    check(g && rc == 1, "one-tensor map: 1 tensor repacked");
    if (g && rc == 1) {
        check_repacked(g, base, "blk.0.attn_qkv.weight");
        exact_on_decoded(g, "blk.0.attn_qkv.weight", FT_BFP_BFP4_B, qkv, x, 0.134);
    }
    pn_gguf_close(g);
    g = open_repacked("blk.0.ssm_alpha.weight bfp4_b\nblk.0.ffn_gate_exps.weight\tbfp4_b\n", &rc);
    check(g && rc == 2, "padded-N + expert-stack map: 2 tensors repacked");
    if (g && rc == 2) {
        exact_on_decoded(g, "blk.0.ssm_alpha.weight", FT_BFP_BFP4_B, alpha, x, 0.134);
        exact_on_decoded(g, "blk.0.ffn_gate_exps.weight", FT_BFP_BFP4_B, exps, x, 0.134);
    }
    pn_gguf_close(g);
    check(file_untouched(base), "the artifact file on disk is unchanged (MAP_PRIVATE copy-on-write)");

    g = open_repacked("# nothing\n\n   \n", &rc);
    check(g && rc == 0 && same_as_base(g, base, 0, 0) && matvecs_identical(g, base, x),
          "empty map: 0 repacked, table + artifact bytes + every matvec output byte-identical");
    pn_gguf_close(g);

    check_refused(base, "blk.1.attn_qkv.weight bfp4_b\n", "a tensor in a layer the model lacks");
    check_refused(base, "blk.0.attn_qkv.weight bfp4_b\nblk.0.attn_qvk.weight bfp4_b\n",
                  "a typo after a good entry (all-or-nothing)");
    check_refused(base, "token_embd.weight bfp4_b\n", "the source-layout embedding");
    check_refused(base, "blk.0.attn_qkv.weight bfp8_b\n", "a format other than bfp4_b");
    check_refused(base, "blk.0.attn_qkv.weight bfp4_b\nblk.0.attn_qkv.weight bfp4_b\n", "a name listed twice");
    check_refused(base, "blk.0.attn_qkv.weight bfp4_b extra\n", "a line of three fields");
    check_refused(base, "blk.0.attn_qkv.weight\n", "a line of one field");
    pn_gguf_close(base);
}

/** `dir/name` -> `target` (relative to the cwd), replacing what is there. */
static int link_as(const char *dir, const char *name, const char *target)
{
    char cwd[1024], from[1400], to[1400];

    if (!getcwd(cwd, sizeof(cwd)))
        return -1;
    snprintf(from, sizeof(from), "%s/%s", dir, name);
    snprintf(to, sizeof(to), "%s/%s", cwd, target);
    unlink(from);
    return symlink(to, from);
}

/** pn_gguf_attach_card_artifact on `dir/name`: its rc, the tensors added, the path chosen. */
static int beside(const char *dir, const char *name, int *added, char *path, size_t cap)
{
    char gguf[512];
    pn_gguf_file *g;
    uint64_t before;
    int rc;

    snprintf(gguf, sizeof(gguf), "%s/%s", dir, name);
    path[0] = '\0';
    *added = -1;
    if (!(g = pn_gguf_open(gguf)))
        return -9;
    before = g->n_tensors;
    rc = pn_gguf_attach_card_artifact(g, gguf, path, cap);
    *added = (int)(g->n_tensors - before);
    if (rc == 1 && !pn_gguf_find_tensor(g, "blk.0.attn_qkv.weight"))
        rc = -8;
    pn_gguf_close(g);
    return rc;
}

#define Q "qwen3.6-35b-a3b-q8-90f46deabc2f"

static void card_artifact_beside(void)
{
    const char *d = "build/fttw_beside";
    char path[1024], what[200];
    int added, rc;

    printf("\n== card artifact beside a metadata GGUF ==\n");
    if (system("rm -rf build/fttw_beside && mkdir -p build/fttw_beside") != 0) {
        check(0, "fixture dir");
        return;
    }
    /* The names a Pantheon Local store holds after a p150 install of Qwen3.6: the files as
     * `<id>-<sha12>.<ext>`, the companion links as `<id>-<weights sha12>.<rest>`. */
    check(link_as(d, Q ".gguf", "build/fttw_oracle_meta.gguf") == 0 &&
              link_as(d, Q ".moe_bfp6.fttw", "build/fttw_oracle.moe_bfp6.fttw") == 0 &&
              link_as(d, Q ".moe_bfp6.mtp.fttw", "build/fttw_oracle.fp32.fttw") == 0 &&
              link_as(d, "qwen3.6-35b-a3b-q8-deec8f87d0d0.fttw", "build/fttw_oracle.bfp8.fttw") == 0 &&
              link_as(d, Q "x.bfp8.fttw", "build/fttw_oracle.bfp8.fttw") == 0 &&
              link_as(d, "full.gguf", "build/fttw_oracle.gguf") == 0 &&
              link_as(d, "full.bfp8.fttw", "build/fttw_oracle.bfp8.fttw") == 0 &&
              link_as(d, "lone.gguf", "build/fttw_oracle_meta.gguf") == 0,
          "store dir: <id>-<meta sha12>.{gguf,moe_bfp6.fttw,moe_bfp6.mtp.fttw}, the trunk's "
          "own store name, another stem's artifact, full.gguf + full.bfp8.fttw, lone.gguf");
    rc = beside(d, Q ".gguf", &added, path, sizeof(path));
    snprintf(what, sizeof(what), "metadata GGUF: attached its artifact (rc %d, %d tensors, %s)", rc,
             added, path);
    check(rc == 1 && added == 5 && strcmp(path, "build/fttw_beside/" Q ".moe_bfp6.fttw") == 0, what);
    rc = beside(d, "full.gguf", &added, path, sizeof(path));
    check(rc == 0 && added == 0, "a full GGUF (has token_embd) is untouched, artifact or not");
    rc = beside(d, "lone.gguf", &added, path, sizeof(path));
    check(rc == 0 && added == 0, "a metadata GGUF with no artifact beside it is untouched");
    link_as(d, Q ".bfp8.fttw", "build/fttw_oracle.bfp8.fttw");
    rc = beside(d, Q ".gguf", &added, path, sizeof(path));
    check(rc == -1 && added == 0, "two artifacts beside it: refused, nothing attached");
}

int main(int argc, char **argv)
{
    float *qkv = malloc((size_t)N * K * sizeof(float));
    float *embd = malloc((size_t)VOCAB * K * sizeof(float));
    float *exps = malloc((size_t)E * EN * K * sizeof(float));
    float *alpha = malloc((size_t)NP * K * sizeof(float));
    float norm[K], x[K];
    ft_gguf_write_tensor t[5], bfp4_tensors[4];
    int red = argc > 1 && strcmp(argv[1], "--red") == 0;

    for (int i = 0; i < N * K; i++)
        qkv[i] = value(1, (unsigned)i);
    for (int i = 0; i < VOCAB * K; i++)
        embd[i] = value(2, (unsigned)i);
    for (int i = 0; i < E * EN * K; i++)
        exps[i] = value(3, (unsigned)i);
    for (int i = 0; i < NP * K; i++)
        alpha[i] = value(6, (unsigned)i);
    for (int i = 0; i < K; i++) {
        norm[i] = 1.0f + value(4, (unsigned)i) / 8.0f;
        x[i] = value(5, (unsigned)i);
    }
    t[0] = tensor("token_embd.weight", 2, K, VOCAB, 1, embd);
    t[1] = tensor("blk.0.attn_qkv.weight", 2, K, N, 1, qkv);
    t[2] = tensor("blk.0.ffn_gate_exps.weight", 3, K, EN, E, exps);
    t[3] = tensor("blk.0.attn_norm.weight", 1, K, 1, 1, norm);
    t[4] = tensor("blk.0.ssm_alpha.weight", 2, K, NP, 1, alpha);
    printf("== fixtures (foundry converter) ==%s\n", red ? "  [RED: PN_TILED_RED=1]" : "");
    check(write_gguf("build/fttw_oracle.gguf", t, 5) == 0 && write_gguf("build/fttw_oracle_meta.gguf", &t[3], 1) == 0,
          "wrote the source GGUF and its metadata GGUF (the 1-D norm only)");
    check(convert("build/fttw_oracle.gguf", "build/fttw_oracle.fp32.fttw", FT_TT_LEDGER_FP32) == 0 &&
              convert("build/fttw_oracle.gguf", "build/fttw_oracle.bfp8.fttw", FT_TT_LEDGER_BFP8) == 0 &&
              convert("build/fttw_oracle.gguf", "build/fttw_oracle.moe_bfp6.fttw", FT_TT_LEDGER_MOE_BFP6) == 0,
          "converted at fp32, bfp8 and moe_bfp6");
    bfp4_tensors[0] = t[1];
    bfp4_tensors[1] = t[2];
    bfp4_tensors[2] = t[3];
    bfp4_tensors[3] = t[4];
    check(write_gguf("build/fttw_oracle_b4.gguf", bfp4_tensors, 4) == 0 &&
              convert("build/fttw_oracle_b4.gguf", "build/fttw_oracle.bfp4.fttw", FT_TT_LEDGER_BFP4) == 0,
          "converted the embedding-free GGUF at bfp4");

    arm("fp32 tiles: EXACT", "build/fttw_oracle.fp32.fttw", qkv, embd, exps, alpha, x, 1, 0.0);
    /* Bfp8_b: 7-bit magnitude under a 16-value shared exponent; values <= 2.0 here, so a
     * weight is within 2^-6 and a 64-term dot of |x| <= 1 within 64 * 2^-6 = 1.0. */
    arm("bfp8 tiles", "build/fttw_oracle.bfp8.fttw", qkv, embd, exps, alpha, x, 0, 1.0);
    arm("moe_bfp6 tiles (experts Bfp6_b)", "build/fttw_oracle.moe_bfp6.fttw", qkv, embd, exps, alpha, x, 0, 1.0);
    bfp4_arm(qkv, exps, alpha, x);
    mixw_repack(qkv, exps, alpha, x);
    card_artifact_beside();

    printf("\n%d checks, %d failed%s\n", checks, failures,
           red ? " (RED run: the exact checks MUST fail)" : "");
    free(qkv);
    free(embd);
    free(exps);
    free(alpha);
    return failures ? 1 : 0;
}
