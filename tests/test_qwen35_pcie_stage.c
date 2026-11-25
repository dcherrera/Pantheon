/* SPDX-License-Identifier: AGPL-3.0-or-later */
/* Copyright (C) 2026 David (Corey) Herrera */
/**
 * @file test_qwen35_pcie_stage.c
 * @brief qwen35moe_pcie_stage: the QWEN35_PCIE_* flags parse, and on two GPUs every lever
 *        set moves the exact bytes in the right order (input fan-out, stage hop, positions).
 *
 * Host part (always): flag parsing.
 * GPU part (USE_CUDA build, QW_PCIE_TEST_GPU=1, two visible devices A and B on one board):
 * roles stage 0 = (A, B), stage 1 = (B, A), so the hop crosses streams. Each round queues
 * long copy work on both dies first, then the round's input fan-out, a slow producer of
 * stage 0's output, the hop and the position uploads, all behind that work; the next round
 * starts at once (the host rewrites the input rows while the last uploads are still queued).
 * After all rounds every destination of every round must hold that round's bytes.
 * Red build (CFLAGS+=-DPN_PCIE_RED, PN_PCIE_RED=hop_wait|d2h_order|peer|reuse) must fail.
 *
 * Run: QW_PCIE_TEST_GPU=1 CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=<a>,<b> \
 *      ./tests/test_qwen35_pcie_stage
 *
 * @status complete
 * @issues None
 * @todo None
 */
#include "../src/models/qwen35moe_pcie_stage.h"

#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static int failures;

#define CHECK(cond, ...) do { if (!(cond)) { failures++; printf("  FAIL: " __VA_ARGS__); \
                                             printf("\n"); } } while (0)

static void test_flags_parse(void)
{
    qw_pcie_flags f = qw_pcie_flags_parse(NULL, NULL, NULL, NULL);
    CHECK(f.pin && f.fanout && f.hop_event && f.pos && qw_pcie_flags_any(f),
          "unset flags are on (the default)");
    f = qw_pcie_flags_parse("0", "0", "0", "0");
    CHECK(!f.pin && !f.fanout && !f.hop_event && !f.pos && !qw_pcie_flags_any(f),
          "\"0\" turns every flag off");
    f = qw_pcie_flags_parse("1", "0", "", "yes");
    CHECK(f.pin && !f.fanout && f.hop_event && !f.pos && qw_pcie_flags_any(f),
          "\"1\" and \"\" are on, anything else is off");
    f = qw_pcie_flags_parse("10", "0", "0", "0");
    CHECK(!f.pin && !qw_pcie_flags_any(f), "\"10\" is not \"1\"");
    f = qw_pcie_flags_parse("0", NULL, "0", "0");
    CHECK(f.fanout && qw_pcie_flags_any(f), "fanout alone counts as on");
    /* QWEN35_SHORT_ROW / QWEN35_TOPK_FAST: same parse, default on. */
    CHECK(qw_env_flag_default_on(NULL) && qw_env_flag_default_on("") && qw_env_flag_default_on("1"),
          "short row / fast top-k: unset, \"\" and \"1\" are on");
    CHECK(!qw_env_flag_default_on("0") && !qw_env_flag_default_on("10") && !qw_env_flag_default_on("yes"),
          "short row / fast top-k: \"0\" and anything else are off");
}

#ifdef USE_CUDA

#define ROUNDS 6
#define ROW_BYTES ((size_t)32768)       /* 4 slots x 2048 fp32 */
#define SMALL_BYTES ((size_t)128)
#define BUSY_BYTES ((size_t)16 << 20)

static void fill(uint8_t *p, size_t n, unsigned seed)
{
    for (size_t i = 0; i < n; i++)
        p[i] = (uint8_t)((i * 151u + seed * 29u + (i >> 7)) & 0xffu);
}

static size_t mismatches(const uint8_t *p, size_t n, unsigned seed)
{
    size_t bad = 0;
    for (size_t i = 0; i < n; i++)
        bad += p[i] != (uint8_t)((i * 151u + seed * 29u + (i >> 7)) & 0xffu);
    return bad;
}

typedef struct {
    fc_cuda_ctx *a, *b;
    fc_buffer *busy_src[2], *busy_dst[2];
    fc_buffer *pats;                     /* on A: stage 0 output pattern of every round */
    fc_buffer *src0;                     /* on A: stage 0 rank 0 output */
    fc_buffer *dst[ROUNDS][4];           /* per round: in r0 (A), in r1 (B), hop r0 (B), hop r1 (A) */
    fc_buffer *pos[ROUNDS][4];           /* per round, per role die */
} gpu_rig;

/** Queue ~16 long copy kernels on ctx (rank i's busy buffers). */
static int queue_busy(gpu_rig *g, int i, fc_cuda_ctx *ctx)
{
    for (int k = 0; k < 16; k++)
        if (fc_buffer_copy_2d_kernel_async(g->busy_dst[i], 0, 16384, g->busy_src[i], 0, 16384,
                                           16384, 1024, ctx) != 0)
            return -1;
    return 0;
}

static int rig_init(gpu_rig *g)
{
    fc_cuda_ctx *role[4] = { g->a, g->b, g->b, g->a };
    for (int i = 0; i < 2; i++) {
        fc_cuda_ctx *c = i ? g->b : g->a;
        if (!(g->busy_src[i] = fc_buffer_alloc_ctx(c, BUSY_BYTES)) ||
            !(g->busy_dst[i] = fc_buffer_alloc_ctx(c, BUSY_BYTES)))
            return -1;
    }
    if (!(g->pats = fc_buffer_alloc_ctx(g->a, ROUNDS * ROW_BYTES)) ||
        !(g->src0 = fc_buffer_alloc_ctx(g->a, ROW_BYTES)))
        return -1;
    uint8_t *h = malloc(ROUNDS * ROW_BYTES);
    if (!h)
        return -1;
    for (int r = 0; r < ROUNDS; r++)
        fill(h + r * ROW_BYTES, ROW_BYTES, 1000u + (unsigned)r);
    fc_buffer_from_cpu_n(g->pats, h, ROUNDS * ROW_BYTES);
    free(h);
    for (int r = 0; r < ROUNDS; r++)
        for (int d = 0; d < 4; d++)
            if (!(g->dst[r][d] = fc_buffer_alloc_ctx(role[d], ROW_BYTES)) ||
                !(g->pos[r][d] = fc_buffer_alloc_ctx(role[d], SMALL_BYTES)))
                return -1;
    return 0;
}

/** Zero every destination so a copy that never lands is seen. */
static void rig_clear(gpu_rig *g)
{
    static uint8_t zero[ROW_BYTES];
    for (int r = 0; r < ROUNDS; r++)
        for (int d = 0; d < 4; d++) {
            fc_buffer_from_cpu_n(g->dst[r][d], zero, ROW_BYTES);
            fc_buffer_from_cpu_n(g->pos[r][d], zero, SMALL_BYTES);
        }
}

/** One round r through the staging; returns 0, or -1 on a device error. */
static int run_round(gpu_rig *g, qw_pcie_stage *st, int r)
{
    float *rows = qw_pcie_stage_input_rows(st);
    if (!rows)
        return -1;
    fill((uint8_t *)rows, ROW_BYTES, 10u + (unsigned)r);
    if (queue_busy(g, 0, g->a) != 0 || queue_busy(g, 1, g->b) != 0)
        return -1;
    if (qw_pcie_stage_fan_in(st, g->dst[r][0], g->dst[r][1], ROW_BYTES) != 0)
        return -1;
    /* The slow producer: stage 0's output is written after A's busy work. */
    if (fc_buffer_copy_2d_kernel_async(g->src0, 0, 16384, g->pats, (size_t)r * ROW_BYTES, 16384,
                                       16384, ROW_BYTES / 16384, g->a) != 0)
        return -1;
    if (qw_pcie_stage_hop(st, g->src0, g->dst[r][2], g->dst[r][3], ROW_BYTES) != 0)
        return -1;
    for (int d = 0; d < 4; d++) {
        uint8_t small[SMALL_BYTES];
        fill(small, SMALL_BYTES, 500u + (unsigned)(r * 4 + d));
        if (qw_pcie_stage_upload_small(st, d, g->pos[r][d], small, SMALL_BYTES) != 0)
            return -1;
    }
    return 0;
}

/** Bytes wrong across every round's destinations. */
static size_t rig_check(gpu_rig *g)
{
    static uint8_t h[ROW_BYTES];
    size_t bad = 0;
    fc_sync_ctx(g->a);
    fc_sync_ctx(g->b);
    for (int r = 0; r < ROUNDS; r++) {
        for (int d = 0; d < 4; d++) {
            fc_buffer_to_cpu_n(g->dst[r][d], h, ROW_BYTES);
            bad += mismatches(h, ROW_BYTES, d < 2 ? 10u + (unsigned)r : 1000u + (unsigned)r);
            fc_buffer_to_cpu_n(g->pos[r][d], h, SMALL_BYTES);
            bad += mismatches(h, SMALL_BYTES, 500u + (unsigned)(r * 4 + d));
        }
    }
    return bad;
}

static void test_gpu_levers(void)
{
    gpu_rig g;
    memset(&g, 0, sizeof(g));
    g.a = fc_cuda_init_device(0);
    g.b = fc_cuda_init_device(1);
    if (!g.a || !g.b || fc_cuda_enable_peer(g.a, g.b) != 0 || rig_init(&g) != 0) {
        failures++;
        printf("  FAIL: GPU rig setup (two peer-capable devices needed)\n");
        return;
    }
    static const struct { const char *name; qw_pcie_flags f; } sets[] = {
        { "pin", { 1, 0, 0, 1 } },
        { "fanout", { 0, 1, 0, 1 } },
        { "hop_event", { 0, 0, 1, 1 } },
        { "pin+hop_event", { 1, 0, 1, 1 } },
        { "pin+fanout", { 1, 1, 0, 1 } },
        { "all", { 1, 1, 1, 1 } },
    };
    fc_cuda_ctx *role[4] = { g.a, g.b, g.b, g.a };
    for (size_t i = 0; i < sizeof(sets) / sizeof(sets[0]); i++) {
        rig_clear(&g);
        qw_pcie_stage *st = qw_pcie_stage_create(role, ROW_BYTES, SMALL_BYTES, sets[i].f);
        CHECK(st != NULL, "%s: create", sets[i].name);
        if (!st)
            continue;
        CHECK(!sets[i].f.pin || qw_pcie_stage_pinned(st), "%s: pinned rows", sets[i].name);
        int rc = 0;
        for (int r = 0; r < ROUNDS && rc == 0; r++)
            rc = run_round(&g, st, r);
        CHECK(rc == 0, "%s: device calls", sets[i].name);
        size_t bad = rig_check(&g);
        printf("  %-14s %zu bytes wrong\n", sets[i].name, bad);
        CHECK(bad == 0, "%s: %zu bytes wrong", sets[i].name, bad);
        qw_pcie_stage_free(st);
    }
}

#endif /* USE_CUDA */

int main(void)
{
    printf("qwen35moe PCIe staging tests\n");
    test_flags_parse();
#ifdef USE_CUDA
    const char *gpu = getenv("QW_PCIE_TEST_GPU");
    if (gpu && gpu[0] == '1') {
        const char *red = getenv("PN_PCIE_RED");
        printf("  GPU part%s%s\n", red ? ", red arm " : "", red ? red : "");
        test_gpu_levers();
    } else {
        printf("  GPU part skipped (QW_PCIE_TEST_GPU unset)\n");
    }
#endif
    printf("%s (%d failure%s)\n", failures ? "FAILED" : "ok", failures, failures == 1 ? "" : "s");
    return failures ? 1 : 0;
}
