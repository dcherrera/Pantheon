/* SPDX-License-Identifier: AGPL-3.0-or-later */
/* Copyright (C) 2026 David (Corey) Herrera */
/**
 * @file   qwen35moe_step_ledger.c
 * @brief  Implementation of the decode step ledger (see the header for the line format).
 *
 * @status Instrument only.
 * @issues Single writer (the model's GPU turn serialises steps).
 * @todo   None
 */
#ifdef USE_CUDA

#include "qwen35moe_step_ledger.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

static const char *const ledger_phase_name[QW_LH_N] = {
    "setup0", "launch0", "mirror0", "wait0", "hop", "setup1", "launch1", "mirror1",
    "head_issue", "pres_wait", "pres_copy", "tail",
};

typedef struct {
    FILE *out;
    int state;                       /* -1 unread, 0 off, 1 on */
    char inject_point[4][16];
    long inject_us[4];
    int n_inject;
    fc_event *ev[QW_LG_N];
    fc_cuda_ctx *ev_ctx[QW_LG_N];
    int recorded[QW_LG_N];
    double t0, last, prev_t0, last_flush;
    double wall_t0;                  /* CLOCK_REALTIME at t0, to line steps up with outside logs */
    double ph[QW_LH_N];
    int nslots, pd, pmin, pmax;
    char mode[2];
    fc_call_census c_begin, c_prev_end;
    int have_prev_end;
    int in_step;
    unsigned long steps;
} step_ledger;

static step_ledger ledger = { .state = -1 };

static double ledger_now(void)
{
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (double)ts.tv_sec + (double)ts.tv_nsec * 1e-9;
}

/** Parse QWEN35_STEP_LEDGER_INJECT=<point>:<us>[,<point>:<us>...] (up to 4). */
static void ledger_read_inject(void)
{
    const char *e = getenv("QWEN35_STEP_LEDGER_INJECT");
    while (e && *e && ledger.n_inject < 4) {
        const char *colon = strchr(e, ':');
        if (!colon || colon == e || (size_t)(colon - e) >= sizeof(ledger.inject_point[0]))
            return;
        int i = ledger.n_inject++;
        memcpy(ledger.inject_point[i], e, (size_t)(colon - e));
        ledger.inject_point[i][colon - e] = '\0';
        char *end = NULL;
        ledger.inject_us[i] = strtol(colon + 1, &end, 10);
        fprintf(stderr, "[ledger] injecting %ld us of host sleep at '%s'\n", ledger.inject_us[i],
                ledger.inject_point[i]);
        e = (end && *end == ',') ? end + 1 : NULL;
    }
}

int qw_ledger_on(void)
{
    if (__builtin_expect(ledger.state >= 0, 1))
        return ledger.state;
    const char *path = getenv("QWEN35_STEP_LEDGER");
    ledger.state = 0;
    if (!path || !*path)
        return 0;
    ledger.out = fopen(path, "a");
    if (!ledger.out) {
        fprintf(stderr, "[ledger] cannot open %s; step ledger off\n", path);
        return 0;
    }
    setvbuf(ledger.out, NULL, _IOFBF, 1 << 16);
    ledger_read_inject();
    fc_call_census_enable(1);
    ledger.state = 1;
    fprintf(stderr, "[ledger] decode step ledger -> %s\n", path);
    return 1;
}

void qw_ledger_step_begin(int nslots, const int *pos, int pd)
{
    ledger.in_step = 1;
    ledger.t0 = ledger.last = ledger_now();
    struct timespec rt;
    clock_gettime(CLOCK_REALTIME, &rt);
    ledger.wall_t0 = (double)rt.tv_sec + (double)rt.tv_nsec * 1e-9;
    memset(ledger.ph, 0, sizeof(ledger.ph));
    memset(ledger.recorded, 0, sizeof(ledger.recorded));
    ledger.mode[0] = ledger.mode[1] = '-';
    ledger.nslots = nslots;
    ledger.pd = pd;
    ledger.pmin = ledger.pmax = nslots > 0 ? pos[0] : 0;
    for (int u = 1; u < nslots; u++) {
        if (pos[u] < ledger.pmin) ledger.pmin = pos[u];
        if (pos[u] > ledger.pmax) ledger.pmax = pos[u];
    }
    fc_call_census_read(&ledger.c_begin);
}

void qw_ledger_lap(qw_ledger_phase phase)
{
    if (!ledger.in_step)
        return;
    const double now = ledger_now();
    ledger.ph[phase] += now - ledger.last;
    ledger.last = now;
}

void qw_ledger_gpu(qw_ledger_point pt, fc_cuda_ctx *ctx)
{
    if (!ledger.in_step)
        return;
    if (ledger.ev[pt] && ledger.ev_ctx[pt] != ctx) {
        fc_event_free(ledger.ev[pt]);
        ledger.ev[pt] = NULL;
    }
    if (!ledger.ev[pt]) {
        ledger.ev[pt] = fc_tevent_create_ctx(ctx);
        ledger.ev_ctx[pt] = ctx;
    }
    ledger.recorded[pt] = ledger.ev[pt] && fc_tevent_record_ctx(ctx, ledger.ev[pt]) == 0;
}

void qw_ledger_stage_mode(int s, char mode)
{
    if (s >= 0 && s < 2 && ledger.mode[s] == '-')   /* a capture's replay stays 'C' */
        ledger.mode[s] = mode;
}

void qw_ledger_graph_census(int s, int nslots, const fc_graph *gr)
{
    if (!gr || !ledger.out)
        return;
    fprintf(ledger.out, "#graph\tstage=%d\tn=%d\tnodes=%zu\tkernel=%zu\tmemcpy=%zu\tmemset=%zu"
            "\tev_wait=%zu\tev_record=%zu\n", s, nslots, fc_graph_node_count(gr, -1),
            fc_graph_node_count(gr, 0), fc_graph_node_count(gr, 1), fc_graph_node_count(gr, 2),
            fc_graph_node_count(gr, 6), fc_graph_node_count(gr, 7));
}

void qw_ledger_inject(const char *point)
{
    if (!ledger.in_step)
        return;
    for (int i = 0; i < ledger.n_inject; i++) {
        if (ledger.inject_us[i] <= 0 || strcmp(point, ledger.inject_point[i]) != 0)
            continue;
        struct timespec ts = { ledger.inject_us[i] / 1000000,
                               (ledger.inject_us[i] % 1000000) * 1000 };
        nanosleep(&ts, NULL);
    }
}

/** ms between two recorded points of one stream; -1 when either is missing. */
static double ledger_span_ms(qw_ledger_point a, qw_ledger_point b)
{
    float ms = -1.0f;
    if (!ledger.recorded[a] || !ledger.recorded[b])
        return -1.0;
    fc_event_sync(ledger.ev[b]);
    if (fc_tevent_elapsed_ms(ledger.ev[a], ledger.ev[b], &ms) != 0)
        return -1.0;
    return ms;
}

/** Census fields as key=value, `tag` prefixed (in-step "i", off-step "o"). */
static void ledger_print_census(const char *tag, const fc_call_census *a, const fc_call_census *b)
{
    fprintf(ledger.out, "\t%sL=%llu\t%sG=%llu\t%sS=%llu\t%sHs=%llu\t%sHa=%llu\t%sHb=%llu"
            "\t%sDs=%llu\t%sDa=%llu\t%sDb=%llu\t%sDD=%llu\t%sDDb=%llu\t%sP=%llu\t%sPb=%llu",
            tag, b->launches - a->launches, tag, b->graph_launches - a->graph_launches,
            tag, b->syncs - a->syncs, tag, b->h2d_sync - a->h2d_sync,
            tag, b->h2d_async - a->h2d_async, tag, b->h2d_bytes - a->h2d_bytes,
            tag, b->d2h_sync - a->d2h_sync, tag, b->d2h_async - a->d2h_async,
            tag, b->d2h_bytes - a->d2h_bytes, tag, b->d2d - a->d2d,
            tag, b->d2d_bytes - a->d2d_bytes, tag, b->peer - a->peer,
            tag, b->peer_bytes - a->peer_bytes);
}

void qw_ledger_step_end(int rc)
{
    fc_call_census c_end;
    if (!ledger.in_step)
        return;
    fc_call_census_read(&c_end);
    qw_ledger_lap(QW_LH_TAIL);
    const double t_end = ledger.last;
    if (rc == 0 && ledger.out) {
        const double period = ledger.prev_t0 > 0.0 ? 1000.0 * (ledger.t0 - ledger.prev_t0) : -1.0;
        fprintf(ledger.out, "ts=%.3f\tn=%d\tpd=%d\tpmin=%d\tpmax=%d\tgraph=%c%c\twall=%.3f\tperiod=%.3f"
                "\ts0=%.3f\ts1=%.3f\thead=%.3f", ledger.wall_t0, ledger.nslots, ledger.pd, ledger.pmin,
                ledger.pmax, ledger.mode[0], ledger.mode[1], 1000.0 * (t_end - ledger.t0),
                period, ledger_span_ms(QW_LG_S0_BEG, QW_LG_S0_END),
                ledger_span_ms(QW_LG_S1_BEG, QW_LG_S1_END),
                ledger_span_ms(QW_LG_H_BEG, QW_LG_H_END));
        for (int p = 0; p < QW_LH_N; p++)
            fprintf(ledger.out, "\t%s=%.3f", ledger_phase_name[p], 1000.0 * ledger.ph[p]);
        ledger_print_census("i", &ledger.c_begin, &c_end);
        if (ledger.have_prev_end)
            ledger_print_census("o", &ledger.c_prev_end, &ledger.c_begin);
        fputc('\n', ledger.out);
        if (++ledger.steps % 256 == 0 || t_end - ledger.last_flush > 2.0) {
            fflush(ledger.out);
            ledger.last_flush = t_end;
        }
    }
    /* A failed step breaks the period chain: the next step starts a new one. */
    ledger.prev_t0 = rc == 0 ? ledger.t0 : 0.0;
    fc_call_census_read(&ledger.c_prev_end);
    ledger.have_prev_end = rc == 0;
    ledger.in_step = 0;
}

#endif /* USE_CUDA */
