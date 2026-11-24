/* SPDX-License-Identifier: AGPL-3.0-or-later */
/* Copyright (C) 2026 David (Corey) Herrera */
/*
 * test_qwen35_ov_plan.c — the overlap's step placement (qwen35moe_prefill_ov_plan.h), NO CARD.
 *
 * A step placed too early reads rows the chain has not finished (a wrong answer nothing
 * reports); a step out of order waits on the wrong grid work or none. Replays every band of a
 * layer (and its tail) for bands 1..80, spans 1..20, pipe depths 0..2, the default offsets and
 * a set of explicit ones, and checks: each job's four stages run once, in order; OUT runs no
 * sooner than `pipe_depth` bands after the job's last band; inside the loop every wait has at
 * least one chain band since the step it waits on; a job's FINISH runs after its DOWN and
 * after the next job's OUT but before its UP (G.2); the jobs cover the bands once, and with
 * more than one job none is under 2 bands (G.3 fold, pn_tt_superband.h).
 * Build with -DPN_OV_PLAN_RED (no pipe-depth floor) or -DPN_TT_SUPERBAND_RED (no fold) to see
 * it fail.
 */
#include "../src/models/qwen35moe_prefill_ov_plan.h"
#include "../src/accel/tt/pn_tt_superband.h"

#include <stdint.h>
#include <stdio.h>
#include <string.h>

static int g_failed;

static void check(int ok, const char *what)
{
    if (!ok) {
        g_failed++;
        if (g_failed < 20)
            fprintf(stderr, "FAIL %s\n", what);
    }
}

static void check_layer(uint32_t bands, uint32_t span, uint32_t depth, const uint32_t *at)
{
    pn_ov_plan plan;
    pn_ov_action due[PN_OV_PLAN_MAX_JOBS * PN_OV_STAGE_COUNT];
    int seen[PN_OV_PLAN_MAX_JOBS][PN_OV_STAGE_COUNT];
    uint32_t when[PN_OV_PLAN_MAX_JOBS][PN_OV_STAGE_COUNT];
    uint32_t order = 0, order_of[PN_OV_PLAN_MAX_JOBS][PN_OV_STAGE_COUNT];
    uint32_t covered = 0;
    char what[128];

    snprintf(what, sizeof(what), "bands %u span %u depth %u at %s", bands, span, depth,
             at ? "given" : "default");
    check(span >= 2u || pn_ov_plan_build(&plan, bands, span, depth, at) != 0, what);
    if (pn_ov_plan_build(&plan, bands, span, depth, at) != 0) {
        check(span < 2u || pn_tt_superband_jobs(bands, span) > PN_OV_PLAN_MAX_JOBS, what);
        return;
    }
    memset(seen, 0, sizeof(seen));
    for (uint32_t band = 0; band <= bands; band++) {
        const size_t n = pn_ov_plan_due(&plan, band, due, sizeof(due) / sizeof(due[0]));

        for (size_t i = 0; i < n; i++) {
            const uint32_t k = due[i].job, s = (uint32_t)due[i].stage;

            check(k < plan.jobs && s < PN_OV_STAGE_COUNT && !seen[k][s], what);
            if (k >= plan.jobs || s >= PN_OV_STAGE_COUNT)
                continue;
            seen[k][s] = 1;
            when[k][s] = band;
            order_of[k][s] = order++;
        }
    }
    for (uint32_t k = 0; k < plan.jobs; k++) {
        uint32_t first, count;

        pn_ov_plan_job_bands(&plan, k, &first, &count);
        check(first == covered && count > 0u, what);
        check(plan.jobs == 1u || count >= 2u, what);
        covered += count;
        for (uint32_t s = 0; s < PN_OV_STAGE_COUNT; s++)
            check(seen[k][s], what);
        for (uint32_t s = 1; s < PN_OV_STAGE_COUNT; s++) {
            check(order_of[k][s] > order_of[k][s - 1u], what);
            /* A wait inside the loop has at least one chain band under the step it waits on. */
            if (when[k][s] < bands && s != PN_OV_STAGE_FINISH)
                check(when[k][s] > when[k][s - 1u], what);
        }
        /* OUT reads `gated` of the job's last band: its gate is issued `depth` bands later. */
        check(when[k][PN_OV_STAGE_OUT] >= first + count - 1u + depth ||
                  when[k][PN_OV_STAGE_OUT] == bands,
              what);
        if (k + 1u < plan.jobs) {
            check(order_of[k][PN_OV_STAGE_FINISH] > order_of[k + 1u][PN_OV_STAGE_OUT], what);
            check(order_of[k][PN_OV_STAGE_FINISH] < order_of[k + 1u][PN_OV_STAGE_UP], what);
            check(order_of[k][PN_OV_STAGE_DOWN] < order_of[k + 1u][PN_OV_STAGE_OUT], what);
            check(when[k][PN_OV_STAGE_FINISH] > when[k][PN_OV_STAGE_DOWN] ||
                      when[k][PN_OV_STAGE_FINISH] == bands,
                  what);
        }
    }
    check(covered == bands, what);
}

int main(void)
{
    static const uint32_t given[][3] = { { 0u, 3u, 8u }, { 2u, 5u, 10u }, { 1u, 2u, 3u },
                                         { 3u, 9u, 20u } };
    uint32_t at[3];

    for (uint32_t bands = 1u; bands <= 80u; bands++)
        for (uint32_t span = 1u; span <= 20u; span++)
            for (uint32_t depth = 0u; depth <= 2u; depth++) {
                check_layer(bands, span, depth, NULL);
                for (uint32_t i = 0; i < sizeof(given) / sizeof(given[0]); i++)
                    check_layer(bands, span, depth, given[i]);
            }
    pn_ov_plan_default_at(13u, 2u, at);
    check(at[0] == 2u && at[1] == 5u && at[2] == 10u, "defaults 13 bands, PIPE 2: 2,5,10");
    check(pn_ov_plan_parse_at("2,5,10", at) == 0 && at[2] == 10u, "parse 2,5,10");
    check(pn_ov_plan_parse_at("5,2,10", at) != 0, "parse: not increasing");
    check(pn_ov_plan_parse_at("2,5", at) != 0, "parse: two values");
    check(pn_ov_plan_parse_at("2,5,10x", at) != 0, "parse: trailing junk");
    check(pn_ov_plan_parse_at("", at) != 0 && pn_ov_plan_parse_at(NULL, at) != 0, "parse: unset");
    printf("test_qwen35_ov_plan: %d failed\n", g_failed);
    return g_failed ? 1 : 0;
}
