/* SPDX-License-Identifier: AGPL-3.0-or-later */
/* Copyright (C) 2026 David (Corey) Herrera */
/**
 * @file   local_routes.c
 * @brief  Implementation of the /local/ route table (see local_routes.h). Each
 *         family of routes lives in its own file; this one dispatches and serves
 *         GET /local/hardware.
 *
 * @status None
 * @issues None
 * @todo   None
 */

#include "local_routes.h"

#include <stdio.h>
#include <string.h>

#include "../serve/json_writer.h"
#include "host_info.h"
#include "hw_mem_poller.h"
#include "hw_report.h"
#include "catalog.h"
#include "local_catalog_routes.h"
#include "local_host_routes.h"
#include "local_key_routes.h"
#include "local_model_routes.h"
#include "local_people_routes.h"
#include "local_relay_routes.h"
#include "local_seats_routes.h"
#include "local_session_routes.h"
#include "local_uninstall.h"

/* Snapshot the box: host facts + the card tree. */
void pn_local_snapshot_hardware(pn_local_ctx *ctx, pn_host_info *host, pn_hw_report *hw)
{
    mr_info models[MR_MAX_MODELS];
    int n = ctx->reg ? mr_snapshot(ctx->reg, models, MR_MAX_MODELS) : 0;

    pn_host_info_collect(ctx->store_dir, ctx->started, host);
    pn_hw_report_build(pn_hw_family_detect(), models, n, host->ram_total, hw);
}

void pn_local_snapshot_hardware_fresh(pn_local_ctx *ctx, pn_host_info *host, pn_hw_report *hw)
{
    if (pn_hw_mem_poller_running() &&
        pn_hw_mem_poller_ensure_fresh(PN_HW_FIT_MAX_AGE_MS, PN_HW_FIT_WAIT_MS) != 0)
        fprintf(stderr, "local: no fresh device memory reading within %d ms; using the last "
                "one\n", PN_HW_FIT_WAIT_MS);
    pn_local_snapshot_hardware(ctx, host, hw);
}

/* The oldest unit reading's age (the snapshot's age), 0 when nothing was read. */
static uint64_t snapshot_age_ms(const pn_hw_report *hw)
{
    uint64_t age = 0;

    for (int i = 0; i < hw->n_units; i++)
        if (hw->units[i].mem_live && hw->units[i].mem_age_ms > age)
            age = hw->units[i].mem_age_ms;
    return age;
}

long long pn_local_catalog_version(const pn_local_ctx *ctx)
{
    return pn_catalog_version(ctx->catalog);
}

/* GET /local/hardware */
static void handle_hardware(int cfd, pn_local_ctx *ctx)
{
    pn_host_info host;
    pn_hw_report hw;
    json_writer jw;

    pn_local_snapshot_hardware(ctx, &host, &hw);
    if (!jw_init(&jw)) {
        hs_send_error(cfd, 500, "server_error", "oom");
        return;
    }
    jw_raw(&jw, "{");
    pn_host_info_write_json(&jw, &host);
    jw_raw(&jw, ",");
    pn_hw_report_write_cards(&jw, &hw);
    jw_printf(&jw, ",\"mem_age_ms\":%llu,\"mem_poll_ms\":%d",
              (unsigned long long)snapshot_age_ms(&hw),
              pn_hw_mem_poller_running() ? PN_HW_POLL_INTERVAL_MS : 0);
    long long cv = pn_local_catalog_version(ctx);
    if (cv >= 0)
        jw_printf(&jw, ",\"catalog_version\":%lld}", cv);
    else
        jw_raw(&jw, ",\"catalog_version\":null}");
    hs_send_json(cfd, 200, jw.data);
    jw_free(&jw);
}

static int route_is(const hs_request *req, const char *method, const char *path)
{
    return strcmp(req->method, method) == 0 && strcmp(req->path, path) == 0;
}

int pn_local_routes_dispatch(int client_fd, pn_local_ctx *ctx, const hs_request *req)
{
    int handled = 0;

    if (pn_local_model_routes(client_fd, ctx, req))
        return 1;
    if (strncmp(req->path, "/local/", 7) != 0)
        return 0;
    if ((handled = pn_local_host_routes(client_fd, ctx, req)) != 0 ||
        (handled = pn_local_relay_routes(client_fd, ctx, req)) != 0)
        return handled;
    if (pn_local_session_routes(client_fd, ctx, req) || pn_local_key_routes(client_fd, ctx, req) ||
        pn_local_people_routes(client_fd, ctx, req) || pn_local_seats_routes(client_fd, ctx, req) ||
        pn_local_catalog_routes(client_fd, ctx, req) ||
        pn_local_uninstall_route(client_fd, ctx, req))
        return 1;
    if (route_is(req, "GET", "/local/hardware"))
        handle_hardware(client_fd, ctx);
    else
        hs_send_error(client_fd, 404, "not_found", "unknown /local route");
    return 1;
}
