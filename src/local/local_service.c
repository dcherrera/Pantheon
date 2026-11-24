/* SPDX-License-Identifier: AGPL-3.0-or-later */
/* Copyright (C) 2026 David (Corey) Herrera */
/**
 * @file   local_service.c
 * @brief  Implementation of the appliance wiring (see local_service.h).
 *
 * @status None
 * @issues None
 * @todo   None
 */

#include "local_service.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "../serve/http_server.h"
#include "../serve/serve_command.h"
#include "catalog.h"
#include "../accel/die_manager.h"
#include "curl_runner.h"
#include "hosts_hub.h"
#include "hw_mem_poller.h"
#include "key_migration.h"
#include "key_store.h"
#include "install_worker.h"
#include "invites.h"
#include "loaded_set.h"
#include "local_catalog_routes.h"
#include "local_load.h"
#include "local_auth.h"
#include "local_routes.h"
#include "mdns_responder.h"
#include "pairing.h"
#include "people_store.h"
#include "rate_limit.h"
#include "session_registry.h"

/* TXT for _pantheon._tcp (spec §11.6): build, port, card summary, backend,
 * and whether the box still needs pairing (no device key yet). */
static int local_txt(void *user, char txt[][PN_MDNS_TXT_ITEM], int max)
{
    pn_local_ctx *ctx = user;
    pn_key_info *keys = calloc(PN_KEY_MAX_KEYS, sizeof(*keys));
    int n_keys = ctx->keys && keys ? pn_key_store_list(ctx->keys, keys, PN_KEY_MAX_KEYS) : 0;
    int paired = 0, n = 0;

    for (int i = 0; i < n_keys; i++)
        paired |= keys[i].kind == PN_KEY_KIND_DEVICE;
    free(keys);
    if (max < 6)
        return 0;
    snprintf(txt[n++], PN_MDNS_TXT_ITEM, "txtvers=1");
    snprintf(txt[n++], PN_MDNS_TXT_ITEM, "v=%s", ctx->build);
    snprintf(txt[n++], PN_MDNS_TXT_ITEM, "port=%d", ctx->port);
    snprintf(txt[n++], PN_MDNS_TXT_ITEM, "cards=%s", ctx->card_summary);
    snprintf(txt[n++], PN_MDNS_TXT_ITEM, "backend=%s", ctx->backend);
    snprintf(txt[n++], PN_MDNS_TXT_ITEM, "pair=%s", paired ? "no" : "yes");
    return n;
}

/* Serve hook: the socket is listening. */
static int local_on_listening(mr_registry *reg, const char *host, int port, void *user)
{
    pn_local_ctx *ctx = user;
    pn_host_info hi;
    pn_hw_report hw;

    ctx->reg = reg;
    pn_hw_family family = pn_hw_family_detect();
    if (family != PN_HW_FAMILY_CPU &&
        pn_hw_mem_poller_start(family, pn_dm_count(), pn_hw_report_read_device,
                               PN_HW_POLL_INTERVAL_MS) != 0)
        fprintf(stderr, "local: background device memory reads did not start; each request "
                "reads the devices\n");
    pn_local_snapshot_hardware(ctx, &hi, &hw);
    pn_hw_report_card_summary(&hw, ctx->card_summary, sizeof(ctx->card_summary));
    snprintf(ctx->build, sizeof(ctx->build), "%s", hi.build);
    snprintf(ctx->backend, sizeof(ctx->backend), "%s", hi.backend);
    fprintf(stderr, "local: appliance ready on %s:%d (cards %s, state %s, store %s, "
            "catalog %s)\n", host, port, ctx->card_summary, ctx->state_dir, ctx->store_dir,
            ctx->catalog_url[0] ? ctx->catalog_url : "none");
    pn_local_catalog_fetch_async(ctx);
    pn_local_reload_async(ctx);
    if (!ctx->no_mdns) {
        ctx->mdns = pn_mdns_start(hi.hostname, (uint16_t)port, local_txt, ctx);
        if (!ctx->mdns)
            fprintf(stderr, "local: LAN discovery is OFF (mdns could not start)\n");
    }
    return 0;
}

/* Serve hook: shutdown, before the models are freed. */
static void local_on_shutdown(void *user)
{
    pn_local_ctx *ctx = user;

    pn_mdns_stop(ctx->mdns);
    ctx->mdns = NULL;
    pn_hw_mem_poller_stop();
    pn_install_worker_stop(ctx->installer);
    ctx->installer = NULL;
    pn_pairing_stop(ctx->pairing);
    ctx->pairing = NULL;
    pn_hub_stop(ctx->hub);   /* held waits hear close shutdown */
    if (ctx->registry && pn_reg_save(ctx->registry) != 0)
        fprintf(stderr, "local: sessions.json was not saved at shutdown\n");
}

/* Serve hook: /local/ routes. */
static int local_route(int cfd, mr_registry *reg, const hs_request *req, void *user)
{
    (void)reg;
    return pn_local_routes_dispatch(cfd, user, req);
}

/* People, keys (migrated from v1 before anything is served), limiters, pairing code. */
static int open_access_control(pn_local_ctx *ctx)
{
    char path[PN_LOCAL_PATH_MAX], backup[PN_LOCAL_PATH_MAX];

    if (pn_local_path_join(path, sizeof(path), ctx->state_dir, "people.json") != 0 ||
        !(ctx->people = pn_people_open(path)))
        return -1;
    if (pn_local_path_join(path, sizeof(path), ctx->state_dir, "keys.json") != 0 ||
        pn_local_path_join(backup, sizeof(backup), ctx->state_dir, "keys.json.v1") != 0 ||
        !(ctx->keys = pn_key_store_open(path)) ||
        pn_key_migrate_v1(ctx->keys, ctx->people, backup) != 0)
        return -1;
    ctx->limiter = pn_rate_limiter_new(PN_LOCAL_RATE_PER_S, PN_LOCAL_RATE_BURST);
    ctx->pair_limiter = pn_rate_limiter_new(PN_LOCAL_PAIR_PER_S, PN_LOCAL_PAIR_BURST);
    if (!ctx->limiter || !ctx->pair_limiter)
        return -1;
    if (pn_local_path_join(path, sizeof(path), ctx->state_dir, "pair_code") != 0 ||
        !(ctx->pairing = pn_pairing_start(path, PN_PAIR_DEFAULT_TTL_S, 1)))
        return -1;
    ctx->invites = pn_invites_new(PN_INVITE_TTL_S);
    return ctx->invites ? 0 : -1;
}

/* A positive integer from the environment, else fallback. Only tests set these
 * (shorter host timings); a box uses the spec's 30 s / 90 s / 5 s / 10 s. */
static int env_positive(const char *name, int fallback)
{
    const char *v = getenv(name);
    char *end = NULL;
    long n = v && *v ? strtol(v, &end, 10) : 0;

    return v && *v && !*end && n > 0 && n < 1000000 ? (int)n : fallback;
}

/* A shared person's handle for the hub's `shares` events. */
static int person_handle(const char *person, char *out, size_t cap, void *user)
{
    pn_local_ctx *ctx = user;
    pn_person p;

    if (pn_people_get(ctx->people, person, &p) != PN_PEOPLE_OK)
        return -1;
    snprintf(out, cap, "%s", p.handle);
    return 0;
}

/* sessions.json and the hub that holds the host waits. */
static int open_registry(pn_local_ctx *ctx)
{
    char path[PN_LOCAL_PATH_MAX];
    pn_hub_cfg cfg = {
        .wait_ms = env_positive("PN_LOCAL_HOST_WAIT_MS", PN_HUB_WAIT_MS),
        .sweep_ms = env_positive("PN_LOCAL_HOST_SWEEP_MS", PN_HUB_SWEEP_MS),
        .handle_of = person_handle,
        .user = ctx,
        .relays = { .host_ms = env_positive("PN_LOCAL_RELAY_HOST_MS", PN_HUB_RELAY_HOST_MS) },
    };

    if (pn_local_path_join(path, sizeof(path), ctx->state_dir, "sessions.json") != 0 ||
        !(ctx->registry = pn_reg_open(path)))
        return -1;
    ctx->host_beat_s = (cfg.wait_ms + 999) / 1000;
    ctx->host_disconnect_s = env_positive("PN_LOCAL_HOST_DISCONNECT_S", PN_REG_DISCONNECT_S);
    pn_reg_set_disconnect_s(ctx->registry, ctx->host_disconnect_s);
    ctx->hub = pn_hub_start(ctx->registry, &cfg);
    return ctx->hub ? 0 : -1;
}

/* Catalog, loaded set, install worker. */
static int open_models(pn_local_ctx *ctx)
{
    char path[PN_LOCAL_PATH_MAX];
    pn_download_cfg cfg = { ctx->store_dir, ctx->hf_base_url,
                            ctx->allow_http ? PN_CURL_PROTOCOLS_ANY : PN_CURL_PROTOCOLS_HTTPS,
                            NULL };

    ctx->catalog = pn_catalog_open(ctx->state_dir, NULL, 0);
    if (!ctx->catalog || pn_local_path_join(path, sizeof(path), ctx->state_dir, "loaded.json") != 0)
        return -1;
    ctx->loaded = pn_loaded_set_open(path);
    if (ctx->loaded && pn_local_loads_claim_unowned(ctx) != 0)
        fprintf(stderr, "local: loaded.json not saved after giving old entries to person 1\n");
    ctx->installer = pn_install_worker_start(&cfg);
    return ctx->loaded && ctx->installer ? 0 : -1;
}

int pn_local_service_install(pn_local_ctx *ctx)
{
    pn_serve_local_hooks hooks = {
        .on_listening = local_on_listening,
        .route = local_route,
        .on_shutdown = local_on_shutdown,
        .user = ctx,
    };

    ctx->started = time(NULL);
    if (open_access_control(ctx) != 0) {
        fprintf(stderr, "local: could not open people / keys / pairing in %s\n", ctx->state_dir);
        pn_local_service_release(ctx);
        return -1;
    }
    if (open_registry(ctx) != 0) {
        fprintf(stderr, "local: could not open the session registry in %s\n", ctx->state_dir);
        pn_local_service_release(ctx);
        return -1;
    }
    if (open_models(ctx) != 0) {
        pn_local_service_release(ctx);
        return -1;
    }
    if (!pn_curl_path())
        fprintf(stderr, "local: curl not found on PATH: catalog fetches and model installs "
                        "are refused until it is installed\n");
    hs_set_auth_fn(pn_local_auth_check, ctx);
    pn_serve_set_local_hooks(&hooks);
    return 0;
}

void pn_local_service_release(pn_local_ctx *ctx)
{
    pn_serve_set_local_hooks(NULL);
    hs_set_auth_fn(NULL, NULL);
    pn_pairing_stop(ctx->pairing);
    ctx->pairing = NULL;
    pn_invites_free(ctx->invites);
    ctx->invites = NULL;
    pn_hub_free(ctx->hub);
    ctx->hub = NULL;
    pn_reg_close(ctx->registry);
    ctx->registry = NULL;
    pn_rate_limiter_free(ctx->limiter);
    pn_rate_limiter_free(ctx->pair_limiter);
    ctx->limiter = ctx->pair_limiter = NULL;
    pn_key_store_close(ctx->keys);
    ctx->keys = NULL;
    pn_people_close(ctx->people);
    ctx->people = NULL;
    pn_install_worker_stop(ctx->installer);
    ctx->installer = NULL;
    pn_loaded_set_close(ctx->loaded);
    ctx->loaded = NULL;
    pn_catalog_close(ctx->catalog);
    ctx->catalog = NULL;
}
