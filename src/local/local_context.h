/* SPDX-License-Identifier: AGPL-3.0-or-later */
/* Copyright (C) 2026 David (Corey) Herrera */
/**
 * @file   local_context.h
 * @brief  The appliance's process-wide state: configured paths and the handles of
 *         each subsystem (people, keys, pairing, discovery, catalog, installer, loaded
 *         set, session registry and its hub). One instance lives for the process; the serve hooks receive it as
 *         their `user` pointer.
 *
 * @status None
 * @issues None
 * @todo   None
 */

#ifndef PN_LOCAL_CONTEXT_H
#define PN_LOCAL_CONTEXT_H

#include <time.h>

#include "../serve/model_registry.h"
#include "local_file_io.h"

#ifdef __cplusplus
extern "C" {
#endif

/* Default port of `pantheon local` (spec §4.1). Clear of the fleet's 8201-8210,
 * the TT dev serve's 8301, the gateway's 8080 and the registry's 8100. */
#define PN_LOCAL_DEFAULT_PORT 8400
#define PN_LOCAL_DEFAULT_HOST "0.0.0.0"
#define PN_LOCAL_URL_MAX      1024

/* The signed catalog a box uses when started without --catalog-url (H2o); its
 * signature is `<url>.minisig`. `--catalog-url <u>` overrides it (mirrors,
 * tests); `--catalog-url ''` means no catalog fetch at all. */
#define PN_LOCAL_DEFAULT_CATALOG_URL \
    "https://huggingface.co/TeamIDE/pantheon-catalog/resolve/main/catalog.json"

typedef struct pn_key_store         pn_key_store;
typedef struct pn_people            pn_people;
typedef struct pn_pairing           pn_pairing;
typedef struct pn_invites           pn_invites;
typedef struct pn_rate_limiter      pn_rate_limiter;
typedef struct pn_mdns_responder    pn_mdns_responder;
typedef struct pn_catalog           pn_catalog;
typedef struct pn_install_worker    pn_install_worker;
typedef struct pn_loaded_set        pn_loaded_set;
typedef struct pn_reg               pn_reg;
typedef struct pn_hub               pn_hub;

typedef struct {
    char               state_dir[PN_LOCAL_PATH_MAX];  /* keys.json, pair_code, ... */
    char               store_dir[PN_LOCAL_PATH_MAX];  /* downloaded model files */
    char               catalog_url[PN_LOCAL_URL_MAX]; /* manifest URL (default
                                                       * PN_LOCAL_DEFAULT_CATALOG_URL); "" = none */
    char               host[64];
    int                port;
    char               hf_base_url[256]; /* --hf-base-url; "" = https://huggingface.co */
    int                no_mdns;          /* --no-mdns (tests) */
    int                allow_http;       /* --allow-http: curl may use http/file
                                          * (local mirrors, tests); integrity is
                                          * still the signature + sha256 */
    time_t             started;
    char               card_summary[32]; /* "k80x8", set when listening */
    char               build[64];        /* Pantheon build commit */
    char               backend[16];
    mr_registry       *reg;              /* set when the serve is listening */
    pn_people         *people;           /* people.json (people registry §3.1) */
    pn_key_store      *keys;
    pn_pairing        *pairing;
    pn_invites        *invites;          /* one-time invite codes (memory only) */
    pn_rate_limiter   *limiter;          /* per key */
    pn_rate_limiter   *pair_limiter;     /* box-wide, POST /local/pair */
    pn_mdns_responder *mdns;
    pn_catalog        *catalog;
    pn_install_worker *installer;
    pn_loaded_set     *loaded;
    pn_reg            *registry;         /* sessions.json (people registry §3.4) */
    pn_hub            *hub;              /* held host waits (§6.3) */
    int                host_beat_s;      /* longest held wait, in POST /local/hosts */
    int                host_disconnect_s;
} pn_local_ctx;

#ifdef __cplusplus
}
#endif

#endif /* PN_LOCAL_CONTEXT_H */
