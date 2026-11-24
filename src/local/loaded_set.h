/* SPDX-License-Identifier: AGPL-3.0-or-later */
/* Copyright (C) 2026 David (Corey) Herrera */
/**
 * @file   loaded_set.h
 * @brief  The catalog models the appliance has loaded, remembered across
 *         restarts (spec §4.4): `<state-dir>/loaded.json`, written on every load
 *         and unload, in load order. Each entry carries everything needed to
 *         reload it without the catalog (file, units, hardware request, load
 *         settings, sampling defaults), so an offline boot still comes back,
 *         and the person who loaded it (people registry spec §4.7).
 *
 * @status None
 * @issues None
 * @todo   None
 */

#ifndef PN_LOCAL_LOADED_SET_H
#define PN_LOCAL_LOADED_SET_H

#include <stdbool.h>

#include "../serve/sampling_defaults.h"
#include "catalog_manifest.h"
#include "unit_place.h"

#ifdef __cplusplus
extern "C" {
#endif

#define PN_LOADED_MAX 16
#define PN_LOADED_PERSON_MAX 16   /* a person id: "p" + 8 hex + NUL fits */

typedef struct {
    char              catalog_id[PN_CAT_ID_MAX];   /* also the served model name */
    int               replica;                     /* 1, 2, ...: copies of one model on
                                                    * different units, one served name */
    char              profile_id[PN_CAT_ID_MAX];
    char              path[1024];                  /* weights file in the store */
    int               units[PN_UNIT_PLACE_MAX];    /* die ids; none for cpu */
    int               n_units;
    pn_unit_request   hardware;                    /* to re-place if units are taken */
    int               ctx_cap, max_sessions, max_concurrent, reasoning_budget;
    int               reasoning_reserve;   /* load.reasoning_reserve; 0 = serve default */
    bool              mtp;
    int               mtp_drafts;                  /* the profile's load.mtp_drafts; 0 = the
                                                    * build's default (written only when > 0) */
    char              kv_type[8];                  /* the profile's options.kv_type ("q4",
                                                    * "fp32", "bf16", a p150 "k8v4" ...);
                                                    * "" = serve default */
    char              kv_long[8];                  /* the profile's options.kv_long (p150
                                                    * "k4h"); "" = none (written only when set) */
    pn_model_sampling sampling;
    char              loaded_by[PN_LOADED_PERSON_MAX];   /* person id; "" = not yet
                                                          * claimed (an entry from
                                                          * before people) */
} pn_loaded_entry;

typedef struct pn_loaded_set pn_loaded_set;

/**
 * Open (missing file = empty set; malformed = NULL, never silently emptied).
 *
 * Thread-safe: yes. Reentrant: yes. Signal-safe: no
 */
pn_loaded_set *pn_loaded_set_open(const char *path);

/** NULL-safe. Thread-safe: no. Reentrant: no. Signal-safe: no */
void pn_loaded_set_close(pn_loaded_set *s);

/**
 * Add (or replace, by catalog id + replica) an entry at the end and save.
 * @return 0, or -1 (full or write failure)
 *
 * Thread-safe: yes (internal lock). Reentrant: no. Signal-safe: no
 */
int pn_loaded_set_add(pn_loaded_set *s, const pn_loaded_entry *e);

/**
 * Remove one replica of catalog_id (replica 0 = every replica) and save.
 * @return 0 if removed, 1 if it was not there, -1 on a write failure
 *
 * Thread-safe: yes (internal lock). Reentrant: no. Signal-safe: no
 */
int pn_loaded_set_remove(pn_loaded_set *s, const char *catalog_id, int replica);

/**
 * The lowest replica number catalog_id does not use yet (1 for a new model).
 *
 * Thread-safe: yes (internal lock). Reentrant: yes. Signal-safe: no
 */
int pn_loaded_set_next_replica(pn_loaded_set *s, const char *catalog_id);

/**
 * Give every entry with no `loaded_by` (written before people existed) to
 * `person` (person 1, spec §4.7) and save when any changed.
 * @return how many entries were claimed, or -1 on a write failure
 *
 * Thread-safe: yes (internal lock). Reentrant: no. Signal-safe: no
 */
int pn_loaded_set_claim_unowned(pn_loaded_set *s, const char *person);

/**
 * Copy the entries in load order. @return how many
 *
 * Thread-safe: yes (internal lock). Reentrant: yes. Signal-safe: no
 */
int pn_loaded_set_list(pn_loaded_set *s, pn_loaded_entry *out, int max);

#ifdef __cplusplus
}
#endif

#endif /* PN_LOCAL_LOADED_SET_H */
