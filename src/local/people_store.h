/* SPDX-License-Identifier: AGPL-3.0-or-later */
/* Copyright (C) 2026 David (Corey) Herrera */
/**
 * @file   people_store.h
 * @brief  The people of a box (people registry spec §3.1): <state-dir>/people.json,
 *         mode 0600, replaced atomically. Each person has an id ("p" + 8 hex), a
 *         unique handle, a display name, an admin flag with the admin who made them
 *         (`admin_by`, the admin tree), and `may_load`. At least one admin always
 *         exists once a person exists. Rights live here; keys only point at people.
 *
 * @status None
 * @issues None
 * @todo   None
 */

#ifndef PN_LOCAL_PEOPLE_STORE_H
#define PN_LOCAL_PEOPLE_STORE_H

#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif

#define PN_PEOPLE_MAX          64
#define PN_PERSON_ID_MAX       16    /* "p" + 8 hex + NUL fits */
#define PN_PERSON_HANDLE_BYTES 24    /* longest handle, bytes */
#define PN_PERSON_NAME_BYTES   64    /* longest name, bytes */
#define PN_PERSON_HANDLE_MAX   (PN_PERSON_HANDLE_BYTES + 1)
#define PN_PERSON_NAME_MAX     (PN_PERSON_NAME_BYTES + 1)

/* Typed results of the store (0 = success). */
typedef enum {
    PN_PEOPLE_OK             = 0,
    PN_PEOPLE_E_INVALID      = -1,   /* bad argument (e.g. first person not admin) */
    PN_PEOPLE_E_BAD_HANDLE   = -2,   /* handle breaks the rules */
    PN_PEOPLE_E_BAD_NAME     = -3,   /* name breaks the rules */
    PN_PEOPLE_E_HANDLE_TAKEN = -4,   /* another person has it (ASCII case ignored) */
    PN_PEOPLE_E_FULL         = -5,   /* PN_PEOPLE_MAX people */
    PN_PEOPLE_E_NOT_FOUND    = -6,   /* no such person */
    PN_PEOPLE_E_LAST_ADMIN   = -7,   /* would leave the box without an admin */
    PN_PEOPLE_E_ADMIN_ABOVE  = -8,   /* target admin is not below the actor in the tree */
    PN_PEOPLE_E_NOT_ADMIN    = -9,   /* the actor is not an admin */
    PN_PEOPLE_E_IO           = -10,  /* RNG or save failure (memory left unchanged) */
} pn_people_result;

/* One person. `admin_by` is "" for a non-admin and for the root admin. */
typedef struct {
    char      id[PN_PERSON_ID_MAX];
    char      handle[PN_PERSON_HANDLE_MAX];
    char      name[PN_PERSON_NAME_MAX];
    int       admin;
    char      admin_by[PN_PERSON_ID_MAX];
    int       may_load;
    long long created;               /* unix seconds */
} pn_person;

/* A partial update: NULL / -1 fields stay as they are. */
typedef struct {
    const char *name;
    const char *handle;
    int         may_load;            /* 0, 1, or -1 = unchanged */
    int         admin;               /* 0, 1, or -1 = unchanged */
} pn_person_change;

typedef struct pn_people pn_people;

/**
 * Open the store at `path` (people.json). A missing file is an empty store; an
 * unreadable or malformed one (bad rules, duplicate handle, broken admin tree,
 * people but no admin) is an error, never silently emptied. Mode forced to 0600.
 * @return the store, or NULL
 *
 * Thread-safe: yes. Reentrant: yes. Signal-safe: no
 */
pn_people *pn_people_open(const char *path);

/** Free the store. NULL-safe. Thread-safe: no. Reentrant: no. Signal-safe: no */
void pn_people_close(pn_people *pp);

/**
 * Add a person and save. The first person must be an admin (the root). An admin
 * added later records `admin_by`, which must name an existing admin.
 * @param admin_by  the maker when `admin` is set, else ignored; NULL = root
 * @param out       receives the stored row; may be NULL
 * @return PN_PEOPLE_OK or a negative pn_people_result
 *
 * Thread-safe: yes (internal lock). Reentrant: no. Signal-safe: no
 */
int pn_people_add(pn_people *pp, const char *handle, const char *name, int admin,
                  const char *admin_by, int may_load, pn_person *out);

/**
 * The oldest admin (lowest `created`, then file order); when the store is empty,
 * first create the root admin from `handle` / `name` and save.
 * @param created_out  set to 1 when this call made the person, else 0; may be NULL
 * @return PN_PEOPLE_OK or a negative pn_people_result
 *
 * Thread-safe: yes (internal lock). Reentrant: no. Signal-safe: no
 */
int pn_people_owner_or_create(pn_people *pp, const char *handle, const char *name,
                              pn_person *out, int *created_out);

/**
 * Change a person and save; all fields are checked before anything changes.
 * Making someone an admin records `actor` as `admin_by`. Demoting an admin
 * follows the admin tree: `actor` may demote only itself or an admin below it
 * (PN_PEOPLE_E_ADMIN_ABOVE otherwise), never the last admin; the demoted admin's
 * child admins move up (spec §3.1). Name, handle and may_load are not checked
 * against `actor` (the route decides who may ask).
 * @param actor  person asking; must be an admin when `admin` changes
 * @return PN_PEOPLE_OK or a negative pn_people_result
 *
 * Thread-safe: yes (internal lock). Reentrant: no. Signal-safe: no
 */
int pn_people_update(pn_people *pp, const char *actor, const char *id,
                     const pn_person_change *change, pn_person *out);

/**
 * Remove a person and save. `actor` must be an admin; removing an admin follows
 * the same tree and last-admin rules as a demotion, and re-parents its children.
 * @return PN_PEOPLE_OK or a negative pn_people_result
 *
 * Thread-safe: yes (internal lock). Reentrant: no. Signal-safe: no
 */
int pn_people_remove(pn_people *pp, const char *actor, const char *id);

/**
 * Copy a person by id. @return PN_PEOPLE_OK or PN_PEOPLE_E_NOT_FOUND
 *
 * Thread-safe: yes (internal lock). Reentrant: yes. Signal-safe: no
 */
int pn_people_get(pn_people *pp, const char *id, pn_person *out);

/**
 * Copy a person by handle, ignoring ASCII case. @return PN_PEOPLE_OK or
 * PN_PEOPLE_E_NOT_FOUND
 *
 * Thread-safe: yes (internal lock). Reentrant: yes. Signal-safe: no
 */
int pn_people_find_handle(pn_people *pp, const char *handle, pn_person *out);

/**
 * Copy up to `max` people in file order. @return the number copied
 *
 * Thread-safe: yes (internal lock). Reentrant: yes. Signal-safe: no
 */
int pn_people_list(pn_people *pp, pn_person *out, int max);

/** Number of admins. Thread-safe: yes. Reentrant: yes. Signal-safe: no */
int pn_people_admin_count(pn_people *pp);

/**
 * The handle / name rules (Argus box-name rules): spaces and tabs trimmed at
 * both ends, then 1..max_bytes bytes of valid UTF-8 with no ASCII control
 * characters. Too long is refused, never cut.
 * @param out  receives the trimmed text; cap must exceed max_bytes
 * @return 0, or -1 when the text breaks a rule
 *
 * Thread-safe: yes. Reentrant: yes. Signal-safe: yes
 */
int pn_people_name_clean(const char *in, size_t max_bytes, char *out, size_t cap);

#ifdef __cplusplus
}
#endif

#endif /* PN_LOCAL_PEOPLE_STORE_H */
