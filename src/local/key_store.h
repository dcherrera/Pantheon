/* SPDX-License-Identifier: AGPL-3.0-or-later */
/* Copyright (C) 2026 David (Corey) Herrera */
/**
 * @file   key_store.h
 * @brief  Client keys of the appliance (people registry spec §3.2). A key is 32
 *         random bytes shown to the client once as `pnk_<64 hex>`; the box keeps
 *         only its SHA-256, with an id, a kind (device / inference), the person
 *         it belongs to, the device it was paired from, a label and the creation
 *         time. File: <state-dir>/keys.json v2, mode 0600, replaced atomically;
 *         a v1 file is read and migrated (see key_migration.h).
 *
 * @status None
 * @issues None
 * @todo   None
 */

#ifndef PN_LOCAL_KEY_STORE_H
#define PN_LOCAL_KEY_STORE_H

#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif

#define PN_KEY_MAX_KEYS        256    /* an office: 20 people x 3 devices + scripts */
#define PN_KEY_ID_MAX          16     /* "k" + 8 hex + NUL fits */
#define PN_KEY_LABEL_MAX       64
#define PN_KEY_SECRET_MAX      72     /* "pnk_" + 64 hex + NUL */
#define PN_KEY_PERSON_MAX      16     /* a person id: "p" + 8 hex + NUL fits */
#define PN_KEY_DEVICE_ID_MAX   18     /* "d" + 16 hex + NUL */
#define PN_KEY_DEVICE_NAME_MAX 65     /* 1-64 bytes, the people name rules */

/* What a key is. Rights come from its person, never from the key. */
typedef enum {
    PN_KEY_KIND_NONE = 0,
    PN_KEY_KIND_INFERENCE,       /* minted by a person: /v1 and /session routes */
    PN_KEY_KIND_DEVICE,          /* from pairing: one Argus install */
} pn_key_kind;

/* Typed results (0 = success). */
typedef enum {
    PN_KEY_OK                = 0,
    PN_KEY_E_INVALID         = -1,  /* bad argument (kind, id format, device name) */
    PN_KEY_E_NOT_FOUND       = -2,  /* no key with that id */
    PN_KEY_E_FULL            = -3,  /* PN_KEY_MAX_KEYS keys */
    PN_KEY_E_IO              = -4,  /* RNG or save failure (memory left unchanged) */
    PN_KEY_E_DEVICE_MISMATCH = -5,  /* the key already claimed another device_id */
} pn_key_result;

/* What the box knows about a key (never the secret). */
typedef struct {
    char        id[PN_KEY_ID_MAX];
    pn_key_kind kind;
    char        person[PN_KEY_PERSON_MAX];
    char        device_id[PN_KEY_DEVICE_ID_MAX];     /* device keys; "" until claimed */
    char        device_name[PN_KEY_DEVICE_NAME_MAX]; /* device keys */
    char        label[PN_KEY_LABEL_MAX];
    long long   created;                             /* unix seconds */
} pn_key_info;

typedef struct pn_key_store pn_key_store;

/**
 * Open the store at `path` (keys.json, v1 or v2). A missing file is an empty v2
 * store; an unreadable or malformed one (or a newer version) is an error, never
 * silently emptied. A v1 file loads with every key's person empty: admin keys as
 * device keys (device_name = label), inference keys as inference keys; nothing
 * verifies until pn_key_store_migrate_v1. The file mode is forced to 0600.
 * @return the store, or NULL
 *
 * Thread-safe: yes
 * Reentrant:   yes
 * Signal-safe: no
 */
pn_key_store *pn_key_store_open(const char *path);

/** Free the store. NULL-safe. Thread-safe: no. Reentrant: no. Signal-safe: no */
void pn_key_store_close(pn_key_store *ks);

/**
 * 1 when the store was read from a v1 file and is not migrated yet, else 0.
 *
 * Thread-safe: yes (internal lock). Reentrant: yes. Signal-safe: no
 */
int pn_key_store_is_v1(pn_key_store *ks);

/** Number of keys. Thread-safe: yes (internal lock). Reentrant: yes. Signal-safe: no */
int pn_key_store_count(pn_key_store *ks);

/**
 * Finish a v1 → v2 migration: every key gets `person`, the v1 file is copied to
 * `backup_path` (0600), then keys.json is written as v2.
 * @return PN_KEY_OK, PN_KEY_E_INVALID (not v1), or PN_KEY_E_IO
 *
 * Thread-safe: yes (internal lock). Reentrant: no. Signal-safe: no
 */
int pn_key_store_migrate_v1(pn_key_store *ks, const char *person, const char *backup_path);

/**
 * Mint a device key for `person` (pairing): random secret, store its hash, save.
 * @param device_id    "d" + 16 hex, or NULL / "" (claimed later)
 * @param device_name  1-64 bytes (people name rules)
 * @param secret_out   receives "pnk_<64 hex>" — the only time it exists in full
 * @param info_out     receives the stored row; may be NULL
 * @return PN_KEY_OK or a negative pn_key_result
 *
 * Thread-safe: yes (internal lock). Reentrant: no. Signal-safe: no
 */
int pn_key_store_mint_device(pn_key_store *ks, const char *person, const char *device_id,
                             const char *device_name, const char *label,
                             char secret_out[PN_KEY_SECRET_MAX], pn_key_info *info_out);

/**
 * Mint an inference key for `person`; as pn_key_store_mint_device otherwise.
 *
 * Thread-safe: yes (internal lock). Reentrant: no. Signal-safe: no
 */
int pn_key_store_mint_inference(pn_key_store *ks, const char *person, const char *label,
                                char secret_out[PN_KEY_SECRET_MAX], pn_key_info *info_out);

/**
 * Look up a presented secret (constant-time hash compare).
 * @return 0 and *info_out when known (and migrated), -1 otherwise
 *
 * Thread-safe: yes (internal lock)
 * Reentrant:   yes
 * Signal-safe: no
 */
int pn_key_store_verify(pn_key_store *ks, const char *secret, pn_key_info *info_out);

/**
 * Copy a key's row by id. @return PN_KEY_OK or PN_KEY_E_NOT_FOUND
 *
 * Thread-safe: yes (internal lock). Reentrant: yes. Signal-safe: no
 */
int pn_key_store_get(pn_key_store *ks, const char *id, pn_key_info *info_out);

/**
 * Claim or rename a device key and save. `device_id` is set only when the key
 * has none or the same one; a different one is PN_KEY_E_DEVICE_MISMATCH.
 * @param device_id    "d" + 16 hex, or NULL to leave it
 * @param device_name  new name, or NULL to leave it
 * @return PN_KEY_OK or a negative pn_key_result
 *
 * Thread-safe: yes (internal lock). Reentrant: no. Signal-safe: no
 */
int pn_key_store_set_device(pn_key_store *ks, const char *id, const char *device_id,
                            const char *device_name, pn_key_info *info_out);

/**
 * Revoke a key by id and save.
 * @return PN_KEY_OK, PN_KEY_E_NOT_FOUND, or PN_KEY_E_IO
 *
 * Thread-safe: yes (internal lock)
 * Reentrant:   no
 * Signal-safe: no
 */
int pn_key_store_revoke(pn_key_store *ks, const char *id);

/**
 * Revoke every key of `person` and save once.
 * @return the number revoked (0 or more), or PN_KEY_E_IO
 *
 * Thread-safe: yes (internal lock). Reentrant: no. Signal-safe: no
 */
int pn_key_store_revoke_person(pn_key_store *ks, const char *person);

/**
 * Copy up to `max` rows. @return the number copied
 *
 * Thread-safe: yes (internal lock)
 * Reentrant:   yes
 * Signal-safe: no
 */
int pn_key_store_list(pn_key_store *ks, pn_key_info *out, int max);

/**
 * Copy up to `max` rows of one person. @return the number copied
 *
 * Thread-safe: yes (internal lock). Reentrant: yes. Signal-safe: no
 */
int pn_key_store_list_person(pn_key_store *ks, const char *person, pn_key_info *out, int max);

/** "device" / "inference" / "none". Thread-safe: yes. Reentrant: yes. Signal-safe: yes */
const char *pn_key_kind_name(pn_key_kind kind);

/** 1 when `s` is "d" + 16 lowercase hex. Thread-safe: yes. Reentrant: yes. Signal-safe: yes */
int pn_key_device_id_valid(const char *s);

#ifdef __cplusplus
}
#endif

#endif /* PN_LOCAL_KEY_STORE_H */
