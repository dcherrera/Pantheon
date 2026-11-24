/* SPDX-License-Identifier: AGPL-3.0-or-later */
/* Copyright (C) 2026 David (Corey) Herrera */
/**
 * @file   catalog.h
 * @brief  The accepted catalog (spec §5, §6): fetch the manifest and its
 *         .minisig with curl, verify the signature with the compiled-in keys,
 *         refuse a version lower than the last accepted one
 *         (<state-dir>/catalog_version), parse, and keep it (plus a verified copy
 *         on disk so the box works offline after a reboot).
 *
 * @status None
 * @issues None
 * @todo   None
 */

#ifndef PN_LOCAL_CATALOG_H
#define PN_LOCAL_CATALOG_H

#include <stddef.h>

#include "catalog_manifest.h"
#include "minisign_verify.h"

#ifdef __cplusplus
extern "C" {
#endif

typedef struct pn_catalog pn_catalog;

/**
 * The trusted keys compiled into this binary (catalog_keys.h; empty ones
 * skipped).
 * @return how many were written to out
 *
 * Thread-safe: yes. Reentrant: yes. Signal-safe: no
 */
int pn_catalog_trusted_keys(pn_minisign_pubkey *out, int max);

/**
 * Open the catalog state in state_dir and load the cached accepted manifest
 * when it still verifies. `keys` = NULL uses the compiled-in keys.
 *
 * Thread-safe: yes. Reentrant: yes. Signal-safe: no
 */
pn_catalog *pn_catalog_open(const char *state_dir, const pn_minisign_pubkey *keys, int n_keys);

/** NULL-safe. Thread-safe: no. Reentrant: no. Signal-safe: no */
void pn_catalog_close(pn_catalog *c);

/**
 * Verify, rollback-check, parse and persist a manifest + signature.
 * @return 0 when accepted, -1 with a reason
 *
 * Thread-safe: yes (internal lock). Reentrant: no. Signal-safe: no
 */
int pn_catalog_accept(pn_catalog *c, const char *manifest, size_t len, const char *sig,
                      char *err, size_t errlen);

/**
 * Fetch <url> and <url>.minisig with curl, then pn_catalog_accept.
 * @return 0 when accepted, -1 with a reason
 *
 * Thread-safe: yes. Reentrant: no. Signal-safe: no
 */
int pn_catalog_refresh(pn_catalog *c, const char *url, const char *protocols, char *err,
                       size_t errlen);

/** Version of the accepted manifest, -1 if none. Thread-safe: yes. */
long long pn_catalog_version(pn_catalog *c);

/**
 * Heap copy of the accepted manifest (caller frees), or NULL if none.
 *
 * Thread-safe: yes. Reentrant: yes. Signal-safe: no
 */
pn_catalog_manifest *pn_catalog_snapshot(pn_catalog *c);

#ifdef __cplusplus
}
#endif

#endif /* PN_LOCAL_CATALOG_H */
