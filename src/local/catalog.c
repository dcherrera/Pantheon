/* SPDX-License-Identifier: AGPL-3.0-or-later */
/* Copyright (C) 2026 David (Corey) Herrera */
/**
 * @file   catalog.c
 * @brief  Implementation of the accepted catalog (see catalog.h).
 *
 * @status None
 * @issues None
 * @todo   None
 */

#include "catalog.h"

#include <pthread.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include <unistd.h>

#include "catalog_keys.h"
#include "curl_runner.h"
#include "local_file_io.h"

#define CATALOG_MAX_BYTES (4u << 20)
#define MAX_KEYS 4

struct pn_catalog {
    pthread_mutex_t      lock;
    char                 dir[PN_LOCAL_PATH_MAX];
    pn_minisign_pubkey   keys[MAX_KEYS];
    int                  n_keys;
    pn_catalog_manifest *manifest;      /* NULL until one is accepted */
};

int pn_catalog_trusted_keys(pn_minisign_pubkey *out, int max)
{
    const char *lines[] = { PN_CATALOG_KEY_CURRENT, PN_CATALOG_KEY_BACKUP, PN_CATALOG_KEY_TEST };
    int n = 0;

    for (size_t i = 0; i < sizeof(lines) / sizeof(lines[0]) && n < max; i++)
        if (lines[i][0] && pn_minisign_parse_pubkey(lines[i], &out[n]) == 0)
            n++;
    return n;
}

static void state_path(const pn_catalog *c, const char *name, char *out, size_t cap)
{
    if (pn_local_path_join(out, cap, c->dir, name) != 0)
        out[0] = '\0';
}

/* Last accepted version from <state>/catalog_version, -1 if none. */
static long long last_accepted(const pn_catalog *c)
{
    char path[PN_LOCAL_PATH_MAX], *text = NULL;
    long long v = -1;

    state_path(c, "catalog_version", path, sizeof(path));
    if (pn_local_read_file(path, 64, &text, NULL) == 0) {
        v = strtoll(text, NULL, 10);
        free(text);
    }
    return v;
}

/* Verify + rollback check + parse into a fresh manifest. */
static pn_catalog_manifest *verify_and_parse(pn_catalog *c, const char *manifest, size_t len,
                                             const char *sig, char *err, size_t errlen)
{
    char why[160];
    pn_catalog_manifest *m = NULL;

    if (c->n_keys == 0) {
        snprintf(err, errlen, "this build trusts no catalog key (catalog_keys.h is empty)");
        return NULL;
    }
    if (pn_minisign_verify((const uint8_t *)manifest, len, sig, c->keys, c->n_keys, NULL, NULL,
                           why, sizeof(why)) != 0) {
        snprintf(err, errlen, "catalog signature refused: %s", why);
        return NULL;
    }
    m = calloc(1, sizeof(*m));
    if (!m || pn_cat_manifest_parse(manifest, m, why, sizeof(why)) != 0) {
        snprintf(err, errlen, "catalog refused: %s", m ? why : "oom");
        free(m);
        return NULL;
    }
    long long last = last_accepted(c);
    if (m->version < last) {
        snprintf(err, errlen, "catalog version %lld is older than the accepted %lld (rollback)",
                 m->version, last);
        free(m);
        return NULL;
    }
    return m;
}

/* Keep the verified files and the version, atomically. */
static int persist(pn_catalog *c, const char *manifest, size_t len, const char *sig,
                   long long version)
{
    char path[PN_LOCAL_PATH_MAX], num[32];

    state_path(c, "catalog.json.minisig", path, sizeof(path));
    if (pn_local_write_file_atomic(path, sig, strlen(sig), 0600) != 0)
        return -1;
    state_path(c, "catalog.json", path, sizeof(path));
    if (pn_local_write_file_atomic(path, manifest, len, 0600) != 0)
        return -1;
    snprintf(num, sizeof(num), "%lld\n", version);
    state_path(c, "catalog_version", path, sizeof(path));
    return pn_local_write_file_atomic(path, num, strlen(num), 0600);
}

int pn_catalog_accept(pn_catalog *c, const char *manifest, size_t len, const char *sig,
                      char *err, size_t errlen)
{
    int rc = -1;

    /* One accept at a time, so the rollback check and the write agree. */
    pthread_mutex_lock(&c->lock);
    pn_catalog_manifest *m = verify_and_parse(c, manifest, len, sig, err, errlen);
    if (m && persist(c, manifest, len, sig, m->version) != 0) {
        snprintf(err, errlen, "could not write the catalog to %s", c->dir);
    } else if (m) {
        free(c->manifest);
        c->manifest = m;
        m = NULL;
        rc = 0;
        fprintf(stderr, "local: catalog version %lld accepted\n", c->manifest->version);
    }
    pthread_mutex_unlock(&c->lock);
    free(m);
    return rc;
}

pn_catalog *pn_catalog_open(const char *state_dir, const pn_minisign_pubkey *keys, int n_keys)
{
    pn_catalog *c = calloc(1, sizeof(*c));
    char path[PN_LOCAL_PATH_MAX], err[200], *manifest = NULL, *sig = NULL;
    size_t len = 0;

    if (!c || snprintf(c->dir, sizeof(c->dir), "%s", state_dir) >= (int)sizeof(c->dir)) {
        free(c);
        return NULL;
    }
    pthread_mutex_init(&c->lock, NULL);
    if (keys) {
        c->n_keys = n_keys < MAX_KEYS ? n_keys : MAX_KEYS;
        memcpy(c->keys, keys, (size_t)c->n_keys * sizeof(*keys));
    } else {
        c->n_keys = pn_catalog_trusted_keys(c->keys, MAX_KEYS);
    }
    state_path(c, "catalog.json", path, sizeof(path));
    if (pn_local_read_file(path, CATALOG_MAX_BYTES, &manifest, &len) == 0) {
        state_path(c, "catalog.json.minisig", path, sizeof(path));
        if (pn_local_read_file(path, 4096, &sig, NULL) == 0)
            c->manifest = verify_and_parse(c, manifest, len, sig, err, sizeof(err));
        if (!c->manifest)
            fprintf(stderr, "local: cached catalog not used: %s\n", sig ? err : "no signature");
    }
    free(manifest);
    free(sig);
    return c;
}

void pn_catalog_close(pn_catalog *c)
{
    if (!c)
        return;
    free(c->manifest);
    pthread_mutex_destroy(&c->lock);
    free(c);
}

int pn_catalog_refresh(pn_catalog *c, const char *url, const char *protocols, char *err,
                       size_t errlen)
{
    char mpath[PN_LOCAL_PATH_MAX], spath[PN_LOCAL_PATH_MAX], sig_url[1100], why[200];
    char *manifest = NULL, *sig = NULL;
    size_t len = 0;
    int rc = -1;

    state_path(c, "catalog.fetch.json", mpath, sizeof(mpath));
    state_path(c, "catalog.fetch.minisig", spath, sizeof(spath));
    snprintf(sig_url, sizeof(sig_url), "%s.minisig", url);
    unlink(mpath);
    unlink(spath);
    if (pn_curl_download(url, mpath, protocols, false, why, sizeof(why)) != 0) {
        snprintf(err, errlen, "catalog download failed: %s", why);
    } else if (pn_curl_download(sig_url, spath, protocols, false, why, sizeof(why)) != 0) {
        snprintf(err, errlen, "catalog is unsigned or the signature is missing: %s", why);
    } else if (pn_local_read_file(mpath, CATALOG_MAX_BYTES, &manifest, &len) != 0 ||
               pn_local_read_file(spath, 4096, &sig, NULL) != 0) {
        snprintf(err, errlen, "downloaded catalog unreadable or too large");
    } else {
        rc = pn_catalog_accept(c, manifest, len, sig, err, errlen);
    }
    free(manifest);
    free(sig);
    unlink(mpath);
    unlink(spath);
    return rc;
}

long long pn_catalog_version(pn_catalog *c)
{
    long long v = -1;

    if (!c)
        return -1;
    pthread_mutex_lock(&c->lock);
    if (c->manifest)
        v = c->manifest->version;
    pthread_mutex_unlock(&c->lock);
    return v;
}

pn_catalog_manifest *pn_catalog_snapshot(pn_catalog *c)
{
    pn_catalog_manifest *copy = NULL;

    if (!c)
        return NULL;
    pthread_mutex_lock(&c->lock);
    if (c->manifest && (copy = malloc(sizeof(*copy))) != NULL)
        memcpy(copy, c->manifest, sizeof(*copy));
    pthread_mutex_unlock(&c->lock);
    return copy;
}
