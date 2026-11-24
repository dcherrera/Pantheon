/* SPDX-License-Identifier: AGPL-3.0-or-later */
/* Copyright (C) 2026 David (Corey) Herrera */
/**
 * @file   model_store_cleanup.c
 * @brief  Implementation of the store listing and uninstall (see
 *         model_store_cleanup.h).
 *
 * @status None
 * @issues None
 * @todo   None
 */

#include "model_store_cleanup.h"

#include <ctype.h>
#include <dirent.h>
#include <errno.h>
#include <stdbool.h>
#include <stdio.h>
#include <string.h>

#include <unistd.h>

#include <sys/stat.h>

#define STORE_SHA_PREFIX 12

/* Whether `s` starts with "-<12 hex>.". */
static bool at_sha_marker(const char *s)
{
    if (s[0] != '-')
        return false;
    for (int i = 1; i <= STORE_SHA_PREFIX; i++)
        if (!isxdigit((unsigned char)s[i]))
            return false;
    return s[STORE_SHA_PREFIX + 1] == '.';
}

int pn_model_store_entry_id(const char *name, char *id, size_t cap)
{
    for (size_t p = 1; name[p]; p++) {
        if (!at_sha_marker(name + p))
            continue;
        if (p >= cap)
            return -1;
        memcpy(id, name, p);
        id[p] = '\0';
        return 0;
    }
    return -1;
}

static bool ends_with(const char *s, const char *suffix)
{
    size_t n = strlen(s), m = strlen(suffix);

    return n >= m && strcmp(s + n - m, suffix) == 0;
}

/* lstat one entry of the store: regular file bytes, or 0 for a link. */
static int entry_stat(const char *store_dir, const char *name, struct stat *sb)
{
    char path[1024];
    int n = snprintf(path, sizeof(path), "%s/%s", store_dir, name);

    return (n < 0 || (size_t)n >= sizeof(path)) ? -1 : lstat(path, sb);
}

/* The group of `id` in out[0..n), added when missing and there is room. */
static pn_store_group *group_of(pn_store_group *out, int *n, int max, const char *id)
{
    for (int i = 0; i < *n; i++)
        if (strcmp(out[i].catalog_id, id) == 0)
            return &out[i];
    if (*n >= max)
        return NULL;
    pn_store_group *g = &out[(*n)++];
    memset(g, 0, sizeof(*g));
    snprintf(g->catalog_id, sizeof(g->catalog_id), "%s", id);
    return g;
}

int pn_model_store_scan(const char *store_dir, pn_store_group *out, int max)
{
    DIR *d = opendir(store_dir);
    struct dirent *de;
    char id[PN_CAT_ID_MAX];
    int n = 0;

    if (!d)
        return 0;
    while ((de = readdir(d)) != NULL) {
        struct stat sb;
        if (pn_model_store_entry_id(de->d_name, id, sizeof(id)) != 0 ||
            entry_stat(store_dir, de->d_name, &sb) != 0 || S_ISDIR(sb.st_mode))
            continue;
        pn_store_group *g = group_of(out, &n, max, id);
        if (!g)
            continue;
        g->n_files++;
        g->n_partial += ends_with(de->d_name, ".part");
        if (S_ISREG(sb.st_mode))
            g->bytes += (uint64_t)sb.st_size;
    }
    closedir(d);
    return n;
}

int pn_model_store_remove_id(const char *store_dir, const char *catalog_id, uint64_t *freed,
                             int *n_removed, char *err, size_t errlen)
{
    DIR *d = opendir(store_dir);
    struct dirent *de;
    char id[PN_CAT_ID_MAX], path[1024];
    int rc = 0;

    *freed = 0;
    *n_removed = 0;
    if (!d) {
        snprintf(err, errlen, "cannot read the model store %s", store_dir);
        return -1;
    }
    while ((de = readdir(d)) != NULL) {
        struct stat sb;
        if (pn_model_store_entry_id(de->d_name, id, sizeof(id)) != 0 ||
            strcmp(id, catalog_id) != 0 || entry_stat(store_dir, de->d_name, &sb) != 0 ||
            S_ISDIR(sb.st_mode))
            continue;
        snprintf(path, sizeof(path), "%s/%s", store_dir, de->d_name);
        if (unlink(path) != 0) {
            if (rc == 0)
                snprintf(err, errlen, "could not remove %s: %s", de->d_name, strerror(errno));
            rc = -1;
            continue;
        }
        (*n_removed)++;
        if (S_ISREG(sb.st_mode))
            *freed += (uint64_t)sb.st_size;
    }
    closedir(d);
    return rc;
}
