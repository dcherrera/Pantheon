/* SPDX-License-Identifier: AGPL-3.0-or-later */
/* Copyright (C) 2026 David (Corey) Herrera */
/**
 * @file   downloader.c
 * @brief  Implementation of download + verify (see downloader.h).
 *
 * @status None
 * @issues None
 * @todo   None
 */

#include "downloader.h"

#include <stdbool.h>
#include <stdio.h>
#include <string.h>

#include <unistd.h>

#include "curl_runner.h"
#include "ingot_fetch.h"
#include "model_store.h"
#include "sha256.h"

/* Hash the part file and compare with the catalog's sha256. */
static int verify_part(const char *part, const pn_cat_file *f, char *err, size_t errlen)
{
    uint8_t digest[PN_SHA256_LEN];
    char hex[PN_SHA256_HEX_LEN + 1];

    if (pn_sha256_file(part, digest) != 0) {
        snprintf(err, errlen, "cannot read %s to verify it", part);
        return -1;
    }
    pn_hex_encode(digest, sizeof(digest), hex);
    if (strcmp(hex, f->sha256) != 0) {
        snprintf(err, errlen, "sha256 mismatch for %s (got %.12s..., catalog %.12s...): deleted",
                 f->filename, hex, f->sha256);
        return -1;
    }
    return 0;
}

static int move_into_place(const char *part, const char *final, char *err, size_t errlen)
{
    if (rename(part, final) != 0) {
        snprintf(err, errlen, "cannot move %s into place", part);
        return -1;
    }
    return 0;
}

/* Whether to fetch the file's .ingot: it has one, a delta's base is installed (its path in
 * `base`, else ""), and the .ingot is smaller than what resuming the whole file from the
 * `have` bytes of its part file still needs. */
static bool ingot_wanted(const pn_download_cfg *cfg, const pn_cat_file *f, uint64_t have,
                         char *base, size_t cap)
{
    base[0] = '\0';
    if (!f->ingot.present || f->ingot.size >= f->size - have)
        return false;
    if (!f->ingot.base_sha256[0] ||
        pn_model_store_find_sha256(cfg->store_dir, cfg->catalog, f->ingot.base_sha256, base,
                                   cap) == 0)
        return true;
    fprintf(stderr, "local: %s: base %.12s... of %s is not installed; downloading the whole "
            "file\n", f->filename, f->ingot.base_sha256, f->ingot.filename);
    return false;
}

/* Fetch + decode the .ingot into the part file, then move it into place.
 * @return 0, PN_INGOT_EBASE (the caller downloads the whole file), or -1 */
static int fetch_ingot(const pn_download_cfg *cfg, const pn_cat_file *f, const char *base,
                       const char *part, const char *final, uint64_t have,
                       pn_download_phase_fn on_phase, void *user, char *err, size_t errlen)
{
    char url[1024];

    if (pn_cat_ingot_url(cfg->base_url, f, url, sizeof(url)) != 0) {
        snprintf(err, errlen, "URL too long");
        return -1;
    }
    if (pn_model_store_check_space(cfg->store_dir, f->size - have, err, errlen) != 0)
        return -1;
    pn_ingot_fetch_req req = { url, cfg->protocols, f->ingot.size, base[0] ? base : NULL, part,
                               f->sha256, f->size };
    if (on_phase)
        on_phase(user, PN_DL_DOWNLOADING, part);
    int rc = pn_ingot_fetch(&req, NULL, err, errlen);
    return rc != 0 ? rc : move_into_place(part, final, err, errlen);
}

/* The whole file: curl into the part file (resumed from `have`), size check, hash, move. */
static int fetch_whole(const pn_download_cfg *cfg, const pn_cat_file *f, const char *part,
                       const char *final, uint64_t have, pn_download_phase_fn on_phase,
                       void *user, char *err, size_t errlen)
{
    char url[1024], why[256];

    if (pn_cat_file_url(cfg->base_url, f, url, sizeof(url)) != 0) {
        snprintf(err, errlen, "URL too long");
        return -1;
    }
    if (pn_model_store_check_space(cfg->store_dir, f->size - have, err, errlen) != 0)
        return -1;
    if (have < f->size) {
        if (on_phase)
            on_phase(user, PN_DL_DOWNLOADING, part);
        int rc = pn_curl_download(url, part, cfg->protocols, have > 0, why, sizeof(why));
        if (rc != 0) {
            snprintf(err, errlen, "download of %s failed (curl %d): %s", f->filename, rc, why);
            return -1;
        }
    }
    if (pn_model_store_file_size(part) != f->size) {
        snprintf(err, errlen, "%s: got %llu bytes, catalog says %llu", f->filename,
                 (unsigned long long)pn_model_store_file_size(part),
                 (unsigned long long)f->size);
        return -1;
    }
    if (on_phase)
        on_phase(user, PN_DL_VERIFYING, part);
    if (verify_part(part, f, err, errlen) != 0) {
        unlink(part);
        return -1;
    }
    return move_into_place(part, final, err, errlen);
}

int pn_download_file(const pn_download_cfg *cfg, const pn_cat_model *m, const pn_cat_file *f,
                     pn_download_phase_fn on_phase, void *user, char *err, size_t errlen)
{
    char final[1024], part[1100], base[1024];

    if (pn_model_store_path(cfg->store_dir, m, f, final, sizeof(final)) != 0) {
        snprintf(err, errlen, "path too long");
        return -1;
    }
    if (pn_model_store_file_size(final) == f->size)
        return 0;
    snprintf(part, sizeof(part), "%s.part", final);
    uint64_t have = pn_model_store_file_size(part);
    if (have > f->size) {
        unlink(part);
        have = 0;
    }
    if (ingot_wanted(cfg, f, have, base, sizeof(base))) {
        int rc = fetch_ingot(cfg, f, base, part, final, have, on_phase, user, err, errlen);
        if (rc != PN_INGOT_EBASE)
            return rc;
        fprintf(stderr, "local: %s; downloading the whole file\n", err);
        have = 0;
    }
    return fetch_whole(cfg, f, part, final, have, on_phase, user, err, errlen);
}
