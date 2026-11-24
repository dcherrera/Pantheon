/* SPDX-License-Identifier: AGPL-3.0-or-later */
/* Copyright (C) 2026 David (Corey) Herrera */
/**
 * @file   model_store.c
 * @brief  Implementation of the model store (see model_store.h).
 *
 * @status None
 * @issues None
 * @todo   None
 */

#include "model_store.h"

#include <stdio.h>
#include <string.h>

#include <unistd.h>

#include <sys/stat.h>
#include <sys/statvfs.h>

int pn_model_store_path(const char *store_dir, const pn_cat_model *m, const pn_cat_file *f,
                        char *out, size_t cap)
{
    const char *dot = strrchr(f->filename, '.');
    const char *ext = dot && dot[1] ? dot + 1 : "bin";
    int n = snprintf(out, cap, "%s/%s-%.12s.%s", store_dir, m->id, f->sha256, ext);

    return (n < 0 || (size_t)n >= cap) ? -1 : 0;
}

int pn_model_store_weights_path(const char *store_dir, const pn_cat_model *need, char *out,
                                size_t cap)
{
    const pn_cat_file *w = pn_cat_weights_file(need);

    return w ? pn_model_store_path(store_dir, need, w, out, cap) : -1;
}

uint64_t pn_model_store_file_size(const char *path)
{
    struct stat sb;

    return stat(path, &sb) == 0 && S_ISREG(sb.st_mode) ? (uint64_t)sb.st_size : 0;
}

bool pn_model_store_installed(const char *store_dir, const pn_cat_model *m)
{
    char path[1024];

    for (int i = 0; i < m->n_files; i++)
        if (pn_model_store_path(store_dir, m, &m->files[i], path, sizeof(path)) != 0 ||
            pn_model_store_file_size(path) != m->files[i].size)
            return false;
    return m->n_files > 0;
}

int pn_model_store_check_space(const char *store_dir, uint64_t need, char *reason, size_t cap)
{
    struct statvfs vfs;

    if (statvfs(store_dir, &vfs) != 0) {
        snprintf(reason, cap, "cannot read free space on %s", store_dir);
        return -1;
    }
    uint64_t have = (uint64_t)vfs.f_bavail * (uint64_t)vfs.f_frsize;
    if (have >= need + PN_STORE_MARGIN_BYTES)
        return 0;
    snprintf(reason, cap, "needs %.1f GB free on %s, has %.1f GB",
             (double)(need + PN_STORE_MARGIN_BYTES) / 1e9, store_dir, (double)have / 1e9);
    return -1;
}

/* Point `link` at `target` (a name in the same directory); atomic via temp + rename. */
static int link_atomically(const char *target, const char *link)
{
    char cur[1024], tmp[1100];
    ssize_t n = readlink(link, cur, sizeof(cur) - 1);

    if (n >= 0) {
        cur[n] = '\0';
        if (strcmp(cur, target) == 0)
            return 0;
    }
    snprintf(tmp, sizeof(tmp), "%s.link-tmp", link);
    unlink(tmp);
    if (symlink(target, tmp) != 0)
        return -1;
    if (rename(tmp, link) != 0) {
        unlink(tmp);
        return -1;
    }
    return 0;
}

int pn_model_store_link_companions(const char *store_dir, const pn_cat_model *m, char *err,
                                   size_t errlen)
{
    const pn_cat_file *w = pn_cat_weights_file(m);
    const char *dot = w ? strrchr(w->filename, '.') : NULL;
    size_t stem = dot ? (size_t)(dot - w->filename) : 0;
    char wpath[1024], fpath[1024], link[1024];

    if (!w || stem == 0 || pn_model_store_path(store_dir, m, w, wpath, sizeof(wpath)) != 0) {
        snprintf(err, errlen, "no weights file to link companions to");
        return -1;
    }
    for (int i = 0; i < m->n_files; i++) {
        const pn_cat_file *f = &m->files[i];
        if (f == w || strncmp(f->filename, w->filename, stem) != 0 || f->filename[stem] != '.')
            continue;
        int n = snprintf(link, sizeof(link), "%s/%s-%.12s%s", store_dir, m->id, w->sha256,
                         f->filename + stem);
        if (n < 0 || (size_t)n >= sizeof(link) || strcmp(link, wpath) == 0 ||
            pn_model_store_path(store_dir, m, f, fpath, sizeof(fpath)) != 0) {
            snprintf(err, errlen, "cannot name the companion of %s", f->filename);
            return -1;
        }
        const char *base = strrchr(fpath, '/');
        if (link_atomically(base ? base + 1 : fpath, link) != 0) {
            snprintf(err, errlen, "cannot link %s in the store", f->filename);
            return -1;
        }
    }
    return 0;
}

int pn_model_store_find_sha256(const char *store_dir, const pn_catalog_manifest *catalog,
                               const char *sha256, char *out, size_t cap)
{
    for (int i = 0; catalog && sha256 && i < catalog->n_models; i++) {
        const pn_cat_model *m = &catalog->models[i];
        for (int j = 0; j < m->n_files; j++) {
            const pn_cat_file *f = &m->files[j];
            if (strcmp(f->sha256, sha256) == 0 &&
                pn_model_store_path(store_dir, m, f, out, cap) == 0 &&
                pn_model_store_file_size(out) == f->size)
                return 0;
        }
    }
    if (cap)
        out[0] = '\0';
    return -1;
}
