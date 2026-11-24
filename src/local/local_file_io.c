/* SPDX-License-Identifier: AGPL-3.0-or-later */
/* Copyright (C) 2026 David (Corey) Herrera */
/**
 * @file   local_file_io.c
 * @brief  Implementation of the appliance file helpers (see local_file_io.h).
 *
 * @status None
 * @issues None
 * @todo   None
 */

#include "local_file_io.h"

#include <errno.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include <fcntl.h>
#include <sys/stat.h>
#include <unistd.h>

int pn_local_mkdir_p(const char *path, mode_t mode)
{
    char buf[PN_LOCAL_PATH_MAX];
    int n = snprintf(buf, sizeof(buf), "%s", path);

    if (n <= 0 || (size_t)n >= sizeof(buf))
        return -1;
    for (char *p = buf + 1; *p; p++) {
        if (*p != '/')
            continue;
        *p = '\0';
        if (mkdir(buf, mode) != 0 && errno != EEXIST)
            return -1;
        *p = '/';
    }
    if (mkdir(buf, mode) != 0 && errno != EEXIST)
        return -1;
    return 0;
}

int pn_local_read_file(const char *path, size_t max_bytes, char **out, size_t *len_out)
{
    FILE *f = fopen(path, "rb");
    char *buf = NULL;
    size_t cap = 4096, len = 0;

    *out = NULL;
    if (!f)
        return -1;
    buf = malloc(cap);
    while (buf) {
        if (len + 1 >= cap) {
            char *grown = realloc(buf, cap * 2);
            if (!grown) { free(buf); buf = NULL; break; }
            buf = grown;
            cap *= 2;
        }
        size_t got = fread(buf + len, 1, cap - len - 1, f);
        len += got;
        if (len > max_bytes) { free(buf); buf = NULL; break; }
        if (got == 0)
            break;
    }
    if (buf && ferror(f)) { free(buf); buf = NULL; }
    fclose(f);
    if (!buf)
        return -1;
    buf[len] = '\0';
    *out = buf;
    if (len_out)
        *len_out = len;
    return 0;
}

int pn_local_write_file_atomic(const char *path, const void *data, size_t len, mode_t mode)
{
    char tmp[PN_LOCAL_PATH_MAX];
    const char *p = data;
    size_t off = 0;
    int fd = -1;

    if (snprintf(tmp, sizeof(tmp), "%s.tmp", path) >= (int)sizeof(tmp))
        return -1;
    unlink(tmp);
    fd = open(tmp, O_WRONLY | O_CREAT | O_EXCL, mode);
    if (fd < 0)
        return -1;
    /* open() honours the umask; force the exact mode (0600 for secrets). */
    if (fchmod(fd, mode) != 0)
        goto fail;
    while (off < len) {
        ssize_t w = write(fd, p + off, len - off);
        if (w < 0 && errno == EINTR)
            continue;
        if (w <= 0)
            goto fail;
        off += (size_t)w;
    }
    if (fsync(fd) != 0)
        goto fail;
    if (close(fd) != 0) {
        fd = -1;
        goto fail;
    }
    fd = -1;
    if (rename(tmp, path) != 0)
        goto fail;
    return 0;
fail:
    if (fd >= 0)
        close(fd);
    unlink(tmp);
    return -1;
}

int pn_local_path_join(char *out, size_t cap, const char *dir, const char *name)
{
    int n = snprintf(out, cap, "%s/%s", dir, name);

    return (n < 0 || (size_t)n >= cap) ? -1 : 0;
}
