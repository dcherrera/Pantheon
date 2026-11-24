/* SPDX-License-Identifier: AGPL-3.0-or-later */
/* Copyright (C) 2026 David (Corey) Herrera */
/**
 * @file   local_file_io.h
 * @brief  Small file helpers for the appliance state dir: mkdir -p with a mode,
 *         whole-file read, and atomic write (temp file + fsync + rename) with an
 *         explicit mode, so secrets are never visible with a looser mode.
 *
 * @status None
 * @issues None
 * @todo   None
 */

#ifndef PN_LOCAL_FILE_IO_H
#define PN_LOCAL_FILE_IO_H

#include <stddef.h>
#include <sys/types.h>

#ifdef __cplusplus
extern "C" {
#endif

#define PN_LOCAL_PATH_MAX 1024

/**
 * Create `path` and its parents with `mode` (existing directories are kept).
 * @return 0, or -1 (errno set)
 *
 * Thread-safe: yes
 * Reentrant:   yes
 * Signal-safe: no
 */
int pn_local_mkdir_p(const char *path, mode_t mode);

/**
 * Read a whole file into a NUL-terminated heap buffer.
 * @param out      receives the buffer (caller frees)
 * @param len_out  receives the byte count (without the NUL); may be NULL
 * @return 0, or -1 (missing/unreadable/over max_bytes)
 *
 * Thread-safe: yes
 * Reentrant:   yes
 * Signal-safe: no
 */
int pn_local_read_file(const char *path, size_t max_bytes, char **out, size_t *len_out);

/**
 * Atomically replace `path` with `data`: write `<path>.tmp` created with
 * `mode`, fsync, rename. A crash leaves either the old or the new file.
 * @return 0, or -1
 *
 * Thread-safe: yes (distinct paths); callers serialize writes to one path
 * Reentrant:   yes
 * Signal-safe: no
 */
int pn_local_write_file_atomic(const char *path, const void *data, size_t len, mode_t mode);

/**
 * Join dir + "/" + name into out[cap].
 * @return 0, or -1 on truncation
 *
 * Thread-safe: yes
 * Reentrant:   yes
 * Signal-safe: yes
 */
int pn_local_path_join(char *out, size_t cap, const char *dir, const char *name);

#ifdef __cplusplus
}
#endif

#endif /* PN_LOCAL_FILE_IO_H */
