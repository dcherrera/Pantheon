/* SPDX-License-Identifier: AGPL-3.0-or-later */
/* Copyright (C) 2026 David (Corey) Herrera */
/**
 * @file   ingot_fetch.h
 * @brief  Download a foundry .ingot and decode it while it arrives (catalog
 *         files[].ingot): curl writes the .ingot into a pipe, foundry's streaming
 *         decoder (ingot/container/ingot_stream.h) reads it once, in order, and
 *         writes the exact file in order into the part file. Memory stays a few
 *         coded blocks; the .ingot is never stored. A cut transfer restarts curl
 *         at the .ingot bytes already read and keeps feeding the same decoder.
 *         The decoder's SHA-256 of the rebuilt file is the integrity check (no
 *         second hashing pass).
 *
 * @status None
 * @issues None
 * @todo   None
 */

#ifndef PN_LOCAL_INGOT_FETCH_H
#define PN_LOCAL_INGOT_FETCH_H

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

#define PN_INGOT_THREADS    2    /* decode threads: the boxes have 2 cores */
#define PN_INGOT_MAX_STALLS 5    /* curl attempts in a row that bring no new byte */
#define PN_INGOT_EBASE      (-2) /* pn_ingot_fetch: the delta base was rejected */

typedef struct {
    const char *url;          /* the .ingot */
    const char *protocols;    /* curl --proto value (PN_CURL_PROTOCOLS_*) */
    uint64_t    ingot_size;   /* bytes of the .ingot (catalog) */
    const char *base_path;    /* installed base file of a delta .ingot, or NULL */
    const char *part_path;    /* output: created or truncated, deleted on failure */
    const char *sha256;       /* 64 lowercase hex: the decoded file's (catalog) */
    uint64_t    size;         /* bytes of the decoded file (catalog) */
} pn_ingot_fetch_req;

typedef struct {
    uint64_t ingot_bytes;     /* .ingot bytes read */
    int      restarts;        /* curl restarts after a cut transfer */
} pn_ingot_fetch_stats;

/**
 * Fetch req->url and decode it into req->part_path. On success the part file
 * holds exactly req->size bytes whose SHA-256 is req->sha256 (checked by the
 * decoder), and every .ingot byte was read. On any failure the part file is
 * deleted.
 *
 * @param stats  may be NULL
 * @param err    the reason on failure (curl's message for a failed transfer)
 * @return 0; PN_INGOT_EBASE when the base is missing, unreadable or not the one
 *         the .ingot was coded against (the caller may download the whole file);
 *         or -1
 *
 * Thread-safe: yes (distinct part files). Reentrant: yes. Signal-safe: no
 */
int pn_ingot_fetch(const pn_ingot_fetch_req *req, pn_ingot_fetch_stats *stats, char *err,
                   size_t errlen);

#ifdef __cplusplus
}
#endif

#endif /* PN_LOCAL_INGOT_FETCH_H */
