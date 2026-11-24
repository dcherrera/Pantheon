/* SPDX-License-Identifier: AGPL-3.0-or-later */
/* Copyright (C) 2026 David (Corey) Herrera */
/**
 * @file   ingot_fetch.c
 * @brief  Implementation of the .ingot download + streaming decode (see ingot_fetch.h).
 *
 * @status None
 * @issues None
 * @todo   None
 */

#include "ingot_fetch.h"

#include <errno.h>
#include <stdbool.h>
#include <stdio.h>
#include <string.h>

#include <fcntl.h>
#include <unistd.h>

#include "ingot/container/ingot_base.h"
#include "ingot/container/ingot_stream.h"
#include "ingot/core/ingot_error.h"
#include "ingot/core/ingot_input.h"

#include "curl_runner.h"
#include "sha256.h"

/* curl exit codes that retrying will not fix: the server answered >= 400 (--fail), or it
 * ignores ranges (--continue-at). */
#define CURL_HTTP_ERROR  22
#define CURL_RANGE_ERROR 33

/* The .ingot byte source: a curl child, restarted at `consumed` after a cut. */
typedef struct {
    const pn_ingot_fetch_req *req;
    pn_curl_stream            curl;
    uint64_t                  consumed;       /* bytes handed to the decoder */
    uint64_t                  attempt_from;   /* `consumed` when the current curl started */
    int                       restarts, stalls, curl_rc;
    bool                      start_failed, gave_up;
    char                      why[256];       /* curl's stderr, or why it did not start */
} ingot_feed;

/* The decoded file's destination: the part file, never past the catalog size. */
typedef struct {
    int      fd;
    uint64_t bytes, limit;
    int      write_errno;
    bool     too_big;
} part_sink;

static const char *base_name(const char *path)
{
    const char *slash = strrchr(path, '/');
    return slash ? slash + 1 : path;
}

/* curl ended before the .ingot did: reap it, decide whether to go on, back off when
 * the attempt brought nothing. 0 = start again from `consumed`. */
static int feed_restart(ingot_feed *f, bool read_failed)
{
    f->curl_rc = pn_curl_stream_close(&f->curl, read_failed, f->why, sizeof(f->why));
    f->stalls = f->consumed > f->attempt_from ? 0 : f->stalls + 1;
    if (f->curl_rc == CURL_HTTP_ERROR || f->curl_rc == CURL_RANGE_ERROR ||
        f->stalls >= PN_INGOT_MAX_STALLS) {
        f->gave_up = true;
        return -INGOT_ETRUNCATED;
    }
    if (f->stalls)
        sleep((unsigned)f->stalls);
    f->restarts++;
    fprintf(stderr, "local: %s cut at %llu of %llu bytes (curl %d%s%s); resuming\n",
            base_name(f->req->url), (unsigned long long)f->consumed,
            (unsigned long long)f->req->ingot_size, f->curl_rc, f->why[0] ? ": " : "", f->why);
    return 0;
}

/* ingot_read_fn: the next bytes of the .ingot; *got = 0 only at its catalog size. */
static int feed_read(void *ctx, void *buf, size_t n, size_t *got)
{
    ingot_feed *f = ctx;
    uint64_t left = f->req->ingot_size - f->consumed;

    *got = 0;
    if (left == 0)
        return 0;
    if (n > left)
        n = (size_t)left;
    for (;;) {
        if (f->curl.pid <= 0) {
            f->attempt_from = f->consumed;
            if (pn_curl_stream_open(f->req->url, f->req->protocols, f->consumed, &f->curl,
                                    f->why, sizeof(f->why)) != 0) {
                f->start_failed = true;
                return -INGOT_EIO;
            }
        }
        ssize_t r = read(f->curl.body_fd, buf, n);
        if (r > 0) {
            f->consumed += (uint64_t)r;
            *got = (size_t)r;
            return 0;
        }
        if (r < 0 && errno == EINTR)
            continue;
        int rc = feed_restart(f, r < 0);
        if (rc != 0)
            return rc;
    }
}

/* ingot_write_fn: append to the part file. */
static int sink_write(void *ctx, const void *buf, size_t n)
{
    part_sink *s = ctx;
    const char *p = buf;

    if (n > s->limit - s->bytes) {
        s->too_big = true;
        return -INGOT_ECORRUPT;
    }
    while (n > 0) {
        ssize_t w = write(s->fd, p, n);
        if (w < 0 && errno == EINTR)
            continue;
        if (w <= 0) {
            s->write_errno = w < 0 ? errno : EIO;
            return -INGOT_EIO;
        }
        p += w;
        n -= (size_t)w;
        s->bytes += (uint64_t)w;
    }
    return 0;
}

/* Run the decoder over the feed into the sink. @return 0 or a negated ingot_error */
static int decode(const pn_ingot_fetch_req *req, ingot_feed *feed, part_sink *sink,
                  ingot_stream_result *res)
{
    ingot_stream_options opt = { .threads = PN_INGOT_THREADS };
    ingot_base_set *base = NULL;
    ingot_input *in = NULL;
    int rc = 0;

    if (req->base_path && ingot_base_set_open(&req->base_path, 1, PN_INGOT_THREADS, &base) != 0)
        return -INGOT_EBASE;
    opt.base = base;
    rc = ingot_input_open_callback(feed_read, feed, &in);
    if (rc == 0)
        rc = ingot_stream_decode(in, sink_write, sink, &opt, res);
    pn_curl_stream_close(&feed->curl, true, NULL, 0);
    ingot_input_close(in);
    ingot_base_set_close(base);
    return rc;
}

/* The reason for a failed decode, in err. */
static void describe(int rc, const pn_ingot_fetch_req *req, const ingot_feed *feed,
                     const part_sink *sink, char *err, size_t errlen)
{
    const char *name = base_name(req->url);

    if (feed->start_failed)
        snprintf(err, errlen, "cannot start curl for %s: %s", name, feed->why);
    else if (feed->gave_up)   /* whatever the decoder made of the early end */
        snprintf(err, errlen, "download of %s stopped at %llu of %llu bytes (curl %d): %s", name,
                 (unsigned long long)feed->consumed, (unsigned long long)req->ingot_size,
                 feed->curl_rc, feed->why[0] ? feed->why : "connection lost");
    else if (rc == -INGOT_EBASE)
        snprintf(err, errlen, "%s: base %s missing or not the one it was coded against", name,
                 req->base_path ? req->base_path : "(none)");
    else if (sink->write_errno)
        snprintf(err, errlen, "cannot write %s: %s", req->part_path, strerror(sink->write_errno));
    else if (sink->too_big)
        snprintf(err, errlen, "%s decodes to more than the catalog's %llu bytes: deleted", name,
                 (unsigned long long)req->size);
    else
        snprintf(err, errlen, "%s failed its checks (%s): deleted", name, ingot_error_name(rc));
}

/* After a clean decode: everything read, and the file is the catalog's. */
static int check_result(const pn_ingot_fetch_req *req, const ingot_feed *feed,
                        const part_sink *sink, const ingot_stream_result *res, char *err,
                        size_t errlen)
{
    char hex[PN_SHA256_HEX_LEN + 1];

    pn_hex_encode(res->sha256, sizeof(res->sha256), hex);
    if (feed->consumed != req->ingot_size) {
        snprintf(err, errlen, "%s ended at %llu bytes, catalog says %llu: deleted",
                 base_name(req->url), (unsigned long long)feed->consumed,
                 (unsigned long long)req->ingot_size);
        return -1;
    }
    if (res->source_bytes != req->size || sink->bytes != req->size) {
        snprintf(err, errlen, "%s decoded to %llu bytes, catalog says %llu: deleted",
                 base_name(req->url), (unsigned long long)sink->bytes,
                 (unsigned long long)req->size);
        return -1;
    }
    if (strcmp(hex, req->sha256) != 0) {
        snprintf(err, errlen, "sha256 mismatch for %s (got %.12s..., catalog %.12s...): deleted",
                 base_name(req->url), hex, req->sha256);
        return -1;
    }
    return 0;
}

int pn_ingot_fetch(const pn_ingot_fetch_req *req, pn_ingot_fetch_stats *stats, char *err,
                   size_t errlen)
{
    ingot_feed feed = { .req = req };
    part_sink sink = { .limit = req->size };
    ingot_stream_result res;
    int rc = 0;

    memset(&res, 0, sizeof(res));
    feed.curl.body_fd = feed.curl.err_fd = -1;
    sink.fd = open(req->part_path, O_WRONLY | O_CREAT | O_TRUNC | O_CLOEXEC, 0644);
    if (sink.fd < 0) {
        snprintf(err, errlen, "cannot create %s: %s", req->part_path, strerror(errno));
        return -1;
    }
    int drc = decode(req, &feed, &sink, &res);
    if (drc != 0) {
        describe(drc, req, &feed, &sink, err, errlen);
        rc = drc == -INGOT_EBASE ? PN_INGOT_EBASE : -1;
    } else {
        rc = check_result(req, &feed, &sink, &res, err, errlen);
    }
    if (close(sink.fd) != 0 && rc == 0) {
        snprintf(err, errlen, "cannot write %s: %s", req->part_path, strerror(errno));
        rc = -1;
    }
    if (rc != 0)
        unlink(req->part_path);
    if (stats) {
        stats->ingot_bytes = feed.consumed;
        stats->restarts = feed.restarts;
    }
    return rc;
}
