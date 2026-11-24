/* SPDX-License-Identifier: AGPL-3.0-or-later */
/* Copyright (C) 2026 David (Corey) Herrera */
/**
 * @file   curl_runner.h
 * @brief  Fetch a URL to a file with the system `curl` as a child process (D2):
 *         fork + execv with a fixed argument vector, never a shell string. TLS
 *         and CA updates come from the OS; integrity comes from the signed
 *         sha256, not from TLS. Resume uses `--continue-at -` on the part file. A
 *         stream variant pipes the body to the caller (.ingot downloads decode it
 *         while it arrives).
 *
 * @status None
 * @issues None
 * @todo   None
 */

#ifndef PN_LOCAL_CURL_RUNNER_H
#define PN_LOCAL_CURL_RUNNER_H

#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>

#include <sys/types.h>

#ifdef __cplusplus
extern "C" {
#endif

#define PN_CURL_PROTOCOLS_HTTPS "=https"
#define PN_CURL_PROTOCOLS_ANY   "=https,http,file"

/**
 * Find `curl` on PATH (once; cached).
 * @return the absolute path, or NULL when curl is missing
 *
 * Thread-safe: yes (first call should happen at startup). Reentrant: yes.
 * Signal-safe: no
 */
const char *pn_curl_path(void);

/**
 * Download url → out_path.
 *
 * @param protocols  curl --proto / --proto-redir value (PN_CURL_PROTOCOLS_*)
 * @param resume     continue from out_path's current size
 * @param err        curl's first stderr line, or a reason, on failure
 * @return 0 on success, else curl's exit code (or -1 if it could not run)
 *
 * Thread-safe: yes. Reentrant: yes. Signal-safe: no
 */
int pn_curl_download(const char *url, const char *out_path, const char *protocols,
                     bool resume, char *err, size_t errlen);

/* A curl child writing a URL's body into a pipe (pn_curl_stream_open). */
typedef struct {
    pid_t pid;       /* 0 = not running */
    int   body_fd;   /* read end: the body from `offset` on */
    int   err_fd;    /* read end: curl's stderr */
} pn_curl_stream;

/**
 * Start curl writing url's body, from byte `offset` on (`--continue-at`, so a server
 * that ignores the range fails instead of sending the wrong bytes), into a pipe the
 * caller reads from s->body_fd. Same protocol restrictions as pn_curl_download, plus
 * a stall guard (aborts when under 1 byte/s for 120 s). Both pipe ends the caller
 * holds are close-on-exec. Finish with pn_curl_stream_close.
 *
 * @return 0, or -1 (curl missing, pipe or fork failed) with the reason in err
 *
 * Thread-safe: yes. Reentrant: yes. Signal-safe: no
 */
int pn_curl_stream_open(const char *url, const char *protocols, uint64_t offset,
                        pn_curl_stream *s, char *err, size_t errlen);

/**
 * End a stream: close the body pipe (SIGTERM first when `stop`, for a transfer the
 * caller abandons), reap curl and zero *s. A no-op on a stream that is not running.
 *
 * @param err  curl's first stderr line, when it wrote one
 * @return curl's exit code (0 = the whole body arrived), or -1 when it died of a
 *         signal or was not running
 *
 * Thread-safe: yes (one caller per stream). Reentrant: yes. Signal-safe: no
 */
int pn_curl_stream_close(pn_curl_stream *s, bool stop, char *err, size_t errlen);

#ifdef __cplusplus
}
#endif

#endif /* PN_LOCAL_CURL_RUNNER_H */
