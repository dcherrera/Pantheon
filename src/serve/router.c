/* SPDX-License-Identifier: AGPL-3.0-or-later */
/* Copyright (C) 2026 David (Corey) Herrera */
/**
 * @file router.c
 * @brief Implementation of the dispatch router (see router.h). Uses json_reader
 *        to pull the `model` + `stream` fields, registry.c to pick the
 *        least-busy live instance, and http_client.c to reverse-proxy with SSE
 *        pass-through. Failover retries the next-least-busy instance as long as
 *        the failure happened BEFORE any body byte was relayed downstream.
 *
 * Relay strategy:
 *   - Streaming (`stream:true`): write the SSE response headers to the client
 *     first, then relay upstream body bytes verbatim as they arrive (the
 *     upstream serve already emits `data: …\n\n` framing incl. the terminal
 *     [DONE]). Unbuffered.
 *   - Non-streaming: buffer the (small) JSON body from upstream, then send it as
 *     one application/json response. Buffering here is correct — a non-streamed
 *     reply is a single JSON object, and it lets a mid-stream upstream error
 *     still fail over cleanly (nothing was sent downstream yet).
 *
 * @status G2.
 * @issues None
 * @todo None
 */

#include "router.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>
#include <errno.h>

#include "http_client.h"
#include "http_server.h"
#include "json_reader.h"
#include "session_key.h"
#include "json_writer.h"

/* Max failover attempts (distinct instances tried for one request). */
#define ROUTER_MAX_TRIES 4
/* Connect timeout for an upstream, ms. */
#define ROUTER_CONNECT_MS 2000

/* ---- streaming relay: write upstream body bytes straight to the client ---- */

typedef struct {
    int client_fd;
    int header_sent;   /* SSE header written yet */
} stream_relay;

static int write_all_fd(int fd, const char *buf, size_t len)
{
    size_t off = 0;
    while (off < len) {
        ssize_t n = write(fd, buf + off, len - off);
        if (n < 0) { if (errno == EINTR) continue; return -1; }
        off += (size_t)n;
    }
    return 0;
}

static int stream_sink(const char *bytes, size_t len, void *user)
{
    stream_relay *sr = user;
    if (!sr->header_sent) {
        hs_begin_sse(sr->client_fd);   /* 200 + text/event-stream headers */
        sr->header_sent = 1;
    }
    /* Relay verbatim — the upstream already frames SSE events. */
    return write_all_fd(sr->client_fd, bytes, len);
}

/* ---- non-streaming relay: accumulate the body into a growable buffer ---- */

typedef struct {
    char  *data;
    size_t len;
    size_t cap;
    int    oom;
} buf_sink;

static int buffer_sink(const char *bytes, size_t len, void *user)
{
    buf_sink *b = user;
    if (b->len + len + 1 > b->cap) {
        size_t ncap = b->cap ? b->cap * 2 : 4096;
        while (ncap < b->len + len + 1) ncap *= 2;
        char *g = realloc(b->data, ncap);
        if (!g) { b->oom = 1; return -1; }
        b->data = g;
        b->cap = ncap;
    }
    memcpy(b->data + b->len, bytes, len);
    b->len += len;
    b->data[b->len] = '\0';
    return 0;
}

/* Outcome of one attempt against a picked instance. TERMINAL means the client
 * was fully answered (success OR an unrecoverable mid-relay failure) and the
 * caller must stop; FAILOVER means nothing was relayed and the caller should try
 * the next instance. */
typedef enum {
    ATTEMPT_TERMINAL = 0,
    ATTEMPT_FAILOVER = 1,
} attempt_outcome;

/* Proxy one request to `pick` and relay the response to the client. `extra_hdrs`
 * is forwarded upstream (the X-Session-Id line, or NULL). Returns whether the
 * caller should fail over to another instance. */
static attempt_outcome proxy_to_instance(int client_fd, const reg_instance *pick,
                                          const char *path, const char *body,
                                          size_t body_len, bool streaming,
                                          const char *extra_hdrs)
{
    int status = 0;
    hc_result rc;
    if (streaming) {
        stream_relay sr = { .client_fd = client_fd, .header_sent = 0 };
        rc = hc_post_stream(pick->endpoint, path, body, body_len, extra_hdrs,
                            ROUTER_CONNECT_MS, stream_sink, &sr, &status);
        if (rc == HC_OK || rc == HC_RELAY_ABORTED)
            return ATTEMPT_TERMINAL;
        if (rc == HC_MIDSTREAM_FAILED) {
            /* Bytes already went downstream — cannot cleanly retry; just end
             * the (broken) SSE stream and stop. */
            if (sr.header_sent) hs_end_sse(client_fd);
            return ATTEMPT_TERMINAL;
        }
        return ATTEMPT_FAILOVER;   /* HC_CONNECT_FAILED / HC_UPSTREAM_ERROR */
    }

    buf_sink b = {0};
    rc = hc_post_stream(pick->endpoint, path, body, body_len, extra_hdrs,
                        ROUTER_CONNECT_MS, buffer_sink, &b, &status);
    if (rc == HC_OK) {
        if (b.oom || !b.data) {
            free(b.data);
            hs_send_error(client_fd, 502, "upstream_error",
                          "empty upstream response");
        } else {
            hs_send_json(client_fd, status ? status : 200, b.data);
            free(b.data);
        }
        return ATTEMPT_TERMINAL;
    }
    free(b.data);
    if (rc == HC_MIDSTREAM_FAILED || rc == HC_RELAY_ABORTED) {
        hs_send_error(client_fd, 502, "upstream_error",
                      "upstream failed mid-response");
        return ATTEMPT_TERMINAL;
    }
    return ATTEMPT_FAILOVER;   /* pre-relay failure */
}

/* Pick the instance to try first. With a session bound to a live instance of the
 * requested model (and not excluded), reuse it — that instance's prefix cache
 * holds the session's KV. Otherwise pick least-busy. Returns true on a pick. */
static bool pick_instance(reg_registry *reg, session_affinity *sa,
                          const char *model, const char *session_id,
                          const char *const *excluded, int n_excluded,
                          reg_instance *out)
{
    char bound[RW_INSTANCE_ID_MAX];
    if (sa && session_id && session_id[0] &&
        sa_lookup(sa, session_id, bound, sizeof(bound))) {
        bool skip = false;
        for (int x = 0; x < n_excluded; x++)
            if (excluded[x] && strcmp(excluded[x], bound) == 0) { skip = true; break; }
        if (!skip && reg_find_live(reg, model, bound, out))
            return true;
        /* Bound instance is gone/unhealthy/excluded → fall through to least-busy. */
    }
    return reg_pick_least_busy(reg, model, excluded, n_excluded, out);
}

void router_handle_completion(int client_fd, reg_registry *reg,
                              session_affinity *sa, const char *path,
                              const char *body, const char *session_id)
{
    if (!body) {
        hs_send_error(client_fd, 400, "invalid_request", "missing request body");
        return;
    }
    jr_value *root = jr_parse(body);
    if (!root || jr_kind(root) != JR_OBJECT) {
        jr_free(root);
        hs_send_error(client_fd, 400, "invalid_request", "malformed JSON body");
        return;
    }
    const char *model = jr_as_string(jr_object_get(root, "model"));
    bool streaming = jr_as_bool(jr_object_get(root, "stream"), false);
    /* A body "session_id" overrides the X-Session-Id header (matches the serve's
     * openai_chat.c precedence), so affinity keys on the SAME id the serve caches
     * under. Copy both out before the parse tree is freed. */
    const char *body_sid = pn_body_session_key(root);
    char model_copy[RW_MODEL_ID_MAX];
    char sid_copy[SA_SESSION_ID_MAX];
    sid_copy[0] = '\0';
    if (body_sid && body_sid[0])
        snprintf(sid_copy, sizeof(sid_copy), "%s", body_sid);
    else if (session_id && session_id[0])
        snprintf(sid_copy, sizeof(sid_copy), "%s", session_id);
    if (!model) {
        jr_free(root);
        hs_send_error(client_fd, 400, "invalid_request",
                      "missing 'model' field");
        return;
    }
    snprintf(model_copy, sizeof(model_copy), "%s", model);
    jr_free(root);   /* done with the parse tree; body stays valid for the proxy */

    size_t body_len = strlen(body);
    const char *sid = sid_copy[0] ? sid_copy : NULL;

    /* Forward the effective session id upstream so the serve keys its prefix
     * cache on it (the header the serve's http_server.c reads). */
    char extra_hdrs[SA_SESSION_ID_MAX + 32];
    const char *fwd_hdrs = NULL;
    if (sid) {
        snprintf(extra_hdrs, sizeof(extra_hdrs), "X-Session-Id: %s\r\n", sid);
        fwd_hdrs = extra_hdrs;
    }

    /* Failover loop: try up to ROUTER_MAX_TRIES distinct instances, excluding
     * ones that already failed pre-relay. The first pick honours affinity. */
    const char *excluded[ROUTER_MAX_TRIES];
    int n_excluded = 0;

    for (int attempt = 0; attempt < ROUTER_MAX_TRIES; attempt++) {
        reg_instance pick;
        if (!pick_instance(reg, sa, model_copy, sid, excluded, n_excluded, &pick)) {
            hs_send_error(client_fd, 503, "model_unavailable",
                          "no live instance for the requested model");
            return;
        }

        attempt_outcome oc = proxy_to_instance(client_fd, &pick, path, body,
                                               body_len, streaming, fwd_hdrs);
        if (oc == ATTEMPT_TERMINAL) {
            /* Bind the session to the instance that served it, so its next turn
             * reuses this instance's warm prefix cache. */
            if (sid)
                sa_record(sa, sid, pick.instance_id);
            return;
        }

        /* Record the failed instance so we don't re-pick it, then retry. */
        if (n_excluded < ROUTER_MAX_TRIES) {
            /* pick.instance_id lives on the stack; stash a stable copy. The
             * excluded[] entries must outlive the loop iteration, so keep a
             * parallel static-lifetime store. */
            static _Thread_local char store[ROUTER_MAX_TRIES][RW_INSTANCE_ID_MAX];
            snprintf(store[n_excluded], RW_INSTANCE_ID_MAX, "%s", pick.instance_id);
            excluded[n_excluded] = store[n_excluded];
            n_excluded++;
        }
    }

    /* Exhausted all attempts with only pre-relay failures. */
    hs_send_error(client_fd, 503, "model_unavailable",
                  "all live instances failed to respond");
}
