/* SPDX-License-Identifier: AGPL-3.0-or-later */
/* Copyright (C) 2026 David (Corey) Herrera */
/**
 * @file   local_host_routes.c
 * @brief  Implementation of the host registry routes (see local_host_routes.h).
 *         Never logs host_secret, host_pub or titles.
 *
 * @status None
 * @issues None
 * @todo   None
 */

#include "local_host_routes.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

#include <arpa/inet.h>
#include <netinet/in.h>
#include <sys/socket.h>

#include "../serve/json_reader.h"
#include "../serve/json_writer.h"
#include "hosts_hub.h"
#include "local_caller.h"
#include "local_people_wire.h"
#include "people_store.h"
#include "session_registry.h"

#define HOSTS_PREFIX "/local/hosts/"

/* The registration's peer IP + host_port, the address LAN clients can reach. */
static int peer_addr(int cfd, int port, char *out, size_t cap)
{
    struct sockaddr_storage ss;
    socklen_t len = sizeof(ss);
    char ip[INET6_ADDRSTRLEN] = "";

    if (port < 1 || port > 65535 || getpeername(cfd, (struct sockaddr *)&ss, &len) != 0)
        return -1;
    if (ss.ss_family == AF_INET) {
        const struct sockaddr_in *a = (const struct sockaddr_in *)&ss;
        if (!inet_ntop(AF_INET, &a->sin_addr, ip, sizeof(ip)))
            return -1;
        return snprintf(out, cap, "%s:%d", ip, port) < (int)cap ? 0 : -1;
    }
    if (ss.ss_family == AF_INET6) {
        const struct sockaddr_in6 *a = (const struct sockaddr_in6 *)&ss;
        if (IN6_IS_ADDR_V4MAPPED(&a->sin6_addr)) {
            if (!inet_ntop(AF_INET, &a->sin6_addr.s6_addr[12], ip, sizeof(ip)))
                return -1;
            return snprintf(out, cap, "%s:%d", ip, port) < (int)cap ? 0 : -1;
        }
        if (!inet_ntop(AF_INET6, &a->sin6_addr, ip, sizeof(ip)))
            return -1;
        return snprintf(out, cap, "[%s]:%d", ip, port) < (int)cap ? 0 : -1;
    }
    return -1;
}

static void write_person_short_id(json_writer *jw, pn_local_ctx *ctx, const char *id)
{
    pn_person p;

    if (pn_people_get(ctx->people, id, &p) == PN_PEOPLE_OK) {
        pn_local_write_person_short(jw, &p);
        return;
    }
    jw_printf(jw, "{\"id\":\"%s\",\"handle\":\"\",\"name\":\"\"}", id);
}

/* The registration answer (spec §4.4). */
static void send_registered(int cfd, pn_local_ctx *ctx, const char *key, long long seq)
{
    pn_reg_row_info row;
    json_writer jw;
    pn_person p;

    if (pn_reg_get(ctx->registry, key, &row) != PN_REG_OK || !jw_init(&jw)) {
        hs_send_error(cfd, 500, "server_error", "the row vanished");
        return;
    }
    jw_printf(&jw, "{\"key\":\"%s\",\"owner\":", row.key);
    write_person_short_id(&jw, ctx, row.owner);
    jw_raw(&jw, ",\"host_addr\":");
    jw_string(&jw, row.host_addr);
    jw_printf(&jw, ",\"seq\":%lld,\"shares\":[", seq);
    for (int s = 0; s < row.n_shares; s++) {
        int known = pn_people_get(ctx->people, row.shares[s].person, &p) == PN_PEOPLE_OK;
        jw_printf(&jw, "%s{\"person\":\"%s\",\"handle\":", s ? "," : "", row.shares[s].person);
        jw_string(&jw, known ? p.handle : "");
        jw_printf(&jw, ",\"role\":\"%s\"}", pn_reg_role_name(row.shares[s].role));
    }
    jw_printf(&jw, "],\"beat_s\":%d,\"disconnect_s\":%d}", ctx->host_beat_s,
              ctx->host_disconnect_s);
    hs_send_json(cfd, 200, jw.data);
    jw_free(&jw);
}

static void send_reg_error(int cfd, int rc)
{
    if (rc == PN_REG_E_KEY_TAKEN)
        hs_send_error(cfd, 409, "key_taken", "another person registered this key");
    else if (rc == PN_REG_E_FULL)
        hs_send_error(cfd, 507, "registry_full", "the registry is full; remove disconnected "
                      "sessions");
    else if (rc == PN_REG_E_INVALID)
        hs_send_error(cfd, 400, "invalid_request_error", "a registry field breaks the rules "
                      "(key c- + 12 hex, title <= 256 bytes, model <= 64, instance 0-64, "
                      "64-hex host_secret and host_pub)");
    else
        hs_send_error(cfd, 507, "registry_error", "the registry could not be saved");
}

/* The string fields and state of a registration; -1 with a 400 sent. */
static int read_host_fields(int cfd, const jr_value *b, pn_reg_host *h)
{
    const char *kind = jr_as_string(jr_object_get(b, "kind"));
    const char *state = jr_as_string(jr_object_get(b, "state"));
    double instance = jr_as_number(jr_object_get(b, "instance"), 0);

    h->key = jr_as_string(jr_object_get(b, "key"));
    h->title = jr_as_string(jr_object_get(b, "title"));
    h->model = jr_as_string(jr_object_get(b, "model"));
    h->host_secret = jr_as_string(jr_object_get(b, "host_secret"));
    h->host_pub = jr_as_string(jr_object_get(b, "host_pub"));
    h->instance = instance == (int)instance ? (int)instance : -1;
    if (pn_reg_kind_parse(kind ? kind : "session", &h->kind) != 0 || !state ||
        pn_reg_state_parse(state, &h->state) != 0 || h->state == PN_REG_STATE_DISCONNECTED) {
        hs_send_error(cfd, 400, "invalid_request_error", "kind is session | desktop; state is "
                      "thinking | needs-you | idle (disconnected is set by the box)");
        return -1;
    }
    return 0;
}

/* POST /local/hosts */
static void handle_register(int cfd, pn_local_ctx *ctx, const pn_local_caller *c,
                            const jr_value *b)
{
    const char *origin = jr_as_string(jr_object_get(b, "origin_device"));
    const char *explicit_addr = jr_as_string(jr_object_get(b, "host_addr"));
    char addr[PN_REG_ADDR_BYTES + 1] = "";
    pn_reg_host h = { .owner = c->person };
    long long seq = 0;
    int rc = 0;

    if (!c->device_id[0]) {
        hs_send_error(cfd, 409, "device_unclaimed", "this device key has no device_id yet "
                      "(POST /local/me/device first)");
        return;
    }
    if (!origin || strcmp(origin, c->device_id) != 0) {
        hs_send_error(cfd, 409, "device_mismatch", "origin_device is not this key's device");
        return;
    }
    if (read_host_fields(cfd, b, &h) != 0)
        return;
    if (explicit_addr)
        snprintf(addr, sizeof(addr), "%s", explicit_addr);
    else if (peer_addr(cfd, (int)jr_as_number(jr_object_get(b, "host_port"), 0), addr,
                       sizeof(addr)) != 0) {
        hs_send_error(cfd, 400, "invalid_request_error", "send host_port (1-65535) or host_addr");
        return;
    }
    h.origin_device = origin;
    h.host_addr = addr;
    rc = pn_reg_upsert(ctx->registry, &h, (long long)time(NULL), &seq);
    if (rc != PN_REG_OK && rc != PN_REG_E_IO) {
        send_reg_error(cfd, rc);
        return;
    }
    if (rc == PN_REG_E_IO)
        fprintf(stderr, "local: host %s registered; sessions.json save failed (retried)\n", h.key);
    send_registered(cfd, ctx, h.key, seq);
}

/* The <key> of /local/hosts/<key>/<action>, copied; -1 when it is not that shape. */
static int path_key(const char *path, const char *action, char *key, size_t cap)
{
    const char *k = path + strlen(HOSTS_PREFIX);
    size_t n = strcspn(k, "/?"), alen = strlen(action);

    if (n == 0 || n >= cap || k[n] != '/' || strncmp(k + n + 1, action, alen) != 0 ||
        (k[n + 1 + alen] != '\0' && k[n + 1 + alen] != '?'))
        return -1;
    memcpy(key, k, n);
    key[n] = '\0';
    return 0;
}

/* ?since=<n> (default 0); -1 when malformed. */
static long long query_since(const char *path)
{
    const char *q = strchr(path, '?'), *p = q ? strstr(q, "since=") : NULL;
    char *end = NULL;
    long long v = 0;

    if (!p)
        return 0;
    if (p != q + 1 && p[-1] != '&')
        return -1;
    v = strtoll(p + 6, &end, 10);
    return end == p + 6 || (*end && *end != '&') || v < 0 ? -1 : v;
}

/* GET /local/hosts/<key>/wait?since=n: 2 when the hub took the fd. Not logged
 * (one per host every 30 s). */
static int handle_wait(int cfd, pn_local_ctx *ctx, const pn_local_caller *c, const char *key,
                       const char *path)
{
    long long since = query_since(path);
    int rc = 0;

    if (since < 0) {
        hs_send_error(cfd, 400, "invalid_request_error", "since is a non-negative integer");
        return 1;
    }
    rc = pn_reg_beat(ctx->registry, key, c->person, since, (long long)time(NULL));
    if (rc == PN_REG_E_NOT_FOUND) {
        hs_send_error(cfd, 404, "not_registered", "no session with that key; register it again");
        return 1;
    }
    if (rc == PN_REG_E_NOT_OWNER) {
        hs_send_error(cfd, 403, "forbidden", "only the owner waits on a session");
        return 1;
    }
    if (pn_hub_add_wait(ctx->hub, cfd, key, c->key_id, since) != 0) {
        hs_send_error_ex(cfd, 503, "hub_full", "too many held waits; retry shortly",
                         "Retry-After: 1\r\n");
        return 1;
    }
    return 2;
}

/* POST /local/hosts/<key>/delete */
static void handle_delete(int cfd, pn_local_ctx *ctx, const pn_local_caller *c, const char *key)
{
    char out[64];
    int rc = pn_reg_delete(ctx->registry, key, c->person, (long long)time(NULL));

    if (rc == PN_REG_E_NOT_FOUND) {
        hs_send_error(cfd, 404, "not_found", "no registered session with that key");
        return;
    }
    if (rc == PN_REG_E_NOT_OWNER) {
        hs_send_error(cfd, 403, "forbidden", "only the owner deletes a session");
        return;
    }
    pn_hub_notify(ctx->hub, key);
    if (rc == PN_REG_E_IO)
        fprintf(stderr, "local: host %s deleted; sessions.json save failed (retried)\n", key);
    fprintf(stderr, "local: host %s deleted by key %s\n", key, c->key_id);
    snprintf(out, sizeof(out), "{\"deleted\":\"%s\"}", key);
    hs_send_json(cfd, 200, out);
}

int pn_local_host_routes(int cfd, pn_local_ctx *ctx, const hs_request *req)
{
    const pn_local_caller *c = pn_local_caller_get();
    const int get = strcmp(req->method, "GET") == 0, post = strcmp(req->method, "POST") == 0;
    char key[PN_REG_KEY_MAX];
    jr_value *body = NULL;
    int wait = 0, del = 0, reg = 0;

    reg = post && strcmp(req->path, "/local/hosts") == 0;
    if (!reg && strncmp(req->path, HOSTS_PREFIX, strlen(HOSTS_PREFIX)) == 0) {
        wait = get && path_key(req->path, "wait", key, sizeof(key)) == 0;
        del = post && path_key(req->path, "delete", key, sizeof(key)) == 0;
    }
    if (!reg && !wait && !del)
        return 0;
    if (!c || !ctx->registry || !ctx->hub || !ctx->people) {
        hs_send_error(cfd, c ? 503 : 401, c ? "unavailable" : "unauthorized",
                      c ? "the registry is not open" : "a device key is required");
        return 1;
    }
    if (wait)
        return handle_wait(cfd, ctx, c, key, req->path);
    if (del) {
        handle_delete(cfd, ctx, c, key);
        return 1;
    }
    body = req->body ? jr_parse(req->body) : NULL;
    if (!body || jr_kind(body) != JR_OBJECT)
        hs_send_error(cfd, 400, "invalid_request_error", "a JSON object body is required");
    else
        handle_register(cfd, ctx, c, body);
    jr_free(body);
    return 1;
}

int pn_local_hosts_person_removed(pn_local_ctx *ctx, const char *person)
{
    char keys[PN_REG_MAX_ROWS][PN_REG_KEY_MAX];
    int removed = 0, n = 0;

    if (!ctx->registry)
        return 0;
    n = pn_reg_remove_person(ctx->registry, person, (long long)time(NULL), keys,
                             PN_REG_MAX_ROWS, &removed);
    for (int i = 0; i < n; i++)
        pn_hub_notify(ctx->hub, keys[i]);
    return removed;
}

void pn_local_hosts_key_revoked(pn_local_ctx *ctx, const char *key_id)
{
    int n = ctx->hub ? pn_hub_close_device(ctx->hub, key_id, PN_REG_CLOSE_REVOKED) : 0;

    if (n)
        fprintf(stderr, "local: key %s revoked; %d held host wait%s closed\n", key_id, n,
                n == 1 ? "" : "s");
}
