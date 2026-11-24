/* SPDX-License-Identifier: AGPL-3.0-or-later */
/* Copyright (C) 2026 David (Corey) Herrera */
/**
 * @file   local_session_routes.c
 * @brief  Implementation of the client registry routes (see local_session_routes.h).
 *         Never logs a ticket, host_secret, host_pub or a title.
 *
 * @status None
 * @issues None
 * @todo   None
 */

#include "local_session_routes.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

#include "../../lib/monocypher/monocypher.h"
#include "../serve/json_reader.h"
#include "../serve/json_writer.h"
#include "attach_ticket.h"
#include "hosts_hub.h"
#include "key_store.h"
#include "local_caller.h"
#include "local_people_wire.h"
#include "people_store.h"
#include "session_registry.h"
#include "sha256.h"

#define SESSIONS_PATH   "/local/sessions"
#define SESSIONS_PREFIX "/local/sessions/"

/* What the list writer looks names up in: people and device names, read once. */
typedef struct {
    pn_person    people[PN_PEOPLE_MAX];
    int          n_people;
    pn_key_info *keys;
    int          n_keys;
} names_snapshot;

static const char *device_name_of(const names_snapshot *ns, const char *device_id)
{
    for (int i = 0; device_id[0] && i < ns->n_keys; i++)
        if (ns->keys[i].kind == PN_KEY_KIND_DEVICE && strcmp(ns->keys[i].device_id, device_id) == 0)
            return ns->keys[i].device_name;
    return "";
}

static const char *handle_of(const names_snapshot *ns, const char *person)
{
    for (int i = 0; i < ns->n_people; i++)
        if (strcmp(ns->people[i].id, person) == 0)
            return ns->people[i].handle;
    return "";
}

static void write_shares(json_writer *jw, const names_snapshot *ns, const pn_reg_row_info *row)
{
    jw_raw(jw, "[");
    for (int s = 0; s < row->n_shares; s++) {
        jw_printf(jw, "%s{\"person\":\"%s\",\"handle\":", s ? "," : "", row->shares[s].person);
        jw_string(jw, handle_of(ns, row->shares[s].person));
        jw_printf(jw, ",\"role\":\"%s\"}", pn_reg_role_name(row->shares[s].role));
    }
    jw_raw(jw, "]");
}

/* One list row (spec §4.5). The share list goes only to the owner; others get []. */
static void write_session(json_writer *jw, const names_snapshot *ns, const pn_reg_row_info *row,
                          pn_reg_role role)
{
    static const pn_reg_row_info no_shares = {0};

    jw_printf(jw, "{\"key\":\"%s\",\"kind\":\"%s\",\"owner\":", row->key,
              pn_reg_kind_name(row->kind));
    pn_local_write_person_short_of(jw, ns->people, ns->n_people, row->owner);
    jw_printf(jw, ",\"origin_device\":{\"id\":\"%s\",\"name\":", row->origin_device);
    jw_string(jw, device_name_of(ns, row->origin_device));
    jw_raw(jw, "},\"title\":");
    jw_string(jw, row->title);
    jw_raw(jw, ",\"model\":");
    jw_string(jw, row->model);
    jw_printf(jw, ",\"instance\":%d,\"state\":\"%s\",\"role\":\"%s\",\"shares\":", row->instance,
              pn_reg_state_name(row->state), pn_reg_role_name(role));
    write_shares(jw, ns, role == PN_REG_ROLE_OWNER ? row : &no_shares);
    jw_printf(jw, ",\"registered_at\":%lld,\"last_beat\":%lld}", row->registered_at,
              row->last_beat);
}

/* ?rev=<n>: 1 with *rev set, 0 when absent, -1 when malformed. */
static int query_rev(const char *path, unsigned long long *rev)
{
    const char *q = strchr(path, '?'), *p = q ? strstr(q, "rev=") : NULL;
    char *end = NULL;

    if (!p)
        return 0;
    if ((p != q + 1 && p[-1] != '&') || p[4] < '0' || p[4] > '9')
        return -1;
    *rev = strtoull(p + 4, &end, 10);
    return *end && *end != '&' ? -1 : 1;
}

/* GET /local/sessions[?rev=n] */
static void handle_list(int cfd, pn_local_ctx *ctx, const pn_local_caller *c, const char *path)
{
    /* rev before the rows: a change in between costs one more fetch, never a miss */
    unsigned long long rev = pn_reg_rev(ctx->registry), asked = 0;
    int has_rev = query_rev(path, &asked), n = 0;
    names_snapshot *ns = NULL;
    pn_reg_row_info *rows = NULL;
    json_writer jw;
    char out[96];

    if (has_rev < 0) {
        hs_send_error(cfd, 400, "invalid_request_error", "rev is a non-negative integer");
        return;
    }
    if (has_rev && asked == rev) {
        snprintf(out, sizeof(out), "{\"rev\":%llu,\"unchanged\":true}", rev);
        hs_send_json(cfd, 200, out);
        return;
    }
    ns = calloc(1, sizeof(*ns));
    rows = calloc(PN_REG_MAX_ROWS, sizeof(*rows));
    if (ns)
        ns->keys = calloc(PN_KEY_MAX_KEYS, sizeof(*ns->keys));
    if (!ns || !rows || !ns->keys || !jw_init(&jw)) {
        hs_send_error(cfd, 500, "server_error", "oom");
        goto done;
    }
    ns->n_people = pn_people_list(ctx->people, ns->people, PN_PEOPLE_MAX);
    ns->n_keys = ctx->keys ? pn_key_store_list(ctx->keys, ns->keys, PN_KEY_MAX_KEYS) : 0;
    ns->n_people = ns->n_people < 0 ? 0 : ns->n_people;
    ns->n_keys = ns->n_keys < 0 ? 0 : ns->n_keys;
    n = pn_reg_list_for(ctx->registry, c->person, rows, PN_REG_MAX_ROWS);
    jw_printf(&jw, "{\"rev\":%llu,\"sessions\":[", rev);
    for (int i = 0; i < n; i++) {
        pn_reg_role role = strcmp(rows[i].owner, c->person) == 0 ? PN_REG_ROLE_OWNER
                                                                  : PN_REG_ROLE_NONE;
        for (int s = 0; role == PN_REG_ROLE_NONE && s < rows[i].n_shares; s++)
            if (strcmp(rows[i].shares[s].person, c->person) == 0)
                role = rows[i].shares[s].role;
        jw_raw(&jw, i ? "," : "");
        write_session(&jw, ns, &rows[i], role);
    }
    jw_raw(&jw, "]}");
    hs_send_json(cfd, 200, jw.data);
    jw_free(&jw);
done:
    if (ns)
        free(ns->keys);
    free(ns);
    free(rows);
}

static void send_share_error(int cfd, int rc)
{
    if (rc == PN_REG_E_NOT_FOUND)
        hs_send_error(cfd, 404, "not_found", "no registered session with that key");
    else if (rc == PN_REG_E_NOT_OWNER)
        hs_send_error(cfd, 403, "forbidden", "only the owner shares a session");
    else if (rc == PN_REG_E_NOT_SHAREABLE)
        hs_send_error(cfd, 400, "not_shareable", "a desktop row is never shared");
    else if (rc == PN_REG_E_SELF_SHARE)
        hs_send_error(cfd, 400, "invalid_request_error", "a session is not shared with its owner");
    else if (rc == PN_REG_E_SHARES_FULL)
        hs_send_error(cfd, 507, "shares_full", "a session has at most 16 shares");
    else
        hs_send_error(cfd, 400, "invalid_request_error", "role is view | control | none");
}

/* The share target from {"person"} or {"handle"}; -1 with an answer sent. */
static int share_target(int cfd, pn_local_ctx *ctx, const jr_value *b, pn_person *p)
{
    const char *id = jr_as_string(jr_object_get(b, "person"));
    const char *handle = jr_as_string(jr_object_get(b, "handle"));
    int rc = PN_PEOPLE_E_NOT_FOUND;

    if (!id && !handle) {
        hs_send_error(cfd, 400, "invalid_request_error", "send person or handle");
        return -1;
    }
    rc = id ? pn_people_get(ctx->people, id, p) : pn_people_find_handle(ctx->people, handle, p);
    if (rc != PN_PEOPLE_OK) {
        hs_send_error(cfd, 404, "not_found", "no such person on this box");
        return -1;
    }
    return 0;
}

/* POST /local/sessions/<key>/share */
static void handle_share(int cfd, pn_local_ctx *ctx, const pn_local_caller *c, const char *key,
                         const jr_value *b)
{
    const char *role_s = jr_as_string(jr_object_get(b, "role"));
    pn_reg_role role = PN_REG_ROLE_NONE;
    names_snapshot *ns = NULL;
    pn_reg_row_info row;
    json_writer jw;
    pn_person p;
    int rc = 0;

    if (!role_s || pn_reg_role_parse(role_s, &role) != 0 || role == PN_REG_ROLE_OWNER) {
        send_share_error(cfd, PN_REG_E_INVALID);
        return;
    }
    if (share_target(cfd, ctx, b, &p) != 0)
        return;
    rc = pn_reg_share(ctx->registry, key, c->person, p.id, role);
    if (rc != PN_REG_OK && rc != PN_REG_E_IO) {
        send_share_error(cfd, rc);
        return;
    }
    pn_hub_notify(ctx->hub, key);
    if (rc == PN_REG_E_IO)
        fprintf(stderr, "local: session %s shares changed; sessions.json save failed "
                "(retried)\n", key);
    if (role == PN_REG_ROLE_NONE)
        fprintf(stderr, "local: session %s unshared from %s by key %s\n", key, p.id, c->key_id);
    else
        fprintf(stderr, "local: session %s shared with %s as %s by key %s\n", key, p.id,
                pn_reg_role_name(role), c->key_id);
    ns = calloc(1, sizeof(*ns));
    if (!ns || pn_reg_get(ctx->registry, key, &row) != PN_REG_OK || !jw_init(&jw)) {
        hs_send_error(cfd, 500, "server_error", "the row vanished");
        free(ns);
        return;
    }
    ns->n_people = pn_people_list(ctx->people, ns->people, PN_PEOPLE_MAX);
    ns->n_people = ns->n_people < 0 ? 0 : ns->n_people;
    jw_printf(&jw, "{\"key\":\"%s\",\"shares\":", key);
    write_shares(&jw, ns, &row);
    jw_raw(&jw, "}");
    hs_send_json(cfd, 200, jw.data);
    jw_free(&jw);
    free(ns);
}

/* The body's cpub, lowercased into out[65]; -1 when missing or not 64 hex. */
static int read_cpub(const jr_value *b, char out[PN_TICKET_CPUB_MAX])
{
    const char *s = jr_as_string(jr_object_get(b, "cpub"));
    uint8_t raw[32];

    if (!s || strlen(s) != 64 || pn_hex_decode(s, raw, sizeof(raw)) != 0)
        return -1;
    pn_hex_encode(raw, sizeof(raw), out);
    return 0;
}

static pn_ticket_role ticket_role_of(pn_reg_role role)
{
    return role == PN_REG_ROLE_OWNER ? PN_TICKET_ROLE_OWNER
         : role == PN_REG_ROLE_CONTROL ? PN_TICKET_ROLE_CONTROL : PN_TICKET_ROLE_VIEW;
}

/* The row checks of an attach; 0, or -1 with the answer sent. */
static int attach_allowed(int cfd, int rc, const pn_reg_row_info *row, pn_reg_role role,
                          const char *secret_hex)
{
    if (rc != PN_REG_OK) {
        hs_send_error(cfd, 404, "not_found", "no registered session with that key");
        return -1;
    }
    if (role == PN_REG_ROLE_NONE ||
        (row->kind == PN_REG_KIND_DESKTOP && role != PN_REG_ROLE_OWNER)) {
        hs_send_error(cfd, 403, "forbidden", "this session is not shared with you");
        return -1;
    }
    if (row->state == PN_REG_STATE_DISCONNECTED || strlen(secret_hex) != 64) {
        hs_send_error(cfd, 409, "disconnected", "the session's host is not connected");
        return -1;
    }
    return 0;
}

/* Mint the ticket and answer; the secret's bytes are wiped here. */
static void send_attach(int cfd, const pn_local_caller *c, const pn_reg_row_info *row,
                        pn_reg_role role, const char *secret_hex, const char *pub,
                        const char *cpub)
{
    uint8_t secret[PN_TICKET_SECRET_LEN];
    char ticket[PN_TICKET_MAX];
    json_writer jw;
    int rc = pn_hex_decode(secret_hex, secret, sizeof(secret)) == 0
           ? pn_ticket_mint(secret, row->key, c->person, c->device_id, cpub,
                            ticket_role_of(role), (long long)time(NULL), NULL, ticket,
                            sizeof(ticket))
           : PN_TICKET_E_INVALID;

    crypto_wipe(secret, sizeof(secret));
    if (rc != PN_TICKET_OK || !jw_init(&jw)) {
        fprintf(stderr, "local: attach %s: no ticket (%s)\n", row->key, pn_ticket_result_name(rc));
        hs_send_error(cfd, 500, "server_error", "could not mint a ticket");
        crypto_wipe(ticket, sizeof(ticket));
        return;
    }
    jw_raw(&jw, "{\"host_addr\":");
    jw_string(&jw, row->host_addr);
    jw_printf(&jw, ",\"host_pub\":\"%s\",\"ticket\":\"%s\",\"role\":\"%s\",\"expires_in\":%d}",
              pub, ticket, pn_reg_role_name(role), PN_TICKET_TTL_S);
    hs_send_json(cfd, 200, jw.data);
    crypto_wipe(jw.data, jw.len);
    jw_free(&jw);
    crypto_wipe(ticket, sizeof(ticket));
    fprintf(stderr, "local: attach %s as %s by key %s\n", row->key, pn_reg_role_name(role),
            c->key_id);
}

/* POST /local/sessions/<key>/attach */
static void handle_attach(int cfd, pn_local_ctx *ctx, const pn_local_caller *c, const char *key,
                          const jr_value *b)
{
    char cpub[PN_TICKET_CPUB_MAX], secret_hex[PN_REG_HEX32_MAX] = "", pub[PN_REG_HEX32_MAX] = "";
    pn_reg_role role = PN_REG_ROLE_NONE;
    pn_reg_row_info row;
    int rc = 0;

    if (read_cpub(b, cpub) != 0) {
        hs_send_error(cfd, 400, "bad_cpub", "cpub is the client's X25519 public key, 64 hex");
        return;
    }
    if (!c->device_id[0]) {
        hs_send_error(cfd, 409, "device_unclaimed", "this device key has no device_id yet "
                      "(POST /local/me/device first)");
        return;
    }
    rc = pn_reg_attach_info(ctx->registry, key, c->person, &row, &role, secret_hex, pub);
    if (attach_allowed(cfd, rc, &row, role, secret_hex) == 0)
        send_attach(cfd, c, &row, role, secret_hex, pub, cpub);
    crypto_wipe(secret_hex, sizeof(secret_hex));
}

/* The <key> of /local/sessions/<key>/<action>, copied; -1 when it is not that shape. */
static int path_key(const char *path, const char *action, char *key, size_t cap)
{
    const char *k = path + strlen(SESSIONS_PREFIX);
    size_t n = strcspn(k, "/?"), alen = strlen(action);

    if (n == 0 || n >= cap || k[n] != '/' || strncmp(k + n + 1, action, alen) != 0 ||
        (k[n + 1 + alen] != '\0' && k[n + 1 + alen] != '?'))
        return -1;
    memcpy(key, k, n);
    key[n] = '\0';
    return 0;
}

int pn_local_session_routes(int cfd, pn_local_ctx *ctx, const hs_request *req)
{
    const pn_local_caller *c = pn_local_caller_get();
    const int get = strcmp(req->method, "GET") == 0, post = strcmp(req->method, "POST") == 0;
    size_t plen = strlen(SESSIONS_PATH);
    char key[PN_REG_KEY_MAX];
    jr_value *body = NULL;
    int list = 0, share = 0, attach = 0;

    list = get && strncmp(req->path, SESSIONS_PATH, plen) == 0 &&
           (req->path[plen] == '\0' || req->path[plen] == '?');
    if (!list && post && strncmp(req->path, SESSIONS_PREFIX, plen + 1) == 0) {
        share = path_key(req->path, "share", key, sizeof(key)) == 0;
        attach = path_key(req->path, "attach", key, sizeof(key)) == 0;
    }
    if (!list && !share && !attach)
        return 0;
    if (!c || !ctx->registry || !ctx->hub || !ctx->people) {
        hs_send_error(cfd, c ? 503 : 401, c ? "unavailable" : "unauthorized",
                      c ? "the registry is not open" : "a device key is required");
        return 1;
    }
    if (list) {
        handle_list(cfd, ctx, c, req->path);
        return 1;
    }
    body = req->body ? jr_parse(req->body) : NULL;
    if (!body || jr_kind(body) != JR_OBJECT)
        hs_send_error(cfd, 400, "invalid_request_error", "a JSON object body is required");
    else if (share)
        handle_share(cfd, ctx, c, key, body);
    else
        handle_attach(cfd, ctx, c, key, body);
    jr_free(body);
    return 1;
}
