/* SPDX-License-Identifier: AGPL-3.0-or-later */
/* Copyright (C) 2026 David (Corey) Herrera */
/**
 * @file   local_people_routes.c
 * @brief  Implementation of the me / device / people routes (see local_people_routes.h).
 *
 * @status None
 * @issues None
 * @todo   None
 */

#include "local_people_routes.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

#include "../serve/json_reader.h"
#include "../serve/json_writer.h"
#include "invites.h"
#include "key_store.h"
#include "local_caller.h"
#include "local_host_routes.h"
#include "local_people_wire.h"
#include "people_store.h"

static void send_jw(int cfd, json_writer *jw)
{
    hs_send_json(cfd, 200, jw->data);
    jw_free(jw);
}

static void write_device(json_writer *jw, const pn_key_info *k)
{
    jw_printf(jw, "{\"key_id\":\"%s\",\"device_id\":\"%s\",\"device_name\":", k->id,
              k->device_id);
    jw_string(jw, k->device_name);
    jw_raw(jw, "}");
}

/* GET /local/me */
static void handle_me(int cfd, pn_local_ctx *ctx, const pn_local_caller *c)
{
    pn_person person;
    pn_key_info key;
    json_writer jw;

    if (pn_people_get(ctx->people, c->person, &person) != PN_PEOPLE_OK ||
        pn_key_store_get(ctx->keys, c->key_id, &key) != PN_KEY_OK) {
        hs_send_error(cfd, 401, "unauthorized", "this key or its person is gone");
        return;
    }
    if (!jw_init(&jw)) {
        hs_send_error(cfd, 500, "server_error", "oom");
        return;
    }
    jw_raw(&jw, "{\"person\":");
    pn_local_write_person_full(&jw, &person);
    jw_raw(&jw, ",\"device\":");
    write_device(&jw, &key);
    jw_raw(&jw, "}");
    send_jw(cfd, &jw);
}

/* POST /local/me/device {device_id?, device_name?} */
static void handle_me_device(int cfd, pn_local_ctx *ctx, const pn_local_caller *c,
                             const jr_value *body)
{
    const char *device_id = jr_as_string(jr_object_get(body, "device_id"));
    const char *device_name = jr_as_string(jr_object_get(body, "device_name"));
    pn_key_info key;
    json_writer jw;
    int rc = 0;

    if (!device_id && !device_name) {
        hs_send_error(cfd, 400, "invalid_request_error", "send device_id and/or device_name");
        return;
    }
    rc = pn_key_store_set_device(ctx->keys, c->key_id, device_id, device_name, &key);
    if (rc == PN_KEY_E_DEVICE_MISMATCH) {
        hs_send_error(cfd, 409, "device_id_mismatch",
                      "this key already belongs to another device_id");
        return;
    }
    if (rc != PN_KEY_OK) {
        hs_send_error(cfd, rc == PN_KEY_E_INVALID ? 400 : rc == PN_KEY_E_NOT_FOUND ? 401 : 507,
                      rc == PN_KEY_E_INVALID ? "invalid_request_error"
                      : rc == PN_KEY_E_NOT_FOUND ? "unauthorized" : "key_store_error",
                      "device_id is \"d\" + 16 hex; device_name 1-64 bytes of text");
        return;
    }
    if (!jw_init(&jw)) {
        hs_send_error(cfd, 500, "server_error", "oom");
        return;
    }
    write_device(&jw, &key);
    fprintf(stderr, "local: key %s device %s (%s)\n", key.id, key.device_id, key.device_name);
    send_jw(cfd, &jw);
}

/* GET /local/devices[?all=1] */
static void handle_devices(int cfd, pn_local_ctx *ctx, const pn_local_caller *c,
                           const char *path)
{
    const int all = pn_local_query_flag(path, "all");
    pn_person people[PN_PEOPLE_MAX];
    pn_key_info *rows = NULL;
    json_writer jw;
    int n = 0, n_people = 0, written = 0;

    if (all && !c->admin) {
        hs_send_error(cfd, 403, "forbidden", "all=1 needs an admin");
        return;
    }
    rows = calloc(PN_KEY_MAX_KEYS, sizeof(*rows));
    if (!rows || !jw_init(&jw)) {
        free(rows);
        hs_send_error(cfd, 500, "server_error", "oom");
        return;
    }
    n = all ? pn_key_store_list(ctx->keys, rows, PN_KEY_MAX_KEYS)
            : pn_key_store_list_person(ctx->keys, c->person, rows, PN_KEY_MAX_KEYS);
    n_people = pn_people_list(ctx->people, people, PN_PEOPLE_MAX);
    jw_raw(&jw, "{\"devices\":[");
    for (int i = 0; i < n; i++) {
        const pn_key_info *k = &rows[i];
        if (k->kind != PN_KEY_KIND_DEVICE)
            continue;
        jw_printf(&jw, "%s{\"key_id\":\"%s\",\"device_id\":\"%s\",\"device_name\":",
                  written++ ? "," : "", k->id, k->device_id);
        jw_string(&jw, k->device_name);
        jw_raw(&jw, ",\"label\":");
        jw_string(&jw, k->label);
        jw_raw(&jw, ",\"person\":");
        pn_local_write_person_short_of(&jw, people, n_people, k->person);
        jw_printf(&jw, ",\"created\":%lld,\"this\":%s}", k->created,
                  strcmp(k->id, c->key_id) == 0 ? "true" : "false");
    }
    jw_raw(&jw, "]}");
    free(rows);
    send_jw(cfd, &jw);
}

static void send_people_error(int cfd, int result, const char *msg)
{
    const char *type = NULL;
    int status = pn_local_people_error(result, &type);

    hs_send_error(cfd, status, type, msg);
}

/* {"person":{full}} */
static void send_person(int cfd, const pn_person *p)
{
    json_writer jw;

    if (!jw_init(&jw)) {
        hs_send_error(cfd, 500, "server_error", "oom");
        return;
    }
    jw_raw(&jw, "{\"person\":");
    pn_local_write_person_full(&jw, p);
    jw_raw(&jw, "}");
    send_jw(cfd, &jw);
}

/* GET /local/people: full objects for an admin, short for anyone else. */
static void handle_people_list(int cfd, pn_local_ctx *ctx, const pn_local_caller *c)
{
    pn_person people[PN_PEOPLE_MAX];
    int n = pn_people_list(ctx->people, people, PN_PEOPLE_MAX);
    json_writer jw;

    if (!jw_init(&jw)) {
        hs_send_error(cfd, 500, "server_error", "oom");
        return;
    }
    jw_raw(&jw, "{\"people\":[");
    for (int i = 0; i < n; i++) {
        jw_raw(&jw, i ? "," : "");
        if (c->admin)
            pn_local_write_person_full(&jw, &people[i]);
        else
            pn_local_write_person_short(&jw, &people[i]);
    }
    jw_raw(&jw, "]}");
    send_jw(cfd, &jw);
}

/* A JSON boolean member as 0/1, -1 when absent; -2 when present but not a bool. */
static int bool_field(const jr_value *body, const char *name)
{
    const jr_value *v = jr_object_get(body, name);

    if (!v)
        return -1;
    return jr_kind(v) == JR_BOOL ? (jr_as_bool(v, false) ? 1 : 0) : -2;
}

/* POST /local/people {handle, name, may_load?} (ADMIN): a non-admin person. */
static void handle_people_add(int cfd, pn_local_ctx *ctx, const pn_local_caller *c,
                              const jr_value *body)
{
    int may_load = bool_field(body, "may_load");
    pn_person p;
    int rc = 0;

    if (may_load == -2) {
        hs_send_error(cfd, 400, "invalid_request_error", "may_load is true or false");
        return;
    }
    rc = pn_people_add(ctx->people, jr_as_string(jr_object_get(body, "handle")),
                       jr_as_string(jr_object_get(body, "name")), 0, NULL, may_load != 0, &p);
    if (rc != PN_PEOPLE_OK) {
        send_people_error(cfd, rc, "could not add that person (handle 1-24, name 1-64 bytes)");
        return;
    }
    fprintf(stderr, "local: person %s (%s) added by %s\n", p.id, p.handle, c->person);
    send_person(cfd, &p);
}

/* POST /local/people/<id> {name?, handle?, may_load?, admin?}: an admin changes
 * anyone (admin per the tree); anyone else only their own name. */
static void handle_people_update(int cfd, pn_local_ctx *ctx, const pn_local_caller *c,
                                 const char *id, const jr_value *body)
{
    pn_person_change ch = { jr_as_string(jr_object_get(body, "name")),
                            jr_as_string(jr_object_get(body, "handle")),
                            bool_field(body, "may_load"), bool_field(body, "admin") };
    pn_person p;
    int rc = 0;

    if (ch.may_load == -2 || ch.admin == -2) {
        hs_send_error(cfd, 400, "invalid_request_error", "may_load and admin are true or false");
        return;
    }
    if (!c->admin && (strcmp(id, c->person) != 0 || ch.handle || ch.may_load >= 0 ||
                      ch.admin >= 0)) {
        hs_send_error(cfd, 403, "forbidden", "only an admin changes people; you may rename "
                      "yourself");
        return;
    }
    rc = pn_people_update(ctx->people, c->person, id, &ch, &p);
    if (rc != PN_PEOPLE_OK) {
        send_people_error(cfd, rc, "could not change that person");
        return;
    }
    fprintf(stderr, "local: person %s changed by %s%s\n", p.id, c->person,
            ch.admin >= 0 ? (p.admin ? " (admin)" : " (not admin)") : "");
    send_person(cfd, &p);
}

/* POST /local/people/remove {id} (ADMIN): the person, their keys, invites and
 * registry rows (their held waits get close removed; shares to them are dropped). */
static void handle_people_remove(int cfd, pn_local_ctx *ctx, const pn_local_caller *c,
                                 const jr_value *body)
{
    const char *id = jr_as_string(jr_object_get(body, "id"));
    char out[160];
    int rc = pn_people_remove(ctx->people, c->person, id);
    int revoked = 0, sessions = 0;

    if (rc != PN_PEOPLE_OK) {
        send_people_error(cfd, rc, "could not remove that person");
        return;
    }
    revoked = pn_key_store_revoke_person(ctx->keys, id);
    pn_invites_void_person(ctx->invites, id);
    sessions = pn_local_hosts_person_removed(ctx, id);
    if (revoked < 0) {
        fprintf(stderr, "local: person %s removed, but revoking their keys failed\n", id);
        hs_send_error(cfd, 507, "key_store_error", "person removed; revoking their keys failed");
        return;
    }
    fprintf(stderr, "local: person %s removed by %s (%d keys revoked, %d sessions)\n", id,
            c->person, revoked, sessions);
    snprintf(out, sizeof(out), "{\"removed\":\"%s\",\"keys_revoked\":%d,\"sessions_removed\":%d}",
             id, revoked, sessions);
    hs_send_json(cfd, 200, out);
}

/* POST /local/invites {person?}: a one-time code for yourself; an admin for anyone. */
static void handle_invite(int cfd, pn_local_ctx *ctx, const pn_local_caller *c,
                          const jr_value *body)
{
    const char *for_id = jr_as_string(jr_object_get(body, "person"));
    char code[PN_INVITE_CODE_DIGITS + 1];
    pn_person p;
    json_writer jw;
    int rc = 0;

    for_id = for_id ? for_id : c->person;
    if (strcmp(for_id, c->person) != 0 && !c->admin) {
        hs_send_error(cfd, 403, "forbidden", "only an admin invites a device for someone else");
        return;
    }
    if (pn_people_get(ctx->people, for_id, &p) != PN_PEOPLE_OK) {
        hs_send_error(cfd, 404, "not_found", "no person with that id");
        return;
    }
    rc = pn_invites_issue(ctx->invites, p.id, time(NULL), code);
    if (rc != 0 || !jw_init(&jw)) {
        hs_send_error(cfd, rc == -1 ? 429 : 500, rc == -1 ? "invites_full" : "server_error",
                      rc == -1 ? "16 invites are live; wait for one to expire" : "no invite");
        return;
    }
    jw_printf(&jw, "{\"code\":\"%s\",\"person\":", code);
    pn_local_write_person_short(&jw, &p);
    jw_printf(&jw, ",\"expires_in\":%d}", PN_INVITE_TTL_S);
    fprintf(stderr, "local: invite issued for %s by key %s\n", p.id, c->key_id);
    hs_send_json(cfd, 200, jw.data);
    memset(code, 0, sizeof(code));
    memset(jw.data, 0, jw.len);
    jw_free(&jw);
}

/* The POST routes with a JSON body. */
static void dispatch_post(int cfd, pn_local_ctx *ctx, const pn_local_caller *c, const char *p,
                          const jr_value *body)
{
    if (strcmp(p, "/local/me/device") == 0)
        handle_me_device(cfd, ctx, c, body);
    else if (strcmp(p, "/local/invites") == 0)
        handle_invite(cfd, ctx, c, body);
    else if (strcmp(p, "/local/people") == 0)
        handle_people_add(cfd, ctx, c, body);
    else if (strcmp(p, "/local/people/remove") == 0)
        handle_people_remove(cfd, ctx, c, body);
    else
        handle_people_update(cfd, ctx, c, p + strlen("/local/people/"), body);
}

static int path_is(const char *path, const char *p)
{
    size_t n = strlen(p);
    return strncmp(path, p, n) == 0 && (path[n] == '\0' || path[n] == '?');
}

/* 1 for a POST /local/people/<id> target (one segment, not "remove"). */
static int is_person_target(const char *p)
{
    const char *id = p + strlen("/local/people/");

    return strncmp(p, "/local/people/", strlen("/local/people/")) == 0 && id[0] &&
           !strchr(id, '/') && !strchr(id, '?');
}

static int is_people_route(int get, int post, const char *p)
{
    if (get)
        return path_is(p, "/local/me") || path_is(p, "/local/devices") ||
               path_is(p, "/local/people");
    return post && (strcmp(p, "/local/me/device") == 0 || strcmp(p, "/local/invites") == 0 ||
                    strcmp(p, "/local/people") == 0 || is_person_target(p));
}

int pn_local_people_routes(int cfd, pn_local_ctx *ctx, const hs_request *req)
{
    const pn_local_caller *c = pn_local_caller_get();
    const int get = strcmp(req->method, "GET") == 0, post = strcmp(req->method, "POST") == 0;
    const char *p = req->path;
    jr_value *body = NULL;

    if (!is_people_route(get, post, p))
        return 0;
    if (!ctx->keys || !ctx->people || !ctx->invites || !c) {
        hs_send_error(cfd, c ? 503 : 401, c ? "unavailable" : "unauthorized",
                      c ? "people are not open" : "a device key is required");
        return 1;
    }
    if (get) {
        if (path_is(p, "/local/me"))
            handle_me(cfd, ctx, c);
        else if (path_is(p, "/local/devices"))
            handle_devices(cfd, ctx, c, p);
        else
            handle_people_list(cfd, ctx, c);
        return 1;
    }
    body = req->body ? jr_parse(req->body) : NULL;
    if (!body || jr_kind(body) != JR_OBJECT)
        hs_send_error(cfd, 400, "invalid_request_error", "a JSON object body is required");
    else
        dispatch_post(cfd, ctx, c, p, body);
    jr_free(body);
    return 1;
}
