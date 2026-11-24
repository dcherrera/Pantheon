/* SPDX-License-Identifier: AGPL-3.0-or-later */
/* Copyright (C) 2026 David (Corey) Herrera */
/**
 * @file   local_key_routes.c
 * @brief  Implementation of the pairing and key routes (see local_key_routes.h).
 *
 * @status None
 * @issues None
 * @todo   None
 */

#include "local_key_routes.h"

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
#include "pairing.h"
#include "people_store.h"

/* The pairing request, checked before the code is spent. */
typedef struct {
    const char *code;
    const char *label;
    const char *device_id;                         /* "" = claim later */
    char        device_name[PN_KEY_DEVICE_NAME_MAX];
    char        handle[PN_PERSON_HANDLE_MAX];      /* read only for person 1 */
    char        name[PN_PERSON_NAME_MAX];
} pair_request;

/* Fill *pr from the body; answers 400 and returns -1 on a bad field. */
static int read_pair_request(int cfd, const jr_value *body, pair_request *pr)
{
    const char *handle = jr_as_string(jr_object_get(body, "handle"));
    const char *name = jr_as_string(jr_object_get(body, "name"));
    const char *device_name = jr_as_string(jr_object_get(body, "device_name"));

    memset(pr, 0, sizeof(*pr));
    pr->code = jr_as_string(jr_object_get(body, "code"));
    pr->label = jr_as_string(jr_object_get(body, "label"));
    pr->label = pr->label ? pr->label : "app";
    pr->device_id = jr_as_string(jr_object_get(body, "device_id"));
    pr->device_id = pr->device_id ? pr->device_id : "";
    if (pr->device_id[0] && !pn_key_device_id_valid(pr->device_id)) {
        hs_send_error(cfd, 400, "invalid_request_error", "device_id is \"d\" + 16 hex");
        return -1;
    }
    if (pn_people_name_clean(device_name ? device_name : pr->label, PN_PERSON_NAME_BYTES,
                             pr->device_name, sizeof(pr->device_name)) != 0) {
        if (device_name) {
            hs_send_error(cfd, 400, "invalid_request_error", "device_name: 1-64 bytes of text");
            return -1;
        }
        snprintf(pr->device_name, sizeof(pr->device_name), "device");
    }
    if (pn_people_name_clean(handle ? handle : "owner", PN_PERSON_HANDLE_BYTES, pr->handle,
                             sizeof(pr->handle)) != 0) {
        hs_send_error(cfd, 400, "bad_handle", "handle: 1-24 bytes of text");
        return -1;
    }
    if (pn_people_name_clean(name ? name : "Owner", PN_PERSON_NAME_BYTES, pr->name,
                             sizeof(pr->name)) != 0) {
        hs_send_error(cfd, 400, "bad_name", "name: 1-64 bytes of text");
        return -1;
    }
    return 0;
}

/* The pairing answer — the one time the device key leaves the box. */
static void send_paired(int cfd, const char *secret, const pn_key_info *key,
                        const pn_person *person, int first_owner)
{
    json_writer jw;

    if (!jw_init(&jw)) {
        hs_send_error(cfd, 500, "server_error", "oom");
        return;
    }
    jw_printf(&jw, "{\"key\":\"%s\",\"id\":\"%s\",\"kind\":\"device\",\"role\":\"%s\",\"label\":",
              secret, key->id, pn_local_key_legacy_role(key->kind, person->admin));
    jw_string(&jw, key->label);
    jw_raw(&jw, ",\"person\":");
    pn_local_write_person_full(&jw, person);
    jw_printf(&jw, ",\"first_owner\":%s}", first_owner ? "true" : "false");
    hs_send_json(cfd, 200, jw.data);
    memset(jw.data, 0, jw.len);
    jw_free(&jw);
}

/* Whose device the code makes: a live invite's person first, else the console
 * code's person 1 (made from handle/name on a fresh box; the oldest admin
 * afterwards: recovery). Returns 0, -1 when the code is refused, -2 when person 1
 * could not be stored. */
static int redeem_code(pn_local_ctx *ctx, const pair_request *pr, pn_person *person, int *first)
{
    char invited[PN_INVITE_PERSON_MAX];
    int rc = 0;

    *first = 0;
    if (ctx->invites && pn_invites_redeem(ctx->invites, pr->code, time(NULL), invited) == 0) {
        pn_pairing_clear_misses(ctx->pairing);
        if (pn_people_get(ctx->people, invited, person) == PN_PEOPLE_OK)
            return 0;
        fprintf(stderr, "local: invite for %s redeemed, but that person is gone\n", invited);
        return -1;
    }
    rc = ctx->pairing ? pn_pairing_redeem(ctx->pairing, pr->code) : -1;
    if (rc == -2) {
        pn_invites_void_all(ctx->invites);
        fprintf(stderr, "local: 5 wrong pairing codes: console code rotated, invites voided\n");
    }
    if (rc != 0)
        return -1;
    return pn_people_owner_or_create(ctx->people, pr->handle, pr->name, person, first) == 0
               ? 0 : -2;
}

/* POST /local/pair: a code binds a device to a person (redeem_code). */
static void handle_pair(int cfd, pn_local_ctx *ctx, const jr_value *body)
{
    char secret[PN_KEY_SECRET_MAX];
    pair_request pr;
    pn_person person;
    pn_key_info key;
    int first = 0, rc = 0;

    if (read_pair_request(cfd, body, &pr) != 0)
        return;
    rc = redeem_code(ctx, &pr, &person, &first);
    if (rc == -1) {
        fprintf(stderr, "local: pairing refused (wrong, expired or used code)\n");
        hs_send_error(cfd, 403, "pairing_refused", "wrong or expired pairing code");
        return;
    }
    if (rc != 0 || pn_key_store_mint_device(ctx->keys, person.id, pr.device_id, pr.device_name,
                                            pr.label, secret, &key) != PN_KEY_OK) {
        hs_send_error(cfd, 507, "key_store_error", "could not store a new key");
        return;
    }
    fprintf(stderr, "local: paired device key %s (%s) for %s (%s)%s\n", key.id, key.label,
            person.handle, person.id, first ? ", the first owner" : "");
    send_paired(cfd, secret, &key, &person, first);
    memset(secret, 0, sizeof(secret));
}

/* One key row of GET /local/keys. */
static void write_key_row(json_writer *jw, const pn_key_info *k, const pn_person *people, int n)
{
    int admin = 0;

    for (int i = 0; i < n; i++)
        if (strcmp(people[i].id, k->person) == 0)
            admin = people[i].admin;
    jw_printf(jw, "{\"id\":\"%s\",\"kind\":\"%s\",\"role\":\"%s\",\"person\":", k->id,
              pn_key_kind_name(k->kind), pn_local_key_legacy_role(k->kind, admin));
    pn_local_write_person_short_of(jw, people, n, k->person);
    jw_raw(jw, ",\"label\":");
    jw_string(jw, k->label);
    jw_printf(jw, ",\"device_id\":\"%s\",\"device_name\":", k->device_id);
    jw_string(jw, k->device_name);
    jw_printf(jw, ",\"created\":%lld}", k->created);
}

/* GET /local/keys[?all=1]: the caller's keys; every key for an admin with all=1. */
static void handle_list(int cfd, pn_local_ctx *ctx, const pn_local_caller *c, const char *path)
{
    const int all = pn_local_query_flag(path, "all");
    pn_key_info *rows = NULL;
    pn_person people[PN_PEOPLE_MAX];
    json_writer jw;
    int n = 0, n_people = 0;

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
    jw_raw(&jw, "{\"keys\":[");
    for (int i = 0; i < n; i++) {
        jw_raw(&jw, i ? "," : "");
        write_key_row(&jw, &rows[i], people, n_people);
    }
    jw_raw(&jw, "]}");
    hs_send_json(cfd, 200, jw.data);
    jw_free(&jw);
    free(rows);
}

/* The one-time answer of POST /local/keys. */
static void send_minted(int cfd, const char *secret, const pn_key_info *key,
                        const pn_person *person)
{
    json_writer jw;

    if (!jw_init(&jw)) {
        hs_send_error(cfd, 500, "server_error", "oom");
        return;
    }
    jw_printf(&jw, "{\"key\":\"%s\",\"id\":\"%s\",\"kind\":\"inference\",\"role\":\"inference\","
              "\"label\":", secret, key->id);
    jw_string(&jw, key->label);
    jw_raw(&jw, ",\"person\":");
    pn_local_write_person_short(&jw, person);
    jw_raw(&jw, "}");
    hs_send_json(cfd, 200, jw.data);
    memset(jw.data, 0, jw.len);
    jw_free(&jw);
}

/* POST /local/keys {label, person?}: an inference key for the caller's person;
 * an admin may name any person. */
static void handle_mint(int cfd, pn_local_ctx *ctx, const pn_local_caller *c,
                        const jr_value *body)
{
    const char *role = jr_as_string(jr_object_get(body, "role"));
    const char *label = jr_as_string(jr_object_get(body, "label"));
    const char *for_id = jr_as_string(jr_object_get(body, "person"));
    char secret[PN_KEY_SECRET_MAX];
    pn_person person;
    pn_key_info key;

    if (role && strcmp(role, "inference") != 0) {
        hs_send_error(cfd, 400, "invalid_request_error",
                      "only inference keys can be minted; device keys come from pairing");
        return;
    }
    for_id = for_id ? for_id : c->person;
    if (strcmp(for_id, c->person) != 0 && !c->admin) {
        hs_send_error(cfd, 403, "forbidden", "only an admin mints keys for another person");
        return;
    }
    if (pn_people_get(ctx->people, for_id, &person) != PN_PEOPLE_OK) {
        hs_send_error(cfd, 404, "not_found", "no person with that id");
        return;
    }
    if (pn_key_store_mint_inference(ctx->keys, person.id, label ? label : "client", secret,
                                    &key) != PN_KEY_OK) {
        hs_send_error(cfd, 507, "key_store_error", "could not store a new key");
        return;
    }
    fprintf(stderr, "local: minted inference key %s (%s) for %s by key %s\n", key.id, key.label,
            person.id, c->key_id);
    send_minted(cfd, secret, &key, &person);
    memset(secret, 0, sizeof(secret));
}

/* POST /local/keys/revoke {id}: own keys of any kind; an admin any key. */
static void handle_revoke(int cfd, pn_local_ctx *ctx, const pn_local_caller *c,
                          const jr_value *body)
{
    const char *id = jr_as_string(jr_object_get(body, "id"));
    pn_key_info key;
    char out[64];
    int rc = 0;

    if (!id || pn_key_store_get(ctx->keys, id, &key) != PN_KEY_OK) {
        hs_send_error(cfd, 404, "not_found", "no key with that id");
        return;
    }
    if (strcmp(key.person, c->person) != 0 && !c->admin) {
        hs_send_error(cfd, 403, "forbidden", "only an admin revokes another person's key");
        return;
    }
    rc = pn_key_store_revoke(ctx->keys, id);
    if (rc != PN_KEY_OK) {
        hs_send_error(cfd, rc == PN_KEY_E_NOT_FOUND ? 404 : 507,
                      rc == PN_KEY_E_NOT_FOUND ? "not_found" : "key_store_error",
                      "could not revoke that key");
        return;
    }
    fprintf(stderr, "local: revoked key %s (person %s) by key %s\n", id, key.person, c->key_id);
    if (key.kind == PN_KEY_KIND_DEVICE)
        pn_local_hosts_key_revoked(ctx, id);
    snprintf(out, sizeof(out), "{\"revoked\":\"%s\"}", id);
    hs_send_json(cfd, 200, out);
}

static int is_key_route(const char *method, const char *p)
{
    const int post = strcmp(method, "POST") == 0, get = strcmp(method, "GET") == 0;
    const size_t keys_len = strlen("/local/keys");

    if (post && (strcmp(p, "/local/pair") == 0 || strcmp(p, "/local/keys") == 0 ||
                 strcmp(p, "/local/keys/revoke") == 0))
        return 1;
    return get && strncmp(p, "/local/keys", keys_len) == 0 &&
           (p[keys_len] == '\0' || p[keys_len] == '?');
}

int pn_local_key_routes(int cfd, pn_local_ctx *ctx, const hs_request *req)
{
    const pn_local_caller *c = pn_local_caller_get();
    const char *p = req->path;
    jr_value *body = NULL;

    if (!is_key_route(req->method, p))
        return 0;
    if (!ctx->keys || !ctx->people) {
        hs_send_error(cfd, 503, "unavailable", "key store is not open");
        return 1;
    }
    if (strcmp(p, "/local/pair") != 0 && !c) {
        hs_send_error(cfd, 401, "unauthorized", "a key is required");
        return 1;
    }
    if (strcmp(req->method, "GET") == 0) {
        handle_list(cfd, ctx, c, p);
        return 1;
    }
    body = req->body ? jr_parse(req->body) : NULL;
    if (!body || jr_kind(body) != JR_OBJECT)
        hs_send_error(cfd, 400, "invalid_request_error", "a JSON object body is required");
    else if (strcmp(p, "/local/pair") == 0)
        handle_pair(cfd, ctx, body);
    else if (strcmp(p, "/local/keys") == 0)
        handle_mint(cfd, ctx, c, body);
    else
        handle_revoke(cfd, ctx, c, body);
    jr_free(body);
    return 1;
}
