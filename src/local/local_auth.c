/* SPDX-License-Identifier: AGPL-3.0-or-later */
/* Copyright (C) 2026 David (Corey) Herrera */
/**
 * @file   local_auth.c
 * @brief  Implementation of the appliance auth hook and route classes (see
 *         local_auth.h).
 *
 * @status None
 * @issues None
 * @todo   None
 */

#include "local_auth.h"

#include <stdio.h>
#include <string.h>
#include <strings.h>

#include "key_store.h"
#include "local_caller.h"
#include "people_store.h"
#include "rate_limit.h"

/* `path` is `prefix` or below it (or `prefix` with a query). */
static int path_under(const char *path, const char *prefix)
{
    size_t n = strlen(prefix);
    return strncmp(path, prefix, n) == 0 && (path[n] == '\0' || path[n] == '/' ||
                                             path[n] == '?');
}

/* `path` is exactly `p`, optionally with a query. */
static int path_is(const char *path, const char *p)
{
    size_t n = strlen(p);
    return strncmp(path, p, n) == 0 && (path[n] == '\0' || path[n] == '?');
}

/* `path` is `prefix` + one non-empty segment (no further '/'), query allowed. */
static int path_one_below(const char *path, const char *prefix)
{
    size_t n = strlen(prefix);
    return strncmp(path, prefix, n) == 0 && path[n] && path[n] != '?' &&
           strcspn(path + n, "/") == strlen(path + n);
}

/* /local/people*: the list is DEVICE; a change to one person is DEVICE at the
 * hook because a person may rename themselves (the route checks the rest). */
static pn_local_route_class people_class(int get, int post, const char *path)
{
    if (get && path_is(path, "/local/people"))
        return PN_ROUTE_DEVICE;
    if (post && path_is(path, "/local/people/remove"))
        return PN_ROUTE_ADMIN;
    if (post && path_one_below(path, "/local/people/"))
        return PN_ROUTE_DEVICE;
    return PN_ROUTE_ADMIN;
}

pn_local_route_class pn_local_route_class_of(const char *method, const char *path)
{
    const int get = strcmp(method, "GET") == 0, post = strcmp(method, "POST") == 0;

    if (strcmp(method, "OPTIONS") == 0 || (post && strcmp(path, "/local/pair") == 0))
        return PN_ROUTE_NONE;
    if (path_under(path, "/v1") || path_under(path, "/session") || strcmp(path, "/") == 0 ||
        (post && path_is(path, "/admin/session/export")))
        return PN_ROUTE_INFERENCE;   /* export: the bound person's own sessions (§4.8) */
    if (post && (path_is(path, "/local/models/install") || path_is(path, "/local/models/load") ||
                 path_is(path, "/admin/models/unload")))
        return PN_ROUTE_LOADER;
    if (path_under(path, "/local/people"))
        return people_class(get, post, path);
    if (get && (path_is(path, "/local/hardware") || path_is(path, "/local/catalog") ||
                path_is(path, "/local/models/loaded") ||
                path_one_below(path, "/local/models/install/") ||
                path_is(path, "/local/devices")))
        return PN_ROUTE_DEVICE;
    if ((post && path_is(path, "/local/invites")) || path_under(path, "/local/me") ||
        path_under(path, "/local/keys") || path_under(path, "/local/hosts") ||
        path_under(path, "/local/sessions") || path_under(path, "/local/relay"))
        return PN_ROUTE_DEVICE;
    return PN_ROUTE_ADMIN;
}

const char *pn_local_route_class_name(pn_local_route_class cls)
{
    static const char *names[] = { "none", "inference", "device", "loader", "admin" };
    return (int)cls >= 0 && cls <= PN_ROUTE_ADMIN ? names[cls] : "unknown";
}

/* The bearer token of the request, or NULL. Points into req. */
static const char *bearer_token(const hs_request *req)
{
    const char *a = req->authorization;

    if (!a || strncasecmp(a, "Bearer ", 7) != 0)
        return NULL;
    a += 7;
    while (*a == ' ')
        a++;
    return *a ? a : NULL;
}

static int refuse(int cfd, int status, const char *type, const char *msg)
{
    if (status == 429)
        hs_send_error_ex(cfd, status, type, msg, "Retry-After: 1\r\n");
    else
        hs_send_error(cfd, status, type, msg);
    return 1;
}

/* Key → person → caller. Returns 0, or -1 when the key or its person is unknown. */
static int resolve_caller(pn_local_ctx *ctx, const hs_request *req, pn_local_caller *out)
{
    pn_key_info key;
    pn_person person;

    if (!ctx->keys || !ctx->people || pn_key_store_verify(ctx->keys, bearer_token(req), &key) != 0)
        return -1;
    if (pn_people_get(ctx->people, key.person, &person) != PN_PEOPLE_OK) {
        fprintf(stderr, "local: key %s belongs to no person; refused\n", key.id);
        return -1;
    }
    memset(out, 0, sizeof(*out));
    snprintf(out->key_id, sizeof(out->key_id), "%s", key.id);
    out->kind = key.kind;
    snprintf(out->person, sizeof(out->person), "%s", key.person);
    snprintf(out->device_id, sizeof(out->device_id), "%s", key.device_id);
    out->admin = person.admin;
    out->may_load = person.admin || person.may_load;
    return 0;
}

/* The refusal for a known caller on a route of `cls`, or 0 when it may pass. */
static int class_refusal(int cfd, const pn_local_caller *c, pn_local_route_class cls,
                         const hs_request *req)
{
    const char *why = NULL, *type = "forbidden";

    if (cls >= PN_ROUTE_DEVICE && c->kind != PN_KEY_KIND_DEVICE)
        why = "this route needs a device key (pair this device with the box)";
    else if (cls == PN_ROUTE_LOADER && !c->may_load) {
        why = "loading models is turned off for this person (ask an admin)";
        type = "may_load_off";
    } else if (cls == PN_ROUTE_ADMIN && !c->admin)
        why = "this route needs an admin";
    if (!why)
        return 0;
    fprintf(stderr, "local: 403 %s key %s (%s, person %s) %s %s\n", type, c->key_id,
            pn_key_kind_name(c->kind), c->person, req->method, req->path);
    return refuse(cfd, 403, type, why);
}

/* GET /local/hosts/<key>/wait: the box holds it and keeps at most one per key,
 * so it does not spend the key's token bucket (spec §4.1). */
static int is_host_wait(const hs_request *req)
{
    const char *prefix = "/local/hosts/";
    const char *k = NULL;
    size_t n = 0;

    if (strcmp(req->method, "GET") != 0 || strncmp(req->path, prefix, strlen(prefix)) != 0)
        return 0;
    k = req->path + strlen(prefix);
    n = strcspn(k, "/?");
    return n > 0 && strncmp(k + n, "/wait", 5) == 0 && (k[n + 5] == '\0' || k[n + 5] == '?');
}

int pn_local_auth_check(int client_fd, const hs_request *req, void *user)
{
    pn_local_ctx *ctx = user;
    pn_local_route_class cls = pn_local_route_class_of(req->method, req->path);
    pn_local_caller caller;

    pn_local_caller_clear();
    hs_set_thread_principal(NULL, 0);
    if (cls == PN_ROUTE_NONE) {
        if (strcmp(req->method, "POST") == 0 && ctx->pair_limiter &&
            !pn_rate_limiter_take(ctx->pair_limiter, "pair", pn_rate_now()))
            return refuse(client_fd, 429, "rate_limited", "too many pairing attempts");
        return 0;
    }
    if (resolve_caller(ctx, req, &caller) != 0) {
        fprintf(stderr, "local: 401 %s %s (no valid key)\n", req->method, req->path);
        return refuse(client_fd, 401, "unauthorized",
                      "a key is required: Authorization: Bearer <key> (pair with the box)");
    }
    if (class_refusal(client_fd, &caller, cls, req))
        return 1;
    if (ctx->limiter && !is_host_wait(req) &&
        !pn_rate_limiter_take(ctx->limiter, caller.key_id, pn_rate_now()))
        return refuse(client_fd, 429, "rate_limited", "rate limit for this key; retry shortly");
    if (cls == PN_ROUTE_ADMIN && strcmp(req->method, "GET") != 0)
        fprintf(stderr, "local: key %s (admin %s) %s %s\n", caller.key_id, caller.person,
                req->method, req->path);
    pn_local_caller_set(&caller);
    hs_set_thread_principal(caller.person, caller.admin);   /* serve sessions bind to it */
    return 0;
}
