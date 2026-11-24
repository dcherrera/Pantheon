/* SPDX-License-Identifier: AGPL-3.0-or-later */
/* Copyright (C) 2026 David (Corey) Herrera */
/**
 * @file   local_relay_routes.c
 * @brief  Implementation of the relay routes (see local_relay_routes.h). Never
 *         logs a ticket, the host's secret or a relayed byte; logs name the
 *         session key, the relay id and the device key id only.
 *
 * @status None
 * @issues None
 * @todo   None
 */

#include "local_relay_routes.h"

#include <stdio.h>
#include <string.h>
#include <time.h>

#include "../../lib/monocypher/monocypher.h"
#include "../serve/json_reader.h"
#include "attach_ticket.h"
#include "hosts_hub.h"
#include "local_caller.h"
#include "session_registry.h"
#include "sha256.h"

#define RELAY_PREFIX "/local/relay/"

/* The ticket's check against the row's secret and the caller; 0, or -1 with
 * the answer sent. */
static int ticket_for_caller(int cfd, const pn_local_caller *c, const char *key,
                             const char *ticket, const char *secret_hex, pn_ticket_fields *f)
{
    uint8_t secret[PN_TICKET_SECRET_LEN];
    int rc = pn_hex_decode(secret_hex, secret, sizeof(secret)) == 0
           ? pn_ticket_check(secret, key, ticket, (long long)time(NULL), f)
           : PN_TICKET_E_INVALID;

    crypto_wipe(secret, sizeof(secret));
    if (rc != PN_TICKET_OK) {
        fprintf(stderr, "local: relay to %s refused for key %s: ticket %s\n", key, c->key_id,
                pn_ticket_result_name(rc));
        hs_send_error(cfd, 403, "bad_ticket", "the ticket is not valid for this session "
                      "(get a new one from POST /local/sessions/<key>/attach)");
        return -1;
    }
    if (strcmp(f->person, c->person) != 0 || strcmp(f->device_id, c->device_id) != 0) {
        fprintf(stderr, "local: relay to %s refused for key %s: ticket of another device\n",
                key, c->key_id);
        hs_send_error(cfd, 403, "bad_ticket", "the ticket was minted for another device");
        return -1;
    }
    return 0;
}

static void send_open_error(int cfd, int rc)
{
    if (rc == PN_HUB_RELAY_E_FULL)
        hs_send_error_ex(cfd, 503, "relay_full", "too many relays open; retry shortly",
                         "Retry-After: 1\r\n");
    else if (rc == PN_HUB_RELAY_E_NONCE)
        hs_send_error(cfd, 409, "ticket_used", "this ticket already opened a relay; attach "
                      "again for a new one");
    else if (rc == PN_HUB_RELAY_E_STOPPED)
        hs_send_error(cfd, 503, "unavailable", "the box is stopping");
    else
        hs_send_error(cfd, 404, "not_found", "no registered session with that key");
}

/* POST /local/relay/<key>: 2 when the hub holds the fd. */
static int handle_open(int cfd, pn_local_ctx *ctx, const pn_local_caller *c, const char *key,
                       const char *ticket)
{
    char secret_hex[PN_REG_HEX32_MAX] = "", pub[PN_REG_HEX32_MAX] = "";
    char id[PN_HUB_RELAY_ID_MAX] = "";
    pn_hub_relay_open_args args = { .key = key, .client_person = c->person };
    pn_ticket_fields f = {0};
    pn_reg_role role = PN_REG_ROLE_NONE;
    pn_reg_row_info row;
    int rc = 0;

    if (!c->device_id[0]) {
        hs_send_error(cfd, 409, "device_unclaimed", "this device key has no device_id yet "
                      "(POST /local/me/device first)");
        return 1;
    }
    rc = pn_reg_attach_info(ctx->registry, key, c->person, &row, &role, secret_hex, pub);
    if (rc != PN_REG_OK) {
        hs_send_error(cfd, 404, "not_found", "no registered session with that key");
        return 1;
    }
    if (row.state == PN_REG_STATE_DISCONNECTED || strlen(secret_hex) != 64) {
        crypto_wipe(secret_hex, sizeof(secret_hex));
        hs_send_error(cfd, 409, "disconnected", "the session's host is not connected");
        return 1;
    }
    rc = ticket_for_caller(cfd, c, key, ticket, secret_hex, &f);
    crypto_wipe(secret_hex, sizeof(secret_hex));
    if (rc != 0)
        return 1;
    args.owner = row.owner;
    args.nonce = f.nonce;
    args.nonce_until = f.exp + PN_TICKET_SKEW_S;
    rc = pn_hub_relay_open(ctx->hub, cfd, &args, id);
    crypto_wipe(&f, sizeof(f));
    if (rc != PN_HUB_RELAY_OK) {
        send_open_error(cfd, rc);
        return 1;
    }
    fprintf(stderr, "local: relay %s to %s opened by key %s\n", id, key, c->key_id);
    return 2;
}

/* POST /local/relay/<id>/host: 2 when the hub piped the fd. */
static int handle_host(int cfd, pn_local_ctx *ctx, const pn_local_caller *c, const char *id)
{
    int rc = pn_hub_relay_host(ctx->hub, cfd, id, c->person);

    if (rc == PN_HUB_RELAY_OK) {
        fprintf(stderr, "local: relay %s joined by key %s\n", id, c->key_id);
        return 2;
    }
    if (rc == PN_HUB_RELAY_E_NOT_OWNER)
        hs_send_error(cfd, 403, "forbidden", "only the session's owner joins its relay");
    else if (rc == PN_HUB_RELAY_E_STOPPED)
        hs_send_error(cfd, 503, "unavailable", "the box is stopping");
    else
        hs_send_error(cfd, 404, "not_found", "no pending relay with that id (it may have "
                      "expired)");
    return 1;
}

/* The path's first segment into seg; *host set when "/host" follows. -1 when
 * the path is neither /local/relay/<seg> nor /local/relay/<seg>/host. */
static int relay_path(const char *path, char *seg, size_t cap, int *host)
{
    const char *s = path + strlen(RELAY_PREFIX);
    size_t n = strcspn(s, "/?");

    if (n == 0 || n >= cap)
        return -1;
    *host = strncmp(s + n, "/host", 5) == 0 && (s[n + 5] == '\0' || s[n + 5] == '?');
    if (!*host && s[n] != '\0' && s[n] != '?')
        return -1;
    memcpy(seg, s, n);
    seg[n] = '\0';
    return 0;
}

int pn_local_relay_routes(int cfd, pn_local_ctx *ctx, const hs_request *req)
{
    const pn_local_caller *c = pn_local_caller_get();
    char seg[PN_HUB_RELAY_ID_MAX];
    const char *ticket = NULL;
    jr_value *body = NULL;
    int host = 0, rc = 1;

    if (strcmp(req->method, "POST") != 0 ||
        strncmp(req->path, RELAY_PREFIX, strlen(RELAY_PREFIX)) != 0 ||
        relay_path(req->path, seg, sizeof(seg), &host) != 0)
        return 0;
    if (!c || !ctx->registry || !ctx->hub) {
        hs_send_error(cfd, c ? 503 : 401, c ? "unavailable" : "unauthorized",
                      c ? "the registry is not open" : "a device key is required");
        return 1;
    }
    if (host)
        return handle_host(cfd, ctx, c, seg);
    if (!pn_reg_key_valid(seg)) {
        hs_send_error(cfd, 404, "not_found", "no registered session with that key");
        return 1;
    }
    body = req->body ? jr_parse(req->body) : NULL;
    ticket = jr_as_string(jr_object_get(body, "ticket"));
    if (!ticket || !ticket[0])
        hs_send_error(cfd, 400, "invalid_request_error", "a JSON object with the attach "
                      "ticket is required");
    else
        rc = handle_open(cfd, ctx, c, seg, ticket);
    jr_free(body);
    return rc;
}
