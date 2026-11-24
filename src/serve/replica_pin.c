/* SPDX-License-Identifier: AGPL-3.0-or-later */
/* Copyright (C) 2026 David (Corey) Herrera */
/**
 * @file   replica_pin.c
 * @brief  Implementation of a client choosing its copy of a model (see
 *         replica_pin.h): parse the request's replica, route with it, announce
 *         the serving copy, and the copy-is-missing / copy-is-full errors.
 *
 * @status None
 * @issues None
 * @todo   None
 */

#include "replica_pin.h"

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "http_server.h"
#include "replica_affinity.h"

/* A copy number from the header: decimal digits only, 1..1000000. */
static int parse_header_number(const char *v)
{
    char *end = NULL;
    long n;

    while (*v == ' ' || *v == '\t')
        v++;
    if (*v < '0' || *v > '9')
        return -1;
    n = strtol(v, &end, 10);
    while (*end == ' ' || *end == '\t')
        end++;
    return (*end == '\0' && n >= 1 && n <= 1000000) ? (int)n : -1;
}

int pn_replica_pin_parse(const jr_value *body, const char *header_value, int *replica_out,
                         char *err, size_t errlen)
{
    const jr_value *field = body ? jr_object_get(body, "replica") : NULL;

    *replica_out = 0;
    if (field && jr_kind(field) != JR_NULL) {
        double v = jr_as_number(field, 0);
        if (jr_kind(field) != JR_NUMBER || v < 1 || v > 1000000 || floor(v) != v) {
            snprintf(err, errlen, "'replica' must be an integer >= 1");
            return -1;
        }
        *replica_out = (int)v;
        return 0;
    }
    if (header_value && *header_value) {
        int n = parse_header_number(header_value);
        if (n < 0) {
            snprintf(err, errlen, PN_REPLICA_HEADER " must be an integer >= 1");
            return -1;
        }
        *replica_out = n;
    }
    return 0;
}

void pn_replica_pin_announce(const mr_entry *entry)
{
    char line[64];

    if (!entry) {
        hs_set_thread_response_headers(NULL);
        return;
    }
    snprintf(line, sizeof(line), PN_REPLICA_HEADER ": %d\r\n", entry->replica);
    hs_set_thread_response_headers(line);
}

/* 404: the name is loaded but not as copy route->replica. Named as routed: with
 * one model loaded the caller routes without the request's name. */
static void send_missing(int client_fd, const mr_route *route)
{
    const char *name = route->name;
    char loaded[128] = "", msg[384];
    size_t len = 0;

    for (int i = 0; i < route->n_replicas && len < sizeof(loaded); i++) {
        int n = snprintf(loaded + len, sizeof(loaded) - len, "%s%d", i ? ", " : "",
                         route->replicas[i]);
        if (n < 0)
            break;
        len += (size_t)n;
    }
    snprintf(msg, sizeof(msg), "model '%s' has no replica %d (loaded replicas: %s)",
             name && *name ? name : "(default)", route->replica, loaded);
    hs_send_error(client_fd, 404, "replica_not_found", msg);
}

mr_entry *pn_replica_pin_acquire(int client_fd, mr_registry *reg, const char *name,
                                 const char *sid, mr_route *route)
{
    mr_entry *entry = mr_acquire_session_on(reg, name, sid, route);

    if (entry) {
        if (route->kept_binding)
            fprintf(stderr, "serve: session %.63s asked for replica %d of '%s'; it stays on "
                    "replica %d (its binding)\n", sid, route->replica, entry->name,
                    entry->replica);
        pn_replica_pin_announce(entry);
    } else if (route->status == MR_ROUTE_NO_REPLICA) {
        send_missing(client_fd, route);
    }
    return entry;
}

void pn_replica_pin_undo(const mr_route *route, const mr_entry *entry, const char *sid)
{
    if (route && entry && route->bound_now)
        pn_replica_affinity_unbind(sid, entry->serial);
}

void pn_replica_pin_send_full(int client_fd, const mr_route *route, const mr_entry *entry,
                              int restart_s)
{
    char msg[384], seats[64];

    /* entry->max_sessions is the seat total; every one of them is taken. */
    if (entry->max_sessions == 1)
        snprintf(seats, sizeof(seats), "its one session is in use");
    else
        snprintf(seats, sizeof(seats), "all %d sessions are in use", entry->max_sessions);
    if (route && route->replica > 0)
        snprintf(msg, sizeof(msg),
                 "replica %d of model '%s' has no free session slot (%s); retry shortly "
                 "or pick another replica. A fresh session re-prefills in ~%ds.",
                 entry->replica, entry->name, seats, restart_s);
    else
        snprintf(msg, sizeof(msg),
                 "model at session capacity (%s); retry shortly. A fresh session "
                 "re-prefills in ~%ds.",
                 seats, restart_s);
    hs_send_error_ex(client_fd, 429, "model_at_session_capacity", msg, "Retry-After: 30\r\n");
}
