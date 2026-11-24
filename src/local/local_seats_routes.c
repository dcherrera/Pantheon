/* SPDX-License-Identifier: AGPL-3.0-or-later */
/* Copyright (C) 2026 David (Corey) Herrera */
/**
 * @file   local_seats_routes.c
 * @brief  Implementation of the admin seat view (see local_seats_routes.h).
 *
 * @status None
 * @issues None
 * @todo   None
 */

#include "local_seats_routes.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "../serve/json_writer.h"
#include "../serve/model_registry.h"
#include "../serve/model_session.h"
#include "people_store.h"
#include "session_registry.h"

/* One person's seats on one model instance. */
typedef struct {
    char model[MR_MAX_NAME];
    int  instance;
    char person[MS_PRINCIPAL_MAX];
    int  held;
} seat_row;

/* Every model instance's seats per person. Returns the rows found (at most max). */
static int collect_seats(mr_registry *reg, seat_row *rows, int max)
{
    mr_info info[MR_MAX_MODELS];
    mr_entry *copies[MR_MAX_MODELS];
    ms_principal_seats seats[PN_PEOPLE_MAX];
    int n_info = reg ? mr_snapshot(reg, info, MR_MAX_MODELS) : 0, n = 0;

    for (int i = 0; i < n_info; i++) {
        int seen = 0;
        for (int j = 0; j < i && !seen; j++)
            seen = strcmp(info[j].name, info[i].name) == 0;
        if (seen)
            continue;
        int n_copies = mr_acquire_replicas(reg, info[i].name, copies, MR_MAX_MODELS);
        for (int c = 0; c < n_copies; c++) {
            int k = ms_seats_by_principal(copies[c], seats, PN_PEOPLE_MAX);
            for (int s = 0; s < k && n < max; s++, n++) {
                snprintf(rows[n].model, sizeof(rows[n].model), "%s", copies[c]->name);
                rows[n].instance = copies[c]->replica;
                snprintf(rows[n].person, sizeof(rows[n].person), "%s", seats[s].principal);
                rows[n].held = seats[s].held;
            }
            mr_release(reg, copies[c]);
        }
    }
    return n;
}

/* "sessions":{...}: the person's own session rows by state (desktop rows are
 * request endpoints, not sessions). */
static void write_session_counts(json_writer *jw, pn_reg *reg, const char *person,
                                 pn_reg_row_info *rows)
{
    int by_state[PN_REG_STATE_DISCONNECTED + 1] = { 0 }, registered = 0;
    int n = reg ? pn_reg_list_for(reg, person, rows, PN_REG_MAX_ROWS) : 0;

    for (int i = 0; i < n; i++) {
        if (strcmp(rows[i].owner, person) != 0 || rows[i].kind != PN_REG_KIND_SESSION)
            continue;
        registered++;
        if ((int)rows[i].state >= 0 && rows[i].state <= PN_REG_STATE_DISCONNECTED)
            by_state[rows[i].state]++;
    }
    jw_printf(jw, ",\"sessions\":{\"registered\":%d,\"thinking\":%d,\"needs_you\":%d,\"idle\":%d,"
                  "\"disconnected\":%d}", registered, by_state[PN_REG_STATE_THINKING],
              by_state[PN_REG_STATE_NEEDS_YOU], by_state[PN_REG_STATE_IDLE],
              by_state[PN_REG_STATE_DISCONNECTED]);
}

static void write_person(json_writer *jw, const pn_person *p, pn_reg *reg, pn_reg_row_info *rows,
                         const seat_row *seats, int n_seats)
{
    int first = 1;

    jw_printf(jw, "{\"id\":\"%s\",\"handle\":", p->id);
    jw_string(jw, p->handle);
    jw_raw(jw, ",\"name\":");
    jw_string(jw, p->name);
    write_session_counts(jw, reg, p->id, rows);
    jw_raw(jw, ",\"seats\":[");
    for (int i = 0; i < n_seats; i++) {
        if (strcmp(seats[i].person, p->id) != 0)
            continue;
        jw_raw(jw, first ? "{\"model\":" : ",{\"model\":");
        jw_string(jw, seats[i].model);
        jw_printf(jw, ",\"instance\":%d,\"held\":%d}", seats[i].instance, seats[i].held);
        first = 0;
    }
    jw_raw(jw, "]}");
}

static void handle_seats_people(int cfd, pn_local_ctx *ctx)
{
    pn_person *people = calloc(PN_PEOPLE_MAX, sizeof(*people));
    pn_reg_row_info *rows = calloc(PN_REG_MAX_ROWS, sizeof(*rows));
    seat_row *seats = calloc(MR_MAX_MODELS * PN_PEOPLE_MAX, sizeof(*seats));
    json_writer jw;

    if (!people || !rows || !seats || !jw_init(&jw)) {
        hs_send_error(cfd, 500, "server_error", "oom");
        goto done;
    }
    int n_people = ctx->people ? pn_people_list(ctx->people, people, PN_PEOPLE_MAX) : 0;
    int n_seats = collect_seats(ctx->reg, seats, MR_MAX_MODELS * PN_PEOPLE_MAX);
    jw_raw(&jw, "{\"people\":[");
    for (int i = 0; i < n_people; i++) {
        if (i)
            jw_raw(&jw, ",");
        write_person(&jw, &people[i], ctx->registry, rows, seats, n_seats);
    }
    jw_raw(&jw, "]}");
    hs_send_json(cfd, 200, jw.data);
    jw_free(&jw);
done:
    free(people);
    free(rows);
    free(seats);
}

int pn_local_seats_routes(int client_fd, pn_local_ctx *ctx, const hs_request *req)
{
    const char *p = req->path;
    size_t n = strlen("/local/seats/people");

    if (strcmp(req->method, "GET") != 0 || strncmp(p, "/local/seats/people", n) != 0 ||
        (p[n] != '\0' && p[n] != '?'))
        return 0;
    handle_seats_people(client_fd, ctx);
    return 1;
}
