/* SPDX-License-Identifier: AGPL-3.0-or-later */
/* Copyright (C) 2026 David (Corey) Herrera */
/**
 * @file   local_people_wire.c
 * @brief  Implementation of the shared people wire pieces (see local_people_wire.h).
 *
 * @status None
 * @issues None
 * @todo   None
 */

#include "local_people_wire.h"

#include <string.h>

void pn_local_write_person_full(json_writer *jw, const pn_person *p)
{
    jw_printf(jw, "{\"id\":\"%s\",\"handle\":", p->id);
    jw_string(jw, p->handle);
    jw_raw(jw, ",\"name\":");
    jw_string(jw, p->name);
    jw_printf(jw, ",\"admin\":%s,\"may_load\":%s,\"created\":%lld}", p->admin ? "true" : "false",
              p->may_load ? "true" : "false", p->created);
}

void pn_local_write_person_short(json_writer *jw, const pn_person *p)
{
    jw_printf(jw, "{\"id\":\"%s\",\"handle\":", p->id);
    jw_string(jw, p->handle);
    jw_raw(jw, ",\"name\":");
    jw_string(jw, p->name);
    jw_raw(jw, "}");
}

void pn_local_write_person_short_of(json_writer *jw, const pn_person *people, int n,
                                    const char *id)
{
    for (int i = 0; i < n; i++) {
        if (strcmp(people[i].id, id) == 0) {
            pn_local_write_person_short(jw, &people[i]);
            return;
        }
    }
    jw_raw(jw, "{\"id\":");
    jw_string(jw, id);
    jw_raw(jw, ",\"handle\":\"\",\"name\":\"\"}");
}

int pn_local_people_error(int result, const char **type_out)
{
    static const struct { int result, status; const char *type; } map[] = {
        { PN_PEOPLE_E_INVALID, 400, "invalid_request_error" },
        { PN_PEOPLE_E_BAD_HANDLE, 400, "bad_handle" },
        { PN_PEOPLE_E_BAD_NAME, 400, "bad_name" },
        { PN_PEOPLE_E_HANDLE_TAKEN, 409, "handle_taken" },
        { PN_PEOPLE_E_FULL, 507, "people_full" },
        { PN_PEOPLE_E_NOT_FOUND, 404, "not_found" },
        { PN_PEOPLE_E_LAST_ADMIN, 409, "last_admin" },
        { PN_PEOPLE_E_ADMIN_ABOVE, 403, "admin_above" },
        { PN_PEOPLE_E_NOT_ADMIN, 403, "forbidden" },
        { PN_PEOPLE_E_IO, 507, "people_store_error" },
    };

    for (size_t i = 0; i < sizeof(map) / sizeof(map[0]); i++) {
        if (map[i].result == result) {
            *type_out = map[i].type;
            return map[i].status;
        }
    }
    *type_out = "server_error";
    return 500;
}

const char *pn_local_key_legacy_role(pn_key_kind kind, int person_is_admin)
{
    if (kind == PN_KEY_KIND_INFERENCE)
        return "inference";
    return person_is_admin ? "admin" : "device";
}

int pn_local_query_flag(const char *path, const char *name)
{
    const char *q = path ? strchr(path, '?') : NULL;
    size_t n = strlen(name);

    while (q) {
        q++;
        if (strncmp(q, name, n) == 0 && q[n] == '=' && q[n + 1] == '1' &&
            (q[n + 2] == '\0' || q[n + 2] == '&'))
            return 1;
        q = strchr(q, '&');
    }
    return 0;
}
