/* SPDX-License-Identifier: AGPL-3.0-or-later */
/* Copyright (C) 2026 David (Corey) Herrera */
/**
 * @file   local_people_wire.h
 * @brief  Wire pieces shared by the pairing, key, device and people routes
 *         (people registry spec §4): person objects (full / short), the error a
 *         people-store result becomes, the role older clients read, and the
 *         `?all=1` query flag.
 *
 * @status None
 * @issues None
 * @todo   None
 */

#ifndef PN_LOCAL_PEOPLE_WIRE_H
#define PN_LOCAL_PEOPLE_WIRE_H

#include "../serve/json_writer.h"
#include "key_store.h"
#include "people_store.h"

#ifdef __cplusplus
extern "C" {
#endif

/**
 * Write a full person object `{id,handle,name,admin,may_load,created}`.
 *
 * Thread-safe: yes. Reentrant: yes. Signal-safe: no
 */
void pn_local_write_person_full(json_writer *jw, const pn_person *p);

/**
 * Write a short person object `{id,handle,name}`.
 *
 * Thread-safe: yes. Reentrant: yes. Signal-safe: no
 */
void pn_local_write_person_short(json_writer *jw, const pn_person *p);

/**
 * Write the short object of person `id`, looked up in `people` (n rows); an id
 * not in the list writes `{id,"handle":"","name":""}`.
 *
 * Thread-safe: yes. Reentrant: yes. Signal-safe: no
 */
void pn_local_write_person_short_of(json_writer *jw, const pn_person *people, int n,
                                    const char *id);

/**
 * The HTTP status and error `type` for a negative pn_people_result.
 * @return the status (400, 403, 404, 409, 507 or 500); *type_out set
 *
 * Thread-safe: yes. Reentrant: yes. Signal-safe: yes
 */
int pn_local_people_error(int result, const char **type_out);

/**
 * The `role` older clients read for a key: "inference", "admin" (a device of an
 * admin person) or "device".
 *
 * Thread-safe: yes. Reentrant: yes. Signal-safe: yes
 */
const char *pn_local_key_legacy_role(pn_key_kind kind, int person_is_admin);

/**
 * 1 when the request target's query has `name=1`, else 0.
 *
 * Thread-safe: yes. Reentrant: yes. Signal-safe: yes
 */
int pn_local_query_flag(const char *path, const char *name);

#ifdef __cplusplus
}
#endif

#endif /* PN_LOCAL_PEOPLE_WIRE_H */
