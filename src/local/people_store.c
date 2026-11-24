/* SPDX-License-Identifier: AGPL-3.0-or-later */
/* Copyright (C) 2026 David (Corey) Herrera */
/**
 * @file   people_store.c
 * @brief  Implementation of the people store and its admin tree (see people_store.h).
 *
 * @status None
 * @issues None
 * @todo   None
 */

#include "people_store.h"

#include <errno.h>
#include <pthread.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

#include <sys/stat.h>

#include "../serve/json_reader.h"
#include "../serve/json_writer.h"
#include "local_file_io.h"
#include "local_random.h"
#include "sha256.h"

struct pn_people {
    pthread_mutex_t lock;
    char            path[PN_LOCAL_PATH_MAX];
    pn_person       rows[PN_PEOPLE_MAX];
    int             n;
};

/* Length of the UTF-8 sequence at p (1-4), or 0 when it is not valid UTF-8. */
static size_t utf8_sequence(const unsigned char *p, size_t left)
{
    size_t n = p[0] < 0x80 ? 1 : (p[0] & 0xE0) == 0xC0 ? 2 : (p[0] & 0xF0) == 0xE0 ? 3
             : (p[0] & 0xF8) == 0xF0 ? 4 : 0;

    if (n == 0 || n > left || (n == 2 && p[0] < 0xC2))   /* overlong 2-byte forms */
        return 0;
    for (size_t i = 1; i < n; i++)
        if ((p[i] & 0xC0) != 0x80)
            return 0;
    return n;
}

int pn_people_name_clean(const char *in, size_t max_bytes, char *out, size_t cap)
{
    const char *s = in ? in : "";
    size_t len = 0;

    while (*s == ' ' || *s == '\t')
        s++;
    len = strlen(s);
    while (len > 0 && (s[len - 1] == ' ' || s[len - 1] == '\t'))
        len--;
    if (len == 0 || len > max_bytes || len >= cap)
        return -1;
    for (size_t i = 0; i < len;) {
        const unsigned char *p = (const unsigned char *)s + i;
        size_t n = utf8_sequence(p, len - i);
        if (n == 0 || (n == 1 && (p[0] < 0x20 || p[0] == 0x7F)))
            return -1;
        i += n;
    }
    memcpy(out, s, len);
    out[len] = '\0';
    return 0;
}

static int ascii_case_equal(const char *a, const char *b)
{
    for (; *a && *b; a++, b++) {
        char ca = (*a >= 'A' && *a <= 'Z') ? (char)(*a + 32) : *a;
        char cb = (*b >= 'A' && *b <= 'Z') ? (char)(*b + 32) : *b;
        if (ca != cb)
            return 0;
    }
    return *a == *b;
}

/* "p" + 8 lowercase hex. */
static int person_id_valid(const char *s)
{
    if (!s || s[0] != 'p' || strlen(s) != 9)
        return 0;
    for (int i = 1; i < 9; i++)
        if (!((s[i] >= '0' && s[i] <= '9') || (s[i] >= 'a' && s[i] <= 'f')))
            return 0;
    return 1;
}

static int index_of_locked(const pn_people *pp, const char *id)
{
    for (int i = 0; id && i < pp->n; i++)
        if (strcmp(pp->rows[i].id, id) == 0)
            return i;
    return -1;
}

static int handle_taken_locked(const pn_people *pp, const char *handle, int except)
{
    for (int i = 0; i < pp->n; i++)
        if (i != except && ascii_case_equal(pp->rows[i].handle, handle))
            return 1;
    return 0;
}

static int admin_count_locked(const pn_people *pp)
{
    int n = 0;

    for (int i = 0; i < pp->n; i++)
        n += pp->rows[i].admin != 0;
    return n;
}

/* 1 when rows[target] sits below rows[ancestor] in the admin tree. */
static int is_below_locked(const pn_people *pp, int ancestor, int target)
{
    int at = target;

    for (int steps = 0; steps < pp->n; steps++) {
        at = index_of_locked(pp, pp->rows[at].admin_by);
        if (at < 0)
            return 0;
        if (at == ancestor)
            return 1;
    }
    return 0;
}

/* May `actor` take admin rights from rows[idx] (demote or remove)? */
static int check_admin_leave_locked(const pn_people *pp, const char *actor, int idx)
{
    int a = index_of_locked(pp, actor);

    if (a < 0 || !pp->rows[a].admin)
        return PN_PEOPLE_E_NOT_ADMIN;
    if (a != idx && !is_below_locked(pp, a, idx))
        return PN_PEOPLE_E_ADMIN_ABOVE;
    if (admin_count_locked(pp) <= 1)
        return PN_PEOPLE_E_LAST_ADMIN;
    return PN_PEOPLE_OK;
}

/* rows[idx] stops being an admin: its child admins move up to its maker; when it
 * is the root, its earliest-made child becomes the root and adopts the others. */
static void reparent_children_locked(pn_people *pp, int idx)
{
    const pn_person *leaving = &pp->rows[idx];
    int new_root = -1;

    for (int j = 0; j < pp->n; j++) {
        pn_person *c = &pp->rows[j];
        if (j == idx || !c->admin || strcmp(c->admin_by, leaving->id) != 0)
            continue;
        if (leaving->admin_by[0])
            snprintf(c->admin_by, sizeof(c->admin_by), "%s", leaving->admin_by);
        else if (new_root < 0 || c->created < pp->rows[new_root].created)
            new_root = j;
    }
    if (new_root < 0)
        return;
    for (int j = 0; j < pp->n; j++) {
        pn_person *c = &pp->rows[j];
        if (j != idx && j != new_root && c->admin && strcmp(c->admin_by, leaving->id) == 0)
            snprintf(c->admin_by, sizeof(c->admin_by), "%s", pp->rows[new_root].id);
    }
    pp->rows[new_root].admin_by[0] = '\0';
}

static int save_locked(const pn_people *pp)
{
    json_writer jw;
    int rc = -1;

    if (!jw_init(&jw))
        return -1;
    jw_raw(&jw, "{\"version\":1,\"people\":[");
    for (int i = 0; i < pp->n; i++) {
        const pn_person *p = &pp->rows[i];
        jw_printf(&jw, "%s\n  {\"id\":\"%s\",\"handle\":", i ? "," : "", p->id);
        jw_string(&jw, p->handle);
        jw_raw(&jw, ",\"name\":");
        jw_string(&jw, p->name);
        jw_printf(&jw, ",\"admin\":%s,\"admin_by\":", p->admin ? "true" : "false");
        if (p->admin_by[0])
            jw_printf(&jw, "\"%s\"", p->admin_by);
        else
            jw_raw(&jw, "null");
        jw_printf(&jw, ",\"may_load\":%s,\"created\":%lld}", p->may_load ? "true" : "false",
                  p->created);
    }
    jw_raw(&jw, "]}\n");
    rc = pn_local_write_file_atomic(pp->path, jw.data, jw.len, 0600);
    jw_free(&jw);
    return rc;
}

/* One row of people.json into *p; -1 when it breaks a rule. */
static int parse_person(const jr_value *v, pn_person *p)
{
    const char *id = jr_as_string(jr_object_get(v, "id"));
    const jr_value *by = jr_object_get(v, "admin_by");
    const jr_value *admin = jr_object_get(v, "admin");

    memset(p, 0, sizeof(*p));
    if (!person_id_valid(id) || jr_kind(admin) != JR_BOOL ||
        pn_people_name_clean(jr_as_string(jr_object_get(v, "handle")), PN_PERSON_HANDLE_BYTES,
                             p->handle, sizeof(p->handle)) != 0 ||
        strcmp(p->handle, jr_as_string(jr_object_get(v, "handle"))) != 0 ||
        pn_people_name_clean(jr_as_string(jr_object_get(v, "name")), PN_PERSON_NAME_BYTES,
                             p->name, sizeof(p->name)) != 0 ||
        strcmp(p->name, jr_as_string(jr_object_get(v, "name"))) != 0)
        return -1;
    snprintf(p->id, sizeof(p->id), "%s", id);
    p->admin = jr_as_bool(admin, false);
    p->may_load = jr_as_bool(jr_object_get(v, "may_load"), true);
    p->created = (long long)jr_as_number(jr_object_get(v, "created"), 0);
    if (by && jr_kind(by) != JR_NULL) {
        if (!p->admin || !person_id_valid(jr_as_string(by)))
            return -1;
        snprintf(p->admin_by, sizeof(p->admin_by), "%s", jr_as_string(by));
    }
    return 0;
}

/* Whole-file rules: unique ids and handles, makers are admins, no cycles, and an
 * admin whenever there are people. */
static int validate_locked(const pn_people *pp)
{
    for (int i = 0; i < pp->n; i++) {
        const pn_person *p = &pp->rows[i];
        if (index_of_locked(pp, p->id) != i || handle_taken_locked(pp, p->handle, i))
            return -1;
        if (p->admin_by[0]) {
            int m = index_of_locked(pp, p->admin_by);
            if (m < 0 || m == i || !pp->rows[m].admin)
                return -1;
            int at = i, steps = 0;
            while (at >= 0 && pp->rows[at].admin_by[0] && steps++ <= pp->n)
                at = index_of_locked(pp, pp->rows[at].admin_by);
            if (steps > pp->n)
                return -1;   /* a cycle never reaches a root */
        }
    }
    return pp->n > 0 && admin_count_locked(pp) == 0 ? -1 : 0;
}

static int load_rows(pn_people *pp, const char *text)
{
    jr_value *root = jr_parse(text);
    const jr_value *people = jr_object_get(root, "people");
    int rc = -1;

    if (!root || jr_as_number(jr_object_get(root, "version"), 0) != 1 ||
        jr_kind(people) != JR_ARRAY || jr_array_len(people) > PN_PEOPLE_MAX)
        goto done;
    for (int i = 0; i < jr_array_len(people); i++) {
        if (parse_person(jr_array_get(people, i), &pp->rows[pp->n]) != 0)
            goto done;
        pp->n++;
    }
    rc = validate_locked(pp);
done:
    jr_free(root);
    return rc;
}

pn_people *pn_people_open(const char *path)
{
    pn_people *pp = calloc(1, sizeof(*pp));
    char *text = NULL;

    if (!pp || !path || snprintf(pp->path, sizeof(pp->path), "%s", path) >= (int)sizeof(pp->path)) {
        free(pp);
        return NULL;
    }
    pthread_mutex_init(&pp->lock, NULL);
    if (pn_local_read_file(path, 1 << 20, &text, NULL) == 0) {
        int bad = load_rows(pp, text);
        free(text);
        if (bad || chmod(path, 0600) != 0) {
            fprintf(stderr, "local: %s is malformed or not ours; refusing to start\n", path);
            pn_people_close(pp);
            return NULL;
        }
    } else if (errno != ENOENT) {
        fprintf(stderr, "local: cannot read %s\n", path);
        pn_people_close(pp);
        return NULL;
    }
    return pp;
}

void pn_people_close(pn_people *pp)
{
    if (!pp)
        return;
    pthread_mutex_destroy(&pp->lock);
    memset(pp, 0, sizeof(*pp));
    free(pp);
}

/* A fresh unused id. Lock held. */
static int new_id_locked(const pn_people *pp, char out[PN_PERSON_ID_MAX])
{
    for (int tries = 0; tries < 8; tries++) {
        uint8_t raw[4];
        char hex[9];
        if (pn_local_random_bytes(raw, sizeof(raw)) != 0)
            return -1;
        pn_hex_encode(raw, sizeof(raw), hex);
        snprintf(out, PN_PERSON_ID_MAX, "p%s", hex);
        if (index_of_locked(pp, out) < 0)
            return 0;
    }
    return -1;
}

static int add_locked(pn_people *pp, const char *handle, const char *name, int admin,
                      const char *admin_by, int may_load, pn_person *out)
{
    pn_person p = {0};
    int maker = index_of_locked(pp, admin_by);

    if (pn_people_name_clean(handle, PN_PERSON_HANDLE_BYTES, p.handle, sizeof(p.handle)) != 0)
        return PN_PEOPLE_E_BAD_HANDLE;
    if (pn_people_name_clean(name, PN_PERSON_NAME_BYTES, p.name, sizeof(p.name)) != 0)
        return PN_PEOPLE_E_BAD_NAME;
    if (handle_taken_locked(pp, p.handle, -1))
        return PN_PEOPLE_E_HANDLE_TAKEN;
    if (pp->n >= PN_PEOPLE_MAX)
        return PN_PEOPLE_E_FULL;
    /* Exactly one root: only the first person may be an admin with no maker. */
    if ((pp->n == 0 && !admin) || (admin && pp->n > 0 && !(admin_by && admin_by[0])))
        return PN_PEOPLE_E_INVALID;
    if (admin && pp->n > 0 && maker < 0)
        return PN_PEOPLE_E_NOT_FOUND;
    if (admin && pp->n > 0 && !pp->rows[maker].admin)
        return PN_PEOPLE_E_NOT_ADMIN;
    if (new_id_locked(pp, p.id) != 0)
        return PN_PEOPLE_E_IO;
    p.admin = admin != 0;
    if (p.admin && pp->n > 0)
        snprintf(p.admin_by, sizeof(p.admin_by), "%s", admin_by);
    p.may_load = may_load != 0;
    p.created = (long long)time(NULL);
    pp->rows[pp->n++] = p;
    if (save_locked(pp) != 0) {
        pp->n--;
        return PN_PEOPLE_E_IO;
    }
    if (out)
        *out = p;
    return PN_PEOPLE_OK;
}

int pn_people_add(pn_people *pp, const char *handle, const char *name, int admin,
                  const char *admin_by, int may_load, pn_person *out)
{
    int rc = 0;

    if (!pp)
        return PN_PEOPLE_E_INVALID;
    pthread_mutex_lock(&pp->lock);
    rc = add_locked(pp, handle, name, admin, admin_by, may_load, out);
    pthread_mutex_unlock(&pp->lock);
    return rc;
}

int pn_people_owner_or_create(pn_people *pp, const char *handle, const char *name,
                              pn_person *out, int *created_out)
{
    int rc = PN_PEOPLE_E_NOT_FOUND, oldest = -1;

    if (created_out)
        *created_out = 0;
    if (!pp || !out)
        return PN_PEOPLE_E_INVALID;
    pthread_mutex_lock(&pp->lock);
    for (int i = 0; i < pp->n; i++)
        if (pp->rows[i].admin && (oldest < 0 || pp->rows[i].created < pp->rows[oldest].created))
            oldest = i;
    if (oldest >= 0) {
        *out = pp->rows[oldest];
        rc = PN_PEOPLE_OK;
    } else if (pp->n == 0) {
        rc = add_locked(pp, handle, name, 1, NULL, 1, out);
        if (rc == PN_PEOPLE_OK && created_out)
            *created_out = 1;
    }
    pthread_mutex_unlock(&pp->lock);
    return rc;
}

/* Build the changed row for rows[idx] into *next; checks every field first. */
static int prepare_change_locked(const pn_people *pp, const char *actor, int idx,
                                 const pn_person_change *change, pn_person *next)
{
    const pn_person *cur = &pp->rows[idx];
    int a = index_of_locked(pp, actor);

    *next = *cur;
    if (change->name &&
        pn_people_name_clean(change->name, PN_PERSON_NAME_BYTES, next->name, sizeof(next->name)))
        return PN_PEOPLE_E_BAD_NAME;
    if (change->handle && pn_people_name_clean(change->handle, PN_PERSON_HANDLE_BYTES,
                                               next->handle, sizeof(next->handle)))
        return PN_PEOPLE_E_BAD_HANDLE;
    if (change->handle && handle_taken_locked(pp, next->handle, idx))
        return PN_PEOPLE_E_HANDLE_TAKEN;
    if (change->may_load >= 0)
        next->may_load = change->may_load != 0;
    if (change->admin == 1 && !cur->admin) {
        if (a < 0 || !pp->rows[a].admin)
            return PN_PEOPLE_E_NOT_ADMIN;
        next->admin = 1;
        snprintf(next->admin_by, sizeof(next->admin_by), "%s", pp->rows[a].id);
    } else if (change->admin == 0 && cur->admin) {
        int rc = check_admin_leave_locked(pp, actor, idx);
        if (rc != PN_PEOPLE_OK)
            return rc;
        next->admin = 0;
        next->admin_by[0] = '\0';
    }
    return PN_PEOPLE_OK;
}

int pn_people_update(pn_people *pp, const char *actor, const char *id,
                     const pn_person_change *change, pn_person *out)
{
    pn_person saved[PN_PEOPLE_MAX], next;
    int rc = 0, idx = -1;

    if (!pp || !change)
        return PN_PEOPLE_E_INVALID;
    pthread_mutex_lock(&pp->lock);
    idx = index_of_locked(pp, id);
    rc = idx < 0 ? PN_PEOPLE_E_NOT_FOUND : prepare_change_locked(pp, actor, idx, change, &next);
    if (rc == PN_PEOPLE_OK) {
        memcpy(saved, pp->rows, sizeof(saved));
        if (pp->rows[idx].admin && !next.admin)
            reparent_children_locked(pp, idx);
        pp->rows[idx] = next;
        if (save_locked(pp) != 0) {
            memcpy(pp->rows, saved, sizeof(saved));
            rc = PN_PEOPLE_E_IO;
        } else if (out) {
            *out = next;
        }
    }
    pthread_mutex_unlock(&pp->lock);
    return rc;
}

int pn_people_remove(pn_people *pp, const char *actor, const char *id)
{
    pn_person saved[PN_PEOPLE_MAX];
    int rc = PN_PEOPLE_OK, idx = -1, a = -1, n_saved = 0;

    if (!pp)
        return PN_PEOPLE_E_INVALID;
    pthread_mutex_lock(&pp->lock);
    idx = index_of_locked(pp, id);
    a = index_of_locked(pp, actor);
    if (idx < 0)
        rc = PN_PEOPLE_E_NOT_FOUND;
    else if (a < 0 || !pp->rows[a].admin)
        rc = PN_PEOPLE_E_NOT_ADMIN;
    else if (pp->rows[idx].admin)
        rc = check_admin_leave_locked(pp, actor, idx);
    if (rc == PN_PEOPLE_OK) {
        memcpy(saved, pp->rows, sizeof(saved));
        n_saved = pp->n;
        if (pp->rows[idx].admin)
            reparent_children_locked(pp, idx);
        memmove(&pp->rows[idx], &pp->rows[idx + 1], (size_t)(pp->n - idx - 1) * sizeof(pn_person));
        pp->n--;
        if (save_locked(pp) != 0) {
            memcpy(pp->rows, saved, sizeof(saved));
            pp->n = n_saved;
            rc = PN_PEOPLE_E_IO;
        }
    }
    pthread_mutex_unlock(&pp->lock);
    return rc;
}

int pn_people_get(pn_people *pp, const char *id, pn_person *out)
{
    int idx = -1;

    if (!pp)
        return PN_PEOPLE_E_INVALID;
    pthread_mutex_lock(&pp->lock);
    idx = index_of_locked(pp, id);
    if (idx >= 0 && out)
        *out = pp->rows[idx];
    pthread_mutex_unlock(&pp->lock);
    return idx >= 0 ? PN_PEOPLE_OK : PN_PEOPLE_E_NOT_FOUND;
}

int pn_people_find_handle(pn_people *pp, const char *handle, pn_person *out)
{
    int found = -1;

    if (!pp || !handle)
        return PN_PEOPLE_E_INVALID;
    pthread_mutex_lock(&pp->lock);
    for (int i = 0; i < pp->n && found < 0; i++)
        if (ascii_case_equal(pp->rows[i].handle, handle))
            found = i;
    if (found >= 0 && out)
        *out = pp->rows[found];
    pthread_mutex_unlock(&pp->lock);
    return found >= 0 ? PN_PEOPLE_OK : PN_PEOPLE_E_NOT_FOUND;
}

int pn_people_list(pn_people *pp, pn_person *out, int max)
{
    int n = 0;

    if (!pp || !out)
        return 0;
    pthread_mutex_lock(&pp->lock);
    for (int i = 0; i < pp->n && n < max; i++)
        out[n++] = pp->rows[i];
    pthread_mutex_unlock(&pp->lock);
    return n;
}

int pn_people_admin_count(pn_people *pp)
{
    int n = 0;

    if (!pp)
        return 0;
    pthread_mutex_lock(&pp->lock);
    n = admin_count_locked(pp);
    pthread_mutex_unlock(&pp->lock);
    return n;
}
