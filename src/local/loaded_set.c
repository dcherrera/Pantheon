/* SPDX-License-Identifier: AGPL-3.0-or-later */
/* Copyright (C) 2026 David (Corey) Herrera */
/**
 * @file   loaded_set.c
 * @brief  Implementation of the remembered loaded set (see loaded_set.h).
 *
 * @status None
 * @issues None
 * @todo   None
 */

#include "loaded_set.h"

#include <errno.h>
#include <pthread.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "../serve/json_reader.h"
#include "../serve/json_writer.h"
#include "local_file_io.h"

struct pn_loaded_set {
    pthread_mutex_t lock;
    char            path[PN_LOCAL_PATH_MAX];
    pn_loaded_entry entries[PN_LOADED_MAX];
    int             n;
};

/* Absent fields are written as out-of-range sentinels (-1, presence -9). */
static void write_sampling(json_writer *jw, const pn_model_sampling *s)
{
    jw_printf(jw, "{\"temperature\":%.9g,\"top_p\":%.9g,\"top_k\":%d,\"presence_penalty\":%.9g}",
              s->has_temperature ? (double)s->temperature : -1.0,
              s->has_top_p ? (double)s->top_p : -1.0, s->has_top_k ? s->top_k : -1,
              s->has_presence_penalty ? (double)s->presence_penalty : -9.0);
}

static void read_sampling(const jr_value *o, pn_model_sampling *s)
{
    double t = jr_as_number(jr_object_get(o, "temperature"), -1);
    double p = jr_as_number(jr_object_get(o, "top_p"), -1);
    double k = jr_as_number(jr_object_get(o, "top_k"), -1);
    double pp = jr_as_number(jr_object_get(o, "presence_penalty"), -9);

    memset(s, 0, sizeof(*s));
    if ((s->has_temperature = t >= 0)) s->temperature = (float)t;
    if ((s->has_top_p = p >= 0)) s->top_p = (float)p;
    if ((s->has_top_k = k >= 0)) s->top_k = (int)k;
    if ((s->has_presence_penalty = pp > -9)) s->presence_penalty = (float)pp;
}

static void write_entry(json_writer *jw, const pn_loaded_entry *e)
{
    jw_printf(jw, "{\"catalog_id\":\"%s\",\"replica\":%d,\"profile_id\":\"%s\",\"path\":",
              e->catalog_id, e->replica, e->profile_id);
    jw_string(jw, e->path);
    jw_raw(jw, ",\"units\":[");
    for (int i = 0; i < e->n_units; i++)
        jw_printf(jw, "%s%d", i ? "," : "", e->units[i]);
    jw_printf(jw, "],\"hardware\":{\"unit_type\":\"%s\",\"count\":%d,\"same_card\":%s,"
                  "\"per_card\":%d,\"min_mem_per_unit\":%llu},",
              e->hardware.unit_type, e->hardware.count, e->hardware.same_card ? "true" : "false",
              e->hardware.per_card, (unsigned long long)e->hardware.min_mem_per_unit);
    jw_printf(jw, "\"ctx_cap\":%d,\"max_sessions\":%d,\"max_concurrent\":%d,"
                  "\"reasoning_budget\":%d,\"mtp\":%s,\"kv_type\":\"%s\",",
              e->ctx_cap, e->max_sessions, e->max_concurrent, e->reasoning_budget,
              e->mtp ? "true" : "false", e->kv_type);
    if (e->mtp_drafts > 0)   /* absent = the default: older files stay byte for byte */
        jw_printf(jw, "\"mtp_drafts\":%d,", e->mtp_drafts);
    if (e->kv_long[0])       /* absent = none: older files stay byte for byte */
        jw_printf(jw, "\"kv_long\":\"%s\",", e->kv_long);
    if (e->reasoning_reserve > 0)   /* likewise */
        jw_printf(jw, "\"reasoning_reserve\":%d,", e->reasoning_reserve);
    jw_raw(jw, "\"sampling\":");
    write_sampling(jw, &e->sampling);
    jw_raw(jw, ",\"loaded_by\":");
    jw_string(jw, e->loaded_by);
    jw_raw(jw, "}");
}

static int read_entry(const jr_value *o, pn_loaded_entry *e)
{
    const char *id = jr_as_string(jr_object_get(o, "catalog_id"));
    const char *prof = jr_as_string(jr_object_get(o, "profile_id"));
    const char *path = jr_as_string(jr_object_get(o, "path"));
    const jr_value *units = jr_object_get(o, "units"), *hw = jr_object_get(o, "hardware");
    const char *type = jr_as_string(jr_object_get(hw, "unit_type"));

    memset(e, 0, sizeof(*e));
    if (!id || !prof || !path || !type || jr_array_len(units) > PN_UNIT_PLACE_MAX)
        return -1;
    snprintf(e->catalog_id, sizeof(e->catalog_id), "%s", id);
    e->replica = (int)jr_as_number(jr_object_get(o, "replica"), 1);   /* before replicas: 1 */
    if (e->replica < 1)
        return -1;
    snprintf(e->profile_id, sizeof(e->profile_id), "%s", prof);
    snprintf(e->path, sizeof(e->path), "%s", path);
    for (int i = 0; i < jr_array_len(units); i++)
        e->units[e->n_units++] = (int)jr_as_number(jr_array_get(units, i), -1);
    snprintf(e->hardware.unit_type, sizeof(e->hardware.unit_type), "%s", type);
    e->hardware.count = (int)jr_as_number(jr_object_get(hw, "count"), 1);
    e->hardware.same_card = jr_as_bool(jr_object_get(hw, "same_card"), false);
    e->hardware.per_card = (int)jr_as_number(jr_object_get(hw, "per_card"), 0);
    e->hardware.min_mem_per_unit = (uint64_t)jr_as_number(jr_object_get(hw, "min_mem_per_unit"), 0);
    e->ctx_cap = (int)jr_as_number(jr_object_get(o, "ctx_cap"), 0);
    e->max_sessions = (int)jr_as_number(jr_object_get(o, "max_sessions"), 0);
    e->max_concurrent = (int)jr_as_number(jr_object_get(o, "max_concurrent"), 0);
    e->reasoning_budget = (int)jr_as_number(jr_object_get(o, "reasoning_budget"), 0);
    e->reasoning_reserve = (int)jr_as_number(jr_object_get(o, "reasoning_reserve"), 0);
    if (e->reasoning_reserve < 0)
        e->reasoning_reserve = 0;
    e->mtp = jr_as_bool(jr_object_get(o, "mtp"), false);
    e->mtp_drafts = (int)jr_as_number(jr_object_get(o, "mtp_drafts"), 0);   /* before: absent */
    if (e->mtp_drafts < 0 || e->mtp_drafts > 15)
        return -1;
    const char *kv = jr_as_string(jr_object_get(o, "kv_type"));   /* before: absent */
    if (kv && strlen(kv) < sizeof(e->kv_type))
        snprintf(e->kv_type, sizeof(e->kv_type), "%s", kv);
    const char *kv_long = jr_as_string(jr_object_get(o, "kv_long"));   /* before: absent */
    if (kv_long && strlen(kv_long) < sizeof(e->kv_long))
        snprintf(e->kv_long, sizeof(e->kv_long), "%s", kv_long);
    read_sampling(jr_object_get(o, "sampling"), &e->sampling);
    const jr_value *by = jr_object_get(o, "loaded_by");   /* before people: absent */
    if (by && (!jr_as_string(by) || strlen(jr_as_string(by)) >= sizeof(e->loaded_by)))
        return -1;
    if (by)
        snprintf(e->loaded_by, sizeof(e->loaded_by), "%s", jr_as_string(by));
    return 0;
}

static int save_locked(const pn_loaded_set *s)
{
    json_writer jw;
    int rc = -1;

    if (!jw_init(&jw))
        return -1;
    jw_raw(&jw, "{\"version\":1,\"models\":[");
    for (int i = 0; i < s->n; i++) {
        if (i)
            jw_raw(&jw, ",");
        write_entry(&jw, &s->entries[i]);
    }
    jw_raw(&jw, "]}\n");
    rc = pn_local_write_file_atomic(s->path, jw.data, jw.len, 0600);
    jw_free(&jw);
    return rc;
}

pn_loaded_set *pn_loaded_set_open(const char *path)
{
    pn_loaded_set *s = calloc(1, sizeof(*s));
    char *text = NULL;

    if (!s || snprintf(s->path, sizeof(s->path), "%s", path) >= (int)sizeof(s->path)) {
        free(s);
        return NULL;
    }
    pthread_mutex_init(&s->lock, NULL);
    if (pn_local_read_file(path, 1 << 20, &text, NULL) != 0) {
        if (errno == ENOENT)
            return s;
        pn_loaded_set_close(s);
        return NULL;
    }
    jr_value *root = jr_parse(text);
    const jr_value *models = jr_object_get(root, "models");
    int bad = jr_kind(models) != JR_ARRAY;
    for (int i = 0; !bad && i < jr_array_len(models) && s->n < PN_LOADED_MAX; i++)
        bad = read_entry(jr_array_get(models, i), &s->entries[s->n++]) != 0;
    jr_free(root);
    free(text);
    if (bad) {
        fprintf(stderr, "local: %s is malformed; refusing to start\n", path);
        pn_loaded_set_close(s);
        return NULL;
    }
    return s;
}

void pn_loaded_set_close(pn_loaded_set *s)
{
    if (!s)
        return;
    pthread_mutex_destroy(&s->lock);
    free(s);
}

int pn_loaded_set_add(pn_loaded_set *s, const pn_loaded_entry *entry)
{
    pn_loaded_entry copy = *entry;
    const pn_loaded_entry *e = &copy;
    int rc = -1;

    if (copy.replica < 1)
        copy.replica = 1;   /* a caller that predates replicas means the first */
    pthread_mutex_lock(&s->lock);
    for (int i = 0; i < s->n; i++) {
        if (strcmp(s->entries[i].catalog_id, e->catalog_id) != 0 ||
            s->entries[i].replica != e->replica)
            continue;
        memmove(&s->entries[i], &s->entries[i + 1], (size_t)(s->n - i - 1) * sizeof(*e));
        s->n--;
        break;
    }
    if (s->n < PN_LOADED_MAX) {
        s->entries[s->n++] = *e;
        rc = save_locked(s);
    }
    pthread_mutex_unlock(&s->lock);
    return rc;
}

int pn_loaded_set_remove(pn_loaded_set *s, const char *catalog_id, int replica)
{
    int removed = 0, rc;

    pthread_mutex_lock(&s->lock);
    for (int i = 0; i < s->n;) {
        if (strcmp(s->entries[i].catalog_id, catalog_id) != 0 ||
            (replica > 0 && s->entries[i].replica != replica)) {
            i++;
            continue;
        }
        memmove(&s->entries[i], &s->entries[i + 1], (size_t)(s->n - i - 1) * sizeof(s->entries[0]));
        s->n--;
        removed++;
    }
    rc = !removed ? 1 : save_locked(s) == 0 ? 0 : -1;
    pthread_mutex_unlock(&s->lock);
    return rc;
}

int pn_loaded_set_next_replica(pn_loaded_set *s, const char *catalog_id)
{
    int replica = 1;
    bool used = true;

    pthread_mutex_lock(&s->lock);
    for (; used; replica += used) {
        used = false;
        for (int i = 0; i < s->n && !used; i++)
            used = strcmp(s->entries[i].catalog_id, catalog_id) == 0 &&
                   s->entries[i].replica == replica;
    }
    pthread_mutex_unlock(&s->lock);
    return replica;
}

int pn_loaded_set_claim_unowned(pn_loaded_set *s, const char *person)
{
    int claimed = 0, rc = 0;

    if (!person || !*person || strlen(person) >= PN_LOADED_PERSON_MAX)
        return 0;
    pthread_mutex_lock(&s->lock);
    for (int i = 0; i < s->n; i++) {
        if (s->entries[i].loaded_by[0])
            continue;
        snprintf(s->entries[i].loaded_by, sizeof(s->entries[i].loaded_by), "%s", person);
        claimed++;
    }
    if (claimed)
        rc = save_locked(s);
    pthread_mutex_unlock(&s->lock);
    return rc == 0 ? claimed : -1;
}

int pn_loaded_set_list(pn_loaded_set *s, pn_loaded_entry *out, int max)
{
    int n = 0;

    pthread_mutex_lock(&s->lock);
    for (int i = 0; i < s->n && n < max; i++)
        out[n++] = s->entries[i];
    pthread_mutex_unlock(&s->lock);
    return n;
}
