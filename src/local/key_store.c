/* SPDX-License-Identifier: AGPL-3.0-or-later */
/* Copyright (C) 2026 David (Corey) Herrera */
/**
 * @file   key_store.c
 * @brief  Implementation of the client key store (see key_store.h).
 *
 * @status None
 * @issues None
 * @todo   None
 */

#include "key_store.h"

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
#include "people_store.h"
#include "sha256.h"

#define KEY_PREFIX "pnk_"

typedef struct {
    pn_key_info info;
    uint8_t     hash[PN_SHA256_LEN];
} key_row;

struct pn_key_store {
    pthread_mutex_t lock;
    char            path[PN_LOCAL_PATH_MAX];
    key_row         rows[PN_KEY_MAX_KEYS];
    int             n;
    int             v1;          /* read from a v1 file, not migrated yet */
};

const char *pn_key_kind_name(pn_key_kind kind)
{
    return kind == PN_KEY_KIND_DEVICE ? "device" : kind == PN_KEY_KIND_INFERENCE ? "inference"
                                                                                : "none";
}

static pn_key_kind kind_from_name(const char *s)
{
    if (s && strcmp(s, "device") == 0) return PN_KEY_KIND_DEVICE;
    if (s && strcmp(s, "inference") == 0) return PN_KEY_KIND_INFERENCE;
    return PN_KEY_KIND_NONE;
}

static int hex_tail_valid(const char *s, char lead, size_t hex_len)
{
    if (!s || s[0] != lead || strlen(s) != hex_len + 1)
        return 0;
    for (size_t i = 1; i <= hex_len; i++)
        if (!((s[i] >= '0' && s[i] <= '9') || (s[i] >= 'a' && s[i] <= 'f')))
            return 0;
    return 1;
}

int pn_key_device_id_valid(const char *s)
{
    return hex_tail_valid(s, 'd', 16);
}

/* Keep printable ASCII only, so a label can never break the log or the JSON. */
static void copy_label(char *dst, const char *src)
{
    size_t j = 0;

    for (const char *p = src ? src : ""; *p && j < PN_KEY_LABEL_MAX - 1; p++)
        if (*p >= 0x20 && *p < 0x7f && *p != '"' && *p != '\\')
            dst[j++] = *p;
    dst[j] = '\0';
}

/* Serialize every row (v2) and replace the file atomically (0600). Lock held. */
static int save_locked(const pn_key_store *ks)
{
    json_writer jw;
    int rc = -1;

    if (!jw_init(&jw))
        return -1;
    jw_raw(&jw, "{\"version\":2,\"keys\":[");
    for (int i = 0; i < ks->n; i++) {
        const pn_key_info *k = &ks->rows[i].info;
        char hex[PN_SHA256_HEX_LEN + 1];
        pn_hex_encode(ks->rows[i].hash, PN_SHA256_LEN, hex);
        jw_printf(&jw, "%s\n  {\"id\":\"%s\",\"sha256\":\"%s\",\"kind\":\"%s\",\"person\":\"%s\"",
                  i ? "," : "", k->id, hex, pn_key_kind_name(k->kind), k->person);
        if (k->kind == PN_KEY_KIND_DEVICE) {
            jw_printf(&jw, ",\"device_id\":\"%s\",\"device_name\":", k->device_id);
            jw_string(&jw, k->device_name);
        }
        jw_raw(&jw, ",\"label\":");
        jw_string(&jw, k->label);
        jw_printf(&jw, ",\"created\":%lld}", k->created);
    }
    jw_raw(&jw, "]}\n");
    rc = pn_local_write_file_atomic(ks->path, jw.data, jw.len, 0600);
    jw_free(&jw);
    return rc;
}

/* A v1 row's role, mapped to v2 (admin → device named after its label). */
static int parse_v1_role(const jr_value *k, pn_key_info *info)
{
    const char *role = jr_as_string(jr_object_get(k, "role"));

    if (role && strcmp(role, "inference") == 0) {
        info->kind = PN_KEY_KIND_INFERENCE;
        return 0;
    }
    if (!role || strcmp(role, "admin") != 0)
        return -1;
    info->kind = PN_KEY_KIND_DEVICE;
    if (pn_people_name_clean(info->label, PN_KEY_DEVICE_NAME_MAX - 1, info->device_name,
                             sizeof(info->device_name)) != 0)
        snprintf(info->device_name, sizeof(info->device_name), "device");
    return 0;
}

/* A v2 row's kind, person and device fields. */
static int parse_v2_fields(const jr_value *k, pn_key_info *info)
{
    const char *person = jr_as_string(jr_object_get(k, "person"));
    const char *device_id = jr_as_string(jr_object_get(k, "device_id"));

    info->kind = kind_from_name(jr_as_string(jr_object_get(k, "kind")));
    if (info->kind == PN_KEY_KIND_NONE || !hex_tail_valid(person, 'p', 8))
        return -1;
    snprintf(info->person, sizeof(info->person), "%s", person);
    if (info->kind != PN_KEY_KIND_DEVICE)
        return 0;
    if (!device_id || (device_id[0] && !pn_key_device_id_valid(device_id)) ||
        pn_people_name_clean(jr_as_string(jr_object_get(k, "device_name")),
                             PN_KEY_DEVICE_NAME_MAX - 1, info->device_name,
                             sizeof(info->device_name)) != 0)
        return -1;
    snprintf(info->device_id, sizeof(info->device_id), "%s", device_id);
    return 0;
}

/* Parse keys.json (v1 or v2) into ks. Returns 0, or -1 on a malformed file. */
static int load_rows(pn_key_store *ks, const char *text)
{
    jr_value *root = jr_parse(text);
    const jr_value *keys = jr_object_get(root, "keys");
    double version = jr_as_number(jr_object_get(root, "version"), 1);
    int rc = -1;

    if (!root || jr_kind(keys) != JR_ARRAY || jr_array_len(keys) > PN_KEY_MAX_KEYS ||
        (version != 1 && version != 2))
        goto done;
    ks->v1 = version == 1;
    for (int i = 0; i < jr_array_len(keys); i++) {
        const jr_value *k = jr_array_get(keys, i);
        const char *id = jr_as_string(jr_object_get(k, "id"));
        const char *hash = jr_as_string(jr_object_get(k, "sha256"));
        key_row *r = &ks->rows[ks->n];
        memset(r, 0, sizeof(*r));
        if (!id || !hash || strlen(id) >= PN_KEY_ID_MAX || strlen(hash) != PN_SHA256_HEX_LEN ||
            pn_hex_decode(hash, r->hash, PN_SHA256_LEN) != 0)
            goto done;
        snprintf(r->info.id, sizeof(r->info.id), "%s", id);
        copy_label(r->info.label, jr_as_string(jr_object_get(k, "label")));
        r->info.created = (long long)jr_as_number(jr_object_get(k, "created"), 0);
        if ((ks->v1 ? parse_v1_role(k, &r->info) : parse_v2_fields(k, &r->info)) != 0)
            goto done;
        ks->n++;
    }
    rc = 0;
done:
    jr_free(root);
    return rc;
}

pn_key_store *pn_key_store_open(const char *path)
{
    pn_key_store *ks = calloc(1, sizeof(*ks));
    char *text = NULL;

    if (!ks || snprintf(ks->path, sizeof(ks->path), "%s", path) >= (int)sizeof(ks->path)) {
        free(ks);
        return NULL;
    }
    pthread_mutex_init(&ks->lock, NULL);
    if (pn_local_read_file(path, 1 << 20, &text, NULL) == 0) {
        int bad = load_rows(ks, text);
        free(text);
        if (bad || chmod(path, 0600) != 0) {
            fprintf(stderr, "local: %s is malformed or not ours; refusing to start\n", path);
            pn_key_store_close(ks);
            return NULL;
        }
    } else if (errno != ENOENT) {
        fprintf(stderr, "local: cannot read %s\n", path);
        pn_key_store_close(ks);
        return NULL;
    }
    return ks;
}

void pn_key_store_close(pn_key_store *ks)
{
    if (!ks)
        return;
    pthread_mutex_destroy(&ks->lock);
    memset(ks, 0, sizeof(*ks));
    free(ks);
}

int pn_key_store_is_v1(pn_key_store *ks)
{
    int v1 = 0;

    pthread_mutex_lock(&ks->lock);
    v1 = ks->v1;
    pthread_mutex_unlock(&ks->lock);
    return v1;
}

int pn_key_store_count(pn_key_store *ks)
{
    int n = 0;

    pthread_mutex_lock(&ks->lock);
    n = ks->n;
    pthread_mutex_unlock(&ks->lock);
    return n;
}

/* Copy the v1 file as it is on disk to `backup_path` (0600). */
static int keep_v1_copy(const char *path, const char *backup_path)
{
    char *text = NULL;
    size_t len = 0;
    int rc = -1;

    if (pn_local_read_file(path, 1 << 20, &text, &len) != 0)
        return errno == ENOENT ? 0 : -1;
    rc = pn_local_write_file_atomic(backup_path, text, len, 0600);
    free(text);
    return rc;
}

int pn_key_store_migrate_v1(pn_key_store *ks, const char *person, const char *backup_path)
{
    int rc = PN_KEY_OK;

    if (!person || !backup_path)
        return PN_KEY_E_INVALID;
    pthread_mutex_lock(&ks->lock);
    if (!ks->v1)
        rc = PN_KEY_E_INVALID;
    else if (keep_v1_copy(ks->path, backup_path) != 0)
        rc = PN_KEY_E_IO;
    if (rc == PN_KEY_OK) {
        for (int i = 0; i < ks->n; i++)
            snprintf(ks->rows[i].info.person, sizeof(ks->rows[i].info.person), "%s", person);
        if (save_locked(ks) == 0) {
            ks->v1 = 0;
        } else {
            for (int i = 0; i < ks->n; i++)
                ks->rows[i].info.person[0] = '\0';
            rc = PN_KEY_E_IO;
        }
    }
    pthread_mutex_unlock(&ks->lock);
    return rc;
}

/* Mint a row from a prepared info (id and created are filled here). */
static int mint_row(pn_key_store *ks, pn_key_info *info, char secret_out[PN_KEY_SECRET_MAX])
{
    uint8_t raw[32], id_raw[4];
    char hex[65], id_hex[9];
    int rc = PN_KEY_E_FULL;

    if (pn_local_random_bytes(raw, sizeof(raw)) != 0 ||
        pn_local_random_bytes(id_raw, sizeof(id_raw)) != 0)
        return PN_KEY_E_IO;
    pn_hex_encode(raw, sizeof(raw), hex);
    pn_hex_encode(id_raw, sizeof(id_raw), id_hex);
    snprintf(info->id, sizeof(info->id), "k%s", id_hex);
    info->created = (long long)time(NULL);
    pthread_mutex_lock(&ks->lock);
    if (ks->v1) {
        rc = PN_KEY_E_INVALID;
    } else if (ks->n < PN_KEY_MAX_KEYS) {
        key_row *r = &ks->rows[ks->n];
        memset(r, 0, sizeof(*r));
        snprintf(secret_out, PN_KEY_SECRET_MAX, KEY_PREFIX "%s", hex);
        pn_sha256(secret_out, strlen(secret_out), r->hash);
        r->info = *info;
        ks->n++;
        rc = save_locked(ks) == 0 ? PN_KEY_OK : PN_KEY_E_IO;
        if (rc != PN_KEY_OK)
            ks->n--;
    }
    pthread_mutex_unlock(&ks->lock);
    memset(raw, 0, sizeof(raw));
    memset(hex, 0, sizeof(hex));
    if (rc != PN_KEY_OK)
        memset(secret_out, 0, PN_KEY_SECRET_MAX);
    return rc;
}

int pn_key_store_mint_device(pn_key_store *ks, const char *person, const char *device_id,
                             const char *device_name, const char *label,
                             char secret_out[PN_KEY_SECRET_MAX], pn_key_info *info_out)
{
    pn_key_info info = {0};
    int rc = 0;

    if (!hex_tail_valid(person, 'p', 8) ||
        (device_id && device_id[0] && !pn_key_device_id_valid(device_id)) ||
        pn_people_name_clean(device_name, PN_KEY_DEVICE_NAME_MAX - 1, info.device_name,
                             sizeof(info.device_name)) != 0)
        return PN_KEY_E_INVALID;
    info.kind = PN_KEY_KIND_DEVICE;
    snprintf(info.person, sizeof(info.person), "%s", person);
    snprintf(info.device_id, sizeof(info.device_id), "%s", device_id ? device_id : "");
    copy_label(info.label, label);
    rc = mint_row(ks, &info, secret_out);
    if (rc == PN_KEY_OK && info_out)
        *info_out = info;
    return rc;
}

int pn_key_store_mint_inference(pn_key_store *ks, const char *person, const char *label,
                                char secret_out[PN_KEY_SECRET_MAX], pn_key_info *info_out)
{
    pn_key_info info = {0};
    int rc = 0;

    if (!hex_tail_valid(person, 'p', 8))
        return PN_KEY_E_INVALID;
    info.kind = PN_KEY_KIND_INFERENCE;
    snprintf(info.person, sizeof(info.person), "%s", person);
    copy_label(info.label, label);
    rc = mint_row(ks, &info, secret_out);
    if (rc == PN_KEY_OK && info_out)
        *info_out = info;
    return rc;
}

/* Constant-time equality of two digests. */
static int digest_equal(const uint8_t *a, const uint8_t *b)
{
    uint8_t diff = 0;

    for (int i = 0; i < PN_SHA256_LEN; i++)
        diff |= (uint8_t)(a[i] ^ b[i]);
    return diff == 0;
}

int pn_key_store_verify(pn_key_store *ks, const char *secret, pn_key_info *info_out)
{
    uint8_t h[PN_SHA256_LEN];
    int found = -1;

    if (!secret || strncmp(secret, KEY_PREFIX, 4) != 0)
        return -1;
    pn_sha256(secret, strlen(secret), h);
    pthread_mutex_lock(&ks->lock);
    for (int i = 0; i < ks->n && !ks->v1; i++) {
        if (digest_equal(h, ks->rows[i].hash)) {
            if (info_out)
                *info_out = ks->rows[i].info;
            found = 0;
        }
    }
    pthread_mutex_unlock(&ks->lock);
    return found;
}

static int index_of_locked(const pn_key_store *ks, const char *id)
{
    for (int i = 0; id && i < ks->n; i++)
        if (strcmp(ks->rows[i].info.id, id) == 0)
            return i;
    return -1;
}

int pn_key_store_get(pn_key_store *ks, const char *id, pn_key_info *info_out)
{
    int i = -1;

    pthread_mutex_lock(&ks->lock);
    i = index_of_locked(ks, id);
    if (i >= 0 && info_out)
        *info_out = ks->rows[i].info;
    pthread_mutex_unlock(&ks->lock);
    return i >= 0 ? PN_KEY_OK : PN_KEY_E_NOT_FOUND;
}

int pn_key_store_set_device(pn_key_store *ks, const char *id, const char *device_id,
                            const char *device_name, pn_key_info *info_out)
{
    char name[PN_KEY_DEVICE_NAME_MAX];
    int rc = PN_KEY_OK, i = -1;

    if ((device_id && !pn_key_device_id_valid(device_id)) ||
        (device_name && pn_people_name_clean(device_name, PN_KEY_DEVICE_NAME_MAX - 1, name,
                                             sizeof(name)) != 0))
        return PN_KEY_E_INVALID;
    pthread_mutex_lock(&ks->lock);
    i = index_of_locked(ks, id);
    if (i < 0) {
        rc = PN_KEY_E_NOT_FOUND;
    } else {
        pn_key_info *k = &ks->rows[i].info, saved = *k;
        if (k->kind != PN_KEY_KIND_DEVICE)
            rc = PN_KEY_E_INVALID;
        else if (device_id && k->device_id[0] && strcmp(k->device_id, device_id) != 0)
            rc = PN_KEY_E_DEVICE_MISMATCH;
        if (rc == PN_KEY_OK) {
            if (device_id)
                snprintf(k->device_id, sizeof(k->device_id), "%s", device_id);
            if (device_name)
                snprintf(k->device_name, sizeof(k->device_name), "%s", name);
            if (save_locked(ks) != 0) {
                *k = saved;
                rc = PN_KEY_E_IO;
            } else if (info_out) {
                *info_out = *k;
            }
        }
    }
    pthread_mutex_unlock(&ks->lock);
    return rc;
}

int pn_key_store_revoke(pn_key_store *ks, const char *id)
{
    int rc = PN_KEY_E_NOT_FOUND, i = -1;

    pthread_mutex_lock(&ks->lock);
    i = index_of_locked(ks, id);
    if (i >= 0) {
        key_row gone = ks->rows[i];
        memmove(&ks->rows[i], &ks->rows[i + 1], (size_t)(ks->n - i - 1) * sizeof(key_row));
        ks->n--;
        rc = save_locked(ks) == 0 ? PN_KEY_OK : PN_KEY_E_IO;
        if (rc != PN_KEY_OK) {   /* keep memory and file in agreement */
            memmove(&ks->rows[i + 1], &ks->rows[i], (size_t)(ks->n - i) * sizeof(key_row));
            ks->rows[i] = gone;
            ks->n++;
        }
    }
    pthread_mutex_unlock(&ks->lock);
    return rc;
}

int pn_key_store_revoke_person(pn_key_store *ks, const char *person)
{
    key_row *saved = NULL;
    int kept = 0, gone = 0, n_saved = 0;

    if (!person || !person[0])
        return 0;
    saved = malloc(sizeof(ks->rows));
    if (!saved)
        return PN_KEY_E_IO;
    pthread_mutex_lock(&ks->lock);
    memcpy(saved, ks->rows, sizeof(ks->rows));
    n_saved = ks->n;
    for (int i = 0; i < ks->n; i++) {
        if (strcmp(ks->rows[i].info.person, person) == 0)
            gone++;
        else
            ks->rows[kept++] = ks->rows[i];
    }
    ks->n = kept;
    if (gone > 0 && save_locked(ks) != 0) {
        memcpy(ks->rows, saved, sizeof(ks->rows));
        ks->n = n_saved;
        gone = PN_KEY_E_IO;
    }
    pthread_mutex_unlock(&ks->lock);
    memset(saved, 0, sizeof(ks->rows));
    free(saved);
    return gone;
}

int pn_key_store_list(pn_key_store *ks, pn_key_info *out, int max)
{
    int n = 0;

    pthread_mutex_lock(&ks->lock);
    for (int i = 0; i < ks->n && n < max; i++)
        out[n++] = ks->rows[i].info;
    pthread_mutex_unlock(&ks->lock);
    return n;
}

int pn_key_store_list_person(pn_key_store *ks, const char *person, pn_key_info *out, int max)
{
    int n = 0;

    pthread_mutex_lock(&ks->lock);
    for (int i = 0; person && i < ks->n && n < max; i++)
        if (strcmp(ks->rows[i].info.person, person) == 0)
            out[n++] = ks->rows[i].info;
    pthread_mutex_unlock(&ks->lock);
    return n;
}
