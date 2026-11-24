/* SPDX-License-Identifier: AGPL-3.0-or-later */
/* Copyright (C) 2026 David (Corey) Herrera */
/**
 * @file   session_registry.c
 * @brief  Implementation of the session registry (see session_registry.h).
 *
 * @status None
 * @issues None
 * @todo   None
 */

#include "session_registry.h"

#include <errno.h>
#include <pthread.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include <arpa/inet.h>
#include <sys/stat.h>

#include "../../lib/monocypher/monocypher.h"
#include "../serve/json_reader.h"
#include "../serve/json_writer.h"
#include "local_file_io.h"

/* One row: what the box shows, plus what it keeps in memory only. */
typedef struct {
    pn_reg_row_info info;
    char            host_secret[PN_REG_HEX32_MAX];
    char            host_pub[PN_REG_HEX32_MAX];
    pn_reg_event    queue[PN_REG_QUEUE_MAX];
    int             n_queue;
} reg_row;

/* A deleted key: its host's next wait gets `close` with this reason. */
typedef struct {
    char                key[PN_REG_KEY_MAX];
    char                owner[PN_REG_PERSON_MAX];
    long long           seq;
    long long           expires;
    pn_reg_close_reason reason;
} reg_tombstone;

struct pn_reg {
    pthread_mutex_t    lock;
    char               path[PN_LOCAL_PATH_MAX];
    reg_row           *rows[PN_REG_MAX_ROWS];
    int                n;
    reg_tombstone      tombs[PN_REG_TOMBSTONES_MAX];
    int                disconnect_s;
    unsigned long long rev;
    long long          dirty_since;            /* 0 = clean */
};

static pn_reg_role role_locked(const reg_row *r, const char *person);

static const char *const k_state_names[] = { "thinking", "needs-you", "idle", "disconnected" };
static const char *const k_kind_names[] = { "session", "desktop" };
static const char *const k_role_names[] = { "none", "view", "control", "owner" };
static const char *const k_event_names[] = { "relay", "shares", "reregister", "resync", "close" };
static const char *const k_reason_names[] = { "shutdown", "removed", "revoked" };

#define NAME_OF(table, v) \
    ((int)(v) >= 0 && (size_t)(v) < sizeof(table) / sizeof(table[0]) ? table[v] : "unknown")

const char *pn_reg_state_name(pn_reg_state state) { return NAME_OF(k_state_names, state); }
const char *pn_reg_kind_name(pn_reg_kind kind) { return NAME_OF(k_kind_names, kind); }
const char *pn_reg_role_name(pn_reg_role role) { return NAME_OF(k_role_names, role); }
const char *pn_reg_event_type_name(pn_reg_event_type t) { return NAME_OF(k_event_names, t); }
const char *pn_reg_close_reason_name(pn_reg_close_reason r) { return NAME_OF(k_reason_names, r); }

static int parse_name(const char *s, const char *const *table, int n)
{
    for (int i = 0; s && i < n; i++)
        if (strcmp(s, table[i]) == 0)
            return i;
    return -1;
}

int pn_reg_state_parse(const char *s, pn_reg_state *out)
{
    int v = parse_name(s, k_state_names, 4);
    if (v < 0)
        return -1;
    *out = (pn_reg_state)v;
    return 0;
}

int pn_reg_kind_parse(const char *s, pn_reg_kind *out)
{
    int v = parse_name(s, k_kind_names, 2);
    if (v < 0)
        return -1;
    *out = (pn_reg_kind)v;
    return 0;
}

int pn_reg_role_parse(const char *s, pn_reg_role *out)
{
    int v = parse_name(s, k_role_names, 4);
    if (v < 0)
        return -1;
    *out = (pn_reg_role)v;
    return 0;
}

/* `s` is `prefix` + exactly `n` lowercase hex digits. */
static int prefixed_hex(const char *s, const char *prefix, size_t n)
{
    size_t p = strlen(prefix);

    if (!s || strncmp(s, prefix, p) != 0 || strlen(s) != p + n)
        return 0;
    for (size_t i = p; i < p + n; i++)
        if (!((s[i] >= '0' && s[i] <= '9') || (s[i] >= 'a' && s[i] <= 'f')))
            return 0;
    return 1;
}

int pn_reg_key_valid(const char *s) { return prefixed_hex(s, "c-", 12); }

static int hex64_valid(const char *s)
{
    if (!s || strlen(s) != 64)
        return 0;
    for (int i = 0; i < 64; i++)
        if (!((s[i] >= '0' && s[i] <= '9') || (s[i] >= 'a' && s[i] <= 'f') ||
              (s[i] >= 'A' && s[i] <= 'F')))
            return 0;
    return 1;
}

int pn_reg_addr_valid(const char *s)
{
    char host[PN_REG_ADDR_BYTES + 1];
    const char *colon = s ? strrchr(s, ':') : NULL;
    unsigned char raw[16];
    size_t hn = colon ? (size_t)(colon - s) : 0;
    long port = 0;
    char *end = NULL;

    if (!colon || strlen(s) > PN_REG_ADDR_BYTES || hn == 0 || !colon[1])
        return 0;
    port = strtol(colon + 1, &end, 10);
    if (*end || port < 1 || port > 65535 || colon[1] == '+' || colon[1] == '-')
        return 0;
    if (s[0] == '[') {
        if (hn < 3 || s[hn - 1] != ']')
            return 0;
        memcpy(host, s + 1, hn - 2);
        host[hn - 2] = '\0';
        return inet_pton(AF_INET6, host, raw) == 1;
    }
    memcpy(host, s, hn);
    host[hn] = '\0';
    return inet_pton(AF_INET, host, raw) == 1;
}

/* A field copied into a fixed buffer: -1 when it is longer than max_bytes. */
static int copy_text(char *out, size_t cap, const char *in, size_t max_bytes)
{
    size_t n = in ? strlen(in) : 0;

    if (n > max_bytes || n >= cap)
        return -1;
    memcpy(out, in ? in : "", n);
    out[n] = '\0';
    return 0;
}

static int host_valid(const pn_reg_host *h)
{
    return h && pn_reg_key_valid(h->key) && prefixed_hex(h->owner, "p", 8) &&
           prefixed_hex(h->origin_device, "d", 16) && pn_reg_addr_valid(h->host_addr) &&
           (!h->title || strlen(h->title) <= PN_REG_TITLE_BYTES) &&
           (!h->model || strlen(h->model) <= PN_REG_MODEL_BYTES) &&
           h->instance >= 0 && h->instance <= PN_REG_INSTANCE_MAX &&
           (h->kind == PN_REG_KIND_SESSION || h->kind == PN_REG_KIND_DESKTOP) &&
           h->state >= PN_REG_STATE_THINKING && h->state < PN_REG_STATE_DISCONNECTED &&
           hex64_valid(h->host_secret) && hex64_valid(h->host_pub);
}

static int index_locked(const pn_reg *reg, const char *key)
{
    for (int i = 0; key && i < reg->n; i++)
        if (strcmp(reg->rows[i]->info.key, key) == 0)
            return i;
    return -1;
}

static reg_tombstone *tomb_locked(pn_reg *reg, const char *key)
{
    for (int i = 0; key && i < PN_REG_TOMBSTONES_MAX; i++)
        if (reg->tombs[i].key[0] && strcmp(reg->tombs[i].key, key) == 0)
            return &reg->tombs[i];
    return NULL;
}

/* Remember a deleted key (the slot of the oldest tombstone when all are used). */
static void add_tomb_locked(pn_reg *reg, const reg_row *row, pn_reg_close_reason reason,
                            long long now)
{
    reg_tombstone *t = &reg->tombs[0];

    for (int i = 0; i < PN_REG_TOMBSTONES_MAX; i++) {
        if (!reg->tombs[i].key[0]) {
            t = &reg->tombs[i];
            break;
        }
        if (reg->tombs[i].expires < t->expires)
            t = &reg->tombs[i];
    }
    snprintf(t->key, sizeof(t->key), "%s", row->info.key);
    snprintf(t->owner, sizeof(t->owner), "%s", row->info.owner);
    t->seq = row->info.seq + 1;
    t->expires = now + PN_REG_TOMBSTONE_S;
    t->reason = reason;
}

static void mark_dirty_locked(pn_reg *reg, long long now)
{
    if (!reg->dirty_since)
        reg->dirty_since = now > 0 ? now : 1;
}

static void write_row(json_writer *jw, const reg_row *r, int first)
{
    const pn_reg_row_info *i = &r->info;

    jw_printf(jw, "%s\n  {\"key\":\"%s\",\"kind\":\"%s\",\"owner\":\"%s\",\"origin_device\":\"%s\","
              "\"host_addr\":", first ? "" : ",", i->key, pn_reg_kind_name(i->kind), i->owner,
              i->origin_device);
    jw_string(jw, i->host_addr);
    jw_raw(jw, ",\"title\":");
    jw_string(jw, i->title);
    jw_raw(jw, ",\"model\":");
    jw_string(jw, i->model);
    jw_printf(jw, ",\"instance\":%d,\"state\":\"%s\",\"shares\":[", i->instance,
              pn_reg_state_name(i->state));
    for (int s = 0; s < i->n_shares; s++)
        jw_printf(jw, "%s{\"person\":\"%s\",\"role\":\"%s\"}", s ? "," : "", i->shares[s].person,
                  pn_reg_role_name(i->shares[s].role));
    jw_printf(jw, "],\"registered_at\":%lld,\"last_beat\":%lld}", i->registered_at, i->last_beat);
}

static int save_locked(pn_reg *reg)
{
    json_writer jw;
    int rc = 0;

    if (!jw_init(&jw))
        return PN_REG_E_IO;
    jw_raw(&jw, "{\"version\":1,\"rows\":[");
    for (int i = 0; i < reg->n; i++)
        write_row(&jw, reg->rows[i], i == 0);
    jw_raw(&jw, "]}\n");
    rc = pn_local_write_file_atomic(reg->path, jw.data, jw.len, 0600);
    jw_free(&jw);
    if (rc != 0) {
        fprintf(stderr, "local: could not write %s\n", reg->path);
        return PN_REG_E_IO;
    }
    reg->dirty_since = 0;
    return PN_REG_OK;
}

/* Save at once; on failure the change stays in memory and the file dirty. */
static int save_now_locked(pn_reg *reg, long long now)
{
    if (save_locked(reg) == PN_REG_OK)
        return PN_REG_OK;
    mark_dirty_locked(reg, now);
    return PN_REG_E_IO;
}

/* One sessions.json row into *r; -1 when it breaks a rule. */
static int parse_row(const jr_value *v, reg_row *r)
{
    pn_reg_row_info *i = &r->info;
    const jr_value *shares = jr_object_get(v, "shares");
    double instance = jr_as_number(jr_object_get(v, "instance"), -1);

    if (copy_text(i->key, sizeof(i->key), jr_as_string(jr_object_get(v, "key")), 14) != 0 ||
        !pn_reg_key_valid(i->key) ||
        pn_reg_kind_parse(jr_as_string(jr_object_get(v, "kind")), &i->kind) != 0 ||
        copy_text(i->owner, sizeof(i->owner), jr_as_string(jr_object_get(v, "owner")), 9) != 0 ||
        !prefixed_hex(i->owner, "p", 8) ||
        copy_text(i->origin_device, sizeof(i->origin_device),
                  jr_as_string(jr_object_get(v, "origin_device")), 17) != 0 ||
        !prefixed_hex(i->origin_device, "d", 16) ||
        copy_text(i->host_addr, sizeof(i->host_addr), jr_as_string(jr_object_get(v, "host_addr")),
                  PN_REG_ADDR_BYTES) != 0 || !pn_reg_addr_valid(i->host_addr) ||
        copy_text(i->title, sizeof(i->title), jr_as_string(jr_object_get(v, "title")),
                  PN_REG_TITLE_BYTES) != 0 ||
        copy_text(i->model, sizeof(i->model), jr_as_string(jr_object_get(v, "model")),
                  PN_REG_MODEL_BYTES) != 0 ||
        instance < 0 || instance > PN_REG_INSTANCE_MAX || instance != (int)instance ||
        pn_reg_state_parse(jr_as_string(jr_object_get(v, "state")), &i->state) != 0 ||
        jr_kind(shares) != JR_ARRAY || jr_array_len(shares) > PN_REG_MAX_SHARES ||
        (i->kind == PN_REG_KIND_DESKTOP && jr_array_len(shares) > 0))
        return -1;
    i->instance = (int)instance;
    for (int s = 0; s < jr_array_len(shares); s++) {
        const jr_value *sh = jr_array_get(shares, s);
        pn_reg_share_entry *out = &i->shares[i->n_shares++];
        if (copy_text(out->person, sizeof(out->person),
                      jr_as_string(jr_object_get(sh, "person")), 9) != 0 ||
            !prefixed_hex(out->person, "p", 8) || strcmp(out->person, i->owner) == 0 ||
            pn_reg_role_parse(jr_as_string(jr_object_get(sh, "role")), &out->role) != 0 ||
            (out->role != PN_REG_ROLE_VIEW && out->role != PN_REG_ROLE_CONTROL))
            return -1;
    }
    i->registered_at = (long long)jr_as_number(jr_object_get(v, "registered_at"), 0);
    i->last_beat = (long long)jr_as_number(jr_object_get(v, "last_beat"), 0);
    i->state = PN_REG_STATE_DISCONNECTED;   /* until its host registers again */
    return 0;
}

static int load_rows(pn_reg *reg, const char *text)
{
    jr_value *root = jr_parse(text);
    const jr_value *rows = jr_object_get(root, "rows");
    int rc = -1;

    if (!root || jr_as_number(jr_object_get(root, "version"), 0) != 1 ||
        jr_kind(rows) != JR_ARRAY || jr_array_len(rows) > PN_REG_MAX_ROWS)
        goto done;
    for (int i = 0; i < jr_array_len(rows); i++) {
        reg_row *r = calloc(1, sizeof(*r));
        if (!r || parse_row(jr_array_get(rows, i), r) != 0 || index_locked(reg, r->info.key) >= 0) {
            free(r);
            goto done;
        }
        reg->rows[reg->n++] = r;
    }
    rc = 0;
done:
    jr_free(root);
    return rc;
}

pn_reg *pn_reg_open(const char *path)
{
    pn_reg *reg = calloc(1, sizeof(*reg));
    char *text = NULL;

    if (!reg || !path ||
        snprintf(reg->path, sizeof(reg->path), "%s", path) >= (int)sizeof(reg->path)) {
        free(reg);
        return NULL;
    }
    pthread_mutex_init(&reg->lock, NULL);
    reg->disconnect_s = PN_REG_DISCONNECT_S;
    if (pn_local_read_file(path, 4u << 20, &text, NULL) == 0) {
        int bad = load_rows(reg, text);
        free(text);
        if (bad || chmod(path, 0600) != 0) {
            fprintf(stderr, "local: %s is malformed or not ours; refusing to start\n", path);
            pn_reg_close(reg);
            return NULL;
        }
    } else if (errno != ENOENT) {
        fprintf(stderr, "local: cannot read %s\n", path);
        pn_reg_close(reg);
        return NULL;
    }
    return reg;
}

static void free_row(reg_row *r)
{
    if (!r)
        return;
    crypto_wipe(r, sizeof(*r));
    free(r);
}

void pn_reg_close(pn_reg *reg)
{
    if (!reg)
        return;
    for (int i = 0; i < reg->n; i++)
        free_row(reg->rows[i]);
    pthread_mutex_destroy(&reg->lock);
    crypto_wipe(reg, sizeof(*reg));
    free(reg);
}

void pn_reg_set_disconnect_s(pn_reg *reg, int seconds)
{
    pthread_mutex_lock(&reg->lock);
    reg->disconnect_s = seconds > 0 ? seconds : PN_REG_DISCONNECT_S;
    pthread_mutex_unlock(&reg->lock);
}

static int owned_by_locked(const pn_reg *reg, const char *person)
{
    int n = 0;

    for (int i = 0; i < reg->n; i++)
        n += strcmp(reg->rows[i]->info.owner, person) == 0;
    return n;
}

/* Drop queued events the host answers by registering again. */
static void drop_reregister_locked(reg_row *r)
{
    int kept = 0;

    for (int i = 0; i < r->n_queue; i++)
        if (r->queue[i].type != PN_REG_EV_REREGISTER && r->queue[i].type != PN_REG_EV_RESYNC)
            r->queue[kept++] = r->queue[i];
    r->n_queue = kept;
}

/* The fields a registration sets; 1 when anything a listing shows changed. */
static int apply_host_locked(reg_row *r, const pn_reg_host *h)
{
    pn_reg_row_info *i = &r->info, before = r->info;

    i->kind = h->kind;
    snprintf(i->origin_device, sizeof(i->origin_device), "%s", h->origin_device);
    snprintf(i->host_addr, sizeof(i->host_addr), "%s", h->host_addr);
    snprintf(i->title, sizeof(i->title), "%s", h->title ? h->title : "");
    snprintf(i->model, sizeof(i->model), "%s", h->model ? h->model : "");
    i->instance = h->instance;
    i->state = h->state;
    snprintf(r->host_secret, sizeof(r->host_secret), "%s", h->host_secret);
    snprintf(r->host_pub, sizeof(r->host_pub), "%s", h->host_pub);
    return before.kind != i->kind || strcmp(before.origin_device, i->origin_device) != 0 ||
           strcmp(before.host_addr, i->host_addr) != 0 || strcmp(before.title, i->title) != 0 ||
           strcmp(before.model, i->model) != 0 || before.instance != i->instance ||
           before.state != i->state;
}

static int insert_locked(pn_reg *reg, const pn_reg_host *h, long long now)
{
    reg_row *r = NULL;
    reg_tombstone *t = tomb_locked(reg, h->key);

    if (reg->n >= PN_REG_MAX_ROWS || owned_by_locked(reg, h->owner) >= PN_REG_MAX_PER_PERSON)
        return PN_REG_E_FULL;
    r = calloc(1, sizeof(*r));
    if (!r)
        return PN_REG_E_IO;
    snprintf(r->info.key, sizeof(r->info.key), "%s", h->key);
    snprintf(r->info.owner, sizeof(r->info.owner), "%s", h->owner);
    r->info.registered_at = now;
    r->info.last_beat = now;
    if (t) {
        r->info.seq = t->seq;   /* numbers stay above what the host has seen */
        memset(t, 0, sizeof(*t));
    }
    apply_host_locked(r, h);
    reg->rows[reg->n++] = r;
    reg->rev++;
    return save_now_locked(reg, now);
}

int pn_reg_upsert(pn_reg *reg, const pn_reg_host *host, long long now, long long *seq_out)
{
    int rc = PN_REG_OK, idx = -1;
    reg_row *r = NULL;

    if (!reg || !host_valid(host))
        return PN_REG_E_INVALID;
    pthread_mutex_lock(&reg->lock);
    idx = index_locked(reg, host->key);
    if (idx < 0) {
        rc = insert_locked(reg, host, now);
        idx = index_locked(reg, host->key);
    } else if (strcmp(reg->rows[idx]->info.owner, host->owner) != 0) {
        rc = PN_REG_E_KEY_TAKEN;
    } else {
        r = reg->rows[idx];
        if (apply_host_locked(r, host))
            reg->rev++;
        r->info.last_beat = now;
        mark_dirty_locked(reg, now);
    }
    if (idx >= 0 && (rc == PN_REG_OK || rc == PN_REG_E_IO)) {
        r = reg->rows[idx];
        drop_reregister_locked(r);
        if (seq_out)
            *seq_out = r->n_queue ? r->queue[0].seq - 1 : r->info.seq;
    }
    pthread_mutex_unlock(&reg->lock);
    return rc;
}

int pn_reg_get(pn_reg *reg, const char *key, pn_reg_row_info *out)
{
    int idx = -1;

    pthread_mutex_lock(&reg->lock);
    idx = index_locked(reg, key);
    if (idx >= 0)
        *out = reg->rows[idx]->info;
    pthread_mutex_unlock(&reg->lock);
    return idx >= 0 ? PN_REG_OK : PN_REG_E_NOT_FOUND;
}

int pn_reg_attach_info(pn_reg *reg, const char *key, const char *person,
                       pn_reg_row_info *row_out, pn_reg_role *role_out,
                       char secret_out[PN_REG_HEX32_MAX], char pub_out[PN_REG_HEX32_MAX])
{
    int idx = -1;

    if (!person)
        return PN_REG_E_INVALID;
    pthread_mutex_lock(&reg->lock);
    idx = index_locked(reg, key);
    if (idx >= 0) {
        const reg_row *r = reg->rows[idx];
        *row_out = r->info;
        *role_out = role_locked(r, person);
        memcpy(secret_out, r->host_secret, PN_REG_HEX32_MAX);
        memcpy(pub_out, r->host_pub, PN_REG_HEX32_MAX);
    }
    pthread_mutex_unlock(&reg->lock);
    return idx >= 0 ? PN_REG_OK : PN_REG_E_NOT_FOUND;
}

/* Remove rows[idx], leaving a tombstone. Lock held. */
static void delete_at_locked(pn_reg *reg, int idx, pn_reg_close_reason reason, long long now)
{
    add_tomb_locked(reg, reg->rows[idx], reason, now);
    free_row(reg->rows[idx]);
    memmove(&reg->rows[idx], &reg->rows[idx + 1], (size_t)(reg->n - idx - 1) * sizeof(reg->rows[0]));
    reg->n--;
    reg->rows[reg->n] = NULL;
    reg->rev++;
}

int pn_reg_delete(pn_reg *reg, const char *key, const char *actor, long long now)
{
    int rc = PN_REG_OK, idx = -1;

    pthread_mutex_lock(&reg->lock);
    idx = index_locked(reg, key);
    if (idx < 0)
        rc = PN_REG_E_NOT_FOUND;
    else if (actor && strcmp(reg->rows[idx]->info.owner, actor) != 0)
        rc = PN_REG_E_NOT_OWNER;
    else {
        delete_at_locked(reg, idx, PN_REG_CLOSE_REMOVED, now);
        rc = save_now_locked(reg, now);
    }
    pthread_mutex_unlock(&reg->lock);
    return rc;
}

/* Queue an event on r (seq assigned); a full queue becomes one `resync`. */
static long long push_locked(reg_row *r, const pn_reg_event *ev)
{
    pn_reg_event e = *ev;

    if (r->n_queue >= PN_REG_QUEUE_MAX) {
        memset(&e, 0, sizeof(e));
        e.type = PN_REG_EV_RESYNC;
        r->n_queue = 0;
    }
    e.seq = ++r->info.seq;
    r->queue[r->n_queue++] = e;
    return e.seq;
}

static void push_type_locked(reg_row *r, pn_reg_event_type type)
{
    pn_reg_event e = { .type = type };
    push_locked(r, &e);
}

/* Set, change or remove one share of r. Lock held. */
static int set_share_locked(reg_row *r, const char *person, pn_reg_role role)
{
    pn_reg_row_info *i = &r->info;
    int at = -1;

    for (int s = 0; s < i->n_shares; s++)
        if (strcmp(i->shares[s].person, person) == 0)
            at = s;
    if (role == PN_REG_ROLE_NONE) {
        if (at < 0)
            return 0;
        memmove(&i->shares[at], &i->shares[at + 1], (size_t)(i->n_shares - at - 1) *
                sizeof(i->shares[0]));
        i->n_shares--;
        return 1;
    }
    if (at < 0 && i->n_shares >= PN_REG_MAX_SHARES)
        return PN_REG_E_SHARES_FULL;
    if (at < 0) {
        at = i->n_shares++;
        snprintf(i->shares[at].person, sizeof(i->shares[at].person), "%s", person);
    } else if (i->shares[at].role == role) {
        return 0;
    }
    i->shares[at].role = role;
    return 1;
}

int pn_reg_share(pn_reg *reg, const char *key, const char *actor, const char *person,
                 pn_reg_role role)
{
    int rc = PN_REG_OK, idx = -1;
    reg_row *r = NULL;

    if (!actor || !prefixed_hex(person, "p", 8) || role == PN_REG_ROLE_OWNER ||
        role < PN_REG_ROLE_NONE || role > PN_REG_ROLE_OWNER)
        return PN_REG_E_INVALID;
    pthread_mutex_lock(&reg->lock);
    idx = index_locked(reg, key);
    r = idx >= 0 ? reg->rows[idx] : NULL;
    if (!r)
        rc = PN_REG_E_NOT_FOUND;
    else if (strcmp(r->info.owner, actor) != 0)
        rc = PN_REG_E_NOT_OWNER;
    else if (r->info.kind == PN_REG_KIND_DESKTOP)
        rc = PN_REG_E_NOT_SHAREABLE;
    else if (strcmp(person, actor) == 0)
        rc = PN_REG_E_SELF_SHARE;
    else if ((rc = set_share_locked(r, person, role)) > 0) {
        push_type_locked(r, PN_REG_EV_SHARES);
        reg->rev++;
        rc = save_now_locked(reg, 0);
    }
    pthread_mutex_unlock(&reg->lock);
    return rc;
}

static pn_reg_role role_locked(const reg_row *r, const char *person)
{
    if (strcmp(r->info.owner, person) == 0)
        return PN_REG_ROLE_OWNER;
    for (int s = 0; s < r->info.n_shares; s++)
        if (strcmp(r->info.shares[s].person, person) == 0)
            return r->info.shares[s].role;
    return PN_REG_ROLE_NONE;
}

int pn_reg_list_for(pn_reg *reg, const char *person, pn_reg_row_info *out, int max)
{
    int n = 0;

    pthread_mutex_lock(&reg->lock);
    for (int pass = 0; pass < 2; pass++)
        for (int i = 0; i < reg->n && n < max; i++) {
            pn_reg_role role = role_locked(reg->rows[i], person);
            if ((pass == 0 && role == PN_REG_ROLE_OWNER) ||
                (pass == 1 && role != PN_REG_ROLE_OWNER && role != PN_REG_ROLE_NONE))
                out[n++] = reg->rows[i]->info;
        }
    pthread_mutex_unlock(&reg->lock);
    return n;
}

pn_reg_role pn_reg_role_of(pn_reg *reg, const char *key, const char *person)
{
    pn_reg_role role = PN_REG_ROLE_NONE;
    int idx = -1;

    pthread_mutex_lock(&reg->lock);
    idx = index_locked(reg, key);
    if (idx >= 0 && person)
        role = role_locked(reg->rows[idx], person);
    pthread_mutex_unlock(&reg->lock);
    return role;
}

static int has_type_locked(const reg_row *r, pn_reg_event_type type)
{
    for (int i = 0; i < r->n_queue; i++)
        if (r->queue[i].type == type)
            return 1;
    return 0;
}

int pn_reg_beat(pn_reg *reg, const char *key, const char *actor, long long since,
                long long now)
{
    int rc = PN_REG_OK, idx = -1, kept = 0;
    reg_tombstone *t = NULL;
    reg_row *r = NULL;

    pthread_mutex_lock(&reg->lock);
    idx = index_locked(reg, key);
    t = idx < 0 ? tomb_locked(reg, key) : NULL;
    if (idx < 0 && !t)
        rc = PN_REG_E_NOT_FOUND;
    else if (actor && strcmp(t ? t->owner : reg->rows[idx]->info.owner, actor) != 0)
        rc = PN_REG_E_NOT_OWNER;
    else if (t)
        rc = PN_REG_GONE;
    if (rc != PN_REG_OK)
        goto done;
    r = reg->rows[idx];
    r->info.last_beat = now;
    mark_dirty_locked(reg, now);
    if (since > r->info.seq)
        r->info.seq = since;
    for (int i = 0; i < r->n_queue; i++)
        if (r->queue[i].seq > since)
            r->queue[kept++] = r->queue[i];
    r->n_queue = kept;
    if (r->info.state == PN_REG_STATE_DISCONNECTED && !has_type_locked(r, PN_REG_EV_REREGISTER))
        push_type_locked(r, PN_REG_EV_REREGISTER);
done:
    pthread_mutex_unlock(&reg->lock);
    return rc;
}

int pn_reg_sweep(pn_reg *reg, long long now)
{
    int changed = 0;

    pthread_mutex_lock(&reg->lock);
    for (int i = 0; i < reg->n; i++) {
        reg_row *r = reg->rows[i];
        if (r->info.state == PN_REG_STATE_DISCONNECTED ||
            now - r->info.last_beat <= reg->disconnect_s)
            continue;
        r->info.state = PN_REG_STATE_DISCONNECTED;
        push_type_locked(r, PN_REG_EV_REREGISTER);
        changed++;
    }
    for (int i = 0; i < PN_REG_TOMBSTONES_MAX; i++)
        if (reg->tombs[i].key[0] && reg->tombs[i].expires <= now)
            memset(&reg->tombs[i], 0, sizeof(reg->tombs[i]));
    if (changed) {
        reg->rev++;
        mark_dirty_locked(reg, now);
    }
    pthread_mutex_unlock(&reg->lock);
    return changed;
}

long long pn_reg_push_event(pn_reg *reg, const char *key, const pn_reg_event *ev)
{
    long long seq = PN_REG_E_NOT_FOUND;
    int idx = -1;

    pthread_mutex_lock(&reg->lock);
    idx = index_locked(reg, key);
    if (idx >= 0)
        seq = push_locked(reg->rows[idx], ev);
    pthread_mutex_unlock(&reg->lock);
    return seq;
}

int pn_reg_pending(pn_reg *reg, const char *key, long long since)
{
    int rc = PN_REG_E_NOT_FOUND, idx = -1;

    pthread_mutex_lock(&reg->lock);
    idx = index_locked(reg, key);
    if (idx >= 0) {
        reg_row *r = reg->rows[idx];
        rc = r->n_queue > 0 && r->queue[r->n_queue - 1].seq > since;
    } else if (tomb_locked(reg, key)) {
        rc = 1;
    }
    pthread_mutex_unlock(&reg->lock);
    return rc;
}

int pn_reg_events_since(pn_reg *reg, const char *key, long long since, pn_reg_event *out,
                        int max, long long *seq_out, pn_reg_share_entry *shares_out,
                        int *n_shares_out)
{
    int n = PN_REG_E_NOT_FOUND, idx = -1;
    reg_tombstone *t = NULL;

    if (n_shares_out)
        *n_shares_out = 0;
    pthread_mutex_lock(&reg->lock);
    idx = index_locked(reg, key);
    if (idx >= 0) {
        const reg_row *r = reg->rows[idx];
        n = 0;
        for (int i = 0; i < r->n_queue && n < max; i++)
            if (r->queue[i].seq > since)
                out[n++] = r->queue[i];
        *seq_out = r->info.seq;
        if (shares_out && n_shares_out) {
            memcpy(shares_out, r->info.shares, (size_t)r->info.n_shares * sizeof(*shares_out));
            *n_shares_out = r->info.n_shares;
        }
    } else if ((t = tomb_locked(reg, key)) && max > 0) {
        memset(&out[0], 0, sizeof(out[0]));
        out[0].seq = t->seq;
        out[0].type = PN_REG_EV_CLOSE;
        out[0].reason = t->reason;
        *seq_out = t->seq;
        n = 1;
    }
    pthread_mutex_unlock(&reg->lock);
    return n;
}

/* Add `key` to keys_out unless it is there or the list is full. */
static void note_key(char (*keys_out)[PN_REG_KEY_MAX], int max, int *n, const char *key)
{
    for (int i = 0; i < *n; i++)
        if (strcmp(keys_out[i], key) == 0)
            return;
    if (*n < max)
        snprintf(keys_out[(*n)++], PN_REG_KEY_MAX, "%s", key);
}

int pn_reg_remove_person(pn_reg *reg, const char *person, long long now,
                         char (*keys_out)[PN_REG_KEY_MAX], int max, int *removed_out)
{
    int n_keys = 0, removed = 0, shares_changed = 0;

    pthread_mutex_lock(&reg->lock);
    for (int i = 0; i < reg->n;) {
        reg_row *r = reg->rows[i];
        if (strcmp(r->info.owner, person) == 0) {
            note_key(keys_out, max, &n_keys, r->info.key);
            delete_at_locked(reg, i, PN_REG_CLOSE_REMOVED, now);
            removed++;
            continue;
        }
        if (set_share_locked(r, person, PN_REG_ROLE_NONE) > 0) {
            push_type_locked(r, PN_REG_EV_SHARES);
            note_key(keys_out, max, &n_keys, r->info.key);
            shares_changed++;
        }
        i++;
    }
    if (removed || shares_changed) {
        reg->rev++;
        if (save_now_locked(reg, now) != PN_REG_OK)
            fprintf(stderr, "local: person %s left the registry; the save is retried\n", person);
    }
    pthread_mutex_unlock(&reg->lock);
    if (removed_out)
        *removed_out = removed;
    return n_keys;
}

int pn_reg_save(pn_reg *reg)
{
    int rc = 0;

    pthread_mutex_lock(&reg->lock);
    rc = save_locked(reg);
    pthread_mutex_unlock(&reg->lock);
    return rc;
}

int pn_reg_save_if_dirty(pn_reg *reg, long long now)
{
    int rc = 0;

    pthread_mutex_lock(&reg->lock);
    if (reg->dirty_since && now >= reg->dirty_since + PN_REG_SAVE_DELAY_S) {
        rc = save_locked(reg) == PN_REG_OK ? 1 : PN_REG_E_IO;
        if (rc < 0)
            reg->dirty_since = now;   /* retry after another delay, never in a spin */
    }
    pthread_mutex_unlock(&reg->lock);
    return rc;
}

long long pn_reg_save_due(pn_reg *reg)
{
    long long due = 0;

    pthread_mutex_lock(&reg->lock);
    if (reg->dirty_since)
        due = reg->dirty_since + PN_REG_SAVE_DELAY_S;
    pthread_mutex_unlock(&reg->lock);
    return due;
}

unsigned long long pn_reg_rev(pn_reg *reg)
{
    unsigned long long rev = 0;

    pthread_mutex_lock(&reg->lock);
    rev = reg->rev;
    pthread_mutex_unlock(&reg->lock);
    return rev;
}
