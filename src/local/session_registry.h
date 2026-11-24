/* SPDX-License-Identifier: AGPL-3.0-or-later */
/* Copyright (C) 2026 David (Corey) Herrera */
/**
 * @file   session_registry.h
 * @brief  The session registry of a box (people registry spec §3.4, §4.4): one
 *         metadata-only row per Argus host (or Desktop request endpoint), its
 *         shares, and a per-row event queue the host's held wait drains.
 *         Persisted in <state-dir>/sessions.json (0600, atomic); the host's
 *         secret and public key, the queues and the tombstones of deleted rows
 *         live in memory only. Every row reads `disconnected` after a load
 *         until its host registers again. No sockets: the hub (hosts_hub.h)
 *         holds the waits and calls in here.
 *
 *         Times are unix seconds passed in by the caller, so tests drive them.
 *
 * @status None
 * @issues None
 * @todo   None
 */

#ifndef PN_LOCAL_SESSION_REGISTRY_H
#define PN_LOCAL_SESSION_REGISTRY_H

#ifdef __cplusplus
extern "C" {
#endif

#define PN_REG_MAX_ROWS        512   /* per box */
#define PN_REG_MAX_PER_PERSON  128
#define PN_REG_MAX_SHARES      16    /* per row */
#define PN_REG_QUEUE_MAX       16    /* events per row; overflow -> one `resync` */
#define PN_REG_TOMBSTONES_MAX  64    /* deleted keys remembered for their host */
#define PN_REG_TOMBSTONE_S     120   /* > the 90 s a live host may stay silent */
#define PN_REG_DISCONNECT_S    90    /* no wait arrival for this long -> disconnected */
#define PN_REG_SAVE_DELAY_S    5     /* dirty (state / title / beat) save delay */
#define PN_REG_KEY_MAX         16    /* "c-" + 12 hex + NUL fits */
#define PN_REG_PERSON_MAX      16    /* "p" + 8 hex + NUL fits */
#define PN_REG_DEVICE_MAX      18    /* "d" + 16 hex + NUL */
#define PN_REG_ADDR_BYTES      64
#define PN_REG_TITLE_BYTES     256
#define PN_REG_MODEL_BYTES     64
#define PN_REG_INSTANCE_MAX    64
#define PN_REG_HEX32_MAX       65    /* 32 bytes as hex + NUL */
#define PN_REG_RELAY_ID_MAX    20    /* "r-" + 16 hex + NUL fits */

typedef enum { PN_REG_KIND_SESSION = 0, PN_REG_KIND_DESKTOP } pn_reg_kind;

typedef enum {
    PN_REG_STATE_THINKING = 0,
    PN_REG_STATE_NEEDS_YOU,
    PN_REG_STATE_IDLE,
    PN_REG_STATE_DISCONNECTED,   /* set by the box only */
} pn_reg_state;

typedef enum {
    PN_REG_ROLE_NONE = 0,
    PN_REG_ROLE_VIEW,
    PN_REG_ROLE_CONTROL,
    PN_REG_ROLE_OWNER,
} pn_reg_role;

typedef enum {
    PN_REG_EV_RELAY = 0,
    PN_REG_EV_SHARES,       /* the full share list after a change */
    PN_REG_EV_REREGISTER,   /* the row reads disconnected: re-POST */
    PN_REG_EV_RESYNC,       /* the queue overflowed: re-POST */
    PN_REG_EV_CLOSE,
} pn_reg_event_type;

typedef enum {
    PN_REG_CLOSE_SHUTDOWN = 0,
    PN_REG_CLOSE_REMOVED,
    PN_REG_CLOSE_REVOKED,
} pn_reg_close_reason;

/* Typed results (0 = success). */
typedef enum {
    PN_REG_OK              = 0,
    PN_REG_GONE            = 1,    /* pn_reg_beat: the key was deleted (a tombstone) */
    PN_REG_E_INVALID       = -1,   /* a field breaks the rules */
    PN_REG_E_NOT_FOUND     = -2,
    PN_REG_E_KEY_TAKEN     = -3,   /* the key belongs to another person */
    PN_REG_E_FULL          = -4,   /* 512 rows, or 128 for this person */
    PN_REG_E_NOT_OWNER     = -5,
    PN_REG_E_NOT_SHAREABLE = -6,   /* a desktop row */
    PN_REG_E_SHARES_FULL   = -7,
    PN_REG_E_SELF_SHARE    = -8,
    PN_REG_E_IO            = -9,   /* save failed: memory changed, the file stays dirty */
} pn_reg_result;

typedef struct {
    char        person[PN_REG_PERSON_MAX];
    pn_reg_role role;                       /* view or control */
} pn_reg_share_entry;

/* A registration as the route checked it (POST /local/hosts). */
typedef struct {
    const char  *key;
    pn_reg_kind  kind;
    const char  *owner;           /* the caller's person */
    const char  *origin_device;
    const char  *host_addr;       /* "a.b.c.d:port" or "[v6]:port" */
    const char  *title;
    const char  *model;
    int          instance;
    pn_reg_state state;           /* never DISCONNECTED */
    const char  *host_secret;     /* 64 hex, memory only */
    const char  *host_pub;        /* 64 hex, memory only */
} pn_reg_host;

/* One row as the box shows it (never the secret or the public key). */
typedef struct {
    char         key[PN_REG_KEY_MAX];
    pn_reg_kind  kind;
    char         owner[PN_REG_PERSON_MAX];
    char         origin_device[PN_REG_DEVICE_MAX];
    char         host_addr[PN_REG_ADDR_BYTES + 1];
    char         title[PN_REG_TITLE_BYTES + 1];
    char         model[PN_REG_MODEL_BYTES + 1];
    int          instance;
    pn_reg_state state;
    pn_reg_share_entry shares[PN_REG_MAX_SHARES];
    int          n_shares;
    long long    registered_at;
    long long    last_beat;
    long long    seq;             /* the last event number given out */
} pn_reg_row_info;

/* One queued event. */
typedef struct {
    long long           seq;
    pn_reg_event_type   type;
    char                relay_id[PN_REG_RELAY_ID_MAX];   /* RELAY */
    long long           relay_expires;                   /* RELAY: unix seconds */
    pn_reg_close_reason reason;                          /* CLOSE */
} pn_reg_event;

typedef struct pn_reg pn_reg;

/**
 * Open the registry at `path` (sessions.json); a missing file is an empty
 * registry. Every loaded row reads DISCONNECTED. NULL when the file is malformed
 * (the box refuses to start rather than empty it) or on OOM. Free with
 * pn_reg_close.
 * Thread-safe: no (call before serving). Reentrant: yes. Signal-safe: no
 */
pn_reg *pn_reg_open(const char *path);

/**
 * Free the registry, wiping the secrets. Does not save (call pn_reg_save first).
 * Thread-safe: no. Reentrant: yes. Signal-safe: no
 */
void pn_reg_close(pn_reg *reg);

/**
 * Seconds without a wait arrival before a row reads DISCONNECTED (default
 * PN_REG_DISCONNECT_S; tests shorten it).
 * Thread-safe: yes. Reentrant: yes. Signal-safe: no
 */
void pn_reg_set_disconnect_s(pn_reg *reg, int seconds);

/**
 * Register a host, or update the owner's existing row (title, address, model,
 * instance, state, kind, secret, public key; shares kept). A new row saves at
 * once; an update marks the file dirty. Queued `reregister` / `resync` events are
 * answered by this call and dropped. *seq_out (may be NULL) is the `since` the
 * host waits from next.
 * @return PN_REG_OK, E_INVALID, E_KEY_TAKEN, E_FULL, or E_IO (row kept, dirty)
 * Thread-safe: yes. Reentrant: yes. Signal-safe: no
 */
int pn_reg_upsert(pn_reg *reg, const pn_reg_host *host, long long now, long long *seq_out);

/** Copy one row. PN_REG_OK or E_NOT_FOUND. Thread-safe: yes. Reentrant: yes. Signal-safe: no */
int pn_reg_get(pn_reg *reg, const char *key, pn_reg_row_info *out);

/**
 * What an attach by `person` needs, read under one lock: the row, the person's
 * role on it (NONE when none), and the host's secret and public key (64 hex,
 * memory only; "" for a row loaded from disk until its host registers again).
 * The caller wipes secret_out after use.
 * @return PN_REG_OK, E_NOT_FOUND, or E_INVALID (no person)
 * Thread-safe: yes. Reentrant: yes. Signal-safe: no
 */
int pn_reg_attach_info(pn_reg *reg, const char *key, const char *person,
                       pn_reg_row_info *row_out, pn_reg_role *role_out,
                       char secret_out[PN_REG_HEX32_MAX], char pub_out[PN_REG_HEX32_MAX]);

/**
 * Delete a row. `actor` must own it (NULL = the box itself). The key becomes a
 * tombstone so its host's next wait gets `close` `removed`. Saves at once.
 * @return PN_REG_OK, E_NOT_FOUND, E_NOT_OWNER, or E_IO (deleted, dirty)
 * Thread-safe: yes. Reentrant: yes. Signal-safe: no
 */
int pn_reg_delete(pn_reg *reg, const char *key, const char *actor, long long now);

/**
 * Share a row with `person` as `role` (VIEW / CONTROL), or unshare (NONE).
 * `actor` must own it. Queues `shares` on the row and saves at once.
 * @return PN_REG_OK, E_NOT_FOUND, E_NOT_OWNER, E_NOT_SHAREABLE, E_SELF_SHARE,
 *         E_SHARES_FULL, E_INVALID, or E_IO (changed, dirty)
 * Thread-safe: yes. Reentrant: yes. Signal-safe: no
 */
int pn_reg_share(pn_reg *reg, const char *key, const char *actor, const char *person,
                 pn_reg_role role);

/**
 * The rows `person` sees: their own, then those shared with them. Returns the
 * number copied (at most max).
 * Thread-safe: yes. Reentrant: yes. Signal-safe: no
 */
int pn_reg_list_for(pn_reg *reg, const char *person, pn_reg_row_info *out, int max);

/**
 * `person`'s role on row `key`: OWNER, CONTROL, VIEW, or NONE (also for an
 * unknown key).
 * Thread-safe: yes. Reentrant: yes. Signal-safe: no
 */
pn_reg_role pn_reg_role_of(pn_reg *reg, const char *key, const char *person);

/**
 * A wait arrived for `key` from `actor` with `since`: sets last_beat (dirty, no
 * rev change), drops the events up to `since`, and on a DISCONNECTED row queues
 * `reregister`. A `since` above the row's counter (a box restart reset it)
 * raises the counter, so new events number above what the host has seen.
 * @return PN_REG_OK (a row), PN_REG_GONE (a tombstone: its close is pending),
 *         E_NOT_FOUND, or E_NOT_OWNER
 * Thread-safe: yes. Reentrant: yes. Signal-safe: no
 */
int pn_reg_beat(pn_reg *reg, const char *key, const char *actor, long long since,
                long long now);

/**
 * Mark rows silent for longer than the disconnect time DISCONNECTED (each gets
 * `reregister`) and expire old tombstones. Returns the rows newly disconnected.
 * Thread-safe: yes. Reentrant: yes. Signal-safe: no
 */
int pn_reg_sweep(pn_reg *reg, long long now);

/**
 * Queue `ev` (its seq is assigned) on row `key`. A full queue is replaced by one
 * `resync`. Returns the event's seq (the resync's on overflow), or E_NOT_FOUND.
 * Thread-safe: yes. Reentrant: yes. Signal-safe: no
 */
long long pn_reg_push_event(pn_reg *reg, const char *key, const pn_reg_event *ev);

/**
 * 1 when a wait on `key` with `since` has something to answer (an event above
 * `since`, or a tombstone), 0 when not, E_NOT_FOUND for an unknown key.
 * Thread-safe: yes. Reentrant: yes. Signal-safe: no
 */
int pn_reg_pending(pn_reg *reg, const char *key, long long since);

/**
 * Copy the events above `since` (at most max) with the row's counter
 * (*seq_out) and its current shares (for `shares` events; may be NULL). A
 * tombstone yields its one `close` event. Returns the number copied, or
 * E_NOT_FOUND.
 * Thread-safe: yes. Reentrant: yes. Signal-safe: no
 */
int pn_reg_events_since(pn_reg *reg, const char *key, long long since, pn_reg_event *out,
                        int max, long long *seq_out, pn_reg_share_entry *shares_out,
                        int *n_shares_out);

/**
 * The person is gone: delete their rows (tombstones, `close` `removed`) and drop
 * them from every share (those rows get `shares`). Saves at once. Writes the
 * keys whose waits must be answered into keys_out (at most max) and returns how
 * many; *removed_out (may be NULL) gets the rows deleted.
 * Thread-safe: yes. Reentrant: yes. Signal-safe: no
 */
int pn_reg_remove_person(pn_reg *reg, const char *person, long long now,
                         char (*keys_out)[PN_REG_KEY_MAX], int max, int *removed_out);

/** Write sessions.json now. 0 or E_IO. Thread-safe: yes. Reentrant: yes. Signal-safe: no */
int pn_reg_save(pn_reg *reg);

/**
 * Write sessions.json when dirty for at least PN_REG_SAVE_DELAY_S. 1 saved, 0
 * nothing to do yet, E_IO on failure (stays dirty).
 * Thread-safe: yes. Reentrant: yes. Signal-safe: no
 */
int pn_reg_save_if_dirty(pn_reg *reg, long long now);

/** When the dirty save is due (unix s), or 0 when clean. Thread-safe: yes. Reentrant: yes. Signal-safe: no */
long long pn_reg_save_due(pn_reg *reg);

/** The box-wide change counter (`rev`). Thread-safe: yes. Reentrant: yes. Signal-safe: no */
unsigned long long pn_reg_rev(pn_reg *reg);

/** Wire names. Thread-safe: yes. Reentrant: yes. Signal-safe: yes */
const char *pn_reg_state_name(pn_reg_state state);
const char *pn_reg_kind_name(pn_reg_kind kind);
const char *pn_reg_role_name(pn_reg_role role);
const char *pn_reg_event_type_name(pn_reg_event_type type);
const char *pn_reg_close_reason_name(pn_reg_close_reason reason);

/** Parse a wire name; 0, or -1 when unknown. Thread-safe: yes. Reentrant: yes. Signal-safe: yes */
int pn_reg_state_parse(const char *s, pn_reg_state *out);
int pn_reg_kind_parse(const char *s, pn_reg_kind *out);
int pn_reg_role_parse(const char *s, pn_reg_role *out);

/** 1 when s is a registry key: ^c-[0-9a-f]{12}$. Thread-safe: yes. Reentrant: yes. Signal-safe: yes */
int pn_reg_key_valid(const char *s);

/** 1 when s is "a.b.c.d:port" or "[v6]:port", at most 64 bytes. Thread-safe: yes. Reentrant: yes. Signal-safe: no */
int pn_reg_addr_valid(const char *s);

#ifdef __cplusplus
}
#endif

#endif /* PN_LOCAL_SESSION_REGISTRY_H */
