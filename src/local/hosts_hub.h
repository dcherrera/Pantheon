/* SPDX-License-Identifier: AGPL-3.0-or-later */
/* Copyright (C) 2026 David (Corey) Herrera */
/**
 * @file   hosts_hub.h
 * @brief  The hub (people registry spec §6.3): one background thread holds
 *         every host's `GET /local/hosts/<key>/wait` in a single poll() loop with
 *         a self-pipe, instead of a parked thread per wait (the host is 2-core).
 *         A wait is answered when its row has events above its `since`, when a
 *         second wait for the same key replaces it, at its deadline (30 s), or at
 *         stop (`close` `shutdown`); a readable wait fd means the peer left. The
 *         same thread runs the registry's 90 s sweep and its delayed save. The
 *         poll timeout is the nearest deadline; there is no fast timer.
 *         The same poll() also pipes the relay pairs (hub_relays.h): the hub
 *         queues the host's `relay` event and wakes for the host leg's deadline.
 *
 * @status None
 * @issues None
 * @todo   None
 */

#ifndef PN_LOCAL_HOSTS_HUB_H
#define PN_LOCAL_HOSTS_HUB_H

#include <stddef.h>

#include "hub_relays.h"
#include "session_registry.h"

#ifdef __cplusplus
extern "C" {
#endif

#define PN_HUB_WAITS_MAX        512     /* = PN_REG_MAX_ROWS */
#define PN_HUB_WAIT_MS          30000   /* a held wait lasts at most this */
#define PN_HUB_SWEEP_MS         5000
#define PN_HUB_SEND_TIMEOUT_MS  2000    /* SO_SNDTIMEO on a wait fd (answers < 4 KiB) */
#define PN_HUB_KEY_ID_MAX       16

/* A person's handle for a `shares` event; 0, or -1 when unknown (sent as ""). */
typedef int (*pn_hub_handle_fn)(const char *person, char *handle_out, size_t cap, void *user);

typedef struct {
    int              wait_ms;     /* 0 = PN_HUB_WAIT_MS */
    int              sweep_ms;    /* 0 = PN_HUB_SWEEP_MS */
    int              max_waits;   /* 0 = PN_HUB_WAITS_MAX */
    pn_hub_handle_fn handle_of;   /* may be NULL */
    void            *user;
    pn_hub_relays_cfg relays;     /* zeros = PN_HUB_RELAYS_MAX, _PER_PERSON, _HOST_MS */
} pn_hub_cfg;

typedef struct pn_hub pn_hub;

/**
 * Start the hub thread over `reg` (borrowed; it must outlive the hub). `cfg`
 * may be NULL (defaults). NULL on failure. Stop with pn_hub_stop, then free
 * with pn_hub_free.
 * Thread-safe: yes. Reentrant: yes. Signal-safe: no
 */
pn_hub *pn_hub_start(pn_reg *reg, const pn_hub_cfg *cfg);

/**
 * Hold `fd` (a validated `GET …/wait` from device key `key_id`) for row `key`.
 * On success the hub owns the fd and answers and closes it; a wait already held
 * for `key` is answered `"replaced":true` first. -1 when the hub is full or
 * stopped (the fd stays the caller's: answer 503 and close it).
 * Thread-safe: yes. Reentrant: yes. Signal-safe: no
 */
int pn_hub_add_wait(pn_hub *hub, int fd, const char *key, const char *key_id, long long since);

/**
 * Something was queued on row `key` (or it was deleted): its held wait, if
 * any, is answered when it has events above its `since`.
 * Thread-safe: yes. Reentrant: yes. Signal-safe: no
 */
void pn_hub_notify(pn_hub *hub, const char *key);

/**
 * Answer every wait held with device key `key_id` with `close` `reason` (a
 * revoked key). Returns how many.
 * Thread-safe: yes. Reentrant: yes. Signal-safe: no
 */
int pn_hub_close_device(pn_hub *hub, const char *key_id, pn_reg_close_reason reason);

/**
 * Stop: join the thread, answer every held wait with `close` `shutdown`, and
 * refuse new waits. The registry is not saved here (the caller saves it).
 * Thread-safe: yes. Reentrant: no. Signal-safe: no
 */
void pn_hub_stop(pn_hub *hub);

/**
 * Free a stopped hub. Kept apart from pn_hub_stop so a connection thread still
 * finishing a request at shutdown meets a refusing hub, never freed memory.
 * Thread-safe: no. Reentrant: no. Signal-safe: no
 */
void pn_hub_free(pn_hub *hub);

/**
 * Hold client `fd` (a validated `POST /local/relay/<key>`) as a pending relay:
 * queues `relay {id, expires_in}` on the row and answers the host's held wait.
 * On PN_HUB_RELAY_OK the hub owns the fd (route returns 2); otherwise it stays
 * the caller's (E_FULL -> 503, E_NONCE, E_INVALID when the row is gone).
 * Thread-safe: yes. Reentrant: yes. Signal-safe: no
 */
int pn_hub_relay_open(pn_hub *hub, int fd, const pn_hub_relay_open_args *args,
                      char id_out[PN_HUB_RELAY_ID_MAX]);

/**
 * Add the host's leg `fd` (`POST /local/relay/<id>/host` from `person`): both
 * legs get the 200 header block, then the hub pipes them. On PN_HUB_RELAY_OK
 * the hub owns the fd; else it stays the caller's (E_NOT_FOUND, E_NOT_OWNER).
 * Thread-safe: yes. Reentrant: yes. Signal-safe: no
 */
int pn_hub_relay_host(pn_hub *hub, int fd, const char *id, const char *person);

/** Relays pending or piped now. Thread-safe: yes. Reentrant: yes. Signal-safe: no */
int pn_hub_relays_held(pn_hub *hub);

/** Most relay buffer bytes ever held at once (tests). Thread-safe: yes. Reentrant: yes. Signal-safe: no */
size_t pn_hub_relay_buffered_peak(pn_hub *hub);

/** Waits held now. Thread-safe: yes. Reentrant: yes. Signal-safe: no */
int pn_hub_waits_held(pn_hub *hub);

/** How often the hub thread has woken from poll() (tests). Thread-safe: yes. Reentrant: yes. Signal-safe: no */
unsigned long pn_hub_wakeups(pn_hub *hub);

#ifdef __cplusplus
}
#endif

#endif /* PN_LOCAL_HOSTS_HUB_H */
