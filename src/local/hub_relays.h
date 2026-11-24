/* SPDX-License-Identifier: AGPL-3.0-or-later */
/* Copyright (C) 2026 David (Corey) Herrera */
/**
 * @file   hub_relays.h
 * @brief  The hub's relay pairs (people registry spec §4.6, §6.3): a client fd
 *         held until the host's leg arrives (10 s, else 504 `relay_timeout`),
 *         then both fds piped both ways by the hub thread. Stores nothing: the
 *         sockets are non-blocking, each direction has one 16 KiB buffer that
 *         exists only while the unsent rest of a partial write is pending, a
 *         side is read only when its outgoing buffer is empty (backpressure: a
 *         slow reader stalls the writer, nothing grows), and every buffer is
 *         wiped before it is freed. Relay sockets get SO_RCVBUF / SO_SNDBUF
 *         64 KiB and SO_KEEPALIVE. Either side closing closes both.
 *
 *         One relay per attach-ticket nonce: a nonce is remembered until the
 *         ticket's last accepted second. Caps: PN_HUB_RELAYS_MAX per box and
 *         PN_HUB_RELAYS_PER_PERSON per opening person (pending and piped both
 *         count; plan P6.3 sets the final values).
 *
 *         Registry-free: the hub (hosts_hub.c) passes a callback that queues
 *         the host's `relay` event. Only the hub thread fills the poll set and
 *         services it; route threads open and add host legs.
 *
 * @status None
 * @issues None
 * @todo   None
 */

#ifndef PN_LOCAL_HUB_RELAYS_H
#define PN_LOCAL_HUB_RELAYS_H

#include <poll.h>
#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

#define PN_HUB_RELAYS_MAX         32      /* per box, until plan P6.3 */
#define PN_HUB_RELAYS_PER_PERSON  8
#define PN_HUB_RELAY_HOST_MS      10000   /* the host's leg must arrive within this */
#define PN_HUB_RELAY_BUF          16384   /* one per direction, only while a write is partial */
#define PN_HUB_RELAY_SOCKBUF      65536   /* SO_RCVBUF / SO_SNDBUF on relay sockets */
#define PN_HUB_RELAY_NONCES_MAX   1024    /* live ticket nonces remembered */
#define PN_HUB_RELAY_ID_MAX       20      /* "r-" + 16 hex + NUL fits */
#define PN_HUB_RELAY_KEY_MAX      16
#define PN_HUB_RELAY_PERSON_MAX   16

typedef enum {
    PN_HUB_RELAY_OK          = 0,
    PN_HUB_RELAY_E_FULL      = -1,   /* the box or the person is at the cap */
    PN_HUB_RELAY_E_NONCE     = -2,   /* this ticket's nonce already opened a relay */
    PN_HUB_RELAY_E_NOT_FOUND = -3,   /* no pending relay with that id (or it expired) */
    PN_HUB_RELAY_E_NOT_OWNER = -4,   /* the host leg is not the row's owner */
    PN_HUB_RELAY_E_STOPPED   = -5,
    PN_HUB_RELAY_E_INVALID   = -6,   /* a bad argument, or the event could not be queued */
} pn_hub_relay_result;

/* Queue the host's `relay` event for relay `id` on row `key`; 0 or -1. Called
 * with the relay lock held, so it must not call back into the relays. */
typedef int (*pn_hub_relay_announce_fn)(const char *key, const char *id, void *user);

typedef struct {
    int max_relays;   /* 0 = PN_HUB_RELAYS_MAX */
    int per_person;   /* 0 = PN_HUB_RELAYS_PER_PERSON */
    int host_ms;      /* 0 = PN_HUB_RELAY_HOST_MS */
} pn_hub_relays_cfg;

/* What a relay open needs, checked by the route (ticket, caller). */
typedef struct {
    const char *key;            /* the row */
    const char *client_person;  /* the opener (counts toward the per-person cap) */
    const char *owner;          /* the row's owner: the only person who may add the host leg */
    const char *nonce;          /* the ticket's nonce */
    long long   nonce_until;    /* the ticket's last accepted second (exp + skew) */
} pn_hub_relay_open_args;

typedef struct pn_hub_relays pn_hub_relays;

/**
 * A relay table. `cfg` may be NULL (defaults). NULL on OOM. Free with
 * pn_hub_relays_free.
 * Thread-safe: yes. Reentrant: yes. Signal-safe: no
 */
pn_hub_relays *pn_hub_relays_new(const pn_hub_relays_cfg *cfg);

/** Close every relay and free the table. Thread-safe: no. Reentrant: yes. Signal-safe: no */
void pn_hub_relays_free(pn_hub_relays *relays);

/**
 * Hold client `fd` as a pending relay: checks the caps and the nonce, makes
 * `r-<16 hex>` into id_out, and calls `announce` (under the relay lock, so the
 * host cannot ask for the id before it exists). On PN_HUB_RELAY_OK the table
 * owns the fd; on any error the fd stays the caller's to answer and close. The
 * nonce is marked only when the relay is held.
 * Thread-safe: yes. Reentrant: yes. Signal-safe: no
 */
int pn_hub_relays_open(pn_hub_relays *relays, int fd, const pn_hub_relay_open_args *args,
                       pn_hub_relay_announce_fn announce, void *user,
                       char id_out[PN_HUB_RELAY_ID_MAX]);

/**
 * Add host leg `fd` from `person` to pending relay `id`: both legs get the 200
 * header block (spec §4.6), then the bytes are piped. On PN_HUB_RELAY_OK the
 * table owns the fd; on an error (E_NOT_FOUND, E_NOT_OWNER, E_STOPPED) it stays
 * the caller's.
 * Thread-safe: yes. Reentrant: yes. Signal-safe: no
 */
int pn_hub_relays_host(pn_hub_relays *relays, int fd, const char *id, const char *person);

/**
 * Hub thread: append the relay fds to pfds (at most cap entries) and lower
 * *nearest_ms to the nearest host-leg deadline. Returns the entries written.
 * Thread-safe: hub thread only. Reentrant: no. Signal-safe: no
 */
int pn_hub_relays_poll_fill(pn_hub_relays *relays, struct pollfd *pfds, int cap,
                            uint64_t *nearest_ms);

/**
 * Hub thread: after poll(), pipe what is ready, close pairs whose side closed
 * or failed, drop a pending client that sent early or left, and answer 504
 * `relay_timeout` to pending clients past their deadline. `pfds`/`n` are what
 * poll_fill wrote (revents filled by poll).
 * Thread-safe: hub thread only. Reentrant: no. Signal-safe: no
 */
void pn_hub_relays_service(pn_hub_relays *relays, const struct pollfd *pfds, int n,
                           uint64_t now_ms);

/**
 * Close every relay (pending clients get no answer) and refuse new ones.
 * Returns how many were closed.
 * Thread-safe: yes. Reentrant: no. Signal-safe: no
 */
int pn_hub_relays_close_all(pn_hub_relays *relays);

/** Relays pending or piped now. Thread-safe: yes. Reentrant: yes. Signal-safe: no */
int pn_hub_relays_count(pn_hub_relays *relays);

/** Most bytes ever held in relay buffers at once (tests: no growth). Thread-safe: yes. Reentrant: yes. Signal-safe: no */
size_t pn_hub_relays_buffered_peak(pn_hub_relays *relays);

/** Monotonic milliseconds, the clock of the deadlines. Thread-safe: yes. Reentrant: yes. Signal-safe: yes */
uint64_t pn_hub_relays_now_ms(void);

#ifdef __cplusplus
}
#endif

#endif /* PN_LOCAL_HUB_RELAYS_H */
