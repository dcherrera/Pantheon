/* SPDX-License-Identifier: AGPL-3.0-or-later */
/* Copyright (C) 2026 David (Corey) Herrera */
/**
 * @file   hub_relays.c
 * @brief  Implementation of the hub's relay pairs (see hub_relays.h). No byte
 *         that passes through a relay is ever logged or written anywhere but
 *         the other socket; logs name relay ids and counts only.
 *
 *         Lock order: the relay lock, then (inside `announce`) the registry
 *         lock. Nothing takes the relay lock while holding the registry's.
 *
 * @status None
 * @issues None
 * @todo   None
 */

#include "hub_relays.h"

#include <errno.h>
#include <fcntl.h>
#include <pthread.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <unistd.h>

#include <sys/socket.h>
#include <sys/time.h>

#include "../../lib/monocypher/monocypher.h"
#include "../serve/http_server.h"
#include "local_random.h"
#include "sha256.h"
#include "used_nonces.h"

#define RELAY_SEND_TIMEOUT_MS 2000   /* the 504 to a pending client (blocking, small) */
#define RELAY_READS_PER_WAKE  4      /* per direction: 64 KiB, then let others run */

typedef enum { SLOT_FREE = 0, SLOT_PENDING, SLOT_PIPED } slot_state;

/* The unsent rest of a partial write in one direction; buf is NULL when empty. */
typedef struct {
    uint8_t *buf;
    size_t   len;
    size_t   off;
} relay_backlog;

typedef struct {
    slot_state    state;
    unsigned      gen;
    int           client_fd;
    int           host_fd;
    char          id[PN_HUB_RELAY_ID_MAX];
    char          key[PN_HUB_RELAY_KEY_MAX];
    char          client_person[PN_HUB_RELAY_PERSON_MAX];
    char          owner[PN_HUB_RELAY_PERSON_MAX];
    uint64_t      deadline_ms;    /* PENDING: the host leg's */
    relay_backlog to_host;        /* read from the client, not yet written to the host */
    relay_backlog to_client;
    int           ready;          /* service scratch: an fd of this pair had events */
} relay_slot;

/* One poll entry's slot, written by poll_fill and read by service. */
typedef struct {
    int      slot;
    unsigned gen;
} relay_poll_ref;

struct pn_hub_relays {
    pn_hub_relays_cfg cfg;
    pthread_mutex_t   lock;
    relay_slot       *slots;
    relay_poll_ref   *refs;          /* 2 per slot */
    int              *timed_out;     /* service scratch: client fds owed a 504 */
    pn_used_nonces   *nonces;
    int               stopped;
    unsigned          gen_next;
    size_t            buffered;      /* bytes in backlog buffers now */
    size_t            buffered_peak;
};

uint64_t pn_hub_relays_now_ms(void)
{
    struct timespec ts;

    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (uint64_t)ts.tv_sec * 1000u + (uint64_t)ts.tv_nsec / 1000000u;
}

static int set_nonblock(int fd)
{
    int fl = fcntl(fd, F_GETFL, 0);
    return fl < 0 ? -1 : fcntl(fd, F_SETFL, fl | O_NONBLOCK);
}

/* Kernel buffering capped, dead idle peers reaped. AF_UNIX test sockets have no
 * keepalive; that refusal is expected and not logged. */
static void tune_relay_socket(int fd)
{
    int size = PN_HUB_RELAY_SOCKBUF, one = 1;

    if (setsockopt(fd, SOL_SOCKET, SO_RCVBUF, &size, sizeof(size)) != 0 ||
        setsockopt(fd, SOL_SOCKET, SO_SNDBUF, &size, sizeof(size)) != 0)
        fprintf(stderr, "local: relay socket buffers not capped: %s\n", strerror(errno));
    if (setsockopt(fd, SOL_SOCKET, SO_KEEPALIVE, &one, sizeof(one)) != 0 &&
        errno != ENOPROTOOPT && errno != EOPNOTSUPP && errno != EINVAL)
        fprintf(stderr, "local: relay keepalive not set: %s\n", strerror(errno));
}

static void backlog_clear(pn_hub_relays *r, relay_backlog *b)
{
    if (!b->buf)
        return;
    crypto_wipe(b->buf, PN_HUB_RELAY_BUF);
    free(b->buf);
    b->buf = NULL;
    b->len = b->off = 0;
    r->buffered -= PN_HUB_RELAY_BUF;
}

/* Keep bytes[0..n) as the backlog; -1 on OOM. */
static int backlog_fill(pn_hub_relays *r, relay_backlog *b, const uint8_t *bytes, size_t n)
{
    if (n > PN_HUB_RELAY_BUF || !(b->buf = malloc(PN_HUB_RELAY_BUF)))
        return -1;
    memcpy(b->buf, bytes, n);
    b->len = n;
    b->off = 0;
    r->buffered += PN_HUB_RELAY_BUF;
    if (r->buffered > r->buffered_peak)
        r->buffered_peak = r->buffered;
    return 0;
}

/* Close both legs and wipe the slot. Lock held. */
static void slot_release(pn_hub_relays *r, relay_slot *s)
{
    if (s->client_fd >= 0)
        hs_close(s->client_fd);
    if (s->host_fd >= 0)
        hs_close(s->host_fd);
    backlog_clear(r, &s->to_host);
    backlog_clear(r, &s->to_client);
    crypto_wipe(s, sizeof(*s));
    s->state = SLOT_FREE;
    s->client_fd = s->host_fd = -1;
}

pn_hub_relays *pn_hub_relays_new(const pn_hub_relays_cfg *cfg)
{
    pn_hub_relays *r = calloc(1, sizeof(*r));

    if (!r)
        return NULL;
    if (cfg)
        r->cfg = *cfg;
    r->cfg.max_relays = r->cfg.max_relays > 0 ? r->cfg.max_relays : PN_HUB_RELAYS_MAX;
    r->cfg.per_person = r->cfg.per_person > 0 ? r->cfg.per_person : PN_HUB_RELAYS_PER_PERSON;
    r->cfg.host_ms = r->cfg.host_ms > 0 ? r->cfg.host_ms : PN_HUB_RELAY_HOST_MS;
    pthread_mutex_init(&r->lock, NULL);
    r->slots = calloc((size_t)r->cfg.max_relays, sizeof(*r->slots));
    r->refs = calloc((size_t)r->cfg.max_relays * 2, sizeof(*r->refs));
    r->timed_out = calloc((size_t)r->cfg.max_relays, sizeof(*r->timed_out));
    r->nonces = pn_used_nonces_new(PN_HUB_RELAY_NONCES_MAX);
    if (!r->slots || !r->refs || !r->timed_out || !r->nonces) {
        pn_hub_relays_free(r);
        return NULL;
    }
    for (int i = 0; i < r->cfg.max_relays; i++)
        r->slots[i].client_fd = r->slots[i].host_fd = -1;
    return r;
}

void pn_hub_relays_free(pn_hub_relays *r)
{
    if (!r)
        return;
    if (r->slots)
        pn_hub_relays_close_all(r);
    pthread_mutex_destroy(&r->lock);
    free(r->slots);
    free(r->refs);
    free(r->timed_out);
    pn_used_nonces_free(r->nonces);
    free(r);
}

/* A free slot when the box and the person are under their caps; -1 when full. Lock held. */
static int free_slot_locked(const pn_hub_relays *r, const char *person)
{
    int slot = -1, mine = 0;

    for (int i = 0; i < r->cfg.max_relays; i++) {
        const relay_slot *s = &r->slots[i];
        if (s->state == SLOT_FREE) {
            if (slot < 0)
                slot = i;
        } else if (strcmp(s->client_person, person) == 0) {
            mine++;
        }
    }
    return mine >= r->cfg.per_person ? -1 : slot;
}

static int make_relay_id(char id[PN_HUB_RELAY_ID_MAX])
{
    uint8_t raw[8];

    if (pn_local_random_bytes(raw, sizeof(raw)) != 0)
        return -1;
    memcpy(id, "r-", 2);
    pn_hex_encode(raw, sizeof(raw), id + 2);
    return 0;
}

static int open_args_valid(int fd, const pn_hub_relay_open_args *a)
{
    return fd >= 0 && a && a->key && a->client_person && a->owner && a->nonce &&
           strlen(a->key) < PN_HUB_RELAY_KEY_MAX &&
           strlen(a->client_person) < PN_HUB_RELAY_PERSON_MAX &&
           strlen(a->owner) < PN_HUB_RELAY_PERSON_MAX;
}

int pn_hub_relays_open(pn_hub_relays *r, int fd, const pn_hub_relay_open_args *a,
                       pn_hub_relay_announce_fn announce, void *user,
                       char id_out[PN_HUB_RELAY_ID_MAX])
{
    struct timeval tv = { RELAY_SEND_TIMEOUT_MS / 1000, (RELAY_SEND_TIMEOUT_MS % 1000) * 1000 };
    long long now = (long long)time(NULL);
    relay_slot *s = NULL;
    int slot = -1, rc = PN_HUB_RELAY_OK;

    if (!r || !open_args_valid(fd, a) || !announce)
        return PN_HUB_RELAY_E_INVALID;
    if (setsockopt(fd, SOL_SOCKET, SO_SNDTIMEO, &tv, sizeof(tv)) != 0)
        fprintf(stderr, "local: relay client send timeout not set: %s\n", strerror(errno));
    pthread_mutex_lock(&r->lock);
    if (r->stopped) {
        rc = PN_HUB_RELAY_E_STOPPED;
    } else if (pn_used_nonces_seen(r->nonces, a->nonce, now)) {
        rc = PN_HUB_RELAY_E_NONCE;
    } else if ((slot = free_slot_locked(r, a->client_person)) < 0 ||
               pn_used_nonces_live(r->nonces, now) >= PN_HUB_RELAY_NONCES_MAX) {
        rc = PN_HUB_RELAY_E_FULL;
    } else if (make_relay_id(id_out) != 0 || announce(a->key, id_out, user) != 0) {
        rc = PN_HUB_RELAY_E_INVALID;
    } else if (pn_used_nonces_mark(r->nonces, a->nonce, a->nonce_until, now) != 0) {
        rc = PN_HUB_RELAY_E_INVALID;   /* not reached: unseen, and room was checked */
    }
    if (rc == PN_HUB_RELAY_OK) {
        s = &r->slots[slot];
        s->state = SLOT_PENDING;
        s->gen = ++r->gen_next;
        s->client_fd = fd;
        s->host_fd = -1;
        memcpy(s->id, id_out, PN_HUB_RELAY_ID_MAX);
        snprintf(s->key, sizeof(s->key), "%s", a->key);
        snprintf(s->client_person, sizeof(s->client_person), "%s", a->client_person);
        snprintf(s->owner, sizeof(s->owner), "%s", a->owner);
        s->deadline_ms = pn_hub_relays_now_ms() + (uint64_t)r->cfg.host_ms;
    }
    pthread_mutex_unlock(&r->lock);
    return rc;
}

/* The 200 header block both legs read before any relayed byte (spec §4.6). */
static int header_block(const char *id, char *out, size_t cap)
{
    int n = snprintf(out, cap, "HTTP/1.1 200 OK\r\nContent-Type: application/octet-stream\r\n"
                     "X-Pantheon-Relay: %s\r\nConnection: close\r\n\r\n", id);
    return n > 0 && (size_t)n < cap ? n : -1;
}

int pn_hub_relays_host(pn_hub_relays *r, int fd, const char *id, const char *person)
{
    char head[256];
    int rc = PN_HUB_RELAY_E_NOT_FOUND, n = 0;

    if (!r || fd < 0 || !id || !person)
        return PN_HUB_RELAY_E_INVALID;
    pthread_mutex_lock(&r->lock);
    for (int i = 0; !r->stopped && i < r->cfg.max_relays; i++) {
        relay_slot *s = &r->slots[i];
        if (s->state != SLOT_PENDING || strcmp(s->id, id) != 0)
            continue;
        if (strcmp(s->owner, person) != 0) {
            rc = PN_HUB_RELAY_E_NOT_OWNER;
            break;
        }
        n = header_block(s->id, head, sizeof(head));
        if (n < 0 || set_nonblock(fd) != 0 || set_nonblock(s->client_fd) != 0 ||
            backlog_fill(r, &s->to_host, (const uint8_t *)head, (size_t)n) != 0 ||
            backlog_fill(r, &s->to_client, (const uint8_t *)head, (size_t)n) != 0) {
            backlog_clear(r, &s->to_host);
            rc = PN_HUB_RELAY_E_INVALID;
            break;
        }
        tune_relay_socket(fd);
        tune_relay_socket(s->client_fd);
        s->host_fd = fd;
        s->state = SLOT_PIPED;
        s->gen = ++r->gen_next;
        rc = PN_HUB_RELAY_OK;
        break;
    }
    if (r->stopped)
        rc = PN_HUB_RELAY_E_STOPPED;
    pthread_mutex_unlock(&r->lock);
    return rc;
}

/* Write the backlog; 0 drained, 1 still pending, -1 the socket failed. */
static int backlog_flush(pn_hub_relays *r, int fd, relay_backlog *b)
{
    while (b->buf && b->off < b->len) {
        ssize_t w = send(fd, b->buf + b->off, b->len - b->off, 0);
        if (w < 0 && errno == EINTR)
            continue;
        if (w < 0)
            return errno == EAGAIN || errno == EWOULDBLOCK ? 1 : -1;
        b->off += (size_t)w;
    }
    backlog_clear(r, b);
    return 0;
}

/* Move bytes src -> dst through backlog b (the dst direction's). Reads src only
 * when b is empty. 0 when src has nothing more now, 1 when dst is full (b
 * holds the rest), -1 when src closed or either socket failed. */
static int relay_pump(pn_hub_relays *r, int src, int dst, relay_backlog *b)
{
    uint8_t chunk[PN_HUB_RELAY_BUF];
    int rc = 0;

    for (int round = 0; round < RELAY_READS_PER_WAKE; round++) {
        ssize_t n = 0, w = 0;
        size_t sent = 0;
        if ((rc = backlog_flush(r, dst, b)) != 0)
            return rc;
        n = recv(src, chunk, sizeof(chunk), 0);
        if (n < 0 && errno == EINTR)
            continue;
        if (n <= 0)
            return n < 0 && (errno == EAGAIN || errno == EWOULDBLOCK) ? 0 : -1;
        do {
            w = send(dst, chunk, (size_t)n, 0);
        } while (w < 0 && errno == EINTR);
        sent = w > 0 ? (size_t)w : 0;
        if (w < 0 && errno != EAGAIN && errno != EWOULDBLOCK)
            rc = -1;
        else if (sent < (size_t)n)
            rc = backlog_fill(r, b, chunk + sent, (size_t)n - sent) == 0 ? 1 : -1;
        crypto_wipe(chunk, (size_t)n);
        if (rc != 0)
            return rc;
    }
    return 0;   /* more may be waiting: poll() reports it again */
}

int pn_hub_relays_poll_fill(pn_hub_relays *r, struct pollfd *pfds, int cap, uint64_t *nearest_ms)
{
    int n = 0;

    pthread_mutex_lock(&r->lock);
    for (int i = 0; i < r->cfg.max_relays && n + 2 <= cap; i++) {
        relay_slot *s = &r->slots[i];
        if (s->state == SLOT_FREE)
            continue;
        pfds[n].fd = s->client_fd;
        pfds[n].revents = 0;
        r->refs[n].slot = i;
        r->refs[n].gen = s->gen;
        if (s->state == SLOT_PENDING) {
            pfds[n++].events = POLLIN;   /* readable before the header = left or broke */
            if (s->deadline_ms < *nearest_ms)
                *nearest_ms = s->deadline_ms;
            continue;
        }
        pfds[n++].events = (short)((s->to_host.buf ? 0 : POLLIN) |
                                   (s->to_client.buf ? POLLOUT : 0));
        pfds[n].fd = s->host_fd;
        pfds[n].revents = 0;
        pfds[n].events = (short)((s->to_client.buf ? 0 : POLLIN) |
                                 (s->to_host.buf ? POLLOUT : 0));
        r->refs[n].slot = i;
        r->refs[n++].gen = s->gen;
    }
    pthread_mutex_unlock(&r->lock);
    return n;
}

/* Mark the pairs that had events; drop pending clients that did. Lock held. */
static void mark_ready_locked(pn_hub_relays *r, const struct pollfd *pfds, int n)
{
    for (int k = 0; k < n; k++) {
        relay_slot *s = &r->slots[r->refs[k].slot];
        if (!pfds[k].revents || s->gen != r->refs[k].gen || s->state == SLOT_FREE)
            continue;
        if (s->state == SLOT_PENDING)
            slot_release(r, s);
        else
            s->ready = 1;
    }
}

/* Pipe one ready pair both ways; close both legs when either side is done. Lock held. */
static void pump_pair_locked(pn_hub_relays *r, relay_slot *s)
{
    int up = relay_pump(r, s->client_fd, s->host_fd, &s->to_host);
    int down = up < 0 ? -1 : relay_pump(r, s->host_fd, s->client_fd, &s->to_client);

    s->ready = 0;
    if (up < 0 || down < 0)
        slot_release(r, s);
}

void pn_hub_relays_service(pn_hub_relays *r, const struct pollfd *pfds, int n, uint64_t now_ms)
{
    int n_timed_out = 0;

    pthread_mutex_lock(&r->lock);
    mark_ready_locked(r, pfds, n);
    for (int i = 0; i < r->cfg.max_relays; i++) {
        relay_slot *s = &r->slots[i];
        if (s->state == SLOT_PIPED && s->ready) {
            pump_pair_locked(r, s);
        } else if (s->state == SLOT_PENDING && s->deadline_ms <= now_ms) {
            r->timed_out[n_timed_out++] = s->client_fd;
            s->client_fd = -1;
            slot_release(r, s);
        }
    }
    pthread_mutex_unlock(&r->lock);
    for (int i = 0; i < n_timed_out; i++) {
        hs_send_error(r->timed_out[i], 504, "relay_timeout", "the host did not join the relay "
                      "in time; try again");
        hs_close(r->timed_out[i]);
    }
}

int pn_hub_relays_close_all(pn_hub_relays *r)
{
    int n = 0;

    pthread_mutex_lock(&r->lock);
    r->stopped = 1;
    for (int i = 0; i < r->cfg.max_relays; i++)
        if (r->slots[i].state != SLOT_FREE) {
            slot_release(r, &r->slots[i]);
            n++;
        }
    pthread_mutex_unlock(&r->lock);
    return n;
}

int pn_hub_relays_count(pn_hub_relays *r)
{
    int n = 0;

    pthread_mutex_lock(&r->lock);
    for (int i = 0; i < r->cfg.max_relays; i++)
        n += r->slots[i].state != SLOT_FREE;
    pthread_mutex_unlock(&r->lock);
    return n;
}

size_t pn_hub_relays_buffered_peak(pn_hub_relays *r)
{
    size_t n = 0;

    pthread_mutex_lock(&r->lock);
    n = r->buffered_peak;
    pthread_mutex_unlock(&r->lock);
    return n;
}
