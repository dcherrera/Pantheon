/* SPDX-License-Identifier: AGPL-3.0-or-later */
/* Copyright (C) 2026 David (Corey) Herrera */
/**
 * @file   hosts_hub.c
 * @brief  Implementation of the hub's held waits (see hosts_hub.h). Writes to a
 *         wait fd happen only after the fd is taken off the table, never under
 *         the hub lock, and the hub never holds its lock while it calls into the
 *         registry (spec §6.2).
 *
 * @status None
 * @issues None
 * @todo   None
 */

#include "hosts_hub.h"

#include <errno.h>
#include <fcntl.h>
#include <limits.h>
#include <poll.h>
#include <pthread.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <unistd.h>

#include <sys/socket.h>
#include <sys/time.h>

#include "../serve/http_server.h"
#include "../serve/json_writer.h"

/* One held wait; fd < 0 marks a free slot. `gen` changes on every reuse so a
 * slot read before a replace is never mistaken for the new wait. */
typedef struct {
    int       fd;
    unsigned  gen;
    char      key[PN_REG_KEY_MAX];
    char      key_id[PN_HUB_KEY_ID_MAX];
    long long since;
    uint64_t  deadline_ms;
    int       check;          /* look for events above `since` */
} hub_wait;

typedef enum { ANSWER_EVENTS, ANSWER_REPLACED, ANSWER_CLOSE, ANSWER_DROP } hub_answer_kind;

/* A wait taken off the table, answered outside the lock. */
typedef struct {
    int                 fd;
    char                key[PN_REG_KEY_MAX];
    long long           since;
    hub_answer_kind     kind;
    pn_reg_close_reason reason;
} hub_answer;

/* A slot to look at for pending events (copied out under the lock). */
typedef struct {
    int       slot;
    unsigned  gen;
    char      key[PN_REG_KEY_MAX];
    long long since;
} hub_check;

struct pn_hub {
    pn_reg         *reg;
    pn_hub_relays  *relays;
    pn_hub_cfg      cfg;
    pthread_mutex_t lock;
    pthread_t       thread;
    int             thread_started;
    int             stop;
    int             wake_r, wake_w;
    hub_wait       *waits;           /* cfg.max_waits slots */
    int             n_held;
    unsigned        gen_next;
    unsigned long   wakeups;
    int             sweep_changed;   /* a sweep disconnected rows: check every wait */
    int             n_pfds;          /* self-pipe + waits + 2 per relay */
    /* Hub-thread scratch, sized to max_waits + 1 (pfds: n_pfds). */
    hub_answer     *answers;
    hub_check      *checks;
    struct pollfd  *pfds;
    int            *poll_slot;
    unsigned       *poll_gen;
};

static uint64_t hub_now_ms(void)
{
    struct timespec ts;

    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (uint64_t)ts.tv_sec * 1000u + (uint64_t)ts.tv_nsec / 1000000u;
}

/* Best-effort wake of poll(); a full pipe already means a wake is pending. */
static void hub_wake(pn_hub *hub)
{
    char b = 1;

    if (write(hub->wake_w, &b, 1) < 0 && errno != EAGAIN && errno != EWOULDBLOCK)
        fprintf(stderr, "local: hub wake failed: %s\n", strerror(errno));
}

/* Take slot i off the table into *out. Lock held. */
static void detach_locked(pn_hub *hub, int i, hub_answer_kind kind, pn_reg_close_reason reason,
                          hub_answer *out)
{
    hub_wait *w = &hub->waits[i];

    out->fd = w->fd;
    snprintf(out->key, sizeof(out->key), "%s", w->key);
    out->since = w->since;
    out->kind = kind;
    out->reason = reason;
    w->fd = -1;
    w->check = 0;
    hub->n_held--;
}

static void write_event(json_writer *jw, pn_hub *hub, const pn_reg_event *e,
                        const pn_reg_share_entry *shares, int n_shares, long long now)
{
    char handle[64];

    jw_printf(jw, "{\"seq\":%lld,\"type\":\"%s\"", e->seq, pn_reg_event_type_name(e->type));
    if (e->type == PN_REG_EV_RELAY)
        jw_printf(jw, ",\"id\":\"%s\",\"expires_in\":%lld", e->relay_id,
                  e->relay_expires > now ? e->relay_expires - now : 0);
    if (e->type == PN_REG_EV_CLOSE)
        jw_printf(jw, ",\"reason\":\"%s\"", pn_reg_close_reason_name(e->reason));
    if (e->type == PN_REG_EV_SHARES) {
        jw_raw(jw, ",\"shares\":[");
        for (int s = 0; s < n_shares; s++) {
            if (!hub->cfg.handle_of ||
                hub->cfg.handle_of(shares[s].person, handle, sizeof(handle), hub->cfg.user) != 0)
                handle[0] = '\0';
            jw_printf(jw, "%s{\"person\":\"%s\",\"handle\":", s ? "," : "", shares[s].person);
            jw_string(jw, handle);
            jw_printf(jw, ",\"role\":\"%s\"}", pn_reg_role_name(shares[s].role));
        }
        jw_raw(jw, "]");
    }
    jw_raw(jw, "}");
}

/* The body of a wait answer: the row's events above `since`. */
static void write_events_body(json_writer *jw, pn_hub *hub, const hub_answer *a)
{
    pn_reg_event ev[PN_REG_QUEUE_MAX];
    pn_reg_share_entry shares[PN_REG_MAX_SHARES];
    long long seq = a->since;
    int n_shares = 0;
    int n = pn_reg_events_since(hub->reg, a->key, a->since, ev, PN_REG_QUEUE_MAX, &seq, shares,
                                &n_shares);

    if (n <= 0)
        seq = a->since;   /* nothing new: the host keeps its `since` */
    jw_printf(jw, "{\"seq\":%lld,\"events\":[", seq);
    for (int i = 0; i < n; i++) {
        if (i)
            jw_raw(jw, ",");
        write_event(jw, hub, &ev[i], shares, n_shares, (long long)time(NULL));
    }
    jw_raw(jw, a->kind == ANSWER_REPLACED ? "],\"replaced\":true}" : "]}");
}

/* A `close` the box sends itself (shutdown, revoked), numbered above both the
 * host's `since` and the row's counter. */
static void write_close_body(json_writer *jw, pn_hub *hub, const hub_answer *a)
{
    pn_reg_row_info row;
    long long s = a->since;

    if (pn_reg_get(hub->reg, a->key, &row) == PN_REG_OK && row.seq > s)
        s = row.seq;
    jw_printf(jw, "{\"seq\":%lld,\"events\":[{\"seq\":%lld,\"type\":\"close\",\"reason\":\"%s\"}]}",
              s + 1, s + 1, pn_reg_close_reason_name(a->reason));
}

static void answer_one(pn_hub *hub, const hub_answer *a)
{
    json_writer jw;

    if (a->kind != ANSWER_DROP) {
        if (!jw_init(&jw)) {
            hs_send_error(a->fd, 500, "server_error", "oom");
        } else {
            if (a->kind == ANSWER_CLOSE)
                write_close_body(&jw, hub, a);
            else if (a->kind == ANSWER_REPLACED) {
                jw_printf(&jw, "{\"seq\":%lld,\"events\":[],\"replaced\":true}", a->since);
            } else {
                write_events_body(&jw, hub, a);
            }
            hs_send_json(a->fd, 200, jw.data);
            jw_free(&jw);
        }
    }
    hs_close(a->fd);
}

/* Timed-out waits, and waits whose row now has something to answer. */
static int collect_due(pn_hub *hub, uint64_t now)
{
    int n_ans = 0, n_checks = 0;

    pthread_mutex_lock(&hub->lock);
    for (int i = 0; i < hub->cfg.max_waits; i++) {
        hub_wait *w = &hub->waits[i];
        if (w->fd < 0)
            continue;
        if (w->deadline_ms <= now) {
            detach_locked(hub, i, ANSWER_EVENTS, PN_REG_CLOSE_SHUTDOWN, &hub->answers[n_ans++]);
        } else if (w->check || hub->sweep_changed) {
            hub_check *c = &hub->checks[n_checks++];
            c->slot = i;
            c->gen = w->gen;
            snprintf(c->key, sizeof(c->key), "%s", w->key);
            c->since = w->since;
            w->check = 0;
        }
    }
    hub->sweep_changed = 0;
    pthread_mutex_unlock(&hub->lock);
    for (int k = 0; k < n_checks; k++) {
        const hub_check *c = &hub->checks[k];
        if (pn_reg_pending(hub->reg, c->key, c->since) == 0)
            continue;
        pthread_mutex_lock(&hub->lock);
        if (hub->waits[c->slot].fd >= 0 && hub->waits[c->slot].gen == c->gen)
            detach_locked(hub, c->slot, ANSWER_EVENTS, PN_REG_CLOSE_SHUTDOWN,
                          &hub->answers[n_ans++]);
        pthread_mutex_unlock(&hub->lock);
    }
    return n_ans;
}

/* The registry's sweep and delayed save, from the hub thread. */
static void run_registry_timers(pn_hub *hub, uint64_t now, uint64_t *next_sweep)
{
    long long unix_now = (long long)time(NULL);

    if (now >= *next_sweep) {
        *next_sweep = now + (uint64_t)hub->cfg.sweep_ms;
        if (pn_reg_sweep(hub->reg, unix_now) > 0) {
            pthread_mutex_lock(&hub->lock);
            hub->sweep_changed = 1;
            pthread_mutex_unlock(&hub->lock);
        }
    }
    if (pn_reg_save_if_dirty(hub->reg, unix_now) < 0)
        fprintf(stderr, "local: sessions.json save failed; retrying in %d s\n",
                PN_REG_SAVE_DELAY_S);
}

/* Fill pfds (self-pipe first, then every held wait) and return the poll
 * timeout: the nearest wait deadline, sweep, or save. Lock held. */
static int build_poll_locked(pn_hub *hub, uint64_t now, uint64_t next_sweep, uint64_t save_ms,
                             int *nf_out)
{
    uint64_t nearest = next_sweep;
    int nf = 1;

    hub->pfds[0].fd = hub->wake_r;
    hub->pfds[0].events = POLLIN;
    hub->pfds[0].revents = 0;
    if (save_ms && save_ms < nearest)
        nearest = save_ms;
    for (int i = 0; i < hub->cfg.max_waits; i++) {
        hub_wait *w = &hub->waits[i];
        if (w->fd < 0)
            continue;
        hub->pfds[nf].fd = w->fd;
        hub->pfds[nf].events = POLLIN;
        hub->pfds[nf].revents = 0;
        hub->poll_slot[nf] = i;
        hub->poll_gen[nf] = w->gen;
        nf++;
        if (w->deadline_ms < nearest)
            nearest = w->deadline_ms;
        if (w->check || hub->sweep_changed)
            nearest = now;
    }
    *nf_out = nf;
    return nearest <= now ? 0 : nearest - now > INT_MAX ? INT_MAX : (int)(nearest - now);
}

/* A readable wait fd: a body-less GET has nothing more to send, so the peer
 * left (or broke the protocol). Drop it without an answer. */
static void drop_readable(pn_hub *hub, int nf)
{
    for (int k = 1; k < nf; k++) {
        hub_answer a;
        int slot = hub->poll_slot[k];
        if (!(hub->pfds[k].revents & (POLLIN | POLLHUP | POLLERR | POLLNVAL)))
            continue;
        a.fd = -1;
        pthread_mutex_lock(&hub->lock);
        if (hub->waits[slot].fd == hub->pfds[k].fd && hub->waits[slot].gen == hub->poll_gen[k])
            detach_locked(hub, slot, ANSWER_DROP, PN_REG_CLOSE_SHUTDOWN, &a);
        pthread_mutex_unlock(&hub->lock);
        if (a.fd >= 0)
            hs_close(a.fd);
    }
}

static void *hub_thread(void *arg)
{
    pn_hub *hub = arg;
    uint64_t next_sweep = hub_now_ms() + (uint64_t)hub->cfg.sweep_ms;

    for (;;) {
        uint64_t now = hub_now_ms(), save_ms = 0;
        long long due = 0;
        int n_ans = 0, nf = 0, timeout = 0, r = 0, n_relay_fds = 0;
        uint64_t relay_deadline = UINT64_MAX;

        run_registry_timers(hub, now, &next_sweep);
        n_ans = collect_due(hub, now);
        for (int i = 0; i < n_ans; i++)
            answer_one(hub, &hub->answers[i]);
        due = pn_reg_save_due(hub->reg);
        if (due)
            save_ms = now + (uint64_t)(due > (long long)time(NULL) ? due - time(NULL) : 0) * 1000u;
        pthread_mutex_lock(&hub->lock);
        if (hub->stop) {
            pthread_mutex_unlock(&hub->lock);
            break;
        }
        timeout = build_poll_locked(hub, now, next_sweep, save_ms, &nf);
        pthread_mutex_unlock(&hub->lock);
        relay_deadline = UINT64_MAX;
        n_relay_fds = pn_hub_relays_poll_fill(hub->relays, hub->pfds + nf, hub->n_pfds - nf,
                                              &relay_deadline);
        if (relay_deadline != UINT64_MAX)
            timeout = relay_deadline <= now ? 0
                    : relay_deadline - now < (uint64_t)timeout ? (int)(relay_deadline - now)
                    : timeout;
        r = poll(hub->pfds, (nfds_t)(nf + n_relay_fds), timeout);
        pthread_mutex_lock(&hub->lock);
        hub->wakeups++;
        pthread_mutex_unlock(&hub->lock);
        if (r < 0) {
            if (errno != EINTR) {
                struct timespec ts = { .tv_sec = 0, .tv_nsec = 50 * 1000000 };
                fprintf(stderr, "local: hub poll failed: %s\n", strerror(errno));
                nanosleep(&ts, NULL);
            }
            continue;
        }
        if (hub->pfds[0].revents & POLLIN) {
            char buf[64];
            while (read(hub->wake_r, buf, sizeof(buf)) > 0) {
            }
        }
        drop_readable(hub, nf);
        pn_hub_relays_service(hub->relays, hub->pfds + nf, n_relay_fds, hub_now_ms());
    }
    return NULL;
}

static int set_nonblock(int fd)
{
    int fl = fcntl(fd, F_GETFL, 0);
    return fl < 0 ? -1 : fcntl(fd, F_SETFL, fl | O_NONBLOCK);
}

static int alloc_scratch(pn_hub *hub)
{
    size_t n = (size_t)hub->cfg.max_waits + 1;

    hub->waits = calloc(n, sizeof(*hub->waits));
    hub->answers = calloc(n, sizeof(*hub->answers));
    hub->checks = calloc(n, sizeof(*hub->checks));
    hub->n_pfds = (int)n + 2 * hub->cfg.relays.max_relays;
    hub->pfds = calloc((size_t)hub->n_pfds, sizeof(*hub->pfds));
    hub->poll_slot = calloc(n, sizeof(*hub->poll_slot));
    hub->poll_gen = calloc(n, sizeof(*hub->poll_gen));
    if (!hub->waits || !hub->answers || !hub->checks || !hub->pfds || !hub->poll_slot ||
        !hub->poll_gen)
        return -1;
    for (size_t i = 0; i < n; i++)
        hub->waits[i].fd = -1;
    return 0;
}

pn_hub *pn_hub_start(pn_reg *reg, const pn_hub_cfg *cfg)
{
    pn_hub *hub = reg ? calloc(1, sizeof(*hub)) : NULL;
    int p[2] = { -1, -1 };

    if (!hub)
        return NULL;
    if (cfg)
        hub->cfg = *cfg;
    hub->cfg.wait_ms = hub->cfg.wait_ms > 0 ? hub->cfg.wait_ms : PN_HUB_WAIT_MS;
    hub->cfg.sweep_ms = hub->cfg.sweep_ms > 0 ? hub->cfg.sweep_ms : PN_HUB_SWEEP_MS;
    hub->cfg.max_waits = hub->cfg.max_waits > 0 && hub->cfg.max_waits <= PN_HUB_WAITS_MAX
                         ? hub->cfg.max_waits : PN_HUB_WAITS_MAX;
    hub->cfg.relays.max_relays = hub->cfg.relays.max_relays > 0 ? hub->cfg.relays.max_relays
                                                                 : PN_HUB_RELAYS_MAX;
    hub->cfg.relays.host_ms = hub->cfg.relays.host_ms > 0 ? hub->cfg.relays.host_ms
                                                           : PN_HUB_RELAY_HOST_MS;
    hub->reg = reg;
    hub->wake_r = hub->wake_w = -1;
    pthread_mutex_init(&hub->lock, NULL);
    hub->relays = pn_hub_relays_new(&hub->cfg.relays);
    if (!hub->relays || alloc_scratch(hub) != 0 || pipe(p) != 0) {
        fprintf(stderr, "local: the hub could not start (memory or pipe)\n");
        pn_hub_free(hub);
        return NULL;
    }
    hub->wake_r = p[0];
    hub->wake_w = p[1];
    if (set_nonblock(hub->wake_r) != 0 || set_nonblock(hub->wake_w) != 0 ||
        pthread_create(&hub->thread, NULL, hub_thread, hub) != 0) {
        fprintf(stderr, "local: the hub thread could not start\n");
        pn_hub_free(hub);
        return NULL;
    }
    hub->thread_started = 1;
    return hub;
}

int pn_hub_add_wait(pn_hub *hub, int fd, const char *key, const char *key_id, long long since)
{
    struct timeval tv = { PN_HUB_SEND_TIMEOUT_MS / 1000, (PN_HUB_SEND_TIMEOUT_MS % 1000) * 1000 };
    hub_answer old = { .fd = -1 };
    int slot = -1, taken = 0;

    if (!hub || fd < 0 || !pn_reg_key_valid(key))
        return -1;
    if (setsockopt(fd, SOL_SOCKET, SO_SNDTIMEO, &tv, sizeof(tv)) != 0)
        fprintf(stderr, "local: wait fd send timeout not set: %s\n", strerror(errno));
    pthread_mutex_lock(&hub->lock);
    for (int i = 0; !hub->stop && i < hub->cfg.max_waits; i++) {
        if (hub->waits[i].fd >= 0 && strcmp(hub->waits[i].key, key) == 0) {
            detach_locked(hub, i, ANSWER_REPLACED, PN_REG_CLOSE_SHUTDOWN, &old);
            slot = i;
            break;
        }
        if (slot < 0 && hub->waits[i].fd < 0)
            slot = i;
    }
    if (!hub->stop && slot >= 0) {
        hub_wait *w = &hub->waits[slot];
        w->fd = fd;
        w->gen = ++hub->gen_next;
        snprintf(w->key, sizeof(w->key), "%s", key);
        snprintf(w->key_id, sizeof(w->key_id), "%s", key_id ? key_id : "");
        w->since = since;
        w->deadline_ms = hub_now_ms() + (uint64_t)hub->cfg.wait_ms;
        w->check = 1;
        hub->n_held++;
        taken = 1;
    }
    pthread_mutex_unlock(&hub->lock);
    if (old.fd >= 0)
        answer_one(hub, &old);
    if (!taken)
        return -1;
    hub_wake(hub);
    return 0;
}

void pn_hub_notify(pn_hub *hub, const char *key)
{
    int found = 0;

    if (!hub || !key)
        return;
    pthread_mutex_lock(&hub->lock);
    for (int i = 0; i < hub->cfg.max_waits; i++)
        if (hub->waits[i].fd >= 0 && strcmp(hub->waits[i].key, key) == 0) {
            hub->waits[i].check = 1;
            found = 1;
        }
    pthread_mutex_unlock(&hub->lock);
    if (found)
        hub_wake(hub);
}

/* Answer every held wait matching key_id (NULL = all) with `close` `reason`. */
static int close_matching(pn_hub *hub, const char *key_id, pn_reg_close_reason reason)
{
    hub_answer a;
    int n = 0;

    for (int i = 0; hub->waits && i < hub->cfg.max_waits; i++) {
        a.fd = -1;
        pthread_mutex_lock(&hub->lock);
        if (hub->waits[i].fd >= 0 && (!key_id || strcmp(hub->waits[i].key_id, key_id) == 0))
            detach_locked(hub, i, ANSWER_CLOSE, reason, &a);
        pthread_mutex_unlock(&hub->lock);
        if (a.fd >= 0) {
            answer_one(hub, &a);
            n++;
        }
    }
    return n;
}

int pn_hub_close_device(pn_hub *hub, const char *key_id, pn_reg_close_reason reason)
{
    return hub && key_id && key_id[0] ? close_matching(hub, key_id, reason) : 0;
}

void pn_hub_stop(pn_hub *hub)
{
    int n = 0;

    if (!hub)
        return;
    pthread_mutex_lock(&hub->lock);
    hub->stop = 1;
    pthread_mutex_unlock(&hub->lock);
    if (hub->thread_started) {
        hub_wake(hub);
        pthread_join(hub->thread, NULL);
        hub->thread_started = 0;
    }
    n = close_matching(hub, NULL, PN_REG_CLOSE_SHUTDOWN);
    if (n)
        fprintf(stderr, "local: told %d held host wait%s the box is stopping\n", n,
                n == 1 ? "" : "s");
    n = hub->relays ? pn_hub_relays_close_all(hub->relays) : 0;
    if (n)
        fprintf(stderr, "local: closed %d relay%s at stop\n", n, n == 1 ? "" : "s");
}

void pn_hub_free(pn_hub *hub)
{
    if (!hub)
        return;
    pn_hub_stop(hub);
    if (hub->wake_r >= 0)
        close(hub->wake_r);
    if (hub->wake_w >= 0)
        close(hub->wake_w);
    pthread_mutex_destroy(&hub->lock);
    pn_hub_relays_free(hub->relays);
    free(hub->waits);
    free(hub->answers);
    free(hub->checks);
    free(hub->pfds);
    free(hub->poll_slot);
    free(hub->poll_gen);
    free(hub);
}

int pn_hub_waits_held(pn_hub *hub)
{
    int n = 0;

    pthread_mutex_lock(&hub->lock);
    n = hub->n_held;
    pthread_mutex_unlock(&hub->lock);
    return n;
}

unsigned long pn_hub_wakeups(pn_hub *hub)
{
    unsigned long n = 0;

    pthread_mutex_lock(&hub->lock);
    n = hub->wakeups;
    pthread_mutex_unlock(&hub->lock);
    return n;
}

/* Queue `relay {id}` on the row; the host must join within the relay deadline. */
static int announce_relay(const char *key, const char *id, void *user)
{
    pn_hub *hub = user;
    pn_reg_event ev = { .type = PN_REG_EV_RELAY };

    ev.relay_expires = (long long)time(NULL) + (hub->cfg.relays.host_ms + 999) / 1000;
    snprintf(ev.relay_id, sizeof(ev.relay_id), "%s", id);
    return pn_reg_push_event(hub->reg, key, &ev) > 0 ? 0 : -1;
}

int pn_hub_relay_open(pn_hub *hub, int fd, const pn_hub_relay_open_args *args,
                      char id_out[PN_HUB_RELAY_ID_MAX])
{
    int rc = hub ? pn_hub_relays_open(hub->relays, fd, args, announce_relay, hub, id_out)
                 : PN_HUB_RELAY_E_INVALID;

    if (rc != PN_HUB_RELAY_OK)
        return rc;
    hub_wake(hub);
    pn_hub_notify(hub, args->key);
    return rc;
}

int pn_hub_relay_host(pn_hub *hub, int fd, const char *id, const char *person)
{
    int rc = hub ? pn_hub_relays_host(hub->relays, fd, id, person) : PN_HUB_RELAY_E_INVALID;

    if (rc == PN_HUB_RELAY_OK)
        hub_wake(hub);
    return rc;
}

int pn_hub_relays_held(pn_hub *hub)
{
    return hub ? pn_hub_relays_count(hub->relays) : 0;
}

size_t pn_hub_relay_buffered_peak(pn_hub *hub)
{
    return hub ? pn_hub_relays_buffered_peak(hub->relays) : 0;
}
