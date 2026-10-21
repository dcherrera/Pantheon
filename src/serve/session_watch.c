/* SPDX-License-Identifier: AGPL-3.0-or-later */
/* Copyright (C) 2026 David (Corey) Herrera */
/**
 * @file session_watch.c
 * @brief Implementation of the liveness-based session-release control plane
 *        (see session_watch.h). One background thread holds every
 *        GET /session/watch connection in a single poll() loop; a peer close
 *        starts a grace timer; a reconnect for the same session cancels it; a
 *        session whose watch stays closed past the grace is released. Idleness
 *        is NEVER a release trigger for a watched session — only death is.
 *
 *        A watch is held on a tab's BASE id `client-<pid>-<tab>`, but the live KV
 *        slots are the mode-suffixed sids `<base>-orch`/`-act`/`-chat` — so a
 *        base watch PROTECTS THE WHOLE `<base>-*` PREFIX: on death it releases
 *        every matching mode slot, and while alive it holds them off the 12 h
 *        backstop (via a watch-hook installed on each model's prefix cache).
 *
 * @status Built; poll loop + grace reaper CPU-tested, CUDA compile-checked.
 * @issues None
 * @todo None
 */

#include "session_watch.h"

#include <errno.h>
#include <fcntl.h>
#include <poll.h>
#include <pthread.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/socket.h>
#include <time.h>
#include <unistd.h>

#include "../core/prefix_cache.h"
#include "http_server.h"
#include "model_session.h"

#define PN_WATCH_SID_MAX  64
#define PN_WATCH_NAME_MAX MR_MAX_NAME

/* One watched session-control connection. `fd < 0` marks a free slot. When a
 * watch's connection closes the fd is closed and set to -1 but the row is KEPT
 * (with closed_at_ms set) until the grace elapses — so a reconnect within grace
 * finds it and cancels the release, and the sweep releases it after. */
typedef struct {
    int       fd;                       /* held client fd, or -1 when closed */
    char      name[PN_WATCH_NAME_MAX];  /* model id (routing + logging) */
    char      sid[PN_WATCH_SID_MAX];    /* prefix-cache session key */
    mr_entry *entry;                    /* the manager holds an mr_acquire ref */
    uint64_t  closed_at_ms;             /* wall-clock ms the watch closed; 0 =
                                         * still held (alive). Nonzero starts
                                         * the grace timer. */
    int       used;                     /* row occupied (fd may be -1 in grace) */
} pn_watch;

/* Max concurrent watches. The K80 test machine holds ~14 model-dies; each model caps its
 * live sessions (max_sessions) — a generous ceiling covers every session on
 * every model holding a watch at once. */
#define PN_WATCH_MAX 256

/* A per-model backstop-hook binding: the ctx handed to each cache's watch-hook.
 * Carries the manager plus the model name so pn_session_watch_base_is_watched
 * can scope the base match to the right model. Owned by the manager (freed at
 * stop). */
typedef struct {
    pn_session_watch_mgr *mgr;
    char                  name[PN_WATCH_NAME_MAX];
} pn_watch_hook_binding;

struct pn_session_watch_mgr {
    mr_registry    *reg;                /* borrowed */
    pn_watch        watches[PN_WATCH_MAX];
    pthread_mutex_t mu;                 /* guards the watch table */
    pthread_t       thread;
    int             thread_started;
    int             stop;               /* set under mu; poll thread exits */
    int             wake_r, wake_w;     /* self-pipe: wakes poll() on change */
    uint64_t        grace_ms;
    /* Backstop-hook bindings (one per model cache the hook is installed on).
     * Borrowed by the caches as their watch-hook ctx; freed at stop. */
    pn_watch_hook_binding *hook_bindings[MR_MAX_MODELS];
    int                    n_hook_bindings;
};

/* Monotonic wall clock in ms — matches oai_now_ms so grace timing is coherent
 * with the request-path reaper (both CLOCK_MONOTONIC). */
static uint64_t watch_now_ms(void)
{
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (uint64_t)ts.tv_sec * 1000u + (uint64_t)ts.tv_nsec / 1000000u;
}

uint64_t pn_session_watch_grace_ms(void)
{
    static uint64_t cached = 0;
    static int loaded = 0;
    if (!loaded) {
        const char *env = getenv("PN_SESSION_WATCH_GRACE_MS");
        if (env && *env) {
            char *end = NULL;
            long v = strtol(env, &end, 10);
            cached = (end != env && *end == '\0' && v >= 0)
                   ? (uint64_t)v : 45u * 1000u;
        } else {
            cached = 45u * 1000u;            /* 45 seconds */
        }
        loaded = 1;
    }
    return cached;
}

uint64_t pn_session_watch_backstop_ms(void)
{
    static uint64_t cached = 0;
    static int loaded = 0;
    if (!loaded) {
        const char *env = getenv("PN_SESSION_IDLE_TTL_MS");
        if (env && *env) {
            char *end = NULL;
            long v = strtol(env, &end, 10);
            cached = (end != env && *end == '\0' && v >= 0)
                   ? (uint64_t)v : 12u * 60u * 60u * 1000u;
        } else {
            cached = 12u * 60u * 60u * 1000u;   /* 12 hours */
        }
        loaded = 1;
    }
    return cached;
}

/* Release a dead base's live sessions through the session facade, which covers
 * BOTH stores under the right lock — the scheduler's warm decode slots on the
 * batched path, the prefix cache on the single-user one — so the release never
 * races a concurrent generation/admit on that model. (Clearing only the prefix
 * cache, what this used to do, freed nothing on a batched model: a dead tab's
 * context stayed bound to its decode slot.) Caller must NOT hold mgr->mu (this
 * takes a model lock; keep the two lock orders independent — mgr->mu is never
 * held across a model lock).
 *
 * `sid` is the watched BASE id `client-<pid>-<tab>`, but the live KV slots are the
 * mode-suffixed sids `<base>-orch`/`-act`/`-chat` — so release the whole
 * `<base>-*` prefix, not just an exact-match base entry (which usually does not
 * even exist). Also release any exact-match base entry, for a client that keyed
 * its live slot on the bare base. */
static void release_session(mr_entry *entry, const char *sid)
{
    if (!entry || !sid || !sid[0])
        return;
    char base_prefix[PN_WATCH_SID_MAX + 1];
    snprintf(base_prefix, sizeof(base_prefix), "%s-", sid);
    ms_session_release_by_prefix(entry, base_prefix);
    ms_session_release(entry, sid);
}

/* Wake the poll thread (the table changed). Best-effort one byte on the
 * self-pipe; a full pipe already means "wakeup pending", so EAGAIN is fine. */
static void watch_wake(pn_session_watch_mgr *m)
{
    if (m->wake_w < 0)
        return;
    char b = 1;
    ssize_t n = write(m->wake_w, &b, 1);
    (void)n;
}

/* Find a live-or-in-grace watch row for (name,sid). Caller holds mu. */
static pn_watch *watch_find(pn_session_watch_mgr *m, const char *name,
                            const char *sid)
{
    for (int i = 0; i < PN_WATCH_MAX; i++) {
        pn_watch *w = &m->watches[i];
        if (w->used && strncmp(w->sid, sid, PN_WATCH_SID_MAX) == 0 &&
            strncmp(w->name, name, PN_WATCH_NAME_MAX) == 0)
            return w;
    }
    return NULL;
}

/* Is `entry_sid` (a live prefix-cache slot, e.g. the mode-suffixed
 * `client-<pid>-<tab>-orch`) covered by a live-or-in-grace watch on model `name`?
 * A watch is held on the BASE id `client-<pid>-<tab>`; it covers `entry_sid` when
 * entry_sid == "<watch.sid>-<something>" — i.e. entry_sid starts with the watch's
 * sid followed by '-'. An exact base match also counts (a slot keyed on the bare
 * base). Takes mu; the caller must NOT hold it. Used by the backstop reaper's
 * watch-hook so a watched base's mode slots are never idle-reaped. */
int pn_session_watch_base_is_watched(pn_session_watch_mgr *m, const char *name,
                                     const char *entry_sid)
{
    if (!m || !name || !entry_sid || !entry_sid[0])
        return 0;
    int watched = 0;
    pthread_mutex_lock(&m->mu);
    for (int i = 0; i < PN_WATCH_MAX; i++) {
        pn_watch *w = &m->watches[i];
        if (!w->used || strncmp(w->name, name, PN_WATCH_NAME_MAX) != 0)
            continue;
        size_t blen = strnlen(w->sid, PN_WATCH_SID_MAX);
        if (blen == 0)
            continue;
        /* Exact base, or entry_sid == "<base>-<suffix>". */
        if (strncmp(entry_sid, w->sid, blen) == 0 &&
            (entry_sid[blen] == '\0' || entry_sid[blen] == '-')) {
            watched = 1;
            break;
        }
    }
    pthread_mutex_unlock(&m->mu);
    return watched;
}

/* Cache watch-hook adapter (matches pn_session_watched_fn): resolve the binding
 * and ask "is this slot's base watched on this model?". */
static int watch_hook_adapter(const char *entry_sid, void *ctx)
{
    pn_watch_hook_binding *b = ctx;
    if (!b)
        return 0;
    return pn_session_watch_base_is_watched(b->mgr, b->name, entry_sid);
}

void pn_session_watch_install_backstop_hooks(pn_session_watch_mgr *m)
{
    if (!m || !m->reg)
        return;
    pthread_mutex_lock(&m->reg->lock);
    for (int i = 0; i < MR_MAX_MODELS; i++) {
        mr_entry *e = &m->reg->entries[i];
        if (!e->used || !e->prefix_cache)
            continue;
        if (m->n_hook_bindings >= MR_MAX_MODELS)
            break;
        pn_watch_hook_binding *b = calloc(1, sizeof(*b));
        if (!b)
            continue;   /* OOM: this cache just keeps the default (no hook) */
        b->mgr = m;
        snprintf(b->name, sizeof(b->name), "%s", e->name);
        pn_prefix_cache_set_watch_hook(e->prefix_cache, watch_hook_adapter, b);
        m->hook_bindings[m->n_hook_bindings++] = b;
    }
    pthread_mutex_unlock(&m->reg->lock);
}

/* Drop a watch row entirely: close its fd if still open, release the manager's
 * registry reference, clear the row. Caller holds mu. Does NOT release the
 * session (that is the sweep's decision) unless `release` is set. */
static void watch_drop(pn_session_watch_mgr *m, pn_watch *w, int release)
{
    if (!w->used)
        return;
    if (release)
        release_session(w->entry, w->sid);
    if (w->fd >= 0) {
        close(w->fd);
        w->fd = -1;
    }
    if (w->entry)
        mr_release(m->reg, w->entry);
    w->entry = NULL;
    w->used = 0;
    w->closed_at_ms = 0;
    w->sid[0] = '\0';
    w->name[0] = '\0';
}

/* Mark a watch's connection closed (peer hung up): close the fd, start the
 * grace timer, keep the row so a reconnect within grace cancels the release.
 * Caller holds mu. */
static void watch_mark_closed(pn_watch *w, uint64_t now_ms)
{
    if (w->fd >= 0) {
        close(w->fd);
        w->fd = -1;
    }
    if (w->closed_at_ms == 0)
        w->closed_at_ms = now_ms ? now_ms : 1;   /* nonzero = grace running */
}

/* Grace sweep: release any watch closed longer than the grace ago (no
 * reconnect cancelled it). Runs under mu for the table scan; the actual
 * pn_prefix_cache_release_session happens on a snapshot AFTER dropping mu so
 * the model lock is never taken under mgr->mu. Returns nothing. */
static void watch_sweep(pn_session_watch_mgr *m, uint64_t now_ms)
{
    /* Collect the (entry,sid) to release + rows to clear under mu, then act
     * without mu held (release_session takes a model lock). */
    struct { mr_entry *entry; char sid[PN_WATCH_SID_MAX]; int idx; }
        due[PN_WATCH_MAX];
    int n_due = 0;

    pthread_mutex_lock(&m->mu);
    for (int i = 0; i < PN_WATCH_MAX; i++) {
        pn_watch *w = &m->watches[i];
        if (!w->used || w->closed_at_ms == 0)
            continue;
        if (now_ms >= w->closed_at_ms &&
            now_ms - w->closed_at_ms >= m->grace_ms) {
            due[n_due].entry = w->entry;
            snprintf(due[n_due].sid, sizeof(due[n_due].sid), "%s", w->sid);
            due[n_due].idx = i;
            n_due++;
            /* Detach the row from the manager's reference bookkeeping here but
             * defer the mr_release + session release to outside mu. Mark the
             * row as "releasing" by clearing fd only; the row stays used until
             * we finish so a concurrent add for the same key still finds it and
             * is handled deterministically (it will fail to match a live fd and
             * we clear below). Simplest correct approach: clear the row now
             * (used=0) and stash entry for release — a concurrent add creates a
             * fresh row. */
            fprintf(stderr, "[session_watch] releasing sid=%s model=%s "
                    "(watch closed %llums ago >= grace %llums)\n",
                    w->sid, w->name,
                    (unsigned long long)(now_ms - w->closed_at_ms),
                    (unsigned long long)m->grace_ms);
            w->used = 0;
            w->closed_at_ms = 0;
            w->entry = NULL;   /* ownership of the ref moves to `due` */
            w->sid[0] = '\0';
            w->name[0] = '\0';
        }
    }
    pthread_mutex_unlock(&m->mu);

    for (int i = 0; i < n_due; i++) {
        release_session(due[i].entry, due[i].sid);
        if (due[i].entry)
            mr_release(m->reg, due[i].entry);
    }
}

void pn_session_watch_sweep_for_test(pn_session_watch_mgr *m, uint64_t now_ms)
{
    if (m)
        watch_sweep(m, now_ms);
}

int pn_session_watch_mark_closed_for_test(pn_session_watch_mgr *m,
                                          const char *name, const char *sid,
                                          uint64_t now_ms)
{
    if (!m || !name || !sid)
        return -1;
    pthread_mutex_lock(&m->mu);
    pn_watch *w = watch_find(m, name, sid);
    if (w)
        watch_mark_closed(w, now_ms);
    pthread_mutex_unlock(&m->mu);
    return w ? 0 : -1;
}

/* The poll loop: hold every open watch fd + the self-pipe. Readability on a
 * watch fd means the peer closed (a GET /session/watch client sends no body) —
 * mark it closed and start its grace. The timeout drives the periodic grace
 * sweep. */
static void *watch_thread(void *arg)
{
    pn_session_watch_mgr *m = arg;
    struct pollfd pfds[PN_WATCH_MAX + 1];
    pn_watch *rows[PN_WATCH_MAX];   /* pfds[k] (k>=1) -> row */

    for (;;) {
        pthread_mutex_lock(&m->mu);
        if (m->stop) {
            pthread_mutex_unlock(&m->mu);
            break;
        }
        int nf = 0;
        pfds[nf].fd = m->wake_r;
        pfds[nf].events = POLLIN;
        pfds[nf].revents = 0;
        nf++;
        for (int i = 0; i < PN_WATCH_MAX; i++) {
            pn_watch *w = &m->watches[i];
            if (w->used && w->fd >= 0) {
                pfds[nf].fd = w->fd;
                pfds[nf].events = POLLIN;   /* POLLIN|POLLHUP|POLLERR all fire */
                pfds[nf].revents = 0;
                rows[nf] = w;
                nf++;
            }
        }
        pthread_mutex_unlock(&m->mu);

        /* Wake at least each grace/4 (min 250 ms, max 5 s) so a closed watch
         * with no other fd activity is still swept promptly. */
        uint64_t g = m->grace_ms ? m->grace_ms : 45000;
        int timeout_ms = (int)(g / 4);
        if (timeout_ms < 250) timeout_ms = 250;
        if (timeout_ms > 5000) timeout_ms = 5000;

        int r = poll(pfds, (nfds_t)nf, timeout_ms);
        if (r < 0) {
            if (errno == EINTR)
                continue;
            /* Unexpected poll failure: back off briefly, keep serving. */
            struct timespec ts = { .tv_sec = 0, .tv_nsec = 50 * 1000000 };
            nanosleep(&ts, NULL);
            continue;
        }

        uint64_t now = watch_now_ms();

        /* Drain the self-pipe (table changed / stop requested). */
        if (pfds[0].revents & POLLIN) {
            char buf[64];
            while (read(m->wake_r, buf, sizeof(buf)) > 0) { /* drain */ }
        }

        /* Any watch fd readable/hung-up → peer closed. Confirm with the same
         * MSG_PEEK==0 test hs_client_alive uses (a stray POLLIN with buffered
         * bytes on a body-less GET is not a close, though clients never send
         * one — belt and suspenders). Mark closed under mu. */
        pthread_mutex_lock(&m->mu);
        for (int k = 1; k < nf; k++) {
            if (!(pfds[k].revents & (POLLIN | POLLHUP | POLLERR)))
                continue;
            pn_watch *w = rows[k];
            /* The row may have been replaced/dropped while mu was released; only
             * act if it is still the same open fd. */
            if (!w->used || w->fd != pfds[k].fd)
                continue;
            if (!hs_client_alive(w->fd)) {
                fprintf(stderr, "[session_watch] watch dropped sid=%s model=%s "
                        "(grace %llums started)\n", w->sid, w->name,
                        (unsigned long long)m->grace_ms);
                watch_mark_closed(w, now);
            }
        }
        pthread_mutex_unlock(&m->mu);

        watch_sweep(m, now);
    }
    return NULL;
}

/* Shared constructor for the started + manual (test) managers. */
static pn_session_watch_mgr *watch_new(mr_registry *reg, int start_thread)
{
    if (!reg)
        return NULL;
    pn_session_watch_mgr *m = calloc(1, sizeof(*m));
    if (!m)
        return NULL;
    m->reg = reg;
    m->grace_ms = pn_session_watch_grace_ms();
    m->wake_r = m->wake_w = -1;
    for (int i = 0; i < PN_WATCH_MAX; i++)
        m->watches[i].fd = -1;
    if (pthread_mutex_init(&m->mu, NULL) != 0) {
        free(m);
        return NULL;
    }

    int p[2];
    if (pipe(p) != 0) {
        pthread_mutex_destroy(&m->mu);
        free(m);
        return NULL;
    }
    m->wake_r = p[0];
    m->wake_w = p[1];
    /* Non-blocking self-pipe so watch_wake never blocks and the drain loop ends
     * on EAGAIN. */
    int fl;
    if ((fl = fcntl(m->wake_r, F_GETFL, 0)) != -1)
        fcntl(m->wake_r, F_SETFL, fl | O_NONBLOCK);
    if ((fl = fcntl(m->wake_w, F_GETFL, 0)) != -1)
        fcntl(m->wake_w, F_SETFL, fl | O_NONBLOCK);

    if (start_thread) {
        if (pthread_create(&m->thread, NULL, watch_thread, m) != 0) {
            close(m->wake_r);
            close(m->wake_w);
            pthread_mutex_destroy(&m->mu);
            free(m);
            return NULL;
        }
        m->thread_started = 1;
    }
    return m;
}

pn_session_watch_mgr *pn_session_watch_start(mr_registry *reg)
{
    return watch_new(reg, 1);
}

/* Test-only manual manager (no background thread; the test drives sweeps). */
pn_session_watch_mgr *pn_session_watch_new_manual(mr_registry *reg)
{
    return watch_new(reg, 0);
}

void pn_session_watch_stop(pn_session_watch_mgr *m)
{
    if (!m)
        return;
    pthread_mutex_lock(&m->mu);
    m->stop = 1;
    pthread_mutex_unlock(&m->mu);
    watch_wake(m);
    if (m->thread_started)
        pthread_join(m->thread, NULL);

    /* Close every held/in-grace watch and release the manager's refs. Does NOT
     * release the sessions — shutdown reclaims VRAM via the model teardown. */
    for (int i = 0; i < PN_WATCH_MAX; i++)
        watch_drop(m, &m->watches[i], 0);

    /* Clear the backstop hooks off the caches (the bindings are about to be
     * freed) and free them. Best-effort under the registry lock. */
    if (m->reg) {
        pthread_mutex_lock(&m->reg->lock);
        for (int i = 0; i < MR_MAX_MODELS; i++)
            if (m->reg->entries[i].used && m->reg->entries[i].prefix_cache)
                pn_prefix_cache_set_watch_hook(m->reg->entries[i].prefix_cache,
                                               NULL, NULL);
        pthread_mutex_unlock(&m->reg->lock);
    }
    for (int i = 0; i < m->n_hook_bindings; i++)
        free(m->hook_bindings[i]);
    m->n_hook_bindings = 0;

    if (m->wake_r >= 0) close(m->wake_r);
    if (m->wake_w >= 0) close(m->wake_w);
    pthread_mutex_destroy(&m->mu);
    free(m);
}

/* The keepalive header a watch connection gets on establish: a 200 with an
 * event-stream content type and a single comment line, then held open. No body
 * follows (the client just watches for the socket to close). */
static void watch_send_established(int fd)
{
    static const char hdr[] =
        "HTTP/1.1 200 OK\r\n"
        "Content-Type: text/event-stream\r\n"
        "Cache-Control: no-cache\r\n"
        "Access-Control-Allow-Origin: *\r\n"
        "Connection: keep-alive\r\n\r\n"
        ": watch established\n\n";
    ssize_t n = write(fd, hdr, sizeof(hdr) - 1);
    (void)n;   /* best-effort; a write failure is caught as a close by poll */
}

int pn_session_watch_add(pn_session_watch_mgr *m, int fd, mr_entry *entry,
                         const char *name, const char *sid)
{
    if (!m || fd < 0 || !entry || !entry->prefix_cache || !name || !sid ||
        !sid[0])
        return -1;

    /* Take the manager's OWN reference for the fd's lifetime (the caller's
     * mr_acquire ref is released by the caller after this returns). */
    mr_entry *held = mr_acquire(m->reg, name);
    if (!held)
        return -1;

    pthread_mutex_lock(&m->mu);

    /* Reconnect: an existing watch for this (name,sid) is REPLACED — cancel any
     * pending grace and adopt the new fd. */
    pn_watch *w = watch_find(m, name, sid);
    if (w) {
        if (w->fd >= 0)
            close(w->fd);          /* the old connection is superseded */
        if (w->entry)
            mr_release(m->reg, w->entry);
        w->entry = held;
        w->fd = fd;
        w->closed_at_ms = 0;       /* alive again — grace cancelled */
        snprintf(w->name, sizeof(w->name), "%s", name);
        snprintf(w->sid, sizeof(w->sid), "%s", sid);
        w->used = 1;
        pthread_mutex_unlock(&m->mu);
        watch_send_established(fd);
        watch_wake(m);
        fprintf(stderr, "[session_watch] watch (re)established sid=%s model=%s\n",
                sid, name);
        return 0;
    }

    /* Fresh watch: take a free row. */
    for (int i = 0; i < PN_WATCH_MAX; i++) {
        if (!m->watches[i].used) {
            w = &m->watches[i];
            break;
        }
    }
    if (!w) {
        pthread_mutex_unlock(&m->mu);
        mr_release(m->reg, held);
        return -1;                  /* table full — caller keeps + closes fd */
    }
    w->entry = held;
    w->fd = fd;
    w->closed_at_ms = 0;
    snprintf(w->name, sizeof(w->name), "%s", name);
    snprintf(w->sid, sizeof(w->sid), "%s", sid);
    w->used = 1;
    pthread_mutex_unlock(&m->mu);

    watch_send_established(fd);
    watch_wake(m);
    fprintf(stderr, "[session_watch] watch established sid=%s model=%s\n",
            sid, name);
    return 0;
}
