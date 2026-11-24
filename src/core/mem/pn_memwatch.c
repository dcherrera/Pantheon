/* SPDX-License-Identifier: AGPL-3.0-or-later */
/* Copyright (C) 2026 David (Corey) Herrera */
/**
 * @file pn_memwatch.c
 * @brief Implementation of the MemoryAware poll loop + accounting hub
 *        (MemoryAware.md §2, §3, §11 Q3/Q6). Owns the single published
 *        pn_mem_state: the host-signal thread and the GPU-side die feed both
 *        write it through a seqlock so many readers get a self-consistent copy
 *        without ever blocking the writer. Derives the pressure level from the
 *        host reserve line + PSI + per-die budgets and logs on a level change.
 *
 * @status Phase 10 — level derivation split into HARD (host bytes, die VRAM;
 *         published immediately) and PSI (leading indicator; passed through a
 *         dwell filter and a noise floor before it can raise the level). Fixes
 *         the ladder tripping TIGHT on a healthy box off a 0.2% PSI blip.
 * @issues None
 * @todo None
 */

#include "pn_memwatch.h"

#include <pthread.h>
#include <stdatomic.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

#include "pn_host_signals.h"

/* ================================================================
 * Published state (seqlock single-writer / many-reader)
 *
 * Writers hold g_write_lock so publishes are serialized (the host thread and
 * the die feed both write). Readers spin the even/odd seq guard, never block a
 * writer, and retry on a torn read. No persistence — transient runtime truth.
 * ================================================================ */

static pn_mem_state       g_state;                 /* the live snapshot        */
static atomic_uint_least64_t g_seq = 0;            /* even = stable, odd = writing */
static pthread_mutex_t    g_write_lock = PTHREAD_MUTEX_INITIALIZER;

static pthread_t          g_thread;
static atomic_bool        g_running = false;
static atomic_bool        g_stop = false;

/* ================================================================
 * Reserve policy (pure helpers, env-overridable)
 * ================================================================ */

/* Parse a byte count from an env var; returns 0 (meaning "unset") if absent or
 * unparseable. */
static uint64_t env_bytes(const char *name)
{
    const char *v = getenv(name);
    if (!v || !*v)
        return 0;
    char *end = NULL;
    unsigned long long parsed = strtoull(v, &end, 10);
    if (end == v)
        return 0;
    return (uint64_t)parsed;
}

uint64_t pn_host_reserve_bytes(uint64_t host_total_bytes)
{
    uint64_t override = env_bytes("PN_HOST_RESERVE_BYTES");
    if (override)
        return override;

    uint64_t frac = (uint64_t)(PN_HOST_RESERVE_FRACTION * (double)host_total_bytes);
    return frac > PN_HOST_RESERVE_MIN_BYTES ? frac : PN_HOST_RESERVE_MIN_BYTES;
}

uint64_t pn_die_reserve_bytes(void)
{
    uint64_t override = env_bytes("PN_DIE_RESERVE_BYTES");
    return override ? override : PN_DIE_RESERVE_BYTES;
}

uint64_t pn_die_load_margin_bytes(void)
{
    uint64_t override = env_bytes("PN_DIE_LOAD_MARGIN_BYTES");
    return override ? override : PN_DIE_LOAD_MARGIN_BYTES;
}

/* Read a positive percentage from an env var, falling back to `dflt`. A zero,
 * negative or unparseable value falls back — a zero threshold would re-arm the
 * any-non-zero-PSI-is-pressure behavior these thresholds exist to remove. */
static double env_pct(const char *name, double dflt)
{
    const char *v = getenv(name);
    if (v && *v) {
        char *end = NULL;
        double parsed = strtod(v, &end);
        if (end != v && parsed > 0.0)
            return parsed;
    }
    return dflt;
}

double pn_psi_full_reject_pct(void)
{
    return env_pct("PN_PSI_FULL_REJECT", PN_PSI_FULL_REJECT_DEFAULT);
}

double pn_psi_full_tight_pct(void)
{
    return env_pct("PN_PSI_FULL_TIGHT", PN_PSI_FULL_TIGHT_DEFAULT);
}

double pn_psi_full_watch_pct(void)
{
    return env_pct("PN_PSI_FULL_WATCH", PN_PSI_FULL_WATCH_DEFAULT);
}

double pn_psi_some_tight_pct(void)
{
    return env_pct("PN_PSI_SOME_TIGHT", PN_PSI_SOME_TIGHT_DEFAULT);
}

double pn_psi_some_watch_pct(void)
{
    return env_pct("PN_PSI_SOME_WATCH", PN_PSI_SOME_WATCH_DEFAULT);
}

int pn_psi_trip_polls(void)
{
    const char *v = getenv("PN_PSI_TRIP_POLLS");
    if (v && *v) {
        char *end = NULL;
        long parsed = strtol(v, &end, 10);
        if (end != v && parsed >= 1)
            return (int)parsed;
    }
    return PN_PSI_TRIP_POLLS_DEFAULT;
}

const char *pn_mem_level_name(pn_mem_level level)
{
    switch (level) {
    case PN_MEM_LEVEL_OK:       return "OK";
    case PN_MEM_LEVEL_WATCH:    return "WATCH";
    case PN_MEM_LEVEL_TIGHT:    return "TIGHT";
    case PN_MEM_LEVEL_CRITICAL: return "CRITICAL";
    }
    return "?";
}

/* ================================================================
 * Level derivation
 * ================================================================ */

/* Host BYTE pressure from the reserve line alone. A fact, not a prediction:
 * the reserve is off-limits, so escalate as avail approaches it. */
static pn_mem_level host_byte_level(const pn_mem_state *s)
{
    uint64_t reserve = pn_host_reserve_bytes(s->host_total_bytes);

    if (s->host_avail_bytes <= reserve)
        return PN_MEM_LEVEL_CRITICAL;

    uint64_t slack = s->host_avail_bytes - reserve;
    /* WATCH when the slack above the reserve dips under one reserve's worth;
     * TIGHT when it dips under a quarter. Coarse but monotonic. */
    if (slack < reserve / 4)
        return PN_MEM_LEVEL_TIGHT;
    if (slack < reserve)
        return PN_MEM_LEVEL_WATCH;
    return PN_MEM_LEVEL_OK;
}

/* Worst per-die level: escalate as a die's free VRAM approaches exhaustion. A
 * die with no reported budget (0) is skipped. */
static pn_mem_level die_level(const pn_mem_state *s)
{
    pn_mem_level worst = PN_MEM_LEVEL_OK;
    for (int i = 0; i < s->n_dies && i < PN_MEM_MAX_DIES; i++) {
        uint64_t budget = s->per_die_budget[i];
        if (budget == 0)
            continue;
        uint64_t free = s->per_die_free[i];
        pn_mem_level lvl = PN_MEM_LEVEL_OK;
        if (free == 0)
            lvl = PN_MEM_LEVEL_CRITICAL;
        else if (free < budget / 16)
            lvl = PN_MEM_LEVEL_TIGHT;
        else if (free < budget / 8)
            lvl = PN_MEM_LEVEL_WATCH;
        if (lvl > worst)
            worst = lvl;
    }
    return worst;
}

pn_mem_level pn_mem_hard_level(const pn_mem_state *s)
{
    if (!s)
        return PN_MEM_LEVEL_OK;
    pn_mem_level h = host_byte_level(s);
    pn_mem_level d = die_level(s);
    return h > d ? h : d;
}

/* What the PSI stall counters predict, on their own. Proportionate: PSI avg10
 * is a decaying 10 s average that sits slightly above zero for ~a minute after
 * any brief reclaim, so a reading has to clear a noise floor before it means
 * anything. See the threshold block in pn_memwatch.h for the measurements. */
pn_mem_level pn_mem_psi_level(const pn_mem_state *s)
{
    if (!s)
        return PN_MEM_LEVEL_OK;

    pn_mem_level lvl = PN_MEM_LEVEL_OK;

    if (s->host_psi_full_avg10 >= pn_psi_full_reject_pct())
        lvl = PN_MEM_LEVEL_CRITICAL;
    else if (s->host_psi_full_avg10 >= pn_psi_full_tight_pct())
        lvl = PN_MEM_LEVEL_TIGHT;
    else if (s->host_psi_full_avg10 >= pn_psi_full_watch_pct())
        lvl = PN_MEM_LEVEL_WATCH;

    if (s->host_psi_some_avg10 >= pn_psi_some_tight_pct() &&
        lvl < PN_MEM_LEVEL_TIGHT)
        lvl = PN_MEM_LEVEL_TIGHT;
    else if (s->host_psi_some_avg10 >= pn_psi_some_watch_pct() &&
             lvl < PN_MEM_LEVEL_WATCH)
        lvl = PN_MEM_LEVEL_WATCH;

    return lvl;
}

void pn_psi_dwell_update(pn_psi_dwell *d, pn_mem_level raw)
{
    if (!d)
        return;

    /* Genuine thrash: publish on the first sample, never delay protection. */
    if (raw >= PN_MEM_LEVEL_CRITICAL) {
        d->trips = pn_psi_trip_polls();
        d->effective = raw;
        return;
    }

    /* Back to quiet: clear the counter and relax immediately (relax-fast is
     * what lets a clamped window re-widen the moment the box is healthy). */
    if (raw <= PN_MEM_LEVEL_OK) {
        d->trips = 0;
        d->effective = PN_MEM_LEVEL_OK;
        return;
    }

    /* Elevated but sub-CRITICAL: it must HOLD before it engages a rung. */
    int need = pn_psi_trip_polls();
    if (d->trips < need)
        d->trips++;
    if (d->trips >= need)
        d->effective = raw;      /* sustained — and follows raw down too */
}

/* The dwell filter's state. Only the host poll advances it (PSI is a host
 * signal sampled on that loop's cadence); the die feed republishes with
 * whatever the filter last settled on. Guarded by g_write_lock. */
static pn_psi_dwell g_psi_dwell = { 0, PN_MEM_LEVEL_OK };

pn_mem_level pn_mem_combine_level(pn_mem_level hard, pn_mem_level psi)
{
    /* PSI is a LEADING indicator of a squeeze — it is not evidence of one.
     * Streaming 17 GB of GGUF off disk to fill eight dies drives full-PSI to
     * ~6% for several seconds (measured on a K80 host during a 3-model load) while
     * MemAvailable never leaves 6.3 GB of 7.8 GB — the box is dropping page
     * cache it does not need, which is the kernel working correctly, not the
     * box running out of memory.
     *
     * So below the whole-box-thrash line, PSI on its own may raise the level to
     * WATCH — which engages only rung 4a (force Q4 KV, near-zero cost) — but no
     * further unless the byte math corroborates. The moment host headroom or a
     * die actually shows a squeeze (hard > OK), PSI is trusted in full and
     * escalates on top of it, which is exactly what a leading indicator is for.
     *
     * A PSI-CRITICAL reading (>= the reject line) is NEVER capped: at that point
     * the whole box is stalled on memory and MemAvailable's opinion is
     * irrelevant. */
    if (psi > PN_MEM_LEVEL_WATCH && psi < PN_MEM_LEVEL_CRITICAL &&
        hard <= PN_MEM_LEVEL_OK)
        psi = PN_MEM_LEVEL_WATCH;

    return hard > psi ? hard : psi;
}

static pn_mem_level derive_level(const pn_mem_state *s)
{
    return pn_mem_combine_level(pn_mem_hard_level(s), g_psi_dwell.effective);
}

/* ================================================================
 * Publish (seqlock write) + logging
 * ================================================================ */

/* Publish an updated snapshot: bump seq odd, copy in, bump seq even. Callers
 * hold g_write_lock. Logs when the derived level changes. */
static void publish_locked(const pn_mem_state *next)
{
    pn_mem_level prev_level = g_state.level;
    uint64_t next_seq = g_state.seq + 1;

    atomic_fetch_add_explicit(&g_seq, 1, memory_order_release); /* -> odd */

    g_state = *next;
    g_state.seq = next_seq;

    atomic_fetch_add_explicit(&g_seq, 1, memory_order_release); /* -> even */

    if (g_state.level != prev_level) {
        /* Name the DRIVING signal, not just the numbers. Phase 10 cost a
         * diagnosis round-trip precisely because the old line printed host
         * avail + PSI side by side and left the reader to infer which one
         * escalated — and the answer (PSI) was the counter-intuitive one. */
        pn_mem_level hard = pn_mem_hard_level(&g_state);
        pn_mem_level byte_lvl = host_byte_level(&g_state);
        pn_mem_level die_lvl  = die_level(&g_state);
        const char *driver = "host-bytes";
        if (g_state.level <= PN_MEM_LEVEL_OK)
            driver = "none";            /* de-escalated to OK: nothing drives */
        else if (g_state.level > hard)
            driver = "psi";
        else if (die_lvl >= byte_lvl && die_lvl == hard)
            driver = "die-vram";

        fprintf(stderr,
                "[memwatch] level %s -> %s by=%s | host avail %llu MiB / %llu MiB "
                "(bytes=%s) psi(some %.1f full %.1f -> %s, dwell %d/%d) "
                "dies=%d (vram=%s)\n",
                pn_mem_level_name(prev_level), pn_mem_level_name(g_state.level),
                driver,
                (unsigned long long)(g_state.host_avail_bytes >> 20),
                (unsigned long long)(g_state.host_total_bytes >> 20),
                pn_mem_level_name(byte_lvl),
                g_state.host_psi_some_avg10, g_state.host_psi_full_avg10,
                pn_mem_level_name(g_psi_dwell.effective),
                g_psi_dwell.trips, pn_psi_trip_polls(),
                g_state.n_dies, pn_mem_level_name(die_lvl));
    }
}

void pn_memwatch_read(pn_mem_state *out)
{
    for (;;) {
        uint64_t s1 = atomic_load_explicit(&g_seq, memory_order_acquire);
        if (s1 & 1u)
            continue;                       /* a write is in progress — retry   */
        *out = g_state;
        uint64_t s2 = atomic_load_explicit(&g_seq, memory_order_acquire);
        if (s1 == s2)
            return;                         /* stable across the copy           */
    }
}

/* ================================================================
 * Host-signal poll: sample -> merge into state -> publish
 * ================================================================ */

static void poll_host_once(void)
{
    pn_host_sample sample;
    pn_host_signals_sample(&sample);

    pthread_mutex_lock(&g_write_lock);

    pn_mem_state next = g_state;            /* keep die numbers, refresh host   */
    next.host_total_bytes    = sample.total_bytes;
    next.host_avail_bytes    = sample.avail_bytes;
    next.host_psi_some_avg10 = sample.psi_some_avg10;
    next.host_psi_full_avg10 = sample.psi_full_avg10;
    next.cgroup_current_bytes = sample.cgroup_current;
    next.cgroup_high_bytes    = sample.cgroup_high;
    next.cgroup_max_bytes     = sample.cgroup_max;

    /* Advance the PSI dwell filter on this sample BEFORE deriving the level:
     * a sub-CRITICAL PSI escalation only counts once it has held for
     * pn_psi_trip_polls() consecutive polls. */
    pn_psi_dwell_update(&g_psi_dwell, pn_mem_psi_level(&next));

    next.level = derive_level(&next);

    publish_locked(&next);

    pthread_mutex_unlock(&g_write_lock);
}

void pn_memwatch_poll_now(void)
{
    poll_host_once();
}

void pn_memwatch_set_die(int die_id, uint64_t free_bytes,
                         uint64_t budget_bytes, uint64_t used_bytes)
{
    if (die_id < 0 || die_id >= PN_MEM_MAX_DIES)
        return;

    pthread_mutex_lock(&g_write_lock);

    pn_mem_state next = g_state;
    next.per_die_free[die_id]   = free_bytes;
    next.per_die_budget[die_id] = budget_bytes;
    next.per_die_used[die_id]   = used_bytes;
    if (die_id + 1 > next.n_dies)
        next.n_dies = die_id + 1;
    next.level = derive_level(&next);

    publish_locked(&next);

    pthread_mutex_unlock(&g_write_lock);
}

/* ================================================================
 * Poll thread
 * ================================================================ */

static void *memwatch_thread(void *arg)
{
    (void)arg;
    while (!atomic_load_explicit(&g_stop, memory_order_acquire)) {
        poll_host_once();
        struct timespec ts = {
            .tv_sec  = PN_MEMWATCH_POLL_MS / 1000,
            .tv_nsec = (long)(PN_MEMWATCH_POLL_MS % 1000) * 1000000L,
        };
        nanosleep(&ts, NULL);
    }
    return NULL;
}

int pn_memwatch_start(void)
{
    bool expected = false;
    if (!atomic_compare_exchange_strong(&g_running, &expected, true))
        return 0;                           /* already running — idempotent     */

    atomic_store(&g_stop, false);

    /* Seed one synchronous poll so readers see real numbers immediately. */
    poll_host_once();

    if (pthread_create(&g_thread, NULL, memwatch_thread, NULL) != 0) {
        atomic_store(&g_running, false);
        fprintf(stderr, "[memwatch] failed to start poll thread\n");
        return -1;
    }

    fprintf(stderr, "[memwatch] poll thread started (%d ms)\n", PN_MEMWATCH_POLL_MS);
    return 0;
}

void pn_memwatch_stop(void)
{
    if (!atomic_load(&g_running))
        return;
    atomic_store(&g_stop, true);
    pthread_join(g_thread, NULL);
    atomic_store(&g_running, false);
}

bool pn_memwatch_running(void)
{
    return atomic_load(&g_running);
}
