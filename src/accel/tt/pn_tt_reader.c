/* SPDX-License-Identifier: AGPL-3.0-or-later */
/* Copyright (C) 2026 David (Corey) Herrera */
/**
 * @file pn_tt_reader.c
 * @brief Implementation of the reader thread. See pn_tt_reader.h for why and what it may touch.
 *
 * @status Written for PHASE W (chain step).
 * @issues None
 * @todo None
 */

#ifdef USE_TT

#include "pn_tt_reader.h"

#include <pthread.h>
#include <sched.h>
#include <stdatomic.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "pn_tt_device_internal.h"

#include "tt/convert/ft_float_convert.h"
#include "tt/ft_tt_runtime_udrv.h"
#include "tt/ft_tt_queue.h"
#include "tt/ft_tt_tilize.h"

/** Jobs in flight: a layer posts two reads a band, 51 bands at 1,630 rows. */
#define READER_RING 256u
/** One job's staged tiles; alpha/beta are one tile, this covers a band of up to 8. */
#define READER_MAX_TILES 8u

typedef struct reader_job {
    uint16_t x, y;
    uint32_t address, tiles, rows, count;
    float *dst;
    pn_tt_reader_gate gate; /**< target 0 = none */
} reader_job;

/** One reader thread on its own window bank, with its own ring: jobs are dealt round-robin,
 *  so a ticket names its worker (`(ticket - 1) % workers`) and its place in that ring. */
typedef struct reader_worker {
    pthread_t thread;
    ft_tt_noc_access access;
    ft_tt_noc_access l1;   /**< the same bank, L1 addresses: the gate polls */
    bool has_l1;
    reader_job ring[READER_RING];
    _Atomic uint64_t posted;    /**< jobs written (producer) */
    _Atomic uint64_t done;      /**< jobs finished (consumer) */
    _Atomic uint64_t failed_at; /**< local index of the last failed job, 0 = none */
    _Atomic int *stop;
    _Atomic int *cancel;   /**< gates fail at once (an abandon: their work may never run) */
    /** IDLE SLEEPS, never spins: a spinning reader took a core from decode on this 2-core
     *  host (dec48 38.7 -> 36.3 tok/s with two of them, run_w_stream2.sh). */
    pthread_mutex_t lock;
    pthread_cond_t wake;
    _Atomic int sleeping;
    double read_seconds;
    uint64_t reads;
    unsigned char stage[READER_MAX_TILES * PN_TT_TILE_BF16];
} reader_worker;

/** `PN_TT_READER_THREADS` (default 1: two did not read faster — the link serialises PIO
 *  reads — and cost decode, run_w_stream2.sh; at most the udrv reader banks opened). */
#define READER_MAX_WORKERS 4u

struct pn_tt_reader {
    reader_worker worker[READER_MAX_WORKERS];
    uint32_t workers;
    uint64_t tickets;           /**< posted over all workers (producer) */
    _Atomic int stop;
    _Atomic int cancel;
    double wait_seconds;        /**< producer side */
};

/** Poll the gate's completion word until it reaches its target (5 s bound). */
static int reader_gate_wait(reader_worker *w, const pn_tt_reader_gate *gate)
{
    const double deadline = pn_tt_now_seconds() + 5.0;
    uint32_t line[4]; /* fields 4..7, the 16-byte completion line ft_tt_queue reads */
    uint32_t count;

    if (gate->target == 0u)
        return 0;
    if (!w->has_l1)
        return -1;
    for (;;) {
        if (w->l1.read(w->l1.user, gate->x, gate->y,
                       gate->control_address + FT_TT_QUEUE_FIELD_STOP_REQUEST * 4u, line,
                       sizeof(line)) != FT_TT_OK)
            return -1;
        count = line[FT_TT_QUEUE_FIELD_COMPLETED_COUNT - FT_TT_QUEUE_FIELD_STOP_REQUEST];
        /* Counts only grow within a run; the difference is wrap-safe. */
        if ((int32_t)(count - gate->target) >= 0)
            return 0;
        if (pn_tt_now_seconds() > deadline ||
            atomic_load_explicit(w->cancel, memory_order_relaxed))
            return -1;
    }
}

static int reader_run_job(reader_worker *w, const reader_job *job)
{
    uint32_t tile, row, column;

    if (reader_gate_wait(w, &job->gate) != 0)
        return -1;
    if (job->rows <= 16u) {
        /* rows <= 16 live in faces 0-1: two 64 B-rounded runs a tile (PN_TT_ROWS_FACES's
         * shape, pn_tt_activation_download_rows()) instead of the whole 2 KiB. */
        const size_t run = ((size_t)job->rows * 32u + 63u) & ~(size_t)63u;

        for (tile = 0; tile < job->tiles; tile++)
            if (w->access.read(w->access.user, job->x, job->y,
                               job->address + tile * PN_TT_TILE_BF16,
                               w->stage + (size_t)tile * PN_TT_TILE_BF16, run) != FT_TT_OK ||
                w->access.read(w->access.user, job->x, job->y,
                               job->address + tile * PN_TT_TILE_BF16 + 512u,
                               w->stage + (size_t)tile * PN_TT_TILE_BF16 + 512u, run) != FT_TT_OK)
                return -1;
    } else if (w->access.read(w->access.user, job->x, job->y, job->address, w->stage,
                              (size_t)job->tiles * PN_TT_TILE_BF16) != FT_TT_OK) {
        return -1;
    }
    for (tile = 0; tile < job->tiles; tile++) {
        const uint16_t *base = (const uint16_t *)w->stage + (size_t)tile * (PN_TT_TILE_BF16 / 2u);

        for (row = 0; row < job->rows; row++)
            for (column = 0; column < PN_TT_TILE_DIM; column++)
                job->dst[(size_t)row * job->count + tile * PN_TT_TILE_DIM + column] =
                    ft_float_from_bf16(base[ft_tt_tile_face_index(row, column)]);
    }
    return 0;
}

static void *reader_main(void *arg)
{
    reader_worker *w = (reader_worker *)arg;

    for (;;) {
        const uint64_t done = atomic_load_explicit(&w->done, memory_order_relaxed);

        if (done == atomic_load_explicit(&w->posted, memory_order_acquire)) {
            if (atomic_load_explicit(w->stop, memory_order_acquire))
                break;
            pthread_mutex_lock(&w->lock);
            atomic_store_explicit(&w->sleeping, 1, memory_order_seq_cst);
            while (done == atomic_load_explicit(&w->posted, memory_order_seq_cst) &&
                   !atomic_load_explicit(w->stop, memory_order_seq_cst))
                pthread_cond_wait(&w->wake, &w->lock);
            atomic_store_explicit(&w->sleeping, 0, memory_order_seq_cst);
            pthread_mutex_unlock(&w->lock);
            continue;
        }
        {
            const double at = pn_tt_now_seconds();

            if (reader_run_job(w, &w->ring[done % READER_RING]) != 0)
                atomic_store_explicit(&w->failed_at, done + 1u, memory_order_relaxed);
            w->read_seconds += pn_tt_now_seconds() - at;
            w->reads++;
        }
        atomic_store_explicit(&w->done, done + 1u, memory_order_release);
    }
    return NULL;
}

/** Start up to `PN_TT_READER_THREADS` workers, one per reader bank that opened. */
static struct pn_tt_reader *reader_start(pn_tt_device *device)
{
    struct pn_tt_reader *r = calloc(1, sizeof(*r));
    const uint32_t want = pn_tt_env_u32("PN_TT_READER_THREADS", 1u);
    uint32_t i;

    if (!r || !device->runtime.udrv)
        return (free(r), NULL);
    for (i = 0; i < want && i < READER_MAX_WORKERS; i++) {
        reader_worker *w = &r->worker[i];

        if (ft_tt_runtime_udrv_reader_access(device->runtime.udrv, i, &w->access) != FT_TT_OK)
            break;
        w->has_l1 = ft_tt_runtime_udrv_reader_l1_access(device->runtime.udrv, i, &w->l1) ==
                    FT_TT_OK;
        w->stop = &r->stop;
        w->cancel = &r->cancel;
        pthread_mutex_init(&w->lock, NULL);
        pthread_cond_init(&w->wake, NULL);
        if (pthread_create(&w->thread, NULL, reader_main, w) != 0)
            break;
        r->workers++;
    }
    if (r->workers == 0u)
        return (free(r), NULL);
    return r;
}

bool pn_tt_reader_available(pn_tt_device *device)
{
    static int enabled = -1;

    if (!device)
        return false;
    if (enabled < 0) {
        const char *value = getenv("PN_TT_READER");

        enabled = (value && *value) ? (strcmp(value, "0") != 0) : 1;
    }
    if (!enabled || device->reader_failed)
        return false;
    if (device->reader)
        return true;
    device->reader = reader_start(device);
    if (!device->reader) {
        device->reader_failed = true;
        fprintf(stderr, "pn_tt: reader thread NOT started (no reader window bank) — strip reads "
                        "stay on the issuing thread\n");
        return false;
    }
    fprintf(stderr, "pn_tt: %u reader thread(s) started, each on its own window bank\n",
            device->reader->workers);
    return true;
}

uint64_t pn_tt_reader_post_rows(pn_tt_device *device, uint32_t bank, uint32_t address,
                                uint32_t tiles, uint32_t rows, float *dst, uint32_t count)
{
    return pn_tt_reader_post_rows_gated(device, bank, address, tiles, rows, dst, count, NULL);
}

uint64_t pn_tt_reader_post_rows_gated(pn_tt_device *device, uint32_t bank, uint32_t address,
                                      uint32_t tiles, uint32_t rows, float *dst,
                                      uint32_t count, const pn_tt_reader_gate *gate)
{
    struct pn_tt_reader *r;
    reader_worker *w;
    uint64_t posted;
    reader_job *job;

    if (!pn_tt_reader_available(device) || !dst || tiles == 0u || tiles > READER_MAX_TILES ||
        rows == 0u || rows > PN_TT_TILE_DIM || count < tiles * PN_TT_TILE_DIM ||
        bank >= device->dram_bank_count)
        return 0u;
    r = device->reader;
    w = &r->worker[r->tickets % r->workers];
    posted = atomic_load_explicit(&w->posted, memory_order_relaxed);
    /* A full ring waits for a slot rather than overwriting one the thread has not read. */
    while (posted - atomic_load_explicit(&w->done, memory_order_acquire) >= READER_RING)
        sched_yield();
    job = &w->ring[posted % READER_RING];
    job->x = (uint16_t)device->dram[bank].x;
    job->y = (uint16_t)device->dram[bank].y;
    job->address = address;
    job->tiles = tiles;
    job->rows = rows;
    job->count = count;
    job->dst = dst;
    if (gate)
        job->gate = *gate;
    else
        memset(&job->gate, 0, sizeof(job->gate));
    atomic_store_explicit(&w->posted, posted + 1u, memory_order_seq_cst);
    if (atomic_load_explicit(&w->sleeping, memory_order_seq_cst)) {
        pthread_mutex_lock(&w->lock);
        pthread_cond_signal(&w->wake);
        pthread_mutex_unlock(&w->lock);
    }
    return ++r->tickets;
}

int pn_tt_reader_wait(pn_tt_device *device, uint64_t ticket)
{
    struct pn_tt_reader *r = device ? device->reader : NULL;
    reader_worker *w;
    uint64_t local;
    double at;

    if (!r || ticket == 0u || ticket > r->tickets)
        return -1;
    w = &r->worker[(ticket - 1u) % r->workers];
    local = (ticket - 1u) / r->workers + 1u;
    at = pn_tt_now_seconds();
    while (atomic_load_explicit(&w->done, memory_order_acquire) < local)
        sched_yield();
    r->wait_seconds += pn_tt_now_seconds() - at;
    /* failed_at only ever names the LAST failure; any failure at or after this job's place
     * is treated as this one's, which errs toward re-reading inline. */
    return atomic_load_explicit(&w->failed_at, memory_order_relaxed) >= local ? -1 : 0;
}

void pn_tt_reader_drain_cancelled(pn_tt_device *device)
{
    struct pn_tt_reader *r = device ? device->reader : NULL;

    if (!r)
        return;
    atomic_store_explicit(&r->cancel, 1, memory_order_seq_cst);
    pn_tt_reader_drain(device);
    atomic_store_explicit(&r->cancel, 0, memory_order_seq_cst);
}

void pn_tt_reader_drain(pn_tt_device *device)
{
    struct pn_tt_reader *r = device ? device->reader : NULL;
    uint64_t t;

    if (!r)
        return;
    for (t = r->tickets; t > 0u && t + r->workers > r->tickets; t--)
        (void)pn_tt_reader_wait(device, t);
}

void pn_tt_reader_close(pn_tt_device *device)
{
    struct pn_tt_reader *r = device ? device->reader : NULL;
    double read_seconds = 0.0;
    uint64_t reads = 0u;
    uint32_t i;

    if (!r)
        return;
    atomic_store_explicit(&r->stop, 1, memory_order_seq_cst);
    for (i = 0; i < r->workers; i++) {
        pthread_mutex_lock(&r->worker[i].lock);
        pthread_cond_signal(&r->worker[i].wake);
        pthread_mutex_unlock(&r->worker[i].lock);
        pthread_join(r->worker[i].thread, NULL);
        pthread_cond_destroy(&r->worker[i].wake);
        pthread_mutex_destroy(&r->worker[i].lock);
        read_seconds += r->worker[i].read_seconds;
        reads += r->worker[i].reads;
    }
    if (reads)
        fprintf(stderr, "pn_tt: reader — %u thread(s), %llu reads, %.3f s reading (summed), "
                        "%.3f s waited on\n",
                r->workers, (unsigned long long)reads, read_seconds, r->wait_seconds);
    free(r);
    device->reader = NULL;
}

void pn_tt_reader_stats(const pn_tt_device *device, double *out_wait_seconds,
                        uint64_t *out_reads, double *out_read_seconds)
{
    const struct pn_tt_reader *r = device ? device->reader : NULL;
    uint32_t i;

    if (out_wait_seconds)
        *out_wait_seconds = r ? r->wait_seconds : 0.0;
    if (out_reads)
        *out_reads = 0u;
    if (out_read_seconds)
        *out_read_seconds = 0.0;
    for (i = 0; r && i < r->workers; i++) {
        if (out_reads)
            *out_reads += r->worker[i].reads;
        if (out_read_seconds)
            *out_read_seconds += r->worker[i].read_seconds;
    }
}

#endif /* USE_TT */
