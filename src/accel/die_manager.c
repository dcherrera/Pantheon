/* SPDX-License-Identifier: AGPL-3.0-or-later */
/* Copyright (C) 2026 David (Corey) Herrera */
/**
 * @file die_manager.c
 * @brief Implementation of the die manager (see die_manager.h). One
 *        process-global table (dies + bindings) behind a mutex, mirroring the
 *        singleton style of pn_memwatch. Placement policy detail lives in
 *        pn_dm_place below; everything else is straight accounting.
 *
 * @status Logical path tested on the Mac; CUDA and TT probes gated on GPU hardware.
 * @issues None
 * @todo None
 */

#include "die_manager.h"

#include <pthread.h>
#include <stdio.h>
#include <string.h>

#ifdef USE_CUDA
#include "cuda/fc_cuda.h"
#include "../core/mem/pn_memwatch.h"
#endif

#ifdef USE_TT
#include "tt/ft_tt_coord.h"
#include "tt/ft_tt_device.h"
#include "tt/ft_tt_runtime.h"
#include "../core/mem/pn_memwatch.h"
#endif

/* One resident model's reservation: which dies it holds and how many bytes it
 * put on each (total split evenly across the set). */
typedef struct {
    bool     used;
    char     name[PN_DM_NAME_MAX];
    int      dies[PN_DM_MAX_DIES];
    int      n_dies;
    uint64_t bytes_per_die;
    uint64_t serial;             /* names this binding in a pn_dm_binding_ref */
} dm_binding;

typedef struct {
    bool     present;
    uint64_t vram_budget;
    uint64_t vram_used;
} dm_die;

static struct {
    pthread_mutex_t lock;
    bool            running;
    bool            hardware;   /* dies probed from CUDA / Tenstorrent, not logical */
    int             n_dies;
    dm_die          die[PN_DM_MAX_DIES];
    dm_binding      binding[PN_DM_MAX_BINDINGS];
    uint64_t        next_serial; /* never reset, so a stale ref never matches */
} g_dm = { .lock = PTHREAD_MUTEX_INITIALIZER };

/* --- init / shutdown --------------------------------------------------- */

int pn_dm_init_logical(int n_dies, uint64_t budget_bytes)
{
    if (n_dies <= 0 || n_dies > PN_DM_MAX_DIES || budget_bytes == 0)
        return -1;
    pthread_mutex_lock(&g_dm.lock);
    if (g_dm.running) {                     /* idempotent while running */
        pthread_mutex_unlock(&g_dm.lock);
        return 0;
    }
    memset(g_dm.die, 0, sizeof(g_dm.die));
    memset(g_dm.binding, 0, sizeof(g_dm.binding));
    g_dm.n_dies = n_dies;
    for (int i = 0; i < n_dies; i++) {
        g_dm.die[i].present = true;
        g_dm.die[i].vram_budget = budget_bytes;
    }
    g_dm.hardware = false;
    g_dm.running = true;
    pthread_mutex_unlock(&g_dm.lock);
    return 0;
}

#ifdef USE_CUDA
int pn_dm_init_cuda(int n_dies)
{
    if (n_dies <= 0 || n_dies > PN_DM_MAX_DIES)
        return -1;
    /* Driver-only gate on purpose: this runs at serve startup, before any
     * model has a placement, so it must not be the thing that forces a
     * process-default context onto whatever die happens to be ordinal 0. */
    if (!fc_driver_available())
        return -1;
    pthread_mutex_lock(&g_dm.lock);
    if (g_dm.running) {
        int n = g_dm.n_dies;
        pthread_mutex_unlock(&g_dm.lock);
        return n;
    }
    memset(g_dm.die, 0, sizeof(g_dm.die));
    memset(g_dm.binding, 0, sizeof(g_dm.binding));
    int found = 0;
    uint64_t reserve = pn_die_reserve_bytes();
    for (int i = 0; i < n_dies; i++) {
        /* Transient bare context per die — created, read, destroyed. Probing
         * with fc_cuda_init_device() stranded a full context (cuBLAS + 12 JIT'd
         * PTX modules, ~124 MiB) on EVERY die the serve enumerated, including
         * dies owned by another serve. A probe must leave no trace. */
        size_t die_free = 0, die_total = 0;
        if (fc_cuda_probe_device(i, &die_free, &die_total) != 0 || die_total == 0)
            break;                      /* ordinals are contiguous — stop */
        uint64_t budget = die_total > reserve ? die_total - reserve : 0;
        g_dm.die[i].present = true;
        g_dm.die[i].vram_budget = budget;
        /* Anything already gone from `free` beyond the reserve is occupancy we
         * did not create (another process / stranded VRAM) — count it so
         * placement doesn't over-commit the die. */
        uint64_t external = (die_total - die_free) > reserve
                          ? (die_total - die_free) - reserve : 0;
        g_dm.die[i].vram_used = external < budget ? external : budget;
        /* 0 bytes ours: the probe context is already destroyed, and no model
         * has been placed on this die yet. */
        if (pn_memwatch_running())
            pn_memwatch_set_die(i, die_free, budget, 0);
        found++;
    }
    if (found == 0) {
        pthread_mutex_unlock(&g_dm.lock);
        return -1;
    }
    g_dm.n_dies = found;
    g_dm.hardware = true;
    g_dm.running = true;
    pthread_mutex_unlock(&g_dm.lock);
    fprintf(stderr, "die_manager: tracking %d CUDA die%s\n",
            found, found == 1 ? "" : "s");
    return found;
}
#endif /* USE_CUDA */

#ifdef USE_TT
bool pn_dm_tt_driver_explicit(void)
{
    ft_tt_runtime_driver driver = FT_TT_RUNTIME_DRIVER_KMD;
    bool explicit_choice = false;

    /* A REFUSED value (neither kmd nor udrv) counts as explicit: somebody asked
     * for a transport and did not get one, which must not read as "no card". */
    if (ft_tt_runtime_driver_select(&driver, &explicit_choice) != FT_TT_OK)
        return true;
    return explicit_choice;
}

/* Capacity of one Blackhole device, in bytes, or 0 if it cannot be measured.
 *
 * The probe leaves no trace: it opens the device, reads the grid and closes
 * again. That matters for the same reason it did on CUDA — this runs at serve
 * startup, before any model has a placement, and a probe that held a context
 * or a TLB window would be holding it against whichever job takes the card.
 *
 * It goes through ft_tt_runtime_open() rather than ft_tt_init_device() so it
 * probes over WHICHEVER TRANSPORT the process is configured for. Opening the
 * chardev here would have refused a machine with no vendor module before the
 * serve ever reached the seam — which is exactly the defect this round exists
 * to remove, one layer up from where it was first seen.
 *
 * The capacity is NOT a constant and is deliberately not defaulted. Bank count
 * comes from ARC telemetry and bank size from the architecture table selected
 * at open; ft_tt_runtime_dram_bytes() answers 0 rather than reporting the
 * nominal eight banks as if it had measured them, so a telemetry failure ends
 * the scan instead of registering a die with an invented budget. */
static uint64_t tt_probe_device_bytes(int device_index)
{
    ft_tt_runtime runtime;
    uint64_t total;

    if (ft_tt_runtime_open(device_index, &runtime) != FT_TT_OK)
        return 0;
    /* ft_tt_runtime_open() already refuses a topology ft_tt_coord cannot map,
     * on both transports, so a card we would not dare address never gets here
     * (spec §7.3). */
    total = ft_tt_runtime_dram_bytes(&runtime);
    ft_tt_runtime_close(&runtime);
    return total;
}

int pn_dm_init_tt(int n_devices)
{
    if (n_devices <= 0 || n_devices > PN_DM_MAX_DIES)
        return -1;
    pthread_mutex_lock(&g_dm.lock);
    if (g_dm.running) {                     /* idempotent while running */
        int n = g_dm.n_dies;
        pthread_mutex_unlock(&g_dm.lock);
        return n;
    }
    memset(g_dm.die, 0, sizeof(g_dm.die));
    memset(g_dm.binding, 0, sizeof(g_dm.binding));
    int found = 0;
    uint64_t reserve = pn_die_reserve_bytes();
    for (int i = 0; i < n_devices; i++) {
        uint64_t total = tt_probe_device_bytes(i);
        if (total == 0)
            break;                          /* ordinals are contiguous — stop */
        g_dm.die[found].present = true;
        g_dm.die[found].vram_budget = total > reserve ? total - reserve : 0;
        /* No external-occupancy term, unlike the CUDA probe: the card exposes
         * no cross-process allocation view, so `free` from a fresh context is
         * always `total`. A second process holding DRAM is invisible here and
         * is a reason to run one serve per card, not a number to fake. */
        g_dm.die[found].vram_used = 0;
        if (pn_memwatch_running())
            pn_memwatch_set_die(found, total, g_dm.die[found].vram_budget, 0);
        found++;
    }
    if (found == 0) {
        pthread_mutex_unlock(&g_dm.lock);
        return -1;
    }
    g_dm.n_dies = found;
    g_dm.hardware = true;
    g_dm.running = true;
    pthread_mutex_unlock(&g_dm.lock);
    fprintf(stderr, "die_manager: tracking %d Tenstorrent device%s (one die each)\n",
            found, found == 1 ? "" : "s");
    return found;
}
#endif /* USE_TT */

void pn_dm_shutdown(void)
{
    pthread_mutex_lock(&g_dm.lock);
    g_dm.running = false;
    g_dm.hardware = false;
    g_dm.n_dies = 0;
    memset(g_dm.die, 0, sizeof(g_dm.die));
    memset(g_dm.binding, 0, sizeof(g_dm.binding));
    pthread_mutex_unlock(&g_dm.lock);
}

/* --- queries ------------------------------------------------------------ */

bool pn_dm_is_hardware(void)
{
    pthread_mutex_lock(&g_dm.lock);
    bool hw = g_dm.running && g_dm.hardware;
    pthread_mutex_unlock(&g_dm.lock);
    return hw;
}

int pn_dm_count(void)
{
    pthread_mutex_lock(&g_dm.lock);
    int n = g_dm.running ? g_dm.n_dies : 0;
    pthread_mutex_unlock(&g_dm.lock);
    return n;
}

/* Locked helper: does `b` cover die `id`? */
static bool binding_has_die(const dm_binding *b, int id)
{
    for (int i = 0; i < b->n_dies; i++)
        if (b->dies[i] == id)
            return true;
    return false;
}

/* Locked helper: bytes the bindings hold on die `id`. */
static uint64_t die_bound_bytes(int id)
{
    uint64_t bound = 0;
    for (int i = 0; i < PN_DM_MAX_BINDINGS; i++)
        if (g_dm.binding[i].used && binding_has_die(&g_dm.binding[i], id))
            bound += g_dm.binding[i].bytes_per_die;
    return bound;
}

int pn_dm_get(int die_id, pn_dm_die *out)
{
    if (!out)
        return -1;
    pthread_mutex_lock(&g_dm.lock);
    if (!g_dm.running || die_id < 0 || die_id >= g_dm.n_dies ||
        !g_dm.die[die_id].present) {
        pthread_mutex_unlock(&g_dm.lock);
        return -1;
    }
    memset(out, 0, sizeof(*out));
    out->id = die_id;
    out->vram_budget = g_dm.die[die_id].vram_budget;
    out->vram_used = g_dm.die[die_id].vram_used;
    out->vram_bound = die_bound_bytes(die_id);
    for (int i = 0; i < PN_DM_MAX_BINDINGS; i++) {
        if (!g_dm.binding[i].used || !binding_has_die(&g_dm.binding[i], die_id))
            continue;
        if (out->n_models == 0) {
            strncpy(out->model_name, g_dm.binding[i].name,
                    PN_DM_NAME_MAX - 1);
            out->model_name[PN_DM_NAME_MAX - 1] = '\0';
        }
        out->n_models++;
    }
    pthread_mutex_unlock(&g_dm.lock);
    return 0;
}

int pn_dm_note_external(int die_id, uint64_t external)
{
    pthread_mutex_lock(&g_dm.lock);
    if (!g_dm.running || die_id < 0 || die_id >= g_dm.n_dies ||
        !g_dm.die[die_id].present) {
        pthread_mutex_unlock(&g_dm.lock);
        return -1;
    }
    dm_die *d = &g_dm.die[die_id];
    uint64_t bound = die_bound_bytes(die_id);
    uint64_t room = d->vram_budget > bound ? d->vram_budget - bound : 0;
    d->vram_used = bound + (external < room ? external : room);
    pthread_mutex_unlock(&g_dm.lock);
    return 0;
}

/* --- placement ----------------------------------------------------------- */

/* Locked helpers. */
static uint64_t die_free_bytes(int id)
{
    const dm_die *d = &g_dm.die[id];
    return d->vram_budget > d->vram_used ? d->vram_budget - d->vram_used : 0;
}

static bool die_occupied(int id)
{
    return g_dm.die[id].vram_used > 0;
}

/* The card mate of a die under the fixed K80 pairing (2i <-> 2i+1); -1 when
 * the mate is outside the tracked range. */
static int die_mate(int id)
{
    int mate = (id % 2 == 0) ? id + 1 : id - 1;
    return (mate >= 0 && mate < g_dm.n_dies) ? mate : -1;
}

int pn_dm_place(uint64_t need_bytes, int n_dies_wanted, int *dies_out)
{
    if (!dies_out || n_dies_wanted < 1 || n_dies_wanted > 2)
        return -1;   /* >2-die pipeline placement arrives with Phase 6.1 */

    pthread_mutex_lock(&g_dm.lock);
    if (!g_dm.running) {
        pthread_mutex_unlock(&g_dm.lock);
        return -1;
    }
    uint64_t per_die = need_bytes / (uint64_t)n_dies_wanted;
    int rc = -1;

    if (n_dies_wanted == 2) {
        /* Same-card pair with room on both halves. Prefer a fully-free pair;
         * tie-break on the most free space (least loaded). */
        int best = -1;
        bool best_free = false;
        uint64_t best_room = 0;
        for (int a = 0; a + 1 < g_dm.n_dies; a += 2) {
            int b = a + 1;
            if (!g_dm.die[a].present || !g_dm.die[b].present)
                continue;
            if (die_free_bytes(a) < per_die || die_free_bytes(b) < per_die)
                continue;
            bool pair_free = !die_occupied(a) && !die_occupied(b);
            uint64_t room = die_free_bytes(a) + die_free_bytes(b);
            if (best < 0 || (pair_free && !best_free) ||
                (pair_free == best_free && room > best_room)) {
                best = a;
                best_free = pair_free;
                best_room = room;
            }
        }
        if (best >= 0) {
            dies_out[0] = best;
            dies_out[1] = best + 1;
            rc = 0;
        }
    } else {
        /* Single die: pack onto a card whose mate is already occupied so free
         * same-card pairs stay available for 2-die models; among candidates
         * take the least-loaded (most free). */
        int best = -1;
        bool best_packed = false;
        uint64_t best_room = 0;
        for (int i = 0; i < g_dm.n_dies; i++) {
            if (!g_dm.die[i].present || die_free_bytes(i) < per_die)
                continue;
            int mate = die_mate(i);
            bool packed = (mate >= 0 && die_occupied(mate)) || die_occupied(i);
            uint64_t room = die_free_bytes(i);
            if (best < 0 || (packed && !best_packed) ||
                (packed == best_packed && room > best_room)) {
                best = i;
                best_packed = packed;
                best_room = room;
            }
        }
        if (best >= 0) {
            dies_out[0] = best;
            rc = 0;
        }
    }
    pthread_mutex_unlock(&g_dm.lock);
    return rc;
}

/* --- bind / release ------------------------------------------------------ */

int pn_dm_bind(const int *dies, int n, const char *model_name,
               uint64_t total_bytes)
{
    return pn_dm_bind_ref(dies, n, model_name, total_bytes, NULL);
}

int pn_dm_bind_ref(const int *dies, int n, const char *model_name,
                   uint64_t total_bytes, pn_dm_binding_ref *out)
{
    if (out)
        *out = PN_DM_BINDING_NONE;
    if (!dies || n < 1 || n > PN_DM_MAX_DIES || !model_name || !*model_name)
        return -1;

    pthread_mutex_lock(&g_dm.lock);
    if (!g_dm.running)
        goto refuse;

    /* Same-name multi-card placement: one model id may be bound to several
     * DISJOINT die-sets (one binding per card) for load-balanced serving.
     * Overlapping dies are still rejected by the per-die room check below, so
     * here we only need a free binding slot — no duplicate-name refusal. Each
     * placement gets its own binding, so a name does not say WHICH one: the
     * registry releases by the ref this returns (pn_dm_release_ref). */
    dm_binding *slot = NULL;
    for (int i = 0; i < PN_DM_MAX_BINDINGS; i++) {
        if (!g_dm.binding[i].used) { slot = &g_dm.binding[i]; break; }
    }
    if (!slot)
        goto refuse;                             /* binding table full */

    uint64_t per_die = total_bytes / (uint64_t)n;
    for (int i = 0; i < n; i++) {
        int id = dies[i];
        if (id < 0 || id >= g_dm.n_dies || !g_dm.die[id].present ||
            die_free_bytes(id) < per_die)
            goto refuse;                         /* unknown die / no room */
    }

    slot->used = true;
    strncpy(slot->name, model_name, PN_DM_NAME_MAX - 1);
    slot->name[PN_DM_NAME_MAX - 1] = '\0';
    slot->n_dies = n;
    slot->bytes_per_die = per_die;
    slot->serial = ++g_dm.next_serial;
    for (int i = 0; i < n; i++) {
        slot->dies[i] = dies[i];
        g_dm.die[dies[i]].vram_used += per_die;
    }
    if (out) {
        out->slot = (int)(slot - g_dm.binding);
        out->serial = slot->serial;
    }
    pthread_mutex_unlock(&g_dm.lock);
    return 0;

refuse:
    pthread_mutex_unlock(&g_dm.lock);
    return -1;
}

int pn_dm_release(const char *model_name)
{
    return pn_dm_release_on(model_name, -1);
}

/* Locked helper: return `b`'s bytes to each of its dies and free the slot. */
static void binding_release_locked(dm_binding *b)
{
    for (int j = 0; j < b->n_dies; j++) {
        dm_die *d = &g_dm.die[b->dies[j]];
        d->vram_used = d->vram_used > b->bytes_per_die
                     ? d->vram_used - b->bytes_per_die : 0;
    }
    memset(b, 0, sizeof(*b));
}

int pn_dm_release_on(const char *model_name, int first_die)
{
    if (!model_name)
        return -1;
    pthread_mutex_lock(&g_dm.lock);
    for (int i = 0; i < PN_DM_MAX_BINDINGS; i++) {
        dm_binding *b = &g_dm.binding[i];
        if (!b->used || strncmp(b->name, model_name, PN_DM_NAME_MAX) != 0)
            continue;
        if (first_die >= 0 && (b->n_dies < 1 || b->dies[0] != first_die))
            continue;
        binding_release_locked(b);
        pthread_mutex_unlock(&g_dm.lock);
        return 0;
    }
    pthread_mutex_unlock(&g_dm.lock);
    return -1;
}

int pn_dm_release_ref(pn_dm_binding_ref *ref)
{
    int rc = -1;

    if (!ref)
        return -1;
    pthread_mutex_lock(&g_dm.lock);
    if (ref->slot >= 0 && ref->slot < PN_DM_MAX_BINDINGS) {
        dm_binding *b = &g_dm.binding[ref->slot];
        if (b->used && b->serial == ref->serial) {
            binding_release_locked(b);
            rc = 0;
        }
    }
    pthread_mutex_unlock(&g_dm.lock);
    *ref = PN_DM_BINDING_NONE;
    return rc;
}
