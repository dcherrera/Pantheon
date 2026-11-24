/* SPDX-License-Identifier: AGPL-3.0-or-later */
/* Copyright (C) 2026 David (Corey) Herrera */
/**
 * @file die_manager.h
 * @brief Minimal die manager — the placement primitive under multi-model
 *        serving. Tracks per-die
 *        state (bound model names, VRAM budget/occupancy) and assigns models
 *        to die-sets, topology-aware: a K80 card is the die pair (2i, 2i+1)
 *        linked by a PLX switch, so 2-die models get a same-card pair and
 *        1-die models pack onto cards whose mate is already occupied (keeping
 *        whole pairs free for the next 2-die model).
 *
 *        Three init paths, one per accelerator family: pn_dm_init_logical()
 *        creates N logical dies with a given budget (the CPU/Mac/test path —
 *        placement bookkeeping needs no GPU), pn_dm_init_cuda() (hardware) probes
 *        real K80 dies via fc_cuda_probe_device, and pn_dm_init_tt() (hardware)
 *        probes Tenstorrent Blackhole cards via ft_tt_init_device + ARC
 *        telemetry. All three feed pn_memwatch_set_die so MemoryAware
 *        admission sees the same per-die truth. This module is bookkeeping
 *        only — it never uploads weights; pn_load_on() does the binding of a
 *        model to the die-set this module picks.
 *
 * @status Logical path exercised on the Mac (tests/test_die_manager.c). The
 *         CUDA probe compiles under USE_CUDA and is gated on GPU hardware (needs live
 *         dies to validate budgets + memwatch feed). The TT probe compiles
 *         under USE_TT and is gated on GPU hardware the same way.
 * @issues Placement assumes the fixed K80 pairing (card = dies 2i/2i+1); a
 *         different topology needs a card-map input. On a TT card the pairing
 *         is inert (one die per device) rather than wrong.
 * @todo
 *   - [ ] (hardware) Validate pn_dm_init_cuda budgets vs nvidia-smi on 16 dies.
 */

#ifndef PN_DIE_MANAGER_H
#define PN_DIE_MANAGER_H

#include <stdbool.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

/* Hard caps: the K80 test machine has 16 GK210 dies (mirrors PN_MEM_MAX_DIES); one
 * binding per resident model. */
#define PN_DM_MAX_DIES      16
#define PN_DM_MAX_BINDINGS  16
#define PN_DM_NAME_MAX      64
#define PN_DM_DIES_PER_CARD 2

/* Copy-out view of one die (pn_dm_get / status surfaces). */
typedef struct {
    int      id;
    char     model_name[PN_DM_NAME_MAX]; /* first bound model, "" if free   */
    int      n_models;                   /* bindings touching this die      */
    uint64_t vram_budget;                /* usable bytes (total - reserve)  */
    uint64_t vram_used;                  /* bindings + other processes' use */
    uint64_t vram_bound;                 /* bytes reserved by bindings only  */
} pn_dm_die;

/* Start the manager with n_dies LOGICAL dies of budget_bytes each — the
 * CPU/test path (a model can be "placed" without CUDA present). Idempotent
 * while running (returns 0 without re-init). Returns 0 on success, -1 on a
 * bad argument.
 *
 * Thread-safe: yes */
int pn_dm_init_logical(int n_dies, uint64_t budget_bytes);

#ifdef USE_CUDA
/* (hardware) Start the manager from real dies: probe CUDA ordinals [0, n_dies)
 * via fc_cuda_init_device + fc_mem_info_ctx, budget = total −
 * pn_die_reserve_bytes(), and feed each die into pn_memwatch_set_die when the
 * watcher is running. Requires fc_init() done and fc_available() true —
 * returns -1 otherwise (caller falls back to pn_dm_init_logical or CPU-only).
 * Idempotent while running. Returns the number of dies probed (>0) or -1. */
int pn_dm_init_cuda(int n_dies);
#endif

#ifdef USE_TT
/* (hardware) Start the manager from Tenstorrent card(s): probe device ordinals
 * [0, n_devices) via ft_tt_init_device + ARC telemetry, register ONE die per
 * DEVICE with budget = total − pn_die_reserve_bytes(), and feed each into
 * pn_memwatch_set_die when the watcher is running. Ordinals are contiguous, so
 * the first one that will not open ends the scan. Idempotent while running.
 * Returns the number of devices probed (>0) or -1.
 *
 * ONE-DEVICE-ONE-DIE, deliberately, and NOT core-group-as-unit (Blackhole spec
 * §9.3). The unit this module models is a MEMORY DOMAIN — every field it
 * stores is a byte budget or a byte occupancy — and a p150a has exactly one:
 * ~31.88 GiB of GDDR6 that all 140 Tensix cores address over the same NoC. A
 * core group is a COMPUTE partition, not a memory domain, so registering N
 * groups as N dies would divide one real budget into N fictional ones (and
 * refuse a model that fits the card), and would assert an independence the
 * hardware does not have — Phase 6.7 measured a per-NoC-row DRAM ceiling
 * shared across groups, and readers on one side of the grid slowing readers on
 * the other.
 *
 * The K80 topology is deliberately not ported forward: pn_dm_card_of()'s
 * die-pair model describes two GK210s with private VRAM behind a PLX switch,
 * and there is no analogue here. At one die per device it degenerates to the
 * identity rather than to something wrong.
 *
 * Multi-model isolation is a real requirement and it already has a mechanism
 * one layer down: ft_tt_group_partition() hands each group disjoint cores AND
 * disjoint DRAM banks with its own allocator. When several models share a
 * card, the field to add is a CORE COUNT on the binding beside the single
 * memory die — memory and compute are separate resources on this part, which
 * is the one structural difference from a K80 die.
 *
 * Note there is no cross-process occupancy signal: ft_tt_mem_info()'s `free`
 * is total minus THIS context's own allocations, so unlike the CUDA probe
 * there is no stranded-memory term and vram_used starts at 0.
 *
 * Thread-safe: yes */
int pn_dm_init_tt(int n_devices);

/**
 * Whether this process was TOLD which Tenstorrent transport to use
 * (`FT_TT_DRIVER=kmd|udrv`) — including having named one that does not exist.
 *
 * The caller needs this because "no Tenstorrent device" has two meanings and
 * only one of them is benign. On a machine with no card a failed pn_dm_init_tt
 * is expected and serving CPU-resident is correct. When somebody named a
 * transport, the same failure means the card they asked for could not be
 * reached — and answering on the CPU would be the silent fallback this project
 * has shipped twice, returning a plausible answer from the wrong thing. It was
 * measured on 2026-08-17 that the card and the CPU fallback return
 * BYTE-IDENTICAL completion text, so nothing downstream can catch it.
 *
 * Thread-safe: yes */
bool pn_dm_tt_driver_explicit(void);
#endif

/* True while the manager tracks real dies (pn_dm_init_cuda / pn_dm_init_tt), false
 * for logical dies or when stopped.
 *
 * Thread-safe: yes */
bool pn_dm_is_hardware(void);

/* Stop tracking and forget all dies/bindings. Safe to call if never started. */
void pn_dm_shutdown(void);

/* Number of tracked dies (0 = manager not running → CPU-only placement). */
int pn_dm_count(void);

/* Copy die_id's state into *out. Returns 0, or -1 if unknown/not running. */
int pn_dm_get(int die_id, pn_dm_die *out);

/* The card index a die sits on (K80 pairing: die/2). Pure. */
static inline int pn_dm_card_of(int die_id) { return die_id / PN_DM_DIES_PER_CARD; }

/* Record what other processes hold on die_id right now (a live device read):
 * vram_used becomes the die's bindings plus `external`, capped at the budget, so
 * pn_dm_place / pn_dm_bind see the same occupancy the hardware report shows.
 * Returns 0, or -1 if the die is unknown or the manager is not running.
 * Thread-safe: yes */
int pn_dm_note_external(int die_id, uint64_t external);

/* Pick a die-set for a model needing need_bytes across n_dies_wanted dies
 * (bytes split evenly). Topology-aware: n=2 → a same-card pair, preferring a
 * fully-free pair, least-loaded tie-break; n=1 → pack onto a die whose card
 * mate is already occupied, least-loaded among those, else the least-loaded
 * free-fitting die. Writes the chosen ids to dies_out[0..n) and returns 0, or
 * -1 when nothing fits (or the manager is not running). Does NOT reserve —
 * call pn_dm_bind to commit.
 *
 * Thread-safe: yes */
int pn_dm_place(uint64_t need_bytes, int n_dies_wanted, int *dies_out);

/* Commit model_name to the die-set: re-checks capacity under the lock and
 * adds total_bytes/n to each die's vram_used. Rejects a duplicate name, an
 * unknown die, or a set that no longer fits. Returns 0 on success, -1 on
 * refusal (nothing is partially bound).
 *
 * Thread-safe: yes */
int pn_dm_bind(const int *dies, int n, const char *model_name,
               uint64_t total_bytes);

/* Names ONE binding exactly. A model name does not: replicas of one model
 * share it, and a by-name release frees whichever binding comes first. */
typedef struct {
    int      slot;     /* binding table index, -1 = none */
    uint64_t serial;   /* that binding's serial, so a reused slot never matches */
} pn_dm_binding_ref;

#define PN_DM_BINDING_NONE ((pn_dm_binding_ref){ .slot = -1, .serial = 0 })

/* pn_dm_bind, and on success *out (may be NULL) names the new binding for
 * pn_dm_release_ref. *out is PN_DM_BINDING_NONE on refusal.
 *
 * Thread-safe: yes */
int pn_dm_bind_ref(const int *dies, int n, const char *model_name,
                   uint64_t total_bytes, pn_dm_binding_ref *out);

/* Release exactly the binding `*ref` names, returning its bytes to each die,
 * and reset *ref to PN_DM_BINDING_NONE. Returns 0, or -1 when *ref names no
 * live binding (never bound, already released, or the manager restarted) — in
 * which case nothing is released.
 *
 * Thread-safe: yes */
int pn_dm_release_ref(pn_dm_binding_ref *ref);

/* Release model_name's FIRST binding, returning its bytes to each die. With
 * replicas of one name this may be another replica's: release a binding you
 * took by its ref. Returns 0, or -1 if the name is unknown. */
int pn_dm_release(const char *model_name);

/* Release the binding of model_name whose die-set starts at first_die (one
 * replica of a name placed on several die-sets); first_die < 0 = the first
 * binding of that name, as pn_dm_release. Returns 0, or -1 if none matches.
 *
 * Thread-safe: yes */
int pn_dm_release_on(const char *model_name, int first_die);

#ifdef __cplusplus
}
#endif

#endif /* PN_DIE_MANAGER_H */
