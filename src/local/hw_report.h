/* SPDX-License-Identifier: AGPL-3.0-or-later */
/* Copyright (C) 2026 David (Corey) Herrera */
/**
 * @file   hw_report.h
 * @brief  The box's accelerator tree, cards → loadable units, built from the die
 *         manager and the model registry (spec §4.2). K80: one card per board,
 *         two GPU units per card. p150: one card = one unit (cores not exposed).
 *         No accelerator: one `cpu` card with one `cpu` unit.
 *         Unit memory comes from the device: the background poller's latest
 *         reading when it runs (hw_mem_poller.h), else a read now (CUDA: NVML,
 *         or cuMemGetInfo through a transient bare context; TT: the card DRAM
 *         allocator of this process's open devices). It is split into what this
 *         Pantheon holds and what other programs hold; a load or unload in
 *         progress counts its live bytes as ours. A unit mostly held by
 *         another program is not free (status "in_use_other"); the fit uses
 *         the same numbers.
 *
 * @status None
 * @issues CUDA has no per-process split without NVML: our share is the die
 *         manager's bindings, and device use beyond that (less a slack for our
 *         own contexts and the reading's context) counts as other programs'.
 *         TT has no cross-process view: other programs' use always reads 0.
 *         Temperature is not read (needs NVML / ARC telemetry); reported null.
 * @todo   None
 */

#ifndef PN_LOCAL_HW_REPORT_H
#define PN_LOCAL_HW_REPORT_H

#include <stdbool.h>
#include <stdint.h>

#include "../serve/json_writer.h"
#include "../serve/model_registry.h"

#ifdef __cplusplus
extern "C" {
#endif

#define PN_HW_MAX_CARDS  16
#define PN_HW_MAX_UNITS  32
#define PN_HW_TYPE_MAX   16

/* Which accelerator family the die manager holds. */
typedef enum {
    PN_HW_FAMILY_CPU = 0,   /* no accelerator dies */
    PN_HW_FAMILY_K80,       /* CUDA dies, paired per board */
    PN_HW_FAMILY_P150,      /* Tenstorrent devices, one die per card */
} pn_hw_family;

/* One loadable unit: a K80 GPU, a whole p150, or the host CPU. */
typedef struct {
    int      id;                         /* unit id == die id; 0 for the cpu unit */
    int      card;                       /* index into cards[] */
    int      die;                        /* die id, -1 for the cpu unit */
    char     type[PN_HW_TYPE_MAX];       /* "k80-gpu", "p150", "cpu" */
    uint64_t mem_total;                  /* usable bytes (die budget / host RAM) */
    uint64_t mem_free;                   /* mem_total minus used (ours + others') */
    uint64_t mem_device_total;           /* device bytes as read, 0 = not read */
    uint64_t mem_used_self;              /* bytes this Pantheon holds */
    uint64_t mem_used_other;             /* bytes other programs hold */
    bool     mem_live;                   /* the numbers come from the device */
    uint64_t mem_age_ms;                 /* age of that reading (0 = read just now) */
    bool     in_use_other;               /* mostly held by another program */
    char     model[MR_MAX_NAME];         /* first model on the unit, "" = free */
    int      n_models;                   /* models touching this unit */
    int      temperature_c;              /* -1 = not available */
} pn_hw_unit;

typedef struct {
    int  id;
    char type[PN_HW_TYPE_MAX];           /* "k80", "p150", "cpu" */
    int  first_unit;                     /* units[first_unit .. +n_units) */
    int  n_units;
} pn_hw_card;

typedef struct {
    pn_hw_family family;
    pn_hw_card   cards[PN_HW_MAX_CARDS];
    int          n_cards;
    pn_hw_unit   units[PN_HW_MAX_UNITS];
    int          n_units;
    uint64_t     die_reserve;            /* bytes each accelerator unit keeps back:
                                          * mem_total = device total - die_reserve */
} pn_hw_report;

/* Device use within this of our own bindings is ours: the CUDA contexts and
 * workspaces a binding estimate leaves out, and the reading's own context. */
#define PN_HW_READ_SLACK_BYTES (256ULL << 20)

/* One device memory reading. */
typedef struct {
    uint64_t total;       /* device bytes */
    uint64_t used;        /* bytes in use on the device, by anyone */
    uint64_t used_self;   /* bytes this process holds (when self_known) */
    bool     self_known;  /* the source measures our share (TT allocator); else
                           * our share is the die manager's bindings */
} pn_hw_dev_mem;

/* Reads one unit's device memory; 0 on success, -1 if it cannot be read. */
typedef int (*pn_hw_mem_reader)(pn_hw_family family, int die, pn_hw_dev_mem *out);

/**
 * Replace the device reader (tests: a fake device). NULL restores the default:
 * CUDA cuMemGetInfo / TT allocator accounting on hardware dies, nothing on
 * logical dies (their numbers stay the die manager's bookkeeping).
 *
 * Thread-safe: no (set before serving)
 * Reentrant:   no
 * Signal-safe: no
 */
void pn_hw_report_set_mem_reader(pn_hw_mem_reader reader);

/**
 * One device reading through the reader in force (the background poller's
 * source; hw_mem_poller.h).
 * @return 0, or -1 when the die cannot be read
 *
 * Reentrant:   yes
 * Signal-safe: no
 */
int pn_hw_report_read_device(pn_hw_family family, int die, pn_hw_dev_mem *out);

/**
 * A unit a new load may take: nothing of ours on it and not held by another
 * program. (Whether its memory suffices is the caller's check on mem_free.)
 *
 * Thread-safe: yes
 * Reentrant:   yes
 * Signal-safe: yes
 */
static inline bool pn_hw_unit_available(const pn_hw_unit *u)
{
    return u->n_models == 0 && !u->in_use_other;
}

/**
 * The family this process's die manager holds: K80 in a CUDA build with a
 * driver, p150 in a TT build, else CPU (also when no die is visible).
 *
 * Thread-safe: yes
 * Reentrant:   yes
 * Signal-safe: no
 */
pn_hw_family pn_hw_family_detect(void);

/**
 * Build the card → unit tree from the running die manager.
 *
 * @param family     accelerator family (pn_hw_family_detect, or forced in tests)
 * @param models     registry snapshot (mr_snapshot), may be NULL
 * @param n_models   rows in models
 * @param host_ram   host RAM bytes, the cpu unit's mem_total
 * @param out        filled
 * @return 0
 *
 * Thread-safe: yes (die manager is locked per read)
 * Reentrant:   yes
 * Signal-safe: no
 */
int pn_hw_report_build(pn_hw_family family, const mr_info *models, int n_models,
                       uint64_t host_ram, pn_hw_report *out);

/**
 * Append `"cards":[...]` (no surrounding braces) to jw.
 *
 * Thread-safe: yes
 * Reentrant:   yes
 * Signal-safe: no
 */
void pn_hw_report_write_cards(json_writer *jw, const pn_hw_report *hw);

/**
 * Short card summary for discovery, e.g. "k80x8", "p150x1", "cpu".
 *
 * Thread-safe: yes
 * Reentrant:   yes
 * Signal-safe: yes
 */
void pn_hw_report_card_summary(const pn_hw_report *hw, char *out, size_t cap);

#ifdef __cplusplus
}
#endif

#endif /* PN_LOCAL_HW_REPORT_H */
