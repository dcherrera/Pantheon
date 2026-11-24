/* SPDX-License-Identifier: AGPL-3.0-or-later */
/* Copyright (C) 2026 David (Corey) Herrera */
/**
 * @file   hw_report.c
 * @brief  Implementation of the card → unit tree (see hw_report.h).
 *
 * @status None
 * @issues None
 * @todo   None
 */

#include "hw_report.h"

#include <stdio.h>
#include <string.h>

#include "../accel/die_manager.h"
#include "../core/load_progress.h"
#include "../core/mem/pn_memwatch.h"
#include "hw_mem_poller.h"
#include "nvml_device_mem.h"

#ifdef USE_CUDA
#include "cuda/fc_cuda.h"
#endif

#ifdef USE_TT
#include "../accel/tt/pn_tt_device.h"
#endif

/* The device reader in force (NULL = default_reader). */
static pn_hw_mem_reader g_reader;

void pn_hw_report_set_mem_reader(pn_hw_mem_reader reader)
{
    g_reader = reader;
}

/* Hardware dies only: a logical die has no device to read. CUDA: NVML (no
 * context at all); without NVML a transient bare context per read
 * (fc_cuda_probe_device), which leaves the die exactly as it was, and never
 * on a die a model of ours is running on (a context there would stall its
 * launches; its numbers stay the die manager's). TT: this process's open
 * devices' DRAM allocators; a card we have not opened holds nothing of ours,
 * and TT has no cross-process view. */
static int default_reader(pn_hw_family family, int die, pn_hw_dev_mem *out)
{
    memset(out, 0, sizeof(*out));
    if (!pn_dm_is_hardware())
        return -1;
#ifdef USE_CUDA
    if (family == PN_HW_FAMILY_K80) {
        size_t free_b = 0, total_b = 0;
        pn_dm_die d;
        if (pn_nvml_read_die(die, out) == 0)
            return 0;
        if (pn_dm_get(die, &d) == 0 && d.vram_bound > 0)
            return -1;
        if (fc_cuda_probe_device(die, &free_b, &total_b) != 0 || total_b == 0)
            return -1;
        out->total = total_b;
        out->used = total_b > free_b ? total_b - free_b : 0;
        return 0;
    }
#endif
#ifdef USE_TT
    if (family == PN_HW_FAMILY_P150) {
        uint64_t used = 0, total = 0;
        pn_dm_die d;
        if (pn_tt_device_dram_usage_on(die, &used, &total) != 0) {
            if (pn_dm_get(die, &d) != 0)
                return -1;
            used = 0;
            total = d.vram_budget + pn_die_reserve_bytes();
        }
        out->total = total;
        out->used = used;
        out->used_self = used;
        out->self_known = true;
        return 0;
    }
#endif
    (void)family;
    (void)die;
    return -1;
}

/* Split one device reading into ours and other programs', and derive free.
 * Without a measured share of our own, ours is the die's bindings and device
 * use within PN_HW_READ_SLACK_BYTES of that is ours too (contexts, the
 * reading's own context). `live_self` = bytes a load / unload in progress
 * reports on the die (load_progress.h), 0 when none. Mostly held = more than
 * half the budget. */
static void apply_reading(pn_hw_unit *u, const pn_hw_dev_mem *m, uint64_t bound,
                          uint64_t live_self)
{
    uint64_t self = m->self_known ? m->used_self : bound;
    uint64_t slack = m->self_known ? 0 : PN_HW_READ_SLACK_BYTES;
    uint64_t other = m->used > self + slack ? m->used - self - slack : 0;
    /* A load or unload running now: the engine's own count is newer than the
     * reading. What it adds is ours, never another program's. */
    if (live_self > self) {
        uint64_t grown = live_self - self;
        other = other > grown ? other - grown : 0;
        self = live_self;
    }
    uint64_t used = self + other;

    u->mem_live = true;
    u->mem_device_total = m->total;
    u->mem_used_self = self;
    u->mem_used_other = other;
    u->mem_free = used < u->mem_total ? u->mem_total - used : 0;
    u->in_use_other = other > u->mem_total / 2;
}

int pn_hw_report_read_device(pn_hw_family family, int die, pn_hw_dev_mem *out)
{
    return (g_reader ? g_reader : default_reader)(family, die, out);
}

pn_hw_family pn_hw_family_detect(void)
{
    if (pn_dm_count() <= 0)
        return PN_HW_FAMILY_CPU;
#ifdef USE_CUDA
    if (fc_driver_available())
        return PN_HW_FAMILY_K80;
#endif
#ifdef USE_TT
    return PN_HW_FAMILY_P150;
#else
    return PN_HW_FAMILY_K80;
#endif
}

/* Record which models sit on unit `u` (by die id). */
static void attach_models(pn_hw_unit *u, const mr_info *models, int n_models)
{
    for (int m = 0; m < n_models; m++) {
        for (int d = 0; d < models[m].n_dies; d++) {
            if (models[m].dies[d] != u->die)
                continue;
            if (u->n_models++ == 0)
                snprintf(u->model, sizeof(u->model), "%s", models[m].name);
        }
    }
}

/* The single cpu card + unit of a box with no accelerator dies. */
static void build_cpu(pn_hw_report *out, const mr_info *models, int n_models, uint64_t ram)
{
    pn_hw_card *c = &out->cards[0];
    pn_hw_unit *u = &out->units[0];

    c->id = 0;
    snprintf(c->type, sizeof(c->type), "cpu");
    c->first_unit = 0;
    c->n_units = 1;
    u->id = 0;
    u->card = 0;
    u->die = -1;
    snprintf(u->type, sizeof(u->type), "cpu");
    u->mem_total = ram;
    u->mem_free = ram;
    u->mem_device_total = ram;
    u->temperature_c = -1;
    for (int m = 0; m < n_models; m++) {
        if (models[m].n_dies != 0)
            continue;
        if (u->n_models++ == 0)
            snprintf(u->model, sizeof(u->model), "%s", models[m].name);
    }
    out->n_cards = 1;
    out->n_units = 1;
}

int pn_hw_report_build(pn_hw_family family, const mr_info *models, int n_models,
                       uint64_t host_ram, pn_hw_report *out)
{
    int n_dies = pn_dm_count();

    memset(out, 0, sizeof(*out));
    out->family = family;
    if (family == PN_HW_FAMILY_CPU || n_dies <= 0) {
        out->family = PN_HW_FAMILY_CPU;
        build_cpu(out, models, n_models, host_ram);
        return 0;
    }
    const bool k80 = family == PN_HW_FAMILY_K80;
    const pn_hw_mem_reader reader = g_reader ? g_reader : default_reader;
    out->die_reserve = pn_die_reserve_bytes();
    for (int die = 0; die < n_dies && out->n_units < PN_HW_MAX_UNITS; die++) {
        pn_dm_die d;
        if (pn_dm_get(die, &d) != 0)
            continue;
        int card_id = k80 ? pn_dm_card_of(die) : die;
        if (card_id >= PN_HW_MAX_CARDS)
            break;
        if (card_id >= out->n_cards) {
            pn_hw_card *c = &out->cards[card_id];
            c->id = card_id;
            snprintf(c->type, sizeof(c->type), "%s", k80 ? "k80" : "p150");
            c->first_unit = out->n_units;
            out->n_cards = card_id + 1;
        }
        pn_hw_unit *u = &out->units[out->n_units++];
        out->cards[card_id].n_units++;
        u->id = die;
        u->card = card_id;
        u->die = die;
        snprintf(u->type, sizeof(u->type), "%s", k80 ? "k80-gpu" : "p150");
        u->mem_total = d.vram_budget;
        u->mem_free = d.vram_used < d.vram_budget ? d.vram_budget - d.vram_used : 0;
        u->mem_used_self = d.vram_bound;
        u->mem_used_other = d.vram_used > d.vram_bound ? d.vram_used - d.vram_bound : 0;
        u->temperature_c = -1;
        pn_hw_dev_mem m;
        uint64_t age = 0;
        int busy = 0;
        uint64_t live = pn_load_progress_die_bytes(die, &busy);
        /* The background snapshot when it runs (instant), else a read now. */
        int got = pn_hw_mem_poller_running() ? pn_hw_mem_poller_get(die, &m, &age)
                                             : reader(family, die, &m);
        if (got == 0) {
            apply_reading(u, &m, d.vram_bound, busy ? live : 0);
            u->mem_age_ms = age;
            /* Placement (pn_dm_place / pn_dm_bind) must see what the tiles show. */
            pn_dm_note_external(die, u->mem_used_other);
        } else if (busy && live > u->mem_used_self) {
            u->mem_used_self = live;
            u->mem_free = live < u->mem_total ? u->mem_total - live : 0;
        }
        attach_models(u, models, n_models);
    }
    return 0;
}

/* "loaded" (a model of ours), "in_use_other" (mostly held by another
 * program), else "free". */
static const char *unit_status(const pn_hw_unit *u)
{
    if (u->n_models > 0)
        return "loaded";
    return u->in_use_other ? "in_use_other" : "free";
}

void pn_hw_report_write_cards(json_writer *jw, const pn_hw_report *hw)
{
    jw_raw(jw, "\"cards\":[");
    for (int c = 0; c < hw->n_cards; c++) {
        const pn_hw_card *card = &hw->cards[c];
        jw_printf(jw, "%s{\"id\":%d,\"type\":\"%s\",\"units\":[", c ? "," : "", card->id,
                  card->type);
        for (int i = 0; i < card->n_units; i++) {
            const pn_hw_unit *u = &hw->units[card->first_unit + i];
            jw_printf(jw, "%s{\"id\":%d,\"type\":\"%s\",\"status\":\"%s\",\"mem_total\":%llu,"
                          "\"mem_free\":%llu,\"mem_device_total\":%llu,\"mem_used_self\":%llu,"
                          "\"mem_used_other\":%llu,\"mem_live\":%s,\"mem_age_ms\":%llu,"
                          "\"model\":",
                      i ? "," : "", u->id, u->type, unit_status(u),
                      (unsigned long long)u->mem_total, (unsigned long long)u->mem_free,
                      (unsigned long long)u->mem_device_total,
                      (unsigned long long)u->mem_used_self,
                      (unsigned long long)u->mem_used_other, u->mem_live ? "true" : "false",
                      (unsigned long long)u->mem_age_ms);
            if (u->model[0])
                jw_string(jw, u->model);
            else
                jw_raw(jw, "null");
            jw_printf(jw, ",\"n_models\":%d,\"temperature_c\":", u->n_models);
            if (u->temperature_c >= 0)
                jw_printf(jw, "%d}", u->temperature_c);
            else
                jw_raw(jw, "null}");
        }
        jw_raw(jw, "]}");
    }
    jw_raw(jw, "]");
}

void pn_hw_report_card_summary(const pn_hw_report *hw, char *out, size_t cap)
{
    if (hw->n_cards <= 0 || hw->family == PN_HW_FAMILY_CPU) {
        snprintf(out, cap, "cpu");
        return;
    }
    snprintf(out, cap, "%sx%d", hw->cards[0].type, hw->n_cards);
}

