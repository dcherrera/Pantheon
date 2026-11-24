/* SPDX-License-Identifier: AGPL-3.0-or-later */
/* Copyright (C) 2026 David (Corey) Herrera */
/**
 * @file   load_options.c
 * @brief  Implementation of the smart load options (see load_options.h).
 *
 * @status None
 * @issues None
 * @todo   None
 */

#include "load_options.h"

#include <stdio.h>
#include <string.h>

#define GB 1e9

static int profile_sessions(const pn_cat_profile *p)
{
    return p->max_sessions > 0 ? p->max_sessions : 1;
}

uint64_t pn_load_option_need(const pn_cat_load_options *o, int ctx, int sessions)
{
    uint64_t per_session = o->mem_per_session + (uint64_t)ctx * o->mem_kv_per_token;

    return o->mem_fixed + (uint64_t)ctx * o->mem_ctx_per_token + (uint64_t)sessions * per_session;
}

/* Insert (ctx, sessions) keeping out[] sorted and unique. */
static int add_choice(pn_load_option *out, int n, int max, int ctx, int sessions)
{
    int i = 0;

    while (i < n && (out[i].ctx_cap < ctx ||
                     (out[i].ctx_cap == ctx && out[i].max_sessions < sessions)))
        i++;
    if ((i < n && out[i].ctx_cap == ctx && out[i].max_sessions == sessions) || n >= max)
        return n;
    memmove(&out[i + 1], &out[i], (size_t)(n - i) * sizeof(*out));
    memset(&out[i], 0, sizeof(*out));
    out[i].ctx_cap = ctx;
    out[i].max_sessions = sessions;
    return n + 1;
}

int pn_load_options_grid(const pn_cat_profile *p, pn_load_option *out, int max)
{
    const pn_cat_load_options *o = &p->options;
    int n = 0;

    if (!o->present) {
        n = add_choice(out, 0, max, p->ctx_cap, profile_sessions(p));
        return n;
    }
    for (int ctx = 4096; ctx <= o->ctx_max; ctx *= 2) {
        if (ctx < o->ctx_min)
            continue;
        for (int s = 1; s <= o->sessions_max; s *= 2)
            n = add_choice(out, n, max, ctx, s);
        n = add_choice(out, n, max, ctx, profile_sessions(p));
    }
    for (int s = 1; s <= o->sessions_max; s *= 2)
        n = add_choice(out, n, max, p->ctx_cap, s);
    n = add_choice(out, n, max, p->ctx_cap, profile_sessions(p));
    for (int i = 0; i < n; i++)
        out[i].need_per_unit = pn_load_option_need(o, out[i].ctx_cap, out[i].max_sessions);
    return n;
}

pn_unit_request pn_load_option_request(const pn_cat_profile *p, const pn_load_option *o,
                                       const pn_hw_report *hw)
{
    uint64_t reserve = hw->die_reserve;
    pn_unit_request r = p->hardware;

    if (p->options.present && strcmp(r.unit_type, "cpu") != 0) {
        uint64_t want = o->need_per_unit + PN_LOAD_FIT_MARGIN_BYTES;
        uint64_t budget = want > reserve ? want - reserve : 0;
        if (budget > r.min_mem_per_unit)
            r.min_mem_per_unit = budget;
    }
    return r;
}

static bool option_fits(const pn_cat_profile *p, const pn_hw_report *hw,
                        const pn_load_option *o)
{
    pn_unit_request r = pn_load_option_request(p, o, hw);
    int ids[PN_UNIT_PLACE_MAX], n = 0;
    char why[160];

    return pn_unit_place(hw, &r, ids, &n, why, sizeof(why)) == 0;
}

/* Index of the recommended choice among the fitting ones, or -1. */
static int pick_recommended(const pn_cat_profile *p, const pn_load_option *out, int n)
{
    int best = -1, smallest = -1, users = profile_sessions(p);

    for (int i = 0; i < n; i++) {
        if (!out[i].fits)
            continue;
        if (out[i].ctx_cap == p->ctx_cap && out[i].max_sessions == users)
            return i;
        if (smallest < 0 || out[i].need_per_unit < out[smallest].need_per_unit)
            smallest = i;
        if (out[i].ctx_cap > p->ctx_cap || out[i].max_sessions > users)
            continue;
        if (best < 0 || out[i].ctx_cap > out[best].ctx_cap ||
            (out[i].ctx_cap == out[best].ctx_cap && out[i].max_sessions > out[best].max_sessions))
            best = i;
    }
    return best >= 0 ? best : smallest;
}

int pn_load_options_fit(const pn_cat_profile *p, const pn_hw_report *hw, pn_load_option *out,
                        int max)
{
    int n = pn_load_options_grid(p, out, max), rec;

    for (int i = 0; i < n; i++)
        out[i].fits = option_fits(p, hw, &out[i]);
    rec = pick_recommended(p, out, n);
    if (rec >= 0)
        out[rec].recommended = true;
    return n;
}

/* The most device memory free on any free unit of the profile's type. */
static uint64_t best_free_device(const pn_cat_profile *p, const pn_hw_report *hw)
{
    uint64_t best = 0, reserve = hw->die_reserve;

    for (int i = 0; i < hw->n_units; i++) {
        const pn_hw_unit *u = &hw->units[i];
        if (strcmp(u->type, p->hardware.unit_type) == 0 && pn_hw_unit_available(u) &&
            u->mem_free + reserve > best)
            best = u->mem_free + reserve;
    }
    return best;
}

/* "8192, 16384, 32768": the distinct contexts (or user counts) on offer. */
static void list_offered(const pn_load_option *opts, int n, bool users, char *out, size_t cap)
{
    size_t len = 0;
    int last = -1;

    out[0] = '\0';
    for (int i = 0; i < n; i++) {
        int v = users ? opts[i].max_sessions : opts[i].ctx_cap;
        bool seen = v == last;
        for (int j = 0; users && j < i; j++)
            seen |= opts[j].max_sessions == v;
        if (seen || len + 16 >= cap)
            continue;
        len += (size_t)snprintf(out + len, cap - len, "%s%d", len ? ", " : "", v);
        last = v;
    }
}

static int refuse_not_offered(const pn_load_option *opts, int n, bool users, int value,
                              char *why, size_t cap)
{
    char offered[160];

    list_offered(opts, n, users, offered, sizeof(offered));
    snprintf(why, cap, "%d %s is not an offered %s (%s)", value, users ? "users" : "tokens",
             users ? "number of users" : "context", offered);
    return -1;
}

int pn_load_options_choose(const pn_cat_profile *p, const pn_hw_report *hw, int ctx,
                           int sessions, pn_load_option *chosen, char *why, size_t cap)
{
    pn_load_option opts[PN_LOAD_OPTIONS_MAX];
    int n = pn_load_options_fit(p, hw, opts, PN_LOAD_OPTIONS_MAX);
    bool ctx_ok = false, users_ok = false;

    if (ctx <= 0 && sessions <= 0) {
        for (int i = 0; i < n; i++) {
            if (opts[i].recommended) {
                *chosen = opts[i];
                return 0;
            }
        }
        ctx = p->ctx_cap;   /* nothing fits: refuse with the default's numbers */
    }
    if (ctx <= 0)
        ctx = p->ctx_cap;
    if (sessions <= 0)
        sessions = profile_sessions(p);
    for (int i = 0; i < n; i++) {
        ctx_ok |= opts[i].ctx_cap == ctx;
        users_ok |= opts[i].max_sessions == sessions;
        if (opts[i].ctx_cap != ctx || opts[i].max_sessions != sessions)
            continue;
        if (opts[i].fits) {
            *chosen = opts[i];
            return 0;
        }
        int ids[PN_UNIT_PLACE_MAX], placed = 0;
        if (pn_unit_place(hw, &p->hardware, ids, &placed, why, cap) != 0)
            return -1;   /* short of units, whatever the memory: say that */
        snprintf(why, cap, "%d tokens x %d user(s) needs %.1f GB on each %s%s; the free "
                 "ones have at most %.1f GB", ctx, sessions,
                 (double)(opts[i].need_per_unit + PN_LOAD_FIT_MARGIN_BYTES) / GB,
                 p->hardware.unit_type,
                 p->hardware.same_card ? " (on one card)"
                 : p->hardware.per_card > 0 ? " (whole cards)" : "",
                 (double)best_free_device(p, hw) / GB);
        return -1;
    }
    if (!ctx_ok)
        return refuse_not_offered(opts, n, false, ctx, why, cap);
    if (!users_ok)
        return refuse_not_offered(opts, n, true, sessions, why, cap);
    snprintf(why, cap, "%d tokens x %d users is not an offered choice", ctx, sessions);
    return -1;
}

int pn_local_kv_format_of(const char *kv_type)
{
    if (kv_type && strcmp(kv_type, "q4") == 0)
        return PN_KV_FORMAT_Q4;
    if (kv_type && strcmp(kv_type, "fp32") == 0)
        return PN_KV_FORMAT_FP32;
    return PN_KV_FORMAT_DEFAULT;
}

const char *pn_local_tt_kv_type_of(const char *unit_type, const char *kv_type)
{
    if (!unit_type || strcmp(unit_type, "p150") != 0 || !kv_type || !kv_type[0] ||
        strcmp(kv_type, "bf16") == 0 || !pn_cat_kv_type_allowed(unit_type, kv_type) ||
        pn_local_kv_format_of(kv_type) != PN_KV_FORMAT_DEFAULT)
        return NULL;
    return kv_type;
}

const char *pn_local_tt_spec_of(const char *unit_type, bool mtp, int mtp_drafts, char *out,
                                size_t cap)
{
    if (!unit_type || strcmp(unit_type, "p150") != 0 || (mtp && mtp_drafts <= 0) || cap < 4)
        return NULL;
    snprintf(out, cap, "%d", mtp ? mtp_drafts : 0);
    return out;
}
