/* SPDX-License-Identifier: AGPL-3.0-or-later */
/* Copyright (C) 2026 David (Corey) Herrera */
/**
 * @file pn_tt_role_table.c
 * @brief The role table and the mode switch (pn_tt_role_table.h).
 *
 * @status P0 (plan P0.3).
 * @issues None
 * @todo None
 */

#include "pn_tt_role_table.h"

#include <stddef.h>
#include <string.h>
#include <time.h>

typedef struct role_entry {
    char name[32];
    uint32_t cores[PN_TT_ROLE_TABLE_MAX_CORES];
    uint32_t count;
    uint32_t modes;
    pn_tt_role_teardown_fn teardown;
    pn_tt_role_arm_fn arm;
    void *user;
    bool armed;
    bool used;
} role_entry;

static role_entry g_roles[PN_TT_ROLE_TABLE_MAX_ROLES];
static uint32_t g_mode;

static double role_now(void)
{
    struct timespec t;

    clock_gettime(CLOCK_MONOTONIC, &t);
    return (double)t.tv_sec + (double)t.tv_nsec * 1e-9;
}

static role_entry *role_find(const char *name)
{
    for (uint32_t r = 0; name && r < PN_TT_ROLE_TABLE_MAX_ROLES; r++)
        if (g_roles[r].used && strcmp(g_roles[r].name, name) == 0)
            return &g_roles[r];
    return NULL;
}

int pn_tt_role_table_claim(const char *name, const uint32_t *cores, uint32_t count,
                           uint32_t modes, pn_tt_role_teardown_fn teardown, pn_tt_role_arm_fn arm,
                           void *user)
{
    role_entry *slot = role_find(name);

    if (!name || !cores || count == 0u || count > PN_TT_ROLE_TABLE_MAX_CORES || modes == 0u ||
        strlen(name) >= sizeof(slot->name))
        return -1;
    for (uint32_t i = 0; i < count; i++) {
        if (cores[i] > 139u)
            return -1;
        for (uint32_t r = 0; r < PN_TT_ROLE_TABLE_MAX_ROLES; r++) {
            const role_entry *other = &g_roles[r];

            if (!other->used || other == slot || (other->modes & modes) == 0u)
                continue;
            for (uint32_t c = 0; c < other->count; c++)
                if (other->cores[c] == cores[i])
                    return -1;
        }
    }
    for (uint32_t r = 0; !slot && r < PN_TT_ROLE_TABLE_MAX_ROLES; r++)
        if (!g_roles[r].used)
            slot = &g_roles[r];
    if (!slot)
        return -1;
    memset(slot, 0, sizeof(*slot));
    strcpy(slot->name, name);
    memcpy(slot->cores, cores, count * sizeof(cores[0]));
    slot->count = count;
    slot->modes = modes;
    slot->teardown = teardown;
    slot->arm = arm;
    slot->user = user;
    slot->armed = true;
    slot->used = true;
    return 0;
}

void pn_tt_role_table_release(const char *name)
{
    role_entry *slot = role_find(name);

    if (slot)
        memset(slot, 0, sizeof(*slot));
}

const char *pn_tt_role_table_holder(uint32_t core, uint32_t mode)
{
    for (uint32_t r = 0; r < PN_TT_ROLE_TABLE_MAX_ROLES; r++) {
        const role_entry *role = &g_roles[r];

        if (!role->used || (role->modes & mode) == 0u)
            continue;
        for (uint32_t c = 0; c < role->count; c++)
            if (role->cores[c] == core)
                return role->name;
    }
    return NULL;
}

int pn_tt_role_table_enter(uint32_t mode, void *context, double *out_teardown_seconds,
                           double *out_arm_seconds)
{
    double started = role_now(), torn;
    int rc = 0;

    for (uint32_t r = PN_TT_ROLE_TABLE_MAX_ROLES; r-- > 0u;) {
        role_entry *role = &g_roles[r];

        if (role->used && role->armed && (role->modes & mode) == 0u) {
            if (role->teardown)
                role->teardown(role->user);
            role->armed = false;
        }
    }
    torn = role_now();
    for (uint32_t r = 0; r < PN_TT_ROLE_TABLE_MAX_ROLES; r++) {
        role_entry *role = &g_roles[r];

        if (!role->used || role->armed || (role->modes & mode) == 0u)
            continue;
        const int armed = role->arm ? role->arm(role->user, context) : 0;

        if (armed < 0)
            rc = -1;
        else if (armed == 0)
            role->armed = true;
    }
    g_mode = mode;
    /* Clamped: a monotonic difference must not print negative under -ffast-math rounding. */
    if (out_teardown_seconds)
        *out_teardown_seconds = torn > started ? torn - started : 0.0;
    if (out_arm_seconds) {
        const double now = role_now();

        *out_arm_seconds = now > torn ? now - torn : 0.0;
    }
    return rc;
}

uint32_t pn_tt_role_table_mode(void)
{
    return g_mode;
}

void pn_tt_role_table_clear(void)
{
    memset(g_roles, 0, sizeof(g_roles));
    g_mode = 0u;
}
