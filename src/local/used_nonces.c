/* SPDX-License-Identifier: AGPL-3.0-or-later */
/* Copyright (C) 2026 David (Corey) Herrera */
/**
 * @file   used_nonces.c
 * @brief  Implementation of the used-nonce set (see used_nonces.h). A flat
 *         array scanned per mark: the relay marks a few nonces a minute and the
 *         set holds about a thousand, so a scan costs microseconds.
 *
 * @status None
 * @issues None
 * @todo   None
 */

#include "used_nonces.h"

#include <stdlib.h>
#include <string.h>

typedef struct {
    char      nonce[PN_USED_NONCE_MAX];   /* "" = never used */
    long long until;
} used_nonce;

struct pn_used_nonces {
    int        max;
    used_nonce entries[];
};

pn_used_nonces *pn_used_nonces_new(int max)
{
    pn_used_nonces *set = NULL;

    if (max < 1)
        return NULL;
    set = calloc(1, sizeof(*set) + (size_t)max * sizeof(set->entries[0]));
    if (set)
        set->max = max;
    return set;
}

void pn_used_nonces_free(pn_used_nonces *set)
{
    free(set);
}

int pn_used_nonces_mark(pn_used_nonces *set, const char *nonce, long long until, long long now)
{
    int room = -1;

    if (!set || !nonce || !nonce[0] || strlen(nonce) >= PN_USED_NONCE_MAX)
        return PN_USED_NONCE_E_ARGS;
    for (int i = 0; i < set->max; i++) {
        used_nonce *e = &set->entries[i];
        int live = e->nonce[0] && e->until >= now;
        if (live && strcmp(e->nonce, nonce) == 0)
            return PN_USED_NONCE_SEEN;
        if (!live && room < 0)
            room = i;
    }
    if (room < 0)
        return PN_USED_NONCE_FULL;
    memcpy(set->entries[room].nonce, nonce, strlen(nonce) + 1);
    set->entries[room].until = until;
    return PN_USED_NONCE_OK;
}

int pn_used_nonces_seen(const pn_used_nonces *set, const char *nonce, long long now)
{
    for (int i = 0; set && nonce && nonce[0] && i < set->max; i++)
        if (set->entries[i].nonce[0] && set->entries[i].until >= now &&
            strcmp(set->entries[i].nonce, nonce) == 0)
            return 1;
    return 0;
}

int pn_used_nonces_live(const pn_used_nonces *set, long long now)
{
    int n = 0;

    for (int i = 0; set && i < set->max; i++)
        n += set->entries[i].nonce[0] && set->entries[i].until >= now;
    return n;
}
