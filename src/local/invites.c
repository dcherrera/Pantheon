/* SPDX-License-Identifier: AGPL-3.0-or-later */
/* Copyright (C) 2026 David (Corey) Herrera */
/**
 * @file   invites.c
 * @brief  Implementation of the one-time invite codes (see invites.h).
 *
 * @status None
 * @issues None
 * @todo   None
 */

#include "invites.h"

#include <pthread.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "local_random.h"

typedef struct {
    int    live;
    char   code[PN_INVITE_CODE_DIGITS + 1];
    char   person[PN_INVITE_PERSON_MAX];
    time_t expires;
} invite_slot;

struct pn_invites {
    pthread_mutex_t lock;
    int             ttl_s;
    invite_slot     slots[PN_INVITES_MAX];
};

pn_invites *pn_invites_new(int ttl_s)
{
    pn_invites *inv = calloc(1, sizeof(*inv));

    if (!inv)
        return NULL;
    pthread_mutex_init(&inv->lock, NULL);
    inv->ttl_s = ttl_s > 0 ? ttl_s : PN_INVITE_TTL_S;
    return inv;
}

void pn_invites_free(pn_invites *inv)
{
    if (!inv)
        return;
    pthread_mutex_destroy(&inv->lock);
    memset(inv, 0, sizeof(*inv));
    free(inv);
}

static int tick_locked(pn_invites *inv, time_t now)
{
    int dropped = 0;

    for (int i = 0; i < PN_INVITES_MAX; i++) {
        if (inv->slots[i].live && now >= inv->slots[i].expires) {
            memset(&inv->slots[i], 0, sizeof(inv->slots[i]));
            dropped++;
        }
    }
    return dropped;
}

/* 1 when `code` is already a live invite's code. */
static int taken_locked(const pn_invites *inv, const char *code)
{
    for (int i = 0; i < PN_INVITES_MAX; i++)
        if (inv->slots[i].live && strcmp(inv->slots[i].code, code) == 0)
            return 1;
    return 0;
}

/* 8 random digits no live invite holds (two live invites never share a code). */
static int roll_unique_locked(const pn_invites *inv, char out[PN_INVITE_CODE_DIGITS + 1])
{
    uint32_t v = 0;

    for (int tries = 0; tries < 8; tries++) {
        if (pn_local_random_below(100000000u, &v) != 0)
            return -1;
        snprintf(out, PN_INVITE_CODE_DIGITS + 1, "%08u", (unsigned)v);
        if (!taken_locked(inv, out))
            return 0;
    }
    return -1;
}

int pn_invites_issue(pn_invites *inv, const char *person, time_t now,
                     char code_out[PN_INVITE_CODE_DIGITS + 1])
{
    int rc = -1;

    if (!inv || !person || !person[0] || strlen(person) >= PN_INVITE_PERSON_MAX)
        return -2;
    pthread_mutex_lock(&inv->lock);
    tick_locked(inv, now);
    for (int i = 0; i < PN_INVITES_MAX && rc == -1; i++) {
        invite_slot *s = &inv->slots[i];
        if (s->live)
            continue;
        if (roll_unique_locked(inv, s->code) != 0) {
            rc = -2;
            break;
        }
        snprintf(s->person, sizeof(s->person), "%s", person);
        s->expires = now + inv->ttl_s;
        s->live = 1;
        memcpy(code_out, s->code, sizeof(s->code));
        rc = 0;
    }
    pthread_mutex_unlock(&inv->lock);
    return rc;
}

/* Digits of `in` (spaces and dashes dropped); -1 on anything else. */
static int normalize_code(const char *in, char out[PN_INVITE_CODE_DIGITS + 1])
{
    int n = 0;

    for (const char *c = in ? in : ""; *c; c++) {
        if (*c == ' ' || *c == '-')
            continue;
        if (*c < '0' || *c > '9' || n >= PN_INVITE_CODE_DIGITS)
            return -1;
        out[n++] = *c;
    }
    out[n] = '\0';
    return n == PN_INVITE_CODE_DIGITS ? 0 : -1;
}

int pn_invites_redeem(pn_invites *inv, const char *code, time_t now,
                      char person_out[PN_INVITE_PERSON_MAX])
{
    char given[PN_INVITE_CODE_DIGITS + 1] = {0};
    int hit = -1;

    if (!inv || normalize_code(code, given) != 0)
        return -1;
    pthread_mutex_lock(&inv->lock);
    tick_locked(inv, now);
    for (int i = 0; i < PN_INVITES_MAX; i++) {
        unsigned char diff = 0;
        for (int j = 0; j < PN_INVITE_CODE_DIGITS; j++)
            diff |= (unsigned char)(given[j] ^ inv->slots[i].code[j]);
        if (inv->slots[i].live && diff == 0 && hit < 0)
            hit = i;
    }
    if (hit >= 0) {
        memcpy(person_out, inv->slots[hit].person, PN_INVITE_PERSON_MAX);
        memset(&inv->slots[hit], 0, sizeof(inv->slots[hit]));
    }
    pthread_mutex_unlock(&inv->lock);
    return hit >= 0 ? 0 : -1;
}

void pn_invites_void_all(pn_invites *inv)
{
    if (!inv)
        return;
    pthread_mutex_lock(&inv->lock);
    memset(inv->slots, 0, sizeof(inv->slots));
    pthread_mutex_unlock(&inv->lock);
}

int pn_invites_void_person(pn_invites *inv, const char *person)
{
    int n = 0;

    if (!inv || !person)
        return 0;
    pthread_mutex_lock(&inv->lock);
    for (int i = 0; i < PN_INVITES_MAX; i++) {
        if (inv->slots[i].live && strcmp(inv->slots[i].person, person) == 0) {
            memset(&inv->slots[i], 0, sizeof(inv->slots[i]));
            n++;
        }
    }
    pthread_mutex_unlock(&inv->lock);
    return n;
}

int pn_invites_tick(pn_invites *inv, time_t now)
{
    int n = 0;

    if (!inv)
        return 0;
    pthread_mutex_lock(&inv->lock);
    n = tick_locked(inv, now);
    pthread_mutex_unlock(&inv->lock);
    return n;
}

int pn_invites_live(pn_invites *inv, time_t now)
{
    int n = 0;

    if (!inv)
        return 0;
    pthread_mutex_lock(&inv->lock);
    for (int i = 0; i < PN_INVITES_MAX; i++)
        n += inv->slots[i].live && now < inv->slots[i].expires;
    pthread_mutex_unlock(&inv->lock);
    return n;
}
