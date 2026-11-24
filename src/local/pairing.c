/* SPDX-License-Identifier: AGPL-3.0-or-later */
/* Copyright (C) 2026 David (Corey) Herrera */
/**
 * @file   pairing.c
 * @brief  Implementation of the pairing code (see pairing.h).
 *
 * @status None
 * @issues None
 * @todo   None
 */

#include "pairing.h"

#include <pthread.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include <unistd.h>

#include "local_file_io.h"
#include "local_random.h"

struct pn_pairing {
    pthread_mutex_t lock;
    pthread_t       ticker;
    int             ticker_running;
    volatile int    stop;
    char            path[PN_LOCAL_PATH_MAX];
    char            code[PN_PAIR_CODE_DIGITS + 1];
    time_t          made;
    int             ttl_s;
    int             failures;
};

/* New code, file, one log line. Lock held (or not yet shared). */
static int rotate_locked(pn_pairing *p, const char *why)
{
    uint32_t v = 0;
    char line[PN_PAIR_CODE_DIGITS + 2];

    if (pn_local_random_below(100000000u, &v) != 0)
        return -1;
    snprintf(p->code, sizeof(p->code), "%08u", (unsigned)v);
    p->made = time(NULL);
    p->failures = 0;
    snprintf(line, sizeof(line), "%s\n", p->code);
    if (pn_local_write_file_atomic(p->path, line, strlen(line), 0600) != 0) {
        fprintf(stderr, "local: cannot write %s\n", p->path);
        return -1;
    }
    fprintf(stderr, "local: pairing code %.4s %.4s (%s; valid %d min; "
                    "`pantheon local pair-code` prints the current one)\n",
            p->code, p->code + 4, why, p->ttl_s / 60);
    return 0;
}

static void *ticker_main(void *arg)
{
    pn_pairing *p = arg;

    while (!p->stop) {
        sleep(1);
        if (!p->stop)
            pn_pairing_tick(p, time(NULL));
    }
    return NULL;
}

pn_pairing *pn_pairing_start(const char *code_path, int ttl_s, int run_thread)
{
    pn_pairing *p = calloc(1, sizeof(*p));

    if (!p)
        return NULL;
    pthread_mutex_init(&p->lock, NULL);
    p->ttl_s = ttl_s > 0 ? ttl_s : PN_PAIR_DEFAULT_TTL_S;
    if (snprintf(p->path, sizeof(p->path), "%s", code_path) >= (int)sizeof(p->path) ||
        rotate_locked(p, "new") != 0) {
        pthread_mutex_destroy(&p->lock);
        free(p);
        return NULL;
    }
    if (run_thread && pthread_create(&p->ticker, NULL, ticker_main, p) == 0)
        p->ticker_running = 1;
    return p;
}

void pn_pairing_stop(pn_pairing *p)
{
    if (!p)
        return;
    p->stop = 1;
    if (p->ticker_running)
        pthread_join(p->ticker, NULL);
    unlink(p->path);
    pthread_mutex_destroy(&p->lock);
    memset(p, 0, sizeof(*p));
    free(p);
}

/* Digits of `in` (spaces and dashes dropped) into out; -1 on anything else. */
static int normalize_code(const char *in, char out[PN_PAIR_CODE_DIGITS + 1])
{
    int n = 0;

    for (const char *c = in ? in : ""; *c; c++) {
        if (*c == ' ' || *c == '-')
            continue;
        if (*c < '0' || *c > '9' || n >= PN_PAIR_CODE_DIGITS)
            return -1;
        out[n++] = *c;
    }
    out[n] = '\0';
    return n == PN_PAIR_CODE_DIGITS ? 0 : -1;
}

int pn_pairing_redeem(pn_pairing *p, const char *code)
{
    char given[PN_PAIR_CODE_DIGITS + 1] = {0};
    int ok = normalize_code(code, given) == 0, rotated = 0;
    unsigned char diff = 0;

    pthread_mutex_lock(&p->lock);
    for (int i = 0; i < PN_PAIR_CODE_DIGITS; i++)
        diff |= (unsigned char)(given[i] ^ p->code[i]);
    ok = ok && diff == 0 && time(NULL) - p->made < p->ttl_s;
    if (ok) {
        rotate_locked(p, "previous one was used");
    } else if (++p->failures >= PN_PAIR_MAX_FAILURES) {
        rotated = 1;   /* the misses are spent even if the new code's file write failed */
        rotate_locked(p, "too many wrong tries");
    }
    pthread_mutex_unlock(&p->lock);
    return ok ? 0 : rotated ? -2 : -1;
}

void pn_pairing_clear_misses(pn_pairing *p)
{
    pthread_mutex_lock(&p->lock);
    p->failures = 0;
    pthread_mutex_unlock(&p->lock);
}

int pn_pairing_tick(pn_pairing *p, time_t now)
{
    int rotated = 0;

    pthread_mutex_lock(&p->lock);
    if (now - p->made >= p->ttl_s)
        rotated = rotate_locked(p, "previous one expired") == 0;
    pthread_mutex_unlock(&p->lock);
    return rotated;
}

void pn_pairing_current(pn_pairing *p, char out[PN_PAIR_CODE_DIGITS + 1])
{
    pthread_mutex_lock(&p->lock);
    memcpy(out, p->code, PN_PAIR_CODE_DIGITS + 1);
    pthread_mutex_unlock(&p->lock);
}
