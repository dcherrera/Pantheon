/* SPDX-License-Identifier: AGPL-3.0-or-later */
/* Copyright (C) 2026 David (Corey) Herrera */
/**
 * @file   local_caller.h
 * @brief  Who is calling (people registry spec §4.1, §6.1): the auth hook
 *         resolves the bearer key to its person and keeps the result in a
 *         thread-local for the route. Each connection thread answers one
 *         request; the hook clears the slot at the start of every request (the
 *         inline fallback path included), so a caller never outlives its request.
 *
 * @status None
 * @issues None
 * @todo   None
 */

#ifndef PN_LOCAL_CALLER_H
#define PN_LOCAL_CALLER_H

#include "key_store.h"

#ifdef __cplusplus
extern "C" {
#endif

/* The resolved caller of one request. */
typedef struct {
    char        key_id[PN_KEY_ID_MAX];
    pn_key_kind kind;
    char        person[PN_KEY_PERSON_MAX];
    char        device_id[PN_KEY_DEVICE_ID_MAX];  /* "" for inference / unclaimed keys */
    int         admin;
    int         may_load;                         /* effective: admin, or the person's flag */
} pn_local_caller;

/**
 * Set this thread's caller (copied).
 *
 * Thread-safe: yes (thread-local). Reentrant: yes. Signal-safe: no
 */
void pn_local_caller_set(const pn_local_caller *caller);

/**
 * This thread's caller, or NULL when the request had no key (or none is set).
 * Valid until the next set / clear on this thread.
 *
 * Thread-safe: yes (thread-local). Reentrant: yes. Signal-safe: no
 */
const pn_local_caller *pn_local_caller_get(void);

/** Forget this thread's caller. Thread-safe: yes. Reentrant: yes. Signal-safe: no */
void pn_local_caller_clear(void);

#ifdef __cplusplus
}
#endif

#endif /* PN_LOCAL_CALLER_H */
