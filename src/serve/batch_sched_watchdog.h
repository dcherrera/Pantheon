/* SPDX-License-Identifier: AGPL-3.0-or-later */
/* Copyright (C) 2026 David (Corey) Herrera */
/**
 * @file   batch_sched_watchdog.h
 * @brief  A decode slot's owner waits for its next token with a watchdog: the
 *         wait wakes on a short poll to ask whether the client is still there,
 *         and gives up after a stall limit with no token. Either way the owner
 *         gets control back and releases the slot, so a vanished client or a
 *         stream the driver stopped feeding can never hold a slot forever.
 *
 *         Before this the owner blocked in bs_slot_pop_token with no timeout
 *         and checked the client only between tokens: a turn the driver
 *         stopped stepping kept its slot (and its open stream) until restart.
 *
 * @status None
 * @issues None
 * @todo   None
 */

#ifndef PN_SERVE_BATCH_SCHED_WATCHDOG_H
#define PN_SERVE_BATCH_SCHED_WATCHDOG_H

#include <stdint.h>

#include "batch_sched.h"

#ifdef __cplusplus
extern "C" {
#endif

/* Outcome of bs_slot_pop_token_watched. */
typedef enum {
    BS_POP_TOKEN    =  1,   /* *tok holds the next token */
    BS_POP_FINISHED =  0,   /* the driver finished the slot (EOS / budget) */
    BS_POP_ERROR    = -1,   /* a step failed or the scheduler shut down */
    BS_POP_ABORTED  = -2,   /* should_abort said stop (client gone, unload) */
    BS_POP_STALLED  = -3,   /* no token for stall_ms */
} bs_pop_status;

/* What the watched wait checks while it waits. */
typedef struct {
    int  (*should_abort)(void *ud);   /* may be NULL */
    void  *abort_ud;
    int    poll_ms;                   /* how often to ask; <= 0 = 1000 */
    int    stall_ms;                  /* no-token limit; <= 0 = never */
} bs_pop_watch;

/**
 * The decode stall limit in ms: PN_DECODE_STALL_MS when set (0 disables the
 * watchdog), else 120000. Read once.
 *
 * Thread-safe: yes. Reentrant: yes. Signal-safe: no
 */
int bs_decode_stall_ms(void);

/**
 * bs_slot_pop_token with a watchdog. Waits for slot `idx`'s next token,
 * waking every poll_ms to call should_abort, and returns BS_POP_ABORTED when it
 * says stop or BS_POP_STALLED after stall_ms with no token. Queued tokens are
 * always drained first. Does not release the slot: the caller does, exactly
 * once, whatever the outcome.
 *
 * @param sched  Scheduler
 * @param idx    The caller's slot
 * @param tok    Out: the token on BS_POP_TOKEN
 * @param watch  Abort check and limits (NULL = wait forever, like
 *               bs_slot_pop_token)
 * @return       A bs_pop_status
 *
 * Thread-safe: yes (takes sched->mu)
 * Reentrant:   no (one owner per slot)
 * Signal-safe: no
 */
bs_pop_status bs_slot_pop_token_watched(pn_batch_sched *sched, int idx, uint32_t *tok,
                                        const bs_pop_watch *watch);

#ifdef __cplusplus
}
#endif

#endif /* PN_SERVE_BATCH_SCHED_WATCHDOG_H */
