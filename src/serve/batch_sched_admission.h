/* SPDX-License-Identifier: AGPL-3.0-or-later */
/* Copyright (C) 2026 David (Corey) Herrera */
/**
 * @file batch_sched_admission.h
 * @brief Session admission for the batch scheduler (internal to batch_sched.c):
 *        who holds each slot, how many session families hold slots, whether a
 *        request may be seated, and the idle reaper.
 *
 * A slot holds a session's KV on the card, so seating a new session on a slot
 * another session holds erases that session. The scheduler therefore admits at
 * most max_sessions session families (session_family.h) and never takes a slot
 * another family holds; the next new family gets BS_ACQUIRE_AT_CAPACITY.
 *
 * A slot is held by the session that claimed it for the running turn
 * (claim_sid), else by whatever session its warm state belongs to: warm history
 * (sid), system-prefix snapshot (sys_sid), or a turn checkpoint (ckpt/partial).
 * A slot held by nobody is "unheld" (cold).
 *
 * @status Complete
 * @issues None
 * @todo None
 */
#ifndef PN_BATCH_SCHED_ADMISSION_H
#define PN_BATCH_SCHED_ADMISSION_H

#include <stdbool.h>

#include "batch_sched.h"

/** The session holding `sl`, or NULL when unheld. Caller holds sched->mu. */
const char *bs_slot_session_owner(const bs_slot *sl);

/** Does some slot (idle or mid-turn) belong to `sid`'s family? Under mu. */
bool bs_family_holds_slot_locked(const pn_batch_sched *sched, const char *sid);

/** Distinct session families holding slots. Under mu. */
int bs_family_count_locked(const pn_batch_sched *sched);

/** Slots nobody holds (idle, or running a session-less turn). Under mu. */
int bs_unheld_count_locked(const pn_batch_sched *sched);
/**
 * May a family that already holds a slot take one more unheld slot (a second
 * copy of its cache)?  Only while the unheld slots left afterwards still seat
 * every family admission could still let in:
 * unheld - 1 >= max_sessions - families.  With max_sessions == cap this never
 * allows it.  Otherwise the family waits for its own slot.  Two Argus agents
 * each holding 2 of 4 slots locked out new sessions at 3/4 families
 * (2026-11-10).  Under mu.
 */
bool bs_family_may_take_extra_locked(const pn_batch_sched *sched);
/** Could a new session family be seated now (under max_sessions, and a slot
 *  nobody holds exists)? Under mu. */
bool bs_admission_seat_free_locked(const pn_batch_sched *sched);

/**
 * May a request for `sid` (NULL/"" = no session) be seated? BS_ACQUIRE_OK when
 * its family already holds a slot, or when it is under max_sessions and some
 * slot is unheld (idle now, or running a session-less turn that leaves it
 * unheld); else BS_ACQUIRE_AT_CAPACITY. Under mu.
 */
bs_acquire_result bs_admission_check_locked(const pn_batch_sched *sched,
                                            const char *sid);

/**
 * Reap idle slots of other families: a slot idle for session_idle_ttl_ms whose
 * holder has no session watch loses its warm state (as bs_session_release), so
 * a dead client cannot hold a seat forever. `sid`'s own family is never
 * reaped. Takes mu itself; calls the watch hook without it.
 */
void bs_admission_reap_idle(pn_batch_sched *sched, const char *sid,
                            double now_ms);

/**
 * Evictable seats (sched->sessions_evictable): take the least recently active
 * idle slot held by another family than `sid`'s (NULL = anonymous) and with no
 * session watch, dropping its warm state as the reaper does, so the caller's
 * admission can succeed. The evicted session simply re-prefills on its next
 * turn. Takes mu itself; calls the watch hook without it.
 * @return true when a seat was freed
 */
bool bs_admission_evict_lru_idle(pn_batch_sched *sched, const char *sid);

/**
 * Drop every session's warm state from slot `sl` (sid, history, projections,
 * turn checkpoints, system-prefix snapshot). The caller has reserved the slot
 * (status PREFILL), so no other thread writes it. Defined in batch_sched.c.
 */
void bs_slot_drop_session(pn_batch_sched *sched, bs_slot *sl);

#endif /* PN_BATCH_SCHED_ADMISSION_H */
