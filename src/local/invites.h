/* SPDX-License-Identifier: AGPL-3.0-or-later */
/* Copyright (C) 2026 David (Corey) Herrera */
/**
 * @file   invites.h
 * @brief  One-time invite codes (people registry spec §3.3): 8 random digits,
 *         each bound to one person, valid 10 min, at most 16 live per box, used
 *         once. Memory only: a box restart voids them. Redeemed through
 *         POST /local/pair before the console code; the console code's 5th
 *         miss voids every live invite. Codes are never logged.
 *
 * @status None
 * @issues None
 * @todo   None
 */

#ifndef PN_LOCAL_INVITES_H
#define PN_LOCAL_INVITES_H

#include <time.h>

#ifdef __cplusplus
extern "C" {
#endif

#define PN_INVITES_MAX        16
#define PN_INVITE_TTL_S       600
#define PN_INVITE_CODE_DIGITS 8
#define PN_INVITE_PERSON_MAX  16

typedef struct pn_invites pn_invites;

/**
 * An empty invite table.
 * @param ttl_s  lifetime of a code (PN_INVITE_TTL_S when <= 0)
 * @return the table, or NULL
 *
 * Thread-safe: yes. Reentrant: yes. Signal-safe: no
 */
pn_invites *pn_invites_new(int ttl_s);

/** Wipe and free. NULL-safe. Thread-safe: no. Reentrant: no. Signal-safe: no */
void pn_invites_free(pn_invites *inv);

/**
 * Issue a code for `person` at time `now` (expired codes are dropped first).
 * @param code_out  receives the 8 digits
 * @return 0, -1 when 16 are live, -2 on a bad person or an RNG failure
 *
 * Thread-safe: yes (internal lock). Reentrant: no. Signal-safe: no
 */
int pn_invites_issue(pn_invites *inv, const char *person, time_t now,
                     char code_out[PN_INVITE_CODE_DIGITS + 1]);

/**
 * Redeem a code (spaces and dashes ignored; constant-time over the table). A
 * match is used up.
 * @param person_out  receives the invite's person on a match
 * @return 0 on a live match, -1 otherwise
 *
 * Thread-safe: yes (internal lock). Reentrant: no. Signal-safe: no
 */
int pn_invites_redeem(pn_invites *inv, const char *code, time_t now,
                      char person_out[PN_INVITE_PERSON_MAX]);

/** Void every live code. Thread-safe: yes (internal lock). Reentrant: no. Signal-safe: no */
void pn_invites_void_all(pn_invites *inv);

/**
 * Void the codes of one person. @return the number voided
 *
 * Thread-safe: yes (internal lock). Reentrant: no. Signal-safe: no
 */
int pn_invites_void_person(pn_invites *inv, const char *person);

/**
 * Drop codes expired at `now`. @return the number dropped
 *
 * Thread-safe: yes (internal lock). Reentrant: no. Signal-safe: no
 */
int pn_invites_tick(pn_invites *inv, time_t now);

/** Live codes at `now`. Thread-safe: yes (internal lock). Reentrant: yes. Signal-safe: no */
int pn_invites_live(pn_invites *inv, time_t now);

#ifdef __cplusplus
}
#endif

#endif /* PN_LOCAL_INVITES_H */
