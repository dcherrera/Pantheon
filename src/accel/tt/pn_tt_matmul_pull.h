/* SPDX-License-Identifier: AGPL-3.0-or-later */
/* Copyright (C) 2026 David (Corey) Herrera */
/**
 * @file pn_tt_matmul_pull.h
 * @brief Which member-0 descriptor of a matmul chain carries the M2 PULL, and the a3/a4
 *        words it carries.
 *
 * The M2 pull makes member 0's writer wait, after its own share of a split projection, until
 * every member has posted that projection's group sequence; only then does member 0 publish
 * its completion, so a flush awaits member 0 alone (resident_matmul_writer.c `pull_group`).
 *
 * Until L2 EVERY split projection's member-0 descriptor pulled. A flush awaits only the
 * chain's LAST descriptor, so the pulls in front of it bought nothing: each one stopped
 * member 0's writer, filled its two output pages, stalled its compute and then its reader,
 * and member 0 restarted the next projection behind the slowest member. In decode that is
 * 256 of the 513 member-0 descriptors a Qwen3.8 token (in-proj chains of 4, gate+up of 2,
 * attention q/k/v/gate of 4).
 *
 * DEFERRED (`PN_TT_MATMUL_PULL_DEFER=1`, default 0: exact but measured +0.00% on the card, plan
 * 6.2): member 0's descriptor is held on the host
 * until the next projection of the same chain is issued (published with no pull) or the chain
 * is flushed (published with the pull, aimed at the chain's last split sequence). Members run
 * their descriptors in order, so "every member reached the last split sequence" covers every
 * projection of the chain, which is all the flush needs. The data path is untouched: the pull
 * never ordered one projection's reads after another's writes (member 0's reader runs ahead of
 * its writer), so a chain was already a set of independent projections.
 *
 * SWEEP: the writer polls the pulled members either one at a time (the shipped walk: member 1
 * until done, then member 2, ...) or all of them per sweep with one read barrier. a3 bit 30
 * asks for the one-at-a-time walk; it is the control arm (`PN_TT_MATMUL_PULL_SWEEP=0`).
 *
 * @status L2. Card A/B staged, not run.
 * @issues None
 * @todo None
 */
#ifndef PN_TT_MATMUL_PULL_H
#define PN_TT_MATMUL_PULL_H

#include <stdbool.h>
#include <stdint.h>

/** a3 bit 31: RED, skip the pull (resident_queue_protocol.h kPullRedSkip). */
#define PN_TT_PULL_RED_SKIP 0x80000000u
/** a3 bit 30: walk the members one at a time (resident_queue_protocol.h kPullSerial). */
#define PN_TT_PULL_SERIAL 0x40000000u

/** What the chain has issued so far that a later pull must cover. */
typedef struct pn_tt_pull_chain {
    uint32_t members; /**< members of the chain's last split projection; 0/1 = none */
    uint32_t target;  /**< that projection's group sequence */
} pn_tt_pull_chain;

/** The two descriptor words member 0's matmul descriptor carries. */
typedef struct pn_tt_pull_words {
    uint32_t a3; /**< pulled member count | flag bits; 0 = no pull */
    uint32_t a4; /**< the group sequence every pulled member must reach */
} pn_tt_pull_words;

/**
 * Record one projection's issue in the chain: a split over `members` (> 1) becomes the
 * chain's pull target at `sequence`; a solo projection leaves the target where it was.
 *
 * Thread-safe: no    Reentrant: yes    Signal-safe: yes
 */
void pn_tt_pull_chain_note(pn_tt_pull_chain *chain, bool split, uint32_t members,
                           uint32_t sequence);

/**
 * The pull member 0's descriptor carries.
 *
 * @param chain    the chain INCLUDING the projection this descriptor belongs to
 * @param last     true when this is the chain's last member-0 descriptor (the flush awaits it)
 *                 or when pulls are not deferred (every descriptor is then treated as last)
 * @param red_skip set bit 31 (the RED arm: the writer skips the pull)
 * @param serial   set bit 30 (the one-member-at-a-time walk)
 * @return the words; a3 == 0 when nothing is pulled
 *
 * Thread-safe: yes    Reentrant: yes    Signal-safe: yes
 */
pn_tt_pull_words pn_tt_pull_words_for(const pn_tt_pull_chain *chain, bool last, bool red_skip,
                                      bool serial);

#endif /* PN_TT_MATMUL_PULL_H */
