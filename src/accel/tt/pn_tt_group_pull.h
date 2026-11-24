/* SPDX-License-Identifier: AGPL-3.0-or-later */
/* Copyright (C) 2026 David (Corey) Herrera */
/**
 * @file pn_tt_group_pull.h
 * @brief S1 GROUP COMPLETION BY PULL for the split families (eltwise, reduce + reduce-member,
 *        render widen/narrow, head-norm, attention lanes, gdn CONV/DELTA, the KV append
 *        split): member 0 posts its completion only
 *        after every member reached the descriptor's group sequence, so the host awaits
 *        member 0 alone and the puller issues ONE AWAIT per split op instead of one a member.
 *
 * The card side is foundry resident_group_pull.h, called from the vector, attention, gdn and
 * place writers: the L2 matmul pull (resident_matmul_writer.c `pull_group_sweep`), reads only,
 * every spin bounded and stoppable.
 *
 *   - THE TABLE, on member 0 at control + 0xF900: magic "FTQP", the member count, then
 *     `x | y << 16` per member in member order. Written at the first split and after every
 *     re-arm, and READ BACK; a table that does not read back leaves the family on
 *     per-member awaits.
 *   - THE WORDS: every member's descriptor carries a5 = the family's group sequence, which its
 *     writer stores in control word 56 before it publishes; member 0's carries a3 = the member
 *     count and a4 = that same sequence.
 *   - THE DONE WORD (control byte 0xE0) is outside what `ft_tt_queue_init()` clears and
 *     survives the process, so it is ZEROED on every core (and every re-arm of one) the table
 *     newly names, before any descriptor of the family pulls it. A stale value from an earlier process would satisfy
 *     the pull before the member ran.
 *
 * Exact by construction: the descriptors' parameter blocks are untouched; only who posts
 * completion, and when, moves. A member's done word is written after its last output write
 * was ACKED, so member 0's completion still means every output byte has landed.
 *
 * KNOB `PN_TT_GROUP_PULL`: 1 (default) decode and verify splits; 2 prefill splits as well; 0 =
 * per-member awaits and today's descriptors, byte for byte. Prefill is out by default: its
 * descriptors are long, so an AWAIT saved is a rounding error there, and it is where the
 * members a decode split never uses sit idle and are retired and re-armed. RED
 * `PN_TT_RED=group_pull_short`: member 0's a3 is one member short, so its completion no
 * longer covers the last member.
 *
 * WHAT THE FIRST CARD RUN TAUGHT (p150, 2026-11-16), and what now prevents each:
 *   - a member idle-retired by its own enqueue AFTER member 0 went out with the pull left
 *     member 0 pulling forever (prefill puller timeout, gdn DELTA "error"): every member is
 *     checked enqueueable first (pn_tt_group_pull_decide());
 *   - a re-arm on the same core keeps the cores but not necessarily member 0's L1: a new queue
 *     nonce plants again, table and done words;
 *   - the gdn reader kept a clock in word 56, so member 0 posted before the members finished,
 *     the host rewrote their parameter slot under them and they wedged: the clock moved, and
 *     foundry tt/tools/check_group_done_word.sh refuses any second writer of word 56.
 *
 * @status S1. Host test
 *         tests/test_tt_group_pull.c; card gate pending.
 * @issues A member whose descriptor the card REFUSES (or whose enqueue fails on a NoC error
 *         after member 0's went out) never posts its done word, so member 0 waits out its spin
 *         bound or the host's op timeout before the per-member fallback names the member;
 *         per-member awaits found it at once. Rare, and already a re-arm.
 * @todo None
 */
#ifndef PN_TT_GROUP_PULL_H
#define PN_TT_GROUP_PULL_H

#ifdef USE_TT

#include <stdbool.h>
#include <stdint.h>

#include "pn_tt_residency.h"

#include "tt/ft_tt_queue.h"

typedef struct pn_tt_device pn_tt_device;

/** The widest split family a pull serves (the eltwise decode + wide set). */
#define PN_TT_GROUP_PULL_MAX_MEMBERS 16u
/** The pull table, from the control block (resident_queue_protocol.h kPullTableOffset). */
#define PN_TT_GROUP_PULL_TABLE_OFFSET 0xF900u
#define PN_TT_GROUP_PULL_MAGIC        0x46545150u /* "FTQP" */
/** Control word 56, the writer's done-sequence (kFieldGroupDone). */
#define PN_TT_GROUP_PULL_DONE_OFFSET  0xE0u
/** Table words: magic, count, one per member. */
#define PN_TT_GROUP_PULL_TABLE_WORDS  (2u + PN_TT_GROUP_PULL_MAX_MEMBERS)

/** One family's pull: the table it planted and the sequence it has issued up to. */
typedef struct pn_tt_group_pull {
    uint16_t core_x[PN_TT_GROUP_PULL_MAX_MEMBERS]; /**< the cores the planted table names */
    uint16_t core_y[PN_TT_GROUP_PULL_MAX_MEMBERS];
    uint32_t nonce[PN_TT_GROUP_PULL_MAX_MEMBERS];  /**< each one's queue run nonce then */
    uint32_t members;  /**< how many it names; 0 = never planted */
    uint32_t sequence; /**< the family's last group sequence (a5); never 0 once issued */
    bool knob_read;    /**< `enabled` / `everywhere` hold PN_TT_GROUP_PULL */
    bool enabled;
    bool everywhere;   /**< PN_TT_GROUP_PULL=2: prefill splits pull too */
    bool table_ok;     /**< the table on member 0 read back */
    uint64_t pulled;   /**< split descriptors issued with the pull */
} pn_tt_group_pull;

/** The three descriptor words of one member of a pulled split. */
typedef struct pn_tt_group_pull_words {
    uint32_t a3; /**< member 0: members pulled; 0 elsewhere */
    uint32_t a4; /**< member 0: the sequence they must reach; 0 elsewhere */
    uint32_t a5; /**< every member: the group sequence it posts */
} pn_tt_group_pull_words;

/**
 * The words member `member` of a `members`-wide split carries at group sequence `sequence`.
 *
 * @param red_short  the RED arm: member 0's a3 one member short
 *
 * Thread-safe: yes    Reentrant: yes    Signal-safe: yes
 */
pn_tt_group_pull_words pn_tt_group_pull_words_for(uint32_t member, uint32_t members,
                                                  uint32_t sequence, bool red_short);

/**
 * Build the pull table for `set[0..members)`: magic, count, `x | y << 16` per member.
 *
 * @param out_words  PN_TT_GROUP_PULL_TABLE_WORDS words
 * @return the words used, or 0 when `members` is outside 2..PN_TT_GROUP_PULL_MAX_MEMBERS
 *
 * Thread-safe: yes    Reentrant: yes    Signal-safe: yes
 */
uint32_t pn_tt_group_pull_table(pn_tt_residency *const *set, uint32_t members,
                                uint32_t *out_words);

/**
 * Is the pull live for a family whose members are `set[0..members)`? Reads the knob on first
 * use; plants the table (and zeroes the done word of every core it newly names) the first time
 * it sees this set of cores, and again after a re-arm moved one. Plant only when no member has
 * a descriptor in flight — every caller does it at the top of an issue, after the previous one
 * was awaited.
 *
 * @param family  the family's name, for the one line a plant logs
 * @return true when the table names exactly these cores and read back
 *
 * Thread-safe: no    Reentrant: no    Signal-safe: no
 */
bool pn_tt_group_pull_ready(pn_tt_group_pull *pull, pn_tt_residency *const *set,
                            uint32_t members, const char *family);

/** Does the planted table cover a split of `members`? */
bool pn_tt_group_pull_covers(const pn_tt_group_pull *pull, uint32_t members);

/**
 * Does THIS split go out pulled? The knob and its scope (`decode`: a decode or verify forward
 * is running, pn_tt_group_pull_decode(); prefill splits pull only at PN_TT_GROUP_PULL=2), the
 * table (planted or re-planted here, pn_tt_group_pull_ready()), and every one of `set[0..
 * members)` enqueueable right now — each member's idle retire is run here, before member 0 is
 * sent a pull of it.
 *
 * @param table_members  the family's whole set, which the table names
 * @param members        this split's members, a prefix of it
 *
 * Thread-safe: no    Reentrant: no    Signal-safe: no
 */
bool pn_tt_group_pull_decide(pn_tt_group_pull *pull, bool decode, pn_tt_residency *const *set,
                             uint32_t table_members, uint32_t members, const char *family);

/** Is a decode or verify forward running on `device` (the S1 default scope)? */
bool pn_tt_group_pull_decode(const pn_tt_device *device);

/**
 * The group sequence of the next pulled split (never 0), counted in `pulled`. For a family
 * that builds its own descriptors; pn_tt_group_pull_enqueue() calls it.
 *
 * Thread-safe: no    Reentrant: no    Signal-safe: yes
 */
uint32_t pn_tt_group_pull_next(pn_tt_group_pull *pull);

/**
 * Enqueue one split, member 0 first. `pulled`: every member carries the next group sequence
 * and member 0 the pull; otherwise the plain enqueue, exactly as before the pull. Every member
 * is attempted even after one fails.
 *
 * @param words  member m's sixteen parameter words from words + m * FT_TT_QUEUE_PARAM_WORDS
 * @param bad    set true for every member whose enqueue failed
 * @return 0, or -1 when any enqueue failed
 *
 * Thread-safe: no    Reentrant: no    Signal-safe: no
 */
int pn_tt_group_pull_enqueue(pn_tt_group_pull *pull, bool pulled, pn_tt_residency *const *set,
                             uint32_t members, uint32_t opcode, const uint32_t *words, bool *bad);

/**
 * Await one split. `pulled` and every enqueue succeeded: member 0 alone, and the others are
 * marked settled on its completion (unless the residencies defer their awaits, when
 * pn_tt_group_pull_settle() does it); if member 0 fails, every other member is awaited on its
 * own so the caller can tell which to re-arm, and the table is planted afresh at the next
 * issue. Otherwise every member, in order, as before the pull.
 *
 * @param bad             set true for every member that failed; a member already bad (its
 *                        enqueue failed) is still awaited on the per-member path, as before
 * @param member_seconds  per-member await seconds added in, or NULL
 * @return 0, or -1 when any member failed
 *
 * Thread-safe: no    Reentrant: no    Signal-safe: no
 */
int pn_tt_group_pull_await(pn_tt_group_pull *pull, bool pulled, pn_tt_residency *const *set,
                           uint32_t members, bool *bad, double *member_seconds);

/**
 * The awaits a `defer_await` window skipped, for a set whose splits may have pulled: member 0,
 * then any member with an await of its own pending; a member the pull covered is marked
 * settled on member 0's completion, or awaited on its own if member 0 failed (and the table
 * is then planted afresh at the next issue).
 *
 * @return 0, or -1 when any member failed
 *
 * Thread-safe: no    Reentrant: no    Signal-safe: no
 */
int pn_tt_group_pull_settle(pn_tt_group_pull *pull, pn_tt_residency *const *set,
                            uint32_t members, uint32_t timeout_ms);

#endif /* USE_TT */
#endif /* PN_TT_GROUP_PULL_H */
