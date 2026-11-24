/* SPDX-License-Identifier: AGPL-3.0-or-later */
/* Copyright (C) 2026 David (Corey) Herrera */
/**
 * @file pn_tt_group_pull.c
 * @brief Implementation of the S1 group pull. See pn_tt_group_pull.h.
 *
 * @status S1. Host test tests/test_tt_group_pull.c; card gate pending.
 * @issues None
 * @todo None
 */

#ifdef USE_TT

#include "pn_tt_group_pull.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "pn_tt_device_internal.h"

#include "tt/ft_tt_op_l1_layout.h"

#define PN_TT_GROUP_PULL_ENV "PN_TT_GROUP_PULL"

pn_tt_group_pull_words pn_tt_group_pull_words_for(uint32_t member, uint32_t members,
                                                  uint32_t sequence, bool red_short)
{
    pn_tt_group_pull_words words = { 0u, 0u, sequence };

    if (member == 0u) {
        words.a3 = red_short && members > 0u ? members - 1u : members;
        words.a4 = sequence;
    }
    return words;
}

uint32_t pn_tt_group_pull_table(pn_tt_residency *const *set, uint32_t members,
                                uint32_t *out_words)
{
    uint32_t m;

    if (!set || !out_words || members < 2u || members > PN_TT_GROUP_PULL_MAX_MEMBERS)
        return 0u;
    out_words[0] = PN_TT_GROUP_PULL_MAGIC;
    out_words[1] = members;
    for (m = 0; m < members; m++)
        out_words[2u + m] = (uint32_t)set[m]->core_x | ((uint32_t)set[m]->core_y << 16);
    return 2u + members;
}

/** Was residency `r` — this core AND this arming of it (the queue's run nonce, new at every
 *  start) — part of the table as planted? */
static bool group_pull_planted(const pn_tt_group_pull *pull, const pn_tt_residency *r)
{
    uint32_t m;

    for (m = 0; m < pull->members; m++)
        if (pull->core_x[m] == r->core_x && pull->core_y[m] == r->core_y &&
            pull->nonce[m] == r->queue.run_nonce)
            return true;
    return false;
}

/** Was the table planted for exactly `set[0..members)`, in order, as armed now? A re-arm on
 *  the same core is a new nonce, so it plants again: member 0's L1 is rewritten and read back,
 *  and the re-armed core's done word zeroed. */
static bool group_pull_names(const pn_tt_group_pull *pull, pn_tt_residency *const *set,
                             uint32_t members)
{
    uint32_t m;

    if (pull->members != members)
        return false;
    for (m = 0; m < members; m++)
        if (pull->core_x[m] != set[m]->core_x || pull->core_y[m] != set[m]->core_y ||
            pull->nonce[m] != set[m]->queue.run_nonce)
            return false;
    return true;
}

/**
 * Zero the done word of every residency the old table did not name (a new core, or a re-arm),
 * write the table to member 0 and read it back. The set is recorded either way, so a refusal
 * is logged once per arm and the family stays on per-member awaits until a re-arm.
 */
static bool group_pull_plant(pn_tt_group_pull *pull, pn_tt_residency *const *set,
                             uint32_t members, const char *family)
{
    const ft_tt_noc_access *noc = &set[0]->runtime->noc;
    uint32_t table[PN_TT_GROUP_PULL_TABLE_WORDS];
    uint32_t back[PN_TT_GROUP_PULL_TABLE_WORDS];
    const uint32_t zero = 0u;
    const uint32_t words = pn_tt_group_pull_table(set, members, table);
    bool ok = words != 0u;
    uint32_t m;

    /* ONE WORD, NOT THE 16-BYTE LINE THE PULL READS: the reduce compute thread keeps a clock
     * in control words 57-59 (resident_reduce_compute.c). */
    for (m = 0; ok && m < members; m++)
        if (!group_pull_planted(pull, set[m]) &&
            noc->write(noc->user, set[m]->core_x, set[m]->core_y,
                       FT_TT_OP_QUEUE_BASE + PN_TT_GROUP_PULL_DONE_OFFSET, &zero,
                       sizeof(zero)) != FT_TT_OK)
            ok = false;
    if (ok)
        ok = noc->write(noc->user, set[0]->core_x, set[0]->core_y,
                        FT_TT_OP_QUEUE_BASE + PN_TT_GROUP_PULL_TABLE_OFFSET, table,
                        words * 4u) == FT_TT_OK &&
             noc->read(noc->user, set[0]->core_x, set[0]->core_y,
                       FT_TT_OP_QUEUE_BASE + PN_TT_GROUP_PULL_TABLE_OFFSET, back,
                       words * 4u) == FT_TT_OK &&
             memcmp(table, back, words * 4u) == 0;
    for (m = 0; m < members; m++) {
        pull->core_x[m] = set[m]->core_x;
        pull->core_y[m] = set[m]->core_y;
        pull->nonce[m] = set[m]->queue.run_nonce;
    }
    pull->members = members;
    pull->table_ok = ok;
    fprintf(stderr, "pn_tt: %s group PULL %s — member 0 (core %u) reads %u members' "
                    "done-sequence before posting%s\n",
            family, ok ? "ARMED" : "REFUSED (the table did not read back; per-member awaits)",
            set[0]->core_index, members,
            pn_tt_red("group_pull_short") ? "; RED PROOF: one member SHORT" : "");
    return ok;
}

/** PN_TT_GROUP_PULL, read once per family: 0 off, 2 every split, else (1, unset) decode and
 *  verify splits only. */
static void group_pull_knob(pn_tt_group_pull *pull)
{
    const char *value;

    if (pull->knob_read)
        return;
    value = getenv(PN_TT_GROUP_PULL_ENV);
    pull->enabled = !(value && *value && atoi(value) == 0);
    pull->everywhere = value && *value && atoi(value) == 2;
    pull->knob_read = true;
}

bool pn_tt_group_pull_ready(pn_tt_group_pull *pull, pn_tt_residency *const *set,
                            uint32_t members, const char *family)
{
    if (!pull || !set)
        return false;
    group_pull_knob(pull);
    if (!pull->enabled || members < 2u || members > PN_TT_GROUP_PULL_MAX_MEMBERS)
        return false;
    if (group_pull_names(pull, set, members))
        return pull->table_ok;
    return group_pull_plant(pull, set, members, family);
}

bool pn_tt_group_pull_covers(const pn_tt_group_pull *pull, uint32_t members)
{
    return pull && pull->enabled && pull->table_ok && members > 1u && members <= pull->members;
}

bool pn_tt_group_pull_decide(pn_tt_group_pull *pull, bool decode, pn_tt_residency *const *set,
                             uint32_t table_members, uint32_t members, const char *family)
{
    uint32_t m;

    if (!pull || !set || members < 2u || members > table_members)
        return false;
    group_pull_knob(pull);
    if (!pull->enabled || (!decode && !pull->everywhere) ||
        !pn_tt_group_pull_ready(pull, set, table_members, family) ||
        !pn_tt_group_pull_covers(pull, members))
        return false;
    /* EVERY MEMBER ENQUEUEABLE BEFORE MEMBER 0 IS SENT ITS PULL. The enqueue's idle retire
     * stops a residency idle past its limit, and a member that refused AFTER member 0 went out
     * left member 0 pulling a descriptor that never came (S1 card run: the wide eltwise and
     * DELTA-only gdn members, idle while the narrower splits ran). A member that would refuse
     * sends the split out unpulled, and it fails and re-arms exactly as it did before. */
    for (m = 0; m < members; m++)
        if (!pn_tt_residency_enqueue_ready(set[m]))
            return false;
    return true;
}

bool pn_tt_group_pull_decode(const pn_tt_device *device)
{
    return device && (device->single_row || device->verify_rows != 0u);
}

uint32_t pn_tt_group_pull_next(pn_tt_group_pull *pull)
{
    /* 0 is "post nothing" on the wire, so the sequence skips it on the wrap. */
    if (++pull->sequence == 0u)
        pull->sequence = 1u;
    pull->pulled++;
    return pull->sequence;
}

int pn_tt_group_pull_enqueue(pn_tt_group_pull *pull, bool pulled, pn_tt_residency *const *set,
                             uint32_t members, uint32_t opcode, const uint32_t *words, bool *bad)
{
    const bool red_short = pulled && pn_tt_red("group_pull_short");
    uint32_t m;
    int failed = 0;

    if (pulled)
        (void)pn_tt_group_pull_next(pull);
    /* MEMBER 0 FIRST, as before the pull. A deferred set relies on it: member 0's enqueue is
     * where the ring-depth settle runs, and its completion is what vouches for the members'
     * earlier descriptors before theirs reuse a parameter slot. */
    for (m = 0; m < members; m++) {
        const uint32_t *member_words = words + m * FT_TT_QUEUE_PARAM_WORDS;
        int rc;

        if (pulled) {
            const pn_tt_group_pull_words w =
                pn_tt_group_pull_words_for(m, members, pull->sequence, red_short);

            rc = pn_tt_residency_enqueue_pull(set[m], opcode, member_words, w.a3, w.a4, w.a5);
        } else {
            rc = pn_tt_residency_enqueue(set[m], opcode, member_words, 1u);
        }
        if (rc != 0) {
            bad[m] = true;
            failed = -1;
        }
    }
    return failed;
}

/** One member's await, its seconds added in. */
static bool group_pull_await_one(pn_tt_residency *residency, double *seconds)
{
    const double started = pn_tt_now_seconds();
    const bool ok = pn_tt_residency_await(residency, PN_TT_OP_TIMEOUT_MS) == 0;
    const double elapsed = pn_tt_now_seconds() - started;

    if (seconds)
        *seconds += elapsed;
    residency->seconds += elapsed;
    return ok;
}

/** Member 0 failed with the pull: plant again (table and done words) at the next issue. A
 *  table L1 no longer holds would otherwise refuse every descriptor a re-armed member 0 ran. */
static void group_pull_forget(pn_tt_group_pull *pull)
{
    pull->members = 0u;
    pull->table_ok = false;
}

int pn_tt_group_pull_await(pn_tt_group_pull *pull, bool pulled, pn_tt_residency *const *set,
                           uint32_t members, bool *bad, double *member_seconds)
{
    uint32_t m = 0u;
    int failed = 0;

    for (m = 0u; m < members; m++)
        pulled = pulled && !bad[m];
    m = 0u;
    if (pulled) {
        if (group_pull_await_one(set[0], member_seconds ? &member_seconds[0] : NULL)) {
            /* Deferred, member 0's await is only pending: pn_tt_group_pull_settle() marks the
             * members once it has really completed. */
            for (m = 1u; m < members; m++)
                if (!set[0]->defer_await && set[m]->unawaited != 0u)
                    pn_tt_residency_mark_settled(set[m]);
            return 0;
        }
        bad[0] = true;
        group_pull_forget(pull);
        m = 1u;
    }
    /* EVERY member is awaited when the pull cannot vouch for it, even one already bad, as
     * the families always did: an enqueue that reached the ring has a descriptor the card
     * will run, and leaving it un-awaited desynchronises that member's cursor. */
    for (; m < members; m++)
        if (!group_pull_await_one(set[m], member_seconds ? &member_seconds[m] : NULL))
            bad[m] = true;
    for (m = 0u; m < members; m++)
        failed |= bad[m] ? -1 : 0;
    return failed;
}

/** An await that does not defer, whatever the residency's window says. */
static int group_pull_await_now(pn_tt_residency *residency, uint32_t timeout_ms)
{
    const bool deferring = residency->defer_await;
    int rc;

    residency->defer_await = false;
    rc = pn_tt_residency_await(residency, timeout_ms);
    residency->defer_await = deferring;
    return rc;
}

int pn_tt_group_pull_settle(pn_tt_group_pull *pull, pn_tt_residency *const *set,
                            uint32_t members, uint32_t timeout_ms)
{
    const int first = pn_tt_residency_settle(set[0], timeout_ms);
    int rc = first;
    uint32_t m;

    if (first != 0 && pull->table_ok)
        group_pull_forget(pull);
    for (m = 1u; m < members; m++) {
        if (set[m]->await_pending) {
            if (pn_tt_residency_settle(set[m], timeout_ms) != 0)
                rc = -1;
        } else if (set[m]->unawaited != 0u) {
            /* Covered by member 0's pull when it completed; on its own when it did not. */
            if (first == 0)
                pn_tt_residency_mark_settled(set[m]);
            else if (group_pull_await_now(set[m], timeout_ms) != 0)
                rc = -1;
        }
    }
    return rc;
}

#endif /* USE_TT */
