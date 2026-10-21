/* SPDX-License-Identifier: AGPL-3.0-or-later */
/* Copyright (C) 2026 David (Corey) Herrera */
/**
 * @file pn_tt_issue.c
 * @brief Implementation of the issue-shape selector (see pn_tt_issue.h for the whole
 *   argument, including the WC/UC store-ordering seam).
 *
 * @status Green.
 * @issues None
 * @todo None
 */

#ifdef USE_TT

#include "pn_tt_issue.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "pn_tt_name_list.h"

#define PN_TT_ISSUE_ENV "PN_TT_ISSUE"

/** Bytes one WORD32 access moves. The other unit is FT_TT_NOC_COPY_UNIT_BYTES. */
#define PN_TT_ISSUE_WORD_BYTES 4u

/** Bytes one BLOCK16 access moves — the same 16 the L1 alignment rule is written in. */
#define PN_TT_ISSUE_BLOCK_BYTES 16u

pn_tt_issue_shape pn_tt_issue_shape_selected(void)
{
    static int cached = -1;

    if (cached < 0) {
        const char *value = getenv(PN_TT_ISSUE_ENV);

        /* DEFAULT `wide` SINCE 2026-08-27, AND THE FLIP IS ITS OWN CONFIRMATION ROUND rather
         * than an inference off the ladder that unblocked it. `wide` was a 5.7% REGRESSION
         * while every issue paid a second TLB re-aim at 8.9 us against the 25 accesses' 0.8;
         * with the window bank absorbing both aims it is a win, and the round that flipped it
         * measured decode 0.1391 (word, three arms, all three to four decimals — a 0.00%
         * control-vs-control drift) against 0.1381/0.1378 (wide): **+0.83%, -1.15 ms/token**.
         * Prefill 165.19/165.27 -> 166.52 tok/s on the `batched prefill —` line's own
         * denominator, **+0.80%** against a 0.048% drift, the gain in the `sequential` term.
         * Every arm `cmp` IDENTICAL at 189 B / 80fc6259, and the two 256-token soaks are
         * byte-identical to each other at 587 B / abca1fa9. The instrument that says the knob
         * armed is the issue clock: 10.0-10.1 -> 5.8-6.1 us/op. See pn_tt_issue.h. */
        cached = (int)PN_TT_ISSUE_WIDE;
        if (value && *value) {
            if (strcmp(value, "word") == 0)
                cached = (int)PN_TT_ISSUE_WORD;
            else if (strcmp(value, "wide") == 0)
                cached = (int)PN_TT_ISSUE_WIDE;
            else
                fprintf(stderr,
                        "pn_tt: %s=%s is not word|wide — using the default (wide)\n",
                        PN_TT_ISSUE_ENV, value);
        }
    }
    return (pn_tt_issue_shape)cached;
}

const char *pn_tt_issue_shape_string(pn_tt_issue_shape shape)
{
    return shape == PN_TT_ISSUE_WIDE ? "wide" : "word";
}

pn_tt_issue_shape pn_tt_issue_shape_for(const ft_tt_runtime *runtime, const ft_tt_queue *queue)
{
    if (pn_tt_issue_shape_selected() != PN_TT_ISSUE_WIDE)
        return PN_TT_ISSUE_WORD;
    /* A DEMOTION, NOT A FAILURE. A queue placed where a 16-byte write cannot reach it would
     * otherwise refuse its first parameter block and take the whole family to the host — a
     * loud failure, but a needless one when the shipped shape is right there. */
    if (!runtime || !runtime->bulk_l1.write || !ft_tt_queue_issue_is_wide_addressable(queue))
        return PN_TT_ISSUE_WORD;
    return PN_TT_ISSUE_WIDE;
}

const ft_tt_noc_access *pn_tt_issue_access(const ft_tt_runtime *runtime,
                                           pn_tt_issue_shape shape)
{
    if (!runtime)
        return NULL;
    return shape == PN_TT_ISSUE_WIDE ? &runtime->bulk_l1 : &runtime->noc;
}

uint32_t pn_tt_issue_unit_bytes(pn_tt_issue_shape shape)
{
    return shape == PN_TT_ISSUE_WIDE ? PN_TT_ISSUE_BLOCK_BYTES : PN_TT_ISSUE_WORD_BYTES;
}

uint64_t pn_tt_issue_accesses(pn_tt_issue_shape shape, uint32_t payload_bytes)
{
    /* The head bump is ALWAYS one WORD32 access, on both shapes, because nothing else can
     * carry a 4-byte write. That single word is why `wide` is 7 and not 6. */
    return (uint64_t)payload_bytes / pn_tt_issue_unit_bytes(shape) + 1u;
}

bool pn_tt_issue_red_head_first(void)
{
    static int cached = -1;

    if (cached < 0) {
        /* SELECTS, NOT CONTAINS. `pn_tt_name_list_contains()` answers TRUE for an UNSET list
         * by design, so arming a red arm with it arms it on every run — measured on silicon
         * 2026-08-25, and it cost a cross-agent bisect the wrong commit. */
        cached = pn_tt_name_list_selects(getenv("PN_TT_RED"), "issue_head_first") ? 1 : 0;
    }
    return cached != 0;
}

#endif /* USE_TT */
