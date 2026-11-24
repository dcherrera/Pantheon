/* SPDX-License-Identifier: AGPL-3.0-or-later */
/* Copyright (C) 2026 David (Corey) Herrera */
/**
 * @file think_history.h
 * @brief The request field `think_history` (what the session's KV must hold
 *        of earlier think blocks) and the resume rung a completion took,
 *        reported back in its usage.
 *
 * @status None
 * @issues None
 * @todo None
 */
/* WHY
 *   The warm rungs keep think blocks in the KV that the client no longer
 *   sends: rung 1b (the `vis` projection) resumes over a KV that still holds
 *   every block, because the GDN layers cannot rewind. A client that drops
 *   reasoning on purpose (a reasoning boundary, a per-turn rewind) needs the
 *   KV to be the request as rendered, and has to be able to see which rung
 *   ran to meter it.
 *
 *   as_sent    the KV must equal this request as rendered. The batched ladder
 *              skips every warm rung and resumes at the system prefix (SYS1);
 *              the single-user cache and the TT card checkpoint never resume
 *              past the request's first think block.
 *   drop_last  the client dropped the last turn's think block: the batched
 *              ladder resumes at that turn's start checkpoint (rung 1c) when
 *              the checkpoint's KV is exactly its client tokens; the
 *              single-user cache restores the turn snapshot it saved at the
 *              last think_history turn's start; TT resumes from the card's
 *              band checkpoint, kept at or below turn_start on think_history
 *              turns. Otherwise as as_sent.
 *   (absent)   today's ladder.
 */

#ifndef PN_CORE_THINK_HISTORY_H
#define PN_CORE_THINK_HISTORY_H

#include <stdint.h>
#include <string.h>

/** The request's `think_history` value. */
typedef enum {
    PN_THINK_HISTORY_KEEP = 0,   /* absent or unknown: today's ladder */
    PN_THINK_HISTORY_AS_SENT,
    PN_THINK_HISTORY_DROP_LAST,
} pn_think_history;

/** The rung a completion resumed on (usage `resume`). */
typedef enum {
    PN_RESUME_COLD = 0,      /* full prefill from token 0 */
    PN_RESUME_EXACT,         /* rung 1: committed history is a prefix */
    PN_RESUME_PROJECTION,    /* rung 1b: the vis / vis_think projection */
    PN_RESUME_PARTIAL,       /* an aborted prefill's prefix, or the plain-KV
                              * longest-common-prefix rewind */
    PN_RESUME_TURN_CKPT,     /* rung 1c, or the TT band checkpoint */
    PN_RESUME_DIV_CKPT,      /* rung 1d: the divergence checkpoint */
    PN_RESUME_SYS_PREFIX,    /* rung 2: the system-prefix snapshot (SYS1) */
} pn_resume_kind;

/** The wire spelling of a resume rung. */
static inline const char *pn_resume_kind_name(pn_resume_kind kind)
{
    switch (kind) {
    case PN_RESUME_EXACT:      return "exact";
    case PN_RESUME_PROJECTION: return "projection";
    case PN_RESUME_PARTIAL:    return "partial";
    case PN_RESUME_TURN_CKPT:  return "turn_ckpt";
    case PN_RESUME_DIV_CKPT:   return "div_ckpt";
    case PN_RESUME_SYS_PREFIX: return "sys_prefix";
    case PN_RESUME_COLD:       break;
    }
    return "cold";
}

/** Parse a `think_history` string. Returns 1 and sets *out for a known value,
 *  0 (and *out = KEEP) for NULL or an unknown one. */
static inline int pn_think_history_parse(const char *text, pn_think_history *out)
{
    *out = PN_THINK_HISTORY_KEEP;
    if (!text)
        return 0;
    if (strcmp(text, "as_sent") == 0)
        *out = PN_THINK_HISTORY_AS_SENT;
    else if (strcmp(text, "drop_last") == 0)
        *out = PN_THINK_HISTORY_DROP_LAST;
    else
        return 0;
    return 1;
}

/** The highest position the Tenstorrent multi-turn checkpoint may sit at for this
 *  request: the end of the client's own messages (`turn_start`, before the assistant
 *  header) when it carries think_history, so the next request, which drops or rewrites
 *  this turn's reply, can resume there; otherwise the end of the history the next
 *  request resends (`n_history`, before the forced generation prompt). */
static inline int pn_think_history_tt_ceiling(pn_think_history think_history,
                                              int turn_start, int n_history)
{
    return think_history != PN_THINK_HISTORY_KEEP && turn_start > 0 ? turn_start
                                                                    : n_history;
}

/** Position of the first `think_open` id in tokens[0, n), or n when none (or
 *  think_open < 0: the model has no think token). */
static inline int pn_think_history_first_think(const uint32_t *tokens, int n,
                                               int think_open)
{
    if (!tokens || think_open < 0)
        return n;
    for (int i = 0; i < n; i++)
        if (tokens[i] == (uint32_t)think_open)
            return i;
    return n;
}

#endif /* PN_CORE_THINK_HISTORY_H */
