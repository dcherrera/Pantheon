/* SPDX-License-Identifier: AGPL-3.0-or-later */
/* Copyright (C) 2026 David (Corey) Herrera */
/**
 * @file reasoned_reply_tracker.h
 * @brief Did this reply reason, and is this its first tool call?  The two facts
 *        behind the request field `required_if_reasoned`, shared by both
 *        tool-call grammars (tool_grammar.c, tool_grammar_json.c).
 *
 * A key listed in `required_if_reasoned` is required on a call only when BOTH
 * hold: the reply's think block held at least one non-whitespace char before its
 * close, and the call is the reply's first (the first to reach a NAME).
 *
 * WHAT COUNTS AS REASONING.  Every char the walk sees in its FREE region while the
 * think block the prompt opened is still open, EXCEPT:
 *   - whitespace (the template's empty "<think>\n\n</think>" is not reasoning);
 *   - the chars of the close itself: a "</think>" text match, or the close token
 *     (reasoned_reply_close);
 *   - the opener of a call written before the close (AUTO / named-function walks
 *     engage the skeleton on an opener mid-think; reasoned_reply_call_opened).
 * A partial "</think>" that breaks off was text, so its chars count.  A block the
 * serve force-closed has ended either by its walked close or by
 * reasoned_reply_end; what came before counts either way.  A walk that never
 * called reasoned_reply_begin (no think block opened) is never reasoned.
 *
 * A small POD: the grammar states embed it by value, so clone/copy carry it.
 *
 * @status Used by tool_grammar.c and tool_grammar_json.c.
 * @issues None
 * @todo None
 */
#ifndef PANTHEON_REASONED_REPLY_TRACKER_H
#define PANTHEON_REASONED_REPLY_TRACKER_H

#include <stdbool.h>

#include "think_close_rule.h"

typedef struct {
    bool watching;     /* the prompt's think block is open and not yet closed */
    int  close_term;   /* rolling "</think>" match (text mode) */
    int  nonws;        /* non-whitespace reasoning chars released so far */
    int  calls_named;  /* calls of this reply that reached a NAME */
} reasoned_reply;

/* Only "any at all" and "more than one opener's worth" are ever asked, so the
 * count saturates instead of overflowing on a very long think. */
#define REASONED_REPLY_COUNT_CAP (1 << 20)

static inline void reasoned_reply_count(reasoned_reply *r, int n) {
    r->nonws = (r->nonws > REASONED_REPLY_COUNT_CAP - n) ? REASONED_REPLY_COUNT_CAP
                                                         : r->nonws + n;
}

/** The prompt left a think block open: start watching it. */
static inline void reasoned_reply_begin(reasoned_reply *r) {
    r->watching = true;
    r->close_term = 0;
    r->nonws = 0;
}

/** The block ended without a walked close (a reasoning-end call or budget trip).
 *  A partial "</think>" still held was text, so it counts. */
static inline void reasoned_reply_end(reasoned_reply *r) {
    if (!r->watching) return;
    reasoned_reply_count(r, r->close_term);
    r->close_term = 0;
    r->watching = false;
}

/** The close token arrived: the block is over, the token itself is not reasoning. */
static inline void reasoned_reply_close(reasoned_reply *r) {
    reasoned_reply_end(r);
}

/** Feed one char the walk saw in its FREE region. */
static inline void reasoned_reply_char(reasoned_reply *r, char c, bool token_mode) {
    if (!r->watching) return;
    int held = r->close_term;
    if (think_close_text_step(&r->close_term, c, token_mode)) {
        r->watching = false;   /* the closer's own chars never count */
        return;
    }
    if (r->close_term == held + 1) return;   /* `c` extends the match: held */
    reasoned_reply_count(r, held);   /* a broken partial "</think>": all text */
    if (r->close_term == 0 && c != ' ' && c != '\t' && c != '\n' &&
        c != '\r' && c != '\f' && c != '\v')
        reasoned_reply_count(r, 1);
}

/** A call opener of `opener_len` non-whitespace chars completed in FREE: those
 *  chars were the call, not reasoning, and the block is over. */
static inline void reasoned_reply_call_opened(reasoned_reply *r, int opener_len) {
    if (!r->watching) return;
    r->nonws = (r->nonws > opener_len) ? r->nonws - opener_len : 0;
    r->close_term = 0;
    r->watching = false;
}

/** Did the reply reason? */
static inline bool reasoned_reply_reasoned(const reasoned_reply *r) {
    return r->nonws > 0;
}

/** A call reached its NAME.  Returns true when it owes the reasoned-only keys:
 *  the reply reasoned and this is its first call. */
static inline bool reasoned_reply_call_named(reasoned_reply *r) {
    bool first = (r->calls_named == 0);
    if (r->calls_named < 1 << 30) r->calls_named++;
    return first && reasoned_reply_reasoned(r);
}

#endif /* PANTHEON_REASONED_REPLY_TRACKER_H */
