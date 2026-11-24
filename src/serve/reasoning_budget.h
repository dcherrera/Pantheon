/* SPDX-License-Identifier: AGPL-3.0-or-later */
/* Copyright (C) 2026 David (Corey) Herrera */
/**
 * @file   reasoning_budget.h
 * @brief  The per-turn reasoning plan: where the AUTO-path reasoning cap
 *         force-closes <think>, and the room it keeps for the answer.
 *
 *         max_tokens counts reasoning and answer together. A budget the size of
 *         max_tokens (CLI 8000 vs Argus 8192) left ~190 tokens for the tool call
 *         after a forced close, so a write/edit was cut and refused. And the close
 *         landed on exactly token N, mid-sentence. The plan fixes both:
 *
 *           reserve = min(reserve, max_tokens / 2)      (default PN_REASONING_RESERVE_DEFAULT)
 *           hard    = min(budget, max_tokens - reserve)  the close fires here regardless
 *           soft    = hard - min(grace, hard / 8)        from here, the close fires
 *                                                        on the first line end
 *
 *         grace is PN_REASONING_LINE_GRACE_DEFAULT, or the PN_REASONING_LINE_GRACE
 *         env (0 = close exactly at hard, the old cut point). The max_tokens / 2
 *         cap keeps a small request from losing its whole budget. The plan never
 *         raises max_tokens: that is the client's own limit.
 *
 *         Header-only: batch_sched.c is linked into many test binaries on its own.
 *
 * @status Used at the budget setup of both serve decode paths (sg_run_completion,
 *         sg_run_completion_batched) and by the batched driver's per-step check.
 * @issues None
 * @todo   None
 */

#ifndef PN_SERVE_REASONING_BUDGET_H
#define PN_SERVE_REASONING_BUDGET_H

#include <limits.h>
#include <stdbool.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#ifdef __cplusplus
extern "C" {
#endif

/* 3072: Qwen3.6 think blocks measured at 4.3k-4.8k fit in 8192 - 3072 = 5120,
 * and ~3k tokens (~10 KB) holds an ordinary write/edit call. */
#define PN_REASONING_RESERVE_DEFAULT    3072
/* 256: Qwen thinking breaks lines every few dozen tokens, so a line end almost
 * always arrives inside the window; it costs at most 256 tokens of thinking. */
#define PN_REASONING_LINE_GRACE_DEFAULT 256

typedef struct {
    int hard;   /* reasoning tokens at which the close fires regardless; 0 = no cap */
    int soft;   /* from this many reasoning tokens on, a line end fires the close */
} pn_reasoning_plan;

/**
 * The plan for one turn. budget <= 0 → {0, 0} (no cap). reserve <= 0 →
 * PN_REASONING_RESERVE_DEFAULT. max_tokens <= 0 → no reserve clamp. grace < 0
 * → 0.
 *
 * Thread-safe: yes
 * Reentrant:   yes
 * Signal-safe: yes
 */
static inline pn_reasoning_plan pn_reasoning_plan_make(int budget, int max_tokens,
                                                       int reserve, int grace)
{
    pn_reasoning_plan p = { 0, 0 };
    int hard = budget, win;

    if (budget <= 0)
        return p;
    if (reserve <= 0)
        reserve = PN_REASONING_RESERVE_DEFAULT;
    if (max_tokens > 0) {
        if (reserve > max_tokens / 2)
            reserve = max_tokens / 2;
        if (hard > max_tokens - reserve)
            hard = max_tokens - reserve;
        if (hard < 1)
            hard = 1;
    }
    if (grace < 0)
        grace = 0;
    win = hard / 8 < grace ? hard / 8 : grace;
    p.hard = hard;
    p.soft = hard - win;
    return p;
}

/**
 * PN_REASONING_LINE_GRACE when it is an integer in [0, 1<<20], else
 * PN_REASONING_LINE_GRACE_DEFAULT.
 *
 * Thread-safe: yes (getenv only)
 * Reentrant:   yes
 * Signal-safe: no
 */
static inline int pn_reasoning_line_grace(void)
{
    const char *s = getenv("PN_REASONING_LINE_GRACE");
    char *end = NULL;
    long v;

    if (!s || !*s)
        return PN_REASONING_LINE_GRACE_DEFAULT;
    v = strtol(s, &end, 10);
    if (*end != '\0' || v < 0 || v > (1L << 20))
        return PN_REASONING_LINE_GRACE_DEFAULT;
    return (int)v;
}

/**
 * pn_reasoning_plan_make with the process grace (pn_reasoning_line_grace()),
 * logging one stderr line when the reserve lowers the budget.
 *
 * @param model_name  for the log only; NULL logs "?"
 *
 * Thread-safe: yes (getenv only)
 * Reentrant:   yes
 * Signal-safe: no
 */
static inline pn_reasoning_plan pn_reasoning_plan_for_turn(int budget, int max_tokens,
                                                           int reserve,
                                                           const char *model_name)
{
    pn_reasoning_plan p = pn_reasoning_plan_make(budget, max_tokens, reserve,
                                                 pn_reasoning_line_grace());

    if (p.hard > 0 && p.hard < budget)
        fprintf(stderr, "[reason_budget] %s: budget %d -> %d to keep %d of max_tokens %d "
                        "for the answer\n", model_name && *model_name ? model_name : "?",
                budget, p.hard, max_tokens - p.hard, max_tokens);
    return p;
}

/**
 * Whether the reasoning token just counted (the `toks`-th) fires the close:
 * toks >= hard, or toks >= soft and the token ended a line. Always false for a
 * plan with no cap.
 *
 * Thread-safe: yes
 * Reentrant:   yes
 * Signal-safe: yes
 */
static inline bool pn_reasoning_should_close(const pn_reasoning_plan *p, int toks,
                                             bool line_end)
{
    if (!p || p->hard <= 0)
        return false;
    return toks >= p->hard || (toks >= p->soft && line_end);
}

/**
 * Whether a decoded token piece ends a line (holds a '\n'). NULL → false.
 *
 * Thread-safe: yes
 * Reentrant:   yes
 * Signal-safe: yes
 */
static inline bool pn_reasoning_piece_ends_line(const char *piece)
{
    return piece && strchr(piece, '\n') != NULL;
}

/**
 * How many tokens one speculative cycle may still accept before the close
 * window opens: soft - toks - 1. A draft run must not step over a possible
 * close, since the close needs the trip token's own logits. INT_MAX when the
 * plan has no cap.
 *
 * Thread-safe: yes
 * Reentrant:   yes
 * Signal-safe: yes
 */
static inline int pn_reasoning_draft_room(const pn_reasoning_plan *p, int toks)
{
    if (!p || p->hard <= 0)
        return INT_MAX;
    return p->soft - toks - 1;
}

#ifdef __cplusplus
}
#endif

#endif /* PN_SERVE_REASONING_BUDGET_H */
