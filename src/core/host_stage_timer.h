/* SPDX-License-Identifier: AGPL-3.0-or-later */
/* Copyright (C) 2026 David (Corey) Herrera */
/**
 * @file host_stage_timer.h
 * @brief Per-request host-time accounting of the decode loop, by stage (`PN_HOST_TIMING=1`).
 *
 * WHY. A batched decode step is GPU forward + dense-row D2H (the timed step) and then host
 * work: sampling (grammar mask, softmax, draw), reasoning bookkeeping, the ring push, the
 * lock and GPU-turn handoffs. On a 2-core host that work is several ms a token, and the
 * user-visible `[req]` tok/s pays it. This splits it so a cut can be priced before it is
 * built.
 *
 * HOW. Each thread binds one accumulator (pn_ht_bind) and stamps stages into it. The
 * driver binds the slot it is sampling, so the sampler's own stages land on that request;
 * the owner binds a local one. Off (the default) every stamp is one predictable branch.
 * Timestamps are CLOCK_MONOTONIC ns (vDSO, ~25 ns a read).
 *
 * @status Instrument only.
 * @issues None
 * @todo None
 */
#ifndef PN_HOST_STAGE_TIMER_H
#define PN_HOST_STAGE_TIMER_H

#include <stdint.h>

typedef enum {
    /* Driver thread, per step (charged to every slot the step ran). */
    PN_HT_GATE = 0,      /* waiting for the gate (cv_work) */
    PN_HT_ACQUIRE,       /* taking the GPU turn */
    PN_HT_STEP,          /* bs_slots_step: forward + logits D2H (the timed step) */
    PN_HT_RELOCK,        /* re-taking sched->mu after the step */
    PN_HT_MTP_PUSH,      /* bs_spec_after_plain_step (MTP head backlog) */
    PN_HT_RELEASE,       /* GPU-turn release + stats line */
    /* Driver thread, per slot. */
    PN_HT_SAMPLE,        /* pn_sample_next_token, whole */
    PN_HT_REASON,        /* driver_reason_step */
    PN_HT_ARM,           /* driver_arm_full_logits */
    PN_HT_EMIT,          /* driver_emit_token + signal */
    /* Sampler sub-stages (inside PN_HT_SAMPLE). */
    PN_HT_S_MASK_SELECT, /* grammar shortlist top-256 select */
    PN_HT_S_MASK_DECIDE, /* shortlist verdicts + row rewrite */
    PN_HT_S_MASK_FULL,   /* chunk / full-vocabulary mask (shortlist undecided) */
    PN_HT_S_TOPN,        /* top-N row verdict (dense_row) */
    PN_HT_S_SURGERY,     /* client surgery + repetition penalty */
    PN_HT_S_GREEDY,      /* temperature 0 argmax */
    PN_HT_S_SPARSE,      /* sparse gather + sparse draw */
    PN_HT_S_FULL_PROBS,  /* sampler_full_probs (dense softmax, top-k, top-p) */
    PN_HT_S_FULL_DRAW,   /* the dense cumulative scan */
    PN_HT_S_MARGIN,      /* the margin draw (sampler_margin_draw.h), decided or refused */
    PN_HT_S_ADVANCE,     /* grammar walk advance over the drawn token */
    /* Owner thread, per token. */
    PN_HT_O_POP_WAIT,    /* bs_slot_pop_token_watched (waiting) */
    PN_HT_O_DETOK,       /* decode_token */
    PN_HT_O_LOGPROB,     /* logprob take/skip */
    PN_HT_O_EMIT,        /* em_feed: stop / tool parse + delta (SSE) */
    PN_HT_O_SSE,         /* the delta callback alone (SSE write + flush) */
    PN_HT_O_RLD,         /* repetition-loop detector */
    PN_HT_N
} pn_ht_stage;

typedef struct {
    uint64_t ns[PN_HT_N];
    uint32_t n[PN_HT_N];
    uint64_t first_launch_ns;  /* driver: this slot's first step launch */
    uint64_t last_launch_ns;   /* driver: its latest step launch */
    uint64_t last_step_ns;     /* driver: that step's duration */
    uint64_t last_deposit_ns;  /* driver: end of its latest deposit */
    uint64_t step_cpu_ns;      /* driver thread CPU time inside PN_HT_STEP */
    uint64_t run_delay0_ns;    /* driver thread schedstat run_delay at the first step */
    uint64_t run_delay_ns;     /* ... and its growth to the latest deposit */
    uint32_t steps;            /* steps this slot ran */
    uint32_t width_sum;        /* sum of the step widths (slots per step) */
    uint32_t masked_rows;      /* draws with a live grammar mask */
    uint32_t full_rows;        /* draws through sampler_full_probs */
} pn_host_timing;

extern int pn_ht_state;   /* -1 unread, 0 off, 1 on */
void pn_ht_init(void);

/** True when PN_HOST_TIMING=1. */
static inline int pn_ht_on(void)
{
    if (__builtin_expect(pn_ht_state < 0, 0))
        pn_ht_init();
    return pn_ht_state;
}

/** CLOCK_MONOTONIC in ns (0 when timing is off). */
uint64_t pn_ht_now(void);

/** This thread's CPU time in ns. */
uint64_t pn_ht_thread_cpu(void);

/** This thread's schedstat run_delay (ns spent runnable, waiting for a CPU); 0 if unreadable. */
uint64_t pn_ht_run_delay(void);

/** Bind (or with NULL unbind) this thread's accumulator. */
void pn_ht_bind(pn_host_timing *acc);
pn_host_timing *pn_ht_bound(void);

/** Charge now - t0 to `stage` of the bound accumulator; returns now (chainable). */
uint64_t pn_ht_lap(pn_ht_stage stage, uint64_t t0);

/** Charge `ns` to `stage` of `acc` directly. */
void pn_ht_add(pn_host_timing *acc, pn_ht_stage stage, uint64_t ns);

/** Sum `src` into `dst` (stage totals and counters; not the launch stamps). */
void pn_ht_merge(pn_host_timing *dst, const pn_host_timing *src);

/** Count a draw kind on the bound accumulator (masked / full row). */
void pn_ht_note_row(int masked, int full);

/** Stage name for the report. */
const char *pn_ht_stage_name(pn_ht_stage stage);

/** Print one `[host]` line for a finished request: per-token ms of every stage, the
 *  driver's off-step time, and the owner/driver scheduling figures. */
void pn_ht_report(const pn_host_timing *drv, const pn_host_timing *own, int slot,
                  int generated, double decode_ms, double first_gap_ms, double tail_gap_ms,
                  uint64_t owner_cpu_ns, uint64_t owner_run_delay_ns);

#endif /* PN_HOST_STAGE_TIMER_H */
