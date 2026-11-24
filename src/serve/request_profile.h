/* SPDX-License-Identifier: AGPL-3.0-or-later */
/* Copyright (C) 2026 David (Corey) Herrera */
/**
 * @file request_profile.h
 * @brief Per-request wall-clock phase profiler for the serve request path.
 *        Attributes one HTTP chat-completion request across transport, JSON
 *        parse, routing, session admission, slot wait, tokenize, prefix-cache
 *        lookup, prefill, first-token sampling and decode.
 *
 * WHY THREAD-LOCAL: the serve handles every connection on its own detached
 * pthread (serve_command.c spawn_connection), so a thread-local record needs no
 * plumbing through the ~8 call frames between the accept loop and the decode
 * loop, and cannot be raced by a concurrent request.
 *
 * OFF BY DEFAULT. Enabled with PN_REQPROF=1; when disabled every entry point is
 * a single cached-int test and no clock is read.
 *
 * @status Complete. Measurement instrumentation for per-request cost.
 * @issues None
 * @todo None
 */
#ifndef PANTHEON_REQUEST_PROFILE_H
#define PANTHEON_REQUEST_PROFILE_H

/**
 * Phases of one chat-completion request, in the order the serve executes them.
 * Each phase is a disjoint wall-clock span; the phases plus PN_REQ_UNATTRIBUTED
 * (total minus the sum) account for the whole server-side request.
 */
typedef enum {
    PN_REQ_HTTP_READ = 0,   /* accept -> request line + headers + body read     */
    PN_REQ_JSON_PARSE,      /* jr_parse + parse_chat_request                    */
    PN_REQ_ROUTE,           /* route_model + tool-dialect swap                  */
    PN_REQ_SESSION_ADMIT,   /* always 0 now: batched admission runs inside the
                             * slot claim (SLOT_WAIT); kept so the "admit"
                             * column of existing profiles keeps its place     */
    PN_REQ_SLOT_WAIT,       /* bs_slot_acquire_or_wait: session admission +
                             * queue for a decode slot                         */
    PN_REQ_TOKENIZE,        /* build_prompt: chat template render + tokenize    */
    PN_REQ_CACHE_LOOKUP,    /* bs_prefill_resume_sys / bs_prefill_begin         */
    PN_REQ_PREFILL,         /* sg_prefill_range_sched over the delta            */
    PN_REQ_FIRST_SAMPLE,    /* sample token 0 from the prefill logits           */
    PN_REQ_DECODE,          /* token 0 emitted -> stream finished               */
    PN_REQ_PHASE_COUNT
} pn_req_phase;

/**
 * @return non-zero when PN_REQPROF=1 in the environment. Cached after the first
 *         call; safe to call on any hot path.
 *
 * Thread-safe: yes (idempotent getenv, benign race on the cache)
 * Reentrant: yes
 * Signal-safe: no
 */
int pn_reqprof_enabled(void);

/**
 * Start profiling the request on THIS thread. Clears all phase accumulators.
 * A second call without an intervening pn_reqprof_end simply restarts.
 */
void pn_reqprof_begin(void);

/** Open a phase span. Nested/overlapping spans of the same phase are ignored. */
void pn_reqprof_phase_start(pn_req_phase phase);

/** Close the open span for `phase` and add its duration to that phase's total. */
void pn_reqprof_phase_end(pn_req_phase phase);

/**
 * Record the request's shape for the report line.
 *
 * @param method       HTTP method (borrowed, copied)
 * @param path         HTTP path (borrowed, copied)
 */
void pn_reqprof_note_route(const char *method, const char *path);

/**
 * Record what the prefix cache did for this request — the single most important
 * field for telling a warm session from a cold one.
 *
 * @param has_sid      non-zero when the client supplied a session id
 * @param prompt_tok   total prompt tokens after templating
 * @param delta_start  first token index actually prefilled (0 == full prefill)
 * @param sys_len      system-prefix boundary length in tokens
 */
void pn_reqprof_note_cache(int has_sid, int prompt_tok, int delta_start,
                           int sys_len);

/** Record the number of tokens generated (reported in the summary line). */
void pn_reqprof_note_generated(int generated_tok);

/**
 * Close the request and print ONE grep-friendly summary line to stderr:
 *
 *   [reqprof] POST /v1/chat/completions total=21800.0ms http=0.4 json=1.2 ...
 *             | sid=1 prompt_tok=812 delta=0 prefilled=812 gen=300
 *
 * No-op when profiling is disabled or pn_reqprof_begin was never called.
 */
void pn_reqprof_end(void);

#endif /* PANTHEON_REQUEST_PROFILE_H */
