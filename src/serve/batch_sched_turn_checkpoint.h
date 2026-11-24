/* SPDX-License-Identifier: AGPL-3.0-or-later */
/* Copyright (C) 2026 David (Corey) Herrera */
/**
 * @file batch_sched_turn_checkpoint.h
 * @brief Turn-start checkpoint for the batch scheduler: a session can resume
 *        from the start of its last turn when the client drops or rewrites
 *        that turn's reply.
 *
 * The resume ladder resumes a session warm only when the next request extends
 * what the slot committed (rung 1) or the serve's projection of it (rung 1b).
 * Three common requests do neither, and each fell to the system-prefix rung, a
 * re-prefill of the whole conversation:
 *   - a reasoning-only turn (no answer, no tool call): Argus drops the turn,
 *     and the serve projects nothing for it;
 *   - an interrupted turn (Esc mid-decode): the serve committed a partial
 *     reply that the client never resends;
 *   - a prefill abort: the slot was reset.
 * The recurrent (GDN) layers cannot rewind, so the state at the start of the
 * turn has to be saved before the turn decodes. This module saves it when the
 * prompt's own messages are prefilled (just before the assistant opener) and
 * adds rung 1c: if the next request starts with exactly those messages, the
 * slot is put back at that point and only the rest is prefilled.
 *
 * What is saved: the recurrent images and the q4 fp16 sink/window shadows, on
 * the card, in buffers carved with the session blocks (engine:
 * pn_qwen35moe_slot_checkpoint_*). KV rows need no copy: the rows before the
 * checkpoint are never rewritten, so setting the position back is enough.
 *
 * Exactness: a rung-1c resume holds the same state as a session that went
 * straight from the checkpoint to the new request — the dropped turn leaves
 * nothing behind. It differs from a cold prefill only the way every warm
 * resume does (think blocks that rung 1b kept in earlier turns).
 *
 * Gate: QWEN35_TURN_CHECKPOINT, default on since 2026-11-03 (=0 turns it off). Off, every
 * function here is a no-op and the ladder behaves as before.
 *
 * Thread-safety: every function is owner-only (the slot's request thread,
 * which owns the slot), and the ones that touch the card need the GPU turn.
 *
 * Aborted prefill: a client that hangs up mid-prefill leaves the state at a
 * chunk boundary, exactly as an uninterrupted prefill of that prompt prefix
 * would (the batched prefill is bit-exact at any chunking). The slot keeps it
 * (bs_partial_prefill_keep) and the session's next request resumes there when
 * it starts with that prefix — the retry of a session's first turn, which has
 * no checkpoint yet, or the part of any turn's prompt already past its
 * checkpoint.
 *
 * @status Complete
 * @issues None
 * @todo None
 */
#ifndef PN_SERVE_BATCH_SCHED_TURN_CHECKPOINT_H
#define PN_SERVE_BATCH_SCHED_TURN_CHECKPOINT_H

#include <stdbool.h>
#include <stdint.h>

struct pn_batch_sched_s;

/* One slot's turn-start checkpoint (host bookkeeping; the state is on the
 * card). Embedded in bs_slot. */
typedef struct {
    bool      valid;
    char      sid[64];      /* session it belongs to */
    uint32_t *client;       /* the request's tokens [0, client_n): what the
                             * next request must start with */
    int       client_n;
    int       client_cap;
    uint32_t *kv;           /* the tokens committed in the KV [0, kv_n) at the
                             * checkpoint (differs from `client` by the think
                             * blocks rung 1b kept) */
    int       kv_n;
    int       kv_cap;
} bs_turn_ckpt;

/** On unless QWEN35_TURN_CHECKPOINT=0 (read once). */
bool bs_turn_ckpt_enabled(void);

/**
 * Save slot `idx`'s state as its turn-start checkpoint. Call mid-prefill,
 * owning the GPU turn, right after the request's tokens [0, cut) are prefilled
 * (state->pos is the matching KV position). Uses the slot's ct_keep /
 * ct_prompt_from to know what the KV holds. A failure leaves the previous
 * checkpoint alone and is benign (the next miss just re-prefills).
 *
 * @param sid     session id (NULL/empty: nothing is saved)
 * @param prompt  the request's prompt tokens
 * @param cut     client position of the checkpoint (sg_prompt_bounds.turn_start)
 * @return 0 saved, -1 not saved
 */
int bs_turn_ckpt_capture(struct pn_batch_sched_s *sched, int idx,
                         const char *sid, const uint32_t *prompt, int cut);

/**
 * Rung 1c. When slot `idx` holds a checkpoint for `sid` whose client tokens
 * are a strict prefix of `tokens`, put the slot back at it: the slot's state,
 * history and committed-log seeding then describe the checkpoint, and *delta
 * is the client position the prefill continues from. Owning the GPU turn.
 *
 * @return 1 resumed, 0 not applicable (nothing changed)
 */
int bs_turn_ckpt_resume(struct pn_batch_sched_s *sched, int idx,
                        const char *sid, const uint32_t *tokens, int n_ctx,
                        int *delta);

/**
 * Rung 1c for `think_history: "drop_last"`: as bs_turn_ckpt_resume, but only
 * when the checkpoint's KV tokens are exactly its client tokens, so the slot
 * resumes holding the request as rendered (KV = request). A checkpoint taken
 * over a KV that kept think blocks the request lacks is refused (logged).
 *
 * @return 1 resumed, 0 not applicable (nothing changed)
 */
int bs_turn_ckpt_resume_exact(struct pn_batch_sched_s *sched, int idx,
                              const char *sid, const uint32_t *tokens,
                              int n_ctx, int *delta);

/** Does `ck` let session `sid` resume `tokens`? (Reroute scan; no card work.) */
bool bs_turn_ckpt_fits(const bs_turn_ckpt *ck, const char *sid,
                       const uint32_t *tokens, int n_ctx);

/**
 * Keep what an aborted prefill computed (owning the GPU turn, before the slot
 * is released): save the live state into the slot's own block and record the
 * prompt prefix it holds in the slot's `partial` record (client tokens, and
 * the KV tokens: hist[0, ct_keep) ++ prompt[ct_prompt_from, ..)). Refused
 * at or below the system prefix (a resume there would skip the session's SYS1
 * capture, and the SYS1 snapshot already resumes at the boundary).
 *
 * @param sid      session id (NULL/empty: nothing is kept)
 * @param prompt   the aborted request's prompt tokens
 * @param n_prompt their count
 * @param sys_len  the request's system-prefix boundary (0: none)
 * @return 0 kept, -1 not kept (the caller resets or keeps the checkpoint)
 */
int bs_partial_prefill_keep(struct pn_batch_sched_s *sched, int idx,
                            const char *sid, const uint32_t *prompt,
                            int n_prompt, int sys_len);

/**
 * Rung 1c, aborted-prefill form: when slot `idx` kept an aborted prefill for
 * `sid` whose client tokens are a strict prefix of `tokens` and the slot's
 * block still holds that state, continue from it (*delta = the client
 * position). Owning the GPU turn. The record is single-use: the ladder drops
 * it whichever rung runs.
 *
 * @return 1 resumed, 0 not applicable (nothing changed)
 */
int bs_partial_prefill_resume(struct pn_batch_sched_s *sched, int idx,
                              const char *sid, const uint32_t *tokens,
                              int n_ctx, int *delta);

/**
 * Rung 1d, the divergence checkpoint. A context compaction (Argus) keeps the
 * head of the conversation byte-identical (system prompt, task, pinned task
 * reads, later user messages) and rewrites everything from the previous
 * summary on, so every compacted request diverges from the slot's history at
 * a point d well past the system prefix, and the NEXT compaction diverges at
 * or after this one's d. Rung 2 resumed every such request at the system
 * prefix. When the ladder resumes a request at r < d (d - r >= the minimum
 * gain), bs_div_ckpt_plan sets the slot's div_cut = d; the prefill saves the
 * state there into the slot's second card checkpoint (bs_div_ckpt_capture),
 * and a later request that starts with those d tokens resumes from it
 * (bs_div_ckpt_resume) instead of the system prefix. KV rows below d are
 * never rewritten while it is valid: any resume below d drops it.
 *
 * On when the load could carve 2 checkpoints per slot (QWEN35_DIV_CHECKPOINT=0
 * turns it off). Exactness is rung 1c's: the state at d is the one a straight
 * prefill of those tokens produces.
 */
bool bs_div_ckpt_enabled(const struct pn_batch_sched_s *sched);

/** QWEN35_DIV_CHECKPOINT != 0 (read once): carve the second checkpoint set. */
bool bs_div_ckpt_wanted(void);

/** Before the ladder mutates the slot: where `tokens` leave what slot `idx`
 *  holds for `sid` (committed history, else the turn checkpoint); -1 if it
 *  holds none of this session. */
int bs_div_point(const struct pn_batch_sched_s *sched, int idx, const char *sid,
                 const uint32_t *tokens, int n_ctx);

/** After the ladder resumed at client position `delta` (state pos = KV pos):
 *  drop a divergence checkpoint the resume went below, and set div_cut to
 *  `div_point` when re-prefilling past it pays. */
void bs_div_ckpt_plan(struct pn_batch_sched_s *sched, int idx, int div_point,
                      int delta, int n_ctx);

/** The client position this turn's prefill should save the divergence
 *  checkpoint at, or -1 (owner-only; set by bs_div_ckpt_plan). */
int bs_div_ckpt_cut(const struct pn_batch_sched_s *sched, int idx);

/** Save the divergence checkpoint at client position `cut` (owner, GPU turn,
 *  state exactly there). 0 saved, -1 not. */
int bs_div_ckpt_capture(struct pn_batch_sched_s *sched, int idx,
                        const char *sid, const uint32_t *prompt, int cut);

/** Rung 1d: resume slot `idx` at its divergence checkpoint when `tokens`
 *  start with its tokens. 1 resumed (*delta set), 0 not applicable. */
int bs_div_ckpt_resume(struct pn_batch_sched_s *sched, int idx,
                       const char *sid, const uint32_t *tokens, int n_ctx,
                       int *delta);

/** Drop the checkpoint (the slot's state was reset or replaced). */
void bs_turn_ckpt_forget(bs_turn_ckpt *ck);

/** Free the checkpoint's host buffers. */
void bs_turn_ckpt_free(bs_turn_ckpt *ck);

#endif /* PN_SERVE_BATCH_SCHED_TURN_CHECKPOINT_H */
