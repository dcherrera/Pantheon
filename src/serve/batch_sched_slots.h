/* SPDX-License-Identifier: AGPL-3.0-or-later */
/* Copyright (C) 2026 David (Corey) Herrera */
/**
 * @file batch_sched_slots.h
 * @brief Where the batch scheduler keeps each session between steps, per
 *        architecture: the ONLY place batch_sched.c reaches a model family's
 *        session storage.
 *
 * Two storage kinds exist:
 *
 *   CARD_BLOCK  qwen35moe / bonsai. Each session owns a block carved on the
 *               model's dies at load (a ctx_cap KV ring row per attention layer
 *               + an SSM image per SSM layer); one batched forward steps every
 *               active row. Every call forwards to the pn_qwen35moe_* slot API.
 *
 *   HOST_KV     the llama family (llama, qwen2, qwen3, phi, mistral, mixtral).
 *               The CUDA forward (llama_forward_cuda.c) keeps the KV cache in
 *               the pn_state on the host and attends there, so a session's
 *               whole state already lives in its own slot's pn_state. Nothing is
 *               carved on the card; a step runs one forward per active slot,
 *               each into its own state. The per-session cost is HOST RAM
 *               (pn_session_host_bytes: fp32 KV ring + arena, eager), so the
 *               load-time fit check is against host memory.
 *
 * Until 2026-10-27 the scheduler only knew CARD_BLOCK: for a llama-arch model
 * the carve's arch guard returned -1 and every --max-sessions was refused
 * ("cannot carve N session blocks"), so llama models only served with
 * QWEN35_BATCH_SLOTS=0.
 *
 * Every function except bs_slot_store_for / bs_slots_supported must be called
 * owning the scheduler's GPU turn (batch_sched.h turnstile).
 *
 * @status Complete
 * @issues HOST_KV steps the active slots one forward after another (the llama
 *         CUDA forward is single-token), so N concurrent sessions share one
 *         die's throughput; they interleave fairly but do not batch.
 * @todo None
 */
#ifndef PN_BATCH_SCHED_SLOTS_H
#define PN_BATCH_SCHED_SLOTS_H

#include <stdbool.h>
#include <stdint.h>

#include "../core/engine.h"

/* How a model's sessions are stored by the scheduler. */
typedef enum {
    BS_SLOT_STORE_NONE = 0,     /* the scheduler cannot hold this arch's sessions */
    BS_SLOT_STORE_CARD_BLOCK,   /* per-session block carved on the dies */
    BS_SLOT_STORE_HOST_KV,      /* the slot's own pn_state host KV cache */
} bs_slot_store;

struct pn_batch_sched_s;

/** Which storage the scheduler uses for `model` (NONE: no scheduler). */
bs_slot_store bs_slot_store_for(const pn_model *model);

/** Can the batch scheduler serve `model` at all? */
bool bs_slots_supported(const pn_model *model);

/**
 * Load-time fit check, BEFORE any slot state is allocated (HOST_KV states are
 * eager host allocations): pn_session_host_check, the one host-RAM fit check.
 * Returns 0 when `n_slots` sessions fit, -1 after logging the refusal.
 */
int bs_slots_precheck(const pn_model *model, int n_slots, const char *model_name);

/** Carve the scheduler's session storage at load. 0, or -1 after logging. */
int bs_slots_carve(struct pn_batch_sched_s *sched);

/** Free what bs_slots_carve allocated on the host (safe to call twice). */
void bs_slots_free(struct pn_batch_sched_s *sched);

/** Register `state` with row `row` (its slot index). 0 or -1. */
int bs_slots_bind(struct pn_batch_sched_s *sched, pn_state *state, int row);

/**
 * One decode step for `n` sessions: token u into states[u] (row rows[u]).
 * logits_out[u] stays valid until that row's next step. Advances each
 * states[u]->pos by 1. 0 or -1.
 */
int bs_slots_step(struct pn_batch_sched_s *sched, const uint32_t *tokens,
                  pn_state **states, const int *rows, int n,
                  float **logits_out);

/**
 * One decode step exactly as bs_slots_step, with the prefill pieces of
 * `cursor` (a pn_qwen35moe_pd_cursor) issued in each board's idle half,
 * at most `pieces` layers per board. QWEN35_BS_PD_OVERLAP only. 0 or -1.
 */
int bs_slots_pd_cycle(struct pn_batch_sched_s *sched, void *cursor, const uint32_t *tokens,
                      pn_state **states, const int *rows, int n, float **logits_out,
                      int pieces);

/** The last overlap cycle's host time per phase (pn_qwen35moe_pd_phase_ms). */
void bs_slots_pd_phase_ms(void *cursor, double *ms);   /* PN_QWEN35MOE_PD_PHASES entries */

/** Make the state's host mirror authoritative (CARD_BLOCK D2H). 0 or -1. */
int bs_slots_state_to_host(struct pn_batch_sched_s *sched, pn_state *state);

/** Does row `row` still hold `state` exactly as committed? 1 or 0. */
int bs_slots_row_holds_state(struct pn_batch_sched_s *sched, int row,
                             pn_state *state);

/** Copy row `row`'s recurrent image into `state`'s host mirror. 0 or -1. */
int bs_slots_row_ssm_to_host(struct pn_batch_sched_s *sched, int row,
                             pn_state *state);

/** Drop every on-card binding of `state` (host bookkeeping only). */
void bs_slots_forget(struct pn_batch_sched_s *sched, pn_state *state);

/** Drain an in-flight pipelined prefill before handing the GPU over. */
void bs_slots_prefill_drain(struct pn_batch_sched_s *sched);

/**
 * Save a prefill stopped part-way (state->pos = what fully cleared) into row
 * `row`'s own block, so the next forward that touches `state` re-seeds from it
 * (eager mode: the host mirror). Drains the pipeline first. 0 or -1.
 */
int bs_slots_save_live(struct pn_batch_sched_s *sched, pn_state *state, int row);

/**
 * Can a resumed turn continue `state` from row `row` as it stands (the row
 * still holds it at state->pos; eager mode: its host mirror is made
 * authoritative)? 1 or 0.
 */
int bs_slots_live_resumable(struct pn_batch_sched_s *sched, pn_state *state,
                            int row);

/**
 * Save `state`'s live image (just prefilled to state->pos) as card checkpoint
 * `ck` (slot i: turn checkpoint i, divergence checkpoint cap + i; CARD_BLOCK
 * only; HOST_KV has none). 0 or -1.
 */
int bs_slots_checkpoint_save(struct pn_batch_sched_s *sched, pn_state *state,
                             int ck);

/**
 * Put row `row` back at checkpoint `ck` (which must be `state`'s, at KV
 * position `pos`) and set state->pos = pos. Nothing changes on -1.
 */
int bs_slots_checkpoint_restore(struct pn_batch_sched_s *sched,
                                pn_state *state, int row, int ck, int pos);

/**
 * The trunk state of `state` was set back (a checkpoint restore): the MTP draft
 * head drops its backlog and continues from state->pos. No-op without a head.
 */
void bs_slots_mtp_rewind(struct pn_batch_sched_s *sched, pn_state *state);

#endif /* PN_BATCH_SCHED_SLOTS_H */
