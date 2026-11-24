/* SPDX-License-Identifier: AGPL-3.0-or-later */
/* Copyright (C) 2026 David (Corey) Herrera */
/**
 * @file batch_sched_slots.c
 * @brief Per-architecture session storage for the batch scheduler — see
 *        batch_sched_slots.h for the two storage kinds and why they differ.
 *
 * CARD_BLOCK forwards one-to-one to the pn_qwen35moe_* slot API (the code the
 * scheduler called directly before this file existed). HOST_KV is the llama
 * family: its session is the slot's own pn_state, so most operations are
 * no-ops and a step is one model->forward per active slot.
 *
 * @status Complete
 * @issues None
 * @todo None
 */

#include "batch_sched_slots.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "batch_sched.h"
#include "../core/mem/pn_session_host_fit.h"
#include "../core/session_snapshot.h"
#include "../models/qwen35moe.h"
#include "tensor/tensor.h"

bs_slot_store bs_slot_store_for(const pn_model *model)
{
    if (!model)
        return BS_SLOT_STORE_NONE;
    if (pn_qwen35moe_owns_state(model))
        return BS_SLOT_STORE_CARD_BLOCK;
    /* The snapshot family list is the enumerated "whole per-session state is
     * a plain causal KV cache" list (session_snapshot.c) — exactly the archs
     * whose pn_state is a pn_llama_kv_cache. */
    if (pn_session_snapshot_supported(model->config.arch))
        return BS_SLOT_STORE_HOST_KV;
    return BS_SLOT_STORE_NONE;
}

bool bs_slots_supported(const pn_model *model)
{
    return bs_slot_store_for(model) != BS_SLOT_STORE_NONE;
}

int bs_slots_precheck(const pn_model *model, int n_slots, const char *model_name)
{
    if (bs_slot_store_for(model) != BS_SLOT_STORE_HOST_KV)
        return 0;   /* CARD_BLOCK is checked by the carve itself, on the dies */
    if (pn_session_host_check(model, n_slots, NULL, 0) == 0)
        return 0;
    fprintf(stderr, "[batch_sched] cannot hold %d host-KV sessions for '%s' "
            "- lower --max-sessions or --ctx-cap\n", n_slots,
            model_name && model_name[0] ? model_name : "?");
    return -1;
}

#ifdef USE_CUDA

static bool host_kv(const pn_batch_sched *sched)
{
    return sched->slot_store == BS_SLOT_STORE_HOST_KV;
}

/* HOST_KV carve: one logits row per slot (the KV already lives in each slot's
 * pn_state, allocated by bs_create). */
static int host_kv_carve(pn_batch_sched *sched)
{
    for (int i = 0; i < sched->cap; i++) {
        sched->host_logits[i] = malloc((size_t)sched->n_vocab * sizeof(float));
        if (!sched->host_logits[i]) {
            fprintf(stderr, "[batch_sched] carve: no host memory for slot %d's "
                    "logits row\n", i);
            bs_slots_free(sched);
            return -1;
        }
    }
    uint64_t per = pn_session_host_bytes(&sched->model->config);
    fprintf(stderr, "[batch_sched] session blocks for '%s': host KV, %llu MiB "
            "per session (ctx %d), %d sessions = %llu MiB host; 0 MiB on the "
            "card\n", sched->model_name[0] ? sched->model_name : "?",
            (unsigned long long)(per >> 20), pn_session_ctx_cap(sched->model),
            sched->cap, (unsigned long long)((per * (uint64_t)sched->cap) >> 20));
    return 0;
}

int bs_slots_carve(pn_batch_sched *sched)
{
    if (host_kv(sched))
        return host_kv_carve(sched);
    if (pn_qwen35moe_decode_slots_carve(sched->model, sched->slots[0].state,
                                        sched->cap) != 0)
        return -1;
    /* Optional: a load that cannot fit the checkpoints serves without them.
     * Two per slot (turn + divergence) when they fit, else the turn ones. */
    sched->div_ckpt_on = false;
    sched->turn_ckpt_on = false;
    if (bs_turn_ckpt_enabled()) {
        if (bs_div_ckpt_wanted() &&
            pn_qwen35moe_slot_checkpoints_carve(sched->model,
                                                sched->slots[0].state,
                                                2 * sched->cap) == 0)
            sched->div_ckpt_on = true;
        sched->turn_ckpt_on = sched->div_ckpt_on ||
            pn_qwen35moe_slot_checkpoints_carve(sched->model,
                                                sched->slots[0].state,
                                                sched->cap) == 0;
    }
    return 0;
}

void bs_slots_free(pn_batch_sched *sched)
{
    for (int i = 0; i < BS_MAX_SLOTS; i++) {
        free(sched->host_logits[i]);
        sched->host_logits[i] = NULL;
    }
}

int bs_slots_bind(pn_batch_sched *sched, pn_state *state, int row)
{
    if (host_kv(sched))
        return 0;   /* the slot's pn_state IS its storage */
    return pn_qwen35moe_decode_slot_bind(sched->model, state, row, sched->cap);
}

/* HOST_KV step: one forward per active slot, in row order, each into its own
 * state. The same model->forward the single-user path calls, so a session's
 * tokens are byte-identical to serving it alone. */
static int host_kv_step(pn_batch_sched *sched, const uint32_t *tokens,
                        pn_state **states, const int *rows, int n,
                        float **logits_out)
{
    pn_model *model = sched->model;
    for (int u = 0; u < n; u++) {
        int row = rows ? rows[u] : u;
        if (row < 0 || row >= sched->cap || !sched->host_logits[row])
            return -1;
        tm_tensor *t = model->forward(model, tokens[u], states[u]);
        if (!t)
            return -1;
        memcpy(sched->host_logits[row], tm_data_f32(t),
               (size_t)sched->n_vocab * sizeof(float));
        tm_free(t);
        logits_out[u] = sched->host_logits[row];
    }
    return 0;
}

int bs_slots_step(pn_batch_sched *sched, const uint32_t *tokens,
                  pn_state **states, const int *rows, int n,
                  float **logits_out)
{
    if (host_kv(sched))
        return host_kv_step(sched, tokens, states, rows, n, logits_out);
    return pn_qwen35moe_decode_slots_rows(sched->model, tokens, states, rows, n,
                                          logits_out);
}

int bs_slots_pd_cycle(pn_batch_sched *sched, void *cursor, const uint32_t *tokens,
                      pn_state **states, const int *rows, int n, float **logits_out,
                      int pieces)
{
    if (host_kv(sched))
        return -1;   /* the overlap exists only for the card-resident path */
    return pn_qwen35moe_pd_cycle_cuda(sched->model, (pn_qwen35moe_pd_cursor *)cursor, tokens,
                                      states, rows, n, logits_out, pieces);
}

void bs_slots_pd_phase_ms(void *cursor, double *ms)
{
    pn_qwen35moe_pd_phase_ms((const pn_qwen35moe_pd_cursor *)cursor, ms);
}

int bs_slots_state_to_host(pn_batch_sched *sched, pn_state *state)
{
    if (host_kv(sched))
        return 0;   /* the host KV is the only copy */
    return pn_qwen35moe_gpu_state_to_host(sched->model, state);
}

int bs_slots_row_holds_state(pn_batch_sched *sched, int row, pn_state *state)
{
    if (host_kv(sched))
        return 1;   /* nothing but this slot ever writes its pn_state */
    return pn_qwen35moe_slot_holds_state(sched->model, row, state);
}

int bs_slots_row_ssm_to_host(pn_batch_sched *sched, int row, pn_state *state)
{
    if (host_kv(sched))
        return 0;   /* no recurrent image */
    return pn_qwen35moe_decode_slot_ssm_to_host(sched->model, row, state);
}

void bs_slots_forget(pn_batch_sched *sched, pn_state *state)
{
    if (!host_kv(sched))
        pn_qwen35moe_gpu_forget_state(sched->model, state);
}

void bs_slots_prefill_drain(pn_batch_sched *sched)
{
    if (!host_kv(sched))
        pn_qwen35moe_prefill_drain_cuda(sched->model);
}

int bs_slots_save_live(pn_batch_sched *sched, pn_state *state, int row)
{
    bs_slots_prefill_drain(sched);
    return pn_qwen35moe_host_kv_mirror_eager()
        ? bs_slots_state_to_host(sched, state)
        : bs_slots_bind(sched, state, row);
}

int bs_slots_live_resumable(pn_batch_sched *sched, pn_state *state, int row)
{
    if (pn_qwen35moe_host_kv_mirror_eager())
        return bs_slots_state_to_host(sched, state) == 0;
    return bs_slots_row_holds_state(sched, row, state);
}

int bs_slots_checkpoint_save(pn_batch_sched *sched, pn_state *state, int ck)
{
    if (host_kv(sched))
        return -1;   /* no recurrent image, no card block */
    return pn_qwen35moe_slot_checkpoint_save_at(sched->model, state, ck);
}

int bs_slots_checkpoint_restore(pn_batch_sched *sched, pn_state *state,
                                int row, int ck, int pos)
{
    if (host_kv(sched))
        return -1;
    return pn_qwen35moe_slot_checkpoint_restore_from(sched->model, state, row,
                                                     ck, pos);
}

void bs_slots_mtp_rewind(pn_batch_sched *sched, pn_state *state)
{
    if (host_kv(sched) || !sched->mtp)
        return;
    (void)pn_qwen35moe_mtp_rewind(sched->model, state);
}

#endif /* USE_CUDA */
