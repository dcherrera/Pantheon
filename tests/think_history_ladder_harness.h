/* SPDX-License-Identifier: AGPL-3.0-or-later */
/* Copyright (C) 2026 David (Corey) Herrera */
/**
 * @file think_history_ladder_harness.h
 * @brief Host harness for the think_history ladder tests: a hand-assembled
 *        batch scheduler and prefix cache over small hand-built qwen35moe
 *        states, with the pn_state lifecycle stubbed (no model, no GPU).
 *
 * Include once, from the test's only translation unit: it defines the link
 * stubs engine.c and the tokenizer would otherwise provide. Same approach as
 * test_prefix_cache_sys.c, whose stubs these mirror.
 *
 * @status None
 * @issues None
 * @todo None
 */

#ifndef PN_TESTS_THINK_HISTORY_LADDER_HARNESS_H
#define PN_TESTS_THINK_HISTORY_LADDER_HARNESS_H

#include <limits.h>
#include <pthread.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "../src/core/prefix_cache.h"
#include "../src/core/session_snapshot.h"
#include "../src/core/think_history.h"
#include "../src/models/qwen35moe.h"
#include "../src/models/qwen35moe_internal.h"
#include "../src/serve/batch_sched.h"

static int failures = 0;
#define CHECK(cond, msg) do { \
    if (cond) { printf("  PASS: %s\n", msg); } \
    else { printf("  FAIL: %s (%s:%d)\n", msg, __FILE__, __LINE__); failures++; } \
} while (0)

/* ---- small fp32 dual state ---- */

enum {
    H_N_SSM = 2, H_N_ATTN = 1, H_CONV_M1 = 3, H_CONV_DIM = 8, H_STATE = 4,
    H_VHEADS = 2, H_HEADS_KV = 2, H_HEAD_DIM = 32, H_MAX_SEQ = 96,
};

static pn_qwen35moe_state *harness_make_state(void)
{
    pn_qwen35moe_state *s = calloc(1, sizeof(*s));
    s->n_ssm = H_N_SSM;
    s->n_attn = H_N_ATTN;
    s->ssm = calloc(H_N_SSM, sizeof(*s->ssm));
    s->kv = calloc(H_N_ATTN, sizeof(*s->kv));
    for (int i = 0; i < H_N_SSM; i++) {
        pn_qwen35moe_ssm_state *st = &s->ssm[i];
        st->conv_dim = H_CONV_DIM;
        st->conv_minus_1 = H_CONV_M1;
        st->state_size = H_STATE;
        st->n_v_heads = H_VHEADS;
        st->conv_state = calloc((size_t)H_CONV_M1 * H_CONV_DIM, sizeof(float));
        st->ssm_state = calloc((size_t)H_STATE * H_STATE * H_VHEADS,
                               sizeof(float));
    }
    pn_qwen35moe_kv *kv = &s->kv[0];
    kv->n_head_kv = H_HEADS_KV;
    kv->head_dim = H_HEAD_DIM;
    kv->max_seq = H_MAX_SEQ;
    size_t n = (size_t)H_HEADS_KV * H_MAX_SEQ * H_HEAD_DIM;
    kv->k = calloc(n, sizeof(float));
    kv->v = calloc(n, sizeof(float));
    return s;
}

/* ---- link stubs (engine.c, the tokenizer, qwen35moe_forward.c) ---- */

unsigned pn_qwen35moe_kv_next_content_gen(void)
{
    static unsigned counter;
    return ++counter;
}

int pn_session_ctx_cap(const pn_model *model)
{
    return model && model->config.n_ctx > 0 ? model->config.n_ctx : INT_MAX;
}

char *ft_decode(ft_tokenizer *tok, int *ids, int length)
{
    (void)tok; (void)ids; (void)length;
    return NULL;
}

int ft_get_model_type(ft_tokenizer *tok)
{
    (void)tok;
    return -1;
}

char *gpt2_bytes_decode(const char *input)
{
    (void)input;
    return NULL;
}

pn_state *pn_state_new(pn_model *model)
{
    pn_state *st = calloc(1, sizeof(*st));
    st->config = &model->config;
    st->kv_cache = harness_make_state();
    return st;
}

void pn_state_free(pn_state *state)
{
    if (!state)
        return;
    pn_qwen35moe_state *s = (pn_qwen35moe_state *)state->kv_cache;
    for (int i = 0; i < s->n_ssm; i++) {
        free(s->ssm[i].conv_state);
        free(s->ssm[i].ssm_state);
    }
    free(s->kv[0].k);
    free(s->kv[0].v);
    free(s->ssm);
    free(s->kv);
    free(s);
    free(state);
}

void pn_state_reset(pn_state *state)
{
    state->pos = 0;
}

/* ---- scheduler helpers ---- */

/* A scheduler with `cap` idle slots; only what the ladder touches is set. */
static pn_batch_sched *harness_sched(pn_model *model, int cap)
{
    pn_batch_sched *s = calloc(1, sizeof(*s));
    s->model = model;
    s->cap = cap;
    pthread_mutex_init(&s->mu, NULL);
    pthread_cond_init(&s->cv_gpu, NULL);
    pthread_cond_init(&s->cv_work, NULL);
    pthread_cond_init(&s->cv_idle, NULL);
    for (int i = 0; i < cap; i++) {
        bs_slot *sl = &s->slots[i];
        sl->slot_idx = i;
        sl->status = BS_SLOT_IDLE;
        sl->gpu_idx = -1;
        pthread_cond_init(&sl->cv, NULL);
        sl->state = pn_state_new(model);
    }
    return s;
}

/* Finalize's warm-history publication (the real one needs a GPU decode
 * slot): the slot holds `sid`'s committed history hist[0, n) at pos n. */
static void harness_publish_warm(bs_slot *sl, const char *sid,
                                 const uint32_t *hist, int n)
{
    snprintf(sl->sid, sizeof(sl->sid), "%s", sid);
    if (sl->hist_cap < n) {
        sl->hist = realloc(sl->hist, (size_t)n * sizeof(uint32_t));
        sl->hist_cap = n;
    }
    memcpy(sl->hist, hist, (size_t)n * sizeof(uint32_t));
    sl->hist_n = n;
    sl->state->pos = n;
}

/* The slot's committed-token log as the driver writes it: the prompt then
 * the generated tokens. */
static void harness_seed_committed(bs_slot *sl, const uint32_t *prompt,
                                   int n_prompt, const uint32_t *gen, int n_gen)
{
    int n = n_prompt + n_gen;
    if (sl->ct_cap < n) {
        sl->ct = realloc(sl->ct, (size_t)n * sizeof(uint32_t));
        sl->ct_cap = n;
    }
    memcpy(sl->ct, prompt, (size_t)n_prompt * sizeof(uint32_t));
    memcpy(sl->ct + n_prompt, gen, (size_t)n_gen * sizeof(uint32_t));
    sl->ct_n = n;
}

/* pn_prefix_cache_acquire_bounded with its options spelled out. */
static pn_state *harness_acquire(pn_prefix_cache *c, const char *sid,
                                 const uint32_t *tokens, int n, int sys_len,
                                 int ceiling, bool drop_last,
                                 pn_prefix_acquire_result *out)
{
    pn_prefix_acquire_opts opts = { sys_len, ceiling, drop_last };
    return pn_prefix_cache_acquire_bounded(c, sid, tokens, n, &opts, out);
}

#endif /* PN_TESTS_THINK_HISTORY_LADDER_HARNESS_H */
