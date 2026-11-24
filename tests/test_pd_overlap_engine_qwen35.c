/* SPDX-License-Identifier: AGPL-3.0-or-later */
/* Copyright (C) 2026 David (Corey) Herrera */
/* Engine exactness gate for the prefill/decode stage overlap
 * (QWEN35_BS_PD_OVERLAP).
 *
 * K decoders and one prefill, run two ways on the same model:
 *   A (sequential): NSTEP plain decode steps (decode_slots_rows), then the
 *     wide pipelined prefill of the prompt -> logits LA.
 *   B (overlap): a prefill cursor over the same prompt; NSTEP overlap cycles
 *     (pd_cycle, PDE_PIECES layers per board per cycle) with a quiesce and a
 *     plain step at QUIESCE_AT, then pd_finish -> logits LB.
 * Then the prefilled session decodes GREEDY_N tokens in both.
 * PASS = every decode step's logits (FNV-1a per user) and greedy tokens agree,
 * LA == LB bytewise, and the prefilled session's greedy decode agrees.
 *
 * Falsifiers (must FAIL): build with -DPN_PDO_RED_SHARED_SCRATCH (decode lane
 * shares the prefill's scratch) or -DPN_PDO_RED_NO_HOP_FENCE (the chunk hop
 * overwrites stage 1's input before stage 1 issued all of the previous chunk;
 * run it with QWEN35_GPU_SPLIT_LAYER=18: on the even 20 + 20 split the stages
 * move in lockstep and the fence never binds).
 *
 * Build (CUDA host): make USE_CUDA=1 tests/test_pd_overlap_engine_qwen35
 * Run: QWEN35_GPU_DIES=0,1,2,3 QWEN35_TP=2 QWEN35_KV_Q4=1 QWEN35_BS_PD_OVERLAP=1 \
 *        QWEN35_MAX_SEQ=4096 ./tests/test_pd_overlap_engine_qwen35 <model.gguf>
 * Exit 0 = PASS. NOT part of the CPU test suite (needs the model + 4 CUDA dies).
 */
#include "../include/pantheon.h"
#include "../src/core/engine.h"
#include "../src/models/qwen35moe.h"
#include "tensor/tensor.h"
#include <inttypes.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#ifndef USE_CUDA
int main(void)
{
    fprintf(stderr, "built without USE_CUDA\n");
    return 1;
}
#else

#define K           3      /* decoders */
#define SEED_LEN    300    /* each decoder's prompt */
#define PROMPT_LEN  1500   /* the overlapped prefill: 12 chunks */
#define CHUNK       128
#define NSTEP       96
#define QUIESCE_AT  40
#define GREEDY_N    8

static int argmax(const float *x, int n)
{
    int best = 0;
    for (int i = 1; i < n; i++)
        if (x[i] > x[best]) best = i;
    return best;
}

static uint64_t fnv(const float *x, int n)
{
    const unsigned char *p = (const unsigned char *)x;
    uint64_t h = 1469598103934665603ull;
    for (size_t i = 0; i < (size_t)n * sizeof(float); i++)
        h = (h ^ p[i]) * 1099511628211ull;
    return h;
}

static void make_prompt(uint32_t *tok, int n, int seed, int n_vocab)
{
    for (int i = 0; i < n; i++)
        tok[i] = (uint32_t)((1000 + seed * 7919 + i * 37) % n_vocab);
}

/* Pipelined prefill of tok[0, n) into `st`; the last logits into `out`. */
static int pipelined(pn_model *model, pn_state *st, const uint32_t *tok, int n, float *out,
                     int n_vocab)
{
    tm_tensor *t = NULL;
    int rc = pn_qwen35moe_prefill_range_pipelined_cuda(model, tok, n, st, CHUNK, NULL, NULL,
                                                       &t);
    if (rc == 0 && t && out)
        memcpy(out, tm_data_f32(t), (size_t)n_vocab * sizeof(float));
    if (t)
        tm_free(t);
    return rc;
}

typedef struct {
    uint64_t step_hash[NSTEP][K];
    uint32_t step_tok[NSTEP][K];
    float   *pf_logits;
    uint32_t pf_greedy[GREEDY_N];
    uint64_t pf_hash[GREEDY_N];
} arm_out;

/* One arm. overlap = 0: sequential, 1: cursor. States are fresh per arm; rows
 * 0..K-1 hold the decoders, row K the prefilled session. 0 or -1. */
static int run_arm(pn_model *model, int overlap, int pieces, arm_out *o, int n_vocab)
{
    pn_state *dec[K];
    pn_state *pf = pn_state_new(model);
    uint32_t seed[SEED_LEN], prompt[PROMPT_LEN], tok[K];
    int rows[K];
    float *lo[K];
    float *tmp = malloc((size_t)n_vocab * sizeof(float));
    if (!pf || !tmp)
        return -1;
    for (int u = 0; u < K; u++) {
        dec[u] = pn_state_new(model);
        rows[u] = u;
        make_prompt(seed, SEED_LEN, u + 1, n_vocab);
        if (!dec[u] || pipelined(model, dec[u], seed, SEED_LEN, tmp, n_vocab) != 0 ||
            pn_qwen35moe_decode_slot_bind(model, dec[u], u, K + 1) != 0)
            return -1;
        tok[u] = (uint32_t)argmax(tmp, n_vocab);
    }
    make_prompt(prompt, PROMPT_LEN, 99, n_vocab);

    pn_qwen35moe_pd_cursor *cur = NULL;
    if (overlap) {
        cur = pn_qwen35moe_pd_begin_cuda(model, pf, prompt, PROMPT_LEN, CHUNK);
        if (!cur) {
            fprintf(stderr, "PDE: the cursor refused (QWEN35_BS_PD_OVERLAP=1, TP2 on 4 dies?)\n");
            return -1;
        }
    }
    int cycles = 0;
    for (int i = 0; i < NSTEP; i++) {
        int rc;
        if (overlap && i == QUIESCE_AT) {
            /* The SRPT / finalize hand-over: retire and drain, then a plain step. */
            if (pn_qwen35moe_pd_quiesce_cuda(cur) != 0)
                return -1;
            rc = pn_qwen35moe_decode_slots_rows(model, tok, dec, rows, K, lo);
        } else if (overlap && !pn_qwen35moe_pd_issued_all(cur)) {
            rc = pn_qwen35moe_pd_cycle_cuda(model, cur, tok, dec, rows, K, lo, pieces);
            cycles++;
        } else {
            rc = pn_qwen35moe_decode_slots_rows(model, tok, dec, rows, K, lo);
        }
        if (rc != 0)
            return -1;
        for (int u = 0; u < K; u++) {
            o->step_hash[i][u] = fnv(lo[u], n_vocab);
            o->step_tok[i][u] = tok[u];
            tok[u] = (uint32_t)argmax(lo[u], n_vocab);
        }
    }
    if (overlap) {
        tm_tensor *t = NULL;
        const int left = pn_qwen35moe_pd_remaining(cur);
        if (pn_qwen35moe_pd_finish_cuda(cur, &t) != 0 || !t)
            return -1;
        memcpy(o->pf_logits, tm_data_f32(t), (size_t)n_vocab * sizeof(float));
        tm_free(t);
        printf("PDE overlap arm: %d cycles at %d piece(s), %d of %d tokens left for finish\n",
               cycles, pieces, left, PROMPT_LEN);
    } else if (pipelined(model, pf, prompt, PROMPT_LEN, o->pf_logits, n_vocab) != 0) {
        return -1;
    }

    /* The prefilled session decodes: its KV and SSM state, not just the last row. */
    if (pn_qwen35moe_decode_slot_bind(model, pf, K, K + 1) != 0)
        return -1;
    pn_state *one[1] = { pf };
    int row[1] = { K };
    uint32_t t1 = (uint32_t)argmax(o->pf_logits, n_vocab);
    for (int i = 0; i < GREEDY_N; i++) {
        float *l1[1];
        o->pf_greedy[i] = t1;
        if (pn_qwen35moe_decode_slots_rows(model, &t1, one, row, 1, l1) != 0)
            return -1;
        o->pf_hash[i] = fnv(l1[0], n_vocab);
        t1 = (uint32_t)argmax(l1[0], n_vocab);
    }

    for (int u = 0; u < K; u++)
        pn_qwen35moe_gpu_forget_state(model, dec[u]);
    pn_qwen35moe_gpu_forget_state(model, pf);
    free(tmp);
    return 0;
}

int main(int argc, char **argv)
{
    if (argc < 2) {
        fprintf(stderr, "usage: %s <model.gguf>\n", argv[0]);
        return 1;
    }
    if (!pn_qwen35moe_pd_overlap_enabled()) {
        fprintf(stderr, "PDE: overlap is off (unset QWEN35_BS_PD_OVERLAP or set it to 1)\n");
        return 1;
    }
    pn_model *model = pn_load(argv[1]);
    if (!model) {
        fprintf(stderr, "load failed\n");
        return 1;
    }
    const int n_vocab = model->config.n_vocab;
    const char *pe = getenv("PDE_PIECES");
    const int pieces = pe && atoi(pe) > 0 ? atoi(pe) : 1;
    static arm_out a, b;
    a.pf_logits = malloc((size_t)n_vocab * sizeof(float));
    b.pf_logits = malloc((size_t)n_vocab * sizeof(float));
    pn_state *geom = pn_state_new(model);
    if (!a.pf_logits || !b.pf_logits || !geom ||
        pn_qwen35moe_decode_slots_carve(model, geom, K + 1) != 0) {
        fprintf(stderr, "PDE: setup failed\n");
        _Exit(1);
    }
    if (run_arm(model, 0, pieces, &a, n_vocab) != 0 ||
        run_arm(model, 1, pieces, &b, n_vocab) != 0) {
        fprintf(stderr, "PDE FAIL: an arm did not run\n");
        fflush(stderr);
        _Exit(1);
    }

    int bad_steps = 0, first_bad = -1;
    for (int i = 0; i < NSTEP; i++)
        for (int u = 0; u < K; u++)
            if (a.step_hash[i][u] != b.step_hash[i][u] || a.step_tok[i][u] != b.step_tok[i][u]) {
                bad_steps++;
                if (first_bad < 0)
                    first_bad = i;
            }
    int differ = 0;
    for (int i = 0; i < n_vocab; i++)
        differ += a.pf_logits[i] != b.pf_logits[i];
    const int pf_same = memcmp(a.pf_logits, b.pf_logits, (size_t)n_vocab * sizeof(float)) == 0;
    const int greedy_same = memcmp(a.pf_greedy, b.pf_greedy, sizeof(a.pf_greedy)) == 0 &&
                            memcmp(a.pf_hash, b.pf_hash, sizeof(a.pf_hash)) == 0;
    uint64_t da = 1469598103934665603ull, db = da;
    for (int i = 0; i < NSTEP; i++)
        for (int u = 0; u < K; u++) {
            da = (da ^ a.step_hash[i][u]) * 1099511628211ull;
            db = (db ^ b.step_hash[i][u]) * 1099511628211ull;
        }
    printf("PDE decode: %d steps x %d users, %d differ (first at step %d); digest "
           "%016" PRIx64 " / %016" PRIx64 "\n", NSTEP, K, bad_steps, first_bad, da, db);
    printf("PDE prefill logits %s (%d of %d differ), hash %016" PRIx64 " / %016" PRIx64 "\n",
           pf_same ? "IDENTICAL" : "DIFFER", differ, n_vocab, fnv(a.pf_logits, n_vocab),
           fnv(b.pf_logits, n_vocab));
    printf("PDE prefilled session greedy %s:", greedy_same ? "same" : "differ");
    for (int i = 0; i < GREEDY_N; i++)
        printf(" %u/%u", a.pf_greedy[i], b.pf_greedy[i]);
    const int pass = bad_steps == 0 && pf_same && greedy_same;
    printf("\nPDE %s\n", pass ? "PASS" : "FAIL");
    fflush(stdout);
    /* CUDA context teardown wedges on the K80: exit without unwinding. */
    _Exit(pass ? 0 : 1);
}
#endif
