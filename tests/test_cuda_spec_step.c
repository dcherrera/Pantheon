/* SPDX-License-Identifier: AGPL-3.0-or-later */
/* Copyright (C) 2026 David (Corey) Herrera */
/**
 * @file test_cuda_spec_step.c
 * @brief Exactness gate for the one-cycle CUDA MTP step (spec_mtp_step.h): a generation
 *        driven by pn_qwen35moe_cuda_mtp_draft + pn_qwen35moe_cuda_spec_step must equal
 *        plain decode token for token, greedy and sampled with a seeded RNG.
 *
 * Per prompt (a code one and an open-ended one) and temperature (0 and 0.7; top_p 0.95,
 * top_k 20, seed 1234) both arms prefill the same chat prompt from a fresh state (chunks of 128; the spec arm feeds the MTP head's backlog),
 * draw the first token from the prefill logits, then generate n_gen tokens:
 *   plain — bound to decode slot 0, one pn_qwen35moe_decode_slots step per token;
 *   spec  — loop of draft(k) + spec_step, the serve's call shape (one cycle per call).
 * STEP_TOKEN_ARM=1 adds an INFORMATIONAL third stream: the single-user serve's plain step
 * (pn_forward_step_begin/poll, the async single-token forward sg_forward_step runs), compared
 * with the decode-slots stream; it never changes the gate verdict.
 * Generation does not stop at EOS (both arms run the full n_gen). A run where no draft was
 * accepted fails (it would test nothing), and so does a temp-0.7 run that drew the greedy
 * stream on every prompt (sampling untested). The red arm is PN_MTP_RED=1 (every draft accepted
 * unchecked): it MUST print a DIFFERENT line and GATE: FAIL.
 *
 * Build (CUDA host): make USE_CUDA=1 tests/test_cuda_spec_step
 * Run:   QWEN35_MTP=1 <production env> ./tests/test_cuda_spec_step <model.gguf> [n_gen=64] [k=2]
 * Prints "GATE: BIT-EXACT PASS" or "GATE: FAIL ..." — exit 0 only on PASS.
 * NOT part of the suite (needs the model + a CUDA GPU); run it by hand on a CUDA host.
 *
 * @status complete
 * @issues None
 * @todo None
 */
#include "../include/pantheon.h"
#include "../src/core/chat.h"
#include "../src/core/engine.h"
#include "../src/core/spec_mtp.h"
#include "../src/core/spec_mtp_step.h"
#include "../src/models/qwen35moe.h"
#include "tensor/tensor.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#ifndef USE_CUDA
int main(void) { fprintf(stderr, "built without USE_CUDA\n"); return 1; }
#else

#define STEP_CHUNK   128
#define STEP_MAX_GEN 1024
#define STEP_RECENT  64

/* A code prompt (high acceptance) and an open-ended one (sampling leaves the greedy path). */
static const char *STEP_PROMPTS[] = {
    "Write a C function `int count_words(const char *s)` that counts whitespace-separated "
    "words in a string. Explain the edge cases (empty string, leading and trailing spaces, "
    "tabs and newlines), then give three short test cases with their expected results.",
    "Invent a short story about a lighthouse keeper who finds a strange object on the beach. "
    "Give the keeper a name, describe the object in vivid detail, and end with a twist.",
};
#define STEP_N_PROMPTS 2

typedef struct {
    long cycles, drafted, accepted;
} step_counts;

static pn_sampling_params step_sampler(float temp)
{
    pn_sampling_params sp = pn_sampling_defaults();
    sp.temperature = temp;
    sp.top_p = 0.95f;
    sp.top_k = 20;
    sp.repeat_penalty = 1.0f;
    sp.repeat_last_n = STEP_RECENT;
    sp.rng_seeded = 1;
    sp.rng_state = 1234;
    return sp;
}

/* Fresh state, prefill in chunks (feeding the MTP backlog when asked), first token drawn. */
static pn_state *step_prefill(pn_model *model, const pn_token_seq *seq, int feed_mtp,
                              pn_sampling_params *sp, uint32_t *recent, int *n_recent,
                              uint32_t *first)
{
    pn_state *st = pn_state_new(model);

    if (!st)
        return NULL;
    for (int off = 0; off < seq->n_tokens; off += STEP_CHUNK) {
        int n = seq->n_tokens - off < STEP_CHUNK ? seq->n_tokens - off : STEP_CHUNK;
        tm_tensor *out = pn_qwen35moe_forward_batch_cuda(model, seq->tokens + off, n, st, NULL);
        if (!out ||
            (feed_mtp && pn_qwen35moe_mtp_push_rows(model, st, seq->tokens + off, n) != 0)) {
            if (out)
                tm_free(out);
            fprintf(stderr, "prefill failed at %d\n", off);
            pn_state_free(st);
            return NULL;
        }
        if (off + n >= seq->n_tokens) {
            *first = pn_sample_next_token(tm_data_f32(out), model->config.n_vocab, sp, recent,
                                          *n_recent);
            pn_mtp_recent_push(recent, n_recent, STEP_RECENT, *first);
        }
        tm_free(out);
    }
    return st;
}

/* Plain arm: slot 0, one decode-slots step per token. Returns tokens written, or -1. */
static int run_plain_arm(pn_model *model, const pn_token_seq *seq, float temp, int n_gen,
                         uint32_t *out)
{
    pn_sampling_params sp = step_sampler(temp);
    uint32_t recent[STEP_RECENT], seed;
    int n_recent = 0, n = 0;
    pn_state *st = step_prefill(model, seq, 0, &sp, recent, &n_recent, &seed);

    if (!st || pn_qwen35moe_decode_slot_bind(model, st, 0, 1) != 0) {
        pn_state_free(st);
        return -1;
    }
    while (n < n_gen) {
        float *logits[1];
        pn_state *sts[1] = { st };
        if (pn_qwen35moe_decode_slots(model, &seed, sts, 1, logits) != 0) {
            n = -1;
            break;
        }
        seed = pn_sample_next_token(logits[0], model->config.n_vocab, &sp, recent, n_recent);
        pn_mtp_recent_push(recent, &n_recent, STEP_RECENT, seed);
        out[n++] = seed;
    }
    pn_state_free(st);
    return n;
}

/* Informational arm: the single-user serve's plain step (sg_forward_step's async path). */
static tm_tensor *token_step(pn_model *model, pn_state *st, uint32_t token)
{
    tm_tensor *out = NULL;
    int rc = pn_forward_step_begin(model, token, st);

    if (rc > 0)
        return model->forward(model, token, st);
    if (rc < 0)
        return NULL;
    do
        rc = pn_forward_step_poll(model, st, &out);
    while (rc > 0);
    return rc == 0 ? out : NULL;
}

static int run_token_arm(pn_model *model, const pn_token_seq *seq, float temp, int n_gen,
                         uint32_t *out)
{
    pn_sampling_params sp = step_sampler(temp);
    uint32_t recent[STEP_RECENT], seed;
    int n_recent = 0, n = 0;
    pn_state *st = step_prefill(model, seq, 0, &sp, recent, &n_recent, &seed);

    if (!st)
        return -1;
    while (n < n_gen) {
        tm_tensor *logits = token_step(model, st, seed);
        if (!logits) {
            n = -1;
            break;
        }
        seed = pn_sample_next_token(tm_data_f32(logits), model->config.n_vocab, &sp, recent,
                                    n_recent);
        tm_free(logits);
        pn_mtp_recent_push(recent, &n_recent, STEP_RECENT, seed);
        out[n++] = seed;
    }
    pn_state_free(st);
    return n;
}

static uint64_t stream_hash(const uint32_t *toks, int n)
{
    uint64_t h = 1469598103934665603ull;
    for (int i = 0; i < n; i++)
        h = (h ^ toks[i]) * 1099511628211ull;
    return h;
}

/* Spec arm: draft(k) + one spec step per cycle, as the serve calls them. */
static int run_spec_arm(pn_model *model, const pn_token_seq *seq, float temp, int n_gen,
                        int k, uint32_t *out, step_counts *c)
{
    pn_sampling_params sp = step_sampler(temp);
    uint32_t recent[STEP_RECENT], seed, drafts[PN_CUDA_SPEC_MAX_DRAFTS];
    uint32_t emit[PN_CUDA_SPEC_MAX_DRAFTS + 1];
    int n_recent = 0, n = 0;
    pn_state *st = step_prefill(model, seq, 1, &sp, recent, &n_recent, &seed);

    if (!st)
        return -1;
    memset(c, 0, sizeof(*c));
    while (n < n_gen) {
        int nd = pn_qwen35moe_cuda_mtp_draft(model, st, seed, k, drafts);
        if (nd < 1) {
            fprintf(stderr, "draft failed at token %d\n", n);
            n = -1;
            break;
        }
        int ne = pn_qwen35moe_cuda_spec_step(model, st, seed, drafts, nd, emit, &sp, recent,
                                             &n_recent, STEP_RECENT);
        if (ne < 1) {
            fprintf(stderr, "spec step failed at token %d\n", n);
            n = -1;
            break;
        }
        c->cycles++;
        c->drafted += nd;
        c->accepted += ne - 1;
        for (int i = 0; i < ne && n < n_gen; i++)
            out[n++] = emit[i];
        seed = emit[ne - 1];
    }
    pn_state_free(st);
    return n;
}

/* One temperature: both arms, compared. 0 = exact, 1 = mismatch, -1 = run failure.
 * *plain_hash = the plain stream's hash. */
static int run_temperature(pn_model *model, const pn_token_seq *seq, float temp, int n_gen,
                           int k, uint64_t *plain_hash)
{
    static uint32_t ref[STEP_MAX_GEN], got[STEP_MAX_GEN], tok[STEP_MAX_GEN];
    step_counts c;
    int n_ref = run_plain_arm(model, seq, temp, n_gen, ref);
    int n_got = run_spec_arm(model, seq, temp, n_gen, k, got, &c);

    if (n_ref != n_gen || n_got != n_gen) {
        printf("TEMP %.1f run failed (plain %d, spec %d of %d)\n", temp, n_ref, n_got, n_gen);
        return -1;
    }
    *plain_hash = stream_hash(ref, n_gen);
    printf("TEMP %.1f cycles=%ld drafted=%ld accepted=%ld rejected=%ld plain_hash=%016llx\n",
           temp, c.cycles, c.drafted, c.accepted, c.drafted - c.accepted,
           (unsigned long long)*plain_hash);
    if (getenv("STEP_TOKEN_ARM") && atoi(getenv("STEP_TOKEN_ARM")) == 1) {
        int n_tok = run_token_arm(model, seq, temp, n_gen, tok), d = -1;
        for (int i = 0; n_tok == n_gen && i < n_gen && d < 0; i++)
            if (tok[i] != ref[i])
                d = i;
        printf("TEMP %.1f INFO single-token serve step vs decode-slots: %s", temp,
               n_tok != n_gen ? "run failed\n" : d < 0 ? "same stream\n" : "differs at ");
        if (n_tok == n_gen && d >= 0)
            printf("%d (%u vs %u)\n", d, tok[d], ref[d]);
    }
    for (int i = 0; i < n_gen; i++) {
        if (ref[i] != got[i]) {
            printf("TEMP %.1f DIFFERENT first_diff=%d plain=%u spec=%u\n", temp, i, ref[i],
                   got[i]);
            return 1;
        }
    }
    if (c.accepted == 0) {
        printf("TEMP %.1f IDENTICAL but no draft was accepted (nothing tested)\n", temp);
        return 1;
    }
    printf("TEMP %.1f IDENTICAL (%d tokens)\n", temp, n_gen);
    return 0;
}

int main(int argc, char **argv)
{
    const float temps[2] = { 0.0f, 0.7f };
    int bad = 0, failed = 0;

    if (argc < 2) {
        fprintf(stderr, "usage: %s <model.gguf> [n_gen=64] [k=2]\n", argv[0]);
        return 2;
    }
    int n_gen = argc > 2 ? atoi(argv[2]) : 64;
    int k = argc > 3 ? atoi(argv[3]) : 2;
    if (n_gen < 1 || n_gen > STEP_MAX_GEN || k < 1 || k > PN_CUDA_SPEC_MAX_DRAFTS) {
        fprintf(stderr, "bad n_gen/k\n");
        return 2;
    }
    pn_model *model = pn_load(argv[1]);
    if (!model)
        return 2;
    if (!pn_qwen35moe_mtp_ready(model)) {
        fprintf(stderr, "no MTP head (QWEN35_MTP=1?)\n");
        return 2;
    }
    printf("cuda spec step: n_gen=%d k=%d red=%s\n", n_gen, k,
           getenv("PN_MTP_RED") ? getenv("PN_MTP_RED") : "0");
    int sampled_left_greedy = 0;
    for (int p = 0; p < STEP_N_PROMPTS && !failed; p++) {
        pn_token_seq seq = pn_chat_format_first((pn_chat_style)model->chat_style,
                                                model->tokenizer, NULL, STEP_PROMPTS[p]);
        uint64_t hash[2] = { 0, 0 };
        if (!seq.tokens)
            return 2;
        printf("PROMPT %d (%d tokens)\n", p, seq.n_tokens);
        for (int t = 0; t < 2; t++) {
            int r = run_temperature(model, &seq, temps[t], n_gen, k, &hash[t]);
            failed |= r < 0;
            bad |= r != 0;
        }
        sampled_left_greedy |= hash[0] != hash[1];
        free(seq.tokens);
    }
    /* The sampled arm must actually sample somewhere, or it only re-tested greedy. */
    if (!failed && !sampled_left_greedy) {
        printf("TEMP 0.7 drew the greedy stream on every prompt (sampling untested)\n");
        bad = 1;
    }
    if (failed)
        printf("GATE: FAIL (run failure)\n");
    else
        printf(bad ? "GATE: FAIL (stream mismatch)\n" : "GATE: BIT-EXACT PASS\n");
    fflush(stdout);
    if (!getenv("QWEN35_EXIT_CLEAN"))
        _Exit(bad);
    pn_model_free(model);
    return bad;
}
#endif
