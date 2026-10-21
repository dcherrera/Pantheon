/* SPDX-License-Identifier: AGPL-3.0-or-later */
/* Copyright (C) 2026 David (Corey) Herrera */
/* Phase 9.2: how often does a prompt-lookup draft actually LAND?
 *
 * Phase 9.1 measured what a D-row verify COSTS and derived a break-even
 * acceptance from it.  It never measured acceptance on real traffic.  This
 * does, on the workload the reopen condition names: nex-n2-mini doing
 * code edits, where the answer re-emits the file that is sitting in the
 * prompt.  (Run it on saber too — the orchestrator's novel reasoning prose
 * is the unfavourable case and belongs in the record.)
 *
 * Per prompt it decodes the SAME continuation twice:
 *   1. plain greedy, single-token forwards  — the reference stream
 *   2. speculative greedy: pn_ngram_draft over (prompt + committed output),
 *      one batched pn_qwen35moe_spec_verify per round, pn_qwen35moe_spec_accept
 * The two streams must be IDENTICAL — speculation is exactness-preserving or
 * it is a bug, so that comparison is a gate, not a statistic.
 *
 * There is no throttle here on purpose: it drafts every round it can, so the
 * reported distribution is the raw behaviour of the traffic rather than the
 * behaviour of one throttle policy.  A round that drafts k tokens verifies
 * D = k+1 rows and delivers 1..D tokens (the pending token's own argmax is
 * always correct and always free).
 *
 * Reasoning vs answer are counted SEPARATELY.  A thinking model spends its
 * first few hundred tokens on novel prose and only then re-emits the file;
 * a single mean over both hides exactly the bimodality that decides this.
 *
 * Build (CUDA host): make USE_CUDA=1 tests/bench_spec_accept
 * Run: QWEN35_GPU_SPLIT_LAYER=20 QWEN35_GPU_DIE_A=6 QWEN35_GPU_DIE_B=7 \
 *        QWEN35_KV_Q4=1 QWEN35_MAX_SEQ=8192 QWEN35_FUSE=0x37 \
 *        ./tests/bench_spec_accept <model.gguf> <prompts.txt> [n_gen]
 *   prompts.txt: user messages separated by a line reading ===PROMPT===
 * Env: QWEN35_SPEC_K (max drafts per round, default 3 -> D=4 rows)
 *      QWEN35_SPEC_NG_MAX (3) / QWEN35_SPEC_NG_MIN (2)  n-gram match window
 *      QWEN35_ACC_SYSTEM  system prompt (default: a short coding-agent one)
 *      QWEN35_ACC_NOPLAIN=1  skip the reference stream (no exactness gate)
 * NOT part of the test suite (needs the model + a CUDA GPU).
 */
#include "../include/pantheon.h"
#include "../src/core/engine.h"
#include "../src/core/chat.h"
#include "../src/core/ngram_draft.h"
#include "../src/models/qwen35moe.h"
#include "tensor/tensor.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

#ifndef USE_CUDA
int main(void) { fprintf(stderr, "built without USE_CUDA\n"); return 1; }
#else

#define ACC_MAX_PROMPTS 32
#define ACC_MAX_GEN     4096
#define ACC_MAX_K       32
#define ACC_MAX_HIST    (65536)

/* Accepted-per-round histogram: index = tokens DELIVERED by the round
 * (1..D), so bucket 1 is "the draft was entirely wrong". */
typedef struct {
    long rounds;              /* rounds that drafted at least one token */
    long plain_rounds;        /* rounds with no draft -> plain forward */
    long drafted;             /* draft tokens proposed */
    long accepted;            /* draft tokens accepted */
    long delivered;           /* tokens emitted by drafting rounds */
    long forwards;            /* verify passes + plain forwards */
    long emitted;             /* tokens emitted overall */
    long hist[ACC_MAX_K + 2]; /* delivered-per-drafting-round */
} acc_stats;

static void acc_add(acc_stats *a, const acc_stats *b)
{
    a->rounds += b->rounds;
    a->plain_rounds += b->plain_rounds;
    a->drafted += b->drafted;
    a->accepted += b->accepted;
    a->delivered += b->delivered;
    a->forwards += b->forwards;
    a->emitted += b->emitted;
    for (int i = 0; i <= ACC_MAX_K + 1; i++)
        a->hist[i] += b->hist[i];
}

static double now_s(void)
{
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (double)ts.tv_sec + (double)ts.tv_nsec * 1e-9;
}

static int argmax(const float *x, int n)
{
    int best = 0;
    for (int i = 1; i < n; i++)
        if (x[i] > x[best]) best = i;
    return best;
}

static int env_int(const char *name, int def)
{
    const char *v = getenv(name);
    return v ? atoi(v) : def;
}

/* Slurp `path`, split on lines reading exactly "===PROMPT===".  Returns the
 * prompt count; out[] entries are malloc'd (leaked deliberately — one-shot
 * bench). */
static int load_prompts(const char *path, char **out, int max_out)
{
    FILE *f = fopen(path, "rb");
    if (!f) return 0;
    fseek(f, 0, SEEK_END);
    long len = ftell(f);
    fseek(f, 0, SEEK_SET);
    char *buf = malloc((size_t)len + 1);
    if (!buf || fread(buf, 1, (size_t)len, f) != (size_t)len) {
        fclose(f);
        return 0;
    }
    buf[len] = '\0';
    fclose(f);

    int n = 0;
    char *cur = buf;
    while (cur && *cur && n < max_out) {
        char *sep = strstr(cur, "\n===PROMPT===\n");
        if (sep) *sep = '\0';
        /* Skip a leading separator line so the file may start with one. */
        if (strncmp(cur, "===PROMPT===\n", 13) == 0) cur += 13;
        if (*cur) out[n++] = cur;
        cur = sep ? sep + 14 : NULL;
    }
    return n;
}

/* Plain greedy decode of up to n_gen tokens from a fresh state.  Writes the
 * stream to gen[] and returns its length, or -1. */
static int decode_plain(pn_model *model, const uint32_t *prefix, int n_prefix,
                        int n_gen, int eos, uint32_t *gen, double *secs)
{
    pn_state *st = pn_state_new(model);
    if (!st) return -1;
    tm_tensor *out = NULL;
    for (int i = 0; i < n_prefix; i++) {
        if (out) tm_free(out);
        out = model->forward(model, prefix[i], st);
        if (!out) { pn_state_free(st); return -1; }
    }
    double t0 = now_s();
    int n = 0;
    uint32_t tok = (uint32_t)argmax(tm_data_f32(out), model->config.n_vocab);
    tm_free(out);
    gen[n++] = tok;
    while (n < n_gen && (int)tok != eos) {
        out = model->forward(model, tok, st);
        if (!out) { pn_state_free(st); return -1; }
        tok = (uint32_t)argmax(tm_data_f32(out), model->config.n_vocab);
        tm_free(out);
        gen[n++] = tok;
    }
    *secs = now_s() - t0;
    pn_state_free(st);
    return n;
}

/* Speculative greedy decode of the same continuation, accumulating stats into
 * pre[] (before the reasoning marker) and post[] (after it).  Returns the
 * stream length, or -1. */
static int decode_spec(pn_model *model, const uint32_t *prefix, int n_prefix,
                       int n_gen, int eos, int think_end, int max_k,
                       int ng_max, int ng_min, uint32_t *gen,
                       acc_stats *pre, acc_stats *post, double *secs)
{
    static uint32_t hist[ACC_MAX_HIST];
    pn_state *st = pn_state_new(model);
    if (!st) return -1;
    int n_vocab = model->config.n_vocab;

    tm_tensor *out = NULL;
    for (int i = 0; i < n_prefix; i++) {
        if (out) tm_free(out);
        out = model->forward(model, prefix[i], st);
        if (!out) { pn_state_free(st); return -1; }
    }
    double t0 = now_s();
    uint32_t pending = (uint32_t)argmax(tm_data_f32(out), n_vocab);
    tm_free(out);

    if (n_prefix > ACC_MAX_HIST - ACC_MAX_GEN) { pn_state_free(st); return -1; }
    memcpy(hist, prefix, (size_t)n_prefix * sizeof(uint32_t));
    int nh = n_prefix;
    int n = 0;
    int reasoning = (think_end >= 0);   /* no marker -> everything is "answer" */

    gen[n++] = pending;
    hist[nh++] = pending;
    if (reasoning && (int)pending == think_end) reasoning = 0;

    while (n < n_gen && (int)gen[n - 1] != eos) {
        acc_stats *bucket = reasoning ? pre : post;
        uint32_t drafts[ACC_MAX_K], block[ACC_MAX_K + 1];
        int32_t am[ACC_MAX_K + 1];
        int k = pn_ngram_draft(hist, nh, ng_max, ng_min, max_k, drafts);
        int stepped = 0;

        if (k > 0) {
            block[0] = pending;
            memcpy(block + 1, drafts, (size_t)k * sizeof(uint32_t));
            if (pn_qwen35moe_spec_verify(model, block, k + 1, st, am) == 0) {
                int acc = 1;                       /* am[0] is always correct */
                while (acc <= k && drafts[acc - 1] == (uint32_t)am[acc - 1])
                    acc++;
                if (pn_qwen35moe_spec_accept(model, st, acc) != 0) {
                    pn_state_free(st);
                    return -1;
                }
                bucket->rounds++;
                bucket->drafted += k;
                bucket->accepted += acc - 1;
                bucket->delivered += acc;
                bucket->forwards++;
                bucket->hist[acc]++;
                for (int i = 0; i < acc && n < n_gen; i++) {
                    gen[n++] = (uint32_t)am[i];
                    hist[nh++] = (uint32_t)am[i];
                    bucket->emitted++;
                    if (reasoning && am[i] == think_end) reasoning = 0;
                }
                pending = (uint32_t)am[acc - 1];
                stepped = 1;
            }
        }
        if (!stepped) {
            out = model->forward(model, pending, st);
            if (!out) { pn_state_free(st); return -1; }
            pending = (uint32_t)argmax(tm_data_f32(out), n_vocab);
            tm_free(out);
            bucket->plain_rounds++;
            bucket->forwards++;
            bucket->emitted++;
            gen[n++] = pending;
            hist[nh++] = pending;
            if (reasoning && (int)pending == think_end) reasoning = 0;
        }
    }
    *secs = now_s() - t0;
    pn_state_free(st);
    return n;
}

static void report(const char *label, const acc_stats *a, int max_k)
{
    long rounds = a->rounds + a->plain_rounds;
    if (!rounds) { printf("  %-9s (no rounds)\n", label); return; }
    printf("  %-9s emitted=%-5ld forwards=%-5ld tok/fwd=%.2f | "
           "draft_rounds=%ld/%ld (%.0f%%) drafted=%ld accepted=%ld "
           "accept=%.1f%% | delivered/round=%.2f\n",
           label, a->emitted, a->forwards,
           a->forwards ? (double)a->emitted / (double)a->forwards : 0.0,
           a->rounds, rounds, 100.0 * (double)a->rounds / (double)rounds,
           a->drafted, a->accepted,
           a->drafted ? 100.0 * (double)a->accepted / (double)a->drafted : 0.0,
           a->rounds ? (double)a->delivered / (double)a->rounds : 0.0);
    printf("  %-9s delivered-per-drafting-round:", "");
    for (int i = 1; i <= max_k + 1; i++)
        printf(" %d:%ld", i, a->hist[i]);
    printf("\n");
}

int main(int argc, char **argv)
{
    if (argc < 3) {
        fprintf(stderr, "usage: %s <model.gguf> <prompts.txt> [n_gen]\n",
                argv[0]);
        return 1;
    }
    int n_gen = argc > 3 ? atoi(argv[3]) : 512;
    if (n_gen < 8) n_gen = 8;
    if (n_gen > ACC_MAX_GEN) n_gen = ACC_MAX_GEN;
    int max_k = env_int("QWEN35_SPEC_K", 3);
    if (max_k < 1) max_k = 1;
    if (max_k > ACC_MAX_K) max_k = ACC_MAX_K;
    int ng_max = env_int("QWEN35_SPEC_NG_MAX", 3);
    int ng_min = env_int("QWEN35_SPEC_NG_MIN", 2);
    int no_plain = env_int("QWEN35_ACC_NOPLAIN", 0);

    char *prompts[ACC_MAX_PROMPTS];
    int n_prompts = load_prompts(argv[2], prompts, ACC_MAX_PROMPTS);
    if (!n_prompts) { fprintf(stderr, "no prompts in %s\n", argv[2]); return 1; }

    pn_model *model = pn_load(argv[1]);
    if (!model) { fprintf(stderr, "load failed\n"); return 1; }
    int eos = ft_eos_id(model->tokenizer);
    int think_end = ft_token_to_id(model->tokenizer, "</think>");
    const char *system = getenv("QWEN35_ACC_SYSTEM");
    if (!system)
        system = "You are a coding agent. Apply the requested edit and "
                 "re-emit the complete file in a single fenced code block. "
                 "Do not explain.";

    printf("== spec acceptance: %s\n", argv[1]);
    printf("   prompts=%d n_gen=%d D=%d rows (max_k=%d) ngram=%d..%d "
           "eos=%d think_end=%d chat_style=%d\n",
           n_prompts, n_gen, max_k + 1, max_k, ng_min, ng_max, eos, think_end,
           model->chat_style);

    acc_stats tot_pre = { 0 }, tot_post = { 0 };
    int gates_run = 0, gates_pass = 0;

    for (int p = 0; p < n_prompts; p++) {
        pn_token_seq seq = pn_chat_format_first((pn_chat_style)model->chat_style,
                                                model->tokenizer, system,
                                                prompts[p]);
        if (!seq.tokens || seq.n_tokens <= 0) {
            fprintf(stderr, "prompt %d: format failed\n", p);
            continue;
        }
        static uint32_t gen_plain[ACC_MAX_GEN], gen_spec[ACC_MAX_GEN];
        double t_plain = 0.0, t_spec = 0.0;
        int n_plain = 0;

        if (!no_plain) {
            n_plain = decode_plain(model, seq.tokens, seq.n_tokens, n_gen, eos,
                                   gen_plain, &t_plain);
            if (n_plain < 0) { fprintf(stderr, "prompt %d: plain failed\n", p);
                               continue; }
        }

        acc_stats pre = { 0 }, post = { 0 };
        int n_spec = decode_spec(model, seq.tokens, seq.n_tokens, n_gen, eos,
                                 think_end, max_k, ng_max, ng_min, gen_spec,
                                 &pre, &post, &t_spec);
        if (n_spec < 0) { fprintf(stderr, "prompt %d: spec failed\n", p);
                          continue; }

        int diff = -1;
        if (!no_plain) {
            gates_run++;
            if (n_plain != n_spec) {
                diff = n_plain < n_spec ? n_plain : n_spec;
            } else {
                for (int i = 0; i < n_spec; i++)
                    if (gen_plain[i] != gen_spec[i]) { diff = i; break; }
            }
            if (diff < 0) gates_pass++;
        }

        printf("\n-- prompt %d: %d prompt tokens, %d generated%s\n", p,
               seq.n_tokens, n_spec,
               no_plain ? "" : (diff < 0 ? ", EXACT vs plain"
                                         : ", ***STREAM DIVERGED***"));
        if (diff >= 0)
            printf("   first_diff=%d plain=%u spec=%u (n_plain=%d n_spec=%d)\n",
                   diff, gen_plain[diff], gen_spec[diff], n_plain, n_spec);
        report("reasoning", &pre, max_k);
        report("answer", &post, max_k);
        if (!no_plain && t_spec > 0.0)
            printf("   wall: plain %.2fs (%.2f tok/s) | spec %.2fs "
                   "(%.2f tok/s) | %.2fx\n",
                   t_plain, n_plain / t_plain, t_spec, n_spec / t_spec,
                   t_plain / t_spec);
        acc_add(&tot_pre, &pre);
        acc_add(&tot_post, &post);
        free(seq.tokens);
        fflush(stdout);
    }

    acc_stats all = { 0 };
    acc_add(&all, &tot_pre);
    acc_add(&all, &tot_post);
    printf("\n== TOTAL (D=%d rows)\n", max_k + 1);
    report("reasoning", &tot_pre, max_k);
    report("answer", &tot_post, max_k);
    report("combined", &all, max_k);
    if (gates_run)
        printf("EXACTNESS GATE: %d/%d prompts bit-identical to plain greedy\n",
               gates_pass, gates_run);

    fflush(stdout);
    if (!getenv("QWEN35_EXIT_CLEAN"))
        _Exit(gates_run && gates_pass != gates_run ? 1 : 0);
    return 0;
}
#endif
