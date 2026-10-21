/* SPDX-License-Identifier: AGPL-3.0-or-later */
/* Copyright (C) 2026 David (Corey) Herrera */
/**
 * @file eval_ppl.c
 * @brief Quality-eval driver for the qwen35moe engine: perplexity over a fixed
 *   corpus (`ppl` mode) and greedy generation over a prompt suite (`gen` mode).
 *   Replaces the bit-exact gate when comparing two engine builds whose logits
 *   are not bit-identical (relaxed reduction order, requantized hot tensors...):
 *   run both modes through build A and build B and diff with evals/eval_quality.sh.
 *
 *   ppl mode: teacher-forced NLL of each actual next token, from the engine's
 *   own forward logits (numerically stable log-softmax in double). Emits
 *   `PPL tokens=N nll=X ppl=Y` on stdout.
 *
 *   gen mode: for each %%-separated prompt, format with the model's chat
 *   template, prefill, then decode greedily (pure argmax, temp-0 equivalent)
 *   up to max_new tokens. Emits machine-parsable `GEN <i> ...: id id ...` plus
 *   a human-readable `TXT <i>: ...` line per prompt. One process, one model
 *   load, all prompts (state reset between prompts).
 *
 *   CPU build:  make USE_BLAS=1 tests/eval_ppl
 *   GPU build (CUDA host): make USE_CUDA=1 tests/eval_ppl, then the usual
 *   QWEN35_GPU_SPLIT_LAYER / QWEN35_GPU_DIE_A / QWEN35_GPU_DIE_B env selects
 *   the dies. An optional leading numeric arg acquires a barswap die first
 *   (legacy rigs only; small-BAR-quirk rigs skip it). NOT part of the test suite.
 *
 * @status Working; validated golden-vs-golden (0 divergence, 0 ppl delta) on M4 CPU
 * @issues None
 * @todo None
 */
#include "../include/pantheon.h"
#include "../src/core/bpe_pretokenize.h"
#include "../src/core/chat.h"
#include "../src/core/engine.h"
#include "../src/core/mem/pn_memwatch.h"
#include "../src/core/ngram_draft.h"
#include "tensor/tensor.h"
#include "tokenizer/tokenizer.h"

#include <ctype.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

#ifdef USE_CUDA
#include <unistd.h>
#include "barswap/barswap.h"
#include "../src/models/qwen35moe.h"
#endif

/* Hard caps so a malformed input can't run away on the 2-core K80 host. */
/* Caps raised for long-context (needle-in-haystack) prompts: a 32k-token
 * needle prompt is ~180 KB of text, and a multi-cell suite of them needs a
 * larger corpus buffer + prompt-token ceiling than the default 4k. */
#define EVAL_MAX_CORPUS_BYTES  (64 * 1024 * 1024)
#define EVAL_MAX_PROMPTS       256
#define EVAL_MAX_GEN_TOKENS    512
#define EVAL_MAX_PROMPT_TOKENS 40000
#define EVAL_SPEC_MAX_K        32

/* Speculative greedy decode (gen mode, QWEN35_SPEC=1, CUDA builds): n-gram
 * drafts verified in one batched pass; emitted tokens are the model's exact
 * greedy argmaxes, so GEN output is unchanged. Stats accumulate across the
 * prompt suite and print once at the end. */
static long g_spec_verifies, g_spec_plain, g_spec_drafted, g_spec_accepted;

static int spec_enabled(void)
{
    const char *v = getenv("QWEN35_SPEC");
    return v && v[0] && v[0] != '0';
}

static int spec_env_int(const char *name, int def)
{
    const char *v = getenv(name);
    return v ? atoi(v) : def;
}

static double now_s(void)
{
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (double)ts.tv_sec + (double)ts.tv_nsec * 1e-9;
}

#ifdef USE_CUDA
static int load_nvidia(void)
{
    if (system("modprobe --ignore-install nvidia 2>/dev/null") != 0)
        return -1;
    sleep(1);
    if (system("modprobe --ignore-install nvidia_uvm 2>/dev/null") != 0)
        return -1;
    sleep(1);
    return 0;
}
#endif

/* Read a whole file into a NUL-terminated heap buffer (caller frees). */
static char *read_text_file(const char *path, size_t *out_len)
{
    char *buf = NULL;
    FILE *f = fopen(path, "rb");
    if (!f) { fprintf(stderr, "eval_ppl: cannot open %s\n", path); return NULL; }

    fseek(f, 0, SEEK_END);
    long len = ftell(f);
    fseek(f, 0, SEEK_SET);
    if (len <= 0 || len > EVAL_MAX_CORPUS_BYTES) {
        fprintf(stderr, "eval_ppl: %s size %ld out of range\n", path, len);
        fclose(f);
        return NULL;
    }

    buf = malloc((size_t)len + 1);
    if (buf && fread(buf, 1, (size_t)len, f) == (size_t)len) {
        buf[len] = '\0';
        if (out_len) *out_len = (size_t)len;
    } else {
        free(buf);
        buf = NULL;
    }
    fclose(f);
    return buf;
}

/* Stable log-softmax probability of `target` under `logits[n]`, in double. */
static double token_log_prob(const float *logits, int n, int target)
{
    double max = logits[0];
    for (int i = 1; i < n; i++)
        if (logits[i] > max) max = logits[i];
    double sum = 0.0;
    for (int i = 0; i < n; i++)
        sum += exp((double)logits[i] - max);
    return (double)logits[target] - max - log(sum);
}

/* Greedy pick: first index with the strictly greatest logit (same tie-break as
 * the CPU oracle, so `gen` mode is comparable with the bit-exact gate's argmax). */
static int argmax_token(const float *logits, int n)
{
    int best = 0;
    for (int i = 1; i < n; i++)
        if (logits[i] > logits[best]) best = i;
    return best;
}

/* One forward step; returns borrowed logits pointer (owned by *out). */
static float *forward_logits(pn_model *model, pn_state *state, uint32_t token,
                             tm_tensor **out, int *n_vocab)
{
    tm_tensor *t = model->forward(model, token, state);
    if (!t) return NULL;
    *out = t;
    *n_vocab = (int)t->shape[0];
    return tm_data_f32(t);
}

/* Teacher-forced perplexity over the tokenized corpus.
 * NLL is accumulated for tokens[1..n-1], each predicted from its true prefix. */
static int run_perplexity(pn_model *model, pn_state *state,
                          const char *corpus_path, int max_tokens)
{
    char *text = read_text_file(corpus_path, NULL);
    if (!text) return 1;

    /* Byte-level BPE models need the qwen35 pre-tokenizer + GPT-2 byte encode
     * (raw ft_encode returns -1 ids for plain UTF-8 text). Matches chat.c. */
    int n_tok = 0;
    int *toks = ft_get_model_type(model->tokenizer) == 0
                    ? pn_bpe_encode(model->tokenizer, text, &n_tok)
                    : ft_encode(model->tokenizer, text, &n_tok);
    free(text);
    if (!toks || n_tok < 2) {
        fprintf(stderr, "eval_ppl: corpus tokenized to %d tokens (<2)\n", n_tok);
        free(toks);
        return 1;
    }
    int n_full = n_tok;
    if (max_tokens > 1 && n_tok > max_tokens) n_tok = max_tokens;
    if (n_tok > state->config->n_ctx) n_tok = state->config->n_ctx;
    for (int i = 0; i < n_tok; i++) {
        if (toks[i] < 0 || toks[i] >= state->config->n_vocab) {
            fprintf(stderr, "eval_ppl: bad token id %d at %d\n", toks[i], i);
            free(toks);
            return 1;
        }
    }
    fprintf(stderr, "eval_ppl: corpus %s -> %d tokens (scoring %d)\n",
            corpus_path, n_full, n_tok);

    /* Pre-flight admission (MemoryAware §6.2): this teacher-forced pass is one
     * long prefill — gate it exactly like serve (engine-level pn_admit, §11 Q7:
     * this harness is the path that crash-looped the K80 test machine). CAP scores a
     * shorter corpus prefix; DENIED refuses the run. */
    int fit_tok = n_tok;
    pn_admit_status admit = pn_admit(model, n_tok, &fit_tok);
    if (admit == PN_ADMIT_DENIED) {
        fprintf(stderr, "eval_ppl: admission refused a %d-token ppl pass "
                "(host/die memory would be exhausted)\n", n_tok);
        free(toks);
        return 1;
    }
    if (admit == PN_ADMIT_CAPPED && fit_tok > 1 && fit_tok < n_tok) {
        fprintf(stderr, "eval_ppl: admission capped the ppl pass %d -> %d "
                "tokens\n", n_tok, fit_tok);
        n_tok = fit_tok;
    }

    double nll_sum = 0.0;
    int scored = 0;
    for (int i = 0; i + 1 < n_tok; i++) {
        tm_tensor *out = NULL;
        int n_vocab = 0;
        float *logits = forward_logits(model, state, (uint32_t)toks[i], &out, &n_vocab);
        if (!logits) { fprintf(stderr, "eval_ppl: forward failed at %d\n", i); free(toks); return 1; }
        nll_sum += -token_log_prob(logits, n_vocab, toks[i + 1]);
        scored++;
        tm_free(out);
        if (scored % 64 == 0)
            fprintf(stderr, "  [%d/%d] running ppl=%.4f\n",
                    scored, n_tok - 1, exp(nll_sum / scored));
    }

    /* The single stdout line eval_quality.sh parses. */
    printf("PPL tokens=%d nll=%.6f ppl=%.6f\n",
           scored, nll_sum / scored, exp(nll_sum / scored));
    free(toks);
    return 0;
}

#ifdef USE_CUDA
/* Teacher-forced perplexity via CHUNKED BATCHED PREFILL: the corpus runs
 * through pn_qwen35moe_forward_batch_cuda with per-position logits, so the
 * score exercises the batch attention path (flash or host softmax —
 * A/B with QWEN35_FLASH=0 vs 1). NLL semantics identical to run_perplexity:
 * token i+1 predicted from its true prefix (chunk boundaries carry the KV
 * cache and SSM state, exactly like one long prefill). Chunk size via
 * EVAL_PPLB_CHUNK (default 256; per-chunk logits buffer is
 * chunk * n_vocab floats — mind the 8 GB K80 host). */
static int run_perplexity_batch(pn_model *model, pn_state *state,
                                const char *corpus_path, int max_tokens)
{
    char *text = read_text_file(corpus_path, NULL);
    if (!text) return 1;
    int n_tok = 0;
    int *toks = ft_get_model_type(model->tokenizer) == 0
                    ? pn_bpe_encode(model->tokenizer, text, &n_tok)
                    : ft_encode(model->tokenizer, text, &n_tok);
    free(text);
    if (!toks || n_tok < 2) {
        fprintf(stderr, "eval_ppl: corpus tokenized to %d tokens (<2)\n", n_tok);
        free(toks);
        return 1;
    }
    int n_full = n_tok;
    if (max_tokens > 1 && n_tok > max_tokens) n_tok = max_tokens;
    if (n_tok > state->config->n_ctx) n_tok = state->config->n_ctx;
    for (int i = 0; i < n_tok; i++) {
        if (toks[i] < 0 || toks[i] >= state->config->n_vocab) {
            fprintf(stderr, "eval_ppl: bad token id %d at %d\n", toks[i], i);
            free(toks);
            return 1;
        }
    }
    fprintf(stderr, "eval_ppl: corpus %s -> %d tokens (scoring %d, batched)\n",
            corpus_path, n_full, n_tok);

    /* Pre-flight admission (MemoryAware §6.2) — same gate as the sequential
     * ppl pass: the batched prefill grows the same KV and is just as capable
     * of exhausting the host/die. CAP scores a prefix; DENIED refuses. */
    int fit_tok = n_tok;
    pn_admit_status admit = pn_admit(model, n_tok, &fit_tok);
    if (admit == PN_ADMIT_DENIED) {
        fprintf(stderr, "eval_ppl: admission refused a %d-token pplb pass "
                "(host/die memory would be exhausted)\n", n_tok);
        free(toks);
        return 1;
    }
    if (admit == PN_ADMIT_CAPPED && fit_tok > 1 && fit_tok < n_tok) {
        fprintf(stderr, "eval_ppl: admission capped the pplb pass %d -> %d "
                "tokens\n", n_tok, fit_tok);
        n_tok = fit_tok;
    }

    int chunk = spec_env_int("EVAL_PPLB_CHUNK", 256);
    if (chunk < 2) chunk = 2;
    int n_vocab = state->config->n_vocab;
    float *pos_logits = malloc((size_t)chunk * n_vocab * sizeof(float));
    uint32_t *chunk_toks = malloc((size_t)chunk * sizeof(uint32_t));
    if (!pos_logits || !chunk_toks) {
        fprintf(stderr, "eval_ppl: pplb scratch alloc failed\n");
        free(pos_logits); free(chunk_toks); free(toks);
        return 1;
    }

    double nll_sum = 0.0;
    int scored = 0;
    for (int off = 0; off < n_tok; off += chunk) {
        int nb = n_tok - off < chunk ? n_tok - off : chunk;
        for (int i = 0; i < nb; i++)
            chunk_toks[i] = (uint32_t)toks[off + i];
        tm_tensor *out = pn_qwen35moe_forward_batch_cuda(model, chunk_toks,
                                                         nb, state,
                                                         pos_logits);
        if (!out) {
            fprintf(stderr, "eval_ppl: batch prefill failed at %d\n", off);
            free(pos_logits); free(chunk_toks); free(toks);
            return 1;
        }
        tm_free(out);
        /* Row i predicts corpus token off+i+1 (the last row of a non-final
         * chunk scores the next chunk's first token). */
        for (int i = 0; i < nb && off + i + 1 < n_tok; i++) {
            nll_sum += -token_log_prob(pos_logits + (size_t)i * n_vocab,
                                       n_vocab, toks[off + i + 1]);
            scored++;
        }
        fprintf(stderr, "  [%d/%d] running ppl=%.4f\n",
                scored, n_tok - 1, exp(nll_sum / (scored > 0 ? scored : 1)));
    }

    printf("PPL tokens=%d nll=%.6f ppl=%.6f\n",
           scored, nll_sum / scored, exp(nll_sum / scored));
    free(pos_logits);
    free(chunk_toks);
    free(toks);
    return 0;
}
#endif /* USE_CUDA */

/* Escape newlines/tabs so each TXT line stays a single greppable line. */
static void print_escaped_line(const char *tag, int idx, const char *text)
{
    printf("%s %d: ", tag, idx);
    for (const char *p = text; *p; p++) {
        if (*p == '\n')      fputs("\\n", stdout);
        else if (*p == '\t') fputs("\\t", stdout);
        else                 fputc(*p, stdout);
    }
    fputc('\n', stdout);
}

/* Greedy-decode one prompt; prints its GEN + TXT lines. Returns 0 on success. */
static int run_one_prompt(pn_model *model, pn_state *state, int idx,
                          const char *prompt, int max_new, int eos_id, int im_end_id)
{
    pn_state_reset(state);
    pn_token_seq seq = pn_chat_format_first((pn_chat_style)model->chat_style,
                                            model->tokenizer, NULL, prompt);
    if (!seq.tokens || seq.n_tokens <= 0) {
        fprintf(stderr, "eval_ppl: prompt %d failed to format\n", idx);
        return 1;
    }

    /* Pre-flight admission (MemoryAware §6.2), the same engine-level gate as
     * serve: DENIED refuses the prompt; CAP keeps the most recent fit tokens
     * (sliding-window trim, mirroring serve_generation). */
    int fit_tok = seq.n_tokens;
    pn_admit_status admit = pn_admit(model, seq.n_tokens, &fit_tok);
    if (admit == PN_ADMIT_DENIED) {
        fprintf(stderr, "eval_ppl: admission refused prompt %d (%d tokens — "
                "host/die memory would be exhausted)\n", idx, seq.n_tokens);
        free(seq.tokens);
        return 1;
    }
    if (admit == PN_ADMIT_CAPPED && fit_tok > 0 && fit_tok < seq.n_tokens) {
        int drop = seq.n_tokens - fit_tok;
        memmove(seq.tokens, seq.tokens + drop,
                (size_t)fit_tok * sizeof(uint32_t));
        seq.n_tokens = fit_tok;
        fprintf(stderr, "eval_ppl: admission capped prompt %d: dropped %d "
                "oldest tokens, prefilling %d\n", idx, drop, seq.n_tokens);
    }

    tm_tensor *out = NULL;
    int n_vocab = 0;
    float *logits = NULL;
#ifdef USE_CUDA
    /* Long-context prompts (needle-in-haystack): prefill through the BATCHED
     * CUDA path (exercises the prefill flash + sliding-window kernels, O(N*W)
     * instead of N sequential decode forwards) then decode from the last row's
     * logits. EVAL_BATCH_PREFILL=0 forces the legacy token-by-token path. */
    const char *bpf = getenv("EVAL_BATCH_PREFILL");
    if (!(bpf && bpf[0] == '0') && seq.n_tokens > 1) {
        int nv = state->config->n_vocab;
        int chunk = spec_env_int("EVAL_PPLB_CHUNK", 256);
        if (chunk < 2) chunk = 2;
        float *pos_logits = malloc((size_t)chunk * nv * sizeof(float));
        uint32_t *ctoks = malloc((size_t)chunk * sizeof(uint32_t));
        if (!pos_logits || !ctoks) {
            fprintf(stderr, "eval_ppl: batch-prefill scratch alloc failed\n");
            free(pos_logits); free(ctoks); free(seq.tokens); return 1;
        }
        int last_row = 0;
        for (int off = 0; off < seq.n_tokens; off += chunk) {
            int nb = seq.n_tokens - off < chunk ? seq.n_tokens - off : chunk;
            for (int i = 0; i < nb; i++) ctoks[i] = seq.tokens[off + i];
            tm_tensor *bo = pn_qwen35moe_forward_batch_cuda(model, ctoks, nb,
                                                            state, pos_logits);
            if (!bo) {
                fprintf(stderr, "eval_ppl: batch prefill failed (prompt %d @ %d)\n",
                        idx, off);
                free(pos_logits); free(ctoks); free(seq.tokens); return 1;
            }
            tm_free(bo);
            last_row = nb - 1;
        }
        n_vocab = nv;
        static float last_logits[1 << 20]; /* >= n_vocab; holds the final row */
        memcpy(last_logits, pos_logits + (size_t)last_row * nv,
               (size_t)nv * sizeof(float));
        logits = last_logits;
        out = NULL;
        free(pos_logits); free(ctoks);
    } else
#endif
    for (int i = 0; i < seq.n_tokens; i++) {
        if (out) tm_free(out);
        logits = forward_logits(model, state, seq.tokens[i], &out, &n_vocab);
        if (!logits) { fprintf(stderr, "eval_ppl: prefill failed (prompt %d)\n", idx); free(seq.tokens); return 1; }
    }

    printf("GEN %d prompt_tokens=%d:", idx, seq.n_tokens);
    char raw[EVAL_MAX_GEN_TOKENS * 16];
    size_t raw_len = 0;
    raw[0] = '\0';
    int emitted = 0;

    /* Decode loop. `cur` is the next greedy token, emitted but NOT yet fed;
     * its successors come from the last verify's queue (spec mode), one
     * batched verify of [cur, n-gram drafts...] (spec mode), or one plain
     * forward of cur. All three produce the model's exact greedy argmaxes,
     * so GEN output is identical with QWEN35_SPEC on or off. */
    int spec = spec_enabled();
    int max_k = spec_env_int("QWEN35_SPEC_K", 6);
    int ng_max = spec_env_int("QWEN35_SPEC_NG_MAX", 3);
    int ng_min = spec_env_int("QWEN35_SPEC_NG_MIN", 3);
    if (max_k < 1) max_k = 1;
    if (max_k > EVAL_SPEC_MAX_K) max_k = EVAL_SPEC_MAX_K;
    /* The economics guard (see pn_spec_throttle): a K80 verify costs ~4
     * plain forwards, so probe-and-cool through phases that draft poorly
     * and skip verifies whose draft count cannot reach break-even. */
    pn_spec_throttle thr;
    pn_spec_throttle_init(&thr, spec_env_int("QWEN35_SPEC_MIN_ACC", 4),
                          spec_env_int("QWEN35_SPEC_MISS", 2),
                          spec_env_int("QWEN35_SPEC_COOL", 48));
#ifndef USE_CUDA
    (void)max_k; (void)ng_max; (void)ng_min; (void)thr; /* CUDA-only path */
#endif
    static uint32_t hist[EVAL_MAX_PROMPT_TOKENS + EVAL_MAX_GEN_TOKENS + 1];
    int nh = seq.n_tokens < EVAL_MAX_PROMPT_TOKENS ? seq.n_tokens
                                                   : EVAL_MAX_PROMPT_TOKENS;
    memcpy(hist, seq.tokens, (size_t)nh * sizeof(uint32_t));
    int32_t queue[EVAL_SPEC_MAX_K + 1];
    int qn = 0, qi = 0;

    double t_dec = now_s();
    int cur = argmax_token(logits, n_vocab);
    while (emitted < max_new) {
        if (cur == eos_id || cur == im_end_id) break;
        printf(" %d", cur);
        const char *piece = ft_id_to_token(model->tokenizer, cur);
        if (piece) {
            size_t plen = strlen(piece);
            if (raw_len + plen < sizeof(raw) - 1) {
                memcpy(raw + raw_len, piece, plen + 1);
                raw_len += plen;
            }
        }
        hist[nh++] = (uint32_t)cur;
        emitted++;
        if (emitted >= max_new) break;

        if (qi < qn) { cur = queue[qi++]; continue; }

        int stepped = 0;
#ifdef USE_CUDA
        if (spec && pn_spec_throttle_gate(&thr)) {
            uint32_t drafts[EVAL_SPEC_MAX_K], block[EVAL_SPEC_MAX_K + 1];
            int32_t am[EVAL_SPEC_MAX_K + 1];
            int k = pn_ngram_draft(hist, nh, ng_max, ng_min, max_k, drafts);
            if (k >= thr.min_accept) {
                block[0] = (uint32_t)cur;
                memcpy(block + 1, drafts, (size_t)k * sizeof(uint32_t));
                if (pn_qwen35moe_spec_verify(model, block, k + 1, state,
                                             am) == 0) {
                    int acc = 1;
                    while (acc <= k && drafts[acc - 1] == (uint32_t)am[acc - 1])
                        acc++;
                    if (pn_qwen35moe_spec_accept(model, state, acc) != 0) {
                        fprintf(stderr, "\neval_ppl: spec accept failed\n");
                        free(seq.tokens);
                        return 1;
                    }
                    g_spec_verifies++;
                    g_spec_drafted += k;
                    g_spec_accepted += acc - 1;
                    pn_spec_throttle_report(&thr, acc - 1);
                    qn = qi = 0;
                    for (int i = 0; i < acc; i++)
                        queue[qn++] = am[i];
                    cur = queue[qi++];
                    stepped = 1;
                }
            }
        }
#endif
        if (!stepped) {
            tm_free(out);
            out = NULL;
            logits = forward_logits(model, state, (uint32_t)cur, &out, &n_vocab);
            if (!logits) { fprintf(stderr, "\neval_ppl: decode failed (prompt %d)\n", idx); free(seq.tokens); return 1; }
            cur = argmax_token(logits, n_vocab);
            g_spec_plain++;
        }
    }
    double dt_dec = now_s() - t_dec;
    printf("\n");
    if (out) tm_free(out);

    char *decoded = gpt2_bytes_decode(raw); /* qwen byte-level BPE -> UTF-8 */
    print_escaped_line("TXT", idx, decoded ? decoded : raw);
    free(decoded);
    fflush(stdout);
    fprintf(stderr, "eval_ppl: prompt %d done (%d tokens, %.2f tok/s%s)\n",
            idx, emitted, dt_dec > 0.0 ? emitted / dt_dec : 0.0,
            spec ? ", spec" : "");
    free(seq.tokens);
    return 0;
}

/* Greedy generation over every %%-separated prompt in the suite file. */
static int run_generation(pn_model *model, pn_state *state,
                          const char *prompts_path, int max_new)
{
    char *text = read_text_file(prompts_path, NULL);
    if (!text) return 1;

    /* Split on lines that contain exactly "%%". */
    char *prompts[EVAL_MAX_PROMPTS];
    int n_prompts = 0;
    char *cursor = text;
    while (cursor && n_prompts < EVAL_MAX_PROMPTS) {
        char *sep = strstr(cursor, "\n%%\n");
        if (sep) *sep = '\0';
        while (*cursor == '\n' || *cursor == ' ') cursor++;
        if (*cursor) prompts[n_prompts++] = cursor;
        cursor = sep ? sep + 4 : NULL;
    }
    fprintf(stderr, "eval_ppl: %d prompts, max_new=%d\n", n_prompts, max_new);

    int eos_id = ft_eos_id(model->tokenizer);
    int im_end_id = ft_token_to_id(model->tokenizer, "<|im_end|>");
    int rc = 0;
    for (int i = 0; i < n_prompts && rc == 0; i++)
        rc = run_one_prompt(model, state, i, prompts[i], max_new, eos_id, im_end_id);
    if (spec_enabled()) {
        long fwd = g_spec_verifies + g_spec_plain;
        long tok = g_spec_verifies + g_spec_accepted + g_spec_plain;
        fprintf(stderr,
                "eval_ppl: SPEC verifies=%ld plain=%ld drafted=%ld "
                "accepted=%ld accept_rate=%.2f tokens/forward=%.2f\n",
                g_spec_verifies, g_spec_plain, g_spec_drafted,
                g_spec_accepted,
                g_spec_drafted ? (double)g_spec_accepted / g_spec_drafted
                               : 0.0,
                fwd ? (double)tok / fwd : 0.0);
    }
    free(text);
    return rc;
}

int main(int argc, char **argv)
{
    int ai = 1;

#ifdef USE_CUDA
    /* Optional leading numeric arg = barswap die to acquire + bring nvidia up
     * (legacy limited-MMIO rigs; unnecessary once the small-BAR quirk maps all
     * dies at boot). */
    if (argc > ai && isdigit((unsigned char)argv[ai][0]) && argc > ai + 2) {
        int die = atoi(argv[ai++]);
        fprintf(stderr, "acquiring die %d ...\n", die);
        bs_context *bs = bs_init();
        if (!bs) { fprintf(stderr, "bs_init failed (bar_swap.ko loaded?)\n"); return 1; }
        if (!bs_acquire(bs, die)) { fprintf(stderr, "bs_acquire(%d) failed\n", die); return 1; }
        if (load_nvidia() != 0) { fprintf(stderr, "nvidia load failed\n"); return 1; }
        fprintf(stderr, "die %d acquired, nvidia up\n", die);
    }
#endif

    if (argc < ai + 3) {
        fprintf(stderr,
                "usage: %s [die] <model.gguf> ppl <corpus.txt> [max_tokens]\n"
                "       %s [die] <model.gguf> pplb <corpus.txt> [max_tokens]"
                " (CUDA: batched-prefill ppl)\n"
                "       %s [die] <model.gguf> gen <prompts.txt> [max_new_tokens]\n",
                argv[0], argv[0], argv[0]);
        return 1;
    }
    const char *model_path = argv[ai];
    const char *mode = argv[ai + 1];
    const char *input_path = argv[ai + 2];
    int limit = argc > ai + 3 ? atoi(argv[ai + 3]) : 0;

    /* MemoryAware: start the memory-watch loop BEFORE the load, exactly like
     * serve — (a) the GPU upload path feeds per-die VRAM truth in, (b) the
     * engine-level pn_admit gates the prefills below with real numbers. This
     * harness is the exact path that crash-looped the K80 test machine (§11 Q7), so it
     * must NOT run fail-open. Non-fatal if the loop cannot start. */
    pn_memwatch_start();

    pn_model *model = pn_load(model_path);
    if (!model) { fprintf(stderr, "eval_ppl: model load failed\n"); return 1; }
    pn_state *state = pn_state_new(model);
    if (!state) { fprintf(stderr, "eval_ppl: state alloc failed\n"); return 1; }

    int rc;
    if (strcmp(mode, "ppl") == 0) {
        rc = run_perplexity(model, state, input_path, limit);
#ifdef USE_CUDA
    } else if (strcmp(mode, "pplb") == 0) {
        rc = run_perplexity_batch(model, state, input_path, limit);
#endif
    } else if (strcmp(mode, "gen") == 0) {
        if (limit <= 0) limit = 48;
        if (limit > EVAL_MAX_GEN_TOKENS) limit = EVAL_MAX_GEN_TOKENS;
        rc = run_generation(model, state, input_path, limit);
    } else {
        fprintf(stderr, "eval_ppl: unknown mode '%s'\n", mode);
        rc = 1;
    }

    /* On the barswap rig, skip CUDA atexit teardown (it hangs/zombies) —
     * same escape hatch as test_kernel_parity. */
#ifdef USE_CUDA
    fflush(stdout); fflush(stderr);
    if (!getenv("QWEN35_EXIT_CLEAN"))
        _Exit(rc);
#endif
    pn_state_free(state);
    pn_model_free(model);
    return rc;
}
