/* SPDX-License-Identifier: AGPL-3.0-or-later */
/* Copyright (C) 2026 David (Corey) Herrera */
/**
 * @file test_tt_prefill_parity.c
 * @brief BYTE-EXACTNESS of the batched TT prefill against the one-at-a-time path.
 *
 * ---------------------------------------------------------------------------
 * WHY BYTES AND NOT PCC
 * ---------------------------------------------------------------------------
 * A batched prefill changes the SHAPE of the pass, not the arithmetic: 32 prompt positions
 * ride rows 0..31 of tiles whose rows 1..31 were previously zero padding the FPU multiplied
 * anyway. Every claim that makes this safe is a claim about INDEPENDENCE — that row m of a
 * matmul reads only row m of A, that a reduce block reduces each of its 32 rows over itself,
 * that an elementwise op is per element. Independence is exactly what a correlation cannot
 * see: PCC saturated at 1.000000000 on a tensor with relRMS 0.235 in this project, and it
 * caught none of four kernel mutations on norm-shaped ops (foundry/tt/DEV_NOTES.md §3). If
 * the two paths are the same arithmetic they are the same BYTES, and anything less than that
 * is a defect however small it scores.
 *
 * ---------------------------------------------------------------------------
 * WHAT IS COMPARED, AND WHY IT IS SUFFICIENT
 * ---------------------------------------------------------------------------
 * Two arms over the same prompt on the same card in the same process:
 *
 *   A. SEQUENTIAL — `pn_llama_forward_tt` once per prompt token, the shipping path.
 *   B. BATCHED    — `pn_llama_prefill_tt`, which runs tokens 0..P-2 in passes of up to 32
 *                   rows and hands token P-1 to `pn_llama_forward_tt`.
 *
 * and it compares:
 *
 *   1. **the whole KV cache**, every layer's K and V, byte for byte over the P positions
 *      actually written. This is the primary instrument: the KV cache is the ENTIRE state a
 *      prefill produces, so two prefills that agree on it are the same prefill.
 *   2. **the last position's logits**, byte for byte. This is a consequence rather than an
 *      independent check — arm B's final token goes through the same function arm A's does —
 *      and it is compared anyway because a consequence that fails is evidence the premise is
 *      wrong.
 *
 * The KV comparison is the strict one and it is what a divergence will show up in first: the
 * logits of the last position depend on the whole cache, but they are one row of one softmax
 * and could in principle absorb a small error in an early position. The cache cannot absorb
 * anything.
 *
 * ---------------------------------------------------------------------------
 * PROVING IT CAN FAIL
 * ---------------------------------------------------------------------------
 * `PN_TT_RED=<name>` breaks the batched arm in one named way and the harness must then
 * report DIFFER. Four vacuous gates have been caught in this project, so a green run of this
 * file means nothing on its own; `tests/run_tt_prefill_red.sh` is the arm that makes it mean
 * something. The mutations are documented at `llama_tt_red` in llama_forward_tt.c:
 *
 *   prefill_pos      every row rotates and appends at the pass's FIRST position
 *   prefill_causal   the per-row loop runs in reverse, so a row reads later rows' K/V
 *   prefill_rowfill  q/k/v rows 1..M-1 are overwritten with row 0 — the bug of a transfer
 *                    that still only moves logical row 0
 *
 * A fourth control needs no mutation at all: `PN_TT_PREFILL_ROWS=1` makes arm B one row per
 * pass, which is the sequential computation reached through the batched code. It must be
 * byte-identical, and if it is not, the divergence is in the batched PLUMBING rather than in
 * the batching.
 *
 * ---------------------------------------------------------------------------
 * USAGE
 * ---------------------------------------------------------------------------
 *   PN_TT_ARTIFACT=<...fttw> PN_TT_ONCARD=norm,swiglu,residual,topk \
 *     ./tests/test_tt_prefill_parity <model.gguf> [--tokens N] [--seed S]
 *
 * ON-CARD ATTENTION IS DELIBERATELY NOT SUPPORTED ON THE LLAMA ARM and the batched path
 * refuses it by name: `pn_tt_rope`, `pn_tt_kv_append` and the flash decode each take ONE
 * position, so a 32-row pass has no way to express 32 of them. The harness says so and exits
 * non-zero rather than quietly scoring a refusal as a pass — a refusal that reads as green is
 * how a gate becomes vacuous.
 *
 * ---------------------------------------------------------------------------
 * THE ORNITH ARM, AND WHY IT COMPARES A DIFFERENT THING
 * ---------------------------------------------------------------------------
 * Ornith (`qwen35`) reaches this file through `pn_qwen35moe_prefill_tt`, and the state a
 * prefill produces there is NOT a host KV cache. It is 24 layers of Gated-Delta-Net
 * recurrence carried in card DRAM plus 8 layers of K/V in card DRAM — **120 MiB of GDN state
 * alone**, and the card->host direction measures **5.3 MB/s flat from 16 KB to 786 KB**
 * (`bench_tt_prefill_quantum`, the TT host, 2026-08-21), so capturing it twice would cost ~46 s of
 * link per comparison and would itself perturb the residencies.
 *
 * So the Ornith arm compares **the last prompt position's logit row, byte for byte**. That is
 * a consequence rather than an independent check, and it is stated as one — but it is a
 * consequence of the WHOLE card state: the row is produced by the ordinary single-token
 * forward reading every one of the 24 recurrent states and all 8 K/V regions the prefill
 * left. A batched pass that wrote one position's state to the wrong slot, ran the recurrence
 * out of order, or served row 0's projection to all 32 rows cannot leave that row intact.
 * The three red arms in `qwen35moe_prefill_tt.c` are what makes that a measurement.
 *
 * **It REFUSES `PN_TT_SPARSE_LOGITS`.** That knob fills 248,312 of 248,320 entries with
 * `-INFINITY`, so two rows would agree everywhere the card did not name — the eighth
 * vacuous-gate class recorded in this project, and the reason it is a refusal and not a
 * warning.
 *
 * @status Working. The llama arm compares the KV cache; the Ornith arm compares the last
 *         logit row and says why.
 * @issues None
 * @todo None
 */

#include "../include/pantheon.h"
#include "../src/core/engine.h"
#include "../src/models/llama.h"
#include "../src/models/qwen35moe.h"
#include "tensor/tensor.h"

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

#ifndef USE_TT
int main(void)
{
    fprintf(stderr, "test_tt_prefill_parity: built without USE_TT — nothing to compare.\n");
    return 77;
}
#else

/** Default prompt length. 100 crosses three 32-row passes AND leaves a ragged last one
 *  (32 + 32 + 32 + 3 + the unbatched final token), so the remainder path is exercised by
 *  the default rather than only by an option nobody passes. */
#define PARITY_DEFAULT_TOKENS 100

/** A deterministic pseudo-prompt. Real ids matter less here than COVERAGE — every row of
 *  every pass must carry a different vector, or a bug that copies row 0 everywhere would be
 *  invisible. A fixed LCG gives that and makes the run reproducible. */
static double parity_now(void)
{
    struct timespec ts;

    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (double)ts.tv_sec + (double)ts.tv_nsec * 1e-9;
}

static uint32_t parity_token(uint32_t *rng, int n_vocab)
{
    *rng = *rng * 1664525u + 1013904223u;
    return (uint32_t)((*rng >> 8) % (uint32_t)n_vocab);
}

/** Copy every layer's K and V for positions [0, positions) into one packed fp32 buffer.
 *  Packed rather than compared in place because the two arms need two snapshots alive at
 *  once, and because a single buffer makes the comparison one memcmp per layer-half. */
static float *parity_capture_kv(pn_model *model, pn_state *state, int positions,
                                size_t *out_floats)
{
    pn_llama_kv_cache *kv = (pn_llama_kv_cache *)state->kv_cache;
    pn_config *cfg = &model->config;
    int head_dim = pn_head_dim(cfg);
    int n_kv_head = cfg->n_kv_head;
    size_t per_half = (size_t)n_kv_head * (size_t)positions * (size_t)head_dim;
    size_t total = per_half * 2u * (size_t)cfg->n_layer;
    float *out = malloc(total * sizeof(float));
    size_t at = 0;

    if (!out)
        return NULL;
    for (int layer = 0; layer < cfg->n_layer; layer++) {
        const float *k = (const float *)kv->k_cache[layer]->data;
        const float *v = (const float *)kv->v_cache[layer]->data;

        for (int h = 0; h < n_kv_head; h++) {
            memcpy(out + at, k + (size_t)h * kv->max_seq * head_dim,
                   (size_t)positions * head_dim * sizeof(float));
            at += (size_t)positions * head_dim;
        }
        for (int h = 0; h < n_kv_head; h++) {
            memcpy(out + at, v + (size_t)h * kv->max_seq * head_dim,
                   (size_t)positions * head_dim * sizeof(float));
            at += (size_t)positions * head_dim;
        }
    }
    *out_floats = total;
    return out;
}

/** First differing float index, or -1. Reported as an INDEX and a pair of values, because
 *  "they differ" without a where is not a diagnosis. */
static long parity_first_diff(const float *a, const float *b, size_t n)
{
    const unsigned char *pa = (const unsigned char *)a;
    const unsigned char *pb = (const unsigned char *)b;
    size_t bytes = n * sizeof(float);

    for (size_t i = 0; i < bytes; i++)
        if (pa[i] != pb[i])
            return (long)(i / sizeof(float));
    return -1;
}

/**
 * WHEN THE TWO ROWS DIFFER, SAY BY HOW MUCH — because one arm this harness now scores is
 * KNOWN not to be byte-identical, and printing only "DIFFER" makes that arm unreadable.
 *
 * The batched ATTENTION chunk is BIT-EXACT against the decode descriptors it replaces
 * (max-abs 0.000e+00, row for row), which is why the batched arm alone is byte-identical and
 * must stay so. The two CHUNKWISE GDN mixers are not: they are the same recurrence with FEWER
 * ROUNDINGS — PHASE CONV crosses one more unpack stage than the decode form, and the
 * chunkwise delta rule measures y 8.6e-03 against 32 sequential decode steps' 2.1e-02 through
 * the same Tf32 unpack. Byte-identity against a per-position path is therefore unreachable BY
 * CONSTRUCTION for that arm, and the honest question is a magnitude.
 *
 * MAX-NORM RELATIVE, `max|a-b| / max|a|`, which is the metric every recorded figure for these
 * two ops is in. The ARGMAX is reported beside it because it is what a greedy completion
 * actually consumes.
 */
static void parity_magnitude(const float *a, const float *b, size_t n, const char *label)
{
    double max_abs = 0.0, max_ref = 0.0, sum_sq = 0.0, ref_sq = 0.0;
    size_t at = 0, over = 0, finite = 0, argmax_a = 0, argmax_b = 0;

    for (size_t i = 0; i < n; i++) {
        const double x = (double)a[i], y = (double)b[i];
        double gap;

        /* A SPARSE ROW IS -INFINITY ALMOST EVERYWHERE and this harness refuses one, but a
         * -INFINITY can still reach a dense row — and `inf - inf` is NaN, which would make
         * every figure below NaN with nothing saying so. */
        if (!(x > -1e30 && x < 1e30) || !(y > -1e30 && y < 1e30))
            continue;
        finite++;
        gap = x > y ? x - y : y - x;
        if (gap > max_abs) { max_abs = gap; at = i; }
        if ((x < 0 ? -x : x) > max_ref) max_ref = (x < 0 ? -x : x);
        sum_sq += gap * gap;
        ref_sq += x * x;
        if (gap > 1e-2) over++;
        if (x > (double)a[argmax_a]) argmax_a = i;
        if (y > (double)b[argmax_b]) argmax_b = i;
    }
    printf("PARITY %s max-norm relative %.3e (max|delta| %.4g at float %zu of %zu finite), "
           "relRMS %.3e, %zu past 1e-2\n",
           label, max_ref > 0.0 ? max_abs / max_ref : 0.0, max_abs, at, finite,
           ref_sq > 0.0 ? sqrt(sum_sq / ref_sq) : 0.0, over);
    printf("PARITY %s ARGMAX sequential %zu (%.6g) vs batched %zu (%.6g) — %s\n", label,
           argmax_a, (double)a[argmax_a], argmax_b, (double)b[argmax_b],
           argmax_a == argmax_b ? "THE SAME TOKEN" : "A DIFFERENT TOKEN");
}

int main(int argc, char **argv)
{
    const char *model_path = NULL;
    int want_tokens = PARITY_DEFAULT_TOKENS;
    uint32_t rng = 0x5eed1234u;
    pn_model *model;
    pn_state *sa, *sb;
    uint32_t *prompt;
    float *kv_a = NULL, *kv_b = NULL;
    float *log_a = NULL, *log_b = NULL;
    size_t kv_floats_a = 0, kv_floats_b = 0;
    tm_tensor *la = NULL, *lb = NULL;
    long kv_diff = -1, logit_diff;
    double t_seq = 0.0, t_batch = 0.0, mark;
    int ornith;
    int rc = 1;

    for (int i = 1; i < argc; i++) {
        if (strcmp(argv[i], "--tokens") == 0 && i + 1 < argc)
            want_tokens = atoi(argv[++i]);
        else if (strcmp(argv[i], "--seed") == 0 && i + 1 < argc)
            rng = (uint32_t)strtoul(argv[++i], NULL, 0);
        else if (!model_path)
            model_path = argv[i];
    }
    if (!model_path || want_tokens < 2) {
        fprintf(stderr, "usage: %s <model.gguf> [--tokens N>=2] [--seed S]\n", argv[0]);
        return 2;
    }

    model = pn_load(model_path);
    if (!model) {
        fprintf(stderr, "PARITY REFUSED: cannot load %s\n", model_path);
        return 2;
    }
    /* THE VACUOUS-RUN GUARD. Without the card there is no batched path to test and the two
     * arms would be the same CPU code compared against itself — a green run that proves
     * nothing, which is the exact failure this project has paid for four times. */
    ornith = (model->forward == pn_qwen35moe_forward_tt);
    if (model->forward != pn_llama_forward_tt && !ornith) {
        fprintf(stderr, "PARITY REFUSED: the engine did not select a Tenstorrent forward. "
                        "Set PN_TT_ARTIFACT and check the card. Comparing the CPU path "
                        "against itself would be vacuous.\n");
        pn_model_free(model);
        return 2;
    }
    /* A SPARSE ROW IS -INFINITY IN 248,312 OF 248,320 ENTRIES, so two of them agree
     * everywhere the card did not name. Scoring that is the eighth vacuous-gate class in
     * this project, and the Ornith arm has nothing to compare BUT that row. */
    if (ornith && getenv("PN_TT_SPARSE_LOGITS")) {
        fprintf(stderr, "PARITY REFUSED: PN_TT_SPARSE_LOGITS is set. The Ornith arm compares "
                        "the last position's LOGIT ROW, and a sparse row is -INFINITY almost "
                        "everywhere — two of them would agree for a reason that is not "
                        "parity. Unset it.\n");
        pn_model_free(model);
        return 2;
    }

    prompt = malloc((size_t)want_tokens * sizeof(*prompt));
    sa = pn_state_new(model);
    sb = pn_state_new(model);
    if (!prompt || !sa || !sb) {
        fprintf(stderr, "PARITY REFUSED: allocation failed\n");
        goto done;
    }
    for (int i = 0; i < want_tokens; i++)
        prompt[i] = parity_token(&rng, model->config.n_vocab);

    /* ── ARM A: the shipping path, one forward per prompt token. */
    mark = parity_now();
    for (int i = 0; i < want_tokens; i++) {
        if (la) tm_free(la);
        la = model->forward(model, prompt[i], sa);
        if (!la) {
            fprintf(stderr, "PARITY REFUSED: sequential forward failed at token %d\n", i);
            goto done;
        }
    }
    t_seq = parity_now() - mark;
    /* ORNITH'S PREFILL STATE IS IN CARD DRAM, not in a host KV cache: 120 MiB of carried
     * Gated-Delta-Net state plus 8 layers of K/V, over a link that reads at 5.3 MB/s. The
     * logit row below is the instrument there, and the header says why. */
    if (!ornith)
        kv_a = parity_capture_kv(model, sa, want_tokens, &kv_floats_a);

    /* ── ARM B: the batched path over the same ids from a fresh state. */
    mark = parity_now();
    lb = ornith ? pn_qwen35moe_prefill_tt(model, prompt, want_tokens, sb)
                : pn_llama_prefill_tt(model, prompt, NULL, want_tokens, sb);
    t_batch = parity_now() - mark;
    if (!lb) {
        fprintf(stderr, "PARITY REFUSED: the batched prefill REFUSED this configuration. "
                        "The llama arm refuses on-card attention (its rotary, KV append and "
                        "flash decode each take ONE position) and the diagnostic arms; the "
                        "Ornith arm defaults OFF and needs PN_TT_PREFILL=1, both on-card "
                        "mixers, and a sequence that starts at position 0. Scoring a refusal "
                        "as a pass is how a gate goes vacuous, so this is an error and not a "
                        "skip.\n");
        goto done;
    }
    if (!ornith) {
        kv_b = parity_capture_kv(model, sb, want_tokens, &kv_floats_b);
        if (!kv_a || !kv_b || kv_floats_a != kv_floats_b) {
            fprintf(stderr, "PARITY REFUSED: KV capture failed or shapes disagree\n");
            goto done;
        }
    }
    if (sa->pos != sb->pos) {
        fprintf(stderr, "PARITY FAIL position: sequential pos=%d batched pos=%d\n",
                sa->pos, sb->pos);
        goto done;
    }
    if (la->numel != lb->numel) {
        fprintf(stderr, "PARITY REFUSED: logit rows differ in length\n");
        goto done;
    }

    log_a = (float *)la->data;
    log_b = (float *)lb->data;
    if (!ornith)
        kv_diff = parity_first_diff(kv_a, kv_b, kv_floats_a);
    logit_diff = parity_first_diff(log_a, log_b, (size_t)la->numel);

    printf("PARITY family=%s tokens=%d rows=%s kv_floats=%zu vocab=%d pos=%d\n",
           ornith ? "ornith/qwen35" : "llama", want_tokens,
           getenv("PN_TT_PREFILL_ROWS") ? getenv("PN_TT_PREFILL_ROWS") : "32(default)",
           kv_floats_a, model->config.n_vocab, sa->pos);
    if (ornith)
        printf("PARITY kv_cache    NOT COMPARED — Ornith's prefill state is 120 MiB of "
               "card-resident GDN recurrence plus card K/V, and the link reads at 5.3 MB/s. "
               "The logit row below is the instrument.\n");
    else if (kv_diff < 0)
        printf("PARITY kv_cache    IDENTICAL over %zu floats\n", kv_floats_a);
    else
        printf("PARITY kv_cache    DIFFER first float %ld sequential=%.9g batched=%.9g\n",
               kv_diff, (double)kv_a[kv_diff], (double)kv_b[kv_diff]);
    if (logit_diff < 0)
        printf("PARITY last_logits IDENTICAL over %lld floats\n", (long long)la->numel);
    else {
        printf("PARITY last_logits DIFFER first float %ld sequential=%.9g batched=%.9g\n",
               logit_diff, (double)log_a[logit_diff], (double)log_b[logit_diff]);
        parity_magnitude(log_a, log_b, (size_t)la->numel, "last_logits");
    }

    /* PREFILL TOK/S, and it is COMPUTED tokens over the wall clock of the prefill alone —
     * both arms start from a fresh state and neither has a prefix cache in front of it, so
     * there is nothing here that a warm cache could inflate. Both arms include the same
     * unbatched final token, which is why the batched rate is a lower bound rather than a
     * best case. */
    printf("PARITY rate sequential %.3f s = %.2f tok/s | batched %.3f s = %.2f tok/s | "
           "speedup %.2fx\n", t_seq, (double)want_tokens / t_seq, t_batch,
           (double)want_tokens / t_batch, t_seq / t_batch);

    rc = (kv_diff < 0 && logit_diff < 0) ? 0 : 1;
    printf("PARITY VERDICT %s\n", rc == 0 ? "BYTE-IDENTICAL" : "DIFFER");

done:
    if (la) tm_free(la);
    if (lb) tm_free(lb);
    free(kv_a);
    free(kv_b);
    free(prompt);
    if (sa) pn_state_free(sa);
    if (sb) pn_state_free(sb);
    pn_model_free(model);
    return rc;
}

#endif /* USE_TT */
