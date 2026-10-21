/* SPDX-License-Identifier: AGPL-3.0-or-later */
/* Copyright (C) 2026 David (Corey) Herrera */
/**
 * @file bench_ngram_accept.c
 * @brief PHASE B's CHEAP GATE: what acceptance rate does prompt-lookup (n-gram) drafting
 *        actually reach on this model, on real traffic — measured BEFORE any verify
 *        machinery exists.
 *
 * ---------------------------------------------------------------------------
 * WHY THIS NEEDS NO VERIFY PATH, NO KERNEL AND NO ROLLBACK
 * ---------------------------------------------------------------------------
 * Greedy speculative decoding accepts a drafted token iff it equals the model's own argmax
 * at that position, and an accepted token leaves the prefix identical to the plain greedy
 * run's. So the whole acceptance distribution is a function of ONE ordinary greedy decode.
 * See ngram_accept_stream.h for the argument written out. This harness therefore:
 *
 *   --generate DUMP MODEL   decode the corpus greedily on the SHIPPED arm, one token per
 *                           forward, and record every token id. Needs the card.
 *   --score DUMP [DUMP...]  replay the recorded streams as speculative rounds and report
 *                           acceptance by K, by corpus, by phase and by context length.
 *                           Loads no model, opens no card, costs seconds.
 *   --selftest              known-answer arms that need neither.
 *
 * The drafter is `pn_ngram_draft` — the one that ships in src/core/ngram_draft.c, not a copy
 * written for the measurement.
 *
 * ---------------------------------------------------------------------------
 * WHAT MAKES THE NUMBER FALSIFIABLE
 * ---------------------------------------------------------------------------
 * This project has hit EIGHT vacuous-gate classes, including a gate that passed 132 checks on
 * a shape the model does not have. A drafting hit counter that can only count hits measures
 * nothing, so the same code path is run three more ways over the same bytes:
 *
 *   --selftest       a periodic stream (must hit the CEILING of K+1 tokens per forward) and
 *                    an i.i.d. stream (must sit at 1.00). Both ends of the scale, no card.
 *   control=shuffle  the real streams with the searched history permuted and only the query
 *                    suffix left intact — same tokens, no structure. Must collapse to ~1.00.
 *   control=random   the real rounds with the drafter's proposals replaced by uniform
 *                    vocabulary ids. Must collapse to ~1.00.
 *
 * ---------------------------------------------------------------------------
 * TEMPERATURE AND SPARSE LOGITS
 * ---------------------------------------------------------------------------
 * The generate pass decodes GREEDILY, which is what `--bench --temperature 0` does and what
 * every speed figure on this card was taken at. Only the ARGMAX of each row is used, so
 * `PN_TT_SPARSE_LOGITS` is safe here and is reported rather than refused — the card selects
 * the exact top-N and the argmax over the sparse row is the argmax over the dense one. Any
 * figure other than an argmax over that row would be noise, and this harness computes none.
 *
 * Under sampling the accept test becomes a rejection-sampling step rather than an equality,
 * so every acceptance figure below is the GREEDY UPPER BOUND.
 *
 * USAGE
 *   ./tests/bench_ngram_accept --generate OUT.bin MODEL.gguf --corpus F --tag NAME
 *                              [--n-gen N] [--max-docs N] [--system TEXT]
 *   ./tests/bench_ngram_accept --score DUMP.bin [DUMP.bin ...]
 *   ./tests/bench_ngram_accept --selftest
 *
 * Dumps are not shipped: record your own with --generate (tests/data/ngram_accept_dumps/
 * is gitignored for that purpose).
 *
 * NOT part of `make test`: --generate needs a model and a card.
 *
 * @status Working
 * @issues None
 * @todo None
 */
#include "../include/pantheon.h"
#include "../src/core/engine.h"
#include "tensor/tensor.h"

#include "../src/core/chat.h"

#include "chat_corpus.h"
#include "ngram_accept_stream.h"
#include "parity_accel_arm.h"
#include "tokenizer/tokenizer.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

#define ACCEPT_ECHO_CHARS   96
#define ACCEPT_MAX_GEN      4096
#define ACCEPT_CONTROL_SEED 0x5DEECE66DULL

/** The K values reported for every corpus. The task names 2, 4 and 8; the neighbours are
 *  free and show whether the curve has already flattened. */
static const int accept_k_ladder[] = { 1, 2, 3, 4, 6, 8, 16 };
#define ACCEPT_K_COUNT ((int)(sizeof(accept_k_ladder) / sizeof(accept_k_ladder[0])))

/** Trailing-history windows, in tokens; 0 = the whole context. Answers "does MORE history
 *  help", which is the mechanism behind any context-length trend. */
static const int accept_window_ladder[] = { 128, 512, 2048, 8192, 0 };
#define ACCEPT_WINDOW_COUNT ((int)(sizeof(accept_window_ladder) / sizeof(accept_window_ladder[0])))

static double accept_now_s(void)
{
    struct timespec ts;

    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (double)ts.tv_sec + (double)ts.tv_nsec * 1e-9;
}

static int accept_argmax(const float *x, int n)
{
    int best = 0;

    for (int i = 1; i < n; i++)
        if (x[i] > x[best])
            best = i;
    return best;
}

/* ================================================================
 * Pass 1 — the recorded greedy stream (needs the card)
 * ================================================================ */

/**
 * Report, once, whether the logit row is the sparse presentation.
 *
 * Not a refusal: this harness reads only the argmax, which is exact on that row. It is
 * printed because a reader of the dump has to know which presentation produced it — the
 * two-pass parity gate REFUSES the same row, and the difference has to be deliberate.
 */
static void accept_report_row_density(const float *row, int n)
{
    static int checked;
    int non_finite = 0;

    if (checked)
        return;
    checked = 1;
    for (int i = 0; i < n; i++)
        if (!(row[i] > -1e30f && row[i] < 1e30f))
            non_finite++;
    printf("ROW n=%d non_finite=%d presentation=%s (only the ARGMAX is read)\n", n, non_finite,
           non_finite * 2 >= n ? "SPARSE" : "dense");
}

/**
 * Prefill `prompt` and greedily decode up to `n_gen` tokens into `doc`.
 *
 * @return 0 on success, -1 if a forward failed
 */
static int accept_decode_document(pn_model *model, const uint32_t *prompt, int n_prompt,
                                  int n_gen, int eos, int think_end, ngram_document *doc)
{
    pn_state *state = pn_state_new(model);
    tm_tensor *logits = NULL;
    double t_prefill, t0;
    int n = 0;

    if (!state)
        return -1;
    doc->prompt = malloc((size_t)n_prompt * sizeof(*doc->prompt));
    doc->gen = malloc((size_t)n_gen * sizeof(*doc->gen));
    if (!doc->prompt || !doc->gen) {
        pn_state_free(state);
        return -1;
    }
    memcpy(doc->prompt, prompt, (size_t)n_prompt * sizeof(*doc->prompt));
    doc->n_prompt = n_prompt;
    doc->think_end_pos = -1;

    t0 = accept_now_s();
    for (int i = 0; i < n_prompt; i++) {
        if (logits)
            tm_free(logits);
        logits = model->forward(model, prompt[i], state);
        if (!logits) {
            fprintf(stderr, "generate: forward failed prefilling position %d\n", i);
            pn_state_free(state);
            return -1;
        }
    }
    t_prefill = accept_now_s() - t0;
    accept_report_row_density(tm_data_f32(logits), (int)logits->shape[0]);

    t0 = accept_now_s();
    while (n < n_gen) {
        uint32_t tok = (uint32_t)accept_argmax(tm_data_f32(logits), model->config.n_vocab);

        tm_free(logits);
        doc->gen[n] = tok;
        if (think_end >= 0 && doc->think_end_pos < 0 && (int)tok == think_end)
            doc->think_end_pos = n;
        n++;
        if ((int)tok == eos || n >= n_gen)
            break;
        logits = model->forward(model, tok, state);
        if (!logits) {
            fprintf(stderr, "generate: forward failed generating token %d\n", n);
            break;
        }
    }
    doc->n_gen = n;
    doc->decode_s = accept_now_s() - t0;
    printf("   prefill %d tok in %.1f s (%.3f s/tok) | decode %d tok in %.1f s (%.3f tok/s) "
           "| think_end_pos=%d\n", n_prompt, t_prefill, t_prefill / (double)n_prompt, n,
           doc->decode_s, (double)n / doc->decode_s, doc->think_end_pos);
    pn_state_free(state);
    return n > 1 ? 0 : -1;
}

/** Echo the first characters of a token run so a reader can see it is real text. */
static void accept_echo(pn_model *model, const uint32_t *ids, int n, const char *label)
{
    char *text = chat_ids_to_text(model, ids, n);

    printf("   %s \"", label);
    for (int i = 0; text && text[i] && i < ACCEPT_ECHO_CHARS; i++)
        putchar(text[i] == '\n' ? ' ' : text[i]);
    printf("...\"\n");
    free(text);
}

/**
 * Slurp a corpus and split it into one system prompt and N user messages.
 *
 * The blocks are separated by lines reading exactly `===PROMPT===`; a block introduced by
 * `===SYSTEM===` is the system prompt instead. A line-tagged format (the chat-parity
 * corpus's `U<TAB>`) cannot carry a source file, because source has blank lines in it and a
 * blank line ends a document there — and a code-edit prompt with its blank lines stripped is
 * not the traffic this is trying to measure.
 *
 * @return number of user messages; out[] points into the returned buffer, which the caller
 *         keeps alive for the run
 */
static int accept_load_corpus(const char *path, char **buffer_out, const char **system_out,
                              char **out, int max_out)
{
    char *buf, *cur;
    long len;
    int n = 0, next_is_system = 0;
    FILE *f = fopen(path, "rb");

    if (!f)
        return -1;
    fseek(f, 0, SEEK_END);
    len = ftell(f);
    fseek(f, 0, SEEK_SET);
    buf = malloc((size_t)len + 2);
    if (!buf || len <= 0 || fread(buf, 1, (size_t)len, f) != (size_t)len) {
        fclose(f);
        free(buf);
        return -1;
    }
    buf[len] = '\0';
    fclose(f);
    *buffer_out = buf;

    cur = buf;
    while (cur && *cur && n < max_out) {
        char *sep = strstr(cur, "\n===PROMPT===\n");
        char *sys = strstr(cur, "\n===SYSTEM===\n");
        int sys_first = sys && (!sep || sys < sep);
        char *at = sys_first ? sys : sep;

        if (strncmp(cur, "===SYSTEM===\n", 13) == 0) {
            next_is_system = 1;
            cur += 13;
            continue;
        }
        if (strncmp(cur, "===PROMPT===\n", 13) == 0) {
            next_is_system = 0;
            cur += 13;
            continue;
        }
        if (at)
            *at = '\0';
        if (*cur) {
            if (next_is_system)
                *system_out = cur;
            else
                out[n++] = cur;
        }
        next_is_system = sys_first;
        cur = at ? at + 14 : NULL;
    }
    return n;
}

#define ACCEPT_MAX_PROMPTS 32

static int accept_run_generate(const char *dump_path, const char *model_path,
                               const char *corpus_path, const char *tag, int n_gen,
                               int max_docs, const char *system_override)
{
    static uint32_t ids[CHAT_CORPUS_MAX_IDS];
    char *prompts[ACCEPT_MAX_PROMPTS];
    const char *system = NULL;
    char *corpus_buffer = NULL;
    const char *arm_name = "?";
    pn_forward_fn arm;
    pn_model *model;
    FILE *dump = NULL;
    int n_prompts, docs = 0, think_end, eos, rc = 0;

    n_prompts = accept_load_corpus(corpus_path, &corpus_buffer, &system, prompts,
                                   ACCEPT_MAX_PROMPTS);
    if (n_prompts <= 0) {
        fprintf(stderr, "generate: no prompts in %s\n", corpus_path);
        free(corpus_buffer);
        return 1;
    }
    if (system_override)
        system = system_override;
    if (n_prompts > max_docs)
        n_prompts = max_docs;
    model = pn_load(model_path);
    if (!model) {
        fprintf(stderr, "generate: load failed\n");
        free(corpus_buffer);
        return 1;
    }
    /* THE ARM HAS TO BE THE ONE THAT SHIPS. Acceptance is a property of the model's own
     * output distribution, so a CPU-forward stream would be measuring a different model.
     * parity_select_accel_arm returns NULL when the engine left the model on its CPU
     * forward, which is a refusal rather than a quiet substitution. */
    arm = parity_select_accel_arm(model, &arm_name);
    if (!arm) {
        fprintf(stderr, "generate: REFUSED — no accelerator arm. Acceptance measured on a "
                        "different forward is not the shipped model's acceptance.\n");
        pn_model_free(model);
        free(corpus_buffer);
        return 1;
    }
    model->forward = arm;
    eos = ft_eos_id(model->tokenizer);
    think_end = ft_token_to_id(model->tokenizer, "</think>");

    printf("GENERATE arm=%s arch=%d n_vocab=%d chat_style=%d eos=%d think_end=%d corpus=%s "
           "prompts=%d tag=%s n_gen=%d out=%s\n", arm_name, (int)model->config.arch,
           model->config.n_vocab, model->chat_style, eos, think_end, corpus_path, n_prompts,
           tag, n_gen, dump_path);
    printf("SYSTEM \"%.*s%s\"\n", 160, system ? system : "(none)",
           system && strlen(system) > 160 ? "..." : "");
    fflush(stdout);

    if (ngram_stream_open_write(dump_path, (uint32_t)model->config.n_vocab,
                                (uint32_t)model->config.arch, (uint32_t)eos,
                                think_end < 0 ? NGRAM_STREAM_NONE : (uint32_t)think_end,
                                arm_name, tag, &dump) != 0) {
        pn_model_free(model);
        free(corpus_buffer);
        return 1;
    }

    for (docs = 0; docs < n_prompts; ) {
        ngram_document doc = {0};
        pn_token_seq seq = pn_chat_format_first((pn_chat_style)model->chat_style,
                                                model->tokenizer, system, prompts[docs]);
        int n_ids = seq.n_tokens;

        if (!seq.tokens || n_ids <= 0 || n_ids > CHAT_CORPUS_MAX_IDS) {
            fprintf(stderr, "generate: prompt %d did not tokenise (%d ids)\n", docs, n_ids);
            free(seq.tokens);
            rc = 1;
            break;
        }
        memcpy(ids, seq.tokens, (size_t)n_ids * sizeof(*ids));
        free(seq.tokens);
        printf("-- doc %d: %d prompt tokens\n", docs, n_ids);
        accept_echo(model, ids, n_ids, "prompt");
        fflush(stdout);
        if (accept_decode_document(model, ids, n_ids, n_gen, eos, think_end, &doc) != 0) {
            free(doc.prompt);
            free(doc.gen);
            rc = 1;
            break;
        }
        accept_echo(model, doc.gen, doc.n_gen, "output");
        if (ngram_stream_write_document(dump, &doc) != 0) {
            fprintf(stderr, "generate: dump write failed at document %d\n", docs);
            rc = 1;
        }
        free(doc.prompt);
        free(doc.gen);
        if (rc != 0)
            break;
        docs++;
        fflush(stdout);
    }
    /* The end marker is what tells the scorer the writer FINISHED. It is written on a clean
     * early break too — the case it exists to catch is a killed process, which never reaches
     * here at all, and throwing away hours of card time because the last document refused to
     * tokenise would be its own defect. The document count is printed on both sides. */
    if (ngram_stream_close_write(dump) != 0)
        rc = 1;
    printf("GENERATE %s: %d document(s)\n", rc == 0 ? "complete" : "FAILED", docs);
    pn_model_free(model);
    free(corpus_buffer);
    return rc;
}

/* ================================================================
 * Pass 2 — scoring (no model, no card)
 * ================================================================ */

/** The drafter's key lengths. 3..2 is what the shipped throttle and the CUDA bench use. */
#define ACCEPT_NGRAM_MAX 3
#define ACCEPT_NGRAM_MIN 2

/** Run one (K, window, control, n-gram) cell over a whole dump. */
static void accept_cell_ng(const ngram_stream *s, int max_k, int window, ngram_control control,
                           int ng_max, int ng_min, ngram_accept_stats *phase)
{
    memset(phase, 0, sizeof(*phase) * NGRAM_PHASE_COUNT);
    for (int d = 0; d < s->n_docs; d++)
        ngram_accept_replay(&s->docs[d], max_k, ng_max, ng_min, window, control, s->n_vocab,
                            ACCEPT_CONTROL_SEED + (uint64_t)d, phase);
}

/** The same at the shipped key lengths. */
static void accept_cell(const ngram_stream *s, int max_k, int window, ngram_control control,
                        ngram_accept_stats *phase)
{
    accept_cell_ng(s, max_k, window, control, ACCEPT_NGRAM_MAX, ACCEPT_NGRAM_MIN, phase);
}

static void accept_print_row(const char *label, const ngram_accept_stats *a, int max_k)
{
    long rounds = a->rounds + a->plain_rounds;

    if (!rounds) {
        printf("  %-14s (no rounds)\n", label);
        return;
    }
    printf("  %-14s tok/fwd=%.3f | draft_rounds=%.0f%% accept=%.1f%% (%ld/%ld) | "
           "E[delivered|drafted]=%.3f | emitted=%ld forwards=%ld\n",
           label, ngram_tokens_per_forward(a), 100.0 * (double)a->rounds / (double)rounds,
           a->drafted ? 100.0 * (double)a->accepted / (double)a->drafted : 0.0, a->accepted,
           a->drafted, a->rounds ? 1.0 + (double)a->accepted / (double)a->rounds : 0.0,
           a->emitted, a->forwards);
    printf("  %-14s delivered histogram:", "");
    for (int i = 1; i <= max_k + 1; i++)
        printf(" %d:%ld", i, a->hist[i]);
    printf("\n");
}

/** The K ladder, split into reasoning and answer, plus the combined figure. */
static void accept_report_k_ladder(const ngram_stream *s)
{
    printf("\n== K LADDER  (window=all, ngram 3..2, control=none)\n");
    for (int i = 0; i < ACCEPT_K_COUNT; i++) {
        ngram_accept_stats phase[NGRAM_PHASE_COUNT], all = {0};

        accept_cell(s, accept_k_ladder[i], 0, NGRAM_CONTROL_NONE, phase);
        ngram_accept_stats_add(&all, &phase[NGRAM_PHASE_REASONING]);
        ngram_accept_stats_add(&all, &phase[NGRAM_PHASE_ANSWER]);
        printf("-- K=%d (verify %d rows)\n", accept_k_ladder[i], accept_k_ladder[i] + 1);
        if (phase[NGRAM_PHASE_REASONING].forwards) {
            accept_print_row("reasoning", &phase[NGRAM_PHASE_REASONING], accept_k_ladder[i]);
            accept_print_row("answer", &phase[NGRAM_PHASE_ANSWER], accept_k_ladder[i]);
        }
        accept_print_row("combined", &all, accept_k_ladder[i]);
    }
}

/** Acceptance against how much history the drafter is allowed to search. */
static void accept_report_windows(const ngram_stream *s, int max_k)
{
    printf("\n== HISTORY WINDOW at K=%d — does MORE context help?\n", max_k);
    for (int i = 0; i < ACCEPT_WINDOW_COUNT; i++) {
        ngram_accept_stats phase[NGRAM_PHASE_COUNT], all = {0};
        char label[32];

        accept_cell(s, max_k, accept_window_ladder[i], NGRAM_CONTROL_NONE, phase);
        ngram_accept_stats_add(&all, &phase[NGRAM_PHASE_REASONING]);
        ngram_accept_stats_add(&all, &phase[NGRAM_PHASE_ANSWER]);
        if (accept_window_ladder[i])
            snprintf(label, sizeof(label), "window %d", accept_window_ladder[i]);
        else
            snprintf(label, sizeof(label), "window all");
        accept_print_row(label, &all, max_k);
    }
}

/**
 * Per-document tokens/forward, so a corpus mean cannot hide a bimodal split.
 *
 * A single blended figure over eight documents is exactly the shape that hid the reasoning
 * / answer bimodality the phase note warned about; the same argument applies one level down.
 */
static void accept_report_per_document(const ngram_stream *s, int max_k)
{
    printf("\n== PER DOCUMENT at K=%d — is the mean a mean, or one document?\n", max_k);
    for (int d = 0; d < s->n_docs; d++) {
        ngram_accept_stats phase[NGRAM_PHASE_COUNT], all = {0};

        memset(phase, 0, sizeof(phase));
        ngram_accept_replay(&s->docs[d], max_k, ACCEPT_NGRAM_MAX, ACCEPT_NGRAM_MIN, 0,
                            NGRAM_CONTROL_NONE, s->n_vocab, ACCEPT_CONTROL_SEED, phase);
        ngram_accept_stats_add(&all, &phase[NGRAM_PHASE_REASONING]);
        ngram_accept_stats_add(&all, &phase[NGRAM_PHASE_ANSWER]);
        printf("  doc %d  %5d prompt %5d gen  tok/fwd=%.3f  accept=%.1f%%  reasoning=%.3f "
               "answer=%.3f%s\n", d, s->docs[d].n_prompt, s->docs[d].n_gen,
               ngram_tokens_per_forward(&all),
               all.drafted ? 100.0 * (double)all.accepted / (double)all.drafted : 0.0,
               ngram_tokens_per_forward(&phase[NGRAM_PHASE_REASONING]),
               ngram_tokens_per_forward(&phase[NGRAM_PHASE_ANSWER]),
               s->docs[d].think_end_pos < 0
                   ? "   [NO </think>: the whole run is reasoning, counted as answer]" : "");
    }
}

/** Acceptance against the context length the round actually ran at. */
static void accept_report_length(const ngram_stream *s, int max_k)
{
    ngram_accept_stats phase[NGRAM_PHASE_COUNT], all = {0};

    accept_cell(s, max_k, 0, NGRAM_CONTROL_NONE, phase);
    ngram_accept_stats_add(&all, &phase[NGRAM_PHASE_REASONING]);
    ngram_accept_stats_add(&all, &phase[NGRAM_PHASE_ANSWER]);
    printf("\n== CONTEXT LENGTH at K=%d — tok/fwd by history length at round start\n", max_k);
    for (int b = 0; b < NGRAM_LEN_BUCKETS; b++) {
        if (!all.bucket_forwards[b])
            continue;
        printf("  %-8s tok/fwd=%.3f  (%ld forwards, %ld tokens)\n", ngram_length_bucket_name[b],
               (double)all.bucket_emitted[b] / (double)all.bucket_forwards[b],
               all.bucket_forwards[b], all.bucket_emitted[b]);
    }
}

/**
 * The drafter's KEY LENGTH, swept.
 *
 * Free to vary offline and it is a real lever: a short key hits more often and proposes worse,
 * a long key the reverse. Reporting only the shipped 3..2 would leave the phase's verdict
 * resting on a parameter nobody had varied.
 */
static void accept_report_ngram(const ngram_stream *s, int max_k)
{
    static const int ng[][2] = { {2, 1}, {2, 2}, {3, 2}, {4, 2}, {6, 3}, {8, 4} };
    char label[32];

    printf("\n== N-GRAM KEY LENGTH at K=%d\n", max_k);
    for (int i = 0; i < (int)(sizeof(ng) / sizeof(ng[0])); i++) {
        ngram_accept_stats phase[NGRAM_PHASE_COUNT], all = {0};

        accept_cell_ng(s, max_k, 0, NGRAM_CONTROL_NONE, ng[i][0], ng[i][1], phase);
        ngram_accept_stats_add(&all, &phase[NGRAM_PHASE_REASONING]);
        ngram_accept_stats_add(&all, &phase[NGRAM_PHASE_ANSWER]);
        snprintf(label, sizeof(label), "ngram %d..%d", ng[i][0], ng[i][1]);
        accept_print_row(label, &all, max_k);
    }
}

/** The two controls that must read LOW on the same bytes as the measurement. */
static void accept_report_controls(const ngram_stream *s, int max_k)
{
    static const ngram_control ladder[] = { NGRAM_CONTROL_NONE, NGRAM_CONTROL_SHUFFLE,
                                            NGRAM_CONTROL_RANDOM };
    static const char *const name[] = { "measurement", "CTRL shuffle", "CTRL random" };

    printf("\n== CONTROLS at K=%d — the two below must collapse to ~1.00 tok/fwd\n", max_k);
    for (int i = 0; i < 3; i++) {
        ngram_accept_stats phase[NGRAM_PHASE_COUNT], all = {0};

        accept_cell(s, max_k, 0, ladder[i], phase);
        ngram_accept_stats_add(&all, &phase[NGRAM_PHASE_REASONING]);
        ngram_accept_stats_add(&all, &phase[NGRAM_PHASE_ANSWER]);
        accept_print_row(name[i], &all, max_k);
    }
}

/** One line of the cross-corpus summary — the table the verdict is read off. */
typedef struct {
    char   tag[NGRAM_STREAM_NAME_LEN];
    int    docs;
    long   gen_tokens;
    double control_tok_s;
    double tpf[3];        /**< tokens per forward at K = 2, 4, 8 */
    double accept_k4;     /**< accepted / drafted at K=4, per cent */
    double shuffle_k4;    /**< the control that must read ~1.00 */
    double random_k4;
    long   reasoning_forwards;
    double tpf_reasoning_k4;
    double tpf_answer_k4;
} accept_summary_row;

/** tokens/forward at one K over both phases. */
static double accept_tpf_at(const ngram_stream *s, int max_k, ngram_control control)
{
    ngram_accept_stats phase[NGRAM_PHASE_COUNT], all = {0};

    accept_cell(s, max_k, 0, control, phase);
    ngram_accept_stats_add(&all, &phase[NGRAM_PHASE_REASONING]);
    ngram_accept_stats_add(&all, &phase[NGRAM_PHASE_ANSWER]);
    return ngram_tokens_per_forward(&all);
}

/** Fill one summary row from a dump. */
static void accept_fill_summary(const ngram_stream *s, accept_summary_row *row)
{
    ngram_accept_stats phase[NGRAM_PHASE_COUNT], all = {0};
    static const int k_of[3] = { 2, 4, 8 };

    snprintf(row->tag, sizeof(row->tag), "%s", s->tag);
    row->docs = s->n_docs;
    for (int d = 0; d < s->n_docs; d++) {
        row->gen_tokens += s->docs[d].n_gen;
        row->control_tok_s += s->docs[d].decode_s;
    }
    row->control_tok_s = row->control_tok_s > 0.0
                             ? (double)row->gen_tokens / row->control_tok_s : 0.0;
    for (int i = 0; i < 3; i++)
        row->tpf[i] = accept_tpf_at(s, k_of[i], NGRAM_CONTROL_NONE);
    row->shuffle_k4 = accept_tpf_at(s, 4, NGRAM_CONTROL_SHUFFLE);
    row->random_k4 = accept_tpf_at(s, 4, NGRAM_CONTROL_RANDOM);
    accept_cell(s, 4, 0, NGRAM_CONTROL_NONE, phase);
    ngram_accept_stats_add(&all, &phase[NGRAM_PHASE_REASONING]);
    ngram_accept_stats_add(&all, &phase[NGRAM_PHASE_ANSWER]);
    row->accept_k4 = all.drafted ? 100.0 * (double)all.accepted / (double)all.drafted : 0.0;
    row->reasoning_forwards = phase[NGRAM_PHASE_REASONING].forwards;
    row->tpf_reasoning_k4 = ngram_tokens_per_forward(&phase[NGRAM_PHASE_REASONING]);
    row->tpf_answer_k4 = ngram_tokens_per_forward(&phase[NGRAM_PHASE_ANSWER]);
}

/**
 * The verdict table.
 *
 * `tokens per forward` is a CEILING on any speculative speedup, whatever the verify ends up
 * costing: a verify that were free would still deliver exactly this many tokens per forward.
 * The two control columns beside it are the same code over the same bytes with the structure
 * removed, and they are printed on every run so nobody has to take the headline on trust.
 */
static void accept_print_summary(const accept_summary_row *rows, int n)
{
    printf("\n================================================================\n");
    printf("== SUMMARY — TOKENS PER FORWARD (a CEILING on any speculative speedup)\n");
    printf("%-10s %4s %8s %9s | %6s %6s %6s | %8s %8s %8s | %8s %8s\n", "corpus", "docs",
           "gen_tok", "ctl tok/s", "K=2", "K=4", "K=8", "acc@K4", "shuffle", "random",
           "K4 reas", "K4 answ");
    for (int i = 0; i < n; i++)
        printf("%-10s %4d %8ld %9.3f | %6.3f %6.3f %6.3f | %7.1f%% %8.3f %8.3f | %8.3f %8.3f\n",
               rows[i].tag, rows[i].docs, rows[i].gen_tokens, rows[i].control_tok_s,
               rows[i].tpf[0], rows[i].tpf[1], rows[i].tpf[2], rows[i].accept_k4,
               rows[i].shuffle_k4, rows[i].random_k4,
               rows[i].reasoning_forwards ? rows[i].tpf_reasoning_k4 : 0.0,
               rows[i].tpf_answer_k4);
    printf("shuffle/random are the SAME code path with the history permuted and with the\n"
           "proposals replaced by uniform ids. Both must sit at ~1.000 or no figure to their\n"
           "left means anything.\n");
}

#define ACCEPT_MAX_DUMPS 16

static int accept_run_score(int n_paths, char **paths)
{
    accept_summary_row rows[ACCEPT_MAX_DUMPS];
    int n_rows = 0;

    memset(rows, 0, sizeof(rows));

    for (int i = 0; i < n_paths; i++) {
        ngram_stream s;
        double gen_tokens = 0.0, gen_seconds = 0.0;

        if (ngram_stream_read(paths[i], &s) != 0)
            return 1;
        for (int d = 0; d < s.n_docs; d++) {
            gen_tokens += s.docs[d].n_gen;
            gen_seconds += s.docs[d].decode_s;
        }
        printf("\n================================================================\n");
        printf("DUMP %s  tag=%s arm=%s arch=%u n_vocab=%u docs=%d\n", paths[i], s.tag, s.arm,
               s.arch, s.n_vocab, s.n_docs);
        printf("  recorded plain greedy decode: %.0f tokens in %.1f s = %.3f tok/s "
               "(%.4f s/token) — TODAY'S CONTROL\n", gen_tokens, gen_seconds,
               gen_tokens / gen_seconds, gen_seconds / gen_tokens);
        for (int d = 0; d < s.n_docs; d++)
            printf("  doc %d: %d prompt + %d generated, think_end_pos=%d\n", d,
                   s.docs[d].n_prompt, s.docs[d].n_gen, s.docs[d].think_end_pos);
        accept_report_k_ladder(&s);
        accept_report_windows(&s, 4);
        accept_report_ngram(&s, 4);
        accept_report_per_document(&s, 4);
        accept_report_length(&s, 4);
        accept_report_controls(&s, 4);
        if (n_rows < ACCEPT_MAX_DUMPS)
            accept_fill_summary(&s, &rows[n_rows++]);
        ngram_stream_free(&s);
    }
    accept_print_summary(rows, n_rows);
    return 0;
}

/* ================================================================
 * Known-answer self-test — proves the counter reads BOTH ends
 * ================================================================ */

static int accept_selftest(void)
{
    const int k = 4, period = 64, vocab = 248320, n_gen = 1024;
    /* THE CEILING IS NOT k+1 EXACTLY. A recorded stream ends, so the last round delivers
     * whatever is left: n_gen-1 tokens over ceil((n_gen-1)/(k+1)) rounds. Asserting the
     * closed form rather than a tolerance is what makes this a known-answer arm — a
     * simulator that mis-advanced by one token would miss it. */
    const long deliverable = n_gen - 1;
    const long full_rounds = (deliverable + k) / (k + 1);
    const double ceiling = (double)deliverable / (double)full_rounds;
    ngram_document periodic, random_doc;
    ngram_accept_stats phase[NGRAM_PHASE_COUNT];
    ngram_accept_stats ctrl[NGRAM_PHASE_COUNT];
    double hi, lo, hi_shuffled;
    int failures = 0;

    ngram_selftest_periodic(&periodic, 1024, n_gen, period, (uint32_t)vocab);
    ngram_selftest_random(&random_doc, 1024, n_gen, (uint32_t)vocab, 0xC0FFEEULL);

    memset(phase, 0, sizeof(phase));
    ngram_accept_replay(&periodic, k, 3, 2, 0, NGRAM_CONTROL_NONE, (uint32_t)vocab, 1, phase);
    hi = ngram_tokens_per_forward(&phase[NGRAM_PHASE_ANSWER]);

    memset(phase, 0, sizeof(phase));
    ngram_accept_replay(&random_doc, k, 3, 2, 0, NGRAM_CONTROL_NONE, (uint32_t)vocab, 1, phase);
    lo = ngram_tokens_per_forward(&phase[NGRAM_PHASE_ANSWER]);

    memset(ctrl, 0, sizeof(ctrl));
    ngram_accept_replay(&periodic, k, 3, 2, 0, NGRAM_CONTROL_SHUFFLE, (uint32_t)vocab, 1, ctrl);
    hi_shuffled = ngram_tokens_per_forward(&ctrl[NGRAM_PHASE_ANSWER]);

    printf("SELFTEST K=%d\n", k);
    printf("  periodic  (period %d, the CEILING)      tok/fwd=%.4f  expect %.4f\n", period,
           hi, ceiling);
    printf("  i.i.d.    (vocab %d, the FLOOR)     tok/fwd=%.4f  expect 1.0000\n", vocab, lo);
    printf("  periodic + shuffled history            tok/fwd=%.4f  expect ~1.0000\n",
           hi_shuffled);

    if (hi < ceiling - 1e-4 || hi > ceiling + 1e-4) {
        printf("  FAIL: the simulator missed the closed-form ceiling %.4f\n", ceiling);
        failures++;
    }
    if (lo > 1.02) {
        printf("  FAIL: an i.i.d. stream must not accept drafts\n");
        failures++;
    }
    if (hi_shuffled > 1.05) {
        printf("  FAIL: the shuffle control did not destroy the structure\n");
        failures++;
    }
    ngram_selftest_free(&periodic);
    ngram_selftest_free(&random_doc);
    printf("SELFTEST %s (%d failure(s))\n", failures ? "FAILED" : "PASSED", failures);
    return failures ? 1 : 0;
}

/**
 * Print what a corpus file parses to, without a model or a card.
 *
 * A corpus is the one input this measurement cannot check any other way: a separator typo
 * silently merges two prompts into one, which changes the very repetition the drafter is
 * being scored on. Cheap to look at, so look at it.
 */
static int accept_corpus_check(const char *path)
{
    char *prompts[ACCEPT_MAX_PROMPTS];
    const char *system = NULL;
    char *buffer = NULL;
    int n = accept_load_corpus(path, &buffer, &system, prompts, ACCEPT_MAX_PROMPTS);

    if (n <= 0) {
        fprintf(stderr, "corpus-check: %s parsed to %d prompts\n", path, n);
        free(buffer);
        return 1;
    }
    printf("CORPUS %s: %d prompt(s), system=%s\n", path, n, system ? "present" : "none");
    if (system)
        printf("  SYSTEM (%zu B): %.100s\n", strlen(system), system);
    for (int i = 0; i < n; i++)
        printf("  prompt %d: %7zu B | first line: %.90s\n", i, strlen(prompts[i]),
               prompts[i]);
    free(buffer);
    return 0;
}

/* ================================================================
 * Entry
 * ================================================================ */

static void accept_usage(const char *argv0)
{
    fprintf(stderr,
            "usage:\n"
            "  %s --generate OUT.bin MODEL.gguf --corpus F --tag NAME [--n-gen N] "
            "[--max-docs N]\n"
            "  %s --score DUMP.bin [DUMP.bin ...]\n"
            "  %s --selftest\n"
            "  %s --corpus-check CORPUS.txt\n",
            argv0, argv0, argv0, argv0);
}

int main(int argc, char **argv)
{
    const char *dump = NULL, *model = NULL, *corpus = NULL, *tag = "untagged";
    const char *system = NULL;
    int n_gen = 512, max_docs = ACCEPT_MAX_PROMPTS;

    if (argc < 2) {
        accept_usage(argv[0]);
        return 1;
    }
    if (strcmp(argv[1], "--selftest") == 0)
        return accept_selftest();
    if (strcmp(argv[1], "--corpus-check") == 0 && argc > 2)
        return accept_corpus_check(argv[2]);
    if (strcmp(argv[1], "--score") == 0) {
        if (argc < 3) {
            accept_usage(argv[0]);
            return 1;
        }
        return accept_run_score(argc - 2, argv + 2);
    }
    if (strcmp(argv[1], "--generate") != 0 || argc < 4) {
        accept_usage(argv[0]);
        return 1;
    }
    dump = argv[2];
    model = argv[3];
    for (int i = 4; i < argc; i++) {
        if (strcmp(argv[i], "--corpus") == 0 && i + 1 < argc)
            corpus = argv[++i];
        else if (strcmp(argv[i], "--tag") == 0 && i + 1 < argc)
            tag = argv[++i];
        else if (strcmp(argv[i], "--n-gen") == 0 && i + 1 < argc)
            n_gen = atoi(argv[++i]);
        else if (strcmp(argv[i], "--max-docs") == 0 && i + 1 < argc)
            max_docs = atoi(argv[++i]);
        else if (strcmp(argv[i], "--system") == 0 && i + 1 < argc)
            system = argv[++i];
        else {
            fprintf(stderr, "unknown argument %s\n", argv[i]);
            return 1;
        }
    }
    if (!corpus) {
        accept_usage(argv[0]);
        return 1;
    }
    if (n_gen < 8)
        n_gen = 8;
    if (n_gen > ACCEPT_MAX_GEN)
        n_gen = ACCEPT_MAX_GEN;
    if (max_docs < 1 || max_docs > ACCEPT_MAX_PROMPTS)
        max_docs = ACCEPT_MAX_PROMPTS;
    return accept_run_generate(dump, model, corpus, tag, n_gen, max_docs, system);
}
