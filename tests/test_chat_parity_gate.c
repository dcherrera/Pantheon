/* SPDX-License-Identifier: AGPL-3.0-or-later */
/* Copyright (C) 2026 David (Corey) Herrera */
/**
 * @file test_chat_parity_gate.c
 * @brief The accelerator parity GATE for chat-templated prompts: real vocab, real template,
 *        every position scored, pass/fail on what the divergence COSTS rather than on one
 *        next-token argmax.
 *
 * ---------------------------------------------------------------------------
 * WHY THE OLD GATE COULD NOT FAIL
 * ---------------------------------------------------------------------------
 * The 20-probe gate read 0/20 mismatches and was believed. It was unrepresentative in KIND,
 * three separate ways, and each one alone would have blinded it:
 *
 *   1. WRONG VOCAB. Decoded with the model's own tokeniser, probe 1 was `</s>ende+}}$--23et`
 *      — Llama-2/SentencePiece ids fed to MiniCPM5's Qwen BPE. Token soup. This gate
 *      tokenises the corpus ITSELF, through the product's own chat template and BPE
 *      (tests/chat_corpus.h), so wrong-vocab ids are impossible by construction.
 *   2. WRONG FRAMING. Every probe was raw ids with no chat wrapper. Chat-templated prompts
 *      cost +22.1% perplexity on this card where plain prose costs +0.4%, and the crossed
 *      control holds content constant and flips only the wrapper (0.994 vs 1.221). A gate
 *      that never sees a template never sees the defect.
 *   3. WRONG QUESTION. It asked whether ONE next-token argmax agreed. An argmax agreement is
 *      not parity and a disagreement is not damage: what a user gets is the whole
 *      distribution over every position of a long prompt. This gate scores EVERY position
 *      against the token that actually follows it.
 *
 * Two more ways to be unrepresentative were found on 2026-08-20 and are fixed here.
 *
 *   4. WRONG ARCHITECTURE. It hardcoded `pn_llama_forward` as the oracle and recognised only
 *      the llama accelerator forwards, so on Ornith it refused with "no accelerator" — which
 *      is why the sole Ornith correctness claim was `cmp` on one 48-token greedy prompt.
 *      Both arms now come from parity_accel_arm.h: the oracle from the engine's own
 *      architecture registry, the arm from whatever the engine actually selected. Ornith
 *      turns out to load as PN_ARCH_BONSAI, whose CPU forward is `pn_bonsai_forward` and
 *      whose TT forward is `pn_qwen35moe_forward_tt` — a pair no per-family table would have
 *      written down, which is why the rule is an identity instead.
 *   5. THE ACCELERATOR IS NOT ALWAYS A FORWARD. On this family it is a matvec HOOK on the
 *      shared config, so calling the architecture's CPU forward in a card-armed process
 *      still ran all 249 projections on the card. Both arms would have been the card; against
 *      the RESIDENT-ACTIVATION arm it SEGFAULTED instead, which is the only reason it was
 *      caught. Every oracle call is now bracketed by `parity_accel_seam_enable`, and the
 *      `seam=` field of the header line says which way it was set.
 *
 * ---------------------------------------------------------------------------
 * THE CRITERIA, AND WHY THESE NUMBERS
 * ---------------------------------------------------------------------------
 * Three thresholds, ANY of which fails the gate. Every figure below is measured on this card
 * with MiniCPM5-1B, not chosen for roundness.
 *
 * The position counts below are from the corpus generation these thresholds were calibrated on.
 * The COMMITTED corpus is smaller — the identity control now runs 1,056 positions over four
 * documents, re-verified exactly 1.000000 on 2026-08-15. The thresholds are unchanged: they are
 * ratios, and the identity floor is an exact zero at either size.
 *
 *   ratio <= 1.05    Teacher-forced perplexity ratio, accelerator over CPU oracle.
 *                    FLOOR:  the identity control is exactly 1.000000 over 1,830 positions;
 *                            the card on plain prose is 1.0036 (n=1327) and 0.9937 (n=391).
 *                    LADDER: iid logit noise on the same positions costs 1.0091 at sigma=0.2
 *                            and 1.0902 at sigma=0.4, so 1.05 lies between "smaller than the
 *                            card's own rounding is worth caring about" and "visible damage".
 *                    DEFECT: the chat framing is 1.2207 — this criterion catches it 4.4x over.
 *                    So 1.05 is ~14x above anything the card does when it is behaving and
 *                    ~4x below the defect. It is the magnitude criterion.
 *
 *   t <= 4.0         Paired t on the per-position ΔNLL, ONE-SIDED: only being worse counts.
 *                    The paired difference cancels content variance, so this detects a
 *                    systematic loss far below the absolute spread — which is exactly the
 *                    regime a future defect will first appear in. The card on plain prose
 *                    reads t=+1.69 and t=-1.47; the defect reads t=+6.76. It is the
 *                    SENSITIVITY criterion and it deliberately grows with corpus size.
 *
 *   disagree <= 8%   Fraction of positions whose argmaxes differ. Plain prose is 3.5% and
 *                    2.6%; the defect is 16.9%. This is the criterion the old gate was
 *                    reaching for, kept because it is the one a human reads fastest — but
 *                    it is the LOOSEST of the three and must never be the only one.
 *
 *   n >= 200         A gate that scored twelve positions would pass anything. Below this the
 *                    gate FAILS rather than passes, because "not enough evidence" is not a
 *                    pass. It is what makes the other three mean something.
 *
 *   worst document   The SAME 1.05 applied to each document of at least 100 scored positions,
 *   ratio <= 1.05    not only to the corpus average. A serve does not average over a corpus —
 *                    a user sends one prompt — so one damaged prompt shape is a defect even
 *                    when the mean is clean. It also removes the gate's dependence on corpus
 *                    composition: adding undamaged documents dilutes a mean and cannot dilute
 *                    a maximum. Measured on this card, the system-framed defect document
 *                    reads 1.23 on its own while the four-document mean reads 1.05.
 *
 * ---------------------------------------------------------------------------
 * PROVING IT CAN FAIL
 * ---------------------------------------------------------------------------
 *   PN_PARITY_IDENTITY=1  makes the accelerator arm the CPU oracle. Every figure must come
 *                         out EXACTLY zero and the gate additionally FAILS if any of them is
 *                         not — a harness whose identity control is not exactly zero is
 *                         measuring itself, and no card number from it can be believed.
 *   --plain               scores the same documents with the chat wrapper removed. Same
 *                         content, same card, same corpus: the crossed control.
 *   --noise SIGMA         adds iid N(0, SIGMA) to the ACCELERATOR arm's logit row before it
 *                         is scored, from a seeded stream (tests/parity_noise.h). WITH
 *                         PN_PARITY_IDENTITY=1 this is the gate's own red arm and needs no
 *                         card at all: both arms are the same CPU oracle, so every reported
 *                         divergence is the injected noise and nothing else. A gate that
 *                         still says PASS at a sigma the thresholds were calibrated against
 *                         is measuring nothing, which is the only way to find that out
 *                         before trusting a green card run.
 *                         The ladder those thresholds were set from: sigma 0.2 costs 1.0091
 *                         and sigma 0.4 costs 1.0902 against a 1.05 limit — so 0.2 must PASS
 *                         and 0.4 must FAIL, and a run where both agree is a broken harness.
 *                         The exact-zero identity assertion is skipped when sigma > 0, since
 *                         a non-zero result is the point.
 *
 * USAGE
 *   ./tests/test_chat_parity_gate <model.gguf> [--corpus FILE] [--plain] [--max-pos N]
 *                                              [--noise SIGMA]
 *
 * The corpus defaults to tests/data/chat_parity_corpus.txt. Build with the accelerator flag
 * the card needs (USE_TT=1 / USE_CUDA=1); without one the gate REFUSES to run rather than
 * scoring the CPU against itself and printing PASS.
 *
 * NOT part of the default `make test`: it needs a model file and a card.
 *
 * @status Working
 * @issues None
 * @todo None
 */
#include "../include/pantheon.h"
#include "../src/core/engine.h"
#include "tensor/tensor.h"

#include "chat_corpus.h"
#include "parity_accel_arm.h"
#include "parity_gate_criteria.h"
#include "parity_noise.h"
#include "parity_score.h"

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define GATE_DEFAULT_CORPUS "tests/data/chat_parity_corpus.txt"

/** The thresholds and the verdict live in parity_gate_criteria.h, so this gate and the
 *  two-pass one cannot drift apart about what a passing arm is. Their derivation stays in
 *  this file's header, which is where the calibration ladder was run. */

/** How much of each document to echo back, decoded, so a reader can SEE it is real text. */
#define GATE_ECHO_CHARS 72

/** The red arm's perturbation, if any. One seeded stream for the whole run, so a repeat of
 *  the same command line draws the same noise and the arm is reproducible rather than
 *  merely random. `gate_noise_row` is the perturbed copy; the row the forward returned is
 *  const and is left alone. */
#define GATE_NOISE_SEED 0x9E3779B97F4A7C15ULL

static double           g_noise_sigma;
static parity_noise_rng g_noise_rng;
static float           *g_noise_row;
static int              g_noise_row_len;

/**
 * The accelerator row as the gate should score it: the forward's own row, or a perturbed
 * copy of it when --noise is active.
 *
 * @return the row to score, or NULL when the perturbed copy could not be allocated
 */
static const float *gate_perturbed_accel_row(const float *row, int n)
{
    if (g_noise_sigma <= 0.0)
        return row;
    if (g_noise_row_len < n) {
        free(g_noise_row);
        g_noise_row = malloc((size_t)n * sizeof(*g_noise_row));
        g_noise_row_len = g_noise_row ? n : 0;
    }
    if (!g_noise_row)
        return NULL;
    parity_noise_add(g_noise_row, row, n, g_noise_sigma, &g_noise_rng);
    return g_noise_row;
}

/**
 * Step both arms through ONE token sequence in lockstep, scoring every position against the
 * token that actually follows it. Both states stay open for the whole sequence and the arms
 * are interleaved token by token; the states are independent, so this computes exactly what
 * two separate passes would while holding host load and card temperature common to both.
 *
 * @return 0 on success, -1 if either forward failed
 */
static int gate_score_sequence(pn_model *model, pn_forward_fn accel, pn_forward_fn cpu,
                               int accel_seam, const uint32_t *toks, int ntok,
                               parity_score *score)
{
    pn_forward_fn saved = model->forward;
    pn_state *state_accel = pn_state_new(model);
    pn_state *state_cpu = pn_state_new(model);
    int rc = 0;

    if (!state_accel || !state_cpu) {
        fprintf(stderr, "gate: state alloc failed\n");
        rc = -1;
        goto cleanup;
    }
    for (int i = 0; i < ntok; i++) {
        tm_tensor *la = NULL, *lc = NULL;
        const float *accel_row, *cpu_row;
        int n, a1 = 0, a2 = 0, c1 = 0, c2 = 0;

        /* `model->forward` is set around each call because a forward may consult it to
         * decide its own path — the accelerator forwards do, and a stale value there is how
         * an arm silently runs the other one. The SEAM is switched for the same reason and
         * it is the stronger of the two: on the qwen35/bonsai family the accelerator is a
         * matvec hook on shared config rather than a separate forward, so without this the
         * "CPU oracle" runs every projection on the card. */
        model->forward = accel;
        parity_accel_seam_enable(model, accel_seam);
        la = accel(model, toks[i], state_accel);
        model->forward = cpu;
        parity_accel_seam_enable(model, 0);
        lc = cpu(model, toks[i], state_cpu);
        if (!la || !lc) {
            fprintf(stderr, "gate: forward failed at position %d\n", i);
            if (la) tm_free(la);
            if (lc) tm_free(lc);
            rc = -1;
            goto cleanup;
        }
        n = (int)la->shape[0];
        accel_row = gate_perturbed_accel_row(tm_data_f32(la), n);
        cpu_row = tm_data_f32(lc);
        if (!accel_row) {
            fprintf(stderr, "gate: the --noise row did not allocate at position %d\n", i);
            tm_free(la);
            tm_free(lc);
            rc = -1;
            goto cleanup;
        }
        /* The last position has no label in the corpus, so it is run but not scored. */
        if (i + 1 < ntok) {
            parity_top2_indices(accel_row, n, &a1, &a2);
            parity_top2_indices(cpu_row, n, &c1, &c2);
            parity_score_add(score, accel_row, cpu_row, n, (int)toks[i + 1], a1, c1, NULL);
        }
        tm_free(la);
        tm_free(lc);
    }

cleanup:
    model->forward = saved;
    parity_accel_seam_enable(model, 1);
    if (state_accel) pn_state_free(state_accel);
    if (state_cpu) pn_state_free(state_cpu);
    return rc;
}

/**
 * Tokenise one document in the requested framing and echo what it decodes back to.
 *
 * The echo is not decoration: the gate this one replaces was fed ids from another model's
 * tokeniser and nobody looked. Printing the decode makes that failure visible in the gate's
 * own output rather than in a later investigation.
 *
 * @return number of ids, or -1
 */
static int gate_tokenise_document(pn_model *model, const chat_document *doc, int chat_arm,
                                  uint32_t *ids, int index)
{
    int n_ids = chat_arm ? chat_document_to_chat_ids(model, doc, ids, CHAT_CORPUS_MAX_IDS)
                         : chat_document_to_plain_ids(model, doc, ids, CHAT_CORPUS_MAX_IDS);
    char *text;

    if (n_ids <= 0)
        return -1;
    text = chat_ids_to_text(model, ids, n_ids);
    printf("GATE doc=%d ids=%d framing=%s text=\"", index, n_ids, chat_arm ? "chat" : "plain");
    for (int i = 0; text && text[i] && i < GATE_ECHO_CHARS; i++)
        putchar(text[i] == '\n' ? ' ' : text[i]);
    printf("...\"\n");
    free(text);
    fflush(stdout);
    return n_ids;
}

int main(int argc, char **argv)
{
    static uint32_t ids[CHAT_CORPUS_MAX_IDS];
    static parity_score per_doc[PARITY_GATE_MAX_DOCUMENTS];
    const char *corpus_path = GATE_DEFAULT_CORPUS;
    const char *accel_name = "CPU", *cpu_name = "CPU";
    chat_document doc = {0};
    parity_score total = {0};
    parity_summary summary;
    pn_forward_fn accel, cpu;
    pn_model *model;
    FILE *corpus;
    int chat_arm = 1, max_pos = 0, docs = 0, accel_seam = 1, rc;

    if (argc < 2) {
        fprintf(stderr, "usage: %s <model.gguf> [--corpus FILE] [--plain] [--max-pos N]"
                        " [--noise SIGMA]\n", argv[0]);
        return 1;
    }
    parity_noise_seed(&g_noise_rng, GATE_NOISE_SEED);
    for (int i = 2; i < argc; i++) {
        if (strcmp(argv[i], "--plain") == 0)
            chat_arm = 0;
        else if (strcmp(argv[i], "--corpus") == 0 && i + 1 < argc)
            corpus_path = argv[++i];
        else if (strcmp(argv[i], "--max-pos") == 0 && i + 1 < argc)
            max_pos = atoi(argv[++i]);
        else if (strcmp(argv[i], "--noise") == 0 && i + 1 < argc)
            g_noise_sigma = atof(argv[++i]);
        else {
            fprintf(stderr, "gate: unknown argument %s\n", argv[i]);
            return 1;
        }
    }

    corpus = fopen(corpus_path, "r");
    if (!corpus) {
        fprintf(stderr, "gate: cannot open corpus %s\n", corpus_path);
        return 1;
    }
    model = pn_load(argv[1]);
    if (!model) {
        fprintf(stderr, "gate: load failed\n");
        fclose(corpus);
        return 1;
    }
    cpu = parity_select_cpu_oracle(model, &cpu_name);
    accel = cpu ? parity_select_accel_arm(model, &accel_name) : NULL;
    if (!accel) {
        pn_model_free(model);
        fclose(corpus);
        return 1;
    }
    /* Under the identity control BOTH arms are the CPU oracle, so the seam must stay
     * detached for both — leaving it attached for the "accelerator" call would run the
     * control on the card and the run would measure neither thing. */
    accel_seam = !parity_identity_control_active();
    printf("GATE arm=%s oracle=%s arch=%d seam=%s corpus=%s framing=%s noise=%.4g "
           "n_vocab=%d\n", accel_name, cpu_name, (int)model->config.arch,
           accel_seam ? "on-for-arm" : "detached-for-both", corpus_path,
           chat_arm ? "chat" : "plain", g_noise_sigma, model->config.n_vocab);
    fflush(stdout);

    while (chat_document_read(corpus, &doc) && docs < PARITY_GATE_MAX_DOCUMENTS) {
        int n_ids = gate_tokenise_document(model, &doc, chat_arm, ids, docs);
        char label[64];

        chat_document_free(&doc);
        if (n_ids <= 0)
            continue;
        if (max_pos > 0 && n_ids > max_pos)
            n_ids = max_pos;
        if (gate_score_sequence(model, accel, cpu, accel_seam, ids, n_ids,
                                &per_doc[docs]) != 0) {
            fprintf(stderr, "gate: sequence %d failed — refusing to report a verdict\n", docs);
            pn_model_free(model);
            fclose(corpus);
            return 1;
        }
        snprintf(label, sizeof(label), "doc=%d", docs);
        parity_score_report(&per_doc[docs], label);
        parity_score_merge(&total, &per_doc[docs]);
        docs++;
        parity_score_report(&total, "running");
    }
    fclose(corpus);

    parity_score_summarise(&total, &summary);
    printf("GATE documents=%d scored_positions=%ld\n", docs, summary.n);
    parity_score_report(&total, "GATE TOTAL");
    rc = parity_gate_verdict(&summary, per_doc, docs, parity_identity_control_active(),
                             g_noise_sigma, chat_arm ? "chat" : "plain");

    /* NOT _Exit: the TT backend's close retires the on-card residencies by ASKING, and
     * skipping it leaves four cores spinning on an armed queue the next probe refuses. */
    pn_model_free(model);
    return rc;
}
