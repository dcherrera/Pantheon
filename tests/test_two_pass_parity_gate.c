/* SPDX-License-Identifier: AGPL-3.0-or-later */
/* Copyright (C) 2026 David (Corey) Herrera */
/**
 * @file test_two_pass_parity_gate.c
 * @brief The accelerator parity gate as TWO SEPARATE PASSES paired offline — same corpus,
 *        same criteria, same verdict as test_chat_parity_gate.c, without ever putting a
 *        26-second CPU forward between two consecutive card forwards.
 *
 * ---------------------------------------------------------------------------
 * WHY THE INTERLEAVED GATE HAD TO BE REPLACED
 * ---------------------------------------------------------------------------
 * MEASURED ON SILICON 2026-08-18. The interleaved gate runs the accelerator and the CPU
 * oracle alternately, position by position. On the TT host the oracle costs **~26 s a scored
 * position** — 4.5 GB of major page faults a token, because a 9.5 GB Q8_0 mmap does not fit a
 * 7.3 GB host — so every card forward is followed by 26 s of silence on the card. A resident
 * queue is BOUNDED on purpose (`idle_spin_limit`), so the triads retire while they wait. The
 * gate logged **802 / 2406 / 442 descriptor refusals** across its arms against **0** on
 * single-arm runs of the same ops at the same length, and the GDN arm died at position 1 of
 * document 0. The recovery is real and the results reproduce; a harness that provokes 2406
 * refusals is still not one to gate an arm on.
 *
 * It is also 3x the work it needs to be. The oracle pass is IDENTICAL for every arm scored
 * against it, and interleaved every arm pays for its own copy — the Phase 6 ladder was ~14 h
 * for what is one ~3 h oracle pass plus minutes of card passes.
 *
 * ---------------------------------------------------------------------------
 * THE THREE MODES
 * ---------------------------------------------------------------------------
 *   --arm FILE     run the ACCELERATOR only, over the whole corpus, dumping every scored
 *                  position's fp32 logit row. Nothing else runs in this process, so the
 *                  residencies are fed at decode cadence exactly as a serve feeds them.
 *   --oracle FILE  run the CPU ORACLE only, the same way. This is the ~3 h pass and it is
 *                  CACHEABLE: score five arms against one oracle dump.
 *   --pair ARM ORACLE
 *                  read both dumps, score every position, apply the criteria, print the
 *                  verdict. Loads NO model and touches NO card, so it costs seconds.
 *
 * ---------------------------------------------------------------------------
 * SCORING A PREFILL ARM: `--prefill L`
 * ---------------------------------------------------------------------------
 * The batched prefill produces NO per-position logits — the last prompt token goes through the
 * ordinary single-token forward and that is the only row a prefill emits. So a prefill arm
 * cannot be scored the way a decode arm is, position against position.
 *
 * What CAN be scored, and is the question a serve actually asks, is the STATE a prefill
 * leaves. `--prefill L` runs the first `L` positions of every document through
 * `pn_qwen35moe_prefill_tt()` — the shipping entry point, the same one `pn_prefill_tt_batched`
 * calls — and then decodes and scores positions `L-1 .. n-1` one at a time. Those decode steps
 * are byte-for-byte the same code in every arm; the ONLY thing that differs between arms is
 * the conv window, the recurrent state and the K/V the prefill left behind. Any divergence the
 * chunkwise mixers introduce is therefore carried into every scored position, autoregressively.
 *
 * The SEQUENTIAL prefill arm needs no separate mode: with `PN_TT_PREFILL` unset the shipping
 * entry point returns NULL WITHOUT TOUCHING THE STATE, and the same `L` positions run through
 * the per-token loop instead. Same scored set, same code, one flag. `--require-prefill` turns
 * a refusal into an abort, because "the chunkwise arm silently ran the sequential path" is the
 * one way this measurement can lie and read plausible.
 *
 * L IS CHOSEN SO THE PREFILL SPANS MORE THAN ONE CHUNK. Every red arm this family has —
 * `gdnc_window_wipe`, `gdnc_state_wipe`, attention's `prior_wipe` — is INVISIBLE IN CHUNK 0,
 * because the incoming window and state are zero there and half the transform is not
 * exercised. A single-chunk prefill would be a gate those bugs pass. At the default 32 rows,
 * `L = 97` is three full chunks (0-31, 32-63, 64-95) plus the forward that emits position 96.
 *
 * A document shorter than `L + 2` ids is SKIPPED, identically in both passes, and said so out
 * loud — the alternative is a document whose "prefill" is the whole of it.
 *
 * ---------------------------------------------------------------------------
 * WHAT KEEPS TWO PASSES HONEST
 * ---------------------------------------------------------------------------
 * Two passes equal one interleaved pass only if they scored the same positions of the same
 * token stream. That is CHECKED, not assumed:
 *
 *   - each record carries its own `(document, position, next_token)` and pairing refuses on
 *     the first mismatch, naming the record;
 *   - the headers must agree on vocabulary, architecture, framing and record count, so a
 *     `--plain` on one pass and not the other is a refusal rather than a misalignment;
 *   - a dump whose writer died before closing records ZERO positions and is refused, instead
 *     of pairing a truncated arm against a whole oracle;
 *   - `argmax` is written by the producer and re-derived by the pairer, so a row read back at
 *     the wrong offset is caught rather than scored.
 *
 * The two arms' `pn_state`s were already independent inside the interleaved gate, so nothing
 * but host load and card temperature was ever held common — and neither is an input to the
 * arithmetic.
 *
 * ---------------------------------------------------------------------------
 * PROVING IT CAN FAIL — AND THIS ONE NEEDS NO CARD AND NO MODEL
 * ---------------------------------------------------------------------------
 * `--pair D D --noise SIGMA` pairs a dump against ITSELF. At sigma 0 every figure must be
 * exactly zero-divergence and the gate additionally FAILS if it is not; at the sigmas the
 * thresholds were calibrated against it must FAIL (0.2 costs ratio 1.0091 and 9.09% disagree
 * against an 8% limit; 0.4 costs 1.0902). A gate that still says PASS at sigma 0.4 is
 * measuring nothing. The whole red ladder is three invocations of the pairing step over one
 * cached dump — seconds, against the ~3 h an interleaved red arm costs.
 *
 * USAGE
 *   ./tests/test_two_pass_parity_gate --arm    ARM.bin    <model.gguf> [--corpus F] [--plain]
 *                                              [--max-pos N] [--prefill L] [--require-prefill]
 *   ./tests/test_two_pass_parity_gate --oracle ORACLE.bin <model.gguf> [--corpus F] [--plain]
 *                                                                      [--max-pos N] [--prefill L]
 *   ./tests/test_two_pass_parity_gate --pair   ARM.bin ORACLE.bin [--noise SIGMA]
 *
 * NOT part of the default `make test`: the dump modes need a model file, and the arm mode
 * needs a card.
 *
 * @status Working
 * @issues None
 * @todo None
 */
#include "../include/pantheon.h"
#include "../src/core/engine.h"
#include "../src/models/qwen35moe.h"
#include "tensor/tensor.h"

#include "chat_corpus.h"
#include "parity_accel_arm.h"
#include "parity_gate_criteria.h"
#include "parity_logit_dump.h"
#include "parity_noise.h"
#include "parity_score.h"

#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

#define GATE_DEFAULT_CORPUS "tests/data/chat_parity_corpus.txt"

/** How much of each document to echo back, decoded, so a reader can SEE it is real text. */
#define GATE_ECHO_CHARS 72

/** The red arm's perturbation stream. One seeded stream for the whole pairing, so a repeat of
 *  the same command line draws the same noise and the arm is reproducible. */
#define GATE_NOISE_SEED 0x9E3779B97F4A7C15ULL

typedef enum { GATE_MODE_NONE, GATE_MODE_ARM, GATE_MODE_ORACLE, GATE_MODE_PAIR } gate_mode;

/** Where one document's wall time went: the forward (on CUDA this includes the logit-row
 *  download, which is synchronous inside it) against the row check and the dump write. */
typedef struct {
    int forwards;
    double forward_s;
    double row_s;
} gate_document_timing;

static double gate_monotonic_seconds(void)
{
    struct timespec ts;

    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (double)ts.tv_sec + (double)ts.tv_nsec * 1e-9;
}

/* ================================================================
 * The dump passes
 * ================================================================ */

/**
 * Tokenise one document in the requested framing and echo what it decodes back to.
 *
 * The echo is not decoration: the gate this family replaces was fed ids from another model's
 * tokeniser and nobody looked.
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
    printf("DUMP doc=%d ids=%d framing=%s text=\"", index, n_ids, chat_arm ? "chat" : "plain");
    for (int i = 0; text && text[i] && i < GATE_ECHO_CHARS; i++)
        putchar(text[i] == '\n' ? ' ' : text[i]);
    printf("...\"\n");
    free(text);
    fflush(stdout);
    return n_ids;
}

/**
 * Refuse a SPARSE logit row before a single position of it is scored.
 *
 * **THIS IS A REAL TRAP ON BOTH ACCELERATORS, NOT A HYPOTHETICAL.** On the card,
 * `PN_TT_SPARSE_LOGITS=1` is on in every speed arm this project runs: it selects the exact
 * top-8 and the host fills the other 248,312 entries with `-INFINITY`. A greedy argmax over
 * that row is exactly the argmax over the dense one — which is why it ships — but a
 * perplexity, a KL and a paired t over it are all meaningless, and `disagree_rate` would read
 * 0% while every other figure was noise. A gate that scored it would be the eighth
 * vacuous-gate class in this project.
 *
 * **THE CUDA ARM IS SPARSE BY DEFAULT, WHICH IS WORSE — MEASURED ON A K80 2026-08-24.**
 * `gpu_topk_k()` returns 40 unless told otherwise, so `pn_qwen35moe_forward_cuda` reduces the
 * head ON the die and presents 40 survivors against 248,280 `-INFINITY`. Nothing has to be
 * switched on for this: an Ornith CUDA parity pass that nobody configured is sparse. The knob
 * is `QWEN35_GPU_FULL_LOGITS=1`, and it changes only the PRESENTATION — the head matmul and
 * the top-k reduction are documented bit-exact, so the full-logits arm is the same arm, fully
 * observable. This guard is what caught it, on the first row, before 339 positions of noise.
 *
 * Checked ONCE, on the first row of the run, because the presentation is a property of the
 * configuration and not of the position.
 *
 * @return 0 when the row is dense, -1 with the reason on stderr
 */
static int gate_refuse_sparse_row(const float *row, int n)
{
    int non_finite = 0;

    for (int i = 0; i < n; i++) {
        if (!(row[i] > -1e30f && row[i] < 1e30f))
            non_finite++;
    }
    if (non_finite * 2 < n)
        return 0;
    fprintf(stderr, "dump: REFUSED — %d of %d logits in the first row are not finite. This is "
                    "a SPARSE logit row, which carries the exact top-N and -INFINITY "
                    "everywhere else. The argmax over it is right and every other figure this "
                    "gate computes would be noise. Fix it at the arm that produced it: on the "
                    "card UNSET PN_TT_SPARSE_LOGITS (opt-in); on CUDA SET "
                    "QWEN35_GPU_FULL_LOGITS=1, because there the device top-k is the DEFAULT "
                    "(k=40) and a pass nobody configured is already sparse.\n", non_finite, n);
    return -1;
}

/**
 * Put a NaN into one logit of the `n`th checked row, so the guard below can be proven to fire.
 *
 * `PN_PARITY_INJECT_NONFINITE=N` — N counted over checked rows, 0 for the first. It needs a
 * card arm of three positions, not a corpus and not an oracle, so the red ladder for this
 * guard costs one model load.
 *
 * The NaN is built from BITS. A `0.0f / 0.0f` would be folded to a constant — or to nothing —
 * under this tree's `-ffast-math`, and a red arm that cannot produce the condition it is
 * testing for is the eleventh vacuous-gate class rather than a proof.
 */
static void gate_inject_non_finite(float *row, int n, int checked_index)
{
    const char *env = getenv("PN_PARITY_INJECT_NONFINITE");
    uint32_t quiet_nan = 0x7FC00000u;

    if (!env || !*env || atoi(env) != checked_index || n < 1)
        return;
    memcpy(&row[0], &quiet_nan, sizeof(quiet_nan));
    fprintf(stderr, "PN_PARITY_INJECT_NONFINITE: logit 0 of checked row %d is now a NaN. The "
                    "dump must REFUSE.\n", checked_index);
}

/**
 * Refuse a row carrying ANY non-finite logit, before it is written.
 *
 * **DO NOT INFER THE ABSENCE OF A NaN FROM A SMALL MAXIMUM.** Every comparison against a NaN
 * is false, so a `max` never selects one and a column full of them reads as zero divergence —
 * a broken chunkwise arm scored 7.775e-15 that way and read green. This counts them instead,
 * on EVERY row rather than the first, because a NaN in this family is data-dependent (the
 * `P_last / P_t` underflow appeared six positions into a chunk and only at real decays).
 *
 * @return 0 when the row is entirely finite, -1 with the count and the position on stderr
 */
static int gate_refuse_non_finite_row(const float *row, int n, int document, int position)
{
    int non_finite = 0;

    for (int i = 0; i < n; i++)
        if (!parity_float_is_finite(row[i]))
            non_finite++;
    if (non_finite == 0)
        return 0;
    fprintf(stderr, "dump: REFUSED — document %d position %d has %d of %d logits non-finite. "
                    "A max over such a row silently drops them and scores near zero; this "
                    "gate counts them instead.\n", document, position, non_finite, n);
    return -1;
}

/**
 * Check one returned row and write it if it is scored.
 *
 * @param scored  0 to run the position without recording it (a prefill prologue, or the last
 *                position of a document, which has no label)
 * @return        0 on success, -1 on a refusal or a short write
 */
static int gate_take_row(tm_tensor *logits, parity_dump_writer *writer, int document,
                         int position, int next_token, int scored)
{
    static int presentation_checked;
    static int checked_rows;

    /* THE ROW THE FORWARD RETURNED HAS TO BE THE WIDTH THE HEADER PROMISED. The writer scans
     * and stores `n_vocab` floats taken from the header; a forward returning a shorter row
     * would make every record an overread of the one after it, and the pairing step's own
     * argmax check would then be comparing two consistent readings of the wrong bytes.
     * Checked once, because a width is a property of the model, as is the sparse presentation. */
    if (!presentation_checked) {
        presentation_checked = 1;
        if ((uint32_t)logits->shape[0] != writer->n_vocab) {
            fprintf(stderr, "dump: REFUSED — the forward returned a %d-wide row and the model "
                            "declares %u\n", (int)logits->shape[0], writer->n_vocab);
            return -1;
        }
        if (gate_refuse_sparse_row(tm_data_f32(logits), (int)logits->shape[0]) != 0)
            return -1;
    }
    gate_inject_non_finite(tm_data_f32(logits), (int)logits->shape[0], checked_rows++);
    if (gate_refuse_non_finite_row(tm_data_f32(logits), (int)logits->shape[0], document,
                                   position) != 0)
        return -1;
    if (!scored)
        return 0;
    return parity_dump_write_row(writer, document, position, next_token, tm_data_f32(logits));
}

/**
 * The PRODUCT's batched prefill, or a refusal that leaves the state untouched.
 *
 * This is the shipping entry point `pn_prefill_tt_batched()` calls, reached one level down
 * because that wrapper is static to the engine. Calling anything else would score a forward
 * the serve does not run.
 *
 * @return the logits for position `n - 1` (caller owns), or NULL for NOT TAKEN
 */
static tm_tensor *gate_batched_prefill(pn_model *model, const uint32_t *toks, int n,
                                       pn_state *state)
{
#ifdef USE_TT
    if (model->forward == pn_qwen35moe_forward_tt)
        return pn_qwen35moe_prefill_tt(model, toks, n, state);
#endif
    (void)model;
    (void)toks;
    (void)n;
    (void)state;
    return NULL;
}

/**
 * Step ONE arm through one token sequence, writing every scored position's row.
 *
 * The forward and the seam are set ONCE by the caller and never touched here — that is the
 * whole point of a single-arm pass, and switching either mid-sequence is what the interleaved
 * gate does.
 *
 * @param prefill        run positions `0 .. prefill-1` through the batched prefill and score
 *                       from `prefill-1` onward; 0 disables and every position is scored
 * @param prefill_taken  set to 1 if the batched entry point ran, 0 if it declined
 * @param timing         accumulates this document's forward count and wall split
 * @return               0 on success, -1 if a forward or a write failed
 */
static int gate_dump_sequence(pn_model *model, pn_forward_fn forward, const uint32_t *toks,
                              int ntok, int document, parity_dump_writer *writer, int prefill,
                              int *prefill_taken, gate_document_timing *timing)
{
    pn_state *state = pn_state_new(model);
    int first = 0, rc = 0;

    *prefill_taken = 0;
    if (!state) {
        fprintf(stderr, "dump: state alloc failed\n");
        return -1;
    }
    if (prefill >= 2 && prefill < ntok) {
        tm_tensor *logits = gate_batched_prefill(model, toks, prefill, state);

        /* A REFUSAL IS NOT A FAILURE — it is the sequential arm. The entry point returns NULL
         * without having touched the state, so falling through to the per-token loop from
         * position 0 scores exactly the same positions by exactly the same code. */
        if (logits) {
            *prefill_taken = 1;
            rc = gate_take_row(logits, writer, document, prefill - 1, (int)toks[prefill], 1);
            tm_free(logits);
            first = prefill;
        }
    }
    for (int i = first; rc == 0 && i < ntok; i++) {
        double t0 = gate_monotonic_seconds();
        tm_tensor *logits = forward(model, toks[i], state);
        double t1 = gate_monotonic_seconds();

        timing->forwards++;
        timing->forward_s += t1 - t0;

        if (!logits) {
            fprintf(stderr, "dump: forward failed at document %d position %d\n", document, i);
            rc = -1;
            break;
        }
        /* The last position has no label in the corpus, so it is RUN — the state must advance
         * exactly as the interleaved gate advanced it — and not scored. Neither is the prefill
         * prologue, which exists to build the state the scored positions then read. */
        rc = gate_take_row(logits, writer, document, i, i + 1 < ntok ? (int)toks[i + 1] : 0,
                           i + 1 < ntok && i >= prefill - 1);
        tm_free(logits);
        timing->row_s += gate_monotonic_seconds() - t1;
    }
    pn_state_free(state);
    return rc;
}

/**
 * Run one whole pass — arm or oracle — over the corpus and write the dump.
 *
 * @return 0 on success, 1 on any failure (the process exit code)
 */
static int gate_run_dump_pass(const char *model_path, const char *corpus_path,
                              const char *dump_path, int chat_arm, int max_pos, gate_mode mode,
                              int allow_armed_oracle, int prefill, int require_prefill)
{
    static uint32_t ids[CHAT_CORPUS_MAX_IDS];
    parity_dump_writer writer;
    const char *arm_name = "CPU", *cpu_name = "CPU";
    chat_document doc = {0};
    pn_forward_fn forward, cpu;
    pn_model *model;
    FILE *corpus;
    int docs = 0, seam, taken_docs = 0, skipped_docs = 0, rc = 0;
    double load_t0;

    corpus = fopen(corpus_path, "r");
    if (!corpus) {
        fprintf(stderr, "dump: cannot open corpus %s\n", corpus_path);
        return 1;
    }
    load_t0 = gate_monotonic_seconds();
    model = pn_load(model_path);
    if (!model) {
        fprintf(stderr, "dump: load failed\n");
        fclose(corpus);
        return 1;
    }
    cpu = parity_select_cpu_oracle(model, &cpu_name);
    if (!cpu) {
        pn_model_free(model);
        fclose(corpus);
        return 1;
    }
    if (mode == GATE_MODE_ARM) {
        forward = parity_select_accel_arm(model, &arm_name);
        if (!forward) {
            pn_model_free(model);
            fclose(corpus);
            return 1;
        }
        /* The identity control makes the arm the oracle, so the seam must stay DETACHED for
         * it too — leaving it attached would run the control on the card. */
        seam = !parity_identity_control_active();
    } else {
        forward = cpu;
        arm_name = cpu_name;
        seam = 0;
    }
    /* The return says whether a seam WAS attached, which is the only way to find out that an
     * oracle pass is running inside a card-armed process. That is not merely wasteful: it
     * leaves every resident triad armed and idle for the ~3 h the oracle takes, which is the
     * exact condition this harness exists to remove. Run the oracle pass from a build or a
     * configuration that never armed the card. */
    if (parity_accel_seam_enable(model, seam) && mode == GATE_MODE_ORACLE) {
        fprintf(stderr, "dump: REFUSED — this is the ORACLE pass and the accelerator seam was "
                        "ATTACHED, so the card is armed. Every resident triad would sit armed "
                        "and idle for the whole pass, which is the defect the two-pass design "
                        "exists to remove. Run the oracle pass unarmed, or pass "
                        "--allow-armed-oracle if you have a reason.\n");
        if (!allow_armed_oracle) {
            pn_model_free(model);
            fclose(corpus);
            return 1;
        }
    }
    model->forward = forward;

    printf("DUMP mode=%s arm=%s oracle=%s arch=%d seam=%s corpus=%s framing=%s n_vocab=%d "
           "prefill=%d out=%s load_s=%.1f\n", mode == GATE_MODE_ARM ? "arm" : "oracle", arm_name,
           cpu_name, (int)model->config.arch, seam ? "attached" : "detached", corpus_path,
           chat_arm ? "chat" : "plain", model->config.n_vocab, prefill, dump_path,
           gate_monotonic_seconds() - load_t0);
    fflush(stdout);

    if (parity_dump_open_write(dump_path, (uint32_t)model->config.n_vocab,
                               (uint32_t)model->config.arch, chat_arm, seam, arm_name,
                               &writer) != 0) {
        pn_model_free(model);
        fclose(corpus);
        return 1;
    }

    while (chat_document_read(corpus, &doc) && docs < PARITY_GATE_MAX_DOCUMENTS) {
        int n_ids = gate_tokenise_document(model, &doc, chat_arm, ids, docs);
        gate_document_timing timing = {0};
        int taken = 0;

        chat_document_free(&doc);
        if (n_ids <= 0)
            continue;
        if (max_pos > 0 && n_ids > max_pos)
            n_ids = max_pos;
        /* SKIPPED IDENTICALLY IN BOTH PASSES, because the rule reads only `n_ids` and the two
         * passes tokenise the same corpus with the same model. A document whose "prefill" is
         * the whole of it scores nothing and would silently shorten one side of the pairing. */
        if (prefill >= 2 && n_ids < prefill + 2) {
            printf("DUMP doc=%d SKIPPED — %d ids is short of prefill %d + 2\n", docs, n_ids,
                   prefill);
            fflush(stdout);
            skipped_docs++;
            continue;
        }
        if (gate_dump_sequence(model, forward, ids, n_ids, docs, &writer, prefill,
                               &taken, &timing) != 0) {
            fprintf(stderr, "dump: document %d failed — refusing to close a partial dump\n",
                    docs);
            rc = 1;
            break;
        }
        taken_docs += taken;
        docs++;
        printf("DUMP doc=%d complete, %llu records so far, prefill=%s scored_from=%d "
               "forwards=%d forward_s=%.1f (%.4f s each) row_s=%.1f\n",
               docs - 1, (unsigned long long)writer.records,
               prefill >= 2 ? (taken ? "TAKEN(batched)" : "declined(per-token)") : "off",
               prefill >= 2 ? prefill - 1 : 0, timing.forwards, timing.forward_s,
               timing.forwards ? timing.forward_s / timing.forwards : 0.0, timing.row_s);
        fflush(stdout);
    }
    fclose(corpus);

    /* THE ONE WAY THIS MEASUREMENT CAN LIE AND STILL READ PLAUSIBLE. A chunkwise arm whose
     * batched entry point quietly declined is the SEQUENTIAL arm wearing its label, and every
     * figure downstream would be a real number about the wrong thing. */
    if (rc == 0 && require_prefill && taken_docs != docs) {
        fprintf(stderr, "dump: REFUSED — --require-prefill, and the batched prefill was taken "
                        "on %d of %d documents. An arm that declined is the SEQUENTIAL path, "
                        "not the one this dump is labelled with.\n", taken_docs, docs);
        rc = 1;
    }

    /* A failed pass deliberately does NOT patch the record count, so the file reads as empty
     * and the pairing step refuses it rather than scoring a truncated arm. */
    if (rc == 0 && parity_dump_close_write(&writer) != 0)
        rc = 1;
    if (rc == 0)
        printf("DUMP COMPLETE documents=%d skipped=%d prefill_taken=%d records=%llu file=%s\n",
               docs, skipped_docs, taken_docs, (unsigned long long)writer.records, dump_path);

    /* NOT _Exit: the TT backend's close retires the on-card residencies by ASKING, and
     * skipping it leaves four cores spinning on an armed queue the next probe refuses. */
    pn_model_free(model);
    return rc;
}

/* ================================================================
 * The pairing pass
 * ================================================================ */

/** The perturbation applied to the ARM's rows at pairing time, if any. */
static double           g_noise_sigma;
static parity_noise_rng g_noise_rng;

/**
 * Score one arm dump against one oracle dump and print the verdict.
 *
 * @param identity  the two paths named the same file — every figure must be exactly zero
 * @return          0 pass, 1 fail or refusal
 */
static int gate_run_pair_pass(const char *arm_path, const char *oracle_path, int identity)
{
    static parity_score per_doc[PARITY_GATE_MAX_DOCUMENTS];
    parity_dump_reader arm = {0}, oracle = {0};
    parity_score total = {0};
    parity_summary summary;
    float *arm_row = NULL, *oracle_row = NULL, *noise_row = NULL;
    uint32_t n_vocab;
    int docs = 0, rc = 1;
    uint64_t index;

    if (parity_dump_open_read(arm_path, &arm) != 0)
        return 1;
    if (parity_dump_open_read(oracle_path, &oracle) != 0) {
        parity_dump_close_read(&arm);
        return 1;
    }
    if (parity_dump_headers_agree(&arm.header, &oracle.header) != 0)
        goto cleanup;

    n_vocab = arm.header.n_vocab;
    arm_row = malloc((size_t)n_vocab * sizeof(*arm_row));
    oracle_row = malloc((size_t)n_vocab * sizeof(*oracle_row));
    noise_row = g_noise_sigma > 0.0 ? malloc((size_t)n_vocab * sizeof(*noise_row)) : NULL;
    if (!arm_row || !oracle_row || (g_noise_sigma > 0.0 && !noise_row)) {
        fprintf(stderr, "pair: two %u-wide rows did not allocate\n", n_vocab);
        goto cleanup;
    }

    printf("PAIR arm=%s(%s) oracle=%s(%s) arch=%u framing=%s records=%llu noise=%.4g%s\n",
           arm_path, arm.header.label, oracle_path, oracle.header.label, arm.header.arch,
           arm.header.framing_is_chat ? "chat" : "plain",
           (unsigned long long)arm.header.record_count, g_noise_sigma,
           identity ? " IDENTITY-CONTROL(same file both sides)" : "");
    if (arm.header.seam_enabled == 0u && !identity)
        printf("PAIR WARNING arm dump records seam=detached — that arm was NOT the "
               "accelerator\n");
    fflush(stdout);

    for (index = 0; index < arm.header.record_count; index++) {
        parity_dump_record ra, ro;
        const float *scored;
        int a1 = 0, a2 = 0, c1 = 0, c2 = 0;

        if (parity_dump_read_row(&arm, &ra, arm_row) != 1 ||
            parity_dump_read_row(&oracle, &ro, oracle_row) != 1) {
            fprintf(stderr, "pair: a dump ended early at record %llu\n",
                    (unsigned long long)index);
            goto cleanup;
        }
        /* THE ALIGNMENT CHECK IS THE WHOLE ARGUMENT FOR PAIRING OFFLINE. Two passes are one
         * interleaved pass only if they scored the same positions of the same stream. */
        if (ra.document != ro.document || ra.position != ro.position ||
            ra.next_token != ro.next_token) {
            fprintf(stderr, "pair: REFUSED — record %llu is arm(doc %d pos %d next %d) against "
                            "oracle(doc %d pos %d next %d). The two passes did not score the "
                            "same stream.\n", (unsigned long long)index, ra.document,
                    ra.position, ra.next_token, ro.document, ro.position, ro.next_token);
            goto cleanup;
        }
        if (ra.document < 0 || ra.document >= PARITY_GATE_MAX_DOCUMENTS) {
            fprintf(stderr, "pair: record %llu names document %d\n",
                    (unsigned long long)index, ra.document);
            goto cleanup;
        }
        if (ra.document + 1 > docs)
            docs = ra.document + 1;

        scored = arm_row;
        if (noise_row) {
            parity_noise_add(noise_row, arm_row, (int)n_vocab, g_noise_sigma, &g_noise_rng);
            scored = noise_row;
        }
        parity_top2_indices(scored, (int)n_vocab, &a1, &a2);
        parity_top2_indices(oracle_row, (int)n_vocab, &c1, &c2);
        /* THE PRODUCER'S OWN ARGMAX, RE-DERIVED. A row read back at the wrong offset scores a
         * real-looking number; this catches it. It is skipped under --noise, where the
         * perturbed argmax is expected to move. */
        if (!noise_row && a1 != ra.argmax) {
            fprintf(stderr, "pair: REFUSED — record %llu re-derives argmax %d and the arm dump "
                            "recorded %d. The row was read at the wrong offset.\n",
                    (unsigned long long)index, a1, ra.argmax);
            goto cleanup;
        }
        if (c1 != ro.argmax) {
            fprintf(stderr, "pair: REFUSED — record %llu re-derives the oracle's argmax as %d "
                            "and the dump recorded %d.\n", (unsigned long long)index, c1,
                    ro.argmax);
            goto cleanup;
        }
        parity_score_add(&per_doc[ra.document], scored, oracle_row, (int)n_vocab,
                         ra.next_token, a1, c1, NULL);
    }

    for (int i = 0; i < docs; i++) {
        char label[64];

        snprintf(label, sizeof(label), "doc=%d", i);
        parity_score_report(&per_doc[i], label);
        parity_score_merge(&total, &per_doc[i]);
    }
    parity_score_summarise(&total, &summary);
    printf("GATE documents=%d scored_positions=%ld\n", docs, summary.n);
    parity_score_report(&total, "GATE TOTAL");
    rc = parity_gate_verdict(&summary, per_doc, docs, identity, g_noise_sigma,
                             arm.header.framing_is_chat ? "chat" : "plain");

cleanup:
    free(arm_row);
    free(oracle_row);
    free(noise_row);
    parity_dump_close_read(&arm);
    parity_dump_close_read(&oracle);
    return rc;
}

/* ================================================================
 * Command line
 * ================================================================ */

static void gate_usage(const char *argv0)
{
    fprintf(stderr,
            "usage:\n"
            "  %s --arm    ARM.bin    <model.gguf> [--corpus F] [--plain] [--max-pos N]\n"
            "                         [--prefill L] [--require-prefill]\n"
            "  %s --oracle ORACLE.bin <model.gguf> [--corpus F] [--plain] [--max-pos N]\n"
            "                         [--prefill L] [--allow-armed-oracle]\n"
            "  %s --pair   ARM.bin ORACLE.bin [--noise SIGMA]\n"
            "\n"
            "The red ladder needs no card and no model:\n"
            "  %s --pair ORACLE.bin ORACLE.bin              must be EXACTLY zero divergence\n"
            "  %s --pair ORACLE.bin ORACLE.bin --noise 0.2  must FAIL (disagree 9.09%%)\n"
            "  %s --pair ORACLE.bin ORACLE.bin --noise 0.4  must FAIL (ratio 1.0902)\n",
            argv0, argv0, argv0, argv0, argv0, argv0);
}

int main(int argc, char **argv)
{
    const char *corpus_path = GATE_DEFAULT_CORPUS;
    const char *dump_path = NULL, *oracle_path = NULL, *model_path = NULL;
    gate_mode mode = GATE_MODE_NONE;
    int chat_arm = 1, max_pos = 0, allow_armed_oracle = 0, prefill = 0, require_prefill = 0;

    parity_noise_seed(&g_noise_rng, GATE_NOISE_SEED);
    for (int i = 1; i < argc; i++) {
        if (strcmp(argv[i], "--arm") == 0 && i + 1 < argc) {
            mode = GATE_MODE_ARM;
            dump_path = argv[++i];
        } else if (strcmp(argv[i], "--oracle") == 0 && i + 1 < argc) {
            mode = GATE_MODE_ORACLE;
            dump_path = argv[++i];
        } else if (strcmp(argv[i], "--pair") == 0 && i + 2 < argc) {
            mode = GATE_MODE_PAIR;
            dump_path = argv[++i];
            oracle_path = argv[++i];
        } else if (strcmp(argv[i], "--corpus") == 0 && i + 1 < argc) {
            corpus_path = argv[++i];
        } else if (strcmp(argv[i], "--max-pos") == 0 && i + 1 < argc) {
            max_pos = atoi(argv[++i]);
        } else if (strcmp(argv[i], "--prefill") == 0 && i + 1 < argc) {
            prefill = atoi(argv[++i]);
        } else if (strcmp(argv[i], "--require-prefill") == 0) {
            require_prefill = 1;
        } else if (strcmp(argv[i], "--noise") == 0 && i + 1 < argc) {
            g_noise_sigma = atof(argv[++i]);
        } else if (strcmp(argv[i], "--plain") == 0) {
            chat_arm = 0;
        } else if (strcmp(argv[i], "--allow-armed-oracle") == 0) {
            allow_armed_oracle = 1;
        } else if (!model_path && argv[i][0] != '-') {
            model_path = argv[i];
        } else {
            fprintf(stderr, "gate: unknown argument %s\n", argv[i]);
            gate_usage(argv[0]);
            return 1;
        }
    }

    if (mode == GATE_MODE_PAIR) {
        if (g_noise_sigma < 0.0) {
            fprintf(stderr, "gate: --noise must not be negative\n");
            return 1;
        }
        /* Same file on both sides IS the identity control, and it needs no environment
         * variable to say so — which is one fewer way for a card run to be mistaken for a
         * control. */
        return gate_run_pair_pass(dump_path, oracle_path,
                                  strcmp(dump_path, oracle_path) == 0);
    }
    if (mode == GATE_MODE_NONE || !model_path) {
        gate_usage(argv[0]);
        return 1;
    }
    if (prefill == 1) {
        fprintf(stderr, "gate: --prefill 1 is not a batched pass and not a control either; use "
                        "0 to disable or a length that spans more than one chunk\n");
        return 1;
    }
    if (require_prefill && mode != GATE_MODE_ARM) {
        fprintf(stderr, "gate: --require-prefill is an ARM check; the oracle has no batched "
                        "path to require\n");
        return 1;
    }
    return gate_run_dump_pass(model_path, corpus_path, dump_path, chat_arm, max_pos, mode,
                              allow_armed_oracle, prefill, require_prefill);
}
