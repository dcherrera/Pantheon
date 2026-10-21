/* SPDX-License-Identifier: AGPL-3.0-or-later */
/* Copyright (C) 2026 David (Corey) Herrera */
/**
 * @file ngram_accept_stream.h
 * @brief The two-pass prompt-lookup ACCEPTANCE measurement: a recorded greedy token stream
 *        (pass 1, needs the card) and an offline simulator of what a speculative round would
 *        have delivered over it (pass 2, needs nothing).
 *
 * ---------------------------------------------------------------------------
 * WHY A RECORDED GREEDY STREAM MEASURES ACCEPTANCE EXACTLY
 * ---------------------------------------------------------------------------
 * Greedy speculative verification accepts a drafted token iff it equals the model's own
 * argmax at that position. If draft token 1 is accepted, the prefix the model sees for
 * draft token 2 is byte-for-byte the prefix it saw when it greedily produced that position
 * in a plain run. So a round that drafts `k` tokens delivers
 *
 *     1 + (length of the longest leading run of drafts matching the true continuation)
 *
 * tokens, capped at k+1 — and every quantity in that sentence is a function of the PLAIN
 * GREEDY STREAM alone. No verify kernel, no batched forward, no rollback. That is what
 * makes this gate cheap: pass 1 is an ordinary decode.
 *
 * The equivalence holds only for temperature 0. Under sampling the accept test is a
 * rejection-sampling step, so this file's figures are the greedy upper bound and say so.
 *
 * ---------------------------------------------------------------------------
 * ROUNDS ARE NOT INDEPENDENT, AND THE SIMULATOR REPLAYS THEM
 * ---------------------------------------------------------------------------
 * Accepting m tokens SKIPS the positions the drafter would otherwise have been asked about,
 * so a per-position hit rate is not the number that converts to speed. The simulator walks
 * the stream the way a real decode would: one draft query per round, advance by what the
 * round delivered. `tokens per forward` is therefore the honest multiplier.
 *
 * ---------------------------------------------------------------------------
 * PROVING THE INSTRUMENT CAN READ LOW
 * ---------------------------------------------------------------------------
 * A hit counter that only ever counts hits measures nothing. Two controls run the SAME code
 * over the SAME stream and must collapse to ~1.00 tokens/forward:
 *   NGRAM_CONTROL_SHUFFLE  the history is permuted before drafting — identical token
 *                          multiset, no sequential structure left to match
 *   NGRAM_CONTROL_RANDOM   the drafter's proposals are replaced by uniform vocabulary ids
 * and `ngram_selftest_*` builds two synthetic streams with known answers — a periodic one
 * that must reach the ceiling k+1, and an i.i.d. one that must sit at 1.00.
 *
 * @status Working
 * @issues None
 * @todo None
 */
#ifndef PN_TESTS_NGRAM_ACCEPT_STREAM_H
#define PN_TESTS_NGRAM_ACCEPT_STREAM_H

#include <stdint.h>
#include <stdio.h>

#define NGRAM_STREAM_MAGIC     "PNNGRAM1"
#define NGRAM_STREAM_ENDIAN    0x01020304u
#define NGRAM_STREAM_END       0xE9D5A17Bu
#define NGRAM_STREAM_NONE      0xFFFFFFFFu
#define NGRAM_STREAM_MAX_DOCS  64
#define NGRAM_STREAM_NAME_LEN  32

/** How the drafter's inputs are perturbed, so the counter can be shown to read low. */
typedef enum {
    NGRAM_CONTROL_NONE = 0,   /**< the real measurement */
    NGRAM_CONTROL_SHUFFLE,    /**< permute the history; same tokens, no structure */
    NGRAM_CONTROL_RANDOM      /**< propose uniform random vocabulary ids */
} ngram_control;

/** One recorded document: the prompt that was prefilled and the greedy continuation. */
typedef struct {
    uint32_t *prompt;
    uint32_t *gen;
    int       n_prompt;
    int       n_gen;
    int       think_end_pos;  /**< index into gen[] of the reasoning terminator, or -1 */
    double    decode_s;       /**< wall seconds spent decoding gen[], for the tok/s control */
} ngram_document;

/** A whole dump: one corpus, one arm, one model. */
typedef struct {
    ngram_document docs[NGRAM_STREAM_MAX_DOCS];
    int            n_docs;
    uint32_t       n_vocab;
    uint32_t       arch;
    uint32_t       eos_id;
    uint32_t       think_end_id;
    char           arm[NGRAM_STREAM_NAME_LEN];
    char           tag[NGRAM_STREAM_NAME_LEN];
} ngram_stream;

/** Context-length buckets a round is attributed to, by history length at round start. */
#define NGRAM_LEN_BUCKETS 6
extern const char *const ngram_length_bucket_name[NGRAM_LEN_BUCKETS];

/** Everything one (stream, K, window, control) cell produces. */
typedef struct {
    long rounds;              /**< rounds that drafted at least one token */
    long plain_rounds;        /**< rounds with no n-gram hit -> a plain single-token forward */
    long drafted;             /**< draft tokens proposed */
    long accepted;            /**< draft tokens accepted */
    long forwards;            /**< verify passes + plain forwards */
    long emitted;             /**< tokens delivered overall */
    long hist[34];            /**< delivered-per-drafting-round, index 1..k+1 */
    long bucket_emitted[NGRAM_LEN_BUCKETS];
    long bucket_forwards[NGRAM_LEN_BUCKETS];
} ngram_accept_stats;

/** Which phase of a reasoning model's output a round belongs to. */
typedef enum { NGRAM_PHASE_REASONING = 0, NGRAM_PHASE_ANSWER = 1, NGRAM_PHASE_COUNT } ngram_phase;

/* ================================================================
 * The dump file
 * ================================================================ */

/** Open a dump for writing and emit its header. @return 0, or -1 with a reason on stderr. */
int ngram_stream_open_write(const char *path, uint32_t n_vocab, uint32_t arch, uint32_t eos_id,
                            uint32_t think_end_id, const char *arm, const char *tag,
                            FILE **out);

/** Append one document. @return 0, or -1. */
int ngram_stream_write_document(FILE *f, const ngram_document *doc);

/** Write the end marker and close. A dump without it is refused by the reader. */
int ngram_stream_close_write(FILE *f);

/** Read a whole dump. @return 0, or -1 with a reason on stderr. Caller frees with _free. */
int ngram_stream_read(const char *path, ngram_stream *out);

/** Release everything ngram_stream_read allocated. */
void ngram_stream_free(ngram_stream *s);

/* ================================================================
 * The simulator
 * ================================================================ */

/**
 * Replay one document as a sequence of speculative rounds and accumulate the outcome.
 *
 * @param doc        the recorded stream
 * @param max_k      draft budget per round (a round verifies max_k+1 rows)
 * @param ngram_max  longest suffix the drafter keys on
 * @param ngram_min  shortest suffix the drafter keys on
 * @param window     how many trailing history tokens the drafter may search; 0 = all
 * @param control    perturbation applied to the drafter's inputs
 * @param n_vocab    vocabulary size (only used by NGRAM_CONTROL_RANDOM)
 * @param seed       control RNG seed, so a repeat of a control arm is reproducible
 * @param out        [NGRAM_PHASE_COUNT] stats, accumulated into (not cleared)
 * @return           0, or -1 if the document is malformed
 *
 * Thread-safe: no (uses a static history scratch)  Reentrant: no  Signal-safe: no
 */
int ngram_accept_replay(const ngram_document *doc, int max_k, int ngram_max, int ngram_min,
                        int window, ngram_control control, uint32_t n_vocab, uint64_t seed,
                        ngram_accept_stats *out);

/** Sum `b` into `a`. */
void ngram_accept_stats_add(ngram_accept_stats *a, const ngram_accept_stats *b);

/** Expected tokens delivered per forward — the figure that converts to speed. */
double ngram_tokens_per_forward(const ngram_accept_stats *s);

/* ================================================================
 * Known-answer self-tests (no model, no card, no dump)
 * ================================================================ */

/**
 * Build a document whose continuation is a period-`period` repetition of the prompt's tail.
 * A correct simulator must reach the ceiling of k+1 tokens per forward on it.
 */
void ngram_selftest_periodic(ngram_document *doc, int n_prompt, int n_gen, int period,
                             uint32_t n_vocab);

/**
 * Build a document whose every token is drawn i.i.d. from a large vocabulary. A correct
 * simulator must sit at ~1.00 tokens per forward on it — this is the LOW end of the scale.
 */
void ngram_selftest_random(ngram_document *doc, int n_prompt, int n_gen, uint32_t n_vocab,
                           uint64_t seed);

/** Free a self-test document. */
void ngram_selftest_free(ngram_document *doc);

#endif /* PN_TESTS_NGRAM_ACCEPT_STREAM_H */
