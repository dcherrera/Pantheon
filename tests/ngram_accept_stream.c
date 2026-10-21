/* SPDX-License-Identifier: AGPL-3.0-or-later */
/* Copyright (C) 2026 David (Corey) Herrera */
/**
 * @file ngram_accept_stream.c
 * @brief Implementation of the recorded-stream dump and the speculative-round simulator
 *        declared in ngram_accept_stream.h.
 *
 * The drafter itself is NOT reimplemented here: `pn_ngram_draft` from src/core/ngram_draft.c
 * is the one that ships, and a second copy written for a measurement is a measurement of the
 * copy.
 *
 * @status Working
 * @issues None
 * @todo None
 */

/* 1. This module's own header */
#include "ngram_accept_stream.h"

/* 2. Standard library */
#include <stdlib.h>
#include <string.h>

/* 4. Project headers */
#include "../src/core/ngram_draft.h"

#define NGRAM_MAX_HISTORY 65536
#define NGRAM_MAX_DRAFT   32

const char *const ngram_length_bucket_name[NGRAM_LEN_BUCKETS] = {
    "<512", "512-1k", "1k-2k", "2k-4k", "4k-8k", ">=8k"
};

/** Which context-length bucket a history of `n` tokens falls in. */
static int ngram_length_bucket(int n)
{
    if (n < 512)   return 0;
    if (n < 1024)  return 1;
    if (n < 2048)  return 2;
    if (n < 4096)  return 3;
    if (n < 8192)  return 4;
    return 5;
}

/** splitmix64 — a seeded stream so a control arm repeats exactly. */
static uint64_t ngram_rng_next(uint64_t *state)
{
    uint64_t z = (*state += 0x9E3779B97F4A7C15ULL);

    z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9ULL;
    z = (z ^ (z >> 27)) * 0x94D049BB133111EBULL;
    return z ^ (z >> 31);
}

/* ================================================================
 * The dump file
 * ================================================================ */

/** Write one little-endian uint32. */
static int ngram_put_u32(FILE *f, uint32_t v)
{
    unsigned char b[4] = { (unsigned char)(v & 0xFF), (unsigned char)((v >> 8) & 0xFF),
                           (unsigned char)((v >> 16) & 0xFF), (unsigned char)((v >> 24) & 0xFF) };

    return fwrite(b, 1, sizeof(b), f) == sizeof(b) ? 0 : -1;
}

/** Read one little-endian uint32. */
static int ngram_get_u32(FILE *f, uint32_t *out)
{
    unsigned char b[4];

    if (fread(b, 1, sizeof(b), f) != sizeof(b))
        return -1;
    *out = (uint32_t)b[0] | ((uint32_t)b[1] << 8) | ((uint32_t)b[2] << 16) |
           ((uint32_t)b[3] << 24);
    return 0;
}

/** Write a fixed-width name field, zero-padded, never truncated mid-run without notice. */
static int ngram_put_name(FILE *f, const char *s)
{
    char buf[NGRAM_STREAM_NAME_LEN] = {0};

    snprintf(buf, sizeof(buf), "%s", s ? s : "");
    return fwrite(buf, 1, sizeof(buf), f) == sizeof(buf) ? 0 : -1;
}

int ngram_stream_open_write(const char *path, uint32_t n_vocab, uint32_t arch, uint32_t eos_id,
                            uint32_t think_end_id, const char *arm, const char *tag,
                            FILE **out)
{
    FILE *f = fopen(path, "wb");

    if (!f) {
        fprintf(stderr, "ngram_stream: cannot create %s\n", path);
        return -1;
    }
    if (fwrite(NGRAM_STREAM_MAGIC, 1, 8, f) != 8 ||
        ngram_put_u32(f, NGRAM_STREAM_ENDIAN) != 0 || ngram_put_u32(f, n_vocab) != 0 ||
        ngram_put_u32(f, arch) != 0 || ngram_put_u32(f, eos_id) != 0 ||
        ngram_put_u32(f, think_end_id) != 0 || ngram_put_name(f, arm) != 0 ||
        ngram_put_name(f, tag) != 0) {
        fprintf(stderr, "ngram_stream: header write failed for %s\n", path);
        fclose(f);
        return -1;
    }
    *out = f;
    return 0;
}

int ngram_stream_write_document(FILE *f, const ngram_document *doc)
{
    uint64_t bits;

    if (ngram_put_u32(f, (uint32_t)doc->n_prompt) != 0 ||
        ngram_put_u32(f, (uint32_t)doc->n_gen) != 0 ||
        ngram_put_u32(f, doc->think_end_pos < 0 ? NGRAM_STREAM_NONE
                                                : (uint32_t)doc->think_end_pos) != 0)
        return -1;
    memcpy(&bits, &doc->decode_s, sizeof(bits));
    if (ngram_put_u32(f, (uint32_t)(bits & 0xFFFFFFFFu)) != 0 ||
        ngram_put_u32(f, (uint32_t)(bits >> 32)) != 0)
        return -1;
    for (int i = 0; i < doc->n_prompt; i++)
        if (ngram_put_u32(f, doc->prompt[i]) != 0)
            return -1;
    for (int i = 0; i < doc->n_gen; i++)
        if (ngram_put_u32(f, doc->gen[i]) != 0)
            return -1;
    return fflush(f);
}

int ngram_stream_close_write(FILE *f)
{
    int rc = ngram_put_u32(f, NGRAM_STREAM_END);

    if (fclose(f) != 0)
        rc = -1;
    return rc;
}

/** Read one document body; returns 0 on success, 1 at the end marker, -1 on damage. */
static int ngram_read_document(FILE *f, ngram_document *doc)
{
    uint32_t first, n_gen, think, lo, hi;
    uint64_t bits;
    int rc = -1;

    if (ngram_get_u32(f, &first) != 0)
        return -1;
    if (first == NGRAM_STREAM_END)
        return 1;
    if (ngram_get_u32(f, &n_gen) != 0 || ngram_get_u32(f, &think) != 0 ||
        ngram_get_u32(f, &lo) != 0 || ngram_get_u32(f, &hi) != 0)
        return -1;
    if (first > NGRAM_MAX_HISTORY || n_gen > NGRAM_MAX_HISTORY)
        return -1;
    bits = (uint64_t)lo | ((uint64_t)hi << 32);
    memcpy(&doc->decode_s, &bits, sizeof(doc->decode_s));
    doc->n_prompt = (int)first;
    doc->n_gen = (int)n_gen;
    doc->think_end_pos = think == NGRAM_STREAM_NONE ? -1 : (int)think;
    doc->prompt = calloc((size_t)first + 1, sizeof(*doc->prompt));
    doc->gen = calloc((size_t)n_gen + 1, sizeof(*doc->gen));
    if (!doc->prompt || !doc->gen)
        goto cleanup;
    for (uint32_t i = 0; i < first; i++)
        if (ngram_get_u32(f, &doc->prompt[i]) != 0)
            goto cleanup;
    for (uint32_t i = 0; i < n_gen; i++)
        if (ngram_get_u32(f, &doc->gen[i]) != 0)
            goto cleanup;
    rc = 0;

cleanup:
    if (rc != 0) {
        free(doc->prompt);
        free(doc->gen);
        doc->prompt = NULL;
        doc->gen = NULL;
    }
    return rc;
}

int ngram_stream_read(const char *path, ngram_stream *out)
{
    char magic[8];
    uint32_t endian;
    FILE *f = fopen(path, "rb");
    int rc = 0;

    memset(out, 0, sizeof(*out));
    if (!f) {
        fprintf(stderr, "ngram_stream: cannot open %s\n", path);
        return -1;
    }
    if (fread(magic, 1, sizeof(magic), f) != sizeof(magic) ||
        memcmp(magic, NGRAM_STREAM_MAGIC, sizeof(magic)) != 0) {
        fprintf(stderr, "ngram_stream: %s is not a stream dump\n", path);
        fclose(f);
        return -1;
    }
    if (ngram_get_u32(f, &endian) != 0 || endian != NGRAM_STREAM_ENDIAN ||
        ngram_get_u32(f, &out->n_vocab) != 0 || ngram_get_u32(f, &out->arch) != 0 ||
        ngram_get_u32(f, &out->eos_id) != 0 || ngram_get_u32(f, &out->think_end_id) != 0 ||
        fread(out->arm, 1, sizeof(out->arm), f) != sizeof(out->arm) ||
        fread(out->tag, 1, sizeof(out->tag), f) != sizeof(out->tag)) {
        fprintf(stderr, "ngram_stream: %s has a damaged header\n", path);
        fclose(f);
        return -1;
    }
    out->arm[NGRAM_STREAM_NAME_LEN - 1] = '\0';
    out->tag[NGRAM_STREAM_NAME_LEN - 1] = '\0';
    while (out->n_docs < NGRAM_STREAM_MAX_DOCS) {
        rc = ngram_read_document(f, &out->docs[out->n_docs]);
        if (rc != 0)
            break;
        out->n_docs++;
    }
    fclose(f);
    /* A dump whose writer died mid-run has no end marker. Refuse it rather than reporting an
     * acceptance rate over however many documents happened to land — the eighth vacuous-gate
     * class in this project was a harness scoring bytes it should have refused. */
    if (rc != 1) {
        fprintf(stderr, "ngram_stream: %s has no end marker — the writer did not finish. "
                        "Refusing rather than scoring %d partial document(s).\n",
                path, out->n_docs);
        ngram_stream_free(out);
        return -1;
    }
    return 0;
}

void ngram_stream_free(ngram_stream *s)
{
    for (int i = 0; i < s->n_docs; i++) {
        free(s->docs[i].prompt);
        free(s->docs[i].gen);
    }
    memset(s, 0, sizeof(*s));
}

/* ================================================================
 * The simulator
 * ================================================================ */

/**
 * Present the drafter's search space for one round, applying the control.
 *
 * NGRAM_CONTROL_SHUFFLE permutes everything EXCEPT the trailing `ngram_max` tokens, so the
 * KEY is still the model's own real suffix and only the structure it is searching for is
 * destroyed. Shuffling the key too would confound "the history has no structure" with "the
 * query is nonsense".
 *
 * @return the number of tokens the drafter may see; *view points at them
 */
static int ngram_present_history(const uint32_t *hist, int nh, int window, int ngram_max,
                                 ngram_control control, uint64_t *rng, uint32_t **scratch,
                                 const uint32_t **view)
{
    int n = nh;
    const uint32_t *base = hist;

    if (window > 0 && n > window) {
        base = hist + (n - window);
        n = window;
    }
    if (control != NGRAM_CONTROL_SHUFFLE) {
        *view = base;
        return n;
    }
    memcpy(*scratch, base, (size_t)n * sizeof(**scratch));
    for (int i = n - ngram_max - 1; i > 0; i--) {
        int j = (int)(ngram_rng_next(rng) % (uint64_t)(i + 1));
        uint32_t t = (*scratch)[i];

        (*scratch)[i] = (*scratch)[j];
        (*scratch)[j] = t;
    }
    *view = *scratch;
    return n;
}

int ngram_accept_replay(const ngram_document *doc, int max_k, int ngram_max, int ngram_min,
                        int window, ngram_control control, uint32_t n_vocab, uint64_t seed,
                        ngram_accept_stats *out)
{
    static uint32_t hist[NGRAM_MAX_HISTORY];
    static uint32_t scratch[NGRAM_MAX_HISTORY];
    uint32_t *scratch_p = scratch;
    uint32_t drafts[NGRAM_MAX_DRAFT];
    uint64_t rng = seed;
    int nh, p;

    if (max_k < 1 || max_k > NGRAM_MAX_DRAFT || doc->n_gen < 2)
        return -1;
    if (doc->n_prompt + doc->n_gen > NGRAM_MAX_HISTORY)
        return -1;
    memcpy(hist, doc->prompt, (size_t)doc->n_prompt * sizeof(*hist));
    nh = doc->n_prompt;
    hist[nh++] = doc->gen[0];

    for (p = 0; p < doc->n_gen - 1; ) {
        int phase = (doc->think_end_pos >= 0 && p < doc->think_end_pos) ? NGRAM_PHASE_REASONING
                                                                       : NGRAM_PHASE_ANSWER;
        ngram_accept_stats *s = &out[phase];
        int bucket = ngram_length_bucket(nh);
        const uint32_t *view = NULL;
        int nv = ngram_present_history(hist, nh, window, ngram_max, control, &rng, &scratch_p,
                                       &view);
        int k = pn_ngram_draft(view, nv, ngram_max, ngram_min, max_k, drafts);
        int delivered, m = 0;

        if (k > 0 && control == NGRAM_CONTROL_RANDOM)
            for (int i = 0; i < k; i++)
                drafts[i] = (uint32_t)(ngram_rng_next(&rng) % (uint64_t)n_vocab);
        if (k > 0) {
            /* The round's free token is the argmax after the last ACCEPTED draft, so
             * accepting m drafts delivers m+1 tokens and needs gen[p+m+1] to exist. */
            int lim = doc->n_gen - p - 2;

            if (lim > k)
                lim = k;
            while (m < lim && drafts[m] == doc->gen[p + 1 + m])
                m++;
            delivered = m + 1;
            s->rounds++;
            s->drafted += k;
            s->accepted += m;
            s->hist[delivered]++;
        } else {
            delivered = 1;
            s->plain_rounds++;
        }
        s->forwards++;
        s->emitted += delivered;
        s->bucket_forwards[bucket]++;
        s->bucket_emitted[bucket] += delivered;
        for (int i = 0; i < delivered; i++)
            hist[nh++] = doc->gen[p + 1 + i];
        p += delivered;
    }
    return 0;
}

void ngram_accept_stats_add(ngram_accept_stats *a, const ngram_accept_stats *b)
{
    a->rounds += b->rounds;
    a->plain_rounds += b->plain_rounds;
    a->drafted += b->drafted;
    a->accepted += b->accepted;
    a->forwards += b->forwards;
    a->emitted += b->emitted;
    for (int i = 0; i < (int)(sizeof(a->hist) / sizeof(a->hist[0])); i++)
        a->hist[i] += b->hist[i];
    for (int i = 0; i < NGRAM_LEN_BUCKETS; i++) {
        a->bucket_emitted[i] += b->bucket_emitted[i];
        a->bucket_forwards[i] += b->bucket_forwards[i];
    }
}

double ngram_tokens_per_forward(const ngram_accept_stats *s)
{
    return s->forwards ? (double)s->emitted / (double)s->forwards : 0.0;
}

/* ================================================================
 * Known-answer self-tests
 * ================================================================ */

void ngram_selftest_periodic(ngram_document *doc, int n_prompt, int n_gen, int period,
                             uint32_t n_vocab)
{
    memset(doc, 0, sizeof(*doc));
    doc->prompt = calloc((size_t)n_prompt, sizeof(*doc->prompt));
    doc->gen = calloc((size_t)n_gen, sizeof(*doc->gen));
    doc->n_prompt = n_prompt;
    doc->n_gen = n_gen;
    doc->think_end_pos = -1;
    for (int i = 0; i < n_prompt; i++)
        doc->prompt[i] = (uint32_t)((i % period) + 1) % n_vocab;
    for (int i = 0; i < n_gen; i++)
        doc->gen[i] = (uint32_t)(((n_prompt + i) % period) + 1) % n_vocab;
}

void ngram_selftest_random(ngram_document *doc, int n_prompt, int n_gen, uint32_t n_vocab,
                           uint64_t seed)
{
    uint64_t rng = seed;

    memset(doc, 0, sizeof(*doc));
    doc->prompt = calloc((size_t)n_prompt, sizeof(*doc->prompt));
    doc->gen = calloc((size_t)n_gen, sizeof(*doc->gen));
    doc->n_prompt = n_prompt;
    doc->n_gen = n_gen;
    doc->think_end_pos = -1;
    for (int i = 0; i < n_prompt; i++)
        doc->prompt[i] = (uint32_t)(ngram_rng_next(&rng) % (uint64_t)n_vocab);
    for (int i = 0; i < n_gen; i++)
        doc->gen[i] = (uint32_t)(ngram_rng_next(&rng) % (uint64_t)n_vocab);
}

void ngram_selftest_free(ngram_document *doc)
{
    free(doc->prompt);
    free(doc->gen);
    memset(doc, 0, sizeof(*doc));
}
