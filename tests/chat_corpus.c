/* SPDX-License-Identifier: AGPL-3.0-or-later */
/* Copyright (C) 2026 David (Corey) Herrera */
/**
 * @file chat_corpus.c
 * @brief Implementation of the corpus-to-token-ids conversion declared in chat_corpus.h.
 *
 * @status Working
 * @issues None
 * @todo None
 */

#include "chat_corpus.h"

#include "../src/core/bpe_pretokenize.h"
#include "../src/core/chat.h"
#include "../src/core/engine.h"
#include "tokenizer/tokenizer.h"

#include <stdlib.h>
#include <string.h>

void chat_document_free(chat_document *doc)
{
    for (int i = 0; i < doc->n_turns; i++) {
        free(doc->text[i]);
        doc->text[i] = NULL;
    }
    doc->n_turns = 0;
}

int chat_document_read(FILE *in, chat_document *doc)
{
    static char line[CHAT_CORPUS_MAX_TEXT_BYTES];

    doc->n_turns = 0;
    while (fgets(line, sizeof(line), in)) {
        size_t len = strlen(line);
        char *body;

        while (len > 0 && (line[len - 1] == '\n' || line[len - 1] == '\r'))
            line[--len] = '\0';
        if (len == 0) {
            if (doc->n_turns > 0)
                return 1;
            continue;
        }
        if (line[0] == '#')
            continue;
        if ((line[0] != 'U' && line[0] != 'A' && line[0] != 'S') || line[1] != '\t') {
            fprintf(stderr, "chat_corpus: line is not S<TAB>, U<TAB> or A<TAB>: %.40s\n", line);
            continue;
        }
        if (doc->n_turns >= CHAT_CORPUS_MAX_TURNS) {
            fprintf(stderr, "chat_corpus: more than %d turns in one document\n",
                    CHAT_CORPUS_MAX_TURNS);
            continue;
        }
        body = strdup(line + 2);
        if (!body)
            return doc->n_turns > 0;
        doc->role[doc->n_turns] = line[0];
        doc->text[doc->n_turns] = body;
        doc->n_turns++;
    }
    return doc->n_turns > 0;
}

/** Append `n` ids, refusing rather than truncating silently. */
static int ids_append(uint32_t *ids, int *n_ids, int max_ids, const uint32_t *add, int n)
{
    if (*n_ids + n > max_ids) {
        fprintf(stderr, "chat_corpus: document exceeds %d ids\n", max_ids);
        return -1;
    }
    for (int i = 0; i < n; i++)
        ids[(*n_ids)++] = add[i];
    return 0;
}

/** BPE-encode `text` and append its ids. The chat arm needs this for the assistant BODY,
 *  which the template functions do not carry — they emit the role frame only. */
static int ids_append_text(pn_model *model, uint32_t *ids, int *n_ids, int max_ids,
                           const char *text)
{
    int n = 0;
    int *enc = pn_bpe_encode(model->tokenizer, text, &n);
    int rc = 0;

    if (!enc || n <= 0) {
        fprintf(stderr, "chat_corpus: encode failed for %.40s\n", text);
        free(enc);
        return -1;
    }
    for (int i = 0; i < n && rc == 0; i++) {
        uint32_t id = (uint32_t)enc[i];

        rc = ids_append(ids, n_ids, max_ids, &id, 1);
    }
    free(enc);
    return rc;
}

int chat_document_to_chat_ids(pn_model *model, const chat_document *doc, uint32_t *ids,
                              int max_ids)
{
    pn_chat_style style = (pn_chat_style)model->chat_style;
    const char *system = NULL;
    int n_ids = 0;
    int first_user = 1;

    /* A leading S turn is the SYSTEM prompt, and it belongs in the template's own system
     * block rather than as another user turn. That is not cosmetic: framed as a system block
     * the defect corpus costs +22.1% perplexity on this card, and the SAME text framed as a
     * second user turn (which also leaves an empty assistant turn between them) does not
     * reproduce it at all. A gate that cannot emit a system block is not scoring what a serve
     * sends. */
    for (int t = 0; t < doc->n_turns; t++)
        if (doc->role[t] == 'S' && !system)
            system = doc->text[t];

    for (int t = 0; t < doc->n_turns; t++) {
        if (doc->role[t] == 'S') {
            continue;
        } else if (doc->role[t] == 'U') {
            pn_token_seq seq = first_user
                                   ? pn_chat_format_first(style, model->tokenizer, system,
                                                          doc->text[t])
                                   : pn_chat_format_turn(style, model->tokenizer, doc->text[t]);
            int rc;

            if (!seq.tokens || seq.n_tokens <= 0) {
                fprintf(stderr, "chat_corpus: chat formatting failed at turn %d\n", t);
                free(seq.tokens);
                return -1;
            }
            rc = ids_append(ids, &n_ids, max_ids, seq.tokens, seq.n_tokens);
            free(seq.tokens);
            if (rc != 0)
                return -1;
            first_user = 0;
        } else if (ids_append_text(model, ids, &n_ids, max_ids, doc->text[t]) != 0) {
            return -1;
        }
    }
    return n_ids;
}

int chat_document_to_plain_ids(pn_model *model, const chat_document *doc, uint32_t *ids,
                               int max_ids)
{
    static char joined[CHAT_CORPUS_MAX_TEXT_BYTES];
    int n_ids = 0;
    size_t used = 0;

    joined[0] = '\0';
    for (int t = 0; t < doc->n_turns; t++) {
        int n = snprintf(joined + used, sizeof(joined) - used, "%s%s",
                         t == 0 ? "" : "\n\n", doc->text[t]);

        if (n < 0 || (size_t)n >= sizeof(joined) - used) {
            fprintf(stderr, "chat_corpus: joined document exceeds %zu bytes\n", sizeof(joined));
            return -1;
        }
        used += (size_t)n;
    }
    if (ids_append_text(model, ids, &n_ids, max_ids, joined) != 0)
        return -1;
    return n_ids;
}

char *chat_ids_to_text(pn_model *model, const uint32_t *ids, int n_ids)
{
    int is_bpe = ft_get_model_type(model->tokenizer) == 0;
    int *signed_ids = malloc((size_t)n_ids * sizeof(int));
    char *raw, *text;

    if (!signed_ids)
        return NULL;
    for (int i = 0; i < n_ids; i++)
        signed_ids[i] = (int)ids[i];
    raw = ft_decode(model->tokenizer, signed_ids, n_ids);
    free(signed_ids);
    if (!raw)
        return NULL;
    text = is_bpe ? gpt2_bytes_decode(raw) : NULL;
    if (!text)
        return raw;
    free(raw);
    return text;
}
