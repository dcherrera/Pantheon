/* SPDX-License-Identifier: AGPL-3.0-or-later */
/* Copyright (C) 2026 David (Corey) Herrera */
/**
 * @file chat_corpus.h
 * @brief Turn a text corpus into the token ids the PRODUCT would prefill — chat-templated
 *        through the model's own template, or plain through its own BPE — and back again.
 *
 *        A corpus assembled by hand from guessed ids proves nothing. The 20-probe accelerator
 *        gate read 0/20 because its ids were Llama-2/SentencePiece fed to a Qwen BPE vocab:
 *        decoded, probe 1 was `</s>ende+}}$--23et`. It never had the chance to fail. Every
 *        harness that scores a chat-framed prompt tokenises through this one module, so token
 *        soup is impossible by construction rather than by review.
 *
 * INPUT FORMAT (one document per blank-line-separated block):
 *
 *   S<TAB>system prompt text        (optional, at most one, anywhere in the document)
 *   U<TAB>user turn text
 *   A<TAB>assistant turn text
 *   <blank line>
 *
 * Lines beginning with '#' are comments, so a corpus can carry the reasoning for what is in
 * it next to the text itself.
 *
 * THE SYSTEM ROLE IS NOT OPTIONAL DECORATION. Framed as a real `<|im_start|>system` block the
 * defect corpus costs +22.1% perplexity on the Blackhole card; the same text framed as a
 * second user turn does not reproduce it. A corpus that cannot express a system prompt cannot
 * express what a serve sends, and a gate built on it is unrepresentative in the same way the
 * 20-probe gate was.
 *
 * The chat arm frames each user turn with pn_chat_format_first / pn_chat_format_turn and
 * encodes the assistant body inside the frame; the plain arm concatenates the same turn texts
 * with a blank line between them and BPE-encodes that. Same content, and the ONLY difference
 * is the wrapper — which is what makes a crossed control possible.
 *
 * @status Working
 * @issues None
 * @todo None
 */
#ifndef PN_TESTS_CHAT_CORPUS_H
#define PN_TESTS_CHAT_CORPUS_H

#include "../include/pantheon.h"

#include <stdint.h>
#include <stdio.h>

#define CHAT_CORPUS_MAX_TURNS      64
#define CHAT_CORPUS_MAX_TEXT_BYTES (256 * 1024)
#define CHAT_CORPUS_MAX_IDS        16384

/** One parsed document: alternating turns, each tagged user ('U') or assistant ('A'). */
typedef struct {
    char  role[CHAT_CORPUS_MAX_TURNS];
    char *text[CHAT_CORPUS_MAX_TURNS];
    int   n_turns;
} chat_document;

/**
 * Read one blank-line-terminated document from `in`.
 *
 * @return 1 when a document was read, 0 at end of input with nothing pending
 */
int chat_document_read(FILE *in, chat_document *doc);

/** Release the turn texts and reset the turn count. */
void chat_document_free(chat_document *doc);

/**
 * Chat-templated ids for one document: the model's own template frames each user turn and
 * opens each assistant turn, and the assistant body is encoded into the frame.
 *
 * @param ids      caller's buffer
 * @param max_ids  its capacity; the function REFUSES rather than truncating silently
 * @return         number of ids written, or -1
 */
int chat_document_to_chat_ids(pn_model *model, const chat_document *doc, uint32_t *ids,
                              int max_ids);

/** The same turn texts with a blank line between them and no role frame — the control arm. */
int chat_document_to_plain_ids(pn_model *model, const chat_document *doc, uint32_t *ids,
                               int max_ids);

/**
 * Decode ids back to text. Byte-level BPE ids come back GPT-2 byte-escaped, so they go
 * through the same unescape the engine uses.
 *
 * @return malloc'd text the caller frees, or NULL
 */
char *chat_ids_to_text(pn_model *model, const uint32_t *ids, int n_ids);

#endif /* PN_TESTS_CHAT_CORPUS_H */
