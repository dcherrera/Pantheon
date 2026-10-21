/* SPDX-License-Identifier: AGPL-3.0-or-later */
/* Copyright (C) 2026 David (Corey) Herrera */
/**
 * @file serve_emitter.h
 * @brief Streaming reasoning/content/tool-call splitter for the serve layer.
 *
 * The assistant turn is prefilled with <think>, so generation begins INSIDE the
 * think block: the model emits reasoning, then "</think>", then the answer.
 * This splitter routes text before "</think>" to the reasoning channel and text
 * after it to the content channel, holding back a partial tail so a marker (the
 * "</think>" close, a request stop string, or a tool-call opener) is still
 * caught when it straddles a token boundary.
 *
 * Tool-call openers.  In the content channel the emitter diverts a tool call
 * into a raw `tool` channel (→ sg_result.tool_raw, parsed by tc_parse_tool_calls)
 * instead of streaming it as content.  Two openers are recognized:
 *   - "<tool_call>"  — the canonical wrapper; detected UNCONDITIONALLY.
 *   - "<function="   — a BARE call with no wrapper (nex-n2-mini at Q3_K_M emits
 *     this most of the time).  Detected ONLY when the request carried `tools`
 *     (`has_tools`) and the tag has a plausible `<function=NAME>` shape (a name
 *     char follows the `=`), to avoid capturing a literal `<function=` fragment
 *     in a normal answer.  Whichever opener appears FIRST in the buffer wins,
 *     mirroring the opener-vs-stop-string "first wins" rule.
 * The token-boundary hold-back covers both openers, so an opener split across
 * two token deltas is not emitted as content and then missed.
 *
 * Every byte diverted into the tool channel is also handed, as it arrives, to an
 * optional tap (em_set_tool_tap) — the streaming serve uses it to put tool-call
 * deltas on the wire while the call is still being written.
 *
 * The emitter functions depend only on libc (no engine/CUDA deps), so the
 * splitter is unit-testable in isolation (tests/test_serve_emitter.c).
 *
 * @status Complete for CPU host serving.
 * @issues None
 * @todo None
 */

#ifndef PN_SERVE_EMITTER_H
#define PN_SERVE_EMITTER_H

#include <stdbool.h>
#include <stddef.h>

#include "serve_generation.h"   /* sg_delta_fn */

/* How many tokens' logprobs can queue up behind a held-back tail before the
 * oldest are dropped.  The hold-back is bounded by the longest marker (a dozen
 * chars), so a handful of tokens is the real ceiling; the cap only matters for a
 * long diverted region (a tool call), whose tokens are not reported anyway. */
#define EM_LOGPROB_PENDING_MAX 32

/* Receives bytes as they are appended to the raw tool channel. */
typedef void (*em_tool_tap_fn)(const char *bytes, size_t n, void *ud);

/* Growable text channel (reasoning, content, or raw-tool accumulator). */
typedef struct {
    char  *data;
    size_t len;
    size_t cap;
} text_channel;

typedef struct {
    int          in_think;
    int          in_tool;      /* capturing a tool-call block into `tool` */
    bool         has_tools;    /* request carried tools → arm bare <function= */
    tc_dialect   dialect;      /* resident model's dialect: ORNITH_JSON diverts a
                                * ```json fence / bare "{" tool block instead of
                                * the XML openers.  Default 0 = QWEN_XML. */
    bool         reasoning_open;
    bool         content_open;
    bool         stopped;
    char        *carry;
    size_t       carry_len;
    size_t       carry_cap;
    text_channel reasoning;
    text_channel content;
    text_channel tool;         /* raw model text from the tool-call opener onward */
    sg_delta_fn  cb;
    void        *ud;
    char       **stops;
    int          n_stop;

    /* OpenAI `logprobs` pass-through.  The splitter holds a partial tail back to
     * catch markers, so ONE content delta can cover several tokens and a token
     * can produce no delta at all — a client cannot pair logprobs with deltas by
     * counting.  The decode loop notes each generated token's logprobs here and
     * em_emit hands the accumulated set to the content delta that finally
     * carries those bytes.  Empty unless the request asked for logprobs. */
    const sg_token_logprob *lp_pending[EM_LOGPROB_PENDING_MAX];
    int                     n_lp_pending;

    /* Tool-channel tap (em_set_tool_tap): every byte appended to `tool` is also
     * handed here, in order, as it is generated — the streaming serve turns it
     * into tool-call deltas (tool_call_stream.h).  NULL = no tap. */
    em_tool_tap_fn          tool_tap;
    void                   *tool_tap_ud;
} emitter;

/* Initialize an emitter.  `has_tools` arms the bare-"<function=" opener (the
 * canonical "<tool_call>" opener is always armed); `dialect` selects the opener
 * set (ORNITH_JSON diverts a ```json fence / bare "{" JSON tool block instead of
 * the XML openers).  `stops`/`n_stop` are the request's stop strings (borrowed). */
void em_init(emitter *em, sg_delta_fn cb, void *ud,
             char **stops, int n_stop, bool has_tools, tc_dialect dialect);

/* Attach a tool-channel tap (NULL detaches).  Call after em_init. */
void em_set_tool_tap(emitter *em, em_tool_tap_fn tap, void *ud);

/* Feed one decoded token's text through the splitter. */
void em_feed(emitter *em, const char *piece);

/* Note the logprobs of the token about to be fed, so they ride the content delta
 * that eventually carries its bytes.  Borrowed — `lp` must outlive the emitter
 * (the serve preallocates the whole array for the turn).  NULL is a no-op. */
void em_note_token_logprob(emitter *em, const sg_token_logprob *lp);

/* Flush any held-back tail at end of generation (no marker/stop completed). */
void em_flush(emitter *em);

/* Free the emitter's owned buffers. */
void em_free(emitter *em);

#endif /* PN_SERVE_EMITTER_H */
