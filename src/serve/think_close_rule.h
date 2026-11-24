/* SPDX-License-Identifier: AGPL-3.0-or-later */
/* Copyright (C) 2026 David (Corey) Herrera */
/**
 * @file think_close_rule.h
 * @brief The one rule the emitter and every decoding grammar share for "where did
 *        the reasoning block end": a "</think>" text match, or the close TOKEN.
 *
 * WHY TOKEN MODE EXISTS.  Qwen3.x has "</think>" as one vocab entry (Qwen3.8: id
 * 248069), and that token is the real close.  The same five bytes spelled out of
 * ordinary tokens are TEXT: a model reasoning about a test that builds
 * "\n</think>\n" writes `</` `think` `>\` (510, 26003, 8189).  Matching the text
 * ended reasoning there: the emitter routed the rest of the thought to content,
 * and a "required" tool grammar compelled the call opener from the next token —
 * the model, still mid-string, could only take the whitespace the opener allows,
 * and wrote "   \r" until max_tokens (Qwen3.8 eval run 3, 2026-11-14: 8192
 * tokens, 10,265 chars of CR and spaces filed as the answer, no tool call).
 *
 * In token mode only a token whose whole text is "</think>" closes the block, so
 * the serve turns it on exactly when the model's close is such a token.  Every
 * caller feeds one token's text per call, which is what makes the whole-piece
 * test sound.  Text mode keeps the rolling char match for models without one.
 *
 * @status Used by serve_emitter.c, tool_grammar.c, tool_grammar_json.c and
 *         response_format_grammar.c.
 * @issues None
 * @todo None
 */
#ifndef PANTHEON_THINK_CLOSE_RULE_H
#define PANTHEON_THINK_CLOSE_RULE_H

#include <stdbool.h>
#include <string.h>

#define THINK_CLOSE_TEXT "</think>"

/**
 * Feed one char to the text-mode "</think>" matcher.
 *
 * @param matched     Rolling count of THINK_CLOSE_TEXT chars matched so far.
 * @param c           The next char of the stream.
 * @param token_mode  True when only the close token counts (never completes then).
 * @return            true when `c` completes the close (`*matched` is reset).
 *
 * Thread-safe: yes (touches only *matched)
 * Reentrant: yes
 * Signal-safe: yes
 */
static inline bool think_close_text_step(int *matched, char c, bool token_mode) {
    if (token_mode)
        return false;
    if (THINK_CLOSE_TEXT[*matched] == c) {
        if (THINK_CLOSE_TEXT[++*matched] != '\0')
            return false;
        *matched = 0;
        return true;
    }
    *matched = (c == THINK_CLOSE_TEXT[0]) ? 1 : 0;
    return false;
}

/**
 * Is this one token's text the close token?  Only ever true in token mode.
 *
 * Thread-safe: yes
 * Reentrant: yes
 * Signal-safe: yes
 */
static inline bool think_close_token_piece(const char *piece, bool token_mode) {
    return token_mode && piece && strcmp(piece, THINK_CLOSE_TEXT) == 0;
}

#endif /* PANTHEON_THINK_CLOSE_RULE_H */
