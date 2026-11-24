/* SPDX-License-Identifier: AGPL-3.0-or-later */
/* Copyright (C) 2026 David (Corey) Herrera */
/**
 * @file tool_raw_excerpt.h
 * @brief The raw tool text on a [tool_call_dropped] line: its head AND its tail.
 *
 * A call dropped at max_tokens (call_incomplete) is usually long, and the head
 * alone cannot tell a run-on value from a loop: the cause is at the end.  Live,
 * rm4 pbthink 3: a grep dropped after 13.3 min, raw cut at 600 chars, cause
 * unknown.  So past `head` bytes the line also carries the last `tail` bytes
 * (never overlapping the head), the tail's start moved forward to a UTF-8 char
 * boundary.
 *
 * Format: `raw_len=N raw=<<<HEAD>>>`, or for a longer text
 * `raw_len=N raw=<<<HEAD…[truncated]>>> raw_tail=<<<TAIL>>>`.
 *
 * @status Used by openai_chat.c's [tool_call_dropped] line.
 * @issues None
 * @todo None
 */
#ifndef PANTHEON_TOOL_RAW_EXCERPT_H
#define PANTHEON_TOOL_RAW_EXCERPT_H

#include <stdio.h>
#include <string.h>

/** Bytes of the raw text the dropped-call line shows from each end. */
#define TOOL_RAW_EXCERPT_HEAD 800
#define TOOL_RAW_EXCERPT_TAIL 800

/**
 * Print `raw_len=… raw=<<<…>>>[ raw_tail=<<<…>>>]` for `raw` to `f`.
 *
 * Thread-safe: yes (stdio locks the stream)
 * Reentrant: yes
 * Signal-safe: no
 */
static inline void tool_raw_excerpt_print(FILE *f, const char *raw,
                                          size_t head, size_t tail) {
    size_t len = strlen(raw);
    if (len <= head) {
        fprintf(f, "raw_len=%zu raw=<<<%s>>>", len, raw);
        return;
    }
    size_t from = len - head > tail ? len - tail : head;
    while (from < len && ((unsigned char)raw[from] & 0xC0) == 0x80) from++;
    fprintf(f, "raw_len=%zu raw=<<<%.*s…[truncated]>>> raw_tail=<<<%s>>>",
            len, (int)head, raw, raw + from);
}

#endif /* PANTHEON_TOOL_RAW_EXCERPT_H */
