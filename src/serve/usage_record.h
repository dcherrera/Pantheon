/* SPDX-License-Identifier: AGPL-3.0-or-later */
/* Copyright (C) 2026 David (Corey) Herrera */
/**
 * @file usage_record.h
 * @brief One always-on `[usage]` log line per chat request, on both the
 *        batched and single-user paths, however the request ended:
 *
 *   [usage] id=chatcmpl-… tenant=acme model=qwen3.6 prompt=812 cached=640
 *           completion=96 choices=1 finish=stop
 *
 * It is the reconciliation record behind the wire `usage`: a gateway whose
 * stream died (client gone, stall, unload) still has the exact counts here,
 * joined on the completion id. `finish` is the wire finish_reason for a turn
 * that completed, else what stopped it: client_gone, model_unloaded, stalled,
 * shutdown, context_over_budget, memory_constrained, slot_unavailable, error.
 * No prompt or output text is ever written.
 *
 * @status Complete
 * @issues None
 * @todo None
 */
#ifndef PANTHEON_SERVE_USAGE_RECORD_H
#define PANTHEON_SERVE_USAGE_RECORD_H

#include <stddef.h>

/** One request's token counts, summed over its completions (`n`). */
typedef struct {
    int prompt;       /* prompt tokens, counted once per request */
    int cached;       /* of `prompt`, reused from the prefix cache */
    int completion;   /* generated tokens over every completion */
    int choices;      /* completions that ran */
} pn_usage_counts;

/**
 * Format the line (no trailing newline). NULL tenant/model print "-".
 * @return the snprintf length (>= cap means truncated)
 *
 * Thread-safe: yes. Reentrant: yes. Signal-safe: no
 */
int pn_usage_line_format(char *buf, size_t cap, const char *id, const char *tenant,
                         const char *model, const pn_usage_counts *u,
                         const char *finish);

/** Format and write the line to stderr. */
void pn_usage_line_log(const char *id, const char *tenant, const char *model,
                       const pn_usage_counts *u, const char *finish);

#endif /* PANTHEON_SERVE_USAGE_RECORD_H */
