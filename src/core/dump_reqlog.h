/* SPDX-License-Identifier: AGPL-3.0-or-later */
/* Copyright (C) 2026 David (Corey) Herrera */
/**
 * @file dump_reqlog.h
 * @brief The request log beside a hidden-state dump (`PN_TT_DUMP_REQLOG=<file>`): one JSON line
 *        per request, so a dump can be split into sequences with their sampling settings
 *        without scraping the serve log (mtpft `pack --reqlog`).
 *
 * The dump writer counts every record it writes (pn_dump_reqlog_count_records); the serve
 * writes one line when a request ends, after its last record was flushed:
 *
 *   {"req":0,"first_record":0,"records":57,"prompt_tokens":20,"completion_tokens":38,
 *    "max_tokens":400,"temperature":0.6,"top_k":20,"top_p":0.95,"seed":1235,
 *    "finish":"length","prompt_sha256":"<64 hex>","spec_cycles":12,"spec_drafts":31,
 *    "spec_accepted":24}
 *
 * `first_record`/`records` index the dump's records (0 records when no dump is armed);
 * `seed` is null when the request sent none; the three `spec_*` fields (this request's own
 * speculative counters) are present only when the backend reports them. `prompt_sha256` is the
 * caller's hash of the prompt (the serve hashes the prompt token ids, int32 little-endian: it
 * renders straight to tokens). Off (unset or empty) = no file and every entry point returns at
 * once; the dump writer only calls in when a dump is armed.
 *
 * status: used by qwen35moe_resident_tt.c (the dump) and serve_generation.c (the request end).
 * issues: none.
 * todo:   none.
 */
#ifndef PN_DUMP_REQLOG_H
#define PN_DUMP_REQLOG_H

#include <stdbool.h>
#include <stdint.h>

/** One finished request, as the log records it. */
typedef struct {
    int prompt_tokens, completion_tokens, max_tokens, top_k;
    float temperature, top_p;
    bool seed_set;
    uint64_t seed;
    const char *finish;              /* "length", "stop", "tool_calls", ... */
    const char *prompt_sha256_hex;   /* 64 lowercase hex chars, or NULL ("") */
    bool has_spec;
    uint64_t spec_cycles, spec_drafts, spec_accepted;
} pn_dump_reqlog_request;

/** Open `path` for the log (truncating), resetting the record and request counters. Reads no
 *  environment. @return 0, or -1 when the file cannot be opened. */
int pn_dump_reqlog_open(const char *path);

/** Close the log (a no-op when none is open). */
void pn_dump_reqlog_close(void);

/** True when a log is open; the first call opens `PN_TT_DUMP_REQLOG` when it names a file. */
bool pn_dump_reqlog_enabled(void);

/** The dump wrote `n` more records. */
void pn_dump_reqlog_count_records(uint64_t n);

/** Write `r`'s line (records since the previous line) and flush. @return 0, -1 on a write error
 *  or when no log is open. */
int pn_dump_reqlog_write_request(const pn_dump_reqlog_request *r);

#endif /* PN_DUMP_REQLOG_H */
