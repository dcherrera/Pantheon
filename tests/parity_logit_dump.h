/* SPDX-License-Identifier: AGPL-3.0-or-later */
/* Copyright (C) 2026 David (Corey) Herrera */
/**
 * @file parity_logit_dump.h
 * @brief A per-position logit-row dump, so a parity gate can run its two arms as two SEPARATE
 *        passes and pair them offline instead of interleaving them inside one process.
 *
 * ---------------------------------------------------------------------------
 * WHY A DUMP EXISTS AT ALL
 * ---------------------------------------------------------------------------
 * The interleaved gate (test_chat_parity_gate.c) calls the accelerator and the CPU oracle
 * alternately, position by position. On this card that puts a **~26 s CPU-oracle forward
 * between two consecutive card forwards**, which leaves every resident triad armed and idle
 * for 26 s at a stretch — and a resident queue is BOUNDED on purpose, so it retires. Measured
 * on silicon 2026-08-18: **802 / 2406 / 442 descriptor refusals** across the gate's arms
 * against **0** on single-arm runs of the same ops at the same length, and the GDN arm died at
 * position 1 of document 0. The recovery is real and the results reproduce, but a harness that
 * provokes 2406 refusals is not one to gate on.
 *
 * It also costs three times what it needs to. The oracle pass is the expensive one (~26 s a
 * scored position, 4.5 GB of major faults a token from a 9.5 GB mmap on a 7.3 GB host) and it
 * is IDENTICAL for every arm scored against it. Interleaved, every arm pays for its own copy:
 * the Phase 6 ladder was ~14 h for what is one ~3 h oracle pass plus minutes of card passes.
 *
 * ---------------------------------------------------------------------------
 * WHAT MAKES OFFLINE PAIRING SAFE
 * ---------------------------------------------------------------------------
 * Two passes are only equivalent to one interleaved pass if they scored **the same positions
 * of the same token stream**. That is not assumed — every record carries its own
 * `(document, position, next_token)` triple and the pairing step REFUSES on the first
 * mismatch. A corpus edited between the two passes, a `--plain` flag on one and not the
 * other, or a tokeniser change is then a loud refusal rather than a silently misaligned score.
 * The two arms' states were already independent inside the interleaved gate — it opens a
 * separate `pn_state` per arm — so nothing but host load and card temperature was ever held
 * common, and neither is an input to the arithmetic.
 *
 * ---------------------------------------------------------------------------
 * FORMAT
 * ---------------------------------------------------------------------------
 * One header, then a flat sequence of (record, fp32 row) pairs. Rows are fp32 because the
 * score is an exact function of them: writing bf16 would inject the dump's own rounding into
 * the very quantity the gate thresholds. At Ornith's vocabulary a row is ~600 KB, so a
 * 400-position pass is ~240 MB — nothing beside the 9.5 GB the model already is.
 *
 * `record_count` is written at CLOSE. A dump from a run that died mid-corpus therefore reads
 * as zero records and is refused, instead of pairing a truncated arm against a whole oracle.
 *
 * @status Working
 * @issues None
 * @todo None
 */
#ifndef PN_TESTS_PARITY_LOGIT_DUMP_H
#define PN_TESTS_PARITY_LOGIT_DUMP_H

#include <stdint.h>
#include <stdio.h>

/** "PNLD" + format version. Bump on any layout change: a reader that accepted an older
 *  layout by ignoring the version would pair rows against the wrong offsets. */
#define PARITY_DUMP_MAGIC   0x504E4C44u
#define PARITY_DUMP_VERSION 1u

/** How long an arm label may be, including the terminator. It is carried in the file so a
 *  pairing report can say WHICH arm each side was without being told twice. */
#define PARITY_DUMP_LABEL_BYTES 48

/** File header. Fixed size and fixed field order; `record_count` is patched in at close. */
typedef struct {
    uint32_t magic;
    uint32_t version;
    uint32_t n_vocab;
    uint32_t arch;
    uint32_t framing_is_chat; /**< 1 chat-templated, 0 plain — a crossed control is a
                                   different corpus and must not pair against this one */
    uint32_t seam_enabled;    /**< what parity_accel_seam_enable was held at for this pass */
    uint64_t record_count;
    char     label[PARITY_DUMP_LABEL_BYTES];
} parity_dump_header;

/** One scored position. `argmax` is the writer's own top-1 over the row it wrote, kept so a
 *  reader can verify it read the row back at the offset it thinks it did. */
typedef struct {
    int32_t document;
    int32_t position;
    int32_t next_token;
    int32_t argmax;
} parity_dump_record;

/** An open dump being written. */
typedef struct {
    FILE    *file;
    uint32_t n_vocab;
    uint64_t records;
} parity_dump_writer;

/** An open dump being read. */
typedef struct {
    FILE              *file;
    parity_dump_header header;
    uint64_t           read_so_far;
} parity_dump_reader;

/**
 * Create `path` and write its header with a zero record count.
 *
 * @param label  a short human name for the arm, e.g. "TT/qwen35moe" or "CPU"
 * @return       0 on success, -1 with a message on stderr
 *
 * Thread-safe: no   Reentrant: no   Signal-safe: no
 */
int parity_dump_open_write(const char *path, uint32_t n_vocab, uint32_t arch,
                           int framing_is_chat, int seam_enabled, const char *label,
                           parity_dump_writer *out);

/**
 * Append one scored position: its identity triple and the whole fp32 logit row.
 *
 * @param row  `n_vocab` floats, exactly as the forward returned them
 * @return     0 on success, -1 on a short write
 *
 * Thread-safe: no   Reentrant: no   Signal-safe: no
 */
int parity_dump_write_row(parity_dump_writer *w, int document, int position, int next_token,
                          const float *row);

/**
 * Patch the record count into the header and close.
 *
 * A dump not closed this way reads as EMPTY and is refused by the reader — which is the
 * intended behaviour for a pass that died half way, because a truncated arm paired against a
 * whole oracle would score a real number over the wrong subset.
 *
 * @return 0 on success, -1 otherwise
 */
int parity_dump_close_write(parity_dump_writer *w);

/**
 * Open `path` for reading and validate its magic, version and record count.
 *
 * @return 0 on success, -1 with a message on stderr
 */
int parity_dump_open_read(const char *path, parity_dump_reader *out);

/**
 * Read the next record and its row.
 *
 * @param row  caller's buffer of at least `header.n_vocab` floats
 * @return     1 when a record was read, 0 at end of the recorded count, -1 on a short read
 */
int parity_dump_read_row(parity_dump_reader *r, parity_dump_record *rec, float *row);

/** Close a reader. Safe on a zeroed struct. */
void parity_dump_close_read(parity_dump_reader *r);

/**
 * Do these two dumps describe the same scored stream?
 *
 * Checks the vocabulary, the architecture, the framing and the record count — everything a
 * pairing depends on that is not per-record. The per-record triple is checked as the rows are
 * read, because a mismatch there names the position it happened at.
 *
 * @return 0 when they may be paired, -1 with the reason on stderr
 */
int parity_dump_headers_agree(const parity_dump_header *arm, const parity_dump_header *oracle);

#endif /* PN_TESTS_PARITY_LOGIT_DUMP_H */
