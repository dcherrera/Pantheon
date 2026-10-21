/* SPDX-License-Identifier: AGPL-3.0-or-later */
/* Copyright (C) 2026 David (Corey) Herrera */
/**
 * @file parity_logit_dump.c
 * @brief Implementation of the two-pass parity dump declared in parity_logit_dump.h.
 *
 * @status Working
 * @issues None
 * @todo None
 */

#include "parity_logit_dump.h"

#include <stddef.h>
#include <stdlib.h>
#include <string.h>

/** The writer's own top-1 over the row it is about to write, so a reader can prove it read
 *  back the row it thinks it did rather than one record's worth of drift. */
static int32_t parity_dump_argmax(const float *row, uint32_t n)
{
    int32_t best = 0;
    uint32_t i;

    for (i = 1u; i < n; i++) {
        if (row[i] > row[best])
            best = (int32_t)i;
    }
    return best;
}

int parity_dump_open_write(const char *path, uint32_t n_vocab, uint32_t arch,
                           int framing_is_chat, int seam_enabled, const char *label,
                           parity_dump_writer *out)
{
    parity_dump_header header;

    if (!path || !out || n_vocab == 0u)
        return -1;
    memset(out, 0, sizeof(*out));
    memset(&header, 0, sizeof(header));
    header.magic = PARITY_DUMP_MAGIC;
    header.version = PARITY_DUMP_VERSION;
    header.n_vocab = n_vocab;
    header.arch = arch;
    header.framing_is_chat = framing_is_chat ? 1u : 0u;
    header.seam_enabled = seam_enabled ? 1u : 0u;
    header.record_count = 0u;
    snprintf(header.label, sizeof(header.label), "%s", label ? label : "?");

    out->file = fopen(path, "wb");
    if (!out->file) {
        fprintf(stderr, "parity_dump: cannot create %s\n", path);
        return -1;
    }
    if (fwrite(&header, sizeof(header), 1, out->file) != 1) {
        fprintf(stderr, "parity_dump: header write failed on %s\n", path);
        fclose(out->file);
        out->file = NULL;
        return -1;
    }
    out->n_vocab = n_vocab;
    return 0;
}

int parity_dump_write_row(parity_dump_writer *w, int document, int position, int next_token,
                          const float *row)
{
    parity_dump_record record;

    if (!w || !w->file || !row)
        return -1;
    record.document = document;
    record.position = position;
    record.next_token = next_token;
    record.argmax = parity_dump_argmax(row, w->n_vocab);
    if (fwrite(&record, sizeof(record), 1, w->file) != 1 ||
        fwrite(row, sizeof(*row), w->n_vocab, w->file) != (size_t)w->n_vocab) {
        fprintf(stderr, "parity_dump: short write at document %d position %d\n", document,
                position);
        return -1;
    }
    w->records++;
    return 0;
}

int parity_dump_close_write(parity_dump_writer *w)
{
    uint64_t count;
    int rc = 0;

    if (!w || !w->file)
        return -1;
    count = w->records;
    /* Seek to the field by `offsetof` rather than rewriting the whole header: a close that
     * only knows one number has no business restating the label, the vocabulary or the
     * framing, and a second copy of them is a second chance to write them differently. */
    if (fseek(w->file, (long)offsetof(parity_dump_header, record_count), SEEK_SET) != 0 ||
        fwrite(&count, sizeof(count), 1, w->file) != 1) {
        fprintf(stderr, "parity_dump: could not patch the record count — this dump will read "
                        "as empty, which is the intended refusal\n");
        rc = -1;
    }
    if (fclose(w->file) != 0)
        rc = -1;
    w->file = NULL;
    return rc;
}

int parity_dump_open_read(const char *path, parity_dump_reader *out)
{
    if (!path || !out)
        return -1;
    memset(out, 0, sizeof(*out));
    out->file = fopen(path, "rb");
    if (!out->file) {
        fprintf(stderr, "parity_dump: cannot open %s\n", path);
        return -1;
    }
    if (fread(&out->header, sizeof(out->header), 1, out->file) != 1) {
        fprintf(stderr, "parity_dump: %s is shorter than one header\n", path);
        goto fail;
    }
    if (out->header.magic != PARITY_DUMP_MAGIC) {
        fprintf(stderr, "parity_dump: %s is not a parity dump (magic %08x)\n", path,
                out->header.magic);
        goto fail;
    }
    if (out->header.version != PARITY_DUMP_VERSION) {
        fprintf(stderr, "parity_dump: %s is format version %u and this build reads %u\n", path,
                out->header.version, PARITY_DUMP_VERSION);
        goto fail;
    }
    if (out->header.record_count == 0u) {
        fprintf(stderr, "parity_dump: %s records ZERO positions. Either the pass that wrote it "
                        "died before closing, or it scored nothing — refusing either way.\n",
                path);
        goto fail;
    }
    return 0;

fail:
    fclose(out->file);
    out->file = NULL;
    return -1;
}

int parity_dump_read_row(parity_dump_reader *r, parity_dump_record *rec, float *row)
{
    if (!r || !r->file || !rec || !row)
        return -1;
    if (r->read_so_far >= r->header.record_count)
        return 0;
    if (fread(rec, sizeof(*rec), 1, r->file) != 1 ||
        fread(row, sizeof(*row), r->header.n_vocab, r->file) != (size_t)r->header.n_vocab) {
        fprintf(stderr, "parity_dump: short read at record %llu of %llu\n",
                (unsigned long long)r->read_so_far, (unsigned long long)r->header.record_count);
        return -1;
    }
    r->read_so_far++;
    return 1;
}

void parity_dump_close_read(parity_dump_reader *r)
{
    if (r && r->file) {
        fclose(r->file);
        r->file = NULL;
    }
}

int parity_dump_headers_agree(const parity_dump_header *arm, const parity_dump_header *oracle)
{
    if (!arm || !oracle)
        return -1;
    if (arm->n_vocab != oracle->n_vocab) {
        fprintf(stderr, "parity_dump: the arm's vocabulary is %u and the oracle's is %u — these "
                        "are not the same model\n", arm->n_vocab, oracle->n_vocab);
        return -1;
    }
    if (arm->arch != oracle->arch) {
        fprintf(stderr, "parity_dump: the arm's architecture is %u and the oracle's is %u\n",
                arm->arch, oracle->arch);
        return -1;
    }
    if (arm->framing_is_chat != oracle->framing_is_chat) {
        fprintf(stderr, "parity_dump: one pass was %s-framed and the other %s-framed. The "
                        "crossed control is a DIFFERENT corpus and must not pair against this "
                        "one.\n", arm->framing_is_chat ? "chat" : "plain",
                oracle->framing_is_chat ? "chat" : "plain");
        return -1;
    }
    if (arm->record_count != oracle->record_count) {
        fprintf(stderr, "parity_dump: the arm scored %llu positions and the oracle %llu. A "
                        "truncated arm paired against a whole oracle would report a real "
                        "number over the wrong subset.\n",
                (unsigned long long)arm->record_count,
                (unsigned long long)oracle->record_count);
        return -1;
    }
    return 0;
}
