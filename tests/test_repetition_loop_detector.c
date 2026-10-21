/* SPDX-License-Identifier: AGPL-3.0-or-later */
/* Copyright (C) 2026 David (Corey) Herrera */
/*
 * Unit tests for the decode-time repetition-cycle detector
 * (src/serve/repetition_loop_detector.c).
 *
 * The failure being guarded: saber-orch (Qwen3.6-35B-A3B, Q3_K_M) at ~40k
 * context stopped producing new text and emitted one ~470-byte paragraph 31
 * times, generating the whole 14,535-byte "answer" to completion and reporting
 * it as a normal finish.
 *
 * THE NEGATIVE TESTS ARE THE POINT.  Truncating a legitimate long code block is
 * a worse failure than the collapse it prevents, because it corrupts good output
 * silently.  So most of this file feeds the detector output that legitimately
 * repeats — a zero-filled byte table, `return NULL;` a hundred times, a JSON
 * array of similar objects, a markdown table, a log excerpt with one line 300
 * times, ASCII art, an enumerated list of near-identical steps — and asserts it
 * stays quiet.  Each of those also prints how close it came to the firing bar,
 * so the margin is a measured number and not an opinion.
 *
 * Token stand-in.  The detector consumes token IDS, so the corpora here are run
 * through a deliberately DENSE word/punctuation splitter (≤4 chars per piece,
 * split at character-class boundaries) rather than a real BPE.  Denser
 * tokenization means every legitimate repeating block gets a LONGER period in
 * tokens than a real tokenizer would give it, which pushes it closer to the
 * min-period bar — the negative tests are therefore stricter here than in
 * production, not laxer.
 *
 * The detector depends only on libc, so this links standalone (no engine/CUDA).
 */

#include "lib/test.h"
#include "../src/serve/repetition_loop_detector.h"

#include <stdarg.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

/* ============================================================
 * Growable text buffer — the corpora are built programmatically so a "200
 * identical rows" case is 200 rows and not an abbreviation of one.
 * ============================================================ */

typedef struct { char *buf; size_t len, cap; } strbuf;

static void sb_init(strbuf *s)
{
    s->cap = 4096;
    s->len = 0;
    s->buf = malloc(s->cap);
    if (s->buf) s->buf[0] = '\0';
}

static void sb_free(strbuf *s) { free(s->buf); s->buf = NULL; s->len = s->cap = 0; }

static void sb_addf(strbuf *s, const char *fmt, ...)
{
    if (!s->buf) return;
    for (;;) {
        va_list ap;
        va_start(ap, fmt);
        int n = vsnprintf(s->buf + s->len, s->cap - s->len, fmt, ap);
        va_end(ap);
        if (n < 0) return;
        if ((size_t)n < s->cap - s->len) { s->len += (size_t)n; return; }
        size_t want = s->cap * 2;
        while (want < s->len + (size_t)n + 1) want *= 2;
        char *grown = realloc(s->buf, want);
        if (!grown) return;
        s->buf = grown;
        s->cap = want;
    }
}

/* ============================================================
 * Dense token stand-in (see the file header for why it is dense on purpose).
 * ============================================================ */

static int char_class(unsigned char c)
{
    if (c == ' ' || c == '\t' || c == '\n' || c == '\r') return 0;
    if ((c >= 'a' && c <= 'z') || (c >= 'A' && c <= 'Z') ||
        (c >= '0' && c <= '9') || c == '_') return 1;
    return 2;
}

static uint32_t piece_id(const char *p, size_t n)
{
    uint32_t h = 2166136261u;
    for (size_t i = 0; i < n; i++) {
        h ^= (unsigned char)p[i];
        h *= 16777619u;
    }
    return h & 0x7fffffffu;   /* stay inside a plausible vocab-ish range */
}

/* Split `text` into ≤4-char pieces that never span a character-class boundary,
 * and hash each piece to an id.  Returns the count; the array is malloc'd. */
static uint32_t *tokenize_dense(const char *text, int *n_out)
{
    size_t len = strlen(text);
    uint32_t *out = malloc((len + 1) * sizeof(*out));
    int n = 0;
    size_t i = 0;
    if (!out) { *n_out = 0; return NULL; }
    while (i < len) {
        int cls = char_class((unsigned char)text[i]);
        size_t start = i;
        while (i < len && char_class((unsigned char)text[i]) == cls &&
               (i - start) < 4)
            i++;
        out[n++] = piece_id(text + start, i - start);
    }
    *n_out = n;
    return out;
}

/* ============================================================
 * Driver: feed a whole stream, report the trip point and the CLOSEST any
 * stretch of it came to the firing bar (as a percentage).  That percentage is
 * the false-positive margin for a legitimate corpus.
 * ============================================================ */

static int feed_stream(rld_detector *rd, const uint32_t *t, int n,
                       double *closest_pct)
{
    int trip = -1;
    double best = 0.0;
    for (int i = 0; i < n; i++) {
        bool hit = rld_feed(rd, t[i]);
        if (rd->locked_period > 0) {
            double need = (double)rd->locked_period * (rd->cfg.min_repeats - 1);
            double pct  = need > 0.0 ? (double)rd->verified_run / need : 0.0;
            if (pct > best) best = pct;
        }
        if (hit && trip < 0) trip = i;
    }
    if (closest_pct) *closest_pct = best * 100.0;
    return trip;
}

/* Run one legitimate corpus through a default-configured detector and assert it
 * does not fire, printing the margin so the number is on the record. */
static int assert_legit(const char *label, const char *text, double *pct_out)
{
    rld_detector rd;
    int n = 0;
    uint32_t *toks = tokenize_dense(text, &n);
    double pct = 0.0;
    int trip;

    if (!toks) return -1;
    if (!rld_init(&rd, NULL)) { free(toks); return -1; }
    trip = feed_stream(&rd, toks, n, &pct);
    printf("      %-28s %6d tokens, closest to firing: %5.1f%% of the bar\n",
           label, n, pct);
    rld_free(&rd);
    free(toks);
    if (pct_out) *pct_out = pct;
    return trip;
}

/* ============================================================
 * The measured failure: one paragraph, cycled.  (The text is the run-11
 * transcript's repeated paragraph.)
 * ============================================================ */

static const char *COLLAPSE_PARAGRAPH =
    "with foreign key relationships, sessions with metadata fields like title "
    "and creation timestamps, messages that reference those sessions through a "
    "foreign key constraint and an index on session_id to optimize lookups.\n"
    "Then there's the file_snapshots table structure starting to take shape as "
    "well.\n";

static void repeat_into(strbuf *s, const char *unit, int times)
{
    for (int i = 0; i < times; i++)
        sb_addf(s, "%s", unit);
}

/* ============================================================
 * POSITIVE — the collapse is caught.
 * ============================================================ */

static void test_collapse_paragraph_trips(void) {
    strbuf s; sb_init(&s);
    repeat_into(&s, COLLAPSE_PARAGRAPH, 31);

    int n_unit = 0, n = 0;
    uint32_t *unit = tokenize_dense(COLLAPSE_PARAGRAPH, &n_unit);
    uint32_t *toks = tokenize_dense(s.buf, &n);
    ASSERT_NOT_NULL(unit);
    ASSERT_NOT_NULL(toks);

    rld_detector rd;
    ASSERT_TRUE(rld_init(&rd, NULL));
    int trip = feed_stream(&rd, toks, n, NULL);

    printf("      cycle unit = %d tokens; tripped at token %d of %d\n",
           n_unit, trip, n);
    ASSERT_TRUE(trip >= 0);
    ASSERT_TRUE(rd.tripped);
    ASSERT_INT_EQ(rd.cycle_tokens, n_unit);
    ASSERT_INT_EQ(rd.repeats, 12);
    /* Fires at the end of the 12th copy — 19 of the 31 copies never generated. */
    ASSERT_INT_EQ(trip, 12 * n_unit - 1);

    rld_free(&rd);
    free(unit); free(toks); sb_free(&s);
}

/* The collapse rarely starts at token 0 — the model writes real prose first.
 * The detector is position-agnostic: the trip must land the same number of
 * tokens after the cycle STARTS, whatever came before it. */
static void test_collapse_after_real_prose_trips(void) {
    strbuf s; sb_init(&s);
    sb_addf(&s, "%s",
        "Looking at the schema more carefully, the CREATE TABLE IF NOT EXISTS "
        "will not add the new column to a database that predates it, so the "
        "CREATE INDEX on that column fails and sqlite3_exec returns an error "
        "before the store is ever initialized. The fix is to detect the old "
        "shape and migrate it. ");
    int n_pre = 0;
    uint32_t *pre = tokenize_dense(s.buf, &n_pre);
    ASSERT_NOT_NULL(pre);

    repeat_into(&s, COLLAPSE_PARAGRAPH, 20);
    int n_unit = 0, n = 0;
    uint32_t *unit = tokenize_dense(COLLAPSE_PARAGRAPH, &n_unit);
    uint32_t *toks = tokenize_dense(s.buf, &n);
    ASSERT_NOT_NULL(unit);
    ASSERT_NOT_NULL(toks);

    rld_detector rd;
    ASSERT_TRUE(rld_init(&rd, NULL));
    int trip = feed_stream(&rd, toks, n, NULL);
    ASSERT_TRUE(trip >= 0);
    ASSERT_INT_EQ(rd.cycle_tokens, n_unit);
    ASSERT_INT_EQ(rd.repeats, 12);
    ASSERT_INT_EQ(trip, n_pre + 12 * n_unit - 1);

    rld_free(&rd);
    free(pre); free(unit); free(toks); sb_free(&s);
}

/* Eleven copies is under the bar and must NOT fire — the twelfth is what fires.
 * This is the exact boundary the whole false-positive argument rests on. */
static void test_eleven_copies_do_not_trip(void) {
    strbuf s; sb_init(&s);
    repeat_into(&s, COLLAPSE_PARAGRAPH, 11);
    int n = 0;
    uint32_t *toks = tokenize_dense(s.buf, &n);
    ASSERT_NOT_NULL(toks);

    rld_detector rd;
    ASSERT_TRUE(rld_init(&rd, NULL));
    double pct = 0.0;
    int trip = feed_stream(&rd, toks, n, &pct);
    printf("      11 copies reached %.1f%% of the bar (12 fires)\n", pct);
    ASSERT_INT_EQ(trip, -1);
    ASSERT_FALSE(rd.tripped);

    rld_free(&rd);
    free(toks); sb_free(&s);
}

/* A cycle that keeps getting interrupted by novel content is not a collapse —
 * the model is still saying new things.  Four copies, a new sentence, four
 * more, forever: never fires however long it runs. */
static void test_cycle_broken_by_novel_content_never_trips(void) {
    strbuf s; sb_init(&s);
    for (int round = 0; round < 12; round++) {
        repeat_into(&s, COLLAPSE_PARAGRAPH, 4);
        sb_addf(&s, "Now round %d introduces a genuinely different observation "
                    "about the migration path and the index rebuild.\n", round);
    }
    int n = 0;
    uint32_t *toks = tokenize_dense(s.buf, &n);
    ASSERT_NOT_NULL(toks);

    rld_detector rd;
    ASSERT_TRUE(rld_init(&rd, NULL));
    double pct = 0.0;
    int trip = feed_stream(&rd, toks, n, &pct);
    printf("      %d tokens, closest to firing: %.1f%%\n", n, pct);
    ASSERT_INT_EQ(trip, -1);

    rld_free(&rd);
    free(toks); sb_free(&s);
}

/* ============================================================
 * NEGATIVE — legitimate output that repeats.  None of these may fire.
 * ============================================================ */

/* A long C function whose body really does contain the same line over and over,
 * plus a zero-filled lookup table.  This is the shape the guard must never
 * truncate: the model's job is largely writing code like this. */
static void test_c_source_with_identical_lines(void) {
    strbuf s; sb_init(&s);
    sb_addf(&s,
        "static const uint8_t crc_seed[256] = {\n");
    for (int i = 0; i < 32; i++)
        sb_addf(&s, "    0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,\n");
    sb_addf(&s, "};\n\n"
        "tc_tool_call *tc_parse_tool_calls(const char *raw, int *n_out)\n"
        "{\n");
    for (int i = 0; i < 40; i++) {
        sb_addf(&s, "    if (!raw)\n");
        sb_addf(&s, "        return NULL;\n");
    }
    for (int i = 0; i < 60; i++)
        sb_addf(&s, "        return NULL;\n");
    sb_addf(&s, "    return calls;\n}\n");

    double pct = 0.0;
    int trip = assert_legit("C source + zero table", s.buf, &pct);
    ASSERT_INT_EQ(trip, -1);
    sb_free(&s);
}

/* A pure exact cycle with a SHORT fundamental period: 200 byte-identical table
 * rows.  The detector locks onto the fundamental period (one row), not a
 * multiple of it, so this is rejected on the period bar and never accumulates.
 * If the smallest-period rule ever regresses, this is the test that catches it. */
static void test_zero_filled_table_two_hundred_rows(void) {
    strbuf s; sb_init(&s);
    sb_addf(&s, "static const uint32_t pad[1600] = {\n");
    for (int i = 0; i < 200; i++)
        sb_addf(&s, "    0x00000000, 0x00000000, 0x00000000, 0x00000000,\n");
    sb_addf(&s, "};\n");

    double pct = 0.0;
    int trip = assert_legit("200 identical table rows", s.buf, &pct);
    ASSERT_INT_EQ(trip, -1);
    sb_free(&s);
}

/* A JSON array of 80 objects with the same SHAPE and different values — the
 * everyday "dump the rows" answer. */
static void test_json_array_of_similar_objects(void) {
    strbuf s; sb_init(&s);
    sb_addf(&s, "[\n");
    for (int i = 0; i < 80; i++)
        sb_addf(&s,
            "  {\n    \"id\": %d,\n    \"name\": \"session_%04d\",\n"
            "    \"created_at\": \"2026-08-%02dT%02d:15:00Z\",\n"
            "    \"message_count\": %d,\n    \"archived\": %s\n  }%s\n",
            i, i, (i % 28) + 1, i % 24, i * 7, (i & 1) ? "true" : "false",
            i == 79 ? "" : ",");
    sb_addf(&s, "]\n");

    double pct = 0.0;
    int trip = assert_legit("JSON array, 80 objects", s.buf, &pct);
    ASSERT_INT_EQ(trip, -1);
    sb_free(&s);
}

/* A markdown table: separator row, then 60 data rows. */
static void test_markdown_table(void) {
    strbuf s; sb_init(&s);
    sb_addf(&s, "| id | table | column | type | nullable | indexed |\n");
    sb_addf(&s, "| --- | --- | --- | --- | --- | --- |\n");
    for (int i = 0; i < 60; i++)
        sb_addf(&s, "| %d | points | scope_%02d | TEXT | no | yes |\n", i, i);

    double pct = 0.0;
    int trip = assert_legit("markdown table, 60 rows", s.buf, &pct);
    ASSERT_INT_EQ(trip, -1);
    sb_free(&s);
}

/* A log excerpt where ONE line recurs 300 times — a real thing a model is asked
 * to reproduce, and a pure exact cycle at a short fundamental period. */
static void test_log_excerpt_same_line_three_hundred_times(void) {
    strbuf s; sb_init(&s);
    sb_addf(&s, "The relevant part of the log is:\n\n");
    for (int i = 0; i < 300; i++)
        sb_addf(&s, "context_store: schema failed: no such column: scope\n");
    sb_addf(&s, "\nThat repeats until the process gives up.\n");

    double pct = 0.0;
    int trip = assert_legit("log line x300", s.buf, &pct);
    ASSERT_INT_EQ(trip, -1);
    sb_free(&s);
}

/* Box-drawing ASCII art: an identical border band repeated 50 times. */
static void test_ascii_art_border_band(void) {
    strbuf s; sb_init(&s);
    sb_addf(&s, "+--------------------------------------------------+\n");
    for (int i = 0; i < 50; i++)
        sb_addf(&s, "|                                                  |\n");
    sb_addf(&s, "+--------------------------------------------------+\n");

    double pct = 0.0;
    int trip = assert_legit("ASCII art border band", s.buf, &pct);
    ASSERT_INT_EQ(trip, -1);
    sb_free(&s);
}

/* SQL schema: 70 column definitions, same shape, different names. */
static void test_sql_schema_column_lines(void) {
    strbuf s; sb_init(&s);
    sb_addf(&s, "CREATE TABLE IF NOT EXISTS points (\n");
    for (int i = 0; i < 70; i++)
        sb_addf(&s, "    column_%03d TEXT NOT NULL DEFAULT '',\n", i);
    sb_addf(&s, "    PRIMARY KEY (column_000)\n);\n");

    double pct = 0.0;
    int trip = assert_legit("SQL schema, 70 columns", s.buf, &pct);
    ASSERT_INT_EQ(trip, -1);
    sb_free(&s);
}

/* An enumerated procedure: 40 near-identical steps differing only by number.
 * Structurally this is the closest legitimate output gets to a paragraph loop. */
static void test_enumerated_near_identical_steps(void) {
    strbuf s; sb_init(&s);
    for (int i = 1; i <= 40; i++)
        sb_addf(&s,
            "%d. Open the file at src/context/context_store.c and locate the "
            "schema string, then confirm that the column referenced by index "
            "number %d is present before the CREATE INDEX statement runs.\n",
            i, i);

    double pct = 0.0;
    int trip = assert_legit("enumerated steps x40", s.buf, &pct);
    ASSERT_INT_EQ(trip, -1);
    sb_free(&s);
}

/* THE WORST CASE.  A byte-identical multi-line block well over the period bar,
 * repeated 11 times back to back with nothing in between — legitimate output
 * has no reason to do this, and this is as close as anything gets to firing.
 * The margin it leaves is the number to quote when the thresholds are argued. */
static void test_worst_case_identical_block_eleven_times(void) {
    const char *block =
        "  {\n"
        "    \"kind\": \"function\",\n"
        "    \"name\": \"read_file\",\n"
        "    \"description\": \"Read a UTF-8 text file from the workspace\",\n"
        "    \"parameters\": { \"path\": { \"type\": \"string\" } },\n"
        "    \"returns\": { \"type\": \"string\" },\n"
        "    \"idempotent\": true\n"
        "  },\n";
    strbuf s; sb_init(&s);
    repeat_into(&s, block, 11);

    int n_unit = 0;
    uint32_t *unit = tokenize_dense(block, &n_unit);
    ASSERT_NOT_NULL(unit);
    ASSERT_TRUE(n_unit >= 48);   /* it really is over the period bar */

    double pct = 0.0;
    int trip = assert_legit("identical block x11", s.buf, &pct);
    ASSERT_INT_EQ(trip, -1);
    /* It locked and climbed — this is a genuine near-miss, not a corpus the
     * detector ignored for some unrelated reason. */
    ASSERT_TRUE(pct > 80.0);
    ASSERT_TRUE(pct < 100.0);
    free(unit);
    sb_free(&s);
}

/* Ordinary varied prose must never lock at all. */
static void test_varied_prose_never_locks(void) {
    strbuf s; sb_init(&s);
    for (int i = 0; i < 120; i++)
        sb_addf(&s,
            "Paragraph %d considers a different part of the store: the %s path "
            "and how it interacts with the %s boundary when the session id is "
            "reused across two conversations at once.\n",
            i, (i & 1) ? "append-resume" : "full-prefill",
            (i % 3) ? "system-prefix" : "context-cap");

    double pct = 0.0;
    int trip = assert_legit("varied prose", s.buf, &pct);
    ASSERT_INT_EQ(trip, -1);
    sb_free(&s);
}

/* ============================================================
 * Configuration
 * ============================================================ */

static void test_defaults_are_the_documented_ones(void) {
    rld_config cfg = {0};
    rld_default_config(&cfg);
    ASSERT_INT_EQ(cfg.min_period, 48);
    ASSERT_INT_EQ(cfg.max_period, 512);
    ASSERT_INT_EQ(cfg.min_repeats, 12);
    ASSERT_TRUE(rld_enabled());
}

/* A cycle LONGER than max_period is not tracked (the ring cannot see that far
 * back) — a documented limit, asserted so it cannot become a surprise. */
static void test_cycle_longer_than_max_period_is_not_tracked(void) {
    rld_config cfg = { .min_period = 8, .max_period = 32, .min_repeats = 4 };
    rld_detector rd;
    ASSERT_TRUE(rld_init(&rd, &cfg));

    uint32_t unit[200];
    for (int i = 0; i < 200; i++) unit[i] = (uint32_t)(1000 + i);
    int trip = -1;
    for (int r = 0; r < 10 && trip < 0; r++)
        for (int i = 0; i < 200; i++)
            if (rld_feed(&rd, unit[i])) { trip = 1; break; }
    ASSERT_INT_EQ(trip, -1);
    rld_free(&rd);
}

/* A NULL / inert detector is safe to feed and free: an allocation failure must
 * never be able to fail a completion. */
static void test_inert_detector_is_safe(void) {
    rld_detector rd;
    memset(&rd, 0, sizeof(rd));
    for (int i = 0; i < 5000; i++)
        ASSERT_FALSE(rld_feed(&rd, (uint32_t)(i % 3)));
    rld_free(&rd);
    rld_free(&rd);          /* idempotent */
    ASSERT_FALSE(rld_feed(NULL, 7));
    ASSERT_FALSE(rld_init(NULL, NULL));
}

/* Raising min_repeats really does hold fire — the operator knob works. */
static void test_min_repeats_knob_holds_fire(void) {
    rld_config cfg = { .min_period = 48, .max_period = 512, .min_repeats = 64 };
    strbuf s; sb_init(&s);
    repeat_into(&s, COLLAPSE_PARAGRAPH, 31);
    int n = 0;
    uint32_t *toks = tokenize_dense(s.buf, &n);
    ASSERT_NOT_NULL(toks);

    rld_detector rd;
    ASSERT_TRUE(rld_init(&rd, &cfg));
    ASSERT_INT_EQ(feed_stream(&rd, toks, n, NULL), -1);
    rld_free(&rd);
    free(toks); sb_free(&s);
}

int main(void) {
    printf("\n=== repetition_loop_detector (decode-time cycle guard) ===\n\n");

    printf("  -- positive: the measured collapse --\n");
    RUN_TEST(collapse_paragraph_trips);
    RUN_TEST(collapse_after_real_prose_trips);
    RUN_TEST(eleven_copies_do_not_trip);
    RUN_TEST(cycle_broken_by_novel_content_never_trips);

    printf("\n  -- negative: legitimate output that repeats --\n");
    RUN_TEST(c_source_with_identical_lines);
    RUN_TEST(zero_filled_table_two_hundred_rows);
    RUN_TEST(json_array_of_similar_objects);
    RUN_TEST(markdown_table);
    RUN_TEST(log_excerpt_same_line_three_hundred_times);
    RUN_TEST(ascii_art_border_band);
    RUN_TEST(sql_schema_column_lines);
    RUN_TEST(enumerated_near_identical_steps);
    RUN_TEST(worst_case_identical_block_eleven_times);
    RUN_TEST(varied_prose_never_locks);

    printf("\n  -- configuration --\n");
    RUN_TEST(defaults_are_the_documented_ones);
    RUN_TEST(cycle_longer_than_max_period_is_not_tracked);
    RUN_TEST(inert_detector_is_safe);
    RUN_TEST(min_repeats_knob_holds_fire);

    TEST_REPORT();
    return __ft_failed == 0 ? 0 : 1;
}
