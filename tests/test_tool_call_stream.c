/* SPDX-License-Identifier: AGPL-3.0-or-later */
/* Copyright (C) 2026 David (Corey) Herrera */
/*
 * Unit tests for incremental tool-call streaming:
 *   - json_canonical_stream.c: fed raw JSON in any split, its output is always a
 *     prefix of — and when complete, byte-identical to — jr_to_json(jr_parse(x)),
 *     and it rejects exactly what jr_parse rejects;
 *   - tool_call_stream.c: an ORNITH_JSON call opens on the wire as soon as its
 *     name is decided, its arguments stream while the model writes them, and
 *     the fragments concatenate to exactly the arguments the end-of-turn parse
 *     returns (repair, abandoned and multi-call cases included);
 *   - serve_emitter.c's tool tap: the streamer sees exactly the bytes that end
 *     up in tool_raw.
 *
 * The streaming-before-the-end assertions (early_*) are the ones that fail on
 * the old behaviour, where nothing of a call left the serve until the turn
 * ended.  libc-only; no engine.
 */

#include "lib/test.h"

#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "../src/serve/json_canonical_stream.h"
#include "../src/serve/json_reader.h"
#include "../src/serve/json_writer.h"
#include "../src/serve/serve_emitter.h"
#include "../src/serve/tool_call_stream.h"
#include "../src/serve/tool_calling.h"

/* ============================================================
 * json_canonical_stream vs jr_parse + jr_to_json
 * ============================================================ */

/* The DOM's canonical form of `doc`, or NULL when jr_parse rejects it. */
static char *dom_canonical(const char *doc)
{
    jr_value *v = jr_parse(doc);
    if (!v) return NULL;
    json_writer jw;
    jw_init(&jw);
    jr_to_json(v, &jw);
    jr_free(v);
    return jw.data;
}

/* Feed `doc` in chunks of `chunk` bytes (0 = random sizes from `seed`).
 * Returns 0 when jcs matches the DOM: same verdict, identical bytes when
 * accepted, and every intermediate output a prefix of the final one. */
static int jcs_matches_dom(const char *doc, size_t chunk, unsigned seed)
{
    char *want = dom_canonical(doc);
    jcs_parser p;
    if (!jcs_init(&p, NULL)) { free(want); return 1; }
    size_t len = strlen(doc), at = 0;
    int bad = 0;
    while (at < len && p.status == JCS_RUNNING) {
        size_t n = chunk;
        if (!n) { seed = seed * 1103515245u + 12345u; n = 1 + (seed >> 16) % 7; }
        if (n > len - at) n = len - at;
        jcs_feed(&p, doc + at, n);
        at += n;
        if (want && (p.out.len > strlen(want) ||
                     memcmp(p.out.data, want, p.out.len) != 0)) { bad = 2; break; }
    }
    if (!bad) {
        if (want) bad = (p.status == JCS_DONE && strcmp(p.out.data, want) == 0) ? 0 : 3;
        else      bad = (p.status == JCS_FAILED) ? 0 : 4;
    }
    if (bad)
        printf("    jcs mismatch (%d) chunk=%zu doc=<<<%s>>> want=<<<%s>>> got=<<<%s>>>\n",
               bad, chunk, doc, want ? want : "(rejected)", p.out.data);
    jcs_free(&p);
    free(want);
    return bad;
}

static const char *const ACCEPTED_DOCS[] = {
    "{}",
    "{ }",
    "[]",
    "{\"a\":1}",
    " {\r\n\t\"a\" :\t[ 1 , 2.50 ,-3e2, 1E+2, 0.1, -0.0, 12345678901234567890 ] } ",
    "{\"n\":[true,false,null],\"o\":{\"x\":{}},\"e\":[]}",
    "{\"s\":\"plain text with spaces\"}",
    "{\"s\":\"esc \\\" \\\\ \\/ \\b \\f \\n \\r \\t end\"}",
    "{\"s\":\"\\u00e9\\u4e2d\\ud83d\\ude00 ok\"}",
    "{\"s\":\"lone high \\ud83d then text\"}",
    "{\"s\":\"high then escape \\ud83d\\n\"}",
    "{\"s\":\"high then quote \\ud83d\\\"\"}",
    "{\"s\":\"high at end \\ud83d\"}",
    "{\"s\":\"bad low \\ud83d\\u0041\"}",
    "{\"s\":\"nul \\u0000 cut here\",\"t\":\"after\"}",
    "{\"k\\u0000ey\":1}",
    "{\"s\":\"raw\ttab and \x01 ctl\"}",
    "{\"s\":\"raw utf8 \xc3\xa9 \xe4\xb8\xad\"}",
    "{\"num\":+1,\"hex\":0x1A,\"dot\":.5,\"inf\":1e400,\"Inf\":inf}",
    "{\"a\":{\"b\":{\"c\":[{\"d\":[[[]]]}]}}}",
    "{\"function\": \"write\", \"arguments\": {\"path\": \"a.py\", \"content\": \"def f():\\n    return {\\\"x\\\": [1, 2]}\\n\"}}",
    "{\"dup\":1,\"dup\":2}",
    "\"top string\"",
    "[1,[2,[3]]]",
};

static const char *const REJECTED_DOCS[] = {
    "{\"a\":1,}",
    "{\"a\" 1}",
    "[1,]",
    "[1 2]",
    "{\"a\":tru}",
    "{\"a\":truex}",
    "{\"a\":nan}",
    "{\"a\":\"\\q\"}",
    "{\"a\":\"\\u12g4\"}",
    "{\"a\":\"\\ud83d\\u12\"}",
    "{\"a\":1.5x}",
    "{\"a\":-}",
    "{a:1}",
    "{\"a\":\"x\" \"y\"}",
    "{\"a\":1}}",            /* the root closes, the trailing '}' is ignored */
    "{\"a\":}",
    "{,}",
    "{\"a\":[}",
    "{\"a\":1 \"b\":2}",
    "{\"a\"::1}",
};

static void test_jcs_matches_dom_fixed_docs(void)
{
    for (size_t i = 0; i < sizeof(ACCEPTED_DOCS) / sizeof(*ACCEPTED_DOCS); i++) {
        char *want = dom_canonical(ACCEPTED_DOCS[i]);
        ASSERT_NOT_NULL(want);              /* the corpus really is accepted */
        free(want);
        size_t len = strlen(ACCEPTED_DOCS[i]);
        for (size_t chunk = 1; chunk <= len; chunk++)
            ASSERT_INT_EQ(jcs_matches_dom(ACCEPTED_DOCS[i], chunk, 0), 0);
        ASSERT_INT_EQ(jcs_matches_dom(ACCEPTED_DOCS[i], 0, (unsigned)i), 0);
    }
}

/* "{\"a\":1}}" is accepted by jr_parse (it ignores trailing bytes) and by jcs
 * (DONE at the root's close) — kept in the list to pin that both agree. */
static void test_jcs_rejects_what_dom_rejects(void)
{
    for (size_t i = 0; i < sizeof(REJECTED_DOCS) / sizeof(*REJECTED_DOCS); i++) {
        for (size_t chunk = 1; chunk <= 3; chunk++)
            ASSERT_INT_EQ(jcs_matches_dom(REJECTED_DOCS[i], chunk, 0), 0);
        ASSERT_INT_EQ(jcs_matches_dom(REJECTED_DOCS[i], 0, (unsigned)i), 0);
    }
}

/* ---- generated documents ---- */

static unsigned rng_state = 12345;
static unsigned rng(void)
{
    rng_state = rng_state * 1103515245u + 12345u;
    return rng_state >> 16;
}

static void gen_ws(json_writer *jw)
{
    static const char ws[] = " \t\n\r";
    int n = (int)(rng() % 3);
    for (int i = 0; i < n; i++) jw_raw_n(jw, &ws[rng() % 4], 1);
}

static void gen_string(json_writer *jw)
{
    static const char *const pieces[] = {
        "a", "Z", " ", "0", "\\n", "\\t", "\\\"", "\\\\", "\\/", "\\b", "\\f",
        "\\r", "\\u00e9", "\\u4E2D", "\\ud83d\\ude00", "\\ud83d", "\\uDC00",
        "\\u0000", "\xc3\xa9", "\t", "{", "}", "[", ":", ",", "\\u001f",
    };
    jw_raw(jw, "\"");
    int n = (int)(rng() % 12);
    for (int i = 0; i < n; i++)
        jw_raw(jw, pieces[rng() % (sizeof(pieces) / sizeof(*pieces))]);
    jw_raw(jw, "\"");
}

static void gen_value(json_writer *jw, int depth)
{
    static const char *const scalars[] = {
        "0", "-1", "1.5", "2e3", "-0.25E-2", "123456789012345678901",
        "3.141592653589793", "true", "false", "null", "1e400", "+7",
    };
    unsigned kind = rng() % (depth > 3 ? 2 : 4);
    gen_ws(jw);
    if (kind == 0) {
        jw_raw(jw, scalars[rng() % (sizeof(scalars) / sizeof(*scalars))]);
    } else if (kind == 1) {
        gen_string(jw);
    } else if (kind == 2) {
        jw_raw(jw, "[");
        int n = (int)(rng() % 4);
        for (int i = 0; i < n; i++) {
            if (i) { gen_ws(jw); jw_raw(jw, ","); }
            gen_value(jw, depth + 1);
            gen_ws(jw);
        }
        if (!n) gen_ws(jw);
        jw_raw(jw, "]");
    } else {
        jw_raw(jw, "{");
        int n = (int)(rng() % 4);
        for (int i = 0; i < n; i++) {
            if (i) jw_raw(jw, ",");
            gen_ws(jw);
            gen_string(jw);
            gen_ws(jw);
            jw_raw(jw, ":");
            gen_value(jw, depth + 1);
            gen_ws(jw);
        }
        if (!n) gen_ws(jw);
        jw_raw(jw, "}");
    }
}

static void test_jcs_matches_dom_generated_docs(void)
{
    for (int i = 0; i < 3000; i++) {
        json_writer jw;
        jw_init(&jw);
        jw_raw(&jw, "{\"k\":");
        gen_value(&jw, 0);
        jw_raw(&jw, "}");
        ASSERT_INT_EQ(jcs_matches_dom(jw.data, 0, (unsigned)i), 0);
        ASSERT_INT_EQ(jcs_matches_dom(jw.data, 1, 0), 0);
        jw_free(&jw);
    }
}

/* ============================================================
 * tool_call_stream
 * ============================================================ */

#define REC_MAX_CALLS 8

/* What a client saw: per wire index, the id/name of the opening delta and the
 * concatenated argument fragments, plus how far the feed had got when each
 * event happened. */
typedef struct {
    int         n_begin;
    int         index[REC_MAX_CALLS];
    char        name[REC_MAX_CALLS][32];
    json_writer args[REC_MAX_CALLS];
    int         fragments[REC_MAX_CALLS];
    size_t      fed;                  /* bytes fed so far (set by the driver) */
    size_t      fed_at_begin[REC_MAX_CALLS];
    size_t      fed_at_first_fragment[REC_MAX_CALLS];
    int         progress;
} recorder;

static int rec_slot(recorder *r, int index)
{
    for (int i = 0; i < r->n_begin; i++)
        if (r->index[i] == index) return i;
    return -1;
}

static void rec_begin(void *ud, int index, const char *id, const char *name)
{
    recorder *r = ud;
    (void)id;
    if (r->n_begin >= REC_MAX_CALLS) return;
    int s = r->n_begin++;
    r->index[s] = index;
    snprintf(r->name[s], sizeof(r->name[s]), "%s", name);
    jw_init(&r->args[s]);
    r->fed_at_begin[s] = r->fed;
    r->fed_at_first_fragment[s] = (size_t)-1;
}

static void rec_args(void *ud, int index, const char *bytes, size_t n)
{
    recorder *r = ud;
    int s = rec_slot(r, index);
    if (s < 0) return;
    if (!r->fragments[s]) r->fed_at_first_fragment[s] = r->fed;
    r->fragments[s]++;
    jw_raw_n(&r->args[s], bytes, n);
}

static void rec_progress(void *ud) { ((recorder *)ud)->progress++; }

static void rec_free(recorder *r)
{
    for (int i = 0; i < r->n_begin; i++) jw_free(&r->args[i]);
}

static const char TOOLS_JSON[] =
    "[{\"type\":\"function\",\"function\":{\"name\":\"write\",\"parameters\":"
    "{\"type\":\"object\",\"properties\":{\"path\":{\"type\":\"string\"},"
    "\"content\":{\"type\":\"string\"}},\"required\":[\"path\",\"content\"]}}},"
    "{\"type\":\"function\",\"function\":{\"name\":\"read\",\"parameters\":"
    "{\"type\":\"object\",\"properties\":{\"path\":{\"type\":\"string\"}}}}}]";

/* Stream `text` through a streamer in `piece`-byte feeds, finish it against the
 * end-of-turn parse (with repair, as collect_tool_calls runs it), and leave the
 * wire record in `r`.  Returns the number of calls the parse produced. */
static int stream_text(recorder *r, const char *text, size_t piece,
                       tc_dialect dialect, tc_tool_call **calls_out)
{
    jr_value *tools = jr_parse(TOOLS_JSON);
    tcs_streamer ts;
    tcs_sink sink = { rec_begin, rec_args, rec_progress, r };
    memset(r, 0, sizeof(*r));
    tcs_init(&ts, dialect, tools, &sink);
    size_t len = strlen(text);
    for (size_t at = 0; at < len; at += piece) {
        size_t n = piece < len - at ? piece : len - at;
        r->fed = at + n;
        tcs_feed(&ts, text + at, n);
    }
    tc_tool_call *calls = NULL;
    int n = 0;
    tc_parse_repair repair = {0};
    tc_parse_tool_calls_diagnosed(text, tools, dialect, &calls, &n, NULL, &repair);
    tc_parse_repair_free(&repair);
    tcs_finish(&ts, calls, n);
    tcs_free(&ts);
    jr_free(tools);
    *calls_out = calls;
    return n;
}

/* The wire contract: the calls whose streamed arguments are valid JSON are
 * exactly the parsed calls, in order, byte for byte; every other wire call
 * carries arguments no client can parse. */
static int wire_matches_parse(const recorder *r, const tc_tool_call *calls, int n)
{
    int j = 0;
    for (int s = 0; s < r->n_begin; s++) {
        jr_value *v = jr_parse(r->args[s].data);
        bool valid = v && jr_kind(v) == JR_OBJECT;
        jr_free(v);
        if (!valid) continue;
        if (j >= n) { printf("    extra valid wire call %d\n", r->index[s]); return 1; }
        if (strcmp(r->name[s], calls[j].name) != 0 ||
            strcmp(r->args[s].data, calls[j].arguments) != 0) {
            printf("    wire call %d != parsed call %d\n      wire  <<<%s>>>\n"
                   "      parse <<<%s>>>\n", r->index[s], j, r->args[s].data,
                   calls[j].arguments);
            return 2;
        }
        j++;
    }
    if (j != n) { printf("    %d parsed calls, %d on the wire\n", n, j); return 3; }
    return 0;
}

static const char LONG_CALL[] =
    "```json\n"
    "{\n  \"function\": \"write\",\n  \"arguments\": {\n"
    "    \"path\": \"inventory.py\",\n"
    "    \"content\": \"\\\"\\\"\\\"Inventory.\\\"\\\"\\\"\\n\\nimport csv\\n\\n"
    "class Product:\\n    def __init__(self, sku, qty):\\n        self.sku = sku\\n"
    "        self.qty = qty\\n\\n    def restock(self, n):\\n"
    "        if n <= 0:\\n            raise ValueError(\\\"n must be > 0\\\")\\n"
    "        self.qty += n\\n\\n\\u00e9t\\u00e9 \\ud83d\\ude00 done\\n\"\n"
    "  }\n}\n```";

static void test_early_call_opens_before_the_object_closes(void)
{
    recorder r;
    tc_tool_call *calls;
    int n = stream_text(&r, LONG_CALL, 3, TC_DIALECT_ORNITH_JSON, &calls);
    ASSERT_INT_EQ(n, 1);
    ASSERT_INT_EQ(r.n_begin, 1);
    ASSERT_STR_EQ(r.name[0], "write");
    /* Opened right after the name, long before the call ends. */
    const char *after_name = strstr(LONG_CALL, "\"write\"") + strlen("\"write\"");
    ASSERT_TRUE(r.fed_at_begin[0] <= (size_t)(after_name - LONG_CALL) + 3);
    ASSERT_INT_EQ(wire_matches_parse(&r, calls, n), 0);
    tc_tool_calls_free(calls, n);
    rec_free(&r);
}

static void test_early_arguments_stream_while_written(void)
{
    recorder r;
    tc_tool_call *calls;
    int n = stream_text(&r, LONG_CALL, 4, TC_DIALECT_ORNITH_JSON, &calls);
    ASSERT_INT_EQ(n, 1);
    /* Many fragments, the first long before the text ends. */
    ASSERT_TRUE(r.fragments[0] > 20);
    const char *content = strstr(LONG_CALL, "\"content\"");
    ASSERT_TRUE(r.fed_at_first_fragment[0] < (size_t)(content - LONG_CALL) + 16);
    ASSERT_INT_EQ(wire_matches_parse(&r, calls, n), 0);
    tc_tool_calls_free(calls, n);
    rec_free(&r);
}

/* Every feed size, token-like and byte-at-a-time: the bytes on the wire are the
 * parse's bytes. */
static void test_fragments_concatenate_to_parsed_arguments(void)
{
    for (size_t piece = 1; piece <= 13; piece++) {
        recorder r;
        tc_tool_call *calls;
        int n = stream_text(&r, LONG_CALL, piece, TC_DIALECT_ORNITH_JSON, &calls);
        ASSERT_INT_EQ(n, 1);
        ASSERT_INT_EQ(wire_matches_parse(&r, calls, n), 0);
        ASSERT_TRUE(r.progress > 0);
        tc_tool_calls_free(calls, n);
        rec_free(&r);
    }
}

static void test_two_calls_two_indices(void)
{
    const char *text =
        "```json\n{\"function\":\"read\",\"arguments\":{\"path\":\"a\"}}\n```\n"
        "```json\n{\"function\":\"write\",\"arguments\":{\"path\":\"b\",\"content\":\"x\"}}\n```";
    recorder r;
    tc_tool_call *calls;
    int n = stream_text(&r, text, 2, TC_DIALECT_ORNITH_JSON, &calls);
    ASSERT_INT_EQ(n, 2);
    ASSERT_INT_EQ(r.n_begin, 2);
    ASSERT_INT_EQ(r.index[0], 0);
    ASSERT_INT_EQ(r.index[1], 1);
    ASSERT_INT_EQ(wire_matches_parse(&r, calls, n), 0);
    tc_tool_calls_free(calls, n);
    rec_free(&r);
}

/* "name" can be outranked by a later "function" key, so it is held to the
 * object's close — then sent whole, still before the turn ends. */
static void test_name_spelling_held_to_object_close(void)
{
    const char *text =
        "{\"name\":\"write\",\"arguments\":{\"path\":\"p\",\"content\":\"long\"}}\ntrailing";
    recorder r;
    tc_tool_call *calls;
    int n = stream_text(&r, text, 1, TC_DIALECT_ORNITH_JSON, &calls);
    ASSERT_INT_EQ(n, 1);
    ASSERT_INT_EQ(r.n_begin, 1);
    size_t close_at = (size_t)(strchr(text, '\n') - text);
    ASSERT_INT_EQ((int)r.fed_at_begin[0], (int)close_at);
    ASSERT_INT_EQ(wire_matches_parse(&r, calls, n), 0);
    tc_tool_calls_free(calls, n);
    rec_free(&r);
}

static void test_arguments_before_function_key(void)
{
    const char *text =
        "{\"arguments\":{\"path\":\"p\",\"content\":\"c\"},\"function\":\"write\"}";
    recorder r;
    tc_tool_call *calls;
    int n = stream_text(&r, text, 1, TC_DIALECT_ORNITH_JSON, &calls);
    ASSERT_INT_EQ(n, 1);
    ASSERT_INT_EQ(wire_matches_parse(&r, calls, n), 0);
    tc_tool_calls_free(calls, n);
    rec_free(&r);
}

static void test_unknown_tool_never_opens(void)
{
    const char *text = "{\"function\":\"rm_rf\",\"arguments\":{\"path\":\"/\"}}";
    recorder r;
    tc_tool_call *calls;
    int n = stream_text(&r, text, 1, TC_DIALECT_ORNITH_JSON, &calls);
    ASSERT_INT_EQ(n, 0);
    ASSERT_INT_EQ(r.n_begin, 0);
    tc_tool_calls_free(calls, n);
    rec_free(&r);
}

/* Cut off mid-string (max_tokens): the parse repairs it, and the stream owes
 * exactly the repaired tail. */
static void test_truncated_call_repaired(void)
{
    const char *text =
        "```json\n{\"function\":\"write\",\"arguments\":{\"path\":\"p\",\"content\":\"abc\\ndef";
    recorder r;
    tc_tool_call *calls;
    int n = stream_text(&r, text, 2, TC_DIALECT_ORNITH_JSON, &calls);
    ASSERT_INT_EQ(n, 1);
    ASSERT_STR_EQ(calls[0].arguments, "{\"path\":\"p\",\"content\":\"abc\\ndef\"}");
    ASSERT_INT_EQ(wire_matches_parse(&r, calls, n), 0);
    tc_tool_calls_free(calls, n);
    rec_free(&r);
}

/* Cut off where no repair is possible: the call was opened on the wire, the
 * parse has none — its streamed arguments must not be valid JSON. */
static void test_truncated_unrepairable_left_unparseable(void)
{
    const char *text = "{\"function\":\"write\",\"arguments\":{\"path\":";
    recorder r;
    tc_tool_call *calls;
    int n = stream_text(&r, text, 1, TC_DIALECT_ORNITH_JSON, &calls);
    ASSERT_INT_EQ(n, 0);
    ASSERT_INT_EQ(r.n_begin, 1);
    ASSERT_INT_EQ(wire_matches_parse(&r, calls, n), 0);
    tc_tool_calls_free(calls, n);
    rec_free(&r);
}

/* Arguments complete, then the object goes bad: the held final '}' is what
 * keeps the abandoned call from looking runnable. */
static void test_malformed_after_arguments_left_unparseable(void)
{
    const char *text =
        "{\"function\":\"write\",\"arguments\":{\"path\":\"p\",\"content\":\"c\"},\"x\":tru}"
        "\n{\"function\":\"read\",\"arguments\":{\"path\":\"q\"}}";
    recorder r;
    tc_tool_call *calls;
    int n = stream_text(&r, text, 1, TC_DIALECT_ORNITH_JSON, &calls);
    ASSERT_INT_EQ(n, 1);
    ASSERT_STR_EQ(calls[0].name, "read");
    ASSERT_INT_EQ(r.n_begin, 2);
    ASSERT_STR_EQ(r.args[0].data, "{\"path\":\"p\",\"content\":\"c\"");
    ASSERT_INT_EQ(wire_matches_parse(&r, calls, n), 0);
    tc_tool_calls_free(calls, n);
    rec_free(&r);
}

/* XML dialect: nothing streams early (values are typed and may be rewritten by
 * recovery), progress still fires for the keepalive, and the end sends whole
 * calls at indices 0.. exactly as before. */
static void test_xml_dialect_sent_whole_at_finish(void)
{
    const char *text =
        "<tool_call>\n<function=write>\n<parameter=path>\na.txt\n</parameter>\n"
        "<parameter=content>\nhello\n</parameter>\n</function>\n</tool_call>";
    recorder r;
    tc_tool_call *calls;
    size_t len = strlen(text);
    int n = stream_text(&r, text, 5, TC_DIALECT_QWEN_XML, &calls);
    ASSERT_INT_EQ(n, 1);
    ASSERT_INT_EQ(r.n_begin, 1);
    ASSERT_INT_EQ((int)r.fed_at_begin[0], (int)len);   /* only at finish */
    ASSERT_INT_EQ(r.index[0], 0);
    ASSERT_INT_EQ(r.fragments[0], 1);
    ASSERT_TRUE(r.progress >= (int)(len / 5));
    ASSERT_INT_EQ(wire_matches_parse(&r, calls, n), 0);
    tc_tool_calls_free(calls, n);
    rec_free(&r);
}

/* ============================================================
 * Emitter tap: the streamer sees exactly tool_raw.
 * ============================================================ */

typedef struct { json_writer seen; } tap_sink;

static void tap_collect(const char *bytes, size_t n, void *ud)
{
    jw_raw_n(&((tap_sink *)ud)->seen, bytes, n);
}

static void test_emitter_tap_sees_exactly_tool_raw(void)
{
    static const char *const pieces[] = {
        "Let me write it", ".\n\n``", "`js", "on\n{\"func", "tion\": \"wr", "ite\", ",
        "\"arguments\": {\"path\": \"x\", \"content\": \"a\\nb\"}}\n``", "`",
    };
    tap_sink t;
    jw_init(&t.seen);
    emitter em;
    em_init(&em, NULL, NULL, NULL, 0, true, TC_DIALECT_ORNITH_JSON);
    em_set_tool_tap(&em, tap_collect, &t);
    em_feed(&em, "</think>");
    for (size_t i = 0; i < sizeof(pieces) / sizeof(*pieces); i++)
        em_feed(&em, pieces[i]);
    em_flush(&em);
    ASSERT_NOT_NULL(em.tool.data);
    ASSERT_STR_EQ(t.seen.data, em.tool.data);
    ASSERT_TRUE(strncmp(em.tool.data, "```json", 7) == 0);
    em_free(&em);
    jw_free(&t.seen);
}

int main(void)
{
    printf("\n  tool-call streaming\n\n");
    RUN_TEST(jcs_matches_dom_fixed_docs);
    RUN_TEST(jcs_rejects_what_dom_rejects);
    RUN_TEST(jcs_matches_dom_generated_docs);
    RUN_TEST(early_call_opens_before_the_object_closes);
    RUN_TEST(early_arguments_stream_while_written);
    RUN_TEST(fragments_concatenate_to_parsed_arguments);
    RUN_TEST(two_calls_two_indices);
    RUN_TEST(name_spelling_held_to_object_close);
    RUN_TEST(arguments_before_function_key);
    RUN_TEST(unknown_tool_never_opens);
    RUN_TEST(truncated_call_repaired);
    RUN_TEST(truncated_unrepairable_left_unparseable);
    RUN_TEST(malformed_after_arguments_left_unparseable);
    RUN_TEST(xml_dialect_sent_whole_at_finish);
    RUN_TEST(emitter_tap_sees_exactly_tool_raw);
    TEST_REPORT();
    return __ft_failed == 0 ? 0 : 1;
}
