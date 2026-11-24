/* SPDX-License-Identifier: AGPL-3.0-or-later */
/* Copyright (C) 2026 David (Corey) Herrera */
/**
 * @file test_template_parity_qwen.c
 * @brief Pantheon's C prompt rendering vs the Qwen3.6 / Qwen3.8 chat templates.
 *
 * Each fixture under tests/data/template_parity_qwen/fixtures/ is an OpenAI
 * wire request body.  The test runs it through the serve's own request parse
 * (parse_chat_request, reached by #including openai_chat.c) and
 * sg_build_prompt_tokens_bounded, then compares the result to the reference
 * Jinja render committed under goldens/ (tests/host/render_qwen_template_goldens.py
 * writes both; Python is not needed at test time).
 *
 * WHAT IS COMPARED.  The tokenizer is synthetic: one id per byte plus the
 * model's added tokens (its own list, extracted beside the template) as single
 * ids.  So the ids decode losslessly into "units": a byte, or an added token.
 * The golden text is split into the same units the way an HF / llama.cpp
 * tokenizer splits added tokens out before BPE (leftmost-longest match).
 * Comparing units rather than real-vocab ids keeps the test model-free and
 * reports differences as bytes; it still sees the one thing a byte compare
 * would miss, an added-token string sent as plain text.  Two passes:
 *   - TEXT: the rendered bytes (added tokens spelled out).  Turns (each starts
 *     at <|im_start|>) are aligned by LCS, so an extra or missing turn does not
 *     shift the rest; a differing run of turns is paired by role and each pair
 *     is diffed by line.  Every differing line run is reported with its byte
 *     offset in both renderings.
 *   - TOKENIZATION: for turns whose text matches, whether every added-token
 *     string is a single id on both sides.  BPE merges inside plain text are
 *     not compared (pn_bpe_encode vs the HF BPE is test_bpe_pretokenize's job).
 *
 * Template modes (goldens/<fixture>.<mode>.txt): qwen36_default; qwen38_default
 * (preserve_thinking undefined = keep all, reasoning_effort xhigh);
 * qwen38_ptfalse; qwen38_medium and qwen38_medium_ptfalse (reasoning_effort
 * "medium" adds no system text, isolating the think-block rules).
 *
 * The differences are PINNED in template_parity_qwen_pins.h.  Card 1.8 fixed
 * D3/D4/D5/D7; the rows now hold only D1, D2 and D6, kept on purpose (spec
 * §2.2).  A change to the rendering, either way, fails this test (it prints
 * the new row) until the pin and the spec are updated together.
 *
 * @status Complete.  CPU, no model, runs in `make test`.
 * @issues None
 * @todo None
 */

#include "lib/test.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

/* The unit under test: the request parse (statics reachable). */
#define main oai_chat_main_unused
#include "../src/serve/openai_chat.c"
#undef main
#include "openai_chat_link_stubs.h"

#include "../src/core/bpe_pretokenize.h"
#include "../src/core/chat.h"
#include "../src/serve/serve_prompt_tokens.h"
#include "tokenizer/tokenizer.h"

#define PARITY_DATA_DIR "tests/data/template_parity_qwen/"
#define MAX_ADDED_TOKENS 64
#define BYTE_TOKEN_COUNT 256

/* ============================================================
 * Files and the per-model tokenizer
 * ============================================================ */

/* Read a whole file; NULL if missing.  Owned. */
static char *read_whole_file(const char *path)
{
    FILE *file = fopen(path, "rb");
    if (!file) return NULL;
    fseek(file, 0, SEEK_END);
    long size = ftell(file);
    fseek(file, 0, SEEK_SET);
    char *data = malloc((size_t)size + 1);
    if (data && fread(data, 1, (size_t)size, file) != (size_t)size) {
        free(data);
        data = NULL;
    }
    if (data) data[size] = '\0';
    fclose(file);
    return data;
}

/* A model's template + added tokens + the synthetic tokenizer built from them.
 * Token ids: 0..255 are the bytes, 256+i is added_tokens[i]. */
typedef struct {
    char *template_text;
    char *added_tokens_file;            /* backing store for added_tokens[] */
    const char *added_tokens[MAX_ADDED_TOKENS];
    int n_added_tokens;
    ft_tokenizer *tokenizer;
    pn_model model;
} parity_model;

static bool load_parity_model(parity_model *pm, const char *model_key)
{
    char path[256];
    memset(pm, 0, sizeof(*pm));
    snprintf(path, sizeof(path), PARITY_DATA_DIR "%s_chat_template.jinja", model_key);
    pm->template_text = read_whole_file(path);
    snprintf(path, sizeof(path), PARITY_DATA_DIR "%s_added_tokens.txt", model_key);
    pm->added_tokens_file = read_whole_file(path);
    if (!pm->template_text || !pm->added_tokens_file) return false;
    for (char *line = strtok(pm->added_tokens_file, "\n");
         line && pm->n_added_tokens < MAX_ADDED_TOKENS; line = strtok(NULL, "\n"))
        pm->added_tokens[pm->n_added_tokens++] = line;

    pm->tokenizer = ft_new();
    ft_set_model_type(pm->tokenizer, 0);   /* BPE: text goes through pn_bpe_encode */
    for (int b = 0; b < BYTE_TOKEN_COUNT; b++) {
        char raw[2] = { (char)b, 0 };
        char *encoded = gpt2_bytes_encode(raw);
        ft_add_token(pm->tokenizer, encoded);
        free(encoded);
    }
    for (int i = 0; i < pm->n_added_tokens; i++)
        ft_add_token(pm->tokenizer, pm->added_tokens[i]);
    pm->model.tokenizer = pm->tokenizer;
    /* Exactly what engine.c derives from tokenizer.chat_template at load. */
    pm->model.chat_style = pn_chat_detect_style(pm->template_text);
    const char *bos = strstr(pm->template_text, "bos_token");
    const char *im = strstr(pm->template_text, "<|im_start|>");
    pm->model.chat_add_bos = (bos && (!im || bos < im)) ? 1 : 0;
    pm->model.chat_trim_content = pn_chat_template_trims_content(pm->template_text) ? 1 : 0;
    return true;
}

static void free_parity_model(parity_model *pm)
{
    if (pm->tokenizer) ft_free(pm->tokenizer);
    free(pm->template_text);
    free(pm->added_tokens_file);
}

/* ============================================================
 * Unit streams: a byte (value 0..255) or an added token (256 + index)
 * ============================================================ */

typedef struct {
    int *units;
    int len;
    int *turn_starts;   /* unit index where each turn begins */
    int n_turns;
} unit_stream;

static bool unit_is_added(int unit) { return unit >= BYTE_TOKEN_COUNT; }

/* Split rendered text into units the way an HF / llama.cpp tokenizer splits
 * added tokens out before BPE: at each position, the longest added token that
 * matches, else one byte. */
static void units_from_text(const parity_model *pm, const char *text, unit_stream *out)
{
    size_t text_len = strlen(text);
    out->units = malloc((text_len + 1) * sizeof(int));
    out->len = 0;
    for (size_t pos = 0; pos < text_len;) {
        int best = -1;
        size_t best_len = 0;
        for (int i = 0; i < pm->n_added_tokens; i++) {
            size_t len = strlen(pm->added_tokens[i]);
            if (len > best_len && strncmp(text + pos, pm->added_tokens[i], len) == 0) {
                best = i;
                best_len = len;
            }
        }
        out->units[out->len++] = best >= 0 ? BYTE_TOKEN_COUNT + best
                                           : (unsigned char)text[pos];
        pos += best >= 0 ? best_len : 1;
    }
}

/* Mark turn starts: unit 0, and every <|im_start|>. */
static void split_turns(const parity_model *pm, unit_stream *stream)
{
    int im_start = -1;
    for (int i = 0; i < pm->n_added_tokens; i++)
        if (strcmp(pm->added_tokens[i], "<|im_start|>") == 0) im_start = BYTE_TOKEN_COUNT + i;
    stream->turn_starts = malloc(((size_t)stream->len + 1) * sizeof(int));
    stream->n_turns = 0;
    for (int i = 0; i < stream->len; i++)
        if (i == 0 || stream->units[i] == im_start)
            stream->turn_starts[stream->n_turns++] = i;
}

static int turn_end(const unit_stream *stream, int turn)
{
    return turn + 1 < stream->n_turns ? stream->turn_starts[turn + 1] : stream->len;
}

/* Append unit `unit`'s spelling (added tokens spelled out) to buf. */
static size_t append_unit_text(const parity_model *pm, int unit, char *buf, size_t at)
{
    if (!unit_is_added(unit)) {
        buf[at] = (char)unit;
        return at + 1;
    }
    const char *spelling = pm->added_tokens[unit - BYTE_TOKEN_COUNT];
    memcpy(buf + at, spelling, strlen(spelling));
    return at + strlen(spelling);
}

/* The text of units [begin, end); owned. */
static char *units_text(const parity_model *pm, const unit_stream *stream, int begin, int end)
{
    char *buf = malloc((size_t)(end - begin) * 32 + 1);
    size_t at = 0;
    for (int i = begin; i < end; i++) at = append_unit_text(pm, stream->units[i], buf, at);
    buf[at] = '\0';
    return buf;
}

/* Byte offset (in the rendered text) where unit `index` begins. */
static size_t text_offset_of_unit(const parity_model *pm, const unit_stream *stream, int index)
{
    size_t offset = 0;
    for (int i = 0; i < index; i++)
        offset += unit_is_added(stream->units[i])
                      ? strlen(pm->added_tokens[stream->units[i] - BYTE_TOKEN_COUNT]) : 1;
    return offset;
}

static void free_unit_stream(unit_stream *stream)
{
    free(stream->units);
    free(stream->turn_starts);
}

/* ============================================================
 * Rendering Pantheon's side
 * ============================================================ */

/* Parse the fixture through the serve's request parse and build the prompt
 * with sg_build_prompt_tokens_bounded.  Returns false on any failure. */
static bool render_pantheon_units(parity_model *pm, const char *fixture_json, unit_stream *out)
{
    jr_value *root = jr_parse(fixture_json);
    if (!root) return false;
    chat_request request;
    const char *error = NULL;
    if (parse_chat_request(root, &request, &error) != 0) {
        fprintf(stderr, "parse_chat_request: %s\n", error ? error : "?");
        jr_free(root);
        return false;
    }
    int n_tokens = 0;
    sg_prompt_bounds bounds;
    uint32_t *tokens = sg_build_prompt_tokens_bounded(&pm->model, request.messages,
                                                      request.n_messages, SG_THINKING_DEFAULT,
                                                      &n_tokens, &bounds);
    chat_request_free(&request);
    jr_free(root);
    if (!tokens) return false;
    out->units = malloc(((size_t)n_tokens + 1) * sizeof(int));
    out->len = n_tokens;
    for (int i = 0; i < n_tokens; i++) out->units[i] = (int)tokens[i];
    free(tokens);
    return true;
}

/* ============================================================
 * Edit scripts (LCS), used at three levels: turns by text, turns by role
 * inside a differing run, and lines inside a paired turn
 * ============================================================ */

typedef enum { EDIT_KEEP, EDIT_TEMPLATE_ONLY, EDIT_PANTHEON_ONLY } edit_op;
typedef bool (*item_equal_fn)(const void *context, int template_index, int pantheon_index);

/* Fill ops[] with an LCS edit script turning n template items into m Pantheon
 * items; returns the op count (<= n + m).  Ties prefer dropping template
 * items first, so a run reads "template had X, pantheon has Y". */
static int lcs_edit_script(int n, int m, item_equal_fn equal, const void *context, edit_op *ops)
{
    int *table = calloc((size_t)(n + 1) * (size_t)(m + 1), sizeof(int));
    bool *same = calloc((size_t)n * (size_t)m + 1, sizeof(bool));
    for (int a = 0; a < n; a++)
        for (int b = 0; b < m; b++) same[a * m + b] = equal(context, a, b);
    for (int a = n - 1; a >= 0; a--)
        for (int b = m - 1; b >= 0; b--) {
            int skip_a = table[(a + 1) * (m + 1) + b], skip_b = table[a * (m + 1) + b + 1];
            table[a * (m + 1) + b] = same[a * m + b] ? table[(a + 1) * (m + 1) + b + 1] + 1
                                                     : (skip_a > skip_b ? skip_a : skip_b);
        }
    int n_ops = 0, a = 0, b = 0;
    while (a < n || b < m) {
        if (a < n && b < m && same[a * m + b] &&
            table[a * (m + 1) + b] == table[(a + 1) * (m + 1) + b + 1] + 1) {
            ops[n_ops++] = EDIT_KEEP;
            a++;
            b++;
        } else if (b >= m || (a < n && table[(a + 1) * (m + 1) + b] >= table[a * (m + 1) + b + 1])) {
            ops[n_ops++] = EDIT_TEMPLATE_ONLY;
            a++;
        } else {
            ops[n_ops++] = EDIT_PANTHEON_ONLY;
            b++;
        }
    }
    free(table);
    free(same);
    return n_ops;
}

/* ============================================================
 * Turn texts and the comparison context
 * ============================================================ */

/* One rendering split into turns: each turn's text (added tokens spelled out)
 * and its byte offset in the full rendering. */
typedef struct {
    char **texts;
    size_t *offsets;
    int n;
} turn_texts;

static void build_turn_texts(const parity_model *pm, const unit_stream *stream, turn_texts *out)
{
    out->n = stream->n_turns;
    out->texts = malloc(((size_t)out->n + 1) * sizeof(char *));
    out->offsets = malloc(((size_t)out->n + 1) * sizeof(size_t));
    for (int t = 0; t < out->n; t++) {
        out->texts[t] = units_text(pm, stream, stream->turn_starts[t], turn_end(stream, t));
        out->offsets[t] = text_offset_of_unit(pm, stream, stream->turn_starts[t]);
    }
}

static void free_turn_texts(turn_texts *tt)
{
    for (int t = 0; t < tt->n; t++) free(tt->texts[t]);
    free(tt->texts);
    free(tt->offsets);
}

#define SIGNATURE_CAP 1024

/* Everything one case's comparison reads and writes. */
typedef struct {
    const parity_model *pm;
    const unit_stream *expected, *got;
    turn_texts expected_turns, got_turns;
    char signature[SIGNATURE_CAP];   /* compact list of every difference, pinned below */
    int n_text_differences;
    int n_tokenization_differences;
} parity_comparison;

static void signature_append(parity_comparison *cmp, const char *format, long x, long y)
{
    size_t used = strlen(cmp->signature);
    snprintf(cmp->signature + used, SIGNATURE_CAP - used, format, x, y);
}

static bool turns_text_equal(const void *context, int a, int b)
{
    const parity_comparison *cmp = context;
    return strcmp(cmp->expected_turns.texts[a], cmp->got_turns.texts[b]) == 0;
}

/* The role of a turn: the text between <|im_start|> and the first newline. */
static void turn_role(const char *turn_text, char *role, size_t cap)
{
    const char *begin = strncmp(turn_text, "<|im_start|>", 12) == 0 ? turn_text + 12 : turn_text;
    size_t len = strcspn(begin, "\n");
    if (len >= cap) len = cap - 1;
    memcpy(role, begin, len);
    role[len] = '\0';
}

static bool turns_role_equal(const void *context, int a, int b)
{
    const parity_comparison *cmp = context;
    char role_a[32], role_b[32];
    turn_role(cmp->expected_turns.texts[a], role_a, sizeof(role_a));
    turn_role(cmp->got_turns.texts[b], role_b, sizeof(role_b));
    return strcmp(role_a, role_b) == 0;
}

/* ============================================================
 * Line-level difference inside one paired turn
 * ============================================================ */

typedef struct {
    const char **starts;   /* each line's start (lines keep their "\n") */
    size_t *lens;
    int n;
} line_list;

static void split_lines(const char *text, line_list *out)
{
    size_t cap = 16;
    out->starts = malloc(cap * sizeof(char *));
    out->lens = malloc(cap * sizeof(size_t));
    out->n = 0;
    for (const char *p = text; *p;) {
        const char *newline = strchr(p, '\n');
        size_t len = newline ? (size_t)(newline - p) + 1 : strlen(p);
        if ((size_t)out->n == cap) {
            cap *= 2;
            out->starts = realloc(out->starts, cap * sizeof(char *));
            out->lens = realloc(out->lens, cap * sizeof(size_t));
        }
        out->starts[out->n] = p;
        out->lens[out->n++] = len;
        p += len;
    }
}

typedef struct { const line_list *left, *right; } line_pair;

static bool lines_equal(const void *context, int a, int b)
{
    const line_pair *lp = context;
    return lp->left->lens[a] == lp->right->lens[b] &&
           memcmp(lp->left->starts[a], lp->right->starts[b], lp->left->lens[a]) == 0;
}

/* Print lines [first, first + count) escaped, or "(nothing)". */
static void print_lines(const char *label, const line_list *lines, int first, int count)
{
    printf("        %s", label);
    if (count == 0) printf(" (nothing)");
    for (int i = first; i < first + count; i++) {
        printf(" \"");
        for (size_t k = 0; k < lines->lens[i] && k < 90; k++) {
            char c = lines->starts[i][k];
            if (c == '\n') printf("\\n");
            else if (c == '\t') printf("\\t");
            else putchar(c);
        }
        printf(lines->lens[i] > 90 ? "...\"" : "\"");
    }
    printf("\n");
}

/* Report every differing line run between template turn a and Pantheon turn b. */
static void report_turn_line_differences(parity_comparison *cmp, int a, int b)
{
    line_list left, right;
    split_lines(cmp->expected_turns.texts[a], &left);
    split_lines(cmp->got_turns.texts[b], &right);
    line_pair lp = { &left, &right };
    edit_op *ops = malloc(((size_t)left.n + (size_t)right.n + 1) * sizeof(edit_op));
    int n_ops = lcs_edit_script(left.n, right.n, lines_equal, &lp, ops);
    char role[32];
    turn_role(cmp->got_turns.texts[b], role, sizeof(role));
    int la = 0, lb = 0;
    size_t off_a = cmp->expected_turns.offsets[a], off_b = cmp->got_turns.offsets[b];
    for (int k = 0; k < n_ops;) {
        if (ops[k] == EDIT_KEEP) {
            off_a += left.lens[la++];
            off_b += right.lens[lb++];
            k++;
            continue;
        }
        int first_a = la, first_b = lb;
        size_t run_off_a = off_a, run_off_b = off_b;
        for (; k < n_ops && ops[k] != EDIT_KEEP; k++) {
            if (ops[k] == EDIT_TEMPLATE_ONLY) off_a += left.lens[la++];
            else off_b += right.lens[lb++];
        }
        printf("    TEXT [template turn %d / pantheon turn %d, %s] template byte %zu, pantheon byte %zu\n",
               a, b, role, run_off_a, run_off_b);
        print_lines("template:", &left, first_a, la - first_a);
        print_lines("pantheon:", &right, first_b, lb - first_b);
        signature_append(cmp, "T%ld/%ld ", (long)run_off_a, (long)run_off_b);
        cmp->n_text_differences++;
    }
    free(ops);
    free(left.starts);
    free(left.lens);
    free(right.starts);
    free(right.lens);
}

/* ============================================================
 * Turn-level comparison
 * ============================================================ */

/* For a text-equal turn pair: is every added-token string a single id on both
 * sides?  Reports the first one that is not. */
static void check_turn_tokenization(parity_comparison *cmp, int a, int b)
{
    const unit_stream *expected = cmp->expected, *got = cmp->got;
    int i = expected->turn_starts[a], j = got->turn_starts[b];
    int end_i = turn_end(expected, a), end_j = turn_end(got, b);
    while (i < end_i && j < end_j && expected->units[i] == got->units[j]) {
        i++;
        j++;
    }
    if (i == end_i && j == end_j) return;
    int template_unit = i < end_i ? expected->units[i] : 0;
    int pantheon_unit = j < end_j ? got->units[j] : 0;
    const int added_unit = unit_is_added(template_unit) ? template_unit : pantheon_unit;
    const char *spelling = unit_is_added(added_unit)
        ? cmp->pm->added_tokens[added_unit - BYTE_TOKEN_COUNT] : "?";
    size_t offset = text_offset_of_unit(cmp->pm, expected, i);
    printf("    TOKENIZATION [template turn %d / pantheon turn %d] template byte %zu: \"%s\" is %s "
           "in the template, %s in pantheon\n", a, b, offset, spelling,
           unit_is_added(template_unit) ? "one added-token id" : "bytes",
           unit_is_added(pantheon_unit) ? "one added-token id" : "bytes");
    signature_append(cmp, "K%ld/%ld ", (long)offset, (long)b);
    cmp->n_tokenization_differences++;
}

/* A run of turns that differ in text: pair them by role, then diff each pair
 * by line; a turn with no counterpart is reported whole. */
static void report_differing_run(parity_comparison *cmp, int a0, int n_a, int b0, int n_b)
{
    parity_comparison shifted = *cmp;   /* role LCS works on the run's own indices */
    shifted.expected_turns.texts += a0;
    shifted.got_turns.texts += b0;
    edit_op *ops = malloc(((size_t)n_a + (size_t)n_b + 1) * sizeof(edit_op));
    int n_ops = lcs_edit_script(n_a, n_b, turns_role_equal, &shifted, ops);
    int a = a0, b = b0;
    for (int k = 0; k < n_ops; k++) {
        if (ops[k] == EDIT_KEEP) {
            report_turn_line_differences(cmp, a++, b++);
            continue;
        }
        bool template_side = ops[k] == EDIT_TEMPLATE_ONLY;
        int turn = template_side ? a++ : b++;
        const turn_texts *tt = template_side ? &cmp->expected_turns : &cmp->got_turns;
        line_list lines;
        split_lines(tt->texts[turn], &lines);
        printf("    TEXT [%s turn %d has no counterpart] byte %zu\n",
               template_side ? "template" : "pantheon", turn, tt->offsets[turn]);
        print_lines(template_side ? "template:" : "pantheon:", &lines, 0, lines.n);
        signature_append(cmp, template_side ? "Xt%ld/%ld " : "Xp%ld/%ld ",
                         (long)tt->offsets[turn], (long)turn);
        cmp->n_text_differences++;
        free(lines.starts);
        free(lines.lens);
    }
    free(ops);
}

/* LCS-align the turns by text; report each differing run and check the
 * matching turns' tokenization. */
static void compare_renderings(parity_comparison *cmp)
{
    int n = cmp->expected_turns.n, m = cmp->got_turns.n;
    edit_op *ops = malloc(((size_t)n + (size_t)m + 1) * sizeof(edit_op));
    int n_ops = lcs_edit_script(n, m, turns_text_equal, cmp, ops);
    int a = 0, b = 0;
    for (int k = 0; k < n_ops;) {
        if (ops[k] == EDIT_KEEP) {
            check_turn_tokenization(cmp, a++, b++);
            k++;
            continue;
        }
        int a0 = a, b0 = b;
        for (; k < n_ops && ops[k] != EDIT_KEEP; k++) {
            if (ops[k] == EDIT_TEMPLATE_ONLY) a++;
            else b++;
        }
        report_differing_run(cmp, a0, a - a0, b0, b - b0);
    }
    free(ops);
}

/* ============================================================
 * Cases
 * ============================================================ */

/* Render one fixture both ways and compare; fills cmp->signature.  Returns
 * false if a file is missing or a render failed. */
static bool run_parity_case(const char *fixture, const char *mode, const char *model_key,
                            parity_comparison *cmp)
{
    char path[256];
    parity_model pm;
    memset(cmp, 0, sizeof(*cmp));
    if (!load_parity_model(&pm, model_key)) return false;
    snprintf(path, sizeof(path), PARITY_DATA_DIR "fixtures/%s.json", fixture);
    char *fixture_json = read_whole_file(path);
    snprintf(path, sizeof(path), PARITY_DATA_DIR "goldens/%s.%s.txt", fixture, mode);
    char *golden = read_whole_file(path);
    unit_stream expected = { 0 }, got = { 0 };
    bool ok = fixture_json && golden && render_pantheon_units(&pm, fixture_json, &got);
    if (ok) {
        units_from_text(&pm, golden, &expected);
        split_turns(&pm, &expected);
        split_turns(&pm, &got);
        printf("  %s.%s (template %d turns, pantheon %d turns)\n", fixture, mode,
               expected.n_turns, got.n_turns);
        cmp->pm = &pm;
        cmp->expected = &expected;
        cmp->got = &got;
        build_turn_texts(&pm, &expected, &cmp->expected_turns);
        build_turn_texts(&pm, &got, &cmp->got_turns);
        compare_renderings(cmp);
        free_turn_texts(&cmp->expected_turns);
        free_turn_texts(&cmp->got_turns);
        cmp->pm = NULL;
        cmp->expected = cmp->got = NULL;
    }
    free_unit_stream(&expected);
    free_unit_stream(&got);
    free(fixture_json);
    free(golden);
    free_parity_model(&pm);
    return ok;
}

/* One pinned case: the fixture, the template mode, and the exact list of
 * differences found (T = text run at template/pantheon byte, Xt/Xp = a turn
 * with no counterpart at byte/turn, K = tokenization at template byte /
 * pantheon turn).  Empty = byte- and token-identical. */
typedef struct {
    const char *fixture;
    const char *mode;
    const char *model_key;
    const char *pinned_signature;
} parity_case;

static const parity_case PARITY_CASES[] = {
#include "template_parity_qwen_pins.h"
};

static void test_template_parity_cases(void)
{
    int mismatches = 0;
    for (size_t i = 0; i < sizeof(PARITY_CASES) / sizeof(PARITY_CASES[0]); i++) {
        const parity_case *pc = &PARITY_CASES[i];
        parity_comparison cmp;
        bool rendered = run_parity_case(pc->fixture, pc->mode, pc->model_key, &cmp);
        ASSERT_TRUE(rendered);
        if (strcmp(cmp.signature, pc->pinned_signature) == 0) continue;
        mismatches++;
        printf("    PIN MISMATCH, pinned \"%s\"; the row for what renders now:\n"
               "{ \"%s\", \"%s\", \"%s\",\n  \"%s\" },\n",
               pc->pinned_signature, pc->fixture, pc->mode, pc->model_key, cmp.signature);
    }
    ASSERT_INT_EQ(mismatches, 0);
}

int main(void)
{
    unsetenv("PANTHEON_NO_THINK");   /* SG_THINKING_DEFAULT must mean thinking on */
    printf("=== template parity: Pantheon prompt vs Qwen chat templates ===\n");
    RUN_TEST(template_parity_cases);
    TEST_REPORT();
    return __ft_failed != 0;
}
