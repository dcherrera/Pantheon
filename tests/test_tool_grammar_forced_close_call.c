/* SPDX-License-Identifier: AGPL-3.0-or-later */
/* Copyright (C) 2026 David (Corey) Herrera */
/**
 * @file test_tool_grammar_forced_close_call.c
 * @brief tool_choice "required": after the thinking close (forced or the model's
 *        own) only the tool-call opener may follow, in both dialects and both
 *        decode-path shapes.
 *
 * @status Complete.  CPU-only — no device, no model.
 * @issues None
 * @todo None
 *
 * The defect (K80 host, Qwen3.6 Q8, keys/budget smoke L2, 2026-11-18): the budget
 * force-closed thinking at 1933/2000 on a line end, then the reply ran to
 * max_tokens (758 s) with no tool call and no drop.  Two holes let a model that
 * wants to keep writing prose do that under a "required" grammar:
 *   1. tg_reasoning_end_injected granted the XML walk a 512-char landing window
 *      of free prose after the injected close;
 *   2. the compelled opener tolerated unbounded leading whitespace, so a model
 *      whose top token was masked took whitespace, its best allowed pick, until
 *      max_tokens.
 *
 * Part 1 checks the grammar entry points.  Part 2 runs a scripted "model"
 * through the real sampler (pn_sample_next_token): after the close it ranks
 * prose first, whitespace next, the call opener after that.  The call must come
 * out with only bounded whitespace before it, for a forced and a natural close,
 * in the single-user shape (the serve advances the walk and calls the grammar's
 * injected-end itself, as sg_tg_reasoning_ended_injected does) and the batched
 * shape (the sampler advances; the driver advances the injected close and calls
 * hook.reasoning_end_injected, as batch_sched.c driver_reason_step does).  AUTO
 * is unchanged: prose and whitespace stay free.
 */

#include "../src/core/engine.h"
#include "../src/serve/json_reader.h"
#include "../src/serve/tool_grammar.h"
#include "../src/serve/tool_grammar_json.h"

#include <stdint.h>
#include <stdio.h>
#include <string.h>

static int g_fail = 0, g_checks = 0;
#define CHECK(cond, ...) do { \
    g_checks++; \
    if (!(cond)) { g_fail++; printf("  FAIL: "); printf(__VA_ARGS__); printf("\n"); } \
} while (0)

static const char TOOLS[] =
    "[{\"type\":\"function\",\"function\":{\"name\":\"read\",\"parameters\":{"
    "\"type\":\"object\",\"properties\":{"
    "\"path\":{\"type\":\"string\",\"minLength\":1}},"
    "\"required\":[\"path\"]}}}]";

#define WS_MAX 16   /* TG_OPENER_WS_MAX / TGJ_OPENER_WS_MAX */

/* ── Part 1: the grammar entry points ── */

static void test_xml_entry(const tg_grammar *g) {
    /* A required walk: the injected close compels the opener at once. */
    tg_state *st = tg_state_new_required(g, true);
    tg_set_think_close_token(st);
    tg_advance(st, "divisors of 96: 1, 2, 3\n");
    tg_advance(st, "</think>");
    tg_reasoning_end_injected(st);
    CHECK(tg_debug_landing_chars(st) == 0, "xml required: injected close grants %d landing chars",
          tg_debug_landing_chars(st));
    CHECK(tg_mask_active(st), "xml required: mask live right after the injected close");
    CHECK(!tg_token_allowed(st, "Wait, divisors"), "xml required: prose refused after the close");
    CHECK(tg_token_allowed(st, "<tool_call>"), "xml required: the opener allowed");
    CHECK(tg_token_allowed(st, "\n\n"), "xml required: the trained \"\\n\\n\" separator allowed");

    /* Whitespace before the opener is bounded. */
    int ws = 0;
    while (ws < 64 && tg_token_allowed(st, " ")) { tg_advance(st, " "); ws++; }
    CHECK(ws == WS_MAX, "xml required: %d whitespace chars before the opener, want %d", ws, WS_MAX);
    CHECK(tg_token_allowed(st, "<tool_call>"), "xml required: opener still allowed after the cap");
    CHECK(!tg_token_allowed(st, "\n"), "xml required: no whitespace past the cap");
    tg_state_free(st);

    /* The named-function form keeps its landing window (frame-capture fix). */
    tg_state *fn = tg_state_new_forced(g, NULL);
    tg_reasoning_end_injected(fn);
    CHECK(tg_debug_landing_chars(fn) == 512, "xml forced (not required): landing window kept");
    tg_state_free(fn);

    /* AUTO: an injected close changes nothing; prose and whitespace stay free. */
    tg_state *au = tg_state_new(g);
    tg_watch_reasoning(au);
    tg_advance(au, "thinking\n</think>");
    tg_reasoning_end_injected(au);
    CHECK(!tg_mask_active(au), "xml auto: no mask after the close");
    CHECK(tg_token_allowed(au, "Wait, divisors"), "xml auto: prose allowed");
    ws = 0;
    while (ws < 64 && tg_token_allowed(au, " ")) { tg_advance(au, " "); ws++; }
    CHECK(ws == 64, "xml auto: whitespace unbounded (%d)", ws);
    tg_state_free(au);
}

static void test_json_entry(const tgj_grammar *g) {
    tgj_state *st = tgj_state_new_required(g, true);
    tgj_set_think_close_token(st);
    tgj_advance(st, "divisors of 96: 1, 2, 3\n");
    tgj_advance(st, "</think>");
    tgj_reasoning_end(st);   /* sg_tg_reasoning_ended_injected's JSON branch */
    CHECK(tgj_mask_active(st), "json required: mask live right after the close");
    CHECK(!tgj_token_allowed(st, "Wait, divisors"), "json required: prose refused after the close");
    CHECK(tgj_token_allowed(st, "```json"), "json required: the opener allowed");
    int ws = 0;
    while (ws < 64 && tgj_token_allowed(st, " ")) { tgj_advance(st, " "); ws++; }
    CHECK(ws == WS_MAX, "json required: %d whitespace chars before the opener, want %d", ws, WS_MAX);
    CHECK(tgj_token_allowed(st, "```json"), "json required: opener still allowed after the cap");
    CHECK(!tgj_token_allowed(st, "\n"), "json required: no whitespace past the cap");
    tgj_state_free(st);

    tgj_state *au = tgj_state_new(g);
    tgj_watch_reasoning(au);
    tgj_advance(au, "thinking\n</think>");
    CHECK(tgj_token_allowed(au, "Wait, divisors"), "json auto: prose allowed");
    ws = 0;
    while (ws < 64 && tgj_token_allowed(au, " ")) { tgj_advance(au, " "); ws++; }
    CHECK(ws == 64, "json auto: whitespace unbounded (%d)", ws);
    tgj_state_free(au);
}

/* ── Part 2: a scripted model through the sampler ── */

enum {
    T_EOS, T_PROSE, T_NL2, T_SP, T_NL, T_CLOSE,
    X_OPEN, X_FN, X_PAR, X_VAL, X_PCLOSE, X_FCLOSE, X_TCLOSE,
    J_OPEN, J_BODY, J_END,
    T_FILLER, N_NAMED
};
#define VOCAB 512   /* wide enough not to read as a device's sparse top-k row */
static const char *const TEXT[N_NAMED] = {
    [T_EOS] = "", [T_PROSE] = "Wait, divisors of 96: 1, 2, 3, 4", [T_NL2] = "\n\n",
    [T_SP] = " ", [T_NL] = "\n", [T_CLOSE] = "</think>",
    [X_OPEN] = "<tool_call>", [X_FN] = "\n<function=read>", [X_PAR] = "\n<parameter=path>",
    [X_VAL] = "\nmain.c", [X_PCLOSE] = "\n</parameter>", [X_FCLOSE] = "\n</function>",
    [X_TCLOSE] = "\n</tool_call>",
    [J_OPEN] = "```json", [J_BODY] = "\n{\"function\": \"read\", \"arguments\": {\"path\": \"main.c\"}}",
    [J_END] = "\n```",
    [T_FILLER] = "zz",
};
static const uint32_t XML_CALL[] = { X_OPEN, X_FN, X_PAR, X_VAL, X_PCLOSE, X_FCLOSE, X_TCLOSE };
static const uint32_t JSON_CALL[] = { J_OPEN, J_BODY, J_END };

static const char *tok_text(void *user, uint32_t id) {
    (void)user;
    if (id >= VOCAB) return NULL;
    return id < N_NAMED ? TEXT[id] : TEXT[T_FILLER];
}

static int xml_allowed(void *st, const char *t) { return tg_token_allowed((tg_state *)st, t); }
static int xml_live(void *st) { return tg_mask_active((tg_state *)st); }
static void xml_advance(void *st, const char *t) { tg_advance((tg_state *)st, t); }
static int xml_end(void *st) { return tg_end_allowed((const tg_state *)st); }
static void xml_injected(void *st) { tg_reasoning_end_injected((tg_state *)st); }
static int json_allowed(void *st, const char *t) { return tgj_token_allowed((tgj_state *)st, t); }
static int json_live(void *st) { return tgj_mask_active((tgj_state *)st); }
static void json_advance(void *st, const char *t) { tgj_advance((tgj_state *)st, t); }
static int json_end(void *st) { return tgj_end_allowed((const tgj_state *)st); }

typedef struct {
    bool json;
    bool forced_close;     /* the serve injected "</think>" (else the model's own) */
    bool driver_advance;   /* batched shape: the sampler advances the walk */
    bool required;         /* tool_choice "required" (else AUTO) */
} run_opts;

typedef struct {
    bool completed;
    bool ended;            /* an end token was drawn */
    int  steps;
    int  ws_before;        /* whitespace chars before the first non-whitespace text */
    bool prose_before;     /* non-whitespace, non-opener text before the call */
    char out[2048];
} run_result;

/* Reason, close, then decode up to `max_tokens` with the model above. */
static run_result run(const void *grammar, run_opts o, int max_tokens) {
    run_result r;
    memset(&r, 0, sizeof(r));
    void *st;
    if (o.json) {
        st = o.required ? tgj_state_new_required(grammar, true) : tgj_state_new(grammar);
        if (!o.required) tgj_watch_reasoning(st);
        tgj_set_think_close_token(st);
    } else {
        st = o.required ? tg_state_new_required(grammar, true) : tg_state_new(grammar);
        if (!o.required) tg_watch_reasoning(st);
        tg_set_think_close_token(st);
    }

    pn_tool_grammar_hook hook;
    memset(&hook, 0, sizeof(hook));
    hook.st = st;
    hook.token_text = tok_text;
    hook.allowed = o.json ? json_allowed : xml_allowed;
    hook.refresh = o.json ? json_live : xml_live;
    hook.n_vocab = VOCAB;
    if (o.required) {
        /* sg_tg_setup's REQUIRED wiring: end ids governed, fail closed to EOS. */
        hook.end_allowed = o.json ? json_end : xml_end;
        hook.fail_closed = 1;
        hook.forced_token = T_EOS;
    }
    /* sg_tg_attach_batched wires the injected-end callback for XML only. */
    if (!o.json) hook.reasoning_end_injected = xml_injected;
    void (*adv)(void *, const char *) = o.json ? json_advance : xml_advance;
    if (o.driver_advance) hook.advance = adv;

    /* The reasoning, ending on a line (reasoning_budget.h's close window). */
    adv(st, "So the divisors of 96 are ..., ..., ...\n");
    if (o.forced_close) {
        /* The close is injected, never sampled: both paths advance the walk
         * over it themselves, then say it was injected. */
        adv(st, TEXT[T_CLOSE]);
        if (o.driver_advance) {
            if (hook.reasoning_end_injected) hook.reasoning_end_injected(st);
        } else if (o.json) {
            tgj_reasoning_end(st);
        } else {
            tg_reasoning_end_injected(st);
        }
    } else {
        /* The model's own close: sampled and advanced like any token; the
         * single-user loop also calls sg_tg_reasoning_ended off the emitter. */
        adv(st, TEXT[T_CLOSE]);
        if (!o.driver_advance) {
            if (o.json) tgj_reasoning_end(st);
            else tg_reasoning_end(st);
        }
    }

    pn_sampling_params sp = pn_sampling_defaults();
    sp.temperature = 0.0f;
    sp.top_k = 1;
    sp.tool_grammar = &hook;
    sp.rng_seeded = 1;
    sp.rng_state = 0x9E3779B97F4A7C15ull;

    const uint32_t *call = o.json ? JSON_CALL : XML_CALL;
    int n_call = o.json ? 3 : 7;
    int call_at = 0;           /* call tokens emitted so far */
    bool seen_text = false;
    for (int step = 0; step < max_tokens; step++) {
        float row[VOCAB];
        for (int i = 0; i < VOCAB; i++) row[i] = -10.0f;
        if (call_at == 0) {
            /* Still writing on: prose first, whitespace next, the call after. */
            row[T_PROSE] = 10.0f;
            row[T_NL2] = 8.0f;
            row[T_SP] = 7.0f;
            row[T_NL] = 6.0f;
            row[call[0]] = 5.0f;
            row[T_EOS] = 4.0f;
        } else if (call_at < n_call) {
            row[call[call_at]] = 10.0f;
            row[T_PROSE] = 5.0f;
        } else {
            row[T_EOS] = 10.0f;
        }
        uint32_t tok = pn_sample_next_token(row, VOCAB, &sp, NULL, 0);
        if (!o.driver_advance) adv(st, tok_text(NULL, tok));
        r.steps++;
        if (tok == T_EOS) { r.ended = true; break; }
        const char *t = tok_text(NULL, tok);
        if (call_at < n_call && tok == call[call_at]) {
            call_at++;
            seen_text = true;
        } else if (!seen_text) {
            if (tok == T_NL2 || tok == T_SP || tok == T_NL) r.ws_before += (int)strlen(t);
            else { r.prose_before = true; seen_text = true; }
        }
        strncat(r.out, t, sizeof(r.out) - strlen(r.out) - 1);
        r.completed = o.json ? tgj_complete(st) : tg_complete(st);
        if (r.completed) break;
    }
    if (o.json) tgj_state_free(st);
    else tg_state_free(st);
    return r;
}

static void test_scripted(const void *g, bool json) {
    const char *dialect = json ? "json" : "xml";
    for (int fc = 0; fc < 2; fc++) {
        for (int da = 0; da < 2; da++) {
            run_opts o = { json, fc != 0, da != 0, true };
            run_result r = run(g, o, 300);
            const char *close = fc ? "forced" : "natural";
            const char *path = da ? "batched" : "single-user";
            CHECK(r.completed, "%s required %s close %s: no call in %d steps (ended=%d), got \"%.120s\"",
                  dialect, close, path, r.steps, r.ended, r.out);
            CHECK(!r.prose_before, "%s required %s close %s: prose before the call: \"%.120s\"",
                  dialect, close, path, r.out);
            CHECK(r.ws_before <= WS_MAX, "%s required %s close %s: %d whitespace chars before the call",
                  dialect, close, path, r.ws_before);
        }
    }
    /* AUTO, forced close: unchanged — the model writes its prose. */
    for (int da = 0; da < 2; da++) {
        run_opts o = { json, true, da != 0, false };
        run_result r = run(g, o, 20);
        CHECK(r.prose_before && strncmp(r.out, TEXT[T_PROSE], strlen(TEXT[T_PROSE])) == 0,
              "%s auto forced close %s: prose should stay free, got \"%.120s\"", dialect,
              da ? "batched" : "single-user", r.out);
    }
}

int main(void)
{
    jr_value *tools = jr_parse(TOOLS);
    tg_grammar *g = tg_build(tools);
    tgj_grammar *gj = tgj_build(tools);
    if (!g || !gj) {
        printf("FAIL: grammar build\n");
        return 1;
    }

    test_xml_entry(g);
    test_json_entry(gj);
    test_scripted(g, false);
    test_scripted(gj, true);

    tg_free(g);
    tgj_free(gj);
    jr_free(tools);
    printf("test_tool_grammar_forced_close_call: %d/%d checks passed\n", g_checks - g_fail, g_checks);
    return g_fail ? 1 : 0;
}
