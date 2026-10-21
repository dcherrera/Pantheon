/* SPDX-License-Identifier: AGPL-3.0-or-later */
/* Copyright (C) 2026 David (Corey) Herrera */
/**
 * @file test_grammar_arm_window.c
 * @brief Gate for the full-logits ARMING WINDOW (PN_GRAMMAR_ARM_WINDOW).
 *
 * @status Proven red twice — see "PROOF THIS GATE CAN FAIL" below.
 * @issues None
 * @todo None
 *
 * WHAT IS BEING GATED
 * ===================
 * The serve arms the full-vocabulary logits row for the steps a grammar mask is
 * live on, instead of for the whole turn.  The invariant that has to survive:
 *
 *     On every step whose sampler evaluates the grammar mask, the row it reads
 *     is the FULL vocabulary row.
 *
 * A missed mask is not a slow step.  The mask can only choose among logits it
 * can see, so a sparse top-k row holding none of the legal tokens has zero
 * survivors and the walk fails open (AUTO) or force-terminates (FORCED) — the
 * response_format defect, reintroduced.
 *
 * THE HAZARD IS EXACTLY ONE TOKEN, AND IT IS THE ONLY THING THIS TEST IS ABOUT
 * ===========================================================================
 * Within one decode step the real loops do, in order:
 *
 *     refresh()  ->  mask + sample  ->  advance(chosen token)  ->  FORWARD
 *
 * `refresh()` reads the walk as it stood after the PREVIOUS token; the forward
 * at the bottom fetches the row the NEXT step will sample.  So the value that
 * has to be known at forward time is the walk AFTER this step's advance —
 * arming from the value `refresh()` produced (which is what `hook.active`
 * holds) is stale by exactly one token, and the token it is stale on is the
 * opener that engages the grammar.
 *
 * This file therefore models the loop ORDERING, not the sampler: it drives the
 * real grammar walks over scripted token text and asks, per step, "was the row
 * this step masked against armed?".  Two policies are simulated:
 *
 *   POLICY_AFTER_ADVANCE (shipped)  armed[T+1] = mask_active(after token T)
 *   POLICY_STALE         (the bug)  armed[T+1] = mask_active(before token T)
 *
 * PROOF THIS GATE CAN FAIL
 * ========================
 * Every scenario asserts all three of:
 *   1. the shipped policy misses ZERO masked steps;
 *   2. the stale policy MISSES AT LEAST ONE — and the first miss is exactly the
 *      transition step (the first step whose mask is active).  If the stale
 *      policy came out clean the script would not be exercising a transition at
 *      all, so the run exits VACUOUS (2) instead of passing;
 *   3. the shipped policy leaves at least one step UNARMED — otherwise the
 *      window never narrows and the change is worth nothing on this script, so
 *      a "green" would again be measuring nothing.
 *
 * (2) is the anti-vacuity assertion the July sparse-sampler failure was missing:
 * a bit-exactness gate that never runs the fast path passes 18/18 while testing
 * the old path against itself.
 *
 * Pure C, libc-only — links the three grammar modules + the JSON reader.  No
 * engine, no model, no CUDA.
 */

#include "lib/test.h"
#include "../src/serve/tool_grammar.h"
#include "../src/serve/tool_grammar_json.h"
#include "../src/serve/response_format_grammar.h"
#include "../src/serve/json_reader.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

/* get_weather(city) — one tool, enough for both dialects. */
static const char *TOOLS_JSON =
    "[{\"type\":\"function\",\"function\":{\"name\":\"get_weather\","
    "\"parameters\":{\"type\":\"object\",\"properties\":{"
    "\"city\":{\"type\":\"string\"}},\"required\":[\"city\"]}}}]";

/* ── The walk under test, behind one vtable so one simulator drives all three
 * grammars exactly as sg_tg_mask_active / sg_tg_step do. ── */
typedef struct {
    void *st;
    void (*advance)(void *st, const char *text);
    int  (*mask_active)(void *st);
    void (*free_st)(void *st);
} walk;

static void  w_adv_xml(void *st, const char *t) { tg_advance((tg_state *)st, t); }
static int   w_act_xml(void *st) { return tg_mask_active((tg_state *)st) ? 1 : 0; }
static void  w_free_xml(void *st) { tg_state_free((tg_state *)st); }
static void  w_adv_json(void *st, const char *t) { tgj_advance((tgj_state *)st, t); }
static int   w_act_json(void *st) { return tgj_mask_active((tgj_state *)st) ? 1 : 0; }
static void  w_free_json(void *st) { tgj_state_free((tgj_state *)st); }
static void  w_adv_rf(void *st, const char *t) { rfg_advance((rfg_state *)st, t); }
static int   w_act_rf(void *st) { return rfg_mask_active((rfg_state *)st) ? 1 : 0; }
static void  w_free_rf(void *st) { rfg_state_free((rfg_state *)st); }

#define MAX_STEPS 64

/* One simulated turn's per-step record. */
typedef struct {
    int n;
    int masked[MAX_STEPS];   /* the sampler evaluated the mask on this step */
    int armed[MAX_STEPS];    /* the row this step read was the FULL row */
} run;

typedef enum { POLICY_AFTER_ADVANCE = 0, POLICY_STALE = 1 } arm_policy;

/**
 * Replay `toks` through `w`, recording per step whether the mask was evaluated
 * and whether the row that step read had been armed under `policy`.
 *
 * Mirrors the real decode loops step for step: the row a step samples was
 * fetched by the PREVIOUS step's forward, so armed[T+1] is decided at the
 * bottom of step T — after the advance under the shipped policy, before it
 * under the stale one.
 *
 * @param w      the grammar walk (consumed: this drives it to the end)
 * @param toks   NULL-terminated token TEXT stream
 * @param policy which arming ordering to simulate
 * @param out    filled with the per-step record
 */
static void simulate(walk *w, const char *const *toks, arm_policy policy,
                     run *out)
{
    memset(out, 0, sizeof(*out));
    /* The turn's first row is armed before any token exists: both real paths
     * arm it from the fresh walk. */
    int armed_next = w->mask_active(w->st);
    for (int t = 0; toks[t] && t < MAX_STEPS; t++) {
        out->armed[t]  = armed_next;
        /* refresh() at the top of the step — what the sampler will gate on. */
        int active_now = w->mask_active(w->st);
        out->masked[t] = active_now;
        out->n = t + 1;
        w->advance(w->st, toks[t]);
        /* Bottom of the step: arm the row the NEXT step reads. */
        armed_next = (policy == POLICY_STALE) ? active_now
                                              : w->mask_active(w->st);
    }
}

/* Steps whose mask was evaluated against a row that was NOT armed. */
static int count_misses(const run *r)
{
    int n = 0;
    for (int i = 0; i < r->n; i++)
        if (r->masked[i] && !r->armed[i]) n++;
    return n;
}

static int first_miss(const run *r)
{
    for (int i = 0; i < r->n; i++)
        if (r->masked[i] && !r->armed[i]) return i;
    return -1;
}

/* The transition step: the first step on which the mask is live. */
static int first_masked(const run *r)
{
    for (int i = 0; i < r->n; i++)
        if (r->masked[i]) return i;
    return -1;
}

static int count_unarmed(const run *r)
{
    int n = 0;
    for (int i = 0; i < r->n; i++)
        if (!r->armed[i]) n++;
    return n;
}

/* Set by any scenario whose STALE arm came out clean — the script did not
 * exercise a transition, so nothing was proved and main() exits VACUOUS. */
static int g_vacuous = 0;
static const char *g_vacuous_why = NULL;

/**
 * Run one scenario under both policies and assert the three gate conditions.
 *
 * @param name    scenario label, printed on failure
 * @param mk      builds a FRESH walk (called once per policy — the walk is
 *                consumed by the simulation)
 * @param ud      passed to `mk`
 * @param toks    the token text stream
 */
static void check_scenario(const char *name, walk (*mk)(void *ud), void *ud,
                           const char *const *toks)
{
    run good, stale;
    walk w = mk(ud);
    simulate(&w, toks, POLICY_AFTER_ADVANCE, &good);
    w.free_st(w.st);
    w = mk(ud);
    simulate(&w, toks, POLICY_STALE, &stale);
    w.free_st(w.st);

    /* 1. The invariant. */
    if (count_misses(&good) != 0)
        printf("    [%s] shipped policy MISSED the mask on step %d "
               "(%d misses of %d steps)\n",
               name, first_miss(&good), count_misses(&good), good.n);
    ASSERT_INT_EQ(0, count_misses(&good));

    /* 2. Anti-vacuity: the stale ordering must break, at the transition step. */
    if (count_misses(&stale) == 0) {
        g_vacuous = 1;
        g_vacuous_why = name;
        printf("    [%s] VACUOUS: the stale policy missed nothing — this "
               "script never transitions, so it tests nothing\n", name);
    }
    ASSERT_TRUE(count_misses(&stale) > 0);
    if (first_miss(&stale) != first_masked(&stale))
        printf("    [%s] stale first miss %d != transition step %d\n",
               name, first_miss(&stale), first_masked(&stale));
    ASSERT_INT_EQ(first_masked(&stale), first_miss(&stale));

    /* 3. The window must actually narrow, or "green" means nothing. */
    if (count_unarmed(&good) == 0)
        printf("    [%s] window never narrowed — every step armed\n", name);
    ASSERT_TRUE(count_unarmed(&good) > 0);

    printf("    [%s] %d steps, %d unarmed (the win), shipped misses 0, "
           "stale misses %d first at step %d (transition %d)\n",
           name, good.n, count_unarmed(&good), count_misses(&stale),
           first_miss(&stale), first_masked(&stale));
}

/* ── walk factories ── */
static walk mk_xml_auto(void *ud) {
    walk w = { tg_state_new((const tg_grammar *)ud), w_adv_xml, w_act_xml,
               w_free_xml };
    return w;
}
static walk mk_xml_forced(void *ud) {
    walk w = { tg_state_new_forced((const tg_grammar *)ud, NULL), w_adv_xml,
               w_act_xml, w_free_xml };
    return w;
}
static walk mk_json_auto(void *ud) {
    walk w = { tgj_state_new((const tgj_grammar *)ud), w_adv_json, w_act_json,
               w_free_json };
    return w;
}
static walk mk_rf(void *ud) {
    (void)ud;
    walk w = { rfg_state_new(RFG_FORMAT_JSON_OBJECT, true), w_adv_rf, w_act_rf,
               w_free_rf };
    return w;
}

/* ── Token streams.  Multi-char pieces on purpose: real BPE tokens are, and the
 * transition lands INSIDE a piece, which is precisely the case a
 * character-granular argument would miss. ── */

/* AUTO + XML: the model reasons, closes, answers a little prose, then opens a
 * call.  This is the shape the client produces on essentially every request. */
static const char *const XML_AUTO[] = {
    "Let", " me", " check", " the", " weather", ".", "</think>",
    "Sure", ",", " one", " moment", ".",
    "<tool_call>", "<function=", "get_weather", ">", "<parameter=", "city",
    ">", "Paris", "</parameter>", "</function>", "</tool_call>", NULL
};

/* FORCED + XML: the mask engages off reasoning-end, with no opener token at
 * all — a different rising edge than in_call, and it must be caught too. */
static const char *const XML_FORCED[] = {
    "Think", "ing", " about", " it", "</think>",
    "<function=", "get_weather", ">", "<parameter=", "city", ">", "Paris",
    "</parameter>", "</function>", NULL
};

/* AUTO + ornith JSON: the fenced dialect. */
static const char *const JSON_AUTO[] = {
    "Check", "ing", ".", "</think>", "Here", " goes", ":", "\n",
    "```json", "\n", "{\"function\":", "\"get_weather\"", ",",
    "\"arguments\":", "{", "\"city\"", ":", "\"Paris\"", "}", "}", "\n",
    "```", NULL
};

/* response_format json_object: the mask engages when reasoning ends, with the
 * whole ANSWER constrained rather than a call. */
static const char *const RF_STREAM[] = {
    "The", " user", " wants", " json", ".", "</think>",
    "{", "\"a\"", ":", " 1", "}", NULL
};

static tg_grammar  *g_xml;
static tgj_grammar *g_json;

static void test_xml_auto_transition(void) {
    check_scenario("xml/auto", mk_xml_auto, g_xml, XML_AUTO);
}
static void test_xml_forced_transition(void) {
    check_scenario("xml/forced", mk_xml_forced, g_xml, XML_FORCED);
}
static void test_json_auto_transition(void) {
    check_scenario("json/auto", mk_json_auto, g_json, JSON_AUTO);
}
static void test_response_format_transition(void) {
    check_scenario("response_format", mk_rf, NULL, RF_STREAM);
}

/* A turn that never calls a tool: the whole point of the window.  No step may
 * be armed, and neither policy can miss (nothing is ever masked). */
static void test_prose_turn_never_arms(void) {
    static const char *const PROSE[] = {
        "The", " capital", " of", " France", " is", " Paris", ".", "</think>",
        "The", " capital", " of", " France", " is", " Paris", ".", NULL
    };
    run r;
    walk w = mk_xml_auto(g_xml);
    simulate(&w, PROSE, POLICY_AFTER_ADVANCE, &r);
    w.free_st(w.st);
    ASSERT_INT_EQ(0, count_misses(&r));
    ASSERT_INT_EQ(r.n, count_unarmed(&r));   /* every step on the fast top-k path */
    printf("    [xml/prose] %d steps, 0 armed — the whole turn stays sparse\n",
           r.n);
}

/* The shipped policy's armed flag must equal what refresh() returns at the NEXT
 * step, everywhere, not merely imply it.  That equality is the mechanism's
 * actual claim: it is the same read, moved to the other side of the advance. */
static void test_armed_equals_next_refresh(void) {
    run r;
    walk w = mk_xml_auto(g_xml);
    simulate(&w, XML_AUTO, POLICY_AFTER_ADVANCE, &r);
    w.free_st(w.st);
    for (int t = 0; t < r.n; t++)
        ASSERT_INT_EQ(r.masked[t], r.armed[t]);
    printf("    [xml/auto] armed[T] == refresh()@T on all %d steps\n", r.n);
}

int main(void) {
    jr_value *tools = jr_parse(TOOLS_JSON);
    if (!tools) { fprintf(stderr, "tools schema failed to parse\n"); return 2; }
    g_xml  = tg_build(tools);
    g_json = tgj_build(tools);
    if (!g_xml || !g_json) {
        fprintf(stderr, "grammar build failed\n");
        return 2;
    }

    printf("\n=== grammar full-logits arming window ===\n");
    RUN_TEST(xml_auto_transition);
    RUN_TEST(xml_forced_transition);
    RUN_TEST(json_auto_transition);
    RUN_TEST(response_format_transition);
    RUN_TEST(prose_turn_never_arms);
    RUN_TEST(armed_equals_next_refresh);

    tg_free(g_xml);
    tgj_free(g_json);
    jr_free(tools);

    TEST_REPORT();
    if (g_vacuous) {
        fprintf(stderr, "\nVACUOUS: scenario '%s' could not break the stale "
                "arming policy — the gate proved nothing.\n", g_vacuous_why);
        return 2;
    }
    return __ft_failed == 0 ? 0 : 1;
}
