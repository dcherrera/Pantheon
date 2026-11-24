/* SPDX-License-Identifier: AGPL-3.0-or-later */
/* Copyright (C) 2026 David (Corey) Herrera */
/**
 * @file test_serve_sched_session_switch.c
 * @brief Session-switch gate: conversations that TAKE TURNS on the batched
 *        scheduler must each resume warm, and every resumed reply must be
 *        token-identical to a cold full prefill (greedy).
 *
 * Two conversations A and B (distinct session ids, each with a long system
 * block so the system-prefix rung is distinguishable from a warm resume) run
 * STRICTLY SEQUENTIALLY through one scheduler: A1 B1 A2 B2 A3 B3.  Only one
 * slot is ever active, so every turn compacts to GPU ring row 0 — the layout
 * that used to overwrite the idle conversation's KV.
 *
 * The same turns also run in SOLO order (A1 A2 A3 B1 B2 B3, fresh scheduler):
 * each conversation resumes with nothing in between — today's single-session
 * warm path — and against a non-cached cold reference (fresh pn_state, full
 * prefill).
 *
 * Gates (switch pass, each turn t >= 2):
 *   WARM  cached_prompt_tokens >= the previous turn's prompt_tokens (a warm
 *         resume keeps the whole previous prompt; the system-prefix fallback
 *         keeps only the system block, so it fails this);
 *   EXACT content + reasoning byte-identical to the SOLO pass — switching
 *         changes nothing about a resumed turn.
 * Reported, not gated: == cold. The think-included resume rung resumes on the
 * ids the model SAMPLED where a cold prefill re-tokenizes the replayed text,
 * so a turn can differ from cold in BOTH orders; the solo column shows it.
 *
 * Red: break the park (qw_kv_slot_park_occupant) and EXACT/WARM must fail.
 *
 * Build (CUDA host): make USE_CUDA=1 tests/test_serve_sched_session_switch
 * Run:  QWEN35_BATCH_SLOTS=1 QWEN35_KV_Q4=1 QWEN35_GPU_SPLIT_LAYER=20 \
 *         QWEN35_GPU_DIE_A=0 QWEN35_GPU_DIE_B=1 \
 *         ./tests/test_serve_sched_session_switch <model.gguf> [max_tokens]
 * max_tokens (default 512) must let each think block close, or the replayed
 * reasoning cannot match the KV and nothing resumes warm.
 * NOT part of the CPU test suite.
 *
 * @status Complete.
 * @issues None
 * @todo None
 */
#include "../include/pantheon.h"
#include "../src/core/engine.h"
#include "../src/serve/batch_sched.h"
#include "../src/serve/serve_generation.h"
#include <stdbool.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#ifndef USE_CUDA
int main(void)
{
    fprintf(stderr, "built without USE_CUDA\n");
    return 1;
}
#else

#define N_CONVOS 2
#define N_TURNS 3
#define SYS_LINES 120

static const char *user_line(int conv, int turn)
{
    static const char *lines[N_CONVOS][N_TURNS] = {
        { "How many items does Line 4 count? One number.",
          "And Line 5? One number.",
          "And Line 6? One number." },
        { "How many items does Row 9 count? One number.",
          "And Row 10? One number.",
          "And Row 11? One number." },
    };
    return lines[conv][turn];
}

/* A long, conversation-specific system block (~2k tokens). */
static char *system_block(int conv)
{
    size_t cap = (size_t)SYS_LINES * 80 + 128;
    char *s = malloc(cap);
    if (!s)
        return NULL;
    size_t n = (size_t)snprintf(s, cap, "%s\n", conv == 0
        ? "You are a careful assistant. Reference ledger A:"
        : "You are a precise analyst. Reference registry B:");
    for (int i = 0; i < SYS_LINES && n < cap; i++)
        n += (size_t)snprintf(s + n, cap - n, "%s %d: the %s counts %d items.\n",
                              conv == 0 ? "Line" : "Row", i,
                              conv == 0 ? "harbor ledger" : "orchard registry",
                              i * (conv == 0 ? 7 : 11));
    return s;
}

typedef struct {
    sg_message msgs[1 + 2 * N_TURNS];
    int        n;
    char      *owned[1 + 3 * N_TURNS];
    int        n_owned;
} convo;

static void convo_init(convo *c, int conv)
{
    memset(c, 0, sizeof(*c));
    c->owned[c->n_owned++] = system_block(conv);
    c->msgs[0].role = "system";
    c->msgs[0].content = c->owned[0];
    c->n = 1;
}

/* `reasoning` (assistant turns) is replayed the way a real client sends
 * reasoning_content back — without it the think block the KV holds never
 * matches the next prompt and no turn can resume warm, fix or no fix. */
static void convo_add(convo *c, const char *role, const char *text,
                      const char *reasoning)
{
    char *copy = strdup(text ? text : "");
    c->owned[c->n_owned++] = copy;
    c->msgs[c->n].role = role;
    c->msgs[c->n].content = copy;
    c->msgs[c->n].reasoning = NULL;
    if (reasoning && reasoning[0]) {
        char *r = strdup(reasoning);
        c->owned[c->n_owned++] = r;
        c->msgs[c->n].reasoning = r;
    }
    c->n++;
}

static void convo_free(convo *c)
{
    for (int i = 0; i < c->n_owned; i++)
        free(c->owned[i]);
}

typedef struct {
    char *content;
    char *reasoning;
    int   prompt;
    int   cached;
} turn_out;

static void gen_params(sg_gen_params *gp, int max_tokens)
{
    memset(gp, 0, sizeof(*gp));
    gp->temperature = 0.0f;   /* greedy — argmax-exact regime */
    gp->top_p = 1.0f;
    gp->max_tokens = max_tokens;
}

static void keep(turn_out *o, const sg_result *res)
{
    o->content = strdup(res->content ? res->content : "");
    o->reasoning = strdup(res->reasoning ? res->reasoning : "");
    o->prompt = res->prompt_tokens;
    o->cached = res->cached_prompt_tokens;
}

/* Non-cached reference: fresh state per turn, full prefill, no session. */
static int reference_turns(pn_model *model, int conv, int max_tokens,
                           turn_out out[N_TURNS])
{
    convo c;
    convo_init(&c, conv);
    sg_gen_params gp;
    gen_params(&gp, max_tokens);
    for (int t = 0; t < N_TURNS; t++) {
        convo_add(&c, "user", user_line(conv, t), NULL);
        pn_state *st = pn_state_new(model);
        sg_result res;
        memset(&res, 0, sizeof(res));
        int rc = st ? sg_run_completion(model, st, c.msgs, c.n, &gp, NULL, NULL,
                                        &res, NULL) : -1;
        pn_state_free(st);
        if (rc != 0) {
            sg_result_free(&res);
            convo_free(&c);
            return rc;
        }
        keep(&out[t], &res);
        convo_add(&c, "assistant", res.content, res.reasoning);
        sg_result_free(&res);
    }
    convo_free(&c);
    return 0;
}

/* One cached batched turn of conversation `c` under session `sid`. */
static int switch_turn(pn_model *model, pn_batch_sched *sched, convo *c,
                       int conv, int t, const char *sid, int max_tokens,
                       turn_out *out)
{
    sg_gen_params gp;
    gen_params(&gp, max_tokens);
    convo_add(c, "user", user_line(conv, t), NULL);
    int slot = bs_slot_acquire_biased(sched, sid);
    if (slot < 0)
        return -100;
    sg_result res;
    memset(&res, 0, sizeof(res));
    int rc = sg_run_completion_batched(model, sched, slot, c->msgs, c->n, &gp,
                                       NULL, NULL, &res, sid);
    if (rc == 0) {
        keep(out, &res);
        convo_add(c, "assistant", res.content, res.reasoning);
    }
    sg_result_free(&res);
    return rc;
}

static bool same_output(const turn_out *a, const turn_out *b)
{
    return strcmp(a->content, b->content) == 0 &&
           strcmp(a->reasoning, b->reasoning) == 0;
}

/* Run all N_CONVOS x N_TURNS turns on a FRESH scheduler, in switch order
 * (A1 B1 A2 B2 A3 B3) or solo order (A1 A2 A3 B1 B2 B3). Returns 0 on success. */
static int run_pass(pn_model *model, bool solo, const char *sid_prefix,
                    int max_tokens, turn_out out[N_CONVOS][N_TURNS])
{
    pn_batch_sched *sched = bs_create(model, N_CONVOS, "switch-gate");
    if (!sched)
        return -1;
    convo c[N_CONVOS];
    char sid[N_CONVOS][32];
    for (int v = 0; v < N_CONVOS; v++) {
        convo_init(&c[v], v);
        snprintf(sid[v], sizeof(sid[v]), "%s-%c", sid_prefix, "ab"[v]);
    }
    int rc = 0;
    for (int k = 0; k < N_TURNS * N_CONVOS && rc == 0; k++) {
        int v = solo ? k / N_TURNS : k % N_CONVOS;
        int t = solo ? k % N_TURNS : k / N_CONVOS;
        rc = switch_turn(model, sched, &c[v], v, t, sid[v], max_tokens,
                         &out[v][t]);
        if (rc != 0)
            fprintf(stderr, "%s %c%d: rc=%d\n", sid_prefix, "AB"[v], t + 1, rc);
    }
    for (int v = 0; v < N_CONVOS; v++)
        convo_free(&c[v]);
    bs_destroy(sched);
    return rc;
}

static void free_turns(turn_out t[N_CONVOS][N_TURNS])
{
    for (int v = 0; v < N_CONVOS; v++)
        for (int k = 0; k < N_TURNS; k++) {
            free(t[v][k].content);
            free(t[v][k].reasoning);
        }
}

int main(int argc, char **argv)
{
    if (argc < 2) {
        fprintf(stderr, "usage: %s <model.gguf> [max_tokens]\n", argv[0]);
        return 1;
    }
    pn_model *model = pn_load(argv[1]);
    if (!model) {
        fprintf(stderr, "load failed\n");
        return 1;
    }
    int max_tokens = argc > 2 ? atoi(argv[2]) : 512;
    if (max_tokens < 4)
        max_tokens = 4;

    turn_out ref[N_CONVOS][N_TURNS], solo[N_CONVOS][N_TURNS];
    turn_out sw[N_CONVOS][N_TURNS];
    memset(ref, 0, sizeof(ref));
    memset(solo, 0, sizeof(solo));
    memset(sw, 0, sizeof(sw));
    for (int v = 0; v < N_CONVOS; v++)
        if (reference_turns(model, v, max_tokens, ref[v]) != 0) {
            fprintf(stderr, "reference conversation %d failed\n", v);
            return 1;
        }
    if (run_pass(model, true, "solo", max_tokens, solo) != 0 ||
        run_pass(model, false, "switch", max_tokens, sw) != 0) {
        fprintf(stderr, "\nGATE: FAIL (a turn errored)\n");
        return 1;
    }

    int failed = 0, cold_diff = 0;
    for (int t = 0; t < N_TURNS; t++)
        for (int v = 0; v < N_CONVOS; v++) {
            const turn_out *g = &sw[v][t];
            bool warm = t == 0 || g->cached >= sw[v][t - 1].prompt;
            bool exact = same_output(g, &solo[v][t]);
            bool cold = same_output(g, &ref[v][t]);
            fprintf(stderr, "%c%d: prompt=%d cached=%d %s, %s solo, %s cold\n",
                    "AB"[v], t + 1, g->prompt, g->cached,
                    t == 0 ? "first turn" : warm ? "WARM" : "NOT-WARM",
                    exact ? "==" : "!=", cold ? "==" : "!=");
            if (!exact || !cold)
                fprintf(stderr, "  cold  : %.400s\n  solo  : %.400s\n"
                        "  switch: %.400s\n", ref[v][t].reasoning,
                        solo[v][t].reasoning, g->reasoning);
            failed |= !warm || !exact;
            cold_diff += !cold;
        }
    free_turns(ref);
    free_turns(solo);
    free_turns(sw);
    fprintf(stderr, "\nGATE: %s (%d switch turn(s) differ from cold; the solo "
            "pass shows whether the resume rung itself does)\n",
            failed ? "FAIL" : "PASS: every switched resume warm and == solo",
            cold_diff);
    fflush(stderr);
    if (!getenv("QWEN35_EXIT_CLEAN"))
        _Exit(failed);
    return failed;
}
#endif
