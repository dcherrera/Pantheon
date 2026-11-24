/* SPDX-License-Identifier: AGPL-3.0-or-later */
/* Copyright (C) 2026 David (Corey) Herrera */
/**
 * @file test_think_flip_backlog.c
 * @brief The batched owner flips its emitter out of <think> at the token where the
 *        grammar's forced reasoning ended, however far the driver has run ahead.
 *
 * @status Complete.  CPU-only: no device, no model.
 * @issues None
 * @todo None
 *
 * The defect (K80 host, Qwen3.6 TP2, REQUIRED tool turns under batched MTP,
 * 2026-11-20): the owner asked the LIVE grammar walk "has forced reasoning
 * ended?" after each token it fed.  The driver advances that walk when it
 * deposits, so with a multi-token deposit in the ring the walk was already past
 * "</think>" while the owner was still feeding the reasoning before it.  The
 * channel flipped early, and "\n</think>\n\n" landed in `content`: same tokens,
 * different response.
 *
 * Here: a hand-built scheduler (the test_decode_watchdog pattern), a real
 * "required" XML walk as the slot's hook, the real deposit (bs_slot_deposit_token
 * = the driver's driver_emit_token) and watched pop, and a real emitter in token
 * mode.  The driver deposits the turn in chunks, then the owner pops each token,
 * feeds it, and applies the coupling rule.  Every chunking (first chunk 1..k,
 * then k, for k = 1..6) must give the lockstep (k = 1) channels.  The pre-fix
 * rule (read the live walk) must not, which shows this reproduces the defect.
 * `--red`: PN_THINK_FLIP_RED=1 drops the position check; it must be caught.
 */

#include "../src/serve/batch_sched.h"
#include "../src/serve/batch_sched_watchdog.h"
#include "../src/serve/json_reader.h"
#include "../src/serve/serve_emitter.h"
#include "../src/serve/tool_grammar.h"

#include <pthread.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static const char TOOLS[] =
    "[{\"type\":\"function\",\"function\":{\"name\":\"read\",\"parameters\":{"
    "\"type\":\"object\",\"properties\":{"
    "\"path\":{\"type\":\"string\",\"minLength\":1}},"
    "\"required\":[\"path\"]}}}]";

/* One REQUIRED tool turn as the model's tokens; token id = index. Token 0 is the
 * owner's own first token (sampled from the prefill logits, never in the ring). */
static const char *const TURN[] = {
    "Now let me check", " the worktree", " exists:", "\n", "</think>", "\n\n",
    "<tool_call>", "\n", "<function=read>", "\n", "<parameter=path>", "\n", "a.txt",
    "\n", "</parameter>", "\n", "</function>", "\n", "</tool_call>",
};
#define N_TURN ((int)(sizeof(TURN) / sizeof(TURN[0])))

enum { RULE_FIXED, RULE_LIVE_WALK };

typedef struct {
    char reasoning[512];
    char content[512];
    int  flip_at;   /* tokens fed when the coupling flipped (-1: never) */
} channels;

static int walk_complete(void *st) { return tg_complete((const tg_state *)st) ? 1 : 0; }
static int walk_ended(void *st) { return tg_forced_reasoning_ended((const tg_state *)st) ? 1 : 0; }

static void copy_channel(char *dst, size_t cap, const text_channel *c)
{
    size_t n = c->len < cap - 1 ? c->len : cap - 1;
    if (n) memcpy(dst, c->data, n);
    dst[n] = '\0';
}

/* One turn: the driver deposits `first` tokens, then `k` at a time; the owner drains
 * each deposit token by token.  Returns 0, or -1 on a scheduler error. */
static int run_turn(const tg_grammar *g, int first, int k, int rule, channels *out)
{
    static pn_batch_sched sched;
    pn_tool_grammar_hook hook;
    emitter em;
    tg_state *walk = tg_state_new_required(g, true);
    int fed = 0, rc = 0;

    memset(&sched, 0, sizeof(sched));
    pthread_mutex_init(&sched.mu, NULL);
    pthread_cond_init(&sched.slots[0].cv, NULL);
    sched.cap = 1;
    sched.eos_id = sched.im_end_id = 1u << 30;
    bs_slot *sl = &sched.slots[0];
    sl->status = BS_SLOT_DECODE;
    sl->ring = calloc(N_TURN, sizeof(uint32_t));
    sl->ring_cap = N_TURN;
    sl->budget = N_TURN;
    memset(&hook, 0, sizeof(hook));
    hook.st = walk;
    hook.complete = walk_complete;
    hook.reasoning_ended = walk_ended;
    sl->sp.tool_grammar = &hook;
    tg_set_think_close_token(walk);

    em_init(&em, NULL, NULL, NULL, 0, true, TC_DIALECT_QWEN_XML);
    em_set_think_close_token(&em);
    out->flip_at = -1;

    /* Token 0: the owner samples, advances and feeds it before the driver runs. */
    tg_advance(walk, TURN[0]);
    em_feed(&em, TURN[0]);
    fed = 1;
    for (int next = 1; next < N_TURN && rc == 0;) {
        const int chunk = next == 1 ? first : k;
        const int end = next + chunk < N_TURN ? next + chunk : N_TURN;

        pthread_mutex_lock(&sched.mu);   /* the driver: sample (advance), deposit */
        for (int t = next; t < end; t++) {
            tg_advance(walk, TURN[t]);
            if (bs_slot_deposit_token(&sched, sl, (uint32_t)t) < 0)
                rc = -1;
        }
        pthread_mutex_unlock(&sched.mu);
        for (; next < end && rc == 0; next++) {   /* the owner: pop, feed, couple */
            uint32_t tok = 0;
            bs_pop_watch w = { NULL, NULL, 10, 1000 };

            if (bs_slot_pop_token_watched(&sched, 0, &tok, &w) != BS_POP_TOKEN) {
                rc = -1;
                break;
            }
            em_feed(&em, TURN[tok]);
            fed++;
            const int ended = rule == RULE_FIXED ? bs_slot_reasoning_ended(&sched, 0)
                                                 : tg_forced_reasoning_ended(walk);
            if (em.in_think && ended) {
                em.in_think = 0;
                out->flip_at = fed;
            }
        }
    }
    em_flush(&em);
    copy_channel(out->reasoning, sizeof(out->reasoning), &em.reasoning);
    copy_channel(out->content, sizeof(out->content), &em.content);
    em_free(&em);
    free(sl->ring);
    tg_state_free(walk);
    return rc;
}

static int same(const channels *a, const channels *b)
{
    return strcmp(a->reasoning, b->reasoning) == 0 && strcmp(a->content, b->content) == 0;
}

/* Every chunking under `rule`; returns how many differ from `ref` (or -1 on error). */
static int sweep(const tg_grammar *g, int rule, const channels *ref, const char *label)
{
    int runs = 0, differ = 0;

    for (int k = 1; k <= 6; k++)
        for (int first = 1; first <= k; first++) {
            channels c;
            if (run_turn(g, first, k, rule, &c) != 0)
                return -1;
            runs++;
            if (!same(&c, ref) && ++differ <= 2)
                printf("  %s k %d first %d: reasoning tail \"%s\" content \"%s\"\n", label, k,
                       first, c.reasoning + (strlen(c.reasoning) > 12 ? strlen(c.reasoning) - 12 : 0),
                       c.content);
        }
    printf("%s: %d chunkings, %d differ from lockstep\n", label, runs, differ);
    return differ;
}

int main(int argc, char **argv)
{
    const int red = argc > 1 && strcmp(argv[1], "--red") == 0;
    jr_value *tools = jr_parse(TOOLS);
    tg_grammar *g = tools ? tg_build(tools) : NULL;
    channels ref;
    int fail = 0;

    if (red)
        setenv("PN_THINK_FLIP_RED", "1", 1);
    if (!g || run_turn(g, 1, 1, RULE_FIXED, &ref) != 0) {
        printf("FAIL: setup\n");
        return 1;
    }
    /* Lockstep: the close ends <think> itself; the reasoning keeps its "\n". */
    if (strstr(ref.content, "</think>") || strcmp(ref.reasoning,
            "Now let me check the worktree exists:\n") != 0) {
        printf("FAIL: lockstep reference: reasoning \"%s\" content \"%s\"\n", ref.reasoning,
               ref.content);
        fail = 1;
    }
    const int fixed = sweep(g, RULE_FIXED, &ref, red ? "fixed rule (red)" : "fixed rule");
    if (red) {
        printf(fixed > 0 ? "SELFTEST PASS (red build caught)\n"
                         : "SELFTEST FAIL (red build not caught)\n");
        tg_free(g);
        jr_free(tools);
        return fixed > 0 ? 0 : 1;
    }
    const int live = sweep(g, RULE_LIVE_WALK, &ref, "pre-fix rule (live walk)");
    if (fixed != 0)
        fail = 1;
    if (live <= 0) {
        printf("FAIL: the pre-fix rule never misroutes: the test does not reproduce the defect\n");
        fail = 1;
    }
    tg_free(g);
    jr_free(tools);
    printf(fail ? "FAIL\n" : "PASS\n");
    return fail;
}
