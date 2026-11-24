/* SPDX-License-Identifier: AGPL-3.0-or-later */
/* Copyright (C) 2026 David (Corey) Herrera */
/**
 * @file test_prefix_projection.c
 * @brief The prefix-cache agreement gate: what the serve PROJECTS for a turn
 *        must be exactly what the client's next request TOKENIZES TO.
 *
 * This is the gate that was missing when the K80 test machine measured
 * `prefilled_tok == prompt_tok - sys_len` on every turn of a live agent-client
 * conversation — TTFT 19 → 38 → 59
 * → 80 s over four turns, ~8 min at 16 k, because every rung of
 * bs_prefill_resume_sys's ladder missed and the whole history was re-prefilled.
 *
 * The property under test, stated as tokens:
 *
 *     sg_project_client_rerender(reply)  ==  the slice of the NEXT turn's
 *     prompt that the client's re-render of that reply occupies.
 *
 * And the same property for the REASONING channel, which is the half a client
 * that sends `reasoning_content` back depends on:
 *
 *     sg_encode_history_think_block(reasoning)  ==  the slice that turn's
 *     `<think>...</think>` occupies in the next prompt.
 *
 * Both sides are computed here for real — no model, no GPU: a synthetic
 * byte-level BPE tokenizer (one id per byte, plus the ChatML/think/tool-call
 * specials) stands in for the saber vocab, and the turn-2 prompt is built by the
 * production builder from the messages a faithful client would send.
 *
 * WHY THIS TEST CAN GO RED, and did.  Every case carries a NEGATIVE CONTROL: the
 * projection the OLD code recorded — the token ids the model actually SAMPLED
 * for its reply.  Those must NOT be a prefix of the next context, and they are
 * not, because "<tool_call>" has two legal tokenizations (one special id, or the
 * bytes) and the char-level tool grammar accepts either.  That is the production
 * defect, reproduced here in tokens; the positive assertion is the fix.
 *
 * @status Complete.  Runs on the Mac in the CPU `make test` suite.  Covers both
 *         recorded client conventions (reasoning dropped, reasoning resent) and
 *         the THIRD one the projection rung does not record — a summarised
 *         block, which matches neither and is why a summary costs a re-prefill.
 * @issues None
 * @todo None
 */

#include "lib/test.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "../src/core/bpe_pretokenize.h"
#include "../src/core/chat.h"
#include "../src/core/engine.h"
#include "../src/serve/json_reader.h"
#include "../src/serve/serve_prompt_tokens.h"
#include "../src/serve/tool_calling.h"
#include "tokenizer/tokenizer.h"

/* ============================================================
 * A synthetic serve tokenizer
 *
 * One id per byte (GPT-2 codepoint order, no merges) plus the special tokens
 * the ChatML builder looks up.  The contrast that matters is deliberate: the
 * text "<tool_call>" encodes to ELEVEN byte ids, while the special token is
 * ONE — the same two-tokenizations-of-one-string situation the saber vocab has
 * (248058 vs `< tool _call >`), scaled up so a divergence is unmissable.
 * ============================================================ */

static const char *const SPECIAL_TOKENS[] = {
    "<|im_start|>", "<|im_end|>", "<think>", "</think>",
    "<tool_call>", "</tool_call>",
};

static ft_tokenizer *make_serve_tok(void)
{
    ft_tokenizer *tok = ft_new();
    ft_set_model_type(tok, 0);   /* BPE — routes text through pn_bpe_encode */
    for (int b = 0; b < 256; b++) {
        char raw[2] = { (char)b, 0 };
        char *enc = gpt2_bytes_encode(raw);
        ft_add_token(tok, enc);
        free(enc);
    }
    for (size_t i = 0; i < sizeof(SPECIAL_TOKENS) / sizeof(SPECIAL_TOKENS[0]); i++)
        ft_add_token(tok, SPECIAL_TOKENS[i]);
    return tok;
}

/* The ids the MODEL produced for a span of its own output: plain text, because
 * that is what sampling emits when it walks the grammar's opener char by char.
 * This is what the old projection recorded, and it is the negative control. */
static uint32_t *sampled_ids(ft_tokenizer *tok, const char *text, int *n_out)
{
    int n = 0;
    int *ids = pn_bpe_encode(tok, text, &n);
    uint32_t *out = malloc((size_t)(n > 0 ? n : 1) * sizeof(uint32_t));
    for (int i = 0; i < n; i++)
        out[i] = (uint32_t)ids[i];
    free(ids);
    *n_out = n;
    return out;
}

/* ============================================================
 * The conversation under test
 * ============================================================ */

#define SYS_PROMPT  "You are a careful orchestrator."
#define USER_TURN_1 "What is item 1?"
#define USER_TURN_2 "And item 2?"

static const char TOOLS_JSON[] =
    "[{\"type\":\"function\",\"function\":{\"name\":\"attempt_completion\","
    "\"parameters\":{\"type\":\"object\",\"properties\":"
    "{\"result\":{\"type\":\"string\"}}}}}]";

/* Everything one turn needs to be replayed as the client would send it back. */
typedef struct {
    const char *name;          /* case label */
    const char *content;       /* the reply's visible content channel */
    const char *tool_raw;      /* the raw diverted tool block, or NULL */
} reply_case;

/* Re-render a reply into the assistant `content` string the SERVE will build
 * from the client's next request — i.e. what combine_assistant_tool_calls does
 * to the content and tool_calls array the client echoes back (it calls
 * tc_join_assistant_history).  Owned; caller frees. */
static char *client_assistant_content(const reply_case *rc, const jr_value *tools)
{
    if (!rc->tool_raw)
        return strdup(rc->content);
    tc_tool_call *calls = NULL;
    int n_calls = 0;
    if (tc_parse_tool_calls(rc->tool_raw, tools, TC_DIALECT_QWEN_XML,
                            &calls, &n_calls) != 0 || n_calls <= 0)
        return NULL;
    /* Rebuild the OpenAI tool_calls array exactly as the reply carries it and
     * The client echoes it, then render it the way build_messages() will. */
    json_writer jw;
    if (!jw_init(&jw)) { tc_tool_calls_free(calls, n_calls); return NULL; }
    jw_raw(&jw, "[");
    for (int i = 0; i < n_calls; i++) {
        if (i) jw_raw(&jw, ",");
        jw_raw(&jw, "{\"id\":");
        jw_string(&jw, calls[i].id);
        jw_raw(&jw, ",\"type\":\"function\",\"function\":{\"name\":");
        jw_string(&jw, calls[i].name);
        jw_raw(&jw, ",\"arguments\":");
        jw_string(&jw, calls[i].arguments);
        jw_raw(&jw, "}}");
    }
    jw_raw(&jw, "]");
    jr_value *dom = jr_parse(jw.data);
    char *joined = tc_join_assistant_history(rc->content, dom, TC_DIALECT_QWEN_XML);
    jr_free(dom);
    jw_free(&jw);
    tc_tool_calls_free(calls, n_calls);
    return joined;
}

/* Is a[0,a_n) a strict prefix of b[0,b_n)? */
static bool is_strict_prefix(const uint32_t *a, int a_n,
                             const uint32_t *b, int b_n)
{
    return a_n > 0 && a_n < b_n &&
           memcmp(a, b, (size_t)a_n * sizeof(uint32_t)) == 0;
}

/* Run one reply shape end to end: project it, rebuild the next turn's prompt
 * from what the client sends back, and require the projection to be a strict
 * prefix of it.  Then require the ids the model SAMPLED not to be. */
static void test_projection_case(pn_model *model, const jr_value *tools,
                     const reply_case *rc)
{
    printf("case: %s\n", rc->name);

    sg_message turn1[2] = {
        { "system", SYS_PROMPT, NULL },
        { "user",   USER_TURN_1, NULL },
    };
    int n1 = 0, sys_len = 0, think_prefill = 0;
    uint32_t *p1 = sg_build_prompt_tokens(model, turn1, 2, SG_THINKING_DEFAULT, &n1, &sys_len,
                                          &think_prefill);
    ASSERT_TRUE(p1 != NULL);
    ASSERT_TRUE(sys_len > 0 && sys_len < n1);
    ASSERT_INT_EQ(think_prefill, 1);      /* the forced "<think>" opener */
    int head_n = n1 - think_prefill;

    int rerender_n = 0;
    uint32_t *rerender = sg_project_client_rerender(model, rc->content,
                                                    rc->tool_raw, tools,
                                                    TC_DIALECT_QWEN_XML,
                                                    &rerender_n);
    ASSERT_TRUE(rerender != NULL);
    ASSERT_TRUE(rerender_n > 0);

    /* The projection the slot records: the client's head + the re-render. */
    int vis_n = head_n + rerender_n;
    uint32_t *vis = malloc((size_t)vis_n * sizeof(uint32_t));
    memcpy(vis, p1, (size_t)head_n * sizeof(uint32_t));
    memcpy(vis + head_n, rerender, (size_t)rerender_n * sizeof(uint32_t));

    /* Turn 2 — what a faithful client posts next. */
    char *assistant_content = client_assistant_content(rc, tools);
    ASSERT_TRUE(assistant_content != NULL);
    sg_message turn2[4] = {
        { "system",    SYS_PROMPT,        NULL },
        { "user",      USER_TURN_1,       NULL },
        { "assistant", assistant_content, NULL },
        { "user",      USER_TURN_2,       NULL },
    };
    int n2 = 0;
    uint32_t *p2 = sg_build_prompt_tokens(model, turn2, 4, SG_THINKING_DEFAULT, &n2, NULL, NULL);
    ASSERT_TRUE(p2 != NULL);

    ASSERT_TRUE(is_strict_prefix(vis, vis_n, p2, n2));

    /* NEGATIVE CONTROL — the projection the old code recorded: the ids the
     * model actually sampled for its reply.  This is the measured production
     * bug, and it must NOT match, or the assertion above proves nothing. */
    const char *sampled_text = rc->tool_raw ? rc->tool_raw : rc->content;
    int sampled_n = 0;
    uint32_t *sampled = sampled_ids(model->tokenizer, sampled_text, &sampled_n);
    int old_n = head_n + sampled_n;
    uint32_t *old_vis = malloc((size_t)old_n * sizeof(uint32_t));
    memcpy(old_vis, p1, (size_t)head_n * sizeof(uint32_t));
    memcpy(old_vis + head_n, sampled, (size_t)sampled_n * sizeof(uint32_t));
    if (rc->tool_raw)
        ASSERT_TRUE(!is_strict_prefix(old_vis, old_n, p2, n2));

    free(old_vis);
    free(sampled);
    free(p2);
    free(assistant_content);
    free(vis);
    free(rerender);
    free(p1);
}

/* ============================================================
 * Cases
 * ============================================================ */

/* The wrapped call the grammar compels — and the exact shape the K80 test machine saw
 * diverge, because the model spelled "<tool_call>" out in plain tokens. */
static const char TOOL_RAW_WRAPPED[] =
    "<tool_call>\n<function=attempt_completion>\n<parameter=result>\n"
    "Item 1 is a widget.\n</parameter>\n</function>\n</tool_call>";

/* The bare form the Q3 model emits much of the time: no wrapper at all, so the
 * client's re-render ADDS one the model never generated. */
static const char TOOL_RAW_BARE[] =
    "<function=attempt_completion>\n<parameter=result>\n"
    "Item 1 is a widget.\n</parameter>\n</function>";

static void test_tool_turn_wrapped(void)
{
    ft_tokenizer *tok = make_serve_tok();
    pn_model model = {0};
    model.tokenizer = tok;
    model.chat_style = PN_CHAT_CHATML_THINK;
    jr_value *tools = jr_parse(TOOLS_JSON);
    reply_case rc = { "wrapped tool call (the observed case)", "",
                      TOOL_RAW_WRAPPED };
    test_projection_case(&model, tools, &rc);
    jr_free(tools);
    ft_free(tok);
}

static void test_tool_turn_bare(void)
{
    ft_tokenizer *tok = make_serve_tok();
    pn_model model = {0};
    model.tokenizer = tok;
    model.chat_style = PN_CHAT_CHATML_THINK;
    jr_value *tools = jr_parse(TOOLS_JSON);
    reply_case rc = { "bare <function=> call the client re-wraps", "",
                      TOOL_RAW_BARE };
    test_projection_case(&model, tools, &rc);
    jr_free(tools);
    ft_free(tok);
}

static void test_tool_turn_with_prose(void)
{
    ft_tokenizer *tok = make_serve_tok();
    pn_model model = {0};
    model.tokenizer = tok;
    model.chat_style = PN_CHAT_CHATML_THINK;
    jr_value *tools = jr_parse(TOOLS_JSON);
    reply_case rc = { "prose + a tool call (the \\n\\n separator)",
                      "Let me look that up.", TOOL_RAW_WRAPPED };
    test_projection_case(&model, tools, &rc);
    jr_free(tools);
    ft_free(tok);
}

static void test_prose_turn(void)
{
    ft_tokenizer *tok = make_serve_tok();
    pn_model model = {0};
    model.tokenizer = tok;
    model.chat_style = PN_CHAT_CHATML_THINK;
    reply_case rc = { "plain prose answer", "Item 1 is a widget.", NULL };
    test_projection_case(&model, NULL, &rc);
    ft_free(tok);
}

/* ============================================================
 * The reasoning-bearing half of the same contract
 *
 * A client that resends `reasoning_content` is the one that keeps the model
 * reasoning at all (18/18 collapse without it, 0/18 with it — Fisher
 * p = 2.2e-10).  But its history no
 * longer matches the think-ELIDED projection, so before this rung every such
 * turn re-prefilled the whole conversation.
 *
 * The property under test:
 *
 *     head ++ sg_encode_history_think_block(reasoning) ++ rerender(reply)
 *       ==  the prefix of the NEXT turn's prompt when that turn's assistant
 *           message carries `reasoning_content`.
 *
 * The client never has to reproduce the ids the model SAMPLED inside its think
 * block — it echoes the string and the serve re-derives the ids through the
 * builder's own encoder.  Both negative controls below are what makes that a
 * claim and not a tautology: each projection must MISS the other client's
 * history, which is exactly the production defect in each direction.
 * ============================================================ */

/* The reasoning the model streamed on this turn, echoed back verbatim by the
 * client.  Contains a "<think>"-adjacent character run on purpose: the block
 * markers must come from the SPECIAL ids the builder pushes, not from bytes
 * that happen to live in the reasoning text. */
#define TURN_1_REASONING \
    "The user wants item 1.  I should call the tool rather than guess.\n" \
    "Note: do not write </thin k> here."

static void assert_reasoning_projection(pn_model *model, const jr_value *tools,
                                        const reply_case *rc)
{
    printf("case: %s\n", rc->name);

    sg_message turn1[2] = {
        { "system", SYS_PROMPT, NULL },
        { "user",   USER_TURN_1, NULL },
    };
    int n1 = 0, sys_len = 0, think_prefill = 0;
    uint32_t *p1 = sg_build_prompt_tokens(model, turn1, 2, SG_THINKING_DEFAULT,
                                          &n1, &sys_len, &think_prefill);
    ASSERT_NOT_NULL(p1);
    int head_n = n1 - think_prefill;

    int rerender_n = 0;
    uint32_t *rerender = sg_project_client_rerender(model, rc->content,
                                                    rc->tool_raw, tools,
                                                    TC_DIALECT_QWEN_XML,
                                                    &rerender_n);
    ASSERT_NOT_NULL(rerender);

    int think_n = 0;
    uint32_t *think = sg_encode_history_think_block(model, TURN_1_REASONING,
                                                    &think_n);
    ASSERT_NOT_NULL(think);
    ASSERT_TRUE(think_n > 0);

    /* The two projections the slot records for this one turn. */
    int elided_n = head_n + rerender_n;
    uint32_t *elided = malloc((size_t)elided_n * sizeof(uint32_t));
    memcpy(elided, p1, (size_t)head_n * sizeof(uint32_t));
    memcpy(elided + head_n, rerender, (size_t)rerender_n * sizeof(uint32_t));

    int bearing_n = head_n + think_n + rerender_n;
    uint32_t *bearing = malloc((size_t)bearing_n * sizeof(uint32_t));
    memcpy(bearing, p1, (size_t)head_n * sizeof(uint32_t));
    memcpy(bearing + head_n, think, (size_t)think_n * sizeof(uint32_t));
    memcpy(bearing + head_n + think_n, rerender,
           (size_t)rerender_n * sizeof(uint32_t));

    /* Turn 2, both ways: the SAME conversation, differing only in whether the
     * assistant history turn carries the reasoning back. */
    char *assistant_content = client_assistant_content(rc, tools);
    ASSERT_NOT_NULL(assistant_content);
    sg_message with_reasoning[4] = {
        { "system",    SYS_PROMPT,        NULL },
        { "user",      USER_TURN_1,       NULL },
        { "assistant", assistant_content, TURN_1_REASONING },
        { "user",      USER_TURN_2,       NULL },
    };
    sg_message without_reasoning[4] = {
        { "system",    SYS_PROMPT,        NULL },
        { "user",      USER_TURN_1,       NULL },
        { "assistant", assistant_content, NULL },
        { "user",      USER_TURN_2,       NULL },
    };
    int n_with = 0, n_without = 0;
    uint32_t *p_with = sg_build_prompt_tokens(model, with_reasoning, 4,
                                              SG_THINKING_DEFAULT, &n_with,
                                              NULL, NULL);
    uint32_t *p_without = sg_build_prompt_tokens(model, without_reasoning, 4,
                                                 SG_THINKING_DEFAULT,
                                                 &n_without, NULL, NULL);
    ASSERT_NOT_NULL(p_with);
    ASSERT_NOT_NULL(p_without);
    /* Sending the reasoning back really does change the prompt — otherwise the
     * two assertions below would be the same assertion twice. */
    ASSERT_TRUE(n_with > n_without);

    /* THE PROPERTY. */
    ASSERT_TRUE(is_strict_prefix(bearing, bearing_n, p_with, n_with));

    /* NEGATIVE CONTROL 1 — the production defect this rung fixes: the elided
     * projection does NOT match a client that sends reasoning, so before the
     * reasoning-bearing candidate existed every such turn re-prefilled. */
    ASSERT_TRUE(!is_strict_prefix(elided, elided_n, p_with, n_with));

    /* NEGATIVE CONTROL 2 — and the new candidate must not cannibalize the old
     * one: it must MISS a client that drops reasoning, so the two are decided
     * by the request and never by luck. */
    ASSERT_TRUE(!is_strict_prefix(bearing, bearing_n, p_without, n_without));
    ASSERT_TRUE(is_strict_prefix(elided, elided_n, p_without, n_without));

    free(p_without);
    free(p_with);
    free(assistant_content);
    free(bearing);
    free(elided);
    free(think);
    free(rerender);
    free(p1);
}

static void test_reasoning_projection_prose(void)
{
    ft_tokenizer *tok = make_serve_tok();
    pn_model model = {0};
    model.tokenizer = tok;
    model.chat_style = PN_CHAT_CHATML_THINK;
    reply_case rc = { "reasoning resent + prose answer",
                      "Item 1 is a widget.", NULL };
    assert_reasoning_projection(&model, NULL, &rc);
    ft_free(tok);
}

static void test_reasoning_projection_tool_call(void)
{
    ft_tokenizer *tok = make_serve_tok();
    pn_model model = {0};
    model.tokenizer = tok;
    model.chat_style = PN_CHAT_CHATML_THINK;
    jr_value *tools = jr_parse(TOOLS_JSON);
    reply_case rc = { "reasoning resent + a bare tool call the client re-wraps",
                      "", TOOL_RAW_BARE };
    assert_reasoning_projection(&model, tools, &rc);
    jr_free(tools);
    ft_free(tok);
}

/* The projection must be EMPTY exactly where the builder renders no block, or
 * the rung would predict a prompt that never arrives and the session would miss
 * every turn while looking like it had a candidate. */
static void test_think_block_projects_nothing_when_the_builder_renders_none(void)
{
    printf("case: no think block to project -> no candidate\n");
    ft_tokenizer *tok = make_serve_tok();
    pn_model model = {0};
    model.tokenizer = tok;
    model.chat_style = PN_CHAT_CHATML_THINK;

    int n = -1;
    ASSERT_TRUE(sg_encode_history_think_block(&model, NULL, &n) == NULL);
    ASSERT_INT_EQ(n, 0);
    ASSERT_TRUE(sg_encode_history_think_block(&model, "", &n) == NULL);
    ASSERT_INT_EQ(n, 0);

    /* A non-think chat style renders a history assistant turn WITHOUT a block
     * whatever `reasoning` says (sg_build_prompt_tokens gates on the style), so
     * projecting one here would be a guaranteed miss. */
    model.chat_style = PN_CHAT_CHATML;
    ASSERT_TRUE(sg_encode_history_think_block(&model, "some reasoning", &n)
                == NULL);
    ASSERT_INT_EQ(n, 0);

    /* And it really is the style, not the text: the same call on a think style
     * produces a block. */
    model.chat_style = PN_CHAT_CHATML_THINK;
    uint32_t *ids = sg_encode_history_think_block(&model, "some reasoning", &n);
    ASSERT_NOT_NULL(ids);
    ASSERT_TRUE(n > 0);
    /* The tags are the SPECIAL ids, not their bytes — a byte-encoded "</think>"
     * would never reach the trained id and the model would read the block as
     * prose. */
    int open_id = ft_token_to_id(tok, "<think>");
    int close_id = ft_token_to_id(tok, "</think>");
    ASSERT_INT_EQ((int)ids[0], open_id);
    bool closed = false;
    for (int i = 0; i < n; i++)
        if ((int)ids[i] == close_id) closed = true;
    ASSERT_TRUE(closed);
    free(ids);
    ft_free(tok);
}

/* ============================================================
 * Verify: history think block matches the jinja chat template exactly
 *
 * The template writes:  + two newlines.  We must reproduce that byte-for-byte:
 *  go in as SPECIAL ids, the middle as ONE text push.
 * ============================================================ */
static void test_history_think_block_matches_template(void)
{
    printf("case: history think block matches the jinja template\n");
    ft_tokenizer *tok = make_serve_tok();
    pn_model model = {0};
    model.tokenizer = tok;
    model.chat_style = PN_CHAT_CHATML_THINK;

    int open_id = ft_token_to_id(tok, "<think>");
    ASSERT_TRUE(open_id >= 0);
    int close_id = ft_token_to_id(tok, "</think>");
    ASSERT_TRUE(close_id >= 0);

    /* Three reasoning variants - trimmed by Jinja|trim, all produce the
     * same middle string: "\nI should call get_weather for Paris.\n" (38 bytes). */
    const char *reasonings[] = {
        "I should call get_weather for Paris.",                          /* R1 */
        "I should call get_weather for Paris.\n",                        /* R2 */
        "  I should call get_weather for Paris. \n\n",                   /* R3 */
    };
    const char *expected_middle = "\nI should call get_weather for Paris.\n";
    int mid_n = 0;
    uint32_t *mid_ids = sampled_ids(tok, expected_middle, &mid_n);

    /* Trailing \n\n after closing tag */
    int tail_n = 0;
    uint32_t *tail_ids = sampled_ids(tok, "\n\n", &tail_n);

    /* Expected full sequence: [open] ++ mid_ids ++ [close] ++ tail_ids */
    int expected_n = 1 + mid_n + 1 + tail_n;
    ASSERT_INT_EQ(expected_n, 42);

    for (int r = 0; r < 3; r++) {
        printf("  reasoning[%d] (R%d): %s\n", r, r + 1, reasonings[r]);
        int n = 0;
        uint32_t *ids = sg_encode_history_think_block(&model, reasonings[r], &n);
        ASSERT_NOT_NULL(ids);

        ASSERT_INT_EQ(n, expected_n);

        for (int i = 0; i < n; i++) {
            uint32_t want;
            if (i == 0)
                want = (uint32_t)open_id;
            else if (i == 1 + mid_n)
                want = (uint32_t)close_id;
            else if (i > 1 + mid_n)
                want = tail_ids[i - (1 + mid_n) - 1];
            else
                want = mid_ids[i - 1];
            ASSERT_INT_EQ(ids[i], want);
        }
        free(ids);
    }

    /* Prompt-level check: build a prompt with a reasoning assistant turn and
     * verify the expected ids occur as a contiguous run. */
    sg_message msgs[3] = {
        { "user",     "What's the weather in Paris?", NULL },
        { "assistant", "Done.",                        reasonings[0] },
        { "user",     "Thanks",                       NULL },
    };
    int prompt_n = 0;
    uint32_t *prompt = sg_build_prompt_tokens(&model, msgs, 3,
                                              SG_THINKING_DEFAULT, &prompt_n,
                                              NULL, NULL);
    ASSERT_NOT_NULL(prompt);

    /* The expected think-block ids: [open] ++ mid_ids ++ [close] ++ tail_ids */
    int block_n = 1 + mid_n + 1 + tail_n;
    uint32_t *block_expected = malloc((size_t)block_n * sizeof(uint32_t));
    block_expected[0] = (uint32_t)open_id;
    for (int i = 0; i < mid_n; i++)
        block_expected[1 + i] = mid_ids[i];
    block_expected[1 + mid_n] = (uint32_t)close_id;
    for (int i = 0; i < tail_n; i++)
        block_expected[1 + mid_n + 1 + i] = tail_ids[i];

    bool found = false;
    for (int i = 0; i + block_n <= prompt_n; i++) {
        bool match = true;
        for (int j = 0; j < block_n; j++)
            if (prompt[i + j] != block_expected[j]) { match = false; break; }
        if (match) { found = true; break; }
    }
    ASSERT_TRUE(found);

    free(block_expected);
    free(prompt);
    free(mid_ids);
    free(tail_ids);
    ft_free(tok);
}

/* ============================================================
 * A THIRD convention: a SUMMARY of the reasoning
 *
 * The harness wants OFF / SUMMARY / FULL per turn.  Two of those are recorded
 * candidates.  The third is a block that is PRESENT but carries different text,
 * so it is a different token sequence and matches neither — asserted here
 * through the production builder rather than by assumption, because this is
 * where the per-turn idea either survives or dies.
 *
 * The same case carries the fix: the block the summary client sends is exactly
 * what sg_encode_history_think_block produces from the summary STRING.  The
 * encoder is already general over any reasoning text; the only thing the serve
 * lacks at record time is which string the client will choose.  So a third
 * recorded projection works — provided the summary comes from a source both
 * sides share.
 * ============================================================ */


#define TURN_1_SUMMARY "Decided to call the tool instead of guessing."

static void test_summarised_reasoning_is_a_third_rendering(void)
{
    printf("case: a summarised block is neither recorded candidate\n");
    ft_tokenizer *tok = make_serve_tok();
    pn_model model = {0};
    model.tokenizer = tok;
    model.chat_style = PN_CHAT_CHATML_THINK;

    sg_message turn1[2] = {
        { "system", SYS_PROMPT, NULL },
        { "user",   USER_TURN_1, NULL },
    };
    int n1 = 0, sys_len = 0, think_prefill = 0;
    uint32_t *p1 = sg_build_prompt_tokens(&model, turn1, 2, SG_THINKING_DEFAULT,
                                          &n1, &sys_len, &think_prefill);
    ASSERT_NOT_NULL(p1);
    int head_n = n1 - think_prefill;

    const char *reply = "Item 1 is a widget.";
    int rerender_n = 0;
    uint32_t *rerender = sg_project_client_rerender(&model, reply, NULL, NULL,
                                                    TC_DIALECT_QWEN_XML,
                                                    &rerender_n);
    ASSERT_NOT_NULL(rerender);

    /* The two projections bs_slot_note_visible records for this turn. */
    int full_n = 0;
    uint32_t *full = sg_encode_history_think_block(&model, TURN_1_REASONING,
                                                   &full_n);
    ASSERT_NOT_NULL(full);
    int elided_n = head_n + rerender_n;
    uint32_t *elided = malloc((size_t)elided_n * sizeof(uint32_t));
    memcpy(elided, p1, (size_t)head_n * sizeof(uint32_t));
    memcpy(elided + head_n, rerender, (size_t)rerender_n * sizeof(uint32_t));
    int bearing_n = head_n + full_n + rerender_n;
    uint32_t *bearing = malloc((size_t)bearing_n * sizeof(uint32_t));
    memcpy(bearing, p1, (size_t)head_n * sizeof(uint32_t));
    memcpy(bearing + head_n, full, (size_t)full_n * sizeof(uint32_t));
    memcpy(bearing + head_n + full_n, rerender,
           (size_t)rerender_n * sizeof(uint32_t));

    /* What a SUMMARY client posts next turn: the same conversation, the same
     * reply, a shorter think block. */
    sg_message summarised[4] = {
        { "system",    SYS_PROMPT,       NULL },
        { "user",      USER_TURN_1,      NULL },
        { "assistant", reply,            TURN_1_SUMMARY },
        { "user",      USER_TURN_2,      NULL },
    };
    int n_summ = 0;
    uint32_t *p_summ = sg_build_prompt_tokens(&model, summarised, 4,
                                              SG_THINKING_DEFAULT, &n_summ,
                                              NULL, NULL);
    ASSERT_NOT_NULL(p_summ);

    /* THE FINDING.  Neither candidate matches, and the divergence is INSIDE the
     * block: everything the client sent for earlier turns still agrees, so this
     * is the newest turn's rendering and nothing else. */
    ASSERT_TRUE(!is_strict_prefix(bearing, bearing_n, p_summ, n_summ));
    ASSERT_TRUE(!is_strict_prefix(elided, elided_n, p_summ, n_summ));
    ASSERT_INT_EQ(memcmp(p_summ, p1, (size_t)head_n * sizeof(uint32_t)), 0);

    /* THE FIX, as far as the encoder is concerned: the same function that built
     * the reasoning-bearing candidate builds a matching one from the summary
     * string.  Recording a third projection is therefore a bookkeeping change,
     * not a new rendering path — the missing input is the STRING. */
    int summ_block_n = 0;
    uint32_t *summ_block = sg_encode_history_think_block(&model,
                                                         TURN_1_SUMMARY,
                                                         &summ_block_n);
    ASSERT_NOT_NULL(summ_block);
    ASSERT_TRUE(summ_block_n < full_n);   /* it really is shorter */
    int third_n = head_n + summ_block_n + rerender_n;
    uint32_t *third = malloc((size_t)third_n * sizeof(uint32_t));
    memcpy(third, p1, (size_t)head_n * sizeof(uint32_t));
    memcpy(third + head_n, summ_block, (size_t)summ_block_n * sizeof(uint32_t));
    memcpy(third + head_n + summ_block_n, rerender,
           (size_t)rerender_n * sizeof(uint32_t));
    ASSERT_TRUE(is_strict_prefix(third, third_n, p_summ, n_summ));

    /* The degenerate summary — the empty string — is not a third convention at
     * all: the builder renders no block, so it IS the elided candidate and
     * costs nothing.  "Summarise to nothing" and OFF are the same request. */
    sg_message empty_summary[4] = {
        { "system",    SYS_PROMPT,  NULL },
        { "user",      USER_TURN_1, NULL },
        { "assistant", reply,       "" },
        { "user",      USER_TURN_2, NULL },
    };
    int n_empty = 0;
    uint32_t *p_empty = sg_build_prompt_tokens(&model, empty_summary, 4,
                                               SG_THINKING_DEFAULT, &n_empty,
                                               NULL, NULL);
    ASSERT_NOT_NULL(p_empty);
    ASSERT_TRUE(is_strict_prefix(elided, elided_n, p_empty, n_empty));

    free(p_empty); free(third); free(summ_block); free(p_summ);
    free(bearing); free(elided); free(full); free(rerender); free(p1);
    ft_free(tok);
}

/* A turn that produced nothing reproducible (a pure-reasoning runaway) must
 * project NOTHING, so the caller leaves the rung disabled rather than record a
 * projection that would silently mismatch. */
static void test_unreproducible_turn_projects_nothing(void)
{
    printf("case: unreproducible turn disables the rung\n");
    ft_tokenizer *tok = make_serve_tok();
    pn_model model = {0};
    model.tokenizer = tok;
    model.chat_style = PN_CHAT_CHATML_THINK;

    int n = -1;
    uint32_t *ids = sg_project_client_rerender(&model, "", NULL, NULL,
                                               TC_DIALECT_QWEN_XML, &n);
    ASSERT_TRUE(ids == NULL);
    ASSERT_INT_EQ(n, 0);

    /* A diverted block that will not parse back into calls is the same case. */
    ids = sg_project_client_rerender(&model, "", "<function=", NULL,
                                     TC_DIALECT_QWEN_XML, &n);
    ASSERT_TRUE(ids == NULL);
    ASSERT_INT_EQ(n, 0);
    ft_free(tok);
}

/* The two sides of the contract are ONE encoder.  If the prompt builder and the
 * projection ever disagree about how assistant text becomes tokens, every tool
 * turn re-prefills — so assert they are literally the same function's output. */
static void test_one_encoder_for_both_sides(void)
{
    printf("case: prompt builder and projection share one encoder\n");
    ft_tokenizer *tok = make_serve_tok();
    pn_model model = {0};
    model.tokenizer = tok;
    model.chat_style = PN_CHAT_CHATML_THINK;

    const char *content = "before <tool_call>\nx\n</tool_call> after";
    int direct_n = 0;
    uint32_t *direct = sg_encode_assistant_content(&model, content, &direct_n);
    ASSERT_TRUE(direct != NULL);

    sg_message msgs[1] = { { "assistant", content, NULL } };
    int n = 0;
    uint32_t *prompt = sg_build_prompt_tokens(&model, msgs, 1, SG_THINKING_DEFAULT, &n, NULL, NULL);
    ASSERT_TRUE(prompt != NULL);

    /* The encoded content sits right after "<|im_start|>" + "assistant\n". */
    int header_n = 1;
    int hdr_ids_n = 0;
    int *hdr = pn_bpe_encode(tok, "assistant\n", &hdr_ids_n);
    free(hdr);
    header_n += hdr_ids_n;
    ASSERT_TRUE(n > header_n + direct_n);
    ASSERT_INT_EQ(memcmp(prompt + header_n, direct,
                         (size_t)direct_n * sizeof(uint32_t)), 0);
    /* And the marker really did become ONE id, not its bytes — otherwise the
     * comparison above would pass on two identically-wrong encodings. */
    int open_id = ft_token_to_id(tok, "<tool_call>");
    ASSERT_TRUE(open_id >= 0);
    bool found = false;
    for (int i = 0; i < direct_n; i++)
        if ((int)direct[i] == open_id) found = true;
    ASSERT_TRUE(found);

    free(prompt);
    free(direct);
    ft_free(tok);
}

/* The tools preamble (a system turn) spells "<tool_call>" out in its format
 * instructions; the chat template tokenizes that as the special id, so the
 * builder must too.  A user turn with the same literal text stays bytes. */
static void test_system_turn_tool_markers_are_special(void)
{
    printf("case: system-turn <tool_call> is the special id, user-turn is bytes\n");
    ft_tokenizer *tok = make_serve_tok();
    pn_model model = {0};
    model.tokenizer = tok;
    model.chat_style = PN_CHAT_CHATML_THINK;
    int open_id = ft_token_to_id(tok, "<tool_call>");
    int close_id = ft_token_to_id(tok, "</tool_call>");

    const char *text = "reply as <tool_call>\nx\n</tool_call>";
    const char *roles[2] = { "system", "user" };
    for (int r = 0; r < 2; r++) {
        sg_message msgs[1] = { { roles[r], text, NULL } };
        int n = 0;
        uint32_t *prompt = sg_build_prompt_tokens(&model, msgs, 1, SG_THINKING_DEFAULT, &n, NULL, NULL);
        ASSERT_TRUE(prompt != NULL);
        int opens = 0, closes = 0;
        for (int i = 0; i < n; i++) {
            if ((int)prompt[i] == open_id) opens++;
            if ((int)prompt[i] == close_id) closes++;
        }
        ASSERT_INT_EQ(opens, r == 0 ? 1 : 0);
        ASSERT_INT_EQ(closes, r == 0 ? 1 : 0);
        free(prompt);
    }
    ft_free(tok);
}

/* Does `enable_thinking` actually REACH THE MODEL?
 *
 * A test that only checks the field parses would pass while the feature does
 * nothing, so this asserts on the token stream the model is prefilled with —
 * the last thing before the forward pass.  SG_THINKING_OFF must land the
 * template's enable_thinking=false serialization, "<think>\n\n</think>\n\n",
 * as SPECIAL ids (byte-encoded "</think>" would never reach the trained id);
 * DEFAULT must not contain a "</think>" at all, because the block is left OPEN
 * for the model to close itself.
 *
 * Asserting BOTH directions is the point: "OFF closes the block" alone would
 * still pass if the builder closed it unconditionally. */
static void assert_prefill_for_style(int chat_style, const char *label)
{
    printf("  style: %s\n", label);
    ft_tokenizer *tok = make_serve_tok();
    pn_model model = {0};
    model.tokenizer = tok;
    model.chat_style = chat_style;

    int close_id = ft_token_to_id(tok, "</think>");
    int open_id  = ft_token_to_id(tok, "<think>");

    sg_message msgs[1] = { { "user", "hi", NULL } };

    int n_def = 0, prefill_def = 0;
    uint32_t *def = sg_build_prompt_tokens(&model, msgs, 1, SG_THINKING_DEFAULT,
                                           &n_def, NULL, &prefill_def);
    ASSERT_NOT_NULL(def);

    int n_off = 0, prefill_off = 0;
    uint32_t *off = sg_build_prompt_tokens(&model, msgs, 1, SG_THINKING_OFF,
                                           &n_off, NULL, &prefill_off);
    ASSERT_NOT_NULL(off);

    int n_on = 0, prefill_on = 0;
    uint32_t *on = sg_build_prompt_tokens(&model, msgs, 1, SG_THINKING_ON,
                                          &n_on, NULL, &prefill_on);
    ASSERT_NOT_NULL(on);

    bool def_closes = false, off_closes = false;
    for (int i = 0; i < n_def; i++) if ((int)def[i] == close_id) def_closes = true;
    for (int i = 0; i < n_off; i++) if ((int)off[i] == close_id) off_closes = true;

    /* The whole feature, in one pair of assertions. */
    ASSERT_TRUE(off_closes);
    ASSERT_FALSE(def_closes);

    /* The closed block is the LAST thing in the prompt: the opener, then the
     * closer, then nothing but newlines.  So the very next token the model emits
     * is already the answer — which is what makes this a dial rather than the
     * mid-sentence guillotine reasoning_budget is.
     * (In this synthetic vocab byte b has id b, so '\n' is id 10.) */
    int i_open = -1, i_close = -1;
    for (int i = 0; i < n_off; i++) {
        if ((int)off[i] == open_id)  i_open  = i;
        if ((int)off[i] == close_id) i_close = i;
    }
    ASSERT_TRUE(i_open >= 0);
    ASSERT_TRUE(i_close > i_open);
    for (int i = i_close + 1; i < n_off; i++)
        ASSERT_INT_EQ((int)off[i], 10);

    /* The forced prefill spans exactly the opener-through-end run, and closing
     * the block costs strictly more forced tokens than leaving it open. */
    ASSERT_INT_EQ(prefill_off, n_off - i_open);
    ASSERT_TRUE(prefill_off > prefill_def);

    /* ON must be byte-identical to DEFAULT here: PANTHEON_NO_THINK is unset in
     * the test env, so "explicitly on" and "unspecified" agree. */
    ASSERT_INT_EQ(n_on, n_def);
    ASSERT_INT_EQ(prefill_on, prefill_def);
    ASSERT_INT_EQ(memcmp(on, def, (size_t)n_def * sizeof(uint32_t)), 0);

    free(def); free(off); free(on);
    ft_free(tok);
}

static void test_enable_thinking_reaches_the_prompt(void)
{
    printf("case: enable_thinking changes the tokens the model is prefilled with\n");
    assert_prefill_for_style(PN_CHAT_CHATML_THINK, "ChatML+think (nex-n2)");
    assert_prefill_for_style(PN_CHAT_CHATML_THINK_NL, "ChatML+think\\n (saber)");
}

/* A non-reasoning model has no think block to close, so the control must be
 * INERT there rather than injecting tags the model was never trained on. */
static void test_enable_thinking_is_inert_on_a_plain_model(void)
{
    printf("case: enable_thinking is inert on a non-think chat style\n");
    ft_tokenizer *tok = make_serve_tok();
    pn_model model = {0};
    model.tokenizer = tok;
    model.chat_style = PN_CHAT_CHATML;

    sg_message msgs[1] = { { "user", "hi", NULL } };
    int n_def = 0, n_off = 0;
    uint32_t *def = sg_build_prompt_tokens(&model, msgs, 1, SG_THINKING_DEFAULT,
                                           &n_def, NULL, NULL);
    uint32_t *off = sg_build_prompt_tokens(&model, msgs, 1, SG_THINKING_OFF,
                                           &n_off, NULL, NULL);
    ASSERT_NOT_NULL(def);
    ASSERT_NOT_NULL(off);
    ASSERT_INT_EQ(n_off, n_def);
    ASSERT_INT_EQ(memcmp(off, def, (size_t)n_def * sizeof(uint32_t)), 0);

    free(def); free(off);
    ft_free(tok);
}

/* ============================================================
 * Turn start (rung 1c, the turn-start checkpoint)
 *
 * The serve saves the slot's state at sg_prompt_bounds.turn_start, and resumes
 * there when the next request starts with prompt[0, turn_start). That holds for
 * every way a client can treat the turn just generated — drop it (Argus with a
 * reasoning-only or interrupted turn), resend it empty (an OpenAI-style client
 * that keeps an empty assistant message), or resend a partial reply — and for
 * nothing that changes an earlier message.
 * ============================================================ */

static void test_turn_start_survives_a_dropped_reply(void)
{
    printf("case: turn_start is a prefix of every next request that keeps the history\n");
    ft_tokenizer *tok = make_serve_tok();
    pn_model model = {0};
    model.tokenizer = tok;
    model.chat_style = PN_CHAT_CHATML_THINK_NL;   /* Qwen3.6: "<think>\n" */

    sg_message turn1[2] = {
        { "system", SYS_PROMPT,  NULL },
        { "user",   USER_TURN_1, NULL },
    };
    int n1 = 0;
    sg_prompt_bounds b;
    uint32_t *p1 = sg_build_prompt_tokens_bounded(&model, turn1, 2,
                                                  SG_THINKING_DEFAULT, &n1, &b);
    ASSERT_TRUE(p1 != NULL);
    /* "<|im_start|>" + "assistant\n" (10 bytes) + "<think>" + "\n". */
    ASSERT_INT_EQ(b.think_prefill, 2);
    ASSERT_INT_EQ(n1 - b.turn_start, 1 + 10 + b.think_prefill);
    ASSERT_INT_EQ((int)p1[b.turn_start], ft_token_to_id(tok, "<|im_start|>"));
    /* The old entry point reports the same boundaries. */
    int sys_len = 0, think_prefill = 0, n1b = 0;
    uint32_t *p1b = sg_build_prompt_tokens(&model, turn1, 2, SG_THINKING_DEFAULT,
                                           &n1b, &sys_len, &think_prefill);
    ASSERT_INT_EQ(n1b, n1);
    ASSERT_INT_EQ(sys_len, b.sys_len);
    ASSERT_INT_EQ(think_prefill, b.think_prefill);
    ASSERT_INT_EQ(memcmp(p1, p1b, (size_t)n1 * sizeof(uint32_t)), 0);

    /* Dropped reply: the rescue note follows the user turn directly. */
    sg_message dropped[3] = {
        { "system", SYS_PROMPT,  NULL },
        { "user",   USER_TURN_1, NULL },
        { "user",   USER_TURN_2, NULL },
    };
    /* Empty reply kept, and a partial reply kept. */
    sg_message empty[4] = {
        { "system",    SYS_PROMPT,  NULL },
        { "user",      USER_TURN_1, NULL },
        { "assistant", "",          NULL },
        { "user",      USER_TURN_2, NULL },
    };
    sg_message partial[4] = {
        { "system",    SYS_PROMPT,  NULL },
        { "user",      USER_TURN_1, NULL },
        { "assistant", "Item 1 is",  NULL },
        { "user",      USER_TURN_2, NULL },
    };
    const sg_message *nexts[3] = { dropped, empty, partial };
    const int n_msgs[3] = { 3, 4, 4 };
    for (int i = 0; i < 3; i++) {
        int n2 = 0;
        uint32_t *p2 = sg_build_prompt_tokens(&model, nexts[i], n_msgs[i],
                                              SG_THINKING_DEFAULT, &n2, NULL, NULL);
        ASSERT_TRUE(p2 != NULL);
        ASSERT_TRUE(is_strict_prefix(p1, b.turn_start, p2, n2));
        /* NEGATIVE CONTROL: the generation prefix is not on the next request,
         * so the full prompt (what rung 1 needs) is not a prefix of it. */
        ASSERT_TRUE(!is_strict_prefix(p1, n1, p2, n2));
        free(p2);
    }

    /* A rewritten earlier message breaks it, as it must. */
    sg_message rewritten[3] = {
        { "system", SYS_PROMPT,         NULL },
        { "user",   "What is item 9?",  NULL },
        { "user",   USER_TURN_2,        NULL },
    };
    int n3 = 0;
    uint32_t *p3 = sg_build_prompt_tokens(&model, rewritten, 3,
                                          SG_THINKING_DEFAULT, &n3, NULL, NULL);
    ASSERT_TRUE(!is_strict_prefix(p1, b.turn_start, p3, n3));

    free(p3);
    free(p1b);
    free(p1);
    ft_free(tok);
}

/* ============================================================
 * Tool loop (card 1.8: consecutive tool results share ONE user turn, content
 * is |trim'd as the Qwen3.5+ templates do, <tool_response> goes in as its id)
 *
 * The think-elided projection, the prefix cache and the turn-start checkpoint
 * all compare prefixes of the token stream ONE builder produces, so a layout
 * change is safe exactly when every earlier message renders the same whatever
 * follows it.  The one message whose rendering depends on what follows is a
 * tool result (it closes the user turn only before a non-tool message); both a
 * generated assistant turn and the generation prompt close it, so every prefix
 * the serve records still holds.  These cases check that end to end.
 * ============================================================ */

/* A Qwen3.6-like model: "<think>\n" prefill, content |trim, and the
 * <tool_response> pair as added tokens. */
static void make_qwen36_like_model(pn_model *model, bool with_tool_response_ids)
{
    ft_tokenizer *tok = make_serve_tok();
    if (with_tool_response_ids) {
        ft_add_token(tok, "<tool_response>");
        ft_add_token(tok, "</tool_response>");
    }
    memset(model, 0, sizeof(*model));
    model->tokenizer = tok;
    model->chat_style = PN_CHAT_CHATML_THINK_NL;
    model->chat_trim_content = 1;
}

/* Text of ids[0,n): a byte id is its byte, any other id its token string. */
static char *ids_as_text(ft_tokenizer *tok, const uint32_t *ids, int n)
{
    size_t cap = (size_t)n * 24 + 1, at = 0;
    char *text = malloc(cap);
    for (int i = 0; i < n; i++) {
        if (ids[i] < 256) { text[at++] = (char)ids[i]; continue; }
        const char *spelling = ft_id_to_token(tok, (int)ids[i]);
        size_t len = strlen(spelling);
        memcpy(text + at, spelling, len);
        at += len;
    }
    text[at] = '\0';
    return text;
}

static int count_id(const uint32_t *ids, int n, int id)
{
    int count = 0;
    for (int i = 0; i < n; i++) count += (int)ids[i] == id;
    return count;
}

static const char TOOL_RAW_TWO_CALLS[] =
    "<tool_call>\n<function=attempt_completion>\n<parameter=result>\n"
    "first\n</parameter>\n</function>\n</tool_call>\n"
    "<tool_call>\n<function=attempt_completion>\n<parameter=result>\n"
    "second\n</parameter>\n</function>\n</tool_call>";

static void test_tool_loop_prefixes_hold_with_merged_results(void)
{
    printf("case: projection and turn_start stay prefixes across a merged tool turn\n");
    pn_model model;
    make_qwen36_like_model(&model, true);
    ft_tokenizer *tok = model.tokenizer;
    jr_value *tools = jr_parse(TOOLS_JSON);

    sg_message turn1[2] = { { "system", SYS_PROMPT, NULL }, { "user", USER_TURN_1, NULL } };
    int n1 = 0;
    sg_prompt_bounds b1;
    uint32_t *p1 = sg_build_prompt_tokens_bounded(&model, turn1, 2, SG_THINKING_DEFAULT, &n1, &b1);
    ASSERT_TRUE(p1 != NULL);

    /* The reply: untrimmed prose (the template trims it) + two calls. */
    reply_case rc = { "two calls", "  Let me check.\n", TOOL_RAW_TWO_CALLS };
    int rerender_n = 0;
    uint32_t *rerender = sg_project_client_rerender(&model, rc.content, rc.tool_raw, tools,
                                                    TC_DIALECT_QWEN_XML, &rerender_n);
    ASSERT_TRUE(rerender != NULL);
    int head_n = n1 - b1.think_prefill, vis_n = head_n + rerender_n;
    uint32_t *vis = malloc((size_t)vis_n * sizeof(uint32_t));
    memcpy(vis, p1, (size_t)head_n * sizeof(uint32_t));
    memcpy(vis + head_n, rerender, (size_t)rerender_n * sizeof(uint32_t));

    char *assistant_content = client_assistant_content(&rc, tools);
    ASSERT_TRUE(assistant_content != NULL);
    sg_message turn2[5] = {
        { "system", SYS_PROMPT, NULL }, { "user", USER_TURN_1, NULL },
        { "assistant", assistant_content, NULL },
        { "tool", "  first done.\n", NULL }, { "tool", "{\"second\": 2}", NULL },
    };
    int n2 = 0;
    sg_prompt_bounds b2;
    uint32_t *p2 = sg_build_prompt_tokens_bounded(&model, turn2, 5, SG_THINKING_DEFAULT, &n2, &b2);
    ASSERT_TRUE(p2 != NULL);
    ASSERT_TRUE(is_strict_prefix(vis, vis_n, p2, n2));      /* the projection rung */

    /* The tail after the projection: ONE user turn holding both results, as ids. */
    int im_start = ft_token_to_id(tok, "<|im_start|>");
    ASSERT_INT_EQ(count_id(p2 + vis_n, n2 - vis_n, im_start), 2);   /* user + generation */
    ASSERT_INT_EQ(count_id(p2, n2, ft_token_to_id(tok, "<tool_response>")), 2);
    ASSERT_INT_EQ(count_id(p2, n2, ft_token_to_id(tok, "</tool_response>")), 2);
    char *tail = ids_as_text(tok, p2 + vis_n, b2.turn_start - vis_n);
    ASSERT_STR_EQ(tail, "<|im_end|>\n<|im_start|>user\n<tool_response>\nfirst done.\n"
                        "</tool_response>\n<tool_response>\n{\"second\": 2}\n"
                        "</tool_response><|im_end|>\n");

    /* The turn-start checkpoint of turn 2 is a prefix of turn 3. */
    sg_message turn3[7] = {
        turn2[0], turn2[1], turn2[2], turn2[3], turn2[4],
        { "assistant", "Both done.", NULL }, { "user", USER_TURN_2, NULL },
    };
    int n3 = 0;
    uint32_t *p3 = sg_build_prompt_tokens(&model, turn3, 7, SG_THINKING_DEFAULT, &n3, NULL, NULL);
    ASSERT_TRUE(is_strict_prefix(p2, b2.turn_start, p3, n3));

    /* NEGATIVE CONTROL: a request that sent only the first result closes the
     * user turn after it, so its turn_start is NOT a prefix of the merged
     * layout — the checkpoint misses (safe) rather than resuming wrongly. */
    int n2a = 0;
    sg_prompt_bounds b2a;
    uint32_t *p2a = sg_build_prompt_tokens_bounded(&model, turn2, 4, SG_THINKING_DEFAULT, &n2a, &b2a);
    ASSERT_TRUE(!is_strict_prefix(p2a, b2a.turn_start, p2, n2));

    free(p2a); free(p3); free(tail); free(p2); free(assistant_content);
    free(vis); free(rerender); free(p1);
    jr_free(tools);
    ft_free(tok);
}

static void test_tool_response_ids_only_when_the_vocab_has_them(void)
{
    printf("case: <tool_response> is one id with the added tokens, the same text without\n");
    sg_message msgs[4] = {
        { "system", SYS_PROMPT, NULL }, { "user", USER_TURN_1, NULL },
        { "assistant", "", NULL }, { "tool", "ok", NULL },
    };
    char *texts[2];
    for (int with_ids = 0; with_ids < 2; with_ids++) {
        pn_model model;
        make_qwen36_like_model(&model, with_ids);
        int n = 0;
        uint32_t *p = sg_build_prompt_tokens(&model, msgs, 4, SG_THINKING_DEFAULT, &n, NULL, NULL);
        ASSERT_TRUE(p != NULL);
        int open_id = ft_token_to_id(model.tokenizer, "<tool_response>");
        ASSERT_INT_EQ(open_id >= 0, with_ids);
        ASSERT_INT_EQ(open_id >= 0 ? count_id(p, n, open_id) : 1, 1);
        texts[with_ids] = ids_as_text(model.tokenizer, p, n);
        free(p);
        ft_free(model.tokenizer);
    }
    ASSERT_STR_EQ(texts[0], texts[1]);
    ASSERT_TRUE(strstr(texts[0], "user\n<tool_response>\nok\n</tool_response><|im_end|>\n") != NULL);
    free(texts[0]);
    free(texts[1]);
}

/* A whitespace-only reply before a call: the template drops the "\n\n"
 * separator (trim(content) is empty); and padded prose with no call, which the
 * builder |trims.  The projection must agree with both. */
static void test_whitespace_only_content_projects_like_the_builder(void)
{
    printf("case: whitespace-only content + a call projects what the builder renders\n");
    pn_model model;
    make_qwen36_like_model(&model, true);
    model.chat_style = PN_CHAT_CHATML_THINK;   /* test_projection_case wants a 1-id prefill */
    jr_value *tools = jr_parse(TOOLS_JSON);
    reply_case rc = { "blank prose", " \n\n ", TOOL_RAW_WRAPPED };
    test_projection_case(&model, tools, &rc);
    char *content = client_assistant_content(&rc, tools);
    ASSERT_TRUE(content != NULL && strncmp(content, "<tool_call>", 11) == 0);
    free(content);
    reply_case padded = { "padded prose, no call", "\n Item 1 is a widget. \n", NULL };
    test_projection_case(&model, NULL, &padded);
    jr_free(tools);
    ft_free(model.tokenizer);
}

int main(void)
{
    printf("\n=== prefix-cache projection agreement ===\n\n");
    RUN_TEST(tool_turn_wrapped);
    RUN_TEST(tool_turn_bare);
    RUN_TEST(tool_turn_with_prose);
    RUN_TEST(prose_turn);
    RUN_TEST(unreproducible_turn_projects_nothing);
    RUN_TEST(one_encoder_for_both_sides);
    RUN_TEST(system_turn_tool_markers_are_special);
    RUN_TEST(history_think_block_matches_template);
    printf("\n=== reasoning-bearing projection (the client sends reasoning back) ===\n\n");
    RUN_TEST(reasoning_projection_prose);
    RUN_TEST(reasoning_projection_tool_call);
    RUN_TEST(think_block_projects_nothing_when_the_builder_renders_none);
    RUN_TEST(summarised_reasoning_is_a_third_rendering);
    printf("\n=== per-request thinking control ===\n\n");
    RUN_TEST(enable_thinking_reaches_the_prompt);
    RUN_TEST(enable_thinking_is_inert_on_a_plain_model);
    printf("\n=== turn start (rung 1c) ===\n\n");
    RUN_TEST(turn_start_survives_a_dropped_reply);
    printf("\n=== tool loop: merged results, content trim, <tool_response> ids ===\n\n");
    RUN_TEST(tool_loop_prefixes_hold_with_merged_results);
    RUN_TEST(tool_response_ids_only_when_the_vocab_has_them);
    RUN_TEST(whitespace_only_content_projects_like_the_builder);
    TEST_REPORT();
    return __ft_failed != 0;
}
