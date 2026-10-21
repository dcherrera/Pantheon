/* SPDX-License-Identifier: AGPL-3.0-or-later */
/* Copyright (C) 2026 David (Corey) Herrera */
/*
 * test_backend_guard.c — the silent-deploy class, in four parts.
 *
 * The incident: a CPU-only `pantheon` binary sat at the path the K80 fleet
 * manifest names, served a 35B-A3B MoE on a 2-core no-AVX Pentium with all 8
 * K80 dies idle at 0 MiB for an unknown period, advertised "max_concurrent":2
 * while actually serving 1, and left nothing in any log that said so.
 * Diagnosing it took `ldd` and `strings` on a remote box.
 *
 * Part A — BACKEND IDENTITY. The process can state its own compiled backend.
 *          Compile-time constants, so this arm is proved by the preprocessor:
 *          on a CUDA build it asserts CUDA, on this Mac build it asserts CPU.
 *
 * Part B — THE REFUSAL. pn_backend_guard_check's full truth table, including
 *          the two rows that matter most: `--dies` on a binary with no
 *          accelerator backend must REFUSE, and no `--dies` at all must stay OK
 *          on every backend (that is how a legitimate CPU serve is asked for).
 *          Also pins the "no accelerator" substring model_admin.c's
 *          status_for_reason matches on, so the 400 mapping cannot rot.
 *
 * Part C — mr_load REFUSES a die list it cannot honour, which is the runtime
 *          POST /admin/models/load path (it reaches mr_load directly and never
 *          passes through pn_serve_main's pre-flight).
 *
 * Part D — THE ADVERTISED NUMBER. mr_snapshot must report SERVED concurrency,
 *          and the two JSON writers must emit it. Driven through the real
 *          oai_handle_models / ma_handle_models over a socketpair, so what is
 *          asserted is the exact bytes a client and an operator receive — not a
 *          restatement of the struct.
 *
 * No GPU, no model file, no network: builds and runs anywhere.
 */

#include "../src/accel/backend_identity.h"
#include "../src/serve/serve_backend_guard.h"
#include "../src/serve/model_admin.h"
#include "../src/serve/model_registry.h"
#include "../src/serve/openai_chat.h"
#include "../src/serve/serve_generation.h"
#include "../src/core/engine.h"

#include <ctype.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/socket.h>
#include <unistd.h>

static int g_failed = 0;

#define CHECK(cond, msg) do { \
    if (!(cond)) { \
        fprintf(stderr, "  FAIL: %s (%s:%d)\n", (msg), __FILE__, __LINE__); \
        g_failed++; \
    } \
} while (0)

/* --- Part A: backend identity -------------------------------------------- */

static void test_backend_identity(void)
{
    printf("backend_identity:\n");
    const char *name  = pn_backend_name();
    const char *token = pn_backend_token();
    const char *build = pn_backend_build_commit();

    CHECK(name && *name, "name is non-empty");
    CHECK(token && *token, "token is non-empty");
    CHECK(build && *build, "build commit is non-empty");

    /* The two spellings are one enumeration. A drift here is a binary that
     * logs one backend and asserts another at deploy time. */
    CHECK(strlen(name) == strlen(token), "name and token are the same value");
    for (size_t i = 0; name[i] && token[i]; i++)
        CHECK(tolower((unsigned char)name[i]) == (unsigned char)token[i],
              "token is the lowercase name");

#if defined(USE_CUDA)
    CHECK(strncmp(token, "cuda", 4) == 0, "CUDA build reports cuda");
    CHECK(pn_backend_has_accelerator(), "CUDA build has an accelerator");
    CHECK(pn_backend_supports_die_placement(), "CUDA build places on dies");
#elif defined(USE_TT)
    CHECK(strncmp(token, "tt", 2) == 0, "TT build reports tt");
    CHECK(pn_backend_has_accelerator(), "TT build has an accelerator");
    CHECK(pn_backend_supports_die_placement(), "TT build places on dies");
#elif defined(USE_METAL)
    CHECK(strcmp(token, "metal") == 0, "Metal build reports metal");
    CHECK(pn_backend_has_accelerator(), "Metal build has an accelerator");
    /* Metal has no pn_dm_init_* path, so --dies can never resolve on it. */
    CHECK(!pn_backend_supports_die_placement(), "Metal build places on no dies");
#else
    CHECK(strcmp(token, "cpu") == 0, "plain build reports cpu");
    CHECK(!pn_backend_has_accelerator(), "plain build has no accelerator");
    CHECK(!pn_backend_supports_die_placement(), "plain build places on no dies");
#endif
    printf("  backend=%s build=%s\n", name, build);
}

/* --- Part B: the refusal -------------------------------------------------- */

static void test_guard_truth_table(void)
{
    printf("guard_truth_table:\n");

    /* THE LEGITIMATE CPU SERVE: no --dies. Must stay OK on every backend, or
     * the guard becomes something operators route around. */
    CHECK(pn_backend_guard_check(0, false, 0) == PN_BACKEND_GUARD_OK,
          "no dies on a CPU build is OK (the deliberate CPU serve)");
    CHECK(pn_backend_guard_check(0, true, 8) == PN_BACKEND_GUARD_OK,
          "no dies on a CUDA build is OK");
    CHECK(pn_backend_guard_check(0, true, 0) == PN_BACKEND_GUARD_OK,
          "no dies with no dies visible is OK");

    /* THE INCIDENT: `--dies 0,1` accepted by a binary with no CUDA at all. */
    CHECK(pn_backend_guard_check(2, false, 0) == PN_BACKEND_GUARD_NO_DIE_BACKEND,
          "dies on a binary with no die backend REFUSED");
    /* Logical dies (pn_dm_init_logical) are bookkeeping, not silicon: a
     * non-zero die count must not launder a CPU binary into an accelerator. */
    CHECK(pn_backend_guard_check(2, false, 4) == PN_BACKEND_GUARD_NO_DIE_BACKEND,
          "dies on a CPU build REFUSED even with logical dies registered");

    /* Right binary, no card / no driver / empty CUDA_VISIBLE_DEVICES. */
    CHECK(pn_backend_guard_check(2, true, 0) == PN_BACKEND_GUARD_NO_DIES_VISIBLE,
          "dies with zero dies visible REFUSED");

    /* The working configuration. */
    CHECK(pn_backend_guard_check(2, true, 8) == PN_BACKEND_GUARD_OK,
          "dies on a die-capable build with dies visible is OK");
    CHECK(pn_backend_guard_check(1, true, 1) == PN_BACKEND_GUARD_OK,
          "one die, one visible, is OK");
}

static void test_guard_message(void)
{
    printf("guard_message:\n");
    const int dies[2] = { 0, 1 };
    char msg[640];

    /* OK writes nothing — a caller that prints unconditionally must not emit a
     * scary empty refusal on the happy path. */
    memset(msg, 'x', sizeof(msg));
    CHECK(pn_backend_guard_message(PN_BACKEND_GUARD_OK, "saber-orch", dies, 2,
                                   msg, sizeof(msg)) == 0,
          "OK verdict writes nothing");
    CHECK(msg[0] == '\0', "OK verdict leaves an empty string");

    int n = pn_backend_guard_message(PN_BACKEND_GUARD_NO_DIE_BACKEND,
                                     "saber-orch", dies, 2, msg, sizeof(msg));
    CHECK(n > 0 && (size_t)n < sizeof(msg), "message fits the buffer");
    CHECK(strstr(msg, "REFUSED") != NULL, "message says REFUSED");
    CHECK(strstr(msg, "saber-orch") != NULL, "message names the model");
    CHECK(strstr(msg, "0,1") != NULL, "message names the dies asked for");
    /* Cause AND fix, both required: an operator reading a fleet log has neither
     * the source nor the build command in front of them. */
    CHECK(strstr(msg, "no accelerator") != NULL ||
          strstr(msg, "NO accelerator") != NULL, "message names the cause");
    CHECK(strstr(msg, "make USE_CUDA=1") != NULL, "message names the fix");
    CHECK(strstr(msg, "--dies") != NULL, "message names the CPU-serve escape");
    CHECK(strchr(msg, '\n') == NULL, "message is ONE greppable line");

    n = pn_backend_guard_message(PN_BACKEND_GUARD_NO_DIES_VISIBLE, "saber-orch",
                                 dies, 2, msg, sizeof(msg));
    CHECK(n > 0, "no-dies-visible message is written");
    CHECK(strstr(msg, "0 dies") != NULL, "no-dies-visible message says 0 dies");

    /* Truncation must not overrun: a 16-byte buffer still NUL-terminates. */
    char tiny[16];
    pn_backend_guard_message(PN_BACKEND_GUARD_NO_DIE_BACKEND, "x", dies, 2,
                             tiny, sizeof(tiny));
    CHECK(tiny[sizeof(tiny) - 1] == '\0', "truncated message stays terminated");

    /* status_for_reason() in model_admin.c matches this substring to answer 400
     * instead of the catch-all 500. Pin it here so the two cannot drift. */
    const char *r1 = pn_backend_guard_short_reason(PN_BACKEND_GUARD_NO_DIE_BACKEND);
    const char *r2 = pn_backend_guard_short_reason(PN_BACKEND_GUARD_NO_DIES_VISIBLE);
    CHECK(r1 && strstr(r1, "no accelerator") != NULL,
          "short reason carries the 'no accelerator' marker");
    CHECK(r2 && strstr(r2, "no accelerator") != NULL,
          "short reason (no dies visible) carries the marker");
    CHECK(pn_backend_guard_short_reason(PN_BACKEND_GUARD_OK) == NULL,
          "OK has no short reason");
}

/* --- Part C: mr_load refuses a die list it cannot honour ------------------- */

static void test_mr_load_refuses_dies(void)
{
    printf("mr_load_refuses_dies:\n");
    mr_registry reg;
    int dies[2] = { 0, 1 };
    char err[192] = "";

    CHECK(mr_init(&reg) == 0, "init");

    /* The guard runs BEFORE the stat(), so a bogus path still exercises it —
     * and proves the refusal is about the dies, not about the file. */
    CHECK(mr_load(&reg, "/no/such/file.gguf", "saber-orch", NULL, dies, 2,
                  0, 0, 0, err, sizeof(err)) == -1,
          "die list refused on this build");
    /* On a die-capable build with no dies probed the refusal is the OTHER arm;
     * either way it must be the backend guard talking, not the missing file. */
    CHECK(strstr(err, "no accelerator") != NULL,
          "refusal is the backend guard, not the file check");
    CHECK(mr_count(&reg) == 0, "refused load leaves the registry empty");

    /* And the CPU path is untouched: with no dies the same call gets as far as
     * the file check, which is the ONLY thing that should stop it. */
    err[0] = '\0';
    CHECK(mr_load(&reg, "/no/such/file.gguf", "saber-orch", NULL, NULL, 0,
                  0, 0, 0, err, sizeof(err)) == -1, "missing file still refused");
    CHECK(strstr(err, "not found") != NULL,
          "no-dies load is refused by the FILE check, not the backend guard");

    mr_free(&reg);
}

/* --- Part D: the advertised number ---------------------------------------- */

/* A zeroed pn_model: mr_snapshot reads only config fields off it
 * (pn_effective_ctx_window / pn_session_ctx_cap) and the residency gauges,
 * all of which handle an all-zero config. No load, no weights, no GPU. */
static pn_model g_fake_model;

/* Hand-build one resident entry: max_concurrent CONFIGURED at `configured`,
 * and no scheduler — precisely the incident's state (bs_create is a
 * `#ifndef USE_CUDA` stub returning NULL, so entry->sched was NULL while the
 * entry kept the cap the operator asked for). */
static void seed_entry(mr_registry *reg, const char *name, int configured)
{
    mr_entry *e = &reg->entries[0];
    memset(e, 0, sizeof(*e));
    e->used = true;
    snprintf(e->name, sizeof(e->name), "%s", name);
    e->model = &g_fake_model;
    e->state = NULL;
    e->sched = NULL;
    e->max_sessions = configured;
    e->max_concurrent = configured;
    reg->n = 1;
}

/* Run one handler against a socketpair and return its whole response body.
 * Caller frees. This is the real writer over a real fd — what is asserted is
 * the bytes a client receives, not a restatement of the struct. */
static char *capture_response(void (*handler)(int, mr_registry *),
                              mr_registry *reg)
{
    int sv[2];
    if (socketpair(AF_UNIX, SOCK_STREAM, 0, sv) != 0)
        return NULL;
    handler(sv[1], reg);
    close(sv[1]);

    size_t cap = 65536, len = 0;
    char *buf = malloc(cap);
    if (!buf) { close(sv[0]); return NULL; }
    for (;;) {
        ssize_t r = read(sv[0], buf + len, cap - len - 1);
        if (r <= 0) break;
        len += (size_t)r;
        if (len + 1 >= cap) break;
    }
    buf[len] = '\0';
    close(sv[0]);
    return buf;
}

static void test_served_concurrency(void)
{
    printf("served_concurrency:\n");
    mr_registry reg;
    mr_info info[MR_MAX_MODELS];

    CHECK(mr_init(&reg) == 0, "init");
    seed_entry(&reg, "saber-orch", 4);

    CHECK(mr_snapshot(&reg, info, MR_MAX_MODELS) == 1, "one model in snapshot");
    CHECK(info[0].max_concurrent == 4, "configured cap is preserved");
    /* THE BUG: the entry says 4, the serve delivers 1. bs_get_load already
     * knew this (capacity 1 for a NULL scheduler) — only the two JSON writers
     * were not asking it. */
    CHECK(info[0].served_concurrency == 1,
          "served concurrency is 1 with no scheduler");

    /* /v1/models — what a CLIENT plans against. */
    char *body = capture_response(oai_handle_models, &reg);
    CHECK(body != NULL, "/v1/models responded");
    if (body) {
        CHECK(strstr(body, "\"max_concurrent\":1") != NULL,
              "/v1/models advertises the SERVED concurrency (1)");
        CHECK(strstr(body, "\"max_concurrent\":4") == NULL,
              "/v1/models does NOT advertise the unserviceable cap (4)");
        CHECK(strstr(body, "\"backend\":") != NULL,
              "/v1/models states the compiled backend");
        CHECK(strstr(body, pn_backend_token()) != NULL,
              "/v1/models states THIS backend");
        CHECK(strstr(body, "\"build\":") != NULL,
              "/v1/models states the build commit");
        free(body);
    }

    /* /admin/models — what an OPERATOR debugs with: both numbers. */
    body = capture_response(ma_handle_models, &reg);
    CHECK(body != NULL, "/admin/models responded");
    if (body) {
        CHECK(strstr(body, "\"max_concurrent\":1") != NULL,
              "/admin/models reports the served concurrency (1)");
        CHECK(strstr(body, "\"max_concurrent_configured\":4") != NULL,
              "/admin/models also reports the configured cap (4)");
        free(body);
    }

    /* Clear the hand-built entry before mr_free: its model pointer is not a
     * loaded model and must never reach pn_model_free. */
    memset(&reg.entries[0], 0, sizeof(reg.entries[0]));
    reg.n = 0;
    mr_free(&reg);
}

/* tool_choice "required" with thinking: the grammar arming policy, and the
 * /v1/models field a client reads before it sends "required" with thinking on.
 * The field NAME is a client contract (clients match it exactly). */
static void test_required_with_thinking_policy(void)
{
    printf("required_with_thinking_policy:\n");
    int dummy_tools = 1;
    sg_gen_params p;
    memset(&p, 0, sizeof(p));
    p.tools = &dummy_tools;

    unsetenv("PANTHEON_TOOL_GRAMMAR");
    p.tool_choice = SG_TOOL_CHOICE_REQUIRED;
    CHECK(sg_tool_grammar_arms(&p), "unset: \"required\" arms the grammar");
    p.tool_choice = SG_TOOL_CHOICE_AUTO;
    CHECK(!sg_tool_grammar_arms(&p), "unset: \"auto\" stays unconstrained");
    p.tool_choice = SG_TOOL_CHOICE_FUNCTION;
    CHECK(!sg_tool_grammar_arms(&p), "unset: a named function stays unconstrained");
    setenv("PANTHEON_TOOL_GRAMMAR", "1", 1);
    CHECK(sg_tool_grammar_arms(&p), "=1: a named function arms");
    p.tool_choice = SG_TOOL_CHOICE_AUTO;
    CHECK(sg_tool_grammar_arms(&p), "=1: \"auto\" arms");
    setenv("PANTHEON_TOOL_GRAMMAR", "0", 1);
    p.tool_choice = SG_TOOL_CHOICE_REQUIRED;
    CHECK(!sg_tool_grammar_arms(&p), "=0: even \"required\" stays unconstrained");
    unsetenv("PANTHEON_TOOL_GRAMMAR");
    p.tools = NULL;
    CHECK(!sg_tool_grammar_arms(&p), "no tools: never arms");

    mr_registry reg;
    CHECK(mr_init(&reg) == 0, "init");
    seed_entry(&reg, "ornith", 1);
    char *body = capture_response(oai_handle_models, &reg);
    CHECK(body && strstr(body, "\"tool_choice_required_with_thinking\":true"),
          "/v1/models: ornith advertises required-with-thinking");
    free(body);
    setenv("PANTHEON_TOOL_GRAMMAR", "0", 1);
    body = capture_response(oai_handle_models, &reg);
    CHECK(body && strstr(body, "\"tool_choice_required_with_thinking\":false"),
          "/v1/models: =0 withdraws it");
    free(body);
    unsetenv("PANTHEON_TOOL_GRAMMAR");
    seed_entry(&reg, "saber-orch", 1);
    body = capture_response(oai_handle_models, &reg);
    CHECK(body && strstr(body, "\"tool_choice_required_with_thinking\":true") &&
          !strstr(body, "\"tool_choice_required_with_thinking\":false"),
          "/v1/models: an XML-dialect model advertises it too");
    free(body);
    /* MiniCPM5's attribute dialect is not modelled by any grammar: a "required"
     * turn there has no guarantee, so the field must stay false. */
    pn_model minicpm_model;
    memset(&minicpm_model, 0, sizeof(minicpm_model));
    minicpm_model.config.arch = PN_ARCH_LLAMA;
    minicpm_model.chat_add_bos = 1;
    seed_entry(&reg, "minicpm5-orch", 1);
    reg.entries[0].model = &minicpm_model;
    body = capture_response(oai_handle_models, &reg);
    CHECK(body && strstr(body, "\"tool_choice_required_with_thinking\":false") &&
          !strstr(body, "\"tool_choice_required_with_thinking\":true"),
          "/v1/models: a MiniCPM-dialect model does not advertise it");
    free(body);
    CHECK(!sg_tool_choice_required_with_thinking(TC_DIALECT_MINICPM_XML),
          "MiniCPM dialect: no required-with-thinking guarantee");
    CHECK(sg_tool_choice_required_with_thinking(TC_DIALECT_QWEN_XML) &&
          sg_tool_choice_required_with_thinking(TC_DIALECT_ORNITH_JSON),
          "modelled dialects keep it");
    memset(&reg.entries[0], 0, sizeof(reg.entries[0]));
    reg.n = 0;
    mr_free(&reg);
}

int main(void)
{
    test_backend_identity();
    test_guard_truth_table();
    test_guard_message();
    test_mr_load_refuses_dies();
    test_served_concurrency();
    test_required_with_thinking_policy();

    if (g_failed) {
        printf("\nFAILED: %d check(s)\n", g_failed);
        return 1;
    }
    printf("\nAll checks passed.\n");
    return 0;
}
