/* SPDX-License-Identifier: AGPL-3.0-or-later */
/* Copyright (C) 2026 David (Corey) Herrera */
/*
 * test_serve_nonstream_disconnect.c — a NON-streaming chat whose client
 * disconnects stops generating and frees the model, on the real chat handler
 * (oai_handle_chat over a socket pair) with the real qwen3-0.6b on the CPU path.
 *
 * Before the fix only streaming turns watched the client: a non-streaming turn
 * ran to max_tokens for an answer nobody would read, holding the slot (17 s on
 * a CPU build, ~47 s per turn on the K80 batched path).
 *
 *   A. Reference: the same request with the client kept open runs long (it
 *      counts to 500, max_tokens 400).
 *   B. The client closes its end 300 ms in: the handler returns in well under
 *      half the reference time.
 *
 * Builds on main (oai_handle_chat + mr_load only) and fails there.
 * Model: $PANTHEON_TEST_MODEL, else models/qwen3-0.6b-Q8_0.gguf. On macOS run
 * model loads through foundry's metal/run_guarded.sh.
 */

#include <pthread.h>
#include <signal.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/socket.h>
#include <time.h>
#include <unistd.h>

#include "../src/serve/model_registry.h"
#include "../src/serve/openai_chat.h"

static int g_failed = 0;

#define CHECK(cond, msg) do { \
    if (!(cond)) { \
        fprintf(stderr, "  FAIL: %s (%s:%d)\n", (msg), __FILE__, __LINE__); \
        g_failed++; \
    } else { \
        printf("  ok: %s\n", (msg)); \
    } \
} while (0)

static const char *BODY =
    "{\"model\":\"small\",\"messages\":[{\"role\":\"user\",\"content\":"
    "\"Count from 1 to 500, one number per line, nothing else.\"}],"
    "\"max_tokens\":400,\"temperature\":0,\"enable_thinking\":false}";

static const char *model_path(void)
{
    const char *env = getenv("PANTHEON_TEST_MODEL");
    return (env && *env) ? env : "models/qwen3-0.6b-Q8_0.gguf";
}

static double now_s(void)
{
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (double)ts.tv_sec + (double)ts.tv_nsec / 1e9;
}

typedef struct {
    mr_registry *reg;
    int          fd;
    double       done_s;
} chat_job;

static void *chat_thread(void *arg)
{
    chat_job *job = arg;
    oai_handle_chat(job->fd, job->reg, BODY, NULL, NULL);
    job->done_s = now_s();
    close(job->fd);   /* as the serve's connection thread does: the reader sees EOF */
    return NULL;
}

/* Run one chat; close the client end after `close_after_s` (< 0 = read the
 * whole answer). Returns the handler's run time; the reply goes to `reply`. */
static double run_chat(mr_registry *reg, double close_after_s, char *reply, size_t cap)
{
    int sv[2];
    if (socketpair(AF_UNIX, SOCK_STREAM, 0, sv) != 0)
        return -1.0;
    chat_job job = { .reg = reg, .fd = sv[1] };
    pthread_t th;
    double t0 = now_s();
    pthread_create(&th, NULL, chat_thread, &job);
    size_t len = 0;
    if (close_after_s >= 0) {
        usleep((useconds_t)(close_after_s * 1e6));
        close(sv[0]);
    } else {
        for (ssize_t r; len + 1 < cap && (r = read(sv[0], reply + len, cap - len - 1)) > 0;)
            len += (size_t)r;
        close(sv[0]);
    }
    if (reply && cap)
        reply[len] = '\0';
    pthread_join(th, NULL);
    return job.done_s - t0;
}

int main(void)
{
    setvbuf(stdout, NULL, _IOLBF, 0);
    signal(SIGPIPE, SIG_IGN);   /* as the serve does (hs_listen): a gone client is EPIPE */
    printf("test_serve_nonstream_disconnect\n");
    FILE *f = fopen(model_path(), "rb");
    if (!f) {
        printf("SKIP: model not found: %s\n", model_path());
        return 0;
    }
    fclose(f);
    mr_registry reg;
    char err[512] = {0};
    if (mr_init(&reg) != 0)
        return 1;
    pn_load_opts opts = { .ctx_cap = 2048 };
    if (mr_load(&reg, model_path(), "small", &opts, NULL, 0, 0, 1, 1, err, sizeof(err)) != 0) {
        fprintf(stderr, "  load error: %s\n", err);
        return 1;
    }

    static char reply[1 << 16];
    double full = run_chat(&reg, -1.0, reply, sizeof(reply));
    const char *ct = strstr(reply, "\"completion_tokens\":");
    int gen = ct ? atoi(ct + 20) : 0;
    printf("  reference: %.2f s, %d tokens\n", full, gen);
    CHECK(gen >= 200, "A. the reference answer runs long (>= 200 tokens)");

    double cut = run_chat(&reg, 0.3, NULL, 0);
    printf("  disconnected at 0.3 s: handler returned after %.2f s\n", cut);
    CHECK(cut < full * 0.5, "B. a disconnected non-streaming turn stops early");

    mr_free(&reg);
    if (g_failed) {
        printf("test_serve_nonstream_disconnect: %d FAILED\n", g_failed);
        return 1;
    }
    printf("test_serve_nonstream_disconnect: all passed\n");
    return 0;
}
