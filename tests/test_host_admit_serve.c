/* SPDX-License-Identifier: AGPL-3.0-or-later */
/* Copyright (C) 2026 David (Corey) Herrera */
/*
 * test_host_admit_serve.c — the serve under host-RAM admission, on the Mac,
 * with a fake MemAvailable (PN_HOST_ADMIT_TEST_AVAIL_MB) and the real
 * qwen3-0.6b on the CPU path.
 *
 *   A. A model load whose explicit window does not fit is refused at load
 *      (mr_load -1 with the reason; POST /admin/models/load answers 507).
 *   B. At runtime a request whose NEW session cannot be admitted gets a clean
 *      503 server_memory_constrained carrying the reason, and the server stays
 *      up: an anonymous request on the same model still answers 200, and the
 *      same session is served once memory is back.
 *
 * Uses only the env hook and serve APIs that exist on main, so it builds on
 * main and fails there (RED): main allocates instead of refusing.
 *
 * Model: $PANTHEON_TEST_MODEL, else models/qwen3-0.6b-Q8_0.gguf.
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include <sys/socket.h>
#include <unistd.h>

#include "../src/core/prefix_cache.h"
#include "../src/serve/model_admin.h"
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

static const char *model_path(void)
{
    const char *env = getenv("PANTHEON_TEST_MODEL");
    return (env && *env) ? env : "models/qwen3-0.6b-Q8_0.gguf";
}

static void set_avail_mb(long mb)
{
    char v[32];
    snprintf(v, sizeof(v), "%ld", mb);
    setenv("PN_HOST_ADMIT_TEST_AVAIL_MB", v, 1);
}

/* Read everything the handler wrote to the other end of a socketpair. */
static char *drain(int fd)
{
    size_t cap = 1 << 16, len = 0;
    char *buf = malloc(cap);
    if (!buf)
        return NULL;
    for (;;) {
        ssize_t r = read(fd, buf + len, cap - len - 1);
        if (r <= 0)
            break;
        len += (size_t)r;
        if (len + 1 >= cap)
            break;
    }
    buf[len] = '\0';
    return buf;
}

static char *chat(mr_registry *reg, const char *sid)
{
    const char *body = "{\"model\":\"small\",\"messages\":[{\"role\":\"user\","
                       "\"content\":\"Hi\"}],\"max_tokens\":2,\"temperature\":0}";
    int sv[2];
    if (socketpair(AF_UNIX, SOCK_STREAM, 0, sv) != 0)
        return NULL;
    oai_handle_chat(sv[1], reg, body, sid, NULL);
    close(sv[1]);
    char *out = drain(sv[0]);
    close(sv[0]);
    return out;
}

static char *admin_load(mr_registry *reg, const char *body)
{
    int sv[2];
    if (socketpair(AF_UNIX, SOCK_STREAM, 0, sv) != 0)
        return NULL;
    ma_handle_load(sv[1], reg, body);
    close(sv[1]);
    char *out = drain(sv[0]);
    close(sv[0]);
    return out;
}

static void test_load_refused(mr_registry *reg)
{
    printf("A. load refusal:\n");
    char err[512] = {0};
    pn_load_opts opts = { .ctx_cap = 20000 };   /* ~4.3 GiB of fp32 KV */
    set_avail_mb(3072);                         /* 1.5 GiB above the reserve */
    int rc = mr_load(reg, model_path(), "big", &opts, NULL, 0, 0, 1, 1, err,
                     sizeof(err));
    CHECK(rc == -1, "mr_load refuses a 20000-position window on a 3 GiB host");
    CHECK(strstr(err, "host RAM admission refused") != NULL,
          "load error says host RAM admission refused");
    CHECK(strstr(err, "needs 4.3 GiB") != NULL, "load error carries the size");

    char body[1024];
    snprintf(body, sizeof(body), "{\"path\":\"%s\",\"name\":\"big\",\"ctx_cap\":20000}",
             model_path());
    char *resp = admin_load(reg, body);
    CHECK(resp && strstr(resp, " 507 ") != NULL, "/admin/models/load answers 507");
    CHECK(resp && strstr(resp, "host KV ring") != NULL,
          "/admin/models/load body carries the reason");
    free(resp);
}

static void test_runtime_session_refused(mr_registry *reg)
{
    printf("B. runtime session refusal:\n");
    char err[512] = {0};
    pn_load_opts opts = { .ctx_cap = 2048 };    /* ~450 MiB per session */
    set_avail_mb(8192);
    int rc = mr_load(reg, model_path(), "small", &opts, NULL, 0, 0, 4, 1, err,
                     sizeof(err));
    CHECK(rc == 0, "small model loads");
    if (rc != 0) {
        fprintf(stderr, "  load error: %s\n", err);
        return;
    }
    /* The registry only builds a prefix cache for die-placed models; attach
     * one so a session id allocates a new state per conversation, which is
     * the runtime allocation under test. */
    mr_entry *e = &reg->entries[0];
    for (int i = 0; i < MR_MAX_MODELS; i++)
        if (strcmp(reg->entries[i].name, "small") == 0)
            e = &reg->entries[i];
    e->prefix_cache = pn_prefix_cache_new(e->model, 4);
    CHECK(e->prefix_cache != NULL, "prefix cache attached");

    /* Admitted so far ~1.05 GiB (593 MiB embedding + the registry state). Leave
     * ~50 MiB above reserve + admitted: no room for a ~450 MiB session. */
    set_avail_mb(1536 + 1100);
    char *resp = chat(reg, "sid-A");
    CHECK(resp && strstr(resp, " 503 ") != NULL, "new session refused with 503");
    CHECK(resp && strstr(resp, "server_memory_constrained") != NULL,
          "error type is server_memory_constrained");
    CHECK(resp && strstr(resp, "host KV ring") != NULL, "error carries the reason");
    free(resp);

    resp = chat(reg, NULL);
    CHECK(resp && strstr(resp, " 200 ") != NULL,
          "server stays up: an anonymous request still answers 200");
    free(resp);

    set_avail_mb(8192);
    resp = chat(reg, "sid-A");
    CHECK(resp && strstr(resp, " 200 ") != NULL,
          "the same session is served once memory is back");
    free(resp);
}

int main(void)
{
    printf("test_host_admit_serve\n");
    FILE *f = fopen(model_path(), "rb");
    if (!f) {
        printf("SKIP: model not found: %s\n", model_path());
        return 0;
    }
    fclose(f);
    mr_registry reg;
    if (mr_init(&reg) != 0)
        return 1;
    test_load_refused(&reg);
    test_runtime_session_refused(&reg);
    mr_free(&reg);
    unsetenv("PN_HOST_ADMIT_TEST_AVAIL_MB");
    if (g_failed) {
        printf("test_host_admit_serve: %d FAILED\n", g_failed);
        return 1;
    }
    printf("test_host_admit_serve: all passed\n");
    return 0;
}
