/* SPDX-License-Identifier: AGPL-3.0-or-later */
/* Copyright (C) 2026 David (Corey) Herrera */
/*
 * test_local_empty_start.c — Pantheon Local A1/A5: `pantheon local` starts with
 * no model, /v1/models is an empty list, a chat is a clean 503, and one model
 * loads and unloads through /admin (with a sampling default that shows on the
 * row). Plain `serve` with no --model still refuses to start.
 *
 * The model step uses models/qwen3-0.6b-Q8_0.gguf (override PN_TEST_MODEL) and
 * is skipped when the file is absent. The server is a forked child of this
 * binary running pn_local_main on 127.0.0.1.
 */

#include "local_test_support.h"

#include "../src/local/local_command.h"
#include "../src/serve/serve_command.h"

#define STATE "build/test_local/empty_start"

static const char *test_model_path(void)
{
    const char *env = getenv("PN_TEST_MODEL");
    return (env && *env) ? env : "models/qwen3-0.6b-Q8_0.gguf";
}

static void test_plain_serve_still_needs_model(void)
{
    printf("plain_serve_still_needs_model:\n");
    pid_t pid = fork();
    if (pid == 0) {
        FILE *f = freopen("/dev/null", "w", stderr);
        (void)f;
        char *argv[] = { "serve", "--port", "1", NULL };
        _exit(pn_serve_main(3, argv));
    }
    int st = 0;
    waitpid(pid, &st, 0);
    CHECK(WIFEXITED(st) && WEXITSTATUS(st) == 1, "serve without --model exits 1");
}

/* H2o: a plain `pantheon local` uses the built-in catalog URL; --catalog-url
 * overrides it, and '' turns the fetch off. */
static void test_default_catalog_url(void)
{
    static pn_local_ctx c;
    char *plain[] = { "local", NULL };
    char *mirror[] = { "local", "--catalog-url", "https://mirror.example/c.json", NULL };
    char *none[] = { "local", "--catalog-url", "", NULL };

    printf("default_catalog_url:\n");
    CHECK(pn_local_parse_args(&c, 1, plain, 1) == 0 &&
          strcmp(c.catalog_url, "https://huggingface.co/TeamIDE/pantheon-catalog/resolve/main/"
                                "catalog.json") == 0, "no flag -> the TeamIDE HF catalog");
    CHECK(c.port == 8400 && strcmp(c.host, "0.0.0.0") == 0, "other defaults kept");
    CHECK(pn_local_parse_args(&c, 3, mirror, 1) == 0 &&
          strcmp(c.catalog_url, "https://mirror.example/c.json") == 0, "--catalog-url overrides");
    CHECK(pn_local_parse_args(&c, 3, none, 1) == 0 && c.catalog_url[0] == '\0',
          "--catalog-url '' = no catalog");
}

static void test_model_round_trip(int port, const char *key)
{
    const char *path = test_model_path();
    struct stat sb;
    char *body = NULL;
    char req[1024];

    printf("model_round_trip:\n");
    if (stat(path, &sb) != 0) {
        printf("  [SKIP] %s not found (set PN_TEST_MODEL)\n", path);
        return;
    }
    snprintf(req, sizeof(req), "{\"path\":\"%s\",\"name\":\"m\",\"top_k\":20}", path);
    int st = lt_http(port, "POST", "/admin/models/load", key, req, &body);
    CHECK(st == 200, "admin load 200");
    CHECK(body && strstr(body, "\"sampling_defaults\":{\"top_k\":20}"), "row carries top_k 20");
    free(body);
    st = lt_http(port, "GET", "/v1/models", key, NULL, &body);
    CHECK(st == 200 && body && strstr(body, "\"id\":\"m\""), "model listed");
    free(body);
    st = lt_http(port, "POST", "/admin/models/unload", key, "{\"name\":\"m\"}", &body);
    CHECK(st == 200, "admin unload 200");
    free(body);
    st = lt_http(port, "GET", "/v1/models", key, NULL, &body);
    CHECK(st == 200 && body && strstr(body, "\"data\":[]"), "list empty again");
    free(body);
}

int main(void)
{
    int port = lt_pick_port(1);
    char port_s[16];
    char *body = NULL;

    test_plain_serve_still_needs_model();
    test_default_catalog_url();

    lt_clean_dir(STATE);
    snprintf(port_s, sizeof(port_s), "%d", port);
    char *argv[] = { "local", "--host", "127.0.0.1", "--port", port_s, "--state-dir",
                     STATE "/state", "--store", STATE "/store", "--no-mdns", "--catalog-url", "", NULL };
    lt_make_dir(STATE);
    pid_t pid = lt_spawn_local(12, argv, port, STATE "/serve.log");
    printf("empty_start:\n");
    CHECK(pid > 0, "pantheon local started");
    if (pid > 0) {
        const char *key = lt_admin_key(port, STATE "/state");
        int st = lt_http(port, "GET", "/v1/models", key, NULL, &body);
        CHECK(st == 200, "GET /v1/models 200");
        CHECK(body && strstr(body, "\"data\":[]"), "no models at start");
        free(body);
        st = lt_http(port, "POST", "/v1/chat/completions", key,
                     "{\"messages\":[{\"role\":\"user\",\"content\":\"hi\"}]}", &body);
        CHECK(st == 503 && body && strstr(body, "no_models_loaded"), "chat with no model is 503");
        free(body);
        test_model_round_trip(port, key);
        lt_stop(pid);
    }
    return lt_report("test_local_empty_start");
}
