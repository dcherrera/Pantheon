/* SPDX-License-Identifier: AGPL-3.0-or-later */
/* Copyright (C) 2026 David (Corey) Herrera */
/*
 * test_local_flow.c — Pantheon Local end to end on the Mac (CPU), the same flow
 * the bank scripts run: start empty → pair → signed catalog (TEST ONLY key) →
 * catalog lists the cpu profile as fitting and the k80/p150 ones greyed out →
 * install (download through curl from a local HF-layout server + sha256) →
 * load with the profile (sampling defaults applied; loaded_by = the paired
 * person) → chat with an inference key → restart: the model comes back by
 * itself, still loaded_by that person → unload through
 * /admin/models/unload → restart: nothing comes back → uninstall (H2m: refused
 * while loaded; removes the files; a model the catalog no longer lists shows as
 * store_only and can be removed) → install again.
 *
 * Uses models/qwen3-0.6b-Q8_0.gguf (PN_TEST_MODEL), pinned by sha256 in
 * tests/local/fixtures/catalog_e2e.json; skipped when absent. Built with
 * PN_LOCAL_TEST_CATALOG_KEY (this test binary only).
 */

#include "local_test_support.h"
#include "local_fixture_httpd.h"

#define DIR "build/test_local/flow"
#define REV "0123456789abcdef0123456789abcdef01234567"
#define ID  "qwen3-0.6b-q8"

static int g_port, g_httpd_port;
static char g_port_s[16], g_cat_url[1200], g_hf[64];

static pid_t start_box(const char *log)
{
    char *argv[] = { "local", "--host", "127.0.0.1", "--port", g_port_s, "--state-dir",
                     DIR "/state", "--store", DIR "/store", "--no-mdns", "--allow-http",
                     "--catalog-url", g_cat_url, "--hf-base-url", g_hf, NULL };
    return lt_spawn_local(15, argv, g_port, log);
}

/* Poll GET path until its body contains needle (or seconds run out). */
static int wait_for(const char *key, const char *path, const char *needle, int seconds)
{
    for (int i = 0; i < seconds * 4; i++) {
        char *body = NULL;
        lt_http(g_port, "GET", path, key, NULL, &body);
        int hit = body && strstr(body, needle);
        free(body);
        if (hit) return 1;
        usleep(250000);
    }
    return 0;
}

/* GET /local/models/loaded names the pairing person as loaded_by (spec §4.7). */
static int loaded_by_me(const char *admin)
{
    char *body = NULL, needle[96], person[32] = "";
    int ok = 0;

    lt_http(g_port, "GET", "/local/me", admin, NULL, &body);
    if (body && strstr(body, "\"person\":{\"id\":\""))
        sscanf(strstr(body, "\"person\":{\"id\":\"") + 16, "%31[^\"]", person);
    free(body);
    body = NULL;
    snprintf(needle, sizeof(needle), "\"loaded_by\":{\"id\":\"%s\"", person);
    lt_http(g_port, "GET", "/local/models/loaded", admin, NULL, &body);
    ok = person[0] == 'p' && body && strstr(body, needle);
    free(body);
    return ok;
}

static void flow_first_boot(char *admin, size_t cap, char *inf, size_t inf_cap)
{
    char *body = NULL;

    printf("first_boot:\n");
    pid_t pid = start_box(DIR "/serve1.log");
    CHECK(pid > 0, "box up");
    const char *k = lt_admin_key(g_port, DIR "/state");
    snprintf(admin, cap, "%s", k ? k : "");
    CHECK(k != NULL, "paired");
    CHECK(wait_for(admin, "/local/catalog", "\"version\":1", 20), "signed catalog accepted");
    lt_http(g_port, "GET", "/local/catalog", admin, NULL, &body);
    CHECK(body && strstr(body, "\"id\":\"cpu\"") && strstr(body, "needs 1 k80-gpu, this box has 0"),
          "cpu fits, k80 greyed with its reason");
    free(body);
    int st = lt_http(g_port, "POST", "/local/models/install", admin, "{\"catalog_id\":\"" ID "\"}", NULL);
    CHECK(st == 202, "install accepted");
    CHECK(wait_for(admin, "/local/models/install/" ID, "\"state\":\"done\"", 120), "installed");
    st = lt_http(g_port, "POST", "/local/models/load", admin,
                 "{\"catalog_id\":\"" ID "\",\"profile_id\":\"cpu\"}", &body);
    CHECK(st == 200, body ? body : "load");
    free(body);
    CHECK(loaded_by_me(admin), "the load records who loaded it (loaded_by)");
    lt_http(g_port, "GET", "/admin/models", admin, NULL, &body);
    CHECK(body && strstr(body, "\"top_k\":20"), "profile sampling defaults on the model");
    free(body);
    st = lt_http(g_port, "POST", "/local/models/uninstall", admin, "{\"catalog_id\":\"" ID "\"}",
                 &body);
    CHECK(st == 409 && body && strstr(body, "is loaded on the CPU. Unload it"),
          "uninstall refused while loaded, with a plain reason");
    free(body);
    st = lt_http(g_port, "POST", "/local/keys", admin, "{\"label\":\"editor\"}", &body);
    if (body && strstr(body, "\"key\":\""))
        sscanf(strstr(body, "\"key\":\"") + 7, "%127[^\"]", inf);
    free(body);
    st = lt_http(g_port, "POST", "/v1/chat/completions", inf,
                 "{\"model\":\"" ID "\",\"max_tokens\":4,\"messages\":[{\"role\":\"user\","
                 "\"content\":\"Say hi.\"}]}", &body);
    CHECK(st == 200 && body && strstr(body, "\"choices\""), "chat through an inference key");
    free(body);
    (void)inf_cap;
    lt_stop(pid);
}

static void flow_reboot(const char *admin)
{
    char *body = NULL;

    printf("reboot_reload_then_unload:\n");
    pid_t pid = start_box(DIR "/serve2.log");
    CHECK(pid > 0, "box up again");
    CHECK(wait_for(admin, "/v1/models", "\"id\":\"" ID "\"", 90), "model reloaded by itself");
    lt_http(g_port, "GET", "/local/models/loaded", admin, NULL, &body);
    CHECK(body && strstr(body, "\"profile_id\":\"cpu\""), "loaded set lists it");
    free(body);
    CHECK(loaded_by_me(admin), "the reload keeps loaded_by");
    int st = lt_http(g_port, "POST", "/admin/models/unload", admin, "{\"name\":\"" ID "\"}", NULL);
    CHECK(st == 200, "unload through /admin/models/unload");
    lt_http(g_port, "GET", "/local/models/loaded", admin, NULL, &body);
    CHECK(body && strstr(body, "\"models\":[]"), "loaded set emptied");
    free(body);
    lt_stop(pid);

    pid = start_box(DIR "/serve3.log");
    sleep(3);
    lt_http(g_port, "GET", "/v1/models", admin, NULL, &body);
    CHECK(body && strstr(body, "\"data\":[]"), "after an unload nothing comes back");
    free(body);
    lt_stop(pid);
}

/* Write a stand-in for a model a newer catalog dropped. */
static void put_dropped_model(void)
{
    FILE *f = fopen(DIR "/store/old-model-0123456789ab.gguf", "w");
    if (f) {
        fputs("weights", f);
        fclose(f);
    }
}

static void flow_uninstall(const char *admin)
{
    char *body = NULL;
    struct stat sb;

    printf("uninstall_then_install_again:\n");
    put_dropped_model();
    pid_t pid = start_box(DIR "/serve4.log");
    CHECK(pid > 0, "box up");
    CHECK(wait_for(admin, "/local/catalog", "\"installed\":true", 20), "installed before");
    lt_http(g_port, "GET", "/local/catalog", admin, NULL, &body);
    CHECK(body && strstr(body, "\"store_only\":[{\"catalog_id\":\"old-model\",\"bytes\":7,"),
          "a model the catalog dropped is listed as store_only");
    CHECK(body && !strstr(body, "\"catalog_id\":\"" ID "\",\"bytes\""),
          "the installed catalog model is not store_only");
    free(body);
    int st = lt_http(g_port, "POST", "/local/models/uninstall", admin,
                     "{\"catalog_id\":\"" ID "\"}", &body);
    CHECK(st == 200 && body && strstr(body, "\"removed_files\":1"), "uninstall 200");
    free(body);
    lt_http(g_port, "GET", "/local/catalog", admin, NULL, &body);
    CHECK(body && strstr(body, "\"installed\":false") && !strstr(body, "\"installed\":true"),
          "catalog row reads not installed");
    free(body);
    st = lt_http(g_port, "POST", "/local/models/uninstall", admin, "{\"catalog_id\":\"" ID "\"}",
                 NULL);
    CHECK(st == 404, "nothing left: 404");
    st = lt_http(g_port, "POST", "/local/models/uninstall", admin, "{\"catalog_id\":\"../x\"}",
                 NULL);
    CHECK(st == 400, "a path is not an id");
    st = lt_http(g_port, "POST", "/local/models/uninstall", admin,
                 "{\"catalog_id\":\"old-model\"}", NULL);
    CHECK(st == 200 && stat(DIR "/store/old-model-0123456789ab.gguf", &sb) != 0,
          "the dropped model is removed");
    lt_http(g_port, "GET", "/local/catalog", admin, NULL, &body);
    CHECK(body && strstr(body, "\"store_only\":[]"), "store_only empty");
    free(body);
    st = lt_http(g_port, "POST", "/local/models/install", admin, "{\"catalog_id\":\"" ID "\"}", NULL);
    CHECK(st == 202, "install again accepted");
    CHECK(wait_for(admin, "/local/models/install/" ID, "\"state\":\"done\"", 120),
          "installed again");
    CHECK(wait_for(admin, "/local/catalog", "\"installed\":true", 5), "catalog reads installed");
    lt_stop(pid);
}

int main(void)
{
    const char *model = getenv("PN_TEST_MODEL");
    char abs_model[1024], cwd[1024], www[1300], link[1400];
    char admin[128] = "", inf[128] = "";
    struct stat sb;

    if (!model || !*model) model = "models/qwen3-0.6b-Q8_0.gguf";
    if (stat(model, &sb) != 0 || !realpath(model, abs_model) || !getcwd(cwd, sizeof(cwd))) {
        printf("test_local_flow: [SKIP] %s not found (set PN_TEST_MODEL)\n", model);
        return 0;
    }
    g_port = lt_pick_port(4);
    g_httpd_port = lt_pick_port(5);
    snprintf(g_port_s, sizeof(g_port_s), "%d", g_port);
    snprintf(g_hf, sizeof(g_hf), "http://127.0.0.1:%d", g_httpd_port);
    snprintf(g_cat_url, sizeof(g_cat_url), "file://%s/tests/local/fixtures/catalog_e2e.json", cwd);
    lt_clean_dir(DIR);
    snprintf(www, sizeof(www), DIR "/www/teamide/test-fixtures/resolve/" REV);
    lt_make_dir(www);
    lt_make_dir(DIR "/store");
    snprintf(link, sizeof(link), "%s/qwen3-0.6b-Q8_0.gguf", www);
    if (symlink(abs_model, link) != 0) perror("symlink");

    fflush(stdout);
    pid_t httpd = fork();
    if (httpd == 0)
        _exit(fx_httpd_run(g_httpd_port, DIR "/www", 0, 0));
    lt_wait_port(g_httpd_port, 5000);
    flow_first_boot(admin, sizeof(admin), inf, sizeof(inf));
    flow_reboot(admin);
    flow_uninstall(admin);
    kill(httpd, SIGKILL);
    waitpid(httpd, NULL, 0);
    return lt_report("test_local_flow");
}
