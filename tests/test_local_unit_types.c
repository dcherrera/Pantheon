/* SPDX-License-Identifier: AGPL-3.0-or-later */
/* Copyright (C) 2026 David (Corey) Herrera */
/*
 * test_local_unit_types.c — Pantheon Local: files[].unit_types (spec §16), end to
 * end through a real `pantheon local` child on the Mac (CPU) with a signed TEST
 * ONLY catalog (tests/local/fixtures/catalog_unit_types.json): one model, a GGUF
 * every profile needs plus two `.fttw` extras tagged ["p150"].
 *   - install for the k80-one profile downloads only the GGUF; the k80-one and
 *     cpu rows read installed, the p150-one row does not; the model is not
 *     listed as store_only;
 *   - an install without profile_id on this CPU-only box needs only the GGUF:
 *     already installed (200);
 *   - install for p150-one downloads both .fttw; its row reads installed;
 *   - an unknown profile is a 404; uninstall removes every file.
 * And a second model with one weights file per unit type (Qwen3.6's shape: a K80
 * GGUF, and a p150 metadata GGUF + trunk + MTP .fttw):
 *   - install for p150-one downloads the three TT files and NOT the K80 GGUF; its
 *     row reads installed with size = those three; the k80-one row does not;
 *   - install for k80-one then downloads the K80 GGUF only; uninstall removes 4.
 */

#include "local_test_support.h"
#include "local_fixture_httpd.h"

#include "../src/serve/json_reader.h"

#define DIR  "build/test_local/unit_types"
#define REV  "0123456789abcdef0123456789abcdef01234567"
#define ID   "orn-mini-q8"
#define GGUF DIR "/store/" ID "-d41d438c3791.gguf"
#define TRUNK DIR "/store/" ID "-e5b95fc91790.fttw"
#define HEAD DIR "/store/" ID "-2b9280e5076b.fttw"
#define QID  "q36-mini-q8"
#define QGGUF DIR "/store/" QID "-0873bfbcb03f.gguf"
#define QMETA DIR "/store/" QID "-f281617c99c5.gguf"
#define QTRUNK DIR "/store/" QID "-8a5fc9b31669.fttw"
#define QHEAD DIR "/store/" QID "-20d0a2125f1c.fttw"

static int g_port, g_httpd_port;
static char g_port_s[16], g_cat_url[1200], g_hf[64];

/* The fixture's files: byte i = i * 31 + salt * 7 (sha256 pinned in the catalog). */
static void make_blob(const char *name, int salt, int size)
{
    char path[512];
    snprintf(path, sizeof(path), DIR "/www/teamide/test-fixtures/resolve/" REV "/%s", name);
    FILE *f = fopen(path, "wb");
    for (int i = 0; f && i < size; i++)
        fputc((i * 31 + salt * 7) & 0xff, f);
    if (f)
        fclose(f);
}

static int exists(const char *path)
{
    struct stat sb;
    return stat(path, &sb) == 0;
}

static pid_t start_box(void)
{
    char *argv[] = { "local", "--host", "127.0.0.1", "--port", g_port_s, "--state-dir",
                     DIR "/state", "--store", DIR "/store", "--no-mdns", "--allow-http",
                     "--catalog-url", g_cat_url, "--hf-base-url", g_hf, NULL };
    return lt_spawn_local(15, argv, g_port, DIR "/serve.log");
}

/* Install model `id` (profile_id may be NULL), wait for done; the HTTP status of the POST. */
static int install_model(const char *key, const char *id, const char *profile_id)
{
    char body[256], path[128], *resp = NULL;

    if (profile_id)
        snprintf(body, sizeof(body), "{\"catalog_id\":\"%s\",\"profile_id\":\"%s\"}", id,
                 profile_id);
    else
        snprintf(body, sizeof(body), "{\"catalog_id\":\"%s\"}", id);
    snprintf(path, sizeof(path), "/local/models/install/%s", id);
    int st = lt_http(g_port, "POST", "/local/models/install", key, body, NULL);
    for (int i = 0; st == 202 && i < 400; i++) {
        lt_http(g_port, "GET", path, key, NULL, &resp);
        int done = resp && (strstr(resp, "\"state\":\"done\"") || strstr(resp, "\"failed\""));
        free(resp);
        if (done)
            break;
        usleep(50000);
    }
    return st;
}

static int install(const char *key, const char *profile_id)
{
    return install_model(key, ID, profile_id);
}

/* The catalog row of model `id`, profile `profile_id`: 1 installed, 0 not, -1 no
 * row; *size = the row's model.size. */
static int row_of(const char *key, const char *id, const char *profile_id, double *size)
{
    char *body = NULL;
    int found = -1;

    lt_http(g_port, "GET", "/local/catalog", key, NULL, &body);
    jr_value *root = body ? jr_parse(body) : NULL;
    const jr_value *entries = jr_object_get(root, "entries");
    for (int i = 0; i < jr_array_len(entries); i++) {
        const jr_value *e = jr_array_get(entries, i);
        const jr_value *model = jr_object_get(e, "model");
        const char *pid = jr_as_string(jr_object_get(jr_object_get(e, "profile"), "id"));
        const char *mid = jr_as_string(jr_object_get(model, "id"));
        if (pid && mid && strcmp(pid, profile_id) == 0 && strcmp(mid, id) == 0) {
            found = jr_as_bool(jr_object_get(model, "installed"), false);
            if (size)
                *size = jr_as_number(jr_object_get(model, "size"), -1);
        }
    }
    jr_free(root);
    free(body);
    return found;
}

static int row_installed(const char *key, const char *profile_id)
{
    return row_of(key, ID, profile_id, NULL);
}

static int body_has(const char *key, const char *path, const char *needle)
{
    char *body = NULL;
    lt_http(g_port, "GET", path, key, NULL, &body);
    int hit = body && strstr(body, needle);
    free(body);
    return hit;
}

static void run_flow(void)
{
    pid_t pid = start_box();
    CHECK(pid > 0, "box up");
    const char *k = lt_admin_key(g_port, DIR "/state");
    char key[128];
    snprintf(key, sizeof(key), "%s", k ? k : "");
    CHECK(k != NULL, "paired");
    for (int i = 0; i < 80 && !body_has(key, "/local/catalog", "\"version\":1"); i++)
        usleep(250000);

    printf("k80_profile_installs_gguf_only:\n");
    CHECK(install(key, "k80-one") == 202, "install k80-one accepted");
    CHECK(exists(GGUF), "GGUF in the store");
    CHECK(!exists(TRUNK) && !exists(HEAD), "no .fttw downloaded");
    CHECK(body_has(key, "/local/models/install/" ID, "\"total\":4096"), "install total = GGUF only");
    CHECK(row_installed(key, "k80-one") == 1, "k80-one row reads installed");
    CHECK(row_installed(key, "cpu") == 1, "cpu row reads installed");
    CHECK(row_installed(key, "p150-one") == 0, "p150-one row reads not installed");
    CHECK(body_has(key, "/local/catalog", "\"store_only\":[]"), "not listed as store_only");

    printf("box_hardware_install:\n");
    CHECK(install(key, NULL) == 200, "no profile_id on a CPU box: already installed");
    CHECK(!exists(TRUNK) && !exists(HEAD), "still no .fttw");

    printf("p150_profile_installs_fttw:\n");
    CHECK(install(key, "p150-one") == 202, "install p150-one accepted");
    CHECK(exists(GGUF) && exists(TRUNK) && exists(HEAD), "GGUF + both .fttw in the store");
    CHECK(row_installed(key, "p150-one") == 1, "p150-one row reads installed");
    CHECK(install(key, "nope") == 404, "unknown profile: 404");
    char *body = NULL;
    int st = lt_http(g_port, "POST", "/local/models/uninstall", key,
                     "{\"catalog_id\":\"" ID "\"}", &body);
    CHECK(st == 200 && body && strstr(body, "\"removed_files\":3"), "uninstall removes all 3");
    free(body);

    printf("per_unit_weights_p150_install:\n");
    double size = 0;
    CHECK(install_model(key, QID, "p150-one") == 202, "install q36 p150-one accepted");
    CHECK(exists(QMETA) && exists(QTRUNK) && exists(QHEAD), "metadata GGUF + both .fttw in the store");
    CHECK(!exists(QGGUF), "the K80 GGUF is NOT downloaded");
    CHECK(body_has(key, "/local/models/install/" QID, "\"total\":2816"),
          "install total = the three TT files");
    CHECK(row_of(key, QID, "p150-one", &size) == 1 && size == 2816,
          "p150-one row: installed, size = the three TT files");
    CHECK(row_of(key, QID, "k80-one", &size) == 0 && size == 4096,
          "k80-one row: not installed, size = the K80 GGUF");

    printf("per_unit_weights_k80_install:\n");
    CHECK(install_model(key, QID, "k80-one") == 202, "install q36 k80-one accepted");
    CHECK(exists(QGGUF), "the K80 GGUF in the store");
    CHECK(body_has(key, "/local/models/install/" QID, "\"total\":4096"),
          "install total = the K80 GGUF only");
    CHECK(row_of(key, QID, "k80-one", NULL) == 1, "k80-one row reads installed");
    body = NULL;
    st = lt_http(g_port, "POST", "/local/models/uninstall", key,
                 "{\"catalog_id\":\"" QID "\"}", &body);
    CHECK(st == 200 && body && strstr(body, "\"removed_files\":4"), "uninstall removes all 4");
    free(body);
    lt_stop(pid);
}

int main(void)
{
    char cwd[1024];

    if (!getcwd(cwd, sizeof(cwd)))
        return 1;
    g_port = lt_pick_port(6);
    g_httpd_port = lt_pick_port(7);
    snprintf(g_port_s, sizeof(g_port_s), "%d", g_port);
    snprintf(g_hf, sizeof(g_hf), "http://127.0.0.1:%d", g_httpd_port);
    snprintf(g_cat_url, sizeof(g_cat_url),
             "file://%s/tests/local/fixtures/catalog_unit_types.json", cwd);
    lt_clean_dir(DIR);
    lt_make_dir(DIR "/www/teamide/test-fixtures/resolve/" REV);
    lt_make_dir(DIR "/store");
    make_blob("orn-mini-Q8_0.gguf", 1, 4096);
    make_blob("orn-mini-Q8_0.bfp8.fttw", 2, 8192);
    make_blob("orn-mini-Q8_0.bfp8.mtp.fttw", 3, 1024);
    make_blob("q36-mini-Q8_0.gguf", 4, 4096);
    make_blob("q36-mini-Q8_0.meta.gguf", 5, 512);
    make_blob("q36-mini-Q8_0.meta.moe_bfp6.fttw", 6, 2048);
    make_blob("q36-mini-Q8_0.meta.moe_bfp6.mtp.fttw", 7, 256);

    fflush(stdout);
    pid_t httpd = fork();
    if (httpd == 0)
        _exit(fx_httpd_run(g_httpd_port, DIR "/www", 0, 0));
    lt_wait_port(g_httpd_port, 5000);
    run_flow();
    kill(httpd, SIGKILL);
    waitpid(httpd, NULL, 0);
    return lt_report("test_local_unit_types");
}
