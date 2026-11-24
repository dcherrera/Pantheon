/* SPDX-License-Identifier: AGPL-3.0-or-later */
/* Copyright (C) 2026 David (Corey) Herrera */
/*
 * test_local_download.c — Pantheon Local F1-F5: the model store and the
 * downloader against a local HTTP fixture (tests/local_fixture_httpd.h).
 *   - store naming <id>-<sha12>.<ext>, installed only at the final name;
 *   - companion extras (<weights stem>.<rest>) linked as <id>-<weights sha12>.<rest>,
 *     only those the unit type needs (files[].unit_types); the shipped Qwen3.6 p150
 *     install links its trunk + MTP .fttw beside its metadata GGUF;
 *   - free-space refusal;
 *   - a download cut mid-way leaves a .part that the next run RESUMES (Range);
 *   - a sha256 mismatch deletes the file and refuses;
 *   - the install worker runs one download at a time and reports progress;
 *   - URLs are built from validated manifest fields only.
 */

#include "local_test_support.h"
#include "local_fixture_httpd.h"

#include "../src/local/curl_runner.h"
#include "../src/local/downloader.h"
#include "../src/local/install_worker.h"
#include "../src/local/model_store.h"
#include "../src/local/sha256.h"

#define DIR   "build/test_local/download"
#define REV   "0123456789abcdef0123456789abcdef01234567"
#define BLOB  (3 * 1024 * 1024 + 123)

static char g_sha[65];

/* A deterministic 3 MB "model" in the fixture's HF layout. */
static void make_blob(const char *name)
{
    char path[512];
    snprintf(path, sizeof(path), DIR "/www/teamide/test-fixtures/resolve/" REV "/%s", name);
    FILE *f = fopen(path, "wb");
    pn_sha256_ctx c;
    pn_sha256_init(&c);
    uint32_t x = 12345;
    for (int i = 0; i < BLOB; i++) {
        x = x * 1103515245u + 12345u;
        uint8_t b = (uint8_t)(x >> 16);
        fputc(b, f);
        pn_sha256_update(&c, &b, 1);
    }
    fclose(f);
    uint8_t d[32];
    pn_sha256_final(&c, d);
    pn_hex_encode(d, 32, g_sha);
}

static void make_model(pn_cat_model *m, const char *id, const char *file, const char *sha)
{
    memset(m, 0, sizeof(*m));
    snprintf(m->id, sizeof(m->id), "%s", id);
    m->n_files = 1;
    pn_cat_file *f = &m->files[0];
    snprintf(f->role, sizeof(f->role), "weights");
    snprintf(f->hf_repo, sizeof(f->hf_repo), "teamide/test-fixtures");
    snprintf(f->revision, sizeof(f->revision), REV);
    snprintf(f->filename, sizeof(f->filename), "%s", file);
    snprintf(f->sha256, sizeof(f->sha256), "%s", sha);
    f->size = BLOB;
}

static void test_store_naming_and_space(void)
{
    pn_cat_model m;
    char path[1024], why[200];

    printf("store_naming_and_space:\n");
    make_model(&m, "tiny-model", "tiny.Q8_0.gguf", g_sha);
    CHECK(pn_model_store_path(DIR "/store", &m, &m.files[0], path, sizeof(path)) == 0, "path");
    char want[256];
    snprintf(want, sizeof(want), DIR "/store/tiny-model-%.12s.gguf", g_sha);
    CHECK(strcmp(path, want) == 0, "named <id>-<sha12>.gguf");
    CHECK(!pn_model_store_installed(DIR "/store", &m), "not installed yet");
    CHECK(pn_model_store_check_space(DIR "/store", (uint64_t)1 << 60, why, sizeof(why)) == -1 &&
          strstr(why, "needs"), "free-space refusal");
    CHECK(pn_model_store_check_space(DIR "/store", 1024, why, sizeof(why)) == 0, "1 KB fits");
    char url[1024];
    pn_cat_file_url(NULL, &m.files[0], url, sizeof(url));
    CHECK(strcmp(url, "https://huggingface.co/teamide/test-fixtures/resolve/" REV
                      "/tiny.Q8_0.gguf") == 0, "HF URL from repo + pinned revision");
}

static void test_resume_after_cut(int port)
{
    pn_cat_model m;
    char base[64], err[256], final[1024], part[1100];

    printf("resume_after_cut:\n");
    snprintf(base, sizeof(base), "http://127.0.0.1:%d", port);
    pn_download_cfg cfg = { DIR "/store", base, PN_CURL_PROTOCOLS_ANY, NULL };
    make_model(&m, "tiny-model", "tiny.Q8_0.gguf", g_sha);
    pn_model_store_path(cfg.store_dir, &m, &m.files[0], final, sizeof(final));
    snprintf(part, sizeof(part), "%s.part", final);

    pn_download_cfg https_only = cfg;
    https_only.protocols = PN_CURL_PROTOCOLS_HTTPS;
    CHECK(pn_download_file(&https_only, &m, &m.files[0], NULL, NULL, err, sizeof(err)) == -1,
          "http refused under https-only");

    CHECK(pn_download_file(&cfg, &m, &m.files[0], NULL, NULL, err, sizeof(err)) == -1,
          "first attempt is cut by the server");
    uint64_t cut = pn_model_store_file_size(part);
    CHECK(cut == 1024 * 1024, "part file holds the bytes before the cut");
    CHECK(pn_model_store_file_size(final) == 0, "nothing at the final name");
    CHECK(pn_download_file(&cfg, &m, &m.files[0], NULL, NULL, err, sizeof(err)) == 0, err);
    CHECK(pn_model_store_file_size(final) == BLOB, "resumed to the full size");
    CHECK(pn_model_store_file_size(part) == 0, "part file gone");
    CHECK(pn_model_store_installed(cfg.store_dir, &m), "installed");
    uint8_t d[32];
    char hex[65];
    pn_sha256_file(final, d);
    pn_hex_encode(d, 32, hex);
    CHECK(strcmp(hex, g_sha) == 0, "resumed file hashes right");
}

static void test_mismatch_deletes(int port)
{
    pn_cat_model m;
    char base[64], err[256], final[1024], part[1100], wrong[65];

    printf("mismatch_deletes:\n");
    snprintf(base, sizeof(base), "http://127.0.0.1:%d", port);
    pn_download_cfg cfg = { DIR "/store", base, PN_CURL_PROTOCOLS_ANY, NULL };
    snprintf(wrong, sizeof(wrong), "%s", g_sha);
    wrong[0] = wrong[0] == 'a' ? 'b' : 'a';
    make_model(&m, "poisoned", "evil.gguf", wrong);
    pn_model_store_path(cfg.store_dir, &m, &m.files[0], final, sizeof(final));
    snprintf(part, sizeof(part), "%s.part", final);
    /* evil.gguf is cut on its first fetch too; run until it is fully fetched. */
    int rc = pn_download_file(&cfg, &m, &m.files[0], NULL, NULL, err, sizeof(err));
    if (rc != 0 && !strstr(err, "sha256"))
        rc = pn_download_file(&cfg, &m, &m.files[0], NULL, NULL, err, sizeof(err));
    CHECK(rc == -1 && strstr(err, "sha256 mismatch"), "mismatch refused");
    CHECK(pn_model_store_file_size(part) == 0 && pn_model_store_file_size(final) == 0,
          "mismatched file deleted, never at the final name");
    CHECK(!pn_model_store_installed(cfg.store_dir, &m), "not installed");
}

static void test_install_worker(int port)
{
    pn_cat_model a, b;
    pn_install_status st;
    char base[64], err[256];

    printf("install_worker:\n");
    snprintf(base, sizeof(base), "http://127.0.0.1:%d", port);
    pn_download_cfg cfg = { DIR "/store", base, PN_CURL_PROTOCOLS_ANY, NULL };
    make_model(&a, "worker-model", "worker.gguf", g_sha);
    make_model(&b, "other-model", "worker.gguf", g_sha);
    pn_install_worker *w = pn_install_worker_start(&cfg);
    CHECK(pn_install_worker_submit(w, &a, NULL, err, sizeof(err)) == 0, "submit");
    CHECK(pn_install_worker_submit(w, &b, NULL, err, sizeof(err)) == -1 &&
          strstr(err, "one at a time"),
          "second install refused while busy");
    for (int i = 0; i < 400; i++) {
        pn_install_worker_status(w, "worker-model", &st);
        if (st.state == PN_INSTALL_DONE || st.state == PN_INSTALL_FAILED) break;
        usleep(50000);
    }
    /* The fixture cuts worker.gguf's first fetch: the worker reports a failure
     * with the part kept, and a resubmit resumes it. */
    if (st.state == PN_INSTALL_FAILED) {
        CHECK(st.bytes == 0 && st.total == BLOB, "failed run reports progress fields");
        pn_install_worker_submit(w, &a, NULL, err, sizeof(err));
        for (int i = 0; i < 400; i++) {
            pn_install_worker_status(w, "worker-model", &st);
            if (st.state == PN_INSTALL_DONE || st.state == PN_INSTALL_FAILED) break;
            usleep(50000);
        }
    }
    CHECK(st.state == PN_INSTALL_DONE && st.bytes == BLOB, "installed through the worker");
    CHECK(pn_install_worker_status(w, "never", &st) == -1, "unknown id");
    pn_install_worker_stop(w);
}

/* Write a small file at `path`. */
static void touch_file(const char *path)
{
    FILE *f = fopen(path, "wb");
    if (f) {
        fputs("x", f);
        fclose(f);
    }
}

static void test_companion_links(void)
{
    const char *w = "1111111111112222222222222222222222222222222222222222222222222222";
    const char *h1 = "aaaaaaaaaaaabbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb";
    const char *h2 = "ccccccccccccdddddddddddddddddddddddddddddddddddddddddddddddddddd";
    const char *st = DIR "/comp";
    char path[1024], cur[256], err[200];
    pn_cat_model m;

    printf("companion_links:\n");
    lt_make_dir(st);
    make_model(&m, "orn", "Orn-Q8_0.gguf", w);
    m.n_files = 3;
    m.files[1] = m.files[0];
    snprintf(m.files[1].role, sizeof(m.files[1].role), "extra");
    snprintf(m.files[1].filename, sizeof(m.files[1].filename), "Orn-Q8_0.bfp8.mtp.fttw");
    snprintf(m.files[1].sha256, sizeof(m.files[1].sha256), "%s", h1);
    m.files[2] = m.files[1];
    snprintf(m.files[2].filename, sizeof(m.files[2].filename), "notes.txt");
    for (int i = 0; i < 3; i++) {
        pn_model_store_path(st, &m, &m.files[i], path, sizeof(path));
        touch_file(path);
    }
    CHECK(pn_model_store_link_companions(st, &m, err, sizeof(err)) == 0, err);
    ssize_t n = readlink(DIR "/comp/orn-111111111111.bfp8.mtp.fttw", cur, sizeof(cur) - 1);
    cur[n > 0 ? n : 0] = '\0';
    CHECK(strcmp(cur, "orn-aaaaaaaaaaaa.fttw") == 0, "MTP head linked at <id>-<w sha12>.bfp8.mtp.fttw");
    CHECK(access(DIR "/comp/orn-111111111111.txt", F_OK) != 0, "a non-companion extra is not linked");
    CHECK(pn_model_store_link_companions(st, &m, err, sizeof(err)) == 0, "idempotent");
    {   /* spec §16: a K80 install has no p150-only head; its links skip it. */
        pn_cat_model k80;
        m.files[1].unit_types = PN_CAT_UNIT_P150;
        unlink(DIR "/comp/orn-111111111111.bfp8.mtp.fttw");
        pn_cat_model_for_units(&m, PN_CAT_UNIT_K80_GPU, &k80);
        CHECK(pn_model_store_link_companions(st, &k80, err, sizeof(err)) == 0, err);
        CHECK(access(DIR "/comp/orn-111111111111.bfp8.mtp.fttw", F_OK) != 0 &&
              readlink(DIR "/comp/orn-111111111111.bfp8.mtp.fttw", cur, sizeof(cur)) < 0,
              "k80: the p150-only head is not linked");
        m.files[1].unit_types = 0;
        CHECK(pn_model_store_link_companions(st, &m, err, sizeof(err)) == 0, "relinked");
    }
    snprintf(m.files[1].sha256, sizeof(m.files[1].sha256), "%s", h2);
    pn_model_store_path(st, &m, &m.files[1], path, sizeof(path));
    touch_file(path);
    CHECK(pn_model_store_link_companions(st, &m, err, sizeof(err)) == 0, err);
    n = readlink(DIR "/comp/orn-111111111111.bfp8.mtp.fttw", cur, sizeof(cur) - 1);
    cur[n > 0 ? n : 0] = '\0';
    CHECK(strcmp(cur, "orn-cccccccccccc.fttw") == 0, "a new head in a new catalog re-points the link");
    snprintf(m.files[2].filename, sizeof(m.files[2].filename), "Orn-Q8_0.gguf");
    CHECK(pn_model_store_link_companions(st, &m, err, sizeof(err)) == -1,
          "an extra that would replace the weights file is refused");
}

/* The shipped catalog's Qwen3.6 p150 install (spec §16): the store links the trunk and
 * the MTP head beside the metadata GGUF, where pn_load / the TT backend look for them
 * (<meta store stem>.moe_bfp6.fttw, <...>.moe_bfp6.mtp.fttw). */
static void test_qwen36_p150_links(void)
{
    pn_catalog_manifest *cat = calloc(1, sizeof(*cat));
    pn_cat_model *tt = calloc(1, sizeof(*tt));
    const char *st = DIR "/q36";
    char path[1024], cur[256], err[256], want[128];
    FILE *f = fopen("catalog/catalog.json", "rb");
    char *json = calloc(1, 1 << 20);
    const pn_cat_model *q;

    printf("qwen36_p150_links:\n");
    lt_make_dir(st);
    size_t n = f && json ? fread(json, 1, (1 << 20) - 1, f) : 0;
    if (f)
        fclose(f);
    CHECK(n > 0 && pn_cat_manifest_parse(json, cat, err, sizeof(err)) == 0, err);
    q = pn_cat_find_model(cat, "qwen3.6-35b-a3b-q8");
    CHECK(q != NULL, "qwen3.6-35b-a3b-q8 in the shipped catalog");
    if (q) {
        pn_cat_model k80;
        pn_cat_model_for_units(q, PN_CAT_UNIT_K80_GPU, &k80);
        CHECK(pn_model_store_weights_path(st, &k80, path, sizeof(path)) == 0 &&
              strcmp(path, DIR "/q36/qwen3.6-35b-a3b-q8-0bd466629210.gguf") == 0,
              "a K80 load opens the K80 GGUF");
        pn_cat_model_for_units(q, PN_CAT_UNIT_P150, tt);
        CHECK(pn_model_store_weights_path(st, tt, path, sizeof(path)) == 0 &&
              strcmp(path, DIR "/q36/qwen3.6-35b-a3b-q8-90f46deabc2f.gguf") == 0,
              "a p150 load opens the metadata GGUF");
        for (int i = 0; i < tt->n_files; i++) {
            pn_model_store_path(st, tt, &tt->files[i], path, sizeof(path));
            touch_file(path);
        }
        CHECK(pn_model_store_link_companions(st, tt, err, sizeof(err)) == 0, err);
        const char *rest[] = { "moe_bfp6.fttw", "moe_bfp6.mtp.fttw" };
        for (int i = 0; i < 2; i++) {
            snprintf(path, sizeof(path), "%s/qwen3.6-35b-a3b-q8-%.12s.%s", st,
                     pn_cat_weights_file(tt)->sha256, rest[i]);
            ssize_t len = readlink(path, cur, sizeof(cur) - 1);
            cur[len > 0 ? len : 0] = '\0';
            snprintf(want, sizeof(want), "qwen3.6-35b-a3b-q8-%.12s.fttw", tt->files[i + 1].sha256);
            CHECK(strcmp(cur, want) == 0, rest[i]);
        }
    }
    free(json);
    free(tt);
    free(cat);
}

int main(void)
{
    int port = lt_pick_port(3);

    lt_clean_dir(DIR);
    lt_make_dir(DIR "/store");
    lt_make_dir(DIR "/www/teamide/test-fixtures/resolve/" REV);
    make_blob("tiny.Q8_0.gguf");
    make_blob("evil.gguf");
    make_blob("worker.gguf");
    test_store_naming_and_space();
    test_companion_links();
    test_qwen36_p150_links();
    if (!pn_curl_path()) {
        printf("  [SKIP] curl not installed\n");
        return lt_report("test_local_download");
    }
    fflush(stdout);
    pid_t httpd = fork();
    if (httpd == 0)
        _exit(fx_httpd_run(port, DIR "/www", 1024 * 1024, 0));
    lt_wait_port(port, 5000);
    test_resume_after_cut(port);
    test_mismatch_deletes(port);
    test_install_worker(port);
    kill(httpd, SIGKILL);
    waitpid(httpd, NULL, 0);
    return lt_report("test_local_download");
}
