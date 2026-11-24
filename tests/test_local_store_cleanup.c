/* SPDX-License-Identifier: AGPL-3.0-or-later */
/* Copyright (C) 2026 David (Corey) Herrera */
/*
 * test_local_store_cleanup.c — H2m: the store's entries are grouped by catalog
 * id (`<id>-<12 hex>.`), and an uninstall removes every entry of one id —
 * verified files, a download in progress, a companion link, the TT card
 * artifact beside the weights — and nothing of another id, even one whose id
 * starts with this one. Store under build/test_local/ (gitignored).
 */

#include "local_test_support.h"

#include "../src/local/model_store_cleanup.h"

#define STORE "build/test_local/store_cleanup"

static void put_file(const char *name, size_t bytes)
{
    char path[512];
    snprintf(path, sizeof(path), STORE "/%s", name);
    FILE *f = fopen(path, "w");
    for (size_t i = 0; f && i < bytes; i++)
        fputc('x', f);
    if (f)
        fclose(f);
}

static int exists(const char *name)
{
    char path[512];
    struct stat sb;
    snprintf(path, sizeof(path), STORE "/%s", name);
    return lstat(path, &sb) == 0;
}

static void test_entry_id(void)
{
    char id[64];

    printf("entry_id:\n");
    CHECK(pn_model_store_entry_id("qwen3.6-35b-a3b-q8-0bd466629210.gguf", id, sizeof(id)) == 0 &&
          strcmp(id, "qwen3.6-35b-a3b-q8") == 0, "id with dashes and dots");
    CHECK(pn_model_store_entry_id("ornith-1.5-9b-q8-a1b2c3d4e5f6.bfp8.mtp.fttw", id,
                                  sizeof(id)) == 0 && strcmp(id, "ornith-1.5-9b-q8") == 0,
          "companion / artifact name");
    CHECK(pn_model_store_entry_id("m-0123456789ab.gguf.part", id, sizeof(id)) == 0 &&
          strcmp(id, "m") == 0, "download in progress");
    CHECK(pn_model_store_entry_id("notes.txt", id, sizeof(id)) != 0, "not a store entry");
    CHECK(pn_model_store_entry_id("m-0123456789a.gguf", id, sizeof(id)) != 0, "11 hex is not");
    CHECK(pn_model_store_entry_id("m-0123456789ab.gguf", id, 1) != 0, "id too long for cap");
}

static void test_scan_and_remove(void)
{
    pn_store_group g[8];
    uint64_t freed = 0;
    int n_removed = 0;
    char err[256];

    printf("scan_and_remove:\n");
    lt_clean_dir(STORE);
    lt_make_dir(STORE);
    put_file("orn-aaaaaaaaaaaa.gguf", 100);
    put_file("orn-bbbbbbbbbbbb.fttw", 30);
    put_file("orn-aaaaaaaaaaaa.bfp8.fttw", 50);           /* TT card artifact */
    put_file("orn-cccccccccccc.gguf.part", 7);           /* download in progress */
    if (symlink("orn-bbbbbbbbbbbb.fttw", STORE "/orn-aaaaaaaaaaaa.mtp.fttw") != 0)
        perror("symlink");
    put_file("orn-2-dddddddddddd.gguf", 40);             /* another id, same prefix */
    put_file("README.txt", 3);
    int n = pn_model_store_scan(STORE, g, 8);
    int orn = -1, orn2 = -1;
    for (int i = 0; i < n; i++) {
        if (strcmp(g[i].catalog_id, "orn") == 0) orn = i;
        if (strcmp(g[i].catalog_id, "orn-2") == 0) orn2 = i;
    }
    CHECK(n == 2 && orn >= 0 && orn2 >= 0, "two ids in the store");
    CHECK(orn >= 0 && g[orn].n_files == 5 && g[orn].n_partial == 1 && g[orn].bytes == 187,
          "orn: 5 entries, 1 partial, 187 bytes (the link counts 0)");
    CHECK(pn_model_store_remove_id(STORE, "orn", &freed, &n_removed, err, sizeof(err)) == 0,
          "remove orn");
    CHECK(n_removed == 5 && freed == 187, "5 entries, 187 bytes freed");
    CHECK(!exists("orn-aaaaaaaaaaaa.gguf") && !exists("orn-aaaaaaaaaaaa.mtp.fttw") &&
          !exists("orn-aaaaaaaaaaaa.bfp8.fttw") && !exists("orn-cccccccccccc.gguf.part"),
          "weights, link, artifact and part gone");
    CHECK(exists("orn-2-dddddddddddd.gguf") && exists("README.txt"), "other id and other files kept");
    CHECK(pn_model_store_remove_id(STORE, "orn", &freed, &n_removed, err, sizeof(err)) == 0 &&
          n_removed == 0, "second removal finds nothing");
    CHECK(pn_model_store_remove_id(STORE "/missing", "orn", &freed, &n_removed, err,
                                   sizeof(err)) != 0, "unreadable store is an error");
}

int main(void)
{
    test_entry_id();
    test_scan_and_remove();
    return lt_report("test_local_store_cleanup");
}
