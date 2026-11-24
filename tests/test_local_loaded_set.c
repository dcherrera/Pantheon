/* SPDX-License-Identifier: AGPL-3.0-or-later */
/* Copyright (C) 2026 David (Corey) Herrera */
/*
 * test_local_loaded_set.c — Pantheon Local G2/G5: the remembered loaded set.
 * Save and restore round-trip every field, entries keep load order (the boot
 * reload walks them one at a time in that order), re-adding moves an entry to
 * the end, removal persists, the file is 0600, and a malformed file is refused
 * rather than silently emptied. L4: replicas of one model (the same catalog id
 * on other units) are separate entries, numbered 1, 2, ...; one or all of them
 * leave the set; a file written before replicas reads as replica 1.
 */

#include "local_test_support.h"

#include "../src/local/loaded_set.h"
#include "../src/local/local_file_io.h"

#define DIR "build/test_local/loaded_set"

static pn_loaded_entry entry(const char *id, int u0, int u1)
{
    pn_loaded_entry e;
    memset(&e, 0, sizeof(e));
    snprintf(e.catalog_id, sizeof(e.catalog_id), "%s", id);
    snprintf(e.profile_id, sizeof(e.profile_id), "k80-pair");
    snprintf(e.path, sizeof(e.path), "/store/%s-0123456789ab.gguf", id);
    e.units[0] = u0;
    e.units[1] = u1;
    e.n_units = 2;
    snprintf(e.hardware.unit_type, sizeof(e.hardware.unit_type), "k80-gpu");
    e.hardware.count = 2;
    e.hardware.same_card = true;
    e.hardware.min_mem_per_unit = 10000000000ull;
    e.ctx_cap = 32768;
    e.max_sessions = 2;
    e.max_concurrent = 1;
    e.reasoning_budget = 4096;
    e.mtp = true;
    snprintf(e.kv_type, sizeof(e.kv_type), "fp32");
    e.sampling.has_top_k = true;
    e.sampling.top_k = 20;
    e.sampling.has_temperature = true;
    e.sampling.temperature = 0.0f;
    return e;
}

int main(void)
{
    pn_loaded_entry a = entry("model-a", 6, 7), b = entry("model-b", 2, 3), out[8];
    struct stat sb;

    a.reasoning_reserve = 3072;   /* b keeps 0 = the serve default */

    printf("loaded_set:\n");
    lt_clean_dir(DIR);
    lt_make_dir(DIR);
    pn_loaded_set *s = pn_loaded_set_open(DIR "/loaded.json");
    CHECK(s && pn_loaded_set_list(s, out, 8) == 0, "missing file = empty set");
    CHECK(pn_loaded_set_add(s, &a) == 0 && pn_loaded_set_add(s, &b) == 0, "add a, b");
    CHECK(stat(DIR "/loaded.json", &sb) == 0 && (sb.st_mode & 0777) == 0600, "file is 0600");
    pn_loaded_set_close(s);

    s = pn_loaded_set_open(DIR "/loaded.json");
    int n = pn_loaded_set_list(s, out, 8);
    CHECK(n == 2 && strcmp(out[0].catalog_id, "model-a") == 0 &&
          strcmp(out[1].catalog_id, "model-b") == 0, "restored in load order");
    CHECK(out[0].n_units == 2 && out[0].units[0] == 6 && out[0].units[1] == 7, "units");
    CHECK(strcmp(out[0].hardware.unit_type, "k80-gpu") == 0 && out[0].hardware.same_card &&
          out[0].hardware.count == 2 && out[0].hardware.min_mem_per_unit == 10000000000ull,
          "hardware request");
    CHECK(out[0].ctx_cap == 32768 && out[0].max_sessions == 2 && out[0].max_concurrent == 1 &&
          out[0].reasoning_budget == 4096 && out[0].mtp, "load settings");
    CHECK(out[0].reasoning_reserve == 3072 && out[1].reasoning_reserve == 0,
          "reasoning_reserve kept; absent stays 0");
    CHECK(strcmp(out[0].kv_type, "fp32") == 0, "the KV choice (a reboot keeps fp32)");
    CHECK(out[0].sampling.has_top_k && out[0].sampling.top_k == 20 &&
          out[0].sampling.has_temperature && out[0].sampling.temperature == 0.0f &&
          !out[0].sampling.has_top_p && !out[0].sampling.has_presence_penalty,
          "sampling (set and unset fields)");
    CHECK(strcmp(out[1].path, "/store/model-b-0123456789ab.gguf") == 0, "path");

    a.units[0] = 0;
    a.units[1] = 1;
    CHECK(pn_loaded_set_add(s, &a) == 0, "re-add a");
    n = pn_loaded_set_list(s, out, 8);
    CHECK(n == 2 && strcmp(out[1].catalog_id, "model-a") == 0 && out[1].units[0] == 0,
          "re-added entry replaced and moved to the end");
    CHECK(pn_loaded_set_remove(s, "model-b", 0) == 0, "remove b");
    CHECK(pn_loaded_set_remove(s, "model-b", 0) == 1, "removing again is a no-op");
    pn_loaded_set_close(s);
    s = pn_loaded_set_open(DIR "/loaded.json");
    CHECK(pn_loaded_set_list(s, out, 8) == 1 && strcmp(out[0].catalog_id, "model-a") == 0,
          "removal persisted");
    pn_loaded_set_close(s);

    /* L4: replicas. */
    s = pn_loaded_set_open(DIR "/loaded.json");
    CHECK(pn_loaded_set_next_replica(s, "model-a") == 2, "model-a is loaded: next copy is 2");
    CHECK(pn_loaded_set_next_replica(s, "model-c") == 1, "a new model starts at 1");
    pn_loaded_entry a2 = entry("model-a", 2, 3), a3 = entry("model-a", 4, 5);
    a2.replica = 2;
    a3.replica = 3;
    CHECK(pn_loaded_set_add(s, &a2) == 0 && pn_loaded_set_add(s, &a3) == 0, "add copies 2 and 3");
    CHECK(pn_loaded_set_list(s, out, 8) == 3, "three replicas, three entries");
    CHECK(pn_loaded_set_remove(s, "model-a", 2) == 0, "remove copy 2 only");
    CHECK(pn_loaded_set_next_replica(s, "model-a") == 2, "its number is free again");
    pn_loaded_set_close(s);
    s = pn_loaded_set_open(DIR "/loaded.json");
    n = pn_loaded_set_list(s, out, 8);
    CHECK(n == 2 && out[0].replica == 1 && out[1].replica == 3 && out[1].units[0] == 4,
          "copies 1 and 3 restored, with their units");
    CHECK(pn_loaded_set_remove(s, "model-a", 0) == 0 && pn_loaded_set_list(s, out, 8) == 0,
          "remove every replica");
    pn_loaded_set_close(s);

    /* A p150 profile's MTP draft count survives a reboot; 0 (every other profile) writes no
     * key, so their files are what an older box wrote and reads. */
    s = pn_loaded_set_open(DIR "/loaded.json");
    pn_loaded_entry tt = entry("model-tt", 0, 0), plain = entry("model-plain", 1, 1);
    snprintf(tt.hardware.unit_type, sizeof(tt.hardware.unit_type), "p150");
    snprintf(tt.kv_type, sizeof(tt.kv_type), "k8v4");
    tt.mtp_drafts = 1;
    CHECK(pn_loaded_set_add(s, &tt) == 0 && pn_loaded_set_add(s, &plain) == 0, "add tt, plain");
    pn_loaded_set_close(s);
    char *text = NULL;
    int keys = 0;
    CHECK(pn_local_read_file(DIR "/loaded.json", 1u << 20, &text, NULL) == 0 && text,
          "read loaded.json");
    for (const char *at = text; at && (at = strstr(at, "\"mtp_drafts\"")) != NULL; at++)
        keys++;
    CHECK(text && strstr(text, "\"mtp_drafts\":1,") && keys == 1,
          "mtp_drafts written for the p150 entry only");
    free(text);
    s = pn_loaded_set_open(DIR "/loaded.json");
    n = pn_loaded_set_list(s, out, 8);
    CHECK(n == 2 && out[0].mtp_drafts == 1 && strcmp(out[0].kv_type, "k8v4") == 0 &&
              out[1].mtp_drafts == 0, "mtp_drafts restored (1 and the default 0)");
    CHECK(pn_loaded_set_remove(s, "model-tt", 0) == 0 && pn_loaded_set_remove(s, "model-plain", 0) == 0,
          "remove tt, plain");
    pn_loaded_set_close(s);

    FILE *f = fopen(DIR "/loaded.json", "w");
    fputs("{\"version\":1,\"models\":[{\"catalog_id\":\"old\",\"profile_id\":\"p\","
          "\"path\":\"/m.gguf\",\"units\":[0],\"hardware\":{\"unit_type\":\"p150\"}}]}", f);
    fclose(f);
    s = pn_loaded_set_open(DIR "/loaded.json");
    CHECK(s && pn_loaded_set_list(s, out, 8) == 1 && out[0].replica == 1,
          "a file from before replicas: replica 1");
    pn_loaded_set_close(s);

    f = fopen(DIR "/loaded.json", "w");
    fputs("{\"models\":[{\"catalog_id\":1}]}", f);
    fclose(f);
    CHECK(pn_loaded_set_open(DIR "/loaded.json") == NULL, "malformed file refused");
    return lt_report("test_local_loaded_set");
}
