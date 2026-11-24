/* SPDX-License-Identifier: AGPL-3.0-or-later */
/* Copyright (C) 2026 David (Corey) Herrera */
/*
 * test_local_key_migration.c — people registry P1.2: keys.json v2.
 *   - a v1 fixture (tests/local/fixtures/people_registry/keys_v1.json) migrates:
 *     admin → device, inference → inference, both of person 1 `owner` (admin);
 *     keys.json.v1 keeps the old bytes (0600); the old secrets still verify;
 *   - a crash between the people save and the keys save re-runs cleanly (the
 *     existing admin is reused, no second owner);
 *   - v2 round trip: device claim once (mismatch refused), rename, list per
 *     person, revoke per person; malformed v2 refused; 256 keys max.
 */

#include "local_test_support.h"

#include "../src/local/key_migration.h"
#include "../src/local/key_store.h"
#include "../src/local/people_store.h"

#define STATE   "build/test_local/keymig"
#define FIXTURE "tests/local/fixtures/people_registry/keys_v1.json"
#define V1_DEVICE_SECRET "pnk_1111111111111111111111111111111111111111111111111111111111111111"
#define V1_INFER_SECRET  "pnk_2222222222222222222222222222222222222222222222222222222222222222"

static char *slurp(const char *path)
{
    FILE *f = fopen(path, "r");
    char *buf = NULL;
    long n = 0;

    if (!f)
        return NULL;
    fseek(f, 0, SEEK_END);
    n = ftell(f);
    fseek(f, 0, SEEK_SET);
    buf = calloc(1, (size_t)n + 1);
    if (buf && fread(buf, 1, (size_t)n, f) != (size_t)n) {
        free(buf);
        buf = NULL;
    }
    fclose(f);
    return buf;
}

static void spit(const char *path, const char *text)
{
    FILE *f = fopen(path, "w");
    if (!f)
        return;
    fputs(text, f);
    fclose(f);
}

static void test_v1_migrates(void)
{
    pn_key_info dev, inf;
    pn_person rows[4];
    struct stat sb;

    printf("v1_migrates:\n");
    lt_clean_dir(STATE "/a");
    lt_make_dir(STATE "/a");
    char *v1 = slurp(FIXTURE);
    CHECK(v1 != NULL, "fixture readable");
    spit(STATE "/a/keys.json", v1 ? v1 : "");
    pn_people *pp = pn_people_open(STATE "/a/people.json");
    pn_key_store *ks = pn_key_store_open(STATE "/a/keys.json");
    CHECK(pp && ks && pn_key_store_is_v1(ks), "v1 file opens as v1");
    CHECK(ks && pn_key_store_verify(ks, V1_DEVICE_SECRET, &dev) != 0,
          "nothing verifies before the migration");
    CHECK(pn_key_migrate_v1(ks, pp, STATE "/a/keys.json.v1") == 0, "migrate");
    CHECK(pn_people_list(pp, rows, 4) == 1 && strcmp(rows[0].handle, "owner") == 0 &&
          strcmp(rows[0].name, "Owner") == 0 && rows[0].admin && rows[0].admin_by[0] == '\0',
          "person 1 owner (admin, root)");
    CHECK(pn_key_store_verify(ks, V1_DEVICE_SECRET, &dev) == 0 && dev.kind == PN_KEY_KIND_DEVICE &&
          strcmp(dev.id, "k0a0b0c0d") == 0 && strcmp(dev.person, rows[0].id) == 0 &&
          dev.device_id[0] == '\0' && strcmp(dev.device_name, "argus") == 0,
          "admin key -> device of person 1, device_name = label, unclaimed");
    CHECK(pn_key_store_verify(ks, V1_INFER_SECRET, &inf) == 0 &&
          inf.kind == PN_KEY_KIND_INFERENCE && strcmp(inf.person, rows[0].id) == 0 &&
          inf.created == 1790000100, "inference key -> inference of person 1");
    char *kept = slurp(STATE "/a/keys.json.v1");
    CHECK(kept && v1 && strcmp(kept, v1) == 0, "keys.json.v1 holds the v1 bytes");
    CHECK(stat(STATE "/a/keys.json.v1", &sb) == 0 && (sb.st_mode & 0777) == 0600,
          "keys.json.v1 0600");
    char *now = slurp(STATE "/a/keys.json");
    CHECK(now && strstr(now, "\"version\":2") && strstr(now, "\"kind\":\"device\""),
          "keys.json is v2");
    free(kept);
    free(now);
    pn_key_store_close(ks);
    pn_people_close(pp);

    pp = pn_people_open(STATE "/a/people.json");
    ks = pn_key_store_open(STATE "/a/keys.json");
    CHECK(ks && !pn_key_store_is_v1(ks), "reopens as v2");
    CHECK(pn_key_migrate_v1(ks, pp, STATE "/a/keys.json.v1") == 0 &&
          pn_people_list(pp, rows, 4) == 1, "re-run is a no-op");
    pn_key_store_close(ks);
    pn_people_close(pp);
    free(v1);
}

/* people.json saved, then the process dies before keys.json is rewritten. */
static void test_crash_between_saves(void)
{
    pn_person owner, rows[4];
    pn_key_info dev;
    int made = 0;

    printf("crash_between_saves:\n");
    lt_clean_dir(STATE "/b");
    lt_make_dir(STATE "/b");
    char *v1 = slurp(FIXTURE);
    spit(STATE "/b/keys.json", v1 ? v1 : "");
    pn_people *pp = pn_people_open(STATE "/b/people.json");
    CHECK(pn_people_owner_or_create(pp, "owner", "Owner", &owner, &made) == 0 && made,
          "first half: the owner is saved");
    pn_people_close(pp);

    pp = pn_people_open(STATE "/b/people.json");
    pn_key_store *ks = pn_key_store_open(STATE "/b/keys.json");
    CHECK(ks && pn_key_store_is_v1(ks), "keys.json still v1 after the crash");
    CHECK(pn_key_migrate_v1(ks, pp, STATE "/b/keys.json.v1") == 0, "re-run migrates");
    CHECK(pn_people_list(pp, rows, 4) == 1, "no second owner");
    CHECK(pn_key_store_verify(ks, V1_DEVICE_SECRET, &dev) == 0 &&
          strcmp(dev.person, owner.id) == 0, "keys belong to the existing admin");
    pn_key_store_close(ks);
    pn_people_close(pp);
    free(v1);
}

static void test_empty_v1(void)
{
    pn_person rows[2];

    printf("empty_v1:\n");
    lt_clean_dir(STATE "/c");
    lt_make_dir(STATE "/c");
    spit(STATE "/c/keys.json", "{\"version\":1,\"keys\":[]}\n");
    pn_people *pp = pn_people_open(STATE "/c/people.json");
    pn_key_store *ks = pn_key_store_open(STATE "/c/keys.json");
    CHECK(pn_key_migrate_v1(ks, pp, STATE "/c/keys.json.v1") == 0 && !pn_key_store_is_v1(ks),
          "empty v1 migrates");
    CHECK(pn_people_list(pp, rows, 2) == 0, "no person made for a box never paired");
    pn_key_store_close(ks);
    pn_people_close(pp);
}

static void test_v2_round_trip(void)
{
    char sd[PN_KEY_SECRET_MAX], si[PN_KEY_SECRET_MAX], so[PN_KEY_SECRET_MAX];
    pn_key_info dev, inf, oth, got, rows[8];

    printf("v2_round_trip:\n");
    lt_clean_dir(STATE "/d");
    lt_make_dir(STATE "/d");
    pn_key_store *ks = pn_key_store_open(STATE "/d/keys.json");
    CHECK(pn_key_store_mint_device(ks, "p00000001", "d0123456789abcdef", "mac-studio", "argus",
                                   sd, &dev) == PN_KEY_OK, "mint device");
    CHECK(pn_key_store_mint_device(ks, "p00000001", "dXYZ", "x", "x", so, &oth) ==
          PN_KEY_E_INVALID, "bad device_id refused");
    CHECK(pn_key_store_mint_device(ks, "p00000001", NULL, "bad\x01", "x", so, &oth) ==
          PN_KEY_E_INVALID, "bad device_name refused");
    CHECK(pn_key_store_mint_inference(ks, "p00000001", "scripts", si, &inf) == PN_KEY_OK,
          "mint inference");
    CHECK(pn_key_store_mint_device(ks, "p00000002", NULL, "phone", "argus", so, &oth) ==
          PN_KEY_OK, "mint unclaimed device of person 2");
    CHECK(pn_key_store_set_device(ks, oth.id, "d00000000000000aa", "Phone", &got) == PN_KEY_OK &&
          strcmp(got.device_id, "d00000000000000aa") == 0, "claim");
    CHECK(pn_key_store_set_device(ks, oth.id, "d00000000000000aa", "Pixel", &got) == PN_KEY_OK &&
          strcmp(got.device_name, "Pixel") == 0, "same id again renames");
    CHECK(pn_key_store_set_device(ks, oth.id, "d00000000000000bb", NULL, &got) ==
          PN_KEY_E_DEVICE_MISMATCH, "a different id is refused");
    CHECK(pn_key_store_set_device(ks, inf.id, "d00000000000000bb", NULL, &got) ==
          PN_KEY_E_INVALID, "inference keys have no device");
    pn_key_store_close(ks);

    ks = pn_key_store_open(STATE "/d/keys.json");
    CHECK(ks && pn_key_store_verify(ks, sd, &got) == 0 && got.kind == PN_KEY_KIND_DEVICE &&
          strcmp(got.device_id, "d0123456789abcdef") == 0 &&
          strcmp(got.device_name, "mac-studio") == 0 && strcmp(got.person, "p00000001") == 0 &&
          strcmp(got.label, "argus") == 0, "device row round trip");
    CHECK(pn_key_store_verify(ks, so, &got) == 0 && strcmp(got.device_name, "Pixel") == 0,
          "claim + rename saved");
    CHECK(pn_key_store_list_person(ks, "p00000001", rows, 8) == 2, "two keys of person 1");
    CHECK(pn_key_store_revoke_person(ks, "p00000001") == 2, "revoke person 1");
    CHECK(pn_key_store_verify(ks, sd, &got) != 0 && pn_key_store_verify(ks, si, &got) != 0,
          "person 1's keys refused");
    CHECK(pn_key_store_verify(ks, so, &got) == 0, "person 2 unaffected");
    pn_key_store_close(ks);
    ks = pn_key_store_open(STATE "/d/keys.json");
    CHECK(ks && pn_key_store_count(ks) == 1, "revocation saved");
    pn_key_store_close(ks);
}

static void test_malformed_and_cap(void)
{
    static const char *bad[] = {
        "{\"version\":3,\"keys\":[]}",
        "{\"version\":2,\"keys\":[{\"id\":\"k1\",\"sha256\":\"%s\",\"kind\":\"admin\","
        "\"person\":\"p00000001\",\"label\":\"x\",\"created\":1}]}",
        "{\"version\":2,\"keys\":[{\"id\":\"k1\",\"sha256\":\"%s\",\"kind\":\"inference\","
        "\"person\":\"bob\",\"label\":\"x\",\"created\":1}]}",
        "{\"version\":2,\"keys\":[{\"id\":\"k1\",\"sha256\":\"%s\",\"kind\":\"device\","
        "\"person\":\"p00000001\",\"device_id\":\"d12\",\"device_name\":\"x\",\"label\":\"x\","
        "\"created\":1}]}",
    };
    const char *why[] = { "newer version refused", "unknown kind refused",
                          "bad person refused", "bad device_id refused" };
    const char *hash = "864f714005ef439dd3bdbe20faa6afd92317a0e74723c94e6ff288577f12cc2d";
    char text[1024], secret[PN_KEY_SECRET_MAX];
    int minted = 0;

    printf("malformed_and_cap:\n");
    lt_clean_dir(STATE "/e");
    lt_make_dir(STATE "/e");
    for (size_t i = 0; i < sizeof(bad) / sizeof(bad[0]); i++) {
        snprintf(text, sizeof(text), bad[i], hash);
        spit(STATE "/e/keys.json", text);
        pn_key_store *ks = pn_key_store_open(STATE "/e/keys.json");
        CHECK(ks == NULL, why[i]);
        pn_key_store_close(ks);
    }
    unlink(STATE "/e/keys.json");
    pn_key_store *ks = pn_key_store_open(STATE "/e/keys.json");
    for (int i = 0; i < PN_KEY_MAX_KEYS; i++)
        minted += pn_key_store_mint_inference(ks, "p00000001", "s", secret, NULL) == PN_KEY_OK;
    CHECK(PN_KEY_MAX_KEYS == 256 && minted == 256, "256 keys fit");
    CHECK(pn_key_store_mint_inference(ks, "p00000001", "s", secret, NULL) == PN_KEY_E_FULL,
          "257th refused");
    pn_key_store_close(ks);
}

int main(void)
{
    test_v1_migrates();
    test_crash_between_saves();
    test_empty_v1();
    test_v2_round_trip();
    test_malformed_and_cap();
    return lt_report("test_local_key_migration");
}
