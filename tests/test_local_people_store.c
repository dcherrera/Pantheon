/* SPDX-License-Identifier: AGPL-3.0-or-later */
/* Copyright (C) 2026 David (Corey) Herrera */
/*
 * test_local_people_store.c — people registry P1.1: people.json.
 *   - handle / name rules: trimmed, too long refused (never cut), control
 *     characters and broken UTF-8 refused, handles unique ignoring ASCII case;
 *   - round trip through the file (0600), malformed files refused;
 *   - the last admin can be neither removed nor demoted;
 *   - the admin tree (spec §3.1): a child admin cannot remove or demote its
 *     maker or a sibling; a maker removes its descendants; children move up when
 *     their admin leaves; the earliest-made child becomes the root;
 *   - 64 people max; a failed save leaves memory unchanged.
 */

#include "local_test_support.h"

#include "../src/local/people_store.h"

#define STATE "build/test_local/people"

static void write_text(const char *path, const char *text)
{
    FILE *f = fopen(path, "w");
    if (!f)
        return;
    fputs(text, f);
    fclose(f);
}

static void test_name_rules(void)
{
    char out[PN_PERSON_NAME_MAX];

    printf("name_rules:\n");
    CHECK(pn_people_name_clean("  david\t", 24, out, sizeof(out)) == 0 &&
          strcmp(out, "david") == 0, "trimmed");
    CHECK(pn_people_name_clean("abcdefghijklmnopqrstuvwx", 24, out, sizeof(out)) == 0,
          "24 bytes accepted");
    CHECK(pn_people_name_clean("abcdefghijklmnopqrstuvwxy", 24, out, sizeof(out)) != 0,
          "25 bytes refused, not cut");
    CHECK(pn_people_name_clean("Jos\xc3\xa9", 24, out, sizeof(out)) == 0, "UTF-8 accepted");
    CHECK(pn_people_name_clean("\xc3\xa9\xc3\xa9\xc3\xa9\xc3\xa9\xc3\xa9\xc3\xa9\xc3\xa9"
                               "\xc3\xa9\xc3\xa9\xc3\xa9\xc3\xa9\xc3\xa9x", 24, out,
                               sizeof(out)) != 0, "limit counts bytes, not characters");
    CHECK(pn_people_name_clean("da\x01vid", 24, out, sizeof(out)) != 0, "control char refused");
    CHECK(pn_people_name_clean("da\tvid", 24, out, sizeof(out)) != 0, "inner tab refused");
    CHECK(pn_people_name_clean("da\x7fvid", 24, out, sizeof(out)) != 0, "DEL refused");
    CHECK(pn_people_name_clean("bad\xff", 24, out, sizeof(out)) != 0, "broken UTF-8 refused");
    CHECK(pn_people_name_clean("   ", 24, out, sizeof(out)) != 0, "blank refused");
    CHECK(pn_people_name_clean(NULL, 24, out, sizeof(out)) != 0, "NULL refused");
}

static void test_add_and_round_trip(void)
{
    pn_person a, b, got;
    struct stat sb;

    printf("add_and_round_trip:\n");
    lt_clean_dir(STATE);
    lt_make_dir(STATE);
    pn_people *pp = pn_people_open(STATE "/people.json");
    CHECK(pp != NULL, "open empty store");
    CHECK(pn_people_add(pp, "bob", "Bob", 0, NULL, 1, &b) == PN_PEOPLE_E_INVALID,
          "first person must be an admin");
    CHECK(pn_people_add(pp, " david ", "David", 1, NULL, 1, &a) == PN_PEOPLE_OK, "add root");
    CHECK(strcmp(a.handle, "david") == 0 && a.id[0] == 'p' && strlen(a.id) == 9, "id + handle");
    CHECK(a.admin && a.admin_by[0] == '\0', "root has no maker");
    CHECK(pn_people_add(pp, "DAVID", "Other", 0, NULL, 1, &b) == PN_PEOPLE_E_HANDLE_TAKEN,
          "handle unique ignoring case");
    CHECK(pn_people_add(pp, "abcdefghijklmnopqrstuvwxy", "X", 0, NULL, 1, &b) ==
          PN_PEOPLE_E_BAD_HANDLE, "too long handle refused");
    CHECK(pn_people_add(pp, "x", "bad\x02name", 0, NULL, 1, &b) == PN_PEOPLE_E_BAD_NAME,
          "bad name refused");
    CHECK(pn_people_add(pp, "eve", "Eve", 1, NULL, 1, &b) == PN_PEOPLE_E_INVALID,
          "a second root is refused");
    CHECK(pn_people_add(pp, "bob", "Bob", 0, NULL, 0, &b) == PN_PEOPLE_OK, "add bob");
    pn_person_change ch = { .name = NULL, .handle = "BOB", .may_load = -1, .admin = -1 };
    CHECK(pn_people_update(pp, a.id, b.id, &ch, &got) == PN_PEOPLE_OK &&
          strcmp(got.handle, "BOB") == 0, "own handle may change case");
    ch.handle = "David";
    CHECK(pn_people_update(pp, a.id, b.id, &ch, &got) == PN_PEOPLE_E_HANDLE_TAKEN,
          "rename onto another handle refused");
    CHECK(stat(STATE "/people.json", &sb) == 0 && (sb.st_mode & 0777) == 0600, "0600");
    pn_people_close(pp);

    pp = pn_people_open(STATE "/people.json");
    CHECK(pp != NULL, "reopen");
    CHECK(pn_people_get(pp, b.id, &got) == PN_PEOPLE_OK && strcmp(got.name, "Bob") == 0 &&
          !got.admin && !got.may_load && strcmp(got.handle, "BOB") == 0, "bob round trip");
    CHECK(pn_people_find_handle(pp, "dAvId", &got) == PN_PEOPLE_OK && strcmp(got.id, a.id) == 0 &&
          got.created == a.created && got.admin, "find by handle ignoring case");
    pn_person rows[PN_PEOPLE_MAX];
    CHECK(pn_people_list(pp, rows, PN_PEOPLE_MAX) == 2, "two people");
    pn_people_close(pp);
}

static void test_malformed(void)
{
    static const char *bad[] = {
        "not json",
        "{\"version\":2,\"people\":[]}",
        "{\"version\":1,\"people\":[{\"id\":\"p00000001\",\"handle\":\"a\",\"name\":\"A\","
        "\"admin\":false,\"admin_by\":null,\"may_load\":true,\"created\":1}]}",
        "{\"version\":1,\"people\":[{\"id\":\"p00000001\",\"handle\":\"a\",\"name\":\"A\","
        "\"admin\":true,\"admin_by\":null,\"created\":1},{\"id\":\"p00000002\",\"handle\":\"A\","
        "\"name\":\"B\",\"admin\":false,\"admin_by\":null,\"created\":2}]}",
        "{\"version\":1,\"people\":[{\"id\":\"p00000001\",\"handle\":\"a\",\"name\":\"A\","
        "\"admin\":true,\"admin_by\":\"p0000000f\",\"created\":1}]}",
        "{\"version\":1,\"people\":[{\"id\":\"p00000001\",\"handle\":\"abcdefghijklmnopqrstuvwxy\","
        "\"name\":\"A\",\"admin\":true,\"admin_by\":null,\"created\":1}]}",
        "{\"version\":1,\"people\":[{\"id\":\"p00000001\",\"handle\":\"a\",\"name\":\"A\","
        "\"admin\":true,\"admin_by\":\"p00000002\",\"created\":1},{\"id\":\"p00000002\","
        "\"handle\":\"b\",\"name\":\"B\",\"admin\":true,\"admin_by\":\"p00000001\",\"created\":2}]}",
    };
    const char *why[] = { "not JSON", "unknown version", "people but no admin",
                          "duplicate handle", "maker missing", "handle too long",
                          "admin cycle" };

    printf("malformed:\n");
    for (size_t i = 0; i < sizeof(bad) / sizeof(bad[0]); i++) {
        write_text(STATE "/bad.json", bad[i]);
        pn_people *pp = pn_people_open(STATE "/bad.json");
        CHECK(pp == NULL, why[i]);
        pn_people_close(pp);
    }
}

static void make_admin(pn_people *pp, const char *actor, const char *id)
{
    pn_person_change ch = { NULL, NULL, -1, 1 };
    pn_people_update(pp, actor, id, &ch, NULL);
}

static void test_admin_tree(void)
{
    pn_person a, b, c, d, n, got;
    pn_person_change demote = { NULL, NULL, -1, 0 };

    printf("admin_tree:\n");
    lt_clean_dir(STATE);
    lt_make_dir(STATE);
    pn_people *pp = pn_people_open(STATE "/people.json");
    pn_people_add(pp, "a", "A", 1, NULL, 1, &a);
    CHECK(pn_people_remove(pp, a.id, a.id) == PN_PEOPLE_E_LAST_ADMIN, "last admin not removed");
    CHECK(pn_people_update(pp, a.id, a.id, &demote, NULL) == PN_PEOPLE_E_LAST_ADMIN,
          "last admin not demoted");
    pn_people_add(pp, "b", "B", 0, NULL, 1, &b);
    pn_people_add(pp, "c", "C", 0, NULL, 1, &c);
    pn_people_add(pp, "d", "D", 0, NULL, 1, &d);
    pn_people_add(pp, "n", "N", 0, NULL, 1, &n);
    CHECK(pn_people_remove(pp, n.id, b.id) == PN_PEOPLE_E_NOT_ADMIN, "non-admin cannot remove");
    make_admin(pp, n.id, b.id);
    CHECK(pn_people_get(pp, b.id, &got) == 0 && !got.admin, "non-admin cannot promote");
    make_admin(pp, a.id, b.id);
    make_admin(pp, b.id, c.id);
    make_admin(pp, a.id, d.id);
    CHECK(pn_people_get(pp, c.id, &got) == 0 && got.admin && strcmp(got.admin_by, b.id) == 0,
          "admin_by records the maker");
    CHECK(pn_people_remove(pp, b.id, a.id) == PN_PEOPLE_E_ADMIN_ABOVE,
          "a child admin cannot remove its maker");
    CHECK(pn_people_update(pp, b.id, a.id, &demote, NULL) == PN_PEOPLE_E_ADMIN_ABOVE,
          "a child admin cannot demote its maker");
    CHECK(pn_people_remove(pp, c.id, a.id) == PN_PEOPLE_E_ADMIN_ABOVE,
          "a grandchild cannot remove the root");
    CHECK(pn_people_update(pp, d.id, b.id, &demote, NULL) == PN_PEOPLE_E_ADMIN_ABOVE,
          "a sibling cannot demote");
    CHECK(pn_people_remove(pp, b.id, n.id) == PN_PEOPLE_OK, "any admin removes a non-admin");

    /* a demotes b: c (b's child) moves up to a. */
    CHECK(pn_people_update(pp, a.id, b.id, &demote, &got) == PN_PEOPLE_OK && !got.admin &&
          got.admin_by[0] == '\0', "maker demotes its child");
    CHECK(pn_people_get(pp, c.id, &got) == 0 && strcmp(got.admin_by, a.id) == 0,
          "children move up to the leaving admin's maker");
    CHECK(pn_people_remove(pp, a.id, c.id) == PN_PEOPLE_OK, "maker removes a descendant");
    CHECK(pn_people_update(pp, d.id, d.id, &demote, NULL) == PN_PEOPLE_OK,
          "an admin may demote itself");
    pn_people_close(pp);
}

/* The root leaves: its EARLIEST-made child (lowest created, here listed last)
 * becomes the root and adopts its siblings; grandchildren stay put. */
static void test_root_leaves(void)
{
    pn_person got;

    printf("root_leaves:\n");
    write_text(STATE "/people.json",
        "{\"version\":1,\"people\":["
        "{\"id\":\"p00000001\",\"handle\":\"root\",\"name\":\"R\",\"admin\":true,"
        "\"admin_by\":null,\"may_load\":true,\"created\":100},"
        "{\"id\":\"p00000002\",\"handle\":\"late\",\"name\":\"L\",\"admin\":true,"
        "\"admin_by\":\"p00000001\",\"may_load\":true,\"created\":300},"
        "{\"id\":\"p00000003\",\"handle\":\"early\",\"name\":\"E\",\"admin\":true,"
        "\"admin_by\":\"p00000001\",\"may_load\":true,\"created\":200},"
        "{\"id\":\"p00000004\",\"handle\":\"grand\",\"name\":\"G\",\"admin\":true,"
        "\"admin_by\":\"p00000002\",\"may_load\":true,\"created\":400}]}\n");
    pn_people *pp = pn_people_open(STATE "/people.json");
    CHECK(pp != NULL, "open tree fixture");
    CHECK(pn_people_remove(pp, "p00000001", "p00000001") == PN_PEOPLE_OK, "root removes itself");
    CHECK(pn_people_get(pp, "p00000003", &got) == 0 && got.admin_by[0] == '\0',
          "earliest-made child is the new root");
    CHECK(pn_people_get(pp, "p00000002", &got) == 0 && strcmp(got.admin_by, "p00000003") == 0,
          "sibling moves under the new root");
    CHECK(pn_people_get(pp, "p00000004", &got) == 0 && strcmp(got.admin_by, "p00000002") == 0,
          "grandchild keeps its maker");
    CHECK(pn_people_remove(pp, "p00000002", "p00000003") == PN_PEOPLE_E_ADMIN_ABOVE,
          "the new root is above its adopted child");
    pn_people_close(pp);
    pp = pn_people_open(STATE "/people.json");
    CHECK(pp && pn_people_get(pp, "p00000002", &got) == 0 &&
          strcmp(got.admin_by, "p00000003") == 0, "re-parenting saved");
    pn_people_close(pp);
}

static void test_limits_and_save_failure(void)
{
    pn_person a, p, rows[PN_PEOPLE_MAX];
    char handle[16];
    int added = 0;

    printf("limits_and_save_failure:\n");
    lt_clean_dir(STATE);
    lt_make_dir(STATE);
    pn_people *pp = pn_people_open(STATE "/people.json");
    pn_people_add(pp, "a", "A", 1, NULL, 1, &a);
    for (int i = 1; i < PN_PEOPLE_MAX; i++) {
        snprintf(handle, sizeof(handle), "u%d", i);
        added += pn_people_add(pp, handle, "U", 0, NULL, 1, &p) == PN_PEOPLE_OK;
    }
    CHECK(added == PN_PEOPLE_MAX - 1, "64 people fit");
    CHECK(pn_people_add(pp, "extra", "X", 0, NULL, 1, &p) == PN_PEOPLE_E_FULL, "65th refused");
    chmod(STATE, 0500);
    CHECK(pn_people_remove(pp, a.id, p.id) == PN_PEOPLE_E_IO, "save failure reported");
    CHECK(pn_people_list(pp, rows, PN_PEOPLE_MAX) == PN_PEOPLE_MAX, "memory unchanged");
    chmod(STATE, 0700);
    pn_people_close(pp);
}

int main(void)
{
    test_name_rules();
    test_add_and_round_trip();
    test_malformed();
    test_admin_tree();
    test_root_leaves();
    test_limits_and_save_failure();
    return lt_report("test_local_people_store");
}
