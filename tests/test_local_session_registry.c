/* SPDX-License-Identifier: AGPL-3.0-or-later */
/* Copyright (C) 2026 David (Corey) Herrera */
/*
 * test_local_session_registry.c — people registry P2.2: sessions.json and the
 * registry rules (spec §3.4, §4.4), no sockets:
 *   - field rules; upsert with the owner check (409 key_taken shape);
 *   - add / delete / share save at once, state changes only after 5 s dirty;
 *   - shares: owner only, never yourself, never a desktop row, 16 max, unshare;
 *   - the per-row queue: since filters, a beat acks, the 17th event -> resync;
 *   - rev moves on visible changes, not on beats;
 *   - 90 s with no wait -> disconnected + reregister; tombstones -> close removed;
 *   - the saved file holds no host_secret / host_pub; restart -> disconnected,
 *     shares kept, numbering above the host's since; malformed files refused;
 *   - caps: 128 rows per person, 512 per box; removing a person.
 * State under build/test_local/ (gitignored).
 */

#include "local_test_support.h"

#include "../src/local/session_registry.h"

#define STATE "build/test_local/session_registry"
#define REG_PATH STATE "/sessions.json"
#define SECRET "5ec7e75ec7e75ec7e75ec7e75ec7e75ec7e75ec7e75ec7e75ec7e75ec7e70000"
#define PUB    "9b0b9b0b9b0b9b0b9b0b9b0b9b0b9b0b9b0b9b0b9b0b9b0b9b0b9b0b9b0b1111"
#define DAVID  "p00000001"
#define BOB    "p00000002"
#define DEV    "d0123456789abcdef"
#define T0     1793000000LL

static pn_reg_host host_of(const char *key, const char *owner)
{
    pn_reg_host h = { .key = key, .kind = PN_REG_KIND_SESSION, .owner = owner,
                      .origin_device = DEV, .host_addr = "192.0.2.20:53211",
                      .title = "Fix the relay", .model = "qwen3.6-35b-a3b-q8", .instance = 2,
                      .state = PN_REG_STATE_THINKING, .host_secret = SECRET, .host_pub = PUB };
    return h;
}

static char *read_text(const char *path)
{
    char *text = NULL;
    FILE *f = fopen(path, "r");
    long n = 0;

    if (!f)
        return strdup("");
    fseek(f, 0, SEEK_END);
    n = ftell(f);
    fseek(f, 0, SEEK_SET);
    text = calloc(1, (size_t)n + 1);
    if (fread(text, 1, (size_t)n, f) != (size_t)n)
        text[0] = '\0';
    fclose(f);
    return text;
}

static int file_has(const char *needle)
{
    char *t = read_text(REG_PATH);
    int has = strstr(t, needle) != NULL;
    free(t);
    return has;
}

static pn_reg *fresh(void)
{
    lt_clean_dir(STATE);
    lt_make_dir(STATE);
    return pn_reg_open(REG_PATH);
}

static void test_fields(void)
{
    pn_reg *reg = fresh();
    pn_reg_host h = host_of("c-0123456789ab", DAVID);
    char long_title[300];

    printf("fields:\n");
    CHECK(reg != NULL, "a missing file opens empty");
    h.key = "c-0123456789AB";
    CHECK(pn_reg_upsert(reg, &h, T0, NULL) == PN_REG_E_INVALID, "upper-case key refused");
    h.key = "c-0123456789a";
    CHECK(pn_reg_upsert(reg, &h, T0, NULL) == PN_REG_E_INVALID, "short key refused");
    h = host_of("c-0123456789ab", DAVID);
    h.host_addr = "192.0.2.300:1";
    CHECK(pn_reg_upsert(reg, &h, T0, NULL) == PN_REG_E_INVALID, "bad IPv4 refused");
    h.host_addr = "192.0.2.20:0";
    CHECK(pn_reg_upsert(reg, &h, T0, NULL) == PN_REG_E_INVALID, "port 0 refused");
    h.host_addr = "[fe80::1]:53211";
    CHECK(pn_reg_addr_valid(h.host_addr), "[v6]:port accepted");
    h = host_of("c-0123456789ab", DAVID);
    memset(long_title, 'x', 257);
    long_title[257] = '\0';
    h.title = long_title;
    CHECK(pn_reg_upsert(reg, &h, T0, NULL) == PN_REG_E_INVALID, "257-byte title refused");
    long_title[256] = '\0';
    CHECK(pn_reg_upsert(reg, &h, T0, NULL) == PN_REG_OK, "256-byte title accepted");
    h = host_of("c-0123456789ab", DAVID);
    h.instance = 65;
    CHECK(pn_reg_upsert(reg, &h, T0, NULL) == PN_REG_E_INVALID, "instance 65 refused");
    h = host_of("c-0123456789ab", DAVID);
    h.state = PN_REG_STATE_DISCONNECTED;
    CHECK(pn_reg_upsert(reg, &h, T0, NULL) == PN_REG_E_INVALID, "state disconnected refused");
    h = host_of("c-0123456789ab", DAVID);
    h.host_secret = "abc";
    CHECK(pn_reg_upsert(reg, &h, T0, NULL) == PN_REG_E_INVALID, "short secret refused");
    pn_reg_close(reg);
}

static void test_upsert_and_saves(void)
{
    pn_reg *reg = fresh();
    pn_reg_host h = host_of("c-0123456789ab", DAVID);
    pn_reg_row_info row;
    long long seq = -1;
    unsigned long long rev = 0;

    printf("upsert_and_saves:\n");
    CHECK(pn_reg_upsert(reg, &h, T0, &seq) == PN_REG_OK && seq == 0, "new row, seq 0");
    CHECK(file_has("c-0123456789ab") && file_has("\"thinking\""), "a new row saves at once");
    rev = pn_reg_rev(reg);
    h.owner = BOB;
    CHECK(pn_reg_upsert(reg, &h, T0, NULL) == PN_REG_E_KEY_TAKEN, "another person: key taken");
    h = host_of("c-0123456789ab", DAVID);
    h.state = PN_REG_STATE_IDLE;
    h.title = "Renamed";
    CHECK(pn_reg_upsert(reg, &h, T0 + 10, &seq) == PN_REG_OK, "the owner updates");
    CHECK(pn_reg_get(reg, "c-0123456789ab", &row) == PN_REG_OK && row.state == PN_REG_STATE_IDLE &&
          strcmp(row.title, "Renamed") == 0 && row.registered_at == T0, "update applied");
    CHECK(pn_reg_rev(reg) == rev + 1, "a visible change moves rev");
    CHECK(file_has("\"thinking\"") && !file_has("Renamed"), "a state change does not save at once");
    CHECK(pn_reg_save_due(reg) == T0 + 10 + PN_REG_SAVE_DELAY_S, "save due 5 s after dirty");
    CHECK(pn_reg_save_if_dirty(reg, T0 + 14) == 0 && !file_has("Renamed"), "not before 5 s");
    CHECK(pn_reg_save_if_dirty(reg, T0 + 15) == 1 && file_has("Renamed") && file_has("\"idle\""),
          "saved at 5 s");
    CHECK(pn_reg_save_due(reg) == 0, "clean after the save");
    rev = pn_reg_rev(reg);
    CHECK(pn_reg_beat(reg, "c-0123456789ab", DAVID, 0, T0 + 20) == PN_REG_OK &&
          pn_reg_rev(reg) == rev, "a beat does not move rev");
    CHECK(pn_reg_beat(reg, "c-0123456789ab", BOB, 0, T0 + 20) == PN_REG_E_NOT_OWNER,
          "a beat from another person is refused");
    CHECK(pn_reg_beat(reg, "c-ffffffffffff", DAVID, 0, T0 + 20) == PN_REG_E_NOT_FOUND,
          "a beat for an unknown key: not found");
    CHECK(!file_has(SECRET) && !file_has(PUB) && !file_has("host_secret") && !file_has("host_pub"),
          "the file holds no host_secret / host_pub");
    struct stat st;
    CHECK(stat(REG_PATH, &st) == 0 && (st.st_mode & 0777) == 0600, "sessions.json is 0600");
    pn_reg_close(reg);
}

static void test_shares(void)
{
    pn_reg *reg = fresh();
    pn_reg_host h = host_of("c-0123456789ab", DAVID);
    pn_reg_row_info rows[4];
    pn_reg_event ev[PN_REG_QUEUE_MAX];
    pn_reg_share_entry shares[PN_REG_MAX_SHARES];
    long long seq = 0;
    int n_shares = 0;
    char person[16];

    printf("shares:\n");
    pn_reg_upsert(reg, &h, T0, NULL);
    CHECK(pn_reg_share(reg, "c-0123456789ab", BOB, DAVID, PN_REG_ROLE_VIEW) == PN_REG_E_NOT_OWNER,
          "only the owner shares");
    CHECK(pn_reg_share(reg, "c-0123456789ab", DAVID, DAVID, PN_REG_ROLE_VIEW) ==
          PN_REG_E_SELF_SHARE, "never with yourself");
    CHECK(pn_reg_share(reg, "c-ffffffffffff", DAVID, BOB, PN_REG_ROLE_VIEW) == PN_REG_E_NOT_FOUND,
          "unknown row");
    CHECK(pn_reg_share(reg, "c-0123456789ab", DAVID, BOB, PN_REG_ROLE_VIEW) == PN_REG_OK,
          "share view with bob");
    CHECK(file_has(BOB), "a share saves at once");
    CHECK(pn_reg_role_of(reg, "c-0123456789ab", BOB) == PN_REG_ROLE_VIEW &&
          pn_reg_role_of(reg, "c-0123456789ab", DAVID) == PN_REG_ROLE_OWNER &&
          pn_reg_role_of(reg, "c-0123456789ab", "p00000003") == PN_REG_ROLE_NONE, "roles");
    CHECK(pn_reg_list_for(reg, BOB, rows, 4) == 1 && pn_reg_list_for(reg, "p00000003", rows, 4) == 0,
          "bob lists the shared row, a stranger nothing");
    {
        char secret[PN_REG_HEX32_MAX] = "", pub[PN_REG_HEX32_MAX] = "";
        pn_reg_role role = PN_REG_ROLE_NONE;
        CHECK(pn_reg_attach_info(reg, "c-0123456789ab", BOB, rows, &role, secret, pub) ==
              PN_REG_OK && role == PN_REG_ROLE_VIEW && strcmp(secret, SECRET) == 0 &&
              strcmp(pub, PUB) == 0 && strcmp(rows[0].key, "c-0123456789ab") == 0,
              "attach info: bob's role, the host's secret and public key");
        CHECK(pn_reg_attach_info(reg, "c-0123456789ab", "p00000003", rows, &role, secret,
                                 pub) == PN_REG_OK && role == PN_REG_ROLE_NONE,
              "attach info: a stranger has no role");
        CHECK(pn_reg_attach_info(reg, "c-ffffffffffff", BOB, rows, &role, secret, pub) ==
              PN_REG_E_NOT_FOUND, "attach info: unknown row");
    }
    CHECK(pn_reg_events_since(reg, "c-0123456789ab", 0, ev, PN_REG_QUEUE_MAX, &seq, shares,
                              &n_shares) == 1 && ev[0].type == PN_REG_EV_SHARES && seq == 1 &&
          n_shares == 1 && shares[0].role == PN_REG_ROLE_VIEW, "a shares event is queued");
    CHECK(pn_reg_share(reg, "c-0123456789ab", DAVID, BOB, PN_REG_ROLE_CONTROL) == PN_REG_OK &&
          pn_reg_role_of(reg, "c-0123456789ab", BOB) == PN_REG_ROLE_CONTROL, "upgrade to control");
    for (int i = 3; i <= 18; i++) {
        snprintf(person, sizeof(person), "p%08d", i);
        CHECK(pn_reg_share(reg, "c-0123456789ab", DAVID, person, PN_REG_ROLE_VIEW) ==
              (i <= 17 ? PN_REG_OK : PN_REG_E_SHARES_FULL), "16 shares per row");
    }
    CHECK(pn_reg_share(reg, "c-0123456789ab", DAVID, BOB, PN_REG_ROLE_NONE) == PN_REG_OK &&
          pn_reg_role_of(reg, "c-0123456789ab", BOB) == PN_REG_ROLE_NONE, "unshare");
    CHECK(!file_has(BOB), "the unshare is on disk at once");
    h = host_of("c-aaaaaaaaaaaa", DAVID);
    h.kind = PN_REG_KIND_DESKTOP;
    pn_reg_upsert(reg, &h, T0, NULL);
    CHECK(pn_reg_share(reg, "c-aaaaaaaaaaaa", DAVID, BOB, PN_REG_ROLE_VIEW) ==
          PN_REG_E_NOT_SHAREABLE, "a desktop row is never shared");
    pn_reg_close(reg);
}

static void test_queue(void)
{
    pn_reg *reg = fresh();
    pn_reg_host h = host_of("c-0123456789ab", DAVID);
    pn_reg_event ev[PN_REG_QUEUE_MAX], relay = { .type = PN_REG_EV_RELAY, .relay_expires = T0 + 10 };
    long long seq = 0;

    printf("queue:\n");
    snprintf(relay.relay_id, sizeof(relay.relay_id), "r-0011223344556677");
    pn_reg_upsert(reg, &h, T0, NULL);
    CHECK(pn_reg_pending(reg, "c-0123456789ab", 0) == 0, "nothing pending");
    CHECK(pn_reg_push_event(reg, "c-0123456789ab", &relay) == 1, "first event is seq 1");
    CHECK(pn_reg_pending(reg, "c-0123456789ab", 0) == 1 &&
          pn_reg_pending(reg, "c-0123456789ab", 1) == 0, "pending above since only");
    for (int i = 2; i <= PN_REG_QUEUE_MAX; i++)
        pn_reg_push_event(reg, "c-0123456789ab", &relay);
    CHECK(pn_reg_events_since(reg, "c-0123456789ab", 10, ev, PN_REG_QUEUE_MAX, &seq, NULL, NULL) ==
          6 && ev[0].seq == 11 && seq == 16, "since filters");
    CHECK(pn_reg_beat(reg, "c-0123456789ab", DAVID, 10, T0 + 1) == PN_REG_OK &&
          pn_reg_events_since(reg, "c-0123456789ab", 0, ev, PN_REG_QUEUE_MAX, &seq, NULL, NULL) ==
          6, "a wait with since 10 acks 1-10");
    for (int i = 0; i < 10; i++)
        pn_reg_push_event(reg, "c-0123456789ab", &relay);
    CHECK(pn_reg_push_event(reg, "c-0123456789ab", &relay) == 27, "17th queued event -> resync");
    CHECK(pn_reg_events_since(reg, "c-0123456789ab", 0, ev, PN_REG_QUEUE_MAX, &seq, NULL, NULL) ==
          1 && ev[0].type == PN_REG_EV_RESYNC && ev[0].seq == 27, "the queue is one resync");
    CHECK(pn_reg_upsert(reg, &h, T0 + 2, &seq) == PN_REG_OK && seq == 27 &&
          pn_reg_pending(reg, "c-0123456789ab", 0) == 0, "re-POST answers the resync");
    CHECK(pn_reg_push_event(reg, "c-ffffffffffff", &relay) == PN_REG_E_NOT_FOUND, "unknown row");
    pn_reg_close(reg);
}

static void test_sweep_and_delete(void)
{
    pn_reg *reg = fresh();
    pn_reg_host h = host_of("c-0123456789ab", DAVID);
    pn_reg_row_info row;
    pn_reg_event ev[4];
    long long seq = 0;
    unsigned long long rev = 0;

    printf("sweep_and_delete:\n");
    pn_reg_upsert(reg, &h, T0, NULL);
    pn_reg_beat(reg, "c-0123456789ab", DAVID, 0, T0 + 100);
    rev = pn_reg_rev(reg);
    CHECK(pn_reg_sweep(reg, T0 + 190) == 0, "90 s after the last wait: still live");
    CHECK(pn_reg_sweep(reg, T0 + 191) == 1 && pn_reg_get(reg, "c-0123456789ab", &row) == 0 &&
          row.state == PN_REG_STATE_DISCONNECTED, "91 s: disconnected");
    CHECK(pn_reg_rev(reg) == rev + 1, "disconnect moves rev");
    CHECK(pn_reg_events_since(reg, "c-0123456789ab", 0, ev, 4, &seq, NULL, NULL) == 1 &&
          ev[0].type == PN_REG_EV_REREGISTER, "reregister queued");
    CHECK(pn_reg_upsert(reg, &h, T0 + 200, &seq) == PN_REG_OK &&
          pn_reg_get(reg, "c-0123456789ab", &row) == 0 && row.state == PN_REG_STATE_THINKING &&
          pn_reg_pending(reg, "c-0123456789ab", 0) == 0 && seq == 1, "re-POST: live again");
    CHECK(pn_reg_delete(reg, "c-0123456789ab", BOB, T0 + 210) == PN_REG_E_NOT_OWNER,
          "only the owner deletes");
    CHECK(pn_reg_delete(reg, "c-0123456789ab", DAVID, T0 + 210) == PN_REG_OK &&
          !file_has("c-0123456789ab"), "delete saves at once");
    CHECK(pn_reg_delete(reg, "c-0123456789ab", DAVID, T0 + 210) == PN_REG_E_NOT_FOUND, "gone");
    CHECK(pn_reg_beat(reg, "c-0123456789ab", DAVID, 1, T0 + 211) == PN_REG_GONE &&
          pn_reg_pending(reg, "c-0123456789ab", 99) == 1, "a wait on the deleted key: pending");
    CHECK(pn_reg_events_since(reg, "c-0123456789ab", 1, ev, 4, &seq, NULL, NULL) == 1 &&
          ev[0].type == PN_REG_EV_CLOSE && ev[0].reason == PN_REG_CLOSE_REMOVED && ev[0].seq == 2,
          "its answer is close removed");
    CHECK(pn_reg_beat(reg, "c-0123456789ab", BOB, 1, T0 + 211) == PN_REG_E_NOT_OWNER,
          "a tombstone answers its owner only");
    pn_reg_sweep(reg, T0 + 210 + PN_REG_TOMBSTONE_S);
    CHECK(pn_reg_beat(reg, "c-0123456789ab", DAVID, 1, T0 + 400) == PN_REG_E_NOT_FOUND,
          "the tombstone expires");
    pn_reg_close(reg);
}

static void test_restart(void)
{
    pn_reg *reg = fresh();
    pn_reg_host h = host_of("c-0123456789ab", DAVID);
    pn_reg_row_info row;
    pn_reg_event ev[4];
    long long seq = 0;

    printf("restart:\n");
    pn_reg_upsert(reg, &h, T0, NULL);
    pn_reg_share(reg, "c-0123456789ab", DAVID, BOB, PN_REG_ROLE_VIEW);
    CHECK(pn_reg_save(reg) == 0, "saved");
    pn_reg_close(reg);
    reg = pn_reg_open(REG_PATH);
    CHECK(reg && pn_reg_get(reg, "c-0123456789ab", &row) == 0, "the row comes back");
    if (!reg)
        return;
    CHECK(row.state == PN_REG_STATE_DISCONNECTED, "reads disconnected after a load");
    {
        char secret[PN_REG_HEX32_MAX] = "x", pub[PN_REG_HEX32_MAX] = "x";
        pn_reg_role role = PN_REG_ROLE_NONE;
        CHECK(pn_reg_attach_info(reg, "c-0123456789ab", DAVID, &row, &role, secret, pub) ==
              PN_REG_OK && role == PN_REG_ROLE_OWNER && !secret[0] && !pub[0],
              "after a load no secret or public key until the host registers");
    }
    CHECK(row.n_shares == 1 && strcmp(row.shares[0].person, BOB) == 0 &&
          row.shares[0].role == PN_REG_ROLE_VIEW && strcmp(row.title, "Fix the relay") == 0 &&
          row.registered_at == T0 && row.instance == 2, "fields and shares kept");
    CHECK(pn_reg_beat(reg, "c-0123456789ab", DAVID, 14, T0 + 50) == PN_REG_OK &&
          pn_reg_events_since(reg, "c-0123456789ab", 14, ev, 4, &seq, NULL, NULL) == 1 &&
          ev[0].type == PN_REG_EV_REREGISTER && ev[0].seq == 15,
          "the host's old since 14 gets reregister 15");
    CHECK(pn_reg_upsert(reg, &h, T0 + 51, &seq) == PN_REG_OK && seq == 15 &&
          pn_reg_get(reg, "c-0123456789ab", &row) == 0 && row.state == PN_REG_STATE_THINKING &&
          row.n_shares == 1, "re-POST: live, shares kept, waits from 15");
    pn_reg_close(reg);
}

static void write_text(const char *text)
{
    FILE *f = fopen(REG_PATH, "w");
    if (f) {
        fputs(text, f);
        fclose(f);
    }
}

static void test_malformed(void)
{
    static const char *bad[] = {
        "not json",
        "{\"version\":2,\"rows\":[]}",
        "{\"version\":1,\"rows\":[{\"key\":\"c-XYZ\",\"kind\":\"session\"}]}",
        "{\"version\":1,\"rows\":[{\"key\":\"c-0123456789ab\",\"kind\":\"session\","
        "\"owner\":\"p00000001\",\"origin_device\":\"d0123456789abcdef\",\"host_addr\":"
        "\"1.2.3.4:5\",\"title\":\"\",\"model\":\"\",\"instance\":99,\"state\":\"idle\","
        "\"shares\":[],\"registered_at\":1,\"last_beat\":1}]}",
    };
    pn_reg *reg = NULL;

    printf("malformed:\n");
    lt_clean_dir(STATE);
    lt_make_dir(STATE);
    for (size_t i = 0; i < sizeof(bad) / sizeof(bad[0]); i++) {
        write_text(bad[i]);
        reg = pn_reg_open(REG_PATH);
        CHECK(reg == NULL, "a malformed sessions.json is refused, not emptied");
        pn_reg_close(reg);
    }
    write_text("{\"version\":1,\"rows\":[]}");
    reg = pn_reg_open(REG_PATH);
    CHECK(reg != NULL, "an empty registry opens");
    pn_reg_close(reg);
}

static void test_caps_and_remove_person(void)
{
    pn_reg *reg = fresh();
    pn_reg_host h;
    char key[16], owner[16], keys[8][PN_REG_KEY_MAX];
    int failures = 0, removed = 0, n = 0;

    printf("caps_and_remove_person:\n");
    for (int i = 0; i < PN_REG_MAX_ROWS; i++) {
        snprintf(key, sizeof(key), "c-%012x", i);
        snprintf(owner, sizeof(owner), "p%08d", i / PN_REG_MAX_PER_PERSON + 1);
        h = host_of(key, owner);
        failures += pn_reg_upsert(reg, &h, T0, NULL) != PN_REG_OK;
    }
    CHECK(failures == 0, "512 rows over 4 people");
    h = host_of("c-ffffffffffff", "p00000009");
    CHECK(pn_reg_upsert(reg, &h, T0, NULL) == PN_REG_E_FULL, "513th row: registry full");
    pn_reg_close(reg);

    reg = fresh();
    for (int i = 0; i < PN_REG_MAX_PER_PERSON; i++) {
        snprintf(key, sizeof(key), "c-%012x", i);
        h = host_of(key, DAVID);
        pn_reg_upsert(reg, &h, T0, NULL);
    }
    h = host_of("c-ffffffffffff", DAVID);
    CHECK(pn_reg_upsert(reg, &h, T0, NULL) == PN_REG_E_FULL, "129th row of one person: full");
    h = host_of("c-ffffffffffff", BOB);
    CHECK(pn_reg_upsert(reg, &h, T0, NULL) == PN_REG_OK, "another person still registers");
    pn_reg_close(reg);

    reg = fresh();
    h = host_of("c-0000000000b1", BOB);
    pn_reg_upsert(reg, &h, T0, NULL);
    h = host_of("c-0000000000b2", BOB);
    pn_reg_upsert(reg, &h, T0, NULL);
    h = host_of("c-0000000000d1", DAVID);
    pn_reg_upsert(reg, &h, T0, NULL);
    pn_reg_share(reg, "c-0000000000d1", DAVID, BOB, PN_REG_ROLE_CONTROL);
    n = pn_reg_remove_person(reg, BOB, T0 + 5, keys, 8, &removed);
    CHECK(n == 3 && removed == 2, "bob's two rows deleted, one share dropped");
    CHECK(pn_reg_role_of(reg, "c-0000000000d1", BOB) == PN_REG_ROLE_NONE &&
          pn_reg_pending(reg, "c-0000000000d1", 1) == 1, "the shared row gets a shares event");
    CHECK(pn_reg_beat(reg, "c-0000000000b1", BOB, 0, T0 + 6) == PN_REG_GONE,
          "bob's hosts get close removed");
    CHECK(!file_has(BOB), "saved at once");
    pn_reg_close(reg);
}

int main(void)
{
    test_fields();
    test_upsert_and_saves();
    test_shares();
    test_queue();
    test_sweep_and_delete();
    test_restart();
    test_malformed();
    test_caps_and_remove_person();
    return lt_report("test_local_session_registry");
}
