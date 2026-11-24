/* SPDX-License-Identifier: AGPL-3.0-or-later */
/* Copyright (C) 2026 David (Corey) Herrera */
/*
 * local_people_test_support.h — shared helpers of the people-registry HTTP
 * tests: read the console pairing code, fork a box on a state dir under
 * build/test_local/, count substrings, and seed a state dir with people.json +
 * keys.json v2 holding TEST secrets (`pnk_` + 64 x one letter). Needs
 * local_test_support.h first.
 */

#ifndef PN_LOCAL_PEOPLE_TEST_SUPPORT_H
#define PN_LOCAL_PEOPLE_TEST_SUPPORT_H

#include "../src/local/sha256.h"

static inline void read_code(const char *state_dir, char code[32])
{
    char path[512];
    FILE *f = NULL;

    snprintf(path, sizeof(path), "%s/pair_code", state_dir);
    code[0] = '\0';
    f = fopen(path, "r");
    if (!f)
        return;
    if (!fgets(code, 32, f))
        code[0] = '\0';
    fclose(f);
    code[strcspn(code, "\r\n")] = '\0';
}

static inline pid_t spawn_box(const char *dir, int port)
{
    char port_s[16], state[256], store[256], log[256];

    snprintf(port_s, sizeof(port_s), "%d", port);
    snprintf(state, sizeof(state), "%s/state", dir);
    snprintf(store, sizeof(store), "%s/store", dir);
    snprintf(log, sizeof(log), "%s/serve.log", dir);
    char *argv[] = { "local", "--host", "127.0.0.1", "--port", port_s, "--state-dir", state,
                     "--store", store, "--no-mdns", "--catalog-url", "", NULL };
    return lt_spawn_local(12, argv, port, log);
}

static inline int count_of(const char *json, const char *needle)
{
    int n = 0;

    for (const char *p = json ? strstr(json, needle) : NULL; p; p = strstr(p + 1, needle))
        n++;
    return n;
}

/* people.json + keys.json v2 with TEST secrets "pnk_" + 64 x c: admin david
 * (p00000001: device 'a', inference 'd') and, with with_bob, non-admin bob
 * (p00000002: device 'b', inference 'c'). */
static inline void seed_state(const char *state_dir, int with_bob)
{
    static const struct { char c; const char *id, *kind, *person, *dev; } keys[] = {
        { 'a', "k000000aa", "device", "p00000001", "d00000000000000aa" },
        { 'd', "k000000dd", "inference", "p00000001", NULL },
        { 'b', "k000000bb", "device", "p00000002", "d00000000000000bb" },
        { 'c', "k000000cc", "inference", "p00000002", NULL },
    };
    char path[512], secret[80], hex[65], text[4096];
    uint8_t h[PN_SHA256_LEN];
    size_t used = 0;
    FILE *f = NULL;

    snprintf(path, sizeof(path), "%s/people.json", state_dir);
    f = fopen(path, "w");
    if (f) {
        fputs("{\"version\":1,\"people\":["
              "{\"id\":\"p00000001\",\"handle\":\"david\",\"name\":\"David\",\"admin\":true,"
              "\"admin_by\":null,\"may_load\":true,\"created\":1793000000}", f);
        if (with_bob)
            fputs(",{\"id\":\"p00000002\",\"handle\":\"bob\",\"name\":\"Bob\",\"admin\":false,"
                  "\"admin_by\":null,\"may_load\":true,\"created\":1793000100}", f);
        fputs("]}\n", f);
        fclose(f);
    }
    used += (size_t)snprintf(text, sizeof(text), "{\"version\":2,\"keys\":[");
    for (size_t i = 0; i < sizeof(keys) / sizeof(keys[0]); i++) {
        if (!with_bob && strcmp(keys[i].person, "p00000002") == 0)
            continue;
        memset(secret, keys[i].c, sizeof(secret));
        memcpy(secret, "pnk_", 4);
        secret[68] = '\0';
        pn_sha256(secret, strlen(secret), h);
        pn_hex_encode(h, PN_SHA256_LEN, hex);
        used += (size_t)snprintf(text + used, sizeof(text) - used,
                                 "%s{\"id\":\"%s\",\"sha256\":\"%s\",\"kind\":\"%s\","
                                 "\"person\":\"%s\"%s%s%s,\"label\":\"seed\",\"created\":1}",
                                 i ? "," : "", keys[i].id, hex, keys[i].kind, keys[i].person,
                                 keys[i].dev ? ",\"device_id\":\"" : "",
                                 keys[i].dev ? keys[i].dev : "",
                                 keys[i].dev ? "\",\"device_name\":\"seed\"" : "");
    }
    snprintf(text + used, sizeof(text) - used, "]}\n");
    snprintf(path, sizeof(path), "%s/keys.json", state_dir);
    f = fopen(path, "w");
    if (f) {
        fputs(text, f);
        fclose(f);
    }
}

static inline void secret_of(char c, char out[80])
{
    memset(out, c, 80);
    memcpy(out, "pnk_", 4);
    out[68] = '\0';
}

#endif /* PN_LOCAL_PEOPLE_TEST_SUPPORT_H */
