/* SPDX-License-Identifier: AGPL-3.0-or-later */
/* Copyright (C) 2026 David (Corey) Herrera */
/*
 * local_fixture_replay.h — replay the people-registry contract fixtures
 * (tests/local/fixtures/people_registry/<endpoint>.json, spec §4 verbatim) against a
 * real `pantheon local`. A fixture is
 *   {"endpoint","class","request":{...}|null,"status":200,"response":{...},
 *    "errors":[{"when","status","type"}]}
 * (a fixture may hold a second answer shape, e.g. "response_short").
 * The request is sent with a few fields replaced by live values (the real
 * pairing code, ids made by this run); the answer must have the fixture's
 * status and exactly the fixture's shape: the same keys at every level, the
 * same JSON types, and array elements shaped like the fixture's first one.
 * Values are not compared (keys, ids and times differ per run); the tests
 * check the values that matter themselves. Needs local_test_support.h first.
 */

#ifndef PN_LOCAL_FIXTURE_REPLAY_H
#define PN_LOCAL_FIXTURE_REPLAY_H

#include "../src/serve/json_reader.h"
#include "../src/serve/json_writer.h"

#define FX_DIR "tests/local/fixtures/people_registry/"

/* Parse fixture `name` (e.g. "pair.json"); NULL when missing or broken. */
static inline jr_value *fx_load(const char *name)
{
    char path[512], *text = NULL;
    FILE *f = NULL;
    long n = 0;
    jr_value *v = NULL;

    snprintf(path, sizeof(path), FX_DIR "%s", name);
    f = fopen(path, "r");
    if (!f)
        return NULL;
    fseek(f, 0, SEEK_END);
    n = ftell(f);
    fseek(f, 0, SEEK_SET);
    text = calloc(1, (size_t)n + 1);
    if (text && fread(text, 1, (size_t)n, f) == (size_t)n)
        v = jr_parse(text);
    free(text);
    fclose(f);
    return v;
}

/* The fixture's request as JSON, with top-level members replaced: `subs` is
 * {name, raw JSON value, name, raw JSON value, ..., NULL}. Caller frees. */
static inline char *fx_request(const jr_value *fx, const char *const *subs)
{
    const jr_value *req = jr_object_get(fx, "request");
    json_writer jw;
    char *out = NULL;

    if (jr_kind(req) != JR_OBJECT || !jw_init(&jw))
        return NULL;
    jw_raw(&jw, "{");
    for (int i = 0; i < jr_object_member_count(req); i++) {
        const char *k = jr_object_key_at(req, i);
        const char *raw = NULL;
        for (int s = 0; subs && subs[s]; s += 2)
            if (strcmp(subs[s], k) == 0)
                raw = subs[s + 1];
        jw_raw(&jw, i ? "," : "");
        jw_string(&jw, k);
        jw_raw(&jw, ":");
        if (raw)
            jw_raw(&jw, raw);
        else
            jr_to_json(jr_object_value_at(req, i), &jw);
    }
    jw_raw(&jw, "}");
    out = strdup(jw.data);
    jw_free(&jw);
    return out;
}

/* 1 when `got` has the shape of `want`; else 0 and `where` names the spot. */
static inline int fx_same_shape(const jr_value *want, const jr_value *got, char *where,
                                size_t cap, const char *at)
{
    if (!got || jr_kind(want) != jr_kind(got)) {
        snprintf(where, cap, "%s: type differs", at);
        return 0;
    }
    if (jr_kind(want) == JR_OBJECT) {
        if (jr_object_member_count(want) != jr_object_member_count(got)) {
            snprintf(where, cap, "%s: %d keys, fixture %d", at, jr_object_member_count(got),
                     jr_object_member_count(want));
            return 0;
        }
        for (int i = 0; i < jr_object_member_count(want); i++) {
            char sub[256];
            const char *k = jr_object_key_at(want, i);
            snprintf(sub, sizeof(sub), "%s.%s", at, k);
            if (!fx_same_shape(jr_object_value_at(want, i), jr_object_get(got, k), where, cap,
                               sub))
                return 0;
        }
    }
    if (jr_kind(want) == JR_ARRAY && jr_array_len(want) > 0) {
        for (int i = 0; i < jr_array_len(got); i++) {
            char sub[256];
            snprintf(sub, sizeof(sub), "%s[%d]", at, i);
            if (!fx_same_shape(jr_array_get(want, 0), jr_array_get(got, i), where, cap, sub))
                return 0;
        }
    }
    return 1;
}

/* Send `body` to `method path` with `key`; 1 when the answer has the fixture's
 * status and the shape of its member `resp_key` (normally "response"). *resp_out
 * (may be NULL) gets the body. Prints why not. */
static inline int fx_replay_as(int port, const jr_value *fx, const char *resp_key,
                               const char *method, const char *path, const char *key,
                               const char *body, char **resp_out)
{
    char *resp = NULL, where[300] = "";
    int want = (int)jr_as_number(jr_object_get(fx, "status"), 0);
    int st = lt_http(port, method, path, key, body, &resp);
    jr_value *got = resp ? jr_parse(resp) : NULL;
    int ok = st == want && fx_same_shape(jr_object_get(fx, resp_key), got, where,
                                         sizeof(where), resp_key);

    if (!ok)
        fprintf(stderr, "  replay %s %s: status %d (fixture %d) %s\n    body %s\n", method, path,
                st, want, where, resp ? resp : "");
    jr_free(got);
    if (resp_out)
        *resp_out = resp;
    else
        free(resp);
    return ok;
}

static inline int fx_replay(int port, const jr_value *fx, const char *method, const char *path,
                            const char *key, const char *body, char **resp_out)
{
    return fx_replay_as(port, fx, "response", method, path, key, body, resp_out);
}

/* 1 when the fixture lists an error with this status and type. */
static inline int fx_has_error(const jr_value *fx, int status, const char *type)
{
    const jr_value *errs = jr_object_get(fx, "errors");

    for (int i = 0; i < jr_array_len(errs); i++) {
        const jr_value *e = jr_array_get(errs, i);
        const char *t = jr_as_string(jr_object_get(e, "type"));
        if ((int)jr_as_number(jr_object_get(e, "status"), 0) == status && t &&
            strcmp(t, type) == 0)
            return 1;
    }
    return 0;
}

/* The error answer of a request: status, with `type` copied out. */
static inline int fx_error(int port, const char *method, const char *path, const char *key,
                           const char *body, char *type, size_t cap)
{
    char *resp = NULL;
    int st = lt_http(port, method, path, key, body, &resp);
    jr_value *v = resp ? jr_parse(resp) : NULL;
    const char *t = jr_as_string(jr_object_get(jr_object_get(v, "error"), "type"));

    snprintf(type, cap, "%s", t ? t : "");
    jr_free(v);
    free(resp);
    return st;
}

/* A string member of a JSON body, copied into out ("" when missing). */
static inline void fx_field(const char *json, const char *a, const char *b, char *out,
                            size_t cap)
{
    jr_value *v = json ? jr_parse(json) : NULL;
    const jr_value *x = jr_object_get(v, a);
    const char *s = NULL;

    if (b)
        x = jr_object_get(x, b);
    s = jr_as_string(x);
    snprintf(out, cap, "%s", s ? s : "");
    jr_free(v);
}

#endif /* PN_LOCAL_FIXTURE_REPLAY_H */
