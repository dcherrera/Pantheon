/* SPDX-License-Identifier: AGPL-3.0-or-later */
/* Copyright (C) 2026 David (Corey) Herrera */
/*
 * test_local_catalog.c — Pantheon Local E1/E4/E7: the accepted catalog.
 * Signed fixtures (TEST ONLY keys) are accepted, persisted and survive a
 * reopen; an older version is refused (rollback) and so is a tampered or an
 * unsigned one; a build that trusts no key refuses everything; refresh over
 * curl (file://) works; manifest validation refuses bad fields; matching ranks
 * fitting profiles first and gives the reason for the rest; files[].unit_types
 * picks the files a unit type needs and refuses unknown types (spec §16); a
 * model may have one weights file per unit type (Qwen3.6: the K80 GGUF, the
 * p150 metadata GGUF), exactly one for each profile.
 */

#include "local_test_support.h"

#include "../src/local/catalog.h"
#include "../src/local/catalog_match.h"
#include "../src/local/curl_runner.h"
#include "../src/local/load_options.h"

#define FX    "tests/local/fixtures/"
#define STATE "build/test_local/catalog"
#define GIB(n) ((uint64_t)(n) << 30)

static char *slurp(const char *path, size_t *len)
{
    FILE *f = fopen(path, "rb");
    if (!f) return NULL;
    fseek(f, 0, SEEK_END);
    long n = ftell(f);
    fseek(f, 0, SEEK_SET);
    char *b = malloc((size_t)n + 1);
    if (fread(b, 1, (size_t)n, f) != (size_t)n) n = 0;
    b[n] = '\0';
    fclose(f);
    if (len) *len = (size_t)n;
    return b;
}

static int test_keys(pn_minisign_pubkey *k)
{
    char *t = slurp(FX "test_catalog.pub", NULL);
    int ok = t && pn_minisign_parse_pubkey(t, k) == 0;
    free(t);
    return ok ? 1 : 0;
}

static int accept_file(pn_catalog *c, const char *json, const char *sig, char *err)
{
    size_t len = 0;
    char *m = slurp(json, &len), *s = sig ? slurp(sig, NULL) : NULL;
    int rc = pn_catalog_accept(c, m, len, s, err, 256);
    free(m);
    free(s);
    return rc;
}

static void test_accept_and_rollback(void)
{
    pn_minisign_pubkey k;
    char err[256];

    printf("accept_and_rollback:\n");
    lt_clean_dir(STATE);
    lt_make_dir(STATE);
    int nk = test_keys(&k);
    pn_catalog *c = pn_catalog_open(STATE, &k, nk);
    CHECK(c && pn_catalog_version(c) == -1, "empty at first");
    CHECK(accept_file(c, FX "catalog_v1.json", NULL, err) == -1, "unsigned refused");
    CHECK(accept_file(c, FX "catalog_v2_tampered.json", FX "catalog_v2_tampered.json.minisig",
                      err) == -1, "tampered refused");
    CHECK(accept_file(c, FX "catalog_v2.json", FX "catalog_v2.json.minisig", err) == 0,
          "v2 accepted");
    CHECK(pn_catalog_version(c) == 2, "version 2");
    CHECK(accept_file(c, FX "catalog_v1.json", FX "catalog_v1.json.minisig", err) == -1 &&
          strstr(err, "rollback"), "v1 after v2 refused (rollback)");
    CHECK(accept_file(c, FX "catalog_v2.json", FX "catalog_v2.json.minisig", err) == 0,
          "same version again accepted");
    pn_catalog_close(c);

    c = pn_catalog_open(STATE, &k, nk);
    CHECK(pn_catalog_version(c) == 2, "accepted catalog survives a reopen (offline boot)");
    pn_catalog_manifest *m = pn_catalog_snapshot(c);
    CHECK(m && m->n_models == 3, "snapshot has 3 models");
    free(m);
    pn_catalog_close(c);

    lt_clean_dir(STATE "/nokeys");
    lt_make_dir(STATE "/nokeys");
    c = pn_catalog_open(STATE "/nokeys", &k, 0);
    CHECK(accept_file(c, FX "catalog_v2.json", FX "catalog_v2.json.minisig", err) == -1 &&
          strstr(err, "trusts no catalog key"), "a build with no key refuses everything");
    pn_catalog_close(c);
}

static void test_refresh_file_url(void)
{
    pn_minisign_pubkey k;
    char err[256], url[1200], cwd[1024];

    printf("refresh_file_url:\n");
    lt_clean_dir(STATE "/refresh");
    lt_make_dir(STATE "/refresh");
    int nk = test_keys(&k);
    pn_catalog *c = pn_catalog_open(STATE "/refresh", &k, nk);
    if (!getcwd(cwd, sizeof(cwd))) cwd[0] = '\0';
    snprintf(url, sizeof(url), "file://%s/" FX "catalog_v1.json", cwd);
    CHECK(pn_catalog_refresh(c, url, PN_CURL_PROTOCOLS_HTTPS, err, sizeof(err)) == -1,
          "file:// refused under https-only");
    CHECK(pn_catalog_refresh(c, url, "=file", err, sizeof(err)) == 0 && pn_catalog_version(c) == 1,
          "fetched + verified over curl");
    snprintf(url, sizeof(url), "file://%s/" FX "nope.json", cwd);
    CHECK(pn_catalog_refresh(c, url, "=file", err, sizeof(err)) == -1, "missing manifest refused");
    pn_catalog_close(c);
}

static void test_validation(void)
{
    pn_catalog_manifest *m = calloc(1, sizeof(*m));
    char err[256];
    char *good = slurp(FX "catalog_v1.json", NULL);

    printf("validation:\n");
    CHECK(pn_cat_manifest_parse(good, m, err, sizeof(err)) == 0, "fixture parses");
    const char *swaps[][2] = {
        { "\"teamide/test-fixtures\"", "\"someone/else\"" },
        { "\"0123456789abcdef0123456789abcdef01234567\"", "\"main\"" },
        { "\"big-two-gpu.gguf\"", "\"../etc/passwd\"" },
        { "\"min_p\": 0", "\"min_p\": 0.1" },
        { "\"8k\": 10.0", "\"8k\": 0" },
        { "\"schema\": 1", "\"schema\": 9" },
    };
    for (size_t i = 0; i < sizeof(swaps) / sizeof(swaps[0]); i++) {
        char *bad = strdup(good), *at = strstr(bad, swaps[i][0]);
        char buf[16384];
        if (!at) { CHECK(0, swaps[i][0]); free(bad); continue; }
        snprintf(buf, sizeof(buf), "%.*s%s%s", (int)(at - bad), bad, swaps[i][1],
                 at + strlen(swaps[i][0]));
        CHECK(pn_cat_manifest_parse(buf, m, err, sizeof(err)) == -1, swaps[i][1]);
        free(bad);
    }
    {   /* HF org names are case-insensitive: any casing of our org is accepted */
        const char *cases[] = { "TeamIDE", "TEAMIDE", "teamide" };
        for (size_t i = 0; i < 3; i++) {
            char *c = strdup(good), *at = strstr(c, "\"teamide/test-fixtures\"");
            char buf[16384];
            if (!at) { CHECK(0, "org fixture"); free(c); continue; }
            snprintf(buf, sizeof(buf), "%.*s\"%s/test-fixtures\"%s", (int)(at - c), c, cases[i],
                     at + strlen("\"teamide/test-fixtures\""));
            CHECK(pn_cat_manifest_parse(buf, m, err, sizeof(err)) == 0, cases[i]);
            free(c);
        }
    }
    free(good);
    free(m);
}

static void make_k80(pn_hw_report *hw, int cards, int busy_unit)
{
    memset(hw, 0, sizeof(*hw));
    for (int c = 0; c < cards; c++) {
        hw->cards[c].id = c;
        snprintf(hw->cards[c].type, sizeof(hw->cards[c].type), "k80");
        hw->cards[c].first_unit = 2 * c;
        hw->cards[c].n_units = 2;
        for (int k = 0; k < 2; k++) {
            pn_hw_unit *u = &hw->units[2 * c + k];
            u->id = u->die = 2 * c + k;
            u->card = c;
            snprintf(u->type, sizeof(u->type), "k80-gpu");
            u->mem_total = u->mem_free = GIB(11);
        }
    }
    hw->n_cards = cards;
    hw->n_units = 2 * cards;
    if (busy_unit >= 0)
        hw->units[busy_unit].n_models = 1;
}

static void test_match(void)
{
    pn_catalog_manifest *m = calloc(1, sizeof(*m));
    pn_cat_match e[PN_CAT_MAX_ENTRIES];
    pn_hw_report hw;
    char err[256], *text = slurp(FX "catalog_v1.json", NULL);

    printf("match:\n");
    pn_cat_manifest_parse(text, m, err, sizeof(err));
    make_k80(&hw, 2, -1);
    int n = pn_cat_match_profiles(m, &hw, GIB(8), e, PN_CAT_MAX_ENTRIES);
    CHECK(n == 4, "4 profiles");
    CHECK(e[0].fits && strcmp(e[0].profile->id, "k80-pair") == 0, "rank 1 fit first: k80-pair");
    CHECK(e[1].fits && strcmp(e[1].profile->id, "k80-one") == 0, "then k80-one (rank 2)");
    CHECK(!e[2].fits && strcmp(e[2].profile->id, "p150-one") == 0 &&
          strcmp(e[2].reason, "needs 1 p150, this box has 0") == 0, "p150 greyed with reason");
    CHECK(!e[3].fits && strcmp(e[3].profile->id, "cpu") == 0, "cpu greyed on a K80 box");

    make_k80(&hw, 1, 0);   /* one card, one GPU busy */
    n = pn_cat_match_profiles(m, &hw, GIB(8), e, PN_CAT_MAX_ENTRIES);
    CHECK(e[0].fits && strcmp(e[0].profile->id, "k80-one") == 0, "only k80-one fits now");
    const pn_cat_match *pair = NULL;
    for (int i = 0; i < n; i++)
        if (strcmp(e[i].profile->id, "k80-pair") == 0) pair = &e[i];
    CHECK(pair && !pair->fits &&
          strcmp(pair->reason, "needs 2 free k80-gpu on one card, you have 1") == 0,
          "k80-pair greyed: needs 2 free GPUs on one card");
    free(text);
    free(m);
}

/* Parse `json` with `from` swapped for `to`; 0 or -1. */
static int parse_swapped(const char *json, const char *from, const char *to,
                         pn_catalog_manifest *m, char *err)
{
    const char *at = strstr(json, from);
    char *buf = malloc(strlen(json) + strlen(to) + 1);
    int rc;

    if (!at || !buf) {
        free(buf);
        return 1;
    }
    sprintf(buf, "%.*s%s%s", (int)(at - json), json, to, at + strlen(from));
    rc = pn_cat_manifest_parse(buf, m, err, 256);
    free(buf);
    return rc;
}

static int files_for(const pn_cat_model *model, unsigned units, pn_cat_model *scratch)
{
    pn_cat_model_for_units(model, units, scratch);
    return scratch->n_files;
}

static void test_unit_types(void)
{
    pn_catalog_manifest *m = calloc(1, sizeof(*m));
    pn_cat_model *need = calloc(1, sizeof(*need));
    char err[256];
    char *fx = slurp(FX "catalog_unit_types.json", NULL);
    const char *tag = "\"unit_types\": [\n      \"p150\"\n     ]";

    printf("unit_types:\n");
    CHECK(pn_cat_manifest_parse(fx, m, err, sizeof(err)) == 0, err);
    const pn_cat_model *o = &m->models[0];
    CHECK(o->files[0].unit_types == 0 && o->files[1].unit_types == PN_CAT_UNIT_P150 &&
          o->files[2].unit_types == PN_CAT_UNIT_P150, "weights untagged, extras p150");
    CHECK(files_for(o, PN_CAT_UNIT_K80_GPU, need) == 1 &&
          strcmp(need->files[0].filename, "orn-mini-Q8_0.gguf") == 0, "k80-gpu: the GGUF only");
    CHECK(files_for(o, PN_CAT_UNIT_CPU, need) == 1, "cpu: the GGUF only");
    CHECK(files_for(o, PN_CAT_UNIT_P150, need) == 3, "p150: GGUF + both .fttw");
    CHECK(files_for(o, PN_CAT_UNITS_ALL, need) == 3, "all units: every file");
    CHECK(parse_swapped(fx, "\"p150\"\n     ]", "\"p151\"\n     ]", m, err) == -1 &&
          strstr(err, "unit_types"), "unknown unit type refused");
    CHECK(parse_swapped(fx, tag, "\"unit_types\": \"p150\"", m, err) == -1 &&
          strstr(err, "non-empty array"), "unit_types not an array refused");
    CHECK(parse_swapped(fx, tag, "\"unit_types\": []", m, err) == -1 &&
          strstr(err, "non-empty array"), "empty unit_types refused");
    CHECK(parse_swapped(fx, "\"role\": \"weights\",",
                        "\"role\": \"weights\", \"unit_types\": [\"p150\"],", m, err) == -1 &&
          strstr(err, "exactly one weights file"),
          "the only weights tagged p150: its k80-gpu / cpu profiles have none, refused");
    CHECK(parse_swapped(fx, "\"k80-gpu\"\n     ]", "\"k80-gpu\", \"p150\"\n     ]", m, err) == -1 &&
          strstr(err, "exactly one weights file"), "two weights files for p150 refused");
    CHECK(parse_swapped(fx, "\"unit_type\": \"p150\"", "\"unit_type\": \"cpu\"", m, err) == -1 &&
          strstr(err, "no unit type"), "a tag no profile uses refused");
    CHECK(pn_cat_manifest_parse(fx, m, err, sizeof(err)) == 0, err);
    o = pn_cat_find_model(m, "q36-mini-q8");
    CHECK(o && files_for(o, PN_CAT_UNIT_K80_GPU, need) == 1 &&
          strcmp(pn_cat_weights_file(need)->filename, "q36-mini-Q8_0.gguf") == 0,
          "per-unit weights, k80-gpu: the K80 GGUF only");
    CHECK(o && files_for(o, PN_CAT_UNIT_P150, need) == 3 &&
          strcmp(pn_cat_weights_file(need)->filename, "q36-mini-Q8_0.meta.gguf") == 0 &&
          strcmp(need->files[1].filename, "q36-mini-Q8_0.meta.moe_bfp6.fttw") == 0 &&
          strcmp(need->files[2].filename, "q36-mini-Q8_0.meta.moe_bfp6.mtp.fttw") == 0,
          "per-unit weights, p150: the metadata GGUF (its weights) + both .fttw, no K80 GGUF");
    free(fx);

    /* The shipped catalog: a K80 Ornith install skips the TT-only .fttw files. */
    fx = slurp("catalog/catalog.json", NULL);
    CHECK(fx && pn_cat_manifest_parse(fx, m, err, sizeof(err)) == 0, "catalog/catalog.json parses");
    o = pn_cat_find_model(m, "ornith-1.5-9b-q8");
    const pn_cat_profile *k80 = o ? pn_cat_find_profile(o, "k80-one") : NULL;
    const pn_cat_profile *tt = o ? pn_cat_find_profile(o, "p150-one") : NULL;
    CHECK(k80 && tt, "Ornith k80-one + p150-one");
    if (k80 && tt) {
        CHECK(files_for(o, pn_cat_unit_type_bit(k80->hardware.unit_type), need) == 1 &&
              strcmp(need->files[0].filename, "Ornith-1.5-9B-Q8_0.gguf") == 0,
              "Ornith k80-one installs the GGUF only");
        CHECK(files_for(o, pn_cat_unit_type_bit(tt->hardware.unit_type), need) == 3,
              "Ornith p150-one installs the GGUF + both .fttw");
    }
    /* Qwen3.6 Q8_0: a p150 install gets the three TT files, never the 37.8 GB K80 GGUF;
     * a K80 install gets that GGUF only. */
    o = pn_cat_find_model(m, "qwen3.6-35b-a3b-q8");
    k80 = o ? pn_cat_find_profile(o, "k80-two-boards") : NULL;
    tt = o ? pn_cat_find_profile(o, "p150-one") : NULL;
    CHECK(k80 && tt, "Qwen3.6 Q8_0 k80-two-boards + p150-one");
    if (k80 && tt) {
        CHECK(files_for(o, pn_cat_unit_type_bit(k80->hardware.unit_type), need) == 1 &&
              strcmp(need->files[0].filename, "Qwen3.6-35B-A3B-Q8_0.gguf") == 0 &&
              strcmp(need->files[0].role, "weights") == 0,
              "Qwen3.6 k80-two-boards installs (and loads) the K80 GGUF only");
        CHECK(files_for(o, pn_cat_unit_type_bit(tt->hardware.unit_type), need) == 3 &&
              strcmp(pn_cat_weights_file(need)->filename,
                     "Qwen_Qwen3.6-35B-A3B-Q8_0.meta.gguf") == 0 &&
              strcmp(need->files[1].filename,
                     "Qwen_Qwen3.6-35B-A3B-Q8_0.meta.moe_bfp6.fttw") == 0 &&
              strcmp(need->files[2].filename,
                     "Qwen_Qwen3.6-35B-A3B-Q8_0.meta.moe_bfp6.mtp.fttw") == 0,
              "Qwen3.6 p150-one installs the metadata GGUF (loaded) + trunk + MTP .fttw only");
        CHECK(tt->ctx_cap == 36864 && tt->options.present &&
              strcmp(tt->options.kv_type, "bf16") == 0 && tt->options.mem_kv_per_token == 22528,
              "Qwen3.6 p150-one: 36,864 tokens, bf16 KV, 22,528 B a token");
    }
    /* Qwen3.8 Q8_0: a K80 install gets the unmodified 29 GB GGUF only; a p150 install never
     * does. */
    o = pn_cat_find_model(m, "qwen3.8-27b-q8");
    k80 = o ? pn_cat_find_profile(o, "k80-two-boards") : NULL;
    tt = o ? pn_cat_find_profile(o, "p150-one") : NULL;
    CHECK(k80 && tt, "Qwen3.8 k80-two-boards + p150-one");
    if (k80 && tt) {
        CHECK(files_for(o, pn_cat_unit_type_bit(k80->hardware.unit_type), need) == 1 &&
              strcmp(need->files[0].filename, "Qwen3.8-27B-Q8_0.gguf") == 0 &&
              strcmp(need->files[0].role, "weights") == 0 &&
              need->files[0].size == 29047086048ull,
              "Qwen3.8 k80-two-boards installs (and loads) the K80 GGUF only");
        CHECK(files_for(o, pn_cat_unit_type_bit(tt->hardware.unit_type), need) == 3 &&
              strcmp(pn_cat_weights_file(need)->filename, "Qwen3.8-27B-Q8_0.meta.gguf") == 0,
              "Qwen3.8 p150-one still installs the metadata GGUF + trunk + MTP .fttw only");
        CHECK(k80->ctx_cap == 32768 && k80->max_sessions == 1 && k80->mtp &&
              k80->mtp_drafts == 0 && strcmp(k80->options.kv_type, "q4") == 0,
              "Qwen3.8 k80-two-boards: 32,768 tokens, one user, Q4 KV, MTP (adaptive depth)");
    }
    free(fx);
    free(need);
    free(m);
}

/* A p150 profile's KV type and MTP draft count (options.kv_type = a PN_TT_KV_TYPE value,
 * load.mtp_drafts): Qwen3.8 p150-one is K8V4 + MTP k1 + its MTP head file; the fields are
 * refused off a p150 and out of range. A box built before them refuses the whole catalog on
 * the kv_type and keeps the version it accepted (catalog.c). */
static void test_tt_profile_fields(void)
{
    pn_catalog_manifest *m = calloc(1, sizeof(*m));
    pn_cat_model *need = calloc(1, sizeof(*need));
    char err[256];
    char *fx = slurp("catalog/catalog.json", NULL);
    const char *k1 = "\"mtp\": true,\n            \"mtp_drafts\": 1";

    printf("tt_profile_fields:\n");
    CHECK(fx && pn_cat_manifest_parse(fx, m, err, sizeof(err)) == 0, "catalog/catalog.json parses");
    const pn_cat_model *o = pn_cat_find_model(m, "qwen3.8-27b-q8");
    const pn_cat_profile *k8v4 = o ? pn_cat_find_profile(o, "p150-one") : NULL;
    const pn_cat_profile *bf16 = o ? pn_cat_find_profile(o, "p150-one-bf16") : NULL;
    CHECK(k8v4 && bf16, "Qwen3.8 p150-one + p150-one-bf16");
    if (k8v4 && bf16) {
        CHECK(k8v4->rank < bf16->rank && k8v4->ctx_cap == 30720 && k8v4->mtp &&
              k8v4->mtp_drafts == 1 && strcmp(k8v4->options.kv_type, "k8v4") == 0 &&
              k8v4->options.mem_kv_per_token == 17u * 1664u,
              "Qwen3.8 p150-one (rank 1): 30,720 tokens, MTP k1, K8V4 KV (17 x 1,664 B a token: "
              "16 layers + the MTP head's)");
        CHECK(bf16->ctx_cap == 30720 && !bf16->mtp && bf16->mtp_drafts == 0 &&
              strcmp(bf16->options.kv_type, "bf16") == 0,
              "Qwen3.8 p150-one-bf16 (rank 2): 30,720 tokens, MTP off, bf16 KV");
        /* Measured (p150, 2026-11-13, K8V4 + k1 serve census): "reserve: prefix checkpoint —
         * 31.592 GiB used of 31.875 GiB", taken at the high end of its 3-decimal rounding. */
        CHECK(pn_load_option_need(&k8v4->options, 30720, 1) ==
                  (uint64_t)(31.5925 * 1073741824.0),
              "K8V4 + MTP k1 + checkpoint at 30,720: the measured card use (31.592 GiB)");
        CHECK(files_for(o, PN_CAT_UNIT_P150, need) == 3 &&
              strcmp(pn_cat_weights_file(need)->filename, "Qwen3.8-27B-Q8_0.meta.gguf") == 0 &&
              strcmp(need->files[2].filename, "Qwen3.8-27B-Q8_0.meta.bfp8.mtp.fttw") == 0,
              "Qwen3.8 p150 installs the metadata GGUF + trunk + MTP head .fttw");
    }
    CHECK(pn_cat_kv_type_allowed("p150", "k8v4") && pn_cat_kv_type_allowed("p150", "bfp4") &&
          pn_cat_kv_type_allowed("p150", "k16v8") && pn_cat_kv_type_allowed("p150", "bf16") &&
          pn_cat_kv_type_allowed("k80-gpu", "q4") && pn_cat_kv_type_allowed("cpu", "fp32"),
          "TT KV types on a p150, q4 / fp32 / bf16 anywhere");
    CHECK(!pn_cat_kv_type_allowed("k80-gpu", "k8v4") && !pn_cat_kv_type_allowed("cpu", "bfp8") &&
          !pn_cat_kv_type_allowed("p150", "k8v8") && !pn_cat_kv_type_allowed("p150", "") &&
          !pn_cat_kv_type_allowed("p150", NULL),
          "no TT KV type off a p150; unknown names refused");
    CHECK(parse_swapped(fx, "\"kv_type\": \"k8v4\"", "\"kv_type\": \"k8v5\"", m, err) == -1 &&
          strstr(err, "kv_type"), "an unknown p150 kv_type refuses the catalog");
    CHECK(parse_swapped(fx, "\"kv_type\": \"q4\"", "\"kv_type\": \"k8v4\"", m, err) == -1 &&
          strstr(err, "k80-pair") && strstr(err, "kv_type"), "k8v4 on a K80 profile refused");
    CHECK(parse_swapped(fx, k1, "\"mtp\": true,\n            \"mtp_drafts\": 16", m, err) == -1 &&
          strstr(err, "mtp_drafts"), "mtp_drafts 16 refused");
    CHECK(parse_swapped(fx, k1, "\"mtp\": true,\n            \"mtp_drafts\": -1", m, err) == -1 &&
          strstr(err, "mtp_drafts"), "mtp_drafts -1 refused");
    CHECK(parse_swapped(fx, k1, "\"mtp\": false,\n            \"mtp_drafts\": 1", m, err) == -1 &&
          strstr(err, "mtp_drafts"), "mtp_drafts with mtp false refused");
    CHECK(parse_swapped(fx, "\"reasoning_budget\": 2000,\n            \"mtp\": true",
                        "\"reasoning_budget\": 2000,\n            \"mtp\": true, \"mtp_drafts\": 1",
                        m, err) == -1 && strstr(err, "k80-pair") && strstr(err, "mtp_drafts"),
          "mtp_drafts on a K80 profile refused");
    CHECK(parse_swapped(fx, k1, "\"mtp\": true,\n            \"mtp_drafts\": 3", m, err) == 0,
          "mtp_drafts 3 parses");
    free(fx);
    free(need);
    free(m);
}

/* files[].ingot (Ingot Phase 1b): an optional compressed download of a file, checked like the
 * file's own fields; repo and revision default to the file's; absent = no .ingot. */
static void test_ingot_fields(void)
{
    pn_catalog_manifest *m = calloc(1, sizeof(*m));
    char err[256], url[512];
    char *fx = slurp(FX "catalog_v1.json", NULL);
    const char *at = "\"filename\": \"big-two-gpu.gguf\",";
    const char *sha = "\"0123456789abcdef0123456789abcdef0123456789abcdef0123456789abcdef\"";
    char to[1024];

    printf("ingot_fields:\n");
    CHECK(fx && pn_cat_manifest_parse(fx, m, err, sizeof(err)) == 0 &&
          !m->models[0].files[0].ingot.present &&
          pn_cat_ingot_url(NULL, &m->models[0].files[0], url, sizeof(url)) == -1,
          "absent: no .ingot, no ingot URL");
    snprintf(to, sizeof(to), "%s \"ingot\": { \"filename\": \"big-two-gpu.gguf.ingot\", "
             "\"size\": 700 },", at);
    CHECK(parse_swapped(fx, at, to, m, err) == 0, err);
    const pn_cat_ingot *g = &m->models[0].files[0].ingot;
    CHECK(g->present && strcmp(g->filename, "big-two-gpu.gguf.ingot") == 0 && g->size == 700 &&
          strcmp(g->hf_repo, "teamide/test-fixtures") == 0 &&
          strcmp(g->revision, "0123456789abcdef0123456789abcdef01234567") == 0 &&
          g->base_sha256[0] == '\0' && m->models[0].files[0].size == 1000,
          "minimal: repo + revision default to the file's; the file keeps its size");
    CHECK(pn_cat_ingot_url("http://h", &m->models[0].files[0], url, sizeof(url)) == 0 &&
          strcmp(url, "http://h/teamide/test-fixtures/resolve/"
                      "0123456789abcdef0123456789abcdef01234567/big-two-gpu.gguf.ingot") == 0,
          "ingot URL");
    snprintf(to, sizeof(to), "%s \"ingot\": { \"filename\": \"d.ingot\", \"size\": 9, "
             "\"hf_repo\": \"TeamIDE/deltas\", \"revision\": "
             "\"fedcba9876543210fedcba9876543210fedcba98\", \"base_sha256\": %s },", at, sha);
    CHECK(parse_swapped(fx, at, to, m, err) == 0, err);
    CHECK(strcmp(g->hf_repo, "TeamIDE/deltas") == 0 && strncmp(g->revision, "fedcba", 6) == 0 &&
          strncmp(g->base_sha256, "0123456789abcdef", 16) == 0 && strlen(g->base_sha256) == 64,
          "full: own repo, revision, base_sha256");
    const char *bad[] = {
        "\"ingot\": 5,",
        "\"ingot\": { \"size\": 700 },",
        "\"ingot\": { \"filename\": \"../x.ingot\", \"size\": 700 },",
        "\"ingot\": { \"filename\": \".x.ingot\", \"size\": 700 },",
        "\"ingot\": { \"filename\": \"x.ingot\" },",
        "\"ingot\": { \"filename\": \"x.ingot\", \"size\": 0 },",
        "\"ingot\": { \"filename\": \"x.ingot\", \"size\": 2e12 },",
        "\"ingot\": { \"filename\": \"x.ingot\", \"size\": 7, \"hf_repo\": \"someone/else\" },",
        "\"ingot\": { \"filename\": \"x.ingot\", \"size\": 7, \"hf_repo\": \"teamide/a/b\" },",
        "\"ingot\": { \"filename\": \"x.ingot\", \"size\": 7, \"revision\": \"main\" },",
        "\"ingot\": { \"filename\": \"x.ingot\", \"size\": 7, \"revision\": \"0123\" },",
        "\"ingot\": { \"filename\": \"x.ingot\", \"size\": 7, \"base_sha256\": \"abc\" },",
        ("\"ingot\": { \"filename\": \"x.ingot\", \"size\": 7, \"base_sha256\": "
         "\"0123456789ABCDEF0123456789abcdef0123456789abcdef0123456789abcdef\" },"),
        "\"ingot\": { \"filename\": \"x.ingot\", \"size\": 7, \"base_sha256\": \"\" },",
    };
    for (size_t i = 0; i < sizeof(bad) / sizeof(bad[0]); i++) {
        snprintf(to, sizeof(to), "%s %s", at, bad[i]);
        CHECK(parse_swapped(fx, at, to, m, err) == -1 && strstr(err, "model "), bad[i]);
    }
    free(fx);
    free(m);
}

/* load.reasoning_reserve (reasoning_budget.h): absent = 0 (the serve default), a
 * value lands on the profile, a negative one refuses the catalog. */
static void test_reasoning_reserve(void)
{
    pn_catalog_manifest *m = calloc(1, sizeof(*m));
    char err[256];
    char *fx = slurp("catalog/catalog.json", NULL);
    const char *from = "\"reasoning_budget\": 2000,\n            \"mtp\": true";

    printf("reasoning_reserve:\n");
    CHECK(fx && pn_cat_manifest_parse(fx, m, err, sizeof(err)) == 0, "catalog/catalog.json parses");
    const pn_cat_model *q = pn_cat_find_model(m, "qwen3.6-35b-a3b-q3km");
    const pn_cat_profile *p = q ? pn_cat_find_profile(q, "k80-pair") : NULL;
    CHECK(p && p->reasoning_budget == 2000 && p->reasoning_reserve == 0,
          "absent reasoning_reserve = 0 (serve default)");
    CHECK(parse_swapped(fx, from, "\"reasoning_budget\": 5120, \"reasoning_reserve\": 3072,\n"
                                  "            \"mtp\": true", m, err) == 0,
          "reasoning_reserve 3072 parses");
    q = pn_cat_find_model(m, "qwen3.6-35b-a3b-q3km");
    p = q ? pn_cat_find_profile(q, "k80-pair") : NULL;
    CHECK(p && p->reasoning_budget == 5120 && p->reasoning_reserve == 3072,
          "reasoning_reserve lands on the profile");
    CHECK(parse_swapped(fx, from, "\"reasoning_budget\": 2000, \"reasoning_reserve\": -1,\n"
                                  "            \"mtp\": true", m, err) == -1 &&
          strstr(err, "load values"), "a negative reasoning_reserve refuses the catalog");
    free(fx);
    free(m);
}

int main(void)
{
    test_accept_and_rollback();
    test_reasoning_reserve();
    test_refresh_file_url();
    test_validation();
    test_match();
    test_unit_types();
    test_tt_profile_fields();
    test_ingot_fields();
    return lt_report("test_local_catalog");
}
