/* SPDX-License-Identifier: AGPL-3.0-or-later */
/* Copyright (C) 2026 David (Corey) Herrera */
/*
 * test_local_load_options.c — Pantheon Local L1-L3: smart load options.
 *
 *   - the grid of context x users choices, the memory model's need per unit;
 *   - which choices fit the units free now (device free = mem_free + reserve),
 *     the recommended one, and the load-time check with its plain refusals;
 *   - the manifest "options" block (Q4 KV refused, default outside the ranges
 *     refused);
 *   - catalog/catalog.json's memory model against the device peaks measured on
 *     the real cards (tests/local/mem_model_measure.sh): every measured point
 *     is predicted within MODEL_TOLERANCE, and every choice the box offers on
 *     an idle card has PN_LOAD_FIT_MARGIN_BYTES of room above its prediction.
 */

#include "local_test_support.h"

#include "../src/local/catalog_manifest.h"
#include "../src/local/catalog_match.h"
#include "../src/local/load_options.h"
#include "../src/core/mem/pn_ladder.h"
#include "../src/local/local_file_io.h"
#include "../src/local/local_kv_policy.h"

#define GIB(n) ((uint64_t)(n) << 30)
#define MIB(n) ((uint64_t)(n) << 20)
/* Measured peaks vs the model: the model never predicts less than a measured
 * peak, and at most this much more (it is fitted to the worst case, a prompt near
 * the cap; a short prompt peaks up to ~30 MiB lower on the K80). */
#define MODEL_TOLERANCE MIB(32)

/* K80 (12,799,180,800 B per die, 1 GiB reserve) or p150 (one unit). */
static void make_box(pn_hw_report *hw, const char *type, int n_cards, uint64_t device_total,
                     const int *busy, int n_busy)
{
    bool k80 = strcmp(type, "k80-gpu") == 0;
    int per_card = k80 ? 2 : 1;

    memset(hw, 0, sizeof(*hw));
    hw->family = k80 ? PN_HW_FAMILY_K80 : PN_HW_FAMILY_P150;
    hw->die_reserve = GIB(1);
    for (int c = 0; c < n_cards; c++) {
        hw->cards[c].id = c;
        snprintf(hw->cards[c].type, sizeof(hw->cards[c].type), "%s", k80 ? "k80" : "p150");
        hw->cards[c].first_unit = per_card * c;
        hw->cards[c].n_units = per_card;
        for (int k = 0; k < per_card; k++) {
            pn_hw_unit *u = &hw->units[hw->n_units++];
            u->id = u->die = per_card * c + k;
            u->card = c;
            snprintf(u->type, sizeof(u->type), "%s", type);
            u->mem_total = u->mem_free = device_total - GIB(1);
        }
    }
    hw->n_cards = n_cards;
    for (int i = 0; i < n_busy; i++) {
        hw->units[busy[i]].n_models = 1;
        hw->units[busy[i]].mem_free = GIB(1);
    }
}

/* A K80-pair profile with a round-number memory model. */
static pn_cat_profile k80_profile(void)
{
    pn_cat_profile p;

    memset(&p, 0, sizeof(p));
    snprintf(p.id, sizeof(p.id), "k80-pair");
    snprintf(p.hardware.unit_type, sizeof(p.hardware.unit_type), "k80-gpu");
    p.hardware.count = 2;
    p.hardware.same_card = true;
    p.ctx_cap = 16384;
    p.max_sessions = 1;
    p.options.present = true;
    snprintf(p.options.kv_type, sizeof(p.options.kv_type), "fp32");
    p.options.ctx_min = 8192;
    p.options.ctx_max = 65536;
    p.options.sessions_max = 4;
    p.options.mem_fixed = GIB(10);
    p.options.mem_ctx_per_token = 20480;
    p.options.mem_per_session = MIB(34);
    p.options.mem_kv_per_token = 20480;
    return p;
}

static const pn_load_option *find(const pn_load_option *o, int n, int ctx, int users)
{
    for (int i = 0; i < n; i++)
        if (o[i].ctx_cap == ctx && o[i].max_sessions == users)
            return &o[i];
    return NULL;
}

static void test_grid_and_need(void)
{
    pn_cat_profile p = k80_profile();
    pn_load_option o[PN_LOAD_OPTIONS_MAX];
    int n = pn_load_options_grid(&p, o, PN_LOAD_OPTIONS_MAX);

    printf("grid_and_need:\n");
    CHECK(n == 4 * 3, "8k/16k/32k/64k x 1/2/4 users");
    CHECK(o[0].ctx_cap == 8192 && o[0].max_sessions == 1, "sorted: smallest first");
    CHECK(o[n - 1].ctx_cap == 65536 && o[n - 1].max_sessions == 4, "largest last");
    CHECK(pn_load_option_need(&p.options, 8192, 2) ==
              GIB(10) + 8192ull * 20480 + 2 * (MIB(34) + 8192ull * 20480),
          "need = fixed + ctx*shared + users*(per_session + ctx*kv)");
    p.ctx_cap = 40960;   /* a non-round default is offered too */
    n = pn_load_options_grid(&p, o, PN_LOAD_OPTIONS_MAX);
    CHECK(find(o, n, 40960, 1) && !find(o, n, 40960, 3), "the default's context, round users");
    pn_cat_profile fixed = k80_profile();
    fixed.options.present = false;
    n = pn_load_options_grid(&fixed, o, PN_LOAD_OPTIONS_MAX);
    CHECK(n == 1 && o[0].ctx_cap == 16384 && o[0].max_sessions == 1,
          "no options block: the profile's fixed settings only");
}

static void test_fit_on_free_units(void)
{
    pn_cat_profile p = k80_profile();
    pn_load_option o[PN_LOAD_OPTIONS_MAX];
    pn_hw_report hw;
    uint64_t dev = GIB(12);
    int busy[] = { 0, 1 };

    printf("fit_on_free_units:\n");
    make_box(&hw, "k80-gpu", 2, dev, NULL, 0);
    int n = pn_load_options_fit(&p, &hw, o, PN_LOAD_OPTIONS_MAX);
    for (int i = 0; i < n; i++) {
        bool room = o[i].need_per_unit + PN_LOAD_FIT_MARGIN_BYTES <= dev;
        CHECK(o[i].fits == room, "fits exactly when need + margin <= the free device");
    }
    CHECK(find(o, n, 16384, 1)->fits && find(o, n, 16384, 1)->recommended,
          "the profile's default fits and is recommended");
    CHECK(!find(o, n, 65536, 4)->fits, "64k x 4 does not fit 12 GiB");
    int recommended = 0;
    for (int i = 0; i < n; i++)
        recommended += o[i].recommended;
    CHECK(recommended == 1, "exactly one recommended");

    make_box(&hw, "k80-gpu", 2, dev, busy, 2);   /* card 0 taken */
    n = pn_load_options_fit(&p, &hw, o, PN_LOAD_OPTIONS_MAX);
    CHECK(find(o, n, 16384, 1)->fits, "the other card still offers it (another copy)");
    make_box(&hw, "k80-gpu", 1, dev, busy, 1);   /* half of the only card taken */
    n = pn_load_options_fit(&p, &hw, o, PN_LOAD_OPTIONS_MAX);
    CHECK(!find(o, n, 8192, 1)->fits, "one free GPU on the card: nothing fits a pair");

    /* A smaller card: the default no longer fits, the biggest smaller one is picked. */
    make_box(&hw, "k80-gpu", 1, GIB(10) + MIB(256) + MIB(34) + MIB(340), NULL, 0);
    n = pn_load_options_fit(&p, &hw, o, PN_LOAD_OPTIONS_MAX);
    CHECK(find(o, n, 8192, 1)->fits && !find(o, n, 16384, 1)->fits, "only 8k x 1 fits");
    CHECK(find(o, n, 8192, 1)->recommended, "so it is recommended instead");
}

static void test_choose_and_refuse(void)
{
    pn_cat_profile p = k80_profile();
    pn_load_option chosen;
    pn_hw_report hw;
    char why[256];
    int busy[] = { 1 };

    printf("choose_and_refuse:\n");
    make_box(&hw, "k80-gpu", 1, GIB(12), NULL, 0);
    CHECK(pn_load_options_choose(&p, &hw, 0, 0, &chosen, why, sizeof(why)) == 0 &&
              chosen.ctx_cap == 16384 && chosen.max_sessions == 1,
          "nothing asked: the recommended choice");
    CHECK(pn_load_options_choose(&p, &hw, 16384, 2, &chosen, why, sizeof(why)) == 0 &&
              chosen.ctx_cap == 16384 && chosen.max_sessions == 2,
          "16k x 2 fits and is taken");
    CHECK(pn_load_options_choose(&p, &hw, 8192, 0, &chosen, why, sizeof(why)) == 0 &&
              chosen.max_sessions == 1, "users absent: the profile's own");
    CHECK(pn_load_options_choose(&p, &hw, 65536, 4, &chosen, why, sizeof(why)) != 0 &&
              strstr(why, "65536 tokens x 4 user(s) needs") && strstr(why, "at most"),
          "too big: refused with the memory it needs");
    CHECK(pn_load_options_choose(&p, &hw, 20000, 1, &chosen, why, sizeof(why)) != 0 &&
              strstr(why, "20000 tokens is not an offered context (8192, 16384, 32768, 65536)"),
          "an off-grid context: refused with the offered ones");
    CHECK(pn_load_options_choose(&p, &hw, 8192, 3, &chosen, why, sizeof(why)) != 0 &&
              strstr(why, "3 users is not an offered number of users (1, 2, 4)"),
          "an off-grid user count: refused with the offered ones");
    make_box(&hw, "k80-gpu", 1, GIB(12), busy, 1);
    CHECK(pn_load_options_choose(&p, &hw, 8192, 1, &chosen, why, sizeof(why)) != 0 &&
              strstr(why, "needs 2 free k80-gpu on one card"),
          "short of units: says so, not memory");
    pn_cat_profile fixed = k80_profile();
    fixed.options.present = false;
    make_box(&hw, "k80-gpu", 1, GIB(12), NULL, 0);
    CHECK(pn_load_options_choose(&fixed, &hw, 32768, 1, &chosen, why, sizeof(why)) != 0 &&
              strstr(why, "not an offered context (16384)"),
          "a fixed profile offers only its own settings");
}

static void test_options_parse(void)
{
    static const char *base =
        "{\"schema\":1,\"version\":2,\"models\":[{\"id\":\"m\",\"name\":\"M\","
        "\"licence\":{\"id\":\"mit\"},\"files\":[{\"role\":\"weights\",\"hf_repo\":"
        "\"TeamIDE/r\",\"revision\":\"0123456789abcdef0123456789abcdef01234567\","
        "\"filename\":\"m.gguf\",\"size\":10,\"sha256\":"
        "\"0000000000000000000000000000000000000000000000000000000000000000\"}],"
        "\"profiles\":[{\"id\":\"p\",\"hardware\":{\"unit_type\":\"p150\",\"count\":1},"
        "\"load\":{\"ctx_cap\":%d,\"max_sessions\":1},\"options\":{\"kv_type\":\"%s\","
        "\"ctx_min\":8192,\"ctx_max\":131072,\"sessions_max\":1,\"mem\":{\"fixed\":100,"
        "\"kv_per_token\":2}},\"measured\":{\"decode_tok_s\":{\"8k\":1},\"ttft\":"
        "{\"prompt_tokens\":1,\"seconds\":1},\"context_window\":1}}]}]}";
    pn_catalog_manifest *m = calloc(1, sizeof(*m));
    char json[2048], err[256];

    printf("options_parse:\n");
    snprintf(json, sizeof(json), base, 32768, "bf16");
    CHECK(pn_cat_manifest_parse(json, m, err, sizeof(err)) == 0 &&
              m->models[0].profiles[0].options.present &&
              m->models[0].profiles[0].options.mem_kv_per_token == 2,
          "a valid options block parses");
    snprintf(json, sizeof(json), base, 32768, "q4");
    CHECK(PN_LOCAL_KV_FP32_ONLY == 0, "local_kv_policy.h: the profile picks its KV (David)");
    CHECK(pn_cat_manifest_parse(json, m, err, sizeof(err)) == 0 &&
              strcmp(m->models[0].profiles[0].options.kv_type, "q4") == 0,
          "Q4 KV is accepted");
    snprintf(json, sizeof(json), base, 32768, "fp32");
    CHECK(pn_cat_manifest_parse(json, m, err, sizeof(err)) == 0 &&
              strcmp(m->models[0].profiles[0].options.kv_type, "fp32") == 0,
          "fp32 KV is accepted");
    {
        char hw[2400];
        const char *two = strstr(base, "\"hardware\":{\"unit_type\":\"p150\",\"count\":1}");
        CHECK(two != NULL, "fixture has the p150 hardware");
        snprintf(hw, sizeof(hw), "%.*s\"hardware\":{\"unit_type\":\"k80-gpu\",\"count\":4,"
                 "\"per_card\":2}%s", (int)(two - base), base,
                 two + strlen("\"hardware\":{\"unit_type\":\"p150\",\"count\":1}"));
        snprintf(json, sizeof(json), hw, 32768, "q4");
        CHECK(pn_cat_manifest_parse(json, m, err, sizeof(err)) == 0 &&
                  m->models[0].profiles[0].hardware.per_card == 2, "per_card parses");
        char *pc = strstr(hw, "\"per_card\":2");
        if (pc)
            pc[11] = '3';
        snprintf(json, sizeof(json), hw, 32768, "q4");
        CHECK(pn_cat_manifest_parse(json, m, err, sizeof(err)) != 0 && strstr(err, "per_card"),
              "per_card must divide count");
    }
    snprintf(json, sizeof(json), base, 32768, "q8");
    CHECK(pn_cat_manifest_parse(json, m, err, sizeof(err)) != 0, "an unknown KV type is refused");
    snprintf(json, sizeof(json), base, 4096, "fp32");
    CHECK(pn_cat_manifest_parse(json, m, err, sizeof(err)) != 0 && strstr(err, "inside"),
          "a default outside the ranges is refused");
    free(m);
}

/* One measured peak (tests/local/mem_model_measure.sh, heaviest unit). */
typedef struct {
    int      ctx, users;
    uint64_t peak;
} measured_peak;

/* K80 Qwen3.6 Q3_K_M pair, dies 0,1, fp32 KV, plain decode (the profile's mtp is off:
 * with fp32 KV the MTP verify is slower than plain decode): die 1 (layers 20-39 +
 * head) is the heavier unit. 2026-10-28, a K80 host (build/memmodel3/k80_q3.csv);
 * 16k x 1 also with a 14,494-token prompt (same peak). */
static const measured_peak k80_q3_pair[] = {
    { 8192, 1, 11172577280ull },
    { 16384, 1, 11508121600ull },
    { 8192, 2, 11377049600ull },
    { 16384, 2, 11880366080ull },
    { 32768, 1, 12179210240ull },
};

/* The same pair with Q4 KV and MTP on (the v1 serving): die 1, 2026-10-28
 * (build/memmodel_q4/k80_q3.csv); 40k x 1 also with a 37,672-token prompt (lower peak). */
static const measured_peak k80_q3_pair_q4[] = {
    { 8192, 1, 12132024320ull },
    { 40960, 1, 12339642368ull },
    { 65536, 1, 12495880192ull },
    { 131072, 1, 12797870080ull },
    { 40960, 2, 12462325760ull },
    { 65536, 2, 12688818176ull },
};

/* The catalog's q4 model over-predicts 128k by ~110 MB (the ring stops growing linearly
 * there), which only makes it refuse 128k, which ran with 1 MB to spare. */

/* Qwen3.6 Q8_0 on two K80 boards (dies 2-5 of a K80 host, 10/10/10/10 + head), Q4 KV,
 * MTP on: die 5 (layers 30-39 + LM head + MTP head) is the heaviest unit. 2026-10-28,
 * build/memmodel_q8/k80_q8.csv; 40k x 1 with a 23,184-token prompt. */
static const measured_peak k80_q8_two_boards[] = {
    { 8192, 1, 11396972544ull },
    { 32768, 1, 11512315904ull },
    { 40960, 1, 11529093120ull },
    { 65536, 1, 11628707840ull },
    { 131072, 1, 11892948992ull },
    { 8192, 2, 11429478400ull },
    { 16384, 2, 11476664320ull },
    { 32768, 2, 11571036160ull },
};

/* Qwen3.8 27B Q8_0 on two K80 boards (dies 8-11 of a K80 host as 0-3, 16/16/16/16 + head),
 * Q4 KV, MTP on, foundry G4 layout: die 3 (layers 48-63 + LM head + MTP head) is the heaviest
 * unit. 2026-11-18, mem_model_measure.sh (200-word prompts); 40k x 1 is the serve with an
 * 8,404-token prompt, 18 MiB over the short-prompt line, which the catalog's fixed covers
 * (+24 MiB). Exactly linear: 1,664 B a token of context + 8,704 B a token per user. */
static const measured_peak k80_q38_two_boards[] = {
    { 8192, 1, 9461301248ull },
    { 32768, 1, 9716105216ull },
    { 40960, 1, 9819914240ull },
    { 65536, 1, 10055843840ull },
    { 131072, 1, 10735321088ull },
    { 32768, 2, 10001317888ull },
};

/* p150 Ornith 1.5 9B, bf16 KV on the card, MTP head armed: the card allocator's
 * "DRAM after arm" (build/memmodel/p150_ornith.csv, p150, 2026-10-28). */
static const measured_peak p150_ornith[] = {
    { 8192, 1, 12241730535ull },
    { 32768, 1, 13146894893ull },
    { 65536, 1, 14354854445ull },
    { 131072, 1, 16770773549ull },
};

/* Ornith 1.5 9B Q8_0 on one K80 GPU (die 6 of a K80 host), Q4 KV, MTP on, 2026-10-29
 * (build/memmodel_orn/q4.csv). 32k x 1 also with a 31,147-token prompt (same peak). The
 * 2-user points peak ~42 MiB under the model: per_session is 0 and the fit is exact for one
 * user, so two users are over-predicted rather than under. 256k x 1 is refused at load. */
static const measured_peak k80_ornith_one_q4[] = {
    { 8192, 1, 10055843840ull },
    { 32768, 1, 10537140224ull },
    { 131072, 1, 12462325760ull },
    { 8192, 2, 10087301120ull },
    { 32768, 2, 10795089920ull },
};

/* The same GPU with fp32 KV, plain decode (build/memmodel_orn/fp32.csv). 32k x 1 is refused
 * at load (the die cannot hold it). */
static const measured_peak k80_ornith_one_fp32[] = {
    { 8192, 1, 10418651136ull },
    { 16384, 1, 11492392960ull },
    { 8192, 2, 11014242304ull },
};

static void check_profile_vs_measured(const pn_cat_profile *p, const measured_peak *pts, int n,
                                      uint64_t device_total, uint64_t tolerance, const char *label)
{
    pn_load_option o[PN_LOAD_OPTIONS_MAX];
    pn_hw_report hw;
    char msg[160];

    CHECK(p->options.present, label);
    for (int i = 0; i < n; i++) {
        uint64_t need = pn_load_option_need(&p->options, pts[i].ctx, pts[i].users);
        snprintf(msg, sizeof(msg), "%s: %d tokens x %d predicted %llu >= measured %llu, within "
                 "%llu MiB", label, pts[i].ctx, pts[i].users, (unsigned long long)need,
                 (unsigned long long)pts[i].peak, (unsigned long long)(tolerance >> 20));
        CHECK(need >= pts[i].peak && need - pts[i].peak <= tolerance, msg);
    }
    int cards = p->hardware.per_card > 0 ? p->hardware.count / p->hardware.per_card : 1;
    make_box(&hw, p->hardware.unit_type, cards, device_total, NULL, 0);
    int k = pn_load_options_fit(p, &hw, o, PN_LOAD_OPTIONS_MAX), offered = 0;
    for (int i = 0; i < k; i++) {
        if (!o[i].fits)
            continue;
        offered++;
        snprintf(msg, sizeof(msg), "%s: offered %d x %d keeps the margin", label, o[i].ctx_cap,
                 o[i].max_sessions);
        CHECK(o[i].need_per_unit + PN_LOAD_FIT_MARGIN_BYTES <= device_total, msg);
    }
    snprintf(msg, sizeof(msg), "%s: an idle card offers at least one choice", label);
    CHECK(offered > 0, msg);
}

/* Qwen3.8 on an idle p150 (31.875 GiB): both profiles offer 8k / 16k / 30,720 x 1 and
 * recommend 30,720 (the card's own fit policy is the real gate: test_tt_q38_fit). */
static void check_qwen38_p150(const pn_catalog_manifest *m)
{
    const pn_cat_model *q = pn_cat_find_model(m, "qwen3.8-27b-q8");
    const char *ids[] = { "p150-one", "p150-one-bf16" };
    char what[160];

    for (int p = 0; p < 2; p++) {
        const pn_cat_profile *prof = q ? pn_cat_find_profile(q, ids[p]) : NULL;
        pn_load_option o[PN_LOAD_OPTIONS_MAX];
        pn_hw_report hw;
        bool r30 = false, f8 = false, f16 = false;

        snprintf(what, sizeof(what), "qwen3.8 %s: 8k, 16k and 30,720 x 1 fit an idle p150, "
                 "30,720 recommended", ids[p]);
        if (!prof) {
            CHECK(0, what);
            continue;
        }
        make_box(&hw, "p150", 1, 34225520640ull, NULL, 0);
        int k = pn_load_options_fit(prof, &hw, o, PN_LOAD_OPTIONS_MAX);
        for (int i = 0; i < k; i++) {
            r30 |= o[i].ctx_cap == 30720 && o[i].max_sessions == 1 && o[i].fits && o[i].recommended;
            f8 |= o[i].ctx_cap == 8192 && o[i].fits;
            f16 |= o[i].ctx_cap == 16384 && o[i].fits;
        }
        CHECK(r30 && f8 && f16, what);
    }
}

static void test_catalog_vs_measured(void)
{
    pn_catalog_manifest *m = calloc(1, sizeof(*m));
    char *text = NULL, err[256];

    printf("catalog_vs_measured:\n");
    if (pn_local_read_file("catalog/catalog.json", 4u << 20, &text, NULL) != 0 ||
        pn_cat_manifest_parse(text, m, err, sizeof(err)) != 0) {
        CHECK(0, "catalog/catalog.json parses");
        free(text);
        free(m);
        return;
    }
    const pn_cat_model *q = pn_cat_find_model(m, "qwen3.6-35b-a3b-q3km");
    const pn_cat_model *o = pn_cat_find_model(m, "ornith-1.5-9b-q8");
    const pn_cat_profile *qp = q ? pn_cat_find_profile(q, "k80-pair") : NULL;
    const pn_cat_profile *qf = q ? pn_cat_find_profile(q, "k80-pair-fp32") : NULL;
    const pn_cat_profile *op = o ? pn_cat_find_profile(o, "p150-one") : NULL;
    CHECK(qp && qf && op, "the catalog profiles are there");
    if (qp && qf) {
        /* David (spec §14): Q4 KV + MTP is the K80 default, fp32 + plain decode the
         * alternative, ranked after it on the same hardware. */
        CHECK(qp->options.present && strcmp(qp->options.kv_type, "q4") == 0 && qp->mtp &&
                  qp->rank < qf->rank,
              "k80-pair: Q4 KV + MTP, ranked first");
        CHECK(qf->options.present && strcmp(qf->options.kv_type, "fp32") == 0 && !qf->mtp &&
                  strcmp(qf->hardware.unit_type, qp->hardware.unit_type) == 0 &&
                  qf->hardware.count == qp->hardware.count &&
                  qf->hardware.same_card == qp->hardware.same_card,
              "k80-pair-fp32: fp32 KV, plain decode, the same hardware");
        check_profile_vs_measured(qf, k80_q3_pair, 5, 12799180800ull, MODEL_TOLERANCE,
                                  "k80-pair-fp32");
        check_profile_vs_measured(qp, k80_q3_pair_q4, 6, 12799180800ull, MIB(128),
                                  "k80-pair q4");
        pn_load_option o[PN_LOAD_OPTIONS_MAX];
        pn_hw_report hw;
        make_box(&hw, "k80-gpu", 1, 12799180800ull, NULL, 0);
        int k = pn_load_options_fit(qp, &hw, o, PN_LOAD_OPTIONS_MAX);
        bool r40 = false, f128 = false;
        for (int i = 0; i < k; i++) {
            r40 |= o[i].ctx_cap == 40960 && o[i].max_sessions == 1 && o[i].fits && o[i].recommended;
            f128 |= o[i].ctx_cap == 131072 && o[i].fits;
        }
        CHECK(r40 && !f128, "k80-pair q4: 40k x 1 (v1) fits and is recommended; 128k is not offered");
        k = pn_load_options_fit(qf, &hw, o, PN_LOAD_OPTIONS_MAX);
        bool r32 = false, f32x2 = false;
        for (int i = 0; i < k; i++) {
            r32 |= o[i].ctx_cap == 32768 && o[i].max_sessions == 1 && o[i].fits && o[i].recommended;
            f32x2 |= o[i].ctx_cap == 32768 && o[i].max_sessions == 2 && o[i].fits;
        }
        CHECK(r32 && !f32x2, "k80-pair-fp32: 32k x 1 fits and is recommended; 32k x 2 is not offered");
    }
    const pn_cat_model *q8 = pn_cat_find_model(m, "qwen3.6-35b-a3b-q8");
    const pn_cat_profile *q8p = q8 ? pn_cat_find_profile(q8, "k80-two-boards") : NULL;
    CHECK(q8p && q8p->hardware.count == 4 && q8p->hardware.per_card == 2 &&
              !q8p->hardware.same_card && q8p->mtp && strcmp(q8p->options.kv_type, "q4") == 0,
          "qwen3.6 Q8_0: 4 k80-gpu as two whole boards, Q4 KV + MTP");
    if (q8p)
        check_profile_vs_measured(q8p, k80_q8_two_boards, 8, 12799180800ull, MODEL_TOLERANCE,
                                  "k80-two-boards q8");
    const pn_cat_model *q38 = pn_cat_find_model(m, "qwen3.8-27b-q8");
    const pn_cat_profile *q38k = q38 ? pn_cat_find_profile(q38, "k80-two-boards") : NULL;
    CHECK(q38k && q38k->hardware.count == 4 && q38k->hardware.per_card == 2 &&
              !q38k->hardware.same_card && q38k->mtp && q38k->mtp_drafts == 0 &&
              q38k->max_sessions == 1 && q38k->options.sessions_max == 1 &&
              strcmp(q38k->options.kv_type, "q4") == 0,
          "qwen3.8 Q8_0: 4 k80-gpu as two whole boards, one user, Q4 KV + MTP (adaptive depth)");
    if (q38k) {
        check_profile_vs_measured(q38k, k80_q38_two_boards, 6, 12799180800ull, MODEL_TOLERANCE,
                                  "qwen3.8 k80-two-boards");
        pn_load_option o4[PN_LOAD_OPTIONS_MAX];
        pn_hw_report hw4;
        make_box(&hw4, "k80-gpu", 2, 12799180800ull, NULL, 0);
        int k4 = pn_load_options_fit(q38k, &hw4, o4, PN_LOAD_OPTIONS_MAX);
        bool r32 = false, f128 = false;
        for (int i = 0; i < k4; i++) {
            r32 |= o4[i].ctx_cap == 32768 && o4[i].max_sessions == 1 && o4[i].fits &&
                   o4[i].recommended;
            f128 |= o4[i].ctx_cap == 131072 && o4[i].fits;
        }
        CHECK(r32 && f128, "qwen3.8 k80-two-boards: 32k x 1 recommended, 128k x 1 offered");
    }
    CHECK(m->version == 8, "catalog version 8 (Qwen3.8 27B p150-one-100k)");
    const pn_cat_profile *ok80 = o ? pn_cat_find_profile(o, "k80-one") : NULL;
    CHECK(ok80 && ok80->hardware.count == 1 && strcmp(ok80->hardware.unit_type, "k80-gpu") == 0 &&
              ok80->mtp && strcmp(ok80->options.kv_type, "q4") == 0,
          "ornith k80-one: one k80-gpu, Q4 KV + MTP");
    if (ok80) {
        check_profile_vs_measured(ok80, k80_ornith_one_q4, 5, 12799180800ull, MIB(48),
                                  "ornith k80-one q4");
        pn_load_option o1[PN_LOAD_OPTIONS_MAX];
        pn_hw_report hw1;
        make_box(&hw1, "k80-gpu", 1, 12799180800ull, NULL, 0);
        int k1 = pn_load_options_fit(ok80, &hw1, o1, PN_LOAD_OPTIONS_MAX);
        bool r32 = false, f128 = false;
        for (int i = 0; i < k1; i++) {
            r32 |= o1[i].ctx_cap == 32768 && o1[i].max_sessions == 1 && o1[i].fits &&
                   o1[i].recommended;
            f128 |= o1[i].ctx_cap == 131072 && o1[i].max_sessions == 1 && o1[i].fits;
        }
        CHECK(r32 && f128, "ornith k80-one: 32k x 1 recommended, 128k x 1 offered");
    }
    const pn_cat_profile *of32 = o ? pn_cat_find_profile(o, "k80-one-fp32") : NULL;
    CHECK(of32 && ok80 && !of32->mtp && strcmp(of32->options.kv_type, "fp32") == 0 &&
              of32->rank > ok80->rank,
          "ornith k80-one-fp32: fp32 KV, plain decode, ranked after Q4");
    if (of32)
        check_profile_vs_measured(of32, k80_ornith_one_fp32, 3, 12799180800ull,
                                  MODEL_TOLERANCE, "ornith k80-one-fp32");
    if (op)
        check_profile_vs_measured(op, p150_ornith, 4, 34225520640ull, MODEL_TOLERANCE,
                                  "p150-one");
    check_qwen38_p150(m);
    free(text);
    free(m);
}

/* Rung 4a and the load's KV choice (local_kv_policy.h at 0): a default-format cache
 * still goes Q4 under pressure; an explicit fp32 cache is never switched, it is
 * refused with a reason; PN_LADDER_KV_Q4=0 turns the rung off for both. */
static void test_ladder_kv_choice(void)
{
    printf("ladder_kv_choice:\n");
    unsetenv("PN_LADDER_KV_Q4");
    unsetenv("PN_LADDER_DISABLE");
    CHECK(pn_ladder_kv_action_for(PN_MEM_LEVEL_OK, false) == PN_LADDER_KV_KEEP &&
              pn_ladder_kv_action_for(PN_MEM_LEVEL_OK, true) == PN_LADDER_KV_KEEP,
          "OK: every cache keeps its format");
    CHECK(pn_ladder_kv_action_for(PN_MEM_LEVEL_WATCH, false) == PN_LADDER_KV_TO_Q4,
          "WATCH: a default-format cache goes Q4 (unchanged)");
    CHECK(pn_ladder_kv_action_for(PN_MEM_LEVEL_WATCH, true) == PN_LADDER_KV_REFUSE &&
              pn_ladder_kv_action_for(PN_MEM_LEVEL_CRITICAL, true) == PN_LADDER_KV_REFUSE,
          "WATCH/CRITICAL: an explicit fp32 cache is refused, never switched to Q4");
    setenv("PN_LADDER_KV_Q4", "0", 1);
    CHECK(pn_ladder_kv_action_for(PN_MEM_LEVEL_CRITICAL, true) == PN_LADDER_KV_KEEP &&
              pn_ladder_kv_action_for(PN_MEM_LEVEL_CRITICAL, false) == PN_LADDER_KV_KEEP,
          "PN_LADDER_KV_Q4=0: the rung is off");
    unsetenv("PN_LADDER_KV_Q4");
    pn_ladder_kv_refusal_clear();
    CHECK(pn_ladder_kv_refusal() == NULL, "no refusal after a clear");
    pn_ladder_kv_refusal_set("host memory pressure (WATCH) would switch");
    CHECK(pn_ladder_kv_refusal() && strstr(pn_ladder_kv_refusal(), "WATCH"),
          "the refusal reason reaches the loader");
    pn_ladder_kv_refusal_clear();
}

/* The box maps a profile's kv_type to the load's format: explicit q4 / fp32, else the
 * serve default (bf16 on the card, no options). */
static void test_kv_format_of(void)
{
    printf("kv_format_of:\n");
    CHECK(pn_local_kv_format_of("q4") == PN_KV_FORMAT_Q4, "q4 -> Q4");
    CHECK(pn_local_kv_format_of("fp32") == PN_KV_FORMAT_FP32, "fp32 -> fp32");
    CHECK(pn_local_kv_format_of("bf16") == PN_KV_FORMAT_DEFAULT &&
              pn_local_kv_format_of("") == PN_KV_FORMAT_DEFAULT,
          "bf16 / none -> the serve default");
}

/* A p150 profile's PN_TT_KV_TYPE / PN_TT_SPEC for its load (local_load.c): set only for a
 * block-float KV, MTP off, or a draft count; never off a p150. In the shipped catalog only
 * Qwen3.8 sets anything (K8V4 + 1 draft; its bf16 profile MTP off): every other profile
 * loads exactly as before. */
static void test_tt_load_env(void)
{
    pn_catalog_manifest *m = calloc(1, sizeof(*m));
    char *text = NULL, err[256], spec[8] = "x", what[200];

    printf("tt_load_env:\n");
    CHECK(pn_local_tt_kv_type_of("p150", "k8v4") &&
              strcmp(pn_local_tt_kv_type_of("p150", "k8v4"), "k8v4") == 0 &&
              pn_local_tt_kv_type_of("p150", "bfp4"),
          "p150 k8v4 / bfp4 -> PN_TT_KV_TYPE");
    CHECK(!pn_local_tt_kv_type_of("p150", "bf16") && !pn_local_tt_kv_type_of("p150", "") &&
              !pn_local_tt_kv_type_of("p150", NULL) && !pn_local_tt_kv_type_of("p150", "q4") &&
              !pn_local_tt_kv_type_of("p150", "fp32") && !pn_local_tt_kv_type_of("p150", "k8v8") &&
              !pn_local_tt_kv_type_of("k80-gpu", "k8v4") && !pn_local_tt_kv_type_of("cpu", "bfp8"),
          "bf16 / none / q4 / fp32 / unknown / off a p150 -> the process's");
    CHECK(pn_local_tt_spec_of("p150", true, 1, spec, sizeof(spec)) == spec &&
              strcmp(spec, "1") == 0, "p150 mtp + 1 draft -> PN_TT_SPEC=1");
    CHECK(pn_local_tt_spec_of("p150", false, 0, spec, sizeof(spec)) == spec &&
              strcmp(spec, "0") == 0, "p150 mtp false -> PN_TT_SPEC=0 (no head)");
    strcpy(spec, "x");
    CHECK(!pn_local_tt_spec_of("p150", true, 0, spec, sizeof(spec)) &&
              !pn_local_tt_spec_of("k80-gpu", false, 0, spec, sizeof(spec)) &&
              !pn_local_tt_spec_of("k80-gpu", true, 2, spec, sizeof(spec)) &&
              !pn_local_tt_spec_of("cpu", false, 0, spec, sizeof(spec)) &&
              !pn_local_tt_spec_of(NULL, false, 0, spec, sizeof(spec)) && strcmp(spec, "x") == 0,
          "mtp true without drafts / off a p150 -> the process's (out untouched)");

    if (pn_local_read_file("catalog/catalog.json", 4u << 20, &text, NULL) != 0 ||
        pn_cat_manifest_parse(text, m, err, sizeof(err)) != 0) {
        CHECK(0, "catalog/catalog.json parses");
        free(text);
        free(m);
        return;
    }
    for (int i = 0; i < m->n_models; i++) {
        for (int j = 0; j < m->models[i].n_profiles; j++) {
            const pn_cat_profile *p = &m->models[i].profiles[j];
            const char *kv = pn_local_tt_kv_type_of(p->hardware.unit_type, p->options.kv_type);
            const char *sp = pn_local_tt_spec_of(p->hardware.unit_type, p->mtp, p->mtp_drafts,
                                                 spec, sizeof(spec));
            const bool q38 = strcmp(m->models[i].id, "qwen3.8-27b-q8") == 0;
            const bool k8v4 = q38 && (strcmp(p->id, "p150-one") == 0 ||
                                      strcmp(p->id, "p150-one-100k") == 0);
            const bool bf16 = q38 && strcmp(p->id, "p150-one-bf16") == 0;

            snprintf(what, sizeof(what), "%s %s: PN_TT_KV_TYPE %s, PN_TT_SPEC %s",
                     m->models[i].id, p->id, kv ? kv : "(process)", sp ? sp : "(process)");
            CHECK(k8v4 ? kv && strcmp(kv, "k8v4") == 0 && sp && strcmp(sp, "1") == 0
                  : bf16 ? !kv && sp && strcmp(sp, "0") == 0
                  : !kv && !sp, what);
        }
    }
    free(text);
    free(m);
}

int main(void)
{
    printf("test_local_load_options\n");
    test_ladder_kv_choice();
    test_kv_format_of();
    test_tt_load_env();
    test_grid_and_need();
    test_fit_on_free_units();
    test_choose_and_refuse();
    test_options_parse();
    test_catalog_vs_measured();
    return lt_report("test_local_load_options");
}
