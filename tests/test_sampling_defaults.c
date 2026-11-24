/* SPDX-License-Identifier: AGPL-3.0-or-later */
/* Copyright (C) 2026 David (Corey) Herrera */
/*
 * test_sampling_defaults.c — Pantheon Local A4: per-model sampling defaults.
 * An absent request field takes the model's default, a present one wins, and
 * with neither the serve fallback (0.7 / 0.95 / 40 / 0) applies. Validation
 * refuses out-of-range defaults and any non-zero min_p (no sampler support).
 */

#include "local_test_support.h"

#include "../src/serve/sampling_defaults.h"

static void parse(const char *json, pn_model_sampling *out)
{
    jr_value *root = jr_parse(json);
    pn_model_sampling_from_json(root, out);
    jr_free(root);
}

static void test_absent_takes_model_default(void)
{
    pn_model_sampling model, request;
    sg_gen_params gen;

    printf("absent_takes_model_default:\n");
    parse("{\"temperature\":0.6,\"top_p\":0.8,\"top_k\":20,\"presence_penalty\":1.5}", &model);
    parse("{\"messages\":[]}", &request);
    memset(&gen, 0, sizeof(gen));
    pn_model_sampling_apply(&request, &model, &gen);
    CHECK(gen.temperature == 0.6f, "temperature from model");
    CHECK(gen.top_p == 0.8f, "top_p from model");
    CHECK(gen.top_k == 20, "top_k from model");
    CHECK(gen.presence_penalty == 1.5f, "presence_penalty from model");
}

static void test_present_field_wins(void)
{
    pn_model_sampling model, request;
    sg_gen_params gen;

    printf("present_field_wins:\n");
    parse("{\"temperature\":0.6,\"top_k\":20}", &model);
    parse("{\"temperature\":1.0,\"top_k\":5,\"presence_penalty\":9}", &request);
    memset(&gen, 0, sizeof(gen));
    pn_model_sampling_apply(&request, &model, &gen);
    CHECK(gen.temperature == 1.0f, "request temperature wins");
    CHECK(gen.top_k == 5, "request top_k wins");
    CHECK(gen.top_p == PN_SAMPLING_FALLBACK_TOP_P, "neither has top_p -> fallback");
    CHECK(gen.presence_penalty == 2.0f, "request presence_penalty clamped to 2");
}

static void test_fallback_without_defaults(void)
{
    pn_model_sampling request;
    sg_gen_params gen;

    printf("fallback_without_defaults:\n");
    parse("{\"top_k\":\"x\"}", &request);   /* non-number = absent */
    memset(&gen, 0, sizeof(gen));
    pn_model_sampling_apply(&request, NULL, &gen);
    CHECK(gen.temperature == PN_SAMPLING_FALLBACK_TEMPERATURE, "fallback temperature 0.7");
    CHECK(gen.top_p == PN_SAMPLING_FALLBACK_TOP_P, "fallback top_p 0.95");
    CHECK(gen.top_k == PN_SAMPLING_FALLBACK_TOP_K, "fallback top_k 40");
    CHECK(gen.presence_penalty == 0.0f, "fallback presence_penalty 0");
}

static void test_validate(void)
{
    pn_model_sampling d;
    char err[128];

    printf("validate:\n");
    parse("{\"temperature\":0.7,\"top_p\":0.95,\"top_k\":20,\"min_p\":0}", &d);
    CHECK(pn_model_sampling_validate(&d, err, sizeof(err)) == 0, "card values accepted");
    parse("{\"min_p\":0.05}", &d);
    CHECK(pn_model_sampling_validate(&d, err, sizeof(err)) != 0 && strstr(err, "min_p"),
          "non-zero min_p refused");
    parse("{\"top_p\":0}", &d);
    CHECK(pn_model_sampling_validate(&d, err, sizeof(err)) != 0, "top_p 0 refused");
    parse("{\"temperature\":-1}", &d);
    CHECK(pn_model_sampling_validate(&d, err, sizeof(err)) != 0, "negative temperature refused");
    parse("{\"presence_penalty\":3}", &d);
    CHECK(pn_model_sampling_validate(&d, err, sizeof(err)) != 0, "presence_penalty 3 refused");
}

int main(void)
{
    test_absent_takes_model_default();
    test_present_field_wins();
    test_fallback_without_defaults();
    test_validate();
    return lt_report("test_sampling_defaults");
}
