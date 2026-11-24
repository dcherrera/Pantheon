/* SPDX-License-Identifier: AGPL-3.0-or-later */
/* Copyright (C) 2026 David (Corey) Herrera */
/**
 * @file   sampling_defaults.c
 * @brief  Implementation of per-model sampling defaults (see sampling_defaults.h).
 *
 * @status None
 * @issues None
 * @todo   None
 */

#include "sampling_defaults.h"

#include <stdio.h>
#include <string.h>

/* Read one numeric field; returns true and writes *out when it is a number. */
static bool read_number_field(const jr_value *obj, const char *key, double *out)
{
    const jr_value *v = jr_object_get(obj, key);

    if (jr_kind(v) != JR_NUMBER)
        return false;
    *out = jr_as_number(v, 0.0);
    return true;
}

void pn_model_sampling_from_json(const jr_value *obj, pn_model_sampling *out)
{
    double v = 0.0;

    memset(out, 0, sizeof(*out));
    if (!obj || jr_kind(obj) != JR_OBJECT)
        return;
    if ((out->has_temperature = read_number_field(obj, "temperature", &v)))
        out->temperature = (float)v;
    if ((out->has_top_p = read_number_field(obj, "top_p", &v)))
        out->top_p = (float)v;
    if ((out->has_top_k = read_number_field(obj, "top_k", &v)))
        out->top_k = (int)v;
    if ((out->has_min_p = read_number_field(obj, "min_p", &v)))
        out->min_p = (float)v;
    if ((out->has_presence_penalty = read_number_field(obj, "presence_penalty", &v)))
        out->presence_penalty = (float)v;
}

int pn_model_sampling_validate(const pn_model_sampling *d, char *err, size_t errlen)
{
    const char *why = NULL;

    if (d->has_temperature && !(d->temperature >= 0.0f && d->temperature <= 5.0f))
        why = "temperature must be in [0,5]";
    else if (d->has_top_p && !(d->top_p > 0.0f && d->top_p <= 1.0f))
        why = "top_p must be in (0,1]";
    else if (d->has_top_k && (d->top_k < 0 || d->top_k > 1000))
        why = "top_k must be in [0,1000]";
    else if (d->has_min_p && d->min_p != 0.0f)
        why = "min_p is not supported by this sampler (only 0 is accepted)";
    else if (d->has_presence_penalty &&
             !(d->presence_penalty >= -2.0f && d->presence_penalty <= 2.0f))
        why = "presence_penalty must be in [-2,2]";
    if (!why)
        return 0;
    snprintf(err, errlen, "%s", why);
    return -1;
}

void pn_model_sampling_apply(const pn_model_sampling *request,
                       const pn_model_sampling *model, sg_gen_params *gen)
{
    static const pn_model_sampling none = {0};
    const pn_model_sampling *m = model ? model : &none;
    float pp = 0.0f;

    gen->temperature = request->has_temperature ? request->temperature
                     : m->has_temperature       ? m->temperature
                                                : PN_SAMPLING_FALLBACK_TEMPERATURE;
    gen->top_p = request->has_top_p ? request->top_p
               : m->has_top_p       ? m->top_p
                                    : PN_SAMPLING_FALLBACK_TOP_P;
    gen->top_k = request->has_top_k ? request->top_k
               : m->has_top_k       ? m->top_k
                                    : PN_SAMPLING_FALLBACK_TOP_K;
    pp = request->has_presence_penalty ? request->presence_penalty
       : m->has_presence_penalty       ? m->presence_penalty
                                       : 0.0f;
    gen->presence_penalty = pp < -2.0f ? -2.0f : pp > 2.0f ? 2.0f : pp;
}
