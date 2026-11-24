/* SPDX-License-Identifier: AGPL-3.0-or-later */
/* Copyright (C) 2026 David (Corey) Herrera */
/**
 * @file   sampling_defaults.h
 * @brief  Per-model sampling defaults (temperature, top_p, top_k, min_p,
 *         presence_penalty) and the rule that resolves a request against them:
 *         a field the request carries wins, else the model's default, else the
 *         serve fallback (0.7 / 0.95 / 40 / 0 / 0).
 *
 * @status Used by openai_chat.c (resolve), model_admin.c (load body) and the
 *         Pantheon Local profile loader.
 * @issues min_p has no sampler support yet: a non-zero min_p default is
 *         refused at load (pn_model_sampling_validate) rather than ignored.
 * @todo   None
 */

#ifndef PN_SERVE_SAMPLING_DEFAULTS_H
#define PN_SERVE_SAMPLING_DEFAULTS_H

#include <stdbool.h>
#include <stddef.h>

#include "json_reader.h"
#include "serve_generation.h"

#ifdef __cplusplus
extern "C" {
#endif

#define PN_SAMPLING_FALLBACK_TEMPERATURE 0.7f
#define PN_SAMPLING_FALLBACK_TOP_P       0.95f
#define PN_SAMPLING_FALLBACK_TOP_K       40

/* A set of sampling fields, each with a presence flag. Used both for a
 * model's defaults and for what one request carried. Zero-init = all absent. */
typedef struct {
    bool  has_temperature;
    float temperature;
    bool  has_top_p;
    float top_p;
    bool  has_top_k;
    int   top_k;
    bool  has_min_p;
    float min_p;
    bool  has_presence_penalty;
    float presence_penalty;
} pn_model_sampling;

/**
 * Read the five fields from a JSON object (a chat request or an admin load
 * body). A field is present only when it is a JSON number; anything else is
 * absent. No range checks here (see pn_model_sampling_validate).
 *
 * @param obj  JSON object, may be NULL (everything absent)
 * @param out  zeroed then filled
 *
 * Thread-safe: yes
 * Reentrant:   yes
 * Signal-safe: no
 */
void pn_model_sampling_from_json(const jr_value *obj, pn_model_sampling *out);

/**
 * Range-check a model's defaults: temperature [0,5], top_p (0,1], top_k
 * [0,1000], min_p 0 only (no sampler support), presence_penalty [-2,2].
 *
 * @return 0 if valid, -1 with a short reason in err[errlen]
 *
 * Thread-safe: yes
 * Reentrant:   yes
 * Signal-safe: no
 */
int pn_model_sampling_validate(const pn_model_sampling *d, char *err, size_t errlen);

/**
 * Write the resolved temperature / top_p / top_k / presence_penalty into gen:
 * request field if present, else the model default if present, else the
 * fallback. presence_penalty is clamped to [-2,2] as the request path always
 * did. `model` may be NULL (no defaults).
 *
 * Thread-safe: yes
 * Reentrant:   yes
 * Signal-safe: no
 */
void pn_model_sampling_apply(const pn_model_sampling *request,
                       const pn_model_sampling *model, sg_gen_params *gen);

#ifdef __cplusplus
}
#endif

#endif /* PN_SERVE_SAMPLING_DEFAULTS_H */
