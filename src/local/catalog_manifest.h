/* SPDX-License-Identifier: AGPL-3.0-or-later */
/* Copyright (C) 2026 David (Corey) Herrera */
/**
 * @file   catalog_manifest.h
 * @brief  The catalog manifest (spec §5, docs/local/catalog_schema.md): parse and
 *         strictly validate catalog.json into fixed structs. Every field that is
 *         later used to build a download URL or a file name is checked against
 *         a narrow character set here, so the downloader only ever sees
 *         validated values.
 *
 * @status None
 * @issues None
 * @todo   None
 */

#ifndef PN_LOCAL_CATALOG_MANIFEST_H
#define PN_LOCAL_CATALOG_MANIFEST_H

#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>

#include "../serve/sampling_defaults.h"
#include "unit_place.h"

#ifdef __cplusplus
extern "C" {
#endif

#define PN_CAT_SCHEMA        1
#define PN_CAT_MAX_MODELS    32
#define PN_CAT_MAX_FILES     4
#define PN_CAT_MAX_PROFILES  8
#define PN_CAT_MAX_CAPS      8
#define PN_CAT_ID_MAX        64
#define PN_CAT_TEXT_MAX      256

/* Unit types as bits (files[].unit_types, a profile's hardware.unit_type). */
#define PN_CAT_UNIT_K80_GPU  1u
#define PN_CAT_UNIT_P150     2u
#define PN_CAT_UNIT_CPU      4u
#define PN_CAT_UNITS_ALL     (PN_CAT_UNIT_K80_GPU | PN_CAT_UNIT_P150 | PN_CAT_UNIT_CPU)

/* Optional compressed download of a file (files[].ingot): a foundry .ingot that decodes to the
 * exact file (its sha256 and size stay the file's identity). A delta .ingot names the file it
 * was coded against (base_sha256); it decodes only when that file is installed. */
typedef struct {
    bool     present;
    char     hf_repo[128];             /* default: the file's */
    char     revision[41];             /* default: the file's */
    char     filename[128];            /* [A-Za-z0-9._-]+ */
    uint64_t size;                     /* bytes of the .ingot */
    char     base_sha256[65];          /* 64 lowercase hex, or "" (not a delta) */
} pn_cat_ingot;

typedef struct {
    char     role[16];                 /* "weights" (exactly one per model) or "extra" */
    char     hf_repo[128];             /* "<org>/<repo>", org in PN_CATALOG_HF_ORGS */
    char     revision[41];             /* pinned 40-hex commit */
    char     filename[128];            /* [A-Za-z0-9._-]+ */
    uint64_t size;
    char     sha256[65];               /* 64 lowercase hex */
    unsigned unit_types;               /* PN_CAT_UNIT_* bits that need it; 0 = every unit type */
    pn_cat_ingot ingot;                /* optional .ingot download */
} pn_cat_file;

typedef struct {
    double decode_tok_s_8k, decode_tok_s_14k, decode_tok_s_32k;   /* 0 = not measured */
    int    ttft_prompt_tokens;
    double ttft_s;
    int    context_window;
} pn_cat_measured;

/* Smart load options (Phase L): the device-memory model of a profile and the
 * ranges the user may pick from. Bytes are per unit (the heaviest unit of the
 * set) as the device reports them (nvidia-smi / the card allocator):
 *   need(ctx, sessions) = mem_fixed + ctx * mem_ctx_per_token
 *                         + sessions * (mem_per_session + ctx * mem_kv_per_token)
 * Absent (`present` false): the profile loads with its fixed `load` settings. */
typedef struct {
    bool     present;
    char     kv_type[8];              /* "fp32", "bf16", "q4" when local_kv_policy.h allows, or
                                       * on a p150 a PN_TT_KV_TYPE value (pn_cat_kv_type_allowed) */
    char     kv_long[8];              /* p150 only, optional: the PN_TT_KV_TYPE the TT fit may
                                       * switch to when the chosen ctx needs it ("k4h");
                                       * "" = kv_type at every ctx */
    int      ctx_min, ctx_max;        /* tokens; the picker offers round steps between */
    int      sessions_max;            /* users at once, >= 1 */
    uint64_t mem_fixed;               /* weights + runtime, per unit */
    uint64_t mem_per_session;         /* per session, independent of ctx */
    uint64_t mem_kv_per_token;        /* per token per session, per unit */
    uint64_t mem_ctx_per_token;       /* per token once per load (ctx-sized scratch) */
} pn_cat_load_options;

typedef struct {
    char              id[PN_CAT_ID_MAX];
    int               rank;             /* lower = better experience, shown first */
    pn_unit_request   hardware;         /* unit type, count, same card, min mem/unit */
    uint64_t          min_host_ram;
    int               ctx_cap, max_sessions, max_concurrent, reasoning_budget;
    int               reasoning_reserve;   /* load.reasoning_reserve; 0 = serve default */
    bool              mtp;
    int               mtp_drafts;       /* load.mtp_drafts: p150 MTP drafts a step (PN_TT_SPEC);
                                         * 0 = the build's default */
    pn_model_sampling sampling;
    pn_cat_measured   measured;
    pn_cat_load_options options;        /* the load's ctx_cap/max_sessions = recommended */
} pn_cat_profile;

typedef struct {
    char           id[PN_CAT_ID_MAX];
    char           name[128];
    char           family[PN_CAT_ID_MAX];
    char           description[PN_CAT_TEXT_MAX];
    char           licence_id[PN_CAT_ID_MAX];
    char           licence_url[PN_CAT_TEXT_MAX];
    char           capabilities[PN_CAT_MAX_CAPS][16];
    int            n_capabilities;
    pn_cat_file    files[PN_CAT_MAX_FILES];
    int            n_files;
    pn_cat_profile profiles[PN_CAT_MAX_PROFILES];
    int            n_profiles;
} pn_cat_model;

typedef struct {
    int          schema;
    long long    version;               /* monotonically increasing (anti-rollback) */
    char         published[32];
    pn_cat_model models[PN_CAT_MAX_MODELS];
    int          n_models;
} pn_catalog_manifest;

/**
 * Parse + validate. On failure `err` names the first bad field.
 * @return 0, or -1
 *
 * Thread-safe: yes. Reentrant: yes. Signal-safe: no
 */
int pn_cat_manifest_parse(const char *json, pn_catalog_manifest *out, char *err, size_t errlen);

/** Model by id, or NULL. Thread-safe: yes. Reentrant: yes. Signal-safe: yes */
const pn_cat_model *pn_cat_find_model(const pn_catalog_manifest *m, const char *id);

/** Profile by id, or NULL. Thread-safe: yes. Reentrant: yes. Signal-safe: yes */
const pn_cat_profile *pn_cat_find_profile(const pn_cat_model *model, const char *id);

/**
 * The model's (first) weights file. Call it on a pn_cat_model_for_units copy for one
 * unit type: a model may have one weights file per unit type (each tagged), and the
 * parser checks every profile's unit type needs exactly one.
 *
 * Thread-safe: yes. Reentrant: yes. Signal-safe: yes
 */
const pn_cat_file *pn_cat_weights_file(const pn_cat_model *model);

/**
 * The PN_CAT_UNIT_* bit of a unit type name ("k80-gpu", "p150", "cpu").
 * @return the bit, or 0 for an unknown name
 *
 * Thread-safe: yes. Reentrant: yes. Signal-safe: yes
 */
unsigned pn_cat_unit_type_bit(const char *unit_type);

/**
 * Whether a profile on `unit_type` may name `kv_type` in options.kv_type: "fp32" and
 * "bf16" anywhere; unless local_kv_policy.h pins fp32, also "q4" anywhere and, on a p150,
 * the other PN_TT_KV_TYPE values (bfp8, bfp4, k8v4, k16v4, k4v16, k16v8; pn_tt_kv_format.h).
 *
 * Thread-safe: yes. Reentrant: yes. Signal-safe: yes
 */
bool pn_cat_kv_type_allowed(const char *unit_type, const char *kv_type);

/**
 * Whether hardware with these unit types needs the file: a file without
 * `unit_types` is needed by every unit type.
 *
 * Thread-safe: yes. Reentrant: yes. Signal-safe: yes
 */
bool pn_cat_file_needed(const pn_cat_file *f, unsigned units);

/**
 * Copy of `model` holding only the files `units` need (spec §16): what an
 * install downloads and what "installed" checks. PN_CAT_UNITS_ALL keeps
 * every file. For one profile's unit type the copy holds exactly one weights file.
 *
 * Thread-safe: yes. Reentrant: yes. Signal-safe: yes
 */
void pn_cat_model_for_units(const pn_cat_model *model, unsigned units, pn_cat_model *out);

/**
 * Download URL: <base>/<hf_repo>/resolve/<revision>/<filename>, base defaulting
 * to https://huggingface.co.
 * @return 0, or -1 on truncation
 *
 * Thread-safe: yes. Reentrant: yes. Signal-safe: yes
 */
int pn_cat_file_url(const char *base, const pn_cat_file *f, char *out, size_t cap);

/**
 * Download URL of the file's .ingot (f->ingot.present):
 * <base>/<ingot.hf_repo>/resolve/<ingot.revision>/<ingot.filename>.
 * @return 0, or -1 when the file has no .ingot or on truncation
 *
 * Thread-safe: yes. Reentrant: yes. Signal-safe: yes
 */
int pn_cat_ingot_url(const char *base, const pn_cat_file *f, char *out, size_t cap);

#ifdef __cplusplus
}
#endif

#endif /* PN_LOCAL_CATALOG_MANIFEST_H */
