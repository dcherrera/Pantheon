/* SPDX-License-Identifier: AGPL-3.0-or-later */
/* Copyright (C) 2026 David (Corey) Herrera */
/**
 * @file   catalog_manifest.c
 * @brief  Implementation of the catalog manifest parser (see catalog_manifest.h).
 *
 * @status None
 * @issues None
 * @todo   None
 */

#include "catalog_manifest.h"

#include <stdio.h>
#include <string.h>
#include <strings.h>

#include "../serve/json_reader.h"
#include "catalog_keys.h"
#include "local_kv_policy.h"

#define CHARS_ID   "abcdefghijklmnopqrstuvwxyz0123456789._-"
#define CHARS_NAME "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789._-"
#define CHARS_HEX  "0123456789abcdef"
#define MAX_FILE_BYTES ((uint64_t)1 << 40)

/* Parse context: the first error wins. */
typedef struct {
    char  *err;
    size_t errlen;
    int    failed;
} parse_ctx;

static int bad(parse_ctx *pc, const char *where, const char *what)
{
    if (!pc->failed)
        snprintf(pc->err, pc->errlen, "%s: %s", where, what);
    pc->failed = 1;
    return -1;
}

/* s is non-empty, shorter than cap, and made only of `allowed`. */
static int only_chars(const char *s, const char *allowed, size_t cap)
{
    size_t n = s ? strlen(s) : 0;
    return n > 0 && n < cap && strspn(s, allowed) == n && strcmp(s, ".") != 0 &&
           strcmp(s, "..") != 0 && s[0] != '.' && s[0] != '-';
}

/* Copy a required (or optional) string field. */
static int get_str(parse_ctx *pc, const jr_value *o, const char *key, char *dst, size_t cap,
                   const char *allowed, bool required, const char *where)
{
    const char *v = jr_as_string(jr_object_get(o, key));

    dst[0] = '\0';
    if (!v)
        return required ? bad(pc, where, key) : 0;
    if (allowed ? !only_chars(v, allowed, cap) : strlen(v) >= cap)
        return bad(pc, where, key);
    snprintf(dst, cap, "%s", v);
    return 0;
}

static double get_num(const jr_value *o, const char *key, double fallback)
{
    return jr_as_number(jr_object_get(o, key), fallback);
}

static int hf_org_allowed(const char *repo)
{
    static const char *orgs[] = PN_CATALOG_HF_ORGS;
    const char *slash = strchr(repo, '/');

    if (!slash || strchr(slash + 1, '/') || !only_chars(slash + 1, CHARS_NAME, 96))
        return 0;
    for (size_t i = 0; i < sizeof(orgs) / sizeof(orgs[0]); i++)
        if ((size_t)(slash - repo) == strlen(orgs[i]) &&
            strncasecmp(repo, orgs[i], (size_t)(slash - repo)) == 0)
            return 1;
    return 0;
}

/* The optional files[].unit_types: a non-empty array of known unit type names. */
static int parse_file_unit_types(parse_ctx *pc, const jr_value *o, pn_cat_file *f,
                                 const char *where)
{
    const jr_value *types = jr_object_get(o, "unit_types");

    f->unit_types = 0;
    if (!types)
        return 0;
    if (jr_kind(types) != JR_ARRAY || jr_array_len(types) < 1)
        return bad(pc, where, "files[].unit_types must be a non-empty array");
    for (int i = 0; i < jr_array_len(types); i++) {
        unsigned bit = pn_cat_unit_type_bit(jr_as_string(jr_array_get(types, i)));
        if (!bit)
            return bad(pc, where, "files[].unit_types: each must be k80-gpu, p150 or cpu");
        f->unit_types |= bit;
    }
    return 0;
}

/* The optional files[].ingot: a compressed download of the file. Repo and revision default to
 * the file's; checked like the file's own fields. */
static int parse_file_ingot(parse_ctx *pc, const jr_value *o, pn_cat_file *f, const char *where)
{
    const jr_value *io = jr_object_get(o, "ingot");
    pn_cat_ingot *g = &f->ingot;

    memset(g, 0, sizeof(*g));
    if (!io)
        return 0;
    if (jr_kind(io) != JR_OBJECT)
        return bad(pc, where, "files[].ingot must be an object");
    g->present = true;
    get_str(pc, io, "filename", g->filename, sizeof(g->filename), CHARS_NAME, true, where);
    get_str(pc, io, "hf_repo", g->hf_repo, sizeof(g->hf_repo), NULL, false, where);
    get_str(pc, io, "revision", g->revision, sizeof(g->revision), CHARS_HEX, false, where);
    get_str(pc, io, "base_sha256", g->base_sha256, sizeof(g->base_sha256), CHARS_HEX, false,
            where);
    double size = get_num(io, "size", 0);
    if (pc->failed)
        return -1;
    if (!g->hf_repo[0])
        snprintf(g->hf_repo, sizeof(g->hf_repo), "%s", f->hf_repo);
    if (!g->revision[0])
        snprintf(g->revision, sizeof(g->revision), "%s", f->revision);
    if (!hf_org_allowed(g->hf_repo))
        return bad(pc, where, "files[].ingot.hf_repo is not <our org>/<repo>");
    if (strlen(g->revision) != 40)
        return bad(pc, where, "files[].ingot.revision must be a pinned 40-hex commit");
    if (g->base_sha256[0] && strlen(g->base_sha256) != 64)
        return bad(pc, where, "files[].ingot.base_sha256 must be 64 hex");
    if (!(size >= 1) || size > (double)MAX_FILE_BYTES)
        return bad(pc, where, "files[].ingot.size");
    g->size = (uint64_t)size;
    return 0;
}

static int parse_file(parse_ctx *pc, const jr_value *o, pn_cat_file *f, const char *where)
{
    get_str(pc, o, "role", f->role, sizeof(f->role), CHARS_ID, true, where);
    get_str(pc, o, "hf_repo", f->hf_repo, sizeof(f->hf_repo), NULL, true, where);
    get_str(pc, o, "revision", f->revision, sizeof(f->revision), CHARS_HEX, true, where);
    get_str(pc, o, "filename", f->filename, sizeof(f->filename), CHARS_NAME, true, where);
    get_str(pc, o, "sha256", f->sha256, sizeof(f->sha256), CHARS_HEX, true, where);
    double size = get_num(o, "size", 0);
    if (pc->failed)
        return -1;
    if (strcmp(f->role, "weights") != 0 && strcmp(f->role, "extra") != 0)
        return bad(pc, where, "role must be weights or extra");
    if (!hf_org_allowed(f->hf_repo))
        return bad(pc, where, "hf_repo is not <our org>/<repo>");
    if (strlen(f->revision) != 40)
        return bad(pc, where, "revision must be a pinned 40-hex commit");
    if (strlen(f->sha256) != 64)
        return bad(pc, where, "sha256 must be 64 hex");
    if (!(size >= 1) || size > (double)MAX_FILE_BYTES)
        return bad(pc, where, "size");
    f->size = (uint64_t)size;
    if (parse_file_unit_types(pc, o, f, where) != 0)
        return -1;
    return parse_file_ingot(pc, o, f, where);
}

static int parse_hardware(parse_ctx *pc, const jr_value *h, pn_cat_profile *p, const char *where)
{
    pn_unit_request *r = &p->hardware;
    const char *t = jr_as_string(jr_object_get(h, "unit_type"));

    if (!pn_cat_unit_type_bit(t))
        return bad(pc, where, "hardware.unit_type must be k80-gpu, p150 or cpu");
    snprintf(r->unit_type, sizeof(r->unit_type), "%s", t);
    r->count = (int)get_num(h, "count", 0);
    r->same_card = jr_as_bool(jr_object_get(h, "same_card"), false);
    r->min_mem_per_unit = (uint64_t)get_num(h, "min_mem_per_unit", 0);
    p->min_host_ram = (uint64_t)get_num(h, "min_host_ram", 0);
    if (r->count < 1 || r->count > PN_UNIT_PLACE_MAX)
        return bad(pc, where, "hardware.count");
    r->per_card = (int)get_num(h, "per_card", 0);
    if (r->per_card < 0 || (r->per_card > 0 && (r->count % r->per_card != 0 || r->same_card)))
        return bad(pc, where, "hardware.per_card must divide count (and not with same_card)");
    return 0;
}

static int parse_measured(parse_ctx *pc, const jr_value *m, pn_cat_measured *out, const char *where)
{
    const jr_value *tok = jr_object_get(m, "decode_tok_s");
    const jr_value *ttft = jr_object_get(m, "ttft");

    out->decode_tok_s_8k = get_num(tok, "8k", 0);
    out->decode_tok_s_14k = get_num(tok, "14k", 0);
    out->decode_tok_s_32k = get_num(tok, "32k", 0);
    out->ttft_prompt_tokens = (int)get_num(ttft, "prompt_tokens", 0);
    out->ttft_s = get_num(ttft, "seconds", 0);
    out->context_window = (int)get_num(m, "context_window", 0);
    /* Spec §5.3: a profile ships only with measured numbers. */
    if (!(out->decode_tok_s_8k > 0) || !(out->ttft_s > 0) || out->ttft_prompt_tokens <= 0 ||
        out->context_window <= 0)
        return bad(pc, where, "measured numbers missing (decode_tok_s.8k, ttft, context_window)");
    return 0;
}

/* The optional "options" block. The profile's own load settings are the
 * recommended choice, so they must lie inside the ranges. */
static int parse_options(parse_ctx *pc, const jr_value *o, pn_cat_profile *p, const char *where)
{
    pn_cat_load_options *op = &p->options;
    const jr_value *mem = jr_object_get(o, "mem");
    int sessions = p->max_sessions > 0 ? p->max_sessions : 1;

    if (!o)
        return 0;
    op->present = true;
    get_str(pc, o, "kv_type", op->kv_type, sizeof(op->kv_type), CHARS_ID, true, where);
    if (pc->failed)
        return -1;
    if (!pn_cat_kv_type_allowed(p->hardware.unit_type, op->kv_type))
        return bad(pc, where, PN_LOCAL_KV_FP32_ONLY
                   ? "options.kv_type must be fp32 or bf16 (Q4 KV is not allowed)"
                   : "options.kv_type must be fp32, bf16 or q4 (a p150 also: bfp8, bfp4, "
                     "k8v4, k16v4, k4v16, k16v8, k4h)");
    /* Optional: a p150 KV type for long contexts only (the TT fit decides at load). */
    if (jr_object_get(o, "kv_long")) {
        get_str(pc, o, "kv_long", op->kv_long, sizeof(op->kv_long), CHARS_ID, true, where);
        if (pc->failed)
            return -1;
        if (strcmp(p->hardware.unit_type, "p150") != 0 || strcmp(op->kv_long, "bf16") == 0 ||
            !pn_cat_kv_type_allowed(p->hardware.unit_type, op->kv_long))
            return bad(pc, where, "options.kv_long must be a p150 block-float KV type (k4h)");
    }
    op->ctx_min = (int)get_num(o, "ctx_min", 0);
    op->ctx_max = (int)get_num(o, "ctx_max", 0);
    op->sessions_max = (int)get_num(o, "sessions_max", 0);
    op->mem_fixed = (uint64_t)get_num(mem, "fixed", 0);
    op->mem_per_session = (uint64_t)get_num(mem, "per_session", 0);
    op->mem_kv_per_token = (uint64_t)get_num(mem, "kv_per_token", 0);
    op->mem_ctx_per_token = (uint64_t)get_num(mem, "ctx_per_token", 0);
    if (op->ctx_min < 1024 || op->ctx_max < op->ctx_min || op->ctx_max > (1 << 20))
        return bad(pc, where, "options.ctx_min/ctx_max");
    if (op->sessions_max < 1 || op->sessions_max > 64)
        return bad(pc, where, "options.sessions_max");
    if (op->mem_fixed == 0 || op->mem_kv_per_token == 0)
        return bad(pc, where, "options.mem needs fixed and kv_per_token (measured)");
    if (p->ctx_cap < op->ctx_min || p->ctx_cap > op->ctx_max || sessions > op->sessions_max)
        return bad(pc, where, "load.ctx_cap / max_sessions must lie inside options");
    return 0;
}

static int parse_profile(parse_ctx *pc, const jr_value *o, pn_cat_profile *p, const char *model)
{
    char where[160];
    const jr_value *load = jr_object_get(o, "load");
    char serr[96];

    snprintf(where, sizeof(where), "model %s profile", model);
    if (get_str(pc, o, "id", p->id, sizeof(p->id), CHARS_ID, true, where) != 0)
        return -1;
    snprintf(where, sizeof(where), "model %s profile %s", model, p->id);
    p->rank = (int)get_num(o, "rank", 100);
    if (parse_hardware(pc, jr_object_get(o, "hardware"), p, where) != 0)
        return -1;
    p->ctx_cap = (int)get_num(load, "ctx_cap", 0);
    p->max_sessions = (int)get_num(load, "max_sessions", 0);
    p->max_concurrent = (int)get_num(load, "max_concurrent", 0);
    p->reasoning_budget = (int)get_num(load, "reasoning_budget", 0);
    p->reasoning_reserve = (int)get_num(load, "reasoning_reserve", 0);
    p->mtp = jr_as_bool(jr_object_get(load, "mtp"), false);
    p->mtp_drafts = (int)get_num(load, "mtp_drafts", 0);
    if (p->ctx_cap < 0 || p->ctx_cap > (1 << 20) || p->max_sessions < 0 ||
        p->max_concurrent < 0 || p->reasoning_budget < 0 || p->reasoning_reserve < 0 ||
        p->reasoning_reserve > (1 << 20))
        return bad(pc, where, "load values out of range");
    /* The draft count is the p150's PN_TT_SPEC (1..15); the CUDA head has no such knob. */
    if (p->mtp_drafts < 0 || p->mtp_drafts > 15 ||
        (p->mtp_drafts > 0 && (!p->mtp || strcmp(p->hardware.unit_type, "p150") != 0)))
        return bad(pc, where, "load.mtp_drafts must be 1-15, with mtp true, on a p150");
    pn_model_sampling_from_json(jr_object_get(o, "sampling"), &p->sampling);
    if (pn_model_sampling_validate(&p->sampling, serr, sizeof(serr)) != 0)
        return bad(pc, where, serr);
    if (parse_options(pc, jr_object_get(o, "options"), p, where) != 0)
        return -1;
    return parse_measured(pc, jr_object_get(o, "measured"), &p->measured, where);
}

/* The model's text fields and capabilities. */
static int parse_model_info(parse_ctx *pc, const jr_value *o, pn_cat_model *m, const char *where)
{
    const jr_value *lic = jr_object_get(o, "licence");
    const jr_value *caps = jr_object_get(o, "capabilities");

    get_str(pc, o, "name", m->name, sizeof(m->name), NULL, true, where);
    get_str(pc, o, "family", m->family, sizeof(m->family), CHARS_ID, false, where);
    get_str(pc, o, "description", m->description, sizeof(m->description), NULL, false, where);
    get_str(pc, lic, "id", m->licence_id, sizeof(m->licence_id), NULL, true, where);
    get_str(pc, lic, "url", m->licence_url, sizeof(m->licence_url), NULL, false, where);
    for (int i = 0; i < jr_array_len(caps) && m->n_capabilities < PN_CAT_MAX_CAPS; i++) {
        const char *c = jr_as_string(jr_array_get(caps, i));
        if (!c || !only_chars(c, CHARS_ID, 16))
            return bad(pc, where, "capabilities");
        snprintf(m->capabilities[m->n_capabilities++], 16, "%s", c);
    }
    return pc->failed ? -1 : 0;
}

/* Every file must be needed by at least one profile, and every profile's unit
 * type must need exactly one weights file (the file its load opens): one untagged
 * weights file, or one per unit type (a K80 GGUF + a p150 metadata GGUF). */
static int check_file_unit_types(parse_ctx *pc, const pn_cat_model *m, const char *where)
{
    unsigned profile_units = 0;

    for (int i = 0; i < m->n_profiles; i++)
        profile_units |= pn_cat_unit_type_bit(m->profiles[i].hardware.unit_type);
    for (int i = 0; i < m->n_files; i++) {
        const pn_cat_file *f = &m->files[i];
        if (f->unit_types && !(f->unit_types & profile_units))
            return bad(pc, where, "files[].unit_types names no unit type of this model's profiles");
    }
    for (int i = 0; i < m->n_profiles; i++) {
        unsigned unit = pn_cat_unit_type_bit(m->profiles[i].hardware.unit_type);
        int weights = 0;
        for (int j = 0; j < m->n_files; j++)
            weights += strcmp(m->files[j].role, "weights") == 0 &&
                       pn_cat_file_needed(&m->files[j], unit);
        if (weights != 1)
            return bad(pc, where, "each profile's unit type must need exactly one weights file");
    }
    return 0;
}

static int parse_model(parse_ctx *pc, const jr_value *o, pn_cat_model *m)
{
    const jr_value *files = jr_object_get(o, "files");
    const jr_value *profiles = jr_object_get(o, "profiles");
    char where[96];

    if (get_str(pc, o, "id", m->id, sizeof(m->id), CHARS_ID, true, "model") != 0)
        return -1;
    snprintf(where, sizeof(where), "model %s", m->id);
    if (parse_model_info(pc, o, m, where) != 0)
        return -1;
    if (jr_array_len(files) < 1 || jr_array_len(files) > PN_CAT_MAX_FILES)
        return bad(pc, where, "files");
    for (int i = 0; i < jr_array_len(files); i++) {
        if (parse_file(pc, jr_array_get(files, i), &m->files[m->n_files++], where) != 0)
            return -1;
    }
    if (jr_array_len(profiles) < 1 || jr_array_len(profiles) > PN_CAT_MAX_PROFILES)
        return bad(pc, where, "profiles");
    for (int i = 0; i < jr_array_len(profiles); i++)
        if (parse_profile(pc, jr_array_get(profiles, i), &m->profiles[m->n_profiles++], m->id))
            return -1;
    return check_file_unit_types(pc, m, where);
}

int pn_cat_manifest_parse(const char *json, pn_catalog_manifest *out, char *err, size_t errlen)
{
    parse_ctx pc = { err, errlen, 0 };
    jr_value *root = jr_parse(json);
    const jr_value *models = jr_object_get(root, "models");

    memset(out, 0, sizeof(*out));
    if (!root || jr_kind(root) != JR_OBJECT) {
        bad(&pc, "manifest", "not a JSON object");
        goto done;
    }
    out->schema = (int)get_num(root, "schema", 0);
    out->version = (long long)get_num(root, "version", -1);
    get_str(&pc, root, "published", out->published, sizeof(out->published), NULL, false,
            "manifest");
    if (out->schema != PN_CAT_SCHEMA)
        bad(&pc, "manifest", "unsupported schema");
    else if (out->version < 1)
        bad(&pc, "manifest", "version must be >= 1");
    else if (jr_kind(models) != JR_ARRAY || jr_array_len(models) > PN_CAT_MAX_MODELS)
        bad(&pc, "manifest", "models");
    for (int i = 0; !pc.failed && i < jr_array_len(models); i++) {
        pn_cat_model *m = &out->models[out->n_models];
        if (parse_model(&pc, jr_array_get(models, i), m) == 0 &&
            pn_cat_find_model(out, m->id) != NULL)
            bad(&pc, m->id, "duplicate model id");
        out->n_models++;
    }
done:
    jr_free(root);
    return pc.failed ? -1 : 0;
}

const pn_cat_model *pn_cat_find_model(const pn_catalog_manifest *m, const char *id)
{
    for (int i = 0; id && i < m->n_models; i++)
        if (strcmp(m->models[i].id, id) == 0)
            return &m->models[i];
    return NULL;
}

const pn_cat_profile *pn_cat_find_profile(const pn_cat_model *model, const char *id)
{
    for (int i = 0; id && i < model->n_profiles; i++)
        if (strcmp(model->profiles[i].id, id) == 0)
            return &model->profiles[i];
    return NULL;
}

const pn_cat_file *pn_cat_weights_file(const pn_cat_model *model)
{
    for (int i = 0; i < model->n_files; i++)
        if (strcmp(model->files[i].role, "weights") == 0)
            return &model->files[i];
    return NULL;
}

unsigned pn_cat_unit_type_bit(const char *unit_type)
{
    if (!unit_type)
        return 0;
    if (strcmp(unit_type, "k80-gpu") == 0)
        return PN_CAT_UNIT_K80_GPU;
    if (strcmp(unit_type, "p150") == 0)
        return PN_CAT_UNIT_P150;
    if (strcmp(unit_type, "cpu") == 0)
        return PN_CAT_UNIT_CPU;
    return 0;
}

bool pn_cat_kv_type_allowed(const char *unit_type, const char *kv_type)
{
    /* The PN_TT_KV_TYPE values besides bf16 (pn_tt_kv_format.h); test_tt_q38_fit checks that
     * each parses there. */
    static const char *const p150_types[] = { "bfp8", "bfp4", "k8v4", "k16v4", "k4v16",
                                              "k16v8", "k4h" };

    if (!kv_type)
        return false;
    if (strcmp(kv_type, "fp32") == 0 || strcmp(kv_type, "bf16") == 0)
        return true;
    if (PN_LOCAL_KV_FP32_ONLY)
        return false;   /* full precision only: no Q4, no block-float KV */
    if (strcmp(kv_type, "q4") == 0)
        return true;
    for (size_t i = 0; unit_type && strcmp(unit_type, "p150") == 0 &&
                       i < sizeof(p150_types) / sizeof(p150_types[0]); i++)
        if (strcmp(kv_type, p150_types[i]) == 0)
            return true;
    return false;
}

bool pn_cat_file_needed(const pn_cat_file *f, unsigned units)
{
    return f->unit_types == 0 || (f->unit_types & units) != 0;
}

void pn_cat_model_for_units(const pn_cat_model *model, unsigned units, pn_cat_model *out)
{
    *out = *model;
    out->n_files = 0;
    for (int i = 0; i < model->n_files; i++)
        if (pn_cat_file_needed(&model->files[i], units))
            out->files[out->n_files++] = model->files[i];
}

int pn_cat_file_url(const char *base, const pn_cat_file *f, char *out, size_t cap)
{
    int n = snprintf(out, cap, "%s/%s/resolve/%s/%s",
                     base && *base ? base : "https://huggingface.co", f->hf_repo, f->revision,
                     f->filename);
    return (n < 0 || (size_t)n >= cap) ? -1 : 0;
}

int pn_cat_ingot_url(const char *base, const pn_cat_file *f, char *out, size_t cap)
{
    if (!f->ingot.present)
        return -1;
    int n = snprintf(out, cap, "%s/%s/resolve/%s/%s",
                     base && *base ? base : "https://huggingface.co", f->ingot.hf_repo,
                     f->ingot.revision, f->ingot.filename);
    return (n < 0 || (size_t)n >= cap) ? -1 : 0;
}
