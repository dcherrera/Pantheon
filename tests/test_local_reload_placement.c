/* SPDX-License-Identifier: AGPL-3.0-or-later */
/* Copyright (C) 2026 David (Corey) Herrera */
/*
 * test_local_reload_placement.c — placement follows the profile, and the boot
 * reload re-checks each remembered model against the hardware found now.
 *
 *   H2f: p150, 2026-10-27, a `cpu`-profile model remembered while the card was
 *        busy came back after a restart that saw the card. A profile that no
 *        longer fits is not loaded, stays remembered, and GET /local/models/loaded
 *        says why. A profile that still fits passes the re-check.
 *   H2g: on that box the `cpu` load went on the card (llama_tt, 0.74 GiB) while
 *        the box said "on the CPU". A cpu load never takes a die, even when the
 *        die manager has free ones to auto-place on.
 *
 * The card box is two logical dies (the hw report then shows a k80 card and no
 * cpu unit, as on a box with an accelerator). Uses models/qwen3-0.6b-Q8_0.gguf
 * (PN_TEST_MODEL); skipped when absent.
 */

#include "local_test_support.h"

#include "../src/accel/die_manager.h"
#include "../src/local/local_load.h"
#include "../src/local/local_model_routes.h"
#include "../src/serve/json_reader.h"
#include "../src/serve/model_admin.h"

#define DIR "build/test_local/reload_placement"
#define GIB(n) ((uint64_t)(n) << 30)

static const char *g_model;

static pn_loaded_entry remembered(const char *id, const char *profile, const char *unit_type,
                                  int unit)
{
    pn_loaded_entry e;

    memset(&e, 0, sizeof(e));
    snprintf(e.catalog_id, sizeof(e.catalog_id), "%s", id);
    snprintf(e.profile_id, sizeof(e.profile_id), "%s", profile);
    snprintf(e.path, sizeof(e.path), "%s", g_model);
    snprintf(e.hardware.unit_type, sizeof(e.hardware.unit_type), "%s", unit_type);
    e.hardware.count = 1;
    if (unit >= 0) {
        e.units[0] = unit;
        e.n_units = 1;
    }
    e.ctx_cap = 2048;
    e.max_sessions = 1;
    return e;
}

/* The loaded-set entry for id in the GET /local/models/loaded body, or NULL. */
static const jr_value *loaded_row(const jr_value *root, const char *id)
{
    const jr_value *models = jr_object_get(root, "models");

    for (int i = 0; i < jr_array_len(models); i++) {
        const jr_value *m = jr_array_get(models, i);
        const char *cid = jr_as_string(jr_object_get(m, "catalog_id"));
        if (cid && strcmp(cid, id) == 0)
            return m;
    }
    return NULL;
}

static int registry_dies(mr_registry *reg, const char *name)
{
    mr_info info[MR_MAX_MODELS];
    int n = mr_snapshot(reg, info, MR_MAX_MODELS);

    for (int i = 0; i < n; i++)
        if (strcmp(info[i].name, name) == 0)
            return info[i].n_dies;
    return -1;
}

static int die_models(int die)
{
    pn_dm_die d;
    return pn_dm_get(die, &d) == 0 ? d.n_models : -1;
}

/* H2f: a cpu profile remembered on a card box is skipped and reported; a k80-gpu
 * profile that still fits comes back on its unit. */
static void test_reload_rechecks_profile(mr_registry *reg, pn_local_ctx *ctx)
{
    pn_loaded_entry cpu = remembered("small-cpu", "cpu", "cpu", -1);
#ifdef USE_TT
    pn_loaded_entry gpu = remembered("small-gpu", "p150-one", "p150", 1);
#else
    pn_loaded_entry gpu = remembered("small-gpu", "k80-one", "k80-gpu", 1);
#endif
    char reason[PN_RELOAD_REASON_MAX];
    json_writer jw;

    printf("reload_rechecks_profile:\n");
    CHECK(pn_loaded_set_add(ctx->loaded, &cpu) == 0, "remember the cpu-profile model");
    CHECK(pn_loaded_set_add(ctx->loaded, &gpu) == 0, "remember the k80-gpu-profile model");
    pn_local_reload_now(ctx);

    CHECK(registry_dies(reg, "small-cpu") < 0, "the cpu profile is not loaded on a card box");
    CHECK(die_models(0) == 0, "no die was taken for it");
    CHECK(pn_local_reload_state("small-cpu", 1, reason, sizeof(reason)) == PN_RELOAD_SKIPPED,
          "its reload is reported as skipped");
    CHECK(strstr(reason, "no longer fits") && strstr(reason, "cpu"), "with the reason");
    /* A CPU build refuses explicit dies (serve_backend_guard) and an accelerator
     * build cannot put a model on a logical die, so the load itself fails here;
     * what matters is that the re-check let it through. */
    CHECK(pn_local_reload_state("small-gpu", 1, reason, sizeof(reason)) != PN_RELOAD_SKIPPED,
          "the card profile still fits the card box: not skipped");

    jw_init(&jw);
    pn_local_write_loaded(&jw, ctx);
    jr_value *root = jr_parse(jw.data);
    const jr_value *c = loaded_row(root, "small-cpu");
    const jr_value *g = loaded_row(root, "small-gpu");
    CHECK(c && strcmp(jr_as_string(jr_object_get(c, "state")), "not_loaded") == 0,
          "loaded API: the cpu model is remembered, state not_loaded");
    CHECK(c && jr_as_string(jr_object_get(c, "reason")) &&
              strstr(jr_as_string(jr_object_get(c, "reason")), "no longer fits"),
          "loaded API: with the reason");
    CHECK(g != NULL, "loaded API: the gpu model is listed");
    jr_free(root);
    jw_free(&jw);
}

/* H2g: a cpu load takes no die even when free dies would auto-place it. */
static void test_cpu_load_takes_no_die(mr_registry *reg)
{
    ma_load_request lr;
    char err[256] = "";

    printf("cpu_load_takes_no_die:\n");
    memset(&lr, 0, sizeof(lr));
    lr.path = g_model;
    lr.name = "plain-cpu";
    lr.opts.ctx_cap = 2048;
    lr.opts.cpu_only = true;
    lr.max_sessions = 1;
    CHECK(ma_load_model(reg, &lr, err, sizeof(err)) == 0, "the cpu load succeeds");
    CHECK(registry_dies(reg, "plain-cpu") == 0, "it holds no die (CPU-resident)");
    CHECK(die_models(0) == 0, "die 0 stays free for the card profiles");
    mr_entry *e = mr_acquire(reg, "plain-cpu");
    CHECK(e && e->model && !pn_model_on_accelerator(e->model),
          "its forward is the CPU one (no CUDA / Tenstorrent upload in any build)");
    if (e)
        mr_release(reg, e);

    lr.name = "cpu-with-dies";
    lr.dies[0] = 0;
    lr.n_dies = 1;
    CHECK(ma_load_model(reg, &lr, err, sizeof(err)) != 0 && strstr(err, "cpu"),
          "a cpu load that names dies is refused");
}

int main(void)
{
    mr_registry reg;
    pn_local_ctx ctx;

    printf("test_local_reload_placement\n");
    g_model = getenv("PN_TEST_MODEL");
    if (!g_model || !*g_model)
        g_model = "models/qwen3-0.6b-Q8_0.gguf";
    if (access(g_model, R_OK) != 0) {
        printf("  SKIP: %s not found\n", g_model);
        return 0;
    }
    lt_clean_dir(DIR);
    lt_make_dir(DIR);
    CHECK(pn_dm_init_logical(2, GIB(11)) == 0, "a card box: two logical dies");
    CHECK(mr_init(&reg) == 0, "registry");
    memset(&ctx, 0, sizeof(ctx));
    snprintf(ctx.state_dir, sizeof(ctx.state_dir), "%s", DIR);
    snprintf(ctx.store_dir, sizeof(ctx.store_dir), "%s", DIR);
    ctx.started = time(NULL);
    ctx.reg = &reg;
    ctx.loaded = pn_loaded_set_open(DIR "/loaded.json");
    CHECK(ctx.loaded != NULL, "loaded set");
    if (ctx.loaded) {
        test_reload_rechecks_profile(&reg, &ctx);
        test_cpu_load_takes_no_die(&reg);
        pn_loaded_set_close(ctx.loaded);
    }
    mr_free(&reg);
    pn_dm_shutdown();
    return lt_report("test_local_reload_placement");
}
