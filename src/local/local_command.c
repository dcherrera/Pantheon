/* SPDX-License-Identifier: AGPL-3.0-or-later */
/* Copyright (C) 2026 David (Corey) Herrera */
/**
 * @file   local_command.c
 * @brief  Implementation of `pantheon local` (see local_command.h).
 *
 * @status None
 * @issues None
 * @todo   None
 */

#include "local_command.h"
#include "local_kv_policy.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include <sys/stat.h>

#include "../serve/serve_command.h"
#include "catalog.h"
#include "local_service.h"

static pn_local_ctx g_local;   /* lives for the process */

static void print_local_usage(void)
{
    printf("Usage: pantheon local [options]\n");
    printf("       pantheon local pair-code [--state-dir <dir>]\n");
    printf("       pantheon local check-catalog <catalog.json> [<catalog.json.minisig>]\n\n");
    printf("Appliance mode: starts with no model, binds %s, is discoverable on the\n",
           PN_LOCAL_DEFAULT_HOST);
    printf("LAN (_pantheon._tcp), pairs clients with a code printed here, and installs\n");
    printf("signed catalog models.\n\n");
    printf("  --port <n>          Port (default %d)\n", PN_LOCAL_DEFAULT_PORT);
    printf("  --host <addr>       Bind address (default %s)\n", PN_LOCAL_DEFAULT_HOST);
    printf("  --state-dir <dir>   Keys, pairing code, catalog version, loaded set\n");
    printf("                      (default ~/.pantheon/local)\n");
    printf("  --store <dir>       Model store (default ~/.pantheon/models)\n");
    printf("  --catalog-url <u>   Catalog manifest URL (signature at <u>.minisig);\n");
    printf("                      default %s,\n", PN_LOCAL_DEFAULT_CATALOG_URL);
    printf("                      '' = no catalog fetch\n");
    printf("  --no-mdns           Do not announce on the LAN\n");
    printf("  --hf-base-url <u>   Download base (default https://huggingface.co)\n");
    printf("  --allow-http        Let downloads use http:// and file:// (mirrors, tests);\n");
    printf("                      files are still checked against the signed sha256\n");
    printf("\n  pair-code           Print the current pairing code (same user as the box)\n");
}

/* Fill the default state/store dirs under $HOME/.pantheon. */
static int set_default_dirs(pn_local_ctx *c)
{
    const char *home = getenv("HOME");

    if (!home || !*home)
        home = ".";
    if (snprintf(c->state_dir, sizeof(c->state_dir), "%s/.pantheon/local", home)
            >= (int)sizeof(c->state_dir) ||
        snprintf(c->store_dir, sizeof(c->store_dir), "%s/.pantheon/models", home)
            >= (int)sizeof(c->store_dir))
        return -1;
    return 0;
}

/* Copy a flag value into a fixed buffer; -1 when it does not fit. */
static int copy_arg(char *dst, size_t cap, const char *src, const char *flag)
{
    if (snprintf(dst, cap, "%s", src) >= (int)cap) {
        fprintf(stderr, "local: %s value too long\n", flag);
        return -1;
    }
    return 0;
}

/* Apply the appliance flags. Returns 0, 1 for --help (exit 0), -1 on error. */
static int apply_local_flags(pn_local_ctx *c, int argc, char **argv, int first)
{
    for (int i = first; i < argc; i++) {
        const char *a = argv[i];
        const char *v = i + 1 < argc ? argv[i + 1] : NULL;
        int rc = 0;

        if (strcmp(a, "--help") == 0 || strcmp(a, "-h") == 0)
            return 1;
        if (strcmp(a, "--no-mdns") == 0) {
            c->no_mdns = 1;
            continue;
        }
        if (strcmp(a, "--allow-http") == 0) {
            c->allow_http = 1;
            continue;
        }
        if (!v) {
            fprintf(stderr, "local: unknown option or missing value '%s'\n", a);
            return -1;
        }
        if (strcmp(a, "--port") == 0) {
            char *end = NULL;
            long p = strtol(v, &end, 10);
            if (end == v || *end || p <= 0 || p > 65535) {
                fprintf(stderr, "local: bad --port '%s'\n", v);
                return -1;
            }
            c->port = (int)p;
        } else if (strcmp(a, "--host") == 0) {
            rc = copy_arg(c->host, sizeof(c->host), v, a);
        } else if (strcmp(a, "--state-dir") == 0) {
            rc = copy_arg(c->state_dir, sizeof(c->state_dir), v, a);
        } else if (strcmp(a, "--store") == 0) {
            rc = copy_arg(c->store_dir, sizeof(c->store_dir), v, a);
        } else if (strcmp(a, "--catalog-url") == 0) {
            rc = copy_arg(c->catalog_url, sizeof(c->catalog_url), v, a);
        } else if (strcmp(a, "--hf-base-url") == 0) {
            rc = copy_arg(c->hf_base_url, sizeof(c->hf_base_url), v, a);
        } else {
            fprintf(stderr, "local: unknown option '%s'\n", a);
            return -1;
        }
        if (rc != 0)
            return -1;
        i++;
    }
    return 0;
}

int pn_local_parse_args(pn_local_ctx *c, int argc, char **argv, int first)
{
    memset(c, 0, sizeof(*c));
    c->port = PN_LOCAL_DEFAULT_PORT;
    snprintf(c->host, sizeof(c->host), "%s", PN_LOCAL_DEFAULT_HOST);
    snprintf(c->catalog_url, sizeof(c->catalog_url), "%s", PN_LOCAL_DEFAULT_CATALOG_URL);
    if (set_default_dirs(c) != 0)
        return -1;
    return apply_local_flags(c, argc, argv, first);
}

/* `pantheon local check-catalog <manifest> [<sig>]`: validate a manifest with the
 * box's own parser, and verify its signature with this build's keys. */
static int run_check_catalog(int argc, char **argv)
{
    char *text = NULL, *sig = NULL, err[256];
    size_t len = 0;
    pn_catalog_manifest *m = calloc(1, sizeof(*m));
    int rc = 1;

    if (argc < 3 || !m || pn_local_read_file(argv[2], 4u << 20, &text, &len) != 0) {
        fprintf(stderr, "local: usage: pantheon local check-catalog <catalog.json> [<sig>]\n");
        goto done;
    }
    if (pn_cat_manifest_parse(text, m, err, sizeof(err)) != 0) {
        fprintf(stderr, "local: manifest refused: %s\n", err);
        goto done;
    }
    printf("manifest ok: version %lld, %d model(s)\n", m->version, m->n_models);
    if (argc >= 4) {
        pn_minisign_pubkey keys[4];
        int n = pn_catalog_trusted_keys(keys, 4);
        if (pn_local_read_file(argv[3], 4096, &sig, NULL) != 0 ||
            pn_minisign_verify((const uint8_t *)text, len, sig, keys, n, NULL, NULL, err,
                               sizeof(err)) != 0) {
            fprintf(stderr, "local: signature refused: %s\n", sig ? err : "unreadable");
            goto done;
        }
        printf("signature ok (trusted by this build)\n");
    }
    rc = 0;
done:
    free(text);
    free(sig);
    free(m);
    return rc;
}

/* Only with PN_LOCAL_KV_FP32_ONLY (local_kv_policy.h): pin fp32 KV for this
 * process before any model loads — the qwen35 KV format (QWEN35_KV_Q4 defaults
 * ON) and the memory ladder's rung 4a. With the policy at 0 (David's decision)
 * each load follows its profile's options.kv_type instead. */
static void pin_kv_format(void)
{
    if (!PN_LOCAL_KV_FP32_ONLY)
        return;   /* the serve's own KV default (local_kv_policy.h) */
    const char *was = getenv("QWEN35_KV_Q4");

    if (was && was[0] != '0')
        fprintf(stderr, "local: ignoring QWEN35_KV_Q4=%s: Pantheon Local runs fp32 KV\n", was);
    setenv("QWEN35_KV_Q4", "0", 1);
    setenv("PN_LADDER_KV_Q4", "0", 1);
    fprintf(stderr, "local: KV cache fp32 (never Q4)\n");
}

/* `pantheon local pair-code`: print the code the running box wrote. */
static int run_pair_code(pn_local_ctx *c)
{
    char path[PN_LOCAL_PATH_MAX];
    char *code = NULL;

    if (pn_local_path_join(path, sizeof(path), c->state_dir, "pair_code") != 0 ||
        pn_local_read_file(path, 64, &code, NULL) != 0) {
        fprintf(stderr, "local: no pairing code at %s (is `pantheon local` running, "
                        "as this user, with this --state-dir?)\n", path);
        return 1;
    }
    code[strcspn(code, "\r\n")] = '\0';
    printf("%s\n", code);
    free(code);
    return 0;
}

int pn_local_main(int argc, char **argv)
{
    pn_local_ctx *c = &g_local;
    int pair_code = argc > 1 && strcmp(argv[1], "pair-code") == 0;

    if (argc > 1 && strcmp(argv[1], "check-catalog") == 0)
        return run_check_catalog(argc, argv);
    int rc = pn_local_parse_args(c, argc, argv, pair_code ? 2 : 1);

    if (rc != 0) {
        print_local_usage();
        return rc > 0 ? 0 : 1;
    }
    if (pair_code)
        return run_pair_code(c);
    if (pn_local_mkdir_p(c->state_dir, 0700) != 0 || pn_local_mkdir_p(c->store_dir, 0700) != 0) {
        fprintf(stderr, "local: cannot create %s or %s\n", c->state_dir, c->store_dir);
        return 1;
    }
    pin_kv_format();
    if (pn_local_service_install(c) != 0)
        return 1;

    char port_arg[16];
    snprintf(port_arg, sizeof(port_arg), "%d", c->port);
    char *serve_argv[] = { "serve", "--local", "--host", c->host, "--port", port_arg, NULL };
    rc = pn_serve_main(6, serve_argv);
    pn_local_service_release(c);
    return rc;
}
