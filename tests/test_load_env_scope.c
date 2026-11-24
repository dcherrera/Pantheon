/* SPDX-License-Identifier: AGPL-3.0-or-later */
/* Copyright (C) 2026 David (Corey) Herrera */
/*
 * test_load_env_scope.c — the environment one model load sets (QWEN35_MTP, PN_TT_KV_TYPE,
 * PN_TT_SPEC: model_admin.c) is in place during the load and put back after: a variable
 * that was set gets its old value again, one that was unset is unset again, and a variable
 * the load leaves alone (value NULL) is never touched.
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "../src/serve/load_env_scope.h"

static int g_checks, g_failures;

static void check(int ok, const char *what)
{
    g_checks++;
    if (!ok) {
        g_failures++;
        printf("  FAIL: %s\n", what);
    } else {
        printf("  ok: %s\n", what);
    }
}

static int env_is(const char *name, const char *value)
{
    const char *v = getenv(name);

    return value ? v && strcmp(v, value) == 0 : v == NULL;
}

int main(void)
{
    setenv("PN_TEST_ENV_SET", "3", 1);
    unsetenv("PN_TEST_ENV_UNSET");
    setenv("PN_TEST_ENV_ALONE", "keep", 1);
    pn_load_env_var vars[] = {
        { "PN_TEST_ENV_SET", "1", NULL },
        { "PN_TEST_ENV_UNSET", "k8v4", NULL },
        { "PN_TEST_ENV_ALONE", NULL, NULL },
    };
    const size_t n = sizeof(vars) / sizeof(vars[0]);

    printf("load_env_scope:\n");
    pn_load_env_apply(vars, n);
    check(env_is("PN_TEST_ENV_SET", "1"), "during the load: a set variable has the load's value");
    check(env_is("PN_TEST_ENV_UNSET", "k8v4"), "during the load: an unset one has it too");
    check(env_is("PN_TEST_ENV_ALONE", "keep"), "during the load: a NULL value leaves it alone");
    pn_load_env_restore(vars, n);
    check(env_is("PN_TEST_ENV_SET", "3"), "after: the old value is back");
    check(env_is("PN_TEST_ENV_UNSET", NULL), "after: the unset variable is unset again");
    check(env_is("PN_TEST_ENV_ALONE", "keep"), "after: the untouched one is unchanged");
    check(vars[0].saved == NULL && vars[1].saved == NULL, "after: nothing left saved");

    /* Nothing to set: apply / restore change nothing (every load without these knobs). */
    pn_load_env_var none[] = { { "PN_TEST_ENV_SET", NULL, NULL } };
    pn_load_env_apply(none, 1);
    pn_load_env_restore(none, 1);
    check(env_is("PN_TEST_ENV_SET", "3"), "a load without the knobs changes nothing");

    printf("test_load_env_scope: %d checks, %d failed\n", g_checks, g_failures);
    return g_failures ? 1 : 0;
}
