/* SPDX-License-Identifier: AGPL-3.0-or-later */
/* Copyright (C) 2026 David (Corey) Herrera */
/**
 * @file test_tt_role_table.c
 * @brief The role table off the card (pn_tt_role_table.h): overlapping claims in a shared mode
 *        are refused and in disjoint modes allowed; a mode switch tears down every role the
 *        target mode does not run (reverse order) BEFORE it re-arms any role it does
 *        (registration order); a failed re-arm is reported and leaves the role down.
 *
 * @status P0.
 * @issues None
 * @todo None
 */

#include "../src/accel/tt/pn_tt_role_table.h"

#include <stdio.h>
#include <string.h>

static int g_failed = 0;
static int g_checked = 0;
static char g_log[256];
static int g_arm_rc = 0;

#define CHECK(cond, msg)                                                                     \
    do {                                                                                     \
        g_checked++;                                                                         \
        if (!(cond)) {                                                                       \
            g_failed++;                                                                      \
            fprintf(stderr, "FAIL: %s (log '%s')\n", msg, g_log);                            \
        }                                                                                    \
    } while (0)

static void note(const char *what, const char *who)
{
    strncat(g_log, what, sizeof(g_log) - strlen(g_log) - 1u);
    strncat(g_log, who, sizeof(g_log) - strlen(g_log) - 1u);
    strncat(g_log, " ", sizeof(g_log) - strlen(g_log) - 1u);
}

static void down(void *user) { note("-", (const char *)user); }

static int up(void *user, void *context)
{
    if (context && strcmp((const char *)context, "short") == 0 && strcmp(user, "grid") == 0) {
        note("~", (const char *)user);
        return 1;
    }
    note("+", (const char *)user);
    return g_arm_rc;
}

int main(void)
{
    static const uint32_t norm[] = { 0, 1, 2, 3, 7, 8, 9, 10 };
    static const uint32_t grid[] = { 0, 1, 2, 3, 4, 5, 6, 7 };
    static const uint32_t attn[] = { 96, 97 };
    double torn = -1.0, armed = -1.0;

    pn_tt_role_table_clear();
    CHECK(pn_tt_role_table_claim("norm", norm, 8u, PN_TT_ROLE_MODE_DECODE, down, up, "norm") == 0,
          "the norm role claims its decode cores");
    CHECK(pn_tt_role_table_claim("x", grid, 8u, PN_TT_ROLE_MODE_DECODE, down, up, "x") != 0,
          "an overlapping decode claim is refused");
    CHECK(pn_tt_role_table_claim("grid", grid, 8u, PN_TT_ROLE_MODE_PREFILL, down, up, "grid") == 0,
          "the same cores in prefill only are allowed");
    CHECK(pn_tt_role_table_claim("attn", attn, 2u,
                                 PN_TT_ROLE_MODE_DECODE | PN_TT_ROLE_MODE_PREFILL, down, up,
                                 "attn") == 0,
          "a role in both modes on its own cores");
    CHECK(strcmp(pn_tt_role_table_holder(2u, PN_TT_ROLE_MODE_DECODE), "norm") == 0 &&
              strcmp(pn_tt_role_table_holder(2u, PN_TT_ROLE_MODE_PREFILL), "grid") == 0 &&
              pn_tt_role_table_holder(50u, PN_TT_ROLE_MODE_DECODE) == NULL,
          "the holder walk names the role per mode");

    g_log[0] = '\0';
    CHECK(pn_tt_role_table_enter(PN_TT_ROLE_MODE_PREFILL, NULL, &torn, &armed) == 0 &&
              strcmp(g_log, "-norm ") == 0 && torn >= 0.0 && armed >= 0.0,
          "into prefill: the norm role down, the both-mode role untouched");
    g_log[0] = '\0';
    (void)pn_tt_role_table_enter(PN_TT_ROLE_MODE_DECODE, NULL, NULL, NULL);
    CHECK(strcmp(g_log, "-grid +norm ") == 0, "back to decode: grid down BEFORE norm up");
    g_log[0] = '\0';
    (void)pn_tt_role_table_enter(PN_TT_ROLE_MODE_PREFILL, NULL, NULL, NULL);
    CHECK(strcmp(g_log, "-norm +grid ") == 0, "prefill again: norm down BEFORE grid up");
    g_arm_rc = -1;
    g_log[0] = '\0';
    CHECK(pn_tt_role_table_enter(PN_TT_ROLE_MODE_DECODE, NULL, NULL, NULL) != 0,
          "a failed re-arm is reported");
    g_arm_rc = 0;
    g_log[0] = '\0';
    (void)pn_tt_role_table_enter(PN_TT_ROLE_MODE_DECODE, NULL, NULL, NULL);
    CHECK(strcmp(g_log, "+norm ") == 0, "and the role it left down re-arms next switch");
    g_log[0] = '\0';
    CHECK(pn_tt_role_table_enter(PN_TT_ROLE_MODE_PREFILL, "short", NULL, NULL) == 0 &&
              strcmp(g_log, "-norm ~grid ") == 0,
          "a declined arm is not a failure and leaves the role down");
    g_log[0] = '\0';
    (void)pn_tt_role_table_enter(PN_TT_ROLE_MODE_PREFILL, NULL, NULL, NULL);
    CHECK(strcmp(g_log, "+grid ") == 0, "and it is asked again at the next switch");
    pn_tt_role_table_release("norm");
    CHECK(pn_tt_role_table_holder(2u, PN_TT_ROLE_MODE_DECODE) == NULL, "a released role frees its cores");
    printf("test_tt_role_table: %d checks, %d failed\n", g_checked, g_failed);
    return g_failed == 0 ? 0 : 1;
}
