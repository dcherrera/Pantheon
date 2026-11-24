/* SPDX-License-Identifier: AGPL-3.0-or-later */
/* Copyright (C) 2026 David (Corey) Herrera */
/*
 * test_host_admission.c — host-RAM admission (pn_host_admission.h) with a fake
 * MemAvailable: fits, refuses with the exact message, the reserve override,
 * cumulative accounting (lazy vs touched), the disable switch, and the
 * default-cap fit helper.
 */

#include "../src/core/mem/pn_host_admission.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static int g_failed = 0;

#define CHECK(cond, msg) do { \
    if (!(cond)) { \
        fprintf(stderr, "  FAIL: %s (%s:%d)\n", (msg), __FILE__, __LINE__); \
        g_failed++; \
    } \
} while (0)

#define MIB(n) ((uint64_t)(n) << 20)
#define GIB_F(x) ((uint64_t)((x) * 1024.0) << 20)

static void reset_env(void)
{
    unsetenv("PN_HOST_ADMIT");
    unsetenv("PN_HOST_RESERVE_MB");
    unsetenv("PN_HOST_ADMIT_TEST_AVAIL_MB");
    pn_host_admit_test_reset();
}

static void test_fits(void)
{
    reset_env();
    pn_host_admit_test_set_memory(MIB(4096), 0);
    char reason[PN_HOST_ADMIT_REASON_LEN];
    CHECK(pn_host_admit(MIB(2048), "host KV ring", reason, sizeof(reason)) == 0,
          "2 GiB fits in 4 GiB available less a 1.5 GiB reserve");
    CHECK(reason[0] == '\0', "no reason on admission");
    CHECK(pn_host_admitted_bytes() == MIB(2048), "admission is charged");
    pn_host_release(MIB(2048));
    CHECK(pn_host_admitted_bytes() == 0, "release returns the charge");
}

static void test_refuses_with_exact_message(void)
{
    reset_env();
    /* The CLI case from 2026-10-27, scaled to the example in the spec. */
    pn_host_admit_test_set_memory(GIB_F(2.1), 0);
    char reason[PN_HOST_ADMIT_REASON_LEN];
    int rc = pn_host_admit(GIB_F(4.6), "host KV ring", reason, sizeof(reason));
    CHECK(rc == -1, "4.6 GiB does not fit in 2.1 GiB");
    const char *want = "host KV ring needs 4.6 GiB, 2.1 GiB available "
                       "(reserve 1.5 GiB); lower --ctx-cap or --max-sessions";
    if (strcmp(reason, want) != 0)
        fprintf(stderr, "  got:  %s\n  want: %s\n", reason, want);
    CHECK(strcmp(reason, want) == 0, "refusal message is exact");
    CHECK(pn_host_admitted_bytes() == 0, "a refusal charges nothing");

    rc = pn_host_admit_hint(MIB(700), "logits buffer", "lower --max-sessions",
                            reason, sizeof(reason));
    CHECK(rc == -1, "700 MiB does not fit above a 1.5 GiB reserve in 2.1 GiB");
    CHECK(strstr(reason, "needs 700 MiB") != NULL, "sub-GiB sizes print in MiB");
    CHECK(strstr(reason, "; lower --max-sessions") != NULL, "custom hint used");
}

static void test_reserve_override(void)
{
    reset_env();
    pn_host_admit_test_set_memory(MIB(3000), 0);
    char reason[PN_HOST_ADMIT_REASON_LEN];
    CHECK(pn_host_admit_reserve_bytes() == MIB(1536), "default reserve 1.5 GiB");
    CHECK(pn_host_admit(MIB(2000), "x", reason, sizeof(reason)) == -1,
          "2000 MiB refused above the default reserve");
    setenv("PN_HOST_RESERVE_MB", "512", 1);
    CHECK(pn_host_admit_reserve_bytes() == MIB(512), "PN_HOST_RESERVE_MB read");
    CHECK(pn_host_admit(MIB(2000), "x", reason, sizeof(reason)) == 0,
          "2000 MiB admitted with a 512 MiB reserve");
    CHECK(strstr(reason, "reserve") == NULL, "no reason when admitted");
    setenv("PN_HOST_RESERVE_MB", "garbage", 1);
    CHECK(pn_host_admit_reserve_bytes() == MIB(1536), "garbage -> default");
}

static void test_cumulative_accounting(void)
{
    reset_env();
    /* 6 GiB available, 1.5 GiB reserve: 4.5 GiB of room. Lazy allocations
     * (anon RSS does not move) must add up. */
    pn_host_admit_test_set_memory(MIB(6144), MIB(100));
    char reason[PN_HOST_ADMIT_REASON_LEN];
    CHECK(pn_host_admit(MIB(2048), "s1", reason, sizeof(reason)) == 0, "1st fits");
    CHECK(pn_host_admit(MIB(2048), "s2", reason, sizeof(reason)) == 0, "2nd fits");
    CHECK(pn_host_admit(MIB(2048), "s3", reason, sizeof(reason)) == -1,
          "3rd refused: 6 GiB of untouched admissions do not fit 4.5 GiB");
    CHECK(strstr(reason, "2.0 GiB available") != NULL,
          "available is net of what this process still owes");

    /* Now the first two are touched: RSS grew by 4 GiB and MemAvailable fell
     * by 4 GiB. They must not be charged a second time. */
    pn_host_admit_test_set_memory(MIB(1536 + 512 + 512), MIB(100 + 4096));
    CHECK(pn_host_admit(MIB(512), "s3", reason, sizeof(reason)) == 0,
          "touched memory is not double-counted");
    CHECK(pn_host_admit(MIB(1024), "s4", reason, sizeof(reason)) == -1,
          "but the untouched 512 MiB still is");

    /* Release: the slot is freed, the charge goes. */
    pn_host_release(MIB(512));
    CHECK(pn_host_admit(MIB(512), "s4", reason, sizeof(reason)) == 0,
          "released bytes are admissible again");
}

static void test_disable_switch(void)
{
    reset_env();
    pn_host_admit_test_set_memory(MIB(100), 0);
    setenv("PN_HOST_ADMIT", "0", 1);
    char reason[PN_HOST_ADMIT_REASON_LEN];
    CHECK(!pn_host_admit_enabled(), "PN_HOST_ADMIT=0 disables");
    CHECK(pn_host_admit(MIB(100000), "x", reason, sizeof(reason)) == 0,
          "disabled admits anything");
    CHECK(pn_host_admit_headroom() == UINT64_MAX, "disabled headroom unbounded");
    setenv("PN_HOST_ADMIT", "1", 1);
    CHECK(pn_host_admit_enabled(), "any other value keeps it on");
}

static void test_env_test_hook(void)
{
    reset_env();
    setenv("PN_HOST_ADMIT_TEST_AVAIL_MB", "2048", 1);
    CHECK(pn_host_admit_headroom() == MIB(512), "env hook pins MemAvailable");
    unsetenv("PN_HOST_ADMIT_TEST_AVAIL_MB");
}

static void test_fit_units(void)
{
    reset_env();
    /* 4 GiB room after the reserve. A 40,000-position ring at 112 KiB/token
     * (qwen3-0.6b K+V fp32) = 4.3 GiB does not fit; the cap lands below it. */
    pn_host_admit_test_set_memory(MIB(4096 + 1536), 0);
    uint64_t per_tok = 2ull * 28 * 8 * 128 * 4;
    long n = pn_host_admit_fit_units(MIB(64), per_tok, 40000, 256);
    CHECK(n > 0 && n < 40000, "default ring capped to what fits");
    CHECK((uint64_t)MIB(64) + (uint64_t)n * per_tok <= MIB(4096), "cap fits");
    CHECK((uint64_t)MIB(64) + (uint64_t)(n + 1) * per_tok > MIB(4096),
          "cap is the largest that fits");
    CHECK(pn_host_admit_fit_units(MIB(64), per_tok, 4096, 256) == 4096,
          "a window that fits is untouched");
    CHECK(pn_host_admit_fit_units(MIB(8192), per_tok, 4096, 256) == 0,
          "fixed cost alone too big -> 0");
}

int main(void)
{
    printf("test_host_admission\n");
    test_fits();
    test_refuses_with_exact_message();
    test_reserve_override();
    test_cumulative_accounting();
    test_disable_switch();
    test_env_test_hook();
    test_fit_units();
    reset_env();
    if (g_failed) {
        printf("test_host_admission: %d FAILED\n", g_failed);
        return 1;
    }
    printf("test_host_admission: all passed\n");
    return 0;
}
