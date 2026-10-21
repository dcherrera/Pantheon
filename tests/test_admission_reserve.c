/* SPDX-License-Identifier: AGPL-3.0-or-later */
/* Copyright (C) 2026 David (Corey) Herrera */
/*
 * test_admission_reserve.c — the per-die reserve is charged exactly once.
 *
 * The die manager sets vram_budget = total - reserve, so anything derived from
 * it has already paid the reserve. The RAW entry point holds it back again, by
 * contract, because it takes raw hardware free. Handing it a budget-derived
 * figure therefore charges 1 GiB twice — which refused a 2-die Qwen3.6-35B-A3B
 * Q4_K_M load whose own printed numbers said it fit.
 *
 * The pair under test is the GPU pair — pn_admit_model_load_die and its
 * _budgeted twin — because that is the pair production actually uses: the split
 * and single-die gates take raw free, mr_load takes a die-manager budget. The
 * CPU-resident gate, pn_admit_model_load, carries the host activation LUT and
 * deliberately has no budgeted twin: an accelerator load must not be priced
 * through it at all, which is the other half of this bug's family.
 */

#include "../src/core/mem/pn_admission.h"

#include <stdio.h>

uint64_t pn_die_reserve_bytes(void);

static int g_failed = 0;

#define CHECK(cond, msg) do { \
    if (!(cond)) { \
        fprintf(stderr, "  FAIL: %s (%s:%d)\n", (msg), __FILE__, __LINE__); \
        g_failed++; \
    } \
} while (0)

#define MIB(n) ((uint64_t)(n) << 20)

int main(void)
{
    printf("test_admission_reserve\n");
    uint64_t reserve = pn_die_reserve_bytes();
    CHECK(reserve > 0, "there is a per-die reserve to get wrong");

    /* Both entry points must agree on WHAT IS NEEDED. Only the free figure they
     * are handed differs. */
    uint64_t need_raw = 0, need_budgeted = 0;
    pn_admit_model_load_die(MIB(10255), MIB(12), 2048, MIB(1), &need_raw);
    pn_admit_model_load_die_budgeted(MIB(10255), MIB(12), 2048, MIB(1),
                                     &need_budgeted);
    CHECK(need_raw == need_budgeted,
          "the projection is the same; only the reserve handling differs");

    /* The K80 test machine case: die budget 11,182 MiB (12,206 total less a 1 GiB reserve),
     * against a die holding 10,255 MiB of weights, 12 MiB of KV at a 2048 cap
     * and the 576 MiB scratch that window reserves -- 10,843 MiB. It fits, and
     * must be admitted. */
    uint64_t need = 0;
    uint64_t budget_free = MIB(11182);
    CHECK(pn_admit_model_load_die_budgeted(MIB(10255), MIB(12), 2048,
                                           budget_free, &need),
          "a die whose remaining budget exceeds the need admits");
    CHECK(need == MIB(10255) + MIB(12) + MIB(576),
          "and the projection is weights + KV + scratch, with no host LUT");
    CHECK(need < budget_free, "and the printed numbers agree it fits");

    /* Same numbers through the RAW entry point are correctly refused: it holds
     * the reserve back, so passing an already-net figure double-charges. That is
     * the bug, reproduced. */
    CHECK(!pn_admit_model_load_die(MIB(10255), MIB(12), 2048, budget_free,
                                   &need),
          "the raw entry point subtracts the reserve again -- the old bug");

    /* The raw entry point is right when it IS given raw free. */
    CHECK(pn_admit_model_load_die(MIB(10255), MIB(12), 2048,
                                  budget_free + reserve, &need),
          "raw free (budget + reserve) admits the same model");

    /* The budgeted gate still refuses what genuinely does not fit -- it drops a
     * double charge, not the check. */
    CHECK(!pn_admit_model_load_die_budgeted(MIB(10255), MIB(12), 2048,
                                            need - MIB(1), &need),
          "one MiB short is still a refusal");
    CHECK(!pn_admit_model_load_die_budgeted(MIB(10255), MIB(12), 2048, 0, &need),
          "no budget left is a refusal");

    /* The CPU-resident gate is a DIFFERENT projection, not a naming variant: at
     * a wide window its host activation LUT dwarfs everything else. This is the
     * regression that would have refused the deployed config -- the same die, the same weights,
     * admitted as a GPU load and refused as a CPU one. */
    uint64_t gpu_need = 0, cpu_need = 0;
    CHECK(pn_admit_model_load_die_budgeted(MIB(7425), MIB(369), 131072,
                                           budget_free, &gpu_need),
          "the deployed config's shape at a 131072 cap admits as the GPU load it is");
    pn_admit_model_load(MIB(7425), 131072, true, 0, budget_free, &cpu_need);
    CHECK(cpu_need > gpu_need + MIB(40000),
          "the CPU projection is tens of GiB larger -- the LUT, on a GPU load");

    if (g_failed) {
        printf("FAILED: %d check(s)\n", g_failed);
        return 1;
    }
    printf("ALL PASS\n");
    return 0;
}
