/* SPDX-License-Identifier: AGPL-3.0-or-later */
/* Copyright (C) 2026 David (Corey) Herrera */
/*
 * test_tt_attn_fast.c — attention fast-math program selection (pn_tt_attn_fast.h), with NO
 * CARD and NO USE_TT.
 *
 * The knob picks which attention program every core arms, so a wrong name is a silently
 * different experiment (or a silently shipped one) on the card. Checked:
 *   - the values parse: unset / "" / "0" off; exp, lofi, both; anything else refuses (off);
 *   - every (base, mode) maps to its own variant name, OFF to the base itself, and a base
 *     with no variants (bf16, bfp8, ...) and NULL map to NULL; the six names are distinct;
 *   - the whole choice: off runs the shipped program with no message; a present variant is
 *     chosen with no message; a bad value, a base without variants and a variant missing from
 *     the program root each run the SHIPPED program and say why.
 * Build with -DPN_TT_ATTN_FAST_RED (`exp` maps to the LoFi program) to see it fail.
 */
#include "../src/accel/tt/pn_tt_attn_fast.h"

#include <stdio.h>
#include <string.h>

static int g_failed;
static int g_checks;

static void check(int ok, const char *what)
{
    g_checks++;
    if (!ok) {
        g_failed++;
        fprintf(stderr, "FAIL %s\n", what);
    }
}

static int same(const char *a, const char *b)
{
    return a && b && strcmp(a, b) == 0;
}

static void test_parse(void)
{
    pn_tt_attn_fast_mode m = PN_TT_ATTN_FAST_BOTH;

    check(pn_tt_attn_fast_parse(NULL, &m) == 0 && m == PN_TT_ATTN_FAST_OFF, "unset: off");
    check(pn_tt_attn_fast_parse("", &m) == 0 && m == PN_TT_ATTN_FAST_OFF, "empty: off");
    check(pn_tt_attn_fast_parse("0", &m) == 0 && m == PN_TT_ATTN_FAST_OFF, "0: off");
    check(pn_tt_attn_fast_parse("exp", &m) == 0 && m == PN_TT_ATTN_FAST_EXP, "exp");
    check(pn_tt_attn_fast_parse("lofi", &m) == 0 && m == PN_TT_ATTN_FAST_LOFI, "lofi");
    check(pn_tt_attn_fast_parse("both", &m) == 0 && m == PN_TT_ATTN_FAST_BOTH, "both");
    m = PN_TT_ATTN_FAST_BOTH;
    check(pn_tt_attn_fast_parse("1", &m) == -1 && m == PN_TT_ATTN_FAST_OFF, "1 refuses (off)");
    check(pn_tt_attn_fast_parse("EXP", &m) == -1 && m == PN_TT_ATTN_FAST_OFF, "EXP refuses");
    check(pn_tt_attn_fast_parse("exp,lofi", &m) == -1, "a list refuses");
    check(same(pn_tt_attn_fast_mode_name(PN_TT_ATTN_FAST_LOFI), "lofi"), "mode name");
}

static void test_names(void)
{
    static const char *const want[2][4] = {
        { "attention_k8v4", "attention_k8v4_fexp", "attention_k8v4_lofi", "attention_k8v4_fast" },
        { "attention_k8v4_split", "attention_k8v4_split_fexp", "attention_k8v4_split_lofi",
          "attention_k8v4_split_fast" },
    };
    const char *seen[6];
    int count = 0;
    int b, m, k;

    for (b = 0; b < 2; b++)
        for (m = 0; m < 4; m++) {
            const char *got = pn_tt_attn_fast_program(want[b][0], (pn_tt_attn_fast_mode)m);
            char what[128];

            snprintf(what, sizeof(what), "%s + %s -> %s", want[b][0],
                     pn_tt_attn_fast_mode_name((pn_tt_attn_fast_mode)m), want[b][m]);
            check(same(got, want[b][m]), what);
            if (m > 0 && got)
                seen[count++] = got;
        }
    for (k = 0; k < count; k++)
        for (m = k + 1; m < count; m++)
            check(!same(seen[k], seen[m]), "the six variant names are distinct");
    check(pn_tt_attn_fast_program("attention", PN_TT_ATTN_FAST_EXP) == NULL, "bf16: no variant");
    check(pn_tt_attn_fast_program("attention_bfp8", PN_TT_ATTN_FAST_BOTH) == NULL,
          "bfp8: no variant");
    check(same(pn_tt_attn_fast_program("attention_bfp8", PN_TT_ATTN_FAST_OFF), "attention_bfp8"),
          "off keeps any base");
    check(pn_tt_attn_fast_program(NULL, PN_TT_ATTN_FAST_EXP) == NULL, "NULL base");
    check(pn_tt_attn_fast_program("attention_k8v4", (pn_tt_attn_fast_mode)7) == NULL,
          "an out-of-range mode");
}

static bool all_present(const char *name)
{
    (void)name;
    return true;
}

static bool none_present(const char *name)
{
    (void)name;
    return false;
}

static void test_choose(void)
{
    char why[256];
    const char *got;

    got = pn_tt_attn_fast_choose_program(NULL, "attention_k8v4_split", all_present, why,
                                         sizeof(why));
    check(same(got, "attention_k8v4_split") && why[0] == '\0', "unset: shipped, silent");
    got = pn_tt_attn_fast_choose_program("0", "attention_k8v4_split", all_present, why,
                                         sizeof(why));
    check(same(got, "attention_k8v4_split") && why[0] == '\0', "0: shipped, silent");
    got = pn_tt_attn_fast_choose_program("exp", "attention_k8v4_split", all_present, why,
                                         sizeof(why));
    check(same(got, "attention_k8v4_split_fexp") && why[0] == '\0', "exp on split: _split_fexp");
    got = pn_tt_attn_fast_choose_program("lofi", "attention_k8v4", all_present, why, sizeof(why));
    check(same(got, "attention_k8v4_lofi") && why[0] == '\0', "lofi on lanes: _lofi");
    got = pn_tt_attn_fast_choose_program("both", "attention_k8v4", all_present, why, sizeof(why));
    check(same(got, "attention_k8v4_fast") && why[0] == '\0', "both on lanes: _fast");
    got = pn_tt_attn_fast_choose_program("fast", "attention_k8v4_split", all_present, why,
                                         sizeof(why));
    check(same(got, "attention_k8v4_split") && strstr(why, "is not 0, exp, lofi or both"),
          "a bad value: shipped, says why");
    got = pn_tt_attn_fast_choose_program("exp", "attention", all_present, why, sizeof(why));
    check(same(got, "attention") && strstr(why, "no fast-math variant"),
          "bf16 cache: shipped, says why");
    got = pn_tt_attn_fast_choose_program("both", "attention_k8v4_split", none_present, why,
                                         sizeof(why));
    check(same(got, "attention_k8v4_split") && strstr(why, "attention_k8v4_split_fast") &&
              strstr(why, "not in the program root"),
          "variant missing from the root: shipped, says why");
    got = pn_tt_attn_fast_choose_program("exp", "attention_k8v4", NULL, NULL, 0);
    check(same(got, "attention_k8v4"), "no presence check and no buffer: shipped");
}

int main(void)
{
    test_parse();
    test_names();
    test_choose();
    printf("test_tt_attn_fast: %d/%d checks passed\n", g_checks - g_failed, g_checks);
    return g_failed ? 1 : 0;
}
