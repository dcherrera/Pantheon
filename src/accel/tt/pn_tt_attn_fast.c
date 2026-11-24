/* SPDX-License-Identifier: AGPL-3.0-or-later */
/* Copyright (C) 2026 David (Corey) Herrera */
/**
 * @file pn_tt_attn_fast.c
 * @brief Attention fast-math program selection. See pn_tt_attn_fast.h.
 * @status Opt-in (default off).
 * @issues None
 * @todo None
 */
#include "pn_tt_attn_fast.h"

#include <stdio.h>
#include <string.h>

/** A base program and its three variants, indexed by mode - 1. */
typedef struct attn_fast_family {
    const char *base;
    const char *variant[3];
} attn_fast_family;

static const attn_fast_family k_families[] = {
#ifdef PN_TT_ATTN_FAST_RED
    /* RED: `exp` arms the LoFi program -- a silently different experiment. */
    { "attention_k8v4", { "attention_k8v4_lofi", "attention_k8v4_lofi", "attention_k8v4_fast" } },
#else
    { "attention_k8v4", { "attention_k8v4_fexp", "attention_k8v4_lofi", "attention_k8v4_fast" } },
#endif
    { "attention_k8v4_split",
      { "attention_k8v4_split_fexp", "attention_k8v4_split_lofi", "attention_k8v4_split_fast" } },
};

int pn_tt_attn_fast_parse(const char *value, pn_tt_attn_fast_mode *mode)
{
    pn_tt_attn_fast_mode parsed = PN_TT_ATTN_FAST_OFF;

    if (!value || !*value || strcmp(value, "0") == 0)
        parsed = PN_TT_ATTN_FAST_OFF;
    else if (strcmp(value, "exp") == 0)
        parsed = PN_TT_ATTN_FAST_EXP;
    else if (strcmp(value, "lofi") == 0)
        parsed = PN_TT_ATTN_FAST_LOFI;
    else if (strcmp(value, "both") == 0)
        parsed = PN_TT_ATTN_FAST_BOTH;
    else {
        if (mode)
            *mode = PN_TT_ATTN_FAST_OFF;
        return -1;
    }
    if (mode)
        *mode = parsed;
    return 0;
}

const char *pn_tt_attn_fast_mode_name(pn_tt_attn_fast_mode mode)
{
    static const char *const names[] = { "off", "exp", "lofi", "both" };

    return (unsigned)mode < sizeof(names) / sizeof(names[0]) ? names[mode] : "?";
}

const char *pn_tt_attn_fast_program(const char *base, pn_tt_attn_fast_mode mode)
{
    size_t i;

    if (!base)
        return NULL;
    if (mode == PN_TT_ATTN_FAST_OFF)
        return base;
    if (mode < PN_TT_ATTN_FAST_EXP || mode > PN_TT_ATTN_FAST_BOTH)
        return NULL;
    for (i = 0; i < sizeof(k_families) / sizeof(k_families[0]); i++)
        if (strcmp(k_families[i].base, base) == 0)
            return k_families[i].variant[mode - 1];
    return NULL;
}

const char *pn_tt_attn_fast_choose_program(const char *value, const char *shipped,
                                           pn_tt_attn_fast_present_fn present, char *why,
                                           size_t why_size)
{
    pn_tt_attn_fast_mode mode;
    const char *variant;

    if (why && why_size)
        why[0] = '\0';
    if (pn_tt_attn_fast_parse(value, &mode) != 0) {
        if (why && why_size)
            snprintf(why, why_size, "%s=%s is not 0, exp, lofi or both -- the shipped `%s` "
                                    "runs", PN_TT_ATTN_FAST_ENV, value, shipped ? shipped : "?");
        return shipped;
    }
    if (mode == PN_TT_ATTN_FAST_OFF)
        return shipped;
    variant = pn_tt_attn_fast_program(shipped, mode);
    if (!variant) {
        if (why && why_size)
            snprintf(why, why_size, "%s=%s, but `%s` has no fast-math variant (K8V4 only) -- "
                                    "it runs as shipped", PN_TT_ATTN_FAST_ENV, value,
                     shipped ? shipped : "?");
        return shipped;
    }
    if (!present || !present(variant)) {
        if (why && why_size)
            snprintf(why, why_size, "%s=%s asks for `%s`, which is not in the program root -- "
                                    "the shipped `%s` runs", PN_TT_ATTN_FAST_ENV, value,
                     variant, shipped);
        return shipped;
    }
    return variant;
}
