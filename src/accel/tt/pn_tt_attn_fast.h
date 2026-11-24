/* SPDX-License-Identifier: AGPL-3.0-or-later */
/* Copyright (C) 2026 David (Corey) Herrera */
/**
 * @file pn_tt_attn_fast.h
 * @brief ATTENTION FAST-MATH (`PN_TT_ATTN_FAST`): which attention PROGRAM every attention core
 *        runs when a fast-math build is asked for. Pure name selection, no device: the arm
 *        glue is pn_tt_attention_fast_choose (pn_tt_attention_splitk.c).
 *
 * The foundry variants (resident_attention_compute.c, FAST-MATH BUILDS; foundry tools/
 * check_attn_fastmath_program_gates.sh) are their base program's recipe plus one or both
 * switches, and differ from it in the MATH kernel only:
 *
 *     PN_TT_ATTN_FAST   switch(es)                       attention_k8v4      attention_k8v4_split
 *     unset, "", "0"    none: the shipped program        attention_k8v4      attention_k8v4_split
 *     exp               -DATTN_FAST_EXP  bf16-grade exp  attention_k8v4_fexp attention_k8v4_split_fexp
 *     lofi              -DATTN_FAST_LOFI LoFi matmuls    attention_k8v4_lofi attention_k8v4_split_lofi
 *     both              both                             attention_k8v4_fast attention_k8v4_split_fast
 *
 * The base is whatever program the run would arm without the knob (the KV type's, or its
 * `_split` build when split-K is armed: pn_tt_attention_program_name), so the knob composes
 * with `PN_TT_ATTN_SPLITK` and `PN_TT_KV_TYPE`. Only K8V4 has variants.
 *
 * DECODE TOO. A load arms ONE attention program on every attention core, so the variant runs
 * decode, verify rows and prefill alike; there is no prefill-only selection.
 *
 * REFUSES LOUDLY, RUNS THE SHIPPED PROGRAM. An unknown value, a base with no variant, or a
 * variant missing from the program root each leave the shipped program armed and say so on
 * stderr: the run is still correct, and the line says it is not the run that was asked for.
 *
 * @status Opt-in (default off: the program, and so every byte on the card, is unchanged).
 *         Red `PN_TT_ATTN_FAST_RED`: `exp` maps to the `lofi` program -- must FAIL.
 * @issues None
 * @todo None
 */
#ifndef PN_TT_ATTN_FAST_H
#define PN_TT_ATTN_FAST_H

#include <stdbool.h>
#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif

#define PN_TT_ATTN_FAST_ENV "PN_TT_ATTN_FAST"

/** What `PN_TT_ATTN_FAST` asks for. */
typedef enum pn_tt_attn_fast_mode {
    PN_TT_ATTN_FAST_OFF = 0,
    PN_TT_ATTN_FAST_EXP = 1,  /**< -DATTN_FAST_EXP */
    PN_TT_ATTN_FAST_LOFI = 2, /**< -DATTN_FAST_LOFI */
    PN_TT_ATTN_FAST_BOTH = 3  /**< both */
} pn_tt_attn_fast_mode;

/** Is program `name` in the program root? (pn_tt_program_is_present on the card.) */
typedef bool (*pn_tt_attn_fast_present_fn)(const char *name);

/**
 * Parse a `PN_TT_ATTN_FAST` value: NULL, "" or "0" off; "exp", "lofi", "both".
 * @return 0, or -1 for anything else (*mode is then OFF)
 *
 * Thread-safe: yes    Reentrant: yes    Signal-safe: yes
 */
int pn_tt_attn_fast_parse(const char *value, pn_tt_attn_fast_mode *mode);

/** "off", "exp", "lofi", "both". */
const char *pn_tt_attn_fast_mode_name(pn_tt_attn_fast_mode mode);

/**
 * The variant of program `base` for `mode`: `base` itself for OFF, NULL when `base` has no
 * such variant (or is NULL).
 *
 * Thread-safe: yes    Reentrant: yes    Signal-safe: yes
 */
const char *pn_tt_attn_fast_program(const char *base, pn_tt_attn_fast_mode mode);

/**
 * The whole choice: the program to arm for knob value `value` over the shipped program
 * `shipped`. Returns the variant when the value parses, `shipped` has one and `present` finds
 * it; otherwise `shipped`, with the refusal written to `why` (empty when nothing was refused
 * or the knob is off). `why` may be NULL.
 *
 * Thread-safe: yes (if `present` is)    Reentrant: yes    Signal-safe: no (snprintf)
 */
const char *pn_tt_attn_fast_choose_program(const char *value, const char *shipped,
                                           pn_tt_attn_fast_present_fn present, char *why,
                                           size_t why_size);

#ifdef __cplusplus
}
#endif

#endif /* PN_TT_ATTN_FAST_H */
