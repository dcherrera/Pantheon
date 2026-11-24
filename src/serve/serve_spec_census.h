/* SPDX-License-Identifier: AGPL-3.0-or-later */
/* Copyright (C) 2026 David (Corey) Herrera */
/**
 * @file serve_spec_census.h
 * @brief Where a serve request's tokens came from: speculative verify commits or plain
 *        one-token forwards, and WHY each plain forward happened.
 *
 * The single-user Tenstorrent serve decodes a request either speculatively (MTP drafts
 * verified in one pass, several tokens a forward) or plainly (one token a forward). A request
 * can be refused speculation outright (logit bias, logprobs, a dense-row grammar, greedy with
 * penalties, ...) or speculate and still take plain forwards at single positions (the
 * reasoning budget's trip, the token budget's edge, no draft surviving the cuts). The census
 * names both, so the share of real traffic that falls back can be counted from a serve log.
 *
 * One line a request on stderr, always (it is cheap and the serve log is where the counts are
 * read):
 *   [spec] census: mode=<mode> gen=<tokens> verify=<committed by a verify> plain=<forwards>
 *                  (trip <n> cap <n> nodraft <n> failed <n> off <n>)
 *                  | free <tokens> tok <ms> ms | masked <tokens> tok <ms> ms
 * The last part splits the decode loop's wall time by whether the grammar mask was active on
 * the step (masked: inside a constrained tool call / JSON answer; free: everything else).
 *
 * status: used by serve_generation.c (sg_run_completion, USE_TT).
 * issues: none.
 * todo:   none.
 */
#ifndef PN_SERVE_SPEC_CENSUS_H
#define PN_SERVE_SPEC_CENSUS_H

/** Why a plain forward was taken in place of a verify cycle. */
typedef enum {
    SG_PLAIN_OFF = 0,     /**< speculation not armed for this request (see mode) */
    SG_PLAIN_TRIP,        /**< the reasoning budget tripped: the force-close needs logits */
    SG_PLAIN_CAP,         /**< no room for a draft under the token / reasoning budget */
    SG_PLAIN_NODRAFT,     /**< every draft was cut (stop token, grammar) or none proposed */
    SG_PLAIN_FAILED,      /**< a verify failed; plain decode for the rest of the request */
    SG_PLAIN_REASONS
} sg_plain_reason;

typedef struct {
    const char *mode;                       /**< "greedy", "sampled", "grammar-…" or "off:<why>" */
    unsigned long plain[SG_PLAIN_REASONS];  /**< plain forwards by reason */
    unsigned long from_verify;              /**< tokens emitted that a verify already committed */
    unsigned long region_tokens[2];         /**< decode-loop steps: [0] free, [1] mask active */
    double region_ms[2];                    /**< their wall time */
} sg_spec_census;

/** Reset; `mode` is a static string naming how the request decodes. */
void sg_spec_census_begin(sg_spec_census *c, const char *mode);

/** One plain forward taken for `reason`. */
void sg_spec_census_plain(sg_spec_census *c, sg_plain_reason reason);

/** One emitted token that a verify cycle had committed (no forward of its own). */
void sg_spec_census_verified(sg_spec_census *c);

/** One decode-loop step of `ms` wall time, under an active grammar mask or not. */
void sg_spec_census_step(sg_spec_census *c, int masked, double ms);

/** Print the request's census line (stderr); `generated` = tokens the request emitted. */
void sg_spec_census_report(const sg_spec_census *c, int generated);

#endif /* PN_SERVE_SPEC_CENSUS_H */
