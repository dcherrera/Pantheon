/* SPDX-License-Identifier: AGPL-3.0-or-later */
/* Copyright (C) 2026 David (Corey) Herrera */
/**
 * @file   local_kv_policy.h
 * @brief  Pantheon Local's KV-format policy, one switch (David, spec §14).
 *         0 (decided 2026-10-28): each catalog profile picks its KV through
 *         `options.kv_type`: "q4" (TurboQuant Q4, the K80 default, lossless in our
 *         tests) or "fp32" (full precision, slower). The box passes that choice to the
 *         load (pn_load_opts.kv_format); the serve never switches an explicit choice
 *         (the memory ladder refuses an fp32 cache it would have made Q4). A profile
 *         without a kv_type gets the serve default (Q4 on the K80 qwen35 path).
 *         1: fp32 only — `pantheon local` pins QWEN35_KV_Q4=0 and PN_LADDER_KV_Q4=0
 *         at start and the catalog parser refuses a "q4" profile.
 *
 * @status None
 * @issues None
 * @todo   None
 */

#ifndef PN_LOCAL_KV_POLICY_H
#define PN_LOCAL_KV_POLICY_H

#ifndef PN_LOCAL_KV_FP32_ONLY
#define PN_LOCAL_KV_FP32_ONLY 0   /* David: 0 = the profile picks (Q4 default), 1 = fp32 only */
#endif

#endif /* PN_LOCAL_KV_POLICY_H */
