/* SPDX-License-Identifier: AGPL-3.0-or-later */
/* Copyright (C) 2026 David (Corey) Herrera */
/**
 * @file serve_dump_reqlog.h
 * @brief The serve's side of the hidden-dump request log (`PN_TT_DUMP_REQLOG`,
 *        core/dump_reqlog.h): hash the request's prompt tokens and write its line.
 *
 * status: used by serve_generation.c (sg_run_completion, the single-user path the TT serve runs).
 * issues: none.
 * todo:   none.
 */
#ifndef PN_SERVE_DUMP_REQLOG_H
#define PN_SERVE_DUMP_REQLOG_H

#include <stdint.h>

#include "../core/dump_reqlog.h"

/** Fill r->prompt_sha256_hex with sha256 over `prompt` (int32 little-endian ids) and write the
 *  request's line. Call only when pn_dump_reqlog_enabled(). @return 0, -1 on a write error. */
int sg_dump_reqlog_request(pn_dump_reqlog_request *r, const uint32_t *prompt, int n_prompt);

#endif /* PN_SERVE_DUMP_REQLOG_H */
