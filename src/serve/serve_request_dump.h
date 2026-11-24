/* SPDX-License-Identifier: AGPL-3.0-or-later */
/* Copyright (C) 2026 David (Corey) Herrera */
/**
 * @file serve_request_dump.h
 * @brief Developer capture of chat request bodies, so real client traffic can be replayed
 *        byte for byte against another build (`PN_SERVE_REQ_DUMP=<dir>`; unset = nothing).
 *
 * Each /v1/chat/completions body is written as `<dir>/req_<unix ms>_<n>.json`, exactly as
 * received. Only the BODY is written: headers (Authorization) never reach this code. The
 * directory must exist; a write failure is logged once a request and never fails the request.
 *
 * status: used by openai_chat.c (oai_handle_chat).
 * issues: none.
 * todo:   none.
 */
#ifndef PN_SERVE_REQUEST_DUMP_H
#define PN_SERVE_REQUEST_DUMP_H

/** Write `body` to the dump directory when `PN_SERVE_REQ_DUMP` names one. */
void sg_request_dump_body(const char *body);

#endif /* PN_SERVE_REQUEST_DUMP_H */
