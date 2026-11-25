/* SPDX-License-Identifier: AGPL-3.0-or-later */
/* Copyright (C) 2026 David (Corey) Herrera */
/**
 * @file gateway_request_gate.c
 * @brief Implementation of gateway_request_gate.h.
 *
 * @status Complete
 * @issues None
 * @todo None
 */

#include "gateway_request_gate.h"

#include "gateway_mode.h"

int pn_gateway_gate_request(int client_fd, const hs_request *req)
{
    pn_gateway_set_thread_tenant(NULL);
    if (!pn_gateway_route_allowed(req->method, req->path)) {
        hs_send_openai_error(client_fd, 404, "invalid_request_error", "unknown_route",
                             NULL, "unknown route");
        return 1;
    }
    if (!pn_gateway_authorized(req->authorization)) {
        hs_send_openai_error(client_fd, 401, "invalid_request_error", "invalid_api_key",
                             NULL, "missing or invalid API key");
        return 1;
    }
    if (req->tenant && !pn_gateway_tenant_valid(req->tenant)) {
        hs_send_openai_error(client_fd, 400, "invalid_request_error", "invalid_tenant",
                             NULL, "X-Pantheon-Tenant must be 1-48 characters of "
                                   "A-Z a-z 0-9 . _ -");
        return 1;
    }
    pn_gateway_set_thread_tenant(req->tenant);
    return 0;
}
