/* SPDX-License-Identifier: AGPL-3.0-or-later */
/* Copyright (C) 2026 David (Corey) Herrera */
/**
 * @file gateway_request_gate.h
 * @brief The check every request on the API listener passes in gateway mode,
 *        before routing: allowed route, gateway bearer key, well-formed tenant.
 *
 * @status Complete
 * @issues None
 * @todo None
 */
#ifndef PANTHEON_SERVE_GATEWAY_REQUEST_GATE_H
#define PANTHEON_SERVE_GATEWAY_REQUEST_GATE_H

#include "http_server.h"

/**
 * Gate one API-listener request in gateway mode. On success sets this thread's
 * gateway tenant (from X-Pantheon-Tenant; NULL when absent) and returns 0. On
 * refusal answers the client in OpenAI shape and returns non-zero:
 *   404 unknown_route    any route outside pn_gateway_route_allowed (checked
 *                        first, so admin paths look absent, not protected)
 *   401 invalid_api_key  missing or wrong `Authorization: Bearer`
 *   400 invalid_tenant   an X-Pantheon-Tenant that pn_gateway_tenant_valid refuses
 *
 * Thread-safe: yes. Reentrant: yes. Signal-safe: no
 */
int pn_gateway_gate_request(int client_fd, const hs_request *req);

#endif /* PANTHEON_SERVE_GATEWAY_REQUEST_GATE_H */
