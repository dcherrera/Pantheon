/* SPDX-License-Identifier: AGPL-3.0-or-later */
/* Copyright (C) 2026 David (Corey) Herrera */
/**
 * @file serve_gateway_setup.c
 * @brief Implementation of serve_gateway_setup.h.
 *
 * @status Complete
 * @issues None
 * @todo None
 */

#include "serve_gateway_setup.h"

#include <stdio.h>
#include <stdlib.h>

#include "gateway_auth.h"
#include "gateway_mode.h"
#include "model_registry.h"
#include "model_session.h"
#include "request_id.h"

int pn_serve_gateway_setup(const char *key_file, char *err, size_t errlen)
{
    pn_gateway_key key;

    if (!key_file || !key_file[0]) {
        snprintf(err, errlen, "--gateway needs --api-key-file <path>");
        return -1;
    }
    if (pn_gateway_key_load(&key, key_file, err, errlen) != 0)
        return -1;
    pn_gateway_enable(&key);
    pn_gateway_key_clear(&key);
    mr_set_sessions_evictable(true);
    ms_set_session_idle_ttl_default_ms(PN_GATEWAY_SESSION_IDLE_TTL_MS);
    srand((unsigned)pn_os_random_seed());
    return 0;
}
