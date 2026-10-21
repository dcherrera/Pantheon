/* SPDX-License-Identifier: AGPL-3.0-or-later */
/* Copyright (C) 2026 David (Corey) Herrera */
/**
 * @file registry_client.c
 * @brief Implementation of the serve-side self-registration client (see
 *        registry_client.h). Env-gated on PANTHEON_REGISTRY_URL; when unset every
 *        call is a no-op so serve is unchanged. Builds the register / heartbeat
 *        / deregister JSON bodies (per registry_wire.h) with json_writer and
 *        sends them via http_client — no new deps.
 *
 * @status G0 client.
 * @issues None
 * @todo None
 */

#include "registry_client.h"

#include <stdlib.h>
#include <string.h>

#include "http_client.h"
#include "json_writer.h"

#define RC_CONNECT_MS 1500

/* Read the registry base URL once (e.g. "http://127.0.0.1:8100"). NULL ⇒ off. */
static const char *registry_url(void)
{
    return getenv("PANTHEON_REGISTRY_URL");
}

bool rc_enabled(void)
{
    const char *u = registry_url();
    return u && u[0];
}

/* Send a JSON body to $PANTHEON_REGISTRY_URL<path> with `method`. Returns true on
 * a 2xx, false on transport error or non-2xx. Frees the writer. */
static bool send_json(const char *method, const char *path, json_writer *jw)
{
    const char *base = registry_url();
    if (jw->oom) { jw_free(jw); return false; }
    int status = 0;
    hc_result rc = hc_request_simple(method, base, path, jw->data, jw->len,
                                     RC_CONNECT_MS, &status);
    jw_free(jw);
    return (rc == HC_OK && status >= 200 && status < 300);
}

bool rc_register(const rc_identity *id)
{
    if (!rc_enabled()) return true;   /* disabled ⇒ no-op success */

    json_writer jw;
    if (!jw_init(&jw)) return false;
    jw_raw(&jw, "{\"model_id\":");
    jw_string(&jw, id->model_id);
    jw_raw(&jw, ",\"instance_id\":");
    jw_string(&jw, id->instance_id);
    jw_raw(&jw, ",\"node_id\":");
    jw_string(&jw, id->node_id);
    jw_raw(&jw, ",\"endpoint\":");
    jw_string(&jw, id->endpoint);
    jw_raw(&jw, ",\"transport\":");
    jw_string(&jw, id->transport[0] ? id->transport : RW_TRANSPORT_LOCAL);
    jw_printf(&jw, ",\"capacity\":%d", id->capacity);
    if (id->context_window > 0)
        jw_printf(&jw, ",\"context_window\":%d", id->context_window);
    jw_raw(&jw, "}");
    return send_json("POST", "/registry/register", &jw);
}

bool rc_heartbeat(const char *instance_id, const rc_load *load)
{
    if (!rc_enabled()) return true;

    json_writer jw;
    if (!jw_init(&jw)) return false;
    jw_raw(&jw, "{\"instance_id\":");
    jw_string(&jw, instance_id);
    jw_printf(&jw, ",\"load\":{\"active_slots\":%d,\"queue_depth\":%d,"
                   "\"free_slots\":%d",
              load->active_slots, load->queue_depth, load->free_slots);
    if (load->context_window > 0)
        jw_printf(&jw, ",\"context_window\":%d", load->context_window);
    jw_printf(&jw, "},\"healthy\":%s}", load->healthy ? "true" : "false");
    return send_json("POST", "/registry/heartbeat", &jw);
}

bool rc_deregister(const char *instance_id)
{
    if (!rc_enabled()) return true;

    json_writer jw;
    if (!jw_init(&jw)) return false;
    jw_raw(&jw, "{\"instance_id\":");
    jw_string(&jw, instance_id);
    jw_raw(&jw, "}");
    return send_json("DELETE", "/registry/deregister", &jw);
}
