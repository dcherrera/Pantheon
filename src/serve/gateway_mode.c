/* SPDX-License-Identifier: AGPL-3.0-or-later */
/* Copyright (C) 2026 David (Corey) Herrera */
/**
 * @file gateway_mode.c
 * @brief Implementation of gateway_mode.h.
 *
 * @status Complete
 * @issues None
 * @todo None
 */

#include "gateway_mode.h"

#include <stdio.h>
#include <string.h>

static bool           g_enabled;
static pn_gateway_key g_key;

void pn_gateway_enable(const pn_gateway_key *key)
{
    g_key = *key;
    g_enabled = true;
}

void pn_gateway_disable(void)
{
    pn_gateway_key_clear(&g_key);
    g_enabled = false;
}

bool pn_gateway_enabled(void)
{
    return g_enabled;
}

bool pn_gateway_authorized(const char *authorization)
{
    return g_enabled && pn_gateway_key_matches(&g_key, authorization);
}

/* `path` up to its query string equals `want`. */
static bool path_is(const char *path, const char *want)
{
    size_t n = strcspn(path, "?");
    return strlen(want) == n && strncmp(path, want, n) == 0;
}

bool pn_gateway_route_allowed(const char *method, const char *path)
{
    if (!method || !path)
        return false;
    if (strcmp(method, "POST") == 0)
        return path_is(path, "/v1/chat/completions");
    if (strcmp(method, "GET") == 0)
        return path_is(path, "/v1/models") || path_is(path, "/health/live") ||
               path_is(path, "/health/ready");
    return false;
}

bool pn_gateway_tenant_valid(const char *tenant)
{
    size_t n = tenant ? strlen(tenant) : 0;

    if (n == 0 || n > PN_GATEWAY_TENANT_MAX)
        return false;
    for (size_t i = 0; i < n; i++) {
        char c = tenant[i];
        bool ok = (c >= 'a' && c <= 'z') || (c >= 'A' && c <= 'Z') ||
                  (c >= '0' && c <= '9') || c == '.' || c == '_' || c == '-';
        if (!ok)
            return false;
    }
    return true;
}

/* FNV-1a 64: a stable stand-in for a key too long for the slot tables. */
static uint64_t fnv1a64(const char *s)
{
    uint64_t h = 0xcbf29ce484222325ull;
    for (; *s; s++) {
        h ^= (unsigned char)*s;
        h *= 0x100000001b3ull;
    }
    return h;
}

const char *pn_gateway_session_key(const char *tenant, const char *client_key,
                                   char *out, size_t cap)
{
    if (!out || cap < PN_GATEWAY_SESSION_KEY_MAX || !pn_gateway_tenant_valid(tenant) ||
        !client_key || !client_key[0])
        return NULL;
    if (strlen(tenant) + 1 + strlen(client_key) < PN_GATEWAY_SESSION_KEY_MAX)
        snprintf(out, cap, "%s:%s", tenant, client_key);
    else
        snprintf(out, cap, "%s:#%016llx", tenant,
                 (unsigned long long)fnv1a64(client_key));
    return out;
}

static _Thread_local char t_tenant[PN_GATEWAY_TENANT_MAX + 1];

void pn_gateway_set_thread_tenant(const char *tenant)
{
    if (!pn_gateway_tenant_valid(tenant))
        tenant = "";
    snprintf(t_tenant, sizeof(t_tenant), "%s", tenant);
}

const char *pn_gateway_thread_tenant(void)
{
    return t_tenant[0] ? t_tenant : NULL;
}
