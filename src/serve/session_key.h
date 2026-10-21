/* SPDX-License-Identifier: AGPL-3.0-or-later */
/* Copyright (C) 2026 David (Corey) Herrera */
/**
 * @file session_key.h
 * @brief The prefix-cache session key a chat request body carries, under any of
 *        the names clients actually send.
 *
 * @status None
 * @issues None
 * @todo None
 */
/* WHY ALIASES
 *   Without a session key the serve caches nothing and every turn re-prefills the
 *   whole conversation, which on a K80 is minutes per turn on a long context.
 *   Pantheon's own field is `session_id`. OpenAI's API calls the same idea
 *   `prompt_cache_key`, so clients built on its SDK set that. opencode sends its
 *   session id in an `x-session-id` header on every request (covered by the header
 *   fallback) and, with `setCacheKey: true` in its provider config, also as
 *   `promptCacheKey` in the body. Accepting all three lets those clients hit the
 *   cache with no Pantheon-specific setup.
 *
 *   Order: `session_id`, then `prompt_cache_key`, then `promptCacheKey`; the first
 *   non-empty string wins. The X-Session-Id header stays the fallback after all
 *   three (callers apply it).
 */

#ifndef PANTHEON_SERVE_SESSION_KEY_H
#define PANTHEON_SERVE_SESSION_KEY_H

#include "json_reader.h"

/** The body's session key (borrowed from `root`), or NULL if none is set. */
static inline const char *pn_body_session_key(const jr_value *root)
{
    static const char *const names[] = { "session_id", "prompt_cache_key", "promptCacheKey" };

    for (unsigned i = 0; i < sizeof(names) / sizeof(names[0]); i++) {
        const char *key = jr_as_string(jr_object_get(root, names[i]));
        if (key && key[0])
            return key;
    }
    return NULL;
}

#endif
