/* SPDX-License-Identifier: AGPL-3.0-or-later */
/* Copyright (C) 2026 David (Corey) Herrera */
/**
 * @file serve_gateway_setup.h
 * @brief Start-up of `pantheon serve --gateway` (gateway_mode.h): load the
 *        gateway key and switch the serve's session policy to gateway mode.
 *
 * @status Complete
 * @issues None
 * @todo None
 */
#ifndef PANTHEON_SERVE_GATEWAY_SETUP_H
#define PANTHEON_SERVE_GATEWAY_SETUP_H

#include <stddef.h>

/**
 * Turn gateway mode on: read the key file, enable gateway_mode, make session
 * seats evictable for every model loaded from now on, shorten the idle seat
 * TTL to PN_GATEWAY_SESSION_IDLE_TTL_MS (PN_SESSION_IDLE_TTL_MS still wins),
 * and seed the process RNG from OS entropy so unseeded sampling does not
 * replay after a restart. Call once, before any model loads.
 * @return 0, or -1 with a reason in err (the key is never in it)
 *
 * Thread-safe: no. Reentrant: no. Signal-safe: no
 */
int pn_serve_gateway_setup(const char *key_file, char *err, size_t errlen);

#endif /* PANTHEON_SERVE_GATEWAY_SETUP_H */
