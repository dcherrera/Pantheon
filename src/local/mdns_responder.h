/* SPDX-License-Identifier: AGPL-3.0-or-later */
/* Copyright (C) 2026 David (Corey) Herrera */
/**
 * @file   mdns_responder.h
 * @brief  Minimal mDNS responder (spec §3, §11.6): one thread on UDP 5353 that
 *         probes for `pantheon-<hostname>` (renaming `-2`, `-3` on a conflict),
 *         announces, answers PTR/SRV/TXT/A for `_pantheon._tcp` and says goodbye
 *         at shutdown. Coexists with avahi / mDNSResponder (SO_REUSEPORT).
 *
 * @status None
 * @issues IPv4 only. The A record is the address of the interface whose subnet
 *         holds the querier (point-to-point links such as wg0 are skipped).
 *         Interfaces are re-read every 30 s; an address change is not
 *         re-announced until a query asks.
 * @todo   None
 */

#ifndef PN_LOCAL_MDNS_RESPONDER_H
#define PN_LOCAL_MDNS_RESPONDER_H

#include <stdint.h>

#include "mdns_packet.h"

#ifdef __cplusplus
extern "C" {
#endif

#define PN_MDNS_TXT_ITEM 96

/* Fills up to `max` "key=value" TXT strings; returns how many. Called from the
 * responder thread for every response, so the TXT is always current. */
typedef int (*pn_mdns_txt_fn)(void *user, char txt[][PN_MDNS_TXT_ITEM], int max);

typedef struct pn_mdns_responder pn_mdns_responder;

/**
 * Start the responder thread.
 * @param hostname  the machine's hostname (label = "pantheon-<first label>")
 * @param port      the HTTP port announced in SRV
 * @return the responder, or NULL (socket/thread failure, logged)
 *
 * Thread-safe: yes. Reentrant: yes. Signal-safe: no
 */
pn_mdns_responder *pn_mdns_start(const char *hostname, uint16_t port, pn_mdns_txt_fn txt,
                                 void *user);

/**
 * Send the goodbye, stop and join the thread, free. NULL-safe.
 *
 * Thread-safe: no. Reentrant: no. Signal-safe: no
 */
void pn_mdns_stop(pn_mdns_responder *r);

/**
 * The label in use after probing (e.g. "pantheon-lab-node-2-2" after a rename).
 *
 * Thread-safe: yes (read after start). Reentrant: yes. Signal-safe: no
 */
void pn_mdns_current_label(pn_mdns_responder *r, char *out, size_t cap);

#ifdef __cplusplus
}
#endif

#endif /* PN_LOCAL_MDNS_RESPONDER_H */
