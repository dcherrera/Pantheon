/* SPDX-License-Identifier: AGPL-3.0-or-later */
/* Copyright (C) 2026 David (Corey) Herrera */
/**
 * @file session_snapshot.h
 * @brief The ONE seam between "a session's state" and the family of snapshot
 *        code that can capture it. Keyed on `pn_arch` and nothing else, so a
 *        caller (the prefix cache, the session facade, the export/restore HTTP
 *        handlers) never names a model family.
 *
 *        Two families exist today and they are not interchangeable:
 *
 *          - QWEN35MOE / BONSAI — a hybrid DUAL state: per-layer SSM recurrent
 *            buffers plus attention KV (fp32 or q4 with fp16 sink/window
 *            shadows). qwen35moe_state_snapshot.c, blob magic "PNS1".
 *          - LLAMA / QWEN2 / QWEN3 / PHI / MISTRAL / MIXTRAL — pure causal
 *            attention, so the whole per-session state is an fp32 KV cache.
 *            llama_state_snapshot.c, blob magic "PNLK".
 *
 *        The blobs carry different magics, so a blob of one family fed to the
 *        other is refused at parse, not misread. `pn_session_snapshot_*` never
 *        guesses: the arch selects the family, and an unsupported arch (RWKVX)
 *        is a refusal.
 *
 * @status Complete. The llama-family arm is what makes session export/restore
 *         reachable on the Tenstorrent backend, which serves exactly that
 *         family; before it, `/admin/session/export` answered 404
 *         `session_not_found` for a live, generating session because the only
 *         predicate in the path asked "is this qwen35moe?".
 * @issues None
 * @todo None
 */

#ifndef PN_SESSION_SNAPSHOT_H
#define PN_SESSION_SNAPSHOT_H

#include <stddef.h>
#include <stdint.h>

#include "engine.h"   /* pn_arch */

/**
 * Can a session on this architecture be captured at all?
 *
 * @param arch  the model's architecture
 * @return      1 if capture/restore/serialize are implemented, 0 otherwise
 *
 * Thread-safe: yes  Reentrant: yes  Signal-safe: yes
 */
int pn_session_snapshot_supported(pn_arch arch);

/**
 * Capture a session's state at `pos` into an opaque, family-tagged snapshot.
 *
 * @param arch      the model's architecture (selects the family)
 * @param kv_cache  pn_state::kv_cache for that model
 * @param pos       committed token count to capture
 * @return          Opaque snapshot (free with pn_session_snapshot_free), or NULL
 *
 * Thread-safe: no — the caller must hold whatever lock stops a turn mutating
 *   the state (gen_lock on the single-user path, the scheduler's on the batched)
 * Reentrant: yes  Signal-safe: no
 */
void *pn_session_snapshot_capture(pn_arch arch, const void *kv_cache, int pos);

/**
 * Write a snapshot back into a live state. Dims are validated against the
 * destination; a mismatch is a refusal, never a partial write.
 *
 * @return 0 on success, -1 if the snapshot does not fit
 *
 * Thread-safe: no (same discipline as capture)  Reentrant: yes  Signal-safe: no
 */
int pn_session_snapshot_restore(pn_arch arch, void *kv_cache, const void *snap);

/** The position a snapshot was taken at, or -1. */
int pn_session_snapshot_pos(pn_arch arch, const void *snap);

/** Free a snapshot of this family. NULL-safe. */
void pn_session_snapshot_free(pn_arch arch, void *snap);

/**
 * Serialize a snapshot into one contiguous little-endian blob.
 *
 * @param weights_version  opaque model-identity tag, echoed in the header and
 *                         validated by the restore handler before any state is
 *                         touched
 * @return 0 on success, -1 on failure (`*bytes_out` untouched on failure)
 *
 * Thread-safe: yes  Reentrant: yes  Signal-safe: no
 */
int pn_session_snapshot_serialize(pn_arch arch, const void *snap,
                                  uint64_t weights_version,
                                  unsigned char **bytes_out, size_t *len_out);

/**
 * Parse a blob back into an opaque snapshot of this family.
 *
 * @return Opaque snapshot, or NULL on any validation failure (including a blob
 *         whose magic belongs to the OTHER family).
 *
 * Thread-safe: yes  Reentrant: yes  Signal-safe: no
 */
void *pn_session_snapshot_deserialize(pn_arch arch, const unsigned char *bytes,
                                      size_t len, uint64_t *weights_version_out);

#endif /* PN_SESSION_SNAPSHOT_H */
