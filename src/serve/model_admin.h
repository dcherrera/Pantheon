/* SPDX-License-Identifier: AGPL-3.0-or-later */
/* Copyright (C) 2026 David (Corey) Herrera */
/**
 * @file model_admin.h
 * @brief Runtime load/unload admin endpoints for multi-model serving:
 *        models add/drop without restarting serve, so
 *        GET /v1/models is a live list.
 *
 *        Routes (wired in serve_command.c):
 *          POST /admin/models/load    {"path":"...", "name":"...",
 *                                      "dies":[0,1], "quantize":false}
 *          POST /admin/models/unload  {"name":"..."}
 *          GET  /admin/dies           die-manager occupancy snapshot
 *
 *        These share the serve port, so keep that port private (localhost or
 *        a private network) and put a gateway in front that forwards only
 *        /v1. Loads run on the caller's connection thread — a big model load
 *        takes a while and the client waits; serving of resident models
 *        continues meanwhile.
 *
 * @status Live on the CPU host; real die placement numbers are gated on GPU hardware.
 * @issues None
 * @todo None
 */

#ifndef PN_SERVE_MODEL_ADMIN_H
#define PN_SERVE_MODEL_ADMIN_H

#include "json_reader.h"
#include "model_registry.h"

/* One model load: what `--model ...` flags or a POST /admin/models/load body
 * asked for. Strings are borrowed. */
typedef struct {
    const char          *path;
    const char          *name;              /* NULL = file basename */
    int                  dies[PN_MODEL_MAX_DIES];
    int                  n_dies;            /* 0 = auto-place / CPU */
    pn_load_opts         opts;              /* quantize, ctx_cap, vocab, config */
    int                  reasoning_budget;  /* 0 = disabled */
    int                  reasoning_reserve; /* answer reserve; 0 = PN_REASONING_RESERVE_DEFAULT */
    int                  max_sessions;      /* 0 = default */
    int                  max_concurrent;    /* 0 = max_sessions */
    bool                 mtp;               /* load the MTP head (CUDA qwen35moe) */
    const char          *tt_kv_type;        /* PN_TT_KV_TYPE for this load (Tenstorrent KV
                                             * cache type); NULL = the process's */
    const char          *tt_kv_long;        /* PN_TT_KV_LONG for this load (the KV type
                                             * the TT fit may switch to at long ctx) */
    const char          *tt_spec;           /* PN_TT_SPEC for this load (MTP drafts a
                                             * step, "0" = no MTP head); NULL = the process's */
    pn_model_sampling sampling;          /* per-model sampling defaults (A4) */
    int                  replica;           /* copy number (mr_load_replica); 0 =
                                             * the lowest the name does not use */
} ma_load_request;

/**
 * Load + register one model and apply its sampling defaults. `mtp` sets
 * QWEN35_MTP for this load only and refuses a model without a head; `tt_kv_type` /
 * `tt_spec` set PN_TT_KV_TYPE / PN_TT_SPEC for this load only (the TT load fixes the
 * KV type, the MTP arm and the verify rows it reserves from them; a later decode
 * drafts no more than those rows).
 *
 * @param reg     model registry
 * @param lr      the load
 * @param err     short reason on failure
 * @param errlen  size of err
 * @return 0 on success, -1 with a reason
 *
 * Thread-safe: yes (loads serialize on an internal lock for the load env)
 * Reentrant: no
 * Signal-safe: no
 */
int ma_load_model(mr_registry *reg, const ma_load_request *lr, char *err, size_t errlen);

/**
 * Parse a POST /admin/models/load body into *out (strings borrow from root).
 * @return 0, or -1 with a reason in err
 *
 * Thread-safe: yes
 * Reentrant: yes
 * Signal-safe: no
 */
int ma_parse_load_request(const jr_value *root, ma_load_request *out, char *err, size_t errlen);

/** POST /admin/models/load — load + register a model at runtime. Accepts the
 *  sampling-default fields (temperature, top_p, top_k, min_p,
 *  presence_penalty) and "mtp" next to the existing ones. */
void ma_handle_load(int client_fd, mr_registry *reg, const char *body);

/** POST /admin/models/unload — drain and unload a resident model. */
void ma_handle_unload(int client_fd, mr_registry *reg, const char *body);

/** GET /admin/dies — die-manager occupancy snapshot (empty list on CPU). */
/**
 * GET /admin/models — every LIVE instance, one row each (NOT deduped by name,
 * unlike /v1/models: two instances of the same model on different dies are two
 * rows, because an operator's question is per-instance).
 *
 * The row is the same one /admin/models/load returns, so until now its
 * operator-only fields — context_window_configured, and the CUDA-graph
 * residency `graphs` — could be read only as a side effect of LOADING a model.
 * That is why task #20 had to infer the memory ladder's runtime clamp instead
 * of reading it. This makes the row readable at any time.
 *
 * Thread-safe: yes (mr_snapshot takes the registry lock)
 * Reentrant: no (writes to a client fd)
 * Signal-safe: no
 */
void ma_handle_models(int client_fd, mr_registry *reg);

void ma_handle_dies(int client_fd);

/**
 * POST /admin/session/export {"name":"...","session_id":"...","release":false}
 * — session warm-tier export (K80 serving-ops Phase 7). Snapshot + serialize the
 * named model's live session state for `session_id` into a portable blob and
 * return it base64 in JSON alongside its weights_version tag and the committed
 * token ids the blob covers, so the CLIENT can store it and later POST
 * /session/restore to warm-restart the session. Read-only w.r.t. the session,
 * unless `release` is true: then the session's slot is freed (as POST
 * /session/release does) right after the snapshot is taken, before the blob is
 * serialized and sent. A client closing a session sets it, so the slot comes back
 * in about a second instead of after the whole transfer (a 30k-token session's
 * blob is ~250 MB; the transfer took ~67 s on the K80 test machine and new
 * sessions got 429 the whole time). An export refused before the snapshot (no
 * session, busy, not yours) releases nothing; a serialize failure after it still
 * releases, which is what the closing client does next anyway.
 *
 * Covers BOTH session models via the model_session.h facade — the batched
 * scheduler's warm decode slot as well as the single-user prefix cache. Since
 * the host KV mirror was removed (Phase 10b / INV3) this is the only path by
 * which a session's state can leave the GPU, so it materializes the KV rows
 * from the decode-slot ring on demand rather than reading any mirror.
 *
 * 404 if the model or session is unknown (including "the session's rows are no
 * longer in any GPU ring" — the client falls back to message replay); 429 if a
 * turn is in flight on that session; 400 if the model has no session store.
 */
void ma_handle_session_export(int client_fd, mr_registry *reg, const char *body);

#endif /* PN_SERVE_MODEL_ADMIN_H */
