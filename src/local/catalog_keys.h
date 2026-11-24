/* SPDX-License-Identifier: AGPL-3.0-or-later */
/* Copyright (C) 2026 David (Corey) Herrera */
/**
 * @file   catalog_keys.h
 * @brief  The catalog trust anchors compiled into the binary (spec §11.2): the
 *         minisign public keys allowed to sign the catalog (current + backup),
 *         and the Hugging Face orgs a catalog file may come from (compared case-insensitively).
 *
 * @status David's current and backup keys are set (generated 2026-10-27; secret
 *         keys in ~/.minisign/ on David's Mac, never in a repo).
 * @issues None
 * @todo None
 */

#ifndef PN_LOCAL_CATALOG_KEYS_H
#define PN_LOCAL_CATALOG_KEYS_H

/* Base64 line of each minisign.pub ("RW..."). Empty = not set. */
#define PN_CATALOG_KEY_CURRENT "RWQtV5kfB8nhIxGvGwhamn2/dZkW8o6+0dP9ljQzGhkKZ/gzXFpJzx8g"
#define PN_CATALOG_KEY_BACKUP  "RWTG4oFKUZEoVc4jwNcY04I6oTJw/lTvvPwdwd+FYS6AaVuSbUDFNsaO"

/* TEST ONLY (tests/local/fixtures/test_catalog.pub): compiled in only with
 * `make LOCAL_TEST_CATALOG_KEY=1`, for the on-hardware e2e scripts. */
#ifdef PN_LOCAL_TEST_CATALOG_KEY
#define PN_CATALOG_KEY_TEST "RWTiSyg6tQZKheDELlvn4wrg39qNe6GPJK/K+d5oohCc50r1/53zB+VI"
#else
#define PN_CATALOG_KEY_TEST ""
#endif

/* Catalog files may only come from these Hugging Face orgs (spec §1: "from
 * our Hugging Face org only"). */
#define PN_CATALOG_HF_ORGS { "TeamIDE" }

#endif /* PN_LOCAL_CATALOG_KEYS_H */
