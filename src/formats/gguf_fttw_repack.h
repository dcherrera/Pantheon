/* SPDX-License-Identifier: AGPL-3.0-or-later */
/* Copyright (C) 2026 David (Corey) Herrera */
/**
 * @file   gguf_fttw_repack.h
 * @brief  Per-tensor format map + in-memory Bfp8 -> Bfp4 repack of an attached FTTW artifact:
 *         the mixed Bfp8/Bfp4 tier's sensitivity arms without a new artifact on disk.
 *
 * MAP FILE (shared with the converter's `--format-map`): one entry per line,
 * `<gguf tensor name> <fttw format name>`; `#` starts a comment (whole line or trailing);
 * blank lines are ignored; anything else is a parse error naming the line. A name listed twice
 * is an error.
 *
 * REPACK: every map entry must name an artifact tensor stored N-major `bfp8_b`
 * (PN_QUANT_TILED_BFP8) and ask for `bfp4_b`; anything else refuses the whole map before a byte
 * is touched. A repacked tensor's tiles are decoded with foundry `ft_bfp_unpack_tiles` and
 * re-encoded with `ft_bfp_pack_tiles` in place, at the head of its own span (576 < 1088 B a
 * tile), and its type becomes PN_QUANT_TILED_BFP4. The artifact is mapped MAP_PRIVATE, so the
 * write is copy-on-write: RAM only, the file is never changed. The bytes past the new end of
 * the span keep stale Bfp8 tiles that nothing reads.
 *
 * `PN_MIXW_BFP4=<map file>` (default off) applies a map at load, with `PN_FTTW_ORACLE`.
 *
 * @status None
 * @issues None
 * @todo   None
 */
#ifndef PN_GGUF_FTTW_REPACK_H
#define PN_GGUF_FTTW_REPACK_H

#include <stddef.h>
#include <stdint.h>

#include "gguf.h"

#ifdef __cplusplus
extern "C" {
#endif

/** Longest tensor name a map entry holds (FT_TT_ARTIFACT_NAME_BYTES, NUL included). */
#define PN_FTTW_FORMAT_MAP_NAME_BYTES 128
/** Longest format name a map entry holds, NUL included. */
#define PN_FTTW_FORMAT_MAP_FORMAT_BYTES 16

/** One map line: a tensor and the FTTW format it should be stored in. */
typedef struct {
    char tensor_name[PN_FTTW_FORMAT_MAP_NAME_BYTES];
    char format_name[PN_FTTW_FORMAT_MAP_FORMAT_BYTES];
} pn_fttw_format_map_entry;

/** A parsed map; `count` may be 0 (an empty map changes nothing). */
typedef struct {
    pn_fttw_format_map_entry *entries;
    size_t count;
} pn_fttw_format_map;

/**
 * Parse a map file into `out` (zeroed first).
 *
 * @return 0, or -1 (stderr names the file and line: unreadable file, a line not of two
 *         fields, a name too long, a name listed twice, out of memory)
 *
 * Thread-safe: yes (own buffers)    Reentrant: yes    Signal-safe: no
 */
int pn_fttw_format_map_load(const char *path, pn_fttw_format_map *out);

/** Release a map from pn_fttw_format_map_load; NULL-safe, leaves it empty. */
void pn_fttw_format_map_free(pn_fttw_format_map *map);

/**
 * Repack every map entry's tensor Bfp8 -> Bfp4 in place (see the file comment). Every entry is
 * checked first: format `bfp4_b`, the tensor present, PN_QUANT_TILED_BFP8, whole tiles, and
 * its span inside the attached artifact mapping; any failure refuses the map with nothing
 * changed.
 *
 * @return tensors repacked (>= 0), or -1 (stderr says which entry and why)
 *
 * Thread-safe: no    Reentrant: no    Signal-safe: no
 */
int pn_gguf_repack_bfp4(pn_gguf_file *file, const pn_fttw_format_map *map);

/**
 * pn_fttw_format_map_load + pn_gguf_repack_bfp4 + free: what `PN_MIXW_BFP4=<path>` runs.
 *
 * @return tensors repacked (>= 0), or -1
 *
 * Thread-safe: no    Reentrant: no    Signal-safe: no
 */
int pn_gguf_repack_bfp4_from_map_file(pn_gguf_file *file, const char *map_path);

#ifdef __cplusplus
}
#endif

#endif /* PN_GGUF_FTTW_REPACK_H */
