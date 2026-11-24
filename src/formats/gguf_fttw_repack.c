/* SPDX-License-Identifier: AGPL-3.0-or-later */
/* Copyright (C) 2026 David (Corey) Herrera */
/**
 * @file   gguf_fttw_repack.c
 * @brief  Implementation of the format map parser and the in-memory Bfp8 -> Bfp4 repack.
 *
 * @status None
 * @issues None
 * @todo   None
 */
#include "gguf_fttw_repack.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/mman.h>
#include <unistd.h>

#include "gguf_dequant.h"
#include "gguf_tiled.h"
#include "tt/convert/ft_bfp.h"

#define MAP_LINE_BYTES 1024
#define REPACK_FORMAT_NAME "bfp4_b"
/* Tiles decoded per ft_bfp call: 256 x 4 KiB of fp32 scratch. */
#define REPACK_CHUNK_TILES 256u

/** Copy the next whitespace-delimited field at `*cursor` into `out`; 0, or -1 if too long / none. */
static int map_take_field(char **cursor, char *out, size_t cap)
{
    char *start = *cursor + strspn(*cursor, " \t");
    size_t len = strcspn(start, " \t");

    if (len == 0 || len >= cap)
        return -1;
    memcpy(out, start, len);
    out[len] = '\0';
    *cursor = start + len;
    return 0;
}

/** Parse one line into `entry`: 1 an entry, 0 blank/comment, -1 malformed. */
static int map_parse_line(char *line, pn_fttw_format_map_entry *entry)
{
    char *cursor = line;

    line[strcspn(line, "#\r\n")] = '\0';
    if (line[strspn(line, " \t")] == '\0')
        return 0;
    if (map_take_field(&cursor, entry->tensor_name, sizeof(entry->tensor_name)) != 0 ||
        map_take_field(&cursor, entry->format_name, sizeof(entry->format_name)) != 0)
        return -1;
    return cursor[strspn(cursor, " \t")] == '\0' ? 1 : -1;
}

/** Append `entry` unless its tensor is already listed; 0, or -1 (stderr says why). */
static int map_append(pn_fttw_format_map *map, const pn_fttw_format_map_entry *entry, const char *path,
                      int line_number)
{
    pn_fttw_format_map_entry *grown;

    for (size_t i = 0; i < map->count; i++)
        if (strcmp(map->entries[i].tensor_name, entry->tensor_name) == 0) {
            fprintf(stderr, "format map %s:%d: '%s' is listed twice\n", path, line_number, entry->tensor_name);
            return -1;
        }
    grown = realloc(map->entries, (map->count + 1u) * sizeof(*grown));
    if (!grown) {
        fprintf(stderr, "format map %s: out of memory\n", path);
        return -1;
    }
    map->entries = grown;
    map->entries[map->count++] = *entry;
    return 0;
}

int pn_fttw_format_map_load(const char *path, pn_fttw_format_map *out)
{
    char line[MAP_LINE_BYTES];
    int line_number = 0, rc = 0;
    FILE *f;

    if (!out)
        return -1;
    memset(out, 0, sizeof(*out));
    if (!path || !(f = fopen(path, "r"))) {
        fprintf(stderr, "format map %s: cannot be read\n", path ? path : "(null)");
        return -1;
    }
    while (rc == 0 && fgets(line, sizeof(line), f)) {
        pn_fttw_format_map_entry entry;
        int parsed;

        line_number++;
        if (!strchr(line, '\n') && !feof(f)) {
            fprintf(stderr, "format map %s:%d: line longer than %d bytes\n", path, line_number, MAP_LINE_BYTES - 1);
            rc = -1;
            break;
        }
        parsed = map_parse_line(line, &entry);
        if (parsed < 0) {
            fprintf(stderr, "format map %s:%d: expected '<gguf tensor name> <fttw format name>' "
                            "(names under %d bytes, '#' comments)\n",
                    path, line_number, PN_FTTW_FORMAT_MAP_NAME_BYTES);
            rc = -1;
        } else if (parsed > 0) {
            rc = map_append(out, &entry, path, line_number);
        }
    }
    if (rc == 0 && ferror(f)) {
        fprintf(stderr, "format map %s: read error\n", path);
        rc = -1;
    }
    fclose(f);
    if (rc != 0)
        pn_fttw_format_map_free(out);
    return rc;
}

void pn_fttw_format_map_free(pn_fttw_format_map *map)
{
    if (!map)
        return;
    free(map->entries);
    map->entries = NULL;
    map->count = 0;
}

/** Stored tiles of an N-major tensor: K whole tiles, N rounded up (padded strip), times experts. */
static uint64_t repack_stored_tiles(const pn_gguf_tensor_info *t)
{
    uint64_t tiles;

    if (t->n_dims < 2u || t->dims[0] == 0u || t->dims[0] % PN_TILED_DIM || t->dims[1] == 0u)
        return 0;
    tiles = (t->dims[0] / PN_TILED_DIM) * ((t->dims[1] + PN_TILED_DIM - 1u) / PN_TILED_DIM);
    for (uint32_t d = 2; d < t->n_dims && d < 4u; d++)
        tiles *= t->dims[d];
    return tiles;
}

/** The tensor a map entry repacks, checked; NULL with stderr saying why. */
static pn_gguf_tensor_info *repack_checked_tensor(pn_gguf_file *file, const pn_fttw_format_map_entry *entry)
{
    pn_gguf_tensor_info *t = pn_gguf_find_tensor(file, entry->tensor_name);
    const unsigned char *lo = file->overlay_base, *hi = lo + file->overlay_size, *span;
    char type_name[PN_GGUF_QUANT_NAME_MAX];
    uint64_t tiles;

    if (strcmp(entry->format_name, REPACK_FORMAT_NAME) != 0) {
        fprintf(stderr, "PN_MIXW_BFP4: '%s' asks for format '%s'; this knob repacks to %s only\n",
                entry->tensor_name, entry->format_name, REPACK_FORMAT_NAME);
        return NULL;
    }
    if (!t) {
        fprintf(stderr, "PN_MIXW_BFP4: '%s' is not a tensor of this model (unknown name, or a "
                        "layer it does not have)\n", entry->tensor_name);
        return NULL;
    }
    if (t->type != PN_QUANT_TILED_BFP8) {
        fprintf(stderr, "PN_MIXW_BFP4: '%s' is %s, not an N-major bfp8_b artifact tensor\n", entry->tensor_name,
                pn_gguf_quant_name(t->type, type_name, sizeof(type_name)));
        return NULL;
    }
    tiles = repack_stored_tiles(t);
    span = (const unsigned char *)file->data + t->offset;
    if (!tiles || !lo || span < lo || tiles > (uint64_t)(hi - span) / ft_bfp_tile_bytes(FT_BFP_BFP8_B)) {
        fprintf(stderr, "PN_MIXW_BFP4: '%s' is not whole tiles inside the attached artifact\n", entry->tensor_name);
        return NULL;
    }
    return t;
}

/** Make [span, span + bytes) writable (`writable`) or read-only again, whole pages. */
static int repack_protect(unsigned char *span, size_t bytes, int writable)
{
    const size_t page = sysconf(_SC_PAGESIZE) > 0 ? (size_t)sysconf(_SC_PAGESIZE) : 4096u;
    const uintptr_t start = (uintptr_t)span / page * page;
    const uintptr_t end = ((uintptr_t)span + bytes + page - 1u) / page * page;

    return mprotect((void *)start, end - start, writable ? PROT_READ | PROT_WRITE : PROT_READ);
}

/**
 * Bfp8 tiles -> Bfp4 tiles at the head of the same span. A chunk's Bfp4 bytes end at
 * (done + n) * 576, before the first unread Bfp8 byte at (done + n) * 1088, and its own Bfp8
 * bytes are already decoded into `scratch`, so writing over them is safe.
 */
static int repack_tensor_tiles(unsigned char *span, uint64_t tiles, float *scratch)
{
    const size_t from = ft_bfp_tile_bytes(FT_BFP_BFP8_B), to = ft_bfp_tile_bytes(FT_BFP_BFP4_B);

    for (uint64_t done = 0; done < tiles;) {
        size_t n = tiles - done < REPACK_CHUNK_TILES ? (size_t)(tiles - done) : REPACK_CHUNK_TILES;

        if (ft_bfp_unpack_tiles(FT_BFP_BFP8_B, span + done * from, n, scratch) != FT_BFP_OK ||
            ft_bfp_pack_tiles(FT_BFP_BFP4_B, scratch, n, span + done * to) != FT_BFP_OK)
            return -1;
        done += n;
    }
    return 0;
}

/** Repack one checked tensor and retype it; 0, or -1. */
static int repack_one_tensor(pn_gguf_file *file, pn_gguf_tensor_info *t, float *scratch)
{
    unsigned char *span = (unsigned char *)file->data + t->offset;
    const uint64_t tiles = repack_stored_tiles(t);
    const size_t bytes = (size_t)tiles * ft_bfp_tile_bytes(FT_BFP_BFP8_B);

    if (repack_protect(span, bytes, 1) != 0) {
        perror("PN_MIXW_BFP4: mprotect");
        return -1;
    }
    if (repack_tensor_tiles(span, tiles, scratch) != 0) {
        fprintf(stderr, "PN_MIXW_BFP4: '%.*s' could not be re-encoded\n", (int)t->name.len, t->name.data);
        return -1;
    }
    t->type = PN_QUANT_TILED_BFP4;
    if (repack_protect(span, bytes, 0) != 0) {
        perror("PN_MIXW_BFP4: mprotect");
        return -1;
    }
    return 0;
}

int pn_gguf_repack_bfp4(pn_gguf_file *file, const pn_fttw_format_map *map)
{
    pn_gguf_tensor_info **targets;
    float *scratch;
    int repacked = 0;

    if (!file || !map)
        return -1;
    if (map->count == 0)
        return 0;
    targets = calloc(map->count, sizeof(*targets));
    scratch = malloc((size_t)REPACK_CHUNK_TILES * PN_TILED_DIM * PN_TILED_DIM * sizeof(float));
    if (!targets || !scratch) {
        fprintf(stderr, "PN_MIXW_BFP4: out of memory\n");
        repacked = -1;
    }
    for (size_t i = 0; repacked == 0 && i < map->count; i++)
        if (!(targets[i] = repack_checked_tensor(file, &map->entries[i])))
            repacked = -1;
    for (size_t i = 0; repacked >= 0 && i < map->count; i++)
        repacked = repack_one_tensor(file, targets[i], scratch) == 0 ? repacked + 1 : -1;
    free(targets);
    free(scratch);
    return repacked;
}

int pn_gguf_repack_bfp4_from_map_file(pn_gguf_file *file, const char *map_path)
{
    pn_fttw_format_map map;
    int repacked;

    if (pn_fttw_format_map_load(map_path, &map) != 0)
        return -1;
    repacked = pn_gguf_repack_bfp4(file, &map);
    pn_fttw_format_map_free(&map);
    if (repacked >= 0)
        fprintf(stderr, "PN_MIXW_BFP4: %d tensors repacked Bfp8 -> Bfp4 in memory from %s (the artifact "
                        "file is untouched)\n", repacked, map_path);
    return repacked;
}
