/* SPDX-License-Identifier: AGPL-3.0-or-later */
/* Copyright (C) 2026 David (Corey) Herrera */
/**
 * @file   gguf_fttw_overlay.c
 * @brief  Implementation of the FTTW artifact overlay on an open GGUF.
 *
 * @status None
 * @issues None
 * @todo   None
 */
#include "gguf_fttw_overlay.h"

#include <dirent.h>
#include <fcntl.h>
#include <stddef.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>
#include <stdbool.h>

#include "gguf_tiled.h"
#include "tt/convert/ft_tt_weight_convert.h"

/**
 * Does the GGUF hold the block an artifact entry belongs to? A truncated metadata checkpoint
 * (`ft_tt_gguf_slim --truncate-blocks N`) beside the full artifact: the blocks it dropped are
 * not this model's, so their entries are skipped, not refused.
 */
static bool overlay_block_present(const pn_gguf_file *file, const char *name)
{
    char prefix[32];
    const char *dot;
    size_t len;

    if (strncmp(name, "blk.", 4) != 0 || !(dot = strchr(name + 4, '.')))
        return true;
    len = (size_t)(dot - name) + 1u;
    if (len >= sizeof(prefix))
        return true;
    memcpy(prefix, name, len);
    prefix[len] = '\0';
    for (uint64_t i = 0; i < file->n_tensors; i++)
        if (file->tensors[i].name.len > len && strncmp(file->tensors[i].name.data, prefix, len) == 0)
            return true;
    return false;
}

/**
 * Tiles the entry stores. K (dims[0]) must be whole tiles. N (dims[1]) may end in a partial
 * tile on a plain 2-D weight: the converter zero-pads the last 32-row strip (Qwen3.8's
 * `ssm_alpha`/`ssm_beta` are N = 48, stored as 2 tile columns) and the CPU decodes the whole
 * column and keeps the valid rows (qweight_tiled.c). An expert stack must be whole tiles (the
 * converter refuses others) or the per-expert stride would be wrong. 0 = a shape not read.
 */
static uint64_t overlay_stored_tiles(const ft_tt_artifact_entry *e)
{
    const uint64_t kt = e->dims[0] / PN_TILED_DIM;
    const uint64_t nt = (e->dims[1] + PN_TILED_DIM - 1u) / PN_TILED_DIM;

    if (e->dims[0] == 0u || e->dims[0] % PN_TILED_DIM || e->dims[1] == 0u)
        return 0;
    if (e->dims[1] % PN_TILED_DIM && e->dims[2] * e->dims[3] != 1u)
        return 0;
    return kt * nt * e->dims[2] * e->dims[3];
}

/** The 2-D (or expert-stack) entries that become tensors; 0 = skip, -1 = refuse. */
static int overlay_entry_usable(const pn_gguf_file *file, const ft_tt_artifact_entry *e, pn_gguf_quant *type)
{
    uint64_t tiles;
    size_t tile_bytes;

    if (e->n_dims < 2u || pn_gguf_find_tensor((pn_gguf_file *)file, e->name) ||
        !overlay_block_present(file, e->name))
        return 0;
    *type = pn_tiled_type_for(e->format, e->layout);
    tile_bytes = pn_tiled_tile_bytes(*type);
    tiles = overlay_stored_tiles(e);
    if (!tile_bytes || !tiles || e->byte_count != tiles * tile_bytes) {
        fprintf(stderr, "gguf_fttw: '%.*s' (format %u, layout %u, [%llu, %llu, %llu]) is not whole "
                        "tiles of a format the CPU decodes — refusing the overlay\n",
                FT_TT_ARTIFACT_NAME_BYTES, e->name, e->format, e->layout, (unsigned long long)e->dims[0],
                (unsigned long long)e->dims[1], (unsigned long long)e->dims[2]);
        return -1;
    }
    return 1;
}

/**
 * Map the GGUF and the artifact back to back: reserve one range for both, map the GGUF file
 * again at its start and the artifact right after it, then retire the GGUF's first mapping
 * and move every pointer into it (the data section, raw array values) by the same delta.
 * A hint address is not enough: Linux placed the artifact BELOW the GGUF on the first try.
 */
static unsigned char *overlay_map_after(pn_gguf_file *file, int fd, size_t bytes)
{
    const long page = sysconf(_SC_PAGESIZE) > 0 ? sysconf(_SC_PAGESIZE) : 4096;
    const size_t head = (file->mmap_size + (size_t)page - 1) / (size_t)page * (size_t)page;
    unsigned char *range = mmap(NULL, head + bytes, PROT_NONE, MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);
    ptrdiff_t delta;

    if (range == MAP_FAILED)
        return NULL;
    if (mmap(range, file->mmap_size, PROT_READ, MAP_PRIVATE | MAP_FIXED, file->fd, 0) == MAP_FAILED ||
        mmap(range + head, bytes, PROT_READ, MAP_PRIVATE | MAP_FIXED, fd, 0) == MAP_FAILED) {
        munmap(range, head + bytes);
        return NULL;
    }
    delta = range - (unsigned char *)file->mmap_base;
    for (uint64_t i = 0; i < file->n_kv; i++)
        if (file->kv[i].type == GGUF_TYPE_ARRAY && file->kv[i].value.str.data)
            file->kv[i].value.str.data += delta;
    munmap(file->mmap_base, file->mmap_size);
    file->mmap_base = range;
    file->data = (unsigned char *)file->data + delta;
    return range + head;
}

/** Append the usable entries; returns the count, or -1. */
static int overlay_append(pn_gguf_file *file, const unsigned char *base, const ft_tt_artifact_header *h)
{
    const ft_tt_artifact_entry *dir = (const ft_tt_artifact_entry *)(base + h->directory_offset);
    pn_gguf_tensor_info *grown = realloc(file->tensors, (size_t)(file->n_tensors + h->tensor_count) * sizeof(*grown));
    int added = 0;

    if (!grown)
        return -1;
    file->tensors = grown;
    for (uint32_t i = 0; i < h->tensor_count; i++) {
        pn_gguf_quant type;
        int usable = overlay_entry_usable(file, &dir[i], &type);
        pn_gguf_tensor_info *t = &file->tensors[file->n_tensors];

        if (usable < 0)
            return -1;
        if (usable == 0)
            continue;
        memset(t, 0, sizeof(*t));
        t->name.len = strnlen(dir[i].name, FT_TT_ARTIFACT_NAME_BYTES);
        t->name.data = strndup(dir[i].name, FT_TT_ARTIFACT_NAME_BYTES);
        if (!t->name.data)
            return -1;
        t->n_dims = dir[i].n_dims;
        memcpy(t->dims, dir[i].dims, sizeof(t->dims));
        t->type = type;
        t->offset = (uint64_t)((base + dir[i].data_offset) - (const unsigned char *)file->data);
        file->n_tensors++;
        added++;
    }
    return added;
}

int pn_gguf_attach_fttw(pn_gguf_file *file, const char *fttw_path)
{
    ft_tt_artifact_header header;
    struct stat st;
    unsigned char *base;
    int added;
    int fd;

    if (!file || !fttw_path || file->overlay_base)
        return -1;
    fd = open(fttw_path, O_RDONLY);
    if (fd < 0 || fstat(fd, &st) != 0 || pread(fd, &header, sizeof(header), 0) != (ssize_t)sizeof(header) ||
        header.magic != FT_TT_ARTIFACT_MAGIC || header.version < FT_TT_ARTIFACT_VERSION_MIN ||
        header.version > FT_TT_ARTIFACT_VERSION ||
        header.directory_offset + (uint64_t)header.tensor_count * FT_TT_ARTIFACT_ENTRY_BYTES > (uint64_t)st.st_size) {
        fprintf(stderr, "gguf_fttw: '%s' is not a readable FTTW artifact (v%u..v%u)\n", fttw_path,
                FT_TT_ARTIFACT_VERSION_MIN, FT_TT_ARTIFACT_VERSION);
        if (fd >= 0)
            close(fd);
        return -1;
    }
    base = overlay_map_after(file, fd, (size_t)st.st_size);
    close(fd);
    if (!base) {
        fprintf(stderr, "gguf_fttw: could not map '%s' after the GGUF\n", fttw_path);
        return -1;
    }
    file->overlay_base = base;
    file->overlay_size = (size_t)st.st_size;
    added = overlay_append(file, base, &header);
    if (added < 0)
        return -1;
    file->data_size = (size_t)((base + st.st_size) - (unsigned char *)file->data);
    fprintf(stderr, "gguf_fttw: %d tensors from %s (FTTW v%u) overlaid on the GGUF — the CPU now "
                    "reads the card's weight bytes\n", added, fttw_path, header.version);
    return added;
}

/** `name` is `<stem>.<rung>.fttw` with a non-empty `<rung>` holding no '.'. */
static bool card_artifact_name(const char *name, const char *stem, size_t stem_len)
{
    size_t n = strlen(name);
    size_t rung_len;

    if (n <= stem_len + 1u + 5u || strncmp(name, stem, stem_len) != 0 || name[stem_len] != '.' ||
        strcmp(name + n - 5u, ".fttw") != 0)
        return false;
    rung_len = n - 5u - (stem_len + 1u);
    return rung_len > 0u && memchr(name + stem_len + 1u, '.', rung_len) == NULL;
}

int pn_gguf_attach_card_artifact(pn_gguf_file *file, const char *gguf_path, char *out,
                                 size_t out_bytes)
{
    const char *slash = gguf_path ? strrchr(gguf_path, '/') : NULL;
    const char *base = slash ? slash + 1 : gguf_path;
    char dir[1024], first[256] = "", second[256] = "";
    size_t stem_len;
    struct dirent *ent;
    struct stat st;
    int found = 0;
    DIR *d;

    if (!file || !gguf_path || !out || out_bytes == 0 ||
        pn_gguf_find_tensor(file, "token_embd.weight"))
        return 0;
    stem_len = strlen(base);
    if (stem_len <= 5u || strcmp(base + stem_len - 5u, ".gguf") != 0)
        return 0;
    stem_len -= 5u;
    if (!slash)
        snprintf(dir, sizeof(dir), ".");
    else if (slash == gguf_path)
        snprintf(dir, sizeof(dir), "/");
    else if ((size_t)(slash - gguf_path) < sizeof(dir))
        snprintf(dir, sizeof(dir), "%.*s", (int)(slash - gguf_path), gguf_path);
    else
        return 0;
    if (!(d = opendir(dir)))
        return 0;
    while ((ent = readdir(d)) != NULL) {
        char path[1300];
        if (!card_artifact_name(ent->d_name, base, stem_len))
            continue;
        snprintf(path, sizeof(path), "%s/%s", dir, ent->d_name);
        if (stat(path, &st) != 0 || !S_ISREG(st.st_mode))
            continue;
        snprintf(found++ == 0 ? first : second, sizeof(first), "%s", ent->d_name);
    }
    closedir(d);
    if (found == 0)
        return 0;
    if (found > 1) {
        fprintf(stderr, "gguf_fttw: %s is a card metadata GGUF with %d artifacts beside it "
                        "(%s, %s, ...); refusing to guess\n", gguf_path, found, first, second);
        return -1;
    }
    if (snprintf(out, out_bytes, "%s/%s", dir, first) >= (int)out_bytes) {
        fprintf(stderr, "gguf_fttw: the card artifact path beside %s is too long\n", gguf_path);
        return -1;
    }
    fprintf(stderr, "gguf_fttw: %s is a card metadata GGUF; attaching its artifact %s\n",
            gguf_path, out);
    return pn_gguf_attach_fttw(file, out) < 0 ? -1 : 1;
}
