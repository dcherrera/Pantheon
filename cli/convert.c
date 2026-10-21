/* SPDX-License-Identifier: AGPL-3.0-or-later */
/* Copyright (C) 2026 David (Corey) Herrera */
/*
 * pantheon-convert: Convert PyTorch .pth files to safetensors format
 *
 * Usage: pantheon-convert <input.pth> -o <output.safetensors> [--inspect]
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "pth/pth.h"
#include "safetensors/safetensors.h"

static void print_usage(const char *prog) {
    fprintf(stderr, "Usage: %s <input.pth> -o <output.safetensors> [--inspect]\n", prog);
    fprintf(stderr, "\nOptions:\n");
    fprintf(stderr, "  --inspect   Print tensor table and exit (no conversion)\n");
}

int main(int argc, char **argv) {
    const char *input = NULL;
    const char *output = NULL;
    int inspect = 0;

    for (int i = 1; i < argc; i++) {
        if (strcmp(argv[i], "-o") == 0 && i + 1 < argc) {
            output = argv[++i];
        } else if (strcmp(argv[i], "--inspect") == 0) {
            inspect = 1;
        } else if (argv[i][0] != '-') {
            input = argv[i];
        } else {
            fprintf(stderr, "Unknown option: %s\n", argv[i]);
            print_usage(argv[0]);
            return 1;
        }
    }

    if (!input) {
        fprintf(stderr, "Error: no input file specified\n");
        print_usage(argv[0]);
        return 1;
    }

    if (!inspect && !output) {
        fprintf(stderr, "Error: no output file specified (use -o or --inspect)\n");
        print_usage(argv[0]);
        return 1;
    }

    /* Open .pth file */
    printf("Opening %s...\n", input);
    pt_file *pth = pt_open(input);
    if (!pth) {
        fprintf(stderr, "Error: failed to open %s\n", input);
        return 1;
    }

    int n = pt_tensor_count(pth);
    printf("Found %d tensors\n\n", n);

    /* Print tensor table */
    if (inspect || !output) {
        printf("%-50s %-8s %-20s %12s\n", "Name", "Dtype", "Shape", "Size");
        printf("%-50s %-8s %-20s %12s\n",
               "--------------------------------------------------",
               "--------", "--------------------", "------------");

        size_t total_bytes = 0;
        for (int i = 0; i < n; i++) {
            const pt_tensor *t = pt_tensor_at(pth, i);
            if (!t) continue;

            char shape_str[128] = {0};
            int off = 0;
            off += snprintf(shape_str + off, sizeof(shape_str) - off, "[");
            for (int d = 0; d < t->ndim; d++) {
                if (d > 0) off += snprintf(shape_str + off, sizeof(shape_str) - off, ", ");
                off += snprintf(shape_str + off, sizeof(shape_str) - off, "%lld",
                                (long long)t->shape[d]);
            }
            snprintf(shape_str + off, sizeof(shape_str) - off, "]");

            char size_str[32];
            if (t->data_size >= 1024 * 1024)
                snprintf(size_str, sizeof(size_str), "%.1f MB", (double)t->data_size / (1024 * 1024));
            else if (t->data_size >= 1024)
                snprintf(size_str, sizeof(size_str), "%.1f KB", (double)t->data_size / 1024);
            else
                snprintf(size_str, sizeof(size_str), "%zu B", t->data_size);

            printf("%-50s %-8s %-20s %12s\n",
                   t->name, fs_dtype_name(t->dtype), shape_str, size_str);

            total_bytes += t->data_size;
        }

        printf("\nTotal: %d tensors, %.1f MB\n", n, (double)total_bytes / (1024 * 1024));

        if (inspect) {
            pt_close(pth);
            return 0;
        }
    }

    /* Convert to safetensors */
    printf("Converting to %s...\n", output);

    fs_writer *w = fs_writer_new();
    if (!w) {
        fprintf(stderr, "Error: failed to create safetensors writer\n");
        pt_close(pth);
        return 1;
    }

    int skipped = 0;
    for (int i = 0; i < n; i++) {
        const pt_tensor *t = pt_tensor_at(pth, i);
        if (!t || !t->data || t->data_size == 0) {
            fprintf(stderr, "  Warning: skipping tensor '%s' (no data)\n", t ? t->name : "?");
            skipped++;
            continue;
        }
        fs_writer_add_tensor(w, t->name, t->dtype, t->shape, t->ndim, t->data);
    }

    if (!fs_writer_save(w, output)) {
        fprintf(stderr, "Error: failed to write %s\n", output);
        fs_writer_free(w);
        pt_close(pth);
        return 1;
    }

    fs_writer_free(w);
    pt_close(pth);

    printf("Done! Converted %d tensors (%d skipped)\n", n - skipped, skipped);
    return 0;
}
