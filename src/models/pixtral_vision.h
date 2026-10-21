/* SPDX-License-Identifier: AGPL-3.0-or-later */
/* Copyright (C) 2026 David (Corey) Herrera */
/**
 * @file   pixtral_vision.h
 * @brief  Pixtral-style vision encoder with a LightOnOCR-style projector (clip mmproj GGUF).
 *
 * Written from a behaviour specification (plans/cleanroom_r2_pixtral_spec.md); no ggml or
 * llama.cpp source was consulted.
 *
 * @status None
 * @issues None
 * @todo   None
 */
#ifndef PN_PIXTRAL_VISION_H
#define PN_PIXTRAL_VISION_H

#include <stdint.h>
#include "../formats/gguf.h"
#include "../formats/gguf_dequant.h"

#ifdef __cplusplus
extern "C" {
#endif

/**
 * Vision tower + projector hyperparameters, read from `clip.*` GGUF metadata.
 * Layout is ABI (callers and tests rely on sizeof/offsetof).
 */
typedef struct {
    int image_size;      /* advertised image size; stored, not used by encode */
    int patch_size;      /* square patch edge in pixels */
    int hidden;          /* tower width D */
    int n_layer;         /* transformer blocks */
    int n_head;          /* attention heads */
    int head_dim;        /* hidden / n_head (integer division) */
    int n_ff;            /* MLP inner width */
    int merge;           /* projector spatial merge factor */
    int proj_dim;        /* output embedding width (decoder n_embd) */
    float eps;           /* RMSNorm epsilon, used by every norm */
    float rope_theta;    /* 2D RoPE base */
    int use_silu;        /* stored, not used: the MLP is always SiLU-gated */
    float image_mean[3]; /* per-channel pixel normalisation mean */
    float image_std[3];  /* per-channel pixel normalisation std */
} pn_vision_config;

/**
 * One transformer block's F32 weights, row-major [out, in]:
 * norms [hidden]; wq/wk/wv/wo [hidden, hidden]; w_gate/w_up [n_ff, hidden];
 * w_down [hidden, n_ff]. No biases.
 */
typedef struct {
    float *ln1;
    float *wq, *wk, *wv, *wo;
    float *ln2;
    float *w_gate, *w_up;
    float *w_down;
} pn_vision_layer;

/**
 * A loaded vision encoder. Owns F32 copies of every weight:
 * patch_embd [hidden, 3*ps*ps] (input ordered channel, ky, kx); pre_ln and
 * mm_input_norm [hidden]; mm_merger [proj_dim, hidden*merge*merge];
 * mm1/mm2 [proj_dim, proj_dim]; layers[cfg.n_layer].
 */
typedef struct {
    pn_vision_config cfg;
    float *patch_embd;
    float *pre_ln;
    pn_vision_layer *layers;
    float *mm_input_norm;
    float *mm_merger;
    float *mm1;
    float *mm2;
} pn_vision_model;

/**
 * Load a vision encoder from a GGUF whose `general.architecture` is "clip" (a vision
 * mmproj file). Weights are dequantized to owned F32 copies, so the caller may close
 * the GGUF right after this returns.
 *
 * @param gguf  Open GGUF file (not modified beyond reads)
 * @return      New model (free with pn_vision_free), or NULL on a NULL argument, a
 *              non-clip file, or missing/undecodable tensors (messages on stderr)
 *
 * Thread-safe: yes (no shared state)
 * Reentrant:   yes
 * Signal-safe: no
 */
pn_vision_model* pn_vision_load(pn_gguf_file *gguf);

/**
 * Free a model returned by pn_vision_load. NULL is a no-op.
 *
 * Thread-safe: yes (for distinct models)
 * Reentrant:   yes
 * Signal-safe: no
 */
void pn_vision_free(pn_vision_model *m);

/**
 * Encode an RGB8 image into decoder-space embeddings. Runs a Pixtral-style tower
 * (patch conv, RMSNorm, 2D rotary attention, SiLU-gated MLP) on the image resized so
 * its long edge is about 1024 px, then a LightOnOCR-style projector (RMSNorm, spatial
 * merge, two linears with GELU between).
 *
 * @param m         Loaded model
 * @param rgb       Interleaved RGB8 pixels, w*h*3 bytes
 * @param w, h      Image size in pixels (> 0)
 * @param out_embd  Receives a malloc'd [rows * proj_dim] float buffer (caller frees)
 * @param out_dim   Receives proj_dim (may be NULL)
 * @return          Number of rows (image tokens, row-major over the merged grid), or -1
 *                  on a NULL model/image or non-positive size (outputs untouched)
 *
 * Thread-safe: TBD (reads the model only; on macOS changes the calling thread's
 *              Accelerate threading during the tower and restores it afterwards)
 * Reentrant:   TBD
 * Signal-safe: no
 */
int pn_vision_encode(pn_vision_model *m, const uint8_t *rgb, int w, int h,
                     float **out_embd, int *out_dim);

#ifdef __cplusplus
}
#endif

#endif /* PN_PIXTRAL_VISION_H */
