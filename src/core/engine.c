/* SPDX-License-Identifier: AGPL-3.0-or-later */
/* Copyright (C) 2026 David (Corey) Herrera */
#include "engine.h"
#include "grammar_mask_shortlist.h"
#include "grammar_row_dump.h"
#include "sampler_topn_tail.h"
#include "sampler_topn_surgery.h"
#include "sampler_margin_draw.h"
#include "chat.h"
#include "bpe_pretokenize.h"
#include "load_guard.h"
#include "../formats/gguf.h"
#include "../formats/gguf_fttw_overlay.h"
#include "../formats/gguf_fttw_repack.h"
#include "../formats/gguf_dequant.h"
#include "../models/llama.h"
#include "../models/phi3.h"
#include "../models/rwkvx.h"
#include "../models/qwen35moe.h"
#include "act_stats.h"
#include "ngram_draft.h"
#include "spec_sampling_chooser.h"
#include "../models/bonsai.h"
#include "../models/pixtral_vision.h"
#include "../memory/arena.h"
#include "mem/pn_admission.h"
#include "mem/pn_kv_geometry.h"
#include "mem/pn_memwatch.h"
#include "mem/pn_ladder.h"
#include "mem/pn_host_admission.h"
#include "mem/pn_session_host_fit.h"
#include "qweight.h"
#include "sampling_modifiers.h"
#include "host_stage_timer.h"
#ifdef USE_THREADS
#include "threadpool/threadpool.h"
#endif
#ifdef USE_CUDA
#include "cuda/fc_cuda.h"
#endif
#ifdef USE_METAL
#include "metal/fm_metal.h"
#endif
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <math.h>
#include <pthread.h>

/* ============================================================
 * GPT-2 byte-level BPE decoding
 *
 * GPT-2 maps bytes to Unicode codepoints to avoid control chars:
 *   bytes 33-126 → same codepoint (printable ASCII)
 *   bytes 161-172 → same codepoint
 *   bytes 174-255 → same codepoint
 *   bytes 0-32, 127-160, 173 → Unicode 256+ (Ā, ā, Ă, ...)
 * This function reverses that mapping.
 * ============================================================ */

static int gpt2_unicode_to_byte(unsigned int cp) {
    /* Direct-mapped ranges: printable ASCII and high Latin-1 */
    if (cp >= 33 && cp <= 126) return (int)cp;
    if (cp >= 161 && cp <= 172) return (int)cp;
    if (cp >= 174 && cp <= 255) return (int)cp;
    /* Remapped bytes: 0-32, 127-160, 173 are mapped to 256+ */
    if (cp >= 256 && cp < 256 + 33 + 34 + 1) {
        static int reverse_map[69];
        static int initialized = 0;
        if (!initialized) {
            int idx = 0;
            for (int b = 0; b < 256; b++) {
                if ((b >= 33 && b <= 126) || (b >= 161 && b <= 172) || (b >= 174 && b <= 255))
                    continue;
                reverse_map[idx++] = b;
            }
            initialized = 1;
        }
        int offset = (int)(cp - 256);
        if (offset >= 0 && offset < 69) return reverse_map[offset];
    }
    return -1;
}

/*
 * GPT-2 byte encoding: maps each raw byte to its GPT-2 Unicode codepoint.
 * This is the inverse of gpt2_bytes_decode.
 * Must be applied to input text before BPE tokenization.
 */
char* gpt2_bytes_encode(const char *input) {
    if (!input) return NULL;

    /* Build forward map: byte -> codepoint */
    static unsigned int byte_to_cp[256];
    static int initialized = 0;
    if (!initialized) {
        int idx = 256;
        for (int b = 0; b < 256; b++) {
            if ((b >= 33 && b <= 126) || (b >= 161 && b <= 172) || (b >= 174 && b <= 255)) {
                byte_to_cp[b] = (unsigned int)b;
            } else {
                byte_to_cp[b] = (unsigned int)idx++;
            }
        }
        initialized = 1;
    }

    size_t len = strlen(input);
    /* Worst case: each byte becomes 2-byte UTF-8 (codepoints 128-2047) */
    char *out = malloc(len * 3 + 1);
    if (!out) return NULL;

    size_t oi = 0;
    const unsigned char *p = (const unsigned char *)input;

    for (size_t i = 0; i < len; i++) {
        unsigned int cp = byte_to_cp[p[i]];
        if (cp < 0x80) {
            out[oi++] = (char)cp;
        } else if (cp < 0x800) {
            out[oi++] = (char)(0xC0 | (cp >> 6));
            out[oi++] = (char)(0x80 | (cp & 0x3F));
        } else {
            out[oi++] = (char)(0xE0 | (cp >> 12));
            out[oi++] = (char)(0x80 | ((cp >> 6) & 0x3F));
            out[oi++] = (char)(0x80 | (cp & 0x3F));
        }
    }

    out[oi] = '\0';
    return out;
}

char* gpt2_bytes_decode(const char *input) {
    if (!input) return NULL;

    size_t len = strlen(input);
    char *out = malloc(len + 1);
    if (!out) return NULL;

    size_t oi = 0;
    const unsigned char *p = (const unsigned char *)input;

    while (*p) {
        unsigned int cp;
        int bytes;

        if (*p < 0x80) {
            cp = *p;
            bytes = 1;
        } else if ((*p & 0xE0) == 0xC0) {
            cp = (*p & 0x1F) << 6;
            if (p[1]) { cp |= (p[1] & 0x3F); bytes = 2; }
            else { out[oi++] = (char)*p; p++; continue; }
        } else if ((*p & 0xF0) == 0xE0) {
            cp = (*p & 0x0F) << 12;
            if (p[1] && p[2]) {
                cp |= (p[1] & 0x3F) << 6;
                cp |= (p[2] & 0x3F);
                bytes = 3;
            } else { out[oi++] = (char)*p; p++; continue; }
        } else if ((*p & 0xF8) == 0xF0) {
            cp = (*p & 0x07) << 18;
            if (p[1] && p[2] && p[3]) {
                cp |= (p[1] & 0x3F) << 12;
                cp |= (p[2] & 0x3F) << 6;
                cp |= (p[3] & 0x3F);
                bytes = 4;
            } else { out[oi++] = (char)*p; p++; continue; }
        } else {
            out[oi++] = (char)*p; p++; continue;
        }

        int byte_val = gpt2_unicode_to_byte(cp);
        if (byte_val >= 0) {
            out[oi++] = (char)byte_val;
        } else {
            for (int i = 0; i < bytes; i++) {
                out[oi++] = (char)p[i];
            }
        }
        p += bytes;
    }

    out[oi] = '\0';
    return out;
}

/* ============================================================
 * Sampling defaults
 * ============================================================ */

pn_sampling_params pn_sampling_defaults(void) {
    return (pn_sampling_params){
        .temperature = 0.8f,
        .top_p = 0.9f,
        .top_k = 40,
        .repeat_penalty = 1.1f,
        .repeat_last_n = 64,
    };
}

/* ============================================================
 * Format detection
 * ============================================================ */

bool pn_is_gguf(const char *path) {
    FILE *f = fopen(path, "rb");
    if (!f) return false;

    char magic[4];
    size_t n = fread(magic, 1, 4, f);
    fclose(f);

    return (n == 4 && memcmp(magic, "GGUF", 4) == 0);
}

bool pn_is_safetensors(const char *path) {
    FILE *f = fopen(path, "rb");
    if (!f) return false;

    /* Safetensors: 8-byte LE header length, then '{' JSON */
    uint8_t buf[9];
    size_t n = fread(buf, 1, 9, f);
    fclose(f);

    if (n < 9) return false;
    uint64_t header_len;
    memcpy(&header_len, buf, 8);
    return (header_len < 10000000 && buf[8] == '{');
}

/* ============================================================
 * Architecture registry
 * ============================================================ */

pn_forward_fn pn_get_forward_fn(pn_arch arch) {
    switch (arch) {
        case PN_ARCH_LLAMA:   return pn_llama_forward;
        case PN_ARCH_QWEN2:   return pn_llama_forward;
        case PN_ARCH_MISTRAL: return pn_llama_forward;
        case PN_ARCH_MIXTRAL: return pn_llama_forward;
        case PN_ARCH_PHI:     return pn_phi3_forward;
        case PN_ARCH_QWEN35MOE: return pn_qwen35moe_forward;
        case PN_ARCH_BONSAI:  return pn_bonsai_forward;
        default:              return NULL;
    }
}

static pn_arch detect_arch_from_gguf(pn_gguf_file *gguf) {
    const char *arch = pn_gguf_get_str(gguf, "general.architecture");
    if (!arch) return PN_ARCH_UNKNOWN;

    if (strcmp(arch, "llama") == 0) {
        /* Mixtral uses arch="llama" but has expert_count metadata */
        uint32_t n_experts = pn_gguf_get_u32(gguf, "llama.expert_count", 0);
        if (n_experts > 0) return PN_ARCH_MIXTRAL;
        return PN_ARCH_LLAMA;
    }
    if (strcmp(arch, "qwen2") == 0) return PN_ARCH_QWEN2;
    if (strcmp(arch, "qwen3") == 0) return PN_ARCH_QWEN3;
    if (strcmp(arch, "phi") == 0)   return PN_ARCH_PHI;
    if (strcmp(arch, "phi3") == 0)  return PN_ARCH_PHI;
    if (strcmp(arch, "mistral") == 0) return PN_ARCH_MISTRAL;
    /* MiniCPM5-1B-SFT (our orchestrator) is a plain LlamaForCausalLM: RoPE + GQA
     * (24L, 16Q/2KV, head_dim=128 decoupled from n_embd/n_head=96) + SwiGLU +
     * RMSNorm, no MiniCPM-4 scale_depth/scale_emb. convert_hf_to_gguf.py emits
     * general.architecture="llama" for it (that is why it runs on vanilla
     * llama.cpp out of the box), so the OFFICIAL openbmb/MiniCPM5-1B-GGUF is
     * already handled by the "llama" branch above with ZERO plumbing.
     *
     * This "minicpm" alias is only a defense-in-depth net for a community GGUF
     * that mis-tags this plain-Llama model as "minicpm". CAVEAT: such a file may
     * write its dims under the "minicpm.*" metadata prefix, which arch_prefix()
     * does not read (it uses "llama" for PN_ARCH_LLAMA) — so if a "minicpm"-
     * tagged file ever fails to load, that prefix mismatch is why. Prefer the
     * official arch="llama" GGUF; warn so the operator knows to check. */
    if (strcmp(arch, "minicpm") == 0) {
        fprintf(stderr, "engine: architecture='minicpm' -> routing to the dense "
                        "Llama-GQA path (PN_ARCH_LLAMA). If dims read wrong, the "
                        "GGUF likely uses a 'minicpm.*' metadata prefix; use the "
                        "official arch=\"llama\" MiniCPM5-1B-GGUF instead.\n");
        return PN_ARCH_LLAMA;
    }
    if (strcmp(arch, "qwen35moe") == 0) return PN_ARCH_QWEN35MOE;
    /* PrismML Bonsai-27B declares general.architecture="qwen35": the SAME hybrid
     * GDN + sparse-full-attention family as Nex-N2-mini, but a DENSE SwiGLU FFN
     * (no MoE) and native-ternary weights. It gets its OWN arch (PN_ARCH_BONSAI)
     * and its own forward entry (bonsai_forward.c), which reuses the shared
     * qwen35moe blocks. A "qwen35" file with routed experts (should any exist)
     * still falls to the qwen35moe MoE path. */
    if (strcmp(arch, "qwen35") == 0)
        return pn_bonsai_is_bonsai_gguf(gguf) ? PN_ARCH_BONSAI
                                              : PN_ARCH_QWEN35MOE;

    fprintf(stderr, "engine: unknown architecture '%s'\n", arch);
    return PN_ARCH_UNKNOWN;
}

/* ============================================================
 * BPE tokenizer extraction from GGUF
 * ============================================================ */

static ft_tokenizer* build_tokenizer_from_gguf(pn_gguf_file *gguf) {
    pn_gguf_arr *tokens_arr = pn_gguf_get_arr(gguf, "tokenizer.ggml.tokens");
    if (!tokens_arr) {
        fprintf(stderr, "engine: no tokenizer.ggml.tokens in GGUF\n");
        return NULL;
    }

    if (tokens_arr->elem_type != GGUF_TYPE_STRING) {
        fprintf(stderr, "engine: tokenizer.ggml.tokens is not string array\n");
        free(tokens_arr);
        return NULL;
    }

    ft_tokenizer *tok = ft_new();
    if (!tok) {
        free(tokens_arr);
        return NULL;
    }

    uint64_t n_vocab = tokens_arr->n;
    const uint8_t *p = (const uint8_t *)tokens_arr->data;

    for (uint64_t i = 0; i < n_vocab; i++) {
        uint64_t slen;
        memcpy(&slen, p, 8);
        p += 8;

        char *token_str = malloc(slen + 1);
        if (!token_str) break;
        memcpy(token_str, p, slen);
        token_str[slen] = '\0';
        p += slen;

        ft_add_token(tok, token_str);
        free(token_str);
    }

    free(tokens_arr);

    fprintf(stderr, "engine: loaded %llu vocab tokens\n", (unsigned long long)n_vocab);

    /* Detect tokenizer model type */
    const char *tok_model = pn_gguf_get_str(gguf, "tokenizer.ggml.model");

    if (tok_model && strcmp(tok_model, "llama") == 0) {
        /* SentencePiece / Unigram tokenizer */
        ft_set_model_type(tok, 1);

        pn_gguf_arr *scores_arr = pn_gguf_get_arr(gguf, "tokenizer.ggml.scores");
        if (scores_arr && scores_arr->elem_type == GGUF_TYPE_FLOAT32) {
            ft_set_scores(tok, (float *)scores_arr->data, (int)scores_arr->n);
            fprintf(stderr, "engine: loaded %llu unigram scores (SentencePiece)\n",
                    (unsigned long long)scores_arr->n);
        }
        free(scores_arr);
    } else {
        /* BPE tokenizer — load merges */
        pn_gguf_arr *merges_arr = pn_gguf_get_arr(gguf, "tokenizer.ggml.merges");
        if (merges_arr && merges_arr->elem_type == GGUF_TYPE_STRING) {
            const uint8_t *mp = (const uint8_t *)merges_arr->data;
            uint64_t n_merges = merges_arr->n;

            for (uint64_t i = 0; i < n_merges; i++) {
                uint64_t slen;
                memcpy(&slen, mp, 8);
                mp += 8;

                char *merge_str = malloc(slen + 1);
                if (!merge_str) break;
                memcpy(merge_str, mp, slen);
                merge_str[slen] = '\0';
                mp += slen;

                char *space = strchr(merge_str, ' ');
                if (space) {
                    *space = '\0';
                    ft_add_merge(tok, merge_str, space + 1);
                }
                free(merge_str);
            }

            fprintf(stderr, "engine: loaded %llu merges (BPE)\n", (unsigned long long)n_merges);
            free(merges_arr);
        }

        /* Select the BPE pre-tokenizer scanner from tokenizer.ggml.pre. */
        const char *pre = pn_gguf_get_str(gguf, "tokenizer.ggml.pre");
        ft_set_pre_type(tok, pn_bpe_pre_from_name(pre));
        fprintf(stderr, "engine: BPE pre-tokenizer = %s\n", pre ? pre : "(unset)");
    }

    uint32_t bos_id = pn_gguf_get_u32(gguf, "tokenizer.ggml.bos_token_id", 0);
    uint32_t eos_id = pn_gguf_get_u32(gguf, "tokenizer.ggml.eos_token_id", 0);

    const char *bos_tok = ft_id_to_token(tok, (int)bos_id);
    const char *eos_tok = ft_id_to_token(tok, (int)eos_id);
    if (bos_tok) ft_set_bos_token(tok, bos_tok);
    if (eos_tok) ft_set_eos_token(tok, eos_tok);

    return tok;
}

/* KV Phase 1: the per-session context window the generic (dense-llama) KV cache
 * and its scores buffer are sized against. Historically hard-clamped to 4096,
 * which is what forced admission to CAP/drop-oldest on a longer chat. Now driven
 * by the configured context, clamped to the per-session hard cap
 * (PN_SESSION_MAX_CTX, default 40000 tokens) so a session can hold a real
 * conversation without the drop-oldest — while still bounding the pre-allocated
 * KV reservation. A sliding_window, if set, still wins (a genuine ring). The
 * hybrid (qwen35moe/bonsai/ornith) path has its own window knob
 * (QWEN35_MAX_SEQ) and does not use this. */
#define PN_SESSION_MAX_CTX_DEFAULT 40000

int pn_dense_session_max_seq(const pn_config *cfg) {
    int cap = PN_SESSION_MAX_CTX_DEFAULT;
    const char *e = getenv("PN_SESSION_MAX_CTX");
    if (e && e[0]) {
        int v = atoi(e);
        if (v > 0)
            cap = v;
    }
    int max_seq = cfg->n_ctx > 0 ? cfg->n_ctx : cap;
    if (cfg->sliding_window > 0 && cfg->sliding_window < max_seq)
        max_seq = cfg->sliding_window;
    /* `--ctx-cap` is the operator's per-model window and it MUST bound the
     * reservation, not just the advertised number. Without this the dense KV
     * ring was sized from the GGUF's own context_length whatever the cap said
     * — a 1.2 GB model reserving ~6 GB of host RSS at --ctx-cap 4096, on an
     * 8 GB box. The hybrid ring has always honoured it (qwen35moe_max_seq_cfg);
     * this is the dense path catching up. */
    if (cfg->ctx_cap > 0 && cfg->ctx_cap < max_seq)
        max_seq = cfg->ctx_cap;
    if (max_seq > cap)
        max_seq = cap;
    return max_seq;
}

/* ============================================================
 * Arena sizing
 * ============================================================ */

size_t pn_dense_arena_bytes(const pn_config *cfg) {
    /* Arena must hold ALL per-layer buffers accumulated across layers,
     * since the arena only resets between forward calls.
     *
     * Per-layer allocations:
     *   normed: n_embd, q: n_head*hd, k: n_kv_head*hd, v: n_kv_head*hd,
     *   attn_out: n_head*hd, attn_proj: n_embd, ffn_normed: n_embd,
     *   gate: n_ff, up: n_ff, ffn_hidden: n_ff, ffn_out: n_embd
     *   = 5*n_embd + 3*n_ff + 2*n_kv_head*head_dim floats per layer
     *
     * One-time allocations:
     *   hidden: n_embd, scores_buf: n_head*max_seq,
     *   final_normed: n_embd, logits: n_vocab
     */
    int max_seq = pn_dense_session_max_seq(cfg);
    int head_dim = pn_head_dim(cfg);

    size_t per_layer = (size_t)(5 * cfg->n_embd)
                     + (size_t)(3 * cfg->n_ff)
                     + (size_t)(2 * cfg->n_kv_head * head_dim);

    /* MoE layers need extra buffers: gate_logits + expert intermediates */
    if (cfg->n_experts > 0) {
        per_layer += (size_t)cfg->n_experts             /* gate_logits */
                   + (size_t)(2 * cfg->n_experts_used)  /* top_idx/top_wt (int->float) */
                   + (size_t)cfg->n_embd                /* ffn_out accumulator */
                   + (size_t)cfg->n_ff                  /* exp_gate */
                   + (size_t)cfg->n_ff                  /* exp_up */
                   + (size_t)cfg->n_ff                  /* exp_hid */
                   + (size_t)cfg->n_embd;               /* exp_out */
    }

    size_t one_time = (size_t)cfg->n_embd              /* hidden */
                    + (size_t)(cfg->n_head * max_seq)   /* scores_buf */
                    + (size_t)cfg->n_embd               /* final_normed */
                    + (size_t)cfg->n_vocab;             /* logits */

    /* Alignment overhead: 16 bytes per allocation */
    size_t n_allocs = (size_t)cfg->n_layer * 11 + 5;
    if (cfg->n_experts > 0) n_allocs += (size_t)cfg->n_layer * 5;
    size_t align_overhead = n_allocs * 16;

    size_t total_floats = per_layer * (size_t)cfg->n_layer + one_time;
    size_t total = total_floats * sizeof(float) + align_overhead;
    /* Add 25% safety margin for alignment rounding accumulation */
    return total + total / 4;
}

/* ============================================================
 * Model loading
 * ============================================================ */

pn_model* pn_load(const char *path) {
    return pn_load_ex(path, NULL);
}

/* ============================================================
 * Pre-flight admission control (MemoryAware §6, §11 Q7)
 *
 * Engine-level so the eval path (eval_ppl) — which crash-looped on the 32k
 * OOM-reboot — is covered as well as the server. Reads the published
 * pn_mem_state (host RAM + PSI + per-die VRAM) and the measured cost LUT;
 * refuses/caps before any memory-growing work starts. Fail-open: if the
 * memory-watch loop is not running there is no truth to check against, so it
 * returns FIT (no behavior change).
 * ============================================================ */

/* Is q4 KV in effect? The load's own choice for a qwen35moe model
 * (pn_qwen35moe_model_kv_q4); else B1's default: ON unless QWEN35_KV_Q4=0. */
static bool pn_admit_kv_q4(const pn_model *model) {
    if (model && model->config.arch == PN_ARCH_QWEN35MOE && model->weights)
        return pn_qwen35moe_model_kv_q4(model->weights) != 0;
    const char *e = getenv("QWEN35_KV_Q4");
    return !(e && e[0] == '0');
}

/* Path-aware cost model (DECIDED 2026-07-09): is this model GPU-resident —
 * weights uploaded, the CUDA forward selected? Then the per-request prefill
 * host cost is ~0 (the O(N^2) scores live on-die; measured on a K80 host) and admission
 * admits the full context up to max_seq. Anything else is the CPU-forward /
 * host-fallback path that the conservative LUT models.
 *
 * Recognizes ALL three CUDA forwards — the MoE forward AND the two dense
 * forwards (bonsai/Ornith and plain-Llama), which are equally GPU-resident.
 * Gating on the MoE pointer alone mislabeled Ornith path=cpu and charged it a
 * phantom host_need from the CPU-fallback LUT, tripping the ladder into
 * throttling a GPU-resident model (the 429 storm / idle-die split). */
bool pn_model_on_accelerator(const pn_model *model) {
#ifdef USE_CUDA
    if (model && (model->forward == pn_qwen35moe_forward_cuda ||
                  model->forward == pn_bonsai_forward_cuda ||
                  model->forward == pn_llama_forward_cuda))
        return true;
#endif
#ifdef USE_TT
    /* The Tenstorrent forward is equally accelerator-resident: its weights live in the
     * card's GDDR6 and never come back. Leaving it out here is the same defect the comment
     * above records for Ornith — a resident model charged a phantom host_need from the
     * CPU-fallback LUT, and then throttled. */
    if (model && (model->forward == pn_llama_forward_tt ||
                  model->forward == pn_qwen35moe_forward_tt))
        return true;
#endif
    (void)model;
    return false;
}

static bool pn_admit_gpu_resident(const pn_model *model) {
    return pn_model_on_accelerator(model);
}

/* The per-session context window admission promises (the per-session hard cap).
 * It MUST equal the window the model's KV ring is actually sized to, or admission
 * either over-promises (admit a prompt the ring can't hold) or under-promises
 * (reject one it could). Resolution, in order:
 *   1. a per-model ctx_cap (set at load) — the KV ring is sized to exactly this
 *      (ornith's 40K), so admission promises exactly this. Wins over everything.
 *   2. QWEN35_MAX_SEQ env — the hybrid ring's global override.
 *   3. else: the HYBRID (qwen35moe/bonsai) ring defaults to PN_QWEN35MOE_MAX_SEQ
 *      (the ring's own default), so admission must match it — NOT the generic
 *      pn_dense_session_max_seq, which governs the DENSE llama KV cache only. This is
 *      what keeps saber (no ctx_cap, unset env) reporting its true 4096 window
 *      instead of a phantom 40K the ring can't hold. Always clamped to the ring
 *      ceiling + trained ctx.
 * A DENSE model short-circuits all of that and returns pn_dense_session_max_seq()
 * verbatim, because on the dense path that function *is* the ring: pn_state_new
 * allocates the KV cache with the very same call. Same function, one number,
 * no drift possible. */
static int pn_admit_max_window(const pn_model *model) {
    int cap = PN_QWEN35MOE_MAX_SEQ_CAP;
    bool hybrid = model && (model->config.arch == PN_ARCH_QWEN35MOE ||
                            model->config.arch == PN_ARCH_BONSAI);
    /* Dense arches: pn_dense_session_max_seq IS the ring (pn_state_new sizes the KV
     * cache with it, and it now honours ctx_cap), so derive the promise from
     * the identical call rather than re-deriving it. Re-deriving is how the
     * promise could exceed the ring — a ctx_cap above PN_SESSION_MAX_CTX_DEFAULT
     * was advertised in full while the ring stopped at 40000. */
    if (model && !hybrid)
        return pn_dense_session_max_seq(&model->config);
    int w;
    if (model && model->config.ctx_cap > 0) {
        w = model->config.ctx_cap;
    } else {
        const char *e = getenv("QWEN35_MAX_SEQ");
        if (e && atoi(e) > 0)
            w = atoi(e);
        else if (hybrid)
            w = PN_QWEN35MOE_MAX_SEQ;   /* match the ring's own default */
        else
            w = model ? pn_dense_session_max_seq(&model->config)
                      : PN_SESSION_MAX_CTX_DEFAULT;
    }
    if (w <= 0) w = PN_SESSION_MAX_CTX_DEFAULT;
    if (w > cap) w = cap;
    /* Never promise more than the model's own trained context. */
    if (model && model->config.n_ctx > 0 && w > model->config.n_ctx)
        w = model->config.n_ctx;
    return w;
}

int pn_session_ctx_cap(const pn_model *model) {
    return pn_admit_max_window(model);
}

/* One prompt through the TT card's BATCHED prefill, or a refusal. Tokens already in the
 * card's own `uint32_t` form.
 *
 * The card multiplies 32 rows against every weight tile it reads whatever we put in them
 * (a matvec is Mt=1 and the other 31 rows are the tile quantum's zero padding), so a
 * 32-position batch costs the card exactly what one position costs. pn_llama_prefill_tt
 * returns NULL WITHOUT having touched the state when it refuses, and the per-token loop at
 * the call site is what runs then — so this is a pure accelerator, never a behaviour fork.
 *
 * EVERY PREFILL SITE IN THE PROCESS MUST COME THROUGH HERE. Measured 2026-08-26: the
 * chunkwise Ornith prefill shipped at 24x and reached exactly ONE caller, because
 * pn_stream_tokens() — which is what the CLI takes for any prompt that is NOT `--raw`, and
 * what the chat REPL takes — had a per-token loop of its own and never asked. A 1,630-token
 * prompt ran 1,632 forwards in 295 s with nothing on stderr, and the shipped rung was
 * invisible rather than broken.
 *
 * Returns the LAST prompt position's logits (caller owns), or NULL to mean "not taken". */
static tm_tensor *pn_prefill_tt_batched_tokens(pn_model *model, pn_state *state,
                                               const uint32_t *tokens, int n_tokens)
{
#ifdef USE_TT
    if (!model || !tokens || n_tokens < 2)
        return NULL;
    /* TWO FAMILIES, TWO BATCHED PREFILLS, and they are not the same code because they are
     * not the same trade. The llama one refuses on-card attention and batches everything
     * position-dependent on the host; Ornith's mixers both run WHOLE on the card and are
     * single-position there, so its batch reaches each row through the link. */
    if (model->forward == pn_llama_forward_tt)
        return pn_llama_prefill_tt(model, tokens, NULL, n_tokens, state);
    if (model->forward == pn_qwen35moe_forward_tt)
        return pn_qwen35moe_prefill_tt(model, tokens, n_tokens, state);
    return NULL;
#else
    (void)model; (void)state; (void)tokens; (void)n_tokens;
    return NULL;
#endif
}

/* Did the batched prefill ABORT part way (not decline)? Then the session was invalidated and
 * the per-token loop must not replay over the half-written card state: the call fails. */
static int pn_prefill_tt_aborted(const pn_model *model)
{
#ifdef USE_TT
    return model && model->forward == pn_qwen35moe_forward_tt &&
           pn_qwen35moe_prefill_tt_aborted(model);
#else
    (void)model;
    return 0;
#endif
}

/* The same, for the tokenizer's signed ids. Negative ids are dropped, which is what the
 * per-token loops at every call site do. */
static tm_tensor *pn_prefill_tt_batched(pn_model *model, pn_state *state,
                                        const int *prompt_tokens, int n_prompt_tokens)
{
#ifdef USE_TT
    uint32_t *toks;
    tm_tensor *logits;
    int n = 0;

    if (!model || !prompt_tokens || n_prompt_tokens < 2)
        return NULL;
    if (model->forward != pn_llama_forward_tt && model->forward != pn_qwen35moe_forward_tt)
        return NULL;
    toks = malloc((size_t)n_prompt_tokens * sizeof(*toks));
    if (!toks)
        return NULL;
    for (int i = 0; i < n_prompt_tokens; i++)
        if (prompt_tokens[i] >= 0)
            toks[n++] = (uint32_t)prompt_tokens[i];
    logits = pn_prefill_tt_batched_tokens(model, state, toks, n);
    free(toks);
    return logits;
#else
    (void)model; (void)state; (void)prompt_tokens; (void)n_prompt_tokens;
    return NULL;
#endif
}

/* Apply the reactive ladder's rung-3 prefill clamp to a configured window.
 * PURE: no logging, no state — pn_admit logs the transitions, every other
 * caller (the /v1/models advertisement, the admin surface, the kv_cap a
 * completion reports) just wants the number. Keeping the clamp in ONE place is
 * what stops the advertised window from drifting away from the window
 * admission actually enforces.
 *
 * The GPU-resident exemption is Phase 10 policy — see the long comment in
 * pn_admit. */
static int pn_apply_window_clamp(const pn_mem_state *st, int configured,
                                 bool gpu_resident, bool kv_q4,
                                 pn_ladder_plan *ladder_out) {
    pn_ladder_plan ladder;
    pn_ladder_plan_from_state(st, 1, configured, kv_q4, &ladder);
    if (ladder_out)
        *ladder_out = ladder;
    if (!gpu_resident && ladder.prefill_window > 0 &&
        ladder.prefill_window < configured)
        return ladder.prefill_window;
    return configured;
}

int pn_effective_ctx_window(const pn_model *model) {
    int configured = pn_admit_max_window(model);
    /* No memory truth to read ⇒ nothing can be clamping, so the configured
     * window IS the effective one (same fail-open as pn_admit). */
    if (!pn_memwatch_running())
        return configured;
    pn_mem_state st;
    pn_memwatch_read(&st);
    return pn_apply_window_clamp(&st, configured, pn_admit_gpu_resident(model),
                                 pn_admit_kv_q4(model), NULL);
}

int pn_model_graph_residency_read(const pn_model *model,
                                  pn_model_graph_residency *out) {
    if (!out)
        return -1;
    memset(out, 0, sizeof(*out));
#ifdef USE_CUDA
    /* Only the hybrid qwen35 family captures graphs today. Bonsai shares that
     * forward wholesale (bonsai_forward.c delegates), so it shares the gauge. */
    if (model && (model->config.arch == PN_ARCH_QWEN35MOE ||
                  model->config.arch == PN_ARCH_BONSAI) && model->weights)
        return pn_qwen35moe_gpu_graph_residency(
            (const pn_qwen35moe_model *)model->weights, out);
#else
    (void)model;
#endif
    return -1;
}

int pn_model_attn_residency_read(const pn_model *model,
                                 pn_model_attn_residency *out) {
    if (!out)
        return -1;
    memset(out, 0, sizeof(*out));
#ifdef USE_CUDA
    /* Same family as the graph gauge, for the same reason: the on-die
     * attention path lives in the hybrid qwen35 forward, which bonsai shares
     * wholesale. */
    if (model && (model->config.arch == PN_ARCH_QWEN35MOE ||
                  model->config.arch == PN_ARCH_BONSAI) && model->weights)
        return pn_qwen35moe_gpu_attn_residency(
            (const pn_qwen35moe_model *)model->weights, out);
#else
    (void)model;
#endif
    return -1;
}

/* Tightest reporting die (worst free VRAM headroom) — the conservative die to
 * schedule against for the single-model server without threading a die id
 * through the CUDA weight structs. Returns -1 if no die reports (CPU/host). */
static int pn_admit_worst_die(const pn_mem_state *st) {
    int worst = -1;
    uint64_t worst_free = 0;
    for (int i = 0; i < st->n_dies && i < PN_MEM_MAX_DIES; i++) {
        if (st->per_die_budget[i] == 0)
            continue;
        if (worst < 0 || st->per_die_free[i] < worst_free) {
            worst = i;
            worst_free = st->per_die_free[i];
        }
    }
    return worst;
}

/* ============================================================
 * Effective-context-window observability (Phase 10)
 *
 * Before this, nothing reported the window a model was actually serving: the
 * ladder logged only when it clamped, never when it released, and no status
 * surface exposed the number. A serve could silently halve a 131072-token
 * window and neither the operator nor the client could tell. That made the
 * clamp bug invisible AND unverifiable.
 *
 * Fix: one greppable `[window]` line per model whenever its effective window
 * CHANGES — in both directions — plus one on the first admission so the
 * steady-state number is always in the log. Change-only, so a healthy serve
 * emits exactly one line per model for the life of the process.
 *
 * The same number is now also ADVERTISED: pn_effective_ctx_window feeds
 * `context_window` on GET /v1/models and GET /admin/models, and it is what a
 * completion reports as `kv_cap`. The log stays because it is the only surface
 * that shows the TRANSITIONS (clamp/restore); the endpoint shows the current
 * value.
 * ============================================================ */

#define PN_WINDOW_LOG_SLOTS 16

static struct {
    const pn_model *model;
    int             window;      /* last-logged effective window, 0 = free slot */
} g_window_log[PN_WINDOW_LOG_SLOTS];
static pthread_mutex_t g_window_log_lock = PTHREAD_MUTEX_INITIALIZER;

/* Short, stable, human-greppable tag for a model: its GGUF basename plus its
 * die placement. The dies are what distinguish two instances of the SAME model
 * file (e.g. nex-n2-mini run twice, on dies 2,3 and 4,5). */
static void pn_model_tag(const pn_model *model, char *buf, size_t n)
{
    const char *base = "model";
    if (model && model->path) {
        const char *slash = strrchr(model->path, '/');
        base = slash ? slash + 1 : model->path;
    }

    char dies[64];
    dies[0] = '\0';
    if (model && model->n_gpu_dies > 0) {
        size_t off = 0;
        off += (size_t)snprintf(dies + off, sizeof(dies) - off, " dies=");
        for (int i = 0; i < model->n_gpu_dies && off < sizeof(dies) - 1; i++)
            off += (size_t)snprintf(dies + off, sizeof(dies) - off, "%s%d",
                                    i ? "," : "", model->gpu_dies[i]);
    }
    snprintf(buf, n, "%s%s", base, dies);
}

static void pn_log_effective_window(const pn_model *model, int configured,
                                    int effective, pn_mem_level level,
                                    const char *note)
{
    if (!model || effective <= 0)
        return;

    pthread_mutex_lock(&g_window_log_lock);

    int slot = -1, free_slot = -1;
    for (int i = 0; i < PN_WINDOW_LOG_SLOTS; i++) {
        if (g_window_log[i].model == model) { slot = i; break; }
        if (!g_window_log[i].model && free_slot < 0) free_slot = i;
    }

    int prev = 0;
    if (slot < 0) {
        slot = free_slot;
        if (slot < 0) {                  /* table full — log every time */
            pthread_mutex_unlock(&g_window_log_lock);
            return;
        }
        g_window_log[slot].model = model;
    } else {
        prev = g_window_log[slot].window;
        if (prev == effective) {         /* unchanged — stay quiet */
            pthread_mutex_unlock(&g_window_log_lock);
            return;
        }
    }
    g_window_log[slot].window = effective;

    pthread_mutex_unlock(&g_window_log_lock);

    char tag[256];
    pn_model_tag(model, tag, sizeof(tag));

    if (prev == 0)
        fprintf(stderr, "[window] %s: effective context %d (configured %d) "
                        "[mem %s]\n",
                tag, effective, configured, pn_mem_level_name(level));
    else if (effective < prev)
        fprintf(stderr, "[window] %s: effective context CLAMPED %d -> %d "
                        "(configured %d) [mem %s: %s]\n",
                tag, prev, effective, configured, pn_mem_level_name(level),
                note ? note : "");
    else
        fprintf(stderr, "[window] %s: effective context RESTORED %d -> %d "
                        "(configured %d) [mem %s]\n",
                tag, prev, effective, configured, pn_mem_level_name(level));
}

pn_admit_status pn_admit(pn_model *model, int n_tokens, int *fit_tokens_out) {
    if (fit_tokens_out)
        *fit_tokens_out = n_tokens;

    if (n_tokens <= 0)
        return PN_ADMIT_FIT;
    /* No memory truth to check against: fail open on memory, but the configured
     * window still binds (H2d). */
    if (!pn_memwatch_running()) {
        int window = pn_admit_max_window(model);
        if (window > 0 && n_tokens > window) {
            if (fit_tokens_out)
                *fit_tokens_out = window;
            return PN_ADMIT_CAPPED;
        }
        return PN_ADMIT_FIT;
    }

    pn_mem_state st;
    pn_memwatch_read(&st);

    bool kv_q4 = pn_admit_kv_q4(model);
    bool gpu_resident = pn_admit_gpu_resident(model);
    int die_id = pn_admit_worst_die(&st);
    int max_window = pn_admit_max_window(model);
    int configured_window = max_window;   /* pre-ladder, for the [window] log */
    /* Price against THIS model's real KV geometry, not a generic constant —
     * the KV Phase 1 correctness fix (pn_kv_geometry.h). 0 falls back to the
     * conservative constant inside admission. */
    uint64_t kv_per_token = pn_kv_bytes_per_token(model, kv_q4);

    /* Rung 3 of the reactive ladder (MemoryAware.md §7): under memory pressure,
     * clamp the admitted prefill window below the configured max so an
     * over-window prompt is CAPPED proactively. Only the plan's prefill_window
     * is consumed here (slots/q4 apply elsewhere), so the concurrency arg is
     * nominal.
     *
     * POLICY (Phase 10, DECIDED 2026-07-27) — the clamp does NOT apply to a
     * GPU-resident model. Two independent reasons:
     *
     *  1. It buys nothing. The rung exists to bound the TRANSIENT HOST
     *     activation peak of a long prefill — pn_admission's O(N^2) LUT. On the
     *     GPU path that peak measured on a K80 host is ~0 (the scores live on-die in
     *     bounded flash tiles) and pn_admit_request hard-sets host_need_bytes
     *     = 0. Every clamp ever observed on the K80 test machine sat next to
     *     `host_need=0 MiB`. Halving the window frees ZERO host RAM.
     *
     *  2. The harm lands on a LIVE session. The window is re-evaluated on every
     *     turn of an existing conversation, and serve turns a CAP into a hard
     *     SG_CONTEXT_OVER_BUDGET rejection. So a transient pressure blip does
     *     not degrade a live session gracefully — it fails its next turn
     *     outright, against a cap the operator never configured. Gating NEW
     *     work is admission's job and admission still does it (the PSI stall
     *     guard and the reserve-line check below both still reject); silently
     *     re-pricing an in-flight session's context is not a memory lever, it
     *     is a correctness-of-service bug.
     *
     * On the host/CPU-fallback path the clamp is kept as-is: those prefills
     * really do allocate lut_activation_peak(N) host bytes, and that is the
     * path that OOM-rebooted the box. */
    pn_ladder_plan ladder;
    int clamped = pn_apply_window_clamp(&st, max_window, gpu_resident, kv_q4,
                                        &ladder);
    if (clamped < max_window) {
        fprintf(stderr, "[ladder] %s -> prefill window clamped %d -> %d (%s)\n",
                pn_mem_level_name(st.level), max_window, clamped, ladder.note);
        max_window = clamped;
    }

    /* Observability (Phase 10): the window a model is ACTUALLY serving must be
     * greppable from the serve log, in both directions, so an operator can tell
     * a configured window from a clamped one. Logs on change only. */
    pn_log_effective_window(model, configured_window, max_window, st.level,
                            ladder.note);

    pn_admit_result res;
    pn_admit_request(n_tokens, kv_q4, kv_per_token, gpu_resident, die_id,
                     max_window, &st, &res);

    if (res.verdict == PN_ADMIT_CAP && fit_tokens_out)
        *fit_tokens_out = res.fit_tokens;

    fprintf(stderr,
            "[admit] n=%d path=%s -> %s (%s) host_need=%llu MiB "
            "host_avail=%llu MiB\n",
            n_tokens, gpu_resident ? "gpu" : "cpu",
            pn_admit_verdict_name(res.verdict), res.reason,
            (unsigned long long)(res.host_need_bytes >> 20),
            (unsigned long long)(res.host_avail_bytes >> 20));

    switch (res.verdict) {
    case PN_ADMIT_OK:     return PN_ADMIT_FIT;
    case PN_ADMIT_CAP:    return PN_ADMIT_CAPPED;
    case PN_ADMIT_REJECT: return PN_ADMIT_DENIED;
    }
    return PN_ADMIT_FIT;
}

/* Record an explicit die placement on a freshly-allocated model (pn_load_on).
 * Bookkeeping on every path; the GPU forwards read it where they support it. */
static void model_record_dies(pn_model *model, const int *dies, int n_dies)
{
    if (!model || !dies || n_dies <= 0)
        return;
    if (n_dies > PN_MODEL_MAX_DIES) {
        /* Truncating would record a placement NARROWER than the one the caller
         * asked for and the die manager bound — the accelerator would then
         * split fewer ways than it holds dies. Refuse the placement instead;
         * the model still loads, with no placement recorded, and the load gate
         * prices it as such. */
        fprintf(stderr, "pn_load_on: REFUSE placement on %d dies — the max is "
                "%d (PN_MODEL_MAX_DIES); ignoring the placement rather than "
                "silently narrowing it\n", n_dies, PN_MODEL_MAX_DIES);
        return;
    }
    for (int i = 0; i < n_dies; i++)
        model->gpu_dies[i] = dies[i];
    model->n_gpu_dies = n_dies;
}

#ifdef USE_CUDA
/* The die this model's process-default CUDA context belongs on. fc_init_on()
 * is first-call-wins, so this must be the placed die and not ordinal 0 —
 * otherwise a serve pinned to `--dies 2,3` still builds a full context on
 * die 0, which another serve owns. Unplaced models keep the legacy default. */
static int model_primary_die(const pn_model *model)
{
    return (model && model->n_gpu_dies > 0) ? model->gpu_dies[0] : 0;
}

/* Bind CUDA on the model's primary die. With no device the load guard
 * refuses the load, so say why (load_guard.h). */
static void engine_cuda_bind(const pn_model *model)
{
    fc_init_on(model_primary_die(model));
    if (!fc_available())
        pn_accel_refusal_note("no CUDA device visible on die %d (driver, "
                              "CUDA_VISIBLE_DEVICES)", model_primary_die(model));
}
#endif

pn_model* pn_load_ex(const char *path, const pn_load_opts *opts) {
    return pn_load_on(path, opts, NULL, 0);
}

static pn_model *pn_load_on_unfitted(const char *path, const pn_load_opts *opts,
                                     const int *dies, int n_dies);

/* Every load ends here. Where it may run (load_guard.h): a load asked to run
 * on an accelerator lands on one or is refused with the reason, never kept on
 * the host CPU forward; a CPU load must fit host RAM before it starts. Then the
 * model's sessions must fit host RAM before anyone allocates one
 * (pn_session_host_fit.h): a default window is capped to fit; an
 * operator-chosen one that does not fit refuses the load. */
pn_model* pn_load_on(const char *path, const pn_load_opts *opts,
                     const int *dies, int n_dies) {
    const bool cpu_planned = pn_load_guard_cpu_planned(opts && opts->cpu_only);
    if (cpu_planned && pn_load_guard_cpu_host_fit(path) != 0)
        return NULL;
    pn_accel_refusal_clear();
    uint64_t admitted_before = pn_host_admit_thread_total();
    pn_model *model = pn_load_on_unfitted(path, opts, dies, n_dies);
    /* The test fake stands where the upload would, and refuses. */
    if (model && !cpu_planned && pn_load_guard_test_fake_reason())
        pn_accel_refusal_note("%s", pn_load_guard_test_fake_reason());
    if (model && !cpu_planned && !pn_model_on_accelerator(model)) {
        pn_load_guard_refuse_accel(path, dies, n_dies);
        pn_model_free(model);   /* a failed upload already released its device memory */
        model = NULL;
    }
    /* Whatever this thread admitted during the load (dequantized weight
     * copies) belongs to the model: pn_model_free returns it. A failed load
     * already freed those copies, so return the charge now. */
    uint64_t admitted = pn_host_admit_thread_total() - admitted_before;
    if (!model) {
        pn_host_release(admitted);
        return NULL;
    }
    model->host_admitted = admitted;
    int sessions = opts && opts->host_sessions > 0 ? opts->host_sessions : 1;
    if (pn_session_host_fit(model, sessions, NULL, 0) != 0) {
        fprintf(stderr, "pn_load: refused - %s\n", pn_host_admit_last_refusal());
        pn_model_free(model);
        return NULL;
    }
    return model;
}

/* `PN_MIXW_BFP4=<map file>` (default off): repack the map's Bfp8
 * artifact tensors to Bfp4 in memory before any loader reads them. Oracle only for now (the card
 * path is plan 4.1), so it refuses without PN_FTTW_ORACLE. 0 = off or applied, -1 = refuse. */
static int engine_apply_mixw_bfp4(pn_gguf_file *gguf, const char *oracle_fttw)
{
    const char *map_path = getenv("PN_MIXW_BFP4");

    if (!map_path || !*map_path)
        return 0;
    if (!oracle_fttw || !*oracle_fttw) {
        fprintf(stderr, "pn_load: PN_MIXW_BFP4 needs PN_FTTW_ORACLE (the CPU oracle over the artifact)\n");
        return -1;
    }
    if (pn_gguf_repack_bfp4_from_map_file(gguf, map_path) < 0) {
        fprintf(stderr, "pn_load: PN_MIXW_BFP4=%s refused\n", map_path);
        return -1;
    }
    return 0;
}

static pn_model *pn_load_on_unfitted(const char *path, const pn_load_opts *opts,
                                     const int *dies, int n_dies) {
    if (!path) {
        fprintf(stderr, "pn_load: path is NULL\n");
        return NULL;
    }

    /* ---- Safetensors path ---- */
    if (pn_is_safetensors(path)) {
        const char *config_path = opts ? opts->config_path : NULL;

        /* If config.json provided, use HF/ternary safetensors loader */
        if (config_path) {
            pn_model *model = calloc(1, sizeof(pn_model));
            if (!model) return NULL;
            model->path = strdup(path);
            model_record_dies(model, dies, n_dies);

            /* Load weights via HF safetensors loader */
            pn_llama_qweights *qw = pn_llama_load_safetensors(path, config_path, &model->config);
            if (!qw) {
                free(model->path);
                free(model);
                return NULL;
            }
            model->weights = qw;
            model->forward = pn_llama_forward;

            /* Load tokenizer from directory containing config.json */
            char tok_dir[1024];
            strncpy(tok_dir, config_path, sizeof(tok_dir) - 1);
            tok_dir[sizeof(tok_dir) - 1] = '\0';
            char *last_slash = strrchr(tok_dir, '/');
            if (last_slash) *last_slash = '\0';
            else strcpy(tok_dir, ".");

            model->tokenizer = pn_llama_load_hf_tokenizer(tok_dir, &model->config);
            if (!model->tokenizer) {
                fprintf(stderr, "engine: warning: no tokenizer loaded (need vocab.json+merges.txt)\n");
            }

            model->chat_style = pn_chat_style_from_arch(model->config.arch);

#ifdef USE_THREADS
            model->pool = fw_pool_new(0);
            if (model->pool) {
                fprintf(stderr, "engine: thread pool created\n");
            }
#endif

            const char *fmt = (qw->layers && qw->layers[0].wq.type == PN_QUANT_TERNARY)
                              ? "ternary" : "F32";
            fprintf(stderr, "engine: loaded %s safetensors (%d layers, %d embd, %s weights)\n",
                   model->config.arch == PN_ARCH_QWEN2 ? "Qwen2" : "LLaMA",
                   model->config.n_layer, model->config.n_embd, fmt);

            return model;
        }

        /* RWKV-X safetensors path (no config.json) */
        bool quantize = opts ? opts->quantize : false;
        const char *vocab = opts ? opts->vocab_path : NULL;

        pn_model *model = calloc(1, sizeof(pn_model));
        if (!model) return NULL;
        model->path = strdup(path);
        model_record_dies(model, dies, n_dies);
        model->config.arch = PN_ARCH_RWKVX;

        model->weights = pn_rwkvx_load(path, &model->config, quantize);
        if (!model->weights) {
            free(model->path);
            free(model);
            return NULL;
        }
        model->forward = pn_rwkvx_forward;
        model->tokenizer = NULL;  /* RWKV-X uses internal trie tokenizer */

        if (vocab) {
            if (pn_rwkvx_load_tokenizer(model->weights, vocab) != 0) {
                fprintf(stderr, "engine: failed to load RWKV vocab: %s\n", vocab);
                pn_rwkvx_free(model->weights);
                free(model->path);
                free(model);
                return NULL;
            }
        }

        model->chat_style = PN_CHAT_NONE;

#ifdef USE_THREADS
        model->pool = fw_pool_new(0);
        if (model->pool) {
            fprintf(stderr, "engine: thread pool created\n");
        }
#endif

        const char *rwkv_type = pn_rwkvx_n_attn_layers(model->weights) > 0 ? "RWKV-X" : "RWKV-7";
        fprintf(stderr, "engine: loaded %s (%d layers, %d embd)\n",
               rwkv_type, model->config.n_layer, model->config.n_embd);

        return model;
    }

    /* ---- GGUF path (Llama, Phi, Qwen, Mistral, Mixtral) ---- */
    if (!pn_is_gguf(path)) {
        fprintf(stderr, "pn_load: unrecognized file format: %s\n", path);
        return NULL;
    }

    pn_gguf_file *gguf = pn_gguf_open(path);
    if (!gguf) return NULL;
    /* The card-format oracle: the CPU forward over the FTTW
     * artifact's own tiles instead of the GGUF's weights. Refused loudly, never skipped. */
    const char *oracle_fttw = getenv("PN_FTTW_ORACLE");
    if (oracle_fttw && *oracle_fttw && pn_gguf_attach_fttw(gguf, oracle_fttw) < 0) {
        fprintf(stderr, "pn_load: PN_FTTW_ORACLE=%s could not be attached\n", oracle_fttw);
        pn_gguf_close(gguf);
        return NULL;
    }
    if (engine_apply_mixw_bfp4(gguf, oracle_fttw) != 0) {
        pn_gguf_close(gguf);
        return NULL;
    }
    /* A card metadata GGUF (Pantheon Local's p150 install of a model too big to ship as a
     * GGUF beside its artifact) loads with the one artifact beside it, as the oracle above
     * does by env: the loaders read its weights, the TT backend uploads it as-is. */
    char card_artifact[1024] = "";
    if (!(oracle_fttw && *oracle_fttw) &&
        pn_gguf_attach_card_artifact(gguf, path, card_artifact, sizeof(card_artifact)) < 0) {
        fprintf(stderr, "pn_load: %s: its card artifact could not be attached\n", path);
        pn_gguf_close(gguf);
        return NULL;
    }

    pn_arch arch = detect_arch_from_gguf(gguf);
    if (arch == PN_ARCH_UNKNOWN) {
        fprintf(stderr, "engine: no architecture detected, trying llama\n");
        arch = PN_ARCH_LLAMA;
    }

    pn_model *model = calloc(1, sizeof(pn_model));
    if (!model) {
        pn_gguf_close(gguf);
        return NULL;
    }

    model->path = strdup(path);
    if (card_artifact[0])
        model->card_artifact = strdup(card_artifact);
    model_record_dies(model, dies, n_dies);
    model->config.arch = arch;
    /* A `cpu` placement is the host CPU forward, whatever accelerators this build
     * carries: no CUDA / Tenstorrent / Metal upload is attempted (Local H2g). */
    const bool accel_ok = !(opts && opts->cpu_only);
    if (!accel_ok)
        fprintf(stderr, "engine: cpu placement requested; no accelerator upload\n");
    (void)accel_ok;   /* a build with no accelerator never reads it */
    /* KV Phase 1: per-model context cap must be in the config BEFORE the arch
     * loader sizes the KV ring, so ornith can run a 40K window without moving
     * saber's. The arch loaders copy it into their own config. */
    if (opts && opts->ctx_cap > 0)
        model->config.ctx_cap = opts->ctx_cap;

    if (arch == PN_ARCH_PHI) {
        /* Phi-3: fused QKV + fused gate+up layout */
        pn_phi3_qweights *qw = pn_phi3_load_gguf_quantized(gguf, &model->config);
        if (qw) {
            model->weights = qw;
            model->forward = pn_phi3_forward;
            fprintf(stderr, "engine: using phi3 quantized weights (zero-copy mmap)\n");
        } else {
            pn_model_free(model);
            pn_gguf_close(gguf);
            return NULL;
        }
    } else if (arch == PN_ARCH_QWEN35MOE) {
        /* Qwen3.5-MoE: hybrid SSM + full-attention + shared-expert MoE */
        pn_qwen35moe_model *qw = pn_qwen35moe_load(gguf, &model->config);
        if (qw) {
            model->weights = qw;
            model->forward = pn_qwen35moe_forward;
            fprintf(stderr, "engine: using qwen35moe weights (zero-copy mmap)\n");
            /* The load's KV format, before the upload's die admission and any state. */
            if (opts && opts->kv_format != PN_KV_FORMAT_DEFAULT) {
                pn_qwen35moe_model_set_kv_format(qw, opts->kv_format);
                fprintf(stderr, "engine: KV cache %s (the load's choice)\n",
                        opts->kv_format == PN_KV_FORMAT_Q4 ? "Q4" : "fp32");
            }
            /* Explicit placement (pn_load_on) must reach the GPU upload's
             * die selection — stored on the arch model before the upload. */
            if (model->n_gpu_dies > 0)
                pn_qwen35moe_model_set_dies(qw, model->gpu_dies,
                                            model->n_gpu_dies);
            /* The session blocks the die gate must price, before the upload. */
            pn_qwen35moe_model_set_card_sessions(qw, opts ? opts->card_sessions : 0);
#ifdef USE_CUDA
            /* Single-die GPU path: upload weights to persistent VRAM and select
             * the CUDA forward. No GPU or a failed upload leaves the CPU forward
             * selected, which pn_load_on refuses (no silent CPU fallback).
             * fc_init_on() is idempotent: it dlopens libcuda + inits CUDA on the
             * ACQUIRED die (not ordinal 0); a no-op (leaving fc_available()
             * false) without a GPU. */
            if (accel_ok)
                engine_cuda_bind(model);
            if (accel_ok && fc_available()) {
                if (pn_qwen35moe_gpu_upload(qw) == 0 &&
                    pn_qwen35moe_model_gpu_loaded(qw)) {
                    model->forward = pn_qwen35moe_forward_cuda;
                    fprintf(stderr, "engine: qwen35moe on GPU (single die)\n");
                } else {
                    fprintf(stderr, "engine: qwen35moe GPU upload skipped/"
                                    "failed\n");
                }
            }
#endif
#ifdef USE_METAL
            /* Apple GPU path, mirroring the CUDA block above. fm_init() is
             * idempotent and returns non-zero on a machine with no Metal
             * device, so this whole block collapses to "keep the CPU forward"
             * off-Mac. No MTLResidencySet is created — deliberate, see
             * foundry/metal/qwen35moe_metal_spec.md §2.
             *
             * PHASE 0: the selected forward has no kernels yet; it reports
             * NOT-IMPLEMENTED per token and delegates to the CPU forward, so
             * selecting it is output-neutral. */
            if (accel_ok && fm_init() == 0 && fm_available() &&
                pn_qwen35moe_metal_upload(qw) == 0) {
                model->forward = pn_qwen35moe_forward_metal;
                fprintf(stderr, "engine: qwen35moe Metal backend selected "
                                "(Phase 0 scaffold — compute still on CPU)\n");
            } else {
                fprintf(stderr, "engine: qwen35moe Metal upload skipped/"
                                "failed; using CPU forward\n");
            }
#endif
#ifdef USE_TT
            /* Tenstorrent, the MoE checkpoint (Qwen3.6): the upload
             * refuses it unless the census arm (PN_TT_Q36_CENSUS=1) is asked for, until the MoE
             * block exists (Phase 5). One model, one residency: only when nothing else claimed it. */
            if (accel_ok && model->forward == pn_qwen35moe_forward) {
                if (pn_qwen35moe_tt_upload(model) == 0 && pn_qwen35moe_model_tt_loaded(model)) {
                    model->forward = pn_qwen35moe_forward_tt;
                    fprintf(stderr, "engine: qwen35moe on Tenstorrent (matmuls on card)\n");
                } else {
                    fprintf(stderr, "engine: qwen35moe Tenstorrent upload skipped/failed; "
                                    "no CPU fallback\n");
                }
            }
#endif
        } else {
            pn_model_free(model);
            pn_gguf_close(gguf);
            return NULL;
        }
    } else if (arch == PN_ARCH_BONSAI) {
        /* PrismML Bonsai-27B: qwen35 hybrid family, dense SwiGLU FFN + native
         * ternary. Own arch + own forward entry (bonsai_forward.c), reusing the
         * shared qwen35moe loader/upload/lifecycle. */
        void *bw = pn_bonsai_load(gguf, &model->config);
        if (bw) {
            model->weights = bw;
            model->forward = pn_bonsai_forward;
            fprintf(stderr, "engine: using bonsai (dense-FFN ternary) weights (zero-copy mmap)\n");
            if (model->n_gpu_dies > 0)
                pn_bonsai_model_set_dies(bw, model->gpu_dies, model->n_gpu_dies);
#ifdef USE_CUDA
            if (accel_ok)
                engine_cuda_bind(model);
            if (accel_ok && fc_available()) {
                if (pn_bonsai_gpu_upload(bw) == 0 &&
                    pn_bonsai_model_gpu_loaded(bw)) {
                    model->forward = pn_bonsai_forward_cuda;
                    fprintf(stderr, "engine: bonsai on GPU (single die)\n");
                } else {
                    fprintf(stderr, "engine: bonsai GPU upload skipped/"
                                    "failed\n");
                }
            }
#endif
#ifdef USE_TT
            /* Tenstorrent: only when no other accelerator already claimed this model —
             * one model, one residency, the same rule the llama branch below states. */
            if (accel_ok && model->forward == pn_bonsai_forward) {
                if (pn_qwen35moe_tt_upload(model) == 0 &&
                    pn_qwen35moe_model_tt_loaded(model)) {
                    model->forward = pn_qwen35moe_forward_tt;
                    fprintf(stderr, "engine: qwen35 (dense) on Tenstorrent "
                                    "(matmuls on card)\n");
                } else {
                    fprintf(stderr, "engine: qwen35 Tenstorrent upload skipped/"
                                    "failed\n");
                }
            }
#endif
        } else {
            pn_model_free(model);
            pn_gguf_close(gguf);
            return NULL;
        }
    } else if (arch == PN_ARCH_LLAMA || arch == PN_ARCH_QWEN2 || arch == PN_ARCH_QWEN3 || arch == PN_ARCH_MISTRAL || arch == PN_ARCH_MIXTRAL) {
        /* Try quantized loader first */
        pn_llama_qweights *qw = pn_llama_load_gguf_quantized(gguf, &model->config);
        if (qw) {
            model->weights = qw;
            model->forward = pn_llama_forward;
            fprintf(stderr, "engine: using quantized weights (zero-copy mmap)\n");
#ifdef USE_CUDA
            /* Plain-Llama GPU forward: upload the (non-MoE) weights to a single
             * die and select the CUDA forward. No GPU or a failed upload is
             * refused by pn_load_on (the CPU oracle needs cpu_only).
             * fc_init_on() is idempotent and binds the ACQUIRED die. */
            if (accel_ok)
                engine_cuda_bind(model);
            if (accel_ok && fc_available()) {
                if (pn_llama_gpu_upload(model) == 0 &&
                    pn_llama_model_gpu_loaded(model)) {
                    model->forward = pn_llama_forward_cuda;
                    fprintf(stderr, "engine: llama on GPU (single die)\n");
                } else {
                    fprintf(stderr, "engine: llama GPU upload skipped/"
                                    "failed\n");
                }
            }
#endif
#ifdef USE_TT
            /* Tenstorrent: only when no other accelerator already claimed this model.
             * The residency handle is one slot (engine.h), and a model has one residency —
             * so a build carrying both families binds whichever family answered first
             * rather than uploading twice and leaking one. */
            if (accel_ok && model->forward == pn_llama_forward) {
                if (pn_llama_tt_upload(model) == 0 && pn_llama_model_tt_loaded(model)) {
                    model->forward = pn_llama_forward_tt;
                    fprintf(stderr, "engine: llama on Tenstorrent (projections on card)\n");
                } else {
                    fprintf(stderr, "engine: llama Tenstorrent upload skipped/"
                                    "failed\n");
                }
            }
#endif
        } else {
            /* Fallback to F32 dequant loader */
            pn_llama_weights *w = pn_llama_load_gguf(gguf, &model->config);
            if (!w) {
                pn_model_free(model);
                pn_gguf_close(gguf);
                return NULL;
            }
            model->weights = w;
            model->forward = pn_llama_forward_f32;
        }
    } else {
        fprintf(stderr, "pn_load: architecture not yet supported\n");
        pn_model_free(model);
        pn_gguf_close(gguf);
        return NULL;
    }

    model->tokenizer = build_tokenizer_from_gguf(gguf);
    if (!model->tokenizer) {
        fprintf(stderr, "pn_load: failed to build tokenizer\n");
        pn_model_free(model);
        pn_gguf_close(gguf);
        return NULL;
    }

    /* Detect chat template style */
    const char *chat_tmpl = pn_gguf_get_str(gguf, "tokenizer.chat_template");
    if (chat_tmpl)
        model->chat_style = pn_chat_detect_style(chat_tmpl);
    else
        model->chat_style = pn_chat_style_from_arch(arch);
    /* Some templates (e.g. MiniCPM5) begin with {{ bos_token }} — the model was
     * trained with BOS at position 0 and degenerates (whitespace) without it.
     * ChatML/Qwen/Nex templates do NOT lead with bos_token. Flag it when the
     * template emits bos_token before the first <|im_start|> role marker so the
     * prompt builder can prepend BOS for exactly those models. */
    if (chat_tmpl) {
        const char *b = strstr(chat_tmpl, "bos_token");
        const char *im = strstr(chat_tmpl, "<|im_start|>");
        model->chat_add_bos = (b && (!im || b < im)) ? 1 : 0;
        model->chat_trim_content = pn_chat_template_trims_content(chat_tmpl) ? 1 : 0;
    }
    fprintf(stderr, "engine: chat style: %s%s%s\n",
            pn_chat_style_name(model->chat_style),
            model->chat_add_bos ? " +bos" : "",
            model->chat_trim_content ? " +trim" : "");

#ifdef USE_THREADS
    {
        const char *nt = getenv("PN_THREADS");
        model->pool = fw_pool_new(nt ? atoi(nt) : 0);
    }
    if (model->pool) {
        fprintf(stderr, "engine: thread pool created\n");
        pn_qmv_set_pool(model->pool);
    }
#endif

    /* PN_ACT_STATS (mixw imatrix): mapped from this GGUF after every loader has taken its
     * weight pointers; refused loudly, never skipped. The weights own the GGUF here, so
     * pn_model_free closes it. */
    if (pn_act_stats_begin_from_env(gguf, model) < 0) {
        pn_model_free(model);
        return NULL;
    }

    const char *name = pn_gguf_get_str(gguf, "general.name");
    fprintf(stderr, "engine: loaded %s (%d layers, %d params roughly)\n",
           name ? name : path,
           model->config.n_layer,
           model->config.n_vocab * model->config.n_embd);

    return model;
}

int pn_model_keepalive(pn_model *model)
{
#ifdef USE_TT
    return pn_qwen35moe_keepalive_tt(model);
#else
    (void)model;
    return 0;
#endif
}

int pn_model_warm(pn_model *model)
{
#ifdef USE_TT
    return pn_qwen35moe_warm_tt(model);
#else
    (void)model;
    return 0;
#endif
}

void pn_model_free(pn_model *model) {
    if (!model) return;

    /* The end of a PN_ACT_STATS run: its files are written once, here. */
    pn_act_stats_finish(model);

    if (model->vision) pn_vision_free((pn_vision_model *)model->vision);

#ifdef USE_TT
    /* Tenstorrent residency shares the `gpu` slot, so it is claimed FIRST and by the
     * forward pointer — the one thing that says which family put a handle there. It NULLs
     * the slot, which is what keeps the CUDA free below from freeing a TT handle in a
     * build that carries both. */
    if (model->gpu && model->forward == pn_llama_forward_tt) pn_llama_tt_free(model);
    /* The qwen35 hybrid keeps its handle INSIDE its own weight struct, like the CUDA arms
     * of that family do, so it is not in the `gpu` slot and is claimed by its own forward
     * pointer. It must run BEFORE the weights are freed below: the residency holds pointers
     * to the very `pn_qweight`s it is keyed on. */
    if (model->forward == pn_qwen35moe_forward_tt) pn_qwen35moe_tt_free(model);
#endif
#ifdef USE_CUDA
    /* Plain-Llama GPU residency (pn_llama_gpu_upload); other arches keep their
     * GPU handle inside their own weight struct and free it there. */
    if (model->gpu) pn_llama_gpu_free(model);
#endif

    if (model->weights) {
        if (model->config.arch == PN_ARCH_RWKVX) {
            pn_rwkvx_free(model->weights);
        } else if (model->config.arch == PN_ARCH_QWEN35MOE) {
            pn_qwen35moe_model *qw = (pn_qwen35moe_model *)model->weights;
            pn_gguf_file *gguf = pn_qwen35moe_gguf(qw);
            pn_qwen35moe_free(qw);
            if (gguf) pn_gguf_close(gguf);
        } else if (model->config.arch == PN_ARCH_BONSAI) {
            void *bw = model->weights;
            pn_gguf_file *gguf = pn_bonsai_gguf(bw);
            pn_bonsai_free(bw);
            if (gguf) pn_gguf_close(gguf);
        } else if (model->config.arch == PN_ARCH_PHI) {
            pn_phi3_qweights *qw = (pn_phi3_qweights *)model->weights;
            pn_gguf_file *gguf = qw->gguf;
            pn_phi3_qweights_free(qw);
            if (gguf) pn_gguf_close(gguf);
        } else if (model->config.arch == PN_ARCH_LLAMA || model->config.arch == PN_ARCH_QWEN2 || model->config.arch == PN_ARCH_QWEN3 || model->config.arch == PN_ARCH_MISTRAL || model->config.arch == PN_ARCH_MIXTRAL) {
            /* Check if it's quantized or F32 weights by checking the tag */
            pn_llama_qweights *qw = (pn_llama_qweights *)model->weights;
            if (qw->is_quantized) {
                pn_gguf_file *gguf = qw->gguf;
                pn_llama_qweights_free(qw);
                if (gguf) pn_gguf_close(gguf);
            } else {
                pn_llama_weights *w = (pn_llama_weights *)model->weights;
                pn_gguf_file *gguf = w->gguf;
                pn_llama_weights_free(w);
                if (gguf) pn_gguf_close(gguf);
            }
        }
    }

    if (model->tokenizer) ft_free(model->tokenizer);
#ifdef USE_THREADS
    if (model->pool) {
        pn_qmv_set_pool(NULL);
        fw_pool_free(model->pool);
    }
#endif
    pn_host_release(model->host_admitted);
    free(model->path);
    free(model->card_artifact);
    free(model);
}

/* ============================================================
 * State management
 * ============================================================ */

pn_state* pn_state_new(pn_model *model) {
    if (!model) return NULL;

    pn_state *state = calloc(1, sizeof(pn_state));
    if (!state) return NULL;

    state->pos = 0;
    state->config = &model->config;

    /* RWKV-X uses its own state (recurrent + KV cache + workspace) */
    if (model->config.arch == PN_ARCH_RWKVX) {
        state->kv_cache = pn_rwkvx_state_new(model->weights);
        state->arena = NULL;
        if (!state->kv_cache) {
            free(state);
            return NULL;
        }
        return state;
    }

    /* Qwen3.5-MoE uses its own dual state (SSM conv/ssm + attention KV). */
    if (model->config.arch == PN_ARCH_QWEN35MOE) {
        state->kv_cache = pn_qwen35moe_state_new(model->weights);
        state->arena = NULL;
        if (!state->kv_cache) {
            free(state);
            return NULL;
        }
        return state;
    }

    /* Bonsai shares the qwen35moe dual-state (same hybrid GDN + attn-KV). */
    if (model->config.arch == PN_ARCH_BONSAI) {
        state->kv_cache = pn_bonsai_state_new(model->weights);
        state->arena = NULL;
        if (!state->kv_cache) {
            free(state);
            return NULL;
        }
        return state;
    }

    /* Allocate KV cache — sized to the per-session window (KV Phase 1: no longer
     * a hard 4096 clamp; driven by config + ctx_cap + PN_SESSION_MAX_CTX).
     *
     * This call is the definition of the ring, and pn_admit_max_window() (hence
     * pn_session_ctx_cap(), the scheduler's per-session gate, and
     * pn_effective_ctx_window(), the advertised number) resolves to the SAME
     * call on the dense path. The invariant that matters: the gate is never
     * above the ring. pn_effective_ctx_window may sit *below* it under memory
     * pressure — that is a transient prefill clamp, and clamping a permanent
     * allocation to a transient signal would strand a slot at whatever the box
     * happened to look like the instant it was created.
     *
     * The bytes here are eager (tm_zeros mallocs and memsets), all layers, fp32:
     *   2 * n_layer * n_kv_head * max_seq * head_dim * 4
     * and one of these exists per decode slot AND per prefix-cache session, so
     * an unbounded max_seq multiplies straight into host RSS. */
    int head_dim = pn_head_dim(&model->config);
    int max_seq = pn_dense_session_max_seq(&model->config);

    /* Host admission BEFORE the eager ring + arena: refuse with a reason
     * instead of OOM-ing the host (pn_host_admission.h). */
    uint64_t session_bytes = pn_session_host_bytes(&model->config);
    char what[96];
    snprintf(what, sizeof(what), "host KV ring + arena (%d positions)", max_seq);
    if (pn_host_admit(session_bytes, what, NULL, 0) != 0) {
        free(state);
        return NULL;
    }
    state->host_admitted = session_bytes;

    pn_llama_kv_cache *cache = pn_llama_kv_cache_new(
        model->config.n_layer,
        model->config.n_kv_head,
        max_seq,
        head_dim
    );
    if (!cache) {
        pn_host_release(session_bytes);
        free(state);
        return NULL;
    }
    state->kv_cache = cache;

    /* Create arena for scratch memory */
    size_t arena_size = pn_dense_arena_bytes(&model->config);
    state->arena = pn_arena_new(arena_size);
    if (!state->arena) {
        pn_llama_kv_cache_free(cache);
        pn_host_release(session_bytes);
        free(state);
        return NULL;
    }
    fprintf(stderr, "engine: arena allocated (%.1f KB)\n", (double)arena_size / 1024.0);

    return state;
}

void pn_state_free(pn_state *state) {
    if (!state) return;
    if (state->config && state->config->arch == PN_ARCH_RWKVX) {
        pn_rwkvx_state_free(state->kv_cache);
        free(state);
        return;
    }
    if (state->config && state->config->arch == PN_ARCH_QWEN35MOE) {
        pn_qwen35moe_state_free(state->kv_cache);
        free(state);
        return;
    }
    if (state->config && state->config->arch == PN_ARCH_BONSAI) {
        pn_bonsai_state_free(state->kv_cache);
        free(state);
        return;
    }
    if (state->kv_cache) {
        pn_llama_kv_cache_free((pn_llama_kv_cache *)state->kv_cache);
    }
    if (state->arena) {
        pn_arena_free(state->arena);
    }
    pn_host_release(state->host_admitted);
    free(state);
}

void pn_state_reset(pn_state *state) {
    if (!state) return;
    if (state->config && state->config->arch == PN_ARCH_RWKVX) {
        pn_rwkvx_state_reset(state->kv_cache);
        state->pos = 0;
        return;
    }
    if (state->config && state->config->arch == PN_ARCH_QWEN35MOE) {
        pn_qwen35moe_state_reset(state->kv_cache);
        state->pos = 0;
        return;
    }
    if (state->config && state->config->arch == PN_ARCH_BONSAI) {
        pn_bonsai_state_reset(state->kv_cache);
        state->pos = 0;
        return;
    }
    state->pos = 0;
    pn_llama_kv_cache *kv = (pn_llama_kv_cache *)state->kv_cache;
    if (kv) {
        for (int i = 0; i < kv->n_layers; i++) {
            if (kv->k_cache[i]) tm_zero_(kv->k_cache[i]);
            if (kv->v_cache[i]) tm_zero_(kv->v_cache[i]);
        }
    }
}

/* ============================================================
 * Sampling
 * ============================================================ */

/*
 * Uniform [0,1) draw for the final cumulative scan.
 *
 * Default: the process-global rand(), exactly as before.  When the request
 * carried an OpenAI `seed` (params->rng_seeded) the draw comes from that
 * request's OWN stream instead, advanced in place — so two identical requests
 * with the same seed replay the same tokens regardless of what other requests
 * are sampling concurrently.  See sampling_modifiers.h.
 */
static float rand_float(pn_sampling_params *params) {
    if (params && params->carried_uniform_set) {
        params->carried_uniform_set = 0;
        return params->carried_uniform;
    }
    if (params && params->rng_seeded)
        return pn_seeded_rng_next_float(&params->rng_state);
    return (float)rand() / ((float)RAND_MAX + 1.0f);
}

/* ------------------------------------------------------------------
 * Optimized token sampler.
 *
 * This is a drop-in, LOSSLESS replacement for the original sampler: for any
 * given logits / params / recent-token window and the same RNG state it
 * returns the exact same token id, bit-for-bit. The correctness contract is
 * enforced by tests/test_sampler_parity.c, which keeps a verbatim copy of the
 * original implementation and diffs the two over thousands of random cases.
 *
 * What changed (all lossless):
 *   - Top-K threshold is found with an O(V) quickselect (order statistic on the
 *     probabilities) instead of the original O(top_k * V) selection sort. The
 *     threshold is the exact same value (the top_k-th largest probability), so
 *     the "keep probs >= threshold" set is bit-identical, including ties.
 *   - Top-P runs over only the <= top_k survivors (sorted once), instead of
 *     repeated full-vocab argmax passes. Survivor probabilities are accumulated
 *     in descending order, reproducing the same cumulative sum and the same
 *     top_p_threshold value.
 *   - The full-vocab softmax (temperature scaling + normalization over the full
 *     vocab) is preserved exactly, so every candidate's probability is
 *     identical to the original, and the final cumulative random draw scans the
 *     same probabilities in the same (ascending index) order.
 *   - Per-call malloc/free of three ~V-sized buffers is replaced by a
 *     thread-local scratch buffer that is reused across calls (and grown on
 *     demand). Thread-local makes it safe if multiple threads ever sample
 *     concurrently, without any locking.
 * ------------------------------------------------------------------ */

typedef struct {
    float *probs;   /* softmax probabilities, indexed by token id */
    float *work;    /* scratch for quickselect / top-p candidate sort */
    int    cap;
} pn_sampler_scratch;

/* ------------------------------------------------------------------
 * Per-thread sampler buffers, freed when the thread exits (B6).
 *
 * The sampler scratch, the logprob snapshot and the grammar verdict/need
 * scratch are reused across calls on one thread.  The serve runs every
 * request on its own short-lived thread (token #1 is sampled there), so when
 * these were bare _Thread_local pointers each request left ~2 MiB behind on
 * the bank (two n_vocab float buffers) — still allocated, so malloc_trim could
 * not return it.  They now live in one heap block registered as a pthread key
 * value; the key's destructor frees the block at thread exit.  The destructor
 * gets the block as its argument and never reads _Thread_local storage, which
 * may already be torn down by then (macOS).
 * ------------------------------------------------------------------ */
typedef struct {
    pn_sampler_scratch scratch;
    float         *logprob_raw;     /* logprob snapshot (logprob_snapshot) */
    int            logprob_cap;
    unsigned char *grammar_verdict; /* grammar_verdict_scratch */
    int            grammar_verdict_cap;
    unsigned char *grammar_need;    /* grammar_need_scratch */
    int            grammar_need_cap;
} pn_sampler_thread_buffers;

static _Thread_local pn_sampler_thread_buffers *t_sampler_buffers = NULL;
static pthread_key_t  g_sampler_buffers_key;
static pthread_once_t g_sampler_buffers_once = PTHREAD_ONCE_INIT;
static int            g_sampler_buffers_key_ok = 0;

static void sampler_buffers_free(void *block)
{
    pn_sampler_thread_buffers *b = (pn_sampler_thread_buffers *)block;
    if (!b)
        return;
    free(b->scratch.probs);
    free(b->scratch.work);
    free(b->logprob_raw);
    free(b->grammar_verdict);
    free(b->grammar_need);
    free(b);
}

static void sampler_buffers_key_create(void)
{
    g_sampler_buffers_key_ok =
        pthread_key_create(&g_sampler_buffers_key, sampler_buffers_free) == 0;
}

/* This thread's buffer block, created (and armed for thread exit) on first
 * use.  NULL on OOM. */
static pn_sampler_thread_buffers *sampler_buffers(void)
{
    if (t_sampler_buffers)
        return t_sampler_buffers;
    pthread_once(&g_sampler_buffers_once, sampler_buffers_key_create);
    pn_sampler_thread_buffers *b = calloc(1, sizeof(*b));
    if (!b)
        return NULL;
    if (g_sampler_buffers_key_ok)
        pthread_setspecific(g_sampler_buffers_key, b);
    t_sampler_buffers = b;
    return b;
}

void pn_sampler_thread_release(void)
{
    if (!t_sampler_buffers)
        return;
    if (g_sampler_buffers_key_ok)
        pthread_setspecific(g_sampler_buffers_key, NULL);
    sampler_buffers_free(t_sampler_buffers);
    t_sampler_buffers = NULL;
}

static pn_sampler_scratch *sampler_scratch_get(int n) {
    pn_sampler_thread_buffers *b = sampler_buffers();
    if (!b)
        return NULL;
    pn_sampler_scratch *s = &b->scratch;
    if (s->cap < n) {
        free(s->probs); free(s->work);
        s->probs = malloc((size_t)n * sizeof(float));
        s->work  = malloc((size_t)n * sizeof(float));
        if (!s->probs || !s->work) {
            free(s->probs); free(s->work);
            s->probs = NULL; s->work = NULL; s->cap = 0;
            return NULL;
        }
        s->cap = n;
    }
    return s;
}

/* Quickselect: returns the k-th smallest element (0-indexed) of a[0..n-1],
 * partitioning a[] in place. Uses a deterministic middle pivot (no rand(), so
 * the global RNG that drives the final draw is untouched). The returned value
 * is the exact order statistic, so it matches the original selection sort's
 * threshold bit-for-bit. */
static float sampler_select_kth(float *a, int n, int k) {
    int lo = 0, hi = n - 1;
    while (lo < hi) {
        int i = lo, j = hi;
        float pivot = a[lo + ((hi - lo) >> 1)];
        while (i <= j) {
            while (a[i] < pivot) i++;
            while (a[j] > pivot) j--;
            if (i <= j) {
                float t = a[i]; a[i] = a[j]; a[j] = t;
                i++; j--;
            }
        }
        if (k <= j) hi = j;
        else if (k >= i) lo = i;
        else break;
    }
    return a[k];
}

/* ------------------------------------------------------------------
 * OpenAI `logprobs` support.
 *
 * The reported numbers must be the MODEL's log-probabilities, so they come from
 * a snapshot of the logits taken before the sampler touches them (bias,
 * penalties, temperature, top-k/top-p and the grammar mask all mutate the array
 * in place).  The snapshot lives in a thread-local buffer that is only ever
 * allocated for requests that actually asked for logprobs — a request without
 * them pays nothing.  The math itself is in sampling_modifiers.c.
 * ------------------------------------------------------------------ */

/* Copy the untouched logits aside (into this thread's buffer block).  Returns
 * NULL on OOM (logprobs are then simply not reported for this step — never a
 * generation failure). */
static const float *logprob_snapshot(const float *logits, int n_vocab) {
    pn_sampler_thread_buffers *b = sampler_buffers();
    if (!b) return NULL;
    if (b->logprob_cap < n_vocab) {
        float *grown = realloc(b->logprob_raw, (size_t)n_vocab * sizeof(float));
        if (!grown) return NULL;
        b->logprob_raw = grown;
        b->logprob_cap = n_vocab;
    }
    memcpy(b->logprob_raw, logits, (size_t)n_vocab * sizeof(float));
    return b->logprob_raw;
}

/* Hand the sink the chosen token's logprob and the top alternatives. */
static void emit_logprobs(const pn_logprob_sink *sink, const float *raw,
                          int n_vocab, uint32_t chosen) {
    uint32_t top_ids[PN_LOGPROBS_MAX_TOP];
    float    top_lp[PN_LOGPROBS_MAX_TOP];
    float    chosen_lp = 0.0f;
    int n_kept = pn_logprobs_extract(raw, n_vocab, chosen, sink->n_top,
                                     &chosen_lp, top_ids, top_lp);
    sink->emit(sink->user, chosen, chosen_lp, top_ids, top_lp, n_kept);
}

static int sampler_cmp_desc(const void *pa, const void *pb) {
    float x = *(const float *)pa, y = *(const float *)pb;
    return (x < y) - (x > y);   /* descending */
}

/* ------------------------------------------------------------------
 * Sparse-survivor sampler fast path (PN_SPARSE_SAMPLER).
 *
 * On the CUDA decode path the LM head's logits never leave the die whole: the
 * device top-k reduces them to k (id, logit) pairs and gpu_present_topk lays
 * those into the host row SPARSELY — every other entry is exactly -inf.  Only
 * k <= 40 tokens can possibly be sampled, but sample_token_core below still
 * walks all 248320 of them EIGHT times and calls expf() on every one.  On the
 * K80 host's 2-core CPU that walk is the entire measured cost of serving at
 * temperature 0.7 instead of 0.  This path does the identical arithmetic over
 * just the survivors.
 *
 * WHY THE DETECTION IS A BIT COMPARE.  The obvious test, `logits[i] >
 * -INFINITY`, is WRONG here and silently so: the engine builds with
 * -ffast-math, which implies -ffinite-math-only, under which the compiler is
 * entitled to assume no operand is infinite and folds the comparison to a
 * constant TRUE.  Measured on the release flags: over a row with 7 real
 * survivors in 1000 entries, `> -INFINITY` counts 1000 and isfinite()/the bit
 * compare both count 7.  An earlier attempt at this fast path used `>
 * -INFINITY`, so every row "overflowed" the survivor cap on entry, the fast
 * path never once executed, and the A/B that concluded "the host sampler costs
 * nothing" was measuring the old path against itself.  Compare the bit pattern
 * the producer actually writes; it is the one test -ffast-math cannot reason
 * away.
 *
 * EXACTNESS.  Bit-identical to the full-vocab path below, not merely close:
 *   - max: non-survivors are -inf and lose every comparison.
 *   - softmax sum: the full path accumulates ascending index and adds
 *     expf(-inf - max) = +0.0f for each non-survivor; adding +0.0f to a float
 *     never changes it, so gathering ascending and skipping those terms yields
 *     the same sum, in the same order, bit for bit.
 *   - top-k: the threshold is the top_k-th largest probability over the WHOLE
 *     row, i.e. over {S survivor probs} u {V-S zeros}.  When top_k >= S the
 *     order statistic falls in the zero region and is 0.0f, so nothing is cut;
 *     otherwise it is the top_k-th largest survivor prob.  Both reproduce the
 *     quickselect's value exactly, ties included.
 *   - top-p: the candidate set is `probs[i] > 0.0f`, which no non-survivor can
 *     join, so the descending sort and the cumulative threshold are identical.
 *   - draw: the cumulative scan crosses r at the same index, because the terms
 *     it skips are +0.0f and a zero term cannot be the crossing point (the
 *     scan would already have returned at the previous index).  The exhausted
 *     fallbacks return the same ids, including the all-zero case's token 0.
 *
 * GREEDY TOO (added after the temp>0 path measured out).  temp 0 is only ONE
 * pass over n_vocab, so it looks like it has nothing to give — but that pass is
 * a serially dependent float compare (`logits[i] > logits[best]`, best carried
 * across the whole row) and the gather's pass is an independent compare against
 * a constant.  Measured on the same row and the same machine, the argmax pass
 * costs 0.394 ms and the gather pass 0.057 ms: same O(n_vocab), 6.9x apart.
 * Greedy therefore goes through the SAME sparse_gather and takes its argmax
 * over the survivors.  Bit-exact for the same reason the max above is: every
 * non-survivor is -inf and loses a strict `>` to any survivor, and both scans
 * run ascending so ties resolve to the same lowest id.
 *
 * Scope: only when the row really is sparse.  A dense row — full_logits mode,
 * which the serve forces for tool grammar, logit_bias and logprobs — trips the
 * survivor cap in the first PN_SURV_MAX entries and falls through to the full
 * path unchanged.
 *
 * NaN caveat (both paths, unchanged by the greedy addition): a NaN logit is not
 * the -inf bit pattern, so it is gathered as a survivor.  `>` is false in both
 * directions against a NaN, so the full scan keeps whatever `best` held when it
 * met one and the survivor scan keeps whatever ITS `best` held — those need not
 * be the same index.  A NaN in the LM head is a model-level failure that has
 * never been observed here, and -ffast-math already declares them out of
 * contract; recorded so the divergence is not a surprise if one ever appears.
 * ------------------------------------------------------------------ */

/* The device top-k caps at FC_TOPK_MAX (256) survivors; anything past that is
 * a dense row by definition. Kept local so engine.c still builds without CUDA. */
#define PN_SURV_MAX 256

/* Bit pattern of -INFINITY as written by the sparse-logits producer. */
#define PN_NEG_INF_BITS 0xff800000u

_Static_assert(sizeof(float) == sizeof(uint32_t),
               "sparse-survivor detection punnes float through uint32_t");

/* ------------------------------------------------------------------
 * SURVIVOR HINT (LAUNCH lane). A backend that built a sparse row itself (the TT forward's
 * top-N row, `PN_TT_ROW_RESET`, default on) knows its finite entries; it hands them to the NEXT draw
 * from that row (pn_sampler_row_hint, called by the decode loop right before sampling the
 * tensor) and the sampler's survivor scans read the list instead of 248K floats. The list is
 * the row's finite set by construction, ascending, so every consumer sees exactly what the
 * scan would have found. Honoured only for the row it names, never under a live tool grammar
 * (the mask rewrites rows), dropped by any dense refill and at the end of every draw.
 * ------------------------------------------------------------------ */
static _Thread_local const float *t_row_hint_row = NULL;
static _Thread_local const uint32_t *t_row_hint_ids = NULL;
static _Thread_local int t_row_hint_n = 0;
static unsigned long g_row_hint_uses = 0;
unsigned long pn_sampler_row_hint_uses(void) { return g_row_hint_uses; }

void pn_sampler_row_hint(const tm_tensor *t)
{
    t_row_hint_row = NULL;
    if (!t || t->is_sparse || t->dtype != TM_F32 || t->nnz <= 0 || t->nnz > PN_SURV_MAX ||
        !t->data)
        return;
    t_row_hint_ids = (const uint32_t *)((const float *)t->data + t->numel);
    t_row_hint_n = (int)t->nnz;
    t_row_hint_row = (const float *)t->data;
}

/* The hinted survivors of `row` (ascending, finite), or -1 when `row` has no hint. */
static int row_hint_gather(const float *row, int n_vocab, int *ids, float *vals, int cap)
{
    int S = 0;

    if (!t_row_hint_row || row != t_row_hint_row)
        return -1;
    g_row_hint_uses++;
    for (int k = 0; k < t_row_hint_n; k++) {
        const uint32_t i = t_row_hint_ids[k];
        uint32_t bits;

        if (i >= (uint32_t)n_vocab)
            continue;
        memcpy(&bits, &row[i], sizeof bits);
        if (bits == PN_NEG_INF_BITS)
            continue;
        if (S >= cap)
            return -2;
        ids[S] = (int)i;
        vals[S] = row[i];
        S++;
    }
    return S;
}

/* ------------------------------------------------------------------
 * SHORT ROW (engine.h pn_sampler_short_row). The row names the backend's exact top 257 of
 * the full row R (value desc, id asc), R holds no NaN and its 256th value is strictly above
 * its 257th, the bound b. The draw keeps the short row only where the decision provably
 * equals R's, and otherwise fetches R through the fill callback and runs the usual path on
 * it. A fill consumes no randomness, so the uniform and the RNG stream are R's either way.
 *
 * WHY IT IS EXACT.  (1) With no modifier that can raise a token (no logit_bias; presence,
 * frequency >= 0; repetition penalty >= 1 or standing down), every step before the draw maps
 * a value to one <= itself and -inf to -inf: tokens outside the list stay <= b, listed tokens
 * get the same values as in R. Anything else, and logprobs (they read all of R), fetch R at
 * entry. (2) Steer: R's argmax is the list's first entry. (3) Live mask: on R the shortlist is
 * the top 256 and its bound the largest value left out; on the short row both are the same
 * (unique set: 256th > 257th; bound = b). A CLEAR verdict depends only on the set, the bound,
 * the verdicts and k, and rewrites the row to the same survivors, so the rest of the draw is
 * R's. The outside bound sums in the shortlist's heap order, which on R depends on the whole
 * scan, so on a short row it does not decide: no CLEAR fetches R, which then takes its usual
 * shortlist / outside bound / full scan. (4) Greedy, mask not live: the listed best w is R's
 * argmax when w > b. (5) Sampled, mask not live: the margin draw reads only the top 64 after
 * surgery; when the short row's 64th is > b that is R's top 64, so the plan and its verdict
 * for the uniform are R's; a refusal (or a 64th <= b) fetches R and redoes the surgery before
 * the uniform is used. (6) The full scan, fail-open and fail-closed run only on R.
 *
 * Cleared by the fill, by a grammar CLEAR rewrite (the row is then the very row R's draw
 * would sample) and at the end of every draw.
 *
 * Red arms (gates only; PN_SHORT_ROW_RED): `bound` lets the grammar shortlist's outside bound
 * (heap-order sums) decide on the short row; `margin` skips the top-64 check; `greedy` skips
 * the argmax check. Each must break the identity gate.
 * ------------------------------------------------------------------ */
_Static_assert(PN_SHORT_ROW_K == PN_SURV_MAX + 1 && PN_SHORT_ROW_K == GMS_MAX + 1,
               "a short row must be one entry past every top-N cap");
_Static_assert(PN_SHORT_ROW_K == PN_QWEN35MOE_SHORT_ROW_K, "the forward presents PN_SHORT_ROW_K");
static _Thread_local const float *t_short_row = NULL;
static _Thread_local float t_short_bound = 0.0f;
static _Thread_local pn_short_row_fill_fn t_short_fill = NULL;
static _Thread_local void *t_short_ctx = NULL;
static _Thread_local int t_short_refilled = 0;
static pn_short_row_stats g_short_stats;

void pn_sampler_short_row(const float *row, float bound, pn_short_row_fill_fn fill, void *ctx)
{
    t_short_row = fill ? row : NULL;
    t_short_bound = bound;
    t_short_fill = fill;
    t_short_ctx = ctx;
}

void pn_sampler_short_row_stats(pn_short_row_stats *out)
{
    if (!out)
        return;
    out->draws = __atomic_load_n(&g_short_stats.draws, __ATOMIC_RELAXED);
    out->decided = __atomic_load_n(&g_short_stats.decided, __ATOMIC_RELAXED);
    for (int i = 0; i < PN_SR_REASONS; i++)
        out->refill[i] = __atomic_load_n(&g_short_stats.refill[i], __ATOMIC_RELAXED);
    out->fill_failed = __atomic_load_n(&g_short_stats.fill_failed, __ATOMIC_RELAXED);
}

static int short_row_red(const char *arm)
{
    const char *e = getenv("PN_SHORT_ROW_RED");
    return e && strcmp(e, arm) == 0;
}

static int short_row_live(const float *logits)
{
    return t_short_row && t_short_row == logits;
}

/* Write R into `logits` when it is still the short row; 1 when it was fetched. */
static int short_row_refill(float *logits, int n_vocab, int reason)
{
    if (!short_row_live(logits))
        return 0;
    t_short_row = NULL;
    t_row_hint_row = NULL;
    t_short_refilled = 1;
    __atomic_fetch_add(&g_short_stats.refill[reason], 1ul, __ATOMIC_RELAXED);
    if (t_short_fill(t_short_ctx, logits, n_vocab) != 0) {
        __atomic_fetch_add(&g_short_stats.fill_failed, 1ul, __ATOMIC_RELAXED);
        fprintf(stderr, "sampler: short row fill failed; sampling the short row\n");
        return 0;
    }
    return 1;
}

/* Entry (WHY (1)): a modifier that can raise a token outside the list, or logprobs
 * (they read the whole raw row), needs R before anything reads the row. Written so a NaN
 * setting also fetches. */
static int short_row_entry_needs_full(const pn_sampling_params *p)
{
    return p->logprobs || (p->logit_bias && p->n_logit_bias > 0) ||
           !(p->presence_penalty >= 0.0f) || !(p->frequency_penalty >= 0.0f) ||
           (p->temperature > 0.0f && !(p->repeat_penalty >= 1.0f));
}

/* The grammar judges ids below its own n_vocab only (steer, shortlist, scan). The short
 * row's view of that range is R's only when every listed id lies inside it. */
static int short_row_past_grammar_vocab(const float *logits, int n_vocab,
                                        const pn_tool_grammar_hook *tg)
{
    if (!tg || tg->n_vocab <= 0 || tg->n_vocab >= n_vocab)
        return 0;
    for (int i = tg->n_vocab; i < n_vocab; i++) {
        uint32_t bits;
        memcpy(&bits, &logits[i], sizeof bits);
        if (bits != PN_NEG_INF_BITS)
            return 1;
    }
    return 0;
}

/* Sampled draw (WHY (5)): R's top 64 after surgery is the short row's when the 64th of the
 * short row is above bound (nothing outside the list can exceed bound). */
static int short_row_top64_clear(const float *logits, int n_vocab)
{
    uint32_t ids[PN_MARGIN_CAND];
    float vals[PN_MARGIN_CAND];

    if (short_row_red("margin"))
        return 1;
    return pn_margin_select_top(logits, n_vocab, PN_MARGIN_CAND, ids, vals) == PN_MARGIN_CAND &&
           vals[PN_MARGIN_CAND - 1] > t_short_bound;
}

/* Runtime override for the tests (the token-identity gate flips arms in
 * process). -1 = follow the env; 0/1 = force. */
static int g_sparse_sampler_override = -1;
void pn_sampler_set_sparse(int on) { g_sparse_sampler_override = on ? 1 : 0; }
void pn_sampler_clear_sparse(void) { g_sparse_sampler_override = -1; }

/* How many times the fast path actually RAN. The gate asserts this advances —
 * an inert fast path is exactly how the previous attempt passed a vacuous
 * bit-exactness test. */
static unsigned long g_sparse_hits = 0;
/* Top-N rows the sampler swapped for the dense row (pn_sampler_topn_row_exact false). */
static unsigned long g_topn_refills = 0;
unsigned long pn_sampler_topn_refills(void) { return g_topn_refills; }
unsigned long pn_sampler_sparse_hits(void) { return g_sparse_hits; }
/* Dense-path draws the margin draw decided (sampler_margin_draw.h). Process-wide, unlocked:
 * a gate counter, like the two above. */
static unsigned long g_margin_draws = 0;
unsigned long pn_sampler_margin_draws(void) { return g_margin_draws; }

/* Per thread: the speculative-sampling chooser asks for the fast path on its own -INF rows
 * (they are sparse by construction; gated byte-identical, run_empty_env_default.sh). */
static _Thread_local int t_sparse_sampler_force = 0;

static int sparse_sampler_enabled(void) {
    if (t_sparse_sampler_force) return 1;
    if (g_sparse_sampler_override >= 0) return g_sparse_sampler_override;
    static int v = -1;
    if (v < 0) v = getenv("PN_SPARSE_SAMPLER") ? 1 : 0;
    return v;
}

/* Set by the grammar wrapper when it APPLIED a mask THIS step. The masked row
 * is dense anyway (the serve arms full_logits while the mask is live), so this
 * is belt-and-braces: it keeps the fast path off even if that ever stops
 * holding.  Thread-local — the batched driver samples per slot on its own
 * thread.
 *
 * IT IS A PER-STEP FLAG AND MUST BE CLEARED PER STEP (sample_token does it, at
 * entry).  It used to be set and never cleared, which on the batched path — one
 * driver thread sampling EVERY slot — meant the first masked step of the first
 * tool call latched the sparse-sampler fast path OFF for every slot, every
 * session, for the life of the process.  That is a whole-process loss of the
 * largest measured win in the project, triggered by one grammar token. */
static _Thread_local int g_grammar_masked = 0;

/* Set while sample_token hands sample_token_core a row the grammar mask has
 * already decided (refilled and/or rewritten): the core's top-N refill must
 * not replace it. */
static _Thread_local int t_grammar_row_final = 0;

/* PN_GRAMMAR_SHORTLIST=0: always run the full-vocabulary mask (A/B, red arm). */
static int grammar_shortlist_on(void)
{
    static int v = -1;
    if (v < 0) {
        const char *env = getenv("PN_GRAMMAR_SHORTLIST");
        v = (env && env[0] == '0') ? 0 : 1;
    }
    return v;
}

/* PN_GRAMMAR_ROW_HINT=0: a shortlist-masked row's draw scans the vocabulary for its
 * survivors instead of reading them from the hint (grammar_hint_kept). */
static int grammar_row_hint_on(void)
{
    static int v = -1;
    if (v < 0) {
        const char *env = getenv("PN_GRAMMAR_ROW_HINT");
        v = (env && env[0] == '0') ? 0 : 1;
    }
    return v;
}

typedef struct {
    const pn_tool_grammar_hook *tg;
} grammar_allowed_ctx;

/* gms_allowed_fn over the hook, with the full mask's rule for a token with no
 * text (grammar_token_verdict). */
static bool grammar_token_allowed(void *ctx, uint32_t id)
{
    const pn_tool_grammar_hook *tg = ((grammar_allowed_ctx *)ctx)->tg;
    const char *txt = tg->token_text(tg->user, id);
    if (!(txt && *txt)) return !tg->end_allowed || tg->end_allowed(tg->st);
    return tg->allowed(tg->st, txt);
}

/* Per-thread n_vocab byte scratch for the full mask's verdicts (NULL on OOM:
 * the caller then asks allowed() again in pass 2). */
static unsigned char *grammar_verdict_scratch(int n)
{
    pn_sampler_thread_buffers *b = sampler_buffers();
    if (!b) return NULL;
    if (n > b->grammar_verdict_cap) {
        unsigned char *grown = realloc(b->grammar_verdict, (size_t)n);
        if (!grown) return NULL;
        b->grammar_verdict = grown;
        b->grammar_verdict_cap = n;
    }
    return b->grammar_verdict;
}

/* Prefix verdicts for one masked step: does the walk survive this 1- or 2-byte
 * prefix?  Every grammar walk here is absorbing at DEAD, so a token whose
 * prefix kills the walk is rejected without walking it.  Rows of the 2-byte
 * table are filled on first use (only for first bytes that survive). */
typedef struct {
    signed char one[256];          /* 1 alive, 0 dead */
    signed char two[256][256];     /* -1 unknown, 1 alive, 0 dead */
    bool two_ready[256];
} grammar_prefix_verdicts;

static void grammar_prefix_verdicts_init(const pn_tool_grammar_hook *tg,
                                         grammar_prefix_verdicts *pv)
{
    char one[2] = { 0, 0 };
    pv->one[0] = 1;
    for (int c = 1; c < 256; c++) {
        one[0] = (char)c;
        pv->one[c] = tg->allowed(tg->st, one) ? 1 : 0;
    }
    memset(pv->two_ready, 0, sizeof(pv->two_ready));
}

static bool grammar_prefix2_alive(const pn_tool_grammar_hook *tg, grammar_prefix_verdicts *pv,
                                  unsigned char c0, unsigned char c1)
{
    if (!pv->two_ready[c0]) {
        char two[3] = { (char)c0, 0, 0 };
        for (int c = 1; c < 256; c++) {
            two[1] = (char)c;
            pv->two[c0][c] = tg->allowed(tg->st, two) ? 1 : 0;
        }
        pv->two_ready[c0] = true;
    }
    return pv->two[c0][c1] != 0;
}

/* The full mask's verdict for one token.  A token with no text (an end token)
 * is allowed unless the hook's end_allowed says otherwise. */
static int grammar_token_verdict(const pn_tool_grammar_hook *tg, grammar_prefix_verdicts *pv,
                                 const char *txt)
{
    if (!(txt && *txt)) return !tg->end_allowed || tg->end_allowed(tg->st);
    const unsigned char c0 = (unsigned char)txt[0], c1 = (unsigned char)txt[1];
    if (!pv->one[c0]) return 0;
    if (!c1) return 1;                         /* a 1-byte token: already judged */
    if (!grammar_prefix2_alive(tg, pv, c0, c1)) return 0;
    return tg->allowed(tg->st, txt) ? 1 : 0;
}

/* Per-thread chunk-need scratch (one byte per 16 ids). */
static unsigned char *grammar_need_scratch(int n_chunks)
{
    pn_sampler_thread_buffers *b = sampler_buffers();
    if (!b) return NULL;
    if (n_chunks > b->grammar_need_cap) {
        unsigned char *grown = realloc(b->grammar_need, (size_t)n_chunks);
        if (!grown) return NULL;
        b->grammar_need = grown;
        b->grammar_need_cap = n_chunks;
    }
    return b->grammar_need;
}

/* The full mask on a TOP-N row without the whole dense row: judge every token
 * (prefix-filtered), fetch only the 16-entry chunks that hold an allowed token
 * (params->dense_chunks), and leave allowed tokens at their true logit and
 * everything else at `mask` — the same row the full mask leaves on a refilled
 * dense row.  Returns 1 when applied, 0 when unavailable (no callback, no
 * scratch, no allowed token, or the fetch failed — the row is then untouched
 * apart from fetched chunks, which hold true logits). */
static int grammar_mask_from_chunks(float *logits, int nv, const pn_sampling_params *params,
                                    const pn_tool_grammar_hook *tg, float mask)
{
    static _Thread_local grammar_prefix_verdicts pv;
    const int n_chunks = (nv + 15) / 16;
    unsigned char *verdict = grammar_verdict_scratch(nv);
    unsigned char *need = grammar_need_scratch(n_chunks);
    int survivors = 0;

    if (!params->dense_chunks || !verdict || !need)
        return 0;
    grammar_prefix_verdicts_init(tg, &pv);
    memset(need, 0, (size_t)n_chunks);
    for (int i = 0; i < nv; i++) {
        verdict[i] = (unsigned char)grammar_token_verdict(tg, &pv, tg->token_text(tg->user,
                                                                                    (uint32_t)i));
        if (verdict[i]) { need[i / 16] = 1; survivors++; }
    }
    if (survivors == 0 || params->dense_chunks(params->dense_ctx, need, logits, nv) != 0)
        return 0;
    for (int i = 0; i < nv; i++)
        if (!verdict[i]) logits[i] = mask;
    return 1;
}

/* Shortlist decisions the outside bound made (row 0 draws and verify rows alike). */
static unsigned long g_grammar_bound_steps = 0;
unsigned long pn_sampler_grammar_bound_steps(void)
{
    return __atomic_load_n(&g_grammar_bound_steps, __ATOMIC_RELAXED);
}

/* PN_GRAMMAR_EXACT_BOUND=0: the shortlist decides only when K allowed tokens clear its bound
 * (the behaviour before the outside bound, grammar_mask_shortlist.h). */
static int grammar_exact_bound_on(void)
{
    static int v = -1;
    if (v < 0) {
        const char *env = getenv("PN_GRAMMAR_EXACT_BOUND");
        v = (env && env[0] == '0') ? 0 : 1;
    }
    return v;
}

/* Per thread: the verify row a draw belongs to (0 = the decode row), for the tail counts. */
static _Thread_local int t_sample_row = 0;

/* The device's tail counts of this row for the outside bound: 1 when `out` was filled. */
static int grammar_outside_counts(const pn_sampling_params *params, float bound, int n_vocab,
                                  tnt_tail_bins *tb, uint32_t *counts, gms_outside *out)
{
    if (!params->tail_counts)
        return 0;
    tnt_tail_bins_make(bound, bound, params->temperature, n_vocab, tb);
    if (params->tail_counts(params->dense_ctx, t_sample_row, tb->keys, tb->bins, counts) != 0)
        return 0;
    out->bins = tb;
    out->bin_counts = counts;
    return 1;
}

/* PN_GRAMMAR_HINT_RED=1 (gates only): the hint drops the highest kept id. */
static int grammar_hint_red(void)
{
    static int v = -1;
    if (v < 0) {
        const char *e = getenv("PN_GRAMMAR_HINT_RED");
        v = (e && e[0] == '1') ? 1 : 0;
    }
    return v;
}

/* After a shortlist rewrite the row's finite entries are exactly the shortlisted ids still
 * finite: hand them, ascending, to this draw's survivor scans as the row hint, so the gather
 * (and a greedy argmax) reads <= GMS_MAX entries instead of the vocabulary. Nothing before
 * the draw can change the finite set (penalties leave -inf at -inf and a finite value finite;
 * the repeat penalty stands down under the mask). A NaN among them (never from a model) keeps
 * the scan: the sparse argmax and the full one would order it differently. */
static void grammar_hint_kept(const float *logits, const uint32_t *ids, int n)
{
    static _Thread_local uint32_t kept[GMS_MAX];
    int k = 0;

    for (int i = 0; i < n; i++) {
        uint32_t bits;
        memcpy(&bits, &logits[ids[i]], sizeof bits);
        if (bits == PN_NEG_INF_BITS)
            continue;
        if ((bits & 0x7f800000u) == 0x7f800000u && (bits & 0x007fffffu))
            return;
        uint32_t id = ids[i];
        int j = k++;
        for (; j > 0 && kept[j - 1] > id; j--) kept[j] = kept[j - 1];
        kept[j] = id;
    }
    if (grammar_hint_red() && k > 0)
        k--;
    t_row_hint_ids = kept;
    t_row_hint_n = k;
    t_row_hint_row = logits;
}

/* The shortlist decision for this row (grammar_mask_shortlist.h); true when the
 * row was rewritten to the masked survivors.  The outside bound charges every token outside
 * the shortlist at its bound first; the device's tail counts only when that cannot decide.
 * `hint`: set this draw's row hint to the survivors (sample_token_draw only). */
static int grammar_mask_from_shortlist(float *logits, int n_vocab,
                                       const pn_sampling_params *params,
                                       const pn_tool_grammar_hook *tg,
                                       const uint32_t *recent, int n_recent, int hint)
{
    uint32_t ids[GMS_MAX], counts[TNT_TAIL_BINS + 1];
    float bound = 0.0f;
    tnt_tail_bins tb;
    grammar_allowed_ctx ctx = { tg };
    const char *red = getenv("PN_TT_RED");
    gms_outside outside = { 0.0, NULL, NULL,
                            red && strcmp(red, "grammar_bound_blind") == 0 ? 0.0 : 1.0 };
    gms_sampling s = {
        .temperature = params->temperature, .top_k = params->top_k, .top_p = params->top_p,
        /* On a short row only CLEAR may decide: the outside bound sums in the shortlist's
         * heap order, which on the full row depends on the whole scan (SHORT ROW, WHY (3)). */
        .outside = grammar_exact_bound_on() &&
                   !(short_row_live(logits) && !short_row_red("bound")) ? &outside : NULL,
        .repeat_penalty = params->repeat_penalty,
        .presence_penalty = params->presence_penalty,
        .frequency_penalty = params->frequency_penalty,
        .has_logit_bias = params->logit_bias && params->n_logit_bias > 0,
        .recent = recent, .n_recent = recent ? n_recent : 0,
    };
    int nv = tg->n_vocab > 0 && tg->n_vocab < n_vocab ? tg->n_vocab : n_vocab;

    if (!grammar_shortlist_on())
        return 0;
    uint64_t ht0 = pn_ht_now();
    int n = gms_select_shortlist(logits, nv, ids, &bound);
    ht0 = pn_ht_lap(PN_HT_S_MASK_SELECT, ht0);
    if (n <= 0)
        return 0;
    outside.count = (double)(n_vocab - n);
    int verdict = gms_mask_from_shortlist(logits, n_vocab, ids, n, bound, &s,
                                          grammar_token_allowed, &ctx);
    if (verdict == GMS_UNDECIDED && s.outside && params->temperature > 0.0f &&
        grammar_outside_counts(params, bound, n_vocab, &tb, counts, &outside))
        verdict = gms_mask_from_shortlist(logits, n_vocab, ids, n, bound, &s,
                                          grammar_token_allowed, &ctx);
    if (verdict == GMS_DECIDED_BOUND)
        __atomic_fetch_add(&g_grammar_bound_steps, 1ul, __ATOMIC_RELAXED);
    if (verdict != GMS_UNDECIDED && short_row_live(logits))
        t_short_row = NULL;
    if (verdict != GMS_UNDECIDED && hint && grammar_row_hint_on())
        grammar_hint_kept(logits, ids, n);
    pn_ht_lap(PN_HT_S_MASK_DECIDE, ht0);
    return verdict != GMS_UNDECIDED;
}

/* Gather the sparse row's survivors in ASCENDING id order. Returns the count,
 * or -1 if the row holds more than PN_SURV_MAX finite entries (dense row → the
 * caller must use the full path). */
static int sparse_gather(const float *logits, int n_vocab,
                         int *ids, float *vals) {
    int S = row_hint_gather(logits, n_vocab, ids, vals, PN_SURV_MAX);

    if (S >= 0) return S;
    if (S == -2) return -1;
    S = 0;
    for (int i = 0; i < n_vocab; i++) {
        uint32_t bits;
        memcpy(&bits, &logits[i], sizeof bits);
        if (bits == PN_NEG_INF_BITS) continue;
        if (S >= PN_SURV_MAX) return -1;
        ids[S] = i; vals[S] = logits[i]; S++;
    }
    return S;
}

/* ------------------------------------------------------------------
 * Grammar-mask density tripwire (see engine.h).
 *
 * The serve arms the full-vocabulary row for the steps the grammar mask is
 * live on.  If it ever arms LATE, the mask lands on the device's sparse top-k
 * presentation and can only choose among ~k tokens — which is not a slow step,
 * it is the response_format defect: no legal token in the top-k means zero
 * survivors, and the walk fails open (AUTO) or force-terminates (FORCED).
 *
 * So count it, in the sampler, where the two facts meet.  The row test is a BIT
 * compare against the -inf pattern the sparse producer writes, never
 * `x > -INFINITY`: -ffast-math implies -ffinite-math-only, which folds the
 * float form to a constant and is exactly what hid the sparse sampler for a
 * month.  The scan early-exits the instant the survivor count passes the device
 * cap, so a correctly-armed DENSE row costs PN_SURV_MAX+1 iterations of a
 * load-and-compare and nothing else.
 * ------------------------------------------------------------------ */
static unsigned long g_grammar_mask_steps = 0;
static unsigned long g_grammar_sparse_rows  = 0;
unsigned long pn_sampler_grammar_mask_steps(void) { return g_grammar_mask_steps; }
/* Where masked steps go and what they cost: decided by the shortlist (no
 * vocabulary scan), or scanned in full; wall time inside the mask (refill,
 * shortlist, scan — not the draw).  Process-wide, like the counters above. */
static unsigned long g_grammar_shortlist_steps = 0;
static unsigned long g_grammar_fullscan_steps = 0;
static unsigned long g_grammar_chunk_steps = 0;
unsigned long pn_sampler_grammar_chunk_steps(void) { return g_grammar_chunk_steps; }
static double g_grammar_mask_seconds = 0.0;
unsigned long pn_sampler_grammar_shortlist_steps(void) { return g_grammar_shortlist_steps; }
unsigned long pn_sampler_grammar_fullscan_steps(void) { return g_grammar_fullscan_steps; }
double pn_sampler_grammar_mask_seconds(void) { return g_grammar_mask_seconds; }

static double grammar_now_seconds(void)
{
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (double)ts.tv_sec + (double)ts.tv_nsec * 1e-9;
}
unsigned long pn_sampler_grammar_sparse_rows(void)  { return g_grammar_sparse_rows; }

/* True when `logits` is still the device's sparse top-k presentation (at most
 * PN_SURV_MAX finite entries) rather than the full vocabulary row. */
static int grammar_row_is_sparse(const float *logits, int n_vocab)
{
    int finite = 0;
    for (int i = 0; i < n_vocab; i++) {
        uint32_t bits;
        memcpy(&bits, &logits[i], sizeof bits);
        if (bits == PN_NEG_INF_BITS) continue;
        if (++finite > PN_SURV_MAX) return 0;
    }
    return 1;
}

/* Record one mask application and shout if it landed on a sparse row. */
static void grammar_note_mask_step(const float *logits, int n_vocab)
{
    g_grammar_mask_steps++;
    if (!grammar_row_is_sparse(logits, n_vocab)) return;
    g_grammar_sparse_rows++;
    fprintf(stderr, "[tool_grammar] INVARIANT VIOLATED: mask applied to a "
            "SPARSE top-k row (masked_steps=%lu sparse_rows=%lu) — the "
            "full-logits row was armed LATE; the grammar can only see ~k "
            "tokens this step and may fail open\n",
            g_grammar_mask_steps, g_grammar_sparse_rows);
}

/* Greedy over S survivors already in ascending-id order.  Mirrors the full
 * path's `for (i=1..n_vocab) if (logits[i] > logits[best]) best = i;` — same
 * strict `>`, same ascending order, so the same lowest-id-wins tie-break. */
static uint32_t sparse_argmax(const int *ids, const float *vals, int S)
{
    int best = 0;
    for (int i = 1; i < S; i++)
        if (vals[i] > vals[best]) best = i;
    return (uint32_t)ids[best];
}

/* The sampler's final distribution over S survivors already in ascending-id order, with
 * temperature > 0 and the repetition penalty already applied to vals[] (scaled in place by
 * 1 / temperature): probs[i] for survivor i, zero where top-k / top-p cut it.
 * Mirrors sample_token_core's tail statement for statement. */
static void sparse_final_probs(float *vals, int S, int n_vocab,
                               const pn_sampling_params *params, float *probs) {
    float work[PN_SURV_MAX];

    float inv_temp = 1.0f / params->temperature;
    for (int i = 0; i < S; i++) vals[i] *= inv_temp;

    int top_k = params->top_k;
    if (top_k <= 0 || top_k > n_vocab) top_k = n_vocab;

    float max_logit = vals[0];
    for (int i = 1; i < S; i++) if (vals[i] > max_logit) max_logit = vals[i];

    float sum = 0.0f;
    for (int i = 0; i < S; i++) { probs[i] = expf(vals[i] - max_logit); sum += probs[i]; }
    float inv_sum = 1.0f / sum;
    for (int i = 0; i < S; i++) probs[i] *= inv_sum;

    /* Top-K over the full row: the order statistic lands in the zero region
     * (threshold 0.0f, nothing cut) whenever top_k >= S. */
    if (top_k < n_vocab) {
        float threshold = 0.0f;
        if (top_k < S) {
            memcpy(work, probs, (size_t)S * sizeof(float));
            qsort(work, (size_t)S, sizeof(float), sampler_cmp_desc);
            threshold = work[top_k - 1];
        }
        for (int i = 0; i < S; i++) if (probs[i] < threshold) probs[i] = 0.0f;
        /* llama.cpp / HF order (2026-10-07): renormalise WITHIN the top-k set before
         * top-p, so top-p measures mass among the candidates, not the full vocab. */
        sum = 0.0f;
        for (int i = 0; i < S; i++) sum += probs[i];
        if (sum > 0.0f) {
            inv_sum = 1.0f / sum;
            for (int i = 0; i < S; i++) probs[i] *= inv_sum;
        }
    }

    if (params->top_p < 1.0f) {
        int nc = 0;
        for (int i = 0; i < S; i++) if (probs[i] > 0.0f) work[nc++] = probs[i];
        qsort(work, (size_t)nc, sizeof(float), sampler_cmp_desc);
        float cum_prob = 0.0f, top_p_threshold = 0.0f;
        for (int t = 0; t < nc; t++) {
            if (!(cum_prob < params->top_p)) break;
            if (work[t] <= 0.0f) break;
            cum_prob += work[t];
            top_p_threshold = work[t];
        }
        for (int i = 0; i < S; i++) if (probs[i] < top_p_threshold) probs[i] = 0.0f;
    }

    sum = 0.0f;
    for (int i = 0; i < S; i++) sum += probs[i];
    if (sum > 0.0f) {
        inv_sum = 1.0f / sum;
        for (int i = 0; i < S; i++) probs[i] *= inv_sum;
    }
}

/* The sampled draw over S survivors (sparse_final_probs, then the cumulative scan). */
static uint32_t sparse_sample(const int *ids, float *vals, int S, int n_vocab,
                              pn_sampling_params *params) {
    float probs[PN_SURV_MAX];

    sparse_final_probs(vals, S, n_vocab, params, probs);
    float r = rand_float(params);
    float cum = 0.0f;
    for (int i = 0; i < S; i++) {
        cum += probs[i];
        if (r < cum) return (uint32_t)ids[i];
    }
    for (int i = S - 1; i >= 0; i--) {
        if (probs[i] > 0.0f) return (uint32_t)ids[i];
    }
    /* Every probability underflowed: the full path's scan falls off the end
     * and returns token 0. Match it — NOT the lowest survivor id. */
    return 0;
}

/* `PN_TT_SAMPLED_TAIL=0`: a top-N row whose request top-k reaches past the N is refilled
 * dense on every draw (the behaviour before the tail planner). */
static int sampler_tail_enabled(void)
{
    static int v = -1;
    if (v < 0) v = !(getenv("PN_TT_SAMPLED_TAIL") && atoi(getenv("PN_TT_SAMPLED_TAIL")) == 0);
    return v;
}

/*
 * Can a TOP-N row (finite on N entries, -INFINITY elsewhere) decide this SAMPLED draw
 * exactly as the dense row would?
 *
 * EXACT: the sampler's top-k set, after the penalties, lies inside the N: every token outside
 * has a raw logit <= the smallest survivor's raw logit (v_min), and presence / frequency (>= 0)
 * and repetition (>= 1) penalties only LOWER logits, so if the k-th largest PENALISED survivor
 * is above v_min by a margin that survives temperature + expf rounding, no outside token can
 * reach the top-k threshold and the renormalised top-k distribution is the dense one's.
 * TAIL: top-k is off or above the N — the tail planner (sampler_topn_tail.h) decides, with
 * *bound = v_min. REFILL: anything else (a logit bias or a penalty can RAISE a token).
 * Not sparse at all (a dense row) is EXACT: nothing to decide.
 */
pn_topn_verdict pn_sampler_topn_row_verdict(const float *logits, int n_vocab,
                                            const pn_sampling_params *params,
                                            const uint32_t *recent, int n_recent, float *bound)
{
    enum { MAX_N = 256 };
    int ids[MAX_N];
    float adj[MAX_N];
    float v_min = 0.0f, kth, limit;
    int S = 0, i, j;
    uint32_t bits;
    /* GREEDY is decided here only when client surgery can reorder across the N
     * (sampler_topn_surgery.h); a plain greedy argmax over the top-N is the dense one. */
    const bool greedy = params && params->temperature <= 0.0f;

    if (!params || (greedy && !pn_topn_surgery_present(params)))
        return PN_TOPN_EXACT;
    S = row_hint_gather(logits, n_vocab, ids, adj, MAX_N);
    if (S == -2)
        return PN_TOPN_EXACT;              /* not a top-N row */
    const bool hinted = S >= 0;
    if (hinted) {
        for (j = 0; j < S; j++)
            if (j == 0 || adj[j] < v_min) v_min = adj[j];
    } else {
        S = 0;
    }
    for (i = 0; !hinted && i < n_vocab; i++) {
        memcpy(&bits, &logits[i], sizeof(bits));
        if (bits == PN_NEG_INF_BITS)
            continue;
        if (S == MAX_N)
            return PN_TOPN_EXACT;          /* not a top-N row */
        ids[S] = i;
        adj[S] = logits[i];
        if (S == 0 || logits[i] < v_min) v_min = logits[i];
        S++;
    }
    if (S == n_vocab)
        return PN_TOPN_EXACT;
    if (greedy) {
        float best;

        if (S < 1 || !pn_topn_surgery_enabled() ||
            !pn_topn_surgery_outside_limit(ids, S, v_min, params, n_recent, &limit))
            return PN_TOPN_REFILL;
        pn_topn_surgery_apply(ids, adj, S, params, recent, n_recent);
        for (best = adj[0], j = 1; j < S; j++)
            if (adj[j] > best) best = adj[j];
        return pn_topn_surgery_clears(best, limit) ? PN_TOPN_EXACT : PN_TOPN_REFILL;
    }
    if (params->logit_bias && params->n_logit_bias > 0 &&
        (!pn_topn_surgery_enabled() || params->top_k < 1 || params->top_k > S))
        return PN_TOPN_REFILL;
    if (n_recent > 0 && (params->presence_penalty < 0.0f || params->frequency_penalty < 0.0f ||
                         params->repeat_penalty < 1.0f))
        return PN_TOPN_REFILL;
    if (params->top_k < 1 || params->top_k > S) {
        if (bound) *bound = v_min;
        return S > 0 && sampler_tail_enabled() ? PN_TOPN_TAIL : PN_TOPN_REFILL;
    }
    /* Without a bias the limit is v_min: every outside token's raw logit is <= it. */
    if (!pn_topn_surgery_outside_limit(ids, S, v_min, params, n_recent, &limit))
        return PN_TOPN_REFILL;
    pn_topn_surgery_apply(ids, adj, S, params, recent, n_recent);
    for (j = 0; j < S && params->repeat_penalty != 1.0f; j++) {
        for (i = 0; i < n_recent; i++)
            if (recent && recent[i] == (uint32_t)ids[j])
                adj[j] = adj[j] > 0.0f ? adj[j] / params->repeat_penalty
                                       : adj[j] * params->repeat_penalty;
    }
    /* k-th largest adjusted value (S <= 256: partial selection). */
    for (i = 0; i < params->top_k; i++) {
        int best = i;

        for (j = i + 1; j < S; j++)
            if (adj[j] > adj[best]) best = j;
        kth = adj[best];
        adj[best] = adj[i];
        adj[i] = kth;
    }
    kth = adj[params->top_k - 1];
    return (kth - limit) / params->temperature > 1e-4f ? PN_TOPN_EXACT : PN_TOPN_REFILL;
}

bool pn_sampler_topn_row_exact(const float *logits, int n_vocab, const pn_sampling_params *params,
                               const uint32_t *recent, int n_recent)
{
    return pn_sampler_topn_row_verdict(logits, n_vocab, params, recent, n_recent, NULL) ==
           PN_TOPN_EXACT;
}

/* Distribution-test red arm only: the pre-2026-10-07 order (top-p over full-vocab mass). */
static int g_sampler_old_order = 0;

/* The full path's sampling distribution: temperature -> full-vocab softmax -> top-k
 * (renormalised) -> top-p -> renormalised, over the row as given (bias and penalties already
 * applied). Returns the scratch probability row, or NULL. Shared by the draw and by the
 * top-N distribution test (pn_sampler_distribution). */
static float *sampler_full_probs(float *logits, int n_vocab, const pn_sampling_params *params)
{
    /* Apply temperature */
    float inv_temp = 1.0f / params->temperature;
    for (int i = 0; i < n_vocab; i++) {
        logits[i] *= inv_temp;
    }

    int top_k = params->top_k;
    if (top_k <= 0 || top_k > n_vocab) top_k = n_vocab;

    /* Full-vocab softmax (preserved exactly: same max, same normalization set) */
    float max_logit = logits[0];
    for (int i = 1; i < n_vocab; i++) {
        if (logits[i] > max_logit) max_logit = logits[i];
    }

    pn_sampler_scratch *sc = sampler_scratch_get(n_vocab);
    if (!sc) return NULL;
    float *probs = sc->probs;

    float sum = 0.0f;
    for (int i = 0; i < n_vocab; i++) {
        probs[i] = expf(logits[i] - max_logit);
        sum += probs[i];
    }
    float inv_sum = 1.0f / sum;
    for (int i = 0; i < n_vocab; i++) {
        probs[i] *= inv_sum;
    }

    /* Build the survivor set (token indices, ascending order).
     *
     * Top-K: find the top_k-th largest probability via quickselect and keep
     * everything >= it (matches the original "if probs[i] < threshold -> 0",
     * including ties at the threshold). When no top-k is requested, every token
     * is a survivor, exactly as in the original. */
    /* Top-K filtering. The original found the top_k-th largest probability with
     * an O(top_k * V) selection sort; we find that exact same value as an order
     * statistic via O(V) quickselect. The zeroing pass below is byte-identical
     * to the original ("keep probs >= threshold", ties included). */
    if (top_k < n_vocab) {
        memcpy(sc->work, probs, (size_t)n_vocab * sizeof(float));
        /* top_k-th largest == (n_vocab - top_k)-th smallest (0-indexed) */
        float threshold = sampler_select_kth(sc->work, n_vocab, n_vocab - top_k);
        for (int i = 0; i < n_vocab; i++) {
            if (probs[i] < threshold) probs[i] = 0.0f;
        }
        /* llama.cpp / HF order (2026-10-07, David): renormalise WITHIN the top-k set
         * before top-p (ascending index order — the sparse path sums the same nonzero
         * terms in the same order, and adding 0.0f is exact). The distribution test's red
         * arm alone switches it off (g_sampler_old_order). */
        sum = 0.0f;
        if (!g_sampler_old_order)
        for (int i = 0; i < n_vocab; i++) sum += probs[i];
        if (sum > 0.0f) {
            inv_sum = 1.0f / sum;
            for (int i = 0; i < n_vocab; i++) probs[i] *= inv_sum;
        }
    }

    /* Top-P filtering. The original repeatedly scanned the full vocab to pull
     * out the next-largest probability, accumulating until the cumulative mass
     * reached top_p and recording the last (smallest) probability it added as
     * the threshold. Instead we gather the (at most top_k) surviving positive
     * probabilities once and sort them descending, then accumulate in that same
     * descending order. This yields the identical cumulative sum and identical
     * top_p_threshold (ties are equal values, so their order is irrelevant); the
     * final zeroing pass is byte-identical to the original. */
    if (params->top_p < 1.0f) {
        float *cand = sc->work;
        int nc = 0;
        for (int i = 0; i < n_vocab; i++) {
            if (probs[i] > 0.0f) cand[nc++] = probs[i];
        }
        qsort(cand, (size_t)nc, sizeof(float), sampler_cmp_desc);

        float cum_prob = 0.0f;
        float top_p_threshold = 0.0f;
        for (int t = 0; t < nc; t++) {
            if (!(cum_prob < params->top_p)) break;
            if (cand[t] <= 0.0f) break;
            cum_prob += cand[t];
            top_p_threshold = cand[t];
        }
        for (int i = 0; i < n_vocab; i++) {
            if (probs[i] < top_p_threshold) probs[i] = 0.0f;
        }
    }

    /* Re-normalize (full-vocab, ascending index order: byte-identical to the
     * original so the final draw crosses at exactly the same token). */
    sum = 0.0f;
    for (int i = 0; i < n_vocab; i++) sum += probs[i];
    if (sum > 0.0f) {
        inv_sum = 1.0f / sum;
        for (int i = 0; i < n_vocab; i++) probs[i] *= inv_sum;
    }

    return probs;
}

/*
 * THE TOP-N DISTRIBUTION TEST (`PN_SAMPLER_DIST_CHECK=1`, a DENSE sampled run).
 *
 * For every sampled draw the dense row reaches the sampler with, build the top-N row the card
 * would hand over (the kernel's comparator: value desc, index asc; N = PN_SAMPLER_DIST_N,
 * default 64) and compute the production sampler's final distribution for BOTH (bias,
 * penalties, then sampler_full_probs). Report the total variation distance on draws the
 * exactness check accepts (where production would sample the top-N row) and count the
 * refills. The draw itself still uses the dense row, so the run's text is the dense run's.
 * Red arms (`PN_SAMPLER_DIST_RED`): `old_order` (the previous top-p) and `topk0` (top-k off,
 * scored on every draw) must show a large distance.
 */
static void sampler_apply_client_surgery(float *logits, int n_vocab,
                                         const pn_sampling_params *params,
                                         const uint32_t *recent_tokens, int n_recent);
static void sampler_apply_repeat_penalty(float *logits, int n_vocab,
                                         const pn_sampling_params *params,
                                         const uint32_t *recent_tokens, int n_recent);

static float sampler_dist_probs(const float *row, int n_vocab, const pn_sampling_params *params,
                                const uint32_t *recent, int n_recent, float *out)
{
    float *work = malloc((size_t)n_vocab * sizeof(float));
    float *probs;

    if (!work) return -1.0f;
    memcpy(work, row, (size_t)n_vocab * sizeof(float));
    pn_apply_logit_bias(work, n_vocab, params->logit_bias, params->n_logit_bias);
    pn_apply_presence_frequency_penalties(work, n_vocab, params->presence_penalty,
                                          params->frequency_penalty, recent, n_recent);
    if (params->repeat_penalty != 1.0f && recent)
        for (int i = 0; i < n_recent; i++)
            if (recent[i] < (uint32_t)n_vocab)
                work[recent[i]] = work[recent[i]] > 0.0f ? work[recent[i]] / params->repeat_penalty
                                                         : work[recent[i]] * params->repeat_penalty;
    probs = sampler_full_probs(work, n_vocab, params);
    if (probs) memcpy(out, probs, (size_t)n_vocab * sizeof(float));
    free(work);
    return probs ? 0.0f : -1.0f;
}

static struct {
    unsigned long draws, scored, inexact;
    double max_tvd, sum_tvd;
} g_dist;

static void sampler_dist_report(void)
{
    if (g_dist.draws)
        fprintf(stderr, "SAMPLER DIST — %lu draws, %lu scored, %lu inexact (refill), TVD max %.3e "
                        "mean %.3e\n", g_dist.draws, g_dist.scored, g_dist.inexact,
                g_dist.max_tvd, g_dist.scored ? g_dist.sum_tvd / (double)g_dist.scored : 0.0);
}

/* The tail planner's share of the test (top-k off or past the N): its plan over the top-N
 * row and the distribution production would draw from — NUCLEUS: the survivors at top_k =
 * the cut; MIXTURE: a * h on the survivors plus the residual max(p - a h, 0) — against the
 * dense one. `PN_SAMPLER_DIST_RED=tail_nobound`: a planner that ignores the tail (B = 0). */
static struct {
    unsigned long nucleus, mixture, dense;
    double max_tvd, sum_residual, sum_tail_mass, max_tail_mass;
} g_dist_tail;

static void sampler_dist_tail_report(void)
{
    if (g_dist_tail.nucleus + g_dist_tail.mixture + g_dist_tail.dense)
        fprintf(stderr, "SAMPLER DIST TAIL — nucleus %lu, mixture %lu (residual rate mean %.3e, "
                        "true tail mass mean %.3e max %.3e), dense %lu, TVD max %.3e\n",
                g_dist_tail.nucleus, g_dist_tail.mixture,
                g_dist_tail.mixture ? g_dist_tail.sum_residual / (double)g_dist_tail.mixture : 0.0,
                g_dist_tail.mixture ? g_dist_tail.sum_tail_mass / (double)g_dist_tail.mixture : 0.0,
                g_dist_tail.max_tail_mass, g_dist_tail.dense, g_dist_tail.max_tvd);
}

static void sampler_dist_tail(const float *logits, const float *sparse, int n_vocab,
                              const pn_sampling_params *p, const uint32_t *recent, int n_recent,
                              float bound, float *pd, float *ps)
{
    const char *red = getenv("PN_SAMPLER_DIST_RED");
    tnt_sampling ts = { p->temperature, p->top_p, p->top_k, n_vocab, 1.0, -1.0, 0 };
    int ids[PN_SURV_MAX], raw_ids[PN_SURV_MAX];
    float vals[PN_SURV_MAX], raw[PN_SURV_MAX];
    tnt_plan plan;
    tnt_tail_bins tb;
    uint32_t counts[TNT_TAIL_BINS + 1];
    pn_sampling_params sub = *p;
    double tvd = 0.0, head = 0.0, m, B, trivial;
    float x_min;
    int S, i, at, in_topk;

    if (red && strcmp(red, "tail_nobound") == 0) ts.tail_scale = 0.0;
    if (sparse_gather(sparse, n_vocab, raw_ids, raw) < 1) return;
    memcpy(ps, sparse, (size_t)n_vocab * sizeof(float));
    sampler_apply_client_surgery(ps, n_vocab, p, recent, n_recent);
    sampler_apply_repeat_penalty(ps, n_vocab, p, recent, n_recent);
    S = sparse_gather(ps, n_vocab, ids, vals);
    if (S < 1 || sampler_dist_probs(logits, n_vocab, p, recent, n_recent, pd) != 0.0f) return;
    /* Production's bound when the device counts (the counts emulated over the dense row);
     * a plan the trivial bound decides is the same plan under the tighter one. */
    m = vals[0];
    x_min = vals[0];
    for (i = 1; i < S; i++) {
        if (vals[i] > m) m = vals[i];
        if (vals[i] < x_min) x_min = vals[i];
    }
    tnt_tail_bins_make(bound, x_min, p->temperature, n_vocab, &tb);
    tnt_tail_bins_count(&tb, logits, n_vocab, counts);
    if (tnt_tail_from_counts(&tb, counts, raw, S, m, 1.0 / (double)p->temperature, x_min,
                             p->top_k, &B, &in_topk) == 0) {
        trivial = (double)(n_vocab - S) * exp(((double)bound - m) / (double)p->temperature);
        ts.tail_mass = B < trivial ? B : trivial;
        ts.survivors_in_topk = in_topk;
    }
    switch (tnt_plan_draw(vals, S, bound, &ts, &plan)) {
    case TNT_NUCLEUS:
        g_dist_tail.nucleus++;
        sub.top_k = plan.nucleus_k;
        sub.top_p = 1.0f;
        if (sampler_dist_probs(sparse, n_vocab, &sub, recent, n_recent, ps) != 0.0f) return;
        break;
    case TNT_MIXTURE:
        g_dist_tail.mixture++;
        for (i = 0; i < n_vocab; i++) ps[i] = pd[i];
        for (at = 0; at < S; at++) {
            const double a_h = plan.head_mass * plan.head_prob[at];
            const double r = (double)pd[ids[at]] - a_h;
            ps[ids[at]] = (float)(a_h + (r > 0.0 ? r : 0.0));
            head += (double)pd[ids[at]];
        }
        g_dist_tail.sum_residual += 1.0 - plan.head_mass;
        g_dist_tail.sum_tail_mass += 1.0 - head;
        if (1.0 - head > g_dist_tail.max_tail_mass) g_dist_tail.max_tail_mass = 1.0 - head;
        break;
    default:
        g_dist_tail.dense++;
        return;
    }
    for (i = 0; i < n_vocab; i++) tvd += fabs((double)pd[i] - (double)ps[i]);
    tvd *= 0.5;
    if (tvd > g_dist_tail.max_tvd) g_dist_tail.max_tvd = tvd;
    if (((g_dist_tail.nucleus + g_dist_tail.mixture) & 63u) == 0u)
        sampler_dist_tail_report();
}

static void sampler_dist_check(const float *logits, int n_vocab, const pn_sampling_params *in,
                               const uint32_t *recent, int n_recent)
{
    static int registered;
    const char *red = getenv("PN_SAMPLER_DIST_RED");
    const char *ns = getenv("PN_SAMPLER_DIST_N");
    const int N = ns ? atoi(ns) : 64;
    pn_sampling_params p = *in;
    float *sparse, *pd, *ps;
    int *top, n = 0, i, j;
    bool exact, score_all = false;
    double tvd = 0.0;
    float bound = 0.0f;

    if (!registered) {
        atexit(sampler_dist_report);
        atexit(sampler_dist_tail_report);
        registered = 1;
    }
    if (red && strcmp(red, "topk0") == 0) { p.top_k = 0; score_all = true; }
    /* `score_all`: score the draws the exactness check REJECTS too (run with a small
     * PN_SAMPLER_DIST_N, e.g. 8 at top_k 40, so the rejected ones carry real mass). */
    if (red && strcmp(red, "score_all") == 0) score_all = true;
    sparse = malloc((size_t)n_vocab * sizeof(float));
    pd = malloc((size_t)n_vocab * sizeof(float));
    ps = malloc((size_t)n_vocab * sizeof(float));
    top = malloc((size_t)(N > 0 ? N : 1) * sizeof(int));
    if (!sparse || !pd || !ps || !top || N < 1) goto done;
    /* top-N: value desc, index asc (the kernel's comparator). */
    for (i = 0; i < n_vocab; i++) {
        if (n < N || logits[i] > logits[top[n - 1]]) {
            int at = n < N ? n++ : N - 1;
            top[at] = i;
            for (j = at; j > 0 && logits[top[j]] > logits[top[j - 1]]; j--) {
                int t = top[j]; top[j] = top[j - 1]; top[j - 1] = t;
            }
        }
    }
    for (i = 0; i < n_vocab; i++) sparse[i] = -INFINITY;
    for (i = 0; i < n; i++) sparse[top[i]] = logits[top[i]];
    g_dist.draws++;
    if (!score_all && pn_sampler_topn_row_verdict(sparse, n_vocab, &p, recent, n_recent,
                                                  &bound) == PN_TOPN_TAIL) {
        sampler_dist_tail(logits, sparse, n_vocab, &p, recent, n_recent, bound, pd, ps);
        goto done;
    }
    exact = pn_sampler_topn_row_exact(sparse, n_vocab, &p, recent, n_recent);
    if (!exact) g_dist.inexact++;
    if (!exact && !score_all) goto done;
    if (red && strcmp(red, "old_order") == 0) g_sampler_old_order = 1;
    if (sampler_dist_probs(logits, n_vocab, &p, recent, n_recent, pd) == 0.0f &&
        sampler_dist_probs(sparse, n_vocab, &p, recent, n_recent, ps) == 0.0f) {
        for (i = 0; i < n_vocab; i++) tvd += fabs((double)pd[i] - (double)ps[i]);
        tvd *= 0.5;
        g_dist.scored++;
        g_dist.sum_tvd += tvd;
        if (tvd > g_dist.max_tvd) g_dist.max_tvd = tvd;
        if ((g_dist.scored & 63u) == 0u)
            sampler_dist_report();
    }
    g_sampler_old_order = 0;
done:
    free(sparse); free(pd); free(ps); free(top);
}

/* Core sampler (repeat_penalty → temperature → top-k, renormalised → top-p →
 * cumulative draw).  The public sample_token wraps this with the OPTIONAL
 * grammar mask; with no grammar hook the wrapper is a straight pass-through, so
 * this path is unchanged from before Phase 5. */
/* Swap a top-N row for the dense one. Returns 1 when the row was replaced. */
static int sampler_refill(float *logits, int n_vocab, pn_sampling_params *params)
{
    t_row_hint_row = NULL; /* the row turns dense: its survivor hint is void */
    g_topn_refills++;
    if (params->dense_row(params->dense_ctx, logits, n_vocab) == 0)
        return 1;
    fprintf(stderr, "sampler: dense refill failed; sampling the top-N row\n");
    return 0;
}

/* Client-requested logit surgery (bias, presence / frequency), in OpenAI's order. */
static void sampler_apply_client_surgery(float *logits, int n_vocab,
                                         const pn_sampling_params *params,
                                         const uint32_t *recent_tokens, int n_recent)
{
    pn_apply_logit_bias(logits, n_vocab, params->logit_bias, params->n_logit_bias);
    pn_apply_presence_frequency_penalties(logits, n_vocab, params->presence_penalty,
                                          params->frequency_penalty, recent_tokens, n_recent);
}

/* The repetition penalty (sampling path only), once per occurrence in the window. */
static void sampler_apply_repeat_penalty(float *logits, int n_vocab,
                                         const pn_sampling_params *params,
                                         const uint32_t *recent_tokens, int n_recent)
{
    if (params->repeat_penalty == 1.0f || !recent_tokens || n_recent <= 0)
        return;
    for (int i = 0; i < n_recent; i++) {
        uint32_t tok = recent_tokens[i];
        if (tok < (uint32_t)n_vocab) {
            if (logits[tok] > 0.0f)
                logits[tok] /= params->repeat_penalty;
            else
                logits[tok] *= params->repeat_penalty;
        }
    }
}

/* Tail-planner draws: decided from the N / mixture residual / dense outright. */
static unsigned long g_tail_from_n = 0, g_tail_residual = 0, g_tail_dense = 0;
void pn_sampler_tail_counts(unsigned long *from_n, unsigned long *residual, unsigned long *dense)
{
    if (from_n) *from_n = g_tail_from_n;
    if (residual) *residual = g_tail_residual;
    if (dense) *dense = g_tail_dense;
}

/* Per thread: speculative sampling's rows past 0 cannot refill (the card's dense row is row
 * 0's). A draw that needs the dense row returns PN_SPEC_CHOOSE_DEFER_TOKEN instead, leaving
 * any uniform it drew in params->carried_uniform for the next draw at that position. */
static _Thread_local int t_refill_forbidden = 0;

static unsigned long g_tail_counted = 0;

/* The device's tail counts for this row (pn_sampling_params.tail_counts), turned into the
 * planner's tight bound: ts->tail_mass and ts->survivors_in_topk. 0 when unavailable. */
static int sampler_tail_counts(pn_sampling_params *params, const float *raw, const float *vals,
                               int S, float bound, int n_vocab, tnt_sampling *ts)
{
    tnt_tail_bins tb;
    uint32_t counts[TNT_TAIL_BINS + 1];
    double m = vals[0], B, trivial;
    const double inv_t = 1.0 / (double)params->temperature;
    float x_min = vals[0];
    int in_topk, i;

    if (!params->tail_counts)
        return 0;
    for (i = 1; i < S; i++) {
        if (vals[i] > m) m = vals[i];
        if (vals[i] < x_min) x_min = vals[i];
    }
    tnt_tail_bins_make(bound, x_min, params->temperature, n_vocab, &tb);
    if (params->tail_counts(params->dense_ctx, t_sample_row, tb.keys, tb.bins, counts) != 0)
        return 0;
    if (tnt_tail_from_counts(&tb, counts, raw, S, m, inv_t, x_min, params->top_k, &B,
                             &in_topk) != 0)
        return 0;
    trivial = (double)(n_vocab - S) * exp(((double)bound - m) * inv_t);
    ts->tail_mass = B < trivial ? B : trivial;   /* two valid bounds: the smaller holds */
    ts->survivors_in_topk = in_topk;
    g_tail_counted++;
    return 1;
}

unsigned long pn_sampler_tail_count_calls(void) { return g_tail_counted; }

/* The draw from the survivors' own softmax with uniform `v` (top-k / top-p off). */
static uint32_t sampler_tail_head(const int *ids, float *vals, int S, int n_vocab,
                                  pn_sampling_params *params, double v)
{
    pn_sampling_params sub = *params;

    sub.top_k = 0;
    sub.top_p = 1.0f;
    sub.carried_uniform_set = 1;
    sub.carried_uniform = v < 1.0 ? (float)v : 0.99999994f;
    if (sub.carried_uniform >= 1.0f) sub.carried_uniform = 0.99999994f;
    return sparse_sample(ids, vals, S, n_vocab, &sub);
}

/* The tail planner's draw over the gathered PENALISED survivors (`raw`: the same survivors
 * before the penalties). Returns 1 with *token when the N decides; 0 when the dense row is
 * needed, with *u_drawn the mixture's uniform (residual branch) or -1 when no draw was taken.
 * The trivial bound first; the device's counts only when it cannot decide. A mixture whose
 * draw lands past the trivial head mass a asks for the counts' a' >= a and splits [a, a')
 * back onto the head — a' is a function of the row alone, so the draw stays exact. */
static int sampler_tail_draw(const int *ids, float *vals, const float *raw, int S, float bound,
                             int n_vocab, pn_sampling_params *params, tnt_plan *plan,
                             uint32_t *token, float *u_drawn)
{
    const char *red = getenv("PN_TT_RED");
    /* RED `tail_nobound`: a planner that ignores the tail (B = 0) must measurably differ. */
    tnt_sampling ts = { params->temperature, params->top_p, params->top_k, n_vocab,
                        red && strcmp(red, "tail_nobound") == 0 ? 0.0 : 1.0, -1.0, 0 };
    pn_sampling_params sub;
    double a;
    float u;
    int counted = 0;

    *u_drawn = -1.0f;
    if (tnt_plan_draw(vals, S, bound, &ts, plan) == TNT_DENSE &&
        (counted = sampler_tail_counts(params, raw, vals, S, bound, n_vocab, &ts)))
        tnt_plan_draw(vals, S, bound, &ts, plan);
    switch (plan->kind) {
    case TNT_NUCLEUS:
        sub = *params;
        sub.top_k = plan->nucleus_k;
        sub.top_p = 1.0f;
        *token = sparse_sample(ids, vals, S, n_vocab, &sub);
        params->rng_state = sub.rng_state;
        params->carried_uniform_set = sub.carried_uniform_set;
        return 1;
    case TNT_MIXTURE:
        u = rand_float(params);
        a = plan->head_mass;
        if ((double)u < a) {
            *token = sampler_tail_head(ids, vals, S, n_vocab, params, (double)u / a);
            return 1;
        }
        if (!counted && sampler_tail_counts(params, raw, vals, S, bound, n_vocab, &ts) &&
            tnt_plan_draw(vals, S, bound, &ts, plan) == TNT_MIXTURE && plan->head_mass > a &&
            (double)u < plan->head_mass) {
            *token = sampler_tail_head(ids, vals, S, n_vocab, params,
                                       ((double)u - a) / (plan->head_mass - a));
            return 1;
        }
        if (plan->kind != TNT_MIXTURE || plan->head_mass < a)
            tnt_plan_draw(vals, S, bound, &(tnt_sampling){ ts.temperature, ts.top_p, ts.top_k,
                                                          n_vocab, ts.tail_scale, -1.0, 0 }, plan);
        *u_drawn = u;
        return 0;
    default:
        return 0;
    }
}

static uint32_t sample_token_core(float *logits, int n_vocab, pn_sampling_params *params,
                                  uint32_t *recent_tokens, int n_recent) {
    float tail_bound = 0.0f;
    int tail = 0;
    int tail_raw_ids[PN_SURV_MAX];
    float tail_raw[PN_SURV_MAX];

    if (params->temperature > 0.0f && getenv("PN_SAMPLER_DIST_CHECK"))
        sampler_dist_check(logits, n_vocab, params, recent_tokens, n_recent);
    uint64_t ht = pn_ht_now();
    /* A top-N row that cannot decide this draw exactly is swapped for the dense row —
     * never one the grammar mask already rewrote (that would drop the mask). A request
     * whose top-k reaches past the N goes to the tail planner below. */
    if (params->dense_row && !t_grammar_row_final && !short_row_live(logits) &&
        (params->temperature > 0.0f || pn_topn_surgery_present(params))) {
        const pn_topn_verdict v = pn_sampler_topn_row_verdict(logits, n_vocab, params,
                                                              recent_tokens, n_recent,
                                                              &tail_bound);
        if (v == PN_TOPN_TAIL) {
            /* The survivors' RAW values, for the device's tail counts (sampler_tail_counts). */
            tail = sparse_gather(logits, n_vocab, tail_raw_ids, tail_raw) > 0;
        } else if (v == PN_TOPN_REFILL) {
            if (t_refill_forbidden)
                return PN_SPEC_CHOOSE_DEFER_TOKEN;
            sampler_refill(logits, n_vocab, params);
        }
    }
    /* Client-requested logit surgery, in OpenAI's order and BEFORE the greedy
     * branch: both are explicit client intent (default 0 / empty = no-op), and
     * both are deterministic functions of the prefix, so greedy stays
     * reproducible.  See the ordering note on
     * apply_presence_frequency_penalties for why `repeat_penalty` is NOT here. */
    ht = pn_ht_lap(PN_HT_S_TOPN, ht);
    sampler_apply_client_surgery(logits, n_vocab, params, recent_tokens, n_recent);
    ht = pn_ht_lap(PN_HT_S_SURGERY, ht);

    /* Temperature == 0 means greedy: PURE argmax on the RAW logits, taken BEFORE
     * any repetition penalty. Greedy decoding must return the model's top token
     * exactly as the reference (llama.cpp --temp 0 applies no repeat penalty by
     * default). Applying repeat_penalty first (the previous behaviour) could
     * demote the true argmax below a near-tied runner-up whenever the top token
     * had appeared in the last repeat_last_n tokens — for Bonsai-27B on the
     * x²+5 prompt this flipped ':' (top, logit 25.41) to ' given' (25.08, Δ0.33)
     * right after "Equation", steering the model into a LaTeX \frac attractor
     * and losing the answer 30. The forward pass was faithful; the greedy
     * decision was corrupted by the penalty. Sampling (temp>0) is unchanged. */
    if (params->temperature <= 0.0f) {
        /* Same sparse fast path as the sampler below, same flag, same gather.
         * Neither pn_apply_logit_bias (+=) nor the presence/frequency penalties
         * (-=) can lift a -inf non-survivor back to finite, so the survivor set
         * is the same set the full scan would have had to beat. */
        if ((sparse_sampler_enabled() || t_row_hint_row == logits) && !g_grammar_masked) {
            int   surv_ids[PN_SURV_MAX];
            float surv_vals[PN_SURV_MAX];
            int S = sparse_gather(logits, n_vocab, surv_ids, surv_vals);
            if (S > 0) {
                g_sparse_hits++;
                uint32_t tok = sparse_argmax(surv_ids, surv_vals, S);
                pn_ht_lap(PN_HT_S_GREEDY, ht);
                return tok;
            }
            /* S == 0 (nothing finite) or S == -1 (dense row): full scan. */
        }
        int best = 0;
        for (int i = 1; i < n_vocab; i++) {
            if (logits[i] > logits[best]) best = i;
        }
        /* Short row (SHORT ROW, WHY (4)): R's argmax is the listed best only above bound. */
        if (short_row_live(logits) && !(logits[best] > t_short_bound) &&
            !short_row_red("greedy") && short_row_refill(logits, n_vocab, PN_SR_GREEDY)) {
            sampler_apply_client_surgery(logits, n_vocab, params, recent_tokens, n_recent);
            best = 0;
            for (int i = 1; i < n_vocab; i++)
                if (logits[i] > logits[best]) best = i;
        }
        pn_ht_lap(PN_HT_S_GREEDY, ht);
        return (uint32_t)best;
    }

    /* Apply repetition penalty (sampling path only) */
    sampler_apply_repeat_penalty(logits, n_vocab, params, recent_tokens, n_recent);
    ht = pn_ht_lap(PN_HT_S_SURGERY, ht);

    /* Sparse-survivor fast path. Placed HERE, after the repetition penalty and
     * before the temperature scale, because the penalty writes through token
     * ids that may not be survivors — and on a non-survivor it is a no-op
     * either way (-inf/rp and -inf*rp are both -inf), so the gathered survivor
     * logits are already penalised. See the block comment above sparse_gather
     * for the bit-exactness argument and for why the row test is a bit
     * compare and not `> -INFINITY`. */
    /* Also whenever a backend hands top-N rows (dense_row set): byte-identical to the full
     * path by test_sparse_sampler_identity, ~10 ms a draw cheaper at 248k vocab. */
    if ((sparse_sampler_enabled() || params->dense_row) && !g_grammar_masked &&
        !short_row_live(logits)) {
        int   surv_ids[PN_SURV_MAX];
        float surv_vals[PN_SURV_MAX];
        int S = sparse_gather(logits, n_vocab, surv_ids, surv_vals);
        if (S > 0 && tail) {
            tnt_plan plan;
            uint32_t tok = 0;
            float u;

            if (sampler_tail_draw(surv_ids, surv_vals, tail_raw, S, tail_bound, n_vocab, params,
                                  &plan, &tok, &u)) {
                g_tail_from_n++;
                g_sparse_hits++;
                return tok;
            }
            if (t_refill_forbidden) {
                /* RED `tail_nocarry`: the deferred row drops its draw (spec != plain). */
                if (u >= 0.0f && !(getenv("PN_TT_RED") &&
                                   strcmp(getenv("PN_TT_RED"), "tail_nocarry") == 0)) {
                    params->carried_uniform_set = 1;
                    params->carried_uniform = u;
                }
                return PN_SPEC_CHOOSE_DEFER_TOKEN;
            }
            if (sampler_refill(logits, n_vocab, params)) {
                sampler_apply_client_surgery(logits, n_vocab, params, recent_tokens, n_recent);
                sampler_apply_repeat_penalty(logits, n_vocab, params, recent_tokens, n_recent);
            }
            if (u >= 0.0f) {
                float *probs = sampler_full_probs(logits, n_vocab, params);

                g_tail_residual++;
                return probs ? tnt_residual_draw(probs, n_vocab, surv_ids, &plan, S, u) : 0u;
            }
            g_tail_dense++;
        } else if (S > 0) {
            g_sparse_hits++;
            uint32_t tok = sparse_sample(surv_ids, surv_vals, S, n_vocab, params);
            pn_ht_lap(PN_HT_S_SPARSE, ht);
            return tok;
        }
        ht = pn_ht_lap(PN_HT_S_SPARSE, ht);
        /* S == 0 (nothing finite), S == -1 (dense row) or a tail plan that needed the dense
         * row: full path below. */
    }

    /* The dense path below, decided from the row's top candidates when every boundary is clear
     * of rounding (sampler_margin_draw.h): the SAME uniform, so a refused draw runs the dense
     * path with it — the same token for the same u, and the same RNG stream. The uniform is
     * drawn first; nothing between consumes randomness (no scratch: the original returns
     * before drawing, so this does too). */
    /* Short row (SHORT ROW, WHY (5)): the margin draw reads R's top 64, which is the short row's
     * when its 64th is above bound; otherwise, or when the margin draw refuses, fetch R and
     * redo the surgery on it (the same uniform follows). */
    if (short_row_live(logits) && !short_row_top64_clear(logits, n_vocab) &&
        short_row_refill(logits, n_vocab, PN_SR_TOP64)) {
        sampler_apply_client_surgery(logits, n_vocab, params, recent_tokens, n_recent);
        sampler_apply_repeat_penalty(logits, n_vocab, params, recent_tokens, n_recent);
    }
    pn_ht_note_row(0, 1);
    if (!sampler_scratch_get(n_vocab)) return 0;
    float r = rand_float(params);
    if (pn_margin_draw_enabled() && !g_sampler_old_order) {
        uint32_t tok;
        const int decided = pn_margin_draw(logits, n_vocab, params->temperature, params->top_k,
                                           params->top_p, r, &tok);
        ht = pn_ht_lap(PN_HT_S_MARGIN, ht);
        if (decided) {
            g_margin_draws++;
            return tok;
        }
    }
    if (short_row_refill(logits, n_vocab, PN_SR_MARGIN)) {
        sampler_apply_client_surgery(logits, n_vocab, params, recent_tokens, n_recent);
        sampler_apply_repeat_penalty(logits, n_vocab, params, recent_tokens, n_recent);
    }
    float *probs = sampler_full_probs(logits, n_vocab, params);
    ht = pn_ht_lap(PN_HT_S_FULL_PROBS, ht);
    if (!probs) return 0;

    /* Sample (identical cumulative scan and fallback as the original). */
    float cum = 0.0f;
    for (int i = 0; i < n_vocab; i++) {
        cum += probs[i];
        if (r < cum) { pn_ht_lap(PN_HT_S_FULL_DRAW, ht); return (uint32_t)i; }
    }
    for (int i = n_vocab - 1; i >= 0; i--) {
        if (probs[i] > 0.0f) { pn_ht_lap(PN_HT_S_FULL_DRAW, ht); return (uint32_t)i; }
    }
    pn_ht_lap(PN_HT_S_FULL_DRAW, ht);
    return 0;
}

int pn_sampler_row_final_probs(float *row, int n_vocab, pn_sampling_params *params,
                               const uint32_t *recent, int n_recent, bool require_exact,
                               int *ids, float *probs, int cap)
{
    int   surv_ids[PN_SURV_MAX];
    float surv_vals[PN_SURV_MAX];
    int S;

    if (!params || params->temperature <= 0.0f || params->tool_grammar || !row || cap < 1)
        return -1;
    /* Decide everything that can refuse BEFORE the row is penalised: a refusal hands the
     * caller its row back as given. */
    S = sparse_gather(row, n_vocab, surv_ids, surv_vals);
    if (S < 1 || S > cap ||
        (require_exact && !pn_sampler_topn_row_exact(row, n_vocab, params, recent, n_recent)))
        return -1;
    /* sample_token_core's order: surgery, repetition penalty, gather, the sparse tail. */
    sampler_apply_client_surgery(row, n_vocab, params, recent, n_recent);
    sampler_apply_repeat_penalty(row, n_vocab, params, recent, n_recent);
    S = sparse_gather(row, n_vocab, ids, surv_vals);
    sparse_final_probs(surv_vals, S, n_vocab, params, probs);
    return S;
}

float pn_sampler_uniform(pn_sampling_params *params) { return rand_float(params); }

static int sampler_pair_id_cmp(const void *a, const void *b)
{
    const int x = *(const int *)a, y = *(const int *)b;
    return (x > y) - (x < y);
}

int pn_sampler_pairs_final_probs(const uint32_t *index, const float *value, uint32_t count,
                                 int n_vocab, const pn_sampling_params *params,
                                 const uint32_t *recent, int n_recent, int *ids, float *probs,
                                 int cap)
{
    int order[PN_SURV_MAX * 2];
    float vals[PN_SURV_MAX];
    int S = 0, i, j;

    if (!params || params->temperature <= 0.0f || cap < 1 || count > PN_SURV_MAX)
        return -1;
    /* (id, pair) packed for an ascending-id sort: the sampler's survivor order. */
    for (i = 0; i < (int)count; i++) {
        if (index[i] >= (uint32_t)n_vocab)
            continue;
        order[2 * S] = (int)index[i];
        order[2 * S + 1] = i;
        S++;
    }
    if (S < 1 || S > cap)
        return -1;
    qsort(order, (size_t)S, 2 * sizeof(int), sampler_pair_id_cmp);
    for (i = 0; i < S; i++) {
        float v = value[order[2 * i + 1]];
        const uint32_t tok = (uint32_t)order[2 * i];

        ids[i] = (int)tok;
        /* sampler_apply_client_surgery + sampler_apply_repeat_penalty on this token alone,
         * occurrence by occurrence in window order (no logit bias: rule requests carry none). */
        for (j = 0; j < n_recent; j++) {
            int k, seen = 0;

            if (recent[j] != tok) continue;
            if (params->presence_penalty != 0.0f || params->frequency_penalty != 0.0f) {
                for (k = 0; k < j; k++) if (recent[k] == tok) { seen = 1; break; }
                v -= params->frequency_penalty;
                if (!seen) v -= params->presence_penalty;
            }
        }
        if (params->repeat_penalty != 1.0f)
            for (j = 0; j < n_recent; j++)
                if (recent[j] == tok) v = v > 0.0f ? v / params->repeat_penalty
                                                   : v * params->repeat_penalty;
        vals[i] = v;
    }
    sparse_final_probs(vals, S, n_vocab, params, probs);
    return S;
}

/*
 * Grammar-constrained wrapper around sample_token_core (n2 tool-calling Phase 5).
 *
 * OFF by default: when params->tool_grammar is NULL (the default) this is a
 * straight pass-through to sample_token_core — logits are untouched and the
 * sampled token is byte-identical to the pre-Phase-5 path.
 *
 * When a hook is present AND active, before sampling we MASK (logit set to a
 * large-negative sentinel) every candidate token whose text would take the
 * grammar off a valid tool-call prefix; after the draw we ADVANCE the grammar
 * by the chosen token's text.  The mask is a CPU-side logit mutation exactly
 * like repeat_penalty, so it is device-agnostic and CPU-testable.  A finite
 * sentinel (not -INFINITY) is used so the core softmax's expf() underflows the
 * masked token to ~0 reliably under -ffast-math (which assumes no infinities).
 *
 * Perf: the scan is O(n_vocab) token-text lookups + allowed() checks; but each
 * allowed() short-circuits on the first char in a fixed-skeleton state, so the
 * per-step cost is dominated by the n_vocab text lookups (a few hundred µs at
 * 248k tokens) and is only paid while `active` (inside a tool call), which is a
 * small fraction of a generation.  A per-state allowed-token cache (see
 * tool_grammar.c) would drop this to a single precomputed bitmask lookup.
 */
static uint32_t sample_token_draw(float *logits, int n_vocab, pn_sampling_params *params,
                                  uint32_t *recent_tokens, int n_recent);

/* Is the hook's mask live for THIS draw?  The same read sample_token_draw makes:
 * the live walk when the hook can refresh itself, else the caller's `active`. */
static int grammar_hook_live(const pn_tool_grammar_hook *tg)
{
    if (!tg) return 0;
    if (tg->refresh && tg->st) return tg->refresh(tg->st);
    return tg->active;
}

/* One draw; a survivor hint (pn_sampler_row_hint) lives for exactly this draw.
 *
 * NO REPETITION PENALTY UNDER A LIVE MASK.  The penalty divides a token's logit
 * by repeat_penalty once PER OCCURRENCE in the last repeat_last_n tokens.  A tool
 * call is structure that repeats by design — "\n" and ">" close every line of
 * <tool_call>/<function=…>/<parameter=…> — so inside a call the penalty
 * compounds on exactly the tokens the format needs.  Recorded on Qwen3.6
 * (2026-10-28, logprobs replay): after "<parameter=path>" the model's own top
 * token was "\n" (0.00) and the draw was "\n\n" (-14.27); from that blank line it
 * wrote "</parameter", the empty-value floor masked ">", and the call shipped
 * `{"path":"</parameter_unnamed>.teamide/…"}`.  The same penalty collapsed the
 * newlines of an `edit`'s old_string (code repeats "\n" too).  The grammar
 * already owns structure while its mask is live, so the penalty stands down for
 * those draws only; prose and reasoning are sampled exactly as before. */
static uint32_t sample_token(float *logits, int n_vocab, pn_sampling_params *params,
                             uint32_t *recent_tokens, int n_recent) {
    uint32_t token;
    float repeat_penalty = 1.0f;
    int exempt = 0;

    if (params && params->tool_grammar) {
        t_row_hint_row = NULL; /* a grammar may rewrite the row: scan it */
        exempt = params->repeat_penalty != 1.0f && grammar_hook_live(params->tool_grammar);
        if (exempt) {
            repeat_penalty = params->repeat_penalty;
            params->repeat_penalty = 1.0f;
        }
    }
    const int short_row = short_row_live(logits);
    t_short_refilled = 0;
    token = sample_token_draw(logits, n_vocab, params, recent_tokens, n_recent);
    if (exempt) params->repeat_penalty = repeat_penalty;
    t_row_hint_row = NULL;
    if (short_row) {
        __atomic_fetch_add(&g_short_stats.draws, 1ul, __ATOMIC_RELAXED);
        if (!t_short_refilled)
            __atomic_fetch_add(&g_short_stats.decided, 1ul, __ATOMIC_RELAXED);
    }
    t_short_row = NULL;
    return token;
}

/* pn_tool_grammar_hook.steer: hand the grammar the model's own top token, read
 * off the row before any mask touches it, when the grammar refuses it.  The
 * argmax, not a draw: it spends no randomness, so the draw that follows is the
 * one the step would have made over the steered mask. */
static void grammar_steer_top_token(const float *logits, int n_vocab,
                                    const pn_tool_grammar_hook *tg, float mask)
{
    int nv = tg->n_vocab > 0 && tg->n_vocab < n_vocab ? tg->n_vocab : n_vocab;
    int best = -1;
    for (int i = 0; i < nv; i++)
        if (logits[i] > mask && (best < 0 || logits[i] > logits[best]))
            best = i;
    if (best < 0)
        return;
    const char *txt = tg->token_text(tg->user, (uint32_t)best);
    if (txt && *txt && !tg->allowed(tg->st, txt))
        tg->steer(tg->st, txt);
}

static uint32_t sample_token_draw(float *logits, int n_vocab, pn_sampling_params *params,
                                  uint32_t *recent_tokens, int n_recent) {
    const pn_tool_grammar_hook *tg = params ? params->tool_grammar : NULL;

    /* g_grammar_masked describes THIS step only.  Clearing it here is what
     * makes it a per-step flag: it used to be set-once-never-cleared, so on the
     * batched path (one driver thread samples every slot) the first masked step
     * of the first tool call disabled the sparse-sampler fast path for every
     * slot and every session until the process exited. */
    g_grammar_masked = 0;
    t_grammar_row_final = 0;
    if (short_row_live(logits) && params &&
        (short_row_entry_needs_full(params) ||
         short_row_past_grammar_vocab(logits, n_vocab, params->tool_grammar)))
        short_row_refill(logits, n_vocab, PN_SR_ENTRY);

    /* Snapshot the untouched logits FIRST when the request asked for logprobs —
     * everything below (mask, bias, penalties, temperature, top-k/top-p) mutates
     * `logits` in place, and the OpenAI field means the model's own
     * distribution, not the post-filter one. */
    const pn_logprob_sink *lp_sink =
        (params && params->logprobs && params->logprobs->emit)
            ? params->logprobs : NULL;
    const float *lp_raw = lp_sink ? logprob_snapshot(logits, n_vocab) : NULL;

    /* Self-refresh: when the hook exposes a refresh callback, recompute the
     * mask-active flag from the live grammar state here (so the batched driver,
     * which samples per-slot in its own thread and never touches `active`, gates
     * correctly).  Falls back to the caller-maintained `active` when NULL. */
    int tg_active = grammar_hook_live(tg);

    static const float TG_MASK = -1e30f;   /* finite sentinel (fast-math safe) */
    uint64_t ht_full = 0;
    if (tg && tg_active && tg->st && tg->token_text && tg->allowed) {
        /* A top-N row (a backend handed `dense_row`) holds only N candidates.
         * First try to decide the masked draw from those N alone
         * (grammar_mask_shortlist.h: exact when enough of them are allowed);
         * else fetch only the dense chunks holding allowed tokens
         * (dense_chunks); else refill the whole dense row, try its top
         * GMS_MAX, and run the full-vocabulary mask below.  Scoped to top-N
         * backends: a dense-row backend goes straight to the full mask, whose
         * verdicts are the same — only faster (prefix filter, one allowed()
         * per token). */
        const double mask_started = grammar_now_seconds();
        if (tg->steer)
            grammar_steer_top_token(logits, n_vocab, tg, TG_MASK);
        if (params->dense_row) {
            if (grammar_mask_from_shortlist(logits, n_vocab, params, tg, recent_tokens,
                                            n_recent, 1)) {
                g_grammar_mask_steps++;
                g_grammar_shortlist_steps++;
                g_grammar_mask_seconds += grammar_now_seconds() - mask_started;
                t_grammar_row_final = 1;
                goto masked;
            }
            if (!short_row_live(logits) &&
                grammar_mask_from_chunks(logits, n_vocab, params, tg, TG_MASK)) {
                g_grammar_mask_steps++;
                g_grammar_chunk_steps++;
                g_grammar_mask_seconds += grammar_now_seconds() - mask_started;
                g_grammar_masked = 1;   /* a masked dense-equivalent row: full path */
                t_grammar_row_final = 1;
                goto masked;
            }
            if (!short_row_refill(logits, n_vocab, PN_SR_MASK)) {
                g_topn_refills++;
                if (params->dense_row(params->dense_ctx, logits, n_vocab) != 0)
                    fprintf(stderr, "sampler: dense refill for the grammar mask failed\n");
            }
            t_grammar_row_final = 1;
            if (grammar_mask_from_shortlist(logits, n_vocab, params, tg, recent_tokens,
                                            n_recent, 1)) {
                g_grammar_mask_steps++;
                g_grammar_shortlist_steps++;
                g_grammar_mask_seconds += grammar_now_seconds() - mask_started;
                goto masked;
            }
        }
        short_row_refill(logits, n_vocab, PN_SR_MASK);
        g_grammar_fullscan_steps++;
        ht_full = pn_ht_now();
        int nv = tg->n_vocab > 0 && tg->n_vocab < n_vocab ? tg->n_vocab : n_vocab;
        /* Two-pass so the mask is only APPLIED when at least one token survives.
         * Pass 1 counts survivors without mutating logits; if none survive (a
         * name/key with no single whole-token continuation, or a tokenizer/
         * grammar mismatch) we DO NOT mask this step — masking all candidates
         * would leave a degenerate all-(-1e30) distribution the sampler draws
         * garbage from (the observed "prose slips through after </think>" bug
         * was this fail-open picking an arbitrary token).  Fail-OPEN cleanly:
         * emit the model's own top token unmasked and let the grammar re-sync on
         * the next step.  Pass 2 applies the mask only when survivors>0. */
        /* The mask is about to be evaluated against this row, so this is the
         * one place that can see both facts at once: check the density
         * invariant here, BEFORE pass 1 mutates anything. */
        grammar_note_mask_step(logits, nv);
        /* One allowed() verdict per token, kept for pass 2, behind a 1- and
         * 2-byte prefix filter (grammar_prefix_verdicts): at a structural
         * position nearly every token dies on its first or second byte. */
        static _Thread_local grammar_prefix_verdicts pv;
        unsigned char *verdict = grammar_verdict_scratch(nv);
        grammar_prefix_verdicts_init(tg, &pv);
        int survivors = 0;
        for (int i = 0; i < nv; i++) {
            if (logits[i] <= TG_MASK) continue;   /* already excluded upstream */
            const char *txt = tg->token_text(tg->user, (uint32_t)i);
            const int ok = grammar_token_verdict(tg, &pv, txt);
            if (verdict) verdict[i] = (unsigned char)ok;
            survivors += ok;
        }
        if (survivors > 0 && verdict && !params->dense_row && pn_grammar_row_dump_enabled())
            pn_grammar_row_dump(logits, nv, verdict, params, recent_tokens, n_recent);
        if (survivors > 0) {
            for (int i = 0; i < nv; i++) {
                if (logits[i] <= TG_MASK) continue;
                const int ok = verdict ? verdict[i]
                    : grammar_token_verdict(tg, &pv, tg->token_text(tg->user, (uint32_t)i));
                if (!ok)
                    logits[i] = TG_MASK;
            }
            g_grammar_masked = 1;   /* sparse fast path stands down this step */
            g_grammar_mask_seconds += grammar_now_seconds() - mask_started;
        } else if (tg->fail_closed) {
            /* Fail-CLOSED (forced-mode hardening): no legal token survives, so
             * rather than fall open to the model's own out-of-grammar top token
             * (the escape hatch that let garbage/empty calls through), force the
             * caller-supplied EOS/end token — a clean, in-grammar termination that
             * never emits garbage.  Advance the grammar by it (so a driver-side
             * hook that tracks completion sees the end) and return immediately;
             * skip normal sampling entirely. */
            fprintf(stderr, "[tool_grammar] 0 survivors — fail-CLOSED (forcing EOS %u)\n",
                    tg->forced_token);
            if (tg->advance && tg->st) {
                const char *etxt = tg->token_text(tg->user, tg->forced_token);
                if (etxt) tg->advance(tg->st, etxt);
            }
            if (lp_raw) emit_logprobs(lp_sink, lp_raw, n_vocab, tg->forced_token);
            return tg->forced_token;
        } else {
            /* Diagnostic detail: a 0-survivor step means the grammar rejected
             * EVERY candidate, which for a whole-answer constraint is normally
             * impossible (whitespace and '{' are always in a BPE vocab).  Report
             * the scan bounds and how many entries had no text, so the cause is
             * visible from the fleet log instead of needing a repro build. */
            int n_empty = 0, n_skipped = 0, n_rejected = 0;
            const char *sample[3] = { NULL, NULL, NULL };
            int n_sample = 0;
            for (int i = 0; i < nv; i++) {
                const char *txt = tg->token_text(tg->user, (uint32_t)i);
                if (!(txt && *txt)) { n_empty++; continue; }
                if (logits[i] <= TG_MASK) { n_skipped++; continue; }
                n_rejected++;
                if (n_sample < 3) sample[n_sample++] = txt;
            }
            fprintf(stderr, "[tool_grammar] WARN: 0 survivors this step — "
                    "fail-open (mask skipped) [nv=%d n_vocab=%d empty=%d "
                    "logit_skipped=%d rejected=%d s0='%s' s1='%s' s2='%s']\n",
                    nv, n_vocab, n_empty, n_skipped, n_rejected,
                    sample[0] ? sample[0] : "", sample[1] ? sample[1] : "",
                    sample[2] ? sample[2] : "");
        }
    }

masked:;
    if (ht_full)
        pn_ht_lap(PN_HT_S_MASK_FULL, ht_full);
    pn_ht_note_row(tg && tg_active, 0);
    uint32_t tok = sample_token_core(logits, n_vocab, params, recent_tokens, n_recent);
    t_grammar_row_final = 0;
    uint64_t ht_adv = pn_ht_now();

    /* Advance the grammar by the chosen token's text.  This must run EVERY step
     * (not only when the mask was active) so the FSM tracks the FREE reasoning
     * region — it needs to see "</think>" and the opener to engage the forced
     * skeleton.  The single-user serve loop advances externally (advance == NULL
     * there); the batched driver relies on this in-thread advance. */
    if (tg && tg->st && tg->token_text && tg->advance) {
        const char *txt = tg->token_text(tg->user, tok);
        if (txt) tg->advance(tg->st, txt);
    }
    pn_ht_lap(PN_HT_S_ADVANCE, ht_adv);
    if (lp_raw) emit_logprobs(lp_sink, lp_raw, n_vocab, tok);
    return tok;
}

int pn_sampler_grammar_shortlist(float *logits, int n_vocab, const pn_sampling_params *params,
                                 const uint32_t *recent_tokens, int n_recent)
{
    const pn_tool_grammar_hook *tg = params ? params->tool_grammar : NULL;

    if (!tg || !tg->st || !tg->token_text || !tg->allowed)
        return 0;
    return grammar_mask_from_shortlist(logits, n_vocab, params, tg, recent_tokens, n_recent, 0);
}

uint32_t pn_sample_decided_row(float *logits, int n_vocab, pn_sampling_params *params,
                               uint32_t *recent_tokens, int n_recent)
{
    uint32_t token;

    t_row_hint_row = NULL;
    t_grammar_row_final = 1;   /* decided: no top-N verdict, no refill, no tail plan */
    token = sample_token_core(logits, n_vocab, params, recent_tokens, n_recent);
    t_grammar_row_final = 0;
    return token;
}

/* Public seam: reuse the internal sampler from external decode drivers (serve
 * layer) without duplicating the sampling logic. See engine.h for contract. */
uint32_t pn_sample_next_token(float *logits, int n_vocab,
                              pn_sampling_params *params,
                              uint32_t *recent_tokens, int n_recent) {
    return sample_token(logits, n_vocab, params, recent_tokens, n_recent);
}

uint32_t pn_sample_next_token_topn(float *logits, int n_vocab, pn_sampling_params *params,
                                   uint32_t *recent_tokens, int n_recent) {
    uint32_t token;

    t_sparse_sampler_force = 1;
    token = sample_token(logits, n_vocab, params, recent_tokens, n_recent);
    t_sparse_sampler_force = 0;
    return token;
}

/* ============================================================
 * Async forward-step seam (C2) — see engine.h. Dispatches to the qwen35moe
 * CUDA async step when that forward is selected (the same discriminator
 * pn_admit_gpu_resident uses); everything else is "unsupported" and the
 * caller keeps the blocking model->forward path.
 * ============================================================ */

/* True for the forwards backed by the shared Qwen3.5 CUDA path (pn_qwen35moe_model
 * weights + qwen35moe kernels). pn_bonsai_forward_cuda (Bonsai/Ornith, dense) just
 * delegates to pn_qwen35moe_forward_cuda, so its weights ARE a pn_qwen35moe_model —
 * it can use the same non-blocking begin/poll. Gating on the exact MoE pointer
 * alone left the dense family stuck on the blocking path (Ornith cap). */
#ifdef USE_CUDA
static int uses_qwen35moe_cuda(const pn_model *model) {
    return model && (model->forward == pn_qwen35moe_forward_cuda ||
                     model->forward == pn_bonsai_forward_cuda);
}
#endif

int pn_forward_step_begin(pn_model *model, uint32_t token, pn_state *state) {
#ifdef USE_CUDA
    if (state && uses_qwen35moe_cuda(model))
        return pn_qwen35moe_forward_step_begin(model, token, state);
#endif
    (void)model; (void)token; (void)state;
    return 1;
}

int pn_forward_step_poll(pn_model *model, pn_state *state,
                         tm_tensor **logits_out) {
#ifdef USE_CUDA
    if (state && uses_qwen35moe_cuda(model))
        return pn_qwen35moe_forward_step_poll(model, state, logits_out);
#endif
    (void)model; (void)state; (void)logits_out;
    return -1;
}

/* ============================================================
 * Generation
 * ============================================================ */

char* pn_generate(pn_model *model, pn_state *state,
                  const char *prompt, int max_tokens,
                  pn_sampling_params *params) {
    if (!model || !state || !prompt || !params) return NULL;
    if (!model->forward) return NULL;
#ifdef USE_TT
    if (pn_qwen35moe_tt_select_rows(model, params, false) > 0u) {
        params->dense_row = pn_qwen35moe_tt_dense_row;
        params->tail_counts = pn_qwen35moe_tt_tail_counts;
        params->dense_ctx = model;
    }
#endif

    /* RWKV-X path: use internal trie tokenizer */
    if (model->config.arch == PN_ARCH_RWKVX) {
        uint32_t prompt_buf[8192];
        int n_prompt = pn_rwkvx_encode(model->weights, prompt, prompt_buf, 8192);
        if (n_prompt == 0) return NULL;

        fprintf(stderr, "generate: prompt tokenized to %d tokens\n", n_prompt);

        int max_recent = params->repeat_last_n;
        uint32_t *recent = calloc(max_recent, sizeof(uint32_t));
        int n_recent = 0;

        /* Prefill */
        tm_tensor *logits = NULL;
        for (int i = 0; i < n_prompt; i++) {
            if (logits) tm_free(logits);
            logits = model->forward(model, prompt_buf[i], state);
            if (!logits) { free(recent); return NULL; }

            if (n_recent < max_recent)
                recent[n_recent++] = prompt_buf[i];
            else {
                memmove(recent, recent + 1, (max_recent - 1) * sizeof(uint32_t));
                recent[max_recent - 1] = prompt_buf[i];
            }
        }

        /* Decode loop */
        int *output_tokens = malloc(max_tokens * sizeof(int));
        int n_output = 0;
        int eos_id = pn_rwkvx_eos_id(model->weights);

        for (int i = 0; i < max_tokens; i++) {
            if (!logits) break;
            float *logit_data = (float *)logits->data;
            uint32_t next_token = sample_token(logit_data, model->config.n_vocab,
                                               params, recent, n_recent);
            tm_free(logits);
            logits = NULL;

            if ((int)next_token == eos_id) break;

            output_tokens[n_output++] = (int)next_token;

            if (n_recent < max_recent)
                recent[n_recent++] = next_token;
            else {
                memmove(recent, recent + 1, (max_recent - 1) * sizeof(uint32_t));
                recent[max_recent - 1] = next_token;
            }

            logits = model->forward(model, next_token, state);
        }

        if (logits) tm_free(logits);
        free(recent);

        /* Decode all output tokens at once */
        char *output = pn_rwkvx_decode(model->weights, (uint32_t*)output_tokens, n_output);
        free(output_tokens);
        return output;
    }

    /* GGUF path: use ft_tokenizer */
    if (!model->tokenizer) return NULL;

    int is_bpe = (ft_get_model_type(model->tokenizer) == 0);

    /* BPE: qwen35 pre-tokenization + GPT-2 byte BPE (matches llama.cpp).
     * SentencePiece: direct encode with BOS. */
    int n_prompt_tokens = 0;
    int *prompt_tokens;
    if (is_bpe) {
        prompt_tokens = pn_bpe_encode(model->tokenizer, prompt, &n_prompt_tokens);
    } else {
        prompt_tokens = ft_encode_with_special(model->tokenizer, prompt,
                                               &n_prompt_tokens,
                                               true,    /* add BOS for SentencePiece */
                                               false);  /* no EOS */
    }
    if (!prompt_tokens || n_prompt_tokens == 0) {
        fprintf(stderr, "generate: tokenization failed\n");
        free(prompt_tokens);
        return NULL;
    }

    fprintf(stderr, "generate: prompt tokenized to %d tokens\n", n_prompt_tokens);

    /* The exact ids, for a harness that takes token ids rather than text. */
    if (getenv("PN_PROMPT_TOKEN_DUMP")) {
        fprintf(stderr, "PROMPT_TOKENS %d:", n_prompt_tokens);
        for (int t = 0; t < n_prompt_tokens; t++)
            fprintf(stderr, " %d", prompt_tokens[t]);
        fprintf(stderr, "\n");
    }

    int max_recent = params->repeat_last_n;
    uint32_t *recent = calloc(max_recent, sizeof(uint32_t));
    int n_recent = 0;

    /* Prefill — the batched card path first; it refuses without touching the state and the
     * per-token loop below is what runs then. `recent` is seeded from the prompt either way. */
    tm_tensor *logits = pn_prefill_tt_batched(model, state, prompt_tokens, n_prompt_tokens);
    int prefill_batched = (logits != NULL);
    if (!logits && pn_prefill_tt_aborted(model)) {
        free(prompt_tokens);
        free(recent);
        return NULL;
    }
    /* The repeat-penalty window is PROMPT state, not forward state: seed it from the same
     * tokens whichever prefill ran, or a batched turn would sample under a different penalty
     * than a sequential one and the difference would look like a card defect. */
    for (int i = 0; prefill_batched && i < n_prompt_tokens; i++) {
        if (prompt_tokens[i] < 0) continue;
        if (n_recent < max_recent) {
            recent[n_recent++] = (uint32_t)prompt_tokens[i];
        } else {
            memmove(recent, recent + 1, (max_recent - 1) * sizeof(uint32_t));
            recent[max_recent - 1] = (uint32_t)prompt_tokens[i];
        }
    }
    for (int i = 0; !prefill_batched && i < n_prompt_tokens; i++) {
        if (prompt_tokens[i] < 0) continue;

        if (logits) tm_free(logits);
        logits = model->forward(model, (uint32_t)prompt_tokens[i], state);
        if (!logits) {
            fprintf(stderr, "generate: forward pass failed at token %d\n", i);
            free(prompt_tokens);
            free(recent);
            return NULL;
        }

        if (n_recent < max_recent) {
            recent[n_recent++] = (uint32_t)prompt_tokens[i];
        } else {
            memmove(recent, recent + 1, (max_recent - 1) * sizeof(uint32_t));
            recent[max_recent - 1] = (uint32_t)prompt_tokens[i];
        }
    }

    free(prompt_tokens);

    /* Decode loop */
    int *output_tokens = malloc(max_tokens * sizeof(int));
    int n_output = 0;

    int eos_id = ft_eos_id(model->tokenizer);

    for (int i = 0; i < max_tokens; i++) {
        if (!logits) break;

        float *logit_data = (float *)logits->data;
        pn_sampler_row_hint(logits); /* a backend's survivor list, this draw only */
        uint32_t next_token = sample_token(logit_data, model->config.n_vocab,
                                           params, recent, n_recent);
        tm_free(logits);
        logits = NULL;

        if (eos_id >= 0 && (int)next_token == eos_id) break;

        output_tokens[n_output++] = (int)next_token;

        if (n_recent < max_recent) {
            recent[n_recent++] = next_token;
        } else {
            memmove(recent, recent + 1, (max_recent - 1) * sizeof(uint32_t));
            recent[max_recent - 1] = next_token;
        }

        logits = model->forward(model, next_token, state);
    }

    if (logits) tm_free(logits);
    free(recent);

    char *raw_output = ft_decode(model->tokenizer, output_tokens, n_output);
    free(output_tokens);

    if (is_bpe) {
        char *output = gpt2_bytes_decode(raw_output);
        free(raw_output);
        return output;
    }
    return raw_output;
}

int pn_stream(pn_model *model, pn_state *state,
              const char *prompt, int max_tokens,
              pn_sampling_params *params,
              pn_token_callback callback, void *user_data) {
    if (!model || !state || !prompt || !params || !callback) return -1;
    if (!model->forward) return -1;
#ifdef USE_TT
    if (pn_qwen35moe_tt_select_rows(model, params, false) > 0u) {
        params->dense_row = pn_qwen35moe_tt_dense_row;
        params->tail_counts = pn_qwen35moe_tt_tail_counts;
        params->dense_ctx = model;
    }
#endif

    /* RWKV-X path: use internal trie tokenizer */
    if (model->config.arch == PN_ARCH_RWKVX) {
        uint32_t prompt_buf[8192];
        int n_prompt = pn_rwkvx_encode(model->weights, prompt, prompt_buf, 8192);
        if (n_prompt == 0) return -1;

        int max_recent = params->repeat_last_n;
        uint32_t *recent = calloc(max_recent, sizeof(uint32_t));
        int n_recent = 0;

        /* Prefill */
        tm_tensor *logits = NULL;
        for (int i = 0; i < n_prompt; i++) {
            if (logits) tm_free(logits);
            logits = model->forward(model, prompt_buf[i], state);
            if (!logits) { free(recent); return -1; }

            if (n_recent < max_recent)
                recent[n_recent++] = prompt_buf[i];
            else {
                memmove(recent, recent + 1, (max_recent - 1) * sizeof(uint32_t));
                recent[max_recent - 1] = prompt_buf[i];
            }
        }

        /* Decode with streaming */
        int n_generated = 0;
        int eos_id = pn_rwkvx_eos_id(model->weights);

        for (int i = 0; i < max_tokens; i++) {
            if (!logits) break;
            float *logit_data = (float *)logits->data;
            uint32_t next_token = sample_token(logit_data, model->config.n_vocab,
                                               params, recent, n_recent);
            tm_free(logits);
            logits = NULL;

            if ((int)next_token == eos_id) break;

            n_generated++;

            char *str = pn_rwkvx_decode(model->weights, &next_token, 1);
            if (str) { callback(str, user_data); free(str); }

            if (n_recent < max_recent)
                recent[n_recent++] = next_token;
            else {
                memmove(recent, recent + 1, (max_recent - 1) * sizeof(uint32_t));
                recent[max_recent - 1] = next_token;
            }

            logits = model->forward(model, next_token, state);
        }

        if (logits) tm_free(logits);
        free(recent);
        return n_generated;
    }

    /* GGUF path */
    if (!model->tokenizer) return -1;

    int is_bpe = (ft_get_model_type(model->tokenizer) == 0);

    int n_prompt_tokens = 0;
    int *prompt_tokens;
    if (is_bpe) {
        prompt_tokens = pn_bpe_encode(model->tokenizer, prompt, &n_prompt_tokens);
    } else {
        prompt_tokens = ft_encode_with_special(model->tokenizer, prompt,
                                               &n_prompt_tokens,
                                               true,    /* add BOS for SentencePiece */
                                               false);  /* no EOS */
    }
    if (!prompt_tokens || n_prompt_tokens == 0) {
        free(prompt_tokens);
        return -1;
    }

    int max_recent = params->repeat_last_n;
    uint32_t *recent = calloc(max_recent, sizeof(uint32_t));
    int n_recent = 0;

    /* Prefill — the batched card path first; it refuses without touching the state and the
     * per-token loop below is what runs then. `recent` is seeded from the prompt either way. */
    tm_tensor *logits = pn_prefill_tt_batched(model, state, prompt_tokens, n_prompt_tokens);
    int prefill_batched = (logits != NULL);
    if (!logits && pn_prefill_tt_aborted(model)) {
        free(prompt_tokens);
        free(recent);
        return -1;
    }
    /* The repeat-penalty window is PROMPT state, not forward state: seed it from the same
     * tokens whichever prefill ran, or a batched turn would sample under a different penalty
     * than a sequential one and the difference would look like a card defect. */
    for (int i = 0; prefill_batched && i < n_prompt_tokens; i++) {
        if (prompt_tokens[i] < 0) continue;
        if (n_recent < max_recent) {
            recent[n_recent++] = (uint32_t)prompt_tokens[i];
        } else {
            memmove(recent, recent + 1, (max_recent - 1) * sizeof(uint32_t));
            recent[max_recent - 1] = (uint32_t)prompt_tokens[i];
        }
    }
    for (int i = 0; !prefill_batched && i < n_prompt_tokens; i++) {
        if (prompt_tokens[i] < 0) continue;

        if (logits) tm_free(logits);
        logits = model->forward(model, (uint32_t)prompt_tokens[i], state);
        if (!logits) {
            free(prompt_tokens);
            free(recent);
            return -1;
        }

        if (n_recent < max_recent) {
            recent[n_recent++] = (uint32_t)prompt_tokens[i];
        } else {
            memmove(recent, recent + 1, (max_recent - 1) * sizeof(uint32_t));
            recent[max_recent - 1] = (uint32_t)prompt_tokens[i];
        }
    }
    free(prompt_tokens);

    /* Decode with streaming */
    int n_generated = 0;
    int eos_id = ft_eos_id(model->tokenizer);

    for (int i = 0; i < max_tokens; i++) {
        if (!logits) break;

        float *logit_data = (float *)logits->data;
        pn_sampler_row_hint(logits); /* a backend's survivor list, this draw only */
        uint32_t next_token = sample_token(logit_data, model->config.n_vocab,
                                           params, recent, n_recent);
        tm_free(logits);
        logits = NULL;

        if (eos_id >= 0 && (int)next_token == eos_id) break;

        n_generated++;

        int tok_id = (int)next_token;
        char *raw_str = ft_decode(model->tokenizer, &tok_id, 1);
        if (raw_str) {
            if (is_bpe) {
                char *tok_str = gpt2_bytes_decode(raw_str);
                free(raw_str);
                if (tok_str) {
                    callback(tok_str, user_data);
                    free(tok_str);
                }
            } else {
                callback(raw_str, user_data);
                free(raw_str);
            }
        }

        if (n_recent < max_recent) {
            recent[n_recent++] = next_token;
        } else {
            memmove(recent, recent + 1, (max_recent - 1) * sizeof(uint32_t));
            recent[max_recent - 1] = next_token;
        }

        logits = model->forward(model, next_token, state);
    }

    if (logits) tm_free(logits);
    free(recent);

    return n_generated;
}

/* ============================================================
 * Multi-modal inference API (Phase 3.9)
 *
 * Infrastructure for audio, vision, and embedding models.
 * Text models work immediately via pn_infer() → PN_OUT_LOGITS.
 * Task-specific functions return "not supported" until
 * model phases (3.10–3.21) implement them.
 * ============================================================ */

pn_output* pn_infer(pn_model *model, pn_state *state, uint32_t token) {
    if (!model || !state || !model->forward) return NULL;

    tm_tensor *logits = model->forward(model, token, state);
    if (!logits) return NULL;

    pn_output *out = calloc(1, sizeof(pn_output));
    if (!out) { tm_free(logits); return NULL; }

    int n = logits->shape[0];
    out->type = PN_OUT_LOGITS;
    out->data = malloc(n * sizeof(float));
    if (!out->data) { tm_free(logits); free(out); return NULL; }
    memcpy(out->data, logits->data, n * sizeof(float));
    out->n = n;

    tm_free(logits);
    return out;
}

pn_output* pn_infer_prefill(pn_model *model, pn_state *state, const uint32_t *tokens,
                            int n_tokens) {
    tm_tensor *logits;
    pn_output *out;
    int n;

    if (!model || !state || !tokens || n_tokens < 2) return NULL;
    logits = pn_prefill_tt_batched_tokens(model, state, tokens, n_tokens);
    if (!logits) return NULL;
    out = calloc(1, sizeof(pn_output));
    n = logits->shape[0];
    if (!out || !(out->data = malloc((size_t)n * sizeof(float)))) {
        free(out);
        tm_free(logits);
        return NULL;
    }
    memcpy(out->data, logits->data, (size_t)n * sizeof(float));
    out->type = PN_OUT_LOGITS;
    out->n = n;
    tm_free(logits);
    return out;
}

void pn_output_free(pn_output *out) {
    if (!out) return;
    free(out->data);
    free(out);
}

char* pn_transcribe(pn_model *model, pn_state *state,
                    const float *audio_samples, int n_samples,
                    int sample_rate) {
    (void)model; (void)state; (void)audio_samples; (void)n_samples; (void)sample_rate;
    fprintf(stderr, "pn_transcribe: not yet supported (requires Whisper model, Phase 3.12)\n");
    return NULL;
}

int pn_synthesize(pn_model *model, pn_state *state,
                  const char *text, float **out_samples,
                  int *out_sample_rate) {
    (void)model; (void)state; (void)text; (void)out_samples; (void)out_sample_rate;
    fprintf(stderr, "pn_synthesize: not yet supported (requires TTS model, Phase 3.14/3.15)\n");
    return -1;
}

int pn_attach_vision(pn_model *model, const char *mmproj_path) {
    if (!model || !mmproj_path) return -1;
    pn_gguf_file *g = pn_gguf_open(mmproj_path);
    if (!g) { fprintf(stderr, "pn_attach_vision: cannot open %s\n", mmproj_path); return -1; }
    uint64_t admitted_before = pn_host_admit_thread_total();
    pn_vision_model *v = pn_vision_load(g);
    pn_gguf_close(g);   /* vision weights are dequantized to owned F32 copies */
    /* Those copies were admitted on this thread; they live as long as the
     * model, so the model's free returns the charge. */
    uint64_t admitted = pn_host_admit_thread_total() - admitted_before;
    if (!v) {
        pn_host_release(admitted);
        return -1;
    }
    model->vision = v;
    model->host_admitted += admitted;
    return 0;
}

/* Build the ChatML vision turn (im_start user \n vstart image_pad*n_img vend
 * prompt im_end \n im_start assistant \n) with the supplied [n_img x n_embd]
 * decoder-space embeds spliced at the <|image_pad|> positions, reset the state,
 * batched-prefill its KV cache, and return the last-position logits (caller
 * tm_free's). The embeds buffer is NOT freed (caller owns it). NULL on error. */
static tm_tensor* vlm_prefill(pn_model *model, pn_state *state,
                              const float *img_embds, int n_img, const char *prompt) {
    int im_start  = ft_token_to_id(model->tokenizer, "<|im_start|>");
    int im_end    = ft_token_to_id(model->tokenizer, "<|im_end|>");
    int vstart    = ft_token_to_id(model->tokenizer, "<|vision_start|>");
    int vend      = ft_token_to_id(model->tokenizer, "<|vision_end|>");
    int image_pad = ft_token_to_id(model->tokenizer, "<|image_pad|>");

    int cap = n_img + 64, ns = 0;
    int *seq = malloc((size_t)cap * sizeof(int));
    if (!seq) return NULL;
    #define PUSH(id) do { if (ns >= cap) { cap *= 2; seq = realloc(seq, (size_t)cap * sizeof(int)); } seq[ns++] = (id); } while (0)
    #define PUSH_TEXT(s) do { int L_ = 0; int *t_ = pn_bpe_encode(model->tokenizer, (s), &L_); for (int i_ = 0; i_ < L_; i_++) PUSH(t_[i_]); free(t_); } while (0)

    if (im_start >= 0) PUSH(im_start);
    PUSH_TEXT("user\n");
    if (vstart >= 0) PUSH(vstart);
    for (int i = 0; i < n_img; i++) PUSH(image_pad);
    if (vend >= 0) PUSH(vend);
    if (prompt && *prompt) PUSH_TEXT(prompt);
    if (im_end >= 0) PUSH(im_end);
    PUSH_TEXT("\n");
    if (im_start >= 0) PUSH(im_start);
    PUSH_TEXT("assistant\n");
    #undef PUSH_TEXT

    pn_state_reset(state);
    int n_embd = model->config.n_embd, img_k = 0;
    tm_tensor *logits = NULL;

    uint32_t *ptoks = malloc((size_t)ns * sizeof(uint32_t));
    const float **prows = calloc((size_t)ns, sizeof(float *));
    if (ptoks && prows) {
        for (int i = 0; i < ns; i++) {
            if (seq[i] == image_pad && img_k < n_img) {
                prows[i] = img_embds + (size_t)(img_k++) * n_embd;
                ptoks[i] = 0;
            } else {
                ptoks[i] = (uint32_t)seq[i];
            }
        }
        logits = pn_llama_prefill(model, ptoks, prows, ns, state);
    }
    free(ptoks);
    free(prows);

    if (!logits) {
        /* Fallback: sequential single-token prefill. */
        img_k = 0;
        for (int i = 0; i < ns; i++) {
            if (logits) tm_free(logits);
            if (seq[i] == image_pad && img_k < n_img) {
                state->embd_override = img_embds + (size_t)(img_k++) * n_embd;
                logits = model->forward(model, 0, state);
                state->embd_override = NULL;
            } else {
                logits = model->forward(model, (uint32_t)seq[i], state);
            }
            if (!logits) { free(seq); return NULL; }
        }
    }
    free(seq);
    return logits;
}

/* Vision-language decode from PRE-SUPPLIED image embeddings: this is exactly
 * what pn_describe_stream does AFTER the vision encode. Splice the supplied
 * [n_img x embd_dim] decoder-space embeds into a ChatML user turn at the
 * <|image_pad|> positions, batched-prefill, and decode the assistant
 * transcription. The embeds buffer is NOT freed here (the caller owns it), so a
 * Core ML vision bridge can produce them and reuse the buffer. Returns newly
 * allocated text (caller frees with free()); NULL on error. */
char* pn_describe_embeds(pn_model *model, pn_state *state,
                         const float *img_embds, int n_img, int embd_dim,
                         const char *prompt, int max_tokens,
                         pn_sampling_params *params,
                         pn_token_callback callback, void *user_data,
                         pn_stop_callback should_stop, void *stop_data) {
    if (!model || !state || !img_embds || n_img <= 0) return NULL;
    if (embd_dim != model->config.n_embd) {
        fprintf(stderr, "pn_describe_embeds: embd_dim %d != n_embd %d\n",
                embd_dim, model->config.n_embd);
        return NULL;
    }

    tm_tensor *logits = vlm_prefill(model, state, img_embds, n_img, prompt);
    if (!logits) return NULL;

    int im_end = ft_token_to_id(model->tokenizer, "<|im_end|>");
    int max_recent = params->repeat_last_n > 0 ? params->repeat_last_n : 64;
    uint32_t *recent = calloc(max_recent, sizeof(uint32_t));
    int n_recent = 0;
    int *out = malloc((size_t)max_tokens * sizeof(int)), no = 0;
    int eos = ft_eos_id(model->tokenizer);
    for (int i = 0; i < max_tokens && logits; i++) {
        float *ld = (float *)logits->data;
        uint32_t nt = sample_token(ld, model->config.n_vocab, params, recent, n_recent);
        if (getenv("PN_VLM_DEBUG") && i < 6) {
            /* show the top-3 logits + the sampled id to diagnose empty output */
            int a = 0, b = 0, cc = 0;
            for (int v = 1; v < model->config.n_vocab; v++) {
                if (ld[v] > ld[a]) { cc = b; b = a; a = v; }
                else if (ld[v] > ld[b]) { cc = b; b = v; }
                else if (ld[v] > ld[cc]) cc = v;
            }
            fprintf(stderr, "  step %d: sampled=%u  top=[%d %.2f, %d %.2f, %d %.2f] (im_end=%d eos=%d)\n",
                    i, nt, a, ld[a], b, ld[b], cc, ld[cc], im_end, eos);
        }
        tm_free(logits);
        logits = NULL;
        if ((eos >= 0 && (int)nt == eos) || (int)nt == im_end) break;
        out[no++] = (int)nt;
        if (n_recent < max_recent) recent[n_recent++] = nt;
        else { memmove(recent, recent + 1, (max_recent - 1) * sizeof(uint32_t)); recent[max_recent - 1] = nt; }
        if (callback) {
            /* Emit the full decoded text so far. Decoding the whole accumulated
             * sequence each step keeps qwen byte-level BPE valid (a partial
             * multi-byte character simply is not shown until it completes). */
            char *raw_i = ft_decode(model->tokenizer, out, no);
            char *txt_i = gpt2_bytes_decode(raw_i);
            free(raw_i);
            if (txt_i) { callback(txt_i, user_data); free(txt_i); }
        }
        if (should_stop && should_stop(stop_data)) break;   // abort generation early
        logits = model->forward(model, nt, state);
    }
    if (logits) tm_free(logits);
    free(recent);
    char *raw = ft_decode(model->tokenizer, out, no);
    free(out);
    char *text = gpt2_bytes_decode(raw);   /* qwen byte-level BPE -> UTF-8 */
    free(raw);
    return text;
}

/* Prefill a state's KV cache from pre-supplied image embeds (ChatML vision turn)
 * and sample the first generated token, WITHOUT running the decode loop. Leaves
 * state ready to be driven by pn_batch_decoder / pn_llama_forward_batch. Returns
 * 0 and writes *first_tok_out on success, -1 on error. */
int pn_vlm_prefill_embeds(pn_model *model, pn_state *state,
                          const float *img_embds, int n_img, int embd_dim,
                          const char *prompt, pn_sampling_params *params,
                          uint32_t *first_tok_out) {
    if (!model || !state || !img_embds || n_img <= 0 || !first_tok_out) return -1;
    if (embd_dim != model->config.n_embd) return -1;
    tm_tensor *logits = vlm_prefill(model, state, img_embds, n_img, prompt);
    if (!logits) return -1;
    pn_sampling_params sp = params ? *params : pn_sampling_defaults();
    *first_tok_out = sample_token((float *)logits->data, model->config.n_vocab, &sp, NULL, 0);
    tm_free(logits);
    return 0;
}

/* ============================================================
 * Continuous-batch VLM decoder
 *
 * Drives pn_llama_forward_batch over up to max_batch prefilled sequences,
 * sampling + EOS + detokenization per sequence, so a caller can keep the batch
 * full (continuous batching): admit a prefilled page into any free slot, call
 * step() repeatedly, and each finished page is returned (with its text and its
 * recyclable state) so a fresh page can take the slot.
 * ============================================================ */

typedef struct {
    pn_state *state;
    int       page_id;
    uint32_t  pending;      /* sampled-but-not-yet-processed token */
    int      *out;          /* generated token ids */
    int       no, ocap;
    uint32_t *recent;       /* repeat-penalty ring */
    int       n_recent;
    int       in_use;
    /* Speculative-decode fields (used only by pn_batch_decoder_step_spec). */
    pn_draft_fn draft_fn;   /* per-slot draft provider (NULL = plain K=1 slot) */
    void       *draft_ud;
    int         spec_on;    /* 1 while this slot still drafts (per-page fallback) */
    int         spec_proved;/* 1 once any cycle accepted >= min_accept_frac */
    long        passes;     /* speculative cycles this slot has run */
    long        drafted;    /* draft tokens proposed for this slot */
    long        accepted;   /* draft tokens accepted for this slot */
} pn_batch_slot;

struct pn_batch_decoder {
    pn_model          *model;
    int                max_batch;
    int                max_tokens;
    pn_sampling_params params;
    int                max_recent;
    int                eos, im_end;
    pn_batch_slot     *slots;
    /* scratch reused across steps */
    pn_state         **act_states;
    uint32_t          *act_toks;
    int               *act_slot;
    float             *logits;
    /* Speculative-decode config + ragged verify scratch (allocated by
     * pn_batch_decoder_spec_config; NULL until then). */
    int                spec_configured;
    int                max_k;
    double             min_accept_frac;
    int                probe_cycles;
    uint32_t          *cat_toks;    /* [PN_QMM_MAX_B] concatenated block tokens */
    pn_state         **blk_states;  /* [max_batch] one state per block */
    int               *blk_len;     /* [max_batch] rows per block (1 + drafts) */
    int               *blk_slot;    /* [max_batch] slot index per block */
    float             *spec_logits; /* [PN_QMM_MAX_B * n_vocab] per-row logits */
    /* Aggregate speculative-decode accounting (observability). */
    long               sp_passes;   /* verify_batch passes actually run */
    long               sp_drafted;  /* draft tokens proposed */
    long               sp_accepted; /* draft tokens accepted */
};

pn_batch_decoder* pn_batch_decoder_new(pn_model *model, int max_batch,
                                       int max_tokens, const pn_sampling_params *params) {
    if (!model || max_batch <= 0) return NULL;
    pn_batch_decoder *d = calloc(1, sizeof(*d));
    if (!d) return NULL;
    d->model = model;
    d->max_batch = max_batch;
    d->max_tokens = max_tokens > 0 ? max_tokens : 4096;
    d->params = params ? *params : pn_sampling_defaults();
    d->max_recent = d->params.repeat_last_n > 0 ? d->params.repeat_last_n : 64;
    d->eos = ft_eos_id(model->tokenizer);
    d->im_end = ft_token_to_id(model->tokenizer, "<|im_end|>");
    d->slots       = calloc(max_batch, sizeof(pn_batch_slot));
    d->act_states  = malloc((size_t)max_batch * sizeof(pn_state *));
    d->act_toks    = malloc((size_t)max_batch * sizeof(uint32_t));
    d->act_slot    = malloc((size_t)max_batch * sizeof(int));
    d->logits      = malloc((size_t)max_batch * model->config.n_vocab * sizeof(float));
    if (!d->slots || !d->act_states || !d->act_toks || !d->act_slot || !d->logits) {
        pn_batch_decoder_free(d); return NULL;
    }
    for (int i = 0; i < max_batch; i++) {
        d->slots[i].ocap = d->max_tokens;
        d->slots[i].out = malloc((size_t)d->max_tokens * sizeof(int));
        d->slots[i].recent = calloc(d->max_recent, sizeof(uint32_t));
        if (!d->slots[i].out || !d->slots[i].recent) { pn_batch_decoder_free(d); return NULL; }
    }
    return d;
}

void pn_batch_decoder_free(pn_batch_decoder *d) {
    if (!d) return;
    if (d->slots) for (int i = 0; i < d->max_batch; i++) {
        free(d->slots[i].out); free(d->slots[i].recent);
    }
    free(d->slots); free(d->act_states); free(d->act_toks);
    free(d->act_slot); free(d->logits);
    free(d->cat_toks); free(d->blk_states); free(d->blk_len);
    free(d->blk_slot); free(d->spec_logits);
    free(d);
}

int pn_batch_decoder_capacity(const pn_batch_decoder *d) { return d ? d->max_batch : 0; }

int pn_batch_decoder_active(const pn_batch_decoder *d) {
    if (!d) return 0;
    int n = 0;
    for (int i = 0; i < d->max_batch; i++) if (d->slots[i].in_use) n++;
    return n;
}

int pn_batch_decoder_admit(pn_batch_decoder *d, pn_state *state, int page_id,
                           uint32_t first_tok) {
    if (!d || !state) return -1;
    for (int i = 0; i < d->max_batch; i++) {
        pn_batch_slot *s = &d->slots[i];
        if (s->in_use) continue;
        s->state = state; s->page_id = page_id;
        s->pending = first_tok; s->no = 0; s->n_recent = 0; s->in_use = 1;
        return 0;
    }
    return -1;   /* full */
}

/* Finalize a slot: detokenize its output to UTF-8, hand back page_id/state/text,
 * and free the slot for reuse. */
static void batch_finalize(pn_batch_decoder *d, pn_batch_slot *s, pn_page_out *o) {
    char *raw = ft_decode(d->model->tokenizer, s->out, s->no);
    char *text = raw ? gpt2_bytes_decode(raw) : NULL;
    free(raw);
    o->page_id = s->page_id;
    o->state = s->state;
    o->text = text ? text : strdup("");
    s->in_use = 0; s->state = NULL;
}

int pn_batch_decoder_step(pn_batch_decoder *d, pn_page_out *out, int max_out) {
    if (!d) return 0;
    int nf = 0;

    /* 1. Process each slot's pending token: EOS/max -> finish; else push it and
     *    enqueue the slot for the batched forward that produces its next token. */
    int na = 0;
    for (int i = 0; i < d->max_batch; i++) {
        pn_batch_slot *s = &d->slots[i];
        if (!s->in_use) continue;
        int done = ((d->eos >= 0 && (int)s->pending == d->eos) ||
                    (int)s->pending == d->im_end || s->no >= d->max_tokens);
        if (done) {
            if (nf < max_out) batch_finalize(d, s, &out[nf++]);
            continue;
        }
        s->out[s->no++] = (int)s->pending;
        if (s->n_recent < d->max_recent) s->recent[s->n_recent++] = s->pending;
        else { memmove(s->recent, s->recent + 1, (d->max_recent - 1) * sizeof(uint32_t));
               s->recent[d->max_recent - 1] = s->pending; }
        d->act_states[na] = s->state;
        d->act_toks[na] = s->pending;
        d->act_slot[na] = i;
        na++;
    }

    /* 2. One batched decode step over the continuing slots; sample each slot's
     *    next pending token.
     *
     *    NOTE ON DETERMINISM: sampling is kept SERIAL on purpose. Each slot's
     *    logits / recent-token ring / candidate set are independent, but the
     *    random draw goes through the process-global rand()/srand() state (see
     *    rand_float()). Parallelizing the per-slot draws would make the order in
     *    which slots consume rand() nondeterministic and change the sampled
     *    tokens run-to-run, breaking reproducibility. Because the per-call
     *    sampler is now O(V) (no O(k*V) sort, no repeated full-vocab top-p
     *    passes, no per-call mallocs), the serial loop is already much cheaper.
     *    If a per-slot deterministic RNG is introduced, this loop can be
     *    parallelized over the thread pool safely. */
    if (na > 0) {
        if (pn_llama_forward_batch(d->model, d->act_toks, NULL, d->act_states, na, d->logits) != 0)
            return -1;   /* unsupported: caller falls back */
        int nv = d->model->config.n_vocab;
        for (int j = 0; j < na; j++) {
            pn_batch_slot *s = &d->slots[d->act_slot[j]];
            s->pending = sample_token(d->logits + (size_t)j * nv, nv, &d->params,
                                      s->recent, s->n_recent);
        }
    }
    return nf;
}

/* Finalize ALL active slots with whatever they have decoded so far (for early
 * stop / drain). Returns the number written to out[]. */
int pn_batch_decoder_flush(pn_batch_decoder *d, pn_page_out *out, int max_out) {
    if (!d) return 0;
    int nf = 0;
    for (int i = 0; i < d->max_batch && nf < max_out; i++) {
        pn_batch_slot *s = &d->slots[i];
        if (s->in_use) batch_finalize(d, s, &out[nf++]);
    }
    return nf;
}

/* ============================================================
 * Batched lossless speculative decode (pn_batch_decoder_step_spec)
 *
 * One verify pass over the concatenation of every active slot's
 * [pending, drafts...] block, then a per-slot greedy accept scan. A slot with
 * no drafts (K=1 block) reduces EXACTLY to pn_batch_decoder_step's per-slot
 * behavior: push the pending seed, verify one row, and set the next pending to
 * the penalized argmax of that row over the same recent-token ring. So the
 * emitted token stream is bit-identical to plain batched greedy decode for any
 * draft; drafts only fold multiple plain steps into one verify pass.
 * ============================================================ */

/* Slide one token into a slot's repeat-penalty ring, byte-for-byte as the plain
 * step does (append until full, then shift). */
static void spec_ring_push(pn_batch_decoder *d, pn_batch_slot *s, uint32_t tok) {
    if (s->n_recent < d->max_recent) {
        s->recent[s->n_recent++] = tok;
    } else {
        memmove(s->recent, s->recent + 1, (d->max_recent - 1) * sizeof(uint32_t));
        s->recent[d->max_recent - 1] = tok;
    }
}

/* Greedy argmax after applying the repeat penalty, a verbatim copy of
 * sample_token's temperature<=0 path (penalty loop over the recent ring, then
 * argmax). Mutates logits[] in place, exactly as sample_token would, so the
 * accepted token matches plain greedy decode bit for bit. */
static uint32_t penalized_argmax(float *logits, int n_vocab, pn_sampling_params *params,
                                 uint32_t *recent_tokens, int n_recent) {
    if (params->repeat_penalty != 1.0f && recent_tokens && n_recent > 0) {
        for (int i = 0; i < n_recent; i++) {
            uint32_t tok = recent_tokens[i];
            if (tok < (uint32_t)n_vocab) {
                if (logits[tok] > 0.0f)
                    logits[tok] /= params->repeat_penalty;
                else
                    logits[tok] *= params->repeat_penalty;
            }
        }
    }
    int best = 0;
    for (int i = 1; i < n_vocab; i++)
        if (logits[i] > logits[best]) best = i;
    return (uint32_t)best;
}

int pn_batch_decoder_spec_config(pn_batch_decoder *d, int max_k,
                                 double min_accept_frac, int probe_cycles) {
    if (!d) return -1;
    /* Speculation is only lossless for greedy decode. */
    if (d->params.temperature > 0.0f) return -1;
    if (max_k < 0) max_k = 0;
    /* A block is 1 seed + up to max_k drafts and the whole step is capped at
     * PN_QMM_MAX_B verify rows, so a single block can hold at most
     * PN_QMM_MAX_B - 1 drafts. */
    if (max_k > PN_QMM_MAX_B - 1) max_k = PN_QMM_MAX_B - 1;
    d->max_k = max_k;
    d->min_accept_frac = min_accept_frac;
    d->probe_cycles = probe_cycles;

    int nv = d->model->config.n_vocab;
    free(d->cat_toks); free(d->blk_states); free(d->blk_len);
    free(d->blk_slot); free(d->spec_logits);
    d->cat_toks    = malloc((size_t)PN_QMM_MAX_B * sizeof(uint32_t));
    d->blk_states  = malloc((size_t)d->max_batch * sizeof(pn_state *));
    d->blk_len     = malloc((size_t)d->max_batch * sizeof(int));
    d->blk_slot    = malloc((size_t)d->max_batch * sizeof(int));
    /* Total verify rows are capped at PN_QMM_MAX_B, so PN_QMM_MAX_B rows of
     * logits is always enough regardless of max_batch / max_k. */
    d->spec_logits = malloc((size_t)PN_QMM_MAX_B * (size_t)nv * sizeof(float));
    if (!d->cat_toks || !d->blk_states || !d->blk_len || !d->blk_slot || !d->spec_logits) {
        free(d->cat_toks); free(d->blk_states); free(d->blk_len);
        free(d->blk_slot); free(d->spec_logits);
        d->cat_toks = NULL; d->blk_states = NULL; d->blk_len = NULL;
        d->blk_slot = NULL; d->spec_logits = NULL;
        return -1;
    }
    d->sp_passes = 0; d->sp_drafted = 0; d->sp_accepted = 0;
    d->spec_configured = 1;
    return 0;
}

void pn_batch_decoder_spec_stats(const pn_batch_decoder *d, long *passes,
                                 long *drafted, long *accepted) {
    if (!d) return;
    if (passes)   *passes   = d->sp_passes;
    if (drafted)  *drafted  = d->sp_drafted;
    if (accepted) *accepted = d->sp_accepted;
}

int pn_batch_decoder_admit_spec(pn_batch_decoder *d, pn_state *state, int page_id,
                                uint32_t first_tok, pn_draft_fn draft_fn, void *draft_ud) {
    if (!d || !state || !d->spec_configured) return -1;
    for (int i = 0; i < d->max_batch; i++) {
        pn_batch_slot *s = &d->slots[i];
        if (s->in_use) continue;
        s->state = state; s->page_id = page_id;
        s->pending = first_tok; s->no = 0; s->n_recent = 0; s->in_use = 1;
        s->draft_fn = draft_fn; s->draft_ud = draft_ud;
        s->spec_on = (draft_fn != NULL && d->max_k > 0);
        s->spec_proved = 0;
        s->passes = 0; s->drafted = 0; s->accepted = 0;
        return 0;
    }
    return -1;   /* full */
}

/* Append a committed (accepted) token to a slot's output + ring, honoring the
 * SAME stop conditions as plain step: EOS / <|im_end|> / max_tokens are never
 * appended — they are routed back through `pending` so the slot finalizes on the
 * next step exactly like plain decode. Returns 1 if a stop was hit (caller must
 * stop emitting this block), 0 otherwise. */
static int spec_emit_committed(pn_batch_decoder *d, pn_batch_slot *s, uint32_t tok) {
    if ((d->eos >= 0 && (int)tok == d->eos) || (int)tok == d->im_end ||
        s->no >= d->max_tokens) {
        s->pending = tok;     /* finalize next step; not appended (matches plain) */
        return 1;
    }
    s->out[s->no++] = (int)tok;
    spec_ring_push(d, s, tok);
    return 0;
}

int pn_batch_decoder_step_spec(pn_batch_decoder *d, pn_page_out *out, int max_out) {
    if (!d) return 0;
    if (!d->spec_configured) return -1;
    int nv = d->model->config.n_vocab;
    int nf = 0;

    /* Phase 1: process each slot's pending token. Finish on EOS/max exactly like
     * plain step; otherwise push the pending "seed", ask the slot's draft_fn for
     * up to K drafts (clamped to max_k, the remaining verify-row budget, and the
     * no-wraparound budget), and lay [seed, drafts...] into the concatenated
     * block buffer. */
    int nblk = 0;   /* number of blocks */
    int M = 0;      /* total verify rows queued so far */
    for (int i = 0; i < d->max_batch; i++) {
        pn_batch_slot *s = &d->slots[i];
        if (!s->in_use) continue;

        int done = ((d->eos >= 0 && (int)s->pending == d->eos) ||
                    (int)s->pending == d->im_end || s->no >= d->max_tokens);
        if (done) {
            if (nf < max_out) batch_finalize(d, s, &out[nf++]);
            continue;
        }

        /* Guard the per-step verify-row cap BEFORE mutating the slot: each block
         * needs at least its seed row. If the batch is already full (only when
         * more than PN_QMM_MAX_B slots are active), leave this slot's pending in
         * place untouched for a later step. */
        if (M + 1 > PN_QMM_MAX_B) continue;

        /* Push the pending seed into out + ring (position row 0 of the block). */
        s->out[s->no++] = (int)s->pending;
        spec_ring_push(d, s, s->pending);

        pn_llama_kv_cache *kv = (pn_llama_kv_cache *)s->state->kv_cache;
        int budget_rows = PN_QMM_MAX_B - M - 1;               /* draft rows left this step */
        int budget_wrap = kv->max_seq - s->state->pos - 1;    /* no cache wraparound */
        int kmax = d->max_k;
        if (kmax > budget_rows) kmax = budget_rows;
        if (kmax > budget_wrap) kmax = budget_wrap;
        if (kmax < 0) kmax = 0;

        int kk = 0;
        if (s->spec_on && s->draft_fn && kmax > 0) {
            /* draft_fn writes straight into the block, right after the seed. */
            kk = s->draft_fn((const uint32_t *)s->out, s->no,
                             d->cat_toks + M + 1, kmax, s->draft_ud);
            if (kk < 0) kk = 0;
            if (kk > kmax) kk = kmax;   /* clamp to the requested budget */
        }

        d->cat_toks[M] = s->pending;             /* seed = row 0 */
        d->blk_states[nblk] = s->state;
        d->blk_len[nblk]    = 1 + kk;
        d->blk_slot[nblk]   = i;
        nblk++;
        M += 1 + kk;

        s->passes++;
        s->drafted += kk;
    }

    if (nblk == 0) return nf;   /* nothing to verify (all finished this step) */

    /* Phase 2: one ragged verify pass over every block. */
    if (pn_llama_verify_batch(d->model, d->cat_toks, d->blk_states, d->blk_len,
                              nblk, d->spec_logits) != 0)
        return -1;   /* unsupported model: caller falls back to plain step */
    d->sp_passes++;

    /* Phase 3: per block, greedy accept scan. logits row (off+i) predicts the
     * token AFTER block row i; row 0 predicts the token after the seed. Emit the
     * penalized argmax per row, accept draft[i] iff it equals that argmax, stop
     * at the first mismatch (the argmax is the correction). On full acceptance
     * take the free bonus argmax. The committed tokens (all but the last emitted)
     * go to out + ring now; the last emitted (correction or bonus) becomes the
     * next pending. Commit pos over the seed + accepted drafts only. */
    int off = 0;
    for (int b = 0; b < nblk; b++) {
        pn_batch_slot *s = &d->slots[d->blk_slot[b]];
        int blk_len = d->blk_len[b];
        int kk = blk_len - 1;                 /* number of drafts in this block */
        int base_pos = s->state->pos;
        const uint32_t *drafts = d->cat_toks + off + 1;

        int n_accept = 0, stopped = 0;
        for (int i = 0; i < kk; i++) {
            float *row = d->spec_logits + (size_t)(off + i) * nv;
            uint32_t g = penalized_argmax(row, nv, &d->params, s->recent, s->n_recent);
            if (g == drafts[i]) {
                /* Accepted: g is a committed token; push to out + ring so the
                 * next row's penalty sees it, exactly as plain decode would. */
                n_accept++;
                if (spec_emit_committed(d, s, g)) { stopped = 1; break; }
            } else {
                /* Mismatch: g is the correction and the last emitted token; it
                 * becomes the next pending (pushed to the ring next step). */
                s->pending = g;
                stopped = 1;
                break;
            }
        }
        if (!stopped) {
            /* Every draft accepted: bonus argmax after the last draft becomes the
             * next pending. */
            float *row = d->spec_logits + (size_t)(off + kk) * nv;
            s->pending = penalized_argmax(row, nv, &d->params, s->recent, s->n_recent);
        }

        /* Commit the seed (base_pos) plus every accepted draft; their K/V is
         * already in the cache from the verify pass. The correction/bonus stays
         * uncommitted and is re-verified as the next seed. */
        s->state->pos = base_pos + 1 + n_accept;
        s->accepted += n_accept;
        d->sp_drafted += kk;
        d->sp_accepted += n_accept;

        /* A page opens with a running header / page number that the draft source
         * cannot match, so early cycles accept nothing; mark the slot "proved"
         * once any single cycle clears min_accept_frac, which the body reliably
         * does. Without this the cumulative rate stays low through the header and
         * the probe below would drop a page that speculates well on its body. */
        if (kk > 0 && (double)n_accept / (double)kk >= d->min_accept_frac)
            s->spec_proved = 1;

        /* Per-page fallback: after probe_cycles, drop a slot that has never once
         * speculated well to plain single-token decode (it then rides as a K=1
         * block). A proved slot is never dropped. */
        if (s->spec_on && !s->spec_proved && s->passes >= d->probe_cycles &&
            s->drafted > 0 &&
            (double)s->accepted / (double)s->drafted < d->min_accept_frac) {
            s->spec_on = 0;
        }

        off += blk_len;
    }

    return nf;
}

/* Vision-language: encode the image (CPU flash-attention path), then delegate to
 * pn_describe_embeds. Kept as the portable fallback when no external (Core ML)
 * vision encoder is available. */
char* pn_describe_stream(pn_model *model, pn_state *state,
                         const uint8_t *image_rgb, int width, int height,
                         const char *prompt, int max_tokens,
                         pn_sampling_params *params,
                         pn_token_callback callback, void *user_data,
                         pn_stop_callback should_stop, void *stop_data) {
    if (!model || !state || !image_rgb) return NULL;
    if (!model->vision) { fprintf(stderr, "pn_describe: no vision encoder attached\n"); return NULL; }

    float *img_embd = NULL;
    int img_dim = 0;
    int n_img = pn_vision_encode((pn_vision_model *)model->vision, image_rgb, width, height,
                                 &img_embd, &img_dim);
    if (n_img <= 0 || img_dim != model->config.n_embd) {
        fprintf(stderr, "pn_describe: vision encode failed (n=%d dim=%d vs n_embd=%d)\n",
                n_img, img_dim, model->config.n_embd);
        free(img_embd);
        return NULL;
    }
    char *text = pn_describe_embeds(model, state, img_embd, n_img, img_dim,
                                    prompt, max_tokens, params,
                                    callback, user_data, should_stop, stop_data);
    free(img_embd);
    return text;
}

char* pn_describe(pn_model *model, pn_state *state,
                  const uint8_t *image_rgb, int width, int height,
                  const char *prompt, int max_tokens,
                  pn_sampling_params *params) {
    return pn_describe_stream(model, state, image_rgb, width, height,
                              prompt, max_tokens, params, NULL, NULL, NULL, NULL);
}

/* Run just the CPU vision encoder for one image, returning the decoder-space
 * embeddings [n_img x n_embd] (caller frees *embeds_out with free()). This is
 * the vision half of pn_describe_stream, split out so a bulk pipeline can stage
 * embeds ahead of the batched decode (and mix with an external Core ML encoder).
 * Returns 0 on success, -1 on error. */
int pn_vision_encode_rgb(pn_model *model, const uint8_t *image_rgb, int width, int height,
                         float **embeds_out, int *n_img_out, int *embd_dim_out) {
    if (!model || !image_rgb || !embeds_out || !n_img_out) return -1;
    if (!model->vision) { fprintf(stderr, "pn_vision_encode_rgb: no vision encoder\n"); return -1; }
    float *emb = NULL;
    int dim = 0;
    int n_img = pn_vision_encode((pn_vision_model *)model->vision, image_rgb, width, height,
                                 &emb, &dim);
    if (n_img <= 0 || dim != model->config.n_embd) { free(emb); return -1; }
    *embeds_out = emb;
    *n_img_out = n_img;
    if (embd_dim_out) *embd_dim_out = dim;
    return 0;
}

float* pn_detect(pn_model *model, pn_state *state,
                 const uint8_t *image_rgb, int width, int height,
                 float confidence_threshold, int *n_boxes_out) {
    (void)model; (void)state; (void)image_rgb; (void)width; (void)height;
    (void)confidence_threshold;
    if (n_boxes_out) *n_boxes_out = 0;
    fprintf(stderr, "pn_detect: not yet supported (requires YOLO/RF-DETR model, Phase 3.18)\n");
    return NULL;
}

float* pn_embed(pn_model *model, pn_state *state,
                const char *text, int *n_dims_out) {
    (void)model; (void)state; (void)text;
    if (n_dims_out) *n_dims_out = 0;
    fprintf(stderr, "pn_embed: not yet supported (requires CLIP/DINOv2 model, Phase 3.17)\n");
    return NULL;
}

/* SPECULATIVE SAMPLING's chooser (pn_spec_choose_fn): one verify row's sparse logits become the
 * row a plain sparse decode step hands the sampler (-INFINITY everywhere the card did not
 * name), and sample_token draws from it under the repeat window plain decode would have at
 * that token — the cycle's own draws appended exactly as the emit loop appends them.
 * With proposals (c->q, true speculative sampling) a draft row is decided by min(1, p/q) and
 * the residual instead (spec_sampling_chooser.h); the plain draw stays the rule for the bonus
 * row and for any row whose p the top-N cannot give exactly. */
/* THE TREE PROBE's sampled half (`PN_TT_SPEC_TREE_PROBE=1`, diagnostics): per verify row the
 * plain sampler's final distribution p (on a COPY of the row), booked as max p and the top-2
 * mass — the old rule's acceptance with a perfect head, and with a perfect width-2 sibling. */
static void spec_maxp_probe(const pn_spec_sample_ctx *c)
{
    static int on = -1;
    static float *scratch;
    static double sum1, sum2;
    static unsigned long rows;
    int ids[64], S, i;
    float probs[64], p1 = 0.0f, p2 = 0.0f;

    if (on < 0)
        on = getenv("PN_TT_SPEC_TREE_PROBE") && atoi(getenv("PN_TT_SPEC_TREE_PROBE")) != 0;
    if (!on || (!scratch && !(scratch = malloc((size_t)c->n_vocab * sizeof(float)))))
        return;
    memcpy(scratch, c->row, (size_t)c->n_vocab * sizeof(float));
    S = pn_sampler_row_final_probs(scratch, c->n_vocab, c->params, c->recent, c->n_recent, true,
                                   ids, probs, 64);
    if (S < 1)
        return;
    for (i = 0; i < S; i++) {
        if (probs[i] > p1) { p2 = p1; p1 = probs[i]; }
        else if (probs[i] > p2) p2 = probs[i];
    }
    sum1 += p1;
    sum2 += p1 + p2;
    if (++rows % 128u == 0u)
        fprintf(stderr, "SPEC MAXP PROBE — rows %lu mean max p %.4f mean top-2 mass %.4f\n", rows,
                sum1 / (double)rows, sum2 / (double)rows);
}

static bool spec_red_window_stale(void)
{
    static int red = -1;

    if (red < 0)
        red = getenv("PN_TT_RED") && strcmp(getenv("PN_TT_RED"), "spec_window_stale") == 0;
    return red != 0;
}

uint32_t pn_spec_sample_choose(void *opaque, const uint32_t *index, const float *value,
                               uint32_t count)
{
    pn_spec_sample_ctx *c = (pn_spec_sample_ctx *)opaque;
    uint32_t i, token;
    const int row = c->row_index++;

    for (i = 0; i < (uint32_t)c->n_vocab; i++)
        c->row[i] = -INFINITY;
    for (i = 0; i < count; i++)
        if (index[i] < (uint32_t)c->n_vocab)
            c->row[index[i]] = value[i];
    spec_maxp_probe(c);
    if (c->q && row < c->n_q && pn_spec_rule_choose_row(c, row, &token))
        goto push;
    /* A row past 0 whose top-N cannot decide this draw: end the cycle there (the next
     * cycle meets the same distribution as its row 0, which the dense refill covers). The
     * sampler says so itself (t_refill_forbidden): before any draw for a REFILL verdict or
     * a DENSE tail plan, after it for a mixture's residual branch — that uniform is carried
     * in params to the next draw, which is this position's. */
    if (row > 0 && !c->params->dense_row &&
        !pn_sampler_topn_row_exact(c->row, c->n_vocab, c->params, c->recent, c->n_recent))
        return PN_SPEC_CHOOSE_DEFER_TOKEN;   /* no refill at all: the pre-sampler check */
    t_refill_forbidden = row > 0;
    t_sample_row = row;
    /* The sampler's sparse fast path (`PN_TT_SPEC_SAMPLE_SPARSE=0`: the full-row path). */
    t_sparse_sampler_force = !(getenv("PN_TT_SPEC_SAMPLE_SPARSE") &&
                               atoi(getenv("PN_TT_SPEC_SAMPLE_SPARSE")) == 0);
    token = sample_token(c->row, c->n_vocab, c->params, c->recent, c->n_recent);
    t_sparse_sampler_force = 0;
    t_refill_forbidden = 0;
    t_sample_row = 0;
    if (token == PN_SPEC_CHOOSE_DEFER_TOKEN)
        return PN_SPEC_CHOOSE_DEFER_TOKEN;
push:
    /* RED `spec_window_stale` (PN_TT_RED): later verify rows draw under the cycle's STARTING
     * window, blind to the tokens accepted before them — a penalised request MUST differ. */
    if (spec_red_window_stale())
        return token;
    if (c->max_recent > 0) {
        if (c->n_recent < c->max_recent) {
            c->recent[c->n_recent++] = token;
        } else {
            memmove(c->recent, c->recent + 1, (size_t)(c->max_recent - 1) * sizeof(uint32_t));
            c->recent[c->max_recent - 1] = token;
        }
    }
    return token;
}

int pn_stream_tokens(pn_model *model, pn_state *state,
                     const uint32_t *tokens, int n_tokens,
                     int max_tokens,
                     pn_sampling_params *params,
                     pn_token_callback callback, void *user_data) {
    if (!model || !state || !tokens || n_tokens <= 0 || !params || !callback) return -1;
    if (!model->forward) return -1;
    if (!model->tokenizer && model->config.arch != PN_ARCH_RWKVX) return -1;

    int max_recent = params->repeat_last_n;
    uint32_t *recent = calloc(max_recent, sizeof(uint32_t));
    int n_recent = 0;

    /* Prefill — the batched card path first, exactly as pn_generate() and pn_stream() do.
     * THIS ENTRY POINT IS THE ONE THE CHAT CLI AND THE REPL TAKE, and it had a per-token
     * loop of its own: the Ornith chunkwise prefill shipped at 24x on 2026-08-25 and never
     * ran here once. A refusal returns NULL without touching the state and the loop below is
     * what runs then, so this is an accelerator and not a behaviour fork. */
#ifdef USE_TT
    /* Top-N rows: the sampler swaps in the dense row whenever one cannot decide a draw. */
    if (pn_qwen35moe_tt_select_rows(model, params, false) > 0u) {
        params->dense_row = pn_qwen35moe_tt_dense_row;
        params->tail_counts = pn_qwen35moe_tt_tail_counts;
        params->dense_ctx = model;
    }
#endif
    tm_tensor *logits = pn_prefill_tt_batched_tokens(model, state, tokens, n_tokens);
    int prefill_batched = (logits != NULL);
    /* The repeat-penalty window is PROMPT state, not forward state: seed it from the same
     * tokens whichever prefill ran, or a batched turn would sample under a different penalty
     * than a sequential one and the difference would look like a card defect. */
    for (int i = 0; prefill_batched && i < n_tokens; i++) {
        if (n_recent < max_recent) {
            recent[n_recent++] = tokens[i];
        } else {
            memmove(recent, recent + 1, (max_recent - 1) * sizeof(uint32_t));
            recent[max_recent - 1] = tokens[i];
        }
    }
    for (int i = 0; !prefill_batched && i < n_tokens; i++) {
        if (logits) tm_free(logits);
        logits = model->forward(model, tokens[i], state);
        if (!logits) {
            free(recent);
            return -1;
        }

        if (n_recent < max_recent) {
            recent[n_recent++] = tokens[i];
        } else {
            memmove(recent, recent + 1, (max_recent - 1) * sizeof(uint32_t));
            recent[max_recent - 1] = tokens[i];
        }
    }

    /* Decode with streaming */
    int n_generated = 0;
    int eos_id;
    int is_bpe = 0;

    if (model->config.arch == PN_ARCH_RWKVX) {
        eos_id = pn_rwkvx_eos_id(model->weights);
    } else {
        eos_id = ft_eos_id(model->tokenizer);
        is_bpe = (ft_get_model_type(model->tokenizer) == 0);
    }

    /* STEP 4 S2: GREEDY SPECULATIVE DECODE on the Tenstorrent qwen35 forward
     * (`PN_TT_SPEC=<drafts a step>`, 0 = off). Lossless: only the model's own greedy tokens
     * are ever emitted. Drafts: prompt-lookup n-grams over the whole history. */
#ifdef USE_TT
    int spec_k = 0;
    /* SHIPPED DEFAULT 3 drafts (2026-10-06); `PN_TT_SPEC=0` off. Without an MTP head the
     * qwen35 draft call returns -1 and the step decodes plainly (n-gram drafts only when
     * PN_TT_SPEC is set explicitly). */
    /* Greedy speculation emits the verify rows' plain argmax: only when that IS the request's
     * greedy pick (no logit bias, presence/frequency penalty or grammar hook). */
    if (pn_qwen35moe_tt_greedy_exact_on_topn(params) &&
        model->forward == pn_qwen35moe_forward_tt)
        spec_k = getenv("PN_TT_SPEC") ? atoi(getenv("PN_TT_SPEC")) : 3;
    /* SPECULATIVE SAMPLING (temperature > 0): the same MTP drafts, each row's token DRAWN by
     * the engine's sampler (spec_sample_choose) — identical to plain sparse sampling, token
     * for token and draw for draw. Only where plain decode's rows are the sparse top-N too
     * (`PN_TT_SPARSE_LOGITS`); `PN_TT_SPEC_SAMPLE=0` off. No n-gram drafts here. */
    const bool spec_sampling = params->temperature > 0.0f && !params->tool_grammar &&
                               model->forward == pn_qwen35moe_forward_tt &&
                               !(getenv("PN_TT_SPEC_SAMPLE") &&
                                 atoi(getenv("PN_TT_SPEC_SAMPLE")) == 0) &&
                               pn_qwen35moe_tt_spec_can_sample(model);
    if (spec_sampling)
        spec_k = getenv("PN_TT_SPEC") ? atoi(getenv("PN_TT_SPEC")) : 3;
    const bool spec_ngram = getenv("PN_TT_SPEC") != NULL && !spec_sampling;
    if (spec_k > 15) spec_k = 15;
    pn_spec_sample_ctx spec_ctx = { .params = params, .n_vocab = model->config.n_vocab,
                                    .max_recent = max_recent };
    if (spec_sampling && spec_k > 0) {
        spec_ctx.row = malloc((size_t)model->config.n_vocab * sizeof(float));
        spec_ctx.recent = malloc((size_t)(max_recent > 0 ? max_recent : 1) * sizeof(uint32_t));
        if (!spec_ctx.row || !spec_ctx.recent)
            spec_k = 0;
        /* TRUE SPECULATIVE SAMPLING (opt-in, `PN_TT_SPEC_SAMPLE_RULE=1`/`2`; off = the
         * shared-randomness rule): drafts drawn from the MTP head's q, rows decided by min(1, p/q). */
        if (spec_k > 0 && pn_spec_rule_applies(params) && pn_spec_sample_rule_arm(&spec_ctx) == 0 &&
            eos_id >= 0)
            spec_ctx.no_draft[spec_ctx.n_no_draft++] = (uint32_t)eos_id;
    }
#else
    int spec_k = 0;
#endif
    uint32_t spec_pending[16];
    int spec_n_pending = 0, spec_at = 0;
    int hist_cap = n_tokens + max_tokens + 1, n_hist = 0;
    uint32_t *hist = spec_k > 0 ? malloc((size_t)hist_cap * sizeof(uint32_t)) : NULL;
    if (hist) { memcpy(hist, tokens, (size_t)n_tokens * sizeof(uint32_t)); n_hist = n_tokens; }
    else spec_k = 0;

    for (int i = 0; i < max_tokens; i++) {
        uint32_t next_token;
        if (spec_at < spec_n_pending) {
            next_token = spec_pending[spec_at++];
            goto spec_emit;
        }
        if (!logits) break;

        float *logit_data = (float *)logits->data;
        /* S0 (bit-exactness): `PN_LOGIT_HEX_DUMP=1` prints the top-8 (id, float bits) of every
         * step, so two arms compare to the BIT, not to a printed digit. */
        if (getenv("PN_LOGIT_HEX_DUMP")) {
            int nv = model->config.n_vocab, top[8], n = 0, v, t;

            for (v = 0; v < nv; v++) {
                if (n < 8 || logit_data[v] > logit_data[top[n - 1]]) {
                    int at = n < 8 ? n++ : 7;

                    top[at] = v;
                    for (t = at; t > 0 && logit_data[top[t]] > logit_data[top[t - 1]]; t--) {
                        int swap = top[t];

                        top[t] = top[t - 1];
                        top[t - 1] = swap;
                    }
                }
            }
            fprintf(stderr, "LOGITHEX step=%d", i);
            for (t = 0; t < n; t++) {
                uint32_t bits;

                memcpy(&bits, &logit_data[top[t]], sizeof(bits));
                fprintf(stderr, " %d:%08x", top[t], bits);
            }
            fprintf(stderr, "\n");
        }
        /* Bonsai debug: BONSAI_LOGIT_DUMP prints top-3 (id,logit,margin) per
         * decode step so the divergence-token margin vs the oracle is visible. */
        if (getenv("BONSAI_LOGIT_DUMP") && model->tokenizer) {
            int nv = model->config.n_vocab;
            int a = 0, b = 0, c = 0;
            for (int v = 1; v < nv; v++) {
                float x = logit_data[v];
                if (x > logit_data[a]) { c = b; b = a; a = v; }
                else if (x > logit_data[b]) { c = b; b = v; }
                else if (x > logit_data[c]) { c = v; }
            }
            int ta = a, tb = b, tc = c;
            char *pa = ft_decode(model->tokenizer, &ta, 1);
            char *pb = ft_decode(model->tokenizer, &tb, 1);
            char *pc = ft_decode(model->tokenizer, &tc, 1);
            fprintf(stderr, "LOGIT step=%d | [%d %.4f '%s'] [%d %.4f '%s'] "
                    "[%d %.4f '%s'] margin=%.4f\n",
                    i, a, logit_data[a], pa ? pa : "?",
                    b, logit_data[b], pb ? pb : "?",
                    c, logit_data[c], pc ? pc : "?",
                    logit_data[a] - logit_data[b]);
            free(pa); free(pb); free(pc);
        }
        pn_sampler_row_hint(logits); /* a backend's survivor list, this draw only */
        next_token = sample_token(logit_data, model->config.n_vocab,
                                  params, recent, n_recent);
        tm_free(logits);
        logits = NULL;
spec_emit:

        if (eos_id >= 0 && (int)next_token == eos_id) break;

        n_generated++;

        if (model->config.arch == PN_ARCH_RWKVX) {
            char *str = pn_rwkvx_decode(model->weights, &next_token, 1);
            if (str) { callback(str, user_data); free(str); }
        } else {
            int tok_id = (int)next_token;
            char *raw_str = ft_decode(model->tokenizer, &tok_id, 1);
            if (raw_str) {
                if (is_bpe) {
                    char *tok_str = gpt2_bytes_decode(raw_str);
                    free(raw_str);
                    if (tok_str) {
                        callback(tok_str, user_data);
                        free(tok_str);
                    }
                } else {
                    callback(raw_str, user_data);
                    free(raw_str);
                }
            }
        }

        if (n_recent < max_recent) {
            recent[n_recent++] = next_token;
        } else {
            memmove(recent, recent + 1, (max_recent - 1) * sizeof(uint32_t));
            recent[max_recent - 1] = next_token;
        }

        if (hist && n_hist < hist_cap) hist[n_hist++] = next_token;
        if (spec_at < spec_n_pending)
            continue; /* more of the last step's tokens to emit; no forward */
#ifdef USE_TT
        if (spec_k > 0) {
            uint32_t drafts[16];
            int nd;
            if (spec_ctx.row) {
                /* The window the cycle's draws (sampled drafts included) start from. */
                memcpy(spec_ctx.recent, recent, (size_t)n_recent * sizeof(uint32_t));
                spec_ctx.n_recent = n_recent;
                spec_ctx.row_index = 0;
                pn_spec_sample_begin_drafts(&spec_ctx);
            }
            nd = spec_ctx.q ? pn_qwen35moe_tt_mtp_draft_ex(model, state, next_token, spec_k,
                                                           drafts, pn_spec_sample_draft,
                                                           &spec_ctx)
                            : pn_qwen35moe_tt_mtp_draft(model, state, next_token, spec_k, drafts);
            if (nd < 1)
                nd = spec_ngram ? pn_ngram_draft(hist, n_hist, 4, 1, spec_k, drafts) : 0;
            pn_spec_sample_cut_drafts(&spec_ctx, nd);
            /* No draft: a one-row verify costs more than the plain decode step. */
            int ne = 0;
            if (nd > 0 && spec_ctx.row) {
                ne = pn_qwen35moe_tt_spec_step_ex(model, state, next_token, drafts, nd,
                                                  spec_pending, pn_spec_sample_choose, &spec_ctx);
            } else if (nd > 0) {
                ne = pn_qwen35moe_tt_spec_step(model, state, next_token, drafts, nd,
                                               spec_pending);
            }

            if (nd == 0) {
                /* fall through to the plain forward */
            } else if (ne > 0) {
                spec_n_pending = ne;
                spec_at = 0;
                continue;
            }
            else {
                fprintf(stderr, "engine: speculative step failed; plain decode from here\n");
                spec_k = 0;
                spec_n_pending = spec_at = 0;
            }
        }
#endif
        logits = model->forward(model, next_token, state);
    }

    if (logits) tm_free(logits);
    free(recent);
    free(hist);
#ifdef USE_TT
    pn_spec_rule_report(&spec_ctx, "cli");
    pn_spec_sample_rule_free(&spec_ctx);
    free(spec_ctx.row);
    free(spec_ctx.recent);
    params->dense_row = NULL;
    params->tail_counts = NULL;
    params->dense_ctx = NULL;
#endif

    return n_generated;
}

/* ============================================================
 * Lossless speculative decode (single sequence, greedy)
 *
 * The whole scheme rests on pn_llama_verify: one batched forward over the seed
 * plus K draft tokens returns, at each position i, the model's own logits for
 * the prefix ending at candidate i (row i is causal, so it is independent of the
 * candidates after it, and of K). The step below only ever emits argmax of those
 * logits, and only "accepts" a draft token when it already equals that argmax.
 * A draft can therefore never inject a token the real model would not have
 * produced: it can only save forward passes when it guesses right. That is the
 * lossless guarantee, and it holds for any draft (see test_spec_decode.c).
 * ============================================================ */

/* Greedy argmax with the SAME tie-break as sample_token's temperature<=0 path
 * (strict >, so the lowest index wins on a tie), keeping the emitted stream
 * bit-identical to plain greedy decode. */
static uint32_t spec_argmax(const float *logits, int n_vocab) {
    int best = 0;
    for (int i = 1; i < n_vocab; i++)
        if (logits[i] > logits[best]) best = i;
    return (uint32_t)best;
}

int pn_spec_decode_greedy(pn_model *model, pn_state *state, uint32_t seed,
                          const uint32_t *draft, int K, uint32_t *out) {
    if (!model || !state || !out || K < 0 || (K > 0 && !draft)) return -1;

    int n_vocab = model->config.n_vocab;
    int M = K + 1;                       /* seed + K drafts fed to the verify pass */
    uint32_t *toks = malloc((size_t)M * sizeof(uint32_t));
    float *logits = malloc((size_t)M * (size_t)n_vocab * sizeof(float));
    if (!toks || !logits) { free(toks); free(logits); return -1; }

    toks[0] = seed;
    for (int i = 0; i < K; i++) toks[1 + i] = draft[i];

    int base_pos = state->pos;           /* seed will occupy this position */
    if (pn_llama_verify(model, toks, NULL, M, state, logits) != 0) {
        free(toks); free(logits);
        return -1;                       /* unsupported model: caller falls back */
    }

    /* logits[i] is the prediction for the token AFTER candidate i, i.e. for the
     * slot that draft[i] occupies. Accept while the draft matches that argmax. */
    int n_emit = 0;
    int n_accept = 0;
    int done = 0;
    for (int i = 0; i < K && !done; i++) {
        uint32_t g = spec_argmax(logits + (size_t)i * n_vocab, n_vocab);
        out[n_emit++] = g;               /* the true greedy token for this slot */
        if (g == draft[i]) n_accept++;   /* draft matched: its K/V is valid */
        else done = 1;                   /* mismatch: g is the correction, stop */
    }
    if (!done) {
        /* Every draft token was accepted; take the free bonus after the last. */
        out[n_emit++] = spec_argmax(logits + (size_t)K * n_vocab, n_vocab);
    }

    free(toks); free(logits);

    /* Commit the seed (position base_pos) plus every accepted draft token; the
     * verify pass already wrote their K/V into the cache at these positions. The
     * final emitted token (correction or bonus) stays uncommitted and becomes
     * the next call's seed. */
    state->pos = base_pos + 1 + n_accept;
    return n_emit;
}

/* Lossless greedy speculative generation loop. See engine.h for the contract.
 *
 * The loop keeps the invariant that `seed` is the last generated token not yet
 * written to the cache; pn_spec_decode_greedy consumes it, commits the seed plus
 * accepted drafts, and returns the greedy continuation whose last element is the
 * next seed. Plain decode is the K=0 special case (one token per pass), so the
 * fallback path just stops calling draft_fn. */
int pn_spec_generate(pn_model *model, pn_state *state, uint32_t seed,
                     int max_tokens, int K, int stop_id,
                     pn_draft_fn draft_fn, void *draft_ud,
                     double min_accept_frac, int probe_cycles,
                     uint32_t *out, pn_spec_stats *stats) {
    if (!model || !state || !out || max_tokens <= 0) return -1;

    if (stats) { stats->verify_passes = 0; stats->tokens_emitted = 0;
                 stats->drafted = 0; stats->accepted = 0; stats->fell_back = 0; }

    /* Probe the verify path once (also the model-support gate). */
    if (K > 0) {
        float *probe = malloc((size_t)model->config.n_vocab * sizeof(float));
        int ok = probe && pn_llama_verify(model, &seed, NULL, 1, state, probe) == 0;
        free(probe);
        if (!ok) return -1;   /* unsupported: caller uses plain pn_describe path */
    }

    int max_k = (K > 0) ? K : 0;
    uint32_t *draft = max_k ? malloc((size_t)max_k * sizeof(uint32_t)) : NULL;
    uint32_t *emit  = malloc((size_t)(max_k + 1) * sizeof(uint32_t));
    if ((max_k && !draft) || !emit) { free(draft); free(emit); return -1; }

    int n = 0;
    long passes = 0, drafted = 0, accepted = 0;
    int use_spec = (K > 0 && draft_fn != NULL);
    int fell_back = 0;

    if ((int)seed == stop_id) { free(draft); free(emit);
        if (stats) *stats = (pn_spec_stats){0,0,0,0,0};
        return 0; }
    out[n++] = seed;

    while (n < max_tokens) {
        int kk = 0;
        if (use_spec) {
            kk = draft_fn(out, n, draft, max_k, draft_ud);
            if (kk < 0) kk = 0;
            if (kk > max_k) kk = max_k;
        }
        int em = pn_spec_decode_greedy(model, state, seed, draft, kk, emit);
        if (em < 0) { free(draft); free(emit); return -1; }
        passes++;
        drafted += kk;
        for (int i = 0; i < em && i < kk; i++) { if (emit[i] == draft[i]) accepted++; else break; }

        int done = 0;
        for (int j = 0; j < em; j++) {
            if ((int)emit[j] == stop_id) { done = 1; break; }
            out[n++] = emit[j];
            if (n >= max_tokens) { done = 1; break; }
        }
        seed = emit[em - 1];
        if (done) break;

        /* Per-page fallback: if early acceptance is poor, stop drafting. */
        if (use_spec && !fell_back && passes >= probe_cycles && drafted > 0 &&
            (double)accepted / (double)drafted < min_accept_frac) {
            use_spec = 0;
            fell_back = 1;
        }
    }

    free(draft); free(emit);
    if (stats) {
        stats->verify_passes = passes;
        stats->tokens_emitted = n;
        stats->drafted = drafted;
        stats->accepted = accepted;
        stats->fell_back = fell_back;
    }
    return n;
}
