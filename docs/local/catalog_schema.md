# Pantheon Local catalog — `catalog.json` schema (schema 1)

The box parses this with `src/local/catalog_manifest.c` and refuses the whole
manifest on the first bad field. `pantheon local check-catalog catalog.json` runs the
same parser.

```json
{
  "schema": 1,
  "version": 12,
  "published": "2026-10-26",
  "models": [{
    "id": "qwen3.6-35b-a3b-q3km",
    "name": "Qwen3.6 35B-A3B",
    "family": "qwen3.6",
    "description": "One line for the app.",
    "licence": { "id": "apache-2.0", "url": "https://huggingface.co/..." },
    "capabilities": ["chat", "tools", "reasoning"],
    "files": [{
      "role": "weights",
      "hf_repo": "TeamIDE/Qwen3.6-35B-A3B-GGUF",
      "revision": "0123456789abcdef0123456789abcdef01234567",
      "filename": "Qwen3.6-35B-A3B-Q3_K_M.gguf",
      "size": 16000000000,
      "sha256": "<64 lowercase hex>"
    }],
    "profiles": [{
      "id": "k80-pair",
      "rank": 1,
      "hardware": { "unit_type": "k80-gpu", "count": 2, "same_card": true,
                    "min_mem_per_unit": 10000000000, "min_host_ram": 7000000000 },
      "load": { "ctx_cap": 16384, "max_sessions": 1, "max_concurrent": 0,
                "reasoning_budget": 4096, "mtp": false },
      "options": { "kv_type": "fp32", "ctx_min": 8192, "ctx_max": 16384, "sessions_max": 2,
                   "mem": { "fixed": 11755585536, "ctx_per_token": 20480,
                            "per_session": 35651584, "kv_per_token": 20480 } },
      "sampling": { "temperature": 0.6, "top_p": 0.95, "top_k": 20, "min_p": 0,
                    "presence_penalty": 0 },
      "measured": { "decode_tok_s": { "8k": 12.1, "14k": 11.0, "32k": 9.2 },
                    "ttft": { "prompt_tokens": 8000, "seconds": 40 },
                    "context_window": 32768 }
    }]
  }]
}
```

## Fields and rules

| Field | Rule |
|---|---|
| `schema` | `1` |
| `version` | integer ≥ 1. The box refuses a version lower than the last one it accepted (anti-rollback). Bump it on every change. |
| `models` | ≤ 32 |
| `models[].id` | `[a-z0-9._-]`, < 64, not starting with `.` or `-`; unique. Also the served model name. |
| `name`, `licence.id` | required text |
| `family`, `capabilities[]` | `[a-z0-9._-]` |
| `files` | 1–4. `"role": "weights"` is the file a load opens; others `"extra"`. Every profile's unit type must need exactly one weights file: one untagged, or one per unit type (`unit_types`), e.g. Qwen3.6 Q8_0's K80 GGUF `["k80-gpu"]` and its p150 metadata GGUF `["p150"]` |
| `files[].hf_repo` | `<org>/<repo>`, `<org>` in `PN_CATALOG_HF_ORGS` (`src/local/catalog_keys.h`) |
| `files[].revision` | a pinned 40-hex commit, never a branch |
| `files[].filename` | `[A-Za-z0-9._-]`, no `/`, not starting with `.` |
| `files[].size` | bytes, 1 … 1 TB |
| `files[].sha256` | 64 lowercase hex; the file is refused if it does not match |
| `files[].unit_types` | optional: the unit types that need the file (`k80-gpu`, `p150`, `cpu`), e.g. `["p150"]` for a TT-only `.fttw`. Absent = every profile needs it. An install, the `installed` check and the `size` of a catalog row, and the file a load opens, use only the files the profile's `hardware.unit_type` needs (an install without `profile_id`: the unit types the box has). Each type must be one some profile of the model uses. Older boxes ignore the key and install every file; a model with two weights files is refused whole by boxes built before per-unit-type weights. |
| `files[].ingot` | optional: a compressed download of the file (foundry `.ingot`, v2), `{ "filename", "size", "hf_repo"?, "revision"?, "base_sha256"? }`. `filename` and `size` (bytes of the `.ingot`, 1 … 1 TB) as for a file; `hf_repo` and `revision` default to the file's and follow the same rules; `base_sha256` (64 lowercase hex) makes it a delta against that file. The file's own `size` and `sha256` stay its identity: the decoded file must match them. See **Ingot downloads**. |
| `profiles` | 1–8 per model |
| `profiles[].rank` | lower = better experience, shown first (default 100) |
| `hardware.unit_type` | `k80-gpu`, `p150` or `cpu` |
| `hardware.count` | 1–8 units; `same_card` puts them on one card |
| `hardware.per_card` | optional: whole cards, this many units on each (`count` 4, `per_card` 2 = two K80 boards, both GPUs of each; idle boards first, units in card order). Must divide `count`; not with `same_card`. |
| `hardware.min_mem_per_unit`, `min_host_ram` | bytes, 0 = no minimum |
| `load.*` | as `pantheon serve` `--ctx-cap`, `--max-sessions`, `--max-concurrent`, `--reasoning-budget`, `--mtp`. On a p150, `mtp: false` loads with no MTP head (`PN_TT_SPEC=0` for that load); `mtp: true` keeps the TT default |
| `load.reasoning_reserve` | optional: tokens of `max_tokens` kept for the answer after a forced close (`--reasoning-reserve`). Absent = the serve default, 3072. Older boxes ignore it |
| `load.mtp_drafts` | optional, p150 only, with `mtp: true`: MTP drafts a step, 1–15 (`PN_TT_SPEC=<n>` for that load; the load reserves that many verify rows). Absent = the TT default. Older boxes ignore it |
| `options` | optional (load options, Phase L). Without it the profile loads with its fixed `load` settings. |
| `options.kv_type` | `q4`, `fp32` or `bf16`: the KV the numbers were measured with, and the KV the box loads with. `q4` (TurboQuant Q4, the K80 default) and `fp32` are explicit: the serve never switches them (under host memory pressure an `fp32` load is refused with a reason, not made Q4). `bf16` (p150) and no `options` = the serve default. A p150 profile may also name a block-float KV: `bfp8`, `bfp4`, `k8v4`, `k16v4`, `k4v16`, `k16v8` (`PN_TT_KV_TYPE=<value>` for that load). Boxes built before these values refuse the whole catalog and keep the version they accepted. |
| `options.ctx_min`, `ctx_max` | context per user the user may pick; the box offers power-of-two steps from 4096 inside the range, plus `load.ctx_cap` |
| `options.sessions_max` | users at once, 1–64; the box offers 1, 2, 4, … up to it, plus `load.max_sessions` |
| `options.mem` | the measured device memory model, bytes on the heaviest unit: `need = fixed + ctx × ctx_per_token + users × (per_session + ctx × kv_per_token)`. `fixed` and `kv_per_token` are required. Measure with `tests/local/mem_model_measure.sh`. |
| `load.ctx_cap`, `load.max_sessions` | with `options`: the recommended choice; must lie inside the ranges |
| `sampling.*` | per-model defaults; temperature [0,5], top_p (0,1], top_k [0,1000], **min_p 0 only** (no sampler support yet), presence_penalty [-2,2] |
| `measured` | required: `decode_tok_s.8k` > 0, `ttft.prompt_tokens` > 0, `ttft.seconds` > 0, `context_window` > 0 (a profile ships only with measured numbers) |

**KV choice.** A model offers both KV formats on the same hardware as two profiles
(e.g. `k80-pair` = `q4` + MTP, rank 1; `k80-pair-fp32` = `fp32` + plain decode, rank 2),
each with its own memory model and measured numbers. A client shows one entry with a
"context precision" choice when a model has several profiles that differ only in
`kv_type` on the same `hardware` (type, count, same_card); the best-ranked is preselected.

**Load options on the box.** `GET /local/catalog` adds per entry
`"options":{"kv_type","ctx_min","ctx_max","sessions_max","choices":[{"ctx_cap","max_sessions","need_per_unit","recommended"}]}`:
only the choices that fit the units free now (each unit needs `need + 256 MiB`
of device memory), one recommended (the profile's own if it fits, else the
biggest fitting one below it). `POST /local/models/load` takes `ctx_cap` and
`max_sessions` (absent = the recommended choice) and refuses anything else
with a plain reason. Loading a model that is already loaded adds a copy
(replica) on free units under the same name.

Download URL: `https://huggingface.co/<hf_repo>/resolve/<revision>/<filename>`.
Store name: `<model id>-<first 12 hex of sha256>.<extension>`.

**Ingot downloads.** A file with `ingot` is fetched as its `.ingot`
(`https://huggingface.co/<ingot.hf_repo>/resolve/<ingot.revision>/<ingot.filename>`) and
decoded while it downloads straight into the store's `.part` file; nothing is held in memory
and the `.ingot` is never stored. The decoder checks every tensor and the rebuilt file's
SHA-256, which must equal `sha256`, and the size must equal `size`, or the part file is
deleted. A cut connection resumes the `.ingot` where it stopped. A delta
(`base_sha256`) is decoded only when a catalog file with that `sha256` is installed in the
store; otherwise, or when that base is rejected, the box downloads the whole file. A box that
already holds a partial `.part` resumes the whole file when that moves fewer bytes than
fetching the `.ingot` again. Older boxes ignore the key.

**Companion extras.** An `extra` whose filename is `<weights stem>.<rest>` (stem = the
weights filename minus its last extension) is linked at load as
`<model id>-<weights sha256[:12]>.<rest>` → its verified store file, where the backend looks
for it. Example: weights `Ornith-1.5-9B-Q8_0.gguf` + extra `Ornith-1.5-9B-Q8_0.bfp8.mtp.fttw`
→ the TT backend finds the MTP head beside its converted `<…>.bfp8.fttw`. A new catalog with
a different extra re-points the link.

**Card metadata GGUF (p150).** A model too big to ship as a GGUF beside its card artifact
(Qwen3.6 Q8_0) ships, for `p150`, a metadata GGUF as its weights (keys and small tensors,
foundry `ft_tt_gguf_slim`; no `token_embd.weight`) plus the artifact as a companion extra:
weights `Qwen_Qwen3.6-35B-A3B-Q8_0.meta.gguf`, extras `….meta.moe_bfp6.fttw` (trunk) and
`….meta.moe_bfp6.mtp.fttw` (MTP head). The load opens the metadata GGUF; `pn_load` attaches the
one `<store stem>.<rung>.fttw` beside it (`pn_gguf_attach_card_artifact`, `<rung>` without a
dot) and the TT backend uploads that artifact as-is (no conversion); the MTP head is found
beside the trunk as before.

`GET /local/catalog` reports a row's `model.size` as the bytes its profile's install
downloads (the files its unit type needs).
