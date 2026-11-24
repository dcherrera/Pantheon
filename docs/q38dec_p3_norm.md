# Q3.8 decode P3 — faster norms, bit-exact

Qwen3.8-27B on the p150, plain decode token 132 ms (p7_2 timeline, run4).

## Bound

| op | per token | each | where the time goes |
|---|---|---|---|
| RMSNorm (`reduce` + member, Wt 160, 2-way column split) | 129 × 65.8 µs = 8.5 ms | 65.8 µs | read done 16.0 µs (gamma 160 KB *then* input 320 KB, nothing overlaps); map+reduce 16.3 µs, starting only once the whole row has landed; final 80 tiles 27.8 µs (347 ns/tile) |
| head norm (`reduce`/`reduce_fp32`, Wt 4/8, 4 members) | 128 ops = 5.1 ms | 34 / 54 µs | reader-bound (input 29.5 of 34.3 µs): CB_IN holds one head, so every head's DRAM latency serialises against compute; full tiles read |

The final chain is gamma then scale through 1-page scratch CBs: the unpacker waits a full unpack→math→pack round trip between a tile's two steps, and re-inits both steps every tile.

## Change (`PN_TT_NORM_FAST=1`, default; `0` = the pre-P3 descriptors and placement)

Kernel: `resident_reduce_{reader,compute}.c`, `resident_reduce_protocol.h` (only the two reduce programs include them). Word 0 of a reduce block (the unused FUNCTION slot) is now a schedule word; 0 = the pre-P3 schedule.

1. **Streamed input** (bit 8): up to 4 blocks in flight on NoC transaction ids 1–4 (matmul M4 pattern); a block is issued only into pages free beyond the ones in flight, and the reader never blocks holding an unpushed block (`ftr_stream_action()`, modelled on the host against a consumer for deadlock/overwrite). The compute thread's x² map waits block by block. Model norm depth 16.
2. **Half reads** (bit 9): with the half-tile bit set and a norm-shaped program (every pass maps x², no center/beta, chain scales cb_in), input and gamma land faces 0–1 only (1 KB/tile bf16, 2 KB fp32).
3. **gamma after row 0**: overlaps the map/reduce; compute waits for gamma just before its first chain step.
4. **Chunked chain** (bits 0–7 = K): step 1 for K tiles, then step 2 for K; inits once per run; scratch CBs and cb_out K pages (model K 16, head K = Wt). A scratch tile is read at index 0 and popped at once, so a run never addresses past a wrap. The reader refuses K > scratch pages (the one way a run wedges PACK).
5. **Head norm**: CB_IN 4 heads deep (reader runs 4 heads ahead; in-place callers stay safe — reads are only ever ahead of writes), the same streamed read, half tile + half reads in decode (`PN_TT_HEAD_NORM_HALF` 0/1/2, gated like `PN_TT_REDUCE_HALF`).

Arena (1320 KB): model norm +44 pages / 88 KB (Wt ceiling 217 → 202, Qwen3.8 needs 160); head norm fp32 Wt 8 = 77 pages / 308 KB. A schedule that does not fit falls back to pre-P3, logged.

## Why it is bit-exact

Only scheduling moves. Every tile sees the same ops, operands, order and bf16/Tf32 pack points; the row sum is the same reduce over the same tiles in the same order; repeated eltwise ops after one init is the shipped eltwise pattern. Half reads leave faces 2–3 of cb_in/gamma stale, and half-tile programs never unpack them (unpack MOP is 1×2 faces; checked per member in `test_reduce_schedule`). Head norm with the half tile: output rows 0–15 bit-identical, rows 16–31 zero instead of normed padding (same contract the model norm's M3 ships with).

## Expected

| op | before | after | per token |
|---|---|---|---|
| RMSNorm | 65.8 µs | ~31–35 µs (read hidden under map; final ≈ writer floor 179 ns × 80) | −4.0 to −4.5 ms |
| head norm | 34 / 54 µs | ~15 / 25 µs | −2.5 to −3 ms |

≈ 7 ms/token (132 → ~125 ms). Writer per-tile flush is the next floor (reduce-only writer: follow-up).

## Card checks

md5/logits gates at `PN_TT_NORM_FAST=1` vs `=0`; `PN_TT_RED=norm_fast_face0` must differ; `PN_TT_HEAD_NORM_HALF=0` isolates the head-norm half tile; timeline: reduce and head-norm µs.
