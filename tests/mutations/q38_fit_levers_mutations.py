#!/usr/bin/env python3
"""Mutation reds for the fit levers (qwen35moe_tt_fit_plan.c) and the weight view
(pn_tt_weight_view.c): each mutation must turn its host test red. Restores every file from git.
Run from the repo root: python3 tests/mutations/q38_fit_levers_mutations.py"""
import re
import subprocess
import sys

P = 'src/models/qwen35moe_tt_fit_plan.c'
H = 'src/models/qwen35moe_tt_fit_plan.h'
V = 'src/accel/tt/pn_tt_weight_view.c'
TESTS = {'levers': 'tests/test_tt_q38_fit_levers', 'view': 'tests/test_tt_weight_view'}
MUTS = [
    ("allowance 0", H, "#define PN_QWEN35_TT_FIT_LEVER_ALLOWANCE (60ull * 1024ull * 1024ull)",
     "#define PN_QWEN35_TT_FIT_LEVER_ALLOWANCE (0ull)", 'levers'),
    ("single allowance as mirrored", H,
     "#define PN_QWEN35_TT_FIT_LEVER_ALLOWANCE_SINGLE (26ull * 1024ull * 1024ull)",
     "#define PN_QWEN35_TT_FIT_LEVER_ALLOWANCE_SINGLE (60ull * 1024ull * 1024ull)", 'levers'),
    ("single allowance for mirrored sets", P,
     "    single = s->tall_rows && s->tall_rows <= 32u && !s->outproj_mirrors;",
     "    single = s->tall_rows && s->tall_rows <= 32u;", 'levers'),
    ("compact verify not taken off", P,
     "    if (s->mtp_bytes && s->mtp_rows > 0u && s->gdn_on_card)\n        even -= set;", "", 'levers'),
    ("checkpoint dropped", P, "    if (checkpoint)\n        even += set;\n    if (floor->kv)\n        kv_bank = pn_qwen35_tt_fit_kv_bank_bytes(s, kv_layers, banks, true);\n    if (kv_bank == 0u)\n        even += floor->kv + head_kv;\n    return (even + banks - 1u) / banks + kv_bank;\n}\n\nint64_t",
     "    if (floor->kv)\n        kv_bank = pn_qwen35_tt_fit_kv_bank_bytes(s, kv_layers, banks, true);\n    if (kv_bank == 0u)\n        even += floor->kv + head_kv;\n    return (even + banks - 1u) / banks + kv_bank;\n}\n\nint64_t", 'levers'),
    ("conv share not taken off", P,
     "        even -= (uint64_t)s->n_ssm * fit_tiles(s->conv_dim) * s->conv_taps * FIT_FP32_TILE_BYTES;\n    /* C:",
     "        even -= 0u;\n    /* C:", 'levers'),
    ("KV spread even", P, "        kv_bank = pn_qwen35_tt_fit_kv_bank_bytes(s, kv_layers, banks, true);\n    if (kv_bank == 0u)\n        even += floor->kv + head_kv;\n    return (even + banks - 1u) / banks + kv_bank;\n}\n\nint64_t pn_qwen35_tt_fit_policy_bank_low",
     "        kv_bank = 0u;\n    if (kv_bank == 0u)\n        even += floor->kv + head_kv;\n    return (even + banks - 1u) / banks + kv_bank;\n}\n\nint64_t pn_qwen35_tt_fit_policy_bank_low", 'levers'),
    ("tall step before copies", P,
     "} fit_tall_steps[] = { { 0u, true }, { 32u, true }, { 32u, false }, { 16u, false } };",
     "} fit_tall_steps[] = { { 32u, true }, { 0u, true }, { 32u, false }, { 16u, false } };", 'levers'),
    ("view before copies", P, "        for (uint32_t v = 0; v < 2u; v++) {",
     "        for (uint32_t v = 1; v < 2u; v++) {", 'levers'),
    ("pins ignored", P, "                if (pins->gdnc_replicas && r != shape->gdnc_replicas)",
     "                if (0)", 'levers'),
    ("draft copy not dropped by V", P,
     "        out->mtp = s->mtp_bytes - (s->draft_view ? s->mtp_draft_bytes : 0u);",
     "        out->mtp = s->mtp_bytes;", 'levers'),
    ("K taken where the ladder fits", P, "    if (!fits && pins->kv_long_k_tile && pins->kv_long_v_tile &&",
     "    if (pins->kv_long_k_tile && pins->kv_long_v_tile &&", 'levers'),
    ("K: long tiles not applied", P, "        s.kv_k_tile_bytes = pins->kv_long_k_tile;\n        s.kv_v_tile_bytes = pins->kv_long_v_tile;\n",
     "", 'levers'),
    ("K: not-smaller long type taken", P,
     "        (uint64_t)pins->kv_long_k_tile + pins->kv_long_v_tile <\n            (uint64_t)shape->kv_k_tile_bytes + shape->kv_v_tile_bytes) {",
     "        true) {", 'levers'),
    ("view: no column offset in panel", V,
     "                source->shard_address[s] + (column - source->shard_column_first[s]) *\n                                               source->tile_rows * source->tile_bytes;",
     "                source->shard_address[s];", 'view'),
    ("view: output offset 0", V, "            out->shard_column_first[piece] = view_column;",
     "            out->shard_column_first[piece] = 0u;", 'view'),
    ("view: no spare members", V, "    for (spare = members - segments; spare > 0u; spare--) {",
     "    for (spare = 0u; spare > 0u; spare--) {", 'view'),
    ("view: door of piece 0", V, "            out->shard_x[piece] = source->shard_x[s];",
     "            out->shard_x[piece] = source->shard_x[0];", 'view'),
]


def run(test):
    subprocess.run(['rm', '-f', test])
    r = subprocess.run(['make', '-s', test], capture_output=True, text=True)
    if r.returncode != 0:
        return None, 'BUILD FAIL'
    out = subprocess.run(['./' + test], capture_output=True, text=True).stdout
    m = re.search(r'(\d+) checks, (\d+) failures', out)
    return (int(m.group(2)) if m else None), (m.group(0) if m else out[-200:])


greens = 0
for name, f, a, b, t in MUTS:
    s = open(f).read()
    if s.count(a) != 1:
        print(f"{name:36s} PATTERN NOT FOUND")
        greens += 1
        continue
    open(f, 'w').write(s.replace(a, b))
    try:
        fails, line = run(TESTS[t])
    finally:
        subprocess.run(['git', 'checkout', '--', f])
    red = fails is not None and fails > 0
    greens += 0 if red else 1
    print(f"{name:36s} {'RED' if red else 'GREEN (mutation survived)'}  {line}")
for t in TESTS.values():
    fails, line = run(t)
    print(f"restored {t}: {line}")
    greens += 1 if fails else 0
sys.exit(1 if greens else 0)
