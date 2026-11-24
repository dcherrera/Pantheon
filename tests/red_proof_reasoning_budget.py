#!/usr/bin/env python3
# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026 David (Corey) Herrera
"""Red proof for the reasoning-plan tests (answer reserve + line-end close).

Applies one mutation at a time, deletes the test binaries (macOS make 3.81 has
1 s mtime granularity), rebuilds, runs the tests and counts failed checks, then
restores the source.  Every mutation must turn at least one test red.  Run from
the repo root:  python3 tests/red_proof_reasoning_budget.py [make args...]
(e.g. FOUNDRY=../foundry).
"""
import os
import re
import subprocess
import sys

PLAN = "src/serve/reasoning_budget.h"
CATALOG = "src/local/catalog_manifest.c"
LOADED = "src/local/loaded_set.c"
TESTS = ["tests/test_reasoning_budget", "tests/test_local_catalog", "tests/test_local_loaded_set"]

# (name, file, old, new)
MUTATIONS = [
    ("no reserve clamp (the old cap)", PLAN,
     "        if (hard > max_tokens - reserve)\n", "        if (0)\n"),
    ("reserve default is 0", PLAN,
     "        reserve = PN_REASONING_RESERVE_DEFAULT;\n", "        reserve = 0;\n"),
    ("reserve not capped at max_tokens / 2", PLAN,
     "        if (reserve > max_tokens / 2)\n", "        if (0)\n"),
    ("no close window (exact cut)", PLAN,
     "    p.soft = hard - win;\n", "    p.soft = hard;\n"),
    ("window ignores the line end", PLAN,
     "(toks >= p->soft && line_end)", "(toks >= p->soft)"),
    ("line end never seen", PLAN,
     "return piece && strchr(piece, '\\n') != NULL;", "return false;"),
    ("drafts run into the window", PLAN,
     "    return p->soft - toks - 1;\n", "    return p->hard - toks - 1;\n"),
    ("grace env ignored", PLAN,
     "    const char *s = getenv(\"PN_REASONING_LINE_GRACE\");\n",
     "    const char *s = NULL;\n"),
    ("catalog drops reasoning_reserve", CATALOG,
     "    p->reasoning_reserve = (int)get_num(load, \"reasoning_reserve\", 0);\n", ""),
    ("catalog accepts a negative reserve", CATALOG,
     "p->reasoning_reserve < 0 ||", "0 ||"),
    ("loaded set does not persist the reserve", LOADED,
     "        jw_printf(jw, \"\\\"reasoning_reserve\\\":%d,\", e->reasoning_reserve);\n", ""),
]


def run_tests(make_args):
    for t in TESTS:
        if os.path.exists(t):
            os.remove(t)
    b = subprocess.run(["make"] + make_args + TESTS, capture_output=True, text=True)
    if b.returncode != 0:
        return "build failed"
    reds = []
    for t in TESTS:
        r = subprocess.run(["./" + t], capture_output=True, text=True)
        if r.returncode != 0:
            n = len(re.findall(r"FAIL", r.stdout + r.stderr))
            reds.append("%s (%d FAIL)" % (os.path.basename(t), n))
    return ", ".join(reds)


def main():
    make_args = sys.argv[1:]
    base = run_tests(make_args)
    if base:
        print("baseline not green: " + base)
        return 1
    print("baseline: all green")
    bad = 0
    for name, path, old, new in MUTATIONS:
        src = open(path).read()
        if src.count(old) != 1:
            print("SKIP (anchor not unique/found): " + name)
            bad += 1
            continue
        open(path, "w").write(src.replace(old, new))
        try:
            reds = run_tests(make_args)
        finally:
            open(path, "w").write(src)
        print(("RED   " if reds else "GREEN ") + name + (": " + reds if reds else ""))
        if not reds:
            bad += 1
    run_tests(make_args)
    print("%d mutations, %d not caught" % (len(MUTATIONS), bad))
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
