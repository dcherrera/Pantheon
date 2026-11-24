#!/usr/bin/env python3
# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026 David (Corey) Herrera
"""Red proof for tests/test_tool_grammar_forced_close_call.c.

Applies one mutation at a time (each a list of (file, old, new) replacements),
deletes the test binaries (macOS make 3.81 has 1 s mtime granularity), rebuilds,
runs the tests and counts failed checks, then restores the sources.  Every
mutation must turn at least one check red.  The first mutation is the code
before this change.  Run from the repo root (FOUNDRY as for `make test`):
    FOUNDRY=... python3 tests/red_proof_forced_close_call.py
"""
import os
import re
import subprocess
import sys

XML = "src/serve/tool_grammar.c"
JSON = "src/serve/tool_grammar_json.c"
TESTS = ["tests/test_tool_grammar_forced_close_call", "tests/test_tool_grammar",
         "tests/test_tool_grammar_json", "tests/test_tool_grammar_reasoned",
         "tests/test_tool_grammar_json_reasoned", "tests/test_frame_capture",
         "tests/test_tool_grammar_no_repeat_keys"]

XML_WINDOW = (XML, "    if (st->think_gated) {\n        st->landing_chars = 0;",
              "    if (0) {\n        st->landing_chars = 0;")
XML_WS = (XML, "            if (st->opener_ws >= TG_OPENER_WS_MAX) { st->phase = TG_DEAD; return; }\n",
          "")
JSON_WS = (JSON, "        if (st->opener_ws >= TGJ_OPENER_WS_MAX) { st->phase = TGJ_DEAD; return; }\n",
           "")

MUTATIONS = [
    ("before this change", [XML_WINDOW, XML_WS, JSON_WS]),
    ("xml: required walk keeps the landing window", [XML_WINDOW]),
    ("xml: unbounded whitespace before the opener", [XML_WS]),
    ("json: unbounded whitespace before the opener", [JSON_WS]),
    ("xml: whitespace cap one too many", [(XML, "st->opener_ws >= TG_OPENER_WS_MAX",
                                           "st->opener_ws > TG_OPENER_WS_MAX")]),
    ("json: whitespace cap one too many", [(JSON, "st->opener_ws >= TGJ_OPENER_WS_MAX",
                                            "st->opener_ws > TGJ_OPENER_WS_MAX")]),
    ("xml: cap applies in AUTO too", [(XML, "        if (forced && st->free_open == 0 &&\n"
                                            "            (c == '\\n'",
                                       "        if ((forced || st->mode == TG_MODE_AUTO) && st->free_open == 0 &&\n"
                                       "            (c == '\\n'")]),
    ("xml: window dropped for every forced walk", [(XML, "    if (st->think_gated) {\n        st->landing_chars = 0;",
                                                    "    if (1) {\n        st->landing_chars = 0;")]),
]


def failed_checks(out):
    """Failed checks in one test's output (either report style); 1 when unreadable."""
    out = re.sub(r"\x1b\[[0-9;]*m", "", out)
    m = re.search(r"(\d+) failed", out)
    if m:
        return int(m.group(1))
    m = re.search(r"(\d+)/(\d+) checks passed", out)
    return int(m.group(2)) - int(m.group(1)) if m else 1


def run_tests():
    """Rebuild the tests from scratch; return failed checks, or None when the build fails."""
    subprocess.run(["rm", "-f"] + TESTS)
    build = subprocess.run(["make", "USE_BLAS=1"] + TESTS, capture_output=True, text=True,
                           env=os.environ)
    if build.returncode != 0:
        return None
    return sum(failed_checks(subprocess.run(["./" + t], capture_output=True, text=True,
                                            errors="replace").stdout)
               for t in TESTS)


def main():
    base = run_tests()
    print(f"baseline: {base} failed")
    if base != 0:
        sys.exit("baseline must be green")
    survivors = 0
    for name, edits in MUTATIONS:
        saved = {}
        for path, old, new in edits:
            src = saved.setdefault(path, open(path).read())
            cur = open(path).read()
            if cur.count(old) != 1:
                for p, s in saved.items():
                    open(p, "w").write(s)
                sys.exit(f"mutation anchor not unique in {path}: {name}")
            open(path, "w").write(cur.replace(old, new))
        try:
            failed = run_tests()
        finally:
            for p, s in saved.items():
                open(p, "w").write(s)
        verdict = "BUILD FAIL" if failed is None else f"{failed} failed"
        print(f"  {name:50s} {verdict}", flush=True)
        if not failed:
            survivors += 1
    print(f"restored; final: {run_tests()} failed; survivors: {survivors}")
    sys.exit(1 if survivors else 0)


if __name__ == "__main__":
    main()
