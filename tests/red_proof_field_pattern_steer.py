#!/usr/bin/env python3
# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026 David (Corey) Herrera
"""Red proof for the field_patterns steer tests.

Applies one mutation at a time to the sources, deletes the test binaries (macOS
make 3.81 has 1 s mtime granularity), rebuilds, runs the tests and counts failed
checks, then restores the source.  Every mutation must turn at least one check
red.  The first mutation is the sampler without the steer, i.e. the code before
this change.  Run from the repo root (FOUNDRY as for `make test`):
    FOUNDRY=... python3 tests/red_proof_field_pattern_steer.py
"""
import re
import subprocess
import sys

FP = "src/serve/field_pattern.c"
XML = "src/serve/tool_grammar.c"
JSON = "src/serve/tool_grammar_json.c"
ENGINE = "src/core/engine.c"
TESTS = ["tests/test_field_pattern", "tests/test_field_pattern_steer",
         "tests/test_tool_grammar_field_patterns", "tests/test_tool_grammar_json_field_patterns"]

# (name, file, old, new)
MUTATIONS = [
    ("sampler: never steers (before this change)", ENGINE,
     "        if (tg->steer)\n            grammar_steer_top_token(logits, n_vocab, tg, TG_MASK);\n", ""),
    ("sampler: steers on an allowed top token too", ENGINE,
     "    if (txt && *txt && !tg->allowed(tg->st, txt))", "    if (txt && *txt)"),
    ("sampler: takes the lowest logit, not the top", ENGINE,
     "(best < 0 || logits[i] > logits[best])", "(best < 0 || logits[i] < logits[best])"),
    ("pattern: a non-unique next byte is still forced", FP,
     "            if (forced >= 0) { forced = -2; break; }", "            if (forced >= 0) continue;"),
    ("pattern: no distance table", FP,
     "        compute_distances(p);\n", ""),
    ("pattern: any closer-or-equal byte counts", FP,
     "set_distance(p, &next) != d - 1", "set_distance(p, &next) > d"),
    ("xml: whitespace close not a close", XML,
     "    if (tg_is_ws(c)) return before->val_term == 0;", "    if (tg_is_ws(c)) return false;"),
    ("xml: '</parameter>' close not a close", XML,
     "    return c == '>' && before->val_term == (int)sizeof(LIT_VALUE_CLOSE) - 2;",
     "    return false;"),
    ("xml: steer literal not enforced", XML,
     "        if (c != st->steer_lit[st->steer_pos]) { st->phase = TG_DEAD; return; }\n", ""),
    ("xml: empty value steers", XML,
     "    if (!st->pat_begun || st->val_term > 0 ||", "    if (st->val_term > 0 ||"),
    ("xml: value mid-'</parameter>' steers", XML,
     "    if (!st->pat_begun || st->val_term > 0 ||", "    if (!st->pat_begun ||"),
    ("xml: steers again while one is pending", XML,
     "st->val_term > 0 || st->steer_pos < st->steer_len) return false;",
     "st->val_term > 0) return false;"),
    ("json: quote close not a close", JSON,
     "            refused = *p == '\"' && in_patterned_string(&before);",
     "            refused = false;"),
    ("json: empty value steers", JSON,
     "    if (!st->val_content || st->steer_pos < st->steer_len) return false;",
     "    if (st->steer_pos < st->steer_len) return false;"),
    ("json: steer literal not enforced", JSON,
     "        if (c != st->steer_lit[st->steer_pos]) { st->phase = TGJ_DEAD; return; }\n", ""),
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
    build = subprocess.run(["make", "USE_BLAS=1"] + TESTS, capture_output=True, text=True)
    if build.returncode != 0:
        return None
    return sum(failed_checks(subprocess.run(["./" + t], capture_output=True, text=True).stdout)
               for t in TESTS)


def main():
    base = run_tests()
    print(f"baseline: {base} failed")
    if base != 0:
        sys.exit("baseline must be green")
    survivors = 0
    for name, path, old, new in MUTATIONS:
        src = open(path).read()
        if src.count(old) != 1:
            sys.exit(f"mutation anchor not unique in {path}: {name}")
        open(path, "w").write(src.replace(old, new))
        try:
            failed = run_tests()
        finally:
            open(path, "w").write(src)
        verdict = "BUILD FAIL" if failed is None else f"{failed} failed"
        print(f"  {name:48s} {verdict}", flush=True)
        if failed == 0:
            survivors += 1
    print(f"restored; final: {run_tests()} failed; survivors: {survivors}")
    sys.exit(1 if survivors else 0)


if __name__ == "__main__":
    main()
