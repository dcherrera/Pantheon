#!/bin/sh
# Mutation check for test_spec_depth: each fix reverted alone must fail the test.
# Run from the repo root: sh tests/mut/run_spec_depth_mutations.sh
set -u
out=tests/mut/build; mkdir -p $out
mut() {
    name=$1; shift
    python3 - "$@" <<'PY' > $out/spec_depth_mut.c
import sys
s = open('src/core/spec_depth.c').read()
old, new = sys.argv[1], sys.argv[2]
assert s.count(old) == 1, old
sys.stdout.write(s.replace(old, new))
PY
    cp src/core/spec_depth.h $out/
    cc -O2 -I$out -o $out/t tests/test_spec_depth.c $out/spec_depth_mut.c -lm 2>/dev/null || { echo "$name: BUILD FAIL"; return; }
    if $out/t >$out/$name.log 2>&1; then echo "$name: SURVIVED (test passed)"; else echo "$name: killed ($(grep -c FAIL $out/$name.log) fails)"; fi
}
mut heat 'double heat = c->plain_seen ? c->plain_ratio : c->ratio;' 'double heat = c->plain_seen || !c->d0_plain ? c->plain_ratio : c->ratio;'
mut dense_noop '    c->dense = 1;
    if (c->env_costs)' '    c->dense = 1;
    if (1)'
mut d0_overrides_dense 'if (!c->env_costs && !c->dense) {' 'if (!c->env_costs) {'
mut more_myopic '    for (int d = drafted + 1; d <= c->dmax; d++) {' '    for (int d = drafted + 1; d <= drafted + 1; d++) {'
mut no_relax_untested '    for (int i = drafted + 1; i <= PN_SPEC_DEPTH_MAX; i++)
        c->accept[i] += 0.05' '    for (int i = drafted + 1; i <= 0; i++)
        c->accept[i] += 0.05'
