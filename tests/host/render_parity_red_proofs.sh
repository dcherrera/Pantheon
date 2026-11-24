#!/usr/bin/env bash
# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026 David (Corey) Herrera
#
# render_parity_red_proofs.sh: prove the card 1.8 render-parity tests can fail.
# Reverts each fix in turn (one source mutation), rebuilds the parity and
# projection tests, and prints how many pinned rows / projection tests fail.
# Every mutation is restored with `git checkout` before the next one, so run it
# on a clean tree.  Usage (repo root): bash tests/host/render_parity_red_proofs.sh
set -u
[ -f tests/template_parity_qwen_pins.h ] || { echo "run from the repo root"; exit 2; }
if ! git diff --quiet -- src; then echo "src/ has uncommitted changes"; exit 2; fi

# name#file#python-literal old#python-literal new
MUTATIONS=(
"D3 merge#src/serve/serve_prompt_tokens.c#'!message_is_tool(messages, n_messages, i - 1),'#'true,'"
"D3 merge (close)#src/serve/serve_prompt_tokens.c#'!message_is_tool(messages, n_messages, i + 1));'#'true);'"
"D4 builder trim#src/serve/serve_prompt_tokens.c#'if (!model->chat_trim_content || !content) return content;'#'return content;'"
"D4 join trim#src/serve/tool_calling.c#'char *trimmed = dup_trim(content ? content : \"\",'#'char *trimmed = strdup(content ? content : \"\"); (void)('"
"D5 tojson#src/serve/tool_calling.c#'else                         jr_to_json_tojson(v, jw);'#'else                         jr_to_json(v, jw);'"
"D5 float repr#src/serve/json_reader.c#'for (int precision = 1; precision <= 17;'#'for (int precision = 17; precision <= 17;'"
"D7 ids#src/serve/serve_prompt_tokens.c#'if (open_id >= 0 && close_id >= 0) {'#'if (0) {'"
"projection trim#src/serve/serve_prompt_tokens.c#'template_content(model, joined ? joined : content, &trimmed);'#'joined ? joined : content; (void)template_content;'"
)

run_case() {
    rm -f tests/test_template_parity_qwen tests/test_prefix_projection
    make USE_BLAS=1 tests/test_template_parity_qwen tests/test_prefix_projection \
        > build/red_proof_build.log 2>&1 || { echo "  BUILD FAILED (see build/red_proof_build.log)"; return; }
    local pins proj
    pins=$(./tests/test_template_parity_qwen 2>&1 | grep -c "PIN MISMATCH")
    proj=$(./tests/test_prefix_projection 2>&1 | grep -E "total" | sed 's/\x1b\[[0-9;]*m//g')
    echo "  parity: $pins of $(grep -c '^{ ' tests/template_parity_qwen_pins.h) rows mismatch; projection: $proj"
}

mkdir -p build
echo "baseline"
run_case
for m in "${MUTATIONS[@]}"; do
    IFS='#' read -r name file old new <<< "$m"
    echo "$name ($file)"
    python3 - "$file" "$old" "$new" <<'EOF' || { echo "  mutation not applied"; continue; }
import ast, sys
path, old, new = sys.argv[1], ast.literal_eval(sys.argv[2]), ast.literal_eval(sys.argv[3])
src = open(path).read()
if old not in src:
    sys.exit(1)
open(path, "w").write(src.replace(old, new, 1))
EOF
    run_case
    git checkout -- "$file"
done
echo "restored"
run_case
