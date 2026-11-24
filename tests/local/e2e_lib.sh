#!/bin/bash
# tests/local/e2e_lib.sh — the H3 flow, run FROM THE MAC (python3 there reads JSON) against a box over the
# LAN (dns-sd is the Mac's), with the box itself driven over ssh. Sourced by a per-box
# driver script, which sets:
#   E2E_HOST     ssh alias of the box             E2E_DIR   repo checkout on the box
#   E2E_LABEL    expected mDNS label              E2E_PROFILE catalog profile to load
#   E2E_ENV      env for the box (e.g. CUDA_VISIBLE_DEVICES=6,7)
#   E2E_MODEL    path of qwen3-0.6b-Q8_0.gguf on the box (relative to E2E_DIR ok)
# The box binary must be a `make LOCAL_TEST_CATALOG_KEY=1` build (TEST ONLY key).
cd "$(dirname "$0")/../.." || exit 1
. tests/local/lib.sh
out=build/local_e2e/${E2E_HOST}
mkdir -p "$out"
PORT=${LT_PORT}
ID=$LT_MODEL_ID

remote() {   # run on the box, retrying a dropped ssh handshake
    for _ in 1 2 3; do
        ssh -o ConnectTimeout=15 -o ServerAliveInterval=20 "$E2E_HOST" \
            "cd $E2E_DIR && export $E2E_ENV LT_MODEL_FILE=$E2E_MODEL && . tests/local/lib.sh && $*" \
            && return 0
        sleep 3
    done
    return 1
}

box_start() { remote "lt_start \$(lt_catalog_flags)"; }
box_stop()  { remote "lt_stop"; }
api() { curl -s -m 600 -H "Authorization: Bearer $KEY" "$URL$1" "${@:2}"; }
loaded_units() { api /local/models/loaded | sed -n 's/.*"units":\[\([0-9,]*\)\].*/\1/p'; }

discover() {   # browse + resolve from the Mac; sets URL
    dns-sd -B _pantheon._tcp > "$out/browse.txt" 2>&1 & local p=$!
    sleep 8; { kill $p; wait $p; } 2>/dev/null
    lt_check "dns-sd sees $E2E_LABEL" grep -q " $E2E_LABEL\$" "$out/browse.txt"
    dns-sd -G v4 "$E2E_LABEL.local" > "$out/addr.txt" 2>&1 & p=$!
    sleep 4; { kill $p; wait $p; } 2>/dev/null
    local ip; ip=$(awk -v h="$E2E_LABEL.local." '$5 == h {print $6}' "$out/addr.txt" | head -1)
    lt_check "$E2E_LABEL.local resolves ($ip)" [ -n "$ip" ]
    URL=http://$ip:$PORT
}

chat_greedy() {   # chat_greedy <url> <key> → raw response
    curl -s -m 600 -H "Authorization: Bearer $2" "$1/v1/chat/completions" -d "{\"model\":\"$ID\",
      \"temperature\":0,\"max_tokens\":16,\"seed\":1,\"enable_thinking\":false,\"messages\":[{\"role\":\"user\",
      \"content\":\"Name three colours of the rainbow.\"}]}"
    echo
}

message_of() {    # message_of <raw response file> → content + reasoning, one line
    python3 -c "import json,sys; m=json.loads(open(sys.argv[1]).read().strip().splitlines()[-1])['choices'][0]['message']; \
print(repr((m.get('content'), m.get('reasoning_content'))))" "$1" 2>/dev/null
}

e2e_run() {
    remote "rm -rf \$LT_DIR/state \$LT_DIR/store; lt_hf_start" || exit 1
    box_start || { echo "box did not start"; exit 1; }
    discover
    code=$(remote "./pantheon local pair-code --state-dir \$LT_DIR/state")
    KEY=$(curl -s -X POST "$URL/local/pair" -d "{\"code\":\"$code\",\"label\":\"e2e\"}" |
          sed -n 's/.*"key":"\([^"]*\)".*/\1/p')
    lt_check "paired over the LAN" [ -n "$KEY" ]
    lt_check "no key -> 401" [ "$(curl -s -o /dev/null -w '%{http_code}' "$URL/v1/models")" = 401 ]
    lt_check "catalog accepted" lt_wait 60 sh -c "curl -s -H 'Authorization: Bearer $KEY' \
        $URL/local/catalog | grep -q '\"version\":1'"
    api /local/catalog > "$out/catalog.json"
    lt_check "$E2E_PROFILE fits" python3 -c "import json,sys; e=json.load(open(sys.argv[1]))['entries']; \
sys.exit(0 if any(x['profile']['id']==sys.argv[2] and x['fits'] for x in e) else 1)" \
        "$out/catalog.json" "$E2E_PROFILE"
    api /local/models/install -X POST -d "{\"catalog_id\":\"$ID\"}" > /dev/null
    lt_check "installed" lt_wait 900 sh -c "curl -s -H 'Authorization: Bearer $KEY' \
        $URL/local/models/install/$ID | grep -q '\"state\":\"done\"'"
    api /local/models/load -X POST -d "{\"catalog_id\":\"$ID\",\"profile_id\":\"$E2E_PROFILE\"}" \
        > "$out/load.json"
    cat "$out/load.json"; echo
    units=$(loaded_units)
    lt_check "loaded on units [$units]" [ -n "$units" ]
    INF=$(api /local/keys -X POST -d '{"label":"editor"}' | sed -n 's/.*"key":"\([^"]*\)".*/\1/p')
    chat_greedy "$URL" "$INF" > "$out/box_raw.txt"
    message_of "$out/box_raw.txt" > "$out/box_reply.txt"
    echo "  box: $(cat "$out/box_reply.txt")"
    lt_check "chat through the box URL with an inference key" grep -q "[A-Za-z]" "$out/box_reply.txt"

    # Exactness: the same request to a plain `pantheon serve` with the same flags.
    api /admin/models/unload -X POST -d "{\"name\":\"$ID\"}" > /dev/null
    if [ -n "$units" ]; then
        remote "tests/local/plain_probe.sh 8413 --dies $units --ctx-cap 4096 --max-sessions 2 \
            --max-concurrent 1" > "$out/plain_raw.txt"
        message_of "$out/plain_raw.txt" > "$out/plain_reply.txt"
        echo "  plain: $(cat "$out/plain_reply.txt")"
        lt_check "plain serve answered" grep -q "[A-Za-z]" "$out/plain_reply.txt"
        lt_check "box reply == plain serve reply (cmp)" cmp -s "$out/box_reply.txt" "$out/plain_reply.txt"
    fi

    # Reboot: load again, restart the box, the model must come back on the same units.
    api /local/models/load -X POST -d "{\"catalog_id\":\"$ID\",\"profile_id\":\"$E2E_PROFILE\"}" > /dev/null
    before=$(loaded_units)
    box_stop; box_start
    lt_check "model reloaded after restart" lt_wait 300 sh -c "curl -s -H 'Authorization: Bearer $KEY' \
        $URL/v1/models | grep -q '\"id\":\"$ID\"'"
    lt_check "same units after restart ([$before] -> [$(loaded_units)])" [ "$before" = "$(loaded_units)" ]
    api /admin/models/unload -X POST -d "{\"name\":\"$ID\"}" > /dev/null
    box_stop
    remote "lt_hf_stop"
    lt_done "e2e_$E2E_HOST"
}
