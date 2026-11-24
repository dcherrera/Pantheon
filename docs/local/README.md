# Pantheon Local

`pantheon local` turns a machine with accelerator cards (K80s, a Tenstorrent p150)
into a home inference appliance. It starts with no model, shows up on the LAN as
`pantheon-<hostname>.local` (`_pantheon._tcp`), pairs with your apps, and installs
models from a signed catalog with settings tuned for your cards.

## Install

Build for your cards (`make USE_CUDA=1` for K80s, `make USE_TT=1` for a p150, plain
`make` for CPU) and run `./pantheon local`, or install it as a service:
[deploy/local/README.md](../../deploy/local/README.md). `curl` must be installed.

```
./pantheon local [--port 8400] [--catalog-url <url>] [--store <dir>] [--state-dir <dir>]
```

With no `--catalog-url` the box uses our signed catalog at
`https://huggingface.co/TeamIDE/pantheon-catalog/resolve/main/catalog.json` (signature
at `.minisig` beside it). `--catalog-url <url>` points it at a mirror;
`--catalog-url ''` turns catalog fetches off.

## Pair

The box prints a pairing code when it starts (and in `journalctl -u pantheon-local`).
On the box, `pantheon local pair-code` prints the current one. It changes after use,
after 10 minutes, and after 5 wrong tries. Nothing on the network shows it.

```sh
BOX=http://pantheon-mybox.local:8400
curl -s -X POST $BOX/local/pair -d '{"code":"12345678","label":"laptop"}'
# → {"key":"pnk_…","id":"k…","role":"admin",…}   keep the key; it is shown once
KEY=pnk_…
```

Every route needs `Authorization: Bearer <key>`. Mint inference-only keys for other
clients (editors, scripts); revoke any key by id:

```sh
curl -s -H "Authorization: Bearer $KEY" -X POST $BOX/local/keys -d '{"label":"editor"}'
curl -s -H "Authorization: Bearer $KEY" $BOX/local/keys
curl -s -H "Authorization: Bearer $KEY" -X POST $BOX/local/keys/revoke -d '{"id":"k…"}'
```

## Load a model

```sh
curl -s -H "Authorization: Bearer $KEY" $BOX/local/hardware   # cards → units
curl -s -H "Authorization: Bearer $KEY" $BOX/local/catalog    # fitting profiles first
curl -s -H "Authorization: Bearer $KEY" -X POST $BOX/local/models/install \
     -d '{"catalog_id":"<model id>"}'                          # download + sha256 (optional "profile_id")
curl -s -H "Authorization: Bearer $KEY" $BOX/local/models/install/<model id>
curl -s -H "Authorization: Bearer $KEY" -X POST $BOX/local/models/load \
     -d '{"catalog_id":"<model id>","profile_id":"<profile id>"}'
```

An install fetches only the files this box's hardware needs (a K80 box skips
p150-only files, `files[].unit_types` in the catalog); with `"profile_id"` it
fetches the files of that profile's unit type.

Optionally pick the units yourself with `"units":[6,7]` (unit ids from
`/local/hardware`). Unload or swap with
`POST /admin/models/unload {"name":"<model id>"}`; the box forgets it. It
remembers what is loaded and reloads it after a restart, after checking each
profile against the hardware it finds then. A `cpu` profile always runs on the
CPU, even on a box with cards.

`GET /local/models/loaded` lists the remembered models (and a load running now)
with `state` (`loaded`, `loading`, `unloading`, `not_loaded`), the `units` and
`placement` each really holds, a `reason` when one was not loaded (e.g. its
profile no longer fits), on a loaded replica `sessions_used` (sessions bound to it
now, out of its `max_sessions`; see `GET /admin/models`), and while a load or
unload runs its live `progress`:
`{"phase","bytes","total","layer","n_layers","units":[{"id","bytes","total"}]}`.

`GET /local/hardware` answers from a device memory snapshot the box refreshes in
the background (every 3 s, and right after a load or unload); `mem_age_ms` is its
age. Loads and the catalog's fit wait for a fresh one.

## Remove a model

```sh
curl -s -H "Authorization: Bearer $KEY" -X POST $BOX/local/models/uninstall \
     -d '{"catalog_id":"<model id>"}'      # → {"catalog_id","removed_files","freed"}
```

Removes the model's files from the store (weights, extras, card caches). Refused
(409) while the model is loaded or set to load at start (unload it first), or still
installing. `GET /local/catalog` lists models whose files are in the store but that
the catalog no longer offers under `"store_only":[{"catalog_id","bytes","files"}]`;
remove them the same way.

## Chat

The box is OpenAI-compatible at `$BOX/v1` with any key:

```sh
curl -s -H "Authorization: Bearer $INFERENCE_KEY" $BOX/v1/chat/completions \
     -d '{"model":"<model id>","messages":[{"role":"user","content":"Hello"}]}'
```

Models outside the catalog can still be run by hand with `pantheon serve --model …`.
Catalog format and signing: [catalog.md](catalog.md).

## Updating a remote box

A box you build from source runs from one checkout pair: `~/box/Pantheon` + `~/box/foundry`
on the box (`BOX_ROOT` to change it). Nothing else on the box is the box. Move it to
`origin/main` from your workstation (`<host>` is any ssh destination):

```sh
tools/box/update_box.sh <host> --backend tt   --test-catalog   # Tenstorrent: make USE_TT=1
tools/box/update_box.sh <host> --backend cuda --test-catalog   # CUDA (K80): make USE_CUDA=1
```

It fetches origin, refuses a box checkout with uncommitted changes, pushes `origin/main`
of both repos, rebuilds detached (`make clean` + `make <flags> FOUNDRY=<box foundry>
pantheon`), restarts the box like `tests/local/lib.sh` (port 8411, state and store in
`build/local_hw`) and prints the built commit. `--test-catalog` builds with
`LOCAL_TEST_CATALOG_KEY=1` and starts on the fixture catalog; pass it until the real
catalog is signed. `--no-restart` only rebuilds. With `--backend tt` the box runs with
`FT_TT_DRIVER=udrv`, which refuses to start while another serve holds the card;
`BOX_RUN_ENV= tools/box/update_box.sh <host> --backend tt …` runs it CPU-resident instead. On a multi-GPU CUDA box,
`BOX_RUN_ENV='CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=0,1,2,3'` keeps the box on
dies 0-3 while another job holds 4-7. `BOX_PANTHEON_REF=origin/<branch>` builds an
unmerged Pantheon branch instead of `origin/main`. Status: `build/box_update.status`,
log: `build/box_update_build.log`.

Smoke it with a paired key (`lt_pair` in `tests/local/lib.sh`):
`PANTHEON_API_KEY=pnk_… tests/serve_long_prompt_smoke.py http://127.0.0.1:8411 <model> 19 1900`.
