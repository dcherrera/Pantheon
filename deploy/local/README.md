# Pantheon Local as a service (systemd)

```sh
make USE_CUDA=1 pantheon            # K80 box (USE_TT=1 for a Tenstorrent box)
sudo install -m 0755 pantheon-cuda /usr/local/bin/pantheon
sudo useradd --system --create-home --home-dir /var/lib/pantheon pantheon
sudo usermod -aG video pantheon     # GPU device access (K80); TT: the card's udev group
sudo install -m 0644 deploy/local/pantheon-local.service /etc/systemd/system/
echo 'PANTHEON_LOCAL_ARGS=""' | sudo tee /etc/default/pantheon-local   # our catalog by default
sudo systemctl daemon-reload
sudo systemctl enable --now pantheon-local
journalctl -u pantheon-local | grep "pairing code"
```

- State (keys, pairing code, accepted catalog, loaded set) is in
  `/var/lib/pantheon/.pantheon/local/`, models in `/var/lib/pantheon/.pantheon/models/`
  (`--state-dir` / `--store` to move them).
- `curl` must be installed (downloads). No Python, no avahi.
- Port 8400 (`--port`). Open it to the LAN only.
- On restart the box reloads the models it had, one at a time, on the same units.
