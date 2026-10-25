# Checkpoints: how to go back fast

## checkpoint-2026-10-25-stable (git tag, on origin)
Known good (Oct 25, ~14:40): dashboard over the SSH tunnel (USB first, Wi-Fi fallback) with depth
**off** in the dashboard, video ~29 fps with no freezes, no-vision debug buttons (scoop Bowl 1/2,
sweep to the mouth spot), Bowl 1 scoops 5 mm deeper, Bowl 2 8 mm deeper.

After it: the dashboard pages show the depth view rendered **on the Orin**
(`dashboard/run_depth_view.sh`, port 8091) instead of fetching raw depth.

### Revert only the pages (no restart; reload the browser tab)
```bash
git checkout checkpoint-2026-10-25-stable -- dashboard/static/
```

### Revert everything since the checkpoint (keeps history; then restart the dashboard)
```bash
git revert --no-edit checkpoint-2026-10-25-stable..HEAD
```

### Safe mode (what the checkpoint ran)
```bash
dashboard/run_mac.sh --token <token> --allow-extend --depth-camera ''
```
Stop the Orin depth viewer: Ctrl-C in its terminal, or `pkill -f run_depth_view`.
Turn depth off on the robot entirely: in the Viam app (orin-dev-kit), disable `webcam-2-depth`
and `webcam-2-range`.

### Tunnel to the Orin (self-healing)
```bash
nohup scripts/tunnel_keeper.sh >/tmp/tunnel_keeper.log 2>&1 &   # start
pkill -f tunnel_keeper                                           # stop
```
