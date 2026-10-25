# Handoff: give the Orin your laptop's internet over the USB cable, then SSH in

Paste this to whoever takes the cable next (or to their AI coding assistant).

---

You're taking over the USB-C cable to the team's **Jetson AGX Orin** (`orin-dev-kit` on Viam,
hostname `rm-sentry`). Its own Wi-Fi is weak at its spot (−80 dBm), so it uses the internet
of the laptop on the cable. While your laptop shares its connection, viam-server stays on the
cloud and teammates can reach the machine from app.viam.com.

**1. Plug in.** USB-C cable from your laptop to the Orin's **device-mode port, the USB-C port
next to the 40-pin header** (not the one by the power jack). After about 10 s your laptop gets
`192.168.55.100` and the Orin is `192.168.55.1`. Check:
```bash
ping -c 2 192.168.55.1
```

**2. Share your internet with it.** You run this yourself; it asks for your password.
On macOS (it finds your internet interface automatically):
```bash
IF=$(route -n get default | awk '/interface:/{print $2}'); sudo sysctl -w net.inet.ip.forwarding=1 && echo "nat on $IF from 192.168.55.0/24 to any -> ($IF) static-port" | sudo pfctl -a com.apple/jetson-share -f - && sudo pfctl -E
```
On Linux:
```bash
IF=$(ip route show default | awk '{print $5; exit}'); sudo sysctl -w net.ipv4.ip_forward=1 && sudo iptables -t nat -A POSTROUTING -s 192.168.55.0/24 -o "$IF" -j MASQUERADE
```
Keep the laptop awake while others use the robot (macOS: run `caffeinate -dims` in a spare terminal).

**3. The Orin side is automatic.** A watchdog on the Orin checks every 10 s and routes through
whichever laptop sits at `192.168.55.100`, falling back to its own Wi-Fi when that fails.
If the Orin has been rebooted, start the watchdog once over SSH (step 4):
```bash
sudo setsid nohup ~/assistive-arm-servo/scripts/usb_uplink_watchdog.sh >/dev/null 2>&1 </dev/null &
```

**4. SSH in over the cable.** Ask the team for the `ultraviolet` password, or have someone add
your public key to `~/.ssh/authorized_keys` on the Orin:
```bash
ssh -o HostKeyAlias=rm-sentry.local ultraviolet@192.168.55.1
```
`HostKeyAlias` matters because every Jetson uses `192.168.55.1`; without it SSH may refuse
with a "host key changed" warning if you've used another Jetson before. Over Wi-Fi instead:
`ssh ultraviolet@10.55.13.77` (or `rm-sentry.local`).

**5. Check it works** (on the Orin):
```bash
ip route get 8.8.8.8      # expect: via 192.168.55.100 dev l4tbr0
ping -c 3 8.8.8.8         # expect ~20 ms (the Orin's own Wi-Fi gave 300-600 ms)
```
The Viam app should show `orin-dev-kit` online and stay connected.

**6. Optional: the face-servo dashboard** (repo `Wafleem/assistive-feeding-arm`):
put your own Viam API key in `.env` (`VIAM_ADDRESS=orin-dev-kit-main.1pabvgfolw.viam.cloud`,
`VIAM_API_KEY`, `VIAM_API_KEY_ID`: Viam app → orin-dev-kit → CONNECT → API keys), then
`JETSON_KEY=~/.ssh/<your key> dashboard/run_mac.sh` and open the printed `localhost:8090` link.
The camera video comes over the cable; arm commands go through the Viam cloud.

**Handing the cable on:** just unplug. The Orin goes back to its Wi-Fi within about 30 s.
To stop sharing on your laptop:
- macOS: `sudo pfctl -a com.apple/jetson-share -F all && sudo sysctl -w net.inet.ip.forwarding=0`
- Linux: `sudo iptables -t nat -D POSTROUTING -s 192.168.55.0/24 -o "$IF" -j MASQUERADE`

Never commit API keys or passwords. Background: `docs/SO101_SETUP.md`, section "Network".
