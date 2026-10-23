#!/usr/bin/env bash
# Run ON THE JETSON (as root): use the laptop's internet over the USB link (l4tbr0 ->
# 192.168.55.100, the laptop shares its connection with NAT) while it works, and fall back
# to the Jetson's own Wi-Fi when it doesn't (cable out, laptop asleep, sharing off).
# Temporary by design: nothing persists across a reboot. Stop it with: sudo pkill -f usb_uplink_watchdog
set -u
DEV=l4tbr0; GW=192.168.55.100; METRIC=50; PROBE=1.1.1.1
sysctl -qw "net.ipv4.conf.$DEV.ignore_routes_with_linkdown=1"
# Anything addressed to the Wi-Fi IP (e.g. teammates' SSH from another subnet) answers over Wi-Fi.
WIFI=$(ip -4 -o route show default | awk '$5 != "'$DEV'" {print $5; exit}')
if [ -n "$WIFI" ]; then
  WIP=$(ip -4 -o addr show "$WIFI" | awk '{print $4}' | cut -d/ -f1)
  WGW=$(ip -4 route show default dev "$WIFI" | awk '{print $3; exit}')
  WNET=$(ip -4 route show dev "$WIFI" scope link proto kernel | awk '{print $1; exit}')
  ip route replace default via "$WGW" dev "$WIFI" table 100
  ip route replace "$WNET" dev "$WIFI" table 100
  ip rule show | grep -q "from $WIP lookup 100" || ip rule add from "$WIP" table 100 priority 100
fi
fails=0
while true; do
  if ping -c 1 -W 2 -I "$DEV" "$PROBE" >/dev/null 2>&1; then
    fails=0
    if ! ip route show default | grep -q "via $GW dev $DEV metric $METRIC"; then
      ip route replace default via "$GW" dev "$DEV" metric "$METRIC"
      resolvectl dns "$DEV" 1.1.1.1 8.8.8.8; resolvectl default-route "$DEV" true; resolvectl domain "$DEV" "~."
      logger -t usb-uplink "laptop uplink up: default route via $GW"
    fi
  else
    fails=$((fails + 1))
    if [ "$fails" -ge 3 ] && ip route show default | grep -q "via $GW dev $DEV metric $METRIC"; then
      ip route del default via "$GW" dev "$DEV" metric "$METRIC"
      resolvectl default-route "$DEV" false
      logger -t usb-uplink "laptop uplink down: back on Wi-Fi"
    fi
  fi
  sleep 10
done
