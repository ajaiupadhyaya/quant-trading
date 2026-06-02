#!/usr/bin/env bash
# Power settings for an always-on Mac mini. Persistent (survives reboot).
set -euo pipefail
sudo pmset -a sleep 0 disablesleep 1 displaysleep 0 disksleep 0 \
  autorestart 1 womp 1 powernap 0 standby 0 tcpkeepalive 1
echo "Applied. Verify with: pmset -g"
sudo systemsetup -setremotelogin on
