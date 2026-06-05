#!/usr/bin/env bash
set -euo pipefail
DOMAIN="gui/$(id -u)"
for label in com.ajaiupadhyaya.quant-tick com.ajaiupadhyaya.quant-guard com.ajaiupadhyaya.quant-engine; do
  launchctl bootout "$DOMAIN/$label" 2>/dev/null || true
  rm -f "$HOME/Library/LaunchAgents/$label.plist"
done
sudo rm -f /etc/newsyslog.d/quant-deploy.conf
echo "Uninstalled launch agents + newsyslog config (logs + pmset left intact)."
