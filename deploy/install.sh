#!/usr/bin/env bash
set -euo pipefail
AGENTS_DIR="$HOME/Library/LaunchAgents"
LOG_DIR="$HOME/Library/Logs/quant-deploy"
DOMAIN="gui/$(id -u)"
HERE="$(cd "$(dirname "$0")" && pwd)"

mkdir -p "$AGENTS_DIR" "$LOG_DIR"
chmod 700 "$LOG_DIR"

for label in com.ajaiupadhyaya.quant-tick com.ajaiupadhyaya.quant-guard com.ajaiupadhyaya.quant-engine; do
  cp "$HERE/$label.plist" "$AGENTS_DIR/$label.plist"
  plutil -lint "$AGENTS_DIR/$label.plist"
  launchctl bootout "$DOMAIN/$label" 2>/dev/null || true
  launchctl bootstrap "$DOMAIN" "$AGENTS_DIR/$label.plist"
  launchctl enable "$DOMAIN/$label"
  echo "loaded $label:"
  launchctl print "$DOMAIN/$label" | sed -n '1,5p'
done

# Log rotation (requires sudo; specify owner so the user-context agents keep write access).
sudo cp "$HERE/newsyslog/quant-deploy.conf" /etc/newsyslog.d/quant-deploy.conf
sudo newsyslog -nv /etc/newsyslog.d/quant-deploy.conf
echo "Install complete. Run ./pmset.sh separately (needs sudo)."
