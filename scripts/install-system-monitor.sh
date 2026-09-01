#!/usr/bin/env bash
set -euo pipefail

PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
sudo install -m 0644 "$PROJECT_DIR/deploy/systemd/gemma4-system-monitor.socket" /etc/systemd/system/gemma4-system-monitor.socket
sudo install -m 0644 "$PROJECT_DIR/deploy/systemd/gemma4-system-monitor.service" /etc/systemd/system/gemma4-system-monitor.service
sudo systemctl daemon-reload
sudo systemctl enable --now gemma4-system-monitor.socket
echo "主 UI：http://<Jetson-IP>:8001/"
echo "系统状态接口：http://<Jetson-IP>:8001/api/system/stats"
