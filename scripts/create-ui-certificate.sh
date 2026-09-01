#!/usr/bin/env bash
set -euo pipefail

PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
CERT_DIR="$PROJECT_DIR/data/certs"
JETSON_IP="$(hostname -I | awk '{print $1}')"

if [[ -z "$JETSON_IP" ]]; then
  echo "无法自动确定 Jetson 的局域网 IP。" >&2
  exit 1
fi

mkdir -p "$CERT_DIR"
openssl req -x509 -newkey rsa:2048 -sha256 -nodes -days 825 \
  -keyout "$CERT_DIR/ui-key.pem" \
  -out "$CERT_DIR/ui-cert.pem" \
  -subj "/CN=$JETSON_IP" \
  -addext "subjectAltName=IP:$JETSON_IP,DNS:localhost"
chmod 600 "$CERT_DIR/ui-key.pem"
echo "HTTPS UI：https://$JETSON_IP:8443/"
