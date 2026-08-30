#!/usr/bin/env bash
set -euo pipefail

sudo docker ps --filter name='^/gemma4-vllm$' \
  --format 'table {{.Names}}\t{{.Status}}\t{{.Image}}'
curl --fail --silent --show-error http://127.0.0.1:8000/health || true
