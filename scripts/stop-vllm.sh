#!/usr/bin/env bash
set -euo pipefail

sudo docker stop gemma4-vllm gemma4-llama 2>/dev/null || true
