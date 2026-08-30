#!/usr/bin/env bash
set -euo pipefail

project_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
config_file="$project_dir/config.env"
container_name="gemma4-vllm"

if [[ ! -f "$config_file" ]]; then
  printf 'Missing %s. Copy config.env.example and set HF_TOKEN.\n' "$config_file" >&2
  exit 1
fi

# This file is local configuration controlled by the operator.
# shellcheck disable=SC1090
source "$config_file"
: "${HF_TOKEN:?Set HF_TOKEN in config.env}"
: "${MODEL_ID:=google/gemma-4-E4B-it-qat-w4a16-ct}"
: "${HOST:=127.0.0.1}"
: "${PORT:=8000}"

if ! getent hosts ghcr.io >/dev/null; then
  printf 'ghcr.io does not resolve. Restore DNS/network access before starting.\n' >&2
  exit 1
fi

mkdir -p "$project_dir/data/huggingface" "$project_dir/logs"

if sudo docker ps -a --format '{{.Names}}' | grep -Fxq "$container_name"; then
  printf 'Removing existing %s container...\n' "$container_name"
  sudo docker rm -f "$container_name"
fi

exec sudo docker run --detach --name "$container_name" --restart unless-stopped \
  --pull always --runtime=nvidia --network host \
  --volume "$project_dir/data/huggingface:/data/models/huggingface" \
  --volume "$project_dir/logs:/var/log/gemma4" \
  --env-file "$config_file" \
  --env "HF_HUB_DISABLE_XET=1" \
  ghcr.io/nvidia-ai-iot/vllm:gemma4-jetson-orin \
  vllm serve "$MODEL_ID" \
    --host "$HOST" \
    --port "$PORT" \
    --max-model-len 4096 \
    --hf-overrides '{"text_config":{"max_position_embeddings":4096}}' \
    --gpu-memory-utilization 0.55 \
    --enable-auto-tool-choice \
    --reasoning-parser gemma4 \
    --tool-call-parser gemma4
