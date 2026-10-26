#!/usr/bin/env bash
set -euo pipefail

project_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
config_file="$project_dir/config.env"
container_name="gemma4-llama"

if [[ ! -f "$config_file" ]]; then
  printf 'Missing %s. Copy config.env.example and set HF_TOKEN.\n' "$config_file" >&2
  exit 1
fi

# shellcheck disable=SC1090
source "$config_file"
: "${HF_TOKEN:?Set HF_TOKEN in config.env}"
: "${HOST:=127.0.0.1}"
: "${LLAMA_PORT:=8080}"
: "${LLAMA_MODEL_ID:=ggml-org/gemma-4-E4B-it-GGUF:Q4_0}"
: "${LLAMA_MODEL_FILE:=gemma-4-E4B-it-Q4_0.gguf}"
: "${LLAMA_MODEL_PATH:=/models/gemma-4-E4B-it-Q4_0.gguf}"
: "${LLAMA_VISION_FILE:=mmproj-gemma-4-E4B-it-Q8_0.gguf}"
: "${LLAMA_VISION_PATH:=/models/mmproj-gemma-4-E4B-it-Q8_0.gguf}"
: "${LLAMA_IMAGE:=gemma4-llama-cuda126:d7a2074}"
: "${LLAMA_CTX_SIZE:=4096}"
: "${LLAMA_PARALLEL:=1}"
: "${LLAMA_GPU_LAYERS:=99}"

mkdir -p "$project_dir/data/llama-huggingface" "$project_dir/logs"

if [[ ! -s "$project_dir/data/models/$LLAMA_MODEL_FILE" ]]; then
  printf 'Missing GGUF: %s\n' "$project_dir/data/models/$LLAMA_MODEL_FILE" >&2
  exit 1
fi

vision_args=()
if [[ -n "$LLAMA_VISION_FILE" ]]; then
  if [[ ! -s "$project_dir/data/models/$LLAMA_VISION_FILE" ]]; then
    printf 'Missing vision GGUF: %s\n' "$project_dir/data/models/$LLAMA_VISION_FILE" >&2
    exit 1
  fi
  vision_args=(--mmproj "$LLAMA_VISION_PATH")
fi

if sudo docker ps -a --format '{{.Names}}' | grep -Fxq "$container_name"; then
  printf 'Removing existing %s container...\n' "$container_name"
  sudo docker rm -f "$container_name"
fi

exec sudo docker run --detach --name "$container_name" --restart unless-stopped \
  --pull never --runtime=nvidia --network host \
  --volume "$project_dir/data/llama-huggingface:/root/.cache/huggingface" \
  --volume "$project_dir/data/models:/models:ro" \
  --volume "$project_dir/logs:/var/log/gemma4" \
  --env-file "$config_file" \
  "$LLAMA_IMAGE" \
  llama-server --model "$LLAMA_MODEL_PATH" \
    "${vision_args[@]}" \
    --host "$HOST" \
    --port "$LLAMA_PORT" \
    --ctx-size "$LLAMA_CTX_SIZE" \
    --parallel "$LLAMA_PARALLEL" \
    --gpu-layers "$LLAMA_GPU_LAYERS" \
    --ui-mcp-proxy \
    --chat-template-kwargs '{"enable_thinking":false}'
