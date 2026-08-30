#!/usr/bin/env bash
set -euo pipefail

project_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
config_file="$project_dir/config.env"

if [[ ! -f "$config_file" ]]; then
  printf 'Missing %s.\n' "$config_file" >&2
  exit 1
fi
# shellcheck disable=SC1090
source "$config_file"
: "${MODEL_ID:=google/gemma-4-E4B-it}"
: "${PORT:=8000}"

curl --fail --silent --show-error "http://127.0.0.1:${PORT}/v1/chat/completions" \
  --header 'Content-Type: application/json' \
  --data "{\"model\":\"${MODEL_ID}\",\"messages\":[{\"role\":\"user\",\"content\":\"Reply with exactly: Gemma 4 is ready.\"}],\"stream\":false}" \
  | python3 -m json.tool
