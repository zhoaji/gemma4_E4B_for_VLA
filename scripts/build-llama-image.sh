#!/usr/bin/env bash
set -euo pipefail

project_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
image="gemma4-llama-cuda126:d7a2074"

exec sudo docker build \
  --network host \
  --file "$project_dir/docker/llama-cpp-gemma4-cuda126.Dockerfile" \
  --tag "$image" \
  "$project_dir"
