#!/usr/bin/env bash
set -euo pipefail

PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
ASR_SOURCE="$PROJECT_DIR/third_party/whisper.cpp"
ASR_BUILD="$ASR_SOURCE/build"
ASR_MODEL_DIR="$PROJECT_DIR/data/asr/whispercpp"

sudo apt-get update
sudo apt-get install -y cmake build-essential curl ffmpeg
if [[ ! -d "$ASR_SOURCE/.git" ]]; then
  git clone --depth 1 https://github.com/ggml-org/whisper.cpp.git "$ASR_SOURCE"
fi
cmake -S "$ASR_SOURCE" -B "$ASR_BUILD" -DGGML_CUDA=1 -DGGML_CUDA_FA=OFF -DCMAKE_CUDA_ARCHITECTURES=87 -DWHISPER_BUILD_SERVER=ON -DWHISPER_BUILD_TESTS=OFF
cmake --build "$ASR_BUILD" --target whisper-server -j1
mkdir -p "$ASR_MODEL_DIR"
bash "$ASR_SOURCE/models/download-ggml-model.sh" base "$ASR_MODEL_DIR"
/home/nvidia/anaconda3/bin/python3 -m pip install --target "$PROJECT_DIR/data/asr/python-deps" -r "$PROJECT_DIR/app/brain/asr/requirements.txt"
sudo install -m 0644 "$PROJECT_DIR/deploy/systemd/gemma4-asr.socket" /etc/systemd/system/gemma4-asr.socket
sudo install -m 0644 "$PROJECT_DIR/deploy/systemd/gemma4-asr.service" /etc/systemd/system/gemma4-asr.service
sudo systemctl daemon-reload
sudo systemctl enable --now gemma4-asr.socket
sudo systemctl enable --now gemma4-asr.service
echo "ASR 已就绪：主 UI 会通过 8001 自动代理 /api/asr 请求。"
