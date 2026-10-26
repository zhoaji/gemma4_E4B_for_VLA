#!/usr/bin/env bash
# Install the already-built ASR as a per-user, socket-activated service.
set -euo pipefail

project_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
asr_python="${ASR_PYTHON:-/home/nvidia/anaconda3/bin/python3}"
unit_dir="${XDG_CONFIG_HOME:-$HOME/.config}/systemd/user"

if [[ ! -x "$project_dir/third_party/whisper.cpp/build/bin/whisper-server" &&
      ! -x "$project_dir/third_party/whisper.cpp/build-cpu/bin/whisper-server" ]]; then
  echo 'ASR 程序缺失，请先完成构建。' >&2
  exit 1
fi
if [[ ! -s "$project_dir/data/asr/whispercpp/ggml-base.bin" ]]; then
  echo 'ASR 程序或模型缺失，请先完成构建和下载。' >&2
  exit 1
fi
[[ -x "$asr_python" ]] || { echo 'ASR Python 不存在。' >&2; exit 1; }
mkdir -p "$unit_dir"
cat > "$unit_dir/gemma4-asr.socket" <<'EOF'
[Unit]
Description=Gemma4 on-demand ASR socket (user)

[Socket]
ListenStream=127.0.0.1:8000
NoDelay=true

[Install]
WantedBy=sockets.target
EOF
cat > "$unit_dir/gemma4-asr.service" <<EOF
[Unit]
Description=Gemma4 on-demand Whisper ASR (user)
Requires=gemma4-asr.socket
After=gemma4-asr.socket

[Service]
Type=simple
WorkingDirectory=$project_dir
Environment=ASR_LANGUAGE=auto
Environment=ASR_FALLBACK_MODEL=base
ExecStart=$asr_python $project_dir/app/brain/asr/server.py --socket-activation --idle-seconds 180
Restart=no
TimeoutStopSec=15
NoNewPrivileges=yes
EOF
systemctl --user daemon-reload
systemctl --user enable --now gemma4-asr.socket
echo 'ASR 用户服务已启用：http://127.0.0.1:8000/api/asr'
echo '状态：systemctl --user status gemma4-asr.socket gemma4-asr.service'
