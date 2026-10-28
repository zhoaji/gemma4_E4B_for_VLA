#!/usr/bin/env bash
# Install persistent shared ASR; socket remains compatible with local clients.
set -euo pipefail

project_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
asr_python="${ASR_PYTHON:-/home/nvidia/anaconda3/bin/python3}"
asr_backend="${ASR_BACKEND:-cuda}"
asr_model="${ASR_MODEL_PATH:-$project_dir/data/asr/whispercpp/ggml-large-v3-turbo.bin}"
unit_dir="${XDG_CONFIG_HOME:-$HOME/.config}/systemd/user"

[[ "$asr_backend" == cuda || "$asr_backend" == cpu ]] || { echo 'ASR_BACKEND 必须是 cuda 或 cpu'; exit 1; }
asr_build=build
[[ "$asr_backend" == cpu ]] && asr_build=build-cpu
if [[ ! -x "$project_dir/third_party/whisper.cpp/$asr_build/bin/whisper-server" ]]; then
  echo "ASR $asr_backend 程序缺失，请先完成构建；CPU 测试请明确设置 ASR_BACKEND=cpu。" >&2
  exit 1
fi
if [[ ! -s "$asr_model" ]]; then
  echo 'ASR 程序或模型缺失，请先完成构建和下载。' >&2
  exit 1
fi
[[ -x "$asr_python" ]] || { echo 'ASR Python 不存在。' >&2; exit 1; }
PYTHONPATH="$project_dir/data/asr/python-deps" "$asr_python" -c 'import aiohttp, webrtcvad; from opencc import OpenCC; OpenCC("t2s")'
mkdir -p "$unit_dir"
cat > "$unit_dir/gemma4-asr.socket" <<'EOF'
[Unit]
Description=Gemma4 shared ASR local socket (user)

[Socket]
ListenStream=127.0.0.1:8000
NoDelay=true

[Install]
WantedBy=sockets.target
EOF
cat > "$unit_dir/gemma4-asr.service" <<EOF
[Unit]
Description=Gemma4 persistent shared Whisper ASR (user)
Requires=gemma4-asr.socket
After=gemma4-asr.socket

[Service]
Type=simple
WorkingDirectory=$project_dir
Environment=ASR_LANGUAGE=auto
Environment=ASR_BACKEND=$asr_backend
Environment="ASR_MODEL_PATH=$asr_model"
ExecStart=$asr_python $project_dir/app/brain/asr/server.py --socket-activation
Restart=on-failure
RestartSec=5
TimeoutStopSec=15
NoNewPrivileges=yes

[Install]
WantedBy=default.target
EOF
# Keep the selected model authoritative even after a previous base/turbo switch.
mkdir -p "$unit_dir/gemma4-asr.service.d"
printf '[Service]\nEnvironment="ASR_MODEL_PATH=%s"\n' "$asr_model" > "$unit_dir/gemma4-asr.service.d/40-asr-model.conf"
systemctl --user daemon-reload
systemctl --user enable --now gemma4-asr.socket
systemctl --user enable gemma4-asr.service
systemctl --user restart gemma4-asr.service
"$asr_python" - "$asr_backend" "$asr_model" <<'PY'
import json
from pathlib import Path
import sys
import time
from urllib.request import urlopen
for _ in range(120):
    try:
        with urlopen('http://127.0.0.1:8000/health', timeout=1) as response:
            status = json.load(response)
        if (status.get('status') == 'ready' and status.get('backend') == sys.argv[1]
                and status.get('model') == Path(sys.argv[2]).name):
            break
    except (OSError, ValueError):
        pass
    time.sleep(1)
else:
    raise SystemExit('ASR 未就绪，请检查 journalctl --user -u gemma4-asr.service')
PY
echo "ASR 常驻用户服务已启动（$asr_backend）：http://127.0.0.1:8000/api/asr"
echo '状态：systemctl --user status gemma4-asr.socket gemma4-asr.service'
