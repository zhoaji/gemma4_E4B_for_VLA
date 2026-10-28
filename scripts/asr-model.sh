#!/usr/bin/env bash
# Switch the persistent shared ASR without changing LAN/TLS configuration.
set -euo pipefail
project_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
asr_python="${ASR_PYTHON:-/home/nvidia/anaconda3/bin/python3}"
profile="${1:-}"
case "$profile" in
  base) filename=ggml-base.bin ;;
  turbo) filename=ggml-large-v3-turbo.bin ;;
  *) echo "用法：bash scripts/asr-model.sh base|turbo [--foreground [server 参数...]]" >&2; exit 2 ;;
esac
shift
model_path="$project_dir/data/asr/whispercpp/$filename"
[[ -s "$model_path" ]] || { echo "模型文件缺失：$model_path" >&2; exit 1; }
if [[ "${1:-}" == --foreground ]]; then
  shift
  exec "$asr_python" "$project_dir/app/brain/asr/server.py" --model-path "$model_path" "$@"
fi
[[ $# == 0 ]] || { echo '未知参数；前台启动请使用 --foreground。' >&2; exit 2; }
unit_dir="${XDG_CONFIG_HOME:-$HOME/.config}/systemd/user/gemma4-asr.service.d"
mkdir -p "$unit_dir"
exec 9>"$unit_dir/.model-switch.lock"
flock -n 9 || { echo '已有模型切换操作正在执行。' >&2; exit 1; }
override="$unit_dir/40-asr-model.conf"
previous="$(mktemp)"
if [[ -f "$override" ]]; then cp "$override" "$previous"; else
  printf '[Service]\nEnvironment="ASR_MODEL_PATH=%s"\n' "$project_dir/data/asr/whispercpp/ggml-base.bin" > "$previous"
fi
trap 'rm -f "$previous"' EXIT
printf '[Service]\nEnvironment="ASR_MODEL_PATH=%s"\n' "$model_path" > "$override"
if systemctl --user daemon-reload && systemctl --user restart gemma4-asr.service && "$asr_python" - "$filename" <<'PY'
import json,sys,time
from urllib.request import urlopen
for _ in range(180):
    try:
        with urlopen('http://127.0.0.1:8000/health', timeout=2) as r:
            health=json.load(r)
        if health.get('status')=='ready' and health.get('model')==sys.argv[1]:
            print('ASR 已就绪：'+health['model']+'，后端：'+health['backend'])
            break
    except (OSError,ValueError):
        pass
    time.sleep(1)
else:
    raise SystemExit('等待模型启动超时。')
PY
then
  echo '已保留现有 LAN/TLS/密钥配置；调试台：http://127.0.0.1:8000/test/'
else
  echo '新模型未就绪，恢复之前的模型配置。' >&2
  cp "$previous" "$override"
  systemctl --user daemon-reload
  systemctl --user restart gemma4-asr.service
  exit 1
fi
