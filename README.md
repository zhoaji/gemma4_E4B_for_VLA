# Jetson Orin NX 上的 Gemma 4 E4B（中文操作手册）

本项目在 JetPack 6.2.1 / CUDA 12.6 的 Jetson Orin NX 16GB 上运行 Gemma 4 E4B。
运行时为本地构建的 `llama.cpp` CUDA 镜像，主模型为 Q4_0 GGUF；默认仅绑定本机回环地址。

## 当前组件与存放位置

| 组件 | 文件或镜像 | 位置 |
| --- | --- | --- |
| 主模型 | `gemma-4-E4B-it-Q4_0.gguf` | `data/models/` |
| 视觉投影模型 | `mmproj-gemma-4-E4B-it-Q8_0.gguf` | `data/models/` |
| 本地运行时镜像 | `gemma4-llama-cuda126:d7a2074` | Docker 本地镜像库 |
| 模型服务 | `gemma4-llama` | Docker 容器 |
| Hugging Face 缓存 | `data/llama-huggingface/` | 本项目目录 |
| 配置（含令牌） | `config.env` | 本项目目录；不要提交或分享 |

## 首次构建运行时

仅在本地镜像不存在或修改 `docker/llama-cpp-gemma4-cuda126.Dockerfile` 后执行：

```bash
cd /home/nvidia/Desktop/gemma4_E4B
./scripts/build-llama-image.sh
```

构建需要约 20–30 分钟。看到以下内容才表示成功：

```text
Successfully tagged gemma4-llama-cuda126:d7a2074
```

## 启动模型

```bash
cd /home/nvidia/Desktop/gemma4_E4B
./scripts/start-llama.sh
```

容器创建后查看模型加载过程：

```bash
sudo docker logs -f gemma4-llama
```

当日志出现 `server is listening`、`HTTP server listening` 或类似的监听信息时，服务可用。

## 网页与接口

在 Jetson 本机浏览器打开：

```text
http://127.0.0.1:8080
```

网页可用于文本聊天和加载视觉投影后上传图片。OpenAI 兼容接口地址为：

```text
http://127.0.0.1:8080/v1/chat/completions
```

检查服务：

```bash
curl http://127.0.0.1:8080/v1/models
```

文本请求示例：

```bash
curl http://127.0.0.1:8080/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{"model":"gemma-4-E4B-it","messages":[{"role":"user","content":"请用中文回复：本地推理正常。"}],"temperature":0,"max_tokens":64}'
```

## 可调整参数

编辑 `config.env` 后重启模型生效。

| 参数 | 默认值 | 作用与建议 |
| --- | ---: | --- |
| `HOST` | `127.0.0.1` | 服务监听地址。保持默认只允许本机访问；不要直接改为 `0.0.0.0`，除非已配置认证和防火墙。 |
| `LLAMA_PORT` | `8080` | 网页与 API 端口。 |
| `LLAMA_MODEL_FILE` | Q4_0 主模型 | 主模型文件名。16GB Orin 建议保持 Q4_0。 |
| `LLAMA_VISION_FILE` | Q8_0 mmproj | 图片理解所需视觉投影文件。设为空可禁用图片、减少约 560MB 磁盘占用。 |
| `LLAMA_CTX_SIZE` | `4096` | 上下文长度。增大可保留更长对话，但会增加统一内存占用并降低可用余量；16GB 建议 2048–4096。 |
| `LLAMA_PARALLEL` | `1` | 同时处理的会话槽位。内存有限时保持 1。 |
| `LLAMA_GPU_LAYERS` | `99` | 尽可能将层放到 GPU。Orin 上保持 99。 |
| `data/models/` | — | 主模型、视觉模型的持久化路径；可迁移到更大 NVMe 分区后相应修改挂载。 |
| `data/llama-huggingface/` | — | Hugging Face 缓存路径。 |

对话“记忆”不是模型自动长期保存的：`LLAMA_CTX_SIZE` 只控制本次上下文窗口。需要长期记忆时，应由后续应用把会话记录或向量数据库放在本项目之外的持久目录，再按需检索后注入提示词。

## 停止、状态与重启

停止模型（不删除模型文件或镜像）：

```bash
./scripts/stop-vllm.sh
```

查看容器状态：

```bash
sudo docker ps --filter name=gemma4-llama
```

修改 `config.env` 后重启：

```bash
./scripts/stop-vllm.sh
./scripts/start-llama.sh
```

## 局域网 UI、按需监控与模型代理（8001）

局域网设备统一访问 `http://<Jetson-IP>:8001/`。该地址由 systemd 的
`gemma4-system-monitor.socket` 持续监听；它本身仅约 8KB，不运行 Python 或
`tegrastats`。首次访问会自动唤醒 `gemma4-system-monitor.service`，无需手动启动。

服务实现位于 `app/brain/monitor/server.py`，只使用 Python 标准库：

| 路径 | 用途 |
| --- | --- |
| `/` | 提供 `app/ui/Mulitiple_model_UI.html` 主界面。 |
| `/api/system/stats` | 返回 CPU、GPU 利用率、系统 RAM、GPU/CPU 温度与设备信息。 |
| `/health`、`/v1/models`、`/v1/chat/completions` | 仅转发到本机 `127.0.0.1:8080` 的 llama.cpp 模型；回答的 SSE 数据会逐块透传。 |
| `/api/monitor/health` | 监控服务自身的健康检查。 |

UI 从 8001 打开时会自动将模型地址改为同源 8001，由后端代理到仅本机监听的
8080。因此模型端口无需暴露到局域网。状态栏每 3 秒请求一次；最后一次 UI 请求后
90 秒，监控 Python 进程与 `tegrastats` 自动退出。下次请求仍由 socket 自动唤醒。

Jetson 使用统一内存，无法准确提供独立 VRAM；状态栏对应位置改为 GPU / CPU 温度，
单位为 `℃`。刚打开页面的前 2–3 秒等待 `tegrastats` 首次采样时可能显示 `--`。

查看与排障：

```bash
sudo systemctl status gemma4-system-monitor.socket gemma4-system-monitor.service
curl http://127.0.0.1:8001/api/system/stats
sudo journalctl -u gemma4-system-monitor.service -n 100 --no-pager
```

若后续加入灯光、相机或动作控制，请扩展该 8001 边缘服务，不要另起服务占用 8001。
8001 当前没有认证，仅应开放在可信局域网；不可信网络应配置防火墙或增加认证。

## 语音输入（Whisper ASR）

UI 的“点击开始说话”会录制完整语音，停止后转换为 16 kHz WAV 并上传；完整上下文可避免
短片段造成的漏字、重复和语义误判，识别文本会在完成后填入文字输入框。8001/8443 将请求
代理到仅本机监听的 8000 ASR 服务。

默认使用已下载的多语言 Whisper `base`，以完整音频提高普通话与中英混说的准确率；语言由
模型自动检测，不会把英文强制识别成中文。结果通过 OpenCC `tw2sp` 词典统一转换为大陆简体中文。
CUDA whisper.cpp 后端完成时会优先使用，尚未就绪时自动回退到 `base` CPU。180 秒没有 ASR
请求即关闭服务并释放占用。模型与 CUDA 服务可通过 `scripts/install-asr.sh` 重建。

```bash
sudo systemctl daemon-reload
sudo systemctl restart gemma4-asr.service
```

远程浏览器的麦克风必须使用安全上下文。局域网语音输入请访问：

```text
https://<Jetson-IP>:8443/
```

8443 使用项目生成的自签名证书，首次访问需在浏览器中确认信任；HTTP 8001 仍可用于
状态查看和文字/图片对话，但多数浏览器会拒绝其麦克风权限。证书创建脚本为
`scripts/create-ui-certificate.sh`，ASR 安装脚本为 `scripts/install-asr.sh`。

排障：

```bash
sudo systemctl status gemma4-asr.socket gemma4-asr.service
curl -k https://127.0.0.1:8443/api/monitor/health
sudo journalctl -u gemma4-asr.service -n 100 --no-pager
```

## 会话记忆

从 8001/8443 打开的 UI 会为每个浏览器标签页维护独立对话 ID；点击“＋ 新对话”会创建完全隔离的
会话，不会混用旧记忆。会话数据保存在本机 `data/memory/conversations/`，不会上传到云端模式。

每轮文本对话保存用户问题和模型回答，并将当前会话的摘要与近期消息注入下一轮本地 E4B 请求。超过
20 条消息或约 9000 字符时，系统调用本地 E4B 压缩较早内容，只保留用户偏好、关键事实/决定、待办
与项目配置摘要，再保留最近 10 条原文消息，从而控制上下文与推理开销。

## 资源与排障

不要同时运行 vLLM、其他大型模型或多个 E4B 容器。需要重新尝试加载时：

```bash
./scripts/stop-vllm.sh
sudo sysctl -w vm.drop_caches=3
./scripts/start-llama.sh
```

监控功耗、温度和统一内存：

```bash
tegrastats
```

查看最近错误：

```bash
sudo docker logs --tail 200 gemma4-llama
```

## 多模态说明

E4B 支持文字、图片与音频输入。当前部署会加载视觉 `mmproj`，因此网页可上传图片。`llama-server` 已在启动日志中初始化音频输入，但该能力仍标记为实验性，质量与接口兼容性需要在接入麦克风前单独验证；GGUF 仓库没有额外需要下载的独立 E4B 音频文件。实际语音交互建议仍以本地语音识别（ASR）将语音转文字，再发送给当前 E4B 服务，稳定性更好。
