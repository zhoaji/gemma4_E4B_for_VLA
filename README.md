# Jetson Orin NX 上的 Gemma 4 E4B（中文操作手册）

本项目在 JetPack 6.2.1 / CUDA 12.6 的 Jetson Orin NX 16GB 上运行 Gemma 4 E4B。
运行时为本地构建的 `llama.cpp` CUDA 镜像，主模型为 Q4_0 GGUF。
示例配置默认仅绑定本机回环地址；本机当前 `config.env` 设置为 `HOST=0.0.0.0`，允许局域网访问。
正式启动脚本已默认添加 `--ui-mcp-proxy`，用于原生网页连接 Home Assistant MCP。

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
cd /home/nvidia/Desktop/gemma4_E4B_for_VLA
./scripts/build-llama-image.sh
```

构建需要约 20–30 分钟。看到以下内容才表示成功：

```text
Successfully tagged gemma4-llama-cuda126:d7a2074
```

## 启动模型

```bash
cd /home/nvidia/Desktop/gemma4_E4B_for_VLA
./scripts/start-llama.sh
```

该脚本会删除并重新创建同名容器，但保留挂载在项目目录中的模型文件。
首次启动、修改配置或增加启动参数时使用它；未修改配置的日常启动使用 `sudo docker start gemma4-llama`。
不要同时运行正式容器与临时测试容器。

如果目前运行的是本次 MCP 测试创建的 `gemma4-llama-mcp-test`，先切换回正式容器：

```bash
sudo docker stop gemma4-llama-mcp-test
bash /home/nvidia/Desktop/gemma4_E4B_for_VLA/scripts/start-llama.sh
```

仅在临时容器存在时执行上面的停止命令。脚本内部会请求 sudo 权限，无需使用 `sudo bash`。

容器创建后查看模型加载过程：

```bash
sudo docker logs -f gemma4-llama
```

启动日志可用于观察加载过程，按 `Ctrl+C` 只退出日志查看，不停止容器。
通过健康接口确认模型已加载完成：

```bash
curl --fail http://127.0.0.1:8080/health
```

返回 HTTP 200 表示就绪；加载期间可能返回 503。

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

编辑 `config.env` 后，重新执行 `scripts/start-llama.sh` 创建容器才会生效。
`docker restart` 或 `docker stop` 后再 `docker start` 不会应用新增的环境变量和启动参数。

| 参数 | 默认值 | 作用与建议 |
| --- | ---: | --- |
| `HOST` | `127.0.0.1` | 示例默认仅本机访问；本机当前为 `0.0.0.0`。需要局域网直连时使用 Jetson 的局域网 IP，并限制访问范围。 |
| `LLAMA_PORT` | `8080` | 网页与 API 端口。 |
| `LLAMA_MODEL_FILE` | Q4_0 主模型 | 主模型文件名。16GB Orin 建议保持 Q4_0。 |
| `LLAMA_VISION_FILE` | Q8_0 mmproj | 图片理解所需视觉投影文件。设为空会停止加载该文件，但不会删除磁盘上的文件。 |
| `LLAMA_CTX_SIZE` | `4096` | 上下文长度。增大可保留更长对话，但会增加统一内存占用并降低可用余量；16GB 建议 2048–4096。 |
| `LLAMA_PARALLEL` | `1` | 同时处理的会话槽位。内存有限时保持 1。 |
| `LLAMA_GPU_LAYERS` | `99` | 尽可能将层放到 GPU。Orin 上保持 99。 |
| `data/models/` | — | 主模型、视觉模型的持久化路径；可迁移到更大 NVMe 分区后相应修改挂载。 |
| `data/llama-huggingface/` | — | Hugging Face 缓存路径。 |

对话“记忆”不是模型自动长期保存的：`LLAMA_CTX_SIZE` 只控制本次上下文窗口。需要长期记忆时，应由后续应用把会话记录或向量数据库放在本项目之外的持久目录，再按需检索后注入提示词。

## 停止、状态与重启

停止模型（不删除模型文件或镜像）：

```bash
sudo docker stop gemma4-llama
```

查看容器状态：

```bash
sudo docker ps --filter name=gemma4-llama
```

不修改配置，启动已存在的容器：

```bash
sudo docker start gemma4-llama
```

修改 `config.env` 或启动参数后，重新创建：

```bash
bash /home/nvidia/Desktop/gemma4_E4B_for_VLA/scripts/start-llama.sh
```

正式容器使用 `--restart unless-stopped`：Docker 随系统启动时会恢复非手动停止的容器；
手动停止后需执行 `docker start`。旧的 `scripts/stop-vllm.sh` 会同时尝试停止 vLLM 和 llama.cpp，
这里只管理 Gemma llama.cpp 时使用上面的明确容器命令。

## Home Assistant MCP：原生网页聊天控制灯

连接链路为：`8080 原生聊天网页 → llama-server MCP 代理 → HA MCP Server → 灯`。
这是 HA 的 MCP Server 接入，与 HA 内部调用模型的 llama.cpp 集成是两个独立入口。

### HA 设置

1. 在“设置 → 设备与服务 → 添加集成”中添加 **Model Context Protocol Server**。
2. 打开其配置，关闭 **Expose all LLM APIs**，勾选 **Assist**。
3. 保持 **Require an administrator account** 开启时，使用管理员账号创建访问令牌。
4. 在“设置 → 语音助手 → 暴露”中，只将允许访问的灯、脚本等实体开放给 Assist。
5. 在个人资料的“安全 → 长期访问令牌”中创建令牌，保存在客户端认证设置中，不放入提示词、仓库或截图。

### 模型原生网页设置

打开 `http://127.0.0.1:8080`，进入 **MCP Servers → Add New Server**：

| 字段 | 内容 |
| --- | --- |
| Server URL | `http://<HA局域网IP>:8123/api/mcp`；如果 HA 提供可复制的 MCP 地址，优先使用它。 |
| Display name | `HA` |
| Authorization | 打开，选择 Bearer，将完整 HA 令牌填入令牌框；若界面已有 Bearer 前缀，不要重复输入。 |
| Use llama-server proxy | 打开。正式启动脚本已包含 `--ui-mcp-proxy`。 |

如果使用 Custom Headers 而不是内置 Authorization，键为 `Authorization`，值为
`Bearer <HA令牌>`（Bearer 后有一个空格）。两种认证方式选一种，不要重复配置同名请求头。
点击 Update，看到 Connected 和可用工具列表后，新建对话测试。

先询问灯的状态，再请求打开它，并核对工具调用记录和实际灯光状态。
若“卧室灯”无法匹配，可先用实体的实际名称（例如 `room1_rgb`）测试，再在 HA 设置中文显示名称、别名和所属区域。
连接成功或模型回复“已打开”本身不代表操作成功，应查看工具返回结果。

### 常见错误

| 现象 | 检查事项 |
| --- | --- |
| `NetworkError` / `Failed to fetch` | 确认代理开关已打开、HA 地址可达；浏览器能打开 HA 登录页不能证明容器能访问 HA。 |
| 代理开关灰色 | 检查正在运行的容器是否包含代理参数，再强制刷新网页。 |
| `401` | 检查令牌及 Bearer 格式。 |
| `403` | 检查账号权限及 HA 的管理员要求。 |
| `404` | 检查 MCP Server 集成和地址。 |
| 找不到灯 | 检查 Assist 暴露设置、实体名称、别名、区域及调用参数。 |

检查启动参数：

```bash
sudo docker inspect gemma4-llama --format '{{json .Config.Cmd}}'
```

输出应包含 `--ui-mcp-proxy`。它属于模型启动参数，不能加到 `docker logs` 后面。
本功能是实验性的网络代理，应仅供可信客户端使用。

网页中的 MCP 设置不会自动赋予直接调用 `/v1/chat/completions` 的 ESP32 或其他客户端工具执行能力。
硬件后端需要实现 MCP 客户端和工具调用循环，或接入已经支持它们的后端。
当前 8001/8443 自定义 UI 仅代理模型对话等已有接口，未实现与原生网页相同的 MCP 工具调用逻辑。

参考：[HA 官方 MCP 文档](https://www.home-assistant.io/integrations/mcp_server/)、
[llama.cpp MCP 文档](https://github.com/ggml-org/llama.cpp/tree/master/tools/server#mcp-servers)。

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

UI 从 8001 打开时会自动将模型地址改为同源 8001，由后端代理到本机
8080。仅使用该代理时，模型端口无需暴露到局域网；当前 `HOST=0.0.0.0` 仍会直接开放模型监听。
状态栏每 3 秒请求一次；最后一次 UI 请求后
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

仓库中的 systemd 服务模板仍有 `/home/nvidia/Desktop/gemma4_E4B` 旧路径。
在本目录重新安装这些服务前，需先将模板中的 WorkingDirectory、ExecStart 等路径调整为
`/home/nvidia/Desktop/gemma4_E4B_for_VLA`，并检查依赖路径。不要直接覆盖已经正确运行的系统服务。

## 语音输入（Whisper ASR）

UI 的“点击开始说话”会录制完整语音，停止后转换为 16 kHz WAV 并上传；完整上下文可避免
短片段造成的漏字、重复和语义误判，识别文本会在完成后填入文字输入框。8001/8443 将请求
代理到仅本机监听的 8000 ASR 服务。

默认使用已下载的多语言 Whisper `base`，以完整音频提高普通话与中英混说的准确率；语言由
模型自动检测，不会把英文强制识别成中文。安装 OpenCC 时，结果通过 `tw2sp` 词典统一转换为大陆简体中文。
当前已构建并验证 whisper.cpp CPU 后端；完整 CUDA 程序存在时优先使用 CUDA。
独立的 Python CPU 回退需要另外安装 `openai-whisper` 及依赖，目前不依赖该路径。
180 秒没有 ASR 请求即关闭服务并释放占用。模型与 CUDA 服务可通过 `scripts/install-asr.sh` 重建。
已补齐模型和程序时，使用以下命令安装用户级按需服务，无需 sudo：

```bash
bash /home/nvidia/Desktop/gemma4_E4B_for_VLA/scripts/install-asr-user.sh
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
systemctl --user status gemma4-asr.socket gemma4-asr.service
curl --fail http://127.0.0.1:8000/health
curl -k https://127.0.0.1:8443/api/monitor/health
journalctl --user -u gemma4-asr.service -n 100 --no-pager
```

用户级 socket 仅监听 `127.0.0.1:8000`，不能直接作为 NAS 上 HA 的 Wyoming 服务使用。
音频接口测试见 `app/brain/asr/README.md`。系统级与用户级 ASR 服务不要同时占用 8000 端口。

## 会话记忆

从 8001/8443 打开的 UI 会为每个浏览器标签页维护独立对话 ID；点击“＋ 新对话”会创建完全隔离的
会话，不会混用旧记忆。会话数据保存在本机 `data/memory/conversations/`，不会上传到云端模式。

每轮文本对话保存用户问题和模型回答，并将当前会话的摘要与近期消息注入下一轮本地 E4B 请求。超过
20 条消息或约 9000 字符时，系统调用本地 E4B 压缩较早内容，只保留用户偏好、关键事实/决定、待办
与项目配置摘要，再保留最近 10 条原文消息，从而控制上下文与推理开销。

## 资源与排障

不要同时运行 vLLM、其他大型模型或多个 E4B 容器。需要重新尝试加载时：

```bash
sudo docker restart gemma4-llama
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

标准 Gemma 4 E4B-it 原生生成文字；理解图片、音频不等于生成图片或声音。
语音播放需要文字转语音（TTS），图片生成需要相应生成服务。MCP 提供工具连接，不会自动增加这些输出能力。
