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
