# 共享语音识别：ESP32-S3 手表与本地应用

本服务只负责音频转文字。手表将识别文字填入输入框，原有 Home Assistant WebSocket 链路继续负责用户提交的指令。本服务不会调用 Gemma、MCP 或 HA。

手表固件设计从[手表接入与 WebSocket 协议 v1](#手表接入与-websocket-协议-v1固件设计输入)开始；
该节包含已核对的当前连接参数、完整消息、错误码、结束条件和验收清单。
证书生成见[CA 证书](#ca-证书生成位置与手表配置)。

```text
ESP32-S3：I2S 麦克风 → PCM → WSS /ws/asr ─┐
本地应用：PCM → WS /ws/asr ──────────────┼→ 认证/会话隔离 → VAD → 共享队列
现有网页：WAV → HTTP /api/asr ──────────┘                       ↓
                                               单个常驻 Whisper base
                                                           ↓
                                  仅返回原连接/请求：partial / final
```

## 文件与实现

| 文件 | 职责 |
| --- | --- |
| server.py | HTTP/WS 双入口、认证、限流、会话生命周期、同进程双监听 |
| segmentation.py | WebRTC VAD、开始/停止说话、前置缓存、分段 |
| scheduler.py | 共享推理队列、优先级、取消、临时结果合并 |
| engine.py | 启动一次 whisper.cpp、内存 WAV、健康检查、日志隐私 |
| ../../../scripts/asr-model.sh | 常驻服务切换 base/turbo、失败恢复配置，也可显式前台启动 |
| auth.py | 离线创建/轮换/撤销每个客户端的密钥 |
| tls.py | 创建本地 CA 与带 IP/DNS SAN 的服务器证书 |
| client.py | WAV 模拟手表实时发送 PCM，也是本地接入参考 |
| test.html | 保留的浏览器整段上传测试页 |
| test/ | 独立调试台：设备、连接事件、音频播放/下载、识别文字；见 [调试台说明](test/README.md) |
| debug.py | 默认关闭的有界内存音频/文字采集，仅本机访问 |
| tests/test_asr.py | 协议、并发与生命周期测试 |

当前常驻服务选择 **多语言 whisper-large-v3-turbo 非量化版**：`data/asr/whispercpp/ggml-large-v3-turbo.bin`（GGML FP16 权重，约 1.5 GiB）。
原 base 模型 `data/asr/whispercpp/ggml-base.bin` 保留，可随时切回。语言保持 `auto`。
历史直接调用 `python3 app/brain/asr/server.py` 的默认模型仍为 base，以保持旧启动方式兼容；常驻服务通过 `ASR_MODEL_PATH` 明确选择 turbo，不依赖这个历史默认值。
中文统一经必需依赖 `opencc-python-reimplemented` 的 `t2s` 转为简体，英文和数字保留原样；
转换器在启动时加载一次，缺少依赖会启动失败，不再静默返回繁体。

服务启动时加载一次模型并保持常驻，所有入口共用一份模型。`--idle-seconds` 仅保留参数兼容，不再空闲退出。首次 socket 唤醒也会先加载模型；安装脚本同时启用 service，正常情况下不必等第一次录音再加载。

默认要求 CUDA；缺少程序、模型或 GPU 初始化证据就启动失败，**不会静默回退到 CPU**。开发测试明确使用 `--backend cpu`。内部 Whisper 仅监听 `127.0.0.1:8002`，不要发布此端口。

Whisper 是整段识别器，临时文字通过重复识别当前句子实现，并非原生增量解码。生成间隔不等于实际延迟，排队与推理可能使它变慢，不能承诺亚秒响应。

## 可视化调试台

启动现有 ASR 后，本机打开 **http://127.0.0.1:8000/test/**。先点击“开启调试采集”，再让手表录音，即可查看在线设备、连接/协议错误、原始 PCM/WAV、波形及实际返回的临时/最终文字。手表、本地 HTTP 请求按设备、会话和请求隔离显示。

默认不采集音频或正文；手动开启后只在内存临时保留，每次采集最长 10 分钟，每条记录保留最多 10 分钟，最多 20 条 / 16 MiB 音频。调试结束可“关闭并清空”。重启会丢失调试数据，不会自动保存到磁盘。详细边界、音频含义与远程电脑 SSH 转发方法见 **[test/README.md](test/README.md)**。

## turbo 与 base 启动、回退

以下命令在工程根目录执行。模型、TLS、客户端密钥分别保存，切换模型只重启 ASR，连接中的手表需要重连。

```bash
# 常驻服务：切换到非量化 turbo，成功后下一次重启仍使用它
bash scripts/asr-model.sh turbo
# 回退：使用保留的 base 模型及现有服务入口
bash scripts/asr-model.sh base
# 确认真正加载的模型，而不只看 service 是否 active
curl --fail http://127.0.0.1:8000/health
```

切换脚本写入 `~/.config/systemd/user/gemma4-asr.service.d/40-asr-model.conf` 的 `ASR_MODEL_PATH`，不改变现有 `30-asr-lan.conf` 的 LAN/TLS 配置。脚本等待健康接口返回目标文件名，启动失败时恢复切换前的模型配置并尝试重启。若以前手工在 ExecStart 写了 `--model-path`，应先移除此显式参数，防止它覆盖环境变量。

前台启动（仅供独立调试；需先停止占用相同端口的常驻 socket/service）：

```bash
systemctl --user stop gemma4-asr.socket gemma4-asr.service
bash scripts/asr-model.sh turbo --foreground --backend cuda --language auto
# 或历史 base：bash scripts/asr-model.sh base --foreground --backend cuda --language auto
# 前台 Ctrl+C 退出后，恢复已选定模型的常驻服务：
systemctl --user start gemma4-asr.socket gemma4-asr.service
```

前台方式不改变持久模型选择，默认仅本机入口；LAN/WSS 仍按后文参数配置。单个模型常驻、同一队列串行推理，勿同时启动两套同端口服务。切换模型不改变 16 kHz PCM、WSS 认证、partial/final 协议及简体转换。

非量化模型来源为 whisper.cpp 官方转换仓库：
`https://huggingface.co/ggerganov/whisper.cpp/resolve/main/ggml-large-v3-turbo.bin`。
下载到 `.part` 后校验再改名；当前固定 SHA1 为 `4af2b29d7ec73d781377bfd1758ca957a807e941`。不使用名称带 `q5_0` 或 `q8_0` 的文件。文件大小不是运行时内存占用；Gemma 与 ASR 共用 Jetson 内存。

本次切换前的服务 unit、LAN drop-in、engine/server、安装脚本和 README 原件备份在 `data/asr/backups/before-turbo-20261028-215254/`。日常回退优先用 `asr-model.sh base`，无需恢复旧代码。

### 本机部署验证（2026-10-28）

Jetson Orin NX 16GB、MAXN、现有 CUDA whisper.cpp、非量化 turbo；Gemma Q4 / 16K 上下文保持加载，但本次没有同时向 Gemma 发起生成请求：

- 模型完整文件 1,624,555,275 字节，通过上述官方 SHA1；文件头为 ftype=1（FP16）、32 层 encoder、4 层 decoder、128 mel。
- `/health` 返回 `model=ggml-large-v3-turbo.bin`、`backend=cuda`、`language=auto`、`output_script=Hans`、`translate=false`。
- 工程自带公开英文示例 `third_party/whisper.cpp/samples/jfk.wav`，时长 11 秒；HTTP 完整上传转写两次分别约 **4.64 / 4.68 秒**。此前 base 同机参考约 1.10 / 0.62 秒，不能把 turbo 名称理解为一定比 base 更快。
- 同音频按实时速度经原 WSS / CA / 临时配对发送，成功收到 final；从开始发送至全部返回约 **17.38 秒**，final 的 `elapsed_ms` 为 **6278**（含排队，不含此前录音时间）。临时识别重复计算会影响最终返回延迟。
- 采样时总 RAM 使用约 12.8GB / 15.7GB（tegrastats MB 显示），`free -h` 可用约 **2.6GiB**，GPU 工作采样约 93–99%；Gemma `/health` 返回 200。这不等于双模型同时生成或多手表并发压力测试已通过。
- 临时测试配对已撤销，调试音频采集保持关闭。回归测试 28 项通过。

以上为部署连通与耗时验证，不代表中文、中英混合及用户手表麦克风的识别准确率。可在调试台开启临时采集，对比自己的实录音频。

## 安装与启动

以下命令在工程根目录执行：

```bash
cd /home/nvidia/Desktop/gemma4_E4B_for_VLA
python3 -m pip install --target data/asr/python-deps -r app/brain/asr/requirements.txt
# 开发验证：不可与占用相同端口的服务同时运行。
python3 app/brain/asr/server.py --backend cpu --language auto
# CUDA 构建完成后：
python3 app/brain/asr/server.py --backend cuda --language auto
```

Jetson Orin CUDA 构建：关闭 FlashAttention 编译以减少构建开销，识别仍在 GPU 上运行。

```bash
cmake -S third_party/whisper.cpp -B third_party/whisper.cpp/build \
  -DGGML_CUDA=ON -DGGML_CUDA_FA=OFF -DCMAKE_CUDA_ARCHITECTURES=87 \
  -DWHISPER_BUILD_SERVER=ON -DWHISPER_BUILD_TESTS=OFF
cmake --build third_party/whisper.cpp/build --target whisper-server -j2
```

安装常驻用户服务（不需要 sudo）：

```bash
# 默认选择非量化 turbo，需先下载该模型。
ASR_BACKEND=cuda bash scripts/install-asr-user.sh
# 若需要按旧模型安装：
# ASR_MODEL_PATH="$PWD/data/asr/whispercpp/ggml-base.bin" ASR_BACKEND=cuda bash scripts/install-asr-user.sh
# CUDA 尚未完成时，可明确选择 ASR_BACKEND=cpu。
systemctl --user status gemma4-asr.socket gemma4-asr.service --no-pager
curl --fail http://127.0.0.1:8000/health
```

健康接口只在模型加载后返回 ready、实际 backend 与 model 文件名；仍需音频请求验证转写质量。不要同时启用系统级和用户级同端口服务。开机免登录运行还需要用户管理器启用 linger。常驻模型占用内存，应与 Gemma 同时运行测资源余量。

## 简体中文与中英混合识别

所有 HTTP、WS partial/final 结果均经过同一个 OpenCC t2s 转换器。它只做繁简转换，不把英文翻译成中文。
例如 `請打開 Wi-Fi 和 Bluetooth。` 返回 `请打开 Wi-Fi 和 Bluetooth。`，英文句子保持英文。
安装/更新环境需执行前文 requirements 安装命令，仅安装操作系统 opencc 命令不再是必需条件。

每个 Whisper 推理请求显式携带 `language=auto`（或服务器指定的语言）、`translate=false`、
`detect_language=false` 和 `temperature=0`。这里 detect_language=false 表示不使用“只检测语言后退出”的模式，
**不禁用 language=auto 本身的语言检测**。不沿用上一位用户/上一段语音的识别文本或语言上下文。

默认提供简短中英转写风格提示；这是 Whisper 解码参考，不是能保证执行的指令。
可通过 `--initial-prompt "你的静态提示"` 或环境变量 `ASR_INITIAL_PROMPT` 配置，传空字符串可禁用。
提示不应放用户隐私内容，也不应把某个用户的历史转写作为全局提示。

Whisper base 对短英文、口音、噪声与同句语言切换仍可能误判；auto 并不保证每个英文词都按英文拼写。
临时结果会随完整音频修正，UI 最终内容以 final 为准；不要把首次 partial 固定为最终文字。
若最终结果仍错误，应使用相同原话复现，并核对麦克风采样、句首是否漏音、录音长度及模型能力。
单独固定 `--language zh` 不能保证英文保留；纯英文场景可选择 `--language en`，中英混合默认仍用 auto。
这里的 language 是服务器配置，当前手表 start 消息没有逐请求语言选择字段，额外字段不会生效。

健康接口新增 `language`、`output_script`（Hans 表示简体）和 `translate` 字段，可检查部署是否生效。
此次修改不改变手表的音频格式、认证或 WebSocket 消息协议。服务升级重启后，手表需重新连接和认证。

## 配对与来源区分

每块手表、每个本地服务分别创建密钥，不要共用：

```bash
python3 app/brain/asr/auth.py --id watch-01 --source watch
python3 app/brain/asr/auth.py --id desktop-ui --source local
python3 app/brain/asr/auth.py --id batch-job --source background
# 撤销；再次创建同 id 即轮换密钥。
python3 app/brain/asr/auth.py --id watch-01 --revoke
```

密钥仅创建时显示一次，存入设备安全存储。服务端只保存 SHA256 摘要，配置位于被 Git 忽略的 `data/asr/clients.json`，权限 0600。配对只能在本机执行，无网络管理接口。变更影响后续认证，立即撤销已有会话需断开连接或重启服务。

client_id/source 由密钥记录确定，不信任客户端传入的设备名、来源或优先级。所有消息带 client_id/source/session_id/request_id；文字结果另带 utterance_id/revision。同一密钥只允许一个 WS 连接，重复连接被拒绝，不会抢走原连接。

旧网页的本机 HTTP 上传可不带密钥，标为 legacy-http；远程来源必须认证。未来本地应用应使用自己的 local 密钥。加 `--require-local-auth` 可取消本机免认证兼容，但原网页代理也需同步添加认证。

## 同时使用的优先级与隔离

| 来源/任务 | 初始优先值（越小越先处理） |
| --- | --- |
| 手表 watch、本地交互 local | 0，同级先来先处理 |
| 原网页 legacy-http | 10 |
| 后台批量 background | 30 |
| partial 临时识别 | 在来源值上额外加 20 |

每等待一秒优先值减 5，避免后台一直饿死。**正在执行的推理不能抢占**；高优先任务等待当前推理结束后先进入，不会并行启动第二份模型。

队列最多 32 个任务，每句最多 2 个排队任务。一个 WS 最多 2 个未完成句子，超出时终止该次录音并报错；HTTP 每客户端最多 2 个并发请求；最多 16 个 WS 连接（含认证中的连接）。

同句的新 partial 替换未执行的旧 partial，final 移除该句的排队 partial。取消后排队任务被删除，执行中的任务完成但不发送旧结果。结果仅返回提交它的原连接或 HTTP 请求，没有全局“最后一次识别结果”。重连生成新 session_id，不恢复旧音频，不把旧文字注入新输入框。

## CA 证书：生成、位置与手表配置

首次给手表接入 WSS 时，在 **Jetson 终端**生成证书。只需生成一次；本机 HTTP/WS 访问不需要证书。
CA 证书用于让手表确认服务器身份，配对 token 用于让服务器确认手表身份，两者分别配置。
手表不需要向服务器上传 CA 证书，而是在固件中保存并信任它。

### 1. 生成证书

依赖 Python 3 和 OpenSSL，可先执行 `openssl version` 确认可用。在本工程的 Jetson 上，可以从任意目录执行：

```bash
python3 /home/nvidia/Desktop/gemma4_E4B_for_VLA/app/brain/asr/tls.py --host 192.168.1.204
```

`192.168.1.204` 必须替换为 **ASR 所在 Jetson 的实际局域网 IP**，不是 HA/NAS 的 IP。
也可填写手表实际使用的 DNS 主机名；证书中的 SAN 必须与连接地址匹配。
其他人部署时，先进入自己的工程根目录，再使用相对路径：

```bash
python3 app/brain/asr/tls.py --host 192.168.1.204
```

默认输出目录相对于工程根目录，不受终端当前目录影响：

```text
data/asr/tls/
├── ca.crt       # CA 公开证书：导入手表
├── ca.key       # CA 私钥：仅保留在服务器
├── server.crt   # 服务器证书：供 ASR WSS 使用
├── server.key   # 服务器私钥：仅保留在服务器
└── ca.srl       # 签发证书使用的序列号文件
```

本机 CA 证书的完整路径：

```text
/home/nvidia/Desktop/gemma4_E4B_for_VLA/data/asr/tls/ca.crt
```

如需自定义输出位置，添加 `--directory /你的证书目录`，并同步修改 ASR 启动参数中的证书路径。
证书生成成功不代表 WSS 已启动；还需按下一节配置监听端口。

### 2. 手表保存哪个文件

只将 **ca.crt 的完整 PEM 内容**放入 ESP32 固件的可信证书配置，例如 `esp_websocket_client` 的 `cert_pem`。
保留 `-----BEGIN CERTIFICATE-----` 和 `-----END CERTIFICATE-----` 两行及中间内容。
所有连接同一服务器的手表可以共用这一份公开 CA，用户日常连接无需手动上传证书。
每块手表仍应使用“配对与来源区分”章节创建的独立配对 token；HA/MCP 的令牌不能代替 ASR 配对密钥。

`ca.key` 和 `server.key` 是私钥，不导入手表，也不提交到 Git。生成目录已被工程的 `data/` 忽略规则覆盖。
手表连接前需同步时间，并校验 CA、证书有效期以及 IP/主机名，不能关闭验证。

### 3. 检查已有证书与重复生成

可以用下面的命令查看证书有效期和服务器地址，不会打印私钥：

```bash
openssl x509 -in data/asr/tls/ca.crt -noout -subject -dates
openssl x509 -in data/asr/tls/ca.crt -noout -fingerprint -sha256
openssl x509 -in data/asr/tls/server.crt -noout -subject -dates -ext subjectAltName
openssl verify -CAfile data/asr/tls/ca.crt data/asr/tls/server.crt
```

最后一条成功时返回 `data/asr/tls/server.crt: OK`。
如果生成脚本提示 `certificate files already exist`，说明目标目录已有证书文件；检查完整性后使用已有证书，
不要为重复启动服务而删除或重建 CA。生成中断留下的文件也会触发这个提示，应先检查，不能直接认定生成成功。

服务器证书有效期一年，CA 十年。当前脚本用于首次签发，不提供原 CA 下的自动续签。
IP/主机名变化或服务器证书到期时，应使用原 CA 签发匹配的新服务器证书；建议为 Jetson 固定 DHCP 地址。
如明确需要创建全新 CA，可用 `--directory` 指定新目录，随后更新服务器证书配置和所有手表的 CA 信任。

## 局域网 WSS 与本地 HTTP 共存

默认仅监听本机。手表需要证书和网段白名单，使用**同一个进程的第二个监听端口**，不要启动第二份模型服务。

先按上一节生成证书。以下命令在工程根目录执行，示例 IP、证书路径请按环境替换。
若用户服务已占用 8000 端口，先停止它，再以前台方式启动同一份 ASR：

```bash
systemctl --user stop gemma4-asr.socket gemma4-asr.service
python3 app/brain/asr/server.py --backend cuda --language auto \
  --bind 127.0.0.1 --port 8000 \
  --lan-bind 192.168.1.204 --lan-port 8444 \
  --allow-subnet 192.168.1.0/24 \
  --tls-cert data/asr/tls/server.crt --tls-key data/asr/tls/server.key
```

本地继续使用 `http://127.0.0.1:8000/api/asr`，手表使用 `wss://192.168.1.204:8444/ws/asr`。LAN 监听没有 TLS 时拒绝启动。证书必须包含所用 IP/主机名的 SAN，手表配置可信 CA，并先同步时间；不能关闭证书验证。不会自动创建配对密钥或开放防火墙。

手表的可信 CA 使用上一节生成的 `data/asr/tls/ca.crt`。前台服务退出后，如需恢复原本机用户服务：

```bash
systemctl --user start gemma4-asr.socket gemma4-asr.service
```

持久化时用 `systemctl --user edit gemma4-asr.service`，在 `[Service]` 下先写空 `ExecStart=`，再写完整的新 `ExecStart=...server.py --socket-activation --lan-bind ...`（其余参数同上）。然后 daemon-reload 和 restart。不要在启动参数中放 token；安装器不覆盖此 drop-in。

白名单只认真实 TCP 对端，不信任 X-Forwarded-For。默认拒绝带 Origin 的浏览器 WS，浏览器流式接入须通过 `--allow-origin https://你的页面地址` 明确允许。ESP32 原生客户端无需 Origin。本机测试页和指标端点禁止远程直接访问。WSS 只暴露本 ASR 路由。

## 手表接入与 WebSocket 协议 v1（固件设计输入）

本节按 `server.py`、`segmentation.py`、`auth.py` 与当前服务配置核对。
标为“必须”的内容是固件互通要求；“建议”是客户端策略，不代表服务器新增功能。
状态名称、字段和大小写按本文发送。**日志 event 名称不是 WebSocket 消息的 state/phase。**

### 1. 当前部署参数与交付配置

以下为 2026-10-27 核对的当前 Jetson 部署，不是其他用户设备的固定地址：

| 项目 | 当前值 / 要求 |
| --- | --- |
| 手表 WebSocket URL | `wss://192.168.1.204:8444/ws/asr`，必须包含 `/ws/asr` |
| 服务器地址 / 端口 | Jetson `192.168.1.204` / TCP `8444` |
| 网络限制 | 实际 TCP 来源须属于 `192.168.1.0/24`；避免访客网络的客户端隔离 |
| 传输 | TLS 1.2 或以上 + 标准 WebSocket，服务器不协商专用子协议 |
| CA | 工程 `data/asr/tls/ca.crt`，以完整 PEM 保存到手表 |
| 本机 CA 完整路径 | `/home/nvidia/Desktop/gemma4_E4B_for_VLA/data/asr/tls/ca.crt` |
| 当前服务器证书 SAN | IP `192.168.1.204`，必须校验地址匹配 |
| 当前服务器证书有效期 | 2026-10-27 13:08:36 UTC 至 2027-10-27 13:08:36 UTC，以部署时证书实际内容为准 |
| 配对身份 | 已创建 `watch-01`，source 为 `watch`；每块新增手表各创建一个身份 |
| 密钥 | 对应身份创建时获得的 token；不在文档中提供，不从哈希文件恢复 |
| 识别后端 / 语言 | 常驻 CUDA Whisper base / `auto`，语言在服务器端配置 |
| 本机兼容入口 | `http://127.0.0.1:8000/api/asr`，供本地应用，不作为手表 URL |
| 服务器内部端口 | `8002` 是 Whisper 内部服务，手表不连接 |

当前 WSS 参数通过 `~/.config/systemd/user/gemma4-asr.service.d/30-asr-lan.conf` 持久化。
设备 IP 变化时需同步修改监听配置、固件 URL 和服务器证书 SAN；建议在路由器固定 DHCP 租约。
核对当前配置可用 `systemctl --user cat gemma4-asr.service` 与下文的监听/日志检查命令。

固件交付时须配置 URL、CA PEM、该设备 token、Wi-Fi 参数及时间同步；**不下发 ca.key/server.key**。
当前不要求手表客户端证书，也没有“CA 上传接口”。token 通过第一条认证 JSON 发送，
不能只放在 HTTP Authorization、URL 参数、MCP 配置或连接用户名中。JSON token 填原始密钥，不加 `Bearer ` 前缀。
服务器不发送时钟；TLS 握手前必须保证设备时间足以校验证书。

### 2. 会话状态机与一次完整交互

建议固件使用以下状态：

```text
DISCONNECTED → TLS/WS_CONNECTING → AUTH_WAIT → IDLE
IDLE → START_WAIT → RECORDING → STOP_WAIT → DRAINING → IDLE
任意已认证状态 → CANCEL_WAIT → IDLE
连接失败 / 致命错误 / 断线 → 清理本次录音 → DISCONNECTED
```

1. 同步时间，连接 Wi-Fi，使用 CA 校验 TLS 与服务器地址，完成 WebSocket 升级。
2. **WebSocket 连上不代表认证成功**。5 秒内发送 auth，等待 listening/ready。
3. 验证 ready 的 `protocol == 1` 和音频参数，随后发送 start，等待 listening/recording。
4. 确认 recording 后，按顺序实时发送二进制 PCM，同时持续处理所有服务器消息。
5. 用户松开录音键时，先停采样，将已采样的完整帧按序发完；不足 20 ms 的尾部补零到 20 ms，
   然后发送 stop。不能先 stop 再继续上传剩余音频。
6. 收到 stopped 后禁止继续发音频，但连接保持，继续等所有已知句子的 final/no_speech/error。
7. 当前请求完成后才能发下一次 start；也可用 cancel 明确放弃未完成请求，再等待 cancelled。

**不要求服务器发送单独的“本次请求全部完成”消息，当前协议也没有这种消息。**
客户端须跟踪 stopped 与句子集合，不能只等“第一条 final”或“固定数量 partial”。

### 3. 客户端文本命令（UTF-8 JSON，WebSocket text 消息）

每个 JSON 对象作为一条完整的 WebSocket 文本消息，不加换行分包协议，不用 JSON-RPC。
以下占位 token 必须替换为真实配对密钥。

认证，必须是连接后的第一条应用消息：

```json
{"type":"auth","token":"REPLACE_WITH_DEVICE_TOKEN"}
```

开始一次录音：

```json
{"type":"start","request_id":"watch01-20261027-000001","sample_rate":16000,"channels":1,"format":"pcm_s16le"}
```

| 字段 | 规则 |
| --- | --- |
| type | 固定为 `start` |
| request_id | 必填，1–64 个 ASCII 字母、数字、`-` 或 `_`；每次录音建议使用全新编号 |
| sample_rate | 整数 `16000`；省略时服务器默认此值，固件应显式发送 |
| channels | 整数 `1`；省略时默认 1 |
| format | 字符串 `pcm_s16le`；省略时默认此值 |

client_id/source/priority 由服务端密钥记录确定，客户端不能通过额外字段改变。
服务器不检查 request_id 的跨请求唯一性，也不据此去重；**唯一性由固件保证**。

控制命令：

```json
{"type":"stop"}
{"type":"cancel"}
{"type":"ping"}
```

| 命令 | 行为和应答 |
| --- | --- |
| stop | 仅 RECORDING 时有效；当前有效语音提交 final，回 listening/stopped。先前已提交的句子继续返回结果 |
| cancel | 放弃本连接当前请求及待回结果，回 listening/cancelled；允许空闲时调用，不影响其他设备 |
| ping | 回 listening/pong；用于应用层保活，不上传音频、不增加音频包序号 |

stop/cancel/ping 不使用 request_id 来选择历史录音，额外携带 request_id 也不会改变其作用范围。
固件应串行安排控制命令，避免上一轮 UI 操作的迟到 stop/cancel 作用到新一轮录音。
认证完成后再次发送 auth 会被视为 unknown_command，不支持连接中更换身份。

### 4. 二进制音频消息格式

| 相对消息开头的偏移 | 类型 | 内容 |
| --- | --- | --- |
| 0–3 | uint32，little-endian | 包序号 seq，从 0 开始，每条二进制消息加 1 |
| 4–末尾 | signed PCM16，little-endian | 16 kHz、单声道裸音频，无 WAV/其他容器头 |

- 每 20 ms 为一个音频帧：320 个采样 × 2 字节 = **640 字节**。
- 每条二进制消息含 1–10 个完整帧，即 PCM 640–6400 字节，总消息长度 **644–6404 字节**。
- 长度判断公式：`640 <= pcm_bytes <= 6400` 且 `pcm_bytes % 640 == 0`。
- 推荐每 100 ms 发送 5 帧：1600 个采样，3200 字节 PCM，**3204 字节总消息**。
- seq 按 WebSocket **消息**计数，不按 TCP 分段、WS 分片、20 ms 帧或 JSON 命令计数。
- TCP 自行重传，应用层不重发 seq。重复/跳号会关闭当前会话，协议没有逐包 ACK、重排或续传。
- 每次成功 start 后 seq 重新从 0 开始；服务端 recording 应答的 next_sequence 固定为 0。
- 输入必须是有符号 16-bit PCM，不接受 MP3、Opus、Float32、立体声、原始 24/32-bit I2S 槽。

第 0 包的头部十六进制为 `00 00 00 00`，第 1 包为 `01 00 00 00`。
PCM 采样 `0` 为 `00 00`，`32767` 为 `FF 7F`，`-32768` 为 `00 80`。
这些仅为编码示例，不要用满幅信号测试麦克风音量。

带宽为 32000 字节/秒 PCM，100 ms 包另加 40 字节/秒应用序号头；TLS/WS/TCP/Wi-Fi 开销另计。
录音期间必须持续发送麦克风数据，包括真实静音；停止发包不是静音结束信号。
服务器按 PCM 帧数判定静音，只有持续采样上传才能自动分句。

### 5. 服务器消息与字段

认证后的消息都有下列公共字段：

| 字段 | 类型 / 含义 |
| --- | --- |
| state | `listening`、`partial`、`final` 或 `error` |
| client_id | string，服务端确认的设备身份，如 watch-01 |
| source | string，服务端确认的来源，如 watch |
| session_id | string，当前连接的服务端会话 UUID hex；重连会改变 |
| request_id | string 或 null；第一次 start 前为 null，此后为最近一次成功 start 的编号 |

认证失败的 error **没有以上身份公共字段**，固件必须允许这些字段缺失。
其他消息的额外字段按下表解析；未识别的可选字段应忽略，不能依赖 JSON 字段顺序。

| state / phase | 额外字段与含义 |
| --- | --- |
| listening / ready | protocol=1、format=pcm_s16le、sample_rate=16000、channels=1；这是认证成功应答，不存在 state=authenticated |
| listening / recording | next_sequence=0；开始发 PCM 的依据 |
| listening / speech | utterance_id：新句子开始；加入本次请求的待完成集合 |
| listening / no_speech | utterance_id：已开始跟踪的句子声音过短，被丢弃；从集合移除 |
| listening / stopped | discarded_utterance_id：string 或 null；停止接收 PCM，非 null 时移除该句；可能是未曾收到 speech 的占位句子 |
| listening / cancelled | 本请求已取消；清除本次请求所有待完成句子，不等待 final |
| listening / pong | ping 应答；不代表收到 PCM 或识别完成 |
| partial | utterance_id、revision、text、elapsed_ms；同句的临时全文 |
| final | utterance_id、revision、text、elapsed_ms；该句最终全文 |
| error | code 必有；utterance_id/message 可能存在，处理见错误表 |

ready 完整示例：

```json
{"state":"listening","client_id":"watch-01","source":"watch","session_id":"aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa","request_id":null,"phase":"ready","protocol":1,"format":"pcm_s16le","sample_rate":16000,"channels":1}
```

recording 完整示例：

```json
{"state":"listening","client_id":"watch-01","source":"watch","session_id":"aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa","request_id":"watch01-20261027-000001","phase":"recording","next_sequence":0}
```

语音及文字消息示例（真实字段值每次不同；每行代表一条独立 WS 消息，不将多行合成一条 JSON）：

```json
{"state":"listening","client_id":"watch-01","source":"watch","session_id":"aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa","request_id":"watch01-20261027-000001","phase":"speech","utterance_id":"bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb"}
{"state":"partial","client_id":"watch-01","source":"watch","session_id":"aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa","request_id":"watch01-20261027-000001","utterance_id":"bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb","revision":1,"text":"打开卧","elapsed_ms":500}
{"state":"final","client_id":"watch-01","source":"watch","session_id":"aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa","request_id":"watch01-20261027-000001","utterance_id":"bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb","revision":3,"text":"打开卧室灯","elapsed_ms":700}
```

revision 是句子内的正整数，入队时递增，可能跳号，不要求从收到的 1 开始；各句独立。
text 为 Unicode 字符串，可能为空；partial/final 都是该句的完整识别结果，不是增量字串。
传输文本消息是 UTF-8 JSON，中文也可能编码为 `\uXXXX` 转义；必须用 JSON 解析器还原，不能直接截取 JSON 原始字节显示。
elapsed_ms 是该次任务从提交到返回的排队+推理耗时，不包含此前录音或等待静音时间，也不是音频时间戳。
服务器不保证固定 partial 数量、间隔或必有 partial；最终结果可以是收到的第一条文字消息。

### 6. stop、结果汇总与 UI 更新算法

**stopped 不等于 final，也不等于本请求所有结果已经返回；final 不代表 WS 已断开。**
由于结果异步返回，固件不要依赖 stopped、final、其他句子结果之间固定的相对顺序。

建议为每个 session_id/request_id 维护：

- pending_utterances：收到 speech 时加入；收到 final、no_speech 或 stopped.discarded_utterance_id 时移除。
- stop_ack：收到 stopped 时置 true；本次 start 时清 false。
- 每句保存最新 revision、临时文字、最终文字和创建顺序，按 speech 的到达顺序拼接各句。
- final 到达后该句标为完成，忽略该句后续 partial；丢弃更旧 revision。

正常完成条件为 `stop_ack == true && pending_utterances 为空`，同时本请求没有未处理错误。
若从头到尾是静音，可只有 recording → stopped，没有 speech/partial/final，应该显示“未识别到语音”并结束等待。
no_speech 用来结束过短声音的跟踪，不能因为已看到 speech 就永远等待 final。

文字只写入手表输入框：partial 替换同句临时内容，final 替换后固定该句；不逐条追加 partial，
也不自动调用 HA。用户确认提交后继续走原有 HA WebSocket 业务，ASR 没有控制设备的职责。
空 text 不提交为设备指令。手表接收和显示成功由固件确认，server result_sent 日志不是设备端 ACK。

用户点击取消后，应立即标记本地请求作废并停止采样/发包，再发 cancel，等 cancelled 后才能 start 新请求。
服务端不能收回已经写入网络的消息，固件必须丢弃已作废 request_id 的迟到结果。
新连接的 session_id、每次录音的 request_id 和每句 utterance_id 都要参与过滤，不能只比较设备名。

### 7. 时序、保活、限流与缓冲预算

| 项目 | 当前服务器行为 / 固件要求 |
| --- | --- |
| 认证时限 | WS 建立后 5 秒内收到首条 auth，否则报错并关闭 |
| 应用消息空闲超时 | recording 时等待下一条应用消息 10 秒，其他状态 60 秒；超时返回 timeout 并关闭 |
| WS heartbeat | 服务端配置 15 秒心跳；必须由 WS 库正确响应协议 ping/pong |
| 应用保活 | 建议 IDLE/DRAINING 时每 5 秒发 `{"type":"ping"}` 并处理 pong；协议层自动 pong 不能代替应用消息 |
| 最长录音 | 收到音频包时检查：从 start 起墙钟不超过 60 秒，累计 PCM 不超过 1,920,000 字节；建议固件不晚于 55 秒主动 stop |
| 录音上限的边界 | 60 秒不会自动为客户端执行 stop/final；只发 ping 可保持接收循环，但下一包超过上限的音频仍会被拒绝 |
| 上传速度 | 累计 PCM 秒数不能超过 start 后墙钟秒数 + 2 秒；不是文件快速上传接口 |
| 单句长度 | 累计最多 750 个 20 ms 帧（约 15 秒，含前置/静音），达到后自动分句，连接和录音仍继续 |
| 默认 VAD | 模式 2；最近 10 帧至少 6 帧有声才开始；前置缓存最多 15 帧；累计至少 13 帧有声才提交转写 |
| 默认分段 | 静音 50 帧即 1000 ms 提交 final；约每 1000 ms 音频触发临时识别，不是网络发送定时保证 |
| 服务器分段参数 | partial-ms：500–2000，silence-ms：800–1200，均步长 20；客户端 start 无法覆盖 |
| 单次识别结果等待 | 单个任务最多等待 120 秒；底层 HTTP 推理超时 90 秒；固件等待时需继续保活，并有可取消的 UI |
| 输入 WS 消息上限 | 8192 字节；超过可直接关闭，无 error JSON 保证；合法 PCM 消息最大仍为 6404 字节 |
| 输出 JSON 大小 | 未定义固定硬上限，不等于输入的 8192；固件须检查完整消息长度并防止缓冲越界 |
| 同一密钥连接数 | 只允许 1 个已认证连接，第二个被拒绝，不会抢走旧连接 |
| 总连接数 | 最多 16 个 WS（含认证中的连接），满额在升级前返回 HTTP 503 |
| 每连接待处理句子 | 最多跟踪 2 个有识别任务的句子，含当前句；超过返回 too_many_pending_utterances 并关闭 |
| 全局队列 | 最多 32 个等待任务（不含正在推理的 1 个）；临时任务会合并/替换，可能完全不回 partial |

建议录音环形缓冲 0.5–1 秒（16–32 KiB），每次网络发送缓冲至少 3204 字节；
接收 JSON 可先预留约 16 KiB，并实现长度检查/可配置扩容。该数值是固件预算建议，不是服务端输出长度承诺。
若超过固件允许长度，明确报错并结束会话，不能截断 JSON 或文字继续执行控制逻辑。
这些预算不含 TLS、Wi-Fi、I2S DMA、任务栈、UI/HA 的内存，需在实际板卡上统计峰值。
网络堵塞导致环形缓冲耗尽时取消录音并清空；不悄悄跳过样本，不追赶发送几十秒历史音频。

### 8. 错误码与恢复策略

认证阶段错误消息只有：

```json
{"state":"error","code":"authentication_failed_or_duplicate"}
```

| code | 连接行为 | 固件处理 |
| --- | --- | --- |
| authentication_failed_or_duplicate | 应答后关闭 | 检查 token、首消息和 5 秒时限；确认同一密钥是否被其他客户端占用，不要无限高频重连 |
| request_in_progress | 保持连接，原请求不变 | 原录音中或仍有旧结果等待，先 stop 并收齐结果，或 cancel 并等 cancelled |
| not_recording | 保持连接 | 重复 stop/空闲 stop；纠正固件状态，不要当作识别成功 |
| busy | 保持连接；针对某句 final 的提交失败 | 消息带 utterance_id；标记该句失败，建议 cancel 清理整次请求，稍后重新录音 |
| recognition_failed | 保持连接；某次推理/发送/等待失败 | 带 utterance_id；对失败句不使用旧文字补全，建议 cancel 后重试 |
| timeout | 应答后关闭 | 应用消息超时，重新连接/认证；检查应用 ping 和音频发送 |
| invalid_json | 应答后关闭 | UTF-8 JSON 解析失败，修正编码/分片组装 |
| invalid_message | 应答后关闭 | 已认证阶段消息不是 JSON 对象 |
| invalid_request_id | 应答后关闭 | 修正 request_id 字符与长度 |
| unsupported_audio_format | 应答后关闭 | start 音频参数不匹配，统一到 PCM16/16k/mono |
| unknown_command | 应答后关闭 | type 不支持或重复 auth；修正指令 |
| start_required | 应答后关闭 | 未进入 recording 就发二进制；先等 recording 应答 |
| invalid_audio_packet | 应答后关闭 | 修正包长度、20 ms 帧边界和 4 字节头 |
| sequence_mismatch_restart_recording | 应答后关闭 | 重复/缺失序号；重连、清缓冲，seq 从 0 重新录音，不补传 |
| recording_limit_60_seconds | 应答后关闭 | 达到墙钟/字节上限；新建录音，不能靠 ping 重置上限 |
| audio_rate_limit | 应答后关闭 | 发包超前实时 2 秒；修正实时节拍，不快速倾倒缓存 |
| too_many_pending_utterances | 应答后关闭 | 识别跟不上，减少持续负载/稍后重录，不继续囤积句子 |

已认证 error 带身份公共字段；code 必有，message 可缺失，不能按中文 message 文案分支。
有 utterance_id 的失败应从本地 pending 集合移除或直接取消该请求，不能继续永等它的 final。
不熟悉的 error code 应显示诊断并安全终止当前请求；不要假定所有 error 都已自动断线。

TLS 失败、来源白名单/Origin 拒绝（HTTP 403）、错路径（404）、错方法（405）、
连接满额（503）、输入消息过大或传输异常，可能根本没有 error JSON。记录连接层状态并清理会话。
不要用 WS 关闭码 1000 判断业务成功，只有对应请求的结果/完成状态才能判断。

### 9. 断线、重连和版本边界

- 建议采用 1/2/4/8 秒并加随机抖动的重连退避；参数属于固件策略，服务器不下发。
- 断线立刻停止采样、清本次 PCM 与待返回句子；连接恢复后必须重新 auth 并等 ready。
- 服务端重新分配 session_id，seq 在下一次成功 start 后从 0 起；不恢复旧录音，不发送旧包。
- 密钥错误、证书错误与网络错误应区分显示；只重复网络重连不能修复证书或配对问题。
- token 撤销/轮换对下一次认证生效，已有连接立即撤销需服务器断开会话/重启服务。
- 当前没有 session resume、逐包 ACK、跨连接 final 重放、exactly-once 投递或设备 UI 确认协议。
- 连接丢失时，即使日志显示 result_sent，手表也可能未收到；重新录音，不能拿别的请求结果代替。
- 新固件只接受已支持的 ready.protocol；当前为 1。未知版本停止录音并提示升级/配置，不盲目发送。

### 10. 固件验收清单

| 场景 | 应验收的结果 |
| --- | --- |
| 正常连接 | 正确 CA/时间/IP/token 获得 ready；WS 建立事件本身不算配对成功 |
| 身份错误/重复连接 | 收到认证错误后关闭，原设备连接不被抢走 |
| 正常短句 | ready → recording → speech → 零或多条 partial → final；只有最终句子固定到输入框 |
| 全静音/过短声音 | 音频字节计数增长，但不要求 final；能按 stopped/no_speech/discarded 完成，不一直转圈 |
| 多句与长语音 | 单 request 下多个 utterance，按句序显示，各句 partial 替换；15 秒分句后仍可继续录音 |
| 松键停录 | 剩余完整音频先发、stop 后不发二进制；stopped 后仍能接收迟到 final |
| 等待期间保活 | 识别和排队期间应用 ping/pong 正常，不因空闲 60 秒被误断开 |
| 取消与新录音 | 取消后忽略旧结果；等 cancelled 后再 start，新的 request_id，seq=0 |
| 错误音频包 | 错长度/序号能正确报错清理，不重试同一个音频包 |
| 断网与后台重启 | 不将上一会话文字填入新输入框，重连后重新认证录音 |
| 两块设备/本地同时用 | 各自 client/session/request 隔离，不复用 token，不误收其他结果 |
| 接收分片与 UTF-8 | 中文跨回调/跨字节边界可完整组装；长度超限有明确失败处理 |
| 数据隐私 | 不在串口/后台日志打印 token、原始音频或识别正文 |

## ESP32-S3 固件注意事项

使用 ESP-IDF I2S 录音、esp_websocket_client 发送，GPIO 和有效位宽由板卡决定。32-bit I2S 槽需先选声道和正确有效位，饱和转换为 signed PCM16，不能直接强转指针发送。

录音与网络各用一个任务，中间环形缓冲 0.5–1 秒（16–32 KB）；网络任务每 100 ms 发包。缓冲满就取消本次录音，不能悄悄丢包。网络阻塞不能阻止 I2S 采样。

包构造片段（pcm 已是小端 int16_t 的 1600 个采样）：

```c
uint8_t packet[4 + 3200];
packet[0] = seq;
packet[1] = seq >> 8;
packet[2] = seq >> 16;
packet[3] = seq >> 24;
memcpy(packet + 4, pcm, 3200);
int sent = esp_websocket_client_send_bin(ws, (const char *)packet,
                                        sizeof(packet), pdMS_TO_TICKS(1000));
if (sent == sizeof(packet)) ++seq;
else cancel_recording_and_reconnect();
```

ws/seq/pcm/cancel_recording_and_reconnect 由固件提供，上段不是完整板级固件。认证等 ready、start 等 recording 后再发音频。
使用 `esp_websocket_client_send_text` 发送 JSON、`esp_websocket_client_send_bin` 发送一条完整二进制消息；
应用序号头不在每个 WS continuation 分片里重复添加。发送失败/短发送时不重发原 seq，应取消并重新建立录音。

接收时区分 WS text、continuation 与 ping/pong/close 控制帧；按 `data_len` 读取 `data_ptr`，不能假定它以 NUL 结尾。
单个 WS 帧也可能拆成多次 DATA 回调，按 `payload_len/payload_offset` 收齐；如为协议分片消息，还须结合 opcode/fin
等到完整文本消息结束，再解析 JSON。中文 UTF-8 字节跨回调时先拼完整消息，不分别解码半个字符。
网络回调只做有限的复制/入队，录音、UI 和重连处理交给应用任务；不能在 WebSocket 事件回调内直接调用 stop/close/destroy。

token 存加密 NVS，不打印串口；cert_pem 使用以 NUL 结尾的完整 CA PEM 缓冲，并在整个连接生命周期内保持有效。
固件必须验证服务器身份，不关闭地址/证书检查。所选 ESP-IDF/TLS 版本的 IP SAN 校验、连接耗时和内存峰值需在板卡上验收。
本服务不提供唤醒词、AEC 回声消除、说话人识别或 TTS 语音合成；需要时由固件音频前端或其他服务完成。

固定并记录所选 ESP-IDF 与 esp_websocket_client 组件版本；下列链接是官方 API 依据，具体字段以所用版本头文件为准。

官方参考：[Espressif WebSocket 客户端](https://docs.espressif.com/projects/esp-protocols/esp_websocket_client/docs/latest/index.html)、[WebRTC VAD 帧规范](https://github.com/wiseman/py-webrtcvad)。

## 验证与本地应用接入

先配对，再从终端读入 token，避免写进历史：

```bash
read -rs -p 'ASR token: ' ASR_TOKEN; echo
export ASR_TOKEN
python3 app/brain/asr/client.py third_party/whisper.cpp/samples/jfk.wav
# LAN WSS 另加 --url wss://192.168.1.204:8444/ws/asr --ca <CA路径>
unset ASR_TOKEN
```

样例为英文，测试服务请选 `--language auto`。client.py 用于短音频联调，没有实现生产固件的自动重连、
周期应用 ping、UI 状态隔离和录音缓冲管理，固件仍需按上文实现完整状态机。
本地应用可用相同 WS 协议，不必另装 Whisper 或加载第二份模型。整段模式保留原接口，仅接受 60 秒内的 16 kHz mono PCM16 WAV：

```bash
curl --fail -F file=@third_party/whisper.cpp/samples/jfk.wav http://127.0.0.1:8000/api/asr
```

认证调用加 Authorization: Bearer 密钥，可提供 X-Request-ID。响应保留 text，新增 client_id/source/request_id。不要在脚本中写真实密钥。

浏览器整段录音测试：本机 `http://127.0.0.1:8000/`。其他电脑执行：

```bash
ssh -N -L 18000:127.0.0.1:8000 nvidia@192.168.1.204
```

保持终端打开，浏览器访问 **http://localhost:18000/** 并允许麦克风。这是整段上传页，流式验证用 client.py。没有部署对应网关时不要访问不存在的 9432/8443。

## 可观测性、隐私与测试

GET /metrics 仅本机访问：完成数、推理错误率、取消/拒绝数、排队长度、连接数、来源计数、累计排队/识别/最终耗时、Jetson GPU 当前使用率（不支持则 null）。监控系统定期采集可计算区间平均值。最终耗时从静音判定/stop 提交任务起算，不含此前录音时间。

日志仅含来源类别、客户端编号、请求编号、任务类型、耗时、状态，不含 token、音频、转写正文。
同为 local 的多个服务也可通过 client/request 字段区分。音频与 WAV 在内存处理，结束/取消后释放；
whisper.cpp 正文输出被关闭。宿主机 swap/core dump 需按部署隐私要求另行配置。

```bash
python3 -m unittest discover -s app/brain/asr/tests -v
curl --fail http://127.0.0.1:8000/metrics
journalctl --user -u gemma4-asr.service -n 50 --no-pager
```

协议测试使用确定性替身，覆盖双客户端、来源防冒用、HTTP/WS 串行共享、优先级/老化、取消、错误序号、重复连接、重连和 VAD 分段。真实 CPU/GPU 与手表麦克风准确率须另外实测，不能用替身代替硬件验证。

## 连接与音频排障

### 后台实时日志

服务启动和连接、认证、开始录音、首个音频包、后续约每秒音频进度、语音检测、结果发出均记录状态日志。
默认同时输出到终端/systemd 和工程内的滚动日志，单文件最大 5 MiB，保留 3 份备份：

```bash
tail -F /home/nvidia/Desktop/gemma4_E4B_for_VLA/data/asr/logs/asr.log
```

其他部署位置对应 `data/asr/logs/asr.log`，可用 `--log-file` 指定其他路径。
若 `journalctl --user` 显示没有 journal 文件或无内容，直接查看上述文件，不必修改系统日志权限。

| 日志事件 | 含义 |
| --- | --- |
| event=service_ready | 模型和监听端口已就绪；lan=disabled 表示尚未开启手表局域网入口 |
| event=ws_connected | WebSocket 已到达服务器，尚未证明配对认证通过 |
| event=authenticated | 配对成功，client 字段是客户端编号 |
| event=recording_started | 收到并接受 start 命令 |
| event=audio_received | 收到合法音频包，包括静音；packets/bytes/audio_ms 显示录音进度 |
| event=speech_detected | VAD 检测到说话 |
| event=result_sent | 文字结果已交给 WebSocket 发送；kind=final 为最终结果，text_chars 为字数 |
| event=authentication_failed_or_duplicate | 密钥错误、认证超时或同一设备重复连接 |
| event=recording_error | 音频格式、序号或时长等错误，查看 code |
| event=disconnected | 已认证会话断开 |

日志不保存配对密钥、音频或识别正文。result_sent 表示服务器发送步骤成功，不等于手表 UI 已收到或显示；
具体文字与最终到达状态仍需查看手表的接收回调。

### 连接和音频接收指标

在 Jetson 查询：

```bash
curl -s http://127.0.0.1:8000/metrics | python3 -m json.tool
```

- connections：WS 连接数，包括认证中的连接。
- authenticated_connections：已认证的在线客户端数量。
- active_clients：当前客户端列表，包含 client_id/source/request_id/recording/audio_bytes/audio_packets/last_audio_age_seconds。
- received_audio_packets、received_audio_bytes：服务启动以来收到的合法 WS 音频包与字节数，静音也会增加。
- final_count：服务器最终结果发送成功的累计数量；识别完成信息也可查看 asr.jobs 日志。

这些累计指标在服务重启后清零。active_clients 中的音频数量按本次录音统计，字节数不含 4 字节包序号。

### 一直显示 0 时

先核对本机服务与监听端口：

```bash
curl --fail http://127.0.0.1:8000/health
ss -ltn '( sport = :8000 or sport = :8444 )'
```

只有 127.0.0.1:8000 而没有局域网 8444 监听时，手表不能直接连接，需要按前文启用第二个 WSS 监听。
手表连接地址为 `wss://<Jetson-IP>:8444/ws/asr`，不是 localhost、HA 地址或内部 Whisper 的 8002 端口。
开启 WSS 后若日志仍没有 ws_connected，应检查手表目标地址、网络和 TLS 证书/时间校验；
若已 authenticated 但音频计数不增加，检查 start 流程与二进制发包；
若音频增加但没有 speech_detected，检查麦克风 PCM 格式、声道、有效位和实际音量。
