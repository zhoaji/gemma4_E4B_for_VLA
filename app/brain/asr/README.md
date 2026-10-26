# 共享语音识别：ESP32-S3 手表与本地应用

本服务只负责音频转文字。手表将识别文字填入输入框，原有 Home Assistant WebSocket 链路继续负责用户提交的指令。本服务不会调用 Gemma、MCP 或 HA。

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
| auth.py | 离线创建/轮换/撤销每个客户端的密钥 |
| tls.py | 创建本地 CA 与带 IP/DNS SAN 的服务器证书 |
| client.py | WAV 模拟手表实时发送 PCM，也是本地接入参考 |
| test.html | 保留的浏览器整段上传测试页 |
| tests/test_asr.py | 协议、并发与生命周期测试 |

模型为多语言 base：`data/asr/whispercpp/ggml-base.bin`。CLI 默认语言 zh；中英混合用 `--language auto`，安装脚本默认 auto。安装 OpenCC 时结果可转为简体。

服务启动时加载一次模型并保持常驻，所有入口共用一份模型。`--idle-seconds` 仅保留参数兼容，不再空闲退出。首次 socket 唤醒也会先加载模型；安装脚本同时启用 service，正常情况下不必等第一次录音再加载。

默认要求 CUDA；缺少程序、模型或 GPU 初始化证据就启动失败，**不会静默回退到 CPU**。开发测试明确使用 `--backend cpu`。内部 Whisper 仅监听 `127.0.0.1:8002`，不要发布此端口。

Whisper 是整段识别器，临时文字通过重复识别当前句子实现，并非原生增量解码。生成间隔不等于实际延迟，排队与推理可能使它变慢，不能承诺亚秒响应。

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
ASR_BACKEND=cuda bash scripts/install-asr-user.sh
# CUDA 尚未完成时，可明确选择 ASR_BACKEND=cpu。
systemctl --user status gemma4-asr.socket gemma4-asr.service --no-pager
curl --fail http://127.0.0.1:8000/health
```

健康接口只在模型加载后返回 ready 和实际 backend；仍需音频请求验证转写质量。不要同时启用系统级和用户级同端口服务。开机免登录运行还需要用户管理器启用 linger。常驻模型占用内存，应与 Gemma 同时运行测资源余量。

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

## 局域网 WSS 与本地 HTTP 共存

默认仅监听本机。手表需要证书和网段白名单，使用**同一个进程的第二个监听端口**，不要启动第二份模型服务。

示例 IP、证书路径请按环境替换：

```bash
python3 app/brain/asr/tls.py --host 192.168.1.204
python3 app/brain/asr/server.py --backend cuda --language auto \
  --bind 127.0.0.1 --port 8000 \
  --lan-bind 192.168.1.204 --lan-port 8444 \
  --allow-subnet 192.168.1.0/24 \
  --tls-cert data/asr/tls/server.crt --tls-key data/asr/tls/server.key
```

本地继续使用 `http://127.0.0.1:8000/api/asr`，手表使用 `wss://192.168.1.204:8444/ws/asr`。LAN 监听没有 TLS 时拒绝启动。证书必须包含所用 IP/主机名的 SAN，手表配置可信 CA，并先同步时间；不能关闭证书验证。不会自动创建配对密钥或开放防火墙。

tls.py 需要 openssl，生成 data/asr/tls/ca.crt、ca.key、server.crt、server.key，拒绝覆盖已有证书。
手表只需导入公开的 ca.crt；ca.key 和 server.key 保留在服务器，不下发。
服务器证书有效期一年，CA 十年；到期前续签需保留原 CA，换 CA 后设备也必须更新信任。
主机 IP 变化也需要签发匹配的新证书，建议固定 DHCP 地址。

持久化时用 `systemctl --user edit gemma4-asr.service`，在 `[Service]` 下先写空 `ExecStart=`，再写完整的新 `ExecStart=...server.py --socket-activation --lan-bind ...`（其余参数同上）。然后 daemon-reload 和 restart。不要在启动参数中放 token；安装器不覆盖此 drop-in。

白名单只认真实 TCP 对端，不信任 X-Forwarded-For。默认拒绝带 Origin 的浏览器 WS，浏览器流式接入须通过 `--allow-origin https://你的页面地址` 明确允许。ESP32 原生客户端无需 Origin。本机测试页和指标端点禁止远程直接访问。WSS 只暴露本 ASR 路由。

## WebSocket 协议 v1

连接 `/ws/asr` 后 5 秒内发送认证文本帧：

```json
{"type":"auth","token":"该客户端自己的配对密钥"}
```

等 `state=listening,phase=ready`，再发送：

```json
{"type":"start","request_id":"watch01-000001","sample_rate":16000,"channels":1,"format":"pcm_s16le"}
```

request_id 是 1–64 个 ASCII 字母、数字、连字符或下划线，每次录音使用新编号。收到 `phase=recording,next_sequence=0` 后发送二进制消息：

| 字节 | 内容 |
| --- | --- |
| 0–3 | uint32 小端包序号，从 0 开始，每消息加 1 |
| 4–末尾 | 裸 PCM：16 kHz、单声道、有符号 16-bit、小端，无 WAV 头 |

每包 1–10 个完整 20 ms 帧。每帧 320 个采样 = 640 字节。推荐每 100 ms 发一次，共 3204 字节，音频带宽约 32 KB/s。不接受 MP3、Opus、Float32、32-bit I2S 槽、立体声或 WAV 头。

丢包/重复序号会报错并关闭连接；清缓存并重新录音，不能拼接有缺口的语音。一次录音最多 60 秒/1.92 MB，上传不得超前实时音频两秒以上。

返回示例（省略部分身份字段，实际返回均包含）：

```json
{"state":"listening","phase":"speech","session_id":"s1","request_id":"watch01-000001","utterance_id":"u1"}
{"state":"partial","client_id":"watch-01","source":"watch","session_id":"s1","request_id":"watch01-000001","utterance_id":"u1","revision":1,"text":"打开卧","elapsed_ms":500}
{"state":"final","client_id":"watch-01","source":"watch","session_id":"s1","request_id":"watch01-000001","utterance_id":"u1","revision":2,"text":"打开卧室灯","elapsed_ms":600}
```

elapsed_ms 为提交任务至结果的排队+推理耗时，示例数值不是性能承诺。partial 必须**替换同句临时文字**，不可逐条追加。final 替换并提交该句文字；按 session/request/utterance 隔离状态，丢弃旧 revision，final 后忽略该句 partial。只填输入框，不自动发送 HA 指令。一次录音可产生多个句子，各有独立 utterance_id。

控制命令：

```json
{"type":"stop"}
{"type":"cancel"}
{"type":"ping"}
```

- stop：结束采样，存在有效语音则生成 final。phase=stopped 先返回，final 可能稍后到达。无有效语音时带 discarded_utterance_id，客户端清除该句临时状态；自然静音丢弃过短声音时返回 phase=no_speech。
- cancel：丢弃本录音及待返回结果，返回 phase=cancelled，不影响其他客户端。
- ping：返回 phase=pong；服务还有 15 秒 WS heartbeat，客户端需处理协议 ping/pong。

无语音不调用 Whisper。VAD：20 ms 帧、约 200 ms 启动投票、300 ms 前置缓存、至少 260 ms 有声。默认每 1 秒生成 partial，静音 1 秒生成 final；`--partial-ms` 可选 500–2000，`--silence-ms` 可选 800–1200，均步长 20 ms。

连续语音 15 秒强制分句；边界不重叠，长篇听写可能断句漏字，当前以短指令和短句为主。认证超时 5 秒，录音中 10 秒无应用消息、空闲 60 秒无消息会关闭连接。录音时持续上传静音 PCM，不能用停发包表示静音。ping 不延长录音总时长。

错误通过 state=error 返回，如 busy、recognition_failed、sequence_mismatch_restart_recording、timeout。客户端显示失败，不复用上次文字。断线按 1/2/4/8 秒带随机抖动退避，不重放旧音频。当前没有跨连接 final 重放/ACK：断线可能丢失未收到的文字，需重新录音；不会自动执行重复动作。

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

ws/seq/pcm/cancel_recording_and_reconnect 由固件提供，上段不是完整板级固件。认证等 ready、start 等 recording 后再发音频。接收 JSON 时按 payload_len/payload_offset 拼完整分片后解析，不能把每次 DATA 回调视为完整 JSON。token 存加密 NVS，不打印串口；cert_pem 配置信任 CA，不关闭校验。

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

样例为英文，测试服务请选 `--language auto`。本地应用可用相同 WS 协议，不必另装 Whisper 或加载第二份模型。整段模式保留原接口，仅接受 60 秒内的 16 kHz mono PCM16 WAV：

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

真实双客户端自检（需要已构建的后端和 openssl；使用临时密钥、证书及本机 18001/18002/18444 端口）：

```bash
TEST_ASR_BACKEND=cpu python3 app/brain/asr/tests/smoke_live.py
# CUDA 完成后使用 TEST_ASR_BACKEND=cuda。
```

一个模拟客户端走带证书验证的 WSS，另一个走本机 WS。报告写入被 Git 忽略的 data/asr/streaming-verification-<后端>.json，不保存录音或识别正文。

## 本机验证记录（2026-10-27）

- 15 项自动测试通过，包含身份、配对轮换/撤销、优先级、取消、重连、静音和故障健康检查。
- 完成 Orin 架构 87 的 CUDA whisper-server 构建；严格 GPU 模式实测拒绝 CPU 后端。
- 真实双客户端测试通过：模拟手表经验证 CA 的 WSS、本地应用经 WS，共享一个 GPU Whisper base。
- 使用 11 秒公开英文样例，两端各收到 1 条最终结果，临时结果分别为 4/5 条；没有串结果或推理错误。
- 该次测试最终任务的排队+识别耗时约 0.72 秒和 1.33 秒，不包含录音时间；CPU 对照约 5–9 秒。
- 此数据只代表本机样例测试，不代表实际 ESP32 麦克风噪声、普通话准确率或多人持续负载性能。
- 未连接真实手表；生产 LAN WSS 需按上文生成证书、为设备配对，并设置第二监听端口。

本机报告：data/asr/streaming-verification-cuda.json 和 streaming-verification-cpu.json。
