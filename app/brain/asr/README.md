# 按需语音识别（ASR）

浏览器录音经 8001/8443 同源代理发送到本机 8000 网关。停止录音后浏览器将完整音频转换为
16 kHz WAV 上传；识别文本完成后自动填入 UI 输入框，以完整上下文优先保证准确率。

网关按需启动 whisper.cpp（内部仅监听 127.0.0.1:8002），加载
`data/asr/whispercpp/ggml-base.bin` 多语言模型。语言自动检测；系统安装 OpenCC 时，结果用
`tw2sp` 转为大陆简体。连续 180 秒没有 ASR
请求时，网关和 GPU 子进程一起退出，释放资源；下次录音会由 socket 自动唤醒。

当前已构建并验证的是 `third_party/whisper.cpp/build-cpu/bin/whisper-server` CPU 后端。
网关在完整 CUDA 程序存在时优先选用 `build/bin/whisper-server`，否则选择 CPU 后端。
CUDA 全量构建尚未完成，中间文件保留在 `build/`；CPU 后端无需 Python Whisper 或 PyTorch。

若 whisper.cpp 程序或模型文件缺失，代码尝试使用 Python Whisper CPU 回退；该路径需要另外安装
`openai-whisper` 及其依赖，不能仅凭存在网关代码就认为回退可用。CUDA 子进程启动失败时会返回错误。

## 当前用户级服务

完成程序构建及模型下载后，可使用用户级 socket 服务，无需 sudo：

```bash
bash /home/nvidia/Desktop/gemma4_E4B_for_VLA/scripts/install-asr-user.sh
systemctl --user status gemma4-asr.socket gemma4-asr.service
journalctl --user -u gemma4-asr.service -n 80 --no-pager
```

该 socket 仅监听本机 `127.0.0.1:8000`。首次请求唤醒服务；服务因空闲退出后 socket 继续监听。
只有 socket 为 active、service 为 inactive 是正常的待命状态。不要同时启用同端口的系统级 ASR 服务。
用户服务已启用时随用户服务管理器启动；无需登录的开机运行取决于系统的用户 linger 配置。

实际转写测试（示例音频为上游自带的 11 秒英文语音，不代表已验证中文识别准确率）：

```bash
curl --fail http://127.0.0.1:8000/health
curl --fail --max-time 150 \
  -F 'file=@/home/nvidia/Desktop/gemma4_E4B_for_VLA/third_party/whisper.cpp/samples/jfk.wav' \
  http://127.0.0.1:8000/api/asr
```

`/health` 仅确认网关可以响应；必须通过 `/api/asr` 的音频请求才能验证模型可用。

## 浏览器麦克风测试

网关根路径提供独立的录音测试页，仅调用 Whisper，不向 Gemma 发送消息或控制设备。
Jetson 本机可打开 `http://127.0.0.1:8000/`。

其他电脑在终端（Windows 可使用 PowerShell）建立到 Jetson 的 SSH 隧道：

```bash
ssh -N -L 18000:127.0.0.1:8000 nvidia@192.168.1.204
```

IP 改为实际 Jetson 局域网地址。完成 SSH 登录后保持终端打开，在该电脑浏览器访问
`http://localhost:18000/`，允许麦克风，点击开始录音、说话、停止并识别。
localhost 满足浏览器麦克风的安全上下文要求；无需新增局域网端口或自签名证书。
关闭 SSH 终端或按 Ctrl+C 即断开隧道。隧道只转发 ASR 本机端口。

局域网麦克风应从已部署的 `https://<Jetson-IP>:8443/` UI 打开，并信任自签名证书。
如果另外选择系统级服务安装，查看状态使用下面命令（与上面的用户级命令不同）：

```bash
sudo systemctl status gemma4-asr.socket gemma4-asr.service
sudo journalctl -u gemma4-asr.service -n 100 --no-pager
```
