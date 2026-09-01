# 按需语音识别（ASR）

浏览器录音经 8001/8443 同源代理发送到本机 8000 网关。停止录音后浏览器将完整音频转换为
16 kHz WAV 上传；识别文本完成后自动填入 UI 输入框，以完整上下文优先保证准确率。

网关按需启动 CUDA 版 whisper.cpp（内部仅监听 127.0.0.1:8002）。CPU 回退默认使用已下载的
多语言 `base` 模型；语言自动检测，结果用 OpenCC `tw2sp` 转为大陆简体。连续 180 秒没有 ASR
请求时，网关和 GPU 子进程一起退出，释放资源；下次录音会由 socket 自动唤醒。

若 CUDA 后端尚未完成首次编译，服务会暂时回退到已安装的 CPU `base`，保证录音功能不失效；
CUDA 文件出现后的下一次按需启动会自动优先使用 CUDA。

局域网麦克风必须从 `https://<Jetson-IP>:8443/` 打开 UI 并信任自签名证书。查看状态：

```bash
sudo systemctl status gemma4-asr.socket gemma4-asr.service
sudo journalctl -u gemma4-asr.service -n 100 --no-pager
```
