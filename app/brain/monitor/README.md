# 按需 UI 与系统监控服务

访问 `http://<Jetson-IP>:8001/` 打开主 UI；需要远程麦克风时使用 HTTPS `8443`。systemd
socket 常驻监听；首次请求自动启动 `server.py`，最后一次 UI 请求后 90 秒自动退出并终止 `tegrastats`。

核心实现：CPU/RAM 读取 `/proc`；GPU 利用率、GPU 温度与 CPU 温度由一个持续运行的
`tegrastats --interval 2000` 解析，避免每次网页请求重复启动进程。服务只代理
`/health`、`/v1/models` 和 `/v1/chat/completions` 到 `127.0.0.1:8080`，并将 `/api/asr`
代理到 `127.0.0.1:8000`；模型和 ASR 均保持本机监听。

接口：`/api/system/stats`、`/api/monitor/health`。Jetson 为统一内存架构，不返回伪造的
独立 VRAM；UI 显示 GPU / CPU 温度，单位 `℃`。

排障：

```bash
sudo systemctl status gemma4-system-monitor.socket gemma4-system-monitor.service
curl http://127.0.0.1:8001/api/system/stats
sudo journalctl -u gemma4-system-monitor.service -n 100 --no-pager
```

安装单元文件使用 `scripts/install-system-monitor.sh`。若扩展硬件控制，请复用该服务的
8001 端口与路由；不要部署第二个监听 8001 的进程。
