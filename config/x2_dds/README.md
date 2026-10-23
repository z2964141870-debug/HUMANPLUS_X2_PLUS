# X2 DDS 传输配置（官方契约 500Hz 的承载前提）

根因（2026-10-23 结题）：rclcpp 默认 UDP 缓冲 ~200KB 扛不住
500Hz×4 命令 + 5000msg/s 状态的组合流，接收队列饱和 → 状态断流 →
watchdog E-STOP。100–150Hz"边界"即默认缓冲饱和阈值。

厂商进程全部经 /agibot/software/entry/cfg/basic_env.sh 加载
ros_dds_configuration.xml（UDPv4 收发各 20MB + SHM 20MB/10240 + 网卡白名单）。

## 使用
| 文件 | 用途 |
|---|---|
| ros_dds_configuration_soc0.xml | SoC0 原件（白名单 10.0.1.40），存档参考 |
| ros_dds_soc1.xml | SoC1 部署用（白名单 10.0.1.41），SoC1 落盘于 ~/dds/ |

部署进程（driver 与 x2_deploy_onnx_ref）启动前：
```bash
export FASTRTPS_DEFAULT_PROFILES_FILE=<对应 SoC 的 xml>
# SoC0 上跑时另需: source /agibot/software/entry/cfg/basic_env.sh
```
实测：SoC1 + 该 profile + 500Hz 真发布，run14/15 两轮 0 strikes。
