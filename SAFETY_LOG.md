# 安全日志

- 2026-07-28：仅只读访问旧 X2/SONIC 工程与 checkpoint。
- 2026-07-28：未启动真机、未连接 SDK、未发送机器人命令。
- 2026-07-28：未重启系统、未重载 NVIDIA 驱动。
- 2026-07-28：所有训练与派生 checkpoint 仅写入 `ZHY/CWI_CrossEmbodiment_Sim`；旧 Stage152-B 与旧脚本均未修改。
- 2026-07-28：Stage4/5 仅运行 Isaac 仿真与离线分析；旧 X2 工程继续只读，未启动真机、未生成新 checkpoint。
- 2026-07-28：IMU 航向回退在仿真中出现反向安全效果后按预注册停止；未将该逻辑部署到任何实机。
- 2026-07-28：Stage6 仅在 Isaac CPU compatibility mode 训练/评估；未连接真机、未调用 SDK、未发送机器人命令，旧 X2/SONIC 工程保持只读。
- 2026-07-28：Stage6 留出 wave 出现方向/倾角回归后按 Gate25 停止，不解锁 SONIC 注入或部署。
