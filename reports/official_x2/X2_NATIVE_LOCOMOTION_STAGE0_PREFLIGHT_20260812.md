# X2 原生 locomotion 路线：阶段0新机预检

日期：2026-08-12
范围：路线图任务一之前的只读资产、复现和容量审计；未运行物理门禁或 optimizer。

## 裁决

`Stage264` 是 Task50 的长训就绪裁决名称，不是独立 checkpoint。其标称后端由以下组合构成：

- Stage219 `model_2600.pt` / `stage219_s2600_actor.onnx`；
- Stage250 修正后的 actor clip、last-action feedback 与转向部署契约；
- 官方 X2 MJCF、`official_kp_ankle` PD profile、Stage208 default pose；
- 40 帧 gait template、stand backend 和既有门禁阈值。

本机当前 `deploy_ready=false`、`training_ready=false`。机器可读证据见
`reports/official_x2/x2_native_locomotion_baseline_preflight_20260812.json`。

## 已验证资产

| 资产 | 结果 |
|---|---|
| Stage219 actor ONNX | SHA-256 `b95bad36…c0f9`，通过 |
| 官方 SDK zip | SHA-256 `5bbcf724…ab35`，通过 |
| 官方 `x2.xml` / `scene.xml` | SHA-256 `3ff43f05…1a3` / `7fceb3e1…b63`，通过 |
| stand ONNX | SHA-256 `edb73c7c…7565`，通过 |
| Docker image | `sha256:52406d45…b7cc`，存在；需经 `sg docker` 访问 |
| GPU | RTX 3090 24 GB，可见 |
| 静态回归 | conda `x2-sonic-isaaclab`，13/13 通过 |

## 缺失且不可替代的资产

Stage219 源 checkpoint 最初缺失，随后从已校验迁移归档
`x2_phase42_dual_track_20260809.tar.gz` 定点恢复，实际 SHA-256
`abcd49a8…9bb` 与历史清单完全一致。

仍缺失且不可替代：

1. `x2_official_forward_gait_phase_template_15dof.npz`，期望 SHA-256
   `16d77b38…f1d`。本机目标路径缺失；部署仓库中同名对象为空目录。
   原始重建输入也不完整，因此当前不得生成近似替代品。没有它不得复跑
   `24/24` 官方闭环。
2. Phase40 live-zero launcher 的 source checkpoint
   `/home/yu/humanoid-GPT/A/sonic_release/last.pt` 缺失。seed 与 Bronze motion
   分别以 SHA `4bd8bc42…ad2` / `06b759e7…4a6` 存在，但尚不足以运行 smoke。

`tools/official_x2/run_official_gate_case.sh` 已加入普通文件检查，会在启动
Docker/ROS 前拒绝空目录或缺失模板，避免产生无效日志与磁盘垃圾。

## 容量门禁

- 根分区：468 GB，总可用约 363 GB，使用率 19%。
- 当前仓库约 206 MB；项目日志约 62 MB；旧 x2_sonic 日志约 190 MB；迁移
  checkpoint 约 330 MB。
- 任务一只允许小型 JSON/MD/MP4 与精确冻结资产；不下载 AMASS。
- PPO 必须先 1–64 env / 0–5 iter smoke，再测 1024/2048/4096；每次记录峰值
  显存、wall time、输出字节数。任何单 run 预估超过 20 GB 前先做容量审计。
- checkpoint/ONNX/Silver 数据产出立即 Git（仅代码/manifest）+ 百度双备份；
  未完成远端 SHA 核验前不删除本地证据。

## 下一解锁条件

1. 从可信原机/备份恢复上述 checkpoint、gait template 和 Phase40 source；
2. 每项 SHA 必须与历史清单完全一致；
3. preflight 变为 `deploy_ready=true`、`training_ready=true`；
4. Phase40 live-zero 通过后才运行任务一官方 24/24 复验；
5. 基线冻结完成后才进入 signed-pitch A/B/C 训练。
