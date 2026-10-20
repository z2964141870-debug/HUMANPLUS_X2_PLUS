# 第二轮准备（2026-10-20 深夜）：明日下地行走部署准备

本轮零带电动作：SSH 只读盘点 + 本地脚本编写/测试。机器人由现场调节，未触碰任何进程。

## 关键发现

1. **作者 v0.2 键盘链在 SoC1 已基本就绪**：`ritual_start_kb.sh`（10-20 kb 变体）已去掉 pad 链、
   给 kplanner 加 `--cmd-bind --command-watchdog-s 0.3`，等 laptop 端 `kbwalk_bridge.py` 提供意图。
   `robot_env.env` 已按 frozen 模板钉住 s1ft16000 + trained_gains_s0 + vendor_20260823 + body + WB=0。
2. **固定提交核对**：上游 c0036a4 的 ritual 用 `--wrist-bypass ik` + `trained_gains_s0`（F07 的
   “--freeze-wrist”是旧说法；以固定提交源码为准）。SoC1 现状与作者配方一致，无需改。
3. **作者顺序 = 根因对策**：deploy_x2.sh 先以 writer 抑制 standby 启动（MC 运行时完成 DDS 匹配/
   ONNX 载入），确认后才 stop_app → touch trigger → CONTROL。与 22:0x supervisor 的 stop→spawn→
   trigger 逆序相反；round1 考古（15:30 spawn+MC运行=干净；15:02 SIGSTOP=干净；四次 spawn+MC停=2s 阻塞）
   支持把“MC 停止期间新参与者加入”定为触发组合。明日一律用作者顺序。
4. **上游 docs 已本地化**：SoC1 `~/projects/x2_sonic_v02_candidate_c0036a4` 为固定提交 checkout，
   docs/x2 全部抓到本目录 `upstream_docs/`（F01/F07/F09/F10 + PORT_REGISTRY 等）。
5. MC stop/start 机制：HTTP POST `http://10.0.1.40:50080/json/{stop,start}_app}`（`aima em` 底层），
   deploy_x2.sh 内含平滑交接（SetMcAction DAMPING→stop，start→wait→STAND）与 RAMP_OUT 软停链。

## 新增工件

- `script/project_tools/candidates/kbwalk_bridge_20261020/kbwalk_bridge.py`：laptop 键盘/定时意图桥。
  20Hz `planner_cmd` multipart、SPACE deadman、W/S/A/D、失焦清零、Q 停稳退出、退出/异常零值脉冲；
  `--script 'walk 2 0.4'` 定时模式用于可重复验收。12/12 单测 + 本机 ZMQ 线格式冒烟（79 帧）通过。
  本机依赖：pyzmq 27.2.0 + pygame 2.6.1（真 keydown/keyup；pynput/pyobjc 构建失败弃用）。
- `reports/X2_TOMORROW_RUNBOOK_20261021.md`：明日分阶段 runbook（Stage 0 隔离测量 → Stage 1 吊起
  作者顺序接管 → Stage 2 停止契约 ×3 → Stage 3 接地承重 → Stage 4 轻扰动 → Stage 5 键盘行走）。

## 目录内容

| 文件 | 说明 |
| --- | --- |
| `soc1_scripts/ritual_start_kb.sh` | SoC1 现场 kb 变体 ritual 快照（审计用原件） |
| `soc1_scripts/start_x2_deploy_ritual.sh` | deploy ritual 快照（--no-confirm 固化版，明日不用其自动路径） |
| `soc1_scripts/start_x2_pose_watchdog_local.sh` | watchdog 启动 |
| `soc1_scripts/ritual_start_sonic.sh` / `robot_env.env` | 其余 ritual 与生效环境快照 |
| `upstream_docs/*.md` | 固定提交 docs/x2 全套 + PORT_REGISTRY |
| `flight_stale_state_1792469868484.json` | 学弟 12:17 首次 stale 的飞行记录器清单（105 行 @50Hz，与已知 2.08s 一致） |

## 遗留决策（明日开跑前需用户拍板）

1. Stage 0（今晚 MC stop/start 隔离测量）是否执行。
2. Stage 1 用原 ws binary（500Hz/2ms）还是 QvhMAD 候选（250Hz/4ms）。
