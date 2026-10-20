# X2 明日下地行走 Runbook（2026-10-21）

编制：2026-10-20 深夜（接班 Agent 第二轮准备）。目标：按 Sonic-X2 v0.2 固定提交
（c0036a4）的作者方法，最小进程集完成 吊起接管 → 承重平衡 → 键盘 KPlanner 行走。
每个阶段完成后立即记录证据并等用户派发下一阶段；不跳阶段、不自批。

## 已就绪（2026-10-20 23:5x 核验）

| 项 | 状态 | 证据/哈希 |
| --- | --- | --- |
| 重启后基线 | 双 SoC 22:48 开机，官方进程齐全，四组 command 仅官方 mc_ros2_node 单 writer，反馈新鲜 | round1 observer JSONL |
| 本 boot 厂商日志 | 零 SHM 警告、零 IMU Publish 卡顿 | `logs/x2_round1_reboot_verify_20261020/` |
| 模型 | `policies/x2_sonic_s1ft16000_g1.onnx` SHA256 7dfdec7c…59a0（与作者 shipped set 一致） | F07 + 实测 |
| robot_env.env | 已按 frozen 模板钉住：s1ft16000 + trained_gains_s0.yaml + vendor_20260823 + body anchor + WB=0 | 逐行核对 |
| 250Hz 候选 | QvhMAD install 完整，binary SHA256 2f159036…6ffa | 实测 |
| 上游固定提交 checkout | SoC1 `~/projects/x2_sonic_v02_candidate_c0036a4`（含 docs/x2 全文） | 已抓取关键文档 |
| 作者配方确认 | 固定提交 ritual = `--wrist-bypass ik` + `trained_gains_s0`（与 SoC1 现状一致，无需改） | 上游源码/模板核对 |
| planner 模型 | `planner_stack/models/planner_onnx/x2_planner_template.onnx` SHA256 190055c7…7bbe0 | 实测 |
| kb 变体 ritual | `ritual_start_kb.sh`：去 pad 链、kplanner `--cmd-bind --command-watchdog-s 0.3`、pose gate | 全文审计 |
| 键盘桥 | 本仓库 `script/project_tools/candidates/kbwalk_bridge_20261020/`，12/12 单测 + ZMQ 线格式冒烟通过 | 本轮 |

## 根因结论对明日的影响（round 1 证据）

四次 powered 接管全部在 **stop_app 停止 MC 期间有新 DDS 参与者加入（spawn+discovery）**
的时刻伴随 ~2s IMU Publish 阻塞 → 反馈空窗 → 原生 stale。作者 deploy_x2.sh 的顺序相反：
**deploy 先以 writer 抑制 standby 在 MC 运行时启动（实体先建、匹配完成、ONNX 先载），
确认后 stop_app → 触发 sentinel → CONTROL**。明日必须用作者顺序，不再用 supervisor 的
stop→spawn→trigger 逆序。15:30 no-publish（spawn 时 MC 运行）与 15:02 SIGSTOP（无新参与者）
均零阻塞，与该结论一致。

## Stage 0（今晚，可选，需授权）：MC 切换隔离测量

目的：测出 stop_app 单独作用（无任何自定义进程、观察者预先匹配）是否造成反馈空窗，
为明日作者顺序的预期窗口提供数值。机器人完全吊起、现场就绪后执行一次：

1. 双端起 60s 只读 observer（探针 02eeff…）。
2. 从 SoC1 `curl -X POST -H 'Content-Type: application/json' -d '{"app_name":"mc"}' http://10.0.1.40:50080/json/stop_app`（与 deploy_x2.sh 同机制）。
3. 等 3s，再 POST start_app。
4. 结束后核对：observer 每秒计数/最大 gap、IMU cost 行、MC fastdds、四组 command 归属、MC 日志恢复。
判读：stop_app 单独不空窗 → 作者顺序预期顺畅；空窗 ~2s → 明日 trigger 前必须等反馈恢复（作者流程的 stop 验证已含此逻辑，仍需盯 observer）。

## Stage 1（明日，吊起）：最小链路 + 作者顺序首次接管

最小进程集（用户已确认不必要的不开）：`x2_pose_watchdog` + `pc2_kplanner` + `x2_deploy`。
不开：hand bridge、motor monitor、thermal、face、scan guard（ritual 中已默认关）。

```bash
# SoC1 tmux: 1) watchdog（必须先起，own :5558）
bash /agibot/data/home/agi/gear-sonic/start_x2_pose_watchdog_local.sh
# 2) kplanner（与 ritual_start_kb.sh 同参数；PYTHONPATH=planner_stack；. robot_env.env）
#    --cmd-bind --command-watchdog-s 0.3 --planner-mode slow_walk
# 3) pose gate：SUB :5558 收到 pose 帧才继续（照抄 ritual 的 15s 探针）
# 4) deploy（作者顺序；交互确认，不 --no-confirm）：
cd /agibot/data/home/agi/gear-sonic/gear_sonic_deploy && ./deploy_x2.sh onbot --no-docker --vla \
  --vla-zmq-host localhost --vla-zmq-port 5558 --vla-zmq-topic pose \
  --vla-debug-port 5557 --vla-debug-topic x2_debug \
  --wrist-bypass ik --head-bypass ref --max-target-dev-head 0.40 \
  --model /agibot/data/home/agi/gear-sonic/policies/x2_sonic_s1ft16000_g1.onnx \
  --intra-op-threads 3 --tuning-config .../trained_gains_s0.yaml \
  --onbot-prefix /agibot/data/home/agi/gear-sonic --onbot-ws <见下> \
  --onbot-venv .../venv --onbot-onnxruntime .../onnxruntime \
  --onbot-aimdk-prefix /agibot/data/home/agi/aimdk_msgs_pc2
```

binary 选择（需用户拍板）：A) 原 ws install（作者原配，writer 500Hz/2ms，0b02df14…）；
B) QvhMAD 候选（writer 250Hz/4ms，2f159036…，`--onbot-ws /agibot/data/home/agi/x2-v02-writer250-resume.QvhMAD/install`）。
历史依据：8 月 D-002 记录本机 500Hz 状态拥堵；22:0x 两次尝试用的是 B。两案都保留原生保护。

验收（吊起窗口 3–5 s）：deploy 面板 `Loaded ONNX: .../s1ft16000_g1.onnx` + `[plant] RUNTIME
PLANT LOADED: vendor_20260823`；CONTROL 后 tick ~50Hz 无 stale；GID 审计四组 command 仅 deploy
一个 publisher；软停 = tmux 窗口内 Ctrl-C（SIGINT → RAMP_OUT → HOLD_FOR_MC → 自动 start_app），
禁止 tmux kill-session；恢复后 observer 复核四组 command 回到官方单 writer。

## Stage 2（吊起）：停止契约复验

同 Stage 1 连续 3 次 接管→软停→官方恢复 循环，记录每次：反馈空窗时长（若有）、tick p95、
恢复耗时、MC 日志。3 次全部干净才进入 Stage 3。

## Stage 3（接地承重，需现场龙门/保护 + 当次授权）

先在吊起状态验证 deploy 承重停止保持（HOLD 静态），再由现场缓慢下放接地；短时自主承重
（建议首窗 ≤10 s），观察后软停恢复。任何时候反馈 stale/异响/踮脚/大幅摆动 → 软停；
紧急 → 物理急停。不落地行走。

## Stage 4（接地轻扰动）

承重稳定后单独派发：小幅推扶/单侧负载，验证恢复，软停。不行走。

## Stage 5（键盘 KPlanner 行走，需当次授权 + 龙门保护 + 独立急停）

1. Mac 建 planner 隧道：`ssh -N -L 127.0.0.1:5563:10.0.1.41:5563 sshuser@100.92.227.62`。
2. 首验收用**定时指令**（不用交互键盘）：
   `python3 kbwalk_bridge.py --endpoint tcp://127.0.0.1:5563 --script 'idle 2' 'walk 2 0.4' 'idle 3'`，
   观察：起步、行走 ~2s、松键停稳、回到静止。0.4 为归一化意图（planner 侧 fixed fwd 0.35 m/s）。
3. 通过后才用 `--interactive`（pygame 窗口，真实 keydown/keyup/失焦清零；SPACE=deadman，
   W/S/A/D，X=立即零意图，Q=停稳后退出）。planner 侧 --command-watchdog-s 0.3 已兜底桥死亡。
4. 行走中桥/隧道断 = watchdog 0.3s 自动停止链；物理急停始终最高优先。

## 全程红线

- 不放宽任何原生门限；不加比固定上游更严的算法门限。
- 单 writer：接管期 deploy 唯一、其他时间官方 MC 唯一；每次交接前后 GID 复核。
- 承重状态绝不 tmux kill-session / kill writer；只走 SIGINT 软停链。
- 不 clear /dev/shm、不重启 HAL、不覆盖厂商 XML、不升级固件。
- 每阶段结束立即写 README + 证据，等用户审阅派发。

## 证据文件索引

- round1：`logs/x2_round1_reboot_verify_20261020/`（重启核验 + 五 boot 考古）
- round2 准备：`logs/x2_round2_prep_20261020/`（SoC1 脚本快照、上游 docs、12:17 stale json）
- 键盘桥：`script/project_tools/candidates/kbwalk_bridge_20261020/`（源码+测试+冒烟日志）
