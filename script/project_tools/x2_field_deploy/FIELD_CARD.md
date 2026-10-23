# X2 键盘行走 · 现场部署任务书（悬空首发）

日期: 2026-10-23 · 适用: 吊装悬空 · 固件 v1.1.0 · SoC1=10.0.1.41(agi) SoC0=10.0.1.40(run)
链路: 键盘 :5563 planner_cmd -> pc2_kplanner(template, 50Hz pose) :5556 -> x2_pose_watchdog :5558 -> deploy(500Hz 写入)

## 0. 开工前检查（学弟逐项打勾）
- [ ] 机器人悬吊稳妥，四脚离地 ≥10cm，吊带不挡髋关节
- [ ] **急停手持有人就位**（真实急停按键全程在手）
- [ ] 笔记本能 SSH 两台 SoC（T1/T2=agi@10.0.1.41，T3=run@10.0.1.40）
- [ ] 无人在机器人 1.5m 半径内
- [ ] （可选）SoC1 预检:  ls ~/gear-sonic/policies/x2_sonic_s1ft16000_g1.onnx 存在;
      ~/gear-sonic/gear_sonic_deploy/configs/real_deploy_tuning/frozen_g1.yaml 存在

## 0.5 上电后先跑一次无风险预检（一次性,~2 分钟）
```bash
# 电池恢复、SSH 通了之后, 先在 SoC1 上:
bash ~/walk/preflight_link.sh
```
等它五步全 OK（kplanner 实机起、5556 流 3936B 帧、键盘被实机接收）。
该预检**不碰 deploy 不碰 MC，机器人零运动**。通过后再走下面的正式流程。
（脚本由工程师在电池恢复后部署到 ~/walk/）

## 1. 三个终端，按顺序执行

### T3（第一个动!）— SoC0 冻结主控，腾出指令权
```bash
ssh run@10.0.1.40
bash ~/walk/pause_mc.sh hold 7200
```
等它打印 `MC identity: pid=... ` 后 `stop_confirmed`（约 20-40s，指令静默期）。
**此窗口 7200s；到点自动恢复 MC。提前收工/中断: 开第二个终端跑 `bash ~/walk/pause_mc.sh release`（或在 T3 窗口按 Ctrl-C）。断 ssh 不会误放行——MC 保持冻结直到期限或 release。**

### T1 — SoC1 点火作者仪式链（watchdog+桥+kplanner+门禁+deploy）
```bash
ssh agi@10.0.1.41
X2_RITUAL_TUNING=/home/agi/gear-sonic/gear_sonic_deploy/configs/real_deploy_tuning/frozen_g1.yaml \
  bash ~/gear-sonic/ritual_start_kb.sh
```
等日志出现 `GATE PASSED`，随后 deploy 自动进入 CONTROL（--no-confirm，
操作确认=pose 门禁，无 y/N 询问）。看各链进度: `tmux ls`（x2_pose_watchdog /
pc2_kplanner / x2_deploy 等会话）；`tmux attach -t x2_deploy` 看 deploy 日志。

### T2 — SoC1 键盘意图源
```bash
ssh agi@10.0.1.41
bash ~/walk/t2_keyboard.sh
```
打印 `PUB tcp://127.0.0.1:5563/planner-cmd` 即就绪。**此时机器人应已收拢双腿、
保持悬空站立——没有人碰键盘它就不会动。**

## 2. 键位（T2 窗口内，单击生效、指令保持）
| 键 | 动作 |
|----|------|
| w  | 前进 0.35 m/s（空中踏步）|
| s  | 后退 0.1 m/s |
| a / d | 原地左转 / 右转 0.65 rad/s |
| space | 归零刹停（回站立）|
| q  | 退出键盘源（先归零再退）|

## 3. 收尾（顺序固定）
1. T2: 按 `space` 再 `q`
2. T1: `tmux send-keys -t x2_deploy C-C`（软停机，坡度归零后退出）
3. T3: Ctrl-C 或 `bash ~/walk/pause_mc.sh release` → 等 `RELEASED`（MC 恢复）
4. 观察机器人 30s 无异动后解除吊装

## 4. 异常处置梯（从轻到重）
1. 步态怪: `space`（归零）
2. 持续异常: T2 `q` → T1 `tmux send-keys -t x2_deploy C-C`
3. 失控/撞吊架: **真实急停**（手持），然后 T3 Ctrl-C 恢复 MC
4. 任何一步卡住: 停在当前步，不要跳步救火；电话工程师

## 5. 本任务书的技术边界（诚实声明）
- kplanner(2026-10-20 跑通过)与 deploy(今日 2/2 验证)的**链路对接是首跑**
- 链路采用作者生产仪式（含 pose 看门狗、电机监控、pose 门禁），但
  tuning 预设用 frozen_g1.yaml 替代缺失的 trained_gains_s0.yaml
- 悬空下按 w = 原地空中踏步，这是预期行为，不是故障
