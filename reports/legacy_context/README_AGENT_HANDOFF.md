# X2 Sonic 真机侧继承说明

更新时间：2026-08-29 01:20 CST

## 先看这里

当前 Codex 工作目录是：

```text
/Users/yu/Documents/ChatGPT/X2
```

真正要修改的 Sonic 主仓库不在当前目录，而是：

```text
/Users/yu/projects/sonic_x2_transfer_v2/GR00T_audit
```

目标机器人 SoC1：

```text
agi@192.168.43.21
```

机器人上的部署副本：

```text
/agibot/data/home/agi/projects/x2_sonic_migrated_20260826/
sonic_x2_transfer_v2/GR00T_audit/gear_sonic_deploy
```

主仓库工作树很脏，包含大量用户已有修改。禁止 `reset`、`clean`、覆盖
或整理无关文件；禁止提交和推送，除非用户明确要求。远端部署目录不是 Git
仓库，部署必须核对源/目标路径与 SHA-256。

## 本任务的范围

只负责目标 X2 的机器人侧：

```text
localhost ZMQ v5 reference
-> Sonic reference/tokenizer
-> Sonic ONNX policy 50 Hz
-> 机器人状态反馈
-> 安全门控
-> HAL writer 250 Hz
-> X2
```

把衣服、TIC/LFP、SMPL、GMR、HMCP 和 ZMQ publisher 当作已经存在的外部
reference 源。不要重做或评估衣服侧，不要引入 PKL、另一套 planner、固定步态
或 Mac/5060 laptop 中继。

最终运行时 reference publisher 与 Sonic 都在目标 X2 SoC1，使用 localhost。
Mac 只做 SSH、部署和看日志。

## 真正要回答的问题

不是“环境能否安装”，而是：

1. ZMQ live reference 能否被真实 Sonic 正确接收并进入策略；
2. Sonic action 能否经现有安全栈和 HAL 让机器人运动；
3. 在静止 reference 和小幅全身 reference 下，受吊架保护的真机能否保持有界；
4. reference 断流时能否安全、平滑地回到进入 policy 前的静态保持。

传输打通和真机稳定是两个不同结论。不能用“收到 ZMQ”宣称机器人稳定，也
不能用 MuJoCo 结果替代 powered test。

## 已经成立的机器人侧事实

- 固件：`test-lx2501_3_t2d5-soc1-dc-v0.9.0-rc7`。
- 消息 ABI：ROS Humble / `aimdk_msgs 0.8.18`。
- Sonic 在 SoC1；官方 MC/HAL 在 SoC0。
- policy 50 Hz，HAL writer 固定 250 Hz。
- 模型：`x2_sonic_frozen_g1core_lora_v2.onnx`。
- 模型 SHA-256：
  `8ccc42a82ea2c446aa708aece58c7a638ea943a79ed88eeab259669e641271e9`。
- 当前唯一可继承的真机参数族是 `neutral_damped`，使用 reconstructed pelvis
  IMU、wrist freeze、现有增益/LPF/clamp/tilt/velocity gates。
- 一次吊架支持的 StandStill policy 连续运行约 300 秒；同一软件配置在另一
  次接触/入场状态下约 30 秒后出现低频踝/接触振荡并自动返回。
- 因此 HAL 接管、反馈、Sonic 推理、250 Hz 写入和 MC 恢复链已证明可工作；
  任意入场状态的稳定性、动态 live reference 和衣服遥操尚未证明。
- C++ deploy 已有 `ZmqPoseInputSource` 和 ZMQ v5 future-window 解码。
- 已有默认 0.5 秒 pose reference starvation watchdog。

## 当前机器人状态：必须重新核验

最后一次只读观察时，机器人处于：

```text
GROUND_LOAD_HOLD
policy=OFF
custom static-PD writer=ON
SoC0 MC=paused
tilt≈8.61 deg
max joint velocity≈0.007 rad/s
max tracking error≈0.301 rad
```

当时运行进程没有 `--input-type zmq`，使用的是 `StandStillReference`，不能在线
切换为 ZMQ。

这是历史快照，不得假设仍然成立。下一个 agent 首先只能做只读检查。若自定义
writer 仍在运行：

- 禁止发送 `policy`；
- 禁止 `Ctrl-C` 或直接关闭会话；
- 只有现场人员明确确认“机器人已完全吊起”后，才能向原控制会话发送
  `lifted`；
- 随后必须验证自定义 writer 退出、官方 MC 恢复、四组 command topic 回到
  官方 publisher。

若无法访问原控制会话，不要启动第二个 writer；先只读诊断 PID、publisher 和
MC 状态。

## 当前机器人侧代码缺口

### 1. policy 入口没有 live-reference readiness 硬门

位置：

```text
gear_sonic_deploy/src/x2/agi_x2_deploy_onnx_ref/
src/x2_deploy_onnx_ref.cpp
```

`RequestSupportedPolicy()` 目前只检查状态机、HAL feedback 和 startup
feedback stability。ZMQ 模式还应要求：

- `has_body_reference()` 为真；
- 至少收到约 40 个有效 body reference 帧，或等价的连续预热条件；
- `LastReceivedMonotonicS()` 对应的 age 小于 0.5 秒；
- 最好要求连续 fresh 窗口，避免刚恢复一帧就允许 policy；
- 拒绝原因必须打印 reference 帧数和 age。

相关入口约在 `x2_deploy_onnx_ref.cpp:2297`。现有可用诊断接口在：

```text
include/zmq/zmq_pose_input_source.hpp:170  total_frames_received()
include/zmq/zmq_pose_input_source.hpp:184  LastReceivedMonotonicS()
include/zmq/zmq_pose_input_source.hpp:188  has_body_reference()
```

注意 `total_frames_received()` 可能包含非 body 消息；实现前核对
`zmq_pose_input_source.cpp:369` 的递增语义。需要时新增单独的有效 body frame
计数，不能把任意 pose topic 消息当作预热完成。

### 2. SUPPORTED_POLICY 断流返回路径不对

当前公共 watchdog 在 `x2_deploy_onnx_ref.cpp:3486` 左右检查 stale，并跳到
`SAFE_IDLE`，目标是 `default_angles + 4x kd`。

对于吊架支持的 `SUPPORTED_POLICY`，这不符合现有受控返回语义。正确行为应是：

```text
pose age >= 0.5 s
-> EnterSupportedPolicyReturn(now, "pose reference stale")
-> 用现有 2 秒 ramp 回到 supported_static_cmd_
-> GROUND_LOAD_HOLD
-> policy OFF
```

恢复 reference 后不得自动再次进入 policy；必须由操作者重新满足 readiness
和 entry gates 后再次明确请求。

`SUPPORTED_POLICY` 分支约在 `x2_deploy_onnx_ref.cpp:2922`，已有通用返回函数
`EnterSupportedPolicyReturn()`。普通非-supported `CONTROL` 模式可以继续使用
`SAFE_IDLE`，不要破坏它原有的双门恢复设计。

### 3. 缺少专用机器人侧 ZMQ launcher

新增一个清楚命名的 wrapper，例如：

```text
gear_sonic_deploy/run_x2_supported_garment_zmq.sh
```

它应复用 `neutral_damped` 的全部参数，只改变 reference 来源：

```text
--input-type zmq
--zmq-pose-host 127.0.0.1
--zmq-pose-port 5556
--zmq-pose-topic pose
--pose-ref-stale-s 0.5
```

不要修改 model、50/250 Hz、增益、滤波、wrist freeze、pelvis reconstruction、
joint clamps、tilt/velocity gates 或 MC/HAL handoff。launcher 必须有独立确认
token，不能复用模糊的实验 wrapper。

### 4. 缺少机器人侧自动测试

至少覆盖：

- ZMQ 模式无 body reference 时 `policy` 被拒绝；
- 未达到预热帧数时被拒绝；
- 连续 fresh reference 达标后可以进入请求队列；
- `SUPPORTED_POLICY` 中 reference stale 触发两秒 return；
- return 完成后停在 `GROUND_LOAD_HOLD / policy=OFF`；
- reference 恢复不自动重启 policy；
- 普通 `CONTROL` 的既有 `SAFE_IDLE` 行为没有回归；
- launcher 参数只比 `neutral_damped` 多 ZMQ/reference watchdog 参数。

优先把 readiness/transition 判定抽成无 ROS、无 ONNX 的纯函数或小状态类，接到
现有 C++ test target，避免只能靠真机测状态机。

## 推荐执行顺序

1. 只读核验机器人实时状态；若旧静态-PD 会话仍在，等待现场“完全吊起”确认后
   正常 `lifted` 并验证 MC 恢复。
2. 在本地 dirty repo 内实现 reference readiness gate 和诊断，不部署、不启动
   HAL writer。
3. 实现 `SUPPORTED_POLICY` stale -> 两秒 static return，并确保不自动恢复。
4. 添加和运行 C++ 离线测试，保留测试输出。
5. 新增专用 localhost-ZMQ launcher；逐项 diff 确认只改变 reference 相关参数。
6. 构建本地/机器人 runtime，但先做不拥有 HAL command 的 ZMQ dry-run：验证
   reference age、body frame count、50 Hz inference 和 watchdog。
7. 只有前述门全部通过，才请求一次新的现场确认并做 powered test：
   - 机器人完全吊起，急停有人握住；
   - 吊架全程承重；
   - live reference 已预热且持续 fresh；
   - 进入 policy 后穿衣者先静止约 5 秒；
   - 静态有界才做约 10 秒小幅全身动作；
   - 踝振荡、脚尖抬起、异响、reference stale、速度增长或现场担忧时立即
     `stop`；
   - 回到 `GROUND_LOAD_HOLD` 后完全吊起，再 `lifted`，验证官方 MC 恢复。
8. 把源码/二进制/launcher/model SHA、日志位置、结果和最终恢复状态写入
   `docs/EXPERIMENTS.md` 和机器人侧 README。

## 权威文件阅读顺序

进入主仓库后必须按以下顺序完整阅读：

1. `/Users/yu/projects/sonic_x2_transfer_v2/GR00T_audit/AGENTS.md`
2. `docs/CURRENT_STATE.md`
3. `docs/DECISIONS.md`
4. `docs/OPERATIONS.md`
5. `docs/EXPERIMENTS.md`
6. `gear_sonic_deploy/README_X2_GARMENT_LIVE.md`
7. `docs/BASELINE_170450.md`
8. `gear_sonic_deploy/README_STAGE_SONIC_BALANCE.md`

本文件是当前任务的单一入口；仓库内文件保存更完整的实验和安全证据。聊天历史
仅作补充，不应覆盖这些文件。

## 禁止事项

- 不研究、修改或替换衣服到 ZMQ publisher 的链路。
- 不使用 HP3090 `.pkl`、固定步态或另一套 planner。
- 不发送 `policy` 来“看看会怎样”。
- 不在 reference readiness 和 stale return 未测试前接 powered HAL。
- 不同时启动两个 HAL writer。
- 不在部分承重时 `Ctrl-C`、退出 SSH 或发送 `lifted`。
- 不改变 250 Hz writer、模型、增益或 pelvis IMU 模式来掩盖接口问题。
- 不删除失败日志，不把吊架支持结果描述成无保护站立或行走成功。

## 2026-08-31 14:19 CST 临时状态补充

- 机器人已完全吊起；`lifted` 已正常结束自定义 writer，自定义控制进程检查为空。
- SoC0 MC worker 恢复后退出，四组 command topic 一度均为 0 publisher；随后机器人
  网络断开。官方 MC 尚未最终恢复核验，联网前保持吊起，禁止再次落地或启动 writer。
- 当前真机日志为
  `runtime_suspended/logs/suspended_sonic_20260831_134829`。Policy 全程 OFF；该次是
  support-geometry 操作流程失败，不是 Sonic policy 失败。
- 初始 `load` 已满足真正的相对入口门。错误在于把 policy-on 的腰部 support-step
  门提前用于 policy-off 调姿，最终触发 tilt fault。
- 吊带会稳定产生约 `0.28-0.35 rad` 腰部偏差。新的本地候选只取消腰绝对误差的
  support-step 阻断，保留双踝、倾角、速度、趋势和连续稳定门。离线 CTest `2/2`
  通过；尚未 ARM64 构建、同步或授权真机运行。
- 下一次不再在 Policy ON 时松绳。先在 Policy OFF 时设定最终支撑并在 `load`
  捕获，然后使用 `run_x2_supported_preloaded_fixed100.sh` 固定支撑运行 100 秒；
  更低支撑放到下一次独立运行。该 wrapper 目前仅在本机，尚未同步。
- fixed-100 wrapper 已增加纯哈希 `--verify-only`、锁定 `deploy_x2.sh`，并清除旧
  adopt/debug 环境变量；SHA-256 为
  `62cb405d2b6077b1a4ae06ced8c7f1e46a07eed85c74688dfe0b5cdf5c8cbc82`。
  离线 argv 回归测试证明它与 frozen relative-300 除 `100/300 s` 外完全一致，
  且未带入 support-step 或 ZMQ。联网后的第一步仍是恢复并核验官方 MC，不是同步
  后直接启动 writer。当前源码已在全新
  `/tmp/x2_fixed100_offline.JGDrD6` 中完成离线构建，CTest `2/2` 通过。
