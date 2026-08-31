# X2 智能服 -> Sonic -> 真机系统总览与实施计划

+ 文档版本：v1.2
+ 更新日期：2026-08-31
+ 最终目标：让 X2 在安全约束下通过 Sonic 跟随 6+5 IMU 智能服的全身动作
+ 当前里程碑：完成“受支撑、穿衣者静止、10 秒平滑融合”的第一段真机衣服闭环

## 0. 先读结论

项目已经不再卡在衣服、蓝牙、HMCP、3588S 算力、ZMQ、Sonic 推理、MC/HAL
接管或固定 StandStill 的短时稳定性。上述链路分别已有实测证据。

原先唯一直接代码缺口是：C++ deploy 在 `GROUND_LOAD_HOLD`（Policy OFF、静态
PD ON）时没有发布只读 `x2_debug`。该缺口已于 2026-08-31 完成 SoC1 隔离构建、
精确候选 process/network/wire/unit 验证，并让代理真实达到
`STANDSTILL_READY`。当前剩余阻塞是用同一候选完成最终父版本 300 秒验收并冻结
manifest，之后才进入第一次带电衣服闭环。

当前阶段不追求走路、不追求无吊绳站立，也不追求明显动作。第一次成功的定义只是：

```text
固定 StandStill 已稳定
-> 安全代理确认穿衣者静止且机器人状态新鲜
-> 显式 ARM
-> 10 秒 smoothstep 从 StandStill 融合到静止衣服参考
-> X2 在既定支撑下保持有界
```

## 1. 文档边界与事实优先级

### 1.1 事实优先级

出现冲突时，按以下顺序判断：

1. 2026-08-30 最新操作者交接和
   `/Users/yu/Documents/ChatGPT/X2/standstill_official_ab_20260830/README.md`；
2. `README_X2_GARMENT_POWERED_INTEGRATION.md`；
3. `README_X2_GARMENT_LIVE.md`、`README_X2_REFERENCE_OFFLOAD.md` 和
   `README_X2_GARMENT_ZMQ.md`；
4. `../docs/CURRENT_STATE.md`、`DECISIONS.md`、`OPERATIONS.md`、
   `EXPERIMENTS.md` 与历史 README。

旧文档中“官方 MC 正在运行”“StandStill 入口仍未解决”“衣服吞吐未达到 30 Hz”
等状态已经过期；其中架构、安全门、单 writer、日志和退出规则仍然有效。

### 1.2 当前项目边界

| 范围 | 本阶段是否包含 | 说明 |
| --- | --- | --- |
| 智能服实时输入 | 是 | 上衣 6 IMU、裤子 5 IMU，V2 BLE |
| 3588S 人体参考 offload | 是 | postprocess、Fast-SMPL、native GMR |
| Sonic 自研底层运控 | 是 | 固定模型、50 Hz policy、250 Hz HAL writer |
| 官方 MC 作为平衡控制器 | 否 | 自定义 writer 工作时官方 MC paused |
| 受支撑固定 StandStill | 是，已完成短时验收 | 证明受支撑稳定，不证明自主站立 |
| 静止穿衣者真机闭环 | 是，当前目标 | 先完成 10 秒融合和短时保持 |
| 小幅全身动作 | 后续单独验收 | 只有静止闭环通过后才开放 |
| 无吊绳站立、推搡恢复、行走 | 否 | 属于后续独立课题 |

## 2. 阶段总览

| 阶段 | 目标 | 当前状态 |
| --- | --- | --- |
| S0 | v0.9 上暂停 MC、证明命令静默、独占 HAL、恢复 MC | 已完成 |
| S1 | 固定 StandStill 在 50/250 Hz 下受支撑稳定 | 多次 30 秒已完成；最终候选 300 秒待做 |
| S2 | 衣服 -> 3588S -> HMCP -> Sonic 全链路 dry-run | 已完成，35.02 Hz |
| S3 | 原始参考安全代理、静止门、yaw 对齐、10 秒 blend、LOCKOUT | 离线实现和测试已完成 |
| S4 | Policy OFF 时向代理提供实时 `x2_debug` | 已完成；SoC1 精确候选和代理网络测试通过 |
| S5 | 冻结最终父版本和批准 manifest | 未完成 |
| S6 | 受支撑、静止穿衣者第一段带电闭环 | 未完成，当前最终里程碑 |
| S7 | 单独的小幅全身动作验收 | 未开始 |

不要再重复相同配置的 30 秒 StandStill，也不要重新优化已通过的衣服传输。下一次
固定参考真机验证是使用最终同一候选二进制和 launcher 的 300 秒验收。

## 3. 最终系统拓扑

```text
上衣 F7:C6:1F:AB:37:E0       裤子 DD:65:A4:4A:22:36
              \                 /
               +---- V2 BLE ----+
                         |
                         v
X2 SoC1 (agi@192.168.43.21)
  AX210 -> TIC/LFP (CUDA)
                         |
                         | pose + velocity，单帧流水线 RPC
                         v
X2 3588S (agi@10.0.1.42)
  physics postprocess -> Fast-SMPL -> native GMR -> G1 qpos36
                         |
                         v
X2 SoC1
  timestamp HMCP -> G1-to-X2 增量映射 -> raw ZMQ :5555
                         |
                         v
  live-reference safety proxy
       ^ robot x2_debug :5557
       | measured base_quat + held SafeCommand
       |
       +---- WAIT_SOURCE / WAIT_ROBOT
             -> WARMUP -> STANDSTILL_READY
             -> explicit ARM -> 10 s BLEND -> LIVE
             -> 任一 post-warmup 错误：终止性 LOCKOUT
                         |
                         v guarded ZMQ :5556
  C++ Sonic deploy 50 Hz
       -> 既有 safety/PD
       -> 唯一 HAL writer 250 Hz
                         |
                         v
X2 SoC0 HAL -> 31 个机身关节
  （自定义 writer 存在时官方 MC paused）
```

Mac 只用于 SSH、监控、日志和代码同步。5060/HP3090 只保留为历史仿真与数据来源，
不进入最终实时链路。不得通过 Mac 或 5060 增加中继。

## 4. 冻结的软件与数据契约

### 4.1 机器人与控制

| 项目 | 冻结值 |
| --- | --- |
| 机器人 | X2 Ultra，v0.9.0-rc7 |
| ROS/message ABI | ROS Humble / `aimdk_msgs 0.8.18` |
| 模型 | `x2_sonic_frozen_g1core_lora_v2.onnx` |
| 模型 SHA-256 | `8ccc42a82ea2c446aa708aece58c7a638ea943a79ed88eeab259669e641271e9` |
| policy / writer | 50 Hz / 250 Hz |
| profile | `neutral_damped` |
| IMU | reconstructed pelvis |
| target slew | `0.12 rad/s` |
| 参考父状态 | 固定 `StandStill` |
| HAL ownership | 任意时刻最多一个自定义 writer |

不得再提高 slew，不得调整 gains、LPF、clamp、action scale 或 joint mapping，也不
得切回 500 Hz、`waist2`、v1.0 `Develop_MC` 或官方 Standing 过渡路线。

### 4.2 衣服与参考语义

- 保持 `pose-scale=0.7`；
- 保持按关节名称的 G1 -> X2 增量映射；
- 实时 HMCP 速度使用原始时间戳，不使用接收 wall-clock，也不固定乘 50；
- HMCP 根四元数为 WXYZ，ZMQ/Sonic wire 使用 XYZW；转换前后均归一化；
- `qpos36 = root xyz[3] + root quaternion WXYZ[4] + G1 joints[29]`；
- 不使用 HP3090 PKL、固定 gait、direct-X2 GMR、新 planner 或 predictor；
- T-Pose 只能在终端明确显示 `T-POSE READY` 后由操作者手动输入 `TPOSE`；
- 3588S 断流后停止 HMCP，不允许自动回退到本地 stateful reference。

### 4.3 安全代理契约

原始衣服参考只允许发布到 `:5555`；只有代理能向 Sonic 的 `:5556` 发布。

进入 `STANDSTILL_READY` 必须同时满足：

- raw reference age `<=0.15 s`；
- robot `x2_debug` age `<=0.10 s`；
- `dry_run=0`；
- 至少 50 个被接受的输入帧；
- 连续静止至少 2 秒；
- 不是校准 T-Pose，且关节偏移、速度、root roll/pitch/angular speed 有界；
- 机器人 debug 时间戳持续递增。

`STANDSTILL_READY` 输出的是精确训练 StandStill、零速度和机器人当前 yaw，不是衣服
动作。显式 ARM 后，代理捕获一次 `robot_yaw - wearer_yaw`，并用同一个 10 秒
smoothstep alpha 融合当前关节、速度、根四元数和全部 9 个 future slots。

任何 post-warmup source/debug/wire/stationary 错误都进入终止性 `LOCKOUT`。数据恢复
不能自动重新接管；必须结束本次代理会话并重新完成全部门禁。

## 5. 已完成的工作与证据

### 5.1 v0.9 底层控制链

- [x] SoC0 MC 可以暂停，命令静默可以验证；
- [x] SoC1 自定义 PD 可以独占 HAL command topics；
- [x] 250 Hz writer 下 state feedback 可持续新鲜；
- [x] Policy 50 Hz、writer 250 Hz、state/control/writer callback 已隔离；
- [x] Policy 返回静态 PD、完全吊起后 `lifted` 退出和恢复 MC 的路径已验证；
- [x] 500 Hz 造成 bursty feedback 的回归已定位并冻结为禁止项。

这些证据证明底层接管链能工作，不需要再回到 `Develop_MC` 或官方 Standing 研究。

### 5.2 固定 StandStill

- [x] 历史 `170450` 完成 299.968 秒，但属于旧二进制和较强吊绳支撑；
- [x] 相对入口版本完成 3 次独立 30 秒受支撑探针；
- [x] 最新降低支撑测试在 Policy OFF 时先建立接触并等待稳定；
- [x] 最新入口 pitch/roll 为 `-8.79/-0.03 deg`，踝误差 `0.123 rad`；
- [x] 30 秒末 tilt `0.72 deg`，无增长摆动、踮脚、异响、clip 或安全门触发；
- [x] 已证明 Policy ON 后改变吊绳会成为大扰动，因此运行中不得改变支撑；
- [ ] 最终补丁后二进制和最终 launcher 的同一版本 300 秒验收。

结论仅限“当前支撑布置下可稳定”。没有证明无吊绳自主站立、推搡恢复或行走。

### 5.3 衣服、3588S 与 dry-run

通过日志：`logs/garment_live_dryrun_20260830_135137`。

| 指标 | 实测结果 |
| --- | ---: |
| live source | `35.02 Hz` |
| source p95 / max gap | `30.5 / 46.6 ms` |
| timestamp HMCP | `1377/1377` |
| fallback / missing / derivative reset | `0 / 0 / 0` |
| Sonic | `1601 ticks`，稳定 50 Hz |
| policy clipped ticks | `38/1500` |
| remote processing median / p95 / max | `16.78 / 18.01 / 19.42 ms` |
| 自定义 HAL publisher | `0`，官方 MC 保持 active |

另有以下证据：

- [x] 600 帧固定 replay 的 one-frame pipeline 达到 `59.48 Hz`，与同步结果
  `qpos error=0`；
- [x] 3588S 故意在 frame 500 断开后，HMCP 停止，无本地 fallback；
- [x] Sonic 在最后 HMCP 后约 `0.518 s` 进入 stale 安全状态；
- [x] 实时速度已从固定 50 倍差分改为 timestamp 语义，fallback/missing/reset 均为 0；
- [x] 真实衣服 BLE、T-Pose、CUDA LFP、Fast-SMPL、native GMR、HMCP、ZMQ、
  C++ tokenizer 和 ONNX 50 Hz 已在同一条只读链上跑通。

因此衣服链路当前不是主要阻塞点，不需要继续做吞吐优化。

### 5.4 安全代理

- [x] `WAIT_SOURCE/WAIT_ROBOT -> WARMUP -> STANDSTILL_READY -> ARM -> BLEND -> LIVE`
  状态机已实现；
- [x] post-warmup 错误进入终止性 `LOCKOUT`，不会自动恢复；
- [x] T-Pose、移动输入、stale、wire 错误、时间戳回退和重复 arm 已有覆盖；
- [x] 真实 1379 帧历史 capture 只停在 `WARMUP`，正确拒绝了 T-Pose/移动入口；
- [x] 当前交接记录为 18 项离线单元测试通过；
- [ ] 精确最终 C++ 候选的 replay/network/wire/unit/process 测试待补丁后重跑；
- [ ] 正式 approved manifest 尚不存在，当前只有 `not-approved` 示例。

## 6. 当前机器人状态与收工边界

截至最后一次权威交接，机器人处于：

```text
state: GROUND_LOAD_HOLD
policy: OFF
custom static PD: ON
official MC: paused
tmux: x2_waist_static
runtime log: runtime_suspended/logs/suspended_sonic_20260830_201850
```

旧 PID `12551/12649` 只是历史观察值，下一次不得直接复用。本次只读 SSH 已用于
比较源文件哈希；在操作者确认完全吊起前，不上传、不在 SoC1 编译、不启动
衣服/offload/proxy、不发送控制命令，也不结束 writer。

下一次工作的第一步必须由操作者确认机器人已经完全吊起且双脚离地，然后只读核验
当前会话，再通过现有会话输入 `lifted` 正常结束 writer。不得用 `Ctrl-C`、kill 或
关闭 SSH 代替。结束后必须确认：

- 自定义 writer 已退出；
- 官方 MC 已恢复；
- leg/waist/arm/head 四个 HAL command topic 各只有一个官方 publisher；
- 没有旧 garment/offload/proxy/deploy 进程或旧 sentinel 被复用。

完成清理前不得同步源码、编译新 runtime 或启动第二个 writer。

## 7. Policy OFF `x2_debug` 接口

### 7.1 2026-08-31 补丁与 SoC1 候选状态

本地补丁已完成：`GROUND_LOAD_HOLD` 现在使用 measured `RobotState` 和 held static
`SafeCommand` 发布现有 `x2_debug` schema。发布必须同时满足快照存在、所有 HAL state
新鲜、debug publisher 已启用，避免用持续 debug 帧掩盖缓存机器人状态。

验证结果：SoC1 新 scratch workspace
`runtime_suspended/ws_ground_debug_20260831` 构建成功，直接 CTest 通过 `2/2`；安全
代理和 parent manifest 在本地及 SoC1 都通过 `18/18`，wire 自检和 1379 帧 replay
通过。精确候选 SHA-256 为：

```text
3c4dee77b9bde50c304b28200eea030b900ac55f9efc1473543068dbc9c3559f
```

进程测试使用 `--dry-run`，因此候选未创建 HAL command publisher，官方
`mc_ros2_node` 始终是四组 command topic 的唯一 publisher。候选进入
`GROUND_LOAD_HOLD` 后，代理达到 `STANDSTILL_READY`；guarded StandStill 最大位置
误差 `2.9564e-08 rad`、速度为零，debug ROS timestamp 增量约 `0.019989 s`。源和
debug 结束后分别在 `0.153 s` 与 `0.102 s` 触发终止性 LOCKOUT。完整 CSV 和进程
证据保存在 `runtime_suspended/logs/m1_ground_debug_dryrun_20260831_104650/`。

本地补丁后 SHA-256：

```text
x2_deploy_onnx_ref.cpp
  ea8595a5eb0573e45b501647ee623e50e636911f0ca5501070216eae39d208e9
supported_policy_gates.hpp
  26532959bfaa435f3abde55ca8e2bab8bad141aa36e0e330b1cc4268d855ed62
test_supported_policy.cpp
  bc8532008fe1d9d027b66f35dea35c65a5e0590c6637830335bae9252e69bdd0
```

### 7.2 原始缺口与最终要求

文件：

```text
src/x2/agi_x2_deploy_onnx_ref/src/x2_deploy_onnx_ref.cpp
```

补丁前，`GROUND_LOAD_HOLD` 已经读取 measured `RobotState rs`，并把当前静态
`latest_cmd_` 复制为 `SafeCommand held`，但没有调用 `PublishDebugFrame()`；因此
代理在 Policy OFF 时收不到机器人 `base_quat`。本地补丁已经补上该调用，最终要求
是在 SoC1 精确候选中证明它能持续提供递增时间戳并让代理达到
`STANDSTILL_READY`。

允许的最小修复是：仅在 state fresh 且 `zmq_debug_pub_` 存在时，用 measured
`rs`、当前 `last_action_il_` 和 held static `SafeCommand` 发布现有 schema 的只读
debug frame。不得改变 writer、状态转换、policy、目标、增益、频率或安全参数。

回归测试至少要证明：

- `GROUND_LOAD_HOLD` 能发布 `x2_debug`；
- `base_quat` 来自当前 measured RobotState；
- command 字段来自当前 held static SafeCommand；
- ROS timestamp 语义正确；`dry_run=true` 已由 M1 无 HAL 进程测试验证，
  `dry_run=false` 的真实父版本字段由 M2 300 秒验收验证；
- Policy OFF 时 control tick 可以不变，但 debug timestamp 必须递增；
- `SAFE_IDLE` 和 `CONTROL` 的既有 debug 行为不回归。

构建必须进入新的隔离 scratch install，不得覆盖任何现有 runtime。由于迁移树中有
重复 package 备份，使用精确 build 目标并直接运行 `ctest`，随后用该精确候选重跑
proxy unit、replay、wire、network 和 process tests。此阶段不得创建 HAL publisher。

## 8. 后续唯一主线

### M0：安全结束当前静态 PD 会话

- [x] 操作者确认完全吊起、双脚离地；
- [x] 只读核验当前 tmux/process/state；机器人已重启，旧会话不存在；
- [x] 因旧 writer 已随重启退出，不发送无目标的 `lifted`；
- [x] 官方 `mc_ros2_node` 为四个 command topic 的唯一 publisher，反馈在线。

里程碑：`M0_CURRENT_SESSION_CLEAN`，2026-08-31 只读验证通过。

### M1：补齐 Policy OFF 的机器人 debug

- [x] 对比本地与 SoC1 源码 SHA-256，确认补丁前完全一致；
- [x] 只做 `GROUND_LOAD_HOLD -> PublishDebugFrame` 最小控制路径改动；
- [x] 增加 stale-safe 回归测试并通过本地离线测试；
- [x] 在 SoC1 新 scratch install 隔离构建并直接运行 C++ tests；
- [x] 用精确候选重跑代理 unit/replay/wire/network/process tests；
- [x] 证明测试阶段没有 HAL publisher，清场后仍为官方单 publisher。

里程碑：`M1_PREPOLICY_DEBUG_READY`，2026-08-31 通过。

### M2：最终固定父版本验收

- [ ] 用 M1 的同一 deploy binary、同一 model、同一 powered/profile launcher；
- [ ] 从完全吊起进入静态 PD；
- [ ] Policy OFF 时建立接触和支撑，等待稳定；
- [ ] Policy ON 后不再改变吊绳、脚位或支撑；
- [ ] 完成一次 300 秒受支撑 StandStill；
- [ ] 保存完整日志、物理支撑描述、哈希和最终 publisher ownership。

历史 `170450` 是旧二进制，不能冒充这次最终父版本证据。不要再增加一组相同的
30 秒探针。

里程碑：`M2_FINAL_FIXED_PARENT_300S_ACCEPTED`。

### M3：冻结批准父版本

- [ ] 固定 deploy binary、ONNX model、powered launcher、profile launcher SHA-256；
- [ ] 创建 `approved_manifests/x2_fixed_standstill_parent.json`；
- [ ] 引用既有独立 30 秒证据和 M2 的最终 300 秒证据；
- [ ] 写入真实审批人、时间、范围和说明，不得伪造 `approved_by`；
- [ ] 让 validator 对全部文件哈希和证据通过。

里程碑：`M3_SUPPORTED_PARENT_APPROVED`。

### M4：组装最终实时链路

按顺序启动：

```text
3588S reference server
-> SoC1 garment source
-> 终端提示后人工 TPOSE
-> raw reference :5555
-> safety proxy :5555 + x2_debug :5557 -> guarded :5556
-> approved powered parent（从完全吊起开始）
```

机器人在 Policy OFF 时建立接触/支撑并等待稳定。代理必须真实经过：

```text
WAIT_SOURCE / WAIT_ROBOT -> WARMUP -> STANDSTILL_READY
```

不得绕过 proxy、manifest 或 `STANDSTILL_READY`，不得使用已故意 `exit 4` 的旧直接
powered launcher。

里程碑：`M4_LIVE_PIPELINE_STANDSTILL_READY`。

### M5：第一次受支撑衣服真机闭环

1. 穿衣者保持静止；
2. X2 先在固定 StandStill Policy 中稳定；
3. 显式 ARM，一次性捕获 yaw 对齐；
4. 完成 10 秒 smoothstep blend；
5. blend 和随后短时保持中，穿衣者不做主动动作；
6. 观察 tilt、速度、跟踪误差、踮脚、异响、支撑和 proxy 状态；
7. 正常结束时先回到静态 PD，再完全吊起，最后 `lifted`。

通过标准：10 秒 blend 完成，代理不进入 LOCKOUT，机器人无增长摆动、踮脚、异响、
速度增长或安全门触发，且全程只有一个 HAL writer。

里程碑：`M5_SUPPORTED_STATIONARY_GARMENT_LOOP_ACCEPTED`。

### M6：单独的小幅动作

只有 M5 通过后，另开一次会话测试短时、小幅全身动作。每次只开放一类动作，并保留
同样的支撑、门禁、停止和日志规则。不得在第一次静止闭环中顺便尝试走路、深蹲或
明显重心转移。

## 9. 停止与恢复规则

- Policy ON 后绝不调整吊绳、机器人高度、脚位或接触；
- 发现 stale、LOCKOUT、摆动增长、踮脚、异响、速度增长或操作者担忧，立即请求
  `stop`，让系统有界返回静态 PD；
- 机器人承重或部分接触时，不得 `Ctrl-C`、kill、关闭 SSH 或发送 `lifted`；
- 回到静态 PD 后，先完全吊起并确认双脚离地，再发送 `lifted`；
- 危险运动时物理急停优先于所有软件流程；
- 电池更换或重启会使 calibration epoch、proxy session、arm request、sentinel、
  debug frame 和 PID 全部失效，必须重新审计；
- 任意时刻只能有一个 HAL writer；发现 ownership 不明确时停止启动新进程，只做
  只读诊断。

## 10. 每阶段必须保存的交付物

- 精确命令、开始/结束时间和操作者确认；
- deploy、model、launcher、proxy 和配置 SHA-256；
- 原始衣服/HMCP/ZMQ/robot debug/Sonic/HAL 日志；
- state transition、LOCKOUT/stop/return 和 publisher ownership 时间线；
- 机器人支撑、脚接触、吊绳松紧及测试中是否变化的文字记录；
- telemetry 摘要和现场视频；
- 成功、失败、停止原因和最终恢复状态；
- 当次结论能证明什么、不能证明什么。

每完成一个阶段，立即更新：

```text
../docs/EXPERIMENTS.md
README_X2_GARMENT_POWERED_INTEGRATION.md
README_STAGE_SONIC_BALANCE.md
/Users/yu/Documents/ChatGPT/X2/standstill_official_ab_20260830/README.md
```

仓库当前很脏，不得 `reset`、`clean`、全局 reformat 或用 Git HEAD 代表运行时身份。
所有候选均以文件 SHA-256 为准。

## 11. 关键路径

```text
本地仓库：
/Users/yu/projects/sonic_x2_transfer_v2/GR00T_audit/gear_sonic_deploy

SoC1 部署：
/agibot/data/home/agi/projects/x2_sonic_migrated_20260826/
  sonic_x2_transfer_v2/GR00T_audit/gear_sonic_deploy

当前 C++ deploy：
src/x2/agi_x2_deploy_onnx_ref/src/x2_deploy_onnx_ref.cpp

安全代理：
scripts/garment_zmq/gate_live_reference.py

衣服 source launcher：
run_x2_garment_offload_source.sh

ARM gate：
arm_x2_supported_garment_live.sh

批准 manifest：
approved_manifests/x2_fixed_standstill_parent.json

当前机器人日志：
runtime_suspended/logs/suspended_sonic_20260830_201850
```

## 12. 当前 Go / No-Go

| 项目 | 结论 |
| --- | --- |
| 衣服和 3588S 继续做 dry-run | GO，但无需重复优化 |
| 修改/构建当前运行中的父版本 | NO-GO，先完全吊起并 `lifted` 清理 |
| 最小补齐 `GROUND_LOAD_HOLD x2_debug` | GO，清理后隔离执行 |
| 用最终候选做固定 StandStill 300 秒 | GO，M1 全部测试通过后 |
| 直接把 raw 衣服参考送进 Sonic | NO-GO |
| 绕过 proxy 或伪造 manifest | NO-GO |
| 第一次静止衣服 10 秒 blend | GO，M2-M4 全部通过后 |
| 无吊绳站立、走路或大幅动作 | NO-GO，当前不在范围内 |

一句话状态：**上游衣服计算和下游受支撑 Sonic 都已经分别跑通；现在只差把 Policy
OFF 的机器人姿态送进安全代理，冻结最终父版本，然后完成第一段静止衣服真机融合。**
