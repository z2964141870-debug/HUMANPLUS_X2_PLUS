# X2 Sonic 实时衣服参考:上电前安全闸门(机器人侧)

更新: 2026-08-29 14:26 CST

## 当前状态(先读这一段)

本轮四项改动**已实现、已在机器人 aarch64 上编译通过、单元测试全通过**。

**没有做任何上电测试。** 机器人是否能在实时衣服参考下保持有界,目前仍然是
未知。历史上跑通过的 `neutral_damped` 用的是不带实时参考的
`StandStillReference`(固定默认姿态);带 ZMQ 实时参考的 garment 路径**一次都
没有上电运行过**。不要把本文当作稳定性结论。

覆盖范围只有机器人侧这一段:

```text
localhost ZMQ v5 参考 → Sonic reference/tokenizer → Sonic ONNX 策略 50 Hz
→ 机器人状态反馈 → 安全闸 → HAL writer 250 Hz → X2
```

衣服 → ZMQ publisher 那一段由另一条线并行完成,本文假定参考已可获得。

---

## 1. v5.1 严格参考速度(发送端拥有微分权)

`src/zmq_pose_input_source.cpp` + `include/zmq/zmq_pose_input_source.hpp`

头文件此前已声明 v5.1 语义,但 `HandleDecoded` 里是空实现:`got_vel` 未被使用、
`joint_vel_mj` 根本没解析、`body_frames_received_` 从不自增、strict 模式不拒帧。
本轮把实现补齐。

**速度契约(与 MuJoCo LiveMotion 对齐,不可改):**

- 发送端按固定 50 Hz 计算
  `reference_joint_velocity[t] = (reference_joint_position[t] - reference_joint_position[t-1]) * 50`;
- 该计算发生在 live-edge clamp **之前**;
- clamp 到最新帧后,未来 9 个 slot 重复最新位姿,**同时重复最新的非零速度,不置零**;
- C++ 侧只做校验、归一化、转发,**绝不**用 wall-clock 接收时间差分重估速度。

**strict 模式(`--strict-reference-velocity`)下的拒帧条件:**

- 缺显式 `joint_vel_mj`;
- 未来窗口不完整(缺 `joint_pos_mj_future` / `root_quat_xyzw_future` /
  `joint_vel_mj_future` 任一)。

拒掉的帧不进缓存、不推进 `body_frames_received()`、不刷新
`LastReceivedMonotonicS()`。也就是说 publisher 若悄悄退回旧版布局,会把
pose-ref watchdog **饿死**(触发有界返回),而不是让策略静默吃到另一套速度语义。

非 strict 模式保留原有 v4/v5 wall-clock 回退,供 mock-VLA 等既有集成使用。

另外补了两处此前缺失的校验:所有解码浮点必须有限;当前帧与全部 9 个未来帧的
`root_quat_xyzw` 都要归一化(范数落在合理区间),未来窗口只有在四元数全部合格后
才会被提升。

---

## 2. 策略入口的实时参考就绪硬闸

`include/supported_policy_gates.hpp`(新增)+ `RequestSupportedPolicy`

在此之前,`policy` 请求不检查实时参考状态——publisher 只要存在就能进策略。
现在 ZMQ 路径下必须按顺序满足全部 5 项,任一不满足即拒绝并打印是哪一条没过:

| 顺序 | 检查 | 参数 | 默认 |
| --- | --- | --- | --- |
| 1 | 收到过 body 参考 | — | — |
| 2 | 已接受的 body 帧数达标 | `--zmq-warmup-body-frames` | 40 |
| 3 | 当前参考足够新 | `--zmq-entry-max-age-s` | 0.5 s |
| 4 | 已**连续** fresh 足够久 | `--zmq-entry-min-fresh-s` | 0.8 s |
| 5 | strict 下拿到过带显式速度的帧 | `--strict-reference-velocity` | 关 |

第 2 项刻意用 `body_frames_received()` 而不是 `total_frames_received()`:后者把
token-only / hand-only / 被拒的帧也算进去,会让一串手部帧就满足"收到 40 帧",
而 body 参考其实还是 bootstrap 的 `default_angles`。

第 4 项是防"单帧复活":参考中断后又回来一帧,age 看起来是新的,但连续窗口已被
重置,所以**不能**立刻重新武装策略。连续性由 `UpdateZmqRefFreshness(now)` 在
`GROUND_LOAD_HOLD` 每个 tick 维护。

非 ZMQ 路径(motion file)该闸门不适用,直接放行——调用方原有的启动闸(状态机、
HAL 新鲜度、交接后稳定性)仍然独立生效。

---

## 3. SUPPORTED_POLICY 断流 → 有界 2 秒回抓拍静态保持

`x2_deploy_onnx_ref.cpp`,`SUPPORTED_POLICY` case

此前 `SUPPORTED_POLICY` 状态下**完全没有** ZMQ 断流检查。现在加了一路独立检查,
判定走纯函数 `ZmqReferenceStale(watchdog_active, ref_age, pose_ref_stale_s)`:

- `age >= --pose-ref-stale-s`(默认 0.5 s)→ 判定 stale;
- `age < 0`(从未收到帧 / 源报"未收到"哨兵值)→ **也判定 stale,fail closed**;
- watchdog 未激活(非 ZMQ 路径、`--disable-pose-ref-watchdog`、
  `--pose-ref-stale-s <= 0`)→ 永不触发。

判定为 stale 后走 `EnterSupportedPolicyReturn(now, reason, clean_stop=false)`:
**2 秒斜坡回到抓拍的静态姿态 → GROUND_LOAD_HOLD,策略关闭,不自动重入**。

这一点和普通 `CONTROL` 态的行为**刻意不同**:`CONTROL` 的 watchdog 仍然跳
`SAFE_IDLE`/`default_angles`,那条两闸恢复路径本轮**没有改动**,不要合并这两条
路径。supported 路径下机器人有龙门架部分承载,直接跳 `default_angles` 的姿态跃变
比有界返回更危险。

参考恢复后不会自动重启策略——必须操作员重新发 `policy`,而且要重新过一遍第 2 节
那 5 项闸门(其中连续 fresh 窗口已被中断重置)。

---

## 4. 专用 launcher

`run_x2_supported_garment_zmq.sh`(新增),确认令牌
`X2_SUPPORTED_GARMENT_ZMQ_START`

```bash
./run_x2_supported_garment_zmq.sh X2_SUPPORTED_GARMENT_ZMQ_START
```

它 `exec` 到 `run_x2_suspended_sonic.sh X2_SUSPENDED_SONIC_START
--supported-neutral-garment-zmq`。新模式与 `--supported-neutral-damped`
**共用同一个 case 分支**,平衡包线逐字复用,因此差异只可能来自新增的 `ZMQ_ARGS`。

**与 `neutral_damped` 的逐项差异(仅参考源与安全闸,别无其他):**

| 项 | neutral_damped | garment_zmq |
| --- | --- | --- |
| 参考源 | StandStillReference | `--vla` → ZMQ v5 `127.0.0.1:5556` topic `pose` |
| 断流 watchdog | 未启用 | `--pose-ref-stale-s 0.5` |
| 严格速度 | — | `--strict-reference-velocity` |
| 入口预热 | — | `--zmq-warmup-body-frames 40` |
| 入口新鲜度 | — | `--zmq-entry-max-age-s 0.5` |
| 入口连续新鲜 | — | `--zmq-entry-min-fresh-s 0.8` |
| 以下全部相同 | | |
| anchor / 时长 / 斜坡 | `--supported-policy-anchor-default` / 300 s / 4 s | 同 |
| 偏差上限 leg/waist/arm/head | 0.60 / 0.20 / 0.25 / 0.08 | 同 |
| tilt delta / abs | 5.0° / 25.0° | 同 |
| joint-vel trip / target rate | 0.8 / 0.12 | 同 |
| 阻尼与 LPF | waist LPF 2.5,kd ankle-pitch 3.31 / ankle-roll 2.20 / waist-pitch 3.0 | 同 |
| writer / max-dev / LPF | 250 Hz / 2.0 / 8.0 | 同 |

`--zmq-*` 与 `--pose-ref-stale-s` 经 `--deploy-extra-arg` 原样转发给 C++ 二进制
(`deploy_x2.sh` 本身不建模这几个旋钮)。`--suspended-start` 使
`deploy_x2.sh:2239` 的 stand-pose YAML 注入分支被跳过,`--vla` 无其他副作用。

**launcher 不会替你发 `policy`。** 策略进入仍然是双重把关:操作员手动输入 +
二进制内部那 5 项就绪闸。不要因为 publisher 起来了就发 `policy`。

---

## 5. 离线测试(无 ROS、无 ONNX、无硬件)

`test/test_supported_policy.cpp`(新增,11 个用例),已在 CMakeLists 的
**离线模式**和 **BUILD_TESTING** 两处都注册。

```text
TestEntryAcceptedWhenReady              全部就绪 → 接受
TestNonZmqAlwaysAccepts                 非 ZMQ 路径不被本闸门拦
TestRejectNoBodyReference               没有 body 参考 → 拒
TestRejectBeforeWarmup                  预热不足 → 拒(含 == 阈值的边界)
TestRejectStaleAtEntry                  入口过旧 / age<0 → 拒
TestRejectUntilContinuouslyFresh        单帧复活不足以武装 → 拒
TestStrictVelocityRequiresExplicitFrame strict 无显式速度 → 拒;非 strict 放行
TestEntryCheckOrdering                  多项同时不满足时,原因指向最根本那条
TestSupportedPolicyStaleTrips           fresh 不触发;>= 阈值触发
TestStaleFailsClosedOnNeverReceived     从未收到 → fail closed
TestStaleNoopWhenWatchdogInactive       watchdog 未激活 → 永不触发
```

跑法(Mac 或机器人皆可,只依赖头文件):

```bash
c++ -std=c++20 -Iinclude -Wall -Wextra -o /tmp/t test/test_supported_policy.cpp && /tmp/t
```

之所以把判定抽成纯函数(`supported_policy_gates.hpp`,header-only、inline):
`x2_deploy_onnx_ref.cpp` 是个持有 ONNX 推理、HAL IO 和硬件定时器的 ROS 2 节点,
在工作站上无法实例化。抽出后部署二进制和测试目标编译的是**同一份**逻辑。

---

## 6. Wire 规格 + 参考 publisher + 字节自校验

三个新增文件,全部**只读地**从 C++ 解码器反推,没有改动任何 C++:

| 文件 | 作用 |
| --- | --- |
| `docs/X2_ZMQ_V51_WIRE_SPEC.md` | 逐条引用 `zmq_pose_input_source.cpp` 行号的权威 wire 规格 + 发送端检查表 |
| `scripts/garment_zmq/publish_v51_reference.py` | 参考 publisher。`--drop-velocity` 触发 strict 拒帧,`--stop-after-s` 触发断流 watchdog |
| `scripts/garment_zmq/verify_v51_bytes.py` | 纯 Python 镜像解码器 + 45 项断言,只依赖 numpy,不需要 pyzmq |

```bash
cd scripts/garment_zmq && python3 verify_v51_bytes.py   # 45/45 通过
```

自校验覆盖的、值得单独点出的几条陷阱:

- **`estop` 只看字段名就永久闭锁**(`cpp:265-268`),根本不读值。`pack_message`
  硬拒绝这个字段名。
- **四元数范数校验抓不到 wxyz/xyzw 顺序错误** —— wxyz 单位四元数范数同样是 1,
  会被接受,然后策略把身体拧向世界 +X。顺序只能由发送端保证。
- **`shape` 为空 → 该字段按 0 字节计**(`subscriber.hpp:155`),之后所有字段
  全部错位。标量必须写 `[1]`。
- **dtype 是字符串精确比较**,`f64` 不会被自动降级成 `f32`,该字段被静默跳过。
- **header 必须 NUL 右补齐到 1280**,否则 `strnlen` 会把二进制吃进 JSON。
- 缺 `joint_vel_mj_future` 时 C++ 的回退分支在 live-edge clamp 下会算出
  **全零**未来速度(`cpp:425-435`),破坏 parity —— 所以 strict 必须要求它。

`jsonl_source()` 刻意留成 `NotImplementedError`:衣服 JSONL 的字段名/关节维度/
四元数顺序/断流表示法必须由衣服那条线确认,在这里瞎猜会制造出第二套规格。

## 7. 实测速率(衣服全链路已跑通)

| 指标 | 实测 |
| --- | --- |
| 衣服 → LFP/TIC/GMR → Sonic → dry-run | 47–50 Hz |
| Shadow 记录 | 50.9 Hz |
| Sonic 推理 | 中位数 7–9 ms,P95 12–15 ms |
| 单帧端到端耗时 | 48–52 ms(各模块并行/流水化) |

三条解读:

1. **速度契约的误差被压到很小。** 差分恒乘 50.0,实测 47–50 Hz 对应真实 dt
   20.0–21.3 ms,速度幅值只被低估 0–6%。这远好于早期 33–35 Hz 时的
   1.43 倍放大。**不要**因此去改 C++ —— 恒乘 50.0 才是 MuJoCo parity。
2. **48–52 ms 是延迟,不是周期。** 约 2.5 个策略周期的端到端延迟。参考落后于
   佩戴者约 50 ms,这是遥操手感问题,不是安全问题:所有闸门阈值
   (`stale 0.5 s` / `entry-max-age 0.5 s`)都比它宽一个数量级。
3. **推理占了 20 ms 预算的 60–75%。** P95 12–15 ms 还有余量,但 P99 若越过
   20 ms 就会掉策略帧。真机测试时要盯尾延迟,不是中位数。

闸门参数与实测速率的匹配关系(无需调整):`--zmq-warmup-body-frames 40` @ 50 Hz
= 0.8 s,正好等于 `--zmq-entry-min-fresh-s 0.8`;`--pose-ref-stale-s 0.5` ≈ 25
帧的容忍窗口。

一个已知的**未被拒帧覆盖**的情况:偶发丢一帧会让间隔变成 ~40 ms,该帧速度被
低估约 50%,但帧本身完全合法、不会被拒。单帧的 50% 速度低估不构成安全问题,
持续丢帧才会 —— 那种情况由 watchdog 兜底。

## 构建与验证记录

机器人: `agi@192.168.43.21`(SoC1,NVIDIA Orin/aarch64),编译时控制器**未运行**
(仅 housekeeper)。

| 项 | 值 |
| --- | --- |
| 源码树 | `~/projects/x2_sonic_migrated_20260826/sonic_x2_transfer_v2/GR00T_audit/gear_sonic_deploy` |
| 新二进制(scratch) | `runtime_suspended/ws/install_gatecheck/.../x2_deploy_onnx_ref` |
| 新二进制 SHA-256 | `e4545d0e3d5a52c86b16bdebb1e7860431f4391dd45d464e3adb2c199cf9a5e7` |
| 原二进制(未动) | `runtime_suspended/ws/install/.../x2_deploy_onnx_ref` |
| 原二进制 SHA-256 | `b750b7d8c78cdee42e3b0927c7bfe8cb7a6636b4e39a11b20da03cd4f058131c` |
| 模型 | `models/x2_sonic_frozen_g1core_lora_v2.onnx` |
| 模型 SHA-256 | `8ccc42a82ea2c446aa708aece58c7a638ea943a79ed88eeab259669e641271e9`(与预期一致) |
| 备份 | `backups/before_garment_zmq_gates_20260829_141508/` |

**刻意编到 scratch install base**,没有覆盖正在使用的 install 空间——8月28
那个可用二进制原封不动,何时提升由操作员决定。提升方法是把上面的 colcon 命令
里 `--build-base`/`--install-base` 换回 `runtime_suspended/ws/build`、
`runtime_suspended/ws/install`。

编译命令(需先 source `aimdk_ws_0_8_18`):

```bash
colcon --log-base "$PWD/runtime_suspended/ws/log_gatecheck" build \
  --packages-select agi_x2_deploy_onnx_ref \
  --base-paths "$PWD/src/x2/agi_x2_deploy_onnx_ref" \
  --build-base "$PWD/runtime_suspended/ws/build_gatecheck" \
  --install-base "$PWD/runtime_suspended/ws/install_gatecheck" \
  --cmake-force-configure --cmake-args \
  -DONNXRUNTIME_ROOT=<repo>/.deps/onnxruntime -DBUILD_TESTING=ON
```

验证结果:

- colcon 编译通过(59.1 s,0 error);
- `ctest`: `test_obs_builder` + `test_supported_policy` **2/2 通过**(aarch64);
- 4 个新 CLI 标志与 3 条闸门原因字符串确认已编入二进制(`strings` 核对);
- 两个 launcher `bash -n` 通过;错误令牌正确打印用法并 `exit 2`;
- 同步的 8 个文件 md5 与 Mac 侧逐一相符;同步前已比对全包 33 个文件,确认机器人
  侧无我不知道的改动(机器人上多出的 5 个文件是它自己的时间戳备份,未触碰)。
