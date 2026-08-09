# BASE Phase15：On-policy Recovery Suffix Aggregation 合同

> 日期：2026-08-09
> 状态：设计与纯测试完成；未采集、未训练、未运行 official MuJoCo。
> Dry schema：[x2_recovery_phase15_suffix_aggregation_dry_schema.json](x2_recovery_phase15_suffix_aggregation_dry_schema.json)

## 假设

Phase14 不支持“handoff 后的 q/dq/当前 action 整体超出 Stage335 训练状态域”，但发现三条健康 episode 的 `previous_action` 15D 组合在 handoff 后约 `0.24–0.26 s` 先离开 Stage335 最近邻支持，约 `1.1 s` 后才出现 gravity/root 失稳。

因此 Phase15 的待验证假设是：

> 当前课程缺的不是更多相互独立的 reset pose，而是 recovery policy 自己运行后产生的、state → issued action → next state 与 action-history 连续一致的 closed-loop suffix。

这与 DAgger 的共同点是采集 learner 实际访问的分布；区别是这里没有外部 expert 给每个状态重新标动作，也不把失败 action 当监督标签。suffix 只作为 RL reset curriculum，使 PPO 能在自身诱发的 action-history 分布上继续优化物理回报。

## 干预

本阶段只加入一个**默认关闭、只记录证据**的 Stage208 入口：

```text
--post-handoff-snapshot-output <sidecar.json>
--post-handoff-snapshot-horizon-seconds 1.5
```

未传第一个参数时，不记录 sidecar、不增加 trace 字段、不改变 summary、不加载 suffix 数据，也不影响控制 action。启用时则强制：

- `clock_mode=step`，50 Hz；
- `stop_controller=curriculum_then_policy`；
- 存在独立 `recovery_model`；
- canonical、非 mirror actor frame；
- 只在 recovery 真正取得 authority 后记录；
- handoff 后只记录 `0.0–1.5 s`。

官方 gate wrapper 仅增加了默认空的环境变量透传，没有启动任何容器。

## 对照

### Control：`suffix_fraction=0`

虚拟合并器在 fraction 为 0 时：

1. 只校验冻结 Stage335 路径和 SHA；
2. 在打开任何 suffix sidecar 之前立即返回；
3. suffix state 数为 0；
4. suffix source mask 全为 false；
5. 不消费 RNG，因此后续 base sampling 随机序列逐项等价。

Dry plan 绑定的 Stage335 为 90 states，SHA256：

`4d8ce06b6e8dd9b43363dadedb781e9feb75f72ac092b475de2a1e2772545013`

原 NPZ 始终只读，不被合并器重写。

### Candidate：非零 suffix fraction（本阶段未执行）

未来只有在拿到真实、失败的 f005 sidecar 后，才允许形成 hash-bound virtual union：

```text
immutable Stage335 90 states
           +
failed f005 closed-loop suffix representatives
           ↓
per-reset source mixture（fraction 预注册）
```

这里的“合并”是采样层虚拟并集，不生成一个看似相同、实则丢失 controller/event 信息的新 NPZ。

## Snapshot schema

每个 `aimdk_x2_recovery_suffix_snapshot_v1` row 必须包含：

- 31 个关节的绝对 q、dq 和固定 joint order；
- root 世界位置、`xyzw` 姿态、世界线/角速度；
- recovery actor 实际看到的 93D obs；
- obs 中的 exact previous-action input；
- 实际下发的 15D issued action；
- command、gait phase/contact、controller sequence clock、handoff-relative time；
- Phase8 完整 controller state，包括三 policy slot 的 previous/issued action、brake/latch、heading、upper、prediction 与 counters；
- physical state SHA、controller state SHA、整行 snapshot SHA。

交叉校验会拒绝：

- obs command/gait/previous-action 与独立字段不一致；
- controller recovery slot 的 issued action 与 row action 不一致；
- controller snapshot 未绑定同一个 physical SHA；
- 非 93D/15D/31DOF、非有限值、非归一 quaternion；
- 非 recovery authority 或超过 1.5 s 的 row。

### 时间边界必须诚实

记录点定义为 `post_inference_pre_physics`：物理状态是本 tick actor 输入时的状态，controller state 已包含本 tick 新下发 action。若未来做精确物理 replay，必须先把该 issued action 应用一个 physics interval，再进入下一 actor tick。

因此这份 sidecar 足够支持“时序一致的 RL reset curriculum 设计”，但本阶段**不声称**它已经实现 ROS mid-event deterministic replay。

## Sidecar、hash 与去重

一个 `aimdk_x2_recovery_suffix_sidecar_v1` 只包含一个 episode，并绑定：

- 主 trace 路径与 SHA；
- adapter 路径与 SHA；
- recovery ONNX 路径与 SHA；
- episode 的 stand/start/move/stop/full outcome；
- 每行 snapshot SHA、sidecar content SHA；
- 完整原始 rows。

去重只影响 reset sampling representative，不删除原始 row。预注册 key 使用 actor 93D、issued action、root z 和 handoff 时间，采用固定细粒度 bin；root XY/yaw 被视为全局等变自由度而忽略，但 event time 被保留，避免把不同 handoff/blend 阶段错误合并。

这比单纯比较 93D 欧氏距离更保守：两个物理状态若 previous/issued action 或 event time 不同，不会因为 q 很接近就被当成同一个 reset contract。

## 结果

本阶段只证明了合同闭合：

- snapshot/sidecar/virtual-union 三层 schema 可构建并 hash 验证；
- physical、actor input、issued action 与 controller state 之间可 fail-closed 交叉校验；
- 保守 near-duplicate 可映射到同一 representative，完整 source row 保留；
- `suffix_fraction=0` 即使给出不存在的 sidecar 路径也不会打开它；
- `suffix_fraction=0` 不消费 RNG；
- 非零 fraction 只接受 outcome 为失败的真实 sidecar；
- live Stage208 adapter 的 capture 开关默认关闭，sidecar 不进入控制路径；
- 相关回归共 `20 passed`。

Phase15 没有产生任何真实 suffix row，`collected_suffix_rows=0`；因此尚不能判断该课程能否改善 recovery。

当前 dry-schema 冻结输入包括：

- f005 ONNX SHA：`9bc672fc3c535dbe6cd2709cdec4531eb9b61990457172e9c813a8ac713fc0ca`
- Phase15 adapter SHA：见 dry-schema JSON（当前为 `46634845d68d1de80bb001b4e102855c80ae21192e5f111460ed9115254a05bd`）

## 结论

设计通过，但训练和 official collection 都保持锁定。

这条路线比继续堆离散 Stage335 reset 更针对 Phase14 的证据，因为它显式保存了 previous-action、issued-action、controller event 与后继物理状态的闭环关系；同时它仍只是假设驱动的数据课程，不能把时间先后相关性写成倒地根因。

失败 suffix 也不能被当作“正确动作监督”。后续若训练，必须保留 immutable Stage335/base control、较小固定 suffix fraction 和原始物理 reward，否则可能只是更频繁地复现失败。

## 下一步

等备份完成并获得明确授权后，最多运行一次 hash-frozen official 采集：使用 f005 recovery、Phase13 已验证的 action-history 同步 handoff、同一 fixed-upper/stiff1.2 matched-event 合同，只新增 sidecar recording，不改变 action。采集成功后先离线执行：

1. sidecar/schema/hash 审计；
2. 确认恰为 recovery authority 后 0–1.5 s；
3. 标记完整 episode outcome；
4. 去重并生成 virtual-union manifest；
5. 再决定唯一的 `fraction=0 vs 固定小 fraction` 5-update paired smoke。

在完成前四项以前，不训练、不扫 fraction、不运行 25-update。

## 文件与复现

- 合同实现：`tools/official_x2/recovery_suffix_aggregation.py`
- 默认关闭挂点：`tools/official_x2/stage208_official_mujoco_adapter.py`
- Dry-schema 工具：`tools/official_x2/build_phase15_recovery_suffix_dry_schema.py`
- 纯测试：`tests/test_phase15_recovery_suffix_aggregation.py`

```bash
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=tools \
conda run -n x2-sonic-isaaclab pytest -q -p no:cacheprovider \
  tests/test_phase15_recovery_suffix_aggregation.py \
  tests/test_controller_snapshot_contract.py \
  tests/test_phase13_handoff_continuity_contract.py \
  tests/test_phase14_recovery_ood_audit.py
```
