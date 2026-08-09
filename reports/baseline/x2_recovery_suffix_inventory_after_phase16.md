# BASE Stop/Recovery Suffix 库存审计（Phase16 后）

> 日期：2026-08-09
> 性质：严格只读审计；未采集、未训练、未运行 official。
> 机器可读库存：[x2_recovery_suffix_inventory_after_phase16.json](x2_recovery_suffix_inventory_after_phase16.json)

## 一句话裁决

现有库**有成功和失败的 observational stop trace，也有一条 state-complete 的临界前缀；但没有在同一冻结合同下同时具备成功、临界、坍塌失败三类 stateful suffix**，因此现在还不能诚实预注册 `fraction=0 vs 固定小 fraction` 的 5-update 训练。

## 审计口径

本报告不事后创造“离门多近算临界”的距离阈值：

- 成功 episode：`stop_gate_pass=true`；
- 临界非倒地 episode：`stop_gate_pass=false`，但 `survived_stop_height_gate=true`；
- 失败 episode：没有通过 height survival；
- 窗口类别单独报告：一个完整 episode 最终失败，其前 1.5 秒仍可能只是 pre-collapse critical prefix。

这里的 observational trace 表示已有连续 `obs/action/root`；stateful suffix 还必须有完整 physical q/dq/root、controller snapshot、actual previous/issued action 和逐行 hash。

## 假设

若 Stage250/306/326/335 与 Phase11/13/16 已经覆盖成功、临界、失败三类 closed-loop suffix，并且这些 suffix 的控制状态契约一致，则可以冻结一次 `fraction=0 vs 固定小 fraction` 5-update 实验；否则先补最小数据合同，不得把大量相关 trace row 当作独立训练证据。

## 干预

只读取：

- Stage250 三条 nominal official trace；
- Stage306 checkpoint 在 Stage350 matched-event gate 下的五条 trace；
- Stage326 stiff1.2 fixed-upper 五条 trace；
- Stage335 90-state reset NPZ 与 source report；
- Phase11 十条、Phase13 十条 trace；
- Phase16 一条 trace 与唯一 sidecar。

没有改变任何原始文件、checkpoint、adapter、门槛或哈希。

## 对照

每个 family 都按同一现有 stop/height gate 分类，并分别审计：

1. 是否有93D obs和15D row action；
2. 是否有明确 physical lower target；
3. 是否有完整 q/dq/root snapshot；
4. 是否有完整 controller snapshot；
5. previous/issued action 是否是单独字段并与 physical/controller hash 绑定。

Stage335 的 90 个状态全部来自 Stage326 同五条 episode，不能作为额外 90 次独立试验计数。

## 结果

### 1. Episode 级库存

| Family | Episode | 成功 | 临界非倒地失败 | 高度坍塌失败 | 完整 controller snapshot |
|---|---:|---:|---:|---:|---:|
| Stage250 nominal | 3 | 3 | 0 | 0 | 0 |
| Stage306 matched source | 5 | 0 | 0 | 5 | 0 |
| Stage326 stiff-fixed stop | 5 | 3 | 0 | 2 | 0 |
| Phase11 recovery role | 10 | 0 | 0 | 10 | 0 |
| Phase13 handoff continuity | 10 | 0 | 0 | 10 | 0 |
| Phase16 f005 trace | 1 | 0 | 0 | 1 | 0（完整状态在独立 sidecar） |
| **合计** | **34** | **6** | **0** | **28** | **0 trace** |

这 34 条并不处于同一控制合同：

- Stage250 使用 Stage219、nominal stop policy，虽然 3/3 通过，但不能直接充当 Stage306/f005 recovery 的成功对照；
- Stage326 在 `stiff1.2 + fixed upper + brake_blend_to_policy` 下 3/5 成功，是当前唯一同一合同内自然产生 pass/fail 的 family；
- Phase11/13/16 使用 `curriculum_then_policy` recovery 角色，现有全部最终倒地。

### 2. 旧 trace 有什么、缺什么

- Stage250/306/326/Phase11 都有连续 93D `obs` 和15D `action`，但没有 physical lower target、full controller snapshot，也没有单独 hash-bound `actual_issued_action`。
- Phase13 额外有15D physical target，并已证明 next previous-action 与实际 action 同步；但 controller snapshot 仍未逐行保存。
- 因此这些是有价值的闭环观察轨迹，不是可直接恢复的 stateful suffix。

### 3. Stage335 是离散 reset，不是 suffix

Stage335：

- 90 states，其中63个来自 eventual-pass episode、27个来自 eventual-fail episode；
- 保存 root、q/dq、body velocity、projected gravity、previous action、gait phase 等 reset 组件；
- 93D 完整 rejoin 依赖不可变 source trace；
- 没有 current issued action、controller event/state，也没有逐状态后继序列；
- eventual-pass/fail 是来源 episode 的未来标签，不是单状态“可恢复性真值”。

所以 `63 + 27` 不能被写成三类 closed-loop teacher，也不能与 Phase16 的76行直接相加成166个独立样本。

### 4. Phase16 是唯一 state-complete 证据

Phase16 sidecar：

- 76行，handoff 后0–1.5秒；
- 每行有完整 physical q/dq/root、93D obs、actual previous/issued action、完整 controller snapshot 与 hashes；
- 完整 episode 最终失败；
- 但窗口内 root z最低 `0.620m`，到 `1.40s` 才首次 tilt 超 `0.30rad`，尚未出现 height collapse。

因此按完整 episode 它是“失败”；按采集窗口形态，它只是一条 **pre-collapse critical suffix**。这两个标签不能混为一谈。

## 结论

### 已有

- [x] observational success episode：Stage250 3条、Stage326 3条；
- [x] observational failure episode：28条；
- [x] Stage335 eventual-pass/fail 离散 reset；
- [x] 一条 hash 完整的 stateful pre-collapse critical suffix（Phase16）。

### 缺失

- [ ] stateful successful suffix；
- [ ] 同一冻结合同下包含实际 height collapse 的 stateful failure suffix；
- [ ] stop command→brake→handoff→recovery 的统一 event-aligned stateful 窗口；
- [ ] 多个独立 episode，避免把单 episode 的76个相关 tick 当76次试验。

### 5-update 是否可预注册

**不能。**

当前如果直接设置非零 suffix fraction，唯一 suffix 来源就是一条 f005 最终失败 episode；这会同时改变“是否使用 suffix、只使用哪个失败轨迹、窗口类别和相关样本数”，无法把结果归因于 closed-loop curriculum 本身。训练继续锁定。

## 唯一下一采集实验

任务名：`outcome-stratified stop-event stateful suffix capture`

- 目标：在同一冻结合同下取得成功 continuation、pre-collapse critical prefix、collapse-containing failure；
- 合同：固定 Stage326 历史 mixed-outcome 设置——Stage306 moving、source stand stationary、`brake_blend_to_policy` 及其既有 post-latch stationary authority、stiff1.2、fixed upper、相同PD/command/50Hz；记录器是唯一 control-path-neutral 改动；
- 窗口：从 stop command 开始，跨过 brake/handoff，记录至 `min(4.0s, height collapse)`；每行沿用 Phase16 physical/controller/action hash；
- 预算：最多5个 episode；只有拿到至少2个成功和1个失败完整 episode才可提前停止；5个用完即停止，不继续补跑；
- 立即停止门：模型/adapter/合同 hash 不符、任一行缺 actual previous/issued 或 controller snapshot、event/authority 时序不一致；
- 采完仍不自动训练：先做 class balance、dedup、reset-integration 离线审计，再裁决是否值得预注册5-update。

这一实验选择 Stage326，是因为它在相同 stiff-fixed 合同下历史上自然产生3 pass/2 fail；它不是把 Stage326 老 trace 伪装成新 stateful 数据。

## 下一步

在上述唯一采集实验获批并完成前：

- 不运行 `fraction>0`；
- 不启动5-update、25-update或长训；
- 不再追加同类 f005 failure-only sidecar；
- 不修改 WBT 或真机链路。

## 文件与复现

- JSON：`reports/baseline/x2_recovery_suffix_inventory_after_phase16.json`
- 审计工具：`tools/official_x2/audit_base_recovery_suffix_inventory.py`
- 测试：`tests/test_base_recovery_suffix_inventory.py`

本阶段只读审计，无 official、无训练、无真机。
