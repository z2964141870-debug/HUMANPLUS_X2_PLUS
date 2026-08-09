# BASE Phase10：独立 recovery 角色合同审计

状态：`BLOCKED_NOOP_RECOVERY_MODEL_UNDER_PHASE9_CONTRACT`
official 新运行：`0`
训练更新：`0`

## 一句话结论

Phase10 原计划在 Phase9 精确合同下只把 `recovery_model` 从 source 换成 f005，但 Phase9 的 `curriculum_then_policy` 在 handoff 后硬编码调用 `stationary`，五条 source 中 recovery 推理次数均为 `0`；因此该候选是严格 no-op，运行五条 official episode 不能回答问题，已在启动前停止。

## 假设

保持：

- moving = Stage306；
- stationary = source stand backend i150；
- prepare / stand / matched move / curriculum stop；
- stiff 1.2、upper fixed、5 repeats、adapter 与所有 hash；

只把 recovery backend 替换为 Phase9 f005，应能隔离 handoff 后的 recovery 性能，同时由 source stationary 保护起步。

## 干预

预定唯一变量：

```text
RECOVERY_MODEL:
source stand_backend_scratch_i150_actor.onnx
→ phase9_stateful_recovery_f005_u5_actor.onnx
```

运行前先审计 hash、Phase9 五条 source 的配置和 policy slot inference counts。没有修改 adapter，也没有启动 AimDK。

## 对照

Phase9 source 五条均来自同一合同：

| 项 | 固定值 |
|---|---|
| moving | Stage306 123D actor |
| stationary / recovery | source stand backend i150 |
| prepare / stand / move / stop | 0.2 / 2.0 / 5.2 / 8.0 s |
| start / stop event | accelerate 1.0 s / `curriculum_then_policy` 2.0 s |
| PD | `official_kp_ankle ×1.2` |
| upper / supervisor | fixed / none |
| clock | step |

五条合同逐字段一致，hash 全部匹配：

- adapter：`d80900b2...ed7e8f9`；
- Stage306：`da95011f...532bc4c`；
- source stand：`edb73c7c...8367565`；
- f005：`9bc672fc...13fc0ca`。

## 结果

### 1. Phase9 source control 指标

| 指标 | source 5-run |
|---|---:|
| stand | 5/5 |
| startup | 5/5 |
| move | 0/5 |
| stop | 0/5 |
| full | 0/5 |
| heading max mean | 0.628 rad |
| lateral displacement mean | 0.551 m |
| stop drift mean | 0.423 m |
| move signed pitch mean | -8.18° |

这些可以继续作为 frozen control，但不能自动成为 recovery-only A/B，因为原 control 根本没有执行 recovery slot。

### 2. Policy slot 权限

五条 source 完全相同：

```text
main       = 360 inference / episode
stationary = 400 inference / episode
recovery   =   0 inference / episode
```

代码级原因：

```python
elif stop_controller == "curriculum_then_policy":
    ...
    # handoff 后
    _policy_targets(..., policy_slot="stationary")
```

`RECOVERY_MODEL_PATH` 只让初始化阶段把 previous/issued action 复制到 recovery slot；该 branch 后续没有用它推理。也就是说：

```text
加载 f005 recovery ONNX ≠ 调用 f005 recovery ONNX
```

### 3. 为什么不直接改用 brake_blend

`brake_blend_to_policy` 的确通过 `stop_policy_slot(recovery_model)` 调用 recovery，但把 Phase9 的 `curriculum_then_policy` 改成 brake blend 同时改变了 stop controller、时序和目标生成方式，已不再是“只改变 recovery model”的 paired control。因此本阶段没有偷换合同。

### 4. Candidate 指标

候选未运行，所以 startup / move / stop / full、heading、lateral、stop drift、pitch 均记为 **N/A**，不能用 Phase9 f005-as-stationary 的结果冒充 recovery-only 结果。

## 结论

原假设在现有合同下不可检验。不是 f005 recovery 被证明无效，而是它根本没有控制权限。若仍跑五条，由于唯一变量从未进入 action path，得到的差异只会是 simulator episode 方差，不能归因于 recovery model。

本次启动前审计避免了五条无效 official 重复，也揭示了一个重要职责事实：Phase9 实际测试的是“把训练后模型同时当 stationary backend 使用”，而不是独立 recovery 角色。

## 下一步

需要重新预注册一个明确的角色合同，二选一：

1. 新增 `curriculum_then_recovery` / 将 handoff 后 slot 显式路由到 `stop_policy_slot`，然后 source-recovery 与 f005-recovery 做 paired A/B；这是 adapter/control-contract 变化，必须另设 source control；
2. 使用已经会调用 recovery 的 `brake_blend_to_policy`，但同样需要在该新合同下重跑 source control，不能复用 Phase9 数字。

在新合同获准前，不运行 official gate、不训练、不改 adapter、不碰 WBT/A3。

## 产物与验证

- 审计 runner：`tools/official_x2/audit_phase10_recovery_role_contract.py`
- 单测：`tests/test_phase10_recovery_role_contract.py`
- JSON：`reports/baseline/x2_recovery_phase10_role_contract.json`
- 回归：`8 passed`
