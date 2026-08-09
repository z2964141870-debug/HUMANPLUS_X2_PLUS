# BASE Phase21：Outcome-aware 5-update Paired Smoke

> 日期：2026-08-09
> 裁决：**不晋级；25-update 继续锁定。** 5% state-role curriculum 保持了接口和二元子门，但没有让 move/stop/full 产生一次通过，且 heading/lateral 相对 source 的均值变差。
> 机器可读结果：[x2_recovery_phase21_outcome_aware_paired_smoke.json](x2_recovery_phase21_outcome_aware_paired_smoke.json)

## 假设

Phase20 已证明 `success_safe` 与 `critical_from_failure` 两类状态可以精确注入。Phase21 检验唯一命题：同一 source、seed、训练预算和物理合同下，将 5% reset 换成两类 state role 1:1、role 内均匀采样，能否改善 official recovery，同时不破坏 source 与同 seed 的 fraction-0 control。

## 干预

两支训练均固定：

- source `model_150.pt`，SHA `4da931cf…251`；
- seed 47，64 env，5 updates，weights-only resume、optimizer reset；
- 每支 `64×24×5=7,680` transitions；
- ideal actuator、sole12、official ankle PD、self-collision off；
- 同一 reward、93D observation、15D action、gait template 与 stand/recovery backend；
- moving Stage306 不加载、不更新。

唯一变量：

```text
f000: outcome-aware reset fraction = 0.00
f005: outcome-aware reset fraction = 0.05
      success_safe : critical_from_failure = 1 : 1
```

训练仅保留最终 `model_155.pt` 与 ONNX。

## 对照

训练后使用冻结 official AimDK matched-event full episode：

- moving：Stage306；
- stationary：source i150；
- recovery：source / f000 / f005；
- Phase13 handoff blend 0.5 s、stiff-fixed 1.2、fixed upper；
- 每组 5 条，共 15 条；
- 每条期望 policy slots：`main=360, stationary=100, recovery=300`。

训练 reward、episode length 和 signed pitch 只作诊断，不参与晋级。

## 结果

### 训练诊断

| 分支 | 最终 mean reward | 最终 episode length | support-set KL(source‖final) mean / p95 | Actor mean delta RMSE |
|---|---:|---:|---:|---:|
| f000 | -4.948 | 53.83 | 2.268 / 3.943 | 0.0881 |
| f005 | -5.517 | 54.57 | 3.518 / 9.264 | 0.0896 |

训练器没有向 TensorBoard 写 PPO minibatch KL。表中 KL 是在 Phase20 全部 432 个 eligible 93D row 上离线计算的 diagonal-Gaussian `KL(source‖final)`，只能说明最终策略相对 source 的漂移，不能冒充训练时 KL 或作为晋级证据。f005 的 support-set KL 更大，没有形成“更小更新、更好物理结果”的证据。

### Official 二元门

15/15 episode 全部有效，policy slot count 全部 exact。

| Recovery | Valid | Stand | Startup | Move | Stop | Full |
|---|---:|---:|---:|---:|---:|---:|
| source | 5/5 | 5/5 | 5/5 | 0/5 | 0/5 | 0/5 |
| f000 | 5/5 | 5/5 | 5/5 | 0/5 | 0/5 | 0/5 |
| f005 | 5/5 | 5/5 | 5/5 | 0/5 | 0/5 | 0/5 |

f005 没有破坏接口、stand 或 startup，但也没有让 move、stop、full 多通过一次。

### Official 连续物理指标

以下均值越小越好；本实验没有预注册连续指标容差，因此只作相对、描述性裁决。

| 指标 | source | f000 | f005 | f005 相对判断 |
|---|---:|---:|---:|---|
| heading max (rad) | 0.530 | 0.601 | 0.619 | 比两组都差 |
| lateral displacement (m) | 0.482 | 0.565 | 0.551 | 优于 f000、差于 source |
| stop drift (m) | 0.327 | 0.347 | 0.292 | 比两组都好 |
| stop settle time (s) | 4.404 | 4.656 | 4.260 | 比两组都好 |

这是一组混合结果：f005 的停车漂移和 settle time 较小，但机器人仍发生 height collapse，stop 仍是 0/5；同时 heading/lateral 没有保护住 source。不能把“倒下后的漂移更小/更快静止”解释为恢复成功。

## 结论

Phase21 否定了当前最小版本的晋级命题：

- outcome-aware state injection 基础设施是真实可用的；
- 5% curriculum 没有引入接口或 startup 灾难；
- 但 5 updates 后仍然是 move/stop/full `0/5`；
- 连续物理指标只出现混合变化，并非一致改善；
- 因此 `phase21_promoted=false`，`updates25_unlocked=false`。

训练 reward 的上升和 episode length 增长没有转化为 official 物理门改善，再次说明它们不能替代完整闭环评估。

## 下一步

不继续 25-update，也不扫 fraction、seed、reward 或 PD。下一步如继续，应先做纯离线 first-violation/attribution：解释为何 f005 的 stop drift/settle 改善却仍在相同 handoff 后坍塌，以及更大的 support-set KL 是否集中在腰/髋/踝与 previous-action OOD；在有单一、可证伪机制前不再训练。

## 关键证据与边界

- official raw panel：[phase21_outcome_aware_paired_gate.json](../official_x2/phase21_outcome_aware_paired_gate.json)，SHA `8d5ea3c7…2a73`；
- final report JSON SHA `561bd42a…47e5`；
- analyzer：`tools/official_x2/analyze_phase21_outcome_aware_smoke.py`；
- f000 ONNX SHA `20e1859a…6449`；f005 ONNX SHA `5735e4ef…cd34`；
- 没有 25-update、参数扫描、WBT 修改、真机、Git 或百度网盘操作；
- official MuJoCo/AimDK 是仿真证据，不是实机足底力、COP 或 GRF 真值。
