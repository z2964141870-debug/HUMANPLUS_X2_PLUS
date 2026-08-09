# BASE Phase25：Role-aware divergence 离线裁决

## 游戏任务

- [x] 只读使用 Phase19/20 冻结 `success_safe=366`、`critical_from_failure=66` 支持行。
- [x] 对每个字段组单独稳健标准化，避免 31D q/dq 支配 3D gravity/velocity。
- [x] 类内 LOO-p95 归一化后做 success/critical 最近邻归属。
- [x] 只有 LOO balanced accuracy `>=0.65` 的组可参与时序解释。
- [x] 连续 5 tick（0.10 s）才算稳定偏离，未扫描窗口或阈值。
- [x] 分析 Phase24 五条 handoff 前0.5 s、后0–1.0 s；没有新 official、训练或控制改动。

## 假设

需要区分两个机制：

1. recovery actor 接管后首先输出训练支持域外动作，随后物理状态被带坏；
2. 交权时物理状态已经不在有效支持域，actor action 只是随后跟随漂移。

## 干预与对照

本阶段无物理干预。对照是 Phase19-v2 两种冻结 state role，查询是 Phase24 五条 candidate 的已有 trace。

距离门均由 reference 自身 LOO 得到；没有使用 Phase24 结果反调阈值。所有动作只按 history/state 解释，绝不是 expert imitation label。

## LOO 判别能力

| 字段组 | balanced accuracy | 可用于角色判断 |
|---|---:|---:|
| base linear velocity | 0.841 | 是 |
| base angular velocity | 0.877 | 是 |
| projected gravity | 0.999 | 是 |
| joint position | 0.932 | 是 |
| joint velocity | 0.568 | 否 |
| previous action | 1.000 | 是 |
| gait phase | 0.543 | 否 |
| actual issued action | 0.977 | 是，但仅历史分布 |
| actor proposal | 1.000 | 仅诊断，不参与因果先后 |
| root z + tilt | 0.927 | 是 |

joint velocity 与 gait phase 的角色判别力不足，后续不以其 success/critical 标签下结论。

## 结果

### 交权瞬间

- 所有可判别组的最近角色均为 `success_safe`，没有任何字段稳定归入 `critical_from_failure`。
- 但 `joint_position` 已有 5/5 超出两类 union LOO-p95。
- `root z + tilt` 有 2/5 在交权瞬间 OOD，并在交权后约 0.02 s 达到 5/5 稳定 OOD。
- `previous_action`、base angular velocity、projected gravity、actual issued action 在交权瞬间均为 0/5 OOD，符合 Phase24 门的直接检查。
- f005 的 unblended actor proposal 为 5/5 OOD；但 reference proposal 来自另一控制器，不是 f005 的 expert label，不能据此声称“actor 首先犯错”。

### 交权后最早稳定偏离

| 字段 | 稳定 union-OOD 中位时刻 |
|---|---:|
| joint position | 0.00 s（5/5） |
| root z + tilt | 0.02 s（5/5） |
| previous action | 0.14 s（5/5） |
| projected gravity | 0.24 s（5/5） |
| base linear velocity | 0.45 s（4/5） |
| base angular velocity | 未稳定 OOD |
| actual issued action | 未稳定 OOD |

在 1.0 s 窗口内，没有字段在多数 episode 中稳定转入 `critical_from_failure` 最近邻；主要现象是离开 union support，而不是清楚地从 success basin 切换到已知 critical basin。

## 结论

现有证据不支持“recovery 一接管，实际执行动作先 OOD，然后才把物理带坏”。实际 issued action 在 1 s 内没有稳定 OOD；相反，joint position 在交权首帧已 5/5 OOD，root posture 几乎同步离域，随后 previous-action 与 gravity 才离域。

更准确的裁决是：Phase24 的四项门只保证 action-history、generator DS、angular velocity 和 gravity；它没有保证完整 q/root 状态进入 recovery 的闭环支持域。交权点看起来更接近 success role，但已经位于两类已知局部轨迹之外。当前失败更像“交权状态不完整/单帧门不足”，而不是已经证实的 actor 首发异常。

这仍不是因果定论：Phase24 普通 trace 没保存完整 controller snapshot，无法逐字段比较 latch、内部 filter 与 clock；也没有真实足底接触。时序只能推翻简单的 action-first 说法，不能证明未观测接触或 PD 是根因。

## 下一步

本阶段按任务卡停止，不新增实验、不解锁训练。若未来继续，不应再叠加单字段时间门；需要先定义一个包含 q/root 的 success-basin closed-loop state contract，或直接使用成功 stateful suffix 验证连续多帧可达性，而不是把单帧 union-support 当作安全吸引域。

## 证据

- 机器可读结果：`reports/official_x2/phase25_role_aware_divergence.json`
- 工具：`tools/official_x2/audit_phase25_role_divergence.py`
- 未训练、未运行 official、未改 adapter 行为、未上真机。
