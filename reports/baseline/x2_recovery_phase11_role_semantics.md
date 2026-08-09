# BASE Phase11：Recovery 角色语义修复与官方配对门

日期：2026-08-09
状态：完成；角色合同通过，f005 未晋级，25-update 与长训继续锁定。

## 假设

Phase9 把 f005 同时放进 stationary 与 recovery 角色，混入了起步退化；若站立/起步始终由 source stand actor 负责，只在 `curriculum_then_policy` 的减速 transition 完成后把控制交给 f005，f005 可能改善停止恢复而不破坏前段。

## 干预

只修复 `curriculum_then_policy` 的控制角色语义：

```text
stop transition 0--2 s：main actor
transition 后，无 recovery：stationary actor（与旧路由逐项等价）
transition 后，有 recovery：recovery actor
```

交权时 `previous_action` 与 `issued_action` 均写入实际接权的同一个 slot。其他 stop controller、移动 actor、站立逻辑、门禁和物理参数均未修改。

纯测试为 `20 passed`：无 recovery 为 `main×100 -> stationary×300`；有 recovery 为 `main×100 -> recovery×300`；静态审计确认 stand/start/main 与减速段未改。

## 对照

两组均固定：

- moving：Stage306；
- stationary：source stand backend i150；
- stiff：1.2；fixed upper；
- prepare/stand/move/stop：0.2/2.0/5.2/8.0 s；
- stop：`curriculum_then_policy`，transition=2.0 s；
- 官方 AimDK MuJoCo，完整 episode 各 5 条。

唯一变量：

- control recovery：source stand i150；
- candidate recovery：Phase9 f005-u5。

冻结 SHA 与逐条原始结果见 [JSON](./x2_recovery_phase11_role_semantics.json) 和 [官方面板](../official_x2/phase11_recovery_role_paired_gate.json)。

历史复现边界：runner 冻结的是 Phase11 当时的 adapter SHA `e682...dae3` 与 handoff SHA `9277...4644`；Phase13 以后当前源码已分别演进为 `4663...05bd` 与 `cd4d...1a32`。旧源码没有另存一份独立快照，因此不能拿当前文件冒充 Phase11 的精确源码复现；但历史报告、runner 和结果仍明确绑定原始 SHA，这不是结果文件损坏。

## 结果

角色合同首先成立：10/10 条 episode 均严格为 `main=360, stationary=100, recovery=300`，所以这次确实测到了 recovery actor，而不是旧合同下的空实验。

| 指标 | source recovery | f005 recovery |
|---|---:|---:|
| full | 0/5 | 0/5 |
| stand | 5/5 | 5/5 |
| startup | 5/5 | 4/5 |
| move | 0/5 | 0/5 |
| stop | 0/5 | 0/5 |
| heading max mean | 0.566 rad | 0.652 rad |
| lateral mean | 0.536 m | 0.603 m |
| stop drift mean | 0.303 m | 0.276 m |
| stop settle mean | 4.396 s | 5.200 s |
| move signed pitch mean | -0.142 rad (-8.12°) | -0.145 rad (-8.30°) |
| stop signed pitch mean | -0.818 rad (-46.87°) | -0.774 rad (-44.34°) |

f005 的停止漂移均值下降约 9.1%，停止后仰均值少约 0.044 rad，但两组均在 stop 阶段跌到约 0.1 m root height，stop/full 仍为 0/5；同时 f005 settle time 变慢约 18.3%。这只是弱连续量信号，不是恢复成功。

candidate 中一次 startup 未通过发生在 recovery 尚未获得控制权之前，因此不能归因于 f005；同理，两组的 move heading/lateral 差异主要反映官方仿真重复间波动，不能作为 recovery 效果。

## 结论

控制角色语义漏洞已经修复，而且验证合同可信；但 Phase9 f005 并没有学会从匹配减速后的真实交权状态恢复。它不能晋级，不能以“stop drift 稍小”解锁 25-update 或长训。

## 下一步

保持训练锁定。先用这 10 条现有 trace 定位交权后的第一个物理分叉：root pitch/height、q/dq、previous action 或 recovery action 中哪个最先越界；只有形成新的、可证伪的 recovery 训练目标或更匹配的 handoff-state 分布后，才值得再做最小 smoke。
