# BASE Phase24：完整物理观测支持域交权

## 游戏任务

- [x] 从 Phase19 recorder-v2 eligible 行冻结物理支持域，不读取本轮结果调阈值。
- [x] `base_ang_vel` 与 `projected_gravity` 分组标准化、分别计算 NN LOO-p95，防止尺度支配。
- [x] gate 默认关闭，且只能在 Phase23 `previous-action + generator DS` 之上取 AND。
- [x] 复用 Phase23 既有 5+5 对照，只新增 5 条 official candidate。
- [x] 5/5 都真实进入完整门并调用 recovery，不存在“靠不交权站住”的假改善。
- [ ] stop/full 通过。
- [ ] 解锁训练。

## 假设

Phase23 仍倒，是因为 action history 与 generator DS 虽已匹配，但交权时 `projected_gravity` 或 `base_ang_vel` 尚未回到 recovery 的 Phase19-v2 训练支持域。

## 干预

候选只有在下列四项同时成立时交权：

1. previous-action robust NN 距离 `<=0.548358`；
2. generator double support；
3. base angular velocity robust NN 距离 `<=2.086444`；
4. projected gravity robust NN 距离 `<=0.384609`。

物理两组阈值来自同一 432 行 Phase19-v2 eligible 数据的独立 LOO-p95；stop horizon 内不满足则继续原 Stage306 brake，不强制交权。

## 对照

- Phase23 fixed：stop+2.0 s 固定交权，5 条；
- Phase23 action-gated：previous-action + generator DS，5 条；
- Phase24 physical-gated：再增加两组物理支持，5 条。

Phase23 原始 trace 与 hash 不变。Phase24 的新代码全部默认关闭，纯测试证明关闭时不会进入物理门，因此没有为旧对照额外消耗 10 条物理 episode。

## 结果

| 指标 | fixed | action-gated | physical-gated |
|---|---:|---:|---:|
| valid | 5/5 | 5/5 | 5/5 |
| recovery 实际调用 | 5/5 | 5/5 | 5/5 |
| timeout | 0/5 | 0/5 | 0/5 |
| stand / startup | 5/5 / 5/5 | 5/5 / 5/5 | 5/5 / 5/5 |
| move / stop / full | 0/5 / 0/5 / 0/5 | 0/5 / 0/5 / 0/5 | 0/5 / 0/5 / 0/5 |
| handoff 时刻中位数 | 2.00 s | 2.36 s | 3.24 s |
| tilt>0.30 时刻中位数 | 3.32 s | 4.12 s | 4.94 s |
| root-z<0.45 时刻中位数 | 3.90 s | 4.74 s | 5.52 s |
| handoff→height collapse 中位数 | 1.90 s | 2.36 s | 2.28 s |
| heading max 中位数 | 0.572 rad | 0.504 rad | 0.627 rad |
| lateral 中位数 | 0.554 m | 0.477 m | 0.558 m |
| stop drift 中位数 | 0.274 m | 0.257 m | 0.238 m |
| settle 中位数 | 4.34 s | 5.14 s | 5.94 s |

Phase24 等待为 `0.36–1.58 s`，五条交权时四项条件均通过。五条事件顺序仍全部为 `tilt → height collapse`。

最关键的辨别量不是绝对倒地时刻，而是交权后的倒地延迟：Phase24 为 `2.28 s`，并未优于 Phase23 action-gated 的 `2.36 s`。绝对倒地晚约 `0.78 s`，主要来自 brake 多保持约 `0.88 s`，不能称为 recovery 能力改善。

## 结论

完整物理门准确运行，但最终假设失败：`previous-action + generator DS + projected gravity + base angular velocity` 仍不是安全 recovery 的充分条件。Phase24 没有改变失稳事件顺序，也没有产生任何 stop/full pass；它只是把交权和随后倒地一起向后平移。

因此不能继续按“再加一个门、再延后一点”推进，也不能解锁训练。较小 stop drift 属于倒地轨迹差异；更大的 settle time 反而更差，均不能包装为恢复成功。

还有一个明确边界：当前 432 行支持集同时包含 `success_safe` 与 `critical_from_failure`。落入 union support 只说明输入不像全局 OOD，不等于已经进入成功吸引域。

## 下一步

停止扩大时间门。若继续，唯一有信息增益的离线工作是：在不跑 official 的前提下，把五个 Phase24 handoff 点对 Phase19-v2 做 role-aware 最近邻归因，区分其更接近 `success_safe` 还是 `critical_from_failure`，并比较 recovery 首 0.5 s action/state divergence。

- 若 handoff 更接近 failure role：现有 union-support gate 过宽，下一候选应是 success-basin reachability，而不是继续训练 actor。
- 若 handoff 已接近 success role 仍同序倒：说明单帧支持域不足，问题在闭环 transition/dynamics，后续必须使用成功 closed-loop suffix 或改变 recovery backend，不能靠静态门解决。

## 证据边界

- JSON：`reports/official_x2/phase24_physical_handoff_gate.json`
- physical manifest：`manifests/x2_phase24_physical_handoff_support_gate.json`，SHA `58484564873d9f13fa105607d1d44a33f5969f6015e2fe507d2dcd7ea1387ffd`
- 官方 AimDK MuJoCo only；无训练、无真机、无 WBT/Git/百度修改。
