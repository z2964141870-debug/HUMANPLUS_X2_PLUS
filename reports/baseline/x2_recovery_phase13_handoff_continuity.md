# BASE Phase13：Handoff continuity 单变量 A/B

日期：2026-08-09
状态：完成；机制通过，物理门失败，不晋级、不训练。

## 假设

Phase12 发现 moving→recovery 在 stop 2.0 s 交权时产生约 0.43–0.47 rad L2 的腿腰目标跳变。如果这是停止后坍塌的主要原因，用 0.5 s C2 smoothstep 消除首帧跳变并限制每帧目标变化，应明显推迟或避免 tilt/root-z 越界。

## 干预

新增 `--curriculum-recovery-handoff-blend-seconds`，默认 `0`。它只在 `curriculum_then_policy` transition 完成后生效：

- 从交权前最后一帧实际 physical lower targets 出发；
- 以 quintic C2 smoothstep 在 0.5 s 内到达每帧 live recovery targets；
- 每帧从最终 physical targets 反算实际 normalized action；
- 同步 recovery slot 的 `issued_actions` 与 `previous_actions`；
- trace 分开记录实际 action、未执行的 recovery proposal、physical target。

因此 actor 下一帧看到的 last action 与实际执行目标逐项一致，没有重新引入 train/eval history mismatch。`blend=0` 不进入上述分支，保持旧物理路径。

纯测试与静态合同共 `33 passed`。

## 对照

官方 AimDK MuJoCo 两组各 5 条，唯一变量为：

- control：blend `0.0 s`；
- candidate：blend `0.5 s`。

两组均固定 Stage306 moving actor、source stand i150 stationary/recovery、stiff 1.2、fixed upper、相同 matched start/move/stop 时序。10/10 条 role counts 均严格为 `main=360, stationary=100, recovery=300`。

## 结果

### 机制验证

| 指标（5 条中位数） | blend 0.0 | blend 0.5 |
|---|---:|---:|
| 实际 normalized action 交权 L2 | 1.317 | 5.96e-8 |
| 未平滑 recovery proposal 交权 L2 | 1.317 | 1.316 |
| physical target 交权 L2 | 0.420 rad | 0.000 rad |
| 前 0.5 s 最大单步 physical target L2 | 0.420 rad | 0.026 rad |
| 0.5 s endpoint 对 live recovery target L∞ 误差 | 8.94e-9 rad | 5.96e-9 rad |
| next previous-action 同步 L∞ 最大误差 | 0 | 0 |

过渡器确实完成了目标：首帧 target jump 被完全消除，前 0.5 s 最大单步变化下降约 94%，结束点精确回到 live recovery，actor history 与实际执行逐项一致。未平滑 recovery proposal 仍有约 1.32 L2 跳变，证明改善来自物理过渡器，而非上游策略偶然变平滑。

### 物理结果

| 门禁 | blend 0.0 | blend 0.5 |
|---|---:|---:|
| full | 0/5 | 0/5 |
| stand | 5/5 | 4/5 |
| startup | 5/5 | 4/5 |
| move | 0/5 | 0/5 |
| stop | 0/5 | 0/5 |

blend0 的 tilt/root-z 首次越界中位数为交权后 `1.34/1.92 s`；blend0.5 全 5 条中位数仍为 `1.34/1.92 s`。candidate 的 r2 在 stand 阶段就已经倒地，发生在 recovery 获得 authority 之前，不能归因于 blend，属于官方仿真跨 episode 初始化/重复波动。

排除这条 pre-handoff 失败后，4 条健康 candidate 的 tilt/root-z 越界中位数约为 `1.37/1.95 s`，仅比 control 推迟约 `0.03 s`，没有工程意义；stop 仍为 0/4。

健康 candidate 的 stop drift 中位数约 `0.266 m`，control 为 `0.282 m`，略有改善；settle time 约 `4.38 vs 4.36 s`，基本不变。机器人仍坍塌到约 0.10 m root height，因此不能用连续指标的微小变化宣称恢复成功。

## 结论

Phase12 的交权跳变是真实且可修复的问题，但不是当前停止后倒地的充分根因。即使首帧 physical target 完全连续、history 完全同步，机器人仍按几乎相同的时间尺度倾斜并坍塌。

因此：

- handoff continuity 基础设施可以保留，默认仍为 0；
- 0.5 s blend 不晋级为最佳控制配置；
- 不继续扫描 blend 时长；
- 不解锁 25-update、长训或 recovery 训练。

## 下一步

下一轮信息增益最高的工作不是再调 blend，而是离线比较交权后 0–1.5 s 内 recovery actor 的目标、实际 q/dq、root pitch 与 source stand 稳态分布，判断 recovery actor 是否从未见过“低速但仍带单支撑/动量”的 handoff state。只有证明训练状态分布缺口并形成新的可证伪 reset/recovery objective，才值得再做最小训练 smoke。

完整逐条 JSON：[phase13_handoff_continuity_ab.json](../official_x2/phase13_handoff_continuity_ab.json)
