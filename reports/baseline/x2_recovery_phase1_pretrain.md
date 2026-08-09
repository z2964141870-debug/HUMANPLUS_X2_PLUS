# X2 BASE Recovery Phase 1：长训前准备

日期：2026-08-09
状态：**reset 地基已打通；长训仍锁定。**

## 总体判断

Stage335 的 90 个官方 MuJoCo 停车边界状态可以安全、精确地注入 IsaacLab 的独立 stand/recovery backend；接口、93D observation、15D action 和 PPO runner 均已跑通。但目前只能证明“课程能运行”，不能证明“长训会改善”。尤其 `50%` recovery reset 比例过于激进，不应直接用于长训。

没有修改或覆盖 `BASE_LOCOMOTION`、`BASE_TRANSITION`、stand 源 checkpoint，也没有触碰 WBT/retarget 文件。

## 数据契约审计

- 数据：`stage335_stage306_stiff_fixed_stop_recovery_states.npz`
- SHA-256：`4d8ce06b6e8dd9b43363dadedb781e9feb75f72ac092b475de2a1e2772545013`
- 总状态：90
- eventual-pass / eventual-fail：63 / 27
- 五个源 trace：21 / 8 / 21 / 21 / 19
- root-z：`0.6026–0.6711 m`
- 最大合成倾角：`0.4812 rad`
- 最大水平 body speed：`0.5021 m/s`
- projected-gravity 单位范数最大误差：`5.96e-8`
- `q/dq`：`90×31`；previous action：`90×15`；gait phase：`90×4`

重要限制：这批数据只能称为**物理 reset-state 数据**。IsaacLab 在 reset event 之后还会清空 action manager 并把 episode clock 归零，所以不能诚实恢复保存的 `previous_action/gait_phase`。本实现只恢复 root pose、root velocity、31DOF q/dq，不把它伪装成完整的 93D observation replay。

## 最小注入点

假设：现有 stand backend 缺少“从停车边界扰动状态恢复”的初始状态覆盖。

干预：在继承事件后追加唯一的新 term：

```text
reset_base
→ reset_robot_joints
→ recovery_state_reset（只覆盖抽中的 env）
```

对照：`recovery_fraction=0` 时 term 严格 no-op，旧 reset 不变；未抽中的 env 始终保留旧行为。

结果：Event Manager 运行时确认顺序正确；关节通过名字重排而非假定数组顺序；软关节限位和速度限位在写入前生效。

结论：这是当前 stand backend 的最小、可撤销、与 WBT 隔离的 reset-state 注入点。

## 实验 1：zero-update 物理 reset smoke

假设：MuJoCo 提取状态可以在 IsaacLab 中形成合法初始物理状态。

干预：16 env，`fraction=1.0`，balanced 抽样 8 个 eventual-pass、8 个 eventual-fail；训练更新为 0。

对照：继承 reset 先运行，新 term 最后覆盖。

结果：

- 16/16 被注入；
- 关节状态最大误差 `0 rad`；root-z 最大误差 `0 m`；
- policy observation `16×93`，action dim `15`；
- 状态与 observation 全部有限；
- 16/16 撑过首个零动作物理步。

结论：物理 reset 接口通过，可以进入极短 runner smoke。

## 实验 2：同 seed、单 update A/B

共同条件：seed 47、32 env、768 timestep、只做 1 次 PPO update、从同一个 stand `model_150.pt` weights-only 分叉。

| 指标 | fraction=0 对照 | fraction=0.5 balanced |
|---|---:|---:|
| mean reward | -7.59 | -9.47 |
| mean episode length | 9.58 step | 10.20 step |
| bad-orientation 日志信号 | 0.2409 | 0.1927 |

结果：两支均完成、runner 无接口错误、各自在新目录写出 `model_151.pt`。源 checkpoint 前后 SHA 均为 `4da931cf...251`，未被修改。

结论：恢复状态没有让 runner 崩溃，也没有在这一轮造成更短 episode；但 reward 更低，符合它引入更困难速度/倾角状态的预期。单 update 的差值不能作为性能结论，两个 smoke checkpoint 都**不晋升**。

## 测试与修复记录

- focused tests：`6 passed in 0.83s`。
- 首次环境 smoke 暴露相对 gait-template 路径错误；已改为已冻结的绝对资产路径。
- 第二次暴露模板保存的 action scale 是 stand backend 的 `2×` 契约；probe 已显式对齐，没有放宽检查。
- 最终 zero-update 与 1-update 均通过。

## 长训门

当前不解锁长训。下一步最小实验应是：

```text
同 checkpoint / seed / optimizer-reset
fraction = 0, 0.05, 0.10
每支最多 5 update
→ 冻结 checkpoint
→ 官方 MuJoCo stand-stop 门禁
```

只有低比例 recovery 分支在多 seed 下提高恢复成功率，同时不损伤 nominal stand，才允许逐步提高 reset 比例。不要从本轮任一 `model_151.pt` 直接长训。

## 实现文件

- `tools/official_x2/recovery_reset_curriculum.py`
- `tests/test_recovery_reset_curriculum.py`
- `scripts/probe_x2_recovery_reset.py`
- `scripts/run_stage336_recovery_reset_env_smoke.sh`
- `scripts/run_stage336_recovery_reset_smoke.sh`
- `scripts/train_x2_stage221_official_ankle_match.py`
