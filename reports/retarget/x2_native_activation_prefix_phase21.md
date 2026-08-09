# X2 Activation Prefix Replay Phase21

- 裁决：**PHASE21_SOURCE_PREFIX_REJECTED_PRESCRIBED_FAILED**。
- 唯一变量是激活/初始化顺序；模型、PD、reference定义、Phase15 gate均未改变。
- reference始终为recorded actual q/root/model-contact；command event只作control input。

## 假设

Phase15 prescribed误差若主要来自动态中段冷启动，则JOINT激活telemetry后，从首个完整snapshot重放至少4秒稳定前缀，再进入RL段，应通过相同trackability门。

## Capture资格

- subscription-ready `0.076s` → JOINT ack/receipt `1.038s` → first complete `5.437s`。
- complete+4s 达成 `9.437s`；RL ack/receipt `10.260s`；可评分prefix `4.823s`；RL覆盖 `20.338s`。
- qualification：`True`，失败项 `[]`。

## Replay结果

| mode | survival | q RMSE(all31/active29/head2) | body rel-pos p95 | body ori p95 | contact | SS(ref/real) | slip p95 | torque sat |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| prescribed_root_trackability | 25.161/25.161s | 0.4896/0.5047/0.1487 | 0.6431m | 2.7398 | 0.578 | 0.219/0.357 | 3.134m/s | 0.1721 |

- prescribed gate：`False`，失败项 `['source_joint_step', 'source_joint_velocity', 'q_trackable', 'body_position_trackable', 'body_orientation_trackable', 'contact_agreement', 'slip_bounded', 'torque_saturation_bounded', 'target_limits_bounded']`。
- free-root executed：`False`；prescribed失败即停。
- source自身：joint-step p95/max `0.1059/0.5821rad`，velocity p95/max `5.4149/31.3145rad/s`，限位违反率 `5.105%`，reference root-z最低 `0.066m`；所以这不是“稳定前缀”。

## 零物理分段归因

| source segment | joint-step p95/max | dq p95/max | limit violation | root-z min/final | flight |
|---|---:|---:|---:|---:|---:|
| JOINT prefix | `0.0108/0.1215rad` | `0.530/8.937rad/s` | `4.479%` | `0.066/0.077m` | `9.9%` |
| mode boundary ±100ms | `0.0880/0.2635rad` | `4.609/16.396rad/s` | `4.516%` | `0.077/0.096m` | `50.0%` |
| RL internal after 100ms | `0.1149/0.5821rad` | `5.872/31.315rad/s` | `5.268%` | `0.075/0.229m` | `45.9%` |

- 冻结的 step/dq 最大值门失败集中在 **RL内部**，并非JOINT插值或JOINT→RL切换边界；最大异常发生于 `19.877s` 的 `right_knee_joint`。
- JOINT prefix虽在q/dq上相对连续，但root-z在首个complete snapshot后约 `1.581s` 已跌破 `0.42m`，RL切换前机器人已经倒下。
- 既有replay JSON没有保存逐帧q/body误差，因此在不重跑physics的约束下，不能判断aggregate q RMSE由哪一段主导。
- 完整分段证据见 `x2_native_activation_prefix_phase21_source_segments.md/json`。

## 边界

- JOINT阶段真实active31 context被保留；RL阶段空head callback不清除最后source head command，未伪造29/31。
- subscriber receipt不是publisher timestamp；solver/contact warmstart仍不可得；source稳定不冒充replay稳定。
- contact为模型/官方MuJoCo碰撞，不是实机GRF/COP/wrench。

## 结论

- 结果：时间资格通过，但captured actual reference自身违反冻结的连续性、速度和关节限位门，同时prescribed失败；未运行free-root。
- 结论：整个JOINT prefix不能作为计分reference；但首个complete state/command context仍可只作为初始化seed。当前大step/dq异常来自RL内部，故也不能未经门禁直接把整段RL当作“稳态reference”。
- 边界：这次混合模式source失败不能归因于WBT/Any2Any，也不能据此宣称冷启动完全无效。
- 下一步：停止event replay；后续若继续，只从现有source中按冻结门选择稳态RL片段作reference，并保留prefix首帧作初始化，不调增益/时移、不训练。
