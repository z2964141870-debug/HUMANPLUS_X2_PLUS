# Official X2 foot-contact 与停止制动裁决（2026-08-08）

## 一句话结论

官方 AimDK v1.0 MuJoCo 发布包的足底 ROS publisher 是空实现，不能靠配置恢复；基于官方 odom 的二维速度闭环制动能显著改善 4 秒停止，但 Stage208 没有稳定站立吸引域，因此“减速”和“站稳”必须拆成两个后端技能。

## 假设

1. 官方配置中被注释的 8 个足底 touch sensor 可以通过独立 MJCF/YAML variant 恢复。
2. 若足底 topic 暂不可用，冻结 Stage208 仍可用已有 odom 做短时闭环制动。
3. 在低速、预测双支撑相位切回零速 policy，可能完成稳定停止。

## 干预

- 未改 vendor；建立独立 `variants/foot_contact`：增加 8 个 touch sites/sensors、恢复两组 `foot_contact` model frame，并注册 `/aima/hal/foot/contact`。
- 对官方闭源二进制做符号与反汇编审计。
- 在相同 Stage208、动作模板、官方 MuJoCo、`official_kp_ankle` PD 下增加两种只读评估控制：
  - `velocity_brake`：在 body frame 中令 `(vx_cmd, vy_cmd)=-K(vx,vy)`，命令限幅 0.30 m/s，保持 gait phase；
  - `brake_then_policy`：低速且预测双支撑时切回原零速 policy。

## 对照

- `ramp_policy`：命令与模板幅值在 2 秒内线性归零；
- `event_hold`：零速 policy，满足速度/倾角阈值后冻结动作；
- `ramp_then_hold`：先 ramp 后固定末帧；
- 新方法只改变停止控制器，不训练、不改 checkpoint、不改官方模型参数。

## 结果

### 1. 官方 foot topic 为能力缺口，不是 YAML 错误

- variant 能被官方 simulator 正确加载；topic、ROS 类型、500 Hz worker 均成功注册。
- 但 topic 始终无消息。
- `libaima-sim-module-publisher.so.0.0.0` 中：

```text
ImuFootDataPublisher<std_msgs::msg::Float64MultiArray>::Publish():
    endbr64
    ret
```

即 `Publish()` 直接返回，没有读取 `ModelFootContactData` 或发布 ROS 消息。继续修改 sensor group、QoS 或 print interval 不可能补回缺失实现。

### 2. 速度闭环制动在 4 秒窗口内有效

| 方法 | 时长 | K | 存活 | stop XY drift | stop z min | 末 1 秒平均水平速度 |
|---|---:|---:|---|---:|---:|---:|
| ramp_policy | 2 s | — | 是 | 0.541 m | 0.664 m | 未统一记录 |
| velocity_brake | 4 s | 0.5 | 是 | 0.776 m | 0.568 m | 0.110 m/s |
| velocity_brake | 4 s | 1.0 | 是 | **0.449 m** | 0.562 m | **0.077 m/s** |
| velocity_brake | 4 s | 1.5 | 是 | **0.366 m** | 0.563 m | 0.101 m/s |

`K=1.0` 的速度收敛最好，`K=1.5` 的总漂移最小；二者均优于“简单线性归零”，且评估时间更长。

### 3. 切回零速 policy 会重新加速

`brake_then_policy, K=1.0` 在约 1.96 秒、水平速度约 0.07 m/s、预测双支撑时成功触发切换：

- 制动前段漂移约 0.288 m；
- 切换后额外漂移约 1.063 m；
- 6 秒停止段总漂移 1.145 m，虽未倒但已不满足停止语义。

说明低速切换条件本身可以实现，失败来自接管后的零速 Stage208，而不是触发器。

### 4. 单独持续制动不是永久站立器

`velocity_brake, K=1.0` 延长到 8 秒后倒地：`z_min=0.146 m`、`tilt_max=1.848 rad`、drift=1.742 m。

因此闭环制动是短时 transition skill，不是稳定 stand skill。

## 结论

1. 封存“从官方 v1.0 ROS foot topic 读取接触”的路线；除非拿到含真实 publisher 实现的新 SDK/源码。
2. 保留 `velocity_brake K=1.0~1.5` 作为当前最好的停止过渡候选，但不能宣称停止门已通过。
3. 当前瓶颈从“走路后如何减速”收窄为“低速后谁来稳定站住”。Stage208 的训练分布/技能并不包含可靠 stand/stop，继续扫 latch 阈值信息增益很低。
4. 这支持分层与安全模型切换方向：locomotion policy 负责走和制动，独立 stand/recovery policy 在安全事件处接管。

## 下一步

1. 不再修改足底 publisher 配置；在仿真评估中用官方 odom + gait phase，并把 contact 明确标为模型估计值。
2. 建立独立 stand backend：优先查现有 X2/官方资产是否有可闭环稳定站立的 policy；没有则在 IsaacLab 训练一个很小的 stand/recovery skill，再到官方 MuJoCo 做 sim-to-sim 门禁。
3. 固定切换门：`speed + tilt + predicted double support + dwell time`；先证明 stand skill 单独 20~60 秒稳定，再测 brake→stand，不再让 locomotion policy 同时承担两种互冲职责。
4. 长训解锁条件更新为：walk 与 brake 分别通过，stand backend 通过，brake→stand 切换通过；不要求足底真值 topic 才能开始仿真长训，但必须在报告中区分“真实 contact”和“模型估计 contact”。
