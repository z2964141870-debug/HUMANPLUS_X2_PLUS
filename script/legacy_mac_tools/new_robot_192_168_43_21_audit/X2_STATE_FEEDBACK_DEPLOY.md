# X2 Sonic policy 真机状态回读

## 结论

Sonic 的 proprioception 必须使用真实的 `qpos/qvel`。旧版
`x2_v2_teleop.py` 把上一帧发送命令当作 qpos，并把 qvel 全部置零；这不是
闭环反馈。

本目录现在提供一个只读桥接：

```text
AimDK ROS2 state topics --(read only)--> x2_state_feedback.py
                                      --(UDP 127.0.0.1:50041)--> x2_v2_teleop.py
```

桥接只有在四组关节都完整且名称匹配，并且 torso IMU 新鲜时才发 UDP 包。
当前 X2 的 odom topic 没有发布者，因此默认使用 torso IMU 的姿态和角速度；
如果以后 odom 恢复，可显式使用 `--base-source odom`。任意必需输入为空或过期
都会 withholding feedback；policy 端也会拒绝进入衣服模式或切回停发模式。

## 已确认根因与修复

2026-08-18 的只读检查确认，机器人 HAL 实际发布了完整关节数据：

- leg/waist/arm/head 分别为 `12/3/14/2` 个关节；
- `state.value=0`，位置和速度字段有效；
- SoC0 HAL 使用 AimRTE/AimDK 消息版本 `0.8.18`。

原 SoC1 工作空间的 `JointStateArray` 只有 `header + joints`，而 HAL 的 0.8.18
定义是 `header + state + joints`。旧定义会把同一条有效消息误解码成空数组。
现已建立独立兼容工作空间 `~/aimdk_ws_0_8_18`，旧的 `~/aimdk_ws` 未修改。

## 只读验证

在 X2 的 SoC1 上先加载 teleop、ROS 和匹配的 AimDK 消息环境：

```bash
conda activate teleop
source /opt/ros/humble/setup.bash
source ~/aimdk_ws_0_8_18/install/setup.bash
ros2 topic type /aima/hal/joint/leg/state
ros2 topic echo --once /aima/hal/joint/leg/state
```

合格的消息必须包含 12 个带正确 `name/position/velocity` 的 `joints`；
看到 `joints: []` 时不要进入 policy 模式。

## 启动只读状态桥

终端 1：

```bash
conda activate teleop
source /opt/ros/humble/setup.bash
source ~/aimdk_ws_0_8_18/install/setup.bash
python ~/x2_v2_teleop/x2_state_feedback.py \
  --udp-host 127.0.0.1 --udp-port 50041
```

只有在终端 1 出现 `feedback sent` 后，才在终端 2 做离线 policy 演练。程序
现在默认就是 dry-run，AIMRT 网络发送处于锁定状态：

```bash
conda activate teleop
python ~/x2_v2_teleop/x2_v2_teleop.py \
  --feedback-bind 127.0.0.1 \
  --feedback-port 50041 \
  --feedback-timeout 0.15
```

以上命令只计算目标并打印 dry-run 数据，不发送真机控制。`--allow-open-loop`
只用于完全离线检查合成 proprioception；它不能和真机解锁参数同时使用。

## 当前安全边界

控制代码现在会在以下任一情况停发并退回 RC 模式：

1. 反馈缺失、过期、倒序/重复或有效位不完整；
2. 状态含非有限值、关节速度或机身倾角超限；
3. 策略目标异常、上一目标与实测状态偏差超限；
4. 推理异常，或发送前反馈再次过期。

目标还会经过站姿包络和每周期变化率限制。代码具备显式真机解锁入口，但本次
没有部署、启动或测试该入口；首次真机测试仍需人在机器人旁、扶机并准备物理
急停。
