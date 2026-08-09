# X2 AimDK v1.0 官方模型 Canonical Contract

日期：2026-08-09

## 裁决

- 官方 MJCF 是 31 个执行关节；官方示例 RL policy 是 29 维并仅排除两个头关节；当前速度后端是腿腰 15 维。三者不得混写。
- WBT 第一阶段采用 29DoF body contract，头 yaw/pitch 锁定 model nominal；不涉及灵巧手。
- 官方足底接触由每脚 12 个半径 5 mm 的球点构成，foot tracking proxy 为 ankle-roll link；不得继续沿用旧推测 offset。
- 官方 MJCF 没有专用 palm/sole tracking site，左右 wrist-roll 与 ankle-roll link 只能作为显式 proxy，后续若获得官方 tracking frame 必须升版 contract。

## 控制边界

- MJCF：`31` DoF。
- 官方 RL 示例：`29` DoF，50 Hz，action scale 0.25。
- 速度保底后端：`15` DoF。
- Any2Any/WBT：目标 29DoF，但 action scale 与 observation contract 必须由忠实基线单独预注册，不能静默照搬官方舞蹈 ONNX。

## 坐标与接触

- 世界系：右手系，+Z 向上；floating root 位于 pelvis。
- pelvis/torso：`pelvis` / `torso_link`。
- feet：`left_ankle_roll_link` / `right_ankle_roll_link`。
- hands：`left_wrist_roll_link` / `right_wrist_roll_link`。
- 足底镜像点 permutation：`[2, 3, 0, 1, 5, 4, 7, 6, 9, 8, 10, 11]`。

## 镜像/FK 门禁

- 关节 round-trip 最大误差：`0.000e+00` rad。
- body position 最大误差：`2.105e-03` m。
- body rotation 最大误差：`5.888e-03`。
- 足底接触点最大误差：`0.000e+00` m。
- body 残差不是映射错误：官方左右 CAD 存在毫米级非对称；预注册容差为位置 3 mm、旋转矩阵元素 0.007。

## 版本与限制

- `x2.xml` SHA-256：`3ff43f05beb57412a804ba9fe05cd9adcdfce78e9ce73a95a71ac58ad20d91a3`。
- `default.yaml` SHA-256：`3d10d8147d9f4ac155070441ea7f6b47d6af3238d0b37c3a81a5eeb41b27ad23`。
- `motion_control.yaml` SHA-256：`07fa8a0151d14c2b0b9f8659302f96dec21fe3b3921f9dbcaab1150dc1c2e400`。
- 官方 README 明确该 RL 示例只在仿真验证，真机仍需重新参数适配；本 contract 不授权真机发送命令。
- model nominal 与官方 RL 示例 deeper-crouch default 同时保留，任何实验必须声明使用哪一套，禁止混用。
