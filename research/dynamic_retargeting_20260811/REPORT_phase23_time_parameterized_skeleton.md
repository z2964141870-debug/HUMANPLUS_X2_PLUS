# Phase23 — X2 接触骨架时间化审计

## 裁决

`TIME_PARAMETERIZED_SKELETON_REJECTED / CONTACT-MANIFOLD INTERPOLATION REQUIRED`

Phase22 的七个硬可行关键帧使用单一预注册方案连接：50 Hz、每段 `0.50 s`、每个关键帧保持 `0.12 s`，root/lower15 使用五次 minimum-jerk，root quaternion 使用同一标量进度的 SLERP。没有时间、阈值、路径或关键帧扫描。

## 结果

| 指标 | 结果 | 门 | 裁决 |
|---|---:|---:|---|
| 全轨迹 active12 最小 signed distance | -3.483 mm | >= -0.500 mm | FAIL |
| hold 支撑接触误差最大值 | 0.455 mm | <= 0.500 mm | PASS |
| 单支撑 hold 摆脚净空最小值 | 24.756 mm | >= 12 mm | PASS |
| lower15 joint-step p95/max | 0.0161 / 0.0165 rad | <= 0.10 / 0.15 | PASS |
| root 水平加速度最大值 | 2.985 m/s² | <= 4.0 | PASS |

总计 `193` 帧、`3.84 s`；`mj_forward=193`、`mj_step=0`、0 GPU、0 训练。

最深穿透发生在 `t=2.86 s`，即 `R_SUPPORT_L_SWING → DS_L_TOUCHDOWN` 转换段：右脚 `-3.483 mm`，左脚仍有 `10.655 mm` 净空。全轨迹有 `64/193` 帧低于 `-0.5 mm`。关键帧 hold 本身全部近似接触正确，因此失败来自关键帧间的关节/root 独立插值，而不是 Phase22 关键帧失效。

## 结论

放慢轨迹不能修复这个几何错误；时间尺度只会降低速度和加速度，不会消除中间构型穿透。因此不得扫描更长 duration，也不得进入 physics/RL。

下一步若继续，应在每个转换段内求解 contact-manifold path：支撑足 signed distance 与 XY 锚点逐帧硬保持、摆脚净空逐帧保持，再在该零空间内最小化平滑项。只在该连续几何层通过后，才加入向前落脚位置或 PHUMA 动作语义。

## 产物

- `phase23_time_parameterized_skeleton_contract.json`
- `run_phase23_time_parameterized_skeleton.py`
- `phase23_result.json`
- `tests/test_phase23_time_parameterized_skeleton.py`
