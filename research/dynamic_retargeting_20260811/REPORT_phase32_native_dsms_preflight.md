# Phase32 — Stage250 原生稳定轨迹 DSMS warm-start 资格门

## 裁决

`NATIVE_DSMS_PREFLIGHT_PASSED / SINGLE SHORT SOLVE STILL LOCKED`

此前并不缺稳定迈步数据。Phase34 已提供 official AimDK closed MuJoCo 的完整稳定直行记录；本阶段验证的是更具体的问题：DSMS 只保存 `qpos/qvel` 作为 shooting state，是否会因为没有 solver/contact hidden state 而立即离开这条成功动力学盆地。

答案是不会。固定 Phase37 的首个 generator-swing / realized-contact mismatch 窗口，从 physics index 2309 起重放 340 ms：

| 指标 | 结果 | 门 |
|---|---:|---:|
| qpos-only/qvel-only replay qpos absmax | `2.66e-15` | `<=1e-6` |
| qvel absmax | `1.35e-13` | `<=1e-4` |
| root-z min | `0.6633 m` | `>=0.55 m` |
| root tilt max | `0.1689 rad` | `<=0.30 rad` |
| recorded torque → equivalent PD target 重建误差 | `2.84e-14` | `<=1e-10` |
| equivalent target joint-limit overshoot | `0` | `0` |
| MuJoCo clipped ctrl vs recorded qfrc_actuator | `0` | `<=1e-10` |

因此 Phase34 原生轨迹可作为下一次 contact-implicit DSMS 短前缀的真实 warm-start；不再需要 PHUMA lunge 失败初态、人工静止 pose 或 Phase27 拼接轨迹作为初始盆地。

## official motor 限幅语义更正

第一次有效计算前的 manifest 解析曾因历史 schema 使用 `artifacts/provenance` 而不是 `assets` 退出；另一次环境选择因 `x2-dsms-agent-b` 缺 `onnxruntime` 在读取前退出。二者均没有进入 `mj_step`，最终在已具备 MuJoCo 3.3.7 的 `h-gpt` 环境执行。

首次结果把“recorded ctrl 必须位于每关节 ctrlrange”误当硬门，发现 340 ms 内有 21 个 command 超范围、最大超出 `31.67`。只读核对确认官方模型 `ctrllimited=1`：MuJoCo 接受上游 PD 的未裁剪 command，再按每 actuator `ctrlrange` 形成实际 actuator force。按正确物理层比较后，`clip(recorded ctrl)` 与记录的 `qfrc_actuator` 逐元素误差为 0。该修正是 actuator 合同更正，不是放松数值门。

## 下一步唯一允许方案

另立 Phase33 单次短求解预注册：

- 仍使用这段 340 ms 原生 qpos/qvel 和 recorded motor command warm-start；
- soft reference 保留 root/前进/全身语义，只新增右脚至少 100 ms 离地和 12 mm clearance 的 contact objective；
- official raw motor、MuJoCo 3.3.7、20 个 1 kHz substep / 20 ms node；
- 1 CPU thread、`nice 10`、MUMPS、最多 50 iterations、0 GPU；
- 不扫权重、不重试；输出必须先过 defect、root/tilt、摆脚、支撑滑移和语义门，才可称动态 teacher；
- 本阶段没有启动该求解，也没有解锁训练。

## 产物

- `phase32_native_dsms_preflight_contract.json`
- `run_phase32_native_dsms_preflight.py`
- `phase32_native_dsms_preflight_result.json`
- `tests/test_phase32_native_dsms_preflight.py`

边界：这是 direct official MuJoCo 3.3.7 的 deterministic suffix，不是 closed ROS 在线回放，也不是实机接触真值。
