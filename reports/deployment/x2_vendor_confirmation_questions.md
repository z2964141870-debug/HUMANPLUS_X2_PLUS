# X2 v1.0 需要用户/智元确认的事实

状态：仅问题清单；未升级 SDK，未向真机发送任何命令。

## 已知边界

- AimDK v1.0 提供 X2 MuJoCo 与 ONNX 部署示例，但 README 明确示例只在仿真验证，真机需要重新适配参数。
- 公开材料尚未证明“原生 MC 独占腿腰 + 外部 HAL 只控上肢”能够并行。
- 旧 v0.8.2 实测中，`GetAllJointState` 可取得 q/dq，而若干分组 state topic 与 odom topic 曾返回空 payload。
- 当前没有核实到真实 foot wrench、COP、COM 或 centroidal momentum 接口。

## P0：决定是否能部署

1. 在 X2 v1.0 developer / `RL_DEFAULT` 模式下，原生 MC 能否继续独占腿和腰，同时外部程序只控制双臂（可选头部）？若支持，请给出 joint-group ownership API 和仲裁规则。
2. MC、HAL joint command、`RL_DEFAULT` 与遥控器命令同时存在时，精确优先级、超时/watchdog 与安全回退状态是什么？
3. 当前 31DoF T2.5 的控制器/固件是否兼容 AimDK/SDK v1.0？从 v0.8.2 升级是否可逆、是否受官方支持？
4. 真机 RL command 的权威 31 关节顺序、单位、命令语义（绝对位置/默认位残差/速度/力矩）、Kp/Kd 所有权、频率与限幅分别是什么？

## P1：决定 observation 与安全交权

5. v1.0 实时 q/dq/effort、base orientation/angular velocity、base linear velocity 的权威数据源是什么？分组 state topic 应该有 joints，还是仍需 `GetAllJointState`？
6. command/state 时间戳是否硬件同步，使用哪个时钟域，端到端 latency/jitter 如何测量？
7. v1.0 是否提供支持的原生 locomotion/WBC 接口，可同时接收 `vx/vy/wz` 和上肢 target？腰部归 locomotion 还是上肢所有权？
8. 本机是否提供左右脚 contact、法向力、六维 wrench 或 COP？请注明接口、单位、频率和“传感器实测/模型估计”。
9. 进入/退出 `RL_DEFAULT`、zero torque、damping、stand 与 emergency stop 的官方状态机是什么？哪些切换允许在机器人承重时执行？

## P2：完善 WBT contract

10. 除 wrist-roll 与 ankle-roll link 外，是否存在官方 palm/wrist、sole/contact tracking frame 及标定 transform？
11. SDK 自带 ONNX 对应哪台 X2、哪套 default/PD、何种 observation/action schema？是否有仅用于验证官方仿真的兼容 T2.5 checkpoint？

## 用户未来必须再次授权

即使厂商回答完毕，真机命令仍需用户单独明确授权，并同时确认现场保护人员、清空区域、急停有效、吊架/保护方案、准确固件版本和控制所有权路线。本 Phase0 不自动解锁任何真机动作。
