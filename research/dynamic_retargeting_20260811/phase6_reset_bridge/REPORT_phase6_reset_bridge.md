# X2 Phase 6：Reset-compatible initial contact projection + fixed bridge

日期：2026-08-11
状态：**INITIAL PROJECTION PASS / DYNAMIC REPLAY 7-of-8 GATES PASS / OVERALL FAIL**
训练：0；参数扫描：0；RL/PPO：0；Phase3–5 文件修改：0。

## 结论

初态不可 reset 确实是此前方法的重要混杂因素。一次冻结配置的 SLSQP 初态投影，只改 root XYZ、双腿 12 DoF 和腰 3 DoF，就能在很小语义误差下让双脚获得真实接触；随后 0.34 s 固定 bridge 在 official raw motor/PD 中通过存活、接触时序、足速、足位移、flight、root 加速度和 head 共 7 类门。

唯一失败是动力学接触中的瞬时软穿透约 `-1.06 mm`，超过 `-0.5 mm` 门。因此总体仍诚实记为 **FAIL**，但它与 Phase3b 的多项大幅失败不同：这条路线已经把问题收缩为一个小量级的动态接触/法向力控制问题。

## 1. Sole geom 合同纠正

`foot_geom_contract()` 每脚返回 13 个 geom，但其中每脚第一个是非碰撞 mesh：

- left geom 14：`contype=0`
- right geom 37：`contype=0`
- 其余每脚 12 个是 `contype=1` 的 sphere，才会产生 floor contact。

把非碰撞 mesh 的 `geom_size[0]` 误当球半径，会得到此前的 `L=-24.706 mm / R=-8.234 mm`。这不是可碰撞 sole sphere 的 clearance。按真实 12 sphere 合同，frame 0 为：

| 指标 | Left | Right |
|---|---:|---:|
| 初始 sphere 最小 clearance | -1.813 mm | +11.779 mm |
| 初始有效 floor contact | 有 | 无 |

完整 12 sphere clearance 已保存于结果 JSON，未用单个最小值掩盖脚掌倾斜。

## 2. 初态投影

冻结变量：root XYZ + 双腿 12 DoF + 腰 3 DoF；root quaternion、双臂、head 保持不变。所有 qvel 明确置零，代表可部署的静止 reset。

约束：所有 12 collidable sole spheres `clearance >= -0.5 mm`，且每脚至少一个 sphere 进入真实 MuJoCo contact。SLSQP 只运行一次，30 iterations，成功退出。

| 语义/状态偏移 | 结果 |
|---|---:|
| root Δx / Δy / Δz | +0.053 / +0.011 / -3.031 mm |
| 15 DoF 变化 RMS | 0.01253 rad |
| 15 DoF 最大变化 | 0.03568 rad（right hip roll） |
| 上肢四关键点误差 RMS | 2.926 mm |
| 上肢四关键点误差 max | 3.078 mm |
| 投影后 min clearance L/R | -0.500 / -0.050 mm |
| 投影后真实 contact geom L/R | 1 / 1 |

初态投影门：**PASS**。

## 3. 固定 0.34 s bridge

未运行优化器或扫描。唯一固定控制为：

```text
q_target(t) = q_ref(t) + (1 - smoothstep(t / 0.34)) * (q_projected - q_ref(0))
```

以 official `motion_control.yaml` 逐关节 PD，在未改写的 raw torque-motor `scene.xml` 中 1 kHz 执行；head target 固定为 0。从新初态 `t=0` 起严格统计接触。

| 共同 raw 门 | 结果 | 阈值 | 判定 |
|---|---:|---:|---|
| 0.34 s 存活 | full | full | PASS |
| 动态最小穿透 L/R | -1.063 / -1.038 mm | ≥ -0.5 mm | **FAIL** |
| stance speed p95 L/R | 0.0402 / 0.0338 m/s | ≤ 0.10 | PASS |
| stance excursion max L/R | 0.00805 / 0.01592 m | ≤ 0.03 | PASS |
| contact contradiction L/R | 0 / 0 | 诊断 | PASS |
| unintended flight | 0 | ≤ 2% | PASS |
| root horiz accel p95 | 3.549 m/s² | ≤ 4.0 | PASS |
| head max abs | 0.00276 rad | ≤ 0.02 | PASS |

附加诊断：root-z min `0.5195 m`，tilt max `0.5323 rad`，torque saturation `0.0949%`。

## 4. 裁决与下一步边界

这次结果支持两个结论：

1. **reset-compatible 初态投影应成为所有后续动态重定向方法的共同前置层。** 固定原始 penetrating/non-contact `x0` 会让某些门先验不可达，也会污染方法间比较。
2. **单纯几何投影不保证整个 prefix 的严格 clearance。** official MuJoCo 软接触在承载时产生约 1 mm 压入；下一阶段需要明确处理动态法向载荷/clearance，而不是再改初态。

按预注册规则，本路线在唯一 bridge 后停止。没有通过事后加大投影高度、修改 penetration 门、改变 smoothstep 时长或添加第二个 optimizer 来制造 PASS。

## 5. 产物

- `prereg_phase6_reset_bridge.json`
- `x2_lunge_phase6_reset_bridge.py`
- `phase6_projected_initial_state.npz`
- `phase6_reset_bridge_result.json`
- `REPORT_phase6_reset_bridge.md`
