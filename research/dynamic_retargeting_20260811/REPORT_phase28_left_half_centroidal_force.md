# Phase28 — X2 左半周期 centroidal/contact-force 可行性

## 裁决

`CENTROIDAL FORCE PREFLIGHT REJECTED / NO RAW PHYSICS`

冻结 Phase27 的 100 帧路径，使用官方 `scene.xml`、MuJoCo quaternion-aware 速度、中心差分 COM 加速度与 `mj_subtreeVel` centroidal angular momentum。每帧所有落在 `[-0.01, 0.5] mm` 真实接触带内的 active12 sphere 都可由 LP 自主选取；没有预设左右 force switch。力满足 6D 总力/总矩、非负法向力与官方摩擦系数的保守 L1 摩擦锥。

## 结果

- 有效中心导数帧：`98`。
- 几何合格：`92/98`。
- force LP 可行：`22/98`。
- 几何+力共同可行：`22/98`。
- COM acceleration norm p95/max：`3.027 / 3.191 m/s²`。
- centroidal Hdot norm p95/max：`23.879 / 25.498 N·m`。

初始 hold 的 6 帧沿用 Phase41 soft-contact 边界，sole min signed distance 约 `-0.306 mm`；它通过 Phase41 的 `-0.5 mm` 原生边界，却不在本 force oracle 的 `-0.01 mm` 接触带内，因此诚实判 geometry fail。

force-feasible 帧主要集中在：

- 左加载结束附近与其 hold：frames `31–37`；
- 右脚 touchdown 末段与最终 hold：frames `88–98`；
- 少量早期双支撑转换帧：`7,18–20`。

完整右脚抬起/单支撑区间没有获得连续 force certificate。LP 的 70 个失败帧返回 HiGHS infeasible，而不是后端崩溃或超时。

## 结论

Phase27 证明连续几何与低 root acceleration 可以同时满足，但这不等价于 centroidal wrench 可实现。当前姿态/时间参数化产生的角动量变化，以及实际落入接触带的稀疏 sole 支点，不能在大多数帧共同满足 6D 平衡。

因此不得进入 raw MuJoCo physics、teacher 保存或 RL。下一方法必须把 centroidal angular momentum/contact force 放进轨迹生成本身，而不是在几何路径完成后事后检查。该结果只否定当前准静态 COM 几何生成器，不否定 DSMS、OmniTrack、X2 或动态重定向总体方向。

## 产物

- `phase28_left_half_centroidal_force_contract.json`
- `run_phase28_left_half_centroidal_force.py`
- `phase28_result.json`
- `tests/test_phase28_left_half_centroidal_force.py`
