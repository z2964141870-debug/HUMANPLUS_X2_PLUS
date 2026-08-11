# X2 Phase4 SBTO/CEM 短前缀动态重定向报告

日期：2026-08-11
状态：**PREFLIGHT PASS / SEARCH OBJECTIVE AUDITOR CONTAMINATED / CORRECTED RAW GATES REJECTED**
训练：0；RL/PPO：0；参数扫描：0；追加重试：0。

## 结论

一次预注册的渐进窗口 SBTO/CEM 已在 X2 官方 raw motor 模型上完成，但事后共同审计发现 penetration auditor 把每脚一个 `contype=0` visual mesh 当作 sole sphere。该错误不仅污染最终 penetration 报告，也污染了 CEM 搜索使用的 scalar loss。因此，原先“loss `689.62→478.89`、穿透改善”的判断撤回。

只使用既有 nominal/candidate artifact、0 次新搜索进行确定性修正重放后，active sole penetration 为 nominal `-1.81/-3.87 mm`、candidate `-1.81/-4.87 mm`；正确 auditor 下 loss 实为 `5.176→6.218`，candidate 比 nominal 更差。最终仍只通过 3/7 个共同硬门，不能作为动态可行参考。

因为搜索排序目标本身受污染，本结果不能用于否定 SBTO 方法、固定预算或控制点表示；它只能拒绝这个已经生成的 candidate，并证明本次搜索实验不具备干净的方法裁决力。未追加候选、未改权重、未跑 RL。

## 事后 penetration auditor 修正

- `physics.foot_geom_contract` 每脚返回 13 个 geom。
- 左 `14`、右 `37` 为 `contype=0` visual mesh；旧 auditor 错把 `geom_size[0]` 当球半径。
- 修正后每脚严格只使用 12 个 `contype!=0` active sole sphere：左 `15..26`，右 `38..49`。
- 只做 2 次确定性重放（nominal/candidate）；CEM 搜索重跑 0 次，physics 搜索 0 次。
- 除 penetration 及其派生 scalar loss 外，所有物理指标与旧 result 完全相同。
- 原 result SHA256：`2febb5c0d669205d3048b2d1204805093ed926614bc9d07d6df9fb1bdf465f42`；artifact SHA256：`4c509112af5d1c1b6e4f8f5c51ef20948308429aa304dcbd3a6de4dda6e462d2`。

## 冻结合同

- 输入：`PHUMA-LUNGE-R-001`，50 Hz 的 18 帧/0.34 s prefix。
- 物理：MuJoCo 3.3.7，官方 scene、Euler、1 kHz；官方 `motion_control.yaml` 逐关节 PD；raw torque motor。
- 表示：5 个线性 correction 控制点，优化 body 29 个 target，head target 固定 0。
- 搜索：seed `20260811`；0.12/0.22/0.34 s 渐进窗口；每窗 40 candidates × 4 iterations，elite=8；单进程、对角 CEM。
- 最终只认 raw 官方重放共同硬门，优化 loss 不作为成功证书。

## Preflight

| 检查 | 结果 |
|---|---:|
| official scene SHA256 | `7fceb3e1357be29b...28bb1b63` |
| official control SHA256 | `07fa8a0151d14c2b...c1c2e400` |
| Phase30 motion SHA256 | `3f218aa8fc2f9441...f7d013ea` |
| MuJoCo / integrator | 3.3.7 / official Euler |
| name→actuator roundtrip residual | 0 |
| zero correction residual | 0 |
| repeated deterministic score delta | 0 |
| full-reference free-root baseline fall | **0.575 s** |

第一次 preflight 曾因错误假定 reference 顺序等于 actuator 顺序得到 0.474 s；preflight 正确拦截，随后在搜索前改为显式 joint-name 映射。修正后精确复现 0.575 s，才进入唯一搜索。这次预物理修正没有改变已预注册的搜索参数、门或 seed。

## 渐进搜索

| 窗口 | 第一轮 best loss（受污染，仅 provenance） | 第四轮 best loss（受污染，仅 provenance） |
|---|---:|---:|
| 7 frames / 0.12 s | 293.03 | 203.15 |
| 12 frames / 0.22 s | 361.52 | 293.98 |
| 18 frames / 0.34 s | 531.70 | 449.95 |

总 wall time `39.88 s`。表内历史 loss 使用了受污染 auditor，不能解释为真实改善；由于没有保存全部 480 个候选，事后不能诚实重排候选。最终 mean artifact 在正确 auditor 下的 loss 是 `6.218`，nominal 是 `5.176`。

## Final raw official replay

| 指标 | Nominal | SBTO 输出 | 硬门 | 结果 |
|---|---:|---:|---:|---|
| 完整 0.34 s 存活 | 是 | 是 | 是 | PASS |
| penetration L/R (mm)，active sole | -1.81 / -3.87 | -1.81 / -4.87 | 均 >= -0.5 | FAIL |
| stance speed p95 L/R (m/s) | 0.0315 / 0.0349 | 0.0860 / **0.1199** | 均 <= 0.10 | FAIL |
| stance excursion L/R (m) | 0.0072 / 0.0145 | **0.0327** / 0.0259 | 均 <= 0.03 | FAIL |
| unintended flight | 0% | 0% | <=2% | PASS |
| root horiz accel p95 (m/s²) | 3.19 | **6.59** | <=4 | FAIL |
| head max abs (rad) | 0.00211 | 0.00333 | <=0.02 | PASS |

诊断：

- 右脚 intended-contact ratio 从 `87.35%` 提高到 `92.06%`；左脚保持 100%。
- candidate 没有改善正确 penetration：左脚持平，右脚恶化约 1.0 mm。
- candidate 同时恶化了足速、左脚位移和 root 加速度。
- torque saturation 为 `0.987%`，root-z min `0.4919 m`，tilt max `0.4813 rad`；短前缀没有倒地。

## 方法判断

本次 CEM 在错误的 visual-mesh penetration penalty 主导下选择候选；其搜索历史无法回答干净 SBTO 是否有效。candidate 的右脚接触率有所增加，但正确 loss、penetration、足速和加速度整体不支持“弱正信号”的旧结论。

若未来重新开 SBTO，必须作为新预注册实验：搜索前冻结 active collision geom 审计器，并重新运行 zero-correction/nominal gate；不能沿用本次污染的 candidate ranking，也不能把失败后的权重修改伪装成同一次实验。

## 产物

- `prereg_phase4_sbto.json`
- `phase4_sbto_preflight.json`
- `phase4_sbto_result.json`
- `phase4_sbto_penetration_audit.json`
- `x2_lunge_phase4_sbto_prefix.npz`
- `x2_lunge_sbto_phase4.py`

结果裁决：`SEARCH_OBJECTIVE_AUDITOR_CONTAMINATED / CORRECTED_RAW_GATES_REJECTED`。路线停止；没有重跑 CEM。
