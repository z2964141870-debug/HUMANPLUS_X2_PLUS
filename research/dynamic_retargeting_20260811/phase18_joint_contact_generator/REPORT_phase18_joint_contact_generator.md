# X2 Phase18：Stage-A 联合 contact / foot-placement / root / lower-q 生成器

日期：2026-08-11  
状态：**5/5 TEMPLATE REJECTED / NO PHYSICS / NO TRAINING**

## 裁决

Phase17 冻结的 5 个接触模板已按固定顺序串行完成，全部未通过完整离线几何门，因此不生成 teacher、不进入 official raw-PD physics，也不解锁训练。

这次不是旧式“在坏 reference 附近加一个短关节 residual”：变量已经包括逐帧 root XYZ/局部旋转、lower12+waist3、每个 stance segment 的足端 XY anchor，并显式把 COM 拉向模板支撑中心。它依然失败，说明当前固定加权稀疏 GN 表示还不能同时兑现支撑足接触、低足速、COM 支撑和上身语义。

## 固定合同

- 输入：`PHUMA-LUNGE-R-001` Phase30 1.46x；Phase15 只作 morphology-normalized 初值。
- 模板：`DS`、`DS-L-DS`、`DS-R-DS`、`DS-L-DS-R-DS`、`DS-R-DS-L-DS`。
- 每模板 8 轮 sparse GN/LSQR，参数和阈值完全相同，不按结果调权重或时序。
- official active12、joint limits 为硬合同；旧 source-height contact label 不作真值。
- 资源：单进程、1 CPU thread、nice 10、0 GPU；总求解约 5.3 秒。
- 执行：`mj_step=0`、policy forward=0、optimizer update=0。

## 结果摘要

| 模板 | 最接近的正信号 | 主要失败 |
|---|---|---|
| DS | qstep、root acceleration、limits 通过 | 无单支撑/摆脚；stance contact p95 69.6mm，足速0.698m/s |
| DS-L-DS | 摆脚 clearance 54.3mm、root accel通过 | COM margin -92.7mm，stance speed 0.468m/s，contact 23.4mm |
| DS-R-DS | dwell/root accel通过 | COM margin -92.7mm，右摆脚仍穿地，stance speed 0.727m/s |
| DS-L-DS-R-DS | **COM margin -3.02mm、clearance 9.79mm，最接近** | qstep 0.10010、stance speed 0.586、contact 42.0mm、root accel 22.35 |
| DS-R-DS-L-DS | qstep/limits/dwell通过 | COM margin -37.9mm、摆脚穿地、root accel 18.80 |

所有模板的 stance excursion 也均超过 30mm；双步模板的 upper keypoint p95 约 0.126–0.128m，仍高于 0.10m。最接近模板仍同时失败多个独立硬门，不能把 `-3mm` 或 `9.8mm` 的近失当成 teacher。

## 解释

本阶段给出两个有用结论：

1. 联合改变 root/足位/contact/lower-q 确实能把原先 20–40cm 量级的单支撑 COM 缺口缩到毫米级，说明换表示方向是对的。
2. 单一加权 least-squares 同时处理接触等式、足速、COM、平滑和语义时出现明显折衷；接触误差仍是厘米级，说明这些条件不能继续只当 soft residual。

因此下一步若继续，应改成分层/硬约束生成：先求 contact/COM/foot-placement 的可行证书，再在其零空间恢复上身和时间平滑。不能通过增加 GN 轮数、提高权重或扫描模板时序来美化当前结果。

## 产物

- `run_phase18_joint_contact_generator.py`
- `prereg_phase18_joint_contact_generator.json`
- `phase18_result.json`
- `tests/test_phase18_joint_contact_generator.py`

当前结论：`STAGE_A_WEIGHTED_GN_REJECTED / HIERARCHICAL_HARD_CONSTRAINT_GENERATOR_REQUIRED`。
