# X2 Phase 9：Reset-projected + contact-phase-aware DSMS

日期：2026-08-11
裁决：**PREFLIGHT PASS / SINGLE NLP FAIL / RAW REPLAY FAIL BEFORE LIFTOFF**
资源：4 CPU threads；solve 1；weight/solver/window scan 0；RL/PPO 0。

## 结论

本轮严格使用 Phase6 projected reset，并首次把左脚在 `10/30 s` 后的 liftoff 写进 0.70 s DSMS 目标；右脚全程 stance。合同全部通过，但唯一 IPOPT 求解未收敛，返回候选的最大 shooting defect 为 `0.1667`。独立 raw replay 在 `0.245 s` 就因 root-z 跌破 `0.42 m` 倒地，早于 liftoff 起点 `0.333 s`，也显著差于 Phase6 frozen bridge 的 `0.683 s`。

因此本轮不能证明“phase-aware liftoff 无效”：有效的 liftoff 区间根本没有被 raw 候选活着到达。能否定的是当前单配置的长窗口 DSMS 求解表示——它没有生成足够连续、可执行的候选。

## 1. 冻结配置与 preflight

- 初态：`phase6_projected_initial_state.npz`，qpos 位级一致、qvel 全零。
- horizon：0.70 s，35 shooting nodes，1 kHz physics / 50 Hz control。
- active sole：每脚 12 个 collidable spheres；非碰撞 mesh 排除。
- 左脚：`0–10/30 s` clearance 0.25 mm；随后 0.10 s cubic liftoff；最终 30 mm。
- 右脚：全程固定 XY anchor、clearance 0.25 mm。
- ankle 权重沿用 Phase3c：XY/Z/V = `100/300/20`；其余 state cost 沿用 Phase3b。
- IPOPT/MUMPS、max_iter=300；CPU 4 threads。
- constraints returned/registered：2700/2700。
- head actuator 15/16 bounds：`[0,0] / [0,0]`。

Preflight：**PASS**。

审计同时纠正了一个旧风险：`G1GaitTO.problem.X_ref` 内部已经展开到 1 kHz，phase target 必须按 1 ms 标时。Phase3c 旧实现按 20 ms 使用该数组，存在目标时序被压缩的合同问题；Phase9 已按 1 kHz 修正后才运行。

## 2. 唯一 NLP

| 指标 | 结果 | 门 |
|---|---:|---:|
| variables / equalities | 3744 / 2700 | 诊断 |
| wall time | 1203.29 s | 诊断 |
| IPOPT iterations | 300 | 固定上限 |
| status | -1, max iterations | status 0/1 |
| returned keep-best iteration | 137 | 诊断 |
| objective（IPOPT final） | 9.7805 | 诊断 |
| max shooting defect | **0.16670** | ≤0.01 |
| defect p95 | 0.006468 | 不可替代 max 门 |

NLP certificate：**FAIL**。没有追加第二 solver、权重或窗口。

## 3. Independent official raw motor/PD replay

候选从同一个 Phase6 projected reset 执行。只把 solver 返回的 controls 送入未改写的 official torque-motor model。

| 指标 | Phase9 | Phase6 comparator | 判定 |
|---|---:|---:|---|
| fall time | **0.245 s** | 0.683 s | FAIL |
| fall cause | root-z < 0.42 m | tilt >0.90 | 更差 |
| root-z min at first fall | 0.41918 m | 0.54673 m | FAIL |
| root XY excursion before fall | 0.03264 m | 0.17309 m @0.683s | 诊断 |
| root accel p95/max | 15.39 / 30.97 m/s² | 4.41 / 4.44 | FAIL |
| semantic joint RMS/max | 0.2896 / 1.0024 rad | — | FAIL |
| torque saturation | 2.710% | 0.047% pre-fall | 更差 |

Active12 跌倒前：

| 指标 | Left | Right |
|---|---:|---:|
| contact fraction | 43.50% | 8.94% |
| min clearance | -2.071 mm | -0.482 mm |
| slip p95 | 0.448 m/s | 0.234 m/s |
| slip max | 0.650 m/s | 0.265 m/s |
| contact switches before fall | 1 | 1 |
| liftoff contact fraction | N/A | N/A |
| liftoff switches | 0 | 0 |

候选在 `0.245 s` 已倒，liftoff 从 `0.333 s` 才开始。因此 liftoff contact fraction 明确记为 `null/N/A`；跌倒前 switch 不能冒充 phase-aware liftoff 成功。原 raw JSON 的 NaN/错误 switch 语义已离线更正，没有重跑仿真。

## 4. 裁决与边界

1. Phase6 reset 没有解决长窗口 DSMS 的数值连续性；本轮 max defect 比 Phase3c 的 0.00115 明显恶化。
2. raw 候选的高 root acceleration、高 slip、高 semantic deviation 与 0.245 s root-z collapse 和 infeasible shooting defects一致。
3. 因候选未到 intervention 时刻，本轮对“显式 liftoff 目标本身”没有形成有效反证；只是当前单配置求解失败。
4. 按预注册和低资源要求停止，不启动追加 solve、replay、新窗口或 RL。

## 5. 产物

- `prereg_phase9_reset_contact_dsms.json`
- `phase9_preflight.json`
- `phase9_solve_result.json`
- `x2_lunge_phase9_prefix.npz`
- `phase9_raw_replay.json`（postmetric correction）
- `x2_lunge_phase9_reset_contact_dsms.py`
- `REPORT_phase9_reset_contact_dsms.md`
