# BASE Phase36：exact-state 单窗摆脚 teacher 裁决

## 一句话裁决

冻结的 240 ms 五模式 CEM 把 Phase34 第一个右脚“想摆但仍触地”窗口从完全不离地改善到连续离地 92 ms、净空 11.28 mm，并保持支撑滑移、root 安全和重新接触，但仍没有通过预注册的 100 ms / 12 mm 双硬门；因此这是很接近的机制改善，不是合格 teacher，当前低维表示被否定，训练继续锁定。

## 游戏任务

- [x] 固定 Phase34 第一个 generator-swing / realized-contact mismatch。
- [x] 从前一 physics tick 恢复 exact `mjSTATE_INTEGRATION`。
- [x] 同一 recorded future ctrl 前 10 ms qpos/qvel 逐位复现。
- [x] 固定 12 control ticks、3 knots、5 modes、seed/budget/bounds。
- [x] 唯一 CEM：120 evaluations，无窗口、权重、阈值或预算扫描。
- [x] 检查离地、净空、支撑 core slip、root/tilt、重新接触和后续短窗。
- [ ] 100 ms 稳定离地。
- [ ] 12 mm 净空。
- [ ] 合格 dynamic teacher。
- [ ] 解锁训练。

## 假设

Phase35 证明 Phase34 的承重核心段不滑，真正缺口是 swing-off 不持续。若缺失只来自少量局部腿腰协调，则在 exact closed 初态、相同后续 ctrl 下，对摆动腿 hip-pitch/knee/ankle-pitch、支撑 hip-roll 和 waist-roll 施加有界三节点 residual，应能产生至少 100 ms 真实离地、12 mm 净空，并在末端重新接触，而不破坏支撑与 root。

## 干预

固定窗口：

- telemetry index：114；move elapsed：0.08 s；
- swing：right；stance：left；
- aligned physics index：2310；snapshot：index 2309 / t=2.310 s；
- snapshot SHA：`226108c4...2f96`。

固定搜索：

- teacher 240 ms（12×20 ms），随后 80 ms recorded-control follow-up；
- 3 knots × 5 modes，共 15 维；
- bounds：hip-pitch ±0.08、knee ±0.12、ankle-pitch ±0.06、stance hip-roll ±0.05、waist-roll ±0.04 rad；
- CEM：seed 3601、population 24、iterations 5、elites 6；
- vendor MuJoCo 3.3.7、原 scene、exact snapshot、未来每 1 ms recorded ctrl 全部冻结。

## 对照

零 residual 完整复现 recorded ctrl 后缀：

- exact prefix 10 ticks：qpos absmax=0、qvel absmax=0；
- 右脚连续离地=0 ms、净空=0；
- 左支撑 core slip p95=0.0326 m/s；
- root-z min=0.663 m、tilt max=0.152 rad；
- terminal recontact / follow-up safety 均通过。

因此搜索不是从 Phase28 的近似 93D reset 起步，也没有把 direct/closed 初态差异误当成 teacher 收益。

## 结果

| 指标 | zero residual | best CEM | 固定硬门 | 裁决 |
|---|---:|---:|---:|---:|
| 连续 swing-off | 0 ms | 92 ms | ≥100 ms | fail |
| 最大 swing clearance | 0 | 11.28 mm | ≥12 mm | fail |
| stance-core slip p95 | 0.0326 | 0.0752 m/s | ≤0.10 | pass |
| root-z min | 0.663 | 0.580 m | ≥0.55 | pass |
| root tilt max | 0.152 | 0.271 rad | ≤0.30 | pass |
| terminal contact 40 ms | pass | pass | required | pass |
| 后续 80 ms 安全 | pass | pass | required | pass |
| actuator residual saturation | 0 | 0 | diagnostic | clean |

cost 从 13.0 降至 0.961，目标区域触地比例降到 0.30。最佳候选的离地发生于 teacher tick 92–184；它不是靠摔倒、取消交权、力矩饱和或破坏支撑换来的假改善。

但它分别比硬门少 8 ms、0.719 mm。由于门和预算在 CEM 前已经冻结，不能把“接近”改写成“通过”，也不能追加一轮搜索。

## 结论

### 成功之处

Phase35 提出的机制判断得到支持：在 exact target-native physics 中，少量腿腰协同确实能把“持续刮地”转化为接近完整的摆脚事件，同时仍能落地并保持短窗安全。这比继续调 reward 或用近似 reset 更直接。

### 失败之处

预注册表示没有找到严格可行解。因此当前 `3-knot × 5-mode × 240 ms` 表示按规则判为 **unsupported**，不能产出训练 label，不能解锁 PPO/长训。

## 下一步

本阶段停止，不增加预算、不改阈值、不换窗口。若继续，必须作为新的独立假设预注册更有物理意义的表示，例如显式 liftoff/touchdown timing 或更早的载荷转移阶段；不能把本轮 92 ms near-miss 当作成功样本直接训练。

## 证据边界

这是从 closed trace exact state 分叉的 direct vendor-MJCF oracle；它不是 closed ROS 完整回放，也不是 X2 实机足底力、GRF、COP 或硬件接触真值。

### Phase37 preflight 后的有效性更正

Phase37 在 CEM 前补充了完整 zero-candidate no-op 门，发现 Phase36 candidate 函数会对五个被干预关节的 recorded `ctrl` 再次执行 MJCF ctrlrange clip；closed trace 中部分 recorded `ctrl` 本来就在声明范围外，因此即使 residual=0，Phase36 的 candidate 后缀也不再逐项等价 closed recorded suffix。Phase36 的 92 ms / 11.28 mm 结果现降级为 **clipped-control diagnostic near-miss**，不能作为严格 matched-control teacher 证据。其“硬门未过、不得训练”的保守裁决仍成立，没有误解锁训练。

证据：[预注册](../official_x2/phase36_exact_swing_teacher_prereg.json) · [结果 JSON](../official_x2/phase36_exact_swing_teacher_result.json)
