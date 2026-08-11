# X2 Phase11：Name-mapped load/swing event SBTO

日期：2026-08-11

裁决：**PREFLIGHT PASS / UNIQUE SEARCH REJECTED / FIXED JOINT-MODE EVENT ROUTE STOPPED**

资源：单进程、单线程、`nice=10`、0 GPU；216 candidate rollouts；41.74 s；搜索1次、重跑0、RL/PPO 0。

## 1. 目的

Phase8 能把首次跌倒从 `0.683 s` 延长到 `0.775 s`，但只产生26次接触抖动，最长左离地/右支撑为21 ms。Phase11 测试一个更平滑、更低维的假设：

> 用严格 name-based body29/head2 映射，将控制限制为“预载荷转移 + 左腿 swing”两段 C2 事件和4个物理可解释关节模式，能否制造至少100 ms稳定左离地/右支撑，同时保护右支撑、root和head。

## 2. 冻结合同

- body29 按官方 actuator joint name 构造；head yaw/pitch 为索引15/16，每个tick固定target=0。
- 事件模式：lateral load、right stance flex、left swing flex、sagittal balance。
- 8个参数；最大联合关节修正0.15 rad。
- preload：0.02–0.40 s C2 pulse；swing：0.28–0.40 s C2 ramp，保持到0.70 s，随后到1.0 s衰减。
- 渐进窗口0.34/0.52/0.70 s；每窗24 candidates×3 iter，elite6，seed20260812。
- official raw motor、官方逐关节PD、Euler、1 kHz/50 Hz、active12不变。
- penetration不进入搜索排序，只用native成功域最小值 `-4.487 mm` 保护。

## 3. Preflight

全部通过：

- official body29 indices明确排除head 15/16；
- modes对head最大系数0；
- t0 event correction严格0；
- bounds下最大修正0.15 rad；
- zero trace两次SHA逐bit一致；
- corrected Phase6 baseline精确复现0.683 s；
- 每脚12 active sole spheres；
- runner SHA `35270ca8aa43dbbc00a815679abdb365036a3f83be3eeecd3d606b745de5c83f` 与prereg/preflight一致；
- result已存在时runner拒绝第二次搜索。

## 4. 唯一搜索结果

| 指标 | Phase6 zero | Phase8 projected | Phase11 |
|---|---:|---:|---:|
| 首次跌倒 | 0.683 s | **0.775 s** | 0.702 s |
| 0.70 s窗口存活 | fail | pass | pass |
| left contact switches | 0 | 26 | **0** |
| 稳定left-off/right-on | 0 ms | 21 ms | **0 ms** |
| swing left contact | 100% | 约75% | **100%** |
| left clearance p50 | — | 未达门 | **-0.495 mm** |
| right support | 100% | 100% | 100% |
| right speed p95 | 0.0329 | 0.0583 | **0.0369 m/s** |
| right excursion | 0.0189 | 0.0389 | **0.0210 m** |
| root accel p95 | 4.4105 | 4.4773 | **4.2096 m/s²** |
| head max | 0.00078 | 0.00059 | **0.00080 rad** |

Phase11 通过 prep双接触、右支撑、右足速度/位移、flight、head和native penetration；失败于稳定liftoff、左接触占比、左净空和root acceleration。

最终事件幅度都较小（最大绝对参数约0.0399 rad）。优化器选择保持双脚稳定接触，而不是付出足够的左腿抬升代价。0.70 s窗后仅2 ms即由tilt触发跌倒，完整动作仍失败。

## 5. 方法边界

0.34 s渐进窗口早于0.40 s swing评分区间，因此其缺失swing指标产生统一的大常数loss；它对该窗所有候选近似共同，只提供tracking/regularization排序。这使第一阶段信息增益很低，但没有改变最终0.52/0.70 s使用完整事件指标的独立选择。该缺点如实记录，不以重跑修饰。

本轮否定的是固定4-mode、8参数、216-rollout的事件关节表示，不否定sampling或显式liftoff。证据表明低维关节模式保护了支撑稳定，却没有足端空间权威性；下一次若回到动态线路，应升级为直接足端/载荷约束表示，而不是继续增加同一模式的population、权重或幅度。

## 6. 停止与下一步

- 按预注册停止，不重跑、不扫模式/时序/阈值，不进入RL。
- 动态线路当前保留Phase6 reset作为公共前置，Phase8作为最好但未合格的sampling证据。
- 为遵守共享设备要求，下一实质任务切换到BASE线路做只读/低资源准备审计，不立即启动另一项物理搜索。

## 7. 产物

- `prereg_phase11_name_mapped_event_sbto.json`
- `phase11_preflight.json`
- `x2_lunge_phase11_name_mapped_event_sbto.py`
- `phase11_event_candidate.npz`
- `phase11_result.json`
- 本报告
