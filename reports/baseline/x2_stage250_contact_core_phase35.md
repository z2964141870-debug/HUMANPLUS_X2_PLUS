# BASE Phase35：Phase34 足滑与 contact chatter 离线裁决

## 一句话裁决

Phase34 的 `0.39–0.49 m/s` 高分位滑移主要由碰撞窗首尾冲击和接触点切换夸大，承重核心段两脚实际都通过既有 `0.10 m/s` 门；但按既有 67 ms 确认和 100 ms 最短驻留去抖后，两脚在整个 move 段都被判为持续接触，真实 `DS→SS→DS=0`，与 generator 的 10 个周期不一致，因此它仍不能作为带可靠 contact phase 的可训练 50 Hz 动态 seed。

## 游戏任务

- [x] 只读 Phase34 完整 closed mmap；0 physics、0 training。
- [x] 每个原始连续接触窗首尾各裁去 50 ms。
- [x] 分别统计原始最大接触点、边沿、承重核心、力/冲量加权、足体平移和支撑位移。
- [x] 用既有 67 ms edge-confirm、100 ms minimum-dwell 重算接触周期。
- [x] 滑移门固定为 0.10 m/s，不按结果调阈值。
- [ ] 带可靠 contact phase 的可训练 50 Hz seed。
- [ ] contact-clean Silver。
- [ ] 实机足底力、COP 或 GRF 真值。

## 假设

Phase34 直接把每个 collision tick 的最大接触点速度放在同一个统计量中。若 p95 超门主要来自落脚冲击、滚动和 contact point 切换，那么裁掉连续接触窗首尾 50 ms 后，法向力/冲量加权速度和足体平移速度应回到 0.10 m/s 以下；若仍超门，才是承重阶段真实滑移。

## 干预

没有控制或数据干预，只增加离线分解：

1. `raw contact`：official closed MuJoCo 的 geom collision 与 `mj_contactForce`；
2. `edge transient`：每个连续 contact window 首尾各 50 ms；
3. `load-bearing core`：同一窗口裁边后的内部段，短于 100 ms 的窗口没有 core；
4. `canonical schedule`：67 ms on/off confirm，再强制 100 ms minimum dwell；
5. point slip 同时报告最大值、每 tick 法向力加权值、全 core 冲量加权值；另独立报告 sole body linear speed 与 stance excursion。

原始 mmap SHA256 仍为 `8bb90c96a17308ce6f3e5be1f19f72a3e3998800f29d6f020692eac318de24ef`，未改写。

## 对照与结果

### 1. 高滑移集中在 contact 边沿，不在承重核心

| 指标（p95） | 左脚 | 右脚 | 0.10 m/s 门 |
|---|---:|---:|---:|
| Phase34/raw 最大接触点速度 | 0.494 | 0.427 | fail/fail |
| 仅首尾 50 ms edge | 0.906 | 0.805 | fail/fail |
| load-bearing core 最大接触点速度 | 0.0666 | 0.0485 | pass/pass |
| core 每 tick 法向力加权点速度 | 0.0339 | 0.0276 | pass/pass |
| core 冲量加权点速度 | 0.0486 | 0.0321 | pass/pass |
| core sole-body 水平速度 | 0.0727 | 0.0805 | pass/pass |

左右脚 core 分别覆盖 1640/1530 个 1 kHz tick。core 最大接触点速度的最大值仍可到 0.110/0.134 m/s，但固定门使用预注册 p95，不能用个别尖峰推翻整体承重段裁决。

### 2. 支撑足没有沿地面持续漂移

| core stance excursion | 左脚 | 右脚 |
|---|---:|---:|
| 有效窗口数 | 14 | 14 |
| net p95 | 2.73 mm | 5.13 mm |
| max-from-start p95 | 3.11 mm | 5.14 mm |
| core duration p50 | 109 ms | 109 ms |

因此 Phase34 的旧滑移 p95 不能解释为“支撑足在承重期间一直滑走”。它主要测到了落脚/离地附近的快速局部接触变化。

### 3. 但 contact chatter 对步态语义是真问题

| contact schedule | 左脚 | 右脚 |
|---|---:|---:|
| raw contact windows | 17 | 15 |
| 67 ms confirm + 100 ms dwell 后 windows | 1 | 1 |
| debounced contact fraction | 1.000 | 1.000 |
| 与 generator agreement | 0.650 | 0.650 |

- generator：10 个 `DS→SS→DS` 周期；
- realized raw 50 Hz（Phase34）：26 个周期，包含抖动/刮擦；
- 固定 debounce 后：0 个周期，因为所有短暂离地缺口都不足以成为稳定 swing/contact-off 状态。

也就是说，旧指标同时混合了两件事：

- **假阳性部分**：高 p95 足滑主要来自边沿瞬态，不能据此说承重脚持续打滑；
- **真实失败部分**：摆动脚没有形成满足既有 dwell 的稳定离地窗，contact phase 仍不可信。

## 结论

### 已修正的认识

Phase34 的稳定直行是一条有效的 **X2-native state/action physical warm-start**。其支撑核心段低滑移、支撑位移仅毫米级，不应再描述成“整段步态严重足滑”。

### 仍然不能晋级的原因

它不是可靠的 **50 Hz contact-conditioned dynamic seed**：去抖后没有任何稳定单支撑周期，无法提供可信的 liftoff/touchdown 监督；Phase34 已知的摆脚净空不足 1 cm 也与这一裁决一致。

最终门：

- closed full gate：pass；
- 两脚 load-bearing core slip：pass；
- debounced cycle 与 generator 一致：fail（0 vs 10）；
- trainable 50 Hz contact seed：**fail**；
- hardware contact truth：false。

## 下一步

唯一有价值的后续不是重新采相同 rollout 或直接长训，而是把 Phase34 当原生 warm-start 做离线 contact-aware repair/bridge：在保持承重核心 p95 `<=0.10 m/s` 的同时，显式创造至少满足既有 100 ms dwell 的 swing-off 窗，再用同一 official closed contact 门复核。未经该修复，不应把 Phase34 的 raw collision labels 作为 contact phase 训练标签。

## 证据边界

所有 force、impulse、collision 与 contact 都来自 official closed MuJoCo 模型；它们是模型物理证据，不是 X2 实机足底力、压力、COP、GRF 或硬件接触真值。

完整数值：[Phase35 JSON](../official_x2/phase35_contact_core_audit.json)
