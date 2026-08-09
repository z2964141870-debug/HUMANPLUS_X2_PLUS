# 🎮 X2 双路线阶段存档卡：BASE Phase10–15 / WBT Phase10–25

> 存档日期：2026-08-09
> 任务性质：只汇总已落盘证据；本卡片没有新增实验、训练、official 仿真或真机运行。
> 机器可读版：[X2_DUAL_TRACK_PHASE10_25_CHECKPOINT_20260809.json](X2_DUAL_TRACK_PHASE10_25_CHECKPOINT_20260809.json)

## 一句话总判定

两条路线都取得了真实的**机制与合同进展**，但都尚未通过最终物理/策略晋级门：BASE recovery 仍是 stop/full `0/5`，WBT prescribed replay 仍失败且 faithful zero-update 尚未执行，所以两条训练线当前都锁定。

## 🗺️ 当前地图

| 路线 | 本阶段任务 | 已完成的证据 | 尚未完成 | 当前锁 |
|---|---|---|---|---|
| BASE | 让独立 recovery 在不破坏起步/行走的条件下接管停车恢复 | Phase10–15 共 6/6 阶段已记录；角色、handoff、history、OOD 与 suffix schema 已审计 | 任一 recovery 模型通过 stop/full；真实 suffix 数据；有效的最小 recovery smoke | 🔒 25-update、长训、suffix训练全锁 |
| WBT | 先建立忠实 WBT29 Any2Any zero-update，再允许优化 | Phase10–25 共 16/16 阶段已记录；Gold、MotionLib、WBT29、LoRA合同与训练物理 provenance 已落盘 | live 29D faithful wiring、Phase25 hash guard、zero-update | 🔒 optimizer/1-update/PPO/LoRA训练全锁 |

这里的“6/6、16/16”只表示阶段证据已完成，不代表模型完成度或物理成功率。

---

# 🛡️ 路线 A：BASE 停止/恢复后端

## 主任务

```text
Stage306 moving
      ↓ matched deceleration
独立 recovery 接权
      ↓
稳定停下并恢复站立
```

目标不是把连续指标刷好看，而是保持 stand/start/move 不退化，并让 stop/full 真正过门。

## Phase10–15 游戏任务

### ✅ Phase10：发现“recovery 根本没上场”

- 假设：只替换 recovery ONNX 就能做纯 recovery A/B。
- 干预：只读审计 Phase9 路由和 policy-slot counts。
- 对照：Phase9 source 五条。
- 结果：`recovery=0 inference/episode`；`curriculum_then_policy` handoff 后硬编码 stationary，候选是严格 no-op，因此没有浪费五条 official 运行。
- 结论：这是控制合同漏洞，不是 f005 无效证据。
- 下一步：先修角色语义。
- 证据：[Phase10 报告](baseline/x2_recovery_phase10_role_contract.md) / [JSON](baseline/x2_recovery_phase10_role_contract.json)，JSON SHA `f2e4b3fe…ccf7`。

### ✅/❌ Phase11：角色机制通过，物理失败

- 假设：source stationary 保护起步，f005 只在 transition 后恢复，可能避免 Phase9 的角色混淆。
- 干预：handoff 后明确路由到 recovery slot，并把 previous/issued action 写入同一接权 slot。
- 对照：source recovery vs f005 recovery，各 5 条完整 official episode。
- 结果：10/10 role counts 精确 `main=360, stationary=100, recovery=300`；两组 full/stop 都 `0/5`。f005 stop drift `0.303→0.276m`，但 settle `4.396→5.200s`，仍坍塌。
- 结论：**角色机制改善成立；物理恢复未成立。**
- 下一步：离线找交权后第一个分叉。
- 证据：[Phase11 报告](baseline/x2_recovery_phase11_role_semantics.md) / [JSON](baseline/x2_recovery_phase11_role_semantics.json)，JSON SHA `8817c641…0689`。

### 🔍 Phase12：锁定交权跳变与延迟坍塌

- 假设：stop 2.0s 的 moving→recovery 切换产生大动作跳变，物理系统随后才倒。
- 干预：只对 Phase11 十条 trace 做 first-violation 对齐审计。
- 对照：source 5 vs f005 5。
- 结果：physical target L2 跳变中位 `0.434/0.471rad`；tilt 在 `+1.32/+1.60s` 越界，root-z 在 `+1.92/+2.14s` 越界。
- 结论：时间先后证据很强，但相关不等于因果。
- 下一步：只做一次有界 handoff continuity A/B。
- 证据：[Phase12 报告](baseline/x2_recovery_phase12_first_violation.md) / [JSON](baseline/x2_recovery_phase12_first_violation.json)，JSON SHA `a4d147f7…11ce`。

### ✅/❌ Phase13：handoff 完全连续，机器人仍倒

- 假设：若跳变是主因，消除跳变应显著推迟或避免坍塌。
- 干预：0.5s C2 physical-target blend；从最终 physical target 反算 actual action，并同步 next previous-action。默认仍为 0。
- 对照：blend `0.0 vs 0.5`，各 5 条 official。
- 结果：handoff physical target L2 `0.420→0.000rad`，前0.5s最大单步 `0.420→0.026rad`，history误差 `0`；但 full/stop 仍均 `0/5`，健康候选只把坍塌推迟约 `0.03s`。
- 结论：**连续性机制修好了，但它不是充分根因；物理门仍失败。**
- 下一步：审计 recovery 输入支持域。
- 证据：[Phase13 报告](baseline/x2_recovery_phase13_handoff_continuity.md) / [官方 A/B JSON](official_x2/phase13_handoff_continuity_ab.json)，JSON SHA `a4b286e8…7f49`。

### 🔍 Phase14：不是 q/dq 全局 OOD，最早异常在 action-history

- 假设：recovery 接权后进入 Stage335 未覆盖的输入域。
- 干预：4 条健康 suffix 对 Stage335 90 states 与 source-stand 250 rows 做分组 robust NN/quantile 审计。
- 对照：完整 Stage335、eventual-pass/fail 子集、source stand稳态。
- 结果：global composite、q、dq、current action OOD 都为 `0`；previous-action 组 OOD `24.67%`，r3/r4/r5 在 `+0.24–0.26s` 先越界，gravity/root 约 `+1.2–1.4s` 才异常。相对 source stand，handoff 从 t=0 即整体 OOD。
- 结论：不支持“物理 q/dq 全局没覆盖”；更窄假设是缺少 rollout-consistent action-history continuation，仍不构成因果证明。
- 下一步：设计 closed-loop suffix aggregation，不再堆离散 reset pose。
- 证据：[Phase14 报告](baseline/x2_recovery_phase14_ood.md) / [JSON](baseline/x2_recovery_phase14_ood.json)，JSON SHA `1bc818dc…7f4`。

### ✅/⏸ Phase15：suffix 合同就绪，尚未采集

- 假设：on-policy closed-loop suffix 比孤立 reset state 更直接覆盖 previous-action 漂移。
- 干预：默认关闭的 recovery-only `0–1.5s` snapshot sidecar；保存 full q/dq/root、93D obs、actual previous/issued action、command/gait/clock、controller state 与多层 hash；保守去重并与 immutable Stage335 做虚拟并集。
- 对照：`suffix_fraction=0` 不打开 sidecar、不消费 RNG、保持 Stage335 exact。
- 结果：schema/hash/no-op 合同 `20 passed`；真实 suffix rows=`0`；无 official、无训练。
- 结论：**数据合同可用，但有效性完全未知。**
- 下一步：备份后最多一次冻结 f005 suffix 采集。
- 证据：[Phase15 报告](baseline/x2_recovery_phase15_suffix_aggregation_design.md) / [Dry JSON](baseline/x2_recovery_phase15_suffix_aggregation_dry_schema.json)，JSON SHA `690604fb…0614`。

## BASE 结算

### 🟢 已确认的机制改善

- [x] 独立 recovery 真的拥有 authority，slot 计数可审计。
- [x] moving→recovery 的首 tick physical target 可做到完全连续。
- [x] actor 下一帧 previous-action 与实际执行 action/target 同步。
- [x] failure timeline 从“大概会倒”收窄到 action-history → gravity/root → height collapse。
- [x] 默认关闭、hash-bound 的 closed-loop suffix 采集/去重/虚拟并集合同已完成。

### 🔴 尚未通过的物理门

- [ ] source recovery 通过 stop/full：当前 `0/5`。
- [ ] f005 recovery 通过 stop/full：当前 `0/5`。
- [ ] 消除 handoff jump 后避免倒地：失败。
- [ ] suffix curriculum 改善 recovery：尚无数据、尚未训练。

### 🔒 当前训练锁

- `25-update`：锁定。
- recovery 长训：锁定。
- suffix fraction 扫描：禁止。
- 只有真实 suffix 通过 schema/hash/authority/window/outcome 审计后，才允许讨论一次 `fraction=0 vs 固定小 fraction` 的 5-update paired smoke。

## 🎯 BASE 下一任务：唯一 suffix 采集门

备份完成后，最多只做一次：

- [ ] 51822 与 ROS domain 独占；不与 WBT official 容器并发。
- [ ] 冻结 f005 ONNX SHA `9bc672fc…c0ca`。
- [ ] 冻结 Stage335 SHA `4d8ce06b…5013`。
- [ ] 固定 Phase13 history-synchronized handoff、fixed upper、stiff1.2、matched-event 合同。
- [ ] 唯一新增变量是启用 sidecar recorder，不改变 action。
- [ ] sidecar 只能包含 recovery authority 后 `+0.0–1.5s`，并绑定 trace/model/adapter/controller/physical hash。
- [ ] 先离线审计和去重；失败立即停，不训练。

---

# 🧠 路线 B：WBT29 / Faithful Any2Any

## 主任务

```text
SONIC/G1 source checkpoint（29D）
          ↓ exact named WBT29 alignment
X2 native Gold sanity + immutable train/held split
          ↓ faithful config / LoRA manifest
declared Isaac sole12 train domain
          ↓
zero-update equivalence gate
          ↓（尚未到达）
1-update smoke / PPO adaptation
```

## Phase10–25 游戏任务

| Phase | 状态 | 已落盘结果 | 属于什么证据 |
|---:|---|---|---|
| 10 | ✅ | 从官方 AimDK MuJoCo + 随包 ONNX 的约60s actual-state舞蹈导出 WBT29 Gold sanity seed；train/4s embargo/held-out 时间块冻结 | 数据/合同机制；不是GMR Silver |
| 11 | ✅ | MotionLib pose/root/FK、WBT29/head-lock round-trip 通过；dq/root velocity/contact 需显式 state adapter | ingestion机制 |
| 12 | ❌ | 裸PD prescribed-root 未过 body/contact/slip，free-root 未运行 | 物理门失败，不评价策略训练 |
| 13 | ⛔ | legacy NPZ 缺 publisher time/sequence、四组原子同步和身份hash，未进physics | 控制事件合同阻塞 |
| 14 | ✅* | active-WBT29 subscriber receipt event order、group index、identity hash 完整；publisher header/time与active head仍缺 | recorder机制改善，有明确边界 |
| 15 | ❌ | recorded q/Kp/Kd receipt-order replay仍未过body/slip，free-root未运行 | 物理门失败 |
| 16 | ⛔ | JOINT→RL表面5.141s，但完整可评分prefix仅0.882s，未进physics | capture资格失败 |
| 17 | ⛔ | 严格ready语义通过静态测试，但5s内未ready，无模式命令/资产 | 握手机制未闭合 |
| 18 | 🔍 | 只读日志无法证明只是等待不足，也不知道具体缺项 | 诊断不充分 |
| 19 | ✅* | 15s zero mode仍0/31且无odom/IMU，证明JOINT必须先激活telemetry | 机制事实；无physics |
| 20 | ⚠ | 51822端口冲突中止；不构成任何控制/物理结果 | 基础设施冲突 |
| 21 | ❌ | 4.823s prefix资格达成，但source已倒；RL段step/dq/limits失格且prescribed失败 | source质量+物理门失败 |
| 22 | 🧭 | readiness审计：离线资产够做sanity，但B1–B5/live zero-update未闭合 | 训练前总门 |
| 23 | ✅* | WBT29 gather/scatter/history、head exclusion、Gold train/held hook通过CPU zero-step | B1/B2机制就绪，未live接入 |
| 24 | ✅* | faithful PPO/reward/DR/std与SONIC decoder+critic、exact-S7两套独立LoRA manifest通过CPU zero-B forward | B3/B4机制就绪，未live运行 |
| 25 | ✅* | Isaac sole12 asset/runtime/solver-control provenance与hash guard冻结，官方差异显式列出 | B5作为“声明训练域”就绪，不等价官方物理 |

逐阶段证据入口：

- [P10 Gold](retarget/x2_native_gold_seed_phase10.md) · [P11 MotionLib](retarget/x2_native_gold_motionlib_phase11.md) · [P12 trackability](retarget/x2_native_gold_trackability_phase12.md)
- [P13 control contract](retarget/x2_native_control_contract_phase13.md) · [P14 event capture](retarget/x2_native_event_replay_phase14.md) · [P15 event replay](retarget/x2_native_event_replay_phase15.md)
- [P16 reset prefix](retarget/x2_native_reset_prefix_phase16.md) · [P17 ready](retarget/x2_native_reset_ready_phase17.md) · [P18 diagnosis](retarget/x2_native_ready_diagnosis_phase18.md)
- [P19 observable ready](retarget/x2_native_observable_ready_phase19.md) · [P20 activation](retarget/x2_native_activation_prefix_phase20.md) · [P21 activation replay](retarget/x2_native_activation_prefix_phase21.md)
- [P22 readiness](retarget/x2_faithful_any2any_readiness_phase22.md) · [P23 WBT29/Gold](retarget/x2_faithful_wbt29_gold_phase23.md) · [P24 faithful entrypoint](retarget/x2_faithful_any2any_phase24.md) · [P25 physics provenance](retarget/x2_physics_provenance_phase25.md)

完整 JSON SHA 已逐阶段记录在本卡片的机器可读版中。

## WBT 结算

### 🟢 已确认的机制改善

- [x] official31 被明确分为 WBT29 + head2，name-driven permutation 可逆。
- [x] policy obs/action/history 可以保持 source-semantic 29D，头部只在 simulator nominal。
- [x] Gold train/held/embargo 分离，MotionLib state adapter 合同闭合。
- [x] SONIC decoder+critic 与 Figure7 exact-S7 保留为两条独立 LoRA 对照，不再混称。
- [x] faithful PPO/reward/DR/std 与 trainable/frozen manifest 已冻结。
- [x] Isaac sole12 训练域 provenance 与运行时 guard 已冻结，官方 MuJoCo 差异没有被隐藏。

### 🔴 物理与学习尚未通过

- [ ] Phase12 bare-PD prescribed replay：失败。
- [ ] Phase15 recorded-control prescribed replay：失败。
- [ ] Phase21 activation source/replay：source本身失格且replay失败。
- [ ] free-root：三处都因 prescribed 失败而未解锁。
- [ ] faithful live zero-update：未执行。
- [ ] 1-update/PPO/LoRA训练：未执行。

### 🔒 当前训练锁

WBT 不是“代码都写了所以可以直接训练”。Phase23–25 分别把 B1/B2、B3/B4、B5 做成了隔离合同，但还需要把这些合同接入同一个 live launcher，并通过 zero-update。

## 🎯 WBT 下一任务：zero-update 前置门

只有以下全部打勾，才允许做 `env_steps=0 / optimizer_steps=0 / fixed_forward_batches=2`：

- [ ] faithful launcher 强制校验 Phase25 manifest SHA `cc02be06…dbb`。
- [ ] launcher 强制校验 resolved runtime snapshot SHA `96ab02bf…c77`，启动前 runtime guard 通过。
- [ ] Phase23 WBT29 gather/scatter/history 与 Gold state hook 真正接入 live Isaac train/eval config，而不是只停留在CPU probe。
- [ ] policy joint semantics严格29D；head不进入obs/action/history，simulator head保持nominal。
- [ ] optimizer sampler只含4个train keys；3个held-out keys与200帧embargo不进入optimizer/callback/early-stop路径。
- [ ] SONIC decoder+critic 与 exact-S7 分开执行、分开结论，不混合trainable scope。
- [ ] LoRA `B=0` 在两份固定batch上复现 source-space action mean 与 reference token，`max_abs≤1e-6`。
- [ ] std与source完全一致且冻结；dense/reference/FSQ/kinematic tensor hash不变。
- [ ] trainable names/mask严格等于所选manifest；critic running stats按语义映射且不静默重置。
- [ ] zero-update只输出小manifest，不输出candidate checkpoint。

zero-update 即使通过，也只证明初始化与接口等价，不证明 X2 性能改善；1-update仍需另行授权。

---

# 🔐 关键资产与哈希

| 资产 | SHA256 |
|---|---|
| Stage335 recovery reset set | `4d8ce06b6e8dd9b43363dadedb781e9feb75f72ac092b475de2a1e2772545013` |
| Phase9 f005 recovery ONNX | `9bc672fc3c535dbe6cd2709cdec4531eb9b61990457172e9c813a8ac713fc0ca` |
| frozen SONIC source checkpoint | `e6bdab3f64a39336b3d41877d4f497d05f58af275f288ec0e6746c283ded8909` |
| Gold train split | `644dc7534b63a7831bfe4156941b01508003f2834d2ccdac7b227369bad8ef8b` |
| Gold held-out split | `45ffda2f8ddc64cbeb4ccc714d37a0c98ca328e8e6473edb12670dcdfd19a3dc` |
| Phase25 physics manifest | `cc02be069637a488a7330928ab8500794f7972531ea093dc2be8cd839e247dbb` |
| Phase25 runtime snapshot | `96ab02bf2a974d1280fc37a3a8890131cbd4dc4f610197f6c5f5127cd9446c77` |

# ⚠️ 不能跨越的证据边界

- 本卡片没有任何 X2 真机运行；所有“official”均指 AimDK v1 官方 MuJoCo。
- official MuJoCo 与 Isaac sole12 是两个不同物理域；Phase25只冻结差异，没有证明等价。
- foot contact 是模型几何/仿真碰撞估计，不是真实足底六维力、GRF、COP、wrench或压力真值。
- Gold 是约60秒的官方仿真舞蹈 sanity seed，不是8小时目标机器人AMASS，也不新增GMR Silver。
- prescribed-root通过与否不等于自由平衡；目前free-root根本没有被解锁。
- BASE 的连续量改善不等于 recovery 成功；WBT 的CPU/static合同通过不等于live zero-update或学习成功。
- 没有第三机器人、没有跨机器人部署、没有真机安全切换结果。

# 📌 下一存档点

```text
BASE：备份 → 唯一一次 f005 suffix 采集 → 离线 sidecar/hash/去重审计 → 再裁决是否允许5-update

WBT：live接入 B1–B5 → Phase25 hash/runtime guard → zero-update（0 optimizer）→ 停止复核
```

在这两个门完成前，不以“继续多训一点”替代合同验证。
