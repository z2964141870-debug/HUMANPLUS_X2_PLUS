# X2 WBT Phase27：B1–B5 与数据证据矩阵（只读审计）

日期：2026-08-09
模式：只读取证；未运行 physics、训练、重定向或数据修改。

## 总裁决

- **24 条诊断面板已经建立，但只是 source selection + SHA 冻结，不是 24 条 official-v1 结果面板。**
- 面板覆盖现状：old-vs-official A/B `3/24`；有明确 Bronze 证据 `2/24`；Silver `0/24`；Gold `0/24`；其余 tier 证据为 `missing`。
- 面板外另有 Phase7 五动作 Bronze（其中 turn-left/right 与面板重合）和官方原生舞蹈 `4 train + 3 held-out` 的 **scoped native Gold sanity seed**。后者不是 GMR/AMASS Silver，且 Phase12 的裸 PD prescribed replay 未过，不能冒充跨具身 Gold。
- Phase22 的 B1–B5 已分别在 Phase23–25 实现，并在 Phase26 同一 fail-closed 入口中通过 zero-update；当前没有尚未解决的 B1–B5 技术阻断。
- **不能执行 faithful Any2Any 1-update。** 它包含 1536 rollout transitions 和 20 个 PPO optimizer minibatch steps，本质上就是 WBT PPO；把它称为 `pipeline sanity` 不会改变任务卡 §6“未通过 Bronze/Silver 诊断面板，不启动 WBT PPO”的硬规则。
- Phase26 只说明网络、数据接口和 B=0 初始化在技术上已就绪；**训练授权仍被数据门阻断**：必须先完成固定 24 条的 official-v1 old/new A/B 与 Bronze/Silver 分级，并得到非空的 GMR Silver/Gold train subset。至少 1 条 train-candidate GMR Silver 才能让 optimizer 数据非空；要满足 faithful baseline 的独立评估，还需至少 1 条不相邻、held-out GMR Silver/Gold。

## Phase22 B1–B5 requirement matrix

| 项 | 当前状态 | 直接证据 | 未覆盖边界 |
|---|---|---|---|
| B1：WBT29 runtime alignment | READY | Phase23 dedicated gather/scatter/history；Phase26 action=29、head 不入 policy、29→G1→29 exact | 未证明训练后性能 |
| B2：Gold state adapter 进入 train/eval | READY | Phase23 immutable train/held hooks；Phase26 两个固定 batch 实际调用且 split 隔离 exact | Gold 是原生 sanity seed，不是 GMR corpus |
| B3：faithful PPO/reward/DR config | READY | Phase24 锁定 epochs=5、minibatches=4、std frozen、source reward/DR；Phase26 hash guard exact | 尚未执行 optimizer |
| B4：exact S7 LoRA 范围 | READY | Phase24 actor proprio columns/backbone/output + critic backbone manifest；Phase26 trainable names exact、仅 A/B trainable、B=0 exact | 尚无非零梯度证据 |
| B5：目标 physics provenance | READY_AS_DECLARED_TRAIN_DOMAIN | Phase25 sole12 Isaac asset/config/runtime hash guard；official AimDK v1 明确为 held-out mismatch domain | 不宣称 Isaac 与 official MuJoCo 数值等价 |
| B6：zero-update | PASS | Phase26 两个固定 batch；token/action/value base-vs-B0 max_abs 全为 `0.0`；0 env、0 physics step、0 optimizer | 不是训练或性能证据 |

## 24 条诊断面板覆盖

面板文件状态为 `selection_only_no_retarget_no_training`，24/24 source 存在并带 SHA；推荐的 14/10 split 仍只是建议。

| 诊断 ID | current official-v1 证据 | tier | old-vs-official A/B |
|---|---|---|---|
| AMASS-STAND-001 | missing | missing | 否 |
| AMASS-UPPER-001 | smoke3 official 输出存在 | unclassified（未跑 tier gate） | 是，数值 exact 相同 |
| AMASS-WALK-001 | smoke3 official 输出存在 | unclassified（未跑 tier gate） | 是，数值 exact 相同 |
| AMASS-TURN-L-001 | Phase7 canonical artifact | Bronze | 否 |
| AMASS-TURN-R-001 | Phase7 canonical artifact | Bronze | 否 |
| AMASS-SQUAT-001 | smoke3 official 输出存在 | unclassified（未跑 tier gate） | 是，数值 exact 相同 |
| AMASS-KICK-L-001 | missing | missing | 否 |
| AMASS-KICK-R-001 | missing | missing | 否 |
| AMASS-COORD-FAST-001 | missing | missing | 否 |
| AMASS-FAIL-CROUCH-001 | missing | missing | 否 |
| AMASS-FAIL-CIRCLE-001 | missing | missing | 否 |
| AMASS-FAIL-THROW-001 | missing | missing | 否 |
| PHUMA-LUNGE-R-001 | missing | missing | 否 |
| PHUMA-LUNGE-L-MIRROR-001 | missing | missing | 否 |
| PHUMA-RAISE-L-001 | missing | missing | 否 |
| PHUMA-RAISE-R-001 | missing | missing | 否 |
| PHUMA-SQUAT-001 | missing | missing | 否 |
| PHUMA-FAIL-MOVE28-001 | missing | missing | 否 |
| PHUMA-COORD-001 | missing | missing | 否 |
| BONES-WALK-FAIL-001 | missing | missing | 否 |
| BONES-TURN-FAIL-001 | missing | missing | 否 |
| BONES-STOP-FAIL-001 | missing | missing | 否 |
| BONES-SIDE-FAIL-001 | missing | missing | 否 |
| BONES-JOG-FAIL-001 | missing | missing | 否 |

old-vs-official A/B 的三条仅证明“把 legacy XML 换成 official x2.xml 后，重定向数值未改变”（global max abs difference `0.0`）；它们没有因此自动成为 Bronze。

## current official-v1 产物分层

| 资产 | 数量 | 分层 | 证据边界 |
|---|---:|---|---|
| smoke3 official retarget outputs | 3 motions | unclassified | 仅模型文件 A/B；未跑 Bronze/Silver/Gold tier gate |
| Phase7 canonical root-ground | 5 motions | Bronze | 五动作 geometry/prescribed/survival 过；walk-to-stand slip 失败，五动作 contact agreement 全失败，故 Silver=0 |
| Phase9 schedule | 5 motions | rejected derivative | 统一 hysteresis/min-dwell 未改善五动作；不改变 Phase7 Bronze |
| Phase10 official native dance seed | 7×8s clips（train4/held3；中间 embargo4s） | scoped native Gold sanity seed | 官方模拟器 actual-state 原生轨迹；不是 GMR/AMASS Silver，不是实机真值 |
| Phase12 held-out clip0 bare-PD replay | 1 clip | rejected trackability smoke | prescribed-root 未过，free-root 未运行；只否定裸 PD replay，不否定 Any2Any |

Phase7 五动作是：`KIT walking_slow02`、`ACCAD turn-left`、`ACCAD turn-right`、`ACCAD stand-to-walk`、`ACCAD walk-to-stand`。只有左右转两条与 24 条诊断面板重合。

## 唯一下一实验

只执行 **固定 24 条诊断面板的 official-v1 old/new A/B + Bronze/Silver 资格审计**，不启动 optimizer：

- 保持已经预注册的 official body/joint/root/contact contract 与 tier threshold 不变；对 24 条逐条生成/核验 official-v1 产物，不根据结果逐 clip 调参。
- 每条输出 old-vs-official 差异、Bronze 各子门、Silver 接触门、reject reason 与 provenance；保留预先指定的 train-candidate/held-out 身份。
- 晋级 WBT 1-update 的最低数据门：分级流程在完整 24 条上稳定完成，且至少出现 `>=1` 条 train-candidate GMR Silver；要执行带独立评估的 faithful baseline，还必须另有 `>=1` 条 held-out GMR Silver/Gold。
- 若完整面板仍为 GMR Silver=0，则继续禁止所有 WBT PPO；结论指向 official-X2 reference 生成器/数据合同，不用 Gold-native seed 绕过。

## 证据文件与 SHA256

- `x2_wbt_diagnostic_panel.json`: `cbb8889b6b900f2a00b46409909449d17b24bdc6b3f79e28b91bbafb241ccf60`
- `x2_wbt_tier_gates.json`: `314edd1970083fa5549929f4c8e79e206e5ece07718955442a9d943e75d2a934`
- `x2_official_retarget_smoke3.json`: `69a98ab6eee76d8ae691afa45a09e90662fdb4c7e05df9296dd700dfbe045a06`
- `x2_wbt_canonical_root_ground_phase7.json`: `bdd703cf7c696c0146494172815e173a4a36ebf74cdad7e74791f6389f02377b`
- `x2_wbt_contact_schedule_phase9.json`: `4dbc7249a6fbee46faf10a970f4587a3ab51fdaba0c248fe2b56bfac9ee448e3`
- `x2_native_gold_seed_phase10.json`: `72b5377f02e2cee266ad5ebc8ea809d4c7b980396926b34f0d3b15436bff8f4e`
- `x2_native_gold_trackability_phase12.json`: `d06db9349e31d67a7233fe4f39eb9ebfb3d751a4e6d4ce2b39653f1e2fbd522d`
- `x2_faithful_any2any_readiness_phase22.json`: `bfcfcf6db4b42b4a3e4ef579d8c99a904a068bfb3467f062fd3170b32d6ded92`
- `x2_faithful_wbt29_gold_phase23.json`: `32e82f3747eebf1c0cd0938a65e30bccf66226f5adc34cd9b14b47075e92475d`
- `x2_faithful_any2any_phase24.json`: `c3e2e58eea9f9166feff3eb55924bab6f61db25bb8adf48dca9c02b73a032245`
- `x2_physics_provenance_phase25.json`: `1a7c0ce96f88749efc6ff886db22989bd6ad82f4c9671cbb4f2a01566c40b02f`
- `x2_faithful_zero_update_phase26.json`: `a77e3421c320b7be572a8d9e29371db68dd8683ab1e5445edf36d8ff748aaf14`
