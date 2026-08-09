# BASE Phase34：Stage250 closed 全程 1 kHz 动力学种子资格审计

## 一句话裁决

完整 Stage250 straight 的 closed 1 kHz 物理真轨迹已经成功拿到，且机器人全程通过 startup/move/stop/full 门；它可作为 **X2 原生物理 warm-start**，但在预先沿用的严格门下仍不能直接晋级为 dynamic Silver seed：真实接触存在高分位滑移和低净空，命令应用时刻不能由现有字段唯一恢复，公开字段重建的 generalized-force 方程也未达到 exact closure。

## 游戏任务

- [x] 冻结 scene、Stage219、stand actor、adapter、template 与 Phase32 observer hash。
- [x] 唯一实际 physics episode；无重试、无训练、无策略修改。
- [x] 完整记录 14.406 秒、14406 个 reset 后 physics step。
- [x] 记录 qpos/qvel/qacc/warmstart/ctrl/qfrc_actuator/qfrc_constraint/contact force 与调用顺序。
- [x] 710 条 telemetry 与物理状态单调精确对齐。
- [x] 用真实 collision/contact 判定支撑、滑移、净空和周期。
- [ ] strict X2-native dynamic seed。
- [ ] contact-clean Silver seed。
- [ ] hardware dynamics truth。

## 假设

Phase27 只能从 50 Hz telemetry 重建接触和动力学，因此 Stage250 三条动作只能当运动学 warm-start。若直接观察 closed official MuJoCo 的全部 1 kHz 子步，可能证明 straight 是接触一致、动力学闭合的 X2 原生 seed。

## 干预

没有控制干预。复用 Phase32 已验证无扰动的 `LD_PRELOAD` observer，把 capture window 从 0.3 秒扩到 15 秒，容量从 4096 扩到 32768 records；原 Stage250 控制合同保持不变。

大 trace 只保存在 official 外部结果目录，repo 只保存 manifest、报告和工具。

## 基础设施 provenance

第一次外层命令误用了 ROS domain 234，CycloneDDS 计算得到端口 65900，adapter 和 simulator 均在 ROS node 初始化时退出；没有进入 MuJoCo、没有 mmap、没有 episode。该次被记为 `PRE_PHYSICS_INFRA_INVALID`。

经明确授权后，只把 domain 改为合法且空闲的 230，并保留原失败日志；scene/controller/observer/容量等所有 physics 合同不变。随后运行唯一实际 physics episode，之后没有补跑。

## 结果

### 1. 完整性和无扰动门通过

- Stage250：startup=true、move=true、stop=true、full=true。
- 位移：forward 1.180 m、lateral -0.084 m。
- 停车：XY drift 0.067 m，settle 2.02 s。
- mmap：623,378,688 bytes（约 594.5 MiB）。
- committed=28,818、dropped=0、sequence 连续、单一 `mjData` pointer。
- 调用：reset=3、forward=14,408、step=14,407。
- 最后一次 reset 后：14,406 个 1 kHz step，时间 0.001–14.406 秒。

observer 没有破坏 Stage250 的原有 full-gate 能力。

### 2. telemetry 可以精确映射，但不是固定 20 ms

不能把 50 Hz row 简单映射为固定每 20 个 physics step。单调最近状态对齐后，710 行 root state 的 score p95=`2.70e-14`、max=`2.14e-11`，属于数值精确匹配；但相邻 callback 实际跨越 15–24 个 physics step：

| Δphysics steps | 15 | 16 | 17 | 18 | 19 | 20 | 21 | 22 | 23 | 24 |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 次数 | 5 | 7 | 18 | 46 | 58 | 523 | 31 | 16 | 4 | 1 |

这再次证明 Phase28/31 的关键错误：ROS 50 Hz telemetry 是可对齐的观测，不是可按固定 cadence 还原的 MuJoCo snapshot 序列。

### 3. action 的应用时刻仍不能唯一恢复

trace schema 本身还有一个限制：

- prepare 10 行没有完整 lower target/action；
- stand+move 300 行只能由 final action 和冻结 scale 重建 target；
- stop 400 行保存了 physical lower target。

按冻结 PD 合同比较 action target 与 realized `ctrl`，0 ms lag 的 median max-error 为 1.293，向 20 ms 持续降至 0.028；但 20 ms 的 p95 仍为 2.038，且没有一行达到 `1e-5` exact。20 ms 是当前搜索边界，不是可信标定值。

能得出的结论仅是：action 在 telemetry sample 之后进入 simulator；现有 trace 没有 command receipt/application timestamp，不能从这条数据唯一恢复每条动作何时开始生效。

### 4. 真实 contact 证实“能走，但不是干净步态”

| 指标 | 左脚 | 右脚 |
|---|---:|---:|
| move realized contact fraction | 0.785 | 0.760 |
| 与 generator contact agreement | 0.645 | 0.570 |
| stance slip p50 | 0.0257 m/s | 0.0263 m/s |
| stance slip p95 | 0.425 m/s | 0.392 m/s |
| swing clearance p95 | 7.66 mm | 9.47 mm |

realized contact 出现 26 个 DS→SS→DS 周期，而 generator 只有 10 个，说明真实碰撞包含额外切换/刮擦。多数支撑帧滑移不大，但 p95 明显超过沿用的 0.20 m/s strict gate；摆脚净空仍不足 1 cm。

这不是“机器人没走”：full gate、1.18 m 位移和真实接触周期都成立；但它也不是可直接监督理想卸载—离地—落脚的 clean Silver。

### 5. qacc 是精确的，完整 force closure 尚未证明

半隐式积分检查中，`Δqvel/0.001` 与记录 qacc 的 absmax p95=`2.12e-13`，证明 1 kHz acceleration/velocity 序列自洽。

但用记录 `qfrc_actuator + qfrc_constraint`、官方模型 passive 与 RNE 重建 generalized equation 时，relative residual p50=`2.96e-4`、p95=`8.89e-3`，没有达到严格 `1e-8`。这可能包含 solver workspace 时序或未记录 applied-force 通道；本阶段不能把它事后解释成 exact dynamics truth。

### 6. 后仰仍然存在

move signed pelvis/root pitch mean=-0.198 rad（约 -11.35°），与 Phase7/27 一致。目标本体闭环能力真实，但姿态风格偏差没有因为使用 official closed physics 自动消失。

## 结论

### 成功之处

首次获得一条完整、稳定、目标 X2 原生、1 kHz closed physics rollout。它比 AMASS/GMR reference 更适合作为：

- X2 contact-aware trajectory optimization 初值；
- brake/bridge/CEM teacher 的 native warm-start；
- closed vs direct 的 exact-state regression oracle；
- reference repair 的目标本体基线。

### 仍未通过之处

它不能原样成为：

- contact-clean dynamic Silver；
- 精确 command→physics 时序标签；
- exact generalized-force truth；
- 自然直立姿态 teacher；
- 真机 GRF/COP 数据。

因此严格裁决为 `NOT_QUALIFIED_X2_NATIVE_DYNAMIC_SEED_UNDER_STRICT_GATE`，但不是“这次采集没价值”：数据层已经从 50 Hz 猜测推进到完整的 closed 1 kHz 物理轨迹，下一步优化终于可以从目标机器人真实闭环轨迹出发。

## 下一步

不应继续采相同 raw rollout，也不应立刻拿它长训。最小高价值路线是对这条 native physical warm-start 做一次离线 contact repair：保留 forward/stop/root 主语义，只优化接触切换附近的低维足端/髋腰轨迹，硬门要求左右 stance-slip p95 降到 0.20 m/s 以下、摆脚净空提高，同时 full-gate 和后仰不退化。

若未来必须把 command timing 或完整力闭合变成标签，recorder 至少还要加入 simulator command receipt/application timestamp、完整 lower target 每阶段，以及 `qfrc_applied/xfrc_applied`；不能靠本条 trace 猜这些缺失量。

## 证据

- Manifest：`reports/official_x2/phase34_closed_full_trace_manifest.json`
- Full audit：`/home/humanplus/projects/ZHY/x2_official_rl_deploy_v1/cache/phase34_full_stage250_trace/phase34_full_offline_audit.json`
- Full mmap：`/home/humanplus/projects/ZHY/x2_official_rl_deploy_v1/results/phase34_full_stage250_trace/phase34_stage250_closed_full_once_d230.mmap`，SHA256 `8bb90c96...24ef`
- Rollout JSON：同目录 `phase34_stage250_closed_full_once_d230.json`，SHA256 `c380b920...c376`
- 预注册与 infra 修订：`reports/official_x2/phase34_closed_full_trace_prereg.json`、`phase34_closed_full_trace_prereg_amendment.json`
- Tests：3 passed；无训练、无策略改动、无 WBT、无真机、无 Git/百度操作。
