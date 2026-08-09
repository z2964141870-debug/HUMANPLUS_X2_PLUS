# X2 WBT official-v1 诊断面板（24 条）

版本：Phase0 / 2026-08-09
状态：**仅选择与冻结源；尚未执行 official-X2 重定向或训练。** 24/24 源路径存在，24/24 已登记逐文件 SHA-256（完整值见同名 JSON）。

## 设计目的

这不是训练集，也不是“挑最好看的动作”。它是用于旧 X2 contract 与 AimDK v1.0 official-X2 contract A/B 的固定诊断面板，覆盖：standing/upper-only、forward walk、左右 turn、squat、左右 lunge、左右 single-leg raise、左右 kick、快速全身协调，以及 10 条明确或系统性旧路线失败动作。

建议分割为 14 条 `train_candidate` 与 10 条 `held_out`。在 official retarget 的 Bronze/Silver 审计完成前，`train_candidate` 也不代表获准训练。

## 面板

| ID | 来源 | 类别 | split 建议 | 源动作 | 选择理由 |
|---|---|---|---|---|---|
| AMASS-STAND-001 | AMASS/ACCAD | standing | train | `Female1General/A1 Stand` | 隔离默认姿态、root 高度和地面 contract |
| AMASS-UPPER-001 | AMASS/ACCAD | upper-only | train | `Male1General/Swing Arms While Stand` | 双脚近固定时检查上肢表达是否扰动下肢 |
| AMASS-WALK-001 | AMASS/ACCAD | forward walk | train | `Female1Walking/B3 walk1` | 标准 overground 直行，与速度型后端可比较 |
| AMASS-TURN-L-001 | AMASS/ACCAD | left turn | train | `B11 walk turn left 135` | 世界 yaw、左转支撑切换与镜像语义 |
| AMASS-TURN-R-001 | AMASS/ACCAD | right turn | held-out | `B13 walk turn right 45` | 方向泛化 held-out，不与左转混成平均分 |
| AMASS-SQUAT-001 | AMASS/KIT | squat | train | `KIT/3/squat05` | 大 root-z、髋膝屈曲，覆盖非步行下肢表达 |
| AMASS-KICK-L-001 | AMASS/ACCAD | left kick | train | `G17 push kick left` | 明确左腿，要求真实摆脚和单支撑 |
| AMASS-KICK-R-001 | AMASS/ACCAD | right kick | held-out | `G18 push kick right` | 左右动作泛化，不允许只记住左腿 |
| AMASS-COORD-FAST-001 | AMASS/Transitions | fast whole-body | held-out | `dance_kick` | 快速舞蹈→踢腿的全身协调高难样本 |
| AMASS-FAIL-CROUCH-001 | AMASS/Transitions | legacy failure | train | `crouchwalk_stand2` | 旧 hard-negative；低 root 与接触转换冲突 |
| AMASS-FAIL-CIRCLE-001 | AMASS/BMLrub | legacy failure | held-out | `rub001/0017 circle walk` | 旧 high-alignment-error 转向 hard-negative |
| AMASS-FAIL-THROW-001 | AMASS/BMLrub | legacy failure | held-out | `rub001/0014 catching and throwing` | 快速上肢造成 root/平衡扰动的 hard-negative |
| PHUMA-LUNGE-R-001 | PHUMA/G1 | right lunge | train | `Side_Lunge_Runner_R clip6` | G1 物理约束源中的大步距、低 root 右弓步 |
| PHUMA-LUNGE-L-MIRROR-001 | PHUMA/G1 | left lunge | train | 上一条的 canonical mirror | 直接验证 X2 左右 mirror round-trip；与右弓步不得拆分 split |
| PHUMA-RAISE-L-001 | PHUMA/G1 | left leg raise | train | `Standing side leg lifts L` | 慢速左腿单支撑 |
| PHUMA-RAISE-R-001 | PHUMA/G1 | right leg raise | held-out | `Standing side leg lifts R` | 独立右腿源作为侧别泛化 |
| PHUMA-SQUAT-001 | PHUMA/G1 | squat | train | `Bodyweight Squat` | 与 AMASS squat 做跨来源一致性对照 |
| PHUMA-FAIL-MOVE28-001 | PHUMA/G1 | legacy failure | held-out | `Move_Walk clip28` | 旧接触候选，但 free-root 四域均 root 过低 |
| PHUMA-COORD-001 | PHUMA/G1 | upper+locomotion | held-out | `Calling and walking` | 上肢、腰部与步态同时存在 |
| BONES-WALK-FAIL-001 | BONES/SOMA | old fail walk | train | `walk_forward_loop_002 A021` | 旧漏斗中性前行失败，用于 official contract A/B |
| BONES-TURN-FAIL-001 | BONES/SOMA | old fail turn | held-out | `Turn_Start_Walk_0225 A018` | 起步+转向组合，暴露 root-foot 时序 |
| BONES-STOP-FAIL-001 | BONES/SOMA | old fail stop | train | `walk_ff_stop_360_R_slow A444` | 连接 WBT 与 stop-to-stand 问题 |
| BONES-SIDE-FAIL-001 | BONES/SOMA | old fail sideways | train | `walk_sideway_090_loop A041` | 横向 root 与支撑脚滑移诊断 |
| BONES-JOG-FAIL-001 | BONES/SOMA | fast old failure | held-out | `jog_arc_cw_loop A093` | 高速接触+弧线转向+全身协调 |

## Split 与泄漏规则

1. `PHUMA-LUNGE-R-001` 和其 mirror 左弓步是同一底层源，必须始终在同一 split；不能用一个训练、另一个宣称 held-out。
2. 左右 kick 来自独立源，左 kick 推荐训练、右 kick 推荐 held-out，用于检验跨侧泛化。
3. AMASS 右转、PHUMA 右腿抬举、BONES 起步转向与弧线 jog 均保留为 held-out；不得因为难而移回训练集。
4. 旧失败动作不能在 official retarget 后静默删除；应报告“仍失败 / 改善 / 新退化”的事件级原因。

## 下一阶段保存合同

对每个 ID，official A/B 应新增而不覆盖：

```text
source motion + SHA-256
old-X2 retarget（若存在）
official-X2 retarget
official model / joint-body-map / mirror-contract hashes
scale、坐标变换、优化目标、target fps
contact labels 与来源（FK estimate / simulator contact，绝不写成真实力）
Bronze/Silver/Gold 指标与 reject reason
```

优先 3 条 smoke 建议：

1. `AMASS-UPPER-001`：最容易隔离 body/hand/foot mapping 和默认站姿；
2. `AMASS-WALK-001`：标准动态基准，与旧 Core24 直接 A/B；
3. `PHUMA-FAIL-MOVE28-001`：旧路线的最强否证样本，可检验官方模型是否真正改善 free-root 动力学。

这三条 smoke 通过 Bronze/Silver 自动审计后，再按本面板扩展到 24 条；本报告本身不授权重定向或训练。

## 证据入口

- AMASS 源与 Core24/Pilot58：`/home/humanplus/x2_teleop_final/x2_sonic/x2_gmr_data_rebuild/outputs/amass_source_summary.md`、`amass_x2_unified_audit_20260717.md`
- AMASS hard-negative：`.../outputs/x2_gmr_hard_negative_panel_4.json`
- PHUMA native/扩展：`.../outputs/phuma17_native_x2_next_audit.md`、`phuma_stage169_all36_warmstart_tier_20260717.md`
- PHUMA clip28 free-root 失败：`.../outputs/x2_three_source_unified_manifest_20260717.json`
- BONES 全量漏斗：`.../outputs/bones_seed_uniform_full_funnel_20260717.md`
