# X2 动态重定向多方法验证任务卡

版本：v1.0
日期：2026-08-10
执行者：独立 agent
审核者：当前项目主 agent / 用户
执行原则：先证据、后优化；逐级解锁；单方法达标即停；所有失败均保留。

## 0. 任务目标

在几何重定向与 WBT/RL 之间建立一条可审计的“动力学可行化”工序，判断动态重定向能否解决 X2 当前 reference 在自由根物理中的失败。

本卡不是要求同时实现四种方法。执行优先级为：

1. **Shooting for Contact（DSMS）最小忠实复现与 X2 单动作验证**；
2. **OmniTrack Stage I 小规模物理生成器**，仅在 DSMS 给出正证据或明确暴露可学习修复目标后启动；
3. **DDR** 等待完整代码，或只做 readiness/最小自研 smoke；
4. **DynaRetarget/SBTO** 作为长时域采样优化备选，不在短弓步上优先投入。

最终业务成功标准不是优化器 loss 下降，而是：用动态可行 reference 训练后，冻结四域门禁中的 ideal 从 `0/4` 提升到至少 `2/4`，且位移过冲下降。

## 1. 已知事实与禁止重写的历史结论

### 1.1 当前失败不是单纯 Sim2Real

- X2 forward-walk 四域门禁：ideal `0/4`、filter `1/4`、delay `0/4`、delay+noise `0/4`。
- ideal 已失败，旧 ideal 平均绝对 progress-ratio error 为 `2.303`；不能把主要问题归因于滤波、延迟或噪声。
- 当前 forward4 来源是 X2 native/readonly-derived，不是 GMR。审计时必须按真实 provenance 描述，不能统称“GMR forward walk”。

### 1.2 PHUMA lunge 是首选最小验证对象

- 固定输入：`PHUMA-LUNGE-R-001`，Phase30 版本，175 帧、30 Hz、5.8 s。
- prescribed-root 裸 PD 能跑满 `5.8/5.8 s`，但 free-root 仅 `0.575 s`；Faithful Any2Any 动态评估约 `0.22 s`。
- exact-joint kinematic oracle 也只存活 `0.324 s`。
- Phase36 已撤销其历史 Silver：官方 sole collision 下左右接触占比约 `0.0343/0`、flight `0.9657`，它只能算 Bronze。
- Phase41 固定局部轨迹在官方几何与 GRF 可行性上为 `0/23` jointly feasible。

这些证据说明：该动作适合验证“动力学可行化是否能救活 reference”，但不得预设 DSMS 必然成功。

### 1.3 旧尝试不能冒充新方法

以下工作已有价值，但不等同于论文方法：

- Phase29/32：全轨迹 LSQR；
- Phase37/38：SLSQP 局部约束；
- Phase39/40：DAQP sequential QP；
- Phase41/42：centroidal force LP/contact-first；
- BASE Phase26/36/37：局部 CEM bridge/swing teacher。

若新实现没有把 MuJoCo 离散动力学、shooting node、defect continuity、控制变量和状态/控制约束纳入同一问题，不得命名为 Shooting for Contact/DSMS。

## 2. 固定资产与执行环境

### 2.1 主项目与 Python 路径

- 主仓库：`/home/humanplus/projects/ZHY/CWI_CrossEmbodiment_Sim`
- `official_x2` Python namespace 位于 `$REPO/tools/official_x2`；运行时使用：

```bash
export PYTHONPATH="$REPO/tools:$REPO/src:$PYTHONPATH"
```

### 2.2 官方 X2 物理合同

- SDK：`/home/humanplus/projects/ZHY/x2_official_rl_deploy_v1`
- scene：`worktree/x2_rl_deploy/x2_rl_deploy_mujoco/configuration/robot/lx2501_3_t2d5/model_info/scene.xml`
- robot XML：同目录 `x2.xml`
- 控制配置：`worktree/x2_rl_deploy/x2_rl_deploy_controller/config/motion_control.yaml`
- 固定合同：MuJoCo 3.3.7、1 kHz physics、50 Hz control、31 DoF articulation、策略/WBT 仅29个非头关节、head2 nominal、官方 joint order/PD/ctrlrange/mesh/sole collision。

不得用简化 URDF、替代碰撞体或非官方 PD 得出 Gold 结论。替代域只能标为 diagnostic。

### 2.3 固定输入

- lunge：`/home/humanplus/projects/ZHY/x2_wbt_official_v1/motion_lib_x2_official_v1/silver_contact/phase30_time_dilation/x2_phase30_time_dilation_1p46.pkl`
- forward4 native source：`/home/humanplus/x2_teleop_final/x2_sonic/data/processed/x2_real_readonly_session04_canonical_31dof.npz`
- forward4 MotionLib：`/home/humanplus/x2_teleop_final/x2_sonic/motion_lib_x2/stage72_official_true_forward4_v1`

### 2.4 迁移服务器已有材料

服务器：`root@59.77.15.15:25142`，根目录 `/opt/data/private/data_fixed/`：

- `x2_official_rl_deploy_v1_source_assets_20260810.tar.gz`
- `Humanoid-GPT_requested_tree_20260810.tar.gz`
- `general_motion_retargeting_actual_runtime_20260810.tar.gz`
- `x2_phase55_native_upper_robust_20260809.tar.gz`
- `x2_phase42_dual_track_20260809.tar.gz`
- `x2_phase30_dual_track_20260809.tar.gz`
- `x2_phase10_recovery_native_gold_20260809.tar.gz`

百度网盘种子包：

- `HUMAN+/HUMANPLUS_X2_PLUS/2026-08-10/dynamics_feasibility/x2_forward4_dynamics_feasibility_seed_v1.tar.gz`
- `HUMAN+/HUMANPLUS_X2_PLUS/2026-08-10/runtime/x2_aimdk_runtime_images_v1.tar.gz.part-00`
- `HUMAN+/HUMANPLUS_X2_PLUS/2026-08-10/runtime/x2_aimdk_runtime_images_v1.tar.gz.part-01`

执行者先做 SHA/manifest 验证，不得静默用相似文件替换。

## 3. 全局实验纪律

1. 每阶段先写 `prereg.json`：输入 SHA、代码 SHA、scene/model/config SHA、seed、预算、变量、指标、阈值、停止条件。
2. 任何阈值只能在运行前固定；运行后不得为通过而调阈值。
3. 一个阶段只允许预注册的重试；infra/pre-physics 失败与物理失败分开记账。
4. train/held-out 严格隔离；不得用 held-out 调权重、窗口、接触时序或 solver 参数。
5. prescribed-root 只证明 trackability；只有 independent free-root replay 才能证明动态可行性。
6. 碰撞标签使用官方 12 个 active sole spheres/foot 与 floor 的真实 collision/signed distance `<=0`；不得复用 10.05 mm Bronze hovering tolerance。
7. optimizer loss、训练 episode length、prescribed 成功都不能代替目标动作 free-root 成功。
8. 不删除失败 artifact；每次报告必须写假设、干预、对照、结果、裁决、边界、下一步。
9. 大文件置于外部 results/cache，Git 只放代码、小 JSON、MD、manifest、hash；不自动上传百度网盘。

## 4. Phase 0：五项全离线前置审计

这一步不使用 GPU，不跑新 physics。对 lunge 与 forward4（含真实存在的镜像/左右样本）分别产出逐帧表和汇总。

### A0.1 重定向/参考伪影

逐帧标注：

- 官方 sole 穿地深度与帧占比；
- contact core 脚滑速度与 stance excursion；
- 关节 limit/ctrlrange 命中；
- severe self-collision；
- 关节单帧突跳；
- root acceleration、COM acceleration、centroidal momentum rate。

### A0.2 接触时序矛盾

分开保存三列：source intent、官方几何 contact、物理 rollout realized contact。报告“意图承重但 official sole 未接触”和“意图 swing 但 realized 持续接触”的比例与典型时间窗。

### A0.3 根速度分布

对真正进入当前训练器的 forward-walk train split 统计 root linear velocity；叠加四域评估命令，给出命令落在训练分布外的比例。若无法证明数据真正进入 optimizer，结论必须是 `BLOCKED_BY_TRAIN_PROVENANCE`。

### A0.4 镜像与对称性

统计镜像/非镜像数量、时长、速度方向与 split；静态审计策略是否有对称 loss、镜像 augmentation、equivariant 层或仅普通 MLP。区分“数据镜像”与“网络归纳偏置”。

### A0.5 延迟随机化

审计训练配置与 runtime：通信延迟、执行延迟、观测历史、filter、noise 的分布及是否实际被采样。只读配置不够，需给 live manifest 或日志证据。

### Phase 0 判定

- 穿地 `>2%` 或接触矛盾 `>5%`：允许进入动态可行化方法。
- 根速度 OOD 高：同时开数据/命令分布问题，但不得假称动态可行化会解决它。
- 若 1/2 不显著而 3 显著：暂停方法试，先交回用户拍板。

## 5. Phase 1：Shooting for Contact 上游最小忠实复现

目标：先证明第三方代码与论文方法在其自带模型/示例上能运行，再接 X2。禁止一上来改 X2 后无法区分算法与适配错误。

### 5.1 来源与许可证

- 论文/项目/代码必须锁 commit SHA。
- 当前候选仓库：`sesteban951/shooting-for-contact`。
- 执行前核查 LICENSE。若仓库仍无许可证：仅限内部研究阅读/运行；不得复制源码进本项目、不得对外分发、不得上传公共仓库。
- MuJoCo、IPOPT、MUMPS/HSL/MA57 分别记录版本与许可证。HSL/MA57 不可随环境包传播。

### 5.2 上游 smoke

按 README 的最小例子完成：

- 原始 example 不改模型；
- 记录 IPOPT status、iteration、objective、constraint violation、wall time、solver；
- 输出轨迹重新用独立 MuJoCo 进程 replay；
- 验证 shooting defect、joint/torque/control constraints 与 replay 一致。

若 MUMPS 可运行但 MA57 不可得，先用 MUMPS；不得因缺商业/受限 solver 伪造 MA57 结果。

### Phase 1 门

上游例子必须可重复完成且独立 replay 不发散。否则停止在“上游复现失败”，不得进入 X2。

## 6. Phase 2：X2 DSMS 适配器与零变量合同

此阶段不求解，仅证明 X2 问题定义无误。

实现并测试：

- official 31DoF ↔ policy WBT29 显式按名字 gather/scatter；head2 始终 nominal；
- 30 Hz reference 到 50 Hz control 的 linear + quaternion SLERP，端点精确保留；
- 50 Hz control 与 1 kHz MuJoCo integration 对齐；
- state 含 root pose/velocity、q/dq，control 为官方 PD target 或论文实现所需控制变量；
- official PD、ctrlrange、joint limits、collision scene 固定；
- shooting node state、defect continuity、cost/constraint 单位全部写入 manifest；
- zero-delta/reference replay 与既有独立 runner 逐项一致；
- finite-difference gradient 做 directional derivative spot-check。

### Phase 2 门

- 名称映射、head lock、端点、时间轴、零变量 roundtrip 全部通过；
- gradient spot-check 误差在预注册容差内；
- 任何一个失败即停止，不进入 NLP。

## 7. Phase 3：PHUMA lunge 的 DSMS 最小验证

### 7.1 顺序

1. 固定一个短前缀（覆盖 DS→SS→DS）做小规模 solve；
2. 短前缀通过 independent free-root replay 后，才扩 5.8 s 全段；
3. 不改变动作语义、时间尺度、root XY、接触标签或门禁来追求收敛；任何扩展必须另立任务。

### 7.2 必须优化/约束的量

- MuJoCo 离散状态与控制；
- shooting defects；
- root/body/keypoint tracking；
- joint/velocity/torque/ctrlrange；
- 自碰撞与官方 sole/floor contact；
- 控制变化率/力矩正则；
- 必要时 receding horizon，但窗口、overlap、warm start 先预注册。

### 7.3 动态 reference 资格门

输出必须同时满足：

- 所有数值有限、joint order/head lock 正确；
- joint/torque/ctrlrange 硬违规为 0；
- official sole 最大穿透 `<=0.5 mm`；
- 两脚 stance speed p95 各 `<=0.10 m/s`；
- stance excursion `<=0.03 m`；
- unintended flight fraction `<=0.02`；
- contact timing error p95 `<=0.10 s`；
- keypoint error p95 `<=0.10 m`，且动作语义指标不得比输入恶化 `>10%`；
- root horizontal acceleration p95 `<=4 m/s²`；
- NLP defect/constraint residual 同时报告 normalized 与物理单位 max；
- **独立 fresh free-root replay 存活 `5.8/5.8 s`**，不是 prescribed-root、kinematic overwrite 或 solver 内部 rollout。

如果短前缀无法满足，不扫阈值。只允许报告是梯度、solver、表示、初始化还是硬不可行；交回审核者决定是否进入其他方法。

## 8. Phase 4：用动态 lunge 做 Faithful Any2Any 最小训练验证

只有 Phase 3 全门通过才可进入。

固定现有 faithful contract：

- fresh original source checkpoint；
- exact-S7/既有 B1–B5 live wiring；
- Gold held-out optimizer samples 始终为 0；
- zero-update → 1-update → 最多5-update；
- 每 update 独立回归并按预注册硬停；
- 不用 Phase47/49 的 rejected last 继续训练。

最低晋升门：

- lunge survival 从旧 `0.22 s` 提升到至少 `1.0 s`，最终目标为全时长或至少 `4×`；
- 不得靠站住不动取得生存，keypoint/root progress/contact 均须保持语义；
- native Gold survival 最坏下降 `<5%`，tracking 最坏恶化 `<10%`；
- KL `<0.02`，frozen hash 不变；
- 任一 update 目标 lunge 全指标无改善则硬停。

## 9. Phase 5：forward4 的同配置验证与四域闭环

仅在 lunge 链路通过后进行。

1. 对四条 forward reference 使用**同一套** DSMS 参数；不得逐 clip 调权重或接触时序。
2. 至少 `2/4` 产生通过第7.3节的动态 reference，才允许训练。
3. 使用既有 DC-PEFT 冻结协议：固定 checkpoint、seed 0、4 env×260 steps、确定性 rollout、同后处理门禁。
4. 四域与旧报告逐项 matched comparison。

业务成功门：

- ideal `>=2/4`；
- mean absolute progress-ratio error 明显低于旧 `2.303`；
- contact-cycle、root、foot gates 不得以牺牲动作前进语义换取；
- 达标即停止，不进入更重方法。

## 10. Phase 6：OmniTrack Stage I 小规模架构验证

启动条件：DSMS 给出至少一个合格动态 reference，或 DSMS 明确失败但形成了可学习、可量化的修复目标，并经审核者授权。

当前官方仓库只有 README/TODO 时，只能做“architecture-inspired implementation”，不得称官方复现。

范围限定：

- 3–10 条 train motions；
- 训练 Physical Motion Generation privileged policy；
- 输入完整 reference、精确仿真状态与域信息；
- rollout 保存 q/dq/root/contact/ctrl/torque/termination 与 source alignment；
- held-out 只评估，不进 optimizer；
- Stage I 输出必须通过与 DSMS 完全相同的第7.3节动态资格门。

失败触发：生成动作被“修平”、不再前进、语义误差恶化>10%，或 free-root 不合格。失败则停止，不启动 Stage II general tracker。

Dynamic-CoM 只能在 Stage I 基线成立后作为单变量 A/B；不能拿它替代动态 reference 可行性。

## 11. Phase 7：DDR readiness / 条件性 smoke

DDR 输入必须是匹配的人体视频/SMPL keypoints，而不是当前 native X2 forward4。执行前必须具备：

- 论文定义的关键点与坐标系；
- X2 keypoint correspondence；
- 官方或可审计的完整算法实现；
- CEM-MPC 控制变量、horizon、execute fraction、budget；
- 人体语义与物理门禁。

若完整代码仍未发布，默认结论为 `BLOCKED_BY_UPSTREAM_CODE`。允许做最小自研 smoke，但必须标为 DDR-inspired，不得与论文数值对比或称复现。

只有 DDR 输出通过第7.3节，并在同一 Faithful Any2Any 门中改善，才晋级。

## 12. Phase 8：DynaRetarget/SBTO readiness / 条件性 smoke

先验证 `Atarilab/sbto` 是否确为论文算法、commit、依赖和许可证。若无 LICENSE，只能内部研究，不能复制/发布。

考虑服务器算力，先做缩小版但必须诚实命名：

- 单 lunge 短前缀；
- 固定 candidate count、control points、horizon growth、CEM iteration 与 wall-time；
- warm-start 必须允许回改前缀，才可称 SBTO-inspired；
- 输出仍过第7.3节 independent free-root gate。

若需要论文级 1024 并行候选/数千万步而当前资源不具备，报告 `COMPUTE_BLOCKED`，不要用低预算失败否定论文方法。

## 13. 方法间切换与全局停止

| 当前方法 | 继续条件 | 换路条件 |
|---|---|---|
| DSMS | 上游复现、X2合同、lunge动态门逐级通过 | 上游不可复现、NLP无证书或free-root失败 |
| OmniTrack | Stage I生成reference通过统一动态门 | 修平动作、held失败或teacher学不会 |
| DDR | 上游代码/人体关键点/映射齐备 | 代码缺失或只有native X2输入 |
| SBTO | 算法与许可证明确、算力预算获准 | 代码/许可证/算力不满足 |

任何方法使 forward4 ideal `>=2/4` 且过冲下降，即停止后续更重路线。

若四路均不能让 ideal 超过 `1/4`，停止站3b投入，回到根速度分布、镜像/对称、控制命令与训练数据问题。

## 14. 版本管理与产物规范

- 开始前记录 `git status --short`，不得覆盖用户已有修改。
- 每个 phase 单独代码/测试/报告/manifest，不混写历史报告。
- 推荐命名：`phase_dsms_upstream`、`phase_dsms_x2_contract`、`phase_dsms_lunge`、`phase_omnitrack_stage1` 等。
- 小文件进入 Git；模型、轨迹、完整 solver log、physics trace 置于外部 artifact 目录，并在 manifest 中写绝对路径、size、SHA-256。
- 不自行 `git commit/push`，不自行上传百度网盘；等审核者统一归档。
- 长任务只按用户约定每30分钟检查一次；除非进程退出、硬门触发或资源异常，不做高频轮询。

## 15. 每阶段报告模板

每份 MD/JSON 必须包含：

1. `hypothesis`
2. `input_provenance` 与全部 SHA
3. `method_fidelity`：官方复现 / inspired / diagnostic
4. `preregistered_contract`
5. `intervention` 与 `control`
6. `compute_budget`、wall time、峰值内存/GPU/CPU
7. `results`：逐动作、逐门、逐失败原因
8. `invalid_runs`：pre-physics、infra、实现偏差单列
9. `decision`：PASS / REJECT / BLOCKED / NOT_PROMOTABLE
10. `evidence_boundary`
11. `next_single_experiment`
12. `artifact_manifest`

最终总报告必须对四种方法采用同一张表比较：实现忠实度、动态 reference 合格数、free-root、训练后 ideal 四域、语义保真、成本、许可证、可部署性。

## 16. 交付给审核者的最小集合

- Phase0 五项审计 MD/JSON/逐帧表；
- 每个实际尝试方法的 prereg、代码、测试、结果报告；
- 原 reference、动态 reference、contact schedule、独立 replay trace 的 SHA manifest；
- 若进入训练：source/final/best/rejected checkpoint 清单与四域结果；
- 一页结论：哪条方法值得继续、哪条被什么证据否定、下一项唯一实验是什么。

## 17. 成功层级，禁止混淆

- **算法复现成功**：上游例子跑通；不代表 X2 成功。
- **X2 动态 reference 成功**：通过第7.3节；不代表 policy 学会。
- **业务闭环成功**：重训后 forward4 ideal `>=2/4` 且过冲下降；只有这一层允许宣布当前问题被实质改善。

## 18. 一手资料

- Shooting for Contact paper: <https://arxiv.org/abs/2608.03116>
- Shooting for Contact project: <https://shooting-for-contact.github.io/>
- Shooting for Contact code: <https://github.com/sesteban951/shooting-for-contact>
- OmniTrack paper: <https://arxiv.org/abs/2602.23832>
- OmniTrack repository: <https://github.com/OmniTrack-Humanoid/OmniTrack>
- DDR paper: <https://arxiv.org/abs/2605.23762>
- DynaRetarget paper: <https://arxiv.org/abs/2602.06827>
- MuJoCo source: <https://github.com/google-deepmind/mujoco>

---

执行者收到本卡后，第一轮只完成 Phase0 与 Phase1 上游可执行性/许可证审计并交回，不自动启动 X2 NLP、RL 或其它方法。审核者确认后才进入下一阶段。
