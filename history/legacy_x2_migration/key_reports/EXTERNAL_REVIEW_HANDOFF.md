# SONIC/G1 → Agibot X2：独立旁观审查入口

更新时间：2026-07-15

项目根目录：

    /home/humanplus/x2_teleop_final/x2_sonic

这份文件是给另一个 Agent 的目录地图。请把现有结论当作待验证假设，不要默认接受“reference 与 policy 冲突”“LoRA 累积漂移”等判断。建议先只读审计，不启动训练、不控制真机。

## 1. 最终目标与边界

最终目标：

    11 IMU 智能服装
        → 人体/遥操作 reference
        → 继承 SONIC/G1 能力的 X2 policy
        → X2 31DOF PD 目标
        → X2 稳定、平衡、准确地跟踪人体动作

当前只解决中间最核心的一段：

    X2 reference
        → SONIC encoder + G1 checkpoint
        → dynamics-decoder LoRA + critic PPO
        → IsaacLab X2 物理仿真

固定边界：

- G1 source 是 29DOF body 版本，没有灵巧手。
- X2 是 31DOF，多出的 head yaw/pitch 锁定，不参与动作迁移。
- 不上真机；当前结果只允许在 IsaacLab/MuJoCo 中判断。
- 用户接受 X2 最终性能略低于 G1，但不能动作语义错误、开局倒地或明显不会走。
- 目前没有要求严格复刻动作时间轴，允许为平衡产生一定相位延迟。

## 2. 建议给审查 Agent 的任务

请独立回答五个问题：

1. 当前完整训练/评估 pipeline 在语义上是否真的等价于“G1 checkpoint + Any2Any 式 X2 dynamics adaptation”？
2. Stage76/77 的 root、接触和 liftoff 互换，是真正的动力学目标冲突，还是 reference 坐标、镜像、checkpoint 加载或评估语义错误？
3. Stage79/80 的倒退是训练目标导致的策略退化，还是训练/评估域不一致、LoRA 重复合并、action scale、forward 方向或 reset 错误？
4. 如果只允许做一个最小实验，哪个实验最能区分上述假设？
5. 是否存在比当前建议更简单、更忠实于 Any2Any 的下一步？

期望输出：

- 一句话总体判断；
- 已证实事实、可疑事实、无效证据三张清单；
- 根因假设按置信度排序，并为每条给出文件/trace 证据；
- 指出任何代码级 bug 或配置不一致；
- 给出一个最小、可证伪的下一实验；
- 明确是否需要新数据；不要笼统建议“更多数据/更长训练”。

## 3. 五分钟最短阅读路径

按顺序读：

1. 原始接续任务：
   /home/humanplus/.codex/attachments/eb5a4715-6e5f-4dfa-a38b-2d956acc5b07/pasted-text.txt
2. 阶段 0–15 总览：
   docs/reports/x2_sonic_stage0_to15_summary.md
3. 真正物理地基：
   docs/reports/x2_stage15_sim_foundation_and_upper_body_summary.md
4. Stage16–18 训练入口：
   docs/reports/x2_stage16_to18_longtrain_foundation_summary.md
5. Stage19 root/foot 重审：
   docs/reports/x2_stage19_root_foot_foundation_conclusion.md
6. 当前主 Agent 的 Stage0–80 复盘，仅作为一种观点：
   docs/reports/x2_stage0_to80_full_reconciliation_20260715.md
7. 更正后的 Stage74–80 完整动作报告：
   docs/reports/x2_stage74_long2000_honest_curve_episodefixed.md
   docs/reports/x2_stage75_long2000_honest_curve_episodefixed.md
   docs/reports/x2_stage76_long1000_honest_curve_episodefixed.md
   docs/reports/x2_stage77_long1000_honest_curve_episodefixed.md
   docs/reports/x2_stage78_long1000_honest_curve_episodefixed.md
   docs/reports/x2_stage79_it215_honest_episodefixed.md
   docs/reports/x2_stage80_d2_honest_curve_episodefixed.md

不要先读 docs/reports 全目录。它约 208 MB，含大量已被后续实验推翻或重复的短训报告。

## 4. 项目目录地图

| 目录 | 大小约 | 用途 | 审查优先级 |
| --- | ---: | --- | --- |
| sonic_x2_sandbox | 4.4 GB | SONIC/GR00T-WBC 源码及 X2 修改 | 最高 |
| scripts | 680 KB | 数据、训练、评估和 Stage 脚本 | 最高 |
| tools | — | 转换、审计、评估、视频工具 | 最高 |
| tests | 2.7 MB | mapping、LoRA、motion、Stage 配置测试 | 高 |
| motion_lib_x2 | 203 MB | X2 reference MotionLib | 高 |
| logs/ppo_dryrun | 103 GB | 训练 checkpoint、config、诊断、rollout trace | 高，按本文指定路径读 |
| docs/reports | 208 MB | 历史报告、JSON、视频 | 中，按阅读清单筛选 |
| data/processed | — | 实机 readonly canonical 数据和动力学窗口 | 中 |
| candidate_files | 296 KB | X2 模型/响应/配置候选 | 中 |
| checkpoints | 290 MB | Stage18 组合 seed 等固定资产 | 中 |
| backups | — | 少量 Stage22/23/67 前备份 | 仅用于回溯 |
| X2_fixed（兄弟目录） | 1.5 MB | 旧版固定测试 | 低 |

注意：

- 项目顶层不是可用的 Git 仓库，不能用顶层 git diff 还原全部历史。
- sonic_x2_sandbox 是嵌套 Git 仓库，HEAD 为官方 release 提交 bc38f6d，但当前有大量 modified/untracked X2 修改。
- 可先执行 git -C sonic_x2_sandbox status --short 和 git -C sonic_x2_sandbox diff。
- 不要 reset、checkout 或清理这些修改；它们是当前工程主体。
- logs/ppo_dryrun 约 103 GB，不要复制或全量扫描 checkpoint tensor。

## 5. 核心源码地图

### X2 机器人与动作接口

- X2 articulation/PD/default pose：
  sonic_x2_sandbox/gear_sonic/envs/manager_env/robots/x2.py
- G1/H2 官方对照：
  sonic_x2_sandbox/gear_sonic/envs/manager_env/robots/g1.py
  sonic_x2_sandbox/gear_sonic/envs/manager_env/robots/h2.py
- 动作、关节语义和 action router：
  sonic_x2_sandbox/gear_sonic/envs/wrapper/manager_env_wrapper.py
- 执行器逻辑：
  sonic_x2_sandbox/gear_sonic/envs/manager_env/mdp/actuators.py
- X2 实机响应模型：
  sonic_x2_sandbox/gear_sonic/envs/manager_env/x2_actuator_response.py

### Reference、接触与 episode

- Motion command、RSI、paired start、contact phase clock、auto-reset：
  sonic_x2_sandbox/gear_sonic/envs/manager_env/mdp/commands.py
- observation：
  sonic_x2_sandbox/gear_sonic/envs/manager_env/mdp/observations.py
- reward：
  sonic_x2_sandbox/gear_sonic/envs/manager_env/mdp/rewards.py
- termination：
  sonic_x2_sandbox/gear_sonic/envs/manager_env/mdp/terminations.py
- MotionLib、adaptive sampling：
  sonic_x2_sandbox/gear_sonic/utils/motion_lib/motion_lib_base.py
- contact label 工具：
  sonic_x2_sandbox/gear_sonic/utils/motion_lib/contact_labels.py

### Any2Any/LoRA/PPO

- PPO rollout、decoder replay、retention、学习率：
  sonic_x2_sandbox/gear_sonic/trl/trainer/ppo_trainer.py
- actor/critic 网络：
  sonic_x2_sandbox/gear_sonic/trl/modules/actor_critic_modules.py
  sonic_x2_sandbox/gear_sonic/trl/modules/universal_token_modules.py
- LoRA checkpoint 合并/恢复：
  sonic_x2_sandbox/gear_sonic/trl/utils/any2any_lora_checkpoint.py
- X2 checkpoint 语义映射：
  sonic_x2_sandbox/gear_sonic/trl/utils/x2_checkpoint_semantic_mapping.py
- 主 X2 experiment config：
  sonic_x2_sandbox/gear_sonic/config/exp/manager/universal_token/all_modes/sonic_x2.yaml

建议重点核查：

- checkpoint 中已有 LoRA 是保留、合并还是再次叠加；
- actor output 的 31DOF 顺序与 X2 articulation 顺序；
- absolute target 与 default-pose-relative residual 是否混用；
- head lock 是否同时作用于 observation、action、reward；
- train/eval 的 action scale、PD、actuator response、contact labels 是否完全一致。

## 6. 数据与 reference 地图

### 通用人体动作与 X2 重定向

- 363 条 AMASS curriculum：
  motion_lib_x2/amass_x2_curriculum_v4_363
- 低 root/foot 冲突子集：
  motion_lib_x2/amass_x2_curriculum_v6_rootfoot_clean67_split
- PHUMA clean291：
  motion_lib_x2/phuma_x2_hybrid1200_clean291_50fps
- 严格基础动作：
  motion_lib_x2/phuma_x2_strict89_foundation_50fps

### 当前步态关键 reference

- 四条 corrected official true-forward：
  motion_lib_x2/stage72_official_true_forward4_v1
- Stage79/80 使用的 D 与 D-mirror：
  motion_lib_x2/stage79_official_true_forward_d2_v1

建议独立检查 D/B 及 mirror：

- raw root XY 起点/终点与 heading；
- forward 的正方向定义；
- 左右 joint、body、contact label 的镜像对应；
- root translation 是否也正确镜像；
- reference foot contact、足端高度、root/COM 是否物理一致；
- reference 在 MuJoCo FK 与 MotionLib FK 中是否相同。

相关工具：

- tools/build_x2_official_gait_motionlib.py
- tools/mirror_x2_motionlib.py
- tools/audit_x2_reference_contact_labels.py
- tools/audit_x2_reference_root_foot_consistency.py
- tools/analyze_x2_motion_cache_kinematics.py
- tools/replay_x2_motion_cache_mujoco.py

### 实机 readonly 数据

- data/processed/x2_real_readonly_session02_canonical_31dof.npz
- data/processed/x2_real_readonly_session03_canonical_31dof.npz
- data/processed/x2_real_readonly_session04_canonical_31dof.npz
- data/processed/x2_real_readonly_session03_session04_canonical_31dof.npz
- candidate_files/config/x2_actuator_response_session03_session04.json

原始 Session03：

    /home/humanplus/humanoid-GPT/A/sonic_release/20260710_session03_full_x2_records.tar.gz

Session04 描述：

    /home/humanplus/humanoid-GPT/A/sonic_release/20260711_session04_dataset_notes.md

已知边界：

- joint q/dq、joint command、stiffness/damping、torso/chest IMU 可用；
- base odometry 不可靠；
- 没有已确认的真实 foot wrench、COP、COM、centroidal momentum；
- 这些日志主要用于执行器响应和动作时序，不应被当成动力学真值。

## 7. 关键 checkpoint 与训练 run

### 上半身和长训入口

- Stage17 upper-body warm actor：
  logs/ppo_dryrun/x2_stage17_upper_oracle_warm200_cont_v1/model_step_000100.pt
- Stage18 组合 seed：
  checkpoints/x2_stage18_warm200_actor_faithful1000_critic_seed.pt

### Stage74–80 主线

| 阶段 | run 目录 | 关键 checkpoint | 主要意图 |
| --- | --- | --- | --- |
| 74 | logs/ppo_dryrun/x2_stage74_stage73c100_trueforward_long2000_v1 | model_step_002000.pt | 极弱 LR + retention 长训 |
| 75 | logs/ppo_dryrun/x2_stage75_trueforward_faithful_capacity_long2000_v1 | model_step_002000.pt | 完整 decoder Any2Any 容量 |
| 76 | logs/ppo_dryrun/x2_stage76_contact_rescue_long1000_v1 | model_step_001000.pt | 连续接触/载荷转移 |
| 77 | logs/ppo_dryrun/x2_stage77_dense_liftoff_long1000_v1 | model_step_000750.pt | 稠密摆动脚高度 |
| 78 | logs/ppo_dryrun/x2_stage78_root020_recovery_long1000_v1 | model_step_000500.pt | 提高 root tracking |
| 79 | logs/ppo_dryrun/x2_stage79_d2_identifiability_long1000_v1 | last.pt，约 it215 | D/D-mirror 数据可辨识对照 |
| 80 | logs/ppo_dryrun/x2_stage80_mixedstart_d2_long1000_v1 | model_step_000100.pt 到 001000.pt | fixed/random mixed start |

每个 run 内优先读：

- config.yaml：最终解析后的真实配置；
- training_diagnostics.jsonl：逐 iteration 训练统计；
- policy_partial_load_report.json；
- policy_lora_merge_report.json；
- any2any_lora_trainable_report.json；
- checkpoint 只在需要比较 tensor 时加载。

不要只根据启动脚本推断真实配置；以 run 内 config.yaml 和 console.log 为准。

## 8. Stage74–80 启动与评估脚本

训练：

- scripts/run_x2_stage73_a2a_priority_continuation.sh
- scripts/run_x2_stage75_trueforward_faithful_capacity.sh
- scripts/run_x2_stage76_contact_rescue_continuation.sh
- scripts/run_x2_stage77_dense_liftoff_continuation.sh
- scripts/run_x2_stage78_root_progress_recovery.sh
- scripts/run_x2_stage79_d2_identifiability_control.sh
- scripts/run_x2_stage80_mixed_start_d2.sh

固定物理评估：

- scripts/run_x2_stage19_foot_panel.sh
- scripts/eval_x2_stage71_a2a_root010_curve.sh
- scripts/eval_x2_stage80_d2_checkpoint.sh
- scripts/eval_x2_stage80_d2_curve.sh

评估器：

- tools/analyze_x2_locomotion_progress.py
- tools/summarize_x2_structure_rollout_traces.py
- tools/analyze_x2_gait_failure_phases.py
- tools/render_x2_rollout_trace_mujoco.py

## 9. 重要的评估纠错

2026-07-15 修复了一个 episode 回卷问题：

- 一条动作约 240/241 帧；
- 旧评估会继续把 260-step rollout 中动作结束后的 auto-reset 尾帧算进同一 episode；
- D reference 的真实水平位移约 0.129 m，旧算法可能错误缩为约 0.023 m；
- 旧 progress ratio、direction 和部分 root RMSE 因此无效；
- structure summarizer 也已改为只统计第一 episode；
- Stage80 的最小 reference 位移门已显式设为 0.10 m。

只信任文件名含 episodefixed 的 Stage74–80 locomotion/structure 报告。

相关测试：

- tests/test_x2_locomotion_progress_analyzer.py
- tests/test_summarize_x2_structure_rollout_traces.py
- tests/test_x2_stage80_mixed_start_config.py

当前 11 个相关测试通过。独立审查仍应验证：

- raw reference 是否确实等于 robot root + anchor residual；
- done 行是否应该包含在 episode 内；
- motion timeout 与物理 fall 是否正确区分；
- 0.10 m 位移阈值是否适合该面板；
- progress 的 forward sign 是否和 reference heading 一致。

## 10. 更正后的关键现象，不是要求接受的结论

统一门禁要求 stable、world progress、contact timing 和 foot tracking 同时通过：

| checkpoint | pass | stable | progress | contact | foot |
| --- | ---: | ---: | ---: | ---: | ---: |
| Stage75-s2000 | 0/4 | 4/4 | 1/4 | 0/4 | 4/4 |
| Stage76-s1000 | 1/4 | 4/4 | 3/4 | 2/4 | 4/4 |
| Stage77-s750 | 1/4 | 4/4 | 1/4 | 4/4 | 4/4 |
| Stage78-s500 | 1/4 | 4/4 | 1/4 | 3/4 | 4/4 |
| Stage79-it215 | 0/2 | 0/2 | 0/2 | 0/2 | 0/2 |
| Stage80-s1000 | 0/2 | 0/2 | 0/2 | 0/2 | 0/2 |

主 Agent 当前解释：

- Stage75 学到低 root error，但通过双脚贴地取巧；
- Stage76 改善载荷转移；
- Stage77 改善真实 liftoff，但破坏部分 root progress；
- Stage78 只提高 root reward 没有统一两者；
- Stage79/80 累积破坏闭环。

请旁观 Agent优先尝试推翻这个解释，而不是在它上面继续调参。

## 11. 最值得挑战的替代假设

### A. forward/mirror 或世界坐标有错

现象：用户视频多次观察到“都在倒着走”，Stage80 更正后也显示负 progress ratio。

需要查：

- reference heading 与 root XY 方向；
- robot root + residual 的重建公式；
- mirror 是否反转了不该反转的轴；
- evaluation camera 方向是否造成视觉误解；
- policy 的 egocentric command 是否根本不需要 world translation。

### B. checkpoint/LoRA 加载不一致

需要查：

- Stage78 seed 进入 Stage79/80 时是否先 merge 再重新注入 LoRA；
- eval 的 preserve-LoRA false 是否与训练保存语义一致；
- base weight 是否重复吸收 LoRA；
- critic 与 actor 是否来自相同阶段；
- checkpoint tensor 映射是否跨 29/31DOF 静默错位。

### C. train/eval 不同域

需要逐项比较：

- actuator response；
- action scale；
- PD gains；
- contact label mode；
- start frame；
- deterministic action/noise；
- termination threshold；
- encoder mode；
- MotionLib cache；
- default pose 和 locked head。

### D. 24-step PPO 统计掩盖 240-step失败

需要查：

- fixed-start 与 RSI-start 是否应分开统计；
- near-end random starts 是否放大 timeout；
- adaptive sampler 如何定义 failure；
- termination 是否有显式负奖励；
- value bootstrap 是否能传播 motion-end 成败；
- per-iteration KL 是否只约束前一批次，而不保护 seed。

### E. reference 本身缺乏动力学可执行性

需要查：

- Stage76 的 progress3/4 是否说明至少部分 reference 可执行；
- 失败动作是否集中在同一接触相位；
- root/COM 是否应该允许受限偏离；
- 是否应先做 dynamics-aware/kinodynamic retargeting；
- 现有 X2 official controller walk 是否能作为 teacher/reference，而不需要真实 COP/GRF。

## 12. 论文与官方资料

- Any2Any：
  /home/humanplus/humanoid-GPT/A/sonic_release/2601.09361v3.pdf
- XHugWBC：
  /home/humanplus/humanoid-GPT/A/sonic_release/XHugWBC.pdf
- X2 用户手册：
  /home/humanplus/humanoid-GPT/A/sonic_release/x2_ultra_user_guide_t2.5_v0.8.1.pdf
- SONIC 官方 new embodiment 训练说明：
  /home/humanplus/.codex/attachments/68bc233f-37a0-48f7-9edb-4c411513e0c8/pasted-text.txt
- 原始 SONIC/GR00T-WBC 工作区：
  /home/humanplus/sonic_lty/GR00T-WholeBodyControl

论文审查重点不是泛泛总结，而是核对：

- Any2Any 实际训练哪些 actor/critic 模块；
- 8000 iteration 使用的动作量、环境数、RSI、reward 和 checkpoint 选择；
- 是否存在 seed-relative behavior retention；
- 新 embodiment 是否先要求零动作站立、reference oracle、分组 actuator；
- X2/G1 结构差异是否超出 Any2Any 实验覆盖。

## 13. 可直接复制给另一个 Agent 的提示词

    你是独立审查者。请只读审计
    /home/humanplus/x2_teleop_final/x2_sonic
    的 SONIC/G1 → Agibot X2 迁移工程。

    先完整阅读 EXTERNAL_REVIEW_HANDOFF.md，再按其中的最短阅读路径和关键源码路径核对。
    不要默认接受主 Agent 在
    docs/reports/x2_stage0_to80_full_reconciliation_20260715.md
    中的根因判断；请优先尝试推翻它。

    当前禁止启动新训练、删除文件或控制真机。允许读取 checkpoint/config/trace，
    运行 CPU 级测试和离线分析。若确需 GPU 物理复评，请先给出必要性和最小命令。

    最终请输出：
    1. 一句话总体判断；
    2. 已证实、可疑、无效证据；
    3. train/eval/reference/checkpoint 语义是否一致；
    4. 根因假设按置信度排序；
    5. 一个信息增益最高的最小实验；
    6. 是否需要新数据；
    7. 哪个现有 checkpoint 最值得保留，哪个不能继续使用。

## 14. 安全与资源提示

- 不上真机。
- 不要运行 rm、git reset、git checkout。
- 不要覆盖现有 checkpoint、trace 或 episodefixed 报告。
- 不要全量复制 103 GB 的 logs/ppo_dryrun。
- CPU 静态测试环境：
  /home/humanplus/anaconda3/envs/x2-sonic-isaaclab/bin/python
- IsaacLab GPU 评估可能占用显存，启动前先检查进程。
- 顶层没有统一 Git 历史；任何代码修改前先单独备份或创建明确 patch。
