# X2 原生运动与全身控制工作路线卡（新机执行版）

> 日期：2026-08-12
> 执行机：新 RTX 3090 Linux 工作站（用户 `yu`，`/home/yu`，软链 `/home/humanplus → /home/yu`）
> 环境基线：conda `x2-sonic-isaaclab`，已通过迁移验收（见附录 A）
> 本卡由迁移验收后整理，给接续 agent 使用。路线卡原作者意图保留，仅补充本机已落地的真实路径/资产与备份纪律。
> **分工：本机承担全部工作**——任务一至六 + BFM-Zero 支线全部在本 3090 工作站顺序推进，不依赖外部服务器。

---

## 0. 起点状态（迁移已完成，可直接复用，不要重建）

迁移任务已 100% 验收通过。**接续 agent 不应重新下载/解压/搭环境**，直接用以下已就位资产：

- 主仓库：`/home/yu/projects/ZHY/CWI_CrossEmbodiment_Sim`，Git HEAD = `ac4d34f97a461b3e5c61bbec754ddef5f89449da`，origin = `git@github.com:z2964141870-debug/HUMANPLUS_X2_PLUS.git`。工作区非 clean（183 行 Phase40 WIP + Cycle20-40 恢复资产，属正常未提交状态）。
- Phase40 live-zero **已闭合**：`src/x2_privileged_generator_live.py` 的 `reset_from_native_generator_seed` / `finalize_native_generator_reset` 已实现，`tests/test_phase40_privileged_generator_live_wiring.py` 4/4 通过。整组验收 8/8 通过。
- conda 环境 `x2-sonic-isaaclab`：Python 3.11.15 / Isaac Sim 5.1.0.0 / IsaacLab `37ddf62…` / PyTorch 2.8.0+cu128 / MuJoCo 3.3.7 / mink 1.2.0（--no-deps）。GPU RTX 3090 可用。
- Docker：`x2-aimdk-humble:1.0`、`g1-deploy-dev:latest` 已 load。
- 官方 SDK：`/home/yu/x2_migration_20260812/downloads/official/aimdk-x2-v1.0.0-official.zip`（含 X2 URDF / scene.xml / x2.xml / onnx predictor / kuailechongbai.onnx）。
- 已下载 checkpoint：`/home/yu/x2_migration_20260812/downloads/checkpoints/stage152_B_dual_equal_split_init.pt`、`stage152_B_materialized_v1.pt`。
- PHUMA 动作库：`/home/yu/x2_teleop_final/x2_sonic/motion_lib_x2/phuma_x2_hybrid1200_clean291_50fps/`（292 pkl）+ broad227/medium169/strict89 四档。
- forward4 seed：`/home/yu/x2_migration_20260812/downloads/seed/x2_forward4_dynamics_feasibility_seed_v1.tar.gz`（已解压到 `$HOME`）。
- 迁移验收报告：`/home/yu/x2_migration_20260812/X2_MIGRATION_ACCEPTANCE_REPORT_20260812.md`。

**老机遗留的关键裁决**（`reports/official_x2/stage264_longtrain_readiness_task50.md`）：
- 标称起步/直行/左右转/停车 **24/24**；上肢固定/慢摆/快摆直行 9/9。
- 未通过：PD 0.9/1.0/1.2 × 固定/快摆臂 14/18（失败以航向越界+停车失稳为主）。
- 老机因"重启后 NVIDIA 设备节点缺失"无法跑 PPO 长训——这正是迁移到新 3090 的原因。**新机 GPU 已恢复，长训能力可用。**

---

## 总目标

基于官方 X2 MuJoCo：

1. 保住 Stage264 已有的起步、行走、转向和停车能力；
2. 解决机器人持续后仰前行；
3. 训练 X2 原生鲁棒 locomotion policy；
4. 重新构建动力学可行的 X2 动作数据；
5. 最终实现稳定的 whole-body tracking，而不只是速度型走路。

---

## 任务一：冻结当前基线

### 工作

- 固定 Stage264 checkpoint、ONNX 和训练配置；
- 固定官方 X2 模型、逐关节 PD、action scale；
- 固定现有标称域与扰动域测试脚本；
- 补充侧视角视频和带符号的 root pitch 指标。

### 本机资产定位（重要）

- Stage264 模型族：`/home/yu/projects/ZHY/x2_official_rl_deploy_v1/models/`，含 `stage298_s2647_transition_actor.onnx`、`stage304_s2647_locomotion_future_actor.onnx`、`stage212_s2649_actor.onnx`、`stage306_s2647_transition_head_actor.onnx` 等。**冻结前先确认哪个 onnx 是 Stage264 的标称后端**（readiness 报告指向的"标称后端已可用"的那一个）。
- 裁决文档：`reports/official_x2/stage264_longtrain_readiness_task50.{md,json}`，已记录 24/24 与 14/18 的明细。
- 官方 MuJoCo 回放/测试脚本：`tools/official_x2/replay_official_trace_direct_mujoco.py`、`tools/official_x2/audit_official_x2_event_replay_phase14.py`、`tests/test_official_x2_event_replay_phase14.py`、`tests/test_x2_native_event_replay_phase15.py`。
- 官方场景：解压 aimdk zip 得 `x2_rl_deploy_mujoco/configuration/robot/lx2501_3_t2d5/`（scene.xml / x2.xml）。

### 验收

- 标称域保持 `24/24`；
- 能完成起步、直行、左右转、停车；
- 记录行走期间：signed root/pelvis pitch mean；pitch p05/p95；速度、航向、足滑和接触时序。

### 产物

`stage264_frozen_baseline/`

---

## 任务二：解决后仰前行

### 推荐方法

从 Stage264 权重继续训练，不从零开始。第一轮暂时固定或弱化上肢扰动。

新增训练约束：
- 带方向的 root/pelvis pitch 奖励；
- CoM 与支撑区域关系；
- 髋、膝、踝 pitch 协同；
- 足底接触与足滑；
- 速度、航向和动作平滑性。

**不要直接缩小腰部 pitch，也不要强制机器人始终完全竖直。**

### 本机训练入口线索

- 训练脚本：`scripts/run_dcpeft_stage152.sh`（PEFT/LoRA 训练入口，含 `FAITHFUL_WBT29_ENV_TARGET=x2_faithful_live_actions_phase46.FaithfulWBT29TrackingEnvCfg`）。
- Phase40 live 接线：`scripts/run_phase40_privileged_generator_live_zero.sh` + `scripts/train_agent_trl_privileged_generator.py`（已通过 import 接线测试，`ITERS=0`/`SAVE_FREQUENCY=-1` 的 live-zero smoke）。
- 训练前务必先跑 Phase40 live-zero smoke 确认接线在新机不退化。

### 对照实验（只做三组）

- A：Stage264 原始基线；
- B：增加 signed pitch reward；
- C：signed pitch + CoM/support reward。

### 验收

- 后仰角相对基线明显下降；
- 不是通过降低行走速度换来的；
- 标称域基础门禁不低于 `23/24`，最终恢复到 `24/24`；
- 不新增膝部僵硬、脚尖拖地或高频摇晃；
- 停车仍稳定。

### 停止条件

如果姿态改善但连续两轮都导致接触或停车明显退化，停止调单一姿态奖励，转向 CoM—接触联合训练。

---

## 任务三：X2 原生鲁棒 locomotion

### 工作

逐步加入随机化：1. 上肢慢速摆动；2. 上肢快速摆动；3. PD/stiffness 偏差；4. 质量、CoM、摩擦偏差；5. 控制延迟与观测噪声；6. 外部推力。

课程训练：`站立 → 慢速直行 → 变速 → 转向 → 停车 → 扰动恢复`

### 验收

- 标称域 `24/24`；
- 关键扰动域稳定通过（含老机未过的 PD 0.9/1.0/1.2 × 固定/快摆臂 14/18 → 目标 18/18）；
- 后仰不重新出现；
- 不以明显降低速度或动作幅度换取通过；
- ONNX 在官方闭环中表现一致。

### 产物

`x2_native_locomotion_v1`（日常遥操作保底后端）

---

## 任务四：重建 X2 动作数据

### 数据来源（本机已就位）

- AMASS：**本机未下载**，需 agent 自行获取（如需）；
- PHUMA：`/home/yu/x2_teleop_final/x2_sonic/motion_lib_x2/phuma_x2_*`（hybrid1200/broad227/medium169/strict89 四档，292+ pkl）；
- SONIC/BONES-seed：`x2_sonic` 资产在 `/home/yu/x2_teleop_final/x2_sonic/`。

### 工作流程

`人体动作 → X2运动学重定向 → 接触修正 → 动力学可行化 → 官方MuJoCo回放 → Silver数据筛选`

重定向工具：`/home/yu/projects/Human+智能服装动作捕捉系统/general_motion_retargeting/`（含 `motion_retarget.py`、`kinematics_model.py`、`ik_configs/`）。注意 `general_motion_retargeting` 的 import 根是这里，不是 `humanoid-GPT/Humanoid-GPT`（后者只有历史报告）。

### 重点检查

root 高度/朝向；足底接触时序；CoM/支撑；关节限位/速度/力矩；脚部穿透与滑动；X2 能否从该动作稳定退出。

### 数据分级

- A：稳定自然，可直接训练；
- B：可执行但需动力学修正；
- C：倒地/滑动/严重饱和，禁止进入训练集。

### 验收

先完成小型高质量集合：行走与转向、下蹲起身、弓步、踢腿、拳击/快速摆臂。**不要一开始处理完整 AMASS。**

---

## 任务五：Whole-body tracking teacher

### 工作

使用 X2 Silver 数据从小动作集开始训练：
- 第一阶段：下蹲、弓步等准静态动作；
- 第二阶段：踢腿和快速重心转移；
- 第三阶段：连续全身表演动作。

训练输入：reference motion；proprioception history；gait/contact phase；root 和未来动作信息。

### 验收

- 不仅跟踪关节，还要保持 root、接触和稳定性；
- 弓步和踢腿的下肢表达确实出现；
- 动作结束后能回到稳定站立；
- 在官方 MuJoCo 中通过动作级测试，而非只看平均 tracking error。

### 产物

`x2_wholebody_teacher_v1`

---

## 任务六：统一部署

双后端：日常遥操作 `x2_native_locomotion_v1`；表演动作 `x2_wholebody_teacher_v1`；异常时回到 stand/recovery backend。

必须实现：状态对齐；平滑进入退出；command bounds；风险检测；失败后恢复与操作者重新接管。

**统一 student 蒸馏放在双后端稳定以后，不提前进行。**

---

## BFM-Zero 支线

**仅在任务三完成后启动。** 第一步只做：
- 跑通官方 G1 最小训练；
- 机器人配置替换为 X2；
- 验证 observation/action/PD/ONNX 闭环；
- 用小规模动作集测试单张 3090 的训练速度和内存需求。

如果小规模 X2 训练无法超过现有 Stage264，不扩大训练规模。

---

## 当前执行顺序

```text
冻结Stage264
   ↓
解决后仰
   ↓
X2原生鲁棒locomotion
   ↓
重新重定向并筛选Silver数据
   ↓
Whole-body tracking teacher
   ↓
安全交权/专家蒸馏
   ↓
真机验证与第三本体
   ↓
BFM-Zero 支线（任务三完成后启动）
```

**全部由本机（单张 RTX 3090）承担**，不依赖外部服务器。

### 单卡 3090 资源约束（重要）

- 显存 24 GB。IsaacLab PPO 的 `NUM_ENVS` 必须按显存调整：locomotion 类建议 4096 起，whole-body tracking 带 reference motion + history 观测时显存占用更高，可能需降到 2048–1024，**以不 OOM 为准**；不得为凑通过数而盲目堆环境。
- 训练速度预期：单卡 3090 远慢于多卡服务器。长训前先用小规模（少量 env / 少量 iter）跑通 smoke，确认吞吐与显存，再决定是否扩大；**小规模跑不通或不超过 Stage264 就不扩大**（BFM-Zero 支线同此原则）。
- 训练 run 写清 `NUM_ENVS`、`ITERS`、batch、显存占用、单 iter 时长到 EXPERIMENTS.md，便于后续判断是否触顶单卡上限。
- 任何 OOM 优先降 `NUM_ENVS` / 观测维度 / 网络规模，不要无原则降精度或关随机化换通过。

---

## ⚠️ 备份纪律（百度网盘 + GitHub 双备份，必须遵守）

迁移任务卡原规则第 5 条要求"每 10 个实质任务一次 Git + 百度网盘双备份"。接续 agent 必须延续：

### 触发节奏

- 每完成 **10 个实质任务**（一次训练 run、一次数据集产出、一次门禁验收 都算一个实质任务）做一次双备份；
- 任何**产物级 checkpoint / onnx / Silver 数据集**产出时，**立即**做一次双备份，不等凑满 10 个；
- 任何**门禁结果反转**（通过→失败 或 解锁→回退）时立即备份并记录。

### GitHub 备份

- 仓库：`git@github.com:z2964141870-debug/HUMANPLUS_X2_PLUS.git`（私密库，SSH 密钥已在 `/home/yu/.ssh/id_ed25519`）。
- 备份内容：代码、脚本、report JSON/MD、manifest。**大文件（checkpoint/onnx/数据集）不进 Git**，走百度。
- 规则：**禁止 `git push --force`、`git reset --hard` 后再 push、`git clean -fd`**。提交前确认 HEAD 仍在冻结基线 `ac4d34f…` 的后代线上。工作区 WIP 用普通 commit，不要覆盖历史。
- push 前先 `git fetch` + 确认不与远程冲突。

### 百度网盘备份

- 工具：`bdpan`（已装于 `/home/yu/.local/bin/bdpan`，已登录，Token 有效约 29 天）。
- 远程根目录：`HUMAN+/HUMANPLUS_X2_PLUS/`（命令中路径不带 `/apps/bdpan/` 前缀）。
- 建议路径约定：`HUMAN+/HUMANPLUS_X2_PLUS/<日期>/<阶段>/`，例如 `2026-08-15/task2_backward_pitch/stage264_signed_pitch_run3.pt`。
- 上传后**必须**记录字节数 + SHA-256 到一个 manifest（沿用迁移清单格式），并在本地留一份 sidecar `.sha256`。
- **硬规则**（继承迁移任务卡）：
  1. 不读取/打印/上传 `~/.config/bdpan/config.json`；
  2. 不向用户索取百度密码/Cookie/Token；Token 过期时让用户自己用 `bdpan login --device-code` 重新授权；
  3. 授权码只能经终端 stdin（`read -s`），禁进命令参数/日志/聊天记录；
  4. 不删除失败证据；上传失败的包保留到确认远程校验通过后才能清理本地。

### 备份清单每次必含

1. 产物文件 + SHA-256；
2. 训练配置（hydra overrides / 脚本入参）；
3. 门禁结果 JSON（标称/扰动域通过数）；
4. EXPERIMENTS.md 的新增账本行。

---

## 停止条件（继承 + 补充）

出现以下任一立即停止并报告，不自行绕过：
- 标称域从 24/24 跌破 23/24 且无法在一轮内恢复；
- Git HEAD 偏离冻结基线后代线；
- Docker 镜像/环境版本漂移（不得无约束 `pip install -U`，不得让 IsaacLab 的 torch 2.7 覆盖项目锁定的 2.8.0+cu128）；
- checkpoint/onnx SHA 与产物清单不符；
- 百度上传/下载 SHA 不符或路径异常；
- 需要用户提供密码/Cookie/Token/私钥；
- 训练出现持续后仰恶化、接触退化、停车失稳且连续两轮调参无效。

---

## 附录 A：迁移验收快照（已完成，勿重做）

- 22/22 下载 SHA 通过；Bootstrap SHA `a9a3e630…`、PHUMA SHA `42152318…` 通过。
- Git HEAD = `ac4d34f97a461b3e5c61bbec754ddef5f89449da`；origin = GitHub 私密库。
- Docker：`x2-aimdk-humble:1.0`（`52406d45fa5c`）、`g1-deploy-dev:latest`（`4cb3e60d89ee`）。
- 版本：Python 3.11.15 / Isaac Sim 5.1.0.0 / IsaacLab `37ddf62…`（isaaclab 0.54.2 等 6 扩展）/ PyTorch 2.8.0+cu128 / MuJoCo 3.3.7 / onnx 1.18.0 / onnxruntime-gpu 1.26.0 / gymnasium 1.2.1 / rsl-rl-lib 3.0.1 / mink 1.2.0 / warp 1.14.0。
- 静态测试 8/8 通过（Phase38/39/40/IO）。
- 6 个必须目录全部就位（见第 0 节）。
- **未执行训练/optimizer/官方物理 A/B/参数扫描**——迁移阶段合规。

## 附录 B：环境坑点提醒（迁移中踩过）

- `isaaclab.sh --install` 会强制装 torch==2.7.0 覆盖 2.8.0+cu128。**不要重跑 `isaaclab.sh --install`**；如需重装扩展用 `pip install --no-deps --editable source/<ext>`，再 `pip install --force-reinstall --no-deps torch==2.8.0+cu128`（带依赖会同步升级 nvidia-nccl 到 2.27.3 修复 `undefined symbol ncclCommWindowRegister`）。
- GitHub HTTPS 不通，git 操作走 SSH（`git@github.com:...`）。
- isaacsim import 需 `OMNI_KIT_ACCEPT_EULA=YES` 环境变量（非交互）。
- PYTHONPATH 必须含：`$REPO:$REPO/src:$REPO/tools:$HOME/x2_teleop_final/x2_sonic:$HOME/projects/Human+智能服装动作捕捉系统`。
- `official_x2` 在 `$REPO/tools/official_x2/`，不是 vendor SDK；`general_motion_retargeting` 在 Human+ 项目树。
