# X2 动力学可行化开发环境与迁移文件总清单

> 版本：2026-08-10
> 适用目标：在另一台 Linux 工作站/服务器复现当前 X2 四域基线，并继续 OmniTrack、SBTO、DDR、DSMS 式动力学可行化。
> 大文件根目录：百度网盘 `/apps/bdpan/HUMAN+/HUMANPLUS_X2_PLUS/`；下文 `bdpan` 命令中的路径不写 `/apps/bdpan/`。

## 1. 结论与最小迁移集合

新机器至少需要以下四层，缺一层都不是完整复现：

1. Git 代码：`https://github.com/z2964141870-debug/HUMANPLUS_X2_PLUS.git`。
2. Host 仿真/训练环境：Ubuntu、NVIDIA 驱动、Python 3.11、Isaac Sim 5.1、固定 IsaacLab、PyTorch CUDA 12.8 和 `requirements.txt`。
3. X2 官方域：AimDK v1.0 SDK/MJCF/ROS workspace，以及已导出的 Docker 镜像分片。
4. 项目数据：精确 Stage152 checkpoint、forward4 reference、原始 X2 31DoF NPZ、四域 trace、旧 X2 源码增量与资产。

当前 `forward4` 的上游是 `x2_real_readonly_session04_canonical_31dof.npz`，属于 **X2 native/readonly-derived reference**，不是普通 AMASS/SMPL→GMR 输出。方法 1/2 可以直接物理化它；方法 3 DDR 若从人体关键点开始，必须另选 matched SMPL/关键点动作。

## 2. 硬件与系统环境

推荐配置：

- Ubuntu 22.04 LTS，x86_64；
- NVIDIA 专有驱动，能够运行 CUDA 12.8 的 PyTorch wheel；
- 已验证 GPU 为 RTX 5060 8 GB；显存更小时必须降低 `NUM_ENVS`；
- 建议至少 100 GB 可用磁盘；若同时保留下载包、Docker 分片、解压镜像和实验结果，建议 200 GB；
- Docker Engine、NVIDIA Container Toolkit；
- headless 运行仍需 Vulkan-capable NVIDIA driver。

Ubuntu 22.04 主机包：

```bash
sudo apt update
sudo apt install -y \
  git git-lfs curl wget jq rsync unzip zip tar \
  build-essential gcc g++ cmake ninja-build pkg-config patchelf \
  ffmpeg xvfb xauth \
  libgl1 libglib2.0-0 libx11-6 libxext6 libxrender1 libsm6 \
  libxi6 libxrandr2 libxinerama1 libxcursor1 libvulkan1 vulkan-tools \
  mesa-utils
git lfs install
```

## 3. Python、Isaac Sim 与 MuJoCo

完整 pip 锁文件位于仓库根目录 `requirements.txt`。安装顺序不可交换：

```bash
conda create -n x2-sonic-isaaclab python=3.11.15 pip=26.1.2 -y
conda activate x2-sonic-isaaclab

python -m pip install "isaacsim[all,extscache]==5.1.0.0" \
  --extra-index-url https://pypi.nvidia.com

git clone https://github.com/isaac-sim/IsaacLab.git
cd IsaacLab
git checkout 37ddf626871758333d6ed89cf64ad702aef127d0
./isaaclab.sh --install

cd "$X2_WORKSPACE/CWI_CrossEmbodiment_Sim"
python -m pip install -r requirements.txt
python -m pip install --no-deps "mink==1.2.0"
```

关键锁定版本：

| 组件 | 版本/提交 |
|---|---|
| Python | 3.11.15 |
| Isaac Sim | 5.1.0.0 |
| IsaacLab | `37ddf626871758333d6ed89cf64ad702aef127d0` |
| PyTorch | 2.8.0+cu128 |
| torchvision / torchaudio | 0.23.0+cu128 / 2.8.0+cu128 |
| MuJoCo | 3.3.7 |
| ONNX / ONNX Runtime GPU | 1.18.0 / 1.26.0 |
| rsl-rl-lib | 3.0.1 |
| Gymnasium | 1.2.1 |
| Mink | 1.2.0，使用 `--no-deps` 安装 |

上游元数据有两处真实冲突：Isaac Sim 5.1 声明 torch 2.7，而当前工作环境和权重使用 torch 2.8+cu128；Mink 1.2 声明 MuJoCo ≥3.8.1，而官方 X2 域锁定 3.3.7。因此不要执行无约束的 `pip install -U`，也不要让 pip 自动重解整个环境。

## 4. 工作区目录

推荐布局：

```text
$X2_WORKSPACE/CWI_CrossEmbodiment_Sim
$X2_WORKSPACE/x2_official_rl_deploy_v1
$X2_WORKSPACE/x2_sonic
$X2_WORKSPACE/Human+智能服装动作捕捉系统
$X2_WORKSPACE/GR00T-WholeBodyControl
```

环境变量：

```bash
export X2_WORKSPACE=/path/to/workspace
export PYTHONPATH="$X2_WORKSPACE/CWI_CrossEmbodiment_Sim/tools:$X2_WORKSPACE/CWI_CrossEmbodiment_Sim/src:$X2_WORKSPACE/x2_sonic:$X2_WORKSPACE/Human+智能服装动作捕捉系统:$X2_WORKSPACE/GR00T-WholeBodyControl:${PYTHONPATH}"
export MUJOCO_GL=egl
```

两个 import 的真实来源：

- `official_x2` 是本 Git 仓库的 `tools/official_x2/` namespace package，不在 vendor SDK。出现 `No module named 'official_x2'` 时，首先检查 `$REPO/tools` 是否在 `PYTHONPATH`；仅复制 SDK 不会修复这个报错。
- `general_motion_retargeting` 的当前实际 import 位置是 `Human+智能服装动作捕捉系统/general_motion_retargeting/`。`/home/humanplus/humanoid-GPT/Humanoid-GPT` 当前只有历史报告，不提供该 Python package。

官方容器内使用 ROS 2 Humble 和 CycloneDDS：

```bash
export RMW_IMPLEMENTATION=rmw_cyclonedds_cpp
export ROS_DOMAIN_ID=220
```

`ROS_DOMAIN_ID` 必须不高于 232；本机已验证 233/234 会让 DDS UDP 端口越界并在 physics 前失败。不要把宿主 ROS Jazzy 的 Python 路径混入 Python 3.11 host 环境。

## 5. 百度网盘：本次新增的可恢复包

### 5.1 动力学可行化 seed bundle

```text
HUMAN+/HUMANPLUS_X2_PLUS/2026-08-10/dynamics_feasibility/
  x2_forward4_dynamics_feasibility_seed_v1.tar.gz
  x2_forward4_dynamics_feasibility_seed_v1.tar.gz.sha256
  x2_forward4_dynamics_feasibility_seed_v1.tar.gz.contents.txt
```

- 大小：`336,075,248 B`
- SHA-256：`660f892728d64c65cee374e898ede7cff15e89fa691b72f6d886aa52cb510033`
- 上传状态：`bdpan upload` 已返回成功；按项目规则未做上传后远端复查。

该包包含：

- 精确 Stage152 四域 source checkpoint `model_step_000200.pt`，SHA `b73c345995c5d468c18d223de96b29fff4cd4866e4d56bb6d5296c4a68540679`；
- `stage72_official_true_forward4_v1` 的 4 条 reference 和 metadata；
- 上游 `x2_real_readonly_session04_canonical_31dof.npz`，SHA `287a134c9e8e260672f3b58c3968747604a5cb7e23af3186cfbba86ca4fc7`；
- ideal/filter/delay/delay+noise 四域 r1 trace；
- X2 assets、IsaacLab USD cache、旧 X2 scripts/tools/tests；
- `sonic_x2_sandbox` 基础 commit、tracked binary patch 和 untracked files archive；
- 当前 baseline/gate 报告和迁移卡。

恢复：

```bash
sha256sum -c x2_forward4_dynamics_feasibility_seed_v1.tar.gz.sha256
tar -xzf x2_forward4_dynamics_feasibility_seed_v1.tar.gz -C "$X2_WORKSPACE_PARENT"
```

归档保留了原机器相对于 `/home/humanplus` 的路径；若新机器用户名或布局不同，解压后按第 4 节移动目录并更新环境变量。

### 5.2 X2/AimDK Docker runtime

```text
HUMAN+/HUMANPLUS_X2_PLUS/2026-08-10/runtime/
  x2_aimdk_runtime_images_v1.tar.gz.part-00
  x2_aimdk_runtime_images_v1.tar.gz.part-01
  x2_aimdk_runtime_images_v1.parts.sha256
  x2_aimdk_runtime_images_v1.images.txt
```

分片：

| 文件 | 大小 | SHA-256 |
|---|---:|---|
| `part-00` | 3,221,225,472 B | `164eb2ee0ae99931e0ef73877b58b312095fd8628cb4d21a6957faa2e15c511b` |
| `part-01` | 2,332,520,747 B | `847a44c26fbc637cfc937fc99296d6854750a9ed943b97080641f69d72593d3a` |

上传状态：两个分片均已由 `bdpan upload` 返回成功；按项目规则未做上传后远端复查。

归档同时保存两个共享层镜像标签：

- `x2-aimdk-humble:1.0`，image ID `sha256:52406d45...81fb7cc`；
- `g1-deploy-dev:latest`，image ID `sha256:4cb3e60d...c128530`。

恢复：

```bash
sha256sum -c x2_aimdk_runtime_images_v1.parts.sha256
cat x2_aimdk_runtime_images_v1.tar.gz.part-* | gzip -dc | docker load
docker image inspect x2-aimdk-humble:1.0 g1-deploy-dev:latest
```

## 6. 百度网盘：已有且按需下载的资产

### 6.1 2026-08-10 补充源码树

```text
HUMAN+/HUMANPLUS_X2_PLUS/2026-08-10/source_trees/
  x2_official_rl_deploy_v1_source_assets_20260810.tar.gz
  Humanoid-GPT_requested_tree_20260810.tar.gz
  general_motion_retargeting_actual_runtime_20260810.tar.gz
```

| 包 | 大小 | SHA-256 | 内容与用途 |
|---|---:|---|---|
| `x2_official_rl_deploy_v1_source_assets_20260810.tar.gz` | 155,344,608 B | `b6f6cecfe08fbfb9666638ead01a9882de16cee73ce7000766ce9a020ae152cf` | SDK source/vendor/worktree、`model_info/x2.xml`、`model_info/scene.xml`、`motion_control.yaml`、官方 meshes、models、Dockerfile；排除已另行归档的 results/cache/downloads |
| `Humanoid-GPT_requested_tree_20260810.tar.gz` | 12,886,212 B | `4b09d83fec09d003f16760ebee3f4e505656545f8fa751362682fca5d191b03a` | 按迁移方要求原样保存该目录；当前实际内容为历史 reports，不能修复 GMR import |
| `general_motion_retargeting_actual_runtime_20260810.tar.gz` | 126,973 B | `c4f1dd76384ce9230106c9ba05d1d32d00a00f8125b742ad890e3dd290382886` | 当前环境实际导入的 GMR runtime、`smplx_to_x2.json`、SMPL loader 及相邻 `gmr` shim |

上传命令均已返回成功；按项目规则未做远端复查。SDK 中的实际路径是 `.../model_info/x2.xml` 和 `.../model_info/scene.xml`，不是字面上的 `scene/x2.xml`。

路径和已盘点大小如下：

| 用途 | 百度网盘路径 | 大小 |
|---|---|---:|
| 官方 AimDK v1.0 SDK/MJCF/ROS | `HUMAN+/HUMANPLUS_X2_PLUS/2026-08-07/official_x2_v1/sdk/aimdk-x2-v1.0.0-official.zip` | 156,945,205 B |
| Phase34/50 closed trace、Phase49 权重 | `HUMAN+/HUMANPLUS_X2_PLUS/2026-08-09/phase55/x2_phase55_native_upper_robust_20260809.tar.gz` | 495,434,849 B |
| Stage219 后端与视频基线 | `HUMAN+/HUMANPLUS_X2_PLUS/2026-08-09/phase42/x2_phase42_dual_track_20260809.tar.gz` | 20,567,491 B |
| 24条面板、Phase29/30 reference | `HUMAN+/HUMANPLUS_X2_PLUS/2026-08-09/phase30/x2_phase30_dual_track_20260809.tar.gz` | 18,973,586 B |
| native Gold MotionLib/recovery | `HUMAN+/HUMANPLUS_X2_PLUS/2026-08-09/phase10/x2_phase10_recovery_native_gold_20260809.tar.gz` | 20,594,628 B |
| faithful exact-S7 | `HUMAN+/HUMANPLUS_X2_PLUS/2026-08-09/phase47/x2_phase47_official_state_faithful_s7_20260809.tar.gz` | 299,929,905 B |
| held-out repair/outcome-aware BASE | `HUMAN+/HUMANPLUS_X2_PLUS/2026-08-09/phase36/x2_phase36_dual_track_20260809.tar.gz` | 29,623,463 B |
| Stage152 dual critic init | `HUMAN+/HUMANPLUS_X2_PLUS/2026-08-07/checkpoints/stage152_B_dual_equal_split_init.pt` | 193,683,894 B |
| Stage152 materialized | `HUMAN+/HUMANPLUS_X2_PLUS/2026-08-07/checkpoints/stage152_B_materialized_v1.pt` | 151,324,629 B |
| 旧工程兼容快照 | `HUMAN+/x2_teleop_final.zip` | 454,053,271 B |

最小恢复先下载 SDK、2026-08-10 seed bundle、Docker runtime、Phase55、Phase42、Phase30、Phase10。仅继续 faithful WBT 时再下载 Phase47/36；旧 zip 只用于补旧路径，不应覆盖较新的 seed bundle。

## 7. 四种动力学可行化方法的额外需求

| 方法 | 可直接复用 | 仍需补充 |
|---|---|---|
| OmniTrack 式 privileged physicalization | 现有 IsaacLab/RL、Stage152、forward4、rollout 导出 | privileged reward/obs 配置与严格导出门；无新外部优化器 |
| DynaRetarget / SBTO | GMR/IK reference、MuJoCo、CEM 基础 | `Atarilab/sbto` 精确 commit 和许可证快照；对外发布前确认许可 |
| DDR | MuJoCo/CEM、现有人体管线 | matched SMPL/人体关键点 forward-walk；当前 X2 native forward4 不能冒充人体输入 |
| DSMS | X2 MJCF、MuJoCo | IPOPT、HSL MA57 授权/安装、具体实现源码与 commit；当前迁移包未包含 |

不要为了开始前置审计而搬完整 AMASS/PHUMA/BONES。只有 DDR 或扩大 train/held-out 时才迁移这些原始人体库。

## 8. 新机器恢复顺序

1. 安装 NVIDIA driver、Docker、NVIDIA Container Toolkit 和系统包。
2. clone Git 仓库，检出本卡对应或更新的提交。
3. 建立第 4 节目录布局。
4. 下载并校验 2026-08-10 seed bundle，恢复源码增量、checkpoint、reference 和 trace。
5. 下载官方 SDK zip。
6. 下载两个 Docker 分片，校验后 `docker load`。
7. 按第 3 节建立 host conda 环境；不要改安装顺序。
8. 按需下载 Phase55/42/30/10 等历史资产。
9. 先跑 smoke checks 和全离线前置审计，不要直接长训。

`bdpan` 下载示例：

```bash
bdpan download \
  'HUMAN+/HUMANPLUS_X2_PLUS/2026-08-10/dynamics_feasibility/x2_forward4_dynamics_feasibility_seed_v1.tar.gz' \
  "$X2_WORKSPACE/downloads"
```

## 9. 恢复验收

```bash
python -c "import torch; assert torch.cuda.is_available(); print(torch.__version__, torch.version.cuda)"
python -c "import isaacsim, isaaclab, mujoco, onnxruntime; print(mujoco.__version__, onnxruntime.__version__)"
python -m pytest -q tests/test_x2_physics_provenance_phase25.py
docker image inspect x2-aimdk-humble:1.0 g1-deploy-dev:latest
```

数据验收：

1. `model_step_000200.pt` SHA 必须为 `b73c...0679`；
2. raw NPZ SHA 必须为 `287a...7fc7`；
3. forward4 metadata 和 4 个 PKL 必须与 bundle 内容清单一致；
4. 四域 r1 trace 能离线复算当前 0/4、1/4、0/4、0/4 结论；
5. 官方 `scene/x2.xml`、`motion_control.yaml`、ROS executable 可见；
6. Docker 两个 tag 均成功加载；
7. 明确 reference provenance 为 X2 native-derived，不写成 GMR-derived。

验收通过后，先执行工单的 5 条全离线前置审计，再决定进入方法 1，还是因根速度分布/镜像问题转向数据侧。任何方法达到 `ideal ≥2/4` 且过冲下降即停止，不继续升级更重的优化器。

## 10. 备份与复查规则

- Git 保存代码、配置、报告和小型 manifest；百度网盘保存 checkpoint、数据、trace、mmap、SDK 和 Docker 分片。
- 大文件每累计约 10 个实质任务归档一次，不需要每个任务上传。
- `bdpan upload` 返回成功即记录成功；远端大小/存在性/哈希由用户指定的每日人工检查执行，不在每次上传后重复检查。
- 本卡是迁移入口；更细的数据来源与方法边界见 `docs/backup/X2_DYNAMICS_FEASIBILIZATION_MIGRATION_CARD.md`。
