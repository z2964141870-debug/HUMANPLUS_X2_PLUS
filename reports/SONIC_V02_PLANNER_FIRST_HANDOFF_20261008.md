# Sonic X2 v0.2 接班路线：先 Planner，再全身控制

整理日期：2026-10-08。本文供下一位同学从已有项目继续工作；它记录文件位置和待验证条件，不描述机器人的实时状态，也不授权真机接管。当前项目的旧实验状态以 [2026-09-29 交接](HANDOFF_20260929.md) 和 [操作规范](../script/sonic_x2_transfer_v2/GR00T_audit/docs/OPERATIONS.md) 为准。

2026-10-19 补充：键盘行走优先读 [操作卡审核与 v0.2 部署交接](SONIC_V02_KEYBOARD_WALK_REVIEW_20261019.md)。其中补齐 F10 真机入口、作者 PC2 与本机 SoC1 的对应关系，以及新旧 binary/CLI、dry-run、触发文件和停止语义的区别。

## 先拿到什么

GitHub 项目：[`HUMANPLUS_X2_PLUS` 的 `humanplus_sonic_x2` 分支](https://github.com/z2964141870-debug/HUMANPLUS_X2_PLUS/tree/humanplus_sonic_x2)。2026-10-08 核验远端分支为 `7284830ac176e8bc481f2bc5f66788f26ac9bf70`。

| 内容 | GitHub 状态 | 入口 |
| --- | --- | --- |
| 旧 X2 C++ deploy、HAL 接管、脚本和测试的代码快照 | 已提交 | [`script/sonic_x2_transfer_v2/GR00T_audit/`](../script/sonic_x2_transfer_v2/GR00T_audit/) |
| 最后已知机器人状态、决策、操作规范和实验记录 | 已提交 | [报告索引](README.md)，再读其链接的 `CURRENT_STATE.md`、`DECISIONS.md`、`OPERATIONS.md` 和 `EXPERIMENTS.md` |
| 衣服到旧 Sonic 的 35.02 Hz dry-run 终端日志、少量 MC 快照 | 已提交 | [`logs/garment_dryrun/`](../logs/garment_dryrun/) |
| 旧 `p3_8900` 的 Phi/action-scale 和 provenance | 已提交 | [`data/model_metadata/`](../data/model_metadata/) |
| 旧 `p3_8900` ONNX、编译产物、完整原始遥测/视频 | **未提交** | [`data/manifests/external_artifacts_20260831.tsv`](../data/manifests/external_artifacts_20260831.tsv) 仅列位置和哈希，不能替代原件 |
| 用户所述“已控制 G1 的 SMPL 输出模型”、其样本及输入协议 | **未定位/未验收** | 接手后记录真实路径、SHA-256、SONIC 版本和一段可回放数据 |

百度网盘 [上传状态](BAIDU_UPLOAD_STATUS.md) 截至 2026-08-31 仍为 `NOT UPLOADED`；不要把 GitHub 有代码误解成大权重和全部实验数据已经备份。新 v0.2 模型可从作者公开的 Git LFS/Hugging Face 发布取得，旧本地大文件不必成为启动新路线的前置条件。

## 两套版本的边界

| 契约 | 我们的历史基线 | 作者 v0.2 真机路线 |
| --- | --- | --- |
| Policy | `x2_sonic_frozen_g1core_lora_v2.onnx`，`p3_8900`，SHA-256 `8ccc42a82ea2c446aa708aece58c7a638ea943a79ed88eeab259669e641271e9` | `x2_sonic_s1ft16000_g1.onnx`，MD5 `5ae9313443068d0fdebdb93480d23746` |
| 输入/输出 | robot-frame pose ONNX `1670 -> 31` | pose actor `1670 -> 31`；全身路径另有 token actor `1670 -> 31` 和 SMPL tokenizer `840 -> 64` |
| Plant/动作尺度 | 旧 `p3_8900` Phi 表和本地 C++ `policy_parameters.hpp` | `vendor_20260823` plant、配套 Phi/action scale 和 `trained_gains_s0.yaml` |
| 运行结构 | SoC1 policy 50 Hz、唯一 HAL writer 250 Hz；SoC0 官方 MC/HAL；reconstructed pelvis | 作者 PC2 是 Jetson Orin NX/aarch64，对应我们的 SoC1；KPlanner + pose actor 可迁移，但须适配路径、ABI、IMU 和 MC 交接，不能原样执行硬编码 ritual |
| 真机证据 | 特定吊绳支撑下约 300 秒固定 StandStill；无自主行走验收 | 作者 v0.2 项目页报告 18 分钟无吊绳行走；仍须核对其精确实验工件，不能当作本项目或 SMPL 全身真机验收 |

新 pose actor 虽然仍是 `1670 -> 31`，不能只把 ONNX 路径换进旧 deploy。旧表中踝 pitch action scale 约 `0.63155`；新 vendor-plant 表约 `0.25789`。腰 pitch 也从约 `0.63155` 变为约 `0.48510`。形状相同不会检测出这个物理语义错配。`trained_gains_s0.yaml` 的当前公开版本还包含腰部增益倍率；论文所称的 `unity gains` 与这个同名文件的精确实验版本需要追溯，不能凭名称推定一致。

作者公开的固定参考：[v0.2 项目页](https://sonic-agibot-x2.github.io/sonic-transfer/)、[代码 `ship-frozen-core`，提交 `c0036a457dd5`](https://github.com/meetsitaram/GR00T-WholeBodyControl-X2/tree/c0036a457dd570b41eaadfa9b4c2115db9df551a)、[模型清单](https://github.com/meetsitaram/GR00T-WholeBodyControl-X2/blob/c0036a457dd570b41eaadfa9b4c2115db9df551a/gear_sonic_deploy/models/x2_sonic_s1ft16000.manifest.txt)、[F07 模型说明](https://github.com/meetsitaram/GR00T-WholeBodyControl-X2/blob/c0036a457dd570b41eaadfa9b4c2115db9df551a/docs/x2/F07_frozen_g1_model.md)、[F08 Planner/全身切换](https://github.com/meetsitaram/GR00T-WholeBodyControl-X2/blob/c0036a457dd570b41eaadfa9b4c2115db9df551a/docs/x2/F08_kplanner_wholebody_switching.md)。作者的发布仓库是模型来源；我们自己的 GitHub 分支是旧部署和实验记录来源。

## 频率和接口速查

| 环节 | 已知设置 | 注意 |
| --- | --- | --- |
| 我们旧衣服源 | 实测 `35.02 Hz`；p95/max gap `30.5/46.6 ms` | 只证明 dry-run；`--diff-hz 50` 与 `--velocity-mode timestamp` 同时出现时，速度以真实时间戳为准 |
| 我们旧 Sonic | `50 Hz` policy | 已有无 HAL publisher 的 live dry-run；带电衣服闭环未验收 |
| 我们旧 X2 HAL | `250 Hz` 唯一 writer | v0.9/`aimdk_msgs 0.8.18` 上验证过反馈新鲜度；C++ 二进制默认值仍是 `500 Hz`，旧 launcher 显式传 `250 Hz`，不能直接跑裸二进制；旧 500 Hz 曾造成回调拥堵，见 D-002 |
| 作者新 KPlanner | 模型帧 `30 fps`，pose wire `50 Hz`，输出 `qpos38`，每次计划最多 `64` 帧 | 发布环调用 `get_next_frame_resampled(OUTPUT_FPS)` 做 30 -> 50 Hz 重采样与跨 replan 淡化，不是每 20 ms 原速消费一帧；future 9 个槽位，间隔 `0.1 s` |
| 作者新 Sonic/token service | pose actor `50 Hz`；token service 按 `50 Hz` 处理意图 | 固定提交 C++ writer 硬编码 2 ms/500 Hz，没有本机旧 `--writer-hz` flag；移植本机 IO 后实测频率/状态年龄，不能把旧参数直接传给新 binary |

我们的旧衣服线路：`BLE -> SoC1 CUDA LFP -> 3588S postprocess/Fast-SMPL/native GMR -> HMCP qpos36 UDP :51234 -> raw ZMQ :5555 -> proxy -> guarded ZMQ :5556 -> Sonic 50 Hz -> HAL 250 Hz`。3588S RPC 为 `:51237`，`x2_debug` 为 `:5557`。这一路输出**机器人关节参考**，不是作者 token service 接收的人体 SMPL 意图；两种输入不能互换。旧链路的详细启动条件见 [衣服实时 README](../script/sonic_x2_transfer_v2/GR00T_audit/gear_sonic_deploy/README_X2_GARMENT_LIVE.md) 和 [powered gate](../script/sonic_x2_transfer_v2/GR00T_audit/gear_sonic_deploy/README_X2_GARMENT_POWERED_INTEGRATION.md)。

## 第一阶段：复现 v0.2 Planner

1. 在独立 checkout 固定作者 `c0036a457dd570b41eaadfa9b4c2115db9df551a`，取得 Git LFS 的三个 ONNX 和 `kplanner_g1core/x2_planner_template_s1d_a05_ws.onnx`。把 planner 文件按作者目录约定命名为 `x2_planner_template.onnx`。核对 pose/token/tokenizer MD5 分别为 `5ae9313443068d0fdebdb93480d23746`、`438a14fea8849b1feaa770e4e9de03c5`、`ee03f06e0211590ece8eff3185efba98`；另记录 planner SHA-256。不要把 LFS pointer 当权重。
2. 先跑作者 [F07 的 Planner-only MuJoCo 命令](https://github.com/meetsitaram/GR00T-WholeBodyControl-X2/blob/c0036a457dd570b41eaadfa9b4c2115db9df551a/docs/x2/F07_frozen_g1_model.md#run-in-sim-acceptance-command-token-path) 中的第二条命令，固定 `MODEL=x2_sonic_s1ft16000_g1.onnx`、`PLANNER_MODEL=kplanner_g1core/`、`KPLANNER_PROFILE=g1core_bare_v1.env`、`SIM_TUNING_YAML=trained_gains_s0.yaml`、`X2_PLANT=vendor_20260823`、`X2_ANCHOR_ORI_MODE=body`。`ALLOW_MISMATCH=1` 只用于该作者文档的隔离仿真命令，不带到真机。记录站立、起步、停止、转向的机器人 qpos、policy 输出、clip、倾角和足接触。仿真画面本身不构成通过证据。

第一天可直接复制的**仿真**入口（需要 Linux/MuJoCo、Git LFS、Hugging Face CLI、作者仓库依赖和可被 `pygame` 识别的手柄；`MODEL_CACHE` 改为存放下载权重的仓库外绝对路径）。下载后核验模型文件不是约 130 字节的 LFS pointer，再启动仿真：

```bash
git clone --branch ship-frozen-core --single-branch https://github.com/meetsitaram/GR00T-WholeBodyControl-X2.git sonic_x2_v02_upstream
cd sonic_x2_v02_upstream
git checkout c0036a457dd570b41eaadfa9b4c2115db9df551a
git lfs pull --include='gear_sonic_deploy/models/x2_sonic_s1ft16000_g1.onnx,gear_sonic_deploy/models/x2_sonic_s1ft16000_g1_token.onnx,gear_sonic_deploy/models/x2_smpl_tokenizer_v11release.onnx'
md5sum gear_sonic_deploy/models/x2_sonic_s1ft16000_g1.onnx gear_sonic_deploy/models/x2_sonic_s1ft16000_g1_token.onnx gear_sonic_deploy/models/x2_smpl_tokenizer_v11release.onnx

MODEL_CACHE=/absolute/path/outside/git
huggingface-cli download tinkerbuggy/sonic-x2 --include 'kplanner_g1core/x2_planner_template_s1d_a05_ws.onnx' --local-dir "$MODEL_CACHE"
ln -s "$MODEL_CACHE/kplanner_g1core/x2_planner_template_s1d_a05_ws.onnx" "$MODEL_CACHE/kplanner_g1core/x2_planner_template.onnx"

ALLOW_MISMATCH=1 X2_PLANT=vendor_20260823 X2_ANCHOR_ORI_MODE=body \
SIM_TUNING_YAML=gear_sonic_deploy/configs/real_deploy_tuning/trained_gains_s0.yaml \
MODEL=gear_sonic_deploy/models/x2_sonic_s1ft16000_g1.onnx \
PLANNER_MODEL="$MODEL_CACHE/kplanner_g1core" \
KPLANNER_PROFILE=gear_sonic/config/kplanner_profiles/g1core_bare_v1.env \
./gear_sonic/scripts/sim_onnx_planner.sh
```

若 `x2_planner_template.onnx` 已存在，先比对哈希，不要覆盖。`huggingface-cli` 若已由新版 `hf` 替代，使用同样参数的 `hf download`。Planner-only 的顺序是：确认模型和配置哈希 -> 启动本地仿真栈 -> 用手柄分别测静止/起步/停止/转向 -> 导出日志 -> `./gear_sonic/scripts/simstack_local.sh --stop`。仿真启动脚本和依赖以作者固定提交的 [F07](https://github.com/meetsitaram/GR00T-WholeBodyControl-X2/blob/c0036a457dd570b41eaadfa9b4c2115db9df551a/docs/x2/F07_frozen_g1_model.md) / [F11](https://github.com/meetsitaram/GR00T-WholeBodyControl-X2/blob/c0036a457dd570b41eaadfa9b4c2115db9df551a/docs/x2/F11_docker.md) 为准；此段未在本机执行，不是一次新的通过记录。
3. 再把这套固定版本作为**新候选**适配我们的 v0.9 SoC1/SoC0 架构。先做接口/数值回放：ONNX 输入 1670、输出 31；关节和四元数顺序；body anchor；vendor plant；动作尺度；PD/LPF/clamp；KPlanner 到 Sonic 的 50 Hz wire；debug/状态回调时序。记录最终二进制、脚本、模型、配置的 SHA-256。保留旧 `p3_8900` 的冻结安装和日志。
4. 真机阶段从当前状态重新做只读 preflight，证明 MC/HAL 归属、反馈和**唯一 writer**；先验证静态参考与安全返回，再验证有界 planner 意图。我们旧控制器的 250 Hz writer 是已测得的本机起点，但**不能仅凭这个旧结果证明新模型在 250 Hz 下与作者真机等价**。迁移时测每类 topic 的实际频率、policy tick p95/p99、状态年龄和丢帧。任何受支撑结果只按其支撑程度命名。

作者的 `x2_pc2/robot_env.env.frozen.template` 集中设置三模型、plant、anchor 和 planner；该文件还可能先于 `g1core_bare_v1.env` 注入部分 `KPLANNER_*` 值。比较 sim 与真机时保存**解析后的有效环境变量**，不能只写 profile 文件名。作者的 PC2 启动脚本含自动停 MC 和不同的控制进程编排，不能直接移植到本机已有 SoC1 HAL handoff。

旧机上的正常启动次序只作架构参考：完全吊起且先确认官方 MC 唯一控制 -> deploy 以 writer 抑制状态待命 -> 停 MC 并证明命令静默、HAL 反馈仍在线 -> 250 Hz 静态 PD 捕获姿态 -> 在 Policy OFF 下确定脚底接触/吊带支撑 -> `load` 捕获该状态 -> 条件满足后 policy -> `stop` 返回静态 PD -> 完全吊起后 `lifted` 恢复 MC。精确命令及异常恢复见 [OPERATIONS.md](../script/sonic_x2_transfer_v2/GR00T_audit/docs/OPERATIONS.md)。新模型不能沿用旧 wrapper 的固定哈希或 `p3_8900` 参数来跳过接口验收。

## 第二阶段：接入 SMPL 全身路径

1. 让用户提供那条已控制 G1 的 SMPL 模型/程序的**准确文件、SHA-256、输入/输出定义、G1 SONIC checkpoint 和可回放片段**。目前我们的 GitHub 分支没有这些工件。先核对其 `SMPL joints/root` 坐标系、四元数顺序、时间戳和 reference window；不要默认“输出 SMPL”就等于 tokenizer 所需的 840 维 observation。
2. 作者 token service 的外部意图入口是 `pico_intent :5573`（含 root-relative `smpl_joints[72]`、人体朝向等字段）；它结合本地 `x2_debug :5557` 的机器人状态构造 `smpl_obs[840]`，经配套 tokenizer 生成 `motion_token[64]`，再在 `:5574` 发给 token actor。复用 G1 的人体前端时，应在这个意图/observation 契约处接线，**旧 HMCP qpos36、G1-to-X2 增量映射和 raw pose proxy 不直接接入该 token 服务**。先在离线 replay 比较输入帧、token、动作和断流行为。公开源码的 `pico_intent` 文档与 tape 元数据对 `human_quat` 分量顺序的描述并不完全一致；适配时以固定提交的编码/解码代码和已知 replay 样本做数值核对，不靠字段名猜测。
3. 在 Planner-only 通过后，先用作者原生 Pico sender 验证同一组 `s1ft16000` pose/token/tokenizer 的**仿真**全身路径，作为衣服适配前的对照（仍在上面的作者仓库目录，并沿用已下载的 `MODEL_CACHE`）：

```bash
TOKEN_SVC_OPERATOR_ROOT_LEVEL=zero ALLOW_MISMATCH=1 SIMSTACK_OMNIHAND=1 \
X2_PLANT=vendor_20260823 X2_ANCHOR_ORI_MODE=body \
SIM_TUNING_YAML=gear_sonic_deploy/configs/real_deploy_tuning/trained_gains_s0.yaml \
MODEL=gear_sonic_deploy/models/x2_sonic_s1ft16000_g1.onnx \
SIMSTACK_TOKEN=gear_sonic_deploy/models/x2_sonic_s1ft16000_g1_token.onnx \
SIMSTACK_TOKENIZER=gear_sonic_deploy/models/x2_smpl_tokenizer_v11release.onnx \
PLANNER_MODEL="$MODEL_CACHE/kplanner_g1core" \
KPLANNER_PROFILE=gear_sonic/config/kplanner_profiles/g1core_bare_v1.env \
./gear_sonic/scripts/sim_onnx_planner.sh --whole-body-teleop
```

另开一个终端，在同一仓库运行作者的 `./gear_sonic/scripts/run_pico_teleop.sh`；没有 Pico 时先用作者的 intent tape/replay 入口，不要把衣服 HMCP 假装成 `pico_intent`。验收静止人体、微小上肢/腰动作、engage/release 和 KPlanner 返回后，用 `./gear_sonic/scripts/simstack_local.sh --stop` 清理仿真。再用 G1 的 SMPL 前端替换**输入端**做离线 replay 与无 HAL 输出 dry-run；旧 proxy 的 `STANDSTILL_READY -> ARM -> 10 s blend -> LOCKOUT` 语义可作为安全需求，但协议是 pose wire，不能未经适配就声称已经保护 token 流。真机 SMPL 测试排在 Planner 基线和新接口安全门之后。

作者 v0.2 的 18 分钟硬件结果是 pad/KPlanner locomotion。论文明确说高动态 whole-body 真机仍在 bring-up；公开评估中 `s1ft16000` 的 27 条 Pico/SMPL capture 通过率为 `37.0%`。不要把 planner 成功写成衣服全身遥操成功。若二阶段表现差，先比较完整的 SMPL 输入契约与 checkpoint，再考虑另一个 checkpoint；不要回头随意调旧 PD 或把 `p3_8900` 的负结果迁移给新模型。

## 每阶段交付

每个实验附一个小 README：源码提交、四类工件 SHA-256、实际命令及解析后的环境、机器人/仿真版本、输入来源、观测和 writer 的实测频率、原始日志位置、通过/失败判据、最终 MC/HAL publisher 归属。先把结论和小型日志提交 Git；大权重/遥测按 [存储策略](STORAGE_POLICY.md) 单独登记，未上传成功就明确写 `NOT UPLOADED`。
