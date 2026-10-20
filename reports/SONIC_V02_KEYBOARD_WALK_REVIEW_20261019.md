# X2 Sonic v0.2：给接班同学的键盘行走交接

核验日期：2026-10-19。目标是复现作者已经报告完成的 **KPlanner locomotion**，把输入设备由手柄换成键盘；不是继续用旧模型循环播放 walk PKL，也不是今天就接入衣服全身遥操。

执行安排见 [分阶段派工与汇报卡](SONIC_V02_WALK_PROGRESS.md)：每轮只做用户明确派发的一阶段，完成后提交证据并等待审核。下面的路线图不构成一次性执行全部阶段的授权。

本文是文档和源码审查，不是真机验收记录。本轮没有连接机器人、停止 MC、运行发布器、上传代码或下载权重。当前机器人固件、控制状态、进程及 publisher ownership 均须现场重新核验。下文的运行命令只用于隔离仿真，不可改一个 IP 就用到真机。

## 1. 结论和最短路线

**作者有真机部署文档，v0.2 也有无吊绳行走的公开报告。学弟应换成完整的 v0.2 模型/plant/Planner 套装，再接键盘意图。旧操作卡不能直接当作该路线的部署说明。**

作者项目页写明 v0.2 更新于 2026-09-12，报告 18 分钟无吊绳粗糙地形行走。这里的“作者完成”指该开源项目作者的实验，不是智元官方 SDK 对我们的旧模型、固件和启动脚本作出的兼容保证，也不是我们已经复现成功。

建议按以下三项交付，不再在旧 `p3_8900` 上堆新 PD 参数：

1. 固定 `s1ft16000 + G1-core KPlanner + vendor_20260823`，在作者的真实 C++ deploy/MuJoCo 物理闭环里测静止、起步、松键停止、转向。
2. 将本机已验证的 MC/HAL 交接、250 Hz writer、状态回调和 IMU 适配迁入一个隔离的新候选。模型/观测/plant 用新套装，旧运行时不覆盖。
3. 键盘只发送 `planner_cmd`。静止/起停和输入断连保护通过后，再安排受支撑短时真机首走。衣服、SMPL、token service、3588S 均不作为此阶段前置。

“仿真通过”不能代替真机验收；“受支撑通过”不能写成自主承重行走。无需为了换模型再重做旧 p3 的同一批 300 秒站立研究。

## 2. 作者文档：按这个顺序读

固定上游：`meetsitaram/GR00T-WholeBodyControl-X2`，分支 `ship-frozen-core`，提交 `c0036a457dd570b41eaadfa9b4c2115db9df551a`（提交日期 2026-09-21）。这是本次审查基准，不把浮动 main 当作固定实验身份。当前默认 launchers 已指向另一套 native `v16ft8_45000`；本路线必须显式指定 frozen 模型。

| 文档 | 用途 |
| --- | --- |
| [Transfer 项目页](https://sonic-agibot-x2.github.io/sonic-transfer/) | 核对 v0.2 硬件结果与模型谱系；不是完整部署步骤 |
| [F10：真机 PC2 / colcon / bringup / push / ritual](https://github.com/meetsitaram/GR00T-WholeBodyControl-X2/blob/c0036a457dd570b41eaadfa9b4c2115db9df551a/docs/x2/F10_real_robot_colcon.md) | **真机部署主文档**：环境、编译、同步、切模型、启动和退出 |
| [F07：Frozen G1 模型](https://github.com/meetsitaram/GR00T-WholeBodyControl-X2/blob/c0036a457dd570b41eaadfa9b4c2115db9df551a/docs/x2/F07_frozen_g1_model.md) | `s1ft16000` 配套图和 Planner-only 仿真入口 |
| [F01：手柄控制](https://github.com/meetsitaram/GR00T-WholeBodyControl-X2/blob/c0036a457dd570b41eaadfa9b4c2115db9df551a/docs/x2/F01_gamepad.md) | 作者已实现的控制流程、`planner_cmd` 入口和 frozen core 真机记录 |
| [F09：MuJoCo 栈](https://github.com/meetsitaram/GR00T-WholeBodyControl-X2/blob/c0036a457dd570b41eaadfa9b4c2115db9df551a/docs/x2/F09_mujoco_sim.md) | 真实 C++ deploy + 物理桥，不是设置 qpos 的动画查看器 |
| [Runbook 索引/依赖准备](https://github.com/meetsitaram/GR00T-WholeBodyControl-X2/blob/c0036a457dd570b41eaadfa9b4c2115db9df551a/docs/x2/README.md) | Ubuntu/Python/Docker/LFS、运动 bank 生成 |
| [端口表](https://github.com/meetsitaram/GR00T-WholeBodyControl-X2/blob/c0036a457dd570b41eaadfa9b4c2115db9df551a/x2_pc2/PORT_REGISTRY.md) | 真机与仿真端口、bind/connect 归属 |

作者公开的主要操作说明是**手柄驱动 Planner**；本轮没有找到已核验的独立 WASD 真机操作卡。键盘需要复用相同意图接口，不等于可以直接复用学弟的 PKL 键盘发布器。

**硬件名称对应：**F10 的 PC2 是 Jetson Orin NX/aarch64，对应我们承担 Sonic 的 **SoC1 Orin**，不是 3588S。架构具备复用基础；仍要适配用户、安装目录、ROS/消息 ABI 和 MC 交接。一些 ritual 实际硬编码 `/home/run/gear-sonic`，不能只改 `PC2_PREFIX` 就认定全部迁移完成。首次键盘测试不需要安装手柄自动配对/开机启动服务。

## 3. 学弟操作卡审查

审查对象：微信提供的 `walking_day_v2_opcard.md`，SHA-256：

```text
b85b080be5809cee2451c4f3b420faab791ef3501c8c22aa991fe7c3e133eda1
```

卡片的 v2 是流程第二版，不代表 Sonic transfer v0.2。卡片提及的 `gate_walk.py`、`keyboard_walk_publisher.py`、`send_resume_chord.py`、`worklog_20261011.md` 本轮未在已搜索的本地项目路径中找到；以下不声称已经审计这些文件。所谓“抽搐根因实锤”和 self-test 数量是卡片陈述，尚未核对原始日志和实际执行二进制。

| 优先级 | 操作卡问题 | 处理 |
| --- | --- | --- |
| 阻断 | 仍加载 `x2_sonic_frozen_g1core_lora_v2.onnx`，W 播放 `x2_relaxed_walk.pkl`；没有 KPlanner | 改为本节后列出的新 actor/plant/Planner 整套；不要只替换文件名 |
| 阻断 | `touch /tmp/walkday_v2.trigger` 在启动旧节点之前执行 | 本仓库旧 C++ 构造时会删除 stale trigger；这次 touch 会被删除，可能一直 STANDBY。由管理 launcher 在节点 ready、MC 静默且反馈新鲜后创建唯一 session trigger |
| 阻断 | `--supported-policy-probe` 未给时长，却要求 policy 后先等 15 秒 | 本仓库旧 C++ 默认 `supported_policy_seconds=5.0`；无法完成该步骤。且默认 2° 倾角增量/0.5 rad/s 速度闸门是站立探针，不是新模型行走配置。先核验机器人实际源码和 binary，不能靠只放大 max-dev 修复 |
| 高 | 将旧踝 Kp/Kd、腰 LPF、0.85 rad clamp 与新模型混用 | 从新模型的配套 plant/tuning 起步，逐项记录有效参数；旧站立 slew 也不当作行走配方 |
| 高 | 认为 MC PASSIVE 已静默，或只看一个 topic 一次 echo | PASSIVE 可能仍发命令；检查四组 command 的实际流量和归属，并确认四组 joint state + IMU 一直新鲜 |
| 高 | broad `pkill -STOP/-CONT -f mc_app_main` 加手工退出 | 使用已确认精确进程的受管理交接和恢复链。SIGSTOP 可能保留 DDS endpoints，不能把 publisher count=2 直接当作两路仍在写，也不能把一次 echo 超时当作已证明单 writer |
| 高 | 松 W “参考冻结 -> 站住”，Q 退出输入等于安全停止 | 冻结摆腿帧不是停止步态。释放键发送零意图，走 Planner 停止链；输入进程死掉也必须触发意图 watchdog |
| 高 | 先 ARM/BLEND/LIVE，再启动 policy | 这没有验证 policy 接管中的真实过渡。旧 garment proxy 不是新 Planner 的启动状态机；不要混接两套工作流 |
| 中 | 用 `ss -lun` 查 ZMQ 5555/5556/5557 | 这些是 TCP，检查 `ss -ltnp`；UDP 51234 与 TCP RPC 51237 分开核验 |

本仓库源码证据（相对仓库根）：

- `script/sonic_x2_transfer_v2/GR00T_audit/gear_sonic_deploy/src/x2/agi_x2_deploy_onnx_ref/src/x2_deploy_onnx_ref.cpp`：第 211 行默认 5 秒；第 2150-2160 行构造时清理旧 trigger。
- `script/sonic_x2_transfer_v2/GR00T_audit/docs/DECISIONS.md`：D-002 记录本机 500 Hz 状态拥堵和 250 Hz 结果，D-005 记录 torso IMU 到 pelvis 的重建原因。这些不能仅因换 ONNX 就丢掉。

这些是已保存代码的结论，不是对学弟当前机器人安装的身份确认。

## 4. 固定完整模型契约

| 工件 | 本阶段取值 |
| --- | --- |
| Pose actor | `x2_sonic_s1ft16000_g1.onnx`；MD5 `5ae9313443068d0fdebdb93480d23746`；1670 输入、31 输出 |
| Planner | HF `tinkerbuggy/sonic-x2/kplanner_g1core/x2_planner_template_s1d_a05_ws.onnx`；记录下载 revision 和 SHA-256；按目录约定提供 `x2_planner_template.onnx` |
| Planner profile | `gear_sonic/config/kplanner_profiles/g1core_bare_v1.env` |
| Plant | `vendor_20260823`；核验实际加载的 YAML、生成表和 SHA-256 |
| Anchor orientation | `X2_ANCHOR_ORI_MODE=body`；用 `gear_sonic/scripts/onnx_obs_contract.py MODEL.onnx --require body` 核对模型 metadata |
| Tuning | 固定提交的 `trained_gains_s0.yaml`，不要拿旧卡片覆盖，也不要误选默认 `trained_gains_s0_inc_waistmc.yaml` |
| Whole-body | 本阶段显式 `X2_WHOLE_BODY_TELEOP=0`；不启 token service、Pico sender、衣服和 offload |
| 手腕 | 保持作者 frozen-core 路线的冻结处理；核验所用新 binary 的实际参数名，不拼旧 CLI |

旧 p3 的模型 SHA-256 是 `8ccc42a82ea2c446aa708aece58c7a638ea943a79ed88eeab259669e641271e9`。新旧 actor 都是 1670 -> 31，**形状相同不证明物理契约相同**：旧 ankle-pitch action scale 约 0.63155，新 vendor 表为 0.25788763。错误表不会通过维度错误自动暴露。

第二阶段全身工件可以一并取下登记，但不作为首次键盘行走的运行依赖：

| 文件 | 作者公布的 MD5 |
| --- | --- |
| `x2_sonic_s1ft16000_g1_token.onnx` | `438a14fea8849b1feaa770e4e9de03c5` |
| `x2_smpl_tokenizer_v11release.onnx` | `ee03f06e0211590ece8eff3185efba98` |

上述 MD5 来自作者工件清单，不是本轮下载实测；本项目正式 manifest 还应记录 SHA-256。[模型说明](https://github.com/meetsitaram/GR00T-WholeBodyControl-X2/blob/c0036a457dd570b41eaadfa9b4c2115db9df551a/MODELS.md)。

配置的两个陷阱：

- `trained_gains_s0.yaml` 注释称 unity，但固定文件实际包含 `kp_scale_waist_pr=2.00`、`kd_scale_waist_pr=5.51`。记录精确文件和解析结果，不写成“全部倍率 1”。
- frozen env template 和 bare profile 均使用 `:=`，已有环境变量会胜出。模板含 brake pulse 0.55 s，而 bare profile 设 0；加载顺序会改变结果。仿真和真机保存相同的最终 `KPLANNER_*`/`X2_*` 有效值，不能只记一个 profile 名。

## 5. 新旧运行时不能拼接

审查的作者 C++：

```text
gear_sonic_deploy/src/x2/agi_x2_deploy_onnx_ref/src/x2_deploy_onnx_ref.cpp
gear_sonic_deploy/src/x2/agi_x2_deploy_onnx_ref/src/aimdk_io.cpp
```

| 项目 | 我们旧 runtime | 作者固定提交 | 本次迁移要求 |
| --- | --- | --- | --- |
| HAL writer | launcher 显式 250 Hz | writer timer 硬编码 2 ms，即 500 Hz；无 `--writer-hz` | 新候选保留本机可配置 250 Hz 的能力，测真实频率和状态年龄；不能给不支持的 binary 传旧 flag |
| 状态 | keep_last(1)、Reentrant、控制/writer 隔离 | Reentrant，SensorDataQoS depth 10 | 移植本机已验证 IO 适配，检查五组反馈 freshness 和控制 p95/p99 |
| IMU | reconstructed pelvis | AimdkIo 直接写入 IMU orientation/omega 为 base | 根据新训练 root 和本机传感器位置核对重建方向/腰链/角速度，不能默认 raw torso 就是 pelvis，也不能叠加两次变换 |
| 生命周期 | `GROUND_LOAD_HOLD / load / policy / stop / lifted` | 另一套控制和交接状态机；没有旧 `--suspended-start`、`--supported-policy-probe` | 以一个明确的新候选实现完整启动/停止契约，不将旧操作口令当作新版已有功能 |
| 禁止发命令 | 旧链路能禁用 command publisher 创建 | 作者 `--dry-run` 只将增益归零，仍发送 HAL；`--no-publish` 才禁止 writer 发消息 | 只读阶段使用经验证的禁止发送路径；不能凭 dry-run 名称并行接官方 MC |

即使 `--no-publish`，作者 AimdkIo 仍会注册四个 command publisher 端点；图上可以出现两个端点而自定义流量为零。本机可复用旧版 `enable_command_publishers=false` 语义，使真正只读时不创建端点。验收同时看**端点身份**与**实际消息**，不混淆它们。

作者 F10 的部分说明称 plant 为 compile-time；同一固定提交 C++ main 已会解析 `X2_PLANT/X2_PLANT_YAML`，并打印 `[plant] RUNTIME PLANT LOADED`。最终以实际代码、启动日志及生效表为准。必须 rebuild 的是改过的 C++/头文件，不能用一句“编辑 YAML 即可”覆盖整个构建链。

MC 交接保留旧项目经验：停止官方 command 写入，但保持 HAL feedback。不要因为 F10 提供了 stop_app 流程，就盲目套到本机历史 v0.9.0-rc7 / aimdk_msgs 0.8.18；旧试验发生过停 MC 后反馈消失。新 launcher 应先加载候选、待命，证明 MC 静默且反馈在线后才启用唯一 writer；恢复链也必须由同一管理流程负责。

## 6. 键盘应该接在哪里

```text
键盘真实按下/松开 + deadman，20 Hz
    -> planner_cmd :5563
    -> G1-core KPlanner：意图规划、30 fps 轨迹、重采样/停止链
    -> pose :5556，50 Hz
    -> pose watchdog :5558
    -> 新 Sonic pose actor，50 Hz
    -> 本机适配的唯一 HAL writer，250 Hz
    -> X2
机器人 joint state + IMU -> Sonic -> x2_debug :5557 -> Planner/watchdog
```

真机首阶段全部核心闭环放 SoC1；不将 Mac/5060/3588S 加入机器人反馈和 HAL 链路。Linux 工作站只用来事先跑隔离仿真。若键盘 UI 在另一台机器，须另验网络中断/窗口失焦的 deadman 行为。

频率与端口不要靠记忆：

| 环节 | 频率/行为 | 真机端口 | 隔离仿真端口 |
| --- | --- | --- | --- |
| 键盘意图 -> Planner | 20 Hz，Planner SUB bind / 键盘 PUB connect | 5563 | 5663 |
| Planner 模型 | 原生轨迹 30 fps、最多 64 帧；推理异步，不要求 50 次推理/s | 无独立命令口 | 同左 |
| Planner pose PUB | 重采样至 50 Hz，跨 replan 淡化 | 5556 | 5656 |
| watchdog downstream/deploy input | 持续参考流 | 5558 | 5658 |
| Sonic policy | 50 Hz | 无额外端口 | 同左 |
| deploy `x2_debug` PUB | 记录实际 debug rate/age | 5557 | 5659，不是 5657 |
| HAL writer | 我们新候选 250 Hz；作者原版 500 Hz | ROS HAL topics | 隔离 ROS domain 73 |

不要启动旧 `gate_walk`/衣服代理占用 :5556；旧衣服 raw :5555 -> guarded :5556 的拓扑不是上表。一个动作意图源有权输出，键盘与手柄 bridge 不同时持续写互相覆盖的命令。

`planner_cmd` 使用 ZMQ multipart：第一段 `b"planner_cmd"`，第二段 JSON UTF-8，例如：

```json
{"intent":"locomotion","magnitude":"continuous","stick_fwd":0.6,"stick_side":0.0,"stick_yaw":0.0}
```

`stick_fwd=0.6` 是归一化意图，不是 0.6 m/s，也不是 PKL 播放倍率。松键、空格、失焦时持续发三轴全零，由 Planner 完成停止/足对齐/站姿衔接；不要把某一摆腿帧永久冻结。A/D 初期只做单独、小幅转向，侧移/后退/蹲走不作为首次目标。

**意图断流必须单独防护：**上游 `pc2_kplanner_onnx.py --command-watchdog-s` 默认 **0（关闭）**。若键盘进程死掉但 Planner 继续出 pose，下游 pose watchdog 不会知道输入已失效。键盘候选显式启用并验证意图 watchdog；`0.3 s` 可作为待测试初值，不冒充作者已验证值或保证停止完成时间。关闭窗口、键盘进程退出、SSH 断连、窗口失焦均要做仿真故障注入，证明请求 IDLE 后实际能完成停止。

普通 SSH TTY 只有字符字节，不能可靠表达真实 keyup/窗口失焦；不要用系统按键自动重复间隔假装可靠的“按住 W”。新键盘 UI 应提供真实 keydown/keyup、失焦清零和显式 deadman；或先使用下节定时指令完成可重复起停验收，再接交互 UI。Q 是正常请求停止并等待确认后退出，不是通过杀输入进程来替代停止流程。物理急停独立于这条软件链。

## 7. 先不用写键盘 UI，也能验证 Planner 起停

在专用 Linux 仿真工作站操作，不在当前真机控制终端操作。先按作者索引完成 Python、Docker、LFS meshes、motion bank 和 Planner 权重准备。不要把约 130 字节的 LFS pointer 当作 ONNX。

```bash
git clone --branch ship-frozen-core --single-branch https://github.com/meetsitaram/GR00T-WholeBodyControl-X2.git sonic_x2_v02_upstream
cd sonic_x2_v02_upstream
git checkout c0036a457dd570b41eaadfa9b4c2115db9df551a
```

先核对 actor 上述 MD5、Planner SHA-256；在一个干净 shell 中准备以下环境。`MODEL_CACHE` 改为实际、仓库外的绝对路径，目录下须有 `kplanner_g1core/x2_planner_template.onnx`：

```bash
export MODEL_CACHE=/absolute/path/outside/git
export CKPT_ROOT="$MODEL_CACHE"
export X2_WHOLE_BODY_TELEOP=0
export X2_PLANT=vendor_20260823
export X2_ANCHOR_ORI_MODE=body
export SIM_TUNING_YAML="$PWD/gear_sonic_deploy/configs/real_deploy_tuning/trained_gains_s0.yaml"
export SIMSTACK_SONIC="$PWD/gear_sonic_deploy/models/x2_sonic_s1ft16000_g1.onnx"
export PLANNER_MODEL="$MODEL_CACHE/kplanner_g1core"
set -a
source gear_sonic/config/kplanner_profiles/g1core_bare_v1.env
set +a
./gear_sonic/scripts/simstack_local.sh --dry-run
```

确认路径和模型正确后，在**这台无真机控制进程的仿真工作站**启动：

```bash
./gear_sonic/scripts/simstack_local.sh
```

选择直接 `simstack_local.sh` 是因为它会令 Planner `--cmd-bind`，没有手柄也能启动，便于定时命令测试。profile 在上面显式 source/export；只设置 `KPLANNER_PROFILE` 文件名不能代替这一步。若接了手柄，不启用其操控，避免其零意图覆盖测试源。此栈的启动/清理脚本含进程扫描和停止逻辑，所以不能在正在控制真机的同一环境盲跑。

待日志确认模型、plant、watchdog LIVE、仿真物理接近实时，再开另一终端，在同一 checkout 测作者现成的命令脚本：

```bash
SIM_CMD_PORT=5663 .venv/bin/python gear_sonic/scripts/simstack_pad_drive.py walk 2 0.6
SIM_CMD_PORT=5663 .venv/bin/python gear_sonic/scripts/simstack_pad_drive.py idle
```

这是作者的**仿真脚本**，不是给真机直接下 2 秒行走的操作命令；本轮未执行。不要省略 `SIM_CMD_PORT=5663`，默认 5563 会接错口。该脚本能测试协议和正常停止，但本身不能代替新键盘 deadman/watchdog 的故障验收。

记录至少：静止 -> 2 秒前进 -> 停稳，重复起停，小幅转向 -> 停稳；保存 intent、Planner served pose、实际 q/dq、IMU、目标/clip、tick 和物理时间。`mj_forward` 动画或显著慢于实时的 physics 不作为通过。结束只在这台隔离工作站运行：

```bash
./gear_sonic/scripts/simstack_local.sh --stop
```

## 8. 真机前的三个明确代码任务

这三项没有在本轮实现，不写成已经能直接真机跑：

1. **v0.2 deploy 的本机 IO 适配。**以作者新 actor/plant/observations 为基础，迁入可配置 250 Hz、keep_last(1)、回调隔离、禁止创建/发送 command publisher 的只读模式；核对 pelvis 重建、关节顺序/零位/角速度。新旧部署状态机择一实现明确契约，不把两套命令混写。
2. **统一 launcher。**在隔离 install 构建并冻结 binary/model/plant/tuning/profile/有效环境哈希；管理精确 MC PID、命令静默/反馈新鲜度、唯一 writer、正常停止与恢复。包含 Planner/watchdog 启动先于 deploy、pose-stream gate、明确现场操作提示；不覆盖旧 runtime。
3. **键盘意图桥。**替换旧 walk-PKL 发布逻辑，使用本页 multipart 协议和 20 Hz cadence；实现 keyup/失焦/deadman；在 Planner 启动参数启用 command watchdog，完成正常停止和故障注入测试。

最快的首个里程碑是 **同一新候选：受支撑静止、短时前进、松键正常停稳、回到静止保持**。首走现场仍有龙门保护和独立急停操作员，人员不进入摆臂/腿的运动范围；不要用“旁边有人扶着”替代保护。没有进入正确停止状态时不降支撑、不重启输入进程。

正常退出和故障退出区别记录：键盘零意图不是立即失力，不是 kill deploy；承重状态不 Ctrl-C、不关 SSH、不结束 tmux。先完成已验证的停止/静态保持，再由现场操作者完全吊起卸载，按**新 launcher 实际实现**结束唯一 writer 和恢复官方控制。旧 `lifted` 只有迁入并验证后才可使用，不能发给不认识它的上游 binary。异常增长摆动、踮脚、异响、反馈 stale、保护或人员担忧立即执行经验证的停止流程；紧急危险直接物理急停。

原 garment proxy 的终止性 LOCKOUT 不得通过重启规避；作者 pose watchdog 的自动 idle-blend 也不等价于原 proxy。新 Planner 路线要单独记录参考中断的停止和重新使能规则，异常后由人明确解除，不因新 pose 到来自动重启行走。

## 9. 学弟每步交什么

每个阶段一份 README 即可，别新增一串重复审批脚本：

| 阶段 | 交付/通过证据 |
| --- | --- |
| 工件固定 | 上游提交 + actor/Planner/plant/tuning/profile SHA-256 + 最终生效环境；不是只有文件名 |
| 物理仿真 | 同候选 binary，静止/起步/正常停止/转向日志；key release、输入退出和断流测试；无真实 HAL |
| 本机只读 | 实际加载的 binary/model/plant、观测契约、50 Hz tick 和五组 state age；确认没有自定义 command 消息 |
| 受支撑真机 | 支撑状态、起止时间、完整日志、实际 writer 频率、停止行为；最后 MC/HAL 归属 |

若继续写最终现场操作卡，须先拿到学弟那三个脚本和实际 binary/source 身份。不要凭此报告假造“已测试的 WASD 启动命令”。下一步要做的是完成上面三项适配并在仿真起停，而不是重新寻找旧 p3 的某个偶然站姿。

## 10. 参考来源和边界

- 作者项目页及固定提交的 F01/F07/F09/F10、`MODELS.md`、`robot_env.env.frozen.template`、`PORT_REGISTRY.md`。
- 固定提交的 `pc2_kplanner_onnx.py`：`get_next_frame_resampled(OUTPUT_FPS)`、20/50 Hz 输入输出契约、默认关闭的 `--command-watchdog-s`；`simstack_pad_drive.py`、`simstack_local.sh`。
- 固定提交的 C++ `x2_deploy_onnx_ref.cpp`、`aimdk_io.cpp`、vendor plant 表、`trained_gains_s0.yaml`。
- 本项目旧 C++、D-002/D-005 和 [10 月 8 日交接](SONIC_V02_PLANNER_FIRST_HANDOFF_20261008.md)。

原操作卡不修改、不删除；新结论不冒充旧试验的根因证明。本报告和索引只写入本地管理分支，本轮未 commit/push，也未检查新的远端上传状态。
