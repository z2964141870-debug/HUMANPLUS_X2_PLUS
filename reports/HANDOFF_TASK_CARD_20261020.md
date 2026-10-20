# X2 Sonic v0.2 接班任务卡

交接日期：2026-10-20，Asia/Shanghai。本文是当前任务入口，覆盖旧文档的实时状态和旧模型路线；不覆盖单 writer、现场急停、承重退出和记录纪律。

## 0. 先知道这六件事

1. 目标是复现作者 **Sonic-X2 v0.2 / s1ft16000 + G1-core KPlanner** 键盘行走，不是继续旧 `p3_8900` 衣服/LoRA 站立调参。路线：吊起接管 -> 接地自主承重 -> 单独轻扰动恢复 -> 键盘起步/松键停稳。
2. 新模型已真实运行：原安装 binary 做过 `--no-publish` 20 s / 1000 ticks，50.002 Hz，无 stale，实收 custom HAL command 为零。但短时 powered 仍未通过，不要要求再跑同样的旧 300 s 站立。
3. 最新阻塞是 **反馈/发布链路失效**：powered 后五路反馈出现秒级空窗，触发上游原生 500 ms / 3 ticks stale -> SAFE_HOLD 纯阻尼。不是已经证明模型不能平衡。
4. 厂商 IMU 日志证明 `Publish()` 曾阻塞约 2000 ms，poll/read 仅约 0.01 ms；MC FastDDS 反复报告 SHM port `NOT Healthy`。**这不是已证明唯一根因或已修复结论**。不要直接改 callback、PD 或放宽 stale。
5. 最后检查无 custom writer；MC 进程存在但收不到前置反馈，实际恢复未通过。用户随后同意**由现场重启整机、保持完全吊起**。交接时没有重新连机器人，**重启是否完成和当前状态未知**，不能复用旧 PID 或现场授权。
6. 用户明确要求快，不增加比固定上游更严格的算法门限；外加 100 ms runtime freshness 中止已删除。保留原生限制、单 writer、完全吊起交接和独立物理急停。一步完成立即报告/记 README，不自批进入落地或行走。

## 1. 仓库与阅读顺序

管理仓库：`https://github.com/z2964141870-debug/HUMANPLUS_X2_PLUS`，分支 **humanplus_sonic_x2**。

本机：`/Users/yu/Documents/ChatGPT/X2/humanplus_sonic_x2`。文件分 `data/`、`logs/`、`script/`、`reports/`。保留脏工作区，不 reset/clean/reformat。不要在 SoC1 同步目录初始化 Git。

先读：

| 文件 | 用途 |
| --- | --- |
| `AGENTS.md` | 当前规则、历史文档适用范围 |
| 本任务卡 | 当前任务、接入、停点、优先排查顺序 |
| `reports/X2_SSH_TAKEOVER_20261020.md` | 今日完整时间线；以最上面的新检查点覆盖后面历史状态 |
| `reports/SONIC_V02_KEYBOARD_WALK_REVIEW_20261019.md` | 作者部署文档入口、完整模型契约、参数、端口和启动脚本陷阱 |
| `reports/SONIC_V02_WALK_PROGRESS.md` | 分阶段汇报纪律；初版“未派发”表是历史，不代表今天没做过 |
| `logs/x2_ssh_takeover_20261020/README.md` | 成功/失败原始证据索引 |
| `data/manifests/x2_takeover_local_artifacts_20261020.tsv` | 未入 Git 的本地二进制/原始 CSV 哈希；未备份项明确列出 |
| `reports/SONIC_V02_PLANNER_FIRST_HANDOFF_20261008.md` | 新 actor/Planner 谱系与后续全身路线，不是当前 live-state |

旧安全/经验文件位于 `script/sonic_x2_transfer_v2/GR00T_audit/AGENTS.md` 及 `docs/{CURRENT_STATE,DECISIONS,OPERATIONS,EXPERIMENTS}.md`。它们的 **8 月 PID、MC 状态、旧模型、neutral_damped gains、重建 pelvis 配方、入口角度、0.12 rad/s 和 30/300 s 验收不能直接套到 10 月新模型**。可读作历史，承重退出和单 writer 原则继续有效。

## 2. 如何访问机器人

网络：Mac -> Windows 桥接机 -> X2 内网。**正确桥接 IP 是 100.92.227.62，不是 100.92.27.62。**

| 主机 | 账号/地址 | 角色 |
| --- | --- | --- |
| Windows 桥接机 | `sshuser@100.92.227.62` | 只做跳板；不要直接对 Windows shell 执行 Linux bash 命令 |
| SoC1 / 作者 PC2 | `agi@10.0.1.41` | Orin，Sonic/Planner；不是 3588S |
| SoC0 / MC/HAL | `run@10.0.1.40` | 厂商控制、EtherCAT、IMU |
| 3588S | `agi@10.0.1.42` | 旧衣服 offload；本轮键盘路线不需要 |

**凭据不入 Git，也不要从日志猜密码。新 Agent 向用户单独取得桥接机和两台 SoC 凭据；交互输入，不写进命令、环境导出记录或脚本。**

先只读检查本机端口是否已有可复用 tunnel；不要杀未知 SSH 会话。没有 tunnel 才在独立管理终端建立：

```bash
ssh -N -o ExitOnForwardFailure=yes -o ServerAliveInterval=10 \
  -L 127.0.0.1:22041:10.0.1.41:22 \
  -L 127.0.0.1:22040:10.0.1.40:22 sshuser@100.92.227.62
```

再登录：

```bash
ssh -p 22041 -o HostKeyAlias=x2-soc1-via-bridge agi@127.0.0.1
ssh -p 22040 -o HostKeyAlias=x2-soc0-via-bridge run@127.0.0.1
```

前一 Agent 用过的 master sockets：`/tmp/x2-takeover-soc1-20261020.sock`、`/tmp/x2-takeover-soc0-20261020.sock`。只可先 `ssh -O check` 核验；电池重启后不能假定仍有效。SCP 使用 `-P`，不要误用 `-p`。

## 3. 精确运行工件，别重新找模型

SoC1 已安装根目录：`/agibot/data/home/agi/gear-sonic`。

| 工件 | 路径/身份 |
| --- | --- |
| 原 binary，未覆盖 | `ws/install/agi_x2_deploy_onnx_ref/lib/agi_x2_deploy_onnx_ref/x2_deploy_onnx_ref` |
| 原 binary SHA256 | `0b02df145874c3e4c6cb05ea4778f09b1bae372212c5bf3754f561b64791864a` |
| 当前 actor | `policies/x2_sonic_s1ft16000_g1.onnx`，1670 输入/31 输出，加载日志来源 `s1ft_it16000_merged.pt` |
| actor SHA256 | `7dfdec7cf1b1ac67635c3cd8f541c27b50d917df2e3381a6bb086b605e1959a0` |
| 250 Hz 候选 install | `/agibot/data/home/agi/x2-v02-writer250-resume.QvhMAD/install/agi_x2_deploy_onnx_ref/lib/agi_x2_deploy_onnx_ref/x2_deploy_onnx_ref` |
| 此候选 binary SHA256 | `2f1590369e4574c9e4debe89d8f34df19dbfbc21b93627b676e2038106c26ffa` |
| 本机候选 binary 副本，非 Git | `data/experiments/x2_v02_candidate_20261020/x2_deploy_onnx_ref`，2202424 bytes，同 SHA256 |
| 候选 C++ 源码，入 Git | `script/project_tools/candidates/v02_writer250_20261020/x2_deploy_onnx_ref.cpp` |
| 候选源码 SHA256 | `9374fd8cf232fe0ee25142b6dcec844ae72d6221a20fb5e31d056fb98c506ef7` |
| 原源码 SHA256 | `59132e76a8204624efb43dd586966d998763ed5f67d7de9862b73397e1fda248` |

**候选只改 writer timer 2 ms -> 4 ms。**没有迁入旧代码的 pelvis 重建、depth=1 或专用 state executor；不能声称这些已部署。当前 AimdkIo 是 SensorDataQoS depth=10、Reentrant state callback group、与其他回调同一个 MultiThreadedExecutor。原始 IO 源码在日志快照中，模型/plant/观测契约仍需按新版本理解。

早先 `/tmp/x2-v02-writer250.VMUMfO/` 已在换电后消失，它的 binary SHA256 `da53bd4b...` 不是当前产物。新候选在持久目录，不应无故重编译；先验证路径/哈希，缺失可用本机精确副本恢复到新隔离目录。二进制不在 GitHub，远程 Agent 无本机文件权限时应先请用户传递此副本或定向重建，不能谎称 Git clone 就有 binary。

机器已观察为 X220025BBZ0029、v0.9.0-rc7、ROS Humble、aimdk_msgs 0.8.18；重启后重新核对，不升级固件。

## 4. 操作部署参照哪些上游文件

固定上游：`meetsitaram/GR00T-WholeBodyControl-X2`，提交 `c0036a457dd570b41eaadfa9b4c2115db9df551a`，不是任意浮动 main。

- `docs/x2/F10_real_robot_colcon.md`：PC2 编译、同步、bringup、MC/自定义控制交接主文档。
- `docs/x2/F07_frozen_g1_model.md`：s1ft16000 frozen G1 actor/Planner 配套。
- `docs/x2/F01_gamepad.md`：Planner 意图、作者手柄流程；键盘应替换输入设备而非播放 walk PKL。
- `docs/x2/F09_mujoco_sim.md`：真实物理闭环，不是 qpos 动画。
- `x2_pc2/PORT_REGISTRY.md`：端口归属。
- 本仓库 `reports/SONIC_V02_KEYBOARD_WALK_REVIEW_20261019.md` 有上述固定提交链接及已知陷阱，无需重新网上乱找。

启动身份保持：`vendor_20260823` plant、`X2_ANCHOR_ORI_MODE=body`、`X2_WHOLE_BODY_TELEOP=0`、`trained_gains_s0.yaml` 语义、Policy 50 Hz、本机候选 HAL 250 Hz。

plant 本次日志实际使用编译表 fallback，**没有加载不存在的 YAML**。trained_gains_s0 注释说 unity，但实际 waist Kp 倍率 2.00 / Kd 5.51；完整实际 argv 在 supervisor JSONL 和 `x2_v02_suspended_probe.py`，不要凭注释改成全 1。

`start_x2_deploy_ritual.sh` 有错误默认模型/路径、auto-confirm、watchdog/wrist 配置差异；`deploy_x2.sh` 的 HTTP 成功后触发不等于证明 MC 实际静默。原件已保存，**不要直接盲跑它们**。这是本机适配任务，不代表弃用作者架构。

## 5. ROS 环境：非常容易再次踩坑

SoC1 使用系统 `/usr/bin/python3`。不要使用 base Conda，也不要 source 未审计的 captured env 文本。

```bash
source /opt/ros/humble/setup.bash
export AMENT_PREFIX_PATH=/agibot/data/home/agi/aimdk_msgs_pc2:$AMENT_PREFIX_PATH
export LD_LIBRARY_PATH=/agibot/data/home/agi/gear-sonic/onnxruntime/lib:/agibot/data/home/agi/aimdk_msgs_pc2/lib:$LD_LIBRARY_PATH
export PYTHONPATH=/agibot/data/home/agi/aimdk_msgs_pc2/local/lib/python3.10/dist-packages:$PYTHONPATH
export ROS_DOMAIN_ID=0 ROS_LOCALHOST_ONLY=0
export FASTRTPS_DEFAULT_PROFILES_FILE=/agibot/data/home/agi/.aima/env/ros_dds_configuration.xml
unset X2_PLANT_YAML
export X2_PLANT=vendor_20260823 X2_ANCHOR_ORI_MODE=body X2_WHOLE_BODY_TELEOP=0
export X2_REPO_ROOT=/agibot/data/home/agi/gear-sonic
```

SoC0 对应环境：

```bash
source /opt/ros/humble/setup.bash
export AMENT_PREFIX_PATH=/agibot/software/housekeeper/bin/aimdk_msgs:/agibot/software/v0/common:$AMENT_PREFIX_PATH
export LD_LIBRARY_PATH=/agibot/software/housekeeper/bin/aimdk_msgs/lib:$LD_LIBRARY_PATH
export PYTHONPATH=/agibot/software/housekeeper/bin/aimdk_msgs/local/lib/python3.10/dist-packages:$PYTHONPATH
export ROS_DOMAIN_ID=0 ROS_LOCALHOST_ONLY=0
export FASTRTPS_DEFAULT_PROFILES_FILE=/agibot/software/entry/cfg/ros_dds_configuration.xml
```

两个 XML 白名单分别为 10.0.1.41 和 10.0.1.40，不能互换；未加载 DDS profile 曾实测全部采集为零。临时 UDP-only observer XML 只用于进程级对照，不是已通过部署配方；不要覆盖厂商 XML。

## 6. 已做过什么，不要重走错误路线

| 事项 | 证据与边界 |
| --- | --- |
| 15:02 MC SIGSTOP/CONT 2.005 s | 连续同一订阅器贯穿，MC 命令静默、HAL/IMU 仍更新；不等于 EM stop_app 加 custom writer 已通过 |
| 15:30 新 actor no-publish20 | 1000 ticks / 50.002 Hz，零 custom command，无 stale；仅计算通过 |
| 第一份 250 Hz 隔离构建 | 7 项直接 C++ 断言 all OK；CTest wrapper 因不生成 GTest XML 报失败，不能写 CTest PASS |
| 15:54 powered | 约 6 ticks / 0.1 s 后被自加 100 ms 外部 freshness 中止；该门已删除。不能当模型失败 |
| 22:13 powered | 原生 stale；独立五路也约 2 s 无数据，不能只归因 deploy callback。SAFE_HOLD 绕过 native max-duration，直到外层 30 s timeout；管理缺陷已记录 |
| 22:23 延长 handoff 等待到 3 s | 同 binary 仍 stale；最多 3 s 探针退出成功，但 MC 实际恢复未通过。不要继续靠延长等待试运气 |
| 厂商日志 | IMU Publish 约 2000 ms 阻塞；MC SHM port unhealthy 重建，MC 等六个依赖。是最优先线索，不是唯一根因定论 |
| 故障后只改 observer 为 UDP-only | 两端都没恢复；未改厂商配置，不等于 clean boot 下 UDP 路线无效，也不等于已修好 |
| 现场观察 | 悬空有挥手/迈腿然后停下，无异常声响，保持吊起；不是落地行走成功 |

其他容易误判的点：

- joint state 的 `sequence` 可停在最后一条 MC command 的序号，timestamp 和接收仍推进；**不以 seq 不变单独判断流**。
- `--dry-run` 在这个新 binary **仍发布零增益 HAL command**；官方 MC 并行时必须用 `--no-publish`。后者仍注册 4 个 publisher endpoints，但实收发送可以为零。
- 因此 publisher count、实际消息 GID、进程存在是三个不同证据。SIGSTOP 留 DDS endpoint 不等于两路仍写；MC start_app success 不等于恢复收流。
- Humble Python MessageInfo/GID 版本不兼容：`x2_command_gid_audit.py` 是失败历史；实际使用 C++ `x2_command_gid_audit.cpp`。其 official-only 总结果在 powered 看到 custom 时报 FAIL 是预期，要分阶段分析，不能据此说双 writer。
- 头部 motor 29 温度字段异常。用户现场明确不是实际高温；保留事实，不继续把它当主阻塞，也不删除原生保护或宣称传感器根因已修复。
- SoC0 MC 的 Linux comm 是 `mc_main`，argv 含 `mc_app_main`。不要用旧 PID，或宽泛 `pkill -f`。
- 尚未建立新 runtime 的承重 stop/静态保持生命周期。不要把旧 `load/policy/stop/lifted` 当新 binary 已有命令，也不要把当前 suspended probe 用在地面承重。

## 7. 证据在哪里

以下都在 `logs/x2_ssh_takeover_20261020/`，保留失败，不修改原文：

- `mc_pause_analysis.json`、`soc{0,1}_active_mc_pause.jsonl`、`soc0_mc_pause_2s.jsonl`：SIGSTOP 诊断。
- `nopublish20_runtime.log`、`nopublish20_analysis.json`、`nopublish_command_gid_cpp.csv`：计算通过。
- `powered3_*`：15:54 外加门中止；`powered3_supervisor_original.py.txt` 是那次实际脚本。
- `resume_powered3_*`：22:13 原生 stale 和 30 s timeout；`resume_powered3_runtime.log` 为完整 runtime。
- `settle_powered3_*`：22:23 3 s 交接等待无效；`settle_powered3_runtime.log` 为完整 runtime。
- `settle_vendor_logs.txt`：MC 等待依赖与 SHM unhealthy；`settle_hal_libraries_logs.txt`：IMU Publish 阻塞。
- `settle_recovery_*`、`udp_only_soc0.jsonl`、`udp_only_soc1.jsonl`：恢复未通过的负证据。
- `ending_soc0_vendor_state.txt`、`ending_soc1_state.txt`：重启前最后观察；不能当重启后当前状态。
- `writer250_source.diff`、`resume_identity_tests.txt`、`resume_writer250_build.log`：单行修改/哈希/构建。

厂商原始日志在 SoC0 `/agibot/log/latest/{mc,hal_imu,hal_ethercat}/`。`latest` 会随重启变；重启前这轮为 `/agibot/data/log/log_20261020_220103/`，优先用该历史路径保留证据。当时 `mc/18834_fastdds.log`、`mc/mc.log`、`hal_imu/hal_imu.log` 有关键信息。

## 8. 工具版本和未完成事项

`script/project_tools/x2_v02_suspended_probe.py` 当前本地版：已删除外加 runtime freshness 门，含 3 s handoff settle、3 s suspended window、恢复失败仍清理 ROS 的 finally；**最后 cleanup 版只测了 22 项本地单元测试，未上传/真机验收**。

此本地版 SHA256 为 `5746e7d1aabcaad35333d86c785c1467a9e635914004d27bd66167b02a152201`；交接整理时再次运行同组 22 项测试全部通过。这里“未上传”指未同步到机器人，不是缺少 GitHub 源码。

远端前一次脚本 `x2_v02_suspended_probe.py` SHA `eebc79559fce56a71a85bf142e3acf34a65b6659f756646f4487a33916659658`；第二次 `x2_v02_suspended_probe_settle.py` SHA `cc2f087b6216623a0bcb5d68a6cc84d075725f7408d7a0460f67b5766d3d8762`。不要把 repo 最新版说成实际执行版。

`script/project_tools/candidates/v02_state_executor_20261020/` 只有未修改的源码副本：**未实施 executor 隔离、未构建、未部署**。这不是另一个通过的 controller。

## 9. 接班第一轮任务与后续停点

当前只派第一轮：**重启后的只读恢复核验 + 收敛发布阻塞**。不要直接再跑相同 powered probe。

1. 先让用户确认现场重启是否完成；建立/复用 SSH tunnel，仅查本次 boot/uptime、官方进程、command ownership、反馈。不要从这份卡推定任何当前状态。
2. 尽早保全本次官方 FastDDS/IMU 日志；比较“仅厂商运行 -> 加入只读 observer”前后是否已有 Publish 卡顿。必要时从 clean boot 用 UDP-only observer 做进程级对照，而非在已故障系统上继续试等待。
3. 分清：发布源阻塞、DDS transport/SHM、订阅调度。使用实际接收、官方 Publish 用时及双 SoC 对齐，不把后段恢复掩盖前段空窗。SHM 不健康原因、触发条件和具体修复仍待查。
4. 不清空 `/dev/shm`，不擅自重启 HAL/升级固件/覆盖厂商 XML，不同时启动第二个 writer；涉及通信栈重启必须说明失力风险并由现场保持吊起、取得该次授权。
5. 第一轮交付：一句话根因范围、已排除什么、唯一下一步；附精确文件/哈希/日志和最后实际 ownership。不得因为需要汇报而杀承重 writer。

后续另行派发：通信链通过后同精确候选吊起接管；验证新 runtime 的承重停止保持契约后接地自主承重；真正自主稳定后单独测试轻扰动；最后键盘 KPlanner 起停。不要把悬空动作或吊带承重写成自主站立。

键盘后续拓扑：真实 keydown/keyup/deadman 20 Hz -> `planner_cmd :5563` -> G1-core KPlanner -> pose 50 Hz `:5556` -> 作者 watchdog `:5558` -> Sonic 50 Hz -> HAL 250 Hz；debug `:5557`。当前未完成此真机拓扑验收，不接衣服/SMPL/旧 walk PKL/Mac 推理。

## 10. 本次保存范围

用户已授权将交接资料推到现有 GitHub 分支。本卡、脚本、compact logs、runtime 文本及哈希索引纳入本次提交。actor 权重、候选 ELF、原始 joint/action CSV 不入 Git；后两类已保存在本机并列 manifest，**没有声称上传百度网盘**。不要把“交接文档上传”说成“所有大工件备份完成”。

若新 Agent 不在这台 Mac 上，从 GitHub 能得到排障证据和源码，但获取权重/精确候选仍需机器人持久目录或用户提供的本地副本。凭据由用户在新会话私下补充。
