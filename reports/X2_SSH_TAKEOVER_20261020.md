# X2 SSH 接管记录：2026-10-20

## 交接补充：用户同意现场重启，当前状态未重查

用户已回复“现场重启整机，保持完全吊起”，随后要求更换 Agent 并上传交接资料。
未代用户执行 reboot/HAL restart，未再次连接机器人确认重启完成。
因此下方 22:41/22:43 状态只是**重启前最后观察**，不是接班时当前状态。
接班入口为 `HANDOFF_TASK_CARD_20261020.md`；不要重新执行失败的同一探针。
本次 GitHub 保存包含所有已记录问题的报告、compact 原始证据和脚本；
精确候选 ELF/原始 CSV 仍本地，哈希见 manifest，不声称已网盘备份。

## 最新检查点：22:41 CST / 暂停 powered，厂商发布阻塞

**实际结束状态：**机器人由现场报告保持完全吊起；自定义 deploy/writer 和有限诊断器均已退出。MC PID 18834 存在，但官方日志持续等待六个前置消息，MC 实际命令流及 HAL/IMU 反馈恢复均未验明，不得写成“MC 已恢复正常”。不降支撑、不接地、不推、不继续接管。已询问用户是否现场重启整机；尚未代为重启 HAL/整机。

1. 22:23 第二轮只延长 stop_app -> trigger 等待到 3 s，使用同一精确候选；仍在接管后触发原生 500 ms / 3 ticks stale。说明简单延长交接等待无效，不作为已修复配方。
2. 第二轮 trigger 后约 0.43 s 独立五路反馈同时停收，约 2.5 s 重新收到少量帧；writer command 持续。3 s 探针窗口正确结束 own child（退出 0），MC start_app 回复 success，但随后的恢复验证超时。端点被发现不等于命令/反馈有实际数据。
3. SoC0 / SoC1 新订阅器均未形成收流基线；SoC0 25 s 延长检查也失败。两端原厂 DDS profile、网卡地址已核对，SoC1 -> SoC0 ICMP RTT 0.248/0.319/0.385 ms，0% 丢包。不能据此排除 DDS 故障。
4. SoC0 官方 MC FastDDS 日志反复出现 `RTPS_TRANSPORT_SHM ... NOT Healthy`、移除/重建共享内存端口。官方 IMU 日志记录 `previous Publish() cost: 2000.659 ms`，同时 poll/read 仅约 0.012/0.010 ms；后续曾记录更长发布停顿。因此至少存在**厂商上游 Publish 阻塞**，不是单独的 policy 推理或 deploy callback 饥饿。
5. 两端仅 UDP 的临时 observer 对照均未恢复收流。只改变诊断进程的 profile，不改厂商文件、不启动 publisher、不重启 HAL。不能由此宣布 UDP 已解决，也不能由 SHM 报错宣布已证明唯一根因。
6. 现场新增观察“挥手迈步，但是停下”：作为吊起模型动作记录，结合 stale 后纯阻尼解释；未证明落地平衡或行走。未执行第三轮 powered、落地或键盘测试。

证据：`settle_powered3_*`、`settle_recovery_*`、`settle_vendor_logs.txt`、`settle_hal_libraries_logs.txt`、`udp_only_soc{0,1}.jsonl`、`ending_soc{0,1}_*.txt`。两轮 CSV/runtime 均已下载。原 runtime/model 未覆盖。

本地工具后续仅补了 recovery 超时后仍必做 node destroy / rclpy.shutdown 的 finally 清理；这版**未上传或带电运行**，不得冒充本轮实测脚本。精确候选二进制已下载到 `data/experiments/x2_v02_candidate_20261020/`，核验后登记哈希，以免下次因 `/tmp` 消失再次编译。未 commit/push/网盘上传。

## 22:13-22:14 / 原生 stale 复现，MC 已恢复

- 精确候选 `2f1590369e4574c9e4debe89d8f34df19dbfbc21b93627b676e2038106c26ffa` 实际启用 writer；源码仍为前次同一 250 Hz 单行修改。19 项 Python 测试和 7 项直接 C++ 断言通过。
- stop_app 后约 0.350 s 触发接管；新 actor 进入 CONTROL，约 0.56 s 后触发原生 `500 ms / 3 ticks` stale 并进入终止 SAFE_HOLD 纯阻尼。外加 100 ms 门已不存在。
- **修正初步口头归因：**独立 supervisor 并非全程连续收 state。trigger+0.08 s 后四组 state 和 IMU 同时停止推进，到约 trigger+2.1 s 才恢复；command 接收持续。这不能归为 deploy 内部 executor 的唯一故障，更不能据此盲改 callback。后段独立反馈正常，不代表前段没有空窗。
- 原生 SAFE_HOLD 不再执行 CONTROL 中的 max-duration，因此原计划 3 s 未自动结束，外层 30 s timeout 才终止 writer；退出 124。约 0.56 s 之后是纯阻尼，不是持续 policy。此为本地限时管理缺陷，必须记录，不把本轮说成 3 秒正常退出。
- 恢复 API 返回 success，独立 graph 核验四组各一个官方 MC publisher。现场“无异常，仍完全吊起”。原始日志 `resume_powered3_*`；完整 CSV/runtime 位于 `data/experiments/x2_nopublish20_20261020/resume_powered3/`。
- 下一轮已获现场授权：**只改管理器交接等待**，MC stop 后固定等待 3 s 并按原 0.5 s freshness 验证后触发；不改 candidate binary、模型、PD、filters 或原生门限。同时给完全吊起探针本身设置 trigger 后 3 s 的实际退出窗口，防止 SAFE_HOLD 绕过时长。不能用于承重退出。
- 曾准备 `v02_state_executor_20261020/` 的未修改源文件副本，尚未实施任何 callback patch、未构建或部署该路线；当前实验不以它为候选。

## 最新检查点：22:06 CST / 换电后恢复同一候选

本节覆盖下方 15:34 的实时状态判断。用户已换电，并重新明确确认完全吊起、活动范围无人、独立急停在手、学弟未并行启动控制，授权本轮 3 秒吊起接管；不授权落地或行走。

- 15:54 上次实际接管已进入新 actor CONTROL，但约 0.1 秒 / 6 ticks 后被管理脚本自加的 100 ms freshness 门中止，不是 deploy 原生 500 ms stale 触发，也不能当作模型平衡失败。四组各实收 20 条 custom command。现场反馈“无异常，只有简单动作，挂起”。原始证据 `powered3_supervisor.jsonl`、`powered3_command_gid.csv` 和 `powered3/` CSV 保留。
- 15:56 独立复核四组 command 各仅官方 MC 一个 publisher，自定义进程已退出。随后电池耗尽；该历史状态不当作本次开机状态。
- 按用户要求，当前本地 supervisor 已删除外加的运行期 100 ms freshness 中止；未改 deploy 原生 stale/tilt/action/velocity 限制。preflight 使用原生同量级 0.5 s，仍需单 writer 和交接静默证明。19 项本地工具测试通过，见 `resume_unit_tests.log`。
- 22:05 SSH 已重新认证；SoC1 开机约 4 分钟，无旧 custom deploy/supervisor。原 binary/model SHA256 与前次一致。旧 `/tmp` 隔离 install 已随重启消失，正在持久隔离目录 `/agibot/data/home/agi/x2-v02-writer250-resume.QvhMAD` 恢复同一个仅 writer 2 -> 4 ms 的源文件，不覆盖原 install，不改模型、PD 或映射。
- 本次 powered 尚未执行；当前仅恢复构建及只读核验。候选重构建后必须记录新 binary 哈希，不冒用旧构建哈希。

本轮顺序：3 秒完全吊起接管 -> 汇报实际结果 -> 另行明确地面保持/停止生命周期后接地平衡 -> 独立受控轻扰动 -> 键盘 KPlanner 起停。禁止把本轮仅适用完全吊起的进程退出方式用于承重机器人。

## 最新检查点：15:34 CST / 新模型零发送计算通过

本节覆盖后面的 15:03 状态。15:30:36 加载实际安装 actor `x2_sonic_s1ft16000_g1.onnx`，完成 20 秒 CONTROL 后正常退出 0；使用 `--no-publish`，没有暂停 MC。

- 精确 binary/model 哈希与下文身份一致；加载日志确认 `s1ft_it16000_merged.pt`、输入 1670。plant 使用 `vendor_20260823` 编译表回退，没有伪称加载不存在的 YAML。
- `tick.csv` 1000 行，观测频率 **50.002 Hz**；周期 p50/p95/max **20.000/20.452/23.366 ms**，reason 全为 `ok`，tilt_trip=0。记录的 action 非全零，最大绝对值 `7.109205`。
- 独立 C++ GID 审计 55 秒正常退出 0：发现 deploy 的四组端点，但各组收到的 deploy 消息均为 **0**；官方 MC 四组分别收到 27447/27444/27448/5498 条。端点存在不等于实际发送；这是观察窗口中的实收证据。
- 独立 50 秒 state/IMU 接收通过，四组 state 与 torso IMU 报告点接收年龄最大分别 3.67/3.06/5.21/4.83/5.16 ms。不是每个物理采样的连续最大延迟。
- **15:34** 最终每组 command 只有一个官方 `mc_ros2_node` publisher；MC/HAL/DRP/IMU 原进程运行，deploy 和诊断器均已退出。未请求落地，未更改控制代码、PD、滤波、映射或 writer 频率。
- 温度日志中 motor 29 跳变；独立 SoC1 原始消息也显示 head_yaw motor_temp=55..200、head_pitch 温度/电压均为 0。用户随后现场明确“非真实高温，报错而已”。按**现场说明**记录，不据此宣称传感器根因已修复，不关闭保护。SoC0 温度交叉读取因本机消息 Python 包路径未设置失败，证据保留；这次失败没有触碰控制。

结论：新 actor 已在机器人上实际计算，纯计算负载下未复现 stale；**尚不能证明打开 writer 后稳定、支持自主平衡或键盘行走**。不把 500 Hz 写入说成已排除或已证明根因。

证据：[运行原文](../logs/x2_ssh_takeover_20261020/nopublish20_runtime.log)、[计算统计](../logs/x2_ssh_takeover_20261020/nopublish20_analysis.json)、[GID 审计](../logs/x2_ssh_takeover_20261020/nopublish_command_gid_cpp.csv)、[最终 ownership](../logs/x2_ssh_takeover_20261020/nopublish_final_ownership.txt)。原始六份 CSV 保存在 `data/experiments/x2_nopublish20_20261020/csv/`，远端原件 `/tmp/x2-nopublish.0zrP2v/csv/`，尚未上传/提交。

只读审计器的初版 Python 接口在此 Humble 不支持 callback MessageInfo，运行失败后改为独立 C++ 订阅器；两次链接缺库和最终成功构建记录均保留。没有因此重编译机器人控制程序。可复跑 no-publish 脚本：`script/project_tools/run_x2_v02_nopublish20.sh`，硬锁 binary/model SHA256，绝不使用会发零增益命令的 `--dry-run`。

用户已在新询问中确认“全部就绪，允许短时吊起接管”：完全吊起、四肢范围无人、独立急停就绪、学弟停止并行改动。下一阶段只允许有界吊起接管，不落地、不走；执行结果另记，**此处不提前写成接管通过**。

## 15:03 CST / 连续 MC 暂停诊断（历史）

本节覆盖下方 14:42 的历史待测状态。用户在现场就绪询问后回复“好了”；本轮按完全吊起、运动范围清空、急停就绪执行，未要求下降或改变支撑。

- **2026-10-20 15:02:21.905-15:02:23.910：只执行一次 MC 有界暂停。** SoC0 本机进程发送 STOP 后约 `2.0052 s` 发送 CONT；恢复状态 `S`，诊断退出 0。
- 同一组 SoC0 / SoC1 订阅器在暂停前完成匹配，贯穿暂停和恢复，没有在暂停期换新订阅器。两轮 45 s 采样正常结束，最后一组结束于 **15:03:00 CST**。
- 暂停内部约 1 s 的共同采样区间，四组 command 收到的新消息均为 **0**；SoC1 四组 state 分别增加 `603/568/537/274` 条，torso IMU 增加 `376` 条。
- SoC1 暂停内部两个报告点的四组 state / torso IMU 接收年龄最大 `3.40 ms`；SoC0 对应最大 `17.26 ms`。这是报告点年龄，不冒充连续每帧最大延迟或实际生产频率。
- Joint state 的 header 时间戳持续前进约 1 s，但 `sequence` 停在最后的 MC command 序号；不能把不变的序号单独当作反馈断流。没有采集逐关节实际数值变化，故不以此证明每一条都是新的物理测量。
- 恢复后四组 command 继续收流，最终各一个官方 `mc_ros2_node` publisher；state / IMU publisher 身份未换。MC / HAL / DRP / IMU 原进程均仍运行，无新 custom writer / policy / Planner。
- **本次证明：当前开机、配置正确且先匹配的订阅路径中，暂停 MC 不会必然停止 HAL / IMU 消息。** 不证明带自定义 500 Hz 写入的新版 deploy 没有 stale，也不宣布历史所有断流已经修复。

原始证据：[暂停事件](../logs/x2_ssh_takeover_20261020/soc0_mc_pause_2s.jsonl)、[SoC0 连续采样](../logs/x2_ssh_takeover_20261020/soc0_active_mc_pause.jsonl)、[SoC1 连续采样](../logs/x2_ssh_takeover_20261020/soc1_active_mc_pause.jsonl)、[结构化对齐结果](../logs/x2_ssh_takeover_20261020/mc_pause_analysis.json)。

执行保护：仅以 `boot_id + PID + start_ticks + argv SHA256 + uid` 校验的 MC 目标打开 pidfd；同一本机进程负责 STOP/CONT，不采用可能提前触发的分离恢复定时器。正常异常路径均在 finally 请求 CONT，并检查非 stopped 状态；HUP/INT/TERM 请求尽早恢复。**这不是硬实时或抗 SIGKILL / 内核故障保证，物理支撑与急停仍必需。** 本地测试 `12/12`，SoC0 临时子进程真实 STOP/CONT 自测退出 0，随后才暂停 MC。工具 SHA256：`2e1cca389b9107def936dd253bf0b7f95e7b9ee74c098b24d8e48a9092b1201a`。

前一轮 35 s 连续采样到期后，自动前置校验拒绝执行暂停；MC 未因此被触碰。两份 `*_continuous_mc_pause.jsonl` 原样保留，实际暂停证据是上述 `*_active_mc_pause.jsonl`，不得混用。

### 不是第一次遇到此类症状

[2026-08-30 ledger](../script/sonic_x2_transfer_v2/GR00T_audit/gear_sonic_deploy/README_STAGE_SONIC_BALANCE.md) 的 15:25-15:36 记录：MC 已暂停，自定义 writer 250 Hz，deploy state 初期延迟 1-2 s、峰值约 3.24 s；独立订阅器却持续收流，之后 deploy 不重启自行恢复。旧项目还记录过 500 Hz 写入 / callback 拥堵。因此“反馈 stale”不是新现象，也不能退回“固件天然不支持 MC 暂停”的旧推断。

今天新增的严格证据分两条：只读采集器缺厂商 DDS profile 会完全看不到消息；正确 profile 下，连续匹配订阅器在 MC 暂停期间仍接收消息。这两条没有定位原 powered deploy 的唯一根因。

### 已读取当前新版源码，下一步不再重测 HAL 存在性

只读保存了 SoC1 实际安装树源码和哈希，没有改动：

- `aimdk_io.cpp` SHA256 `b1f31a59dc7d22e165452b532943a7de7b88287c6e4371bc69f294358bc5ce35`。
- `aimdk_io.hpp` SHA256 `f6995dd965766992452bdcd06b0f5c446311b77b5f43ef13c0583b393dd57068`。
- `x2_deploy_onnx_ref.cpp` SHA256 `59132e76a8204624efb43dd586966d998763ed5f67d7de9862b73397e1fda248`。

当前 AimdkIo 使用 `SensorDataQoS` depth 10、独立 Reentrant state callback group，以回调本机 steady time 判 freshness，不以 state sequence 变化判新鲜度。Writer 仍配置 2 ms / 500 Hz；control 与 writer timer 使用默认 callback group。源码存在这些差异，不等于本轮已经证明 500 Hz 是唯一根因；也未证明该源码到当前 binary 的完整构建来源。

**建议下一阶段：同一新 deploy/model/plant/tuning 做明确禁止发命令的计算负载对照，官方 MC 保持运行，独立订阅器同时采样。** 若独立接收正常但 deploy stale，检查 deploy 消费 / executor；若 no-publish 稳定而实际写入才失败，再隔离 writer 负载和交接整合。不放宽 stale 门、不改 gains、不换模型、不直接上行走。本轮完成上述单次暂停诊断后先汇报，没有执行这个下一阶段。

源码摘录与完整哈希见 [日志索引](../logs/x2_ssh_takeover_20261020/README.md)。本轮所有采集器和暂停工具均已退出；SSH 连接不包含自定义控制。

## 14:42 只读基线（历史）

- 当前任务：用户要求直接 SSH 接手，复现 v0.2 键盘 / KPlanner 真机链路；不是继续旧 p3 衣服站立调参。
- 本阶段只做只读诊断。最后一组采样结束于 **2026-10-20 14:42:22 CST**；不能把这个时间点的状态当成以后仍然有效。
- 观察到官方 MC、HAL EtherCAT、IMU 进程运行；四组 command 都只有官方 `mc_ros2_node` publisher。
- 没有运行自定义 deploy、Planner、watchdog、衣服、offload 或 HAL writer；全部有限时长采集器已退出。
- **没有执行 MC pause / stop / resume，没有发 HAL、启动 policy、编译或覆盖运行时，也没有杀机器人进程。**
- 只上传了一个无 publisher / client / signal 的采集器到两台 SoC 的隔离 `/tmp` 目录；没有改厂商 DDS 文件。
- **下一停止点：取得当次完全吊起、运动范围无人、现场独立急停就绪确认，再审核并执行一次有本机限时恢复的连续暂停诊断。当前未执行。**

本记录覆盖旧文档对“当前在线状态”的判断，不废除旧文档的安全、单 writer、证据保存规则。8 月旧 runtime 的 250 Hz / PD / slew 等参数不得直接当作新 v0.2 actor 已验收配方。阶段完成不等于用户批准进入下一带电阶段。

## 接入与工件身份

连接路径：`Mac -> sshuser@100.92.227.62 (Windows) -> 机器人内网`。此前给出的 `100.92.27.62` 不正确。凭据不写入文档。

| 角色 | 地址 / 账号 | 观察 |
| --- | --- | --- |
| Sonic SoC1 | `agi@10.0.1.41` | X2 `X220025BBZ0029`，v0.9.0-rc7，ROS Humble |
| MC / HAL SoC0 | `run@10.0.1.40` | 本次约 14:00 开机；不要复用上次开机的 PID |
| 本地管理仓库 | `/Users/yu/Documents/ChatGPT/X2/humanplus_sonic_x2` | 原有脏工作区保留，未 commit / push |
| SoC1 实际安装 | `/agibot/data/home/agi/gear-sonic` | 没有在这里重新初始化 Git 或覆盖控制代码 |

本轮已实测的文件身份，不代表整套模型 / plant / tuning / launcher 已验收：

```text
SoC1 ws/install/agi_x2_deploy_onnx_ref/lib/agi_x2_deploy_onnx_ref/x2_deploy_onnx_ref
SHA256 0b02df145874c3e4c6cb05ea4778f09b1bae372212c5bf3754f561b64791864a

SoC1 policies/x2_sonic_s1ft16000_g1.onnx
SHA256 7dfdec7cf1b1ac67635c3cd8f541c27b50d917df2e3381a6bb086b605e1959a0

script/project_tools/x2_hal_readonly_probe.py
SHA256 02eeff5175519a0e902da852acb1d285613c244b7ec4b949acd844ab31b5fb2e

script/project_tools/test_x2_hal_readonly_probe.py
SHA256 5295e1dbda4013a5df27a41f27e23d2a2cf30cadc9b506ecf0d4cf8abda96630
```

上游源码审查基准：`meetsitaram/GR00T-WholeBodyControl-X2`，提交 `c0036a457dd570b41eaadfa9b4c2115db9df551a`。不把该提交号冒充当前安装产物的完整来源证明。

## 已验证的反馈归属

保留完整 endpoint 类型、node、GID、QoS，而非从过滤后的 topic 输出猜测：

| Topics | 实际 publisher | 当前结论 |
| --- | --- | --- |
| 四组 `/aima/hal/joint/*/command` | 各 1 个 `mc_ros2_node` | 官方命令正在发布，没有自定义 command publisher |
| 四组 `/aima/hal/joint/*/state` | `hal_ethercat_x21546` | 当前开机由 EtherCAT HAL 发布；不是 `soc0_drp` |
| torso / chest IMU | `soc0_hal_imu2958` | 当前开机由 IMU 节点发布 |
| `soc0_drp` | state 的 subscriber | 不能据此推断暂停 MC 是否会影响 HAL |

观测到的官方命令 position / velocity / effort / stiffness / damping 均为零。不据此认定机器人已经自主站立、处于某个明确 MC 模式，或没有其他硬件状态要求。

SoC0 约 14:39 的进程快照：MC PID `2784`、HAL PID `1546`、DRP PID `1544`、IMU PID `2958`。**这些只是历史观察值，操作前必须重新核验 argv、start identity 和状态。** MC 的 Linux `comm` 是 `mc_main`，argv 才含 `./mc_app_main`；`pgrep -x mc_app_main` 在当前进程上不成立。

## DDS 对照与测量边界

| 采集 | 条件 | 结果 |
| --- | --- | --- |
| SoC0 初次 | 非交互 shell 未加载厂商 DDS 配置 | 没有发现厂商 publishers，接收 0，退出 2 |
| SoC1 基线 | SoC1 厂商 DDS 配置 | 九组必需流全部接收且连续新鲜，有限采集通过 |
| SoC0 14:37 | 显式加载 SoC0 厂商 DDS 配置 | 十组流全部接收，15 s 采集通过，退出 0 |
| SoC0 14:42 对照 | 保持上一组其余环境，仅 unset `FASTRTPS_DEFAULT_PROFILES_FILE` | 十组流接收均为 0，10 s 未建立基线，退出 2 |
| SoC1 14:42 复核 | SoC1 厂商 DDS 配置；MC 未变 | 15 s 通过，四组 command 仍各一个官方 publisher |

因此，**当前采集进程的 DDS 配置差异能导致“看不到反馈”的现象**。这不能证明此前 powered stale 的唯一根因，也没有证明暂停 MC 时反馈持续。未加载配置的采集器收不到命令，不能用来证明 MC 已静默。

SoC1 最后一次 15 s：四组 state 和 torso IMU 最终接收年龄约 `0.95-3.22 ms`，最大接收间隔 `10.53-13.44 ms`；四组 command 最大间隔约 `16.49-18.52 ms`。SoC0 配置正确的采样中 state / IMU 最大间隔约 `61-65 ms`，waist command 曾达 `118.8 ms`。

这些是 **Python 十路订阅、depth=1、单 executor、周期性 graph 查询下的接收统计**，不是 HAL 实际生产频率，更不是新 deploy 的控制时序验收。不能把采集器约 60 Hz 等读数写成 HAL 降频。

六个消息定义的 SHA256 在 SoC0 / SoC1 一致：`JointCommandArray`、`JointCommand`、`JointStateArray`、`JointState`、`MessageHeader`、`DomainErrorState`。原始哈希文件见日志索引。这只证明这些定义一致，不是全部链接库 / typesupport / 运行时 ABI 闭合证明。

## 可复用的只读采集环境

必须使用 `/usr/bin/python3`，不要使用 base Conda。以下只是采集环境，不是启动控制命令。厂商配置含各自主机的网卡地址，不能交叉使用。

SoC0：

```bash
source /opt/ros/humble/setup.bash
export AMENT_PREFIX_PATH=/agibot/software/housekeeper/bin/aimdk_msgs:/agibot/software/v0/common:$AMENT_PREFIX_PATH
export LD_LIBRARY_PATH=/agibot/software/housekeeper/bin/aimdk_msgs/lib:$LD_LIBRARY_PATH
export PYTHONPATH=/agibot/software/housekeeper/bin/aimdk_msgs/local/lib/python3.10/dist-packages:$PYTHONPATH
export ROS_DOMAIN_ID=0
export ROS_LOCALHOST_ONLY=0
export FASTRTPS_DEFAULT_PROFILES_FILE=/agibot/software/entry/cfg/ros_dds_configuration.xml
/usr/bin/python3 /tmp/x2-takeover-readonly.dMqSMc/x2_hal_readonly_probe.py --tag soc0 --seconds 15 --discovery-seconds 10
```

SoC1：

```bash
source /opt/ros/humble/setup.bash
export AMENT_PREFIX_PATH=/agibot/data/home/agi/aimdk_msgs_pc2:$AMENT_PREFIX_PATH
export LD_LIBRARY_PATH=/agibot/data/home/agi/aimdk_msgs_pc2/lib:$LD_LIBRARY_PATH
export PYTHONPATH=/agibot/data/home/agi/aimdk_msgs_pc2/local/lib/python3.10/dist-packages:$PYTHONPATH
export ROS_DOMAIN_ID=0
export FASTRTPS_DEFAULT_PROFILES_FILE=/agibot/data/home/agi/.aima/env/ros_dds_configuration.xml
/usr/bin/python3 /tmp/x2-takeover-readonly.XPVWDB/x2_hal_readonly_probe.py --tag soc1 --seconds 15 --discovery-seconds 10
```

`/tmp` 文件可能随重启消失；重新上传前比较路径和哈希。采集器只订阅，不创建 publisher/client、不发进程信号、不调用 MC API；接收基线通过不是带电接管授权。

本地复核：`python3 -m unittest discover -s script/project_tools -p test_x2_hal_readonly_probe.py -v`，3 项通过。修正了旧采集器“每秒计数不清零”的统计问题；旧 `/tmp/t01_collector.py` 未改写、未复用。

## 先前失败证据如何解释

学弟 T01 证据包 SHA256：`1395c7e4b37c6bcabd6c191a6069c96fadc01679cb2fb769ca890f90bc799550`。原件不删除。

- 首次 powered 记录约 `105 帧 / 2.079921 s`，五组反馈年龄一起涨到约 `0.53 s`；能确认 stale，不能区分源停止、DDS 路径、deploy 消费阻塞。
- 先前 STOP 前 / 中 / 后使用不同新订阅器，暂停期订阅器没有匹配 state endpoint；存在 discovery 混杂，不能作为“HAL 被 MC 关闭”的定论。
- 包内 `kb_deploy_launch.log` 是约 12:48 的第二次待命启动，不是 12:17 的 powered 失败启动。首次完整 launcher / args 仍缺失。
- 不使用报告里的 v2 暂停脚本：限时 CONT 可能早于实际 STOP，且恢复状态未强验证；旧进程名检查也与当前 `comm` 不一致。

## 14:42 的下一步计划（15:02 已完成）

要验证的是：**先在两台 SoC 建立并保持同一组订阅匹配，再短暂暂停精确 MC 进程时，命令能否静默且 joint / IMU 继续更新。** 不启动任何 custom writer / policy / Planner。

前提与验收：

1. 当前完全吊起、运动范围清空、独立急停由现场人员掌握；本轮尚未得到新确认。
2. 两台 SoC 采集器已连续收流，使用各自 DDS 配置；暂停前记录 GID、序列、时间戳和进程身份。
3. 恢复机制运行于 SoC0 本机，限定暂停时长；消除先 CONT 后 STOP 的竞态，绑定 PID 与 start identity，先离线 / mock 测试恢复路径；禁止直接复用旧脚本。
4. 暂停 / 恢复全过程同一订阅器连续采样，区分 endpoint 存在与实际新消息；不以 publisher count 判定 SIGSTOP 静默。
5. 结束后核验 MC 非 stopped、命令恢复、反馈新鲜、四组 command 各一个官方 publisher。任何恢复异常不继续部署，由现场保持吊起并处理。
6. 保存完整日志后先汇报，不自动进入 powered handoff、静止 policy 或行走。

离线 / 只读结论已经形成，不因为暂缺物理确认去启动其他控制实验，也不要求重跑旧模型的 30 / 300 秒站立测试。

## 证据与保存规则

- [本轮原始日志及文件索引](../logs/x2_ssh_takeover_20261020/README.md)
- [订阅采集器源码](../script/project_tools/x2_hal_readonly_probe.py)
- [采集器单元测试](../script/project_tools/test_x2_hal_readonly_probe.py)
- [阶段汇报规则](SONIC_V02_WALK_PROGRESS.md)
- [v0.2 部署审查与模型契约](SONIC_V02_KEYBOARD_WALK_REVIEW_20261019.md)

每个阶段结束立即追加本记录，写清时间、做了什么 / 没做什么、完整证据、实际最终 ownership 与下一停止点。密码、tokens、私钥均不得记录。没有用户上传指令时只保存在本地仓库，不把“已写文件”说成“已同步 GitHub”。
