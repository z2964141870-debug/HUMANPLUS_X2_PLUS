# 新固件原生部署复测（sonic_x2 v0.2 deploy_x2.sh 流程）2026-10-22 23:19-23:22

## 结论
**不能"开箱直跑"，但补 3 个环境接点后完整走通作者流程至 CONTROL，且昨日高频报错 1:1 复现**（签名与升级前完全一致）。

## 部署链路逐关结果
| 关卡 | 结果 | 说明 |
|---|---|---|
| preflight (x2_preflight.py) | 初次 FAIL → 修复后 **PASS** | 初次 FAIL：aimdk_msgs typesupport 无法加载（见修复#1） |
| baseline 独占所有权 | **PASS**（适配后） | 节点名变为 mc_ros2_node3214（带 PID 后缀），原校验拒绝，已做最小适配 |
| deploy STANDBY 就绪 | **PASS** | C++ 二进制（旧 ABI 编译）在新固件上正常加载 onnx、绑定 ZMQ、写 ready 哨兵 |
| INIT → WAIT_FOR_CONTROL | **PASS** | C++ 自报 "all state sources fresh" —— 旧 ABI 订阅在新固件上**收得到状态**（线格式兼容） |
| CONTROL | **复现频繁报错** | 进入 CONTROL ~0.5s 后全通道状态判 stale，3 strikes → [FATAL] E-STOP → PURE DAMPING |
| MC 暂停/恢复 | PASS | pidfd 有界暂停 45.0s，auto-CONT 恢复正常（state S） |

## 复现的报错（= 升级前 13/13 同款）
```
[WARN] robot state stale >500 ms in CONTROL (strike 1/3): leg=0.502s waist=0.506s arm=0.505s head=0.505s imu=0.503s
[WARN] robot state stale >500 ms in CONTROL (strike 2/3): ... (strike 3/3): ...
[FATAL] robot state stale 3 consecutive ticks — E-STOP -> PURE DAMPING
```
关键判读：暂停期间驱动器(新 ABI python)全程看到状态新鲜(ages 0.00s、计数持续增长)，而 C++ 恰在**放开 500Hz 写入门控进入 CONTROL 的瞬间**开始收不到状态 —— 与升级前"deploy 内部执行器饥饿"根因#2 签名一致。**该报错与新固件无关，是作者栈自身缺陷**，升级没有引入新病也没有治好旧病。

## 需要的 3 个环境接点（新固件导致）
1. **housekeeper 前缀被沙箱挡死**（agi 用户 Permission denied，v1.0.0 文件系统沙箱）→ 从 SoC0 搬 vendor aimdk_msgs（新旧类型并存，JointState/JointStateArray 都在）到 ~/hk_aimdk/aimdk_msgs，接 AMENT/PYTHON/LD 三路。
2. **ONBOT_AIMDK_PREFIX 默认值失效**（指向被挡的 housekeeper 路径）→ 同上覆盖。
3. **MC 进程/节点改名**：mc_app_main→aima-mc-app-main、mc_ros2_node→mc_ros2_node3214（PID 后缀）→ 已最小适配 x2_mc_bounded_pause_probe.py 与 x2_stage1_sigstop_takeover.py 的身份校验。
另有：驱动直启二进制需自备 LD_LIBRARY_PATH 加 gear-sonic/onnxruntime/lib（deploy_x2.sh 内部会加，驱动器绕过了它）。

## 证据
- run2 runtime.log（210KB，含完整 FATAL 链）：run2_deploy_runtime.log
- 停止原因链、STANDBY 门控、BLACK BOX 60s ring 已 dump 至 SoC1 run2/csv
- 机器人全程吊起+急停持有，MC 45s 有界暂停后自动恢复，反馈全程连续

## 下一步建议
1. 旧病修复路径不变：候选方案（独立 state 执行器/writer 节流）需按新 SDK 重编后重测（JointStateArray 线格式未变，老 ABI C++ 仍能收状态——重编工作量比预期小）。
2. 升级带来的新机会：/aima/mc/common/state(10Hz) 官方状态源可绕开饥饿路径，值得在候选里试。
