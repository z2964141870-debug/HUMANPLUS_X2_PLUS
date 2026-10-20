# 第一轮（2026-10-20 晚）重启后只读核验与发布阻塞收敛

接班 Agent 第一轮证据目录。范围：重启后只读确认官方反馈与命令归属 + 结合当日全部
boot 的厂商日志收敛发布阻塞根因范围。本轮**未**触碰任何机器人进程、未发任何命令、
未执行 stop_app/接管；全部操作为 SSH 读取、日志 grep 与两个 15 秒只读订阅 observer。

## 现场与会话

- 用户确认现场已完成整机重启并保持完全吊起；本目录所有状态均为 23:1x 实测，不引自旧记录。
- 复用既有隧道（Mac PID 98540，127.0.0.1:22041→10.0.1.41:22、22040→10.0.1.40:22）。
- 本轮认证会话 ControlMaster：`/tmp/x2-round1-20261020/soc{1,0}.sock`（4h，密码仅经 expect 交互输入，未写入任何仓库/脚本）。
- 远端只读工具目录：SoC1 `/tmp/x2-r1-obs-soc1/`、SoC0 `/tmp/x2-r1-obs-soc0/`（探针 SHA256 与仓库一致，02eeff…）。

## 重启后状态（23:18 实测）

| 项 | SoC1 (agi@10.0.1.41) | SoC0 (run@10.0.1.40) |
| --- | --- | --- |
| 开机 | 2026-10-20 22:48:52 CST | 2026-10-20 22:48:36 CST |
| 内核 | 5.15.148-tegra, Ubuntu 22.04.5 | 6.1.99-rt36 |
| 官方进程 | soc1_drp 2061、hal_sensor 3191 | mc_main 2547（argv mc_app_main, cfg lx2501_3_t2d5/mc.yaml）、hal_ethercat 1572、soc0_hal_imu 3057、soc0_drp 1570、hal_pmu 2844、hal_485 2968 |
| 自定义进程 | 无（deploy/supervisor/planner 均无） | 无 |
| 5555/5556/5557/5558/5563 监听 | 无 | — |
| 模型哈希 | x2_sonic_s1ft16000_g1.onnx = 7dfdec7c…59a0（与交接一致） | — |
| 原 binary 哈希 | x2_deploy_onnx_ref = 0b02df14…864a（与交接一致） | — |

SoC0 ssh host key 在重启后变化（已知 known_hosts 旧条目已清理，新指纹已接受）。

## 命令归属与反馈（双端 15 秒只读 observer，探针 02eeff…）

- 四组 `/aima/hal/joint/{leg,waist,arm,head}/command`：**每組恰 1 个 publisher，均为官方 mc_ros2_node（GID 010f39c2…）**，双端一致；无任何自定义 writer。
- 十路流全部新鲜、baseline 通过。SoC1 端 rate：cmd_leg 483.5 Hz / state_leg 495.6 Hz / imu_torso 319.6 Hz（age 0.03–2.9 ms）。SoC0 端受 Python 采集上限约 78–98 Hz，age ≤ 9.6 ms。
- 两轮 observer 前后 SoC0 厂商日志**均无新增** Publish 卡顿或 SHM 警告 → 只读 observer 不是触发因素。

## 发布阻塞考古（当日五个 boot 全量，SoC0 厂商日志）

| Boot | 事件 | hal_imu Publish cost 行数 / 量级 | MC SHM NOT Healthy |
| --- | --- | --- | --- |
| 09:30 | 纯厂商上午（无自定义代码） | 12 行瞬态 ≤34 ms / 2.5 h | 0 |
| 11:47 | 12:17 学弟 powered → 原生 stale | **85 行爆发（12:17–18），max ≈2005 ms** | 0 |
| 11:47 | 12:48 学弟第二次待命启动 | **持续恶化 272 行（12:48–54），max ≈2010 ms；13:17 后同 boot 内自行清零** | 0 |
| 14:00 | 15:02 MC SIGSTOP 2 s（无 writer） | 9 行 ≤37 ms | 0 |
| 14:00 | 15:30 候选 no-publish 20 s（MC 运行，零命令） | 2 行 ≤58 ms | 0 |
| 14:00 | 15:54 powered3（stop_app+writer，0.1 s 外部门中止） | **18 行，max ≈2003 ms** | 0 |
| 22:01 | 22:13 resume powered（stale→SAFE_HOLD→30 s 外层杀） | 47 行爆发，max ≈2052 ms，之后清零 | 0（22:24 前） |
| 22:01 | 22:23 settle powered（3 s 等待；恢复失败） | **持续 1971 行至关机（24 min），max ≈75618 ms** | **42 次，全部在 22:24:15 之后** |
| 22:48 | 本次 clean boot + 双端 observer | 5 行开机瞬态 ≤20.5 ms，40 min 内零新增 | 0 |

上表修正了本轮早期一条错误读数（"14:00 boot 零卡顿"）：那是一条本地变量展开 bug 造成的空输出，以 `imu_shm_archaeology.txt` 为准。

## 结论边界（证据能支撑的范围）

1. **~2 s IMU Publish 阻塞与 powered 接管触发强相关**：四次接管（12:17/15:54/22:13/22:23，两套不同自定义 runtime、三个 boot）全部出现；纯厂商运行、MC SIGSTOP、no-publish 计算、只读 observer 全部不出现。它不是模型/PD/单一候选 binary 的属性。
2. SHM NOT Healthy 仅出现在 22:01 boot 的 22:24 之后（即故障后）；学弟 12:17/12:48 两次 IMU 恶化均未伴随任何 SHM 警告 → SHM unhealthy 是**后果/伴生**，不是前置根因。
3. 两次持续恶化（12:48、22:23）都在"接管未按预期完成/恢复失败"之后；12:48 那次曾同 boot 内自行恢复（机制未知），22:23 那次直到重启才清除。
4. 尚未把触发链分解到"stop_app 官方切换本身"还是"自定义 writer 实体/命令与切换的交叠"——该组合从未被单独测试。

## 文件与哈希

| 文件 | SHA256 |
| --- | --- |
| boot_processes_soc1.txt | 84394264f5b9e80b3672aed25cb11f38d02e01ebca12ea61b2969ce64d3e180b |
| boot_processes_soc0.txt | 407942e0e33a61ea8c383cffe9ff196db8c0f3db599c79c49581b19464308abc |
| imu_shm_archaeology.txt | afaa22c24e8888353eac44f98b8b308c0e51f1ea88e1625a8478a695c84c4e13 |
| prev_boot_fastdds_identity.txt | 0b349a816a385f7b122e063db1407b37d0972eafd4e59b427101070ef12a17f6 |
| this_boot_vendor_health.txt | 11fdc08bc5ef39d59533a86a77a0b35b33fff9d994608ea853b09eb353a5c1a3 |
| soc1_observer15.jsonl | 34171e7da0338031afcb1f8efcd99d63f98af720300a0be78357dd5395e289e8 |
| soc0_observer15.jsonl | 920009277977e80ca7d711f7a41dfaf4f4204218bacc0f404e349fcc8a4d0ba4 |
| soc{0,1}_observer15.stderr | 空（e3b0c442…） |

补充记录：上一 boot 中 12426_fastdds.log 属于一个 22:13:30–22:23:09 期间在 SoC0 上运行、
加载了厂商 MC DDS profile、订阅全部 joint/IMU 话题的自定义诊断进程（对应当时的 GID
审计），其存在已并入时间线解读。本轮未运行任何同类进程（observer 为 Python、无 fastdds 独立日志）。

本轮结束状态：两端 master 会话保留；远端 /tmp 只读探针目录保留（重启即失）；机器人保持
完全吊起、官方 MC 单 writer、无任何自定义进程。唯一下一步建议见接管报告新检查点。
