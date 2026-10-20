# SSH 接管原始记录：2026-10-20

交接保存范围：本目录 compact 文本、报告及诊断源码纳入本次 GitHub 交接提交；提交身份以 Git 历史为准。密码、权重、ELF 和原始 joint/action CSV 不入库。完整本地工件大小/哈希见 [manifest](../../data/manifests/x2_takeover_local_artifacts_20261020.tsv)，未上传百度网盘。用户已同意现场重启，但这里没有重启后的观察，以下状态均为历史。

新增 `powered3_runtime.log`、`resume_powered3_runtime.log`、`settle_powered3_runtime.log`、`writer250_observe10_runtime.log` 是从本地原件逐字节复制的完整 runtime 文本，不是重新运行结果；原始文件保留。

22:41 最新：两轮 powered 结果分别为 `resume_powered3_*` / `settle_powered3_*`，均原生 stale，不是接管通过。第一轮恢复确认过；第二轮官方恢复未通过。`settle_vendor_logs.txt` 保存 MC 缺消息和 FastDDS SHM unhealthy；`settle_hal_libraries_logs.txt` 保存 IMU Publish 约 2000 ms 阻塞。`udp_only_soc0/1.*` 为失败的仅 UDP 只读对照。`ending_soc0_vendor_state.txt` / `ending_soc1_state.txt` 记录最后进程和原件哈希。以接管报告最新节为准，不沿用下方历史“MC 正常”的状态。

22:06 恢复记录见接管报告最新节。`powered3_*` 保存 15:54 实际短接管及外加 100 ms 门中止；`writer250_*` 保存旧隔离构建、no-publish10、单行源码 diff。不得把 official-only GID 审计在 powered 窗口的 FAIL 当作控制故障：此时 custom 流量本来就是预期。`resume_writer250_build.log` / `resume_unit_tests.log` 为换电后的恢复构建与 19 项本地测试；实际接管另存新前缀，不覆盖旧失败原件。

结论与恢复入口：[接管记录](../../reports/X2_SSH_TAKEOVER_20261020.md)。15:34 最新观测：新 actor 已完成 20 秒无发送计算，MC 全程运行，最后每组一个官方 command publisher。后续吊起接管已获用户当次物理确认，但本检查点不提前声称执行。

新增证据：`nopublish20_runtime.log` 为完整命令/加载/CONTROL/正常退出；`nopublish20_analysis.json` 为 1000 帧时序统计；`nopublish_command_gid_cpp.csv` 为 55 秒按 GID 归属的实际收流；`nopublish_state_probe.jsonl` 为并行独立反馈；`nopublish_final_ownership.txt` / `nopublish_final_soc0.txt` 为最终 ownership / 进程。`nopublish_raw_thermal.json` 保留温度跳变；现场称非真实高温，未关闭任何保护。`soc0_raw_thermal*` 为未配置 Python 消息包的失败交叉读取，不能当作成功采样。

`nopublish_command_gid.jsonl` 和 stderr 是 Python MessageInfo 不兼容的失败原件；实际成功审计使用 `script/project_tools/x2_command_gid_audit.cpp`。`gid_audit_build*.log` 保留两次失败链接及最终成功。控制 runtime 未重编译。原始 CSV 本机放 `data/experiments/x2_nopublish20_20261020/csv/`，未提交/上传。

| 文件 | 作用 / 退出结果 |
| --- | --- |
| `soc0_baseline.jsonl` | 初次未加载厂商 DDS 配置，未接收反馈，退出 2 |
| `soc1_baseline.jsonl` | SoC1 初次 15 s 基线，通过，退出 0 |
| `soc0_vendor_dds_baseline.jsonl` | SoC0 14:37 厂商 DDS 配置，15 s，通过，退出 0 |
| `soc0_no_dds_control.jsonl` | SoC0 14:42 仅 unset DDS profile 的对照，未接收反馈，退出 2 |
| `soc1_final_baseline.jsonl` | SoC1 14:42 复核，15 s，通过，退出 0 |
| `*.stderr.log` | 对应采样 stderr；空文件表示没有 stderr 输出，不是漏存 |
| `soc0_message_hashes.txt` / `soc1_message_hashes.txt` | 六种实际消息定义 SHA256，逐一一致 |
| `soc0_processes.txt` | MC / HAL / DRP / IMU 的 PID、start time、comm、完整 argv |
| `soc0_custom_processes.txt` / `soc1_custom_processes.txt` | 当时定向进程检索无匹配；空文件不是机器人所有进程清单 |
| `soc1_control_listeners.txt` | 当时定向检查 5555/5556/5557/5558/5563 无监听 |
| `probe_unit_tests.log` | 采集器 3 项本地单元测试通过 |
| `pause_probe_unit_tests.log` | 采集器 + 有界暂停工具共 12 项测试通过 |
| `pause_probe_remote_hash.txt` | 上传到 SoC0 临时目录的暂停工具实际 SHA256 |
| `pause_probe_soc0_self_test.jsonl` | 只暂停临时 Python 子进程 0.1 s 的本机实测，退出 0；不是 MC 测试 |
| `soc0_pre_pause_identity.json` / `soc0_pause_preflight.jsonl` | 当前 boot/PID/start_ticks/uid/argv 的身份快照与不发信号校验 |
| `soc0_continuous_mc_pause.jsonl` / `soc1_continuous_mc_pause.jsonl` | 首轮 35 s 采样，时间窗已结束而拒绝暂停；这两轮没有暂停 MC |
| `soc0_active_mc_pause.jsonl` / `soc1_active_mc_pause.jsonl` | 实际暂停诊断的连续 45 s 采集，贯穿暂停和恢复；两轮正常结束 |
| `soc0_mc_pause_2s.jsonl` | 唯一一次 MC STOP/CONT，实际 STOP 至 CONT 约 2.0052 s，恢复为 S，退出 0 |
| `pause_orchestrator.log` | 双 host 接收匹配/新鲜度/ownership 前置门和暂停工具退出结果 |
| `mc_pause_analysis.json` | 暂停内部采样区间的 command/state/IMU 增量、时间戳与最终 publisher/GID |
| `soc0_post_pause_processes.txt` | MC、HAL、DRP、IMU 原进程仍运行 |
| `soc1_runtime_source_paths.txt` / `soc1_runtime_source_hashes.txt` | 当前实际安装树源文件路径及 SHA256，不是完整 binary 构建来源证明 |
| `soc1_runtime_aimdk_io.cpp.txt` / `soc1_runtime_scheduling_excerpt.txt` | 只读下载的 AimdkIo 原文与 control/writer/executor 源码摘录 |

JSONL 内 UTC 与 host monotonic 时间同时保留；CST = UTC+8。DDS 文本可能混入 stdout，解析时只取以 `{` 开头的行。`rate_hz` 是本采集器的接收频率，不是 HAL 生产频率；`max_rx_gap_s` 是整轮 callback 间隔最大值。未发现 endpoint / 未收到样本不是命令静默证明。

关键证据 SHA256：

```text
2eaaf7d5b02984386260159daee113e9238a67729095a9cfcac4f1f3b76be469  soc0_baseline.jsonl
212ee3919420057fa26cecc2c7b57059ee7a64ac5cba8d1aacbd05b1cf965497  soc1_baseline.jsonl
fadaa6b30bf511611cd123ccd5852cd8e477e1dd84d89d5cd9ecb707f354f500  soc0_vendor_dds_baseline.jsonl
622903e0654b4ef1aab38f5bc273b743658e80298f82c2c5576f7d239d46725c  soc0_no_dds_control.jsonl
6361379ce7666b6fb2fe26539e143e68c0b08b6ec99ff92eec821a35b565552b  soc1_final_baseline.jsonl
54a6ab5990d4e089cc3a27ab006300b2fe07805b675da66f7258c9744f9c86be  soc0_active_mc_pause.jsonl
876aa032c813d7b3f9c38c5fbabb7b21a36a25e7db5212b05c3729bb28a3bbe7  soc1_active_mc_pause.jsonl
dc11fb8b338b18df7296b10939b1e2ff5e750c528283ee6ed18d2a73e53a16ee  soc0_mc_pause_2s.jsonl
4281bb7eb2071d29eac46894435f1be77c56c02ef6af32d6ca41d6943801626b  mc_pause_analysis.json
```

本地原件未删除；采集当时尚未 commit / push，此后用户已授权本次交接上传，范围见本页开头及 Git 历史。没有上传百度网盘；若后续产生大体量 rosbag / telemetry，按仓库存储规则登记到 `data/` 并外存。
