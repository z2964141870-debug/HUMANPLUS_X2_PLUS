# DISK LEDGER

| 日期 | 事件 | / 可用 | 本目录大小 | 备注 |
|---|---|---|---|---|
| 2026-07-26 | 任务启动基线 | 404G | 40K | x2_sonic 旧工程 69G（只读，不计入本任务预算） |
| 2026-07-26 | P0-B0 八次评价完成 | 403G | ~1M | 旧工程 logs 下 dcpeft_p0 前缀 16 目录共 ~129M（trace+日志，增量写入，属本任务可清理范围；r1 为冻结基线永久保留，r2 可在 Phase 1 后清理） |
| 2026-07-28 | Phase 3-I25 完成 | 未用 NVML 统计 | 1.3G | logs 1.1G、dual-init 185M、reports 292K、results 1.7M；step-1/step-5 checkpoint 已清理 |
| 2026-07-28 | 最终 checkpoint 审计 | 未用 NVML 统计 | 1.3G | 仅保留 B1/E1/N1 step-25 三个 final evidence（各约 187M）与 dual-init；无冗余中间 checkpoint |
| 2026-07-28 | 第二轮 gradient probe 完成 | 未用 NVML 统计 | 1.3G | 12 个边界 probe + 1 条25轮轨迹；保存 JSON/日志但未生成任何新 checkpoint |
| 2026-07-28 | 第三轮 FBP/LBP 5/25 完成 | 未用 NVML 统计 | 2.6G | 新增共同物化基座 151,324,629 B 与 4 个训练 checkpoint；所有 SHA256 已写入 round3 manifest |
| 2026-07-28 | 第三轮清理候选登记 | 未用 NVML 统计 | 2.6G | 拟保留共同基座 + LBP-I5；FBP-I5、FBP-I25、LBP-I25 共 463,406,917 B 待在报告复核后删除 |
| 2026-07-28 | 第三轮 checkpoint 精简完成 | 未用 NVML 统计 | 2.1G | 已按 manifest SHA 删除 FBP-I5、FBP-I25、LBP-I25，共 463,406,917 B；共同基座与 LBP-I5 保留 |
| 2026-07-28 | 第四轮接口门禁完成 | NVML 失效 | 2.1G | 新增约 5.4M JSON/NPZ 报告与小型代码；零训练、零新 checkpoint，无需清理旧工程 |
| 2026-07-28 | 第五轮覆盖与回退面板完成 | NVML 失效 | 2.1G | Stage5 报告目录约 8.1M；零训练、零新 checkpoint；保留全部小型反例 trace，无过剩 checkpoint 可清理 |
| 2026-07-28 | 第六轮 Gate1/5/25 完成（清理前） | 381G | 2.2G | Stage6 共 93 个 checkpoint、113,354,817 B；正式指标、审计与 SHA manifest 已落盘 |
| 2026-07-28 | 第六轮清理计划 | 381G | 2.2G | 删除 87 个 Gate1/中间 checkpoint，共 106,041,603 B；保留三支 Gate5 final 与三支 Gate25 final |
| 2026-07-28 | 第六轮 checkpoint 精简完成 | 381G | 2.1G | 已删除登记的 87 个文件；复核仅余 6 个 final，共 7,313,214 B，SHA 与 manifest 一致 |
| 2026-07-28 | 第七轮评估完成（清理前） | 未用 NVML 统计 | ~2.1G | 新增 2 个 Gate5 checkpoint 各 1,218,869 B；Stage7 报告约 3.0M，含 6 份逐步轨迹 |
| 2026-07-28 | 第七轮清理计划 | 未用 NVML 统计 | ~2.1G | 保留 NEUTRAL05 研究候选；RISK05 已有 SHA、审计和物理报告，计划删除其 checkpoint，预计释放 1,218,869 B |
| 2026-07-28 | 第七轮 checkpoint 精简完成 | 未用 NVML 统计 | ~2.1G | 已删除 RISK05 权重 1,218,869 B；保留其小型 params/events 与全部裁决证据，NEUTRAL05 final 继续保留 |

规则：
- 每次生成/清理 checkpoint 后追加一行；
- 只允许清理本任务在本目录新生成且指标已落盘+SHA256 已记录的文件；
- 永久保留：baseline、best、final、首个明确失败、配置、指标、manifest、关键视频、最小反例。
- 当前约 0.5G 为四域 raw trace；属于未来可压缩/清理候选，但在完成独立复核前不做隐藏式删除。
