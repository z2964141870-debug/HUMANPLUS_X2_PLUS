# SONIC/G1 → AgiBot X2 历史尝试索引

> 归档日期：2026-08-07  
> 原始工程：`/home/humanplus/x2_teleop_final/x2_sonic`  
> 原始文件保持不变；本目录只保存关键报告副本、索引和校验清单。

## 1. 这份历史解决什么

新工程 `CWI_CrossEmbodiment_Sim` 主要保存后期 CWI、Action Ownership、
Future-Intent Adapter 等实验。更早的 Stage 0–163 包含关节映射、
Any2Any/LoRA、执行器响应、root/foot 审计、长训和 GMR 数据漏斗，
仍位于旧工程。本索引把两段历史连起来，避免后续重复已否定实验。

## 2. 五分钟阅读路径

| 顺序 | 报告 | 用途 |
| --- | --- | --- |
| 1 | `key_reports/x2_sonic_stage0_to15_summary.md` | 目标、映射、仿真、数据、执行器与 LoRA 总览 |
| 2 | `key_reports/x2_stage15_sim_foundation_and_upper_body_summary.md` | 确认物理地基与上肢测试 |
| 3 | `key_reports/x2_stage16_to18_longtrain_foundation_summary.md` | 长训前基础与早期长训 |
| 4 | `key_reports/x2_stage19_root_foot_foundation_conclusion.md` | root/foot 矛盾重审 |
| 5 | `key_reports/x2_stage75_to77_longtrain_conclusion.md` | 长训趋势与失败边界 |
| 6 | `key_reports/x2_stage84_timescale200_root_foot_audit.md` | 时间尺度与 root/foot 因果审计 |
| 7 | `key_reports/x2_stage102_kdmr_lite_final_verdict.md` | KDMR-lite 路线裁决 |
| 8 | `key_reports/x2_stage103b_final_verdict.md` | 后续结构化修正的最终结论 |
| 9 | `key_reports/x2_stage163_reference_data_funnel_decision.md` | AMASS/PHUMA/BONES 数据漏斗结论 |
| 10 | `key_reports/EXTERNAL_REVIEW_HANDOFF.md` | 旧工程的详细目录地图与旁观复盘入口 |

当前 CWI 及分层 Adapter 阶段继续阅读仓库根目录的
`STATUS.md`、`EXPERIMENTS.md`、`FAILURES.md` 和 `ROUND7_RESULT_CARD.md`。

## 3. 已完成的主要尝试

- G1 29DOF 到 X2 31DOF 的语义映射、头部锁定与仿真闭环。
- X2 执行器的 ideal、nominal、delay、noise 评估域。
- Any2Any 式 LoRA、checkpoint 合并/恢复、短训和长训。
- root、足端、腕部和上肢的分项门禁与因果审计。
- AMASS、PHUMA、BONES 的 X2 GMR 重定向与动态数据漏斗。
- CWI 式语义多 critic、随机分组对照与 actor 梯度冲突探针。
- 腰腿 Action Ownership、X2 下层独占腿腰、有界上肢意图接口。
- Future-Intent + gait phase 的小型腰髋 Adapter 与多种子扰动测试。

## 4. 已否定或不应盲目重复的方向

- 只扩大 LoRA 或只调单个 reward 不能统一 root/contact/upper 闭环。
- CWI 能改善 value estimation，但未形成 X2 物理性能 Pareto 改善。
- 持续 actor 梯度冲突不是 CWI 失败的主因，当前不应盲目上 PCGrad。
- 单纯增加 AMASS/PHUMA/BONES 数量，在旧重定向器下不会自动产生动态 Silver 集。
- 固定幅值、固定滤波或事后 supervisor 只有局部工作区，不是通用安全保证。
- 训练 reward/episode length 上升不能代替 held-out 物理门禁。

## 5. 仍未完成

- 跨动作、跨速度、跨执行器域稳定通过的 X2 迁移模型。
- 可信的动态 X2 reference/Silver 数据，或能绕开严格全身 reference 的稳定分层接口。
- root、接触时序、横向平衡和上肢扰动的统一解。
- 满足“长训持续改善且不破坏基座能力”的训练契约。
- 真机部署、自动安全切换和第三机器人的跨具身验证。

## 6. 完整归档与恢复

完整压缩包不进入 Git，应保存在：

```text
百度网盘：/HUMAN+/HUMANPLUS_X2_PLUS/2026-08-07/history/
文件：x2_legacy_reports_stage0_163_20260807.tar.gz
SHA-256：b856517c36ffa8353bbb80595a9cc87f1a45ae4cd46858bda7860726dbe822f0
原始大小：145275616 bytes
```

压缩包包含旧工程 `docs/reports` 全目录和 `EXTERNAL_REVIEW_HANDOFF.md`。
`LEGACY_REPORT_SHA256SUMS.txt` 含原目录 3065 个报告文件的逐文件校验值。

恢复后在 `x2_sonic` 根目录校验：

```bash
sha256sum -c LEGACY_REPORT_SHA256SUMS.txt
```

## 7. 保护状态语义

- `key_reports/` + 本索引 + 校验清单：由 Git/GitHub 保护。
- 完整压缩包：由百度网盘保护。
- 原始旧工程：仍在本机保持原位，不因归档而删除。

