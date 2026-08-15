# X2-Sonic PHUMA 分层闭环与输入稳定域实验

日期：2026-08-15
项目：`/home/yu/projects/BFM-Zero`
分支：`work/codex-x2-extremity-contract`

## 目的与边界

本轮只在 3090 工位机的官方 MuJoCo 场景中验证公开的 `x2_sonic_policy.onnx`。没有连接机器人、Orin、BLE，也没有修改 policy 权重、机器人固件或控制器。实验目标不是宣称“人体动作可以原样安全复现”，而是测量公开 policy 对 PHUMA/X2 动作输入的动力学稳定域，并验证最小的、可回滚的输入适配。

固定运行契约：1670-D observation → 31-D action，50 Hz，官方 X2 XML，policy SHA256：

`e7ccd6522010ea660facfb7265fb129dac2b580dd1483cc1dbada73c205309d7`

## 数据与去重

- PHUMA-X2 源文件：776 个，来自四个已有 subset（strict89、medium169、broad227、hybrid1200-clean291）。
- 分层选择：每个 subset 24 个，共 96 条选择记录；按 root 平移速度、关节速度、姿态幅度和源类别分层。
- 四个 subset 之间存在同一内容的复制文件，因此 96 条记录只有 51 个唯一源 SHA256。所有科研结论同时报告选择层面和唯一内容层面，避免重复文件放大结果。
- 数据仅写入：`/media/yu/FAFF-E977/data/BFM-Zero/`；原始文件和失败结果未删除。

## 诊断结果

### 原始 canonical 输入

96 条记录中 50 条通过；去重后 27/51 个唯一动作通过。根部倾斜 p95 与生存时间的 Pearson 相关为 **−0.862**；低倾斜组 15/15 通过，中倾斜组 7/10，通过率在高倾斜组降到 5/26。相比之下，root XY 路径、root XY 速度和简单关节速度限幅的信号弱得多。

### 单变量/组合对照（5 秒闭环，唯一内容）

| 输入处理 | 通过 | 改善/回退 | 说明 |
|---|---:|---:|---|
| canonical 原始 | 27/51 | — | 基线 |
| 关节速度限幅 2 rad/s | 27/51 | 0/0 | 仅略微改变生存时间 |
| 关节速度限幅 3 rad/s | 27/51 | 0/0 | 与基线几乎完全相同 |
| 姿态幅度 ×0.75 | 29/51 | 2/0 | 有限改善 |
| 姿态幅度 ×0.50 | 29/51 | 2/0 | 单独使用仍有失败 |
| 根部 roll/pitch ×0.50，保留 yaw | 32/51 | 6/1 | 明显改善 |
| 根部 roll/pitch ×0，保留 yaw | 48/51 | 21/0 | 主要稳定性增益 |
| 根部 ×0 + 姿态 ×0.75 | 50/51 | 23/0 | 仅剩一个高幅度动作失败 |
| 根部 ×0 + 姿态 ×0.50 | 51/51 | 24/0 | 96/96 选择记录全部通过 |

这里的“根部 ×0”只去掉根部 roll/pitch，不改变 yaw；“姿态 ×0.50”是相对于官方站立默认关节角缩放，不是把关节角缩到零。

### 30 秒重复闭环

最终适配（根部 roll/pitch ×0 + 姿态 ×0.50）在 96 条选择记录、51 条唯一内容上全部运行满 30 秒：

- `96/96` 通过，`51/51` 唯一内容通过；
- 最小生存时间：`30.0 s`；
- 最大根部 XY 漂移：`1.5937 m`；
- 最大机体倾斜：`0.1038 rad`；
- policy 仍为官方原始 ONNX，没有重新训练。

## 解释与限制

1. 这证明公开 X2-Sonic policy 在官方仿真场景中是可运行的；之前失败的关键原因更接近输入根部姿态/动作幅度超出稳定域，而不是 ONNX 不能用或 1670-D 接口错位。
2. 这不是最终的衣服遥操策略。完全去除人体 roll/pitch 会牺牲后仰、侧倾等表现力；它应被视为安全基线和稳定域探针。
3. 当前官方 policy 的 tokenizer 使用 root orientation，不使用 root position；因此 root XY 缩放作为负对照几乎不改变结果。后续若要恢复全身位移，需要重新设计 locomotion/root-phase 接口，而不是继续调 root XY。
4. 所有结果来自单一公开权重、单一官方 MuJoCo 场景和现有控制增益；还不能外推到真实 X2。真实机测试必须另行经过安全审批、低幅度动作和急停验证。

## 可复现实验产物

代码（将纳入本分支）：

- `tools/official_x2/inventory_x2_motion_bank.py`：源动作统计和分层选择；
- `tools/official_x2/analyze_x2_stratified_results.py`：按类别/速度/姿态/根部倾斜归因，并按 SHA256 去重；
- `tools/official_x2/adapt_x2_motion_distribution.py`：关节速度、姿态幅度、根部倾斜单变量适配；
- `tools/official_x2/compare_x2_closed_loop_variants.py`：按唯一源内容比较通过/失败转移。

关键 manifest/report：

- 统计：`/media/yu/FAFF-E977/data/BFM-Zero/processed/2026-08-15/phuma_motion_inventory_v2.json`；
- canonical 选择：`/media/yu/FAFF-E977/data/BFM-Zero/manifests/2026-08-15/phuma_stratified96_canonical_manifest_v4.json`；
- 原始闭环：`/media/yu/FAFF-E977/data/BFM-Zero/processed/2026-08-15/official_sonic_x2_stratified96_5s.json`；
- 归因：`/media/yu/FAFF-E977/data/BFM-Zero/processed/2026-08-15/baseline_analysis_v2.md`；
- 最终 5 秒：`/media/yu/FAFF-E977/data/BFM-Zero/processed/2026-08-15/official_sonic_x2_stratified96_rottilt0_pose05_5s.json`；
- 最终 30 秒：`/media/yu/FAFF-E977/data/BFM-Zero/processed/2026-08-15/official_sonic_x2_stratified96_rottilt0_pose05_30s.json`；
- 变量比较：`/media/yu/FAFF-E977/data/BFM-Zero/processed/2026-08-15/x2_sonic_all_variant_comparison.md`。

## 下一步

1. 在不改变 policy 的前提下，把“根部倾斜限幅”和“姿态幅度限幅”改成连续、可调的实时 adapter，输出安全状态和被限幅量；
2. 用衣服离线日志做 observation parity，确认衣服的 root orientation 是否正是高倾斜来源；
3. 只在仿真中加入 root-phase/contact 约束，寻找比 `roll/pitch=0` 更能保留表现力的安全边界；
4. 形成 A/B/C 任务级指标（稳定性、姿态误差、表现力、延迟），再决定是否值得训练 X2 专用 policy；
5. 未完成上述离线证据前，不把任何适配结果部署到真实机器人。
