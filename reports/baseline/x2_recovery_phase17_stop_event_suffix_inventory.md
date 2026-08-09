# BASE Phase17：Stage326 结果分层 Stop-event Stateful Suffix 库存

> 日期：2026-08-09
> 状态：冻结 official AimDK MuJoCo 采集完成；只做离线审计，未训练。
> 机器可读结果：[x2_recovery_phase17_stop_event_suffix_inventory.json](x2_recovery_phase17_stop_event_suffix_inventory.json)

## 假设

Phase16 只有一条 f005 失败 recovery suffix，不能支持 `fraction=0` 与固定小 fraction 的训练预注册。本阶段验证一个更窄的数据合同问题：

> 冻结 Stage326 mixed-outcome 合同后，最多五条 official episode 能否同时得到至少两条完整成功和一条 height-collapse fail，并为每条保留 stop-event 对齐的完整 physical/controller/action-history continuation。

它不假设失败 action 是 expert，不验证 suffix curriculum 有效，也不解锁训练。

## 干预

- moving：Stage306，SHA `da95011f…bc4c`；
- stationary：source stand backend，SHA `edb73c7c…7565`；
- `brake_blend_to_policy`、fixed upper、stiff1.2、step clock、50 Hz；
- 0.5 s future-stop preview、4 s move、8 s stop；
- lateral-recovery supervisor 0.6、ankle-roll common 0.20；
- 只启用默认关闭的 stop-event recorder，窗口为 stop command 后 `0.0–4.0 s`；
- 每条只运行一次；达到 `2 full-success + 1 height-collapse fail` 立即提前停止。

没有修改 policy、PD、blend、horizon 或 seed 选择逻辑，没有训练。

## 对照

四条 episode 共用逐项相同的模型、adapter SHA `61c73512…1077` 和冻结控制合同。r1–r4 都是同一合同的随机重复，不把某一条成功/失败解释为单变量因果实验。

Stage335 90-state 只作为描述性 93D 支持域参照；Phase17 的 stop-event 行实际由 main/stationary actor 获得 authority，而 Stage335 是 recovery-reset 集。因此 OOD 数字只回答“相对旧 recovery reset 支持域有多远”，不能直接判定当前 stop controller 正确或错误。

## 结果

### 1. 结果类别达到预注册目标

| Episode | 严格类别 | Full | Move | Stop | Height | Stop-window 物理摘要 |
|---|---|---:|---:|---:|---:|---|
| r1 | failure | ✗ | ✓ | ✗ | ✗ | z `0.669→0.145 m`；tilt 首次 >0.30 rad：1.56 s；z 首次 <0.45 m：2.30 s |
| r2 | success | ✓ | ✓ | ✓ | ✓ | z min `0.628 m`；tilt max `0.221 rad` |
| r3 | critical | ✗ | ✗ | ✓ | ✓ | 安全停住，但移动航向门失败；z min `0.625 m` |
| r4 | success | ✓ | ✓ | ✓ | ✓ | z min `0.623 m`；tilt max `0.222 rad` |

累计：`success=2`、`critical=1`、`failure=1`。目标在 r4 后满足，r5 未运行。

这里的 success 严格等于 `full_gate_pass=true`。r3 虽然 stop/height 通过，但不能算“完整成功”。

### 2. Schema / hash / action-history 合同完整

- sidecar schema：`aimdk_x2_stop_event_suffix_sidecar_v1`；
- row schema：`aimdk_x2_stop_event_suffix_snapshot_v1`；
- 四条各 `201` 行，共 `804` 行，窗口严格为 `0.00–4.00 s`；
- 每行均包含 31DOF q/dq、root pose/velocity、93D observation、actual previous action、actual issued action、physical target、完整 controller snapshot 与逐层 hash；
- physical target 可由 `default + actual_issued × scale` 逐项重建；
- authority 覆盖 `brake_main → brake_stationary_blend → stationary`，并保留对应 observation slot；
- 所有 trace/sidecar/content/controller/physical hash 验证通过。

固定物理 bin 去重后仍为 `804/804` representatives，没有跨 outcome bin collision。这个结果说明四条 continuation 并非重复文件；它不等于有 804 条独立动作语义。

### 3. 相对 Stage335 的描述性 OOD

| Outcome | previous-action OOD | gravity OOD | root-state OOD | equal-group composite OOD |
|---|---:|---:|---:|---:|
| success（r2/r4） | 6.72% | 1.99% | 0.25% | 1.74% |
| critical（r3） | 6.47% | 0% | 0% | 0% |
| failure（r1） | 13.43% | 55.22% | 52.74% | 47.26% |

r1 中 gravity 组在 `1.72 s` 首次越 Stage335 分组门，之后才在 `2.30 s` 出现 height collapse；其 gravity/root/composite 距离尾部远大于成功/临界样本。这说明失败 continuation 确实补上了 Phase16 库存没有的“完整坍塌后段”。

但 root/composite 在 r1 很早就越门（约 `0.04–0.06 s`），r2/r4 的 previous-action 也在 `0–0.02 s` 越门。这主要反映 Stage326 stop-event 起点与 Stage335 recovery-reset 支持集的角色/时序不匹配，不能写成“早期 OOD 导致倒地”。Phase17 的证据是覆盖差异与时间先后，不是因果分类器。

相对 Phase14 全部健康 handoff 汇总，Phase17 overall previous-action/gravity/root OOD 分别为 `8.33% / 14.80% / 13.31%`；由于窗口、actor slot 与 outcome 构成不同，不做优劣排名。

## Provenance：两次 runner 分类问题

1. r1 后 runner 使用 `conda run python -` 从 stdin 分类，stdin 未正确转发，导致分类为空并退出。r1 的 trace/sidecar 已完整落盘并通过独立 validator，因此没有重跑。
2. r3 后旧 validator 错把 `stop_gate_pass` 当“完整成功”，曾短暂误报 `2 success + 1 fail`。发现后没有补跑/覆盖 r1–r3，而是把成功定义 fail-closed 修为 `full_gate_pass`；r3 正确归入 critical，仅新增 r4。

这两个问题只影响 runner 编排/标签，不影响冻结控制输入或已经记录的物理 episode。

## 结论

### 已完成

- 成功、临界、height-collapse 三类 stop/recovery closed-loop suffix 已具备；
- 其中完整成功两条、失败一条，满足 Phase17 库存目标；
- controller snapshot、physical state、actual previous/issued action 和 93D 输入均 hash 绑定；
- 失败样本覆盖到真正 height collapse，而不是只截取倾倒前 1.5 s。

### 仍未完成

- 没有证明 suffix curriculum 能改善 recovery；
- 没有定义失败 continuation 应作为 reset 分布、负例还是代价信号；
- 不能把失败 action 直接当作 imitation target；
- official MuJoCo 结果不等于真机结果。

因此裁决是：**库存/数据合同门通过，训练门继续锁定。**

## 下一步

若要预注册唯一一次 5-update paired smoke，必须先固定：

1. success / critical / failure continuation 如何进入 reset curriculum；
2. 失败 action 不作为 expert label；
3. `fraction=0` 对照和一个固定小 fraction，source checkpoint、seed、5 updates、official full gate 全部一致；
4. candidate 不低于 source/control，且 stiff-fixed 朝 5/5 改善；否则立即停止，25-update 继续锁定。

## 文件与复现

- 审计工具：`tools/official_x2/audit_phase17_stop_event_suffix.py`
- 采集 runner：`tools/official_x2/run_phase17_stop_event_suffix_capture.sh`
- 单条 fail-closed validator：`tools/official_x2/validate_phase17_stop_event_episode.py`
- 纯测试：`tests/test_phase17_stop_event_suffix_audit.py`
- JSON：`reports/baseline/x2_recovery_phase17_stop_event_suffix_inventory.json`

```bash
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=tools \
conda run -n x2-sonic-isaaclab pytest -q -p no:cacheprovider \
  tests/test_phase17_stop_event_suffix_audit.py \
  tests/test_phase17_stop_event_suffix_contract.py \
  tests/test_phase14_recovery_ood_audit.py
```

结果：`10 passed`。

## 边界

- official 指 AimDK v1 MuJoCo，不是真机；
- 没有真实足底六维力、COP、GRF；
- sidecar 是 post-inference/pre-physics 序列，不是 closed ROS mid-event restore 证明；
- 未运行训练、WBT、真机、Git 或百度网盘任务。
