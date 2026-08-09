# BASE Phase14：Recovery 输入支持域离线审计

> 日期：2026-08-09
> 性质：纯离线诊断；未训练、未运行官方仿真、未修改 adapter、未改变门禁。
> 机器可复现结果：[x2_recovery_phase14_ood.json](x2_recovery_phase14_ood.json)

## 假设

Phase13 已消除了交权首 tick 的 physical target/action-history 不连续，但健康 4 条仍在约 1.2–1.4 s 后明显倾倒。待证伪假设是：

1. recovery policy 在交权后很快进入 Stage335 训练支持域之外；
2. 越界若存在，应先出现在某个明确特征组，并早于 root 倾倒；
3. 若 93D 输入整体仍在 Stage335 支持域，则“继续加入更多静态 reset state”没有直接证据，应转查闭环 rollout/action-history 课程。

## 干预

没有控制干预。只做三组不可变数据之间的离线比较：

- Query：Phase13 `blend=0.5` 中通过 stand/start 的 r1、r3、r4、r5，取 recovery handoff 后 `0.0–1.5 s`，共 `304` 行（每条 76 行）。r2 因交权前 stand/start 已失败而排除。
- Stage335：冻结的 90 个 recovery reset state，其中 source episode eventual-pass 63、eventual-fail 27。
- Source stand stable：Phase13 `blend=0` 五条 control 的 stand 末 1 s，共 250 行，只用来描述原 stand policy 的稳态支持域。

比较采用与 recovery actor 一致的 93D 输入分组：base linear/angular velocity、projected gravity、command、31D q、31D dq、15D previous action、4D gait phase；另外只作诊断加入 current action 与 root `[z, tilt, signed pitch, horizontal speed]`。

每个特征用 reference median/IQR 标准化（退化维用 std，再退化则 1）；每组距离取 RMS，因此 31 维 q/dq 不会仅因维数大而压过 3 维 base 特征。组间 composite 等权。OOD 门为 reference leave-one-out 最近邻距离的 p95，同时报告逐字段 reference p01–p99 覆盖和 leave-one-group-out ablation。

## 对照与数据契约

- Stage335 NPZ SHA256：`4d8ce06b6e8dd9b43363dadedb781e9feb75f72ac092b475de2a1e2772545013`。
- 90/90 行均按 `source_index + trace_index` 回接到 5 个冻结 source trace；obs 各切片与 NPZ 的最大容差为 `1e-6`。
- command、current action 与 root state 都来自同一批 hash 绑定 source trace，不是后补伪造值。
- 旧 Stage335 trace 没有显式 signed pitch，按其 projected gravity 恢复同一 ZYX pitch；负值表示后仰。没有构造 quaternion。
- Phase13 adapter SHA256 仍为 `a929ccec61f91af9d8492b11c4e50c7a61379d27a69b02327b984b2a9dd6759f`。

## 结果

### 1. 相对完整 Stage335 90-state，未发现全局 93D OOD

| 指标/分组 | Query OOD 比例 | 首次 OOD |
|---|---:|---:|
| 等权 composite | 0.0% | 无 |
| q（31D） | 0.0% | 无 |
| dq（31D） | 0.0% | 无 |
| current action（15D） | 0.0% | 无 |
| previous action（15D） | 24.67% | r3 0.26 s；r4/r5 0.24 s；r1 1.02 s |
| projected gravity | 20.07% | 四条 1.04–1.36 s，中位约 1.23 s |
| root state | 11.18% | r1 0.80 s；r4 1.36 s；r3 1.40 s |
| base linear velocity | 2.30% | 仅 r1 1.38 s |
| base angular velocity | 1.97% | 仅 r1 1.40 s |
| command / gait phase | 0.0% | 无 |

等权 composite 的 reference LOO-p95 为 `11.507`，query median/p95 仅 `1.088/1.933`。所有 leave-one-group-out composite 仍基本不越界（仅去掉 dq 后有 0.66% 的 r1 晚期越界）。因此结果不是 q/dq 高维尺度把其他组“淹没”造成的假阴性；但 Stage335 本身很异质，global composite 门较宽，所以结论必须以 per-group 为主。

### 2. 最早可重复的局部异常是 action-history 相关结构，不是 q/dq 爆出范围

previous-action 最近邻距离在 r3/r4/r5 的 `0.24–0.26 s` 已越 Stage335 组门，明显早于这三条的 tilt 门 `1.34–1.40 s`。但 previous-action 单字段 p01–p99 最早到 `1.12 s` 才退出（主要是 right hip roll）。这意味着早期信号更像 15D action-history 的**组合/相关结构**脱离训练样本，而非某一个关节 action 数值简单超界。

q、dq、current action 的组最近邻 OOD 都是 0%。逐字段分位数仍暴露出局部边界差异：

- left hip yaw q 在四条 handoff 时已落到 Stage335 p01–p99 外，但整个 31D q 仍在最近邻支持内；
- waist pitch q、少量 ankle/hip 特征也有局部退出；
- current action 的 left hip yaw 在 handoff 时有边界退出，但 15D current-action 组合仍未越组门；
- dq 的逐字段超界比例仅 0.18%，没有系统性 q/dq 支持域崩塌。

因此不能把结果简化成“所有输入都正常”，但也不支持“Stage335 没覆盖当前物理 q/dq，因此站立 recovery 完全无从下手”。

### 3. 相对 source stand 的窄稳态分布，handoff 从第一帧就是 OOD

相对 source stand stable，等权 composite 100% OOD，且除 command/gait phase 外，base velocity、gravity、q、dq、previous/current action、root state 都从 handoff 第一帧 100% OOD。

这只能证明 recovery slot 接手的是一个远离 stand 平衡点的移动状态，不能证明该状态物理上不可恢复；stand 对照本身非常窄。但它解释了为什么“直接复用 stand policy 做 recovery”没有可靠依据：这个 policy 的自然稳态访问分布和实际 handoff 分布完全不同。

### 4. eventual-pass / fail 拆分不能充当因果分类器

- 相对 eventual-pass 63 states：composite 仅 2.63% OOD（只 r1 晚期）；root 53.29%、gravity 48.36%，而 q/dq 仍 0%。
- 相对 eventual-fail 27 states：composite 0%，但 q 与 previous action 从 t=0 就 100% OOD，current action/dq 仍 0%。

这里的 pass/fail 是“该 reset 所在 source episode 后续是否通过”，不是单状态的可恢复性标签。两组小样本的覆盖形状差异很大，不能据此声称 query 更像 pass 或 fail，也不能训练一个伪 failure oracle。

### 5. 物理退化发生在局部支持异常之后

1.5 s 窗口内，r1/r3/r4 首次 `root_tilt > 0.30 rad` 分别为 `1.18/1.40/1.34 s`；r5 未越。四条都未出现 `root_z < 0.45 m`，所以这里观察到的是后仰/倾倒开始，不是完整落地坍塌。signed pitch 最小值为：

- r1 `-0.557 rad`
- r3 `-0.339 rad`
- r4 `-0.379 rad`
- r5 `-0.279 rad`

previous-action 组异常在三条中提前约 `1.1 s`，与后续失稳有时间先后关系；但这是相关性，不是因果证明。另一个限制是 1.5 s 窗口早于 Phase13 后续约 1.7–2.1 s 的 root-height collapse，因此本审计不能描述完整倒地阶段。

## 结论

假设只得到**局部支持**，没有得到“93D 整体 OOD”的证据：

1. Stage335 90 states 对当前 handoff 的 q、dq、current action 和全局等权输入覆盖总体足够；“再堆更多互不连续的静态 reset states”不是首选修复。
2. source stand policy 的稳态分布与 recovery handoff 从 t=0 完全不同，直接把 stand backend 当 recovery backend 缺少数据支持。
3. 最早且可重复的 Stage335 局部异常是 rollout 过程中 previous-action 的多维相关结构在约 0.25 s 后漂出，而重力/root 异常在约 1.2–1.4 s 后才出现。
4. 这更支持“静态状态覆盖尚可，但缺少与 policy action-history 同步的闭环 recovery continuation”这一窄化解释；仍不能据此宣称 action-history 是倒地根因。

Phase14 不解锁训练，也不改变 Phase13 的失败结论。

## 下一步

若后续获准做数据课程，优先采集/生成**同一 controller snapshot 起点下的连续 closed-loop recovery rollout**，让 93D state、previous action、actual issued action 和下一物理状态保持时序一致；按 eventual physical outcome 标注完整 continuation，而不是继续增加离散 reset state。

最小下一证伪应比较：现有静态 90-state curriculum vs 数量相近的 rollout-consistent short suffix curriculum，并保持 source checkpoint、模型、种子、更新数和 official gate 完全一致。当前阶段不应直接长训、扫 recovery fraction 或把 eventual-pass/fail 当单状态标签。

## 复现

```bash
PYTHONPATH=tools conda run -n x2-sonic-isaaclab \
  python tools/official_x2/audit_phase14_recovery_ood.py \
  --output reports/baseline/x2_recovery_phase14_ood.json

PYTHONPATH=tools conda run -n x2-sonic-isaaclab \
  pytest -q tests/test_phase14_recovery_ood_audit.py \
            tests/test_phase13_handoff_continuity_contract.py
```

测试结果：`9 passed`。pytest cache 因当前目录只读产生 warning，不影响测试结果。
