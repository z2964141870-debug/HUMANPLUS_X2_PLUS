# BASE Phase16：唯一 f005 On-policy Recovery Suffix 采集

> 日期：2026-08-09
> 状态：唯一一次 official AimDK MuJoCo episode 已完成；只做离线审计，未训练。
> 机器可读结果：[x2_recovery_phase16_suffix_capture.json](x2_recovery_phase16_suffix_capture.json)

## 假设

Phase14 发现：健康 handoff episode 的 93D 输入并没有整体离开 Stage335 支持域，但 `previous_action` 组通常在 handoff 后约 `0.24–0.26s` 先越界，gravity/root 约一秒后才异常。

本阶段只验证一个更窄的数据假设：

> f005 真正取得 recovery authority 后的 closed-loop suffix，能否形成 schema/hash 完整、时序连续、包含 actual previous/issued action 的真实课程候选；它是否复现 Phase14 的异常先后顺序。

这不是“suffix 一定能治好 recovery”的假设，也不把失败 action 当作正确监督。

## 干预

只运行一次冻结的 official matched-event episode：

- moving：Stage306，SHA `da95011f…bc4c`；
- stationary：source stand backend，SHA `edb73c7c…7565`；
- recovery：Phase9 f005，SHA `9bc672fc…c0ca`；
- Phase13 action-history-synchronized `0.5s C2` handoff；
- fixed upper、stiff1.2、step clock、50Hz；
- 只新增默认关闭的 sidecar recorder，窗口为 recovery handoff 后 `0.0–1.5s`；
- `MAX_ATTEMPTS=1`，没有因物理失败补跑。

51822 在启动前无监听。主机上另有 A3 AimSim，但位于独立 ROS domain 232；本次使用 domain 176。

## 对照

- 控制与 episode 合同保持 Phase13/f005 冻结；唯一新增变量是旁路记录器，不参与 action。
- Stage335 90-state NPZ 保持只读，SHA `4d8ce06b…5013`。
- 没有采 source/f000 扩展对照，没有扫描 horizon、blend、PD 或 recovery fraction。
- OOD 相对量复用 Phase14 的 Stage335 reference、robust scaling 与 leave-one-out p95 threshold；三组阈值逐项相同。

## 结果

### 1. Schema / hash / authority

- sidecar schema：`aimdk_x2_recovery_suffix_sidecar_v1`；
- row schema：`aimdk_x2_recovery_suffix_snapshot_v1`；
- `76` 行，时间严格从 `0.00s` 到 `1.50s`，tick 连续，最大步长误差 `1.35e-15s`；
- 76/76 都是 recovery authority；
- policy-slot counts 精确为 `main=360, stationary=100, recovery=300`；
- 主 trace、adapter、f005、Stage335 哈希全部通过；
- raw rows `76`，固定预注册 bin 去重后 representatives `76`，没有删除原始行。

主资产：

- trace SHA：`6e32e7487e19b6e88695f7d26445a31eb412874f4cb83cd461cecb9ff3888da6`；
- sidecar 文件 SHA：`4e8a8751c7644b7fde0d78109f8d07215baf8874fdea62146b3e57676f866694`；
- sidecar content SHA：`643b998cd9e176f0bb369d46bb16bc44feee41cac562ac4db90546450d172dd0`。

sidecar 内的 `/results/...`、`/repo/...`、`/models/...` 是容器命名空间路径；审计使用对应主机文件重新计算 SHA，而不是假定路径字符串可在主机直接打开。

### 2. Episode 物理结果

- stand：通过；
- startup：通过；
- move：失败；
- stop：失败；
- full：失败。

采集窗口内 root z 为 `0.623→0.657m`，最低 `0.620m`；tilt 首次超过 `0.30rad` 在 `+1.40s`，窗口内尚未低于 `0.45m`。但完整 stop 后 root z 最低 `0.098m`、tilt 最大 `1.519rad`，机器人最终仍坍塌。

### 3. 相对 Phase14 的描述性覆盖

| Stage335 reference group | Phase16 OOD | Phase14 OOD | Phase16 first OOD | Phase14 first OOD median |
|---|---:|---:|---:|---:|
| previous action | 38.16% | 24.67% | 0.24s | 0.25s |
| projected gravity | 19.74% | 20.07% | 1.22s | 1.23s |
| root state | 7.89% | 11.18% | 1.40s | 1.36s |
| equal-group composite | 0% | 0% | — | — |

本次单 episode 精确复现了 Phase14 的时间顺序：`previous_action` 先于 gravity/root 约一秒离开分组支持域。previous-action OOD 比 Phase14 四条汇总更高，但这只是一次随机 episode，不能据此写成总体性能差异或因果效应。

## 结论

### 已通过的机制门

- Phase15 默认关闭记录入口可在真实 official matched-event episode 中工作；
- 每行 physical、93D actor input、actual previous/issued action 与完整 controller snapshot 已做 fail-closed hash 绑定；
- 采集严格限定在 recovery authority 的 `0–1.5s`；
- 真实 suffix inventory 已形成，且未污染 Stage335。

### 未通过的物理/学习门

- f005 仍没有通过 move/stop/full；
- sidecar 没有证明 suffix curriculum 有效；
- previous-action 先越界仍是相关性，不是倒地原因；
- 没有建立训练 fraction，也没有生成 candidate checkpoint。

因此当前裁决为：**采集合同通过，物理门失败，训练继续锁定。**

## 下一步

先停在离线 inventory：

1. 保留 immutable Stage335、trace、sidecar 与本报告的 SHA；
2. 不再采第二条 source/f000/f005 episode，不扫采集参数；
3. 由主线复核单条 suffix 是否足以预注册一次 `fraction=0 vs 固定小 fraction` 的 5-update paired smoke；
4. 即使获准，仍必须固定 source checkpoint/seed/updates/official gate，并在任何接口或 official 退化时停止；
5. 25-update 与长训继续锁定。

## 文件与复现

- 审计工具：`tools/official_x2/audit_phase16_recovery_suffix.py`
- 纯测试：`tests/test_phase16_recovery_suffix_audit.py`
- JSON：`reports/baseline/x2_recovery_phase16_suffix_capture.json`
- official trace：`/home/humanplus/projects/ZHY/x2_official_rl_deploy_v1/results/official_native_strict_20260807/phase16_f005_recovery_suffix_stiff1p2_fixed_r1.json`
- sidecar：`/home/humanplus/projects/ZHY/x2_official_rl_deploy_v1/results/official_native_strict_20260807/phase16_f005_recovery_suffix_sidecar.json`

```bash
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=tools \
conda run -n x2-sonic-isaaclab pytest -q -p no:cacheprovider \
  tests/test_phase16_recovery_suffix_audit.py \
  tests/test_phase15_recovery_suffix_aggregation.py
```

结果：`8 passed`。

## 边界

- 这里的 official 是 AimDK v1 MuJoCo，不是真机；
- 未使用真实足底六维力、COP、GRF 或 pressure truth；
- sidecar 是 post-inference/pre-physics sequence，不是 closed ROS mid-event restore 证明；
- 本阶段没有训练、没有真机、没有 WBT 修改。
