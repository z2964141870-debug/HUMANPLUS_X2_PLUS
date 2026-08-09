# BASE Phase20：Outcome-aware State Role 与 Zero-update 门

> 日期：2026-08-09
> 裁决：**两类 state role 的 16-env zero-update 注入全部通过；只解锁一份 5-update 预注册供复核，尚未训练。**
> 机器可读结果：[x2_recovery_phase20_outcome_aware_zero_update.json](x2_recovery_phase20_outcome_aware_zero_update.json)

## 假设

“episode 最终失败”和“某一时刻是否是可用 critical reset”不是同一个标签。

Phase19 没采到独立 critical episode，但两条 height-failure 轨迹在真正倒下前有一段状态仍满足：

- root z `>=0.55 m`；
- root tilt `<=0.30 rad`；
- q/dq 位于 sole12 URDF 硬限位内；
- 未来 `0.5–1.5 s` 内首次发生 root z `<0.45 m` 的 height collapse。

本阶段验证：这些状态能否诚实标作 `critical_from_failure` state role，与 full-success 中的 `success_safe` 一起形成 reset curriculum；不能把它写成“采到了一条 critical episode”，也不能把失败 action 当 expert。

## 干预

从 Phase19 四条有效 v2 sidecar 构建 source-reference-only manifest：

```text
2 full-success episodes
    └── success_safe：366 个 eligible states

2 height-failure episodes
    └── critical_from_failure：66 个 eligible pre-collapse states
```

terminal、post-collapse 和不安全状态全部排除。manifest 只存 source row/hash，不复制 q/dq、93D 或 action payload；原始 Phase19 sidecar 不改写。

随后每类固定抽取 16 个 state，在 `StatefulRecoveryRLEnv` 做 0-update 注入：

- physical q/dq/root pose/root velocity；
- actor previous action、实际 issued action；
- command、clock、完整 controller snapshot；
- v2 torso IMU/frame 与精确 gait/contact source；
- actor 实际 93D。

torso IMU/base/gait override 只在 reset-return 首帧有效，episode step 增加后自动失效，避免把历史 observation 永久钉死。

## 对照

- `fraction=0` 在打开 manifest 和抽 RNG 之前直接返回；纯测试证明 RNG state 与 I/O 均严格 no-op；
- 两类 probe 均为 seed 47、16 env、50 Hz、sole12、无 observation corruption、无 push/external force；
- physical 容差 `1e-5`；93D、action/history、command、clock 容差 `1e-6`；hash 必须逐项 exact；
- 0 PPO update、0 optimizer step。

manifest SHA：`16b8ee05…85ea`，content SHA：`9cdaf372…724d`。

## 结果

### 两类角色均非空

| State role | Eligible | Source episodes | 语义 |
|---|---:|---:|---|
| success_safe | 366 | 2 | full-success episode 中的安全状态 |
| critical_from_failure | 66 | 2 | height-failure episode 中安全且 0.5–1.5 s 后坍塌的前兆状态 |

这里没有 fabricated critical episode。episode outcome 仍是 height failure，只是其中一部分前缀具有 critical state role。

### 16-env zero-update 全过

| Role | 93D max error | q/dq/root max error | action/history/command/clock | Controller hash | Actor-source hash |
|---|---:|---:|---:|---:|---:|
| success_safe | **0** | `5.96e-8` | **0** | exact | exact |
| critical_from_failure | **0** | `5.96e-8` | **0** | exact | exact |

两组 source ref 也逐项 exact，所有值 finite。结果文件 SHA：

- success_safe：`aed24b1e…a6b`；
- critical_from_failure：`55347050…15ea`。

纯测试 `15 passed`，相关完整回归 `101 passed`；覆盖 fraction0 no-op、role 窗口、terminal 排除、manifest/source hash fail-closed 和 action 非 expert 语义。

## 结论

Phase20 通过了两个此前没有同时通过的门：

1. **数据语义门**：从失败 episode 提取“当前安全、未来会倒”的 state role，而不篡改 episode outcome；
2. **注入等价门**：两类状态在 Isaac reset-return 中都能逐元素复现物理状态、93D、action history、command、clock 和 controller/actor source。

这只证明可以做一个干净的小实验，不证明训练一定有效。尤其，首帧 sensor/gait override 在下一 simulator step 失效，后续闭环是否平稳仍必须由 5-update paired smoke 和 official full episode 判定。

## 唯一 5-update 预注册（未启动）

```text
同一 stand/recovery source checkpoint、seed=47、64 env、5 updates

control:   outcome_aware_reset_fraction = 0.00
candidate: outcome_aware_reset_fraction = 0.05

candidate 内：
state role 1:1 采样
→ role 内 eligible row 均匀采样
```

固定：

- source checkpoint SHA `4da931cf…251`；
- ideal、sole12、self-collision off、official ankle PD、50 Hz；
- moving Stage306 训练中不加载、不更新；
- 两支使用相同 optimizer reset、minibatch、seed、reward、obs/action contract 和 manifest hash；
- failure action 只恢复 history/state，没有 BC/imitation loss；
- 每支 `64×24×5=7,680` transitions。

训练后 source / fraction0 / fraction0.05 各跑 5 条冻结 official matched-event full episode。candidate 必须不低于 source 与 fraction0，并让 stiff-fixed recovery 朝 `5/5` 改善；任何接口错配、无效 episode 或 startup/move/stop 回归立即停止，25-update 继续锁定。

当前状态是 `READY_NOT_STARTED`，本阶段没有启动该训练。

## 关键文件

- manifest：[x2_phase19_outcome_aware_state_role.json](../../manifests/x2_phase19_outcome_aware_state_role.json)
- contract：`tools/official_x2/outcome_aware_state_role_v2.py`
- Isaac glue：`tools/official_x2/stateful_recovery_isaac.py`
- probe：`scripts/probe_x2_phase20_outcome_aware_reset.py`
- runner：`scripts/run_phase20_outcome_aware_zero_update.sh`
- tests：`tests/test_phase20_outcome_aware_state_role.py`

## 边界

- `critical_from_failure` 是 state role，不是独立 critical episode；
- reset-return exact 不等于 vendor ROS mid-event restore 或未来 suffix 成功；
- 仿真证据不是真机、真实足底力、COP 或 GRF；
- 没有训练、WBT、真机、Git 或百度网盘操作。
