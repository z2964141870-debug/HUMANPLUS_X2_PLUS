# BASE Phase18：Role-aware Reset Curriculum 与 Zero-update 注入门

> 日期：2026-08-09
> 裁决：`STOP_LOCKED`；资产门通过，三类精确注入门未全过，禁止训练。
> 机器结果：[x2_recovery_phase18_role_aware_zero_update.json](x2_recovery_phase18_role_aware_zero_update.json)

## 假设

Phase17 已有同一冻结合同下的 success / critical / height-collapse 三类 stateful suffix。本阶段验证：

> 能否在不复制或改写 Phase17 源的前提下，形成显式分层、安全筛选、hash-bound 的 reset curriculum，并把每类固定 16 个状态精确注入 `StatefulRecoveryRLEnv` 的 physical、93D、action history、clock、command 和 controller state。

所有 actual action 只允许作为 history/state；没有 imitation target、behavior cloning 或 expert label。

## 干预

### 1. Source-reference-only manifest

资产：[x2_phase17_role_aware_reset_curriculum.json](../../manifests/x2_phase17_role_aware_reset_curriculum.json)

- 文件 SHA：`8904ebd256ee49279ac8011db04dc0044b2ad75ea86d85f53d6dce174cc2c7a7`；
- 内容 SHA：`d17c1836e297bcd5c10ad85b577ed8affe8848651bbae37898cfff30d1bc575a`；
- 只保存 `source sidecar + row index + snapshot/controller hash`；
- 不复制 q/dq/root/93D/action，不改写 Phase17 sidecar；
- source 被读取时重新验证 file/content/row/controller hash。

eligible 门固定为：

- root z ≥ `0.55 m`；
- root tilt ≤ `0.30 rad`；
- q/dq 必须在 Phase6 同一 `sole12` URDF hard limits 内，禁止注入时 projection；
- 4 s capture 中至少保留 `0.50 s` continuation；
- post-collapse 不能作为正常 reset。

| Outcome role | Raw | Eligible | Ineligible |
|---|---:|---:|---:|
| success | 402 | 303 | 99 |
| critical | 201 | 151 | 50 |
| height_fail | 201 | 60 | 141 |

height-fail 的 141 条排除原因包括 tilt 123、root-z 90、continuation 25、q-limit 18（可重叠），因此真正倒地后的状态没有混进安全 reset。

### 2. 16-env fixed-batch zero-update probe

每个 outcome role 用 deterministic fixed batch 注入 16 个 eligible state；`PPO updates=0`、`optimizer steps=0`，没有调用训练 runner。

## 对照

- `reset_fraction=0` 在打开 manifest 前直接返回，不消费 suffix RNG；纯测试使用不存在的 manifest 路径仍严格 no-op。
- 非零 probe 只改变 reset source；Isaac X2 sole12、50 Hz、fixed action/observation contract 与 Phase6 一致。
- physical state 不允许 clamp 后冒充精确恢复。
- 93D 必须由恢复后的 simulator state/buffer 自然重算一致；禁止直接覆盖 observation 来“做出通过”。

## 结果

| Role | Physical/action/history max error | 93D max error | Controller hash | 结果 |
|---|---:|---:|---:|---|
| success | `5.96e-8` | `1.0` | exact | 失败 |
| critical | `5.96e-8` | `4.68e-7` | exact | 通过 |
| height_fail | `1.19e-7` | `0.04672` | exact | 失败 |

三类的 q/dq/root pose+velocity、previous/issued action、clock、command、controller registry 与 source reference 都精确。失败集中在“保存的 93D 是否能由同一 physical/controller snapshot 重新生成”。

### Success 的首个不等价字段：gait/contact

success fixed batch 中最大误差为 `1.0`。离线重建定位到 gait contact bit：某条 source row 的 clock 为约 `[0.454, 0.891]`，记录 contact 为 `[1,1]`，而当前 deterministic phase contract 重建为 `[1,0]`。

因此不能同时满足 source clock 和 source contact observation；直接把 93D contact 位写死会制造 reset 后下一 tick 立刻改变的 observation mismatch。

### Height-fail 的首个不等价字段：IMU/root 角速度语义

height-fail fixed batch 的 93D 最大误差 `0.04672`。离线从保存的 root quaternion 与 world angular velocity 重建 body angular velocity，source 自身最大差同样是 `0.04672`；projected gravity 还有 `0.00274` 差异。

这说明失败不是 Isaac 写入 q/dq/root 不精确，而是 sidecar 中 actor 93D 更接近某个 IMU/滤波 frame，不能由当前保存的 floating-root state 唯一重建。critical 恰好一致不代表该语义合同在三类上成立。

## Provenance 边界

首次 wrapper 依次启动三类。Isaac Kit 在 probe 已把 `passed=false` 写入 JSON 后仍以 clean process status 退出，所以 shell 没在 success 后立即停止，随后也运行了 critical 与 height-fail。发现后没有补跑或调参；wrapper 已改为以 JSON `passed` 字段 fail-closed。三次都为 zero-update，没有训练污染。

## 结论

### 已通过

- role-aware、source-reference-only、hash-bound 资产已建立；
- 三类都有安全 eligible 状态；
- post-collapse 与 q/dq 越限状态被排除；
- fraction=0 严格 no-op；
- physical、action history、command、clock、controller registry 注入基础设施成立；
- actual action 没有变成 expert label。

### 未通过

- success gait/contact observation 不能由当前 clock contract精确重建；
- height-fail IMU angular velocity/gravity 不能由保存的 root snapshot精确重建；
- 三类 exact 93D gate 只有 critical 通过。

因此：**资产门通过，state injection 门失败，训练保持锁定。**

不能用直接覆盖 93D 的办法绕过，因为那会重新引入 reset/rollout observation mismatch。

## 唯一 5-update 预注册草案（当前 BLOCKED）

只有三类 exact injection 全部通过后，才允许审核以下草案：

```text
source checkpoint：同一独立 stand/recovery backend
control：role_aware_reset_fraction = 0.00
candidate：role_aware_reset_fraction = 0.05
suffix 内部采样：success : critical : height_fail = 1 : 1 : 1
role 内采样：eligible source row 均匀采样
action 用途：history/state only，无 imitation loss
updates：5
seed：47
其他：checkpoint / optimizer / minibatch / RNG / official gate 全冻结
```

晋级要求：candidate 不低于 source/control，且 stiff-fixed full success 朝 `5/5` 改善；任何 actor/interface mismatch、official 退化或 reset 字段不精确立即停止。现在不授权运行。

## 下一步

唯一合理修复是回到 source contract：

1. 明确保存 actor 使用的 IMU frame、filter state 与 angular velocity，而不是只保存 floating-root world velocity；
2. 保存/恢复 contact/gait phase 的生成状态，使 clock 与 contact 标签来自同一状态机；
3. 重新运行 exact injection gate；
4. 三类均通过前，不运行 5-update。

## 文件与验证

- manifest 逻辑：`tools/official_x2/role_aware_recovery_curriculum.py`
- builder：`scripts/build_phase18_role_aware_curriculum.py`
- 0-update probe：`scripts/probe_x2_phase18_role_aware_reset.py`
- offline audit：`tools/official_x2/audit_phase18_role_aware_probe.py`
- tests：`tests/test_phase18_role_aware_curriculum.py`
- probe results：`results/phase18_role_aware_zero_update/{success,critical,height_fail}.json`

纯合同回归：`16 passed`。

## 边界

- official source 是 AimDK MuJoCo，不是真机；target injection 是 IsaacLab；
- controller snapshot 已 hash-exact 保存到 env registry，但尚未证明未来 Isaac manager 会消费全部 official brake/latch 字段；
- 没有 GRF/COP/foot-wrench truth；
- 未运行 PPO、optimizer、WBT、真机、Git 或百度网盘。
