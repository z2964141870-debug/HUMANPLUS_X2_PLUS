# BASE Phase9：stateful recovery 5-update paired smoke 预注册

状态：`READY_NOT_STARTED`
训练上限：每支 `5 updates`；`25-update/长训` 保持锁定。

## 假设

在 Phase6 projected physical snapshot 与 Phase8 deterministic suffix 基础上，低比例（5%）精确 stateful recovery RSI 能扩大独立 stand/recovery backend 的恢复域，同时不破坏冻结的 locomotion actor 与 full matched-event 闭环。

## 干预

同一 source checkpoint、seed、PPO 配置和 5 updates，只改变：

```text
control:   recovery_fraction = 0.00
candidate: recovery_fraction = 0.05
```

两支都显式启用 Phase4/6 stateful contract：physical `root/q/dq`、episode clock、command、previous raw/issued action、gait phase 与 low-command moving latch。旧 physical-only Stage337 入口不再允许复用。

## 对照

- source checkpoint：stand backend `model_150.pt`；
- moving actor：训练期间完全不加载，官方评估固定 Stage306；
- seed：47；
- env：64；每 update 24 steps/env；
- 训练域：ideal、sole12、self-collision off、ankle 40/20；
- optimizer：两支均 weights-only resume，避免继承未知旧 optimizer；
- official panel：source / f000 / f005 各 5 条完整 `prepare→stand→matched start/move→curriculum stop→stationary handoff`；
- stiff-fixed、upper fixed、无 supervisor；signed pitch 只报告、不进 gate。

## 冻结 SHA-256

| 资产 | SHA-256 |
|---|---|
| source PT | `4da931cf8ab1f094056a9d3bd024ef555b2c94944f09fe38924d7fde10a52251` |
| Stage335 NPZ | `4d8ce06b6e8dd9b43363dadedb781e9feb75f72ac092b475de2a1e2772545013` |
| Stage335 source report | `b4755acd98ecfa72717de7057d5ab9b23c54585dd2acd40e322dde55afc8f926` |
| gait template | `16d77b382a5c016c9ca0ec05d8811b333ee9d805d0436381d80ab9202444ff1d` |
| trainer | `8dbb7f5f67bdac9e47247c62db8700e71aaec3612f9521e1fa5dee085d4db6e8` |
| stateful glue | `36c1931bc193bd1d702ce06ef67d2dbf46b1bfe1daada93d4854ba744eb5b79f` |
| reset contract | `58fc61112d4e7dba62cbea8a29c575beb66c78cc49e29a5c9d7ae3ad7a61fbaf` |
| moving Stage306 ONNX | `da95011f7c7153bc538ab2d31948ddaee2e8916429d0ebca961ce0dc7532bc4c` |
| source stand ONNX | `edb73c7c3c5a64c3c3fcfe3020295b27058ecf0ae39b45d3fe0029cfc8367565` |
| official adapter | `d80900b2e7e1fac98aa586dc44c9dfb3d8561592e93167d931b8177afed7e8f9` |

## 确切命令和预算

```bash
NUM_ENVS=64 DEVICE=cuda:0 bash scripts/run_phase9_stateful_recovery_paired_u5.sh
bash tools/official_x2/run_phase9_stateful_recovery_paired_gate.sh
```

- 每支训练：`64 × 24 × 5 = 7,680` transitions；两支共 15,360；
- RTX 5060 8 GB；旧 64-env smoke 规模内，预计训练总 2–6 分钟；
- official panel 15 条 full episode，预计 5–10 分钟；
- 大文件只保留 source、每支 final `model_155.pt` 与 ONNX/export manifest；没有必要中间点。

## 晋级/停止门

必须同时满足：

1. 15/15 episode 均形成有效结果；
2. f005 full-gate 通过数不低于 source 与 f000；
3. f005 相对两者向 `5/5` 改善（或直接达到 `5/5`）；
4. 无 actor shape、93D接口、stateful finalizer 或 official runtime 错配；
5. signed pitch 仅相对报告，不允许事后新增姿态阈值。

任一 actor/接口错配、official 退化或门禁不满足即停止。即使本 smoke 通过，也只提交人工复核，脚本明确写死 `updates25_unlocked=false`。

静态/纯回归：`24 passed`。训练、真机、WBT、Git、百度网盘尚未触发。
