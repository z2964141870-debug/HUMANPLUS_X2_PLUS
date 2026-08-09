# WBT/BASE 交叉 Phase54：上肢扰动鲁棒下层后端短训准备审计

## 假设

仅让 Stage219 的 lower12+waist3 actor 在有界 upper14 **物理目标扰动**下适配，可以学习对上肢惯性扰动的因果反应，同时保持 Stage250 的 15D 部署接口；不加入 Future-intent action residual，不把 GMR 下肢叠回去。

## 干预

本阶段只读/静态审计，没有 Isaac physics、没有 optimizer。唯一拟议训练变量是：A 无上肢扰动，B 加入 Phase50 范围（scale 0.25、excursion≤0.12rad、slew≤0.20rad/s）的 upper14 position target；不同时加入外力。

## 资产链

| 资产 | SHA-256 | pass |
|---|---|:---:|
| `model_2600.pt` | `abcd49a823385bcd02e00480c9476ad5bc381e68252ad6930c85d731178f49bb` | True |
| `env.yaml` | `f702a358bdbc1df94ac2a54b83aa4f6d7c98c76b091ac05a65fd074c44e6f9d7` | True |
| `agent.yaml` | `38d462ad726e0e74d797f8a0ce3799aaadc02737e6e14a5cdf3443f7da0a8368` | True |
| `stage219_s2600_actor.onnx` | `b95bad3680658c7c25be50f236f070c80b7ff7ba8992355cec2ddfb1ee53c0f9` | True |
| `x2_official_forward_gait_phase_template_15dof.npz` | `16d77b382a5c016c9ca0ec05d8811b333ee9d805d0436381d80ab9202444ff1d` | True |
| `sitecustomize.py` | `e4e320e52633f86e9edca1a8d0134742a750e66d951ea02604b2c18fb5bf5f4f` | True |
| `stage208_official_mujoco_adapter.py` | `91a67ab6c583d48bbf09fea1d02de33f0b77961a404d8e67b7fab7764662d733` | True |
| `x2_hybrid_phase44_upper_motion.npz` | `71db36d0206c44da05640f6e3616f918945524e891a5df8051533fcbeb2ab2ef` | True |

- Stage219 PT fresh re-export 与冻结 Stage250 ONNX **byte-identical**，64×93 随机 batch 输出 max-abs=`0.0`。
- checkpoint：标准 ActorCritic `93→256→128→128→15`；critic `93→...→1`。
- 记录配置：dt=`0.005`s、decimation=`4`、control dt=`0.02`s、sole12、self-collision off、template scale=`0.15`。

## 29/31 与动作边界

- BASE_LOCOMOTION **不是 WBT29 actor**：policy action 只有 lower12+waist3=`15D`。
- simulator/observation 是 official/Isaac `31D`；upper14 由独立物理 target path 控制，head2 nominal。
- upper target/future 不拼入 actor，也不形成 action residual；但实际 arm q/dq 仍在既有 93D proprioception 中，因此 lower actor 能看到扰动结果并闭环反应。
- partition、Isaac/action/official adapter joint order、upper-only hook AST 全部通过：`True`。

## 训练/冻结预注册

- A：同一 source/config/seed，upper 全固定。
- B：同一 source/config/seed，50% env fixed、50% env replay upper；其余完全相同。
- trainable：actor 与 critic 的 `0/2/4/6` weight+bias；actor 输出仍只有 lower12+waist3。
- frozen：std、upper generator/PD、head、template、obs/action order、asset/PD/DR/reward、closed adapter/scene/supervisor/stand backend。
- 两支都 weights-only resume，actor anchor coeff=`1.0`，不扫系数。
- 固定 64 env×24 step=`1536` transitions/update；5 epoch×4 minibatch=`20` optimizer steps/update。

顺序门：live zero-update → 唯一 1 update → 若全部趋势门通过才最多累计 5 update。closed 端先保 ordinary straight/right/left/stop，再跑唯一 Phase50-style B；不重试。

## 对照

现有 Stage265/280 不是本方案：它冻结 Stage219 actor、增加 123D Future-intent suffix，只训练 8-mode residual adapter。本方案保持 93D→15D 接口，直接训练 lower/waist actor，让它通过实际 upper q/dq 反馈学习抗扰。

## 结果

静态资产、PT→ONNX、15→31 映射和 upper-only target path均闭合；但尚有三个 fail-closed 缺口：

- `LIVE_ZERO_ENTRYPOINT_NOT_YET_FROZEN`：The current Stage265 launcher instantiates a frozen-base FutureIntentActorCritic with a 123D suffix.  Phase54 requires a dedicated standard 93D ActorCritic launcher that rehydrates and hash-checks the archived Stage219 env/agent contract.
- `PHASE50_TRAINING_SOURCE_ADAPTER_PENDING`：The Isaac hook consumes a one-motion MotionLib pkl, while the immutable Phase50 artifact is NPZ and its source entry lives in a 24-motion cache.  A minimal immutable one-entry adapter/extractor must be hash-guarded before the live zero gate.
- `STD_FREEZE_AND_PAIRED_A_B_GUARD_PENDING`：The legacy trainer optimizes std by default and has no paired A/B zero-equivalence guard.

## 结论

**READY_FOR_DEDICATED_LIVE_ZERO_UPDATE_ONLY**。source PT/config/部署导出链对应，方向可进入 dedicated live zero-update；但 optimizer 仍为 **禁止**，不能拿现有 Future-intent launcher 冒充本方案。

## 下一步

只实现一个最小 dedicated 93D launcher + immutable Phase50 source adapter + paired zero guard；live zero 全过后先回报，再决定是否授权 1 update。
