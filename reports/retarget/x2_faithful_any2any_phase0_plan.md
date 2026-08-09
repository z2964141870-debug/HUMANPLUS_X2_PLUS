# X2 忠实 Any2Any Phase0 预注册计划

状态：只锁定复现协议，尚未训练。
论文哈希：`912e7425a37e8e2436870dc6b1c7f3c515500597534f38356e8c7494e64a4ec8`。

## 先纠正一个容易混淆的点

Any2Any 对 SONIC transfer 的明确口径是：在 **actor dynamics decoder 与 critic** 中注入 LoRA，FSQ 和其他预训练部分冻结。Figure 7 的 S7 最佳范围是 actor backbone、proprioception input、action output 和 critic backbone，但该消融来自 `Oli-WBT→Luna`，不能静默替换 SONIC-specific 配置。

因此首个忠实基线使用 SONIC-specific 范围；S7-style 的独立 proprio/output 注入只能作为基线完成后的单变量消融。之前 X2 的完整 `g1_dyn + critic` LoRA 方向基本符合这一范围，真正需要重做的是官方 contract、数据分布和评估，而不是假装过去只训练了末层。

## 冻结源资产

- G1/SONIC checkpoint：`/home/humanplus/humanoid-GPT/A/sonic_release/last.pt`
  SHA-256：`e6bdab3f64a39336b3d41877d4f497d05f58af275f288ec0e6746c283ded8909`
- source config：`/home/humanplus/humanoid-GPT/A/sonic_release/config.yaml`
  SHA-256：`f08187795fa16a839a28bc1c18e0555d38d9420e03733744341cdcb56ab629c7`
- source body：G1 29DoF；目标 X2 同样暴露 29DoF body，两个 head DoF 锁定。

## Kinematic alignment

在任何参数更新前同时完成：observation semantic alignment、reference body alignment、action alignment、action clipping/previous-action 语义和 target PD 声明。唯一目标映射是 `x2_official_joint_body_map.json`，不能再从旧实验脚本隐式推断。

## 网络冻结与 LoRA

首个 faithful baseline：

- 训练：SONIC `g1 dynamics decoder` 的全部 Linear LoRA；critic 的全部 Linear LoRA；
- 冻结：robot/reference motion encoder、SMPL/reference encoder、FSQ、kinematic decoder、全部 source dense weight、action std；
- 禁止混入：contact-preview、response-aware adapter、自定义 root/foot reward、CEM teacher、CWI/multi-critic、recovery switching。

实例化后必须从 `named_modules()` 记录精确 tensor、rank、trainable/frozen 参数和比例。旧 Stage152-B 是约 `0.662%`，而论文通用网络报告约 `5.26%`；两者结构不同，不能拿百分比硬对齐。

## PPO/source contract

尽量原样保持 source：4096 env、50 Hz（`sim_dt=0.005 s`、decimation 4）、24 rollout steps、5 PPO epochs、4 minibatches、clip 0.2、gamma 0.99、lambda 0.95、actor LR `2e-5`、critic LR `1e-3`、adaptive KL target 0.01、max grad norm 0.1。reward、termination、sampling 和 domain randomization 只能做有记录的 morphology-equivalent 替换。

## 数据与 held-out

- train：诊断面板中 `train_candidate` 且通过 Silver/Gold 的动作，之后扩展为独立版本的 official-X2 AMASS/PHUMA/BONES corpus；
- held-out：固定 10 条，绝不用于训练、阈值调整、early stopping 或 checkpoint 选择；
- 旧官方 B/D：保留作 regression pressure test，不再作为唯一训练集；
- 若后续拿到合规原始资产，再增加论文可比的 OMOMO/LAFAN official-X2 held-out。

## 算力阶梯与停止规则

| 阶段 | iterations | 允许进入下一阶段的证据 |
| --- | ---: | --- |
| zero-update | 0 | 初始化等价、冻结范围、tensor 映射全部通过 |
| smoke | 5 | rollout/loss/gradient 有限，所有目标 LoRA group 有梯度 |
| trend | 25 | 多动作一致改善，无动作语义错误或 held-out 灾难性回退 |
| pilot | 200 | 至少两类动作在官方物理门上改善，3 seeds 可复现 |
| medium | 1000 | 趋势继续上升或安全平台期，源能力未被破坏 |
| paper-scale | 8000 | 仅在高算力平台解锁；按 held-out 物理门选 checkpoint |

Any2Any 论文的完整对照采用 8000 iterations；SONIC→Oli 实机示例约 90 GPU-hours。我们不能把 25/200 update 当成论文级收敛，也不能在 Bronze/Silver 未通过时用长训掩盖 reference 问题。

## 必须保留的对照

1. Frozen source + kinematic alignment；
2. Aligned full fine-tuning；
3. Faithful Any2Any LoRA；
4. X2 from-scratch / 可比 target specialist；
5. `BASE_LOCOMOTION`，只在它支持的基础 locomotion 范围比较。

每次报告必须包含动作数/时长、GPU-hours、wall-clock、精确 trainable ratio、各动作类物理门、held-out 指标以及失败发生的事件和身体区域。
