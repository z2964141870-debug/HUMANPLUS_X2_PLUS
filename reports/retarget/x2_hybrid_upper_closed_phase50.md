# WBT Phase50：closed AimDK 上肢合成单次 A/B

## 假设

在 Stage219/Stage250 的腿、腰、root、PD、命令和 stand→move→stop 合同完全冻结时，只在 move 窗口叠加有界 `AMASS-UPPER-001` upper14，可以改善对该上肢意图的跟踪，而不破坏原生速度后端。

## 干预与对照

- A：复用 BASE Phase34 的 immutable closed AimDK episode，没有重跑。
- B：唯一一条新 episode；只有 upper14 target path 开启。新 physics episode 数：`1`，重试：`0`。
- domain233 在 simulator 启动前被 CycloneDDS 拒绝，physics episode=`0`；按 amendment 只改为 domain229 后完成唯一实际 B。
- Phase32 `LD_PRELOAD` observer 保存了 B 的完整 1 kHz physics；接触是官方 MuJoCo 模型真值，不是实机 GRF/COP。

## 静态合同

- Phase34 adapter hash exact：`True`。
- model/PD/command/stand-move-stop 等冻结配置逐项 exact：`True`。
- `_apply_upper_motion` 不引用腿/腰/root/head group，直接写 mask 仅 ARM14：`True`。
- bounded target excursion/speed：0.099633 rad / 0.200000 rad/s。

## 结果

| 指标 | A | B | 门 | 通过 |
|---|---:|---:|---:|:---:|
| AMASS upper counterfactual RMSE | 0.04850 | 0.03925 | B < A | True |
| AMASS upper counterfactual p95 | 0.11190 | 0.09182 | B < A | True |
| move forward (m) | 1.1796 | 1.1810 | drop≤0.15 | True |
| abs lateral (m) | 0.0836 | 0.1506 | increase≤0.05 | False |
| heading max (rad) | 0.1285 | 0.2053 | increase≤0.035 | False |
| stop drift (m) | 0.0673 | 0.0907 | increase≤0.05 | True |
| stop settle (s) | 2.020 | 1.96 | increase≤0.50 | True |
| move signed pitch mean (rad) | -0.1981 | -0.1929 | abs increase≤0.035 | True |

- B full/start/move/stop gates：`True`。
- 左/右 realized-contact agreement gate：`False`；stance-slip p95 relative gate：`False`。
- upper fallback steps：`0`。

## 结论

**NOT_PROMOTABLE_UNDER_PREREGISTERED_GATE**

唯一B episode未同时满足上肢改善与冻结后端的全部预注册门；不重试、不改腿腰补偿，不把局部改善表述为WBT晋级。

## 下一步

停止该固定upper14 composition候选；如需继续，应提出新的单变量假设，而不是重跑或放宽门。
