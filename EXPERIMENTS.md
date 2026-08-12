# 实验账本

| ID | 状态 | 初始化 | Actor | Critic | Reward | 预算 | 结果 |
|---|---|---|---|---|---|---:|---|
| NATIVE-S0-PREFLIGHT | complete/blocked-assets | Stage219/Stage250 freeze contract | frozen | n/a | reporting only | 0 physics / 0 optimizer | Stage219 source 已由 Phase42 归档恢复且 SHA exact；仅缺 exact gait template，deploy/training 尚未解锁；13/13 static tests passed |
| P0-AUDIT | complete | Stage152-B | frozen | original | scalar | 0 | 旧训练为 single critic；trainer 多头路径未被实际使用 |
| P1-STATIC | complete | n/a | n/a | n/a | scalar ↔ dual | unit | 3/3 tests passed |
| P1-RUNTIME | complete | Stage152-B | merged/frozen | 2-head plumbing | scalar ↔ dual | 1 env × 8 steps | max abs error 1.49e-08; shape/GAE/PPO path passed |
| B0-R1/R2 | complete | Stage152-B | frozen | original | scalar | 4 domains × 2 | exact repeat: max metric diff 0; ideal 0/4, filter 1/4, delay 0/4, noise 0/4 |
| P2-DUAL-INIT | complete | Stage152-B equal split | unchanged | dual | semantic | 4 env × 260 | 90/90 actor tensors exact; 402,480 rollout numbers exact; 45/45 value tensors loaded |
| B1-I1 | complete | Stage152-B | same PEFT | single | scalar | 1 | finite; actor grad 7.753; KL 0.001533 |
| E1-I1 | complete | Stage152-B equal split | same PEFT | dual | semantic | 1 | finite; actor grad 7.648; KL 0.001511; reward error 1.49e-08 |
| N1-I1 | complete | Stage152-B equal split | same PEFT | dual | shuffled | 1 | finite; actor grad 7.623; KL 0.001527; reward error 2.24e-08 |
| B1-I5 | complete | Stage152-B | same PEFT | single | scalar | 5 + four-domain | 0/16 strict; stable 16/16 |
| E1-I5 | complete | Stage152-B equal split | same PEFT | dual | semantic | 5 + four-domain | 1/16 strict; delay continuous metrics best, noise stable 3/4 |
| N1-I5 | complete | Stage152-B equal split | same PEFT | dual | shuffled | 5 + four-domain | 2/16 strict; ideal best, but delay/noise stable only 3/4 and large drift |
| B1-I25 | complete | Stage152-B | same PEFT | single | scalar | 25 + four-domain | 0/16 strict; 15/16 stable; wrist mean 55.53 mm |
| E1-I25 | complete | Stage152-B equal split | same PEFT | dual | semantic | 25 + four-domain | 1/16 strict; 12/16 stable; value EV 0.832/0.830; wrist mean 71.07 mm |
| N1-I25 | complete | Stage152-B equal split | same PEFT | dual | shuffled | 25 + four-domain | 0/16 strict; 16/16 stable; best root/progress continuous metrics; wrist mean 54.38 mm |
| PHASE4-200x3 | locked/stopped | n/a | n/a | n/a | n/a | 0 | E1 violates upper-preservation/Pareto gate; no evidence-based unlock |
| GRAD-BOUNDARY-E1 | complete | dual-init / E1-I25 | frozen | semantic dual | semantic | 2 checkpoints × 3 seeds × 4 microbatches | 24/24 positive cosine; boundary actor delta exactly 0 |
| GRAD-BOUNDARY-N1 | complete | dual-init / N1-I25 | frozen | shuffled dual | shuffled | 2 checkpoints × 3 seeds × 4 microbatches | 24/24 positive cosine; boundary actor delta exactly 0 |
| GRAD-TRAJECTORY-E1 | complete | dual-init | same PEFT | semantic dual | semantic | 25 × 12 microbatches | 2/300 negative; 0/25 negative aggregate; mean aggregate cosine 0.897 |
| PCGRAD/GCR-PPO | locked/rejected | n/a | n/a | n/a | n/a | 0 | persistent negative-gradient prerequisite absent |
| R3-UPPER-FIXED | complete | B0 / E1-I25 | frozen | n/a | n/a | 3 motions × ideal/filter | filter 3/3 stable；E1 wrist mean +1.78%，upper 独立能力保持 |
| R3-MATERIALIZE | complete | Stage152-B | folded 7-layer LoRA | folded 7-layer LoRA | n/a | unit + one-step | 11 tests passed；FBP/LBP initial 4×31 action max delta=0 |
| R3-FBP-I5 | complete | materialized Stage152-B | 29-DOF projected PEFT | scalar | original | 5 + four-domain | 2/16 strict；16/16 stable |
| R3-LBP-I5 | complete | materialized Stage152-B | 15-DOF waist/lower projected PEFT | scalar | original | 5 + four-domain | 1/16 strict；delay root 0.176 m、contact 4/4；upper preserve 4/4 domains |
| R3-FBP-I25 | complete/rejected | materialized Stage152-B | 29-DOF projected PEFT | scalar | original | 25 + four-domain | 1/16 strict；15/16 stable；不得续训 |
| R3-LBP-I25 | complete/rejected | materialized Stage152-B | 15-DOF waist/lower projected PEFT | scalar | original | 25 + four-domain | 0/16 strict；delay stable 2/4、root 0.402 m；5-step 收益反转 |
| R3-LONG | locked/stopped | n/a | n/a | n/a | n/a | 0 | train reward 上升主要来自 episode survival；四域趋势未同向 |
| R4-ZERO-EQUIV | complete | Stage208-s2550 | lower frozen | n/a | upper scale=0 | delay × 150 | 所有 trace 数组 max diff=0.0；注入器不改变旧闭环 |
| R4-UPPER-025-IDEAL | complete/pass | Stage208-s2550 | lower frozen + 14DOF oracle upper | n/a | fixed vx | 500 steps | 10 s stable；heading/lateral 相对 control 改善 |
| R4-UPPER-025-FILTER | complete/pass | Stage208-s2550 | lower frozen + 14DOF oracle upper | n/a | fixed vx | 500 steps | 10 s stable；无显式 delay 时相对门通过 |
| R4-UPPER-025-DELAY | complete/rejected | Stage208-s2550 | lower frozen + 14DOF oracle upper | n/a | fixed vx | 500 steps | stable，但 lateral 0.612→0.915 m；原速接口不合格 |
| R4-CF-ZD-ARMS | diagnostic/non-comparable | Stage208-s2550 | lower frozen | n/a | arm delay=0 | 500 steps | 静态 control 4.78 s 跌倒、动态候选 10 s；揭示窄吸引域，不作正式相对结论 |
| R4-CF-ZD-WAIST | complete/pass | Stage208-s2550 | lower frozen + oracle upper | n/a | waist delay=0 | 500 steps | 动态相对静态 heading/lateral 不退化；腰部 1 帧时序是强交互项 |
| R4B-UPPER-025-TS05 | complete/pass | Stage208-s2550 | lower frozen + bounded slow upper | n/a | fixed vx | 500 steps | 10 s；heading 0.510→0.491、lateral 0.612→0.505 m、双脚约 39 mm |
| R4B-UPPER-050-TS05 | complete/rejected | Stage208-s2550 | lower frozen + bounded slow upper | n/a | fixed vx | 500 steps | stable，但 heading +0.172 rad、lateral +0.467 m；安全幅值上限未到 0.50 |
| R4-TRAINING | locked/stopped | n/a | n/a | n/a | n/a | 0 | 本轮为接口因果验证；无证据授权长训 |
| R5-STATIC | complete | n/a | n/a | n/a | bounded upper contract | 9 tests | excursion/velocity/fallback/wrapped-yaw 单测 9/9 |
| R5-SMOKE | complete/pass | Stage208-s2550 | frozen lower + bounded oracle upper | n/a | fixed vx | 100 steps | 100/100；max target `0.0677 rad/0.159 rad/s`；横漂仅 +1.15 cm |
| R5-PANEL | complete/partial | Stage208-s2550 | frozen lower + bounded oracle upper | n/a | fixed vx | 4 motions × 2 unique speeds | vx0.30: 3/4 pass、4/4 survive；vx0.20: 0/4 pass、2/4 survive |
| R5-SEED-AUDIT | complete/rejected-axis | Stage208-s2550 | frozen | n/a | deterministic eval | seed42 vs 7 | 关闭 randomization 后结果逐项相同；不得当独立重复 |
| R5C-HEADING-GUARD | complete/rejected | Stage208-s2550 | frozen lower + latched IMU guard | n/a | fixed vx | 4 motions × 2 speeds | 3/8 pass；低速 survive 2/4→0/4；事后收臂是新扰动 |
| R5-SONIC-INJECT | locked/stopped | n/a | n/a | n/a | n/a | 0 | 多动作/低速门未过；只允许未来 shadow 或预期协调实验 |
| R6-G0 | complete/pass | Stage208-s2550 | frozen + zero-init Adapter | original | original | tensor + Isaac 100 | actor/critic load exact；39 个 trace 数组 max diff=0 |
| R6-G1 | complete/pass | Stage208-s2550 | frozen base + C/F/FNP Adapter | original trainable | original | 1 update × 3 | finite、reload、mask、actor frozen、residual≤0.10 全通过 |
| R6-G5 | complete/unlocked-25 | Stage208-s2550 independent | CURRENT/FUTURE/FNP | original trainable | original | 5 updates × 3 + 12-case | survive 5/5/4；FUTURE heading/lateral 0.614/0.473，方向信号优于对照 |
| R6-G25 | complete/partial-rejected | Stage208-s2550 independent | CURRENT/FUTURE/FNP | original trainable | original | 25 updates × 3 + 12-case | FUTURE 总步数3218、heading/lateral 0.560/0.282；但低速945<962，held-out wave heading 1.068、tilt 0.509 |
| R6-PROMOTION | locked/stopped | Stage208 retained | n/a | n/a | n/a | 0 | 一致性门失败；不续训、不接 SONIC、不部署 |
| R7A-BLEND | complete/rejected | FUTURE-s2575 frozen | residual blend 0.25/0.50/0.75 | original | n/a | 3×12-case | alpha0.75 selection 最优但 held-out wave-left heading/tilt 回归；不晋级 |
| R7B-G5 | complete/partial | Stage208 independent | NEUTRAL05 / RISK05 | original trainable | original / +heading risk | 5 updates×2 + 12-case | RISK05 失败；NEUTRAL05 固定面板形成 Pareto 候选 |
| R7C-ROBUST | complete/rejected | Stage208 vs NEUTRAL05 | frozen eval | n/a | n/a | 3 seeds×12 paired×400 | 初态 max diff0；生存15→17、heading改善，但 lateral 0.4657→0.4872、0/3 seed 三项不劣 |
| R7D-MATCHED | complete/diagnostic | Stage208 vs NEUTRAL05 | frozen eval | n/a | n/a | 36 paired traces | matched lateral max略好、时间均值略差；删失偏差仅部分成立，停止当前 objective |
| R7-LONG | locked/stopped | n/a | n/a | n/a | n/a | 0 | 不解锁同目标25/200/1000；Stage208 retained，NEUTRAL05仅研究候选 |
| STAGE8-CONTRACT | complete | Stage208-s2550 | frozen | BASE/FUTURE/FUTURE-PHASE | event-gate | 0 | 冻结最小原生后端门禁；既有证据覆盖0/6核心事件，待BASE事件矩阵 |
| PHASE40-NEW-MACHINE-LIVE-ZERO | complete/pass-zero-only | native generator seed + SONIC `last.pt` | frozen | frozen | observation/reset contract only | 1 env × 1 reset; 0 control/optimizer | `PASS_LIVE_ZERO_ONLY`；WBT29 与 10×58 future observation live 接线通过；未解锁训练 |
| NATIVE-P60-A | complete/baseline | Stage219-s2600 SHA `abcd49a8…` | frozen 93→15D | frozen | original | 64 env × 200 steps × 3 deterministic seeds | signed pitch `-0.193130 rad`；speed RMSE `0.105676 m/s`；4 s survival；seed 输出相同（只证明复现，不是独立样本） |
| NATIVE-P60-B-U1 | complete/rejected | fresh Stage219 + zero LoRA | rank4 LoRA only | rank4 LoRA only | + one-sided backward-pitch | 64 env × 24 steps; 1 update = 20 optimizer steps; 3 deterministic eval seeds | pitch `+0.001840 rad` 未达 `+0.002` 门，p05 `-0.000138 rad` 退化；KL mean `5.26e-5`；不续训 |
| NATIVE-P60-C-U1 | complete/rejected | fresh Stage219 + zero LoRA | rank4 LoRA only | rank4 LoRA only | + one-sided pitch + actual-contact COM/support | 64 env × 24 steps; 1 update = 20 optimizer steps; 3 deterministic eval seeds | pitch `+0.002053 rad`，但 p05 `-0.000495 rad`、support outside `+0.000201 m`，严格淘汰；KL mean `5.17e-5`；不续训 |
| NATIVE-P61-JOINT-EVENT-U1 | complete/rejected | fresh Stage219 + zero LoRA | rank4 LoRA only | rank4 LoRA only | Phase60 joint reward + aligned 10.24 s start/cruise/decelerate/hold event | 64 env × 512 steps; 32,768 transitions; 1 update = 20 optimizer steps | mean pitch `+0.001208 rad` 未达门，p05 `-0.001580 rad` 退化；support `-0.000690 m`、terminal speed `-0.004917 m/s` 改善；ideal 48/48 存活但 response 5/16 终止；严格淘汰，不导出、不续训；peak GPU `3335 MiB` |
| NATIVE-P62-ACTION-SENSITIVITY | complete/diagnostic | Stage219-s2600 SHA `abcd49a8…` | frozen + batched action probes | frozen | original | 64 env × 200 steps; base + 8 sagittal `±0.02` groups; 0 optimizer | 9/9 groups 4 s survival、0 termination；双髋 pitch 负方向最敏感（`0.2776 rad/action`），`-0.02` mean pitch `+0.005962 rad`、p05 `+0.002043 rad`，但 lateral/yaw 明显退化；仅解锁小剂量验证，不是 checkpoint/部署候选；peak GPU `3252 MiB` |
| NATIVE-P63-HIP-DOSE | complete/rejected | Stage219-s2600 SHA `abcd49a8…` | frozen + bilateral hip-pitch bias | frozen | original | 64 env × 200 steps; `0/-0.004/-0.006/-0.008/-0.010/-0.012/-0.016/-0.020`; 0 optimizer | 无恒定剂量通过：`-0.004` 姿态增益仅 `+0.000068 rad`；`-0.010` 虽过 mean/p05 但 lateral/yaw `+0.02442/+0.02195`；停止 constant-hip-dose，转协调/相位或膝小剂量诊断；25.68 s，peak GPU `3385 MiB`，无 checkpoint |

Phase60 的两次训练均为 64 env、1536 transitions、9,000 个可训练 LoRA 参数、fixed LR `5e-5`，单进程墙钟包含 Isaac 启动约几十秒，未生成中间 checkpoint。此次未对 peak VRAM 单独采样（64 env 无 OOM）；这是后续训练入口必须补齐的资源账本缺口，不能据此推算 1024/2048/4096 env 的容量。

Phase61 已补齐资源账本：source eval / train / candidate eval 墙钟分别约 `43.3/44.0/44.0 s`，峰值显存 `3204/3335/3207 MiB`。训练报告中的 `terminal_zero_steps=0` 是“全 64 env 命令均值必须精确为零”的保守计数；4 个 response env 中途 reset 后使批均值非零，不能解释为其余 60 个 env 没有到达 terminal hold。该诊断缺陷不改变 pitch mean/p05 和 response survival 已失败的淘汰结论。

Phase62 为单次 batched 闭环方向筛查，墙钟 `26.85 s`、峰值显存 `3252 MiB`、磁盘 used delta `135,966,720 bytes`（Isaac 缓存/日志；无 checkpoint）。首次报告因复用 evaluator 被标成 `mode=eval`，但 `phase=62`、`posture_variant=action_sensitivity`、9 组载荷和输入 SHA 均完整；归档结果显式记录 `legacy_eval_mode_tag_accepted=true`，后续运行已修为 `mode=screen`，未浪费一次物理重跑。

Phase63 按 Phase62 解锁边界只做了一次 ideal/fixed-upper 恒定双髋小剂量筛查。所有剂量均 4 s 存活、0 termination，速度/support/slip/root/action-rate 门也均满足；失败边界清楚分成两段：小于 `-0.010` 时 pitch mean 改善不足，大于等于 `-0.010` 时 lateral/yaw 越门。该负结果禁止继续扫 constant hip bias，但不否定 Phase62 中横向代价更小的膝 pitch 方向。运行墙钟 `25.68 s`、峰值显存 `3385 MiB`、磁盘 used delta `136,028,160 bytes`（Isaac 临时 USD；无 checkpoint/ONNX）。
