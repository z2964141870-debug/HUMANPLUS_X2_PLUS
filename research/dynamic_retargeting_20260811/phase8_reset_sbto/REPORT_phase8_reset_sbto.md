# X2 Phase8：Projected reset + corrected active12 SBTO

日期：2026-08-11
状态：**PREFLIGHT PASS / UNIQUE SEARCH PARTIAL IMPROVEMENT / CONTACT-SWITCH GATES REJECTED**
搜索重跑：0；调参：0；RL/PPO：0。

## 结论

这条独立路线给出了目前 sampling route 最可信的局部改善，但仍未生成合格动态参考：

- 固定 Phase6 projected reset 后，zero-correction 精确复现 `0.683 s` 跌倒。
- 唯一 reset-aware SBTO candidate 撑过完整 `0.70 s` 搜索窗；完整 continuation 在 `0.775 s` 首次跌倒，比 Phase6 延长 `92 ms`，比原始 `0.575 s` 延长 `200 ms`。
- 左脚不再从头到跌倒 100% stuck：出现 26 次 contact switch，非意图期接触降至 `79.19%`（0.70 s 内为 `74.93%`）。
- 但最长连续“左离地、右支撑”只有 `21 ms`，未达到预注册 `30 ms`；大量短切换更像 contact chatter，而非稳定 liftoff。
- 0.70 s 共同门还失败于左脚 stuck、足速、右脚位移和 root 加速度。因此 candidate 被拒绝，不进入 RL。

结论边界：正确 contact-aware SBTO 能推动卸载并延迟倾倒，说明路线有进一步研究价值；当前固定 480-rollout/低维 correction 配置不足以形成稳定支撑转换。

## 1. 为什么搜索窗是 0.70 s

左脚 source intent 在 `10/30 = 0.333 s` 结束。旧 `0.34 s` prefix 只有约 7 ms 非意图区间，不可能认证 30 ms liftoff。搜索前冻结：

- 36 帧 / 0.70 s；
- 渐进窗 14/25/36 帧，即 0.26/0.48/0.70 s；
- 5 个控制点 `[0,8,18,27,35]`；
- 40 candidates、elite 8、每窗 4 轮、seed `20260811`，与 Phase4 相同总候选预算。

## 2. 合同与 preflight

- Phase6 reset SHA256：`141c6d7d9c006634eb00ddf25d2f593080abd0d8cfee8376767c14674d8cc2ea`。
- MuJoCo 3.3.7、官方 Euler、1 kHz、raw torque motor、官方逐关节 PD。
- 每脚严格 12 个 `contype!=0` active sole spheres。
- 搜索 loss 显式包含 source-intent support、左脚非意图 stuck、左脚离地高度不足、右脚支撑、flight、fall、足速、root 加速度和 tracking。
- penetration **不进入 CEM 排序**，只按官方 native 分布报告。
- deterministic loss delta `0`；双脚 t0 真实接触；nominal fall 精确 `0.683 s`。

Preflight：**PASS**。

## 3. 唯一搜索

| 窗口 | 第一轮 best loss | 第四轮 best loss |
|---|---:|---:|
| 0.26 s | 45.017 | 45.024 |
| 0.48 s | 32.990 | 23.120 |
| 0.70 s | 32.418 | 27.131 |

Wall time `90.77 s`。输出按预注册选择最后一轮最低 corrected-loss candidate；没有因结果追加候选。

## 4. 0.70 s raw replay

| 指标 | Phase6 nominal | Phase8 candidate | 门 | 结果 |
|---|---:|---:|---:|---|
| 完整窗口存活 | 0.683 s 倒 | full 0.70 s | full | PASS |
| 左非意图接触 | 100% | **74.93%** | ≤10% | FAIL |
| 左离地+右支撑最长连续 | 0 ms | **21 ms** | ≥30 ms | FAIL |
| 左非意图期右支撑 | 100% | 100% | ≥95% | PASS |
| active penetration L/R | -1.063/-1.038 mm | -1.671/-0.976 mm | native min -4.487 | PASS |
| legacy absolute penetration | FAIL | FAIL | ≥-0.5 mm | 仅保留诊断 |
| foot speed p95 L/R | 0.0344/0.0331 | **0.1311**/0.0583 m/s | ≤0.10 | FAIL |
| stance excursion L/R | 0.0081/0.0201 | 0.0093/**0.0389 m** | ≤0.03 | FAIL |
| flight | 0 | 0 | ≤2% | PASS |
| root accel p95 | 4.408 | 4.474 m/s² | ≤4 | FAIL |
| head max | 0.00449 | 0.00446 rad | ≤0.02 | PASS |

Candidate active penetration 比 native success p01（L `-2.116`、R `-2.491 mm`）更浅，属于官方成功仿真的软接触分布；绝对 `-0.5 mm` 仍如实报告，但不参与排序或 calibrated verdict。

## 5. 首次跌倒前 continuation

为避免跌倒后的碰撞数据污染解释，使用同一 candidate 做一次确定性 stop-at-first-fall 审计；没有重跑搜索。

| 指标 | Phase6 | Phase8 |
|---|---:|---:|
| 首次跌倒 | 0.683 s | **0.775 s** |
| 相对 Phase6 | — | **+92 ms / +13.5%** |
| 相对原始 0.575 s | +108 ms | **+200 ms / +34.8%** |
| 跌倒触发 | tilt | tilt `0.90024 rad` |
| 跌倒时 root-z | 0.5467 m | 0.4820 m |
| 左非意图 contact | 100% | **79.19%** |
| 左 switches | 0 | **26** |
| 左离地+右支撑最长 | 0 ms | **21 ms** |
| 右支撑/左非意图 | 100% | 99.55% |
| flight | 0 | 0 |

它不是完整动作成功，也不是稳定单支撑。证据更符合：优化器开始尝试卸载左脚并延后倾倒，但只形成毫秒级抖动切换；质心/角动量仍未被充分控制，最终继续以 tilt 倒下。

## 6. 裁决

Primary contact-switch verdict：**FAIL**。

相比受 visual penetration 污染的 Phase4，这次结果具有干净解释力：projected reset 消除了 t0 混杂，active12 objective 没有 penetration 污染，并自然产生了生存延长和部分卸载。不过稳定 liftoff、足速、支撑足位移与 root 加速度仍没有同时达标。

按预注册停止：不增大 CEM 预算、不改 stuck/liftoff 权重、不扫控制点、不运行 RL。

## 7. 产物

- `prereg_phase8_reset_sbto.json`
- `phase8_preflight.json`
- `phase8_result.json`
- `phase8_reset_sbto_candidate.npz`
- `phase8_candidate_prefall_audit.json`
- `x2_lunge_phase8_reset_sbto.py`
- `REPORT_phase8_reset_sbto.md`
