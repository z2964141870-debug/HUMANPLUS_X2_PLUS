# DC-PEFT Phase 3-I5 裁决

日期：2026-07-28

## 训练内曲线

| 第 5 轮 | B1 | E1 | N1 |
|---|---:|---:|---:|
| episode reward | 1.955 | 2.054 | 1.932 |
| episode length | 28.07 | 30.06 | 28.93 |
| actor grad preclip | 8.218 | 8.608 | 8.591 |
| approximate KL | 0.001521 | 0.001742 | 0.001928 |
| root error | 0.277 | 0.159 | 0.221 |
| body error | 0.0738 | 0.0596 | 0.0721 |

E1 的训练日志同时改善，但这里只视为待验证信号。

## 四域物理门

| 方法 | ideal | filter | delay | noise | 合计 |
|---|---:|---:|---:|---:|---:|
| B0 | 0/4 | 1/4 | 0/4 | 0/4 | 1/16 |
| B1-I5 | 0/4 | 0/4 | 0/4 | 0/4 | 0/16 |
| E1-I5 | 0/4 | 1/4 | 0/4 | 0/4 | 1/16 |
| N1-I5 | 1/4 | 1/4 | 0/4 | 0/4 | 2/16 |

## 连续指标揭示的差异

- E1 delay：4/4 stable、4/4 contact，root XY RMSE `0.184 m`，progress-ratio error `1.827`，三支中最好。
- N1 ideal：root XY RMSE `0.126 m`、progress-ratio error `0.926`，并严格通过 1 条。
- N1 delay/noise：各跌倒 1 条，root XY RMSE `0.417/0.345 m`，鲁棒性明显差。
- B1：全域稳定但 0 条严格通过，说明普通 single-critic 的五轮更新已破坏原有 filter 通过项。

## 裁决

五轮不足以证明 semantic grouping；随机分组在 ideal 域更强，必须保留 N1。E1 的价值在当前阶段体现为 delay 域的连续鲁棒指标，而不是严格通过数。允许进入 25 轮，目的不是继续“刷分”，而是观察这种 trade-off 是否稳定分化。
