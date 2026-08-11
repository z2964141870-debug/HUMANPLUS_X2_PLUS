# X2 动态重定向赛马：官方软接触门校准

日期：2026-08-11
新仿真：0；优化：0；训练：0。

## 结论

原共同门“任意 active sole sphere clearance 必须 `>=-0.5 mm`”不符合官方 MuJoCo 软接触的成功行为，不能继续作为动态 teacher 的绝对淘汰线。

只读审计既有 Phase34 成功 closed AimDK 直行 trace（14,406 个 1 kHz physics steps）后，真实 active-sole contact distance 为：

| 成功 native trace | Left | Right |
|---|---:|---:|
| samples | 112,217 | 111,574 |
| minimum | -4.487 mm | -4.487 mm |
| p01 | -2.116 mm | -2.491 mm |
| p05 | -0.683 mm | -0.827 mm |
| median | -0.249 mm | -0.292 mm |

Phase6 reset-bridge 最坏值为左/右 `-1.063/-1.038 mm`：虽然没有通过事先注册的 `-0.5 mm`，但比官方成功 trace 的 p01 和 minimum 都更浅。

因此保留两层裁决：

1. Phase6 原预注册结果仍是 `7/8 FAIL`，不事后改写。
2. 物理解释更正为：Phase6 的动态软接触压入处于官方成功域内，不应再为追求 `-0.5 mm` 进行 clearance/控制调参。

## 方法学纠正

- 每脚只使用 12 个 `contype!=0` 的 5 mm sole spheres。
- 排除 visual mesh：left geom14、right geom37。
- 未来 penetration/contact-compression 门必须由同一 official scene 的成功 native trace 标定，并同时报告分位数与极值。
- 这只是仿真接触距离，不是真机形变、GRF、COP 或硬件安全阈值。

## 对赛马的影响

- Phase3b/3c、SBTO、DDR 的 visual-mesh penetration 数值均已撤回并按 active12 重审。
- 当前最有希望的是 Phase6：reset-compatible 投影是必要共同前置层；短前缀的其余动态门全部通过。
- 下一门不是继续压 penetration，而是将同一固定 bridge 延长到完整 5.8 s，检查 survival 与 contact phase 是否保持。
