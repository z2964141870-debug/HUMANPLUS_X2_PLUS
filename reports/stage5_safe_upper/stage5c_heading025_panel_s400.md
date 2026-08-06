# Stage5 有界上肢安全 Adapter 面板

- 完成：`8/8`
- 通过：`3/8`
- 生存失败：`4`
- 裁决：`DO_NOT_UNLOCK`
- 建议：不接 SONIC，转向下层 yaw/lateral 扰动鲁棒性。

| case | pass | survive | active | exc RMS/max rad | vel max rad/s | track p95 rad | Δheading rad | Δlateral m | tilt rad | feet lift mm L/R | hazard |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| box_lift_seed42_vx0p20_delay_s400 | ✗ | ✗ | ✗ | 0.009/0.060 | 0.200 | 0.068 | -0.169 | -0.454 | 0.772 | 75.6/191.9 | 0.053 |
| box_lift_seed42_vx0p30_delay_s400 | ✓ | ✓ | ✓ | 0.068/0.120 | 0.200 | 0.092 | -0.106 | -0.210 | 0.219 | 37.0/38.3 | 0.000 |
| knocking_seed42_vx0p20_delay_s400 | ✗ | ✗ | ✓ | 0.024/0.120 | 0.200 | 0.081 | -0.117 | -0.442 | 0.765 | 74.1/202.4 | 0.049 |
| knocking_seed42_vx0p30_delay_s400 | ✓ | ✓ | ✓ | 0.061/0.120 | 0.200 | 0.111 | 0.117 | -0.085 | 0.272 | 36.9/41.9 | 0.000 |
| swing_arms_seed42_vx0p20_delay_s400 | ✗ | ✗ | ✗ | 0.005/0.048 | 0.200 | 0.067 | -0.188 | -0.460 | 0.791 | 77.2/194.0 | 0.056 |
| swing_arms_seed42_vx0p30_delay_s400 | ✓ | ✓ | ✓ | 0.027/0.104 | 0.200 | 0.092 | -0.094 | -0.009 | 0.223 | 36.9/39.9 | 0.000 |
| wave_real_seed42_vx0p20_delay_s400 | ✗ | ✗ | ✓ | 0.013/0.104 | 0.200 | 0.071 | -0.201 | -0.482 | 0.773 | 86.4/88.9 | 0.043 |
| wave_real_seed42_vx0p30_delay_s400 | ✗ | ✓ | ✓ | 0.016/0.104 | 0.200 | 0.079 | 0.196 | 0.215 | 0.214 | 42.2/39.1 | 0.000 |

## 未通过项

- `box_lift_seed42_vx0p20_delay_s400`：active_upper_target, hazard_fraction_le_0p02, survived, tilt_le_0p45_rad
- `knocking_seed42_vx0p20_delay_s400`：hazard_fraction_le_0p02, survived, tilt_le_0p45_rad
- `swing_arms_seed42_vx0p20_delay_s400`：active_upper_target, hazard_fraction_le_0p02, survived, tilt_le_0p45_rad
- `wave_real_seed42_vx0p20_delay_s400`：hazard_fraction_le_0p02, survived, tilt_le_0p45_rad
- `wave_real_seed42_vx0p30_delay_s400`：heading_degradation_le_0p15_rad, lateral_degradation_le_0p15_m
