# Stage5 有界上肢安全 Adapter 面板

- 完成：`12/12`
- 通过：`6/12`
- 生存失败：`2`
- 裁决：`DO_NOT_UNLOCK`
- 建议：不接 SONIC，转向下层 yaw/lateral 扰动鲁棒性。

| case | pass | survive | active | exc RMS/max rad | vel max rad/s | track p95 rad | Δheading rad | Δlateral m | tilt rad | feet lift mm L/R | hazard |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| box_lift_seed42_vx0p20_delay_s400 | ✗ | ✓ | ✓ | 0.068/0.120 | 0.200 | 0.088 | 0.171 | 0.297 | 0.187 | 38.0/35.5 | 0.000 |
| box_lift_seed42_vx0p30_delay_s400 | ✓ | ✓ | ✓ | 0.068/0.120 | 0.200 | 0.092 | -0.106 | -0.210 | 0.219 | 37.0/38.3 | 0.000 |
| box_lift_seed7_vx0p30_delay_s400 | ✓ | ✓ | ✓ | 0.068/0.120 | 0.200 | 0.092 | -0.106 | -0.210 | 0.219 | 37.0/38.3 | 0.000 |
| knocking_seed42_vx0p20_delay_s400 | ✗ | ✓ | ✓ | 0.063/0.120 | 0.200 | 0.107 | 0.156 | 0.147 | 0.205 | 54.4/39.3 | 0.000 |
| knocking_seed42_vx0p30_delay_s400 | ✓ | ✓ | ✓ | 0.063/0.120 | 0.200 | 0.114 | 0.115 | -0.085 | 0.272 | 36.9/41.8 | 0.000 |
| knocking_seed7_vx0p30_delay_s400 | ✓ | ✓ | ✓ | 0.063/0.120 | 0.200 | 0.114 | 0.115 | -0.085 | 0.272 | 36.9/41.8 | 0.000 |
| swing_arms_seed42_vx0p20_delay_s400 | ✗ | ✗ | ✓ | 0.027/0.104 | 0.200 | 0.081 | -0.191 | -0.447 | 0.759 | 84.7/125.2 | 0.046 |
| swing_arms_seed42_vx0p30_delay_s400 | ✓ | ✓ | ✓ | 0.027/0.104 | 0.200 | 0.092 | -0.094 | -0.009 | 0.223 | 36.9/39.9 | 0.000 |
| swing_arms_seed7_vx0p30_delay_s400 | ✓ | ✓ | ✓ | 0.027/0.104 | 0.200 | 0.092 | -0.094 | -0.009 | 0.223 | 36.9/39.9 | 0.000 |
| wave_real_seed42_vx0p20_delay_s400 | ✗ | ✗ | ✓ | 0.019/0.104 | 0.200 | 0.074 | -0.163 | -0.475 | 0.798 | 86.3/88.9 | 0.046 |
| wave_real_seed42_vx0p30_delay_s400 | ✗ | ✓ | ✓ | 0.020/0.104 | 0.200 | 0.085 | 0.201 | 0.223 | 0.214 | 43.0/39.8 | 0.000 |
| wave_real_seed7_vx0p30_delay_s400 | ✗ | ✓ | ✓ | 0.020/0.104 | 0.200 | 0.085 | 0.201 | 0.223 | 0.214 | 43.0/39.8 | 0.000 |

## 未通过项

- `box_lift_seed42_vx0p20_delay_s400`：heading_degradation_le_0p15_rad, lateral_degradation_le_0p15_m
- `knocking_seed42_vx0p20_delay_s400`：heading_degradation_le_0p15_rad
- `swing_arms_seed42_vx0p20_delay_s400`：hazard_fraction_le_0p02, survived, tilt_le_0p45_rad
- `wave_real_seed42_vx0p20_delay_s400`：hazard_fraction_le_0p02, survived, tilt_le_0p45_rad
- `wave_real_seed42_vx0p30_delay_s400`：heading_degradation_le_0p15_rad, lateral_degradation_le_0p15_m
- `wave_real_seed7_vx0p30_delay_s400`：heading_degradation_le_0p15_rad, lateral_degradation_le_0p15_m
