# X2 Native Gold → SONIC MotionLib Phase11

- 裁决：**PHASE11_MOTIONLIB_INGESTION_PASSED**。
- 仅做 Phase10 held-out MotionLib ingestion/FK round-trip；没有 PPO、LoRA、checkpoint、BASE、官方物理重放或真机。
- source trace 自身60秒稳定不等于 reference replay 稳定；本阶段没有把二者混写。

## 假设

Phase10 Gold 的 actual pose/root 能被 SONIC 原生 MotionLib 无损读入；原生 loader 对速度/contact 的派生行为可通过独立最小 adapter 显式恢复，而不修改源 Gold。

## 干预与对照

- 对照：原生 `MotionLibRobot`，`no_fix`、50Hz、关闭随机 heading/wrist/freeze augmentation。
- 干预：只在内存中恢复 `dof_vel`、root lin/ang velocity 和 Phase10 model contact；pose/root/FK/fps/clip 不允许变化。
- zero-update gate：固定 held-out clip0，比较源 Gold、MotionLib FK 与官方 `x2.xml` FK；无动作修正。

## 结果

- clips/fps/starts：`[400, 400, 400]` / `[50.0, 50.0, 50.0]` / `[0, 400, 800]`。
- 原生 q p95/max：`5.96e-08/2.38e-07rad`；root pos max `0.00e+00m`；root quat max `4.30e-07rad`。
- 原生 loader 会显式重算 dq：与 actual dq p95/max `0.932/8.292rad/s`，因此启用独立 state adapter。
- adapter 后 dq/root-lin/root-ang max error：`0.00e+00/0.00e+00/0.00e+00`；contact exact `{'left': True, 'right': True}`。
- official FK position p95/max `7.64e-08/3.11e-07m`；orientation p95/max `2.19e-07/4.58e-07rad`。
- WBT29 q/dq max error `2.38e-07/0.00e+00`；head q/dq max `0.00e+00/0.00e+00`。

## 结论

- 结果：Phase10 held-out Gold已通过SONIC MotionLib原生pose/root/FK与最小state adapter round-trip；未发现clip、fps、WBT29或head-lock静默改写。
- 结论：Gold可作为SONIC/X2 pipeline ingestion sanity set；actual velocity/contact必须通过独立adapter显式恢复，不能假设原生loader会保留。
- 下一步：若继续，只允许在held-out clip0做一次冻结prescribed/free reference-trackability smoke；仍不得训练。
