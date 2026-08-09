# X2 旧模型 vs AimDK v1.0 官方模型 Retarget Smoke（3 条）

## 裁决

- 数值最大差：`0.000e+00`。
- 两个 MJCF 的关节、body 拓扑、关节位置、限位、质量和惯量相同；旧文件只额外内置了一个不参与接触的 floor 及可视化资源。
- 因此，单纯把旧 GMR 的机器人 XML 换成 AimDK v1.0 官方 `x2.xml` **不会改善运动学重定向**。这是否定性但有效的 Phase0 结果，不能宣称官方模型修好了 reference。
- 后续差异应来自 canonical body/contact contract、IK objective、contact-aware repair 与官方动态回放，而不是来自这两个机器人 XML 的运动学本体差异。
- robot numeric contract exact equal: `True`；legacy-only geom: `['floor']`。

## 固定条件

- 同一源动作、同一 SMPL-X 模型、同一 v4 IK config。
- 30 Hz、full root、GMR root-z、smooth window 9、quadprog、damping 0.5。
- 两个分支均重新运行；不复用旧 cache，不训练，不做 postprocess。

| 动作 | 帧数 | dof max | root xyz max | root quat max | pose-aa max |
| --- | ---: | ---: | ---: | ---: | ---: |
| `AMASS-WALK-001` | 228 | 0.000e+00 | 0.000e+00 | 0.000e+00 | 0.000e+00 |
| `AMASS-UPPER-001` | 172 | 0.000e+00 | 0.000e+00 | 0.000e+00 | 0.000e+00 |
| `AMASS-SQUAT-001` | 165 | 0.000e+00 | 0.000e+00 | 0.000e+00 | 0.000e+00 |

## 版本

- legacy MJCF SHA-256: `8018c9a8d5729d1a6325ed55af9ecfcea2efe4e4cc6e06b9ecd882b24db60755`
- official MJCF SHA-256: `3ff43f05beb57412a804ba9fe05cd9adcdfce78e9ce73a95a71ac58ad20d91a3`
- IK config SHA-256: `822cbcf4909e595d3e957454e6c6a2a2bf17bd77a98c4ddd69d05b3a68f3dd44`

大 cache 位于 Git 外的版本目录；Git 只保存本报告、脚本和哈希。
