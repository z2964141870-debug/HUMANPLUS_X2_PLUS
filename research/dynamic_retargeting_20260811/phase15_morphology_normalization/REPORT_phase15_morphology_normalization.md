# X2 Phase15：确定性髋外展形态归一化

日期：2026-08-11  
状态：**FOOT WIDTH IMPROVED / SINGLE SUPPORT STILL 100% OUTSIDE / NO PHYSICS**

## 假设

Phase13/14 的主要几何异常是足距中位0.801m。若它主要来自 G1→X2 后 hip-roll 幅度不匹配，则按 X2 neutral 足距比例缩放左右 hip-roll，可能在不引入优化器的情况下恢复合理足位。

## 唯一公式

```text
scale = X2 official neutral foot separation / Phase30 median foot separation
      = 0.274300 / 0.801188
      = 0.342367
```

只缩放 `left/right_hip_roll_joint` 相对neutral的幅度；没有扫比例。root XY/orientation 和其余29个关节不变。随后仅用既有Phase7式MA9 root-ground把最小active12 clearance校到5.05mm。

## 结果

| 指标 | Phase30 | Phase15 |
|---|---:|---:|
| 足距 p50 | 0.801 m | **0.464 m** |
| 足距 p95 / max | 0.978 / 0.994 m | **0.536 / 0.546 m** |
| 单支撑 COM-outside | 137/137 | **137/137** |
| 单支撑 gap p50 / p95 | 0.214 / 0.392 m | **0.0767 / 0.1520 m** |
| 双支撑 outside | 30 | **0** |
| 双支撑最小margin | -24.8 mm | **+3.29 mm** |

形态归一化确实消除了双支撑几何矛盾并显著缩小单支撑gap，说明hip-roll/足宽是重要根因；但原stance schedule仍完全不成立。

代价也不可忽略：

- canonical re-ground 的root-z最大修正 `103.7 mm`；
- body position误差p95/max `176.5/246.7 mm`；
- upper四点最大位置变化 `103.7 mm`（来自整体root-z变化）；
- qstep p95 `0.0954 rad`、关节越限0。

## 裁决

`TWO-HIP MORPHOLOGY NORMALIZATION INSUFFICIENT`

这条公式提供了一个有价值的初始化方向：缩窄X2足位后，双支撑支撑面可覆盖COM。但把它直接当reference会：

- 把全部动态单支撑语义退化为双支撑；
- 引入约10cm整体高度变化；
- 仍保留7.7–15.2cm单支撑COM缺口。

因此不进入physics/RL，也不继续扫hip-roll scale。下一步应把该候选仅作为联合优化初值，同时允许root/foot placement/contact schedule与多关节下肢共同调整，并以人体关键点语义作软约束。

## 产物

- `audit_phase15_morphology_normalization.py`
- `prereg_phase15_morphology_normalization.json`
- `phase15_result.json`
- 本报告
