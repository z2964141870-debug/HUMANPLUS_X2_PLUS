# X2 Phase17：下一代动态重定向生成器冻结合同

日期：2026-08-11  
状态：**OFFLINE GEOMETRY IMPLEMENTATION READY / PHYSICS AND TRAINING LOCKED**

## 为什么必须换表示

当前证据已经排除四类小修：

1. Phase8/11：joint-mode sampling只能在chatter与stuck contact之间切换；
2. Phase12/16：固定现有支撑足，即使摆脚自由也没有局部可达单支撑帧；
3. Phase13/14：原contact标签137/137单支撑不成立，仅重标DS仍有30帧不覆盖COM；
4. Phase15：按X2比例缩髋能修复DS覆盖，却仍完全不能支持原SS，并需约10cm重落地。

因此 Bronze 的 lower q、root/COM和source-height stance标签不能再作为硬中心。下一生成器要从新的X2可行初态生成整段动作，而不是从坏初态做短bridge。

## 方法选择

采用一个本项目可立即实现的 X2-specific staged generator，方法论介于 contact-first、DDR 和 DynaRetarget 之间：

```text
人体/PHUMA上肢与躯干语义（软目标）
        + official X2 geometry/limits
        ↓
离散contact模板枚举
        + foot-placement/root/lower-q联合稀疏优化
        ↓
离线几何硬门
        ↓（仅通过后）
一次official raw-PD physics
        ↓（仅通过后）
DC-PEFT tracker
```

这不是忠实DDR/OmniTrack复现：两者当前上游代码/checkpoint不可用。它是针对现有证据的最小可执行替代。

## 本机能力

- Python 3.12.13、MuJoCo 3.3.7、SciPy 1.15.3。
- `mj_jacGeom`、`mj_jacBody`、`mj_jacSubtreeCom`、`mj_geomDistance` 全部可用。
- sparse LSQR、SLSQP、SciPy MILP可用；qpsolvers有DAQP/quadprog。
- `cyipopt`在h-gpt环境不可用，但Stage A不依赖它。
- official scene、Phase30、Phase15和shooting-for-contact源码树均存在。

因此 Stage A 没有依赖阻塞，不需要联网装包或占GPU。

## 冻结 Stage A

### 输入边界

- official X2模型/limits/active12为硬合同；
- 人体/PHUMA上肢关键点、torso方向、时长和forward/lunge语义为软目标；
- Phase15 lower pose仅作初始化；
- Phase30 lower q、root/COM、stance labels明确不是truth。

### 变量

- 30Hz逐帧root xyz/orientation、lower12+waist3 q；
- 每个stance段的support-foot XY anchor；
- 固定顺序枚举5个schedule：`DS`、`DS-L-DS`、`DS-R-DS`、两个双步模板；最小dwell 100ms。

### 离线硬门

- joint limit；qstep p95 `<=0.10rad/frame`；
- 单支撑COM margin `>=0`；
- stance speed `<=0.10m/s`、excursion `<=0.03m`；
- swing clearance `>=12mm`、flight `<=2%`；
- root horizontal acceleration p95 `<=4m/s²`；
- upper keypoint p95 `<=0.10m`、head `<=0.02rad`。

固定配置 sparse GN/LSQR；模板按顺序运行，第一个完整通过即冻结。没有模板通过则停止，不进入physics；不按结果扫权重、阈值或模板时序。

## 后续门禁

Stage B 当前未授权。只有 Stage A 出现完整pass并冻结hash后，才允许一次 official raw motor/per-joint PD replay；失败不重试。Stage B还必须全时长存活、contact agreement>=0.90并通过同一foot/root/upper门。

Stage C训练仍锁。只有Stage B通过，才允许沿用DC-PEFT frozen协议验证ideal `>=2/4`且过冲下降。

## 资源与项目管理

- 单进程、CPU线程<=4、0GPU、每模板wall-time<=30分钟；
- 不并行subagent，不影响其他工友；
- 每个模板结束先裁决；
- 本项是上次百度归档后的第10个实质任务，完成后触发Git和百度完整归档。

## 产物

- `audit_phase17_capabilities.py`
- `phase17_capability.json`
- `phase17_generator_contract.json`
- 本报告
