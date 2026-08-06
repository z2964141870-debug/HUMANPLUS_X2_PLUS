# 第一轮结果卡：CWI / DC-PEFT 仿真因果实验

日期：2026-07-28  
地图：`ZHY/CWI_CrossEmbodiment_Sim`  
真机：未使用  
旧 X2/SONIC 工程：只读

## 🎯 主线任务

验证“把 locomotion 与 imitation/upper 交给不同 critic，能否缓解 X2 迁移中的多目标干扰”。

## ✅ 已完成

- [x] 冻结并两次复现 Stage152-B 四域基线；
- [x] 建立 scalar/vector 等价 Reward Contract；
- [x] 单 critic → 双 critic 公平等分初始化；
- [x] actor action 与 checkpoint 载入兼容验证；
- [x] B1/E1/N1 的 1、5、25 iterations matched 对照；
- [x] ideal / nominal(filter) / delay / noise 四域评价；
- [x] value、TD、GAE、KL、gradient、action drift 诊断；
- [x] 上肢能力保持与最终 Pareto 审计；
- [x] 完整测试 `6/6 passed`；
- [x] step-1/step-5 冗余 checkpoint 清理。

## 🧪 本轮 Boss 判定

**工程链路：通过。** 多 critic 实现可靠、等价、可保存恢复。  
**机制假设：部分通过。** 语义分组让 critic 的 value 估计明显优于随机分组。  
**迁移效果：未通过。** E1 没有形成物理 Pareto，且 wrist mean 相比 B0 恶化 26.38%。

因此：

- [ ] 不解锁 200×3 seeds；
- [ ] 不把训练 reward 更高误报成 X2 迁移改善；
- [x] 将 multi-critic 保留为可复用工程模块和有限机制证据。

## 📊 最关键战绩

| 模型 | strict / 16 | stable / 16 | wrist mean | 判定 |
|---|---:|---:|---:|---|
| B0 frozen | 1 | 16 | 56.23 mm | 基线 |
| B1-I25 | 0 | 15 | 55.53 mm | 单 critic 短训未改善 |
| E1-I25 | 1 | 12 | 71.07 mm | critic 更准，但能力交换 |
| N1-I25 | 0 | 16 | 54.38 mm | 负控制连续指标不输 E1 |

## 🔒 未解锁任务

- 200 / 1000 / 3000 iterations；
- triple critic；
- PCGrad / GCR-PPO；
- 真机与 SDK。

## 🗝️ 下一张任务卡的解锁条件

先测各 objective 对共享 actor/LoRA 参数的 gradient cosine：

```text
若持续负冲突成立
    → 解锁 PCGrad/GCR-PPO 的 5/25 最小对照
若不成立
    → 结束 CWI critic 主线
    → 回到 reference feasibility / data-use decoupling
```

## 📁 证据入口

- 最终裁决：`reports/phase3_i25_final_report.md`
- 当前状态：`STATUS.md`
- 实验账本：`EXPERIMENTS.md`
- 决策记录：`DECISIONS.md`
- 失败与反例：`FAILURES.md`
