# 第二轮结果卡：Actor 梯度冲突审判

## 🎯 任务

确认 E1 的失败是否来自 locomotion 与 upper objective 在共享 actor 上互相打架，并决定是否解锁 PCGrad/GCR-PPO。

## ✅ 已完成

- [x] 逐 head clipped PPO loss 精确恢复；
- [x] 123,432 维 actor LoRA 梯度测量；
- [x] E1/N1 × INIT/I25 × 3 seeds；
- [x] 48 个零更新边界 microbatch；
- [x] E1 完整 25 轮、300 microbatch 梯度轨迹；
- [x] 12/12 actor 参数严格零漂移；
- [x] 插桩复跑与原训练核心轨迹完全一致；
- [x] 完整单测 9/9。

## 🧪 Boss 结果

```text
边界矩阵：48/48 cosine > 0
25轮轨迹：298/300 cosine > 0
每轮聚合：25/25 cosine > 0
```

当前两个目标不是持续冲突，而是高度同向。E1 上肢退化不能归因于 PCGrad 所处理的负梯度问题。

## 🔒 未解锁

- [ ] PCGrad；
- [ ] GCR-PPO；
- [ ] 200/1000 长训。

## 🗝️ 下一主线

优先检查 CWI 的另一半：**数据使用解耦与纯 objective contract**。若该方向也不能形成 upper preservation，再回到 reference feasibility / 动态可执行性，而不是继续换 PPO 优化器。

完整报告：`reports/round2_actor_gradient_conflict_report.md`
