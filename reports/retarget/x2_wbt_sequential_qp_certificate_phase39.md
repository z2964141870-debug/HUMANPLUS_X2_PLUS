# X2 WBT Phase39：sequential-convex QP hard certificate

## 裁决

- **PHASE39_LOCAL_SEQUENTIAL_QP_CERTIFICATE_FAILED**
- 同Phase38 frames `[0,25)`、同变量/门；DAQP固定trust-region外层最多20轮，未扫参数。
- active最低sphere允许随迭代切换；全部12 sphere每帧都有nonpenetration线性约束。

## 结果

- QP solved outer iterations：20；failure：`None`。
- active sphere switches L/R累计：`[1, 0]`。
- stance distance p95/max：`[0.000515457841201924, 0.0010351528592163805]`m；minimum all-sphere distance：0.000000m。
- stance speed max：0.3090m/s；excursion：0.0207m；qstep：0.0572rad。
- max exact constraint violation：0.00696568；local feasible：`False`。

## 结论

固定sequential-QP合同仍未取得局部证书；这是solver/表示证书失败，不证明X2或该动作物理上不可能。

## 下一步

按门停止；不调trust-region、不换solver/窗口、不扩完整轨迹或physics。
