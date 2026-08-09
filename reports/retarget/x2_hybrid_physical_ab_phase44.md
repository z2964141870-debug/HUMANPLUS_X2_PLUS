# X2 WBT Phase44：hybrid physical A/B runner合同

## 状态

**BLOCKED_BY_NATIVE_SEED**

- BASE Phase28 manifest exists：`True`；qualified+instrumented：`False`。
- 本次physics执行：`False`；PPO/optimizer：`False`。

## A/B

- A：Phase28/Stage250 native straight，lower/root/waist/head原生。
- B：完全相同native backend + `AMASS-UPPER-001` upper14；GMR lower/root/contact禁止进入。
- upper raw 30→50Hz由Phase43冻结；official adapter合同：`default + clip(0.25*(q-q0), ±0.12rad)`，再以0.20rad/s限速，输出绝对PD target。
- shoulder/elbow PD=40/5，wrist PD=30/3；official joint order固定。

## 静态结果

- raw/bounded upper shapes：[200, 14] / [200, 14]。
- bounded target qstep max：0.004000rad（门=0.004000）。
- lower/waist/root/head composition round-trip：`True`。

## 门禁

- A必须先完全复现qualified seed；A失败不运行B。
- prescribed/free严格分栏；prescribed不得表述为free balance。
- B报告upper RMSE/qstep、survival、realized contact、slip、signed pitch；composer不得直接改lower target。

## 结论

runner/schema/hash/upper-target合同已实现，但BASE Phase28尚未提供qualified且带substep realized-contact的native dynamic seed；默认fail-closed，未运行physics。

## 下一步

等待BASE Phase28 manifest；若qualified则另行显式--execute，先A复现再B。
