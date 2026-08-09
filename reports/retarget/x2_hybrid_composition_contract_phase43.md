# X2 WBT Phase43：hybrid composition合同/preflight

## 目标合同

```text
Stage250 native lower12 + waist3 + root + intended gait schedule (frozen)
                         +
AMASS-UPPER-001 upper14 only (first 4.0s, real-time 30→50Hz)
                         ↓
X2 official31 / WBT29 / MotionLib-compatible dry composite
```

GMR lower、GMR root、GMR contact 永远不进入组合；头部保持Stage250；腰部本阶段保持Stage250，尚未开放有界意图。

## Phase27资格边界

- Phase27 available：`True`；native warm-start：`True`；contact-consistent seed：`False`。
- Stage250 trace含93D obs/15D action/50Hz intended gait phase；realized sole collision是否可用：`False`。
- 内部gait phase是**意图接触**，不是realized collision、更不是实机GRF/COP。

## Schema / mapping / round-trip

- Gold Phase11 ingestion：`True`；WBT29 Phase23：`True`。
- A zero-change official31→WBT29→official31 max error：0rad。
- B lower12/waist3/root/head/intended-contact unchanged：`True`。
- AMASS upper source only：`AMASS-UPPER-001` Bronze，first4.0s实时插值；upper qstep max=0.0287rad。

## 最小未来A/B门（本阶段未运行）

- A：Stage250 straight 原生全身。
- B：同一Stage250 lower/root/contact schedule + 冻结AMASS upper14；waist/head不变。
- prescribed 与 free 分栏；source trace稳定不得冒充replay稳定。
- upper error、survival、root/contact、slip与signed pitch门见JSON `future_gate_contract`。

## 裁决

**PHASE43_HYBRID_COMPOSITION_PREFLIGHT_BLOCKED**

Composition schema本身可构造，但Stage250 native teacher资格/realized-contact合同未完整通过；fail-closed，不把内部gait phase伪装成物理接触。Blockers: base_phase27_qualified; stage250_realized_contact_available; BASE Phase27 did not explicitly qualify Stage250 native dynamic seed; Stage250 realized sole collision/contact unavailable; gait phase is intent only

## 下一步

保留Stage250为native warm-start；若要解锁physical composition A/B，需按Phase27最小扩展补录physics-substep contact identity/position/impulse、完整root pose/velocity、applied torque和clip前后target。
