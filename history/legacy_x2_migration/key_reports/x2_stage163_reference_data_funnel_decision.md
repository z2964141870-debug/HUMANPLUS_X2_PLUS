# Stage 163 — Reference-data funnel decision

## Current evidence

| source | processed scope | strict / Silver dynamic result | decision |
| --- | --- | --- | --- |
| AMASS | 11,796 registered; 58 unified 30 Hz X2 pilot clips | certified Gold 0, Silver 0 | retain as a source; do not full-retarget with the old generator |
| PHUMA | 5,201 dynamic source candidates registered; fresh balanced X2 panel 36 | strict dynamic contact 0; 10 near; 2 mechanically clean but contact-incomplete | current best repair testbed |
| BONES-Seed | 344 locomotion clips fully streamed, retargeted and audited | strict 0, near 0, Silver 0 | stop additional BONES ingestion under the current retargeter |

The result does **not** say that AMASS, PHUMA, or BONES lacks walking. The
same failure signature recurs after each source is mapped to X2:

- stance-foot drift/slip is too high;
- lower-body target steps are too abrupt;
- single-support clearance/contact timing is unreliable.

Therefore the bottleneck is now the X2 retarget/trajectory contract, not raw
motion count. Adding another large source through the unchanged generator is
unlikely to produce a dynamic Silver set.

## What is frozen

- Existing AMASS/PHUMA/BONES source files and caches are retained unchanged.
- BONES receives no further ingestion with the current method.
- No strict/near candidate is promoted to policy training.
- No four-domain rollout is run for a panel with zero strict dynamic-contact
motions.

## Next data intervention

Use the 10 PHUMA near candidates as a small, reproducible A/B panel for a
contact-aware **lower-body plus root** retargeter. Preserve the GMR upper body;
solve a short window over root XY/yaw and lower-body joint trajectories with:

1. inferred stance foot world-position lock and swing-foot clearance;
2. X2 joint bounds and a 30 Hz joint-step/rate constraint;
3. limited root/pelvis correction rather than arbitrary multi-metre root edits;
4. lower-body deviation penalty against the original GMR solution;
5. the same FK/contact, prescribed-root tracking, and semantic reports used by
   the current funnel.

This is a kinematic contact repair, not a claim of measured force/COP or a
substitute for the later free-root dynamics teacher. Its only promotion
criterion is a measurable improvement over the frozen PHUMA baseline without
breaking upper-body semantics. If it fails on this ten-clip panel, further raw
data expansion should stop until a stronger kinodynamic method is available.

## Relevant artifacts

- `x2_gmr_data_rebuild/outputs/amass_x2_unified_audit_20260717.md`
- `x2_gmr_data_rebuild/outputs/phuma_registered_locomotion_turn_step_x2_next36_tier_report_20260717.md`
- `x2_gmr_data_rebuild/outputs/bones_seed_uniform_full_funnel_20260717.md`
