# Reports

**Successor: read [HANDOFF_TASK_CARD_20261020.md](HANDOFF_TASK_CARD_20261020.md) first.**
It includes access through the Windows bridge, exact artifact hashes,
evidence, known traps, unresolved publication blocking and the first task.
The operator agreed to reboot; completion/current state is unknown here.

**Last observed checkpoint (2026-10-20 22:41 CST, before the approved onsite reboot):** see the [SSH takeover record](X2_SSH_TAKEOVER_20261020.md). Two suspended powered attempts hit native stale. Custom writer exited; MC process existed but actual official command/feedback recovery was not verified. Vendor IMU Publish blocked for about 2000 ms and SHM errors were recorded. Reboot completion and present ownership are unknown. The next task is read-only recovery verification, not a third powered test or ground trial. Earlier no-publish results remain valid for computation only.

For staged execution and reporting, use the [v0.2 walking progress card](SONIC_V02_WALK_PROGRESS.md). Only the explicitly assigned stage is active; completion requires an evidence report and user review before further work. The first drafted task is a read-only inventory, not robot control.

For the keyboard-walking task, start with the [2026-10-19 v0.2 deployment and operation-card review](SONIC_V02_KEYBOARD_WALK_REVIEW_20261019.md). It links the author's real-robot runbooks, identifies the old card's blocking issues, and specifies the model/plant/runtime and keyboard-intent migration.

For the new `s1ft16000` route, start with the [2026-10-08 Planner-first successor handoff](SONIC_V02_PLANNER_FIRST_HANDOFF_20261008.md). It inventories what is already on GitHub, pins the public v0.2 artifacts, and separates the planner and SMPL phases.

Start with the [2026-09-29 handoff](HANDOFF_20260929.md). It distinguishes
the 2026-08-31 verified snapshot from later public model information and
does not claim to describe the robot's live state.

The copied Sonic tree contains the earlier operational history and safety contract (not the current October robot state):

- [Current state](../script/sonic_x2_transfer_v2/GR00T_audit/docs/CURRENT_STATE.md)
- [Decisions](../script/sonic_x2_transfer_v2/GR00T_audit/docs/DECISIONS.md)
- [Operations](../script/sonic_x2_transfer_v2/GR00T_audit/docs/OPERATIONS.md)
- [Experiments](../script/sonic_x2_transfer_v2/GR00T_audit/docs/EXPERIMENTS.md)
- [Garment live integration](../script/sonic_x2_transfer_v2/GR00T_audit/gear_sonic_deploy/README_X2_GARMENT_LIVE.md)
- [StandStill ledger](../script/sonic_x2_transfer_v2/GR00T_audit/gear_sonic_deploy/README_STAGE_SONIC_BALANCE.md)
- [Complete system description](../script/sonic_x2_transfer_v2/GR00T_audit/gear_sonic_deploy/README_X2_SONIC_GARMENT_SYSTEM.md)

Repository management:

- [Storage policy](STORAGE_POLICY.md)
- [2026-08-31 organization report](PROJECT_FILE_ORGANIZATION_20260831.md)
- [Cleanup ledger](CLEANUP_LEDGER_20260831.md)
- [Baidu upload status](BAIDU_UPLOAD_STATUS.md)

`analysis/` contains compact machine-generated summaries. `experiments/`
contains curated experiment evidence. `legacy_context/` is historical and
must not override the current-state documents above.
