# mc_transition_next

Offline-only design work for a future Sonic to `Develop_MC` handoff.  Nothing
in this directory imports ROS, AimDK, HAL, sockets, SSH, or process-control
APIs.  It cannot migrate robot state or publish robot commands.

Run the offline tests:

```bash
cd /Users/yu/Documents/ChatGPT/X2/mc_transition_next
python3 -m unittest -v
```

They can also be run from the X2 workspace root:

```bash
cd /Users/yu/Documents/ChatGPT/X2
python3 -m unittest -v mc_transition_next.test_state_machine
```

Files:

- `state_machine.py`: pure freshness tracker and authority state machine.
- `test_state_machine.py`: deterministic offline fault and recovery tests.
- `DESIGN.md`: state model, invariants, and adapter contract.
- `AUDIT_AND_RISKS.md`: audit findings, risk register, and hard hardware gates.
- `INTEGRATION.md`: future adapter boundaries; not a deployment procedure.

This directory is a design candidate, not hardware authorization.  The state
machine's `transition_request` field is only data.  No code here acts on it.
