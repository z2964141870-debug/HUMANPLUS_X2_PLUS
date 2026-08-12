# X2 native posture Phase61

Decision: **FAIL_LOCAL_JOINT_TRANSITION_STOP**.

- Moving signed pitch mean delta: `+0.001208 rad`.
- Moving signed pitch p05 delta: `-0.001580 rad`.
- Actual support outside-mean delta: `-0.000690 m`.
- Terminal speed mean delta: `-0.004917 m/s`.
- Training peak GPU memory: `3335.0 MiB`.
- Failed local gates: `['source_survival', 'candidate_survival', 'source_termination', 'candidate_termination', 'moving_pitch_mean', 'moving_pitch_p05', 'event_terminal_coverage']`.

This single complete-event update is not long-training evidence. The default official stop phase remains owned by the frozen stationary actor; the candidate can only alter locomotion and the state delivered at handoff.
