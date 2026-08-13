# X2 state-feedback brake panel attempt 0

Decision: `FAIL_INVALID_PANEL_CONFIG_STOP`.

All 24 physics cases completed, but the six left-turn commands used
`MIRROR_POLICY=1`.  The frozen inner runner only treats the literal string
`true` as enabled, and all six raw summaries correctly reported
`mirror_policy=false`.  The frozen finalizer caught this before writing an
aggregate result.

The other 18 cases are contract-valid.  They passed 18/18 startup gates and
18/18 stop gates; 15/18 passed the movement and complete gates.  The three
failures were right-turn movement failures before the stop phase, so they do
not establish a stop-controller regression, but they do expose continuing
locomotion repeatability limitations.

The only allowed repair is a new preregistered run of the six invalid
left-turn cases with `MIRROR_POLICY=true`.  The 18 valid cases may not be
rerun, controller parameters and thresholds may not change, and repaired
evidence cannot promote this attempt into a confirmatory 24/24 result.

No optimizer, backward pass, checkpoint, Baidu Netdisk access, or GitHub
remote access was used.
