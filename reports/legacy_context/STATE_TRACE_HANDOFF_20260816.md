# X2-Sonic state-trace continuation

## Current boundary

The last verified remote commit is:

~~~text
d7ad44af53e7d2cf37a5df9ad8c4ec5c1b531d93
~~~

The already-verified JSONL observation/action parity used a deterministic
synthetic qpos/qvel trace. The next experiment must replace that trace with
states captured from the official MuJoCo loop.

## Pending local files

~~~text
capture_x2_sonic_state_trace.py
sha256 773177372418513ac07a01fa33496dfda2c6392a77ef97a50961ff940345672d

parity_x2_sonic_jsonl.py
sha256 522420c8b7ea260119765732fea3c4c4d2f1d41479cc084f64443208d74918e1
~~~

The first file records the state before each policy tick:

~~~text
qpos[74], qvel[72], tick, motion_frame, time_s
~~~

The second file accepts that report through the optional --state-trace
argument and compares two JSONL streams under the same state sequence.

## Remote execution after SSH recovery

Project root:

~~~text
/home/yu/projects/BFM-Zero
~~~

Data root:

~~~text
/media/yu/FAFF-E977/data/BFM-Zero
~~~

Use the process-local CUDA library path; do not modify system libraries:

~~~bash
ENVROOT=/home/yu/miniconda3/envs/x2-sonic-isaaclab
CUDA_LIBS=$(find "$ENVROOT/lib/python3.11/site-packages/nvidia" -type d -name lib -print | paste -sd: -)
export LD_LIBRARY_PATH="$CUDA_LIBS${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
~~~

Capture a 100-tick trace from the stable root-tilt/pose variant:

~~~bash
PYTHONPATH=/home/yu/projects/BFM-Zero/tools/official_x2 +  /home/yu/miniconda3/envs/x2-sonic-isaaclab/bin/python +  tools/official_x2/capture_x2_sonic_state_trace.py +  --motion /media/yu/FAFF-E977/data/BFM-Zero/processed/2026-08-15/canonical_motion_bank/phuma_stratified96_rottilt0_pose05/phuma_x2_broad227_50fps__EgoBody__recording_20210911_S06_S07_01__body_idx_0__005_chunk_0001.json +  --model /media/yu/FAFF-E977/data/BFM-Zero/raw/sonic-x2/x2_sonic_policy.onnx +  --scene /home/yu/projects/sonic-web-demo-x2/assets/robot/scene.xml +  --ticks 100 --require-cuda +  --output /media/yu/FAFF-E977/data/BFM-Zero/manifests/2026-08-16/state_trace_rottilt0_pose05_100ticks.json
~~~

Then run parity with:

~~~bash
PYTHONPATH=/home/yu/projects/BFM-Zero/tools/official_x2 +  /home/yu/miniconda3/envs/x2-sonic-isaaclab/bin/python +  tools/official_x2/parity_x2_sonic_jsonl.py +  --reference /media/yu/FAFF-E977/data/BFM-Zero/processed/2026-08-16/canonical_replay_smoke/batch_reference_120.jsonl +  --candidate /media/yu/FAFF-E977/data/BFM-Zero/processed/2026-08-16/canonical_replay_smoke/adapted_root0_pose05.jsonl +  --model /media/yu/FAFF-E977/data/BFM-Zero/raw/sonic-x2/x2_sonic_policy.onnx +  --ticks 100 +  --state-trace /media/yu/FAFF-E977/data/BFM-Zero/manifests/2026-08-16/state_trace_rottilt0_pose05_100ticks.json +  --require-cuda +  --output /media/yu/FAFF-E977/data/BFM-Zero/manifests/2026-08-16/canonical_replay_smoke_real_state_parity.json
~~~

If the 100-tick trace falls early, preserve the partial report and classify it
as a dynamics failure; do not silently substitute synthetic states.

## Safety and versioning

- Offline MuJoCo only; no Orin, BLE, or physical X2.
- Commit both pending scripts before interpreting the new report.
- Generate a new manifest containing commit, file hashes, state-trace hash,
  parity report hash, and fall/capture summary.
- Preserve all previous manifests and reports.
