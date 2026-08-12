# X2 native posture Phase68 — one-step physical residual PPO

Decision: **FAIL_NO_LOCAL_SIGNAL_STOP**.

Technical checks: `{'prereg_sidecar': True, 'base_checkpoint': True, 'runner_hash': True, 'module_hash': True, 'interface_hash': True, 'train_decision': True, 'all_train_gates': True, 'optimizer_step_exact_one': True, 'checkpoint_count_exact_two': True, 'resource_exit_codes': True, 'resource_launch_count': True, 'resource_disk_delta': True, 'resource_gpu_peak': True, 'all_finite': True}`.

Source checks: `{'finite': True, 'role': True, 'survival': True, 'termination': True, 'zero_residual': True, 'zero_processed_delta': True}`.

Candidate safety checks: `{'finite': True, 'role': True, 'survival': True, 'termination': True, 'terminal_speed_mean': True, 'terminal_speed_p95': True, 'terminal_double_support': True, 'standing_zero': True, 'invalid_zero': True, 'non_knee_zero': True, 'residual_bound': True, 'residual_nonempty': True, 'effective_ratio': True, 'velocity': True, 'lateral': True, 'yaw': True, 'support': True, 'slip': True, 'flight': True, 'root_height': True, 'root_tilt': True, 'action_delta': True, 'clearance': True, 'knee_left': True, 'knee_right': True, 'knee_asymmetry': True}`.

Signed-pitch candidate minus source: `{'mean': -0.001026943325996399, 'p05': -0.002094358205795288, 'p50': -0.0005534142255783081, 'p95': -0.00035694241523742676}`.

The update was technically valid and safety-bounded, but it is rejected unless every preregistered efficacy check passes. No long training, export, or deployment was unlocked.
