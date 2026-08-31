X2 Sonic shadow migration bundle
================================

Purpose
-------
This bundle prepares another AgiBot X2 Ultra for read-only Sonic shadow tests.
It does not authorize or enable HAL publishing, Develop_MC takeover, or live motion.

Expected target layout
----------------------
Extract the archive under /home/agi so these paths exist:

  /home/agi/projects/X2_sonic_real
  /home/agi/projects/smartwear_v2
  /home/agi/projects/humanplus_x2_bridge
  /home/agi/mocap_teleop/x2_pc1_ble_teleop

Target-machine prerequisites (not bundled)
------------------------------------------
- AgiBot X2 Ultra with compatible 31-joint layout and v1.0.x firmware.
- Robot-provided AIMDK/ROS installation matching its firmware.
- CUDA-capable Python environment at /home/agi/venvs/x2_sonic_cuda.
- Working AX210 Bluetooth controller accessible by user agi.
- Jacket FB:08:4D:3B:D6:06 and pants CC:D1:FA:DD:6D:B8, or update the script variables.

Model verification
------------------
The transfer-v2 model must have SHA256:

  8ccc42a82ea2c446aa708aece58c7a638ea943a79ed88eeab259669e641271e9

Safe first test
---------------
Run only the read-only shadow entry point:

  cd /home/agi/projects/X2_sonic_real
  ./run_x2_ax210_full_body_shadow.sh

The script must state that no HAL publisher or system-state migration is created.
Do not use live-control scripts until closed-loop simulation and handoff validation pass.
