# GMR native prototype

This directory is an isolated offline prototype. It does not publish HAL commands,
change MC state, or modify the deployed garment/launcher files.

The current GMR already uses native MuJoCo and DAQP libraries. Profiling found a
Python call-site bug that rebuilt `mink.ConfigurationLimit` for every IK solve:
the cached limits list was passed as the positional `safety_break` argument.
`LimitsFixedGMR` passes it as `limits=self.ik_limits`.

`native_preprocess.cpp` additionally fuses human-frame scaling, quaternion offset,
position-offset rotation, and ground offset into one C++ call. The IK solve remains
Mink/MuJoCo/DAQP so numerical behavior can be compared before a larger rewrite.

Build and run the standalone kernel test:

```bash
./build.sh
python3 test_native_preprocess.py
```

On the X2 SoC1, with the existing shadow environment exported, run:

```bash
./build.sh
python benchmark_gmr.py --frames 220 --warmup 20 --max-iter 2
```

The benchmark reports baseline, cached-limit fix, native preprocessing timing, and
absolute qpos error. It is offline and creates no BLE, UDP, HAL, MC, or MuJoCo process.
