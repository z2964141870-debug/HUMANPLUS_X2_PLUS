# Script

- `sonic_x2_transfer_v2/`: current source snapshot, launchers, C++ controller,
  safety proxy, offload client/server, tests, and integration documentation.
- `smartwear_runtime/`: Mac-side copies of the garment/offload runtime files
  used during migration and profiling.
- `gmr_cpp_prototype/`: native low-latency GMR prototype.
- `legacy_mac_tools/`: earlier X2 analysis and replay utilities retained for
  provenance.
- `project_tools/`: repository storage and payload audits.

Generated builds, virtual environments, compiled runtimes, model weights, and
credentials are excluded. The robot deployment tree is synchronized only
after exact source/destination SHA-256 comparison.
