# Data

This directory contains only small, Git-suitable reproducibility data.

- `model_metadata/`: provenance and action-scale metadata, never the ONNX
  weight itself.
- `motions/`: small motion fixtures used by the current Sonic tools.
- `evidence/`: small fixed inputs or images needed to interpret a report.
- `approved_manifests/`: approval templates and future hash-frozen manifests.
- `manifests/`: inventories linking local large artifacts to Baidu Netdisk.

Raw telemetry, rosbag, large CSV/JSONL, model weights, complete asset trees,
and archives must not be added here. Their local files remain intact until a
verified Baidu copy exists.
