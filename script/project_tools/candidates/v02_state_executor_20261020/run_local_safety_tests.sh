#!/usr/bin/env bash
# Endpoint-free tests only. No ROS or remote connection is used.
set -euo pipefail
D="$(cd -- "$(dirname -- "$0")" && pwd)"
BUILD="$(mktemp -d /tmp/x2-plan-a-unit.XXXXXX)"
# Keep artifacts for inspection; do not delete computed paths in this runner.
printf 'Local test artifacts: %s\n' "$BUILD"
for sanitizer in address,undefined thread; do
  for name in test_state_executor_worker test_command_publication_gate test_independent_writer_liveness; do
    clang++ -std=c++17 -pthread -g -fsanitize="$sanitizer" "$D/$name.cpp" -o "$BUILD/$name-${sanitizer}"
    "$BUILD/$name-${sanitizer}"
  done
done
python3 "$D/test_offline_guard_source.py"
python3 "$D/test_harness_cleanup_model.py"
# Process-group signal tests are separate: host process policy may reject killpg.
