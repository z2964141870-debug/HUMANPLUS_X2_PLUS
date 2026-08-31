#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CXX_BIN="${CXX:-c++}"

case "$(uname -s)" in
  Darwin)
    OUTPUT="$ROOT/libgmr_native.dylib"
    SHARED_FLAGS=(-dynamiclib)
    ;;
  Linux)
    OUTPUT="$ROOT/libgmr_native.so"
    SHARED_FLAGS=(-shared -fPIC)
    ;;
  *)
    echo "Unsupported platform: $(uname -s)" >&2
    exit 2
    ;;
esac

"$CXX_BIN" -O3 -DNDEBUG -std=c++17 "${SHARED_FLAGS[@]}" \
  "$ROOT/native_preprocess.cpp" -o "$OUTPUT"
echo "$OUTPUT"
