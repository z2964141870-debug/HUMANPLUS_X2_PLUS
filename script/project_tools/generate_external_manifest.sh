#!/usr/bin/env bash
set -euo pipefail

repo_root="$(git rev-parse --show-toplevel)"
output="$repo_root/data/manifests/external_artifacts_20260831.tsv"
tmp_output="${output}.tmp"
remote_root='HUMAN+/HUMANPLUS_X2_PLUS/humanplus_sonic_x2/2026-08-31'

sources=(
  /Users/yu/projects/sonic_x2_transfer_v2/models
  /Users/yu/projects/sonic_x2_transfer_v2/assets
  /Users/yu/projects/sonic_x2_transfer_v2/GR00T_audit/gear_sonic_deploy/analysis_logs
  /Users/yu/Documents/ChatGPT/X2/_pending_baidu
)

file_size() {
  stat -f '%z' "$1" 2>/dev/null || stat -c '%s' "$1"
}

category_for() {
  case "$1" in
    *.onnx|*.pt|*.pth|*.ckpt) echo models ;;
    *analysis_logs*|*/logs/*|*.db3|*.mcap|*.bag) echo logs ;;
    *.tar|*.tar.gz|*.tar.zst|*.tgz|*.zip) echo archives ;;
    *) echo data ;;
  esac
}

printf 'local_path\tbytes\tsha256\tcategory\tremote_directory\tupload_status\tverification\n' >"$tmp_output"

append_file() {
  local item="$1"
  local category
  category="$(category_for "$item")"
  printf '%s\t%s\t%s\t%s\t%s/%s\tpending\tnot_uploaded\n' \
    "$item" \
    "$(file_size "$item")" \
    "$(shasum -a 256 "$item" | awk '{print $1}')" \
    "$category" \
    "$remote_root" \
    "$category" >>"$tmp_output"
}

for source_item in "${sources[@]}"; do
  if [[ -f "$source_item" ]]; then
    append_file "$source_item"
  elif [[ -d "$source_item" ]]; then
    while IFS= read -r -d '' artifact_file; do
      append_file "$artifact_file"
    done < <(find "$source_item" -type f -print0 | sort -z)
  else
    echo "warning: manifest source is absent: $source_item" >&2
  fi
done

mv "$tmp_output" "$output"
echo "wrote $output"
