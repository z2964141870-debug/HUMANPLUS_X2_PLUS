#!/usr/bin/env bash
set -euo pipefail

ATTEMPTS="${BDPAN_UPLOAD_ATTEMPTS:-4}"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
if PROJECT_ROOT="$(git -C "$SCRIPT_DIR" rev-parse --show-toplevel 2>/dev/null)"; then
  :
else
  PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
fi
MANIFEST_DIR="${BDPAN_UPLOAD_MANIFEST_DIR:-$PROJECT_ROOT/docs/backup/manifests}"

usage() {
  echo "usage: $0 LOCAL_FILE REMOTE_DIRECTORY" >&2
  echo "remote paths are relative to Baidu's /apps/bdpan/ directory" >&2
}

if [[ $# -ne 2 ]]; then
  usage
  exit 64
fi

local_file="$1"
remote_dir="${2%/}"
remote_name="$(basename "$local_file")"

if [[ ! -f "$local_file" ]]; then
  echo "local file does not exist: $local_file" >&2
  exit 66
fi

if [[ -z "$remote_dir" || "$remote_dir" == /* || "$remote_dir" == *".."* || "$remote_dir" == *"~"* ]]; then
  echo "unsafe remote directory: $remote_dir" >&2
  exit 65
fi

for command_name in bdpan jq sha256sum stat flock; do
  if ! command -v "$command_name" >/dev/null 2>&1; then
    echo "required command is unavailable: $command_name" >&2
    exit 69
  fi
done

if ! bdpan whoami 2>/dev/null | grep -q "已登录"; then
  echo "bdpan is not logged in; use the official baidu-drive login.sh first." >&2
  exit 2
fi

local_size="$(stat -c '%s' "$local_file")"
local_sha256="$(sha256sum "$local_file" | awk '{print $1}')"
remote_path="$remote_dir/$remote_name"
safe_id="$(printf '%s' "$remote_path" | sha256sum | awk '{print $1}')"
mkdir -p "$MANIFEST_DIR"
manifest_path="$MANIFEST_DIR/${safe_id}.json"
lock_path="/tmp/x2-bdpan-${safe_id}.lock"

exec 9>"$lock_path"
if ! flock -n 9; then
  echo "another upload owns the lock for: $remote_path" >&2
  exit 75
fi

list_remote() {
  bdpan ls "$remote_dir" --json
}

remote_size_from_json() {
  jq -r --arg name "$remote_name" \
    'if type == "array" then
       .[] | select(.server_filename == $name and .isdir == false) | .size
     else
       empty
     end' \
    | head -n 1
}

write_manifest() {
  local status="$1"
  local verified_size="$2"
  local tmp_manifest="${manifest_path}.tmp"
  jq -n \
    --arg status "$status" \
    --arg local_path "$local_file" \
    --arg remote_path "$remote_path" \
    --arg sha256 "$local_sha256" \
    --arg uploaded_at "$(date --iso-8601=seconds)" \
    --argjson local_size "$local_size" \
    --argjson remote_size "$verified_size" \
    '{
      status: $status,
      local_path: $local_path,
      remote_path_relative_to_apps_bdpan: $remote_path,
      local_size: $local_size,
      remote_size: $remote_size,
      local_sha256: $sha256,
      verification: "remote path and byte size only; cryptographic readback pending",
      uploaded_at: $uploaded_at
    }' >"$tmp_manifest"
  mv "$tmp_manifest" "$manifest_path"
}

existing_json=""
preflight_ok="false"
for preflight_attempt in 1 2 3; do
  if existing_json="$(list_remote 2>/dev/null)" \
    && printf '%s\n' "$existing_json" | jq -e 'type == "array"' >/dev/null; then
    preflight_ok="true"
    break
  fi
  sleep 2
done

if [[ "$preflight_ok" != "true" ]]; then
  echo "remote preflight list did not return a valid array; refusing to upload." >&2
  exit 4
fi

existing_size="$(printf '%s\n' "$existing_json" | remote_size_from_json)"
if [[ -n "$existing_size" ]]; then
  if [[ "$existing_size" == "$local_size" ]]; then
    write_manifest "already_present_same_size" "$existing_size"
    echo "remote artifact already exists with matching size: $remote_path"
    echo "manifest: $manifest_path"
    exit 0
  fi
  echo "refusing to overwrite remote artifact with different size: $remote_path" >&2
  echo "remote=$existing_size local=$local_size" >&2
  exit 73
fi

delays=(2 5 15)
for ((attempt=1; attempt<=ATTEMPTS; attempt++)); do
  echo "upload attempt $attempt/$ATTEMPTS: $local_file -> $remote_path"
  if bdpan upload "$local_file" "$remote_path"; then
    for verify_attempt in 1 2 3; do
      if remote_json="$(list_remote 2>/dev/null)"; then
        verified_size="$(printf '%s\n' "$remote_json" | remote_size_from_json)"
        if [[ "$verified_size" == "$local_size" ]]; then
          write_manifest "uploaded_size_verified" "$verified_size"
          echo "verified remote byte size: $verified_size"
          echo "local sha256: $local_sha256"
          echo "manifest: $manifest_path"
          exit 0
        fi
      fi
      sleep 2
    done
    echo "upload returned success, but remote size verification did not pass." >&2
  fi

  if ((attempt < ATTEMPTS)); then
    delay_index=$((attempt - 1))
    sleep "${delays[$delay_index]}"
  fi
done

echo "upload failed after $ATTEMPTS attempts; source file remains unchanged." >&2
exit 3
