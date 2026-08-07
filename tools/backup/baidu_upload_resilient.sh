#!/usr/bin/env bash
set -u

PCS="${PCS:-/home/humanplus/projects/ZHY/baidupcs}"
CONFIG_DIR="${BAIDUPCS_GO_CONFIG_DIR:-/home/humanplus/projects/ZHY/.baidupcs_go}"
ATTEMPTS="${BAIDU_UPLOAD_ATTEMPTS:-4}"

usage() {
  echo "usage: $0 LOCAL_FILE REMOTE_DIRECTORY" >&2
}

if [[ $# -ne 2 ]]; then
  usage
  exit 64
fi

local_file="$1"
remote_dir="${2%/}"
remote_name="$(basename "$local_file")"
export BAIDUPCS_GO_CONFIG_DIR="$CONFIG_DIR"

if [[ ! -f "$local_file" ]]; then
  echo "local file does not exist: $local_file" >&2
  exit 66
fi

if ! "$PCS" who >/dev/null 2>&1; then
  echo "BaiduPCS-Go session is unavailable; refresh the saved BDUSS + STOKEN cookie once." >&2
  exit 2
fi

delays=(2 5 15 30)
for ((attempt=1; attempt<=ATTEMPTS; attempt++)); do
  log_file="$(mktemp /tmp/baidupcs-upload.XXXXXX.log)"
  "$PCS" upload "$local_file" "$remote_dir/" 2>&1 | tee "$log_file"
  if ! rg -q "以下文件上传失败|获取用户uk错误|网络错误" "$log_file"; then
    if "$PCS" ls "$remote_dir" 2>/dev/null | rg -Fq "$remote_name"; then
      rm -f "$log_file"
      echo "verified remote artifact: $remote_dir/$remote_name"
      exit 0
    fi
  fi
  rm -f "$log_file"
  if ((attempt < ATTEMPTS)); then
    delay_index=$((attempt - 1))
    sleep "${delays[$delay_index]}"
  fi
done

echo "upload failed after $ATTEMPTS attempts; the local artifact is intact." >&2
echo "Run '$PCS who'. Only refresh BDUSS + STOKEN if account lookup also fails or repeated UK errors persist." >&2
exit 3
