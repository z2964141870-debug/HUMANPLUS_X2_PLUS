#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT=/home/humanplus/projects/ZHY/CWI_CrossEmbodiment_Sim
OUT_DIR=${1:-${PROJECT_ROOT}/backups/migration_20260810/runtime}
PREFIX=${OUT_DIR}/x2_aimdk_runtime_images_v1.tar.gz.part-

mkdir -p "${OUT_DIR}"
rm -f "${PREFIX}"* "${OUT_DIR}/x2_aimdk_runtime_images_v1.parts.sha256" \
  "${OUT_DIR}/x2_aimdk_runtime_images_v1.images.txt"

docker image inspect x2-aimdk-humble:1.0 g1-deploy-dev:latest \
  --format '{{.RepoTags}} {{.Id}} {{.Size}} {{.Architecture}}' \
  > "${OUT_DIR}/x2_aimdk_runtime_images_v1.images.txt"

# Save both tags together so their common base layers are stored only once.
# Each part stays below 3 GiB to avoid the known large-part-count failure mode
# of the official bdpan client. Restore with:
#   cat x2_aimdk_runtime_images_v1.tar.gz.part-* | gzip -dc | docker load
docker save x2-aimdk-humble:1.0 g1-deploy-dev:latest \
  | gzip -1 \
  | split -b 3072m -d -a 2 - "${PREFIX}"

sha256sum "${PREFIX}"* > "${OUT_DIR}/x2_aimdk_runtime_images_v1.parts.sha256"
stat --format='%n %s bytes' "${PREFIX}"*
cat "${OUT_DIR}/x2_aimdk_runtime_images_v1.parts.sha256"
