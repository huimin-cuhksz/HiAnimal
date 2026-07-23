#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
GPU_ID="${GPU_ID:-${CUDA_VISIBLE_DEVICES:-7}}"
ANIMAL="${1:-cat}"

case "${ANIMAL}" in
    cat|sheep) ;;
    *)
        echo "Usage: $0 [cat|sheep]" >&2
        exit 2
        ;;
esac

cd "${ROOT_DIR}/code"
CUDA_VISIBLE_DEVICES="${GPU_ID}" python launch.py \
    --config custom/threestudio_meshfitting/configs/registration.yaml \
    --train \
    "data.target_mesh=../../hianimal_train/test_results/${ANIMAL}/${ANIMAL}_prediction.obj" \
    "data.corr_path=../inputs/${ANIMAL}/correspondence.npy" \
    "tag=${ANIMAL}-registration"

RUN_DIR="$(find "${ROOT_DIR}/runs/hianimal-registration" \
    -mindepth 1 -maxdepth 1 -type d -name "${ANIMAL}-registration@*" \
    -printf '%f\n' | sort | tail -n 1)"
mkdir -p "${ROOT_DIR}/result"
cp "${ROOT_DIR}/runs/hianimal-registration/${RUN_DIR}/save/deform_mesh.obj" \
    "${ROOT_DIR}/result/${ANIMAL}_registered_5000.obj"
echo "Saved result/${ANIMAL}_registered_5000.obj"
