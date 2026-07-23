#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ANIMAL="${1:-cat}"

case "${ANIMAL}" in
    cat|sheep) ;;
    *)
        echo "Usage: $0 [cat|sheep]" >&2
        exit 2
        ;;
esac

PREDICTION_DIR="${ROOT_DIR}/../hianimal_train/test_results/${ANIMAL}"

python "${ROOT_DIR}/correspondence/generate_correspondence.py" \
    --target-obj "${PREDICTION_DIR}/${ANIMAL}_prediction.obj" \
    --target-npz "${PREDICTION_DIR}/${ANIMAL}_prediction.npz" \
    --output-dir "${ROOT_DIR}/inputs/${ANIMAL}"
