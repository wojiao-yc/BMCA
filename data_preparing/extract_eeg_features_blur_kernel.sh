#!/usr/bin/env bash
set -euo pipefail

SCRIPT="/home/wenxiao/workspace/qhy/BMCA/data_preparing/extract_eeg_features.py"
CONFIG="${CONFIG:-/home/wenxiao/workspace/qhy/BMCA/configs/ubp.yaml}"
DATA_ROOT="${DATA_ROOT:-/mnt/dataset4/qhy/THINGS-EEG/things_eeg/Preprocessed_data_250Hz_whiten/sub-01}"
TRAIN_PT="${TRAIN_PT:-${DATA_ROOT}/train.pt}"
TEST_PT="${TEST_PT:-${DATA_ROOT}/test.pt}"
OUTPUT_DIR="${OUTPUT_DIR:-/home/wenxiao/workspace/qhy/BMCA/data}"
MODEL_TYPE="${MODEL_TYPE:-RN50}"
DEVICE="${DEVICE:-auto}"
IMAGE_BATCH_SIZE="${IMAGE_BATCH_SIZE:-128}"
TEXT_BATCH_SIZE="${TEXT_BATCH_SIZE:-256}"
BLUR_CURVE="${BLUR_CURVE:-exp}"
BLUR_SYSTEM_G="${BLUR_SYSTEM_G:-3}"
BLUR_H="${BLUR_H:-224}"
BLUR_W="${BLUR_W:-224}"
PRETRAINED="${PRETRAINED:-}"

KERNEL_SIZES=(1)

EXTRA_ARGS=()
if [[ -n "${IMAGE_ROOT:-}" ]]; then
  EXTRA_ARGS+=(--image-root "${IMAGE_ROOT}")
fi
if [[ -n "${BLUR_CURVE}" ]]; then
  EXTRA_ARGS+=(--blur-curve "${BLUR_CURVE}")
fi
if [[ -n "${BLUR_SYSTEM_G}" ]]; then
  EXTRA_ARGS+=(--blur-system-g "${BLUR_SYSTEM_G}")
fi
if [[ -n "${BLUR_H}" ]]; then
  EXTRA_ARGS+=(--blur-h "${BLUR_H}")
fi
if [[ -n "${BLUR_W}" ]]; then
  EXTRA_ARGS+=(--blur-w "${BLUR_W}")
fi
if [[ -n "${PRETRAINED}" ]]; then
  EXTRA_ARGS+=(--pretrained "${PRETRAINED}")
fi

if [[ ! -f "${TRAIN_PT}" ]]; then
  echo "Train .pt not found: ${TRAIN_PT}" >&2
  exit 1
fi
if [[ ! -f "${TEST_PT}" ]]; then
  echo "Test .pt not found: ${TEST_PT}" >&2
  exit 1
fi

mkdir -p "${OUTPUT_DIR}"

for k in "${KERNEL_SIZES[@]}"; do
  for mode in train test; do
    if [[ "${mode}" == "train" ]]; then
      data_path="${TRAIN_PT}"
    else
      data_path="${TEST_PT}"
    fi
    output="${OUTPUT_DIR}/${MODEL_TYPE}_${mode}_blur${k}.pt"
    # output="${OUTPUT_DIR}/${MODEL_TYPE}_resize.pt"
    python "${SCRIPT}" \
      --config "${CONFIG}" \
      --mode "${mode}" \
      --data-path "${data_path}" \
      --model-type "${MODEL_TYPE}" \
      --device "${DEVICE}" \
      --image-batch-size "${IMAGE_BATCH_SIZE}" \
      --text-batch-size "${TEXT_BATCH_SIZE}" \
      --blur-kernel-size "${k}" \
      --output "${output}" \
      "${EXTRA_ARGS[@]}"
  done
done
