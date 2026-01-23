#!/usr/bin/env bash
set -euo pipefail

# Wrapper for extract_eeg_features_degrade.py that runs lowres/jpeg/subject once.

SCRIPT="/home/wenxiao/workspace/qhy/BMCA/data_preparing/extract_eeg_features_degrade.py"
CONFIG="${CONFIG:-/home/wenxiao/workspace/qhy/BMCA/configs/ubp.yaml}"
DATA_ROOT="${DATA_ROOT:-/mnt/dataset4/qhy/THINGS-EEG/things_eeg/Preprocessed_data_250Hz_whiten/sub-01}"
TRAIN_PT="${TRAIN_PT:-${DATA_ROOT}/train.pt}"
TEST_PT="${TEST_PT:-${DATA_ROOT}/test.pt}"
OUTPUT_DIR="${OUTPUT_DIR:-/home/wenxiao/workspace/qhy/BMCA/data}"
MODEL_TYPE="${MODEL_TYPE:-ViT-g-14}"
DEVICE="${DEVICE:-auto}"
IMAGE_BATCH_SIZE="${IMAGE_BATCH_SIZE:-128}"
TEXT_BATCH_SIZE="${TEXT_BATCH_SIZE:-256}"

# lowres parameters
LOWRES_SCALE="${LOWRES_SCALE:-0.25}"

# jpeg parameters (single quality)
JPEG_QUALITY="${JPEG_QUALITY:-15}"

# subject parameters (only used when BLUR_METHOD=subject)
SUBJECT_BLUR_RADIUS="${SUBJECT_BLUR_RADIUS:-51}"
SUBJECT_RECT_SCALE="${SUBJECT_RECT_SCALE:-0.6}"
SUBJECT_MASK_BLUR="${SUBJECT_MASK_BLUR:-11}"
SUBJECT_ITER="${SUBJECT_ITER:-5}"

EXTRA_ARGS=()
if [[ -n "${IMAGE_ROOT:-}" ]]; then
  EXTRA_ARGS+=(--image-root "${IMAGE_ROOT}")
fi
if [[ -n "${PRETRAINED:-}" ]]; then
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

run_one() {
  local mode="$1"
  local data_path="$2"
  local tag="$3"
  local method="$4"
  shift 4
  local output="${OUTPUT_DIR}/${MODEL_TYPE}_${mode}_${tag}.pt"

  python "${SCRIPT}" \
    --config "${CONFIG}" \
    --mode "${mode}" \
    --data-path "${data_path}" \
    --model-type "${MODEL_TYPE}" \
    --device "${DEVICE}" \
    --image-batch-size "${IMAGE_BATCH_SIZE}" \
    --text-batch-size "${TEXT_BATCH_SIZE}" \
    --blur-method "${method}" \
    --output "${output}" \
    "$@" \
    "${EXTRA_ARGS[@]}"
}

for mode in train test; do
  if [[ "${mode}" == "train" ]]; then
    data_path="${TRAIN_PT}"
  else
    data_path="${TEST_PT}"
  fi

  tag="lowres${LOWRES_SCALE}"
  run_one "${mode}" "${data_path}" "${tag}" lowres \
    --lowres-scale "${LOWRES_SCALE}"

  tag="jpeg${JPEG_QUALITY}"
  run_one "${mode}" "${data_path}" "${tag}" jpeg \
    --jpeg-quality "${JPEG_QUALITY}"

  # tag="subject_r${SUBJECT_BLUR_RADIUS}_s${SUBJECT_RECT_SCALE}_m${SUBJECT_MASK_BLUR}"
  # run_one "${mode}" "${data_path}" "${tag}" subject \
  #   --subject-blur-radius "${SUBJECT_BLUR_RADIUS}" \
  #   --subject-rect-scale "${SUBJECT_RECT_SCALE}" \
  #   --subject-mask-blur "${SUBJECT_MASK_BLUR}" \
  #   --subject-iter "${SUBJECT_ITER}"
done
