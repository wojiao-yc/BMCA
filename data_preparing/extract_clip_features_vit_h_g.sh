#!/usr/bin/env bash
set -euo pipefail

SCRIPT="/home/wenxiao/workspace/qhy/BMCA/data_preparing/extract_clip_features.py"
CONFIG="${CONFIG:-/home/wenxiao/workspace/qhy/BrainFLORA/configs/config.yaml}"
OUTPUT_DIR="${OUTPUT_DIR:-/home/wenxiao/workspace/qhy/BMCA/data/clip_features}"
GPU="${GPU:-cuda:3}"
TEXT_BATCH_SIZE="${TEXT_BATCH_SIZE:-256}"
IMAGE_BATCH_SIZE="${IMAGE_BATCH_SIZE:-20}"

MODELS=("ViT-H-14" "ViT-g-14")
PRETRAINED=("laion2b_s32b_b79k" "laion2b_s34b_b88k")

if [[ ! -f "${SCRIPT}" ]]; then
  echo "Missing script: ${SCRIPT}" >&2
  exit 1
fi
if [[ ! -f "${CONFIG}" ]]; then
  echo "Missing config: ${CONFIG}" >&2
  echo "Set CONFIG to a YAML file that defines eegdataset.img_directory_training/test." >&2
  exit 1
fi
if [[ "${#MODELS[@]}" -ne "${#PRETRAINED[@]}" ]]; then
  echo "MODELS and PRETRAINED length mismatch." >&2
  exit 1
fi

mkdir -p "${OUTPUT_DIR}"

EXTRA_ARGS=()
if [[ -n "${USE_CPU:-}" ]]; then
  EXTRA_ARGS+=(--cpu)
else
  EXTRA_ARGS+=(--gpu "${GPU}")
fi

for i in "${!MODELS[@]}"; do
  model_type="${MODELS[$i]}"
  pretrained="${PRETRAINED[$i]}"
  for mode in train test; do
    output="${OUTPUT_DIR}/${model_type}_${pretrained}_${mode}.pt"
    python "${SCRIPT}" \
      --config "${CONFIG}" \
      --model-type "${model_type}" \
      --pretrained "${pretrained}" \
      --mode "${mode}" \
      --output "${output}" \
      --text-batch-size "${TEXT_BATCH_SIZE}" \
      --image-batch-size "${IMAGE_BATCH_SIZE}" \
      "${EXTRA_ARGS[@]}"
  done
done
