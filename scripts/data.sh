#!/usr/bin/env bash
set -euo pipefail

RADII=(3 5 7 9 11 21 31 41)

for r in "${RADII[@]}"; do
  for mode in train test; do
    python BMCA/data/extract_clip_features_blur.py \
      --model-type RN50 \
      --pretrained openai \
      --mode "$mode" \
      --blur-radius "$r" \
      --output "RN50_openai_${mode}_blur${r}.pt"
  done
done
