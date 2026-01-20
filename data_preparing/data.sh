# !/usr/bin/env bash
set -euo pipefail

RADII=(51 41 31 15 7 5 3)

for r in "${RADII[@]}"; do
  # for mode in train test; do
  for mode in train test; do
    python /home/wenxiao/workspace/qhy/BMCA/data/extract_clip_features_blur.py \
      --model-type RN50 \
      --pretrained openai \
      --mode "$mode" \
      --blur-radius "$r" \
      --output "RN50_openai_${mode}_g3_blur${r}.pt"
  done
done

# python /home/wenxiao/workspace/qhy/BMCA/data/extract_clip_features_blur.py \
#   --model-type RN50 \
#   --pretrained openai \
#   --mode "test" \
#   --blur-radius "51" \
#   --output "RN50_openai_test_g3_blur51.pt"
