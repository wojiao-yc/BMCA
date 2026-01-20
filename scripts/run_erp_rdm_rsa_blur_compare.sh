#!/usr/bin/env bash
set -euo pipefail

PYTHON_BIN="${PYTHON_BIN:-python}"
SCRIPT="${SCRIPT:-/home/wenxiao/workspace/qhy/BMCA/erp_rdm_rsa.py}"
DATA_PATH="${DATA_PATH:-/mnt/dataset1/ldy/datasets/datasets/THINGS_EEG/Preprocessed_data_250Hz}"
IMG_DIR="${IMG_DIR:-/home/wenxiao/workspace/qhy/BMCA/data}"
BLUR_REF="${BLUR_REF:-1}"
T0="${T0:-0.0}"
T1="${T1:-1.0}"
WIN_MS="${WIN_MS:-50}"
STEP_MS="${STEP_MS:-10}"
OUT_DIR="${OUT_DIR:-/home/wenxiao/workspace/qhy/BMCA/results/erp_rdm_rsa}"

BLUR_LEVELS=(0)
# BLUR_LEVELS=(51 41 31 1)
SUBJECTS=(sub-01 sub-02 sub-03 sub-04 sub-05 sub-06 sub-07 sub-08 sub-09 sub-10)

mkdir -p "${OUT_DIR}"

blur_ref_pt="${IMG_DIR}/RN50_test_blur${BLUR_REF}.pt"
if [[ ! -f "${blur_ref_pt}" ]]; then
  echo "Missing blur reference pt: ${blur_ref_pt}" >&2
  exit 1
fi

for blur in "${BLUR_LEVELS[@]}"; do
  # orig_pt="${IMG_DIR}/RN50_test_blur${blur}.pt"
  orig_pt="/home/wenxiao/workspace/qhy/BMCA/data/RN50_openai_test.pt"
  if [[ ! -f "${orig_pt}" ]]; then
    echo "Missing orig pt: ${orig_pt} (skip)" >&2
    continue
  fi

  for sub in "${SUBJECTS[@]}"; do
    out_npz="${OUT_DIR}/rsa_raw_eeg_${sub}_orig${blur}_vs${BLUR_REF}.npz"
    plot_png="${OUT_DIR}/rsa_raw_eeg_${sub}_orig${blur}_vs${BLUR_REF}.png"
    "${PYTHON_BIN}" "${SCRIPT}" \
      --data_path "${DATA_PATH}" \
      --img_pt_orig "${orig_pt}" \
      --img_pt_blur "${blur_ref_pt}" \
      --t0 "${T0}" \
      --t1 "${T1}" \
      --win_ms "${WIN_MS}" \
      --step_ms "${STEP_MS}" \
      --out_npz "${out_npz}" \
      --plot_png "${plot_png}" \
      --subjects "${sub}"
  done

  out_npz="${OUT_DIR}/rsa_raw_eeg_all_orig${blur}_vs${BLUR_REF}.npz"
  plot_png="${OUT_DIR}/rsa_raw_eeg_all_orig${blur}_vs${BLUR_REF}.png"
  "${PYTHON_BIN}" "${SCRIPT}" \
    --data_path "${DATA_PATH}" \
    --img_pt_orig "${orig_pt}" \
    --img_pt_blur "${blur_ref_pt}" \
    --t0 "${T0}" \
    --t1 "${T1}" \
    --win_ms "${WIN_MS}" \
    --step_ms "${STEP_MS}" \
    --out_npz "${out_npz}" \
    --plot_png "${plot_png}" \
    --subjects "${SUBJECTS[@]}"
done
