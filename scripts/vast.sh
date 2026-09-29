#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."
mode="${1:-smoke}"
case "$mode" in smoke|pilot|full) ;; *) echo 'Usage: bash scripts/vast.sh [smoke|pilot|full]'; exit 2 ;; esac
experiment_name="${RUN_NAME:-${mode}_$(date -u +%Y%m%dT%H%M%SZ)}"
[[ "$experiment_name" =~ ^[a-zA-Z0-9_-]+$ ]] || { echo 'RUN_NAME must contain only letters, digits, _ or -'; exit 2; }
experiment_out="runs/$experiment_name"
mkdir -p "$experiment_out" .cache/matplotlib
exec > >(tee -a "$experiment_out/driver.log") 2>&1
export HF_HOME="${HF_HOME:-$PWD/.cache/huggingface}"
export MPLCONFIGDIR="$PWD/.cache/matplotlib"
export MPLBACKEND=Agg
export TOKENIZERS_PARALLELISM=false
export PYTHONUNBUFFERED=1
python3 - <<'PY'
import sys
assert (3,10) <= sys.version_info[:2] < (3,13), 'Use Python 3.10-3.12'
PY
if [[ ! -x .venv/bin/python ]]; then python3 -m venv .venv; fi
source .venv/bin/activate
if [[ "${SKIP_INSTALL:-0}" != 1 ]]; then
  python -m pip install --upgrade pip
  python -m pip install torch==2.6.0 --index-url https://download.pytorch.org/whl/cu124
  python -m pip install -e '.[gpu]'
fi
python - <<'PY'
import torch
assert torch.cuda.is_available(), 'CUDA unavailable: use a GPU CUDA image and compatible NVIDIA driver'
assert torch.cuda.is_bf16_supported(), 'BF16 support required'
print('GPU:', torch.cuda.get_device_name(0))
print('VRAM GiB:', torch.cuda.get_device_properties(0).total_memory / 2**30)
PY
python -m unittest discover -s tests -v
python -m pip freeze > "$experiment_out/packages.txt"
nvidia-smi > "$experiment_out/nvidia-smi.txt"
case "$mode" in
  smoke) count="${SAMPLES:-2}"; length="${LENGTH:-64}" ;;
  pilot) count="${SAMPLES:-100}"; length="${LENGTH:-256}" ;;
  full) count="${SAMPLES:-500}"; length="${LENGTH:-256}" ;;
esac
# Fresh forward pass over the full response. Two paired policies on exactly the same questions.
dataset_args=()
for policy in top1 threshold; do
  python -m confidence_geography.run \
    --out "$experiment_out/$policy" --policy "$policy" \
    --samples "$count" --length "$length" --block-length "${BLOCK_LENGTH:-0}" \
    --seed "${SEED:-1729}" --offset "${OFFSET:-0}" \
    --commit-threshold "${COMMIT_THRESHOLD:-0.9}" "${dataset_args[@]}"
  python -m confidence_geography.analyze \
    --run "$experiment_out/$policy" --out "$experiment_out/$policy/analysis" \
    --max-plots "${MAX_PLOTS:-8}"
  dataset_revision="$(python -c 'import json,sys; print(json.load(open(sys.argv[1]))["dataset_info"]["revision"])' "$experiment_out/$policy/manifest.json")"
  dataset_args=(--dataset-revision "$dataset_revision")
done
if [[ "${RUN_LTR:-0}" == 1 ]]; then
  python -m confidence_geography.run --out "$experiment_out/left_to_right" \
    --policy left_to_right --samples "$count" --length "$length" \
    --block-length "${BLOCK_LENGTH:-0}" --seed "${SEED:-1729}" --offset "${OFFSET:-0}" "${dataset_args[@]}"
  python -m confidence_geography.analyze --run "$experiment_out/left_to_right" \
    --out "$experiment_out/left_to_right/analysis" --max-plots "${MAX_PLOTS:-8}"
fi
bash scripts/export.sh "$experiment_out"
echo "Complete: $experiment_out"
