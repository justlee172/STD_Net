#!/usr/bin/env bash
set -euo pipefail

repo_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$repo_dir"

python_bin="${PYTHON_BIN:-$repo_dir/.venv_stdnet_paper/bin/python}"
cuda_visible_devices="${CUDA_VISIBLE_DEVICES:-0}"
seeds="${SEEDS:-0,1,2,3,4}"
epochs="${EPOCHS:-1500}"
pseudo_spots="${PSEUDO_SPOTS:-30000}"
jobs="${JOBS:-1}"
export CUDA_VISIBLE_DEVICES="$cuda_visible_devices"
export PYTHONPATH="$repo_dir${PYTHONPATH:+:$PYTHONPATH}"
export MPLCONFIGDIR="${MPLCONFIGDIR:-/tmp/mpl}"

"$python_bin" - <<'PY'
import torch

if not torch.cuda.is_available() or torch.cuda.device_count() < 1:
    raise SystemExit("CUDA is unavailable; refusing to run the rare-type GPU ablation")
print(f"GPU: {torch.cuda.get_device_name(0)}")
PY

run_one() {
    local profile="$1"
    local output_root="$2"
    local log_file="$3"
    if [[ -e "$output_root" ]]; then
        echo "Output already exists: $output_root" >&2
        exit 2
    fi
    echo "START profile=$profile output=$output_root"
    "$python_bin" server_validation/run_paper_aligned.py \
        --profile "$profile" \
        --seeds "$seeds" \
        --epochs "$epochs" \
        --pseudo-spots "$pseudo_spots" \
        --jobs "$jobs" \
        --output "$output_root" 2>&1 | tee "$log_file"
}

timestamp="$(date +%Y%m%d_%H%M%S)"
mkdir -p server_logs
run_one \
    support_calibrated \
    "paper_aligned_runs/rare_baseline_support_gpu_${timestamp}" \
    "server_logs/rare_baseline_support_gpu_${timestamp}.log"
run_one \
    support_rare_balanced \
    "paper_aligned_runs/rare_balanced_gpu_${timestamp}" \
    "server_logs/rare_balanced_gpu_${timestamp}.log"

echo "RARE-BALANCED GPU ABLATION COMPLETED"
