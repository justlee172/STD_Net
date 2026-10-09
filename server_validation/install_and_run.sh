#!/usr/bin/env bash
set -euo pipefail

repo_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$repo_dir"
export PYTHONPATH="$repo_dir${PYTHONPATH:+:$PYTHONPATH}"

python_bin="${PYTHON_BIN:-python3}"
venv_dir="${VENV_DIR:-$repo_dir/.venv_stdnet_paper}"
cuda_backend="${CUDA_BACKEND:-cu121}"
run_mode="${RUN_MODE:-quick}"
seeds="${SEEDS:-0,1,2,3,4}"
epochs="${EPOCHS:-1500}"
pseudo_spots="${PSEUDO_SPOTS:-30000}"
jobs="${JOBS:-1}"
datasets="${DATASETS:-}"
protocol="${PROTOCOL:-standard}"
profile="${PROFILE:-full_v2}"
raw_sc_counts="${RAW_SC_COUNTS:-$repo_dir/original_data/Lohoff_seqFISH/counts.txt}"

"$python_bin" -c 'import sys; assert (3, 9) <= sys.version_info[:2] <= (3, 11), "Use Python 3.9, 3.10, or 3.11"'

if [[ ! -x "$venv_dir/bin/python" ]]; then
  "$python_bin" -m venv "$venv_dir"
fi

python_exec="$venv_dir/bin/python"
pip_exec="$venv_dir/bin/pip"
"$pip_exec" install --upgrade pip setuptools wheel

case "$cuda_backend" in
  cu121|cu118)
    "$pip_exec" install "torch==2.3.0" --index-url "https://download.pytorch.org/whl/$cuda_backend"
    pyg_wheel="https://data.pyg.org/whl/torch-2.3.0+${cuda_backend}.html"
    ;;
  cpu)
    "$pip_exec" install "torch==2.3.0" --index-url "https://download.pytorch.org/whl/cpu"
    pyg_wheel="https://data.pyg.org/whl/torch-2.3.0+cpu.html"
    ;;
  *)
    echo "CUDA_BACKEND must be cu121, cu118, or cpu" >&2
    exit 2
    ;;
esac

"$pip_exec" install \
  "pyg_lib" "torch_scatter" "torch_sparse" "torch_cluster" "torch_spline_conv" \
  -f "$pyg_wheel"
"$pip_exec" install -r server_validation/requirements-paper-aligned.txt

"$python_exec" server_validation/preflight.py
"$python_exec" -m unittest server_validation/test_paper_alignment.py -v

run_arguments=(
  --raw-sc-counts "$raw_sc_counts"
  --seeds "$seeds"
  --epochs "$epochs"
  --pseudo-spots "$pseudo_spots"
  --jobs "$jobs"
  --protocol "$protocol"
  --profile "$profile"
)
if [[ -n "$datasets" ]]; then
  run_arguments+=(--datasets "$datasets")
fi
if [[ "$run_mode" == "quick" ]]; then
  run_arguments+=(--quick)
elif [[ "$run_mode" != "formal" ]]; then
  echo "RUN_MODE must be quick or formal" >&2
  exit 2
fi

mkdir -p server_logs
log_file="server_logs/${profile}_${protocol}_${run_mode}_$(date +%Y%m%d_%H%M%S).log"
if [[ "$run_mode" == "formal" ]]; then
  echo "Running a two-epoch smoke test before the formal jobs"
  "$python_exec" server_validation/run_paper_aligned.py \
    --raw-sc-counts "$raw_sc_counts" \
    --seeds "${seeds%%,*}" --epochs 2 --pseudo-spots 100 --jobs "$jobs" \
    --protocol "$protocol" --profile "$profile" --quick
fi
"$python_exec" server_validation/run_paper_aligned.py "${run_arguments[@]}" 2>&1 | tee "$log_file"
