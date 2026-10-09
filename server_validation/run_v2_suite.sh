#!/usr/bin/env bash
set -euo pipefail

repo_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$repo_dir"

cuda_backend="${CUDA_BACKEND:-cu121}"
suite_mode="${SUITE_MODE:-primary}"
seeds="${SEEDS:-0,1,2,3,4}"
epochs="${EPOCHS:-1500}"
pseudo_spots="${PSEUDO_SPOTS:-30000}"
jobs="${JOBS:-1}"
datasets="${DATASETS:-}"
venv_dir="${VENV_DIR:-$repo_dir/.venv_stdnet_paper}"

# Bootstrap the environment and exercise the full data path once.  The quick
# result is intentionally separate from all formal timestamped run folders.
CUDA_BACKEND="$cuda_backend" \
RUN_MODE=quick PROFILE=full_v2 SEEDS="${seeds%%,*}" JOBS="$jobs" \
bash server_validation/install_and_run.sh

python_exec="$venv_dir/bin/python"
if [[ ! -x "$python_exec" ]]; then
  echo "Expected Python environment at $python_exec" >&2
  exit 2
fi

common_arguments=(
  --seeds "$seeds"
  --epochs "$epochs"
  --pseudo-spots "$pseudo_spots"
  --jobs "$jobs"
)
if [[ -n "$datasets" ]]; then
  common_arguments+=(--datasets "$datasets")
fi

mkdir -p server_logs
case "$suite_mode" in
  primary)
    profiles=(full_v2)
    protocols=(standard slice animal)
    ;;
  ablation)
    profiles=(
      legacy pseudo_panel sparse_supervision reconstruction full_v2
      no_reconstruction no_alignment no_spatial no_sparse_head
      no_platform_match no_full_panel uniform_pseudo
    )
    protocols=(standard)
    ;;
  all)
    profiles=(
      legacy pseudo_panel sparse_supervision reconstruction full_v2
      no_reconstruction no_alignment no_spatial no_sparse_head
      no_platform_match no_full_panel uniform_pseudo
    )
    protocols=(standard slice animal)
    ;;
  *)
    echo "SUITE_MODE must be primary, ablation, or all" >&2
    exit 2
    ;;
esac

for protocol in "${protocols[@]}"; do
  for profile in "${profiles[@]}"; do
    log_file="server_logs/suite_${profile}_${protocol}_$(date +%Y%m%d_%H%M%S).log"
    echo "START profile=$profile protocol=$protocol log=$log_file"
    "$python_exec" server_validation/run_paper_aligned.py \
      "${common_arguments[@]}" \
      --profile "$profile" \
      --protocol "$protocol" \
      2>&1 | tee "$log_file"
  done
done

"$python_exec" server_validation/build_v2_comparison.py
