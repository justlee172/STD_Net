#!/usr/bin/env bash
set -euo pipefail

repo_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$repo_dir"
export PYTHONPATH="$repo_dir${PYTHONPATH:+:$PYTHONPATH}"

python_bin="${PYTHON_BIN:-python3}"
venv_dir="${VENV_DIR:-$repo_dir/.venv_stdnet_paper}"
cuda_backend="${CUDA_BACKEND:-cu121}"
run_mode="${RUN_MODE:-quick}"
methods="${METHODS:-nlsdeconv,card}"
protocols="${PROTOCOLS:-standard,slice,animal}"
datasets="${DATASETS:-}"
seeds="${SEEDS:-0}"
rscript_bin="${RSCRIPT_BIN:-Rscript}"
nls_commit="3bbe0a53a8b0f6798eacf7eab9f7298742fd27fd"
nls_dir="$repo_dir/baseline_validation/vendor/NLSDeconv_3bbe0a53"
card_commit="2d64b91abb5cdd0c7f576b1c5d4727c84e7c93a0"
card_library="$repo_dir/baseline_validation/R_library"

"$python_bin" -c 'import sys; assert (3, 9) <= sys.version_info[:2] <= (3, 11), "Use Python 3.9, 3.10, or 3.11"'
if [[ ! -x "$venv_dir/bin/python" ]]; then
  "$python_bin" -m venv "$venv_dir"
fi
python_exec="$venv_dir/bin/python"
pip_exec="$venv_dir/bin/pip"
"$pip_exec" install --upgrade pip setuptools wheel

if ! "$python_exec" -c 'import torch' >/dev/null 2>&1; then
  case "$cuda_backend" in
    cu121|cu118)
      "$pip_exec" install "torch==2.3.0" --index-url "https://download.pytorch.org/whl/$cuda_backend"
      ;;
    cpu)
      "$pip_exec" install "torch==2.3.0" --index-url "https://download.pytorch.org/whl/cpu"
      ;;
    *)
      echo "CUDA_BACKEND must be cu121, cu118, or cpu" >&2
      exit 2
      ;;
  esac
fi
"$pip_exec" install -r baseline_validation/requirements-baselines.txt

if [[ "$methods" == *"nlsdeconv"* ]]; then
  mkdir -p "$(dirname "$nls_dir")"
  if [[ ! -d "$nls_dir/.git" ]]; then
    git clone https://github.com/tinachentc/NLSDeconv.git "$nls_dir"
    git -C "$nls_dir" checkout "$nls_commit"
  fi
  actual_commit="$(git -C "$nls_dir" rev-parse HEAD)"
  if [[ "$actual_commit" != "$nls_commit" ]]; then
    echo "Unexpected NLSDeconv checkout: $actual_commit" >&2
    echo "Expected: $nls_commit" >&2
    exit 2
  fi
fi

if [[ "$methods" == *"card"* ]]; then
  if ! command -v "$rscript_bin" >/dev/null 2>&1; then
    echo "Rscript was not found. On Ubuntu, install R and CARD system libraries first:" >&2
    echo "sudo apt-get update" >&2
    echo "sudo apt-get install -y r-base r-base-dev libcurl4-openssl-dev libssl-dev libxml2-dev libfontconfig1-dev libharfbuzz-dev libfribidi-dev libfreetype6-dev libpng-dev libtiff-dev libjpeg-dev libudunits2-dev libgdal-dev libgeos-dev libproj-dev" >&2
    exit 2
  fi
  mkdir -p "$card_library"
  export R_LIBS_USER="$card_library"
  if [[ -d /usr/lib/x86_64-linux-gnu/pkgconfig ]]; then
    export PKG_CONFIG_LIBDIR="/usr/lib/x86_64-linux-gnu/pkgconfig:/usr/share/pkgconfig"
  fi
  "$rscript_bin" baseline_validation/install_card.R "$card_library" "$card_commit"
fi

"$python_exec" -m unittest baseline_validation.test_common -v
"$python_exec" baseline_validation/preflight.py --methods "$methods" --rscript "$rscript_bin"

arguments=(
  --methods "$methods"
  --protocols "$protocols"
  --seeds "$seeds"
  --rscript "$rscript_bin"
)
if [[ -n "$datasets" ]]; then
  arguments+=(--datasets "$datasets")
fi
if [[ "$run_mode" == "quick" ]]; then
  arguments+=(--quick)
elif [[ "$run_mode" != "formal" ]]; then
  echo "RUN_MODE must be quick or formal" >&2
  exit 2
fi

mkdir -p baseline_logs
log_file="baseline_logs/${run_mode}_$(date +%Y%m%d_%H%M%S).log"
"$python_exec" baseline_validation/run_new_baselines.py "${arguments[@]}" 2>&1 | tee "$log_file"
