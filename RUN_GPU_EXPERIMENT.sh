#!/usr/bin/env bash
set -euo pipefail
repo_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
if [[ -z "${PYTHON_BIN:-}" ]]; then
    if [[ -x "$repo_dir/.venv_stdnet_paper/bin/python" ]]; then
        export PYTHON_BIN="$repo_dir/.venv_stdnet_paper/bin/python"
    else
        export PYTHON_BIN="$(command -v python3 || command -v python)"
    fi
fi
exec bash "$repo_dir/server_validation/run_rare_balanced_gpu_ablation.sh"
