#!/usr/bin/env bash
set -euo pipefail
repo_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
python_bin="${PYTHON_BIN:-$(command -v python3 || command -v python)}"
cd "$repo_dir"
PYTHONPATH="$repo_dir${PYTHONPATH:+:$PYTHONPATH}" "$python_bin" -m unittest server_validation.test_paper_alignment -v
