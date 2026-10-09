# STD-Net

STD-Net is a support-aware dual-graph neural network for spatial transcriptomic cell-type deconvolution. This repository contains the executable source code for the current `support_rare_balanced` implementation, together with the paper-aligned training wrapper and tests.

## What is included

- `STD_Net/`: core model, sparse graph construction, pseudo-spot generation and optimization code.
- `Run_STD.py`: legacy-compatible entry point.
- `server_validation/`: paper-aligned training/evaluation wrappers and alignment tests.
- `baseline_validation/`: optional baseline evaluation utilities.
- Requirements files and shell wrappers used by the included tests and experiments.

The repository intentionally excludes raw expression data, coordinates, truth labels, predictions, checkpoints, logs, result tables, figures, cached environments and manuscript files. Obtain third-party datasets from their original repositories and follow their licenses.

## Installation

Use Python 3.11 and install the paper-aligned dependencies:

```bash
python -m pip install -r server_validation/requirements-paper-aligned.txt
```

Baseline utilities use:

```bash
python -m pip install -r baseline_validation/requirements-baselines.txt
```

GPU training requires a compatible CUDA-enabled PyTorch installation and `torch-geometric` wheels for that installation.

## Tests

Run the paper-alignment tests without any expression dataset:

```bash
python -m unittest discover -s server_validation -p 'test_*.py'
```

## Running with user-provided data

The wrappers expect reference expression, cell-type labels, aligned reference coordinates, target expression and target coordinates supplied by the user. Set the data paths and output directory in a local configuration or wrapper invocation. Do not commit raw data, predictions, checkpoints or logs to this repository.

The `support_rare_balanced` profile jointly uses inverse-abundance weighting and a bounded rare-type ranking term. The implementation and parameter defaults are defined in `server_validation/run_paper_aligned.py` and `STD_Net/optimization.py`.

## Reproducibility

Record the commit identifier, Python version, dependency versions, random seeds, CUDA/PyTorch build and input-data accession when reproducing an analysis. The repository contains code and tests only; numerical results are not part of this public code release.

## Archive and citation

The `v0.2.0` code release is archived on Zenodo with the version DOI [10.5281/zenodo.23266460](https://doi.org/10.5281/zenodo.23266460). Cite this version DOI when referring to the released implementation.

## License

This code release is distributed under the MIT License; see [LICENSE](LICENSE).
