# STD-Net: Spatially Topology-aware and Dynamically Fused Network for Cell-type Deconvolution in Spatial Transcriptomics

**Software version:** 0.0.1

**STD-Net (Spatially Topology-aware Deconvolution Network)** is a graph-based deep learning framework designed for high-resolution cell-type deconvolution of spatial transcriptomics data. The model integrates single-cell RNA sequencing data with spatial topology information to accurately resolve cellular heterogeneity within tissue microenvironments. By incorporating a spatially topology-enhanced data generation module and a dynamic graph attention network (DGAT), STD-Net achieves superior performance in identifying rare cell types and preserving spatial domain integrity. The implementation demonstrates robust capability across multiple platforms including seqFISH, seqFISH+, and MERFISH datasets.

![图片1](C:\Users\冷寻非\Desktop\图片1.png)

---

## Table of Contents

- [Software Name and Version](#software-name-and-version)
- [Installation](#installation)
- [Requirements and Dependencies](#requirements-and-dependencies)
- [How to Run STD-Net](#how-to-run-std-net)
- [Input Data Format](#input-data-format)
- [Output Files Description](#output-files-description)
- [Test Data Description](#test-data-description)
- [How to Reproduce Results from the Paper](#how-to-reproduce-results-from-the-paper)
- [License](#license)
- [Citation](#citation)
- [Contact Information](#contact-information)
- [Acknowledgements](#acknowledgements)

---

## Software Name and Version

- **Software name:** STD-Net
- **Version:** 0.0.1
- **Description:** Spatially Topology-aware and Dynamically Fused Graph Neural Network for cell-type deconvolution of spatial transcriptomics data.
- **Programming language:** Python 3
- **License:** MIT License (see [License](#license) section)

---

## Installation

### Prerequisites

STD-Net requires Python 3.8 or later. We recommend using a virtual environment (e.g., conda or venv) to manage dependencies.

### Step-by-step installation

```bash
# 1. Clone the repository
git clone https://github.com/justlee172/STD_Net.git
cd STD_Net

# 2. Create and activate a virtual environment (recommended)
conda create -n stdnet python=3.9
conda activate stdnet

# 3. Install PyTorch with CUDA support (GPU recommended)
# For CUDA 12.1:
pip install torch==2.3.0 torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

# 4. Install PyTorch Geometric dependencies
pip install torch-scatter==2.1.2+cu121 torch-sparse==0.6.18+cu121 torch-cluster==1.6.3+cu121 -f https://data.pyg.org/whl/torch-2.3.0+cu121.html

# 5. Install remaining dependencies
pip install numpy==1.26.4 pandas scipy matplotlib seaborn scanpy anndata scanorama tensorflow networkx shap lime umap-learn scikit-learn tqdm multiprocess
```

> **Note:** If you do not have a CUDA-compatible GPU, install the CPU-only versions of PyTorch and skip the PyTorch Geometric dependencies. However, GPU acceleration is strongly recommended for training the DGAT model on full datasets.

---

## Requirements and Dependencies

The following table lists all required Python packages with the tested versions:

| Package | Tested Version | Purpose |
|---------|---------------|---------|
| numpy | 1.26.4 | Numerical computing |
| pandas | 2.0.3 | Data manipulation and analysis |
| scipy | 1.6.1 | Scientific computing and sparse matrices |
| matplotlib | 3.7.1 | Plotting and visualization |
| seaborn | 0.13.2 | Statistical data visualization |
| scanpy | 1.9.3 | Single-cell analysis toolkit |
| anndata | 0.10.3 | Annotated data containers |
| scanorama | 1.7.3 | Batch correction and data integration |
| torch | 2.3.0 | Deep learning framework (with CUDA 12.1) |
| torch-geometric | 2.5.3 | Graph neural network library |
| torch-scatter | 2.1.2+cu121 | Scatter operations for PyG |
| torch-sparse | 0.6.18+cu121 | Sparse matrix operations for PyG |
| torch-cluster | 1.6.3+cu121 | Graph clustering for PyG |
| tensorflow | 2.13.0 | Optional backend (primarily uses PyTorch) |
| networkx | 3.1 | Network/graph analysis |
| shap | 0.44.0 | SHAP explainability (optional) |
| lime | 0.2.0.1 | LIME explainability (optional) |
| umap-learn | 0.5.5 | Dimensionality reduction for visualization |
| scikit-learn | 1.4.2 | Machine learning utilities and metrics |
| tqdm | 4.66.2 | Progress bars |
| multiprocess | 0.70.16 | Parallel processing |

### Hardware requirements

- **Minimum:** 16 GB RAM, 4 CPU cores
- **Recommended:** 32 GB RAM, 8+ CPU cores, NVIDIA GPU with 8+ GB VRAM (CUDA 12.1 compatible)
- **Storage:** At least 10 GB free disk space for datasets and output files

---

## How to Run STD-Net

STD-Net is designed for straightforward execution with a one-click operation mode.

### Quick start

Simply execute the following command to start the complete workflow, including data preprocessing, model training, evaluation, and visualization:

```bash
python Run_STD.py
```

If you are using an IDE like PyCharm or a code editor, you can directly open and run the `Run_STD.py` script from within your development environment.

### Command-line parameters

All parameters are configured within `Run_STD.py` through the dedicated parameter dictionaries. The key parameter groups are:

#### Path Configuration (`paths`)

```python
paths = {
    'sc_path': './data/sc_data',       # Path to single-cell RNA-seq data
    'ST_path': './data/ST_data',       # Path to spatial transcriptomics data
    'output_path': './results/',       # Output directory for results
    'dataset_name': "SEQFISH+",        # Dataset name
    'slice_name': "SVZ",               # Slice identifier
}
```

#### Pseudo-spot Simulation Parameters (`pseudo_spot_simulation_paras`)

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `spot_num` | int | 30000 | Number of pseudo-spots to generate |
| `min_cell_num_in_spot` | int | 8 | Minimum cells per pseudo-spot |
| `max_cell_num_in_spot` | int | 12 | Maximum cells per pseudo-spot |
| `generation_method` | str | 'celltype' | Method: 'cell' (equal cell selection) or 'celltype' (equal cell-type selection) |
| `max_cell_types_in_spot` | int | 4 | Maximum cell types per pseudo-spot (when `generation_method='celltype'`) |
| `hybrid_p` | float | 0.7 | Probability weight for multimodal fusion |
| `enable_low_abundance_boost` | bool | True | Enable oversampling of rare cell types |

#### Data Normalization Parameters (`data_normalization_paras`)

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `normalize` | bool | True | Normalize each spot to total counts = 10,000 |
| `log` | bool | True | Apply log(X+1) transformation |
| `scale` | bool | False | Scale genes to unit variance and zero mean |

#### DGAT Model Parameters (`DGAT_paras`)

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `use_gatv2` | bool | True | Use GATv2 attention (True) or standard GCN (False) |
| `gat_heads` | int | 4 | Number of attention heads in GATv2 |
| `epoch_n` | int | 3000 | Maximum training epochs |
| `dim` | int | 80 | Hidden layer dimension |
| `common_hid_layers_num` | int | 1 | Number of common hidden graph convolution layers |
| `fcnn_hid_layers_num` | int | 1 | Number of fully-connected hidden layers |
| `dropout` | float | 0.0 | Dropout rate |
| `learning_rate_SGD` | float | 0.2 | SGD learning rate |
| `weight_decay_SGD` | float | 3e-4 | Weight decay (L2 regularization) |
| `momentum` | float | 0.9 | SGD momentum |
| `early_stopping_patience` | int | 20 | Patience for early stopping |
| `clip_grad_max_norm` | float | 1.0 | Maximum gradient norm for clipping |
| `DGAT_device` | str | 'GPU' | Device: 'GPU' or 'CPU' |

### Running on multiple datasets

The default `Run_STD.py` script automatically processes all dataset directories under `final_data/`. Each subdirectory should contain the required data structure (see [Input Data Format](#input-data-format)).

---

## Input Data Format

STD-Net expects data organized in the following directory structure for each dataset slice:

```
final_data/
└── <dataset_slice_name>/
    ├── sc_data/
    │   ├── sc_data.tsv        # Single-cell expression matrix (genes x cells)
    │   └── sc_label.tsv       # Cell type annotations for each cell
    └── st_data/
        ├── ST_data.tsv        # Spatial transcriptomics expression matrix (genes x spots)
        ├── coordinates.csv    # Spatial coordinates (x, y) for each spot
        └── ST_ground_truth.tsv  # (Optional) Ground truth cell-type proportions
```

### File format specifications

#### `sc_data.tsv` (Single-cell expression matrix)
- **Format:** Tab-separated values (TSV)
- **Rows:** Cells (cell barcodes as index)
- **Columns:** Genes (gene symbols)
- **Values:** Raw or normalized gene expression counts

#### `sc_label.tsv` (Cell type annotations)
- **Format:** Tab-separated values (TSV)
- **Content:** Two columns: cell barcode (index) and `cell_type` (categorical label)
- **Example:**
  ```
  cell_id    cell_type
  AAACCTGAG  Excitatory_Neuron
  AAACCTGCA  Inhibitory_Neuron
  ...
  ```

#### `ST_data.tsv` (Spatial transcriptomics expression matrix)
- **Format:** Tab-separated values (TSV)
- **Rows:** Spatial spots (spot IDs as index)
- **Columns:** Genes (gene symbols)
- **Values:** Gene expression counts per spatial spot

#### `coordinates.csv` (Spatial coordinates)
- **Format:** Comma-separated values (CSV)
- **Required columns:** `x` and `y` (spatial coordinates in micrometers or pixels)
- **Index:** Spot IDs matching those in `ST_data.tsv`

#### `ST_ground_truth.tsv` (Ground truth proportions, optional)
- **Format:** Tab-separated values (TSV)
- **Rows:** Spatial spots (matching ST_data.tsv index)
- **Columns:** Cell types (matching those in sc_label.tsv)
- **Values:** Proportion of each cell type in each spot (rows sum to 1)

> **Important:** Gene names must be consistent across `sc_data.tsv` and `ST_data.tsv`. The intersection of genes between these two files determines the feature space used for training.

---

## Output Files Description

After training completes, results are saved under the `results/` directory, organized by dataset name. The following files are generated:

### Directory structure

```
results/
└── <dataset_name>/
    └── <slice_name>/
        ├── marker_genes.tsv
        ├── Loss_function.jpg
        ├── metrics_<slice_id>.txt
        ├── metrics_chart_<slice_id>.png
        ├── model_parameters.pth
        ├── predict_result.csv
        ├── umap_<slice_id>/
        ├── type_<cell_type>/       (per-cell-type analysis)
        │   ├── metrics_*.txt
        │   └── metrics_chart_*.png
        └── results.h5ad
```

### File descriptions

| File | Description |
|------|-------------|
| `marker_genes.tsv` | List of cell type-specific marker genes identified and used for model training. Contains one gene symbol per line. |
| `Loss_function.jpg` | Image (300 dpi) displaying the training, validation, and test loss curves over epochs. Used for monitoring model convergence and diagnosing overfitting. |
| `metrics_<slice_id>.txt` | Text file recording detailed quantitative evaluation results: Root Mean Square Error (RMSE), Jensen-Shannon Divergence (JSD), Spearman correlation coefficient, and Spatial Correlation Score (SCS) for the specified slice. |
| `metrics_chart_<slice_id>.png` | Bar chart visualization of performance metrics (RMSE, JSD, Spearman, SCS) for the specified slice, enabling intuitive comparison across metrics. |
| `model_parameters.pth` | PyTorch file containing the trained parameter weights of the STD-Net (conDGAT) model. Can be loaded for subsequent prediction, fine-tuning, or model inspection. |
| `predict_result.csv` | **Core output file.** Contains predicted cell-type proportions for each spatial spot in the transcriptomics data. Rows are spots, columns are cell types, and values are predicted proportions (summing to 1 per row). |
| `umap_<slice_id>/` | Directory containing UMAP dimensionality reduction visualization results. Includes side-by-side comparisons of predicted versus true cell-type spatial distributions, colored by dominant cell type. |
| `type_<cell_type>/` | Per-cell-type directories containing granular performance metrics (RMSE, Spearman, JSD, SCS) for each individual cell type, enabling evaluation of rare cell-type detection. |
| `results.h5ad` | AnnData HDF5 file containing the complete processed dataset with predictions stored in `obsm['predict_result']`, suitable for downstream analysis with Scanpy. |

---

## Test Data Description

To facilitate quick setup and debugging, we provide a subset of the seqFISH dataset within the `final_data` directory. Due to the large size of the complete original dataset, this example includes only three selected embryo slices for demonstration purposes.

The seqFISH dataset slice names follow the convention `embryo[number]_z[slice number]` (e.g., `embryo1_z2`) as described in Lohoff et al. (2021), where the number represents the embryonic individual and the z value indicates the position of the slice along the head-tail axis of the embryo.

### Provided test slices

| Slice ID | Description |
|----------|-------------|
| `embryo1_z2` | Embryo 1, slice position 2 along the head-tail axis. Contains multiple cell types with moderate spatial mixing. |
| `embryo1_z5` | Embryo 1, slice position 5. Deeper tissue section with distinct spatial domain organization. |
| `embryo2_z2` | Embryo 2, slice position 2. Cross-individual validation slice demonstrating generalizability across biological replicates. |

These three slices represent different spatial positions within the developing mouse embryo, capturing the spatial heterogeneity of cell-type distributions that STD-Net is designed to deconvolve. The data include ground truth cell-type annotations derived from the original seqFISH imaging-based measurements, enabling quantitative evaluation of deconvolution accuracy.

For access to the full seqFISH, seqFISH+, and MERFISH datasets and a detailed description of the complete data processing pipeline, please refer to our associated research paper and the original data publications (Lohoff et al., 2021; Eng et al., 2019; Chen et al., 2015).

---

## How to Reproduce Results from the Paper

This section provides step-by-step instructions to reproduce the key results reported in the paper.

### 1. Download the datasets

Download the full seqFISH, seqFISH+, and MERFISH datasets from their respective repositories:
- **seqFISH:** Available from the original publication (Lohoff et al., 2021)
- **seqFISH+:** Available from Eng et al. (2019)
- **MERFISH:** Available from Chen et al. (2015)

### 2. Prepare the data

Organize each dataset slice into the directory structure described in [Input Data Format](#input-data-format). The provided test data (`final_data/`) can be used as a reference for the expected format.

```bash
# Example directory structure after data preparation
final_data/
├── embryo1_z2/      # seqFISH slice
├── embryo1_z5/      # seqFISH slice
├── embryo2_z2/      # seqFISH slice
├── seqFISH+_slice1/ # seqFISH+ data
└── MERFISH_slice1/  # MERFISH data
```

### 3. Configure parameters

Edit `Run_STD.py` to set the appropriate parameters for your experiment. The default parameters are configured to reproduce the main benchmark results. See [How to Run STD-Net](#how-to-run-std-net) for detailed parameter descriptions.

### 4. Run STD-Net training and evaluation

```bash
python Run_STD.py
```

This will:
1. Load and preprocess the single-cell and spatial transcriptomics data
2. Identify marker genes for each cell type
3. Generate pseudo-spots via multimodal probabilistic fusion
4. Construct the spatial topology and expression adjacency matrices
5. Train the DGAT model with GATv2 attention
6. Predict cell-type proportions for each spatial spot
7. Generate evaluation metrics (RMSE, JSD, Spearman, SCS)
8. Produce visualizations (loss curves, UMAP plots, metric charts)
9. Save all outputs to the `results/` directory

### 5. Compare with benchmark methods

To run the comparison methods (SPOTlight, Cell2location, DestVI, Tangram), set the `deconv_method` parameter in `Run_STD.py`:

```python
spatial_deconv_params = {
    'method': 'SPOTlight',  # Options: 'STD_Net', 'SPOTlight', 'Cell2location', 'DestVI', 'Tangram'
    ...
}
```

Run each method separately and collect the metrics from the output files for comparison.

### 6. Reproduce ablation study

To reproduce the ablation study (Fig. 3), modify individual components of the STD-Net pipeline:

| Ablation | Code Modification |
|----------|------------------|
| Remove spatial topology | Set `spatial_adj_paras['link_method']` to `None` |
| Remove pseudo-spot generation | Set `generate_new_pseudo_spots = False` |
| Disable GATv2 (use standard GCN) | Set `DGAT_paras['use_gatv2'] = False` |
| Remove residual gating | Modify `conDGAT.forward()` to remove residual connections |
| Remove boundary decay | Set `boundary_decay = 1.0` in adjacency construction |
| Replace KL divergence | Change `loss_fn` to MSE in training loop |

### 7. Expected runtime

On a system with an NVIDIA GPU (8 GB+ VRAM), 32 GB RAM, and 8 CPU cores:
- Data preprocessing: ~5-10 minutes per slice
- Pseudo-spot generation: ~2-5 minutes
- DGAT model training (3000 epochs): ~15-30 minutes per slice
- Evaluation and visualization: ~5-10 minutes per slice

Total runtime for a full experiment with 3 slices: approximately 1-2 hours.

---

## License

STD-Net is released under the MIT License.

```
MIT License

Copyright (c) 2024 Xiaolin Li, Juehan Lu, Zunren Chen, Haihan Zeng,
                   Qinhua Zhang, Yilan Sun, Shiguo Huang

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
```

---

## Citation

If you use STD-Net in your research, please cite our paper:

```bibtex
@article{li2025stdnet,
  title={STD-Net: Spatially topology-aware and dynamically fused network
         for cell-type deconvolution in spatial transcriptomics},
  author={Li, Xiaolin and Lu, Juehan and Chen, Zunren and Zeng, Haihan
          and Zhang, Qinhua and Sun, Yilan and Huang, Shiguo},
  journal={Bioinformatics},
  year={2025},
  publisher={Oxford University Press},
  note={In press}
}
```

> **Note:** This citation is a placeholder. Please update with the final publication details once the paper is accepted.

### Data and software availability

The source code is archived at:
- **GitHub:** https://github.com/justlee172/STD_Net
- **Zenodo/Figshare:** [DOI: 10.5281/zenodo.20842487](https://doi.org/10.5281/zenodo.20842487)

---

## Contact Information

For questions, bug reports, or collaboration inquiries, please contact:

**Corresponding Authors:**

- **Shiguo Huang** (sghuang@fafu.edu.cn)
  - Fujian Agriculture and Forestry University
  - College of Computer and Information Sciences

- **Yilan Sun** (yilansun@126.com)
  - Fujian University of Traditional Chinese Medicine

**Lead Developer:**

- **Xiaolin Li**
  - Fujian Agriculture and Forestry University

Please use "STD-Net" in the subject line when contacting us via email.

---

## Acknowledgements

This work was supported by [funding information to be added]. We thank the developers of Scanpy, PyTorch Geometric, and the authors of SPOTlight, Cell2location, DestVI, Tangram, and STdGCN for making their code and methods publicly available. We also thank the original data generators of the seqFISH, seqFISH+, and MERFISH datasets.
