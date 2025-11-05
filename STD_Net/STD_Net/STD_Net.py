import pickle
import sys

from .__init__ import *
import re

import importlib
import subprocess
from typing import Dict, Any


class SpatialDeconvWrapper:

    def __init__(self, method_name: str, params: Dict[str, Any]):
        self.method_name = method_name
        self.params = params.get(method_name, {})
        self._load_dependencies()

    def _load_dependencies(self):

        required_libs = {
            'SPOTlight': ['SPOTlight'],
            'Cell2location': ['cell2location'],
            'Tangram': ['tangram-sc'],
            'DestVI': ['scvi-tools']
        }

        try:
            if self.method_name in required_libs:
                for lib in required_libs[self.method_name]:
                    try:
                        importlib.import_module(lib)
                    except ImportError:

                        try:
                            subprocess.run(
                                [sys.executable, "-m", "pip", "install", lib],
                                check=True,
                                capture_output=True
                            )
                            importlib.import_module(lib)
                        except subprocess.CalledProcessError as e:
                            raise RuntimeError(
                                f"Error details: {e.stderr.decode()}"
                            ) from e
        except ImportError:
            print(f"Installing {self.method_name} dependencies...")
            subprocess.run(['pip', 'install', lib])

    def train_and_predict(self, sc_data, st_data, sc_labels):

        if not isinstance(sc_data, sc.AnnData) or not isinstance(st_data, sc.AnnData):
            raise ValueError("sc_data and st_data must be AnnData objects.")
        if 'cell_type' not in sc_data.obs:
            raise KeyError("sc_data must have 'cell_type' in obs.")

        method_map = {
            'SPOTlight': self._run_spotlight,
            'Cell2location': self._run_cell2location,
            'Tangram': self._run_tangram,
            'DestVI': self._run_destvi
        }

        return method_map[self.method_name](sc_data, st_data, sc_labels)

    def _run_spotlight(self, sc_data, st_data, sc_labels):

        from sklearn.decomposition import NMF
        import numpy as np
        from scipy.sparse import issparse
        import scanpy as sc
        import pandas as pd

        sc.pp.highly_variable_genes(sc_data, n_top_genes=3000, flavor='seurat_v3')
        selected_genes = sc_data.var_names[sc_data.var.highly_variable].tolist()

        common_genes = sc_data.var_names.intersection(st_data.var_names)
        common_genes = common_genes.intersection(selected_genes)
        sc_data = sc_data[:, common_genes].copy()
        st_data = st_data[:, common_genes].copy()

        first_cell = next(iter(sc_data.obs.index))
        filtered_sc = sc_data[sc_data.obs.index == first_cell].copy()
        cell_type_counts = sc_data.obs['cell_type'].value_counts()
        valid_cell_types = cell_type_counts[cell_type_counts >= 2].index.tolist()

        if not valid_cell_types:
            raise ValueError("There are not enough cells of the specific type for the calculation.")

        for ct in valid_cell_types:
            ct_cells = sc_data[sc_data.obs['cell_type'] == ct]
            if ct_cells.shape[0] > 100:
                selected_indices = np.random.choice(ct_cells.shape[0], 100, replace=False)
                ct_cells = ct_cells[selected_indices]

            if ct != sc_data[first_cell].obs['cell_type'].values[0]:
                filtered_sc = filtered_sc.concatenate(ct_cells, batch_key="temp")

        if 'temp' in filtered_sc.obs:
            del filtered_sc.obs['temp']

        sc_data = filtered_sc

        if len(valid_cell_types) >= 2:
            try:
                sc.tl.rank_genes_groups(sc_data, 'cell_type', method='wilcoxon')
                marker_genes = {}
                for ct in valid_cell_types:
                    try:
                        ct_markers = sc.get.rank_genes_groups_df(sc_data, group=ct)

                        n_markers = min(50, len(ct_markers))
                        marker_genes[ct] = ct_markers.names[:n_markers].tolist()
                    except Exception as e:
                        print(f"Warning: Unable to obtain the marker gene of {ct}:{str(e)}")
                        marker_genes[ct] = []
            except Exception as e:
                print(f"Warning: Unable to perform differential expression analysis: {str(e)}")
                marker_genes = {ct: [] for ct in valid_cell_types}
        else:
            print("Warning: Insufficient cell types, unable to perform differential expression analysis")
            marker_genes = {ct: [] for ct in valid_cell_types}

        H = np.zeros((len(valid_cell_types), len(common_genes)))

        if not any(marker_genes.values()):

            for i, ct in enumerate(valid_cell_types):
                ct_mask = sc_data.obs['cell_type'] == ct
                ct_data = sc_data[ct_mask].X
                if issparse(ct_data):
                    H[i] = ct_data.mean(axis=0).A1
                else:
                    H[i] = np.mean(ct_data, axis=0)
        else:
            for i, ct in enumerate(valid_cell_types):
                ct_mask = sc_data.obs['cell_type'] == ct
                ct_data = sc_data[ct_mask].X

                for gene in marker_genes[ct]:
                    if gene in common_genes:
                        gene_idx = list(common_genes).index(gene)

                        if issparse(ct_data):
                            H[i, gene_idx] = ct_data[:, gene_idx].mean()
                        else:
                            H[i, gene_idx] = np.mean(ct_data[:, gene_idx])

        def safe_matrix(adata):
            X = adata.X
            if issparse(X):
                return X.toarray()
            return X

        sc_counts = safe_matrix(sc_data)
        st_counts = safe_matrix(st_data)

        from scipy.optimize import nnls

        W = np.zeros((st_counts.shape[0], H.shape[0]))

        for i in range(st_counts.shape[0]):
            spot_data = st_counts[i] + 1e-6
            W[i], _ = nnls(H.T, spot_data)

        proportions = W / (W.sum(axis=1, keepdims=True) + 1e-8)
        return pd.DataFrame(proportions,
                            index=st_data.obs.index,
                            columns=valid_cell_types)

    def _run_cell2location(self, sc_data, st_data, sc_labels):

        from cell2location.models import Cell2location, RegressionModel
        import pandas as pd
        import numpy as np
        import scanpy as sc
        from scipy.sparse import issparse

        def ensure_integer_counts(adata):
            if issparse(adata.X):
                if not np.issubdtype(adata.X.data.dtype, np.integer):
                    adata.X.data = np.rint(adata.X.data).astype(np.int32)
            else:
                if not np.issubdtype(adata.X.dtype, np.integer):
                    adata.X = np.rint(adata.X).astype(np.int32)
            return adata

        sc_data.var_names = sc_data.var_names.str.lower().str.replace('-', '_')
        st_data.var_names = st_data.var_names.str.lower().str.replace('-', '_')

        common_genes = sc_data.var_names.intersection(st_data.var_names)
        sc_data = sc_data[:, common_genes].copy()
        st_data = st_data[:, common_genes].copy()

        sc_data = ensure_integer_counts(sc_data)
        st_data = ensure_integer_counts(st_data)

        sc_data.obs['cell_type'] = sc_data.obs['cell_type'].astype('category')
        cell_types = sc_data.obs['cell_type'].cat.categories.tolist()
        n_cell_types = len(cell_types)

        RegressionModel.setup_anndata(
            adata=sc_data,
            labels_key='cell_type',
            batch_key='sample' if 'sample' in sc_data.obs.columns else None
        )

        reg_mod = RegressionModel(sc_data)

        def validate_obs_count(adata):
            if adata.n_obs == 0:
                raise ValueError(f"The dataset contains 0 observation samples. Please check the input data.")
            return adata.n_obs

        n_obs_sc = validate_obs_count(sc_data)
        n_obs_st = validate_obs_count(st_data)

        safe_batch_size = min(256, n_obs_sc)

        reg_mod.train(
            max_epochs=100,
            batch_size=safe_batch_size,
            lr=0.1
        )

        safe_batch_size_st = min(256, n_obs_st)

        try:

            if hasattr(reg_mod.module, 'w_means'):
                cell_state_df = reg_mod.module.w_means

            else:

                sc_data = reg_mod.export_posterior(sc_data)
                if 'means_per_cluster_mu_fg' in sc_data.varm:
                    cell_state_df = sc_data.varm['means_per_cluster_mu_fg']
                elif 'post_sample_means' in sc_data.varm:
                    cell_state_df = sc_data.varm['post_sample_means']
                else:

                    raise AttributeError("Unable to find an effective cell state matrix")
        except Exception as e:

            cell_state_df = pd.DataFrame(
                np.random.dirichlet(
                    np.ones(n_cell_types),
                    size=len(sc_data.var.index)
                ),
                index=sc_data.var.index,
                columns=cell_types
            )

        Cell2location.setup_anndata(adata=st_data)

        model_params = {
            'adata': st_data,
            'cell_state_df': cell_state_df,
            'N_cells_per_location': 8
        }

        try:

            model_params.update({
                'cell_number_prior': {
                    'factors_per_spot': 3,
                    'combs_per_spot': 1
                }
            })
        except:

            model_params.update({
                'max_cell_types_per_spot': 3
            })

        try:
            model = Cell2location(**model_params)
        except TypeError as e:

            model = Cell2location(
                st_data,
                cell_state_df=cell_state_df,
                N_cells_per_location=8
            )

        model.train(
            max_epochs=200,
            batch_size=safe_batch_size_st,
            lr=0.01
        )

        try:
            st_data = model.export_posterior(st_data)

            if 'q05_cell_abundance_w_sf' in st_data.obsm:
                proportions = st_data.obsm['q05_cell_abundance_w_sf']
            elif 'cell_abundance' in st_data.obsm:
                proportions = st_data.obsm['cell_abundance']
            elif 'means_per_cell_type' in st_data.obsm:
                proportions = st_data.obsm['means_per_cell_type']
            else:

                proportions = model.module.cell_abundance
        except Exception as e:
            print(f"Failed to export spatial results: {e}")

            proportions = model.module.cell_abundance

        if isinstance(proportions, pd.DataFrame):
            proportions = proportions.values
        elif not isinstance(proportions, np.ndarray):
            proportions = np.array(proportions)

        row_sums = proportions.sum(axis=1)

        row_sums[row_sums == 0] = 1e-8

        proportions = proportions / row_sums.reshape(-1, 1)

        return pd.DataFrame(
            proportions,
            index=st_data.obs.index,
            columns=cell_state_df.columns
        )

    def _run_destvi(self, sc_data, st_data, sc_labels):

        from scvi.model import SCVI
        import pandas as pd
        import numpy as np
        import scipy
        from scipy.optimize import nnls
        from scipy.special import digamma

        params = self.params.get('DestVI', {})
        sc_epochs = params.get('sc_epochs', 20)
        st_epochs = params.get('st_epochs', 4000)
        n_latent = params.get('n_latent', 4)
        n_layers = params.get('n_layers', 2)
        n_hidden = params.get('n_hidden', 128)
        lr = params.get('lr', 0.001)

        sc_data.var_names = sc_data.var_names.str.lower()
        st_data.var_names = st_data.var_names.str.lower()
        common_genes = sc_data.var_names.intersection(st_data.var_names)

        common_genes = sorted(common_genes)

        sc_data = sc_data[:, common_genes].copy()
        st_data = st_data[:, common_genes].copy()

        sc_data.layers["counts"] = sc_data.X.copy()
        st_data.layers["counts"] = st_data.X.copy()

        sc_data.obs["cell_type"] = sc_labels.astype("category")
        sc_data.obs["cell_type"] = sc_data.obs["cell_type"].cat.remove_unused_categories()
        cell_type_mapping = sc_data.obs["cell_type"].cat.categories.tolist()
        n_cell_types = len(cell_type_mapping)
        n_genes = len(common_genes)



        SCVI.setup_anndata(
            sc_data,
            layer="counts",
            labels_key="cell_type"
        )

        safe_batch_size = min(64, sc_data.shape[0])
        if safe_batch_size < 2:
            if sc_data.shape[0] < 2:
                raise ValueError("At least 2 training samples are required.")
            else:
                safe_batch_size = 2

        scvi_model = SCVI(
            sc_data,
            n_latent=n_latent,
            n_layers=n_layers,
            n_hidden=n_hidden,
            gene_likelihood="nb",
            use_batch_norm="none"
        )

        scvi_model.train(
            max_epochs=sc_epochs,
            batch_size=safe_batch_size,
            plan_kwargs={'lr': lr},
            early_stopping=False,
            check_val_every_n_epoch=None
        )

        if scipy.sparse.issparse(sc_data.X):
            sc_matrix = sc_data.X.toarray()
        else:
            sc_matrix = sc_data.X

        cell_type_means = []
        cell_type_dispersions = []

        for cell_type in cell_type_mapping:
            idx = sc_data.obs["cell_type"] == cell_type

            if np.sum(idx) > 1:

                mean_expr = np.mean(sc_matrix[idx], axis=0)

                variances = np.var(sc_matrix[idx], axis=0)
                dispersions = np.maximum((variances - mean_expr) / (mean_expr ** 2 + 1e-8), 1e-4)
            else:

                mean_expr = np.mean(sc_matrix, axis=0)
                variances = np.var(sc_matrix, axis=0)
                dispersions = np.maximum((variances - mean_expr) / (mean_expr ** 2 + 1e-8), 1e-4)
                print(f"Warning: Insufficient cells of type '{cell_type}'. Use global statistics instead.")

            cell_type_means.append(mean_expr)
            cell_type_dispersions.append(dispersions)

        profile_matrix = np.array(cell_type_means).T
        dispersion_matrix = np.array(cell_type_dispersions).T



        if scipy.sparse.issparse(st_data.X):
            st_matrix = st_data.X.toarray()
        else:
            st_matrix = st_data.X

        proportions = np.ones((st_matrix.shape[0], n_cell_types)) / n_cell_types

        for iteration in range(st_epochs):

            E = np.zeros_like(proportions)

            for spot_idx in range(st_matrix.shape[0]):
                spot_data = st_matrix[spot_idx, :]
                prop = proportions[spot_idx, :]

                contributions = profile_matrix * prop[None, :]

                total_expr = np.sum(contributions, axis=1) + 1e-8

                posterior = np.zeros((n_cell_types, n_genes))
                for k in range(n_cell_types):
                    mean_k = profile_matrix[:, k]
                    disp_k = dispersion_matrix[:, k]

                    ratio = mean_k / (mean_k + disp_k + 1e-8)
                    log_prob = spot_data * np.log(ratio) + disp_k * np.log(1 - ratio)

                    log_prob += np.log(prop[k] + 1e-8)
                    posterior[k, :] = log_prob

                max_log = np.max(posterior, axis=0, keepdims=True)
                posterior = np.exp(posterior - max_log)
                posterior /= np.sum(posterior, axis=0, keepdims=True)

                E[spot_idx, :] = np.sum(posterior, axis=1)

            proportions = E / np.sum(E, axis=1, keepdims=True)

            if iteration % 100 == 0:
                delta = np.mean(np.abs(proportions - E / np.sum(E, axis=1, keepdims=True)))
                print(f"Iteration {iteration}/{st_epochs}, Delta={delta:.6f}")
                if delta < 1e-5:
                    print("The EM algorithm has converged.")
                    break

        proportions_df = pd.DataFrame(
            proportions,
            index=st_data.obs.index,
            columns=cell_type_mapping
        )

        return proportions_df

    def _run_tangram(self, sc_data, st_data, sc_labels):

        import tangram as tg
        import scanpy as sc
        import torch
        import pandas as pd
        import numpy as np

        common_genes = sc_data.var_names.intersection(st_data.var_names)
        if len(common_genes) == 0:
            raise ValueError("Single-cell and spatial data do not share common genes.")

        sc_data = sc_data[:, common_genes].copy()
        st_data = st_data[:, common_genes].copy()

        def ensure_category(adata, column):
            if not pd.api.types.is_categorical_dtype(adata.obs[column]):
                adata.obs[column] = adata.obs[column].fillna('Unknown')

                adata.obs[column] = adata.obs[column].astype('category')


        ensure_category(sc_data, 'cell_type')

        tg.pp_adatas(sc_data, st_data, genes=common_genes.tolist())

        device = 'cuda' if torch.cuda.is_available() else 'cpu'

        training_params = {
            'num_epochs': 1000,
            'learning_rate': 0.01,
            'density_prior': 'rna_count_based',
            'random_state': 42,
            'device': device,
            'mode': 'clusters',
            'cluster_label': 'cell_type'
        }

        try:
            ad_map = tg.map_cells_to_space(
                adata_sc=sc_data,
                adata_sp=st_data,
                **training_params
            )
        except Exception as e:
            raise RuntimeError(f"Tangram failed: {str(e)}") from e

        tg.project_cell_annotations(ad_map, st_data)
        if 'tangram_ct_pred' not in st_data.obsm:
            raise KeyError("The Tangram projection result was not found. Please check the model output.")

        proportions = st_data.obsm['tangram_ct_pred']

        cell_types = sc_data.obs['cell_type'].cat.categories.tolist()

        if proportions.shape[1] != len(cell_types):
            print(
                f"Warning: The number of prediction dimensions ({proportions.shape[1]}) does not match the number of cell types ({len(cell_types)}).")

            base_matrix = np.zeros((proportions.shape[0], len(cell_types)))

            for i, ct in enumerate(cell_types):
                if i < proportions.shape[1]:
                    base_matrix[:, i] = proportions[:, i]
                else:
                    print(f"Type of missing data filling: {ct}")

            proportions = base_matrix

        proportions_np = proportions.values
        proportions = proportions_np / proportions_np.sum(axis=1, keepdims=True)
        proportions = pd.DataFrame(proportions, index=st_data.obs.index, columns=cell_types)

        return pd.DataFrame(
            proportions,
            index=st_data.obs.index,
            columns=cell_types
        )


def run_STD_Net(paths,
                deconv_method,
                method_params,
                find_marker_genes_paras,
                pseudo_spot_simulation_paras,
                data_normalization_paras,
                integration_for_adj_paras,
                inter_exp_adj_paras,
                spatial_adj_paras,
                real_intra_exp_adj_paras,
                pseudo_intra_exp_adj_paras,
                integration_for_feature_paras,
                DGAT_paras,
                load_test_groundtruth=False,
                use_marker_genes=True,
                external_genes=False,
                generate_new_pseudo_spots=True,
                fraction_pie_plot=True,
                cell_type_distribution_plot=True,
                n_jobs=-1,
                DGAT_device='GPU'
                ):
    dataset_name = paths['dataset_name']
    slice_name = paths['slice_name']

    # Experimental Plan 2:
    output_dir = f"{paths['output_path']}{dataset_name}/"
    output_path = f"{paths['output_path']}{dataset_name}/{slice_name}/"
    os.makedirs(output_path, exist_ok=True)

    sc_path = paths['sc_path']
    ST_path = paths['ST_path']
    # output_path = paths['output_path']

    sc_adata = sc.read_csv(sc_path + "/sc_data.tsv", delimiter='\t')

    sc_label = pd.read_table(sc_path + "/sc_label.tsv", sep='\t', header=0, index_col=0, encoding="utf-8")

    # sc_adata = sc_adata[:500, :]

    # sc_label = sc_label.iloc[:500, :]
    sc_label.columns = ['cell_type']
    sc_adata.obs['cell_type'] = sc_label['cell_type'].values
    cell_type_num = len(sc_adata.obs['cell_type'].unique())
    cell_types = sc_adata.obs['cell_type'].unique()

    word_to_idx_celltype = {word: i for i, word in enumerate(cell_types)}
    idx_to_word_celltype = {i: word for i, word in enumerate(cell_types)}

    celltype_idx = [word_to_idx_celltype[w] for w in sc_adata.obs['cell_type']]
    sc_adata.obs['cell_type_idx'] = celltype_idx

    sc_adata.obs['cell_type'].value_counts()

    ST_adata = sc.read_csv(ST_path + "/ST_data.tsv", delimiter='\t')
    ST_coor = pd.read_table(ST_path + "/coordinates.csv", sep=',', header=0, index_col=0, encoding="utf-8")

    # ST_adata = ST_adata[:500, :]
    # ST_coor = ST_coor.iloc[:500, :]
    # ST_adata = sc.read_csv(
    #     ST_path + "/ST_data.tsv",
    #     delimiter='\t',
    #     index_col=0
    # )
    ST_adata.obs['coor_X'] = ST_coor['x']
    ST_adata.obs['coor_Y'] = ST_coor['y']

    if load_test_groundtruth == True:
        ST_groundtruth = pd.read_table(ST_path + "/ST_ground_truth.tsv", sep='\t', header=0, index_col=0,
                                       encoding="utf-8")
        # ST_groundtruth = ST_groundtruth.iloc[:500, :]
        for i in cell_types:
            ST_adata.obs[i] = ST_groundtruth[i]

    # 1、
    if use_marker_genes == True:
        if external_genes == True:

            with open(sc_path + "/marker_genes.tsv", 'r') as f:
                selected_genes = [line.rstrip('\n') for line in f]
        else:
            selected_genes, cell_type_marker_genes = find_marker_genes(sc_adata,
                                                                       preprocess=find_marker_genes_paras['preprocess'],
                                                                       highly_variable_genes=find_marker_genes_paras[
                                                                           'highly_variable_genes'],
                                                                       PCA_components=find_marker_genes_paras[
                                                                           'PCA_components'],
                                                                       filter_wilcoxon_marker_genes=
                                                                       find_marker_genes_paras[
                                                                           'filter_wilcoxon_marker_genes'],
                                                                       marker_gene_method=find_marker_genes_paras[
                                                                           'marker_gene_method'],
                                                                       pvals_adj_threshold=find_marker_genes_paras[
                                                                           'pvals_adj_threshold'],
                                                                       log_fold_change_threshold=
                                                                       find_marker_genes_paras[
                                                                           'log_fold_change_threshold'],
                                                                       min_within_group_fraction_threshold=
                                                                       find_marker_genes_paras[
                                                                           'min_within_group_fraction_threshold'],
                                                                       max_between_group_fraction_threshold=
                                                                       find_marker_genes_paras[
                                                                           'max_between_group_fraction_threshold'],
                                                                       top_gene_per_type=find_marker_genes_paras[
                                                                           'top_gene_per_type'])
            print("marker gene", len(selected_genes))
            print("Cell type marker genes", len(cell_type_marker_genes))

            with open(output_path + "/marker_genes.tsv", 'w') as f:
                for gene in selected_genes:
                    f.write(str(gene) + '\n')

    print("{} genes have been selected as marker genes.".format(len(selected_genes)))

    # 2
    if generate_new_pseudo_spots == True:

        pseudo_adata = pseudo_spot_generation(
            sc_exp=sc_adata,
            idx_to_word_celltype=idx_to_word_celltype,
            spot_num=pseudo_spot_simulation_paras['spot_num'],
            min_cell_number_in_spot=pseudo_spot_simulation_paras['min_cell_num_in_spot'],
            max_cell_number_in_spot=pseudo_spot_simulation_paras['max_cell_num_in_spot'],
            max_cell_types_in_spot=pseudo_spot_simulation_paras['max_cell_types_in_spot'],
            generation_method=pseudo_spot_simulation_paras['generation_method'],
            n_jobs=n_jobs
        )

        with open(output_path + '/pseudo_ST.pkl', 'wb') as data_file:
            pickle.dump(pseudo_adata, data_file)
            data_file.close()
    else:
        data_file = open(output_path + '/pseudo_ST.pkl', 'rb')
        pseudo_adata = pickle.load(data_file)
        data_file.close()

    # 3、
    ST_adata = sc.read_csv(ST_path + "/ST_data.tsv", delimiter='\t')
    ST_coor = pd.read_table(ST_path + "/coordinates.csv", sep=',', header=0, index_col=0, encoding="utf-8")

    # ST_adata = ST_adata[:500, :]

    # ST_coor = ST_coor.iloc[:500, :]

    ST_adata.obs['coor_X'] = ST_coor['x']
    ST_adata.obs['coor_Y'] = ST_coor['y']

    if load_test_groundtruth == True:
        ST_groundtruth = pd.read_table(ST_path + "/ST_ground_truth.tsv", sep='\t', header=0, index_col=0,
                                       encoding="utf-8")
        # ST_groundtruth = ST_groundtruth.iloc[:500, :]
        for i in cell_types:
            ST_adata.obs[i] = ST_groundtruth[i]

    ST_genes = ST_adata.var.index.values
    pseudo_genes = pseudo_adata.var.index.values

    # 4、

    common_genes = set(ST_genes).intersection(set(pseudo_genes))


    ST_adata_filter = ST_adata[:, list(common_genes)]

    pseudo_adata_filter = pseudo_adata[:, list(common_genes)]


    ST_adata_filter_norm = ST_preprocess(ST_adata_filter,
                                         normalize=data_normalization_paras['normalize'],
                                         log=data_normalization_paras['log'],
                                         scale=data_normalization_paras['scale'],
                                         )[:, selected_genes]

    try:
        try:
            ST_adata_filter_norm.obs.insert(0, 'cell_num', ST_adata_filter.obs['cell_num'])
        except:
            ST_adata_filter_norm.obs['cell_num'] = ST_adata_filter.obs['cell_num']

    except:
        ST_adata_filter_norm.obs.insert(0, 'cell_num', [0] * ST_adata_filter_norm.obs.shape[0])

    for i in cell_types:
        try:

            ST_adata_filter_norm.obs[i] = ST_adata_filter.obs[i]
        except:
            ST_adata_filter_norm.obs[i] = [0] * ST_adata_filter_norm.obs.shape[0]
    try:
        ST_adata_filter_norm.obs['cell_type_num'] = (ST_adata_filter_norm.obs[cell_types] > 0).sum(axis=1)
    except:
        ST_adata_filter_norm.obs['cell_type_num'] = [0] * ST_adata_filter_norm.obs.shape[0]

    pseudo_adata_norm = ST_preprocess(pseudo_adata_filter,
                                      normalize=data_normalization_paras['normalize'],
                                      log=data_normalization_paras['log'],
                                      scale=data_normalization_paras['scale'],
                                      )[:, selected_genes]

    pseudo_adata_norm.obs['cell_type_num'] = (pseudo_adata_norm.obs[cell_types] > 0).sum(axis=1)

    # 5、
    ST_integration = data_integration(ST_adata_filter_norm,
                                      pseudo_adata_norm,
                                      batch_removal_method=integration_for_adj_paras['batch_removal_method'],
                                      dim=min(integration_for_adj_paras['dim'], int(ST_adata_filter_norm.shape[1] / 2)),
                                      dimensionality_reduction_method=integration_for_adj_paras[
                                          'dimensionality_reduction_method'],
                                      scale=integration_for_adj_paras['scale'],
                                      cpu_num=n_jobs,
                                      AE_device=DGAT_device
                                      )

    # 6、

    # real-pseudo expression
    A_inter_exp = inter_adj(ST_integration,
                            find_neighbor_method=inter_exp_adj_paras['find_neighbor_method'],
                            dist_method=inter_exp_adj_paras['dist_method'],
                            corr_dist_neighbors=inter_exp_adj_paras['corr_dist_neighbors'],
                            )

    spatial_clusters = spatial_clustering(ST_adata_filter_norm, n_clusters=7)

    spatial_coords = ST_adata_filter_norm.obs[['coor_X', 'coor_Y']].values
    boundary_flags = detect_boundary_points(spatial_coords, spatial_clusters)

    A_intra_space = intra_dist_adj_with_boundary_decay(
        ST_adata_filter_norm,
        spatial_clusters,
        boundary_flags,
        link_method=spatial_adj_paras['link_method'],
        space_dist_neighbors=spatial_adj_paras.get('space_dist_neighbors', 27),
        space_dist_threshold=spatial_adj_paras['space_dist_threshold'],
        boundary_decay=0.3
    )

    A_real_intra_exp = intra_exp_adj_with_clusters(ST_adata_filter_norm,
                                                   spatial_clusters,
                                                   find_neighbor_method=real_intra_exp_adj_paras[
                                                       'find_neighbor_method'],
                                                   dist_method=real_intra_exp_adj_paras['dist_method'],
                                                   PCA_dimensionality_reduction=real_intra_exp_adj_paras[
                                                       'PCA_dimensionality_reduction'],
                                                   corr_dist_neighbors=real_intra_exp_adj_paras['corr_dist_neighbors'])

    A_pseudo_intra_exp = intra_exp_adj(pseudo_adata_norm,
                                       find_neighbor_method=pseudo_intra_exp_adj_paras['find_neighbor_method'],
                                       dist_method=pseudo_intra_exp_adj_paras['dist_method'],
                                       PCA_dimensionality_reduction=pseudo_intra_exp_adj_paras[
                                           'PCA_dimensionality_reduction'],
                                       corr_dist_neighbors=pseudo_intra_exp_adj_paras['corr_dist_neighbors'])

    # 7、

    real_num = ST_adata_filter.shape[0]
    pseudo_num = pseudo_adata.shape[0]

    adj_inter_exp = A_inter_exp.values
    adj_pseudo_intra_exp = A_intra_transfer(A_pseudo_intra_exp, 'pseudo', real_num, pseudo_num)
    adj_real_intra_exp = A_intra_transfer(A_real_intra_exp, 'real', real_num, pseudo_num)
    adj_intra_space = A_intra_transfer(A_intra_space, 'real', real_num, pseudo_num)

    adj_alpha = 1
    adj_beta = 1
    diag_power = 20
    adj_balance = (1 + adj_alpha + adj_beta) * diag_power

    adj_exp = torch.tensor(
        adj_inter_exp + adj_alpha * adj_pseudo_intra_exp + adj_beta * adj_real_intra_exp) / adj_balance + torch.eye(
        adj_inter_exp.shape[0])

    adj_sp = torch.tensor(adj_intra_space) / diag_power + torch.eye(adj_intra_space.shape[0])

    norm = True
    if (norm == True):
        adj_exp = torch.tensor(adj_normalize(adj_exp, symmetry=True))
        adj_sp = torch.tensor(adj_normalize(adj_sp, symmetry=True))

    # 8、
    ST_integration_batch_removed = data_integration(ST_adata_filter_norm,
                                                    pseudo_adata_norm,
                                                    batch_removal_method=integration_for_feature_paras[
                                                        'batch_removal_method'],
                                                    dim=min(int(ST_adata_filter_norm.shape[1] * 1 / 2),
                                                            integration_for_feature_paras['dim']),
                                                    dimensionality_reduction_method=integration_for_feature_paras[
                                                        'dimensionality_reduction_method'],
                                                    scale=integration_for_feature_paras['scale'],
                                                    cpu_num=n_jobs,
                                                    AE_device=DGAT_device
                                                    )

    feature = torch.tensor(ST_integration_batch_removed.iloc[:, 3:].values)

    # 9、
    input_layer = feature.shape[1]
    hidden_layer = min(int(ST_adata_filter_norm.shape[1] * 1 / 2), DGAT_paras['dim'])

    if deconv_method == 'STD_Net':
        output_layer1 = len(word_to_idx_celltype)
        epoch_n = DGAT_paras['epoch_n']
        common_hid_layers_num = DGAT_paras['common_hid_layers_num']
        fcnn_hid_layers_num = DGAT_paras['fcnn_hid_layers_num']
        dropout = DGAT_paras['dropout']
        learning_rate_SGD = DGAT_paras['learning_rate_SGD']
        weight_decay_SGD = DGAT_paras['weight_decay_SGD']
        momentum = DGAT_paras['momentum']
        dampening = DGAT_paras['dampening']
        nesterov = DGAT_paras['nesterov']
        early_stopping_patience = DGAT_paras['early_stopping_patience']
        clip_grad_max_norm = DGAT_paras['clip_grad_max_norm']
        LambdaLR_scheduler_coefficient = 0.997
        ReduceLROnPlateau_factor = 0.1
        ReduceLROnPlateau_patience = 5

        scheduler = 'scheduler_CosineAnn'
        print_epoch_step = DGAT_paras['print_loss_epoch_step']
        cpu_num = n_jobs

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Using device: {device}")

        model = conDGAT(nfeat=input_layer,
                        nhid=hidden_layer,
                        common_hid_layers_num=common_hid_layers_num,
                        fcnn_hid_layers_num=fcnn_hid_layers_num,
                        dropout=dropout,
                        nout1=output_layer1,
                        use_gatv2=True,
                        gat_heads=4)
        model.to(device)


        optimizer = torch.optim.SGD(model.parameters(),
                                    lr=learning_rate_SGD,
                                    momentum=momentum,
                                    weight_decay=weight_decay_SGD,
                                    dampening=dampening,
                                    nesterov=nesterov)

        scheduler_LambdaLR = torch.optim.lr_scheduler.LambdaLR(optimizer,
                                                               lr_lambda=lambda
                                                                   epoch: LambdaLR_scheduler_coefficient ** epoch)
        scheduler_ReduceLROnPlateau = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer,
                                                                                 mode='min',
                                                                                 factor=ReduceLROnPlateau_factor,
                                                                                 patience=ReduceLROnPlateau_patience,
                                                                                 threshold=0.0001,
                                                                                 threshold_mode='rel',
                                                                                 cooldown=0,
                                                                                 min_lr=0)

        if scheduler == 'scheduler_LambdaLR':
            scheduler = scheduler_LambdaLR
        elif scheduler == 'scheduler_ReduceLROnPlateau':
            scheduler = scheduler_ReduceLROnPlateau
        elif scheduler == 'scheduler_CosineAnn':

            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                optimizer,
                T_max=epoch_n,
                eta_min=0.0001
            )
        else:
            scheduler = None
    elif deconv_method in ['SPOTlight', 'Tangram', 'Cell2location', 'DestVI']:
        deconv_wrapper = SpatialDeconvWrapper(
            method_name=deconv_method,
            params=method_params
        )

        if not isinstance(ST_adata_filter_norm, sc.AnnData):
            ST_adata_filter_norm = sc.AnnData(X=ST_adata_filter_norm.values,
                                              obs=ST_adata_filter_norm.obs)

        proportions = deconv_wrapper.train_and_predict(
            sc_data=sc_adata,
            st_data=ST_adata_filter_norm,
            sc_labels=sc_adata.obs['cell_type']
        )

        deconv_cell_types = proportions.columns.tolist()

        missing_types = set(cell_types) - set(deconv_cell_types)

        if missing_types:
            print(f"Warning: The deconvolution result is missing the following cell types: {missing_types}")
            for cell_type in missing_types:
                proportions[cell_type] = 0.0

        ordered_proportions = proportions[cell_types]

        raw_proportions = ordered_proportions.values

        row_sums = raw_proportions.sum(axis=1)
        row_sums[row_sums == 0] = 1
        normalized_proportions = raw_proportions / row_sums[:, np.newaxis]

        output1 = torch.tensor(normalized_proportions)

        predict_table = pd.DataFrame(normalized_proportions,
                                     index=ST_adata_filter_norm.obs.index,
                                     columns=cell_types)

        ST_adata_filter_norm.obsm['predict_result'] = normalized_proportions

        predict_table.to_csv(output_path + '/predict_result.csv', index=True, header=True)

    else:
        raise ValueError(f"Unsupported algorithm type: {deconv_method}")

    loss_fn1 = nn.KLDivLoss(reduction='batchmean')

    train_valid_len = pseudo_adata.shape[0]
    test_len = ST_adata_filter.shape[0]

    table1 = ST_adata_filter_norm.obs.copy()
    label1 = pd.concat([table1[pseudo_adata.obs.iloc[:, :-1].columns], pseudo_adata.obs.iloc[:, :-1]], axis=0)
    label1 = torch.tensor(label1.values)

    adjs = [adj_exp.float(), adj_sp.float()]

    # 9（2）

    if deconv_method == 'STD_Net':

        output1, loss, trained_model = conDGAT_train(model=model,
                                                     train_valid_len=train_valid_len,
                                                     train_valid_ratio=0.9,
                                                     test_len=test_len,
                                                     feature=feature,
                                                     adjs=adjs,
                                                     label=label1,
                                                     epoch_n=epoch_n,
                                                     loss_fn=loss_fn1,
                                                     optimizer=optimizer,
                                                     scheduler=scheduler,
                                                     early_stopping_patience=early_stopping_patience,
                                                     clip_grad_max_norm=clip_grad_max_norm,
                                                     load_test_groundtruth=load_test_groundtruth,
                                                     print_epoch_step=print_epoch_step,
                                                     cpu_num=cpu_num,
                                                     DGAT_device=DGAT_device
                                                     )

        loss_table = pd.DataFrame(loss, columns=['train', 'valid', 'test'])

        # 9 (3）、

        fig, ax = plt.subplots(figsize=(7, 7))
        ax.plot(loss_table.index, loss_table['train'], label='train')
        ax.plot(loss_table.index, loss_table['valid'], label='valid')
        if load_test_groundtruth == True:
            ax.plot(loss_table.index, loss_table['test'], label='test')
        ax.set_xlabel('Epoch', fontsize=20)
        ax.set_ylabel('Loss', fontsize=20)
        ax.set_title('Loss function curve', fontsize=20)
        ax.legend(fontsize=15)
        plt.tight_layout()

        plt.savefig(output_path + '/Loss_function.jpg', dpi=300)
        plt.close('all')

        predict_table = pd.DataFrame(np.exp(output1[:test_len].cpu().detach().numpy()).tolist(),
                                     index=ST_adata_filter_norm.obs.index, columns=pseudo_adata_norm.obs.columns[:-2])
        predict_table.to_csv(output_path + '/predict_result.csv', index=True, header=True)

        torch.save(trained_model, output_path + '/model_parameters.pth')

        pred_use = np.round_(output1.exp().cpu().detach()[:test_len], decimals=4)
        cell_type_list = cell_types
        coordinates = ST_adata_filter_norm.obs[['coor_X', 'coor_Y']]

        ST_adata_filter_norm.obsm['predict_result'] = np.exp(output1[:test_len].cpu().detach().numpy())

        # 10、
        if load_test_groundtruth:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            trained_model = trained_model.to(device)
            feature = feature.to(device)
            adjs = [adj.to(device) for adj in adjs]

            ground_truth = pd.read_csv(ST_path + "/ST_ground_truth.tsv", sep='\t', index_col=0)
            # ground_truth = ground_truth.iloc[:500, :]
            predict_table_path = os.path.join(output_path, "predict_result.csv")
            predictions = predict_table

            coordinates = ST_adata_filter_norm.obs[['coor_X', 'coor_Y']]

        # Experimental Plan 1

        # rmse, spearman, jsd, pred_scs, true_scs = calculate_metrics(
        #    predictions,
        #    ground_truth,
        #    coordinates=coordinates,
        #    cell_types=cell_types,
        #    dataset_name=dataset_name,
        #    slice_name=slice_name,
        #    output_dir=output_path
        # )

        # print(f"""
        #    Metrics Report:
        #    - RMSE: {rmse:.4f}
        #    - Spearman Correlation: {spearman:.4f}
        #    - JSD: {jsd:.4f}
        #    - Predicted SCS: {pred_scs:.4f}
        #    - True SCS: {true_scs:.4f}
        #    """)

        # plot_metrics(rmse, spearman, jsd, pred_scs, true_scs,
        #            dataset_name, slice_name, output_dir=output_path)

        #
        # common_cols = list(set(predictions.columns).intersection(set(ground_truth.columns)))

        # coordinates = ST_adata_filter_norm.obs[['coor_X', 'coor_Y']]
        # plot_umap_comparison(
        #     ground_truth[common_cols],
        #     predictions[common_cols],
        #      coordinates,
        #      dataset_name,
        #      slice_name,
        #      output_dir=output_path
        # )
        #


        # # Experimental Plan 2
        #        cell_num = ground_truth['cell_num'].values

        #        groups = [
        #            ('cell_leq5', cell_num <= 5),
        #            ('cell_gt5', cell_num > 5)]
        #
        #        for group_name, mask in groups:
        #
        #            group_output_dir = os.path.join(output_dir,group_name)
        #            os.makedirs(group_output_dir, exist_ok=True)
        #
        #
        #            group_pred = predictions.loc[mask]
        #            group_gt = ground_truth.loc[mask]
        #            group_coord = coordinates.loc[mask]
        #     #

        #            if len(group_pred) == 0:
        #                print(f"Warning: {group_name} group is empty")
        #               continue
        #     #

        #            rmse, spearman, jsd, pred_scs, true_scs = calculate_metrics(
        #                group_pred,
        #                group_gt,
        #                coordinates=group_coord,
        #                cell_types=cell_types,
        #                dataset_name=dataset_name,
        #                slice_name=slice_name,
        #                group_name=group_name,
        #                output_dir=group_output_dir
        #            )
        #     #
        #
        #            print(f"""
        #                    {group_name} Group Metrics:
        #                    - RMSE: {rmse:.4f}
        #                    - Spearman Correlation: {spearman:.4f}
        #                    - JSD: {jsd:.4f}
        #                    - Predicted SCS: {pred_scs:.4f}
        #                    - True SCS: {true_scs:.4f}
        #                    """)
        #

        #            plot_metrics(rmse, spearman, jsd, pred_scs, true_scs,
        #                     dataset_name, slice_name, group_name=group_name,output_dir=group_output_dir)
        #
        #     common_cols = list(set(predictions.columns).intersection(set(ground_truth.columns)))
        #
        #     coordinates = ST_adata_filter_norm.obs[['coor_X', 'coor_Y']]
        #     plot_umap_comparison(
        #         group_gt[common_cols],
        #         group_pred[common_cols],
        #         group_coord,
        #         dataset_name,
        #         slice_name,
        #         group_name=group_name,
        #         output_dir=group_output_dir
        #     )
        #
        #     if DGAT_paras.get('enable_explain', False):
        #
        #         original_test_indices = np.arange(train_valid_len, train_valid_len + test_len)
        #

        #         selected_test_indices = original_test_indices[mask]
        #         if len(selected_test_indices) == 0:
        #             print(f"Skip explanation for {group_name} (no test samples)")
        #             continue
        #

        #         selected_indices = np.concatenate([
        #             np.arange(train_valid_len),
        #             selected_test_indices
        #         ])
        #

        #         group_feature = feature[selected_indices]
        #         group_adj_exp = adj_exp[selected_indices][:, selected_indices]
        #         group_adj_sp = adj_sp[selected_indices][:, selected_indices]
        #

        #         group_train_indices = list(range(train_valid_len))
        #         group_test_indices = list(range(train_valid_len, len(selected_indices)))
        #


        # Experimental Plan 3
        #       ground_truth = pd.read_csv(ST_path + "/ST_ground_truth.tsv", sep='\t', index_col=0)
        #       columns_to_remove = ['cell_num', 'coor_X', 'coor_Y']
        #
        #       existing_columns = [col for col in columns_to_remove if col in ground_truth.columns]
        #       if existing_columns:
        #           ground_truth = ground_truth.drop(columns=existing_columns)
        #
        #       else:
        #           print("No target column was found. No deletion is necessary.")
        #       cell_type_counts = (ground_truth > 0).sum(axis=1).values
        #
        #
        #
        #       groups = [
        #           ('cell_type_1', cell_type_counts == 1),  # Singleton
        #           ('cell_type_2', cell_type_counts == 2),  # Doubleton
        #           ('cell_type_3', cell_type_counts == 3),  # Tripleton
        #           ('cell_type_4+', cell_type_counts >= 4)  # Multiton
        #       ]

        #    #
        #       for group_name, mask in groups:
        #
        #           group_output_dir = os.path.join(output_dir, group_name)
        #           os.makedirs(group_output_dir, exist_ok=True)
        #

        #           group_pred = predictions.loc[mask]
        #           group_gt = ground_truth.loc[mask]
        #            group_coord = coordinates.loc[mask]
        #
        #
        #            if len(group_pred) == 0:
        #                print(f"Warning: {group_name} group is empty")
        #                continue
        #            else:
        #                print(f" {group_name} group is {len(group_pred)}")

        #            if len(group_pred) < 2:
        #                print(f"Skip {group_name}: only {len(group_pred)} samples")
        #                continue
        #            print(f"{group_name} group length:{len(group_pred)}")
        #

        #            rmse, spearman, jsd, pred_scs, true_scs = calculate_metrics(
        #                group_pred,
        #                group_gt,
        #                coordinates=group_coord,
        #                cell_types=cell_types,
        #                dataset_name=dataset_name,
        #                slice_name=slice_name,
        #                group_name = group_name,
        #                output_dir=group_output_dir
        #            )

        #            print(f"""
        #                    {group_name} Group Metrics:
        #                    - RMSE: {rmse:.4f}
        #                    - Spearman Correlation: {spearman:.4f}
        #                   - JSD: {jsd:.4f}
        #                    - Predicted SCS: {pred_scs:.4f}
        #                    - True SCS: {true_scs:.4f}
        #                    """)

        #           plot_metrics(rmse, spearman, jsd, pred_scs, true_scs,
        #                          dataset_name, slice_name,                                    #group_name=group_name,output_dir=group_output_dir)
        # #
        #         common_cols = list(set(predictions.columns).intersection(set(ground_truth.columns)))
        #
        #         coordinates = ST_adata_filter_norm.obs[['coor_X', 'coor_Y']]
        #         plot_umap_comparison(
        #             group_gt[common_cols],
        #             group_pred[common_cols],
        #             group_coord,
        #             dataset_name,
        #             slice_name,
        #             group_name=group_name,
        #             output_dir=group_output_dir
        #         )
        #
        #         if DGAT_paras.get('enable_explain', False):
        #
        #             original_test_indices = np.arange(train_valid_len, train_valid_len + test_len)
        #
        #
        #             selected_test_indices = original_test_indices[mask]
        #             if len(selected_test_indices) == 0:
        #                 print(f"Skip explanation for {group_name}")
        #                 continue
        #
        #
        #             selected_indices = np.concatenate([
        #                 np.arange(train_valid_len),
        #                 selected_test_indices
        #             ])
        #
        #
        #             group_feature = feature[selected_indices]
        #             group_adj_exp = adj_exp[selected_indices][:, selected_indices]
        #             group_adj_sp = adj_sp[selected_indices][:, selected_indices]
        #
        #
        #             group_train_indices = list(range(train_valid_len))
        #             group_test_indices = list(range(train_valid_len, len(selected_indices)))
        #


        # # Experimental Plan 4

        ground_truth = pd.read_csv(ST_path + "/ST_ground_truth.tsv", sep='\t', index_col=0)
        columns_to_remove = ['cell_num', 'coor_X', 'coor_Y']

        existing_columns = [col for col in columns_to_remove if col in ground_truth.columns]
        if existing_columns:
            ground_truth = ground_truth.drop(columns=existing_columns)
            print(f"Deleted columns: {existing_columns}")
        else:
            print("The target column was not found. No deletion is required.")
        cell_types = ground_truth.columns.tolist()


        for cell_type in cell_types:

            type_output_dir = os.path.join(output_path, f"type_{cell_type}")
            os.makedirs(type_output_dir, exist_ok=True)


            type_mask = (ground_truth[cell_type] > 0)
            if type_mask.sum() < 3:
                print(f"Skipping rare type: {cell_type} ({type_mask.sum()} spots)")
            #              continue


            type_pred = predictions.loc[type_mask]
            type_gt = ground_truth.loc[type_mask]
            type_coord = coordinates.loc[type_mask]


            type_rmse, type_spearman, type_jsd, type_scs = calculate_type_metrics(
                cell_type=cell_type,
                predictions=type_pred,
                ground_truth=type_gt,
                target_type=cell_type,
                coordinates=type_coord,
                output_dir=type_output_dir,
                dataset_name=dataset_name,
                slice_name=slice_name
            )

        #
        #     common_cols = list(set(predictions.columns).intersection(set(ground_truth.columns)))
        #
        #     coordinates = ST_adata_filter_norm.obs[['coor_X', 'coor_Y']]
        #     plot_umap_comparison(
        #         type_gt[common_cols],
        #         type_pred[common_cols],
        #         type_coord,
        #         dataset_name,
        #         slice_name,
        #         group_name=cell_type,
        #         output_dir=type_output_dir
        #     )
        #
        #     if DGAT_paras.get('enable_explain', False):
        #
        #         original_test_indices = np.arange(train_valid_len, train_valid_len + test_len)
        #
        #
        #         selected_test_indices = original_test_indices[type_mask]
        #         if len(selected_test_indices) == 0:
        #             print(f"Skip explanation for {cell_type}")
        #             continue
        #
        #
        #         selected_indices = np.concatenate([
        #             np.arange(train_valid_len),
        #             selected_test_indices
        #         ])
        #
        #
        #         type_feature = feature[selected_indices]
        #         type_adj_exp = adj_exp[selected_indices][:, selected_indices]
        #         type_adj_sp = adj_sp[selected_indices][:, selected_indices]
        #
        #
        #         type_train_indices = list(range(train_valid_len))
        #         type_test_indices = list(range(train_valid_len, len(selected_indices)))
        #
        #



    #     stats_files = glob("results/*/stats_*.json")
    #     all_results = []
    #
    #     for f in stats_files:
    #         with open(f, 'r') as fp:
    #             data = json.load(fp)
    #             data['file'] = f
    #             all_results.append(data)
    #
    #     # 转换为DataFrame分析
    #     df = pd.DataFrame(all_results)
    #     df_exploded = df.explode('p_values').reset_index()
    #
    #
    #     significant_improvements = df_exploded[
    #         (df_exploded['corrected_p'] < 0.05) &
    #         (df_exploded['significance'] == True)
    #         ]
    #

    #     print(significant_improvements[['dataset', 'slice', 'metric', 'corrected_p']])

    else:



        pred_use = normalized_proportions

        # pred_use = np.round_(output1.exp().cpu().detach()[:test_len], decimals=4)

        cell_type_list = cell_types
        coordinates = ST_adata_filter_norm.obs[['coor_X', 'coor_Y']]


        # if fraction_pie_plot == True:
        #     plot_frac_results(paths, pred_use, cell_type_list, coordinates, point_size=300, size_coefficient=0.0009, file_name=output_path+'/predict_results_pie_plot.jpg', if_show=False)
        #
        # if cell_type_distribution_plot == True:
        #     plot_scatter_by_type(paths, pred_use, cell_type_list, coordinates, point_size=300, file_path=output_path, if_show=False)


        ST_adata_filter_norm.obsm['predict_result'] = np.exp(output1[:test_len].cpu().detach().numpy())

        if load_test_groundtruth:

            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            feature = feature.to(device)
            adjs = [adj.to(device) for adj in adjs]

            ground_truth = pd.read_csv(ST_path + "/ST_ground_truth.tsv", sep='\t', index_col=0)
            # ground_truth = ground_truth.iloc[:500, :]
            predict_table_path = os.path.join(output_path, "predict_result.csv")
            predictions = pd.read_csv(
                predict_table_path,
                index_col=0
            )

            coordinates = ST_adata_filter_norm.obs[['coor_X', 'coor_Y']]







    torch.cuda.empty_cache()

    return ST_adata_filter_norm
