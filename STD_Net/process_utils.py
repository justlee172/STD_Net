import os
import warnings

from matplotlib import pyplot as plt
from sklearn.neighbors import NearestNeighbors
from tqdm.notebook import tqdm
from sklearn.decomposition import NMF
import random
from sklearn.metrics.pairwise import cosine_similarity
from .autoencoder import *

from scipy.stats import spearmanr
from scipy.spatial.distance import jensenshannon
import numpy as np
import pandas as pd
import re


def _run_safe_pca(adata, requested_components):
    """Run ARPACK PCA without requesting an impossible component count."""
    min_dimension = min(adata.n_obs, adata.n_vars)
    max_components = min_dimension - 1
    if max_components < 1:
        raise ValueError(
            "PCA requires at least two observations and two features; "
            f"received shape {adata.shape}"
        )
    n_components = min(int(requested_components), max_components)
    if n_components < int(requested_components):
        warnings.warn(
            "Reducing PCA components from "
            f"{requested_components} to {n_components} for data shape "
            f"{adata.shape} because ARPACK requires n_components < "
            "min(n_samples, n_features).",
            RuntimeWarning,
            stacklevel=2,
        )
    sc.tl.pca(
        adata,
        n_comps=n_components,
        svd_solver='arpack',
        random_state=None,
    )



def dense_to_sparse(adj):

    edge_index = (adj > 0).nonzero(as_tuple=False).t()
    edge_attr = adj[adj > 0]
    return edge_index, edge_attr

def ST_preprocess(ST_exp, 
                  normalize=True,
                  log=True,
                  highly_variable_genes=False, 
                  regress_out=False, 
                  scale=False,
                  scale_max_value=None,
                  scale_zero_center=True,
                  hvg_min_mean=0.0125,
                  hvg_max_mean=3,
                  hvg_min_disp=0.5,
                  highly_variable_gene_num=None
                 ):
    
    adata = ST_exp.copy()


    if normalize == True:
        sc.pp.normalize_total(adata, target_sum=1e4)


    if log == True:
        sc.pp.log1p(adata)
        
    adata.layers['scale.data'] = adata.X.copy()

    if highly_variable_genes == True:
        sc.pp.highly_variable_genes(adata, 
                                    min_mean=hvg_min_mean, 
                                    max_mean=hvg_max_mean, 
                                    min_disp=hvg_min_disp,
                                    n_top_genes=highly_variable_gene_num,
                                   )
        adata = adata[:, adata.var.highly_variable]


    if regress_out == True:
        mito_genes = adata.var_names.str.startswith('MT-')
        adata.obs['percent_mito'] = np.sum(adata[:, mito_genes].X, axis=1) / np.sum(adata.X, axis=1)
        sc.pp.filter_cells(adata, min_counts=0)
        sc.pp.regress_out(adata, ['n_counts', 'percent_mito'])

    if scale == True:
        sc.pp.scale(adata, max_value=scale_max_value, zero_center=scale_zero_center)
    
    return adata


def find_marker_genes(sc_exp, 
                      preprocess = True,
                      highly_variable_genes = True,
                      regress_out = False,
                      scale = False,
                      PCA_components = 50, 
                      marker_gene_method = 'wilcoxon',
                      filter_wilcoxon_marker_genes = True, 
                      top_gene_per_type = 20, 
                      pvals_adj_threshold = 0.10,
                      log_fold_change_threshold = 1,
                      min_within_group_fraction_threshold = 0.7,
                      max_between_group_fraction_threshold = 0.3,
                     ):


    if preprocess == True:
        sc_adata_marker_gene = ST_preprocess(sc_exp.copy(), 
                                             normalize=True,
                                             log=True,
                                             highly_variable_genes=highly_variable_genes, 
                                             regress_out=regress_out, 
                                             scale=scale,
                                            )
    else:
        sc_adata_marker_gene = sc_exp.copy()



    _run_safe_pca(sc_adata_marker_gene, PCA_components)

    layer = 'scale.data'

    sc.tl.rank_genes_groups(sc_adata_marker_gene, 'cell_type', layer=layer, use_raw=False, pts=True, 
                            method=marker_gene_method, corr_method='benjamini-hochberg', key_added=marker_gene_method)


    if marker_gene_method == 'wilcoxon':

        if filter_wilcoxon_marker_genes == True:

            gene_dict = {}

            gene_list = []

            for name in sc_adata_marker_gene.obs['cell_type'].unique():

                data = sc.get.rank_genes_groups_df(sc_adata_marker_gene, group=name, key=marker_gene_method).sort_values('pvals_adj')

                if pvals_adj_threshold != None:
                    data = data[data['pvals_adj'] < pvals_adj_threshold]

                if log_fold_change_threshold != None:
                    data = data[data['logfoldchanges'] >= log_fold_change_threshold]

                if min_within_group_fraction_threshold != None:
                    data = data[data['pct_nz_group'] >= min_within_group_fraction_threshold]

                if max_between_group_fraction_threshold != None:
                    data = data[data['pct_nz_reference'] < max_between_group_fraction_threshold]

                gene_dict[name] = data['names'].values[:top_gene_per_type].tolist()
                gene_list = gene_list + data['names'].values[:top_gene_per_type].tolist()
                gene_list = list(set(gene_list))
        else:
            gene_table = pd.DataFrame(sc_adata_marker_gene.uns[marker_gene_method]['names'][:top_gene_per_type])
            gene_dict = {}
            for i in gene_table.columns:
                gene_dict[i] = gene_table[i].values.tolist()
            gene_list = list(set([item   for sublist in gene_table.values.tolist()   for item in sublist]))
    elif marker_gene_method == 'logreg':
        gene_table = pd.DataFrame(sc_adata_marker_gene.uns[marker_gene_method]['names'][:top_gene_per_type])
        gene_dict = {}
        for i in gene_table.columns:
            gene_dict[i] = gene_table[i].values.tolist()
        gene_list = list(set([item   for sublist in gene_table.values.tolist()   for item in sublist]))
    else:
        print("marker_gene_method should be 'logreg' or 'wilcoxon'")

    return gene_list, gene_dict


import numpy as np
import anndata

def identify_tissue_boundary(spatial_coords, bandwidth=0.1):


    from sklearn.neighbors import KernelDensity
    kde = KernelDensity(bandwidth=bandwidth).fit(spatial_coords)
    densities = np.exp(kde.score_samples(spatial_coords))


    threshold = np.percentile(densities, 20)
    boundary_mask = densities < threshold
    return boundary_mask


def spatial_constrained_oversampling(sc_exp,
                                     boundary_mask,
                                     low_abundance_types,
                                     oversample_ratio=3.0):

    boundary_cells = sc_exp.obs.index[boundary_mask]
    boundary_low_abundance = sc_exp.obs[
        (sc_exp.obs.index.isin(boundary_cells)) &
        (sc_exp.obs['cell_type'].isin(low_abundance_types))
        ].index.tolist()


    n_extra = int(len(boundary_low_abundance) * oversample_ratio)

    if n_extra == 0:
        return sc_exp


    synthetic_cells = []
    for _ in range(n_extra):

        idx1, idx2 = np.random.choice(boundary_low_abundance, 2, replace=False)
        cell1 = sc_exp[sc_exp.obs.index == idx1].X
        cell2 = sc_exp[sc_exp.obs.index == idx2].X


        alpha = np.random.uniform(0.3, 0.7)
        new_cell = alpha * cell1 + (1 - alpha) * cell2

        synthetic_cells.append(new_cell.flatten())


    synthetic_adata = anndata.AnnData(
        X=np.vstack(synthetic_cells),
        obs=pd.DataFrame({
            'cell_type': ['synthetic'] * n_extra,
            'cell_type_idx': [-1] * n_extra,
            'is_synthetic': [True] * n_extra
        }),
        var=sc_exp.var.copy()
    )


    combined_adata = anndata.concat([sc_exp, synthetic_adata])
    return combined_adata

def generate_a_spot(sc_exp,
                    min_cell_number_in_spot,
                    max_cell_number_in_spot,
                    max_cell_types_in_spot,
                    generation_method,
                    boundary_mask=None):


    if generation_method == 'cell':

        cell_num = random.randint(min_cell_number_in_spot, max_cell_number_in_spot)
        cell_list = list(sc_exp.obs.index.values)
        picked_cells = random.choices(cell_list, k=cell_num)
        return sc_exp[picked_cells]


    elif generation_method == 'celltype':

        cell_num = random.randint(min_cell_number_in_spot, max_cell_number_in_spot)


        if boundary_mask is not None:

            boundary_cells = sc_exp.obs.index[boundary_mask]


            min_boundary_cells = max(1, int(cell_num * 0.2))
            boundary_selected = np.random.choice(
                boundary_cells,
                size=min_boundary_cells,
                replace=True
            ).tolist()
        else:
            boundary_selected = []


        cell_type_counts = sc_exp.obs['cell_type'].value_counts()
        N_t = cell_type_counts.values.astype(float)
        T = len(N_t)

        P_unbalance = N_t / N_t.sum()
        P_balance = np.ones(T) / T
        P_sqrt = np.sqrt(N_t) / np.sqrt(N_t).sum()


        gamma = 0.2
        median_Nt = np.median(N_t)
        sigmoid_factor = 1 / (1 + np.exp(-gamma * (N_t - median_Nt)))
        P_curve = (1 / np.sqrt(N_t + 1e-6)) * sigmoid_factor
        P_curve /= P_curve.sum()


        prob_mode = random.choices(
            ['unbalance', 'balance', 'sqrt', 'curve'],
            weights=[0.25, 0.3, 0.3, 0.15]
        )[0]
        probs = {
            'unbalance': P_unbalance,
            'balance': P_balance,
            'sqrt': P_sqrt,
            'curve': P_curve
        }[prob_mode]


        probs = (probs + 1e-6) / (1 + 1e-6 * T)
        probs = probs / probs.sum()


        if boundary_mask is not None:

            boundary_type_counts = sc_exp[boundary_mask].obs['cell_type'].value_counts()
            boundary_probs = boundary_type_counts / boundary_type_counts.sum()


            combined_probs = 0.7 * probs + 0.3 * boundary_probs
            combined_probs = combined_probs / combined_probs.sum()
        else:
            combined_probs = probs


        cell_type_num = random.randint(1, min(max_cell_types_in_spot, len(cell_type_counts.index)))
        all_types = cell_type_counts.index.tolist()


        selected_types = []
        remaining_types = all_types.copy()

        while len(selected_types) < cell_type_num and remaining_types:

            remaining_probs = combined_probs[cell_type_counts.index.isin(remaining_types)]
            remaining_probs = remaining_probs / remaining_probs.sum()

            chosen_type = np.random.choice(remaining_types, p=remaining_probs)
            selected_types.append(chosen_type)
            remaining_types.remove(chosen_type)


        cell_pools = {}
        for ct in selected_types:

            cell_pools[ct] = sc_exp[sc_exp.obs['cell_type'] == ct].obs.index.tolist()


        picked_cells = boundary_selected.copy()


        remaining_cells = cell_num - len(picked_cells)


        type_weights = [combined_probs[cell_type_counts.index == t][0] for t in selected_types]
        type_weights = np.array(type_weights) / sum(type_weights)

        for _ in range(remaining_cells):

            chosen_type = random.choices(selected_types, weights=type_weights, k=1)[0]


            if cell_pools[chosen_type]:
                chosen_cell = random.choice(cell_pools[chosen_type])
                picked_cells.append(chosen_cell)


                cell_pools[chosen_type].remove(chosen_cell)

        return sc_exp[picked_cells]

    else:
        raise ValueError('generation_method is "cell" or "celltype"')



def pseudo_spot_generation(sc_exp,
                           idx_to_word_celltype,
                           spot_num,
                           min_cell_number_in_spot,
                           max_cell_number_in_spot,
                           max_cell_types_in_spot,
                           generation_method,
                           spatial_coords=None,
                           n_jobs=-1):

    if spatial_coords is not None:

        boundary_mask = identify_tissue_boundary(spatial_coords)


        type_counts = sc_exp.obs['cell_type'].value_counts()
        low_abundance_threshold = 0.05
        low_abundance_types = type_counts[type_counts / type_counts.sum() < low_abundance_threshold].index.tolist()


        sc_exp = spatial_constrained_oversampling(
            sc_exp,
            boundary_mask,
            low_abundance_types
        )


    cell_type_num = len(sc_exp.obs['cell_type'].unique())

    cores = multiprocessing.cpu_count()
    if n_jobs == -1:
        pool = multiprocessing.Pool(processes=cores)
    else:
        pool = multiprocessing.Pool(processes=n_jobs)


    args = [(sc_exp, min_cell_number_in_spot, max_cell_number_in_spot,
             max_cell_types_in_spot, generation_method, boundary_mask if spatial_coords else None)
            for i in range(spot_num)]

    generated_spots = pool.starmap(generate_a_spot, tqdm(args, desc='Generating pseudo-spots'))
    pseudo_spots = []
    pseudo_spots_table = np.zeros((spot_num, sc_exp.shape[1]), dtype=float)

    pseudo_fraction_table = np.zeros((spot_num, cell_type_num), dtype=float)
    for i in range(spot_num):
        one_spot = generated_spots[i]
        pseudo_spots.append(one_spot)
        pseudo_spots_table[i] = one_spot.X.sum(axis=0)
        for j in one_spot.obs.index:
            type_idx = one_spot.obs.loc[j, 'cell_type_idx']
            pseudo_fraction_table[i, type_idx] += 1

    pseudo_spots_table = pd.DataFrame(pseudo_spots_table, columns=sc_exp.var.index.values)
    pseudo_spots = anndata.AnnData(X=pseudo_spots_table.iloc[:,:].values)
    pseudo_spots.obs.index = pseudo_spots_table.index[:]
    pseudo_spots.var.index = pseudo_spots_table.columns[:]
    type_list = [idx_to_word_celltype[i] for i in range(cell_type_num)]
    pseudo_fraction_table = pd.DataFrame(pseudo_fraction_table, columns=type_list)

    pseudo_fraction_table['cell_num'] = pseudo_fraction_table.sum(axis=1)
    for i in pseudo_fraction_table.columns[:-1]:
        pseudo_fraction_table[i] = pseudo_fraction_table[i]/pseudo_fraction_table['cell_num']

    pseudo_spots.obs = pseudo_spots.obs.join(pseudo_fraction_table)


    return pseudo_spots




def data_integration(real, 
                     pseudo, 
                     batch_removal_method="combat",
                     dimensionality_reduction_method='PCA', 
                     dim=50, 
                     scale=True,
                     autoencoder_epoches=2000,
                     autoencoder_LR=1e-3,
                     autoencoder_drop=0,
                     cpu_num=-1,
                     AE_device='GPU'
                    ):


    if batch_removal_method == 'mnn':
        mnn = sc.external.pp.mnn_correct(pseudo, real, svd_dim=dim, k=50, batch_key='real_pseudo', save_raw=True, var_subset=None)

        adata = mnn[0]

        if dimensionality_reduction_method == 'PCA':

            if scale == True:
                sc.pp.scale(adata, max_value=None, zero_center=True)
            _run_safe_pca(adata, dim)

            table = pd.DataFrame(adata.obsm['X_pca'], index=[str(i)[:-2] for i in adata.obs.index])
        elif dimensionality_reduction_method == 'autoencoder':
            data = torch.tensor(adata.X)
            x_size = data.shape[1]
            latent_size = dim
            hidden_size = int((x_size + latent_size)/2)
            nets = autoencoder(x_size=x_size, hidden_size=hidden_size, embedding_size=latent_size, p_drop=autoencoder_drop)
            optimizer_ae = torch.optim.Adam(nets.parameters(), lr=autoencoder_LR)
            loss_ae = nn.MSELoss(reduction = 'mean')
            embedding = auto_train(model=nets, 
                                   epoch_n=autoencoder_epoches, 
                                   loss_fn=loss_ae, 
                                   optimizer=optimizer_ae, 
                                   data=data,
                                   cpu_num=cpu_num,
                                   device=AE_device
                                  ).detach().numpy()
            if scale == True:
                embedding = (embedding-embedding.mean(axis=0))/embedding.std(axis=0)
            table = pd.DataFrame(embedding, index=[str(i)[:-2] for i in adata.obs.index])
        elif dimensionality_reduction_method == 'nmf':
            nmf = NMF(n_components=dim).fit_transform(adata.X)
            if scale == True:
                nmf = (nmf-nmf.mean(axis=0))/nmf.std(axis=0)
            table = pd.DataFrame(nmf, index=[str(i)[:-2] for i in adata.obs.index])
        elif dimensionality_reduction_method == None:
            if scale == True:
                sc.pp.scale(adata, max_value=None, zero_center=True)
            table = pd.DataFrame(adata.X, index=[str(i)[:-2] for i in adata.obs.index], columns=adata.var.index.values)

        table = table.iloc[pseudo.shape[0]:,:].append(table.iloc[:pseudo.shape[0],:])

        table.insert(0, 'ST_type', ['real']*real.shape[0]+['pseudo']*pseudo.shape[0])
        
    elif batch_removal_method == 'scanorama':
        import scanorama
        scanorama.integrate_scanpy([real, pseudo], dimred = dim)
        table1 = pd.DataFrame(real.obsm['X_scanorama'], index=real.obs.index.values)
        table2 = pd.DataFrame(pseudo.obsm['X_scanorama'], index=pseudo.obs.index.values)
        table = table1.append(table2)
        table.insert(0, 'ST_type', ['real']*real.shape[0]+['pseudo']*pseudo.shape[0])
        
    elif batch_removal_method == 'combat':
        aaa = real.copy()
        aaa.obs = pd.DataFrame(index = aaa.obs.index)
        bbb = pseudo.copy()
        bbb.obs = pd.DataFrame(index = bbb.obs.index)
        adata = aaa.concatenate(bbb, batch_key='real_pseudo')
        sc.pp.combat(adata, key='real_pseudo')
        if dimensionality_reduction_method == 'PCA':
            if scale == True:
                sc.pp.scale(adata, max_value=None, zero_center=True)
            _run_safe_pca(adata, dim)
            table = pd.DataFrame(adata.obsm['X_pca'], index=[str(i)[:-2] for i in adata.obs.index])
        elif dimensionality_reduction_method == 'autoencoder':
            data = torch.tensor(adata.X)
            x_size = data.shape[1]
            latent_size = dim
            hidden_size = int((x_size + latent_size)/2)
            nets = autoencoder(x_size=x_size, hidden_size=hidden_size, embedding_size=latent_size, p_drop=autoencoder_drop)
            optimizer_ae = torch.optim.Adam(nets.parameters(), lr=autoencoder_LR)
            loss_ae = nn.MSELoss(reduction = 'mean')
            embedding = auto_train(model=nets, 
                                   epoch_n=autoencoder_epoches, 
                                   loss_fn=loss_ae, 
                                   optimizer=optimizer_ae, 
                                   data=data,
                                   cpu_num=cpu_num,
                                   device=AE_device
                                  ).detach().numpy()
            if scale == True:
                embedding = (embedding-embedding.mean(axis=0))/embedding.std(axis=0)
            table = pd.DataFrame(embedding, index=[str(i)[:-2] for i in adata.obs.index])
        elif dimensionality_reduction_method == 'nmf':
            nmf = NMF(n_components=dim).fit_transform(adata.X)
            if scale == True:
                nmf = (nmf-nmf.mean(axis=0))/nmf.std(axis=0)
            table = pd.DataFrame(nmf, index=[str(i)[:-2] for i in adata.obs.index])
        elif dimensionality_reduction_method == None:
            if scale == True:
                sc.pp.scale(adata, max_value=None, zero_center=True)
            table = pd.DataFrame(adata.X, index=[str(i)[:-2] for i in adata.obs.index], columns=adata.var.index.values)
        table.insert(0, 'ST_type', ['real']*real.shape[0]+['pseudo']*pseudo.shape[0])
        
    else:
        aaa = real.copy()
        aaa.obs = pd.DataFrame(index = aaa.obs.index)
        bbb = pseudo.copy()
        bbb.obs = pd.DataFrame(index = bbb.obs.index)
        adata = aaa.concatenate(bbb, batch_key='real_pseudo')
        if dimensionality_reduction_method == 'PCA':
            if scale == True:
                sc.pp.scale(adata, max_value=None, zero_center=True)
            _run_safe_pca(adata, dim)
            table = pd.DataFrame(adata.obsm['X_pca'], index=[str(i)[:-2] for i in adata.obs.index])
        elif dimensionality_reduction_method == 'autoencoder':
            data = torch.tensor(adata.X)
            x_size = data.shape[1]
            latent_size = dim
            hidden_size = int((x_size + latent_size)/2)
            nets = autoencoder(x_size=x_size, hidden_size=hidden_size, embedding_size=latent_size, p_drop=autoencoder_drop)
            optimizer_ae = torch.optim.Adam(nets.parameters(), lr=autoencoder_LR)
            loss_ae = nn.MSELoss(reduction = 'mean')
            embedding = auto_train(model=nets, 
                                   epoch_n=autoencoder_epoches,
                                   loss_fn=loss_ae, 
                                   optimizer=optimizer_ae, 
                                   data=data,
                                   cpu_num=cpu_num,
                                   device=AE_device
                                  ).detach().numpy()
            if scale == True:
                embedding = (embedding-embedding.mean(axis=0))/embedding.std(axis=0)
            table = pd.DataFrame(embedding, index=[str(i)[:-2] for i in adata.obs.index])
        elif dimensionality_reduction_method == 'nmf':
            nmf = NMF(n_components=dim).fit_transform(adata.X)
            if scale == True:
                nmf = (nmf-nmf.mean(axis=0))/nmf.std(axis=0)
            table = pd.DataFrame(nmf, index=[str(i)[:-2] for i in adata.obs.index])
        elif dimensionality_reduction_method == None:
            if scale == True:
                sc.pp.scale(adata, max_value=None, zero_center=False)
            table = pd.DataFrame(adata.X, index=[str(i)[:-2] for i in adata.obs.index], columns=adata.var.index.values)
        table.insert(0, 'ST_type', ['real']*real.shape[0]+['pseudo']*pseudo.shape[0])


    table.insert(1, 'cell_num', real.obs['cell_num'].values.tolist()+pseudo.obs['cell_num'].values.tolist())
    table.insert(2, 'cell_type_num', real.obs['cell_type_num'].values.tolist()+pseudo.obs['cell_type_num'].values.tolist())

    return table

# Experiment Plan 1,2,3
def calculate_metrics(predictions, ground_truth, coordinates, output_dir,
                     cell_types=None, dataset_name="SEQFISH+", slice_name="SVZ",group_name="cell_leq5"):
    # Experiment Plan 3
    assert (predictions > 0).sum(axis=1).min() >= 1, "There are prediction points of the cell-free type."

    # # Experiment Plan 2
    # assert predictions.shape[0] == ground_truth.shape[0], "The data volumes of Predictions and Ground Truth do not match."
    # assert predictions.shape[0] == coordinates.shape[0], "The quantity of coordinate data does not match the quantity of predicted data."
    if (predictions > 0).sum(axis=1).min() < 1:
        warnings.warn("There are prediction points of the cell-free type, which have been automatically filled with a uniform distribution.")
        zero_mask = (predictions.sum(axis=1) == 0)
        n_celltypes = predictions.shape[1]
        predictions[zero_mask] = 1 / n_celltypes


    raw_intersect = set(predictions.columns) & set(ground_truth.columns)

    common_cols = list(raw_intersect)

    pred_reordered = predictions[common_cols].columns.tolist()

    true_reordered = ground_truth[common_cols].columns.tolist()

    order_match = pred_reordered == true_reordered

    pred_matrix = predictions[common_cols].values.astype(np.float32)
    true_matrix = ground_truth[common_cols].values.astype(np.float32)



    pred_matrix = predictions[common_cols].values.astype(np.float32)
    true_matrix = ground_truth[common_cols].values.astype(np.float32)


    common_cols = list(set(predictions.columns) & set(ground_truth.columns))

    coord_matrix = coordinates.values.astype(np.float32)


    def _safe_normalize(matrix):
        matrix = np.clip(matrix, 1e-10, 1.0)
        return matrix / matrix.sum(axis=1, keepdims=True)

    gt_norm = _safe_normalize(true_matrix)
    pred_norm = _safe_normalize(pred_matrix)


    jsd = np.mean([jensenshannon(gt_row, pred_row)
                   for gt_row, pred_row in zip(gt_norm, pred_norm)])


    rmse_per_spot = np.sqrt(np.mean((gt_norm - pred_norm) ** 2, axis=1))
    rmse = np.mean(rmse_per_spot)


    spearman_list = []
    for i in range(len(true_matrix)):
        corr, _ = spearmanr(gt_norm[i], pred_norm[i])
        spearman_list.append(corr if not np.isnan(corr) else 0.0)
    spearman = np.mean(spearman_list)


    def _calculate_scs(matrix, coords, n_neighbors=5):

        actual_neighbors = min(n_neighbors, len(coords) - 1)
        if actual_neighbors < 1:
            return 0.0

        try:
            nn_model = NearestNeighbors(n_neighbors=actual_neighbors + 1)
            nn_model.fit(coords)
            _, indices = nn_model.kneighbors(coords)
            neighbor_indices = indices[:, 1:]

            similarities = []
            for i in range(len(matrix)):
                vec = matrix[i].reshape(1, -1)
                neighbors = matrix[neighbor_indices[i]]
                sim = cosine_similarity(vec, neighbors).mean()
                similarities.append(sim)
            return np.mean(similarities)
        except ValueError as e:
            print(f"SCS Calculation Warning: {str(e)}, return 0.0")
            return 0.0

    pred_scs = _calculate_scs(pred_norm, coord_matrix)
    true_scs = _calculate_scs(gt_norm, coord_matrix)

    print("Current save path:",output_dir)
    # Experiment Plan 1
    #save_metrics_to_file(rmse, spearman, jsd, pred_scs, true_scs,
    #                     dataset_name, slice_name,group_name,output_dir=output_dir)

    # Experiment Plan 2 and 3:
    os.makedirs(output_dir, exist_ok=True)
    save_metrics_to_file(rmse, spearman, jsd, pred_scs, true_scs,
                         dataset_name, slice_name, group_name,output_dir)

    return rmse, spearman, jsd, pred_scs, true_scs

# Experimental Plan 4

def calculate_type_metrics(cell_type,predictions, ground_truth, target_type, coordinates,
                          output_dir, dataset_name, slice_name):


    assert target_type in predictions.columns, f"The prediction data is missing the target type column: {target_type}"
    assert target_type in ground_truth.columns, f"The true data is missing the target type column: {target_type}"


    target_name=cell_type
    target_gt = ground_truth[target_type].values
    target_pred = predictions[target_type].values


    def _rank_based_spearman(true_values, pred_values):
        from scipy.stats import rankdata


        true_ranks = rankdata(true_values, method='average')
        pred_ranks = rankdata(pred_values, method='average')


        d_sq = np.sum((true_ranks - pred_ranks)**2)
        n = len(true_ranks)


        if n <= 1:
            return 0.0
        spearman = 1 - (6 * d_sq) / (n * (n**2 - 1))
        return spearman

    type_spearman = _rank_based_spearman(target_gt, target_pred)


    def _binary_jsd(gt_row_tuple, pred_row_tuple, target_type):


        gt_row = gt_row_tuple[1]  # (index, Series)
        pred_row = pred_row_tuple[1]


        p = np.array([gt_row[target_type], 1 - gt_row[target_type]])
        q = np.array([pred_row[target_type], 1 - pred_row[target_type]])
        return jensenshannon(p, q)



    type_jsd = np.mean([_binary_jsd(gt_row, pred_row, target_type)
                        for gt_row, pred_row in zip(ground_truth.iterrows(), predictions.iterrows())])


    type_rmse = np.sqrt(np.mean((target_gt - target_pred)**2))


    def _type_scs(matrix, coords, n_neighbors=5):

        if len(coords) < 2:
            return 0.0

        actual_neighbors = min(n_neighbors, len(coords)-1)
        if actual_neighbors < 1:
            return 0.0

        nn_model = NearestNeighbors(n_neighbors=actual_neighbors)
        nn_model.fit(coords)
        _, indices = nn_model.kneighbors(coords)

        similarities = []
        for i in range(len(matrix)):
            vec = matrix[i].reshape(1, -1)
            neighbors = matrix[indices[i]]
            sim = cosine_similarity(vec, neighbors).mean()
            similarities.append(sim)
        return np.mean(similarities)


    scs_matrix = predictions[[target_type]].values.astype(np.float32)
    type_scs = _type_scs(scs_matrix, coordinates.values) if len(coordinates) >=2 else 0.0



    filename = os.path.join(output_dir, f"{target_name}.txt")
    with open(filename, 'w') as f:
        f.write(f"RMSE: {type_rmse:.4f}\n")
        f.write(f"Spearman Correlation: {type_spearman:.4f}\n")
        f.write(f"Jensen-Shannon Divergence: {type_jsd:.4f}\n")
        f.write(f"SCS: {type_scs:.4f}\n")
    print(f"The indicator has been saved as: {filename}")


    # plot_type_metrics(type_rmse, type_spearman, type_jsd, type_scs,
    #                 dataset_name, slice_name, target_type, output_dir)

    return type_rmse, type_spearman, type_jsd, type_scs

## Experiment Plans 1, 2, 3:
def save_metrics_to_file(rmse, spearman, jsd, pred_scs, true_scs,
                         dataset_name, slice_name, group_name,output_dir="results/"):
    """保存指标到文件"""
    # Experiment Plan 1

    if dataset_name == "total_average":
        filename = "total_average_original_STD_Net.txt"
    else:
        filename = f"metrics_{dataset_name}_{group_name}.txt"

    #file_path = os.path.join(output_dir, filename)


    #with open(file_path, 'w') as f:
    #    if dataset_name != "total_average":
    #        f.write(f"Dataset: {dataset_name}\n")
    #        f.write(f"Slice: {slice_name}\n")
    #    f.write(f"RMSE: {rmse:.4f}\n")
    #    f.write(f"Spearman Correlation: {spearman:.4f}\n")
    #    f.write(f"Jensen-Shannon Divergence: {jsd:.4f}\n")
    #    f.write(f"Predicted SCS: {pred_scs:.4f}\n")
     #   f.write(f"True SCS: {true_scs:.4f}\n")

    #print(f"The indicator has been saved to: {file_path}")

    # Experiment Plan 2, 3
    filename = f"{output_dir}/metrics_{dataset_name}_{group_name}.txt"
    with open(filename, 'w') as f:
        f.write(f"Dataset: {dataset_name}\ngroup_name: {group_name}\n")
        f.write(f"RMSE: {rmse:.4f}\n")
        f.write(f"Spearman Correlation: {spearman:.4f}\n")
        f.write(f"Jensen-Shannon Divergence: {jsd:.4f}\n")
        f.write(f"Predicted SCS: {pred_scs:.4f}\n")
        f.write(f"True SCS: {true_scs:.4f}\n")
    print(f"The indicator has been saved as: {filename}")

#Experimental Plan 4：

def plot_type_metrics(rmse, spearman, jsd, scs,
                      dataset_name, slice_name, cell_type, output_dir):

    metrics = {
        'RMSE': rmse,
        'Spearman': spearman,
        'JSD': jsd,
        'SCS': scs
    }

    fig, ax = plt.subplots(figsize=(10, 6))
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728']
    bars = ax.bar(metrics.keys(), metrics.values(), color=colors)


    for bar in bars:
        height = bar.get_height()
        ax.annotate(f'{height:.3f}',
                    xy=(bar.get_x() + bar.get_width() / 2, height),
                    xytext=(0, 3),
                    textcoords="offset points",
                    ha='center', va='bottom')

    ax.set_ylabel('Score', fontsize=12)
    ax.set_title(f'{cell_type} Metrics - {dataset_name} ({slice_name})', fontsize=14)
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()


    plt.savefig(os.path.join(output_dir, f"type_metrics_{cell_type}.png"), dpi=300)
    plt.close()







def collect_and_average_metrics(output_root, final_data_dir):

    metrics_files = []


    for root, dirs, files in os.walk(output_root):
        for file in files:
            if file.startswith("metrics_") and file.endswith(".txt"):
                metrics_files.append(os.path.join(root, file))


    total_metrics = {
        'rmse': 0.0,
        'spearman': 0.0,
        'jsd': 0.0,
        'pred_scs': 0.0,
        'true_scs': 0.0,
        'count': 0
    }

    for file_path in metrics_files:
        try:
            with open(file_path, 'r') as f:
                content = f.read()


            rmse = float(re.search(r"RMSE: (\d+\.\d+)", content).group(1))
            spearman = float(re.search(r"Spearman Correlation: (-?\d+\.\d+)", content).group(1))
            jsd = float(re.search(r"Jensen-Shannon Divergence: (\d+\.\d+)", content).group(1))
            pred_scs = float(re.search(r"Predicted SCS: (\d+\.\d+)", content).group(1))
            true_scs = float(re.search(r"True SCS: (\d+\.\d+)", content).group(1))

            total_metrics['rmse'] += rmse
            total_metrics['spearman'] += spearman
            total_metrics['jsd'] += jsd
            total_metrics['pred_scs'] += pred_scs
            total_metrics['true_scs'] += true_scs
            total_metrics['count'] += 1

        except Exception as e:
            print(f"Warning: Unable to parse file {file_path}. Error message：{str(e)}")


    if total_metrics['count'] > 0:
        avg_metrics = {
            'rmse': total_metrics['rmse'] / total_metrics['count'],
            'spearman': total_metrics['spearman'] / total_metrics['count'],
            'jsd': total_metrics['jsd'] / total_metrics['count'],
            'pred_scs': total_metrics['pred_scs'] / total_metrics['count'],
            'true_scs': total_metrics['true_scs'] / total_metrics['count']
        }


        save_metrics_to_file(
            avg_metrics['rmse'],
            avg_metrics['spearman'],
            avg_metrics['jsd'],
            avg_metrics['pred_scs'],
            avg_metrics['true_scs'],
            dataset_name="total_average",
            slice_name="all_slices",
            group_name="original_STD_Net",
            output_dir=output_root
        )
        print(f"Successfully aggregated {total_metrics['count']} datasets of metrics")
    else:
        print("No valid indicator file was found.")

def deep_clean():

    import matplotlib.pyplot as plt
    plt.close('all')

    try:
        import tensorflow as tf
        tf.keras.backend.clear_session()
        if tf.executing_eagerly():
            tf.compat.v1.reset_default_graph()
    except:
        pass

    try:
        import torch
        torch.cuda.empty_cache()
    except:
        pass

    import gc
    gc.collect()
