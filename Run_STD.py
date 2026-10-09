#!/usr/bin/env python
# coding: utf-8



import matplotlib
matplotlib.use('Agg')
import multiprocessing
multiprocessing.set_start_method('spawn', force=True)



import os
import sys
import traceback
import warnings

warnings.filterwarnings("ignore")
sys.path.append(os.getcwd())
from STD_Net.STD_Net import run_STD_Net

# from STD_Net.deconvolution_models import SpatialDeconvolution

def init_worker():

    import matplotlib
    matplotlib.use('Agg')
    import scanpy as sc
    sc.settings.autoshow = False

spatial_deconv_params = {
    'method': 'STD_Net',
    'params': {
        'RCTD': {
            'doublet_mode': 'full',
            'max_cores': 8
        },
        'SPOTlight': {
            'n_iter': 1000,
            'clustering': 'louvain'
        },
        'Cell2location': {
            'N_cells_per_location': 7,
            'detection_alpha': 200
        },
        'CARD': {
            'lambda_g': 0.5,
            'lambda_r': 0.1
        },
        'Tangram': {
            'mode': 'clusters',
            'cluster_label': 'cell_type'
        },
        'Stereoscope': {
            'n_epochs': 10000,
            'lr': 0.01
        }
    }
}


paths = {
    'sc_path': './data/sc_data',
    'ST_path': './data/ST_data',
    'output_path': './results/',
    'dataset_name' : "SEQFISH+",
    'slice_name' : "SVZ",
}



find_marker_genes_paras = {
    'preprocess': True,
    'normalize': True,
    'log': True,
    'highly_variable_genes': True,
    'highly_variable_gene_num': None,
    'regress_out': True,
    'PCA_components': 15,
    'marker_gene_method': 'logreg',
    'top_gene_per_type': 100,
    'filter_wilcoxon_marker_genes': True,
    'pvals_adj_threshold': 0.10,
    'log_fold_change_threshold': 1,
    'min_within_group_fraction_threshold': None,
    'max_between_group_fraction_threshold': None,
    'preserve_full_targeted_panel': True,
    'targeted_panel_gene_limit': 2000,
}



'''
This module is used to simulate pseudo-spots.

Parameters:
'spot_num': [int]. The number of pseudo-spots.
'min_cell_num_in_spot': [int]. The minimum number of cells in a pseudo-spot.
'max_cell_num_in_spot': [int]. The maximum number of cells in a pseudo-spot.
'generation_method': ['cell' or 'celltype']. STD_Net provides two pseudo-spot simulation methods. When 'generation_method'='cell', each cell is equally selected. When 
                    'generation_method'='celltype', each cell type is equally selected. See manuscript for more details.
'max_cell_types_in_spot': [int]. When 'generation_method'='celltype', choose the maximum number of cell types in a pseudo-spot.
'''
pseudo_spot_simulation_paras = {
    'spot_num': 30000,
    'min_cell_num_in_spot': 8,
    'max_cell_num_in_spot': 12,
    'generation_method': 'celltype',
    'max_cell_types_in_spot': 4,
    'low_density_quantile': 0.20,
    'kde_bandwidth': 0.1,
    'low_abundance_threshold': 0.05,
    'candidate_cell_fraction': 0.20,
    'candidate_spot_probability': 0.25,
    'oversample_ratio': 3.0,
    'interpolation_alpha_min': 0.3,
    'interpolation_alpha_max': 0.7,
    'probability_mode_weights': (0.25, 0.30, 0.30, 0.15),
    'hybrid_p': 0.7,
    'enable_low_abundance_boost': True,
    'require_spatial_coordinates': True,
    'random_seed': 0,
    'spot_ratio_to_real': 8.0,
    'minimum_spot_count': 100,
    'cell_type_count_weights': (0.70, 0.22, 0.06, 0.02),
    'dirichlet_alpha': 0.35,
    'platform_match': True,
    'platform_ratio_clip': (0.25, 4.0),
    # Optional unsupervised calibration of the pseudo-spot support-size prior
    # from the observed target expression.  The historical default remains off
    # so the original benchmark profile stays reproducible.
    'target_complexity_calibration': False,
}



'''
This module is used for real- and pseudo- spots normalization.

Parameters:
'normalize': [bool]. Select whether you need to normalize each cell/spot by total counts = 10,000, so that every cell/spot has the same total count after normalization.
'log': [bool]. Select whether you need to logarithmize (X=log(X+1)) the expression matrix.
'scale': [bool]. Select whether you need to scale each gene to unit variance and zero mean.
'''
data_normalization_paras = {
    'normalize': True, 
    'log': True, 
    'scale': False,
}




integration_for_adj_paras = {
    'batch_removal_method': None, 
    'dim': 30, 
    'dimensionality_reduction_method': 'PCA',
    'scale': True,
}

inter_exp_adj_paras = {
    'find_neighbor_method': 'MNN', 
    'dist_method': 'cosine', 
    'corr_dist_neighbors': 8,
}
real_intra_exp_adj_paras = {
    'find_neighbor_method': 'MNN', 
    'dist_method': 'cosine',  
    'corr_dist_neighbors': 8,
    'PCA_dimensionality_reduction': False,
    'dim': 50,
}
pseudo_intra_exp_adj_paras = {
    'find_neighbor_method': 'MNN', 
    'dist_method': 'cosine', 
    'corr_dist_neighbors': 8,
    'PCA_dimensionality_reduction': False,
    'dim': 50,
}

spatial_adj_paras = {
    'link_method': 'soft',
    'space_dist_neighbors': 8,
    'space_dist_threshold': 2,
    'n_clusters': 7,
    'boundary_threshold_percent': 0.1,
    'boundary_decay': 0.3,
}

integration_for_feature_paras = {
    'batch_removal_method': None, 
    'dimensionality_reduction_method': None, 
    'dim': 80,
    'scale': True,
}


DGAT_paras = {

    'optimization_profile': 'full_v2',

    'use_gatv2': True,
    'gat_heads': 4,
    'use_sparse_graph': True,
    'output_transform': 'sparsemax',
    # A small training-time softmax component prevents sparsemax from
    # permanently killing rare cell-type logits.  Hard sparsemax is restored
    # for the final prediction when this is non-zero.
    'support_smoothing_epsilon': 0.0,
    'domain_adversarial': False,
    'domain_adversarial_lambda': 1.0,
    'adaptive_signature': False,
    'adaptive_signature_rank': 8,
    'use_pseudo_spatial_graph': False,
    'pseudo_spatial_neighbors': 4,
    'pseudo_spatial_edge_scale': 0.25,
    'prediction_zero_threshold': 1e-7,
    'gated_fusion': True,
    'use_gene_weights': True,

    'epoch_n': 1500,
    'dim': 64,
    'common_hid_layers_num': 2,
    'fcnn_hid_layers_num': 1,
    'dropout': 0.2,
    'optimizer': 'AdamW',
    'learning_rate': 1e-3,
    'learning_rate_SGD': 2e-1,
    'weight_decay_SGD': 1e-4,
    'adamw_betas': (0.9, 0.999),
    'momentum': 0.9,
    'dampening': 0,
    'nesterov': True,
    'early_stopping_patience': 60,
    'clip_grad_max_norm': 1.0,
    'print_loss_epoch_step': 20,
    'objective': {
        'enabled': True,
        'composition': {
            'js_weight': 1.0,
            'rmse_weight': 1.0,
            'support_weight': 0.20,
            'entropy_weight': 0.10,
            'rare_weight_power': 0.5,
            'support_threshold': 1e-4,
            'focal_gamma': 2.0,
        },
        'reconstruction_weight': 0.05,
        'prototype_weight': 0.05,
        'spatial_weight': 0.02,
        'confidence_threshold': 0.35,
        'early_stopping_min_delta': 1e-5,
    },
    'enable_explain': False,
    'use_shap': False,
    'use_lime': False,
    'explain_samples': 50,
}

runtime_n_jobs = -1



def process_single_dataset(final_data_root,output_root,dataset_dir):
    """Function for processing a single dataset"""
    try:
        print(f"-------------------processing {dataset_dir} slicing data：------------------------")
        dataset_path = os.path.join(final_data_root, dataset_dir)

        if os.path.isdir(dataset_path):
            print("yes")

            dynamic_paths = {
                'sc_path': os.path.join(dataset_path, "sc_data"),
                'ST_path': os.path.join(dataset_path, "st_data"),
                'output_path': os.path.join(output_root, dataset_dir) + "/",
                'dataset_name': dataset_dir,
                'slice_name': "slice_0"
            }


            required_files = {
                'sc': ["sc_data.tsv", "sc_label.tsv"],
                'st': ["ST_data.tsv", "coordinates.csv"]
            }

            missing_files = []
            for f in required_files['sc']:
                if not os.path.exists(os.path.join(dynamic_paths['sc_path'], f)):
                    missing_files.append(f"sc_data/{f}")
            for f in required_files['st']:
                if not os.path.exists(os.path.join(dynamic_paths['ST_path'], f)):
                    missing_files.append(f"st_data/{f}")
            if missing_files:
                print(f"Skipping dataset {dataset_dir}, missing files: {', '.join(missing_files)}")
                return

            os.makedirs(dynamic_paths['output_path'], exist_ok=True)

            results = run_STD_Net(
                dynamic_paths,
                deconv_method=spatial_deconv_params['method'],
                method_params=spatial_deconv_params['params'],
                load_test_groundtruth=False,
                use_marker_genes=True,
                external_genes=False,
                find_marker_genes_paras=find_marker_genes_paras,
                generate_new_pseudo_spots=True,
                pseudo_spot_simulation_paras=pseudo_spot_simulation_paras,
                data_normalization_paras=data_normalization_paras,
                integration_for_adj_paras=integration_for_adj_paras,
                inter_exp_adj_paras=inter_exp_adj_paras,
                spatial_adj_paras=spatial_adj_paras,
                real_intra_exp_adj_paras=real_intra_exp_adj_paras,
                pseudo_intra_exp_adj_paras=pseudo_intra_exp_adj_paras,
                integration_for_feature_paras=integration_for_feature_paras,
                DGAT_paras=DGAT_paras,
                fraction_pie_plot=True,
                cell_type_distribution_plot=True,
                n_jobs=runtime_n_jobs,
                DGAT_device='GPU',
            )
            results.write_h5ad(f"{dynamic_paths['output_path']}/results.h5ad")
            print(f"Successfully processed the dataset: {dataset_dir}")
        else:
            print("NO")

        return True
    except Exception as e:
        print(f"Failure in handling: {str(e)}")
        traceback.print_exc()
        return False
    finally:
        from STD_Net.process_utils import deep_clean
        deep_clean()


if __name__ == '__main__':

    import matplotlib

    matplotlib.use('Agg')
    import scanpy as sc

    sc.settings.autoshow = False

    final_data_root = "final_data"
    output_root = "./results"

    dataset_dirs = [d for d in os.listdir(final_data_root)
                    if os.path.isdir(os.path.join(final_data_root, d))]

    for dataset_dir in dataset_dirs:
        results = process_single_dataset(final_data_root,output_root,dataset_dir)


    # ------------------ Summing and averaging of Experiment 1 ------------------
    # print(f"\nSuccessfully processed {sum(results)}/{len(dataset_dirs)} datasets")
    # from STD_Net.process_utils import collect_and_average_metrics

    # collect_and_average_metrics(output_root, final_data_root)
# if __name__ == '__main__':


#     save results.h5ad
#     results.write_h5ad(f"{paths['output_path']}{paths['dataset_name']}/{paths['slice_name']}/results.h5ad")
