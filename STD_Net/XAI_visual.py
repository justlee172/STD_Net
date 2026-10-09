import random

from tqdm.notebook import tqdm
import matplotlib.pyplot as plt


import numpy as np
import os



import torch




def plot_metrics(rmse, spearman, jsd, pred_scs, true_scs,
                 dataset_name, slice_name, group_name="cell_leq5",output_dir="results/"):

    metrics = {
        'RMSE': rmse,
        'Spearman': spearman,
        'JSD': jsd,
        'Pred_SCS': pred_scs,
        'True_SCS': true_scs
    }

    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd']

    fig, ax = plt.subplots(figsize=(10, 6))
    bars = ax.bar(metrics.keys(), metrics.values(), color=colors)


    for bar in bars:
        height = bar.get_height()
        ax.annotate(f'{height:.2f}',
                    xy=(bar.get_x() + bar.get_width() / 2, height),
                    xytext=(0, 3),  # 3 points vertical offset
                    textcoords="offset points",
                    ha='center', va='bottom')

    ax.set_ylabel('Score', fontsize=12)
    ax.set_title(f'Metrics - {dataset_name} ({slice_name})', fontsize=14)
    plt.xticks(rotation=45, ha='right', fontsize=10)
    plt.tight_layout()


    # Experimental Plan 1
    #filename = f"{output_dir}/metrics_chart_{dataset_name}_{slice_name}.png"

    # Experimental Plan 2,3
    filename = f"{output_dir}/metrics_chart_{dataset_name}_{group_name}.png"


    plt.savefig(filename, dpi=300, bbox_inches='tight')
    plt.close()



def plot_umap_comparison(ground_truth, predictions, coordinates, dataset_name, slice_name,group_name="cell_leq5",output_dir="results/"):

    import umap
    import pandas as pd

    os.makedirs(output_dir, exist_ok=True)


    combined_data = np.vstack([ground_truth.values, predictions.values])
    labels = ['True'] * len(ground_truth) + ['Pred'] * len(predictions)


    reducer = umap.UMAP(n_neighbors=15, min_dist=0.1, metric='cosine', random_state=42)
    umap_emb = reducer.fit_transform(combined_data)


    fig, ax = plt.subplots(1, 2, figsize=(14, 6))


    for label, color in zip(['True', 'Pred'], ['#1f77b4', '#ff7f0e']):
        mask = np.array(labels) == label
        ax[0].scatter(umap_emb[mask, 0], umap_emb[mask, 1], label=label, alpha=0.6, c=color)
    ax[0].set_xlabel('UMAP1', fontsize=12)
    ax[0].set_ylabel('UMAP2', fontsize=12)
    ax[0].set_title('UMAP Projection: True vs Pred', fontsize=14)
    ax[0].legend()


    spatial_plot = ax[1].scatter(
        coordinates['coor_X'], coordinates['coor_Y'],
        c=umap_emb[:len(ground_truth), 0], cmap='viridis', alpha=0.8, s=50
    )
    ax[1].set_xlabel('Spatial X', fontsize=12)
    ax[1].set_ylabel('Spatial Y', fontsize=12)
    ax[1].set_title('Spatial Coordinates Colored by UMAP1', fontsize=14)
    plt.colorbar(spatial_plot, ax=ax[1], label='UMAP1')

    plt.tight_layout()
    # # Experimental Plan 1
    #plt.savefig(f"{output_dir}/umap_{dataset_name}_{slice_name}.png", dpi=300)
    # Experimental Plan 2,3:
    plt.savefig(f"{output_dir}/umap_{dataset_name}_{group_name}.png", dpi=300)

    plt.close()



