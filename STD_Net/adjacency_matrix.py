import torch
import numpy as np
import pandas as pd
import scanpy as sc
import scipy.sparse as sp
from sklearn.cluster import KMeans
from sklearn.neighbors import KDTree
from sklearn.metrics import DistanceMetric
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.neighbors import NearestNeighbors

# Add the spatial clustering function before constructing the adjacency matrix
def spatial_clustering(ST_adata, n_clusters=7):

    # Obtain spatial coordinates
    spatial_coords = ST_adata.obs[['coor_X', 'coor_Y']].values

    # Perform spatial clustering using KMeans
    kmeans = KMeans(n_clusters=n_clusters, random_state=0)
    cluster_labels = kmeans.fit_predict(spatial_coords)

    return cluster_labels

def intra_dist_adj_with_clusters(ST_exp,
                                 cluster_labels,
                                 link_method='soft',
                                 space_dist_neighbors=27,
                                 space_dist_threshold=None,
                                 restrict_within_cluster=True):

    # Use KNN to find the nearest neighbors
    knn = NearestNeighbors(n_neighbors=space_dist_neighbors, metric='minkowski')
    knn.fit(ST_exp.obs[['coor_X', 'coor_Y']])
    dist, ind = knn.kneighbors()

    n_points = ST_exp.shape[0]
    A_space = np.zeros((n_points, n_points), dtype=float)

    for i in range(ind.shape[0]):
        for j in range(ind.shape[1]):
            neighbor = int(ind[i, j])
            same_cluster = cluster_labels[i] == cluster_labels[neighbor]
            if same_cluster or not restrict_within_cluster:
                if space_dist_threshold is not None and dist[i, j] >= space_dist_threshold:
                    continue

                if link_method == 'hard':
                    A_space[i, neighbor] = 1
                    A_space[neighbor, i] = 1
                else:
                    weight = 1 / max(float(dist[i, j]), 1e-12)
                    A_space[i, neighbor] = weight
                    A_space[neighbor, i] = weight

    A_space = pd.DataFrame(A_space,
                           index=ST_exp.obs.index.values,
                           columns=ST_exp.obs.index.values)
    return A_space


def detect_boundary_points(coords, cluster_labels, threshold_percent=0.1):

    from scipy.spatial import ConvexHull, distance_matrix
    import numpy as np

    unique_clusters = np.unique(cluster_labels)
    boundary_flags = np.zeros(len(coords), dtype=bool)

    # Calculate each boundary point of the cluster
    for cluster_id in unique_clusters:
        cluster_indices = np.where(cluster_labels == cluster_id)[0]
        cluster_points = coords[cluster_indices]

        # Calculate the convex hull of the clustering
        if len(cluster_points) >= 3:
            try:
                hull = ConvexHull(cluster_points)
            except Exception:
                continue
            boundary_indices = cluster_indices[hull.vertices]
            boundary_flags[boundary_indices] = True

    # Calculate the average inter-cluster distance
    cluster_centers = []
    for cluster_id in unique_clusters:
        cluster_points = coords[cluster_labels == cluster_id]
        cluster_centers.append(np.mean(cluster_points, axis=0))

    min_dist = np.inf
    for i in range(len(cluster_centers)):
        for j in range(i + 1, len(cluster_centers)):
            dist = np.linalg.norm(cluster_centers[i] - cluster_centers[j])
            if dist < min_dist:
                min_dist = dist

    # Set the distance threshold
    dist_threshold = min_dist * threshold_percent

    # Adjust the boundary points based on the distance between adjacent clusters
    dist_matrix = distance_matrix(coords, coords)
    for i in np.where(boundary_flags)[0]:
        neighbor_indices = np.where(dist_matrix[i] < dist_threshold)[0]
        neighbor_clusters = np.unique(cluster_labels[neighbor_indices])
        if len(neighbor_clusters) > 1 and cluster_labels[i] in neighbor_clusters:
            boundary_flags[i] = True
        else:
            boundary_flags[i] = False

    return boundary_flags


def intra_dist_adj_with_boundary_decay(ST_exp,
                                       cluster_labels,
                                       boundary_flags,
                                       link_method='soft',
                                       space_dist_neighbors=27,
                                       space_dist_threshold=None,
                                       boundary_decay=0.3):

    # Create the basic adjacency matrix
    A_base = intra_dist_adj_with_clusters(
        ST_exp,
        cluster_labels,
        link_method,
        space_dist_neighbors,
        space_dist_threshold,
        restrict_within_cluster=False,
    ).values

    # The number of points obtained
    n = ST_exp.shape[0]

    # Create the attenuation weight matrix
    decay_matrix = np.ones((n, n))

    # Application of boundary attenuation
    for i in range(n):
        for j in range(i + 1, n):
            is_cross_domain = cluster_labels[i] != cluster_labels[j]
            touches_interface = boundary_flags[i] or boundary_flags[j]
            if is_cross_domain and touches_interface and A_base[i, j] > 0:
                decay_matrix[i, j] = boundary_decay
                decay_matrix[j, i] = boundary_decay

    # Application attenuation
    A_decayed = A_base * decay_matrix

    return pd.DataFrame(A_decayed,
                        index=ST_exp.obs.index.values,
                        columns=ST_exp.obs.index.values)


def intra_exp_adj_with_clusters(adata,
                                cluster_labels,
                                find_neighbor_method='KNN',
                                dist_method='euclidean',
                                PCA_dimensionality_reduction=True,
                                dim=50,
                                corr_dist_neighbors=10,
                                ):

    ST_exp = adata.copy()
    sc.pp.scale(ST_exp, max_value=None, zero_center=True)

    if PCA_dimensionality_reduction:
        sc.tl.pca(ST_exp, n_comps=dim, svd_solver='arpack', random_state=None)
        input_data = ST_exp.obsm['X_pca']
    else:
        input_data = ST_exp.X

    n_points = ST_exp.shape[0]
    A_exp = np.zeros((n_points, n_points), dtype=float)

    if find_neighbor_method == 'KNN':
        if dist_method == 'cosine':
            cos_sim = cosine_similarity(input_data, input_data)
            k_index = torch.topk(torch.tensor(cos_sim), k=corr_dist_neighbors, dim=1)[1].numpy()
        else:
            dist_metric = DistanceMetric.get_metric(dist_method)
            k_index = dist_metric.pairwise(input_data)
            k_index = np.argsort(k_index, axis=1)[:, :corr_dist_neighbors]

        for i in range(n_points):
            for j_idx in k_index[i]:
                j = j_idx
                # Only connect the points within the same cluster
                if i != j and cluster_labels[i] == cluster_labels[j]:
                    A_exp[i, j] = 1
                    A_exp[j, i] = 1

    elif find_neighbor_method == 'MNN':
        mut = find_mutual_nn(input_data, input_data,
                             dist_method=dist_method,
                             k1=corr_dist_neighbors,
                             k2=corr_dist_neighbors)
        mut = pd.DataFrame(mut, columns=['data1', 'data2'])

        for _, row in mut.iterrows():
            i, j = row['data1'], row['data2']
            # Only connect the points within the same cluster
            if i != j and cluster_labels[i] == cluster_labels[j]:
                A_exp[i, j] = 1
                A_exp[j, i] = 1

    A_exp = pd.DataFrame(A_exp,
                         index=ST_exp.obs.index.values,
                         columns=ST_exp.obs.index.values)
    return A_exp

# Output all index pairs that satisfy the condition of being nearest neighbors to provide a basis for constructing a weighted graph or adjacency matrix.
def find_mutual_nn(data1, 
                   data2, 
                   dist_method, 
                   k1, 
                   k2, 
                  ):
    if dist_method == 'cosine':
        cos_sim1 = cosine_similarity(data1, data2)
        cos_sim2 = cosine_similarity(data2, data1)
        k_index_1 = torch.topk(torch.tensor(cos_sim2), k=k2, dim=1)[1]
        k_index_2 = torch.topk(torch.tensor(cos_sim1), k=k1, dim=1)[1]
    else:
        dist = DistanceMetric.get_metric(dist_method)
        k_index_1 = KDTree(data1, metric=dist).query(data2, k=k2, return_distance=False)
        k_index_2 = KDTree(data2, metric=dist).query(data1, k=k1, return_distance=False)
    mutual_1 = []
    mutual_2 = []
    mutual = []
    for index_2 in range(data2.shape[0]):
        for index_1 in k_index_1[index_2]:
            if index_2 in k_index_2[index_1]: 
                mutual_1.append(index_1)
                mutual_2.append(index_2)
                mutual.append([index_1, index_2])
    return mutual


def inter_adj(ST_integration, 
              find_neighbor_method='MNN',
              dist_method='euclidean',
              corr_dist_neighbors=20, 
             ):
    # The MNN/KNN method selects a certain number of neighbors and constructs an adjacency matrix.
    #Perform the adjacency matrix construction operation on the anndata format dataset that was synthesized and mapped out from the data just created above in STD_Net.py

    # Neighbor search method
    if find_neighbor_method == 'KNN':
        real = ST_integration[ST_integration['ST_type'] == 'real']
        pseudo = ST_integration[ST_integration['ST_type'] == 'pseudo']
        data1 = real.iloc[:, 3:]
        data2 = pseudo.iloc[:, 3:]
        real_num = real.shape[0]
        pseudo_num = pseudo.shape[0]
        # Measurement method - Calculate the distance according to the different types of distances using specific formulas
        if dist_method == 'cosine':
            # Calculate the distance between real and pseudo
            cos_sim = cosine_similarity(data1, data2)
            # 用It is used to select the k nearest neighbors for each real data point based on the cosine similarity with the pseudo data points, and then return the indices of these neighbors.
            k_index = torch.topk(torch.tensor(cos_sim), k=corr_dist_neighbors, dim=1)[1]
        else:
            dist = DistanceMetric.get_metric(dist_method)
            k_index = KDTree(data2, metric=dist).query(data1, k=corr_dist_neighbors, return_distance=False)
        A_exp = np.zeros((ST_integration.shape[0], ST_integration.shape[0]), dtype=float)
        for i in range(k_index.shape[0]):
            for j in k_index[i]:
                A_exp[i, j+real_num] = 1;
                A_exp[j+real_num, i] = 1;  
        A_exp = pd.DataFrame(A_exp, index=ST_integration.index, columns=ST_integration.index)

        # MNN looks for the nearest neighbors that are mutually recognized between two sets of data. It is generally used for cross-domain data matching, such as finding the most similar data pairs from different sources (such as fake data and real data).
    elif find_neighbor_method == 'MNN':
        real = ST_integration[ST_integration['ST_type'] == 'real']
        pseudo = ST_integration[ST_integration['ST_type'] == 'pseudo']
        data1 = real.iloc[:, 3:]
        data2 = pseudo.iloc[:, 3:]
        mut = find_mutual_nn(data2, data1, dist_method=dist_method, k1=corr_dist_neighbors, k2=corr_dist_neighbors)
        mut = pd.DataFrame(mut, columns=['pseudo', 'real'])
        real_num = real.shape[0]
        pseudo_num = pseudo.shape[0]
        A_exp = np.zeros((real_num+pseudo_num, real_num+pseudo_num), dtype=float)
        for i in mut.index:
            A_exp[mut.loc[i, 'real'], mut.loc[i, 'pseudo']+real_num] = 1
            A_exp[mut.loc[i, 'pseudo']+real_num, mut.loc[i, 'real']] = 1
        A_exp = pd.DataFrame(A_exp, index=ST_integration.index, columns=ST_integration.index)
    
    return A_exp


def intra_exp_adj(adata,
                  find_neighbor_method='KNN',
                  dist_method='euclidean',
                  PCA_dimensionality_reduction=True,
                  dim=50,
                  corr_dist_neighbors=10,
                  ):

    ST_exp = adata.copy()

    sc.pp.scale(ST_exp, max_value=None, zero_center=True)
    if PCA_dimensionality_reduction == True:
        sc.tl.pca(ST_exp, n_comps=dim, svd_solver='arpack', random_state=None)
        input_data = ST_exp.obsm['X_pca']
        if find_neighbor_method == 'KNN':
            if dist_method == 'cosine':
                cos_sim = cosine_similarity(input_data, input_data)
                k_index = torch.topk(torch.tensor(cos_sim), k=corr_dist_neighbors, dim=1)[1]
            else:
                dist = DistanceMetric.get_metric(dist_method)
                k_index = KDTree(input_data, metric=dist).query(input_data, k=corr_dist_neighbors, return_distance=False)
            A_exp = np.zeros((ST_exp.shape[0], ST_exp.shape[0]), dtype=float)
            for i in range(k_index.shape[0]):
                for j in k_index[i]:
                    if i != j:
                        A_exp[i, j] = 1;
                        A_exp[j, i] = 1;
            A_exp = pd.DataFrame(A_exp, index=ST_exp.obs.index.values, columns=ST_exp.obs.index.values)
        elif find_neighbor_method == 'MNN':
            mut = find_mutual_nn(input_data, input_data, dist_method=dist_method, k1=corr_dist_neighbors, k2=corr_dist_neighbors)
            mut = pd.DataFrame(mut, columns=['data1', 'data2'])
            A_exp = np.zeros((ST_exp.shape[0], ST_exp.shape[0]), dtype=float)
            for i in mut.index:
                A_exp[mut.loc[i, 'data1'], mut.loc[i, 'data2']] = 1
                A_exp[mut.loc[i, 'data2'], mut.loc[i, 'data1']] = 1
            A_exp = A_exp - np.eye(A_exp.shape[0])
            A_exp = pd.DataFrame(A_exp, index=ST_exp.obs.index.values, columns=ST_exp.obs.index.values)
    else:
        sc.pp.scale(ST_exp, max_value=None, zero_center=True)
        input_data = ST_exp.X
        if find_neighbor_method == 'KNN':
            if dist_method == 'cosine':
                cos_sim = cosine_similarity(input_data, input_data)
                k_index = torch.topk(torch.tensor(cos_sim), k=corr_dist_neighbors, dim=1)[1]
            else:
                dist = DistanceMetric.get_metric(dist_method)
                k_index = KDTree(input_data, metric=dist).query(input_data, k=corr_dist_neighbors, return_distance=False)
            A_exp = np.zeros((ST_exp.shape[0], ST_exp.shape[0]), dtype=float)
            for i in range(k_index.shape[0]):
                for j in k_index[i]:
                    if i != j:
                        A_exp[i, j] = 1;
                        A_exp[j, i] = 1;
            A_exp = pd.DataFrame(A_exp, index=ST_exp.obs.index.values, columns=ST_exp.obs.index.values)
        elif find_neighbor_method == 'MNN':
            mut = find_mutual_nn(input_data, input_data, dist_method=dist_method, k1=corr_dist_neighbors, k2=corr_dist_neighbors)
            mut = pd.DataFrame(mut, columns=['data1', 'data2'])
            A_exp = np.zeros((ST_exp.shape[0], ST_exp.shape[0]), dtype=float)
            for i in mut.index:
                A_exp[mut.loc[i, 'data1'], mut.loc[i, 'data2']] = 1
                A_exp[mut.loc[i, 'data2'], mut.loc[i, 'data1']] = 1
            A_exp = A_exp - np.eye(A_exp.shape[0])
            A_exp = pd.DataFrame(A_exp, index=ST_exp.obs.index.values, columns=ST_exp.obs.index.values)

    return A_exp


# Based on the sample type, a global adjacency matrix is constructed to integrate the local adjacency relationships of both the real samples and the pseudo samples.

def A_intra_transfer(data, data_type, real_num, pseudo_num):
    
    adj = np.zeros((real_num+pseudo_num, real_num+pseudo_num), dtype=float)
    if data_type == 'real':      
        adj[:real_num, :real_num] = data
    elif data_type == 'pseudo':
        adj[real_num:, real_num:] = data
        
    return adj


def adj_normalize(mx, symmetry=True):

    mx = sp.csr_matrix(mx)

    rowsum = np.array(mx.sum(1))
    r_inv = np.power(rowsum, -1).flatten()
    r_inv[np.isinf(r_inv)] = 0.

    if symmetry == True:
        r_mat_inv = sp.diags(np.sqrt(r_inv))
        mx = r_mat_inv.dot(mx).dot(r_mat_inv)
    else:
        r_mat_inv = sp.diags(r_inv)
        mx = r_mat_inv.dot(mx)
    
    return mx.todense()
