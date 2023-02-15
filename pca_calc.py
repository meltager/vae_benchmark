# This file is a test file to calc. the PCA and it so far it is not a part of the project so don't include it for the
# calculation of the project.
import numpy as np
from sklearn.decomposition import PCA
from sklearn.neighbors import NearestNeighbors
import  igraph as ig
import leidenalg as la
from sklearn import metrics


pca = PCA(n_components=10)  # The number of PC's required for the calc.
transformed = pca.fit_transform(np.nan_to_num(self.rna_data_subset))

#This part it a copy and paste from the experiment.py file : Function : get_data_cluster()
# Now the data is transformed to PCA, we should now Do the custering and then calculate the ARI
neighbours = NearestNeighbors(n_neighbors = 15, metric='minkowski',p=2)
neighbours.fit(transformed)
neighbours_list = neighbours.kneighbors(transformed, return_distance = False)
adj_mtx = np.zeros((transformed.shape[0],transformed.shape[0]))
tmp_idx = 0
for i in neighbours_list:
    for j in i:
        adj_mtx[tmp_idx,j] = 1
    tmp_idx+=1

g = ig.Graph.Adjacency(adj_mtx, mode='undirected')
partition = la.find_partition(g, la.ModularityVertexPartition, n_iterations=-1, seed=42)
ari = metrics.adjusted_rand_score(self.meta_data.iloc[:, 1], np.array(partition._membership))
