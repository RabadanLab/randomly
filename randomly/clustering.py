"""Implementations of several clustering
algorithms based on scikit-learn library
"""

from sklearn.cluster import KMeans
from scipy.cluster import hierarchy

class Cluster():
    '''Attributes
       ----------

       X: array-like or sparse matrix, shape=(n_cells, n_genes)
          Training instances to clustering

       labels:
       Labels for each data point
    '''
    
    def __init__(self):
        self.X = None

    def fit_kmeans(self, n_clusters=2, random_state=1):
        if self.X_vis is None:
            raise ValueError('Nothing to cluster, please fit tsne first')
        kmeans_model = KMeans(n_clusters=n_clusters,
                              random_state=1).fit(self.X_vis)
        self.labels_kmeans = kmeans_model.labels_

    def fit_hierarchical(self,method='ward'):
        if self.X_vis is None:
            raise ValueError('Nothing to cluster, please fit tsne first')
        self.h2 = hierarchy.linkage(self.X_vis, method=method)
        self.h1 = hierarchy.linkage(self.X_vis.T, method=method)
    
    
    
    
    
