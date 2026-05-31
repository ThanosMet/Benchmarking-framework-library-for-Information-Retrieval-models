import numpy as np
from sklearn.manifold import SpectralEmbedding

class SpectralClustering:
    def __init__(self, n_clusters=10, affinity='precomputed', n_neighbors=3, assign_labels='kmeans', n_init=10):
        self.n_clusters = n_clusters
        self.affinity = affinity
        self.n_neighbors = n_neighbors
        self.assign_labels = assign_labels
        self.n_init = n_init
        
    def fit_predict(self, X):
        # Compute affinity matrix
        if self.affinity == 'nearest_neighbors':
            from sklearn.neighbors import NearestNeighbors
            nn = NearestNeighbors(n_neighbors=self.n_neighbors, metric='euclidean')
            nn.fit(X)
            A = nn.kneighbors_graph(X, mode='connectivity').toarray()
            A = np.maximum(A, A.T)  # Make A symmetric

        elif self.affinity == 'rbf':
            from sklearn.metrics.pairwise import rbf_kernel
            A = rbf_kernel(X, gamma=1.0)

        elif self.affinity == 'precomputed':
            A = X

        else:
            raise ValueError("Invalid affinity parameter")


        emb_model = SpectralEmbedding(n_components=self.n_clusters, affinity=self.affinity, n_jobs=-1)
        embeddings = emb_model.fit_transform(A)

        # Cluster using k-means or discretize
        if self.assign_labels == 'kmeans':
            from sklearn.cluster import KMeans

            kmeans = KMeans(n_clusters=self.n_clusters, n_init=self.n_init)
            labels = kmeans.fit_predict(embeddings)

        elif self.assign_labels == 'discretize':
            from sklearn.cluster._spectral import discretize
            labels = discretize(embeddings)


        elif self.assign_labels == 'cluster_qr':
            from sklearn.cluster._spectral import cluster_qr
            labels = cluster_qr(embeddings)
            
        else:
            raise ValueError("Invalid assign_labels parameter")
        
        return labels, embeddings
    


