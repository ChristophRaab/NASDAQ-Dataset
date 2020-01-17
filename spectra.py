
import matplotlib.pyplot as plt
import numpy as np
from sklearn import cluster, datasets, mixture
from sklearn.cluster import k_means,KMeans
from sklearn.neighbors import NearestNeighbors,KNeighborsClassifier

def spectral_clustering(X,gamma,n_cluster):
    X_norm = np.sum(X ** 2, axis=-1)

    A = np.exp(-2 * (X_norm[:, None] + X_norm[None, :] - 2 * np.dot(X, X.T)))
    D = np.diag(np.sum(A,axis=1))
    L = D*-0.5 @ A @ D*-0.5

    _,v = np.linalg.eig(L)
    v = np.real(v)
    k = v[:,:n_cluster]

    row_sums = k.sum(axis=1)
    k = k / k.sum(axis=1)[:, None]

    clf = KMeans(n_clusters=n_clusters)
    clf.fit(X)
    idx = clf.predict(X)
    return idx

n_samples = 1000
centers = 4
n_clusters = 2
X, y = datasets.make_blobs(n_samples=n_samples,centers=centers)
X,y = datasets.make_circles(n_samples=n_samples, factor=.5,
                                      noise=.05)
idx = spectral_clustering(X,2,2)

plt.scatter(X[:, 0], X[:, 1], s=10, c=idx)
plt.show()

spectral = cluster.SpectralClustering(
    n_clusters=n_clusters, eigen_solver='arpack',
    affinity="nearest_neighbors")

idx = spectral.fit_predict(X)
plt.scatter(X[:, 0], X[:, 1], s=10, c=idx)
plt.show()