import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from typing import List
import random

class ClusteringBase:
    def __init__(self):
        self.labels_ = None
        self.cluster_centers_ = None

    @staticmethod
    def _euclidean_distance(x1: np.ndarray, x2: np.ndarray) -> float:
        return np.sqrt(np.sum((x1 - x2) ** 2, axis=-1))

    @staticmethod
    def _pairwise_distances(X: np.ndarray, Y: np.ndarray = None) -> np.ndarray:
        if Y is None:
            Y = X
        diff = X[:, np.newaxis, :] - Y[np.newaxis, :, :]
        return np.sqrt(np.sum(diff * diff, axis=-1))

    def plot_clusters(self, X: np.ndarray, feature_names: List[str], title: str):
        plt.figure(figsize=(12, 8))
        plt.scatter(X[:, 0], X[:, 1], c=self.labels_, cmap='viridis')

        if self.cluster_centers_ is not None:
            plt.scatter(
                self.cluster_centers_[:, 0],
                self.cluster_centers_[:, 1],
                c='red',
                marker='x',
                s=200,
                linewidths=3,
                label='Centroids'
            )

        plt.xlabel(feature_names[0])
        plt.ylabel(feature_names[1])
        plt.title(title)
        plt.legend()
        plt.show()

class KMeans(ClusteringBase):
    def __init__(self, n_clusters: int, max_iter: int = 300):
        super().__init__()
        self.n_clusters = n_clusters
        self.max_iter = max_iter

    def fit(self, X: np.ndarray) -> None:
        random_indices = random.sample(range(len(X)), self.n_clusters)
        self.cluster_centers_ = X[random_indices].copy()

        for _ in range(self.max_iter):
            old_centroids = self.cluster_centers_.copy()
            distances = self._pairwise_distances(X, self.cluster_centers_)
            self.labels_ = np.argmin(distances, axis=1)

            for k in range(self.n_clusters):
                if sum(k == self.labels_) > 0:
                    self.cluster_centers_[k] = X[self.labels_ == k].mean(axis=0)

            if np.all(np.abs(old_centroids - self.cluster_centers_) < 1e-6):
                break

class KMedoids(ClusteringBase):
    def __init__(self, n_clusters: int, max_iter: int = 300):
        super().__init__()
        self.n_clusters = n_clusters
        self.max_iter = max_iter

    def fit(self, X: np.ndarray) -> None:
        random_indices = random.sample(range(len(X)), self.n_clusters)
        self.cluster_centers_ = X[random_indices].copy()

        for _ in range(self.max_iter):
            old_medoids = self.cluster_centers_.copy()
            distances = self._pairwise_distances(X, self.cluster_centers_)
            self.labels_ = np.argmin(distances, axis=1)

            for k in range(self.n_clusters):
                cluster_points = X[self.labels_ == k]
                if len(cluster_points) > 0:
                    distances = self._pairwise_distances(cluster_points)
                    total_distances = distances.sum(axis=1)
                    medoid_idx = np.argmin(total_distances)
                    self.cluster_centers_[k] = cluster_points[medoid_idx]

            if np.all(np.abs(old_medoids - self.cluster_centers_) < 1e-6):
                break

class DBSCAN(ClusteringBase):
    def __init__(self, eps: float = 0.5, min_samples: int = 5):
        super().__init__()
        self.eps = eps
        self.min_samples = min_samples

    def fit(self, X: np.ndarray) -> None:
        self.labels_ = np.full(len(X), -1)
        cluster_id = 0
        distances_matrix = self._pairwise_distances(X)

        for point_idx in range(len(X)):
            if self.labels_[point_idx] != -1:
                continue

            neighbors = self._find_neighbors(distances_matrix, point_idx)

            if len(neighbors) < self.min_samples:
                self.labels_[point_idx] = -1
                continue

            cluster_id += 1
            self.labels_[point_idx] = cluster_id

            seed_set = neighbors.copy()
            while seed_set:
                current_point = seed_set.pop()
                if self.labels_[current_point] == -1:
                    self.labels_[current_point] = cluster_id
                    current_neighbors = self._find_neighbors(distances_matrix, current_point)
                    if len(current_neighbors) >= self.min_samples:
                        seed_set.update(current_neighbors)

    def _find_neighbors(self, distances_matrix: np.ndarray, point_idx: int) -> set:
        return set(np.where(distances_matrix[point_idx] <= self.eps)[0])

def test_clustering_algorithms():
    df = pd.read_csv('data.csv')
    features = ['Repos on GitHub (K)', 'Average Salary ($K/year)']
    X = df[features].values
    X = (X - X.mean(axis=0)) / X.std(axis=0)

    kmeans = KMeans(n_clusters=5)
    kmeans.fit(X)
    kmeans.plot_clusters(X, features, "K-Means Clustering")

    kmedoids = KMedoids(n_clusters=5)
    kmedoids.fit(X)
    kmedoids.plot_clusters(X, features, "K-Medoids Clustering")

    dbscan = DBSCAN(eps=0.5, min_samples=4)
    dbscan.fit(X)
    dbscan.plot_clusters(X, features, "DBSCAN Clustering")

    original_languages = df['Language']
    for algorithm, labels in zip(['K-Means', 'K-Medoids', 'DBSCAN'],
                               [kmeans.labels_, kmedoids.labels_, dbscan.labels_]):
        print(f"\n{algorithm} Clusters:")
        for cluster in sorted(set(labels)):
            cluster_languages = original_languages[labels == cluster].tolist()
            print(f"Cluster {cluster}: {', '.join(cluster_languages)}")

if __name__ == "__main__":
    test_clustering_algorithms()