import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# This data is generated by ChatGPT(model: GTP-4o)
data = pd.read_csv('data.csv')

X = data[['Repos on GitHub (K)', 'Learning Difficulty (1-5)']].values

np.random.seed(1)
k = 3

def calculate_distance(a, b):
    return np.sqrt(np.sum((a - b) ** 2, axis=1))

def k_means(X, k, max_iter=100):
    m = X.shape[0]
    centroids = X[np.random.choice(m, k, replace=False)]
    labels = np.zeros(m)
    for iteration in range(max_iter):
        clusters = [[] for _ in range(k)]

        for i, x in enumerate(X):
            distances = calculate_distance(x, centroids)
            cluster_idx = np.argmin(distances)
            clusters[cluster_idx].append(x)
            labels[i] = cluster_idx

        new_centroids = np.array(
            [np.mean(cluster, axis=0) if cluster else centroids[i] for i, cluster in enumerate(clusters)])

        if np.all(centroids == new_centroids):
            break
        centroids = new_centroids

    return centroids, labels

centroids, labels = k_means(X, k)

plt.figure(figsize=(8, 6))
colors = ['r', 'g', 'b']

for i in range(k):
    cluster_points = X[labels == i]
    plt.scatter(cluster_points[:, 0], cluster_points[:, 1], s=50, color=colors[i])

plt.scatter(centroids[:, 0], centroids[:, 1], s=100, c='black', marker='X', label='Centroids')
plt.title('K-means Clustering')
plt.xlabel('Repos on GitHub (K)')
plt.ylabel('Learning Difficulty (1-5)')
plt.show()
