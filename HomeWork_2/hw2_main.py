import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Create 'figures' directory if it doesn't exist
os.makedirs("figures", exist_ok=True)


# ----------------------------- k-means functions ------------------------------
def load_data(file_path):
    data = pd.read_csv(file_path)
    return data.values


def initialize_centroids(data, k):
    indices = np.random.choice(len(data), size=k, replace=False)
    return data[indices]


def assign_clusters(data, centroids):
    distances = np.linalg.norm(data[:, np.newaxis] - centroids, axis=2)
    return np.argmin(distances, axis=1)


def update_centroids(data, labels, k):
    new_centroids = []
    for i in range(k):
        cluster_points = data[labels == i]
        if len(cluster_points) > 0:
            new_centroids.append(np.mean(cluster_points, axis=0))
        else:
            new_centroids.append(data[np.random.randint(0, len(data))])
    return np.array(new_centroids)


def compute_loss(data, centroids, labels):
    loss = 0.0
    for i, point in enumerate(data):
        loss += np.sum((point - centroids[labels[i]]) ** 2)
    return loss


def run_kmeans(data, k, max_iters=100):
    centroids = initialize_centroids(data, k)
    for _ in range(max_iters):
        labels = assign_clusters(data, centroids)
        new_centroids = update_centroids(data, labels, k)
        if np.allclose(new_centroids, centroids):
            break
        centroids = new_centroids
    loss = compute_loss(data, centroids, labels)
    return labels, centroids, loss


# --------------------------------- main loop ----------------------------------
data = load_data("data/data.csv")
k_values = range(1, 7)
mean_losses = []
best_clusterings = {}

for k in k_values:
    trial_losses = []
    best_loss = float("inf")
    best_labels = None
    best_centroids = None

    for _ in range(10):
        labels, centroids, loss = run_kmeans(data, k)
        trial_losses.append(loss)
        if loss < best_loss:
            best_loss = loss
            best_labels = labels
            best_centroids = centroids

    mean_losses.append(np.mean(trial_losses))
    best_clusterings[k] = (best_labels, best_centroids)

# -------------------------------- plot loss -----------------------------------
plt.figure(figsize=(8, 5))
plt.plot(k_values, mean_losses, marker='o')
plt.title("Mean Reconstruction Loss vs k")
plt.xlabel("Number of Clusters (k)")
plt.ylabel("Mean Reconstruction Loss")
plt.grid(True)
plt.savefig("figures/mean_loss_plot.png")
plt.close()

# --------------------------- plot best clustering -----------------------------
for k in k_values:
    labels, centroids = best_clusterings[k]
    plt.figure(figsize=(6, 5))
    for i in range(k):
        cluster_points = data[labels == i]
        plt.scatter(cluster_points[:, 0], cluster_points[:, 1], label=f"Cluster {i + 1}")
    plt.scatter(centroids[:, 0], centroids[:, 1], color='black', marker='x', s=100, label="Centroids")
    plt.title(f"Best Clustering for k = {k}")
    plt.legend()
    plt.savefig(f"figures/best_clustering_k{k}.png")
    plt.close()
