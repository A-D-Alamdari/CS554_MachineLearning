import os
import csv
import random
import math
import numpy as np
import matplotlib.pyplot as plt

# Ensure figures directory exists
os.makedirs("figures", exist_ok=True)


def load_fashion_mnist_csv(ignore_header=True, ignore_labels=True):
    """
        Loads the Fashion-MNIST CSV file as a list of lists, optionally ignoring the header row and the first column of labels.

        Args:
            ignore_header: Whether to ignore the header row.
            ignore_labels: Whether to ignore the first column of labels.

        Returns:
            A list of lists, where each inner list contains the pixel values of a single image.
    """

    data = []
    with open('data/fashion_mnist.csv', 'r') as f:
        reader = csv.reader(f)

        # Skip the header row, if necessary.
        if ignore_header:
            next(reader)

        for row in reader:
            image_data = []

            # Skip the first column of labels, if necessary.
            if ignore_labels:
                row = row[1:]

            for pixel_value in row:
                image_data.append(int(pixel_value))
            data.append(image_data)
    return data


def initialize_centroids(data, k):
    """
        Initialize k cluster centroids randomly within the range of the data.

        Parameters:
            - data: List of data points
            - k: Number of clusters

        Returns:
            - centroids: List of k cluster centroids
    """
    # Transpose the data to get dimensions
    dimensions = len(data[0])

    # Initialize centroids as lists of random values within the range of each dimension
    centroids = [
        [random.uniform(min(point[j] for point in data), max(point[j] for point in data)) for j in range(dimensions)]
        for _ in range(k)]

    return centroids


def calculate_distance(point1, point2):
    """
        Calculate the Euclidean distance between two points.

        Parameters:
            - point1: List representing the coordinates of the first point
            - point2: List representing the coordinates of the second point

        Returns:
            - distance: Euclidean distance between the two points
    """
    # Ensure both points have the same number of dimensions
    assert len(point1) == len(point2), "Points must have the same number of dimensions"

    # Calculate Euclidean distance
    distance = math.sqrt(sum((point1[i] - point2[i]) ** 2 for i in range(len(point1))))

    return distance


def assign_to_clusters(data, centroids):
    """
        Assign each data point to the nearest cluster centroid.

        Parameters:
            - data: List of data points
            - centroids: List of cluster centroids

        Returns:
            - clusters: List where each element represents the index of the assigned cluster for the
                        corresponding data point
    """
    # Initialize clusters
    clusters = []

    # Iterate over each data point
    for point in data:
        # Calculate distances to all centroids
        distances = [calculate_distance(point, centroid) for centroid in centroids]

        # Find the index of the centroid with the minimum distance
        nearest_centroid_index = min(range(len(distances)), key=distances.__getitem__)

        # Assign the data point to the nearest cluster
        clusters.append(nearest_centroid_index)

    return clusters


def update_centroids(data, clusters, k):
    """
        Recalculate the centroids of each cluster by taking the mean of all the data points assigned to that cluster.

        Parameters:
            - data: List of data points
            - clusters: List where each element represents the index of the assigned cluster for the
                        corresponding data point
            - k: Number of clusters

        Returns:
            - new_centroids: List of updated cluster centroids
    """
    # Initialize new centroids
    new_centroids = []

    # Iterate over each cluster
    for cluster_index in range(k):
        # Extract data points assigned to the current cluster
        cluster_points = [data[i] for i in range(len(data)) if clusters[i] == cluster_index]

        # If the cluster is not empty, calculate the mean as the new centroid
        if cluster_points:
            new_centroid = [sum(coord) / len(cluster_points) for coord in zip(*cluster_points)]
        else:
            # If the cluster is empty, keep the centroid unchanged
            new_centroid = [0] * len(data[0])

        # Append the new centroid to the list
        new_centroids.append(new_centroid)

    return new_centroids


def has_converged(prev_centroids, new_centroids, threshold=1e-4):
    """
        Check whether the centroids have changed significantly.

        Parameters:
            - prev_centroids: List of centroids from the previous iteration
            - new_centroids: List of centroids from the current iteration
            - threshold: Convergence threshold

        Returns:
            - converged: True if the centroids have converged, False otherwise
    """
    # Check the Euclidean distance between corresponding centroids
    centroid_distances = [calculate_distance(prev, new) for prev, new in zip(prev_centroids, new_centroids)]

    # If all centroid distances are below the threshold, the algorithm has converged
    converged = all(distance < threshold for distance in centroid_distances)

    return converged


def k_means(data, k, max_iterations=15):
    """
        Perform k-means clustering on the given data.

        Parameters:
            - data: Numpy array of data points
            - k: Number of clusters
            - max_iterations: Maximum number of iterations

        Returns:
            - centroids: List of final cluster centroids
            - loss_history: List of reconstruction loss for each iteration
    """
    centroids = initialize_centroids(data, k)
    loss_history = []

    for iteration in range(max_iterations):
        prev_centroids = centroids.copy()

        # Assign data points to clusters
        clusters = assign_to_clusters(data, centroids)

        # Update centroids based on assigned clusters
        centroids = update_centroids(data, clusters, k)

        # Calculate and record the reconstruction loss
        loss = sum(calculate_distance(data[i], centroids[clusters[i]]) for i in range(len(data)))
        loss_history.append(loss)

        # Check for convergence
        if iteration > 0 and has_converged(prev_centroids, centroids):
            break

    return centroids, loss_history


def plot_loss(iterations, loss_history, k):
    """
        Plot the reconstruction loss as a function of iterations.

        Parameters:
            - iterations: List of iteration numbers
            - loss_history: List of reconstruction loss values
    """

    plt.plot(iterations, loss_history, marker='o')
    plt.title('Reconstruction Loss over Iterations')
    plt.xlabel('Iterations')
    plt.ylabel('Reconstruction Loss')
    plt.savefig("figures/loss_plot.png")
    plt.close()


def plot_centroids(centroids):
    """
        Plot the cluster centroids as 28x28 grayscale images.

        Parameters:
            - centroids: List of cluster centroids (NumPy arrays)
    """
    num_centroids = len(centroids)

    # Create a single plot with subplots for each centroid
    plt.figure(figsize=(num_centroids * 2, 2))

    for i in range(num_centroids):
        plt.subplot(1, num_centroids, i + 1)

        # Reshape the centroid into a 28x28 array
        centroid_image = centroids[i].reshape((28, 28))

        # Plot the image
        for row in range(28):
            for col in range(28):
                pixel_value = centroid_image[row, col]
                plt.plot(col, 28 - row, color=str(pixel_value / 255.0), marker='s', markersize=8)

        plt.title(f'Centroid {i + 1}')
        plt.axis('off')

    plt.savefig("figures/centroid_plot.png")
    plt.close()


def plot_centroids(centroids, k):
    """
        Plot the cluster centroids as 28x28 grayscale images.

        Parameters:
            - centroids: List of cluster centroids (NumPy arrays)
            - k: Number of clusters
    """
    num_rows = k // 10  # Calculate the number of rows based on k
    num_columns = 10    # Fixed number of columns

    # Create a single plot with subplots for each centroid
    plt.figure(figsize=(num_columns * 2, num_rows * 2))

    for i in range(k):
        plt.subplot(num_rows, num_columns, i + 1)

        # Reshape the centroid into a 28x28 array
        centroid_image = centroids[i].reshape((28, 28))

        # Plot the image
        plt.imshow(centroid_image, cmap='gray')
        plt.title(f'Centroid {i + 1}')
        plt.axis('off')

    plt.savefig("figures/data_plot_k_clusters.png")
    plt.close()


def plot_loss_all_k(iterations, loss_history_list, k_values):
    """
        Plot the reconstruction loss as a function of iterations for different k values in the same plot.

        Parameters:
            - iterations: List of iteration numbers
            - loss_history_list: List of lists, each containing reconstruction loss values for a specific k
            - k_values: List of k values
    """
    plt.figure(figsize=(10, 6))

    for i, loss_history in enumerate(loss_history_list):
        plt.plot(iterations, loss_history, label=f'k = {k_values[i]}', marker='o')

    plt.title('Reconstruction Loss over Iterations for Different k Values')
    plt.xlabel('Iterations')
    plt.ylabel('Reconstruction Loss')
    plt.legend()
    plt.savefig("figures/loss_plot_all.png")
    plt.close()


data = load_fashion_mnist_csv(ignore_header=True, ignore_labels=True)

# Use this function to plot the reconstruction loss for different k values
k_values = [10, 20, 30]
loss_history_list = []

for k in k_values:
    centroids, loss_history = k_means(data, k)
    loss_history_list.append(loss_history)
    plot_centroids(np.array(centroids), k)

plot_loss_all_k(range(1, len(loss_history_list[0]) + 1), loss_history_list, k_values)

#
# data = load_fashion_mnist_csv(ignore_header=True, ignore_labels=True)
#
# k_values = [10, 20, 30]
#
# for k in k_values:
#     centroids, loss_history = k_means(data, k)
#     plot_loss(range(1, len(loss_history) + 1), loss_history)
#     plot_centroids(np.array(centroids))


# centroids, loss_history = k_means(data, 10)
# plot_loss(range(1, len(loss_history) + 1), loss_history)