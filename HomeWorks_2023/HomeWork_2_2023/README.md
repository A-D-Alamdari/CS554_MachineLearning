# CS454/554 Homework 2 (2023): k-Means Clustering for Unsupervised Learning (Description)

## Overview

In this assignment, you will implement the **k-means clustering algorithm** from scratch to discover cluster centroids on the **Fashion-MNIST** dataset. The class labels will be ignored, making this a fully unsupervised task.

## Dataset

- **File**: `fashion_mnist.csv` (extracted from `fashion_mnist.zip`)
- **Format**: 60,000 instances
  - **First column**: Label (to be ignored)
  - **Remaining 784 columns**: Flattened 28×28 grayscale image pixel values

> 📦 **Download Dataset**
>
> 🧪 [Download `fashion_mnist.csv`](https://drive.google.com/drive/folders/1VgTbWu6WCLZoA-On0H1_l3ujGeoWAlt5?usp=sharing)
> 
> Use this file to implement the **k-means clustering algorithm**.




## Task

For **k ∈ {10, 20, 30}**, perform the following:

1. Implement the **k-means algorithm**:
   - Initialize centroids randomly
   - Assign instances to the nearest centroid
   - Recompute centroids

2. For each `k`, produce:
   - A **plot of reconstruction loss vs. iteration**
   - A **visualization of the final centroids** as 28×28 grayscale images

## Constraints

- You **may use Pandas** to load the data and **NumPy arrays** for storage
- You **must not use** any library functions for statistical or numerical calculations:
  - ❌ `np.sum`, `np.mean`, `np.linalg.norm`, `dataframe.mean()` etc.
  - ✔️ Manually compute distances, centroids, and updates

