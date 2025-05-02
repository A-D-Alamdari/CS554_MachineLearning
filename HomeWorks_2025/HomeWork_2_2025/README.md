# CS554 Spring 2025 - Homework 2: k-Means Clustering for Unsupervised Learning (Description)

## Overview

This repository contains the solution for Homework 2 of CS554 (Spring 2025), which involves implementing the **k-means clustering algorithm** from scratch and analyzing clustering performance on a 2D dataset.

## Dataset

You are provided with:
- `data.csv`: a 2D dataset to be used for clustering

## Task Description

1. **Clustering Implementation**  
   - Implement the **k-means algorithm** (no library implementations allowed).
   - Use values of **k = 1 through 6**.
   - For each `k`, run the algorithm **10 times** and record the reconstruction (clustering) loss.

2. **Visualization and Analysis**  
   - Plot the **mean reconstruction loss vs. k** on a single graph.
   - For each `k`, visualize the **best clustering result** (with the least loss) using distinct colors/symbols for each cluster.

## Allowed Libraries

- `pandas`
- `numpy`
- `matplotlib`

> ❗ **Do not use any external libraries for k-means implementation** (e.g., `sklearn.cluster.KMeans` is not allowed).


