# CS454/554 Homework 3: Single and Multi-Layer Perceptrons for Binary Classification

## Overview

In this homework, you will implement **single-layer and multi-layer perceptrons** for **binary classification** on 2D input data. You’ll train your models using the **back-propagation algorithm** and visualize results using decision contours and learning curves.

---

## Dataset Description

- 200 instances in `train.csv` and 200 in `test.csv`
- Each instance has:
  - 2 input features (numerical)
  - 1 binary class label (0 or 1)

---

## Tasks

Implement the following networks **from scratch**:

- `a)` Single-layer perceptron  
- `b)` Multi-layer perceptron with 1 hidden layer (2 hidden units)  
- `c)` Multi-layer perceptron with 1 hidden layer (4 hidden units)  
- `d)` Multi-layer perceptron with 1 hidden layer (8 hidden units)

---

## Requirements

- Use **binary cross-entropy** as the loss function
- Train via **back-propagation**
- Include **bias nodes** in both input and hidden layers
- Use **online**, **mini-batch**, or **batch learning**
- **Do not** use machine learning libraries (e.g., PyTorch, Scikit-learn)
- You may use `pandas` and `numpy` for data loading only
- **Do not use** NumPy functions for math operations (e.g., `np.dot`, `np.sum`)

---

## Visualizations

Include the following plots in your report:

1. **Contour plots** of network output overlaid with training data (1 per model)
2. **Epoch vs. binary cross-entropy** (1 combined or 4 separate plots)
3. **Model complexity vs. error** (training/test loss vs. number of hidden units; use `x=0` for the single-layer perceptron)


