# CS454/554 Homework 1 (2023): Parametric Classification (Description)

## Overview

In this assignment, you will implement a **parametric classification** algorithm under the assumption that class-conditional densities are **Gaussian**. You will estimate class priors and Gaussian parameters from training data, then evaluate classification performance on test data.

## Dataset

- `training.csv`: 150 instances (used to estimate class priors, means, and variances)
- `testing.csv`: 150 instances (used to evaluate model generalization)

Each instance has:
- **First column**: Input value `x` (float)
- **Second column**: Class label `C` (1, 2, or 3)

## Objectives

1. **Estimate parameters**:
   - Class priors \( P(C_k) \)
   - Class-conditional Gaussian parameters \( \mu_k, \sigma_k^2 \)

2. **Plot the following** (on the same graph):
   - Likelihood functions \( P(x | C_k) \)
   - Posterior probabilities \( P(C_k | x) \)
   - Training and test data points (different colors for each class)

3. **Performance Evaluation**:
   - Assume 0/1 loss
   - Compute **3×3 confusion matrices** on both training and test sets

## Constraints

- **Do NOT use any statistical library functions**
- Code can be written in `.py` (Python) or `.m` (MATLAB)
- Use manual implementation for all calculations (e.g., means, variances)

