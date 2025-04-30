# CS554 Spring 2025 - Homework 1: Polynomial Regression (Description)

## Overview

This repository contains the solution for Homework 1 of CS554 (Spring 2025), focused on polynomial regression. The task is to fit polynomials of varying degrees to a dataset and evaluate model performance.

## Dataset

Two CSV files are provided:
- `train.csv`: Training data (each row contains input `x` and output `r`)
- `test.csv`: Testing data (same format)

Each file includes two columns:
1. Input feature `x`
2. Target value `r`

## Task Description

1. **Fit Polynomial Models**  
   - Fit polynomials of degrees **0 through 7** using the training data.
   - For each model, plot the fitted polynomial alongside the training data.
   - Generate separate plots for each degree.

2. **Evaluate Performance**  
   - Compute the **Sum of Squared Errors (SSE)** for each model on both the training and test sets.
   - Plot SSE as a function of polynomial degree for both datasets.

## Tools and Libraries

- Python 3.x
- NumPy
- Matplotlib (for plotting)
- (Optional) scikit-learn or other libraries for polynomial fitting

## Output Requirements

Your submission must include:
- A **report** (`report.pdf`) summarizing:
  - Your approach
  - All generated plots
  - Key observations and findings
- Your **source code** (`.py` files), including:
  - Code for fitting models
  - Code for plotting results
  - Code for computing SSE

**Do not compress your submission. Submit all files individually.**

## Usage

To run the code:

```bash
python polynomial_regression.py
