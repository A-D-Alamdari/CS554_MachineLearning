# CS454/554 Homework 3: Single and Multi-Layer Perceptrons for Regression

## Overview

This homework focuses on implementing single-layer and multi-layer perceptrons for regression tasks. You will build and train neural networks using back-propagation and compare performance across different configurations.

## Dataset

- **Training Data**: `train.csv` (20 instances)
- **Testing Data**: `test.csv` (80 instances)

Each row corresponds to one input instance and its corresponding output.

## Network Configurations

You are required to implement and compare the following architectures:

- **a)** Single-layer perceptron  
- **b)** Multi-layer perceptron with 1 hidden layer (2 hidden units)  
- **c)** Multi-layer perceptron with 1 hidden layer (4 hidden units)  
- **d)** Multi-layer perceptron with 1 hidden layer (8 hidden units)  

## Requirements

- Use **tanh** activation function for hidden layers in multi-layer perceptrons
- Include **bias nodes** in both input and hidden layers
- Implement **back-propagation algorithm** from scratch
- You may use **NumPy** or **Pandas** for computation and data handling
- Do **not** use any machine learning libraries (e.g., PyTorch, Scikit-learn)
- You may choose **online**, **mini-batch**, or **batch** learning
- Tune **learning rate** and **number of epochs** via experimentation


## Example Figures

- Output vs. Training Data (1 plot per model)
- Epochs vs. MSE (1 plot with all models or separate plots)
- Hidden Units vs. Training/Test MSE (with x=0 representing the single-layer perceptron)

