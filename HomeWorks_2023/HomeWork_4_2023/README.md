# CS454/554 Homework 4 (2023): Convolutional Autoencoders (Description)

## 📌 Overview

In this homework, you will implement and evaluate **Convolutional Autoencoders** to upscale **FashionMNIST** images from **7×7** to **28×28** resolution. You will explore the impact of different architectures on performance and complexity.

---

> 📦 **Dataset**  
> Use the standard [FashionMNIST](https://github.com/zalandoresearch/fashion-mnist) dataset from PyTorch.  
> - Use `torchvision.transforms.Resize((7, 7))` to create the **low-resolution inputs**  
> - The original **28×28 images** serve as the **output targets**

---

## 🎯 Objectives

You will implement **three different convolutional autoencoder architectures**, varying in:
- Number of layers
- Layer sizes
- Connectivity (e.g., kernel sizes, strides)

You are **not required to find the best model**, but to observe the trade-offs between:
- Model complexity (parameter count)
- Reconstruction performance (MSE)

---

## 🛠 Requirements

- Language: **Python**
- Framework: **PyTorch**
- Modify or extend the provided `autoencoder_example.py` template

---

## 📈 Your Report Must Include:

1. 🔧 **Architecture Descriptions**  
   - Brief explanation of each encoder-decoder setup

2. 📊 **Comparison Table**  
   - Architecture name  
   - Number of trainable parameters  
   - Final **training** and **test MSE**

3. 📉 **Training Curves**  
   - Plot of **epochs vs Train/Test MSE** for each architecture

4. 🖼 **Visual Results**  
   - For each class (0–9), pick **one test image** and display:  
     - The 7×7 input  
     - The predicted 28×28 output  
     - The ground truth 28×28 output

