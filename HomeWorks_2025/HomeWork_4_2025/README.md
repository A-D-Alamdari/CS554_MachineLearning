# CS454/554 Homework 4: KMNIST Classification with PyTorch

## 📌 Overview

In this assignment, you will implement and evaluate three different neural network architectures for classifying the [KMNIST dataset](https://github.com/rois-codh/kmnist) using **PyTorch**. Your goal is to compare the performance of models with varying levels of complexity.

---

> 📦 **Dataset**
>
> Load the KMNIST dataset using:  
> `torchvision.datasets.KMNIST`  
> - Contains 10 classes of Japanese characters  
> - Each image is 28×28 grayscale

---

## 🧠 Architectures to Implement

1. **Linear Model**  
   - A single fully connected layer

2. **MLP Model**  
   - One hidden layer with 40 neurons

3. **CNN Model**  
   - Must include **at least one convolutional layer** and **one fully connected layer**  
   - You may experiment and report the CNN architecture that performs best

---

## 📝 Your Report Must Include:

- 📐 **Architecture Details**  
  - Layers, activation functions, and dimensions of each model

- 📉 **Training/Test Loss Plots**  
  - Show how the loss evolves over epochs for each model

- 📊 **Training/Test Accuracy Plots**  
  - Track how accuracy changes during training and testing

---

## 🛠 Tools

- Programming Language: **Python**  
- Framework: **PyTorch**  
- Dataset Loader: `torchvision.datasets.KMNIST`  
