import os
import math
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Ensure figures directory exists
os.makedirs("figures", exist_ok=True)


# Load data
def load_data():
    train_df = pd.read_csv("data/training.csv", header=None, names=["x", "class"])
    test_df = pd.read_csv("data/testing.csv", header=None, names=["x", "class"])
    return train_df, test_df


# Plot training data
def plot_training_data(class_data):
    plt.figure(figsize=(10, 4))
    for c in class_data:
        plt.scatter(class_data[c], [c] * len(class_data[c]), label=f"Training Class {c}", marker='x')
    plt.xlabel("Value of Data Points")
    plt.ylabel("Class")
    plt.legend()
    plt.yticks(sorted(class_data.keys()))
    plt.savefig("figures/data_plot.png")
    plt.close()


# Plot likelihoods
def plot_likelihoods(x_values, likelihoods):
    plt.figure(figsize=(10, 6))
    for c in likelihoods:
        plt.plot(x_values, likelihoods[c], label=f"Likelihood Class {c}")
    plt.xlabel("Value of Data Points")
    plt.ylabel("Likelihood")
    plt.legend()
    plt.savefig("figures/likelihood_plot.png")
    plt.close()


# Plot posteriors
def plot_posteriors(x_values, posteriors):
    plt.figure(figsize=(10, 6))
    for c in posteriors:
        plt.plot(x_values, posteriors[c], linestyle="--", label=f"Posterior Class {c}")
    plt.xlabel("Value of Data Points")
    plt.ylabel("Posterior")
    plt.legend()
    plt.savefig("figures/posterior_plot.png")
    plt.close()


# Combined plot
def plot_combined(x_values, likelihoods, posteriors, class_data, X_test):
    plt.figure(figsize=(10, 6))
    for c in likelihoods:
        plt.plot(x_values, likelihoods[c], label=f"Likelihood Class {c}")
        plt.plot(x_values, posteriors[c], linestyle="--", label=f"Posterior Class {c}")
    for c in class_data:
        plt.scatter(class_data[c], [0.1 * c] * len(class_data[c]), marker='+', label=f"Train Class {c}")
    plt.scatter(X_test, [0.8] * len(X_test), color='black', marker='x', label='Test Data')
    plt.xlabel("Value of Data Points")
    plt.ylabel("Probability Density")
    plt.legend()
    plt.savefig("figures/part1_plot.png")
    plt.close()


# Likelihood function
def likelihood(x, mean, var):
    return (1 / math.sqrt(2 * math.pi * var)) * math.exp(-((x - mean) ** 2) / (2 * var))


# Prediction function
def predict(x, priors, means, variances):
    numerators = {c: priors[c] * likelihood(x, means[c], variances[c]) for c in priors}
    denominator = sum(numerators.values())
    posteriors = {c: numerators[c] / denominator for c in priors}
    return max(posteriors, key=posteriors.get)


# Confusion matrix function
def compute_confusion_matrix(X, y_true, priors, means, variances):
    classes = sorted(priors.keys())
    y_pred = [predict(x, priors, means, variances) for x in X]
    matrix = np.zeros((len(classes), len(classes)), dtype=int)
    for true, pred in zip(y_true, y_pred):
        matrix[true - 1][pred - 1] += 1
    return matrix


# Main logic
def main():
    train_df, test_df = load_data()
    X_train = train_df["x"].values
    y_train = train_df["class"].values
    X_test = test_df["x"].values
    y_test = test_df["class"].values

    classes = sorted(np.unique(y_train))
    class_data = {c: X_train[y_train == c] for c in classes}

    plot_training_data(class_data)

    priors = {c: len(class_data[c]) / len(X_train) for c in classes}
    means = {c: np.mean(class_data[c]) for c in classes}
    variances = {c: np.var(class_data[c]) for c in classes}

    for c in classes:
        print(f"Class {c} => Prior: {priors[c]:.4f}, Mean: {means[c]:.4f}, Variance: {variances[c]:.4f}")
    print("\n" + "*" * 80 + "\n")

    x_values = np.sort(X_train)
    likelihoods = {
        c: [likelihood(x, means[c], variances[c]) for x in x_values] for c in classes
    }
    plot_likelihoods(x_values, likelihoods)

    total_likelihood = np.sum([np.array(likelihoods[c]) * priors[c] for c in classes], axis=0)
    posteriors = {
        c: (np.array(likelihoods[c]) * priors[c]) / total_likelihood for c in classes
    }
    plot_posteriors(x_values, posteriors)
    plot_combined(x_values, likelihoods, posteriors, class_data, X_test)

    train_cm = compute_confusion_matrix(X_train, y_train, priors, means, variances)
    test_cm = compute_confusion_matrix(X_test, y_test, priors, means, variances)

    print("Confusion Matrix for Training Data:")
    print(train_cm)
    print("\nConfusion Matrix for Testing Data:")
    print(test_cm)


if __name__ == "__main__":
    main()
