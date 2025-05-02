import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Create 'figures' directory if it doesn't exist
os.makedirs("figures", exist_ok=True)

# Load data
train_df = pd.read_csv("data/train.csv")
test_df = pd.read_csv("data/test.csv")

# Extract features (X) and target values (y) from datasets
X_train = train_df["x"].values.reshape(-1, 1)
y_train = train_df["r"].values
X_test = test_df["x"].values.reshape(-1, 1)
y_test = test_df["r"].values

# Define the range of polynomial degrees to fit
degrees = range(8)


def generate_polynomial_features(X, degree):
    X_poly = np.ones((len(X), degree + 1))
    for d in range(1, degree + 1):
        X_poly[:, d] = X[:, 0] ** d
    return X_poly


def compute_weights(X, y):
    X_transpose = X.T
    XTX = np.dot(X_transpose, X)
    XTy = np.dot(X_transpose, y)
    weights = np.linalg.pinv(XTX).dot(XTy)
    return weights


def make_predictions(X, weights):
    return np.dot(X, weights)


def calculate_sse(y_true, y_pred):
    errors = y_true - y_pred
    squared_errors = errors ** 2
    return np.sum(squared_errors)


# Lists to store SSE values for different polynomial degrees
sse_train = []
sse_test = []

for degree in degrees:
    X_train_poly = generate_polynomial_features(X_train, degree)
    X_test_poly = generate_polynomial_features(X_test, degree)

    weights = compute_weights(X_train_poly, y_train)

    y_train_pred = make_predictions(X_train_poly, weights)
    y_test_pred = make_predictions(X_test_poly, weights)

    train_sse = calculate_sse(y_train, y_train_pred)
    test_sse = calculate_sse(y_test, y_test_pred)

    sse_train.append(train_sse)
    sse_test.append(test_sse)

    X_smooth = np.linspace(min(X_train), max(X_train), 100).reshape(-1, 1)
    X_smooth_poly = generate_polynomial_features(X_smooth, degree)
    y_smooth = make_predictions(X_smooth_poly, weights)

    plt.figure(figsize=(6, 4))
    plt.scatter(X_train, y_train, color="blue", label="Training Data")
    plt.plot(X_smooth, y_smooth, color="red", label=f"Degree {degree} Fit")
    plt.xlabel("x")
    plt.ylabel("r")
    plt.title(f"Polynomial Fit of Degree {degree}")
    plt.legend()
    plt.savefig(f"figures/polynomial_fit_{degree}.png")  # Save figure
    plt.close()  # Close the figure to free memory

# Plot SSE vs Degree
plt.figure(figsize=(6, 4))
plt.plot(degrees, sse_train, marker="o", label="Train SSE", linestyle="-")
plt.plot(degrees, sse_test, marker="s", label="Test SSE", linestyle="--")
plt.xlabel("Polynomial Degree")
plt.ylabel("Sum of Squared Errors (SSE)")
plt.title("SSE vs Polynomial Degree")
plt.legend()
plt.savefig("figures/sse_degree_plot.png")  # Save SSE plot
plt.close()
