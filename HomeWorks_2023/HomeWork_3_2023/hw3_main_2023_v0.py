import os
import csv
import math
import random
import numpy as np
import matplotlib.pyplot as plt

# Ensure figures directory exists
os.makedirs("figures", exist_ok=True)


# Loading data
def load_csv(file_path, ignore_header=True, ignore_labels=True):
    """Loads a custom CSV file as a list of lists, optionally ignoring the header row and the first column of labels.

    Args:
        file_path (str): The path to the CSV file.
        ignore_header (bool): Whether to ignore the header row.
        ignore_labels (bool): Whether to ignore the first column of labels.

    Returns:
        list: A list of lists, where each inner list contains the feature values.
    """
    data = []
    with open(file_path, 'r') as f:
        reader = csv.reader(f)

        # Skip the header row, if necessary.
        if ignore_header:
            next(reader)

        for row in reader:
            feature_values = []

            # Skip the first column of labels, if necessary.
            if ignore_labels:
                row = row[:]

            for value in row:
                feature_values.append(float(value))
            data.append(feature_values)
    return data


train_data = load_csv("data/train.csv")
test_data = load_csv("data/test.csv")

X_train = []
y_train = []
for i in range(len(train_data)):
    inp = []
    for j in range(len(train_data[i]) - 1):
        inp.append(train_data[i][j])
    y_train.append(train_data[i][-1])
    X_train.append(inp)

X_test = []
y_test = []
for i in range(len(test_data)):
    inp = []
    for j in range(len(test_data[i]) - 1):
        inp.append(test_data[i][j])
    y_test.append(test_data[i][-1])
    X_test.append(inp)


class MLP:
    def __init__(self, input_size, hidden_units):
        self.input_size = input_size
        self.hidden_units = hidden_units
        self.weights_hidden = [random.uniform(-1, 1) for _ in range(input_size * hidden_units)]
        self.bias_hidden = [random.uniform(-1, 1) for _ in range(hidden_units)]
        self.weights_output = [random.uniform(-1, 1) for _ in range(hidden_units)]
        self.bias_output = random.uniform(-1, 1)

    def sigmoid(self, x):
        return 1 / (1 + math.exp(-x))

    def predict(self, x):
        hidden_output = [self.sigmoid(self.bias_hidden[i] + sum(
            self.weights_hidden[i * self.input_size + j] * x[j] for j in range(self.input_size))) for i in
                         range(self.hidden_units)]
        output = self.sigmoid(
            self.bias_output + sum(self.weights_output[i] * hidden_output[i] for i in range(self.hidden_units)))
        return output

    def binary_cross_entropy(self, label, output):
        epsilon = 1e-15
        output = max(epsilon, min(1 - epsilon, output))
        return -(label * math.log(output) + (1 - label) * math.log(1 - output))

    def train(self, train_data, learning_rate, epochs):
        errors = []

        for epoch in range(epochs):
            total_error = 0

            for data_point in train_data:
                x = data_point[:self.input_size]
                label = data_point[self.input_size]

                # Forward pass
                hidden_output = [self.sigmoid(self.bias_hidden[i] + sum(
                    self.weights_hidden[i * self.input_size + j] * x[j] for j in range(self.input_size))) for i in
                                 range(self.hidden_units)]
                output = self.sigmoid(
                    self.bias_output + sum(self.weights_output[i] * hidden_output[i] for i in range(self.hidden_units)))

                # Calculate binary cross-entropy
                error = self.binary_cross_entropy(label, output)
                total_error += error

                # Backward pass
                delta_output = label - output
                delta_hidden = [hidden_output[i] * (1 - hidden_output[i]) * self.weights_output[i] * delta_output for i
                                in range(self.hidden_units)]

                # Update weights and bias
                for i in range(self.hidden_units):
                    for j in range(self.input_size):
                        self.weights_hidden[i * self.input_size + j] += learning_rate * delta_hidden[i] * x[j]
                    self.bias_hidden[i] += learning_rate * delta_hidden[i]

                for i in range(self.hidden_units):
                    self.weights_output[i] += learning_rate * delta_output * hidden_output[i]
                self.bias_output += learning_rate * delta_output

            errors.append(total_error)

        return errors

    def plot_output_contour(self, data):
        x1_values = [point[0] for point in data]
        x2_values = [point[1] for point in data]

        x1_range = (min(x1_values) - 1, max(x1_values) + 1)
        x2_range = (min(x2_values) - 1, max(x2_values) + 1)

        xx, yy = self.generate_meshgrid(x1_range, x2_range, 0.1)

        predictions = []
        for i in range(len(xx)):
            row_preds = []
            for j in range(len(xx[0])):
                x = [xx[i][j], yy[i][j]]
                row_preds.append(self.predict(x))
            predictions.append(row_preds)

        plt.figure(figsize=(8, 6))
        plt.contourf(xx, yy, predictions, levels=50, cmap='viridis', alpha=0.6)
        plt.scatter(x1_values, x2_values, c=[point[2] for point in data], cmap='viridis', edgecolors='k')
        plt.title('Network Output Contour Plot')
        plt.xlabel('X1')
        plt.ylabel('X2')
        plt.savefig("figures/output_contour_plot.png")
        plt.close()

    def plot_learning_epochs(self, errors):
        plt.figure(figsize=(8, 6))
        plt.plot(range(1, len(errors) + 1), errors)
        plt.title('Learning Epochs vs Error')
        plt.xlabel('Epochs')
        plt.ylabel('Error')
        plt.savefig("figures/learning_epochs_plot.png")
        plt.close()

    def generate_meshgrid(self, x1_range, x2_range, step):
        x1_values = np.arange(x1_range[0], x1_range[1], step)
        x2_values = np.arange(x2_range[0], x2_range[1], step)
        xx, yy = np.meshgrid(x1_values, x2_values)
        return xx, yy

    def plot_network_complexity(self, train_data, test_data, hidden_units_range):
        train_errors = []
        test_errors = []

        for hidden_units in hidden_units_range:
            if hidden_units == 0:
                model = MLP(self.input_size, 0)
            else:
                model = MLP(self.input_size, hidden_units)

            train_errors.append(self.calculate_total_error(train_data, model))
            test_errors.append(self.calculate_total_error(test_data, model))

        plt.figure(figsize=(8, 6))
        plt.plot(hidden_units_range, train_errors, label='Training Error')
        plt.plot(hidden_units_range, test_errors, label='Test Error')
        plt.title('Network Complexity vs Error')
        plt.xlabel('Number of Hidden Units')
        plt.ylabel('Binary Cross-Entropy Error')
        plt.legend()
        plt.savefig("figures/network_complexity.png")
        plt.close()

    def calculate_total_error(self, data, model):
        total_error = 0
        for data_point in data:
            x = data_point[:self.input_size]
            label = data_point[self.input_size]
            output = model.predict(x)
            total_error += self.binary_cross_entropy(label, output)
        return total_error


# Example usage:
input_size = 2
learning_rate = 0.1
epochs = 1000

# Assuming you have your training data in 'train_data'
# Load train_data from your dataset
# train_data = ...

# Single-layer perceptron
slp = MLP(input_size, 0)
errors_slp = slp.train(train_data, learning_rate, epochs)
slp.plot_output_contour(train_data)
slp.plot_learning_epochs(errors_slp)

# Multi-layer perceptron with one hidden layer of 2 hidden units
mlp_2_units = MLP(input_size, 2)
errors_mlp_2_units = mlp_2_units.train(train_data, learning_rate, epochs)
mlp_2_units.plot_output_contour(train_data)
mlp_2_units.plot_learning_epochs(errors_mlp_2_units)

# Multi-layer perceptron with one hidden layer of 4 hidden units
mlp_4_units = MLP(input_size, 4)
errors_mlp_4_units = mlp_4_units.train(train_data, learning_rate, epochs)
mlp_4_units.plot_output_contour(train_data)
mlp_4_units.plot_learning_epochs(errors_mlp_4_units)

# Multi-layer perceptron with one hidden layer of 8 hidden units
mlp_8_units = MLP(input_size, 8)
errors_mlp_8_units = mlp_8_units.train(train_data, learning_rate, epochs)
mlp_8_units.plot_output_contour(train_data)
mlp_8_units.plot_learning_epochs(errors_mlp_8_units)


def calculate_accuracy(model, test_data):
    correct_predictions = 0

    for data_point in test_data:
        x = data_point[:model.input_size]
        label = data_point[model.input_size]
        predicted_output = model.predict(x)

        # Assuming a threshold of 0.5 for binary classification
        predicted_label = 1 if predicted_output >= 0.5 else 0

        if predicted_label == label:
            correct_predictions += 1

    accuracy = correct_predictions / len(test_data)
    return accuracy


# Calculate accuracy for each model
accuracy_slp = calculate_accuracy(slp, test_data)
accuracy_mlp_2_units = calculate_accuracy(mlp_2_units, test_data)
accuracy_mlp_4_units = calculate_accuracy(mlp_4_units, test_data)
accuracy_mlp_8_units = calculate_accuracy(mlp_8_units, test_data)

print("Accuracy - Single-layer Perceptron:", accuracy_slp)
print("Accuracy - MLP (2 Hidden Units):", accuracy_mlp_2_units)
print("Accuracy - MLP (4 Hidden Units):", accuracy_mlp_4_units)
print("Accuracy - MLP (8 Hidden Units):", accuracy_mlp_8_units)

hidden_units_range = [0, 2, 4, 8]
slp.plot_network_complexity(train_data, test_data, hidden_units_range)
