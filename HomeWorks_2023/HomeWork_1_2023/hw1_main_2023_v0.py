"""
        CS554 - Homework 1
       Amin Deldari Alamdari
             S033174
            Fall 2023
"""

# Importing required libraries
import matplotlib.pyplot as plt
import math

# Loading Data from the Provided CSV Files:

# Loading training data
training_data = []
with open('data/training.csv', 'r') as file:
    for line in file:
        data_point = line.strip().split(',')
        data_point = [float(data_point[0]), int(data_point[1])]
        training_data.append(data_point)

# Loading testing data
testing_data = []
with open('data/testing.csv', 'r') as file:
    for line in file:
        data_point = line.strip().split(',')
        data_point = [float(data_point[0]), int(data_point[1])]
        testing_data.append(data_point)

# ------------------------------------------------------------------------------------------

# Classify the training data into three classes
class1 = [item[0] for item in training_data if item[1] == 1]
class2 = [item[0] for item in training_data if item[1] == 2]
class3 = [item[0] for item in training_data if item[1] == 3]

# ------------------------------------------------------------------------------------------

# Plotting the data
plt.figure(figsize=(10, 4))

plt.scatter(class1, [1] * len(class1), label='Training Data (Classe 1)', color='red', marker='x')
plt.scatter(class2, [2] * len(class2), label='Training Data (Classe 2)', color='green', marker='x')
plt.scatter(class3, [3] * len(class3), label='Training Data (Classe 3)', color='blue', marker='x')

plt.xlabel('Value of Data Points')
plt.ylabel('Class of Data Points')

plt.legend()
plt.yticks([1, 2, 3], ['1', '2', '3'])  # Set y-axis labels

# Save the plot as an image file (change 'plot_image.png' to your desired file name and extension)
plt.savefig('data_plot.png')

plt.show()

# ------------------------------------------------------------------------------------------

# Calculate class priors
total_instances = len(training_data)
prior_class1 = len(class1) / total_instances
prior_class2 = len(class2) / total_instances
prior_class3 = len(class3) / total_instances

print("Prior Probability of Class 1: ", prior_class1)
print("Prior Probability of Class 2: ", prior_class2)
print("Prior Probability of Class 3: ", prior_class3)
print("\n********************************************************************************\n")

# ------------------------------------------------------------------------------------------

# Calculate means and variances for each class
mean_class1 = sum(class1) / len(class1)
variance_class1 = sum((x - mean_class1) ** 2 for x in class1) / len(class1)

mean_class2 = sum(class2) / len(class2)
variance_class2 = sum((x - mean_class2) ** 2 for x in class2) / len(class2)

mean_class3 = sum(class3) / len(class3)
variance_class3 = sum((x - mean_class3) ** 2 for x in class3) / len(class3)

print("Mean Value of Class 1: ", mean_class1, "; Variance of Class 1: ", variance_class1)
print("Mean Value of Class 2: ", mean_class2, "; Variance of Class 2: ", variance_class2)
print("Mean Value of Class 3: ", mean_class3, "; Variance of Class 3: ", variance_class3)
print("\n********************************************************************************\n")

# ------------------------------------------------------------------------------------------
# Generate x values for the likelihood plots
x_values = [x[0] for x in training_data]
x_values.sort()
likelihood_class1 = [(1 / ((variance_class1 ** 0.5) * (2 * math.pi) ** 0.5)) *
                     math.exp(-(x - mean_class1) ** 2 / (2 * variance_class1)) for x in x_values]

likelihood_class2 = [(1 / (variance_class2 ** 0.5 * (2 * math.pi) ** 0.5)) *
                     math.exp(-(x - mean_class2) ** 2 / (2 * variance_class2)) for x in x_values]

likelihood_class3 = [(1 / (variance_class3 ** 0.5 * (2 * math.pi) ** 0.5)) *
                     math.exp(-(x - mean_class3) ** 2 / (2 * variance_class3)) for x in x_values]

plt.figure(figsize=(10, 6))

plt.plot(x_values, likelihood_class1, label='Likelihood of Class 1', color='blue')
plt.plot(x_values, likelihood_class2, label='Likelihood of Class 2', color='green')
plt.plot(x_values, likelihood_class3, label='Likelihood of Class 3', color='red')

plt.xlabel('Value of Data Points')
plt.ylabel('Likelihood')

plt.legend()

plt.savefig('likelihood_plot.png')
plt.show()

# ------------------------------------------------------------------------------------------

c1 = [prior_class1 * l1 for l1 in likelihood_class1]
c2 = [prior_class2 * l2 for l2 in likelihood_class2]
c3 = [prior_class3 * l3 for l3 in likelihood_class3]

cTot = []
for i in range(len(c1)):
    cTot.append(c1[i] + c2[i] + c3[i])


def calculate_posteriors(prior, likelihoods):
    posteriors = []
    for i, likelihood in enumerate(likelihoods):
        posterior = ((prior * likelihood) / cTot[i])
        posteriors.append(posterior)
    return posteriors


# Calculate posteriors for each class
posteriors_class1 = calculate_posteriors(prior_class1, likelihood_class1)
posteriors_class2 = calculate_posteriors(prior_class2, likelihood_class2)
posteriors_class3 = calculate_posteriors(prior_class3, likelihood_class3)

# ------------------------------------------------------------------------------------------

# Plot the posterior distributions
plt.figure(figsize=(10, 6))

plt.plot(x_values, posteriors_class1, label='Posterior Class 1', linestyle='--', color='blue')
plt.plot(x_values, posteriors_class2, label='Posterior Class 2', linestyle='--', color='green')
plt.plot(x_values, posteriors_class3, label='Posterior Class 3', linestyle='--', color='red')

plt.xlabel('Value of Data Points')
plt.ylabel('Posterior')
plt.legend()

plt.savefig('posterior_plot.png')

plt.show()

# ------------------------------------------------------------------------------------------

# Plotting the likelihood and posterior distributions
plt.figure(figsize=(10, 6))

plt.plot(x_values, likelihood_class1, label='Likelihood Class 1', color='blue')
plt.plot(x_values, likelihood_class2, label='Likelihood Class 2', color='green')
plt.plot(x_values, likelihood_class3, label='Likelihood Class 3', color='red')

plt.plot(x_values, posteriors_class1, label='Posterior Class 1', linestyle='--', color='blue')
plt.plot(x_values, posteriors_class2, label='Posterior Class 2', linestyle='--', color='green')
plt.plot(x_values, posteriors_class3, label='Posterior Class 3', linestyle='--', color='red')

# Plot training data
plt.scatter(class1, [0.1] * len(class1), label='Training Class 1', color='blue', marker='+')
plt.scatter(class2, [0.2] * len(class2), label='Training Class 2', color='green', marker='.')
plt.scatter(class3, [0.3] * len(class3), label='Training Class 3', color='red', marker='x')

# Plot testing data
test_x_values = [x[0] for x in testing_data]
test_labels = [x[1] for x in testing_data]
plt.scatter(test_x_values, [0.80] * len(testing_data), label='Test Data', color='black', marker='x')

plt.xlabel('Value of Data Points')
plt.ylabel('Probability Density')
plt.legend()

plt.savefig('part1_plot.png')

plt.show()


# ------------------------------------------------------------------------------------------

# Define a function to predict class based on the maximum posterior probability
def predict_class(x):
    likelihood_class1 = (1 / ((variance_class1 ** 0.5) * (2 * math.pi) ** 0.5)) * math.exp(
        -(x - mean_class1) ** 2 / (2 * variance_class1))

    likelihood_class2 = (1 / (variance_class2 ** 0.5 * (2 * math.pi) ** 0.5)) * math.exp(
        -(x - mean_class2) ** 2 / (2 * variance_class2))

    likelihood_class3 = (1 / (variance_class3 ** 0.5 * (2 * math.pi) ** 0.5)) * math.exp(
        -(x - mean_class3) ** 2 / (2 * variance_class3))

    c1 = prior_class1 * likelihood_class1
    c2 = prior_class2 * likelihood_class2
    c3 = prior_class3 * likelihood_class3

    cTot = c1 + c2 + c3

    posterior1 = (prior_class1 * likelihood_class1) / (cTot)
    posterior2 = (prior_class2 * likelihood_class2) / (cTot)
    posterior3 = (prior_class3 * likelihood_class3) / (cTot)

    posteriors = [posterior1, posterior2, posterior3]
    return posteriors.index(max(posteriors)) + 1  # Adding 1 to get the class label (1, 2, or 3)


# Calculate confusion matrix for the training data
train_predictions = []
train_true_labels = [int(item[1]) for item in training_data]  # Ensure labels are integers

for i, point in enumerate(training_data):
    x_val = point[0]
    # Calculate posterior probabilities for each class (same as before)
    predicted_class = predict_class(x_val)
    train_predictions.append(predicted_class)

# Initialize confusion matrix
train_confusion_matrix = [[0, 0, 0], [0, 0, 0], [0, 0, 0]]

# Update confusion matrix with integer indices
for i in range(len(train_true_labels)):
    train_confusion_matrix[int(train_true_labels[i]) - 1][int(train_predictions[i]) - 1] += 1

# Print confusion matrix for training data
print("Confusion Matrix for Training Data:")
for row in train_confusion_matrix:
    print(row)

# Calculate confusion matrix for the test data
test_predictions = []
test_true_labels = [int(item[1]) for item in testing_data]  # Ensure labels are integers
for i, point in enumerate(testing_data):
    x_test = point[0]
    # Calculate posterior probabilities for each class (same as before)
    predicted_class = predict_class(x_test)
    test_predictions.append(predicted_class)

# Initialize confusion matrix
test_confusion_matrix = [[0, 0, 0], [0, 0, 0], [0, 0, 0]]

# Update confusion matrix with integer indices
for i in range(len(test_true_labels)):
    test_confusion_matrix[int(test_true_labels[i]) - 1][int(test_predictions[i]) - 1] += 1

# Print confusion matrix for test data
print("\nConfusion Matrix for Test Data:")
for row in test_confusion_matrix:
    print(row)

# ------------------------------------------------------------------------------------------
