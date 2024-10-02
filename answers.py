import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

dataset = pd.read_csv("data_problem2.csv", header=None)
dataset = dataset.transpose() # Transpose the data
dataset.columns = ["Feature", "Label"]  # Give the column names
dataset["Label"] = dataset["Label"].astype(int)  # Convert label to integer type

# Printing info
print("\nInformation:")
print(dataset.info())
print("\nMissing values:")
print(dataset.isnull().sum())
print("\nStatistics:")
print(dataset.describe())

# Splitting the dataset into the Training set and Test set
from sklearn.model_selection import train_test_split
X = dataset["Feature"].values
y = dataset["Label"]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)

# Calculate the mean and variance for each class
parameters = {}
classes = np.unique(y_train) # Find the unique classes
for clss in classes:
  features = X_train[y_train == clss] # Extract features for the current class
  parameters[clss] = {"mean": np.mean(features), "variance": np.var(features)} # Store the parameters

def Bayes_Classifier(x, parameters):
  output = []
  for clss in classes:
    mean = parameters[clss]["mean"]
    variance = parameters[clss]["variance"]
    p = (1/np.sqrt(2* np.pi * variance)) * np.exp(-0.5 * ((x - mean) ** 2) / variance)     # apply Gaussian formula
    output.append(p)
  return np.argmax(output)

# Apply the classifier on the test set
y_pred = np.array([Bayes_Classifier(x, parameters) for x in X_test])
accuracy = np.mean(y_pred == y_test)
print(f"Accuracy = {accuracy:.2f}")

misclassified = (y_pred != y_test)
classified = (y_pred == y_test)


# Ploting histograms
plt.figure(figsize=(10, 6))
plt.hist(X, bins=30, edgecolor="r", alpha=0.7)
plt.title("Histogram of feature values")
plt.xlabel("Feature Values")
plt.ylabel("Number of samples")
plt.show()


plt.scatter(X_test[classified], y_test[classified], color='green', label='Classified', alpha=0.6, s= 50)
plt.scatter(X_test[misclassified], y_test[misclassified], color='red', label='Misclassified', alpha=0.6, s = 50)
plt.title("Classified vs Misclassified Data")
plt.xlabel("Feature Values")
plt.ylabel("Class Labels")
plt.legend()
plt.show()
