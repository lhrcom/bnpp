# BPNN - Numpy-based Backpropagation Neural Network

[![Python](https://img.shields.io/badge/python-3.8%2B-blue.svg)](https://www.python.org/)
[![License](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)

BPNN is a lightweight **fully connected Backpropagation Neural Network** implemented in **NumPy**, designed for **regression tasks**. It provides a simple, educational framework for experimenting with neural network fundamentals without relying on heavy deep learning libraries.

---

## 🚀 Features

- Fully connected feedforward neural network
- Manual forward and backward propagation
- He initialization for weights
- Sigmoid activation for hidden layers, linear output for regression
- Standardization utilities to avoid data leakage
- Configurable number of layers and neurons
- Easy visualization of training loss and predictions

---

## 📦 Installation

Clone the repository:

```bash
git clone https://github.com/你的用户名/bpnn.git
cd bpnn
import pandas as pd
from sklearn.model_selection import train_test_split
from bpnn.preprocessing import standardize_train, standardize_test
from bpnn.model import BPRegressor
from sklearn.metrics import mean_squared_error, r2_score

# Load dataset
df = pd.read_excel('/path/to/fetch_california_housing.xlsx')
X = df.drop('target', axis=1).values
y = df['target'].values

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Standardize features
X_train_std, X_mean, X_std = standardize_train(X_train)
X_test_std = standardize_test(X_test, X_mean, X_std)

# Standardize target
y_train_std, y_mean, y_std = standardize_train(y_train.reshape(-1,1))
y_train_std = y_train_std.flatten()

# Build and train model
model = BPRegressor(layers=[X_train_std.shape[1], 32, 8, 1], lr=0.3, epochs=1000)
model.fit(X_train_std, y_train_std)

# Make predictions
y_pred = model.predict(X_test_std) * y_std + y_mean

# Evaluate
print("Test MSE:", mean_squared_error(y_test, y_pred))
print("Test R2:", r2_score(y_test, y_pred))
