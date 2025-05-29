import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import PolynomialFeatures
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt

# Generate data
m = 100
X = 6 * np.random.rand(m, 1) - 4
y = 0.5 * X ** 2 + X + 2 + np.random.randn(m, 1)

def plot_learning_curves(model, X, y, title):
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2)
    train_errors, val_errors = [], []
    
    for m in range(1, len(X_train)):
        model.fit(X_train[:m], y_train[:m])
        y_train_predict = model.predict(X_train[:m])
        y_val_predict = model.predict(X_val)
        train_errors.append(mean_squared_error(y_train_predict, y_train[:m]))
        val_errors.append(mean_squared_error(y_val_predict, y_val))
    
    plt.plot(np.sqrt(train_errors), "r-+", linewidth=2, label="train")
    plt.plot(np.sqrt(val_errors), "b-", linewidth=3, label="val")
    plt.legend()
    plt.xlabel("Training set size")
    plt.ylabel("RMSE")
    plt.title(title)
    return np.mean(np.sqrt(train_errors)), np.mean(np.sqrt(val_errors))

# Linear regression
lin_reg = LinearRegression()

# Create figure
plt.figure(figsize=(12, 6))

# Plot linear regression learning curves
plt.subplot(1, 2, 1)
train_error_lin, val_error_lin = plot_learning_curves(lin_reg, X, y, "Learning Curves for Linear Regression")

# Polynomial regression pipeline
polynomial_regression = Pipeline([
    ("poly_features", PolynomialFeatures(degree=10, include_bias=False)),
    ("lin_reg", LinearRegression()),
])

# Plot polynomial regression learning curves
plt.subplot(1, 2, 2)
train_error_poly, val_error_poly = plot_learning_curves(polynomial_regression, X, y, "Learning Curves for Polynomial Regression")

plt.tight_layout()
plt.show()

# Print results
print("Linear Regression - Training Error: {:.2f}, Validation Error: {:.2f}".format(train_error_lin, val_error_lin))
print("Polynomial Regression - Training Error: {:.2f}, Validation Error: {:.2f}".format(train_error_poly, val_error_poly))