import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import Ridge, LinearRegression
from sklearn.datasets import make_regression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import StandardScaler

# Set random seed
np.random.seed(0)

# Generate regression data
X, y = make_regression(
    n_samples=100,
    n_features=10,
    n_informative=5,
    noise=10.0
)

# Introduce multicollinearity
X[:, 5:] = X[:, :5] + np.random.normal(0, 0.1, size=(100, 5))

# Standardize features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Split data
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.3, random_state=42
)

# Initialize Ridge regression model
ridge = Ridge(alpha=1.0)

# Fit the model
ridge.fit(X_train, y_train)

# Predict on test data
y_pred_ridge = ridge.predict(X_test)

# Calculate Mean Squared Error
mse_ridge = mean_squared_error(y_test, y_pred_ridge)
print(f"Ridge Regression MSE: {mse_ridge:.2f}")

# Fit OLS model
ols = LinearRegression()
ols.fit(X_train, y_train)

# Predict on test data
y_pred_ols = ols.predict(X_test)

# Calculate Mean Squared Error
mse_ols = mean_squared_error(y_test, y_pred_ols)
print(f"OLS Regression MSE: {mse_ols:.2f}")

# Plotting coefficients
plt.figure(figsize=(12, 6))
indices = np.arange(len(ridge.coef_))
width = 0.35

plt.bar(indices - width/2, ols.coef_, width=width, label='OLS Coefficients')
plt.bar(indices + width/2, ridge.coef_, width=width, label='Ridge Coefficients')
plt.xlabel('Feature Index')
plt.ylabel('Coefficient Value')
plt.title('Comparison of OLS and Ridge Regression Coefficients')
plt.legend()
plt.show()
