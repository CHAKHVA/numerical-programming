import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import Lasso, LinearRegression
from sklearn.datasets import make_regression
from sklearn.preprocessing import StandardScaler

# Set random seed for reproducibility
np.random.seed(0)

# Generate regression data
X, y, coef_true = make_regression(
    n_samples=200,
    n_features=50,
    n_informative=5,
    noise=0.1,
    coef=True
)

# Standardize features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Initialize Lasso regression model with alpha (regularization strength)
lasso = Lasso(alpha=0.1)

# Fit the model
lasso.fit(X_scaled, y)

# Get the coefficients
lasso_coefs = lasso.coef_

# Plotting the coefficients
plt.figure(figsize=(12, 6))
plt.stem(lasso_coefs)
plt.title('Lasso Regression Coefficients')
plt.xlabel('Feature Index')
plt.ylabel('Coefficient Value')
plt.show()

# Fit ordinary least squares regression
ols = LinearRegression()
ols.fit(X_scaled, y)
ols_coefs = ols.coef_

# Plotting OLS vs. Lasso Coefficients
plt.figure(figsize=(12, 6))
plt.stem(ols_coefs, markerfmt='C0o', label='OLS Coefficients')
plt.stem(lasso_coefs, markerfmt='C1x', label='Lasso Coefficients')
plt.title('Comparison of OLS and Lasso Coefficients')
plt.xlabel('Feature Index')
plt.ylabel('Coefficient Value')
plt.legend()
plt.show()