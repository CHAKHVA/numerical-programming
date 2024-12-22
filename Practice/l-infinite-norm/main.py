import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import linprog

# Define the nonlinear function
def g(x):
    return np.sin(x)

# Interval [a, b]
a, b = 0, 2 * np.pi

# Number of sample points
n_points = 100

# Sample points
x_samples = np.linspace(a, b, n_points)
y_samples = g(x_samples)

# Number of variables: m, c, E
n_vars = 3

# Objective: Minimize E (the third variable)
c_lp = np.array([0, 0, 1])

# Inequality constraints: A_ub * x <= b_ub
A_ub = []
b_ub = []

for i in range(n_points):
    x_i = x_samples[i]
    g_i = y_samples[i]

    # (mx_i + c) - g_i - E <= 0
    A_ub.append([x_i, 1, -1])
    b_ub.append(g_i)

    # -(mx_i + c) + g_i - E <= 0
    A_ub.append([-x_i, -1, -1])
    b_ub.append(-g_i)

A_ub = np.array(A_ub)
b_ub = np.array(b_ub)

# Bounds for variables (None implies no bounds)
bounds = [(None, None), (None, None), (0, None)]  # E >= 0

# Solve LP problem
result = linprog(
    c=c_lp,
    A_ub=A_ub,
    b_ub=b_ub,
    bounds=bounds,
    method='highs'
)

# Extract the optimal values
m_opt, c_opt, E_min = result.x
print(f"Optimal m: {m_opt:.4f}")
print(f"Optimal c: {c_opt:.4f}")
print(f"Minimum Maximum Error (E): {E_min:.4f}")

# Compute the approximation
f_approx = m_opt * x_samples + c_opt

# Plot the original function and the approximation
plt.figure(figsize=(12, 6))
plt.plot(x_samples, y_samples, label='Original Function g(x) = sin(x)')
plt.plot(x_samples, f_approx, label='Chebyshev Approximation f(x) = mx + c', linestyle='--')
plt.fill_between(x_samples, f_approx - E_min, f_approx + E_min, color='grey', alpha=0.2, label='Maximum Error Band')
plt.title('Chebyshev Approximation Using L-infinity Norm')
plt.xlabel('x')
plt.ylabel('y')
plt.legend()
plt.show()
