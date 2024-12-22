import numpy as np
import matplotlib.pyplot as plt

def f(x):
    return np.sin(x)

x_points = np.array([0, np.pi/6, np.pi/4, np.pi/3, np.pi/2])
y_points = f(x_points)

def lagrange_term(i, x, x_points):
    term = 1
    for j in range(len(x_points)):
        if i != j:
            term *= (x - x_points[j]) / (x_points[i] - x_points[j])
    return term

def L(x):
    lagrange_poly = 0
    for i in range(len(x_points)):
        lagrange_poly += y_points[i] * lagrange_term(i, x, x_points)
    return lagrange_poly

def error(x):
    return np.abs(f(x) - L(x))

x_values = np.linspace(0, np.pi/2, 100)

L_values = [L(x) for x in x_values]
error_values = error(x_values)

fig, ax1 = plt.subplots(figsize=(10, 5))

ax1.plot(x_values, L_values, label="L(x)", color="red")
ax1.plot(x_points, y_points, 'o', color='black')
ax1.set_xlabel("x")
ax1.set_ylabel("L(x)")
ax1.legend(loc="upper left")
ax1.set_title("Interpolation of sin(x) using Lagrange Polynomial with Error")

ax2 = ax1.twinx()
ax2.plot(x_values, error_values, label="Error(x)", color="purple")
ax2.set_ylabel("Error(x)")
ax2.legend(loc="upper right")

fig.tight_layout()
plt.grid(True)
plt.show()