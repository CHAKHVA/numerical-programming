def function(x):
    return 3 * x ** 3 - 2 * x ** 2 + x
def derivative(x):
    return 9 * x ** 2 - 4 * x + 1

def forward_difference(x, h=0.1):
    return (function(x + h) - function(x)) / h

def backward_difference(x, h=0.1):
    return  (function(x) - function(x - h)) / h

def central_difference(x, h=0.1):
    return (function(x + h) - function(x - h)) / (2 * h)

import numpy as np
import matplotlib.pyplot as plt

x_values = np.linspace(-1, 1, 5000)

exact_derivative = derivative(x_values)
forward_diff = forward_difference(x_values)
backward_diff = backward_difference(x_values)
central_diff = central_difference(x_values)

plt.figure(figsize=(10, 6))
plt.plot(x_values, exact_derivative, label='Exact Derivative', color='black', linewidth=1)
plt.plot(x_values, forward_diff, label='Forward Difference', linestyle='--', color='green')
plt.plot(x_values, backward_diff, label='Backward Difference', linestyle='--', color='blue')
plt.plot(x_values, central_diff, label='Central Difference', linestyle='--', color='red')

plt.title('Comparison of Exact Derivative and Approximation Methods')
plt.xlabel('x')
plt.ylabel('Derivative')
plt.legend()
plt.grid(True)
plt.show()
