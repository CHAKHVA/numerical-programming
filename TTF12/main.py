import numpy as np
import matplotlib.pyplot as plt


def f(x):
    return x ** 3 - 3 * x + 1


def df(x):
    return 3 * x ** 2 - 3


def newtons_method(f, df, x0, tolerance, max_iterations=100):
    x = x0
    approximations = [x]

    for _ in range(max_iterations):
        fx = f(x)
        dfx = df(x)

        if dfx == 0:
            print("Zero derivative. Stopping iteration.")
            break

        x_new = x - fx / dfx
        approximations.append(x_new)

        if abs(x_new - x) < tolerance:
            x = x_new
            break

        x = x_new

    return x, approximations


x0 = 1.5 # If we start from 1, code stops immediately because there's zero derivative
tolerance = 1e-6
root, approximations = newtons_method(f, df, x0, tolerance)

x_values = np.linspace(-2, 2, 400)
y_values = f(x_values)

plt.figure(figsize=(10, 6))
plt.plot(x_values, y_values, label='f(x) = x^3 - 3x + 1', color='blue')
plt.axhline(0, color='gray', linestyle='--', linewidth=0.8, label='y = 0')
plt.scatter(approximations, [f(x) for x in approximations], color='red', label='Approximations', zorder=5)
plt.title('Newton\'s Method: Solving f(x) = x^3 - 3x + 1')
plt.xlabel('x')
plt.ylabel('f(x)')
plt.legend()
plt.grid(True)
plt.scatter(root, f(root), color='green', label='Root (Converged)', zorder=6)
plt.legend()
plt.show()

print(f"Root: {root}")
print("Sequence of Approximations:")
for i, approx in enumerate(approximations):
    print(f"x{i}: {approx}")
