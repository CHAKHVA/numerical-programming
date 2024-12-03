import numpy as np
import matplotlib.pyplot as plt

def f(x):
    return np.sin(x)

def df_exact(x):
    return np.cos(x)

def finite_differences(x, h):
    forward = (f(x + h) - f(x)) / h

    backward = (f(x) - f(x - h)) / h

    central = (f(x + h) - f(x - h)) / (2 * h)

    return forward, backward, central

def plot_error_analysis():
    x0 = np.pi/4

    h_values = np.logspace(-15, 0, 100)

    forward_errors = []
    backward_errors = []
    central_errors = []

    true_derivative = df_exact(x0)
    for h in h_values:
        forward, backward, central = finite_differences(x0, h)

        forward_errors.append(abs(forward - true_derivative))
        backward_errors.append(abs(backward - true_derivative))
        central_errors.append(abs(central - true_derivative))

    plt.figure(figsize=(10, 6))
    plt.loglog(h_values, forward_errors, 'r-', label='Forward Difference')
    plt.loglog(h_values, backward_errors, 'g-', label='Backward Difference')
    plt.loglog(h_values, central_errors, 'b-', label='Central Difference')

    plt.grid(True)
    plt.xlabel('Step size (h)')
    plt.ylabel('Absolute Error')
    plt.title('Finite Difference Approximation Errors')
    plt.legend()
    plt.show()

def plot_tangent_approximation():
    x0 = np.pi/4
    h = 0.1

    x = np.linspace(x0 - 1, x0 + 1, 200)
    y = f(x)

    derivative_approx = (f(x0 + h) - f(x0 - h)) / (2 * h)
    tangent = f(x0) + derivative_approx * (x - x0)

    plt.figure(figsize=(10, 6))
    plt.plot(x, y, 'b-', label='f(x) = sin(x)')
    plt.plot(x, tangent, 'r--', label='Tangent Approximation')
    plt.plot(x0, f(x0), 'ko', label='Point of Tangency')

    plt.grid(True)
    plt.xlabel('x')
    plt.ylabel('y')
    plt.title('Function and Its Tangent Line Approximation')
    plt.legend()
    plt.show()

# Run the analysis
if __name__ == "__main__":
    plot_error_analysis()
    plot_tangent_approximation()

'''
import numpy as np
import matplotlib.pyplot as plt


def demonstrate_roundoff():
    print("\n1. Round-off Accumulation:")
    sum1 = sum([0.1] * 10)
    print(f"Sum of 0.1 ten times: {sum1}")
    print(f"Direct value 1.0: {1.0}")
    print(f"Difference: {abs(sum1 - 1.0)}")


def demonstrate_catastrophic_cancellation():
    print("\n2. Catastrophic Cancellation:")
    a = 1.0
    b = a + 1e-8
    c = b - a
    print(f"Expected: 1e-8")
    print(f"Actual: {c}")
    print(f"Relative Error: {abs(c - 1e-8) / 1e-8}")


def demonstrate_associative_violation():
    print("\n3. Associative Property Violation:")
    x = 1e20
    y = -1e20
    z = 1.0

    result1 = (x + y) + z
    result2 = x + (y + z)

    print(f"(a + b) + c = {result1}")
    print(f"a + (b + c) = {result2}")


def plot_precision_error():
    x = np.linspace(0, 1, 1000)
    float32_sum = np.float32(0)
    float64_sum = np.float64(0)

    float32_results = []
    float64_results = []

    for val in x:
        float32_sum += np.float32(val)
        float64_sum += np.float64(val)
        float32_results.append(float(float32_sum))
        float64_results.append(float(float64_sum))

    plt.figure(figsize=(10, 6))
    plt.plot(x, float32_results, label='32-bit float')
    plt.plot(x, float64_results, label='64-bit float')
    plt.plot(x, x * x.size / 2, '--', label='Exact sum')
    plt.title('Cumulative Sum: 32-bit vs 64-bit Precision')
    plt.xlabel('Number of additions')
    plt.ylabel('Sum')
    plt.legend()
    plt.grid(True)
    plt.show()


# Run all demonstrations
if __name__ == "__main__":
    demonstrate_roundoff()
    demonstrate_catastrophic_cancellation()
    demonstrate_associative_violation()
    plot_precision_error()
'''