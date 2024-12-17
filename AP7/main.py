import numpy as np
import matplotlib.pyplot as plt

def neural_dynamics(t, y, params):
    W, M, grad_E = y
    alpha, beta, gamma, delta, epsilon, zeta, T = params
    dW_dt = -alpha * W * grad_E + beta * M
    dM_dt = gamma * W - delta * M
    dgrad_E_dt = epsilon * (T - W) - zeta * grad_E
    return np.array([dW_dt, dM_dt, dgrad_E_dt])

def adams_bashforth_moulton(f, y0, t_span, params, h):
    t_values = np.arange(t_span[0], t_span[1] + h, h)
    y_values = [y0]

    for i in range(2):
        t = t_values[i]
        y_next = y_values[-1] + h * f(t, y_values[-1], params)
        y_values.append(y_next)

    for i in range(2, len(t_values) - 1):
        t = t_values[i]
        y_pred = y_values[-1] + h * (
            23 * f(t, y_values[-1], params) -
            16 * f(t_values[i - 1], y_values[-2], params) +
            5 * f(t_values[i - 2], y_values[-3], params)) / 12

        y_corr = y_values[-1] + h * (
            5 * f(t_values[i + 1], y_pred, params) +
            8 * f(t, y_values[-1], params) -
            f(t_values[i - 1], y_values[-2], params)) / 12

        y_values.append(y_corr)

    return t_values, np.array(y_values)

experiments = [
    ([0.1, 0.02, 0.15, 0.1, 0.05, 0.1, 1.0], [0.5, 0.1, 0.05], "Standard Learning"),
    ([0.2, 0.04, 0.3, 0.2, 0.1, 0.2, 1.5], [0.7, 0.15, 0.1], "Fast Learning"),
    ([0.05, 0.01, 0.1, 0.05, 0.02, 0.05, 0.8], [0.3, 0.05, 0.02], "Slow Learning")
]

plt.figure(figsize=(15, 10))

for idx, (params, y0, label) in enumerate(experiments):
    t_span = [0, 20]
    h = 0.1
    t_values, y_values = adams_bashforth_moulton(neural_dynamics, y0, t_span, params, h)

    plt.subplot(3, 1, idx + 1)
    plt.plot(t_values, y_values[:, 0], label="Weights (W)", color="b")
    plt.plot(t_values, y_values[:, 1], label="Momentum (M)", color="g")
    plt.plot(t_values, y_values[:, 2], label="Error Gradient (∇E)", color="r")
    plt.title(f"{label} Dynamics")
    plt.xlabel("Time")
    plt.ylabel("Value")
    plt.legend()
    plt.grid(True)

plt.tight_layout()
plt.show()
