import numpy as np
import matplotlib.pyplot as plt

def quantum_dot_dynamics(t, y, params):
    E, T, V = y
    alpha, beta, gamma, kappa, lambd, omega, eta, T0, V0 = params

    dE_dt = -alpha * E + beta * V - gamma * E * T
    dT_dt = lambd * E * T - kappa * (T - T0)
    dV_dt = omega * (V0 - V) - eta * E

    return np.array([dE_dt, dT_dt, dV_dt])

def embedded_rk_fast(f, y0, t_span, params, h_init=0.5, tol=1e-4):
    t_values = [t_span[0]]
    y_values = [np.array(y0)]
    h = h_init
    t = t_span[0]
    y = np.array(y0)

    while t < t_span[1]:
        k1 = f(t, y, params)
        k2 = f(t + h / 5, y + h * k1 / 5, params)
        k3 = f(t + 3 * h / 10, y + h * (3 * k1 / 40 + 9 * k2 / 40), params)
        k4 = f(t + 4 * h / 5, y + h * (44 * k1 / 45 - 56 * k2 / 15 + 32 * k3 / 9), params)
        k5 = f(t + 8 * h / 9, y + h * (19372 * k1 / 6561 - 25360 * k2 / 2187 +
                                       64448 * k3 / 6561 - 212 * k4 / 729), params)
        k6 = f(t + h, y + h * (9017 * k1 / 3168 - 355 * k2 / 33 +
                               46732 * k3 / 5247 + 49 * k4 / 176 - 5103 * k5 / 18656), params)

        y4 = y + h * (35 * k1 / 384 + 500 * k3 / 1113 + 125 * k4 / 192 - 2187 * k5 / 6784 + 11 * k6 / 84)
        y5 = y + h * (5179 * k1 / 57600 + 7571 * k3 / 16695 +
                      393 * k4 / 640 - 92097 * k5 / 339200 + 187 * k6 / 2100)

        error = np.linalg.norm(y5 - y4, ord=np.inf)

        if error <= tol:
            t += h
            y = y5
            t_values.append(t)
            y_values.append(y)

        h = h * min(max(0.1, 0.9 * (tol / error) ** 0.2), 5.0)

    return np.array(t_values), np.array(y_values)

experiments = [
    ([0.1, 0.05, 0.02, 0.05, 0.1, 0.04, 0.02, 300, 1.5], [0.5, 300, 1.0], "Standard Conditions"),
    ([0.2, 0.1, 0.04, 0.1, 0.2, 0.08, 0.04, 310, 1.8], [0.7, 310, 1.2], "High Excitation"),
    ([0.05, 0.03, 0.01, 0.03, 0.08, 0.02, 0.01, 290, 1.2], [0.3, 290, 0.8], "Low Energy State")
]

plt.figure(figsize=(18, 12))

for idx, (params, y0, label) in enumerate(experiments):
    t_span = [0, 50]
    t_values, y_values = embedded_rk_fast(quantum_dot_dynamics, y0, t_span, params)

    plt.subplot(3, 1, idx + 1)
    plt.plot(t_values, y_values[:, 0], label="Excitation Level (E)", color="b")
    plt.plot(t_values, y_values[:, 1], label="Temperature (T)", color="r")
    plt.plot(t_values, y_values[:, 2], label="Applied Voltage (V)", color="g")
    plt.title(f"{label} Dynamics")
    plt.xlabel("Time")
    plt.ylabel("State Variables")
    plt.grid(True)
    plt.legend()

plt.tight_layout()
plt.show()
