import numpy as np
import matplotlib.pyplot as plt

def crypto_market(t, y, params):
    P, S, N = y
    alpha, beta, gamma, delta, epsilon, zeta, F, K = params
    dP_dt = alpha * S * (1 - P / K) - beta * P * N
    dS_dt = gamma * (F - P) - delta * S
    dN_dt = epsilon * (P - F)**2 - zeta * N
    return np.array([dP_dt, dS_dt, dN_dt])

butcher_table = {
    "A": np.array([[0.5, 0], [0.5, 0.5]]),
    "b": np.array([0.5, 0.5]),
    "c": np.array([0.5, 1.0])
}

def dirk_solver(f, y0, t_span, params, h):
    t_values = np.arange(t_span[0], t_span[1] + h, h)
    y_values = [y0]

    for t in t_values[:-1]:
        y_current = y_values[-1]
        Y = np.zeros((3, len(butcher_table['b'])))

        Y_flat = Y.ravel()
        for _ in range(10):
            Y_matrix = Y_flat.reshape((3, -1))
            for i in range(len(butcher_table['b'])):
                Yi = y_current + h * sum(
                    butcher_table['A'][i, j] * f(t + butcher_table['c'][j] * h, Y_matrix[:, j], params)
                    for j in range(len(butcher_table['b']))
                )
                Y_matrix[:, i] = Yi
            Y_flat = Y_matrix.ravel()

        Y = Y_flat.reshape((3, -1))
        y_next = y_current + h * sum(butcher_table['b'][i] * f(t + butcher_table['c'][i] * h, Y[:, i], params) for i in range(len(butcher_table['b'])))
        y_values.append(y_next)

    return t_values, np.array(y_values)

experiments = [
    ([0.1, 0.02, 0.1, 0.05, 0.01, 0.1, 100, 200], [50, 20, 10], "Baseline Experiment"),
    ([0.2, 0.04, 0.15, 0.1, 0.02, 0.15, 150, 300], [60, 25, 15], "High Market Volatility"),
    ([0.05, 0.01, 0.05, 0.02, 0.005, 0.05, 80, 150], [40, 15, 8], "Low Market Volatility")
]

plt.figure(figsize=(15, 10))
for idx, (params, y0, label) in enumerate(experiments):
    t_span = [0, 100]
    h = 0.5
    t_values, y_values = dirk_solver(crypto_market, y0, t_span, params, h)

    plt.subplot(3, 1, idx+1)
    plt.plot(t_values, y_values[:, 0], label="Price (P)", color="b")
    plt.plot(t_values, y_values[:, 1], label="Supply (S)", color="g")
    plt.plot(t_values, y_values[:, 2], label="Network Effect (N)", color="r")
    plt.title(f"{label}")
    plt.xlabel("Time")
    plt.ylabel("Value")
    plt.legend()
    plt.grid(True)

plt.tight_layout()
plt.show()
