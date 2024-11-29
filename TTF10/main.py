import numpy as np
import matplotlib.pyplot as plt

a = 0.8
b = 0.1
c = 1.0
d = 0.2

def predator_prey(x0, y0, t_max, dt):
    t = np.arange(0, t_max, dt)
    x = np.zeros(len(t))
    y = np.zeros(len(t))
    x[0] = x0
    y[0] = y0
    for i in range(1, len(t)):
        dxdt = x[i - 1] * (a - b * y[i - 1])
        dydt = -y[i - 1] * (c - d * x[i - 1])
        x[i] = x[i - 1] + dxdt * dt
        y[i] = y[i - 1] + dydt * dt
    return t, x, y

x0 = 10
y0 = 5
t_max = 50
dt = 0.01

t, x, y = predator_prey(x0, y0, t_max, dt)

plt.figure(figsize=(12, 6))
plt.plot(t, x, label='Prey Population (x(t))', color='blue')
plt.plot(t, y, label='Predator Population (y(t))', color='red')
plt.xlabel('Time')
plt.ylabel('Population')
plt.title('Predator-Prey Model')
plt.legend()
plt.grid()
plt.show()

plt.figure(figsize=(8, 6))
plt.plot(x, y, label='Predator-Prey Phase Plot', color='green')
plt.xlabel('Prey Population (x(t))')
plt.ylabel('Predator Population (y(t))')
plt.title('Phase Plot of Predator-Prey Model')
plt.grid()
plt.legend()
plt.show()


'''
The prey and predator populations exhibit stable oscillations over time.
Neither species goes extinct, and the populations do not stabilize at fixed values but cycle around equilibrium points.
The closed loops in the phase plot indicate a limit cycle, meaning the system maintains periodic oscillations without divergence or collapse.
The predator-prey model demonstrates neutral stability: populations oscillate indefinitely in a predictable pattern, influenced by the initial conditions and system parameters.
'''