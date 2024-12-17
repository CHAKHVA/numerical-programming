# Model Description: Quantum Dot Energy States

The Quantum Dot Energy States model describes the evolution of excitation level (E), temperature (T), and applied voltage (V) in a quantum dot system. The system is modeled using a set of three coupled nonlinear ordinary differential equations (ODEs).

## System of Equations

1. **Excitation Level Equation (dE/dt):**
   
   $$\frac{dE}{dt} = -\alpha E + \beta V - \gamma E T$$
   
   - **Term 1:** $-\alpha E$ represents the natural decay of the excitation level.
   - **Term 2:** $+\beta V$ reflects excitation due to applied voltage.
   - **Term 3:** $-\gamma E T$ accounts for temperature-induced excitation decay.

2. **Temperature Equation (dT/dt):**
   
   $$\frac{dT}{dt} = \lambda E T - \kappa (T - T_0)$$
   
   - **Term 1:** $+\lambda E T$ models heat generation due to excitation.
   - **Term 2:** $-\kappa (T - T_0)$ represents cooling toward ambient temperature $T_0$.

3. **Voltage Equation (dV/dt):**
   
   $$\frac{dV}{dt} = \omega (V_0 - V) - \eta E$$
   
   - **Term 1:** $\omega (V_0 - V)$ restores the applied voltage toward the target voltage $V_0$.
   - **Term 2:** $-\eta E$ models voltage reduction due to excitation.

## Model Parameters

- $\alpha, \beta, \gamma, \kappa, \lambda, \omega, \eta$: System-specific constants
- $T_0$: Ambient temperature
- $V_0$: Applied voltage
- $E, T, V$: State variables representing excitation level, temperature, and applied voltage

## Numerical Method

The system is solved using the Embedded Runge-Kutta (Dormand-Prince-like) method with adaptive step size control. The method estimates solutions of different orders simultaneously and adjusts the step size based on the local truncation error. This ensures efficient and accurate solutions for stiff ODE systems.

## Simulation Setup

- **Initial Conditions:** 
  - Standard: $E=0.5$, $T=300$, $V=1.0$
  - High Excitation: $E=0.7$, $T=310$, $V=1.2$
  - Low Energy: $E=0.3$, $T=290$, $V=0.8$
- **Time Span:** 0 to 50
- **Step Size:** Adaptive

## Experiments & Test Methods

We conducted three numerical experiments under different parameter settings:

1. **Standard Conditions:**
   - Parameters: $\alpha=0.1$, $\beta=0.05$, $\gamma=0.02$, $\kappa=0.05$, $\lambda=0.1$, $\omega=0.04$, $\eta=0.02$, $T_0=300$, $V_0=1.5$
   - Insight: Balanced dynamics with moderate excitation, temperature, and voltage variations.

2. **High Excitation:**
   - Parameters: $\alpha=0.2$, $\beta=0.1$, $\gamma=0.04$, $\kappa=0.1$, $\lambda=0.2$, $\omega=0.08$, $\eta=0.04$, $T_0=310$, $V_0=1.8$
   - Insight: Rapid and large-scale oscillations due to high excitation and heating effects.

3. **Low Energy State:**
   - Parameters: $\alpha=0.05$, $\beta=0.03$, $\gamma=0.01$, $\kappa=0.03$, $\lambda=0.08$, $\omega=0.02$, $\eta=0.01$, $T_0=290$, $V_0=1.2$
   - Insight: Minimal variations with stable, low-energy behavior.

## Results & Insights

- **Standard Conditions:** The system exhibits moderate oscillations in excitation, temperature, and voltage, indicating stable operation.
- **High Excitation:** Large-scale oscillations emerge due to strong feedback between temperature, excitation, and voltage.
- **Low Energy State:** The system remains near equilibrium with low excitation and voltage fluctuations.

The experiments demonstrate the model's ability to capture key dynamics of a quantum dot system while providing insight into energy transfer and feedback processes.

#### I used Claude AI to create Cryptocurrency Market Psychology equations system