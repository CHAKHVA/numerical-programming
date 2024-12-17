# Model Description: Neural Network Learning Dynamics

The Neural Network Learning Dynamics model represents the evolution of neural network weights, momentum, and error gradients during the learning process. This model uses a system of three nonlinear ordinary differential equations (ODEs) to simulate learning dynamics.

## System of Equations

1. **Weight Update Equation (dW/dt):**
   
   $$\frac{dW}{dt} = -\alpha W \nabla E + \beta M$$
   
   - **Term 1:** $-\alpha W \nabla E$ represents the gradient descent process where weights are updated in proportion to the error gradient.
   - **Term 2:** $+\beta M$ reflects the momentum contribution, which helps smooth out weight updates.

2. **Momentum Update Equation (dM/dt):**
   
   $$\frac{dM}{dt} = \gamma W - \delta M$$
   
   - **Term 1:** $+\gamma W$ adds the influence of weights to the momentum.
   - **Term 2:** $-\delta M$ represents decay in momentum over time.

3. **Error Gradient Update Equation (d∇E/dt):**
   
   $$\frac{d\nabla E}{dt} = \epsilon (T - W) - \zeta \nabla E$$
   
   - **Term 1:** $\epsilon (T - W)$ accounts for the mismatch between the target value $T$ and the current weights.
   - **Term 2:** $-\zeta \nabla E$ describes the natural decay of the error gradient.

## Model Parameters

- $\alpha, \beta, \gamma, \delta, \epsilon, \zeta$: Constants governing the dynamics of the system
- $T$: Target weight value
- $W$, $M$, $\nabla E$: Variables representing weights, momentum, and error gradient, respectively

## Numerical Method

The system is solved using the 3-step Adams-Bashforth-Moulton predictor-corrector method, which combines explicit and implicit integration techniques for improved accuracy:

1. **Predictor:** Adams-Bashforth uses previous steps to estimate the next value.
2. **Corrector:** Adams-Moulton refines the prediction by incorporating the new step value.

This method is well-suited for moderately stiff ODE systems and provides stability for this learning dynamics problem.

## Simulation Setup

- **Initial Conditions:** 
  - Standard Learning: $W=0.5$, $M=0.1$, $\nabla E=0.05$
  - Fast Learning: $W=0.7$, $M=0.15$, $\nabla E=0.1$
  - Slow Learning: $W=0.3$, $M=0.05$, $\nabla E=0.02$
- **Time Span:** 0 to 20
- **Step Size:** 0.1

## Experiments & Test Methods

Three experiments were conducted to analyze learning dynamics under varying conditions:

1. **Standard Learning:**
   - Parameters: $\alpha=0.1$, $\beta=0.02$, $\gamma=0.15$, $\delta=0.1$, $\epsilon=0.05$, $\zeta=0.1$, $T=1.0$
   - Insight: The weights converge gradually, with steady momentum and error gradient decay.

2. **Fast Learning:**
   - Parameters: $\alpha=0.2$, $\beta=0.04$, $\gamma=0.3$, $\delta=0.2$, $\epsilon=0.1$, $\zeta=0.2$, $T=1.5$
   - Insight: Rapid weight updates with large oscillations in momentum and gradient.

3. **Slow Learning:**
   - Parameters: $\alpha=0.05$, $\beta=0.01$, $\gamma=0.1$, $\delta=0.05$, $\epsilon=0.02$, $\zeta=0.05$, $T=0.8$
   - Insight: Slow convergence with minimal fluctuations in weights, momentum, and gradient.

## Results & Insights

- **Standard Learning:** Demonstrates balanced learning dynamics, achieving smooth convergence of weights while controlling momentum and error gradients.
- **Fast Learning:** Highlights the risks of aggressive updates, including overshooting and instability.
- **Slow Learning:** Achieves highly stable results but converges slowly, which may be impractical for real-time applications.

The visualizations show the interplay between weights, momentum, and error gradients, providing insights into optimizing learning rates and dynamic behavior.