# Model Description: Cryptocurrency Market Psychology

The proposed model simulates the dynamics of a cryptocurrency market using a system of three nonlinear ordinary differential equations (ODEs). The equations describe how the cryptocurrency's price (P), supply (S), and network effect (N) evolve over time based on market psychology principles.

## System of Equations

1. **Price Equation (dP/dt):**
   
   $$\frac{dP}{dt} = \alpha S (1 - \frac{P}{K}) - \beta P N$$
   
   - **Term 1:** The first term $\alpha S (1 - \frac{P}{K})$ reflects price growth driven by supply availability and market saturation. $K$ represents the maximum market capacity.
   - **Term 2:** The second term $-\beta P N$ indicates how increased network effect (N) can reduce the price if it leads to speculative behavior.

2. **Supply Equation (dS/dt):**
   
   $$\frac{dS}{dt} = \gamma (F - P) - \delta S$$
   
   - **Term 1:** $\gamma (F - P)$ represents the adjustment of supply based on the difference between the desired target price $F$ and the current price $P$.
   - **Term 2:** $-\delta S$ captures the natural decay of supply due to market factors like mining costs.

3. **Network Effect Equation (dN/dt):**
   
   $$\frac{dN}{dt} = \epsilon (P - F)^2 - \zeta N$$
   
   - **Term 1:** $\epsilon (P - F)^2$ models how significant price deviations from the target $F$ increase network activity due to speculation.
   - **Term 2:** $-\zeta N$ accounts for network decay over time.

## Model Parameters

- $\alpha, \beta, \gamma, \delta, \epsilon, \zeta$: Constants determining the influence of various market dynamics
- $F$: The desired target price
- $K$: Market capacity

## Numerical Method

The system is solved using a Diagonally Implicit Runge-Kutta (DIRK) method with a 2-stage Gauss-Legendre scheme. This method ensures stability for stiff equations, making it suitable for capturing complex market dynamics.

## Simulation Setup

- **Initial Conditions:** $P=50$, $S=20$, $N=10$
- **Time Span:** 0 to 100
- **Step Size:** 0.5

## Experiments & Test Methods

We conducted three distinct experiments by adjusting the model parameters to observe different market dynamics:

1. **Baseline Experiment:**
   - Parameters: $\alpha=0.1$, $\beta=0.02$, $\gamma=0.1$, $\delta=0.05$, $\epsilon=0.01$, $\zeta=0.1$, $F=100$, $K=200$
   - Initial Values: $P=50$, $S=20$, $N=10$
   - Insight: The market shows stable oscillations with moderate price volatility.

2. **High Market Volatility:**
   - Parameters: $\alpha=0.2$, $\beta=0.04$, $\gamma=0.15$, $\delta=0.1$, $\epsilon=0.02$, $\zeta=0.15$, $F=150$, $K=300$
   - Initial Values: $P=60$, $S=25$, $N=15$
   - Insight: The system shows highly volatile price swings due to amplified network effects and supply adjustments.

3. **Low Market Volatility:**
   - Parameters: $\alpha=0.05$, $\beta=0.01$, $\gamma=0.05$, $\delta=0.02$, $\epsilon=0.005$, $\zeta=0.05$, $F=80$, $K=150$
   - Initial Values: $P=40$, $S=15$, $N=8$
   - Insight: The market stabilizes quickly, with minimal price fluctuations.

## Results & Insights

The numerical experiments reveal cyclic dynamics in price, supply, and network effects. These cycles highlight how market psychology can induce volatility, driven by supply-demand adjustments and speculative behavior within the network. Each experiment showcased distinct market responses under different parameter settings, emphasizing the model's adaptability in capturing complex cryptocurrency market behaviors.

#### I used Claude AI to create Cryptocurrency Market Psychology equations system