# Ball Trajectory Reconstruction and Interception Project

## Table of Contents

1. [Problem Statement](#1-problem-statement)
   - [1.1 Project Objectives](#11-project-objectives)
   - [1.2 Input](#12-input)
   - [1.3 Output Requirements](#13-output-requirements)

2. [Mathematical Model](#2-mathematical-model)
   - [2.1 Physical Model](#21-physical-model)
   - [2.2 First-Order System](#22-first-order-system)
   - [2.3 Initial and Boundary Conditions](#23-initial-and-boundary-conditions)

3. [Numerical Methods](#3-numerical-methods)
   - [3.1 Implemented Methods](#31-implemented-methods)
     - [3.1.1 Euler Method](#311-euler-method)
     - [3.1.2 RK4 Method](#312-rk4-method)
   - [3.2 Shooting Method Implementation](#32-shooting-method-implementation)
   - [3.3 Error Analysis](#33-error-analysis)

4. [Algorithm](#4-algorithm)
   - [4.1 Ball Detection and Position Extraction](#41-ball-detection-and-position-extraction)
   - [4.2 Trajectory Reconstruction](#42-trajectory-reconstruction)
     - [4.2.1 Core Problem](#421-core-problem)
     - [4.2.2 Initial Velocity Estimation](#422-initial-velocity-estimation)
     - [4.2.3 Shooting Method Implementation](#423-shooting-method-implementation)
     - [4.2.4 Proof of Convergence](#424-proof-of-convergence)
   - [4.3 Interception Calculation](#43-interception-calculation)
     - [4.3.1 Problem Formulation](#431-problem-formulation)
     - [4.3.2 Mathematical Approach](#432-mathematical-approach)
     - [4.3.3 Solution Strategy](#433-solution-strategy)
     - [4.3.4 Proof of Feasibility](#434-proof-of-feasibility)
     - [4.3.5 Implementation Details](#435-implementation-details)

5. [Implementation Details](#5-implementation-details)
   - [5.1 Project Structure](#51-project-structure)
   - [5.2 Key Components](#52-key-components)

6. [Test Cases and Results](#6-test-cases-and-results)
   - [6.1 Test Video Specifications](#61-test-video-specifications)
   - [6.2 Validation Tests](#62-validation-tests)
   - [6.3 Performance Metrics](#63-performance-metrics)

7. [Future Improvements](#7-future-improvements)

8. [References](#8-references)

## 1. Problem Statement

### 1.1 Project Objectives

This project aims to solve two main tasks:

1. Given a partial video of a thrown ball's trajectory, reconstruct its complete path
2. Simulate another ball to intercept the original ball along its trajectory

### 1.2 Input

- A video showing a ball in motion under gravity
- Video captures only part of the trajectory
- Ball is visually distinct (blue colored)

### 1.3 Output Requirements

1. Reconstructed full trajectory of the original ball
2. Simulated trajectory of the intercepting ball
3. Animation showing both trajectories and their intersection

## 2. Mathematical Model

### 2.1 Physical Model

The motion of both balls is governed by projectile motion under gravity with the following equations:

```math
\frac{d^2x}{dt^2} = 0 \\
\frac{d^2y}{dt^2} = g
```

where:

- g is the acceleration due to gravity (positive downward in screen coordinates)
- x, y are positions in screen coordinates
- t is time

### 2.2 First-Order System

Converting to a first-order system:

```math
\frac{dx}{dt} = v_x \\
\frac{dy}{dt} = v_y \\
\frac{dv_x}{dt} = 0 \\
\frac{dv_y}{dt} = g \\
```

### 2.3 Initial and Boundary Conditions

1. Original ball:
   - Initial positions extracted from video
   - Initial velocities determined through shooting method
   - Boundary condition: trajectory must pass through observed points

2. Intercepting ball:
   - Initial position chosen randomly within specified ranges
   - Initial velocities determined to achieve intersection
   - Boundary condition: must intersect with original ball's trajectory

## 3. Numerical Methods

### 3.1 Implemented Methods

#### 3.1.1 Euler Method

```math
y_{n+1} = y_n + hf(t_n, y_n)
```

- Simplest method
- First-order accurate: O(h)
- Less computationally intensive
- Greater accumulated error

#### 3.1.2 RK4 Method

```math
k_1 = f(t_n, y_n)
k_2 = f(t_n + \frac{h}{2}, y_n + \frac{h}{2}k_1) \\
k_3 = f(t_n + \frac{h}{2}, y_n + \frac{h}{2}k_2) \\
k_4 = f(t_n + h, y_n + hk_3) \\
y_{n+1} = y_n + \frac{h}{6}(k_1 + 2k_2 + 2k_3 + k_4)
```

- Fourth-order accurate: O(h⁴)
- More computationally intensive
- Better stability and accuracy

### 3.2 Shooting Method Implementation

1. Initialize with estimated velocity from first two points
2. Simulate trajectory using ODE solver
3. Calculate error between simulated and observed positions
4. Update initial velocities using optimization (Nelder-Mead)
5. Repeat until convergence

### 3.3 Error Analysis

1. Truncation Error:
   - Euler: O(h²) local, O(h) global
   - RK4: O(h⁵) local, O(h⁴) global

2. Stability Analysis:
   - Euler is conditionally stable
   - RK4 has larger stability region
   - Both methods stable for this problem due to simple ODEs

## 4. Algorithm

### 4.1 Ball Detection and Position Extraction

1. Convert frame to HSV color space
2. Apply color thresholding for blue ball
3. Find contours and filter by area and circularity
4. Extract center coordinates of detected ball
5. Store frame number and position for each detection

### 4.2 Trajectory Reconstruction

#### 4.2.1 Core Problem

Given a set of observed positions {(x₁,y₁), (x₂,y₂), ..., (xₙ,yₙ)} from the video, we need to:

1. Find initial velocities (vx₀, vy₀)
2. Reconstruct the complete trajectory

This is a boundary value problem converted to an initial value problem using the shooting method.

#### 4.2.2 Initial Velocity Estimation

First, we make an educated initial guess for velocities:

```math
v_{x0} = \frac{x_2 - x_1}{\Delta t} \\

v_{y0} = \frac{y_2 - y_1}{\Delta t} - \frac{1}{2}g\Delta t
```

Where:

- Δt = 1/fps (time between frames)
- Second term in vy₀ accounts for gravitational acceleration
- This gives a first-order approximation based on kinematics

#### 4.2.3 Shooting Method Implementation

The shooting method works by iteratively refining initial velocities through these steps:

1. **Forward Integration**:

   ```math
   \begin{cases}
   x(t) = x_0 + v_{x0}t \\
   y(t) = y_0 + v_{y0}t + \frac{1}{2}gt^2
   \end{cases}
   ```

2. **Error Function**:

   ```math
   E(v_{x0}, v_{y0}) = \sum_{i=1}^{n} \sqrt{(x_i - x_{sim}(t_i))^2 + (y_i - y_{sim}(t_i))^2}
   ```

   where (x_sim, y_sim) are simulated positions at observation times.

3. **Optimization**:
   Using Nelder-Mead method to minimize E(vx₀, vy₀):

   ```math
   \nabla E = \begin{bmatrix}
   \frac{\partial E}{\partial v_{x0}} \\
   \frac{\partial E}{\partial v_{y0}}
   \end{bmatrix} \rightarrow 0
   ```

#### 4.2.4 Proof of Convergence

The shooting method converges because:

1. **Existence**: By physics, we know a unique solution exists (real ball followed this trajectory)

2. **Continuity**: The error function E is continuous in (vx₀, vy₀) because:
   - ODEs are linear
   - Error measurement is Euclidean distance (continuous)

3. **Convexity**: For projectile motion without air resistance:
   - Trajectories are parabolic
   - Error surface is quadratic near solution
   - Guarantees unique minimum

4. **Error Bounds**:

   ```math
   \|E_{n+1}\| \leq C\|E_n\|^2
   ```

   where C is a constant dependent on g and Δt, showing quadratic convergence near solution.

### 4.3 Interception Calculation

#### 4.3.1 Problem Formulation

Given:

- Original ball trajectory r₁(t) = (x₁(t), y₁(t))
- Shooter position (x₀, y₀)
Find:
- Initial velocities (vx₀, vy₀) for interceptor
Such that:
- ∃t: r₁(t) = r₂(t) (trajectories intersect)

#### 4.3.2 Mathematical Approach

The problem can be solved through these equations:

1. **Time of Flight Equation**:
   For intersection at time t:

   ```math
   \begin{cases}
   x_0 + v_{x0}t = x_1(t) \\
   y_0 + v_{y0}t + \frac{1}{2}gt^2 = y_1(t)
   \end{cases}
   ```

2. **Intersection Condition**:

   ```math
   \min_{t,v_{x0},v_{y0}} \|r_1(t) - r_2(t)\| = 0
   ```

#### 4.3.3 Solution Strategy

We use a multi-stage optimization:

1. **Initial Guess**:
   - Estimate intersection time tᵢ based on target speed
   - Calculate required velocity:

   ```math
   \begin{cases}
   v_{x0} = \frac{x_1(t_i) - x_0}{t_i} \\
   v_{y0} = \frac{y_1(t_i) - y_0}{t_i} - \frac{1}{2}gt_i
   \end{cases}
   ```

2. **Distance Minimization**:
   Define distance function:

   ```math
   D(v_{x0}, v_{y0}) = \min_t \|r_1(t) - r_2(t; v_{x0}, v_{y0})\|
   ```

3. **Optimization Process**:
   a. For each velocity guess:
      - Simulate interceptor trajectory
      - Find minimum distance to target trajectory
   b. Update velocities using Nelder-Mead
   c. Repeat until convergence

#### 4.3.4 Proof of Feasibility

The interception problem is solvable because:

1. **Controllability**:
   - Projectile motion spans 2D space
   - Any point in range can be reached
   - Range equation:

   ```math
   R = \frac{v^2\sin(2\theta)}{g}
   ```

2. **Time Window**:
   - Target trajectory is known
   - Multiple intersection points possible
   - At least one solution exists if target within range

3. **Optimization Convergence**:
   - Error function is continuous
   - Physical constraints bound search space
   - Local minimum is global minimum due to physics

#### 4.3.5 Implementation Details

1. **Speed Matching**:
   - Use similar velocity magnitude as target
   - Helps ensure realistic interception

2. **Multiple Solutions**:
   - Choose solution with earliest intersection
   - Minimize numerical error accumulation

3. **Numerical Stability**:
   - Normalize distances for optimization
   - Use adaptive step sizes
   - Handle edge cases (no intersection possible)

## 5. Implementation Details

### 5.1 Project Structure

``` python
FinalProject2/
├── constants.py            # Configuration parameters
├── ball_detector.py     # Ball detection module
├── ode_solvers.py      # Numerical integration methods
├── trajectory_predictor.py  # Trajectory calculation
├── animator.py         # Visualization
└── main.py            # Main execution
```

### 5.2 Key Components

1. Ball Detector:
   - HSV color filtering
   - Contour detection and filtering
   - Position extraction

2. ODE Solver:
   - Euler method implementation
   - RK4 method implementation
   - Adaptable time step

3. Trajectory Predictor:
   - Shooting method optimization
   - Full trajectory generation
   - Intersection calculation

4. Animator:
   - Real-time visualization
   - Trajectory plotting
   - Velocity vector display

## 6. Test Cases and Results

### 6.1 Test Video Specifications

- Resolution: 640x480
- Frame rate: 30 fps
- Ball color: Blue
- Partial trajectory visible
- Clear background for reliable detection

### 6.2 Validation Tests

1. Ball Detection Accuracy:
   - Verified position extraction accuracy
   - Analyzed detection consistency

2. Trajectory Reconstruction:
   - Compared Euler vs RK4 accuracy
   - Verified physics model correctness
   - Tested with different partial trajectories

3. Interception Accuracy:
   - Tested various shooter positions
   - Verified intersection achievement

### 6.3 Performance Metrics

1. Numerical Method Comparison:
   - RK4 shows better accuracy but higher computation time
   - Euler sufficient for simple trajectories
   - Trade-off between speed and accuracy

2. Optimization Performance:
   - Convergence typically achieved in <100 iterations
   - Initial velocity estimation improves convergence
   - Adaptive step size improves stability

## 7. Future Improvements

1. Physics Model:
   - Add air resistance
   - Include rotational effects
   - Consider 3D trajectories

2. Numerical Methods:
   - Implement adaptive step size
   - Add more integration methods
   - Improve shooting method convergence

3. Visualization:
   - Add error visualization
   - Show physics parameters
   - Include interactive controls

## 8. References

1. Numerical Methods:
   - Numerical Analysis

2. Computer Vision:
   - OpenCV Documentation

3. Physics:
   - Projectile Motion with Air Resistance
