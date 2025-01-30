# Projectile Motion and Shape Detection Project Documentation

## Table of Contents

1. [Project Overview](#1-project-overview)
2. [Image Processing](#2-image-processing)
3. [Numerical Methods](#3-numerical-methods)
4. [Project Structure](#4-project-structure)
5. [Mathematical Analysis](#5-mathematical-analysis)
6. [Implementation Details](#6-implementation-details)
7. [Performance Optimization](#7-performance-optimization)

## 1. Project Overview

### 1.1 Project Description

This project combines computer vision and numerical methods to simulate projectile motion targeting detected shapes in an image. The system:

1. Detects shapes (circles and ellipses) in an input image
2. Calculates trajectories to hit each target using projectile motion
3. Visualizes the trajectories and targets

### 1.2 Core Components

- Image processing and shape detection
- Numerical methods for solving ODEs
- Projectile motion simulation
- Trajectory visualization

## 2. Image Processing

### 2.1 Edge Detection (Canny Algorithm)

The Canny edge detection algorithm consists of five steps:

1. **Gaussian Blur**

   ```math
   G(x,y) = (1/(2πσ²))e^(-(x²+y²)/(2σ²))
   ```

   where σ is the standard deviation.

2. **Gradient Calculation**
   Using Sobel operators:

   ```math
   Gx = [[-1, 0, 1],
         [-2, 0, 2],
         [-1, 0, 1]]

   Gy = [[-1, -2, -1],
         [ 0,  0,  0],
         [ 1,  2,  1]]
   ```

   Magnitude and direction:

   ```math
   G = √(Gx² + Gy²) \\
   θ = arctan(Gy/Gx)
   ```

3. **Non-Maximum Suppression**
   - Compare each pixel with neighbors in gradient direction
   - Keep only local maxima

4. **Double Thresholding**

   ```python
   if pixel >= highThreshold:
       mark as strong edge
   else if pixel >= lowThreshold:
       mark as weak edge
   ```

5. **Hysteresis**
   - Connect weak edges to strong edges
   - Remove isolated weak edges

### 2.2 Shape Detection

#### 2.2.1 Contour Detection

- Follow edges to form continuous contours
- Filter contours based on:

  ```python
  min_area = image_size * 0.0001  # 0.01% of image size
  max_area = image_size * 0.5    # 50% of image size
  ```

#### 2.2.2 Shape Classification

- **Ellipse Detection**:
  - Minimum 5 points required
  - Aspect ratio check: 0.5 ≤ major_axis/minor_axis ≤ 2.0

- **Circle Detection**:
  - Area comparison: 0.8 ≤ area/circular_area ≤ 1.2
  - Where circular_area = πr²

## 3. Numerical Methods

### 3.1 Differential Equations

The projectile motion with air resistance is described by:

```math
d²x/dt² = -(k/m)v·vx \\
d²y/dt² = -g - (k/m)v·vy
```

where:

- g is the acceleration due to gravity (9.81 m/s²)
- k = ½ρCᴅA (air resistance coefficient)
- ρ is air density (1.225 kg/m³)
- Cᴅ is drag coefficient (0.47 for a sphere)
- A is cross-sectional area (πr² for a sphere)
- v is velocity magnitude √(vx² + vy²)
- m is projectile mass (0.1 kg)

As a system of first-order ODEs:

```math
dx/dt = vx \\
dy/dt = vy \\
dvx/dt = -(k/m)v·vx \\
dvy/dt = -g - (k/m)v·vy
```

### 3.2 Numerical Integration Methods

Four methods are implemented and compared:

#### 3.2.1 Euler Method

```python
yₙ₊₁ = yₙ + h·f(tₙ, yₙ)
```

- Simplest method
- Local Error: O(h²)
- Global Error: O(h)
- Less accurate but fastest

#### 3.2.2 Improved Euler Method

```python
k₁ = f(tₙ, yₙ)
k₂ = f(tₙ + h, yₙ + h·k₁)
yₙ₊₁ = yₙ + h/2·(k₁ + k₂)
```

- Better accuracy than basic Euler
- Local Error: O(h³)
- Global Error: O(h²)

#### 3.2.3 RK2 Method (Heun's Method)

```python
k₁ = f(tₙ, yₙ)
k₂ = f(tₙ + 2h/3, yₙ + 2h·k₁/3)
yₙ₊₁ = yₙ + h/4·(k₁ + 3k₂)
```

- Balanced between accuracy and speed
- Local Error: O(h³)
- Global Error: O(h²)

#### 3.2.4 RK4 Method

```python
k₁ = f(tₙ, yₙ)
k₂ = f(tₙ + h/2, yₙ + h·k₁/2)
k₃ = f(tₙ + h/2, yₙ + h·k₂/2)
k₄ = f(tₙ + h, yₙ + h·k₃)
yₙ₊₁ = yₙ + (h/6)·(k₁ + 2k₂ + 2k₃ + k₄)
```

- Most accurate method
- Local Error: O(h⁵)
- Global Error: O(h⁴)
- Default method for simulation

### 3.3 Shooting Method

Two update strategies are implemented for finding initial velocities:

#### 3.3.1 Fixed Update Strategy

- Constant learning rate
- Simple and reliable
- May be slower to converge
- Update rule: `v₀ += error * learning_rate`

#### 3.3.2 Adaptive Update Strategy

- Dynamic learning rate adjustment
- Faster convergence when near solution
- More robust to different target positions
- Update rules:

  ```python
  if current_error < previous_error:
      learning_rate *= 1.1  # Increase rate
  else:
      learning_rate *= 0.5  # Decrease rate
  v₀ += error * learning_rate
  ```

Performance comparison based on test results:

- Adaptive strategy generally achieves better convergence rates
- Fixed strategy may be more stable for simple trajectories
- RK4 with adaptive strategy provides best overall accuracy
- Convergence rates and errors are saved in `tests/results/` for analysis

## 4. Project Structure

```shell
projectile_simulation/
├── images/                    # Directory containing input images
├── src/
│   ├── constants.py          # Global constants and configuration
│   ├── core/
│   │   ├── image_processor.py
│   │   ├── ode_solver.py
│   │   └── shooting_method.py
│   ├── models/
│   │   └── shape.py         # Shape data model
│   ├── visualization/
│   │   └── visualizer.py    # Plotting and animation
│   └── simulation.py        # Main simulation logic
├── main.py                  # Entry point with parallel processing
├── requirements.txt         # Project dependencies
└── documentation.md        # Project documentation
```

## 5. Mathematical Analysis

### 5.1 Error Analysis

For RK4 method:

```math
Local Error = O(h⁵) \\
Global Error = O(h⁴)
```

Error sources:

1. Numerical integration error
2. Shooting method convergence
3. Discretization error

### 5.2 Convergence Analysis

Shooting method convergence depends on:

1. Initial guess quality
2. Update factor (1/10)
3. Target position relative to launch point

## 6. Implementation Details

### 6.1 Optimization Techniques

1. **Image Processing**:
   - Vectorized operations
   - Early filtering of contours
   - Efficient shape validation

2. **Numerical Methods**:
   - Optimized RK4 implementation
   - Adaptive step size control
   - Efficient memory usage

### 6.2 Parameters and Tuning

Key parameters that affect performance:

```python
GAUSSIAN_KERNEL_SIZE = (5, 5)
GAUSSIAN_SIGMA = 1.0
CANNY_LOW_THRESHOLD = 50
CANNY_HIGH_THRESHOLD = 150
CONTOUR_EPSILON_FACTOR = 0.02
ELLIPSE_MIN_POINTS = 5
```

## 7. Performance Optimization

### 7.1 Shape Detection Optimization

1. Preprocessing with bilateral filter and CLAHE
2. Area-based filtering
3. Distance-based duplicate removal
4. Aspect ratio validation

### 7.2 Numerical Computation Optimization

1. Vectorized operations
2. Efficient memory management
3. Early termination conditions
4. Optimized convergence criteria

### 7.3 Visualization Optimization

1. Efficient trajectory animation
2. Batch plotting operations
3. Memory-efficient drawing
