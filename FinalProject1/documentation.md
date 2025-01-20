# Ball Detection and Trajectory Simulation Project

## Problem Statement

This project implements a ball detection and trajectory simulation system that:

1. Detects balls in an input image using edge detection and circle detection
2. Simulates shooting trajectories from a fixed point to each detected ball
3. Visualizes the trajectories using projectile motion physics

## Mathematical Model

### Ball Detection

1. **Edge Detection (Canny Algorithm)**:
   - Gaussian smoothing: $G(x,y) = \frac{1}{2\pi\sigma^2}e^{-\frac{x^2+y^2}{2\sigma^2}}$
   - Gradient calculation using Sobel operators
   - Non-maximum suppression
   - Double thresholding and edge tracking

2. **Circle Detection (Hough Transform)**:
   - Circle equation: $(x - x_c)^2 + (y - y_c)^2 = r^2$
   - Parameter space accumulation
   - Peak detection for circle centers

### Trajectory Simulation

1. **Projectile Motion Equations**:

   ```
   x(t) = x₀ + v₀cos(θ)t
   y(t) = y₀ + v₀sin(θ)t - ½gt²
   ```

   where:
   - (x₀, y₀): Initial position
   - v₀: Initial velocity
   - θ: Launch angle
   - g: Gravitational acceleration (9.81 m/s²)
   - t: Time

2. **Boundary Conditions**:
   - Initial position: Fixed shooting point (x₀, y₀)
   - Target position: Center of detected ball (x_target, y_target)
   - y(t) ≥ 0 (Above ground constraint)

## Numerical Methods

### Shooting Method Implementation

The shooting method finds optimal initial velocity (v₀) and angle (θ) to hit targets:

1. **Parameter Estimation**:
   - Initial velocity range: `[0.5 * √(gd), 1.5 * √(gd)]` where d is target distance
   - Angle range: [0, π/2] radians

2. **Grid Search Algorithm**:
   - Sample velocities and angles in ranges
   - Calculate trajectories for each combination
   - Find parameters minimizing distance to target
   - Stop when within tolerance or best parameters found

3. **Error Analysis**:
   - Distance error: `√[(x - x_target)² + (y - y_target)²]`
   - Tolerance: 0.1 units
   - Parameters refined through grid search

### Numerical Properties

1. **Discretization**:
   - Time step: 0.01s for trajectory calculation
   - Angular step: π/40 for angle search
   - Velocity step: Adaptive based on distance

2. **Stability**:
   - Time step chosen small enough for stable trajectories
   - Grid search ensures robust parameter finding
   - Parameter ranges prevent unrealistic solutions

## Algorithm

1. **Ball Detection**:

   ```
   Input: Image
   1. Apply Gaussian blur
   2. Compute image gradients
   3. Apply non-maximum suppression
   4. Perform double thresholding
   5. Track edges by hysteresis
   6. Apply Hough transform for circles
   Output: List of (x, y, r) for detected balls
   ```

2. **Trajectory Calculation**:

   ```
   Input: Shooting point, target positions
   For each target:
   1. Calculate target distance
   2. Define parameter search ranges
   3. Grid search for optimal v₀ and θ
   4. Calculate trajectory with best parameters
   Output: List of trajectory points
   ```

## Test Cases

### Ball Detection Tests

1. **Simple Test**: Image with clear, non-overlapping balls
2. **Complex Test**: Image with varying ball sizes and textures
3. **Edge Cases**:
   - Different lighting conditions
   - Overlapping balls
   - Various backgrounds

### Trajectory Tests

1. **Horizontal Shots**: Targets at same height
2. **Upward Shots**: Targets above shooting point
3. **Distance Tests**: Near and far targets
4. **Precision Test**: Multiple shots to same target

## Results Validation

1. **Ball Detection**:
   - Compare detected circles with known ball positions
   - Verify radius consistency
   - Check edge detection accuracy

2. **Trajectory Simulation**:
   - Verify trajectories reach targets within tolerance
   - Check physical plausibility of solutions
   - Validate parameter ranges

## Test Data Description

Test images include:

1. `test1.png`: Simple balls with textures background
2. `test2.png`: More Balls with textured background
3. `test3.png`: More Balls with simple background
4. `test4.png`: More Balls with simple background
5. `test3.png`: More Balls with different background
6. `test4.png`: More Balls with different background

## Implementation Details

- Programming Language: Python
- Key Libraries: NumPy, OpenCV, Matplotlib
- Project Structure: Modular design with separate classes for detection and simulation
