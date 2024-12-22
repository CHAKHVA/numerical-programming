# Ball Motion Analysis: Algorithm and Implementation

## Algorithm and Approach

The approach to analyzing ball motion from video consists of three main components:

### 1. Ball Detection and Tracking

- Convert video frames to HSV color space for robust color detection
- Apply color thresholding to isolate the ball
- Use morphological operations to reduce noise
- Find contours and filter based on:
  - Area (minimum size threshold)
  - Circularity (how close to a perfect circle)
  - Size constraints (minimum and maximum radius)
- Convert pixel coordinates to real-world units using calibration factor

### 2. Motion Analysis

- Track ball positions over time
- Calculate velocities using finite differences:

  ```python
  v(t) = [x(t+dt) - x(t)] / dt
  ```

- Calculate accelerations:

  ```python
  a(t) = [v(t+dt) - v(t)] / dt
  ```

- Estimate mass using trajectory peak:

  ```python
  At peak: F_g = ma
  mg = ma_measured
  m = g/a_measured
  ```

- Calculate drag coefficient using:

  ```python
  F_drag = kv²
  ma = kv²
  k = ma/v²
  ```

### 3. Physical Model (ODEs)

The motion is modeled using four coupled ODEs:

```python
dx/dt = vx
dy/dt = vy
dvx/dt = -(k/m)v*vx
dvy/dt = -g - (k/m)v*vy
```

Where:

- (x,y) is position
- (vx,vy) is velocity
- k is drag coefficient
- m is mass
- g is gravitational acceleration
- v is total velocity magnitude

## Properties of Numerical Methods

### 1. Runge-Kutta 4th Order (RK4)

Used for solving the ODEs, RK4 has these properties:

- 4th order accuracy: local error O(h⁵), global error O(h⁴)
- Self-starting method (doesn't need previous steps)
- Stable for reasonable step sizes
- Good balance of accuracy and computational cost
- Formula:

  ```python
  k1 = f(yn)
  k2 = f(yn + h*k1/2)
  k3 = f(yn + h*k2/2)
  k4 = f(yn + h*k3)
  y(n+1) = yn + h*(k1 + 2k2 + 2k3 + k4)/6
  ```

### 2. Finite Differences

Used for velocity and acceleration calculations:

- Forward differences for real-time processing
- First-order accuracy
- Simple implementation
- Sensitive to noise
- Requires small time steps for accuracy

## Limitations and Application Domain

### Successful Cases

The method works well under these conditions:

1. **Ideal Video Conditions**:
   - Good lighting
   - High contrast between ball and background
   - Static camera
   - Minimal motion blur
   - Frame rate ≥ 30 fps

2. **Simple Motion Types**:
   - Basketball free throws
   - Simple projectile motion
   - Minimal spin
   - Motion primarily in 2D plane

3. **Environmental Conditions**:
   - Indoor settings
   - No wind
   - Consistent lighting
   - Clear background

### Failure Cases

The method fails or becomes less accurate in these situations:

1. **Complex Motion**:
   - Significant spin (Magnus effect)
   - 3D motion
   - Multiple bounces
   - Complex trajectories

2. **Poor Video Quality**:
   - Low frame rate
   - Motion blur
   - Poor lighting
   - Moving camera
   - Complex backgrounds
   - Ball occlusion

3. **Physical Limitations**:
   - Simplified drag model doesn't capture all effects
   - No consideration of:
     - Ball rotation
     - Air density variations
     - Wind effects
     - Magnus force
     - Ball deformation

4. **Measurement Issues**:
   - Pixel discretization errors
   - Color detection limitations
   - Calibration inaccuracies
   - Noise in position measurements

### Reasons for Success/Failure

1. **Success Cases**:
   - Simple projectile motion closely matches our physical model
   - Good video quality allows accurate position tracking
   - Static camera maintains consistent coordinate system
   - Clear ball visibility enables reliable detection

2. **Failure Cases**:
   - Complex aerodynamics not captured by simple drag model
   - Motion blur makes ball detection unreliable
   - Camera motion invalidates coordinate system
   - 3D motion cannot be accurately analyzed from single view
   - Spin effects require more sophisticated physical model
