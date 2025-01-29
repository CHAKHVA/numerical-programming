# Sturm-Liouville Problem Solver Documentation

## Problem Statement

This project implements a numerical solver for the Sturm-Liouville eigenvalue problem on a finite interval with singular coefficients. The solver finds the first eight eigenvalues and corresponding eigenfunctions.

### Mathematical Model

The Sturm-Liouville equation is given by:

-1/2(cos⁴(x)u'' + cos³(x)cos(2x)/sin(x)u') + (m²cos²(x)/(2sin²(x)) - cos(x)/sin(x))u = λu

with boundary conditions:

- u(0) = 0
- u(π/2) = 0

where:

- x ∈ [0, π/2] is the spatial variable
- λ is the eigenvalue parameter
- u(x) is the eigenfunction
- m is a given parameter

### Numerical Methods Properties

1. **Discretization Method**:
   - Uses finite difference method for spatial discretization
   - Second-order central differences for the second derivative
   - First-order central differences for the first derivative
   - Grid size h = π/(2(N-1)) where N is the number of points

2. **Handling of Singular Points**:
   - At x = 0: Singularity due to sin(x) in denominator
   - At x = π/2: Vanishing coefficients due to cos(x)
   - Uses Taylor series expansions near singular points:
     - Near x = 0: sin(x) ≈ x + ε
     - Near x = π/2: cos(x) ≈ -(x - π/2)

3. **Eigenvalue Solver**:
   - Uses direct eigenvalue computation via numpy.linalg.eigh
   - Provides all eigenvalues simultaneously
   - Guarantees orthogonal eigenvectors
   - Maintains numerical stability

4. **Error Analysis**:
   - Truncation error: O(h²) for interior points
   - Boundary condition accuracy: Monitored via residuals
   - Solution verification through eigenfunction orthogonality

## Algorithm

1. **Initialization**:
   - Set parameters: N (grid points), m (equation parameter)
   - Create uniform grid on [0, π/2]
   - Initialize coefficient matrices

2. **Matrix Construction**:

   ```
   For each interior point i = 1 to N-2:
       1. Compute coefficients p(x), q(x), w(x)
       2. Handle singular points using Taylor expansions
       3. Fill tridiagonal matrix A using finite differences:
          - A[i,i-1] = p/h² - q/(2h)
          - A[i,i] = -2p/h² + w
          - A[i,i+1] = p/h² + q/(2h)
   ```

3. **Eigenvalue Problem Solution**:

   ```
   1. Solve generalized eigenvalue problem Au = λWu
   2. Sort eigenvalues in ascending order
   3. Select first 8 eigenvalues and eigenvectors
   4. Normalize eigenvectors
   ```

4. **Post-processing**:

   ```
   For each eigenpair:
       1. Convert to full domain including boundary points
       2. Normalize with proper weight function
       3. Verify boundary conditions
       4. Compute error metrics
   ```

## Test Case

### Input Parameters

- m = 2 (equation parameter)
- N = 200 (grid points)
- Number of eigenvalues = 8
- Domain: [0, π/2]

### Expected Properties

1. Eigenvalues:
   - All positive and real
   - Strictly increasing
   - Asymptotic growth rate proportional to n²

2. Eigenfunctions:
   - Satisfy boundary conditions u(0) = u(π/2) = 0
   - n-th eigenfunction has n-1 internal zeros
   - Orthogonal with respect to weight function

### Verification Criteria

1. Boundary condition residuals < 10⁻⁸
2. Eigenfunction orthogonality check
3. Node counting for each eigenfunction
4. Asymptotic behavior of eigenvalues

## Implementation Notes

The code is organized into three main classes:

1. `SturmLiouvilleParameters`: Problem parameters
2. `SturmLiouvilleSolver`: Main solver implementation
3. `SturmLiouvilleVisualizer`: Results visualization

### Key Features

- Robust handling of singular points
- Direct eigenvalue computation
- Comprehensive error analysis
- Clear visualization of results

## Output

The solver produces:

1. Eigenvalues λ₁ through λ₈
2. Corresponding eigenfunctions u₁(x) through u₈(x)
3. Visualization plots:
   - Eigenfunction plots
   - Error analysis
4. Numerical data in JSON format

## Usage

```python
# Example usage
params = SturmLiouvilleParameters(m=2, N=200)
solver = SturmLiouvilleSolver(params)
eigenvalues, eigenfunctions = solver.solve()

# Visualize results
visualizer = SturmLiouvilleVisualizer()
visualizer.plot_eigenfunctions(solver.x, eigenvalues, eigenfunctions)
```
