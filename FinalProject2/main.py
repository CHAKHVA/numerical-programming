from typing import List

import numpy as np

from src.solvers.finite_difference import FiniteDifference
from src.solvers.parameters import SturmLiouvilleParameters
from src.solvers.power_iteration import PowerIterationSolver
from src.visualization.visualizer import SturmLiouvilleVisualizer


def compute_boundary_errors(eigenfunctions: np.ndarray) -> List[float]:
    """Compute boundary condition errors for each eigenfunction."""
    # Check both boundary conditions
    left_errors = [abs(ef[0]) for ef in eigenfunctions]  # Error at x=0
    right_errors = [abs(ef[-1]) for ef in eigenfunctions]  # Error at x=π/2
    return [(l + r) / 2 for l, r in zip(left_errors, right_errors)]


def compute_analytical_eigenvalues(m: float, n: int) -> List[float]:
    """
    Compute first n analytical eigenvalues for the Sturm-Liouville problem.
    """
    return [(k * (k + 2 * m)) ** 2 for k in range(1, n + 1)]


def main():
    """Main function to solve and visualize the Sturm-Liouville problem."""
    # Initialize parameters with improved grid resolution
    params = SturmLiouvilleParameters(
        m=2,
        N=400,  # Increased for better accuracy
        num_eigenvalues=8,
        epsilon=1e-12,  # Smaller epsilon for better handling of singularity
    )

    # Create solvers
    fd = FiniteDifference(params)
    A, W = fd.build_matrices()
    solver = PowerIterationSolver(A, W)

    # Solve eigenvalue problem
    print("Computing eigenvalues and eigenfunctions...")
    eigenvalues, eigenvectors = solver.find_eigenvalues(params.num_eigenvalues)

    # Get analytical eigenvalues for comparison
    analytical_eigenvalues = compute_analytical_eigenvalues(
        params.m, params.num_eigenvalues
    )

    # Convert to full domain eigenfunctions with proper normalization
    x = fd.x
    full_eigenfunctions = np.zeros((len(eigenvalues), len(x)))

    print("\nNormalizing eigenfunctions...")
    for i, v in enumerate(eigenvectors):
        # Add boundary points (explicitly set to 0)
        full_eigenfunctions[i, 0] = 0  # Boundary condition at x=0
        full_eigenfunctions[i, 1:-1] = v
        full_eigenfunctions[i, -1] = 0  # Boundary condition at x=π/2

        # Normalize with weight function w(x) = sin(x)
        weight = np.sin(x)
        norm = np.sqrt(np.trapz(weight * full_eigenfunctions[i] ** 2, x))
        full_eigenfunctions[i] /= norm

    # Compute boundary condition errors
    boundary_errors = compute_boundary_errors(full_eigenfunctions)

    # Initialize visualizer and create plots
    print("\nCreating visualizations...")
    visualizer = SturmLiouvilleVisualizer(output_dir="output")

    # Plot eigenfunctions
    visualizer.plot_eigenfunctions(x, eigenvalues, full_eigenfunctions)

    # Save numerical results
    visualizer.save_results(x, eigenvalues, full_eigenfunctions)

    # Plot error analysis
    visualizer.plot_error_analysis(x, full_eigenfunctions, boundary_errors)

    # Print results and comparison with analytical solution
    print("\nNumerical vs Analytical Eigenvalues:")
    print("n  |  Numerical λ  |  Analytical λ  |  Relative Error  | Boundary Error")
    print("-" * 65)
    for i, (num, ana, err) in enumerate(
        zip(eigenvalues, analytical_eigenvalues, boundary_errors)
    ):
        rel_error = abs(num - ana) / ana
        print(f"{i+1:2d} | {num:12.6f} | {ana:12.6f} | {rel_error:14.2e} | {err:12.2e}")

    print("\nResults have been saved in the 'output' directory.")


if __name__ == "__main__":
    main()
