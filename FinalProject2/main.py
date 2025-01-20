from typing import List

import numpy as np

from src.models.parameters import SturmLiouvilleParameters
from src.solvers.finite_difference import FiniteDifference
from src.solvers.power_iteration import PowerIterationSolver
from src.visualization.visualizer import SturmLiouvilleVisualizer


def compute_boundary_errors(eigenfunctions: np.ndarray) -> List[float]:
    """Compute boundary condition errors for each eigenfunction."""
    return [abs(ef[-1]) for ef in eigenfunctions]  # Error at x=π/2


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

    # Convert to full domain eigenfunctions with proper normalization
    x = fd.x
    full_eigenfunctions = np.zeros((len(eigenvalues), len(x)))

    print("\nNormalizing eigenfunctions...")
    for i, v in enumerate(eigenvectors):
        # Add boundary points
        full_eigenfunctions[i, 1:-1] = v
        # Normalize with weight function
        weight = 1.0  # w(x) = 1 for this problem
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

    # Print results
    print("\nComputed eigenvalues:")
    for i, (ev, err) in enumerate(zip(eigenvalues, boundary_errors)):
        print(f"λ_{i+1} = {ev:.6f} (boundary error: {err:.2e})")

    print("\nResults have been saved in the 'output' directory.")


if __name__ == "__main__":
    main()
