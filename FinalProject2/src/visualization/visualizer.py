import json
from pathlib import Path
from typing import List

import matplotlib.pyplot as plt
import numpy as np


class SturmLiouvilleVisualizer:
    """Class for visualizing Sturm-Liouville solutions."""

    def __init__(self, output_dir: str = "output"):
        """
        Initialize visualizer.

        Args:
            output_dir: Directory to save plots and results
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)

    def plot_eigenfunctions(
        self,
        x: np.ndarray,
        eigenvalues: List[float],
        eigenfunctions: np.ndarray,
        show: bool = True,
    ) -> None:
        """
        Plot eigenfunctions with corresponding eigenvalues.

        Args:
            x: Grid points
            eigenvalues: List of eigenvalues
            eigenfunctions: Array of eigenfunction values
            show: Whether to display the plot
        """
        plt.figure(figsize=(12, 8))
        colors = plt.cm.tab10(np.linspace(0, 1, len(eigenvalues)))

        for i, (eigenval, eigenfunc, color) in enumerate(
            zip(eigenvalues, eigenfunctions, colors)
        ):
            plt.plot(
                x,
                eigenfunc,
                label=f"λ_{i+1} = {eigenval:.4f}",
                color=color,
                linewidth=1.5,
            )

        plt.title("Sturm-Liouville Eigenfunctions", fontsize=14, pad=20)
        plt.xlabel("x", fontsize=12)
        plt.ylabel("u(x)", fontsize=12)

        # Customize grid
        plt.grid(True, linestyle="--", alpha=0.3)
        plt.axhline(y=0, color="k", linestyle="-", alpha=0.2)
        plt.axvline(x=0, color="k", linestyle="-", alpha=0.2)

        # Add legend with good placement
        plt.legend(bbox_to_anchor=(1.05, 1), loc="upper left", borderaxespad=0.0)

        # Save plot with high quality
        plt.savefig(
            self.output_dir / "eigenfunctions.png", dpi=300, bbox_inches="tight"
        )

        if show:
            plt.show()
        plt.close()

    def save_results(
        self, x: np.ndarray, eigenvalues: List[float], eigenfunctions: np.ndarray
    ) -> None:
        """
        Save numerical results to JSON file.

        Args:
            x: Grid points
            eigenvalues: List of eigenvalues
            eigenfunctions: Array of eigenfunction values
        """
        results = {
            "x_points": x.tolist(),
            "eigenvalues": eigenvalues,
            "eigenfunctions": eigenfunctions.tolist(),
        }

        with open(self.output_dir / "results.json", "w") as f:
            json.dump(results, f, indent=2)

    def plot_error_analysis(
        self, x: np.ndarray, eigenfunctions: np.ndarray, boundary_errors: List[float]
    ) -> None:
        """
        Plot error analysis for boundary conditions and orthogonality.

        Args:
            x: Grid points
            eigenfunctions: Array of eigenfunction values
            boundary_errors: List of boundary condition errors
        """
        plt.figure(figsize=(10, 6))
        plt.semilogy(
            range(1, len(boundary_errors) + 1),
            boundary_errors,
            "o-",
            label="Boundary Error",
        )
        plt.grid(True, which="both", ls="-", alpha=0.2)
        plt.xlabel("Eigenfunction Number")
        plt.ylabel("Error (log scale)")
        plt.title("Boundary Condition Error Analysis")
        plt.legend()
        plt.savefig(
            self.output_dir / "error_analysis.png", dpi=300, bbox_inches="tight"
        )
        plt.close()
