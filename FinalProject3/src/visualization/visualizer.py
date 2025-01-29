import json
from pathlib import Path
from typing import List

import matplotlib.pyplot as plt
import numpy as np


class SturmLiouvilleVisualizer:
    def __init__(self, output_dir: str = "output"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)

    def plot_eigenfunctions(
        self,
        x: np.ndarray,
        eigenvalues: List[float],
        eigenfunctions: np.ndarray,
        show: bool = True,
    ) -> None:
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
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))

        # Plot boundary errors
        ax1.semilogy(
            range(1, len(boundary_errors) + 1),
            boundary_errors,
            "o-",
            label="Boundary Error",
        )
        ax1.grid(True, which="both", ls="-", alpha=0.2)
        ax1.set_xlabel("Eigenfunction Number")
        ax1.set_ylabel("Error (log scale)")
        ax1.set_title("Boundary Condition Error Analysis")
        ax1.legend()

        # Plot orthogonality matrix
        n = len(eigenfunctions)
        orthogonality_matrix = np.zeros((n, n))
        for i in range(n):
            for j in range(n):
                # Compute W-inner product using sin(x) weight
                weight = np.sin(x)
                orthogonality_matrix[i, j] = abs(
                    np.trapezoid(weight * eigenfunctions[i] * eigenfunctions[j], x)
                )

        im = ax2.imshow(orthogonality_matrix, cmap="viridis", aspect="equal")
        plt.colorbar(im, ax=ax2)
        ax2.set_title("Orthogonality Check Matrix")
        ax2.set_xlabel("Eigenfunction j")
        ax2.set_ylabel("Eigenfunction i")

        plt.tight_layout()
        plt.savefig(
            self.output_dir / "error_analysis.png", dpi=300, bbox_inches="tight"
        )
        plt.close()
