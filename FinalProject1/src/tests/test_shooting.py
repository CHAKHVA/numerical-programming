import csv
import os
from datetime import datetime

import numpy as np

from src.constants import DEFAULT_INITIAL_POSITION, SHOOTING_TOLERANCE
from src.core.ode_solver import ODESolver, SolverMethod
from src.core.shooting_method import ShootingMethod, UpdateStrategy


def test_shooting_strategies():
    """Test different shooting method strategies and save results"""
    # Test setup
    ode_solver = ODESolver()
    shooting = ShootingMethod(ode_solver)

    # Create results directory if it doesn't exist
    results_dir = os.path.join(os.path.dirname(__file__), "results")
    os.makedirs(results_dir, exist_ok=True)

    # Test cases: different target positions
    test_targets = [
        (500, 400),  # High target
        (600, 300),  # Medium height
        (700, 200),  # Lower target
        (800, 350),  # Far target
        (300, 450),  # Close high target
    ]

    # Results storage
    results = []

    # Test each combination
    for target_x, target_y in test_targets:
        for strategy in UpdateStrategy:
            for solver in SolverMethod:
                print(
                    f"\nTesting target ({target_x}, {target_y}) with {strategy.value} strategy and {solver.value} solver"
                )

                # Run shooting method
                v0, t, trajectory = shooting.solve(
                    target_x=target_x,
                    target_y=target_y,
                    initial_position=DEFAULT_INITIAL_POSITION,
                    solver_method=solver,
                    update_strategy=strategy,
                )

                # Calculate final error if solution found
                if trajectory is not None:
                    final_pos = trajectory[-1, :2]
                    error = np.sqrt(
                        (final_pos[0] - target_x) ** 2 + (final_pos[1] - target_y) ** 2
                    )
                    converged = error < SHOOTING_TOLERANCE
                else:
                    error = float("inf")
                    converged = False

                # Store results
                results.append(
                    {
                        "target_x": target_x,
                        "target_y": target_y,
                        "strategy": strategy.value,
                        "solver": solver.value,
                        "error": error,
                        "converged": converged,
                        "initial_velocity": str(v0) if v0 is not None else "None",
                    }
                )

    # Save results in the results directory
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = os.path.join(results_dir, f"shooting_test_results_{timestamp}.csv")

    with open(filename, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=results[0].keys())
        writer.writeheader()
        writer.writerows(results)

    print(f"\nResults saved to {filename}")

    # Print summary
    print("\nTest Summary:")
    for strategy in UpdateStrategy:
        strategy_results = [r for r in results if r["strategy"] == strategy.value]
        convergence_rate = sum(1 for r in strategy_results if r["converged"]) / len(
            strategy_results
        )
        avg_error = np.mean(
            [r["error"] for r in strategy_results if r["error"] != float("inf")]
        )
        print(f"\n{strategy.value} strategy:")
        print(f"Convergence rate: {convergence_rate:.2%}")
        print(f"Average error: {avg_error:.6f}")


if __name__ == "__main__":
    test_shooting_strategies()
