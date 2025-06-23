from src.core.ode_solver import SolverMethod
from src.simulation import ProjectileSimulation


def main():
    # Get the path to the image file
    image_path = "images/random_shapes_random_positions.png"

    # Create and run simulation
    try:
        simulation = ProjectileSimulation(image_path, solver_method=SolverMethod.RK4)
        simulation.run()
    except Exception as e:
        print(f"Error running simulation: {e}")
        raise


if __name__ == "__main__":
    main()
