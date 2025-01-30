from src.core.ode_solver import SolverMethod
from src.simulation import ProjectileSimulation


def main():
    # Get the path to the image file
    image_path = "images/test3.png"

    # Create and run simulation
    try:
        simulation = ProjectileSimulation(image_path, solver_method=SolverMethod.RK4)
        simulation.run()
    except Exception as e:
        print(f"Error running simulation: {e}")
        raise


if __name__ == "__main__":
    main()
