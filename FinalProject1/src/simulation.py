from .constants import DEFAULT_INITIAL_POSITION, DEFAULT_INITIAL_VELOCITY
from .core.image_processor import ImageProcessor
from .core.ode_solver import ODESolver, SolverMethod
from .core.shooting_method import ShootingMethod
from .models.shape import Shape
from .visualization.visualizer import Visualizer


class ProjectileSimulation:
    """Main simulation class"""

    def __init__(self, image_path: str, solver_method: SolverMethod = SolverMethod.RK4):
        self.image_processor = ImageProcessor(image_path)
        self.ode_solver = ODESolver(method=solver_method)
        self.shooting_method = ShootingMethod(self.ode_solver)
        self.visualizer = None
        self.shapes: list[Shape] = []

    def detect_targets(self) -> None:
        """Detect targets from the image"""
        self.shapes = self.image_processor.detect_shapes()
        print(f"Detected shapes: {self.shapes}")

    def initialize_visualization(self) -> None:
        """Set up the visualization"""
        self.visualizer = Visualizer(
            self.image_processor.width, self.image_processor.height
        )
        self.visualizer.setup_plot()

    def plot_all_targets(self) -> None:
        """Plot all detected targets"""
        for shape in self.shapes:
            self.visualizer.plot_target(shape)

    def process_target(self, shape: Shape, index: int) -> None:
        """Process a single target"""
        target_x, target_y = shape.position
        print(f"\nTarget {index+1}: {shape}")

        # Find trajectory
        v0, t, trajectory = self.shooting_method.solve(
            target_x,
            target_y,
            DEFAULT_INITIAL_POSITION,
            DEFAULT_INITIAL_VELOCITY,
            solver_method=SolverMethod.RK4,
        )

        if trajectory is not None:
            self.visualizer.animate_trajectory(trajectory, target_x, target_y)
            self.visualizer.add_trajectory_to_legend(shape, index + 1)
        else:
            print("Failed to find a solution for this target.")

        self.visualizer.pause_between_targets()

    def run(self) -> None:
        """Run the complete simulation"""
        # Setup
        self.detect_targets()
        self.initialize_visualization()
        self.plot_all_targets()

        # Process each target
        for i, shape in enumerate(self.shapes):
            self.process_target(shape, i)

        # Show legend after all trajectories are drawn
        self.visualizer.show_legend()

        # Show final result
        self.visualizer.show()
