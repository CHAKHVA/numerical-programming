from scipy.optimize import minimize


def ball_ode(t, state, g):
    x, y, vx, vy = state
    dxdt = vx
    dydt = vy
    dvxdt = 0
    dvydt = -g
    return [dxdt, dydt, dvxdt, dvydt]


def solve_ode(initial_state, t_span, g, method="RK45"):
    from scipy.integrate import solve_ivp

    sol = solve_ivp(ball_ode, t_span, initial_state, args=(g,), method=method)
    return sol.y


def shooting_method_error(params, target_positions, t_span, g):
    vx0, vy0 = params
    initial_state = [target_positions[0][0], target_positions[0][1], vx0, vy0]
    sol = solve_ode(initial_state, t_span, g)
    predicted_positions = list(zip(sol[0], sol[1]))
    error = np.sum(
        [
            np.linalg.norm(np.array(pred) - np.array(actual))
            for pred, actual in zip(predicted_positions, target_positions)
        ]
    )
    return error


def reconstruct_trajectory(ball_positions, g=9.81):
    t_span = (0, len(ball_positions))
    initial_guess = [1.0, 1.0]  # Initial guess for vx0 and vy0
    result = minimize(
        shooting_method_error, initial_guess, args=(ball_positions, t_span, g)
    )
    vx0, vy0 = result.x
    initial_state = [ball_positions[0][0], ball_positions[0][1], vx0, vy0]
    sol = solve_ode(initial_state, t_span, g)
    return sol


# Example usage
g = 9.81
trajectory = reconstruct_trajectory(ball_positions, g)


def intercept_ball(trajectory, intercept_point, g=9.81):
    # Assume intercept_point is a random point (x, y) from which we launch the second ball
    x0, y0 = intercept_point
    # Calculate the time and position to intercept
    # This is a simplified approach; a more accurate method would involve solving for the intersection
    t_intercept = len(trajectory[0]) // 2  # Example: intercept at the midpoint
    target_x, target_y = trajectory[0][t_intercept], trajectory[1][t_intercept]

    # Calculate initial velocity for the second ball
    vx0 = (target_x - x0) / t_intercept
    vy0 = (target_y - y0 + 0.5 * g * t_intercept**2) / t_intercept

    # Simulate the second ball's trajectory
    initial_state = [x0, y0, vx0, vy0]
    t_span = (0, t_intercept)
    sol = solve_ode(initial_state, t_span, g)
    return sol


# Example usage
intercept_point = (100, 100)  # Random point
intercept_trajectory = intercept_ball(trajectory, intercept_point, g)


def euler_method(ball_ode, initial_state, t_span, g, dt):
    t = np.arange(t_span[0], t_span[1], dt)
    states = [initial_state]
    for _ in t[1:]:
        current_state = states[-1]
        dxdt, dydt, dvxdt, dvydt = ball_ode(0, current_state, g)
        new_state = [
            current_state[0] + dxdt * dt,
            current_state[1] + dydt * dt,
            current_state[2] + dvxdt * dt,
            current_state[3] + dvydt * dt,
        ]
        states.append(new_state)
    return np.array(states).T


# Example usage
dt = 0.01
initial_state = [ball_positions[0][0], ball_positions[0][1], 1.0, 1.0]
t_span = (0, len(ball_positions))
euler_trajectory = euler_method(ball_ode, initial_state, t_span, g, dt)

# Compare with Runge-Kutta
rk_trajectory = solve_ode(initial_state, t_span, g, method="RK45")

# Compare the results
print("Euler Method Trajectory:", euler_trajectory)
print("Runge-Kutta Trajectory:", rk_trajectory)

import matplotlib.animation as animation
import matplotlib.pyplot as plt


def create_animation(trajectory, intercept_trajectory):
    fig, ax = plt.subplots()
    ax.set_xlim(0, 800)
    ax.set_ylim(0, 600)

    (ball,) = ax.plot([], [], "bo", markersize=10)
    (intercept_ball,) = ax.plot([], [], "ro", markersize=10)

    def animate(i):
        ball.set_data(trajectory[0][i], trajectory[1][i])
        intercept_ball.set_data(intercept_trajectory[0][i], intercept_trajectory[1][i])
        return ball, intercept_ball

    ani = animation.FuncAnimation(
        fig, animate, frames=len(trajectory[0]), interval=50, blit=True
    )
    plt.show()


# Example usage
create_animation(trajectory, intercept_trajectory)  # Example usage
create_animation(trajectory, intercept_trajectory)
