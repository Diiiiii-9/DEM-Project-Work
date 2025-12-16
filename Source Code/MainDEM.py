import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation

from dem.particle import Particle
from dem.solver import DEMSolver
from dem.contact_model import get_contact_model
from dem.time_integration import get_time_integration_method
from results.plot_utils import plot_trajectories, plot_velocities, plot_forces, plot_torques

def main():
    # --- Inputs ---

    # input for contact model
    coeff_of_restitution = 1.0
    mu_friction = 0.0

    # particle parameter
    m1, m2 = 1.0, 1.0
    E1, E2 = 10000, 10000
    nu1, nu2 =0, 0
    radius1, radius2 = 0.5, 0.5
    
    # external load
    gravity = np.array([0.0, 0.0, 0.0])

    # simulation properties
    delta_t = 1e-3
    t_end = 1.0

    # Contact and integration models
    contact_params = {
        "coeff_of_restitution": coeff_of_restitution,
        "mu": mu_friction
    }

    contact_model = get_contact_model("HM+D", contact_params)
    time_integration = get_time_integration_method("velocity-verlet")

    # --- Create particles ---
    particles = [
        Particle(position=[0.0, 0.0, 0.0], velocity=[1.0, 0.0, 0.0], omega=[0.0, 0.0, 0.0],radius=radius1, mass=m1, E=E1, nu=nu1, particle_id=1),
        Particle(position=[1.5, 0.0, 0.0], velocity=[-1.0, 0.0, 0.0], omega=[0.0, 0.0, 0.0],radius=radius2, mass=m2,E=E2, nu=nu2, particle_id=2)
    ]

    solver = DEMSolver(particles, contact_model, time_integration, gravity)

    # --- Logs ---
    
    times = []
    trajectories = [[] for _ in particles]
    velocities = [[] for _ in particles]
    forces = [[] for _ in particles]
    torques = [[] for _ in particles]
    omegas = [[] for _ in particles]

    # Plotting selector
    plots_to_show = ["trajectory", "velocity", "force"]

    # --- Main loop ---
    t = 0.0
    while t < t_end:
        solver.solve_time_step(delta_t)

        for i, p in enumerate(particles):
            trajectories[i].append(p.position.copy())
            velocities[i].append(p.velocity.copy())
            forces[i].append(np.linalg.norm(p.force))
            torques[i].append(p.torque.copy())
            omegas[i].append(p.omega.copy())
            

        times.append(t)
        t += delta_t

    # --- Convert logs to arrays ---
    time = np.array(times)
    trajectories = [np.array(traj) for traj in trajectories]
    omegas = [np.array(om) for om in omegas]

    angles = []
    for omega_hist in omegas:
        omega_z = omega_hist[:, 2]             # extract z-component
        angles_i = np.cumsum(omega_z * delta_t)  # integrate over simulation timestep
        angles.append(angles_i)

    # --- Animation ---
    max_radius = np.maximum(radius1, radius2)
    animate_trajectories(particles, trajectories, angles, max_radius)
    

    # --- Plotting ---

    if "velocity" in plots_to_show:
        plot_velocities(velocities, time)

    if "force" in plots_to_show:
        plot_forces(forces, time)

    if "torque" in plots_to_show:
        plot_torques(torques, time)


def animate_trajectories(particles, trajectories, angles, max_radius):
    """
    Animate multiple particles with rotating arrows according to ωz.
    
    Parameters:
    - particles: list of particle objects (must have .radius)
    - trajectories: list of arrays, each shape (N_steps, 2 or 3)
    - angles: list of arrays, each shape (N_steps,), precomputed orientation angles
    - max_radius: maximum particle radius (for margins)
    """

    fig, ax = plt.subplots()
    N = len(particles)
    colors = ['r', 'g', 'b', 'm', 'c']

    # Compute axis limits
    all_positions = np.concatenate(trajectories)
    margin = 0.1 + max_radius
    x_min, x_max = all_positions[:,0].min() - margin, all_positions[:,0].max() + margin
    y_min, y_max = all_positions[:,1].min() - margin, all_positions[:,1].max() + margin
    ax.set_xlim(x_min, x_max)
    ax.set_ylim(y_min, y_max)
    ax.set_aspect('equal')
    ax.set_title("DEM Particle Animation with Rotating Arrows")

    # Initialize circles
    circles = []
    for i, p in enumerate(particles):
        circle = plt.Circle(trajectories[i][0], radius=p.radius,
                            color=colors[i % len(colors)], fill=False)
        ax.add_patch(circle)
        circles.append(circle)

    # Initialize arrows using quiver
    pos = np.array([traj[0,:2] for traj in trajectories])  # initial positions
    dx = np.array([p.radius * np.cos(angles[i][0]) for i,p in enumerate(particles)])
    dy = np.array([p.radius * np.sin(angles[i][0]) for i,p in enumerate(particles)])
    quiv = ax.quiver(pos[:,0], pos[:,1], dx, dy, color=colors[:N],
                     angles='xy', scale_units='xy', scale=1)

    def update(frame):
        # Update circles
        for i, circle in enumerate(circles):
            circle.center = trajectories[i][frame,:2]

        # Update arrows
        X = np.array([trajectories[i][frame,0] for i in range(N)])
        Y = np.array([trajectories[i][frame,1] for i in range(N)])
        U = np.array([particles[i].radius * np.cos(angles[i][frame]) for i in range(N)])
        V = np.array([particles[i].radius * np.sin(angles[i][frame]) for i in range(N)])
        quiv.set_offsets(np.c_[X,Y])
        quiv.set_UVC(U,V)

        return circles + [quiv]

    ani = animation.FuncAnimation(fig, update, frames=len(trajectories[0]),
                                  interval=30, blit=False)
    plt.show()


if __name__ == "__main__":
    main()
