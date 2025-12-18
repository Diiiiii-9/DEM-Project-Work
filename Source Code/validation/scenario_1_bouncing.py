# Particle bouncing on a rigid floor scenario
import numpy as np
import matplotlib.pyplot as plt
import sys
import os
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.append(parent_dir)

from dem.particle import Particle
from dem.boundary import Boundary
from dem.solver import DEMSolver
from dem.contact_model import get_contact_model
from dem.time_integration import get_time_integration_method
from results.plot_utils import plot_trajectories, plot_velocities

from results.plot_field import plot_field
def main():
    # --- Inputs ---

    # Contact model parameters
    coeff_of_restitution = 0.8
    mu_friction = 0.0

    # Particle parameters
    mass = 1.0
    E = 10000
    nu = 0.0
    radius = 0.5

    # External load
    gravity = np.array([0.0, -9.81, 0.0])

    # Simulation properties
    delta_t = 1e-4
    t_end = 2.0

    # Contact and integration models
    contact_params = {
        "coeff_of_restitution": coeff_of_restitution,
        "mu": mu_friction
    }

    contact_model = get_contact_model("HM+D", contact_params)
    time_integration = get_time_integration_method("velocity-verlet")

    # --- Create particle ---
    particle = Particle(position=[5.0, 1.0, 0.0], velocity=[0.0, 0.0, 0.0], omega=[0.0, 0.0, 0.0],
                        radius=radius, mass=mass, E=E, nu=nu, particle_id=1)

    # Rigid floor boundary (y=0)
    Boundary_point = np.array([0.0, 0.0, 0.0])
    Boundary_normal = np.array([0.0, 1.0, 0.0])

    floor_boundary = Boundary(boundary_id_in=1, point_in=Boundary_point, normal_in=Boundary_normal,
                              E_in=1e10, nu_in=0.0, mu_in=mu_friction)


    solver = DEMSolver([particle], contact_model, time_integration, gravity, boundaries=[floor_boundary])

    # --- Logs ---
    times = []
    trajectories = [[]]
    velocities = [[]]
    forces = [[]]
    torques = [[]]
    omegas = [[]]
    # --- Main loop ---
    t = 0.0
    while t < t_end:
        solver.solve_time_step(delta_t)

        # Log data
        times.append(t)
        trajectories[0].append(particle.position.copy())
        velocities[0].append(particle.velocity.copy())
        forces[0].append(particle.force.copy())

        t += delta_t

    # Convert logs to numpy arrays for easier handling
    trajectories = [np.array(traj) for traj in trajectories]
    velocities = [np.array(vel) for vel in velocities]
    forces = [np.array(frc) for frc in forces]

    # --- Plot results ---
    output_filename = "scenario_1_bouncing.gif"
    
    particle.history_time = np.array(times)
    particle.history_position = trajectories[0]
    particle.history_velocity = velocities[0]
    particle.history_force = particle.history_force = forces[0]

    #plot_trajectories(trajectories)
    #plot_velocities(velocities, times)
    plot_field([floor_boundary], [particle], "bouncing_scenario.gif")
# --- Run the main function ---
if __name__ == "__main__":
    main()