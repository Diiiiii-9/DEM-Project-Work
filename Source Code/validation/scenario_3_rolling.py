# validation/scenario_3_rolling.py

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.patches import Circle
import os
import sys

current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.append(parent_dir)

from base_validation import BaseValidationScenario
from dem.particle import Particle
from dem.boundary import Boundary
from dem.solver import DEMSolver
from dem.contact_model import get_contact_model
from dem.time_integration import get_time_integration_method
from results.plot_utils import plot_trajectories, plot_velocities, plot_forces
from results.plot_field import plot_field

def main():
    """
    main function to run the rolling friction test scenario
    """
    
    # --- Inputs ---
    # Contact model parameters
    coeff_of_restitution = 0.5
    mu_friction = 0.3
    mu_rolling = 0.1
    rolling_stiffness_ratio = 0.1
    rolling_damping_ratio = 0.1
    
    # Particle parameters
    mass = 1.0
    E = 1e5
    nu = 0.3
    radius = 0.2
    
    # External load
    # Modified: Gravity is required to generate Normal Force, which is required for Friction
    gravity = np.array([0.0, -9.81, 0.0])
    
    # Simulation properties
    delta_t = 1e-3
    t_end = 2.0
    
    # Contact and integration models
    contact_params = {
        "coeff_of_restitution": coeff_of_restitution,
        "mu": mu_friction,
        "mu_rolling": mu_rolling,
        "rolling_stiffness_ratio": rolling_stiffness_ratio,
        "rolling_damping_ratio": rolling_damping_ratio
    }
    contact_model = get_contact_model("HM+D", contact_params)
    time_integration = get_time_integration_method("velocity-verlet")
    
    # --- Create particles ---
    # Particle setup
    particle = Particle(position=[-3.0, 0.2, 0.0], velocity=[3.0, 0.0, 0.0], omega=[0.0, 0.0, 10.0],
                        radius=radius, mass=mass, E=E, nu=nu, particle_id=1)
    
    # Boundary setup
    floor_boundary = Boundary(boundary_id_in=1, point_in=np.array([0.0, 0.0, 0.0]),
                              normal_in=np.array([0.0, 1.0, 0.0]),
                              E_in=1e10, nu_in=0.0, mu_in=mu_friction)
    
    # Modified: Passed boundaries=[floor_boundary] to the solver
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
        # Solve one time step
        solver.solve_time_step(delta_t)
        
        # Log data
        times.append(t)
        trajectories[0].append(particle.position.copy())
        velocities[0].append(particle.velocity.copy())
        forces[0].append(particle.force.copy())
        torques[0].append(particle.torque.copy())
        omegas[0].append(particle.omega.copy())
        
        t += delta_t
        
    # Convert logs to numpy arrays for easier handling
    trajectories = [np.array(traj) for traj in trajectories]
    velocities = [np.array(vel) for vel in velocities]
    forces = [np.array(frc) for frc in forces]
    
    # --- Plot results ---
    output_filename = "scenario_3_rolling.gif"  
    particle.history_time = np.array(times)
    particle.history_position = trajectories[0]
    particle.history_velocity = velocities[0]
    particle.history_force = forces[0]
    
    plot_field([floor_boundary], [particle], output_filename)

# --- Run the main function ---
if __name__ == "__main__":
    main()