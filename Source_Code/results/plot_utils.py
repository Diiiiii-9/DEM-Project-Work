# plot_utils.py

import matplotlib.pyplot as plt
import numpy as np

def plot_trajectories(trajectories):
    plt.figure(figsize=(6, 6))
    for i, traj in enumerate(trajectories):
        traj = np.array(traj)
        plt.plot(traj[:, 0], traj[:, 1], label=f"Particle {i}")
    plt.title("Particle Trajectories")
    plt.xlabel("x [m]")
    plt.ylabel("y [m]")
    plt.legend()
    plt.grid(True)
    plt.axis("equal")
    plt.tight_layout()
    plt.show()

def plot_forces(forces_over_time, time):
    plt.figure(figsize=(10, 4))
    for i, forces in enumerate(forces_over_time):
        plt.plot(time, forces, label=f"Particle {i}")
    plt.title("Contact Forces Over Time")
    plt.xlabel("Time [s]")
    plt.ylabel("Force Magnitude [N]")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()

def plot_torques(torques_over_time, time):
    plt.figure(figsize=(10, 4))
    for i, torques in enumerate(torques_over_time):
        torques = np.array(torques)        # shape (N_steps, 3)
        torques_z = torques[:, 2]          # extract z-component
        plt.plot(time, torques_z, label=f"Particle {i}")
    plt.title("Torque (z-component) Over Time")
    plt.xlabel("Time [s]")
    plt.ylabel("Torque [Nm]")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()

def plot_velocities(velocities_over_time, time):
    plt.figure(figsize=(10, 4))
    for i, vels in enumerate(velocities_over_time):
        plt.plot(time, vels, label=f"Particle {i}")
    plt.title("Velocity Magnitude Over Time")
    plt.xlabel("Time [s]")
    plt.ylabel("Velocity [m/s]")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()
